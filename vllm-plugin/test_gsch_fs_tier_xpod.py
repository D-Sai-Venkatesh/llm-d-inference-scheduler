# SPDX-License-Identifier: Apache-2.0
"""Cross-pod reads: two GschFsTier instances on two roots of one filesystem (engine b and engine c), each WRITING and
EVICTING only its own root, each reading the other's files through the other's published index. Real vLLM modules (the
same harness as test_gsch_fs_tier). Run: python -m unittest -v test_gsch_fs_tier_xpod"""
import json, os, tempfile, time, unittest

import numpy as np
from test_gsch_fs_tier import ROW, _Region, fill, keys, spec, wait  # also installs the vllm package stub
from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
import gsch_fs_tier as T

END = ScheduleEndContext(new_req_ids=[], preempted_req_ids=[])
SYNC = 0.05


class Pair(unittest.TestCase):
    """Engine b and engine c, roots under one shared base (the PVC)."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="gschfs-xpod-")
        self.rb, self.rc = os.path.join(self.base, "b"), os.path.join(self.base, "c")
        self.jid = 0

    def mk(self, root, peer, chunks=32, **kw):
        arr = np.zeros((chunks, ROW), dtype=np.uint8)
        kw.setdefault("n_read_threads", 4); kw.setdefault("n_write_threads", 4); kw.setdefault("stats_every_s", 1e9)
        kw.setdefault("peer_sync_s", SYNC)
        t = T.GschFsTier(spec(), memoryview(arr), "GschFsTier", root, peer_roots=[peer] if peer else None, **kw)
        self.addCleanup(t.shutdown)
        return t, arr

    def job(self, ks, rows, promotion):
        self.jid += 1
        return TransferJob(job_id=self.jid, keys=ks, chunk_ids=np.array(rows, dtype=np.int32), is_promotion=promotion,
                           req_context=ReqContext(req_id=f"r{self.jid}"))

    def finish(self, tier, jid, timeout=10.0):
        got = {}
        ok = wait(lambda: (tier.on_schedule_end(END) or True)
                  and (got.update({r.job_id: r for r in tier.get_finished_jobs()}) or jid in got), timeout)
        self.assertTrue(ok, f"job {jid} never finished")
        return got[jid]

    def store(self, tier, arr, ks, rows, seed):
        fill(arr, rows, seed)
        data = {k: arr[r].copy() for k, r in zip(ks, rows)}
        j = self.job(ks, rows, False); tier.submit_store(j)
        self.assertTrue(self.finish(tier, j.job_id).success)
        return data

    def sees(self, reader, owner, ks, timeout=5.0):
        """The owner publishes (its scheduler hands off on on_schedule_end), the reader's peer thread picks it up."""
        return wait(lambda: (owner.on_schedule_end(END) or True) and (reader.on_schedule_end(END) or True)
                    and all(reader.lookup(k, None) is LookupResult.HIT for k in ks), timeout)


class TestCrossPod(Pair):
    def test_peer_chain_is_read_identical_and_never_indexed_here(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        ks = keys(50, 6); data = self.store(tc, ac, ks, list(range(6)), 50)
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in ks), "nothing before c publishes")
        self.assertTrue(self.sees(tb, tc, ks), "b sees c's chain through c's published index")
        j = self.job(ks, list(range(10, 16)), True); tb.submit_load(j)
        self.assertTrue(self.finish(tb, j.job_id).success)
        for k, r in zip(ks, range(10, 16)):
            np.testing.assert_array_equal(ab[r], data[k], "b's CPU slot holds c's bytes")
        self.assertEqual(tb.stats["peer_load_ok"], 6)
        self.assertEqual(len(tb._index), 0, "a peer's chunk is never indexed as ours")
        self.assertTrue(all(os.path.exists(tc.file_mapper.get_file_name(k)) for k in ks))
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.HIT for k in ks))

    def test_failed_peer_read_misses_and_never_deletes_the_peers_files(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc, n_load_slices=4)
        ks = keys(51, 4); self.store(tc, ac, ks, list(range(4)), 51)
        self.assertTrue(self.sees(tb, tc, ks))
        os.unlink(tc.file_mapper.get_file_name(ks[2]))  # the owner evicted it after its last snapshot
        deletes, real_delete = [], tb._delete
        tb._delete = lambda k: (deletes.append(k), real_delete(k))  # our delete path is for OUR files only
        j = self.job(ks, list(range(4)), True); tb.submit_load(j)
        r = self.finish(tb, j.job_id)
        self.assertFalse(r.success)
        self.assertEqual(tb.stats["peer_load_failed"], 1)
        self.assertIs(tb.lookup(ks[2], None), LookupResult.MISS, "the failed key is not promoted again")
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.HIT for k in (ks[0], ks[1], ks[3])))
        time.sleep(0.3)
        self.assertTrue(all(os.path.exists(tc.file_mapper.get_file_name(k)) for k in (ks[0], ks[1], ks[3])),
                        "b deleted nothing of c's")
        self.assertEqual(tb.stats["deleted"], 0)
        self.assertEqual(deletes, [], "a failed peer read never goes down our delete path (it could hit our own copy)")

    def test_mixed_chain_own_and_peer_in_one_job(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        pk = keys(52, 3); dp = self.store(tc, ac, pk, [0, 1, 2], 52)  # the prefix, written while the session was on c
        ok_ = keys(53, 3); do = self.store(tb, ab, ok_, [3, 4, 5], 53)  # its later turns, written on b
        self.assertTrue(self.sees(tb, tc, pk))
        chain = pk + ok_
        j = self.job(chain, list(range(20, 26)), True); tb.submit_load(j)
        self.assertTrue(self.finish(tb, j.job_id).success)
        for k, r in zip(chain, range(20, 26)):
            np.testing.assert_array_equal(ab[r], {**dp, **do}[k])
        self.assertEqual((tb.stats["load_ok"], tb.stats["peer_load_ok"]), (6, 3))

    def test_touches_are_forwarded_so_the_owner_keeps_a_moved_chain(self):
        tc, ac = self.mk(self.rc, self.rb, max_bytes=4 * ROW)
        tb, ab = self.mk(self.rb, self.rc)
        a = keys(54, 4); self.store(tc, ac, a, [0, 1, 2, 3], 54)  # c's disk is full; a[0] is c's least recent
        self.assertTrue(self.sees(tb, tc, a))
        tb.touch([a[0]], ReqContext(req_id="moved"))  # the session now runs on b and uses its prefix
        self.assertTrue(wait(lambda: (tb.on_schedule_end(END) or True) and (tc.on_schedule_end(END) or True)
                             and tc.stats["touch_applied"] >= 1), "c applies b's touch")
        self.assertEqual(list(tc._index)[-1], a[0], "a[0] is now c's most recent")
        self.store(tc, ac, keys(55, 1), [8], 55)  # c needs room for one more
        self.assertIn(a[0], tc._index, "the moved session's prefix survives")
        self.assertNotIn(a[1], tc._index, "the least recent of the rest goes instead")

    def test_follows_the_peers_newest_run(self):
        tc1, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        x = keys(56, 2); self.store(tc1, ac, x, [0, 1], 56)
        self.assertTrue(self.sees(tb, tc1, x))
        tc1.shutdown()
        time.sleep(1.1)  # a later start timestamp
        sweep = T.GschFsTier._sweep_old_runs
        T.GschFsTier._sweep_old_runs = lambda self, *a, **k: None  # the old run is still on disk (a sweep takes minutes)
        try:
            tc2, ac2 = self.mk(self.rc, self.rb)  # c restarts: a fresh run dir
        finally:
            T.GschFsTier._sweep_old_runs = sweep
        self.assertEqual(len([n for n in os.listdir(self.rc) if n.startswith(T.RUN_PREFIX)]), 2)
        y = keys(57, 2); self.store(tc2, ac2, y, [0, 1], 57)
        self.assertTrue(self.sees(tb, tc2, y), "b follows c's new incarnation")
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in x), "the old run is not served")

    def test_a_peer_run_with_another_layout_is_refused(self):
        tb, ab = self.mk(self.rb, self.rc)
        run = os.path.join(self.rc, T.RUN_PREFIX + "29990101T000000-deadbeef")
        mapper = T.FileMapper.from_offloading_spec(root_dir=run, offloading_spec=tb._spec, blocks_per_file=1,
                                                   parallel_agnostic=True)
        os.makedirs(os.path.dirname(mapper.get_config_file_path()), exist_ok=True)
        cfg = json.loads(json.dumps(tb.file_mapper.get_run_config())); cfg["__other__"] = 1
        with open(mapper.get_config_file_path(), "w") as f:
            json.dump(cfg, f)
        ks = keys(58, 2)
        T._write_keys(os.path.join(run, T.INDEX_FILE), ks, 1)
        self.assertTrue(wait(lambda: tb.stats["peer_layout_refused"] >= 1))
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in ks))

    def test_no_forwarded_touch_is_lost_when_hand_offs_pile_up(self):
        tb, _ = self.mk(self.rb, self.rc, peer_sync_s=1e6)  # the peer thread never drains during the test
        a, b = keys(61, 2), keys(62, 2)
        tb._outbox.put((a, set(a))); tb._outbox.put((b, set(b)))
        idx, touched = tb._drain_outbox()
        self.assertEqual((idx, touched), (b, set(a) | set(b)), "the newest index, every touch")

    def test_index_file_round_trip_and_torn_file_rejected(self):
        p = os.path.join(self.base, "idx.bin"); os.makedirs(self.base, exist_ok=True)
        T._write_keys(p, [], 3)
        self.assertEqual(T._read_keys(p, None), (3, []), "an empty index reads back empty")
        ks = keys(59, 5)
        T._write_keys(p, ks, 7)
        self.assertEqual(T._read_keys(p, None), (7, ks))
        self.assertEqual(T._read_keys(p, 7), (7, None), "an unchanged sequence is not re-read")
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 3)
        with self.assertRaises(ValueError):
            T._read_keys(p, None)


class TestCrossPodEndToEnd(unittest.TestCase):
    """Two engines, each vLLM's TieringOffloadingManager + CPU primary (GroupMemberLRU) + GschFsTier: a chain computed on c is
    promoted into b's CPU tier from c's files, bytes identical — the returner that the router placed on the other engine."""

    def engine(self, root, peer):
        arr = np.zeros((8, ROW), dtype=np.uint8)
        prim = CPUPrimaryTierOffloadingManager(num_chunks=8, mmap_region=_Region(arr), cache_policy="GroupMemberLRU",
                                               cache_policy_module_path="gsch_member_policy")
        tier = T.GschFsTier(spec(), prim.get_kv_memoryview(), "GschFsTier", root, n_read_threads=2, n_write_threads=2,
                            stats_every_s=1e9, peer_roots=[peer], peer_sync_s=SYNC)
        mgr = TieringOffloadingManager(prim, [tier]); self.addCleanup(mgr.shutdown)
        return mgr, tier, arr

    def test_chain_written_on_c_is_promoted_on_b(self):
        base = tempfile.mkdtemp(prefix="gschfs-xpod-e2e-")
        mb, tb, ab = self.engine(os.path.join(base, "b"), os.path.join(base, "c"))
        mc, tc, ac = self.engine(os.path.join(base, "c"), os.path.join(base, "b"))
        a = keys(60, 6)
        ctx = ReqContext(req_id="A@c"); mc.on_new_request(ctx)
        out = mc.prepare_store(a, ctx); rows = [int(c) for c in out.store_spec.chunk_ids]
        fill(ac, rows, 60); data = {k: ac[r].copy() for k, r in zip(out.keys_to_store, rows)}
        mc.complete_store(out.keys_to_store, ctx, True); mc.on_request_finished(ctx)
        self.assertTrue(wait(lambda: (mc.on_schedule_end(END) or True) and (mb.on_schedule_end(END) or True)
                             and all(tb.lookup(k, None) is LookupResult.HIT for k in a)), "written on c, visible on b")
        ctx = ReqContext(req_id="A@b"); mb.on_new_request(ctx)
        res = [mb.lookup(k, ctx) for k in a]
        self.assertTrue(all(r is LookupResult.HIT_PENDING for r in res), res)
        self.assertTrue(wait(lambda: (mb.on_schedule_end(END) or True)
                             and all(mb.lookup(k, ctx) is LookupResult.HIT for k in a)), "promoted into b's CPU tier")
        spec_ = mb.prepare_load(a, ctx)
        for k, r in zip(a, [int(c) for c in spec_.chunk_ids]):
            np.testing.assert_array_equal(ab[r], data[k], "b's CPU tier holds c's bytes")
        mb.complete_load(a, ctx); mb.on_request_finished(ctx)
        self.assertEqual(tb.stats["peer_load_ok"], 6)


if __name__ == "__main__":
    unittest.main()
