# SPDX-License-Identifier: Apache-2.0
"""Independent verifier's tests for the cross-pod reads of gsch_fs_tier (commit bdfe9e78, 2026-11-29). Same harness as
test_gsch_fs_tier_xpod (vLLM's REAL tiering sources, stub vllm package, two tiers on two roots of one temp filesystem).
Tests marked expectedFailure expose a bug or a risk; reasons and fixes are in
sai-ka-knowledge-bank/explore/agent-bench-replay/independent-analysis/GPFS-TIER-XPOD-VERIFY-2026-11-29.md.
Run: python -m unittest -v test_gsch_fs_tier_xpod_verify      (XPOD_BENCH=1 also runs the production-size benchmark)"""
import hashlib, os, random, statistics, sys, tempfile, threading, time, unittest

import numpy as np
from test_gsch_fs_tier import ROW, _Region, keys, spec, wait  # also installs the vllm package stub
from test_gsch_fs_tier_xpod import END, SYNC, Pair
import test_gsch_fs_tier_verify as V  # for _build_fsio_c (the C fast path compiled from csrc/fs_io.cpp)
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
import vllm.v1.kv_offload.tiering.fs.io as fsio
import gsch_fs_tier as T


def pattern(k):
    """Deterministic bytes for a key: a re-store of the same key writes the same content (content addressing)."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(k).digest()[:8], "big")).integers(0, 256, ROW, dtype=np.uint8)


def wrap_tasks(tier, before):
    """Wrap every pool slice the tier builds: before(job, i, kind, paths) runs first on the pool thread; returning "fail"
    makes the slice fail as an I/O error would (started, 0 keys ok). Independent of which io function the tier calls."""
    orig = tier._task

    def task(job, i, kind, paths, offs):
        run = orig(job, i, kind, paths, offs)

        def wrapped():
            if before(job, i, kind, paths) == "fail":
                job.started[i] = True; job.ok[i] = 0
                return
            run()
        return wrapped
    tier._task = task
    return lambda: tier.__dict__.pop("_task", None)


def tree(root):
    out = {}
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


class XV(Pair):
    def quiesce(self, reader, owner=None, settle=0.25):
        """Stop driving the owner's scheduler: it hands off nothing more, so the reader's view generation stays fixed."""
        time.sleep(settle)
        return reader._peers[0].view.gen

    def bump_gen(self, reader, owner):
        """Let the owner publish once more (same keys, new seq) so the reader swaps in a new view generation."""
        g = reader._peers[0].view.gen
        time.sleep(SYNC * 1.5)
        owner.on_schedule_end(END)
        self.assertTrue(wait(lambda: reader._peers[0].view.gen > g), "reader never picked up the new snapshot")
        time.sleep(0.25)
        return reader._peers[0].view.gen


class TestPeerFilesAreNeverModified(XV):
    @unittest.expectedFailure  # BUG-X1: vLLM's io (_load_block / csrc fs_io.cpp _load_block) unlinks the SOURCE file on a
    def test_short_peer_file_is_never_deleted_by_the_reader(self):  # short read: a reader deletes the OWNER's file
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        ks = keys(70, 2); self.store(tc, ac, ks, [0, 1], 70)
        self.assertTrue(self.sees(tb, tc, ks))
        path = tc.file_mapper.get_file_name(ks[1])
        with open(path, "r+b") as f:
            f.truncate(ROW // 2)  # a corrupt / short file in c's root (c still indexes it)
        j = self.job(ks, [4, 5], True); tb.submit_load(j)
        self.assertFalse(self.finish(tb, j.job_id).success)
        self.assertIs(tb.lookup(ks[1], None), LookupResult.MISS)
        self.assertTrue(os.path.exists(path), "b (a reader) deleted a file in c's root")

    @unittest.expectedFailure  # BUG-X1 on the production path: the C fast path (csrc/fs_io.cpp _load_block) unlinks too
    def test_short_peer_file_is_never_deleted_c_fast_path(self):
        C = V._build_fsio_c()
        if C is None:
            self.skipTest("no C++ compiler / csrc")
        saved = (fsio._HAS_FSIO_C, getattr(fsio, "batch_load_block_C", None), getattr(fsio, "batch_store_block_C", None))
        fsio._HAS_FSIO_C, fsio.batch_load_block_C, fsio.batch_store_block_C = True, C.batch_load_block, C.batch_store_block
        try:
            tc, ac = self.mk(self.rc, self.rb)
            tb, ab = self.mk(self.rb, self.rc)
            ks = keys(71, 2); self.store(tc, ac, ks, [0, 1], 71)
            self.assertTrue(self.sees(tb, tc, ks))
            path = tc.file_mapper.get_file_name(ks[1])
            with open(path, "r+b") as f:
                f.truncate(ROW // 2)
            j = self.job(ks, [4, 5], True); tb.submit_load(j)
            self.assertFalse(self.finish(tb, j.job_id).success)
            self.assertTrue(os.path.exists(path), "b (a reader, C io) deleted a file in c's root")
        finally:
            fsio._HAS_FSIO_C, fsio.batch_load_block_C, fsio.batch_store_block_C = saved

    def test_reader_never_writes_creates_or_deletes_in_the_peer_root(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc, n_load_slices=2, load_deadline_s=0.0)
        ks = keys(72, 6); self.store(tc, ac, ks, list(range(6)), 72)
        own = keys(73, 2); self.store(tb, ab, own, [10, 11], 73)
        self.assertTrue(self.sees(tb, tc, ks))
        os.unlink(tc.file_mapper.get_file_name(ks[4]))  # a stale-snapshot miss (c evicted it after publishing)
        self.quiesce(tb, tc)
        before = tree(self.rc)
        for _ in range(3):  # lookups, touches, loads (ok, failed, deadline-cancelled), publishes
            [tb.lookup(k, None) for k in ks + own]
            tb.touch(ks + own, ReqContext(req_id="x"))
            j = self.job(ks + own, list(range(16, 24)), True); tb.submit_load(j); self.finish(tb, j.job_id)
            tb.on_schedule_end(END)
        time.sleep(0.3)
        self.assertEqual(tree(self.rc), before, "c's root changed while only b was active")


class TestBadMarksAcrossGenerations(XV):
    # fixed — was xfail:  # BUG-X2: get_finished_jobs resets the peer's bad set to the FINISHING job's gen, so a late
    def test_late_failure_from_older_gen_does_not_clobber_current_marks(self):  # old-gen failure erases current marks
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        k1, k2 = keys(74, 2); self.store(tc, ac, [k1, k2], [0, 1], 74)
        self.assertTrue(self.sees(tb, tc, [k1, k2]))
        g = self.quiesce(tb, tc)
        gate = threading.Event()
        slow = tc.file_mapper.get_file_name(k1)
        unwrap = wrap_tasks(tb, lambda job, i, kind, paths: "fail" if slow in paths and gate.wait(10) else None)
        try:
            j1 = self.job([k1], [4], True); tb.submit_load(j1)  # read under gen g, still in flight
            g2 = self.bump_gen(tb, tc)
            self.assertGreater(g2, g)
            os.unlink(tc.file_mapper.get_file_name(k2))
            j2 = self.job([k2], [5], True); tb.submit_load(j2)  # read under gen g2, fails at once
            self.assertFalse(self.finish(tb, j2.job_id).success)
            self.assertIs(tb.lookup(k2, None), LookupResult.MISS, "k2 marked bad in the current gen")
            gate.set()
            self.assertFalse(self.finish(tb, j1.job_id).success)
        finally:
            gate.set(); unwrap()
        self.assertIs(tb.lookup(k2, None), LookupResult.MISS, "the late gen-g failure erased k2's current-gen mark")

    # fixed — was xfail:  # BUG-X2 (variant): the peer thread swaps the view in the middle of submit_load's per-key
    def test_one_job_spanning_two_gens_keeps_every_mark(self):  # loop; one job then carries two gens and loses marks
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc, n_load_slices=1)
        ks = keys(75, 3); self.store(tc, ac, ks, [0, 1, 2], 75)
        self.assertTrue(self.sees(tb, tc, ks))
        self.quiesce(tb, tc)
        for k in ks:
            os.unlink(tc.file_mapper.get_file_name(k))
        orig, n = tb._peer_of, [0]

        def swapping(key):
            r = orig(key)
            n[0] += 1
            if n[0] == 1:  # what the peer thread may do between two iterations of submit_load's loop
                p = tb._peers[0]; v = p.view
                p.view = T._PeerView(v.run_dir, v.mapper, v.keys, v.gen + 1)
            return r
        tb._peer_of = swapping
        j = self.job(ks, [4, 5, 6], True); tb.submit_load(j)
        del tb._peer_of
        self.assertFalse(self.finish(tb, j.job_id).success)
        self.assertEqual([tb.lookup(k, None) for k in ks], [LookupResult.MISS] * 3, "a failed key is served again")

    def test_view_swapped_between_lookup_and_submit_load_is_safe(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        k = keys(76, 1); self.store(tc, ac, k, [0], 76)
        self.assertTrue(self.sees(tb, tc, k))
        self.quiesce(tb, tc)
        self.assertIs(tb.lookup(k[0], None), LookupResult.HIT)
        p = tb._peers[0]; v = p.view
        p.view = T._PeerView(v.run_dir, v.mapper, frozenset(), v.gen + 1)  # c's next snapshot no longer lists it
        j = self.job(k, [3], True); tb.submit_load(j)
        self.assertFalse(self.finish(tb, j.job_id).success, "falls back to our own (absent) path: a clean miss")
        self.assertTrue(os.path.exists(tc.file_mapper.get_file_name(k[0])), "c's file untouched")
        self.assertEqual((tb.stats["load_failed"], tb.stats["peer_load_failed"]), (1, 0), "counted as an own failure (nit)")
        self.assertTrue(wait(lambda: not tb._pending_del), "the spurious own delete drains")

    def test_deadline_cancelled_peer_read_marks_bad_for_one_gen_only(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc, n_load_slices=4, n_read_threads=1, n_write_threads=1, load_deadline_s=0.05)
        ks = keys(77, 8); self.store(tc, ac, ks, list(range(8)), 77)
        self.assertTrue(self.sees(tb, tc, ks))
        self.quiesce(tb, tc)
        unwrap = wrap_tasks(tb, lambda job, i, kind, paths: time.sleep(0.3))
        try:
            j = self.job(ks, list(range(16, 24)), True); tb.submit_load(j)
            time.sleep(0.1); tb.on_schedule_end(END)
            r = self.finish(tb, j.job_id)
        finally:
            unwrap()
        self.assertFalse(r.success)
        cancelled = [k for k in ks if k not in set(r.successful_keys or ())]
        self.assertTrue(cancelled)
        self.assertTrue(all(os.path.exists(tc.file_mapper.get_file_name(k)) for k in ks), "nothing of c's deleted")
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in cancelled), "skipped in this gen")
        self.bump_gen(tb, tc)
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.HIT for k in cancelled), "served again next snapshot")


class TestOwnVersusPeer(XV):
    def test_same_key_in_both_reads_own_and_falls_back_to_the_peer(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc, n_load_slices=1)
        k = keys(78, 1)
        data = self.store(tc, ac, k, [0], 78)
        self.assertTrue(self.sees(tb, tc, k))  # before b has its own copy (sees() checks lookup, own index first)
        self.store(tb, ab, k, [0], 78)
        def load():
            j = self.job(k, [3], True); tb.submit_load(j)
            src = tb._jobs[j.job_id].src
            return src, self.finish(tb, j.job_id)
        src, r = load()
        self.assertTrue(r.success); self.assertIsNone(src, "own copy preferred (no peer source)")
        os.unlink(tb.file_mapper.get_file_name(k[0]))  # our own copy is lost
        src, r = load()
        self.assertFalse(r.success); self.assertIsNone(src)
        self.assertNotIn(k[0], tb._index, "own failure: dropped from our index")
        self.assertIs(tb.lookup(k[0], None), LookupResult.HIT, "now served from the peer")
        ab[3] = 0
        src, r = load()
        self.assertTrue(r.success); self.assertEqual(src[0][0], 0, "read from peer 0")
        np.testing.assert_array_equal(ab[3], data[k[0]])
        self.assertTrue(os.path.exists(tc.file_mapper.get_file_name(k[0])))

    def test_touch_of_a_key_in_both_stays_local(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        k = keys(79, 1)
        self.store(tc, ac, k, [0], 79)
        self.assertTrue(self.sees(tb, tc, k))
        self.store(tb, ab, k, [0], 79)
        tb._touched_out.clear()
        tb.touch(k, ReqContext(req_id="x"))
        self.assertEqual(tb._touched_out, set(), "our own copy is refreshed; the peer's duplicate may age out")


class TestPeerLifecycle(XV):
    def test_peer_down_keeps_serving_its_last_snapshot(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        ks = keys(80, 3); data = self.store(tc, ac, ks, [0, 1, 2], 80)
        self.assertTrue(self.sees(tb, tc, ks))
        tc.shutdown()  # the pod is gone; its files stay on the PVC (only its next start sweeps them)
        time.sleep(0.3)
        j = self.job(ks, [8, 9, 10], True); tb.submit_load(j)
        self.assertTrue(self.finish(tb, j.job_id).success)
        for kk, r in zip(ks, [8, 9, 10]):
            np.testing.assert_array_equal(ab[r], data[kk])

    def test_restart_window_old_view_then_switch(self):
        tc1, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        x = keys(81, 3); self.store(tc1, ac, x, [0, 1, 2], 81)
        self.assertTrue(self.sees(tb, tc1, x))
        old_paths = [tc1.file_mapper.get_file_name(k) for k in x]
        tc1.shutdown(); time.sleep(1.1)
        sweep = T.GschFsTier._sweep_old_runs
        T.GschFsTier._sweep_old_runs = lambda self, *a, **k: None
        try:
            tc2, ac2 = self.mk(self.rc, self.rb)  # new run dir + config.json, no index until its first hand-off
        finally:
            T.GschFsTier._sweep_old_runs = sweep
        time.sleep(0.3)
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.HIT for k in x), "old snapshot kept until c2 publishes")
        os.unlink(old_paths[1])  # c2's sweep has reached one of them
        j = self.job(x, [8, 9, 10], True); tb.submit_load(j)
        r = self.finish(tb, j.job_id)
        self.assertFalse(r.success)
        self.assertEqual(set(r.successful_keys), {x[0], x[2]})
        self.assertTrue(os.path.exists(old_paths[0]) and os.path.exists(old_paths[2]), "b deleted nothing")
        y = keys(82, 1); self.store(tc2, ac2, y, [0], 82)
        self.assertTrue(self.sees(tb, tc2, y))
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in x))

    def test_layout_mismatch_is_never_read_and_is_reported(self):  # fixed N-X1: it used to be silent
        tb, ab = self.mk(self.rb, self.rc)
        other = spec(); other.config.model.name = "another-model"  # a real mismatch changes the digest -> the path
        arr = np.zeros((8, ROW), dtype=np.uint8)
        tc = T.GschFsTier(other, memoryview(arr), "GschFsTier", self.rc, peer_roots=[self.rb], peer_sync_s=SYNC,
                          stats_every_s=1e9, n_read_threads=2, n_write_threads=2)
        self.addCleanup(tc.shutdown)
        ks = keys(83, 2); self.store(tc, arr, ks, [0, 1], 83)
        for _ in range(20):
            tc.on_schedule_end(END); tb.on_schedule_end(END); time.sleep(SYNC)
        self.assertIsNone(tb._peers[0].view, "never followed")
        self.assertTrue(all(tb.lookup(k, None) is LookupResult.MISS for k in ks))
        self.assertEqual(tb.stats["peer_layout_refused"], 1, "a real mismatch is refused once and reported")

    def test_publish_enospc_keeps_the_old_snapshot_and_recovers(self):
        tc, ac = self.mk(self.rc, self.rb)
        tb, ab = self.mk(self.rb, self.rc)
        ks = keys(84, 2); self.store(tb, ab, ks, [0, 1], 84)
        self.assertTrue(self.sees(tc, tb, ks))
        g, real = self.quiesce(tc, tb), T._write_keys  # let b's last hand-off land first (else gen moves under us)

        def full(path, keys_, seq):
            if path.startswith(tb._run_root):
                raise OSError(28, "No space left on device")
            return real(path, keys_, seq)
        T._write_keys = full
        try:
            more = keys(85, 1); self.store(tb, ab, more, [2], 85)
            self.assertTrue(wait(lambda: (tb.on_schedule_end(END) or True) and tb.stats["publish_errors"] >= 2))
            self.assertEqual(tc._peers[0].view.gen, g, "c keeps b's last good snapshot")
            self.assertIs(tc.lookup(more[0], None), LookupResult.MISS)
        finally:
            T._write_keys = real
        self.assertTrue(self.sees(tc, tb, ks + more), "publishing resumes")


class TestTouchForwarding(XV):
    def run_rounds(self, forward):
        tc, ac = self.mk(self.rc, self.rb, max_bytes=6 * ROW)
        tb, ab = self.mk(self.rb, self.rc)
        chain = keys(86, 2); self.store(tc, ac, chain, [0, 1], 86)  # the moved session's prefix, on c
        self.assertTrue(self.sees(tb, tc, chain))
        for i in range(12):  # c keeps writing its own sessions: 2x its cap over the test
            self.store(tc, ac, keys(87, 12)[i:i + 1], [2 + i % 8], 87 + i)
            if forward:  # the session keeps running on b, whose connector touches its whole chain
                n = tc.stats["touch_applied"]
                tb.touch(chain, ReqContext(req_id="moved"))
                self.assertTrue(wait(lambda: (tb.on_schedule_end(END) or True) and (tc.on_schedule_end(END) or True)
                                     and tc.stats["touch_applied"] > n), f"round {i}: touch never reached c")
        return tc, chain

    def test_forwarded_touches_keep_a_moved_chain_resident_over_two_cap_turnovers(self):
        tc, chain = self.run_rounds(True)
        self.assertTrue(all(k in tc._index for k in chain), "the moved prefix survived 2x cap of c's own writes")

    def test_control_without_forwarding_the_chain_ages_out(self):
        tc, chain = self.run_rounds(False)
        self.assertFalse(any(k in tc._index for k in chain), "without forwarding c's LRU drops it")

    def test_owner_without_peer_roots_never_applies_forwarded_touches(self):
        tc, ac = self.mk(self.rc, None)  # c publishes but does not follow b: an ASYMMETRIC config
        tb, ab = self.mk(self.rb, self.rc)
        chain = keys(88, 2); self.store(tc, ac, chain, [0, 1], 88)
        self.assertTrue(self.sees(tb, tc, chain))
        for _ in range(10):
            tb.touch(chain, ReqContext(req_id="moved"))
            tb.on_schedule_end(END); tc.on_schedule_end(END); time.sleep(SYNC)
        self.assertGreater(tb.stats["touch_forwarded"], 0)
        self.assertEqual(tc.stats["touch_applied"], 0, "config requirement: the OWNER must list the reader in peer_roots")

    # fixed — was xfail:  # RISK-X1: while the owner's scheduler is idle (no on_schedule_end) its peer thread keeps
    def test_idle_owner_touch_backlog_is_bounded(self):  # queueing every forwarded-touch list: unbounded memory, and the
        tc, ac = self.mk(self.rc, self.rb)  # whole backlog is applied in ONE scheduler step when it wakes up
        tb, ab = self.mk(self.rb, self.rc)
        chain = keys(89, 4); self.store(tc, ac, chain, [0, 1, 2, 3], 89)
        self.assertTrue(self.sees(tb, tc, chain))
        time.sleep(0.3)  # c idle from here on
        while not tc._touch_in.empty():
            tc._touch_in.get_nowait()
        for _ in range(15):  # b keeps serving the moved session
            tb.touch(chain, ReqContext(req_id="moved"))
            time.sleep(SYNC * 1.2); tb.on_schedule_end(END)
        time.sleep(0.3)
        self.assertLessEqual(tc._touch_in.qsize(), 2, f"{tc._touch_in.qsize()} touch lists queued on an idle owner")


class TestManagerInterplay(unittest.TestCase):
    """vLLM's real TieringOffloadingManager on both engines: a stale peer snapshot names a missing file."""

    def engine(self, root, peer, n=16):
        arr = np.zeros((n, ROW), dtype=np.uint8)
        prim = CPUPrimaryTierOffloadingManager(num_chunks=n, mmap_region=_Region(arr), cache_policy="GroupMemberLRU",
                                               cache_policy_module_path="gsch_member_policy")
        tier = T.GschFsTier(spec(), prim.get_kv_memoryview(), "GschFsTier", root, n_read_threads=2, n_write_threads=2,
                            stats_every_s=1e9, peer_roots=[peer], peer_sync_s=SYNC)
        mgr = TieringOffloadingManager(prim, [tier]); self.addCleanup(mgr.shutdown)
        return mgr, tier, arr

    def step(self, mgr, ctx, ks):
        res = [mgr.lookup(k, ctx) for k in ks]
        mgr.touch(ks, ctx)
        mgr.on_schedule_end(END)
        return res

    def test_failed_peer_promotion_is_retried_at_most_once_per_snapshot(self):
        base = tempfile.mkdtemp(prefix="gschfs-xpodv-mgr-")
        mb, tb, ab = self.engine(os.path.join(base, "b"), os.path.join(base, "c"))
        mc, tc, ac = self.engine(os.path.join(base, "c"), os.path.join(base, "b"))
        a = keys(90, 6)
        ctx = ReqContext(req_id="A@c"); mc.on_new_request(ctx)
        out = mc.prepare_store(a, ctx); rows = [int(c) for c in out.store_spec.chunk_ids]
        for k, r in zip(out.keys_to_store, rows):
            ac[r] = pattern(k)
        mc.complete_store(out.keys_to_store, ctx, True); mc.on_request_finished(ctx)
        self.assertTrue(wait(lambda: (mc.on_schedule_end(END) or True) and (mb.on_schedule_end(END) or True)
                             and all(tb.lookup(k, None) is LookupResult.HIT for k in a)))
        time.sleep(0.3)  # c idle: b's view gen fixed
        os.unlink(tc.file_mapper.get_file_name(a[3]))  # the file is gone but c still lists it
        ctx = ReqContext(req_id="A@b"); mb.on_new_request(ctx)

        def prefix(res):
            n = 0
            for r in res:
                if r is LookupResult.MISS:
                    break
                n += 1
            return n
        res = self.step(mb, ctx, a)
        self.assertTrue(all(r is LookupResult.HIT_PENDING for r in res), res)
        self.assertTrue(wait(lambda: self.step(mb, ctx, a)[:3] == [LookupResult.HIT] * 3))
        jobs = tb.stats["load_jobs"]
        for _ in range(20):  # same snapshot: the bad mark holds, no re-promotion loop
            res = self.step(mb, ctx, a); time.sleep(0.005)
            self.assertEqual(prefix(res), 3)
            self.assertIs(res[3], LookupResult.MISS)
        self.assertEqual(tb.stats["load_jobs"], jobs, "no new promotion while the snapshot is unchanged")
        g = tb._peers[0].view.gen
        time.sleep(SYNC * 1.5); mc.on_schedule_end(END)  # c publishes again, still listing a[3]
        self.assertTrue(wait(lambda: tb._peers[0].view.gen > g))
        time.sleep(0.2)
        self.assertTrue(wait(lambda: self.step(mb, ctx, a)[3] is LookupResult.MISS and tb.stats["load_jobs"] > jobs))
        self.assertEqual(tb.stats["load_jobs"], jobs + 1, "exactly one wasted promotion per new snapshot")
        self.assertEqual(tb.stats["peer_load_failed"], 2)
        self.assertTrue(os.path.exists(tc.file_mapper.get_file_name(a[0])))
        prim_view = {k: mb.primary_tier.lookup(k, ctx) for k in a[:3]}
        self.assertTrue(all(v is LookupResult.HIT for v in prim_view.values()))
        spec_ = mb.prepare_load(a[:3], ctx)
        for k, r in zip(a[:3], [int(c) for c in spec_.chunk_ids]):
            np.testing.assert_array_equal(ab[r], pattern(k), "b's CPU slot holds c's bytes")
        mb.complete_load(a[:3], ctx); mb.on_request_finished(ctx)


class TestConcurrentChurn(Pair):
    """b's scheduler (this thread) promotes c's chunks while c's scheduler (another thread) stores and EVICTS at full speed
    under a small cap, and both peer threads publish every 20 ms. Every loaded byte must equal the key's content."""

    def test_no_torn_or_wrong_bytes_no_lost_ids_no_peer_deletes(self):
        sync = 0.02
        tc, ac = self.mk(self.rc, self.rb, chunks=32, max_bytes=24 * ROW, peer_sync_s=sync)
        tb, ab = self.mk(self.rb, self.rc, chunks=64, n_load_slices=4, peer_sync_s=sync)
        universe = keys(91, 60)
        stop, errors, deleted_paths = threading.Event(), [], []
        real_put = tb._deletes.put
        tb._deletes.put = lambda item, *a, **k: (deleted_paths.append(item[1]), real_put(item, *a, **k))
        real_remove = os.remove
        os.remove = lambda p, *a, **k: (deleted_paths.append(p), real_remove(p, *a, **k))

        def c_sched():
            jid, rng = 10_000, random.Random(2)
            try:
                while not stop.is_set():
                    batch = rng.sample(universe, 4)
                    rows = rng.sample(range(32), 4)
                    for k, r in zip(batch, rows):
                        ac[r] = pattern(k)
                    jid += 1
                    from vllm.v1.kv_offload.tiering.base import TransferJob
                    tc.submit_store(TransferJob(job_id=jid, keys=batch, chunk_ids=np.array(rows, dtype=np.int32),
                                                is_promotion=False, req_context=ReqContext(req_id="c")))
                    done = False
                    while not done and not stop.is_set():
                        tc.on_schedule_end(END)
                        done = any(r.job_id == jid for r in tc.get_finished_jobs())
                        time.sleep(0.001)
                    tc.touch(rng.sample(universe, 10), None)
            except Exception as e:  # noqa: BLE001
                errors.append(("c", repr(e)))
        th = threading.Thread(target=c_sched); th.start()
        reported, mismatches, rng, t0 = set(), 0, random.Random(3), time.monotonic()
        loads = ok_bytes = 0
        try:
            while time.monotonic() - t0 < 3.0:
                hits = [k for k in rng.sample(universe, 12) if tb.lookup(k, None) is LookupResult.HIT]
                tb.touch(universe, None)
                if hits:
                    rows = rng.sample(range(64), len(hits))
                    ab[rows] = 0
                    j = self.job(hits, rows, True); tb.submit_load(j)
                    got = None
                    while got is None:
                        tb.on_schedule_end(END)
                        for r in tb.get_finished_jobs():
                            if r.job_id in reported:
                                errors.append(("dup", r.job_id))
                            reported.add(r.job_id)
                            if r.job_id == j.job_id:
                                got = r
                        time.sleep(0.0005)
                    good = set(hits) if got.success else set(got.successful_keys or ())
                    loads += 1
                    for k, row in zip(hits, rows):
                        if k in good:
                            ok_bytes += 1
                            if not np.array_equal(ab[row], pattern(k)):
                                mismatches += 1
                else:
                    tb.on_schedule_end(END)
        finally:
            stop.set(); th.join(10)
            os.remove = real_remove
        self.assertEqual(errors, [])
        self.assertEqual(mismatches, 0, "a successful peer read returned bytes that are not the key's")
        self.assertGreater(ok_bytes, 20, "the test must actually read across pods")
        self.assertEqual([p for p in deleted_paths if p.startswith(self.rc)], [], "b deleted in c's root")
        self.assertGreater(tc.stats["evicted"], 20, "c must actually evict under the readers")
        print(f"\n  churn: {loads} load jobs, {ok_bytes} chunks verified, peer_load_ok={tb.stats['peer_load_ok']} "
              f"peer_load_failed={tb.stats['peer_load_failed']} c evicted={tc.stats['evicted']} "
              f"c published={tc.stats['published']} b peer_reads={tb.stats['peer_reads']}", file=sys.stderr)


@unittest.skipUnless(os.environ.get("XPOD_BENCH"), "set XPOD_BENCH=1 for the production-size benchmark")
class TestBenchmark(unittest.TestCase):
    """Production sizes: 143K 36-byte keys per index, 1,300-key chains. Prints; asserts nothing about speed."""

    def test_bench(self):
        N, L = 143_000, 1300
        mk = lambda n: [os.urandom(32) + bytes(4) for _ in range(n)]  # noqa: E731
        own, peer, fresh = mk(N), mk(N), mk(L)
        from collections import OrderedDict
        tier = T.GschFsTier.__new__(T.GschFsTier)
        tier._index = OrderedDict((k, None) for k in own); tier._touched_out = set()
        p = T._Peer("/x"); p.view = T._PeerView("/x/r", None, frozenset(peer), 0); tier._peers = [p]

        def tm(f, rep=5):
            xs = []
            for _ in range(rep):
                t = time.perf_counter(); f(); xs.append(time.perf_counter() - t)
            return min(xs) * 1e3, statistics.median(xs) * 1e3
        d = tempfile.mkdtemp(prefix="xpod-bench-"); path = os.path.join(d, T.INDEX_FILE); lst = list(tier._index)
        T._write_keys(path, lst, 1)
        _, ks = T._read_keys(path, None)
        rows = [("scheduler: list(self._index) 143K", tm(lambda: list(tier._index))),
                ("peer: _write_keys 143K", tm(lambda: T._write_keys(path, lst, 1))),
                ("peer: _read_keys 143K", tm(lambda: T._read_keys(path, None))),
                ("peer: frozenset(143K) (one C call, GIL held)", tm(lambda: frozenset(ks))),
                ("peer: b''.join(143K) (one C call, GIL held)", tm(lambda: b"".join(lst)))]
        idx = tier._index

        def apply():
            for k in own:
                if k in idx:
                    idx.move_to_end(k)
        rows.append(("scheduler: apply 143K forwarded touches", tm(apply, 3)))
        for name, ch in (("own", own[:L]), ("peer", peer[:L]), ("neither", fresh)):
            rows.append((f"scheduler: touch() {L} keys, all {name}", tm(lambda ch=ch: tier.touch(ch, None), 50)))

        def old_touch(ch=fresh):
            for k in ch:
                if k in idx:
                    idx.move_to_end(k)
        rows.append((f"scheduler: 18083e4f touch() {L} keys, none own", tm(old_touch, 50)))

        def fast_touch(ch):  # candidate: one C-level set intersection per peer instead of _peer_of per key
            mte, rest = idx.move_to_end, []
            for k in ch:
                if k in idx:
                    mte(k)
                else:
                    rest.append(k)
            if rest:
                for pp in tier._peers:
                    v = pp.view
                    if v is not None:
                        hit = v.keys.intersection(rest)
                        if hit and pp.bad_gen == v.gen and pp.bad:
                            hit -= pp.bad
                        tier._touched_out |= hit
        for name, ch in (("peer", peer[:L]), ("neither", fresh)):
            rows.append((f"scheduler: candidate fast touch() {L} keys, all {name}", tm(lambda ch=ch: fast_touch(ch), 50)))
        for name, val in rows:
            print(f"  {name:55s} min {val[0]:8.3f} ms  med {val[1]:8.3f} ms", file=sys.stderr)

        # GIL: a scheduler-like thread runs ~1 ms pure-Python steps; the peer thread runs one full production cycle
        # (publish index 143K + touched 100K, read the peer's index 143K -> frozenset + touched 100K).
        tpath = os.path.join(d, T.TOUCH_FILE); T._write_keys(tpath, peer[:100_000], 1)

        def cycle():
            T._write_keys(path + ".pub", lst, 2); T._write_keys(tpath + ".pub", peer[:100_000], 2)
            _, k1 = T._read_keys(path, None); frozenset(k1); T._read_keys(tpath, None)
        for label, run_cycle in (("baseline", False), ("with peer cycle", True)):
            steps, stop = [], threading.Event()

            def sched():
                while not stop.is_set():
                    t = time.perf_counter(); x = 0
                    while time.perf_counter() - t < 0.001:
                        x += 1
                    steps.append(time.perf_counter() - t)
            th = threading.Thread(target=sched); th.start(); time.sleep(0.1); steps.clear()
            t0 = time.perf_counter()
            if run_cycle:
                cycle()
            else:
                time.sleep(0.15)
            wall = time.perf_counter() - t0
            time.sleep(0.02); stop.set(); th.join()
            xs = sorted(steps)
            lost = sum(xs) - len(xs) * 0.001
            print(f"  GIL {label:16s}: peer wall {wall*1e3:6.1f} ms, sched steps {len(xs)}, p50 {xs[len(xs)//2]*1e3:.2f} ms, "
                  f"p99 {xs[int(len(xs)*.99)]*1e3:.2f} ms, max {xs[-1]*1e3:.2f} ms, sched time lost {lost*1e3:.1f} ms",
                  file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
