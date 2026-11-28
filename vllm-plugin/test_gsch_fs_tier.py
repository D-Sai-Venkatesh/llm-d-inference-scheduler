# SPDX-License-Identifier: Apache-2.0
"""Tests for gsch_fs_tier against vLLM's REAL sources (VLLM_SRC, default /tmp/vllm-v0.30.0): FileSystemTierManager, io.py,
the thread pool, FileMapper, and — end to end — TieringOffloadingManager + CPUPrimaryTierOffloadingManager with our
GroupMemberLRU policy, on an in-memory CPU region and a temp directory. vllm/__init__.py is bypassed (no GPU, no build);
needs a Python env with torch (CPU) + vLLM's pure-Python deps. Run: python -m unittest -v test_gsch_fs_tier"""
import os, sys, tempfile, time, types, unittest
from types import SimpleNamespace as NS

SRC = os.environ.get("VLLM_SRC", "/tmp/vllm-v0.30.0")
if "vllm" not in sys.modules:
    _pkg = types.ModuleType("vllm"); _pkg.__path__ = [SRC + "/vllm"]; sys.modules["vllm"] = _pkg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext, make_offload_key  # noqa: E402
from vllm.v1.kv_offload.tiering.base import TransferJob  # noqa: E402
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager  # noqa: E402
import gsch_fs_tier as T  # noqa: E402

ROW = 16384  # bytes per chunk row in the tests (real: 6 MiB)


def spec():
    cfg = NS(groups=[NS(tokens_per_block=64, layer_names=["l0"])],
             parallel=NS(tp_size=4, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True),
             canonical_layout=False, kv_cache_layout=None, model=NS(name="test-model", dtype="bfloat16"),
             cache=NS(tokens_per_hash=64), replicated_layout=False)
    return NS(config=cfg, blocks_per_chunk=1, kv_events_config=NS(enable_kv_cache_events=False), extra_config={})


def keys(tag, n):
    return [make_offload_key(bytes([tag]) * 28 + i.to_bytes(4, "big"), 0) for i in range(n)]


def fill(arr, rows, seed):
    rng = np.random.default_rng(seed)
    for r in rows:
        arr[r] = rng.integers(0, 256, ROW, dtype=np.uint8)


def wait(cond, timeout=10.0):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


class TierFixture(unittest.TestCase):
    def mk(self, chunks=64, **kw):
        self.dir = tempfile.mkdtemp(prefix="gschfs-")
        self.arr = np.zeros((chunks, ROW), dtype=np.uint8)
        kw.setdefault("n_read_threads", 4); kw.setdefault("n_write_threads", 4); kw.setdefault("stats_every_s", 1e9)
        self.tier = T.GschFsTier(spec(), memoryview(self.arr), "GschFsTier", self.dir, **kw)
        self.addCleanup(self.tier.shutdown)
        self.jid = 0
        return self.tier

    def job(self, ks, rows, promotion):
        self.jid += 1
        return TransferJob(job_id=self.jid, keys=ks, chunk_ids=np.array(rows, dtype=np.int32), is_promotion=promotion,
                           req_context=ReqContext(req_id=f"r{self.jid}"))

    def finish(self, jid, timeout=10.0):
        got = {}
        ok = wait(lambda: got.update({r.job_id: r for r in self.tier.get_finished_jobs()}) or jid in got, timeout)
        self.assertTrue(ok, f"job {jid} never finished")
        return got[jid]


class TestTier(TierFixture):
    def test_round_trip_bytes_equal_and_lookup_is_index_only(self):
        t = self.mk(n_load_slices=8, n_store_slices=4)
        ks = keys(1, 10); fill(self.arr, range(10), 1); orig = self.arr[:10].copy()
        j = self.job(ks, list(range(10)), False); t.submit_store(j)
        self.assertTrue(self.finish(j.job_id).success)
        self.assertTrue(all(t.lookup(k, None) is LookupResult.HIT for k in ks))
        real_exists = os.path.exists
        os.path.exists = lambda p: (_ for _ in ()).throw(AssertionError("lookup touched the disk"))
        try:
            t.lookup(ks[0], None)
        finally:
            os.path.exists = real_exists
        j2 = self.job(ks, list(range(20, 30)), True); t.submit_load(j2)
        r = self.finish(j2.job_id)
        self.assertTrue(r.success and r.successful_keys is None)
        np.testing.assert_array_equal(self.arr[20:30], orig)

    def test_parallel_slices_equal_one_slice(self):
        for slices in (1, 3, 8):
            t = self.mk(n_load_slices=slices, n_store_slices=slices)
            ks = keys(2, 11); fill(self.arr, range(11), 2); orig = self.arr[:11].copy()
            j = self.job(ks, list(range(11)), False); t.submit_store(j); self.finish(j.job_id)
            j2 = self.job(ks, list(range(30, 41)), True); t.submit_load(j2); self.finish(j2.job_id)
            np.testing.assert_array_equal(self.arr[30:41], orig, f"{slices} slices")
            self.assertEqual(len(T._split(11, slices)), min(slices, 11))

    def test_partial_failure_keeps_the_good_keys_and_drops_the_bad(self):
        t = self.mk(n_load_slices=2, n_store_slices=1)
        ks = keys(3, 8); fill(self.arr, range(8), 3)
        j = self.job(ks, list(range(8)), False); t.submit_store(j); self.finish(j.job_id)
        os.remove(t.file_mapper.get_file_name(ks[5]))  # slice 2 = keys 4..7: 4 loads, 5 fails, 6,7 never read
        j2 = self.job(ks, list(range(40, 48)), True); t.submit_load(j2); r = self.finish(j2.job_id)
        self.assertFalse(r.success)
        self.assertEqual(set(r.successful_keys), set(ks[:5]))
        for k in ks[5:]:
            self.assertIs(t.lookup(k, None), LookupResult.MISS, "a failed key is never served again")

    def test_failed_write_is_not_indexed_and_leaves_no_orphans(self):
        t = self.mk(n_store_slices=2)
        ks = keys(9, 6); fill(self.arr, range(6), 9)
        real = T.batch_store_block
        def flaky(paths, view, offs, bs, direct):  # like real io: writes until the failing block, raises WITHOUT a count
            if t.file_mapper.get_file_name(ks[3]) in paths:  # slice 2 = keys 3..5: key 3 written, key 4 fails
                real(paths[:1], view, offs[:1], bs, direct)
                raise OSError(28, "No space left on device")
            return real(paths, view, offs, bs, direct)
        T.batch_store_block = flaky
        try:
            j = self.job(ks, list(range(6)), False); t.submit_store(j); r = self.finish(j.job_id)
        finally:
            T.batch_store_block = real
        self.assertFalse(r.success)
        self.assertEqual({k for k in ks if t.lookup(k, None) is LookupResult.HIT}, set(ks[:3]), "the failed slice is not served")
        self.assertEqual(t.stats["store_failed"], 3)
        self.assertTrue(wait(lambda: not os.path.exists(t.file_mapper.get_file_name(ks[3]))), "its written file is deleted")

    def test_backlog_drop_finishes_at_once_and_indexes_nothing(self):
        t = self.mk(max_store_backlog_bytes=2 * ROW)
        real = T.batch_store_block
        T.batch_store_block = lambda *a, **k: (time.sleep(0.5), real(*a, **k))[1]  # keep the first store in flight
        try:
            first = keys(40, 2); fill(self.arr, range(10, 12), 40)
            j1 = self.job(first, [10, 11], False); t.submit_store(j1)  # 2 chunks writing = the backlog bound
            ks = keys(4, 5)
            j = self.job(ks, list(range(5)), False); t.submit_store(j)
            r = t.get_finished_jobs()
            self.assertEqual([x.job_id for x in r], [j.job_id], "dropped store is reported on the very next poll")
            self.assertTrue(all(t.lookup(k, None) is LookupResult.MISS for k in ks))
            self.assertEqual(t.stats["store_dropped_backlog"], 5)
            self.assertFalse(any(os.path.exists(t.file_mapper.get_file_name(k)) for k in ks))
            self.finish(j1.job_id)
        finally:
            T.batch_store_block = real

    def test_cap_evicts_least_recent_and_deletes_files(self):
        t = self.mk(max_bytes=5 * ROW)
        a = keys(5, 5); fill(self.arr, range(5), 5)
        j = self.job(a, list(range(5)), False); t.submit_store(j); self.finish(j.job_id)
        t.touch([a[0]], None)  # a0 becomes the most recent
        b = keys(6, 3); fill(self.arr, range(5, 8), 6)
        j = self.job(b, list(range(5, 8)), False); t.submit_store(j); self.finish(j.job_id)
        present = {k for k in a + b if t.lookup(k, None) is LookupResult.HIT}
        self.assertEqual(present, {a[0], a[4]} | set(b), "LRU: a1..a3 evicted; the touched a0 kept")
        self.assertTrue(wait(lambda: not any(os.path.exists(t.file_mapper.get_file_name(k)) for k in a[1:4])))

    def test_deadline_cancels_only_slices_not_started(self):
        t = self.mk(n_load_slices=4, n_read_threads=1, n_write_threads=1, load_deadline_s=0.05)
        ks = keys(7, 8); fill(self.arr, range(8), 7)
        j = self.job(ks, list(range(8)), False); t.submit_store(j); self.finish(j.job_id)
        real = T.batch_load_block
        T.batch_load_block = lambda *a, **k: (time.sleep(0.3), real(*a, **k))[1]
        try:
            self.arr[50:58] = 7  # sentinel
            j2 = self.job(ks, list(range(50, 58)), True); t.submit_load(j2)
            time.sleep(0.1); t.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=[]))
            r = self.finish(j2.job_id)
        finally:
            T.batch_load_block = real
        self.assertFalse(r.success)
        good = set(r.successful_keys or ())
        self.assertTrue(set(ks[:2]) <= good, "the slices running at the deadline finish")
        self.assertTrue(set(ks[4:]).isdisjoint(good), "slices not started are cancelled")
        self.assertTrue((self.arr[54:58] == 7).all(), "a cancelled slice never touches its CPU slots")
        self.assertEqual(t.stats["deadline_cancels"], 1)

    def test_fresh_directory_and_old_runs_swept(self):
        base = tempfile.mkdtemp(prefix="gschfs-base-")
        old = os.path.join(base, T.RUN_PREFIX + "old"); os.makedirs(old); open(os.path.join(old, "x.bin"), "w").close()
        arr = np.zeros((8, ROW), dtype=np.uint8)
        t = T.GschFsTier(spec(), memoryview(arr), "GschFsTier", base, stats_every_s=1e9)
        self.addCleanup(t.shutdown)
        self.assertTrue(wait(lambda: not os.path.exists(old)), "the previous incarnation is removed")
        self.assertTrue(t._run_root.startswith(os.path.join(base, T.RUN_PREFIX)))

    def test_a_root_locked_by_another_process_is_refused(self):
        import subprocess
        base = tempfile.mkdtemp(prefix="gschfs-xproc-")
        holder = subprocess.Popen([sys.executable, "-c",
            "import fcntl,os,sys,time; fd=os.open(sys.argv[1], os.O_CREAT|os.O_RDWR, 0o644); fcntl.lockf(fd, fcntl.LOCK_EX);"
            " print('locked', flush=True); time.sleep(30)", os.path.join(base, ".gsch.lock")], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            with self.assertRaises(RuntimeError):
                T.GschFsTier(spec(), memoryview(np.zeros((4, ROW), dtype=np.uint8)), "GschFsTier", base, stats_every_s=1e9)
        finally:
            holder.kill(); holder.wait()
        t = T.GschFsTier(spec(), memoryview(np.zeros((4, ROW), dtype=np.uint8)), "GschFsTier", base, stats_every_s=1e9)
        t.shutdown()  # once the other owner is gone, the root is free

    def test_each_job_reported_once_and_submits_do_no_io(self):
        t = self.mk(n_store_slices=4)
        real = T.batch_store_block
        T.batch_store_block = lambda *a, **k: (time.sleep(0.2), real(*a, **k))[1]
        try:
            ks = keys(8, 40); fill(self.arr, range(40), 8)
            j = self.job(ks, list(range(40)), False)
            t0 = time.monotonic(); t.submit_store(j); dt = time.monotonic() - t0
            self.assertLess(dt, 0.05, "submit_store must not do the I/O on the calling thread")
            self.finish(j.job_id)
        finally:
            T.batch_store_block = real
        self.assertEqual(list(t.get_finished_jobs()), [], "nothing is reported twice")

    def test_row_size_assertion(self):
        with self.assertRaises(ValueError):
            self.mk(expected_row_bytes=ROW + 1)


class _Region:
    def __init__(self, arr):
        self.arr = arr

    def create_kv_memoryview(self):
        return memoryview(self.arr)

    def cleanup(self):
        pass


class TestEndToEnd(unittest.TestCase):
    """vLLM's own TieringOffloadingManager + CPU primary (GroupMemberLRU) + GschFsTier: a request's chain is written through,
    evicted from the CPU tier by others, then promoted back from disk with identical bytes."""

    def setUp(self):
        self.arr = np.zeros((8, ROW), dtype=np.uint8)
        self.primary = CPUPrimaryTierOffloadingManager(num_chunks=8, mmap_region=_Region(self.arr),
                                                       cache_policy="GroupMemberLRU",
                                                       cache_policy_module_path="gsch_member_policy")
        self.dir = tempfile.mkdtemp(prefix="gschfs-e2e-")
        self.tier = T.GschFsTier(spec(), self.primary.get_kv_memoryview(), "GschFsTier", self.dir,
                                 n_read_threads=2, n_write_threads=2, stats_every_s=1e9)
        self.mgr = TieringOffloadingManager(self.primary, [self.tier])
        self.addCleanup(self.mgr.shutdown)
        self.end = ScheduleEndContext(new_req_ids=[], preempted_req_ids=[])

    def store(self, rid, ks, seed):
        ctx = ReqContext(req_id=rid)
        self.mgr.on_new_request(ctx)
        out = self.mgr.prepare_store(ks, ctx)
        self.assertIsNotNone(out, f"{rid}: the CPU tier must accept the store")
        rows = [int(c) for c in out.store_spec.chunk_ids]
        fill(self.arr, rows, seed)
        data = {k: self.arr[r].copy() for k, r in zip(out.keys_to_store, rows)}
        self.mgr.complete_store(out.keys_to_store, ctx, True)
        self.mgr.on_request_finished(ctx)
        self.assertTrue(wait(lambda: (self.mgr.on_schedule_end(self.end) or True)
                             and all(self.tier.lookup(k, None) is LookupResult.HIT for k in ks)), f"{rid} written through")
        return data

    def test_chain_comes_back_from_disk_identical(self):
        a = keys(10, 6); data_a = self.store("A", a, 10)
        self.store("B", keys(11, 6), 11)  # pushes most of A out of the 8-chunk CPU tier
        evicted = [k for k in a if self.primary.lookup(k, ReqContext(req_id="probe")) is LookupResult.MISS]
        self.assertGreaterEqual(len(evicted), 4, "A must have left the CPU tier")
        ctx = ReqContext(req_id="A2"); self.mgr.on_new_request(ctx)
        res = [self.mgr.lookup(k, ctx) for k in a]
        self.assertTrue(all(r in (LookupResult.HIT, LookupResult.HIT_PENDING) for r in res), res)
        self.assertIn(LookupResult.HIT_PENDING, res, "the evicted part is promoted from disk")
        self.assertTrue(wait(lambda: (self.mgr.on_schedule_end(self.end) or True)
                             and all(self.mgr.lookup(k, ctx) is LookupResult.HIT for k in a)), "promotion completes")
        spec_ = self.mgr.prepare_load(a, ctx)
        for k, r in zip(a, [int(c) for c in spec_.chunk_ids]):
            np.testing.assert_array_equal(self.arr[r], data_a[k], "promoted bytes equal what was stored")
        self.mgr.complete_load(a, ctx)
        self.mgr.on_request_finished(ctx)
        self.assertGreaterEqual(self.tier.stats["load_ok"], len(evicted))


if __name__ == "__main__":
    unittest.main()
