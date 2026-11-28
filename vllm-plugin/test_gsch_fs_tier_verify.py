# SPDX-License-Identifier: Apache-2.0
"""Verifier's tests for gsch_fs_tier (2026-11-29), same harness as test_gsch_fs_tier (vLLM's REAL tiering sources, stub
vllm package). Tests marked expectedFailure expose a bug or a gap; the reason and the proposed fix are in
independent-analysis/GPFS-TIER-VERIFY-2026-11-29.md. Run: python -m unittest -v test_gsch_fs_tier_verify"""
import glob, importlib.util, os, shutil, subprocess, sys, sysconfig, tempfile, threading, time, unittest

import test_gsch_fs_tier as B  # sets up the stub vllm package and sys.path
from test_gsch_fs_tier import ROW, TierFixture, _Region, fill, keys, spec, wait  # noqa: F401
import numpy as np
from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
import vllm.v1.kv_offload.tiering.fs.io as fsio
import gsch_fs_tier as T

END = ScheduleEndContext(new_req_ids=[], preempted_req_ids=[])


def on_disk_files(tier):
    return {p for p in glob.glob(os.path.join(tier._run_root, "**", "*.bin"), recursive=True)}


class TestOrphansAndRaces(TierFixture):
    # fixed — was xfail # BUG-1: cancelled (and failed) load keys are dropped from the index but their files stay on disk
    def test_deadline_cancelled_keys_leave_no_orphan_files(self):
        t = self.mk(n_load_slices=4, n_read_threads=1, n_write_threads=1, load_deadline_s=0.05)
        ks = keys(21, 8); fill(self.arr, range(8), 21)
        j = self.job(ks, list(range(8)), False); t.submit_store(j); self.finish(j.job_id)
        real = T.batch_load_block
        T.batch_load_block = lambda *a, **k: (time.sleep(0.3), real(*a, **k))[1]
        try:
            j2 = self.job(ks, list(range(50, 58)), True); t.submit_load(j2)
            time.sleep(0.1); t.on_schedule_end(END); self.finish(j2.job_id)
        finally:
            T.batch_load_block = real
        time.sleep(1.0)  # let the deleter run
        orphans = [k for k in ks if t.lookup(k, None) is LookupResult.MISS and os.path.exists(t.file_mapper.get_file_name(k))]
        self.assertEqual(orphans, [], "cancelled keys: files on disk, not indexed, not counted against max_bytes")

    # fixed — was xfail # BUG-2: io.batch_store_block never sets num_succeeded on a STORE error -> written files orphaned
    def test_partial_store_failure_with_real_io_leaves_no_orphans(self):
        t = self.mk(n_store_slices=1)
        ks = keys(22, 4); fill(self.arr, range(4), 22)
        bad = t.file_mapper.get_file_name(ks[2])
        real_replace = os.replace
        def flaky_replace(src, dst, *a, **k):
            if dst == bad:
                raise OSError(28, "No space left on device")
            return real_replace(src, dst, *a, **k)
        os.replace = flaky_replace
        try:
            j = self.job(ks, list(range(4)), False); t.submit_store(j); r = self.finish(j.job_id)
        finally:
            os.replace = real_replace
        self.assertFalse(r.success)
        indexed = {t.file_mapper.get_file_name(k) for k in ks if t.lookup(k, None) is LookupResult.HIT}
        self.assertEqual(on_disk_files(t) - indexed, set(), "files written before the failing block are neither indexed nor deleted")

    # fixed — was xfail # BUG-3: deleter check-then-unlink races a re-store whose io exists() short-circuit skips the write
    def test_restore_during_pending_unlink_never_serves_a_missing_file(self):
        t = self.mk(max_bytes=2 * ROW, n_store_slices=1)
        ks = keys(23, 3); fill(self.arr, range(3), 23)
        target = t.file_mapper.get_file_name(ks[0])
        entered, release = threading.Event(), threading.Event()
        real_unlink = os.unlink
        def slow_unlink(p, *a, **k):
            if p == target and threading.current_thread().name == "gsch_fs_delete":
                entered.set(); release.wait(5)
            return real_unlink(p, *a, **k)
        os.unlink = slow_unlink
        try:
            j = self.job(ks[:2], [0, 1], False); t.submit_store(j); self.finish(j.job_id)
            j = self.job(ks[2:], [2], False); t.submit_store(j); self.finish(j.job_id)  # evicts ks[0] -> delete queued
            self.assertTrue(entered.wait(5), "deleter reached the unlink of ks[0]")
            j = self.job(ks[:1], [0], False); t.submit_store(j); self.finish(j.job_id)  # re-store: exists() -> skipped
            release.set(); time.sleep(0.3)
        finally:
            release.set(); os.unlink = real_unlink
        if t.lookup(ks[0], None) is LookupResult.HIT:
            self.assertTrue(os.path.exists(target), "indexed key whose file the deleter removed")

    # fixed (RISK-1): a second tier on a live root_dir refuses to start instead of sweeping the live directory
    def test_second_tier_on_same_root_does_not_sweep_a_live_one(self):
        base = tempfile.mkdtemp(prefix="gschfs-shared-")
        a1, a2 = np.zeros((4, ROW), dtype=np.uint8), np.zeros((4, ROW), dtype=np.uint8)
        t1 = T.GschFsTier(spec(), memoryview(a1), "GschFsTier", base, stats_every_s=1e9); self.addCleanup(t1.shutdown)
        time.sleep(1.1)  # a distinct start timestamp
        with self.assertRaises(RuntimeError):
            T.GschFsTier(spec(), memoryview(a2), "GschFsTier", base, stats_every_s=1e9)
        time.sleep(0.5)
        self.assertTrue(os.path.isdir(t1._run_root), "engine 1's live directory was swept by engine 2")
        t1.shutdown()
        t3 = T.GschFsTier(spec(), memoryview(a2), "GschFsTier", base, stats_every_s=1e9); self.addCleanup(t3.shutdown)
        self.assertTrue(wait(lambda: not os.path.exists(t1._run_root)), "after its owner is gone, the old run is swept")

    def test_job_larger_than_backlog_is_written_when_idle(self):
        # fixed (NIT-1): the backlog bound applies only while something is in flight; an idle tier writes a big job
        t = self.mk(max_store_backlog_bytes=2 * ROW)
        ks = keys(24, 3); fill(self.arr, range(3), 24)
        j = self.job(ks, [0, 1, 2], False); t.submit_store(j)
        self.assertTrue(self.finish(j.job_id).success)
        self.assertEqual(t.stats["store_dropped_backlog"], 0)
        self.assertTrue(all(t.lookup(k, None) is LookupResult.HIT for k in ks))


class TestConfigAndContracts(unittest.TestCase):
    def test_factory_builds_the_tier_from_the_design_config(self):
        d = tempfile.mkdtemp(prefix="gschfs-cfg-")
        arr = np.zeros((4, ROW), dtype=np.uint8)
        cfg = {"type": "GschFsTier", "module_path": "gsch_fs_tier", "root_dir": d, "max_bytes": 1000000000000,
               "n_load_slices": 8, "n_store_slices": 4, "stats_every_s": 1e9}
        self.assertIs(SecondaryTierFactory.get_tier_class(cfg), T.GschFsTier)
        self.assertEqual(T.GschFsTier.build_metric_definitions(cfg), {})
        t = SecondaryTierFactory.create_secondary_tier(cfg, memoryview(arr), spec()); self.addCleanup(t.shutdown)
        self.assertIsInstance(t, T.GschFsTier)
        with self.assertRaises(TypeError):  # a typo in a key is a start-up crash, not a silent default
            SecondaryTierFactory.create_secondary_tier(dict(cfg, n_load_slice=8), memoryview(arr), spec())

    def test_reset_cache_with_inflight_and_dropped_stores_does_not_assert(self):
        arr = np.zeros((8, ROW), dtype=np.uint8)
        prim = CPUPrimaryTierOffloadingManager(num_chunks=8, mmap_region=_Region(arr), cache_policy="lru")
        tier = T.GschFsTier(spec(), prim.get_kv_memoryview(), "GschFsTier", tempfile.mkdtemp(prefix="gschfs-rst-"),
                            n_read_threads=1, n_write_threads=1, max_store_backlog_bytes=3 * ROW, stats_every_s=1e9)
        mgr = TieringOffloadingManager(prim, [tier]); self.addCleanup(mgr.shutdown)
        real = T.batch_store_block
        T.batch_store_block = lambda *a, **k: (time.sleep(0.3), real(*a, **k))[1]
        try:
            for rid, tag in (("A", 25), ("B", 26)):  # A in flight (slow), B dropped by backpressure (immediate)
                ctx = ReqContext(req_id=rid); mgr.on_new_request(ctx)
                out = mgr.prepare_store(keys(tag, 3), ctx); mgr.complete_store(out.keys_to_store, ctx, True)
                mgr.on_request_finished(ctx)
            self.assertTrue(tier._immediate and tier._jobs)
            mgr.reset_cache()  # drain_jobs -> process -> assert not self._jobs
        finally:
            T.batch_store_block = real
        self.assertFalse(mgr.has_pending_work())


class TestPromotionSelfEviction(unittest.TestCase):
    """RISK-2 (vLLM behaviour, not GschFsTier's): a lookup scan does not pin the chunks it has already counted as HIT,
    so the promotions it starts further down the chain can evict the request's OWN CPU prefix (LRU front). The next
    scan re-promotes that prefix from disk: extra disk reads and one extra promotion round."""

    def _returner(self, policy, module):
        """A returner whose chain is partly in the CPU tier (a0..a3) and partly only on disk (a4, a5): how many chunks does
        its lookup read from disk? 2 if the scan's HITs are protected from its own promotions, more if they self-evict."""
        arr = np.zeros((8, ROW), dtype=np.uint8)
        prim = CPUPrimaryTierOffloadingManager(num_chunks=8, mmap_region=_Region(arr), cache_policy=policy,
                                               cache_policy_module_path=module)
        tier = T.GschFsTier(spec(), prim.get_kv_memoryview(), "GschFsTier", tempfile.mkdtemp(prefix="gschfs-self-"),
                            n_read_threads=2, n_write_threads=2, stats_every_s=1e9)
        mgr = TieringOffloadingManager(prim, [tier]); self.addCleanup(mgr.shutdown)
        def store(rid, ks):
            ctx = ReqContext(req_id=rid); mgr.on_new_request(ctx)
            out = mgr.prepare_store(ks, ctx); self.assertIsNotNone(out)
            mgr.complete_store(out.keys_to_store, ctx, True); mgr.on_request_finished(ctx)
            self.assertTrue(wait(lambda: (mgr.on_schedule_end(END) or True) and not mgr.has_pending_work()))
        a, b = keys(27, 6), keys(28, 4)
        store("A", a); mgr.touch(a, ReqContext(req_id="A")); store("B", b)  # B evicts a5, a4 (LRU front after the touch)
        ctx = ReqContext(req_id="A2"); mgr.on_new_request(ctx)
        for _ in range(10):
            res = []
            for k in a:  # the connector's _maximal_prefix_lookup: scan to the first MISS
                r = mgr.lookup(k, ctx); res.append(r)
                if r is LookupResult.MISS: break
            mgr.touch(a, ctx); mgr.on_schedule_end(END)
            if all(r is LookupResult.HIT for r in res) and len(res) == len(a): break
            wait(lambda: (mgr.on_schedule_end(END) or True) and not tier._jobs)
        return tier.stats["load_ok"]

    @unittest.expectedFailure  # vLLM's own LRU: the scan does not pin its HITs (documented; GroupMemberLRU fixes it)
    def test_returner_reads_only_its_missing_tail_from_disk(self):
        self.assertEqual(self._returner("lru", None), 2, "only a4, a5 had left the CPU tier")

    def test_group_member_lru_protects_the_scan(self):
        self.assertEqual(self._returner("GroupMemberLRU", "gsch_member_policy"), 2, "only a4, a5 are read from disk")


def _build_fsio_c():
    cxx = shutil.which("clang++") or shutil.which("g++")
    src = os.path.join(B.SRC, "csrc", "fs_io.cpp")
    if not cxx or not os.path.exists(src):
        return None
    out = os.path.join(tempfile.mkdtemp(prefix="fsioc-"), "fs_io_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    r = subprocess.run([cxx, "-std=c++17", "-O2", "-shared", "-fPIC", *flags, "-I" + sysconfig.get_paths()["include"],
                        src, "-o", out], capture_output=True)
    if r.returncode != 0:
        return None
    s = importlib.util.spec_from_file_location("fs_io_C", out); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


class TestCFastPath(TierFixture):
    """The real image ships vllm/fs_io_C (csrc/fs_io.cpp): io.py then calls the C batch functions (GIL released for
    the whole batch). Compile the SAME source here and route io.py through it."""

    @classmethod
    def setUpClass(cls):
        cls.C = _build_fsio_c()
        if cls.C is None:
            raise unittest.SkipTest("no C++ compiler / csrc")

    def setUp(self):
        self._saved = (fsio._HAS_FSIO_C, getattr(fsio, "batch_load_block_C", None), getattr(fsio, "batch_store_block_C", None))
        fsio._HAS_FSIO_C, fsio.batch_load_block_C, fsio.batch_store_block_C = True, self.C.batch_load_block, self.C.batch_store_block

    def tearDown(self):
        fsio._HAS_FSIO_C, fsio.batch_load_block_C, fsio.batch_store_block_C = self._saved

    def test_c_round_trip_sliced(self):
        t = self.mk(n_load_slices=3, n_store_slices=2)
        ks = keys(31, 9); fill(self.arr, range(9), 31); orig = self.arr[:9].copy()
        j = self.job(ks, list(range(9)), False); t.submit_store(j); self.assertTrue(self.finish(j.job_id).success)
        j2 = self.job(ks, list(range(40, 49)), True); t.submit_load(j2); self.assertTrue(self.finish(j2.job_id).success)
        np.testing.assert_array_equal(self.arr[40:49], orig)

    def test_c_partial_load_failure_keeps_prefix_of_each_slice(self):
        t = self.mk(n_load_slices=2, n_store_slices=1)
        ks = keys(32, 8); fill(self.arr, range(8), 32)
        j = self.job(ks, list(range(8)), False); t.submit_store(j); self.finish(j.job_id)
        os.remove(t.file_mapper.get_file_name(ks[5]))
        j2 = self.job(ks, list(range(40, 48)), True); t.submit_load(j2); r = self.finish(j2.job_id)
        self.assertFalse(r.success); self.assertEqual(set(r.successful_keys), set(ks[:5]))

    def test_c_short_file_is_removed_and_never_served(self):
        t = self.mk(n_load_slices=1, n_store_slices=1)
        ks = keys(33, 2); fill(self.arr, range(2), 33)
        j = self.job(ks, [0, 1], False); t.submit_store(j); self.finish(j.job_id)
        p = t.file_mapper.get_file_name(ks[1])
        with open(p, "r+b") as f:
            f.truncate(ROW // 2)
        j2 = self.job(ks, [4, 5], True); t.submit_load(j2); r = self.finish(j2.job_id)
        self.assertEqual(set(r.successful_keys), {ks[0]})
        self.assertFalse(os.path.exists(p), "the C loader unlinks a provably short file")
        self.assertIs(t.lookup(ks[1], None), LookupResult.MISS)

    def test_c_store_error_has_no_num_succeeded(self):
        # documents BUG-2's root: the C store raises a plain OSError (no num_succeeded), so a failed slice reports 0
        # written keys although the blocks before the failing one ARE on disk (orphans). (Note: the C store treats ANY
        # existing dest path -- even a directory -- as "already present" and reports success.)
        t = self.mk(n_store_slices=1)
        ks = [keys(tag, 1)[0] for tag in (34, 35, 36)]; fill(self.arr, range(3), 34)  # 3 keys in 3 directories
        p0, p1 = (t.file_mapper.get_file_name(k) for k in ks[:2])
        self.assertNotEqual(os.path.dirname(p0), os.path.dirname(p1))
        os.makedirs(os.path.dirname(p1)); os.chmod(os.path.dirname(p1), 0o555)  # EACCES on the temp file of ks[1]
        self.addCleanup(os.chmod, os.path.dirname(p1), 0o755)
        j = self.job(ks, [0, 1, 2], False); t.submit_store(j); r = self.finish(j.job_id)
        self.assertFalse(r.success)
        self.assertEqual(t.stats["store_failed"], 3, "all keys of the slice counted failed")
        self.assertTrue(os.path.exists(p0), "although ks[0] was written (an orphan: on disk, not indexed, not capped)")
        self.assertIs(t.lookup(ks[0], None), LookupResult.MISS)


if __name__ == "__main__":
    unittest.main()
