# SPDX-License-Identifier: Apache-2.0
"""GschFsTier — a bounded, parallel filesystem tier for vLLM's tiered KV offloading (an out-of-tree SecondaryTierManager).

It is vLLM's own FileSystemTierManager (file layout, atomic temp+rename writes, the dual-queue thread pool) with four changes:

1. PARALLEL I/O — each load or store job is split into slices, one pool task each (the pool supports several tasks per job;
   the stock tier submits one, so a whole request is read by one thread).
2. AN IN-MEMORY INDEX — lookup answers HIT/MISS at once from what this tier itself wrote: no disk access on the scheduler
   thread and no RETRY step. The tier starts in a fresh directory per start, so the index always equals the directory.
3. A CAPACITY CAP — beyond max_bytes the least recently used chunks (recency from the tiering manager's touch fan-out) are
   dropped from the index at once and their files deleted by a background thread.
4. WRITE BACKPRESSURE — when the bytes being written exceed max_store_backlog_bytes, a store job is reported finished at once,
   unwritten: the CPU tier's slots are never held hostage by a slow disk (the stock tier has no bound and no timeout).

Every doubt is a MISS, so vLLM recomputes — today's behaviour. Job ids are only ever the ones vLLM gave us, each reported once
(an unknown id crashes the engine). Loads that pass load_deadline_s cancel their slices that have not started (a started copy
must finish: aborting it would corrupt the CPU slot).

CROSS-POD READS (peer_roots): engines share one filesystem, each WRITES and EVICTS only its own root_dir. Every peer_sync_s
each tier publishes, in its run directory, a snapshot of its index (gsch-index.bin) and the peer keys it touched over the last
3 intervals (gsch-touched.bin), atomically (temp + rename). A background thread follows each peer's newest run directory
(only if its run config equals ours), reads its index and its touches. Lookups stay in memory: own index, then the peers'
snapshots — no RETRY, no disk on the scheduler thread. A load reads each chunk from wherever it is. Touches of peer keys are
forwarded so the OWNER keeps a moved session's chain recent (otherwise its disk LRU ages it out ~cap / write rate later).
A peer file is never deleted or indexed here; a failed peer read marks that key bad until the peer's next snapshot. Staleness
is safe: a snapshot naming a just-evicted file → that read fails → recompute.

Config (kv_connector_extra_config of TieringOffloadingSpec):
  "secondary_tiers": [{"type": "GschFsTier", "module_path": "gsch_fs_tier", "root_dir": "/kvtier/<engine>",
                       "max_bytes": 900000000000, "n_load_slices": 8, "n_store_slices": 4,
                       "peer_roots": ["/kvtier/<other engine>"], "peer_sync_s": 5}]
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import queue
import struct
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import JobId, JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.io import batch_load_block, batch_store_block
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = logging.getLogger("vllm.gsch_fs_tier")
RUN_PREFIX = "gsch-run-"
_LIVE_ROOTS: set[str] = set()  # POSIX locks are per process: this catches a second tier on one root inside one process
INDEX_FILE = "gsch-index.bin"
TOUCH_FILE = "gsch-touched.bin"
_MAGIC = b"GSCHKEY1"
_HDR = struct.Struct("<8sHQI")  # magic | key length | sequence | number of keys; then the keys, concatenated


def _write_keys(path: str, keys, seq: int) -> None:
    keys = list(keys)
    klen = len(keys[0]) if keys else 0
    keys = [k for k in keys if len(k) == klen]
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(_HDR.pack(_MAGIC, klen, seq, len(keys)))
        f.write(b"".join(keys))
    os.replace(tmp, path)  # readers see the old file or the new one, never a torn one


def _read_keys(path: str, last_seq):
    """(seq, keys) — keys None when the sequence has not changed since last_seq."""
    with open(path, "rb") as f:
        data = f.read()
    magic, klen, seq, n = _HDR.unpack_from(data)
    if magic != _MAGIC or len(data) != _HDR.size + klen * n:
        raise ValueError(f"bad key file {path}")
    if seq == last_seq:
        return seq, None
    body = memoryview(data)[_HDR.size:]
    return seq, [bytes(body[i * klen:(i + 1) * klen]) for i in range(n)]


class _PeerView:
    """One immutable snapshot of a peer: swapped in whole by the peer thread, read by the scheduler thread."""

    __slots__ = ("run_dir", "mapper", "keys", "gen")

    def __init__(self, run_dir, mapper, keys, gen):
        self.run_dir, self.mapper, self.keys, self.gen = run_dir, mapper, keys, gen


class _Peer:
    __slots__ = ("root", "view", "idx_seq", "touch_seq", "bad", "bad_gen", "refused")

    def __init__(self, root):
        self.root = root
        self.view: _PeerView | None = None
        self.idx_seq = self.touch_seq = None  # peer thread only
        self.bad: set = set()  # scheduler thread only: keys whose read failed in this view generation
        self.bad_gen = -1
        self.refused: str | None = None  # a run dir skipped because its layout differs from ours


class _Job:
    """One load or store job as this tier split it. Slice outcomes are written by pool threads before the pool marks the
    task done and read on the scheduler thread only after the pool reports the job finished (the stock tier's ordering)."""

    __slots__ = ("kind", "keys", "bounds", "ok", "started", "cancel", "t0", "src")

    def __init__(self, kind: str, keys: list, bounds: list[tuple[int, int]], src: list | None = None):
        self.kind = kind
        self.src = src  # loads: per key, -1 = our own file, i = peer i (its view generation alongside)
        self.keys = keys
        self.bounds = bounds
        self.ok = [0] * len(bounds)  # per slice: number of leading keys that succeeded
        self.started = [False] * len(bounds)
        self.cancel = False
        self.t0 = time.monotonic()


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _load_summary(samples: list[tuple[int, float]]) -> str:
    """Load-job seconds by chain size (chunks of 64 tokens): n / p50 / p99 / max per bucket — the data for the deadline."""
    out = []
    for name, lo, hi in (("<=16K", 0, 256), ("16-64K", 256, 1024), ("64-131K", 1024, 2048), (">131K", 2048, 1 << 30)):
        xs = sorted(t for n, t in samples if lo < n <= hi)
        if xs:
            q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
            out.append(f"{name}:n={len(xs)} p50={q(0.5):.2f} p99={q(0.99):.2f} max={xs[-1]:.2f}")
    return " ".join(out) or "none"


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, n))
    step, extra = divmod(n, parts)
    out, lo = [], 0
    for i in range(parts):
        hi = lo + step + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


class GschFsTier(FileSystemTierManager):
    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        max_bytes: int = 1_000_000_000_000,
        n_load_slices: int = 8,
        n_store_slices: int = 4,
        n_read_threads: int = 16,
        n_write_threads: int = 8,
        max_store_backlog_bytes: int = 8_000_000_000,
        load_deadline_s: float = 10.0,
        stats_every_s: float = 60.0,
        fresh_dir_per_start: bool = True,
        expected_row_bytes: int | None = None,
        enable_kv_events: bool = False,
        locality: str | None = None,
        peer_roots: list[str] | str | None = None,
        peer_sync_s: float = 5.0,
        publish_index: bool = True,
    ):
        if enable_kv_events:
            raise ValueError("GschFsTier does not emit KV events; set enable_kv_events false")
        fresh_dir_per_start = _as_bool(fresh_dir_per_start)
        self._base_root = root_dir
        os.makedirs(root_dir, exist_ok=True)
        # ONE live tier per root_dir: the start-up sweep deletes every other run directory under it
        self._root_key = os.path.abspath(root_dir)
        if self._root_key in _LIVE_ROOTS:
            raise RuntimeError(f"GschFsTier: {root_dir} is owned by another live tier in this process (one root_dir per engine)")
        self._lock_fd = os.open(os.path.join(root_dir, ".gsch.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.lockf(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # cluster-wide on GPFS (POSIX byte-range lock)
        except OSError as exc:
            os.close(self._lock_fd)
            raise RuntimeError(f"GschFsTier: {root_dir} is owned by another live tier (one root_dir per engine)") from exc
        _LIVE_ROOTS.add(self._root_key)
        run_root = root_dir
        if fresh_dir_per_start:
            run_root = os.path.join(root_dir, f"{RUN_PREFIX}{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")
            os.makedirs(run_root, exist_ok=True)
        super().__init__(offloading_spec, primary_kv_view, tier_type, run_root,
                         n_read_threads=n_read_threads, n_write_threads=n_write_threads,
                         enable_kv_events=enable_kv_events, locality=locality)
        row = self._block_size  # bytes of one chunk row (all TP ranks)
        if expected_row_bytes is not None and row != expected_row_bytes:
            raise ValueError(f"GschFsTier: chunk row is {row} B, expected {expected_row_bytes} B")
        self._run_root = run_root
        self._spec = offloading_spec
        if isinstance(peer_roots, str):
            peer_roots = [r for r in peer_roots.split(",") if r.strip()]
        self._peers = [_Peer(r.strip()) for r in (peer_roots or [])
                       if os.path.abspath(r.strip()) != self._root_key]
        self._publish = _as_bool(publish_index) and fresh_dir_per_start
        self._sync_s = max(0.05, float(peer_sync_s))
        self._next_sync = 0.0
        self._outbox: queue.SimpleQueue = queue.SimpleQueue()  # scheduler -> peer thread: (index keys, touched peer keys)
        self._touched_out: set = set()  # scheduler thread: peer keys touched since the last hand-off
        self._touch_in: queue.SimpleQueue = queue.SimpleQueue()  # peer thread -> scheduler: keys peers touched
        self._cap_chunks = max(1, int(max_bytes) // row)
        self._backlog_chunks = max(1, int(max_store_backlog_bytes) // row)
        self._n_load_slices = max(1, int(n_load_slices))
        self._n_store_slices = max(1, int(n_store_slices))
        self._deadline = float(load_deadline_s)
        self._stats_every = float(stats_every_s)
        self._index: OrderedDict[OffloadKey, None] = OrderedDict()  # on disk and readable; LRU order (last = most recent)
        self._writing: set[OffloadKey] = set()
        self._loading: dict[OffloadKey, int] = {}  # key -> number of load jobs reading it
        self._jobs: dict[JobId, _Job] = {}
        self._immediate: list[JobResult] = []
        self._deletes: queue.Queue = queue.Queue()
        self._pending_del: set[OffloadKey] = set()  # queued for unlink: never re-stored until the unlink is done
        self._low_space = False  # set by the deleter thread from statvfs: below the free-space floor, writes are dropped
        self._space_check_every = 10.0
        self._stop = threading.Event()
        self._deleter = threading.Thread(target=self._delete_loop, name="gsch_fs_delete", daemon=True)
        self._deleter.start()
        self.stats = {k: 0 for k in ("store_jobs", "store_written", "store_dropped_backlog", "store_dropped_space",
                                       "store_dropped_pending_delete", "store_already", "store_failed",
                                       "load_jobs", "load_ok", "load_failed", "deadline_cancels", "evicted",
                                       "deleted", "delete_errors", "old_runs_removed", "old_files_removed",
                                       "peer_load_ok", "peer_load_failed", "touch_forwarded", "touch_applied",
                                       "published", "publish_errors", "peer_reads", "peer_read_errors", "peer_switches",
                                       "peer_layout_refused")}
        self._load_secs: list[tuple[int, float]] = []  # (chunks, seconds) per load job since the last stats line
        self._last_log = time.monotonic()
        if fresh_dir_per_start:
            threading.Thread(target=self._sweep_old_runs, name="gsch_fs_sweep", daemon=True).start()
        if self._peers or self._publish:
            threading.Thread(target=self._peer_loop, name="gsch_fs_peers", daemon=True).start()
        logger.info("GschFsTier: root %s, row %d B, cap %d chunks (%.1f GB), backlog %d chunks, slices load %d / store %d, "
                    "peers %s, publish %s every %.1f s", run_root, row, self._cap_chunks, self._cap_chunks * row / 1e9,
                    self._backlog_chunks, self._n_load_slices, self._n_store_slices,
                    [p.root for p in self._peers], self._publish, self._sync_s)

    # ---- background housekeeping (never on the scheduler thread) ---------------------------------------------------
    def _delete_loop(self) -> None:
        next_space = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_space:
                next_space = now + self._space_check_every
                try:
                    st = os.statvfs(self._run_root)
                    self._low_space = st.f_bavail < 0.05 * st.f_blocks
                except OSError:
                    pass
            try:
                key, path = self._deletes.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                os.unlink(path)
                self.stats["deleted"] += 1
            except FileNotFoundError:
                pass
            except OSError:
                self.stats["delete_errors"] += 1
            self._pending_del.discard(key)  # only after the unlink: a re-store can never meet the old file

    def _delete(self, key: OffloadKey) -> None:
        self._pending_del.add(key)
        self._deletes.put((key, self.file_mapper.get_file_name(key)))

    def _sweep_old_runs(self, max_unlinks_per_s: float = 500.0) -> None:
        """Delete earlier incarnations under root_dir at a bounded rate (we hold root_dir's lock, so they are dead)."""
        pause = 1.0 / max_unlinks_per_s
        try:
            names = os.listdir(self._base_root)
        except OSError:
            return
        for name in names:
            p = os.path.join(self._base_root, name)
            if not name.startswith(RUN_PREFIX) or os.path.abspath(p) == os.path.abspath(self._run_root):
                continue
            for dirpath, _dirs, files in os.walk(p, topdown=False):
                for f in files:
                    if self._stop.is_set():
                        return
                    try:
                        os.unlink(os.path.join(dirpath, f))
                        self.stats["old_files_removed"] += 1
                    except OSError:
                        pass
                    time.sleep(pause)
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
            self.stats["old_runs_removed"] += 1
            logger.info("GschFsTier: removed old run %s", p)

    # ---- cross-pod: publish our index + forwarded touches, follow the peers (peer thread) ----------------------------
    def _peer_loop(self) -> None:
        seq, window = 0, deque(maxlen=3)  # touches stay published for 3 intervals: a reader polling twice per interval sees each
        while not self._stop.wait(self._sync_s / 2):
            index_keys, touched = self._drain_outbox()
            if index_keys is not None and self._publish:
                seq += 1
                window.append(touched)
                try:
                    _write_keys(os.path.join(self._run_root, INDEX_FILE), index_keys, seq)
                    _write_keys(os.path.join(self._run_root, TOUCH_FILE), set().union(*window), seq)
                    self.stats["published"] += 1
                except OSError:
                    self.stats["publish_errors"] += 1
            for p in self._peers:
                self._refresh_peer(p)

    def _drain_outbox(self):
        """The newest index snapshot and the union of every touched set handed off since the last drain (none lost)."""
        index_keys, touched = None, set()
        while True:
            try:
                index_keys, t = self._outbox.get_nowait()
            except queue.Empty:
                return index_keys, touched
            touched |= t

    def _refresh_peer(self, p: _Peer) -> None:
        try:
            runs = [n for n in os.listdir(p.root) if n.startswith(RUN_PREFIX)]
        except OSError:
            return
        if not runs:
            return
        run = os.path.join(p.root, max(runs))  # names start with the start time: the newest incarnation
        v = p.view
        mapper = v.mapper if v is not None and v.run_dir == run else None
        if mapper is None:
            if p.refused == run:
                return
            mapper = FileMapper.from_offloading_spec(root_dir=run, offloading_spec=self._spec,
                                                     blocks_per_file=self._spec.blocks_per_chunk, parallel_agnostic=True)
            try:
                with open(mapper.get_config_file_path()) as f:
                    theirs = json.load(f)
            except (OSError, ValueError):
                return  # not written yet
            if theirs != json.loads(json.dumps(self.file_mapper.get_run_config())):
                p.refused = run
                self.stats["peer_layout_refused"] += 1
                logger.warning("GschFsTier: peer run %s has a different layout; not reading it", run)
                return
            p.idx_seq = p.touch_seq = None
        try:
            p.idx_seq, ks = _read_keys(os.path.join(run, INDEX_FILE), p.idx_seq)
            if ks is not None or mapper is not (v.mapper if v is not None else None):
                if v is None or v.run_dir != run:
                    self.stats["peer_switches"] += 1
                p.view = _PeerView(run, mapper, frozenset(ks) if ks is not None else frozenset(),
                                   (v.gen + 1) if v is not None else 0)
                self.stats["peer_reads"] += 1
            p.touch_seq, touched = _read_keys(os.path.join(run, TOUCH_FILE), p.touch_seq)
            if touched:
                self._touch_in.put(touched)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, struct.error):
            self.stats["peer_read_errors"] += 1

    def _peer_of(self, key: OffloadKey):
        for i, p in enumerate(self._peers):
            v = p.view
            if v is not None and key in v.keys:
                if p.bad_gen == v.gen and key in p.bad:
                    continue
                return i, v
        return -1, None

    # ---- lookups and recency ----------------------------------------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._index:
            return LookupResult.HIT
        if self._peers and self._peer_of(key)[0] >= 0:
            return LookupResult.HIT
        return LookupResult.MISS

    def touch(self, keys, req_context: ReqContext):
        idx = self._index
        for k in keys:
            if k in idx:
                idx.move_to_end(k)
            elif self._peers and len(self._touched_out) < 4_000_000 and self._peer_of(k)[0] >= 0:
                self._touched_out.add(k)

    # ---- stores (write-through from the CPU tier) ---------------------------------------------------------------------
    def submit_store(self, job_metadata: TransferJob) -> None:
        self.stats["store_jobs"] += 1
        keys = list(job_metadata.keys)
        cids = [int(c) for c in job_metadata.chunk_ids]
        new, pend = [], 0
        for k, c in zip(keys, cids):
            if k in self._index or k in self._writing:
                continue
            if k in self._pending_del:
                pend += 1
                continue
            new.append((k, c))
        self.stats["store_already"] += len(keys) - len(new) - pend
        self.stats["store_dropped_pending_delete"] += pend
        if not new:
            self._immediate.append(JobResult(job_id=job_metadata.job_id, success=True))
            return
        if self._low_space:
            self.stats["store_dropped_space"] += len(new)
            self._immediate.append(JobResult(job_id=job_metadata.job_id, success=True))
            return
        if self._writing and len(self._writing) + len(new) > self._backlog_chunks:
            self.stats["store_dropped_backlog"] += len(new)
            self._immediate.append(JobResult(job_id=job_metadata.job_id, success=True))
            return
        self._make_room(len(new))
        nkeys = [k for k, _ in new]
        self._writing.update(nkeys)
        bounds = _split(len(new), self._n_store_slices)
        job = _Job("store", nkeys, bounds)
        self._jobs[job_metadata.job_id] = job
        paths = [self.file_mapper.get_file_name(k) for k in nkeys]
        offs = [c * self._block_size for _, c in new]
        tasks = [self._task(job, i, "store", paths[lo:hi], offs[lo:hi]) for i, (lo, hi) in enumerate(bounds)]
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    def _make_room(self, n_new: int) -> None:
        over = len(self._index) + len(self._writing) + n_new - self._cap_chunks
        if over <= 0:
            return
        idx, skipped = self._index, []
        while over > 0 and idx:
            k, _ = idx.popitem(last=False)
            if k in self._loading:  # a promotion is reading it: keep it, as most recent
                skipped.append(k)
                continue
            self._delete(k)
            self.stats["evicted"] += 1
            over -= 1
        for k in skipped:
            idx[k] = None

    # ---- loads (promotion into CPU slots vLLM reserved) ---------------------------------------------------------------
    def submit_load(self, job_metadata: TransferJob) -> None:
        self.stats["load_jobs"] += 1
        keys = list(job_metadata.keys)
        cids = [int(c) for c in job_metadata.chunk_ids]
        for k in keys:
            self._loading[k] = self._loading.get(k, 0) + 1
        bounds = _split(len(keys), self._n_load_slices)
        paths, src = [], None
        for j, k in enumerate(keys):
            if k in self._index or not self._peers:
                paths.append(self.file_mapper.get_file_name(k))
                continue
            i, v = self._peer_of(k)
            if i < 0:
                paths.append(self.file_mapper.get_file_name(k))  # neither: the read fails cleanly -> recompute
                continue
            if src is None:
                src = [None] * len(keys)
            src[j] = (i, v.gen)
            paths.append(v.mapper.get_file_name(k))
        job = _Job("load", keys, bounds, src)
        self._jobs[job_metadata.job_id] = job
        offs = [c * self._block_size for c in cids]
        tasks = [self._task(job, i, "load", paths[lo:hi], offs[lo:hi]) for i, (lo, hi) in enumerate(bounds)]
        self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)

    def _task(self, job: _Job, i: int, kind: str, paths: list[str], offs: list[int]):
        view, bs, direct = self._primary_kv_view, self._block_size, self._use_o_direct

        def run() -> None:
            if job.cancel:  # not started before the deadline: never touches the CPU slots
                job.ok[i] = 0
                return
            job.started[i] = True
            try:
                if kind == "store":
                    batch_store_block(paths, view, offs, bs, direct)
                else:
                    batch_load_block(paths, view, offs, bs, direct)
                job.ok[i] = len(paths)
            except OSError as exc:
                # loads report how many leading blocks were read; stores never do (io.py / fs_io.cpp), so a failed
                # store slice counts as wholly failed and its files are deleted on completion
                job.ok[i] = int(getattr(exc, "num_succeeded", 0)) if kind == "load" else 0
            except Exception:  # never let a slice take the pool thread down
                job.ok[i] = 0
        return run

    # ---- completions ---------------------------------------------------------------------------------------------
    def get_finished_jobs(self) -> Iterable[JobResult]:
        out, self._immediate = self._immediate, []
        for job_id, _pool_ok, transfer_time in self._pool.get_finished():
            job = self._jobs.pop(job_id, None)
            if job is None:  # not ours to report (cannot happen: every pool job is one vLLM gave us)
                continue
            good, bad, peer_good, peer_bad = [], [], 0, []
            for (lo, hi), n in zip(job.bounds, job.ok):
                good.extend(job.keys[lo:lo + n])
                bad.extend(job.keys[lo + n:hi])
                if job.src is not None:
                    peer_good += sum(1 for j in range(lo, lo + n) if job.src[j] is not None)
                    peer_bad.extend(j for j in range(lo + n, hi) if job.src[j] is not None)
            if job.kind == "store":
                self._writing.difference_update(job.keys)
                for k in good:
                    self._index[k] = None
                for k in bad:
                    self._delete(k)  # partly written slices leave no orphans
                self.stats["store_written"] += len(good)
                self.stats["store_failed"] += len(bad)
                out.append(JobResult(job_id=job_id, success=not bad, transfer_time=transfer_time))
            else:
                for k in job.keys:
                    c = self._loading.get(k, 0) - 1
                    if c > 0:
                        self._loading[k] = c
                    else:
                        self._loading.pop(k, None)
                pb = {job.keys[j] for j in peer_bad}
                for j in peer_bad:  # a peer's file: never ours to delete; skip it until that peer's next snapshot
                    i, gen = job.src[j]
                    p = self._peers[i]
                    if p.bad_gen != gen:
                        p.bad, p.bad_gen = set(), gen
                    p.bad.add(job.keys[j])
                for k in bad:
                    if k in pb:
                        continue
                    self._index.pop(k, None)  # failed or cancelled: never served again (no re-promote loop)
                    self._delete(k)  # and no orphan file outside the cap
                self.stats["load_ok"] += len(good)
                self.stats["load_failed"] += len(bad)
                self.stats["peer_load_ok"] += peer_good
                self.stats["peer_load_failed"] += len(peer_bad)
                if len(self._load_secs) < 20000:
                    self._load_secs.append((len(job.keys), time.monotonic() - job.t0))
                out.append(JobResult(job_id=job_id, success=not bad,
                                     successful_keys=(tuple(good) if bad and good else None),
                                     transfer_time=transfer_time))
        return out

    def has_pending_work(self) -> bool:
        return bool(self._immediate) or bool(self._jobs)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        now = time.monotonic()
        if (self._publish or self._peers) and now >= self._next_sync:
            self._next_sync = now + self._sync_s
            self._outbox.put((list(self._index), self._touched_out))
            self.stats["touch_forwarded"] += len(self._touched_out)
            self._touched_out = set()
        idx = self._index
        while True:
            try:
                ks = self._touch_in.get_nowait()
            except queue.Empty:
                break
            for k in ks:  # a peer used our chunks: keep them recent here, their owner
                if k in idx:
                    idx.move_to_end(k)
                    self.stats["touch_applied"] += 1
        for job in self._jobs.values():
            if job.kind == "load" and not job.cancel and now - job.t0 > self._deadline:
                job.cancel = True
                self.stats["deadline_cancels"] += 1
        if now - self._last_log >= self._stats_every:
            self._last_log = now
            logger.info("gsch_fs_tier: %s index=%d (%.1f GB) writing=%d pending_del=%d jobs=%d low_space=%s peers=%s load_s %s",
                        self.stats, len(self._index), len(self._index) * self._block_size / 1e9, len(self._writing),
                        len(self._pending_del), len(self._jobs), self._low_space,
                        [(os.path.basename(p.view.run_dir), len(p.view.keys)) if p.view else None for p in self._peers],
                        _load_summary(self._load_secs))
            self._load_secs = []

    def on_request_finished(self, req_context: ReqContext) -> None:
        return  # no per-request state (the stock async lookup manager is not used)

    def shutdown(self) -> None:
        self._stop.set()
        super().shutdown()
        _LIVE_ROOTS.discard(self._root_key)
        try:
            os.close(self._lock_fd)  # releases the root_dir lock
        except OSError:
            pass
