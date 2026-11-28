# SPDX-License-Identifier: Apache-2.0
"""Tests for gsch_member_policy against vLLM's REAL CachePolicy / LRUCachePolicy source (VLLM_SRC, default
/tmp/vllm-v0.30.0). Only vllm.v1.kv_offload.base (which imports torch) is replaced by the two types the
policies use: OffloadKey and ReqContext, copied from that file. Run: python3 -m unittest -v test_gsch_member_policy"""
import importlib.util, os, random, sys, types, unittest
from dataclasses import dataclass, field
from typing import Any, NewType

SRC = os.environ.get("VLLM_SRC", "/tmp/vllm-v0.30.0")


def _install_vllm():
    if "vllm.v1.kv_offload.cpu.policies.lru" in sys.modules:
        return
    for name in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu", "vllm.v1.kv_offload.cpu.policies"):
        m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
    base = types.ModuleType("vllm.v1.kv_offload.base")
    base.OffloadKey = NewType("OffloadKey", bytes)

    @dataclass
    class ReqContext:  # same fields and scratch methods as vllm/v1/kv_offload/base.py
        req_id: str
        kv_transfer_params: dict[str, Any] | None = None
        _state: dict = field(default_factory=dict, repr=False, init=False)

        def set_state(self, val):
            self._state[type(val)] = val

        def get_state(self, cls):
            return self._state.get(cls)

    base.ReqContext = ReqContext
    sys.modules["vllm.v1.kv_offload.base"] = base
    for mod in ("base", "lru"):
        name = f"vllm.v1.kv_offload.cpu.policies.{mod}"
        spec = importlib.util.spec_from_file_location(name, f"{SRC}/vllm/v1/kv_offload/cpu/policies/{mod}.py")
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)


_install_vllm()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm.v1.kv_offload.base import ReqContext  # noqa: E402
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus  # noqa: E402
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy  # noqa: E402
import gsch_member_policy as G  # noqa: E402


def K(*names):
    return [n.encode() for n in names]


def ctx(thread=None, members=None, epoch=0, source="s", rid="r"):
    p = None if members is None else {G.PARAMS_KEY: {"thread": thread, "members": list(members), "epoch": epoch, "source": source}}
    return ReqContext(req_id=rid, kv_transfer_params=p)


def store(pol, keys, c):
    """What vLLM does for one request: lookup-touch, then insert+ready (mark_evictable) every new chunk."""
    pol.touch(keys, c)
    for i, k in enumerate(keys):
        if pol.get(k) is None:
            ch = ChunkStatus(i); pol.insert(k, ch); ch.ref_cnt = 0; pol.mark_evictable(k)
    pol.touch(keys, c)  # vLLM touches again after prepare_store


def order(pol):
    return [k.decode() for k in pol.evictable_chunks]  # front = next evicted


class TestGroupMemberLRU(unittest.TestCase):
    def test_no_params_is_exactly_lru(self):
        rnd = random.Random(7)
        a, b = LRUCachePolicy(50), G.GroupMemberLRU(50)
        for step in range(400):
            keys = K(*[f"t{rnd.randrange(6)}-{j}" for j in range(rnd.randrange(1, 9))])
            for pol in (a, b):
                c = ReqContext(req_id=str(step))
                pol.touch(keys, c)
                for i, k in enumerate(keys):
                    if pol.get(k) is None:
                        if len(pol.chunks) >= 50:
                            ev = pol.evict(1, set(keys))
                            if ev is None:
                                continue
                        ch = ChunkStatus(i); pol.insert(k, ch); ch.ref_cnt = 0; pol.mark_evictable(k)
            self.assertEqual(list(a.evictable_chunks), list(b.evictable_chunks), f"step {step}")

    def test_departed_member_moves_to_the_eviction_end_tail_first(self):
        p = G.GroupMemberLRU(100)
        store(p, K("a0", "a1", "a2"), ctx("A", {"A", "B"}, 1))
        store(p, K("b0", "b1"), ctx("B", {"A", "B"}, 1))
        self.assertEqual(order(p), ["a2", "a1", "a0", "b1", "b0"], "LRU: the touch is tail-first, so a chain's head is its most recent")
        # B is newest; now A leaves: the next request (B's) carries the list without A
        p.touch(K("b0", "b1"), ctx("B", {"B"}, 2, rid="r2"))
        self.assertEqual(order(p)[:3], ["a2", "a1", "a0"], "A's chain is first in line, its tail first")
        ev = p.evict(2, set())
        self.assertEqual([k.decode() for k, _ in ev], ["a2", "a1"], "evictions take the departed tail first")

    def test_shared_blocks_stay_with_a_staying_member(self):
        p = G.GroupMemberLRU(100)
        store(p, K("s0", "s1", "a2"), ctx("A", {"A", "B"}, 1))       # A: shared prefix s0,s1 + own a2
        store(p, K("s0", "s1", "b2"), ctx("B", {"A", "B"}, 1, rid="r1"))
        store(p, K("c0"), ctx("C", {"A", "B", "C"}, 2, rid="r2"))
        p.touch(K("c0"), ctx("C", {"B", "C"}, 3, rid="r3"))           # A departs
        self.assertEqual(order(p)[0], "a2", "only A's unshared block is demoted")
        self.assertGreater(order(p).index("s0"), order(p).index("b2"), "the shared prefix keeps its place (B still holds it)")
        self.assertGreater(order(p).index("s1"), 0)

    def test_stale_list_is_ignored_and_new_source_accepted(self):
        p = G.GroupMemberLRU(100)
        store(p, K("a0"), ctx("A", {"A", "B"}, 5))
        store(p, K("b0"), ctx("B", {"A", "B"}, 5, rid="r1"))
        p.touch(K("b0"), ctx("B", {"B"}, 4, rid="r2"))                # older epoch: A did NOT depart
        self.assertEqual(p.stats["departed"], 0)
        p.touch(K("b0"), ctx("B", {"B"}, 0, source="new", rid="r3"))  # router restarted: accepted
        self.assertEqual(p.stats["departed"], 1)
        self.assertEqual(order(p)[0], "a0")

    def test_notice_applied_once_per_request(self):
        p = G.GroupMemberLRU(100)
        c = ctx("A", {"A"}, 1)
        store(p, K("a0"), c)
        self.assertEqual(p.stats["notices"], 1, "two touches of one request apply its notice once")

    def test_chunk_in_transfer_is_not_moved(self):
        p = G.GroupMemberLRU(100)
        store(p, K("a0", "a1"), ctx("A", {"A", "B"}, 1))
        store(p, K("b0"), ctx("B", {"A", "B"}, 1, rid="r1"))
        p.mark_non_evictable(b"a1"); p.chunks[b"a1"].ref_cnt = 1      # a1 is being loaded
        p.touch(K("b0"), ctx("B", {"B"}, 2, rid="r2"))
        self.assertEqual(order(p)[0], "a0")
        self.assertNotIn("a1", order(p))

    def test_returning_member_is_refreshed_by_its_own_touch(self):
        p = G.GroupMemberLRU(100)
        store(p, K("a0", "a1"), ctx("A", {"A", "B"}, 1))
        store(p, K("b0"), ctx("B", {"A", "B"}, 1, rid="r1"))
        p.touch(K("b0"), ctx("B", {"B"}, 2, rid="r2"))                 # A departs → front
        p.touch(K("a0", "a1"), ctx("A", {"A", "B"}, 3, rid="r3"))      # A seated again, its turn touches
        self.assertEqual(order(p)[-2:], ["a1", "a0"], "A's own touch makes its chain the most recent again")
        self.assertEqual(order(p)[0], "b0", "A is no longer first in line")

    def test_request_without_thread_or_from_non_member_records_nothing(self):
        p = G.GroupMemberLRU(100)
        store(p, K("x0"), ctx(None, {"A"}, 1))
        store(p, K("y0"), ctx("Z", {"A"}, 1, rid="r1"))
        self.assertEqual(p._lists, {})


class TestScanProtection(unittest.TestCase):
    def test_lookup_hits_are_protected_until_the_next_touch(self):
        p = G.GroupMemberLRU(100)
        store(p, K("a0", "a1"), ctx("A", {"A", "B"}, 1))
        store(p, K("b0", "b1"), ctx("B", {"A", "B"}, 1, rid="r1"))
        p.touch(K("b0", "b1"), ctx("B", {"B"}, 2, rid="r2"))       # A departs: a1, a0 at the front
        self.assertEqual(order(p)[:2], ["a1", "a0"])
        for k in K("a0", "a1"):                                     # A returns: its lookup scan finds them
            self.assertIsNotNone(p.get(k))
        ev = p.evict(1, set())                                      # a promotion further down A's chain needs room
        self.assertEqual([k.decode() for k, _ in ev], ["b1"], "the scan's own HITs are not evicted")
        p.touch(K("a0", "a1"), ctx("A", {"A", "B"}, 3, rid="r3"))  # the scan's touch ends the protection
        self.assertEqual(p._scan, set())

    def test_pinned_chunks_leave_the_protection(self):
        p = G.GroupMemberLRU(100)
        store(p, K("x0"), ctx("X", {"X"}, 1))
        p.get(b"x0"); p.mark_non_evictable(b"x0"); p.chunks[b"x0"].ref_cnt = 1  # a transfer read, not a lookup HIT
        self.assertNotIn(b"x0", p._scan)

    def test_protection_falls_back_when_nothing_else_is_evictable(self):
        p = G.GroupMemberLRU(100)
        store(p, K("y0"), ctx("Y", {"Y"}, 1))
        p.get(b"y0")
        ev = p.evict(1, set())
        self.assertEqual([k for k, _ in ev], [b"y0"], "never fail an eviction vLLM needs")
        self.assertEqual(p.stats["scan_protected_evictions"], 1)


if __name__ == "__main__":
    unittest.main()
