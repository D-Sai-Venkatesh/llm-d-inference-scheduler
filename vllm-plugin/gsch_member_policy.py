# SPDX-License-Identifier: Apache-2.0
"""Group-membership-aware eviction for vLLM's CPU offloading tier (an out-of-tree CachePolicy).

The group scheduler (llm-d EPP, groupsched) seats a set of threads per engine. When a thread LEAVES the
group, its KV in the CPU tier is not dropped: it is MOVED TO THE EVICTION END of the LRU, so the next write
that needs room takes it first, and a seated member's blocks go only once every departed block is gone.
Everything else is vLLM's own LRU (LRUCachePolicy): recency, eviction from the front, ref-count handling.

Membership arrives with each request, in kv_transfer_params["gsch_member"] (written by the router):
    {"thread": "<this request's thread>", "members": ["<staying threads on this engine>", ...],
     "epoch": <int, grows with every membership change on this engine>, "source": "<router-side incarnation>"}
A list is accepted when its source differs from the last accepted one or its epoch is newer. Threads that
were members and are not any more are DEPARTED: their last chain's keys that no staying member also holds
move to the eviction end, in chain order, so a chain's tail goes first and what survives stays a prefix.
A request without the field changes nothing: the policy is then exactly LRUCachePolicy.

Enable (kv_transfer_config.kv_connector_extra_config):
    "eviction_policy": "GroupMemberLRU", "cache_policy_module_path": "gsch_member_policy"
with this file importable by the engine (e.g. mounted on PYTHONPATH).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

PARAMS_KEY = "gsch_member"
LOG_EVERY_S = 60.0
logger = logging.getLogger("vllm.gsch_member_policy")


class _Notice:
    """One request's parsed notice (kept in the request's ReqContext scratch, so it is parsed and applied
    once although vLLM touches a request's chunks several times)."""

    __slots__ = ("thread", "members", "epoch", "source", "applied")

    def __init__(self, thread, members, epoch, source):
        self.thread = thread
        self.members = members
        self.epoch = epoch
        self.source = source
        self.applied = False


def _parse(req_context: ReqContext) -> _Notice | None:
    cached = req_context.get_state(_Notice)
    if cached is not None:
        return cached
    params = req_context.kv_transfer_params
    raw = params.get(PARAMS_KEY) if isinstance(params, dict) else None
    if not isinstance(raw, dict):
        return None
    members = raw.get("members")
    epoch = raw.get("epoch")
    if not isinstance(members, list) or type(epoch) is not int:
        return None
    thread = raw.get("thread")
    n = _Notice(thread if isinstance(thread, str) else None,
                frozenset(m for m in members if isinstance(m, str)), epoch, str(raw.get("source", "")))
    req_context.set_state(n)
    return n


class GroupMemberLRU(LRUCachePolicy):
    """LRU whose departed-member chunks move to the eviction end (see the module doc)."""

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._members: frozenset[str] = frozenset()
        self._source: str | None = None
        self._epoch = -1
        # the last chain (offload keys, in chain order) each STAYING member's requests touched
        self._lists: dict[str, tuple[OffloadKey, ...]] = {}
        # key -> how many staying members' last chains hold it (a shared key is not demoted)
        self._stay: dict[OffloadKey, int] = {}
        # keys a lookup scan found since the last touch: protected from eviction until that touch, so a request's own
        # promotions further down its chain cannot evict the prefix it just counted as a HIT (vLLM does not pin them)
        self._scan: set[OffloadKey] = set()
        self.stats = {"notices": 0, "lists_accepted": 0, "departed": 0, "demoted": 0, "scan_protected_evictions": 0}
        self._last_log = time.monotonic()

    # --- the one override of LRU behaviour -------------------------------------------------------
    def get(self, key: OffloadKey):
        chunk = self.chunks.get(key)
        if chunk is not None and key in self.evictable_chunks:
            self._scan.add(key)
        return chunk

    def mark_non_evictable(self, key: OffloadKey) -> None:
        # a chunk pinned for a transfer (prepare_load / a cascade read) was not a lookup HIT of the scan in progress
        self._scan.discard(key)
        super().mark_non_evictable(key)

    def evict(self, n: int, protected: set[OffloadKey]):
        if self._scan:
            out = super().evict(n, protected | self._scan)
            if out is not None:
                return out
            self.stats["scan_protected_evictions"] += 1  # not enough room otherwise: fall back to plain order
        return super().evict(n, protected)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self._scan.clear()
        notice = _parse(req_context)
        if notice is not None and not notice.applied:
            notice.applied = True
            keys = list(keys)
            self._apply(notice, keys)
        super().touch(keys, req_context)

    # --- membership -------------------------------------------------------------------------------
    def _apply(self, n: _Notice, keys: list[OffloadKey]) -> None:
        self.stats["notices"] += 1
        if n.source != self._source or n.epoch > self._epoch:
            self.stats["lists_accepted"] += 1
            departed = self._members - n.members
            self._members = n.members
            self._source, self._epoch = n.source, n.epoch
            for t in departed:
                chain = self._lists.pop(t, None)
                self.stats["departed"] += 1
                if chain:
                    self._count(chain, -1)
                    self._demote(chain)
        if n.thread is not None and n.thread in self._members:
            old = self._lists.get(n.thread)
            new = tuple(keys)
            if old != new:
                if old:
                    self._count(old, -1)
                self._count(new, +1)
                self._lists[n.thread] = new
        self._maybe_log()

    def _count(self, chain: tuple[OffloadKey, ...], d: int) -> None:
        stay = self._stay
        for k in set(chain):
            c = stay.get(k, 0) + d
            if c > 0:
                stay[k] = c
            else:
                stay.pop(k, None)

    def _demote(self, chain: tuple[OffloadKey, ...]) -> None:
        # forward order with last=False: each key goes to the front, so the chain's LAST key ends first
        # in line (a chain's tail is evicted first and what survives stays a prefix from block 0)
        ev, stay = self.evictable_chunks, self._stay
        moved = 0
        for k in chain:
            if k in ev and k not in stay:
                ev.move_to_end(k, last=False)
                moved += 1
        self.stats["demoted"] += moved

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log >= LOG_EVERY_S:
            self._last_log = now
            logger.info("gsch_member_policy: %s members=%d tracked_keys=%d", self.stats,
                        len(self._members), len(self._stay))

    def clear(self) -> None:
        super().clear()
        self._scan.clear()
        self._lists.clear()
        self._stay.clear()
