/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package groupsched

// THE CPU-TIER SHADOW (PREREG-multipod-offload §"THE OFFLOAD SOLUTION", piece 1).
//
// vLLM's CPU offload tier is an LRU of the blocks the engine has stored out of HBM:
// on a request's arrival the engine looks the prefix up in GPU memory, THEN in the
// CPU tier, and restores any CPU hit automatically (a restore is ~1-4 % of a
// prefill). So a block is *cheap* while it is in either place, and the ONLY question
// the scheduler needs answered is "is this block still somewhere the engine can reach
// without recomputing it?".
//
// This is that answer, as a MODEL — the same dictate-don't-observe discipline as the
// GPU residue shadow it sits next to: it reads NO engine cache state, no vLLM metric,
// no hook. It replays the tier's own eviction rule over the blocks the scheduler
// itself dispatched:
//
//	capacity = the configured tier size in blocks (cpuTierBytes ÷ kvBlockBytes) —
//	           a DEPLOYMENT FACT, not a knob: it is the engine's tmpfs tier divided by
//	           one KV block, exactly as the seat budget is a fraction of the engine's
//	           measured num_gpu_blocks;
//	on DISPATCH of a request to engine e — its chain blocks move to MRU (the engine
//	           holds them, and vLLM re-stores what it later evicts);
//	on SETTLE  — the request's blocks are INSERTED (they are now stored: the engine
//	           keeps a request's blocks in the tier once it is done with them);
//	whenever len > capacity — the LRU tail is evicted, which is where the measured
//	           survival cliff comes from (c128: restored share 90 % under 1 min, 30 %
//	           at 1-3 min, 14 % at 3-5 min).
//
// It is nil when no tier is configured, and every method is nil-safe: an arm that
// runs without offload pays nothing and its decisions cannot depend on this file.
//
// NOT MODELLED, deliberately (each is a place the model is OPTIMISTIC, and the
// shadow-validation metric is what measures the resulting error):
//   - the store rate: the real tier fills at PCIe speed, so between a settle and the
//     store's completion a block is neither resident nor cold — the model calls it
//     resident immediately (the tier's fill time is seconds, the cliff is minutes);
//   - "don't store" hints: there are none, so a branch that never returns still
//     occupies the tier until the LRU pushes it out (62 % of threads are one-shot
//     leaves — the measured store pollution);
//   - the tier is PER ENGINE (a block on engine A's tier is worthless to B), so the
//     shadow is per engine like every other engine fact here.

import (
	"container/list"
)

// tierShadow is ONE engine's model of its CPU offload tier: a bounded LRU of block
// keys. It is not safe for concurrent use on its own — the Scheduler owns it and
// touches it only under s.mu, like every other per-engine fact.
type tierShadow struct {
	// capacity is the tier in BLOCKS: cpuTierBytes ÷ kvBlockBytes, floored. 0 means
	// "no tier configured" (the shadow is then nil, see newTierShadow).
	capacity int
	// version bumps whenever RESIDENCY can have changed (an insert or an eviction).
	// Recency-only moves (Touch) do not bump it: the estimate cache keys on "is this
	// block resident", not on where it sits in the LRU.
	version uint64
	order   *list.List // front = MRU, back = the next eviction
	where   map[Hash]*list.Element
}

// newTierShadow builds the shadow for one engine. capacityBlocks <= 0 returns nil:
// with no configured tier there is nothing to model, and nil is the value every
// caller can test cheaply (Enabled).
func newTierShadow(capacityBlocks int) *tierShadow {
	if capacityBlocks <= 0 {
		return nil
	}
	return &tierShadow{
		capacity: capacityBlocks,
		order:    list.New(),
		where:    make(map[Hash]*list.Element, capacityBlocks),
	}
}

// Enabled reports whether a tier is configured on this engine.
func (t *tierShadow) Enabled() bool { return t != nil && t.capacity > 0 }

// Len is how many blocks the shadow believes are stored in the tier.
func (t *tierShadow) Len() int {
	if t == nil {
		return 0
	}
	return len(t.where)
}

// Capacity is the tier in blocks (0 when no tier is configured).
func (t *tierShadow) Capacity() int {
	if t == nil {
		return 0
	}
	return t.capacity
}

// Version is the residency counter (the tier-aware estimate cache's key).
func (t *tierShadow) Version() uint64 {
	if t == nil {
		return 0
	}
	return t.version
}

// Holds reports whether the tier is believed to hold this block. A nil shadow holds
// nothing, which is what makes every tier-aware computation degenerate to the
// GPU-only one when no tier is configured.
func (t *tierShadow) Holds(h Hash) bool {
	if t == nil {
		return false
	}
	_, ok := t.where[h]
	return ok
}

// Touch moves the blocks the engine ALREADY holds to MRU (the dispatch hook). Blocks
// the tier does not hold are not inserted here: on dispatch they are being prefilled
// into HBM, and they enter the tier at SETTLE (Insert) — the two events are different
// facts and keeping them apart is what makes the model's fill semantics explicit.
func (t *tierShadow) Touch(keys []Hash) {
	if !t.Enabled() {
		return
	}
	for _, h := range keys {
		if el, ok := t.where[h]; ok {
			t.order.MoveToFront(el)
		}
	}
}

// Insert registers the blocks of a SETTLED request: a block the tier does not hold
// goes to MRU, a block it already holds is left where it is (the dispatch Touch owns
// recency — re-ranking here would let a settle jump blocks dispatched later), and the
// LRU tail is evicted until the tier is back inside its capacity.
func (t *tierShadow) Insert(keys []Hash) {
	if !t.Enabled() {
		return
	}
	changed := false
	for _, h := range keys {
		if _, ok := t.where[h]; ok {
			continue
		}
		t.where[h] = t.order.PushFront(h)
		changed = true
	}
	for len(t.where) > t.capacity {
		tail := t.order.Back()
		if tail == nil {
			break // unreachable: where and order are maintained together
		}
		t.order.Remove(tail)
		delete(t.where, tail.Value.(Hash))
		changed = true
	}
	if changed {
		t.version++
	}
}

// tierBlocksFor is the shadow's capacity in BLOCKS from the two deployment facts:
// the tier's byte size and one KV block's byte size. 0 (the shadow OFF) whenever
// either fact is missing, and also when the tier is smaller than one block — a tier
// that can hold nothing predicts nothing, and silently rounding it up to 1 would make
// every arm look warm.
func tierBlocksFor(cpuTierBytes, kvBlockBytes int64) int {
	if cpuTierBytes <= 0 || kvBlockBytes <= 0 {
		return 0
	}
	n := cpuTierBytes / kvBlockBytes
	if n <= 0 {
		return 0
	}
	return int(n)
}
