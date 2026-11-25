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

package approximateprefix

import "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"

// TierLRU is ONE endpoint's block-hash LRU — the same structure the residency index keeps per pod
// (hashLRU), with the same insertion order (AddChain is the indexer's Add: tail-first) — exported for a
// consumer that owns its own copy of an engine's CPU-tier model instead of reading this producer's.
//
// It has NO lock: the owner serialises every call (the group scheduler calls it only under its own
// mutex). Nothing here is a second implementation of the index: the LRU, the insertion order and the
// prefix-run search are this package's own.
type TierLRU struct {
	l *hashLRU
	// gen advances on every call that changes membership or recency, so a reader can memoise a walk.
	gen uint64
	// onEvict, when set, is told every key the LRU drops to make room for an insert.
	onEvict func(prefixhash.BlockHash)
}

// NewTierLRU returns an empty LRU holding at most capacity blocks.
func NewTierLRU(capacity int) *TierLRU {
	return &TierLRU{l: newHashLRU(capacity)}
}

// OnEvict installs the eviction observer (nil removes it). It is called synchronously inside AddChain,
// once per dropped key, before the key is gone.
func (t *TierLRU) OnEvict(f func(prefixhash.BlockHash)) { t.onEvict = f }

// AddChain records that the engine now holds keys — a prompt's block hashes from block 0 — exactly as
// the indexer's Add does: tail-first, so the chain's head ends most recent and its tail is evicted first,
// which keeps every held chain a contiguous run from block 0 (what ResidentPrefix relies on). A key
// already held is only refreshed.
func (t *TierLRU) AddChain(keys []prefixhash.BlockHash) {
	if len(keys) == 0 || t.l.capacity <= 0 {
		return
	}
	// REFRESH FIRST (vLLM's lookup touches the whole chain before any store): every key already held
	// moves to the fresh end, so the inserts below evict OTHER content, never this chain's own surviving
	// blocks (which would only be re-added: membership would end right, but the evictions — and an
	// eviction observer — would name the chain's own blocks). Tail-first, so the order is the one the
	// loop below leaves anyway.
	for idx := len(keys) - 1; idx >= 0; idx-- {
		if t.l.Contains(keys[idx]) {
			t.l.Add(keys[idx]) // a held key: moveFront only
		}
	}
	for idx := len(keys) - 1; idx >= 0; idx-- {
		k := keys[idx]
		if t.onEvict != nil && !t.l.Contains(k) && len(t.l.nodes) >= t.l.capacity && t.l.tail != nodeNil {
			t.onEvict(t.l.nodes[t.l.tail].key)
		}
		t.l.Add(k)
	}
	t.gen++
}

// ResidentPrefix is how many leading blocks of keys the LRU holds (the indexer's own run search).
func (t *TierLRU) ResidentPrefix(keys []prefixhash.BlockHash) int {
	if len(keys) == 0 {
		return 0
	}
	return runLength(t.l, keys)
}

// Contains reports whether one key is held (does not refresh it).
func (t *TierLRU) Contains(k prefixhash.BlockHash) bool { return t.l.Contains(k) }

// Len is the number of blocks held; Cap the capacity.
func (t *TierLRU) Len() int { return t.l.Len() }

// Cap is the capacity in blocks.
func (t *TierLRU) Cap() int { return t.l.capacity }

// Gen is the change counter (see gen).
func (t *TierLRU) Gen() uint64 { return t.gen }

// TailRoom is how many blocks an insert can take WITHOUT evicting a protected block: the free capacity,
// plus the run of blocks at the eviction end (the tail) up to the first block stop reports — blocks skip
// reports are walked over but not counted (the inserting chain's own blocks: evicting them would cost
// the insert itself). The walk ends early once need blocks are counted; the result is then >= need.
func (t *TierLRU) TailRoom(need int, skip, stop func(prefixhash.BlockHash) bool) int {
	room := t.l.capacity - len(t.l.index)
	if room < 0 {
		room = 0
	}
	for i := t.l.tail; i != nodeNil && room < need; i = t.l.nodes[i].prev {
		k := t.l.nodes[i].key
		// skip BEFORE stop: the inserting chain's own key is refreshed by its own touch, so it never
		// ends the walk — even when another (seated) chain shares it
		if skip != nil && skip(k) {
			continue
		}
		if stop != nil && stop(k) {
			break
		}
		room++
	}
	return room
}
