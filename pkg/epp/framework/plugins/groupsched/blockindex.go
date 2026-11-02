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

// THE BLOCK INDEX — one engine's model of which KV blocks it holds.
//
// It answers three questions, all in BLOCKS (the seat budget's unit):
//   - what does this branch cost on this engine? the cold part of its chains, exactly;
//   - how full is the engine? two refcounts, one per question (below);
//   - may this branch be seated? fit and seat in ONE critical section.
//
// The keys are chained block hashes of the engine's own tokens (prefixhash: block i
// commits to every block before it). Two consequences the whole design rests on:
//
//   - RESIDENCY IS PREFIX-CLOSED. A chain's blocks are added and removed as a whole, and
//     a key commits to its prefix, so if block i of a chain is held then 0..i-1 are held
//     too. The resident part of a chain is therefore a PREFIX, found by binary search —
//     never by scanning a set.
//   - SHARING IS A SHARED PREFIX. Two branches share blocks exactly when their prompts
//     share a prefix, which is what the engine's own prefix cache dedups.
//
// TWO REFCOUNTS, ONE PER QUESTION:
//
//	blocks     (USED) — every seated branch, exiting ones included. An exiting branch's
//	                    blocks are still pinned by the engine until its request drains, so
//	                    ADMISSION asks about this one.
//	blocksStay (STAY) — the seated branches that are NOT exiting. Those are the blocks
//	                    that will still be there when the retirees are gone, so
//	                    ENFORCEMENT asks about this one. A block an exiting branch
//	                    shares with a staying one stays counted here, which is the
//	                    under-eviction the two views exist to avoid.
//
// A branch's record exists exactly as long as it holds a seat: it is created by the
// seat and dropped by the unseat, so nothing here is a history.

import (
	"errors"
	"fmt"
	"sync"
)

// ErrBlockSizeUnknown is the fail-closed startup error: the block size must be
// measured, never assumed (a hardcoded default once admitted four times the cache).
var ErrBlockSizeUnknown = errors.New("groupsched: engine block size not set (fail closed)")

// Hash is one chained block hash (prefixhash's key space).
type Hash uint64

// Chain is one prompt's EXACT prefill chain: the block keys of its tokens (the last
// block is partial, so its key is private to that exact token sequence) and the
// prompt's token count.
//
// The chains are computed ONCE per request, at its first Pick, from the tokens the
// token producer rendered at arrival — so every cost the scheduler takes is exact for
// the prefill it will run. Decode growth is not part of it: the seat budget's headroom
// pays for that.
type Chain struct {
	Keys   []Hash
	Tokens int
}

// BlockIndex is one engine's KV-block accounting.
//
// Safe for concurrent use; the scheduler calls the Locked variants under its own lock.
type BlockIndex struct {
	mu              sync.Mutex
	blockSizeTokens int

	blocks     map[Hash]int
	blocksStay map[Hash]int
	// seats holds every seated branch's record. There is no record for a branch without
	// a seat: the head carries its own exact chains, so the index has nothing to
	// remember between residencies.
	seats map[string]*seatBlocks
}

// seatBlocks is one seated branch's footprint: its newest chains (the turns of its
// branch extend one another, so the newest chain is its set) and the keys actually
// charged for it.
type seatBlocks struct {
	chains []Chain
	// charged is the deduplicated union of the chains' keys: what the two refcounts were
	// incremented by at the seat, and what the unseat releases.
	charged map[Hash]struct{}
	exiting bool
}

// NewBlockIndex returns the index for an engine whose KV blocks hold blockSizeTokens
// tokens. It fails closed on a non-positive block size.
func NewBlockIndex(blockSizeTokens int) (*BlockIndex, error) {
	if blockSizeTokens <= 0 {
		return nil, fmt.Errorf("%w: got %d", ErrBlockSizeUnknown, blockSizeTokens)
	}
	return &BlockIndex{
		blockSizeTokens: blockSizeTokens,
		blocks:          make(map[Hash]int),
		blocksStay:      make(map[Hash]int),
		seats:           make(map[string]*seatBlocks),
	}, nil
}

// BlockSizeTokens is the engine block size in tokens.
func (b *BlockIndex) BlockSizeTokens() int { return b.blockSizeTokens }

// chargeSet deduplicates the union of the chains' keys. Duplicate keys are possible
// within one request: two identical prompts render identical chains, and the engine
// holds that block once.
func chargeSet(chains []Chain) map[Hash]struct{} {
	var out map[Hash]struct{}
	for _, c := range chains {
		if out == nil {
			out = make(map[Hash]struct{}, len(c.Keys))
		}
		for _, h := range c.Keys {
			out[h] = struct{}{}
		}
	}
	return out
}

// residentPrefixLocked is how many of the chain's blocks this engine already holds —
// the resident PREFIX, which is a binary search because residency is prefix-closed.
func (b *BlockIndex) residentPrefixLocked(keys []Hash) int {
	lo, hi := 0, len(keys)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if _, held := b.blocks[keys[mid]]; held {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// CostLocked is the exact admission cost of a branch on this engine, in blocks: what
// the engine would have to compute, i.e. every block beyond its resident prefix, per
// chain.
func (b *BlockIndex) CostLocked(chains []Chain) int {
	n := 0
	for _, c := range chains {
		n += len(c.Keys) - b.residentPrefixLocked(c.Keys)
	}
	return n
}

// ColdTokensLocked is the same cost in TOKENS (the cold valve's unit): the prompt's
// tokens beyond the resident prefix. It counts the partial last block exactly, which
// the block-rounded cost cannot, and a fully resident chain is zero.
func (b *BlockIndex) ColdTokensLocked(chains []Chain) int64 {
	var n int64
	for _, c := range chains {
		p := b.residentPrefixLocked(c.Keys)
		if p == len(c.Keys) {
			continue // nothing to prefill
		}
		n += int64(c.Tokens - p*b.blockSizeTokens)
	}
	return n
}

// TrySeatLocked seats a branch if its exact cost fits the budget, and inserts its chains
// in the same critical section: the fit test and the seat can never disagree, because
// there is one number and one moment.
func (b *BlockIndex) TrySeatLocked(id string, chains []Chain, budgetBlocks int) (bool, int) {
	if _, seated := b.seats[id]; seated {
		return true, 0
	}
	cost := b.CostLocked(chains)
	if len(b.blocks)+cost > budgetBlocks {
		return false, cost
	}
	b.insertLocked(id, chains)
	return true, cost
}

// insertLocked creates the seat's record and charges its keys to both counters.
func (b *BlockIndex) insertLocked(id string, chains []Chain) {
	s := &seatBlocks{chains: chains, charged: chargeSet(chains)}
	b.seats[id] = s
	for h := range s.charged {
		b.blocks[h]++
		b.blocksStay[h]++
	}
}

// SeatForceLocked seats a branch WITHOUT the budget check: the blocks are taken as
// already present on this engine (an observed placement said so — the engine, not the
// model, is authoritative about what it holds). Enforcement reconciles the budget; a
// branch absent from the engine serving it would not be recoverable.
func (b *BlockIndex) SeatForceLocked(id string, chains []Chain) {
	if s, seated := b.seats[id]; seated {
		b.GrowLocked(id, chains)
		_ = s
		return
	}
	b.insertLocked(id, chains)
}

// GrowLocked makes a seated branch's chains the NEWEST ones it has: a branch's turns
// extend one another, so the newest chain is its set, and the difference is charged (or
// released, for the partial block the growing tail replaced).
func (b *BlockIndex) GrowLocked(id string, chains []Chain) {
	s, ok := b.seats[id]
	if !ok || sameChains(s.chains, chains) {
		return
	}
	old := s.charged
	s.chains = chains
	s.charged = chargeSet(chains)
	b.applyDeltaLocked(s, old)
}

// sameChains reports whether two chain sets are the same length for every prompt — the
// O(1) check that a branch's set did not change (the common case at a 1 kHz Pick).
func sameChains(a, c []Chain) bool {
	if len(a) != len(c) {
		return false
	}
	for i := range a {
		if len(a[i].Keys) != len(c[i].Keys) || a[i].Tokens != c[i].Tokens {
			return false
		}
	}
	return true
}

// applyDeltaLocked charges the keys the branch gained and releases the keys it lost.
// A non-exiting branch moves both counters; an exiting one only the USED side (its stay
// charge was released at its mark).
func (b *BlockIndex) applyDeltaLocked(s *seatBlocks, old map[Hash]struct{}) {
	for h := range s.charged {
		if _, had := old[h]; !had {
			b.blocks[h]++
			if !s.exiting {
				b.blocksStay[h]++
			}
		}
	}
	for h := range old {
		if _, has := s.charged[h]; !has {
			b.dec(b.blocks, h)
			if !s.exiting {
				b.dec(b.blocksStay, h)
			}
		}
	}
}

// UnseatLocked releases a branch's seat and drops its record: this is the ONE release
// point, and the record does not outlive the seat (a re-seat costs the branch nothing to
// rebuild — its head carries the chain).
func (b *BlockIndex) UnseatLocked(id string) {
	s, ok := b.seats[id]
	if !ok {
		return
	}
	for h := range s.charged {
		b.dec(b.blocks, h)
		if !s.exiting {
			b.dec(b.blocksStay, h) // an exiting branch left the STAY side at its mark
		}
	}
	delete(b.seats, id)
}

// Unseat is UnseatLocked under the index's own lock.
func (b *BlockIndex) Unseat(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.UnseatLocked(id)
}

// dec decrements a refcount, panicking on underflow: a refcount below zero is a
// programming error, and a silent one would poison every later decision.
func (b *BlockIndex) dec(m map[Hash]int, h Hash) {
	c, ok := m[h]
	if !ok || c <= 0 {
		panic(fmt.Sprintf("groupsched: refcount underflow for hash %d", h))
	}
	if c == 1 {
		delete(m, h)
	} else {
		m[h] = c - 1
	}
}

// MarkExitingLocked marks a branch as on its way out: it is not dispatched and not
// re-seated, and it leaves at its next drain. Its blocks stay on the USED side — the
// engine pins them until the request finishes — and leave the STAY side, which is what
// tells enforcement that evicting it again would free nothing. Idempotent.
func (b *BlockIndex) MarkExitingLocked(id string) {
	s, ok := b.seats[id]
	if !ok || s.exiting {
		return
	}
	s.exiting = true
	for h := range s.charged {
		b.dec(b.blocksStay, h)
	}
}

// MarkExiting is MarkExitingLocked under the index's own lock.
func (b *BlockIndex) MarkExiting(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.MarkExitingLocked(id)
}

// ExitingLocked reports the branch's exiting state.
func (b *BlockIndex) ExitingLocked(id string) bool {
	s, ok := b.seats[id]
	return ok && s.exiting
}

// Exiting is ExitingLocked under the index's own lock.
func (b *BlockIndex) Exiting(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.ExitingLocked(id)
}

// Seated reports whether the branch holds a seat here.
func (b *BlockIndex) Seated(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, ok := b.seats[id]
	return ok
}

// ---- the counted views ----

// UniqueBlocksLocked is the USED side: every seated branch's blocks, exiting included.
func (b *BlockIndex) UniqueBlocksLocked() int { return len(b.blocks) }

// UniqueBlocks is UniqueBlocksLocked under the index's own lock.
func (b *BlockIndex) UniqueBlocks() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.blocks)
}

// StayBlocksLocked is the STAY side: the blocks of the branches that are not exiting.
// Enforcement measures the overage against it.
func (b *BlockIndex) StayBlocksLocked() int { return len(b.blocksStay) }

// StayBlocks is StayBlocksLocked under the index's own lock.
func (b *BlockIndex) StayBlocks() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.blocksStay)
}

// ExitingBlocksLocked is the part of the USED side that belongs only to exiting branches
// (used − stay): the overage enforcement must NOT evict again, and admission still
// refuses.
func (b *BlockIndex) ExitingBlocksLocked() int { return len(b.blocks) - len(b.blocksStay) }

// ExitingBlocks is ExitingBlocksLocked under the index's own lock.
func (b *BlockIndex) ExitingBlocks() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.blocks) - len(b.blocksStay)
}

// HasNELocked reports whether a key is on the USED side.
func (b *BlockIndex) HasNELocked(h Hash) bool {
	_, ok := b.blocks[h]
	return ok
}

// ChainsLocked is a seated branch's chains: what a rebind carries to the engine that is
// observed to be serving it.
func (b *BlockIndex) ChainsLocked(id string) []Chain {
	s, ok := b.seats[id]
	if !ok {
		return nil
	}
	return s.chains
}

// TrackedCount is the number of seat records (one per seat, by construction).
func (b *BlockIndex) TrackedCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.seats)
}

// checkInvariantLocked recomputes both refcounts from the records and panics on
// divergence. Tests call it after every mutation.
func (b *BlockIndex) checkInvariantLocked() {
	want := make(map[Hash]int)
	wantStay := make(map[Hash]int)
	for _, s := range b.seats {
		for h := range s.charged {
			want[h]++
			if !s.exiting {
				wantStay[h]++
			}
		}
	}
	if !sameCounts(want, b.blocks) {
		panic(fmt.Sprintf("invariant: blocks %v != recomputed %v", b.blocks, want))
	}
	if !sameCounts(wantStay, b.blocksStay) {
		panic(fmt.Sprintf("invariant: blocksStay %v != recomputed %v", b.blocksStay, wantStay))
	}
}

func sameCounts(a, c map[Hash]int) bool {
	if len(a) != len(c) {
		return false
	}
	for h, n := range a {
		if c[h] != n {
			return false
		}
	}
	return true
}
