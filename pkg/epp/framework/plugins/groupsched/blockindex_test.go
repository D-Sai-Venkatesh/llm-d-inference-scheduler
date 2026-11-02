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

// THE BLOCK INDEX's own suite: the two refcounts, the exact cost, and the property the
// binary search rests on (residency is prefix-closed).

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

func newIdx(t *testing.T) *BlockIndex {
	t.Helper()
	idx, err := NewBlockIndex(64)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

// chain is a one-prompt chain of n blocks drawn from a shared or private key space:
// same seed ⇒ same prefix (the shared-prefix shape), different seed ⇒ no sharing.
func chain(seed string, n int) []Chain {
	return []Chain{{Keys: keys(n, seedFor(seed)), Tokens: n * 64}}
}

// chainWithPrefix is a chain that REUSES another branch's first prefixBlocks keys: the
// shape a fork (or a growing turn) has on one engine.
func chainWithPrefix(prefix string, prefixBlocks int, own string, ownBlocks int) []Chain {
	k := append(keys(prefixBlocks, seedFor(prefix)), keys(ownBlocks, seedFor(own))...)
	return []Chain{{Keys: k, Tokens: (prefixBlocks + ownBlocks) * 64}}
}

func mustSeat(t *testing.T, idx *BlockIndex, id string, chains []Chain, budget int) int {
	t.Helper()
	ok, cost := idx.TrySeatLocked(id, chains, budget)
	if !ok {
		t.Fatalf("TrySeat(%s) rejected (cost %d)", id, cost)
	}
	return cost
}

func TestFailClosedOnUnknownBlockSize(t *testing.T) {
	if _, err := NewBlockIndex(0); err == nil {
		t.Fatal("block size 0 must fail closed")
	}
	if _, err := NewBlockIndex(-1); err == nil {
		t.Fatal("negative block size must fail closed")
	}
	idx := newIdx(t)
	if idx.BlockSizeTokens() != 64 {
		t.Fatalf("block size: %d want 64", idx.BlockSizeTokens())
	}
}

// ---- the exact cost ----

// A head's cost is everything beyond the engine's resident PREFIX: the shared prefix is
// free, and the partial last block is paid for (it must be computed).
func TestCostIsTheColdPartOfTheChain(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 1000)
	if got := idx.CostLocked(chainWithPrefix("A", 3, "B", 2)); got != 2 {
		t.Fatalf("a head sharing 3 of a's 4 blocks costs %d, want 2 (1 new shared tail + 1 private)", got)
	}
	if got := idx.CostLocked(chain("A", 4)); got != 0 {
		t.Fatalf("a fully resident chain costs %d, want 0", got)
	}
	if got := idx.ColdTokensLocked(chain("A", 4)); got != 0 {
		t.Fatalf("a fully resident chain has %d cold tokens, want 0", got)
	}
	// the partial tail: 100 tokens at 64/block is 2 blocks, the second partial
	partial := []Chain{{Keys: keys(2, seedFor("P")), Tokens: 100}}
	if got := idx.CostLocked(partial); got != 2 {
		t.Fatalf("a 100-token prompt costs %d blocks, want 2 (the partial block counts)", got)
	}
	if got := idx.ColdTokensLocked(partial); got != 100 {
		t.Fatalf("a 100-token prompt is %d cold tokens, want 100", got)
	}
}

// One number, one moment: TrySeat's cost is the same computation the placement module
// makes, and a rejected seat charges nothing.
func TestTrySeatAtomicBudget(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 100)
	// budget 6: b shares 2 of a's blocks and adds 2 → exactly 6
	ok, cost := idx.TrySeatLocked("b", chainWithPrefix("A", 2, "B", 2), 6)
	if !ok || cost != 2 {
		t.Fatalf("seat b: ok=%v cost=%d want true/2", ok, cost)
	}
	ok, cost = idx.TrySeatLocked("c", chain("C", 5), 6)
	if ok || cost != 5 {
		t.Fatalf("seat c: ok=%v cost=%d want false/5", ok, cost)
	}
	if idx.Seated("c") {
		t.Fatal("a rejected seat must not be seated")
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// ---- the two refcounts (R4) ----

// An exiting branch's blocks are still pinned by the engine, so they stay on the USED
// side and leave the STAY side: that difference is what enforcement acts on.
func TestExitingMovesStayNotUsed(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 3), 1000)
	used, stay := idx.UniqueBlocks(), idx.StayBlocks()
	idx.MarkExitingLocked("a")
	if idx.UniqueBlocks() != used {
		t.Fatalf("exiting moved the used side: %d -> %d", used, idx.UniqueBlocks())
	}
	if idx.StayBlocks() != stay-3 {
		t.Fatalf("exiting must release the stay side: %d -> %d", stay, idx.StayBlocks())
	}
	if idx.ExitingBlocksLocked() != 3 {
		t.Fatalf("exiting blocks: %d want 3", idx.ExitingBlocksLocked())
	}
	idx.UnseatLocked("a")
	if idx.UniqueBlocks() != 0 || idx.StayBlocks() != 0 {
		t.Fatalf("after the drain: used %d stay %d, want 0/0", idx.UniqueBlocks(), idx.StayBlocks())
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// A block an exiting branch SHARES with a staying one must stay on the staying branch's
// account: excusing the exiting branch's whole set is the under-eviction R4 exists for.
func TestExitingDoesNotExcuseASharedBlock(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "parent", chain("P", 3), 1000)
	mustSeat(t, idx, "child", chainWithPrefix("P", 3, "C", 2), 1000) // 3 shared + 2 own
	if used, stay := idx.UniqueBlocksLocked(), idx.StayBlocksLocked(); used != 5 || stay != 5 {
		t.Fatalf("premise: used %d stay %d, want 5/5", used, stay)
	}
	idx.MarkExitingLocked("child")
	// Only the child's OWN two blocks leave the stay side: the three it shares are still
	// held by the parent, so they stay. (The excuse-based arithmetic this replaces read
	// the child's whole set as leaving — used 5 − `exit` 5 = nothing over the budget,
	// while the parent's three blocks were still pinned.)
	if got := idx.StayBlocksLocked(); got != 3 {
		t.Fatalf("stay after the child exits: %d want 3 (the 3 shared blocks stay with the parent)", got)
	}
	if got := idx.UniqueBlocksLocked(); got != 5 {
		t.Fatalf("used after the child exits: %d want 5 (the child's blocks are still pinned)", got)
	}
	idx.UnseatLocked("child")
	if got := idx.StayBlocksLocked(); got != 3 {
		t.Fatalf("stay after the child drains: %d want 3", got)
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// MarkExiting is idempotent: repeating it cannot double-release the stay side.
func TestMarkExitingIdempotent(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 1000)
	idx.MarkExitingLocked("a")
	stay := idx.StayBlocksLocked()
	idx.MarkExitingLocked("a")
	idx.MarkExitingLocked("a")
	if idx.StayBlocksLocked() != stay {
		t.Fatalf("triple MarkExiting moved the stay side")
	}
	if idx.UniqueBlocksLocked() != 4 {
		t.Fatalf("used after triple MarkExiting: %d want 4", idx.UniqueBlocksLocked())
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// An exiting branch's growth is real KV the engine pins, so it is charged — to the USED
// side only.
func TestExitingGrowthStaysCharged(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 3), 1000)
	idx.MarkExitingLocked("a")
	used, stay := idx.UniqueBlocksLocked(), idx.StayBlocksLocked()
	idx.GrowLocked("a", chain("A", 5))
	if got := idx.UniqueBlocksLocked(); got != used+2 {
		t.Fatalf("used after an exiting branch grows: %d want %d", got, used+2)
	}
	if got := idx.StayBlocksLocked(); got != stay {
		t.Fatalf("stay after an exiting branch grows: %d want %d", got, stay)
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// Growth replaces the partial tail block: the grown chain's tail is a NEW key, and the
// old one leaves. The counters follow exactly.
func TestGrowthReplacesThePartialTail(t *testing.T) {
	idx := newIdx(t)
	// 182 tokens is two whole blocks and a partial third; 212 tokens makes that third
	// whole and adds a partial fourth, so the third block's key CHANGES and a fourth
	// appears.
	full := keys(2, seedFor("A"))
	oldTail, newTail, fourth := Hash(0x1001), Hash(0x1002), Hash(0x1003)
	short := []Chain{{Keys: append(append([]Hash{}, full...), oldTail), Tokens: 182}}
	idx.TrySeatLocked("a", short, 1000)
	if got := idx.UniqueBlocksLocked(); got != 3 {
		t.Fatalf("used: %d want 3", got)
	}
	grown := []Chain{{Keys: append(append([]Hash{}, full...), newTail, fourth), Tokens: 212}}
	idx.GrowLocked("a", grown)
	if got := idx.UniqueBlocksLocked(); got != 4 {
		t.Fatalf("used after growth: %d want 4 (the tail was replaced, one block added)", got)
	}
	if idx.HasNELocked(oldTail) {
		t.Fatal("the replaced partial tail must leave the used side")
	}
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// The record does not outlive the seat: a re-seat costs the branch's chain again,
// because nothing was remembered between residencies.
func TestNoResidencyMemoryBetweenSeats(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 4), 1000)
	idx.UnseatLocked("a")
	if idx.TrackedCount() != 0 {
		t.Fatalf("a record outlived its seat: %d records", idx.TrackedCount())
	}
	ok, cost := idx.TrySeatLocked("a", chain("A", 4), 1000)
	if !ok || cost != 4 {
		t.Fatalf("re-seat: ok=%v cost=%d want true/4 (nothing is remembered)", ok, cost)
	}
}

func TestDoubleUnseatIsSafeNoOp(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chain("A", 3), 1000)
	idx.UnseatLocked("a")
	after := idx.UniqueBlocksLocked()
	idx.UnseatLocked("a") // no record, so nothing to release — not an underflow
	if idx.UniqueBlocksLocked() != after {
		t.Fatalf("double unseat changed the counters: %d -> %d", after, idx.UniqueBlocksLocked())
	}
}

// ---- R2's property: prefix closure, and the binary search equals a brute-force scan ----

// TestPrefixClosureAndSearchCost runs random seat/grow/exit sequences and checks, after
// every step, (1) that residency is PREFIX-CLOSED — if block i of a chain is held, every
// block before it is — and (2) that the O(log n) cost equals the brute-force count of the
// chain's blocks the engine does not hold. (2) is what the search is worth on; (1) is why
// it is legal.
//
// The chains are built the way production builds them: as prefixhash chains of TOKEN
// streams, so sharing is a shared token prefix and every key commits to the blocks before
// it. (Synthetic key lists cannot express that commitment, and a test built from them can
// "share" a block at two different positions — which no real chain can.)
func TestPrefixClosureAndSearchCost(t *testing.T) {
	idx := newIdx(t)
	rng := rand.New(rand.NewSource(42))
	ids := []string{"s0", "s1", "s2", "s3", "s4"}
	// three shared prompt prefixes of different lengths: every branch extends one of them
	// with its own tail, which is the fork/growth shape the engine dedups
	shared := [][]uint32{}
	for i := 0; i < 3; i++ {
		shared = append(shared, tokenRun(1+3*i, 40+i*17))
	}
	own := func(n int) []uint32 { return tokenRun(500+rng.Intn(50), n) }
	tokens := map[string][]uint32{}
	chains := map[string][]Chain{}

	for step := 0; step < 3000; step++ {
		id := ids[rng.Intn(len(ids))]
		switch rng.Intn(6) {
		case 0, 1: // (re)seat: a shared prefix + a private tail
			toks := append(append([]uint32{}, shared[rng.Intn(len(shared))]...), own(1+rng.Intn(90))...)
			c := chainsOfTokens(toks, 64)
			idx.mu.Lock()
			_, already := idx.seats[id]
			idx.mu.Unlock()
			if already {
				continue
			}
			want := bruteCost(idx, c) // measured BEFORE the seat changes the index
			ok, cost := idx.TrySeatLocked(id, c, 1+rng.Intn(40))
			if !ok {
				continue
			}
			if cost != want {
				t.Fatalf("step %d: seat cost %d != brute force %d", step, cost, want)
			}
			tokens[id], chains[id] = toks, c
		case 2: // grow: the branch's next turn continues its own prompt
			if _, seated := chains[id]; !seated {
				continue
			}
			tokens[id] = append(tokens[id], own(1+rng.Intn(70))...)
			chains[id] = chainsOfTokens(tokens[id], 64)
			idx.GrowLocked(id, chains[id])
		case 3:
			idx.MarkExitingLocked(id)
		case 4:
			idx.UnseatLocked(id)
			delete(tokens, id)
			delete(chains, id)
		case 5: // ask for a cost without changing anything
			if c, ok := chains[id]; ok {
				if got, want := idx.CostLocked(c), bruteCost(idx, c); got != want {
					t.Fatalf("step %d: cost %d != brute force %d", step, got, want)
				}
			}
		}
		idx.mu.Lock()
		idx.checkInvariantLocked()
		for _, s := range idx.seats {
			for _, c := range s.chains {
				// prefix closure: once a block is missing, nothing after it may be held
				missing := false
				for _, h := range c.Keys {
					_, held := idx.blocks[h]
					if !held {
						missing = true
						continue
					}
					if missing {
						t.Fatalf("step %d: residency is not prefix-closed (chain of %d blocks)", step, len(c.Keys))
					}
				}
			}
		}
		idx.mu.Unlock()
	}
}

// tokenRun is a deterministic run of n distinct-ish token ids from seed.
func tokenRun(seed, n int) []uint32 {
	out := make([]uint32, n)
	v := uint32(seed)*2654435761 + 1
	for i := range out {
		v = v*1664525 + 1013904223
		out[i] = v%50000 + 1
	}
	return out
}

// bruteCost is the cost without the prefix argument: every chain block the engine does
// not hold.
func bruteCost(idx *BlockIndex, chains []Chain) int {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	seen := map[Hash]struct{}{}
	for _, c := range chains {
		for _, h := range c.Keys {
			if _, held := idx.blocks[h]; held {
				continue
			}
			seen[h] = struct{}{} // the union is DEDUPLICATED: cost == charge
		}
	}
	return len(seen)
}

// The index is called under the scheduler's lock, but its own methods are safe too: a
// concurrent mix of seat/grow/exit must keep both refcounts consistent.
func TestConcurrentSharedPrefixes(t *testing.T) {
	idx := newIdx(t)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			id := fmt.Sprintf("s%d", w)
			for i := 0; i < 40; i++ {
				c := chainWithPrefix("shared", 2, fmt.Sprintf("own%d", w), 1+i%3)
				idx.mu.Lock()
				idx.TrySeatLocked(id, chainsClone(c), 1000)
				idx.MarkExitingLocked(id)
				idx.GrowLocked(id, chainsClone(c))
				if i%5 == 0 {
					idx.UnseatLocked(id)
				}
				idx.mu.Unlock()
				idx.UniqueBlocks()
			}
		}(w)
	}
	wg.Wait()
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

// chainsClone copies a chain set (the concurrency test mutates shared slices).
func chainsClone(c []Chain) []Chain {
	out := make([]Chain, len(c))
	for i := range c {
		out[i] = Chain{Keys: append([]Hash(nil), c[i].Keys...), Tokens: c[i].Tokens}
	}
	return out
}

// THE GROWTH'S SAME-SHAPE TRAP, pinned: a successor chain of the SAME length and token
// count but DIFFERENT content is a different chain, and the index must charge it. Branch
// identity's tolerance allows exactly this (a continuation may differ in its last full
// block plus the partial block), so a shape-only check kept the OLD keys charged — the next
// arrival extending the old chain was costed 0 extra (over-admission) and a fork of the new
// one was costed fully cold (under-utilisation).
func TestGrowChargesAnEqualShapeChainWithDifferentContent(t *testing.T) {
	idx := newIdx(t)
	old := []Chain{{Keys: []Hash{1, 2}, Tokens: 128}}
	mustSeat(t, idx, "b", old, 1000)
	if !idx.HasNELocked(2) {
		t.Fatal("fixture: the seated chain's second block must be charged")
	}
	// same length, same token count, different tip: the branch's newest chain
	grown := []Chain{{Keys: []Hash{1, 99}, Tokens: 128}}
	idx.GrowLocked("b", grown)
	if idx.HasNELocked(2) {
		t.Fatal("the replaced tail block is still charged: an equal-shape chain was skipped")
	}
	if !idx.HasNELocked(99) {
		t.Fatal("the new tail block is not charged: the index keeps modelling the OLD chain")
	}
	if !idx.HasNELocked(1) {
		t.Fatal("the shared prefix block was released by a growth that kept it")
	}
	if got, want := idx.UniqueBlocksLocked(), 2; got != want {
		t.Fatalf("used side %d, want %d (one shared prefix + the new tail)", got, want)
	}
	// and the SAME chain again is still a no-op (the 1 kHz Pick's common case)
	before := idx.UniqueBlocksLocked()
	idx.GrowLocked("b", grown)
	if got := idx.UniqueBlocksLocked(); got != before {
		t.Fatalf("re-growing the same chain moved the used side: %d -> %d", before, got)
	}
}

// COST == CHARGE, the case that used to disagree: a multi-prompt request whose prompts are
// byte-identical. The charge deduplicated the union; the cost summed per chain, so the fit
// test saw two prompts' worth of blocks where the seat took one.
func TestCostEqualsTheChargeForIdenticalPrompts(t *testing.T) {
	idx := newIdx(t)
	one := []Chain{{Keys: []Hash{7, 8}, Tokens: 128}}
	two := []Chain{{Keys: []Hash{7, 8}, Tokens: 128}, {Keys: []Hash{7, 8}, Tokens: 128}}
	if got, want := idx.CostLocked(two), idx.CostLocked(one); got != want {
		t.Fatalf("two identical prompts cost %d blocks, one costs %d: the cost must be the deduplicated union", got, want)
	}
	cost := mustSeat(t, idx, "m", two, 1000)
	if got := idx.UniqueBlocksLocked(); got != cost {
		t.Fatalf("the seat charged %d blocks but the cost said %d: cost != charge (R1's claim)", got, cost)
	}
	// and a half-resident duplicate: one shared block, one new
	if got := idx.CostLocked([]Chain{{Keys: []Hash{7}, Tokens: 64}, {Keys: []Hash{7}, Tokens: 64}}); got != 0 {
		t.Fatalf("a fully resident prompt repeated is %d blocks, want 0", got)
	}
}
