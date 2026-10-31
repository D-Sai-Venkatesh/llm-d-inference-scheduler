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

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"testing"
)

// newIdx builds an index at the campaign's measured block size (64
// tokens — the review's S0-3: 40,183 blocks x 64 = 2.57M).
func newIdx(t *testing.T) *BlockIndex {
	idx, err := NewBlockIndex(64)
	if err != nil {
		t.Fatalf("NewBlockIndex: %v", err)
	}
	return idx
}

// body builds a body from aligned chunks with a per-chunk distinct
// filler so within-body dedup cannot hide counting errors.
func body(sysChunks int, privChunks int, base rune) []byte {
	var b strings.Builder
	for i := 0; i < sysChunks; i++ {
		b.WriteString(strings.Repeat("S", 256)) // 64 tokens x 4 B = 256 B chunks
	}
	for i := 0; i < privChunks; i++ {
		b.WriteString(strings.Repeat(string(base+rune(i%26)), 256))
	}
	return []byte(b.String())
}

func mustSeat(t *testing.T, idx *BlockIndex, id string, b []byte, budget int) (cost int) {
	idx.Track(id, 0)
	if err := idx.UpdateWaiting(id, "r", b); err != nil {
		t.Fatalf("UpdateWaiting: %v", err)
	}
	ok, cost := idx.TrySeat(id, budget)
	if !ok {
		t.Fatalf("TrySeat(%s) rejected", id)
	}
	return cost
}

// ---- S0-3: units ----

func TestFailClosedOnUnknownBlockSize(t *testing.T) {
	if _, err := NewBlockIndex(0); err == nil {
		t.Fatal("block size 0 must fail closed")
	}
	if _, err := NewBlockIndex(-1); err == nil {
		t.Fatal("negative block size must fail closed")
	}
	for _, bs := range []int{16, 64, 256} {
		idx, err := NewBlockIndex(bs)
		if err != nil {
			t.Fatalf("bs=%d: %v", bs, err)
		}
		want := bs * 4
		if bs*4 < minChunkBytes {
			want = minChunkBytes
		}
		if idx.ChunkBytes() != want {
			t.Fatalf("bs=%d chunkBytes=%d want %d", bs, idx.ChunkBytes(), want)
		}
		// one chunk per engine block (subject to the 64-byte floor)
		wantChunks := (256 + idx.ChunkBytes() - 1) / idx.ChunkBytes()
		if n := len(idx.HashBody(make([]byte, 256))); n != wantChunks {
			t.Fatalf("bs=%d: 256B body -> %d chunks, want %d", bs, n, wantChunks)
		}
		if idx.EstimateTokens(40183) != int64(40183*bs) {
			t.Fatalf("token estimate wrong at bs=%d", bs)
		}
	}
}

// ---- S0-2: chained hashes ----

func TestChainedHashPositionSensitive(t *testing.T) {
	idx := newIdx(t)
	sysA := body(2, 0, 'A') // "SSSS..." x2
	sysB := body(1, 1, 'Z') // "SSS..." + "ZZZ..." — same total, different structure
	// the same private chunk after different prefixes must NOT match
	tailX := []byte(strings.Repeat("X", 256))
	h1 := idx.HashBody(append(append([]byte{}, sysA...), tailX...))
	h2 := idx.HashBody(append(append([]byte{}, sysB...), tailX...))
	if sameHashes(h1[2:], h2[2:]) {
		t.Fatal("identical content after divergent prefixes must hash differently (chaining)")
	}
}

func TestChainedHashNoWithinBodyDedup(t *testing.T) {
	idx := newIdx(t)
	// a body of 5 IDENTICAL chunks must count as 5 blocks, not 1
	rep := strings.Repeat(strings.Repeat("R", 256), 5)
	if n := len(idx.HashBody([]byte(rep))); n != 5 {
		t.Fatalf("repeated chunks: %d blocks, want 5 (no within-body dedup)", n)
	}
}

func TestChainedHashSharedPrefixShares(t *testing.T) {
	idx := newIdx(t)
	a := idx.HashBody(body(3, 1, 'A'))
	b := idx.HashBody(body(3, 2, 'N'))
	shared := 0
	for i := 0; i < 3 && i < len(a) && i < len(b); i++ {
		if a[i] == b[i] {
			shared++
		} else {
			break
		}
	}
	if shared != 3 {
		t.Fatalf("aligned shared prefix: %d shared chunks, want 3", shared)
	}
}

func TestEmptyBodyNoPhantomBlock(t *testing.T) {
	idx := newIdx(t)
	if n := len(idx.HashBody(nil)); n != 0 {
		t.Fatalf("empty body: %d chunks, want 0 (no phantom shared block)", n)
	}
}

// ---- S0-1: exiting state (the review's R1/R2/R3, restated for the charged-drain rule) ----

func TestExitingUnseatDoesNotStealNE(t *testing.T) {
	// R1: a and b share 3 chunks; MarkExiting(a) then Unseat(a) must
	// leave b's budget contribution intact.
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(3, 1, 'A'), 1000)
	mustSeat(t, idx, "b", body(3, 2, 'N'), 1000)
	neBoth := idx.UniqueBlocksNE() // 3 shared + 1 + 2 = 6
	if neBoth != 6 {
		t.Fatalf("NE both: %d want 6", neBoth)
	}
	idx.MarkExiting("a")
	idx.Unseat("a")
	if got := idx.UniqueBlocksNE(); got != 5 { // 3 shared + 2 private to b
		t.Fatalf("R1: NE after exiting-unseat: %d want 5 (stolen from b!)", got)
	}
}

// An exiting member's blocks STAY on the used side until it drains: mark the
// member, and the budget still pays for its blocks (the engine pins them until the
// running request finishes). Releasing them at the mark was the accounting bug the
// multi-pod wave measured at 14-18 % of the pool.
func TestExitingMemberStaysChargedUntilUnseat(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(3, 1, 'A'), 1000)
	before := idx.UniqueBlocksNE()
	idx.MarkExiting("a")
	if got := idx.UniqueBlocksNE(); got != before {
		t.Fatalf("NE after MarkExiting: %d want %d (the blocks are still held)", got, before)
	}
	idx.Unseat("a")
	if got := idx.UniqueBlocksNE(); got != 0 {
		t.Fatalf("NE after the drain (Unseat): %d want 0", got)
	}
	idx.checkInvariantLocked()
}

// MarkExiting changes no counter, so repeating it cannot double-release anything.
func TestMarkExitingIdempotent(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(3, 1, 'A'), 1000)
	ne := idx.UniqueBlocksNE()
	idx.MarkExiting("a")
	idx.MarkExiting("a")
	idx.MarkExiting("a")
	if got := idx.UniqueBlocksNE(); got != ne {
		t.Fatalf("R2: NE after triple MarkExiting: %d want %d", got, ne)
	}
	if got := idx.UniqueBlocks(); got != 4 {
		t.Fatalf("still seated+counted: %d want 4", got)
	}
	idx.checkInvariantLocked()
}

// An exiting member's growth is pinned by the engine like anyone else's, so it is
// charged: the retiree's next turn is real KV, and the budget must see it.
func TestExitingMemberGrowthStaysCharged(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(3, 1, 'A'), 1000)
	mustSeat(t, idx, "b", body(3, 2, 'N'), 1000)
	idx.MarkExiting("a")
	before := idx.UniqueBlocksNE()
	// a grows while exiting: the new blocks enter both counters
	if err := idx.UpdateWaiting("a", "r", body(0, 2, 'Q')); err != nil {
		t.Fatal(err)
	}
	if err := idx.SettleDispatched("a", "r", body(0, 2, 'Q')); err != nil {
		t.Fatal(err)
	}
	if got := idx.UniqueBlocksNE(); got != idx.UniqueBlocks() || got <= before-2 {
		t.Fatalf("R3: NE after exiting member growth: %d (union %d) want the exiting member fully charged", got, idx.UniqueBlocks())
	}
	idx.checkInvariantLocked()
}

// ---- seating semantics ----

func TestTrySeatAtomicBudget(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(2, 2, 'A'), 100) // 4 blocks
	// budget exactly 6: b shares 2, adds 2 private -> total 6, fits
	if err := idx.UpdateWaiting("b", "r", body(2, 2, 'B')); err != nil {
		t.Fatal(err)
	}
	ok, cost := idx.TrySeat("b", 6)
	if !ok || cost != 2 {
		t.Fatalf("seat b: ok=%v cost=%d want true/2", ok, cost)
	}
	// c would exceed
	if err := idx.UpdateWaiting("c", "r", body(0, 5, 'C')); err != nil {
		t.Fatal(err)
	}
	ok, cost = idx.TrySeat("c", 6)
	if ok || cost != 5 {
		t.Fatalf("seat c: ok=%v cost=%d want false/5", ok, cost)
	}
	if idx.Seated("c") {
		t.Fatal("c must not be seated on rejection")
	}
}

func TestSeatWithSetNoPreSettle(t *testing.T) {
	// R6: the seat request must NOT be pre-settled; settling the same
	// body later must occupy exactly one window slot.
	idx := newIdx(t)
	bd := body(2, 1, 'A')
	mustSeat(t, idx, "a", bd, 1000)
	if err := idx.SettleDispatched("a", "r", bd); err != nil {
		t.Fatal(err)
	}
	s := idx.sessions["a"]
	if len(s.dispHist) != 1 {
		t.Fatalf("R6: dispatch history after one settle: %d want 1 (no pre-settle)", len(s.dispHist))
	}
	// settling again (same body, second turn) is a second real dispatch
	if err := idx.SettleDispatched("a", "r", bd); err != nil {
		t.Fatal(err)
	}
	if len(s.dispHist) != 2 {
		t.Fatalf("second settle: %d want 2", len(s.dispHist))
	}
}

func TestRecordPersistsAcrossResidencies(t *testing.T) {
	idx := newIdx(t)
	bd := body(2, 2, 'A')
	mustSeat(t, idx, "a", bd, 1000)
	if err := idx.SettleDispatched("a", "r", bd); err != nil {
		t.Fatal(err)
	}
	idx.Unseat("a")
	// re-seat: the session's history still counts (multi-chain re-entry)
	if err := idx.UpdateWaiting("a", "r", body(0, 1, 'Z')); err != nil {
		t.Fatal(err)
	}
	ok, cost := idx.TrySeat("a", 1000)
	if !ok {
		t.Fatal("re-seat rejected")
	}
	// 4 original blocks + 1 new private = 5 in the all-seated counter
	if got := idx.UniqueBlocks(); got != 5 {
		t.Fatalf("re-entry counts history: %d want 5", got)
	}
	_ = cost
}

func TestSettleWindowSlideReleases(t *testing.T) {
	idx := newIdx(t)
	// five divergent turns: the 4-window slides, the first turn's private
	// blocks leave the set and the counters
	mustSeat(t, idx, "a", body(2, 1, 'A'), 1000)
	if err := idx.SettleDispatched("a", "r", body(2, 1, 'A')); err != nil {
		t.Fatal(err) // settle the entry request so the window arithmetic is clean
	}
	peak := 0
	for i := 0; i < 5; i++ {
		bd := body(2, 1, rune('A'+i))
		if err := idx.UpdateWaiting("a", fmt.Sprintf("w%d", i), bd); err != nil {
			t.Fatal(err)
		}
		if err := idx.SettleDispatched("a", fmt.Sprintf("w%d", i), bd); err != nil {
			t.Fatal(err)
		}
		if u := idx.UniqueBlocks(); u > peak {
			peak = u
		}
	}
	// window = 4 bodies x 1 private + 2 shared = 6 max; first body's
	// private block must be released by the slide
	if got := idx.UniqueBlocks(); got > 6 {
		t.Fatalf("window slide did not release: %d > 6", got)
	}
}

func TestReplaceWaitingCancelAndMutation(t *testing.T) {
	idx := newIdx(t)
	orig := body(2, 2, 'A')
	if err := idx.UpdateWaiting("a", "r", orig); err != nil {
		t.Fatal(err)
	}
	ok, _ := idx.TrySeat("a", 1000)
	if !ok {
		t.Fatal("seat")
	}
	before := idx.UniqueBlocksNE()
	// cancel: the waiting request's blocks leave the budget
	if err := idx.Cancel("a", "r"); err != nil {
		t.Fatal(err)
	}
	if got := idx.UniqueBlocksNE(); got != 0 {
		t.Fatalf("cancel: NE %d want 0 (was %d)", got, before)
	}
	// mutation: swap in a new body
	if err := idx.UpdateWaiting("a", "r", orig); err != nil {
		t.Fatal(err)
	}
	mutated := body(2, 3, 'M')
	if err := idx.UpdateWaiting("a", "r", mutated); err != nil {
		t.Fatal(err)
	}
	if got := idx.UniqueBlocksNE(); got != 5 {
		t.Fatalf("mutation: NE %d want 5", got)
	}
	// settling the MUTATED body works (no orphan waiting entry)
	if err := idx.SettleDispatched("a", "r", mutated); err != nil {
		t.Fatal(err)
	}
	if n := len(idx.sessions["a"].waiting); n != 0 {
		t.Fatalf("orphan waiting entries: %d", n)
	}
}

func TestDoubleUnseatIsSafeNoOp(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", body(2, 1, 'A'), 1000)
	idx.Unseat("a")
	after := idx.UniqueBlocks()
	idx.Unseat("a") // the seated-set guard makes this a no-op, not an underflow
	if idx.UniqueBlocks() != after {
		t.Fatalf("double unseat changed counters: %d -> %d", after, idx.UniqueBlocks())
	}
}

// ---- the invariant + randomized property test (the review's demand) ----

func TestRandomizedPropertyInvariants(t *testing.T) {
	idx := newIdx(t)
	rng := rand.New(rand.NewSource(42))
	alphabet := "ABCDEFGHIJKLMNOP"
	state := map[string][]byte{}
	ids := []string{"s0", "s1", "s2", "s3"}
	for step := 0; step < 2000; step++ {
		id := ids[rng.Intn(len(ids))]
		switch rng.Intn(8) {
		case 0, 1: // new waiting body (shared prefix + random tail)
			sysN := rng.Intn(3)
			privN := 1 + rng.Intn(3)
			var b []byte
			b = append(b, body(sysN, 0, 'S')...)
			b = append(b, body(0, privN, rune(alphabet[rng.Intn(len(alphabet))]))...)
			_ = idx.UpdateWaiting(id, "r", b)
			state[id] = b
		case 2:
			_ = idx.UpdateWaiting(id, "r", state[id])
			ok, _ := idx.TrySeat(id, 3+rng.Intn(20))
			_ = ok
		case 3:
			idx.Unseat(id)
		case 4:
			idx.MarkExiting(id)
		case 5:
			if len(state[id]) > 0 {
				_ = idx.SettleDispatched(id, "r", state[id])
			}
		case 6:
			idx.Forget(id)
			delete(state, id)
		case 7:
			idx.FitBlocks(idx.HashBody(state[id]))
		}
		idx.mu.Lock()
		idx.checkInvariantLocked()
		idx.mu.Unlock()
	}
}

func TestConcurrentSharedPrefixes(t *testing.T) {
	idx := newIdx(t)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			id := fmt.Sprintf("s%d", w)
			for i := 0; i < 40; i++ {
				bd := append(body(2, 0, 'S'), body(0, 1+rngInt(3), rune('A'+w))...)
				_ = idx.UpdateWaiting(id, "r", bd)
				idx.TrySeat(id, 1000)
				_ = idx.SettleDispatched(id, "r", bd)
				if i%7 == 0 {
					idx.MarkExiting(id)
				}
				if i%5 == 0 {
					idx.Unseat(id)
				}
				idx.UniqueBlocksNE()
			}
		}(w)
	}
	wg.Wait()
	idx.mu.Lock()
	idx.checkInvariantLocked()
	idx.mu.Unlock()
}

var rngMu sync.Mutex
var prng = rand.New(rand.NewSource(7))

func rngInt(n int) int {
	rngMu.Lock()
	defer rngMu.Unlock()
	return prng.Intn(n)
}
