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

// The reviewer's 8 repro scenarios (review-groupsched/review_repro_test.go),
// ported to the post-fix API. Each R-number is the original scenario; the
// assertions are the reviewer's. Run at 256-token blocks so the original
// 1024-byte body arithmetic carries over.

import (
	"fmt"
	"strings"
	"testing"
)

// privChunks returns n chunks of distinct content (letter base+i per chunk).
func privChunks(n, chunkBytes int, base rune) []byte {
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteString(strings.Repeat(string(base+rune(i%26)), chunkBytes))
	}
	return []byte(b.String())
}

func reproIdx(t *testing.T) *BlockIndex {
	idx, err := NewBlockIndex(256) // 1024-byte chunks, as the original repro assumed
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func reproSeat(t *testing.T, idx *BlockIndex, id string, body []byte) {
	if err := idx.UpdateWaiting(id, "r", body); err != nil {
		t.Fatal(err)
	}
	if ok, _ := idx.TrySeat(id, 1<<20); !ok {
		t.Fatalf("seat %s rejected", id)
	}
}

func reproAligned(s string) []byte {
	b := []byte(s)
	for len(b)%1024 != 0 {
		b = append(b, ' ')
	}
	return b
}

// R1: MarkExiting then Unseat must not steal the OTHER member's non-exiting
// refcount on shared blocks.
func TestReproNEStealOnUnseat(t *testing.T) {
	sys := reproAligned(strings.Repeat("common ", 300)) // 3 shared chunks
	ha := append(append([]byte{}, sys...), privChunks(1, 1024, 'A')...)
	hb := append(append([]byte{}, sys...), privChunks(1, 1024, 'N')...)
	idx := reproIdx(t)
	reproSeat(t, idx, "a", ha)
	reproSeat(t, idx, "b", hb)
	idx.MarkExiting("a")
	ne := idx.UniqueBlocksNE() // b alone: 3 shared + 1 private = 4
	if ne != 4 {
		t.Fatalf("NE before unseat: %d want 4", ne)
	}
	idx.Unseat("a")
	if idx.UniqueBlocksNE() != 4 {
		t.Errorf("R1: unseating an exiting member removed b's shared blocks from NE: %d", idx.UniqueBlocksNE())
	}
}

// R2: MarkExiting twice must be idempotent.
func TestReproMarkExitingTwice(t *testing.T) {
	sys := reproAligned(strings.Repeat("common ", 300))
	ha := append(append([]byte{}, sys...), privChunks(1, 1024, 'A')...)
	hb := append(append([]byte{}, sys...), privChunks(1, 1024, 'N')...)
	idx := reproIdx(t)
	reproSeat(t, idx, "a", ha)
	reproSeat(t, idx, "b", hb)
	idx.MarkExiting("a")
	idx.MarkExiting("a")
	if idx.UniqueBlocksNE() != 4 {
		t.Errorf("R2: second MarkExiting(a) stole b's NE refcounts: NE=%d want 4", idx.UniqueBlocksNE())
	}
}

// R3: an exiting member's new blocks must not be counted into NE.
func TestReproExitingGrowsNE(t *testing.T) {
	idx := reproIdx(t)
	reproSeat(t, idx, "a", reproAligned("member a turn 1"))
	idx.MarkExiting("a")
	if err := idx.UpdateWaiting("a", "r", append(reproAligned("member a turn 1"), privChunks(3, 1024, 'Q')...)); err != nil {
		t.Fatal(err)
	}
	if idx.UniqueBlocksNE() != 0 {
		t.Errorf("R3: exiting member's new waiting blocks counted against the budget: NE=%d want 0", idx.UniqueBlocksNE())
	}
}

// R4: chained hashes — identical content after DIFFERENT prefixes must not
// count as shared (the engine's prefix cache cannot reuse it).
func TestReproUnchainedFalseSharing(t *testing.T) {
	tool := privChunks(4, 1024, 'T') // the same file contents read by two sessions
	a := append(reproAligned("system prompt of tenant ALPHA"), tool...)
	b := append(reproAligned("system prompt of tenant BETA"), tool...)
	idx := reproIdx(t)
	reproSeat(t, idx, "a", a)
	cost := idx.FitBlocks(idx.HashBody(b))
	t.Logf("admission cost of b: %d blocks (engine needs 5: nothing shareable after a divergent first block)", cost)
	if cost < 5 {
		t.Errorf("R4: false sharing: cost %d < 5", cost)
	}
}

// R5: repeated chunks within one body must each count (the engine stores each).
func TestReproWithinBodyRepetition(t *testing.T) {
	rep := []byte(strings.Repeat("x", 1024*10)) // 10 identical chunks
	idx := reproIdx(t)
	if got := idx.FitBlocks(idx.HashBody(rep)); got != 10 {
		t.Errorf("R5: 10-block body costed as %d", got)
	}
}

// R6: seating must not pre-settle the request into the dispatch window.
func TestReproDoublePush(t *testing.T) {
	idx := reproIdx(t)
	bd := reproAligned("turn one")
	reproSeat(t, idx, "a", bd)
	if err := idx.SettleDispatched("a", "r", bd); err != nil {
		t.Fatal(err)
	}
	if n := len(idx.sessions["a"].dispHist); n != 1 {
		t.Errorf("R6: the seat request occupies %d window slots", n)
	}
}

// R7: a per-turn field serialized BEFORE messages shifts every chunk — no
// cross-turn dedup. Informational: the alignment ratio is CP2's measurement.
func TestReproMisalignedTurns(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 1200; i++ {
		sb.WriteString(fmt.Sprintf("msg%05d ", i)) // ~10.8KB non-repeating
	}
	msgs := sb.String()
	t1 := []byte(`{"max_tokens":512,"messages":"` + msgs + `"}`)
	t2 := []byte(`{"max_tokens":1024,"messages":"` + msgs + `next turn"}`)
	idx := reproIdx(t)
	reproSeat(t, idx, "a", t1)
	if err := idx.SettleDispatched("a", "r", t2); err != nil {
		t.Fatal(err)
	}
	setBlocks := len(idx.sessions["a"].set())
	latestBlocks := len(idx.HashBody(t2))
	t.Logf("R7: session set after 2 turns: %d blocks for a %d-block latest prompt (ratio %.2f; CP2 measures this on real bodies)",
		setBlocks, latestBlocks, float64(setBlocks)/float64(latestBlocks))
}

// R8: settling must remove exactly ONE matching waiting entry.
func TestReproNilSettleDropsDuplicate(t *testing.T) {
	idx := reproIdx(t)
	bd := reproAligned("same body")
	if err := idx.UpdateWaiting("a", "q1", bd); err != nil {
		t.Fatal(err)
	}
	if err := idx.UpdateWaiting("a", "q2", bd); err != nil {
		t.Fatal(err)
	}
	if err := idx.SettleDispatched("a", "q1", bd); err != nil {
		t.Fatal(err)
	}
	if n := len(idx.sessions["a"].waiting); n != 1 {
		t.Errorf("R8: one settle removed %d waiting entries", 2-n)
	}
}
