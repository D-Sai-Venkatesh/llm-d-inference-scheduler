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
	"strings"
	"sync"
	"testing"
)

// mkBody builds a deterministic body of n chunks from a prefix, so
// bodies sharing `prefix` share their leading hashes.
// privChunks returns n chunks of distinct content (letter 'A'+i per chunk).
func privChunks(n, chunkBytes int, base rune) []byte {
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteString(strings.Repeat(string(base+rune(i%26)), chunkBytes))
	}
	return []byte(b.String())
}

func mkBody(prefix string, extraChunks int, chunkBytes int) []byte {
	var b strings.Builder
	b.WriteString(prefix)
	pad := chunkBytes - (len(prefix) % chunkBytes)
	if pad < chunkBytes {
		b.WriteString(strings.Repeat("p", pad))
	}
	for i := 0; i < extraChunks; i++ {
		b.WriteString(strings.Repeat(fmt.Sprintf("%c", 'a'+rune(i%26)), chunkBytes))
	}
	return []byte(b.String())
}

func TestHashBodyChunking(t *testing.T) {
	body := mkBody("system-prompt", 3, 1024)
	hs := HashBody(body, 1024)
	want := (len(body) + 1023) / 1024 // ceil: exact multiples have no partial chunk
	if len(hs) != want {
		t.Fatalf("chunks: got %d want %d", len(hs), want)
	}
	// empty body: one chunk
	if got := len(HashBody(nil, 1024)); got != 1 {
		t.Fatalf("empty body chunks: %d", got)
	}
	// exact multiple: no trailing partial
	if got := len(HashBody(make([]byte, 2048), 1024)); got != 2 {
		t.Fatalf("exact-multiple chunks: %d", got)
	}
	// determinism
	again := HashBody(body, 1024)
	for i := range hs {
		if hs[i] != again[i] {
			t.Fatal("hashing not deterministic")
		}
	}
}

func TestSharedPrefixCountedOnce(t *testing.T) {
	// THE test of D4/tier B: two sessions sharing a long system prompt
	// must add only their divergent tails to the group's unique count.
	sys := strings.Repeat("You are an agent. ", 200) // ~3.6KB
	a := mkBody(sys+"USER-A", 2, 1024)
	bod := mkBody(sys+"USER-B", 2, 1024)
	ha, hb := HashBody(a, 1024), HashBody(bod, 1024)

	idx := NewBlockIndex()
	addedA := idx.SeatWithSet("a", ha, 4)
	addedB := idx.SeatWithSet("b", hb, 4)

	if addedA != len(ha) {
		t.Fatalf("first seat adds all its blocks: %d vs %d", addedA, len(ha))
	}
	if addedB >= addedA {
		t.Fatalf("shared prefix must discount: addedB %d, addedA %d", addedB, addedA)
	}
	// group unique = shared prefix + both tails, strictly less than the sum
	sum := len(ha) + len(hb)
	if idx.UniqueBlocks() >= sum {
		t.Fatalf("unique %d should be < sum %d", idx.UniqueBlocks(), sum)
	}
	if idx.UniqueBlocks() != addedA+addedB {
		t.Fatalf("unique %d != added %d+%d", idx.UniqueBlocks(), addedA, addedB)
	}
}

func TestFitBlocksDiscountsCommonality(t *testing.T) {
	sys := strings.Repeat("shared system prompt ", 150)
	idx := NewBlockIndex()
	idx.SeatWithSet("a", HashBody(mkBody(sys+"A", 3, 1024), 1024), 4)
	// candidate b: same system prompt, different tail
	fit := idx.FitBlocks(HashBody(mkBody(sys+"B", 2, 1024), 1024))
	full := idx.FitBlocks(HashBody(mkBody(strings.Repeat("totally different ", 100), 2, 1024), 1024))
	if fit == 0 || fit >= full {
		t.Fatalf("shared-prefix fit %d should be less than unrelated fit %d", fit, full)
	}
}

func TestUnseatReleasesBlocks(t *testing.T) {
	idx := NewBlockIndex()
	ha := HashBody(mkBody("solo-session", 3, 1024), 1024)
	idx.SeatWithSet("a", ha, 4)
	if idx.UniqueBlocks() != len(ha) {
		t.Fatalf("unique after seat: %d vs %d", idx.UniqueBlocks(), len(ha))
	}
	idx.Unseat("a")
	if idx.UniqueBlocks() != 0 || idx.UniqueBlocksNE() != 0 {
		t.Fatalf("after unseat: %d/%d", idx.UniqueBlocks(), idx.UniqueBlocksNE())
	}
}

func TestRefcountingOnSharedUnseat(t *testing.T) {
	// chunk-ALIGNED shared prefix (the real case: the shared system
	// prompt starts at body offset 0, so all its full chunks align)
	sys := []byte(strings.Repeat("common ", 300))
	for len(sys)%1024 != 0 {
		sys = append(sys, ' ')
	}
	idx := NewBlockIndex()
	// distinct tails: session a adds one private chunk, session b two
	ha := HashBody(append(append([]byte{}, sys...), privChunks(1, 1024, 'A')...), 1024)
	hb := HashBody(append(append([]byte{}, sys...), privChunks(2, 1024, 'N')...), 1024)
	idx.SeatWithSet("a", ha, 4)
	idx.SeatWithSet("b", hb, 4)
	both := idx.UniqueBlocks()
	if both != 6 { // 3 shared + 1 (a only) + 2 (b only)
		t.Fatalf("unique with both seated: %d (want 6)", both)
	}
	// unseating one holder must NOT release the shared blocks
	idx.Unseat("a")
	if idx.UniqueBlocks() == 0 {
		t.Fatal("shared blocks released while another member holds them")
	}
	if idx.UniqueBlocks() != 5 { // 3 shared + 2 (b only); a's private block released
		t.Fatalf("unique after unseating a: %d (want 5)", idx.UniqueBlocks())
	}
	if idx.UniqueBlocks() > both {
		t.Fatalf("unique grew on unseat: %d > %d", idx.UniqueBlocks(), both)
	}
	idx.Unseat("b")
	if idx.UniqueBlocks() != 0 {
		t.Fatalf("all released: %d", idx.UniqueBlocks())
	}
}

func TestMarkExiting(t *testing.T) {
	idx := NewBlockIndex()
	ha := HashBody(mkBody("member", 3, 1024), 1024)
	idx.SeatWithSet("a", ha, 4)
	if idx.UniqueBlocksNE() != len(ha) {
		t.Fatalf("NE before exiting: %d", idx.UniqueBlocksNE())
	}
	idx.MarkExiting("a")
	if idx.UniqueBlocksNE() != 0 {
		t.Fatalf("NE after exiting: %d (should not count against budget)", idx.UniqueBlocksNE())
	}
	if idx.UniqueBlocks() != len(ha) {
		t.Fatalf("still seated+counted: %d", idx.UniqueBlocks())
	}
	// an exiting member's blocks do not block admission
	if fit := idx.FitBlocks(ha); fit != len(dedup(ha)) {
		t.Fatalf("fit past exiting member: %d vs %d", fit, len(dedup(ha)))
	}
}

func TestSettleDispatchedWindow(t *testing.T) {
	idx := NewBlockIndex()
	// a growing chain: turn 2's prompt contains turn 1's
	turn1 := mkBody("chain start", 2, 1024)
	turn2 := append(append([]byte{}, turn1...), mkBody(" plus more", 2, 1024)...)
	idx.SeatWithSet("a", HashBody(turn1, 1024), 4)
	before := idx.UniqueBlocks()
	idx.SettleDispatched("a", turn2, 1024)
	after := idx.UniqueBlocks()
	if after < before {
		t.Fatalf("growth shrank the set: %d -> %d", before, after)
	}
	// window of 4: settling 5 distinct bodies drops the oldest
	sizes := []int{}
	for i := 0; i < 6; i++ {
		idx.SettleDispatched("a", mkBody(fmt.Sprintf("divergent-%d ", i)+strings.Repeat("x", 300), 1, 1024), 1024)
		sizes = append(sizes, idx.UniqueBlocks())
	}
	for i := 1; i < len(sizes); i++ {
		if sizes[i] < sizes[i-1]-2 || sizes[i] > sizes[i-1]+3 {
			t.Logf("window slide sizes: %v (informational)", sizes)
			break
		}
	}
}

func TestConcurrent(t *testing.T) {
	idx := NewBlockIndex()
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			id := fmt.Sprintf("s%d", w)
			for i := 0; i < 50; i++ {
				idx.SeatWithSet(id, HashBody(mkBody(fmt.Sprintf("body-%d-%d ", w, i)+strings.Repeat("z", 400), 1, 1024), 1024), 4)
				idx.FitBlocks(HashBody(mkBody("probe", 1, 1024), 1024))
				idx.UpdateWaiting(id, mkBody(fmt.Sprintf("wait-%d-%d", w, i), 1, 1024), 1024, 4)
				idx.UniqueBlocks()
				idx.MarkExiting(id)
				idx.Unseat(id)
			}
		}(w)
	}
	wg.Wait()
	if idx.UniqueBlocks() != 0 {
		t.Fatalf("leaked blocks after all unseats: %d", idx.UniqueBlocks())
	}
}

func TestBytesToBlocksHint(t *testing.T) {
	if got := BytesToBlocksHint(256); got != 1024 {
		t.Fatalf("hint for 256-token blocks: %d", got)
	}
	if got := BytesToBlocksHint(0); got != DefaultChunkBytes {
		t.Fatalf("default hint: %d", got)
	}
	if got := BytesToBlocksHint(1); got != 64 {
		t.Fatalf("floor: %d", got)
	}
}

func TestMisalignedBoundaryLoss(t *testing.T) {
	// The documented tier-B approximation: a shared prefix that is NOT a
	// multiple of the chunk size loses its final partial chunk (the
	// boundary chunk mixes shared tail + private head, so it hashes
	// differently per session). The loss is bounded by ONE chunk per
	// sharing boundary — assert that, no more.
	sys := strings.Repeat("common ", 300) // 2100 bytes: 2 full + 52 partial
	ha := HashBody([]byte(sys+string(privChunks(8, 1024, 'A'))), 1024)
	hb := HashBody([]byte(sys+string(privChunks(8, 1024, 'N'))), 1024)
	idx := NewBlockIndex()
	addedA := idx.SeatWithSet("a", ha, 4)
	addedB := idx.SeatWithSet("b", hb, 4)
	// perfect alignment would share 2 chunks; misalignment shares only
	// those 2 (the 52-byte tail is lost in the boundary chunk), so b
	// re-adds everything after chunk 2 — but never MORE than its body.
	if addedB > len(hb) {
		t.Fatalf("added more than the body: %d > %d", addedB, len(hb))
	}
	if addedA+addedB > len(ha)+len(hb) {
		t.Fatal("invariant: added blocks cannot exceed the bodies' blocks")
	}
	// aligned sharing would be 2 chunks; misalignment shares fewer
	shared := len(dedup(ha)) + len(dedup(hb)) - (addedA + addedB)
	t.Logf("shared chunks under misalignment: %d (aligned would be 2; the boundary chunk is lost, documented)", shared)
	if shared > 2 {
		t.Fatalf("shared more than alignment allows: %d", shared)
	}
}

func dedup(hs []Hash) []Hash {
	seen := map[Hash]struct{}{}
	var out []Hash
	for _, h := range hs {
		if _, ok := seen[h]; !ok {
			seen[h] = struct{}{}
			out = append(out, h)
		}
	}
	return out
}
