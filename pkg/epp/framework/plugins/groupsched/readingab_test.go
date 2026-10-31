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

// THE READING A / READING B REPLAY (one replay, the root-cause doc's §13 experiment).
//
// THE QUESTION. The multi-pod wave's core arms preempted 13 times against FCFS's 5 on
// an engine both arms keep at the capacity edge. Two readings of the artifacts survived:
//
//	Reading A  the SEAT INDEX inherits the branch-identity layer's byte-derived block
//	           count (bytes/256), which measures 0.895 of the true token count
//	           (n = 1,206 real requests, p05 0.848 / p95 0.909) — so the budget is
//	           ~10.5 % short every sample and the engine is oversubscribed continuously.
//	Reading B  the index is token-exact; the 0.895 bias belongs to a different layer.
//
// THE REPLAY. Real prompt-token counts (the empirical distribution of the wave's own
// 2,450 records), the measured byte ratio, and the engine the arms ran on (40,183
// blocks of 64 tokens). For every request it drives the REAL paths — Pick seats, the
// token producer's token IDs become the tier-A keys (HashTokens retaining the trailing
// partial block), PreRequest lands them — and then reads back three numbers for the
// same request:
//
//	blocks_in_index  what the member is CHARGED (its set in the engine's BlockIndex)
//	bytes/256        what a byte-derived counter would say
//	entry estimate   what a QUEUED head is charged at admission (IndexSizer.Cost)
//
// THE ANSWER (measured, printed by the test): blocks_in_index is token-exact —
// ceil(tokens / 64) for every request, ratio 1.000, no bias — so READING B HOLDS for
// the budget's used side. The 0.895 bias is real but it lives in the ADMISSION
// ESTIMATE, which cannot be token-exact: at Pick time the tokenizer has not run for
// that request yet (the data producers run AFTER the queue is released), so the only
// available quantity is the wire size. That estimate carries the byte bias for at most
// the Pick->PreRequest window and is then replaced by the exact keys (D13).
//
// THE MEASURED BYTE RATIO. The block-count ratio 0.895 inverted through the estimator
// is 0.895 x 256 / 64 = 3.58 bytes per token, and that is the value used here. (The
// root-cause doc's prose says "~4.47 bytes/token", which would make a bytes/4 count
// read HIGH; the measurement it reports is 0.895, i.e. LOW. The measurement is what the
// replay uses.)
//
// PROVENANCE of promptSizes: records/profile_export.jsonl of results/exp-ops/mp-q4-c64
// (2,450 rows), input_sequence_length, sampled at every 5th percentile:
//
//	tar xzOf records/*.tar.gz --wildcards '*/profile_export.jsonl' \
//	  | python3 -c 'import sys,json;v=sorted(int(json.loads(l)["metrics"]["input_sequence_length"]["value"]) for l in sys.stdin);n=len(v);print([v[min(n-1,round(p/100*(n-1)))] for p in range(0,101,5)])'

import (
	"fmt"
	"sort"
	"testing"
	"time"
)

// promptSizeQuantiles is the wave's own prompt-token distribution (p0..p100, step 5);
// mean of the full set is 74,493 tokens, p50 71,489, max 209,035.
var promptSizeQuantiles = []int{
	264, 2954, 20936, 31358, 41983, 46500, 50930, 56435, 62381, 66565, 71487,
	75933, 81134, 86547, 91884, 98033, 104611, 115559, 131899, 157181, 209035,
}

const (
	// readingABByteRatio is the measured corpus ratio: the branch-identity block count
	// (bytes/256) read 0.895 of the true token count, i.e. 0.895 x 256 / 64 bytes per
	// token.
	readingABByteRatio = 0.895 * 256.0 / 64.0
	readingABBlocks    = 40183 // the campaign engine's num_gpu_blocks
)

func pct(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	i := int(p / 100 * float64(len(s)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

func TestSeatIndexBlockCountReadings(t *testing.T) {
	s, err := NewScheduler(defaultConfig()) // fraction 0.95, blockSize 64
	if err != nil {
		t.Fatal(err)
	}
	const engine = "engine-a"
	if err := s.SetNumBlocksFor(engine, 64, readingABBlocks); err != nil {
		t.Fatal(err)
	}
	g := s.groups[engine]
	now := time.Unix(1793000000, 0).UTC()
	sizer := IndexSizer{Fraction: 0.95, BlockSize: 64}

	var inIndex, byteDerived, estimate []float64
	var exact, partial int
	const n = 420 // 20 draws per quantile bucket
	for i := 0; i < n; i++ {
		tokens := promptSizeQuantiles[(i*7)%len(promptSizeQuantiles)]
		sizeBytes := int64(float64(tokens)*readingABByteRatio + 0.5)
		member := fmt.Sprintf("m%03d", i)
		reqID := fmt.Sprintf("r%03d", i)
		now = now.Add(time.Millisecond)

		// the ADMISSION estimate: what a queued head is charged before its exact keys
		// exist (this is the only quantity available at Pick time).
		estBlocks := sizer.Cost(g.index, member, sizeBytes)

		// Pick: the head is seated (the engine has room; prefill is acked below).
		h := Head{Member: member, ReqID: reqID, Arrival: now.Add(-time.Second), SizeBytes: sizeBytes}
		if got := s.PickFromHeads([]Head{h}, now); got != 0 {
			t.Fatalf("request %d: head not seated (pick %d)", i, got)
		}
		// PreRequest: the token producer's token IDs become the tier-A keys.
		ids := make([]uint32, tokens)
		for j := range ids {
			ids[j] = uint32(j % 50257)
		}
		keys := s.HashTokens(ids, "", 64)
		s.DispatchedOn(member, reqID, engine, keys, sizeBytes, now)

		charged := int64(len(g.index.SetBlocks(member)))
		trueBlocks := float64((tokens + 63) / 64) // the engine's own allocation (tail block included)
		if charged != int64(trueBlocks) {
			t.Fatalf("request %d: index charged %d blocks for %d tokens, want %d (token-exact)",
				i, charged, tokens, int64(trueBlocks))
		}
		if tokens%64 != 0 {
			partial++
		}
		inIndex = append(inIndex, float64(charged)/trueBlocks)
		byteDerived = append(byteDerived, float64((sizeBytes+255)/256)/trueBlocks)
		estimate = append(estimate, float64(estBlocks)/trueBlocks)
		exact++

		// the request finishes: prefill acked, then it leaves (the replay measures
		// per-request counts; the budget's own accumulation is the goldens' business).
		s.ChunkAt(member, now)
		s.Settle(member, reqID, struct{ Prompt, Cached int }{tokens, 0}, true, now)
		s.mu.Lock()
		g.index.MarkExitingLocked(member)
		s.mu.Unlock()
		s.Settle(member, "", struct{ Prompt, Cached int }{0, 0}, true, now.Add(time.Millisecond))
	}
	s.mu.Lock()
	left := len(g.members)
	s.mu.Unlock()
	if left != 0 {
		t.Fatalf("%d members still seated at the end of the replay", left)
	}

	t.Logf("READING A/B REPLAY — n=%d requests, %d with a partial tail block, engine %d blocks x 64 tokens, %.2f B/token",
		n, partial, readingABBlocks, readingABByteRatio)
	t.Logf("  blocks charged in the INDEX      / true blocks: p05 %.3f  p50 %.3f  p95 %.3f   (1.000 = token-exact)",
		pct(inIndex, 5), pct(inIndex, 50), pct(inIndex, 95))
	t.Logf("  bytes/256 (the byte-derived count) / true blocks: p05 %.3f  p50 %.3f  p95 %.3f",
		pct(byteDerived, 5), pct(byteDerived, 50), pct(byteDerived, 95))
	t.Logf("  the ADMISSION estimate (at Pick)   / true blocks: p05 %.3f  p50 %.3f  p95 %.3f",
		pct(estimate, 5), pct(estimate, 50), pct(estimate, 95))

	// VERDICT, asserted so it cannot drift: the seat index is TOKEN-EXACT (Reading B),
	// and the 0.895 byte bias is confined to the admission estimate.
	for i, r := range inIndex {
		if r != 1 {
			t.Fatalf("request %d: the index charged %.4f of the true block count — the used side must be token-exact", i, r)
		}
	}
	if m := pct(estimate, 50); m > 0.92 || m < 0.87 {
		t.Fatalf("the admission estimate reads %.3f of the true block count at p50: the replay's premise (a ~0.895 byte-derived estimate) is gone", m)
	}
	t.Logf("  VERDICT: Reading B — the budget's USED side is token-exact; the %.3f byte bias is the ENTRY ESTIMATE's, which is structural (no tokens exist at Pick time)",
		pct(estimate, 50))
}

// THE TRAILING PARTIAL BLOCK (fix 7), at the key counter and at the estimate. Both are
// the same defect in two places: a partial block was dropped where the engine allocates
// it, and both directions cost headroom. Runs at N=1 and N=2 because the estimate is
// per engine (the delta is costed against ONE engine's index).
func TestPartialBlockIsCountedInKeysAndEstimate(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		s := fx.s
		ids := make([]uint32, 65)
		for i := range ids {
			ids[i] = uint32(i)
		}
		if got := len(s.HashTokens(ids, "", 64)); got != 2 {
			t.Fatalf("65 tokens at 64/block: %d keys, want 2 (the tail block is allocated)", got)
		}
		if got := len(s.HashTokens(ids[:64], "", 64)); got != 1 {
			t.Fatalf("64 tokens at 64/block: %d keys, want 1", got)
		}
		if got := len(s.HashTokens(nil, "", 64)); got != 0 {
			t.Fatalf("empty body: %d keys, want 0 (no phantom block)", got)
		}
		// the estimate: a delta that is not a whole number of blocks rounds UP
		g := fx.g()
		sizer := IndexSizer{Fraction: 0.95, BlockSize: 64}
		if got := sizer.Cost(g.index, "fresh", 256); got != 1 {
			t.Fatalf("a 256-byte head (exactly one block): estimate %d, want 1", got)
		}
		if got := sizer.Cost(g.index, "fresh", 257); got != 2 {
			t.Fatalf("a 257-byte head (one block + 1 byte): estimate %d, want 2 — a partial block occupies a block", got)
		}
		if got := sizer.Cost(g.index, "fresh", 0); got != 0 {
			t.Fatalf("an empty head: estimate %d, want 0", got)
		}
	})
}
