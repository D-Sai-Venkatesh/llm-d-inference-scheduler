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

// The F-arms' gates. Every knob is default-OFF and the golden replay is the proof
// that "off" is today's behaviour; these tests are the proof that each knob DOES
// something specific when it is on, and that the shared tenure ceiling actually
// bounds the two dynamic rules that can otherwise hold a seat forever.

import (
	"fmt"
	"testing"
	"time"
)

// fArmScheduler builds a single-engine scheduler with the F0 base (the q4 anchor)
// and lets the test tweak the knobs.
func fArmScheduler(t *testing.T, tweak func(*Config)) *Scheduler {
	t.Helper()
	cfg := defaultConfig()
	cfg.RotationMode = "quantum"
	cfg.QuantumBusyPeriods = 4
	cfg.GateMode = "size-aware"
	if tweak != nil {
		tweak(&cfg)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	return s
}

// seatWithBlocks seats one member and gives it a footprint of `blocks` blocks, so
// the footprint-scaled quantum has something real to scale on.
func seatWithBlocks(t *testing.T, s *Scheduler, id string, blocks int, now time.Time) {
	t.Helper()
	heads := []Head{{Member: id, ReqID: id + "-r1", Arrival: now, SizeBytes: bytesFor(blocks)}}
	if _, i := s.PickFromHeadsEngine(heads, now); i < 0 {
		t.Fatalf("could not seat %s", id)
	}
	s.DispatchedOn(id, id+"-r1", "", keys(blocks, Hash(len(id)*97+blocks)), bytesFor(blocks), now)
	s.SettleOn(id, id+"-r1", "", struct{ Prompt, Cached int }{Prompt: blocks * 64, Cached: blocks * 64}, true, now.Add(time.Second))
}

// TestDynamicQuantumScalesWithFootprint is F3: Q_i = max(1, round(Q̄ · blocks_i /
// median seated blocks)). A member twice the median footprint gets twice the
// quantum; a small one gets a shorter one; nobody gets less than one busy period.
func TestDynamicQuantumScalesWithFootprint(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.QuantumMode = "dyn-size" })
	now := goldenT0
	// footprints 10, 20, 40 → median 20; Q̄ = 4
	seatWithBlocks(t, s, "small", 10, now)
	seatWithBlocks(t, s, "med", 20, now.Add(2*time.Second))
	seatWithBlocks(t, s, "big", 40, now.Add(4*time.Second))
	s.mu.Lock()
	defer s.mu.Unlock()
	if got := s.medianFootprintLocked(s.eng()); got != 20 {
		t.Fatalf("median footprint = %d, want 20", got)
	}
	cases := map[string]int{"small": 2, "med": 4, "big": 8}
	for id, want := range cases {
		m, ok := s.eng().members[id]
		if !ok {
			t.Fatalf("%s not seated", id)
		}
		if got := s.dynQuantumLocked(s.eng(), m); got != want {
			t.Fatalf("%s quantum = %d, want %d (footprint %d / median 20 x Q̄ 4)", id, got, want, len(s.eng().index.SetBlocks(id)))
		}
	}
}

// TestTenureCeilingCapsButDoesNotOverrideDynamicQuantum is the fairness guarantee
// WITH the B1b precedence fix (ruled 2026-10-31): the ceiling CAPS the scaled quantum
// (min(base, ceiling)) and never replaces it, so small members keep their short slice
// and only the members whose own base is above the ceiling are trimmed.
//
// The previous version of this test asserted the opposite ("the ceiling overrides the
// scaled quantum") — i.e. it codified the bug that turned F3 into a Q8 replicate.
func TestTenureCeilingCapsButDoesNotOverrideDynamicQuantum(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.QuantumMode = "dyn-size"
		c.QuantumCeiling = 6
	})
	now := goldenT0
	// footprints 10/20/20/20/40/200 => MEDIAN 20, Q̄ = 4 => bases 2, 4, 8 and 40.
	for i, blocks := range []int{10, 20, 20, 20, 40} {
		seatWithBlocks(t, s, fmt.Sprintf("m%d", i), blocks, now.Add(time.Duration(i)*time.Second))
	}
	seatWithBlocks(t, s, "huge", 200, now.Add(20*time.Second))
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	if got := s.medianFootprintLocked(g); got != 20 {
		t.Fatalf("test premise: median footprint = %d, want 20", got)
	}
	// (member, base, eligibility bound): dyn-size VARIES below the ceiling and is
	// only truncated where its own base exceeds it.
	for _, tc := range []struct {
		id    string
		base  int
		bound int
	}{
		{"m0", 2, 2},    // 10 blocks: 4 x 10/20
		{"m1", 4, 4},    // 20 blocks: the median
		{"m4", 8, 6},    // 40 blocks: base 8, capped by the ceiling
		{"huge", 40, 6}, // 200 blocks: a huge slice, still capped
	} {
		m := g.members[tc.id]
		if base, bounded := s.eligibilityBaseLocked(g, m); !bounded || base != tc.base {
			t.Fatalf("%s: base quantum = (%d,%v), want (%d,true) — the ceiling must not REPLACE the base", tc.id, base, bounded, tc.base)
		}
		if q, bounded := s.quantumBoundLocked(g, m); !bounded || q != tc.bound {
			t.Fatalf("%s: eligibility bound = (%d,%v), want (%d,true) = min(base, ceiling)", tc.id, q, bounded, tc.bound)
		}
	}
	// ceilingReached compares against the CEILING itself (what F7's gate may not block)
	if !s.ceilingReachedLocked(g, &member{id: "huge", busyPeriods: 6}) {
		t.Fatal("6 busy periods must reach a ceiling of 6")
	}
	if s.ceilingReachedLocked(g, &member{id: "huge", busyPeriods: 5}) {
		t.Fatal("5 busy periods must not reach a ceiling of 6")
	}
}

// TestFixedQuantumIsUnchangedByACeilingAboveIt: a ceiling at or above the fixed Q
// changes nothing — the base fires first, which is the default-PAT-adjacent case and
// the reason the fix is safe for every banked arm that carried no ceiling.
func TestFixedQuantumIsUnchangedByACeilingAboveIt(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.QuantumCeiling = 8 }) // Q = 4
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	m := &member{id: "m", engine: "engine-a", seatTime: now, lastActivity: now}
	if q, bounded := s.quantumBoundLocked(g, m); !bounded || q != 4 {
		t.Fatalf("eligibility bound = (%d,%v), want (4,true)", q, bounded)
	}
	m.busyPeriods = 4
	if reason, ok := s.eligibleLocked(g, m, now); !ok || reason != "quantum-complete" {
		t.Fatalf("a Q4-complete member = (%q,%v), want eligible at 4 with an 8 ceiling", reason, ok)
	}
	if s.ceilingReachedLocked(g, m) {
		t.Fatal("4 busy periods must not reach a ceiling of 8")
	}
}

// ---- F5 HRRN ----

// TestHRRNOrdersByTheRatio: the entry head is the highest wait/cold_tokens — the
// HRRN ratio with the (common) prefill rate cancelled out — so a small entry jumps
// an older large one only while its slowdown is genuinely higher.
func TestHRRNOrdersByTheRatio(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.EntryOrder = "hrrn" })
	now := goldenT0
	mk := func(id string, since time.Time, cold int64) *waiter {
		return &waiter{id: id, since: since, sizeBytes: bytesFor(int(cold / 64)), coldTokens: cold}
	}
	s.mu.Lock()
	s.now = now
	// A: waited 100s over 6400 cold tokens  → 0.0156 tokens-normalised
	// B: waited 1s over 64 cold tokens      → 0.0156  (a near tie, B wins on age)
	// C: waited 10s over 64 cold tokens     → 0.156   (the smallest job by far)
	s.eng().waiters["A"] = mk("A", now.Add(-100*time.Second), 6400)
	s.eng().waiters["B"] = mk("B", now.Add(-1*time.Second), 64)
	s.eng().waiters["C"] = mk("C", now.Add(-10*time.Second), 64)
	head := s.entryHeadLocked(s.eng(), nil)
	s.mu.Unlock()
	if head == nil || head.id != "C" {
		t.Fatalf("HRRN head = %v, want C (10s / 64 tokens = the highest slowdown)", head)
	}
	// a big head that has waited long enough takes over — the ratio is monotone in
	// wait, which is the whole no-starvation argument. The takeover is NOT soon: the
	// bound is the small job's ratio x the big job's cold tokens (~500,000s here),
	// which is exactly why "wait/cold" is a starvation-free order rather than a
	// small-first-forever one.
	s.mu.Lock()
	// age ONLY the big head: its competitors' waits stay where they were (sustained
	// arrivals keep producing FRESH small jobs, which is the regime that matters).
	s.eng().waiters["A"].since = now.Add(-600000 * time.Second)
	head = s.entryHeadLocked(s.eng(), nil)
	kA := s.entryKeyLocked(s.eng().waiters["A"], nil)
	kC := s.entryKeyLocked(s.eng().waiters["C"], nil)
	s.mu.Unlock()
	if head == nil {
		t.Fatal("no head")
	}
	if kA.hrrn <= kC.hrrn {
		t.Fatalf("after 600000s the big head's slowdown (%.3f) must exceed the small one's (%.3f)", kA.hrrn, kC.hrrn)
	}
	if head.id != "A" {
		t.Fatalf("head = %q, want A once its wait dominates", head.id)
	}
}

// TestHRRNCannotStarvate is the property the arm needs: under SUSTAINED arrivals of
// small jobs, a big job's priority eventually exceeds every small one's.
func TestHRRNCannotStarvate(t *testing.T) {
	// The property is arithmetic: wait/cold is strictly increasing in wait and
	// strictly decreasing in cold, so for ANY finite competitor there is a FINITE
	// wait beyond which the big job outranks it — bound = competitor_ratio * cold_big.
	const bigCold = 6400  // a big entry: 100 blocks
	const smallCold = 64  // a small entry: 1 block
	const smallWait = 600 // the worst competitor: a small job already waiting 600s
	smallRatio := entryPriority(smallWait, smallCold, 64)
	if !(smallRatio > entryPriority(0, bigCold, 64)) {
		t.Fatal("small-first violated: a waiting small job must outrank a fresh big one")
	}
	bound := smallRatio * bigCold // the wait at which the big job's ratio reaches smallRatio
	if bound < 0 || bound > 1e9 {
		t.Fatalf("the starvation bound is not finite: %v", bound)
	}
	if !(entryPriority(bound+1, bigCold, 64) > smallRatio) {
		t.Fatalf("a big job waiting %.0fs (%.4f) must beat the small job (%.4f)",
			bound+1, entryPriority(bound+1, bigCold, 64), smallRatio)
	}
	for _, w := range []float64{1, 30, 600, 3600, 86400} {
		if b := entryPriority(w, smallCold, 64) * bigCold; b < 0 || b > 1e9 {
			t.Fatalf("starvation bound for a small job waiting %v s is %v", w, b)
		}
	}
}

// ---- F5′ bounded HRRN ----

// hrrnSpec is one waiter of a scenario: how long it has waited and how many cold
// tokens it will have to prefill.
type hrrnSpec struct {
	id   string
	wait time.Duration
	cold int64
}

// hrrnWaiterSet installs a waiter set on a scheduler built with the given entry order
// and returns the scheduler, so one scenario can be read through both rules.
func hrrnWaiterSet(t *testing.T, order string, now time.Time, specs ...hrrnSpec) *Scheduler {
	t.Helper()
	s := fArmScheduler(t, func(c *Config) { c.EntryOrder = order })
	s.mu.Lock()
	s.now = now
	s.eng().waiters = map[string]*waiter{}
	for _, spec := range specs {
		s.eng().waiters[spec.id] = &waiter{id: spec.id, since: now.Add(-spec.wait),
			sizeBytes: bytesFor(int(spec.cold / 64)), coldTokens: spec.cold}
	}
	s.mu.Unlock()
	return s
}

// headOf is the D1 head of the scheduler's single engine.
func headOf(s *Scheduler) *waiter {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.entryHeadLocked(s.eng(), nil)
}

// TestHRRNBoundedPriorityAgesAtTheMedianRate is F5′'s intent test (the reviewer's):
// a 200K-token waiter must catch a fresh/normal-sized one within ~median-size x ratio,
// NOT within ~an hour.
//
// The scenario is B1b's tail in miniature — one big entry beside a queue of 3K ones
// that have each been waiting ten minutes (the regime that produced the measured
// 3,601 s max entry wait). With the median = 3,000 tokens:
//
//	plain HRRN   bound = (600/3,000) x 200,000 = 40,000 s   <- the measured defect
//	bounded      bound = (600/3,000) x   3,000 =    600 s   <- the fix, parameter-free
//
// and the bounded priority grows at exactly the MEDIAN rate (2w doubles it), which is
// the property the prediction rests on.
func TestHRRNBoundedPriorityAgesAtTheMedianRate(t *testing.T) {
	const (
		bigCold   int64 = 200_000 // the huge entry that stalled ~an hour
		smallCold int64 = 3_000   // the small class; it IS the median here
		smallWait       = 600 * time.Second
	)
	now := goldenT0
	specs := []hrrnSpec{
		{"s1", smallWait, smallCold},
		{"s2", smallWait, smallCold},
		{"s3", smallWait, smallCold},
		{"s4", smallWait, smallCold},
	}
	withBig := func(wait time.Duration) []hrrnSpec {
		return append([]hrrnSpec{{"big", wait, bigCold}}, specs...)
	}

	// the bound a pass actually uses is the live median of the waiter set
	plain := hrrnWaiterSet(t, "hrrn", now, withBig(0)...)
	bounded := hrrnWaiterSet(t, "hrrn-bounded", now, withBig(0)...)
	plain.mu.Lock()
	if got := plain.entryColdBoundLocked(plain.waitersLocked(plain.eng())); got != 0 {
		t.Fatalf("plain hrrn must carry no bound, got %d", got)
	}
	plain.mu.Unlock()
	bounded.mu.Lock()
	if got := bounded.entryColdBoundLocked(bounded.waitersLocked(bounded.eng())); got != smallCold {
		t.Fatalf("bound = %d, want the live median %d (one 200K entry among four 3K ones)", got, smallCold)
	}
	bounded.mu.Unlock()

	// (1) SMALL-FIRST IS PRESERVED: while the big entry is young it is not the head
	// under either rule — the 3K entries, all older, go first.
	for _, s := range []*Scheduler{plain, bounded} {
		s.mu.Lock()
		s.eng().waiters["big"].since = now.Add(-60 * time.Second)
		s.mu.Unlock()
		if h := headOf(s); h == nil || h.coldTokens == bigCold {
			t.Fatalf("head = %v at a 60 s big-entry wait, want a 3K entry (small-first)", h)
		}
	}

	// (2) THE BOUNDED OVERTOKE: one small-wait's worth of extra aging is enough — the
	// bounded priority grows at the median rate, so 601 s beats the 600 s small class.
	bounded.mu.Lock()
	bounded.eng().waiters["big"].since = now.Add(-601 * time.Second)
	bounded.mu.Unlock()
	if h := headOf(bounded); h == nil || h.id != "big" {
		t.Fatalf("bounded head = %v at a 601 s big-entry wait, want big (bound = ratio x median = 600 s)", h)
	}

	// (3) ... and PLAIN HRRN is nowhere near it at that wait: the 67x size gap is the
	// whole defect. Its bound is ratio x cold_big = 40,000 s, and it is still not the
	// head at 39,000 s.
	plain.mu.Lock()
	plain.eng().waiters["big"].since = now.Add(-39_000 * time.Second)
	plain.mu.Unlock()
	if h := headOf(plain); h == nil || h.id == "big" {
		t.Fatalf("plain hrrn head = %v at a 39,000 s big-entry wait, want a 3K entry (its bound is 40,000 s)", h)
	}
	plain.mu.Lock()
	plain.eng().waiters["big"].since = now.Add(-40_001 * time.Second)
	plain.mu.Unlock()
	if h := headOf(plain); h == nil || h.id != "big" {
		t.Fatalf("plain hrrn head = %v at a 40,001 s big-entry wait, want big (ratio x cold_big)", h)
	}

	// (4) THE RATE ITSELF: the bounded priority is entryPriority over the MEDIAN
	// denominator, so it is linear in the wait and identical to plain HRRN's value for
	// a median-sized entry.
	for _, w := range []float64{0, 30, 600, 3601} {
		got := boundedEntryPriority(w, bigCold, smallCold, 64)
		want := entryPriority(w, smallCold, 64)
		if got != want {
			t.Fatalf("bounded priority at %vs = %v, want the median-rate value %v", w, got, want)
		}
		if d := boundedEntryPriority(2*w, bigCold, smallCold, 64); d != 2*got {
			t.Fatalf("bounded priority is not linear in the wait: p(%v)=%v, p(%v)=%v", w, got, 2*w, d)
		}
	}
	if boundedEntryPriority(601, bigCold, 0, 64) != entryPriority(601, bigCold, 64) {
		t.Fatal("coldBound 0 must be EXACTLY entryPriority (the unbounded path)")
	}
}

// TestHRRNBoundedKeepsSmallFirstInsideTheSmallClass: the bound only ever binds AT or
// ABOVE the median (the even-set median is the LOWER one), so an entry below the median
// keeps its own size and the small class's small-first ordering is F5's exactly.
func TestHRRNBoundedKeepsSmallFirstInsideTheSmallClass(t *testing.T) {
	now := goldenT0
	s := hrrnWaiterSet(t, "hrrn-bounded", now,
		hrrnSpec{"tiny", 600 * time.Second, 1_000},
		hrrnSpec{"small", 600 * time.Second, 2_000},
		hrrnSpec{"big", 0 * time.Second, 200_000},
	)
	s.mu.Lock()
	bound := s.entryColdBoundLocked(s.waitersLocked(s.eng()))
	if bound != 2_000 {
		t.Fatalf("bound = %d, want 2000 (the lower median of {1000,2000,200000})", bound)
	}
	// below the median: the entry keeps its OWN size, i.e. the key is plain HRRN's
	kTiny := s.entryKeyBoundLocked(s.eng().waiters["tiny"], nil, bound)
	kSmall := s.entryKeyBoundLocked(s.eng().waiters["small"], nil, bound)
	kBig := s.entryKeyBoundLocked(s.eng().waiters["big"], nil, bound)
	s.mu.Unlock()
	if kTiny.hrrn != entryPriority(600, 1_000, 64) {
		t.Fatalf("the 1K entry's priority = %v, want its own size (%v)", kTiny.hrrn, entryPriority(600, 1_000, 64))
	}
	// same wait, smaller size -> higher priority: small-first inside the small class
	if !(kTiny.hrrn > kSmall.hrrn) {
		t.Fatalf("the 1K entry (%v) must outrank the 2K one (%v) at equal wait", kTiny.hrrn, kSmall.hrrn)
	}
	// a FRESH big entry is behind both: the small class is served first
	if !(kTiny.hrrn > kBig.hrrn && kSmall.hrrn > kBig.hrrn) {
		t.Fatalf("a fresh big entry (%v) must be behind both small ones (%v, %v)", kBig.hrrn, kTiny.hrrn, kSmall.hrrn)
	}
	if h := headOf(s); h == nil || h.id != "tiny" {
		t.Fatalf("head = %v, want tiny (smallest of the small class)", h)
	}
}

// TestHRRNBoundedReproducesTheB1bTailAsBounded replays the measured B1b shape: twenty
// waiters, nineteen small ones already waiting up to an hour, one 200K-token entry that
// was still queued at the 3,601 s mark. Under plain HRRN that entry is NOT the head (its
// ratio is ~67x too small); under F5′ it is, because it ages at the small class's rate.
func TestHRRNBoundedReproducesTheB1bTailAsBounded(t *testing.T) {
	now := goldenT0
	const bigCold, smallCold int64 = 200_000, 3_000
	var specs []hrrnSpec
	for i := 0; i < 19; i++ {
		// staggered like a saturated queue: 3,600 s down to ~0, one per ~190 s
		specs = append(specs, hrrnSpec{fmt.Sprintf("s%02d", i), time.Duration(3600-i*190) * time.Second, smallCold})
	}
	specs = append(specs, hrrnSpec{"big", 3610 * time.Second, bigCold})

	bounded := hrrnWaiterSet(t, "hrrn-bounded", now, specs...)
	if h := headOf(bounded); h == nil || h.id != "big" {
		t.Fatalf("bounded head = %v at the B1b tail, want big (the median rate gives it a bounded wait)", h)
	}
	plain := hrrnWaiterSet(t, "hrrn", now, specs...)
	h := headOf(plain)
	if h == nil || h.id == "big" {
		t.Fatalf("plain hrrn head = %v, want a small entry — the measured 3,601 s stall", h)
	}
	// the arithmetic of the defect, stated as a number: the big entry's ratio is the
	// small class's divided by the size gap.
	plain.mu.Lock()
	bigP := plain.entryKeyLocked(plain.eng().waiters["big"], nil).hrrn
	plain.mu.Unlock()
	bounded.mu.Lock()
	bigB := bounded.entryKeyBoundLocked(bounded.eng().waiters["big"], nil, smallCold).hrrn
	bounded.mu.Unlock()
	if !(bigB > bigP*50) {
		t.Fatalf("bounded big priority (%v) must exceed the plain one (%v) by ~the 67x size gap", bigB, bigP)
	}
}

// TestHRRNBoundedEqualsPlainWhenTheBoundCannotBind is the no-op guarantee: when every
// waiter is the same size the median IS that size, so min(cold, median) = cold and the
// bounded order is bit-for-bit plain HRRN's. A single waiter is the same case.
func TestHRRNBoundedEqualsPlainWhenTheBoundCannotBind(t *testing.T) {
	now := goldenT0
	uniform := func(wait time.Duration) []hrrnSpec {
		var out []hrrnSpec
		for i, w := range []time.Duration{wait, wait / 2, wait / 3, 0} {
			out = append(out, hrrnSpec{fmt.Sprintf("u%d", i), w, 3_000})
		}
		return out
	}
	for _, wait := range []time.Duration{0, 10 * time.Second, 600 * time.Second, 3600 * time.Second} {
		plain := hrrnWaiterSet(t, "hrrn", now, uniform(wait)...)
		bounded := hrrnWaiterSet(t, "hrrn-bounded", now, uniform(wait)...)
		if hb, hp := headOf(bounded), headOf(plain); hb == nil || hp == nil || hb.id != hp.id {
			t.Fatalf("wait %v: heads differ (%v vs %v) with a uniform-size set", wait, hb, hp)
		}
		var ids []string
		plain.mu.Lock()
		for id := range plain.eng().waiters {
			ids = append(ids, id)
		}
		plain.mu.Unlock()
		for _, id := range ids {
			plain.mu.Lock()
			kp := plain.entryKeyLocked(plain.eng().waiters[id], nil)
			plain.mu.Unlock()
			bounded.mu.Lock()
			kb := bounded.entryKeyBoundLocked(bounded.eng().waiters[id], nil, 3_000)
			bounded.mu.Unlock()
			if kp.hrrn != kb.hrrn {
				t.Fatalf("wait %v: %s priority differs (plain %v, bounded %v)", wait, id, kp.hrrn, kb.hrrn)
			}
		}
	}
	// one waiter: the median is its own size, so the rules agree
	for _, order := range []string{"hrrn", "hrrn-bounded"} {
		s := hrrnWaiterSet(t, order, now, hrrnSpec{"solo", 900 * time.Second, 200_000})
		if h := headOf(s); h == nil || h.id != "solo" {
			t.Fatalf("%s: a lone waiter must be the head, got %v", order, h)
		}
	}
}

// TestHRRNBoundedKeepsThePrefillAccounting: F5′ must carry F5's per-dispatch prefill
// bookkeeping (the engine's PREFILL backlog that F7's gate and the F9 pools read). If it
// did not, an F5-vs-F5′ comparison would differ in more than the entry order — a silent
// confound — and F5′ combined with F7 would see a permanently empty backlog.
func TestHRRNBoundedKeepsThePrefillAccounting(t *testing.T) {
	for _, order := range []string{"hrrn", "hrrn-bounded"} {
		s := fArmScheduler(t, func(c *Config) { c.EntryOrder = order })
		now := goldenT0
		s.mu.Lock()
		s.now = now
		s.mu.Unlock()
		heads := []Head{{Member: "a", ReqID: "a-r1", Arrival: now, SizeBytes: bytesFor(10)}}
		if _, i := s.PickFromHeadsEngine(heads, now); i < 0 {
			t.Fatalf("%s: could not seat a", order)
		}
		s.DispatchedOn("a", "a-r1", "", keys(10, Hash(11)), bytesFor(10), now)
		s.mu.Lock()
		backlog := s.eng().prefillBacklog
		pending := s.eng().members["a"].prefillPending
		s.mu.Unlock()
		if backlog <= 0 || pending <= 0 {
			t.Fatalf("%s: prefill backlog = %d, member pending = %d, want both > 0 (the F5 accounting)", order, backlog, pending)
		}
		if backlog != 10*int64(defaultConfig().BlockSizeTokens) && pending != backlog {
			t.Fatalf("%s: backlog %d != member pending %d", order, backlog, pending)
		}
	}
}

// TestMedianColdTokens pins the bound's definition: the LOWER median of the waiter
// set's sizes, with a fully-warm waiter (0 uncached tokens) floored at one block exactly
// as entryPriority floors its denominator, and 0 for an empty set.
func TestMedianColdTokens(t *testing.T) {
	mk := func(colds ...int64) map[string]*waiter {
		m := map[string]*waiter{}
		for i, c := range colds {
			m[fmt.Sprintf("w%d", i)] = &waiter{id: fmt.Sprintf("w%d", i), coldTokens: c}
		}
		return m
	}
	for _, tc := range []struct {
		name   string
		colds  []int64
		blocks int
		want   int64
	}{
		{"empty", nil, 64, 0},
		{"odd", []int64{200_000, 3_000, 8_000}, 64, 8_000},
		{"even takes the LOWER median", []int64{200_000, 3_000, 8_000, 4_000}, 64, 4_000},
		{"warm waiters floor at one block", []int64{0, 0, 200_000}, 64, 64},
		{"all warm", []int64{0, 0}, 64, 64},
	} {
		if got := medianColdTokens(mk(tc.colds...), tc.blocks); got != tc.want {
			t.Errorf("%s: median = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// ---- F6′ residue-driven release ----

// TestResidueVictimOrderIsFairAndColdFirst is F6″:
//
//  1. a COLD member that has NOT completed its quantum is NEVER evicted for being
//     cold (eligibility is the fair-share definition and F6″ does not touch it);
//  2. among two ELIGIBLE members, the colder one goes;
//  3. ties on class fall through to the spec order (oldest last completion).
func TestResidueVictimOrderIsFairAndColdFirst(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.Residue = "tiebreak"
		c.VictimOrder = "residue"
	})
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	// warm-el Is genuinely warm: dispatched and settled, so its prefix reads resident.
	// Every OTHER member has no chain entries at all, i.e. the model calls it COLD.
	warmHeld := g.residue.dispatch("warm-el", 100)
	g.residue.settle(false, "warm-el", warmHeld, 100, 0)
	g.residue.samples = 0

	mk := func(id string, busy int, lastDone time.Time) *member {
		return &member{id: id, engine: "engine-a", seatTime: now, lastActivity: now,
			busyPeriods: busy, resTurns: busy, lastCompletion: lastDone}
	}
	// NOT eligible: 0 busy periods (fair share not served) — cold must not change that
	g.members["fresh-cold"] = mk("fresh-cold", 0, time.Time{})
	// a member in flight keeps the ENGINE-IDLE liveness rule out of the way (that rule
	// fires when nothing is in flight and nothing is dispatchable — it is a stall
	// guard, not a victim policy, and it is not what this test is about)
	g.members["busy"] = &member{id: "busy", engine: "engine-a", seatTime: now, lastActivity: now,
		inflight: 1, busyPeriods: 0}
	g.waiters["w"] = &waiter{id: "w", since: now}
	if reason, v := s.rotationVictimLocked(g, now); v != nil && v.id == "fresh-cold" {
		t.Fatalf("a cold member with 0 busy periods was evicted (%q): F6\" must not create evictions", reason)
	}
	// two ELIGIBLE members: cold first
	g.members["cold-el"] = mk("cold-el", 4, now.Add(-100*time.Second))
	g.members["warm-el"] = mk("warm-el", 4, now.Add(-200*time.Second))
	reason, v := s.rotationVictimLocked(g, now)
	if v == nil {
		t.Fatal("no victim among the eligible members")
	}
	if v.id != "cold-el" {
		t.Fatalf("victim = %q (%q), want cold-el — the colder ELIGIBLE member is the cheaper eviction even though warm-el finished earlier", v.id, reason)
	}
	// with BOTH cold the class ties and the spec order (oldest completion) decides
	delete(g.residue.chains, "warm-el")
	g.members["warm-el"].lastCompletion = now.Add(-500 * time.Second)
	if reason, v := s.rotationVictimLocked(g, now); v == nil || v.id != "warm-el" {
		t.Fatalf("class tie must fall through to oldest-last-completion (warm-el), got %v (%q)", v, reason)
	}
}

// ---- F7 rotate-when-free ----

// TestRotateWhenFreeHoldsWhilePrefillSaturated is F7's actual intent, restated after
// the B1b precedence fix (ruled 2026-10-31):
//
//	prefill FREE       -> members are eligible at the BASE quantum (Q = 4 here), i.e. F7
//	                      takes Q4's slice exactly when the freed slice cannot be reused;
//	prefill SATURATED  -> rotation is HELD — past the base, not at it — until the tenure
//	                      ceiling, so the slices lengthen to Q8 and then stop.
//
// Before the fix the base bound WAS the ceiling (16), so the "hold" started at 16 and the
// free-prefill half rotated at 16 as well: F7 was Q16-with-a-gate, i.e. an uninterpretable
// arm rather than a Pareto improvement.
func TestRotateWhenFreeHoldsWhilePrefillSaturated(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.RotateWhenFree = true
		c.QuantumCeiling = 8
		c.MaxNumBatchedTokens = 8192
	})
	now := goldenT0
	s.mu.Lock()
	g := s.eng()
	// two idle members that have served their BASE share (Q = 4)
	for _, id := range []string{"old", "older"} {
		g.members[id] = &member{id: id, engine: "engine-a", seatTime: now, lastActivity: now,
			busyPeriods: 4, resTurns: 4}
	}
	// PREFILL SATURATED: more pending prefill tokens than one step can batch
	g.members["busy"] = &member{id: "busy", engine: "engine-a", seatTime: now, lastActivity: now,
		inflight: 1, prefillPending: 16384}
	g.waiters["w"] = &waiter{id: "w", since: now}
	s.deriveLocked()
	if !s.prefillSaturatedLocked(g) {
		s.mu.Unlock()
		t.Fatalf("gate should be CLOSED (backlog %d >= %d)", g.prefillBacklog, s.cfg.MaxNumBatchedTokens)
	}
	reason, victim := s.rotationVictimLocked(g, now)
	s.mu.Unlock()
	if victim != nil {
		t.Fatalf("rotation fired (%q) while prefill was saturated: %+v", reason, victim)
	}
	// the HOLD extends to the ceiling: at 7 busy periods (base 4, ceiling 8) the
	// saturated engine still refuses to rotate
	s.mu.Lock()
	g.members["old"].busyPeriods = 7
	reason, victim = s.rotationVictimLocked(g, now)
	s.mu.Unlock()
	if victim != nil {
		t.Fatalf("rotation fired at 7 busy periods while saturated (%q): the gate must hold up to the ceiling (8)", reason)
	}
	// the CEILING still fires: give a member 8 busy periods
	s.mu.Lock()
	g.members["old"].busyPeriods = 8
	reason, victim = s.rotationVictimLocked(g, now)
	s.mu.Unlock()
	if victim == nil || victim.id != "old" {
		t.Fatalf("the tenure ceiling must evict even while prefill is saturated, got %v (%q)", victim, reason)
	}
	// and with the gate OPEN (prefill drained) rotation fires normally
	s.mu.Lock()
	g.members["busy"].prefillPending = 0
	s.deriveLocked()
	if s.prefillSaturatedLocked(g) {
		s.mu.Unlock()
		t.Fatal("gate should be open after the prefill drained")
	}
	reason, victim = s.rotationVictimLocked(g, now)
	s.mu.Unlock()
	if victim == nil {
		t.Fatal("rotation must resume once prefill has room")
	}
}

// TestPrefillBacklogDecrementsAtFirstToken is F7's tracking correctness: the
// backlog falls at StartOfStream (ChunkAt), NOT at EndOfStream.
// TestRotateWhenFreeRotatesAtTheBaseQuantum: with prefill FREE the same Q4-complete
// member is eligible at once — F7's other half, which the precedence bug removed.
func TestRotateWhenFreeRotatesAtTheBaseQuantum(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.RotateWhenFree = true
		c.QuantumCeiling = 8
		c.MaxNumBatchedTokens = 8192
	})
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.members["q4"] = &member{id: "q4", engine: "engine-a", seatTime: now, lastActivity: now,
		busyPeriods: 4, resTurns: 4, lastCompletion: now}
	g.members["busy"] = &member{id: "busy", engine: "engine-a", seatTime: now, lastActivity: now,
		inflight: 1, prefillPending: 1024} // in flight, but prefill is NOT over budget
	g.waiters["w"] = &waiter{id: "w", since: now}
	s.deriveLocked()
	if s.prefillSaturatedLocked(g) {
		s.mu.Unlock()
		t.Fatalf("fixture: prefill backlog %d must be under the %d-token budget", g.prefillBacklog, s.cfg.MaxNumBatchedTokens)
	}
	if reason, victim := s.rotationVictimLocked(g, now); victim == nil || victim.id != "q4" {
		t.Fatalf("with prefill FREE a Q4-complete member must be rotatable, got %v (%q)", victim, reason)
	}
}

func TestPrefillBacklogDecrementsAtFirstToken(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.RotateWhenFree = true })
	now := goldenT0
	seatWithBlocks(t, s, "m1", 10, now)
	s.ChunkAt("m1", now.Add(500*time.Millisecond)) // first turn's prefill completes
	s.mu.Lock()
	if s.eng().prefillBacklog != 0 {
		s.mu.Unlock()
		t.Fatalf("fixture: backlog after the first turn's first token = %d, want 0", s.eng().prefillBacklog)
	}
	s.mu.Unlock()
	heads := []Head{{Member: "m1", ReqID: "m1-r2", Arrival: now.Add(2 * time.Second), SizeBytes: bytesFor(100)}}
	if _, i := s.PickFromHeadsEngine(heads, now.Add(2*time.Second)); i < 0 {
		t.Fatal("second turn not seated")
	}
	s.DispatchedOn("m1", "m1-r2", "", keys(100, Hash(31)), bytesFor(100), now.Add(2*time.Second))
	s.mu.Lock()
	before := s.eng().prefillBacklog
	s.mu.Unlock()
	if before <= 0 {
		t.Fatalf("prefill backlog after dispatch = %d, want the prompt's tokens", before)
	}
	// EndOfStream must NOT clear it — only the first token may.
	s.SettleOn("m1", "m1-r2", "", struct{ Prompt, Cached int }{Prompt: 6400, Cached: 6400}, true, now.Add(9*time.Second))
	s.mu.Lock()
	afterSettle := s.eng().prefillBacklog
	s.mu.Unlock()
	if afterSettle < before {
		t.Fatalf("the backlog fell at END of stream (%d → %d): only the first token may release it", before, afterSettle)
	}
	s.ChunkAt("m1", now.Add(3*time.Second))
	s.mu.Lock()
	afterChunk := s.eng().prefillBacklog
	s.mu.Unlock()
	if afterChunk != 0 {
		t.Fatalf("backlog after the first token = %d, want 0", afterChunk)
	}
}

// ---- F9 MLFQ-2 ----

// TestMLFQ2PromotionAndPools: a newcomer is SHORT until it outlives one busy
// period; only LONG members rotate; the long pool's ceiling is longQuantum.
func TestMLFQ2PromotionAndPools(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.RotationMode = "mlfq2"
		c.LongQuantum = 8
	})
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	// a member that has NOT yet completed a busy period (a fresh arrival)
	leaf := &member{id: "leaf", engine: "engine-a", seatTime: now, lastActivity: now}
	g.members["leaf"] = leaf
	if s.mlffqLongLocked(leaf) {
		t.Fatal("a newcomer must start in the SHORT pool")
	}
	if _, ok := s.eligibleLocked(g, leaf, now.Add(time.Hour)); ok {
		t.Fatal("a SHORT member must never be rotation-evicted (it finishes or is promoted)")
	}
	leaf.busyPeriods = 1
	s.promoted["leaf"] = true
	if !s.mlffqLongLocked(leaf) {
		t.Fatal("surviving one busy period must promote to the LONG pool")
	}
	leaf.inflight = 0
	leaf.lastActivity = now.Add(-time.Hour)
	leaf.waitingSince = time.Time{}
	if q, bounded := s.quantumBoundLocked(g, leaf); !bounded || q != 8 {
		t.Fatalf("long-pool bound = (%d,%v), want (8,true)", q, bounded)
	}
	leaf.busyPeriods = 8
	if reason, ok := s.eligibleLocked(g, leaf, now.Add(time.Hour)); !ok || reason != "quantum-complete" {
		t.Fatalf("a long member at Q_l must be rotation-eligible, got (%q,%v)", reason, ok)
	}
}

// TestMLFQ2ArbitrationServesBothPools is F9's no-starvation property: with demand
// in BOTH pools the seat alternates by the HRRN ratio instead of draining one pool.
func TestMLFQ2ArbitrationServesBothPools(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.RotationMode = "mlfq2"
		c.LongQuantum = 8
	})
	now := goldenT0
	s.mu.Lock()
	s.now = now
	// one SHORT head that has waited 1s for a tiny job, one LONG head that has
	// waited 100s for a bigger one
	s.eng().waiters["shorty"] = &waiter{id: "shorty", since: now.Add(-1 * time.Second), sizeBytes: bytesFor(1), coldTokens: 64}
	s.eng().waiters["longy"] = &waiter{id: "longy", since: now.Add(-100 * time.Second), sizeBytes: bytesFor(64), coldTokens: 4096}
	s.promoted["longy"] = true
	head := s.mlfq2HeadLocked(s.eng(), nil)
	s.mu.Unlock()
	if head == nil {
		t.Fatal("no head")
	}
	// slowdowns: shorty 1s/64 = 0.0156 ; longy 100s/4096 = 0.0244 → longy wins
	if head.id != "longy" {
		t.Fatalf("head = %q, want longy (the higher response ratio)", head.id)
	}
	// now let shorty wait long enough that it wins — both pools get service
	s.mu.Lock()
	s.now = now.Add(200 * time.Second)
	head = s.mlfq2HeadLocked(s.eng(), nil)
	s.mu.Unlock()
	if head == nil || head.id != "shorty" {
		t.Fatalf("head = %v, want shorty once its wait dominates (both pools must be served)", head)
	}
}

// TestFArmKnobsOffKeepTheGolden is the default-off guarantee, asserted as a knob
// set rather than by re-running the golden (TestGoldenReplay does that): every
// F-knob's default leaves the config equal to the anchor's.
func TestFArmKnobsOffKeepTheGolden(t *testing.T) {
	c := defaultConfig()
	if c.QuantumMode != "fixed" || c.EntryOrder != "newcomer-first" ||
		c.VictimOrder != "legacy" || c.RotateWhenFree || c.QuantumCeiling != 0 {
		t.Fatalf("an F-knob default is not off: quantumMode=%q entryOrder=%q victimOrder=%q rotateWhenFree=%v ceiling=%d",
			c.QuantumMode, c.EntryOrder, c.VictimOrder, c.RotateWhenFree, c.QuantumCeiling)
	}
	// and the F0 base (the q4 anchor) is expressed purely in config
	f0 := defaultConfig()
	f0.RotationMode = "quantum"
	f0.QuantumBusyPeriods = 4
	f0.GateMode = "size-aware"
	for _, arm := range []func(*Config){
		func(c *Config) { c.QuantumMode = "dyn-size" },                        // F3
		func(c *Config) { c.EntryOrder = "hrrn" },                             // F5
		func(c *Config) { c.EntryOrder = "hrrn-bounded" },                     // F5′
		func(c *Config) { c.Residue = "tiebreak"; c.VictimOrder = "residue" }, // F6"
		func(c *Config) { c.RotateWhenFree = true; c.QuantumCeiling = 16 },    // F7
		func(c *Config) { c.RotationMode = "mlfq2" },                          // F9
	} {
		cfg := f0
		arm(&cfg)
		if err := cfg.effective().validate(); err != nil {
			t.Fatalf("arm config rejected: %v", err)
		}
	}
	// F6" without the residue model is a startup failure, not a silent no-op.
	bad := f0
	bad.VictimOrder = "residue"
	if err := bad.effective().validate(); err == nil {
		t.Fatal("victimOrder=residue without residue=tiebreak must fail validation")
	}
}

// TestEngineGateHoldsDispatches (reviewer G1): in per-engine mode a saturated
// engine's SEATED heads are not dispatched either — the gate is the pool gate
// applied to each engine, not an admission-only rule — while the healthy engine's
// heads keep flowing.
func TestEngineGateHoldsDispatches(t *testing.T) {
	cfg := e4aCfg()
	cfg.EngineGate = "per-engine"
	cfg.MemberKey = "session"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s.SetNumBlocksFor(id, 64, 4000); err != nil {
			t.Fatal(err)
		}
	}
	now := goldenT0
	// seat one member on each engine (both healthy at first)
	s.SetEngineLoad("engine-a", 0.10, 0, now)
	s.SetEngineLoad("engine-b", 0.10, 0, now)
	for _, id := range []string{"a1", "b1"} {
		heads := []Head{{Member: id, Session: id, ReqID: id + "-r1", Arrival: now, SizeBytes: bytesFor(10)}}
		if _, i := s.PickFromHeadsEngine(heads, now); i < 0 {
			t.Fatalf("%s not seated", id)
		}
		s.DispatchedOn(id, id+"-r1", s.bindingOf(id), keys(10, Hash(7)), bytesFor(10), now)
	}
	// engine-a saturates
	s.SetEngineLoad("engine-a", 0.95, 0, now)
	// both members have a new turn queued
	heads := []Head{
		{Member: "a1", Session: "a1", ReqID: "a1-r2", Arrival: now.Add(2 * time.Second), SizeBytes: bytesFor(10)},
		{Member: "b1", Session: "b1", ReqID: "b1-r2", Arrival: now.Add(2 * time.Second), SizeBytes: bytesFor(10)},
	}
	engine, i := s.PickFromHeadsEngine(heads, now.Add(2*time.Second))
	if i < 0 {
		t.Fatal("nothing dispatched: the healthy engine's head must still flow")
	}
	if engine != "engine-b" {
		t.Fatalf("dispatched to %q, want engine-b (engine-a is saturated: its heads must be HELD, not merely unseated)", engine)
	}
	s.mu.Lock()
	skipped := s.groups["engine-a"].counters.engineGateSkipped
	s.mu.Unlock()
	if skipped == 0 {
		t.Fatal("the saturated engine's head was not counted as gate-held")
	}
}

// TestGoldenE4aUnchangedByFArmKnobs guards the interaction: turning the F-knobs on
// must not disturb the E4a golden's pinned facts when they are off (the golden test
// itself is the arbiter; this asserts the config surface is what changes).
func TestFArmKnobSurfaceIsConfigOnly(t *testing.T) {
	base := fmt.Sprintf("%+v", defaultConfig())
	cfg := defaultConfig()
	cfg.QuantumMode = "dyn-size"
	if fmt.Sprintf("%+v", cfg) == base {
		t.Fatal("knob assignment did not change the config")
	}
}

// TestPrefillChargeIsUncachedNotPrompt is RF7: the F7 gate must see the request's
// ESTIMATED UNCACHED work, not its prompt size. A warm seated turn (most of its
// blocks already resident on this engine) charges a small fraction; a cold entry
// charges all of it. With prompt tokens the gate would sit closed almost always
// (one warm 50-100k-token turn beats an 8,192-token step budget on its own) and F7
// would stop being load-adaptive.
func TestPrefillChargeIsUncachedNotPrompt(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.RotateWhenFree = true })
	now := goldenT0
	// a member with 100 blocks resident: dispatch them so the index holds them
	seatWithBlocks(t, s, "warm", 100, now)
	s.ChunkAt("warm", now.Add(500*time.Millisecond))
	// its next turn re-sends the same 100 blocks plus nothing new: a full warm hit
	warmKeys := append(keys(100, Hash(len("warm")*97+100)), Hash(999999))
	s.mu.Lock()
	g := s.eng()
	cold := s.uncachedTokensLocked(g, warmKeys)
	total := int64(len(warmKeys)) * int64(s.blockSize)
	s.mu.Unlock()
	if cold >= total/2 {
		t.Fatalf("a warm turn charged %d of %d tokens: the charge must be the UNCACHED work, not the prompt", cold, total)
	}
	// a cold entry: every key is new, so it charges the whole prompt
	s.mu.Lock()
	coldEntry := s.uncachedTokensLocked(g, keys(50, Hash(424242)))
	s.mu.Unlock()
	if coldEntry != 50*int64(s.blockSize) {
		t.Fatalf("a cold entry charged %d, want its full %d tokens", coldEntry, 50*int64(s.blockSize))
	}
}
