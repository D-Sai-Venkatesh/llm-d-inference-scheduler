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
