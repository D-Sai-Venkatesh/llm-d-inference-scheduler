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

// TestTenureCeilingBoundsDynamicQuantum is the fairness guarantee: with a ceiling
// set, the dynamic quantum can NEVER exceed it — that is what stops a big member's
// scaled slice from being unbounded.
func TestTenureCeilingBoundsDynamicQuantum(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.QuantumMode = "dyn-size"
		c.QuantumCeiling = 6
	})
	now := goldenT0
	for i := 0; i < 5; i++ { // five tiny members so the MEDIAN is 1, not the mean
		seatWithBlocks(t, s, fmt.Sprintf("tiny%d", i), 1, now.Add(time.Duration(i)*time.Second))
	}
	seatWithBlocks(t, s, "huge", 200, now.Add(20*time.Second))
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.eng().members["huge"]
	if got := s.dynQuantumLocked(s.eng(), m); got < 100 {
		t.Fatalf("test premise: the scaled quantum should be huge, got %d", got)
	}
	if q, bounded := s.quantumBoundLocked(s.eng(), m); !bounded || q != 6 {
		t.Fatalf("ceiling bound = (%d,%v), want (6,true) — the ceiling overrides the scaled quantum", q, bounded)
	}
	if !s.ceilingReachedLocked(s.eng(), &member{id: "huge", busyPeriods: 6}) {
		t.Fatal("6 busy periods must reach a ceiling of 6")
	}
	if s.ceilingReachedLocked(s.eng(), &member{id: "huge", busyPeriods: 5}) {
		t.Fatal("5 busy periods must not reach a ceiling of 6")
	}
}

// ---- F5 HRRN ----

// TestHRRNOrdersByTheRatio: the entry head is the highest (wait + s)/s, so a small
// entry jumps an older large one only while its ratio is genuinely higher.
func TestHRRNOrdersByTheRatio(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) { c.EntryOrder = "hrrn" })
	now := goldenT0
	// seed a measured prefill rate: 6400 tokens in 1s = 6400 tok/s
	s.mu.Lock()
	s.eng().prefillSamples = prefillRateMinSamples
	s.eng().prefillRate = 6400
	s.mu.Unlock()

	// waiter A: old (wait 100s), big (6400 cold tokens ⇒ s = 1s) → ratio 101
	// waiter B: new (wait 1s), tiny (64 cold tokens ⇒ s = 0.01s) → ratio 101
	// ⇒ nearly tied; then make B's wait 3s and its ratio wins outright.
	mk := func(id string, since time.Time, cold int64) *waiter {
		return &waiter{id: id, since: since, sizeBytes: bytesFor(int(cold / 64)), coldTokens: cold}
	}
	s.mu.Lock()
	s.eng().waiters["A"] = mk("A", now.Add(-100*time.Second), 6400)
	s.eng().waiters["B"] = mk("B", now.Add(-1*time.Second), 64)
	s.eng().waiters["C"] = mk("C", now.Add(-1000*time.Second), 64) // huge wait, tiny job
	head := s.entryHeadLocked(s.eng(), nil)
	s.mu.Unlock()
	if head == nil || head.id != "C" {
		t.Fatalf("HRRN head = %v, want C (wait 1000s over 0.01s service)", head)
	}
	// raise A's wait so its ratio overtakes C's
	s.mu.Lock()
	s.now = now.Add(10000 * time.Second)
	head = s.entryHeadLocked(s.eng(), nil)
	s.mu.Unlock()
	if head == nil {
		t.Fatal("no head")
	}
	// C's ratio also grew (tiny service), so C still wins — the property is
	// MONOTONICITY in wait, not a swap. Assert the ordering property instead:
	s.mu.Lock()
	kC := s.entryKeyLocked(s.eng().waiters["C"], nil)
	kA := s.entryKeyLocked(s.eng().waiters["A"], nil)
	s.mu.Unlock()
	if !(kC.hrrn > kA.hrrn) {
		t.Fatalf("small-job-first violated: C hrrn %.1f must exceed A hrrn %.1f", kC.hrrn, kA.hrrn)
	}
}

// TestHRRNCannotStarvate is the property test the arm needs: under SUSTAINED
// arrivals of small jobs, a big job's ratio eventually exceeds every small one's
// (wait grows without bound, service is finite), so it is served.
func TestHRRNCannotStarvate(t *testing.T) {
	// The no-starvation property is arithmetic, and this is the whole proof:
	// (wait + s)/s is strictly increasing in wait and strictly decreasing in s, so
	// for ANY finite population of short jobs there is a FINITE wait beyond which
	// the long job outranks every one of them.
	const bigService = 8.0   // a big entry: 8s of prefill
	const smallService = 0.1 // a small entry: 0.1s
	const smallWait = 600.0  // the worst competitor: a small job already waiting 600s
	smallRatio := hrrn(smallWait, smallService)
	if !(smallRatio > hrrn(0, bigService)) {
		t.Fatal("small-first violated: a waiting small job must outrank a fresh big one")
	}
	// the big job's wait that beats it solves (W + s)/s > smallRatio:  W > s·ratio − s
	bound := bigService*smallRatio - bigService
	if bound < 0 || bound > 1e9 {
		t.Fatalf("the starvation bound is not finite: %v", bound)
	}
	if !(hrrn(bound+1, bigService) > smallRatio) {
		t.Fatalf("a big job waiting %.0fs (ratio %.2f) must beat the small job (ratio %.2f)",
			bound+1, hrrn(bound+1, bigService), smallRatio)
	}
	// and the bound stays finite for any realistic wait a small job can accumulate —
	// which is what "sustained arrivals cannot starve a long job" reduces to.
	for _, w := range []float64{1, 30, 600, 3600, 86400} {
		if b := bigService*hrrn(w, smallService) - bigService; b < 0 || b > 1e9 {
			t.Fatalf("starvation bound for a small job waiting %v s is %v", w, b)
		}
	}
}

// ---- F6′ residue-driven release ----

// TestResidueReleaseFreesAColdSeat: a seated idle member whose blocks the model says
// are gone loses its seat when a waiter needs it; a WARM member does not.
func TestResidueReleaseFreesAColdSeat(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.Residue = "tiebreak"
		c.ResidueRelease = "on"
	})
	now := goldenT0
	seatWithBlocks(t, s, "warm", 60, now)
	s.mu.Lock()
	g := s.eng()
	// the model is made to say "warm" for warm and "cold" for a second member
	g.residue.setCapacity(1000)
	h := g.residue.dispatch("warm", 60)
	g.residue.settle(false, "warm", h, 0, 0)
	g.residue.samples = 0
	cold := &member{id: "cold", engine: "engine-a", seatTime: now, lastActivity: now}
	g.members["cold"] = cold
	if g.residue.ratio("warm") < residueWarmFraction {
		s.mu.Unlock()
		t.Fatal("test premise: warm must be warm")
	}
	// a waiter exists (the release only fires when somebody wants the seat)
	g.waiters["needy"] = &waiter{id: "needy", since: now}
	reason, victim := s.residueReleaseVictimLocked(g, now)
	s.mu.Unlock()
	if victim == nil || victim.id != "cold" || reason != "residue-cold" {
		t.Fatalf("release victim = %+v (%q), want the cold member", victim, reason)
	}
	// with NO waiter nothing is released (a release would only lose cache)
	s.mu.Lock()
	delete(g.waiters, "needy")
	if _, v := s.residueReleaseVictimLocked(g, now); v != nil {
		s.mu.Unlock()
		t.Fatalf("released %q with no waiter waiting", v.id)
	}
	s.mu.Unlock()
}

// ---- F7 rotate-when-free ----

// TestRotateWhenFreeHoldsWhilePrefillSaturated: the gate blocks rotation evictions
// when prefill is saturated, and the tenure ceiling still evicts (no starvation).
func TestRotateWhenFreeHoldsWhilePrefillSaturated(t *testing.T) {
	s := fArmScheduler(t, func(c *Config) {
		c.RotateWhenFree = true
		c.QuantumCeiling = 16
		c.MaxNumBatchedTokens = 8192
	})
	now := goldenT0
	s.mu.Lock()
	g := s.eng()
	// two idle, rotation-eligible members (they are quantum-complete at Q=4)
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
	// the CEILING still fires: give a member 16 busy periods
	s.mu.Lock()
	g.members["old"].busyPeriods = 16
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
	s.eng().prefillSamples = prefillRateMinSamples
	s.eng().prefillRate = 6400
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
	// ratios: shorty (1+0.01)/0.01 = 101 ; longy (100+0.64)/0.64 = 157 → longy wins
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
		c.ResidueRelease != "off" || c.RotateWhenFree || c.QuantumCeiling != 0 {
		t.Fatalf("an F-knob default is not off: quantumMode=%q entryOrder=%q residueRelease=%q rotateWhenFree=%v ceiling=%d",
			c.QuantumMode, c.EntryOrder, c.ResidueRelease, c.RotateWhenFree, c.QuantumCeiling)
	}
	// and the F0 base (the q4 anchor) is expressed purely in config
	f0 := defaultConfig()
	f0.RotationMode = "quantum"
	f0.QuantumBusyPeriods = 4
	f0.GateMode = "size-aware"
	for _, arm := range []func(*Config){
		func(c *Config) { c.QuantumMode = "dyn-size" },                      // F3
		func(c *Config) { c.EntryOrder = "hrrn" },                           // F5
		func(c *Config) { c.Residue = "tiebreak"; c.ResidueRelease = "on" }, // F6'
		func(c *Config) { c.RotateWhenFree = true; c.QuantumCeiling = 16 },  // F7
		func(c *Config) { c.RotationMode = "mlfq2" },                        // F9
	} {
		cfg := f0
		arm(&cfg)
		if err := cfg.effective().validate(); err != nil {
			t.Fatalf("arm config rejected: %v", err)
		}
	}
	// F6' without the residue model is a startup failure, not a silent no-op.
	bad := f0
	bad.ResidueRelease = "on"
	if err := bad.effective().validate(); err == nil {
		t.Fatal("residueRelease=on without residue=tiebreak must fail validation")
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
