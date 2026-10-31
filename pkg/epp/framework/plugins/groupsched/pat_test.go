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

// PAT's gates (eligibilityMode "pat", pat.go). The knob is default-OFF and the golden
// replay is the proof that "off" is today's behaviour; the first test below proves the
// same for the WRITTEN-OUT default value. The rest are the proof that PAT's four claims
// are true of the code and not only of the prose:
//
//  1. the fairness guard — a COLD member is never released before its fair share,
//     however long it pauses and however cold its cache reads;
//  2. the early exit — a member that pauses past its residue horizon AFTER its fair
//     share becomes eligible, and its cache class comes from ALLOCATIONS (no clock, no
//     cache read);
//  3. the ceiling — a member that stays warm (a rapid-fire caller) is never
//     early-released and runs to the 8-busy-period ceiling, exactly as under Q8;
//  4. the victim order among the eligible is F6″'s (coldest class, then raw ratio,
//     then oldest last completion).

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// patCfg is the PAT arm expressed purely in config: the anchor's quantum (Q_fair = 4
// busy periods), the residue model ON (PAT's COLD class is its verdict) and F6″'s
// victim order (PAT's own order among the eligible).
func patCfg() Config {
	cfg := defaultConfig()
	cfg.RotationMode = "quantum"
	cfg.QuantumBusyPeriods = 4
	cfg.GateMode = "size-aware"
	cfg.EligibilityMode = "pat"
	cfg.Residue = "tiebreak"
	cfg.VictimOrder = "residue"
	return cfg
}

func patScheduler(t *testing.T, tweak func(*Config)) *Scheduler {
	t.Helper()
	cfg := patCfg()
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

// residueSeed writes `blocks` RESIDENT blocks of member id into the engine's shadow:
// dispatch then settle, i.e. exactly the sequence one completed request produces. The
// chain's prefix ends up deepest in the free queue (blocks are freed tail-first).
func residueSeed(g *engineGroup, id string, blocks int) {
	held := g.residue.dispatch(id, blocks)
	g.residue.settle(false, id, held, blocks, 0)
}

// residueDrain ALLOCATES `blocks` blocks for another member: that is the ONLY thing
// that evicts anybody (the pool is finite and a pop recycles the head identity). No
// clock moves and no cache is read — this is the mechanism PAT's COLD class rests on.
func residueDrain(g *engineGroup, by string, blocks int) {
	g.residue.dispatch(by, blocks)
}

// patMember plants a seated member record with the given completed busy periods.
func patMember(s *Scheduler, id string, busy int, lastDone time.Time, now time.Time) *member {
	g := s.eng()
	m := &member{id: id, engine: "engine-a", seatTime: now, lastActivity: now,
		busyPeriods: busy, resTurns: busy, lastCompletion: lastDone}
	g.members[id] = m
	return m
}

// TestPatOffIsGoldenNeutral: writing the knob out at its default ("off") must reproduce
// the committed single-engine golden byte for byte. The knob is therefore inert unless
// it is asked for — the B1b requirement, stated as a test.
func TestPatOffIsGoldenNeutral(t *testing.T) {
	cfg := defaultConfig()
	cfg.EligibilityMode = "off"
	got := runGoldenScenarioCfg(t, 2600, cfg, nil)
	want, err := os.ReadFile(filepath.Join("testdata", "golden.txt"))
	if err != nil {
		t.Fatalf("golden file missing: %v", err)
	}
	if got != string(want) {
		gotL := strings.Split(got, "\n")
		wantL := strings.Split(string(want), "\n")
		for i := 0; i < len(gotL) || i < len(wantL); i++ {
			var a, b string
			if i < len(gotL) {
				a = gotL[i]
			}
			if i < len(wantL) {
				b = wantL[i]
			}
			if a != b {
				t.Fatalf("eligibilityMode=off diverged from the golden at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
			}
		}
		t.Fatal("eligibilityMode=off diverged from the golden (length)")
	}
}

// TestPatConfigIsFailClosed: PAT reads three existing knobs and every pair is a startup
// failure, never a silent fallback to a differently-named Q-arm.
func TestPatConfigIsFailClosed(t *testing.T) {
	if err := patCfg().effective().validate(); err != nil {
		t.Fatalf("the PAT arm config must validate: %v", err)
	}
	cases := []struct {
		name  string
		tweak func(*Config)
	}{
		{"without the residue model", func(c *Config) { c.Residue = "off"; c.VictimOrder = "legacy" }},
		{"without residue's victim order", func(c *Config) { c.VictimOrder = "legacy" }},
		{"on a ledger rotation", func(c *Config) { c.RotationMode = "ledger" }},
		{"on mlfq2", func(c *Config) { c.RotationMode = "mlfq2" }},
		{"with an unknown value", func(c *Config) { c.EligibilityMode = "pause" }},
		// the ceiling CAPS; a cap below the fair share would release members BEFORE their
		// share, which is the one thing PAT guarantees never happens.
		{"with a ceiling below the fair share", func(c *Config) { c.QuantumCeiling = 3 }},
		{"with an unbounded quantum", func(c *Config) { c.QuantumBusyPeriods = -1; c.QuantumCeiling = 4 }},
	}
	for _, tc := range cases {
		cfg := patCfg()
		tc.tweak(&cfg)
		if err := cfg.effective().validate(); err == nil {
			t.Fatalf("eligibilityMode=pat %s must fail validation", tc.name)
		} else {
			t.Logf("rejected as intended (%s): %v", tc.name, err)
		}
	}
	// and the DEFAULT (no eligibilityMode at all) still validates: every banked arm.
	def := defaultConfig()
	if err := def.effective().validate(); err != nil {
		t.Fatalf("the default config must validate: %v", err)
	}
}

// TestPatFairShareGuard: the COLD class alone NEVER evicts. A member with a cache that
// is provably gone — every block recycled by somebody else's allocations — stays seated
// until its fair share is served. That is the human's fairness point, and the part the
// removed F6′ got wrong.
func TestPatFairShareGuard(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	residueSeed(g, "a", 100)
	// a member with a live request keeps the engine-idle liveness rule out of the way:
	// that rule is a stall guard (nothing in flight and nothing dispatchable), not the
	// rule under test.
	g.members["busy"] = &member{id: "busy", engine: "engine-a", seatTime: now, lastActivity: now, inflight: 1}
	residueDrain(g, "hog", 960) // 900 anonymous slots + 60 pops: 40 of a's 100 blocks survive
	if r := g.residue.ratio("a"); r >= residueWarmFraction {
		t.Fatalf("fixture: a's residue ratio = %.2f, want COLD (< %.2f)", r, residueWarmFraction)
	}
	for busy := 0; busy < 4; busy++ {
		m := patMember(s, "a", busy, now.Add(-time.Hour), now)
		if reason, ok := s.eligibleLocked(g, m, now.Add(time.Hour)); ok {
			t.Fatalf("a COLD member with %d busy periods was rotation-eligible (%q): nothing may be released before fair share", busy, reason)
		}
	}
	// fair share served ⇒ eligible, on the early-exit reason
	m := g.members["a"]
	m.busyPeriods = 4
	reason, ok := s.eligibleLocked(g, m, now.Add(time.Hour))
	if !ok || reason != reasonPatFairShareCold {
		t.Fatalf("a COLD, idle member at fair share must be eligible on %q, got (%q,%v)", reasonPatFairShareCold, reason, ok)
	}
	// ... but not while it has a queued turn: it wants to run
	m.waitingSince = now
	if reason, ok := s.eligibleLocked(g, m, now.Add(time.Hour)); ok {
		t.Fatalf("a member with a QUEUED turn was eligible (%q): rotation would throw away work it is about to need", reason)
	}
}

// TestPatColdClassDerivesFromAllocations: the class is a fact about BLOCK ALLOCATIONS.
// Advancing the clock by an hour changes nothing (no time constant — the legacy 20 s
// silence rule is deliberately not part of PAT); allocating other members' prompts
// changes everything, with the clock held still; and the MEASURED hit never moves the
// state (open loop — dictate, don't observe), only the model's own accounting does.
func TestPatColdClassDerivesFromAllocations(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	residueSeed(g, "a", 100)
	m := patMember(s, "a", 4, now, now)
	if r := g.residue.ratio("a"); r < residueWarmFraction {
		t.Fatalf("fixture: a's ratio right after its turn = %.2f, want WARM", r)
	}
	// AN HOUR PASSES AND NOTHING IS ALLOCATED: the cache is still there, so the member
	// is NOT eligible — however long the pause. (Under Q8's 20 s silent rule this member
	// was evicted hours ago.)
	if reason, ok := s.eligibleLocked(g, m, now.Add(time.Hour)); ok {
		t.Fatalf("an hour of silence made a WARM member eligible (%q): the class must come from allocations, not from a clock", reason)
	}
	// THE ENGINE'S CACHE IS READ: a wildly optimistic measured hit must not move the
	// model (residueReconcile is off — the measured pair is a VALIDATION signal).
	before := g.residue.ratio("a")
	hogHeld := g.residue.dispatch("hog", 20)
	g.residue.settle(false, "hog", hogHeld, 0, 0) // reported hit 0, prediction 20 blocks
	if after := g.residue.ratio("a"); after != before {
		t.Fatalf("the measured hit moved the shadow (ratio %.3f → %.3f): the model must not read the cache", before, after)
	}
	// ALLOCATION EVICTS IT — one drain, clock unchanged (same `now`) — and the member
	// becomes eligible on exactly the block that crosses the bimodal cut.
	g.residue.reset("") // isolate the drain from the probe above (capacity 1000, a re-seeds)
	g.residue.setCapacity(1000)
	residueSeed(g, "a", 100)
	residueDrain(g, "hog2", 940) // 900 anonymous + 40 pops ⇒ 60 of a's blocks survive
	if r := g.residue.ratio("a"); r < residueWarmFraction {
		t.Fatalf("fixture at 60%% survival: ratio = %.2f, want WARM — the cut must be the bimodal gap, not a hair trigger", r)
	}
	if reason, ok := s.eligibleLocked(g, m, now); ok {
		t.Fatalf("a WARM member (60%% resident) was eligible at the SAME instant (%q): eligibility must track the class", reason)
	}
	residueDrain(g, "hog3", 20) // pops take it under half
	if r := g.residue.ratio("a"); r >= residueWarmFraction {
		t.Fatalf("fixture: ratio after the extra drain = %.2f, want COLD", r)
	}
	reason, ok := s.eligibleLocked(g, m, now)
	if !ok || reason != reasonPatFairShareCold {
		t.Fatalf("a member whose pause OUTLIVED ITS CACHE must be eligible at the same clock instant (%q,%v), want %q", reason, ok, reasonPatFairShareCold)
	}
}

// TestPatRapidFireRunsToTheCeiling: a member that keeps coming back keeps its cache
// warm (its own re-dispatch touches its blocks), so rule (a) never fires and it runs to
// the safety ceiling — i.e. PAT behaves exactly like Q8 for the rapid-fire class, which
// is the point: the adaptive rule only differs where the class differs.
func TestPatRapidFireRunsToTheCeiling(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	residueSeed(g, "rapid", 100)
	m := patMember(s, "rapid", 4, now, now)
	for busy := 4; busy < patCeilingBusyPeriods; busy++ {
		m.busyPeriods = busy
		if r := g.residue.ratio("rapid"); r < residueWarmFraction {
			t.Fatalf("fixture: the rapid-fire member is not warm (%.2f)", r)
		}
		if reason, ok := s.eligibleLocked(g, m, now); ok {
			t.Fatalf("a WARM member at %d busy periods was eligible (%q): rule (a) must not fire while the cache lives", busy, reason)
		}
	}
	m.busyPeriods = patCeilingBusyPeriods
	reason, ok := s.eligibleLocked(g, m, now)
	if !ok || reason != reasonPatCeiling {
		t.Fatalf("a member at the ceiling (%d busy periods) must be eligible on %q, got (%q,%v)",
			patCeilingBusyPeriods, reasonPatCeiling, reason, ok)
	}
	// BOTH halves of the tenure story are asserted (reviewer item 5): the FAIR SHARE is
	// the gate that rule (a) reads, and the CEILING is the cap rule (b) fires on. A
	// reader of the artifacts must be able to see that 4 and 8 are different numbers
	// doing different jobs — the class the f3/f7 precedence bug came from.
	if base, bounded := s.eligibilityBaseLocked(g, m); !bounded || base != 4 {
		t.Fatalf("PAT's fair-share gate = (%d,%v), want (4,true) = the quantum", base, bounded)
	}
	if c := s.tenureCeilingLocked(); c != patCeilingBusyPeriods {
		t.Fatalf("PAT's ceiling = %d, want %d (the measured knee)", c, patCeilingBusyPeriods)
	}
	// and an explicit quantumCeiling overrides the measured knee — the ceiling CAPS, so
	// with ceiling 6 the same warm member rotates at 6, not at 8
	s.cfg.QuantumCeiling = 6
	if c := s.tenureCeilingLocked(); c != 6 {
		t.Fatalf("explicit quantumCeiling = %d, want 6", c)
	}
	m.busyPeriods = 6
	if reason, ok := s.eligibleLocked(g, m, now); !ok || reason != reasonPatCeiling {
		t.Fatalf("with a 6-busy-period ceiling the member must be eligible at 6, got (%q,%v)", reason, ok)
	}
}

// TestPatVictimOrderAmongEligibleIsResidueOrder: eligibility decides WHO MAY go; the
// victim order decides WHO GOES. Among PAT's eligible set it is F6″'s: cold class
// first, then the raw ratio, then oldest last completion. Eligibility itself is NOT
// re-ordered by it — a warm member that reached the ceiling is eligible, it is simply
// the more expensive eviction.
func TestPatVictimOrderAmongEligibleIsResidueOrder(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	// give order decides eviction order (the FIFO pops its head first): coldOld loses
	// all of its chain, coldNew keeps a tenth, warmCeil keeps everything.
	residueSeed(g, "coldOld", 100)
	residueSeed(g, "coldNew", 100)
	residueSeed(g, "warmCeil", 100)
	residueDrain(g, "hog", 890) // 700 anonymous + 190 pops: coldOld loses all of its chain, coldNew keeps a tenth
	got := map[string]float64{}
	for _, id := range []string{"coldOld", "coldNew", "warmCeil"} {
		got[id] = g.residue.ratio(id)
	}
	if !(got["coldOld"] < got["coldNew"] && got["coldNew"] < residueWarmFraction && got["warmCeil"] >= residueWarmFraction) {
		t.Fatalf("fixture: ratios = %v, want coldOld < coldNew < %.2f <= warmCeil", got, residueWarmFraction)
	}
	patMember(s, "coldOld", 4, now.Add(-100*time.Second), now)
	patMember(s, "coldNew", 4, now.Add(-200*time.Second), now)
	patMember(s, "warmCeil", patCeilingBusyPeriods, now.Add(-10_000*time.Second), now)
	// all three are PAT-eligible, on two different reasons
	for _, id := range []string{"coldOld", "coldNew", "warmCeil"} {
		if _, ok := s.eligibleLocked(g, g.members[id], now); !ok {
			t.Fatalf("%s must be PAT-eligible", id)
		}
	}
	reason, v := s.rotationVictimLocked(g, now)
	if v == nil {
		t.Fatal("no victim among three eligible members")
	}
	// coldOld is the COLDEST, so it goes first although warmCeil has waited far longer
	// (oldest last completion) and coldNew finished earlier than coldOld.
	if v.id != "coldOld" {
		t.Fatalf("victim = %q (%q), want coldOld — the coldest ELIGIBLE member is the cheaper eviction", v.id, reason)
	}
	if reason != reasonPatFairShareCold {
		t.Fatalf("the victim's eligibility reason = %q, want %q", reason, reasonPatFairShareCold)
	}
	// with the coldest gone, the next-cheapest cold member goes before the WARM one
	delete(g.members, "coldOld")
	if reason, v := s.rotationVictimLocked(g, now); v == nil || v.id != "coldNew" {
		t.Fatalf("second victim = %v (%q), want coldNew — a warm member is not the cheaper eviction", v, reason)
	}
	// class ties fall through to the spec order (oldest last completion)
	delete(g.members, "coldNew")
	if reason, v := s.rotationVictimLocked(g, now); v == nil || v.id != "warmCeil" {
		t.Fatalf("last victim = %v (%q), want warmCeil", v, reason)
	}
}

// TestPatEligibilityCountersCountTransitions: the two mechanism counters must count
// MEMBERS BECOMING ELIGIBLE (a transition per residency), not decision passes — the
// rotation loop re-tests every member on every pass, so an unlatched counter would
// measure contention and the artifact would say nothing about the rule.
func TestPatEligibilityCountersCountTransitions(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	var fair, ceil int
	s.onPatEligible = func(_ string, reason string) {
		switch reason {
		case reasonPatFairShareCold:
			fair++
		case reasonPatCeiling:
			ceil++
		default:
			t.Errorf("unknown PAT eligibility reason %q", reason)
		}
	}
	s.mu.Lock()
	g := s.eng()
	g.residue.setCapacity(1000)
	residueSeed(g, "a", 100)
	residueDrain(g, "hog", 960)
	m := patMember(s, "a", 4, now.Add(-time.Hour), now)
	for i := 0; i < 3; i++ {
		if reason, ok := s.eligibleLocked(g, m, now); !ok || reason != reasonPatFairShareCold {
			s.mu.Unlock()
			t.Fatalf("pass %d: (%q,%v), want %q", i, reason, ok, reasonPatFairShareCold)
		}
	}
	if fair != 1 {
		s.mu.Unlock()
		t.Fatalf("three decision passes over one eligible member booked %d fair-share transitions, want 1 (the latch must make this a MEMBER count)", fair)
	}
	m.busyPeriods = patCeilingBusyPeriods
	if reason, ok := s.eligibleLocked(g, m, now); !ok || reason != reasonPatCeiling {
		s.mu.Unlock()
		t.Fatalf("(%q,%v), want %q", reason, ok, reasonPatCeiling)
	}
	s.mu.Unlock()
	if fair != 1 || ceil != 1 {
		t.Fatalf("counters after the ceiling transition: fair=%d ceiling=%d, want 1/1 (a change of REASON is a new mechanism event)", fair, ceil)
	}
}

// TestPatDegradesToTheCeilingWithoutAModel: an engine whose residue model is disabled
// (sustained mis-calibration — a restart, a second EPP, traffic we do not see) has no
// honest COLD verdict, and the response is to claim nothing: PAT then behaves exactly
// like Q8 (ceiling only) instead of evicting on a guess. The run's residue_off gauge is
// what tells a reader that the arm ran in this state.
func TestPatDegradesToTheCeilingWithoutAModel(t *testing.T) {
	s := patScheduler(t, nil)
	now := goldenT0
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.eng()
	g.residue.setCapacity(1000)
	residueSeed(g, "a", 100)
	residueDrain(g, "hog", 960) // genuinely cold
	m := patMember(s, "a", 4, now.Add(-time.Hour), now)
	if _, ok := s.eligibleLocked(g, m, now); !ok {
		t.Fatal("fixture: the member must be fair-share-cold-eligible with a working model")
	}
	g.residue.disabled = true
	if reason, ok := s.eligibleLocked(g, m, now); ok {
		t.Fatalf("a DISABLED model still produced an early exit (%q): no verdict must mean no verdict", reason)
	}
	m.busyPeriods = patCeilingBusyPeriods
	if reason, ok := s.eligibleLocked(g, m, now); !ok || reason != reasonPatCeiling {
		t.Fatalf("the ceiling must still fire with the model disabled, got (%q,%v)", reason, ok)
	}
}

// TestPatEvictsOnlyTheCacheDeadMemberEndToEnd drives the REAL admission loop
// (syncWaiters → budget → rotate) on a full engine:
//
//	a seated member holds 940 of the 950-block budget, a newcomer needs a seat, and
//	the only difference between the two runs is the seated member's RESIDUE CLASS.
//
// The cache-dead member is rotated — with the PAT reason in the exit counter, and the
// waiter gets the seat. The cache-WARM member at the SAME tenure (4 busy periods) is
// NOT rotated, while a fixed Q4 would evict it on the spot (quantum-complete) and buy a
// cold re-prefill for nothing: that pair is the mechanism.
func TestPatEvictsOnlyTheCacheDeadMemberEndToEnd(t *testing.T) {
	run := func(t *testing.T, cold bool) (exits map[string]int64, seated bool) {
		t.Helper()
		s := patScheduler(t, nil)
		now := goldenT0
		seatWithBlocks(t, s, "patient", 940, now) // 940 of the 950-block budget: nothing else fits
		s.mu.Lock()
		g := s.eng()
		g.members["patient"].busyPeriods = 4 // fair share served
		// a member with a live request keeps the ENGINE-IDLE liveness rule out of the
		// way: that rule evicts somebody whenever the engine has nothing in flight and
		// nothing dispatchable while a waiter is blocked (A1's stall guard, not the rule
		// under test — and with 32 seats it almost never fires in a real run).
		g.members["busy"] = &member{id: "busy", engine: "engine-a", seatTime: now, lastActivity: now, inflight: 1}
		g.residue.setCapacity(1000)
		g.residue.reset("fixture") // the seat path already modelled the dispatch; rebuild
		residueSeed(g, "patient", 940)
		s.mu.Unlock()
		if cold {
			s.mu.Lock()
			residueDrain(s.eng(), "hog", 1000) // 60 anonymous slots + 940 pops
			s.mu.Unlock()
		}
		heads := []Head{{Member: "newcomer", ReqID: "newcomer-r1", Arrival: now.Add(time.Second), SizeBytes: bytesFor(20)}}
		if _, i := s.PickFromHeadsEngine(heads, now.Add(time.Second)); i < 0 {
			t.Log("the newcomer is queued (not seated)")
		}
		return s.Snapshot().Exits, s.Seated("newcomer")
	}

	exits, seated := run(t, true)
	if exits[reasonPatFairShareCold] != 1 {
		t.Fatalf("cache-dead run: exits = %v, want exactly one %q — the pause outlived the cache after fair share", exits, reasonPatFairShareCold)
	}
	if !seated {
		t.Fatal("cache-dead run: the waiter must get the freed seat")
	}
	exits, seated = run(t, false)
	if exits[reasonPatFairShareCold] != 0 || exits["quantum-complete"] != 0 || exits["idle"] != 0 {
		t.Fatalf("cache-warm run: exits = %v, want NO rotation — a warm member at fair share is the expensive eviction", exits)
	}
	if seated {
		t.Fatal("cache-warm run: the waiter must NOT be seated by evicting a warm member")
	}
}
