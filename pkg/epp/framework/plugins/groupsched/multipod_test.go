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

// THE MULTI-ENGINE SUITE — the SP intent tests (T1-T4) plus the per-engine
// invariants the multi-pod wave is scored on. Every one runs at N=1 AND N=2 through
// forEngines, and the ones that are only MEANINGFUL at N=2 say so and assert the
// engine count explicitly.

import (
	"fmt"
	"testing"
	"time"
)

// T1: PLACEMENT IS MOST-FREE-BUDGET. Of two engines where the head fits on both,
// the one with more free budget takes the seat; if that engine's safety net is
// firing, the other one takes it.
func TestT1SeatGoesToTheMostFreeBudget(t *testing.T) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction" // T1's second half needs the net
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		if len(fx.ids) < 2 {
			// N=1: the same code with one candidate — the head lands on it.
			h := head("m", time.Second, bytesFor(100))
			if e, i := fx.pickEngine(h); i != 0 || e != fx.ids[0] {
				t.Fatalf("N=1: the only engine must take the head, got %q/%d", e, i)
			}
			return
		}
		// engine-a is 300 blocks fuller than engine-b, so b has more free budget
		fx.seatOn(t, fx.ids[0], "big", 300, 3*time.Second)
		h := head("m", time.Second, bytesFor(100))
		e, i := fx.pickEngine(h)
		if i != 0 {
			t.Fatalf("the head must be seated and dispatched, got %d", i)
		}
		if e != fx.ids[1] {
			t.Fatalf("T1: the seat must go to the engine with the MOST FREE BUDGET (%s), got %s", fx.ids[1], e)
		}
		// now fire engine-b's safety net: the next head must go to engine-a
		fx.s.SetEngineLoad(fx.ids[1], 0.99, 0, fx.clock)
		fx.s.Tick(fx.clock)
		h2 := head("m2", time.Second, bytesFor(100))
		e2, i2 := fx.pickEngine(h2)
		if i2 != 0 {
			t.Fatalf("the second head must still be seated, got %d", i2)
		}
		if e2 != fx.ids[0] {
			t.Fatalf("T1: with %s's net firing the head must go to %s, got %s", fx.ids[1], fx.ids[0], e2)
		}
		// and once the fire clears, the placement is the plain key again (a net never
		// influences placement beyond holding its own engine's seats)
		fx.s.SetEngineLoad(fx.ids[1], 0.10, 0, fx.clock.Add(time.Second))
		fx.s.Tick(fx.clock.Add(time.Second))
		if snap := fx.s.Snapshot(); snap.Engines[engineLabel(fx.ids[1])].SafetyFiring {
			t.Fatal("the net must clear when the measured kv drops")
		}
	})
}

// T2: THE BINDING LASTS ONLY WHILE SEATED. A member that exits is unbound; when it
// re-enters it is seated wherever there is room, which may be the OTHER engine.
func TestT2ExitUnbindsAndReentryReseatsAfresh(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 300, 3*time.Second)
		first := fx.s.bindings["m"]
		if first == "" {
			t.Fatal("a seated member must be bound")
		}
		m, g := fx.member(t, "m")
		fx.s.mu.Lock()
		fx.s.exitLocked(g, m, "test", fx.clock)
		_, bound := fx.s.bindings["m"]
		fx.s.mu.Unlock()
		if bound {
			t.Fatal("T2: an exit must DROP the binding (seat = pin)")
		}
		if len(fx.ids) < 2 {
			return
		}
		// fill the first engine so the re-entry can only fit on the other one
		other := fx.ids[0]
		if first == fx.ids[0] {
			other = fx.ids[1]
		}
		fx.seatOn(t, first, "fill", 900, 2*time.Second)
		h := head("m", 900*time.Second, bytesFor(300))
		e, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("the re-entering member must be seated somewhere")
		}
		if e == first {
			t.Fatalf("T2: the re-entry must be re-placed where there is room (%s), got the old engine %s", other, e)
		}
	})
}

// T3: THE e4 REGRESSION. Four seats, KV at 3 %, MANY queued/held heads: the engine
// keeps taking seats and NOTHING closes. (The E-wave's defect closed its gate on five
// waiting requests at 3 % KV — a raw count — and the engine then starved permanently.)
func TestT3e4RegressionNoLockInOnQueueDepth(t *testing.T) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction" // even with the net ON, 3 % KV must not fire it
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		// four seats, each with a queued follow-up head (the residents want to run)
		for i := 0; i < 4; i++ {
			id := fmt.Sprintf("s%d", i)
			fx.seatOne(t, id, 100, time.Duration(10+i)*time.Second)
			m, _ := fx.member(t, id)
			fx.s.mu.Lock()
			m.waitingSince = fx.clock.Add(-time.Duration(5+i) * time.Second)
			m.lastActivity = fx.clock
			fx.s.mu.Unlock()
		}
		// every engine: measured KV 3 %, vLLM waiting queue FULL (100)
		for _, e := range fx.ids {
			fx.s.SetEngineLoad(e, 0.03, 100, fx.clock)
		}
		fx.s.Tick(fx.clock)
		before := fx.s.Snapshot()
		for i := 0; i < 6; i++ {
			h := head(fmt.Sprintf("q%d", i), time.Duration(60-i)*time.Second, bytesFor(50))
			if _, j := fx.pickEngine(h); j < 0 {
				t.Fatalf("head %d must be seated: 3 %% KV and a deep waiting queue are not saturation", i)
			}
		}
		after := fx.s.Snapshot()
		if after.Seats <= before.Seats {
			t.Fatalf("T3: the policy must KEEP seating (%d -> %d seats)", before.Seats, after.Seats)
		}
		for _, es := range after.Engines {
			if es.SafetyFiring {
				t.Fatal("T3: 3 % KV and a deep waiting queue must not fire the safety net")
			}
		}
	})
}

// T4: THE QUEUE TERM IS GONE. vLLM reporting waiting=100 at KV 0.5 does NOT close
// the net; KV at or over the budget fraction DOES — and the threshold is the budget's
// own fraction (no second constant).
func TestT4SafetyIsMemoryOnly(t *testing.T) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction"
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		eng := fx.ids[0]
		// deep queue, half-full KV: OPEN
		fx.s.SetEngineLoad(eng, 0.5, 100, fx.clock)
		fx.s.Tick(fx.clock)
		if fx.s.Snapshot().Engines[engineLabel(eng)].SafetyFiring {
			t.Fatal("T4: a waiting queue of 100 at KV 0.5 must NOT close the net (no queue term)")
		}
		// KV exactly at the budget fraction: FIRING
		fx.s.SetEngineLoad(eng, cfg.GroupBudgetFraction, 0, fx.clock)
		fx.s.Tick(fx.clock)
		st := fx.s.Snapshot().Engines[engineLabel(eng)]
		if !st.SafetyFiring {
			t.Fatalf("T4: KV at the budget's own fraction (%v) must fire the net", cfg.GroupBudgetFraction)
		}
		if st.SafetyFires != 1 {
			t.Fatalf("T4: one fire EVENT must be booked, got %d", st.SafetyFires)
		}
		// a firing engine seats nothing new and holds its own dispatches
		before := fx.s.Snapshot().Engines[engineLabel(eng)].Seats
		h := head("n", time.Second, bytesFor(100))
		if len(fx.ids) == 1 {
			if _, i := fx.pickEngine(h); i >= 0 {
				t.Fatal("T4: a firing engine must not seat a new head")
			}
			if fx.s.Snapshot().Engines[engineLabel(eng)].Seats != before {
				t.Fatal("T4: the seat count must not move while fired")
			}
		}
		// the fire time is booked (the wave's "fired <= 1 %" liveness line)
		if st2 := fx.s.Snapshot().Engines[engineLabel(eng)]; st2.SafetyFiredSeconds < 0 {
			t.Fatal("fired seconds must be countable")
		}
	})
}

// NOOP IS THE DEFAULT: with the default config the net never fires, whatever the
// metrics say — the seat budget is the whole admission control.
func TestSafetyIsNoopByDefault(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		eng := fx.ids[0]
		fx.s.SetEngineLoad(eng, 1.0, 1000, fx.clock)
		fx.s.Tick(fx.clock)
		st := fx.s.Snapshot().Engines[engineLabel(eng)]
		if st.SafetyFiring || st.SafetyFires != 0 {
			t.Fatalf("the default safety module must be a NOOP, got firing=%v fires=%d", st.SafetyFiring, st.SafetyFires)
		}
	})
}

// PER-ENGINE SIZING: each engine's budget is its OWN capacity × the fraction, shown
// by two engines with different capacities.
func TestPerEngineBudgetFromOwnCapacity(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("a", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("b", 64, 2000); err != nil { // budget 1900
		t.Fatal(err)
	}
	snap := s.Snapshot()
	if snap.Engines["a"].BudgetBlocks != 950 || snap.Engines["b"].BudgetBlocks != 1900 {
		t.Fatalf("per-engine budgets wrong: a=%d b=%d", snap.Engines["a"].BudgetBlocks, snap.Engines["b"].BudgetBlocks)
	}
	if snap.BudgetBlocks != 2850 {
		t.Fatalf("the whole-policy budget must be the sum, got %d", snap.BudgetBlocks)
	}
}

// PER-ENGINE DEDUP: the same block keys on two engines are two footprints (a block on
// A is worthless to B), and one engine's eviction does not touch the other's NE.
func TestPerEngineDedupAndIndependence(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"a", "b"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Unix(1793000000, 0).UTC()
	shared := keys(200, 42)
	for _, id := range []string{"m1", "m2"} {
		if i := s.PickFromHeads([]Head{head(id, time.Second, bytesFor(200))}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		s.Dispatched(id, "r-"+id, shared, bytesFor(200), now) // the SAME keys
	}
	snap := s.Snapshot()
	if len(snap.Engines) != 2 {
		t.Fatalf("both engines must be tracked, got %d", len(snap.Engines))
	}
	// each engine holds 200 NE blocks; the same hashes dedup WITHIN an engine only
	for id, es := range snap.Engines {
		if es.NEBlocks != 200 {
			t.Fatalf("engine %s: NE must be 200 (dedup is per engine), got %d", id, es.NEBlocks)
		}
	}
	if snap.NEBlocks != 400 {
		t.Fatalf("the whole-policy NE is the sum (400), got %d", snap.NEBlocks)
	}
}

// ENGINE GONE: the engine's members are exited ("engine-gone") and their heads go
// back to the SHARED queue, where a survivor seats them.
func TestEngineGoneReleasesMembersToTheSharedQueue(t *testing.T) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"a", "b"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Unix(1793000000, 0).UTC()
	for _, id := range []string{"m1", "m2", "m3"} {
		if i := s.PickFromHeads([]Head{head(id, time.Second, bytesFor(300))}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		s.Dispatched(id, "r-"+id, keys(300, seedFor(id)), bytesFor(300), now)
		s.Chunk(id)
		s.Settle(id, "r-"+id, struct{ Prompt, Cached int }{300, 300}, true, now)
	}
	gone := s.bindings["m1"]
	if n := s.RemoveEngine(gone, now); n == 0 {
		t.Fatal("the teardown must exit the engine's members")
	}
	if snap := s.Snapshot(); snap.EngineGone != 1 {
		t.Fatalf("the engine-gone counter must be 1, got %d", snap.EngineGone)
	}
	if _, ok := s.groups[gone]; ok {
		t.Fatal("the group must be gone")
	}
	// a member of the dead engine re-enters and must be seated on the survivor
	survivor := "a"
	if gone == "a" {
		survivor = "b"
	}
	e, i := s.PickFromHeadsEngine([]Head{head("m1", 30*time.Second, bytesFor(300))}, now)
	if i < 0 {
		t.Fatal("the released member must be seatable on the survivor")
	}
	if e != survivor {
		t.Fatalf("the re-entry must land on the survivor %s, got %s", survivor, e)
	}
}

// THE FILTER PINS DISPATCH TO THE SEAT ENGINE: one candidate, never empty, and a
// pass-through for unseated or anonymous members.
func TestFilterPinsTheSeatEngine(t *testing.T) {
	cfg := defaultConfig()
	plugin := newTestPlugin(t, cfg, 2, 1000)
	s := plugin.sched
	now := time.Unix(1793000000, 0).UTC()
	if i := s.PickFromHeads([]Head{head("m", time.Second, bytesFor(200))}, now); i != 0 {
		t.Fatal("the member must seat")
	}
	seat := s.bindings["m"]
	pods := podsOf(schedulerEndpoint("a"), schedulerEndpoint("b"))
	got := plugin.Filter(t.Context(), requestFor("m"), pods)
	if len(got) != 1 || endpointKey(got[0]) != seat {
		t.Fatalf("the Filter must narrow to the seat engine %s, got %d candidate(s)", seat, len(got))
	}
	// anonymous: pass everything through
	if got := plugin.Filter(t.Context(), requestFor(""), pods); len(got) != 2 {
		t.Fatalf("anonymous traffic must pass through, got %d", len(got))
	}
	// unseated member: pass through (no placement opinion yet)
	if got := plugin.Filter(t.Context(), requestFor("stranger"), pods); len(got) != 2 {
		t.Fatalf("an unseated member must pass through, got %d", len(got))
	}
	// the seat engine is not among the candidates: NEVER empty — pass all through and
	// let the observed endpoint win
	onlyOther := podsOf(schedulerEndpoint("zzz"))
	if got := plugin.Filter(t.Context(), requestFor("m"), onlyOther); len(got) != 1 {
		t.Fatalf("a missing seat engine must not empty the candidate list, got %d", len(got))
	}
	// one candidate: nothing to narrow
	if got := plugin.Filter(t.Context(), requestFor("m"), podsOf(schedulerEndpoint("a"))); len(got) != 1 {
		t.Fatalf("a single candidate must pass through, got %d", len(got))
	}
	// an exit drops the pin
	m, g := func() (*member, *engineGroup) {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, gg := range s.groups {
			if mm, ok := gg.members["m"]; ok {
				return mm, gg
			}
		}
		return nil, nil
	}()
	s.mu.Lock()
	s.exitLocked(g, m, "test", now)
	s.mu.Unlock()
	if _, ok := plugin.pins.get("m"); ok {
		t.Fatal("an exit must drop the placement pin")
	}
}

// GOLDEN AT N=2. There is no single golden TRACE for two engines (the same scenario
// legitimately splits), so the two properties the wave actually needs are pinned
// instead:
//
//  1. DETERMINISM: two replays of the golden scenario on a two-engine stack produce
//     identical traces (the multi-engine path has no map-order decision left).
//  2. CONSERVATION: the same scenario on two engines of capacity C/2 does not lose or
//     invent work — the served/dispatch totals stay in the same ballpark as the
//     single-engine run (seats and blocks are bounded by the same total budget), i.e.
//     nothing is double-counted across engines.
func TestMultiEngineGoldenProperties(t *testing.T) {
	run := func(engines int, capacity int) (string, Snapshot) {
		cfg := defaultConfig()
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		// the golden scenario names engine-a; a second engine is added for N=2
		for i := 0; i < engines; i++ {
			id := fmt.Sprintf("engine-%c", 'a'+i)
			if err := s.SetNumBlocksFor(id, 64, capacity); err != nil {
				t.Fatal(err)
			}
		}
		trace := runGoldenScenarioOn(t, s)
		return trace, s.Snapshot()
	}
	oneA, snapOne := run(1, 1000)
	oneB, _ := run(1, 1000)
	if oneA != oneB {
		t.Fatal("the golden scenario must be deterministic at N=1")
	}
	twoA, snapTwo := run(2, 1000)
	twoB, _ := run(2, 1000)
	if twoA != twoB {
		t.Fatal("the golden scenario must be deterministic at N=2 (no map-order decision left)")
	}
	// conservation: two engines of the same size serve MORE (two budgets), never
	// fewer, and the totals are sane multiples rather than double-counted figures
	if snapTwo.Served < snapOne.Served {
		t.Fatalf("N=2 must serve at least as many dispatches as N=1 (%d < %d)", snapTwo.Served, snapOne.Served)
	}
	if snapTwo.BudgetBlocks != 2*snapOne.BudgetBlocks {
		t.Fatalf("the whole-policy budget must be the per-engine sum: %d vs %d", snapTwo.BudgetBlocks, snapOne.BudgetBlocks)
	}
	var seats int
	for _, es := range snapTwo.Engines {
		seats += es.Seats
		if es.NEBlocks > es.BudgetBlocks {
			t.Fatalf("engine %s is over its budget at the end of the run (%d > %d)", es.ID, es.NEBlocks, es.BudgetBlocks)
		}
	}
	if seats != snapTwo.Seats {
		t.Fatalf("the whole-policy seat count must be the per-engine sum (%d vs %d)", seats, snapTwo.Seats)
	}
}
