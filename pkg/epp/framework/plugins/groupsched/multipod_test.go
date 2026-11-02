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
// the one with more free budget takes the seat.
func TestT1SeatGoesToTheMostFreeBudget(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		if len(fx.ids) < 2 {
			// N=1: the same code with one candidate — the head lands on it.
			h := head("m", time.Second, 100)
			if e, i := fx.pickEngine(h); i != 0 || e != fx.ids[0] {
				t.Fatalf("N=1: the only engine must take the head, got %q/%d", e, i)
			}
			return
		}
		// engine-a is 300 blocks fuller than engine-b, so b has more free budget
		fx.seatOn(t, fx.ids[0], "big", 300, 3*time.Second)
		// the resident's prefill lands, so its engine's cold-entry slot is free again
		// (the derived cap is 5 % of the engine, and an outstanding prefill owns it)
		fx.s.Chunk("big")
		h := head("m", time.Second, 100)
		e, i := fx.pickEngine(h)
		if i != 0 {
			t.Fatalf("the head must be seated and dispatched, got %d", i)
		}
		if e != fx.ids[1] {
			t.Fatalf("T1: the seat must go to the engine with the MOST FREE BUDGET (%s), got %s", fx.ids[1], e)
		}
	})
}

// T2: THE BINDING LASTS ONLY WHILE SEATED. A branch that exits is unbound; when it
// re-enters it is seated wherever there is room, which may be the OTHER engine.
func TestT2ExitUnbindsAndReentryReseatsAfresh(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 300, 3*time.Second)
		first := fx.s.bindings["m"]
		if first == "" {
			t.Fatal("a seated branch must be bound")
		}
		m, g := fx.branch(t, "m")
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
		h := head("m", 900*time.Second, 300)
		e, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("the re-entering branch must be seated somewhere")
		}
		if e == first {
			t.Fatalf("T2: the re-entry must be re-placed where there is room (%s), got the old engine %s", other, e)
		}
	})
}

// T3: THE e4 REGRESSION. Four seats, KV at 3 %, MANY queued/held heads: the engine
// keeps taking seats and NOTHING closes. (The E-wave's defect closed its gate on five
// waiting requests at 3 % KV — a raw count — and the engine then starved permanently.
// The queue term is gone (the net itself is gone with the trim), so the property this
// pins is narrower and still worth pinning: a deep vLLM queue is not a reason to stop
// seating.)
func TestT3e4RegressionNoLockInOnQueueDepth(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		// four seats, each with a queued follow-up head (the residents want to run)
		for i := 0; i < 4; i++ {
			id := fmt.Sprintf("s%d", i)
			fx.seatOne(t, id, 100, time.Duration(10+i)*time.Second)
			fx.s.Chunk(id) // each resident's prefill has landed: its cold slot is free
			m, _ := fx.branch(t, id)
			fx.s.mu.Lock()
			m.waitingSince = fx.clock.Add(-time.Duration(5+i) * time.Second)
			m.lastActivity = fx.clock
			fx.s.mu.Unlock()
		}
		fx.s.PickFromHeads(nil, fx.clock)
		before := fx.s.Snapshot()
		for i := 0; i < 6; i++ {
			id := fmt.Sprintf("q%d", i)
			h := head(id, time.Duration(60-i)*time.Second, 50)
			if _, j := fx.pickEngine(h); j < 0 {
				t.Fatalf("head %d must be seated: 3 %% KV and a deep waiting queue are not saturation", i)
			}
			// the head is dispatched by the same pass: its prefill lands, which is what
			// frees the engine's single cold-entry slot for the next newcomer
			fx.s.Chunk(id)
		}
		after := fx.s.Snapshot()
		if after.Seats <= before.Seats {
			t.Fatalf("T3: the policy must KEEP seating (%d -> %d seats)", before.Seats, after.Seats)
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
		// the SAME chain: two branches whose prompts share every block
		h := headWithChain(id, time.Second, []Chain{{Keys: shared, Tokens: 200 * 64}})
		if i := s.PickFromHeads([]Head{h}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		s.Dispatched(id, "r-"+id, now)
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

// ENGINE GONE: the engine's branches are exited ("engine-gone") and their heads go
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
		if i := s.PickFromHeads([]Head{head(id, time.Second, 300)}, now); i != 0 {
			t.Fatalf("%s must seat", id)
		}
		s.Dispatched(id, "r-"+id, now)
		s.Chunk(id)
		s.Settle(id, "r-"+id, struct{ Prompt, Cached int }{300, 300}, true, now)
	}
	gone := s.bindings["m1"]
	if n := s.RemoveEngine(gone, now); n == 0 {
		t.Fatal("the teardown must exit the engine's branches")
	}
	if snap := s.Snapshot(); snap.EngineGone != 1 {
		t.Fatalf("the engine-gone counter must be 1, got %d", snap.EngineGone)
	}
	if _, ok := s.groups[gone]; ok {
		t.Fatal("the group must be gone")
	}
	// a branch of the dead engine re-enters and must be seated on the survivor
	survivor := "a"
	if gone == "a" {
		survivor = "b"
	}
	e, i := s.PickFromHeadsEngine([]Head{head("m1", 30*time.Second, 300)}, now)
	if i < 0 {
		t.Fatal("the released branch must be seatable on the survivor")
	}
	if e != survivor {
		t.Fatalf("the re-entry must land on the survivor %s, got %s", survivor, e)
	}
}

// THE FILTER PINS DISPATCH TO THE SEAT ENGINE: one candidate, never empty, and a
// pass-through for unseated or anonymous branches.
func TestFilterPinsTheSeatEngine(t *testing.T) {
	cfg := defaultConfig()
	plugin := newTestPlugin(t, cfg, 2, 1000)
	s := plugin.sched
	now := time.Unix(1793000000, 0).UTC()
	if i := s.PickFromHeads([]Head{head("m", time.Second, 200)}, now); i != 0 {
		t.Fatal("the branch must seat")
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
	// unseated branch: pass through (no placement opinion yet)
	if got := plugin.Filter(t.Context(), requestFor("stranger"), pods); len(got) != 2 {
		t.Fatalf("an unseated branch must pass through, got %d", len(got))
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
	m, g := func() (*branch, *engineGroup) {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, gg := range s.groups {
			if mm, ok := gg.branches["m"]; ok {
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
		trace := runScenarioOn(t, s)
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
		// THE BUDGET, I1 + R4: admission counts the blocks an exiting branch still pins
		// (the engine holds them until its drain), so the USED side may exceed the budget
		// by exactly those and by nothing else. Enforcement acts on the STAY side and
		// evicts until that is inside the budget, so anything beyond the exiting blocks
		// here is a real over-commit.
		if es.NEBlocks > es.BudgetBlocks+es.ExitingBlocks {
			t.Fatalf("engine %s is over its budget at the end of the run: used %d > budget %d + exiting %d",
				es.ID, es.NEBlocks, es.BudgetBlocks, es.ExitingBlocks)
		}
		// the derived cold-entry cap is per engine and the charge never exceeds it
		if es.ColdCapTokens <= 0 {
			t.Fatalf("engine %s has no derived cold-entry cap", es.ID)
		}
		if es.ColdInFlight > es.ColdCapTokens {
			t.Fatalf("engine %s: cold charge %d exceeds its derived cap %d", es.ID, es.ColdInFlight, es.ColdCapTokens)
		}
	}
	if seats != snapTwo.Seats {
		t.Fatalf("the whole-policy seat count must be the per-engine sum (%d vs %d)", seats, snapTwo.Seats)
	}
}

// THE COLD VALVE NEVER EVICTS (the defect the core fixes): a head that fits an
// engine's BUDGET but is held by the cold-entry valve WAITS for the prefill slot.
// Evicting branches frees budget, never the valve, so every eviction taken on such a
// pass was wasted — and the engine's seats sat cold-empty until the prefill finished.
//
// The cap is derived per engine (5 % of 10,000 blocks = 500 blocks = 32,000 tokens),
// so a 600-block resident fills its engine's whole cold slot while leaving 8,900
// blocks of budget free: the valve is the only thing holding the next head.
func TestColdValveHeldHeadNeverEvicts(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		// ONE resident per ENGINE seats (its cold charge alone fills the cap, and the
		// placement spreads them one per engine because a closed valve is not a
		// candidate), and their prefills stay in flight (no Chunk), so EVERY engine's
		// valve has no room for another entry
		for i := range fx.ids {
			id := fmt.Sprintf("res%d", i)
			fx.seatOne(t, id, 600, time.Duration(5+i)*time.Second)
		}
		before := fx.s.Snapshot()
		// the residents' prefills are RUNNING (in flight, no chunk), which is what keeps
		// their cold charge live — and a running branch is never a rotation victim, so
		// nothing can be evicted for the head that waits.
		for attempt := 0; attempt < 4; attempt++ {
			h := head(fmt.Sprintf("cold%d", attempt), time.Second, 900)
			_, i := fx.pickEngine(h)
			if i >= 0 {
				t.Fatalf("attempt %d: a valve-held head must not be dispatched, got %d", attempt, i)
			}
		}
		after := fx.s.Snapshot()
		if after.Seats != before.Seats {
			t.Fatalf("THE VALVE MUST NEVER EVICT: seats moved %d -> %d for a head that only the valve held",
				before.Seats, after.Seats)
		}
		if len(after.Exits) != len(before.Exits) {
			t.Fatalf("no exit may be booked for a valve-held head: %v -> %v", before.Exits, after.Exits)
		}
		// R11 liveness: the valve opens when the residents' prefills complete
		for i := range fx.ids {
			fx.s.Chunk(fmt.Sprintf("res%d", i))
		}
		if _, i := fx.pickEngine(head("coldX", time.Second, 900)); i < 0 {
			t.Fatal("R11: once the cold prefills land, the waiting head must be seatable")
		}
	})
}

// AT N=2 THE HEAD MOVES TO THE SIBLING rather than waiting: engine A's valve refuses
// it, engine B both fits by budget AND allows it.
func TestColdValveHeldHeadSeatsOnTheSibling(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		if len(fx.ids) < 2 {
			t.Skip("needs two engines: the sibling is the point")
		}
		// one engine takes the first cold entry (fits AND allowed: nothing cold in
		// flight there), so ITS valve is now closed while the sibling's is open
		fx.seatOne(t, "coldA", 300, 5*time.Second)
		engA := fx.s.bindings["coldA"]
		// the next head fits the budget on both; A's valve refuses it and B's allows
		h := head("next", time.Second, 50)
		_, i := fx.pickEngine(h)
		if i < 0 {
			t.Fatal("the head must seat (on the SIBLING), not wait")
		}
		if got := fx.s.bindings["next"]; got == engA {
			t.Fatalf("the head was refused by %s's cold valve and must go to the sibling, not %s", engA, got)
		}
	})
}
