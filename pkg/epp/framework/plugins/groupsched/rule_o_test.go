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

// RULE O — THE EMPTY ENGINE ADMITS ONE (the oversize-request liveness fix).
//
// A head whose EXACT cost exceeds every engine's seat budget cannot fit an engine that is
// empty, because the budget is 0.95 of the engine's cache. The 0.95 margin is a safety
// margin on CONCURRENT residency, not a statement that one request is inadmissible: vLLM
// refuses to start unless the model's max-model-len fits its own KV capacity, so any request
// the engine accepts physically fits its WHOLE cache. Without this rule such a head waited
// for ever — a permanent stall of one branch, and (strict head-of-line) of every branch
// behind it.
//
// The rule, in the order the code applies it:
//
//  1. TRIGGER — the head's cost exceeds the budget on EVERY engine. Otherwise placement is
//     exactly what it was.
//  2. An EMPTY engine takes it, subject to the cold valve as usual (an empty engine holds no
//     charge, and a capped charge always fits the cap, so the valve is never the blocker).
//  3. With no empty engine, the ordinary demand-driven rotation runs on the engine placement
//     would pick anyway (most free budget, then id), and it CONVERGES on that one engine:
//     every eviction raises its free budget, so it stays the pick until it is empty — never a
//     spread of evictions across the pool.
//  4. The seat charges the FULL exact chain (the index stays exact; the engine reads over
//     budget, which is the truth — the branch uses nearly the whole GPU).
//  5. NO THRASH: enforcement never takes an engine's SOLE staying resident as a pressure
//     victim when its own footprint exceeds the budget. It leaves by TENURE (Q busy periods,
//     or S of silence when a waiter needs the room), like everyone else.
//  6. While it is seated the engine's free budget is negative, so nothing else seats there —
//     conservative, and exactly what the accounting should say.
//
// These tests are the six families: the empty-engine admit at N=1 and N=2, the convergence on
// one engine, the no-thrash passes, the tenure release when a waiter exists, and (in
// invariant_test.go) the randomized stream with oversize heads in it.

import (
	"fmt"
	"testing"
	"time"
)

// pickAndLand releases a turn for the head and brings it to a pod (the live order: Pick, a
// decision pass inside the window, then PreRequest).
func pickAndLand(t *testing.T, fx *engineSet, h Head) string {
	t.Helper()
	engine, i := fx.pickEngine(h)
	if i < 0 {
		t.Fatalf("the head must be served, got %d", i)
	}
	fx.s.DispatchedOn("", h.ReqID, engine, nil, fx.clock)
	return engine
}

// seatIncumbentLocked places a tenure-eligible resident on a NAMED engine directly: the
// convergence family needs both engines occupied, and the public seat path would choose the
// engine for us.
func seatIncumbent(fx *engineSet, engine, id string, blocks, busyPeriods int) {
	g := fx.s.groups[engine]
	g.index.TrySeatLocked(id, chainOf(id, blocks), g.budgetBlocks)
	g.branches[id] = &branch{id: id, engine: engine, seatTime: fx.clock,
		lastActivity: fx.clock, busyPeriods: busyPeriods, coldDone: true}
	fx.s.bindings[id] = engine
}

// settleTurn completes one turn of a branch the fixture dispatched by hand (one busy period).
func settleTurn(fx *engineSet, branch string, reqID string, blocks int) {
	fx.s.SettleOnUsage(branch, reqID, "", blocks*64, blocks*64, 0, true, fx.clock)
}

// TestRuleOEmptyEngineAdmitsOneOversizeHead is families 1, 4 and 6 at once: the oversize head
// is seated on an EMPTY engine, the seat charges its full exact chain (so the engine reads
// over budget), and nothing else seats on that engine while it is there — at N=2 the small
// head goes to the other engine, at N=1 it waits (family 5 is its own test).
func TestRuleOEmptyEngineAdmitsOneOversizeHead(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Second, int(budget)+250) // over the budget on every engine
		small := head("small", 500*time.Millisecond, 100)

		// ONE pass sees both heads: big is the queue head, the small one is behind it
		engine, i := fx.pickEngine(big, small)
		if i < 0 {
			t.Fatal("RULE O: an oversize head must be admitted on an EMPTY engine")
		}
		fx.s.DispatchedOn("", big.ReqID, engine, nil, fx.clock)

		if !fx.s.Seated("big") {
			t.Fatal("the oversize branch must hold a seat")
		}
		g := fx.s.groups[engine]
		if got, want := int64(g.index.UniqueBlocksLocked()), budget+250; got != want {
			t.Fatalf("the seat must charge the FULL exact chain: used %d, want %d", got, want)
		}
		if used := int64(g.index.UniqueBlocksLocked()); used <= budget {
			t.Fatalf("an oversize seat reads %d blocks against a %d-block budget: the engine must read OVER budget", used, budget)
		}

		// the small head: at N=2 it seats on the OTHER (still empty) engine; at N=1 it waits
		if len(fx.ids) == 1 {
			if fx.s.Seated("small") {
				t.Fatal("nothing may seat beside an oversize resident: its engine's free budget is negative")
			}
			if i := fx.pick(small); i >= 0 {
				t.Fatal("a small head must not be dispatchable while it cannot be seated")
			}
		} else {
			fx.pick(small) // the head is still queued: one more pass admits it elsewhere
			if !fx.s.Seated("small") {
				t.Fatal("at N=2 the small head must seat on the other engine")
			}
			if got := fx.s.bindings["small"]; got == engine {
				t.Fatalf("the small head seated beside the oversize resident (engine %s)", got)
			}
		}
	})
}

// TestRuleOEvictionsConvergeOnOneEngine is family 3: with incumbents everywhere and no empty
// engine, the evictions all land on ONE engine — the one placement picks (most free budget,
// then id) — whose free budget rises with each exit until it is empty and takes the head.
func TestRuleOEvictionsConvergeOnOneEngine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		type exit struct{ engine, reason string }
		var exits []exit
		fx.s.onExit = func(engine, reason string, _ int) { exits = append(exits, exit{engine, reason}) }

		// two tenure-eligible incumbents on EVERY engine, so any engine could be emptied
		fx.s.mu.Lock()
		for _, engine := range fx.ids {
			for j := 0; j < 2; j++ {
				seatIncumbent(fx, engine, fmt.Sprintf("inc-%s-%d", engine, j), 400, fx.s.cfg.Quantum)
			}
		}
		fx.s.mu.Unlock()

		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine, i := fx.pickEngine(big)
		if i < 0 {
			t.Fatal("RULE O: the head must be seated once an engine is freed for it")
		}
		fx.s.DispatchedOn("", big.ReqID, engine, nil, fx.clock)

		if !fx.s.Seated("big") {
			t.Fatal("the oversize head must hold the seat it freed")
		}
		if got := fx.s.bindings["big"]; got != engine {
			t.Fatalf("the head seated on %s, but placement picked %s", got, engine)
		}
		// CONVERGENCE: every eviction happened on the engine that took the head, and it is now
		// empty of incumbents. Nothing was taken from another engine for this one head.
		for _, e := range exits {
			if e.engine != engine {
				t.Fatalf("an eviction for ONE oversize head landed on %s, not on the engine that took it (%s): the order did not converge", e.engine, engine)
			}
		}
		if want := 2; len(exits) != want {
			t.Fatalf("the engine held 2 incumbents: want %d exits on it, got %d (%v)", want, len(exits), exits)
		}
		for _, other := range fx.ids {
			if other == engine {
				continue
			}
			g := fx.s.groups[other]
			if len(g.branches) != 2 {
				t.Fatalf("engine %s lost residents (%d left) for a head that seated elsewhere", other, len(g.branches))
			}
		}
	})
}

// TestRuleOWaitsForTenureNotPressure is family 5: once the oversize branch holds its engine,
// no amount of pressure passes touches it — not the enforcement sweep, not a waiting small
// head — and it is not even MARKED. Pressure would free the engine and admit nothing (the head
// it was admitted for is gone), so the pass is done; tenure is how it leaves.
func TestRuleOWaitsForTenureNotPressure(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine := pickAndLand(t, fx, big)
		if !fx.s.Seated("big") {
			t.Fatal("fixture: the oversize branch must be seated")
		}
		// the other engines are FULL, so a waiting head cannot escape to one of them (at N=1
		// there is none): the pressure this test applies is pressure on the oversize engine
		fx.closeEnginesExcept(t, engine, 950)

		var exits int
		fx.s.onExit = func(engine, reason string, _ int) { exits++ }

		// many passes with growth and a waiting small head in the queue
		for round := 0; round < 40; round++ {
			fx.clock = fx.clock.Add(100 * time.Millisecond)
			small := head(fmt.Sprintf("small-%d", round), time.Second, 100)
			fx.pick(small)
			// every few rounds the resident's own next turn grows into a new chain
			if round%7 == 3 {
				grown := headWithChain("big", time.Second, chainOf("big", int(budget)+400))
				grown.ReqID = fmt.Sprintf("r-big-%d", round)
				if i := fx.pick(grown); i >= 0 {
					fx.s.DispatchedOn("", grown.ReqID, engine, nil, fx.clock)
					settleTurn(fx, "big", grown.ReqID, int(budget)+400)
				}
			}
		}

		if !fx.s.Seated("big") {
			t.Fatal("RULE O: pressure must never take the engine's sole oversize resident")
		}
		if exits != 0 {
			t.Fatalf("the sole oversize resident was exited %d times: that is the thrash the rule forbids", exits)
		}
		g := fx.s.groups[engine]
		if g.index.ExitingLocked("big") {
			t.Fatal("the sole oversize resident was MARKED exiting: the mark is the thrash, one pass later")
		}
		// and no small head slipped onto the engine beside it
		for _, id := range []string{"small-0", "small-39"} {
			if fx.s.Seated(id) {
				t.Fatalf("%s seated on an engine whose free budget is negative", id)
			}
		}
	})
}

// TestRuleOTenureFreesTheEngineForAWaiter is family 5's other half at BOTH engine counts: the
// oversize branch leaves by TENURE once its Q busy periods are done, and the waiting head then
// takes the engine — the stall ends through the normal rotation, not through pressure.
func TestRuleOTenureFreesTheEngineForAWaiter(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		budget := int64(fx.g().budgetBlocks)
		big := head("big", time.Minute, int(budget)+250)
		engine := pickAndLand(t, fx, big)
		if !fx.s.Seated("big") {
			t.Fatal("fixture: the oversize branch must be seated")
		}
		// one engine only, at both N: the waiter must take THIS engine once it is free
		fx.closeEnginesExcept(t, engine, 950)

		small := head("small", time.Second, 100)
		if i := fx.pick(small); i >= 0 {
			t.Fatal("the small head cannot be seated beside the oversize resident")
		}

		// the entry turn completes, then Q further busy periods make the resident
		// rotation-eligible
		settleTurn(fx, "big", big.ReqID, int(budget)+250)
		for turn := 1; turn <= fx.s.cfg.Quantum; turn++ {
			h := headWithChain("big", time.Second, chainOf("big", int(budget)+250))
			h.ReqID = fmt.Sprintf("r-big-%d", turn)
			if i := fx.pick(h); i < 0 {
				t.Fatalf("turn %d: the seated branch's own turn must dispatch", turn)
			}
			fx.s.DispatchedOn("", h.ReqID, engine, nil, fx.clock)
			settleTurn(fx, "big", h.ReqID, int(budget)+250)
		}

		// the waiter is still there: the tenure rule now releases the resident, and it seats
		if i := fx.pick(small); i < 0 {
			t.Fatalf("the waiting head must take the engine the tenure rule released (engine %s)", engine)
		}
		if fx.s.Seated("big") {
			t.Fatal("the resident completed its quantum: it must have been rotated out")
		}
		if !fx.s.Seated("small") {
			t.Fatal("the waiting head must hold the seat the rotation freed")
		}
		if got := fx.s.bindings["small"]; got != engine {
			t.Fatalf("the waiter seated on %s, want the freed engine %s", got, engine)
		}
	})
}

// TestRuleOIsDeterministic: the same oversize scenario decides the same way twice (the
// placement tie-break and the eviction order are total orders, so nothing here reads map
// order).
func TestRuleOIsDeterministic(t *testing.T) {
	run := func() string {
		s, err := NewScheduler(defaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		fx := &engineSet{s: s, clock: time.Unix(1793000000, 0).UTC()}
		for i := 0; i < 2; i++ {
			id := fmt.Sprintf("e%d", i)
			fx.ids = append(fx.ids, id)
			if err := s.SetNumBlocksFor(id, 64, 1000); err != nil {
				t.Fatal(err)
			}
		}
		var trace string
		s.onExit = func(engine, reason string, _ int) { trace += engine + ":" + reason + ";" }
		fx.s.mu.Lock()
		for _, engine := range fx.ids {
			for j := 0; j < 2; j++ {
				seatIncumbent(fx, engine, fmt.Sprintf("inc-%s-%d", engine, j), 400, s.cfg.Quantum)
			}
		}
		fx.s.mu.Unlock()
		engine, i := fx.pickEngine(head("big", time.Minute, 1200))
		trace += fmt.Sprintf("seat=%s:%d", engine, i)
		return trace
	}
	first := run()
	for i := 0; i < 3; i++ {
		if again := run(); again != first {
			t.Fatalf("run %d decided differently: %q vs %q", i, again, first)
		}
	}
}
