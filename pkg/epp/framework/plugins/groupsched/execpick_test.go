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

// EXEC v2 — THE PICK-RACE FIX (PREREG-multipod-offload, "EXEC v2"). The execution
// budget's steady overrun (c128: 73.6 % of samples over budget, max 1.71) was mostly a
// RACE, not the estimate's bias:
//
//   - at Pick the winner got a PROVISIONAL inflight record that added NO blocks to the
//     running set ("the exact keys arrive at PreRequest"), and flow control calls Pick
//     about every 1 ms, so every turn inside the Pick -> PreRequest window saw the same
//     stale running count and passed the gate. Over-admission by construction.
//   - and nothing anywhere priced the OUTPUT GROWTH a dispatched turn allocates while it
//     decodes, so even a perfectly sequenced running set understated the engine's blocks.
//
// The three pieces these tests pin:
//
//	(1) RESERVE AT PICK     the estimate the gate admitted the turn with goes onto the
//	                        engine's used side immediately, is REPLACED by the exact keys
//	                        at PreRequest, and is RELEASED on every path where the
//	                        dispatch does not happen;
//	(2) GROWTH RESERVATION  ceil(max_tokens / blockSize) blocks per dispatched turn,
//	                        from the request's own cap, released at settle;
//	(3) the LEDGER           running keys + reservations + this head's charge <= budget,
//	                        at N=1 and at N=2 (a per-engine property is asserted at both).
//
// Everything here is invisible unless an arm asks for the execution budget: with the
// default selector nothing is maintained and the f0p replay stays byte-identical
// (TestExecutionBudgetDefaultIsGoldenNeutral / ...OnChangesNothingWhenItNeverBinds).

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// ---- helpers ----

// usedOf is the engine's DISPATCH used side as the GATE reads it (exact keys + charges).
func (fx *engineSet) usedOf(engine string) int64 {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.usedBlocksLocked(fx.s.groups[engine])
}

// reservedOf is the part of the used side that carries no keys.
func (fx *engineSet) reservedOf(engine string) int64 {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.groups[engine].reserved
}

// seatIdle seats one member and leaves it with NOTHING in flight: its first turn is
// dispatched and settled, so its cold-entry charge is behind it and the next Pick on this
// fixture is about the dispatch budget only.
func (fx *engineSet) seatIdle(t *testing.T, id string, blocks int) {
	t.Helper()
	if got := fx.seatAndDispatch(t, id, blocks, keys(blocks, seedFor(id))); got < 0 {
		t.Fatalf("%s must be seated and dispatched on an empty engine", id)
	}
	fx.settleAt(id, blocks)
}

// turn is the next queued turn for a member that is already seated and idle: same chain
// size as its residency, no output cap unless one is given.
func turn(id string, ago time.Duration, blocks int, maxTokens int64) Head {
	return Head{Member: id, ReqID: "t2-" + id, Arrival: time.Unix(1793000000, 0).UTC().Add(-ago),
		SizeBytes: bytesFor(blocks), MaxTokens: maxTokens}
}

// ---- (1) THE RACE ----

// TEN PICKS INSIDE ONE Pick -> PreRequest WINDOW NEVER EXCEED THE BUDGET.
//
// The fixture seats ten members, settles their first turns (so their cold-entry charges
// are behind them and nothing is in flight), then queues one turn each and runs ten Picks
// WITHOUT ever calling PreRequest — the window, held open. Before the fix every one of the
// ten passed: the running set was still empty because no exact keys had arrived.
//
// After the fix the used side moves at the moment the gate answers, so a turn is admitted
// only while the engine has room for it: floor(budget / per-turn) turns per engine, and
// the running KEY set is still empty the whole time — which is exactly the point (it is
// the reservation, not the keys, that holds the rest).
func TestExecutionBudgetPickReservationClosesTheRace(t *testing.T) {
	const (
		capacity = 1000 // dispatch budget 950
		tier     = 3000 // seat budget 0.95 x 3000, so all ten members are SEATED
		perTurn  = 200  // 4 turns fit (800), the 5th would be 1000 > 950
		members  = 10
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		budget := execBudgetOf(cfg.GroupBudgetFraction, capacity)
		wantPerEngine := int(budget) / perTurn

		ids := make([]string, 0, members)
		for i := 0; i < members; i++ {
			id := fmt.Sprintf("m%02d", i)
			ids = append(ids, id)
			fx.seatIdle(t, id, perTurn)
		}

		// THE WINDOW, HELD OPEN: ten Picks, no PreRequest.
		admitted := map[string]string{} // member -> engine it was served from
		for i, id := range ids {
			engine, idx := fx.pickEngine(turn(id, time.Duration(i+1)*time.Second, perTurn, 0))
			if idx >= 0 {
				admitted[id] = engine
			}
			// AFTER EVERY PICK: no engine's used side is over its budget.
			for _, e := range fx.ids {
				if used, keys := fx.usedOf(e), fx.runningSetSize(e); used > budget {
					t.Fatalf("pick %d: engine %s used side %d > budget %d (running keys %d) — the window over-admitted",
						i+1, e, used, budget, keys)
				}
			}
		}
		if want := wantPerEngine * len(fx.ids); len(admitted) != want {
			t.Fatalf("admits inside the window = %d, want %d (%d per engine x %d): the reservation must close the race",
				len(admitted), want, wantPerEngine, len(fx.ids))
		}
		for _, e := range fx.ids {
			if got := fx.runningSetSize(e); got != 0 {
				t.Fatalf("engine %s: no exact keys have arrived, yet the running key set is %d", e, got)
			}
			if got, want := fx.usedOf(e), int64(wantPerEngine*perTurn); got != want {
				t.Fatalf("engine %s: used side = %d, want %d (the %d reservations)", e, got, want, wantPerEngine)
			}
		}

		// THE REPLACEMENT, NOT AN ADDITION: PreRequest arrives for every admitted turn and
		// the exact keys take the reservation's place. The used side does not move.
		for _, id := range ids {
			engine, ok := admitted[id]
			if !ok {
				continue
			}
			fx.s.DispatchedOnUsage(id, "t2-"+id, engine, keys(perTurn, seedFor(id)), bytesFor(perTurn), 0, fx.clock)
			if got, want := fx.usedOf(engine), int64(wantPerEngine*perTurn); got != want {
				t.Fatalf("engine %s: after PreRequest the used side is %d, want %d — the reservation is REPLACED by the exact keys, never added to them",
					engine, got, want)
			}
		}
		for _, e := range fx.ids {
			if got := fx.reservedOf(e); got != 0 {
				t.Fatalf("engine %s: %d blocks still reserved after every admitted turn's exact keys arrived (a dispatched turn carries no charge when it declares no output cap)", e, got)
			}
			if got, want := fx.runningSetSize(e), wantPerEngine*perTurn; got != want {
				t.Fatalf("engine %s: running keys = %d, want %d", e, got, want)
			}
		}

		// and the turns that were held are STILL held: nothing freed.
		for _, id := range ids {
			if _, ok := admitted[id]; ok {
				continue
			}
			if _, idx := fx.pickEngine(turn(id, time.Second, perTurn, 0)); idx >= 0 {
				t.Fatalf("%s was admitted after the window closed with every engine full", id)
			}
		}

		// a settle frees exactly one turn's room, and the next Pick takes it. The settle
		// carries the id this turn DISPATCHED under (t2-...), not the residency's first one.
		first := ""
		for _, id := range ids {
			if _, ok := admitted[id]; ok {
				first = id
				break
			}
		}
		engine := fx.s.bindings[first]
		fx.s.ChunkAt(first, fx.clock)
		fx.s.SettleOnUsage(first, "t2-"+first, engine, perTurn, perTurn, 0, true, fx.clock)
		if got, want := fx.usedOf(engine), int64((wantPerEngine-1)*perTurn); got != want {
			t.Fatalf("engine %s: used side after a settle = %d, want %d", engine, got, want)
		}
		held := ""
		for _, id := range ids {
			if _, ok := admitted[id]; !ok && fx.s.bindings[id] == engine {
				held = id
				break
			}
		}
		if held == "" {
			t.Fatalf("engine %s must still be holding a turn (5 members seated, 4 admitted)", engine)
		}
		if _, idx := fx.pickEngine(turn(held, time.Second, perTurn, 0)); idx < 0 {
			t.Fatal("a settled turn frees exactly one turn's room and the held turn must take it (the gate re-checks every pass)")
		}
		if got, want := fx.usedOf(engine), int64(wantPerEngine*perTurn); got != want {
			t.Fatalf("engine %s: after the held turn was admitted the used side is %d, want %d", engine, got, want)
		}
		fx.checkRunning(t, "after the window closed and refilled")
	})
}

// THE RELEASE, ON EVERY PATH WHERE THE DISPATCH DOES NOT HAPPEN. A reservation that is not
// released is not a conservative estimate: it is a block the engine will never use and can
// never get back (until the reap), which is how a gate becomes a leak. One subtest per path
// the scheduler can take between Pick and a first response chunk — plus the two paths that
// end a record without any settle at all.
func TestExecutionBudgetReservationReleasedOnEveryNonDispatchPath(t *testing.T) {
	const (
		capacity = 1000
		tier     = 3000
		perTurn  = 200
	)
	cfg := execBudgetConfig(capacity, tier)
	for _, tc := range []struct {
		name string
		// run ends the request without a dispatch and returns nothing; the test asserts the
		// used side is back to zero afterwards.
		run func(t *testing.T, fx *engineSet, engine, id string)
	}{
		{"client-cancel", func(t *testing.T, fx *engineSet, engine, id string) {
			// the client went away before the turn ran: the framework's forced EndOfStream
			// arrives as a non-natural settle with the request id it dispatched under.
			fx.s.SettleOnUsage(id, "t2-"+id, "", 0, 0, 0, false, fx.clock)
		}},
		{"error-before-first-chunk", func(t *testing.T, fx *engineSet, engine, id string) {
			// an error upstream: the same non-natural settle, but with NO request id — the
			// path that must fall back to the member's oldest record instead of silently
			// keeping the reservation alive.
			fx.s.SettleOnUsage(id, "", "", 0, 0, 0, false, fx.clock)
		}},
		{"re-pick", func(t *testing.T, fx *engineSet, engine, id string) {
			// the same turn is picked again (a retry inside the framework): the first
			// reservation must come out with the record it belonged to.
			if _, idx := fx.pickEngine(turn(id, time.Second, perTurn, 0)); idx < 0 {
				t.Fatalf("%s must still be admitted on a used side of one turn", id)
			}
			if got := fx.reservedOf(engine); got != perTurn {
				t.Fatalf("a re-picked turn reserved twice: %d blocks on %s, want %d", got, engine, perTurn)
			}
			// and one release clears it (the record is one, not two)
			fx.s.SettleOnUsage(id, "t2-"+id, "", 0, 0, 0, false, fx.clock)
		}},
		{"unobserved-termination-reap", func(t *testing.T, fx *engineSet, engine, id string) {
			// nothing observed at all: the provisional record (and its reservation) is
			// reaped at its 30 s bound.
			fx.tick(31 * time.Second)
		}},
		{"member-exit", func(t *testing.T, fx *engineSet, engine, id string) {
			fx.s.mu.Lock()
			for _, g := range fx.allGroups() {
				if m, ok := g.members[id]; ok {
					fx.s.exitLocked(g, m, "test", fx.clock)
				}
			}
			fx.s.mu.Unlock()
		}},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
				fx.seatIdle(t, "m", perTurn)
				engine, idx := fx.pickEngine(turn("m", time.Second, perTurn, 0))
				if idx < 0 {
					t.Fatal("an empty engine's first turn must be admitted")
				}
				if got := fx.usedOf(engine); got != perTurn {
					t.Fatalf("after the Pick the used side is %d, want the reservation %d", got, perTurn)
				}
				tc.run(t, fx, engine, "m")
				if got := fx.usedOf(engine); got != 0 {
					t.Fatalf("%s: the reservation was not released — used side %d on engine %s, want 0", tc.name, got, engine)
				}
				if got := fx.runningSetSize(engine); got != 0 {
					t.Fatalf("%s: running keys = %d, want 0", tc.name, got)
				}
				fx.checkRunning(t, "after "+tc.name)
			})
		})
	}
}

// ---- (2) THE GROWTH RESERVATION ----

// THE GROWTH TERM IS COUNTED AND RELEASED. A dispatched turn allocates
// ceil(max_tokens / blockSize) blocks of KV while it decodes; under the harness's
// ignore_eos that count is exact, and it is blocks the engine holds whether or not
// anything is in the running key set.
//
// The second half is the one that matters for the overrun: a turn whose CHAIN alone fits
// is HELD when its own growth does not — the case the old ledger could not see at all.
func TestExecutionBudgetGrowthReservation(t *testing.T) {
	const (
		capacity = 1000 // dispatch budget 950
		tier     = 3000
		blocks   = 300
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		// 640 max_tokens / 64-token blocks = exactly 10 blocks of decode growth; 650
		// rounds UP to 11 (a partial block still occupies a block).
		const (
			maxTok   = 640
			wantGrow = 10
		)
		if got := fx.s.growthBlocksLocked(maxTok); got != wantGrow {
			t.Fatalf("growth for %d tokens at blockSize %d = %d, want %d", maxTok, fx.s.blockSize, got, wantGrow)
		}
		if got := fx.s.growthBlocksLocked(maxTok + 9); got != wantGrow+1 {
			t.Fatalf("a partial trailing block must round UP: %d, want %d", got, wantGrow+1)
		}
		if got := fx.s.growthBlocksLocked(0); got != 0 {
			t.Fatalf("a turn with no declared cap reserves no growth, got %d", got)
		}

		fx.seatIdle(t, "m", blocks)
		engine, idx := fx.pickEngine(turn("m", time.Second, blocks, maxTok))
		if idx < 0 {
			t.Fatal("the empty engine's first turn must be admitted")
		}
		if got, want := fx.usedOf(engine), int64(blocks+wantGrow); got != want {
			t.Fatalf("after the Pick the used side is %d, want %d (the chain estimate PLUS its %d-block growth)", got, want, wantGrow)
		}
		// PreRequest: the exact keys replace the chain estimate; the growth stays.
		fx.s.DispatchedOnUsage("m", "t2-m", engine, keys(blocks, seedFor("m")), bytesFor(blocks), maxTok, fx.clock)
		if got := fx.runningSetSize(engine); got != blocks {
			t.Fatalf("running keys = %d, want %d", got, blocks)
		}
		if got, want := fx.reservedOf(engine), int64(wantGrow); got != want {
			t.Fatalf("the growth reservation = %d, want %d (it is the only charge a dispatched turn carries)", got, want)
		}
		if got, want := fx.usedOf(engine), int64(blocks+wantGrow); got != want {
			t.Fatalf("used side = %d, want %d", got, want)
		}
		// and the exported numbers say the same thing
		if es := fx.s.Snapshot().Engines[engineLabel(engine)]; es.ReservedBlocks != wantGrow {
			t.Fatalf("snapshot reservedBlocks = %d, want %d", es.ReservedBlocks, wantGrow)
		}
		// SETTLE releases it
		fx.s.ChunkAt("m", fx.clock)
		fx.s.SettleOnUsage("m", "t2-m", "", blocks, blocks, maxTok, true, fx.clock)
		if got := fx.usedOf(engine); got != 0 {
			t.Fatalf("after the settle the used side is %d, want 0: the growth is released with the turn", got)
		}
		fx.checkRunning(t, "after the growth turn")
	})
}

// THE GROWTH TERM ALONE CAN HOLD A TURN. Two turns of the same chain cost the same, so
// the only difference between "runs" and "waits" here is the decode growth the old
// accounting never priced.
func TestExecutionBudgetGrowthAloneHoldsATurn(t *testing.T) {
	const (
		capacity = 1000 // dispatch budget 950
		tier     = 3000
		running  = 700 // a resident turn of 700 blocks
		next     = 200 // the next turn's chain: 700 + 200 = 900 fits, its growth does not
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		if got := fx.seatAndDispatch(t, "a", running, keys(running, seedFor("a"))); got < 0 {
			t.Fatal("the first turn on an empty engine must run")
		}
		engine := fx.s.bindings["a"]
		// every OTHER engine is closed, so BOTH turns below are on a's engine (at N=2 an
		// open second engine would serve b's turn by liveness, and the fixture would be
		// measuring the wrong engine)
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		fx.seatIdle(t, "b", next)
		// 700 + 200 = 900 leaves 50 blocks of room, so a cap of 3840 tokens (60 blocks of
		// decode) is what tips the turn over: the CHAIN fits, the GROWTH does not.
		const maxTok = 3840
		grow := fx.s.growthBlocksLocked(maxTok)
		if int64(running+next)+grow <= execBudgetOf(cfg.GroupBudgetFraction, capacity) {
			t.Fatalf("the fixture must make the GROWTH the binding term: %d + %d + %d vs %d",
				running, next, grow, execBudgetOf(cfg.GroupBudgetFraction, capacity))
		}
		if _, idx := fx.pickEngine(turn("b", time.Second, next, 0)); idx < 0 {
			t.Fatal("WITHOUT a cap the same turn must fit (700 + 200 <= 950): the fixture is wrong, not the gate")
		}
		// undo that admission (it was the control) and re-ask with the cap
		fx.s.SettleOnUsage("b", "t2-b", "", 0, 0, 0, false, fx.clock)
		if _, idx := fx.pickEngine(turn("b", time.Second, next, maxTok)); idx >= 0 {
			t.Fatalf("a turn whose %d-block decode growth overruns the budget must WAIT (its chain alone fits)", grow)
		}
		if got := fx.usedOf(engine); got != running {
			t.Fatalf("a held turn must not reserve: used side %d, want %d", got, running)
		}
	})
}

// ---- (3) golden neutrality ----

// THE TWO GOLDEN CLAIMS, restated where the fix lives:
//
//   - the DEFAULT arm is byte-identical (nothing is maintained, no reservation, no growth
//     term: the f0p replay is the committed one);
//   - the MODULE ON A DEPLOYMENT WITH ROOM is byte-identical as well — the reservations
//     close a window that never binds, so not one decision moves.
func TestExecutionBudgetPickReservationGoldenNeutral(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "golden-f0p-bugfixes.txt"))
	if err != nil {
		t.Fatalf("f0p golden missing: %v", err)
	}
	if got := runGoldenScenario(t); got != string(want) {
		t.Fatal("the DEFAULT config's trace moved: the Pick reservation must be invisible unless an arm asks for the execution budget")
	}
	const huge = 200000
	off := runGoldenAt(t, defaultConfig(), huge)
	on := runGoldenAt(t, execConfig(defaultConfig()), huge)
	if off != on {
		t.Fatal("with the module ON and the running set always inside the budget, the reservation changed a decision")
	}
}

// ---- (4) the export ----

// DISPATCH_HOLDS_TOTAL MOVES. The pass counter used to be instantiated with Add(0) and
// incremented nowhere: the series existed on every arm and could never leave zero, so the
// execution budget's measured cost was invisible. It is now exported by DELTA from the
// scheduler's own cumulative per-engine count.
func TestExecutionBudgetHoldsAreExported(t *testing.T) {
	const (
		capacity = 1000
		tier     = 3000
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		reg := prometheus.NewRegistry()
		m := newGroupMetrics(fx.s)
		if err := m.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		if got := fx.seatAndDispatch(t, "a", 800, keys(800, seedFor("a"))); got < 0 {
			t.Fatal("the first turn on an empty engine must run")
		}
		engine := fx.s.bindings["a"]
		// every OTHER engine is closed, so b is SEATED on a's engine — where 800 blocks are
		// already running, so 800 + 300 over the 950-block dispatch budget: held every pass
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		if _, idx := fx.pickEngine(turn("b", time.Second, 300, 0)); idx >= 0 {
			t.Fatalf("b's turn must be HELD on %s (800 running + 300)", engine)
		}
		if got := fx.s.bindings["b"]; got != engine {
			t.Fatalf("b must be seated on a's engine %s (that is the state the gate holds), got %s", engine, got)
		}
		m.Refresh(0)
		if got, ok := gatherValue(t, reg, subsystem+"_dispatch_holds_total", map[string]string{"engine": engineLabel(engine)}); !ok || got < 1 {
			t.Fatalf("dispatch_holds_total = %v (present=%v) on %s: the pass count must be exported, not merely instantiated",
				got, ok, engineLabel(engine))
		}
		if got, ok := gatherValue(t, reg, subsystem+"_execution_reserved_blocks", map[string]string{"engine": engineLabel(engine)}); !ok || got != 0 {
			t.Fatalf("execution_reserved_blocks = %v (present=%v): a held turn reserves nothing, and the series must exist regardless", got, ok)
		}
		// the counter follows the scheduler's own count, pass for pass
		fx.s.mu.Lock()
		want := fx.s.groups[engine].counters.dispatchHolds
		fx.s.mu.Unlock()
		m.Refresh(0)
		got, ok := gatherValue(t, reg, subsystem+"_dispatch_holds_total", map[string]string{"engine": engineLabel(engine)})
		if !ok || got != float64(want) {
			t.Fatalf("dispatch_holds_total = %v, want the scheduler's own count %d", got, want)
		}
	})
}
