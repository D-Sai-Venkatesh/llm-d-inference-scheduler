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

// THE EXECUTION BUDGET (PREREG-multipod-offload, "BATCH 2 + THE EXECUTION BUDGET"): the
// second of the two budgets.
//
//	SEATING  (residency) = fraction x (num_gpu_blocks + tierBlocks)  — the SIZING module,
//	                                                                  unchanged
//	DISPATCH (execution) = fraction x  num_gpu_blocks                — the DISPATCH module
//	                                                                  ("execution-budget")
//
// The measured defect it answers, batch 2 at c64: the tier-sized seat budget let many
// members be SEATED on one engine, and seated members RUN; their running KV exceeded the
// GPU, so vLLM queued and preempted them (43 preemptions vs 0, TTFT p50 0.9 s -> 17.8 s,
// newcomer p90 9.9 s -> 142 s). The CPU tier widens where IDLE chains live, not how many
// requests may RUN at once.
//
// Every property below runs at N=1 AND at N=2: the gate is per engine by construction, so
// N is a case of the code and not a branch of it. The fixtures set sizing "gpu-plus-tier"
// with a tier — the arm this module ships for — because that is the only configuration in
// which the SEAT budget can exceed the GPU, and therefore the only one where the gate can
// bind for long.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ---- fixtures ----

// execBudgetConfig is the default config with the EXECUTION BUDGET on. The tier is what
// lets the seat budget outgrow the GPU:
// seat = 0.95 x (capacity + tier), dispatch = 0.95 x capacity.
func execBudgetConfig(capacity, tier int) Config {
	c := defaultConfig()
	c.ExecutionBudget = "execution-budget"
	c.Sizing = "gpu-plus-tier"
	c.KvBlockBytes = 4 * 64 // the index's 4-bytes/token convention (see sizetier_test.go)
	c.CpuTierBytes = int64(tier) * c.KvBlockBytes
	return c
}

// execConfig turns the execution budget on for a given config (used for the golden replay,
// which keeps the GPU-sized default seat budget).
func execConfig(c Config) Config {
	c.ExecutionBudget = "execution-budget"
	return c
}

// execBudgetOf is the rule under test, written the way the deployment reads it.
func execBudgetOf(fraction float64, numBlocks int) int64 {
	return int64(fraction * float64(numBlocks))
}

// seatBudgetOf is the OTHER budget (unchanged, the sizing module's): fraction x (GPU + tier).
func seatBudgetOf(cfg Config, capacity, tier int) int64 {
	return int64(cfg.GroupBudgetFraction * float64(capacity+tier))
}

// newFixture builds a scheduler with n engines of the given measured capacity — the twin
// rig of forEnginesCap, for the tests that must run TWO configurations over the SAME
// geometry (the arm with the module and the arm without it).
func newFixture(t *testing.T, cfg Config, n, capacity int) *engineSet {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	fx := &engineSet{s: s, clock: time.Unix(1793000000, 0).UTC()}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("e%d", i)
		fx.ids = append(fx.ids, id)
		if err := s.SetNumBlocksFor(id, 64, capacity); err != nil {
			t.Fatal(err)
		}
	}
	fx.now = fx.clock
	return fx
}

// forNs runs fn at N=1 and N=2 (the suite's rule: a per-engine property is asserted at
// both, because "N engines are the N=1 code run N times" is a claim, not a comment).
func forNs(t *testing.T, cfg Config, capacity int, fn func(t *testing.T, fx *engineSet)) {
	t.Helper()
	for _, n := range []int{1, 2} {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			fn(t, newFixture(t, cfg, n, capacity))
		})
	}
}

// runningSetSize is an engine's running set from the EPP's own accounting.
func (fx *engineSet) runningSetSize(engine string) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.groups[engine].index.RunningBlocksLocked()
}

// seatAndDispatch queues one head, lets the Pick seat it, and — when the Pick returned it —
// drives the released-request path with an EXPLICIT chain (the fixtures control which
// blocks two members share; dedup is the point of one test) and the first response chunk,
// which releases the cold-entry charge so the NEXT member can be seated (the derived valve
// cap is 5 % of the engine, and one outstanding prefill owns it).
func (fx *engineSet) seatAndDispatch(t *testing.T, id string, blocks int, chain []Hash) int {
	t.Helper()
	i := fx.pick(head(id, 30*time.Second, bytesFor(blocks)))
	if i >= 0 {
		fx.s.DispatchedOn(id, "r-"+id, "", chain, bytesFor(blocks), fx.clock)
		fx.s.ChunkAt(id, fx.clock) // prefill complete: the cold-entry slot is free again
	}
	return i
}

// settleAt is the natural termination of one in-flight request on the fixture's clock.
func (fx *engineSet) settleAt(id string, prompt int) {
	fx.s.ChunkAt(id, fx.clock)
	fx.s.SettleOnUsage(id, "r-"+id, "", prompt, prompt, 0, true, fx.clock)
}

// closeOtherEngines seats ONE filler per engine other than the first, sized to fill that
// engine's tier-sized SEAT budget exactly. The most-free-budget placement routes each
// filler to the only engine where it still fits, so from here on every new member is
// seated on the first engine — which is what a test about ONE engine's running set needs
// (at N=2 a second member would otherwise be seated on the idle engine, where the gate has
// nothing to hold).
func (fx *engineSet) closeOtherEngines(t *testing.T, seatBudget int64) {
	t.Helper()
	for i, e := range fx.ids {
		if i == 0 {
			continue
		}
		id := "closed-" + e
		if got := fx.seatAndDispatch(t, id, int(seatBudget), keys(int(seatBudget), seedFor(id))); got < 0 {
			t.Fatalf("the filler for %s must be seated and dispatched (it is the first turn on that engine)", e)
		}
		if fx.s.bindings[id] != e {
			t.Fatalf("the filler must close %s: it was seated on %s", e, fx.s.bindings[id])
		}
	}
}

// checkRunning is the ACCOUNTING INVARIANT: every engine's maintained running set equals
// the union re-derived from the scheduler's own in-flight records (the tick's heal).
func (fx *engineSet) checkRunning(t *testing.T, when string) {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	for _, g := range fx.allGroups() {
		want := fx.s.deriveRunningLocked(g)
		if !sameCounts(want, g.index.blocksRun) {
			t.Fatalf("running set drift %s on engine %s: maintained %v != derived %v", when, g.id, g.index.blocksRun, want)
		}
	}
}

// ---- (1) the fix ----

// A TURN WHOSE CHAIN WOULD PUSH THE RUNNING SET OVER THE GPU BUDGET **WAITS AT THE EPP**,
// and dispatches when a running request settles. The wait has moved out of vLLM's
// preempting queue, where it was unordered and cost a recompute.
func TestExecutionBudgetTurnWaitsThenDispatches(t *testing.T) {
	const (
		capacity = 1000 // GPU blocks: the DISPATCH budget is 950
		tier     = 3000 // the SEAT budget is 0.95 x 4000 = 3800, so both turns can be seated
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		budget := execBudgetOf(cfg.GroupBudgetFraction, capacity)
		// A's turn runs: 600 blocks of running set.
		if i := fx.seatAndDispatch(t, "a", 600, keys(600, seedFor("a"))); i < 0 {
			t.Fatal("the first turn on an idle engine must run (the gate's liveness rule)")
		}
		engine := fx.s.bindings["a"]
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		if got := fx.runningSetSize(engine); got != 600 {
			t.Fatalf("the running set must be the dispatched chain: %d blocks, wanted 600", got)
		}
		// B is SEATED (residency is untouched: 600 + 500 is inside the tier-sized seat
		// budget) but its turn does not fit the EXECUTION budget: 600 + 500 > 950.
		if i := fx.seatAndDispatch(t, "b", 500, keys(500, seedFor("b"))); i >= 0 {
			t.Fatalf("B dispatched into a running set of 600 blocks (budget %d): the execution budget must hold the turn at the EPP", budget)
		}
		if fx.s.bindings["b"] != engine {
			t.Fatalf("B must be SEATED on %s while its turn waits — that is the state the second budget creates", engine)
		}
		if got := fx.runningSetSize(engine); got != 600 {
			t.Fatalf("a held turn must not grow the running set: %d blocks, wanted 600", got)
		}
		// A settles: 600 blocks stop RUNNING, and the next pass dispatches B.
		fx.settleAt("a", 600)
		if got := fx.runningSetSize(engine); got != 0 {
			t.Fatalf("a settled request's blocks are not running any more: %d blocks", got)
		}
		if i := fx.seatAndDispatch(t, "b", 500, keys(500, seedFor("b"))); i < 0 {
			t.Fatal("with the running request settled, the waiting turn must dispatch (the gate re-checks every pass)")
		}
		if got := fx.runningSetSize(engine); got != 500 {
			t.Fatalf("after B's turn ran the running set must be 500, got %d", got)
		}
	})
}

// (1b) ONE BIT OF DIFFERENCE, ONE BEHAVIOUR: the same fixture and the same turn, held with
// the module on and dispatched with it off. The gate is a function of the running set and of
// nothing else.
func TestExecutionBudgetIsTheOnlyDifference(t *testing.T) {
	const (
		capacity = 1000
		tier     = 3000
	)
	for _, on := range []bool{false, true} {
		on := on
		t.Run(fmt.Sprintf("executionBudget=%v", on), func(t *testing.T) {
			cfg := execBudgetConfig(capacity, tier)
			if !on {
				cfg.ExecutionBudget = "none"
			}
			forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
				fx.seatAndDispatch(t, "a", 800, keys(800, seedFor("a"))) // 800 running
				fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
				i := fx.seatAndDispatch(t, "b", 300, keys(300, seedFor("b")))
				if on && i >= 0 {
					t.Fatal("with the execution budget on, a turn that pushes the running set over the GPU budget must WAIT")
				}
				if !on && i < 0 {
					t.Fatal("with the execution budget off, the same turn must dispatch (the default arm is unchanged)")
				}
			})
		})
	}
}

// ---- (2) liveness ----

// ONE REQUEST IS ALWAYS ALLOWED when nothing is in flight on the engine — even a chain
// larger than the whole dispatch budget. Without the rule an engine could wait for itself
// (and the first turn after an idle period could never start).
func TestExecutionBudgetLivenessOnAnIdleEngine(t *testing.T) {
	const (
		capacity = 1000 // dispatch budget 950
		tier     = 3000
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		// a chain three times the dispatch budget, seated under the tier-sized seat budget
		if i := fx.seatAndDispatch(t, "huge", 2900, keys(2900, seedFor("huge"))); i < 0 {
			t.Fatal("the first turn on an idle engine must run whatever its size")
		}
		engine := fx.s.bindings["huge"]
		if got, budget := fx.runningSetSize(engine), execBudgetOf(cfg.GroupBudgetFraction, capacity); int64(got) <= budget {
			t.Fatalf("the fixture must make the running set exceed the budget: %d vs %d", got, budget)
		}
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		// the NEXT turn on that (now busy) engine waits
		if i := fx.seatAndDispatch(t, "other", 100, keys(100, seedFor("other"))); i >= 0 {
			t.Fatalf("with %d blocks running, a second turn on that engine must wait", fx.runningSetSize(engine))
		}
	})
}

// ---- (3) no effect when it fits ----

// NO EFFECT WHEN THE RUNNING SET FITS: on an engine with room the gate holds nothing, and
// every decision is the one the arm WITHOUT the module makes.
func TestExecutionBudgetNoEffectWhenTheRunningSetFits(t *testing.T) {
	const (
		capacity = 1000
		tier     = 3000
	)
	for _, n := range []int{1, 2} {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			script := func(fx *engineSet) []bool {
				var got []bool
				fx.seatAndDispatch(t, "a", 300, keys(300, seedFor("a"))) // 300 of 950
				fx.seatAndDispatch(t, "b", 300, keys(300, seedFor("b"))) // 600 of 950
				got = append(got, fx.seatAndDispatch(t, "a", 300, keys(300, seedFor("a"))) >= 0)
				got = append(got, fx.seatAndDispatch(t, "b", 300, keys(300, seedFor("b"))) >= 0)
				return got
			}
			offCfg := execBudgetConfig(capacity, tier)
			offCfg.ExecutionBudget = "none"
			off := script(newFixture(t, offCfg, n, capacity))
			on := script(newFixture(t, execBudgetConfig(capacity, tier), n, capacity))
			if fmt.Sprint(off) != fmt.Sprint(on) {
				t.Fatalf("the arms disagree although the running set fits: off=%v on=%v", off, on)
			}
			for i, ok := range on {
				if !ok {
					t.Fatalf("decision %d must dispatch: the running set fits the budget (all %v)", i, on)
				}
			}
		})
	}
}

// ---- (4) dedup ----

// THE RUNNING SET IS A REFCOUNT (dedup-exact): a block two in-flight requests share is
// counted ONCE, and a head whose chain is already running adds NOTHING — so a gate that
// priced heads without the union would hold turns this one dispatches.
func TestExecutionBudgetDedupAcrossInFlightMembers(t *testing.T) {
	const (
		// 2000 GPU blocks: the dispatch budget is 1900, above the fixture's 1300-block
		// union but BELOW |run| + one more 900-block chain (2200) — so the union, and not
		// the sum, is what decides the last turn.
		capacity = 2000
		tier     = 3000
	)
	cfg := execBudgetConfig(capacity, tier)
	shared := keys(500, seedFor("shared"))
	aChain := append(append([]Hash(nil), shared...), keys(400, seedFor("a-only"))...)
	bChain := append(append([]Hash(nil), shared...), keys(400, seedFor("b-only"))...)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		if i := fx.seatAndDispatch(t, "b", 900, bChain); i < 0 {
			t.Fatal("B's turn must run (nothing is in flight)")
		}
		engine := fx.s.bindings["b"]
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		// A's chain shares 500 of its 900 blocks with B's. A is a NEWCOMER, so the index
		// has nothing to dedup yet and the gate prices its whole 900 — 900 + 900 = 1800,
		// inside the 1900 budget.
		if i := fx.seatAndDispatch(t, "a", 900, aChain); i < 0 {
			t.Fatalf("A's turn must fit: 900 running + its own 900 <= %d", execBudgetOf(cfg.GroupBudgetFraction, capacity))
		}
		// ... and the RUNNING SET now counts the 500 shared blocks ONCE: 900 + 900 = 1800
		// in flight, 1300 in blocks.
		if got := fx.runningSetSize(engine); got != 1300 {
			t.Fatalf("the running set must count the %d shared blocks ONCE: %d blocks, wanted 500 + 400 + 400 = 1300", len(shared), got)
		}
		// A's next turn over the SAME chain adds nothing to the union: 1300 + 0 <= 1900,
		// so it runs. (Sum-based pricing would need 1300 + 900 = 2200 and hold it — this
		// is the dedup deciding a dispatch.)
		if i := fx.seatAndDispatch(t, "a", 900, aChain); i < 0 {
			t.Fatalf("a turn whose whole chain is already running adds 0 blocks and must dispatch: %d running <= %d",
				fx.runningSetSize(engine), execBudgetOf(cfg.GroupBudgetFraction, capacity))
		}
		if got := fx.runningSetSize(engine); got != 1300 {
			t.Fatalf("a shared refcount must not grow the union: %d blocks, wanted 1300", got)
		}
		// B settles: A's two in-flight turns still hold all 900 of its blocks
		fx.settleAt("b", 900)
		if got := fx.runningSetSize(engine); got != 900 {
			t.Fatalf("A's two turns hold the same 900 blocks: %d, wanted 900", got)
		}
	})
}

// ---- (5) no starvation ----

// FIFO AMONG THE DISPATCH-ELIGIBLE: a big turn that cannot fit does not block the engine (a
// smaller turn behind it runs), and when it fits again it is the OLDEST eligible, so no
// later arrival can overtake it. Starvation-freedom without a second queue.
//
// The two small members are seated and idle BEFORE the big turn arrives — which is the
// realistic shape (a returning member whose prefill already completed) and also avoids a
// second effect this test does not claim: the derived cold-entry valve admits one
// outstanding prefill per engine, so a HELD turn legitimately occupies that slot while it
// waits.
func TestExecutionBudgetBigTurnIsNotStarved(t *testing.T) {
	const (
		capacity = 1000 // dispatch budget 950
		tier     = 3000
	)
	cfg := execBudgetConfig(capacity, tier)
	forNs(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
		// 900 blocks running (run is the first member, so the placement tie puts it on the
		// first engine), then every other engine is closed so the queue is ONE engine's
		fx.seatAndDispatch(t, "run", 900, keys(900, seedFor("run")))
		engine := fx.s.bindings["run"]
		fx.closeOtherEngines(t, seatBudgetOf(cfg, capacity, tier))
		if got := fx.runningSetSize(engine); got != 900 {
			t.Fatalf("the fixture needs 900 blocks running, got %d", got)
		}
		// two small members: seated, prefilled (the cold-entry valve released) and idle again
		for _, id := range []string{"small", "late"} {
			if i := fx.seatAndDispatch(t, id, 40, keys(40, seedFor(id))); i < 0 {
				t.Fatalf("%s must be seated and dispatched: 900 + 40 = 940 fits", id)
			}
			fx.settleAt(id, 40)
		}
		if got := fx.runningSetSize(engine); got != 900 {
			t.Fatalf("the small members settled: 900 blocks must still be running, got %d", got)
		}
		// big (900) does not fit (900 + 900 > 950); small's second turn (40) does (940)
		i := fx.pick(head("big", 30*time.Second, bytesFor(900)), head("small", 5*time.Second, bytesFor(40)))
		if i != 1 {
			t.Fatalf("the small turn behind a held big one must dispatch (index 1), got %d", i)
		}
		if fx.s.bindings["big"] != engine || fx.s.bindings["small"] != engine || fx.s.bindings["late"] != engine {
			t.Fatalf("every turn must be seated on %s while the gate decides: %v", engine, fx.s.bindings)
		}
		fx.s.DispatchedOn("small", "r-small", "", keys(40, seedFor("small")), bytesFor(40), fx.clock)
		if got := fx.runningSetSize(engine); got != 940 {
			t.Fatalf("the running set must be 900 + 40 = 940, got %d", got)
		}
		// the 900-block request settles: the big turn now fits (40 + 900 = 940) — and so
		// does the later 40-block arrival
		fx.settleAt("run", 900)
		if i := fx.pick(head("big", 30*time.Second, bytesFor(900)), head("late", time.Second, bytesFor(40))); i != 0 {
			t.Fatalf("once the big turn fits it is the oldest eligible head and must go first, got index %d", i)
		}
	})
}

// ---- (6) golden neutrality ----

// THE DEFAULT CONFIG'S TRACE IS UNCHANGED. The default selector is "none", nothing is
// maintained, and the f0p golden replay is reproduced byte for byte.
func TestExecutionBudgetDefaultIsGoldenNeutral(t *testing.T) {
	mod, err := buildModules(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if got := mod.gate.Name(); got != "none" {
		t.Fatalf("the default dispatch selector must stay %q, got %q", "none", got)
	}
	if mod.gateOn {
		t.Fatal("the default must leave the gate OFF, so nothing is maintained and no comparator arm pays for it")
	}
	if got := mod.gate.BudgetBlocks(40183); got != 0 {
		t.Fatalf("the noop gate has no budget to enforce, got %d", got)
	}
	want, err := os.ReadFile(filepath.Join("testdata", "golden-f0p-bugfixes.txt"))
	if err != nil {
		t.Fatalf("f0p golden missing: %v", err)
	}
	if got := runGoldenScenario(t); got != string(want) {
		t.Fatal("the DEFAULT config's trace moved: the execution budget must be invisible unless an arm asks for it")
	}
}

// AND ENABLING IT WHERE THE RUNNING SET ALWAYS FITS MOVES NOTHING IN THE WHOLE REPLAY —
// 2,600 s, byte for byte, with the module ON. "No effect when it fits" at scale: not one
// pick, seat or exit differs.
func TestExecutionBudgetOnChangesNothingWhenItNeverBinds(t *testing.T) {
	const huge = 200000 // blocks: the dispatch budget is 190,000 and the replay never nears it
	off := runGoldenAt(t, defaultConfig(), huge)
	on := runGoldenAt(t, execConfig(defaultConfig()), huge)
	if off == on {
		return
	}
	offL, onL := strings.Split(off, "\n"), strings.Split(on, "\n")
	for i := 0; i < len(offL) && i < len(onL); i++ {
		if offL[i] != onL[i] {
			t.Fatalf("the execution budget changed decision %d on a deployment where the running set always fits:\n  off: %s\n  on:  %s", i+1, offL[i], onL[i])
		}
	}
	t.Fatalf("the two traces differ in length (%d vs %d lines)", len(offL), len(onL))
}

// ---- (7) the accounting ----

// THE RUNNING SET IS EXACT under every path that changes it: dispatch, settle, a second
// concurrent turn, an unobserved termination (the reap), a rebind to the observed engine and
// an exit — checked against the DERIVED union of the scheduler's own in-flight records.
func TestExecutionBudgetRunningSetAccounting(t *testing.T) {
	cfg := execConfig(defaultConfig())
	cfg.InflightReapSeconds = 60 // the reap path needs a bound the test can cross
	cfg.TTLSeconds = 600         // ... and the member must survive the advance that reaps it
	forNs(t, cfg, 20000, func(t *testing.T, fx *engineSet) {
		totalRunning := func() int {
			n := 0
			for _, id := range fx.ids {
				n += fx.runningSetSize(id)
			}
			return n
		}
		fx.seatAndDispatch(t, "a", 300, keys(300, seedFor("a")))
		fx.checkRunning(t, "after a seat+dispatch")
		if got := totalRunning(); got != 300 {
			t.Fatalf("one dispatched turn is 300 running blocks, got %d", got)
		}
		fx.seatAndDispatch(t, "b", 200, keys(200, seedFor("b")))
		fx.checkRunning(t, "after a second member")
		// a second concurrent turn for A (a sub-agent thread): both count
		fx.s.DispatchedOn("a", "r-a2", "", keys(120, seedFor("a2")), bytesFor(120), fx.clock)
		fx.checkRunning(t, "after a second in-flight turn")
		if got := totalRunning(); got != 620 {
			t.Fatalf("A's two turns (300+120) plus B's (200) is the running set, got %d", got)
		}
		fx.settleAt("a", 300) // settles "r-a": its 300 blocks leave, A's 120 stay
		fx.checkRunning(t, "after a settle")
		if got := totalRunning(); got != 320 {
			t.Fatalf("A's remaining turn (120) plus B's (200) is the running set, got %d", got)
		}
		// an unobserved termination: past InflightReapSeconds the id is reaped
		fx.clock = fx.clock.Add(2 * time.Minute)
		fx.s.Tick(fx.clock)
		fx.checkRunning(t, "after the reap")
		if got := totalRunning(); got != 0 {
			t.Fatalf("every confirmed id is past the reap bound: the running set must be empty, got %d", got)
		}
		// a rebind: the observed endpoint wins and the running blocks move WITH the member
		fx.seatAndDispatch(t, "c", 400, keys(400, seedFor("c")))
		seat := fx.s.bindings["c"]
		if len(fx.ids) > 1 {
			other := fx.ids[0]
			if seat == other {
				other = fx.ids[1]
			}
			if got := fx.runningSetSize(seat); got != 400 {
				t.Fatalf("C's turn is running on its seat: %d, wanted 400", got)
			}
			fx.s.DispatchedOn("c", "r-c", other, keys(400, seedFor("c")), bytesFor(400), fx.clock)
			fx.checkRunning(t, "after a rebind")
			if got := fx.runningSetSize(other); got != 400 {
				t.Fatalf("the rebound member's blocks must be running on the OBSERVED engine: %d, wanted 400", got)
			}
			if got := fx.runningSetSize(seat); got != 0 {
				t.Fatalf("... and NOT on the seat it left: %d, wanted 0", got)
			}
			seat = other
		}
		// an exit takes the member's in-flight ids out of the set
		fx.s.mu.Lock()
		for _, g := range fx.allGroups() {
			if m, ok := g.members["c"]; ok {
				fx.s.exitLocked(g, m, "test", fx.clock)
			}
		}
		fx.s.mu.Unlock()
		fx.checkRunning(t, "after an exit")
		if got := fx.runningSetSize(seat); got != 0 {
			t.Fatalf("an exited member's in-flight blocks leave the running set, got %d", got)
		}
	})
}

// ---- (8) the contract ----

// THE SELECTOR IS CONFORMANT: buildModules accepts both values, fails CLOSED on anything
// else, and the module's arithmetic is what the deployment reads. (The type-checked half —
// the emitter's key and both values through the loader's strict decode — is in
// config_conformance_test.go, and the loader path itself in cmd/epp/runner.)
func TestExecutionBudgetSelectorIsConformant(t *testing.T) {
	for _, v := range []string{"none", "execution-budget"} {
		cfg := defaultConfig()
		cfg.ExecutionBudget = v
		mod, err := buildModules(cfg)
		if err != nil {
			t.Fatalf("buildModules rejects executionBudget=%q: %v", v, err)
		}
		if mod.gate.Name() != v {
			t.Fatalf("executionBudget=%q wired the %q gate", v, mod.gate.Name())
		}
	}
	cfg := defaultConfig()
	cfg.ExecutionBudget = "exec-budget" // the plausible typo
	if _, err := buildModules(cfg); err == nil {
		t.Fatal("an unknown dispatch value must fail CLOSED (a typo would otherwise measure the arm without the budget)")
	}
	g := ExecutionBudget{Fraction: 0.95}
	if got := g.BudgetBlocks(1000); got != 950 {
		t.Fatalf("the dispatch budget is fraction x num_gpu_blocks: got %d, wanted 950", got)
	}
	if got := g.BudgetBlocks(0); got != 0 {
		t.Fatalf("an unmeasured engine has no dispatch budget, got %d", got)
	}
	if !g.Allows(DispatchCtx{RunningBlocks: 900, ChainBlocks: 50, BudgetBlocks: 950}) {
		t.Fatal("exactly filling the budget must be allowed (the bound is inclusive)")
	}
	if g.Allows(DispatchCtx{RunningBlocks: 900, ChainBlocks: 51, BudgetBlocks: 950}) {
		t.Fatal("one block over the budget must be held")
	}
	if !g.Allows(DispatchCtx{RunningBlocks: 10, ChainBlocks: 5000, BudgetBlocks: 950, Idle: true}) {
		t.Fatal("liveness: the first turn on an idle engine is always allowed")
	}
	if !g.Allows(DispatchCtx{RunningBlocks: 10, ChainBlocks: 5000, BudgetBlocks: 0}) {
		t.Fatal("defensive: a gate with no measured budget must be OPEN, never closed")
	}
}

// ---- (8b) the interface contract the gate shares with the seat path ----

// Cost IS TierCost's cold term, and total IS cold + resident: the dispatch gate takes both
// halves from ONE sizing call, so this contract is what keeps the execution budget and the
// seat budget from pricing the same head two different ways. Pinned for every wired sizer
// (a future implementation that breaks it fails here, not in a run).
func TestExecutionBudgetSizerContractHolds(t *testing.T) {
	for _, sizing := range []string{"index", "gpu-plus-tier"} {
		sizing := sizing
		t.Run(sizing, func(t *testing.T) {
			cfg := execBudgetConfig(1000, 2000)
			cfg.Sizing = sizing
			forEnginesCap(t, cfg, 1000, func(t *testing.T, fx *engineSet) {
				fx.seatAndDispatch(t, "m", 300, keys(300, seedFor("m")))
				fx.s.mu.Lock()
				g, mod := fx.g(), fx.s.mod.sizer
				fx.s.mu.Unlock()
				for _, size := range []int64{bytesFor(40), bytesFor(100), bytesFor(300) * 3} {
					cold, resident, total := mod.TierCost(g.index, nil, "next", size)
					if got := mod.Cost(g.index, "next", size); got != cold {
						t.Fatalf("%s: Cost(%d) = %d but TierCost's cold term is %d — Cost MUST be TierCost with no tier", sizing, size, got, cold)
					}
					if total != cold+resident {
						t.Fatalf("%s: total %d != cold %d + resident %d", sizing, total, cold, resident)
					}
					if cold <= 0 || total < cold {
						t.Fatalf("%s: a new member's head must cost something: cold=%d total=%d", sizing, cold, total)
					}
				}
			})
		})
	}
}

// ---- (9) the hot path ----

// THE GATE'S QUESTION IS ASKED PER CANDIDATE HEAD ON EVERY DECISION PASS, so it must
// allocate nothing: the numbers come from the caches, the shared-block count is computed in
// place, and the decision travels by value.
func BenchmarkExecutionBudgetDispatchCheck(b *testing.B) {
	const (
		capacity = 40183 // the campaign engine
		chainLen = 800   // a ~51K-token prefix chain
	)
	s, err := NewScheduler(execConfig(defaultConfig()))
	if err != nil {
		b.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, capacity); err != nil {
		b.Fatal(err)
	}
	now := time.Unix(1793000000, 0).UTC()
	// seat + dispatch one resident so the running set is non-empty (the interesting case)
	s.PickFromHeads([]Head{{Member: "m", ReqID: "r-m", Arrival: now.Add(-time.Minute), SizeBytes: bytesFor(chainLen)}}, now)
	s.DispatchedOn("m", "r-m", "", keys(chainLen, seedFor("m")), bytesFor(chainLen), now)

	h := &Head{Member: "m", ReqID: "r-m2", Arrival: now, SizeBytes: bytesFor(chainLen)}
	budget := s.mod.gate.BudgetBlocks(capacity)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.mu.Lock()
		ok := s.dispatchAllowedLocked(s.groups["engine-a"], h, budget)
		s.mu.Unlock()
		if !ok {
			b.Fatal("the fixture's head must fit (its chain IS the running set)")
		}
	}
}

// BenchmarkExecutionBudgetDispatchCheckColdScan is the worst case of the same check: the
// per-head cache is cold (the member's chain and the running set both moved since the last
// pass), so the exact union is recomputed — one pass over the member's own block set, in
// place, still without allocating. It runs once per head per state change, not per pass.
func BenchmarkExecutionBudgetDispatchCheckColdScan(b *testing.B) {
	const (
		capacity = 40183
		chainLen = 800
	)
	s, err := NewScheduler(execConfig(defaultConfig()))
	if err != nil {
		b.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, capacity); err != nil {
		b.Fatal(err)
	}
	now := time.Unix(1793000000, 0).UTC()
	s.PickFromHeads([]Head{{Member: "m", ReqID: "r-m", Arrival: now.Add(-time.Minute), SizeBytes: bytesFor(chainLen)}}, now)
	s.DispatchedOn("m", "r-m", "", keys(chainLen, seedFor("m")), bytesFor(chainLen), now)

	h := &Head{Member: "m", ReqID: "r-m2", Arrival: now, SizeBytes: bytesFor(chainLen)}
	budget := s.mod.gate.BudgetBlocks(capacity)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.mu.Lock()
		s.groups["engine-a"].gateCache = nil // force the scan
		ok := s.dispatchAllowedLocked(s.groups["engine-a"], h, budget)
		s.mu.Unlock()
		if !ok {
			b.Fatal("the fixture's head must fit")
		}
	}
}

// ---- helpers ----

// runGoldenAt replays the golden scenario on ONE engine of the given capacity.
func runGoldenAt(t *testing.T, cfg Config, capacity int) string {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, capacity); err != nil {
		t.Fatal(err)
	}
	return runGoldenScenarioOnSched(t, 2600, s)
}
