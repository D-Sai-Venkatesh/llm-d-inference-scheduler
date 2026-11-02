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

// THE RELEASED WINDOW — every way it closes, and the one eviction it may justify.
//
// A request that a Pick has released is neither queued nor in flight until PreRequest
// records it. The branch carries `released` for that window, because a cold charge nobody
// can explain closes the engine's valve. The window is closed by EVENTS (PreRequest, the
// terminal callback, the director's abandon hook, an engine teardown), and for the paths
// that reach none of them it is freed ON DEMAND by the one eviction I2 allows a valve-held
// head to take: the branch that HOLDS the charge, once the tenure rule already releases it.

import (
	"context"
	"testing"
	"time"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

// the two release-path extension points this plugin takes part in.
var (
	_ fwkrc.DataProducer     = (*GroupScheduler)(nil)
	_ fwkrc.RequestAbandoned = (*GroupScheduler)(nil)
)

// closeOtherEngines fills every engine but the first, so "the engine cannot take it" is
// true at N=1 and N=2 alike (the 10,000-block capacity fixture needs its own filler size).
func (fx *engineSet) closeEnginesExcept(t *testing.T, keep string, blocks int) {
	t.Helper()
	for _, e := range fx.ids {
		if e == keep {
			continue
		}
		id := "closed-" + e
		fx.seatOne(t, id, blocks, 3*time.Second)
		if got := fx.s.bindings[id]; got != e {
			t.Fatalf("%s must hold %s, it holds %q", id, e, got)
		}
	}
}

// engineOf is where a branch is seated.
func (fx *engineSet) engineOf(t *testing.T, id string) string {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.bindings[id]
}

// openWindow seats branch "m" and then releases a SECOND turn that never reaches a pod:
// one request in flight, one window open.
func openWindow(t *testing.T, fx *engineSet, reqID string) {
	t.Helper()
	fx.seatOne(t, "m", 100, time.Second)
	h := head("m", time.Second, 100)
	h.ReqID = reqID
	if i := fx.pick(h); i != 0 {
		t.Fatalf("the seated branch's head must be picked, got %d", i)
	}
	if got := windowOf(t, fx, "m"); got != reqID {
		t.Fatalf("the window must name %q, it names %q", reqID, got)
	}
}

// windowOf is the request the branch has released and not yet brought to a pod ("" when
// none): the record's own view, never a copy of it.
func windowOf(t *testing.T, fx *engineSet, id string) string {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.oldestPendingLocked(id)
}

// Every EVENT that proves a request is no longer only "on its way" closes the window.
func TestReleasedWindowClosesOnItsEvents(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("PreRequest", func(t *testing.T) {
			openWindow(t, fx, "r-a")
			fx.s.Dispatched("m", "r-a", fx.clock)
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("PreRequest must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("terminal-callback", func(t *testing.T) {
			openWindow(t, fx, "r-b")
			// the handler forces this call for every request it scheduled a pod for
			fx.s.Settle("m", "r-b", struct{ Prompt, Cached int }{100, 0}, false, fx.clock)
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("the terminal callback must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("abandon-hook", func(t *testing.T) {
			openWindow(t, fx, "r-c")
			// the director's hook for a request that left the path before PreRequest
			fx.s.Abandon("m", "r-c")
			if got := windowOf(t, fx, "m"); got != "" {
				t.Fatalf("RequestAbandoned must close the window, it is %q", got)
			}
		})
	})
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		t.Run("engine-gone", func(t *testing.T) {
			openWindow(t, fx, "r-d")
			fx.s.RemoveEngine(fx.ids[0], fx.clock)
			if fx.s.Seated("m") {
				t.Fatal("the teardown must unseat the branch")
			}
		})
	})
}

// A window that no event ever closes (the paths that reach no hook: no endpoint candidates
// at the director's entry, an item finalized while it was being dispatched) must not be
// able to freeze anything: it never bars the tenure rule, the engine-idle rule or the
// in-flight cap.
func TestStaleWindowNeverBarsLiveness(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-stale")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		noInFlightLocked(fx.s, m) // its other request is over
		// the release is an HOUR old: past S, so the window is presumed dead (a live one
		// counts as in flight and is protected — see branch.busy)
		pendingLocked(fx.s, m, fx.clock.Add(-time.Hour), "r-stale")
		m.lastActivity = fx.clock.Add(-time.Hour) // silent
		fx.s.mu.Unlock()
		// one pass with no heads: the released turn is no longer queued (the next Pick in
		// production clears it the same way)
		fx.pick()
		reason, victim := fx.s.rotationVictimLocked(fx.g(), fx.clock)
		if victim == nil {
			t.Fatal("a dead window must not stop the tenure rule: a silent branch is releasable")
		}
		if reason != "idle" {
			t.Fatalf("the dead window's branch is releasable as %q, want idle", reason)
		}
		if victim.id != "m" {
			t.Fatalf("the victim is %q, want m", victim.id)
		}
	})
}

// A LIVE window (a release a few milliseconds old) is IN FLIGHT: a valve-held head waits for
// it (S has not elapsed), and no rule may evict the branch out from under its request.
func TestLiveWindowIsNotEvicted(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // its cold charge fills the engine's cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take the head
		fx.s.mu.Lock()
		m := fx.g().branches["holder"]
		noInFlightLocked(fx.s, m)
		pendingLocked(fx.s, m, fx.clock, "r-live") // a request on its way, released just now
		m.lastActivity = fx.clock
		fx.s.mu.Unlock()
		fx.pick() // the released turn is no longer queued
		before := fx.s.Snapshot()
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i >= 0 {
			t.Fatalf("a valve-held head must not be served over a live window, got %d", i)
		}
		after := fx.s.Snapshot()
		if after.Seats != before.Seats || len(after.Exits) != len(before.Exits) {
			t.Fatalf("nothing may be evicted for a live window: seats %d->%d exits %v->%v",
				before.Seats, after.Seats, before.Exits, after.Exits)
		}
		if got := windowOf(t, fx, "holder"); got != "r-live" {
			t.Fatalf("the live window must still be open, it is %q", got)
		}
	})
}

// A DEAD window (> S) frees its valve BY ITSELF: the derivation releases a charge nothing
// explains any more, so the head that the valve was holding is served — and NOTHING is
// evicted for it.
//
// This replaces the I2 holder correction. That rule let a valve-held head evict the branch
// holding the charge; with the window counted as in flight and bounded by S, the charge is
// released at the pass the window is presumed dead, exactly when an eviction would have been
// taken — so the rule was redundant, and the code that could evict a resident for a head
// that fits its budget is gone.
func TestDeadWindowFreesTheValveWithoutEviction(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // charges min(cost, cap) = the whole cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take it instead
		fx.s.mu.Lock()
		m := fx.g().branches["holder"]
		noInFlightLocked(fx.s, m)
		pendingLocked(fx.s, m, fx.clock.Add(-21*time.Second), "r-dead") // dead: 21s > S
		m.lastActivity = fx.clock.Add(-21 * time.Second)
		fx.s.mu.Unlock()
		fx.pick() // the released turn is no longer queued; the charge has nothing to explain it
		if got := fx.g().coldInFlight; got != 0 {
			t.Fatalf("a dead window's charge must be released by the derivation, coldInFlight=%d", got)
		}
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i < 0 {
			t.Fatal("the head the valve was holding must be served once the dead window's charge is released")
		}
		if !fx.s.Seated("holder") {
			t.Fatal("nothing may be evicted for a head that fits the budget: the holder keeps its seat")
		}
		if exits := fx.s.Snapshot().Exits; len(exits) != 0 {
			t.Fatalf("a dead window needs no eviction at all, exits=%v", exits)
		}
	})
}

// THE BUG1 REGRESSION: a valve-held head never evicts a branch that is NOT the holder —
// not even a releasable one. (BUG1 evicted every eligible member and left the seats
// cold-empty; that is what I2 exists to forbid.)
func TestValveHeldHeadNeverEvictsANonHolder(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		// an innocent, releasable branch whose charge is already gone (seated FIRST: the
		// holder's charge will fill the valve)
		fx.seatOne(t, "innocent", 100, 2*time.Second)
		fx.s.Chunk("innocent") // its prefill landed: no cold charge left to explain
		// the holder keeps its charge (in flight, so nothing can release it)
		fx.seatOne(t, "holder", 600, time.Second)
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000)
		fx.s.mu.Lock()
		m := fx.g().branches["innocent"]
		noInFlightLocked(fx.s, m)
		m.lastActivity = fx.clock.Add(-time.Hour) // it looks idle — but it holds no charge
		fx.s.mu.Unlock()
		before := fx.s.Snapshot()
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i >= 0 {
			t.Fatalf("a valve-held head must not be served by evicting a NON-holder, got %d", i)
		}
		after := fx.s.Snapshot()
		if after.Seats != before.Seats {
			t.Fatalf("BUG1: seats moved %d -> %d for a valve-held head", before.Seats, after.Seats)
		}
		if !fx.s.Seated("innocent") || !fx.s.Seated("holder") {
			t.Fatal("BUG1: a branch was evicted for a valve-held head")
		}
	})
}

// Two releases of the same branch before either reaches a pod (a retry merged into one
// queue) are TWO records — the branch is held busy by either of them — and a callback ends
// only the record it NAMES, in either order, leaving nothing open at the end.
func TestTwoReleasesClearOnlyTheirOwnWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		// the retry released a second request before the first reached a pod
		fx.s.reqs["r-second"] = &reqRec{branch: "m", released: fx.clock}
		m.pending++
		fx.s.mu.Unlock()
		// a callback for the FIRST request ends the first record only
		fx.s.Abandon("m", "r-first")
		if got := windowOf(t, fx, "m"); got != "r-second" {
			t.Fatalf("a callback for the first request closed the second's window: it names %q", got)
		}
		// and the matching one ends the second
		fx.s.Abandon("m", "r-second")
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("the matching callback must close the window, it is %q", got)
		}
		// a late callback for an ended request invents nothing
		fx.s.Abandon("m", "r-first")
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("a late callback reopened the window (%q)", got)
		}
	})
	// the same, in the other order
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		fx.s.reqs["r-second"] = &reqRec{branch: "m", released: fx.clock}
		m.pending++
		fx.s.mu.Unlock()
		fx.s.Abandon("m", "r-second")
		if got := windowOf(t, fx, "m"); got != "r-first" {
			t.Fatalf("the older window must still be open, it names %q", got)
		}
		fx.s.Abandon("m", "r-first")
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("the matching callback must close the window, it is %q", got)
		}
	})
	_ = context.Background()
}

// A branch evicted while its ONLY claim on the engine was a released window is seated by
// the OBSERVATION when its request reaches PreRequest, and drains when the request ends.
//
// Enforcement marks such a branch exiting (it is prefilling — a turn is on its way), and the
// drain reads in-flight only, so the branch is unseated BEFORE its request arrives. Without
// the observed seat, that request would run untracked: its blocks absent from the index and
// its cold charge gone, and the engine's next admission would hand its blocks away twice.
func TestReleasedWindowVictimIsAccountedAtPreRequest(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		engine := fx.ids[0]
		// a keeps a request in flight for the whole test, so no idle/done victim class exists
		fx.seatOne(t, "a", 300, 5*time.Second)
		fx.s.Chunk("a") // its prefill landed: the cold-entry valve is a separate budget
		// nothing may escape to a sibling engine (the N=1 shape must hold at N=2)
		fx.closeEnginesExcept(t, engine, 950)
		// b joins a on the same engine and completes one turn
		fx.seatOne(t, "b", 300, 4*time.Second)
		if got := fx.engineOf(t, "b"); got != engine {
			t.Fatalf("b must share a's engine for this shape, it is on %q", got)
		}
		settle(fx.s, "b", 300*64, 300*64)

		// b's next turn is released with a prompt that grew past the engine's headroom:
		// charged at the dispatch, so 300 (a) + 900 (b) > the 950-block budget, and b has
		// nothing in flight — its only claim on the seat is the window.
		h := headWithChain("b", time.Second, chainOf("b", 900))
		h.ReqID = "r-b2"
		if i := fx.pick(h); i != 0 {
			t.Fatalf("b's grown turn must dispatch on its seat, got %d", i)
		}
		if got := windowOf(t, fx, "b"); got != "r-b2" {
			t.Fatalf("the released window must name b's request, it names %q", got)
		}

		// THE WINDOW IS IN FLIGHT WHILE IT IS LIVE, so the branch is not draggable out from
		// under it: for S the pass leaves it alone even though it is over budget.
		fx.pick()
		if !fx.s.Seated("b") {
			t.Fatal("a LIVE window is a turn on its way: the branch must not be drained before it runs")
		}
		// past S the window is presumed dead (its request never arrived): it stops counting
		// as in flight, so the pressure pass takes the branch like any other headless
		// resident — a quarter of the engine's budget, nothing running.
		fx.clock = fx.clock.Add(time.Duration(fx.s.cfg.IdleSeconds+1) * time.Second)
		fx.pick()
		if fx.s.Seated("b") {
			t.Fatal("the dead window must no longer bar the pressure pass: the branch is evictable on demand")
		}
		// the mark was taken while the window was still live (the branch was over budget and
		// in flight — a state, not an exit); the departure is booked once, here
		if exits := fx.s.Snapshot().Exits; exits["exiting-drain"] == 0 {
			t.Fatalf("the dead-window branch must leave at the drain, exits=%v", exits)
		}
		before := fx.s.Snapshot()

		// PreRequest: the request IS on the engine, and the observation wins
		fx.s.DispatchedOn("b", "r-b2", engine, chainOf("b", 900), fx.clock)
		if !fx.s.Seated("b") {
			t.Fatal("the observation must seat the branch its request landed on: otherwise it runs untracked")
		}
		after := fx.s.Snapshot()
		if got, want := after.NEBlocks, before.NEBlocks+900; got != want {
			t.Fatalf("the observed seat must account the request's chain: used %d, want %d", got, want)
		}
		if got := after.Engines[engineLabel(engine)].ForcedSeats; got != 1 {
			t.Fatalf("the observed seat must be counted, forcedSeats=%d", got)
		}
		// the observed seat is not an admission: it never dispatches a turn
		if i := fx.pick(h); i >= 0 {
			t.Fatalf("an exiting branch must not dispatch, got %d", i)
		}

		// the request drains: the seat goes with it, and the blocks leave the budget
		fx.s.Settle("b", "r-b2", struct{ Prompt, Cached int }{900 * 64, 900 * 64}, true, fx.clock)
		if fx.s.Seated("b") {
			t.Fatal("the observed seat must leave when its request drains")
		}
		if got := fx.s.Snapshot().NEBlocks; got != before.NEBlocks {
			t.Fatalf("the drained seat must release its blocks: used %d, want %d", got, before.NEBlocks)
		}
	})
}

// ---- THE THREE PROBES, as regressions ----
//
// The window is the COMMON case, not a corner: every dispatch passes through it and Pick
// runs at ~1 kHz, so a decision almost always lands inside one. Each probe below pins one
// rule that used to read a branch on its way to the pod as IDLE, and the damage that
// followed (the numbers are the measured ones: two engines, in-flight cap 4).

// usedWithinBudget asserts the engine's used side never exceeds its budget plus what is
// already on its way out: the invariant every probe broke (used 1200 against 950; 1500
// against 1100).
func usedWithinBudget(t *testing.T, g *engineGroup) {
	t.Helper()
	used, budget, exiting := g.index.UniqueBlocksLocked(), g.budgetBlocks, int(g.index.ExitingBlocksLocked())
	if used > budget+exiting {
		t.Fatalf("the engine is over its budget with nothing on the way out to excuse it: used %d, budget %d, exiting %d", used, budget, exiting)
	}
}

// (i) THE ENGINE-IDLE RULE never evicts a branch whose request is on its way. Evicting it
// left the engine 250 blocks over budget when the observed seat put the branch straight back.
func TestEngineIdleRuleNeverEvictsALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		engine := fx.ids[0]
		fx.seatOne(t, "m", 900, 2*time.Second)
		fx.s.Chunk("m")
		fx.closeEnginesExcept(t, engine, 950) // no sibling may take the new head instead
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		noInFlightLocked(fx.s, m) // nothing at a pod: the window is m's ONLY claim
		pendingLocked(fx.s, m, fx.clock, "r-live")
		fx.s.mu.Unlock()
		before := fx.s.Snapshot().Seats
		// a head that does not fit the budget: the old rule read "nothing in flight, nothing
		// dispatchable" and evicted m for it
		if _, i := fx.pickEngine(head("cold", time.Second, 500)); i >= 0 {
			t.Fatalf("a head that does not fit must wait, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("the engine-idle rule evicted a branch whose request is on its way")
		}
		if after := fx.s.Snapshot().Seats; after != before {
			t.Fatalf("no seat may move for a live window: %d -> %d", before, after)
		}
		if exits := fx.s.Snapshot().Exits; len(exits) != 0 {
			t.Fatalf("nothing may leave for a live window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
	})
}

// (ii) A MARKED branch is not DRAINED inside a live window: the drain used to release it
// before its request ran, and the observed seat then put 1500 blocks back against 1100.
func TestMarkedBranchIsNotDrainedInsideALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		fx.s.Chunk("m")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		noInFlightLocked(fx.s, m)
		pendingLocked(fx.s, m, fx.clock, "r-live") // a turn on its way
		fx.g().index.MarkExitingLocked("m")        // the pressure pass already marked it
		fx.s.mu.Unlock()
		fx.pick() // the pass runs the drain
		if !fx.s.Seated("m") {
			t.Fatal("a marked branch must not be drained while its window is live: its request is on its way")
		}
		if exits := fx.s.Snapshot().Exits; exits["exiting-drain"] != 0 {
			t.Fatalf("the drain released a branch mid-window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
		// past S the window is presumed dead and the drain lets it go — exactly once
		fx.tick(time.Duration(fx.s.cfg.IdleSeconds+1) * time.Second)
		if fx.s.Seated("m") {
			t.Fatal("a dead window must not hold a marked branch's seat")
		}
		if exits := fx.s.Snapshot().Exits; exits["exiting-drain"] != 1 {
			t.Fatalf("the departure must be booked exactly once, exits=%v", exits)
		}
	})
}

// (iii) NO ROTATION EXIT inside a live window, even for a branch the tenure rule would
// release on its own (Q complete).
func TestNoRotationExitInsideALiveWindow(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 1000, func(t *testing.T, fx *engineSet) {
		engine := fx.ids[0]
		fx.seatOne(t, "m", 800, 2*time.Second)
		fx.s.Chunk("m")
		fx.closeEnginesExcept(t, engine, 950)
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		m.busyPeriods = fx.s.cfg.quantum() // the rotation quantum is complete
		noInFlightLocked(fx.s, m)
		pendingLocked(fx.s, m, fx.clock, "r-live") // and a turn is on its way
		m.lastActivity = fx.clock.Add(-time.Hour)
		fx.s.mu.Unlock()
		fx.s.mu.Lock()
		reason, victim := fx.s.rotationVictimLocked(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if victim != nil {
			t.Fatalf("the tenure rule released %q (%s) inside a live window", victim.id, reason)
		}
		// and a blocked head does not get the seat by eviction either
		if _, i := fx.pickEngine(head("cold", time.Second, 500)); i >= 0 {
			t.Fatalf("a head that does not fit must wait, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("a quantum-complete branch with a live window must keep its seat until the window ends")
		}
		if exits := fx.s.Snapshot().Exits; len(exits) != 0 {
			t.Fatalf("no rotation exit may be taken inside a live window, exits=%v", exits)
		}
		usedWithinBudget(t, fx.g())
	})
}
