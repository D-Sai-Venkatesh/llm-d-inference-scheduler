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

func windowOf(t *testing.T, fx *engineSet, id string) string {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	if m, ok := fx.g().branches[id]; ok {
		return m.released
	}
	return "-gone-"
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
		m.inflightIDs = map[string]inflightRec{} // its other request is over
		m.inflight = 0
		m.lastActivity = fx.clock.Add(-time.Hour) // silent: the window is dead
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

// A LIVE window (a release a few milliseconds old) is protected: a valve-held head may not
// evict it, because S has not elapsed.
func TestLiveWindowIsNotEvicted(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // its cold charge fills the engine's cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take the head
		fx.s.mu.Lock()
		m := fx.g().branches["holder"]
		m.inflightIDs = map[string]inflightRec{}
		m.inflight = 0
		m.released = "r-live" // a request on its way, released just now
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

// THE I2 HOLDER CORRECTION: a valve-held head MAY evict the branch holding the charge, once
// the tenure rule releases it — evicting the holder frees the valve, and that is the one
// case where I2's premise ("evicting frees budget, never the valve") is false.
func TestValveHeldHeadEvictsItsEligibleHolder(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "holder", 600, time.Second)                // charges min(cost, cap) = the whole cap
		fx.closeEnginesExcept(t, fx.engineOf(t, "holder"), 9000) // no sibling may take it instead
		fx.s.mu.Lock()
		m := fx.g().branches["holder"]
		m.inflightIDs = map[string]inflightRec{}
		m.inflight = 0
		m.released = "r-dead" // a window nothing will ever close
		m.lastActivity = fx.clock.Add(-21 * time.Second)
		fx.s.mu.Unlock()
		fx.pick() // the released turn is no longer queued
		if _, i := fx.pickEngine(head("cold", time.Second, 900)); i < 0 {
			t.Fatal("the valve-held head must be served once the holder is releasable: it frees the valve")
		}
		if fx.s.Seated("holder") {
			t.Fatal("the HOLDER must be the victim (its charge is what holds the head back)")
		}
		if fx.s.Snapshot().Exits["exiting-drain"]+fx.s.Snapshot().Exits["engine-idle"] == 0 &&
			fx.s.Snapshot().Exits["idle"] == 0 {
			t.Fatalf("the holder must leave through the tenure rule, exits=%v", fx.s.Snapshot().Exits)
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
		m.inflightIDs = map[string]inflightRec{}
		m.inflight = 0
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
// queue): a callback closes only the window it NAMES, in either order, and nothing is left
// open at the end.
func TestTwoReleasesClearOnlyTheirOwnWindow(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		m.released = "r-second" // the retry replaced the window
		// a callback for the FIRST request must not clear the second's window
		fx.s.closeReleasedLocked(m, "r-first")
		after := m.released
		fx.s.mu.Unlock()
		if after != "r-second" {
			t.Fatalf("a callback for the first request cleared the second's window (%q)", after)
		}
		// and the matching one does close it
		fx.s.Abandon("m", "r-second")
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("the matching callback must close the window, it is %q", got)
		}
	})
	// the same, in the other order
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		openWindow(t, fx, "r-first")
		fx.s.mu.Lock()
		m := fx.g().branches["m"]
		m.released = "r-second"
		fx.s.closeReleasedLocked(m, "r-second")
		after := m.released
		fx.s.mu.Unlock()
		if after != "" {
			t.Fatalf("the matching callback must close the window, it is %q", after)
		}
		fx.s.Abandon("m", "r-first") // a late callback for the older request: nothing to do
		if got := windowOf(t, fx, "m"); got != "" {
			t.Fatalf("a late callback reopened nothing, it must leave %q", got)
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

		// the pressure pass marks b EXITING, and the next pass's drain lets it go: nothing
		// was in flight to wait for
		fx.pick()
		snap := fx.s.Snapshot()
		if got := snap.Engines[engineLabel(engine)].ExitingMembers; got != 1 {
			t.Fatalf("the over-budget pass must mark the window-only branch exiting (exiting=%d, exits=%v)", got, snap.Exits)
		}
		fx.pick()
		if fx.s.Seated("b") {
			t.Fatal("the drain must have released the window-only branch before its request arrived")
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
