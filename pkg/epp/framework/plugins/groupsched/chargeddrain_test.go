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

// THE CHARGED DRAIN'S RELEASE INVARIANT, ONE TEST PER EXIT PATH.
//
// (iii) made an exiting member's blocks stay on the budget's used side (blocksNE)
// until it is unseated, because the engine pins them while the request drains. That is
// sound ONLY if every path that takes a member out ends at the ONE release point
// (Unseat/Forget). MarkExiting moves no counter any more, so a member that becomes
// exiting and never reaches Unseat/Forget keeps its blocks charged to the budget
// FOREVER — the engine then looks permanently full, TrySeat refuses every seat, and
// marking frees nothing, so that engine stops serving: the whole pool, silently.
// Before (iii) the same lifecycle hole was invisible, because the mark itself dropped
// the blocks out of NE "by definition".
//
// So the inventory has to be exhaustive. A member becomes EXITING at exactly one site
// (enforceBudgetLocked's mid-flight class); its in-flight count reaches zero at
// exactly three (Settle, reapInflightLocked, and nowhere else). Every one of those must
// end at the release. These tests drive each path for real and assert the same three
// things every time:
//
//	1. the mark moved NO counter (it is a mark, not a release);
//	2. when the path completes, the exiting member's charge is GONE — NE is exactly the
//	   SURVIVING members' blocks, so the release was neither short nor taken out of
//	   somebody else's charge (the asymmetry check);
//	3. the index invariant still holds (blocks == blocksNE here, and nothing
//	   underflowed — a double release panics).
//
// Every test runs at N=1 and N=2 (the twins rule). At N=2 the sibling engine's charge
// is asserted untouched: a release that reaches across engines is the same class of
// bug, one engine over.

import (
	"testing"
	"time"
)

// ---- helpers ----

// neOn is the engine's budget-side (NE) block count, under the scheduler's lock.
func neOn(fx *engineSet, g *engineGroup) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return g.index.UniqueBlocksNELocked()
}

// uniqueOn is the all-seated block count (the other side of the same coin).
func uniqueOn(fx *engineSet, g *engineGroup) int {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return g.index.UniqueBlocksLocked()
}

// chargedPath runs one path test at N=1 and N=2 with the fixture every path needs:
// "keep" (a survivor whose charge must never be touched) and "gone" (the member that
// becomes exiting), BOTH on the fixture's first engine.
//
// fx.tighten closes every other engine with a filler at its whole budget, so both
// members land on e0 at N=1 AND at N=2 — the per-engine assertions then mean the same
// thing at both counts, and the sibling's filler is the cross-engine check. "keep"'s
// prefill completes (Chunk) so its cold charge is off the valve: otherwise the
// stagger cap would hold "gone" at the door instead of seating it.
func chargedPath(t *testing.T, cfg Config, path string, fn func(t *testing.T, fx *engineSet, g *engineGroup)) {
	t.Helper()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "keep", 400, 5*time.Second) // e0 (the tie goes to e0)
		fx.s.Chunk("keep")
		fx.tighten(t) // every other engine full: "gone" has one home
		fx.seatOne(t, "gone", 300, 5*time.Second)
		g := fx.g()
		for _, id := range []string{"keep", "gone"} {
			if _, ok := g.members[id]; !ok {
				t.Fatalf("fixture: %q must be seated on %s", id, g.id)
			}
		}
		if got, want := neOn(fx, g), 700; got != want {
			t.Fatalf("fixture: NE = %d want %d", got, want)
		}
		fn(t, fx, g)
	})
}

// markExitingInflight makes "gone" an exiting member with one in-flight request,
// exactly as the pressure valve's mid-flight class does, and asserts the mark moved no
// counter (the whole point of (iii): it is a mark, not a release).
func markExitingInflight(t *testing.T, fx *engineSet, g *engineGroup, reqID string, confirmed bool) {
	t.Helper()
	fx.s.mu.Lock()
	m := g.members["gone"]
	before, beforeU := g.index.UniqueBlocksNELocked(), g.index.UniqueBlocksLocked()
	m.inflightIDs = map[string]inflightRec{reqID: {since: fx.clock, confirmed: confirmed}}
	m.inflight = 1
	g.index.MarkExitingLocked("gone")
	n, u := g.index.UniqueBlocksNELocked(), g.index.UniqueBlocksLocked()
	fx.s.mu.Unlock()
	if n != before {
		t.Fatalf("MarkExiting moved the budget counter: NE %d -> %d (the mark must move nothing)", before, n)
	}
	if u != beforeU {
		t.Fatalf("MarkExiting moved the all-seated counter: %d -> %d", beforeU, u)
	}
	if !fx.s.Seated("gone") {
		t.Fatal("a marked member is still seated: it drains, it is not evicted")
	}
}

// assertReleased is the assertion every path test ends with: "gone" is unseated, its
// 300 blocks are off the engine's used side, "keep"'s 400 are untouched, the sibling is
// untouched, and no counter diverged.
func assertReleased(t *testing.T, fx *engineSet, g *engineGroup, path string) {
	t.Helper()
	if fx.s.Seated("gone") {
		t.Fatalf("%s: the exiting member is still seated — its blocks are still charged (LEAK)", path)
	}
	if got, want := neOn(fx, g), 400; got != want {
		why := "over-released (somebody else's charge was taken)"
		if got > want {
			why = "never released (LEAK)"
		}
		t.Fatalf("%s: NE = %d want %d — %s", path, got, want, why)
	}
	if got, want := uniqueOn(fx, g), 400; got != want {
		t.Fatalf("%s: all-seated blocks = %d want %d", path, got, want)
	}
	if !fx.s.Seated("keep") {
		t.Fatalf("%s: the release took the OTHER member's seat (asymmetry)", path)
	}
	for _, id := range fx.ids[1:] {
		if got, want := neOn(fx, fx.s.groups[id]), 950; got != want {
			t.Fatalf("%s: sibling %s NE = %d want %d (a release crossed engines)", path, id, got, want)
		}
	}
	fx.s.mu.Lock()
	g.index.checkInvariantLocked() // panics on a double release / divergence
	fx.s.mu.Unlock()
}

// ---- path 1: the ordinary drain (the last in-flight request settles) ----

func TestChargedDrainSettleReleases(t *testing.T) {
	chargedPath(t, defaultConfig(), "settle-drain", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)
		// the request terminates: the finalizer removes the record, sees the member
		// exiting and drained, and unseats it.
		fx.s.SettleOnUsage("gone", "r-gone", g.id, 300, 0, 0, false, fx.clock)
		assertReleased(t, fx, g, "settle-drain")
	})
}

// A settle that does NOT drain the member (a second request still running) must not
// release: the blocks are still pinned. Release happens on the LAST drain, once.
func TestChargedDrainWaitsForTheLastRequest(t *testing.T) {
	chargedPath(t, defaultConfig(), "settle-drain-last", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r1", true)
		fx.s.mu.Lock()
		m := g.members["gone"]
		m.inflightIDs["r2"] = inflightRec{since: fx.clock, confirmed: true}
		m.inflight = 2
		fx.s.mu.Unlock()

		fx.s.SettleOnUsage("gone", "r1", g.id, 300, 0, 0, false, fx.clock)
		if !fx.s.Seated("gone") {
			t.Fatal("one of two in-flight requests settled: the member must stay charged")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("NE = %d want 700 while one request still runs", got)
		}
		fx.s.SettleOnUsage("gone", "r2", g.id, 300, 0, 0, false, fx.clock)
		assertReleased(t, fx, g, "settle-drain (last)")
	})
}

// A LATE/DUPLICATE settle of an id the record no longer holds is a STRAY: it removes
// nothing, so it must not be read as a drain (and must not exit anybody).
func TestChargedDrainStraySettleDoesNotRelease(t *testing.T) {
	chargedPath(t, defaultConfig(), "stray-settle", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r1", true)
		fx.s.mu.Lock()
		m := g.members["gone"]
		m.inflightIDs["r2"] = inflightRec{since: fx.clock, confirmed: true}
		m.inflight = 2
		fx.s.mu.Unlock()

		fx.s.SettleOnUsage("gone", "not-a-known-id", g.id, 300, 0, 0, false, fx.clock)
		if !fx.s.Seated("gone") {
			t.Fatal("an unknown req id removed nothing, so nothing drained: the seat must stand")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("NE = %d want 700 after a stray settle", got)
		}
	})
}

// ---- path 2: engine-gone (the engine disappears under the member) ----

func TestChargedDrainEngineGoneReleases(t *testing.T) {
	chargedPath(t, defaultConfig(), "engine-gone", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)
		if n := fx.s.RemoveEngine(g.id, fx.clock); n == 0 {
			t.Fatal("the teardown must exit the members it tears down")
		}
		if got, want := neOn(fx, g), 0; got != want {
			t.Fatalf("engine-gone: NE = %d want %d — a dead engine's charge leaked", got, want)
		}
	})
}

// ---- path 3: the reap — a termination nobody observed ----
//
// This is the class the reviewer pointed at: the settle is NOT the only way an
// in-flight count reaches zero. reapInflightLocked drops ids whose termination was
// never observed — a client cancel whose forced EndOfStream never reached the plugin,
// or a request that errored before its first chunk. Dropping the LAST id IS a drain, so
// it must finish the exit; the TTL is not a rescue (its idle test needs no queued turn,
// and a member that still has a head is exactly the member that is waiting).
//
// The TTL is pushed out of reach (1e5 s) so the test can only pass on the reap path.
func TestChargedDrainReapOfCancelledRequestReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60 // the campaign shape's bound, scaled down
	cfg.TTLSeconds = 1e5         // no TTL rescue: the reap must do it
	cfg.IdleSeconds = 1e5
	chargedPath(t, cfg, "reap-cancelled", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-cancelled", true)
		fx.s.lastHeads = nil      // no queued heads: the release must not be masked by a re-admission
		fx.tick(61 * time.Second) // the confirmed-id reap bound
		assertReleased(t, fx, g, "reap (cancel, no settle)")
	})
}

// The same for a PROVISIONAL id (picked, never confirmed at PreRequest — an error
// before the first chunk): the 30 s provisional bound drops it. The cold charge is
// still outstanding here (no chunk ever released it), so the exit must release that too.
func TestChargedDrainReapOfUnconfirmedRequestReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.TTLSeconds = 1e5
	cfg.IdleSeconds = 1e5
	chargedPath(t, cfg, "reap-provisional", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-provisional", false)
		fx.s.mu.Lock()
		charged := g.members["gone"].coldEntry
		fx.s.mu.Unlock()
		if charged == 0 {
			t.Fatal("fixture: the entry's cold charge must still be outstanding (no chunk, no settle)")
		}
		fx.s.mu.Lock()
		fx.s.lastHeads = nil
		fx.s.mu.Unlock()
		fx.tick(31 * time.Second) // the provisional reap bound
		assertReleased(t, fx, g, "reap (error before first chunk)")
		if got := fx.s.groups[g.id].coldInFlight; got != 0 {
			t.Fatalf("the cold-entry charge leaked on the reap path: %d want 0", got)
		}
	})
}

// ---- path 4: TTL, and the idle exit while already exiting ----
//
// A member that is exiting with nothing in flight is NOT a rotation victim: the victim
// rules skip it (it is already leaving, and marking it again frees nothing). So the TTL
// sweep is the only thing between it and a permanent charge — and the TTL sweep's own
// test is "no queued turn". A member with a queued head keeps its waiting-since fresh
// on every queue sync, which is the NORMAL state of an oversubscribed engine (measured:
// 97-112 heads in the shared queue at c128). Hence: the release must not depend on the
// TTL at all — it must happen the moment the member is exiting AND drained.
func TestChargedDrainHeadlessIdleReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	chargedPath(t, cfg, "ttl", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)
		fx.s.mu.Lock()
		m := g.members["gone"]
		m.inflightIDs = map[string]inflightRec{}
		m.inflight = 0
		m.lastActivity = fx.clock
		m.waitingSince = time.Time{} // headless: nothing queued for it
		fx.s.lastHeads = nil         // ... and the queue view agrees (no stale head snapshot)
		fx.s.mu.Unlock()
		fx.tick(31 * time.Second) // past the TTL
		assertReleased(t, fx, g, "ttl")
	})
}

// The same member WITH a queued head still on the books (the common case at c128: it has
// its next turn waiting). The TTL's idle test never fires for it and rotation and the
// pressure valve both refuse an exiting member — so the decision point is the only thing
// between it and a permanent charge.
func TestChargedDrainExitingWithQueuedHeadReleases(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60
	chargedPath(t, cfg, "exiting-with-head", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)
		fx.s.mu.Lock()
		m := g.members["gone"]
		m.inflightIDs = map[string]inflightRec{}
		m.inflight = 0
		m.lastActivity = fx.clock
		m.waitingSince = fx.clock.Add(-10 * time.Second) // its next turn is queued
		fx.s.mu.Unlock()
		// a Pick with no heads: the reconciliation runs, nothing can be re-admitted
		if i := fx.pick(); i != -1 {
			t.Fatalf("no heads: nothing to dispatch, got %d", i)
		}
		assertReleased(t, fx, g, "exiting-with-head")
	})
}

// ... and while it DOES have a request in flight it must never be dispatched for its
// queued head: the seat is not being handed out twice on the way out.
func TestChargedDrainExitingIsNeverDispatched(t *testing.T) {
	chargedPath(t, defaultConfig(), "exiting-no-dispatch", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)
		if _, i := fx.pickEngine(head("gone", 10*time.Second, bytesFor(300))); i >= 0 {
			t.Fatal("an exiting member must not be dispatched for its queued head")
		}
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("the member is still draining: NE = %d want 700", got)
		}
	})
}

// ---- path 5: Forget, and the double-release asymmetry ----

// Forget is the OTHER half of the one release point (the TTL sweep and the silence GC
// use it). On a SEATED member it must release both counters — the exiting member's
// included — and a later Unseat must be a harmless no-op rather than a second release
// (which would underflow and panic). Then a fresh residency must start non-exiting and
// charge exactly once.
func TestChargedDrainForgetReleasesAndUnseatIsIdempotent(t *testing.T) {
	chargedPath(t, defaultConfig(), "forget", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true)

		fx.s.mu.Lock()
		g.index.ForgetLocked("gone")
		g.index.checkInvariantLocked()
		fx.s.mu.Unlock()
		if got, want := neOn(fx, g), 400; got != want {
			t.Fatalf("Forget on a seated exiting member: NE = %d want %d", got, want)
		}
		// the double release: Unseat and a second Forget must both be no-ops
		fx.s.mu.Lock()
		g.index.UnseatLocked("gone")
		g.index.ForgetLocked("gone")
		g.index.checkInvariantLocked()
		ok, _ := g.index.TrySeatLocked("gone", 950)
		n := g.index.UniqueBlocksNELocked()
		fx.s.mu.Unlock()
		if !ok {
			t.Fatal("the record must be re-seatable after the Forget")
		}
		if got, want := n, 400; got != want {
			// a forgotten record has no block set left, so the fresh residency adds
			// nothing: the 400 is the survivor's. What matters is that neither the
			// release nor the re-seat moved anybody else's charge.
			t.Fatalf("a fresh residency after Forget: NE = %d want %d (charged once)", got, want)
		}
		if g.index.Exiting("gone") {
			t.Fatal("a fresh residency must not inherit the exiting flag")
		}
	})
}

// ---- path 6: the other direction — a mark on a member that holds no charge ----

// MarkExiting refuses an unseated session (a seated member is the only thing that can
// be exiting), so no path can release a charge that was never taken. That symmetry is
// what makes the unconditional release in exitLocked/Forget safe: the counters are
// rebuilt from the seated sets, so an unseated record contributes nothing to release.
func TestChargedDrainMarkOnUnseatedRecordIsANoOp(t *testing.T) {
	chargedPath(t, defaultConfig(), "mark-unseated", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.mu.Lock()
		g.index.MarkExitingLocked("never-seated")
		g.index.MarkExitingLocked("keep-then-gone")
		g.index.checkInvariantLocked()
		n := g.index.UniqueBlocksNELocked()
		fx.s.mu.Unlock()
		if got, want := n, 700; got != want {
			t.Fatalf("marking an unseated id moved the budget counter: %d want %d", got, want)
		}
		if g.index.Exiting("keep-then-gone") {
			t.Fatal("an unseated session is not exiting")
		}
	})
}
