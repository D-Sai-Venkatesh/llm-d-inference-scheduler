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

// THE TIER SHADOW's SUITE (PREREG-multipod-offload, "THE OFFLOAD SOLUTION" piece 1).
//
// Two layers, both required:
//
//	(1) the MODEL's own semantics, unit-level with no scheduler in the way — touch to
//	    MRU, insert on settle, evict the tail past capacity, capacity from the two
//	    deployment facts, and nil-safety (a deployment without a tier pays nothing);
//	(2) the WIRING: the scheduler's dispatch and settle hooks are the only things that
//	    may feed it, and they must feed it with the RIGHT keys — a request's own chain
//	    at settle, never the member's whole set (which contains turns that have not run).
//
// The wiring half runs at N=1 and N=2 through forEngines: the shadow is per ENGINE, so a
// rule that only held on the single-engine fixture would be a bug by definition.

import (
	"testing"
	"time"
)

// ---- (1) the model ----

// TestTierShadowCapacityFromTheDeploymentFacts pins the ONE arithmetic the shadow owns
// (bytes ÷ bytes-per-block) including its degenerate answers: no tier, a tier smaller
// than one block, and exactly one block.
func TestTierShadowCapacityFromTheDeploymentFacts(t *testing.T) {
	const kb = 6291456 // one 64-token block of Qwen3-30B-A3B at TP4, in bytes
	cases := []struct {
		name          string
		bytes, kvByte int64
		want          int
	}{
		{"no tier configured", 0, 0, 0},
		{"bytes given but no block size", 549755813888, 0, 0},
		{"the campaign tier: 512 GiB / 6 MiB blocks", 549755813888, kb, 87381},
		{"smaller than one block", kb - 1, kb, 0},
		{"exactly one block", kb, kb, 1},
	}
	for _, tc := range cases {
		if got := tierBlocksFor(tc.bytes, tc.kvByte); got != tc.want {
			t.Errorf("%s: tierBlocksFor(%d, %d) = %d, want %d", tc.name, tc.bytes, tc.kvByte, got, tc.want)
		}
	}
}

// TestTierShadowLRUSemantics is the model's whole contract, as the four rules the frozen
// spec names: TOUCH moves a held block to MRU (and inserts nothing), INSERT adds a new
// block at MRU, the tail is evicted past capacity, and a nil shadow holds nothing.
func TestTierShadowLRUSemantics(t *testing.T) {
	if newTierShadow(0) != nil {
		t.Fatal("capacity 0 must be a NIL shadow (no tier configured = nothing modelled)")
	}
	if (&tierShadow{}).Enabled() {
		t.Fatal("a zero-value shadow must not report Enabled")
	}
	var none *tierShadow
	if none.Holds(1) || none.Len() != 0 || none.Capacity() != 0 || none.Version() != 0 {
		t.Fatal("a nil shadow must answer 'holds nothing' to every query (nil-safety)")
	}
	none.Touch([]Hash{1})
	none.Insert([]Hash{1}) // must not panic
	if none.Len() != 0 {
		t.Fatal("a nil shadow must swallow writes")
	}

	a, b, c, d := Hash(10), Hash(20), Hash(30), Hash(40)
	sh := newTierShadow(3)

	// INSERT: new blocks enter at MRU.
	sh.Insert([]Hash{a, b, c})
	if sh.Len() != 3 || !sh.Holds(a) || !sh.Holds(b) || !sh.Holds(c) {
		t.Fatalf("after inserting {a,b,c} the tier holds %d blocks (%v)", sh.Len(), sh.where)
	}
	if got := sh.order.Front().Value.(Hash); got != c {
		t.Fatalf("MRU = %d, want c (the last inserted)", got)
	}
	if got := sh.order.Back().Value.(Hash); got != a {
		t.Fatalf("LRU = %d, want a (the first inserted)", got)
	}

	// TOUCH: a held block moves to MRU; an unheld one is NOT inserted.
	v0 := sh.Version()
	sh.Touch([]Hash{a, d})
	if got := sh.order.Front().Value.(Hash); got != a {
		t.Fatalf("after Touch(a) the MRU = %d, want a", got)
	}
	if sh.Holds(d) {
		t.Fatal("Touch must not INSERT: a dispatch alone does not store a block")
	}
	if sh.Version() != v0 {
		t.Fatal("a recency-only move must not bump the residency version (the estimate cache keys on membership, not order)")
	}

	// EVICT: inserting d past a capacity of 3 pushes out the LRU tail, which is now b
	// (a was touched, c was more recent than b).
	sh.Insert([]Hash{d})
	if sh.Len() != 3 {
		t.Fatalf("after inserting past capacity the tier holds %d blocks, want 3", sh.Len())
	}
	if sh.Holds(b) {
		t.Fatal("b was the LRU tail after Touch(a) and must have been evicted")
	}
	for _, h := range []Hash{a, c, d} {
		if !sh.Holds(h) {
			t.Fatalf("block %d must still be held: %v", h, sh.where)
		}
	}
	if sh.Version() <= v0 {
		t.Fatal("an insert/eviction must bump the residency version (it changes what the tier holds)")
	}

	// a settle of an ALREADY-held block does not re-rank it: the dispatch Touch owns
	// recency, so a settle cannot jump a block dispatched after it.
	front := sh.order.Front().Value.(Hash)
	sh.Insert([]Hash{a})
	if got := sh.order.Front().Value.(Hash); got != front {
		t.Fatalf("Insert of a held block moved it to MRU (%d -> %d): settle must not re-rank", front, got)
	}

	// capacity 1: one insert evicts everything older, and the newest insert survives.
	tiny := newTierShadow(1)
	tiny.Insert([]Hash{a})
	tiny.Insert([]Hash{b})
	if tiny.Len() != 1 || !tiny.Holds(b) || tiny.Holds(a) {
		t.Fatalf("a 1-block tier must hold only the newest block, holds %v", tiny.where)
	}
}

// ---- (2) the wiring ----

// TestTierShadowFedByDispatchAndSettle pins WHICH keys reach the shadow, and when. The
// spec's two hooks are not interchangeable: a queued turn is NOT stored, a dispatched
// turn only REFRESHES what is already stored, and a SETTLED turn is what enters.
func TestTierShadowFedByDispatchAndSettle(t *testing.T) {
	forEngines(t, tierConfig(400), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second) // seats AND dispatches turn 1
		g := fx.s.groups[fx.s.bindings["m"]]
		turn1 := keys(100, seedFor("m"))
		if g.tier.Len() != 0 {
			t.Fatalf("a DISPATCHED turn must not be in the tier yet (it is in HBM, not stored), holds %d", g.tier.Len())
		}
		settle(fx.s, "m", 100, 100)
		if got := g.tier.Len(); got != 100 {
			t.Fatalf("after the settle the tier holds %d blocks, want the request's own 100", got)
		}
		for _, h := range turn1 {
			if !g.tier.Holds(h) {
				t.Fatalf("block %d of the settled request must be in the tier", h)
			}
		}

		// a QUEUED (arrived, not run) turn must never reach the tier — neither while it
		// waits nor when it is dispatched, only when it settles.
		turn2 := keys(40, seedFor("m-turn2"))
		fx.s.mu.Lock()
		g.index.UpdateWaitingLocked("m", "r-m", turn2)
		fx.s.mu.Unlock()
		if g.tier.Holds(turn2[0]) {
			t.Fatal("a WAITING turn's blocks must never be in the tier shadow")
		}
		dispatch(t, fx.s, "m", turn2, bytesFor(40))
		if g.tier.Holds(turn2[0]) {
			t.Fatal("a DISPATCH must not insert new blocks: the store is on settle")
		}
		if !g.tier.Holds(turn1[0]) {
			t.Fatal("the previous turn's stored blocks must still be held (no eviction happened)")
		}
		settle(fx.s, "m", 140, 100)
		if !g.tier.Holds(turn2[0]) {
			t.Fatal("the settled turn's blocks must be inserted")
		}
		if got, want := g.tier.Len(), 140; got != want {
			t.Fatalf("the tier holds %d blocks, want %d (100 stored + 40 new)", got, want)
		}
	})
}

// TestTierShadowEvictsTheTailAcrossMembers drives the tier past capacity through the
// PUBLIC path (three members' settles) and asserts the LRU TAIL is what leaves — the
// mechanism behind the measured survival cliff, and the reason a returner's cost rises
// with the time it has been away.
func TestTierShadowEvictsTheTailAcrossMembers(t *testing.T) {
	forEngines(t, tierConfig(250), func(t *testing.T, fx *engineSet) {
		// every engine but the first is closed off, so all three members are seated on
		// ONE engine (at N=1 there is only that engine) and share one tier.
		fx.tighten(t)
		seatAndSettle := func(id string, blocks int, wantEngine string) (*engineGroup, []Hash) {
			t.Helper()
			fx.seatOne(t, id, blocks, 5*time.Second)
			engine := fx.s.bindings[id]
			if wantEngine != "" && engine != wantEngine {
				t.Fatalf("member %s seated on %s, wanted %s: the three members must share one tier", id, engine, wantEngine)
			}
			settle(fx.s, id, blocks, blocks)
			return fx.s.groups[engine], keys(blocks, seedFor(id))
		}
		g, first := seatAndSettle("first", 100, "")
		_, second := seatAndSettle("second", 100, g.id)
		if got := g.tier.Len(); got != 200 {
			t.Fatalf("the tier holds %d blocks after two 100-block settles, want 200 (capacity 250)", got)
		}
		if !g.tier.Holds(first[0]) || !g.tier.Holds(second[0]) {
			t.Fatal("both stored sets must be held while the tier has room")
		}
		_, third := seatAndSettle("third", 100, g.id) // 300 blocks into a 250-block tier
		if got := g.tier.Len(); got != 250 {
			t.Fatalf("the tier holds %d blocks, want exactly its capacity 250", got)
		}
		if !g.tier.Holds(third[0]) {
			t.Fatal("the newest blocks must be held (they were inserted last)")
		}
		if g.tier.Holds(first[0]) {
			t.Fatal("'first' was stored first and never touched again: its oldest blocks must be the evicted tail")
		}
		if !g.tier.Holds(first[99]) {
			t.Fatal("only the TAIL goes: 'first' was 100 blocks, the tier overflowed by 50, so its MRU half must survive")
		}
	})
}
