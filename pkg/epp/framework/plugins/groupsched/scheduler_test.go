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

// THE CORE MECHANISM SUITE.
//
// EVERY test here runs at N=1 AND N=2 (the twins rule): the single-engine stack is
// the N=1 case of the same code, so a behaviour that only holds at one engine count
// is a bug by definition. forEngines(t, fn) is the twin runner; fn receives a
// fixture whose engines are named e0, e1 and whose capacity is 1000 blocks each
// (budget 950) unless the test overrides it.

import (
	"fmt"
	"testing"
	"time"
)

// ---- shared helpers ----

// keys builds a deterministic distinct-key set of n blocks.
func keys(n int, seed Hash) []Hash {
	out := make([]Hash, 0, n)
	prev := seed
	for i := 0; i < n; i++ {
		prev = prev*0x9E3779B97F4A7C15 ^ Hash(i+1) ^ 0xC2B2AE3D27D4EB4F
		out = append(out, prev)
	}
	return out
}

// bytesFor converts blocks to the wire size that estimates them at 4 bytes/token
// (the estimate path's convention).
func bytesFor(blocks int) int64 { return int64(blocks) * 4 * 64 }

// seedFor derives a DISTINCT block key set per (member, size) — two members must not
// share blocks by accident, or the index's dedup would hide the second one's footprint.
func seedFor(id string) Hash {
	h := Hash(1469598103934665603)
	for i := 0; i < len(id); i++ {
		h = (h ^ Hash(id[i])) * 1099511628211
	}
	return h
}

// head is one queue head for member id (ago = how long it has waited).
func head(id string, ago time.Duration, sizeBytes int64) Head {
	return Head{Member: id, ReqID: "r-" + id, Arrival: time.Now().Add(-ago), SizeBytes: sizeBytes}
}

// dispatch drives the released-request path: Pick chose the head, the producer
// tokenized, PreRequest corrects with the exact keys.
func dispatch(t *testing.T, s *Scheduler, id string, k []Hash, size int64) {
	t.Helper()
	s.Dispatched(id, "r-"+id, k, size, time.Now())
}

// settleChunk is the first response chunk (prefill complete) + the natural settle.
func settle(s *Scheduler, id string, prompt, cached int) {
	s.Chunk(id)
	s.Settle(id, "", struct{ Prompt, Cached int }{prompt, cached}, true, time.Now())
}

// engineSet is one twin fixture: n engines each with measured capacity.
type engineSet struct {
	s     *Scheduler
	ids   []string
	now   time.Time
	sim   int
	clock time.Time
}

// forEngines runs fn at N=1 and at N=2 with identical per-engine capacity.
func forEngines(t *testing.T, cfg Config, fn func(t *testing.T, fx *engineSet)) {
	t.Helper()
	forEnginesCap(t, cfg, 1000, fn)
}

// forEnginesCap is forEngines with a chosen per-engine measured capacity in blocks.
func forEnginesCap(t *testing.T, cfg Config, capacity int, fn func(t *testing.T, fx *engineSet)) {
	t.Helper()
	for _, n := range []int{1, 2} {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
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
			fn(t, fx)
		})
	}
}

// member finds a seated member and the engine it holds, on any engine.
func (fx *engineSet) member(t *testing.T, id string) (*member, *engineGroup) {
	t.Helper()
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	for _, g := range fx.allGroups() {
		if m, ok := g.members[id]; ok {
			return m, g
		}
	}
	t.Fatalf("member %s is not seated on any engine", id)
	return nil, nil
}

// g is the fixture's FIRST engine group (the single-engine assertions' target; at
// N=2 a test that means "any engine" must say so explicitly).
func (fx *engineSet) g() *engineGroup { return fx.s.groups[fx.ids[0]] }

// allGroups is every engine group of the fixture, in id order.
func (fx *engineSet) allGroups() []*engineGroup { return fx.s.activeGroupsLocked() }

// tick advances the fixture's virtual clock by d and runs the scheduler's ticker.
func (fx *engineSet) tick(d time.Duration) {
	fx.clock = fx.clock.Add(d)
	fx.s.Tick(fx.clock)
}

// pick runs one Pick over the given heads at the fixture's clock.
func (fx *engineSet) pick(hs ...Head) int {
	return fx.s.PickFromHeads(hs, fx.clock)
}

// pickEngine runs one Pick and returns the engine it served from.
func (fx *engineSet) pickEngine(hs ...Head) (string, int) {
	return fx.s.PickFromHeadsEngine(hs, fx.clock)
}

// seatOn seats one member on a NAMED engine through the public path (the seat-time
// choice must agree with the ask, or the test's premise is broken).
func (fx *engineSet) seatOn(t *testing.T, engine, id string, blocks int, ago time.Duration) {
	t.Helper()
	fx.seatOne(t, id, blocks, ago)
	if got := fx.s.bindings[id]; got != engine {
		t.Fatalf("member %s seated on %s, wanted %s", id, got, engine)
	}
}

// tighten makes room scarce on the FIRST engine and absent everywhere else: the
// test seats its own members first, then every OTHER engine is filled to its budget
// with one 950-block filler (the most-free-budget rule puts each one there). A test
// that asserts "only one seat fits" is then true at N=1 and at N=2 alike.
func (fx *engineSet) tighten(t *testing.T) {
	t.Helper()
	for i, e := range fx.ids {
		if i == 0 {
			continue
		}
		id := "closed-" + e
		fx.seatOne(t, id, 950, 5*time.Second)
	}
	fx.s.mu.Lock()
	for _, g := range fx.allGroups() {
		for _, m := range g.members {
			m.lastActivity = fx.clock
		}
	}
	fx.s.mu.Unlock()
}

// newSched is the legacy single-engine fixture// newSched is the legacy single-engine fixture (the "" engine id, capacity 1000).
func newSched(t *testing.T) *Scheduler {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil { // budget 950 blocks
		t.Fatal(err)
	}
	return s
}

// seatOne seats one member on the fixture through the public path.
func (fx *engineSet) seatOne(t *testing.T, id string, blocks int, ago time.Duration) {
	t.Helper()
	h := head(id, ago, bytesFor(blocks))
	if i := fx.pick(h); i < 0 {
		t.Fatalf("member %s must be seated", id)
	}
	dispatch(t, fx.s, id, keys(blocks, seedFor(id)), bytesFor(blocks))
}

// ---- the tests ----

// A member with a queued head dispatches immediately once seated; an unknown member
// is never seated by dispatching to it (non-members never run unseated).
func TestMemberServedAndUnknownNeverSeated(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		if i := fx.pick(head("m", time.Second, bytesFor(100))); i != 0 {
			t.Fatalf("a seated member's head must dispatch, got %d", i)
		}
		if !fx.s.Seated("m") {
			t.Fatal("the member must be seated")
		}
		settle(fx.s, "m", 100, 100)
		if fx.s.Sessions() != 1 {
			t.Fatalf("only the seated member has a record, got %d", fx.s.Sessions())
		}
	})
}

// Newcomers go first among the WAITING heads (D1 as f0p ran it), and the seat goes
// to the newcomer even though the returning member has waited longer.
func TestNewcomerFirstThenOldest(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// fill EVERY engine's budget so exactly ONE of the two queued heads can be
		// seated (the twin rule: at N=2 the second engine must not simply take it)
		// the filler lands on e0 (id tie), THEN the other engines are closed off
		fx.seatOne(t, "filler", 800, 5*time.Second)
		settle(fx.s, "filler", 800, 800)
		fx.tighten(t)
		// the returning member HAS been served before (non-newcomer); the other has not
		fx.s.mu.Lock()
		fx.s.everSeated["ret"] = struct{}{}
		fx.s.mu.Unlock()
		hs := []Head{head("ret", 30*time.Second, bytesFor(100)), head("new", time.Second, bytesFor(100))}
		fx.pick(hs...)
		if !fx.s.Seated("new") {
			t.Fatal("the newcomer must be seated first (D1 newcomer-first)")
		}
		if fx.s.Seated("ret") {
			t.Fatal("only one seat fits: the returning member must wait behind the newcomer")
		}
	})
}

// Among two WAITING heads of the same class, the OLDEST arrival is the head, and it
// is the one that gets the seat.
func TestOldestWaiterIsTheHead(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// the filler lands on e0 (id tie), THEN the other engines are closed off
		fx.seatOne(t, "filler", 800, 5*time.Second)
		settle(fx.s, "filler", 800, 800)
		fx.tighten(t)
		hs := []Head{head("young", time.Second, bytesFor(100)), head("old", 30*time.Second, bytesFor(100))}
		fx.pick(hs...)
		if !fx.s.Seated("old") {
			t.Fatal("the oldest waiting head must be seated first")
		}
		if fx.s.Seated("young") {
			t.Fatal("the younger head must wait")
		}
	})
}

// The quantum Q is the tenure rule: a member that completed Q busy periods is
// rotation-eligible, one mid-quantum is not. Q4 vs Q8 is a VALUE, same code.
func TestQuantumIsTheTenureRule(t *testing.T) {
	for _, q := range []int{4, 8} {
		q := q
		t.Run(fmt.Sprintf("Q=%d", q), func(t *testing.T) {
			cfg := defaultConfig()
			cfg.Quantum = q
			forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
				fx.seatOne(t, "m", 400, time.Second)
				settle(fx.s, "m", 400, 400) // one busy period completes (main chain drains)
				fx.s.mu.Lock()
				m := fx.g().members["m"]
				bp := m.busyPeriods
				_, eligible := fx.s.eligibleLocked(m, fx.clock)
				fx.s.mu.Unlock()
				if bp != 1 {
					t.Fatalf("one drained main-chain request is one busy period, got %d", bp)
				}
				if eligible {
					t.Fatalf("Q=%d: a member with 1 busy period must NOT be rotation-eligible", q)
				}
				for i := 1; i < q; i++ {
					fx.s.mu.Lock()
					m.busyPeriods++
					fx.s.mu.Unlock()
				}
				fx.s.mu.Lock()
				reason, ok := fx.s.eligibleLocked(m, fx.clock)
				fx.s.mu.Unlock()
				if !ok || reason != "quantum-complete" {
					t.Fatalf("Q=%d: %d busy periods must be eligible (quantum-complete), got %q %v", q, q, reason, ok)
				}
			})
		})
	}
}

// A member in flight is NEVER eligible: rotation must not evict running work.
func TestInFlightNeverEligible(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.mu.Lock()
		m := fx.g().members["m"]
		m.busyPeriods = 100 // way past any Q
		m.inflight = 1
		_, ok := fx.s.eligibleLocked(m, fx.clock)
		fx.s.mu.Unlock()
		if ok {
			t.Fatal("a member with a request in flight must never be rotation-eligible")
		}
	})
}

// The budget valve (MEMORY PRESSURE): when an engine's footprint grows past its
// budget, the victim is the DONE member (no queued turn) — never the member that is
// waiting to run (f0p's order).
func TestBudgetValvePrefersTheDoneMember(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// order matters: the first member lands on e0 (id tie), THEN the other
		// engines are closed off, so the second member has to share e0
		fx.seatOne(t, "done", 450, 5*time.Second)
		settle(fx.s, "done", 450, 450) // finished its turn: no queued head
		fx.tighten(t)
		fx.seatOne(t, "busy", 450, 4*time.Second)
		// `busy` wants to run again (a queued head), and its request GROWS to 200
		// blocks: 450 + 450 + 200 > 950 budget => the valve must fire
		fx.s.mu.Lock()
		fx.g().members["busy"].waitingSince = fx.clock.Add(-time.Second)
		fx.g().members["busy"].lastActivity = fx.clock
		fx.s.mu.Unlock()
		if i := fx.pick(head("busy", time.Second, bytesFor(200))); i != 0 {
			t.Fatalf("the member's grown head must dispatch, got %d", i)
		}
		dispatch(t, fx.s, "busy", keys(200, 77), bytesFor(200)) // the growth lands
		fx.s.Tick(fx.clock)                                     // and the valve enforces it
		fx.s.mu.Lock()
		_, doneSeated := fx.g().members["done"]
		_, busySeated := fx.g().members["busy"]
		fx.s.mu.Unlock()
		if doneSeated {
			t.Fatal("the DONE member (no queued turn) must be the budget victim")
		}
		if !busySeated {
			t.Fatal("the member with a queued turn must survive (f0p's valve order)")
		}
		if snap := fx.s.Snapshot(); snap.Exits["budget-noqueue"] == 0 && snap.Exits["budget-idle"] == 0 {
			t.Fatalf("the exit must be a budget-class exit, got %v", snap.Exits)
		}
	})
}

// No eviction while in flight: a member with a running request is marked exiting at
// most, never removed, until the drain completes.
func TestNoEvictionWhileInFlight(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 400, 2*time.Second)
		fx.s.Chunk("m") // m's prefill landed (its turn keeps decoding); the cold slot is free
		fx.seatOne(t, "m2", 400, time.Second)
		// m's first turn is still running when its second (grown) head is served
		i := fx.pick(head("m", time.Millisecond, bytesFor(700)))
		if i != 0 {
			t.Fatalf("the member's second head must be served (in-flight cap), got %d", i)
		}
		dispatch(t, fx.s, "m", keys(700, 3), bytesFor(700))
		fx.s.Tick(fx.clock)
		if !fx.s.Seated("m") {
			t.Fatal("a member in flight must never be evicted (marked exiting at most)")
		}
		settle(fx.s, "m", 400, 0)
		settle(fx.s, "m", 700, 0)
		if fx.s.Seated("m") {
			t.Fatal("after the drain the exiting member must leave")
		}

	})
}

// AN EXITING MEMBER'S BLOCKS STAY ON THE USED SIDE UNTIL IT DRAINS. Marking a
// member exiting says "do not dispatch, do not re-seat" — it says nothing about the
// engine, which keeps the member's blocks pinned until the running request finishes.
// Excluding them from the budget at the mark was a plain accounting bug: the budget
// then believed it had 14-18 % of the pool free while the pool was still paying for
// it (measured on the multi-pod wave).
func TestExitingMemberStaysOnTheUsedSideUntilDrain(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		for i := range fx.ids {
			fx.seatOne(t, fmt.Sprintf("m%d", i), 400, time.Duration(5+i)*time.Second)
		}
		min, g := fx.member(t, "m0")
		fx.s.mu.Lock()
		// in flight and marked exiting (the mid-flight pressure class)
		min.inflightIDs = map[string]inflightRec{"r-m0": {since: fx.clock, confirmed: true}}
		min.inflight = 1
		freeBefore := fx.s.freeBudgetLocked(g)
		neBefore := g.index.UniqueBlocksNELocked()
		g.index.MarkExitingLocked("m0")
		neAfter := g.index.UniqueBlocksNELocked()
		freeAfter := fx.s.freeBudgetLocked(g)
		fx.s.mu.Unlock()
		if neAfter != neBefore {
			t.Fatalf("marking a member exiting released %d blocks from the used side (NE %d -> %d)",
				neBefore-neAfter, neBefore, neAfter)
		}
		if freeAfter != freeBefore {
			t.Fatalf("the mark freed budget (free %d -> %d): the engine still holds those blocks",
				freeBefore, freeAfter)
		}
		// the drain is what releases them
		settle(fx.s, "m0", 400, 400)
		fx.s.mu.Lock()
		defer fx.s.mu.Unlock()
		if _, seated := g.members["m0"]; seated {
			t.Fatal("a drained exiting member must leave")
		}
		if got := g.index.UniqueBlocksNELocked(); got >= neBefore {
			t.Fatalf("the drain did not release the member's blocks: NE %d want < %d", got, neBefore)
		}
	})
}

// An exiting member that cannot be freed must END the enforcement pass: marking is
// idempotent and its blocks stay charged, so re-picking it would spin the loop
// forever under the scheduler's mutex.
func TestExitingVictimDoesNotSpinEnforcement(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		m, g := fx.member(t, "m")
		fx.s.mu.Lock()
		m.inflightIDs = map[string]inflightRec{"r-m": {since: fx.clock, confirmed: true}}
		m.inflight = 1
		g.index.MarkExitingLocked("m")
		// way over budget with the only member already exiting
		g.budgetBlocks = 100
		fx.s.mu.Unlock()
		// The enforcement pass runs BEFORE Tick's queue re-sync, so a clock this far
		// ahead leaves the queue view untrustworthy — the one path that names an
		// already-exiting member as the victim.
		fx.clock = fx.clock.Add(60 * time.Second)
		done := make(chan struct{})
		go func() {
			fx.s.Tick(fx.clock)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("Tick spun: an exiting in-flight victim was re-picked forever")
		}
		if !fx.s.Seated("m") {
			t.Fatal("the exiting member must still be seated (it drains, it is not evicted)")
		}
	})
}

// The cold-entry valve: one cold entry is ALWAYS allowed (R11); a second waits until
// the first's prefill completes — and the cap is PER ENGINE, so at N=2 the second
// oversized entry may seat immediately on the sibling engine.
//
// The cap is DERIVED from the engine: 10,000 blocks at 0.95 leaves 500 blocks of
// headroom = 32,000 tokens, so a 1,000-block entry is twice the cap and still fits
// the budget by a mile — the valve, not the budget, is the binding constraint.
func TestColdEntryValveIsPerEngine(t *testing.T) {
	forEnginesCap(t, defaultConfig(), 10000, func(t *testing.T, fx *engineSet) {
		big := bytesFor(1000) // twice the derived cap (500 blocks), far inside the budget
		if got, want := fx.g().coldCapTokens(0.95, 64), int64(500*64); got != want {
			t.Fatalf("derived cold cap: %d want %d", got, want)
		}
		if i := fx.pick(head("big1", time.Second, big)); i != 0 {
			t.Fatal("a big newcomer must be seated when nothing cold is in flight (R11)")
		}
		dispatch(t, fx.s, "big1", keys(1000, 1), big)
		h2 := []Head{head("big2", time.Second, big)}
		i := fx.pick(h2...)
		if len(fx.ids) == 1 {
			if i != -1 {
				t.Fatalf("the second big entry must wait while the first's prefill is cold (got %d)", i)
			}
			fx.s.Chunk("big1")
			if i := fx.pick(h2...); i != 0 {
				t.Fatal("the second big entry must seat after the first's prefill completes")
			}
			return
		}
		// N=2: the cap is per engine, so the sibling has its own cold slot
		if i != 0 {
			t.Fatalf("the second cold entry must seat on the SIBLING engine (per-engine cap), got %d", i)
		}
		fx.s.mu.Lock()
		e1, e2 := fx.s.bindings["big1"], fx.s.bindings["big2"]
		fx.s.mu.Unlock()
		if e1 == e2 {
			t.Fatalf("two oversized cold entries must not share an engine's cold slot: both on %s", e1)
		}
	})
}

// Anonymous traffic (R9) is served when no member head is dispatchable, is never
// tracked, and never outranks a member's head.
func TestAnonymousPassThrough(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		anon := Head{Member: "", ReqID: "anon-1", Arrival: fx.clock.Add(-time.Second), SizeBytes: 1000}
		if i := fx.pick(anon); i != 0 {
			t.Fatalf("an anonymous head must be served when no member head is dispatchable, got %d", i)
		}
		if fx.s.Sessions() != 0 {
			t.Fatal("anonymous traffic must not be tracked")
		}
		fx.seatOne(t, "m", 10, time.Second)
		anonLater := Head{Member: "", ReqID: "anon-2", Arrival: fx.clock, SizeBytes: 1000}
		if i := fx.pick(anonLater, head("m", time.Second, bytesFor(10))); i != 1 {
			t.Fatalf("the member's head must outrank the anonymous one: got %d", i)
		}
	})
}

// Hold the line: a blocked head does not let a later, smaller head jump it.
func TestHoldTheLine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// both residents must live on ONE engine, so the tests' own members are seated
		// first (the first lands on e0 by the id tie) and THEN the other engines are
		// closed off; a resident that still lands elsewhere makes the shape invalid
		fx.seatOne(t, "m1", 450, 3*time.Second)
		settle(fx.s, "m1", 450, 450)
		fx.tighten(t)
		fx.seatOne(t, "m2", 450, 2*time.Second)
		m1, g1 := fx.member(t, "m1")
		m2, g2 := fx.member(t, "m2")
		if g1 != g2 {
			t.Skip("residents landed on different engines; the hold-the-line shape needs them together")
		}
		fx.s.mu.Lock()
		m1.waitingSince = fx.clock.Add(-3 * time.Second)
		m1.lastActivity = fx.clock
		m2.waitingSince = fx.clock.Add(-2 * time.Second)
		m2.lastActivity = fx.clock
		fx.s.mu.Unlock()
		hs := []Head{
			head("m1", 3*time.Second, bytesFor(400)),
			head("bighead", 2*time.Second, bytesFor(900)),
			head("small", 100*time.Millisecond, bytesFor(100)),
			head("m2", 2*time.Second, bytesFor(400)),
		}
		for round := 0; round < 4; round++ {
			i := fx.pick(hs...)
			if i == 2 {
				t.Fatal("hold the line violated: small jumped the blocked big head")
			}
			if i >= 0 && hs[i].Member != "" {
				dispatch(t, fx.s, hs[i].Member, keys(400, 9), bytesFor(400))
				settle(fx.s, hs[i].Member, 400, 400)
			}
		}
	})
}

// No starvation: under steady load every waiter is eventually seated (rotation and
// the valve both fire, the queue keeps moving).
func TestNoStarvation(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// three residents fill the budget; eight waiters queue
		for _, id := range []string{"m0", "m1", "m2"} {
			fx.seatOne(t, id, 300, time.Second)
			settle(fx.s, id, 300, 300)
		}
		var waiters []string
		for i := 0; i < 8; i++ {
			waiters = append(waiters, fmt.Sprintf("w%d", i))
		}
		gotSeat := map[string]bool{}
		// The synthetic head list has to rotate: with the cold-entry cap derived from
		// the engine (50 blocks here) a seam is only admitted while the engine's cold
		// slot is free, and a slot is freed when the seated member's head RUNS. Two
		// heads with the same synthetic arrival let one member win every Pick, so the
		// other stayed seated-but-never-dispatched and the test measured that artifact
		// instead of starvation. The least-served member gets the oldest head.
		served := map[string]int{}
		maxServed := func() int {
			m := 0
			for _, n := range served {
				if n > m {
					m = n
				}
			}
			return m
		}
		seatedCount := func() int {
			n := 0
			for _, id := range waiters {
				if fx.s.Seated(id) || gotSeat[id] {
					gotSeat[id] = true
					n++
				}
			}
			return n
		}
		allHeads := func() []Head {
			hs := make([]Head, 0, len(waiters)+3)
			age := func(id string) time.Duration {
				return time.Duration(maxServed()+1-served[id]) * time.Millisecond
			}
			for _, id := range []string{"m0", "m1", "m2"} {
				if fx.s.Seated(id) {
					hs = append(hs, head(id, age(id), bytesFor(300)))
				}
			}
			for i, id := range waiters {
				if fx.s.Seated(id) {
					hs = append(hs, head(id, age(id), bytesFor(300)))
				} else {
					hs = append(hs, head(id, time.Duration(len(waiters)-i)*100*time.Millisecond, bytesFor(300)))
				}
			}
			return hs
		}
		for cycles := 0; seatedCount() < len(waiters) && cycles < 80; cycles++ {
			for round := 0; round < 6; round++ {
				hs := allHeads()
				i := fx.pick(hs...)
				if i < 0 {
					break
				}
				id := hs[i].Member
				if id != "" && fx.s.Seated(id) {
					dispatch(t, fx.s, id, keys(300, Hash(uint64(id[len(id)-1]))), bytesFor(300))
					settle(fx.s, id, 300, 300)
					served[id]++
				}
			}
			fx.tick(time.Second)
		}
		if n := seatedCount(); n < len(waiters) {
			t.Fatalf("starvation: only %d/%d waiters were ever seated", n, len(waiters))
		}
	})
}

// TTL: a silent, unseated record is forgotten; a member with a live head is not.
func TestTTLForgetsSilentRecordsKeepsWaiters(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "quiet", 100, time.Second)
		settle(fx.s, "quiet", 100, 100)
		// a second member seats, then leaves so it is a returning WAITER — its record
		// must survive any TTL while it has a live head
		fx.seatOne(t, "waiter", 100, time.Second)
		settle(fx.s, "waiter", 100, 100)
		m, g := fx.member(t, "waiter")
		fx.s.mu.Lock()
		fx.s.exitLocked(g, m, "test", fx.clock)
		fx.s.mu.Unlock()
		longAgo := fx.clock.Add(-time.Duration(10*defaultConfig().TTLSeconds) * time.Second)
		fx.clock = longAgo
		fx.pick(head("waiter", 10*time.Second, bytesFor(100)))
		fx.clock = longAgo.Add(time.Duration(3*defaultConfig().TTLSeconds) * time.Second)
		fx.s.Tick(fx.clock)
		if fx.s.Sessions() == 0 {
			t.Fatal("a waiter with a live head must never be forgotten (R2)")
		}
		if fx.s.isNewcomerLocked("waiter") {
			t.Fatal("a returning waiter keeps its non-newcomer status while it waits")
		}
	})
}

// C8: a member with a queued head is not idle — gate closure (no Picks) must not
// manufacture idle exits.
func TestWaitingMemberIsNotIdleEvicted(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 400, time.Second)
		settle(fx.s, "m", 400, 400)
		fx.s.mu.Lock()
		fx.g().members["m"].waitingSince = fx.clock.Add(-time.Minute)
		fx.s.mu.Unlock()
		fx.tick(5 * time.Minute) // no Picks at all: the gate is "closed"
		if !fx.s.Seated("m") {
			t.Fatal("a member with a queued turn must not be idle-evicted while no Picks run (C8)")
		}
	})
}

// C3: a provisional in-flight id (chosen at Pick, never reaching PreRequest) is
// reaped; a confirmed one survives until InflightReapSeconds.
func TestProvisionalInflightReaped(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		// one more head, picked but never dispatched => provisional (a NEW req id, or
		// the record would simply overwrite the confirmed one)
		h2 := head("m", time.Second, bytesFor(100))
		h2.ReqID = "r-m-2"
		fx.pick(h2)
		fx.s.mu.Lock()
		m := fx.g().members["m"]
		before := m.inflight
		fx.s.mu.Unlock()
		if before != 2 {
			t.Fatalf("expected one confirmed + one provisional in flight, got %d", before)
		}
		fx.tick(31 * time.Second)
		fx.s.mu.Lock()
		after := fx.g().members["m"].inflight
		fx.s.mu.Unlock()
		if after != 1 {
			t.Fatalf("the provisional id must be reaped, %d in flight remain", after)
		}
	})
}

// A leaked cold charge is re-derived away at the next decision point
// (derive-don't-accumulate).
func TestLeakedColdChargeVanishesOnDerive(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 100, time.Second)
		fx.s.mu.Lock()
		fx.g().coldInFlight += 999999
		fx.s.deriveLocked()
		got := fx.g().coldInFlight
		cap := fx.g().coldCapTokens(fx.s.cfg.GroupBudgetFraction, fx.s.blockSize)
		fx.s.mu.Unlock()
		if got > cap {
			t.Fatalf("the derived cold charge must replace the leak, got %d (cap %d)", got, cap)
		}
	})
}

// The SEAT/EXIT event log carries the fairness id and the engine (the join key the
// E4a instrumentation exists for).
func TestSeatExitEventsCarryFairnessIDAndEngine(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		h := head("m", time.Second, bytesFor(100))
		h.FairnessID = "thread-7"
		if i := fx.pick(h); i != 0 {
			t.Fatal("the member must seat")
		}
		dispatch(t, fx.s, "m", keys(100, 1), bytesFor(100))
		evs := fx.s.Events()
		var seat, exit *Event
		for i := range evs {
			switch evs[i].Kind {
			case "SEAT":
				seat = &evs[i]
			}
		}
		if seat == nil {
			t.Fatal("a SEAT event must be recorded")
		}
		if seat.Engine == "" || seat.Engine == "single" && len(fx.ids) > 1 {
			t.Fatalf("the SEAT event must carry the engine, got %q", seat.Engine)
		}
		if want := "fairness_id=thread-7"; !contains(seat.Detail, want) {
			t.Fatalf("the SEAT event must carry the wire identity (%s), got %q", want, seat.Detail)
		}
		fx.s.mu.Lock()
		m := fx.g().members["m"]
		fx.s.exitLocked(fx.g(), m, "test", fx.clock)
		fx.s.mu.Unlock()
		evs = fx.s.Events()
		for i := range evs {
			if evs[i].Kind == "EXIT" {
				exit = &evs[i]
			}
		}
		if exit == nil || !contains(exit.Detail, "fairness_id=thread-7") {
			t.Fatalf("the EXIT event must carry the fairness id, got %+v", exit)
		}
	})
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// BUG 2: the MID-QUANTUM victim order never takes a seat back from a member that has
// not served a turn in this residency while a member that HAS served one is available.
func TestClass3VictimPrefersAServedMember(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		// two residents, both with a queued head (so class 2 finds nothing), neither
		// eligible by the tenure rule
		fx.seatOne(t, "served", 200, 9*time.Second)
		settle(fx.s, "served", 200, 200) // served one turn
		fx.seatOne(t, "fresh", 200, 8*time.Second)
		for _, id := range []string{"served", "fresh"} {
			m, _ := fx.member(t, id)
			fx.s.mu.Lock()
			m.waitingSince = fx.clock.Add(-time.Second)
			m.lastActivity = fx.clock
			m.inflight = 0
			m.busyPeriods = 0
			fx.s.mu.Unlock()
		}
		var reason string
		var victim *member
		fx.s.mu.Lock()
		fx.s.queueViewAt = fx.clock // a trustworthy queue view (both have heads)
		reason, victim = fx.s.budgetVictimLocked(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if victim == nil {
			t.Fatal("a victim must exist")
		}
		if victim.id == "fresh" {
			t.Fatalf("BUG 2: a member that has served ZERO turns must not be taken before one that has (reason=%s)", reason)
		}
		if victim.id != "served" {
			t.Fatalf("expected the served member, got %s", victim.id)
		}
	})
}

// BUG 3 (the C8 race): a head that arrives just before a Tick protects its member, and
// a STALE queue view must never justify an idle eviction.
func TestStaleQueueViewNeverIdleEvicts(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 200, time.Second)
		settle(fx.s, "m", 200, 200)
		// (a) a head arrives, Pick sees it, then a Tick lands just after -> protected
		fx.pick(head("m", time.Second, bytesFor(200)))
		fx.tick(25 * time.Second) // silence past S, but the head is queued
		if !fx.s.Seated("m") {
			t.Fatal("BUG 3: a head present at the last sync must protect its member from an idle eviction")
		}
		// (b) no Pick has run for longer than S (the flow controller's gate is closed):
		// the queue view is stale, so silence is NOT evidence and nothing may be evicted
		fx.s.mu.Lock()
		m := fx.g().members["m"]
		m.waitingSince = time.Time{} // the flag went stale exactly as in the race
		m.lastActivity = fx.clock.Add(-time.Hour)
		fx.s.mu.Unlock()
		fx.tick(5 * time.Minute) // Tick re-syncs from the LAST snapshot (which has the head)
		if !fx.s.Seated("m") {
			t.Fatal("BUG 3: a stale queue view must never justify an idle eviction")
		}
	})
}

// BUG 4(a): a mid-flight exit reaches the exit HOOK (the metric series), not only the
// snapshot. BUG 4(b): a rotation victim is never a member with something in flight.
func TestExitingIsCountedAndRotationNeverPicksInflight(t *testing.T) {
	cfg := defaultConfig()
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		var hooked []string
		fx.s.onExit = func(engine, reason string, turns int) { hooked = append(hooked, reason) }
		// a member filling most of the budget, in flight
		fx.seatOne(t, "run", 800, 5*time.Second)
		// over budget by growth: the pressure module takes the mid-flight class
		fx.s.mu.Lock()
		m := fx.g().members["run"]
		m.waitingSince = time.Time{}
		m.lastActivity = fx.clock.Add(-time.Hour) // silent, so class 2 has a victim...
		fx.s.mu.Unlock()
		// ...but the member is in flight, so class 2 skips it: force class 4 by making
		// every member in flight and the engine over budget
		fx.dispatchBig(t, "run", 400)
		fx.s.mu.Lock()
		fx.s.enforceBudgetLocked(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		// (b) the rotation never returns an in-flight member
		fx.s.mu.Lock()
		_, rv := fx.s.rotationVictimLocked(fx.g(), fx.clock)
		fx.s.mu.Unlock()
		if rv != nil && rv.inflight > 0 {
			t.Fatal("BUG 4(b): a rotation victim must never have requests in flight")
		}
		// (a) every booked exit reason reaches the hook
		found := false
		for _, r := range hooked {
			if r == "exiting" || r == "budget-exiting" {
				found = true
			}
		}
		if !found {
			t.Fatalf("BUG 4(a): a mid-flight exit must reach the exit hook, got %v", hooked)
		}
	})
}

// dispatchBig grows a member's footprint with an extra request (the pressure path's
// trigger), through the public API.
func (fx *engineSet) dispatchBig(t *testing.T, id string, blocks int) {
	t.Helper()
	h := head(id, time.Second, bytesFor(blocks))
	h.ReqID = "r-" + id + "-grow"
	if i := fx.pick(h); i < 0 {
		t.Fatalf("the member's growth head must dispatch, got %d", i)
	}
	fx.s.Dispatched(id, h.ReqID, keys(blocks, seedFor(id)+97), bytesFor(blocks), fx.clock)
}
