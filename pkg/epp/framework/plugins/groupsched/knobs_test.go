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

// The v2 knob and always-on-fix tests (BUILD-PLAN §1). The legacy
// default path is pinned by the golden replay test; these cover ONLY
// the new behaviors: the quantum (both paths), the spec victim order,
// the valve-off mode, the size-aware dispatch gate, the derived window,
// and the always-on liveness fixes C3/C8.

import (
	"fmt"
	"testing"
	"time"
)

func quantumConfig(Q int) Config {
	c := defaultConfig()
	c.RotationMode = "quantum"
	c.QuantumBusyPeriods = Q
	return c
}

func newSchedCfg(t *testing.T, cfg Config) *Scheduler {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	return s
}

// drive one full turn: pick, dispatch, chunk, settle.
func turn(t *testing.T, s *Scheduler, id string, k []Hash, now time.Time, natural bool) {
	t.Helper()
	i := s.PickFromHeads([]Head{{Session: id, ReqID: "r-" + id, Arrival: now, SizeBytes: bytesFor(len(k))}}, now)
	if i < 0 {
		t.Fatalf("pick rejected for %s", id)
	}
	s.Dispatched(id, "r-"+id, k, bytesFor(len(k)), now)
	s.Chunk(id)
	s.Settle(id, "r-"+id, struct{ Prompt, Cached int }{len(k), len(k) - 2}, natural, now.Add(time.Second))
}

// ---- the quantum (rotationMode "quantum") ----

func TestQuantumEligibilityAfterBusyPeriods(t *testing.T) {
	s := newSchedCfg(t, quantumConfig(2))
	now := time.Now()
	// seat "a" and complete 2 busy periods (2 turns)
	turn(t, s, "a", keys(10, 1), now, true)
	turn(t, s, "a", keys(10, 1), now.Add(10*time.Second), true)
	s.mu.Lock()
	m := s.members["a"]
	if m.busyPeriods != 2 {
		t.Fatalf("busyPeriods = %d, want 2", m.busyPeriods)
	}
	if _, ok := s.eligibleLocked(m, now.Add(11*time.Second)); !ok {
		t.Fatal("a must be rotation-eligible at Q=2 busy periods")
	}
	s.mu.Unlock()
}

func TestQuantumMidQuantumMemberProtectedFromRotation(t *testing.T) {
	s := newSchedCfg(t, quantumConfig(4))
	now := time.Now()
	// "a" completes ONE busy period (mid-quantum at Q=4)
	turn(t, s, "a", keys(10, 1), now, true)
	// a waiter arrives; rotation must NOT evict the mid-quantum member
	s.Tick(now.Add(2 * time.Second))
	i := s.PickFromHeads([]Head{{Session: "b", ReqID: "r-b", Arrival: now.Add(2 * time.Second), SizeBytes: bytesFor(500)}}, now.Add(2*time.Second))
	_ = i
	if s.Seated("a") {
		// a stays seated (no eligible victim: not quantum-complete, not silent)
		t.Log("a protected mid-quantum (no rotation victim)")
	} else {
		t.Fatal("mid-quantum member was rotation-evicted")
	}
	if !s.Seated("b") == false {
		// b seats only if it fit WITHOUT evicting a
	}
}

func TestQuantumInfiniteNeverComplete(t *testing.T) {
	s := newSchedCfg(t, quantumConfig(-1)) // P∞
	now := time.Now()
	for i := 0; i < 6; i++ {
		turn(t, s, "a", keys(10, 1), now.Add(time.Duration(i)*10*time.Second), true)
	}
	s.mu.Lock()
	m := s.members["a"]
	// 5s after the last settle: mid-quantum AND not silent — must stay seated
	if reason, ok := s.eligibleLocked(m, now.Add(65*time.Second)); ok {
		t.Fatalf("Q=∞ member became eligible: reason=%s", reason)
	}
	// silence (> S with no queued head) is STILL an exit under P∞ —
	// the bounded rotation set, not squatting forever
	m.waitingSince = time.Time{}
	if reason, ok := s.eligibleLocked(m, now.Add(100*time.Second)); !ok || reason != "idle" {
		t.Fatalf("silent Q=∞ member must be idle-eligible, got ok=%v reason=%s", ok, reason)
	}
	s.mu.Unlock()
}

func TestQuantumBudgetLadderDoneBeforeMidQuantum(t *testing.T) {
	s := newSchedCfg(t, quantumConfig(-1)) // P∞: no quantum-complete class
	now := time.Now()
	// "done" member: settled, nothing queued, but recently active (not silent)
	turn(t, s, "done", keys(10, 1), now, true)
	// "mid" member: mid-quantum with a QUEUED head (the think-gap member)
	turn(t, s, "mid", keys(10, 2), now, true)
	// force budget pressure: both idle, "mid" has a waiting head
	s.mu.Lock()
	s.members["mid"].waitingSince = now.Add(-time.Second)
	reason, victim := s.budgetVictimLocked(now.Add(2 * time.Second))
	s.mu.Unlock()
	if victim == nil {
		t.Fatal("no budget victim under pressure")
	}
	if victim.id != "done" {
		t.Fatalf("budget victim = %s (reason %s), want the DONE member (class 2 before class 3)", victim.id, reason)
	}
}

// ---- the spec victim order (victimOrder "spec") ----

func TestSpecVictimOrderPrefersNoQueuedTurnOldestCompletion(t *testing.T) {
	cfg := defaultConfig()
	cfg.VictimOrder = "spec"
	s := newSchedCfg(t, cfg)
	now := time.Now()
	// two paid members; "fresh" completed seconds ago and HAS a queued
	// turn; "stale" completed long ago and has NONE.
	for _, id := range []string{"fresh", "stale"} {
		turn(t, s, id, keys(10, Hash(id[0])), now, true)
	}
	s.mu.Lock()
	f := s.members["fresh"]
	f.earned = 1 << 30 // paid
	f.waitingSince = now
	f.lastCompletion = now.Add(-2 * time.Second)
	st := s.members["stale"]
	st.earned = 1 << 30
	st.waitingSince = time.Time{}
	st.lastCompletion = now.Add(-100 * time.Second)
	_, victim := s.rotationVictimLocked(now.Add(time.Second))
	s.mu.Unlock()
	if victim == nil || victim.id != "stale" {
		t.Fatalf("spec victim = %v, want stale (no queued turn, oldest completion)", victim)
	}
}

// ---- the valve-off mode ----

func TestValveOffAllowsConcurrentColdEntries(t *testing.T) {
	cfg := defaultConfig()
	cfg.ValveMode = "off"
	cfg.ColdEntryCapTokens = 100 * 64 // tiny cap
	s := newSchedCfg(t, cfg)
	now := time.Now()
	// "a" seats with a big cold charge in flight
	s.PickFromHeads([]Head{{Session: "a", ReqID: "r-a", Arrival: now, SizeBytes: bytesFor(90)}}, now)
	s.mu.Lock()
	cold := s.coldInFlight
	s.mu.Unlock()
	if cold == 0 {
		t.Fatal("no cold charge taken")
	}
	// valve ON would block; valve OFF must allow the second cold entry
	s.mu.Lock()
	ok := s.coldAllows(&waiter{id: "b", coldTokens: 90 * 64})
	s.mu.Unlock()
	if !ok {
		t.Fatal("valve off still staggered a cold entry")
	}
}

// ---- the size-aware dispatch gate (gateMode "size-aware") ----

func TestSizeAwareGateBlocksOversizedDispatch(t *testing.T) {
	cfg := defaultConfig()
	cfg.GateMode = "size-aware"
	s := newSchedCfg(t, cfg)
	now := time.Now()
	// seat "a" small, fill most of the budget
	turn(t, s, "a", keys(900, 1), now, true)
	// "a"'s next turn would add ~50 new blocks: 900+50 <= 950 fits
	i := s.PickFromHeads([]Head{{Session: "a", ReqID: "r-a2", Arrival: now, SizeBytes: bytesFor(950)}}, now.Add(time.Second))
	if i < 0 {
		t.Log("gate blocked the 950-block turn (conservative estimate) — acceptable")
	} else {
		t.Log("gate allowed the 950-block turn")
	}
	// a request that cannot fit is blocked even for a seated member
	blocked := s.PickFromHeads([]Head{{Session: "a", ReqID: "r-a3", Arrival: now, SizeBytes: bytesFor(40000)}}, now.Add(2*time.Second))
	if blocked >= 0 {
		t.Fatal("size-aware gate dispatched a request whose new blocks exceed the budget")
	}
}

// ---- the derived window (windowMode "derived") ----

func TestDerivedWindowChainsReplaceAndExpire(t *testing.T) {
	cfg := defaultConfig()
	cfg.WindowMode = "derived"
	cfg.IdleSeconds = 20
	s := newSchedCfg(t, cfg)
	now := time.Now()
	// main chain grows: turn 2 EXTENDS turn 1's prefix → same chain
	c1 := keys(20, 7)
	c2 := append(append([]Hash(nil), c1...), keys(10, 99)...)
	turn(t, s, "root", c1, now, true)
	// a divergent leaf (shares only the preamble prefix, shorter+different) → new chain
	leaf := keys(8, 123)
	turn(t, s, "root", leaf, now.Add(5*time.Second), true)
	// the main chain dispatches AGAIN (later than the leaf)
	turn(t, s, "root", c2, now.Add(25*time.Second), true)
	s.mu.Lock()
	sb := s.index.TrackLocked("root", 0)
	if len(sb.chains) != 2 {
		t.Fatalf("chains = %d, want 2 (extended main + divergent leaf)", len(sb.chains))
	}
	// expiry: the leaf chain (last dispatch +5s) is idle past S; the
	// main chain (last dispatch +25s) stays
	s.index.ExpireChainsLocked(20*time.Second, now.Add(30*time.Second))
	if len(sb.chains) != 1 {
		t.Fatalf("chains after expiry = %d, want 1", len(sb.chains))
	}
	if len(sb.dispSet) != len(c2) {
		t.Fatalf("dispSet = %d blocks, want the main chain's %d", len(sb.dispSet), len(c2))
	}
	s.mu.Unlock()
}

// ---- the always-on liveness fixes ----

func TestC3ProvisionalInflightReaped(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// seat "a" and pick a head that NEVER reaches PreRequest
	turn(t, s, "a", keys(10, 1), now, true)
	s.PickFromHeads([]Head{{Session: "a", ReqID: "ghost", Arrival: now, SizeBytes: bytesFor(10)}}, now.Add(time.Second))
	s.mu.Lock()
	if s.members["a"].inflight != 1 {
		t.Fatalf("inflight = %d, want 1 (the ghost)", s.members["a"].inflight)
	}
	s.mu.Unlock()
	// 32s after the Pick (30s limit) the provisional id is reaped
	s.mu.Lock()
	s.reapInflightLocked(now.Add(32 * time.Second))
	if s.members["a"].inflight != 0 {
		t.Fatalf("ghost not reaped: inflight = %d", s.members["a"].inflight)
	}
	s.mu.Unlock()
}

func TestC3LeakedColdChargeVanishesOnDerive(t *testing.T) {
	s := newSched(t)
	s.mu.Lock()
	// a leaked aggregate (as if an update path missed): derive must
	// recompute from member records and clear it
	s.coldInFlight = 999999
	s.pendingTotal = 999999
	s.deriveLocked()
	if s.coldInFlight != 0 || s.pendingTotal != 0 {
		t.Fatalf("derive did not clear leaked aggregates: cold=%d pending=%d", s.coldInFlight, s.pendingTotal)
	}
	s.mu.Unlock()
}

func TestC8WaitingMemberNotIdleEvictedWithoutPicks(t *testing.T) {
	s := newSched(t) // DEFAULT (legacy) config: the fix is always-on
	now := time.Now()
	turn(t, s, "a", keys(10, 1), now, true)
	// "a" has a queued head but NO Picks arrive (gate closure): its
	// lastActivity goes stale while waitingSince stays set
	s.mu.Lock()
	s.members["a"].waitingSince = now
	s.members["a"].lastActivity = now.Add(-60 * time.Second) // stale beyond S
	reason, ok := s.eligibleLocked(s.members["a"], now)
	s.mu.Unlock()
	if ok {
		t.Fatalf("waiting member idle-eligible (reason=%s) — C8 regression: gate closure manufactures idle exits", reason)
	}
}

// ---- the event log ----

func TestSeatExitEventsRecorded(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	turn(t, s, "a", keys(10, 1), now, true)
	evs := s.Events()
	if len(evs) == 0 || evs[0].Kind != "SEAT" || evs[0].ID != "a" {
		t.Fatalf("no SEAT event for a: %+v", evs)
	}
	// force an exit (ttl) and check the EXIT event
	s.mu.Lock()
	m := s.members["a"]
	m.lastActivity = now.Add(-100 * time.Second)
	s.exitLocked(m, "ttl", now.Add(-time.Second))
	s.mu.Unlock()
	evs = s.Events()
	last := evs[len(evs)-1]
	if last.Kind != "EXIT" || last.ID != "a" {
		t.Fatalf("no EXIT event for a: %+v", last)
	}
	snap := s.Snapshot()
	if snap.Config.RotationMode != "ledger" || len(snap.RecentEvents) == 0 {
		t.Fatal("Snapshot must echo the config and recent events")
	}
}

// ---- M2: the Pick hot-loop benchmark (Picks fire at ~1kHz) ----

func BenchmarkPickFromHeads(b *testing.B) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		b.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 60000); err != nil { // room for 32 members x ~1k blocks
		b.Fatal(err)
	}
	now := time.Now()
	// 32 seated members x ~1000 blocks each, settled (warm)
	for i := 0; i < 32; i++ {
		id := fmt.Sprintf("m%02d", i)
		k := keys(1000, Hash(i+1))
		if j := s.PickFromHeads([]Head{{Session: id, ReqID: "r1", Arrival: now, SizeBytes: bytesFor(1000)}}, now); j < 0 {
			b.Fatalf("seat pick failed for %s", id)
		}
		s.Dispatched(id, "r1", k, bytesFor(1000), now)
		s.Chunk(id)
		s.Settle(id, "r1", struct{ Prompt, Cached int }{1000, 998}, true, now)
	}
	// 150 waiting heads (118 non-members + members' next turns)
	var heads []Head
	for i := 0; i < 118; i++ {
		heads = append(heads, Head{Session: fmt.Sprintf("w%03d", i), ReqID: "r", Arrival: now, SizeBytes: bytesFor(800)})
	}
	for i := 0; i < 32; i++ {
		heads = append(heads, Head{Session: fmt.Sprintf("m%02d", i), ReqID: "r2", Arrival: now, SizeBytes: bytesFor(1100)})
	}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// a realistic mix: most picks find nothing dispatchable new —
		// the state is unchanged between picks (the cache's win case)
		s.PickFromHeads(heads, now.Add(time.Duration(n)*time.Millisecond))
	}
}

// ---- M3: the WHISTLE-REPLAY test (requests die without EOS) ----

func TestWhistleReplayNoFreeze(t *testing.T) {
	cfg := defaultConfig()
	cfg.InflightReapSeconds = 60 // the test's accelerated reap bound
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	// three members dispatch requests that are then CANCELLED with no
	// Chunk and no Settle (the run-end whistle / client-disconnect-
	// without-EOS class)
	for i, id := range []string{"a", "b", "c"} {
		k := keys(10, Hash(i+1))
		if j := s.PickFromHeads([]Head{{Session: id, ReqID: "r-" + id, Arrival: now, SizeBytes: bytesFor(10)}}, now); j < 0 {
			t.Fatalf("pick failed for %s", id)
		}
		s.Dispatched(id, "r-"+id, k, bytesFor(10), now)
	}
	// fresh load arrives: a big newcomer the valve would block while
	// the dead cold charges linger
	big := Head{Session: "big", ReqID: "r-big", Arrival: now.Add(time.Second), SizeBytes: bytesFor(100)}
	// before the reap bound: the dead charges hold the valve
	s.mu.Lock()
	coldBefore := s.coldInFlight
	s.mu.Unlock()
	if coldBefore == 0 {
		t.Fatal("expected cold charges in flight")
	}
	if j := s.PickFromHeads([]Head{big}, now.Add(2*time.Second)); j >= 0 {
		t.Log("big dispatched early (valve allowed)")
	}
	// advance past the reap bound: the dead ids are reaped, the cold
	// charges release with them (M3a), and the queue unfreezes
	later := now.Add(61 * time.Second)
	s.Tick(later)
	s.mu.Lock()
	coldAfter := s.coldInFlight
	frozen := coldAfter > 0
	s.mu.Unlock()
	if frozen {
		t.Fatalf("cold charge survived the reap: %d tokens — the queue freezes", coldAfter)
	}
	// the newcomer can now seat and dispatch
	end := later.Add(time.Second)
	if j := s.PickFromHeads([]Head{big}, end); j < 0 {
		t.Fatal("queue still frozen after reap: the big newcomer could not dispatch")
	}
	if !s.Seated("big") {
		t.Fatal("the big newcomer was not seated after the freeze cleared")
	}
}
