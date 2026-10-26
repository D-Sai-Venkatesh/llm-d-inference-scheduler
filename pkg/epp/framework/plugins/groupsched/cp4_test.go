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

import (
	"testing"
	"time"
)

// ---- the four re-review fixes ----

// R1+R11: cold charges in tokens; one cold entry is always allowed, so a
// session bigger than the cap does not deadlock the queue.
func TestColdUnitsAndBigSessionEntry(t *testing.T) {
	// cap = 10 blocks worth of tokens (640); a 250k-token session is
	// ~3906 blocks — far bigger than the cap
	s, err := NewScheduler(Config{GroupBudgetFraction: 0.95, Rho: 4, IdleSeconds: 20, TTLSeconds: 30,
		ColdEntryCapTokens: 640, InFlightCap: 64, BlockSizeTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 100000); err != nil { // huge budget: the cold cap is the binding constraint
		t.Fatal(err)
	}
	now := time.Now()
	// a big newcomer seats when nothing cold is in flight (R11)
	big := bytesFor(20000) // ~20000 blocks = 1.28M tokens > cap
	if i := s.PickFromHeads([]Head{head("big1", time.Second, big)}, now); i != 0 {
		t.Fatal("a big newcomer must be seated when nothing cold is in flight (R11)")
	}
	dispatch(t, s, "big1", keys(20000, 1), big)
	// a second big one WAITS (cold in flight; the cap would be exceeded)
	h2 := []Head{head("big2", time.Second, big)}
	if i := s.PickFromHeads(h2, now); i != -1 {
		t.Fatal("the second big entry must wait while the first's prefill is cold (R1 units)")
	}
	// the first's prefill completes: the charge releases; the second seats
	s.Chunk("big1")
	if i := s.PickFromHeads(h2, now); i != 0 {
		t.Fatal("the second big entry must seat after the first's prefill completes")
	}
}

// R2: no TTL on waiters — a session waiting longer than TTL keeps its
// history (the cost estimate) and its non-newcomer status.
func TestWaiterKeepsHistoryPastTTL(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// seat and pay out "re" so it has history and everSeated
	s.PickFromHeads([]Head{head("re", time.Second, bytesFor(100))}, now)
	dispatch(t, s, "re", keys(100, 1), bytesFor(100))
	settle(s, "re", 100, 0)
	for turn := 0; turn < 3; turn++ {
		s.PickFromHeads([]Head{head("re", time.Second, bytesFor(100))}, now)
		dispatch(t, s, "re", keys(100, 1), bytesFor(100))
		settle(s, "re", 100, 150)
	}
	s.PickFromHeads([]Head{head("big", time.Second, bytesFor(900))}, now)
	dispatch(t, s, "big", keys(900, 2), bytesFor(900))
	settle(s, "big", 900, 0)
	// re waits (blocked by the budget) for LONGER than the TTL; ticks run
	later := now.Add(time.Duration(defaultConfig().TTLSeconds*float64(time.Second)) * 3)
	h := []Head{head("re", time.Second, bytesFor(100))}
	for i := 0; i < 5; i++ {
		s.PickFromHeads(h, later) // keeps the head present (a live waiter)
		s.Tick(later)
	}
	if s.Sessions() == 0 {
		t.Fatal("a waiter with a live head must never be forgotten (R2)")
	}
	// the history survives (the cost estimate stays small, not full-size)
	s.mu.Lock()
	est := s.estimateCostLocked("re", bytesFor(100))
	_, everSeated := s.everSeated["re"]
	s.mu.Unlock()
	if est > 100 {
		t.Fatalf("history must survive the long wait: estimate inflated to %d blocks", est)
	}
	if !everSeated {
		t.Fatal("non-newcomer status must survive the long wait (R2)")
	}
}

// R4: a fully warm entry (uncached 0) still records and later turns pay.
func TestWarmEntryPaysBack(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	s.PickFromHeads([]Head{head("w", time.Second, bytesFor(100))}, now)
	dispatch(t, s, "w", keys(100, 1), bytesFor(100))
	settle(s, "w", 100, 100) // FULLY warm entry: uncached = 0
	// repeat turns with cached hits: earned accumulates (the old code
	// re-detected the entry forever and earned stayed 0)
	for turn := 0; turn < 3; turn++ {
		s.PickFromHeads([]Head{head("w", time.Second, bytesFor(100))}, now)
		dispatch(t, s, "w", keys(100, 1), bytesFor(100))
		settle(s, "w", 100, 100) // earned += 100 each
	}
	s.mu.Lock()
	m := s.members["w"]
	earned, cost, recorded := m.earned, m.entryCost, m.entryRecorded
	s.mu.Unlock()
	if !recorded || cost != 0 {
		t.Fatalf("warm entry must record with cost 0: recorded=%v cost=%d", recorded, cost)
	}
	if earned != 300 {
		t.Fatalf("earned must accumulate past a warm entry: %d", earned)
	}
	// paid: earned 300 >= rho*max(entryCost,1) = 4 — the member is eligible
	// (prove it: a waiter needing the room rotates it out)
	s.PickFromHeads([]Head{head("x", time.Second, bytesFor(900))}, now)
	if snap := s.Snapshot(); snap.Exits["paid"] != 1 {
		t.Fatalf("the warm-entry member must be paid-eligible: exits=%v", snap.Exits)
	}
}

// R9: anonymous traffic is served when no member head is dispatchable.
func TestAnonymousPassThrough(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// a member mid-cap... simplest: only an anonymous head exists
	anon := Head{Session: "", ReqID: "anon-1", Arrival: now.Add(-time.Second), SizeBytes: 1000}
	if i := s.PickFromHeads([]Head{anon}, now); i != 0 {
		t.Fatalf("an anonymous head must be served when no member head is dispatchable, got %d", i)
	}
	// and it is never tracked
	if s.Sessions() != 0 {
		t.Fatal("anonymous traffic must not be tracked")
	}
	// a member's head outranks it
	s.PickFromHeads([]Head{head("m", 100*time.Millisecond, bytesFor(10))}, now) // seats m; its head dispatches
	anonLater := Head{Session: "", ReqID: "anon-2", Arrival: now, SizeBytes: 1000}
	member := head("m", 100*time.Millisecond, bytesFor(10))
	if i := s.PickFromHeads([]Head{anonLater, member}, now); i != 1 {
		t.Fatalf("the member's head must outrank the anonymous one: got %d", i)
	}
}

// ---- CP4: the mechanism suite ----

// Hold the line: a big head is not overtaken by a smaller one behind it.
func TestHoldTheLine(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// fill the group so little headroom remains (800 of 950)
	s.PickFromHeads([]Head{head("m1", 3*time.Second, bytesFor(400))}, now)
	dispatch(t, s, "m1", keys(400, 1), bytesFor(400))
	settle(s, "m1", 400, 0)
	s.PickFromHeads([]Head{head("m2", 2*time.Second, bytesFor(400))}, now)
	dispatch(t, s, "m2", keys(400, 2), bytesFor(400))
	settle(s, "m2", 400, 0)
	// the head of the line is big (needs 400; only 150 free). m1 and m2
	// have their OWN next turns queued (their heads are present), so the
	// engine-idle rule cannot fire and no member is eligible: the big
	// head is genuinely blocked — and NOBODY behind it jumps ahead.
	h := []Head{
		head("m1", 3*time.Second, bytesFor(400)),
		head("bighead", time.Second, bytesFor(400)),
		head("small", 100*time.Millisecond, bytesFor(100)),
		head("m2", 2*time.Second, bytesFor(400)),
	}
	i := s.PickFromHeads(h, now)
	if i != 0 && i != 3 {
		t.Fatalf("hold the line: only member heads (m1/m2) may dispatch, got head %d", i)
	}
	if i >= 0 { // the served member head is dispatched and settled (production shape)
		id := h[i].Session
		dispatch(t, s, id, keys(400, 1), bytesFor(400))
		settle(s, id, 400, 400)
	}
	// small (head 2) never dispatches while bighead (head 1) is blocked;
	// the member heads that DO dispatch are settled (production shape)
	for round := 0; round < 5; round++ {
		j := s.PickFromHeads(h, now)
		if j == 2 {
			t.Fatal("hold the line violated: small jumped the blocked big head")
		}
		if j >= 0 {
			id := h[j].Session
			dispatch(t, s, id, keys(400, 1), bytesFor(400))
			settle(s, id, 400, 400)
		}
	}
	// m1 pays out fully; the demand rotation frees room — the BIG head
	// (not small) is next through the door
	for turn := 0; turn < 6; turn++ {
		s.PickFromHeads(h, now)
		dispatch(t, s, "m1", keys(400, 1), bytesFor(400))
		settle(s, "m1", 400, 400) // earned += 400/turn
	}
	// after room frees, the line moves IN ORDER: the big head seats (and
	// small behind it, once it fits — hold-the-line blocks only while the
	// head is stuck, it does not serialize the drain)
	k := s.PickFromHeads(h, now)
	_ = k
	if !s.Seated("bighead") {
		t.Fatal("after room frees the big head must be seated (before or with small)")
	}
}

// No starvation: under steady load every waiter is seated within a
// bounded number of payback cycles.
func TestNoStarvation(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// three 300-block members fill the 950 budget; eight waiters queue.
	// Each cycle: every seated member gets one turn (fully warm: paid in
	// one repeat); the demand rotation seats the next waiter. All eight
	// must be seated within a small multiple of the group size.
	for _, id := range []string{"m0", "m1", "m2"} {
		s.PickFromHeads([]Head{head(id, time.Second, bytesFor(300))}, now)
		dispatch(t, s, id, keys(300, Hash(uint64(id[1]))), bytesFor(300))
		settle(s, id, 300, 300) // warm entry: entryCost 0
	}
	var waiters []string
	for i := 0; i < 8; i++ {
		waiters = append(waiters, string(rune('a'+i)))
	}
	// the metric is EVER-seated: a waiter that got its seat and later
	// rotated out (paid) has NOT been starved — starvation is never
	// getting in at all.
	gotSeat := map[string]bool{}
	seated := func() (n int) {
		for _, id := range waiters {
			if s.Seated(id) {
				gotSeat[id] = true
				n++
			} else if gotSeat[id] {
				n++ // seated before, rotated out paid
			}
		}
		return
	}
	// every pick sees the FULL head set (the production shape: the band
	// iterates every queue) — member next-turns and waiting newcomers
	// together; the served member is whichever head the pick returns.
	allHeads := func() []Head {
		hs := make([]Head, 0, len(waiters)+3)
		for _, id := range []string{"m0", "m1", "m2"} {
			if s.Seated(id) {
				hs = append(hs, head(id, 10*time.Millisecond, bytesFor(300)))
			}
		}
		for i, id := range waiters {
			if s.Seated(id) {
				hs = append(hs, head(id, 5*time.Millisecond, bytesFor(300)))
			} else {
				hs = append(hs, head(id, time.Duration(len(waiters)-i)*100*time.Millisecond, bytesFor(300)))
			}
		}
		return hs
	}
	cycles := 0
	for seated() < len(waiters) && cycles < 60 {
		cycles++
		// take up to (members + 1) served turns per cycle
		for round := 0; round < 5; round++ {
			hs := allHeads()
			i := s.PickFromHeads(hs, now)
			if i < 0 {
				break
			}
			id := hs[i].Session
			if id != "" && s.Seated(id) {
				dispatch(t, s, id, keys(300, Hash(uint64(id[len(id)-1]))), bytesFor(300))
				settle(s, id, 300, 300)
			}
		}
	}
	if n := seated(); n < len(waiters) {
		t.Fatalf("starvation: only %d/%d waiters seated after %d cycles", n, len(waiters), cycles)
	}
}

// No eviction while in flight: a member with a running request is marked
// exiting at most, never removed, until the drain completes.
func TestNoEvictionWhileInFlight(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	s.PickFromHeads([]Head{head("m", 2*time.Second, bytesFor(400))}, now)
	dispatch(t, s, "m", keys(400, 1), bytesFor(400))
	s.PickFromHeads([]Head{head("m2", time.Second, bytesFor(400))}, now)
	dispatch(t, s, "m2", keys(400, 2), bytesFor(400))
	// m's FIRST turn is still running (inflight=1, unsettled) when its
	// SECOND head (grown to 700) is served: m's inflight becomes 2 and
	// the budget overflows (400+700 > 950 with m2's 400... 1100+)
	i := s.PickFromHeads([]Head{head("m", time.Millisecond, bytesFor(700))}, now)
	if i != 0 {
		t.Fatalf("the member's second head must be served (in-flight cap): got %d", i)
	}
	dispatch(t, s, "m", keys(700, 1), bytesFor(700))
	s.Tick(now)
	if !s.Seated("m") {
		t.Fatal("a member in flight must never be evicted (marked exiting at most)")
	}
	// the drain completes: m leaves when its running turns settle
	settle(s, "m", 400, 0)
	settle(s, "m", 700, 0)
	if s.Seated("m") {
		t.Fatal("after the drain the exiting member must leave")
	}
	if !s.Seated("m2") {
		t.Fatal("m2 (never marked exiting) must survive the overflow and the drain")
	}
}
