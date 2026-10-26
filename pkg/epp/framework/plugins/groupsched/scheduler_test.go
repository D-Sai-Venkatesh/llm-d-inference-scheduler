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

// bytesFor converts blocks to the wire size that estimates them at 4
// bytes/token (the estimate path's convention).
func bytesFor(blocks int) int64 { return int64(blocks) * 4 * 64 }

func newSched(t *testing.T) *Scheduler {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil { // budget 950 blocks
		t.Fatal(err)
	}
	return s
}

func head(session string, ago time.Duration, sizeBytes int64) Head {
	return Head{Session: session, ReqID: "r-" + session, Arrival: time.Now().Add(-ago), SizeBytes: sizeBytes}
}

// dispatch drives the released-request path: Pick chose the head, the
// producer tokenized, PreRequest corrects with the exact keys.
func dispatch(t *testing.T, s *Scheduler, session string, k []Hash, size int64) {
	t.Helper()
	s.Dispatched(session, "r-"+session, k, size, time.Now())
}

func settle(s *Scheduler, session string, prompt, cached int) {
	s.Chunk(session)
	s.Settle(session, struct{ Prompt, Cached int }{prompt, cached}, true, time.Now())
}

func TestMemberServedImmediately(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// one queued head; rotation seats the newcomer; Pick serves it
	i := s.PickFromHeads([]Head{head("a", time.Second, bytesFor(10))}, now)
	if i != 0 {
		t.Fatalf("pick: %d", i)
	}
	dispatch(t, s, "a", keys(10, 1), bytesFor(10))
	if !s.Seated("a") {
		t.Fatal("a must be seated")
	}
}

func TestNonMembersNeverRunUnseated(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// two 600-block sessions: only one fits the 950 budget
	h := []Head{head("a", 2*time.Second, bytesFor(600)), head("b", time.Second, bytesFor(600))}
	i := s.PickFromHeads(h, now)
	if i != 0 {
		t.Fatalf("the older newcomer a must be served first, got %d", i)
	}
	dispatch(t, s, "a", keys(600, 1), bytesFor(600))
	settle(s, "a", 600, 0)
	// a is quiet (nothing waiting, nothing in flight); b is blocked: the
	// engine-idle rule seats b — b dispatches SEATED, never unseated (D10)
	j := s.PickFromHeads(h[1:], now)
	if j != 0 {
		t.Fatalf("b must dispatch via the engine-idle rotation, got %d", j)
	}
	if !s.Seated("b") {
		t.Fatal("b must be the seated member")
	}
}

func TestPaybackRotation(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// member a: entry cost 100 uncached, needs earned >= 400 to leave
	i := s.PickFromHeads([]Head{head("a", time.Second, bytesFor(100))}, now)
	if i != 0 {
		t.Fatal("seat a")
	}
	dispatch(t, s, "a", keys(100, 1), bytesFor(100))
	settle(s, "a", 100, 0)
	for turn := 0; turn < 3; turn++ {
		j := s.PickFromHeads([]Head{head("a", time.Second, bytesFor(100))}, now)
		if j != 0 {
			t.Fatal("pick turn", turn)
		}
		dispatch(t, s, "a", keys(100, 1), bytesFor(100))
		settle(s, "a", 100, 150) // 450 earned >= 4 x 100
	}
	if snap := s.Snapshot(); snap.Exits["paid"] != 0 {
		t.Fatalf("a left before a waiter needed the room (rotation must be demand-driven): %v", snap.Exits)
	}
	// b arrives needing the room: a (paid, nothing waiting) leaves; b seats
	k := s.PickFromHeads([]Head{head("b", time.Second, bytesFor(900))}, now)
	if k != 0 {
		t.Fatal("b must dispatch after the paid rotation")
	}
	snap := s.Snapshot()
	if snap.Exits["paid"] != 1 || !s.Seated("b") {
		t.Fatalf("paid exit + b seated: exits=%v seated(b)=%v", snap.Exits, s.Seated("b"))
	}
}

func TestNewcomerFirst(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// seat and pay out "re" (an ex-member), then a fresh newcomer competes
	i := s.PickFromHeads([]Head{head("re", time.Second, bytesFor(100))}, now)
	if i != 0 {
		t.Fatal("seat re")
	}
	dispatch(t, s, "re", keys(100, 1), bytesFor(100))
	settle(s, "re", 100, 0)
	for turn := 0; turn < 3; turn++ {
		s.PickFromHeads([]Head{head("re", time.Second, bytesFor(100))}, now)
		dispatch(t, s, "re", keys(100, 1), bytesFor(100))
		settle(s, "re", 100, 150)
	}
	// a big newcomer rotates the paid re out
	j := s.PickFromHeads([]Head{head("fresh", time.Second, bytesFor(900))}, now)
	if j != 0 {
		t.Fatal("fresh seats")
	}
	dispatch(t, s, "fresh", keys(900, 2), bytesFor(900))
	settle(s, "fresh", 900, 0)
	// re re-arrives (waiter, NOT newcomer); a second newcomer arrives after
	h := []Head{head("re", 2*time.Second, bytesFor(100)), head("fresh2", time.Second, bytesFor(900))}
	// budget seats one: the NEWCOMER (fresh2) goes before the ex-member
	k := s.PickFromHeads(h, now.Add(time.Second))
	if k != 1 {
		t.Fatalf("newcomer must be seated before the ex-member waiter, got head %d", k)
	}
}

func TestTTLSilenceForgets(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	s.PickFromHeads([]Head{head("a", time.Second, bytesFor(10))}, now)
	dispatch(t, s, "a", keys(10, 1), bytesFor(10))
	settle(s, "a", 10, 0)
	if n := s.Sessions(); n != 1 {
		t.Fatalf("tracked sessions: %d", n)
	}
	later := now.Add(time.Duration(defaultConfig().TTLSeconds*float64(time.Second)) + time.Second)
	s.Tick(later)
	if n := s.Sessions(); n != 0 {
		t.Fatalf("record leaked after TTL: %d sessions (CP1 condition)", n)
	}
	if s.Seated("a") {
		t.Fatal("a must be unseated")
	}
}

func TestGrowthReEnforcement(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// two 400-block members (800 <= 950)
	s.PickFromHeads([]Head{head("a", 2*time.Second, bytesFor(400))}, now)
	dispatch(t, s, "a", keys(400, 1), bytesFor(400))
	settle(s, "a", 400, 0)
	s.PickFromHeads([]Head{head("b", time.Second, bytesFor(400))}, now)
	dispatch(t, s, "b", keys(400, 2), bytesFor(400))
	settle(s, "b", 400, 0)
	if snap := s.Snapshot(); snap.Seats != 2 {
		t.Fatalf("seats: %d", snap.Seats)
	}
	// both grow to 700 (1400 > 950): mid-flight members are marked
	// exiting (§6.3); after the drain (settle), the next enforcement
	// removes one for real
	s.PickFromHeads([]Head{head("a", time.Second, bytesFor(700))}, now)
	dispatch(t, s, "a", keys(700, 1), bytesFor(700))
	s.PickFromHeads([]Head{head("b", time.Second, bytesFor(700))}, now)
	dispatch(t, s, "b", keys(700, 2), bytesFor(700))
	s.Tick(now) // the overflow marks the earliest-seated member exiting
	snap := s.Snapshot()
	if snap.NEBlocks > snap.BudgetBlocks {
		t.Fatalf("budget violated after growth: %d > %d", snap.NEBlocks, snap.BudgetBlocks)
	}
	// drain both (their turns complete); the next enforcement evicts
	settle(s, "a", 700, 0)
	settle(s, "b", 700, 0)
	s.Tick(now)
	snap = s.Snapshot()
	if snap.Seats > 1 {
		t.Fatalf("after the drain one member must leave: %d seats (exits %v)", snap.Seats, snap.Exits)
	}
}

func TestColdCapReleasedAtChunkNotDecodeEnd(t *testing.T) {
	// B3 + S0-2: the stagger releases at prefill complete (the first
	// chunk), not at end of decode; and non-natural entry turns release too.
	s, err := NewScheduler(Config{GroupBudgetFraction: 0.95, Rho: 4, IdleSeconds: 20, TTLSeconds: 30,
		ColdEntryCapTokens: bytesFor(10), InFlightCap: 64, BlockSizeTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	if i := s.PickFromHeads([]Head{head("a", time.Second, bytesFor(10))}, now); i != 0 {
		t.Fatal("seat a")
	}
	dispatch(t, s, "a", keys(10, 1), bytesFor(10))
	// decode is still running (no settle); the FIRST CHUNK releases the cap
	s.Chunk("a")
	s.mu.Lock()
	cold := s.coldInFlight
	s.mu.Unlock()
	if cold != 0 {
		t.Fatalf("cold cap must be released at the first chunk, still %d", cold)
	}
	// a second entry can now use the full cap (B6: re-entries are capped too)
	if i := s.PickFromHeads([]Head{head("b", time.Second, bytesFor(10))}, now); i != 0 {
		t.Fatal("b seats after the release")
	}
	// non-natural termination releases as well (settle without Chunk)
	s.Settle("b", struct{ Prompt, Cached int }{}, false, now)
	s.mu.Lock()
	cold = s.coldInFlight
	s.mu.Unlock()
	if cold != 0 {
		t.Fatalf("non-natural termination must release the charge, still %d", cold)
	}
}

func TestEveryTurnCountsTowardNMax(t *testing.T) {
	// A8: a session whose turns keep failing (non-natural) still reaches
	// n_max and becomes eligible to leave.
	s := newSched(t)
	now := time.Now()
	s.PickFromHeads([]Head{head("a", time.Second, bytesFor(10))}, now)
	dispatch(t, s, "a", keys(10, 1), bytesFor(10))
	settle(s, "a", 10, 0) // natural entry turn, resTurns=1
	for turn := 0; turn < 7; turn++ {
		s.PickFromHeads([]Head{head("a", time.Second, bytesFor(10))}, now)
		dispatch(t, s, "a", keys(10, 1), bytesFor(10))
		// NON-natural terminations (errors): no payback ledger, but turns count
		s.Settle("a", struct{ Prompt, Cached int }{}, false, now)
	}
	// resTurns = 8 = n_max: eligible even though earned == 0
	s.mu.Lock()
	m := s.members["a"]
	eligible := m != nil && m.resTurns >= s.cfg.nMax()
	s.mu.Unlock()
	if !eligible {
		t.Fatal("n_max must count every settled turn, natural or not (A8)")
	}
}

func TestPreCapacityFCFS(t *testing.T) {
	// A5: before num_gpu_blocks arrives, FCFS pass-through (oldest head).
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	h := []Head{head("x", 2*time.Second, 100), head("y", time.Second, 100)}
	if i := FCFSPick(h); i != 0 {
		t.Fatalf("FCFS must pick the oldest, got %d", i)
	}
	if s.HasCapacity() {
		t.Fatal("no capacity yet")
	}
}
