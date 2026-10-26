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

// keys builds a deterministic chained-key set of n blocks.
func keys(n int, seed Hash) []Hash {
	out := make([]Hash, 0, n)
	prev := seed
	for i := 0; i < n; i++ {
		prev = prev*0x9E3779B97F4A7C15 ^ Hash(i+1) ^ 0xC2B2AE3D27D4EB4F
		out = append(out, prev)
	}
	return out
}

func newSched(t *testing.T) *Scheduler {
	s := NewScheduler(defaultConfig())
	if err := s.SetCapacity(64, 1000); err != nil { // 1000 blocks; budget 950
		t.Fatal(err)
	}
	return s
}

func TestSmokeMemberServedImmediately(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	s.Arrive("a", "r1", keys(10, 1), now)
	// rotation seats the newcomer (fits)
	s.Tick(now)
	if snap := s.Snapshot(); snap.Seats != 1 {
		t.Fatalf("seats after tick: %d", snap.Seats)
	}
	// the member's request dispatches
	reqID, sess, ok := s.Pick(now)
	if !ok || reqID != "r1" || sess != "a" {
		t.Fatalf("pick: %v %v %v", reqID, sess, ok)
	}
}

func TestSmokeNonMembersNeverRunUnseated(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// two sessions, 600 blocks each: only one fits the 950 budget
	s.Arrive("a", "ra", keys(600, 1), now)
	s.Arrive("b", "rb", keys(600, 2), now)
	s.Tick(now)
	snap := s.Snapshot()
	if snap.Seats != 1 {
		t.Fatalf("seats: %d (budget 950, two 600s cannot both fit)", snap.Seats)
	}
	// pick serves only the member
	_, _, ok := s.Pick(now)
	if !ok {
		t.Fatal("the member's request must dispatch")
	}
	// settle it; a goes quiet (nothing waiting, nothing in flight) while
	// b is blocked: the engine-idle rule rotates b in — b dispatches
	// SEATED (a member), never unseated (D10)
	s.Settle("a", "ra", keys(600, 1), struct{ Prompt, Cached int }{600, 0}, true, now)
	reqID, sess, ok := s.Pick(now)
	if !ok {
		t.Fatal("the engine-idle rule must seat b and dispatch it")
	}
	if sess != "b" || reqID != "rb" {
		t.Fatalf("dispatch went to %v/%v, want b/rb", sess, reqID)
	}
	if snap := s.Snapshot(); snap.Seats != 1 {
		t.Fatalf("seats: %d (b must be the seated member)", snap.Seats)
	}
}

func TestSmokePaybackRotation(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// member a: entry cost 100 uncached; needs earned >= 4*100 = 400 to leave
	s.Arrive("a", "r1", keys(100, 1), now)
	s.Tick(now)
	if _, _, ok := s.Pick(now); !ok {
		t.Fatal("seat+pick a")
	}
	s.Settle("a", "r1", keys(100, 1), struct{ Prompt, Cached int }{100, 0}, true, now)
	// three more turns earning 150 cached each (450 >= 400): paid back
	for i := 2; i <= 4; i++ {
		s.Arrive("a", "r"+string(rune('0'+i)), keys(100, 1), now)
		if _, _, ok := s.Pick(now); !ok {
			t.Fatal("pick turn", i)
		}
		s.Settle("a", "r"+string(rune('0'+i)), keys(100, 1), struct{ Prompt, Cached int }{100, 150}, true, now)
	}
	if snap := s.Snapshot(); snap.Exits["paid"] != 0 {
		t.Fatalf("a left before a waiter needed the room (rotation must be demand-driven)")
	}
	// now b arrives and needs the room: a (paid, nothing waiting, nothing
	// in flight) leaves; b seats
	s.Arrive("b", "rb", keys(900, 2), now)
	s.Tick(now)
	snap := s.Snapshot()
	if snap.Seats != 1 {
		t.Fatalf("seats after rotation: %d", snap.Seats)
	}
	if snap.Exits["paid"] != 1 {
		t.Fatalf("paid exits: %v", snap.Exits)
	}
	// b's request dispatches (a's would not: a is now a waiter)
	if _, _, ok := s.Pick(now); !ok {
		t.Fatal("b must dispatch")
	}
}

func TestSmokeNewcomerFirst(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// "re" is a RE-ENTERING session: seated once, paid out, exited with a
	// new waiting request (an ex-member waiter, not a newcomer)
	s.Arrive("re", "r1", keys(100, 1), now)
	s.Tick(now)
	if _, _, ok := s.Pick(now); !ok {
		t.Fatal("seat re")
	}
	s.Settle("re", "r1", keys(100, 1), struct{ Prompt, Cached int }{100, 0}, true, now)
	for i := 2; i <= 4; i++ {
		s.Arrive("re", "r"+string(rune('0'+i)), keys(100, 1), now)
		if _, _, ok := s.Pick(now); !ok {
			t.Fatal("pick turn", i)
		}
		s.Settle("re", "r"+string(rune('0'+i)), keys(100, 1), struct{ Prompt, Cached int }{100, 150}, true, now)
	}
	// re paid (450 >= 400); a big newcomer arrives needing the room -> re exits
	s.Arrive("fresh", "rf", keys(900, 2), now)
	s.Tick(now)
	if snap := s.Snapshot(); snap.Exits["paid"] != 1 {
		t.Fatalf("re must have been rotated out: %v", snap.Exits)
	}
	// settle fresh's entry request so its state is clean for round 2
	if _, _, ok := s.Pick(now); !ok {
		t.Fatal("fresh must dispatch")
	}
	s.Settle("fresh", "rf", keys(900, 2), struct{ Prompt, Cached int }{900, 0}, true, now)
	// re re-arrives (a waiter, NOT a newcomer); a fresh newcomer arrives after
	s.Arrive("re", "r5", keys(100, 1), now.Add(time.Millisecond))
	s.Arrive("fresh2", "rf2", keys(900, 3), now.Add(2*time.Millisecond))
	s.Tick(now.Add(time.Second))
	// budget seats one: the NEWCOMER (fresh2) goes before the ex-member (re)
	if _, sess, ok := s.Pick(now.Add(time.Second)); !ok || sess != "fresh2" {
		t.Fatalf("newcomer must be seated before the ex-member waiter, got %v", sess)
	}
}

func TestSmokeTTLSilenceForgets(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	s.Arrive("a", "r1", keys(10, 1), now)
	s.Tick(now)
	if _, _, ok := s.Pick(now); !ok {
		t.Fatal("pick")
	}
	s.Settle("a", "r1", keys(10, 1), struct{ Prompt, Cached int }{10, 0}, true, now)
	if n := s.Sessions(); n != 1 {
		t.Fatalf("tracked sessions: %d", n)
	}
	// silence past the TTL: unseat AND forget (the CP1 condition)
	later := now.Add(time.Duration(defaultConfig().TTLSeconds*float64(time.Second)) + time.Second)
	s.Tick(later)
	if n := s.Sessions(); n != 0 {
		t.Fatalf("record leaked after TTL: %d sessions tracked (CP1 condition)", n)
	}
	if snap := s.Snapshot(); snap.Seats != 0 {
		t.Fatalf("seats after TTL: %d", snap.Seats)
	}
}

func TestSmokeGrowthReEnforcement(t *testing.T) {
	s := newSched(t)
	now := time.Now()
	// two members of 400 blocks each (800 <= 950): both seat
	s.Arrive("a", "ra", keys(400, 1), now)
	s.Arrive("b", "rb", keys(400, 2), now)
	s.Tick(now)
	if snap := s.Snapshot(); snap.Seats != 2 {
		t.Fatalf("seats: %d", snap.Seats)
	}
	// both grow (D13): each adds 300 new blocks on the next turn
	s.Arrive("a", "ra2", keys(700, 1), now) // 400 old + 300 new
	s.Arrive("b", "rb2", keys(700, 2), now)
	s.Tick(now)
	snap := s.Snapshot()
	if snap.NEBlocks > snap.BudgetBlocks {
		t.Fatalf("budget violated after growth: %d > %d", snap.NEBlocks, snap.BudgetBlocks)
	}
	if snap.Seats > 2 {
		t.Fatalf("seats grew: %d", snap.Seats)
	}
}
