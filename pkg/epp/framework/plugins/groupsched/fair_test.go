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
	"fmt"
	"testing"
	"time"
)

func fairSched(t *testing.T) *Scheduler {
	t.Helper()
	return roomSched(t, true, func(c *Config) {
		c.RoomDisplace = true
		c.RoomHoldMaxSeconds = 0
		c.Quantum = 4
		c.NewcomersFirst = true
		c.FairRotation = true
	})
}

// One design, not a stack: fairRotation refuses the older rotation knobs and needs the room gate +
// displace.
func TestFairRotationIsExclusive(t *testing.T) {
	base := func(c *Config) {
		c.TierCapacityTokens = 64000
		c.OwnTierIndex = true
		c.RoomGate = true
		c.RoomDisplace = true
		c.RoomHoldMaxSeconds = 0
		c.FairRotation = true
	}
	for name, f := range map[string]func(*Config){
		"with roomVictims":     func(c *Config) { c.RoomVictims = true },
		"with quantumLastTurn": func(c *Config) { c.QuantumLastTurn = true },
		"with quantumDoneLast": func(c *Config) { c.QuantumDoneLast = true },
		"with a forced seat":   func(c *Config) { c.RoomHoldMaxSeconds = 90 },
		"without roomDisplace": func(c *Config) { c.RoomDisplace = false },
		"without ownTierIndex": func(c *Config) { c.OwnTierIndex = false },
	} {
		cfg := defaultConfig()
		base(&cfg)
		f(&cfg)
		if _, err := NewScheduler(cfg); err == nil {
			t.Errorf("fairRotation %s must be refused", name)
		}
	}
	cfg := defaultConfig()
	base(&cfg)
	if _, err := NewScheduler(cfg); err != nil {
		t.Fatalf("the plain fair config must load: %v", err)
	}
	// ungated (the CPU-warm group's successor): no room gate, no displace — loads with the own tier index
	cfg.RoomGate, cfg.RoomDisplace = false, false
	if _, err := NewScheduler(cfg); err != nil {
		t.Fatalf("ungated fair rotation must load: %v", err)
	}
}

// The budget's order: idle first, then the LONGEST TIME IN THE GROUP since joining among quantum-done
// members — not most turns (a fast member that made more turns but joined later goes after).
func TestFairRotationBudgetOrderIsSeatTime(t *testing.T) {
	s := fairSched(t)
	runTurn(t, s, headN("i", 0, time.Second, chain("i", 50)), epoch) // i: one turn, then silent
	at := epoch.Add(time.Duration(s.cfg.IdleSeconds+5) * time.Second)
	for k := 0; k < 5; k++ { // slow: joined FIRST (of the active two), 5 turns
		runTurn(t, s, headN("slow", k, time.Second, chain("slow", 50)), at)
		at = at.Add(3 * time.Second)
	}
	for k := 0; k < 9; k++ { // fast: joined later, 9 turns
		runTurn(t, s, headN("fast", k, time.Second, chain("fast", 50)), at)
		at = at.Add(time.Second)
	}
	var order []string
	for k := 0; k < 3; k++ {
		s.mu.Lock()
		v := s.fairVictim(s.groups[""], nil, at)
		if v != nil {
			order = append(order, v.id)
			s.exit(s.groups[""], v, "test", at)
		}
		s.mu.Unlock()
	}
	if fmt.Sprint(order) != "[i slow fast]" {
		t.Fatalf("fair budget order must be idle first, then longest in the group: got %v", order)
	}
}

// A quantum-done member that is IN FLIGHT is a victim only past its first token; removed, it is out at
// once — its seat budget is free (the stay side) before its request ends — and it leaves at the end.
func TestFairRotationRunningVictimAfterFirstToken(t *testing.T) {
	s := fairSched(t)
	at := epoch
	for k := 0; k < 3; k++ { // q: 3 turns, then its 4th (quantum-completing) RUNNING
		runTurn(t, s, headN("q", k, time.Second, chain("q", 300)), at)
		at = at.Add(time.Second)
	}
	q4 := headN("q", 3, time.Second, chain("q", 300))
	if i := s.Pass([]Head{q4}, at); i != 0 {
		t.Fatal("q's 4th turn must dispatch")
	}
	s.AtPod(q4.ReqID, "", at)
	s.mu.Lock()
	before := s.fairVictim(s.groups[""], nil, at)
	s.mu.Unlock()
	if before != nil {
		t.Fatal("before its first token q is not a victim (its prefill may still write into the tier)")
	}
	s.FirstChunk(q4.ReqID, at)
	s.mu.Lock()
	g := s.groups[""]
	v := s.fairVictim(g, nil, at)
	if v == nil || v.id != "q" {
		s.mu.Unlock()
		t.Fatalf("past its first token q (running its quantum-completing turn) is a victim, got %v", v)
	}
	free0 := s.freeBudget(g)
	s.fairRemove(g, v, &waiter{id: "z", since: at}, "fair-budget", at)
	free1 := s.freeBudget(g)
	exiting := g.index.Exiting("q")
	s.mu.Unlock()
	if !exiting || free1 != free0+300 {
		t.Fatalf("removed in flight: marked leaving and its 300 blocks of budget free at once: exiting %v free %d -> %d", exiting, free0, free1)
	}
	s.Settle(q4.ReqID, "", at.Add(time.Second))
	if seated(s, "q") || s.Snapshot(at).Exits["fair-budget"] != 1 {
		t.Fatalf("q must leave at its request's end: seated %v exits %+v", seated(s, "q"), s.Snapshot(at).Exits)
	}
}

// The room takes only the member AT THE BOTTOM, and only if it may leave: a mid-quantum member at the
// bottom makes the head wait (nobody is removed); a quantum-done one at the bottom is removed and the
// head admitted — while the longest-seated quantum-done member elsewhere is untouched by the room.
func TestFairRotationRoomTakesOnlyTheBottom(t *testing.T) {
	for _, bottomDone := range []bool{false, true} {
		s := fairSched(t)
		at := epoch
		turns := 2 // mid-quantum
		if bottomDone {
			turns = 4
		}
		for k := 0; k < turns; k++ { // b: the OLDEST blocks — the bottom
			runTurn(t, s, headN("b", k, time.Second, chain("b", 200)), at)
			at = at.Add(time.Second)
		}
		for k := 0; k < 6; k++ { // f: quantum done, joined later, fresh blocks
			runTurn(t, s, headN("f", k, time.Second, chain("f", 300)), at)
			at = at.Add(time.Second)
		}
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 600)), at)
		evict(t, s, "x") // free 2000-200-300-600 = 900
		at = at.Add(time.Second)
		z := headN("z", 0, time.Second, chain("z", 1000)) // budget 200+300+1000 <= 1900; free 900 < 1000
		s.Pass([]Head{z}, at)
		if !seated(s, "f") {
			t.Fatalf("bottomDone=%v: the room never takes a member that is not in the way (f)", bottomDone)
		}
		if bottomDone {
			if seated(s, "b") || !seated(s, "z") {
				t.Fatalf("quantum-done member at the bottom: removed, head admitted: b %v z %v", seated(s, "b"), seated(s, "z"))
			}
			if s.Snapshot(at).Exits["fair-room"] != 1 {
				t.Fatalf("one fair-room removal, got %+v", s.Snapshot(at).Exits)
			}
		} else if !seated(s, "b") || seated(s, "z") {
			t.Fatalf("mid-quantum member at the bottom: protected, head waits: b %v z %v", seated(s, "b"), seated(s, "z"))
		}
	}
}

// costFairRotation needs fairRotation (it is fair rotation's victim order, not a design of its own).
func TestCostFairRotationNeedsFairRotation(t *testing.T) {
	cfg := defaultConfig()
	cfg.TierCapacityTokens = 64000
	cfg.OwnTierIndex = true
	cfg.RoomGate = true
	cfg.RoomDisplace = true
	cfg.RoomHoldMaxSeconds = 0
	cfg.CostFairRotation = true
	if _, err := NewScheduler(cfg); err == nil {
		t.Fatal("costFairRotation without fairRotation must be refused")
	}
	cfg.FairRotation = true
	if _, err := NewScheduler(cfg); err != nil {
		t.Fatalf("costFairRotation with fairRotation must load: %v", err)
	}
}

// The one change: among quantum-done members the highest seat time ÷ context size leaves first. A big member
// seated long ago (100 s, 300 blocks → 0.33 s/block) stays before a small one seated later (30 s, 50 blocks →
// 0.6 s/block); plain fair rotation (seat time alone) takes the big one. Idle still goes first in both.
func TestCostFairRotationOrderIsSeatTimePerBlock(t *testing.T) {
	for _, cost := range []bool{false, true} {
		s := roomSched(t, true, func(c *Config) {
			c.RoomDisplace = true
			c.RoomHoldMaxSeconds = 0
			c.Quantum = 4
			c.NewcomersFirst = true
			c.FairRotation = true
			c.CostFairRotation = cost
		})
		runTurn(t, s, headN("i", 0, time.Second, chain("i", 40)), epoch) // i: one turn, then silent (idle)
		for k, at := range []int{0, 30, 60, 90} {                        // big: seated at 0, 4 turns
			runTurn(t, s, headN("big", k, time.Second, chain("big", 300)), epoch.Add(time.Duration(at)*time.Second))
		}
		for k, at := range []int{70, 80, 90, 95} { // small: seated at 70, 4 turns
			runTurn(t, s, headN("small", k, time.Second, chain("small", 50)), epoch.Add(time.Duration(at)*time.Second))
		}
		now := epoch.Add(100 * time.Second)
		var order []string
		for k := 0; k < 3; k++ {
			s.mu.Lock()
			if v := s.fairVictim(s.groups[""], nil, now); v != nil {
				order = append(order, v.id)
				s.exit(s.groups[""], v, "test", now)
			}
			s.mu.Unlock()
		}
		want := "[i big small]"
		if cost {
			want = "[i small big]"
		}
		if fmt.Sprint(order) != want {
			t.Fatalf("costFairRotation=%v: victim order %v, want %s", cost, order, want)
		}
	}
}

// The cost counts only blocks the member ALONE holds among staying members: a shared prefix stays cached and
// charged by the others, so it is neither freed nor rebuilt when this member leaves.
func TestCostFairCostIsExclusiveBlocks(t *testing.T) {
	s := roomSched(t, true, func(c *Config) {
		c.RoomDisplace, c.RoomHoldMaxSeconds, c.Quantum, c.NewcomersFirst = true, 0, 4, true
		c.FairRotation, c.CostFairRotation = true, true
	})
	at := epoch
	for k := 0; k < 4; k++ { // p: a parent with a 200-block prefix
		runTurn(t, s, headN("p", k, time.Second, chain("p", 200)), at)
		at = at.Add(time.Second)
	}
	for k := 0; k < 4; k++ { // sub: 200 shared + 100 own = 300 charged, 100 exclusive
		runTurn(t, s, headN("sub", k, time.Second, chainWithPrefix("p", 200, "sub", 100)), at)
		at = at.Add(time.Second)
	}
	for k := 0; k < 4; k++ { // solo: 250 own, all exclusive
		runTurn(t, s, headN("solo", k, time.Second, chain("solo", 250)), at)
		at = at.Add(time.Second)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groups[""]
	cSub, _ := s.fairCost(g, g.threads["sub"], fairTail{})
	cSolo, _ := s.fairCost(g, g.threads["solo"], fairTail{})
	if cSub != 100 || cSolo != 250 {
		t.Fatalf("exclusive cost: sub %d (want 100 — its 200 shared blocks stay), solo %d (want 250)", cSub, cSolo)
	}
}

func stayingEvicted(s *Scheduler) int64 {
	var n int64
	for _, v := range s.Snapshot(epoch).Room.SeatedEvicted {
		n += v
	}
	return n
}

// UNGATED, the tail term: a head's write evicts the bottom of the LRU whoever owns it. Choosing as victim the
// eligible member whose blocks are there turns that thrash into its own loss — no staying block evicted —
// while seat time alone takes the longest-seated member and the write thrashes the one at the bottom.
func TestCostFairUngatedVictimAtTheBottomSavesThrash(t *testing.T) {
	for _, cost := range []bool{false, true} {
		s := roomSched(t, false, func(c *Config) {
			c.TierCapacityTokens = 1000 * 64
			c.RoomGate, c.RoomDisplace, c.RoomHoldMaxSeconds, c.RoomReserveTokens = false, false, 0, 0
			c.Quantum, c.NewcomersFirst = 4, true
			c.FairRotation, c.CostFairRotation = true, cost
		})
		// d: joined first (longest seat time); c: joined later, its LAST dispatch is the oldest -> its blocks
		// are at the bottom. Both run their 4th (quantum-completing) turn, past first token: both eligible.
		for k, at := range []int{0, 10, 20} {
			runTurn(t, s, headN("d", k, time.Second, chain("d", 300)), epoch.Add(time.Duration(at)*time.Second))
		}
		for k, at := range []int{5, 6, 7} {
			runTurn(t, s, headN("c", k, time.Second, chain("c", 300)), epoch.Add(time.Duration(at)*time.Second))
		}
		c4 := headN("c", 3, time.Second, chain("c", 300))
		s.Pass([]Head{c4}, epoch.Add(8*time.Second))
		s.AtPod(c4.ReqID, "", epoch.Add(8*time.Second))
		s.FirstChunk(c4.ReqID, epoch.Add(9*time.Second))
		d4 := headN("d", 3, time.Second, chain("d", 300))
		s.Pass([]Head{d4}, epoch.Add(30*time.Second))
		s.AtPod(d4.ReqID, "", epoch.Add(30*time.Second))
		s.FirstChunk(d4.ReqID, epoch.Add(31*time.Second))
		before := stayingEvicted(s)
		// z: 500 cold blocks; free tier 400 -> its write evicts the bottom 100 (c's); budget 600+500 > 950
		z := headN("z", 0, time.Second, chain("z", 500))
		s.Pass([]Head{z}, epoch.Add(40*time.Second))
		s.AtPod(z.ReqID, "", epoch.Add(40*time.Second))
		thrash := stayingEvicted(s) - before
		if !seated(s, "z") {
			t.Fatalf("cost=%v: ungated, the head is admitted once the budget fits", cost)
		}
		s.mu.Lock()
		cOut, dOut := s.groups[""].index.Exiting("c"), s.groups[""].index.Exiting("d")
		s.mu.Unlock()
		saved := s.Snapshot(epoch).Room.FairTailSaved
		if cost {
			if !cOut || dOut || thrash != 0 || saved != 100 {
				t.Fatalf("cost-fair: c (at the bottom) leaves, nothing staying is evicted: c out %v d out %v thrash %d saved %d",
					cOut, dOut, thrash, saved)
			}
		} else if !dOut || cOut || thrash != 100 {
			t.Fatalf("seat time alone: d (longest seated) leaves and the write thrashes c's 100 bottom blocks: c out %v d out %v thrash %d",
				cOut, dOut, thrash)
		}
	}
}
