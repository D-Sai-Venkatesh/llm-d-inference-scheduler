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

// armB is wave 3's arm B: ungated, cost-fair rotation, cold valve off — the base leastRecentlyServed is
// built on (one change on top of it).
func armB(t *testing.T, lrs bool) *Scheduler {
	t.Helper()
	return roomSched(t, false, func(c *Config) {
		c.RoomGate, c.RoomDisplace, c.RoomHoldMaxSeconds, c.RoomReserveTokens = false, false, 0, 0
		c.Quantum, c.NewcomersFirst = 8, true
		c.FairRotation, c.CostFairRotation, c.ColdValveOff = true, true, true
		c.LeastRecentlyServed = lrs
	})
}

// x was served 5 min ago (a long decode since), y 1 min ago; y's next turn ARRIVED earlier. Arrival order serves
// y; least-recently-served serves x — the chain closest to the tier's eviction end — and counts the overtake.
// The first turn of a stay (n, just seated, never served in it) goes before both.
func TestLeastRecentlyServedServesTheStalestFirst(t *testing.T) {
	for _, c := range []struct {
		name    string
		lrs     bool
		withNew bool
		want    string
	}{
		{"arrival order without it", false, false, "y"},
		{"served longest ago first", true, false, "x"},
		{"the first turn of a stay first of all", true, true, "n"},
	} {
		s := armB(t, c.lrs)
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 20)), epoch.Add(-300*time.Second))
		runTurn(t, s, headN("y", 0, time.Second, chain("y", 20)), epoch.Add(-60*time.Second))
		heads := []Head{
			headN("x", 1, time.Second, chain("x", 24)),
			headN("y", 1, 30*time.Second, chain("y", 24)),
		}
		if c.withNew {
			heads = append(heads, headN("n", 0, 500*time.Millisecond, chain("n", 10)))
		}
		i := s.Pass(heads, epoch)
		got := "none"
		if i >= 0 {
			got = heads[i].Thread
		}
		if got != c.want {
			t.Fatalf("%s: dispatched %s, want %s", c.name, got, c.want)
		}
		if over := s.counters.sizeWaitOvertakes; (c.lrs && over != 1) || (!c.lrs && over != 0) {
			t.Fatalf("%s: overtakes %d", c.name, over)
		}
		// the served member goes to the back: the next pass serves the other one (round robin)
		if !c.withNew {
			rest := []Head{heads[1-i]}
			if j := s.Pass(rest, epoch.Add(time.Second)); j != 0 {
				t.Fatalf("%s: the other member must be served next, got %d", c.name, j)
			}
		}
	}
}

// One in-group order at a time.
func TestLeastRecentlyServedIsOneOrder(t *testing.T) {
	for _, mut := range []func(*Config){
		func(c *Config) { c.SizeWaitOrder = true },
		func(c *Config) { c.QuantumDoneLast = true },
	} {
		cfg := defaultConfig()
		cfg.LeastRecentlyServed = true
		mut(&cfg)
		if _, err := NewScheduler(cfg); err == nil {
			t.Fatal("leastRecentlyServed with another in-group order must be refused")
		}
	}
	cfg := defaultConfig()
	cfg.LeastRecentlyServed = true
	if _, err := NewScheduler(cfg); err == nil {
		t.Fatal("leastRecentlyServed without ownTierIndex must be refused (its protection reads the tier index)")
	}
	cfg.TierCapacityTokens, cfg.OwnTierIndex = 2000*64, true
	if _, err := NewScheduler(cfg); err != nil {
		t.Fatalf("leastRecentlyServed with ownTierIndex must be accepted: %v", err)
	}
}

// tinyTier is arm B on a 200-block tier with a 1000-block GPU: the seat budget (950) exceeds the tier, so the
// seated members alone fill it and the bottom of the LRU is set exactly by the order of their turns.
func tinyTier(t *testing.T, plain bool, quantum int) *Scheduler {
	t.Helper()
	s := roomSched(t, false, func(c *Config) {
		c.TierCapacityTokens = 200 * 64
		c.RoomGate, c.RoomDisplace, c.RoomHoldMaxSeconds, c.RoomReserveTokens = false, false, 0, 0
		c.Quantum, c.NewcomersFirst = quantum, true
		c.FairRotation, c.CostFairRotation, c.ColdValveOff = true, true, true
		c.LeastRecentlyServed = true
	})
	s.plainLRS = plain
	return s
}

// A cold write (a newcomer's first turn: 60 blocks, 5 free) would evict y's whole chain, which sits at the
// bottom (y served longest ago). Plain least-recently-served sends the newcomer first (the cold cascade);
// the strict rule serves y first: a = 30 of y's blocks in the newcomer's danger zone, b = 0 (y's own write is
// its new context, 2 blocks, which the free room absorbs).
func TestStrictRuleProtectsAQueuedChainFromAColdWrite(t *testing.T) {
	for _, plain := range []bool{true, false} {
		s := tinyTier(t, plain, 4)
		runTurn(t, s, headN("y", 0, time.Second, chain("y", 30)), epoch.Add(-300*time.Second))
		runTurn(t, s, headN("f", 0, time.Second, chain("f", 165)), epoch.Add(-30*time.Second))
		heads := []Head{
			headN("y", 1, 10*time.Second, chain("y", 32)),
			headN("n", 0, time.Second, chain("n", 60)),
		}
		i := s.Pass(heads, epoch)
		want := "y"
		if plain {
			want = "n"
		}
		if i < 0 || heads[i].Thread != want {
			t.Fatalf("plain=%v: dispatched %d, want %s", plain, i, want)
		}
		snap := s.Snapshot(epoch)
		if !plain && (snap.Room.Protected != 1 || snap.Room.ProtectedBlocks != 30) {
			t.Fatalf("strict rule: protected %d blocks %d, want 1 / 30", snap.Room.Protected, snap.Room.ProtectedBlocks)
		}
		if plain && snap.Room.Protected != 0 {
			t.Fatalf("plain order must never protect, got %d", snap.Room.Protected)
		}
	}
}

// Net save. x (served longest ago, its tail at the very bottom) is warm: its write is only its new context.
// y sits just above x. 3 blocks free.
//   - x writes 4 → its danger zone is 1 of y's blocks (a = 1); y's write (6) would evict 3 of x's tail
//     blocks that x's write spares (b = 3): no net save → x goes (declined).
//   - x writes 10 → 7 of y's blocks at risk (a = 7 > b = 3) → y goes first (protected).
func TestStrictRuleNetSave(t *testing.T) {
	for _, c := range []struct {
		name           string
		xBlocks        int
		want           string
		prot, declined int64
	}{
		{"tiny save, x would lose its tail: declined", 44, "x", 0, 1},
		{"a real save: protected", 50, "y", 1, 0},
	} {
		s := tinyTier(t, false, 4)
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 40)), epoch.Add(-300*time.Second))
		runTurn(t, s, headN("y", 0, time.Second, chain("y", 30)), epoch.Add(-250*time.Second))
		runTurn(t, s, headN("f", 0, time.Second, chain("f", 127)), epoch.Add(-30*time.Second))
		heads := []Head{
			headN("x", 1, 2*time.Second, chain("x", c.xBlocks)),
			headN("y", 1, 5*time.Second, chain("y", 36)), // arrived first: arrival order would serve y
		}
		i := s.Pass(heads, epoch)
		if i < 0 || heads[i].Thread != c.want {
			t.Fatalf("%s: dispatched %d, want %s", c.name, i, c.want)
		}
		snap := s.Snapshot(epoch)
		if snap.Room.Protected != c.prot || snap.Room.ProtectDeclined != c.declined {
			t.Fatalf("%s: protected %d declined %d, want %d / %d", c.name, snap.Room.Protected, snap.Room.ProtectDeclined, c.prot, c.declined)
		}
	}
}

// Protectable: a quantum-complete member is protected only while nobody waits outside (with someone waiting
// it leaves soon, and its chain is lost during the exile anyway).
func TestStrictRuleSkipsQuantumCompleteWhileSomeoneWaits(t *testing.T) {
	s := tinyTier(t, false, 1) // quantum 1: y is quantum-complete after its first turn
	runTurn(t, s, headN("y", 0, time.Second, chain("y", 30)), epoch.Add(-300*time.Second))
	runTurn(t, s, headN("f", 0, time.Second, chain("f", 165)), epoch.Add(-30*time.Second))
	g := tierOf(s)
	n, y := headN("n", 0, time.Second, chain("n", 60)), headN("y", 1, 10*time.Second, chain("y", 32))
	if j, a, _ := s.protectFirst(g, n, []Head{y}); j != 0 || a != 30 {
		t.Fatalf("nobody waiting: the quantum-complete member is protectable, got j=%d a=%d", j, a)
	}
	s.waiters["w"] = &waiter{id: "w", since: epoch}
	if j, _, _ := s.protectFirst(g, n, []Head{y}); j != -1 {
		t.Fatalf("someone waiting: a quantum-complete member is not protected, got j=%d", j)
	}
}
