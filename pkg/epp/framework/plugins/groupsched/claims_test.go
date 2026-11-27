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
	"strings"
	"testing"
	"time"
)

// armCConfig is arm C's design (ungated, cost-fair, cold valve off, the strict rule) with claimOrder on top.
func armCConfig(tierBlocks int64, quantum int, claimOrder string) Config {
	cfg := defaultConfig()
	cfg.TierCapacityTokens = tierBlocks * 64
	cfg.OwnTierIndex, cfg.RoomGate, cfg.RoomDisplace, cfg.RoomHoldMaxSeconds, cfg.RoomReserveTokens = true, false, false, 0, 0
	cfg.Quantum, cfg.NewcomersFirst = quantum, true
	cfg.FairRotation, cfg.CostFairRotation, cfg.ColdValveOff = true, true, true
	cfg.VictimOldestDispatch, cfg.LeastRecentlyServed = true, true
	cfg.ClaimOrder = claimOrder
	return cfg
}

func TestClaimOrderConfig(t *testing.T) {
	for _, order := range []string{"compute", "size"} {
		if _, err := NewScheduler(armCConfig(2000, 8, order)); err != nil {
			t.Fatalf("claimOrder %q on arm C's line must be accepted: %v", order, err)
		}
	}
	bad := armCConfig(2000, 8, "nearest")
	if _, err := NewScheduler(bad); err == nil {
		t.Fatal("an unknown claimOrder must be refused")
	}
	gated := armCConfig(2000, 8, "compute")
	gated.RoomGate = true
	if _, err := NewScheduler(gated); err == nil {
		t.Fatal("claimOrder with the room gate must be refused (built for the ungated line)")
	}
	valve := armCConfig(2000, 8, "size")
	valve.ColdValveOff = false
	if _, err := NewScheduler(valve); err == nil {
		t.Fatal("claimOrder with the cold valve on must be refused")
	}
}

// With ONE engine every decision is today's: the replay's event log (every SEAT, EXIT, …) and its snapshot are
// identical with claimOrder off, "compute" and "size".
func TestClaimsOneEngineIsToday(t *testing.T) {
	run := func(seed, tier int64, order string) (string, Snapshot) {
		s, err := NewScheduler(armCConfig(tier, 4, order))
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		driveReplay(t, s, seed, 500, &out, replayOpts{noMoves: true, engines: []string{"a"}}, func(int) {})
		return out.String(), s.Snapshot(epoch)
	}
	for _, tier := range []int64{700, 1000, 2000} {
		for seed := int64(21); seed <= 24; seed++ {
			base, bs := run(seed, tier, "")
			for _, order := range []string{"compute", "size"} {
				got, gs := run(seed, tier, order)
				if got != base {
					t.Fatalf("tier %d seed %d claimOrder %q: one engine must decide exactly as today (event logs differ)", tier, seed, order)
				}
				if gs.Served != bs.Served || fmt.Sprint(gs.Exits) != fmt.Sprint(bs.Exits) {
					t.Fatalf("tier %d seed %d claimOrder %q: served %d vs %d, exits %v vs %v", tier, seed, order, gs.Served, bs.Served, gs.Exits, bs.Exits)
				}
			}
		}
	}
}

// claimTwoEngines seats m1 (300) + m2 (250) on a (free 400), then adds b and seats m3 (350) + m4 (300) on it (free
// 300); every member has run `turns` turns, all in the last 20 s (none idle).
func claimTwoEngines(t *testing.T, quantum int, order string, m4Turns int) *Scheduler {
	t.Helper()
	s, err := NewScheduler(armCConfig(200, quantum, order)) // tier share 0.95×200 < the GPU budget: the seat budget is 950
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	runTurn(t, s, headN("m1", 0, time.Second, chain("m1", 300)), epoch.Add(-9*time.Second))
	runTurn(t, s, headN("m2", 0, time.Second, chain("m2", 250)), epoch.Add(-8*time.Second))
	if err := s.SetNumBlocksFor("b", 64, 1000); err != nil {
		t.Fatal(err)
	}
	runTurn(t, s, headN("m3", 0, time.Second, chain("m3", 350)), epoch.Add(-7*time.Second))
	for k := 0; k < m4Turns; k++ {
		runTurn(t, s, headN("m4", k, time.Second, chain("m4", 300)), epoch.Add(time.Duration(-6+k)*time.Second))
	}
	for id, want := range map[string]string{"m1": "a", "m2": "a", "m3": "b", "m4": "b"} {
		if got := s.bindings[id]; got != want {
			t.Fatalf("setup: %s seated on %q, want %q", id, got, want)
		}
	}
	return s
}

// The head-of-line fix. H (500) fits neither a (free 400) nor b (free 300) and nobody may leave; S (200) is
// behind it and fits b. Today S waits behind H; with claims H claims a (shortfall 100 < 200) and S is seated on b.
// T (350) fits only a — closed behind H — so it claims b, the last open engine.
func TestClaimsLetOthersPastABlockedHead(t *testing.T) {
	for _, order := range []string{"", "compute", "size"} {
		s := claimTwoEngines(t, 8, order, 1)
		// T (350) comes after S: it fits only a (free 400; b keeps 100 after S) — and a is H's claim, closed to T
		heads := []Head{headN("H", 0, 3*time.Second, chain("H", 500)), headN("S", 0, 2*time.Second, chain("S", 200)),
			headN("T", 0, time.Second, chain("T", 350))}
		s.Pass(heads, epoch)
		_, hSeated := s.bindings["H"]
		sEngine, sSeated := s.bindings["S"]
		if _, tSeated := s.bindings["T"]; tSeated {
			t.Fatalf("claimOrder %q: T fits only H's claimed engine, which is closed behind H: T must wait", order)
		}
		snap := s.Snapshot(epoch)
		if order == "" {
			if hSeated || sSeated {
				t.Fatalf("today: H blocks the queue, nobody is seated (H %v, S %v)", hSeated, sSeated)
			}
			continue
		}
		if hSeated || !sSeated || sEngine != "b" {
			t.Fatalf("claimOrder %q: H waits on its claim and S is seated on b; got H %v, S %v on %q", order, hSeated, sSeated, sEngine)
		}
		if s.waiters["H"] == nil || s.waiters["H"].claim != "a" {
			t.Fatalf("claimOrder %q: H must claim a (the smaller shortfall)", order)
		}
		// T, fitting nowhere open (a is H's; b keeps 100), becomes b's blocked head: two claims, every engine held
		if s.waiters["T"] == nil || s.waiters["T"].claim != "b" {
			t.Fatalf("claimOrder %q: T must claim b (the only open engine)", order)
		}
		if snap.Room.Claims != 2 || snap.Room.SeatedPastClaim != 1 || snap.Room.ClaimVictimsElsewhere != 0 {
			t.Fatalf("claimOrder %q: claims %d, seated past claim %d, victims elsewhere %d; want 2 / 1 / 0",
				order, snap.Room.Claims, snap.Room.SeatedPastClaim, snap.Room.ClaimVictimsElsewhere)
		}
	}
}

// Victims come only from the claim (no scattering) — and the trade-off it carries, recorded here on purpose:
// m4 on b is quantum-complete (quantum 2). Today H fits nowhere, the eviction goes to the engine that has a
// victim (b): m4 leaves, H is seated on b. With claims H claims a (smaller shortfall), nobody on a may leave,
// so H WAITS there; m4 is NOT removed for H, and S (200) is seated on b without any eviction.
func TestClaimsVictimsOnlyFromTheClaim(t *testing.T) {
	for _, order := range []string{"", "compute"} {
		s := claimTwoEngines(t, 2, order, 2)
		heads := []Head{headN("H", 0, 3*time.Second, chain("H", 500)), headN("S", 0, 2*time.Second, chain("S", 200))}
		s.Pass(heads, epoch)
		_, m4Seated := s.bindings["m4"]
		hEngine, hSeated := s.bindings["H"]
		if order == "" {
			if m4Seated || !hSeated || hEngine != "b" {
				t.Fatalf("today: m4 (b) leaves for H, H is seated on b; got m4 seated %v, H on %q (%v)", m4Seated, hEngine, hSeated)
			}
			continue
		}
		if !m4Seated || hSeated {
			t.Fatalf("claims: victims for H only from its claim a: m4 stays, H waits; got m4 seated %v, H seated %v", m4Seated, hSeated)
		}
		if e, ok := s.bindings["S"]; !ok || e != "b" {
			t.Fatalf("claims: S is seated on b; got %q (%v)", e, ok)
		}
	}
}

// Three engines, the replay's churn (engine c leaves and returns): claims never starve the workload, and the
// counters stay sane (victims removed for a waiter then seated elsewhere ≈ 0).
func TestClaimsReplayThreeEngines(t *testing.T) {
	run := func(seed, tier int64, order string) Snapshot {
		s, err := NewScheduler(armCConfig(tier, 4, order))
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		var bad error
		driveReplay(t, s, seed, 500, &out, replayOpts{noMoves: true}, func(int) {
			if bad == nil {
				bad = runSideExact(s)
			}
		})
		if bad != nil {
			t.Fatalf("claimOrder %q seed %d: %v", order, seed, bad)
		}
		return s.Snapshot(epoch)
	}
	for _, tier := range []int64{700, 1000, 1400} {
		for seed := int64(21); seed <= 26; seed++ {
			off := run(seed, tier, "")
			for _, order := range []string{"compute", "size"} {
				on := run(seed, tier, order)
				if on.Served < off.Served/2 {
					t.Fatalf("tier %d seed %d claimOrder %q must not starve the workload: served %d vs %d", tier, seed, order, on.Served, off.Served)
				}
				if on.Room.ClaimVictimsElsewhere > on.Room.Claims {
					t.Fatalf("tier %d seed %d claimOrder %q: victims elsewhere %d > claims %d", tier, seed, order, on.Room.ClaimVictimsElsewhere, on.Room.Claims)
				}
				var ev, ov int64
				for _, v := range on.Room.SeatedEvicted {
					ev += v
				}
				for _, v := range off.Room.SeatedEvicted {
					ov += v
				}
				t.Logf("tier %4d seed %d %-7s: served %d vs %d (off); claims %d, seated past a claim %d, victims elsewhere %d; staying evicted %d vs %d; tokens given up %d",
					tier, seed, order, on.Served, off.Served, on.Room.Claims, on.Room.SeatedPastClaim, on.Room.ClaimVictimsElsewhere, ev, ov, on.Room.TokensGivenUp)
			}
		}
	}
}
