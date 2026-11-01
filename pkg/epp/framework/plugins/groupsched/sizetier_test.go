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

// THE SIZING ALTERNATIVE "gpu-plus-tier" (PREREG-multipod-offload):
//
//	BudgetBlocks = floor(groupBudgetFraction x (num_gpu_blocks + tierBlocks))
//
// tierBlocks is the CPU tier SHADOW's own capacity, so the value adds no constant. Four
// properties are pinned here, at N=1 AND at N=2 (forEngines / forEnginesCap — the value
// is selected per ARM, and each engine's budget comes from its OWN capacity and its OWN
// tier):
//
//	(1) the budget IS the sum when a tier is configured (per engine, and the
//	    whole-policy budget is their sum);
//	(2) the DEFAULT is unchanged — same selector value, same budget, with or without a
//	    tier configured (the golden replay TestGolden runs on that config, byte for
//	    byte);
//	(3) the cold-entry valve's cap stays the GPU headroom: it staggers GPU allocation
//	    bursts, and the tier cannot absorb one;
//	(4) tier = 0 ⇒ the value is arithmetically "index": same budget, same decisions,
//	    same costs.
//
// The fifth check lives in config_conformance_test.go (the selector value the emitter
// renders) and the sixth in cmd/epp/runner (the loader path a real arm starts through).

import (
	"fmt"
	"testing"
	"time"
)

// tierBudgetConfig is the default config with the SIZING selector set to gpu-plus-tier
// and a CPU tier of capacityBlocks. The tier facts follow the index's own 4-bytes/token
// convention (bytesFor/IndexSizer), so a block means one thing everywhere.
func tierBudgetConfig(capacityBlocks int) Config {
	c := defaultConfig()
	c.Sizing = "gpu-plus-tier"
	c.KvBlockBytes = 4 * 64 // 4 bytes/token x 64-token blocks (the index's convention)
	c.CpuTierBytes = int64(capacityBlocks) * c.KvBlockBytes
	return c
}

// budgetFor is the rule under test, written the way the deployment reads it.
func budgetFor(fraction float64, numBlocks, tierBlocks int) int {
	return int(fraction * float64(numBlocks+tierBlocks))
}

// (1) THE BUDGET IS THE SUM.
func TestGpuPlusTierBudgetIsTheSum(t *testing.T) {
	const (
		capacity   = 1000
		tierBlocks = 2000
	)
	forEnginesCap(t, tierBudgetConfig(tierBlocks), capacity, func(t *testing.T, fx *engineSet) {
		want := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, tierBlocks)
		gpuOnly := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, 0)
		if want <= gpuOnly {
			t.Fatalf("the fixture must make the two budgets different: sum %d vs GPU-only %d", want, gpuOnly)
		}
		snap := fx.s.Snapshot()
		for _, id := range fx.ids {
			es := snap.Engines[engineLabel(id)]
			if es.NumBlocks != capacity {
				t.Fatalf("engine %s: measured capacity %d, wanted the fixture's %d", id, es.NumBlocks, capacity)
			}
			if es.TierCapacityBlocks != tierBlocks {
				t.Fatalf("engine %s: the tier shadow's capacity must be the number added to the budget, got %d (wanted %d)", id, es.TierCapacityBlocks, tierBlocks)
			}
			if es.BudgetBlocks != want {
				t.Fatalf("engine %s: budget %d, wanted floor(%v x (%d + %d)) = %d — sizing gpu-plus-tier sizes the budget from the engine's WHOLE reachable working set",
					id, es.BudgetBlocks, fx.s.cfg.GroupBudgetFraction, capacity, tierBlocks, want)
			}
		}
		if got, expected := snap.BudgetBlocks, want*len(fx.ids); got != expected {
			t.Fatalf("the whole-policy budget must be the per-engine sum, got %d wanted %d", got, expected)
		}
	})
}

// (2) THE DEFAULT IS UNCHANGED. The selector value the arms render ("index") keeps the
// GPU-only budget even when the deployment carries a tier — which is the shape of
// core-q4-offload.env (tier facts set, default sizing) and of the golden replay.
func TestGpuPlusTierDefaultIsUnchanged(t *testing.T) {
	mod, err := buildModules(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if got := mod.sizer.Name(); got != "index" {
		t.Fatalf("the default sizing value must stay %q, got %q", "index", got)
	}
	cfg := defaultConfig()
	cfg.KvBlockBytes = 4 * 64 // a tier IS configured, and it must not move the default
	cfg.CpuTierBytes = int64(2000) * cfg.KvBlockBytes
	forEnginesCap(t, cfg, 1000, func(t *testing.T, fx *engineSet) {
		snap := fx.s.Snapshot()
		for _, id := range fx.ids {
			es := snap.Engines[engineLabel(id)]
			if es.BudgetBlocks != 950 {
				t.Fatalf("engine %s: the default budget must stay floor(0.95 x 1000) = 950, got %d", id, es.BudgetBlocks)
			}
			if es.TierCapacityBlocks != 2000 {
				t.Fatalf("engine %s: the tier shadow must still be sized (%d blocks) — the default queue module and the shadow-validation series run on it", id, es.TierCapacityBlocks)
			}
			// the valve's cap is the GPU headroom in both cases (property (3) below)
			if want := int64(50) * 64; es.ColdCapTokens != want {
				t.Fatalf("engine %s: cold cap %d, wanted the GPU headroom %d", id, es.ColdCapTokens, want)
			}
		}
		if got := snap.BudgetBlocks; got != 950*len(fx.ids) {
			t.Fatalf("whole-policy budget %d, wanted %d", got, 950*len(fx.ids))
		}
	})
}

// (3) THE VALVE CAP IS UNCHANGED BY THE TIER. Same capacity, same fraction, two configs:
// the budget triples and the cap does not move, because a cold prefill allocates HBM
// whatever the tier holds.
func TestGpuPlusTierValveCapStaysGPUHeadroom(t *testing.T) {
	const (
		capacity   = 1000
		tierBlocks = 2000
	)
	// the GPU headroom in tokens: floor((1 - 0.95) x 1000) x 64 = 3200
	const wantCap = int64(50) * 64
	caps := map[string]int64{}
	for _, name := range []string{"gpu-plus-tier", "index"} {
		cfg := defaultConfig()
		if name == "gpu-plus-tier" {
			cfg = tierBudgetConfig(tierBlocks)
		}
		forEnginesCap(t, cfg, capacity, func(t *testing.T, fx *engineSet) {
			snap := fx.s.Snapshot()
			for _, id := range fx.ids {
				es := snap.Engines[engineLabel(id)]
				if es.ColdCapTokens != wantCap {
					t.Fatalf("%s engine %s: cold-entry cap %d tokens, wanted the GPU headroom %d — the cap staggers GPU allocation bursts and the tier cannot absorb one",
						name, id, es.ColdCapTokens, wantCap)
				}
				if es.ColdCapTokens >= int64(es.BudgetBlocks)*64 {
					t.Fatalf("%s engine %s: the cap (%d tokens) must stay well inside the budget (%d blocks)", name, id, es.ColdCapTokens, es.BudgetBlocks)
				}
				caps[name] = es.ColdCapTokens
			}
			// the budget, by contrast, DID move with the tier (property 1)
			es := snap.Engines[engineLabel(fx.ids[0])]
			wantBudget := budgetFor(cfg.GroupBudgetFraction, capacity, 0)
			if name == "gpu-plus-tier" {
				wantBudget = budgetFor(cfg.GroupBudgetFraction, capacity, tierBlocks)
			}
			if es.BudgetBlocks != wantBudget {
				t.Fatalf("%s: budget %d, wanted %d", name, es.BudgetBlocks, wantBudget)
			}
		})
	}
	if caps["gpu-plus-tier"] != caps["index"] {
		t.Fatalf("the tier must not move the valve: cap %d with the tier vs %d without", caps["gpu-plus-tier"], caps["index"])
	}
}

// driveFourBigMembers seats four 700-block members on a fresh scheduler of `engines`
// engines and returns the end state. One cold entry is admitted at a time (the derived
// valve cap), so each turn settles before the next head arrives, and members hold their
// seats afterwards. Under GPU-only sizing one 700-block member per 950-block budget fits;
// under gpu-plus-tier all four do (4 x 700 = 2800 <= 2850).
func driveFourBigMembers(t *testing.T, cfg Config, engines int) Snapshot {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < engines; i++ {
		if err := s.SetNumBlocksFor(fmt.Sprintf("e%d", i), 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 4; i++ {
		id := fmt.Sprintf("m%d", i)
		if s.PickFromHeads([]Head{head(id, time.Duration(i+1)*time.Second, bytesFor(700))}, time.Now()) < 0 {
			continue // no room on any engine: the head waits (the GPU-sized arm's outcome)
		}
		dispatch(t, s, id, keys(700, seedFor(id)), bytesFor(700))
		settle(s, id, 700, 700)
	}
	return s.Snapshot()
}

// The RULE'S EFFECT ON THE STATE MACHINE: the same script admits four members under
// gpu-plus-tier and exactly one per engine under the GPU-only default.
func TestGpuPlusTierSeatsMoreThanGPUOnly(t *testing.T) {
	for _, engines := range []int{1, 2} {
		engines := engines
		t.Run(fmt.Sprintf("N=%d", engines), func(t *testing.T) {
			tier := driveFourBigMembers(t, tierBudgetConfig(2000), engines)
			if tier.Seats != 4 {
				t.Fatalf("gpu-plus-tier: %d seats, wanted all four 700-block members (budget %d blocks per engine)",
					tier.Seats, budgetFor(0.95, 1000, 2000))
			}
			if tier.NEBlocks != 2800 {
				t.Fatalf("gpu-plus-tier: %d blocks charged, wanted 4 x 700", tier.NEBlocks)
			}
			for id, es := range tier.Engines {
				if es.NEBlocks > es.BudgetBlocks {
					t.Fatalf("engine %s is over its budget at the end of the run (%d > %d)", id, es.NEBlocks, es.BudgetBlocks)
				}
			}
			// the comparator: the SAME script on the GPU-only budget seats one member
			// per engine, and the rest wait
			gpu := driveFourBigMembers(t, defaultConfig(), engines)
			if gpu.Seats != engines {
				t.Fatalf("index: %d seats, wanted %d (one 700-block member per 950-block budget)", gpu.Seats, engines)
			}
			if gpu.Seats >= tier.Seats {
				t.Fatalf("the tier-sized budget must seat strictly more (%d vs %d)", tier.Seats, gpu.Seats)
			}
		})
	}
}

// (4) TIER = 0 ⇒ IDENTICAL TO "index". Not just the same budget: the same script lands on
// the same state, and the cost of a queued head is the same number on a populated index.
func TestGpuPlusTierZeroTierEqualsGPU(t *testing.T) {
	zero := tierBudgetConfig(0)
	if zero.CpuTierBytes != 0 || zero.tierBlocks() != 0 {
		t.Fatalf("the fixture must carry no tier, got cpuTierBytes=%d tierBlocks=%d", zero.CpuTierBytes, zero.tierBlocks())
	}
	mod, err := buildModules(zero)
	if err != nil {
		t.Fatal(err)
	}
	if got := mod.sizer.Name(); got != "gpu-plus-tier" {
		t.Fatalf("the selector must still wire gpu-plus-tier with no tier configured, got %q", got)
	}
	for _, engines := range []int{1, 2} {
		engines := engines
		t.Run(fmt.Sprintf("N=%d", engines), func(t *testing.T) {
			gpu := driveFourBigMembers(t, defaultConfig(), engines)
			degen := driveFourBigMembers(t, zero, engines)
			if gpu.BudgetBlocks != degen.BudgetBlocks {
				t.Fatalf("budget %d (index) vs %d (gpu-plus-tier, tier=0)", gpu.BudgetBlocks, degen.BudgetBlocks)
			}
			if gpu.Seats != degen.Seats || gpu.NEBlocks != degen.NEBlocks ||
				gpu.UniqueBlocks != degen.UniqueBlocks || gpu.Served != degen.Served {
				t.Fatalf("tier=0 must be the same state machine: index {seats %d ne %d unique %d served %d} vs gpu-plus-tier {seats %d ne %d unique %d served %d}",
					gpu.Seats, gpu.NEBlocks, gpu.UniqueBlocks, gpu.Served,
					degen.Seats, degen.NEBlocks, degen.UniqueBlocks, degen.Served)
			}
			for id, es := range gpu.Engines {
				other := degen.Engines[id]
				if es.BudgetBlocks != other.BudgetBlocks || es.NEBlocks != other.NEBlocks || es.Seats != other.Seats {
					t.Fatalf("engine %s differs at tier=0: budget %d/%d ne %d/%d seats %d/%d",
						id, es.BudgetBlocks, other.BudgetBlocks, es.NEBlocks, other.NEBlocks, es.Seats, other.Seats)
				}
			}
		})
	}
}

// The COST is IndexSizer's at either selector value: gpu-plus-tier sizes seats, it never
// discounts a head (the same rule hrrn-tier's tier-aware cost obeys).
func TestGpuPlusTierCostIsTheGPUOnlyCost(t *testing.T) {
	forEnginesCap(t, tierBudgetConfig(2000), 1000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 300, 3*time.Second)
		settle(fx.s, "m", 300, 300)
		fx.s.mu.Lock()
		g := fx.g()
		fx.s.mu.Unlock()
		plus := GpuPlusTier{Fraction: 0.95, BlockSize: 64}.Cost(g.index, "next", bytesFor(100))
		gpu := IndexSizer{Fraction: 0.95, BlockSize: 64}.Cost(g.index, "next", bytesFor(100))
		if plus != gpu {
			t.Fatalf("gpu-plus-tier cost %d blocks, index cost %d: the sizing value must not change what a head costs", plus, gpu)
		}
		if plus <= 0 {
			t.Fatalf("the equality must not be vacuous: a new member's head costs %d blocks", plus)
		}
	})
}
