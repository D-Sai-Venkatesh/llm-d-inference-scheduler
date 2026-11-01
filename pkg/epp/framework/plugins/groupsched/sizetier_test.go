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

// THE SIZING ALTERNATIVE "tier-inclusive" (PREREG-multipod-offload, the tier-inclusive
// correction):
//
//	BudgetBlocks = floor(groupBudgetFraction x max(num_gpu_blocks, tierBlocks))
//
// tierBlocks is the CPU tier SHADOW's own capacity, so the value adds no constant. The
// MAX is the subject: vLLM's native offload is WRITE-THROUGH at compute time (every chunk
// is stored to the tier as it is computed, nothing on GPU eviction), so the tier holds a
// copy of what HBM holds and the two capacities do NOT add — a sum prices blocks that live
// in both places at once. Properties pinned here, at N=1 AND at N=2 (forEnginesCap: the
// value is selected per ARM, and each engine's budget comes from its OWN capacity and its
// OWN tier):
//
//	(1) the budget IS the max where the tier is the larger (per engine, and the
//	    whole-policy budget is their sum);
//	(2) a tier SMALLER than the GPU is invisible: the budget stays the GPU's, because a
//	    tier that only covers HBM adds nothing to what the engine can reach;
//	(3) the DEFAULT is unchanged — same selector value, same budget, with or without a
//	    tier configured (the golden replay TestGolden runs on that config, byte for
//	    byte);
//	(4) the cold-entry valve's cap stays the GPU headroom: it staggers GPU allocation
//	    bursts, and the tier cannot absorb one;
//	(5) tier = 0 ⇒ the value is arithmetically "index": same budget, same decisions,
//	    same costs;
//	(6) the campaign deployment's arithmetic as numbers (40,183 GPU blocks, 87,381 tier
//	    blocks ⇒ 83,011, NOT the 121,185 the retired additive value gave);
//	(7) the RETIRED value ("gpu-plus-tier") fails closed.
//
// The conformance guard lives in config_conformance_test.go (the selector value the
// emitter renders) and the loader path in cmd/epp/runner (what a real arm starts through).

import (
	"fmt"
	"testing"
	"time"
)

// tierInclusiveConfig is the default config with the SIZING selector set to tier-inclusive
// and a CPU tier of capacityBlocks. The tier facts follow the index's own 4-bytes/token
// convention (bytesFor/IndexSizer), so a block means one thing everywhere.
func tierInclusiveConfig(capacityBlocks int) Config {
	c := defaultConfig()
	c.Sizing = "tier-inclusive"
	c.KvBlockBytes = 4 * 64 // 4 bytes/token x 64-token blocks (the index's convention)
	c.CpuTierBytes = int64(capacityBlocks) * c.KvBlockBytes
	return c
}

// budgetFor is the rule under test, written the way the deployment reads it.
func budgetFor(fraction float64, numBlocks, tierBlocks int) int {
	unique := numBlocks
	if tierBlocks > unique {
		unique = tierBlocks
	}
	return int(fraction * float64(unique))
}

// (1) THE BUDGET IS THE MAX, NOT THE SUM.
func TestTierInclusiveBudgetIsTheMax(t *testing.T) {
	const (
		capacity   = 1000
		tierBlocks = 2000
	)
	forEnginesCap(t, tierInclusiveConfig(tierBlocks), capacity, func(t *testing.T, fx *engineSet) {
		want := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, tierBlocks)
		gpuOnly := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, 0)
		summed := int(fx.s.cfg.GroupBudgetFraction * float64(capacity+tierBlocks))
		if want <= gpuOnly {
			t.Fatalf("the fixture must make the two budgets different: max %d vs GPU-only %d", want, gpuOnly)
		}
		if want == summed {
			t.Fatalf("the fixture must make max and sum different: max %d vs sum %d", want, summed)
		}
		snap := fx.s.Snapshot()
		for _, id := range fx.ids {
			es := snap.Engines[engineLabel(id)]
			if es.NumBlocks != capacity {
				t.Fatalf("engine %s: measured capacity %d, wanted the fixture's %d", id, es.NumBlocks, capacity)
			}
			if es.TierCapacityBlocks != tierBlocks {
				t.Fatalf("engine %s: the tier shadow's capacity must be the number the max reads, got %d (wanted %d)", id, es.TierCapacityBlocks, tierBlocks)
			}
			if es.BudgetBlocks != want {
				t.Fatalf("engine %s: budget %d, wanted floor(%v x max(%d, %d)) = %d — sizing tier-inclusive sizes the budget from the engine's UNIQUE capacity, because the write-through tier COPIES HBM and adds nothing to it",
					id, es.BudgetBlocks, fx.s.cfg.GroupBudgetFraction, capacity, tierBlocks, want)
			}
			if es.BudgetBlocks == summed {
				t.Fatalf("engine %s: budget %d is the SUM (%d) — an additive capacity the inclusive hierarchy does not have", id, es.BudgetBlocks, summed)
			}
		}
		if got, expected := snap.BudgetBlocks, want*len(fx.ids); got != expected {
			t.Fatalf("the whole-policy budget must be the per-engine sum, got %d wanted %d", got, expected)
		}
	})
}

// (2) A TIER SMALLER THAN THE GPU IS INVISIBLE. The hierarchy is inclusive in ONE
// direction: the tier holds a copy of HBM, so where the GPU is the larger of the two it is
// the whole unique capacity and the tier's blocks add nothing (a tier-evicted block is
// still on the GPU; nothing new becomes reachable).
func TestTierInclusiveSmallerTierIsTheGPU(t *testing.T) {
	const (
		capacity   = 3000
		tierBlocks = 500
	)
	forEnginesCap(t, tierInclusiveConfig(tierBlocks), capacity, func(t *testing.T, fx *engineSet) {
		want := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, 0) // floor(0.95 x 3000) = 2850
		if got := budgetFor(fx.s.cfg.GroupBudgetFraction, capacity, tierBlocks); got != want {
			t.Fatalf("the fixture must make the tier the SMALLER capacity: max %d vs tier %d", want, got)
		}
		snap := fx.s.Snapshot()
		for _, id := range fx.ids {
			es := snap.Engines[engineLabel(id)]
			if es.TierCapacityBlocks != tierBlocks {
				t.Fatalf("engine %s: the tier shadow must still be sized (%d blocks) — it is what the max reads", id, es.TierCapacityBlocks)
			}
			if es.BudgetBlocks != want {
				t.Fatalf("engine %s: budget %d, wanted the GPU's %d — a tier under the GPU adds no reachable capacity", id, es.BudgetBlocks, want)
			}
		}
	})
}

// (3) THE DEFAULT IS UNCHANGED. The selector value the arms render ("index") keeps the
// GPU-only budget even when the deployment carries a tier — which is the shape of
// core-q4-offload.env (tier facts set, default sizing) and of the golden replay.
func TestTierInclusiveDefaultIsUnchanged(t *testing.T) {
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
			// the valve's cap is the GPU headroom in both cases (property (4) below)
			if want := int64(50) * 64; es.ColdCapTokens != want {
				t.Fatalf("engine %s: cold cap %d, wanted the GPU headroom %d", id, es.ColdCapTokens, want)
			}
		}
		if got := snap.BudgetBlocks; got != 950*len(fx.ids) {
			t.Fatalf("whole-policy budget %d, wanted %d", got, 950*len(fx.ids))
		}
	})
}

// (4) THE VALVE CAP IS UNCHANGED BY THE TIER. Same capacity, same fraction, two configs:
// the budget grows (the tier is the larger capacity) and the cap does not move, because a
// cold prefill allocates HBM whatever the tier holds.
func TestTierInclusiveValveCapStaysGPUHeadroom(t *testing.T) {
	const (
		capacity   = 1000
		tierBlocks = 2000
	)
	// the GPU headroom in tokens: floor((1 - 0.95) x 1000) x 64 = 3200
	const wantCap = int64(50) * 64
	caps := map[string]int64{}
	for _, name := range []string{"tier-inclusive", "index"} {
		cfg := defaultConfig()
		if name == "tier-inclusive" {
			cfg = tierInclusiveConfig(tierBlocks)
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
			if name == "tier-inclusive" {
				wantBudget = budgetFor(cfg.GroupBudgetFraction, capacity, tierBlocks)
			}
			if es.BudgetBlocks != wantBudget {
				t.Fatalf("%s: budget %d, wanted %d", name, es.BudgetBlocks, wantBudget)
			}
		})
	}
	if caps["tier-inclusive"] != caps["index"] {
		t.Fatalf("the tier must not move the valve: cap %d with the tier vs %d without", caps["tier-inclusive"], caps["index"])
	}
}

// driveFourBigMembers seats four 700-block members on a fresh scheduler of `engines`
// engines and returns the end state. One cold entry is admitted at a time (the derived
// valve cap), so each turn settles before the next head arrives, and members hold their
// seats afterwards. Under GPU-only sizing one 700-block member per 950-block budget fits;
// under tier-inclusive two do (2 x 700 = 1400 <= 1900 = max(1000, 2000) x 0.95, and a third
// would not) — the tier is the larger capacity here, not an addition to it.
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

// The RULE'S EFFECT ON THE STATE MACHINE: the same script admits TWO members per engine
// under tier-inclusive (1400 of the 1900-block budget) and exactly ONE per engine under the
// GPU-only default.
func TestTierInclusiveSeatsMoreThanGPUOnly(t *testing.T) {
	for _, engines := range []int{1, 2} {
		engines := engines
		t.Run(fmt.Sprintf("N=%d", engines), func(t *testing.T) {
			tier := driveFourBigMembers(t, tierInclusiveConfig(2000), engines)
			want := 2 * engines // 2 x 700 fits 0.95 x 2000 = 1900; a third 700 does not
			if tier.Seats != want {
				t.Fatalf("tier-inclusive: %d seats, wanted %d (budget %d blocks per engine, 3 x 700 does not fit)",
					tier.Seats, want, budgetFor(0.95, 1000, 2000))
			}
			if tier.NEBlocks != 700*want {
				t.Fatalf("tier-inclusive: %d blocks charged, wanted %d x 700", tier.NEBlocks, want)
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

// (5) TIER = 0 ⇒ IDENTICAL TO "index". Not just the same budget: the same script lands on
// the same state, and the cost of a queued head is the same number on a populated index.
func TestTierInclusiveZeroTierEqualsGPU(t *testing.T) {
	zero := tierInclusiveConfig(0)
	if zero.CpuTierBytes != 0 || zero.tierBlocks() != 0 {
		t.Fatalf("the fixture must carry no tier, got cpuTierBytes=%d tierBlocks=%d", zero.CpuTierBytes, zero.tierBlocks())
	}
	mod, err := buildModules(zero)
	if err != nil {
		t.Fatal(err)
	}
	if got := mod.sizer.Name(); got != "tier-inclusive" {
		t.Fatalf("the selector must still wire tier-inclusive with no tier configured, got %q", got)
	}
	for _, engines := range []int{1, 2} {
		engines := engines
		t.Run(fmt.Sprintf("N=%d", engines), func(t *testing.T) {
			gpu := driveFourBigMembers(t, defaultConfig(), engines)
			degen := driveFourBigMembers(t, zero, engines)
			if gpu.BudgetBlocks != degen.BudgetBlocks {
				t.Fatalf("budget %d (index) vs %d (tier-inclusive, tier=0)", gpu.BudgetBlocks, degen.BudgetBlocks)
			}
			if gpu.Seats != degen.Seats || gpu.NEBlocks != degen.NEBlocks ||
				gpu.UniqueBlocks != degen.UniqueBlocks || gpu.Served != degen.Served {
				t.Fatalf("tier=0 must be the same state machine: index {seats %d ne %d unique %d served %d} vs tier-inclusive {seats %d ne %d unique %d served %d}",
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

// The COST is IndexSizer's at either selector value: tier-inclusive sizes seats, it never
// discounts a head (the same rule hrrn-tier's tier-aware cost obeys).
func TestTierInclusiveCostIsTheGPUOnlyCost(t *testing.T) {
	forEnginesCap(t, tierInclusiveConfig(2000), 1000, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 300, 3*time.Second)
		settle(fx.s, "m", 300, 300)
		fx.s.mu.Lock()
		g := fx.g()
		fx.s.mu.Unlock()
		tier := TierInclusive{Fraction: 0.95, BlockSize: 64}.Cost(g.index, "next", bytesFor(100))
		gpu := IndexSizer{Fraction: 0.95, BlockSize: 64}.Cost(g.index, "next", bytesFor(100))
		if tier != gpu {
			t.Fatalf("tier-inclusive cost %d blocks, index cost %d: the sizing value must not change what a head costs", tier, gpu)
		}
		if tier <= 0 {
			t.Fatalf("the equality must not be vacuous: a new member's head costs %d blocks", tier)
		}
	})
}

// (6) THE CAMPAIGN DEPLOYMENT'S ARITHMETIC, as numbers: 40,183 measured GPU blocks and a
// 512 GiB tier of 6,291,456-byte blocks. The tier is 2.17x the GPU pool, so it IS the
// unique capacity: the budget is 83,011 blocks, not the 121,185 the retired additive value
// produced (46 % of which were blocks that live in both places at once).
func TestTierInclusiveCampaignArithmetic(t *testing.T) {
	const (
		gpuBlocks  = 40183
		tierBytes  = 549755813888 // DECODE_TIER_BYTES / CPU_TIER_BYTES (512 GiB)
		blockBytes = 6291456      // KV_BLOCK_BYTES: one 64-token block of Qwen3-30B-A3B at TP4
	)
	tier := tierBlocksFor(tierBytes, blockBytes)
	if tier != 87381 {
		t.Fatalf("the tier's capacity in blocks is %d, wanted 87,381 (the number every config header quotes)", tier)
	}
	inclusive := TierInclusive{Fraction: 0.95, BlockSize: 64}.BudgetBlocks(gpuBlocks, tier)
	if inclusive != 83011 {
		t.Fatalf("budget %d, wanted floor(0.95 x 87,381) = 83,011 — the tier is the larger capacity, so the budget is a fraction of the TIER", inclusive)
	}
	gpuOnly := IndexSizer{Fraction: 0.95, BlockSize: 64}.BudgetBlocks(gpuBlocks, tier)
	if gpuOnly != 38173 {
		t.Fatalf("the GPU-only budget is %d, wanted floor(0.95 x 40,183) = 38,173", gpuOnly)
	}
	summed := int(0.95 * float64(gpuBlocks+tier))
	if summed != 121185 {
		t.Fatalf("the retired additive value was %d, wanted 121,185 (the number batch 2 ran)", summed)
	}
	over := (float64(summed) - float64(inclusive)) / float64(inclusive)
	if over < 0.45 || over > 0.47 {
		t.Fatalf("the additive value over-seats by %.1f %%, wanted ~46 %% — the size of the correction", 100*over)
	}
}

// (7) THE RETIRED VALUE FAILS CLOSED. "gpu-plus-tier" sized the budget from the SUM of the
// two capacities — an additive working set the write-through hierarchy does not have, and
// 46 % of it on the campaign deployment was blocks that live in both places at once. Batch
// 2 ran it; HEAD does not offer it, so a copied arm that still names it must fail at
// buildModules instead of silently measuring a policy nobody chose. The archived proof is
// the batch-2 measurement plus the config it ran (docs/gsch-core/README.md §4f, commit
// 17a10e8c / digest groupsched-17a10e8c).
func TestGpuPlusTierValueIsRetired(t *testing.T) {
	cfg := defaultConfig()
	cfg.Sizing = "gpu-plus-tier"
	if _, err := buildModules(cfg); err == nil {
		t.Fatal(`sizing "gpu-plus-tier" must be REJECTED: the additive capacity is retired (the tier is inclusive), and an arm still carrying the value has to fail closed at startup rather than run something else`)
	}
	cfg.Sizing = "tier-inclusive"
	if _, err := buildModules(cfg); err != nil {
		t.Fatalf(`sizing "tier-inclusive" is the shipped value and must be accepted: %v`, err)
	}
}
