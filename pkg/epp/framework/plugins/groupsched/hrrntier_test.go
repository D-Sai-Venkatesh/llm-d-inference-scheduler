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

// THE QUEUE MODULE WITH THE CPU TIER (`entryOrder: "hrrn-tier"`, F5′ + the tier shadow)
// and the SHADOW-VALIDATION metric that has to prove the shadow is worth believing
// (PREREG-multipod-offload, "THE OFFLOAD SOLUTION" pieces 2 and 3).
//
// The claims under test, in the order the spec states them:
//
//	(1) a WARM returner (its blocks still restorable from the engine's CPU tier) beats a
//	    COLD returner at an equal wait;
//	(2) the COLD returner still OVERTAKES after a bounded wait — the bound is the COST
//	    RATIO times the warm wait, not starvation;
//	(3) the newcomer OUTER key is unchanged (a returner never jumps a newcomer);
//	(4) TIER OFF ⇒ hrrn-tier ≡ F5′ — the equivalence replay against the archive F5′
//	    golden, byte for byte;
//	(5) the tier-aware cost is a PRIORITY input ONLY: the seat budget and the cold valve
//	    keep the GPU-only cost (offload adds no seats);
//	(6) the validation pair (prediction at seat, measured at settle) is produced,
//	    joinable by fairness id, and exported in a form an AUC can be computed from.
//
// Every behavioural test runs at N=1 AND N=2 through forEngines: the config value is
// selected per ARM, never per topology.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// ---- fixtures ----

// tierConfig is the default config with the CPU tier SWITCHED ON and the queue module
// set to hrrn-tier. capacityBlocks is the shadow's capacity, expressed through the two
// deployment facts the plugin takes (cpuTierBytes ÷ kvBlockBytes) at the index's own
// 4-bytes/token convention, so bytesFor() and the shadow agree on what a block is.
func tierConfig(capacityBlocks int) Config {
	c := defaultConfig()
	c.EntryOrder = "hrrn-tier"
	c.KvBlockBytes = 4 * 64 // 4 bytes/token x 64-token blocks (the index's convention)
	c.CpuTierBytes = int64(capacityBlocks) * c.KvBlockBytes
	return c
}

// tierConfigNoTier is the SAME selector with no tier configured — the degenerate arm the
// equivalence test pins.
func tierConfigNoTier() Config {
	c := defaultConfig()
	c.EntryOrder = "hrrn-tier"
	return c
}

// tierSpec is one waiter of a tier-aware scenario. `cold` is F5′'s denominator (tokens
// resident in NEITHER the GPU index), `tier` is hrrn-tier's (tokens resident in neither
// the index NOR the tier) — so `cold - tier` is exactly the part the CPU copy saves.
type tierSpec struct {
	id       string
	wait     time.Duration
	cold     int64
	tier     int64
	newcomer bool
}

// tierQueue installs a waiter set with BOTH costs, as the scheduler's sync pass fills
// them (tierCostSet = true, exactly what setWaiterCostsLocked writes).
func (fx *engineSet) tierQueue(specs ...tierSpec) {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	fx.s.waiters = make(map[string]*waiter, len(specs))
	for _, sp := range specs {
		fx.s.waiters[sp.id] = &waiter{
			id: sp.id, newcomer: sp.newcomer, since: fx.clock.Add(-sp.wait),
			coldTokens: sp.cold, tierCostTokens: sp.tier, tierCostSet: true,
		}
	}
}

// tierPriorityOf is one waiter's hrrn-tier priority in the CURRENT pass.
func (fx *engineSet) tierPriorityOf(id string) float64 {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	h := fx.s.mod.entry.(HrrnTier)
	bound := lowerMedianTokens(fx.s.waiters, h.BlockSize, tierCostOf)
	return h.priority(fx.s.waiters[id], bound, fx.clock)
}

// ---- (1) warm before cold at an equal wait ----

// TestHrrnTierWarmReturnBeatsColdAtEqualWait is the rule's whole point. Three returners
// have waited exactly as long; ONE still has its history in the CPU tier (it left a
// minute ago, the tier still holds its copy) and the other two are cold. The warm one is
// the head.
//
// The POPULATION is not incidental (the same caveat F5′'s own suite documents): the cap
// binds at or above the LOWER median, so a queue of equal-sized waiters cannot express a
// size preference at all. Here the median is a COLD waiter's size, which leaves the warm
// waiter's small cost intact — and that is the realistic shape: most returns are past the
// tier's cliff (the measured 30 %/14 %/6.4 % bins), a few are inside it.
func TestHrrnTierWarmReturnBeatsColdAtEqualWait(t *testing.T) {
	forEngines(t, tierConfig(2000), func(t *testing.T, fx *engineSet) {
		const (
			wait     = 60 * time.Second
			fullCold = int64(200_000) // a 200K-token history with no copy anywhere
			newTurn  = int64(6_400)   // its history is restorable: only this turn is cold
		)
		fx.tierQueue(
			tierSpec{id: "cold-a", wait: wait, cold: fullCold, tier: fullCold},
			tierSpec{id: "cold-b", wait: wait, cold: fullCold, tier: fullCold},
			tierSpec{id: "warm", wait: wait, cold: fullCold, tier: newTurn},
		)
		// the two denominators, side by side: the warm returner's s is 31x smaller, so at
		// an equal wait its ratio is 31x larger.
		warmP, coldP := fx.tierPriorityOf("warm"), fx.tierPriorityOf("cold-a")
		if !(warmP > coldP) {
			t.Fatalf("at an equal wait the WARM returner (%v) must outrank the cold one (%v): a live CPU copy is a deadline", warmP, coldP)
		}
		if got := fx.headOf(); got == nil || got.id != "warm" {
			t.Fatalf("head = %v, want warm", got)
		}
		// and the mechanism is the TIER-AWARE cost, not a second rule: with the same
		// population under F5′ (no tier) the three tie at an equal wait and the id decides.
		fx.s.mu.Lock()
		for _, w := range fx.s.waiters {
			w.tierCostSet = false // F5′'s fallback: the cold cost
		}
		fx.s.mu.Unlock()
		if got := fx.headOf(); got == nil || got.id != "cold-a" {
			t.Fatalf("without the tier the same set must fall back to F5′'s ordering (cold-a, oldest-then-id), got %v", got)
		}
	})
}

// ---- (2) the cold returner still overtakes, at the cost ratio ----

// TestHrrnTierColdReturnerOvertakesAtTheCostRatio pins the no-starvation half AND its
// arithmetic. The warm waiter's ratio is wait/6,400 and the cold one's is wait/200,000, so
// the cold returner becomes the head exactly when its wait exceeds the COST RATIO
// (200,000/6,400 = 31.25x) times the warm waiter's wait — a bound derived from the queue's
// own sizes, and 31x SHORTER than plain FIFO-among-returners would make a cold one wait
// for the warm class to drain.
func TestHrrnTierColdReturnerOvertakesAtTheCostRatio(t *testing.T) {
	forEngines(t, tierConfig(2000), func(t *testing.T, fx *engineSet) {
		const (
			fullCold = int64(200_000)
			newTurn  = int64(6_400)
			warmWait = 10 * time.Second
			// crossing: wait_cold > warmWait x (fullCold / newTurn)
			crossing = time.Duration(float64(warmWait) * float64(fullCold) / float64(newTurn)) // 312.5s
		)
		base := []tierSpec{
			{id: "cold-a", wait: 0, cold: fullCold, tier: fullCold},
			{id: "cold-b", wait: 0, cold: fullCold, tier: fullCold},
			{id: "warm", wait: warmWait, cold: fullCold, tier: newTurn},
		}
		// (a) below the crossing the warm returner keeps the head: the cold one has NOT
		// aged enough yet, and it must not be able to buy its way in with size (the cap).
		set := append([]tierSpec{{id: "cold-young", wait: crossing - time.Second, cold: fullCold, tier: fullCold}}, base...)
		fx.tierQueue(set...)
		if got := fx.headOf(); got == nil || got.id != "warm" {
			t.Fatalf("head = %v at %v, want warm (the crossing is %v)", got, crossing-time.Second, crossing)
		}
		// (b) one instant past it, the COLD returner is the head — it aged past the warm
		// class's ratio, which is the whole no-starvation argument.
		set = append([]tierSpec{{id: "cold-old", wait: crossing + time.Second, cold: fullCold, tier: fullCold}}, base...)
		fx.tierQueue(set...)
		if got := fx.headOf(); got == nil || got.id != "cold-old" {
			t.Fatalf("head = %v at %v, want cold-old: a cold returner must overtake once its wait buys the ratio", got, crossing+time.Second)
		}
		// (c) a fully COLD returner is priced exactly as F5′ prices it: the tier can only
		// ever LOWER a denominator, so the cold class is never slower than it was.
		if got, want := fx.tierPriorityOf("cold-old"), (crossing+time.Second).Seconds()/float64(fullCold); got != want {
			t.Fatalf("a cold returner's priority = %v, want F5′'s own rate %v", got, want)
		}
	})
}

// ---- (3) the newcomer outer key ----

// TestHrrnTierKeepsTheNewcomerOuterKey: the tier can make a returning member CHEAP, and a
// cheap member with a long wait could otherwise outrank everything. It must not: the
// archive's outer partition is the newcomer CLASS, and it is compared before any ratio, so
// a warm returner — however cheap and however long it has waited — stays behind a
// newcomer. This is the property that keeps the arm's newcomer TTFT gate meaningful.
func TestHrrnTierKeepsTheNewcomerOuterKey(t *testing.T) {
	forEngines(t, tierConfig(2000), func(t *testing.T, fx *engineSet) {
		fx.tierQueue(
			tierSpec{id: "warm-returner", wait: 3600 * time.Second, cold: 200_000, tier: 64},
			tierSpec{id: "cold-returner", wait: 3600 * time.Second, cold: 200_000, tier: 200_000},
			// a newcomer with a full cold prefill ahead of it and ONE SECOND of waiting
			tierSpec{id: "newcomer", wait: time.Second, cold: 200_000, tier: 200_000, newcomer: true},
		)
		if got := fx.headOf(); got == nil || got.id != "newcomer" {
			t.Fatalf("head = %v, want the newcomer: the class is the OUTER key, so no tier-saving returner may jump it", got)
		}
		// INSIDE the newcomer class the ratio still rules (the tier changes nothing there
		// except the honest cost): a small newcomer beats a big one at an equal wait.
		fx.tierQueue(
			tierSpec{id: "small", wait: 10 * time.Second, cold: 3_000, tier: 3_000, newcomer: true},
			tierSpec{id: "med", wait: 10 * time.Second, cold: 6_000, tier: 6_000, newcomer: true},
			tierSpec{id: "med2", wait: 10 * time.Second, cold: 6_000, tier: 6_000, newcomer: true},
			tierSpec{id: "big", wait: 10 * time.Second, cold: 200_000, tier: 200_000, newcomer: true},
		)
		if got := fx.headOf(); got == nil || got.id != "small" {
			t.Fatalf("head = %v, want small (the small-first ordering inside the class is F5′'s)", got)
		}
	})
}

// ---- (4) TIER OFF ⇒ hrrn-tier ≡ F5′ ----

// TestHrrnTierNoTierEqualsF5P is the DEGENERACY golden: with no CPU tier configured
// (cpuTierBytes 0), `entryOrder: "hrrn-tier"` must reproduce the ARCHIVE F5′ trace BYTE
// FOR BYTE — no normalisation, no "close enough". The mechanism is asserted at its
// source: setWaiterCostsLocked writes the tier-aware cost EQUAL to the cold cost when no
// engine has a tier, so the arithmetic is literally F5′'s.
//
// The golden is the same file TestHrrnBoundedEqualsTheArchiveF5P uses, so the two values
// are pinned against ONE archive trace: a divergence here means the new value changed
// something it promised not to.
func TestHrrnTierNoTierEqualsF5P(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "golden-f5p-archive.txt"))
	if err != nil {
		t.Fatalf("F5′ equivalence golden missing: %v", err)
	}
	got := runGoldenScenarioCfg(t, 2600, tierConfigNoTier())
	if got == string(want) {
		return
	}
	gotL := strings.Split(got, "\n")
	wantL := strings.Split(string(want), "\n")
	diff := 0
	for i := 0; i < len(gotL) || i < len(wantL); i++ {
		var a, b string
		if i < len(gotL) {
			a = gotL[i]
		}
		if i < len(wantL) {
			b = wantL[i]
		}
		if a == b {
			continue
		}
		diff++
		if diff <= 5 {
			t.Errorf("divergence at line %d:\n  hrrn-tier (no tier): %s\n  archive F5′:       %s", i+1, a, b)
		}
	}
	t.Fatalf("hrrn-tier with NO tier configured differs from the archive F5′ in %d lines: the degenerate case must BE F5′, not resemble it", diff)
}

// TestHrrnTierConfiguredTraceIsPinned is the arm's OWN golden: the same 2,600-second replay
// with a CPU tier configured (2,000 blocks per engine) and `entryOrder: hrrn-tier`. It is
// pinned as a new trace (nothing existing moves), and the test also asserts the trace is
// DIFFERENT from F5′'s — a tier-aware module that produced F5′'s decisions would mean the
// tier never reached the decision path.
func TestHrrnTierConfiguredTraceIsPinned(t *testing.T) {
	// the replay on a scheduler we keep, so the arm's own state can be asserted after it:
	// the trace says WHAT was decided, the snapshot says the tier pieces were actually live.
	s, err := NewScheduler(tierConfig(2000))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	got := runGoldenScenarioOn(t, s)
	path := filepath.Join("testdata", "golden-hrrn-tier.txt")
	if *updateGolden {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
		t.Logf("wrote %s (%d lines)", path, strings.Count(got, "\n"))
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("hrrn-tier golden missing: %v (regenerate with -update-golden)", err)
	}
	if got != string(want) {
		gotL := strings.Split(got, "\n")
		wantL := strings.Split(string(want), "\n")
		for i := 0; i < len(gotL) || i < len(wantL); i++ {
			var a, b string
			if i < len(gotL) {
				a = gotL[i]
			}
			if i < len(wantL) {
				b = wantL[i]
			}
			if a != b {
				t.Fatalf("hrrn-tier trace diverges at line %d:\n  now:    %s\n  golden: %s", i+1, a, b)
			}
		}
		t.Fatal("hrrn-tier trace diverges (length)")
	}
	f5p, err := os.ReadFile(filepath.Join("testdata", "golden-f5p-archive.txt"))
	if err != nil {
		t.Fatalf("F5′ golden missing: %v", err)
	}
	if got == string(f5p) {
		t.Fatal("the tier-configured arm reproduced the F5′ trace exactly: the tier shadow is not on the decision path")
	}
	// COVERAGE, not a value: the arm must actually FILL its tier and actually produce
	// validation pairs. A run whose shadow stays empty, or whose metric never fires, looks
	// identical in the trace and would make the live AUC gate unanswerable.
	snap := s.Snapshot()
	es := snap.Engines["engine-a"]
	if es.TierCapacityBlocks != 2000 {
		t.Fatalf("tier capacity = %d blocks, want the configured 2000", es.TierCapacityBlocks)
	}
	if es.TierShadowBlocks != 2000 {
		t.Fatalf("the tier shadow holds %d blocks, want it pinned at capacity 2000 (the replay stores far more than that)", es.TierShadowBlocks)
	}
	if snap.ShadowPairs < 50 {
		t.Fatalf("the replay produced %d shadow-validation pairs, want the seated returns' worth (>= 50): the prediction/measurement pairing is not firing", snap.ShadowPairs)
	}
	// determinism: the whole replay, run again, is identical (no map order, no wall clock).
	for i := 0; i < 2; i++ {
		if again := runGoldenScenarioCfg(t, 2600, tierConfig(2000)); again != got {
			t.Fatalf("the hrrn-tier replay is not deterministic (run %d differs)", i+2)
		}
	}
}

// ---- (5) the tier-aware cost is a PRIORITY input only ----

// TestHrrnTierDoesNotDiscountTheSeatBudget is the invariant that keeps the arm honest:
// a returner whose history is entirely restorable costs ~nothing IN THE QUEUE, and must
// still pay its FULL GPU footprint against the seat budget. Offload adds no seats — vLLM
// runs only from HBM — so an admission priced off the tier-aware cost would overbook the
// engine exactly when the restores land.
func TestHrrnTierDoesNotDiscountTheSeatBudget(t *testing.T) {
	forEngines(t, tierConfig(400), func(t *testing.T, fx *engineSet) {
		// a member whose 100 blocks are stored in its engine's tier
		fx.seatOne(t, "m", 100, time.Second)
		g := fx.s.groups[fx.s.bindings["m"]]
		settle(fx.s, "m", 100, 100)
		chain := keys(100, seedFor("m"))
		if !g.tier.Holds(chain[0]) {
			t.Fatal("premise: the member's blocks must be in the tier")
		}
		// it leaves and comes back with the same body (delta = 0, so F5′'s cost is the
		// 100 blocks the engine does not hold while it is unseated)
		m := g.members["m"]
		fx.s.mu.Lock()
		fx.s.exitLocked(g, m, "test", fx.clock)
		w := &waiter{id: "m", since: fx.clock.Add(-time.Second), sizeBytes: bytesFor(100),
			fairnessID: "fb-m"}
		fx.s.setWaiterCostsLocked(w)
		cold, tier := w.coldTokens, w.tierCostTokens
		cand := fx.s.seatEngineLocked(w, fx.clock)
		fx.s.mu.Unlock()
		if cold != w.coldTokens || cold != int64(100*64) {
			t.Fatalf("the GPU-only cost = %d tokens, want the member's 100 blocks = %d tokens", cold, 100*64)
		}
		if tier != 0 {
			t.Fatalf("the tier-aware cost = %d tokens, want 0 (every block is restorable from the tier)", tier)
		}
		if cand == nil {
			t.Fatal("the head must have a placement candidate")
		}
		if cand.costTokens != cold {
			t.Fatalf("the SEAT chose on %d tokens but the GPU-only cost is %d: the admission must never be discounted by the CPU tier", cand.costTokens, cold)
		}
		if len(fx.s.waiters) != 0 {
			t.Fatal("premise: no other waiter")
		}
	})
}

// ---- (6) the shadow-validation metric ----

// TestTierShadowValidationPairsPredictionWithMeasurement is piece 3 end to end: a returning
// member is seated (prediction taken from the index + the tier shadow), the turn that seat
// was for settles (the engine's measured cached/prompt), and the pair is emitted with the
// fairness id that joins it to the wire-level request — in BOTH exports: the event stream
// (the durable, per-id record) and the metrics matrix (the AUC's input).
func TestTierShadowValidationPairsPredictionWithMeasurement(t *testing.T) {
	forEngines(t, tierConfig(400), func(t *testing.T, fx *engineSet) {
		type pair struct {
			engine              string
			predicted, measured float64
		}
		var pairs []pair
		fx.s.onShadowValidation = func(engine string, predicted, measured float64) {
			pairs = append(pairs, pair{engine, predicted, measured})
		}
		// turn 1 stores the member's 100 blocks in its engine's tier
		fx.seatOne(t, "m", 100, time.Second)
		g := fx.s.groups[fx.s.bindings["m"]]
		settle(fx.s, "m", 100, 100)
		if g.tier.Len() != 100 {
			t.Fatalf("premise: the tier holds %d blocks, want 100", g.tier.Len())
		}
		// it leaves; its blocks are NOT in the engine's index any more (unseated) but they
		// ARE still in the tier — exactly the state the prediction exists for.
		fx.s.mu.Lock()
		fx.s.exitLocked(g, g.members["m"], "test", fx.clock)
		fx.s.mu.Unlock()
		if len(pairs) != 0 {
			t.Fatal("a newcomer's first settlement must not produce a validation pair (the metric is defined over seated RETURNS)")
		}

		// it comes back, same body: the seat takes the prediction...
		h := Head{Member: "m", ReqID: "r-m-2", Arrival: fx.clock.Add(-30 * time.Second),
			SizeBytes: bytesFor(100), FairnessID: "fb-m"}
		if i := fx.pick(h); i < 0 {
			t.Fatal("the returning member must be seated")
		}
		m, gg := fx.member(t, "m")
		if !m.predSet {
			t.Fatal("a seated RETURN must carry a shadow-validation prediction")
		}
		if m.predResident != 1.0 {
			t.Fatalf("predicted resident fraction = %v, want 1.0 (every block of the returning body is in the tier)", m.predResident)
		}
		// ...and the turn that seat was for settles with the engine's measured hit.
		dispatch(t, fx.s, "m", keys(100, seedFor("m")), bytesFor(100))
		settle(fx.s, "m", 100, 90)
		if len(pairs) != 1 {
			t.Fatalf("got %d validation pairs, want exactly 1", len(pairs))
		}
		if pairs[0].predicted != 1.0 || pairs[0].measured != 0.9 {
			t.Fatalf("pair = (pred %v, measured %v), want (1.0, 0.9)", pairs[0].predicted, pairs[0].measured)
		}
		if pairs[0].engine != engineLabel(gg.id) {
			t.Fatalf("pair attributed to engine %q, want %q", pairs[0].engine, engineLabel(gg.id))
		}
		// one prediction, one measurement: the second settle of the same residency has
		// nothing left to pair.
		dispatch(t, fx.s, "m", keys(100, seedFor("m")), bytesFor(100))
		settle(fx.s, "m", 100, 80)
		if len(pairs) != 1 {
			t.Fatalf("got %d pairs after a second settle, want 1 (a prediction must not be reused)", len(pairs))
		}
		// the durable record: an event carrying BOTH numbers and the fairness id, so the
		// AUC is computable from the run's own log.
		var found bool
		for _, ev := range fx.s.Events() {
			if ev.Kind != "SHADOW-VALIDATION" {
				continue
			}
			found = true
			for _, want := range []string{"pred=1.0000", "measured=0.9000", "fairness_id=fb-m", "prompt=100", "cached=90"} {
				if !strings.Contains(ev.Detail, want) {
					t.Errorf("SHADOW-VALIDATION detail %q is missing %q", ev.Detail, want)
				}
			}
			if ev.Member != "m" || ev.Engine != engineLabel(gg.id) {
				t.Errorf("SHADOW-VALIDATION event attributed to member=%q engine=%q", ev.Member, ev.Engine)
			}
		}
		if !found {
			t.Fatal("no SHADOW-VALIDATION event was emitted: the analysis cannot join prediction to measurement")
		}
		if snap := fx.s.Snapshot(); snap.ShadowPairs != 1 {
			t.Fatalf("Snapshot.ShadowPairs = %d, want 1", snap.ShadowPairs)
		}
		if got := g.tier.Len(); got != 100 {
			t.Fatalf("the tier holds %d blocks, want 100", got)
		}
	})
}

// TestShadowValidationMatrixIsScrapable asserts the WIRE FORM of the AUC input: the
// (predicted, measured) pair lands in the 11x11 binned matrix, so an analysis — or a
// scrape — can compute the ranking without the event log. A counter nobody can gather is
// not observability.
func TestShadowValidationMatrixIsScrapable(t *testing.T) {
	reg := prometheus.NewRegistry()
	gm := newGroupMetrics(nil)
	if err := gm.register(reg); err != nil {
		t.Fatalf("register: %v", err)
	}
	gm.ObserveShadowValidation("e0", 0.92, 0.51) // bins 9 and 5
	gm.ObserveShadowValidation("e0", 0.92, 0.02) // bins 9 and 0
	gm.ObserveShadowValidation("e1", 1.0, 1.3)   // saturating bins: 10 and 10
	v, ok := gatherValue(t, reg, "llm_d_epp_group_tier_shadow_pairs_total",
		map[string]string{"engine": "e0", "pred_bin": "9", "meas_bin": "5"})
	if !ok || v != 1 {
		t.Fatalf("pair (0.92, 0.51) on e0 = %v (present=%v), want 1", v, ok)
	}
	if v, ok := gatherValue(t, reg, "llm_d_epp_group_tier_shadow_pairs_total",
		map[string]string{"engine": "e0", "pred_bin": "9", "meas_bin": "0"}); !ok || v != 1 {
		t.Fatalf("pair (0.92, 0.02) on e0 = %v (present=%v), want 1", v, ok)
	}
	if v, ok := gatherValue(t, reg, "llm_d_epp_group_tier_shadow_pairs_total",
		map[string]string{"engine": "e1", "pred_bin": "10", "meas_bin": "10"}); !ok || v != 1 {
		t.Fatalf("out-of-range fractions must saturate into bin 10, got %v (present=%v)", v, ok)
	}
	for f, want := range map[float64]string{-0.1: "0", 0: "0", 0.05: "0", 0.5: "5", 0.99: "9", 1: "10", 2: "10"} {
		if got := fracBin(f); got != want {
			t.Errorf("fracBin(%v) = %q, want %q", f, got, want)
		}
	}
}

// TestTierConfiguredIsShadowNeutralForTheOtherArms pins the other half of the degeneracy:
// a deployment that HAS a CPU tier, running a queue module that does not price with it,
// makes exactly the same decisions as a deployment with no tier at all. That is what makes
// the wave's OFF/ON pairs readable — core-q4 vs core-q4-offload differ in the DEPLOYMENT,
// and any decision difference between them is the engine's behaviour, not a shadow that
// leaked into the policy.
//
// It is asserted against the two existing goldens (the default arm's and the archive F5′'s),
// so no new trace is introduced for a claim that is really "nothing changed".
func TestTierConfiguredIsShadowNeutralForTheOtherArms(t *testing.T) {
	tierFacts := func(entry string) Config {
		c := defaultConfig()
		c.EntryOrder = entry
		c.KvBlockBytes = 4 * 64
		c.CpuTierBytes = 2000 * c.KvBlockBytes
		return c
	}
	cases := []struct {
		name, entry, golden string
	}{
		{"newcomer-first (the anchor arms)", "newcomer-first", "golden-f0p-bugfixes.txt"},
		{"hrrn-bounded (F5′, the tier-blind comparator)", "hrrn-bounded", "golden-f5p-archive.txt"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want, err := os.ReadFile(filepath.Join("testdata", tc.golden))
			if err != nil {
				t.Fatalf("golden missing: %v", err)
			}
			cfg := tierFacts(tc.entry)
			if cfg.tierBlocks() == 0 {
				t.Fatal("premise: the config must actually carry a tier")
			}
			got := runGoldenScenarioCfg(t, 2600, cfg)
			if got == string(want) {
				return
			}
			gotL := strings.Split(got, "\n")
			wantL := strings.Split(string(want), "\n")
			for i := 0; i < len(gotL) || i < len(wantL); i++ {
				var a, b string
				if i < len(gotL) {
					a = gotL[i]
				}
				if i < len(wantL) {
					b = wantL[i]
				}
				if a != b {
					t.Fatalf("a configured CPU tier changed a decision at line %d under entryOrder=%s:\n  with tier: %s\n  golden:    %s\nThe tier must be invisible to every other module (it is a PRIORITY input for hrrn-tier only).",
						i+1, tc.entry, a, b)
				}
			}
			t.Fatalf("trace length differs under entryOrder=%s with a tier configured", tc.entry)
		})
	}
}
