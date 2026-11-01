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

// THE TIER SHADOW'S VALIDATION (PREREG-multipod-offload, "THE OFFLOAD SOLUTION" piece 3):
// the prediction taken at seat and the measurement taken at the turn's settle must be
// produced, joinable by fairness id, and exported in a form an AUC can be computed from.
//
// (The hrrn-tier queue module that this file was created with has been removed with the
// parked queue implementations; the shadow and its validation remain until the trim
// removes them too.)

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// tierConfig is the default config with the CPU tier SWITCHED ON. capacityBlocks is the
// shadow's capacity, expressed through the two deployment facts the plugin takes
// (cpuTierBytes ÷ kvBlockBytes) at the index's own 4-bytes/token convention, so bytesFor()
// and the shadow agree on what a block is.
func tierConfig(capacityBlocks int) Config {
	c := defaultConfig()
	c.KvBlockBytes = 4 * 64 // 4 bytes/token x 64-token blocks (the index's convention)
	c.CpuTierBytes = int64(capacityBlocks) * c.KvBlockBytes
	return c
}

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
			for _, want := range []string{"pred=1.0000", "measured=0.9000", "pred_blocks=100", "fairness_id=fb-m", "prompt=100", "cached=90"} {
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
					t.Fatalf("a configured CPU tier changed a decision at line %d under entryOrder=%s:\n  with tier: %s\n  golden:    %s\nThe tier must be invisible to the modules that ship (it was a PRIORITY input for hrrn-tier only, and that module is gone).",
						i+1, tc.entry, a, b)
				}
			}
			t.Fatalf("trace length differs under entryOrder=%s with a tier configured", tc.entry)
		})
	}
}

// TestTierShadowValidationIncludesColdReturns pins the pair population's HONESTY: a return
// the shadow says is COLD (prediction 0.0) must still produce a validation pair. Keeping
// only the warm predictions (a `resident > 0` filter) would bias the population toward the
// cases the model gets right and flatter the AUC — the one number this arm is gated on.
func TestTierShadowValidationIncludesColdReturns(t *testing.T) {
	forEngines(t, tierConfig(50), func(t *testing.T, fx *engineSet) {
		type pair struct{ predicted, measured float64 }
		var pairs []pair
		fx.s.onShadowValidation = func(_ string, predicted, measured float64) {
			pairs = append(pairs, pair{predicted, measured})
		}
		fx.tighten(t) // one engine takes every seat, so both members share one tier
		fx.seatOne(t, "cold-return", 100, time.Second)
		g := fx.s.groups[fx.s.bindings["cold-return"]]
		settle(fx.s, "cold-return", 100, 100)
		// a SECOND member's settle overflows the 50-block tier, evicting cold-return's copy
		fx.seatOne(t, "other", 100, time.Second)
		if got := fx.s.bindings["other"]; got != g.id {
			t.Fatalf("premise: 'other' must share the tier's engine, got %s want %s", got, g.id)
		}
		settle(fx.s, "other", 100, 100)
		chain := keys(100, seedFor("cold-return"))
		for _, h := range chain {
			if g.tier.Holds(h) {
				t.Fatal("premise: cold-return's copy must have been evicted (the tier held 50 of its 100 blocks)")
			}
		}
		// it comes back: the prediction must be 0.0 and it must STILL be paired.
		fx.s.mu.Lock()
		fx.s.exitLocked(g, g.members["cold-return"], "test", fx.clock)
		fx.s.mu.Unlock()
		h := Head{Member: "cold-return", ReqID: "r-cold-2", Arrival: fx.clock.Add(-20 * time.Second),
			SizeBytes: bytesFor(100), FairnessID: "fb-cold"}
		if i := fx.pick(h); i < 0 {
			t.Fatal("the returning member must be seated")
		}
		m, _ := fx.member(t, "cold-return")
		if !m.predSet || m.predResident != 0 {
			t.Fatalf("a COLD return's prediction = %v (set=%v), want 0.0 and set", m.predResident, m.predSet)
		}
		dispatch(t, fx.s, "cold-return", chain, bytesFor(100))
		settle(fx.s, "cold-return", 100, 5)
		if len(pairs) != 1 {
			t.Fatalf("got %d validation pairs, want 1 — a cold return's prediction must not be dropped", len(pairs))
		}
		if pairs[0].predicted != 0 || pairs[0].measured != 0.05 {
			t.Fatalf("pair = (%v, %v), want (0, 0.05)", pairs[0].predicted, pairs[0].measured)
		}
	})
}
