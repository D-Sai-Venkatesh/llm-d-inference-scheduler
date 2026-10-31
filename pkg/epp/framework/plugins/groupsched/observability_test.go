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

// THE OBSERVABILITY SUITE — the two series that made a real defect invisible
// (2026-11-01, reviewer's ask; both observability-only, golden-neutral):
//
//  1. cold_valve_held_total was attributed `engine="single"` on a multi-engine stack,
//     because the admission loop passed an empty engine id. A series whose label is a
//     LIE at N=2 cannot answer "which engine's prefill slot is the stagger costing?".
//  2. The C8 guard ("a stale queue view never justifies an eviction") had no clock of
//     its own: an arm whose queue view is never refreshed and one that is always fresh
//     looked identical in every series. Now queue_view_age_seconds is the clock and
//     queue_view_untrusted_passes_total is the count of passes that needed the guard.
//
// Both run at N=1 AND N=2, and both assert the WIRE FORM (the Prometheus series), not
// just the hook: a counter nobody can scrape is not observability.

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// gatherValue pulls one sample out of a fresh gather, matching every label pair given.
func gatherValue(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) (float64, bool) {
	t.Helper()
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			match := true
			seen := map[string]string{}
			for _, lp := range m.GetLabel() {
				seen[lp.GetName()] = lp.GetValue()
			}
			for k, v := range labels {
				if seen[k] != v {
					match = false
					break
				}
			}
			if !match {
				continue
			}
			return sampleValue(m), true
		}
	}
	return 0, false
}

// sampleValue reads whichever of the metric's value fields this series uses.
func sampleValue(m *dto.Metric) float64 {
	switch {
	case m.GetGauge() != nil:
		return m.GetGauge().GetValue()
	case m.GetCounter() != nil:
		return m.GetCounter().GetValue()
	case m.GetUntyped() != nil:
		return m.GetUntyped().GetValue()
	}
	return 0
}

// TestColdValveHeldIsAttributedToTheEngine: the valve-held series must name the engine
// whose prefill slot the head is waiting for, never "" (which the exporter renders
// `engine="single"` on a two-engine stack).
//
// The scenario holds the head on EVERY engine (each one carries an outstanding cold
// charge, so the small cap refuses it while its BUDGET would admit it) and gives the
// engines different free budget, so the assertion can tell the right answer ("the engine
// placement would have picked") from the wrong one ("the first engine in the loop").
func TestColdValveHeldIsAttributedToTheEngine(t *testing.T) {
	cfg := defaultConfig()
	cfg.ColdEntryCapTokens = 20_000 // tiny on purpose: 200 blocks of cold charge ~ 12.8K tokens
	forEngines(t, cfg, func(t *testing.T, fx *engineSet) {
		var held []string
		fx.s.onColdValveHeld = func(engine string) { held = append(held, engine) }

		// one cold charge per engine, and the FIRST engine deliberately fuller so the
		// two engines' free budgets differ (300 blocks used on e0, 200 on e1).
		fx.seatOn(t, fx.ids[0], "cold-0", 300, 3*time.Second)
		if len(fx.ids) > 1 {
			fx.seatOn(t, fx.ids[1], "cold-1", 200, 3*time.Second)
		}

		// a head that fits every engine's BUDGET but is refused by every engine's VALVE
		if i := fx.pick(head("waiting", time.Second, bytesFor(200))); i >= 0 {
			t.Fatalf("the head must NOT be seated (the valve holds it), got pick %d", i)
		}
		if len(held) == 0 {
			t.Fatal("the valve held a head and fired no signal: cold_valve_held_total would stay 0 on a cap-binding arm")
		}
		for _, e := range held {
			if e == "" {
				t.Fatal("the series was booked with an empty engine id — that renders engine=\"single\" on a multi-engine stack (the defect this pins)")
			}
			found := false
			for _, id := range fx.ids {
				if id == e {
					found = true
				}
			}
			if !found {
				t.Fatalf("engine %q is not one of this stack's engines %v", e, fx.ids)
			}
		}
		if len(fx.ids) > 1 {
			// e1 has the most free budget (650 vs 750 blocks free), so it is the engine
			// the head is actually waiting for: the attribution follows placement's own
			// order, not the loop's.
			if want := fx.ids[1]; held[0] != want {
				t.Fatalf("valve-held engine = %q, want %q (the engine with the MOST FREE BUDGET among those that fit by budget)", held[0], want)
			}
		} else if want := fx.ids[0]; held[0] != want {
			t.Fatalf("valve-held engine = %q, want %q", held[0], want)
		}
	})
}

// TestQueueViewAgeAndUntrustedPassesAreVisible makes the C8 guard's clock scrapeable and
// pins what it means:
//
//   - a fresh view (the pass that just synced) reports age 0 and books nothing;
//   - a pass older than idleSeconds (S) reports age > S, books ONE untrusted pass for the
//     engine it enforced on, and — the property the guard exists for — does NOT evict the
//     member whose "no queued head" cannot be trusted;
//   - the very next pass with a fresh view takes the eviction, so the guard DELAYS the
//     decision by at most one tick instead of blocking it.
func TestQueueViewAgeAndUntrustedPassesAreVisible(t *testing.T) {
	// a small capacity so one member's growth puts its engine over budget
	forEnginesCap(t, defaultConfig(), 400, func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "big", 300, 3*time.Second) // budget 380, footprint 300
		// the head is CONSUMED (a Pick with nothing queued clears "has a queued head"),
		// so the member is exactly the headless-but-over-budget case the C8 guard governs.
		fx.pick()
		// its next request brings 300 NEW blocks: the accounted footprint is 600 > 380,
		// i.e. the engine is over budget and enforcement must choose a victim.
		fx.s.Dispatched("big", "r-second", keys(300, seedFor("big-2")), bytesFor(300), fx.clock)

		reg := prometheus.NewRegistry()
		m := newGroupMetrics(fx.s)
		if err := m.register(reg); err != nil {
			t.Fatalf("register: %v", err)
		}
		// the pass that just ran OBSERVED the queue: the evidence is 0 s old, and no pass
		// has needed the guard.
		m.Refresh(0)
		snap := fx.s.Snapshot()
		if snap.QueueViewAgeSeconds != 0 {
			t.Fatalf("a pass that just observed the queue reports age %v s, want 0", snap.QueueViewAgeSeconds)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_queue_view_age_seconds", nil); !ok || got != 0 {
			t.Fatalf("queue_view_age_seconds = %v (present=%v), want 0", got, ok)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_queue_view_untrusted_passes_total", nil); !ok || got != 0 {
			t.Fatalf("queue_view_untrusted_passes_total = %v (present=%v), want 0 with fresh evidence", got, ok)
		}

		// advance past S WITHOUT a Pick (a closed pool gate is the normal cause): the head
		// EVIDENCE goes stale, which is the C8 window.
		stale := time.Duration(defaultConfig().IdleSeconds+1) * time.Second
		fx.clock = fx.clock.Add(stale)
		fx.s.Tick(fx.clock)

		// the enforcement pass ran on that stale view: it must book the pass, and it must
		// NOT evict on "no queued head" (that is the race the guard closes).
		snap = fx.s.Snapshot()
		if snap.QueueViewUntrusted != 1 {
			t.Fatalf("untrusted enforcement passes = %d, want 1 (one engine over budget, one stale-viewed pass)", snap.QueueViewUntrusted)
		}
		if !fx.s.Seated("big") {
			t.Fatal("the member must SURVIVE the stale-viewed pass: \"no queued head\" is not evidence when the view is older than S (the C8 race)")
		}
		// Tick re-APPLIES the last head snapshot (so the waiting flags stay honest) but it
		// does not OBSERVE the queue: the evidence age keeps measuring the window.
		if snap.QueueViewAgeSeconds <= defaultConfig().IdleSeconds {
			t.Fatalf("evidence age = %v s after a %v s gap, want > S (%v): a Tick re-sync must not mask how stale the OBSERVATION is",
				snap.QueueViewAgeSeconds, stale, defaultConfig().IdleSeconds)
		}
		m.Refresh(0)
		if got, ok := gatherValue(t, reg, subsystem+"_queue_view_age_seconds", nil); !ok || got <= defaultConfig().IdleSeconds {
			t.Fatalf("queue_view_age_seconds = %v (present=%v), want > S (%v)", got, ok, defaultConfig().IdleSeconds)
		}
		if got, ok := gatherValue(t, reg, subsystem+"_queue_view_untrusted_passes_total", nil); !ok || got != 1 {
			t.Fatalf("queue_view_untrusted_passes_total = %v (present=%v), want 1", got, ok)
		}

		// settle BOTH of the member's requests: a member with anything in flight is never
		// evictable by any class (a separate rule), so this leaves the stale-evidence guard
		// as the only thing keeping the member seated.
		usage := struct{ Prompt, Cached int }{Prompt: 600, Cached: 600}
		fx.s.Settle("big", "r-big", usage, true, fx.clock)
		fx.s.Settle("big", "r-second", usage, true, fx.clock)
		if !fx.s.Seated("big") {
			t.Fatal("settling must not evict: the guard is what kept this member seated, not the settle")
		}

		// the next pass OBSERVES the queue, and then the SAME over-budget state does evict:
		// the guard costs one tick, never a datapath stall.
		fx.pick()
		if fx.s.Seated("big") {
			t.Fatal("with fresh evidence the over-budget member must be evicted (the guard must not be a permanent shield)")
		}
		snap = fx.s.Snapshot()
		if snap.QueueViewUntrusted != 1 {
			t.Fatalf("untrusted passes = %d after a fresh pass, want still 1 (a fresh view must not be counted)", snap.QueueViewUntrusted)
		}
		if snap.QueueViewAgeSeconds != 0 {
			t.Fatalf("evidence age = %v s after a Pick, want 0", snap.QueueViewAgeSeconds)
		}
	})
}
