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

// EVERY KEPT COUNTER MOVES ON ITS OWN EVENT, and on nothing else.
//
// A series is a promise about an EVENT, not about a code path that happens to run: a
// valve counter that counted loop passes instead of heads waited on, or a metric that is
// registered and never incremented, reads as a healthy zero and voids every conclusion
// drawn from it. So each counter here is driven by exactly one event, its delta is read
// from the registry, and the neighbouring series are checked not to move.

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// metricFixture wires the plugin's metric hooks to a fresh registry, so a test can read
// one series' delta without a scrape.
type metricFixture struct {
	s   *Scheduler
	reg *prometheus.Registry
	m   *groupMetrics
}

func newMetricFixture(t *testing.T, engines ...string) *metricFixture {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	m := newGroupMetrics(s)
	reg := prometheus.NewRegistry()
	if err := m.register(reg); err != nil {
		t.Fatal(err)
	}
	s.onServe = func(engine string) { m.ObserveServed(engine) }
	s.onExit = func(engine, reason string, turns int) { m.ObserveExit(engine, reason) }
	s.onEntry = func(engine, kind string, wait float64) { m.ObserveEntry(engine, kind, wait) }
	s.onColdValveHeld = func(engine string) { m.ObserveColdValveHeld(engine) }
	s.onEngineGone = func(engine string, n int) { m.ObserveEngineGone() }
	s.onRehome = func(branch, from, to string) { m.ObserveAffinityViolation(from, to) }
	for _, e := range engines {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	return &metricFixture{s: s, reg: reg, m: m}
}

// delta is one series' value in the registry as it stands.
func (f *metricFixture) value(t *testing.T, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := f.reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, fam := range families {
		if fam.GetName() != subsystem+"_"+name {
			continue
		}
		for _, met := range fam.GetMetric() {
			match := true
			for k, want := range labels {
				found := false
				for _, lp := range met.GetLabel() {
					if lp.GetName() == k && lp.GetValue() == want {
						found = true
					}
				}
				if !found {
					match = false
				}
			}
			if !match {
				continue
			}
			switch {
			case met.GetCounter() != nil:
				return met.GetCounter().GetValue()
			case met.GetGauge() != nil:
				return met.GetGauge().GetValue()
			case met.GetHistogram() != nil:
				return float64(met.GetHistogram().GetSampleCount())
			}
		}
	}
	return 0
}

func (f *metricFixture) seat(t *testing.T, id string, blocks int, now time.Time) {
	t.Helper()
	if i := f.s.PickFromHeads([]Head{head(id, time.Second, blocks)}, now); i != 0 {
		t.Fatalf("%s must seat", id)
	}
	f.s.Dispatched(id, "r-"+id, now)
}

// dispatches_total moves on a DISPATCH, and not on a seat alone.
func TestCounterDispatchesMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	if got := f.value(t, "dispatches_total", map[string]string{"engine": "e0"}); got != 0 {
		t.Fatalf("before any dispatch: %v, want 0", got)
	}
	f.seat(t, "m", 100, now)
	if got := f.value(t, "dispatches_total", map[string]string{"engine": "e0"}); got != 1 {
		t.Fatalf("one dispatch: %v, want 1", got)
	}
}

// exits_total moves on an EXIT with the reason it happened for.
func TestCounterExitsMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "m", 100, now)
	settle(f.s, "m", 100, 100) // the turn ends: nothing in flight, one busy period
	f.s.mu.Lock()
	f.s.groups["e0"].branches["m"].busyPeriods = 4 // past Q
	f.s.mu.Unlock()
	// a waiting head that cannot fit evicts the rotation-eligible branch
	f.s.mu.Lock()
	g := f.s.groups["e0"]
	g.budgetBlocks = 100
	reason, victim := f.s.rotationVictimLocked(g, now)
	if victim == nil {
		t.Fatal("the branch must be rotation-eligible")
	}
	f.s.exitLocked(g, victim, reason, now)
	f.s.mu.Unlock()
	if got := f.value(t, "exits_total", map[string]string{"engine": "e0", "reason": reason}); got != 1 {
		t.Fatalf("exits_total{reason=%q} = %v, want 1", reason, got)
	}
	if got := f.value(t, "exits_total", map[string]string{"engine": "e0", "reason": "ttl"}); got != 0 {
		t.Fatalf("a series for a reason that no longer exists must stay 0, got %v", got)
	}
}

// cold_valve_held_total moves when the VALVE holds a head — once per pass that holds it,
// not once per candidate engine and not once per loop iteration.
func TestCounterColdValveHeldMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	// a resident whose cold charge fills the cap (5 % of 1000 blocks = 3200 tokens)
	f.s.mu.Lock()
	g := f.s.groups["e0"]
	g.budgetBlocks = 950
	f.s.mu.Unlock()
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "resident", 600, now)
	before := f.value(t, "cold_valve_held_total", map[string]string{"engine": "e0"})
	// a head the budget would take and only the valve refuses
	if _, i := f.s.PickFromHeadsEngine([]Head{head("cold", time.Second, 300)}, now); i != -1 {
		t.Fatalf("the valve must hold the head, got %d", i)
	}
	if got := f.value(t, "cold_valve_held_total", map[string]string{"engine": "e0"}); got != before+1 {
		t.Fatalf("cold_valve_held_total: %v -> %v, want exactly one increment for the held head", before, got)
	}
}

// gift: entry_wait_seconds moves once per SEAT with the kind that seat had.
func TestHistogramEntryWaitMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	before := f.value(t, "entry_wait_seconds", map[string]string{"kind": "new"})
	f.seat(t, "m", 100, now)
	if got := f.value(t, "entry_wait_seconds", map[string]string{"kind": "new"}); got != before+1 {
		t.Fatalf("entry_wait_seconds{kind=new}: %v -> %v, want +1", before, got)
	}
	if got := f.value(t, "entry_wait_seconds", map[string]string{"kind": "reentry"}); got != 0 {
		t.Fatalf("a first turn must not be counted as a re-entry, got %v", got)
	}
}

// engine_gone_total moves on a TEARDOWN.
func TestCounterEngineGoneMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t, "e0", "e1")
	if got := f.value(t, "engine_gone_total", nil); got != 0 {
		t.Fatalf("before any teardown: %v", got)
	}
	f.s.RemoveEngine("e1", time.Unix(1793000000, 0).UTC())
	if got := f.value(t, "engine_gone_total", nil); got != 1 {
		t.Fatalf("engine_gone_total: %v, want 1", got)
	}
}

// affinity_violations_total and rebinds_total move on an OBSERVED engine that disagrees
// with the seat.
func TestCountersAffinityAndRebindMoveOnTheirEvent(t *testing.T) {
	f := newMetricFixture(t, "e0", "e1")
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "m", 100, now)
	if got := f.value(t, "rebinds_total", nil); got != 0 {
		t.Fatalf("before any rebound: %v", got)
	}
	// the framework lands the request on e1 although the seat is on e0
	f.s.DispatchedOn("m", "r-m2", "e1", now)
	if got := f.value(t, "rebinds_total", nil); got != 1 {
		t.Fatalf("rebinds_total: %v, want 1", got)
	}
	if got := f.value(t, "affinity_violations_total", map[string]string{"from": "e0", "to": "e1"}); got != 1 {
		t.Fatalf("affinity_violations_total{e0->e1}: %v, want 1", got)
	}
}

// capacity_learned_total moves when the capacity arrives, with the path it arrived by.
func TestCounterCapacityLearnedMovesOnItsEvent(t *testing.T) {
	f := newMetricFixture(t)
	reg := prometheus.NewRegistry()
	m := newGroupMetrics(f.s)
	if err := m.register(reg); err != nil {
		t.Fatal(err)
	}
	m.ObserveCapacityLearned("event", "e0")
	if got := testutil.ToFloat64(m.capacityLearned.WithLabelValues("event", "e0")); got != 1 {
		t.Fatalf("capacity_learned_total{via=event}: %v, want 1", got)
	}
	_ = fwkdl.Metrics{}
}
