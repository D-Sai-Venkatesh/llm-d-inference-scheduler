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
	"github.com/prometheus/client_golang/prometheus"
)

// groupMetrics exports the §13 legibility set: group size in seats and
// unique blocks, exits by reason, residency turns, entry waits, member
// repeat-turn hit, and the refill share.
//
// EVERY per-engine series carries an `engine` label (PREREG-multipod-offload:
// the reducer-before-the-run rule — without it the per-engine equivalence
// test E1 is unscoreable and the split-skew precondition is uncomputable).
// In engineGroups=single the label value is "single" and the SUMS over label
// values equal the pre-split numbers exactly, so a banked single-pod scrape
// and a new one stay comparable: `sum(llm_d_epp_group_seats)` is the
// whole-policy seat count in both worlds.
type groupMetrics struct {
	seats        *prometheus.GaugeVec
	uniqueBlocks *prometheus.GaugeVec
	neBlocks     *prometheus.GaugeVec
	budgetBlocks *prometheus.GaugeVec
	waiters      *prometheus.GaugeVec
	// members is the per-engine seating count (= seats; kept as its own
	// series because "members" is the name the equivalence test's skew
	// precondition is stated in).
	members *prometheus.GaugeVec
	// sessions is the per-engine count of DISTINCT session units among the
	// seated members (the second half of the skew precondition: sessions
	// within ±15% of an even split; sessions span engines, so this is a
	// per-engine view of a global identity population).
	sessions *prometheus.GaugeVec
	// boundMembers counts the members HOMED to each engine — including
	// members that are currently unseated. It is the binding store's size
	// per engine (the homing decision's footprint).
	boundMembers *prometheus.GaugeVec
	// coldInFlight is the per-engine cold-entry charge in tokens: N engines
	// may hold N concurrent cold prefills, so the concurrency has to be
	// visible per engine to re-derive R11's deadlock proof under N.
	coldInFlight *prometheus.GaugeVec
	capacity     *prometheus.GaugeVec
	exits        *prometheus.CounterVec
	residency    *prometheus.HistogramVec
	entryWait    *prometheus.HistogramVec
	refillShare  prometheus.Gauge
	served       *prometheus.CounterVec
	// capacityMissing is 1 while the measured capacity (num_gpu_blocks) of
	// THAT engine is unknown past CapacityLivenessSeconds — the per-endpoint
	// liveness test (A5's pre-capacity pass-through is otherwise invisible,
	// and one late-ready engine is invisible entirely).
	capacityMissing *prometheus.GaugeVec
	// capacityLearned counts how each engine's capacity arrived (endpoint-
	// event | metrics-poll | pre-request) — the acquisition audit.
	capacityLearned *prometheus.CounterVec
	// capacityHeterogeneous is 1 when a measured num_gpu_blocks deviates >1%
	// from the first engine's: a heterogeneous pool must never silently
	// inherit another engine's budget.
	capacityHeterogeneous prometheus.Gauge
	// affinityViolations counts requests that landed off their member's
	// engine (from = the engine the member was charged to, to = the engine
	// the framework actually picked). CORRECTNESS, not instrumentation.
	affinityViolations *prometheus.CounterVec
	rebinds            prometheus.Counter
	homings            *prometheus.CounterVec
	rehomes            prometheus.Counter
	engineGone         prometheus.Counter

	sched *Scheduler
	// known tracks the engine label values already instantiated, so Refresh
	// can zero the gauges of an engine that disappeared (a stale series
	// reads as "engine alive" forever otherwise).
	known map[string]struct{}
}

func newGroupMetrics(sched *Scheduler) *groupMetrics {
	return &groupMetrics{sched: sched, known: map[string]struct{}{}}
}

func (g *groupMetrics) register(reg prometheus.Registerer) error {
	const subsystem = "llm_d_epp_group"
	engineLabels := []string{"engine"}
	g.seats = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "seats",
		Help: "group members (seated sessions) per engine"}, engineLabels)
	g.members = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "members",
		Help: "seated members per engine (the E1 skew precondition's numerator)"}, engineLabels)
	g.sessions = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "sessions",
		Help: "distinct session units among the seated members, per engine"}, engineLabels)
	g.boundMembers = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "bound_members",
		Help: "members HOMED to this engine (bindings, seated or not)"}, engineLabels)
	g.coldInFlight = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_inflight_tokens",
		Help: "cold-entry charge in flight per engine (tokens)"}, engineLabels)
	g.uniqueBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "unique_blocks",
		Help: "unique blocks held by all seated members, per engine"}, engineLabels)
	g.neBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "ne_blocks",
		Help: "unique blocks over non-exiting members (the per-engine budget counter)"}, engineLabels)
	g.budgetBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "budget_blocks",
		Help: "the engine's group budget in blocks (0.95 x its num_gpu_blocks)"}, engineLabels)
	g.capacity = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "num_gpu_blocks",
		Help: "the engine's measured num_gpu_blocks (0 = not learned yet)"}, engineLabels)
	g.waiters = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters",
		Help: "non-members with waiting requests, per engine"}, engineLabels)
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "member exits by reason and engine"}, []string{"engine", "reason"})
	g.residency = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "residency_turns",
		Help: "turns per residency at exit", Buckets: prometheus.LinearBuckets(1, 1, 20)}, engineLabels)
	g.entryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help: "newcomer seat waits", Buckets: prometheus.ExponentialBuckets(0.1, 2, 14)}, engineLabels)
	g.refillShare = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "refill_share",
		Help: "uncached share of settled prompt tokens (the refill cost)"})
	g.served = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "requests dispatched to members, per engine"}, engineLabels)
	g.capacityMissing = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_missing",
		Help: "1 while this engine's num_gpu_blocks is unknown past capacityLivenessSeconds (FCFS pass-through in effect)"}, engineLabels)
	g.capacityLearned = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "capacity_learned_total",
		Help: "capacity acquisition by path (endpoint-event | metrics-poll | pre-request) and engine"}, []string{"via", "engine"})
	g.capacityHeterogeneous = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_heterogeneous",
		Help: "1 when a measured num_gpu_blocks deviates >1% from the first engine's (a pool that must NOT share a budget)"})
	g.affinityViolations = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "affinity_violations_total",
		Help: "requests that landed off their member's engine (from = charged engine, to = observed engine)"}, []string{"from", "to"})
	g.rebinds = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "rebinds_total",
		Help: "unseat-and-rebind corrections after an affinity violation"})
	g.homings = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "homings_total",
		Help: "first-sight member homing decisions per engine"}, engineLabels)
	g.rehomes = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "rehomes_total",
		Help: "silent-member rehoming (rate-bounded defrag; a counter stub until it is wired)"})
	g.engineGone = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gone_total",
		Help: "engines torn down by an endpoint delete event"})
	fcfsCounter = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: "llm_d_epp_group", Name: "fcfs_dispatches_total",
		Help: "pre-capacity FCFS pass-through dispatches, per engine"}, engineLabels)
	for _, c := range []prometheus.Collector{g.seats, g.members, g.sessions, g.boundMembers,
		g.coldInFlight, g.uniqueBlocks, g.neBlocks, g.budgetBlocks, g.capacity,
		g.waiters, g.exits, g.residency, g.entryWait, g.refillShare, g.served, fcfsCounter,
		g.capacityMissing, g.capacityLearned, g.capacityHeterogeneous,
		g.affinityViolations, g.rebinds, g.homings, g.rehomes, g.engineGone} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	return nil
}

// Refresh pushes a scheduler snapshot into the gauges (the ticker calls
// it every second; the counters and histograms are observed at their
// events by the scheduler).
func (g *groupMetrics) Refresh() {
	if g.seats == nil {
		return
	}
	snap := g.sched.SnapshotLite()
	seen := make(map[string]struct{}, len(snap.Engines))
	for label, es := range snap.Engines {
		seen[label] = struct{}{}
		g.known[label] = struct{}{}
		g.seats.WithLabelValues(label).Set(float64(es.Seats))
		g.members.WithLabelValues(label).Set(float64(es.Seats))
		g.sessions.WithLabelValues(label).Set(float64(es.Sessions))
		g.boundMembers.WithLabelValues(label).Set(float64(es.BoundMembers))
		g.coldInFlight.WithLabelValues(label).Set(float64(es.ColdInFlight))
		g.uniqueBlocks.WithLabelValues(label).Set(float64(es.UniqueBlocks))
		g.neBlocks.WithLabelValues(label).Set(float64(es.NEBlocks))
		g.budgetBlocks.WithLabelValues(label).Set(float64(es.BudgetBlocks))
		g.capacity.WithLabelValues(label).Set(float64(es.NumBlocks))
		g.waiters.WithLabelValues(label).Set(float64(es.Waiters))
	}
	// an engine that disappeared must not keep its last gauge value
	for label := range g.known {
		if _, ok := seen[label]; ok {
			continue
		}
		delete(g.known, label)
		for _, vec := range []*prometheus.GaugeVec{g.seats, g.members, g.sessions, g.boundMembers,
			g.coldInFlight, g.uniqueBlocks, g.neBlocks, g.budgetBlocks, g.capacity, g.waiters} {
			vec.DeleteLabelValues(label)
		}
	}
	g.refillShare.Set(snap.RefillShare)
	if len(snap.HeterogeneousEngines) > 0 {
		g.capacityHeterogeneous.Set(1)
	} else {
		g.capacityHeterogeneous.Set(0)
	}
}

// ObserveExit records a member exit (residency turns + the reason + engine).
func (g *groupMetrics) ObserveExit(engine, reason string, resTurns int) {
	if g.exits == nil {
		return
	}
	g.exits.WithLabelValues(engine, reason).Inc()
	g.residency.WithLabelValues(engine).Observe(float64(resTurns))
}

// ObserveEntry records a newcomer seat wait.
func (g *groupMetrics) ObserveEntry(engine string, waitSeconds float64) {
	if g.entryWait == nil {
		return
	}
	g.entryWait.WithLabelValues(engine).Observe(waitSeconds)
}

// ObserveServed records a dispatch.
func (g *groupMetrics) ObserveServed(engine string) {
	if g.served != nil {
		g.served.WithLabelValues(engine).Inc()
	}
}

// ObserveHoming records a first-sight homing decision.
func (g *groupMetrics) ObserveHoming(engine string) {
	if g.homings != nil {
		g.homings.WithLabelValues(engine).Inc()
	}
}

// ObserveAffinityViolation records a request that landed off its member's
// engine (and the rebind that repaired it).
func (g *groupMetrics) ObserveAffinityViolation(from, to string) {
	if g.affinityViolations == nil {
		return
	}
	g.affinityViolations.WithLabelValues(from, to).Inc()
	if g.rebinds != nil {
		g.rebinds.Inc()
	}
}

// ObserveRehome records a (stubbed) silent-member rehoming.
func (g *groupMetrics) ObserveRehome() {
	if g.rehomes != nil {
		g.rehomes.Inc()
	}
}

// ObserveEngineGone records an engine torn down by an endpoint delete.
func (g *groupMetrics) ObserveEngineGone() {
	if g.engineGone != nil {
		g.engineGone.Inc()
	}
}

// ObserveCapacityLearned records the capacity acquisition path.
func (g *groupMetrics) ObserveCapacityLearned(via, engine string) {
	if g.capacityLearned == nil {
		return
	}
	g.capacityLearned.WithLabelValues(via, engine).Inc()
}

// SetCapacityMissing raises/clears the per-engine capacity-liveness gauge.
func (g *groupMetrics) SetCapacityMissing(engine string, missing bool) {
	if g.capacityMissing == nil {
		return
	}
	v := 0.0
	if missing {
		v = 1
	}
	g.capacityMissing.WithLabelValues(engine).Set(v)
}

// fcfsCounter is the pre-capacity pass-through counter.
var fcfsCounter *prometheus.CounterVec

// ObserveFCFS records a pre-capacity FCFS dispatch.
func (g *groupMetrics) ObserveFCFS(engine string) {
	if fcfsCounter != nil {
		fcfsCounter.WithLabelValues(engine).Inc()
	}
}
