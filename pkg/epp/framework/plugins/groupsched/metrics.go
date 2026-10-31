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
	// pooledEndpoints is the number of endpoints the policy has DISCOVERED
	// while engineGroups=single: 1 is normal, >1 means the pooled group is
	// summing independent engines into one budget/index (AUDIT N1 — the knob's
	// default fails OPEN, so the state must be visible as a series, not only as
	// a startup log line). Always 0/absent when engineGroups=per-engine.
	pooledEndpoints prometheus.Gauge
	// capacityHeterogeneous is 1 when a measured num_gpu_blocks deviates >1%
	// from the first engine's: a heterogeneous pool must never silently
	// inherit another engine's budget.
	capacityHeterogeneous prometheus.Gauge
	// filterDecisions books every dispatch the scheduling Filter saw, by what it
	// did (pinned | passthrough-*). An arm whose decisions are all passthrough-
	// unseated is NOT a pinned arm, and that is the state the E3 external filter's
	// silent-no-op class made invisible (PREREG-multipod-offload §E2).
	filterDecisions *prometheus.CounterVec
	// ---- E4b residue export (the arm scores its OWN argmax post-hoc) ----
	//
	// residuePredicted is the model's predicted surviving FRACTION at each
	// seat-time engine choice; residueActual is the measured fraction at settle
	// (usage.cached_tokens / prompt). A calibrated model puts the two in the same
	// place; the gap IS the model's error, with no ground-truth cache read.
	residuePredicted *prometheus.HistogramVec
	residueActual    *prometheus.HistogramVec
	// residueDecided counts seat choices the MODEL decided (it was the
	// discriminating key), per engine; residueAgreed counts the rest. An arm whose
	// decisions are all "agreed" cannot attribute any outcome to the model.
	residueDecided *prometheus.CounterVec
	residueResets  *prometheus.CounterVec
	residueOff     *prometheus.GaugeVec
	// valveBlockedEvictions counts rotations the cold-entry VALVE forced: the
	// engine's budget would have admitted the head, the valve refused it, so the
	// only way forward was to evict somebody. It is the C1 sizing signal (what the
	// stagger costs in evictions), emitted in BOTH binding models so the
	// member-stable and seat-pin arms stay comparable.
	valveBlockedEvictions *prometheus.CounterVec
	// engineGateSkips counts DISPATCHES the per-engine saturation gate held back
	// (engineGate=per-engine): the pool gate applied to one engine. Together with
	// engineGateClosed it is the "the gate is per-engine now" evidence in the run.
	engineGateSkips  *prometheus.CounterVec
	engineGateClosed *prometheus.GaugeVec
	// engineKV / engineWaiting export the gate's OWN inputs per engine (the pushed
	// scraped values the gate reads), so a run's gate behaviour is auditable from
	// the series alone.
	engineKV      *prometheus.GaugeVec
	engineWaiting *prometheus.GaugeVec
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
	// lastHomings mirrors the scheduler's per-engine homing count into the
	// Prometheus counter as a DELTA on each Refresh. Without it
	// llm_d_epp_group_homings_total stayed at its pre-instantiated 0 while the
	// snapshot carried the real number (measured on live arm e1: 203 homings in
	// the dump, 0 in the series) — a metric that silently disagrees with the
	// state dump is worse than no metric.
	lastHomings map[string]int64
}

func newGroupMetrics(sched *Scheduler) *groupMetrics {
	return &groupMetrics{sched: sched, known: map[string]struct{}{}, lastHomings: map[string]int64{}}
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
	g.pooledEndpoints = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "pooled_endpoints",
		Help: "endpoints pooled into the ONE group while engineGroups=single (>1 means independent engines are summed: wrong budget, wrong dedup)"})
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
	g.filterDecisions = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "filter_decisions_total",
		Help: "placement-Filter decisions by outcome (pinned | engine-not-a-candidate | passthrough-unseated | passthrough-anonymous | passthrough-single-candidate | passthrough-binding-mode)"}, []string{"decision"})
	g.residuePredicted = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "residue_predicted_ratio",
		Help: "predicted surviving FRACTION of the returning member's chain on the chosen engine (E4b, at seat time)", Buckets: residueBuckets}, engineLabels)
	g.residueActual = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "residue_actual_ratio",
		Help: "MEASURED surviving fraction at settle (usage.cached_tokens/prompt) — the model's outcome, not a cache read", Buckets: residueBuckets}, engineLabels)
	g.residueDecided = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "residue_decided_total",
		Help: "seat-time engine choices by whether the residue model DECIDED them (decision=model|agreed) and engine"}, []string{"engine", "decision"})
	g.residueResets = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "residue_reset_total",
		Help: "per-engine residue self-model resets (the model was voided and must rebuild)"}, []string{"engine", "reason"})
	g.residueOff = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "residue_off",
		Help: "1 when the residue model is disabled on this engine (sustained prediction error): it ranks by fit → backlog only"}, engineLabels)
	g.valveBlockedEvictions = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "rotation_evictions_valve_blocked_total",
		Help: "rotation steps where the cold-entry valve (not the budget) refused a head the budget would have taken, forcing an eviction"}, engineLabels)
	g.engineGateSkips = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gate_skipped_total",
		Help: "dispatches (and seats) the per-engine saturation gate held back, by engine"}, engineLabels)
	g.engineGateClosed = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "engine_gate_closed",
		Help: "1 when THIS engine's own saturation_e >= 1 (the per-engine gate is closed for it)"}, engineLabels)
	g.engineKV = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "engine_kv_usage",
		Help: "the SCRAPED per-engine kv_cache_usage this policy's per-engine gate reads (not the pool average)"}, engineLabels)
	g.engineWaiting = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "engine_waiting_queue",
		Help: "the SCRAPED per-engine waiting queue depth this policy's per-engine gate reads"}, engineLabels)
	fcfsCounter = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: "llm_d_epp_group", Name: "fcfs_dispatches_total",
		Help: "pre-capacity FCFS pass-through dispatches, per engine"}, engineLabels)
	for _, c := range []prometheus.Collector{g.seats, g.members, g.sessions, g.boundMembers,
		g.coldInFlight, g.uniqueBlocks, g.neBlocks, g.budgetBlocks, g.capacity,
		g.waiters, g.exits, g.residency, g.entryWait, g.refillShare, g.served, fcfsCounter,
		g.capacityMissing, g.capacityLearned, g.pooledEndpoints, g.capacityHeterogeneous,
		g.affinityViolations, g.rebinds, g.homings, g.rehomes, g.engineGone, g.filterDecisions,
		g.residuePredicted, g.residueActual, g.residueDecided, g.residueResets, g.residueOff,
		g.valveBlockedEvictions, g.engineGateSkips, g.engineGateClosed, g.engineKV, g.engineWaiting} {
		if err := reg.Register(c); err != nil {
			// A SECOND instantiation in one process (a config reload, or a second
			// plugin config in the same test binary) must not fail startup: the
			// series exist and belong to the first instance. Failing here meant a
			// duplicate-metrics crash for a situation that is a duplicate
			// configuration, not a broken build — and it made the runner's own
			// config test unable to instantiate a second arm config.
			if _, dup := err.(prometheus.AlreadyRegisteredError); dup {
				fmt.Printf("group-scheduler: WARNING metrics already registered in this process — this instance will not emit its own series (one group scheduler per process is the supported configuration)\n")
				continue
			}
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
	if g.engineKV != nil {
		for engine, st := range g.sched.EngineLoads() {
			g.engineKV.WithLabelValues(engineLabel(engine)).Set(st.KV)
			g.engineWaiting.WithLabelValues(engineLabel(engine)).Set(float64(st.Waiting))
			closed := 0.0
			if st.KV >= 0.80 || st.Waiting >= 5 {
				closed = 1
			}
			g.engineGateClosed.WithLabelValues(engineLabel(engine)).Set(closed)
		}
	}
	// PRE-INSTANTIATE the per-engine COUNTER series for every known engine.
	// A *CounterVec has no series until its first observation, so before any
	// dispatch a scrape would carry no llm_d_epp_group_dispatches_total at all —
	// and the deploy pre-flight (which requires that series, because the analysis
	// reads it) failed on exactly that at 2026-10-30 18:23Z on the first live
	// multi-pod stack. A ZERO is a fact; an absent series is a question.
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
		g.served.WithLabelValues(label).Add(0)
		if prev, ok := g.lastHomings[label]; !ok || es.Homings != prev {
			delta := es.Homings - prev
			if delta > 0 {
				g.homings.WithLabelValues(label).Add(float64(delta))
			}
			g.lastHomings[label] = es.Homings
		}
		if fcfsCounter != nil {
			fcfsCounter.WithLabelValues(label).Add(0)
		}
		for _, reason := range []string{"paid", "idle", "nmax", "exiting", "engine-idle",
			"quantum-complete", "ttl", "exiting-drain", "budget-idle", "budget-noqueue",
			"budget-exiting", "engine-gone"} {
			g.exits.WithLabelValues(label, reason).Add(0)
		}
	}
	// an engine that disappeared must not keep its last gauge value
	for label := range g.known {
		if _, ok := seen[label]; ok {
			continue
		}
		delete(g.known, label)
		delete(g.lastHomings, label)
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

// ObserveFilterDecision books one placement-Filter outcome (nil-safe: a unit-mode
// run has no registrar and must still take the same code path).
// residueBuckets spans the measured bimodality: cold members read 0.12-0.17 and
// warm ones ~0.98, so the buckets must resolve both ends and the gap.
var residueBuckets = []float64{0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}

func (g *groupMetrics) ObserveResiduePredicted(engine string, predicted float64, decided bool) {
	if g == nil || g.residuePredicted == nil {
		return
	}
	g.residuePredicted.WithLabelValues(engine).Observe(predicted)
	d := "agreed"
	if decided {
		d = "model"
	}
	if g.residueDecided != nil {
		g.residueDecided.WithLabelValues(engine, d).Inc()
	}
}

func (g *groupMetrics) ObserveResidueSettled(engine string, predicted, actual float64) {
	if g == nil || g.residueActual == nil {
		return
	}
	g.residueActual.WithLabelValues(engine).Observe(actual)
}

func (g *groupMetrics) SetResidueDisabled(engine string) {
	if g == nil || g.residueOff == nil {
		return
	}
	g.residueOff.WithLabelValues(engine).Set(1)
}

func (g *groupMetrics) ObserveEngineGateSkip(engine string) {
	if g == nil || g.engineGateSkips == nil {
		return
	}
	g.engineGateSkips.WithLabelValues(engine).Inc()
}

func (g *groupMetrics) ObserveValveBlockedEviction(engine string) {
	if g == nil || g.valveBlockedEvictions == nil {
		return
	}
	g.valveBlockedEvictions.WithLabelValues(engine).Inc()
}

func (g *groupMetrics) ObserveResidueReset(engine, reason string) {
	if g == nil || g.residueResets == nil {
		return
	}
	g.residueResets.WithLabelValues(engine, reason).Inc()
}

func (g *groupMetrics) ObserveFilterDecision(decision string) {
	if g == nil || g.filterDecisions == nil {
		return
	}
	g.filterDecisions.WithLabelValues(decision).Inc()
}

// SetPooledEndpoints reports how many endpoints the single group is pooling
// (AUDIT N1: engineGroups=single is the default and it fails OPEN, so a
// multi-endpoint pool must be visible as a series).
func (g *groupMetrics) SetPooledEndpoints(n int) {
	if g.pooledEndpoints == nil {
		return
	}
	g.pooledEndpoints.Set(float64(n))
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
