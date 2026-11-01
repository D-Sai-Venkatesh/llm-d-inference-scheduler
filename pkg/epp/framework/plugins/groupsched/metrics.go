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
	"math"

	"github.com/prometheus/client_golang/prometheus"
)

// groupMetrics exports the CORE legibility set: per engine the seats, the queues,
// the budget accounting, the exits by reason, the residency and entry-wait
// histograms, the dispatch count and the SAFETY NET (its live verdict, its fire
// events and the time spent fired). Plus the whole-stack correctness counters and
// the placement Filter's decisions.
//
// EVERY per-engine series carries an `engine` label, and the label value is the
// endpoint id. `sum(llm_d_epp_group_seats)` is therefore the whole-policy seat count
// with one engine and with N.
//
// The safety series are the ONE place the 15 s per-engine scrape reads the gate
// (the D3 fix): they are filled from Scheduler.SafetyStates(), which answers from the
// SAME safetyFiredLocked the decision path uses — the old metric re-derived closure
// from the literals 0.80/5 and would report a gate that was not the gate.
type groupMetrics struct {
	sched *Scheduler

	seats        *prometheus.GaugeVec
	uniqueBlocks *prometheus.GaugeVec
	neBlocks     *prometheus.GaugeVec
	budgetBlocks *prometheus.GaugeVec
	waiters      *prometheus.GaugeVec
	coldInFlight *prometheus.GaugeVec
	coldCap      *prometheus.GaugeVec
	capacity     *prometheus.GaugeVec
	// waitersShared is the SHARED waiter queue's size (one number, no engine label,
	// because a waiter belongs to no engine until one seats it).
	waitersShared prometheus.Gauge
	// queueViewAge is how old the queue-head snapshot behind "has a queued head" was on
	// the decision clock, and queueViewUntrusted counts the budget-enforcement passes
	// that ran with it too old to trust. The C8 guard SKIPS the head-presence classes in
	// exactly those passes, so without these two the guard's own clock is invisible — a
	// stack whose queue view is never refreshed and one that is always fresh look
	// identical in every other series.
	queueViewAge       prometheus.Gauge
	queueViewUntrusted prometheus.Counter
	// queueViewUntrustedLast is the scheduler's cumulative counter as last exported:
	// Refresh adds the DELTA, so the series has exactly one source of truth (the
	// scheduler's own count) and nothing can be double-booked or forgotten.
	queueViewUntrustedLast int64
	// eventRingDropped / eventRingDroppedLast are the same delta pattern for the ring's
	// wrap count.
	eventRingDropped     prometheus.Counter
	eventRingDroppedLast int64
	exits                *prometheus.CounterVec
	residency            *prometheus.HistogramVec
	entryWait            *prometheus.HistogramVec
	reentryWait          *prometheus.HistogramVec
	refillShare          prometheus.Gauge
	served               *prometheus.CounterVec
	fcfs                 *prometheus.CounterVec

	capacityMissing *prometheus.GaugeVec
	capacityLearned *prometheus.CounterVec

	// ---- the safety net (per engine) ----
	safetyFired        *prometheus.GaugeVec
	safetyFires        *prometheus.CounterVec
	safetyFiredSeconds *prometheus.CounterVec
	safetySkips        *prometheus.CounterVec
	engineKV           *prometheus.GaugeVec
	engineWaiting      *prometheus.GaugeVec

	// poolScore is the pool saturation the plugin last handed the flow controller as
	// the framework's detector (0 = not saturated, the default answer).
	poolScore prometheus.Gauge
	// staleFired counts per-engine samples the detector had to EXCLUDE because the
	// metrics had not been refreshed (stale → open).
	staleSamples *prometheus.CounterVec
	// capacityHeterogeneous is 1 when a measured num_gpu_blocks deviates >1% from
	// the first engine's: a heterogeneous pool must never silently inherit another
	// engine's budget.
	capacityHeterogeneous prometheus.Gauge
	// filterDecisions books every dispatch the scheduling Filter saw, by what it
	// did (pinned | passthrough-*). An arm whose decisions are all passthrough-
	// unseated is NOT a pinned arm, and that state must be visible.
	filterDecisions *prometheus.CounterVec
	// affinityViolations counts requests that landed off their member's seat engine.
	affinityViolations *prometheus.CounterVec
	rebinds            prometheus.Counter
	engineGone         prometheus.Counter
	// coldValveHeld counts heads the cold-entry VALVE held back: the budget would have
	// admitted the head, the valve refused it, so the head WAITED for the engine's
	// prefill slot. Nothing is evicted for it (the sizing signal: what the stagger
	// costs in waiting).
	coldValveHeld *prometheus.CounterVec

	// ---- the CPU-tier shadow (piece 1) and its VALIDATION (piece 3) ----
	//
	// shadowBlocks / shadowCapacity are the model's own state per engine: a shadow
	// pinned at capacity is the tier's steady state, a shadow far below it means the
	// store traffic the tier was sized for never arrived.
	shadowBlocks   *prometheus.GaugeVec
	shadowCapacity *prometheus.GaugeVec
	// shadowPairs is the VALIDATION matrix: for every seated return, the PREDICTED
	// resident fraction and the MEASURED cached/prompt, each binned to ten parts (bins
	// 0..10). The 11x11 matrix is what an AUC is computed from at analysis time (bin i
	// vs bin j ordering, summed); a single counter or a mean could not express it, and a
	// histogram per side would lose the JOINT distribution the AUC needs. Cardinality
	// is bounded by construction (11 x 11 per engine).
	shadowPairs *prometheus.CounterVec

	known map[string]struct{}
}

// subsystem namespaces every series (llm_d_epp_group_*).
const subsystem = "llm_d_epp_group"

var engineLabels = []string{"engine"}

func newGroupMetrics(sched *Scheduler) *groupMetrics {
	return &groupMetrics{sched: sched, known: map[string]struct{}{}}
}

func (g *groupMetrics) register(reg prometheus.Registerer) error {
	g.seats = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "seats",
		Help: "seated members (residencies) on this engine"}, engineLabels)
	g.uniqueBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "unique_blocks",
		Help: "distinct blocks the engine's index holds for this policy"}, engineLabels)
	g.neBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "ne_blocks",
		Help: "distinct blocks charged to this engine's budget: every seated member's set, exiting members included until they drain"}, engineLabels)
	g.budgetBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "budget_blocks",
		Help: "seat budget in blocks (groupBudgetFraction x this engine's num_gpu_blocks; at sizing gpu-plus-tier, of num_gpu_blocks + the CPU tier's blocks)"}, engineLabels)
	g.waiters = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters",
		Help: "waiting heads ACCOUNTED to this engine (0 by construction: the queue is shared; see waiters_shared)"}, engineLabels)
	g.coldInFlight = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_inflight_tokens",
		Help: "outstanding cold-entry charge in tokens against this engine's stagger cap"}, engineLabels)
	g.coldCap = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_cap_tokens",
		Help: "DERIVED cold-entry stagger cap in tokens: floor((1-groupBudgetFraction) x this engine's num_gpu_blocks) x blockSizeTokens"}, engineLabels)
	g.capacity = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "num_gpu_blocks",
		Help: "measured KV capacity in blocks for this engine"}, engineLabels)
	g.waitersShared = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters_shared",
		Help: "heads waiting in the ONE shared admission queue"})
	g.queueViewAge = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "queue_view_age_seconds",
		Help: "age of the queue-head snapshot the 'has a queued head' evidence comes from, on the decision clock (-1 = no snapshot yet, always > idleSeconds = the evidence is untrustworthy)"})
	g.queueViewUntrusted = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "queue_view_untrusted_passes_total",
		Help: "budget-enforcement passes whose victim choice ran with a queue view older than idleSeconds (the C8 guard skipped the head-presence classes)"})
	g.eventRingDropped = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "event_ring_dropped_total",
		Help: "SEAT/EXIT events the bounded ring has wrapped past (the DumpState tail no longer covers the whole window; the stdout event stream still does)"})
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "member exits by reason"}, []string{"engine", "reason"})
	g.residency = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "residency_turns",
		Help: "turns per residency at exit", Buckets: prometheus.LinearBuckets(1, 1, 20)}, engineLabels)
	g.entryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help: "newcomer seat waits", Buckets: prometheus.ExponentialBuckets(0.1, 2, 14)}, engineLabels)
	g.reentryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "reentry_wait_seconds",
		Help: "re-entering member seat waits", Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, engineLabels)
	g.refillShare = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "refill_share",
		Help: "uncached prompt tokens / total prompt tokens over the run"})
	g.served = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "dispatches served from this engine's seats"}, engineLabels)
	g.fcfs = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "fcfs_dispatches_total",
		Help: "pre-capacity FCFS pass-through dispatches (label: nocapacity)"}, engineLabels)
	g.capacityMissing = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_missing",
		Help: "1 while this engine's num_gpu_blocks is still unknown past capacityLivenessSeconds"}, engineLabels)
	g.capacityLearned = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "capacity_learned_total",
		Help: "how each engine's capacity was acquired"}, []string{"via", "engine"})
	g.safetyFired = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "safety_fired",
		Help: "1 while this engine's memory-only safety net is firing (measured kv >= groupBudgetFraction)"}, engineLabels)
	g.safetyFires = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "safety_fires_total",
		Help: "safety net fire EVENTS per engine"}, engineLabels)
	g.safetyFiredSeconds = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "safety_fired_seconds_total",
		Help: "seconds this engine's safety net has been firing"}, engineLabels)
	g.safetySkips = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "safety_skips_total",
		Help: "decisions the safety net held back (dispatch or seat attempts) on this engine"}, engineLabels)
	g.engineKV = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "engine_kv_usage",
		Help: "the engine's scraped kv_cache_usage (the safety net's input)"}, engineLabels)
	g.engineWaiting = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "engine_waiting_queue",
		Help: "the engine's scraped vLLM waiting queue depth (REPORTING only: not part of the net)"}, engineLabels)
	g.poolScore = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "detector_pool_score",
		Help: "the pool saturation this plugin handed the flow controller as the saturation detector (0 = not saturated)"})
	g.staleSamples = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "detector_stale_samples_total",
		Help: "per-engine samples the detector excluded because the metrics were not refreshed"}, engineLabels)
	g.capacityHeterogeneous = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_heterogeneous",
		Help: "1 when an engine's num_gpu_blocks deviates >1% from the first measured engine"})
	g.filterDecisions = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "filter_decisions_total",
		Help: "placement Filter outcomes (pinned | passthrough-*)"}, []string{"decision"})
	g.affinityViolations = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "affinity_violations_total",
		Help: "requests that landed on an engine other than the member's seat"}, []string{"from", "to"})
	g.rebinds = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "rebinds_total",
		Help: "seat moves repaired by the observed endpoint winning"})
	g.engineGone = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gone_total",
		Help: "engines torn down by an endpoint delete"})
	g.coldValveHeld = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "cold_valve_held_total",
		Help: "heads the cold-entry valve held back (the budget would have admitted them; they WAIT, nothing is evicted)"}, engineLabels)
	g.shadowBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "tier_shadow_blocks",
		Help: "blocks this engine's CPU-tier SHADOW believes are stored in the offload tier (0 = no tier configured)"}, engineLabels)
	g.shadowCapacity = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "tier_shadow_capacity_blocks",
		Help: "the tier shadow's capacity in blocks (cpuTierBytes / kvBlockBytes), the deployment's tier size"}, engineLabels)
	g.shadowPairs = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "tier_shadow_pairs_total",
		Help: "seated returns with a SHADOW-VALIDATION pair: predicted resident fraction and measured cached/prompt, each in ten bins (0..10) — the AUC's input"},
		[]string{"engine", "pred_bin", "meas_bin"})

	for _, c := range []prometheus.Collector{g.seats, g.uniqueBlocks, g.neBlocks, g.budgetBlocks,
		g.waiters, g.coldInFlight, g.coldCap, g.capacity, g.waitersShared, g.exits, g.residency, g.entryWait,
		g.reentryWait, g.refillShare, g.served, g.fcfs, g.capacityMissing, g.capacityLearned,
		g.safetyFired, g.safetyFires, g.safetyFiredSeconds, g.safetySkips, g.engineKV, g.engineWaiting,
		g.poolScore, g.staleSamples, g.capacityHeterogeneous, g.filterDecisions, g.affinityViolations, g.rebinds, g.engineGone,
		g.coldValveHeld, g.queueViewAge, g.queueViewUntrusted, g.eventRingDropped,
		g.shadowBlocks, g.shadowCapacity, g.shadowPairs} {
		if err := reg.Register(c); err != nil {
			// A SECOND instantiation in one process (a config reload, or a second
			// plugin config in the same test binary) must not fail startup: the series
			// exist and belong to the first instance.
			if _, dup := err.(prometheus.AlreadyRegisteredError); dup {
				fmt.Printf("group-scheduler: WARNING metrics already registered in this process — this instance will not emit its own series (one group scheduler per process is the supported configuration)\n")
				continue
			}
			return err
		}
	}
	return nil
}

// Refresh pushes a scheduler snapshot into the gauges (the ticker calls it every
// second; the counters and histograms are observed at their events by the
// scheduler).
func (g *groupMetrics) Refresh(poolScoreBits uint64) {
	if g == nil || g.seats == nil {
		return
	}
	snap := g.sched.SnapshotLite()
	loads := map[string]SafetyState{}
	for _, st := range g.sched.SafetyStates() {
		loads[engineLabel(st.Engine)] = st
	}
	// PRE-INSTANTIATE the per-engine counter series for every known engine. A
	// *CounterVec has no series until its first observation, so before any dispatch a
	// scrape would carry no llm_d_epp_group_dispatches_total at all — and the deploy
	// pre-flight (which requires that series) failed on exactly that on the first
	// live multi-pod stack. A ZERO is a fact; an absent series is a question.
	seen := make(map[string]struct{}, len(snap.Engines))
	for label, es := range snap.Engines {
		seen[label] = struct{}{}
		g.known[label] = struct{}{}
		g.seats.WithLabelValues(label).Set(float64(es.Seats))
		g.uniqueBlocks.WithLabelValues(label).Set(float64(es.UniqueBlocks))
		g.neBlocks.WithLabelValues(label).Set(float64(es.NEBlocks))
		g.budgetBlocks.WithLabelValues(label).Set(float64(es.BudgetBlocks))
		g.capacity.WithLabelValues(label).Set(float64(es.NumBlocks))
		g.waiters.WithLabelValues(label).Set(float64(es.Waiters))
		g.coldInFlight.WithLabelValues(label).Set(float64(es.ColdInFlight))
		g.coldCap.WithLabelValues(label).Set(float64(es.ColdCapTokens))
		g.shadowBlocks.WithLabelValues(label).Set(float64(es.TierShadowBlocks))
		g.shadowCapacity.WithLabelValues(label).Set(float64(es.TierCapacityBlocks))
		g.served.WithLabelValues(label).Add(0)
		g.fcfs.WithLabelValues("nocapacity").Add(0)
		g.safetySkips.WithLabelValues(label).Add(0)
		for _, reason := range []string{"quantum-complete", "idle", "exiting", "engine-idle", "ttl",
			"exiting-drain", "budget-idle", "budget-noqueue", "budget-exiting", "engine-gone"} {
			g.exits.WithLabelValues(label, reason).Add(0)
		}
		if st, ok := loads[label]; ok {
			// ONE definition, TWO readers (D3): fired comes from the scheduler's own
			// verdict, never from the thresholds re-typed here.
			fired := 0.0
			if st.Fired {
				fired = 1
			}
			g.safetyFired.WithLabelValues(label).Set(fired)
			g.safetyFires.WithLabelValues(label).Add(0)
			g.safetyFiredSeconds.WithLabelValues(label).Add(0)
			g.engineKV.WithLabelValues(label).Set(st.KV)
			g.engineWaiting.WithLabelValues(label).Set(float64(st.Waiting))
		}
	}
	// an engine that disappeared must not keep its last gauge value
	for label := range g.known {
		if _, ok := seen[label]; ok {
			continue
		}
		delete(g.known, label)
		for _, vec := range []*prometheus.GaugeVec{g.seats, g.uniqueBlocks, g.neBlocks,
			g.budgetBlocks, g.capacity, g.waiters, g.coldInFlight, g.coldCap, g.safetyFired,
			g.engineKV, g.engineWaiting, g.shadowBlocks, g.shadowCapacity} {
			vec.DeleteLabelValues(label)
		}
	}
	g.waitersShared.Set(float64(snap.WaitersShared))
	g.queueViewAge.Set(snap.QueueViewAgeSeconds)
	if d := snap.QueueViewUntrusted - g.queueViewUntrustedLast; d > 0 {
		g.queueViewUntrusted.Add(float64(d))
		g.queueViewUntrustedLast = snap.QueueViewUntrusted
	}
	if d := snap.EventRingDropped - g.eventRingDroppedLast; d > 0 {
		g.eventRingDropped.Add(float64(d))
		g.eventRingDroppedLast = snap.EventRingDropped
	}
	g.poolScore.Set(math.Float64frombits(poolScoreBits))
	g.refillShare.Set(snap.RefillShare)
	if len(snap.HeterogeneousEngines) > 0 {
		g.capacityHeterogeneous.Set(1)
	} else {
		g.capacityHeterogeneous.Set(0)
	}
}

// observeSafety books a fire EVENT (a transition into firing) and the accumulated
// fired time. Called from the scheduler's settle hook.
func (g *groupMetrics) observeSafety(engine string, fires int64, firedSeconds float64, fired bool) {
	if g == nil || g.safetyFires == nil {
		return
	}
	label := engineLabel(engine)
	f := 0.0
	if fired {
		f = 1
	}
	g.safetyFired.WithLabelValues(label).Set(f)
	g.safetyFires.WithLabelValues(label).Add(float64(fires))
	g.safetyFiredSeconds.WithLabelValues(label).Add(firedSeconds)
}

// ObserveExit records a member exit (residency turns + the reason + engine).
func (g *groupMetrics) ObserveExit(engine, reason string, resTurns int) {
	if g == nil || g.exits == nil {
		return
	}
	g.exits.WithLabelValues(engine, reason).Inc()
	g.residency.WithLabelValues(engine).Observe(float64(resTurns))
}

// ObserveEntry records a newcomer seat wait.
func (g *groupMetrics) ObserveEntry(engine string, waitSeconds float64) {
	if g == nil || g.entryWait == nil {
		return
	}
	g.entryWait.WithLabelValues(engine).Observe(waitSeconds)
}

// ObserveReentry records a re-entering member's seat wait.
func (g *groupMetrics) ObserveReentry(engine string, waitSeconds float64) {
	if g == nil || g.reentryWait == nil {
		return
	}
	g.reentryWait.WithLabelValues(engine).Observe(waitSeconds)
}

// ObserveServed records a dispatch.
func (g *groupMetrics) ObserveServed(engine string) {
	if g == nil || g.served == nil {
		return
	}
	g.served.WithLabelValues(engine).Inc()
}

// ObserveAffinityViolation records a request that landed off its member's seat
// engine (and the rebind that repaired it).
func (g *groupMetrics) ObserveAffinityViolation(from, to string) {
	if g == nil || g.affinityViolations == nil {
		return
	}
	g.affinityViolations.WithLabelValues(from, to).Inc()
	g.rebinds.Inc()
}

// ObserveEngineGone records an engine torn down by an endpoint delete.
func (g *groupMetrics) ObserveEngineGone() {
	if g == nil || g.engineGone == nil {
		return
	}
	g.engineGone.Inc()
}

// ObserveFilterDecision books one placement-Filter outcome (nil-safe: a unit-mode
// run has no registrar and must still take the same code path).
func (g *groupMetrics) ObserveFilterDecision(decision string) {
	if g == nil || g.filterDecisions == nil {
		return
	}
	g.filterDecisions.WithLabelValues(decision).Inc()
}

// ObserveQuality learns how an engine's capacity arrived.
func (g *groupMetrics) ObserveCapacityLearned(via, engine string) {
	if g == nil || g.capacityLearned == nil {
		return
	}
	g.capacityLearned.WithLabelValues(via, engine).Inc()
}

// SetCapacityMissing raises/clears the per-engine capacity-liveness gauge.
func (g *groupMetrics) SetCapacityMissing(engine string, missing bool) {
	if g == nil || g.capacityMissing == nil {
		return
	}
	v := 0.0
	if missing {
		v = 1
	}
	g.capacityMissing.WithLabelValues(engine).Set(v)
}

// ObserveStaleSample books an engine sample the detector had to exclude.
func (g *groupMetrics) ObserveStaleSample(engine string) {
	if g == nil || g.staleSamples == nil {
		return
	}
	g.staleSamples.WithLabelValues(engineLabel(engine)).Inc()
}

// ObserveFCFS records a pre-capacity FCFS dispatch (label "nocapacity": B2 — a
// pre-capacity dispatch belongs to no engine and must not be booked to one).
func (g *groupMetrics) ObserveFCFS(label string) {
	if g == nil || g.fcfs == nil {
		return
	}
	g.fcfs.WithLabelValues(label).Inc()
}

// ObserveColdValveHeld counts a head the cold-entry valve held back.
func (g *groupMetrics) ObserveColdValveHeld(engine string) {
	if g == nil || g.coldValveHeld == nil {
		return
	}
	g.coldValveHeld.WithLabelValues(engineLabel(engine)).Inc()
}

// ObserveShadowValidation books ONE shadow-validation pair: the predicted resident
// fraction the tier shadow made at seat time and the measured cached/prompt the engine
// reported for the turn that seat was for. Both sides are binned to ten parts so the
// scrape carries the JOINT distribution — which is what an AUC needs and what two
// separate histograms cannot reconstruct.
func (g *groupMetrics) ObserveShadowValidation(engine string, predicted, measured float64) {
	if g == nil || g.shadowPairs == nil {
		return
	}
	g.shadowPairs.WithLabelValues(engine, fracBin(predicted), fracBin(measured)).Inc()
}

// fracBin is the shared binning for the validation matrix: a fraction in [0,1] to one
// of eleven labels (0..10), saturating outside the range rather than panicking on a
// measured value above 1 (an engine may report cached > prompt under a broken counter).
func fracBin(f float64) string {
	switch {
	case f <= 0:
		return "0"
	case f >= 1:
		return "10"
	}
	return fmt.Sprintf("%d", int(f*10))
}
