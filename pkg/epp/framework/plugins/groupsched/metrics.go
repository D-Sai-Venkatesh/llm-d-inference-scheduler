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
	"context"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groupMetrics exports the MINIMAL set an operator needs to see liveness and fairness
// or to diagnose a known failure class — nineteen names, and nothing else. The test for
// every one of them: would its absence cost an operator a decision or a diagnosis?
//
//	admission state   seats · ne_blocks · budget_blocks · waiters_shared
//	drain / cascade   exiting_branches · exiting_blocks · exiting_oldest_age_seconds
//	valve             cold_inflight_tokens · cold_cap_tokens · cold_valve_held_total
//	startup           capacity_missing · capacity_learned_total
//	decisions         dispatches_total · exits_total{reason}
//	fairness          entry_wait_seconds{kind=new|reentry}   (ONE histogram)
//	lifecycle         engine_gone_total · rebinds_total · affinity_violations_total
//	integrity         anomalies_total{kind}   (every kind should be 0)
//
// EVERY per-engine series carries an `engine` label, whose value is the endpoint id:
// `sum(llm_d_epp_group_seats)` is the whole-policy seat count with one engine and with N.
type groupMetrics struct {
	sched *Scheduler

	seats        *prometheus.GaugeVec
	neBlocks     *prometheus.GaugeVec
	budgetBlocks *prometheus.GaugeVec
	// The EXITING ledger: the enforcement view's inputs, per engine. The used side includes
	// branches that are already leaving, so "how much of it is leaving" is a first-class
	// number: without these three, an engine that is busy and an engine whose drains have
	// stopped look the same in every other series.
	exitingBranches *prometheus.GaugeVec
	exitingBlocks   *prometheus.GaugeVec
	exitingOldest   *prometheus.GaugeVec
	coldInFlight    *prometheus.GaugeVec
	coldCap         *prometheus.GaugeVec
	// waitersShared is the SHARED waiter queue's size (one number, no engine label,
	// because a waiter belongs to no engine until one seats it).
	waitersShared prometheus.Gauge
	exits         *prometheus.CounterVec
	// entryWait is ONE histogram for BOTH seat waits (newcomer and re-entry), split by
	// `kind`: they answer the same question (how long did admission make this branch
	// wait), and two names meant two bucket sets and two queries.
	entryWait *prometheus.HistogramVec
	served    *prometheus.CounterVec

	capacityMissing *prometheus.GaugeVec
	capacityLearned *prometheus.CounterVec

	// affinityViolations counts requests that landed off their branch's seat engine.
	affinityViolations *prometheus.CounterVec
	rebinds            prometheus.Counter
	engineGone         prometheus.Counter
	// anomalies is the ONE kept counter for the events the policy cannot act on, by kind
	// (anomalyKinds). Every kind is SHOULD-BE-ZERO: a lost render, an unnameable request, a
	// settle for a request nothing tracks. One counter, one question: "is the accounting
	// intact?"
	anomalies *prometheus.CounterVec
	// coldValveHeld counts heads the cold-entry VALVE held back: the budget would have
	// admitted the head, the valve refused it, so the head WAITED for the engine's
	// prefill slot. Nothing is evicted for it (the sizing signal: what the stagger
	// costs in waiting).
	coldValveHeld *prometheus.CounterVec

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
		Help: "seated branches (residencies) on this engine"}, engineLabels)
	g.exitingBranches = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_branches",
		Help: "branches on this engine that have been marked exiting and still hold their seat until their in-flight requests drain"}, engineLabels)
	g.exitingBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_blocks",
		Help: "distinct blocks of this engine's ne_blocks that belong to exiting branches: the part of the used side that is already leaving (admission still refuses it, enforcement no longer evicts for it)"}, engineLabels)
	g.exitingOldest = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_oldest_age_seconds",
		Help: "how long this engine's OLDEST exiting branch has been on its way out (a value that only grows is a drain that is not happening)"}, engineLabels)
	g.neBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "ne_blocks",
		Help: "distinct blocks charged to this engine's budget: every seated branch's set, exiting branches included until they drain"}, engineLabels)
	g.budgetBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "budget_blocks",
		Help: "seat budget in blocks (groupBudgetFraction x this engine's num_gpu_blocks)"}, engineLabels)
	g.coldInFlight = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_inflight_tokens",
		Help: "outstanding cold-entry charge in tokens against this engine's stagger cap"}, engineLabels)
	g.coldCap = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_cap_tokens",
		Help: "DERIVED cold-entry stagger cap in tokens: floor((1-groupBudgetFraction) x this engine's num_gpu_blocks) x blockSizeTokens"}, engineLabels)
	g.waitersShared = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters_shared",
		Help: "heads waiting in the ONE shared admission queue"})
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "branch exits by reason"}, []string{"engine", "reason"})
	g.entryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help:    "seat waits by kind (new = a branch's first turn, reentry = a returning branch): how long admission made it wait, TOP BUCKET 13,107 s so nothing lands only in +Inf",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"engine", "kind"})
	g.served = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "dispatches served: per engine, by that engine's seats; under engine=\"nocapacity\", the pre-capacity pass-through (no engine had a measured num_gpu_blocks yet, so nothing was seated and the request was served unseated)"}, engineLabels)
	g.capacityMissing = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_missing",
		Help: "1 while this engine's num_gpu_blocks is still unknown past capacityLivenessSeconds"}, engineLabels)
	g.capacityLearned = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "capacity_learned_total",
		Help: "how each engine's capacity was acquired"}, []string{"via", "engine"})
	g.affinityViolations = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "affinity_violations_total",
		Help: "requests that landed on an engine other than the branch's seat (an invariant: must be 0)"}, []string{"from", "to"})
	g.rebinds = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "rebinds_total",
		Help: "seat moves repaired by the observed endpoint winning"})
	g.engineGone = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gone_total",
		Help: "engines torn down by an endpoint delete"})
	g.anomalies = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "anomalies_total",
		Help: "should-be-zero events the policy could not act on, by kind: no_tokens_head = a Pick pass saw a branch head with no rendered tokens (a lost render; it is served last, unseated); no_request_id = a Pick released a request with no id (nothing can name it); stray_settle = a settle named a request nothing tracks (a late reap, a duplicate terminal callback, a branch that left). All three read 0 on a healthy stack"}, []string{"kind"})
	g.coldValveHeld = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "cold_valve_held_total",
		Help: "Pick passes in which the queue head was held by this engine's cold-entry valve (the budget would have admitted it; it WAITS, nothing is evicted). A PASS count, not a head count: Pick runs at ~1 kHz, so a head held for a second counts about a thousand times"}, engineLabels)

	for _, c := range []prometheus.Collector{g.seats, g.neBlocks, g.budgetBlocks,
		g.exitingBranches, g.exitingBlocks, g.exitingOldest,
		g.coldInFlight, g.coldCap, g.waitersShared,
		g.exits, g.entryWait, g.served,
		g.capacityMissing, g.capacityLearned,
		g.affinityViolations, g.rebinds, g.engineGone, g.coldValveHeld, g.anomalies,
	} {
		if err := reg.Register(c); err != nil {
			// A SECOND instantiation in one process (a config reload, or a second
			// plugin config in the same test binary) must not fail startup: the series
			// exist and belong to the first instance.
			if _, dup := err.(prometheus.AlreadyRegisteredError); dup {
				// One group scheduler per process is the supported configuration; a second
				// instance would double-register every series.
				log.FromContext(context.Background()).V(logutil.DEFAULT).Info("group-scheduler: WARNING metrics already registered in this process — this instance will not emit its own series")
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
func (g *groupMetrics) Refresh() {
	if g == nil || g.seats == nil {
		return
	}
	snap := g.sched.SnapshotLite()
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
		g.neBlocks.WithLabelValues(label).Set(float64(es.NEBlocks))
		g.budgetBlocks.WithLabelValues(label).Set(float64(es.BudgetBlocks))
		g.coldInFlight.WithLabelValues(label).Set(float64(es.ColdInFlight))
		g.coldCap.WithLabelValues(label).Set(float64(es.ColdCapTokens))
		g.exitingBranches.WithLabelValues(label).Set(float64(es.ExitingMembers))
		g.exitingBlocks.WithLabelValues(label).Set(float64(es.ExitingBlocks))
		g.exitingOldest.WithLabelValues(label).Set(es.ExitingOldestAgeS)
		g.served.WithLabelValues(label).Add(0)
		// One series per DEPARTURE reason. "exiting" is not one: the mark of a mid-flight
		// branch is a STATE (exitingBranches / exitingBlocks / exitingOldest), and booking it
		// as an exit too counted every mid-flight departure twice in this same series.
		// "budget-exiting" survives only as the name enforcement would use if a marked
		// branch could not be re-marked — it is never a departure either.
		for _, reason := range []string{"quantum-complete", "idle", "engine-idle",
			"exiting-drain", "budget-idle", "budget-noqueue", "engine-gone"} {
			g.exits.WithLabelValues(label, reason).Add(0)
		}
	}
	// an engine that disappeared must not keep its last gauge value
	for label := range g.known {
		if _, ok := seen[label]; ok {
			continue
		}
		delete(g.known, label)
		for _, vec := range []*prometheus.GaugeVec{g.seats, g.neBlocks,
			g.exitingBranches, g.exitingBlocks, g.exitingOldest,
			g.budgetBlocks, g.coldInFlight, g.coldCap} {
			vec.DeleteLabelValues(label)
		}
	}
	g.waitersShared.Set(float64(snap.WaitersShared))
	// the pre-capacity bucket exists from the first scrape: a zero is a fact, and this is the
	// series that says "no request was served before capacity arrived"
	g.served.WithLabelValues(nocapacityEngine).Add(0)
	// ONE zero series per anomaly kind: a zero is a fact, an absent series is a question (and
	// a should-be-zero count with no series is the shape of a bug nobody notices).
	for _, kind := range anomalyKinds {
		g.anomalies.WithLabelValues(kind).Add(0)
	}
}

// ObserveExit records a branch exit (the reason + engine).
func (g *groupMetrics) ObserveExit(engine, reason string) {
	if g == nil || g.exits == nil {
		return
	}
	g.exits.WithLabelValues(engine, reason).Inc()
}

// ObserveEntry records ONE seat wait, split by kind: "new" for a branch's first turn,
// "reentry" for a returning branch. Both were separate histograms; they answer the same
// question.
func (g *groupMetrics) ObserveEntry(engine, kind string, waitSeconds float64) {
	if g == nil || g.entryWait == nil {
		return
	}
	g.entryWait.WithLabelValues(engine, kind).Observe(waitSeconds)
}

// ObserveServed records a dispatch.
func (g *groupMetrics) ObserveServed(engine string) {
	if g == nil || g.served == nil {
		return
	}
	g.served.WithLabelValues(engine).Inc()
}

// ObserveAffinityViolation records a request that landed off the engine it was charged to.
// It is fired from BOTH sites — the PreRequest observation (which also rebinds) and the
// settle — so the series covers the whole violation class, not half of it.
func (g *groupMetrics) ObserveAffinityViolation(from, to string) {
	if g == nil || g.affinityViolations == nil {
		return
	}
	g.affinityViolations.WithLabelValues(from, to).Inc()
}

// ObserveRebind records the seat MOVE that repaired a violation. Separate from the
// violation itself: a violation found at the settle repairs nothing, so counting rebinds off
// the same event would overstate the repairs.
func (g *groupMetrics) ObserveRebind() {
	if g == nil || g.rebinds == nil {
		return
	}
	g.rebinds.Inc()
}

// ObserveEngineGone records an engine torn down by an endpoint delete.
func (g *groupMetrics) ObserveEngineGone() {
	if g == nil || g.engineGone == nil {
		return
	}
	g.engineGone.Inc()
}

// ObserveCapacityLearned records how an engine's capacity arrived (the acquisition path).
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

// ObserveAnomaly counts ONE should-be-zero event by kind (anomalyKinds).
func (g *groupMetrics) ObserveAnomaly(kind string) {
	if g == nil || g.anomalies == nil {
		return
	}
	g.anomalies.WithLabelValues(kind).Inc()
}

// ObserveColdValveHeld counts a head the cold-entry valve held back.
func (g *groupMetrics) ObserveColdValveHeld(engine string) {
	if g == nil || g.coldValveHeld == nil {
		return
	}
	g.coldValveHeld.WithLabelValues(engineLabel(engine)).Inc()
}
