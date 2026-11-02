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
//	                  · queue_view_untrusted_passes_total
//
// EVERY per-engine series carries an `engine` label, whose value is the endpoint id:
// `sum(llm_d_epp_group_seats)` is the whole-policy seat count with one engine and with N.
type groupMetrics struct {
	sched *Scheduler

	seats        *prometheus.GaugeVec
	neBlocks     *prometheus.GaugeVec
	budgetBlocks *prometheus.GaugeVec
	// The EXITING ledger: the enforcement view's inputs, per engine. (iii) made the
	// used side include members that are already leaving, so "how much of it is
	// leaving" is a first-class number: without these three, an engine that is busy
	// and an engine whose drains have stopped look the same in every other series.
	exitingBranches *prometheus.GaugeVec
	exitingBlocks   *prometheus.GaugeVec
	exitingOldest   *prometheus.GaugeVec
	coldInFlight    *prometheus.GaugeVec
	coldCap         *prometheus.GaugeVec
	// waitersShared is the SHARED waiter queue's size (one number, no engine label,
	// because a waiter belongs to no engine until one seats it).
	waitersShared prometheus.Gauge
	// queueViewUntrusted counts the budget-enforcement passes that ran with a queue
	// view too old to trust (the C8 guard skipped the head-presence classes in exactly
	// those passes, so without it the guard's own input is invisible). It is exported
	// as the DELTA of the scheduler's cumulative count: one source of truth.
	queueViewUntrusted     prometheus.Counter
	queueViewUntrustedLast int64
	exits                  *prometheus.CounterVec
	// entryWait is ONE histogram for BOTH seat waits (newcomer and re-entry), split by
	// `kind`: they answer the same question (how long did admission make this branch
	// wait), and two names meant two bucket sets and two queries.
	entryWait *prometheus.HistogramVec
	served    *prometheus.CounterVec

	capacityMissing *prometheus.GaugeVec
	capacityLearned *prometheus.CounterVec

	// affinityViolations counts requests that landed off their member's seat engine.
	affinityViolations *prometheus.CounterVec
	rebinds            prometheus.Counter
	engineGone         prometheus.Counter
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
		Help: "seated members (residencies) on this engine"}, engineLabels)
	g.exitingBranches = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_branches",
		Help: "branches on this engine that have been marked exiting and still hold their seat until their in-flight requests drain"}, engineLabels)
	g.exitingBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_blocks",
		Help: "distinct blocks of this engine's ne_blocks that belong to exiting members: the part of the used side that is already leaving (admission still refuses it, enforcement no longer evicts for it)"}, engineLabels)
	g.exitingOldest = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "exiting_oldest_age_seconds",
		Help: "how long this engine's OLDEST exiting branch has been on its way out (a value that only grows is a drain that is not happening)"}, engineLabels)
	g.neBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "ne_blocks",
		Help: "distinct blocks charged to this engine's budget: every seated member's set, exiting members included until they drain"}, engineLabels)
	g.budgetBlocks = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "budget_blocks",
		Help: "seat budget in blocks (groupBudgetFraction x this engine's num_gpu_blocks)"}, engineLabels)
	g.coldInFlight = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_inflight_tokens",
		Help: "outstanding cold-entry charge in tokens against this engine's stagger cap"}, engineLabels)
	g.coldCap = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "cold_cap_tokens",
		Help: "DERIVED cold-entry stagger cap in tokens: floor((1-groupBudgetFraction) x this engine's num_gpu_blocks) x blockSizeTokens"}, engineLabels)
	g.waitersShared = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters_shared",
		Help: "heads waiting in the ONE shared admission queue"})
	g.queueViewUntrusted = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "queue_view_untrusted_passes_total",
		Help: "budget-enforcement passes whose victim choice ran with a queue view older than idleSeconds (the C8 guard skipped the head-presence classes)"})
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "member exits by reason"}, []string{"engine", "reason"})
	g.entryWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help:    "seat waits by kind (new = a branch's first turn, reentry = a returning branch): how long admission made it wait, TOP BUCKET 13,107 s so nothing lands only in +Inf",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"engine", "kind"})
	g.served = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "dispatches served from this engine's seats"}, engineLabels)
	g.capacityMissing = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: subsystem, Name: "capacity_missing",
		Help: "1 while this engine's num_gpu_blocks is still unknown past capacityLivenessSeconds"}, engineLabels)
	g.capacityLearned = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "capacity_learned_total",
		Help: "how each engine's capacity was acquired"}, []string{"via", "engine"})
	g.affinityViolations = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "affinity_violations_total",
		Help: "requests that landed on an engine other than the member's seat (an invariant: must be 0)"}, []string{"from", "to"})
	g.rebinds = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "rebinds_total",
		Help: "seat moves repaired by the observed endpoint winning"})
	g.engineGone = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "engine_gone_total",
		Help: "engines torn down by an endpoint delete"})
	g.coldValveHeld = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "cold_valve_held_total",
		Help: "heads the cold-entry valve held back (the budget would have admitted them; they WAIT, nothing is evicted)"}, engineLabels)

	for _, c := range []prometheus.Collector{g.seats, g.neBlocks, g.budgetBlocks,
		g.exitingBranches, g.exitingBlocks, g.exitingOldest,
		g.coldInFlight, g.coldCap, g.waitersShared,
		g.queueViewUntrusted, g.exits, g.entryWait, g.served,
		g.capacityMissing, g.capacityLearned,
		g.affinityViolations, g.rebinds, g.engineGone, g.coldValveHeld,
	} {
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
		for _, reason := range []string{"quantum-complete", "idle", "exiting", "engine-idle", "ttl",
			"exiting-drain", "budget-idle", "budget-noqueue", "budget-exiting", "engine-gone"} {
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
	if d := snap.QueueViewUntrusted - g.queueViewUntrustedLast; d > 0 {
		g.queueViewUntrusted.Add(float64(d))
		g.queueViewUntrustedLast = snap.QueueViewUntrusted
	}
}

// ObserveExit records a member exit (the reason + engine).
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

// ObserveColdValveHeld counts a head the cold-entry valve held back.
func (g *groupMetrics) ObserveColdValveHeld(engine string) {
	if g == nil || g.coldValveHeld == nil {
		return
	}
	g.coldValveHeld.WithLabelValues(engineLabel(engine)).Inc()
}
