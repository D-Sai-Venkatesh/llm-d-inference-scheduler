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
type groupMetrics struct {
	seats        prometheus.Gauge
	uniqueBlocks prometheus.Gauge
	neBlocks     prometheus.Gauge
	budgetBlocks prometheus.Gauge
	waiters      prometheus.Gauge
	exits        *prometheus.CounterVec
	residency    prometheus.Histogram
	entryWait    prometheus.Histogram
	refillShare  prometheus.Gauge
	served       prometheus.Counter

	sched *Scheduler
}

func newGroupMetrics(sched *Scheduler) *groupMetrics {
	return &groupMetrics{sched: sched}
}

func (g *groupMetrics) register(reg prometheus.Registerer) error {
	const subsystem = "llm_d_epp_group"
	g.seats = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "seats",
		Help: "group members (seated sessions)"})
	g.uniqueBlocks = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "unique_blocks",
		Help: "unique blocks held by all seated members"})
	g.neBlocks = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "ne_blocks",
		Help: "unique blocks over non-exiting members (the budget counter)"})
	g.budgetBlocks = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "budget_blocks",
		Help: "the group budget in blocks (0.95 x num_gpu_blocks)"})
	g.waiters = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "waiters",
		Help: "non-members with waiting requests"})
	g.exits = prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: subsystem, Name: "exits_total",
		Help: "member exits by reason"}, []string{"reason"})
	g.residency = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: subsystem, Name: "residency_turns",
		Help: "turns per residency at exit", Buckets: prometheus.LinearBuckets(1, 1, 20)})
	g.entryWait = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: subsystem, Name: "entry_wait_seconds",
		Help: "newcomer seat waits", Buckets: prometheus.ExponentialBuckets(0.1, 2, 14)})
	g.refillShare = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: subsystem, Name: "refill_share",
		Help: "uncached share of settled prompt tokens (the refill cost)"})
	g.served = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: subsystem, Name: "dispatches_total",
		Help: "requests dispatched to members"})
	fcfsCounter = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: "llm_d_epp_group", Name: "fcfs_dispatches_total",
		Help: "pre-capacity FCFS pass-through dispatches"})
	for _, c := range []prometheus.Collector{g.seats, g.uniqueBlocks, g.neBlocks, g.budgetBlocks,
		g.waiters, g.exits, g.residency, g.entryWait, g.refillShare, g.served, fcfsCounter} {
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
	g.seats.Set(float64(snap.Seats))
	g.uniqueBlocks.Set(float64(snap.UniqueBlocks))
	g.neBlocks.Set(float64(snap.NEBlocks))
	g.budgetBlocks.Set(float64(snap.BudgetBlocks))
	g.waiters.Set(float64(snap.Waiters))
	g.refillShare.Set(snap.RefillShare)
	g.served.Add(0) // counter; the scheduler path adds directly below
}

// ObserveExit records a member exit (residency turns + the reason).
func (g *groupMetrics) ObserveExit(reason string, resTurns int) {
	if g.exits == nil {
		return
	}
	g.exits.WithLabelValues(reason).Inc()
	g.residency.Observe(float64(resTurns))
}

// ObserveEntry records a newcomer seat wait.
func (g *groupMetrics) ObserveEntry(waitSeconds float64) {
	if g.entryWait == nil {
		return
	}
	g.entryWait.Observe(waitSeconds)
}

// ObserveServed records a dispatch.
func (g *groupMetrics) ObserveServed() {
	if g.served != nil {
		g.served.Inc()
	}
}

// fcfsCounter is the pre-capacity pass-through counter.
var fcfsCounter prometheus.Counter

// ObserveFCFS records a pre-capacity FCFS dispatch.
func (g *groupMetrics) ObserveFCFS() {
	if fcfsCounter != nil {
		fcfsCounter.Inc()
	}
}
