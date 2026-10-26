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
	for _, c := range []prometheus.Collector{g.seats, g.uniqueBlocks, g.neBlocks, g.budgetBlocks,
		g.waiters, g.exits, g.residency, g.entryWait, g.refillShare, g.served} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	return nil
}

// Describe and Collect satisfy prometheus.Collector lazily via the
// scheduler snapshot; the register path above uses the concrete
// collectors, and the snapshot refresh runs on scrape via Collect.
