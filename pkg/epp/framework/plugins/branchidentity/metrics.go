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

package branchidentity

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const metricsSubsystem = "llm_d_epp_branch"

// branchMetrics is the legibility set for the derivation: how often a request
// continued a branch, forked one, or found nothing, how big the table is, how
// much the flattening and the resolve cost on the (untimed, pre-admission)
// request path, and how often the derivation was bypassed.
type branchMetrics struct {
	matched    prometheus.Counter
	forked     prometheus.Counter
	fresh      prometheus.Counter
	sticky     prometheus.Counter
	retried    prometheus.Counter
	evicted    prometheus.Counter
	unflatten  prometheus.Counter
	overridden prometheus.Counter
	records    prometheus.Gauge
	flattenNs  prometheus.Histogram
	resolveNs  prometheus.Histogram
}

func newBranchMetrics() *branchMetrics { return &branchMetrics{} }

func (m *branchMetrics) register(reg prometheus.Registerer) error {
	m.matched = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "matched_total",
		Help: "requests that continued an existing branch (same fairness id as the branch's previous turn)"})
	m.forked = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "forks_total",
		Help: "requests that extended a BUSY branch tip and therefore forked a new branch"})
	m.fresh = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "new_total",
		Help: "requests that matched no branch tip and started an unrelated branch"})
	m.sticky = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "sticky_busy_total",
		Help: "requests whose continuation was proven by content (H1 reply guard) despite a busy branch"})
	m.retried = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "retry_total",
		Help: "requests byte-identical to a tip with nothing in flight, treated as a client retry of a still-queued turn (S2)"})
	m.evicted = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "evicted_total",
		Help: "branches removed by the table cap (the only eviction path; there is no idle GC)"})
	m.unflatten = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "unflattened_total",
		Help: "requests whose body has no canonical stream; the fairness id falls back to the header/agent identity"})
	m.overridden = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: metricsSubsystem, Name: "explicit_override_total",
		Help: "requests that carried an explicit fairness id, so the derived branch did not become the queue key (a run whose whole point is the derivation must keep this at 0)"})
	m.records = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: metricsSubsystem, Name: "records",
		Help: "live branches in the table"})
	m.flattenNs = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: metricsSubsystem, Name: "flatten_ns",
		Help:    "identity flattening + block chaining time per request (the pre-admission cost of derivation)",
		Buckets: prometheus.ExponentialBuckets(2000, 4, 8)})
	m.resolveNs = prometheus.NewHistogram(prometheus.HistogramOpts{Subsystem: metricsSubsystem, Name: "resolve_ns",
		Help:    "branch table resolve time per request (under the table lock: match + mint)",
		Buckets: prometheus.ExponentialBuckets(500, 4, 8)})
	for _, c := range []prometheus.Collector{
		m.matched, m.forked, m.fresh, m.sticky, m.retried, m.evicted, m.unflatten,
		m.overridden, m.records, m.flattenNs, m.resolveNs,
	} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	return nil
}

func (m *branchMetrics) observeFlatten(d time.Duration) {
	if m.flattenNs != nil {
		m.flattenNs.Observe(float64(d.Nanoseconds()))
	}
}

func (m *branchMetrics) observeResolveLatency(d time.Duration) {
	if m.resolveNs != nil && d >= 0 {
		m.resolveNs.Observe(float64(d.Nanoseconds()))
	}
}

func (m *branchMetrics) observeResolve(res ResolveResult) {
	switch res.Outcome {
	case OutcomeMatched:
		if m.matched != nil {
			m.matched.Inc()
		}
	case OutcomeForked:
		if m.forked != nil {
			m.forked.Inc()
		}
	case OutcomeNew:
		if m.fresh != nil {
			m.fresh.Inc()
		}
	}
	if res.StickyBusy && m.sticky != nil {
		m.sticky.Inc()
	}
	if res.Retry && m.retried != nil {
		m.retried.Inc()
	}
}

func (m *branchMetrics) observeUnflattened() {
	if m.unflatten != nil {
		m.unflatten.Inc()
	}
}

func (m *branchMetrics) observeExplicitOverride() {
	if m.overridden != nil {
		m.overridden.Inc()
	}
}

func (m *branchMetrics) addEvicted(n int) {
	if m.evicted != nil && n > 0 {
		m.evicted.Add(float64(n))
	}
}

func (m *branchMetrics) setRecords(n int) {
	if m.records != nil {
		m.records.Set(float64(n))
	}
}
