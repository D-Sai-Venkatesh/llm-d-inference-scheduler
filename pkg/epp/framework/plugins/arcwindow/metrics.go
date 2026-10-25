/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package arcwindow

import (
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
)

// arcMetrics holds the direct collectors; everything else flows through
// the ledger collector at scrape time (the sibling's pattern, B5).
type arcMetrics struct {
	ledger       *arcLedgerCollector
	waiting      prometheus.Gauge
	turnHitRatio prometheus.Histogram
}

func newArcMetrics(mgr *manager) *arcMetrics {
	return &arcMetrics{
		ledger: &arcLedgerCollector{mgr: mgr},
		waiting: prometheus.NewGauge(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "arc_waiting_turns",
			Help:      metricsutil.HelpMsgWithStability("Waiting turns seen at the last Pick.", compbasemetrics.ALPHA),
		}),
		turnHitRatio: prometheus.NewHistogram(prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "arc_turn_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Per-turn prefix hit ratio (cached/prompt tokens).", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01},
		}),
	}
}

func (a *arcMetrics) register(reg prometheus.Registerer) error {
	if err := reg.Register(a.ledger); err != nil {
		return err
	}
	for _, c := range []prometheus.Collector{a.waiting, a.turnHitRatio} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	return nil
}

// arcLedgerCollector computes the arc's gauges and counters from the
// manager's snapshot at scrape time.
type arcLedgerCollector struct {
	mgr *manager
}

func (c *arcLedgerCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range c.descs() {
		ch <- d
	}
}

func (c *arcLedgerCollector) descs() []*prometheus.Desc {
	fq := func(name, help string, labels []string) *prometheus.Desc {
		return prometheus.NewDesc(prometheus.BuildFQName("", eppmetrics.LLMDRouterEndpointPickerSubsystem, name),
			metricsutil.HelpMsgWithStability(help, compbasemetrics.ALPHA), labels, nil)
	}
	return []*prometheus.Desc{
		fq("arc_members", "Sessions currently covered by the arc.", nil),
		fq("arc_member_sum_tokens", "Sum of member footprints (committed + inflight).", nil),
		fq("arc_width_members", "The arc's width target (count).", nil),
		fq("arc_idle_pool_tokens", "Measured idle KV pool (1-KV)*C.", nil),
		fq("arc_budget_tokens", "The arc's token budget theta*C.", nil),
		fq("arc_ring_size", "Live sessions on the ring.", nil),
		fq("arc_demand_live", "Sessions with activity in the last 5 minutes (the pacing population).", nil),
		fq("arc_holes", "Persistent drop holes currently inside the arc's span.", nil),
		fq("arc_epochs_total", "Epochs started.", nil),
		fq("arc_dispatches_total", "Dispatches made by the policy (members, guests, off-switch, anon).", nil),
		fq("arc_membership_growth_total", "Membership churn: sessions newly covered by the arc (not arc motion).", nil),
		fq("arc_uncovered_serves_total", "Dispatches to uncovered sessions (the hybrid path).", nil),
		fq("arc_immediate_redrop_total", "Sum-fit drops (the fit-treadmill detector).", nil),
		fq("arc_covered_never_served_total", "Covered-then-dropped sessions never dispatched (the vacuous-bound detector).", nil),
		fq("arc_advance_blocked_no_room_total", "Advances that could not make room (must be 0: coverage never blocks).", nil),
		fq("arc_swept_sessions_total", "Sessions removed from the ring by the stale sweep.", nil),
		fq("arc_advances_total", "Arc advances by reason (heartbeat, ddemand — arc motion only).", []string{"reason"}),
		fq("arc_cycle_seconds", "Time between arc advances (the heartbeat's period — the fairness-cycle gauge).", nil),
		fq("arc_drops_total", "Arc membership drops by policy.", []string{"policy"}),
	}
}

func (c *arcLedgerCollector) Collect(ch chan<- prometheus.Metric) {
	snap := c.mgr.arcSnapshot()
	d := c.descs()
	ch <- prometheus.MustNewConstMetric(d[0], prometheus.GaugeValue, float64(snap.members))
	ch <- prometheus.MustNewConstMetric(d[1], prometheus.GaugeValue, snap.memberSum)
	ch <- prometheus.MustNewConstMetric(d[2], prometheus.GaugeValue, float64(snap.width))
	if snap.pool >= 0 {
		ch <- prometheus.MustNewConstMetric(d[3], prometheus.GaugeValue, snap.pool)
	}
	ch <- prometheus.MustNewConstMetric(d[4], prometheus.GaugeValue, snap.budget)
	ch <- prometheus.MustNewConstMetric(d[5], prometheus.GaugeValue, float64(snap.ringSize))
	ch <- prometheus.MustNewConstMetric(d[6], prometheus.GaugeValue, float64(snap.demandLive))
	ch <- prometheus.MustNewConstMetric(d[7], prometheus.GaugeValue, float64(snap.holes))
	ch <- prometheus.MustNewConstMetric(d[8], prometheus.CounterValue, float64(snap.epochs))
	ch <- prometheus.MustNewConstMetric(d[9], prometheus.CounterValue, float64(snap.served))
	ch <- prometheus.MustNewConstMetric(d[10], prometheus.CounterValue, float64(snap.membershipGrowth))
	ch <- prometheus.MustNewConstMetric(d[11], prometheus.CounterValue, float64(snap.uncoveredServed))
	ch <- prometheus.MustNewConstMetric(d[12], prometheus.CounterValue, float64(snap.immediateRedrop))
	ch <- prometheus.MustNewConstMetric(d[13], prometheus.CounterValue, float64(snap.coveredNeverServed))
	ch <- prometheus.MustNewConstMetric(d[14], prometheus.CounterValue, float64(snap.advanceBlocked))
	ch <- prometheus.MustNewConstMetric(d[15], prometheus.CounterValue, float64(snap.swept))
	if snap.cycleSeconds > 0 {
		ch <- prometheus.MustNewConstMetric(d[17], prometheus.GaugeValue, snap.cycleSeconds)
	}
	for reason, n := range snap.advances {
		ch <- prometheus.MustNewConstMetric(d[16], prometheus.CounterValue, float64(n), reason)
	}
	for policy, n := range snap.drops {
		ch <- prometheus.MustNewConstMetric(d[18], prometheus.CounterValue, float64(n), policy)
	}
}
