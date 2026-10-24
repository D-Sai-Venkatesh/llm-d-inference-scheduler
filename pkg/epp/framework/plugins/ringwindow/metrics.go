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

package ringwindow

import (
	"errors"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// ringMetrics holds the plugin's Prometheus collectors. The gauges are
// collected from the manager at scrape time, so they stay current when no
// request arrives.
type ringMetrics struct {
	ledger                *ringLedgerCollector
	pickRotations         prometheus.Counter
	pickRecoveries        prometheus.Counter
	pickMemberServes      prometheus.Counter
	pickJoins             prometheus.Counter
	pickGuests            prometheus.Counter
	pickOversized         prometheus.Counter
	pickAnonymous         prometheus.Counter
	pickDelayedDispatches prometheus.Counter
	pickTransparent       prometheus.Counter
	pickGuestTokens       prometheus.Counter
	pickNoVictims         prometheus.Counter
	rotationInterval      prometheus.Histogram
	turnHitRatio          prometheus.Histogram
}

func newRingMetrics(mgr *ringManager) *ringMetrics {
	return &ringMetrics{
		ledger: &ringLedgerCollector{mgr: mgr},
		pickRotations: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_rotations_total",
			Help:      metricsutil.HelpMsgWithStability("Window rotations: a waiting outsider's D-clock expired, so it joined and was dispatched first.", compbasemetrics.ALPHA),
		}),
		pickRecoveries: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_member_recoveries_total",
			Help:      metricsutil.HelpMsgWithStability("Member dispatches promoted by their own expired slack (quiet-member recovery), ahead of the back-and-forth cursor.", compbasemetrics.ALPHA),
		}),
		pickMemberServes: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_member_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches of window members in back-and-forth order.", compbasemetrics.ALPHA),
		}),
		pickJoins: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_joins_total",
			Help:      metricsutil.HelpMsgWithStability("Head outsiders that joined the window without any demotion.", compbasemetrics.ALPHA),
		}),
		pickGuests: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_guest_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Never-idle guest dispatches: a non-fitting outsider ran while no member waited. The guest rate bounds the window's payoff.", compbasemetrics.ALPHA),
		}),
		pickOversized: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_oversized_guests_total",
			Help:      metricsutil.HelpMsgWithStability("Guests no subset of members could cover: served as non-members, window cleared.", compbasemetrics.ALPHA),
		}),
		pickAnonymous: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_anonymous_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Untracked (session-less) traffic passed through the policy.", compbasemetrics.ALPHA),
		}),
		pickDelayedDispatches: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_delayed_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches whose request waited at least 1s in the flow-control queue.", compbasemetrics.ALPHA),
		}),
		pickTransparent: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_transparent_dispatches_total",
			Help:      metricsutil.HelpMsgWithStability("Dispatches in transparent mode: the idle pool cannot retain anything, so oldest-first is served without membership churn.", compbasemetrics.ALPHA),
		}),
		pickGuestTokens: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_guest_tokens_total",
			Help:      metricsutil.HelpMsgWithStability("Token estimates of never-idle guest dispatches; with the guest count this is the window's displacement cost.", compbasemetrics.ALPHA),
		}),
		pickNoVictims: prometheus.NewCounter(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_no_victims_total",
			Help:      metricsutil.HelpMsgWithStability("Admissions that found no demotable member (every member running): the candidate ran as a non-member with the window intact. Not the oversized case.", compbasemetrics.ALPHA),
		}),
		rotationInterval: prometheus.NewHistogram(prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_rotation_interval_seconds",
			Help:      metricsutil.HelpMsgWithStability("Time between window rotations — the rotation pacing, i.e. the thing under test.", compbasemetrics.ALPHA),
			Buckets:   []float64{0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600},
		}),
		turnHitRatio: prometheus.NewHistogram(prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "ring_window_turn_hit_ratio",
			Help:      metricsutil.HelpMsgWithStability("Per-turn prefix-cache hit ratio (cached/prompt tokens), when the engine reports prompt token details.", compbasemetrics.ALPHA),
			Buckets:   []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1},
		}),
	}
}

func (m *ringMetrics) register(reg prometheus.Registerer) error {
	return errors.Join(
		reg.Register(m.ledger),
		registerOrReuse(reg, &m.pickRotations),
		registerOrReuse(reg, &m.pickRecoveries),
		registerOrReuse(reg, &m.pickMemberServes),
		registerOrReuse(reg, &m.pickJoins),
		registerOrReuse(reg, &m.pickGuests),
		registerOrReuse(reg, &m.pickOversized),
		registerOrReuse(reg, &m.pickAnonymous),
		registerOrReuse(reg, &m.pickDelayedDispatches),
		registerOrReuse(reg, &m.pickTransparent),
		registerOrReuse(reg, &m.pickGuestTokens),
		registerOrReuse(reg, &m.pickNoVictims),
		registerOrReuse(reg, &m.rotationInterval),
		registerOrReuse(reg, &m.turnHitRatio),
	)
}

func registerOrReuse[C prometheus.Collector](reg prometheus.Registerer, target *C) error {
	err := reg.Register(*target)
	if err == nil {
		return nil
	}
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		if existing, ok := already.ExistingCollector.(C); ok {
			*target = existing
			return nil
		}
	}
	return err
}

// ringLedgerCollector reports the window state at scrape time.
type ringLedgerCollector struct {
	mgr *ringManager
}

func (c *ringLedgerCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range c.descs() {
		ch <- d
	}
}

func (c *ringLedgerCollector) descs() []*prometheus.Desc {
	fq := func(name, help string, labels []string) *prometheus.Desc {
		return prometheus.NewDesc(prometheus.BuildFQName("", eppmetrics.LLMDRouterEndpointPickerSubsystem, name),
			metricsutil.HelpMsgWithStability(help, compbasemetrics.ALPHA), labels, nil)
	}
	return []*prometheus.Desc{
		fq("ring_window_members", "Window member count (the member set, ring order).", nil),
		fq("ring_window_member_sum_tokens", "Sum of member footprints, idle members included.", nil),
		fq("ring_window_budget_tokens", "Membership budget theta*C in tokens.", nil),
		fq("ring_window_capacity_tokens", "Pool KV capacity in tokens.", nil),
		fq("ring_window_capacity_from_fallback", "1 while the capacity is still the configured fallback rather than a reported cache_config_info — assert it drops to 0 at run start.", nil),
		fq("ring_window_mean_turn_seconds", "Running mean completed-turn duration in seconds (the elastic-D and D-width input).", nil),
		fq("ring_window_waiting_turns", "Waiting queue heads at the last Pick (N-hat, half of every pacing formula).", nil),
		fq("ring_window_inflight_turns", "Sessions with a turn in flight (c-hat, the concurrency divisor of the elastic D and the D-width).", nil),
		fq("ring_window_d_effective_seconds", "The elastic budget of the last dispatched candidate: max(D, mean*sN/c).", nil),
		fq("ring_window_width_d_members", "The D-width of the window at the last admission check (W_D).", nil),
		fq("ring_window_width_binding", "1 on the width that bound the last failed admission: budget or d.", []string{"binding"}),
		fq("ring_window_demotions_total", "Demotions from the window, by reason.", []string{"reason"}),
	}
}

func (c *ringLedgerCollector) Collect(ch chan<- prometheus.Metric) {
	snap := c.mgr.snapshot(time.Now())
	descs := c.descs()
	ch <- prometheus.MustNewConstMetric(descs[0], prometheus.GaugeValue, float64(snap.members))
	ch <- prometheus.MustNewConstMetric(descs[1], prometheus.GaugeValue, snap.memberSum)
	ch <- prometheus.MustNewConstMetric(descs[2], prometheus.GaugeValue, snap.budget)
	ch <- prometheus.MustNewConstMetric(descs[3], prometheus.GaugeValue, snap.capacity)
	ch <- prometheus.MustNewConstMetric(descs[4], prometheus.GaugeValue, boolFloat(snap.capacityFromConfig))
	ch <- prometheus.MustNewConstMetric(descs[5], prometheus.GaugeValue, snap.meanTurnDur.Seconds())
	ch <- prometheus.MustNewConstMetric(descs[6], prometheus.GaugeValue, float64(snap.waiting))
	ch <- prometheus.MustNewConstMetric(descs[7], prometheus.GaugeValue, float64(snap.inflight))
	ch <- prometheus.MustNewConstMetric(descs[8], prometheus.GaugeValue, snap.dEff.Seconds())
	ch <- prometheus.MustNewConstMetric(descs[9], prometheus.GaugeValue, float64(snap.widthD))
	if snap.binding != "" {
		ch <- prometheus.MustNewConstMetric(descs[10], prometheus.GaugeValue, 1, snap.binding)
	}
	for reason, n := range snap.counters.demotions {
		ch <- prometheus.MustNewConstMetric(descs[11], prometheus.CounterValue, float64(n), reason)
	}
}

func boolFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
