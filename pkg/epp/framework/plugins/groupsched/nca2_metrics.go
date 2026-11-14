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

// THE nca2 METRIC SURFACE (nca2 spec §7). Every series carries the llm_d_epp_nca2_ prefix and is
// registered ONLY in nca2: OFF's metric surface is the shipped one, unchanged.
//
// Every method is nil-safe on purpose. The scheduler owns one of these (s.nca2m), nil in OFF and nil
// in a bare unit-test Scheduler, so an nca2 decision can be recorded without any wiring and OFF
// cannot reach it at all.

import (
	"sort"

	"github.com/prometheus/client_golang/prometheus"
)

// nca2Metrics is the nca2 counters and histograms. The per-pod GAUGES are not here: they are
// rendered from ONE Snapshot at scrape time (nca2Gauges), so a scrape never mutates state (I9).
type nca2Metrics struct {
	// admissions counts A2.1 seatings, by the outside class that won them.
	admissions *prometheus.CounterVec
	// returnerSeats counts A2.2 seatings: a warm returner, seated on its home.
	returnerSeats *prometheus.CounterVec
	// decisions counts every decision a pod took, by outcome.
	decisions *prometheus.CounterVec
	// pushouts counts every loss of admission, by the event that caused it.
	pushouts *prometheus.CounterVec
	// activePushouts counts the push-outs of a thread that was ACTIVE: the C13/A6 class, the one
	// that keeps its seat for the in-flight request.
	activePushouts *prometheus.CounterVec
	// moves counts L3: the home followed the observed pod.
	moves *prometheus.CounterVec
	// lineTouchTokens is the CPU WRITE RATE: the tokens of every touch (L1) per pod.
	lineTouchTokens *prometheus.CounterVec
	// leakedRequests counts requests that outlived maxInflightSeconds (once each, at the derive).
	leakedRequests prometheus.Counter
	// doorWait is the arrival → the admission, by the class that won it.
	doorWait *prometheus.HistogramVec
	// seatWait is the wait for a seat, by the class of the seat itself.
	seatWait *prometheus.HistogramVec
}

// nca2Subsystem is the nca2 series' namespace (llm_d_epp_nca2_*).
const nca2Subsystem = "llm_d_epp_nca2"

// newNCA2Metrics declares every nca2 series. It does NOT register them: registration is the one
// nca2-only step in Factory, so a second instance in the same process cannot duplicate them.
func newNCA2Metrics() *nca2Metrics {
	return &nca2Metrics{
		admissions: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "admissions_total",
			Help: "A2.1 admissions: the outside head was admitted on this pod and seated at once, split by the class it came from (priority = it was admitted before and was pushed out; newcomer = it never was)"}, []string{"pod", "path"}),
		returnerSeats: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "returner_seats_total",
			Help: "A2.2 seatings: an admitted thread of this pod was WAITING with no seat and took a freed one, on its home"}, []string{"pod"}),
		decisions: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "decisions_total",
			Help: "decisions taken on this pod, by outcome (admit = A2.1, returner = A2.2). A pod that can decide nothing does not move this counter; head_blocked_seconds is what measures that"}, []string{"pod", "outcome"}),
		pushouts: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "pushouts_total",
			Help: "every LOSS of admission, by the event that caused it: room = an admission's provisional entry moved it past the line (always an inactive thread: I2), growth = an inactive entry pushed by another thread's touch, overflow = an ACTIVE entry pushed by a touch (A6/C13: the active threads alone exceeded the line), restart = the pod's line was reset (S5)"}, []string{"pod", "reason"}),
		activePushouts: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "active_pushouts_total",
			Help: "the push-outs of a thread that was ACTIVE at the push-out: it keeps its seat for its in-flight request (marked exiting) and goes outside when its next head arrives"}, []string{"pod"}),
		moves: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "moves_total",
			Help: "L3: a request reached a pod other than the thread's home, so the home (and the CPU line's entry) moved to the pod that served it — the main source of cold"}, []string{"pod"}),
		lineTouchTokens: prometheus.NewCounterVec(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "line_touch_tokens_total",
			Help: "the CPU WRITE RATE: the prompt tokens of every touch (L1, a request reaching this pod), i.e. how much the router's model writes into this pod's CPU tier"}, []string{"pod"}),
		leakedRequests: prometheus.NewCounter(prometheus.CounterOpts{Subsystem: nca2Subsystem, Name: "leaked_requests_total",
			Help: "requests in flight longer than maxInflightSeconds, counted ONCE each: they stop holding their thread ACTIVE (and stop pinning the room)"}),
		doorWait: prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: nca2Subsystem, Name: "door_wait_seconds",
			Help:    "the DOOR: the head's arrival to its admission, by the class that won it (priority | newcomer). TOP BUCKET 13,107 s so nothing lands only in +Inf",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"path"}),
		seatWait: prometheus.NewHistogramVec(prometheus.HistogramOpts{Subsystem: nca2Subsystem, Name: "seat_wait_seconds",
			Help:    "the wait for a GPU SEAT, by the class of the seat: returner (A2.2, an admitted thread of this pod) or admitted_now (A2.1, a thread admitted and seated in one decision). Same buckets as the door, so the two compose",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 18)}, []string{"class"}),
	}
}

// collectors is every nca2 series to register, in one list: the plugin's register installs them
// (and the gauges collector) ONLY when nca2 runs, so OFF's metric surface gains nothing.
func (m *nca2Metrics) collectors() []prometheus.Collector {
	if m == nil {
		return nil
	}
	// the global zero series: a zero is a fact, an absent series is a question
	m.leakedRequests.Add(0)
	for _, path := range []string{pathPriority, pathNewcomer} {
		m.doorWait.WithLabelValues(path) // creates the series; no observation
	}
	for _, class := range []string{nca2ClassReturner, nca2ClassAdmittedNow} {
		m.seatWait.WithLabelValues(class)
	}
	return []prometheus.Collector{m.admissions, m.returnerSeats, m.decisions, m.pushouts,
		m.activePushouts, m.moves, m.lineTouchTokens, m.leakedRequests, m.doorWait, m.seatWait}
}

// preInstantiatePod creates this pod's zero-valued counter series when its group is created: "no
// admission", "no push-out of this reason", "no move" are readable as zeros from the pod's first
// scrape (and survive the pod's removal, as a gauge must not).
func (m *nca2Metrics) preInstantiatePod(pod string) {
	if m == nil {
		return
	}
	for _, path := range []string{"priority", "newcomer"} {
		m.admissions.WithLabelValues(pod, path).Add(0)
	}
	for _, outcome := range []string{"admit", "returner"} {
		m.decisions.WithLabelValues(pod, outcome).Add(0)
	}
	for _, reason := range pushoutReasons {
		m.pushouts.WithLabelValues(pod, reason).Add(0)
	}
	m.returnerSeats.WithLabelValues(pod).Add(0)
	m.activePushouts.WithLabelValues(pod).Add(0)
	m.moves.WithLabelValues(pod).Add(0)
	m.lineTouchTokens.WithLabelValues(pod).Add(0)
}

// ObserveAdmission records an A2.1 admission and the door wait it ended.
func (m *nca2Metrics) ObserveAdmission(pod, path string, doorWaitSeconds float64) {
	if m == nil {
		return
	}
	m.admissions.WithLabelValues(pod, path).Inc()
	m.decisions.WithLabelValues(pod, "admit").Inc()
	m.doorWait.WithLabelValues(path).Observe(nonNegativeSeconds(doorWaitSeconds))
}

// ObserveReturnerSeat records an A2.2 seating: a warm returner took a freed seat on its home.
func (m *nca2Metrics) ObserveReturnerSeat(pod string) {
	if m == nil {
		return
	}
	m.returnerSeats.WithLabelValues(pod).Inc()
	m.decisions.WithLabelValues(pod, "returner").Inc()
}

// ObserveSeatWait records ONE seat wait, by the class of the seat.
func (m *nca2Metrics) ObserveSeatWait(class string, seconds float64) {
	if m == nil {
		return
	}
	m.seatWait.WithLabelValues(class).Observe(nonNegativeSeconds(seconds))
}

// ObservePushout records one loss of admission.
func (m *nca2Metrics) ObservePushout(pod, reason string) {
	if m == nil {
		return
	}
	m.pushouts.WithLabelValues(pod, reason).Inc()
}

// ObserveActivePushout records the push-out of a thread that was ACTIVE.
func (m *nca2Metrics) ObserveActivePushout(pod string) {
	if m == nil {
		return
	}
	m.activePushouts.WithLabelValues(pod).Inc()
}

// ObserveMove records L3: the home followed the pod that served the request.
func (m *nca2Metrics) ObserveMove(pod string) {
	if m == nil {
		return
	}
	m.moves.WithLabelValues(pod).Inc()
}

// ObserveLineTouch records one L1 write on a pod's CPU line.
func (m *nca2Metrics) ObserveLineTouch(pod string, tokens int64) {
	if m == nil {
		return
	}
	m.lineTouchTokens.WithLabelValues(pod).Add(float64(tokens))
}

// ObserveLeaked records ONE request that outlived maxInflightSeconds.
func (m *nca2Metrics) ObserveLeaked() {
	if m == nil {
		return
	}
	m.leakedRequests.Inc()
}

// nonNegativeSeconds clamps a duration in seconds: a clock that went backwards must not observe a
// negative wait into an exponential histogram.
func nonNegativeSeconds(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// ---- the gauges: ONE Snapshot per scrape, nothing written back (I9) ----

// nca2Gauges renders every nca2 GAUGE from ONE Snapshot at scrape time. A gauge read never changes a
// decision — the state is only read, and the ages it reports take the caller's clock, because the
// scheduler's own clock only advances when its pass runs.
type nca2Gauges struct {
	r     *GroupScheduler
	sched *Scheduler

	room             *prometheus.Desc
	laaPosition      *prometheus.Desc
	lineTokens       *prometheus.Desc
	admitted         *prometheus.Desc
	admittedActive   *prometheus.Desc
	admittedThreads  *prometheus.Desc
	activeThreads    *prometheus.Desc
	warmReturners    *prometheus.Desc
	pendingHeads     *prometheus.Desc
	seatsInflight    *prometheus.Desc
	seatsReclaimable *prometheus.Desc
	idleWithWaiters  *prometheus.Desc

	priorityWaiting    *prometheus.Desc
	newcomerWaiting    *prometheus.Desc
	headBlockedSeconds *prometheus.Desc
}

// nca2GaugeDesc is one per-pod nca2 gauge descriptor (the pod label is the endpoint id).
func nca2GaugeDesc(name, help string) *prometheus.Desc {
	return prometheus.NewDesc(prometheus.BuildFQName("", nca2Subsystem, name), help, []string{"pod"}, nil)
}

// newNCA2Gauges builds the collector for one plugin instance.
func newNCA2Gauges(r *GroupScheduler) *nca2Gauges {
	return &nca2Gauges{
		r:     r,
		sched: r.sched,
		room: nca2GaugeDesc("room_tokens",
			"the space an admission on this pod may consume: the admission line minus the position of the LOWEST admitted active entry (the whole line when there is none)"),
		laaPosition: nca2GaugeDesc("laa_position_tokens",
			"the position of the LOWEST admitted active entry (the LAA): the entry the room is measured below"),
		lineTokens: nca2GaugeDesc("line_tokens",
			"this pod's CPU line: the router's model of the pod's CPU KV tier, in tokens"),
		admitted: nca2GaugeDesc("admitted_tokens",
			"the tokens of this pod's ADMITTED entries that sit inside the admission line"),
		admittedActive: nca2GaugeDesc("admitted_active_tokens",
			"the part of admitted_tokens that belongs to ACTIVE threads (the tokens an admission has to protect)"),
		admittedThreads: nca2GaugeDesc("admitted_threads",
			"the threads admitted on this pod (the STATE, granted at A2.1 and lost at a push-out)"),
		activeThreads: nca2GaugeDesc("active_threads",
			"the threads with an entry on this pod that are ACTIVE (waiting or in flight)"),
		warmReturners: nca2GaugeDesc("warm_returners_waiting",
			"A2.2's list: this pod's admitted threads that are waiting with no seat"),
		pendingHeads: nca2GaugeDesc("pending_heads",
			"the demand this pod's own threads are holding back: its warm returners plus its seated threads with a queued head"),
		seatsInflight: nca2GaugeDesc("seats_inflight",
			"seats on this pod whose thread has a request in flight (a seat no rotation may take)"),
		seatsReclaimable: nca2GaugeDesc("seats_reclaimable",
			"seats on this pod the shipped rotation would accept as a victim right now (Q completed periods, or idleSeconds of silence)"),
		idleWithWaiters: nca2GaugeDesc("idle_with_waiters",
			"1 when this pod has nothing in flight while an outside head or one of its warm returners waits — the shape a liveness bug has on a dashboard"),
		priorityWaiting: prometheus.NewDesc(prometheus.BuildFQName("", nca2Subsystem, "priority_waiting"),
			"the GLOBAL priority list: threads waiting outside with no admission that had one before", nil, nil),
		newcomerWaiting: prometheus.NewDesc(prometheus.BuildFQName("", nca2Subsystem, "newcomer_waiting"),
			"the GLOBAL newcomer queue: threads waiting outside that were never admitted", nil, nil),
		headBlockedSeconds: prometheus.NewDesc(prometheus.BuildFQName("", nca2Subsystem, "head_blocked_seconds"),
			"how long the CURRENT outside head has been the head without fitting any pod (0 when the queue is empty)", nil, nil),
	}
}

// Describe sends every descriptor this collector can produce.
func (c *nca2Gauges) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range []*prometheus.Desc{c.room, c.laaPosition, c.lineTokens, c.admitted,
		c.admittedActive, c.admittedThreads, c.activeThreads, c.warmReturners, c.pendingHeads,
		c.seatsInflight, c.seatsReclaimable, c.idleWithWaiters, c.priorityWaiting,
		c.newcomerWaiting, c.headBlockedSeconds} {
		ch <- d
	}
}

// Collect reads ONE snapshot and emits const gauges from it.
func (c *nca2Gauges) Collect(ch chan<- prometheus.Metric) {
	snap := c.sched.Snapshot(c.r.now()).NCA2
	if snap == nil {
		return
	}
	gauge := func(d *prometheus.Desc, v float64, labels ...string) {
		ch <- prometheus.MustNewConstMetric(d, prometheus.GaugeValue, v, labels...)
	}
	pods := make([]string, 0, len(snap.Engines))
	for id := range snap.Engines {
		pods = append(pods, id)
	}
	sort.Strings(pods) // deterministic emit order: a scraper compares two scrapes row by row
	for _, id := range pods {
		p := snap.Engines[id]
		gauge(c.room, float64(p.RoomTokens), p.ID)
		gauge(c.laaPosition, float64(p.LAAPositionTokens), p.ID)
		gauge(c.lineTokens, float64(p.LineTokens), p.ID)
		gauge(c.admitted, float64(p.AdmittedTokens), p.ID)
		gauge(c.admittedActive, float64(p.AdmittedActiveTokens), p.ID)
		gauge(c.admittedThreads, float64(p.AdmittedThreads), p.ID)
		gauge(c.activeThreads, float64(p.ActiveThreads), p.ID)
		gauge(c.warmReturners, float64(p.WarmReturnersWaiting), p.ID)
		gauge(c.pendingHeads, float64(p.PendingHeads), p.ID)
		gauge(c.seatsInflight, float64(p.SeatsInflight), p.ID)
		gauge(c.seatsReclaimable, float64(p.SeatsReclaimable), p.ID)
		gauge(c.idleWithWaiters, float64(p.IdleWithWaiters), p.ID)
	}
	gauge(c.priorityWaiting, float64(snap.PriorityWaiting))
	gauge(c.newcomerWaiting, float64(snap.NewcomerWaiting))
	gauge(c.headBlockedSeconds, snap.HeadBlockedSeconds)
}
