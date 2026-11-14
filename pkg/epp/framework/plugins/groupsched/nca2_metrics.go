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

// register installs the nca2 series (called only when nca2 runs) and pre-instantiates the global
// zero series: a zero is a fact, an absent series is a question.
func (m *nca2Metrics) register(reg prometheus.Registerer) error {
	if m == nil {
		return nil
	}
	for _, c := range []prometheus.Collector{m.admissions, m.returnerSeats, m.decisions, m.pushouts,
		m.activePushouts, m.moves, m.lineTouchTokens, m.leakedRequests, m.doorWait, m.seatWait} {
		if err := reg.Register(c); err != nil {
			return err
		}
	}
	m.leakedRequests.Add(0)
	return nil
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
