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

// THE INVARIANTS (nca2 spec §9), checked on the STATE after every step of a randomized closed loop
// rather than argued from the code: I1 (one entry inside the line, on the home), I2 (an admission
// never pushes an ACTIVE entry out), I3 (an outside thread holds no seat), I4 (an admitted thread is
// seated only on its home), I6 (liveness: every started turn is served), I7 (work conservation: a pass
// never ends with capacity it could have used), I8 (the cost of a pass at scale).
//
// The driver is a deterministic PRNG: same seed, same run, so a failure is reproducible.

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// invariantCheck is the whole state's invariant report for one moment.
type invariantCheck struct {
	I1, I3, I4 []string
}

// checkInvariants reads the state under the scheduler's lock and reports every violation, in a stable
// order (the message names the rule and the thread, so a failure reads like a bug report).
func checkInvariants(s *Scheduler) invariantCheck {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out invariantCheck
	// I1: an admitted thread has exactly ONE entry, on its HOME pod, at a position inside the line.
	entries := map[string]int{}
	places := map[string]string{}
	for _, id := range s.groupOrder {
		g := s.groups[id]
		if g.nca2 == nil {
			continue
		}
		var pos int64
		for _, e := range g.nca2.line {
			pos += e.tokens
			entries[e.thread]++
			places[e.thread] += fmt.Sprintf(" %s@%d", g.id, pos)
		}
	}
	for thread, home := range s.nca2.homes {
		if !s.nca2.admitted[thread] {
			continue
		}
		if entries[thread] != 1 {
			out.I1 = append(out.I1, fmt.Sprintf("%s is admitted with %d entries (%s)", thread, entries[thread], places[thread]))
			continue
		}
		pos, ok := s.nca2PositionOn(s.groups[home], thread)
		if !ok {
			out.I1 = append(out.I1, fmt.Sprintf("%s is admitted but has no entry on its home %s", thread, home))
			continue
		}
		if pos > s.nca2.AdmitLineTokens {
			out.I1 = append(out.I1, fmt.Sprintf("%s is admitted at position %d, past the line %d", thread, pos, s.nca2.AdmitLineTokens))
		}
	}
	// I3: an outside thread (waiting with no admission) never holds a seat.
	for id := range s.waiters {
		if s.groups[s.bindings[id]] != nil {
			out.I3 = append(out.I3, fmt.Sprintf("%s waits outside and holds a seat on %s", id, s.bindings[id]))
		}
	}
	// I4: an admitted thread is seated only on its home.
	for thread := range s.nca2.admitted {
		if !s.nca2.admitted[thread] {
			continue
		}
		if seatPod, seated := s.bindings[thread]; seated && seatPod != s.nca2.homes[thread] {
			out.I4 = append(out.I4, fmt.Sprintf("%s is admitted on %s but seated on %s", thread, s.nca2.homes[thread], seatPod))
		}
	}
	return out
}

// assertInvariants fails the test on any violation.
func assertInvariants(t *testing.T, s *Scheduler, where string) {
	t.Helper()
	got := checkInvariants(s)
	if len(got.I1) > 0 || len(got.I3) > 0 || len(got.I4) > 0 {
		t.Fatalf("%s: I1 %v; I3 %v; I4 %v", where, got.I1, got.I3, got.I4)
	}
}

// checkWorkConservation is I7: no pass may end with a pod that could have taken the outside head, or
// one of its own warm returners, and did neither. The fit tests are pure, so asking costs nothing.
func checkWorkConservation(s *Scheduler, now time.Time) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	head := s.nca2OutsideHead()
	memo := map[fitKey]fitFacts{}
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 {
			continue
		}
		if head != nil {
			f := PromptTokens(head.chains)
			room := s.nca2View(g).room(s.nca2.AdmitLineTokens)
			if f <= room && s.ncaV2Fits(g, head, now, memo) {
				out = append(out, fmt.Sprintf("head %s fits %s (f=%d room=%d, cold=%d coldInFlight=%d seats=%d free=%d) and was not admitted",
					head.id, g.id, f, room, g.index.ColdTokens(head.chains), g.coldInFlight, len(g.threads), s.freeBudget(g)))
			}
		}
		for _, w := range s.nca2WarmReturners(g) {
			if s.ncaV2Fits(g, w, now, memo) {
				out = append(out, fmt.Sprintf("returner %s fits %s and was not seated", w.id, g.id))
			}
		}
	}
	return out
}

// simThread is one session in the closed loop: a fixed number of turns, a gap between them, and one
// request at a time.
type simThread struct {
	id       string
	turns    int
	gap      int
	req      string
	waiting  bool
	inflight bool
	settleIn int
	waits    int
	maxWait  int
	session  string
	// arrival is the turn's FIRST arrival: a queued head keeps it, like a real queue's enqueue time
	arrival time.Time
}

// sim is the closed loop: N sessions, arrivals and settles in a fixed PRNG order.
type sim struct {
	s       *Scheduler
	threads []*simThread
	rng     *rand.Rand
	now     time.Time
	seq     int
	blocks  int
	// heads / served are the pass in flight: the heads offered and the index it returned
	heads  []Head
	served int
	// pushouts is every L4 push-out with the pushed thread's activity, in order (I2's evidence)
	pushouts []pushoutEvent
	// checked counts the passes I7 could be asked about (the clean ones)
	checked int
}

// pushoutEvent is one push-out as the invariant tests see it.
type pushoutEvent struct {
	reason string
	active bool
}

// newSim builds the loop. tier is the CPU tier in tokens: a small one puts every admission under the
// line's pressure, the real one leaves the CPU space free and makes the loop about the GPU layer.
func newSim(t *testing.T, threads, blocks int, tier int64) *sim {
	t.Helper()
	return newSimSeeded(t, 20261114, threads, blocks, tier)
}

// newSimSeeded is the same loop with a chosen PRNG seed: the property test runs several.
func newSimSeeded(t *testing.T, seed int64, threads, blocks int, tier int64) *sim {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) { c.CPUTierTokens = i64(tier) }, []string{"p0", "p1", "p2"})
	m := &sim{s: s, rng: rand.New(rand.NewSource(seed)), now: epoch, blocks: blocks}
	s.nca2.onPushout = func(reason string, active bool) {
		m.pushouts = append(m.pushouts, pushoutEvent{reason: reason, active: active})
	}
	for i := 0; i < threads; i++ {
		// a SESSION is a thread: its turns are the same id's successive requests
		m.threads = append(m.threads, &simThread{id: fmt.Sprintf("s%03d", i), session: fmt.Sprintf("sess-%03d", i),
			turns: 3 + i%3, gap: i % 4})
	}
	return m
}

// step is ONE pass of the loop: the arrivals that are due, the pass, the release's at-pod stub, and
// the settles that are due.
func (m *sim) step(t *testing.T) bool {
	t.Helper()
	m.now = m.now.Add(time.Second)
	for _, th := range m.threads {
		if th.waiting {
			th.waits++
			if th.waits > th.maxWait {
				th.maxWait = th.waits
			}
		}
	}
	var heads []Head
	for _, th := range m.threads {
		if th.inflight || th.turns == 0 {
			continue
		}
		if !th.waiting {
			if th.gap > 0 {
				th.gap--
				continue
			}
			m.seq++
			th.req = fmt.Sprintf("%s-r%d", th.id, m.seq)
			th.arrival = m.now
			th.waiting = true
		}
		// a QUEUED head is offered again every pass: the router keeps it in the queue until it is
		// served, and the queue key is the FIRST arrival
		heads = append(heads, Head{Thread: th.id, ReqID: th.req, Arrival: th.arrival,
			Chains: chain(th.id, m.blocks)})
	}
	m.heads = heads
	served := m.s.Pass(heads, m.now)
	m.served = served
	return served >= 0
}

// hooks is the request path AFTER the pass: the admitted turn reaches its pod (L1) and the due settles
// run. It is separate because I7 judges the pass on the state the pass itself left.
func (m *sim) hooks(t *testing.T) {
	t.Helper()
	servedThread := ""
	served := m.served
	heads := m.heads
	if served >= 0 {
		servedThread = heads[served].Thread
	}
	for _, th := range m.threads {
		if th.id != servedThread {
			continue
		}
		// the turn ran: the request reached its pod (the at-pod hook is the truth for L1)
		th.waiting, th.inflight = false, true
		th.settleIn = 1 + m.rng.Intn(2)
		th.waits = 0
		m.s.AtPod(th.req, m.homeOf(th.id), m.now)
	}
}

// settleDue ends the requests whose time is up: prefill complete, then the stream's end. It runs AFTER
// the pass's own checks, because a settle releases a cold charge and a request's counts — the pass must
// be judged on the state it decided in.
func (m *sim) settleDue(t *testing.T) {
	t.Helper()
	for _, th := range m.threads {
		if !th.inflight {
			continue
		}
		if th.settleIn > 0 {
			th.settleIn--
			continue
		}
		m.s.FirstChunk(th.req, m.now)
		m.s.Settle(th.req, m.homeOf(th.id), m.now)
		th.inflight, th.req = false, ""
		th.turns--
		th.gap = m.rng.Intn(3)
	}
}

// exitCount is every booked departure, under the lock: a pass that evicted anybody is not a clean
// pass for I7 (its own evictions changed the state it had already decided on).
func (m *sim) exitCount() int64 {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	var n int64
	for _, v := range m.s.counters.exits {
		n += v
	}
	return n
}

// homeOf is the thread's home under the lock ("" when it has none).
func (m *sim) homeOf(id string) string {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	return m.s.nca2.homes[id]
}

// run drives the loop for at most horizon passes and checks every invariant after each one.
func (m *sim) run(t *testing.T, horizon int) {
	t.Helper()
	for i := 0; i < horizon; i++ {
		exitsBefore := m.exitCount()
		m.step(t)
		// the pass is judged on the state it left, BEFORE any hook of the request path runs
		assertInvariants(t, m.s, fmt.Sprintf("pass %d", i))
		if s := m.s.snapshot(m.now).NCA2; s.DecisionsThisPass == 0 && !s.DecisionCapHit && m.exitCount() == exitsBefore {
			// a pass that admitted nothing and evicted nothing leaves the state exactly as it found it:
			// a candidate that fits now was a candidate it could have taken (I7)
			m.checked++
			if got := checkWorkConservation(m.s, m.now); len(got) > 0 {
				t.Fatalf("I7 at pass %d: %v", i, got)
			}
		}
		m.hooks(t)
		m.settleDue(t)
		assertInvariants(t, m.s, fmt.Sprintf("pass %d, after %d settles", i, i))
		alive := 0
		for _, th := range m.threads {
			if th.turns > 0 {
				alive++
			}
		}
		if alive == 0 {
			return
		}
	}
	remaining := 0
	worst := 0
	var worstTh *simThread
	for _, th := range m.threads {
		if th.turns > 0 {
			remaining++
		}
		if th.maxWait > worst {
			worst, worstTh = th.maxWait, th
		}
	}
	m.s.mu.Lock()
	detail := ""
	if worstTh != nil {
		detail = fmt.Sprintf("%s: turns=%d inflight=%v waiting=%v admitted=%v home=%q seated=%v exiting=%v",
			worstTh.id, worstTh.turns, worstTh.inflight, worstTh.waiting, m.s.nca2.admitted[worstTh.id],
			m.s.nca2.homes[worstTh.id], m.s.bindings[worstTh.id], m.s.groups[m.s.bindings[worstTh.id]] != nil &&
				m.s.groups[m.s.bindings[worstTh.id]].index.Exiting(worstTh.id))
	}
	m.s.mu.Unlock()
	t.Fatalf("I6: %d of %d sessions never finished in %d passes (worst wait %d passes; %s)",
		remaining, len(m.threads), horizon, worst, detail)
}

// TestInvariantsUnderCPUPressure: a SMALL tier (3200 tokens) with prompts close to the line, so every
// admission presses the line and the push-outs are frequent: I1-I4 and I7 still hold on every pass.
func TestInvariantsUnderCPUPressure(t *testing.T) {
	for _, seed := range []int64{1, 7, 20261114, 99, 4242} {
		// 40 sessions of 8-block prompts (512 tokens) into a 3040-token line: ~6 entries per pod, so a
		// touch re-orders a nearly full line — the class mix L4 is about (room, growth and overflow)
		m := newSimSeeded(t, seed, 40, 16, lineC)
		m.run(t, 400)
		checkPushoutMix(t, m, seed)
	}
}

// checkPushoutMix is I2's evidence AND the fixture's own coverage: every push-out class must have
// happened at least once for the run to say anything about it.
func checkPushoutMix(t *testing.T, m *sim, seed int64) {
	t.Helper()
	if len(m.pushouts) == 0 {
		t.Fatalf("seed %d: a tier this small must push threads out", seed)
	}
	seen := map[string]int{}
	for _, ev := range m.pushouts {
		seen[ev.reason]++
	}
	// I2: an ACTIVE entry is only ever pushed by a TOUCH (overflow) or a pod reset — never by an
	// admission (room) or by another thread's touch finding it inactive (growth).
	for _, ev := range m.pushouts {
		switch ev.reason {
		case pushoutRoom, pushoutGrowth:
			if ev.active {
				t.Fatalf("seed %d: I2: an ACTIVE entry was pushed out with reason %q", seed, ev.reason)
			}
		case pushoutOverflow:
			if !ev.active {
				t.Fatalf("seed %d: A6: reason overflow names an ACTIVE entry, got an inactive one", seed)
			}
		case pushoutRestart:
		default:
			t.Fatalf("seed %d: unknown push-out reason %q", seed, ev.reason)
		}
	}
	// a closed loop of admissions into a nearly full line produces `room` push-outs; the OTHER two
	// classes need a touch that re-orders the line under a candidate and are covered deterministically
	// in the line tests (growth and overflow), so this run only has to have pushed somebody out
	if seen[pushoutRoom] == 0 {
		t.Fatalf("seed %d: the fixture produced no room push-out (seen %v)", seed, seen)
	}
}

// TestLivenessOnTheRealTier: I6 — with the CPU tier at its real size the loop is the GPU layer's, and
// every started turn of every session is served.
//
// ONE PASS IS ONE PICK, and a Pick serves one head: 24 sessions of 3-5 turns therefore put up to 24
// heads in the queue at once, so a head's wait is bounded by the heads ahead of it plus the seat
// rotation (the quantum, and the 5 s grace, which this loop's 1-3 pass gaps stay inside). Measured
// 18-31 passes; the bound is 60 so the test reports a REGRESSION rather than a schedule.
func TestLivenessOnTheRealTier(t *testing.T) {
	m := newSim(t, 24, 10, defaultCPUTierTokens)
	m.run(t, 400)
	worst := 0
	for _, th := range m.threads {
		if th.maxWait > worst {
			worst = th.maxWait
		}
	}
	t.Logf("I6: 24 sessions, 3-5 turns each, one Pick per pass: worst head wait %d passes", worst)
	if worst > 60 {
		t.Fatalf("I6: a head waited %d passes on a tier that holds the whole working set", worst)
	}
}

// TestCostAtScale (I8): a pass over three 1000-entry lines and 1000 waiters stays inside a generous
// wall-clock bound — a quadratic (or worse) derive would take orders of magnitude longer.
func TestCostAtScale(t *testing.T) {
	s := newNCA2(t, []string{"p0", "p1", "p2"})
	now := epoch
	s.mu.Lock()
	for _, pod := range []string{"p0", "p1", "p2"} {
		g := s.groups[pod]
		for i := 0; i < 1000; i++ {
			g.nca2.line = append(g.nca2.line, nca2Entry{thread: fmt.Sprintf("%s-t%d", pod, i), tokens: 100})
		}
	}
	s.mu.Unlock()
	heads := make([]Head, 0, 1000)
	for i := 0; i < 1000; i++ {
		id := fmt.Sprintf("n%d", i)
		heads = append(heads, Head{Thread: id, ReqID: "r-" + id, Arrival: now.Add(-time.Duration(i) * time.Millisecond),
			Chains: chain(id, 40)})
	}
	start := time.Now()
	s.Pass(heads, now.Add(time.Second))
	took := time.Since(start)
	if took > 5*time.Second {
		t.Fatalf("I8: a pass over 3 x 1000 entries and 1000 waiters took %v", took)
	}
	t.Logf("I8: 3 x 1000-entry lines, 1000 waiters: %v for one pass", took)
}

// newRng is the loop's deterministic PRNG: the same seed every run.
func newRng() *rand.Rand { return rand.New(rand.NewSource(20261114)) }
