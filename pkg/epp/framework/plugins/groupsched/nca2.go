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

// THE CPU TIER (nca2 spec §2–§3): each pod's CPU line — the router's model of vLLM's CPU LRU — the
// thread's ACTIVITY, the admission STATE, and the line events L0 (the provisional entry), L1 (the
// touch), L2 (the trim), L3 (the move) and L4 (the push-out).
//
// The line is a list of entries ordered the MOST RECENT TOUCH FIRST. An entry's POSITION is the sum
// of the entry tokens from the head through it, i.e. where vLLM's LRU would place it inside the CPU
// tier: the head is the hottest, the tail the coldest. The admission line L_a is a position, so an
// entry is INSIDE the line when its position is ≤ L_a. Room is the space below the LOWEST ADMITTED
// ACTIVE entry (the LAA): everything past the LAA — inactive admitted entries and empty space — is
// what the next admission may consume.
//
// The gate, once, for every function here: s.nca2 is nil in OFF, and nothing below runs then.

import (
	"sort"
	"time"
)

// nca2Policy is the resolved nca2 policy AND its state: nil in OFF (the shipped behaviour), one
// instance in nca2. The numbers are resolved once, at construction (config.go's resolveNCA2).
type nca2Policy struct {
	// CPUTierTokens is C: the pod's CPU KV tier, in tokens. The trim drops an entry past it.
	CPUTierTokens int64
	// AdmitLineTokens is L_a = admitLine x C: the admission line, a POSITION.
	AdmitLineTokens int64
	// TrustLineTokens is L_t = trustLine x C: the line the WARM PREDICTION trusts (position ≤ L_t
	// before a touch is predicted warm). Nothing in a decision reads it.
	TrustLineTokens int64
	// MaxInflightSeconds bounds a request's in-flight age for the ACTIVITY test: past it the record
	// is leaked and stops holding its thread active.
	MaxInflightSeconds float64
	// LostRequestSeconds is the released-window liveness bound (S2).
	LostRequestSeconds float64

	// homes is thread → its home pod. It is retained through a push-out (A4's "old pod") and
	// cleared only when the pod goes away (S5): a home on a pod that no longer exists is not a
	// placement.
	homes map[string]string
	// admitted is the admission STATE (§2): GRANTED at A2.1, LOST at a push-out (L4), TRANSFERRED
	// by a move (L3). It is explicit, not derived from a position: a move or a trim lowers the
	// positions below it, and a derived rule would admit threads that never won a decision.
	admitted map[string]bool
	// ever is "admitted at some point": it separates the PRIORITY class (pushed out) from the
	// NEWCOMER class (never admitted) among the outside threads.
	ever map[string]bool
	// inf is the in-flight request count per thread (§2): a request RELEASED by dispatch that has
	// not completed, the leak rule applied. It is re-derived from the request records every pass
	// (derive-don't-accumulate) and maintained at the request hooks.
	inf map[string]int
	// blockedHead / blockedSince track the outside head's tenure: how long the CURRENT outside head
	// has been the head. Written by the pass, only READ by the Snapshot (I9).
	blockedHead  string
	blockedSince time.Time
}

// maxInflightWindow is MaxInflightSeconds as a duration.
func (n *nca2Policy) maxInflightWindow() time.Duration {
	return time.Duration(n.MaxInflightSeconds * float64(time.Second))
}

// lostRequestWindow is LostRequestSeconds as a duration.
func (n *nca2Policy) lostRequestWindow() time.Duration {
	return time.Duration(n.LostRequestSeconds * float64(time.Second))
}

// initNCA2 allocates the policy's state: the maps are created once, at construction, so no decision
// path ever creates one lazily.
func (s *Scheduler) initNCA2() {
	s.nca2.homes = map[string]string{}
	s.nca2.admitted = map[string]bool{}
	s.nca2.ever = map[string]bool{}
	s.nca2.inf = map[string]int{}
}

// nca2Pod is ONE pod's CPU line and the per-pod number the pass maintains on it.
type nca2Pod struct {
	// line is the ordered entries, the MOST RECENT TOUCH FIRST.
	line []nca2Entry
	// inflight is the number of requests IN FLIGHT on this pod: A5's first sort key. Derived every
	// pass from the request records (and maintained at the pod hook for a between-pass read).
	inflight int
}

// nca2Entry is one thread's place in a pod's CPU line.
type nca2Entry struct {
	thread string
	// tokens is the prompt tokens of the thread's last request that REACHED this pod (vLLM offloads
	// the prompts, so that is what the pod's CPU tier holds for it).
	tokens int64
	// lastTouch is when the entry was last written: the LRU order, kept explicitly so a debug read
	// can name it (the list order IS the recency).
	lastTouch time.Time
}

// nca2PodOf is a group's line, or nil when the group predates nca2 (a group created in OFF).
func nca2PodOf(g *engineGroup) *nca2Pod { return g.nca2 }

// ---- the line's geometry ----

// nca2PositionOn is one entry's POSITION on this pod: the sum of the entry tokens from the head
// through it. ok is false when the thread has no entry there.
func (s *Scheduler) nca2PositionOn(g *engineGroup, id string) (int64, bool) {
	if g == nil || g.nca2 == nil {
		return 0, false
	}
	var pos int64
	for _, e := range g.nca2.line {
		pos += e.tokens
		if e.thread == id {
			return pos, true
		}
	}
	return 0, false
}

// nca2AdmittedOn reports whether the thread is ADMITTED ON THIS POD, and its position: home = P
// (§2's definition) AND the state, which only A2.1 grants and only L4 clears. The position test is
// kept as the invariant's own reading of I1 — an admitted thread whose entry is past the line is a
// push-out the pass has not booked yet, and it is NOT counted as admitted here.
func (s *Scheduler) nca2AdmittedOn(g *engineGroup, id string) (int64, bool) {
	if s.nca2 == nil || g == nil || !s.nca2.admitted[id] || s.nca2.homes[id] != g.id {
		return 0, false
	}
	pos, ok := s.nca2PositionOn(g, id)
	if !ok || pos > s.nca2.AdmitLineTokens {
		return 0, false
	}
	return pos, true
}

// nca2View is ONE pod's line as a decision (and a scrape) reads it.
type nca2View struct {
	// lineTokens is the whole line (the CPU tier's resident size).
	lineTokens int64
	// admittedTokens is the tokens of the ADMITTED entries inside the line.
	admittedTokens int64
	// admittedActiveTokens is the part of it that belongs to ACTIVE threads.
	admittedActiveTokens int64
	// admittedThreads, activeThreads count entries by class.
	admittedThreads int
	activeThreads   int
	// laa is the LOWEST ADMITTED ACTIVE entry: the admitted, active entry with the largest
	// position ("" when there is none), laaPosition its position.
	laa         string
	laaPosition int64
	laaSet      bool
}

// nca2Room is the space an admission on this pod may consume (§2): the admission line minus the
// LAA's position, or the whole line when no admitted active entry exists. It is ≥ 0 by
// construction, because the LAA is admitted and an admitted position is ≤ L_a.
func (v nca2View) room(admitLine int64) int64 {
	if !v.laaSet {
		return admitLine
	}
	if room := admitLine - v.laaPosition; room > 0 {
		return room
	}
	return 0
}

// nca2View is the line's geometry, in ONE walk (I8): the position sums, the class counts, the LAA.
func (s *Scheduler) nca2View(g *engineGroup) nca2View {
	var v nca2View
	if s.nca2 == nil || g == nil || g.nca2 == nil {
		return v
	}
	var pos int64
	for _, e := range g.nca2.line {
		pos += e.tokens
		v.lineTokens = pos
		active := s.nca2Active(e.thread)
		if active {
			v.activeThreads++
		}
		if !s.nca2.admitted[e.thread] || s.nca2.homes[e.thread] != g.id || pos > s.nca2.AdmitLineTokens {
			continue
		}
		v.admittedThreads++
		v.admittedTokens += e.tokens
		if active {
			v.admittedActiveTokens += e.tokens
			// the LAA: the admitted ACTIVE entry with the LARGEST position (the nearest the tail)
			if !v.laaSet || pos > v.laaPosition {
				v.laa, v.laaPosition, v.laaSet = e.thread, pos, true
			}
		}
	}
	return v
}

// nca2Active is §2's ACTIVITY: the thread is WAITING (a head is queued at the router) or IN FLIGHT
// (a released request has not completed; a leaked one does not count — nca2Derive applies the
// bound). A thread with a seat and neither is INACTIVE, the class the 5 s grace is about.
func (s *Scheduler) nca2Active(id string) bool {
	if s.nca2 == nil || id == "" {
		return false
	}
	if s.nca2.inf[id] > 0 {
		return true
	}
	if _, waiting := s.waiters[id]; waiting {
		return true
	}
	if g := s.groupOf(id); g != nil {
		if m := g.threads[id]; m != nil && !m.noQueuedHead() {
			return true
		}
	}
	return false
}

// ---- the line events ----

// nca2LineSet is L0/L1/L2/L4 for ONE thread on ONE pod: it sets the thread's entry at the HEAD of
// the line with these tokens (inserting it when the thread had none), trims the line past C, and
// books every admitted thread the mutation pushed past the admission line.
//
// cause names the mutation for the push-out's reason: pushoutRoom for an admission's provisional
// entry (L0), pushoutTouch for a request reaching the pod (L1/L3). The pushed entry's own ACTIVITY
// then splits the touch class: an inactive one (growth) may be pushed, an active one (overflow) is
// the C13 event.
func (s *Scheduler) nca2LineSet(g *engineGroup, id string, tokens int64, now time.Time, cause string) {
	if s.nca2 == nil || g == nil || g.nca2 == nil || id == "" {
		return
	}
	before := s.nca2AdmittedSet(g)
	p := g.nca2
	// L0/L1: THE ENTRY AT THE HEAD. One entry per thread: the thread's place IS its recency, so a
	// touch removes the old place and prepends the new one.
	kept := make([]nca2Entry, 0, len(p.line)+1)
	for _, e := range p.line {
		if e.thread != id {
			kept = append(kept, e)
		}
	}
	p.line = append([]nca2Entry{{thread: id, tokens: tokens, lastTouch: now}}, kept...)
	// L2: THE TRIM, past C. Everything from the first entry beyond the tier is gone: positions only
	// grow down the list, so one cut is the whole trim.
	var pos int64
	cut := len(p.line)
	for i, e := range p.line {
		pos += e.tokens
		if pos > s.nca2.CPUTierTokens {
			cut = i
			break
		}
	}
	if cut < len(p.line) {
		for i := cut; i < len(p.line); i++ {
			p.line[i] = nca2Entry{} // let the dropped entries go; the slice is reused
		}
		p.line = p.line[:cut]
	}
	// L4: THE PUSH-OUTS. Every thread that was admitted on this pod and is not any more lost its
	// admission HERE, at the mutation that moved its position past the line.
	after := s.nca2AdmittedSet(g)
	for gone := range before {
		if _, still := after[gone]; still {
			continue
		}
		reason := pushoutGrowth
		switch {
		case cause == pushoutRoom:
			reason = pushoutRoom
		case s.nca2Active(gone):
			reason = pushoutOverflow
		}
		s.nca2Pushout(g, gone, reason, now)
	}
}

// nca2AdmittedSet is the set of threads ADMITTED on this pod right now: the L4 detector's input.
func (s *Scheduler) nca2AdmittedSet(g *engineGroup) map[string]struct{} {
	out := map[string]struct{}{}
	if s.nca2 == nil || g == nil || g.nca2 == nil {
		return out
	}
	var pos int64
	for _, e := range g.nca2.line {
		pos += e.tokens
		if !s.nca2.admitted[e.thread] || s.nca2.homes[e.thread] != g.id {
			continue
		}
		if pos <= s.nca2.AdmitLineTokens {
			out[e.thread] = struct{}{}
		}
	}
	return out
}

// The push-out reasons (§3 L4, A6). Each names ONE event; the counters carry them, so the classes a
// run has to be able to distinguish stay distinguishable.
const (
	// pushoutRoom: an admission's provisional entry (or its own touch) pushed the thread past the
	// line. By construction such a thread is INACTIVE (room is measured below the LAA), which is why
	// I2 holds: an ACTIVE admitted entry is never pushed by A2.1.
	pushoutRoom = "room"
	// pushoutGrowth: an INACTIVE entry pushed past the line by another thread's touch.
	pushoutGrowth = "growth"
	// pushoutOverflow: an ACTIVE entry pushed past the line by a touch (A6, C13: the physical form
	// of "the active threads alone exceed the line"). No action beyond the push-out itself.
	pushoutOverflow = "overflow"
	// pushoutRestart: the pod's line was reset (S5: an engine removal), so its threads lost their
	// admission.
	pushoutRestart = "restart"
	// pushoutTouch is not a reason: it is the CAUSE (a request reaching the pod, L1/L3), which L4
	// splits into growth or overflow by the pushed entry's own activity.
	pushoutTouch = "touch"
)

// pushoutReasons is every reason, in a fixed order (the metric pre-instantiates one zero series per
// reason in nca2: a zero is a fact, an absent series is a question).
var pushoutReasons = []string{pushoutRoom, pushoutGrowth, pushoutOverflow, pushoutRestart}

// nca2Pushout is L4: the thread loses its admission, and the pod's GPU layer reacts exactly as the
// base's mid-flight rule says it must (the spec owner's Q2):
//   - INACTIVE and seated → it is evicted NOW (exit reason pushout): it holds a seat it is not
//     using and no longer owns.
//   - ACTIVE and seated → it KEEPS the seat for its in-flight request, MARKED exiting: dispatch
//     skips an exiting thread, the base's drain exits it once nothing is in flight, and that exit
//     re-queues its waiting head into the shared queue — i.e. it goes OUTSIDE as the priority class
//     when its next head arrives (A0's base dispatch never serves it again).
//
// Nothing else happens: the entry stays in the line (up to C), which is what makes it a warm
// returner of its old pod when it comes back (S4, A4).
func (s *Scheduler) nca2Pushout(g *engineGroup, id, reason string, now time.Time) {
	if s.nca2 == nil || g == nil {
		return
	}
	s.nca2.admitted[id] = false
	s.nca2m.ObservePushout(engineLabel(g.id), reason)
	m := g.threads[id]
	if m == nil {
		return // not seated: nothing on the GPU layer follows from it
	}
	if s.nca2Active(id) {
		s.nca2m.ObserveActivePushout(engineLabel(g.id))
		if !g.index.Exiting(id) {
			g.index.MarkExiting(id)
			m.exitingSince = now
		}
		return
	}
	s.exit(g, m, "pushout", now)
}

// ---- the hooks: L1/L3 at the pod, and the derive ----

// nca2AtPod is L1 (the touch) and L3 (the move), run when a request of the thread REACHES a pod —
// the authoritative moment (the base's at-pod hook). It returns the span's own line facts:
// positionAtTouch (the entry's position AFTER the write) and predictedWarm (its position ≤ L_t
// BEFORE it).
func (s *Scheduler) nca2AtPod(id, engine string, tokens int64, now time.Time) (positionAtTouch int64, predictedWarm bool) {
	if s.nca2 == nil || id == "" || engine == "" {
		return 0, false
	}
	g, ok := s.groups[engine]
	if !ok || g.nca2 == nil {
		return 0, false
	}
	// L3: THE HOME FOLLOWS THE DATA. An observation on a pod other than the home moves the entry
	// there (vLLM holds the prompt on the pod that served it) and the home with it. An admission
	// that already exists is TRANSFERRED, never created (Q1): a pushed-out thread's entry follows
	// its data too, and stays unadmitted.
	if home, hadHome := s.nca2.homes[id]; hadHome && home != engine {
		s.nca2m.ObserveMove(engineLabel(g.id))
	}
	s.nca2HomeMove(id, engine)
	if pos, has := s.nca2PositionOn(g, id); has {
		// the prediction is read BEFORE the write: after it the entry is at the head and the answer
		// would be a constant
		predictedWarm = pos <= s.nca2.TrustLineTokens
	}
	s.nca2LineSet(g, id, tokens, now, pushoutTouch)
	s.nca2m.ObserveLineTouch(engineLabel(g.id), tokens)
	positionAtTouch, _ = s.nca2PositionOn(g, id)
	return positionAtTouch, predictedWarm
}

// nca2DropEntry removes a thread's entry from a line (L3's other half).
func nca2DropEntry(line []nca2Entry, id string) []nca2Entry {
	kept := make([]nca2Entry, 0, len(line))
	for _, e := range line {
		if e.thread != id {
			kept = append(kept, e)
		}
	}
	if len(kept) < len(line) {
		for i := len(kept); i < len(line); i++ {
			line[i] = nca2Entry{}
		}
	}
	return kept
}

// nca2Derive re-derives the nca2 activity aggregates from the request records, once per pass: the
// in-flight count per thread and per pod, and the LEAK flag.
//
// §2's three request lifetimes are applied here, and they are three different questions:
//   - lostRequestSeconds (300): a RELEASED request that never reached a pod stops counting as in
//     flight. Nothing is reported: it is the same bound the base's released window uses (S2), read
//     from the constructor's one field.
//   - maxInflightSeconds (1,200): an AT-POD request older than this is LEAKED for the activity test
//     only — it stops holding its thread active (and stops pinning the room) — and it is counted
//     ONCE, here, in leaked_requests_total.
//   - inflightReapSeconds (9,300): the base's accounting reap, untouched (reapRequests).
func (s *Scheduler) nca2Derive(now time.Time) {
	if s.nca2 == nil {
		return
	}
	for k := range s.nca2.inf {
		delete(s.nca2.inf, k)
	}
	for _, g := range s.groups {
		if g.nca2 != nil {
			g.nca2.inflight = 0
		}
	}
	leakLimit := s.nca2.maxInflightWindow()
	for _, rec := range s.reqs {
		if rec.pod.IsZero() {
			// RELEASED, NOT YET AT A POD: in flight (a released turn counts) until the lost-request
			// bound, after which the router presumes it never arrives.
			if now.Sub(rec.released) > s.lostWindow {
				continue
			}
			s.nca2.inf[rec.thread]++
			continue
		}
		if !rec.leaked && now.Sub(rec.pod) > leakLimit {
			rec.leaked = true
			s.nca2m.ObserveLeaked()
		}
		if rec.leaked {
			continue
		}
		s.nca2.inf[rec.thread]++
		if g := s.groups[rec.engine]; g != nil && g.nca2 != nil {
			g.nca2.inflight++
		}
	}
}

// nca2HomeMove puts the thread's home on `engine` and its entry with every other pod: ONE entry per
// thread, at most, across all the pods (L0's merge, L3's move). The caller inserts the new entry.
func (s *Scheduler) nca2HomeMove(id, engine string) {
	if s.nca2 == nil {
		return
	}
	if home, had := s.nca2.homes[id]; had && home != engine {
		if old, alive := s.groups[home]; alive && old.nca2 != nil {
			old.nca2.line = nca2DropEntry(old.nca2.line, id)
		}
	}
	s.nca2.homes[id] = engine
}

// nca2ResetPod is S5: the pod's line is gone (an engine removal), so its threads lose their
// admission — as a push-out with reason restart, which is what the counters record. The SEATS are
// the base's business (RemoveEngine exits them as engine-gone): this is only the CPU tier's state.
func (s *Scheduler) nca2ResetPod(g *engineGroup, now time.Time) {
	if s.nca2 == nil || g == nil {
		return
	}
	if g.nca2 != nil {
		for admitted := range s.nca2AdmittedSet(g) {
			s.nca2.admitted[admitted] = false
			s.nca2m.ObservePushout(engineLabel(g.id), pushoutRestart)
			if s.nca2Active(admitted) {
				s.nca2m.ObserveActivePushout(engineLabel(g.id))
			}
		}
	}
	for id, home := range s.nca2.homes {
		if home == g.id {
			delete(s.nca2.homes, id) // no home on a pod that no longer exists
		}
	}
	if g.nca2 != nil {
		g.nca2.line = nil
		g.nca2.inflight = 0
	}
}

// nca2WarmReturners is A2.2's list: this pod's ADMITTED threads that are WAITING and hold NO seat
// (a waiter is unseated by construction), LOWEST position first — the nearest the tail, whose
// service moves it to the head and hands the space below it to the room.
//
// The order is total (position, then the oldest arrival, then the id): map iteration never reaches a
// decision.
func (s *Scheduler) nca2WarmReturners(g *engineGroup) []*waiter {
	if s.nca2 == nil || g == nil || g.nca2 == nil {
		return nil
	}
	type warm struct {
		w   *waiter
		pos int64
	}
	var found []warm
	for id, w := range s.waiters {
		if s.nca2.homes[id] != g.id || !s.nca2.admitted[id] {
			continue
		}
		pos, ok := s.nca2AdmittedOn(g, id)
		if !ok {
			continue
		}
		found = append(found, warm{w: w, pos: pos})
	}
	sort.Slice(found, func(i, j int) bool {
		a, b := found[i], found[j]
		if a.pos != b.pos {
			return a.pos < b.pos
		}
		if !a.w.since.Equal(b.w.since) {
			return a.w.since.Before(b.w.since)
		}
		return a.w.id < b.w.id
	})
	out := make([]*waiter, 0, len(found))
	for _, f := range found {
		out = append(out, f.w)
	}
	return out
}

// nca2OutsideHead is A3's head: the oldest on the GLOBAL priority list (a thread that was admitted
// before), and — only when that list is empty — the oldest on the GLOBAL newcomer queue. The lists
// are GLOBAL (all pods), keyed by the head's arrival (waiter.since, the base's own key), with the id
// as the total tie-break.
func (s *Scheduler) nca2OutsideHead() *waiter {
	if s.nca2 == nil {
		return nil
	}
	var priority, newcomer *waiter
	for id, w := range s.waiters {
		if s.nca2.ever[id] {
			if priority == nil || outsideLess(w, priority) {
				priority = w
			}
			continue
		}
		if newcomer == nil || outsideLess(w, newcomer) {
			newcomer = w
		}
	}
	if priority != nil {
		return priority
	}
	return newcomer
}

// outsideLess is the outside lists' total order: the oldest arrival first, then the id.
func outsideLess(a, b *waiter) bool {
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since)
	}
	return a.id < b.id
}

// nca2TrackHead records how long the CURRENT outside head has been the head, at the end of a pass:
// the gauge reads it at scrape time and never writes it (I9).
func (s *Scheduler) nca2TrackHead(now time.Time) {
	if s.nca2 == nil {
		return
	}
	head := s.nca2OutsideHead()
	if head == nil {
		s.nca2.blockedHead, s.nca2.blockedSince = "", time.Time{}
		return
	}
	if head.id != s.nca2.blockedHead {
		s.nca2.blockedHead, s.nca2.blockedSince = head.id, now
	}
}
