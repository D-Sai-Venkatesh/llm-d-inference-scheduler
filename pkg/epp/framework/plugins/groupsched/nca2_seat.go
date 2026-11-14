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

// THE nca2 ADMISSION (nca2 spec §4, A0–A7): who gets the space, who gets the seat, and on which pod
// — inside the base Pass, after syncQueue + deriveColdCharges + enforceBudgets and before dispatch.
//
// nca2 replaces EXACTLY ONE step of the pass: seatWaiters, the unseated heads. Everything else — the
// seated threads' dispatch (A0), the budget pressure, the quantum/idle rotation, the cold valve,
// Rule O, the in-flight cap — is the shipped policy, untouched.
//
// The decision is ONE question per repetition: on the first pod of the A5 order that can decide,
// either the outside HEAD is admitted (A2.1: it fits the room and the GPU) or one of that pod's warm
// returners is SEATED (A2.2: nearest the tail first). The pods are re-ordered before every
// repetition, which is what makes the loop converge and keeps it from funnelling.

import (
	"sort"
	"time"
)

// ncaV2SeatWaiters is A1: repeat A2 until it seats nobody. The safety cap (waiters + 4 × pods) bounds the
// pass: every decision either seats a thread (removing it from the waiter queue) or ends the pass,
// and the slack covers the pods an admission may evict on and the merges of a chained push-out.
func (s *Scheduler) ncaV2SeatWaiters(now time.Time) {
	if s.nca2 == nil {
		return
	}
	cap := len(s.waiters) + 4*len(s.groups)
	for i := 0; i < cap && len(s.waiters) > 0; i++ {
		if !s.ncaV2DecideOnce(now) {
			break
		}
	}
	// nothing more could be seated: if the outside head is held by an engine's cold-entry valve (it
	// would fit the budget, the valve refuses it), that is the shipped valve counter's event
	if head := s.nca2OutsideHead(); head != nil {
		if engine, held := s.coldValveHeldOn(head); held {
			s.metrics.ObserveColdValveHeld(engineLabel(engine))
		}
	}
	// head_blocked_seconds' input: which thread is the outside head, and since when. Written here,
	// only read by the Snapshot (I9).
	s.nca2TrackHead(now)
}

// ncaV2DecideOnce is ONE repetition of A1: the pods in the A5 order, and the first one that can
// decide. The order is recomputed here, after the previous decision, which is why one admission can
// never funnel the next one onto the same pod.
func (s *Scheduler) ncaV2DecideOnce(now time.Time) bool {
	for _, g := range s.ncaV2PodOrder() {
		if g.numBlocks <= 0 {
			continue // no measured capacity: the pod cannot be costed (the base's rule)
		}
		if s.ncaV2Decide(g, now) {
			return true
		}
	}
	return false
}

// podRank is one pod's A5 keys, computed once per ordering: map iteration and repeated line walks
// never decide an order.
type podRank struct {
	g        *engineGroup
	room     int64
	inFlight int
}

// ncaV2PodOrder is A5, plus A4's preference: the fewest requests IN FLIGHT first (the pod whose GPU
// is the least busy), then the largest ROOM (the most free CPU), then the pod id — and, when the
// outside head is a pushed-out thread whose old entry still exists on a pod, that pod FIRST (A4: a
// priority thread prefers its old pod; if it does not fit there, the rest of the order follows).
func (s *Scheduler) ncaV2PodOrder() []*engineGroup {
	groups := s.activeGroups()
	ranks := make([]podRank, 0, len(groups))
	for _, g := range groups {
		r := podRank{g: g}
		if g.nca2 != nil {
			r.inFlight = g.nca2.inflight
		}
		r.room = s.nca2View(g).room(s.nca2.AdmitLineTokens)
		ranks = append(ranks, r)
	}
	sort.SliceStable(ranks, func(i, j int) bool {
		a, b := ranks[i], ranks[j]
		if a.inFlight != b.inFlight {
			return a.inFlight < b.inFlight
		}
		if a.room != b.room {
			return a.room > b.room
		}
		return a.g.id < b.g.id
	})
	// A4: a pushed-out thread PREFERS its old pod, when its old entry still exists there (the fit test
	// is the decision's own: if it does not fit there, the rest of the order is tried next).
	if head := s.nca2OutsideHead(); head != nil {
		if home, ok := s.nca2.homes[head.id]; ok {
			for i := range ranks {
				if ranks[i].g.id != home {
					continue
				}
				if _, has := s.nca2PositionOn(ranks[i].g, head.id); has && i > 0 {
					preferred := ranks[i]
					copy(ranks[1:i+1], ranks[:i])
					ranks[0] = preferred
				}
				break
			}
		}
	}
	ordered := make([]*engineGroup, 0, len(ranks))
	for _, r := range ranks {
		ordered = append(ordered, r.g)
	}
	return ordered
}

// ncaV2Decide is A2 on ONE pod, in the contract's own order: the outside head first (A2.1), then the
// pod's warm returners nearest the tail (A2.2), and no decision at all if neither can be taken.
func (s *Scheduler) ncaV2Decide(g *engineGroup, now time.Time) bool {
	room := s.nca2View(g).room(s.nca2.AdmitLineTokens)
	head := s.nca2OutsideHead()
	if head != nil {
		// f(X): the prompt tokens of the head's OWN request — the size the provisional entry costs
		// the CPU tier at this decision.
		f := PromptTokens(head.chains)
		if f <= room && s.ncaV2Fits(g, head, now) {
			if s.nca2Admit(g, head, f, room, now) {
				return true
			}
		}
	}
	// A2.2: this pod's admitted threads that are waiting with no seat, the one nearest the tail
	// first. Serving it moves it to the head, so the space below it joins the room (the [H] reason
	// the order is not "the longest waiting").
	for _, w := range s.nca2WarmReturners(g) {
		if !s.ncaV2Fits(g, w, now) {
			continue
		}
		if s.nca2SeatReturner(g, w, now) {
			return true
		}
	}
	return false
}

// ncaV2Fits is A2's fits(X, P): the BASE fit, unchanged — the head's exact cost inside the free
// budget, OR Rule O for an empty pod, AND the cold-entry valve. The valve is checked FIRST and
// alone, with no eviction: evicting frees budget, never the valve (C9), so a head the valve holds
// must WAIT.
//
// When the budget is the blocker, the pod's RECLAIMABLE seats (S1: the base's eligible(), i.e. Q
// completed busy periods or idleSeconds of silence, never an in-flight thread) are evicted in the
// base rotationVictim order until the head fits (X2).
func (s *Scheduler) ncaV2Fits(g *engineGroup, w *waiter, now time.Time) bool {
	// THE VALVE FIRST, and alone: it is not budget, so no eviction can ever change its answer (C9).
	if !s.coldAllowsTokens(g, g.index.ColdTokens(w.chains)) {
		return false
	}
	cost := int64(g.index.Cost(w.chains))
	budget := int64(g.budgetBlocks)
	for {
		if cost <= s.freeBudget(g) || (len(g.threads) == 0 && cost > budget) {
			// the budget's two halves: it fits by itself, or Rule O admits one oversize head onto an
			// EMPTY pod (the base's rule, unchanged)
			return true
		}
		reason, victim := s.rotationVictim(g, now)
		if victim == nil {
			return false
		}
		s.exit(g, victim, reason, now)
	}
}

// nca2Admit is A2.1: the head ADMITTED on this pod — home, the admission STATE, and L0's
// provisional entry — and SEATED at once (the seat frees first: S1's rotation is what a blocked
// admission evicts).
func (s *Scheduler) nca2Admit(g *engineGroup, w *waiter, f, room int64, now time.Time) bool {
	path, class := pathNewcomer, nca2ClassAdmittedNow
	if s.nca2.ever[w.id] {
		path = pathPriority
	}
	if !s.ncaV2Seat(g, w, class, room, now) {
		return false
	}
	// L0: the provisional entry at the HOME's head with tokens = f. The home is this pod (A4 let it
	// prefer its old one, the A5 order decided otherwise), so an entry left on another pod is
	// DELETED — one entry per thread, at most, across all the pods.
	s.nca2HomeMove(w.id, g.id)
	s.nca2.admitted[w.id] = true
	s.nca2.ever[w.id] = true
	s.nca2LineSet(g, w.id, f, now, pushoutRoom)
	doorWait := now.Sub(w.since)
	s.nca2m.ObserveAdmission(engineLabel(g.id), path, nonNegative(doorWait))
	s.nca2m.ObserveSeatWait(class, nonNegative(doorWait))
	return true
}

// nca2SeatReturner is A2.2: the pod's admitted thread takes the freed seat. Nothing about the line
// changes here — its entry is already inside the line, and L1 moves it to the head when its request
// reaches the pod.
func (s *Scheduler) nca2SeatReturner(g *engineGroup, w *waiter, now time.Time) bool {
	if !s.ncaV2Seat(g, w, nca2ClassReturner, s.nca2View(g).room(s.nca2.AdmitLineTokens), now) {
		return false
	}
	s.nca2m.ObserveReturnerSeat(engineLabel(g.id))
	s.nca2m.ObserveSeatWait(nca2ClassReturner, nonNegative(now.Sub(w.since)))
	return true
}

// ncaV2Seat takes the GPU seat through the BASE's own seat path (the same TrySeat, Rule O, cold
// charge, SEAT event and entry histogram a shipped seat takes) and stamps the turn's nca2 class and
// the room it was decided on, for the span PreRequest sets.
func (s *Scheduler) ncaV2Seat(g *engineGroup, w *waiter, class string, room int64, now time.Time) bool {
	tokens := g.index.ColdTokens(w.chains)
	if !s.seat(g, w, tokens, now) {
		return false
	}
	if m := g.threads[w.id]; m != nil {
		m.nca2Class, m.nca2Room = class, room
		if !s.nca2.ever[w.id] {
			m.nca2Pushed = false
		} else {
			m.nca2Pushed = !s.nca2.admitted[w.id]
		}
		if m.seatObs != nil {
			m.seatObs.nca2 = nca2Obs{class: class, room: room, pushedOutBefore: m.nca2Pushed}
		}
	}
	return true
}

// The outside classes and the seat classes (§7's labels), spelled once.
const (
	pathPriority = "priority"
	pathNewcomer = "newcomer"
	// pathNone is a request that was not admitted by a decision: a seated thread's later turn.
	pathNone = "none"

	// nca2ClassAdmittedNow is a seat taken by A2.1 (the thread came from OUTSIDE).
	nca2ClassAdmittedNow = "admitted_now"
	// nca2ClassReturner is a seat taken by A2.2 (the thread was admitted and is warm on this pod).
	nca2ClassReturner = "returner"
	// nca2ClassSeated is a later turn of a thread that already holds its seat.
	nca2ClassSeated = "seated"
)
