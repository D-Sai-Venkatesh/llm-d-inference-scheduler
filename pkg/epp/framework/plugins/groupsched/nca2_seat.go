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
	limit := ncaV2DecisionCap(len(s.waiters), len(s.groups))
	s.nca2.passDecisions, s.nca2.capHit = 0, false
	decisions := 0
	for ; decisions < limit && len(s.waiters) > 0; decisions++ {
		if !s.ncaV2DecideOnce(now) {
			break
		}
		s.nca2.passDecisions++
	}
	// the CAP was the reason the pass stopped, not the absence of a candidate: the value is reported so
	// a run can tell the two apart (and the invariant test can tell a real work-conservation gap from a
	// cap that was doing its job)
	if decisions >= limit && len(s.waiters) > 0 {
		s.nca2.capHit = true
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

// ncaV2DecisionCap is A1's safety cap, as a function because the RULE is the function: a pass takes at
// most (waiters + 4 × pods) decisions. It only ever binds when a pass's own evictions re-queue waiters
// (a pathological loop); in every ordinary pass each decision removes one waiter, so the cap cannot be
// reached. It is reported (`decisionCapHit`) rather than silently truncating.
func ncaV2DecisionCap(waiters, pods int) int { return waiters + 4*pods }

// podState is ONE pod's facts for ONE repetition (I8/R3): its A5 keys and its room, computed once,
// so a decision costs O(pods × (line + warm returners)) and the same number is never computed twice.
type podState struct {
	g        *engineGroup
	room     int64
	inFlight int
}

// fitKey / fitFacts memoize one candidate's cost ON ONE POD for the span of one repetition: the block
// index cannot change inside a repetition except by a reclaim, and a reclaim ends the repetition.
type fitKey struct{ pod, thread string }

type fitFacts struct {
	cost int64 // the head's exact cost in blocks on that pod
	cold int64 // the same cost in tokens (the valve's unit)
}

// ncaV2DecideOnce is ONE repetition of A1, in the contract's OWN order: the outside head is tried
// FIRST, over the whole fleet (A2.1), and only if it fits no pod are the warm returners tried, again
// over the fleet (A2.2). The alternative — per pod, head then returners — would let a returner on the
// head's preferred pod end the repetition before the head was tried anywhere else.
func (s *Scheduler) ncaV2DecideOnce(now time.Time) bool {
	head := s.nca2OutsideHead()
	pods := s.ncaV2PodOrder(head)
	memo := map[fitKey]fitFacts{}
	// PHASE 1 — A2.1: the outside head, the pods in the A5 order (A4's old pod first).
	if head != nil {
		f := PromptTokens(head.chains)
		for _, p := range pods {
			if p.g.numBlocks <= 0 {
				continue // no measured capacity: the pod cannot be costed (the base's rule)
			}
			if f > p.room {
				continue // f(X) ≤ room(P): the provisional entry must fit the space this pod has
			}
			if !s.ncaV2Fits(p.g, head, now, memo) {
				continue
			}
			if !s.ncaV2Reclaim(p.g, head, now, memo) {
				// the evictions fell short (shared blocks): this pod cannot take the head, and the
				// seats it did not need stay in place
				clear(memo)
				continue
			}
			if s.nca2Admit(p.g, head, f, p.room, now) {
				return true
			}
		}
	}
	// PHASE 2 — A2.2: the warm returners of each pod in the same order, the deepest position first.
	for _, p := range pods {
		if p.g.numBlocks <= 0 {
			continue
		}
		for _, w := range s.nca2WarmReturners(p.g) {
			if !s.ncaV2Fits(p.g, w, now, memo) {
				continue
			}
			if !s.ncaV2Reclaim(p.g, w, now, memo) {
				clear(memo)
				continue
			}
			if s.nca2SeatReturner(p.g, w, p.room, now) {
				return true
			}
		}
	}
	return false
}

// ncaV2PodOrder is A5, plus A4's preference: the fewest requests IN FLIGHT first (the pod whose GPU
// is the least busy), then the largest ROOM (the most free CPU), then the pod id — and, when the
// outside head is a pushed-out thread whose old entry still exists on a pod, that pod FIRST (A4: a
// priority thread prefers its old pod; if it does not fit there, the rest of the order follows).
func (s *Scheduler) ncaV2PodOrder(head *waiter) []podState {
	groups := s.activeGroups()
	pods := make([]podState, 0, len(groups))
	for _, g := range groups {
		p := podState{g: g, room: s.nca2View(g).room(s.nca2.AdmitLineTokens)}
		if g.nca2 != nil {
			p.inFlight = g.nca2.inflight
		}
		pods = append(pods, p)
	}
	sort.SliceStable(pods, func(i, j int) bool {
		a, b := pods[i], pods[j]
		if a.inFlight != b.inFlight {
			return a.inFlight < b.inFlight
		}
		if a.room != b.room {
			return a.room > b.room
		}
		return a.g.id < b.g.id
	})
	if head != nil {
		if home, ok := s.nca2.homes[head.id]; ok {
			for i := range pods {
				if pods[i].g.id != home {
					continue
				}
				if _, has := s.nca2PositionOn(pods[i].g, head.id); has && i > 0 {
					preferred := pods[i]
					copy(pods[1:i+1], pods[:i])
					pods[0] = preferred
				}
				break
			}
		}
	}
	return pods
}

// ncaV2Cost is one candidate's cost on one pod, memoized for the repetition (the block index is the
// expensive read: it walks the candidate's keys).
func (s *Scheduler) ncaV2Cost(g *engineGroup, w *waiter, memo map[fitKey]fitFacts) fitFacts {
	key := fitKey{pod: g.id, thread: w.id}
	if f, ok := memo[key]; ok {
		return f
	}
	f := fitFacts{cost: int64(g.index.Cost(w.chains)), cold: g.index.ColdTokens(w.chains)}
	memo[key] = f
	return f
}

// ncaV2Fits is A2's fits(X, P): the BASE fit, unchanged — the head's exact cost inside the free
// budget plus the blocks this pod's RECLAIMABLE seats would return, OR Rule O for an empty pod, AND
// the cold-entry valve. It is PURE (R1): nothing is evicted to answer the question.
//
// The valve is checked FIRST and alone: it is not budget, so no eviction can change its answer, and a
// head it holds must WAIT (C9).
func (s *Scheduler) ncaV2Fits(g *engineGroup, w *waiter, now time.Time, memo map[fitKey]fitFacts) bool {
	// THE VALVE FIRST, and alone: it is not budget, so no eviction could ever change its answer (C9),
	// and a head it refuses must WAIT rather than take a seat from somebody.
	facts := s.ncaV2Cost(g, w, memo)
	if !s.coldAllowsTokens(g, facts.cold) {
		return false
	}
	if facts.cost <= s.freeBudget(g) {
		return true
	}
	if len(g.threads) == 0 && facts.cost > int64(g.budgetBlocks) {
		return true // RULE O: an empty pod admits one oversize head
	}
	return facts.cost <= s.freeBudget(g)+s.ncaV2ReclaimableBlocks(g, now)
}

// ncaV2ReclaimableBlocks is the most this pod can free WITHOUT touching a thread with work in flight
// (S1: the base's eligible() — Q completed busy periods, or idleSeconds of silence).
func (s *Scheduler) ncaV2ReclaimableBlocks(g *engineGroup, now time.Time) int64 {
	var blocks int64
	for id, m := range g.threads {
		if g.index.Exiting(id) {
			continue // already leaving: the base rotation never returns one, and it frees nothing twice
		}
		if _, eligible := s.eligible(m, now); !eligible {
			continue
		}
		blocks += int64(g.index.ChargedBlocks(id))
	}
	return blocks
}

// ncaV2Reclaim evicts the pod's reclaimable seats in the base rotationVictim order, ONE at a time,
// until the candidate fits. It is the only place a seat is taken for an admission, and only for the
// candidate the fit test already accepted (R1). If the evictions fall short — seats that SHARE blocks
// free less than their own footprint — it stops and leaves the rest in place.
func (s *Scheduler) ncaV2Reclaim(g *engineGroup, w *waiter, now time.Time, memo map[fitKey]fitFacts) bool {
	facts := s.ncaV2Cost(g, w, memo)
	budget := int64(g.budgetBlocks)
	for {
		if facts.cost <= s.freeBudget(g) || (len(g.threads) == 0 && facts.cost > budget) {
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
	if !s.ncaV2Seat(g, w, class, path, room, now) {
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
func (s *Scheduler) nca2SeatReturner(g *engineGroup, w *waiter, room int64, now time.Time) bool {
	if !s.ncaV2Seat(g, w, nca2ClassReturner, pathNone, room, now) {
		return false
	}
	s.nca2m.ObserveReturnerSeat(engineLabel(g.id))
	s.nca2m.ObserveSeatWait(nca2ClassReturner, nonNegative(now.Sub(w.since)))
	return true
}

// ncaV2Seat takes the GPU seat through the BASE's own seat path (the same TrySeat, Rule O, cold
// charge, SEAT event and entry histogram a shipped seat takes) and stamps the turn's nca2 class, path
// and the room it was decided on, for the span PreRequest sets.
func (s *Scheduler) ncaV2Seat(g *engineGroup, w *waiter, class, path string, room int64, now time.Time) bool {
	tokens := g.index.ColdTokens(w.chains)
	if !s.seat(g, w, tokens, now) {
		return false
	}
	if m := g.threads[w.id]; m != nil {
		m.nca2Class, m.nca2Room = class, room
		if m.seatObs != nil {
			m.seatObs.nca2 = nca2Obs{class: class, path: path, room: room,
				pushedOutBefore: s.nca2.pushed[w.id],
				doorWait:        now.Sub(w.since), seatWait: now.Sub(w.since),
				clientRequestID: w.clientReqID, wireID: w.wireID, threadID: w.id, valid: true}
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
