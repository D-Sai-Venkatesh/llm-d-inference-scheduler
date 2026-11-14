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
	// THE PASS'S OWN CACHES (V2): a candidate's prompt size, each pod's warm-returner list sorted
	// deepest-first, and the exact-cost memo. The lists are built once here; their ORDER survives the
	// pass because an admission shifts every remaining entry by the same f, and a returner that is
	// seated leaves the waiter queue (the list entry is re-checked before use).
	s.nca2.passMemo = map[fitKey]fitFacts{}
	s.nca2.cand = map[string]nca2Cand{}
	s.nca2.warm = map[string][]*waiter{}
	s.nca2.warmBuilt = false
	s.nca2.passSeq++
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
// fitKey keys the PASS's cost memo (V2). One entry per (pod, candidate) for the whole pass: the index
// changes inside a pass only when a seat is taken or a victim is evicted, and both directions of the
// resulting staleness are safe — a stale HIGH cost makes a head wait one pass, a stale LOW cost is
// caught by the seat's own exact TrySeat/cold charge before anything is granted.
type fitKey struct{ pod, thread string }

type fitFacts struct {
	cost int64 // the head's exact cost in blocks on that pod
	cold int64 // the same cost in tokens (the valve's unit)
	// costKnown / coldKnown: the two numbers are computed LAZILY and independently (each is a walk of
	// the candidate's block keys), so a valve-blocked head never pays for the block cost and a
	// budget-blocked one never pays for the tokens.
	costKnown, coldKnown bool
}

// ncaV2DecideOnce is ONE repetition of A1, in the contract's OWN order: the outside head is tried
// FIRST, over the whole fleet (A2.1), and only if it fits no pod are the warm returners tried, again
// over the fleet (A2.2). The alternative — per pod, head then returners — would let a returner on the
// head's preferred pod end the repetition before the head was tried anywhere else.
func (s *Scheduler) ncaV2DecideOnce(now time.Time) bool {
	head := s.nca2OutsideHead()
	pods := s.ncaV2PodOrder(head)
	memo := s.nca2.passMemo
	// A9: a reservation belongs to ONE head. When the head changes (admitted, gone, replaced) every
	// pod's budget is released with it.
	headID := ""
	if head != nil {
		headID = head.id
	}
	for _, p := range pods {
		if p.g.nca2 != nil && p.g.nca2.reservedHead != headID {
			p.g.nca2.reservedHead, p.g.nca2.reservedBlocks = "", 0
		}
	}
	// PHASE 1 — A2.1: the outside head, the pods in the A5 order (A4's old pod first).
	if head != nil {
		f := s.nca2CandOf(head).tokens
		for _, p := range pods {
			if p.g.numBlocks <= 0 {
				continue // no measured capacity: the pod cannot be costed (the base's rule)
			}
			if f > p.room {
				// f(X) ≤ room(P): the provisional entry must fit the space this pod has. A pod that
				// cannot take the head in SPACE is not asked to hold seats for it either.
				continue
			}
			if !s.ncaV2AdmissionBudget(p.g, head, now, memo) {
				continue // the pod is now holding what it freed for the head (A9)
			}
			if s.nca2Admit(p.g, head, f, p.room, now) {
				return true
			}
		}
	}
	// PHASE 2 — A2.2: the warm returners of each pod in the same order, the deepest position first.
	// A pod that is budgeting for the head does not seat returners: the space it freed is the head's.
	for _, p := range pods {
		if p.g.numBlocks <= 0 {
			continue
		}
		if p.g.nca2 != nil && p.g.nca2.reservedBlocks > 0 && p.g.nca2.reservedHead == headID && headID != "" {
			continue // this pod is holding seats for the head (A9)
		}
		if !s.nca2.warmBuilt {
			// built ONCE per pass, on the first phase-2 visit: a pass that never gets there pays nothing
			for _, g := range s.activeGroups() {
				if g.nca2 != nil && g.numBlocks > 0 {
					s.nca2.warm[g.id] = s.nca2WarmReturners(g)
				}
			}
			s.nca2.warmBuilt = true
		}
		for _, w := range s.nca2.warm[p.g.id] {
			if _, live := s.waiters[w.id]; !live {
				continue // seated by an earlier decision in this pass
			}
			if !s.ncaV2Fits(p.g, w, now, memo) {
				continue
			}
			if !s.ncaV2Reclaim(p.g, w, now, memo) {
				continue
			}
			if s.nca2SeatReturner(p.g, w, p.room, now) {
				return true
			}
		}
	}
	return false
}

// ncaV2AdmissionBudget is A9, the ADMISSION BUDGET of the outside head on one pod: the pod evicts its
// reclaimable seats (the base rotation order) FOR THIS HEAD, records how many blocks it freed on the
// head's behalf, and reports whether the head fits now.
//
// The recorded blocks are what makes the rule work across passes: a prompt larger than one seat needs
// several seats freed, and they become reclaimable one quantum at a time. Without the reservation the
// space freed in one pass was taken by a returner in the same pass (A2.2's phase runs right after),
// the head restarted from zero every ~20 ms, and the whole outside line froze behind it (the X3
// freeze, verifier 2). While the reservation stands this pod's phase 2 is skipped, so the head
// accumulates until it fits. A3 (strict head-of-line) and A2.2's own eligibility are untouched.
//
// The cold valve is checked FIRST: it is not budget, so no eviction can change its answer (C9).
func (s *Scheduler) ncaV2AdmissionBudget(g *engineGroup, head *waiter, now time.Time, memo map[fitKey]fitFacts) bool {
	if s.ncaV2CheapFits(g, head) {
		return true // the cheap upper bound already fits: no index walk, no eviction
	}
	// THE VALVE FIRST and alone (C9), and only the token cost it needs: a head the valve holds never
	// pays for the block walk.
	if !s.coldAllowsTokens(g, s.ncaV2Cold(g, head, memo).cold) {
		return false
	}
	budget := int64(g.budgetBlocks)
	var facts fitFacts
	for {
		facts = s.ncaV2Cost(g, head, memo)
		if facts.cost <= s.freeBudget(g) || (len(g.threads) == 0 && facts.cost > budget) {
			return true
		}
		reason, victim := s.rotationVictim(g, now)
		if victim == nil {
			return false // nothing reclaimable right now: the budget keeps what it freed, for the next pass
		}
		freed := int64(g.index.ChargedBlocks(victim.id))
		s.exit(g, victim, reason, now)
		if g.nca2 != nil {
			// THE RESERVATION STARTS AT THE FIRST SEAT THIS POD FREES FOR THE HEAD: a pod that freed
			// nothing has nothing to hold, and must keep seating its returners (A2.2 untouched)
			g.nca2.reservedHead = head.id
			g.nca2.reservedBlocks += freed
		}
	}
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

// ncaV2Cost is one candidate's exact block cost on one pod, memoized for the PASS (the block index is
// the expensive read: it walks every key of the candidate's chains).
func (s *Scheduler) ncaV2Cost(g *engineGroup, w *waiter, memo map[fitKey]fitFacts) fitFacts {
	key := fitKey{pod: g.id, thread: w.id}
	f, ok := memo[key]
	if !ok {
		f = fitFacts{}
	}
	if !f.costKnown {
		f.cost, f.costKnown = int64(g.index.Cost(w.chains)), true
		memo[key] = f
	}
	return f
}

// ncaV2Cold is the same cost in TOKENS (the cold valve's unit), computed only when the valve is asked.
func (s *Scheduler) ncaV2Cold(g *engineGroup, w *waiter, memo map[fitKey]fitFacts) fitFacts {
	key := fitKey{pod: g.id, thread: w.id}
	f, ok := memo[key]
	if !ok {
		f = fitFacts{}
	}
	if !f.coldKnown {
		f.cold, f.coldKnown = g.index.ColdTokens(w.chains), true
		memo[key] = f
	}
	return f
}

// ncaV2Fits is A2's fits(X, P): the BASE fit, unchanged — the head's exact cost inside the free
// budget plus the blocks this pod's RECLAIMABLE seats would return, OR Rule O for an empty pod, AND
// the cold-entry valve. It is PURE (R1): nothing is evicted to answer the question.
//
// The valve is checked FIRST and alone: it is not budget, so no eviction can change its answer, and a
// head it holds must WAIT (C9).
func (s *Scheduler) ncaV2Fits(g *engineGroup, w *waiter, now time.Time, memo map[fitKey]fitFacts) bool {
	if s.ncaV2CheapFits(g, w) {
		return true // the cheap upper bound fits: no index walk
	}
	// THE VALVE FIRST, and alone: it is not budget, so no eviction could ever change its answer (C9),
	// and a head it refuses must WAIT rather than take a seat from somebody.
	if !s.coldAllowsTokens(g, s.ncaV2Cold(g, w, memo).cold) {
		return false
	}
	if facts := s.ncaV2Cost(g, w, memo); facts.cost <= s.freeBudget(g) {
		return true
	}
	cost := s.ncaV2Cost(g, w, memo).cost
	if len(g.threads) == 0 && cost > int64(g.budgetBlocks) {
		return true // RULE O: an empty pod admits one oversize head
	}
	return cost <= s.freeBudget(g)+s.ncaV2ReclaimableBlocks(g, now)
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
	budget := int64(g.budgetBlocks)
	for {
		facts := s.ncaV2Cost(g, w, memo)
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
	// THE WARM PREDICTION FOR AN ADMISSION IS READ AT THE DECISION, BEFORE L0 (§7, the owner's
	// intent: "is the thread's data already on that pod's CPU tier"): a priority thread re-admitted at
	// its old pod whose entry is still inside the trust line is warm; a newcomer, or an entry past
	// L_t, is not. L0's provisional entry must not answer this — it is the request's own tokens, not
	// data the pod holds.
	warm := s.nca2PreTouchWarm(g, w.id)
	if !s.ncaV2Seat(g, w, class, path, room, warm, now) {
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

// nca2PreTouchWarm is §7's predicted_warm for an A2.1 admission: the thread's OWN entry, already on
// this pod (a priority thread re-admitted at its old pod), at a position inside the trust line. No
// entry — or one past L_t — means the pod does not hold the thread's data warmly.
func (s *Scheduler) nca2PreTouchWarm(g *engineGroup, id string) bool {
	pos, has := s.nca2PositionOn(g, id)
	return has && pos <= s.nca2.TrustLineTokens
}

// nca2SeatReturner is A2.2: the pod's admitted thread takes the freed seat. Nothing about the line
// changes here — its entry is already inside the line, and L1 moves it to the head when its request
// reaches the pod.
func (s *Scheduler) nca2SeatReturner(g *engineGroup, w *waiter, room int64, now time.Time) bool {
	// a returner's prediction is the position its entry holds when its request REACHES the pod (the
	// at-pod hook reads it before the touch, and this seat changes nothing on the line)
	if !s.ncaV2Seat(g, w, nca2ClassReturner, pathNone, room, false, now) {
		return false
	}
	s.nca2m.ObserveReturnerSeat(engineLabel(g.id))
	s.nca2m.ObserveSeatWait(nca2ClassReturner, nonNegative(now.Sub(w.since)))
	return true
}

// ncaV2Seat takes the GPU seat through the BASE's own seat path (the same TrySeat, Rule O, cold
// charge, SEAT event and entry histogram a shipped seat takes) and stamps the turn's nca2 class, path
// and the room it was decided on, for the span PreRequest sets.
func (s *Scheduler) ncaV2Seat(g *engineGroup, w *waiter, class, path string, room int64, warm bool, now time.Time) bool {
	tokens := g.index.ColdTokens(w.chains)
	if !s.seat(g, w, tokens, now) {
		return false
	}
	if m := g.threads[w.id]; m != nil {
		m.nca2Class, m.nca2Room = class, room
		if m.seatObs != nil {
			m.seatObs.nca2 = nca2Obs{class: class, path: path, room: room,
				predictedWarm:   warm,
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
