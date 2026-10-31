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

// SEAT = PIN (bindingMode "seat", arm E4a).
//
// The multi-pod binding model where the member→engine binding IS the member's
// CURRENT SEAT: created when the member is seated on an engine, dropped the
// moment it exits. Nothing else creates a binding — no first-sight homing, no
// TTL, no residue — so an unseated member has NO placement opinion and its head
// is ordinary queue demand. That is the whole difference from member-stable
// (E3), where the engine is chosen at first sight and kept for
// BindingTTLSeconds so a silent member returns to the engine holding its
// blocks.
//
// Three structural consequences follow, and each has exactly one home here:
//
//  1. WAITERS ARE ONE GLOBAL POOL (scheduler.go: s.gWaiters). A waiter is not
//     homed, so there is nothing for an engine to own; each rotate pass offers
//     the global head to an engine that has room. rotateSeatPoolLocked is that
//     loop.
//  2. THE ENGINE IS CHOSEN AT SEAT TIME by seatEngineLocked: fit → least
//     relative backlog per seat → cheapest-victim engine. It takes a SCORED
//     CANDIDATE LIST, which is the E4b seam: the parameter-free residue model
//     (predicted surviving prefix blocks per engine) slots in as one more field
//     on seatCandidate plus one more ordering step, without touching the fit or
//     victim logic. NO residue term is computed here — whether it is built at
//     all is decided by the offline calibration, not by this build.
//  3. PLACEMENT IS PINNED BY OUR OWN FILTER (filter.go). The framework picks
//     the endpoint; with seat=pin the policy's Filter narrows the candidate set
//     to the engine the member is SEATED on, so PreRequest's observed endpoint
//     agrees with the charge instead of repairing it every turn (the E3 arm
//     needed the external session-affinity-filter for exactly this, and the
//     un-pinned E1 defect measured the cost of not having it: 50 % of
//     dispatches off-binding).

import "time"

// gapHistoryLimit bounds the per-member inter-turn gap history (the F4 pause
// rule's own median). 16 is a sample bound, not a tuning constant: a median over
// the last 16 gaps tracks a client's think time without remembering the whole
// run, and it is small enough to keep the pool scan O(1) per look.
const gapHistoryLimit = 16

// noteTurnArrivalLocked records a member's inter-turn gap from its head's own
// enqueue time. Called from syncWaitersLocked for every head sighting; a head is
// re-seen by every Pick pass until it dispatches, so the guard is "strictly
// later arrival than the last one recorded" — one gap per TURN, never per pass.
//
// The history is member-scoped and survives rotation, exit and re-entry on
// purpose: the gap is a property of the client's think time, and the population
// the pause rule must judge is exactly the one that rotates.
func (s *Scheduler) noteTurnArrivalLocked(id string, arrival time.Time) {
	if id == "" || arrival.IsZero() {
		return
	}
	prev, seen := s.turnSeen[id]
	if !seen {
		s.turnSeen[id] = arrival
		return
	}
	if !arrival.After(prev) {
		return // the same head, re-observed
	}
	s.turnSeen[id] = arrival
	gap := arrival.Sub(prev).Seconds()
	if gap <= 0 {
		return
	}
	h := s.gapHistory[id]
	if len(h) >= gapHistoryLimit {
		h = append(h[:0], h[1:]...) // ring: keep the most recent gaps
	}
	s.gapHistory[id] = append(h, gap)
}

// forgetTurnHistoryLocked drops a member's gap history with its record (the
// silent-record GC): a member the policy has forgotten must not steer the
// population median forever.
func (s *Scheduler) forgetTurnHistoryLocked(id string) {
	delete(s.turnSeen, id)
	delete(s.gapHistory, id)
}

// medianGap is the median of a gap slice (0 when empty). Even counts average the
// two middle samples — a median, not a mean, because think-time gaps are
// heavy-tailed and one 3,000 s outlier must not make a member un-evictable.
func medianGap(h []float64) float64 {
	n := len(h)
	if n == 0 {
		return 0
	}
	cp := append([]float64(nil), h...)
	for i := 1; i < len(cp); i++ { // insertion sort: n <= 16
		for j := i; j > 0 && cp[j] < cp[j-1]; j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

// poolColdTokensLocked is the cold-entry estimate of a waiting head that has NO
// engine yet: the BEST CASE over the live engines (what the head would cost if
// it landed where most of its blocks already are). The global pool cannot know
// its engine before the seat-time choice runs, and the cold-entry stagger bounds
// concurrent cold PREFILL, so the optimistic cost is the right input here; the
// per-engine fit test re-costs the head against the engine actually chosen.
func (s *Scheduler) poolColdTokensLocked(h Head) int64 {
	best := int64(-1)
	for _, g := range s.activeGroupsLocked() {
		e := s.estimateCostCachedLocked(g, h.Member, h.SizeBytes)
		if best < 0 || e < best {
			best = e
		}
	}
	if best < 0 {
		best = 0
	}
	return best * int64(s.blockSize)
}

// seatCandidate is ONE engine's score for seating ONE waiting head — the input
// of the engine-choice function, and the E4B SEAM.
// E4a orders on (fits, backlogPerSeat). The parameter-free residue model, IF the
// offline calibration passes, adds `residueBlocks` (predicted surviving prefix
// blocks of this member on this engine, from the scheduler's own residue FIFO)
// and takes the argmax among fitting engines before falling back to backlog.
// Nothing else moves: the fit test, the cold-cap check and the victim step are
// engine-agnostic.
type seatCandidate struct {
	g *engineGroup
	// fits is this head's admission cost against THIS engine's budget.
	fits bool
	// costTokens / costBlocks are the estimate the fit test used (kept so the
	// caller can hand the chosen engine's numbers to seatLocked instead of the
	// pool's best-case guess).
	costTokens int64
	costBlocks int64
	// backlogPerSeat is the engine's relative backlog (queued demand per seat
	// held) — the cross-engine fairness key, never kv_cache_usage (pinned ~0.95
	// by construction under the group scheduler).
	backlogPerSeat float64
	// seats is the seats the engine currently holds — the SPREAD tie-break.
	// Under seat=pin the queued demand is one global pool, so backlogPerSeat is
	// the same demand divided by a bigger number for a bigger engine: it
	// strictly prefers the engine that already holds more seats. That is right
	// when a queue exists (the engine with more capacity should absorb it) and
	// degenerate when it does not (every engine ties at 0 and the id order would
	// pile every newcomer onto one engine). Equal backlog therefore breaks on
	// FEWER seats, which spreads the cold newcomers — they have no cache to lose
	// — and leaves the concentration to the residue tie-break (E4b), which
	// prefers the engine that actually holds the member's blocks.
	seats int
	// hasVictim marks an engine that can evict an eligible incumbent right now
	// (the cheapest-victim engine's input).
	hasVictim bool
}

// seatEngineLocked is the SEAT-TIME ENGINE CHOICE. Order:
//
//  1. FIT — an engine that can take this head without evicting anyone.
//  2. among the fitting: LEAST RELATIVE BACKLOG PER SEAT (the same key Pick uses
//     across engines; ties break on the engine id so the decision is total and
//     every multi-engine run stays reproducible).
//  3. nothing fits: the CHEAPEST-VICTIM ENGINE — among engines holding an
//     eligible rotation victim, the one with the lowest relative backlog per
//     seat (evicting on the least-pressured engine gives up the least warm state
//     per unit of queued demand). Ties break on the engine id. E4b's residue
//     argmax, if it ships, replaces this ordering with "the engine whose victim's
//     residue is least valuable" — the field is already on the candidate.
//
// Returns nil when there is no engine at all (pre-capacity, no endpoints).
func (s *Scheduler) seatEngineLocked(head *waiter, now time.Time) *seatCandidate {
	groups := s.activeGroupsLocked()
	if len(groups) == 0 {
		return nil
	}
	// prefer engines whose capacity is MEASURED (an engine whose metrics never
	// arrived cannot be costed); with none, cost against the whole set so the
	// pre-capacity behaviour is the member-stable one.
	ready := make([]*engineGroup, 0, len(groups))
	for _, g := range groups {
		if g.numBlocks > 0 {
			ready = append(ready, g)
		}
	}
	if len(ready) == 0 {
		ready = groups
	}
	cands := make([]seatCandidate, 0, len(ready))
	for _, g := range ready {
		// THE FIT TEST USES THE HEAD'S OWN SYNC-TIME ESTIMATE — the SAME input the
		// member-stable admission loop uses (head.coldTokens, refreshed by
		// syncWaiters on every Pick pass). A per-engine, recomputed index estimate
		// here would be a SECOND undeclared variable between the E3 and E4a arms,
		// and a large one: measured on the golden scenario, a fresh estimate for a
		// returning member jumps ~10x the instant a member that held its shared
		// prefix is evicted (28 → 328 blocks at t=360), so the two models would
		// admit on different numbers. The per-engine index state still decides the
		// BUDGET side (ne) and, later, the E4b residue tie-break.
		blocks := head.coldTokens / int64(s.blockSize)
		tokens := head.coldTokens
		ne := int64(g.index.UniqueBlocksNELocked()) + g.pendingTotal
		c := seatCandidate{
			g: g, fits: g.budgetBlocks <= 0 || ne+blocks <= int64(g.budgetBlocks),
			costTokens: tokens, costBlocks: blocks,
			backlogPerSeat: s.seatBacklogLocked(g),
			seats:          len(g.members),
		}
		cands = append(cands, c)
	}
	// 2. fit first.
	var best *seatCandidate
	for i := range cands {
		c := &cands[i]
		if !c.fits {
			continue
		}
		if betterSeat(*c, best) {
			best = c
		}
	}
	if best != nil {
		return best
	}
	// 3. nothing fits: the cheapest-victim engine. hasVictim is filled lazily —
	// rotationVictimLocked allocates nothing, but it is O(members) and the common
	// path (a fitting engine exists) must not pay for it.
	for i := range cands {
		cands[i].hasVictim = s.hasRotationVictimLocked(cands[i].g, now)
	}
	for i := range cands {
		c := &cands[i]
		if !c.hasVictim {
			continue
		}
		if betterSeat(*c, best) {
			best = c
		}
	}
	return best
}

// seatBacklogLocked is the seat choice's relative-backlog key: the engine's OWN
// queued demand — seated members that have a head waiting RIGHT NOW — per seat it
// holds.
//
// It is deliberately NOT backlogPerSeatLocked (which adds the pool's unseated
// waiters). Under seat=pin the unseated demand is ONE GLOBAL POOL, so that term
// is identical for every engine and the key degenerates to "demand / seats":
// the engine already holding more seats scores LOWER and attracts the next seat,
// a runaway that hands a burst's whole seat set to whichever engine took the
// first seat (measured on the first cut of this loop: 11 seats on engine-a, 1 on
// engine-b, from a 12-member burst). Owned queued demand is the honest per-engine
// number: a head nobody has seated yet is not waiting ON any engine.
func (s *Scheduler) seatBacklogLocked(g *engineGroup) float64 {
	seats := len(g.members)
	if seats <= 0 {
		seats = 1
	}
	queued := 0
	for _, m := range g.members {
		if !m.waitingSince.IsZero() {
			queued++
		}
	}
	return float64(queued) / float64(seats)
}

// betterSeat is the total order over seat candidates: least relative backlog per
// seat, then FEWER seats held (the spread tie-break), then engine id (so the
// decision is total and every multi-engine run is reproducible).
func betterSeat(c seatCandidate, best *seatCandidate) bool {
	if best == nil {
		return true
	}
	if c.backlogPerSeat != best.backlogPerSeat {
		return c.backlogPerSeat < best.backlogPerSeat
	}
	if c.seats != best.seats {
		return c.seats < best.seats
	}
	return c.g.id < best.g.id
}

// hasRotationVictimLocked reports whether an engine can evict somebody right
// now. It reuses rotationVictimLocked so the victim criteria (quantum/ledger
// eligibility, the per-engine idle fallback, the F-knobs) have ONE definition.
func (s *Scheduler) hasRotationVictimLocked(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictimLocked(g, now)
	return v != nil
}

// seatableHeadLocked walks the GLOBAL pool and returns the D1-BEST head that
// some engine can seat RIGHT NOW (fit AND cold-cap), with its engine choice.
// FIRST-FIT over the ORDERED pool, not head-only.
//
// The distinction is the 02:13Z v2 freeze class: the pre-v2 code checked only
// the single oldest head, so one head that never fits (a huge cold entry, or a
// head blocked by the cold-entry cap) blocked EVERY dispatch — the oldest head
// never fit, no busy period completed, no quantum-complete victim existed, and
// waiting members were never idled. Under one global pool that hazard is
// strictly worse than it was per engine, so the scan looks past blocked heads
// and seats the best head that CAN go (the blocked head keeps its `since`, so
// it stays first in D1 order for the next pass).
//
// Implemented as a single min-scan over the pool (no sort): the pass stays O(W)
// with no allocation, which matters because rotate runs inside every Pick.
func (s *Scheduler) seatableHeadLocked(seats map[string]int, now time.Time) (*waiter, *seatCandidate) {
	var bestW *waiter
	var bestC *seatCandidate
	var bestK entryKey
	for _, w := range s.gWaiters {
		cand := s.seatEngineLocked(w, now)
		if cand == nil || !cand.fits {
			continue
		}
		if !s.coldAllows(cand.g, w) {
			continue // the cold cap, not the budget: try the next head
		}
		k := s.entryKeyLocked(w, seats)
		if bestW == nil || k.less(bestK) {
			bestW, bestC, bestK = w, cand, k
		}
	}
	return bestW, bestC
}

// rotateSeatPoolLocked is the seat=pin admission loop: ONE global pool, the D1
// head at a time, the engine chosen by the seat-time rule.
//
// This is the seat model's answer to "the engine-idle fallback and the rotate
// loop are per engine". The VICTIM side stays per engine (rotationVictimLocked
// takes one group and evaluates only its members — an engine whose members are
// all in flight must not borrow another engine's idleness, and an idle engine
// must not spend its decision on a head another engine owns). The ADMISSION side
// cannot be per engine once the pool is global: "whichever engine frees a seat
// takes the global head" is a pool-wide decision, and N per-engine loops over the
// same shared map would just mean engine id order decides — which the seat-time
// rule (fit → least relative backlog) deliberately does not.
//
// The loop is structurally the shipped admission loop with the engine lifted out
// of it: take the D1 head, ask the seat-time rule which engine, seat it there if
// that engine can take it, else try to evict on that engine. It is HEAD-ONLY on
// purpose (the D1 head, not "the first head that fits"): that is the admission
// semantics every banked run was produced with, and the N=1 golden gate requires
// seat=pin to reproduce it byte for byte. The first-fit rule belongs to the
// DISPATCH gate (gateAllowsLocked), where the 02:13Z freeze required it.
//
// Bounded: every iteration either seats a head (the pool shrinks), evicts a
// member (the engine's member count shrinks) or returns.
func (s *Scheduler) rotateSeatPoolLocked(now time.Time) {
	if len(s.gWaiters) == 0 {
		return
	}
	var seats map[string]int
	if s.cfg.SessionOrder == "lsf" {
		seats = s.sessionSeatCountsLocked() // the LSF key's seat component, summed across engines
	}
	budget := len(s.gWaiters)
	for _, g := range s.activeGroupsLocked() {
		budget += len(g.members)
	}
	for ; budget > 0; budget-- {
		// the LSF key's seat component follows this loop's seats; entryHeadLocked
		// refreshes it itself when it is asked with nil.
		head := s.entryHeadLocked(nil, seats)
		if head == nil {
			return
		}
		cand := s.seatEngineLocked(head, now)
		if cand == nil || cand.g == nil {
			return
		}
		if cand.fits && s.coldAllows(cand.g, head) {
			// the chosen engine's own cost replaces the pool's best-case guess:
			// the cold charge, the pending blocks and the budget all belong to
			// THAT engine from here on.
			head.coldTokens = cand.costTokens
			if !s.seatLocked(cand.g, head, now) {
				// C4, PER ENGINE, always-on: the estimate passed but TrySeat
				// rejected — refresh from the authoritative index state and
				// RETURN (continuing with the stale estimate spins this loop
				// forever under s.mu: the 13:00 deadlock class).
				head.coldTokens = s.estimateCostLocked(cand.g, head.id, head.sizeBytes) * int64(s.blockSize)
				return
			}
			if seats != nil {
				seats[sessionUnitOf(head.session, head.id)]++
			}
			continue
		}
		// Nothing fits on the chosen engine: eviction is the only way a blocked
		// head ever runs, and it is demand-driven (a head is waiting right now).
		reason, victim := s.rotationVictimLocked(cand.g, now)
		if victim == nil {
			return // this engine can free nothing; another pass will re-evaluate
		}
		if victim.inflight > 0 {
			cand.g.index.MarkExitingLocked(victim.id)
			s.countExitLocked(cand.g, "exiting")
			return
		}
		s.exitLocked(cand.g, victim, reason, now)
		if seats != nil {
			seats = s.sessionSeatCountsLocked() // an exit frees a seat
		}
	}
}
