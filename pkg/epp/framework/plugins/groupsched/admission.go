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

// ADMISSION — one shared queue, one engine choice, ONE code path for N=1..N.
//
// The model, in full:
//
//  1. WAITING IS POOLED. There is exactly ONE waiter queue (s.waiters) for the
//     whole policy. A waiter is not seated, so it belongs to no engine; "one
//     queue" is what makes N engines one admission system instead of N queues
//     whose depths drift apart. Nothing is pre-homed: an unseated member carries
//     no placement opinion.
//
//  2. THE HEAD IS THE HEAD. Entry order is newcomer-first, then oldest arrival,
//     then id (f0p's order). The pool hands out ONE head per step, and the head
//     is never skipped: if no engine can take it, it keeps its place. (A head that
//     could jump the queue to a roomier engine would be served before older
//     demand — the queue would stop being a queue.)
//
//  3. THE ENGINE IS CHOSEN AT SEAT TIME, by the most FREE BUDGET among the
//     engines where the head FITS (fit = the head's own per-engine admission cost
//     plus the engine's accounted footprint is inside that engine's budget, and
//     the cold-entry valve allows it). Ties break on the engine id, so the decision
//     is total and every run is reproducible. Absent a fit, the head evicts on the
//     engine that has an eligible victim.
//
//  4. THE BINDING LASTS ONLY WHILE SEATED. seatLocked creates it, exitLocked drops
//     it, and nothing else touches it — so a member's next seat is chosen afresh,
//     by the state of the engines at that moment. (No TTL, no pre-seat homing, no
//     residue: there is nothing to keep warm between seats because the blocks die
//     during the unseated interval — measured, not assumed.)
//
// At N=1 every rule above collapses to "the head goes to the only engine that
// fits", which is exactly what the single-pod golden trace pins.

import (
	"time"
)

// debugSeat is a TEST-ONLY hook (nil in production) for tracing the admission loop.
var debugSeat func(s *Scheduler, now time.Time, head *waiter, cand *seatChoice)

// debugVictim is a TEST-ONLY hook for tracing the victim choice.
var debugVictim func(s *Scheduler, now time.Time, g *engineGroup, reason string, victim *member)

// coldTokensForEngine is one waiter's admission cost against ONE engine, in
// tokens: the per-engine estimate frozen at sync time (each engine is costed
// against its OWN index — the member's blocks are warm or cold PER ENGINE), with
// the pool's best-case scalar as the fallback for an engine that appeared after the
// sync.
func coldTokensForEngine(w *waiter, g *engineGroup, blockSize int) int64 {
	if t, ok := w.coldTokensByEngine[g.id]; ok {
		return t
	}
	return w.coldTokens
}

// setWaiterCostsLocked fills a waiting head's admission costs: the per-engine
// vector (each engine's OWN index estimate, frozen for the decision pass — reviewer
// R2) and the pool's best-case scalar (the minimum, used for the cold-entry charge
// and for the turnover numbers).
func (s *Scheduler) setWaiterCostsLocked(w *waiter) {
	groups := s.activeGroupsLocked()
	if len(groups) == 0 {
		return
	}
	vec := make(map[string]int64, len(groups))
	best := int64(-1)
	for _, g := range groups {
		e := s.mod.sizer.Tokens(s.estimateCostCachedLocked(g, w.id, w.sizeBytes))
		vec[g.id] = e
		if best < 0 || e < best {
			best = e
		}
	}
	if best < 0 {
		best = 0
	}
	w.coldTokensByEngine = vec
	w.coldTokens = best
}

// queuedPerSeatLocked is the engine's OWN queued demand per seat it holds: seated
// members that have a head waiting right now, divided by its seats. It orders the
// DISPATCH pass across engines (the engine whose own residents are waiting most per
// seat releases first); it is never a placement key — placement is the fit +
// most-free-budget rule alone.
func (s *Scheduler) queuedPerSeatLocked(g *engineGroup) float64 {
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

// freeBudgetLocked is what the engine can still admit: its budget minus its
// accounted footprint (NE blocks + the seat-time estimates not yet converted to
// exact keys). Negative when the engine is over budget (enforcement is about to
// catch up).
func (s *Scheduler) freeBudgetLocked(g *engineGroup) int64 {
	return int64(g.budgetBlocks) - (int64(g.index.UniqueBlocksNELocked()) + g.pendingTotal)
}

// coldAllows staggers ALL cold entries IN ONE ENGINE (B6: re-entries too — WEKA
// re-entries are fully cold), in TOKENS (R1). R11's rule: one cold entry is ALWAYS
// allowed — a member bigger than the cap must be seatable when nothing cold is in
// flight, or newcomers-first + hold-the-line deadlocks the queue on it forever.
//
// The cap is PER ENGINE, so the system-wide cold-entry concurrency is at most N, and
// the "always one" clause keeps each engine's own queue live.
func (s *Scheduler) coldAllows(g *engineGroup, w *waiter) bool {
	return g.coldInFlight == 0 || g.coldInFlight+w.coldTokens <= s.cfg.ColdEntryCapTokens
}

// seatChoice is the PLACEMENT module's answer bound back to the engine.
type seatChoice struct {
	g *engineGroup
	// costTokens / costBlocks are the estimate the fit test used, so the chosen
	// engine's own numbers become the member's cold charge and pending blocks
	// (instead of the pool's best-case scalar).
	costTokens int64
	costBlocks int64
	// freeBudget is what the engine could still admit when it was chosen.
	freeBudget int64
	// fits: the head's cost was inside the free budget.
	fits bool
}

// seatEngineLocked is THE seat-time engine choice: the scheduler gathers the FACTS
// (per-engine cost, free budget, fit, whether a victim exists) and the PLACEMENT
// module decides. Engines without measured capacity are not candidates (an unknown
// budget must not take seats); an engine whose safety net is firing is excluded —
// a gate on that engine's own admissions, never a ranking of the others.
func (s *Scheduler) seatEngineLocked(head *waiter, now time.Time) *seatChoice {
	groups := s.activeGroupsLocked()
	cands := make([]EngineCandidate, 0, len(groups))
	owners := make([]*engineGroup, 0, len(groups))
	for _, g := range groups {
		if g.numBlocks <= 0 {
			// NO measured capacity: the engine cannot be costed, and seating into an
			// unknown budget is how an engine takes unbounded seats during the metrics
			// race. Its heads wait; the plugin serves FCFS pass-through until capacity
			// lands (and alarms loudly if it never does).
			continue
		}
		if s.safetyFiredLocked(g) {
			g.counters.safetySkips++
			continue
		}
		tokens := coldTokensForEngine(head, g, s.blockSize)
		blocks := tokens / int64(s.blockSize)
		free := s.freeBudgetLocked(g)
		gg := g
		cands = append(cands, EngineCandidate{
			Engine: gg.id, CostTokens: tokens, CostBlocks: blocks,
			FreeBudget: free, Fits: blocks <= free,
			HasVictim: func() bool { return s.hasRotationVictimLocked(gg, now) },
		})
		owners = append(owners, gg)
	}
	idx := s.mod.place.Choose(cands)
	if idx < 0 || idx >= len(cands) {
		return nil
	}
	return &seatChoice{g: owners[idx], costTokens: cands[idx].CostTokens,
		costBlocks: cands[idx].CostBlocks, freeBudget: cands[idx].FreeBudget,
		fits: cands[idx].Fits}
}

// hasRotationVictimLocked reports whether an engine can evict somebody right now
// (one definition of the victim criteria, in rotationVictimLocked).
func (s *Scheduler) hasRotationVictimLocked(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictimLocked(g, now)
	return v != nil
}

// entryHeadLocked is D1: the QUEUE module picks the head of the ONE shared queue.
func (s *Scheduler) entryHeadLocked() *waiter {
	return s.mod.entry.Head(s.waiters)
}

// rotateLocked is the admission loop: hand the shared queue's head to an engine
// until nothing more can be seated.
//
// Bounded: every iteration either seats a head (the queue shrinks), evicts a member
// (an engine's member count shrinks) or returns.
func (s *Scheduler) rotateLocked(now time.Time) {
	if len(s.waiters) == 0 {
		return
	}
	budget := len(s.waiters)
	for _, g := range s.activeGroupsLocked() {
		budget += len(g.members)
	}
	for ; budget > 0; budget-- {
		head := s.entryHeadLocked()
		if head == nil {
			return
		}
		cand := s.seatEngineLocked(head, now)
		if debugSeat != nil {
			debugSeat(s, now, head, cand)
		}
		if cand == nil || cand.g == nil {
			return
		}
		if cand.fits && s.coldAllows(cand.g, head) {
			// the chosen engine's own cost replaces the pool's best-case scalar: the
			// cold charge, the pending blocks and the budget all belong to THAT
			// engine from here on.
			head.coldTokens = cand.costTokens
			if !s.seatLocked(cand.g, head, now) {
				// C4, PER ENGINE, always-on: the estimate passed but TrySeat rejected —
				// refresh from the authoritative index state and RETURN (continuing
				// with the stale estimate spins this loop forever under s.mu: the 13:00
				// deadlock class).
				head.coldTokens = s.mod.sizer.Tokens(s.estimateCostLocked(cand.g, head.id, head.sizeBytes))
				return
			}
			continue
		}
		if cand.fits && !s.coldAllows(cand.g, head) && s.onValveBlockedEviction != nil {
			// the budget would have taken this head, but the cold-entry valve refused
			// it, so the only way forward is an EVICTION (the sizing signal).
			s.onValveBlockedEviction(cand.g.id)
		}
		// Nothing fits on the chosen engine: eviction is the only way a blocked head
		// ever runs, and it is demand-driven (a head is waiting right now).
		reason, victim := s.rotationVictimLocked(cand.g, now)
		if debugVictim != nil {
			debugVictim(s, now, cand.g, reason, victim)
		}
		if victim == nil {
			return // this engine can free nothing; another pass will re-evaluate
		}
		if victim.inflight > 0 {
			cand.g.index.MarkExitingLocked(victim.id)
			s.countExitLocked(cand.g, "exiting")
			return
		}
		s.exitLocked(cand.g, victim, reason, now)
	}
}
