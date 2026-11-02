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
//     whose depths drift apart. Nothing is pre-homed: an unseated branch carries
//     no placement opinion.
//
//  2. THE HEAD IS THE HEAD. Entry order is newcomer-first, then oldest arrival,
//     then id. The pool hands out ONE head per step, and the head
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
//  4. THE BINDING LASTS ONLY WHILE SEATED. seatLocked creates it, exitLocked drops it,
//     and nothing else touches it — so a branch's next seat is chosen afresh, by the
//     state of the engines at that moment. There is no TTL and no pre-seat homing:
//     there is nothing to keep warm between seats, because the blocks die during the
//     unseated interval (measured, not assumed — the unseated interval IS a
//     multi-minute admission wait).
//
// At N=1 every rule above collapses to "the head goes to the only engine that
// fits", which is exactly what the single-pod golden trace pins.

import (
	"time"
)

// queuedPerSeatLocked is the engine's OWN queued demand per seat it holds: seated
// branches that have a head waiting right now, divided by its seats. It orders the
// DISPATCH pass across engines (the engine whose own residents are waiting most per
// seat releases first); it is never a placement key — placement is the fit +
// most-free-budget rule alone.
func (s *Scheduler) queuedPerSeatLocked(g *engineGroup) float64 {
	seats := len(g.branches)
	if seats <= 0 {
		seats = 1
	}
	queued := 0
	for _, m := range g.branches {
		if !m.waitingSince.IsZero() {
			queued++
		}
	}
	return float64(queued) / float64(seats)
}

// freeBudgetLocked is what the engine can still admit: its budget minus its USED side
// (every seated branch's blocks, exiting ones included — they still hold them).
// Negative when the engine is over budget, which enforcement is about to catch up on.
func (s *Scheduler) freeBudgetLocked(g *engineGroup) int64 {
	return int64(g.budgetBlocks) - int64(g.index.UniqueBlocksLocked())
}

// coldAllowsTokens asks whether this head's cold prefill fits the engine's stagger, for the
// exact cost it brings.
//
// The cap is an INVARIANT, not a threshold. A seat charges min(cost, cap), so the test is
// exactly "after this entry, is the engine's outstanding cold charge inside its headroom?" —
// and two properties fall out of that single rule: an EMPTY valve always admits one entry
// whatever its size (its capped charge IS the headroom), and the outstanding charge can
// never exceed the cap.
func (s *Scheduler) coldAllowsTokens(g *engineGroup, tokens int64) bool {
	cap := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
	if cap <= 0 {
		return true // no measured capacity: the admission pass already refuses these engines
	}
	return g.coldInFlight+min(tokens, cap) <= cap
}

// seatChoice is the PLACEMENT module's answer bound back to the engine.
type seatChoice struct {
	g *engineGroup
	// costTokens is the head's cold tokens ON THIS ENGINE: the fit test's input and the
	// cold-entry charge the seat takes (capped by the engine's own valve).
	costTokens int64
	// fits: the head's cost was inside the free budget.
	fits bool
	// valveHeld: the budget would take the head and only the cold-entry valve refuses it,
	// so the ONLY victim that may be taken is the branch holding the charge.
	valveHeld bool
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
		// the head's exact cost ON THIS ENGINE: what this engine would have to compute
		// (its own resident prefix discounted) and the cold tokens behind it
		tokens := g.index.ColdTokensLocked(head.chains)
		blocks := int64(g.index.CostLocked(head.chains))
		free := s.freeBudgetLocked(g)
		budgetFits := blocks <= free
		gg := g
		cands = append(cands, EngineCandidate{
			Engine: gg.id, CostTokens: tokens, CostBlocks: blocks,
			FreeBudget: free, BudgetFits: budgetFits,
			// FITS = budget AND cold valve: a head the valve holds must wait for the
			// valve, never evict for it (the defect this replaces evicted every
			// eligible branch on the engine and left the seats cold-empty).
			Fits:      budgetFits && s.coldAllowsTokens(g, tokens),
			HasVictim: func() bool { return s.hasRotationVictimLocked(gg, now) },
			HasValveHolder: func() bool {
				_, holder := s.valveHolderVictimLocked(gg, now)
				return holder != nil
			},
		})
		owners = append(owners, gg)
	}
	idx := s.mod.place.Choose(cands)
	if idx < 0 || idx >= len(cands) {
		return nil
	}
	return &seatChoice{g: owners[idx], costTokens: cands[idx].CostTokens,
		fits: cands[idx].Fits,
		// not fitting while the BUDGET would take it = the valve is the blocker
		valveHeld: !cands[idx].Fits && cands[idx].BudgetFits}
}

// coldValveHeldOn reports WHICH engine's cold-entry valve is the only thing keeping
// this head out — the engine whose BUDGET would take it while the valve refuses it — and
// whether any engine is in that state at all. The engine id is the attribution: the
// series said `engine="single"` on a two-engine stack until this returned a name
// (2026-11-01). When several engines hold the head the answer is the one the placement
// module would have chosen, most free budget first then engine id, so the metric names
// the engine the head is actually waiting for instead of the first one in a loop.
func (s *Scheduler) coldValveHeldOn(head *waiter) (string, bool) {
	best := EngineCandidate{Engine: ""}
	for _, g := range s.activeGroupsLocked() {
		if g.numBlocks <= 0 {
			continue // no measured capacity: the engine is not taking seats at all
		}
		tokens := g.index.ColdTokensLocked(head.chains)
		if int64(g.index.CostLocked(head.chains)) > s.freeBudgetLocked(g) {
			continue // the BUDGET is the blocker on this engine: the valve is not the reason
		}
		if s.coldAllowsTokens(g, tokens) {
			continue // the valve allows it: whatever held the head, it was not this
		}
		c := EngineCandidate{Engine: g.id, FreeBudget: s.freeBudgetLocked(g)}
		if best.Engine == "" || betterEngine(c, best) {
			best = c
		}
	}
	return best.Engine, best.Engine != ""
}

// hasRotationVictimLocked reports whether an engine can evict somebody right now
// (one definition of the victim criteria, in rotationVictimLocked).
func (s *Scheduler) hasRotationVictimLocked(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictimLocked(g, now)
	return v != nil
}

// valveHolderVictimLocked answers the ONE case in which a valve-held head may evict.
//
// I2 says a valve-held head waits and never evicts, because evicting frees budget and never
// frees the valve. That premise is false for exactly one member: the branch HOLDING the
// cold charge. Evicting it releases the charge, which opens the valve — and it is only ever
// done on demand, by the head that is waiting, for ONE holder, and only one the TENURE rule
// already releases (nothing in flight, and Q complete or silent past S). It evicts exactly
// the holder, never "every eligible member" — the eviction storm that leaves seats cold-empty
// (the eviction-storm class) cannot come back.
//
// The silence bound is what protects a LIVE window: a release a few milliseconds old
// belongs to a branch whose S has not elapsed, so its charge stays.
func (s *Scheduler) valveHolderVictimLocked(g *engineGroup, now time.Time) (string, *branch) {
	var best *branch
	var bestReason string
	bestWaiting := false
	for _, m := range g.branches {
		if m.coldEntry <= 0 || g.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(m, now)
		if !ok {
			continue
		}
		waiting := !m.waitingSince.IsZero()
		better := best == nil ||
			(waiting && !bestWaiting) ||
			(waiting == bestWaiting && m.seatTime.Before(best.seatTime)) ||
			(waiting == bestWaiting && m.seatTime.Equal(best.seatTime) && m.id < best.id)
		if better {
			best, bestReason, bestWaiting = m, reason, waiting
		}
	}
	return bestReason, best
}

// entryHeadLocked is D1: the QUEUE module picks the head of the ONE shared queue. The
// module gets the pass's clock and nothing else — never the scheduler — and it answers
// for the waiter set it was handed.
func (s *Scheduler) entryHeadLocked(now time.Time) *waiter {
	return s.mod.entry.Head(s.waiters, EntryCtx{Now: now})
}

// rotateLocked is the admission loop: hand the shared queue's head to an engine
// until nothing more can be seated.
//
// Bounded: every iteration either seats a head (the queue shrinks), evicts a branch
// (an engine's branch count shrinks) or returns.
func (s *Scheduler) rotateLocked(now time.Time) {
	if len(s.waiters) == 0 {
		return
	}
	budget := len(s.waiters)
	for _, g := range s.activeGroupsLocked() {
		budget += len(g.branches)
	}
	for ; budget > 0; budget-- {
		head := s.entryHeadLocked(now)
		if head == nil {
			return
		}
		cand := s.seatEngineLocked(head, now)
		if cand == nil || cand.g == nil {
			// nothing to do this pass. If the reason is the cold-entry valve holding a
			// head that WOULD fit the budget, book it AGAINST THAT ENGINE: the head waits
			// for the prefill slot, and that is a measurable cost of the stagger (never a
			// reason to evict — see EngineCandidate.Fits). The attribution is the engine
			// id, not "": the series must name the engine the head is actually waiting for
			// (it read `engine="single"` on a two-engine stack while this passed an empty
			// id).
			if engine, held := s.coldValveHeldOn(head); held {
				if s.onColdValveHeld != nil {
					s.onColdValveHeld(engine)
				}
			}
			return
		}
		if cand.fits {
			// ONE number, one moment: the fit test the chooser just ran and the seat's
			// own cost are the same computation inside the same critical section, so
			// TrySeat cannot reject a head the chooser admitted.
			s.seatLocked(cand.g, head, cand.costTokens, now)
			continue
		}
		// Nothing fits BY BUDGET on the chosen engine: eviction is the only way a
		// budget-blocked head ever runs, and it is demand-driven (a head is waiting right
		// now). A VALVE-held head reaches here only when the engine's charge holder is
		// itself releasable — then that holder is the victim, and nothing else is touched.
		reason, victim := s.rotationVictimLocked(cand.g, now)
		if cand.valveHeld {
			reason, victim = s.valveHolderVictimLocked(cand.g, now)
		}
		if victim == nil {
			return // this engine can free nothing; another pass will re-evaluate
		}
		// NOTE: a rotation victim is never in flight — the tenure rule refuses in-flight
		// branches — so there is no mid-flight exit to handle here (the mid-flight class
		// lives in the PRESSURE module, which does reach running branches).
		s.exitLocked(cand.g, victim, reason, now)
	}
}
