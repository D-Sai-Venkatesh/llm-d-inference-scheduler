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

import (
	"time"
)

// queuedPerSeatLocked is the engine's OWN queued demand per seat it holds: seated branches
// that have a head waiting right now, divided by its seats.
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

// freeBudgetLocked is what the engine can still admit: its budget minus its USED side (every
// seated branch's blocks, exiting ones included — they still hold them).
func (s *Scheduler) freeBudgetLocked(g *engineGroup) int64 {
	return int64(g.budgetBlocks) - int64(g.index.UniqueBlocksLocked())
}

// coldAllowsTokens asks whether this head's cold prefill fits the engine's stagger, for the
// exact cost it brings.
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
	// costTokens is the head's cold tokens ON THIS ENGINE: the fit test's input and the cold-
	// entry charge the seat takes (capped by the engine's own valve).
	costTokens int64
	// fits: the head's cost was inside the free budget AND the cold-entry valve allowed it.
	fits bool
}

// seatEngineLocked is THE seat-time engine choice: the scheduler gathers the FACTS (per-engine
// cost, free budget, fit, whether a victim exists) and the PLACEMENT module decides.
func (s *Scheduler) seatEngineLocked(head *waiter, now time.Time) *seatChoice {
	groups := s.activeGroupsLocked()
	cands := make([]EngineCandidate, 0, len(groups))
	owners := make([]*engineGroup, 0, len(groups))
	for _, g := range groups {
		if g.numBlocks <= 0 {
			// NO measured capacity: the engine cannot be costed, and seating into an unknown budget is
			// how an engine takes unbounded seats during the metrics race.
			continue
		}
		// the head's exact cost ON THIS ENGINE: what this engine would have to compute (its own
		// resident prefix discounted) and the cold tokens behind it
		tokens := g.index.ColdTokensLocked(head.chains)
		blocks := int64(g.index.CostLocked(head.chains))
		free := s.freeBudgetLocked(g)
		budget := int64(g.budgetBlocks)
		budgetFits := blocks <= free
		// RULE O — the EMPTY ENGINE ADMITS ONE.
		ruleO := len(g.branches) == 0 && blocks > budget
		gg := g
		cands = append(cands, EngineCandidate{
			Engine: gg.id, CostTokens: tokens, CostBlocks: blocks,
			FreeBudget: free, Budget: budget, BudgetFits: budgetFits,
			// FITS = (budget OR Rule O's empty engine) AND the cold valve: a head the valve holds must
			// wait for the valve, never evict for it (the defect this replaces evicted every eligible
			// branch on the engine and left the seats cold-empty).
			Fits:      (budgetFits || ruleO) && s.coldAllowsTokens(g, tokens),
			HasVictim: func() bool { return s.hasRotationVictimLocked(gg, now) },
		})
		owners = append(owners, gg)
	}
	idx := s.mod.place.Choose(cands)
	if idx < 0 || idx >= len(cands) {
		return nil
	}
	return &seatChoice{g: owners[idx], costTokens: cands[idx].CostTokens, fits: cands[idx].Fits}
}

// coldValveHeldOn reports WHICH engine's cold-entry valve is the only thing keeping this head
// out — the engine whose BUDGET would take it while the valve refuses it — and whether any
// engine is in that state at all.
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

// hasRotationVictimLocked reports whether an engine can evict somebody right now (one
// definition of the victim criteria, in rotationVictimLocked).
func (s *Scheduler) hasRotationVictimLocked(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictimLocked(g, now)
	return v != nil
}

// entryHeadLocked is the QUEUE module's answer: it picks the head of the ONE shared queue.
func (s *Scheduler) entryHeadLocked(now time.Time) *waiter {
	return s.mod.entry.Head(s.waiters, EntryCtx{Now: now})
}

// rotateLocked is the admission loop: hand the shared queue's head to an engine until nothing
// more can be seated.
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
			// nothing to do this pass.
			if engine, held := s.coldValveHeldOn(head); held {
				if s.onColdValveHeld != nil {
					s.onColdValveHeld(engine)
				}
			}
			return
		}
		if cand.fits {
			// ONE number, one moment: the fit test the chooser just ran and the seat's own cost are the
			// same computation inside the same critical section, so TrySeat can only reject a head the
			// chooser admitted if the two disagree — and then the loop must STOP, not re-pick the same
			// head until the budget runs out under the scheduler's mutex.
			if !s.seatLocked(cand.g, head, cand.costTokens, now) {
				return
			}
			continue
		}
		// Nothing fits BY BUDGET on the chosen engine: eviction is the only way a budget-blocked
		// head ever runs, and it is demand-driven (a head is waiting right now).
		reason, victim := s.rotationVictimLocked(cand.g, now)
		if victim == nil {
			return // this engine can free nothing; another pass will re-evaluate
		}
		// NOTE: a rotation victim is never in flight — the tenure rule refuses in-flight branches —
		// so there is no mid-flight exit to handle here (the mid-flight class lives in the PRESSURE
		// module, which does reach running branches).
		s.exitLocked(cand.g, victim, reason, now)
	}
}
