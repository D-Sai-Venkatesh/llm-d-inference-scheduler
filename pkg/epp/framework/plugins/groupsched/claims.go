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

import (
	"time"
)

// ONE BLOCKED HEAD PER ENGINE (claimOrder). Today's admission works on the queue head alone: a head that fits
// on no engine, with no eligible victim, stops the loop, and every waiter behind it waits — even one that fits
// on another engine now. And the eviction engine is re-picked at every step, so one big head can scatter
// victims over engines and be seated nowhere. Here each pass walks the shared queue in its order (newcomers
// first, then oldest), and for each waiter W:
//
//  1. W fits on an OPEN engine without eviction → seated there (today's placement). GREEDY: a claim W held
//     elsewhere is released; the budget its victims freed there goes to the next waiter.
//  2. Otherwise W keeps its claim if that engine is still open (STICKY — no scattering), or claims the open
//     engine chosen by claimOrder ("compute": fewest tokens to really prefill there, then smallest shortfall;
//     "size": smallest shortfall), and victims for W come ONLY from that engine (the fair victim), one at a
//     time, until W fits.
//  3. If nobody there may leave, the engine is CLAIMED for this pass: closed to every waiter behind W (so small
//     waiters cannot starve it), while they go on to the other engines.
//  4. Every engine claimed → the rest wait.
//
// Claims follow the queue order on every pass: an engine is closed only to the waiters BEHIND its claimer, so
// newcomers-first stays as it is, and with one engine every decision is today's.

// seatWaitersClaims is the admission loop under claimOrder. The next waiter is picked exactly as headWaiter
// picks the head (the shared queue's order), skipping only the waiters already blocked in this pass — so a
// victim re-queued during the pass competes at once, as today, and with one engine every step is today's.
func (s *Scheduler) seatWaitersClaims(now time.Time) {
	groups := s.activeGroups()
	if len(s.waiters) == 0 || len(groups) == 0 {
		return
	}
	claimed := make(map[string]bool, len(groups))
	open := func(e string) bool { return !claimed[e] }
	blocked := map[string]bool{}
	steps := len(s.waiters)
	for _, g := range groups {
		steps += len(g.threads)
	}
	claimAhead := false // a waiter ahead of this one is blocked on a claim in this pass
	for steps > 0 {
		if len(claimed) >= len(groups) {
			return // every engine holds a blocked head: the rest wait
		}
		var w *waiter
		for _, x := range s.waiters {
			if !blocked[x.id] && (w == nil || s.entryLess(x, w)) {
				w = x
			}
		}
		if w == nil {
			return
		}
		for steps > 0 {
			steps--
			cand := s.seatEngineFor(w, now, open)
			if cand != nil && cand.g != nil && cand.fits {
				prev, victims, since := w.claim, w.claimVictims, w.claimSince
				if !s.seat(cand.g, w, cand.costTokens, now) {
					return // the chooser and the seat disagreed: stop, as today
				}
				s.notePlacement(cand, w, now)
				s.afterSeat(cand, w, now)
				s.noteClaimSeat(cand.g, prev, victims, since, claimAhead, now)
				break
			}
			g := s.claimEngine(w, groups, open)
			if g == nil {
				blocked[w.id] = true // no open engine with known capacity
				break
			}
			if w.claim != g.id {
				if w.claim != "" && w.claimVictims > 0 {
					// its old claim was taken by a waiter ahead: the victims removed there seat someone else
					s.counters.claimVictimsElsewhere += int64(w.claimVictims)
					s.metrics.ObserveClaimVictimsElsewhere(engineLabel(w.claim), w.claimVictims)
				}
				if w.claim == "" {
					w.claimSince = now
				}
				w.claim, w.claimVictims = g.id, 0
				s.counters.claims++
				s.metrics.ObserveClaim(engineLabel(g.id))
			}
			if s.fairStep(&seatChoice{g: g}, w, now) {
				w.claimVictims++
				continue // a victim left there: re-check W
			}
			claimed[g.id] = true // nobody there may leave now: W waits on it, the engine is closed behind W
			blocked[w.id] = true
			claimAhead = true
			break
		}
	}
}

// claimEngine is the engine a waiter that fits on no open engine waits on: its current claim while that engine
// is open and known (sticky), else the open engine claimOrder ranks first.
func (s *Scheduler) claimEngine(w *waiter, groups []*engineGroup, open func(string) bool) *engineGroup {
	if w.claim != "" && open(w.claim) {
		if g := s.groups[w.claim]; g != nil && g.numBlocks > 0 {
			return g
		}
	}
	resident := s.residentTokens(w.chains, groups)
	prompt := PromptTokens(w.chains)
	var best *engineGroup
	var bestCold, bestShort int64
	for gi, g := range groups {
		if g.numBlocks <= 0 || !open(g.id) {
			continue
		}
		cold := physCold(prompt, resident, gi)
		short := int64(g.index.Cost(w.chains)) - s.freeBudget(g)
		if best == nil || claimBetter(s.cfg.ClaimOrder, cold, short, g.id, bestCold, bestShort, best.id) {
			best, bestCold, bestShort = g, cold, short
		}
	}
	return best
}

// claimBetter is claimOrder's total order (lower wins): "compute" — fewer tokens to really prefill, then a
// smaller shortfall; "size" — a smaller shortfall; then the engine id.
func claimBetter(order string, cold, short int64, id string, bestCold, bestShort int64, bestID string) bool {
	if order == "compute" && cold != bestCold {
		return cold < bestCold
	}
	if short != bestShort {
		return short < bestShort
	}
	return id < bestID
}

// noteClaimSeat books a seat under claimOrder: the claim's duration, victims removed for the waiter on
// another engine than the one it was seated on, and a seat made while an older waiter was blocked on a claim.
func (s *Scheduler) noteClaimSeat(g *engineGroup, claim string, victims int, since time.Time, claimAhead bool, now time.Time) {
	if claim != "" {
		s.metrics.ObserveClaimSeconds(engineLabel(claim), now.Sub(since).Seconds())
		if claim != g.id && victims > 0 {
			s.counters.claimVictimsElsewhere += int64(victims)
			s.metrics.ObserveClaimVictimsElsewhere(engineLabel(claim), victims)
		}
	}
	if claimAhead {
		s.counters.seatedPastClaim++
		s.metrics.ObserveSeatedPastClaim(engineLabel(g.id))
	}
}
