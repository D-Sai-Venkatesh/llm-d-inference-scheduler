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
	"fmt"
	"time"
)

// queuedPerSeat is the engine's OWN queued demand per seat it holds: seated threads

// freeBudget is what the engine can still admit: its budget minus its USED side (every
// seated thread's blocks, exiting ones included — they still hold them).
func (s *Scheduler) freeBudget(g *engineGroup) int64 {
	return int64(g.budgetBlocks) - int64(g.index.UniqueBlocks())
}

// coldAllowsTokens asks whether this head's cold prefill fits the engine's stagger, for the
// exact cost it brings.
func (s *Scheduler) coldAllowsTokens(g *engineGroup, tokens int64) bool {
	if s.cfg.RoomGate {
		return true // the room gate replaces the valve: the tier's room paces cold entries
	}
	cap := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
	if cap <= 0 {
		return true // no measured capacity: the admission pass already refuses these engines
	}
	return g.coldInFlight+min(tokens, cap) <= cap
}

// seatChoice is placement's answer (chooseEngine) bound back to the engine it named.
type seatChoice struct {
	g *engineGroup
	// costTokens is the head's cold tokens ON THIS ENGINE: the fit test's input and the cold-
	// entry charge the seat takes (capped by the engine's own valve).
	costTokens int64
	// fits: the head's cost was inside the free budget AND the cold-entry valve allowed it.
	fits bool
	// resident is the head's prompt tokens the CPU-residency index says THIS engine's tier still
	// holds, and bestResident the most any candidate engine holds (both 0 with residency off).
	resident, bestResident int64
	// roomHeld (roomGate): the budget takes the head on this engine but the tier's room does not;
	// roomForced: the room did not fit but the head's hold reached roomHoldMaxSeconds, so it is admitted.
	roomHeld, roomForced bool
	// writeBlocks is what the head will write into this engine's tier, in blocks, reserve included.
	writeBlocks int
	// heldOn names the engine whose room held the head when no engine was chosen (g nil).
	heldOn string
}

// roomNeed is the blocks a head's write takes from the tier (its prompt beyond the resident prefix,
// plus the reserve), rounded up to whole blocks.
func (s *Scheduler) roomNeed(writeTokens int64) int {
	t := max(writeTokens, 0) + s.cfg.RoomReserveTokens
	bs := int64(s.blockSize)
	return int((t + bs - 1) / bs)
}

// tierRoom is the engine's tier room for the head: free capacity plus the run of non-seated blocks at
// the eviction end, skipping the head's own blocks, up to the first STAYING (seated, not exiting) block
// — blocks of also are treated as not seated (the rotation guard's "if this member left" read). The
// walk stops once need is reached.
func (s *Scheduler) tierRoom(g *engineGroup, w *waiter, need int, also map[Hash]struct{}) int {
	if g.tier == nil {
		return 0
	}
	own := w.ownKeys()
	skip := func(k Hash) bool { _, ok := own[k]; return ok }
	stop := func(k Hash) bool {
		if also != nil {
			if _, ok := also[k]; ok {
				return false
			}
		}
		return g.index.Staying(k)
	}
	return g.tier.TailRoom(need, skip, stop)
}

// roomGrowsWithout is the room-rotation guard: would rotating victim out give the head more room?
// Only if victim's blocks lie below the first staying block of somebody else (a busy member at the
// eviction end — a long decode — blocks every rotation behind it, and then none is made).
func (s *Scheduler) roomGrowsWithout(g *engineGroup, w *waiter, victim *thread, need int) bool {
	keys := map[Hash]struct{}{}
	for _, c := range g.index.Chains(victim.id) {
		for _, k := range c.Keys {
			keys[k] = struct{}{}
		}
	}
	return s.tierRoom(g, w, need, keys) > s.tierRoom(g, w, need, nil)
}

// seatEngine is THE seat-time engine choice: the scheduler gathers the FACTS (per-engine
// cost, free budget, fit, whether a victim exists) and chooseEngine decides.
func (s *Scheduler) seatEngine(head *waiter, now time.Time) *seatChoice {
	groups := s.activeGroups()
	cands := make([]EngineCandidate, 0, len(groups))
	owners := make([]*engineGroup, 0, len(groups))
	resident := s.residentTokens(head.chains, groups)
	prompt := PromptTokens(head.chains)
	for gi, g := range groups {
		if g.numBlocks <= 0 {
			// NO measured capacity: the engine cannot be costed, and seating into an unknown budget is
			// how an engine takes unbounded seats during the metrics race.
			continue
		}
		// the head's exact cost ON THIS ENGINE: what this engine would have to compute (its own
		// resident prefix discounted) and the cold tokens behind it
		tokens := g.index.ColdTokens(head.chains)
		blocks := int64(g.index.Cost(head.chains))
		free := s.freeBudget(g)
		budget := int64(g.budgetBlocks)
		budgetFits := blocks <= free
		// RULE O — the EMPTY ENGINE ADMITS ONE.
		ruleO := len(g.threads) == 0 && blocks > budget
		gg := g
		// THE ROOM GATE (roomGate): the tier must absorb what the head will write without evicting a
		// seated block. Measured only where the budget already takes the head (an empty engine admits
		// as Rule O always has); the hold's last resort admits it anyway.
		roomFits, roomHeld, roomForced, need := true, false, false, 0
		if s.cfg.RoomGate && gg.tier != nil && (budgetFits || ruleO) && len(gg.threads) > 0 {
			need = s.roomNeed(physCold(prompt, resident, gi))
			if s.tierRoom(gg, head, need, nil) < need {
				if !head.roomHeldSince.IsZero() && now.Sub(head.roomHeldSince).Seconds() >= s.cfg.RoomHoldMaxSeconds {
					roomForced = true
				} else {
					roomFits, roomHeld = false, true
				}
			}
		}
		cands = append(cands, EngineCandidate{
			Engine: gg.id, CostTokens: tokens, CostBlocks: blocks,
			FreeBudget: free, Budget: budget, BudgetFits: budgetFits,
			// FITS = (budget OR Rule O's empty engine) AND the cold valve: a head the valve holds must
			// WAIT for the valve, never evict for it — evicting frees budget, and the valve is not
			// budget. With roomGate the valve is off and the ROOM is the second condition instead.
			Fits:      (budgetFits || ruleO) && s.coldAllowsTokens(g, tokens) && roomFits,
			RoomHeld:  roomHeld,
			RoomForce: roomForced,
			Need:      need,
			HasVictim: func() bool { return s.hasRotationVictim(gg, now) },
			// what this engine would REALLY prefill: the prompt beyond what its CPU tier still holds
			// (0 for every engine with residency off, so the orders below are today's exactly)
			PhysColdTokens: physCold(prompt, resident, gi),
			Resident:       residentAt(resident, gi),
		})
		owners = append(owners, gg)
	}
	idx := chooseEngine(cands)
	if idx < 0 || idx >= len(cands) {
		// nothing to seat and no eviction to make: say whether the ROOM is (part of) why, so the pass
		// can book the hold (roomGate)
		for _, c := range cands {
			if c.RoomHeld {
				return &seatChoice{roomHeld: true, heldOn: c.Engine}
			}
		}
		return nil
	}
	var best int64
	for _, c := range cands {
		best = max(best, c.Resident)
	}
	return &seatChoice{g: owners[idx], costTokens: cands[idx].CostTokens, fits: cands[idx].Fits,
		resident: cands[idx].Resident, bestResident: best,
		roomHeld: cands[idx].RoomHeld, roomForced: cands[idx].RoomForce, writeBlocks: cands[idx].Need}
}

// residentTokens is the CPU-residency read for one head, per group (positional): the prompt tokens
// of each chain the engine's tier still holds as a leading run, summed over the head's prompts. nil
// when residency is off. ONE read per chain covers every engine.
func (s *Scheduler) residentTokens(chains []Chain, groups []*engineGroup) []int64 {
	if s.residency == nil || len(groups) == 0 {
		return nil
	}
	ids := make([]string, len(groups))
	for i, g := range groups {
		ids[i] = g.id
	}
	out := make([]int64, len(groups))
	for _, c := range chains {
		runs := s.residency(c.Keys, ids)
		for i := range out {
			if i < len(runs) {
				out[i] += min(int64(runs[i])*int64(s.blockSize), int64(c.Tokens))
			}
		}
	}
	return out
}

func residentAt(resident []int64, i int) int64 {
	if i < len(resident) {
		return resident[i]
	}
	return 0
}

// physCold is the prompt beyond what engine i's tier holds; 0 with residency off (no ordering).
func physCold(prompt int64, resident []int64, i int) int64 {
	if resident == nil {
		return 0
	}
	return max(prompt-residentAt(resident, i), 0)
}

// coldValveHeldOn reports WHICH engine's cold-entry valve is the only thing keeping this head
// out — the engine whose BUDGET would take it while the valve refuses it — and whether any
// engine is in that state at all.
func (s *Scheduler) coldValveHeldOn(head *waiter) (string, bool) {
	best := EngineCandidate{Engine: ""}
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 {
			continue // no measured capacity: the engine is not taking seats at all
		}
		tokens := g.index.ColdTokens(head.chains)
		if int64(g.index.Cost(head.chains)) > s.freeBudget(g) {
			continue // the BUDGET is the blocker on this engine: the valve is not the reason
		}
		if s.coldAllowsTokens(g, tokens) {
			continue // the valve allows it: whatever held the head, it was not this
		}
		c := EngineCandidate{Engine: g.id, FreeBudget: s.freeBudget(g)}
		if best.Engine == "" || betterEngine(c, best) {
			best = c
		}
	}
	return best.Engine, best.Engine != ""
}

// hasRotationVictim reports whether an engine can evict somebody right now (one
// definition of the victim criteria, in rotationVictim).
func (s *Scheduler) hasRotationVictim(g *engineGroup, now time.Time) bool {
	_, v := s.rotationVictim(g, now)
	return v != nil
}

// headWaiter picks the head of the ONE shared queue: newcomers before returning
// threads, then the oldest arrival, then the id.
func (s *Scheduler) headWaiter() *waiter {
	var head *waiter
	for _, w := range s.waiters {
		if head == nil || s.entryLess(w, head) {
			head = w
		}
	}
	return head
}

// entryLess is the shared queue's order: newcomerLess, or with newcomersFirst false plain oldest
// arrival first (then id) for every head.
func (s *Scheduler) entryLess(a, b *waiter) bool {
	if s.cfg.NewcomersFirst {
		return newcomerLess(a, b)
	}
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since)
	}
	return a.id < b.id
}

// newcomerLess is the entry order's total comparison (lower wins): a total order, so map
// iteration cannot leak into a decision.
func newcomerLess(a, b *waiter) bool {
	if a.returner != b.returner {
		return !a.returner // newcomers first
	}
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since) // oldest first
	}
	return a.id < b.id
}

// seat seats one waiting head on the engine the admission pass chose.
func (s *Scheduler) seat(g *engineGroup, w *waiter, coldTokens int64, now time.Time) bool {
	// THE TURN'S PREDICTION IS MEASURED HERE, BEFORE THE SEAT RESERVES ANYTHING: from this moment the
	// thread's blocks are resident by construction, so a measurement taken after the insert would
	// report the reservation as cache and every seat-entry dispatch would predict its whole prompt.
	seatObs := dispatchObservables{
		predictedCached: g.index.CachedTokens(w.chains),
		promptTokens:    PromptTokens(w.chains),
	}
	if ok, cost := g.index.TrySeat(w.id, w.chains, g.budgetBlocks); !ok {
		// RULE O — AN EMPTY ENGINE ADMITS ONE OVERSIZE HEAD, charging its FULL exact chain.
		if len(g.threads) != 0 || cost <= g.budgetBlocks {
			// Not Rule O: the chooser and this seat read the same numbers in the same critical section,
			// so this means they disagreed.
			return false
		}
		g.index.SeatForce(w.id, w.chains)
	}
	s.bindings[w.id] = g.id
	delete(s.waiters, w.id)
	m := &thread{
		id:           w.id,
		seatTime:     now,
		lastActivity: now,
		// the head that triggered this seating is still queued (the serve happens after the seat):
		// inherit its presence so the engine-idle rule cannot evict a thread that is about to
		// dispatch — and its own chains and class, so a later exit re-queues with this head's cost
		// rather than the chain of whatever ran.
		waitingSince: w.since,
		fairnessID:   w.fairnessID,
		wireID:       w.wireID,
		headChains:   w.chains,
		headReturner: w.returner,
		seatObs:      &seatObs,
	}
	// EVERY entry carries a cold charge (in tokens), CAPPED at the engine's own stagger cap: the
	// valve's job is to serialise cold prefills inside the budget's headroom, not to account the
	// entry's whole footprint (the seat's blocks do that).
	m.coldEntry = min(coldTokens, g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
	if s.cfg.RoomGate {
		m.coldEntry = 0 // no valve: nothing to charge or release
	}
	g.coldInFlight += m.coldEntry
	// record EVERY seat wait, split by kind (a thread's first turn vs a returning thread): both
	// answer "how long did admission make this thread wait".
	eventKind, waitKind := "re-entry", "reentry"
	if !w.returner {
		eventKind, waitKind = "newcomer", "new"
	}
	s.metrics.ObserveEntry(engineLabel(g.id), waitKind, now.Sub(w.since).Seconds())
	// the fields a run's forensics need: thread id, wire id, engine (the Event's own field), the
	// seat kind, the wait, and the arrival the wait is measured from
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f fairness_id=%s wire_id=%s since=%s",
		eventKind, now.Sub(w.since).Seconds(), fairnessOrNone(w.fairnessID), fairnessOrNone(w.wireID),
		w.since.UTC().Format(time.RFC3339Nano)), now)
	g.threads[w.id] = m
	return true
}

// ---- PLACEMENT ---------------------------------------------------------------

// EngineCandidate is one engine's offer for one waiting head — the FACTS chooseEngine decides on.
type EngineCandidate struct {
	// Engine is the engine id (the deterministic tie-break).
	Engine string
	// CostTokens is this head's admission cost ON THIS ENGINE (tokens), and CostBlocks its block-
	// quantised form.
	CostTokens int64
	CostBlocks int64
	// FreeBudget is what the engine can still admit (budget − accounted footprint).
	FreeBudget int64
	// Budget is the engine's SEAT BUDGET in blocks (fraction × its own num_gpu_blocks): the
	// number an EMPTY engine could admit, which is what RULE O's trigger is measured against.
	Budget int64
	// BudgetFits reports CostBlocks <= FreeBudget.
	BudgetFits bool
	// Fits reports BudgetFits AND that the COLD-ENTRY VALVE allows this head on this engine.
	Fits bool
	// HasVictim is LAZY (it costs a scan): may the engine evict an eligible incumbent right now?
	HasVictim func() bool
	// PhysColdTokens is what this engine would REALLY prefill for the head: its prompt beyond the
	// leading run the CPU-residency index says the engine's tier still holds. 0 for every engine when
	// residency is off, which leaves both placement orders exactly as without it.
	PhysColdTokens int64
	// Resident is the head's prompt tokens that tier holds (observability).
	Resident int64
	// RoomHeld (roomGate): BudgetFits, but the tier's room does not absorb the head's write. It is the
	// ONE "does not fit" that may still rotate a member out (the room-rotation guard decides), where a
	// valve-held head only waits. RoomForce: admitted by the hold's last resort. Need: the write in
	// blocks.
	RoomHeld  bool
	RoomForce bool
	Need      int
}

// chooseEngine answers "which engine takes this head?": among the engines where the head FITS
// (budget AND the cold valve), the one with the FEWEST TOKENS TO REALLY PREFILL (PhysColdTokens —
// all 0 with residency off), then the MOST FREE BUDGET AFTER the seat — free minus this head's cost
// on it; ties → lowest engine id. When nothing fits, the eviction that today's rule makes anyway is
// made on the engine chosen by the same order among the engines with an eligible victim: the
// residency index changes WHERE, never WHETHER or WHOM. It returns the index into cands, or -1.
func chooseEngine(cands []EngineCandidate) int {
	best := -1
	for i := range cands {
		if !cands[i].Fits {
			continue
		}
		if best < 0 || betterFit(cands[i], cands[best]) {
			best = i
		}
	}
	if best >= 0 {
		return best
	}
	if oversizeEverywhere(cands) {
		// RULE O, the no-empty-engine half: evict on the engine placement would pick anyway, and let
		// the next iteration re-ask.
		for i := range cands {
			if cands[i].HasVictim == nil || !cands[i].HasVictim() {
				continue
			}
			if best < 0 || betterVictimEngine(cands[i], cands[best]) {
				best = i
			}
		}
		return best
	}
	// The cold valve is holding this head somewhere: WAIT — never evict — because evicting frees
	// budget and never the valve. A ROOM-held head (roomGate) is different: a member leaving can make
	// room, so it falls through to the eviction choice (the room-rotation guard in seatWaiters decides
	// whether that exit is made).
	for i := range cands {
		if cands[i].BudgetFits && !cands[i].RoomHeld {
			return -1
		}
	}
	// Nothing fits by BUDGET either: eviction is the way forward.
	for i := range cands {
		if cands[i].HasVictim == nil || !cands[i].HasVictim() {
			continue
		}
		if best < 0 || betterVictimEngine(cands[i], cands[best]) {
			best = i
		}
	}
	return best
}

// betterFit is the placement total order among engines the head FITS: the fewest tokens to really
// prefill (equal everywhere with residency off), then the most free budget left AFTER the seat,
// then the engine id.
func betterFit(c, best EngineCandidate) bool {
	if c.PhysColdTokens != best.PhysColdTokens {
		return c.PhysColdTokens < best.PhysColdTokens
	}
	if c.FreeBudget-c.CostBlocks != best.FreeBudget-best.CostBlocks {
		return c.FreeBudget-c.CostBlocks > best.FreeBudget-best.CostBlocks
	}
	return c.Engine < best.Engine
}

// oversizeEverywhere reports RULE O's trigger: the head's exact cost exceeds the SEAT BUDGET
// of every candidate engine, so it could not fit one even if that engine were empty.
func oversizeEverywhere(cands []EngineCandidate) bool {
	if len(cands) == 0 {
		return false
	}
	for i := range cands {
		if cands[i].CostBlocks <= cands[i].Budget {
			return false
		}
	}
	return true
}

// betterVictimEngine is the order for the engine an eviction is made on when the head fits nowhere:
// the fewest tokens to really prefill (equal everywhere with residency off), then betterEngine.
func betterVictimEngine(c, best EngineCandidate) bool {
	if c.PhysColdTokens != best.PhysColdTokens {
		return c.PhysColdTokens < best.PhysColdTokens
	}
	return betterEngine(c, best)
}

// betterEngine is the placement total order for the engines that cannot take the head (and for
// naming the engine a valve-held head waits for): most free budget, then engine id.
func betterEngine(c, best EngineCandidate) bool {
	if c.FreeBudget != best.FreeBudget {
		return c.FreeBudget > best.FreeBudget
	}
	return c.Engine < best.Engine
}
