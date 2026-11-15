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

// THE DOOR FIX (session-layer contract Part A, `doorBackfill`): ONE outside thread — the head — may
// collect GPU seats, on ONE pod, and only when no pod can take it right now; every other outsider is
// admitted the moment it fits what is already free (backfill). It replaces ncaV2DecideOnce, and nothing
// else, when the switch is on; with it off (n.door == nil) none of this file runs (C7).
//
// One decision, in the contract's order (§7.7):
//
//	H1-H3  who is the head (sticky; a strictly higher outsider that cannot backfill takes over)
//	H4     the head fits some pod NOW: admit it there, nothing released anywhere (H5)
//	H6-H9  otherwise it collects on ONE pod, chosen once (not for a Part-B new-session head: H8)
//	L2-1   the pods' returners, v2's step unchanged (free budget excludes the held blocks)
//	H10    a Part-B new-session head collects only if no returner of its pod waits
//	BF     everyone else, in ranking order, into what is already free; never a release
//
// The decision is ONE admission or ONE seating at most, like v2's: the pass repeats it (A1's cap).

import (
	"fmt"
	"sort"
	"strconv"
	"time"
)

// nca2Door is the door's configuration and its state across passes (ST3).
type nca2Door struct {
	// sessionAware is Part B; header is the lower-case session header it reads (C1/C4).
	sessionAware bool
	header       string
	// backfillReclaim is BF4 (LOGIC Amendment 2, C8): backfill's second chance by reclaiming seats.
	backfillReclaim bool

	// ST3: the head and its collection pod survive the passes. The held blocks themselves stay in v2's
	// per-pod reservedHead/reservedBlocks, and only collectPod may hold any (I2).
	headID     string
	collectPod string

	// the PASS's ranking (R3): every outsider of the pass, sorted once; a decision skips the entries
	// that stopped being outsiders. rankIdx finds a thread's entry (its class for H3).
	rank    []doorCand
	rankIdx map[string]int
	// onLine is A4's question for every candidate of ONE decision: the pod each thread has an entry on.
	// Rebuilt (in place) per decision, because an admission's trim can drop an entry.
	onLine map[string]string
	// podBuf is the per-candidate pod order, reused so backfill allocates nothing per outsider.
	podBuf []podState
	// refused is the last exhaustive backfill sweep's refusals by reason (O5), flushed once per pass.
	refused [backfillRefusalReasons]int

	// sessions is Part B's state (nil in Part A).
	sessions *nca2Sessions

	// onStage is a TEST-ONLY seam: nil in production, called under the lock at fixed points of every
	// decision, so the invariant tests can read the state the rules are about (I4, I6, I7, I11) without
	// the decision computing anything for them.
	onStage func(doorStage)
}

// doorCand is one outsider in the pass's ranking with its ranking key.
type doorCand struct {
	w *waiter
	// class is R1's PRIORITY (0) / NEWCOMER (1), or R2's group A (0) / B (1) / C (2).
	class int
	// active is R2's within-group key for group A: the session's active threads.
	active int
}

// doorStage is what the test seam is told: the stage, the head after H1-H3 and whether a takeover kept
// the held blocks, and (at the end) the decision's path.
type doorStage struct {
	name string
	head string
	kept bool
	path string
	ok   bool
}

// The seam's stages.
const (
	doorStageStart    = "start"    // after H1-H3, before anything is admitted or released
	doorStageBackfill = "backfill" // BF begins
	doorStageEnd      = "end"      // the decision's result
)

// The door paths (session-layer O1/O7): how a turn got its seat.
const (
	doorPathSeated        = "seated"
	doorPathReturner      = "returner"
	doorPathHeadFit       = "head_fit"
	doorPathHeadCollected = "head_collected"
	doorPathBackfill      = "backfill"
)

// The collection events (session-layer O2, LOGIC Amendment 1 Q5).
const (
	collectStart            = "start"
	collectKeptOnTakeover   = "kept_on_takeover"
	collectReleasedTakeover = "released_takeover"
	collectReleasedCPU      = "released_cpu"
	collectReleasedHeadGone = "released_head_gone"
	collectReleasedPodGone  = "released_pod_gone"
	collectCompleted        = "completed"
)

// The head changes (session-layer O2).
const (
	headChangeAdmitted = "admitted"
	headChangeGone     = "gone"
	headChangeTakeover = "takeover"
)

// The backfill refusal reasons (O5), in the order a candidate gets FURTHER: a refusal is booked under
// the furthest check the candidate reached on any pod.
const (
	refusedCPU = iota
	refusedGPU
	refusedValve
	refusedValveHeadHeadroom
	backfillRefusalReasons
)

// backfillRefusalNames are O5's label values, indexed by the reasons above.
var backfillRefusalNames = [backfillRefusalReasons]string{"cpu", "gpu", "valve", "valve_head_headroom"}

// doorEcho is C6's (and C8's) part of the nca2 config echo: the resolved values, whether or not they were set.
func (n *nca2Policy) doorEcho() string {
	door, aware, header, reclaim := false, false, "", false
	if n != nil && n.door != nil {
		door, aware, header, reclaim = true, n.door.sessionAware, n.door.header, n.door.backfillReclaim
	}
	return fmt.Sprintf("doorBackfill=%s sessionAware=%s sessionHeader=%s backfillReclaim=%s",
		strconv.FormatBool(door), strconv.FormatBool(aware), header, strconv.FormatBool(reclaim))
}

// ---- the pass ----

// doorBeginPass is the pass's own derivation (R3, ST1): the session facts (Part B) and the ranking of
// every outsider, once. It runs after ncaV2SeatWaiters reset the pass caches.
func (s *Scheduler) doorBeginPass() {
	d := s.nca2.door
	if d.rankIdx == nil {
		d.rankIdx = map[string]int{}
		d.onLine = map[string]string{}
	}
	s.sessionBeginPass()
	d.rank = d.rank[:0]
	for id, w := range s.waiters {
		if s.nca2.admitted[id] {
			continue // a returner (A2.2), never an outsider
		}
		class, active := s.doorRankKey(w)
		d.rank = append(d.rank, doorCand{w: w, class: class, active: active})
	}
	sort.Slice(d.rank, func(i, j int) bool { return d.rankLess(d.rank[i], d.rank[j]) })
	clear(d.rankIdx)
	for i, c := range d.rank {
		d.rankIdx[c.w.id] = i
	}
	d.refused = [backfillRefusalReasons]int{}
}

// doorRankKey is R1's class (PRIORITY before NEWCOMER) or, in Part B, R2's group and active count.
func (s *Scheduler) doorRankKey(w *waiter) (class, active int) {
	if s.nca2.door.sessionAware {
		return s.sessionRankKey(w.id)
	}
	if s.nca2.ever[w.id] {
		return 0, 0
	}
	return 1, 0
}

// rankLess is R1/R2's total order: the class (group), then — in Part B's group A — the session's
// active threads, then the queue position, then the id.
func (d *nca2Door) rankLess(a, b doorCand) bool {
	if a.class != b.class {
		return a.class < b.class
	}
	if d.sessionAware && a.class == 0 && a.active != b.active {
		return a.active < b.active
	}
	return outsideLess(a.w, b.w)
}

// doorIsOutsider is §2's outsider: waiting and not admitted.
func (s *Scheduler) doorIsOutsider(id string) bool {
	if _, waiting := s.waiters[id]; !waiting {
		return false
	}
	return !s.nca2.admitted[id]
}

// doorFirstOutsider is the first entry of the pass's ranking that is still an outsider.
func (s *Scheduler) doorFirstOutsider() *doorCand {
	d := s.nca2.door
	for i := range d.rank {
		if s.doorIsOutsider(d.rank[i].w.id) {
			return &d.rank[i]
		}
	}
	return nil
}

// doorClassOf is a thread's class in the pass's ranking (the worst class when it is not ranked).
func (s *Scheduler) doorClassOf(id string) int {
	d := s.nca2.door
	if i, ok := d.rankIdx[id]; ok {
		return d.rank[i].class
	}
	return 3
}

// nca2CurrentHead is the outside head as the head-tenure gauge and the valve counter mean it (U8):
// v2's outside head, or — with the door on — the door's head (the first-ranked outsider while none is
// chosen yet).
func (s *Scheduler) nca2CurrentHead() *waiter {
	if s.nca2.door == nil {
		return s.nca2OutsideHead()
	}
	if id := s.nca2.door.headID; id != "" && s.doorIsOutsider(id) {
		return s.waiters[id]
	}
	if c := s.doorFirstOutsider(); c != nil {
		return c.w
	}
	return nil
}

// ---- one decision ----

// ncaDoorDecideOnce is ONE decision of the door, in §7.7's order.
func (s *Scheduler) ncaDoorDecideOnce(now time.Time) bool {
	d := s.nca2.door
	memo := s.nca2.passMemo
	// THE PODS OF THIS DECISION: A5's keys and rooms once (v2 computes them per decision too), and A4's
	// entry test for every candidate from one walk of the lines.
	base := s.ncaV2PodOrder(nil)
	s.doorIndexLines()
	head, kept := s.doorSelectHead(base, memo, now)
	if d.onStage != nil {
		d.onStage(doorStage{name: doorStageStart, head: d.headID, kept: kept})
	}
	path, ok := s.doorDecide(base, head, memo, now)
	if d.onStage != nil {
		d.onStage(doorStage{name: doorStageEnd, head: d.headID, path: path, ok: ok})
	}
	return ok
}

// doorDecide is the decision after the head is known: H4, the collection, the returners, H10, backfill.
func (s *Scheduler) doorDecide(base []podState, head *waiter, memo map[fitKey]fitFacts, now time.Time) (string, bool) {
	d := s.nca2.door
	deferred := false
	var collect *podState
	if head != nil {
		pods := s.doorPodOrder(base, head)
		// H4/H5: every pod is tried with the budget AS IT IS before any seat is released anywhere
		if s.doorHeadFitNow(pods, head, now) {
			return doorPathHeadFit, true
		}
		// H6/H7: the one collection pod (a copy: the pod-order buffer is reused by every later candidate)
		if c := s.doorCollectionPod(pods, head, now); c != nil {
			cp := *c
			collect = &cp
		}
		// H8: a Part-B new-session head waits for its pod's returners (H10 below)
		deferred = d.sessionAware && s.doorClassOf(head.id) == 2
		if collect != nil && !deferred && s.doorCollect(collect, head, now) {
			return doorPathHeadCollected, true
		}
	}
	// L2-1: the returners, v2's step (the A5 order with the head's A4 preference, as v2 has it)
	if s.doorReturners(s.doorPodOrder(base, head), memo, now) {
		return doorPathReturner, true
	}
	// H10: a Part-B new-session head collects only budget no returner of its pod is waiting for
	if head != nil && deferred && collect != nil && !s.doorReturnerWaitingOn(collect.g) &&
		s.doorCollect(collect, head, now) {
		return doorPathHeadCollected, true
	}
	if d.onStage != nil {
		d.onStage(doorStage{name: doorStageBackfill, head: d.headID})
	}
	if s.doorBackfill(base, head, memo, now) {
		return doorPathBackfill, true
	}
	// BF3: nothing could be admitted or seated: the pass ends here, and this sweep was exhaustive (O5)
	s.doorFlushRefusals()
	return "", false
}

// ---- H1-H3: the head ----

// doorSelectHead is H1-H3. kept reports a takeover that kept the held blocks (H3 on the same pod).
func (s *Scheduler) doorSelectHead(base []podState, memo map[fitKey]fitFacts, now time.Time) (*waiter, bool) {
	d := s.nca2.door
	// H1: the head left the queue (cancelled, gone): its collection goes with it (I11)
	if d.headID != "" && !s.doorIsOutsider(d.headID) {
		s.doorReleaseAll(collectReleasedHeadGone, now)
		s.nca2m.ObserveHeadChange(headChangeGone)
		d.headID = ""
	}
	first := s.doorFirstOutsider()
	if d.headID == "" {
		if first == nil {
			return nil, false
		}
		d.headID, d.collectPod = first.w.id, ""
		return first.w, false
	}
	head := s.waiters[d.headID]
	// H2: the head is sticky. H3: a strictly higher class/group that cannot be admitted by backfill
	// anywhere now takes over (LOGIC Amendment 1 Q3).
	if first == nil || first.w.id == d.headID || first.class >= s.doorClassOf(d.headID) {
		return head, false
	}
	if s.doorBackfillsAnywhere(base, first.w, head, memo) {
		return head, false
	}
	x := first.w
	kept := false
	if d.collectPod != "" {
		// the taker keeps the held blocks only if ITS collection pod (H6, computed for x) is this one
		if p := s.doorChoosePod(s.doorPodOrder(base, x), x); p != nil && p.g.id == d.collectPod {
			p.g.nca2.reservedHead = x.id
			kept = true
			s.nca2m.ObserveCollection(collectKeptOnTakeover)
			s.recordEvent("COLLECT", x.id, fmt.Sprintf("event=%s pod=%s held=%d from=%s",
				collectKeptOnTakeover, engineLabel(p.g.id), p.g.nca2.reservedBlocks, d.headID), now)
		} else {
			s.doorReleaseAll(collectReleasedTakeover, now)
		}
	}
	s.nca2m.ObserveHeadChange(headChangeTakeover)
	d.headID = x.id
	return x, kept
}

// doorReleaseAll releases every pod's held blocks (H1, H3, H7, E9, the head's admission) and books the
// collection event when a collection existed.
func (s *Scheduler) doorReleaseAll(event string, now time.Time) {
	d := s.nca2.door
	var held int64
	for _, g := range s.groups {
		if g.nca2 == nil {
			continue
		}
		held += g.nca2.reservedBlocks
		g.nca2.reservedHead, g.nca2.reservedBlocks = "", 0
	}
	if d.collectPod != "" {
		s.nca2m.ObserveCollection(event)
		s.recordEvent("COLLECT", d.headID, fmt.Sprintf("event=%s pod=%s held=%d",
			event, engineLabel(d.collectPod), held), now)
	}
	d.collectPod = ""
}

// ---- H4: the head fits now ----

// doorHeadFitNow is H4: the head is admitted on the first pod of its order that takes it NOW — the
// valve, the room, and its exact cost inside the free budget (all of it: the held blocks are its own),
// or Rule O on an empty pod (E4). Nothing is released (H5). The head's numbers are computed fresh, so a
// shared prefix moved by this pass's own seats cannot make it collect while it fits (I6).
func (s *Scheduler) doorHeadFitNow(pods []podState, head *waiter, now time.Time) bool {
	f := s.nca2CandOf(head).tokens
	for i := range pods {
		p := &pods[i]
		if !s.doorHeadFitsPod(p, head, f) {
			continue
		}
		if s.doorAdmit(p, head, doorPathHeadFit, now) {
			return true
		}
	}
	return false
}

// doorHeadFitsPod is fit-now(head, P): V, C and G with the budget as it is (Rule O included).
func (s *Scheduler) doorHeadFitsPod(p *podState, head *waiter, f int64) bool {
	g := p.g
	if g.numBlocks <= 0 || f > p.room {
		return false
	}
	if !s.coldAllowsTokens(g, g.index.ColdTokens(head.chains)) {
		return false
	}
	cost := int64(g.index.Cost(head.chains))
	return cost <= s.ncaV2FreeFor(g, head) || (len(g.threads) == 0 && cost > int64(g.budgetBlocks))
}

// ---- H6-H9: the collection ----

// doorChoosePod is H6's choice for a head: the first pod of its order that passes the valve and the CPU
// check (nil when none does: the head waits for room or the valve).
func (s *Scheduler) doorChoosePod(pods []podState, head *waiter) *podState {
	f := s.nca2CandOf(head).tokens
	for i := range pods {
		p := &pods[i]
		if p.g.numBlocks <= 0 || p.g.nca2 == nil || f > p.room {
			continue
		}
		if !s.coldAllowsTokens(p.g, p.g.index.ColdTokens(head.chains)) {
			continue
		}
		return p
	}
	return nil
}

// doorCollectionPod is H6/H7: the collection pod is chosen ONCE and kept while the head passes the CPU
// check there. Losing the room releases it and the next pod is chosen; the valve refusing the head is
// transient and keeps it (LOGIC Amendment 1 Q1).
func (s *Scheduler) doorCollectionPod(pods []podState, head *waiter, now time.Time) *podState {
	d := s.nca2.door
	f := s.nca2CandOf(head).tokens
	if d.collectPod != "" {
		for i := range pods {
			if pods[i].g.id != d.collectPod {
				continue
			}
			if f <= pods[i].room {
				return &pods[i] // H7: kept
			}
			break
		}
		// H7: no longer passes C there (or the pod is gone: E9 releases at the reset already)
		s.doorReleaseAll(collectReleasedCPU, now)
	}
	p := s.doorChoosePod(pods, head)
	if p == nil {
		return nil
	}
	d.collectPod = p.g.id
	p.g.nca2.reservedHead = head.id
	s.nca2m.ObserveCollection(collectStart)
	s.recordEvent("COLLECT", head.id, fmt.Sprintf("event=%s pod=%s held=%d",
		collectStart, engineLabel(p.g.id), p.g.nca2.reservedBlocks), now)
	return p
}

// doorCollect is H9, v2's A9 loop restricted to the ONE collection pod: the valve first (a refusal keeps
// what is held and releases nothing: LOGIC Amendment 1 Q1), then the eligible seats released one at a
// time in the rotationVictim order until the head fits — admitted — or none is left.
//
// The held amount is LOGIC Amendment 1 Q7 and the parent dry-run's D1: EVERYTHING free on the collection
// pod is the head's, up to its cost, whatever freed it (the head's own releases, a push-out, a seat that
// ended) — design A3's level-2 order, the head first. Held only grows: v2's deficit cap (cost − free) would
// hand 2·free − cost back to the pod's other candidates once the head has gathered half its cost, and with
// backfill that surplus is always taken. A valve refusal holds nothing new (Q1).
func (s *Scheduler) doorCollect(p *podState, head *waiter, now time.Time) bool {
	g := p.g
	if !s.coldAllowsTokens(g, g.index.ColdTokens(head.chains)) {
		return false
	}
	budget := int64(g.budgetBlocks)
	for {
		cost := int64(g.index.Cost(head.chains))
		if cost <= s.freeBudget(g) || (len(g.threads) == 0 && cost > budget) {
			return s.doorAdmit(p, head, doorPathHeadCollected, now)
		}
		g.nca2.reservedHead = head.id
		g.nca2.reservedBlocks = min(cost, max(g.nca2.reservedBlocks, s.freeBudget(g)))
		reason, victim := s.rotationVictim(g, now)
		if victim == nil {
			if g.nca2.reservedBlocks > cost {
				g.nca2.reservedBlocks = cost
			}
			return false
		}
		s.exit(g, victim, reason, now)
	}
}

// ---- the admission ----

// doorAdmit is A2.1 through the door: v2's admission and seat, the door's counters, and — for the head —
// the end of its tenure (its collection is complete, the next decision's H1 picks the next head).
func (s *Scheduler) doorAdmit(p *podState, w *waiter, doorPath string, now time.Time) bool {
	d := s.nca2.door
	class := s.doorClassLabel(w.id)
	if !s.nca2AdmitPath(p.g, w, s.nca2CandOf(w).tokens, p.room, doorPath, now) {
		return false
	}
	s.nca2m.ObserveDoorAdmission(doorPath, class)
	if w.id == d.headID {
		if d.collectPod != "" {
			s.doorReleaseAll(collectCompleted, now)
		}
		s.nca2m.ObserveHeadChange(headChangeAdmitted)
		d.headID = ""
	}
	return true
}

// doorClassLabel is O1's class label: priority | newcomer in Part A, A | B | C in Part B.
func (s *Scheduler) doorClassLabel(id string) string {
	class := s.doorClassOf(id)
	if s.nca2.door.sessionAware {
		switch class {
		case 0:
			return "A"
		case 1:
			return "B"
		}
		return "C"
	}
	if class == 0 {
		return pathPriority
	}
	return pathNewcomer
}

// ---- L2-1: the returners ----

// doorReturners is L2-1: v2's returner step, unchanged — each pod's returners in the seat order (L2-2),
// v2's fit (the valve, then the cost inside the free budget that excludes the held blocks, plus what the
// pod's reclaimable seats would return), v2's reclaim, the seat.
func (s *Scheduler) doorReturners(pods []podState, memo map[fitKey]fitFacts, now time.Time) bool {
	for i := range pods {
		p := &pods[i]
		if p.g.numBlocks <= 0 {
			continue
		}
		s.doorBuildWarm()
		for _, w := range s.nca2.warm[p.g.id] {
			if _, live := s.waiters[w.id]; !live {
				continue
			}
			if !s.ncaV2Fits(p.g, w, now, memo) || !s.ncaV2Reclaim(p.g, w, now, memo) {
				continue
			}
			if s.nca2SeatReturnerPath(p.g, w, p.room, doorPathReturner, now) {
				return true
			}
		}
	}
	return false
}

// doorBuildWarm builds the pass's returner lists once (lazily, like v2's), in L2-2's seat order.
func (s *Scheduler) doorBuildWarm() {
	if s.nca2.warmBuilt {
		return
	}
	for _, g := range s.activeGroups() {
		if g.nca2 != nil && g.numBlocks > 0 {
			s.nca2.warm[g.id] = s.sessionSeatOrder(s.nca2WarmReturners(g))
		}
	}
	s.nca2.warmBuilt = true
}

// doorReturnerWaitingOn is H10's question: does any returner of this pod still wait?
func (s *Scheduler) doorReturnerWaitingOn(g *engineGroup) bool {
	s.doorBuildWarm()
	for _, w := range s.nca2.warm[g.id] {
		if _, live := s.waiters[w.id]; live {
			return true
		}
	}
	return false
}

// ---- BF: backfill ----

// doorBackfill is BF1: every outsider but the head, in ranking order, on each pod of its own order,
// admitted if it passes the three checks with what is ALREADY free — no rotationVictim, no exit, no
// reclaim (I4) — and, on the collection pod, neither the held blocks nor the valve headroom the head
// needs (I5). The first admission ends the decision.
func (s *Scheduler) doorBackfill(base []podState, head *waiter, memo map[fitKey]fitFacts, now time.Time) bool {
	d := s.nca2.door
	var maxRoom int64
	for i := range base {
		if base[i].g.numBlocks > 0 && base[i].room > maxRoom {
			maxRoom = base[i].room
		}
	}
	var collect *engineGroup
	var headNeed int64
	if head != nil && d.collectPod != "" {
		if g := s.groups[d.collectPod]; g != nil {
			collect = g
			headNeed = min(g.index.ColdTokens(head.chains), g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
		}
	}
	d.refused = [backfillRefusalReasons]int{}
	for i := range d.rank {
		w := d.rank[i].w
		if (head != nil && w.id == head.id) || !s.doorIsOutsider(w.id) {
			continue
		}
		f := s.nca2CandOf(w).tokens
		// BF2: the CPU prefilter is exact — no pod has the room
		if f > maxRoom {
			d.refused[refusedCPU]++
			continue
		}
		furthest := refusedCPU
		pods := s.doorPodOrder(base, w)
		for j := range pods {
			reason, ok := s.doorBackfillFits(&pods[j], w, f, collect, headNeed, memo)
			if ok {
				if s.doorAdmit(&pods[j], w, doorPathBackfill, now) {
					return true
				}
				continue
			}
			furthest = max(furthest, reason)
		}
		d.refused[furthest]++
	}
	return false
}

// doorBackfillFits is BF1's three checks for one candidate on one pod, cheapest first, with the reason a
// refusal is booked under (O5). The exact cost is re-read before the grant: the pass memo may predate
// this pass's own releases, and a stale LOW cost would let the seat spend held blocks (I5).
func (s *Scheduler) doorBackfillFits(p *podState, w *waiter, f int64, collect *engineGroup, headNeed int64,
	memo map[fitKey]fitFacts) (int, bool) {
	g := p.g
	if g.numBlocks <= 0 || f > p.room {
		return refusedCPU, false
	}
	free := s.ncaV2FreeFor(g, w)
	// BF2 (LOGIC Amendment 1 Q2): a SOUND lower bound before the exact cost
	if s.doorCostLowerBound(g, w) > free || s.ncaV2Cost(g, w, memo).cost > free {
		return refusedGPU, false
	}
	cold := s.ncaV2Cold(g, w, memo).cold
	if !s.coldAllowsTokens(g, cold) {
		return refusedValve, false
	}
	if g == collect {
		if capTokens := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize); capTokens > 0 &&
			g.coldInFlight+min(cold, capTokens)+headNeed > capTokens {
			return refusedValveHeadHeadroom, false
		}
	}
	exact := fitFacts{cost: int64(g.index.Cost(w.chains)), cold: g.index.ColdTokens(w.chains), costKnown: true, coldKnown: true}
	memo[fitKey{pod: g.id, thread: w.id}] = exact
	if exact.cost > free {
		return refusedGPU, false
	}
	if !s.coldAllowsTokens(g, exact.cold) {
		return refusedValve, false
	}
	return 0, true
}

// doorBackfillsAnywhere is H3's "cannot backfill anywhere" (LOGIC Amendment 1 Q3), read with the
// backfill's own checks against the current head's collection; it changes nothing.
func (s *Scheduler) doorBackfillsAnywhere(base []podState, x, head *waiter, memo map[fitKey]fitFacts) bool {
	d := s.nca2.door
	var collect *engineGroup
	var headNeed int64
	if head != nil && d.collectPod != "" {
		if g := s.groups[d.collectPod]; g != nil {
			collect = g
			headNeed = min(g.index.ColdTokens(head.chains), g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
		}
	}
	f := s.nca2CandOf(x).tokens
	for i := range base {
		if _, ok := s.doorBackfillFits(&base[i], x, f, collect, headNeed, memo); ok {
			return true
		}
	}
	return false
}

// doorCostLowerBound is BF2's GPU prefilter: no thread costs fewer blocks than the longest non-resident
// SUFFIX of any one of its chains. Residency is prefix-closed (the same property the valve's ColdTokens
// relies on), so the suffix past the resident prefix is all new blocks; one binary search per chain, no
// allocation. A token-size bound would be an UPPER bound — it would skip a sub-agent whose parent's
// prefix is resident while it fits.
func (s *Scheduler) doorCostLowerBound(g *engineGroup, w *waiter) int64 {
	var lb int64
	for _, c := range w.chains {
		lb = max(lb, int64(len(c.Keys)-g.index.residentPrefix(c.Keys)))
	}
	return lb
}

// doorFlushRefusals books the exhaustive sweep's refusals (O5): once per pass, one count per refused
// outsider under the furthest check it reached.
func (s *Scheduler) doorFlushRefusals() {
	d := s.nca2.door
	for reason, n := range d.refused {
		if n > 0 {
			s.nca2m.ObserveBackfillRefused(backfillRefusalNames[reason], n)
		}
	}
	d.refused = [backfillRefusalReasons]int{}
}

// ---- the debug state (O9) ----

// doorSnapshot fills the door's part of the nca2 snapshot. It only READS (I9): the session groups are
// derived here from the live state, without the pass's memo.
func (s *Scheduler) doorSnapshot(snap *NCA2Snapshot) {
	d := s.nca2.door
	if d == nil {
		return
	}
	if d.headID != "" && s.doorIsOutsider(d.headID) {
		snap.HeadID = d.headID
	}
	snap.CollectPod = d.collectPod
	var held int64
	if g := s.groups[d.collectPod]; g != nil && g.nca2 != nil {
		held = g.nca2.reservedBlocks
	}
	snap.HeldBlocks = &held
	if !d.sessionAware {
		snap.DoorWaitingByGroup = map[string]int{pathPriority: 0, pathNewcomer: 0}
		for id := range s.waiters {
			switch {
			case s.nca2.admitted[id]:
			case s.nca2.ever[id]:
				snap.DoorWaitingByGroup[pathPriority]++
			default:
				snap.DoorWaitingByGroup[pathNewcomer]++
			}
		}
		return
	}
	inside := s.sessionsInsideNow()
	n := len(inside)
	snap.SessionsInside = &n
	snap.DoorWaitingByGroup = map[string]int{"A": 0, "B": 0, "C": 0}
	for id := range s.waiters {
		if s.nca2.admitted[id] {
			continue
		}
		key := s.sessionKey(id)
		switch {
		case inside[key]:
			snap.DoorWaitingByGroup["A"]++
		case s.sessionEverInNow(key):
			snap.DoorWaitingByGroup["B"]++
		default:
			snap.DoorWaitingByGroup["C"]++
		}
	}
}

// ---- the pod order of one candidate ----

// doorIndexLines is A4's entry test for every candidate of this decision: one walk of the lines.
func (s *Scheduler) doorIndexLines() {
	d := s.nca2.door
	clear(d.onLine)
	for _, g := range s.groups {
		if g.nca2 == nil {
			continue
		}
		for _, e := range g.nca2.line {
			d.onLine[e.thread] = g.id
		}
	}
}

// doorPodOrder is pod order(w): the decision's A5 order, with w's old pod first when its entry still
// exists there (A4) — v2's ncaV2PodOrder, without recomputing the pods per candidate. The slice is the
// door's reusable buffer: valid until the next call.
func (s *Scheduler) doorPodOrder(base []podState, w *waiter) []podState {
	d := s.nca2.door
	d.podBuf = append(d.podBuf[:0], base...)
	if w == nil {
		return d.podBuf
	}
	home, ok := s.nca2.homes[w.id]
	if !ok || d.onLine[w.id] != home {
		return d.podBuf
	}
	for i := range d.podBuf {
		if d.podBuf[i].g.id == home {
			if i > 0 {
				preferred := d.podBuf[i]
				copy(d.podBuf[1:i+1], d.podBuf[:i])
				d.podBuf[0] = preferred
			}
			break
		}
	}
	return d.podBuf
}
