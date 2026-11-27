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

// FAIR ROTATION (fairRotation): one design for WHO LEAVES the group, so a session's progress is
// guaranteed (the quantum is a floor) and seat time is shared (the longest-seated eligible member leaves
// first) — never chosen by what is cached.
//
//   eligible   = quantum done (Q turns this stay, or running the turn that completes them, past its first
//                token) OR idle (nothing in flight, nothing queued, idleSeconds quiet)
//   BUDGET     — the head does not fit: remove idle members first (longest idle first), then quantum-done
//                members by the LONGEST TIME IN THE GROUP since they joined (costFairRotation: the highest
//                seat time ÷ context size), until it fits
//   ROOM       — counted free + non-member + idle + removed blocks from the tail; a held head removes only
//                the member whose block ends the walk, and only if it is eligible; otherwise it WAITS
//   REMOVED    — out at once: no further turns, seat budget free, blocks room; an in-flight request
//                finishes and counts on the GPU until it ends; its next turn queues behind the head

import (
	"sort"
	"time"
)

// fairIdle: nothing in flight, nothing queued, quiet for idleSeconds.
func (s *Scheduler) fairIdle(m *thread, now time.Time) bool { return !m.busy() && s.silent(m, now) }

// fairQuantumDone: the member has used its quantum this stay — Q completed turns between turns, or
// running the turn that completes them once that request is past its first token (its prefill no
// longer writes into the CPU tier, and the request in flight is its last).
func (s *Scheduler) fairQuantumDone(m *thread) bool {
	if !m.busy() {
		return m.busyPeriods >= s.cfg.Quantum
	}
	return m.busyPeriods >= s.cfg.Quantum-1 && s.pastFirstToken(m.id)
}

// pastFirstToken: every request the thread has released (pending ones included) has produced its first
// chunk.
func (s *Scheduler) pastFirstToken(id string) bool {
	n := 0
	for _, rec := range s.reqs {
		if rec.thread != id {
			continue
		}
		if !rec.firstToken {
			return false
		}
		n++
	}
	return n > 0
}

// roomEligible is who the ROOM may take without a choice: the rotation's eligible members, or — with
// fairRotation — only IDLE ones (removed members are already out: their blocks are not staying).
func (s *Scheduler) roomEligible(m *thread, now time.Time) bool {
	if s.cfg.FairRotation {
		return s.fairIdle(m, now)
	}
	_, ok := s.eligible(m, now)
	return ok
}

// fairTail is what a head's write would evict on g (costFairRotation): the blocks beyond the tier's free
// capacity, from the eviction end, the head's own blocks skipped (its own touch refreshes them), and how
// many of them a STAYING member holds (t0). Empty with no head (the growth path) or no tier.
type fairTail struct {
	bottom map[Hash]struct{}
	t0     int
}

func (s *Scheduler) fairTailFor(g *engineGroup, head *waiter) fairTail {
	var ft fairTail
	if head == nil || g.tier == nil {
		return ft
	}
	resident := s.residentTokens(head.chains, []*engineGroup{g})
	n := s.roomNeed(physCold(PromptTokens(head.chains), resident, 0)) - (g.tier.Cap() - g.tier.Len())
	if n <= 0 {
		return ft
	}
	own := head.ownKeys()
	keys := g.tier.TailKeys(n, func(k Hash) bool { _, ok := own[k]; return ok })
	ft.bottom = make(map[Hash]struct{}, len(keys))
	for _, k := range keys {
		ft.bottom[k] = struct{}{}
		if g.index.blocksStay[k] > 0 {
			ft.t0++
		}
	}
	return ft
}

// fairCost is what sending m out costs, in blocks (>= 1): the blocks only m holds among staying members
// (what its leaving frees, and what it will rebuild) + the staying blocks the head's write evicts anyway −
// those m alone holds there; saved = that last term (the thrash, or the room block, its leaving removes).
func (s *Scheduler) fairCost(g *engineGroup, m *thread, ft fairTail) (cost, saved int) {
	sb := g.index.seats[m.id]
	if sb == nil {
		return 1, 0
	}
	excl := 0
	for h := range sb.charged {
		if g.index.blocksStay[h] == 1 { // m is the only staying holder
			excl++
			if _, ok := ft.bottom[h]; ok {
				saved++
			}
		}
	}
	if s.cfg.MemberTierEviction {
		// m's leaving moves ALL its exclusive blocks to the tier's eviction end, so the head's write takes them
		// before any staying block: the others lose max(0, t0-excl), and the cost is max(excl, t0)
		saved = min(excl, ft.t0)
	}
	return max(1, excl+ft.t0-saved), saved
}

// fairBefore is the budget's victim order: idle first (longest idle first); then among quantum-done members
// the longest time in the group since joining — or, with costFairRotation, the highest seat time ÷ cost
// (compared by cross-multiplication: no division); then the oldest dispatch, then id.
func (s *Scheduler) fairBefore(ia bool, a *thread, ca int, ib bool, b *thread, cb int, now time.Time) bool {
	if ia != ib {
		return ia
	}
	if ia {
		if !a.lastActivity.Equal(b.lastActivity) {
			return a.lastActivity.Before(b.lastActivity)
		}
	} else if s.cfg.CostFairRotation {
		sa, sb := nonNegative(now.Sub(a.seatTime)), nonNegative(now.Sub(b.seatTime))
		if l, r := sa*float64(cb), sb*float64(ca); l != r {
			return l > r
		}
	} else if !a.seatTime.Equal(b.seatTime) {
		return a.seatTime.Before(b.seatTime)
	}
	if !touchedAt(a).Equal(touchedAt(b)) {
		return touchedAt(a).Before(touchedAt(b))
	}
	return a.id < b.id
}

// fairVictim is the budget's next victim on g (nil = none eligible). head may be nil.
func (s *Scheduler) fairVictim(g *engineGroup, head *waiter, now time.Time) *thread {
	v, _ := s.fairVictimSaved(g, head, now)
	return v
}

// fairVictimSaved is fairVictim plus the staying blocks at the tail its leaving takes off the head's write.
func (s *Scheduler) fairVictimSaved(g *engineGroup, head *waiter, now time.Time) (*thread, int) {
	var ft fairTail
	if s.cfg.CostFairRotation {
		ft = s.fairTailFor(g, head)
	}
	var best *thread
	bestIdle, bestCost, bestSaved := false, 1, 0
	for _, m := range g.threads {
		if (head != nil && m.id == head.id) || g.index.Exiting(m.id) {
			continue
		}
		idle := s.fairIdle(m, now)
		if !idle && !s.fairQuantumDone(m) {
			continue
		}
		cost, saved := 1, 0
		if s.cfg.CostFairRotation && !idle {
			cost, saved = s.fairCost(g, m, ft)
		}
		if best == nil || s.fairBefore(idle, m, cost, bestIdle, best, bestCost, now) {
			best, bestIdle, bestCost, bestSaved = m, idle, cost, saved
		}
	}
	return best, bestSaved
}

// fairBottom is the member(s) holding the block that ends head's room walk on g, when EVERY staying
// holder of it is quantum-done (the walk never stops at idle or removed members' blocks); nil when the
// walk did not stop at a block, or a holder is protected — the head then waits.
func (s *Scheduler) fairBottom(g *engineGroup, head *waiter, need int) []*thread {
	if g.tier == nil {
		return nil
	}
	skip, stop := s.roomPreds(g, head.ownKeys(), nil)
	_, k, stopped := g.tier.TailRoomStop(need, skip, stop)
	if !stopped {
		return nil
	}
	var out []*thread
	for _, m := range g.threads {
		if g.index.Exiting(m.id) {
			continue
		}
		sb := g.index.seats[m.id]
		if sb == nil {
			continue
		}
		if _, ok := sb.charged[k]; !ok {
			continue
		}
		if !s.fairQuantumDone(m) {
			return nil // a protected member is in the way: wait
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// fairRemove takes m out of the group for head: between turns it exits now; in flight it is marked
// leaving (not dispatched again, its blocks no longer staying, its seat budget free on the stay side)
// and drains at its request's end. A re-queued turn goes behind the head it left for.
func (s *Scheduler) fairRemove(g *engineGroup, m *thread, head *waiter, reason string, now time.Time) {
	s.counters.fairRemovals[reason]++
	if m.busy() {
		g.index.MarkExiting(m.id)
		m.exitingSince = now
		m.exitReason = reason
		return
	}
	s.exit(g, m, reason, now)
	if head == nil {
		return // the growth path: no head to queue behind
	}
	if w, ok := s.waiters[m.id]; ok && !entryKey(w).After(entryKey(head)) {
		w.floor = entryKey(head).Add(time.Nanosecond)
	}
}

// fairStep is fair rotation's one move for a head that does not fit (the caller re-evaluates the head
// when it returns true). Budget: remove the fair victim. Room: remove the member at the bottom if it may
// leave.
func (s *Scheduler) fairStep(c *seatChoice, head *waiter, now time.Time) bool {
	g := c.g
	if c.roomHeld {
		vs := s.fairBottom(g, head, c.writeBlocks)
		for _, v := range vs {
			s.fairRemove(g, v, head, "fair-room", now)
		}
		return len(vs) > 0
	}
	v, saved := s.fairVictimSaved(g, head, now)
	if v == nil {
		return false
	}
	s.counters.fairTailSaved += int64(saved)
	s.metrics.ObserveFairTailSaved(engineLabel(g.id), saved)
	reason := "fair-budget"
	if s.fairIdle(v, now) {
		reason = "idle"
	}
	s.fairRemove(g, v, head, reason, now)
	return true
}
