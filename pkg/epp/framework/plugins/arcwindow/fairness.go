/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package arcwindow

import (
	"context"
	"sort"
	"time"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// candidate mirrors the sibling plugin's queue-head view.
type candidate struct {
	queue     fwkfc.FlowQueueAccessor
	session   string
	arrival   time.Time
	d         time.Duration // the queue's TTL (the deadline D when set)
	sizeBytes int64
	anon      bool

	est     float64
	shat    time.Duration
	dEff    time.Duration
	slack   time.Duration
	offRing bool // parked outside the ring budget (population tier)
}

// NewState satisfies the FairnessPolicy contract; Pick reads the shared
// manager, so the per-band state is nil.
func (r *ArcWindow) NewState(_ context.Context) any { return nil }

// Pick is the arc's dispatch decision, one queue per Pick:
//
//  0. OFF-SWITCH — the idle pool cannot hold one mean footprint: serve
//     oldest-arrival-first without moving the arc (the honest FCFS regime).
//  1. D-DEMAND — the oldest waiting session past its deadline is served
//     now, and if it was uncovered the arc JUMPS to cover it (re-anchored
//     epoch). The bound is on DISPATCH, not coverage (Revision 1 fix 2).
//  2. EPOCH SWEEP — serve waiting epoch-members in ring order
//     (trailing → leading). When every epoch member that waited has been
//     served, the epoch completes; the arc advances one position (the
//     heartbeat) only if an uncovered session waits (the freeze rule).
//  3. THE FORK (A5) — no member waits:
//     serveUncovered=false: nothing dispatches (the pure arc — cache
//     protection by construction; the engine waits for the arc);
//     serveUncovered=true: the longest-waiting uncovered session
//     dispatches without moving the arc (the hybrid guest path).
func (r *ArcWindow) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()
	m := r.mgr

	var cands []candidate
	waiting := map[string]bool{}
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		req := head.OriginalRequest()
		sizeBytes := int64(0)
		if req != nil {
			if ir := req.InferenceRequest(); ir != nil && ir.RequestSizeBytes > 0 {
				sizeBytes = int64(ir.RequestSizeBytes)
			} else if n := req.ByteSize(); n > 0 {
				sizeBytes = int64(n)
			}
		}
		c := candidate{
			queue:     queue,
			session:   headSessionID(head),
			arrival:   head.EnqueueTime(),
			sizeBytes: sizeBytes,
		}
		c.d = head.EffectiveTTL()
		if c.d < 0 {
			c.d = 0
		}
		c.anon = c.session == ""
		if !c.anon {
			waiting[c.session] = true
		}
		cands = append(cands, c)
		return true
	})
	r.metrics.waiting.Set(float64(len(cands)))
	if len(cands) == 0 {
		return nil, nil //nolint:nilnil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.sweepLocked(now)
	m.updateWidth(now)

	// Ledger view: every waiting session exists on the ring before it is
	// ever picked (first sight is the queue, same as the sibling).
	ringMassNow := m.ringMass() // computed once; the population tier's gate
	var meanDur time.Duration
	{
		var sum time.Duration
		var n int
		for _, s := range m.sessions {
			if s.lastTurn > 0 {
				sum += s.lastTurn
				n++
			}
		}
		if n > 0 {
			meanDur = sum / time.Duration(n)
		}
	}
	for i := range cands {
		c := &cands[i]
		if c.anon {
			continue
		}
		s := m.ensureSession(c.session, now)
		estimate := max(estimateTokens(int(c.sizeBytes)), 1)
		c.est = max(s.committed, float64(estimate))
		// The population tier: first-seen sessions beyond the ring
		// budget park off-ring (oldest-first admission when mass frees).
		// ringMass is computed ONCE per Pick (the O(N^2) fix).
		if len(m.sessions) == 1 || s.turns == 0 {
			if m.maybeOffRingMass(c.session, c.est, now, ringMassNow) {
				c.offRing = true
			}
		}
		c.shat = s.lastTurn
		if c.shat <= 0 {
			c.shat = meanDur
		}
		c.dEff = c.d
		// The elastic floor stretches the deadline by queue depth over
		// CONCURRENCY (ŝ·N̂/ĉ) — never over token capacity (B3: token
		// units made D=0 press every candidate within microseconds and
		// the policy degenerated to FCFS). ĉ=0 at cold start means no
		// stretch yet (guard n>0).
		if ĉ := m.inflightCountLocked(); ĉ > 0 {
			if stretch := meanDur * time.Duration(len(cands)) / time.Duration(ĉ); stretch > c.dEff {
				c.dEff = stretch
			}
		}
		c.slack = c.arrival.Add(c.dEff).Sub(now) - c.shat
	}
	// Deterministic candidate order: ring order (position), anon last.
	sort.Slice(cands, func(i, j int) bool {
		si, sj := m.sessions[cands[i].session], m.sessions[cands[j].session]
		if cands[i].anon || cands[j].anon {
			return !cands[i].anon && cands[j].anon
		}
		if si == nil || sj == nil {
			return si != nil
		}
		return si.pos < sj.pos
	})

	// 0. OFF-SWITCH.
	if f := m.meanFootprint(); f > 0 {
		if p := m.idlePool(); p >= 0 && p < f {
			var oldest *candidate
			for i := range cands {
				c := &cands[i]
				if c.anon {
					continue
				}
				if oldest == nil || c.arrival.Before(oldest.arrival) {
					oldest = c
				}
			}
			if oldest != nil {
				m.noteServed(oldest.session, now)
				return oldest.queue, nil
			}
		}
	}

	// 1. D-DEMAND: the oldest waiting session past its deadline dispatches
	// now; an uncovered one is covered first (the arc jumps, epoch
	// re-anchors — the bound is on dispatch, not coverage).
	{
		var pressed *candidate
		for i := range cands {
			c := &cands[i]
			if c.anon || c.slack > 0 {
				continue
			}
			if pressed == nil || c.arrival.Before(pressed.arrival) {
				pressed = c
			}
		}
		// The demand path requires a CONFIGURED deadline (D > 0): with D
		// parked the arc moves ONLY by the heartbeat (the freeze rule) —
		// the elastic stretch must not self-generate a deadline (the
		// 2026-10-25 finding: dDemand dominated 1139:1 at c256 with D=0).
		if pressed != nil && pressed.d > 0 {
			// The population tier: a deadline-pressed off-ring session
			// is admitted to the ring first (the budget does not bound D).
			if pressed.offRing {
				delete(m.offRing, pressed.session)
				m.counters.ringAdmissions++
			}
			// Q2: capture the quota clock so the re-anchored epoch continues it.
			var savedDispatches map[string]int64
			var savedTotal int64
			if m.ep != nil {
				savedDispatches = m.ep.dispatches
				savedTotal = m.ep.total
			}
			if s := m.sessions[pressed.session]; s != nil && !s.member {
				m.coverTo(pressed.session, now)
				// Q2: the D-cover must not reset the quota clock — carry
				// the in-flight epoch's counters into the re-anchored epoch.
				if savedDispatches != nil && m.ep != nil {
					for k, v := range savedDispatches {
						m.ep.dispatches[k] = v
					}
					m.ep.total = savedTotal
				}
			} else if m.ep != nil {
				// re-anchor the cursor at the pressed session
				for i, mid := range m.ep.members {
					if mid == pressed.session {
						m.ep.cursor = i
						break
					}
				}
			}
			m.noteServed(pressed.session, now)
			return pressed.queue, nil
		}
	}

	// Admission: the population tier admits oldest-first when the
	// budget allows (off-ring waiters never dispatch before admission).
	m.admitOffRing(waiting, now)

	// 2. EPOCH SWEEP — a served-anchored state machine (B2/M1/M3):
	// the sweep loops; an epoch completes ONLY when its cursor is
	// exhausted AND it served at least one turn; the advance happens only
	// then, and the sweep RE-RUNS (a fresh epoch serves immediately — no
	// per-Pick rotation, no out-of-order fallback).
	for range 64 { // a Pick serves at most one turn; the loop bounds epoch boundaries per Pick
		if m.ep == nil {
			m.startEpoch(now)
		}
		if m.cfg.EpochPolicy == "quota" {
			// QUOTA: the epoch is exactly W dispatches; the selection
			// prefers members not yet dispatched this epoch, then the
			// least-dispatched (deficit round robin over the arc).
			target := pickQuotaMember(m, waiting)
			if target != "" {
				for i := range cands {
					if cands[i].session == target && !cands[i].offRing {
						m.ep.dispatches[target]++
						m.ep.total++
						if m.ep.total >= int64(m.width) {
							// Quota complete: epoch boundary.
							if anyUncoveredWaiting(m, waiting) {
								m.advance("heartbeat", now)
							}
							m.ep = nil
						}
						m.noteServed(target, now)
						return cands[i].queue, nil
					}
				}
			}
			// No member dispatchable this Pick (quota mid-flight, nobody
			// waiting inside): fall to the fork below.
			break
		}
		for m.ep.cursor < len(m.ep.members) {
			id := m.ep.members[m.ep.cursor]
			m.ep.cursor++
			s := m.sessions[id]
			if s == nil || !waiting[id] {
				continue // not waiting: skipped (the sweep's unit is cursor motion)
			}
			if _, parked := m.offRing[id]; parked {
				continue // R2: off-ring sessions never dispatch (any path)
			}
			m.ep.waited[id] = true
			for i := range cands {
				if cands[i].session == id {
					m.noteServed(id, now)
					return cands[i].queue, nil
				}
			}
		}
		// Cursor exhausted. B2: an epoch that served nothing does NOT
		// complete the heartbeat — restart it and stop (the freeze rule:
		// no advance without service).
		if !m.anyoneServedThisEpoch() {
			m.ep = nil
			break
		}
		// The epoch completed with service. Reset it, then advance only
		// if an uncovered session waits (the freeze rule) — and RE-RUN
		// the sweep with the fresh epoch (no fallback path).
		m.ep = nil
		if anyUncoveredWaiting(m, waiting) {
			m.advance("heartbeat", now)
			continue
		}
		break
	}

	// 3. THE FORK: no member dispatched by the sweep (no member waiting).
	anyMemberWaiting := false
	for id := range waiting {
		if s := m.sessions[id]; s != nil && s.member {
			anyMemberWaiting = true
			break
		}
	}
	if anyMemberWaiting {
		// The sweep will reach them next Pick (the epoch state carries).
		return nil, nil //nolint:nilnil
	}
	if m.cfg.ServeUncovered {
		// The hybrid: the longest-waiting uncovered session dispatches
		// without moving the arc (v0-ring's guest path). Off-ring
		// sessions never dispatch (the population tier bounds them).
		var oldest *candidate
		for i := range cands {
			c := &cands[i]
			if c.anon || c.offRing {
				continue
			}
			if s := m.sessions[c.session]; s != nil && s.member {
				continue
			}
			if oldest == nil || c.arrival.Before(oldest.arrival) {
				oldest = c
			}
		}
		if oldest != nil {
			m.counters.uncoveredServed++
			m.noteServed(oldest.session, now)
			return oldest.queue, nil
		}
	}
	// The pure arc: the engine waits for the arc. Anon traffic still
	// dispatches (it is outside the session world).
	for i := range cands {
		if cands[i].anon {
			return cands[i].queue, nil
		}
	}
	return nil, nil //nolint:nilnil
}

// anyoneServedThisEpoch reports whether the current epoch served at least
// one turn (B2: the heartbeat's service anchor).
func (m *manager) anyoneServedThisEpoch() bool {
	return m.ep != nil && len(m.ep.waited) > 0
}

// inflightCountLocked returns the number of sessions with in-flight turns
// (the concurrency denominator for the elastic floor, B3).
func (m *manager) inflightCountLocked() int64 {
	var n int64
	for _, s := range m.sessions {
		if s.inflight > 0 {
			n++
		}
	}
	return n
}

// noteServed records a dispatch for the gauges (and the
// covered-never-served detector's service signal).
func (m *manager) noteServed(id string, now time.Time) {
	if s, ok := m.sessions[id]; ok {
		s.lastSeen = now
		s.turns++
	}
	m.counters.served++
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// pickQuotaMember selects the next member to serve under the quota
// policy: unserved-this-epoch first (tier 1), then least-dispatched
// (tier 2), ring order breaking ties. Only waiting, on-ring members.
func pickQuotaMember(m *manager, waiting map[string]bool) string {
	cov := m.arcMembers()
	best, bestCount := "", int64(-1)
	for _, s := range cov {
		if !waiting[s.id] {
			continue
		}
		if _, parked := m.offRing[s.id]; parked {
			continue // R2: parked members are not quota targets
		}
		if s2, ok := m.sessions[s.id]; !ok || s2 == nil {
			continue
		}
		n := m.ep.dispatches[s.id]
		if best == "" || n < bestCount {
			best, bestCount = s.id, n
		}
	}
	return best
}

// anyUncoveredWaiting reports whether an on-ring, uncovered session waits.
func anyUncoveredWaiting(m *manager, waiting map[string]bool) bool {
	for id := range waiting {
		if _, parked := m.offRing[id]; parked {
			continue
		}
		if s := m.sessions[id]; s != nil && !s.member {
			return true
		}
	}
	return false
}
