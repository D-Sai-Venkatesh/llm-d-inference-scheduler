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

package ringwindow

import (
	"context"
	"math"
	"sort"
	"time"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// bytesPerToken is the average number of request bytes per prompt token,
// mirroring upstream requesthandling.BytesPerToken (4); this tree predates
// the constant, so it is local to the port.
const bytesPerToken = 4

// NewState satisfies the FairnessPolicy contract; Pick reads the shared
// ringManager instead of per-band state (the window is one pool).
func (r *RingWindow) NewState(_ context.Context) any { return nil }

// candidate is one dispatchable queue head under consideration.
type candidate struct {
	queue fwkfc.FlowQueueAccessor
	// session is the agent-identity session id; "" is anonymous traffic.
	session string
	// arrival is the head's enqueue time; d is its EffectiveTTL — the
	// queue-wait budget for this request, band and deployment (decision D5).
	arrival time.Time
	d       time.Duration
	// dEff is the elastic deadline: max(d, ŝ·N̂) — when the SLO is
	// infeasible for every waiting turn, the budget stretches to the
	// measured line instead of letting the rescue degenerate into EDF
	// (decision D4).
	dEff time.Duration
	// slack is arrival + dEff − now − ŝ_i: negative means the turn must be
	// dispatched now to finish inside its budget. The −ŝ margin acts one
	// dispatch ahead, because one dispatch costs one slot.
	slack time.Duration
	// est is the session's footprint once this turn is charged (the whole
	// history is resent each turn).
	est float64
	// ŝ is the service prediction for this session: its own last completed
	// turn, or the running mean for a session with no history yet (D10).
	shat     time.Duration
	member   bool
	anon     bool
	waitMs   float64
}

// Pick is the ring-window dispatch order:
//
//  1. ROTATE — a waiting outsider whose slack has expired rotates the window
//     one place: it joins (members are demoted in eviction order until the
//     budget covers it) and is dispatched first in the new window. Members
//     never trigger rotation; this is the only rotation (decision D4).
//  2. MEMBER — a waiting member whose own slack has expired is served first
//     (quiet-member recovery); otherwise members are served in
//     back-and-forth order with a rotating turnaround (decision D1). A
//     member with no turn waiting is skipped and keeps its place.
//  3. OUTSIDER — with no member waiting, the head outsider (earliest in
//     ring order) runs: it joins when it fits both window widths; otherwise
//     it runs anyway (never idle, the weak sense) with members demoted in
//     eviction order until it fits; an oversized candidate that no subset
//     can cover runs as a non-member and the window clears (decision D6).
//
// The flow-control gate stays exactly as deployed: it decides how much runs,
// this policy only decides who. Anonymous traffic passes through last, with
// no ledger effect.
func (r *RingWindow) Pick(_ context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()

	var cands []*candidate
	band.IterateQueues(func(queue fwkfc.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		cands = append(cands, &candidate{
			queue:   queue,
			session: headSessionID(head),
			arrival: head.EnqueueTime(),
			d:       head.EffectiveTTL(),
			waitMs:  float64(now.Sub(head.EnqueueTime()).Milliseconds()),
		})
		return true
	})
	if len(cands) == 0 {
		return nil, nil //nolint:nilnil
	}

	m := r.mgr
	m.mu.Lock()
	defer m.mu.Unlock()

	nWaiting := len(cands)
	meanDur := m.meanTurnDurLocked()

	// Classify and compute slacks.
	waiting := make(map[string]bool, len(cands))
	for _, c := range cands {
		c.anon = c.session == ""
		if c.anon {
			continue
		}
		waiting[c.session] = true
		// Every waiting session enters the ledger at first sight, so its
		// ring position exists before it is ever picked: PreRequest only
		// runs after a dispatch, so first sight is the queue.
		s := m.ensureSessionLocked(c.session)
		estimate := max(estimateTokens(headSizeBytes(c)), 1)
		c.member = s.member
		c.est = max(float64(s.committedTokens), float64(estimate))
		c.shat = s.lastTurnDur
		if c.shat <= 0 {
			c.shat = meanDur
		}
		c.dEff = c.d
		if stretch := meanDur * time.Duration(nWaiting); stretch > c.dEff {
			c.dEff = stretch
		}
		c.slack = c.arrival.Add(c.dEff).Sub(now) - c.shat
	}
	// Deterministic candidate order: ring order. Pick must not depend on
	// queue iteration order.
	sort.Slice(cands, func(i, j int) bool {
		si, sj := m.sessions[cands[i].session], m.sessions[cands[j].session]
		if cands[i].anon {
			si = nil
		}
		if cands[j].anon {
			sj = nil
		}
		if si == nil || sj == nil {
			return si != nil
		}
		return si.seq < sj.seq
	})

	// 1. ROTATE: the least-slack waiting outsider whose D-clock has expired.
	var rot *candidate
	for _, c := range cands {
		if c.anon || c.member {
			continue
		}
		if rot == nil || c.slack < rot.slack {
			rot = c
		}
	}
	if rot != nil && rot.slack <= 0 {
		joined, oversized := m.rotate(rot.session, rot.est, rot.dEff, waiting, now)
		m.reserveLocked(rot.session, rot.est, now)
		r.metrics.pickRotations.Inc()
		if oversized {
			r.metrics.pickOversized.Inc()
		}
		_ = joined
		return rot.queue, nil
	}

	// 2. MEMBER: recovery first, then the back-and-forth cursor.
	var memberCand []*candidate
	for _, c := range cands {
		if !c.anon && c.member {
			memberCand = append(memberCand, c)
		}
	}
	if len(memberCand) > 0 {
		var rec *candidate
		for _, c := range memberCand {
			if c.slack <= 0 && (rec == nil || c.slack < rec.slack) {
				rec = c
			}
		}
		var pick *session
		if rec != nil {
			// Quiet-member recovery: serve the most urgent member first.
			pick = m.sessions[rec.session]
			r.metrics.pickRecoveries.Inc()
		} else {
			pick = m.pickNextMemberLocked(waiting)
		}
		if pick != nil {
			for _, c := range memberCand {
				if c.session == pick.id {
					m.serveMemberLocked(pick)
					m.reserveLocked(pick.id, c.est, now)
					r.metrics.pickMemberServes.Inc()
					if c.waitMs >= 1000 {
						r.metrics.pickDelayedDispatches.Inc()
					}
					return c.queue, nil
				}
			}
		}
	}

	// 3. OUTSIDER (no member waiting): the head outsider, never idle.
	if head := m.headOutsiderLocked(nonMemberIDs(cands)); head != nil {
		var hc *candidate
		for _, c := range cands {
			if c.session == head.id {
				hc = c
				break
			}
		}
		if hc != nil {
			if m.fitsLocked(hc.est, hc.dEff, now) {
				m.joinQuiet(hc.session, hc.est, hc.dEff, now)
				m.reserveLocked(hc.session, hc.est, now)
				r.metrics.pickJoins.Inc()
			} else {
				_, oversized := m.guest(hc.session, hc.est, hc.dEff, waiting, now)
				m.reserveLocked(hc.session, hc.est, now)
				r.metrics.pickGuests.Inc()
				if oversized {
					r.metrics.pickOversized.Inc()
				}
			}
			return hc.queue, nil
		}
	}

	// Anonymous traffic passes through last, with no ledger effect.
	for _, c := range cands {
		if c.anon {
			r.metrics.pickAnonymous.Inc()
			return c.queue, nil
		}
	}
	return nil, nil //nolint:nilnil
}

// nonMemberIDs maps the waiting non-member (and non-anonymous) session ids.
func nonMemberIDs(cands []*candidate) map[string]bool {
	out := make(map[string]bool)
	for _, c := range cands {
		if !c.anon && !c.member {
			out[c.session] = true
		}
	}
	return out
}

// estimateTokens converts a request body size to a token estimate. The body
// also holds the JSON envelope, tool schemas and images, so the estimate is
// above the prompt token count.
// TODO(#3221): use the request's token count once it is available before flow
// control.
func estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 0
	}
	return int64(sizeBytes / bytesPerToken)
}

// headSizeBytes returns the request body size of a queue head, from the
// scheduling request when the item carries one, else from the item's byte
// size.
func headSizeBytes(c *candidate) int {
	req := c.queue.Peek().OriginalRequest()
	if req == nil {
		return 0
	}
	if ir := req.InferenceRequest(); ir != nil && ir.RequestSizeBytes > 0 {
		return ir.RequestSizeBytes
	}
	if n := req.ByteSize(); n <= math.MaxInt {
		return int(n)
	}
	return math.MaxInt
}
