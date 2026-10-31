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

// THE F-ARMS (single pod): five rotation-policy knobs, each one INDEPENDENT and
// each defaulting to today's behaviour (so the golden replay is untouched):
//
//	F3  quantumMode: fixed|dyn-size      per-member quantum scaled by footprint
//	F5  entryOrder:  newcomer-first|hrrn  HRRN over the entry head (D1)
//	F6′ residueRelease: off|on            release a seat whose blocks are gone
//	F7  rotateWhenFree                    rotate only while prefill is not saturated
//	F9  rotationMode: mlfq2 (+longQuantum) two pools, HRRN arbitration
//
// and ONE cross-cutting guarantee they all share:
//
//	quantumCeiling  the tenure bound (in completed busy periods) of every DYNAMIC
//	                rule. A dynamic rule without a ceiling can starve somebody;
//	                F6/F7/F9 all lean on it. 0 = off, and off is the pinned default.
//
// Everything here is derived from state the scheduler already has (busy periods,
// block footprints, dispatch and first-token times, own accounting). No wall
// clock, no cache read, no fitted constant: the two rates are MEASURED from the
// run (prefill tokens per second from dispatch→first-token, the population median
// footprint from the seated set).

import "time"

// prefillRateMinSamples is how many completed prefill observations an engine needs
// before its measured rate is used at all. Below it the scheduler has no honest
// rate and the HRRN ordering degrades to newcomer-first (the anchor's behaviour)
// rather than dividing by a guess.
const prefillRateMinSamples = 3

// quantumBoundLocked is a member's tenure bound in completed busy periods, or
// (bound=false) "unbounded". Precedence:
//
//  1. quantumCeiling > 0 — the FAIRNESS GUARANTEE of every dynamic rule: whatever
//     else is going on, nobody holds a seat longer than this many busy periods.
//  2. rotationMode != quantum / mlfq2 — the ledger model does not use a quantum.
//  3. quantumMode dyn-size — Q_i = max(1, round(Q̄ · blocks_i / median blocks)).
//     Q̄ is the anchor's own Q (so the AVERAGE scale matches and only the SIZE
//     effect varies): a big member gets a longer slice, a small one a shorter one.
//  4. otherwise the fixed Q.
func (s *Scheduler) quantumBoundLocked(g *engineGroup, m *member) (int, bool) {
	if s.cfg.QuantumCeiling > 0 {
		return s.cfg.QuantumCeiling, true
	}
	switch s.cfg.RotationMode {
	case "quantum":
		if s.cfg.QuantumMode == "dyn-size" {
			if !s.cfg.quantumFinite() {
				return 0, false
			}
			return s.dynQuantumLocked(g, m), true
		}
		q := s.cfg.quantum()
		return q, q >= 0
	case "mlfq2":
		// long pool: Q_l is the tenure bound; a member still in the SHORT pool is
		// bounded by Q_s = 1 busy period (promotion, not eviction).
		if s.mlffqLongLocked(m) {
			return s.longQuantum(), true
		}
		return 1, true
	}
	return 0, false
}

// dynQuantumLocked is F3: the member's quantum scaled by its FOOTPRINT relative to
// the median seated footprint on that engine.
func (s *Scheduler) dynQuantumLocked(g *engineGroup, m *member) int {
	qbar := s.cfg.quantum()
	if qbar <= 0 {
		return qbar // ∞ stays ∞
	}
	median := s.medianFootprintLocked(g)
	if median <= 0 {
		return qbar // no population yet: the anchor's Q
	}
	mine := int64(len(g.index.SetBlocks(m.id)))
	if mine <= 0 {
		return 1
	}
	q := int(float64(qbar) * float64(mine) / float64(median))
	if q < 1 {
		q = 1
	}
	return q
}

// medianFootprintLocked is the median block footprint over the engine's seated,
// non-exiting members (the population the quantum is relative to).
func (s *Scheduler) medianFootprintLocked(g *engineGroup) int64 {
	if g == nil {
		return 0
	}
	sizes := make([]int64, 0, len(g.members))
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) {
			continue
		}
		sizes = append(sizes, int64(len(g.index.SetBlocks(m.id))))
	}
	if len(sizes) == 0 {
		return 0
	}
	for i := 1; i < len(sizes); i++ { // insertion sort; the seated set is small
		for j := i; j > 0 && sizes[j] < sizes[j-1]; j-- {
			sizes[j], sizes[j-1] = sizes[j-1], sizes[j]
		}
	}
	n := len(sizes)
	if n%2 == 1 {
		return sizes[n/2]
	}
	return (sizes[n/2-1] + sizes[n/2]) / 2
}

// longQuantum is Q_l (F9): the long pool's slice length. Default 8 (the B1a
// winner will set the value from the config).
func (s *Scheduler) longQuantum() int {
	if s.cfg.LongQuantum > 0 {
		return s.cfg.LongQuantum
	}
	return 8
}

// mlffqLongLocked reports whether a member is in the LONG pool: it has already
// outlived its first busy period (the only promotion rule — "did it outlive one
// busy period?" — a natural event, not a constant).
func (s *Scheduler) mlffqLongLocked(m *member) bool {
	if m == nil {
		return false
	}
	if m.busyPeriods >= 1 {
		return true
	}
	return s.promoted[m.id] // survives rotation/exit: the class is a property of the member
}

// ---- F5 / F9: HRRN over the entry head ----

// prefillRateLocked is the engine's MEASURED prefill rate in tokens/second, from
// the running observations the scheduler already collects: each tracked request is
// timed from its dispatch (PreRequest) to its FIRST TOKEN (ResponseBody's
// StartOfStream), and paired at settle with that request's uncached prompt tokens.
// EWMA (alpha 0.2) so a slow first observation does not pin the estimate.
//
// The confound is stated where it belongs: dispatch→first-token includes whatever
// engine-side queueing the request suffered, so under load the estimate is a
// conservative (lower) rate — which makes s LARGER for everybody, i.e. it changes
// the SHAPE of the HRRN ratio less than it changes its scale.
func (s *Scheduler) prefillRateLocked(g *engineGroup) (float64, bool) {
	if g == nil || g.prefillSamples < prefillRateMinSamples || g.prefillRate <= 0 {
		return 0, false
	}
	return g.prefillRate, true
}

// notePrefillLocked books one prefill observation: seconds from dispatch to first
// token, and the uncached prompt tokens the engine actually had to prefill.
func (s *Scheduler) notePrefillLocked(g *engineGroup, seconds float64, uncachedTokens int64) {
	if g == nil || seconds <= 0 || uncachedTokens <= 0 {
		return
	}
	rate := float64(uncachedTokens) / seconds
	g.prefillSamples++
	if g.prefillSamples == 1 {
		g.prefillRate = rate
		return
	}
	g.prefillRate = 0.8*g.prefillRate + 0.2*rate
}

// estimatedPrefillSeconds is s in the HRRN ratio: the head's cold tokens over the
// MEASURED rate. Returns (0,false) when no honest rate exists yet.
func (s *Scheduler) estimatedPrefillSeconds(g *engineGroup, w *waiter) (float64, bool) {
	rate, ok := s.prefillRateLocked(g)
	if !ok {
		return 0, false
	}
	if w.coldTokens <= 0 {
		return 0, true // nothing to prefill: an infinite ratio, handled by the caller
	}
	return float64(w.coldTokens) / rate, true
}

// nowLocked is the current decision pass's clock (never time.Now — a golden replay
// must be reproducible).
func (s *Scheduler) nowLocked() time.Time { return s.now }

// hrrn returns the response ratio (wait + s) / s: the LONGER a head waits, the
// higher its priority, and the SMALLER its estimated service, the higher its
// priority. That is shortest-job-first that cannot starve: wait grows without
// bound, so every head eventually outranks every other.
func hrrn(waitSeconds, serviceSeconds float64) float64 {
	if serviceSeconds <= 0 {
		return 1e18 // no prefill work: maximally attractive, still a finite ORDER
	}
	return (waitSeconds + serviceSeconds) / serviceSeconds
}

// ---- F6′: residue-driven seat release ----

// residueReleaseVictimLocked is F6′: when waiters exist, a SEATED IDLE member whose
// predicted surviving fraction is below the warm/cold cut holds a seat that is
// protecting nothing — releasing it costs no cache and gives the seat-time to a
// waiter. Among the candidates the longest-idle goes first.
//
// This is the residue model's SINGLE-ENGINE use: the same shadow, read for "are
// this member's blocks still there" instead of "which engine has them".
func (s *Scheduler) residueReleaseVictimLocked(g *engineGroup, now time.Time) (string, *member) {
	if g == nil || !g.residue.usable() {
		return "", nil
	}
	if len(s.waitersLocked(g)) == 0 && len(s.gWaiters) == 0 {
		return "", nil // nobody wants the seat: a release would only lose the cache
	}
	var best *member
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) || m.inflight > 0 {
			continue
		}
		if !m.waitingSince.IsZero() {
			continue // it has a turn queued: it is about to run, not idle
		}
		if g.residue.ratio(m.id) >= residueWarmFraction {
			continue // warm: the seat is protecting a real prefix
		}
		if best == nil || m.lastActivity.Before(best.lastActivity) ||
			(m.lastActivity.Equal(best.lastActivity) && m.id < best.id) {
			best = m
		}
	}
	if best == nil {
		return "", nil
	}
	return "residue-cold", best
}

// ---- F7: rotate only while prefill has room ----

// prefillSaturatedLocked is F7's gate: the engine's EPP-side PREFILL BACKLOG —
// dispatched requests that have not produced a first token yet, in TOKENS — at or
// over the engine's own per-step token budget (max_num_batched_tokens).
//
// Waiting is conserved in a closed loop: rotation only redistributes it. Its real
// cost is the cold re-prefill a rotation buys. So rotate while that prefill costs
// nobody anything, and HOLD when the engine's prefill is already the bottleneck —
// slices lengthen automatically, and quantumCeiling is what stops them lengthening
// forever.
func (s *Scheduler) prefillSaturatedLocked(g *engineGroup) bool {
	if !s.cfg.RotateWhenFree || g == nil {
		return false
	}
	limit := s.cfg.MaxNumBatchedTokens
	if limit <= 0 {
		return false
	}
	return g.prefillBacklog >= int64(limit)
}

// ---- rotation eligibility, all knobs in one place ----

// eligibleLocked is the exit-eligibility model. LEGACY ledger rule (paid / idle /
// nmax) unless rotationMode says quantum or mlfq2, with the C8 fix always-on: a
// member with a live queued head is never idle (its lastActivity cannot be
// refreshed when Picks stop, so gate closure used to manufacture idle exits).
func (s *Scheduler) eligibleLocked(g *engineGroup, m *member, now time.Time) (string, bool) {
	if m.inflight > 0 {
		return "", false
	}
	switch s.cfg.RotationMode {
	case "quantum", "mlfq2":
		if s.cfg.RotationMode == "mlfq2" && !s.mlffqLongLocked(m) {
			// SHORT pool: never rotation-evicted. It either finishes inside its first
			// busy period (the common single-request leaf) or is promoted by
			// completing one — the promotion IS the eligibility.
			return "", false
		}
		if q, bounded := s.quantumBoundLocked(g, m); bounded && m.busyPeriods >= q {
			return "quantum-complete", true
		}
		if s.silentLocked(m, now) {
			return "idle", true
		}
		return "", false
	}
	if m.resTurns >= 1 && m.earned >= int64(s.cfg.Rho)*max(m.entryCost, 1) {
		return "paid", true
	}
	if s.silentLocked(m, now) {
		return "idle", true
	}
	if m.resTurns >= s.cfg.nMax() {
		return "nmax", true
	}
	return "", false
}

// ceilingReachedLocked reports whether a member has hit the tenure ceiling — the
// one eviction F7's gate may not block (nobody starves).
func (s *Scheduler) ceilingReachedLocked(g *engineGroup, m *member) bool {
	q, bounded := s.quantumBoundLocked(g, m)
	return bounded && m.busyPeriods >= q
}
