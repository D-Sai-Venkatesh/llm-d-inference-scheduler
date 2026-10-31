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
//	F5′ entryOrder:  hrrn-bounded         HRRN whose DENOMINATOR is capped at the live
//	                                     MEDIAN waiter size (bounded far tail)
//	F6″ victimOrder:    residue             evict the COLDEST eligible member first
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

import (
	"sort"
	"time"
)

// prefillRateMinSamples is how many completed prefill observations an engine needs
// before its measured rate is used at all. Below it the scheduler has no honest
// rate and the HRRN ordering degrades to newcomer-first (the anchor's behaviour)
// rather than dividing by a guess.
const prefillRateMinSamples = 3

// ---- the two tenure quantities: the rule's BASE and the hard CEILING ----
//
// THE CEILING IS A CAP, NEVER AN OVERRIDE (the B1b precedence bug, ruled 2026-10-31).
// When this code returned `quantumCeiling` as THE bound for any rule that had one, the
// rule's own base quantum never applied: F3 (dyn-size) released every member at the
// ceiling (dyn-size = Q8 exactly) and F7's "rotate at Q4 while prefill is free" half
// never ran either (F7 ≈ Q8 + a minor idle-hold). Both arms then read as Q8 replicates
// — an uninterpretable arm, not a wrong one. The two quantities are therefore separate:
//
//   base    — the tenure THIS rule releases a member at (the fixed Q, the footprint-
//             scaled Q_i, the pool's Q_s/Q_l, PAT's fair share Q_fair).
//   ceiling — the hard bound nobody may exceed (Config.QuantumCeiling; for PAT the
//             measured-safe knee when unset). 0 = none.
//
// and the eligibility bound is min(base, ceiling).

// eligibilityBaseLocked is the tenure the member's OWN eligibility rule releases at:
// the fixed Q, the dyn-size Q_i (F3), the pool quantum (F9) or PAT's fair share. It
// answers "when does this rule fire?", which is a different question from
// tenureCeilingLocked's "when must it fire anyway?".
func (s *Scheduler) eligibilityBaseLocked(g *engineGroup, m *member) (int, bool) {
	if s.cfg.patOn() {
		// PAT's rule (a) fires at the FAIR SHARE (Q_fair = the quantum); its rule (b) is
		// the ceiling and lives in tenureCeilingLocked. Reporting the fair share here is
		// what makes "nothing is released before fair share" checkable from one number.
		if q, bounded := s.patFairShare(); bounded {
			return q, true
		}
		return 0, false
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
		// long pool: Q_l is the slice; a member still in the SHORT pool is bounded by
		// Q_s = 1 busy period (promotion, not eviction).
		if s.mlffqLongLocked(m) {
			return s.longQuantum(), true
		}
		return 1, true
	}
	return 0, false // the ledger model does not use a quantum
}

// tenureCeilingLocked is the HARD bound in completed busy periods, 0 = none. It is what
// F7's hold-gate may not block and what PAT's rule (b) fires on — the fairness guarantee
// of every dynamic rule, measured against the ceiling ITSELF and never against a
// min() bound (otherwise a member capped down to 3 would "reach the ceiling" at 3).
func (s *Scheduler) tenureCeilingLocked() int {
	if s.cfg.patOn() {
		return s.patCeiling() // explicit quantumCeiling, else the measured 8-knee
	}
	if s.cfg.QuantumCeiling > 0 {
		return s.cfg.QuantumCeiling
	}
	return 0
}

// quantumBoundLocked is the member's ELIGIBILITY bound in completed busy periods, or
// (bound=false) "unbounded": min(base, ceiling). The ceiling caps the base; it never
// replaces it — that distinction is the whole fix.
func (s *Scheduler) quantumBoundLocked(g *engineGroup, m *member) (int, bool) {
	base, bounded := s.eligibilityBaseLocked(g, m)
	ceiling := s.tenureCeilingLocked()
	if ceiling <= 0 {
		return base, bounded
	}
	if !bounded {
		return ceiling, true // an unbounded rule is bounded by the ceiling alone
	}
	if ceiling < base {
		return ceiling, true // the cap binds
	}
	return base, true
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

// uncachedTokensLocked is the request's ESTIMATED UNCACHED prompt work against ONE
// engine, in tokens: the tier-A keys this engine does not already hold, times the
// block size. It is the policy's OWN accounting (dedup against the engine's index,
// which is exactly what the NE-counter delta already computes) — no cache read.
//
// WHY NOT prompt tokens: prompt tokens make F7 degenerate. A warm seated turn is
// 50-100k tokens at ~0.9 hit, so ONE such turn in prefill would exceed an 8,192-token
// step budget 6-12x while its real prefill work is a few k — the gate would sit closed
// almost permanently and F7 would be a fixed-Q arm (P∞ + ceiling) rather than a
// load-adaptive one.
func (s *Scheduler) uncachedTokensLocked(g *engineGroup, keys []Hash) int64 {
	if g == nil || len(keys) == 0 {
		return 0
	}
	var missing int64
	for _, h := range keys {
		if !g.index.HasNELocked(h) {
			missing++
		}
	}
	return missing * int64(s.blockSize)
}

// ---- F5 / F9: HRRN over the entry head ----

// entryPriorityLocked is F5's priority: the HRRN response ratio
//
//	(wait + s) / s
//
// with s the entry's estimated prefill SECONDS. The engine's measured prefill rate
// is COMMON to every waiter at one decision instant, so it CANCELS out of the
// ordering — (wait + cold/rate)/(cold/rate) is a monotone function of wait/cold — so
// the key is computed directly as
//
//	wait / cold_tokens   (descending)
//
// which deletes the rate estimator, its EWMA constant, its warm-up period and its
// fallback ordering: the rule is active from the first decision. This is the "equal
// slowdown" criterion (waits proportional to entry size) — shortest-job-first that
// cannot starve, because wait grows without bound while cold_tokens is fixed.
//
// The denominator is floored at one block so a zero-cold head is not a division by
// zero; ties (wait 0, or equal ratios) fall through to the age/id tie-breaks of the
// D1 key.
func entryPriority(waitSeconds float64, coldTokens int64, blockSize int) float64 {
	den := coldTokens
	if den <= 0 {
		den = int64(blockSize)
	}
	if waitSeconds < 0 {
		waitSeconds = 0
	}
	return waitSeconds / float64(den)
}

// ---- F5′: bounded HRRN (entryOrder "hrrn-bounded") ----
//
// THE MEASURED DEFECT OF PLAIN HRRN (B1b, 2026-11-01). F5 won the throughput race
// (+25% tok/s, p10 +16%, the best >600 s fraction, newcomer waits DOWN) but its far
// tail was broken: max entry wait 3,601 s and 20 stalled sessions. The mechanism is
// arithmetic, not a tuning problem — wait/cold_tokens ages a 200K-token entry ~67x
// slower than a 3K one, so the big entry's takeover bound is
//
//	small_ratio x cold_big      (hours, at the measured ratios)
//
// i.e. starvation-free in theory and an hour in practice.
//
// THE FIX (parameter-free). Replace the entry's OWN size in the denominator with
//
//	s_eff = min(cold_tokens, the live MEDIAN cold_tokens of the current waiters)
//
// Nothing else changes: no time constant, no threshold, no new state. The median is
// the scheduler's OWN accounting over the waiter pool it already tracks (never a
// cache read), recomputed once per decision pass, so the rule stays
// dictate-don't-observe. What it buys:
//
//   - SMALL ENTRIES ARE UNTOUCHED. The even-set median is taken as the LOWER median,
//     so an entry below it keeps its own size and small-first ordering inside the
//     small class is exactly F5's.
//   - LARGE ENTRIES AGE AT THE MEDIAN RATE, so their takeover bound becomes
//     small_ratio x median instead of small_ratio x cold_big. At B1b's ratios that is
//     ~600 s instead of ~40,000 s — bounded, and derived from the queue itself rather
//     than from a constant somebody has to fit.
//   - THE WAITERS SELF-REGULATE. If the queue fills with big entries the median rises
//     and they age slowly TOGETHER (equal slowdown, oldest first among them); if it is
//     mostly small entries the big one is served at the small class's rate. The bound
//     is the population's own size, not a policy choice.
//
// The plain "hrrn" stays available unchanged: it is the comparison arm.

// entryColdBoundLocked is the bound one decision pass uses: the live MEDIAN cold-token
// size of its waiter set, or 0 when the ordering rule is not hrrn-bounded. 0 means
// unbounded, so plain HRRN and every non-HRRN order are bit-for-bit unaffected.
func (s *Scheduler) entryColdBoundLocked(waiters map[string]*waiter) int64 {
	if s.cfg.EntryOrder != "hrrn-bounded" {
		return 0
	}
	return medianColdTokens(waiters, s.blockSize)
}

// medianColdTokens is the LOWER median of the waiter set's cold-token sizes, each
// floored at one block exactly as entryPriority floors its denominator (so a fully
// WARM waiter — 0 uncached tokens — contributes the smallest possible size instead of
// collapsing the median to zero and silently restoring unbounded HRRN). The lower
// median is deliberate: every size BELOW it is then strictly below it, so
// min(cold, median) can only bind on entries AT or ABOVE the median, and the small
// class's ordering is provably untouched. 0 when nothing is waiting.
func medianColdTokens(waiters map[string]*waiter, blockSize int) int64 {
	if len(waiters) == 0 {
		return 0
	}
	sizes := make([]int64, 0, len(waiters))
	for _, w := range waiters {
		n := w.coldTokens
		if n <= 0 {
			n = int64(blockSize)
		}
		sizes = append(sizes, n)
	}
	sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })
	return sizes[(len(sizes)-1)/2]
}

// boundedEntryPriority is entryPriority with the denominator capped at the pass's
// bound: min(coldTokens, coldBound) holds a big job's age rate at the median waiter's,
// so its priority grows at the MEDIAN rate instead of its own (much slower) one.
// coldBound 0 (unbounded, or no bound computed for this pass) makes it EXACTLY
// entryPriority — that equality is what keeps every other path pinned.
func boundedEntryPriority(waitSeconds float64, coldTokens int64, coldBound int64, blockSize int) float64 {
	den := coldTokens
	if coldBound > 0 && coldBound < den {
		den = coldBound
	}
	return entryPriority(waitSeconds, den, blockSize)
}

// nowLocked is the current decision pass's clock (never time.Now — a golden replay
// must be reproducible).
func (s *Scheduler) nowLocked() time.Time { return s.now }

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
// eligibilityMode "pat" substitutes PAT's rule (pat.go) for the whole model.
func (s *Scheduler) eligibleLocked(g *engineGroup, m *member, now time.Time) (string, bool) {
	if m.inflight > 0 {
		return "", false
	}
	if s.cfg.patOn() {
		// PAT (pat.go) replaces the whole eligibility model — not the victim order, and
		// not anything downstream of it.
		return s.patEligibleLocked(g, m)
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

// ceilingReachedLocked reports whether a member has hit the tenure CEILING itself — the
// one eviction F7's gate may not block (nobody starves). It compares against the ceiling
// and not against the eligibility bound: a member whose base was capped down must still
// be able to run to the ceiling when the gate holds the base back.
//
// With NO ceiling configured there is nothing to escape to, so the rule's own base is
// the only bound there is — which keeps F7-without-a-ceiling exactly a fixed-Q arm
// instead of an unbounded hold (the starvation the ceiling exists to prevent).
func (s *Scheduler) ceilingReachedLocked(g *engineGroup, m *member) bool {
	if c := s.tenureCeilingLocked(); c > 0 {
		return m.busyPeriods >= c
	}
	q, bounded := s.eligibilityBaseLocked(g, m)
	return bounded && m.busyPeriods >= q
}
