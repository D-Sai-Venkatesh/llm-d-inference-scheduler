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

// PAT — PAUSE-ALIGNED TENURE (eligibilityMode "pat"; the reviewer's B2 adaptive
// candidate).
//
// THE RULE. A member is rotation-eligible iff
//
//	(a) it has completed its FAIR SHARE (Q_fair busy periods — the standard quantum,
//	    4 by default) AND it is IDLE (no queued turn, nothing in flight) AND its
//	    residue class is COLD, or
//	(b) it reaches the safety CEILING (quantumCeiling; 8 busy periods — the measured
//	    B1a knee — when unset).
//
// The victim order among the eligible is F6″'s (coldest class first, then raw ratio,
// then oldest last completion, then id) — see colderVictim. Everything else in the
// policy is untouched: the entry order (D1), the cold-entry valve, the budget ladder,
// the exit accounting and the per-engine groups are the same code as the anchor.
//
// WHY IT IS PART OF THE SAME FRONTIER AS THE FIXED QUANTUM. B1a measured the fixed-Q
// frontier: Q4 balances, Q8 is the knee, Q16 fails (tenures run past the useful point
// into tail starvation: a starved thread, 68 requests unserved >20 min). A fixed Q is
// the wrong instrument for that job because the useful tenure is not a property of the
// SCHEDULER — it is a property of the CACHE. What makes a long tenure expensive is
// that the member's blocks are gone by the time it returns: the pause has outlived the
// cache, so the saved cold re-prefill is a fiction and the seat was simply idle.
//
// THE RESIDUE MODEL IS THE MEASUREMENT (residue.go, the E4b per-engine shadow of
// vLLM v1's free-block queue). It answers exactly that question in the scheduler's own
// accounting: how much of this member's chain is still resident on this engine, given
// every allocation and free that has happened since. It has no clock, no threshold and
// no cache read — the decay curve measured offline comes out of the pool size and the
// allocation counts (knee at ≈4-8× the chain's own size, 15-40 s of idle on the
// campaign engine). So PAT's "is the pause cache-dead?" test INHERITS that
// parameter-free property, which is the whole point of the exercise (the τ=120
// objection: no magic constant).
//
// WHAT IT BUYS, stated as the frozen prediction: a member that pauses past its residue
// horizon returns COLD under both PAT and Q8, so the cold tokens are the same either
// way; the difference is that under Q8 the member held the seat IDLE through the pause
// while a waiter queued, whereas PAT hands that seat to the waiter at the cache-dead
// moment — utilisation at ZERO cache cost — and rotates at a moment when the eviction
// costs nothing instead of at the 8th busy period (which may well be WARM). The cost
// it pays, also frozen: a released member's RETURN turn gets an entry wait instead of
// an instant seat. That TTFT is the watched number.
//
// FAIRNESS (the human's point, and the fair version of the removed F6′): nobody is
// released before its fair share. The residue class alone never evicts — it only
// decides WHEN a member that has already had its slice leaves. And the ceiling bounds
// the other direction, so PAT can never hold a seat longer than Q8 does.
//
// The one exception is the one every arm shares: the §6.3 BUDGET valve
// (budgetVictimLocked) still evicts idle members when the engine's blocks exceed 95 % of
// its HBM — that is an OOM backstop, not a rotation, and it is byte-identical in the Q4,
// Q8 and PAT arms, so it cannot explain a difference between them. PAT's own rule acts
// only through rotationVictimLocked.

const (
	// reasonPatFairShareCold is PAT's early-exit reason: fair share served, the member
	// is idle and its pause has outlived its cache.
	reasonPatFairShareCold = "pat-fair-share-cold"
	// reasonPatCeiling is PAT's safety-ceiling reason (b).
	reasonPatCeiling = "pat-ceiling"

	// patCeilingBusyPeriods is PAT's ceiling when quantumCeiling is unset: 8 busy
	// periods, the MEASURED-safe knee (B1a f-arms: Q8 balances, Q16 starves). It is
	// deliberately the same bound f1p runs, so "PAT never holds a seat longer than the
	// frontier point it is read against" is arithmetic, not a hope.
	patCeilingBusyPeriods = 8
)

// patFairShare is Q_fair. It is the QUANTUM (Config.quantum()) and not a constant of
// its own: PAT's fairness unit has to be the anchor's slice, or "fair share" would be
// a second, private definition of the same idea. bounded=false is the P∞ arm
// (quantumBusyPeriods = -1): nobody is ever fair-share complete, so only the ceiling
// can fire — which is exactly the meaning that arm's Q already carries.
func (s *Scheduler) patFairShare() (q int, bounded bool) {
	q = s.cfg.quantum()
	return q, q >= 0
}

// patCeiling is PAT's safety ceiling in completed busy periods: explicit
// quantumCeiling when set, else the measured knee.
func (s *Scheduler) patCeiling() int {
	if s.cfg.QuantumCeiling > 0 {
		return s.cfg.QuantumCeiling
	}
	return patCeilingBusyPeriods
}

// patEligibleLocked is PAT's exit-eligibility model for ONE member. The caller
// (eligibleLocked) has already required m.inflight == 0.
//
// It takes no clock on purpose: neither half of the rule has a time constant. (a) asks
// whether the member is USING the seat now and whether its cache is still there — both
// are state, and the second one is the residue model's own accounting. The legacy
// "silent for more than idleSeconds" test is deliberately NOT part of PAT: that test
// released members at 20 s whether or not their cache had survived, which is the
// rotate-at-a-warm-moment behaviour this arm exists to remove.
func (s *Scheduler) patEligibleLocked(g *engineGroup, m *member) (string, bool) {
	// (b) THE SAFETY CEILING. Checked first and unconditionally (class and idleness do
	// not matter): it is the guarantee that stops "hold, the cache is still warm" from
	// becoming starvation, and it is the exact bound f1p runs. It compares against the
	// CEILING itself (tenureCeilingLocked), never against a min() eligibility bound.
	if c := s.tenureCeilingLocked(); c > 0 && m.busyPeriods >= c {
		s.notePatEligible(g, m, reasonPatCeiling)
		return reasonPatCeiling, true
	}
	// (a) FAIR SHARE + PAUSE-ALIGNED EXIT.
	q, bounded := s.patFairShare()
	if !bounded || m.busyPeriods < q || !patIdleLocked(m) || !s.patColdLocked(g, m.id) {
		m.patReason = "" // not eligible now: a later cold pause is a NEW event
		return "", false
	}
	s.notePatEligible(g, m, reasonPatFairShareCold)
	return reasonPatFairShareCold, true
}

// patIdleLocked is PAT's "currently idle": nothing in flight (the caller guarantees
// it) and NO QUEUED TURN. A member with a queued head WANTS to run — rotating it would
// throw away work it is about to need (the C8 rule, unchanged).
func patIdleLocked(m *member) bool { return m.waitingSince.IsZero() }

// patColdLocked is PAT's COLD class: the residue model's own BIMODAL CUT, on the
// member's chain on THIS engine.
//
// It is a CLASS, not a score (reviewer R3): the measured distribution is bimodal
// (warm ≈ 0.98, cold ≈ 0.12-0.17), so any cut in the gap gives the same verdict and
// residueWarmFraction is the point furthest from both modes. It is derived from
// ALLOCATIONS — the FIFO of block identities that the scheduler itself drove — so it
// has no clock, no fitted threshold and no cache read.
//
// An engine with no usable model (residue off, or disabled after sustained
// mis-calibration) has NO honest verdict, and the response is to claim nothing: COLD is
// false, so PAT degrades EXACTLY to its ceiling — i.e. to Q8 — rather than evicting on
// a guess. The run's residue_off gauge says when that happened.
func (s *Scheduler) patColdLocked(g *engineGroup, id string) bool {
	if g == nil || !g.residue.usable() {
		return false
	}
	return g.residue.ratio(id) < residueWarmFraction
}

// notePatEligible books an eligibility TRANSITION and latches it on the member record,
// so the two mechanism counters count MEMBERS BECOMING ELIGIBLE rather than one
// increment per decision pass (the rotation loop re-tests every member on every pass;
// an unlatched counter would measure contention, not the rule). A change of reason
// counts again: "this member ran to the ceiling" and "this member's pause outlived its
// cache" are different mechanisms and the artifacts must be able to tell them apart.
func (s *Scheduler) notePatEligible(g *engineGroup, m *member, reason string) {
	if m.patReason == reason {
		return
	}
	m.patReason = reason
	if s.onPatEligible != nil {
		s.onPatEligible(g.id, reason)
	}
}
