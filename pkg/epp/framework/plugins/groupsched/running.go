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

// THE RUNNING INDEX — TIER 2 of the two-tier policy: what is IN FLIGHT on one engine, and the gate a
// member's head has to pass before it may be dispatched.
//
// It is NOT the group's BlockIndex and never enters it: the BlockIndex is the SEAT side (a member's
// whole history, charged for its whole tenure, against the membership budget), and this is the
// RUNNING side (the chains of the requests that are in flight right now, against the GPU's own
// capacity). One engine has both, they answer different questions, and summing them would count the
// same blocks twice.
//
// It is refcounted and deduplicated like the BlockIndex — two requests of one thread share their
// prefix, and a block in flight twice is one block of GPU memory — and it is exact: a request's
// charge is its chains' distinct keys, taken at the release (the same moment and the same chains the
// seat side is grown with) and released where the record ends.

// runningIndex is ONE engine's in-flight state: the deduplicated blocks of its in-flight requests,
// and how many requests those are. It has NO lock: every call comes from the scheduler under its own
// mutex (see state.go's convention).
type runningIndex struct {
	blocks map[Hash]int
	seqs   int
}

// newRunningIndex returns the empty index of an engine with no requests in flight.
func newRunningIndex() *runningIndex {
	return &runningIndex{blocks: make(map[Hash]int)}
}

// add charges one request's deduplicated block set, and ONE sequence for it. One record is one sequence:
// a multi-prompt request (len(chains) > 1) is n sequences inside vLLM, so the gate's sequence half reads
// one lower than the engine's own count for those requests. Rare (one prompt per request in every cell so
// far), and it errs toward admitting, never toward holding.
func (r *runningIndex) add(set map[Hash]struct{}) {
	r.seqs++
	for h := range set {
		r.blocks[h]++
	}
}

// release returns one request's deduplicated block set and its sequence. It reports false when it had
// to MASK an underflow — a sequence with nothing charged, or a block that is not held — which means the
// same request was released twice. The counter is still clamped (a panic under s.mu would take the whole
// scheduler down for an accounting bug), but the caller books an anomaly: an underflow that nobody counts
// is a running set that silently under-admits, which is exactly the failure the gate exists to prevent.
func (r *runningIndex) release(set map[Hash]struct{}) bool {
	ok := true
	if r.seqs > 0 {
		r.seqs--
	} else {
		ok = false
	}
	for h := range set {
		c, held := r.blocks[h]
		switch {
		case !held:
			ok = false
		case c > 1:
			r.blocks[h] = c - 1
		default:
			delete(r.blocks, h)
		}
	}
	return ok
}

// Seqs is how many requests are in flight on this engine (the gate's sequence half).
func (r *runningIndex) Seqs() int { return r.seqs }

// Blocks is how many DISTINCT blocks the in-flight requests hold (the gate's memory half).
func (r *runningIndex) Blocks() int { return len(r.blocks) }

// wouldAdd is how many NEW blocks a head's chains would add to the running set: the union's size is
// Blocks() + wouldAdd, computed without touching the index.
func (r *runningIndex) wouldAdd(chains []Chain) int {
	n := 0
	for h := range chargeSet(chains) {
		if _, held := r.blocks[h]; !held {
			n++
		}
	}
	return n
}

// ---- the gate ----

// The two halves of the dispatch gate, as metric label values and as the reason a head waited.
const (
	// gateReasonSeqs: the engine already has maxNumSeqs requests in flight.
	gateReasonSeqs = "seqs"
	// gateReasonMemory: the head's chain does not fit beside the in-flight requests' blocks inside
	// fraction × num_gpu_blocks.
	gateReasonMemory = "memory"
)

// gateReasons is every half, in a fixed order (the metric pre-instantiates one zero series per
// reason per engine: a zero is a fact, an absent series is a question).
var gateReasons = []string{gateReasonSeqs, gateReasonMemory}

// dispatchGate answers TIER 2 for ONE head on ONE engine: "" when the head may be dispatched now,
// otherwise the half that held it. It is read in the SAME critical section as the dispatch, so the
// gate and the charge can never disagree (A7).
//
// The MEMORY half is armed exactly while the group's seat budget EXCEEDS the running cap — the case the
// gate exists for. When the two are equal (no CPU tier: the GPU-sized group) the seat budget IS the
// bound, and the one-tier policy already lived with a running set that transiently exceeds it: a turn's
// chain is charged at dispatch and enforcement runs at the START of the next pass, and a marked-exiting
// member keeps running (its blocks are off the STAY side but its request is still in flight). Arming the
// gate there would hold heads the one-tier policy used to dispatch, for an over-count the seat side
// already tolerates — a decision change with no bound behind it. That is also why OFF mode is
// decision-identical to the commit this branch was cut from (verified: arming it under gpu moves the
// pinned digest).
//
// An engine with NOTHING in flight runs one head whatever its size: the dispatch twin of RULE O (A8).
// Without it a single member whose chain exceeds the running cap — a head RULE O itself seated on an
// empty engine — could never be dispatched at all, and its seat would be a stall.
func (s *Scheduler) dispatchGate(g *engineGroup, h Head) string {
	if s.cfg.MaxNumSeqs > 0 && g.running.Seqs() >= s.cfg.MaxNumSeqs {
		return gateReasonSeqs
	}
	if g.runningCapBlocks <= 0 || g.running.Seqs() == 0 || g.budgetBlocks <= g.runningCapBlocks {
		return ""
	}
	if g.running.Blocks()+g.running.wouldAdd(h.Chains) > g.runningCapBlocks {
		return gateReasonMemory
	}
	return ""
}

// ---- the record's charge, and where it moves ----

// chargeRunning opens one released request's place in its engine's running set. It is the SAME moment
// and the same chains as the seat's Grow: the request is committed to that engine from here on, not
// from its PreRequest, because the policy's own dispatch is what put it there.
func (s *Scheduler) chargeRunning(g *engineGroup, rec *reqRec) {
	if g == nil || rec == nil {
		return
	}
	if rec.runSet == nil {
		rec.runSet = chargeSet(rec.chains)
	}
	g.running.add(rec.runSet)
	rec.runningOn = g
}

// unchargeRunning closes one record's place in the running set, wherever it was opened. It holds the
// engine GROUP the record was charged to, not its thread's binding: an exit deletes the binding before
// it sweeps the records, and a record must give back what it holds to the engine that is holding it.
// (An engine torn down under its records takes its whole index with it, so releasing into the dead
// group is a no-op on state nothing reads any more.)
func (s *Scheduler) unchargeRunning(rec *reqRec) {
	if rec == nil || rec.runningOn == nil {
		return
	}
	if !rec.runningOn.running.release(rec.runSet) {
		s.noteAnomaly(anomalyRunningUnderflow)
	}
	rec.runningOn = nil
}

// moveRunning moves one in-flight record's charge to the engine that is REALLY serving it: the
// observed engine wins (rebind), and the running set has to follow it or one engine is charged for a
// request another is running.
func (s *Scheduler) moveRunning(rec *reqRec, to *engineGroup) {
	if rec == nil || rec.runningOn == nil || rec.runningOn == to {
		return
	}
	if !rec.runningOn.running.release(rec.runSet) {
		s.noteAnomaly(anomalyRunningUnderflow)
	}
	if to != nil {
		to.running.add(rec.runSet)
	}
	rec.runningOn = to
}
