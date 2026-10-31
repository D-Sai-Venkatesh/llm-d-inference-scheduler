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

// THE PARAMETER-FREE RESIDUE MODEL (E4b) — per engine, no clocks, no constants,
// no cache reads.
//
// The scheduler WRITES the cache: it decides what enters each engine and when a
// member leaves. So it can DERIVE whether a departed member's blocks are still
// resident instead of asking the engine. The mechanism is vLLM v1's free-block
// queue (v0.30.0 as deployed, --block-size 64, prefix caching on):
//
//   - a request ALLOCATES the prompt blocks it does not cache (pop from the HEAD
//     of the free queue) and the output blocks as it decodes;
//   - on completion its blocks are FREED IN REVERSE ORDER (KVCacheManager.free:
//     "we free the blocks in reverse order so that the tail blocks are evicted
//     first"), so they land at the TAIL of the free queue and the chain's PREFIX
//     ends up deepest and survives longest;
//   - a block with a live refcount is never in the queue (vLLM touches it out).
//
// Consequence: a departed member's surviving blocks are a PREFIX of its chain.
// That is why the state here is a FIFO plus a per-member highest position, not a
// per-block set.
//
// This mirrors the offline validation's residue-sim.py (the reference
// implementation, r 0.88-0.92 against measured cached_tokens). One deliberate
// difference: the simulator replays a finished run and knows each request's
// measured cached_tokens AT dispatch, while an online scheduler only learns them
// at Settle. So the model predicts at dispatch (what it believes is resident) and
// RECONCILES at settle against usage.cached_tokens — the reference's
// settle-corrected variant, one request later. The reconciliation is counted
// (overCredit / resyncs) and drives the calibration gauge.
//
// THE MODEL IS NOT A RANKER OF COLD ENGINES (§11 + the reviewer's ruling): it
// only breaks ties among engines that already PASSED the fit gate, and only when
// it has something to say (two cold engines have a residue of 0 and fall through
// to the backlog key).

import (
	"container/list"
)

// residueWarmFraction is the model's BIMODAL CLASS cut (reviewer R3): at or above
// it the member's chain is treated as WARM on that engine, below it as COLD. The
// measured distribution is bimodal — warm ≈ 0.98, cold ≈ 0.12-0.17 — so any cut in
// the gap gives the same class; ½ is the point furthest from both modes. It is a
// CLASS boundary, never a score: two cold engines tie and the backlog/free-budget
// keys decide, which is what stops post-EXIT returns (predicted ~0.001 vs ~0.002)
// from deciding a seat on noise.
const residueWarmFraction = 0.5

// residueMinCalibration is how many reconciled samples an engine needs before its
// calibration verdict may disable the model there. 16 is a sample bound, not a
// tunable: below it a single unlucky eviction would look like a broken model.
const residueMinCalibration = 16

// residueErrLimit is the sustained per-engine error that drops the engine to
// fit-only. The model is BIMODAL by construction — a warm member reads ~0.98 and
// a cold one ~0.12-0.17 — so a mean absolute error above 0.5 means the model is
// not tracking that engine at all (a restart, a second EPP, traffic we do not
// see), and the honest response is to stop trusting it THERE rather than to keep
// silently mis-ranking. It is a CALIBRATION verdict, not a policy constant.
const residueErrLimit = 0.5

// resNode is one resident block identity: chain (member), position in that
// member's chain, its refcount, and its place in the free queue (inQ ⇒ cached and
// evictable; !inQ ⇒ allocated/live).
type resNode struct {
	member string
	pos    int
	refs   int
	elem   *list.Element // its place in the free queue (nil when live)
}

// residueHeld is what one in-flight request holds: the nodes it must free at
// completion (in reverse position order) plus the prediction it was admitted on.
type residueHeld struct {
	nodes   []*resNode
	predHit int // surviving prefix blocks the model claimed at dispatch
	full    int // prompt blocks the request asked for
}

// residueEngine is ONE engine's shadow of the free-block queue.
type residueEngine struct {
	capacity int
	// anonFree counts never-used slots. The reference pre-allocates the whole
	// pool and appends freed identities at the TAIL, so anonymous slots are
	// always consumed before any identity is recycled — modelling them as a
	// counter is exact and avoids a 40k-node slab per engine.
	anonFree int
	free     *list.List // of *resNode, head = next to be recycled
	chains   map[string]map[int]*resNode
	hi       map[string]int
	pops     uint64

	// calibration (per engine)
	samples  int
	errEWMA  float64
	resyncs  uint64 // a measured hit on a block the shadow had evicted
	overCred uint64 // a block the shadow called cached but the engine re-allocated
	disabled bool   // sustained mis-calibration ⇒ fit-only on this engine
	gen      uint64 // reset generation (restart / resync burst)
	resets   uint64
	picks    uint64 // times this engine won the residue tie-break
}

func newResidueEngine(capacity int) *residueEngine {
	return &residueEngine{capacity: capacity, anonFree: capacity, free: list.New(),
		chains: map[string]map[int]*resNode{}, hi: map[string]int{}}
}

// setCapacity re-sizes the shadow. The engine's measured capacity is the pool
// size; a capacity change means the pool was re-created, so the residue is void.
func (e *residueEngine) setCapacity(capacity int) {
	if capacity == e.capacity {
		return
	}
	e.capacity = capacity
	e.reset("capacity-change")
}

// reset clears everything: the pool starts with every block free and no cached
// content (a restart, a re-created pool, or a resync burst we cannot reconcile).
func (e *residueEngine) reset(reason string) {
	e.anonFree = e.capacity
	e.free.Init()
	e.chains = map[string]map[int]*resNode{}
	e.hi = map[string]int{}
	e.gen++
	e.resets++
	e.samples = 0
	e.errEWMA = 0
	e.disabled = false
}

// alloc gives a fresh block identity (member, pos) a slot: an anonymous slot when
// the pool still has one, else the head of the free queue — whose identity is
// LOST (that is the eviction the model is trying to predict).
func (e *residueEngine) alloc(member string, pos int) *resNode {
	if e.anonFree > 0 {
		e.anonFree--
	} else if front := e.free.Front(); front != nil {
		victim := front.Value.(*resNode)
		e.free.Remove(front)
		victim.elem = nil
		if d := e.chains[victim.member]; d != nil {
			if d[victim.pos] == victim {
				delete(d, victim.pos)
			}
		}
	}
	e.pops++
	n := &resNode{member: member, pos: pos}
	if e.chains[member] == nil {
		e.chains[member] = map[int]*resNode{}
	}
	e.chains[member][pos] = n
	return n
}

// live takes a cached block out of the free queue (vLLM touch + refcount): from
// now on it cannot be evicted.
func (e *residueEngine) live(member string, pos int) *resNode {
	d := e.chains[member]
	if d == nil {
		return nil
	}
	n := d[pos]
	if n == nil {
		return nil
	}
	if n.elem != nil {
		e.free.Remove(n.elem)
		n.elem = nil
	}
	n.refs++
	return n
}

// release drops one ref; at zero the block goes to the TAIL of the free queue —
// already-freed blocks of the same chain are therefore ahead of it, which is what
// puts a chain's prefix deepest.
func (e *residueEngine) release(n *resNode) {
	if n == nil {
		return
	}
	n.refs--
	if n.refs > 0 {
		return
	}
	n.refs = 0
	if n.elem == nil {
		n.elem = e.free.PushBack(n)
	}
}

// survivingPrefix is the model's answer: how many of the member's LEADING blocks
// are still resident (in the free queue and not yet recycled).
func (e *residueEngine) survivingPrefix(member string) int {
	d := e.chains[member]
	if d == nil {
		return 0
	}
	hi := e.hi[member]
	pos := 0
	for pos < hi {
		n := d[pos]
		// resident = still in the free queue (cached, evictable) OR live (a
		// refcount is held, so it cannot be evicted at all). A member with an
		// in-flight request reads as warm for the blocks that request holds.
		if n == nil || (n.elem == nil && n.refs == 0) {
			break
		}
		pos++
	}
	return pos
}

// ratio is the predicted surviving FRACTION of the member's chain (0 when the
// member never used this engine, which is the "cold engine" case the model must
// not rank on).
func (e *residueEngine) ratio(member string) float64 {
	hi := e.hi[member]
	if hi <= 0 {
		return 0
	}
	return float64(e.survivingPrefix(member)) / float64(hi)
}

// dispatch models one request's admission: the blocks the model believes are
// resident get touched (live, unevictable), the rest are allocated.
func (e *residueEngine) dispatch(member string, promptBlocks int) residueHeld {
	predHit := e.survivingPrefix(member)
	if predHit > promptBlocks {
		predHit = promptBlocks
	}
	held := residueHeld{predHit: predHit, full: promptBlocks}
	for pos := 0; pos < predHit; pos++ {
		n := e.live(member, pos)
		if n == nil {
			// the model called it resident but holds no identity for it: treat it
			// as a resync (the reference does the same on a measured hit).
			e.resyncs++
			n = e.alloc(member, pos)
			n.refs = 1
		}
		held.nodes = append(held.nodes, n)
	}
	for pos := max(predHit, 0); pos < promptBlocks; pos++ {
		n := e.alloc(member, pos)
		n.refs = 1
		held.nodes = append(held.nodes, n)
	}
	if promptBlocks > e.hi[member] {
		e.hi[member] = promptBlocks
	}
	return held
}

// settle reconciles one request against what the ENGINE actually reported
// (usage.prompt / usage.cached — the OUTCOME of our decisions, not a cache read),
// models the decode allocations, and frees the request's blocks in REVERSE
// position order.
func (e *residueEngine) settle(member string, held residueHeld, hitBlocks, outBlocks int) (predicted, actual float64) {
	if hitBlocks < 0 {
		hitBlocks = 0
	}
	if hitBlocks > held.full {
		hitBlocks = held.full
	}
	// OVER-CREDIT: the model claimed [actHit, predHit) were cached; the engine
	// allocated them. Give them identities so the queue stays honest.
	for pos := hitBlocks; pos < held.predHit; pos++ {
		// the model believed this block resident; the engine allocated it instead,
		// which means SOMEONE ELSE recycled the slot — so the identity is gone and
		// the slot is taken. Drop the stale identity, then re-allocate (the pop is
		// what keeps the queue's accounting honest).
		if d := e.chains[member]; d != nil {
			if old := d[pos]; old != nil {
				if old.elem != nil {
					e.free.Remove(old.elem)
					old.elem = nil
				}
				delete(d, pos)
			}
		}
		e.overCred++
		n := e.alloc(member, pos)
		n.refs = 1
		held.nodes = append(held.nodes, n)
	}
	// UNDER-CREDIT (resync): the engine reported a hit on blocks the model had
	// evicted — it was wrong about the eviction order, so re-assign them.
	resynced := 0
	for pos := held.predHit; pos < hitBlocks; pos++ {
		if e.chains[member] != nil && e.chains[member][pos] != nil {
			continue
		}
		e.resyncs++
		resynced++
		n := e.alloc(member, pos)
		n.refs = 1
		held.nodes = append(held.nodes, n)
	}
	// decode allocations
	for pos := held.full; pos < held.full+outBlocks; pos++ {
		n := e.alloc(member, pos)
		n.refs = 1
		held.nodes = append(held.nodes, n)
	}
	if hi := held.full + outBlocks; hi > e.hi[member] {
		e.hi[member] = hi
	}
	// free in REVERSE position order (the tail blocks enter the queue first)
	for i := len(held.nodes) - 1; i >= 0; i-- {
		e.release(held.nodes[i])
	}
	// calibration: predicted vs measured surviving fraction of the PROMPT
	predicted = 0.0
	if held.full > 0 {
		predicted = float64(held.predHit) / float64(held.full)
	}
	actual = 0.0
	if held.full > 0 {
		actual = float64(hitBlocks) / float64(held.full)
	}
	err := predicted - actual
	if err < 0 {
		err = -err
	}
	e.samples++
	if e.samples == 1 {
		e.errEWMA = err
	} else {
		e.errEWMA = 0.8*e.errEWMA + 0.2*err
	}
	// THE SELF-MODEL RESET (2): a sustained per-engine miss means the model is not
	// tracking this engine (a restart we did not see, a second EPP, traffic outside
	// our view). Drop the engine to FIT-ONLY — never read the cache, never invent a
	// threshold: just stop trusting the model there.
	if !e.disabled && e.samples >= residueMinCalibration && e.errEWMA > residueErrLimit {
		e.disabled = true
	}
	// SELF-MODEL RESET (1b): a RESYNC BURST — most of one request's claimed hits were
	// blocks the model had already evicted — means the shadow lost the engine's
	// state (a restart we did not observe, a second EPP, a pool re-creation). Voiding
	// the residue is strictly better than carrying a model we know is wrong: the
	// engine keeps serving, it just stops being ranked on residue until it rebuilds.
	if held.full > 0 && resynced*2 >= held.full && !e.disabled {
		e.reset("resync-burst")
	}
	return predicted, actual
}

// residueResetFloors is the engine-side hook set the scheduler calls when an
// engine's shadow must be voided. Kept as a method so the reasons are enumerated.
func (e *residueEngine) usable() bool { return e != nil && !e.disabled }
