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

// THE CORE'S MODULES.
//
// The policy is a set of SMALL modules behind narrow interfaces. Each module owns
// exactly one question, has exactly ONE default implementation, and is selected by
// exactly ONE config value — so a future idea is a new implementation of ONE
// interface plus ONE value, never a new knob on five modules.
//
//	IDENTITY    the branch-identity plugin (separate plugin, unchanged): the
//	            member id IS the queue key. This package only READS it.
//	SIZING      FootprintSizer: what a head costs on one engine, in blocks, from
//	            the engine's own BlockIndex (dedup + cold-token estimate), and the
//	            engine's seat budget from its measured KV capacity. Default
//	            IndexSizer (budget = fraction x num_gpu_blocks); the alternative
//	            GpuPlusTier sizes the budget from GPU blocks PLUS the CPU tier's
//	            blocks (gpu-plus-tier).
//	QUEUE       EntryOrder: which waiter is the head. Default newcomer-first, then
//	            oldest arrival, then id. Alternatives: bounded HRRN (equal slowdown,
//	            denominator capped at the median waiter size) and bounded HRRN with
//	            the CPU-tier shadow in the denominator (hrrn-tier: a returner whose
//	            blocks are still in the engine's offload tier is CHEAP to re-seat).
//	PLACEMENT   EngineChooser: which engine takes the head. Default the most free
//	            budget among the engines where it fits, tie → lowest engine id.
//	TENURE      RotationPolicy: when a resident may be released. Default the fixed
//	            quantum Q (one value, default 4; Q8 is the same code with 8).
//	PRESSURE    VictimChooser: who leaves under MEMORY pressure. Default f0p's
//	            budget valve.
//	SAFETY      SafetyDetector: the per-engine net. DEFAULT = NONE: the seat budget
//	            is the whole admission control (the human's model). The memory-only
//	            detector (threshold = the seat budget's own fraction) is the
//	            ALTERNATIVE implementation, OFF until its input is verified as
//	            in-use work. Run-level safety is preemptions <= FCFS's.
//	OBSERVABILITY the hooks + groupMetrics (metrics.go) + the event log.
//
// The modules never see *Scheduler: they receive the facts they are allowed to use
// as arguments (a small context struct with the READ-ONLY callbacks they need), and
// they return decisions, never state. The scheduler owns all state.

import (
	"sort"
	"time"
)

// ---- TENURE ------------------------------------------------------------------

// TenureCtx is what a rotation policy may consult about one resident.
type TenureCtx struct {
	// Now is the decision pass's clock (never time.Now — a golden replay must be
	// reproducible).
	Now time.Time
	// IdleSeconds is S, the silence bound: a member silent that long (and with no
	// queued head — the C8 fix) is not going to use its seat.
	IdleSeconds float64
	// QueueView is when the queue-head snapshot behind this decision pass was taken,
	// and QueueViewKnown whether there is one at all. A member's "has a queued head"
	// lives in `waitingSince`, which only a PASS OVER THE QUEUE can maintain
	// (syncWaitersLocked). When the flow controller's gate is closed no Pick runs, so
	// that flag goes STALE — and an eviction rule that trusted a stale "no head" took
	// seats away from members whose head was queued the whole time (the C8 race: heads
	// arriving just before a Tick, victims' heads then waiting 4.7-25 min, cold).
	QueueView    time.Time
	QueueViewSet bool
}

// Headless reports "this member has no queued head" as EVIDENCE rather than as a
// guess: it is true only when the queue view is recent enough for the absence to mean
// something (a view older than S cannot tell a busy member from an idle one).
func (c TenureCtx) Headless(m *member) bool {
	if !m.waitingSince.IsZero() {
		return false
	}
	if !c.QueueViewSet || c.Now.Sub(c.QueueView) > time.Duration(c.IdleSeconds*float64(time.Second)) {
		return false // no trustworthy view: silence is NOT evidence
	}
	return true
}

// Silent reports the idle test for a member: headless (see Headless) and no activity
// for S.
func (c TenureCtx) Silent(m *member) bool {
	if !c.Headless(m) {
		return false
	}
	return c.Now.Sub(m.lastActivity) > time.Duration(c.IdleSeconds*float64(time.Second))
}

// RotationPolicy answers "may this resident be released now, and why?".
type RotationPolicy interface {
	// Name is the config value that selects the policy.
	Name() string
	// Eligible reports the reason a member may be rotated ("" and false = it may
	// not). A member with anything in flight is NEVER eligible.
	Eligible(m *member, ctx TenureCtx) (string, bool)
}

// FixedQuantum is the ONE tenure rule: Q completed busy periods, or S of silence.
type FixedQuantum struct {
	Q int
}

func (FixedQuantum) Name() string { return "quantum" }

func (p FixedQuantum) Eligible(m *member, ctx TenureCtx) (string, bool) {
	if m.inflight > 0 {
		return "", false
	}
	if m.busyPeriods >= p.Q {
		return "quantum-complete", true
	}
	if ctx.Silent(m) {
		return "idle", true
	}
	return "", false
}

// ---- PRESSURE ----------------------------------------------------------------

// PressureCtx is what a victim chooser may consult about ONE engine.
type PressureCtx struct {
	Now time.Time
	// Exiting reports that the member is already on its way out (it drains, then
	// goes).
	Exiting func(id string) bool
	// RotationVictim is the TENURE module's demand-driven choice over this engine
	// (nil when no member is release-eligible under it).
	RotationVictim func() (string, *member)
	// Silent is the same idle test the tenure policy uses.
	Silent func(m *member) bool
	// Headless reports whether the member has no queued head, as EVIDENCE (false when
	// the queue view is stale — see TenureCtx.Headless).
	Headless func(m *member) bool
	// QueueViewTrusted reports whether the queue view is recent enough that "this
	// member has no queued head" means anything at all. When it is false, the classes
	// that reason about HEAD PRESENCE (done and mid-quantum) are SKIPPED: the policy
	// cannot tell an idle member from a busy one, so it must not reclaim a seat on a
	// guess — that guess is exactly the C8 race (heads arriving just before a Tick,
	// victims' heads then waiting 4.7-25 minutes, cold). Memory enforcement then waits
	// for the next pass, which is at most one tick away.
	QueueViewTrusted bool
}

// VictimChooser answers "who leaves when the engine is over its budget?".
type VictimChooser interface {
	Name() string
	Victim(members []*member, ctx PressureCtx) (string, *member)
}

// BudgetValve is f0p's memory-pressure order, unchanged:
//
//	(1) a rotation victim (the tenure rule already releases this member);
//	(2) DONE or SILENT — no queued turn, or silent past S;
//	(3) mid-quantum IDLE, oldest last completion first (the safety valve that only
//	    genuine GROWTH pressure reaches: every admission was budget-checked);
//	(4) the EXITING class, mid-flight.
type BudgetValve struct{}

func (BudgetValve) Name() string { return "valve" }

func (BudgetValve) Victim(members []*member, ctx PressureCtx) (string, *member) {
	if ctx.RotationVictim != nil {
		if reason, m := ctx.RotationVictim(); m != nil {
			return reason, m
		}
	}
	var best *member
	for _, m := range members {
		if ctx.Exiting(m.id) || m.inflight > 0 {
			continue
		}
		if ctx.Headless(m) || ctx.Silent(m) {
			if best == nil || olderCompletion(m, best) {
				best = m
			}
		}
	}
	if best != nil {
		return "budget-noqueue", best
	}
	if !ctx.QueueViewTrusted {
		// no trustworthy view of who has a queued head: classes (2) and (3) are guesses.
		// Only members already on their way out can be named.
		for _, m := range members {
			if !ctx.Exiting(m.id) {
				continue
			}
			if best == nil || olderCompletion(m, best) {
				best = m
			}
		}
		if best != nil {
			return "budget-exiting", best
		}
		return "", nil
	}
	// class (3): mid-quantum. THE ORDER MATTERS: a member that has not served a turn
	// in this residency, or that has a queued head right now, is the LAST choice —
	// taking its seat back before its first turn makes it wait another admission cycle
	// (measured on f0p's class-3 victims: 100 % had a queued head and 53/54 had served
	// ZERO turns in the residency). Only then "oldest last completion", then id.
	for _, m := range members {
		if ctx.Exiting(m.id) || m.inflight > 0 {
			continue
		}
		if best == nil || class3Less(m, best) {
			best = m
		}
	}
	if best != nil {
		return "budget-idle", best
	}
	for _, m := range members {
		if ctx.Exiting(m.id) {
			continue
		}
		if best == nil || olderCompletion(m, best) {
			best = m
		}
	}
	if best != nil {
		return "budget-exiting", best
	}
	return "", nil
}

// ---- QUEUE -------------------------------------------------------------------

// EntryCtx is what an entry order may consult about ONE decision pass. It is an
// argument and not hidden state on purpose: the module sees only the facts the
// scheduler hands it (never *Scheduler, never time.Now), so a replay is reproducible
// and a policy cannot quietly read a clock its inputs never carried.
type EntryCtx struct {
	// Now is the decision pass's clock (never time.Now — a golden replay must be
	// reproducible). "How long this head has waited" is derived from it and the
	// waiter's own oldest arrival, never stored.
	Now time.Time
}

// EntryOrder answers "which waiter is the head?" over the ONE shared queue.
//
// The comparison MUST be a TOTAL order ending in the member id. That is not a style
// preference: the archive build compared unset timestamps in map-iteration order and
// produced three different traces in four replays of the same scenario.
type EntryOrder interface {
	Name() string
	// Head returns the head of the waiter set, or nil when nothing is waiting.
	Head(waiters map[string]*waiter, ctx EntryCtx) *waiter
}

// NewcomerFirst is the default: newcomers before returning members, then the oldest
// arrival, then the id (a total order, so map iteration cannot leak into a decision).
type NewcomerFirst struct{}

func (NewcomerFirst) Name() string { return "newcomer-first" }

func (NewcomerFirst) Head(waiters map[string]*waiter, _ EntryCtx) *waiter {
	var head *waiter
	for _, w := range waiters {
		if head == nil || newcomerLess(w, head) {
			head = w
		}
	}
	return head
}

// newcomerLess is the default entry order's total comparison (lower wins).
func newcomerLess(a, b *waiter) bool {
	if a.newcomer != b.newcomer {
		return a.newcomer // newcomers first
	}
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since) // oldest first
	}
	return a.id < b.id
}

// HrrnBounded is F5′: BOUNDED HRRN — the equal-slowdown criterion with the
// denominator capped at the median waiter size. It is the archive's
// `entryOrder: "hrrn-bounded"` (dsai/groupsched-tier-aware @ dd5350b5) moved here as ONE
// EntryOrder implementation.
//
// THE RULE. Every waiter's priority is
//
//	wait ÷ s_eff,   s_eff = min(this waiter's cold tokens, the LOWER-median cold
//	                       tokens of the waiters in THIS decision pass)
//
// and the highest-priority waiter is the head. "Cold tokens" is the admission cost the
// sizing module already computed for the head (one unit everywhere, R1), so the rule
// adds no estimate, no rate, no cache read and no clock of its own.
//
// WHY THE CAP (the measured defect of plain HRRN, B1b 2026-11-01). Uncapped, the
// denominator is the waiter's OWN size, so a 200K-token entry ages ~67x slower than a 3K
// one: its takeover bound is (a small waiter's ratio) × cold_big — starvation-free in
// theory, ~an hour in practice (max entry wait 3,601 s, 20 stalled sessions). Capped at
// the median, the bound becomes (that ratio) × median ≈ ten minutes, derived from the
// queue rather than fitted.
//
// THREE PROPERTIES THE CAP BUYS, all from the LOWER median (index (n-1)/2):
//
//   - waiters BELOW the median keep their own size, so small-first ordering inside the
//     small class is exactly uncapped HRRN's (the cap can only bind at or above it);
//   - waiters AT or ABOVE it age at the same rate, so they are served oldest-first
//     among themselves instead of in a size-ordered starvation;
//   - the bound is the POPULATION's own size: a queue of big entries raises the median
//     and they all age slowly together; a queue of small ones serves them at the small
//     class's rate. Nothing here is a tuning constant.
//
// THE TOTAL ORDER. The newcomer CLASS is the outer key, exactly as in the archive's
// entryKey (its isNew is compared before the response ratio), then the priority
// DESCENDING, then the older arrival, then the id. So this is not "newcomer-first plus a
// ratio": the class is the archive's outermost partition, and INSIDE a class the ratio
// replaces age ordering entirely.
type HrrnBounded struct {
	// BlockSize is the engine's KV block size in tokens: the unit that floors a
	// waiter's size (and therefore any age rate) at ONE BLOCK, so a fully warm head
	// contributes the smallest possible size instead of collapsing the median to zero
	// and silently restoring the unbounded rule. A wiring fact (Config, once).
	BlockSize int
}

func (HrrnBounded) Name() string { return "hrrn-bounded" }

func (h HrrnBounded) Head(waiters map[string]*waiter, ctx EntryCtx) *waiter {
	// ONE bound per CALL, over the CURRENT waiter set — the same pool this loop orders.
	// Computed here and never stored: a cached median would be a fact about an older
	// queue. (The admission loop calls this once per head it hands out, so a pass that
	// seats several heads recomputes the bound as its queue shrinks — which is the
	// archive's accounting, and the equivalence golden depends on it.)
	bound := lowerMedianColdTokens(waiters, h.BlockSize)
	var head *waiter
	var headPriority float64
	for _, w := range waiters {
		p := h.priority(w, bound, ctx.Now)
		if head == nil || hrrnLess(w, p, head, headPriority) {
			head, headPriority = w, p
		}
	}
	return head
}

// priority is wait ÷ s_eff for ONE waiter under one pass's bound: the waiter's own cold
// tokens, capped at the bound, floored at one block.
func (h HrrnBounded) priority(w *waiter, bound int64, now time.Time) float64 {
	return priorityWith(w, w.coldTokens, bound, now, h.BlockSize)
}

// priorityWith is the ONE priority formula, over an explicit cost: wait ÷ s_eff, where
// s_eff = min(cost, the pass's bound) floored at one block. Both F5′ and hrrn-tier are
// this function; they differ only in WHAT they hand it as the cost (the cold tokens of
// the GPU-estimate, or the tier-aware cold tokens), which is what keeps the two arms a
// one-value comparison rather than two rules.
func priorityWith(w *waiter, cost int64, bound int64, now time.Time, blockSize int) float64 {
	den := cost
	if den <= 0 {
		den = int64(blockSize) // a fully warm head: the smallest possible size
	}
	if bound > 0 && bound < den {
		den = bound // the cap binds only AT or ABOVE the median (see lowerMedianTokens)
	}
	wait := now.Sub(w.since).Seconds()
	if wait < 0 {
		wait = 0 // a head whose arrival is in the pass's future waits zero
	}
	return wait / float64(den)
}

// hrrnLess is bounded HRRN's total comparison (lower wins): the newcomer class, then the
// higher response ratio, then the older arrival, then the id.
func hrrnLess(a *waiter, aPriority float64, b *waiter, bPriority float64) bool {
	if a.newcomer != b.newcomer {
		return a.newcomer // the newcomer class is the OUTER key (the archive's isNew)
	}
	if aPriority != bPriority {
		return aPriority > bPriority // higher response ratio first
	}
	if !a.since.Equal(b.since) {
		return a.since.Before(b.since) // tie → the older arrival
	}
	return a.id < b.id // tie → the id (a TOTAL order)
}

// lowerMedianColdTokens is the LOWER median of the waiter set's cold-token sizes, each
// floored at one block exactly as priority floors its denominator. The LOWER median is
// deliberate: for n waiters it is sizes[(n-1)/2], so every size strictly below it is
// below it in the sorted list too, and min(cold, median) can only bind at or above the
// median — which is what leaves the small class's ordering untouched. 0 when nothing is
// waiting (0 means "no cap": a lone waiter is its own median).
func lowerMedianColdTokens(waiters map[string]*waiter, blockSize int) int64 {
	return lowerMedianTokens(waiters, blockSize, func(w *waiter) int64 { return w.coldTokens })
}

// lowerMedianTokens is the same bound over a CHOSEN cost, so the two bounded-HRRN
// implementations cannot drift apart in the part they share (the median, the floor, the
// cap): F5′ passes the cold tokens, hrrn-tier passes the tier-aware ones.
func lowerMedianTokens(waiters map[string]*waiter, blockSize int, cost func(*waiter) int64) int64 {
	if len(waiters) == 0 {
		return 0
	}
	sizes := make([]int64, 0, len(waiters))
	for _, w := range waiters {
		n := cost(w)
		if n <= 0 {
			n = int64(blockSize)
		}
		sizes = append(sizes, n)
	}
	sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })
	return sizes[(len(sizes)-1)/2]
}

// HrrnTier is F5′ WITH THE CPU TIER IN THE DENOMINATOR (PREREG-multipod-offload,
// "THE OFFLOAD SOLUTION" piece 2): byte-for-byte F5′'s rule — the newcomer class as the
// OUTER key, bounded HRRN inside a class, the LOWER-median clamp, the same total tie
// order — with ONE input changed:
//
//	s = tokens predicted resident in NEITHER the engine's GPU index NOR its CPU-tier
//	    shadow     (restorable tokens count as CACHED)
//
// WHY. Measured on the c128 order cut: the share of a returning branch's prefix that
// comes back from the CPU tier falls off a cliff between one and three minutes away
// (90 % under 1 min, 30 % at 1-3 min, 14 % at 3-5 min, 6.4 % at 5-10 min), and the
// scheduler's own entry wait for a seat-loss return is 328-420 s — i.e. the queue
// itself is what pushes returners over that cliff. A returner with a live CPU copy has
// a DEADLINE: seating it a minute later costs a full cold prefill, seating it now costs
// a restore. Plain F5′ is FIFO-shaped among returners of equal size and therefore serves
// the most-likely-cold first — the wrong way round.
//
// MECHANISM, and why it is not a new knob. A returner whose blocks are still in the
// engine's tier costs ~its NEW TURN's tokens (its history is restorable), so its ratio
// wait ÷ s is large almost immediately and it is seated before it falls off the cliff. A
// returner whose copy is gone (a COLD return, s = min(full, median) — the F5′ bound
// unchanged) keeps a small ratio and therefore AGES; when its wait does overtake, the
// takeover bound is the COST RATIO times the warm waiter's wait, so the landscape is F5′'s
// bounded HRRN with the warm class served first. Newcomers are untouched: the class key
// is still the outer partition, so a returner — warm or cold — never jumps a newcomer.
//
// WHAT IT MAY NOT DO. The tier-aware cost is a PRIORITY input ONLY: the seat budget, the
// cold-entry valve and placement keep the GPU-only cost (offload adds no SEATS — vLLM
// runs only from HBM — so admitting against a tier-aware cost would overbook the engine
// the moment the restores land).
//
// TIER OFF ⇒ F5′ EXACTLY. With no tier configured the scheduler writes the tier-aware
// cost EQUAL to the cold cost for every waiter (setWaiterCostsLocked), so this module is
// arithmetically the same function as HrrnBounded. TestHrrnTierNoTierEqualsF5P pins that on
// the full replay, byte for byte, against the archive's F5′ trace.
type HrrnTier struct {
	// BlockSize is the engine's KV block size in tokens (see HrrnBounded.BlockSize).
	BlockSize int
}

func (HrrnTier) Name() string { return "hrrn-tier" }

func (h HrrnTier) Head(waiters map[string]*waiter, ctx EntryCtx) *waiter {
	bound := lowerMedianTokens(waiters, h.BlockSize, tierCostOf)
	var head *waiter
	var headPriority float64
	for _, w := range waiters {
		p := h.priority(w, bound, ctx.Now)
		if head == nil || hrrnLess(w, p, head, headPriority) {
			head, headPriority = w, p
		}
	}
	return head
}

// priority is F5′'s priority over the TIER-AWARE cost.
func (h HrrnTier) priority(w *waiter, bound int64, now time.Time) float64 {
	return priorityWith(w, tierCostOf(w), bound, now, h.BlockSize)
}

// tierCostOf is the cost hrrn-tier prices a waiter with: the tier-aware cold tokens the
// scheduler computed for this decision pass. A waiter whose tier cost was never computed
// (no engine had a tier during its sync — the tier-off case, and the unit-test fixtures
// that build a queue directly) falls back to its plain cold tokens, which is exactly
// F5′'s denominator: the degenerate case is the DEFAULT, not a special path.
func tierCostOf(w *waiter) int64 {
	if w.tierCostSet {
		return w.tierCostTokens
	}
	return w.coldTokens
}

// ---- PLACEMENT ---------------------------------------------------------------

// EngineCandidate is one engine's offer for one waiting head — the FACTS the
// placement module may decide on. The scheduler computes them; the module never
// touches an engine.
type EngineCandidate struct {
	// Engine is the engine id (the deterministic tie-break).
	Engine string
	// CostTokens is this head's admission cost ON THIS ENGINE (tokens), and
	// CostBlocks its block-quantised form.
	CostTokens int64
	CostBlocks int64
	// FreeBudget is what the engine can still admit (budget − accounted footprint).
	FreeBudget int64
	// BudgetFits reports CostBlocks <= FreeBudget.
	BudgetFits bool
	// Fits reports BudgetFits AND that the COLD-ENTRY VALVE allows this head on this
	// engine. Both halves matter, and keeping them apart is the fix for a real defect
	// (inherited from f0p): a head that fits by budget but is held by the valve must
	// WAIT, because evicting members frees budget and never frees the valve — the old
	// loop therefore evicted every eligible member on the engine and left the seats
	// empty until the cold prefill finished (measured on f0p's final-epp.prom:
	// 913,162 valve-blocked passes, and every one that found a victim evicted for
	// nothing).
	Fits bool
	// HasVictim is LAZY (it costs a scan): may the engine evict an eligible
	// incumbent right now?
	HasVictim func() bool
}

// EngineChooser answers "which engine takes this head?".
type EngineChooser interface {
	Name() string
	// Choose returns the index into cands of the engine to use, or -1.
	Choose(cands []EngineCandidate) int
}

// MostFreeBudget is the default: among the engines where the head FITS (budget AND the
// cold valve), the one with the MOST FREE BUDGET; ties → lowest engine id.
//
//   - If NOTHING fits but some engine fits BY BUDGET (i.e. the cold-entry valve is
//     holding the head), the answer is "wait": the valve opens by itself when that
//     engine's cold prefill produces its first token, and no eviction can hurry it.
//   - Only when nothing fits by BUDGET either is eviction the way forward: the engine
//     that can free a seat by an eligible rotation, again most free budget first
//     (evicting on the least-pressured engine gives up the least warm state).
type MostFreeBudget struct{}

func (MostFreeBudget) Name() string { return "most-free-budget" }

func (MostFreeBudget) Choose(cands []EngineCandidate) int {
	best := -1
	for i := range cands {
		if !cands[i].Fits {
			continue
		}
		if best < 0 || betterEngine(cands[i], cands[best]) {
			best = i
		}
	}
	if best >= 0 {
		return best
	}
	// the cold valve is holding this head somewhere: WAIT, never evict (see Fits)
	for i := range cands {
		if cands[i].BudgetFits {
			return -1
		}
	}
	for i := range cands {
		if cands[i].HasVictim == nil || !cands[i].HasVictim() {
			continue
		}
		if best < 0 || betterEngine(cands[i], cands[best]) {
			best = i
		}
	}
	return best
}

// betterEngine is the placement total order: most free budget, then engine id.
func betterEngine(c, best EngineCandidate) bool {
	if c.FreeBudget != best.FreeBudget {
		return c.FreeBudget > best.FreeBudget
	}
	return c.Engine < best.Engine
}

// ---- SIZING ------------------------------------------------------------------

// FootprintSizer answers "what does this head cost on this engine?" and "how many
// blocks may this engine seat?".
type FootprintSizer interface {
	Name() string
	// BudgetBlocks is one engine's seat budget, from its measured KV capacity and
	// that engine's CPU tier capacity in blocks (0 = no tier configured). Both
	// numbers are per-engine deployment facts; a GPU-only implementation ignores
	// the tier's, so neither implementation reaches for the other's state.
	BudgetBlocks(numBlocks, tierBlocks int) int
	// Cost is the conservative admission cost of a QUEUED request against one
	// engine's index, in blocks.
	Cost(idx *BlockIndex, member string, sizeBytes int64) int64
	// TierCost is Cost with that engine's CPU-TIER SHADOW as a second residence
	// oracle, split into the three numbers its two readers need:
	//
	//	cold     blocks resident in NEITHER the GPU index NOR the tier — the tier-aware
	//	         cold estimate that hrrn-tier's denominator uses. (It is a PRIORITY input:
	//	         the seat budget and the cold valve keep cost()'s GPU-only number.)
	//	resident blocks the index or the tier already holds (restorable ≈ free);
	//	total    the head's whole estimated footprint (cold + resident).
	//
	// A nil tier means "no offload tier configured" and reproduces Cost exactly
	// (Cost is this function's cold term), so the tier-aware path and the default
	// path can never disagree about the same engine state.
	TierCost(idx *BlockIndex, tier *tierShadow, member string, sizeBytes int64) (cold, resident, total int64)
	// Tokens converts a block count to tokens (R1: one unit everywhere).
	Tokens(blocks int64) int64
}

// IndexSizer is the default: the engine's own BlockIndex (dedup against what the
// engine already holds) plus the wire-size delta over the member's last dispatched
// body, and a budget of fraction × num_gpu_blocks.
type IndexSizer struct {
	Fraction  float64
	BlockSize int
}

func (IndexSizer) Name() string { return "index" }

func (z IndexSizer) BudgetBlocks(numBlocks, _ int) int {
	return int(z.Fraction * float64(numBlocks))
}

func (z IndexSizer) Cost(idx *BlockIndex, member string, sizeBytes int64) int64 {
	cold, _, _ := z.TierCost(idx, nil, member, sizeBytes)
	return cold
}

func (z IndexSizer) TierCost(idx *BlockIndex, tier *tierShadow, member string, sizeBytes int64) (cold, resident, total int64) {
	for _, h := range idx.SetBlocks(member) { // persisted across residencies (CP1)
		if idx.HasNELocked(h) || tier.Holds(h) {
			// the engine can reach this block without recomputing it: in HBM now, or
			// restorable from its CPU tier for ~1-4 % of a prefill.
			resident++
			continue
		}
		cold++
	}
	// the baseline is the member's last DISPATCHED body size (the index record,
	// updated at PreRequest) — never the current head's size, or the second sight of
	// the same head would estimate zero new blocks.
	delta := sizeBytes - idx.LastSizeBytesLocked(member)
	if delta < 0 {
		delta = 0
	}
	// ROUNDED UP: a partial block still occupies a block. Truncating the division
	// understates every head whose growth is not a whole block, in the same
	// direction as (and on top of) the byte-to-token conversion this estimate is
	// stuck with. The GROWTH is always cold: a block that never existed cannot be
	// restorable, whatever the tier holds.
	blockBytes := 4 * int64(z.BlockSize)
	cold += (delta + blockBytes - 1) / blockBytes
	return cold, resident, cold + resident
}

func (z IndexSizer) Tokens(blocks int64) int64 { return blocks * int64(z.BlockSize) }

// GpuPlusTier is the SIZING alternative "gpu-plus-tier": the seat budget is a fraction
// of the engine's whole reachable working set — its measured GPU blocks PLUS its CPU
// tier's blocks — while a head's admission cost stays IndexSizer's GPU-only estimate.
//
//	BudgetBlocks = floor(fraction x (num_gpu_blocks + tierBlocks))
//
// WHY (PREREG-multipod-offload, "the seat budget counts GPU memory only"). The budget
// sizes the working set an engine may hold, and what a miss costs is what it reserves
// for: with no tier a miss is a full prefill, with a tier ~1-4 % of one (a restore). A
// seated member whose chain has left HBM is therefore still cheap to serve, and the
// engine's reachable capacity is GPU + tier — 40,183 GPU blocks + 87,381 tier blocks =
// 3.17x the GPU pool on the campaign deployment. Sizing from GPU alone reserves for the
// expensive miss the deployment no longer pays: on the c64 cells stock FCFS gained ~2x
// from offload (1.390 req/s) where the GPU-sized core gained 1.267.
//
// WHAT DOES NOT GROW. The cold-entry valve's cap stays floor((1-fraction) x
// num_gpu_blocks) x blockSize (engineGroup.coldCapTokens): it staggers GPU allocation
// bursts, and the tier cannot absorb one. The per-head cost stays GPU-only, so a head is
// admitted on what it will occupy in HBM — the same rule hrrn-tier's tier-aware cost
// obeys (restorable blocks price priority, never seats).
//
// tierBlocks is the engine's TIER SHADOW capacity (tiershadow.go): the same two
// deployment facts that size the shadow, read back from it, so this implementation adds
// no constant of its own. 0 (no tier configured) makes BudgetBlocks IndexSizer's number
// exactly; TestGpuPlusTierZeroTierEqualsGPU pins that degeneracy on the state machine.
type GpuPlusTier struct {
	Fraction  float64
	BlockSize int
}

func (GpuPlusTier) Name() string { return "gpu-plus-tier" }

func (z GpuPlusTier) BudgetBlocks(numBlocks, tierBlocks int) int {
	return int(z.Fraction * float64(numBlocks+tierBlocks))
}

// The cost functions are IndexSizer's, delegated: this implementation changes WHICH
// heads may be seated, never what a head costs.
func (z GpuPlusTier) Cost(idx *BlockIndex, member string, sizeBytes int64) int64 {
	return z.gpu().Cost(idx, member, sizeBytes)
}

func (z GpuPlusTier) TierCost(idx *BlockIndex, tier *tierShadow, member string, sizeBytes int64) (cold, resident, total int64) {
	return z.gpu().TierCost(idx, tier, member, sizeBytes)
}

func (z GpuPlusTier) Tokens(blocks int64) int64 { return z.gpu().Tokens(blocks) }

// gpu is this implementation's wiring facts as the GPU-only sizer, so the two selector
// values share one cost implementation instead of keeping two copies of it.
func (z GpuPlusTier) gpu() IndexSizer {
	return IndexSizer{Fraction: z.Fraction, BlockSize: z.BlockSize}
}

// ---- SAFETY ------------------------------------------------------------------

// SafetyDetector is THE definition of the net. The DEFAULT is NoopSafety: the seat
// budget alone governs dispatch and admission, which is the human's model and makes
// the core's no-constant claim exact. BudgetFraction (the memory-only detector) is
// the ALTERNATIVE, kept wired and tested but OFF: its input is vLLM's kv_cache_usage,
// and whether that measures IN-USE work (blocks with ref_cnt > 0) or cache occupancy
// is still being verified against the vLLM source — a metric gate that latches on a
// wrongly-understood input is worse than no gate (the E-wave measured exactly that).
//
// The run-level safety that replaces it: preemptions <= the FCFS arm's (a hard fail),
// plus the wave's own liveness gates. One implementation is
// used by two readers: the scheduler (per-engine dispatch/seat holds) and the
// framework's detector plugin (the pool-level Saturation signal) — see
// detector.go, which is the same value wired into the framework's detector slot.
type SafetyDetector interface {
	Name() string
	// Threshold is the utilisation at which the net fires (the budget's own
	// fraction: no second constant).
	Threshold() float64
	// Fired reports whether a scrape fires the net. have=false (no snapshot, or a
	// stale one) is OPEN: the seat budget is the primary control, so a detector that
	// fails closed would only add a stall mode of its own.
	Fired(kv float64, have bool) bool
	// Score is the utilisation the framework's Saturation signal uses: kv /
	// Threshold, so 1.0 means exactly "firing".
	Score(kv float64, have bool) float64
}

// class3Less is the mid-quantum (class 3) victim order: UNSTARTED or WANTING members
// sort last (they have not had the seat long enough for it to be worth reclaiming),
// then the oldest last completion, then the id.
func class3Less(a, b *member) bool {
	aRaw := a.resTurns == 0 || !a.waitingSince.IsZero()
	bRaw := b.resTurns == 0 || !b.waitingSince.IsZero()
	if aRaw != bRaw {
		return !aRaw // a member that has used its seat goes first
	}
	return olderCompletion(a, b)
}

// NoopSafety never fires: dispatch and admission are governed by the seat budget
// alone.
type NoopSafety struct{}

func (NoopSafety) Name() string { return "none" }

func (NoopSafety) Threshold() float64 { return 0 }

func (NoopSafety) Fired(float64, bool) bool { return false }

func (NoopSafety) Score(float64, bool) float64 { return 0 }

// BudgetFraction is the ALTERNATIVE: the memory-only detector at the seat budget's own
// fraction (stale/absent metrics are OPEN — the budget is the primary control).
type BudgetFraction struct {
	Fraction float64
}

func (BudgetFraction) Name() string { return "budget-fraction" }

func (d BudgetFraction) Threshold() float64 { return d.Fraction }

func (d BudgetFraction) Fired(kv float64, have bool) bool {
	return have && kv >= d.Fraction
}

func (d BudgetFraction) Score(kv float64, have bool) float64 {
	if !have {
		return 0 // no evidence → open (the budget is the primary control)
	}
	return kv / d.Fraction
}

// ---- the bundle --------------------------------------------------------------

// modules is the wired set. NewScheduler builds it ONCE from Config; nothing below
// ever consults Config for a policy decision again (no cross-module knobs).
type modules struct {
	sizer    FootprintSizer
	entry    EntryOrder
	place    EngineChooser
	tenure   RotationPolicy
	pressure VictimChooser
	safety   SafetyDetector
	// entryPricesTier reports that the wired QUEUE module prices waiters with the
	// TIER-AWARE cost (hrrn-tier), so the scheduler should compute it. It is a fact about
	// the module set, not a second knob: an arm that carries a CPU tier but queues with
	// F5′/newcomer-first pays NOTHING for a number nothing reads — which matters, because
	// the EPP's own CPU is a hidden TTFT variable and the comparator arms are exactly the
	// ones that must not be perturbed by work their policy never uses.
	entryPricesTier bool
}

// buildModules resolves one implementation per interface from the config's single
// selector value, failing closed on an unknown value.
func buildModules(cfg Config) (modules, error) {
	var m modules
	switch cfg.Sizing {
	case "", "index":
		m.sizer = IndexSizer{Fraction: cfg.GroupBudgetFraction, BlockSize: cfg.BlockSizeTokens}
	case "gpu-plus-tier":
		m.sizer = GpuPlusTier{Fraction: cfg.GroupBudgetFraction, BlockSize: cfg.BlockSizeTokens}
	default:
		return m, errValue("sizing", cfg.Sizing, "index|gpu-plus-tier")
	}
	switch cfg.EntryOrder {
	case "", "newcomer-first":
		m.entry = NewcomerFirst{}
	case "hrrn-bounded":
		m.entry = HrrnBounded{BlockSize: cfg.BlockSizeTokens}
	case "hrrn-tier":
		m.entry = HrrnTier{BlockSize: cfg.BlockSizeTokens}
		m.entryPricesTier = true
	default:
		return m, errValue("entryOrder", cfg.EntryOrder, "newcomer-first|hrrn-bounded|hrrn-tier")
	}
	switch cfg.EngineChoice {
	case "", "most-free-budget":
		m.place = MostFreeBudget{}
	default:
		return m, errValue("engineChoice", cfg.EngineChoice, "most-free-budget")
	}
	switch cfg.Rotation {
	case "", "quantum":
		m.tenure = FixedQuantum{Q: cfg.quantum()}
	default:
		return m, errValue("rotation", cfg.Rotation, "quantum")
	}
	switch cfg.Victim {
	case "", "valve":
		m.pressure = BudgetValve{}
	default:
		return m, errValue("victim", cfg.Victim, "valve")
	}
	switch cfg.Safety {
	case "", "none":
		m.safety = NoopSafety{}
	case "budget-fraction":
		m.safety = BudgetFraction{Fraction: cfg.GroupBudgetFraction}
	default:
		return m, errValue("safety", cfg.Safety, "none|budget-fraction")
	}
	return m, nil
}
