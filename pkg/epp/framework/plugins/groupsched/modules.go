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
//	            TierInclusive sizes the budget from the engine's UNIQUE capacity
//	            when a CPU offload tier is configured: fraction x max(num_gpu_blocks,
//	            the tier's blocks) — the tier is INCLUSIVE (vLLM's native offload is
//	            write-through), so it holds a copy of what HBM holds and adds no
//	            capacity of its own (tier-inclusive).
//	QUEUE       EntryOrder: which waiter is the head — newcomer-first, then oldest
//	            arrival, then id (a TOTAL order; the archive's map-order tie-break was
//	            a real bug).
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
//	DISPATCH    DispatchGate: whether a seated member's turn may RUN on its engine now.
//	            DEFAULT = NONE (an admitted turn runs). The ALTERNATIVE
//	            execution-budget tests the engine's RUNNING set — the dedup-exact union
//	            of its in-flight requests' chain blocks, from this package's own
//	            BlockIndex — against the same fraction of num_gpu_blocks: the CPU tier
//	            widens RESIDENCY, never how many requests may run at once.
//	OBSERVABILITY the hooks + groupMetrics (metrics.go) + the event log.
//
// The modules never see *Scheduler: they receive the facts they are allowed to use
// as arguments (a small context struct with the READ-ONLY callbacks they need), and
// they return decisions, never state. The scheduler owns all state.

import (
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
	// the tier's, so neither implementation reaches for the other's state. A
	// tier-aware one must combine them the way the deployment's offload actually
	// combines them: the hierarchy is INCLUSIVE, so max, never the sum.
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

// TierInclusive is the SIZING alternative "tier-inclusive": the seat budget is a fraction
// of the engine's UNIQUE KV capacity — max(num_gpu_blocks, tierBlocks) — while a head's
// admission cost stays IndexSizer's GPU-only estimate.
//
//	BudgetBlocks = floor(fraction x max(num_gpu_blocks, tierBlocks))
//
// WHY THE MAX AND NOT THE SUM (the tier-inclusive correction, 2026-11-02). vLLM's native
// CPU offload is WRITE-THROUGH at compute time: every chunk is stored to the tier as it is
// computed, and nothing is stored on GPU eviction. The tier therefore holds a copy of
// essentially everything HBM holds — a block on the GPU is, by construction, also in the
// tier — so the hierarchy is INCLUSIVE: what an engine can reach is the MAX of the two
// capacities, because a block needs a place in ONE of them and the tier already covers
// everything HBM covers. A sum prices twice-billed blocks that do not exist.
//
// WHY (PREREG-multipod-offload, "the seat budget counts GPU memory only"). The budget
// sizes the working set an engine may hold, and what a miss costs is what it reserves
// for: with no tier a miss is a full prefill, with a tier ~1-4 % of one (a restore). A
// seated member whose chain has left HBM is therefore still cheap to serve, and the
// engine's reachable capacity is the max of the two. Sizing from GPU alone reserves for
// the expensive miss the deployment no longer pays: on the c64 cells stock FCFS gained ~2x
// from offload (1.390 req/s) where the GPU-sized core gained 1.267.
//
// THE CAMPAIGN DEPLOYMENT: 40,183 GPU blocks, and a tier of 512 GiB ÷ 6,291,456-byte
// blocks = 87,381 blocks (2.17x the GPU pool). The tier IS the unique capacity there, so
// the budget is floor(0.95 x 87,381) = 83,011 blocks — 2.17x the GPU-only 38,173, where the
// summed value this replaces said 121,185 (3.02x: 46 % of it did not exist).
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
// exactly; TestTierInclusiveZeroTierEqualsGPU pins that degeneracy on the state machine.
//
// THE VALUE IT REPLACES. This module was "gpu-plus-tier" (fraction x the SUM of the two
// capacities) through the batch-2 arm, which ran 121,185 blocks of seats on a 40,183-block
// GPU. The summed value is RETIRED rather than kept beside this one: HEAD must not offer a
// selector that prices capacity the deployment does not have, or a copied arm pins the
// over-seating again. Its measurement and its config stay in the archive —
// docs/gsch-core/README.md §4f, at commit 17a10e8c / digest groupsched-17a10e8c.
type TierInclusive struct {
	Fraction  float64
	BlockSize int
}

func (TierInclusive) Name() string { return "tier-inclusive" }

func (z TierInclusive) BudgetBlocks(numBlocks, tierBlocks int) int {
	unique := numBlocks
	if tierBlocks > unique {
		unique = tierBlocks
	}
	return int(z.Fraction * float64(unique))
}

// The cost functions are IndexSizer's, delegated: this implementation changes WHICH
// heads may be seated, never what a head costs.
func (z TierInclusive) Cost(idx *BlockIndex, member string, sizeBytes int64) int64 {
	return z.gpu().Cost(idx, member, sizeBytes)
}

func (z TierInclusive) TierCost(idx *BlockIndex, tier *tierShadow, member string, sizeBytes int64) (cold, resident, total int64) {
	return z.gpu().TierCost(idx, tier, member, sizeBytes)
}

func (z TierInclusive) Tokens(blocks int64) int64 { return z.gpu().Tokens(blocks) }

// gpu is this implementation's wiring facts as the GPU-only sizer, so the two selector
// values share one cost implementation instead of keeping two copies of it.
func (z TierInclusive) gpu() IndexSizer {
	return IndexSizer{Fraction: z.Fraction, BlockSize: z.BlockSize}
}

// ---- DISPATCH ----------------------------------------------------------------

// DispatchCtx is what a dispatch gate may consult about ONE seated head on ONE engine.
// It is a VALUE, and every field is a number the scheduler already has in hand: the
// check runs per candidate head on every decision pass, so it must neither allocate nor
// reach into an engine.
type DispatchCtx struct {
	// Engine is the engine the head would run on (the member's seat — a head never
	// moves engines, so this is also the attribution of a hold).
	Engine string
	// Member is the head's member id (reporting only; the gate decides on blocks).
	Member string
	// RunningBlocks is the engine's running set: the dedup-exact union of its in-flight
	// requests' chain blocks, from the EPP's own accounting. It is NEVER an endpoint
	// metric — that is what keeps a gate out of the old per-engine gate's failure class
	// (a metrics clone that latched and locked an engine out).
	RunningBlocks int64
	// ReservedBlocks is the engine's used side OUTSIDE that key set: the charges its
	// in-flight requests hold that carry no chain keys (a provisional estimate before
	// PreRequest, and a dispatched turn's output growth).
	ReservedBlocks int64
	// ChainBlocks is what THIS head would ADD to the running set: its whole estimated
	// footprint MINUS the blocks the running set already holds (dedup), PLUS its own
	// output growth.
	ChainBlocks int64
	// BudgetBlocks is this engine's dispatch budget in blocks (DispatchGate.BudgetBlocks).
	BudgetBlocks int64
	// Idle reports that NOTHING is in flight on the engine — no chain running and no
	// reservation outstanding (a turn admitted in the Pick->PreRequest window IS in
	// flight, even though its exact keys have not arrived yet).
	Idle bool
}

// DispatchGate answers exactly one question: may this seated member's turn RUN on its
// engine now?
//
// THE SEAM IS EMPTY, DELIBERATELY. The seat budget is the whole admission control: an
// admitted turn runs, which is what NoopDispatchGate answers and the only value the
// config accepts. A gate that BINDS needs the engine's running-set accounting (the
// dedup-exact union of its in-flight requests' chain blocks, plus the charges those
// requests hold outside their keys); the trim removed that accounting with its parked
// implementation, so a future gate re-adds the numbers it prices — one implementation of
// this interface plus one config value, per the module contract. What must NOT come back
// with it is the OLD per-engine gate's input: endpoint metrics (a construction-time
// clone that froze and latched) or a raw request count. The accounting must be the EPP's
// own, per engine, by construction.
//
// A gate does not RANK: it excludes a head for this pass, and the scheduler keeps its
// own order (FIFO by turn arrival among the dispatch-eligible — see PickFromHeadsEngine).
type DispatchGate interface {
	Name() string
	// BudgetBlocks is this engine's dispatch budget in blocks, from its measured GPU
	// capacity (num_gpu_blocks). 0 means NO dispatch budget: the gate can never bind,
	// which is what the default answers and what an unmeasured engine gets (it takes no
	// seats anyway, so it has no seated head to gate).
	BudgetBlocks(numBlocks int) int64
	// Allows is the decision. It must not store the context (the hot path reuses none of
	// it and relies on the value staying on the stack).
	Allows(c DispatchCtx) bool
}

// NoopDispatchGate never holds a dispatch: the seat budget is the admission control and
// an admitted turn runs. It is the ONE wired value ("none").
type NoopDispatchGate struct{}

func (NoopDispatchGate) Name() string { return "none" }

func (NoopDispatchGate) BudgetBlocks(int) int64 { return 0 }

func (NoopDispatchGate) Allows(DispatchCtx) bool { return true }

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
	gate     DispatchGate
}

// buildModules resolves one implementation per interface from the config's single
// selector value, failing closed on an unknown value.
func buildModules(cfg Config) (modules, error) {
	var m modules
	switch cfg.Sizing {
	case "", "index":
		m.sizer = IndexSizer{Fraction: cfg.GroupBudgetFraction, BlockSize: cfg.BlockSizeTokens}
	case "tier-inclusive":
		m.sizer = TierInclusive{Fraction: cfg.GroupBudgetFraction, BlockSize: cfg.BlockSizeTokens}
	default:
		return m, errValue("sizing", cfg.Sizing, "index|tier-inclusive")
	}
	switch cfg.EntryOrder {
	case "", "newcomer-first":
		m.entry = NewcomerFirst{}
	default:
		return m, errValue("entryOrder", cfg.EntryOrder, "newcomer-first")
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
	switch cfg.ExecutionBudget {
	case "", "none":
		m.gate = NoopDispatchGate{}
	default:
		return m, errValue("executionBudget", cfg.ExecutionBudget, "none")
	}
	return m, nil
}
