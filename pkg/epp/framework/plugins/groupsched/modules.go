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
//	            branch id IS the queue key. This package only READS it.
//	SIZING      SeatBudget: one engine's seat budget, from its measured KV capacity — the
//	            ONE implementation IndexSizer: budget = fraction x num_gpu_blocks. What a
//	            head COSTS is not a module's answer any more: it is exact, from the
//	            engine's own BlockIndex. The CPU offload tier is a DEPLOYMENT fact, not a
//	            sizing input: native offload is write-through, so it adds no seat.
//	QUEUE       EntryOrder: which waiter is the head — newcomer-first, then oldest
//	            arrival, then id (a TOTAL order: a map-order tie-break is a real bug).
//	PLACEMENT   EngineChooser: which engine takes the head. Default: the most free budget
//	            left AFTER the seat, among the engines where it fits, tie → lowest id.
//	TENURE      RotationPolicy: when a resident may be released. Default the fixed
//	            quantum Q (one value, default 4; Q8 is the same code with 8).
//	PRESSURE    VictimChooser: who leaves under MEMORY pressure. Default: the budget valve.
//	SAFETY      SafetyDetector: a seam with ONE wired value, NONE — the seat budget is the
//	            whole admission control, so a net has nothing to add and no implementation
//	            ships (a future one brings its own evidence and one config value).
//	DISPATCH    DispatchGate: whether a seated branch's turn may RUN on its engine now.
//	            ONE wired value, NONE (an admitted turn runs). A gate that BINDS would need
//	            the engine's running-set accounting, which the trim removed with its parked
//	            implementation — it must be re-added with the gate that prices it.
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
	// Now is the decision pass's clock (never time.Now — a replayed decision stream must be
	// reproducible).
	Now time.Time
	// IdleSeconds is S, the silence bound: a branch silent that long (and with no
	// queued head) is not going to use its seat.
	IdleSeconds float64
}

// Headless reports that this branch has no queued head. Decisions run INSIDE Pick, where
// `waitingSince` was just synced from the live queue, so the absence is a fact of this pass
// rather than a guess about an old snapshot.
func (c TenureCtx) Headless(m *branch) bool { return m.waitingSince.IsZero() }

// Silent reports the idle test for a branch: no queued head and no activity for S.
func (c TenureCtx) Silent(m *branch) bool {
	return c.Headless(m) && c.Now.Sub(m.lastActivity) > time.Duration(c.IdleSeconds*float64(time.Second))
}

// RotationPolicy answers "may this resident be released now, and why?".
type RotationPolicy interface {
	// Name is the config value that selects the policy.
	Name() string
	// Eligible reports the reason a branch may be rotated ("" and false = it may
	// not). A branch with anything in flight is NEVER eligible.
	Eligible(m *branch, ctx TenureCtx) (string, bool)
}

// FixedQuantum is the ONE tenure rule: Q completed busy periods, or S of silence.
type FixedQuantum struct {
	Q int
}

func (FixedQuantum) Name() string { return "quantum" }

func (p FixedQuantum) Eligible(m *branch, ctx TenureCtx) (string, bool) {
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
	// Exiting reports that the branch is already on its way out (it drains, then
	// goes).
	Exiting func(id string) bool
	// RotationVictim is the TENURE module's demand-driven choice over this engine
	// (nil when no branch is release-eligible under it).
	RotationVictim func() (string, *branch)
	// Silent is the same idle test the tenure policy uses.
	Silent func(m *branch) bool
	// Headless reports whether the branch has no turn wanting to run: no queued head AND no
	// request released and on its way to a pod. A window counts here (it is a turn on its
	// way, so a branch that has one is never "done"), and it does NOT count for the tenure
	// rule's silence test — liveness must never be barred by a window.
	Headless func(m *branch) bool
}

// VictimChooser answers "who leaves when the engine is over its budget?".
type VictimChooser interface {
	Name() string
	Victim(branches []*branch, ctx PressureCtx) (string, *branch)
}

// BudgetValve is the memory-pressure order:
//
//	(1) a rotation victim (the tenure rule already releases this branch);
//	(2) DONE or SILENT — no queued turn, or silent past S;
//	(3) mid-quantum IDLE, oldest last completion first (the safety valve that only
//	    genuine GROWTH pressure reaches: every admission was budget-checked);
//	(4) anything else — a branch with a request RUNNING. It is marked exiting and leaves at
//	    its drain: the mark is progress, because its blocks leave the stay side immediately.
type BudgetValve struct{}

func (BudgetValve) Name() string { return "valve" }

func (BudgetValve) Victim(branches []*branch, ctx PressureCtx) (string, *branch) {
	if ctx.RotationVictim != nil {
		if reason, m := ctx.RotationVictim(); m != nil {
			return reason, m
		}
	}
	var best *branch
	for _, m := range branches {
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
	// class (3): mid-quantum. THE ORDER MATTERS: a branch that has not served a turn
	// in this residency, or that has a queued head right now, is the LAST choice —
	// taking its seat back before its first turn makes it wait another admission cycle
	// (measured: 100 % of the old class-3 victims had a queued head, 53/54 had served
	// ZERO turns in the residency). Only then "oldest last completion", then id.
	for _, m := range branches {
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
	for _, m := range branches {
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
	// Now is the decision pass's clock (never time.Now — a replayed decision stream must be
	// reproducible). "How long this head has waited" is derived from it and the waiter's
	// own oldest arrival, never stored.
	Now time.Time
}

// EntryOrder answers "which waiter is the head?" over the ONE shared queue.
//
// The comparison MUST be a TOTAL order ending in the branch id. That is not a style
// preference: a comparison that falls back to map iteration order produced three
// different traces in four replays of one scenario.
type EntryOrder interface {
	Name() string
	// Head returns the head of the waiter set, or nil when nothing is waiting.
	Head(waiters map[string]*waiter, ctx EntryCtx) *waiter
}

// NewcomerFirst is the default: newcomers before returning branches, then the oldest
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
	if a.returner != b.returner {
		return !a.returner // newcomers first
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
	// engine. Both halves are kept apart on purpose: a head that fits by budget but is held
	// by the valve must WAIT, because evicting frees budget and never frees the valve — a
	// loop that evicted for such a head emptied the seats and hurried nothing.
	Fits bool
	// HasVictim is LAZY (it costs a scan): may the engine evict an eligible
	// incumbent right now?
	HasVictim func() bool
	// HasValveHolder is LAZY: may the engine evict the branch HOLDING its cold-entry
	// charge right now (tenure permitting)? That is the ONE eviction a valve-held head may
	// take, because it frees the valve rather than the budget.
	HasValveHolder func() bool
}

// EngineChooser answers "which engine takes this head?".
type EngineChooser interface {
	Name() string
	// Choose returns the index into cands of the engine to use, or -1.
	Choose(cands []EngineCandidate) int
}

// MostFreeBudget is the default: among the engines where the head FITS (budget AND the
// cold valve), the one with the MOST FREE BUDGET AFTER the seat — free minus this head's
// cost on it; ties → lowest engine id.
//
// "After" rather than "before" is the same principle evaluated on the state the decision
// produces, and it is dedup-aware for free: a fork whose parent's prefix is already
// resident on A costs almost nothing there, so it lands on A unless B is genuinely
// emptier — no separate affinity rule, and nothing to tune.
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
		if best < 0 || betterFit(cands[i], cands[best]) {
			best = i
		}
	}
	if best >= 0 {
		return best
	}
	// The cold valve is holding this head somewhere: WAIT — never evict — because evicting
	// frees budget and never the valve. The one exception is the branch HOLDING the charge
	// on that engine: evicting IT releases the charge and opens the valve, and it is only
	// ever a branch the tenure rule already releases (see valveHolderVictimLocked).
	for i := range cands {
		if !cands[i].BudgetFits {
			continue
		}
		if cands[i].HasValveHolder != nil && cands[i].HasValveHolder() {
			if best < 0 || betterEngine(cands[i], cands[best]) {
				best = i
			}
			continue
		}
		return -1 // the valve holds it and nothing may be evicted for it
	}
	if best >= 0 {
		return best
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

// betterFit is the placement total order among engines the head FITS: the most free budget
// left AFTER the seat, then the engine id.
func betterFit(c, best EngineCandidate) bool {
	if c.FreeBudget-c.CostBlocks != best.FreeBudget-best.CostBlocks {
		return c.FreeBudget-c.CostBlocks > best.FreeBudget-best.CostBlocks
	}
	return c.Engine < best.Engine
}

// betterEngine is the placement total order for the engines that cannot take the head (and
// for naming the engine a valve-held head waits for): most free budget, then engine id.
func betterEngine(c, best EngineCandidate) bool {
	if c.FreeBudget != best.FreeBudget {
		return c.FreeBudget > best.FreeBudget
	}
	return c.Engine < best.Engine
}

// ---- SIZING ------------------------------------------------------------------

// SeatBudget answers how many blocks one engine may seat.
//
// It used to answer the FOOTPRINT too. It no longer does: every cost is now exact, from
// the request's own tokens against the engine's own index (BlockIndex.CostLocked), so
// there is one number and one place that computes it. What is left of the module is the
// one deployment fact it alone knows — the fraction of the engine's measured KV the
// seats may hold.
type SeatBudget interface {
	Name() string
	// BudgetBlocks is one engine's seat budget, from its measured KV capacity
	// (num_gpu_blocks): the deployment's own statement about how much of the engine's
	// KV may be held. The CPU offload tier does not widen it — native offload is
	// write-through, so it adds no seat (the cold valve's cap stays GPU-only for the
	// same reason).
	BudgetBlocks(numBlocks int) int
}

// IndexSizer is the one implementation: fraction × num_gpu_blocks.
type IndexSizer struct {
	Fraction float64
}

func (IndexSizer) Name() string { return "index" }

func (z IndexSizer) BudgetBlocks(numBlocks int) int {
	return int(z.Fraction * float64(numBlocks))
}

// ---- DISPATCH ----------------------------------------------------------------

// DispatchCtx is what a dispatch gate may consult about ONE seated head on ONE engine.
// It is a VALUE, and every field is a number the scheduler already has in hand: the
// check runs per candidate head on every decision pass, so it must neither allocate nor
// reach into an engine.
type DispatchCtx struct {
	// Engine is the engine the head would run on (the branch's seat — a head never
	// moves engines, so this is also the attribution of a hold).
	Engine string
	// Branch is the head's branch id (reporting only; the gate decides on blocks).
	Branch string
	// RunningBlocks is the engine's running set: the dedup-exact union of its in-flight
	// requests' chain blocks, from the EPP's own accounting. It is NEVER an endpoint
	// metric — that is what keeps a gate out of the old per-engine gate's failure class
	// (a metrics clone that latched and locked an engine out).
	RunningBlocks int64
	// ReservedBlocks is the engine's used side OUTSIDE that key set: the charges its
	// in-flight requests hold that carry no chain keys (an unborn dispatch's growth).
	ReservedBlocks int64
	// ChainBlocks is what THIS head would ADD to the running set: the cold part of its
	// chain against that set (dedup), PLUS its own output growth.
	ChainBlocks int64
	// BudgetBlocks is this engine's dispatch budget in blocks (DispatchGate.BudgetBlocks).
	BudgetBlocks int64
	// Idle reports that NOTHING is in flight on the engine — no chain running and no
	// reservation outstanding (a turn admitted in the Pick->PreRequest window IS in
	// flight, even though its exact keys have not arrived yet).
	Idle bool
}

// DispatchGate answers exactly one question: may this seated branch's turn RUN on its
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

// SafetyDetector is the SAFETY seam. The ONE wired value is NoopSafety: the seat
// budget is the whole admission control ("safety: none").
//
// The memory-only net (threshold = the budget's own fraction, live kv scrape) was the
// parked alternative and is REMOVED with its readers: the per-engine holds that asked
// it, the pool score the framework detector returned, and the live-sample staleness
// rule. What remains of the seam is the type: a future net is one implementation of
// this interface plus one config value, and it must bring its own evidence — the trim
// also removed the verdict path (safetyFiredLocked) rather than leaving a hook nothing
// can reach.
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

// class3Less is the mid-quantum (class 3) victim order: UNSTARTED or WANTING branches
// sort last (they have not had the seat long enough for it to be worth reclaiming),
// then the oldest last completion, then the id.
func class3Less(a, b *branch) bool {
	aRaw := a.seatTurns == 0 || !a.waitingSince.IsZero()
	bRaw := b.seatTurns == 0 || !b.waitingSince.IsZero()
	if aRaw != bRaw {
		return !aRaw // a branch that has used its seat goes first
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

// ---- the bundle --------------------------------------------------------------

// modules is the wired set. NewScheduler builds it ONCE from Config; nothing below
// ever consults Config for a policy decision again (no cross-module knobs).
type modules struct {
	sizer    SeatBudget
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
		m.sizer = IndexSizer{Fraction: cfg.GroupBudgetFraction}
	default:
		return m, errValue("sizing", cfg.Sizing, "index")
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
	default:
		return m, errValue("safety", cfg.Safety, "none")
	}
	switch cfg.ExecutionBudget {
	case "", "none":
		m.gate = NoopDispatchGate{}
	default:
		return m, errValue("executionBudget", cfg.ExecutionBudget, "none")
	}
	return m, nil
}
