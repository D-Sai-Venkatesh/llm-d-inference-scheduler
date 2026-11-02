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

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// ---- the policy's parameters ----
//
// One code path for one engine and for N: a single-pod stack is the N=1 case of the
// per-engine state, never a branch of its own. Each knob below is a fact the policy cannot
// work without — a measured deployment value, a wiring fact, or one policy number. A
// selector that changes behaviour lives in the module set (modules.go), one implementation
// per question.
type Config struct {
	// GroupBudgetFraction sizes each engine's SEAT BUDGET from that engine's own
	// measured KV capacity: budgetBlocks = fraction × num_gpu_blocks. It is the one
	// deployment number the policy cannot derive — how much of the engine's KV the seats
	// may hold — and its complement is the cold-entry valve's headroom (see
	// engineGroup.coldCapTokens).
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`

	// Quantum is Q — the fixed tenure, in COMPLETED busy periods, after which a
	// resident becomes rotation-eligible. ONE value, no mode: Q4 (the default)
	// and Q8 are the same code with a different number.
	Quantum int `json:"quantum"`

	// IdleSeconds is S: a resident with no queued turn and no activity for S is
	// rotation-eligible (it is not going to use its seat).
	IdleSeconds float64 `json:"idleSeconds"`
	// (The cold-entry stagger cap is NOT a knob: it is DERIVED per engine as the
	// seat budget's own headroom — floor((1 - groupBudgetFraction) x
	// num_gpu_blocks) blocks x blockSizeTokens — so the valve can only ever spend
	// the margin the budget reserved for it. See engineGroup.coldCapTokens.)
	// InFlightCap bounds one branch's concurrent requests.
	InFlightCap int `json:"inFlightCap"`
	// BlockSizeTokens is the engine's KV block size in tokens (the token
	// processor's value; 64 on the campaign stacks). It is a WIRING FACT, not a
	// knob: it must equal the engine's --block-size so the index's unit matches,
	// and a measured mismatch fails loudly. Required > 0.
	BlockSizeTokens int `json:"blockSizeTokens"`

	// CapacityLivenessSeconds bounds how long an endpoint's measured capacity
	// (num_gpu_blocks) may stay unknown after it is first observed. Past it the
	// plugin logs loudly and raises llm_d_epp_group_capacity_missing: an engine
	// without capacity seats nothing while every other series looks healthy.
	CapacityLivenessSeconds float64 `json:"capacityLivenessSeconds"`

	// InflightReapSeconds bounds the age of a tracked request before reconciliation drops it:
	// the belt for a termination that no callback reported (a record is written when a Pick
	// releases the turn and ended at its settle, its abandon, its branch's exit — so a record
	// that outlives this bound has no owner left).
	//
	// It MUST sit above the client's own request timeout, or the belt cuts a request that is
	// still legitimately running: the campaign harness runs a 9,000 s timeout and every pin
	// sets this to 9,300 s. The default is that same 9,300 s, so a config that forgets the key
	// is safe rather than silently short.
	InflightReapSeconds float64 `json:"inflightReapSeconds"`

	// ---- ONE selector per module (a future idea = one new implementation + one
	// value here; nothing else in the config moves). Validated by buildModules. ----
	//
	//	Sizing       "index"            budget = fraction x num_gpu_blocks (the ONE
	//	                                implementation; a head's cost is not the sizer's
	//	                                answer any more — it is exact, from the engine's
	//	                                own BlockIndex)
	//	EntryOrder   "newcomer-first"   newcomers, then oldest, then id
	//	EngineChoice "most-free-budget" most free budget left AFTER the seat, where the
	//	                                head fits; tie → engine id
	//	Rotation     "quantum"          fixed Q busy periods (plus S of silence)
	//	Victim       "valve"            the valve's memory-pressure order
	//	Safety       "none"            no net (the seat budget IS the control): the ONE
	//	                               value
	//	ExecutionBudget
	//	             "none"            no dispatch gate (an admitted turn runs) — the ONE
	//	                               value: the seam stays (modules.go DispatchGate) but
	//	                               its parked execution-budget implementation, and the
	//	                               running-set accounting it priced, are gone
	Sizing          string `json:"sizing"`
	EntryOrder      string `json:"entryOrder"`
	EngineChoice    string `json:"engineChoice"`
	Rotation        string `json:"rotation"`
	Victim          string `json:"victim"`
	Safety          string `json:"safety"`
	ExecutionBudget string `json:"executionBudget"`
}

// errValue is the fail-closed error for an unknown module value (a typo must never
// silently measure the default twice).
func errValue(knob, got, allowed string) error {
	return fmt.Errorf("%s: invalid value %q (%s)", knob, got, allowed)
}

func defaultConfig() Config {
	return Config{
		GroupBudgetFraction:     0.95,
		Quantum:                 4,
		IdleSeconds:             20,
		InFlightCap:             64,
		BlockSizeTokens:         64,
		CapacityLivenessSeconds: 120,
		InflightReapSeconds:     defaultInflightReapSeconds,
		Sizing:                  "index",
		EntryOrder:              "newcomer-first",
		EngineChoice:            "most-free-budget",
		Rotation:                "quantum",
		Victim:                  "valve",
		Safety:                  "none",
		ExecutionBudget:         "none",
	}
}

// effective fills the zero values with their defaults so the startup echo and
// Snapshot.Config show EFFECTIVE values (the config the policy actually runs).
func (c Config) effective() Config {
	def := defaultConfig()
	if c.GroupBudgetFraction == 0 {
		c.GroupBudgetFraction = def.GroupBudgetFraction
	}
	if c.Quantum == 0 {
		c.Quantum = def.Quantum
	}
	if c.IdleSeconds == 0 {
		c.IdleSeconds = def.IdleSeconds
	}
	if c.InFlightCap == 0 {
		c.InFlightCap = def.InFlightCap
	}
	if c.BlockSizeTokens == 0 {
		c.BlockSizeTokens = def.BlockSizeTokens
	}
	if c.CapacityLivenessSeconds == 0 {
		c.CapacityLivenessSeconds = def.CapacityLivenessSeconds
	}
	if c.InflightReapSeconds == 0 {
		c.InflightReapSeconds = def.InflightReapSeconds
	}
	if c.Sizing == "" {
		c.Sizing = def.Sizing
	}
	if c.EntryOrder == "" {
		c.EntryOrder = def.EntryOrder
	}
	if c.EngineChoice == "" {
		c.EngineChoice = def.EngineChoice
	}
	if c.Rotation == "" {
		c.Rotation = def.Rotation
	}
	if c.Victim == "" {
		c.Victim = def.Victim
	}
	if c.Safety == "" {
		c.Safety = def.Safety
	}
	return c
}

// quantum is the effective Q (one value, never unbounded).
func (c Config) quantum() int { return c.Quantum }

func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.Quantum < 1 {
		return fmt.Errorf("quantum must be >= 1 (Q is the fixed tenure in completed busy periods; the pause arm is gone), got %d", c.Quantum)
	}
	if c.IdleSeconds <= 0 {
		return fmt.Errorf("idleSeconds must be > 0")
	}
	if c.InFlightCap <= 0 {
		return fmt.Errorf("inFlightCap must be > 0")
	}
	if c.BlockSizeTokens <= 0 {
		return fmt.Errorf("blockSizeTokens is required (the token processor's block size; got %d)", c.BlockSizeTokens)
	}
	if c.InflightReapSeconds != 0 && c.InflightReapSeconds < 60 {
		return fmt.Errorf("inflightReapSeconds must be >= 60 (it is the belt over an unobserved termination, not a timeout: the harness's own request timeout is 9000s and every pin sets 9300s; got %v)", c.InflightReapSeconds)
	}
	if c.CapacityLivenessSeconds < 0 {
		return fmt.Errorf("capacityLivenessSeconds must be >= 0 (0 = default; got %v)", c.CapacityLivenessSeconds)
	}
	return nil
}

// defaultInflightReapSeconds is the reap bound a config that omits the key gets: the campaign
// harness's own request timeout (9000 s) plus headroom, so the belt can only ever cut a leak.
const defaultInflightReapSeconds = 9300

// reapSeconds is the effective confirmed-id reap bound (0 → default).
func (c Config) reapSeconds() float64 {
	if c.InflightReapSeconds > 0 {
		return c.InflightReapSeconds
	}
	return defaultInflightReapSeconds
}

// ---- the queue-head snapshot (waiting state comes from the flow-control
// queues at Pick time, never from a hook) ----

// Head is one waiting queue head as seen at Pick: the branch id (the fairness unit AND
// the queue key), the request id, the arrival, and the request's exact prefill chains.
type Head struct {
	// Branch is the group policy's unit: the derived branch id, or "" for
	// anonymous traffic (never tracked, never seated).
	Branch  string
	ReqID   string
	Arrival time.Time
	// Chains is the request's EXACT prefill chains, computed once from the tokens the
	// token producer rendered at arrival (one chain per prompt). An empty set means the
	// request has no tokens: it is never seated (it is dispatched without a seat, like
	// anonymous traffic).
	Chains []Chain
	// FairnessID is the request's flow-control identity — its QUEUE KEY. It is the
	// derived branch id unless the client sent an explicit fairness header, so it is a
	// join key only in the header case; the branch id itself joins to nothing in the
	// harness's own request records (measured 0/768 matches). Every SEAT/EXIT event
	// carries it, and WireID below is the field that actually joins to a request record.
	FairnessID string
	// WireID is the client's OWN id for this request, from its headers
	// (x-correlation-id, else x-request-id): the one field that joins a SEAT/EXIT line to
	// the harness's request records, whatever the queue key turned out to be. Read at Pick
	// (the only unscoped hook) and carried on the branch, so an EXIT names it too.
	WireID string
	// Returner reports whether this request is NOT its branch's first turn — the
	// fairness class, from branch identity's arrival record. It is phrased as the
	// exception on purpose: the zero value must be the NEWCOMER (a head whose arrival
	// record is missing is a branch with no history we know of, and branch identity
	// defaults the same way), so a caller that forgets the field cannot demote a
	// first-turn branch behind incumbents.
	Returner bool
}

// ---- per-branch scheduling state ----

// branch is one seated residency: a record that exists exactly as long as the
// branch holds a seat on ONE engine (the seat IS the binding — there is no
// pre-seat homing and no post-exit placement claim).
type branch struct {
	id string
	// engine is the engineGroup id this branch is SEATED on — the same value as
	// the identity-level binding, kept on the record so a dump shows the seat
	// without a second lookup.
	engine string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// seatTurns counts settled turns in this residency (the residency histogram's
	// unit).
	seatTurns int
	// inflight and pending are DERIVED each decision point from the request map
	// (derive-don't-accumulate): a leaked increment cannot persist past a
	// reconciliation. They are kept on the record so a decision reads a count, not
	// the whole map: inflight = the branch's requests at a pod, pending = the
	// requests it RELEASED that have not reached one yet (the "on its way" window).
	//
	// The window is the one fact that says the branch is still busy: without it a
	// cold-charge derivation would free the charge of a prefill that is about to
	// start, and rotation could evict the branch that is on its way.
	inflight int
	pending  int
	// busyPeriods counts completed busy periods: the branch's in-flight count reaching
	// zero and staying there, which — one branch, one chain — is the whole of its work
	// draining. It is the rotation quantum's unit.
	busyPeriods int
	// lastCompletion is the last time the branch's in-flight count returned to zero.
	lastCompletion time.Time
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this branch's outstanding cold-entry charge in TOKENS against
	// the stagger cap; released EXACTLY once, at the first of: prefill
	// complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// waitingSince is the branch's oldest queued head: the wait its victims and its SEAT
	// event are measured from, kept OLDEST across fan-out re-arrivals.
	waitingSince time.Time
	// headChains / headReturner are the LAST queued head observed for this branch, from the
	// queue view of a decision pass. They exist for ONE consumer: exitLocked's re-queue.
	// Re-queueing with the chain of the turn that just RAN charges the next turn the wrong
	// cost (the seat path prices the head's own chain), and forcing returner=true told the
	// entry order a branch was an incumbent when its queued head might be a newcomer.
	headChains   []Chain
	headReturner bool
	// exitingSince is when this branch was marked exiting (the mid-flight class): the
	// age of a retiree is the time between "nothing more will be admitted for it" and
	// "its blocks are off the budget", i.e. how long the engine pays for a seat it has
	// already written off. Zero on a branch that is not exiting (a fresh residency is a
	// fresh struct, so the flag never has to be cleared).
	exitingSince time.Time
	// fairnessID is the last fairness id (queue key) this branch was seen with —
	// the queue key its SEAT/EXIT events name (see Head.FairnessID).
	fairnessID string
	// wireID is the last client request id this branch was seen with: the field a
	// SEAT/EXIT line joins a request record on (see Head.WireID).
	wireID string
}

// busy reports whether the branch has work in progress: a request at a pod, or a LIVE
// released window (a turn that a Pick released and PreRequest has not yet brought to a pod).
//
// THE WINDOW COUNTS AS IN FLIGHT EVERYWHERE — tenure eligibility, the engine-idle rule,
// every victim class, the drain, the cold-charge derivation, the in-flight cap. That is not
// a convenience: the window is the COMMON case, not a corner. Every dispatch passes through
// it and Pick runs at ~1 kHz, so a decision almost always lands inside one, and a branch on
// its way to the pod looks IDLE to every rule that ignores the window. Measured, at N=1 and
// N=2: the engine-idle rule evicted such a branch and the observed seat put it straight back
// (used 1200 against a 950 budget); a marked branch was DRAINED before its request ran (used
// 1500 against 1100); a quantum-complete or silent branch was rotated out mid-window.
//
// The window is bounded by S (Config.IdleSeconds, the same silence bound the tenure rule
// uses), and m.pending counts only the windows that are still live: one that outlives S is
// presumed dead, stops counting, and its branch becomes evictable on demand like any silent
// branch — which is also what frees the cold charge that would otherwise close the valve.
// Liveness is evaluated in the decision pass (reapRequestsLocked): no timer, no new
// constant.
func (m *branch) busy() bool { return m.inflight > 0 || m.pending > 0 }

// reqRec is ONE tracked request's record, and the scheduler's only memory of a request
// that is not in a queue. It spans both halves of the request's life outside the queue:
// the RELEASED window (a Pick released it, PreRequest has not reached a pod yet — pod is
// zero) and the in-flight period (pod is set).
//
// It is keyed by request id because that is the ONLY thing the later hooks hand back.
// The framework scopes a plugin's request-attribute access to what the plugin declares,
// and an undeclared read RESOLVES AS ABSENT — so a branch id another plugin wrote at
// RequestHeader is not readable at Filter/PreRequest/ResponseBody at all. Resolving the
// branch from this record instead makes the policy's identity independent of the scoping
// model (and of whether the flow-control Pick path is scoped: it is not, so Pick still
// reads the arrival attributes, and it is the only hook that does).
//
// chains is the request's exact prefill chain as the Pick released it, so the observed
// seat at PreRequest can charge the ENGINE the same chain the seat did, without reading
// the request's tokens a second time.
type reqRec struct {
	// branch is the branch that released the request (never empty: anonymous traffic
	// is not tracked).
	branch string
	// chains is the prefill chain the release was charged with.
	chains []Chain
	// engine is the engine the request landed on, once an observation named one.
	engine string
	// released is the Pick that released the request.
	released time.Time
	// pod is when the request reached a pod (PreRequest); zero while it is pending.
	pod time.Time
}

// waiter is one head in the SHARED waiter queue. A waiter is not seated, so it
// belongs to no engine: its admission cost is a per-engine VECTOR (what it would
// cost on each engine, from that engine's own index) and the engine is chosen at
// SEAT time (admission.go).
type waiter struct {
	id    string
	since time.Time // the OLDEST arrival: a fan-out re-arrival does not reset it
	// returner is the fairness class: this branch's queued turn is NOT its first
	// one. It comes from the head (Head.Returner), i.e. from branch identity's
	// arrival record — never from a history the scheduler keeps.
	returner bool
	// chains is the head's exact prefill chain set. Every cost the seat path takes is
	// computed from it against the engine's own index, so a waiter needs no cost of its
	// own: an engine that does not hold the branch's blocks is costed as cold, which is
	// the truth about that engine.
	chains []Chain
	// fairnessID rides with the head (see Head.FairnessID) so a SEAT event can name
	// the queue key even before the branch has a record, and wireID rides with it for the
	// join to the client's own request record (see Head.WireID).
	fairnessID string
	wireID     string
}

// ---- the per-engine group ----
//
// Everything that is a fact about ONE engine's HBM lives here: its block index (dedup is per
// engine — a block on A is worthless to B), its budget, its measured capacity and liveness,
// its seats, its cold-entry charge and its counters. One engine and N engines are the same
// struct and the same decisions; there is no pooled group and no single-engine case.
//
// What is NOT here (identity, because a branch's identity is a property of the
// client, not of the seat): the fairness class (it arrives on the request, from
// branch identity), the branch→engine bindings, and the SHARED waiter queue.
type engineGroup struct {
	// id is the endpoint id the EPP discovered.
	id string

	index *BlockIndex // attached at creation (block size from config)
	// budgetBlocks is what this engine may seat, in blocks: the SIZING module's answer
	// for THIS engine's measured capacity. 0 until THIS engine's num_gpu_blocks
	// arrives.
	budgetBlocks int
	numBlocks    int
	// blockSizeMismatch is the loud per-engine wiring failure (a measured
	// CacheBlockSize that differs from the config's BlockSizeTokens).
	blockSizeMismatch error

	branches map[string]*branch
	// coldInFlight is this engine's outstanding cold-entry charge against its own stagger
	// cap, so N engines allow N concurrent cold prefills and no more.
	coldInFlight int64

	counters struct {
		served int64 // dispatches served on this engine
		// entryWaits / reentryWaits are THIS engine's seat-wait samples (the per-engine
		// wait ladder a dump reports).
		entryWaits   []float64
		reentryWaits []float64
		// forcedSeats counts seats that were taken on an OBSERVED engine rather than on a
		// fit: an affinity reconciliation, and a request whose branch held no seat when it
		// reached the pod (seatObservedLocked). Enforcement catches up on the next pass.
		forcedSeats int64
		exits       map[string]int64
	}
}

// coldCapTokens is THIS engine's cold-ENTRY stagger cap, in tokens: the seat
// budget's own headroom, floor((1 - fraction) x num_gpu_blocks) blocks x blockSize.
// The valve staggers cold prefills INSIDE the margin the budget reserved for them,
// which is why it needs no value of its own — with the budget at 0.95 of 40,183
// blocks and 64-token blocks the cap is 2009 blocks = 128,576 tokens.
//
// IT READS num_gpu_blocks AND NOTHING ELSE: the valve staggers GPU allocation bursts, and
// a CPU tier cannot absorb one (a cold prefill allocates HBM whatever the tier holds).
//
// 0 means "no measured capacity yet", and the places that call this are already
// behind the capacity check (an engine without capacity takes no seats).
func (g *engineGroup) coldCapTokens(fraction float64, blockSize int) int64 {
	if g.numBlocks <= 0 || blockSize <= 0 {
		return 0
	}
	headroomBlocks := int64((1 - fraction) * float64(g.numBlocks))
	return headroomBlocks * int64(blockSize)
}

func newEngineGroup(id string, blockSizeTokens int) (*engineGroup, error) {
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return nil, err
	}
	g := &engineGroup{id: id, index: idx,
		branches: map[string]*branch{}}
	g.counters.exits = map[string]int64{}
	return g, nil
}

// engineLabel is the metrics/event label value for an engine id.
func engineLabel(id string) string { return id }

// Scheduler is the group scheduler state machine over an engine set. The ONE
// piece of state that is deliberately global is the SHARED waiter queue: a waiter
// is not homed, so it belongs to no engine until an engine seats it. Everything
// else that is per engine lives in engineGroup; everything per branch (its class, its
// seat engine, its own clock) lives here.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// mod is the wired module set (modules.go) — the ONE place a policy decision
	// comes from.
	mod modules
	// observers are the optional metric hooks (nil in tests).
	onExit       func(engine, reason string, seatTurns int)
	onEntry      func(engine, kind string, waitSeconds float64)
	onServe      func(engine string)
	onEvent      func(Event)
	onRehome     func(branch, from, to string)
	onEngineGone func(engine string, branches int)
	// onAffinityViolation fires for a request that LANDED on an engine other than the one it
	// was charged to, when the violation is found at the SETTLE (the observation at
	// PreRequest already rebinds, and reports through onRehome). Both sites must reach the
	// counter: the settle-side one is the only signal when the placement path silently
	// ignored the pin, and the metric's whole claim is that it must be 0.
	onAffinityViolation func(branch, charged, settled string)
	// onBind / onUnbind are the PLACEMENT-PIN hooks: they fire when a branch's
	// seat (and therefore its engine binding) is created or dropped, so the
	// plugin's lock-light Filter table can follow the seats without ever touching
	// s.mu. nil in tests.
	onBind   func(branch, engine string)
	onUnbind func(branch string)
	// onRelease / onForget are the REQUEST-IDENTITY hooks: they fire when a Pick
	// releases a request (branch, request id) and when its record ends (request id),
	// so the Filter can resolve a dispatching request to its branch by request id
	// alone — under its own lock, never s.mu (Pick holds s.mu at ~1 kHz). nil in
	// tests.
	onRelease func(branch, reqID string)
	onForget  func(reqID string)
	// onColdValveHeld fires when the cold-entry valve HOLDS a head that would fit an
	// engine's budget. The head WAITS (the valve opens when that engine's cold prefill
	// produces its first token); nothing is evicted for it. This is the sizing signal:
	// how often the stagger makes demand wait.
	onColdValveHeld func(engine string)
	// groups holds one engineGroup per DISCOVERED engine, keyed by endpoint id.
	groups map[string]*engineGroup
	// groupOrder is the sorted key list (deterministic iteration: every
	// cross-engine decision is taken in id order, never map order).
	groupOrder []string

	blockSize int

	// bindings is the branch→engine association. It IS the seat: created when the
	// branch is seated, dropped the moment it exits. An unseated branch has no
	// placement opinion at all.
	bindings map[string]string

	// waiters is THE SHARED WAITER QUEUE: one entry per branch with a queued head
	// and no seat. Whichever engine frees a seat (or has room) takes the head from
	// here — the queue is where waiting is pooled, which is what makes N engines
	// one admission system instead of N.
	waiters map[string]*waiter

	// reqs is EVERY tracked request, keyed by request id: the record written when a
	// Pick releases a turn and dropped at its settle, its abandon, its reap, or its
	// branch's exit. It is the scheduler's OWN answer to "which branch is this
	// request?" for every hook the framework scopes (see reqRec).
	reqs map[string]*reqRec

	// now is the virtual clock of the CURRENT decision pass (set at the top of
	// every public decision entry point). Nothing here calls time.Now: a replayed
	// decision stream has to be reproducible.
	now    time.Time
	events []Event

	counters struct {
		entryWaits   []float64
		reentryWaits []float64
		// emptyReqIDs / straySettles count events the policy cannot act on: a request with
		// no id (nothing can name it) and a settle for an id that was never in flight.
		// Both read 0 with the gateway, which sends a real id end to end.
		emptyReqIDs  int64
		straySettles int64
		// noCost counts heads that arrived without tokens, so could not be seated (the
		// render failed; branch identity counts that too). They are served FCFS, last.
		noCost int64
		// affinityViolations counts requests that LANDED on an engine other than
		// the one they were seated on (CORRECTNESS: the observed endpoint wins, so
		// each violation is repaired by moving the seat; a nonzero steady rate
		// means the placement path is not honouring the pin).
		affinityViolations int64
		rebinds            int64
		engineGone         int64
		exits              map[string]int64
		seatTurns          []int
		refillShare        float64
		uncachedTok        int64
		totalTok           int64
		served             int64
		fcfsServed         int64
		// eventsDropped counts events the SEAT/EXIT ring wrapped past. The ring is
		// bounded (memory), so on a long window the oldest events fall out of the
		// DumpState tail; the stdout stream is the durable copy. A non-zero value here
		// means "the dump's RecentEvents does not cover the window", which must be
		// visible instead of inferred.
		eventsDropped int64
	}
}

// NewScheduler builds the state machine. The block index attaches per engine as
// the engine is discovered, with the block size from the token processor's config.
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg = cfg.effective()
	mod, err := buildModules(cfg)
	if err != nil {
		return nil, err
	}
	s := &Scheduler{cfg: cfg, mod: mod, blockSize: cfg.BlockSizeTokens,
		groups:   map[string]*engineGroup{},
		bindings: map[string]string{},
		reqs:     map[string]*reqRec{},
		waiters:  map[string]*waiter{}}
	s.counters.exits = map[string]int64{}
	return s, nil
}

// groupForLocked returns (creating) the group for an engine id.
func (s *Scheduler) groupForLocked(id string) (*engineGroup, error) {
	if g, ok := s.groups[id]; ok {
		return g, nil
	}
	g, err := newEngineGroup(id, s.blockSize)
	if err != nil {
		return nil, err
	}
	s.groups[id] = g
	s.rebuildGroupOrderLocked()
	return g, nil
}

func (s *Scheduler) rebuildGroupOrderLocked() {
	ids := make([]string, 0, len(s.groups))
	for id := range s.groups {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	s.groupOrder = ids
}

// existingGroupLocked returns the group for an engine id WITHOUT creating it
// (a settle for an already-torn-down engine must not resurrect the group).
func (s *Scheduler) existingGroupLocked(id string) *engineGroup { return s.groups[id] }

// groupOfLocked is the branch's SEAT engine (nil when the branch is unseated).
func (s *Scheduler) groupOfLocked(branch string) *engineGroup {
	id, ok := s.bindings[branch]
	if !ok {
		return nil
	}
	return s.groups[id]
}

// activeGroupsLocked is the deterministic engine list every decision pass
// iterates: every discovered engine, in id order.
func (s *Scheduler) activeGroupsLocked() []*engineGroup {
	out := make([]*engineGroup, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		out = append(out, s.groups[id])
	}
	return out
}

// SetNumBlocksFor attaches ONE engine's measured capacity. Capacity is per engine
// by construction: two engines each holding 40,183 blocks is 2 × HBM, not one
// pool of 40,183.
func (s *Scheduler) SetNumBlocksFor(engine string, blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, err := s.groupForLocked(engine)
	if err != nil {
		return err
	}
	if blockSizeTokens > 0 && blockSizeTokens != s.blockSize {
		g.blockSizeMismatch = fmt.Errorf(
			"groupsched[%s]: measured CacheBlockSize %d != configured BlockSizeTokens %d (the token processor's value); refusing the capacity",
			engineLabel(g.id), blockSizeTokens, s.blockSize)
		return g.blockSizeMismatch
	}
	if g.numBlocks > 0 || numBlocks <= 0 {
		return nil // first capacity wins, PER ENGINE
	}
	g.numBlocks = numBlocks
	g.budgetBlocks = s.mod.sizer.BudgetBlocks(numBlocks)
	return nil
}

// HasCapacity reports whether ANY engine has a measured num_gpu_blocks (the
// pre-capacity FCFS pass-through's whole-stack condition). Seating itself is
// per engine and needs THAT engine's capacity.
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, g := range s.groups {
		if g.numBlocks > 0 {
			return true
		}
	}
	return false
}

// HasCapacityFor reports whether THAT engine's capacity has arrived — the
// per-endpoint liveness test (an engine without metrics serves nothing while
// looking healthy).
func (s *Scheduler) HasCapacityFor(engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, ok := s.groups[engine]
	return ok && g.numBlocks > 0
}

// GroupLabel resolves an endpoint key to the label the metrics and events carry for it.
func (s *Scheduler) GroupLabel(engine string) string {
	return engineLabel(engine)
}

// HeterogeneousEngines reports the engine ids whose measured num_gpu_blocks
// deviates from the FIRST measured engine by more than tolFrac. A heterogeneous
// pool must not silently inherit another engine's budget (the plugin logs this
// loudly; it is a wiring fact, not a tuning knob).
func (s *Scheduler) HeterogeneousEngines(tolFrac float64) map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heterogeneousLocked(tolFrac)
}

func (s *Scheduler) heterogeneousLocked(tolFrac float64) map[string]int {
	out := map[string]int{}
	var base int
	for _, id := range s.groupOrder {
		g := s.groups[id]
		if g.numBlocks <= 0 {
			continue
		}
		if base == 0 {
			base = g.numBlocks
			continue
		}
		diff := float64(g.numBlocks-base) / float64(base)
		if diff < 0 {
			diff = -diff
		}
		if diff > tolFrac {
			out[id] = g.numBlocks
		}
	}
	return out
}

// BlockSizeMismatches is every mismatched engine, in id order. ONE engine's misconfigured
// block size must not be able to hide behind another's: the startup warning names them all
// rather than the first, and the dump carries each engine's own error.
func (s *Scheduler) BlockSizeMismatches() map[string]error {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]error{}
	for _, id := range s.groupOrder {
		if err := s.groups[id].blockSizeMismatch; err != nil {
			out[engineLabel(id)] = err
		}
	}
	return out
}

// ---- Pick: the waiting set is the queue-head snapshot ----

// PickFromHeads runs one dispatch decision against the current queue heads and
// returns the index into heads of the chosen request, or -1.
//
// The engine is chosen INSIDE (see PickFromHeadsEngine); the plugin needs only the
// head, because the engine the request actually landed on arrives authoritatively at
// PreRequest (pickedEndpoint).
func (s *Scheduler) PickFromHeads(heads []Head, now time.Time) int {
	_, i := s.PickFromHeadsEngine(heads, now)
	return i
}

// PickFromHeadsEngine is PickFromHeads with the engine it served from ("" for an
// anonymous head). One pass, in this order: derive and reap, SYNC the shared waiter
// queue, ENFORCE each engine's budget, ROTATE/SEAT waiters, then DISPATCH — and the
// dispatch writes the chosen turn's exact chain into the budget in the same critical
// section. A seated branch's growth is therefore charged when its turn goes out, never
// before: the seat is an admission right, and growth that no longer fits is enforced
// against the group on the next pass.
//
// The pass is written for N engines as THE code and for one engine as its N=1
// case: nothing below is conditioned on how many engines exist.
//
// Within an engine the branch heads go OLDEST FIRST (a resident whose turn is due
// dispatches before its own later turns). ACROSS engines the order is the engine
// whose own residents are waiting most per seat — a dispatch-order tie-break, not
// a placement key — then the head's arrival, then the engine id, so the decision is
// total and reproducible.
func (s *Scheduler) PickFromHeadsEngine(heads []Head, now time.Time) (string, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	s.reapRequestsLocked(now)
	s.drainExitingLocked(now)
	s.syncWaitersLocked(heads, now)
	s.deriveColdChargesLocked()
	groups := s.activeGroupsLocked()
	for _, g := range groups {
		s.enforceBudgetLocked(g, now)
	}
	s.rotateLocked(now)

	type cand struct {
		i      int
		h      Head
		engine string
	}
	perEngine := make(map[string][]cand, len(groups))
	for i, h := range heads {
		if h.Branch == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		g := s.groupOfLocked(h.Branch)
		if g == nil {
			continue // unseated: its head is ADMISSION, handled by rotateLocked
		}
		m, seated := g.branches[h.Branch]
		if !seated || g.index.ExitingLocked(h.Branch) {
			continue
		}
		if m.inflight+m.pending >= s.cfg.InFlightCap {
			continue
		}
		perEngine[g.id] = append(perEngine[g.id], cand{i: i, h: h, engine: g.id})
	}
	type best struct {
		c     cand
		score float64
	}
	var winner *best
	for _, g := range groups {
		cands := perEngine[g.id]
		if len(cands) == 0 {
			continue
		}
		sort.SliceStable(cands, func(x, y int) bool { return cands[x].h.Arrival.Before(cands[y].h.Arrival) })
		pick := 0 // FIFO: the candidates are oldest-first, so the head is the first one
		chosen := &cands[pick]
		score := s.queuedPerSeatLocked(g)
		if winner == nil || score > winner.score ||
			(score == winner.score && chosen.h.Arrival.Before(winner.c.h.Arrival)) ||
			(score == winner.score && chosen.h.Arrival.Equal(winner.c.h.Arrival) && g.id < winner.c.engine) {
			winner = &best{c: *chosen, score: score}
		}
	}
	if winner != nil {
		g := s.groups[winner.c.engine]
		m := g.branches[winner.c.h.Branch]
		// THE DISPATCHED TURN'S CHAIN IS CHARGED HERE, as the pass's winner — in this
		// pass's critical section, so the charge is exact and atomic with the decision.
		//
		// A SEAT IS AN ADMISSION RIGHT: a seated branch's next turn runs even when its
		// prompt grew past what the engine has left, and the overage that growth creates
		// is enforcement's business on the NEXT pass, with the grower in flight. Charging
		// the growth at sync instead would let a branch's own growth condemn it before its
		// turn ever ran — and that buys nothing: the branch re-queues demanding the same
		// blocks, re-enters through the cold valve, and loses a cycle.
		if len(winner.c.h.Chains) > 0 {
			g.index.GrowLocked(m.id, winner.c.h.Chains)
		}
		// The request is RELEASED: it is neither queued nor in flight until PreRequest
		// records it, and the record written here is what every later hook resolves it
		// by. No synthetic id is minted. The record ends at its settle, its abandon, its
		// branch's exit — or the reap, which is what bounds the map on the paths where no
		// hook ever arrives.
		s.releaseLocked(m, winner.c.h.ReqID, winner.c.h.Chains, now)
		m.lastActivity = now
		g.counters.served++
		s.counters.served++
		if s.onServe != nil {
			s.onServe(engineLabel(g.id))
		}
		return g.id, winner.c.i
	}
	// No branch head is dispatchable — serve the OLDEST anonymous head (FCFS pass-through,
	// the siblings' behaviour; anonymous traffic is never tracked, never seated, never
	// starved). A head with no tokens is served the same way: it cannot be costed, and
	// refusing it would starve it forever. The order is by arrival, not by the order the
	// queues were iterated in — a fallback that took whichever such head came first would
	// serve an arbitrary one of them.
	fcfs := -1
	for i, h := range heads {
		if h.Branch != "" && len(h.Chains) > 0 {
			continue // a branch head that could be costed is not this pass's business
		}
		if fcfs < 0 || h.Arrival.Before(heads[fcfs].Arrival) {
			fcfs = i
		}
	}
	return "", fcfs
}

// FCFSPick returns the oldest head (the pre-capacity pass-through).
func FCFSPick(heads []Head) int {
	best := -1
	var bestArrival time.Time
	for i, h := range heads {
		if best < 0 || h.Arrival.Before(bestArrival) {
			best, bestArrival = i, h.Arrival
		}
	}
	return best
}

// FCFSServed counts a pre-capacity dispatch: it belongs to no engine (a
// pre-capacity stack has none to attribute it to), so it is only counted.
func (s *Scheduler) FCFSServed() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
}

// syncWaitersLocked reconciles the SHARED waiter queue with the queue-head
// snapshot: a branch with a queued head and no seat joins it, a branch whose head
// vanished leaves it, and every seated branch's waiting-since tracks its oldest
// queued head (the victim preference's input).
func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Branch == "" {
			continue
		}
		if len(h.Chains) == 0 {
			// no tokens, no cost: the request cannot be seated or charged, so it is served
			// like anonymous traffic (FCFS, last) and never enters the shared queue. It is
			// counted, because a run that starts seeing these is losing its render.
			s.counters.noCost++
			continue
		}
		present[h.Branch] = h.Arrival
		if g := s.groupOfLocked(h.Branch); g != nil {
			if m, seated := g.branches[h.Branch]; seated {
				if h.FairnessID != "" && m.fairnessID != h.FairnessID {
					m.fairnessID = h.FairnessID
				}
				if h.WireID != "" {
					m.wireID = h.WireID
				}
				if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
					m.waitingSince = h.Arrival
				}
				// the head's own cost and class, for the re-queue an exit may have to make
				m.headChains, m.headReturner = h.Chains, h.Returner
				m.lastActivity = now
				continue
			}
		}
		// unseated: its demand is ADMISSION, so it waits in the one shared queue
		w, ok := s.waiters[h.Branch]
		if !ok {
			w = &waiter{id: h.Branch}
			s.waiters[h.Branch] = w
		}
		// the class rides with the HEAD: branch identity knows whether this is the
		// branch's first turn
		w.returner = h.Returner
		w.fairnessID = h.FairnessID
		w.wireID = h.WireID
		w.chains = h.Chains
		// the oldest arrival is what the wait is measured from: a fan-out re-arrival
		// does not reset it
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
	}
	// vanished: branches clear their waiting marker; waiters leave
	for _, g := range s.activeGroupsLocked() {
		for id, m := range g.branches {
			if _, ok := present[id]; !ok {
				m.waitingSince = time.Time{}
			}
		}
	}
	for id := range s.waiters {
		if _, ok := present[id]; !ok {
			delete(s.waiters, id)
		}
	}
}

// reconcileEngineLocked is the observation winning: a seated branch whose request lands on
// another engine is MOVED to it (one branch's correction instead of unbounded accounting
// drift — the index is an optimistic model of where blocks ARE, and a stale binding drifts
// permanently and silently).
func (s *Scheduler) reconcileEngineLocked(id, observed string, now time.Time) *engineGroup {
	from := s.groupOfLocked(id)
	if from == nil || from.id == observed {
		return from
	}
	to, err := s.groupForLocked(observed)
	if err != nil || to == nil {
		return from
	}
	m := from.branches[id]
	if m == nil {
		s.bindings[id] = to.id
		return to
	}
	s.counters.affinityViolations++
	s.counters.rebinds++
	from.coldInFlight = max(from.coldInFlight-m.coldEntry, 0)
	chains := from.index.ChainsLocked(id)
	from.index.UnseatLocked(id) // the blocks are NOT on this engine (observation wins)
	delete(from.branches, id)
	to.branches[id] = m
	m.engine = to.id
	s.bindings[id] = to.id
	to.coldInFlight += m.coldEntry
	// the engine is serving this branch, so its blocks ARE there: seat them without a fit
	// test and let enforcement reconcile the budget (an over-budget counter is
	// recoverable; a branch absent from the engine that is serving it is not).
	to.index.SeatForceLocked(id, chains)
	to.counters.forcedSeats++
	s.recordEvent("REBIND", id, fmt.Sprintf("from=%s to=%s", engineLabel(from.id), engineLabel(to.id)), now)
	if s.onRehome != nil {
		s.onRehome(id, from.id, to.id)
	}
	if s.onBind != nil {
		s.onBind(id, to.id) // the placement pin follows the seat
	}
	return to
}

// ---- the request map: ONE record per tracked request ----
//
// A request leaves the queue at Pick and comes back to the scheduler only through hooks
// the framework SCOPES: Filter and PreRequest get a request whose attribute access is
// confined to what this plugin declares, and an undeclared read resolves as absent. The
// branch id another plugin wrote at RequestHeader is therefore not readable there — and
// declaring it would fail startup (the DAG rejects a flow-control consumer of a
// request-control producer). So the branch a later hook belongs to is resolved from THIS
// map, by request id, which is the one thing every hook still carries.
//
// The map is also the released window and the in-flight set, folded into one record per
// request rather than two branch-level fields: the window and the in-flight entry are the
// SAME request, they differ only in whether PreRequest has reached a pod yet.

// releaseLocked records the turn a Pick just released.
func (s *Scheduler) releaseLocked(m *branch, reqID string, chains []Chain, now time.Time) {
	if m == nil {
		return
	}
	if reqID == "" {
		// An id-less request cannot be tracked: nothing names it, so no window opens and
		// the branch is not held busy by it. It is counted, because a run that starts
		// seeing these has lost the id the accounting joins on.
		s.counters.emptyReqIDs++
		return
	}
	if _, tracked := s.reqs[reqID]; tracked {
		// a RE-release of a request already tracked (a retry that re-entered the queue):
		// replacing the record must correct the counts it was holding, or the branch is
		// charged twice for one turn.
		s.dropRequestLocked(reqID, now)
	}
	s.reqs[reqID] = &reqRec{branch: m.id, chains: chains, released: now}
	m.pending++
	if s.onRelease != nil {
		s.onRelease(m.id, reqID)
	}
}

// windowLive reports whether a released record's window is still inside S: the branch is
// presumed to be using its seat for that long and evictable after it (see branch.busy).
// The clock is the decision pass's, so a replayed stream decides identically.
func (s *Scheduler) windowLive(rec *reqRec, now time.Time) bool {
	if rec == nil || !rec.pod.IsZero() {
		return false
	}
	return now.Sub(rec.released) <= time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// atPodLocked closes the released window: the request is running on an engine.
func (s *Scheduler) atPodLocked(rec *reqRec, m *branch, engine string, now time.Time) {
	rec.engine = engine
	if !rec.pod.IsZero() {
		return // a repeated PreRequest: the record is already in flight
	}
	rec.pod = now
	if m != nil && rec.branch == m.id {
		m.pending = max(m.pending-1, 0)
		m.inflight++
	}
}

// dropRequestLocked deletes one record and re-derives the counts of the branch that owned
// it. Every way a record ends (settle, abandon, the sweep on its branch's exit) goes through
// here, so the counts cannot be updated on one path and missed on another.
//
// A window that already outlived S was never counted (branch.busy counts live windows
// only), so ending it decrements nothing.
func (s *Scheduler) dropRequestLocked(reqID string, now time.Time) {
	rec, ok := s.reqs[reqID]
	if !ok {
		return
	}
	delete(s.reqs, reqID)
	if g := s.groupOfLocked(rec.branch); g != nil {
		if m := g.branches[rec.branch]; m != nil {
			if rec.pod.IsZero() {
				if s.windowLive(rec, now) {
					m.pending = max(m.pending-1, 0)
				}
			} else {
				m.inflight = max(m.inflight-1, 0)
			}
		}
	}
	if s.onForget != nil {
		s.onForget(reqID)
	}
}

// sweepBranchLocked drops every record a branch owns: they name a seat that no longer
// exists, so no later hook could act on them (and a settle for one is a stray, not a
// mis-charged engine). The ruling that keeps the map bounded: a dead pre-PreRequest
// request costs one entry until its branch leaves.
func (s *Scheduler) sweepBranchLocked(id string, now time.Time) {
	for reqID, rec := range s.reqs {
		if rec.branch == id {
			s.dropRequestLocked(reqID, now)
		}
	}
}

// branchOfRequestLocked is the ONE resolution the scoped hooks go through: the branch a
// request id names, or "" when nothing tracks it (anonymous traffic, a pre-capacity FCFS
// dispatch, a request whose branch has left, or one that no Pick released).
func (s *Scheduler) branchOfRequestLocked(reqID string) string {
	if reqID == "" {
		return ""
	}
	if rec, ok := s.reqs[reqID]; ok {
		return rec.branch
	}
	return ""
}

// oldestAtPodLocked returns the branch's oldest in-flight record (its id), for the one
// caller that cannot name a request: an id-less settle or abandon. "" when there is none.
func (s *Scheduler) oldestAtPodLocked(branch string) string {
	oldest, at := "", time.Time{}
	for reqID, rec := range s.reqs {
		if rec.branch != branch || rec.pod.IsZero() {
			continue
		}
		if oldest == "" || rec.pod.Before(at) {
			oldest, at = reqID, rec.pod
		}
	}
	return oldest
}

// oldestPendingLocked returns the branch's oldest released-but-not-at-a-pod record.
func (s *Scheduler) oldestPendingLocked(branch string) string {
	oldest, at := "", time.Time{}
	for reqID, rec := range s.reqs {
		if rec.branch != branch || !rec.pod.IsZero() {
			continue
		}
		if oldest == "" || rec.released.Before(at) {
			oldest, at = reqID, rec.released
		}
	}
	return oldest
}

// RequestBranch is the scheduler's own request→branch resolution, for a caller that only
// has the request id (the scoped-hooks test asserts every hook agrees with it; nothing in the
// hooks reads a request attribute to get the same answer).
func (s *Scheduler) RequestBranch(reqID string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b := s.branchOfRequestLocked(reqID)
	return b, b != ""
}

// TrackedRequests is how many request records exist. Every record ends at a settle, an
// abandon, a reap, or its branch's exit, so a growing count on a steady run is a leak.
func (s *Scheduler) TrackedRequests() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.reqs)
}

// ---- PreRequest (post-dispatch): the in-flight record, and the observed engine ----

// Dispatched records a released request at PreRequest without an observed endpoint (the
// single-engine form): the branch's seat decides the engine.
func (s *Scheduler) Dispatched(id, reqID string, now time.Time) {
	s.DispatchedOn(id, reqID, "", nil, now)
}

// DispatchedOn is the endpoint-aware Dispatched (a unit fixture's form).
func (s *Scheduler) DispatchedOn(id, reqID, engine string, chains []Chain, now time.Time) {
	s.DispatchedOnUsage(id, reqID, engine, chains, now)
}

// DispatchedOnUsage records a released request at PreRequest on the endpoint the
// framework ACTUALLY picked. Nothing is charged here: the seat charged the exact
// chains at Pick, so this call only confirms the record (the drain's input) and,
// when the observation disagrees with the seat, moves the seat.
//
// id is the branch, and it is normally EMPTY: the plugin cannot read a branch id out of a
// scoped request, so it passes the request id and the branch comes from the record the
// Pick released. A fixture that knows the branch passes it explicitly (and then a missing
// record is not an error — the fixture's branch is the authority).
//
// engine=="" means "the branch's SEAT" (no observation); chains is that request's own
// prefill chain, and it is used for ONE thing: a branch that holds no seat at all when
// its request arrives is seated by the observation (seatObservedLocked). It defaults to
// the chain the release recorded, which is the same chain.
//
// A branch with NO seat and NO observation stays UNTRACKED, and is never given a
// binding: a binding to an engine that no decision visits is a branch that is
// permanently unseated while the dump counts it.
func (s *Scheduler) DispatchedOnUsage(id, reqID, engine string, chains []Chain, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := s.reqs[reqID]
	if id == "" {
		// the scoped path: the branch is OUR record's, and no record means untracked
		// (anonymous, pre-capacity FCFS, or a branch that has already left).
		if rec == nil {
			return
		}
		id = rec.branch
	}
	if len(chains) == 0 && rec != nil {
		chains = rec.chains
	}
	g := s.groupOfLocked(id)
	switch {
	case g == nil && engine == "":
		return // unseated and no observation: nothing says where this request runs
	case g == nil:
		gg, err := s.groupForLocked(engine)
		if err != nil || gg == nil {
			return
		}
		if _, seated := gg.branches[id]; !seated {
			if _, ok := s.seatObservedLocked(gg, id, chains, now); !ok {
				return
			}
		}
		g = gg
	case engine != "" && engine != g.id:
		// the observed endpoint is authoritative
		g = s.reconcileEngineLocked(id, engine, now)
		if g == nil {
			return
		}
	}
	if m, ok := g.branches[id]; ok {
		m.lastActivity = now
		switch {
		case rec != nil:
			// THE WINDOW CLOSES: the request is at the pod.
			s.atPodLocked(rec, m, engine, now)
		case reqID != "":
			// A request the fixture dispatched without a release (a unit path that skips
			// Pick): its record opens AT the pod, so the request is still tracked and
			// still named. The live path always has a release first.
			s.reqs[reqID] = &reqRec{branch: id, chains: chains, engine: engine, released: now, pod: now}
			m.inflight++
			if s.onRelease != nil {
				s.onRelease(id, reqID)
			}
		}
	}
}

// seatObservedLocked accounts for a request that landed on an engine while its branch
// held no seat: the seat was released — enforcement marked it exiting, its drain found
// nothing in flight and let it go — while the request was still in the released window.
//
// The observation wins, as it does in reconcileEngineLocked: the engine is the authority on
// what it is running, so the branch is seated where the request landed, with that request's
// own chain and without a fit test, and marked EXITING at once. The seat exists to account
// for a request in progress (its blocks are in the index, so the budget sees them and the
// drain can release them); it is not an admission, so it takes no cold-entry charge and
// never dispatches a turn, and it leaves when the request drains.
//
// ok=false means there is nothing to account: an engine without measured capacity takes no
// seats, and a request with no chain has no cost to charge. The branch then stays untracked,
// as it was before.
func (s *Scheduler) seatObservedLocked(g *engineGroup, id string, chains []Chain, now time.Time) (*branch, bool) {
	if g.numBlocks <= 0 || len(chains) == 0 {
		return nil, false
	}
	g.index.SeatForceLocked(id, chains)
	g.index.MarkExitingLocked(id)
	m := &branch{id: id, engine: g.id, seatTime: now, lastActivity: now, exitingSince: now}
	g.branches[id] = m
	s.bindings[id] = g.id
	// THE INVARIANT: never seated AND waiting. The branch may still have a head in the SHARED
	// queue — its seat was released while its request was on the way, so no admission pass
	// ever seated it, and an exit re-queued the head it had — and a seat now exists. Leaving
	// the waiter behind would let the admission loop seat an id that is already seated, i.e.
	// charge one branch's blocks twice.
	delete(s.waiters, id)
	g.counters.forcedSeats++
	// the observed seat is a MARK too (it leaves at the drain): one departure, one booking,
	// and it is booked when it actually leaves, not here
	if s.onBind != nil {
		s.onBind(id, g.id) // the placement pin follows the seat
	}
	return m, true
}

// Abandon ends a request's record from OUTSIDE the decision path: the request is gone
// before a pod served it (the pinned engine vanished, a screener refused it, scheduling
// failed), so the branch is not waiting on anything any more. Nothing else changes — a
// seat is not a per-request resource, and an idle seat is evicted on demand.
//
// id is normally empty: the hook is handed a request, and the branch comes from the
// record. An id-less abandon (the legacy form) drops the branch's oldest pending record,
// which is the one it could mean.
func (s *Scheduler) Abandon(id, reqID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id == "" {
		id = s.branchOfRequestLocked(reqID)
		if id == "" {
			return // nothing tracks this request: nothing to close
		}
	}
	if reqID == "" {
		reqID = s.oldestPendingLocked(id)
		if reqID == "" {
			return
		}
	}
	if rec, ok := s.reqs[reqID]; ok && rec.branch == id {
		s.dropRequestLocked(reqID, s.now)
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry charge is released
// HERE, at prefill completion rather than at end of decode, exactly once.
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groupOfLocked(id); g != nil {
		s.releaseColdLocked(g, id)
	}
}

// ChunkAt is Chunk for a caller that has the clock (kept so the live path and the
// virtual-clock tests take the same code).
func (s *Scheduler) ChunkAt(id string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	if g := s.groupOfLocked(id); g != nil {
		s.releaseColdLocked(g, id)
	}
}

// ChunkForRequest is ChunkAt for the scoped hook, which is handed a request and not a
// branch: the branch comes from the request's record. A request nothing tracks releases
// nothing (there is no branch whose charge it could be).
func (s *Scheduler) ChunkForRequest(reqID string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	id := s.branchOfRequestLocked(reqID)
	if id == "" {
		return
	}
	if g := s.groupOfLocked(id); g != nil {
		s.releaseColdLocked(g, id)
	}
}

func (s *Scheduler) releaseColdLocked(g *engineGroup, id string) {
	m, ok := g.branches[id]
	if !ok || m.coldDone || m.coldEntry <= 0 {
		return
	}
	g.coldInFlight = max(g.coldInFlight-m.coldEntry, 0)
	m.coldEntry = 0
	m.coldDone = true
}

// ---- settle (ResponseBody at EndOfStream: the finalizer) ----

// Settle records a terminated dispatched request without an observed endpoint (the
// single-engine form).
func (s *Scheduler) Settle(id, reqID string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.SettleOn(id, reqID, "", usage, natural, now)
}

// SettleOn records a terminated dispatched request on the endpoint it ran on: the
// in-flight decrement for EVERY cause, the residency turn count for every settled turn, and
// the busy period when the branch's last request drains.
//
// An engine mismatch here means the request landed on an engine other than the
// branch's seat AFTER PreRequest reconciled — a CORRECTNESS violation (it corrupts
// the charged engine's record). It is counted, never silently folded: the
// accounting stays with the seat the PreRequest committed, and the counter is the
// run-level alarm.
func (s *Scheduler) SettleOn(id, reqID, engine string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.SettleOnUsage(id, reqID, engine, usage.Prompt, usage.Cached, 0, natural, now)
}

// SettleOnUsage is SettleOn with the completion token count.
//
// id is normally EMPTY: the ResponseBody hook is handed a request, and the branch comes
// from the request's record (the scoped hook cannot read the branch id out of the
// request). A settle for a request nothing tracks is counted as a stray and changes
// nothing — after a late reap or a duplicate settle the record is gone, and inventing a
// branch for it would charge an engine twice.
func (s *Scheduler) SettleOnUsage(id, reqID, engine string, promptTokens, cachedTokens, completionTokens int, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id == "" {
		id = s.branchOfRequestLocked(reqID)
		if id == "" {
			s.counters.straySettles++
			return
		}
	}
	g := s.groupOfLocked(id)
	if g == nil && engine != "" {
		g = s.existingGroupLocked(engine) // never resurrect a torn-down engine
	}
	if g == nil {
		return
	}
	// WHERE IT LANDED is the record's engine (the endpoint the observation named at
	// PreRequest), falling back to the seat binding when nothing recorded one. The binding
	// can have moved since (a rebind), so comparing against it alone would miss a request
	// that ran on an engine other than the one it was seen landing on.
	landed := g.id
	if rec, ok := s.reqs[reqID]; ok && rec.engine != "" {
		landed = rec.engine
	}
	if engine != "" && landed != engine {
		s.counters.affinityViolations++
		s.recordEvent("AFFINITY-VIOLATION", id, fmt.Sprintf("bound=%s settled_on=%s", engineLabel(landed), engineLabel(engine)), now)
		if s.onAffinityViolation != nil {
			s.onAffinityViolation(id, landed, engine)
		}
	}
	m, seated := g.branches[id]
	if seated {
		// THE RECORD ENDS HERE, and the counts follow the map (derive-don't-accumulate).
		// A missing reqID ends the OLDEST record (defensive: an id-less caller cannot name
		// one). A NON-EMPTY id that names no record ends NOTHING and is counted: the
		// oldest-fallback would delete a different live request's record after a late reap
		// or a duplicate settle — the branch would undercount and evict mid-flight.
		removed := false
		if reqID != "" {
			if rec, had := s.reqs[reqID]; had && rec.branch == id {
				s.dropRequestLocked(reqID, now)
				removed = true
			} else {
				s.counters.straySettles++
			}
		} else if oldest := s.oldestAtPodLocked(id); oldest != "" {
			s.dropRequestLocked(oldest, now)
			removed = true
		} else if pending := s.oldestPendingLocked(id); pending != "" {
			// THE TERMINAL CALLBACK ENDS A PENDING RECORD TOO: a request that died between
			// its release and PreRequest has no in-flight record, so this is the only event
			// that can say it is gone — and the handler forces this call whenever a pod was
			// scheduled.
			s.dropRequestLocked(pending, now)
			removed = true
		}
		m.lastActivity = now
		s.releaseColdLocked(g, id) // any termination releases the charge
		// THE BUSY PERIOD: the branch's in-flight count reached zero, and one branch is one
		// chain, so that is the whole of its work draining — the rotation quantum's unit.
		if removed && m.inflight == 0 {
			m.busyPeriods++
			m.lastCompletion = now
		}
	}
	if !seated {
		return
	}
	m.seatTurns++ // every settled turn counts in the residency
	// the settled request was the tracked queued head: clear the marker (the next
	// Pick re-establishes it if more heads remain)
	m.waitingSince = time.Time{}
	// an exiting branch leaves when its in-flight requests drain
	if g.index.ExitingLocked(id) && m.inflight == 0 {
		s.exitLocked(g, m, "exiting-drain", now)
		return
	}
	if !natural {
		return
	}
	uncached := int64(promptTokens - cachedTokens)
	if uncached < 0 {
		uncached = 0
	}
	s.counters.seatTurns = append(s.counters.seatTurns, m.seatTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(promptTokens)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// ---- rotation: WHO leaves, and when ----

// deriveLocked recomputes the derived aggregates from branch records
// (derive-don't-accumulate): a leaked or missed increment anywhere cannot persist
// past the next decision point. Per engine.
func (s *Scheduler) deriveLocked() {
	for _, g := range s.groups {
		var cold int64
		for _, m := range g.branches {
			cold += m.coldEntry
		}
		g.coldInFlight = cold
	}
}

// reapRequestsLocked re-derives every branch's request counts from the map
// (derive-don't-accumulate: a leaked increment cannot survive a decision pass) and drops the
// records that outlived InflightReapSeconds. The handler forces a terminal ResponseBody
// whenever a pod was scheduled, so in the common case this never fires; it is the belt to
// the terminal callback's braces, and on a path that reaches NO hook (an admission rejection
// after the release, a director that returned before its abort hook) it is the only end the
// record gets.
//
// A PENDING record is reaped by the same bound. Nothing else would ever end one whose hooks
// never arrive (a request finalized while it was being dispatched, an endpoint set that
// emptied at the director's entry), and a record that never ends is the one way this map
// could grow without bound.
func (s *Scheduler) reapRequestsLocked(now time.Time) {
	for _, g := range s.groups {
		for _, m := range g.branches {
			m.pending, m.inflight = 0, 0
		}
	}
	limit := time.Duration(s.cfg.reapSeconds() * float64(time.Second))
	reaped := map[string]*branch{} // branch id → its record, for the cold-charge release
	for rid, rec := range s.reqs {
		g := s.groupOfLocked(rec.branch)
		var m *branch
		if g != nil {
			m = g.branches[rec.branch]
		}
		if m == nil {
			// the branch left without its records being swept (defensive): a record that
			// names no seat can never be acted on.
			delete(s.reqs, rid)
			if s.onForget != nil {
				s.onForget(rid)
			}
			continue
		}
		if rec.pod.IsZero() {
			if now.Sub(rec.released) > limit {
				delete(s.reqs, rid)
				if s.onForget != nil {
					s.onForget(rid)
				}
				continue
			}
			if s.windowLive(rec, now) {
				m.pending++
			}
			continue
		}
		if now.Sub(rec.pod) > limit {
			delete(s.reqs, rid)
			if s.onForget != nil {
				s.onForget(rid)
			}
			reaped[rec.branch] = m
			continue
		}
		m.inflight++
	}
	for id, m := range reaped {
		if m.inflight != 0 {
			continue
		}
		m.lastCompletion = now
		// the branch's requests are all dead — release its cold charge, or the valve
		// freezes the queue until the reap bound.
		if g := s.groupOfLocked(id); g != nil {
			s.releaseColdLocked(g, id)
		}
	}
}

// deriveColdChargesLocked releases a cold charge whose owner can no longer be prefilling:
// no queued head, no LIVE released window, and nothing in flight.
//
// A cold charge left behind CLOSES THE ENGINE'S VALVE for everyone — the valve, not the
// budget, is what binds admission — so it is derived from the branch's own records at
// every decision point instead of waiting for a timer. Nothing here evicts anything: a
// charge that is still explained by a head or a request stays.
func (s *Scheduler) deriveColdChargesLocked() {
	for _, g := range s.activeGroupsLocked() {
		for _, m := range g.branches {
			if m.coldEntry == 0 || m.busy() || !m.waitingSince.IsZero() {
				continue
			}
			s.releaseColdLocked(g, m.id)
		}
	}
}

// drainExitingLocked releases every branch that is on its way out and has nothing left in
// flight.
//
// The release must not depend on WHICH path brought the in-flight count to zero: a settle is
// only one of them (a reap for an unobserved termination is another), and a termination
// nobody observed would otherwise leave the branch charged, undispatchable and never
// releasable — the engine looks permanently full while it serves nothing. Deriving the
// condition at every decision point makes it hold by construction, one pass late at worst.
func (s *Scheduler) drainExitingLocked(now time.Time) {
	for _, g := range s.activeGroupsLocked() {
		var ids []string
		for id, m := range g.branches {
			// a LIVE window is in flight: a marked branch may not be drained out from
			// under its own request (measured: the request then ran on an unseated branch
			// and the observed seat put 1500 blocks back against a 1100 budget)
			if m.busy() || !g.index.ExitingLocked(id) {
				continue
			}
			ids = append(ids, id)
		}
		if len(ids) == 0 {
			continue
		}
		// deterministic order: this writes EXIT events, and the event stream's order has
		// to be reproducible in the artifacts
		sort.Strings(ids)
		for _, id := range ids {
			if m, ok := g.branches[id]; ok {
				s.exitLocked(g, m, "exiting-drain", now)
			}
		}
	}
}

// eligibleLocked asks the TENURE module whether a resident may be released: Q — the
// one rotation value — in COMPLETED BUSY PERIODS, or S of silence.
func (s *Scheduler) eligibleLocked(m *branch, now time.Time) (string, bool) {
	return s.mod.tenure.Eligible(m, s.tenureCtxLocked(now))
}

// tenureCtxLocked is the read-only view the TENURE module gets: the pass's clock and S.
// Every decision runs inside Pick, on the queue view that pass synced, so there is no
// snapshot age to account for.
func (s *Scheduler) tenureCtxLocked(now time.Time) TenureCtx {
	return TenureCtx{Now: now, IdleSeconds: s.cfg.IdleSeconds}
}

// pressureCtxLocked is the read-only view the PRESSURE module gets of ONE engine.
func (s *Scheduler) pressureCtxLocked(g *engineGroup, now time.Time) PressureCtx {
	tctx := s.tenureCtxLocked(now)
	return PressureCtx{
		Now:            now,
		Exiting:        g.index.ExitingLocked,
		RotationVictim: func() (string, *branch) { return s.rotationVictimLocked(g, now) },
		Silent:         tctx.Silent,
		// the classification's view: a released window is a turn on its way
		Headless: func(m *branch) bool { return m.waitingSince.IsZero() && m.pending == 0 },
	}
}

// rotationVictimLocked is the demand-driven exit over ONE engine's branches, in the
// order: among the ELIGIBLE, prefer a branch WITH a queued turn (it will
// use the seat again the moment it gets it), then earliest-seated, then id.
//
// The engine-idle rule is the liveness fallback below it: an engine with nothing in
// flight and nothing dispatchable, while a waiter is blocked, must free a seat or
// the queue stalls — and that rule scans ONLY THIS ENGINE, because an engine whose
// branches are all in flight must not borrow another engine's idleness to justify an
// eviction.
func (s *Scheduler) rotationVictimLocked(g *engineGroup, now time.Time) (string, *branch) {
	var best *branch
	var bestReason string
	bestHasWaiting := false
	for _, m := range g.branches {
		if g.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(m, now)
		if !ok {
			continue
		}
		hasWaiting := !m.waitingSince.IsZero()
		better := best == nil ||
			(hasWaiting && !bestHasWaiting) ||
			(hasWaiting == bestHasWaiting && m.seatTime.Before(best.seatTime)) ||
			(hasWaiting == bestHasWaiting && m.seatTime.Equal(best.seatTime) && m.id < best.id)
		if better {
			best, bestReason, bestHasWaiting = m, reason, hasWaiting
		}
	}
	if best != nil {
		return bestReason, best
	}
	// engine-idle rule: no branch in flight AND no branch dispatchable, while a waiter is
	// blocked (a livelock guard: the seats are held by branches that cannot use them)
	idle := true
	for _, m := range g.branches {
		if m.busy() || !m.waitingSince.IsZero() {
			idle = false
			break
		}
	}
	if idle {
		for _, m := range g.branches {
			if g.index.ExitingLocked(m.id) {
				continue
			}
			if best == nil || m.lastActivity.Before(best.lastActivity) ||
				(m.lastActivity.Equal(best.lastActivity) && m.id < best.id) {
				best, bestReason = m, "engine-idle"
			}
		}
	}
	return bestReason, best
}

// olderCompletion breaks a tie between two equally-costly victims: a REAL last
// completion beats "never completed", then oldest completion first, then the id.
//
// The last clause is a correctness requirement, not a taste: every victim rule iterates a
// slice built from the branch map, so without a total order the "first one in map iteration
// order" wins — and Go randomises that, which makes two replays of one scenario disagree
// (TestDeterminism_OneAndTwoEngines is the proof it does not).
func olderCompletion(a, b *branch) bool {
	az, bz := a.lastCompletion.IsZero(), b.lastCompletion.IsZero()
	if az != bz {
		return bz // a real completion is the cheaper victim
	}
	if az {
		return a.id < b.id // TOTAL tie-break among branches that never completed
	}
	return a.lastCompletion.Before(b.lastCompletion) ||
		(a.lastCompletion.Equal(b.lastCompletion) && a.id < b.id)
}

// budgetVictimLocked is MEMORY PRESSURE on the growth path: the PRESSURE module picks the
// victim among ONE engine's branches, in the budget valve's order.
// Only memory pressure gets past the tenure rule's first class: rotation itself never
// evicts a branch that is not eligible.
func (s *Scheduler) budgetVictimLocked(g *engineGroup, now time.Time) (string, *branch) {
	branches := make([]*branch, 0, len(g.branches))
	for _, m := range g.branches {
		branches = append(branches, m)
	}
	return s.mod.pressure.Victim(branches, s.pressureCtxLocked(g, now))
}

// exitLocked unseats one branch. The BINDING IS DROPPED HERE — the seat is the
// binding, so an exiting branch keeps no placement claim: its next seat is chosen
// fresh, by the fit/most-free-budget rule, and may land on a different engine.
func (s *Scheduler) exitLocked(g *engineGroup, m *branch, reason string, now time.Time) {
	s.releaseColdLocked(g, m.id) // unseat releases the charge
	g.index.UnseatLocked(m.id)   // and the exact chains leave the budget here
	delete(g.branches, m.id)
	s.countExitLocked(g, reason)
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d fairness_id=%s wire_id=%s",
		reason, m.seatTurns, fairnessOrNone(m.fairnessID), fairnessOrNone(m.wireID)), now)
	if s.onExit != nil {
		s.onExit(engineLabel(g.id), reason, m.seatTurns)
	}
	delete(s.bindings, m.id)
	if s.onUnbind != nil {
		s.onUnbind(m.id)
	}
	// THE RECORDS GO WITH THE BRANCH: they name a seat that no longer exists, so no later
	// hook could act on one (and its settle, if it ever comes, is a stray rather than a
	// charge against an engine that never ran it).
	s.sweepBranchLocked(m.id, now)
	// a branch with a queued head re-enters at its ORIGINAL waiting-since (the queue position
	// survives the exit) — back in the SHARED queue, with no engine attached, carrying the
	// cost its HEAD will pay and the class its head has. Both come from the head itself (the
	// last queue view that showed it), never from the turn that just ran and never from an
	// assumption: rotation re-picks the queue inside this very pass, so the cost has to be
	// right before the next sync reads the head again.
	if !m.waitingSince.IsZero() {
		s.waiters[m.id] = &waiter{id: m.id, since: m.waitingSince,
			returner: m.headReturner, chains: m.headChains,
			fairnessID: m.fairnessID, wireID: m.wireID}
	}
}

// fairnessOrNone renders an absent fairness id as "-" so a parser never sees an
// empty field.
func fairnessOrNone(id string) string {
	if id == "" {
		return "-"
	}
	return id
}

// countExitLocked books ONE departure on the engine AND in the global tally (Snapshot.Exits
// stays the whole-policy view). Marks are states, never departures: a mid-flight branch is
// marked exiting (exitingBranches/exitingBlocks/exitingOldest) and booked here exactly once, at
// the exit that actually releases its seat.
func (s *Scheduler) countExitLocked(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

// seatLocked seats one waiting head on the engine the admission pass chose. It is the
// ONE place a binding is created: seat = pin, and the seat is the whole of it.
//
// coldTokens is the head's exact cold cost on THIS engine (the same number the placement
// module's fit test used), so the charge and the fit can never disagree.
func (s *Scheduler) seatLocked(g *engineGroup, w *waiter, coldTokens int64, now time.Time) bool {
	if ok, _ := g.index.TrySeatLocked(w.id, w.chains, g.budgetBlocks); !ok {
		return false // unreachable: the chooser tested the same number in the same pass
	}
	s.bindings[w.id] = g.id
	if s.onBind != nil {
		s.onBind(w.id, g.id) // the placement pin's mirror (filter.go)
	}
	delete(s.waiters, w.id)
	m := &branch{
		id:           w.id,
		engine:       g.id,
		seatTime:     now,
		lastActivity: now,
		// the head that triggered this seating is still queued (the serve happens after the
		// seat): inherit its presence so the engine-idle rule cannot evict a branch that is
		// about to dispatch — and its own chains and class, so a later exit re-queues with
		// this head's cost rather than the chain of whatever ran.
		waitingSince: w.since,
		fairnessID:   w.fairnessID,
		wireID:       w.wireID,
		headChains:   w.chains,
		headReturner: w.returner,
	}
	// EVERY entry carries a cold charge (in tokens), CAPPED at the engine's own stagger
	// cap: the valve's job is to serialise cold prefills inside the budget's headroom, not
	// to account the entry's whole footprint (the seat's blocks do that). Capping is also
	// what makes the cap an invariant — an entry bigger than the headroom charges the
	// headroom, so an empty valve always admits exactly one entry, and the outstanding
	// charge can never exceed the cap.
	m.coldEntry = min(coldTokens, g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
	g.coldInFlight += m.coldEntry
	// record EVERY seat wait, split by kind (a branch's first turn vs a returning branch):
	// both answer "how long did admission make this branch wait".
	eventKind := "re-entry"
	waitKind := "reentry"
	if !w.returner {
		eventKind = "newcomer"
		waitKind = "new"
		s.counters.entryWaits = append(s.counters.entryWaits, now.Sub(w.since).Seconds())
		g.counters.entryWaits = append(g.counters.entryWaits, now.Sub(w.since).Seconds())
	} else {
		s.counters.reentryWaits = append(s.counters.reentryWaits, now.Sub(w.since).Seconds())
		g.counters.reentryWaits = append(g.counters.reentryWaits, now.Sub(w.since).Seconds())
	}
	if s.onEntry != nil {
		s.onEntry(engineLabel(g.id), waitKind, now.Sub(w.since).Seconds())
	}
	// the fields a run's forensics need, without the event ring: branch id, wire id,
	// engine (the Event's own field), the seat kind, the wait, and the arrival the wait is
	// measured from
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f fairness_id=%s wire_id=%s since=%s",
		eventKind, now.Sub(w.since).Seconds(), fairnessOrNone(w.fairnessID), fairnessOrNone(w.wireID),
		w.since.UTC().Format(time.RFC3339Nano)), now)
	g.branches[w.id] = m
	return true
}

// enforceBudgetLocked is MEMORY PRESSURE on the growth path: while the STAY side — the
// blocks of the branches that are not already leaving — is over the engine's budget, take
// victims in the valve's order until it fits or nothing is evictable.
//
// TWO NUMBERS, ONE ACCOUNTING, and the difference between them is the whole of this
// function:
//
//	ADMISSION (freeBudgetLocked, TrySeatLocked) asks about the USED side: every seated
//	  branch's blocks, exiting ones included, because the engine pins them until the
//	  request drains. A head that would only fit if the retirees' blocks were free waits.
//	ENFORCEMENT (here) asks about the STAY side: the branches that are NOT exiting. Those
//	  are the blocks that will still be there when the retirees are gone, so they are the
//	  overage anything can still act on.
//
// That is also why a MARK is progress: marking a prefilling victim moves its blocks off the
// stay side (it leaves at its drain), so every pass strictly reduces the overage. Without
// that the loop would be free of consequences — it marks a victim, the footprint does not
// move, it marks the next in-flight branch, and the next, on every pass, and a pass is
// every Pick. That is the cascade, and `TestEveryEnforcementMarkReducesTheOverage` is its
// regression proof: a pass is DONE once what is over is already on its way out.
func (s *Scheduler) enforceBudgetLocked(g *engineGroup, now time.Time) {
	if g.budgetBlocks <= 0 {
		return // no measured capacity yet for THIS engine: nothing to enforce
	}
	for s.overBudgetLocked(g) > 0 {
		reason, victim := s.budgetVictimLocked(g, now)
		if victim == nil {
			return
		}
		if victim.busy() {
			if g.index.ExitingLocked(victim.id) {
				// Already on its way out. Its blocks stay charged until it drains, so
				// marking it again frees nothing: this pass is DONE (continuing would
				// re-pick the same branch and spin under the scheduler's mutex).
				return
			}
			g.index.MarkExitingLocked(victim.id)
			victim.exitingSince = now
			// THE MARK IS A STATE, NOT AN EVENT: one departure is booked ONCE, when the
			// branch actually leaves (reason "exiting-drain"). Booking the mark as an exit
			// too double-counted every mid-flight departure in the same series — and the
			// mark is already visible as state (ExitingMembers / ExitingBlocks /
			// ExitingOldestAgeS), which is where an operator looks for "on its way out".
			continue
		}
		s.exitLocked(g, victim, reason, now)
	}
}

// overBudgetLocked is enforcement's view of the overage, in blocks: the STAY side (the
// seated branches that are not exiting) over the budget.
// An exiting branch's blocks are excused by construction — they are not on the stay side,
// and if a staying branch shares them the shared blocks still count there, which is
// exactly the under-eviction this replaces (used − |exit| excused them wholesale).
func (s *Scheduler) overBudgetLocked(g *engineGroup) int64 {
	over := int64(g.index.StayBlocksLocked()) - int64(g.budgetBlocks)
	if over < 0 {
		return 0
	}
	return over
}

// ---- lifecycle: the ~1s ticker ----

func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	// DECISIONS LIVE IN PICK. A Tick only does what has no event to hang on: derive the
	// aggregates from the records, end the request records that outlived the reap bound,
	// and release branches whose drain has nothing left to wait for. Admission,
	// enforcement and rotation run in the Pick pass, on the queue view that pass just
	// synced.
	s.deriveLocked()
	s.reapRequestsLocked(now)
	s.drainExitingLocked(now)
}

func (s *Scheduler) seatedAnywhereLocked(id string) bool {
	for _, g := range s.groups {
		if _, ok := g.branches[id]; ok {
			return true
		}
	}
	return false
}

// ---- engine lifecycle (the DISCOVERED engine set) ----

// RemoveEngine tears one engine's group down when the EPP reports the
// endpoint gone (an endpoint delete — node rotation exercises this every wave).
// Its branches are exited (reason "engine-gone") — their seats die with it and
// their heads go back to the SHARED queue — and the group's budget/index/seats
// disappear with it. Leaving charges on a dead engine is how a scale-down
// silently poisons every later budget decision.
func (s *Scheduler) RemoveEngine(engine string, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, ok := s.groups[engine]
	if !ok {
		return 0
	}
	// deterministic order: the teardown writes one EXIT event per branch and
	// a torn-down group's event order must be reproducible in the artifacts
	ids := make([]string, 0, len(g.branches))
	for id := range g.branches {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	n := 0
	for _, id := range ids {
		if m, ok := g.branches[id]; ok {
			s.exitLocked(g, m, "engine-gone", now)
			n++
		}
	}
	// The waiter QUEUE is shared and survives a teardown: those heads can seat on
	// a survivor, which is the whole point of pooling the waits.
	for branch, id := range s.bindings {
		if id == engine {
			delete(s.bindings, branch)
		}
	}
	delete(s.groups, engine)
	s.rebuildGroupOrderLocked()
	s.counters.engineGone++
	if s.onEngineGone != nil {
		s.onEngineGone(engine, n)
	}
	return n
}

// ---- introspection ----

// ---- the seat/exit event log (spans cannot see seats, so every run needs it) ----

// Event is one seat/exit/rebind decision, per id, with reason and timestamp. Kept as a
// bounded ring (dumped in DumpState) and emitted to the onEvent hook (the plugin logs one
// line per event).
//
// Engine is the engine label the event happened on: a FIELD of its own, never spliced into
// the detail, so a per-engine reducer can read it without parsing.
type Event struct {
	T      time.Time `json:"t"`
	Kind   string    `json:"kind"`   // SEAT | EXIT | REBIND | AFFINITY-VIOLATION
	ID     string    `json:"id"`     // the branch id the decision was taken for
	Engine string    `json:"engine"` // the engine label the decision happened on
	Detail string    `json:"detail"` // reason=… turns=… / kind=… wait=… / from=… to=… / fairness_id=… wire_id=…
	// Branch is the fairness unit the event's id resolves to at emit time
	// (the affinity-violation join needs branch→engine-at-seat reconstructable
	// from the artifacts alone).
	Branch string `json:"branch,omitempty"`
}

const eventRingSize = 2048

func (s *Scheduler) recordEvent(kind, id, detail string, now time.Time) {
	ev := Event{T: now, Kind: kind, ID: id, Detail: detail}
	if g := s.groupOfLocked(id); g != nil {
		ev.Engine = engineLabel(g.id)
		ev.Branch = id
	}
	s.recordEventOn(ev)
}

// recordEventOn appends an already-attributed event (the caller knows the
// engine: a torn-down group's events are emitted after its branches are
// gone).
func (s *Scheduler) recordEventOn(ev Event) {
	s.events = append(s.events, ev)
	if len(s.events) > eventRingSize {
		s.counters.eventsDropped += int64(len(s.events) - eventRingSize)
		s.events = s.events[len(s.events)-eventRingSize:]
	}
	if s.onEvent != nil {
		s.onEvent(ev)
	}
}

// Events returns the recent event ring (newest last).
func (s *Scheduler) Events() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Event(nil), s.events...)
}

// Snapshot is the whole-policy view of the state machine (DumpState), plus the
// per-engine section the multi-pod arm reports and the lights-out dumps are scored
// on.
type Snapshot struct {
	Seats        int
	UniqueBlocks int
	NEBlocks     int
	BudgetBlocks int
	Waiters      int
	// WaitersShared is the SHARED waiter queue's size. It is the same number as
	// Waiters (a waiter belongs to no engine), and it is reported separately so a
	// reader cannot mistake the per-engine Waiters fields for a partition.
	WaitersShared  int
	Exits          map[string]int64
	MeanResTurns   float64
	EntryWaitP50   float64
	ReentryWaitP50 float64
	// RefillShare is the share of prompt tokens that had to be recomputed (uncached), and
	// −1 before any turn has settled (see `absent`).
	RefillShare float64 `json:"refillShare"`
	Served      int64
	FCFSServed  int64
	// Config echoes the effective knob values (auditable runs).
	Config Config
	// RecentEvents is the tail of the seat/exit ring.
	RecentEvents []Event
	// Engines is the PER-ENGINE section. The top-level fields above remain the
	// whole-policy sums, so a single-engine dump is unchanged and a multi-engine
	// dump is scoreable.
	Engines map[string]EngineSnapshot `json:"engines,omitempty"`
	// AffinityViolations / Rebinds are the CORRECTNESS counters (a request landing
	// off its branch's seat engine).
	AffinityViolations int64 `json:"affinityViolations"`
	Rebinds            int64 `json:"rebinds"`
	EngineGone         int64 `json:"enginesGone"`
	// Bindings is the number of LIVE seat bindings per engine (a branch is bound
	// exactly while it is seated, so this equals that engine's seats; a binding to
	// an engine that holds no such branch must be 0).
	Bindings map[string]int `json:"bindings,omitempty"`
	// HeterogeneousEngines lists engines whose num_gpu_blocks deviates >1% from the
	// first measured engine (the wiring alarm).
	HeterogeneousEngines []string `json:"heterogeneousEngines,omitempty"`
	// EventRingDropped is how many SEAT/EXIT events the bounded ring has wrapped past:
	// the dump's RecentEvents covers the LAST eventRingSize events, and this says how
	// much of the window lies before them.
	EventRingDropped int64 `json:"eventRingDropped"`
}

// EngineSnapshot is ONE engine's group state: seats, the budget, the exit mix, the
// exiting ledger and the per-engine wait ladder.
type EngineSnapshot struct {
	ID           string `json:"id"`
	Seats        int    `json:"seats"`
	Waiters      int    `json:"waiters"`
	UniqueBlocks int    `json:"uniqueBlocks"`
	NEBlocks     int    `json:"neBlocks"`
	BudgetBlocks int    `json:"budgetBlocks"`
	NumBlocks    int    `json:"numBlocks"`
	CapacityOK   bool   `json:"capacityOk"`
	ColdInFlight int64  `json:"coldInFlightTokens"`
	// ColdCapTokens is the DERIVED cold-entry stagger cap (the budget's own
	// headroom, in tokens). A scored run divides coldInFlightTokens by THIS, not by
	// a constant from the arm's config: the cap moves with the engine.
	ColdCapTokens int64 `json:"coldCapTokens"`
	// The EXITING ledger (the enforcement view's own inputs, per engine): branches on
	// their way out, the blocks of the used side they still hold, and how long the
	// oldest one has been leaving. Read together they say whether an engine is busy or
	// stuck: exiting_blocks inside ne_blocks is a drain in progress, and an oldest age
	// that only grows is a drain that is not happening.
	ExitingMembers    int     `json:"exitingMembers"`
	ExitingBlocks     int     `json:"exitingBlocks"`
	ExitingOldestAgeS float64 `json:"exitingOldestAgeSeconds"`
	BoundMembers      int     `json:"boundMembers"`
	Served            int64   `json:"served"`
	ForcedSeats       int64   `json:"forcedSeats"`
	// EntryWaitP50 / ReentryWaitP50 are THIS engine's median seat waits, or -1 when nothing
	// has waited yet (see `absent`): the per-engine wait ladder, which the Prometheus
	// histograms cannot give a lights-out dump.
	EntryWaitP50   float64          `json:"entryWaitP50"`
	ReentryWaitP50 float64          `json:"reentryWaitP50"`
	Exits          map[string]int64 `json:"exits,omitempty"`
	BlockSizeErr   string           `json:"blockSizeMismatch,omitempty"`
}

// SnapshotLite is Snapshot without the event ring and config echo — the metrics
// Refresh path, which runs every second and must not copy the 2048-event ring.
func (s *Scheduler) SnapshotLite() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshotLocked(false)
}

func (s *Scheduler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshotLocked(true)
}

func (s *Scheduler) snapshotLocked(full bool) Snapshot {
	snap := Snapshot{Exits: map[string]int64{},
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed,
		AffinityViolations: s.counters.affinityViolations, Rebinds: s.counters.rebinds,
		EngineGone:       s.counters.engineGone,
		EventRingDropped: s.counters.eventsDropped,
		Engines:          map[string]EngineSnapshot{}, Bindings: map[string]int{}}
	if full {
		snap.Config = s.cfg
		snap.RecentEvents = append([]Event(nil), s.events...)
	}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	for _, engine := range s.bindings {
		snap.Bindings[engineLabel(engine)]++
	}
	var seats, uniq, ne int
	for _, id := range s.groupOrder {
		g := s.groups[id]
		label := engineLabel(id)
		exitingMembers, exitingOldest := 0, time.Duration(0)
		for id, m := range g.branches {
			if !g.index.ExitingLocked(id) {
				continue
			}
			exitingMembers++
			if age := s.now.Sub(m.exitingSince); age > exitingOldest {
				exitingOldest = age
			}
		}
		es := EngineSnapshot{ID: label, Seats: len(g.branches),
			ExitingMembers: exitingMembers, ExitingBlocks: g.index.ExitingBlocksLocked(),
			ExitingOldestAgeS: exitingOldest.Seconds(),
			UniqueBlocks:      g.index.UniqueBlocks(), NEBlocks: g.index.UniqueBlocksLocked(),
			BudgetBlocks: g.budgetBlocks, NumBlocks: g.numBlocks,
			CapacityOK:   g.numBlocks > 0,
			ColdInFlight: g.coldInFlight, Served: g.counters.served,
			ColdCapTokens:  g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize),
			ForcedSeats:    g.counters.forcedSeats,
			BoundMembers:   snap.Bindings[label],
			EntryWaitP50:   p50(g.counters.entryWaits),
			ReentryWaitP50: p50(g.counters.reentryWaits),
		}
		if g.blockSizeMismatch != nil {
			es.BlockSizeErr = g.blockSizeMismatch.Error()
		}
		if len(g.counters.exits) > 0 {
			es.Exits = map[string]int64{}
			for k, v := range g.counters.exits {
				es.Exits[k] = v
			}
		}
		snap.Engines[label] = es
		seats += es.Seats
		uniq += es.UniqueBlocks
		ne += es.NEBlocks
		snap.BudgetBlocks += g.budgetBlocks
	}
	snap.Seats, snap.UniqueBlocks, snap.NEBlocks = seats, uniq, ne
	// A waiter belongs to NO engine (the queue is shared), so the per-engine
	// Waiters above are 0 by construction; the whole-policy number is the queue.
	snap.WaitersShared = len(s.waiters)
	snap.Waiters = snap.WaitersShared
	if n := len(s.counters.seatTurns); n > 0 {
		sum := 0
		for _, t := range s.counters.seatTurns {
			sum += t
		}
		snap.MeanResTurns = float64(sum) / float64(n)
	}
	snap.EntryWaitP50 = p50(s.counters.entryWaits)
	snap.ReentryWaitP50 = p50(s.counters.reentryWaits)
	snap.RefillShare = s.counters.refillShare
	if s.counters.totalTok == 0 {
		snap.RefillShare = absent // no turn has settled: unmeasured, not zero
	}
	for id := range s.heterogeneousLocked(0.01) {
		snap.HeterogeneousEngines = append(snap.HeterogeneousEngines, engineLabel(id))
	}
	sort.Strings(snap.HeterogeneousEngines)
	return snap
}

// absent is THE absent-signal convention: a quantity that was never observed reads -1,
// never 0. A zero is a measurement ("nothing waited", "no time passed"); an unmeasured
// quantity read as zero is a good-looking answer to a question nobody asked, and a gate
// that reads it cannot tell the two apart.
const absent = -1

// p50 is the median of a wait sample, or `absent` when nothing has been sampled.
func p50(w []float64) float64 {
	if len(w) == 0 {
		return absent
	}
	cp := append([]float64(nil), w...)
	sort.Float64s(cp)
	return cp[len(cp)/2]
}

// Sessions returns the tracked-branch-record count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, g := range s.groups {
		n += g.index.TrackedCount()
	}
	return n
}

// seatedIDs is the union of every engine's seated branch ids (the replay's state
// observation).
func (s *Scheduler) seatedIDs() map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]bool{}
	for _, g := range s.groups {
		for id := range g.branches {
			out[id] = true
		}
	}
	return out
}

// Seated reports whether the branch holds a seat on ANY engine.
func (s *Scheduler) Seated(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seatedAnywhereLocked(id)
}
