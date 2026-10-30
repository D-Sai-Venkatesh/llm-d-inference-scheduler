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
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ---- §9 parameters (the only knobs; spec defaults) ----

// Config is the group scheduler's parameter surface: ALGORITHM-group.md
// §9 plus BlockSizeTokens, a WIRING FACT (not a tuning knob): it must
// equal the token processor's block size so the index's unit matches the
// engine's. Measured capacity (num_gpu_blocks) arrives from the endpoint
// metrics; a mismatch with the measured CacheBlockSize fails loudly.
type Config struct {
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`
	Rho                 int     `json:"rho"`
	NMax                int     `json:"nMax"`
	IdleSeconds         float64 `json:"idleSeconds"`
	TTLSeconds          float64 `json:"ttlSeconds"`
	ColdEntryCapTokens  int64   `json:"coldEntryCapTokens"`
	InFlightCap         int     `json:"inFlightCap"`
	// BlockSizeTokens is the engine's KV block size in tokens (the token
	// processor's value; 64 on the campaign stacks, M4). Required > 0.
	BlockSizeTokens int `json:"blockSizeTokens"`

	// ---- v2 knobs (BUILD-PLAN §1; the defaults reproduce the legacy
	// decision path exactly — pinned by the golden replay test) ----

	// RotationMode selects the exit-eligibility model:
	//   "ledger"  — the legacy payback ledger (rho, nMax). DEFAULT.
	//   "quantum" — the busy-period quantum: a member is rotation-
	//               eligible when it has completed QuantumBusyPeriods
	//               busy periods (inflight 0→>0→0) or is silent > S.
	RotationMode string `json:"rotationMode"`
	// QuantumBusyPeriods is Q (rotationMode "quantum" only):
	// 0 → 4; -1 → ∞ (the pause arm P∞: no member is ever
	// quantum-complete, so the waiting excess never displaces residents).
	QuantumBusyPeriods int `json:"quantumBusyPeriods"`
	// ValveMode: "on" (legacy cold-entry stagger) | "off" (the stagger
	// moves to the engine: --scheduling-policy priority +
	// long_prefill_token_threshold + X-Vllm-Priority; Phase-2 deployment).
	ValveMode string `json:"valveMode"`
	// VictimOrder: "legacy" (earliest-seated, prefers members WITH a
	// queued turn — C2) | "spec" (prefers members with NO queued turn,
	// then oldest last completion — ALGORITHM §7).
	VictimOrder string `json:"victimOrder"`
	// WindowMode: "last4" (legacy fixed 4-request dispatch window) |
	// "derived" (one entry per live prefix-chain, expiring after
	// IdleSeconds — MANDATORY under root-session identity, where fan-out
	// within one id is real and a fixed window undercounts).
	WindowMode string `json:"windowMode"`
	// PriorityHeader (V3, valve-off deployments): "off" (default) |
	// "engine" — PreRequest sets X-Vllm-Priority on every tracked
	// request: 0 (high) for a seated member's turn, 10 (low) for an
	// entry (newcomer) request. Requires the Phase-2 engine config
	// (--scheduling-policy priority); without it, cold prefills block
	// resident turns in vLLM's waiting loop (the D14 damage).
	PriorityHeader string `json:"priorityHeader"`
	// GateMode: "detector" (legacy: the stack's saturation detector
	// gates dispatch at ~0.80 running KV) | "size-aware" (the plugin
	// gates each dispatch: NE + pending + this request's new blocks ≤
	// GroupBudgetFraction × HBM, from the EPP's own accounting; deploy
	// with the detector raised to ~0.95 so it no longer binds first).
	GateMode string `json:"gateMode"`
	// InflightReapSeconds bounds the age of a CONFIRMED in-flight
	// request id before reconciliation drops it (C3: requests whose
	// termination is never observed — the harness request timeout is
	// 3600s, so 3700 reaps only true leaks). Provisional ids (chosen at
	// Pick but never reaching PreRequest) are reaped after 30s.
	InflightReapSeconds float64 `json:"inflightReapSeconds"`

	// ---- WAVE-1 knobs (branch identity: base mode + the session layer) ----

	// MemberKey selects which request attribute the policy treats as the
	// MEMBER — the fairness unit (seats, budget accounting, rotation,
	// newcomer status, in-flight and waiting accounting):
	//
	//	"session" — the agent-identity SESSION attribute. This is the
	//	            behaviour of every arm before WAVE-1.
	//	"branch"  — the derived BRANCH id (branch-identity's
	//	            BranchIdentityKey). BASE MODE: one implicit id is
	//	            queue key AND member AND fairness unit, which is what
	//	            moves a headerless client off R9 FCFS pass-through
	//	            (BRANCH-IDENTITY-DESIGN §V2 BASE-MODE FRAMING).
	//	"auto"    — branch when present, else the session, else
	//	            anonymous (→ R9). DEFAULT: base mode and session mode
	//	            are the same build; attribute presence picks the mode.
	//
	// The branch id is NEVER written into the agent-identity slot: the
	// session attribute keeps meaning "the client's session" in every
	// mode (the session layer reads it as such; nothing here writes it).
	MemberKey string `json:"memberKey"`
	// SessionOrder selects the SESSION ordering layer at D1 — the entry
	// head ONLY (victim choice D2 and dispatch order D3 are untouched):
	//
	//	"off"      — no session layer: newcomers first, oldest within the
	//	             class, then id (today's behaviour). DEFAULT.
	//	"newcomer" — NEWCOMER STATUS becomes session-level: only a NEW
	//	             session's head is newcomers-first; later branches of
	//	             an already-served session queue FIFO (the anti
	//	             newcomer-jumping rule).
	//	"lsf"      — least-seats-first: entry key = (session seats held ↑,
	//	             session-is-new first, oldest arrival, id).
	//
	// The session unit is the request's session ATTRIBUTE when it carries
	// one and the MEMBER key otherwise, so with session := branch (base
	// mode) every mode reduces EXACTLY to today's order — pinned by
	// TestSessionOrderGoldenEquality.
	SessionOrder string `json:"sessionOrder"`
	// CapacityLivenessSeconds bounds how long the measured capacity
	// (num_gpu_blocks) may stay unknown after a decode endpoint is first
	// observed. Past it the plugin logs loudly and raises
	// llm_d_epp_group_capacity_missing: the pre-capacity FCFS pass-through
	// (A5) is otherwise invisible in the outcome metrics.
	CapacityLivenessSeconds float64 `json:"capacityLivenessSeconds"`

	// ---- MULTI-POD knob (PREREG-multipod-offload task 1) ----

	// EngineGroups selects the fairness state's scope:
	//
	//	"single"     — ONE group for the whole policy: every engine an
	//	               endpoint-discovery populates is summed into one
	//	               budget, index and seat set (today's behaviour, and
	//	               the DEFAULT — pinned byte-identically by the golden
	//	               replay). Correct only for one engine.
	//	"per-engine" — one engineGroup per DISCOVERED endpoint, each with
	//	               its own BlockIndex, budget (95% of THAT engine's
	//	               HBM), capacity liveness, seats, waiters, pending
	//	               total, cold-entry cap and rotation loop. Identity
	//	               (everSeated / session bookkeeping / lastSeen) and
	//	               session seat counts stay global — a session spans
	//	               engines. Members are HOMED to one engine at first
	//	               seat and never migrate while active.
	//
	// The engine set is DISCOVERED, never configured: groups appear on the
	// endpoints the EPP already sees and are torn down on EventDelete.
	EngineGroups string `json:"engineGroups"`
}

func defaultConfig() Config {
	return Config{
		GroupBudgetFraction:     0.95,
		Rho:                     4,
		NMax:                    0, // -> 2ρ
		IdleSeconds:             20,
		TTLSeconds:              30,
		ColdEntryCapTokens:      131072,
		InFlightCap:             64,
		BlockSizeTokens:         64,
		RotationMode:            "ledger",
		QuantumBusyPeriods:      0, // -> 4
		ValveMode:               "on",
		VictimOrder:             "legacy",
		WindowMode:              "last4",
		GateMode:                "detector",
		PriorityHeader:          "off",
		InflightReapSeconds:     3700,
		MemberKey:               "auto",
		SessionOrder:            "off",
		CapacityLivenessSeconds: 120,
		EngineGroups:            "single",
	}
}

func (c Config) nMax() int {
	if c.NMax > 0 {
		return c.NMax
	}
	return 2 * c.Rho
}

// perEngine reports whether the policy keeps one group per engine.
func (c Config) perEngine() bool { return c.EngineGroups == "per-engine" }

// effective fills the knob defaults (the zero value of each string
// knob means "default") so the startup echo and Snapshot.Config show
// EFFECTIVE values (S2 — validate()'s value receiver made its fills
// dead code).
func (c Config) effective() Config {
	def := defaultConfig()
	if c.RotationMode == "" {
		c.RotationMode = def.RotationMode
	}
	if c.ValveMode == "" {
		c.ValveMode = def.ValveMode
	}
	if c.VictimOrder == "" {
		c.VictimOrder = def.VictimOrder
	}
	if c.WindowMode == "" {
		c.WindowMode = def.WindowMode
	}
	if c.GateMode == "" {
		c.GateMode = def.GateMode
	}
	if c.InflightReapSeconds == 0 {
		c.InflightReapSeconds = def.InflightReapSeconds
	}
	if c.MemberKey == "" {
		c.MemberKey = def.MemberKey
	}
	if c.SessionOrder == "" {
		c.SessionOrder = def.SessionOrder
	}
	if c.CapacityLivenessSeconds == 0 {
		c.CapacityLivenessSeconds = def.CapacityLivenessSeconds
	}
	if c.EngineGroups == "" {
		c.EngineGroups = def.EngineGroups
	}
	return c
}

// quantum is the effective Q (rotationMode "quantum").
func (c Config) quantum() int {
	if c.QuantumBusyPeriods != 0 {
		return c.QuantumBusyPeriods
	}
	return 4
}

// quantumFinite reports whether Q is finite (false = the pause arm P∞).
func (c Config) quantumFinite() bool { return c.quantum() >= 0 }

func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.Rho < 1 {
		return fmt.Errorf("rho must be >= 1, got %d", c.Rho)
	}
	if c.IdleSeconds <= 0 || c.TTLSeconds <= 0 {
		return fmt.Errorf("idleSeconds and ttlSeconds must be > 0")
	}
	if c.ColdEntryCapTokens <= 0 || c.InFlightCap <= 0 {
		return fmt.Errorf("coldEntryCapTokens and inFlightCap must be > 0")
	}
	if c.BlockSizeTokens <= 0 {
		return fmt.Errorf("blockSizeTokens is required (the token processor's block size; got %d)", c.BlockSizeTokens)
	}
	if c.RotationMode == "" {
		c.RotationMode = "ledger"
	}
	switch c.RotationMode {
	case "ledger", "quantum":
	default:
		return fmt.Errorf("rotationMode: invalid value %q (ledger|quantum)", c.RotationMode)
	}
	if c.ValveMode == "" {
		c.ValveMode = "on"
	}
	switch c.ValveMode {
	case "on", "off":
	default:
		return fmt.Errorf("valveMode: invalid value %q (on|off)", c.ValveMode)
	}
	if c.VictimOrder == "" {
		c.VictimOrder = "legacy"
	}
	switch c.VictimOrder {
	case "legacy", "spec":
	default:
		return fmt.Errorf("victimOrder: invalid value %q (legacy|spec)", c.VictimOrder)
	}
	if c.WindowMode == "" {
		c.WindowMode = "last4"
	}
	switch c.WindowMode {
	case "last4", "derived":
	default:
		return fmt.Errorf("windowMode: invalid value %q (last4|derived)", c.WindowMode)
	}
	if c.GateMode == "" {
		c.GateMode = "detector"
	}
	switch c.GateMode {
	case "detector", "size-aware":
	default:
		return fmt.Errorf("gateMode: invalid value %q (detector|size-aware)", c.GateMode)
	}
	switch c.PriorityHeader {
	case "", "off", "engine":
	default:
		return fmt.Errorf("priorityHeader: invalid value %q (off|engine)", c.PriorityHeader)
	}
	if c.InflightReapSeconds != 0 && c.InflightReapSeconds < 60 {
		return fmt.Errorf("inflightReapSeconds must be >= 60 (the harness request timeout is 3600s; got %v)", c.InflightReapSeconds)
	}
	switch c.MemberKey {
	case "", "auto", "branch", "session":
	default:
		return fmt.Errorf("memberKey: invalid value %q (auto|branch|session)", c.MemberKey)
	}
	switch c.SessionOrder {
	case "", "off", "newcomer", "lsf":
	default:
		return fmt.Errorf("sessionOrder: invalid value %q (off|newcomer|lsf)", c.SessionOrder)
	}
	if c.CapacityLivenessSeconds < 0 {
		return fmt.Errorf("capacityLivenessSeconds must be >= 0 (0 = default; got %v)", c.CapacityLivenessSeconds)
	}
	switch c.EngineGroups {
	case "", "single", "per-engine":
	default:
		return fmt.Errorf("engineGroups: invalid value %q (single|per-engine)", c.EngineGroups)
	}
	return nil
}

// reapSeconds is the effective confirmed-id reap bound (0 → default).
func (c Config) reapSeconds() float64 {
	if c.InflightReapSeconds > 0 {
		return c.InflightReapSeconds
	}
	return 3700
}

// ---- the queue-head snapshot (S0-1: waiting state comes from the
// flow-control queues at Pick time, not from PreRequest) ----

// Head is one waiting queue head as seen at Pick: the MEMBER id (the
// fairness unit — see Config.MemberKey), the request's SESSION attribute
// when it carried one, the request id, the enqueue time, and the wire
// size (the conservative-cost input).
type Head struct {
	// Member is the group policy's unit: the branch id, the session, or
	// "" for anonymous traffic (never tracked, never seated, R9).
	Member string
	// Session is the request's session ATTRIBUTE (read-only here). Empty
	// means the request carried none — base mode — and the session layer
	// then treats the member key as the session (session := branch), which
	// is what makes every sessionOrder mode reduce to today's order.
	Session   string
	ReqID     string
	Arrival   time.Time
	SizeBytes int64
	// Source names the attribute the member id came from ("branch" |
	// "session" | ""). Counted per dispatch (the run is auditable from the
	// artifacts without re-deriving it from the config).
	Source string
}

// ---- per-session scheduling state (§3) ----

type member struct {
	id string
	// engine is the engineGroup id this member is HOMED to (the same value
	// as the identity-level binding; kept on the record so a dump shows the
	// binding without a second lookup).
	engine string
	// session is the ordering layer's unit for this member (the session
	// attribute at seat time, or the member id when the request carried
	// none). Fixed at seat; read only by entryHeadLocked.
	session string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// residency payback ledger (§7, tokens from usage reports)
	entryCost int64
	earned    int64
	resTurns  int
	// inflight is DERIVED each decision point from inflightIDs
	// (derive-don't-accumulate, BUILD-PLAN §1): a leaked increment
	// cannot persist past a reconciliation.
	inflight int
	// inflightIDs tracks the member's in-flight request ids with their
	// confirmation state (C3): provisional (chosen at Pick, not yet at
	// PreRequest) ids are reaped after 30s; confirmed ids after
	// InflightReapSeconds. m.inflight = len(inflightIDs).
	inflightIDs map[string]inflightRec
	// busyPeriods counts completed MAIN-CHAIN busy periods (V1: the
	// main chain draining to zero — under root identity a busy root
	// with overlapping sub-agent threads never fully drains, which
	// would make it de-facto Q=∞).
	busyPeriods int
	// mainInflight is the in-flight count on the MAIN chain (the
	// busy-period unit's input).
	mainInflight int
	// lastCompletion is the last time inflight returned to zero (the
	// spec victim order's tie-break: oldest last completion first).
	lastCompletion time.Time
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this member's outstanding cold-entry charge in TOKENS
	// against the stagger cap; released EXACTLY once (S0-2) at the first
	// of: prefill complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// entryRecorded marks that the entry turn's cost has been measured
	// (R4: a fully warm entry has uncached 0, which the old
	// entryCost==0&&earned==0 test misread as "still the entry turn").
	entryRecorded bool
	// entryDispatched marks that this residency's FIRST dispatch has
	// happened (R6: the priority header's entry/resident split — the
	// entry request gets low engine priority, resident turns high).
	entryDispatched bool
	// lastSizeBytes is the last known wire size (the estimate delta's
	// baseline; updated at PreRequest with the exact size).
	lastSizeBytes int64
	// pendingBlocks is the seat-time admission ESTIMATE still counted
	// against the engine's budget; converts to exact blocks at PreRequest
	// (Dispatched) and releases at exit. Without it a newcomer seats at
	// index-cost 0 and the budget understates until the exact keys land.
	pendingBlocks int64
	// awaitingEngine marks a pendingBlocks charge that Pick made on a
	// GUESSED engine: the binding when the member had one, else the
	// least-backlogged engine. PreRequest's pickedEndpoint is
	// AUTHORITATIVE and MOVES the charge when it disagrees
	// (reconcileEngineLocked) — the estimate error is therefore bounded
	// to admissions in flight between Pick and PreRequest, never
	// unbounded drift.
	awaitingEngine bool
	// waitingSince is the member's oldest queued head (B5: kept oldest).
	waitingSince time.Time
}

// inflightRec is one in-flight request's reconciliation record (C3).
type inflightRec struct {
	since     time.Time // Pick time (provisional) / PreRequest time (confirmed)
	confirmed bool
	main      bool // landed on the session's main chain (V1)
}

// estEntry is one cached admission estimate (M2): blocks for a given
// head size; recomputed when the index version or the size changes.
type estEntry struct {
	sizeBytes int64
	blocks    int64
}

type waiter struct {
	id string
	// session is the ordering layer's unit for this waiter (the session
	// attribute, or the member id when the request carried none). Refreshed
	// with the head at sync time.
	session  string
	since    time.Time // the OLDEST arrival (B5)
	newcomer bool
	// sizeBytes is the last seen head's wire size (the C4 refresh input).
	sizeBytes int64
	// coldTokens is the entry's cold-prefill estimate in TOKENS
	// (R1: one unit everywhere — blocks x blockSize).
	coldTokens int64
}

// ---- the per-engine group (PREREG-multipod-offload task 1) ----
//
// EVERYTHING that is a fact about ONE engine's HBM lives here: its block
// index (dedup is per engine — a block on A is worthless to B, and the
// global refcount over-counted up to N×), its budget, its measured
// capacity and liveness, its seats, waiters, pending estimates, cold-entry
// charge, estimate cache, rotation victims and counters.
//
// What is NOT here (identity, because a session spans engines): everSeated
// / everSeatedSession / recordSession / sessionRefs / lastSeen / the
// member→engine bindings / session seat counts (which SUM across engines —
// LSF's unit is the session, not the seat).
type engineGroup struct {
	// id is the endpoint id the EPP discovered ("", the sole group, in
	// engineGroups=single).
	id string

	index        *BlockIndex // attached at creation (block size from config)
	budgetBlocks int         // 0 until THIS engine's num_gpu_blocks arrives
	numBlocks    int
	// blockSizeMismatch is the loud per-engine wiring failure (a measured
	// CacheBlockSize that differs from the config's BlockSizeTokens).
	blockSizeMismatch error

	members map[string]*member
	waiters map[string]*waiter
	// pendingTotal is the sum of seated members' pendingBlocks (the
	// budget's estimate component).
	pendingTotal int64
	// coldInFlight is this engine's outstanding cold-entry charge: the
	// stagger cap is PER ENGINE (N engines ⇒ N concurrent cold prefills;
	// R11's "one cold entry is always allowed" proves liveness inside one
	// engine's cap, and it holds independently for each engine).
	coldInFlight int64

	// estCache/estVersion cache waiter admission estimates keyed on the
	// index's mutation version (M2: Picks run at ~1kHz; unchanged state
	// must not recompute ~150 × ~1k-block history scans). Per engine
	// because the versions are.
	estCache   map[string]estEntry
	estVersion uint64

	counters struct {
		served      int64 // dispatches served on this engine
		fcfsServed  int64 // pre-capacity FCFS dispatches counted here
		gateSkipped int64 // heads the size-aware gate turned away
		// forcedSeats counts seats taken OVER budget by an affinity
		// reconciliation (the observed endpoint wins; enforcement catches
		// up on the next pass). Must be ~0 in a healthy run.
		forcedSeats int64
		// homings counts members HOMED to this engine at first sight (the
		// E1/E2 split-skew accounting: per-engine member counts).
		homings int64
		exits   map[string]int64
	}
}

func newEngineGroup(id string, blockSizeTokens int, windowMode string) (*engineGroup, error) {
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return nil, err
	}
	idx.SetWindowMode(windowMode)
	g := &engineGroup{id: id, index: idx,
		members: map[string]*member{}, waiters: map[string]*waiter{},
		estCache: map[string]estEntry{}}
	g.counters.exits = map[string]int64{}
	return g, nil
}

// engineLabel is the metrics/event label value for a group id ("" — the
// sole single-mode group — renders as "single" so an unlabelled-series
// scrape never shows an empty label value).
func engineLabel(id string) string {
	if id == "" {
		return "single"
	}
	return id
}

// Scheduler is the group scheduler state machine over per-engine
// BlockIndexes. The 80% dispatch gate stays in the stack's saturation
// detector.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// observers are the optional metric hooks (nil in tests).
	onExit         func(engine, reason string, resTurns int)
	onEntry        func(engine string, waitSeconds float64)
	onServe        func(engine string)
	onEvent        func(Event)
	onRehome       func(member, from, to string)
	onEngineGone   func(engine string, members int)
	onMemberSource func(source string)

	// groups holds one engineGroup per engine, keyed by endpoint id. In
	// single mode there is exactly one, keyed "".
	groups map[string]*engineGroup
	// groupOrder is the sorted key list (deterministic iteration: every
	// cross-engine decision is taken in id order, never map order).
	groupOrder []string

	blockSize int

	// ---- GLOBAL identity state (a session spans engines) ----
	everSeated map[string]struct{}
	// everSeatedSession is the SESSION-level mirror of everSeated — the
	// ordering layer's "session-is-new" test ("new" == never seated within
	// the session's lifetime, the same everSeated semantics the member
	// newcomer flag has, and forgotten with the records themselves via
	// gcSilentRecordsLocked). It is kept EXACTLY in step by
	// markEverSeatedLocked/forgetEverSeatedLocked: with session := member
	// (no session attribute) the mirror is one-to-one, so the session keys
	// below reduce to today's newcomer test byte-for-byte.
	everSeatedSession map[string]struct{}
	// recordSession maps an everSeated record to its session unit, and
	// sessionRefs counts the live everSeated records per session (that
	// count IS what "session := member" degenerates to the member flag).
	recordSession map[string]string
	sessionRefs   map[string]int
	// lastSeen is the per-record activity time for the silence GC (R2):
	// updated at Dispatched and Settle; a headless, unseated record
	// older than TTL is forgotten.
	lastSeen map[string]time.Time
	// bindings is the member→engine HOMING (PREREG-multipod-offload):
	// chosen ONCE, at first sight of the member, and never changed while
	// the member is active. It survives rotation (a re-entering member
	// returns to the engine that holds its cache) and is dropped when the
	// member's record is forgotten (TTL/GC — a silent member may then be
	// re-homed) or when its engine disappears.
	bindings map[string]string

	// pickSeq synthesizes request ids for heads that lack one (M1).
	pickSeq int64
	// homeSeq rotates the tie-break among equally-loaded engines at homing
	// time. Round-robin rather than random: same anti-burst property (a
	// burst of newcomers splits instead of racing to one engine) and it
	// keeps EVERY multi-engine run reproducible — a randomized homing
	// would make the E1 equivalence test unrepeatable.
	homeSeq int64
	events  []Event

	counters struct {
		entryWaits   []float64
		reentryWaits []float64
		// R1/R2 diagnostics: both must read 0 with the gateway (every
		// request carries a real id end to end).
		emptyReqIDs  int64
		straySettles int64
		// affinityViolations counts requests that LANDED on an engine other
		// than the one their member is charged to (CORRECTNESS: the
		// observed endpoint wins, so each violation is repaired by a
		// rebind; a nonzero steady rate means the placement path is not
		// honouring the homing and every such request corrupts the
		// accounting until PreRequest repairs it).
		affinityViolations int64
		rebinds            int64
		rehomes            int64 // silent-member rehoming (stub: not wired)
		engineGone         int64
		exits              map[string]int64
		resTurns           []int
		refillShare        float64
		uncachedTok        int64
		totalTok           int64
		served             int64
		fcfsServed         int64
		// memberSources counts dispatches by the attribute the member id
		// came from (the run's effective-member-source echo).
		memberSources map[string]int64
	}
}

// NewScheduler builds the state machine. The group index attaches NOW (A5:
// block size from the token processor's config, known at startup).
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg = cfg.effective()
	s := &Scheduler{cfg: cfg, blockSize: cfg.BlockSizeTokens,
		groups:     map[string]*engineGroup{},
		everSeated: map[string]struct{}{}, lastSeen: map[string]time.Time{},
		everSeatedSession: map[string]struct{}{}, recordSession: map[string]string{},
		sessionRefs: map[string]int{}, bindings: map[string]string{}}
	s.counters.exits = map[string]int64{}
	s.counters.memberSources = map[string]int64{}
	if _, err := s.groupForLocked(""); err != nil {
		return nil, err
	}
	return s, nil
}

// groupForLocked returns (creating) the group for an engine id. In single
// mode every id collapses to the one group keyed "" (the pre-multi-pod
// behaviour: one index, one budget, "first capacity wins").
func (s *Scheduler) groupForLocked(id string) (*engineGroup, error) {
	if !s.cfg.perEngine() {
		id = ""
	}
	if g, ok := s.groups[id]; ok {
		return g, nil
	}
	g, err := newEngineGroup(id, s.blockSize, s.cfg.WindowMode)
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

// eng is the SINGLE-ENGINE accessor: the group every engine collapses to
// when engineGroups=single (and the group the legacy tests drive). It is
// the only place the multi-engine code is allowed to assume one engine.
//
// It does NOT lock — callers hold s.mu (or own the scheduler in a test), so
// it can be used from inside an already-locked decision path without
// re-entering the mutex.
func (s *Scheduler) eng() *engineGroup { return s.groups[""] }

// existingGroupLocked returns the group for an engine id WITHOUT creating it
// (a settle for an already-torn-down engine must not resurrect the group).
func (s *Scheduler) existingGroupLocked(id string) *engineGroup {
	if !s.cfg.perEngine() {
		id = ""
	}
	return s.groups[id]
}

// groupOfLocked is the member's group via its homing binding (nil when the
// member is unbound).
func (s *Scheduler) groupOfLocked(member string) *engineGroup {
	id, ok := s.bindings[member]
	if !ok {
		return nil
	}
	return s.groups[id]
}

// activeGroupsLocked is the deterministic engine list a Pick/decision pass
// iterates: the sole group in single mode, every discovered group (sorted)
// in per-engine mode.
func (s *Scheduler) activeGroupsLocked() []*engineGroup {
	if !s.cfg.perEngine() {
		g, err := s.groupForLocked("")
		if err != nil || g == nil {
			return nil
		}
		return []*engineGroup{g}
	}
	out := make([]*engineGroup, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		if !s.visibleEngineLocked(id) {
			continue
		}
		out = append(out, s.groups[id])
	}
	return out
}

// visibleEngineLocked reports whether a group id is a REAL engine in the
// current mode: the pooled "" group is an engine only when the policy pools
// (single mode). In per-engine mode it survives as the anonymous-gate
// fallback and must never be counted, reported or homed into.
func (s *Scheduler) visibleEngineLocked(id string) bool {
	if id == "" {
		return !s.cfg.perEngine()
	}
	return true
}

// SetNumBlocks attaches the measured capacity (num_gpu_blocks from the
// endpoint metrics). A measured block size that differs from the config
// value fails loudly (recorded; HasCapacity stays until it matches).
//
// It is SetNumBlocksFor("") — the single-pool form the legacy tests and
// single-mode runs use.
func (s *Scheduler) SetNumBlocks(blockSizeTokens, numBlocks int) error {
	return s.SetNumBlocksFor("", blockSizeTokens, numBlocks)
}

// SetNumBlocksFor attaches ONE engine's measured capacity. Capacity is
// per engine by construction: two engines each holding 40,183 blocks is
// 2 × HBM, not one pool of 40,183 ("first capacity wins" was correct only
// while a single pool was the model).
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
	g.budgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	return nil
}

// HasCapacity reports whether ANY engine has a measured num_gpu_blocks
// (the pre-capacity FCFS gate's condition).
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
// per-endpoint liveness test (an engine without metrics serves nothing
// while looking healthy).
func (s *Scheduler) HasCapacityFor(engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.cfg.perEngine() {
		engine = ""
	}
	g, ok := s.groups[engine]
	return ok && g.numBlocks > 0
}

// EngineGroupIDs returns the discovered engine ids (sorted) — the audit of
// the group set the policy is actually driving.
func (s *Scheduler) EngineGroupIDs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		if s.visibleEngineLocked(id) {
			out = append(out, id)
		}
	}
	return out
}

// GroupLabel resolves an endpoint key to the label the metrics and events
// carry for it: the engine's own id in per-engine mode, "single" for the one
// pooled group (an unlabelled-series scrape never sees an empty label value).
func (s *Scheduler) GroupLabel(engine string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.cfg.perEngine() {
		return engineLabel("")
	}
	return engineLabel(engine)
}

// HeterogeneousEngines reports the engine ids whose measured num_gpu_blocks
// deviates from the FIRST measured engine by more than tolFrac. A
// heterogeneous pool must not silently inherit another engine's budget
// (the plugin logs this loudly; it is a wiring fact, not a tuning knob).
func (s *Scheduler) HeterogeneousEngines(tolFrac float64) map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heterogeneousEnginesLocked(tolFrac)
}

func (s *Scheduler) heterogeneousEnginesLocked(tolFrac float64) map[string]int {
	out := map[string]int{}
	var base int
	var baseID string
	for _, id := range s.groupOrder {
		if !s.visibleEngineLocked(id) {
			continue
		}
		g := s.groups[id]
		if g.numBlocks <= 0 {
			continue
		}
		if base == 0 {
			base, baseID = g.numBlocks, id
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
	_ = baseID
	return out
}

// CapacityBlockSizes reports the configured-vs-measured block-size
// mismatches per engine (the loud failure surface).
func (s *Scheduler) BlockSizeMismatch() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range s.groupOrder {
		if err := s.groups[id].blockSizeMismatch; err != nil {
			return err
		}
	}
	return nil
}

// HashTokens computes the chained token-block keys — the tier-A input
// (CP2b): blockSizeTokens-token blocks, each key chained from the
// previous, the cache salt folded into the first block.
func (s *Scheduler) HashTokens(tokens []uint32, cacheSalt string, blockSizeTokens int) []Hash {
	if blockSizeTokens <= 0 {
		blockSizeTokens = s.blockSize
	}
	if blockSizeTokens <= 0 || len(tokens) == 0 {
		return nil
	}
	n := len(tokens) / blockSizeTokens
	out := make([]Hash, 0, n)
	var parent uint64
	salt := sha256.Sum256([]byte(cacheSalt))
	saltSeed := binary.LittleEndian.Uint64(salt[:8])
	for i := 0; i < n; i++ {
		h := sha256.New()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], saltSeed^parent)
		h.Write(b[:])
		for j := i * blockSizeTokens; j < (i+1)*blockSizeTokens; j++ {
			binary.LittleEndian.PutUint32(b[:4], tokens[j])
			h.Write(b[:4])
		}
		parent = binary.LittleEndian.Uint64(h.Sum(nil)[:8])
		out = append(out, Hash(parent))
	}
	return out
}

// ---- Pick (§5): the waiting set is the queue-head snapshot ----

// PickFromHeads runs one dispatch decision against the current queue
// heads and returns the index into heads of the chosen request, or -1.
//
// The multi-engine engine is chosen INSIDE (see PickFromHeadsEngine); the
// plugin needs only the head, because the engine attribution arrives
// authoritatively at PreRequest (pickedEndpoint) — the pendingEngine idiom.
func (s *Scheduler) PickFromHeads(heads []Head, now time.Time) int {
	_, i := s.PickFromHeadsEngine(heads, now)
	return i
}

// PickFromHeadsEngine is PickFromHeads with the engine it served from ("" for
// an anonymous R9 head). Per pass: derive, reap, sync waiters (homing
// unbound newcomers), enforce each engine's budget, rotate each engine,
// then ONE outer engine-selection step whose fairness key is relative
// backlog per seat.
//
// Cross-engine fairness is deliberately NOT global oldest-first: taken over
// the union of engines that re-creates head-of-line blocking across them
// (engine A's 900s-old head starves engine B's queue while A is full).
// The engine with the deepest queue per seat goes first; ties break on the
// candidate's arrival and then on the engine id, so the decision is total
// and reproducible.
func (s *Scheduler) PickFromHeadsEngine(heads []Head, now time.Time) (string, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deriveLocked() // derive-don't-accumulate: recomputed, never trusted across paths
	s.reapInflightLocked(now)
	s.syncWaitersLocked(heads, now)
	groups := s.activeGroupsLocked()
	for _, g := range groups {
		s.enforceBudgetLocked(g, now)
	}
	s.rotateLocked(now)

	// member heads, OLDEST-FIRST WITHIN THEIR ENGINE (the cross-engine
	// order comes from the backlog-per-seat step below). The size-aware
	// dispatch gate (v2) is FIRST-FIT: dispatch the first head of that
	// engine that fits, SKIP the ones that don't (they stay queued). The
	// first deployed v2 checked only the single oldest head and returned -1
	// when it didn't fit — with the budget near-full the oldest head NEVER
	// fit, nothing else could dispatch, no busy period ever completed (no
	// quantum-complete victims), and the C8 fix kept the waiting members
	// from being idle-evicted: a self-sustaining head-of-line freeze (the
	// 02:13Z v2 incident; dump banked as v2-freeze-goroutines.txt).
	type cand struct {
		i      int
		h      Head
		engine string
	}
	perEngine := make(map[string][]cand, len(groups))
	for i, h := range heads {
		if h.Member == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		g := s.groupOfLocked(h.Member)
		if g == nil {
			continue
		}
		m, seated := g.members[h.Member]
		if !seated || g.index.ExitingLocked(h.Member) {
			continue
		}
		if m.inflight >= s.cfg.InFlightCap {
			continue
		}
		perEngine[g.id] = append(perEngine[g.id], cand{i, h, g.id})
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
		var chosen *cand
		for k := range cands {
			if s.cfg.GateMode == "size-aware" && !s.gateAllowsLocked(g, cands[k].h) {
				g.counters.gateSkipped++ // diagnostic: heads the gate turned away
				continue
			}
			chosen = &cands[k]
			break
		}
		if chosen == nil {
			continue
		}
		score := s.backlogPerSeatLocked(g)
		if winner == nil || score > winner.score ||
			(score == winner.score && chosen.h.Arrival.Before(winner.c.h.Arrival)) ||
			(score == winner.score && chosen.h.Arrival.Equal(winner.c.h.Arrival) && g.id < winner.c.engine) {
			winner = &best{c: *chosen, score: score}
		}
	}
	if winner != nil {
		g := s.groups[winner.c.engine]
		m := g.members[winner.c.h.Member]
		s.observeMemberSourceLocked(winner.c.h.Source)
		// M1: a dispatch is NEVER counted without an id — synthesize one
		// when the head lacks ReqID (the counter follows the map).
		rid := winner.c.h.ReqID
		if rid == "" {
			s.pickSeq++
			rid = fmt.Sprintf("synthetic-%d", s.pickSeq)
		}
		if m.inflightIDs == nil {
			m.inflightIDs = map[string]inflightRec{}
		}
		m.inflightIDs[rid] = inflightRec{since: now} // provisional (C3): confirmed at Dispatched
		m.inflight = len(m.inflightIDs)
		m.lastActivity = now
		g.counters.served++
		s.counters.served++
		if s.onServe != nil {
			s.onServe(engineLabel(g.id))
		}
		return g.id, winner.c.i
	}
	// R9: no member head is dispatchable — serve the oldest ANONYMOUS
	// head (FCFS pass-through, the siblings' behaviour; anonymous
	// traffic is never tracked, never seated, never starved).
	for i, h := range heads {
		if h.Member == "" {
			if s.cfg.GateMode == "size-aware" && !s.gateAllowsLocked(s.defaultGroupLocked(), h) {
				continue
			}
			return "", i
		}
	}
	return "", -1
}

// backlogPerSeatLocked is the cross-engine fairness key: queued members per
// seat held on that engine. NOT kv_cache_usage — under the group scheduler
// every engine sits at ~0.95 by construction (F3), so any least-loaded rule
// keyed on cache usage degenerates to round-robin.
func (s *Scheduler) backlogPerSeatLocked(g *engineGroup) float64 {
	seats := len(g.members)
	if seats <= 0 {
		seats = 1
	}
	// queued DEMAND on this engine: unseated waiters plus seated members that
	// have a queued head (a resident waiting for a turn is backlog too — it is
	// what the engine cannot serve yet).
	backlog := len(g.waiters)
	for _, m := range g.members {
		if !m.waitingSince.IsZero() {
			backlog++
		}
	}
	return float64(backlog) / float64(seats)
}

// defaultGroupLocked is the group single-mode decisions use (and the one an
// anonymous head's gate is checked against).
func (s *Scheduler) defaultGroupLocked() *engineGroup {
	g, err := s.groupForLocked("")
	if err != nil {
		return nil
	}
	return g
}

// gateAllowsLocked is the size-aware dispatch gate for ONE engine: current
// usage (NE + pending estimates on that engine) plus this request's new
// blocks must fit the budget. Computed from the EPP's OWN accounting — never
// the scraped kv_cache_usage (stale by a scrape interval; kept as a
// cross-check gauge only). A member's own admission estimate (pendingBlocks)
// is NOT double-counted: its new blocks are est minus what its pending
// already reserves.
func (s *Scheduler) gateAllowsLocked(g *engineGroup, h Head) bool {
	if g == nil || g.budgetBlocks <= 0 {
		return true
	}
	est := s.estimateCostCachedLocked(g, h.Member, h.SizeBytes)
	if m, ok := g.members[h.Member]; ok && m.pendingBlocks > 0 {
		est = max(est-m.pendingBlocks, 0)
	}
	// M4: ALL seated blocks (exiting members' blocks stay pinned until
	// they drain) + pending estimates + this request's new blocks.
	all := int64(g.index.UniqueBlocksLocked()) + g.pendingTotal
	return all+est <= int64(g.budgetBlocks)
}

// FCFSPick returns the oldest head (the pre-capacity pass-through, A5).
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

// FCFSServed counts a pre-capacity dispatch (the gauge substrate) and
// returns the engine label it was booked to (the sole group in single mode;
// with per-engine groups and no capacity anywhere yet, the label is the
// default group's "single").
func (s *Scheduler) FCFSServed() (int64, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
	label := engineLabel("")
	if g := s.defaultGroupLocked(); g != nil {
		g.counters.fcfsServed++
		label = engineLabel(g.id)
	}
	return s.counters.fcfsServed, label
}

// syncWaitersLocked reconciles the waiter view with the queue-head
// snapshot: new members become waiters IN THEIR HOMED ENGINE (the binding
// when they have one — a re-entrant returns to the engine holding its
// cache — else the least-backlogged engine), vanished waiters leave, and
// member waiting-since tracks the oldest queued head (the victim
// preference's input).
func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Member == "" {
			continue
		}
		present[h.Member] = h.Arrival
		if g := s.groupOfLocked(h.Member); g != nil {
			if m, seated := g.members[h.Member]; seated {
				if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
					m.waitingSince = h.Arrival
				}
				m.lastActivity = now
				continue
			}
		}
		g := s.homeEngineLocked(h.Member)
		if g == nil {
			continue
		}
		w, ok := g.waiters[h.Member]
		if !ok {
			w = &waiter{id: h.Member, newcomer: s.isNewcomerLocked(h.Member)}
			g.waiters[h.Member] = w
		}
		// the ordering layer's unit follows the head (session attribute,
		// else the member key — base mode)
		w.session = sessionUnitOf(h.Session, h.Member)
		// B5: keep the OLDEST arrival across fan-out re-arrivals
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
		w.sizeBytes = h.SizeBytes
		w.coldTokens = s.estimateCostCachedLocked(g, h.Member, h.SizeBytes) * int64(s.blockSize) // tokens (R1)
	}
	// vanished: members clear their waiting marker; waiters leave
	for _, g := range s.activeGroupsLocked() {
		for id, m := range g.members {
			if _, ok := present[id]; !ok {
				m.waitingSince = time.Time{}
			}
		}
		for id := range g.waiters {
			if _, ok := present[id]; !ok {
				delete(g.waiters, id)
			}
		}
	}
}

// homeEngineLocked resolves (and on first sight RECORDS) a member's engine.
//
// PREREG-multipod-offload: the engine is chosen ONCE, at first seat, and
// never changed while the member is active; rehoming is limited to SILENT
// members (dropped bindings after TTL/GC, and engines that disappear). The
// choice is made on EPP-side backlog per seat — never kv_cache_usage, which
// the scheduler pins at ~0.95 by construction (F3). Ties break round-robin
// so a burst of newcomers splits across the engines instead of racing to
// one (and so every multi-engine run stays reproducible).
func (s *Scheduler) homeEngineLocked(member string) *engineGroup {
	if id, ok := s.bindings[member]; ok {
		if g, ok := s.groups[id]; ok {
			return g
		}
		// the bound engine is gone: re-home on a live one below
		delete(s.bindings, member)
	}
	groups := s.activeGroupsLocked()
	if len(groups) == 0 {
		return nil
	}
	// prefer engines that HAVE capacity (an engine whose metrics never
	// arrived cannot seat); with none, fall back to the whole set so the
	// behaviour is the pre-capacity one.
	var ready []*engineGroup
	for _, g := range groups {
		if g.numBlocks > 0 {
			ready = append(ready, g)
		}
	}
	if len(ready) == 0 {
		ready = groups
	}
	best := ready
	bestScore := s.backlogPerSeatLocked(ready[0])
	for _, g := range ready[1:] {
		sc := s.backlogPerSeatLocked(g)
		switch {
		case sc < bestScore:
			best, bestScore = []*engineGroup{g}, sc
		case sc == bestScore:
			best = append(best, g)
		}
	}
	g := best[0]
	if len(best) > 1 {
		g = best[int(s.homeSeq%int64(len(best)))]
		s.homeSeq++
	}
	s.bindings[member] = g.id
	g.counters.homings++
	return g
}

// reconcileEngineLocked is the pendingEngine idiom's second half: the
// OBSERVED endpoint wins. A member charged to engine A whose request lands
// on B is MOVED to B (unseat-and-rebind: one member's correction instead of
// unbounded accounting drift — the index is an optimistic model of where
// blocks ARE, and a stale binding drifts permanently and silently).
func (s *Scheduler) reconcileEngineLocked(id, observed string, now time.Time) *engineGroup {
	if !s.cfg.perEngine() {
		return s.defaultGroupLocked()
	}
	from := s.groupOfLocked(id)
	if from == nil {
		// unbound (never homed): bind on the observed engine
		g, err := s.groupForLocked(observed)
		if err != nil {
			return nil
		}
		s.bindings[id] = g.id
		return g
	}
	if from.id == observed {
		return from
	}
	to, err := s.groupForLocked(observed)
	if err != nil || to == nil {
		return from
	}
	m := from.members[id]
	if m == nil {
		s.bindings[id] = to.id
		return to
	}
	s.counters.affinityViolations++
	s.counters.rebinds++
	from.coldInFlight = max(from.coldInFlight-m.coldEntry, 0)
	from.pendingTotal -= m.pendingBlocks
	from.index.UnseatLocked(id) // the blocks are NOT on this engine (observation wins)
	delete(from.members, id)
	to.members[id] = m
	m.engine = to.id
	s.bindings[id] = to.id
	to.pendingTotal += m.pendingBlocks
	to.coldInFlight += m.coldEntry
	if ok, _ := to.index.TrySeatLocked(id, to.budgetBlocks); !ok {
		// the engine is over its budget for this member: seat it anyway —
		// it demonstrably holds the blocks — and let enforcement evict
		// (an over-budget counter is recoverable; a member absent from the
		// engine that is serving it is not).
		to.index.SeatForceLocked(id)
		to.counters.forcedSeats++
	}
	s.recordEvent("REBIND", id, fmt.Sprintf("from=%s to=%s", engineLabel(from.id), engineLabel(to.id)), now)
	if s.onRehome != nil {
		s.onRehome(id, from.id, to.id)
	}
	return to
}

// estimateCostCachedLocked is estimateCostLocked behind the M2 cache:
// valid while the engine's index version is unchanged and the head size
// matches.
func (s *Scheduler) estimateCostCachedLocked(g *engineGroup, id string, sizeBytes int64) int64 {
	v := g.index.VersionLocked()
	if g.estVersion != v {
		g.estCache = map[string]estEntry{}
		g.estVersion = v
	}
	if e, ok := g.estCache[id]; ok && e.sizeBytes == sizeBytes {
		return e.blocks
	}
	b := s.estimateCostLocked(g, id, sizeBytes)
	g.estCache[id] = estEntry{sizeBytes: sizeBytes, blocks: b}
	return b
}

// estimateCostLocked is the conservative admission cost of a QUEUED
// request (S0-1 b/c) against ONE engine's index: the session's persisted
// history blocks not held by that engine's non-exiting group, plus the
// wire-size DELTA over the last known body (bytes/4/blockSize, zero sharing
// assumed). A session with no history costs its full wire size the same way.
func (s *Scheduler) estimateCostLocked(g *engineGroup, id string, sizeBytes int64) int64 {
	hist := g.index.SetBlocks(id) // persisted across residencies (CP1)
	known := int64(0)
	for _, h := range hist {
		if !g.index.HasNELocked(h) {
			known++
		}
	}
	// the baseline is the session's last DISPATCHED body size (the index
	// record, updated at PreRequest) — never the current head's size, or
	// the second sight of the same head would estimate zero new blocks
	// (found by the integration test: w seated on the retry pick)
	lastBytes := g.index.LastSizeBytesLocked(id)
	delta := sizeBytes - lastBytes
	if delta < 0 {
		delta = 0
	}
	return known + delta/4/int64(s.blockSize)
}

// ---- PreRequest (post-dispatch): the exact tier-A correction ----

// Dispatched records a released request at PreRequest without an observed
// endpoint (the legacy/single-engine form): the member's binding decides.
func (s *Scheduler) Dispatched(id, reqID string, keys []Hash, sizeBytes int64, now time.Time) {
	s.DispatchedOn(id, reqID, "", keys, sizeBytes, now)
}

// DispatchedOn records a released request at PreRequest on the endpoint the
// framework ACTUALLY picked: the exact keys replace the estimate (D13
// absorbs the error at the next enforcement), in-flight and the cold charge
// are taken, and the size baseline moves. engine=="" means "the member's
// binding" (no observation).
func (s *Scheduler) DispatchedOn(id, reqID, engine string, keys []Hash, sizeBytes int64, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOfLocked(id)
	if g == nil {
		if engine == "" || !s.cfg.perEngine() {
			g = s.defaultGroupLocked()
		} else if gg, err := s.groupForLocked(engine); err == nil {
			g = gg
		}
		if g == nil {
			return
		}
		if s.cfg.perEngine() {
			s.bindings[id] = g.id
		}
	} else if engine != "" {
		// pendingEngine reconciliation: the observed endpoint is authoritative
		g = s.reconcileEngineLocked(id, engine, now)
		if g == nil {
			return
		}
	}
	g.index.SetSizeLocked(id, sizeBytes) // the estimate baseline
	onMain := g.index.SettleDispatchedAtLocked(id, reqID, keys, now)
	if m, ok := g.members[id]; ok {
		m.lastActivity = now
		m.lastSizeBytes = sizeBytes
		m.awaitingEngine = false
		// C3/M1/R1: reconcile the in-flight id. A non-empty reqID
		// confirms-or-adds. An EMPTY reqID must not mint a second id
		// (Pick already recorded a provisional one — two records for
		// one request leak past the reap): confirm the OLDEST
		// PROVISIONAL record, and synthesize only when none exists.
		if reqID == "" {
			s.counters.emptyReqIDs++
			reqID = ""
			oldestKey, oldestTime := "", time.Time{}
			for rid, rec := range m.inflightIDs {
				if rec.confirmed {
					continue
				}
				if oldestKey == "" || rec.since.Before(oldestTime) {
					oldestKey, oldestTime = rid, rec.since
				}
			}
			if oldestKey != "" {
				reqID = oldestKey
			} else {
				s.pickSeq++
				reqID = fmt.Sprintf("synthetic-%d", s.pickSeq)
			}
		}
		if m.inflightIDs == nil {
			m.inflightIDs = map[string]inflightRec{}
		}
		m.inflightIDs[reqID] = inflightRec{since: now, confirmed: true, main: onMain}
		m.inflight = len(m.inflightIDs)
		m.mainInflight = 0
		for _, rec := range m.inflightIDs {
			if rec.main {
				m.mainInflight++
			}
		}
		if m.pendingBlocks > 0 {
			g.pendingTotal -= m.pendingBlocks
			m.pendingBlocks = 0 // the exact keys are in NE now (D13 reconciles)
		}
		if !m.coldDone && m.coldEntry > 0 {
			// refine the charge to the exact token count (R1: one unit)
			g.coldInFlight = max(g.coldInFlight-m.coldEntry, 0)
			m.coldEntry = int64(len(keys)) * int64(s.blockSize)
			g.coldInFlight += m.coldEntry
		}
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry
// charge is released HERE (B3: not at end of decode), exactly once.
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groupOfLocked(id); g != nil {
		s.releaseColdLocked(g, id)
	}
}

func (s *Scheduler) releaseColdLocked(g *engineGroup, id string) {
	m, ok := g.members[id]
	if !ok || m.coldDone || m.coldEntry <= 0 {
		return
	}
	g.coldInFlight = max(g.coldInFlight-m.coldEntry, 0)
	m.coldEntry = 0
	m.coldDone = true
}

// ---- settle (ResponseBody at EndOfStream: the finalizer) ----

// Settle records a terminated dispatched request without an observed
// endpoint (the legacy/single-engine form).
func (s *Scheduler) Settle(id, reqID string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.SettleOn(id, reqID, "", usage, natural, now)
}

// SettleOn records a terminated dispatched request on the endpoint it ran
// on: the in-flight decrement for EVERY cause; n_max counts every settled
// turn (A8); the payback ledger only on natural completion.
//
// F6: an engine mismatch here means the request landed on an engine other
// than the member's binding AFTER PreRequest reconciled — a CORRECTNESS
// violation (it corrupts the charged engine's record). It is counted, never
// silently folded: the accounting stays with the binding the PreRequest
// committed, and the counter is the run-level alarm.
func (s *Scheduler) SettleOn(id, reqID, engine string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSeen[id] = now
	g := s.groupOfLocked(id)
	if g == nil {
		if !s.cfg.perEngine() || engine == "" {
			g = s.defaultGroupLocked()
		} else {
			g = s.existingGroupLocked(engine) // never resurrect a torn-down engine
		}
	}
	if g == nil {
		return
	}
	if engine != "" && s.cfg.perEngine() && g.id != engine {
		s.counters.affinityViolations++
		s.recordEvent("AFFINITY-VIOLATION", id, fmt.Sprintf("bound=%s settled_on=%s", engineLabel(g.id), engineLabel(engine)), now)
	}
	m, seated := g.members[id]
	if seated {
		// C3/M1: remove the in-flight record; the counter follows the
		// map (derive-don't-accumulate). A missing reqID removes the
		// OLDEST record (defensive legacy path — never blind).
		removed := false
		removedMain := false
		if reqID != "" {
			// R2: a known id removes itself; an UNKNOWN non-empty id
			// removes NOTHING (the oldest-fallback would delete a
			// different live request's record after a late reap or a
			// duplicate settle — the member would undercount and evict
			// mid-flight). Counted as a stray.
			if rec, had := m.inflightIDs[reqID]; had {
				delete(m.inflightIDs, reqID)
				removed, removedMain = true, rec.main
			} else {
				s.counters.straySettles++
			}
		} else if len(m.inflightIDs) > 0 {
			// empty reqID (defensive legacy path): remove the OLDEST
			key, oldest := "", time.Time{}
			for rid, rec := range m.inflightIDs {
				if key == "" || rec.since.Before(oldest) {
					key, oldest = rid, rec.since
				}
			}
			if rec, had := m.inflightIDs[key]; had {
				delete(m.inflightIDs, key)
				removed, removedMain = true, rec.main
			}
		}
		if removed {
			m.inflight = len(m.inflightIDs)
			m.mainInflight = 0
			for _, rec := range m.inflightIDs {
				if rec.main {
					m.mainInflight++
				}
			}
		}
		m.lastActivity = now
		s.releaseColdLocked(g, id) // S0-2: any termination releases the charge
		// S4/V1/R3a: the busy period closes only when a MAIN-chain
		// request was removed AND the main chain drained to zero — a
		// sub-agent request settling while the main chain is idle must
		// not burn a quantum unit.
		if removed && removedMain && m.mainInflight == 0 {
			m.busyPeriods++
			m.lastCompletion = now
		}
	}
	if !seated {
		return
	}
	m.resTurns++ // A8: every settled turn counts toward n_max
	// the settled request was the tracked queued head: clear the marker
	// (the next Pick re-establishes it if more heads remain — the TTL
	// path cannot see stale queue presence)
	m.waitingSince = time.Time{}
	// §6.3: an exiting member leaves when its in-flight requests drain
	if g.index.ExitingLocked(id) && m.inflight == 0 {
		s.exitLocked(g, m, "exiting-drain", now)
		return
	}
	if !natural {
		return
	}
	uncached := int64(usage.Prompt - usage.Cached)
	if uncached < 0 {
		uncached = 0
	}
	if !m.entryRecorded {
		m.entryCost = uncached // the entry turn's cold cost (measured; may be 0 when fully warm)
		m.entryRecorded = true
	} else {
		m.earned += int64(usage.Cached)
	}
	s.counters.resTurns = append(s.counters.resTurns, m.resTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(usage.Prompt)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// ---- rotation (§7) ----

// eligibleLocked is the LEGACY ledger rule (rotationMode "ledger"):
// paid / idle / nmax. The C8 fix is always-on: a member with a live
// queued head is never idle (its lastActivity cannot be refreshed when
// Picks stop — gate closure used to manufacture idle exits).
func (s *Scheduler) eligibleLocked(m *member, now time.Time) (string, bool) {
	if m.inflight > 0 {
		return "", false
	}
	if s.cfg.RotationMode == "quantum" {
		// the busy-period quantum: eligible when Q busy periods are
		// complete (never, under Q=∞ — the pause arm) or on silence.
		if s.cfg.quantumFinite() && m.busyPeriods >= s.cfg.quantum() {
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

// silentLocked is the idle test with the C8 fix: silent means no
// activity AND no queued head (a waiting member wants to run).
func (s *Scheduler) silentLocked(m *member, now time.Time) bool {
	if !m.waitingSince.IsZero() {
		return false
	}
	return now.Sub(m.lastActivity) > time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// deriveLocked recomputes the derived aggregates from member records
// (derive-don't-accumulate, BUILD-PLAN §1): a leaked or missed increment
// anywhere cannot persist past the next decision point. Per engine.
func (s *Scheduler) deriveLocked() {
	for _, g := range s.groups {
		var cold, pend int64
		for _, m := range g.members {
			cold += m.coldEntry
			pend += m.pendingBlocks
		}
		g.coldInFlight = cold
		g.pendingTotal = pend
	}
}

// reapInflightLocked drops in-flight ids whose termination was never
// observed (C3): provisional ids (chosen at Pick, never reaching
// PreRequest) after 30s; confirmed ids after InflightReapSeconds (the
// harness request timeout is 3600s — anything older is a leak).
func (s *Scheduler) reapInflightLocked(now time.Time) {
	for _, g := range s.groups {
		for _, m := range g.members {
			if len(m.inflightIDs) == 0 {
				continue
			}
			reaped := false
			for rid, rec := range m.inflightIDs {
				limit := 30 * time.Second
				if rec.confirmed {
					limit = time.Duration(s.cfg.reapSeconds() * float64(time.Second))
				}
				if now.Sub(rec.since) > limit {
					delete(m.inflightIDs, rid)
					reaped = true
				}
			}
			if reaped {
				m.inflight = len(m.inflightIDs)
				m.mainInflight = 0
				for _, rec := range m.inflightIDs {
					if rec.main {
						m.mainInflight++
					}
				}
				if m.inflight == 0 {
					m.lastCompletion = now
					// M3(a): the member's requests are all dead — release
					// its cold charge or the valve freezes the queue until
					// the reap bound (the whistle scenario).
					s.releaseColdLocked(g, m.id)
				}
			}
		}
	}
}

// rotateLocked seats waiters, PER ENGINE: each engine's own budget, its own
// cold-entry cap and its own victim set, so one engine's head-of-line can
// never stall another's admission.
func (s *Scheduler) rotateLocked(now time.Time) {
	groups := s.activeGroupsLocked()
	anyWaiters := false
	for _, g := range groups {
		if len(g.waiters) > 0 {
			anyWaiters = true
			break
		}
	}
	if !anyWaiters {
		return
	}
	// the LSF key's seat component is computed ONCE per rotate and kept
	// current as this loop seats (and re-computed after an exit): the
	// per-waiter cost stays O(1) instead of O(members) — with base mode's
	// ~+29% members and a deep waiter list the naive form is quadratic.
	// It SUMS across engines: LSF's unit is the session, and a session's
	// seats may be spread over several engines even though budgets are not.
	var seats map[string]int
	if s.cfg.SessionOrder == "lsf" {
		seats = s.sessionSeatCountsLocked()
	}
	for _, g := range groups {
		s.rotateEngineLocked(g, &seats, now)
	}
}

func (s *Scheduler) rotateEngineLocked(g *engineGroup, seatsP *map[string]int, now time.Time) {
	seats := *seatsP
	for {
		head := s.entryHeadLocked(g, seats)
		if head == nil {
			return
		}
		ne := int64(g.index.UniqueBlocksNELocked()) + g.pendingTotal
		estBlocks := head.coldTokens / int64(s.blockSize)
		if ne+estBlocks <= int64(g.budgetBlocks) && s.coldAllows(g, head) {
			seatedSession := sessionUnitOf(head.session, head.id)
			if !s.seatLocked(g, head, now) {
				// C4 (always-on, now PER ENGINE): TrySeat failed although
				// the estimate passed — refresh the head's estimate from
				// the authoritative index state and RETURN. Continuing
				// with the stale estimate spins this loop forever under
				// s.mu (the 13:00 deadlock class, reproduced by the golden
				// scenario at sim≈1114).
				head.coldTokens = s.estimateCostLocked(g, head.id, head.sizeBytes) * int64(s.blockSize)
				return
			}
			if seats != nil {
				seats[seatedSession]++
			}
			continue
		}
		if reason, victim := s.rotationVictimLocked(g, now); victim != nil {
			if victim.inflight > 0 {
				g.index.MarkExitingLocked(victim.id)
				s.countExitLocked(g, "exiting")
				*seatsP = seats
				return
			}
			s.exitLocked(g, victim, reason, now)
			if seats != nil {
				seats = s.sessionSeatCountsLocked() // an exit frees a seat
				*seatsP = seats
			}
			continue
		}
		*seatsP = seats
		return
	}
}

// entryHeadLocked is D1 — the ONLY place the session ordering layer acts.
// Victim choice (D2) and dispatch order (D3) are untouched by it. It scans
// ONE engine's waiters (the entry decision is that engine's admission; the
// cross-engine order is Pick's backlog-per-seat step).
//
//	"off"      — newcomers first (D7), oldest within the class then id,
//	             else oldest waiting (A4/A7: re-entrants keep their
//	             original since). Today's order.
//	"newcomer" — the newcomer CLASS is the session's: a branch of an
//	             already-served session queues FIFO instead of jumping.
//	"lsf"      — (session seats held ↑, session-is-new first, oldest, id):
//	             least-seats-first. At c128 the fair share is 0.25 seats
//	             per session, so the work-conserving cap IS this order
//	             (WAVE-1 pre-registration: one rule, not two).
//
// With session := member (no session attribute — base mode) all three modes
// reduce EXACTLY to "off": a waiter's session holds no seat (all zero) and
// the session mirror of everSeated is one-to-one with the member flag. That
// equality is structural here and pinned end-to-end by
// TestSessionOrderGoldenEquality.
func (s *Scheduler) entryHeadLocked(g *engineGroup, seats map[string]int) *waiter {
	if seats == nil && s.cfg.SessionOrder == "lsf" {
		seats = s.sessionSeatCountsLocked() // a direct caller (tests) gets a fresh count
	}
	var head *waiter
	var headKey entryKey
	for _, w := range g.waiters {
		k := s.entryKeyLocked(w, seats)
		// the key is a TOTAL order (id breaks every tie), so the map's
		// iteration order cannot leak into the decision.
		if head == nil || k.less(headKey) {
			head, headKey = w, k
		}
	}
	return head
}

// entryKey is the D1 ordering key. seats/isNew are 0-or-1 ranks (lower =
// first) so the comparison is plain lexicographic.
type entryKey struct {
	seats int
	isNew int
	since time.Time
	id    string
}

func (k entryKey) less(o entryKey) bool {
	if k.seats != o.seats {
		return k.seats < o.seats
	}
	if k.isNew != o.isNew {
		return k.isNew < o.isNew
	}
	if !k.since.Equal(o.since) {
		return k.since.Before(o.since)
	}
	return k.id < o.id
}

func (s *Scheduler) entryKeyLocked(w *waiter, seats map[string]int) entryKey {
	k := entryKey{since: w.since, id: w.id}
	switch s.cfg.SessionOrder {
	case "lsf":
		sess := sessionUnitOf(w.session, w.id)
		k.seats = seats[sess]
		k.isNew = s.sessionNewRankLocked(sess)
	case "newcomer":
		k.isNew = s.sessionNewRankLocked(sessionUnitOf(w.session, w.id))
	default: // "off"
		if !w.newcomer {
			k.isNew = 1
		}
	}
	return k
}

// sessionUnitOf is the ordering layer's unit: the request's session ATTRIBUTE
// when it carried one, else the member key (base mode: session := branch).
// This single line is what makes base mode and session mode the same code.
func sessionUnitOf(sessionAttr, member string) string {
	if sessionAttr != "" {
		return sessionAttr
	}
	return member
}

// sessionNewRankLocked is 0 for a session that has never been seated, 1 for a
// served one — the session-level counterpart of the member newcomer flag.
func (s *Scheduler) sessionNewRankLocked(sess string) int {
	if _, served := s.everSeatedSession[sess]; served {
		return 1
	}
	return 0
}

// sessionSeatCountsLocked counts the seats held per session over EVERY
// engine (a session spans engines; LSF's unit is the session), in one
// O(members) pass. rotateLocked builds it once and keeps it current
// (increment on a seat, recompute after an exit) so the entry key stays O(1)
// per waiter.
func (s *Scheduler) sessionSeatCountsLocked() map[string]int {
	counts := make(map[string]int, len(s.groups))
	for _, g := range s.groups {
		for _, m := range g.members {
			counts[sessionUnitOf(m.session, m.id)]++
		}
	}
	return counts
}

// coldAllows staggers ALL cold entries IN ONE ENGINE (B6: re-entries too —
// WEKA re-entries are fully cold), in TOKENS (R1). R11's rule: one cold
// entry is ALWAYS allowed — a session bigger than the cap must be seatable
// when nothing cold is in flight, or newcomers-first + hold-the-line
// deadlocks the queue on it forever.
//
// Under N engines the cap is per engine, so the system-wide cold-entry
// concurrency is at most N. The proof is unchanged inside each engine: its
// charge is its own, its admission loop is its own, and the "always one"
// clause keeps ITS queue live. Per-engine is also what makes the stagger
// meaningful: two engines have two independent prefill slots.
func (s *Scheduler) coldAllows(g *engineGroup, w *waiter) bool {
	if s.cfg.ValveMode == "off" {
		return true // the stagger moved to the engine (Phase-2 deployment)
	}
	return g.coldInFlight == 0 || g.coldInFlight+w.coldTokens <= s.cfg.ColdEntryCapTokens
}

// rotationVictimLocked (§7 demand-driven exit) over ONE engine's members.
// LEGACY order (B4, C2-as-shipped): eligible members sorted by
// (has-waiting, seatTime) — inverted vs ALGORITHM §7, kept as the default.
// SPEC order (victimOrder "spec"): no-queued-turn members first, then oldest
// last completion. The engine-idle rule stays the liveness fallback — and it
// scans ONLY THIS ENGINE: an engine whose members are all in flight must not
// borrow another engine's idleness to justify an eviction, and an idle
// engine must not spend a decision on waiters bound elsewhere.
func (s *Scheduler) rotationVictimLocked(g *engineGroup, now time.Time) (string, *member) {
	var best *member
	var bestReason string
	bestHasWaiting := false
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(m, now)
		if !ok {
			continue
		}
		hasWaiting := !m.waitingSince.IsZero()
		var better bool
		if s.cfg.VictimOrder == "spec" {
			// spec: prefer NO queued turn, then oldest last completion
			better = best == nil ||
				(!hasWaiting && bestHasWaiting) ||
				(hasWaiting == bestHasWaiting && olderCompletion(m, best))
		} else {
			// legacy: prefer members WITH a queued turn, earliest-seated
			better = best == nil ||
				(hasWaiting && !bestHasWaiting) ||
				(hasWaiting == bestHasWaiting && m.seatTime.Before(best.seatTime)) ||
				(hasWaiting == bestHasWaiting && m.seatTime.Equal(best.seatTime) && m.id < best.id)
		}
		if better {
			best, bestReason, bestHasWaiting = m, reason, hasWaiting
		}
	}
	if best != nil {
		return bestReason, best
	}
	// engine-idle rule: no member in flight AND no member dispatchable
	// (A1's livelock guard) while a waiter is blocked
	idle := true
	for _, m := range g.members {
		if m.inflight > 0 || !m.waitingSince.IsZero() {
			idle = false
			break
		}
	}
	if idle {
		for _, m := range g.members {
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

// olderCompletion is the spec order's tie-break: oldest last
// completion first (a member that finished seconds ago is the LAST
// candidate, not the first).
func olderCompletion(a, b *member) bool {
	if a.lastCompletion.IsZero() {
		return false
	}
	if b.lastCompletion.IsZero() {
		return true
	}
	return a.lastCompletion.Before(b.lastCompletion) ||
		(a.lastCompletion.Equal(b.lastCompletion) && a.id < b.id)
}

// budgetVictimLocked is the §6 growth order, over ONE engine's members.
// LEGACY: eligible, then idle (nothing in flight), then the exiting class —
// earliest-seated EVEN MID-FLIGHT (the caller marks it exiting, which drops
// its NE contribution immediately; it leaves when its requests drain, §6.3).
// QUANTUM mode replaces the whole ladder with the BUILD-PLAN §1 order:
// (1) quantum-complete idle; (2) done or silent > S; (3) mid-quantum idle,
// oldest last completion first; (4) the exiting class, mid-flight. Budget
// enforcement only — rotation never evicts an un-eligible member.
func (s *Scheduler) budgetVictimLocked(g *engineGroup, now time.Time) (string, *member) {
	if reason, m := s.rotationVictimLocked(g, now); m != nil {
		return reason, m
	}
	if s.cfg.RotationMode == "quantum" {
		// class (2): done (no queued head) or silent > S
		var best *member
		for _, m := range g.members {
			if g.index.ExitingLocked(m.id) || m.inflight > 0 {
				continue
			}
			if m.waitingSince.IsZero() || s.silentLocked(m, now) {
				if best == nil || olderCompletion(m, best) {
					best = m
				}
			}
		}
		if best != nil {
			return "budget-noqueue", best
		}
		// class (3): mid-quantum idle — the safety valve on GENUINE growth
		// pressure only (admissions were budget-checked; under Q=∞ the
		// waiting excess never reaches here — it waits without
		// displacing residents, the bounded rotation set).
		for _, m := range g.members {
			if g.index.ExitingLocked(m.id) || m.inflight > 0 {
				continue
			}
			if best == nil || olderCompletion(m, best) {
				best = m
			}
		}
		if best != nil {
			return "budget-idle", best
		}
		// class (4): the exiting class, mid-flight
		for _, m := range g.members {
			if g.index.ExitingLocked(m.id) {
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
	var best *member
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) || m.inflight > 0 {
			continue
		}
		if best == nil || m.seatTime.Before(best.seatTime) ||
			(m.seatTime.Equal(best.seatTime) && m.id < best.id) {
			best = m
		}
	}
	if best != nil {
		return "budget-idle", best
	}
	// the exiting class: earliest-seated regardless of inflight
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) {
			continue
		}
		if best == nil || m.seatTime.Before(best.seatTime) ||
			(m.seatTime.Equal(best.seatTime) && m.id < best.id) {
			best = m
		}
	}
	if best != nil {
		return "budget-exiting", best
	}
	return "", nil
}

func (s *Scheduler) exitLocked(g *engineGroup, m *member, reason string, now time.Time) {
	s.releaseColdLocked(g, m.id) // S0-2: unseat releases the charge
	g.pendingTotal -= m.pendingBlocks
	m.pendingBlocks = 0
	g.index.UnseatLocked(m.id)
	delete(g.members, m.id)
	s.countExitLocked(g, reason)
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d", reason, m.resTurns), now)
	if s.onExit != nil {
		s.onExit(engineLabel(g.id), reason, m.resTurns)
	}
	// A7: a session with a queued head re-enters at its ORIGINAL
	// waiting-since (FIFO survives the exit) — on the SAME engine (the
	// binding survives rotation: its cache is there).
	if !m.waitingSince.IsZero() {
		g.waiters[m.id] = &waiter{id: m.id, session: m.session, since: m.waitingSince,
			sizeBytes:  m.lastSizeBytes,
			coldTokens: s.estimateCostLocked(g, m.id, m.lastSizeBytes) * int64(s.blockSize)}
	}
}

// countExitLocked books an exit (or an exiting-mark) on the engine AND in
// the global tally (Snapshot.Exits stays the whole-policy view).
func (s *Scheduler) countExitLocked(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

func (s *Scheduler) seatLocked(g *engineGroup, w *waiter, now time.Time) bool {
	// seat in the index first (history-based cost; the estimate-based
	// budget check already ran in rotate — D13 reconciles the difference).
	// Returns false when TrySeat rejects (the caller must refresh and
	// return — C4).
	if ok, _ := g.index.TrySeatLocked(w.id, g.budgetBlocks); !ok {
		return false
	}
	sess := sessionUnitOf(w.session, w.id)
	s.markEverSeatedLocked(w.id, sess)
	if s.cfg.perEngine() {
		s.bindings[w.id] = g.id // the homing, recorded at FIRST SEAT
	} else if _, ok := s.bindings[w.id]; !ok {
		s.bindings[w.id] = g.id
	}
	delete(g.waiters, w.id)
	m := &member{
		id:            w.id,
		engine:        g.id,
		session:       sess,
		seatTime:      now,
		lastActivity:  now,
		lastSizeBytes: g.index.LastSizeBytesLocked(w.id),
		// the head that triggered this seating is still queued (the
		// serve happens after rotate): inherit its presence so the
		// engine-idle rule cannot evict a member that is about to
		// dispatch (A1's livelock class, entering via the fresh record's
		// zero waitingSince — found by TestNewcomerFirst)
		waitingSince: w.since,
	}
	// B6: ALL entries carry a cold charge (tokens, R1)
	m.coldEntry = w.coldTokens
	g.coldInFlight += m.coldEntry
	// the budget counts the admission estimate (blocks) until the exact
	// keys land at PreRequest
	m.pendingBlocks = w.coldTokens / int64(s.blockSize)
	m.awaitingEngine = true
	g.pendingTotal += m.pendingBlocks
	// C7: record ALL seat waits (newcomers AND re-entries — the last
	// campaign's re-entry waits were invisible); the newcomer counter
	// and the onEntry hook keep their legacy semantics (golden trace).
	kind := "re-entry"
	if w.newcomer {
		kind = "newcomer"
		wait := now.Sub(w.since).Seconds()
		s.counters.entryWaits = append(s.counters.entryWaits, wait)
		if s.onEntry != nil {
			s.onEntry(engineLabel(g.id), wait)
		}
	} else {
		s.counters.reentryWaits = append(s.counters.reentryWaits, now.Sub(w.since).Seconds())
	}
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f", kind, now.Sub(w.since).Seconds()), now)
	g.members[w.id] = m
	return true
}

// ---- budget enforcement (§6, growth) ----

func (s *Scheduler) enforceBudgetLocked(g *engineGroup, now time.Time) {
	if g.budgetBlocks <= 0 {
		return // pre-capacity (A5) for THIS engine: no budget yet
	}
	for int64(g.index.UniqueBlocksNELocked())+g.pendingTotal > int64(g.budgetBlocks) {
		reason, victim := s.budgetVictimLocked(g, now)
		if victim == nil {
			return
		}
		if victim.inflight > 0 {
			g.index.MarkExitingLocked(victim.id)
			s.countExitLocked(g, "exiting")
			continue
		}
		s.exitLocked(g, victim, reason, now)
	}
}

// ---- lifecycle (§8): the ~1s ticker ----

func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deriveLocked() // derive-don't-accumulate, every decision point
	s.reapInflightLocked(now)
	for _, g := range s.activeGroupsLocked() {
		// the derived window's chain expiry (windowMode "derived"):
		// chains with no dispatch for > S leave the session's set.
		if s.cfg.WindowMode == "derived" {
			g.index.ExpireChainsLocked(time.Duration(s.cfg.IdleSeconds*float64(time.Second)), now)
		}
		s.enforceBudgetLocked(g, now)
		for id, m := range g.members {
			if now.Sub(m.lastActivity) > time.Duration(s.cfg.TTLSeconds*float64(time.Second)) &&
				m.inflight == 0 && m.waitingSince.IsZero() {
				s.exitLocked(g, m, "ttl", now)
				g.index.ForgetLocked(id)
				s.forgetEverSeatedLocked(id) // B7: prune consistently with the waiter path
			}
		}
	}
	// R2: NO TTL on waiters — a waiter exists because its head was
	// present at the last sync; rotation waits are minutes. Headless,
	// unseated records are GC'd by lastSeen (silence, not queue age).
	s.gcSilentRecordsLocked(now)
	s.rotateLocked(now)
}

// gcSilentRecordsLocked forgets headless, unseated records whose last
// activity is older than the TTL (R2: silence, not queue age — a waiter
// with a live head is NEVER forgotten, however long it waits). A forgotten
// record loses its HOMING too: rehoming a silent member is exactly the
// defrag the design allows (there is nothing warm to lose).
func (s *Scheduler) gcSilentRecordsLocked(now time.Time) {
	ttl := time.Duration(s.cfg.TTLSeconds * float64(time.Second))
	for id, seen := range s.lastSeen {
		if s.seatedAnywhereLocked(id) {
			continue
		}
		if s.waiterAnywhereLocked(id) {
			continue // has a live head (syncWaiters keeps waiters honest)
		}
		if now.Sub(seen) > ttl {
			for _, g := range s.groups {
				g.index.ForgetLocked(id)
			}
			s.forgetEverSeatedLocked(id)
			delete(s.lastSeen, id)
			delete(s.bindings, id) // the silent member may be re-homed on return
		}
	}
}

func (s *Scheduler) seatedAnywhereLocked(id string) bool {
	for _, g := range s.groups {
		if _, ok := g.members[id]; ok {
			return true
		}
	}
	return false
}

func (s *Scheduler) waiterAnywhereLocked(id string) bool {
	for _, g := range s.groups {
		if _, ok := g.waiters[id]; ok {
			return true
		}
	}
	return false
}

func (s *Scheduler) isNewcomerLocked(id string) bool {
	_, seen := s.everSeated[id]
	return !seen
}

// markEverSeatedLocked records that a record has been seated and mirrors the
// mark on its SESSION (the ordering layer's is-new test). The refcount makes
// the mirror exact: with session := member the ref is 1:1 with the record
// flag, so everSeatedSession[s] exists iff everSeated[s] does — at every
// instant, independent of GC timing.
func (s *Scheduler) markEverSeatedLocked(id, sess string) {
	s.everSeated[id] = struct{}{}
	if _, had := s.recordSession[id]; had {
		return // a re-seat of a still-marked record already holds its session ref
	}
	s.recordSession[id] = sess
	s.sessionRefs[sess]++
	s.everSeatedSession[sess] = struct{}{}
}

// forgetEverSeatedLocked drops a record's "was served" mark and the session
// mirror with it: a session is forgotten exactly when the LAST record that
// carried it is forgotten (the same TTL/GC semantics as the record itself).
func (s *Scheduler) forgetEverSeatedLocked(id string) {
	delete(s.everSeated, id)
	sess, ok := s.recordSession[id]
	if !ok {
		return
	}
	delete(s.recordSession, id)
	s.sessionRefs[sess]--
	if s.sessionRefs[sess] <= 0 {
		delete(s.sessionRefs, sess)
		delete(s.everSeatedSession, sess)
	}
}

// observeMemberSourceLocked counts a dispatch by its member-key source and
// fires the once-per-source hook (the run's effective-member-source echo).
func (s *Scheduler) observeMemberSourceLocked(source string) {
	if source == "" {
		source = "none"
	}
	first := s.counters.memberSources[source] == 0
	s.counters.memberSources[source]++
	if first && s.onMemberSource != nil {
		s.onMemberSource(source)
	}
}

// ---- engine lifecycle (the DISCOVERED engine set) ----

// RemoveEngine tears one engine's group down when the EPP reports the
// endpoint gone (EventDelete — node rotation exercises this every wave).
// Its members are exited (reason "engine-gone"), their HOMING is dropped so
// the next head sighting re-homes them onto a live engine, and the group's
// budget/index/seats disappear with it. Leaving charges on a dead engine is
// how a scale-down silently poisons every later budget decision.
func (s *Scheduler) RemoveEngine(engine string, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.cfg.perEngine() {
		return 0
	}
	g, ok := s.groups[engine]
	if !ok {
		return 0
	}
	// deterministic order: the teardown writes one EXIT event per member and
	// a torn-down group's event order must be reproducible in the artifacts
	ids := make([]string, 0, len(g.members))
	for id := range g.members {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	n := 0
	for _, id := range ids {
		if m, ok := g.members[id]; ok {
			s.exitLocked(g, m, "engine-gone", now)
			n++
		}
	}
	for id := range g.waiters {
		delete(g.waiters, id)
	}
	for member, id := range s.bindings {
		if id == engine {
			delete(s.bindings, member)
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

// ---- the seat/exit event log (BUILD-PLAN §1 instrumentation: spans
// cannot see seats — every future run needs this) ----

// Event is one seat/exit/exiting decision, per id, with reason and
// timestamp. Kept as a bounded ring (dumped in DumpState) and emitted
// to the onEvent hook (the plugin logs one line per event).
//
// Engine is the engine label the event happened on (PREREG-multipod-offload
// instrumentation: without it the per-engine equivalence test is
// unscoreable). It is carried as a FIELD and appended to the log line's
// detail, so the pre-multi-pod reducer regex keeps matching unchanged.
type Event struct {
	T      time.Time `json:"t"`
	Kind   string    `json:"kind"`   // SEAT | EXIT | EXITING | REBIND | AFFINITY-VIOLATION
	ID     string    `json:"id"`     // the session id
	Engine string    `json:"engine"` // the engine label ("" single-mode group -> "single")
	Detail string    `json:"detail"` // reason=... turns=... / kind=... wait=...
	// Member is the fairness unit the event's id resolves to at emit time
	// (the affinity-violation join needs member→engine-at-seat reconstructable
	// from the artifacts alone).
	Member string `json:"member,omitempty"`
}

const eventRingSize = 2048

func (s *Scheduler) recordEvent(kind, id, detail string, now time.Time) {
	ev := Event{T: now, Kind: kind, ID: id, Detail: detail}
	if g := s.groupOfLocked(id); g != nil {
		ev.Engine = engineLabel(g.id)
		ev.Member = id
	}
	s.recordEventOn(ev)
}

// recordEventOn appends an already-attributed event (the caller knows the
// engine: a torn-down group's events are emitted after its members are
// gone).
func (s *Scheduler) recordEventOn(ev Event) {
	s.events = append(s.events, ev)
	if len(s.events) > eventRingSize {
		s.events = s.events[len(s.events)-eventRingSize:]
	}
	if s.onEvent != nil {
		s.onEvent(ev)
	}
}

// EntryTurn reports whether this dispatch is the member's ENTRY (the
// residency's first dispatch) and marks it — the priority header's
// input (R6). Unseated/unknown sessions are entries by definition.
func (s *Scheduler) EntryTurn(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOfLocked(id)
	if g == nil {
		return true
	}
	m, ok := g.members[id]
	if !ok {
		return true
	}
	if m.entryDispatched {
		return false
	}
	m.entryDispatched = true
	return true
}

// Events returns the recent event ring (newest last).
func (s *Scheduler) Events() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Event(nil), s.events...)
}

type Snapshot struct {
	Seats          int
	UniqueBlocks   int
	NEBlocks       int
	BudgetBlocks   int
	Waiters        int
	Exits          map[string]int64
	MeanResTurns   float64
	EntryWaitP50   float64
	ReentryWaitP50 float64
	RefillShare    float64
	Served         int64
	FCFSServed     int64
	// Config echoes the effective knob values (auditable runs).
	Config Config
	// RecentEvents is the tail of the seat/exit ring.
	RecentEvents []Event
	// MemberSources counts dispatches by the member-key source that resolved
	// them ("branch" | "session" | "none") — the effective-member echo.
	MemberSources map[string]int64
	// Engines is the PER-ENGINE section (PREREG-multipod-offload item 8:
	// the schema moves with the state split — the pre-split top-level
	// fields above remain the whole-policy sums, so a single-engine dump
	// is unchanged and a multi-engine dump is scoreable).
	Engines map[string]EngineSnapshot `json:"engines,omitempty"`
	// EngineGroups echoes the mode ("single" | "per-engine").
	EngineGroups string `json:"engineGroups"`
	// AffinityViolations / Rebinds are the CORRECTNESS counters (a request
	// landing off its member's engine).
	AffinityViolations int64 `json:"affinityViolations"`
	Rebinds            int64 `json:"rebinds"`
	Rehomes            int64 `json:"rehomes"`
	EngineGone         int64 `json:"enginesGone"`
	// Bindings is the member→engine homing count per engine (identity-level:
	// it includes bound members that are currently unseated).
	Bindings map[string]int `json:"bindings,omitempty"`
	// HeterogeneousEngines lists engines whose num_gpu_blocks deviates >1%
	// from the first measured engine (the wiring alarm).
	HeterogeneousEngines []string `json:"heterogeneousEngines,omitempty"`
}

// EngineSnapshot is ONE engine's group state: the unit the equivalence test
// is scored on (per-engine seats/waiters/blocks/budget must reproduce the
// banked single-pod run).
type EngineSnapshot struct {
	ID            string           `json:"id"`
	Seats         int              `json:"seats"`
	Waiters       int              `json:"waiters"`
	Sessions      int              `json:"sessions"`
	UniqueBlocks  int              `json:"uniqueBlocks"`
	NEBlocks      int              `json:"neBlocks"`
	BudgetBlocks  int              `json:"budgetBlocks"`
	NumBlocks     int              `json:"numBlocks"`
	CapacityOK    bool             `json:"capacityOk"`
	PendingBlocks int64            `json:"pendingBlocks"`
	ColdInFlight  int64            `json:"coldInFlightTokens"`
	BoundMembers  int              `json:"boundMembers"`
	Served        int64            `json:"served"`
	FCFSServed    int64            `json:"fcfsServed"`
	GateSkipped   int64            `json:"gateSkipped"`
	ForcedSeats   int64            `json:"forcedSeats"`
	Homings       int64            `json:"homings"`
	Exits         map[string]int64 `json:"exits,omitempty"`
	BlockSizeErr  string           `json:"blockSizeMismatch,omitempty"`
}

// SnapshotLite is Snapshot without the event ring and config echo —
// the metrics Refresh path (S5: Refresh runs every second and must not
// copy the 2048-event ring).
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
	snap := Snapshot{Exits: map[string]int64{}, EngineGroups: s.cfg.EngineGroups,
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed,
		AffinityViolations: s.counters.affinityViolations, Rebinds: s.counters.rebinds,
		Rehomes: s.counters.rehomes, EngineGone: s.counters.engineGone,
		Engines: map[string]EngineSnapshot{}, Bindings: map[string]int{}}
	if full {
		snap.Config = s.cfg
		snap.RecentEvents = append([]Event(nil), s.events...)
	}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	snap.MemberSources = map[string]int64{}
	for k, v := range s.counters.memberSources {
		snap.MemberSources[k] = v
	}
	for member, engine := range s.bindings {
		_ = member
		snap.Bindings[engineLabel(engine)]++
	}
	var seats, waiters, uniq, ne int
	for _, id := range s.groupOrder {
		if !s.visibleEngineLocked(id) {
			continue
		}
		g := s.groups[id]
		label := engineLabel(id)
		es := EngineSnapshot{ID: label, Seats: len(g.members), Waiters: len(g.waiters),
			UniqueBlocks: g.index.UniqueBlocks(), NEBlocks: g.index.UniqueBlocksNE(),
			BudgetBlocks: g.budgetBlocks, NumBlocks: g.numBlocks,
			CapacityOK: g.numBlocks > 0, PendingBlocks: g.pendingTotal,
			ColdInFlight: g.coldInFlight, Served: g.counters.served,
			FCFSServed: g.counters.fcfsServed, GateSkipped: g.counters.gateSkipped,
			ForcedSeats: g.counters.forcedSeats, Homings: g.counters.homings,
			BoundMembers: snap.Bindings[label]}
		if g.blockSizeMismatch != nil {
			es.BlockSizeErr = g.blockSizeMismatch.Error()
		}
		if len(g.counters.exits) > 0 {
			es.Exits = map[string]int64{}
			for k, v := range g.counters.exits {
				es.Exits[k] = v
			}
		}
		sessions := map[string]struct{}{}
		for _, m := range g.members {
			sessions[sessionUnitOf(m.session, m.id)] = struct{}{}
		}
		es.Sessions = len(sessions)
		snap.Engines[label] = es
		seats += es.Seats
		waiters += es.Waiters
		uniq += es.UniqueBlocks
		ne += es.NEBlocks
	}
	snap.Seats, snap.Waiters, snap.UniqueBlocks, snap.NEBlocks = seats, waiters, uniq, ne
	// BudgetBlocks is the whole-policy sum; with one engine (single mode, and
	// every pre-split dump) it is exactly that engine's budget.
	for _, id := range s.groupOrder {
		if !s.visibleEngineLocked(id) {
			continue
		}
		snap.BudgetBlocks += s.groups[id].budgetBlocks
	}
	if n := len(s.counters.resTurns); n > 0 {
		sum := 0
		for _, t := range s.counters.resTurns {
			sum += t
		}
		snap.MeanResTurns = float64(sum) / float64(n)
	}
	if n := len(s.counters.entryWaits); n > 0 {
		w := append([]float64(nil), s.counters.entryWaits...)
		sort.Float64s(w)
		snap.EntryWaitP50 = w[n/2]
	}
	if n := len(s.counters.reentryWaits); n > 0 {
		w := append([]float64(nil), s.counters.reentryWaits...)
		sort.Float64s(w)
		snap.ReentryWaitP50 = w[n/2]
	}
	snap.RefillShare = s.counters.refillShare
	if s.cfg.perEngine() {
		for id, n := range s.heterogeneousEnginesLocked(0.01) {
			_ = n
			snap.HeterogeneousEngines = append(snap.HeterogeneousEngines, engineLabel(id))
		}
		sort.Strings(snap.HeterogeneousEngines)
	}
	return snap
}

// Sessions returns the tracked-session count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, g := range s.groups {
		n += g.index.TrackedCount()
	}
	return n
}

// seatedIDs is the union of every engine's seated member ids — the
// identity-level membership set (a member is seated on exactly one engine,
// but WHICH engine is the split's business, not membership's). Used by the
// dumps, the golden trace and the equivalence tests.
func (s *Scheduler) seatedIDs() map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]bool{}
	for _, g := range s.groups {
		for id := range g.members {
			out[id] = true
		}
	}
	return out
}

// Seated reports membership (tests).
func (s *Scheduler) Seated(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.seatedAnywhereLocked(id)
}
