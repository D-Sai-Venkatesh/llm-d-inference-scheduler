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

// ---- GSCH-CORE parameters ----
//
// ONE code path for one engine and for N engines: a single-pod stack is the
// N=1 case of the per-engine state, never a branch of its own. Everything
// that was an ARM (an option that changed the policy's behaviour class) was
// removed; what is left is the six things the policy actually does —
// identity, sizing, admission, rotation, memory pressure, safety — plus the
// wiring facts they cannot work without.
//
// Removed with the arm machinery (each still lives on the archive branch
// dsai/groupsched-tier-aware and may return only as a small addition with
// evidence): entry orders (HRRN / hrrn-bounded), every F-arm
// (rotateWhenFree, dyn-size quantum, quantum ceilings, MLFQ-2), PAT, the
// residue model and its reconcile, the victim-order variants, the ledger
// rotation model (rho/nMax), member-stable binding, the pooled
// engineGroups=single mode, the session layer (sessionOrder / LSF / session
// caps) and the per-engine saturation gate with its 0.80/5 constants.
type Config struct {
	// GroupBudgetFraction sizes each engine's SEAT BUDGET from that engine's own
	// measured KV capacity: budgetBlocks = fraction × num_gpu_blocks. It is ONE
	// number with TWO uses, and that is deliberate:
	//
	//	(1) SIZING — the budget a seat's footprint is admitted against;
	//	(2) SAFETY  — the memory-only net's threshold (safety.go): the net fires
	//	    when the engine's MEASURED kv utilisation reaches the fraction its
	//	    budget already assumes. No second constant exists.
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`

	// Quantum is Q — the fixed tenure, in COMPLETED busy periods, after which a
	// resident becomes rotation-eligible. ONE value, no mode: Q4 (the default)
	// and Q8 are the same code with a different number.
	Quantum int `json:"quantum"`

	// IdleSeconds is S: a resident with no queued turn and no activity for S is
	// rotation-eligible (it is not going to use its seat).
	IdleSeconds float64 `json:"idleSeconds"`
	// TTLSeconds is how long a headless, unseated record is remembered (the
	// silence GC — a waiter with a live head is never forgotten).
	TTLSeconds float64 `json:"ttlSeconds"`
	// (The cold-entry stagger cap is NOT a knob: it is DERIVED per engine as the
	// seat budget's own headroom — floor((1 - groupBudgetFraction) x
	// num_gpu_blocks) blocks x blockSizeTokens — so the valve can only ever spend
	// the margin the budget reserved for it. See engineGroup.coldCapTokens.)
	// InFlightCap bounds one member's concurrent requests.
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

	// ---- the CPU offload tier (PREREG-multipod-offload §"THE OFFLOAD SOLUTION") ----
	//
	// These two numbers are DEPLOYMENT FACTS about the engine, not tuning knobs: they
	// describe the CPU tier the decode pods are running with, exactly as
	// GroupBudgetFraction describes the HBM budget. Together they size the TIER SHADOW
	// (tiershadow.go): capacityBlocks = CpuTierBytes ÷ KvBlockBytes.
	//
	// CpuTierBytes is vLLM's `cpu_bytes_to_use` (exp-ops: DECODE_TIER_BYTES, 512 GiB on
	// the campaign stacks). 0 = this deployment has NO CPU tier: the shadow is absent,
	// nothing is modelled, and `entryOrder: "hrrn-tier"` degenerates to F5′ exactly.
	CpuTierBytes int64 `json:"cpuTierBytes"`
	// KvBlockBytes is ONE KV block's size in bytes on this deployment
	// (2 x layers x kv_heads x head_dim x dtype_bytes x blockSizeTokens — a model+TP
	// fact). Required > 0 whenever CpuTierBytes > 0: bytes ÷ bytes is the only unit
	// conversion in this file that has no measurement behind it, so it is given, never
	// guessed.
	KvBlockBytes int64 `json:"kvBlockBytes"`
	// InflightReapSeconds bounds the age of a CONFIRMED in-flight request id
	// before reconciliation drops it (requests whose termination is never
	// observed — the harness request timeout is 3600 s, so 3700 reaps only true
	// leaks). Provisional ids (chosen at Pick, never reaching PreRequest) are
	// reaped after 30 s.
	InflightReapSeconds float64 `json:"inflightReapSeconds"`

	// ---- ONE selector per module (a future idea = one new implementation + one
	// value here; nothing else in the config moves). Validated by buildModules. ----
	//
	//	Sizing       "index"            footprint from the engine's own BlockIndex
	//	EntryOrder   "newcomer-first"   newcomers, then oldest, then id |
	//	             "hrrn-bounded"     bounded HRRN: wait ÷ min(cold tokens, the median
	//	                                waiter's cold tokens), newcomers as the outer
	//	                                class, then the ratio, then oldest, then id |
	//	             "hrrn-tier"        F5′ with the CPU-tier shadow in the denominator:
	//	                                wait ÷ min(tier-aware cold tokens, the median of
	//	                                them) — restorable tokens count as CACHED, so a
	//	                                returner with a live CPU copy is re-seated before
	//	                                it falls off the tier's cliff
	//	EngineChoice "most-free-budget" most free budget where the head fits, tie → id
	//	Rotation     "quantum"          fixed Q busy periods (plus S of silence)
	//	Victim       "valve"            f0p's memory-pressure order
	//	Safety       "none"            no net (the seat budget IS the control) |
	//	             "budget-fraction" memory-only net at the budget's own fraction
	Sizing       string `json:"sizing"`
	EntryOrder   string `json:"entryOrder"`
	EngineChoice string `json:"engineChoice"`
	Rotation     string `json:"rotation"`
	Victim       string `json:"victim"`
	Safety       string `json:"safety"`
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
		TTLSeconds:              30,
		InFlightCap:             64,
		BlockSizeTokens:         64,
		CapacityLivenessSeconds: 120,
		InflightReapSeconds:     3700,
		Sizing:                  "index",
		EntryOrder:              "newcomer-first",
		EngineChoice:            "most-free-budget",
		Rotation:                "quantum",
		Victim:                  "valve",
		Safety:                  "none",
	}
}

// effective fills the zero values with their defaults so the startup echo and
// Snapshot.Config show EFFECTIVE values (S2 — validate()'s value receiver made
// its fills dead code).
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
	if c.TTLSeconds == 0 {
		c.TTLSeconds = def.TTLSeconds
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

// tierBlocks is the TIER SHADOW's capacity in blocks: the configured CPU tier ÷ one
// KV block. 0 means "no tier on this deployment" — the shadow is then absent and the
// tier-aware cost is the GPU-only cost (hrrn-tier ≡ F5′).
func (c Config) tierBlocks() int {
	return tierBlocksFor(c.CpuTierBytes, c.KvBlockBytes)
}

func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.Quantum < 1 {
		return fmt.Errorf("quantum must be >= 1 (Q is the fixed tenure in completed busy periods; the pause arm is gone), got %d", c.Quantum)
	}
	if c.IdleSeconds <= 0 || c.TTLSeconds <= 0 {
		return fmt.Errorf("idleSeconds and ttlSeconds must be > 0")
	}
	if c.InFlightCap <= 0 {
		return fmt.Errorf("inFlightCap must be > 0")
	}
	if c.BlockSizeTokens <= 0 {
		return fmt.Errorf("blockSizeTokens is required (the token processor's block size; got %d)", c.BlockSizeTokens)
	}
	if c.InflightReapSeconds != 0 && c.InflightReapSeconds < 60 {
		return fmt.Errorf("inflightReapSeconds must be >= 60 (the harness request timeout is 3600s; got %v)", c.InflightReapSeconds)
	}
	if c.CapacityLivenessSeconds < 0 {
		return fmt.Errorf("capacityLivenessSeconds must be >= 0 (0 = default; got %v)", c.CapacityLivenessSeconds)
	}
	if c.CpuTierBytes < 0 {
		return fmt.Errorf("cpuTierBytes must be >= 0 (0 = no CPU tier; got %d)", c.CpuTierBytes)
	}
	if c.KvBlockBytes < 0 {
		return fmt.Errorf("kvBlockBytes must be >= 0 (0 = no CPU tier; got %d)", c.KvBlockBytes)
	}
	if c.CpuTierBytes > 0 && c.KvBlockBytes <= 0 {
		return fmt.Errorf("kvBlockBytes is required when cpuTierBytes > 0: the tier shadow's capacity is cpuTierBytes ÷ kvBlockBytes, and a bytes-per-block constant would be a magic number (got cpuTierBytes=%d, kvBlockBytes=%d)", c.CpuTierBytes, c.KvBlockBytes)
	}
	if c.CpuTierBytes > 0 && c.tierBlocks() == 0 {
		return fmt.Errorf("cpuTierBytes (%d) is smaller than one KV block (%d bytes): a tier that can hold nothing predicts nothing", c.CpuTierBytes, c.KvBlockBytes)
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

// Head is one waiting queue head as seen at Pick: the MEMBER id (the branch
// identity — the fairness unit AND the queue key), the request id, the enqueue
// time, and the wire size (the conservative-cost input).
type Head struct {
	// Member is the group policy's unit: the derived branch id, or "" for
	// anonymous traffic (never tracked, never seated, R9).
	Member    string
	ReqID     string
	Arrival   time.Time
	SizeBytes int64
	// Source names the attribute the member id came from ("branch" |
	// "session" | ""). Counted per dispatch (the run is auditable from the
	// artifacts without re-deriving it from the config).
	Source string
	// FairnessID is the request's flow-control identity — the id the GATEWAY and
	// the harness record on the wire (x_correlation_id / the client's thread id).
	// The member id is a DERIVED branch id and joins to nothing in the request
	// records (measured 0/768 matches), so every SEAT/EXIT event must carry this
	// too or a run's seat membership cannot be joined to its own requests.
	FairnessID string
}

// ---- per-member scheduling state ----

// member is one seated residency: a record that exists exactly as long as the
// member holds a seat on ONE engine (the seat IS the binding — there is no
// pre-seat homing and no post-exit placement claim).
type member struct {
	id string
	// engine is the engineGroup id this member is SEATED on — the same value as
	// the identity-level binding, kept on the record so a dump shows the seat
	// without a second lookup.
	engine string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// resTurns counts settled turns in this residency (the residency histogram's
	// unit).
	resTurns int
	// inflight is DERIVED each decision point from inflightIDs
	// (derive-don't-accumulate): a leaked increment cannot persist past a
	// reconciliation.
	inflight int
	// inflightIDs tracks the member's in-flight request ids with their
	// confirmation state (C3): provisional (chosen at Pick, not yet at
	// PreRequest) ids are reaped after 30s; confirmed ids after
	// InflightReapSeconds. m.inflight = len(inflightIDs).
	inflightIDs map[string]inflightRec
	// busyPeriods counts completed MAIN-CHAIN busy periods (V1: the main chain
	// draining to zero — under root identity a busy root with overlapping
	// sub-agent threads never fully drains, which would make it de-facto Q=∞).
	// It is the rotation quantum's unit.
	busyPeriods int
	// mainInflight is the in-flight count on the MAIN chain (the busy-period
	// unit's input).
	mainInflight int
	// lastCompletion is the last time the main chain returned to zero.
	lastCompletion time.Time
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this member's outstanding cold-entry charge in TOKENS against
	// the stagger cap; released EXACTLY once (S0-2) at the first of: prefill
	// complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// lastSizeBytes is the last known wire size (the estimate delta's baseline;
	// updated at PreRequest with the exact size).
	lastSizeBytes int64
	// pendingBlocks is the seat-time admission ESTIMATE still counted against the
	// engine's budget; it converts to exact blocks at PreRequest (Dispatched) and
	// releases at exit. Without it a newcomer seats at index-cost 0 and the budget
	// understates until the exact keys land.
	pendingBlocks int64
	// awaitingEngine marks a pendingBlocks charge made at SEAT time on the engine
	// the seat chose. PreRequest's pickedEndpoint is AUTHORITATIVE and MOVES the
	// charge when it disagrees (reconcileEngineLocked) — the estimate error is
	// bounded to admissions in flight between Pick and PreRequest.
	awaitingEngine bool
	// waitingSince is the member's oldest queued head (B5: kept oldest).
	waitingSince time.Time
	// fairnessID is the last fairness id (queue key) this member was seen with —
	// the join key for its SEAT/EXIT events (see Head.FairnessID).
	fairnessID string
	// predResident / predSet carry the SHADOW-VALIDATION prediction (piece 3) from the
	// moment this member was SEATED as a RETURN to the settlement of the turn that
	// seat was taken for: the fraction of that turn's estimated footprint the engine's
	// GPU index + its CPU-tier shadow said was already resident. It is paired there
	// with the MEASURED cached/prompt the engine reports, then cleared — one
	// prediction, one measurement.
	predResident float64
	predSet      bool
}

// inflightRec is one in-flight request's reconciliation record (C3).
type inflightRec struct {
	since     time.Time // Pick time (provisional) / PreRequest time (confirmed)
	confirmed bool
	main      bool // landed on the member's main chain (V1)
	// keys are the exact tier-A chain blocks recorded at PreRequest. They are kept
	// because SETTLE is where a request's blocks enter the CPU tier (a model of vLLM
	// storing them) and settle otherwise carries no hash list — re-deriving the set
	// from the index at settle would insert blocks of turns that have NOT run yet.
	keys []Hash
}

// estEntry is one cached admission estimate (M2): blocks for a given head size;
// recomputed when the index version or the size changes.
type estEntry struct {
	sizeBytes int64
	blocks    int64
}

// waiter is one head in the SHARED waiter queue. A waiter is not seated, so it
// belongs to no engine: its admission cost is a per-engine VECTOR (what it would
// cost on each engine, from that engine's own index) and the engine is chosen at
// SEAT time (admission.go).
type waiter struct {
	id       string
	since    time.Time // the OLDEST arrival (B5)
	newcomer bool
	// sizeBytes is the last seen head's wire size (the C4 refresh input).
	sizeBytes int64
	// coldTokens is the head's cold-prefill estimate in TOKENS (R1: one unit
	// everywhere — blocks × blockSize): the BEST CASE over the live engines, and
	// the value the cold-entry stagger is charged with once an engine is chosen.
	coldTokens int64
	// coldTokensByEngine is the per-engine admission cost, FROZEN at sync time for
	// this decision pass (reviewer R2). Each entry comes from that engine's own
	// index, so an engine that does not hold the member's blocks is costed as cold
	// — which is the truth about that engine — instead of inheriting the best case.
	coldTokensByEngine map[string]int64
	// tierCostTokens is coldTokens' TIER-AWARE twin: the same best-case scalar with
	// each engine's CPU-tier shadow counted as residence, so a block that is still
	// restorable from some engine's tier is NOT part of it. It is the denominator
	// `entryOrder: "hrrn-tier"` prices a waiter with, and it is deliberately a
	// SEPARATE field: the admission cost above must stay GPU-only (offload adds no
	// seats, so admitting against a tier-aware cost would overbook the engine).
	// tierCostSet records that it was computed for this pass; when no engine has a
	// tier the scheduler writes it EQUAL to coldTokens, which is what makes
	// hrrn-tier degenerate to F5′ rather than to a special case.
	tierCostTokens int64
	tierCostSet    bool
	// fairnessID rides with the head (see Head.FairnessID) so a SEAT event can name
	// the wire-level identity even before the member has a record.
	fairnessID string
}

// ---- the per-engine group ----
//
// EVERYTHING that is a fact about ONE engine's HBM lives here: its block index
// (dedup is per engine — a block on A is worthless to B), its budget, its
// measured capacity and liveness, its seats, its cold-entry charge, its estimate
// cache and its counters. One engine id and two engines are the same struct, the
// same functions and the same decisions; there is no pooled group and no
// single-engine special case.
//
// What is NOT here (identity, because a member's identity is a property of the
// client, not of the seat): everSeated (the newcomer test), lastSeen (the silence
// GC), the member→engine bindings, and the SHARED waiter queue.
type engineGroup struct {
	// id is the endpoint id the EPP discovered.
	id string

	index        *BlockIndex // attached at creation (block size from config)
	budgetBlocks int         // 0 until THIS engine's num_gpu_blocks arrives
	numBlocks    int
	// blockSizeMismatch is the loud per-engine wiring failure (a measured
	// CacheBlockSize that differs from the config's BlockSizeTokens).
	blockSizeMismatch error

	members map[string]*member
	// coldInFlight is this engine's outstanding cold-entry charge: the stagger
	// cap is PER ENGINE (N engines ⇒ N concurrent cold prefills; R11's "one cold
	// entry is always allowed" proves liveness inside one engine's cap, and it
	// holds independently for each engine).
	coldInFlight int64
	// pendingTotal is the sum of seated members' pendingBlocks (the budget's
	// estimate component).
	pendingTotal int64

	// safetyFiredOn is the net's LATCHED verdict as of the last settle, and
	// safetySince the start of the current fire interval (safety.go).
	safetyFiredOn bool
	safetySince   time.Time

	// estCache/estVersion cache waiter admission estimates keyed on the index's
	// mutation version (M2: Picks run at ~1kHz; unchanged state must not recompute
	// ~150 × ~1k-block history scans). Per engine because the versions are.
	estCache   map[string]estEntry
	estVersion uint64

	// tier is THIS engine's CPU offload tier, as a model (tiershadow.go): nil when the
	// deployment configures no tier. It is a per-engine fact like the index — a block
	// in engine A's CPU tier is worthless to B — and it is consulted ONLY by the two
	// readers that need it: the tier-aware cold estimate (hrrn-tier's denominator) and
	// the shadow-validation metric. Dispatch, the budget and the cold valve never see
	// it: offload adds no seats.
	tier *tierShadow
	// estTierCache caches the TIER-AWARE estimate keyed on BOTH mutation versions —
	// the index's (a block arriving or leaving the GPU set changes the answer) and the
	// tier's (a store or an eviction does). One cache would let a tier mutation read a
	// stale cost for the rest of the index's version epoch.
	estTierCache    map[string]estEntry
	estTierIndexVer uint64
	estTierTierVer  uint64

	counters struct {
		served     int64 // dispatches served on this engine
		fcfsServed int64 // pre-capacity FCFS dispatches counted here
		// safetySkips counts dispatches the SAFETY NET held back (safety.go).
		safetySkips int64
		// safetyFires / safetyFiredSeconds are the net's events and fired time.
		safetyFires        int64
		safetyFiredSeconds float64
		// entryWaits / reentryWaits are THIS engine's seat-wait samples (the
		// per-engine wait ladder the dump reports, B7).
		entryWaits   []float64
		reentryWaits []float64
		// forcedSeats counts seats taken OVER budget by an affinity
		// reconciliation (the observed endpoint wins; enforcement catches up on
		// the next pass). Must be ~0 in a healthy run.
		forcedSeats int64
		exits       map[string]int64
	}
}

// coldCapTokens is THIS engine's cold-ENTRY stagger cap, in tokens: the seat
// budget's own headroom, floor((1 - fraction) x num_gpu_blocks) blocks x blockSize.
// The valve staggers cold prefills INSIDE the margin the budget reserved for them,
// which is why it needs no value of its own — with the budget at 0.95 of 40,183
// blocks and 64-token blocks the cap is 2009 blocks = 128,576 tokens (the constant it
// replaces was 131,072 in a unit the budget never used).
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

func newEngineGroup(id string, blockSizeTokens, tierBlocks int) (*engineGroup, error) {
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return nil, err
	}
	g := &engineGroup{id: id, index: idx,
		members: map[string]*member{}, estCache: map[string]estEntry{},
		tier: newTierShadow(tierBlocks)}
	g.counters.exits = map[string]int64{}
	return g, nil
}

// engineLabel is the metrics/event label value for an engine id (the legacy
// single-engine alias's "" id renders as "single" so an unlabelled-series scrape
// never shows an empty label value).
func engineLabel(id string) string {
	if id == "" {
		return "single"
	}
	return id
}

// Scheduler is the group scheduler state machine over an engine set. The ONE
// piece of state that is deliberately global is the SHARED waiter queue: a waiter
// is not homed, so it belongs to no engine until an engine seats it. Everything
// else that is per engine lives in engineGroup; everything that is per MEMBER
// (identity, GC clock, the seat binding) lives here.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// mod is the wired module set (modules.go) — the ONE place a policy decision
	// comes from.
	mod modules
	// observers are the optional metric hooks (nil in tests).
	onExit         func(engine, reason string, resTurns int)
	onEntry        func(engine string, waitSeconds float64)
	onReentry      func(engine string, waitSeconds float64)
	onServe        func(engine string)
	onEvent        func(Event)
	onRehome       func(member, from, to string)
	onEngineGone   func(engine string, members int)
	onMemberSource func(source string)
	// onBind / onUnbind are the PLACEMENT-PIN hooks: they fire when a member's
	// seat (and therefore its engine binding) is created or dropped, so the
	// plugin's lock-light Filter table can follow the seats without ever touching
	// s.mu. nil in tests.
	onBind   func(member, engine string)
	onUnbind func(member string)
	// onColdValveHeld fires when the cold-entry valve HOLDS a head that would fit an
	// engine's budget. The head WAITS (the valve opens when that engine's cold prefill
	// produces its first token); nothing is evicted for it. This is the sizing signal:
	// how often the stagger makes demand wait.
	onColdValveHeld func(engine string)
	// onSafetyFire fires when an engine's safety net OPENS (a fire event).
	onSafetyFire func(engine string)
	// onShadowValidation receives one (predicted resident fraction, measured
	// cached/prompt) pair per seated return — the CPU-tier shadow's accuracy, exported
	// for the analysis-time AUC. nil in tests.
	onShadowValidation func(engine string, predicted, measured float64)

	// groups holds one engineGroup per DISCOVERED engine, keyed by endpoint id.
	groups map[string]*engineGroup
	// groupOrder is the sorted key list (deterministic iteration: every
	// cross-engine decision is taken in id order, never map order).
	groupOrder []string

	blockSize int

	// everSeated is the NEWCOMER test: a member that has never been seated queues
	// ahead of the members that have (D1's newcomer-first order).
	everSeated map[string]struct{}
	// lastSeen is the per-record activity time for the silence GC (R2): updated at
	// Dispatched and Settle; a headless, unseated record older than TTL is
	// forgotten.
	lastSeen map[string]time.Time
	// bindings is the member→engine association. It IS the seat: created when the
	// member is seated, dropped the moment it exits. An unseated member has no
	// placement opinion at all.
	bindings map[string]string

	// waiters is THE SHARED WAITER QUEUE: one entry per member with a queued head
	// and no seat. Whichever engine frees a seat (or has room) takes the head from
	// here — the queue is where waiting is pooled, which is what makes N engines
	// one admission system instead of N.
	waiters map[string]*waiter

	// load is the PER-ENGINE utilisation view the plugin pushes up from the
	// endpoint metrics it already polls (safety.go).
	load map[string]engineLoad

	// pickSeq synthesizes request ids for heads that lack one (M1).
	pickSeq int64
	// queueViewAt is when the queue-head snapshot was last reconciled
	// (syncWaitersLocked) — the evidence clock behind "this member has no queued
	// head". A stale view must never justify an eviction (the C8 race).
	queueViewAt time.Time
	// headsObservedAt is when the head list was last OBSERVED — a Pick, which carries a
	// candidate set. A Tick RE-APPLIES the last observation (so the waiting flags stay
	// honest) and must NOT refresh this clock: this is the age of the EVIDENCE itself,
	// i.e. the size of the window in which a head can be queued with nobody looking.
	headsObservedAt time.Time
	// now is the virtual clock of the CURRENT decision pass (set at the top of
	// every public decision entry point). Nothing here calls time.Now: a golden
	// replay has to be reproducible.
	now time.Time
	// homeSeq rotates the tie-break among equally-empty engines at SEAT time.
	// Round-robin rather than random: same anti-burst property (a burst of
	// newcomers splits instead of racing to one engine) and it keeps EVERY
	// multi-engine run reproducible.
	homeSeq int64
	events  []Event
	// lastHeads is the most recent queue-head snapshot (Pick's view), kept so the
	// ticker's admission pass can re-sync the same evidence instead of guessing.
	lastHeads []Head

	counters struct {
		entryWaits   []float64
		reentryWaits []float64
		// R1/R2 diagnostics: both must read 0 with the gateway (every request
		// carries a real id end to end).
		emptyReqIDs  int64
		straySettles int64
		// affinityViolations counts requests that LANDED on an engine other than
		// the one they were seated on (CORRECTNESS: the observed endpoint wins, so
		// each violation is repaired by moving the seat; a nonzero steady rate
		// means the placement path is not honouring the pin).
		affinityViolations int64
		rebinds            int64
		engineGone         int64
		exits              map[string]int64
		resTurns           []int
		refillShare        float64
		uncachedTok        int64
		totalTok           int64
		served             int64
		fcfsServed         int64
		// queueViewUntrusted counts budget-enforcement passes that ran with an
		// untrusted queue view (the C8 guard skipped the head-presence classes).
		queueViewUntrusted int64
		// eventsDropped counts events the SEAT/EXIT ring wrapped past. The ring is
		// bounded (memory), so on a long window the oldest events fall out of the
		// DumpState tail; the stdout stream is the durable copy. A non-zero value here
		// means "the dump's RecentEvents does not cover the window", which must be
		// visible instead of inferred.
		eventsDropped int64
		// memberSources counts dispatches by the attribute the member id came from
		// (the run's effective-member-source echo).
		memberSources map[string]int64
		// shadowPairs counts the SHADOW-VALIDATION pairs collected (one per seated
		// return whose first turn settled with a measured cache hit): the denominator
		// of any AUC the analysis computes, reported so an arm with no pairs cannot be
		// read as "the shadow was right".
		shadowPairs int64
	}
}

// NewScheduler builds the state machine. The block index attaches per engine as
// the engine is discovered (A5: block size from the token processor's config,
// known at startup).
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
		groups:     map[string]*engineGroup{},
		everSeated: map[string]struct{}{},
		lastSeen:   map[string]time.Time{},
		bindings:   map[string]string{},
		waiters:    map[string]*waiter{},
		load:       map[string]engineLoad{}}
	s.counters.exits = map[string]int64{}
	s.counters.memberSources = map[string]int64{}
	return s, nil
}

// groupForLocked returns (creating) the group for an engine id.
func (s *Scheduler) groupForLocked(id string) (*engineGroup, error) {
	if g, ok := s.groups[id]; ok {
		return g, nil
	}
	g, err := newEngineGroup(id, s.blockSize, s.cfg.tierBlocks())
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

// eng is the single-engine TEST accessor: the group the legacy test fixtures
// drive through the no-engine forms of the API (SetNumBlocks / Dispatched /
// Settle, which resolve to the engine id ""). It is an accessor, never a code
// path: no decision reads it.
//
// It does NOT lock — callers hold s.mu (or own the scheduler in a test), so it
// can be used from inside an already-locked decision path without re-entering the
// mutex.
func (s *Scheduler) eng() *engineGroup { return s.groups[""] }

// existingGroupLocked returns the group for an engine id WITHOUT creating it
// (a settle for an already-torn-down engine must not resurrect the group).
func (s *Scheduler) existingGroupLocked(id string) *engineGroup { return s.groups[id] }

// groupOfLocked is the member's SEAT engine (nil when the member is unseated).
func (s *Scheduler) groupOfLocked(member string) *engineGroup {
	id, ok := s.bindings[member]
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

// visibleEngineLocked reports whether a group id is a real DISCOVERED engine: the
// legacy single-engine alias's "" id is one (it renders as "single'), a torn-down
// engine has no group at all.
func (s *Scheduler) visibleEngineLocked(id string) bool {
	_, ok := s.groups[id]
	return ok
}

// SetNumBlocks attaches the measured capacity (num_gpu_blocks from the endpoint
// metrics). A measured block size that differs from the config value fails loudly
// (recorded; the engine is not usable until it matches).
//
// It is SetNumBlocksFor("") — the single-engine form the legacy tests use.
func (s *Scheduler) SetNumBlocks(blockSizeTokens, numBlocks int) error {
	return s.SetNumBlocksFor("", blockSizeTokens, numBlocks)
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

// EngineGroupIDs returns the discovered engine ids (sorted) — the audit of the
// group set the policy is actually driving.
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

// GroupLabel resolves an endpoint key to the label the metrics and events carry
// for it (an unlabelled-series scrape never sees an empty label value).
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

// BlockSizeMismatch reports the configured-vs-measured block-size mismatch (the
// loud failure surface). With N engines every mismatch is loud: this returns the
// first in id order for the startup error, and the per-engine dump carries each
// engine's own error.
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
//
// The TRAILING PARTIAL BLOCK IS COUNTED. The engine allocates a block for the last
// tokens of a prompt whether or not they fill it, so a count of full blocks is short
// by up to one block per request — an undercount in exactly the direction that costs
// preemption headroom, and the one part of "token-exact block counting" that needs no
// tokenizer (the tokens are already here).
func (s *Scheduler) HashTokens(tokens []uint32, cacheSalt string, blockSizeTokens int) []Hash {
	if blockSizeTokens <= 0 {
		blockSizeTokens = s.blockSize
	}
	if blockSizeTokens <= 0 || len(tokens) == 0 {
		return nil
	}
	n := (len(tokens) + blockSizeTokens - 1) / blockSizeTokens
	out := make([]Hash, 0, n)
	var parent uint64
	salt := sha256.Sum256([]byte(cacheSalt))
	saltSeed := binary.LittleEndian.Uint64(salt[:8])
	for i := 0; i < n; i++ {
		h := sha256.New()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], saltSeed^parent)
		h.Write(b[:])
		end := (i + 1) * blockSizeTokens
		if end > len(tokens) {
			end = len(tokens)
		}
		for j := i * blockSizeTokens; j < end; j++ {
			binary.LittleEndian.PutUint32(b[:4], tokens[j])
			h.Write(b[:4])
		}
		parent = binary.LittleEndian.Uint64(h.Sum(nil)[:8])
		out = append(out, Hash(parent))
	}
	return out
}

// ---- Pick (§5): the waiting set is the queue-head snapshot ----

// PickFromHeads runs one dispatch decision against the current queue heads and
// returns the index into heads of the chosen request, or -1.
//
// The engine is chosen INSIDE (see PickFromHeadsEngine); the plugin needs only the
// head, because the engine attribution arrives authoritatively at PreRequest
// (pickedEndpoint) — the pendingEngine idiom.
func (s *Scheduler) PickFromHeads(heads []Head, now time.Time) int {
	_, i := s.PickFromHeadsEngine(heads, now)
	return i
}

// PickFromHeadsEngine is PickFromHeads with the engine it served from ("" for an
// anonymous R9 head). One pass: derive, reap, sync the shared waiter queue,
// enforce each engine's budget, seat waiters, then choose one dispatchable head.
//
// The pass is written for N engines as THE code and for one engine as its N=1
// case: nothing below is conditioned on how many engines exist.
//
// Within an engine the member heads go OLDEST FIRST (a resident whose turn is due
// dispatches before its own later turns). ACROSS engines the order is the engine
// whose own residents are waiting most per seat — a dispatch-order tie-break, not
// a placement key — then the head's arrival, then the engine id, so the decision is
// total and reproducible.
func (s *Scheduler) PickFromHeadsEngine(heads []Head, now time.Time) (string, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	s.headsObservedAt = now // a Pick is the only FRESH observation of the queue
	s.reapInflightLocked(now)
	s.syncWaitersLocked(heads, now)
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
		if h.Member == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		g := s.groupOfLocked(h.Member)
		if g == nil {
			continue // unseated: its head is ADMISSION, handled by rotateLocked
		}
		m, seated := g.members[h.Member]
		if !seated || g.index.ExitingLocked(h.Member) {
			continue
		}
		if m.inflight >= s.cfg.InFlightCap {
			continue
		}
		if s.safetyFiredLocked(g) {
			// THE SAFETY NET holds this engine's own DISPATCHES (not just its seats):
			// the net exists to leave preemption headroom for RUNNING work, and a
			// running request's KV keeps growing after it was admitted. First-fit, like
			// the seat loop: the head stays queued, other engines carry on, and a fire
			// that clears restores the engine to the ordinary path with no state to undo.
			g.counters.safetySkips++
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
		chosen := &cands[0]
		score := s.queuedPerSeatLocked(g)
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
		// M1: a dispatch is NEVER counted without an id — synthesize one when the
		// head lacks ReqID (the counter follows the map).
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
	// R9: no member head is dispatchable — serve the oldest ANONYMOUS head (FCFS
	// pass-through, the siblings' behaviour; anonymous traffic is never tracked,
	// never seated, never starved).
	for i, h := range heads {
		if h.Member == "" {
			return "", i
		}
	}
	return "", -1
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

// FCFSServed counts a pre-capacity dispatch (the gauge substrate) and returns the
// label it is booked to. B2: it is booked to NO engine — a pre-capacity stack has
// no engine to attribute a dispatch to, and booking it to one (the old "single"
// label, which at N≥2 was a series that owned no pod) made a counter lie. The
// plugin labels it "nocapacity".
func (s *Scheduler) FCFSServed() (int64, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
	return s.counters.fcfsServed, "nocapacity"
}

// syncWaitersLocked reconciles the SHARED waiter queue with the queue-head
// snapshot: a member with a queued head and no seat joins it, a member whose head
// vanished leaves it, and every seated member's waiting-since tracks its oldest
// queued head (the victim preference's input).
func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
	s.queueViewAt = now
	s.lastHeads = heads
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Member == "" {
			continue
		}
		present[h.Member] = h.Arrival
		if g := s.groupOfLocked(h.Member); g != nil {
			if m, seated := g.members[h.Member]; seated {
				if h.FairnessID != "" && m.fairnessID != h.FairnessID {
					m.fairnessID = h.FairnessID
				}
				if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
					m.waitingSince = h.Arrival
				}
				m.lastActivity = now
				continue
			}
		}
		// unseated: its demand is ADMISSION, so it waits in the one shared queue
		w, ok := s.waiters[h.Member]
		if !ok {
			w = &waiter{id: h.Member, newcomer: s.isNewcomerLocked(h.Member)}
			s.waiters[h.Member] = w
		}
		w.fairnessID = h.FairnessID
		// B5: keep the OLDEST arrival across fan-out re-arrivals
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
		w.sizeBytes = h.SizeBytes
		s.setWaiterCostsLocked(w) // per-engine admission costs, frozen for this pass
	}
	// vanished: members clear their waiting marker; waiters leave
	for _, g := range s.activeGroupsLocked() {
		for id, m := range g.members {
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

// reconcileEngineLocked is the pendingEngine idiom's second half: the OBSERVED
// endpoint wins. A seated member whose request lands on another engine is MOVED
// to it (one member's correction instead of unbounded accounting drift — the index
// is an optimistic model of where blocks ARE, and a stale binding drifts
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
		// the engine is over its budget for this member: seat it anyway — it
		// demonstrably holds the blocks — and let enforcement evict (an over-budget
		// counter is recoverable; a member absent from the engine that is serving it
		// is not).
		to.index.SeatForceLocked(id)
		to.counters.forcedSeats++
	}
	s.recordEvent("REBIND", id, fmt.Sprintf("from=%s to=%s", engineLabel(from.id), engineLabel(to.id)), now)
	if s.onRehome != nil {
		s.onRehome(id, from.id, to.id)
	}
	if s.onBind != nil {
		s.onBind(id, to.id) // the placement pin follows the seat
	}
	return to
}

// estimateCostCachedLocked is estimateCostLocked behind the M2 cache: valid while
// the engine's index version is unchanged and the head size matches.
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

// estimateCostLocked is the SIZE module's admission cost for one head on one
// engine (blocks). It is a thin seam: the scheduler owns the cache, the module owns
// the formula.
func (s *Scheduler) estimateCostLocked(g *engineGroup, id string, sizeBytes int64) int64 {
	return s.mod.sizer.Cost(g.index, id, sizeBytes)
}

// estimateTierCostCachedLocked is estimateCostCachedLocked over the TIER-AWARE cost:
// the head's blocks that are resident in NEITHER this engine's GPU index NOR its CPU
// tier, in tokens. The cache is keyed on BOTH mutation versions (index and tier): a
// store or an eviction changes the answer just as surely as a block arriving does, and
// one version alone would let a tier change read stale for the rest of the epoch.
func (s *Scheduler) estimateTierCostCachedLocked(g *engineGroup, id string, sizeBytes int64) int64 {
	iv, tv := g.index.VersionLocked(), g.tier.Version()
	if g.estTierCache == nil || g.estTierIndexVer != iv || g.estTierTierVer != tv {
		g.estTierCache = map[string]estEntry{}
		g.estTierIndexVer, g.estTierTierVer = iv, tv
	}
	if e, ok := g.estTierCache[id]; ok && e.sizeBytes == sizeBytes {
		return s.mod.sizer.Tokens(e.blocks)
	}
	cold, _, _ := s.mod.sizer.TierCost(g.index, g.tier, id, sizeBytes)
	g.estTierCache[id] = estEntry{sizeBytes: sizeBytes, blocks: cold}
	return s.mod.sizer.Tokens(cold)
}

// ---- PreRequest (post-dispatch): the exact tier-A correction ----

// Dispatched records a released request at PreRequest without an observed endpoint
// (the legacy/single-engine form): the member's seat decides the engine.
func (s *Scheduler) Dispatched(id, reqID string, keys []Hash, sizeBytes int64, now time.Time) {
	s.DispatchedOn(id, reqID, "", keys, sizeBytes, now)
}

// DispatchedOn records a released request at PreRequest on the endpoint the
// framework ACTUALLY picked: the exact keys replace the estimate (D13 absorbs the
// error at the next enforcement), in-flight and the cold charge are taken, and the
// size baseline moves.
//
// engine=="" means "the member's SEAT" (no observation).
//
// B1 (the phantom binding): a member with NO seat is UNTRACKED. The old code fell
// back to the pooled "" group and wrote a binding to it, so under per-engine groups
// a member could be bound to an engine that no decision ever visits — permanently
// unseated while the pool size and the dump counted it. There is no pooled group
// here to fall back to: no seat, no charge, no binding.
func (s *Scheduler) DispatchedOn(id, reqID, engine string, keys []Hash, sizeBytes int64, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOfLocked(id)
	switch {
	case g == nil && engine == "":
		return // unseat and no observation: nothing to charge (B1)
	case g == nil:
		gg, err := s.groupForLocked(engine)
		if err != nil || gg == nil {
			return
		}
		if _, seated := gg.members[id]; !seated {
			return // an engine we do not seat this member on: untracked, no binding
		}
		g = gg
	case engine != "" && engine != g.id:
		// pendingEngine reconciliation: the observed endpoint is authoritative
		g = s.reconcileEngineLocked(id, engine, now)
		if g == nil {
			return
		}
	}
	g.index.SetSizeLocked(id, sizeBytes) // the estimate baseline
	onMain := g.index.SettleDispatchedAtLocked(id, reqID, keys, now)
	// THE TIER SHADOW's DISPATCH HOOK: the request's chain blocks move to MRU in this
	// engine's CPU-tier model. Touching only REFRESHES blocks the tier already holds —
	// the new ones enter at settle (a block being prefilled is not stored yet).
	g.tier.Touch(keys)
	if m, ok := g.members[id]; ok {
		m.lastActivity = now
		m.lastSizeBytes = sizeBytes
		m.awaitingEngine = false
		// C3/M1/R1: reconcile the in-flight id. A non-empty reqID confirms-or-adds.
		// An EMPTY reqID must not mint a second id (Pick already recorded a
		// provisional one — two records for one request leak past the reap): confirm
		// the OLDEST PROVISIONAL record, and synthesize only when none exists.
		if reqID == "" {
			s.counters.emptyReqIDs++
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
		m.inflightIDs[reqID] = inflightRec{since: now, confirmed: true, main: onMain, keys: keys}
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
			// refine the charge to the exact token count (R1: one unit), still capped:
			// the refinement arrives AFTER the admission check, so an uncapped
			// refinement is exactly how the charge overran its own cap.
			g.coldInFlight = max(g.coldInFlight-m.coldEntry, 0)
			m.coldEntry = min(s.mod.sizer.Tokens(int64(len(keys))),
				g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
			g.coldInFlight += m.coldEntry
		}
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry charge is
// released HERE (B3: not at end of decode), exactly once.
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

// Settle records a terminated dispatched request without an observed endpoint (the
// legacy/single-engine form).
func (s *Scheduler) Settle(id, reqID string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.SettleOn(id, reqID, "", usage, natural, now)
}

// SettleOn records a terminated dispatched request on the endpoint it ran on: the
// in-flight decrement for EVERY cause, the residency turn count for every settled
// turn (A8), and the busy period on a main-chain drain.
//
// An engine mismatch here means the request landed on an engine other than the
// member's seat AFTER PreRequest reconciled — a CORRECTNESS violation (it corrupts
// the charged engine's record). It is counted, never silently folded: the
// accounting stays with the seat the PreRequest committed, and the counter is the
// run-level alarm.
func (s *Scheduler) SettleOn(id, reqID, engine string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.SettleOnUsage(id, reqID, engine, usage.Prompt, usage.Cached, 0, natural, now)
}

// SettleOnUsage is SettleOn with the completion token count.
func (s *Scheduler) SettleOnUsage(id, reqID, engine string, promptTokens, cachedTokens, completionTokens int, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSeen[id] = now
	g := s.groupOfLocked(id)
	if g == nil && engine != "" {
		g = s.existingGroupLocked(engine) // never resurrect a torn-down engine
	}
	if g == nil {
		return
	}
	if engine != "" && g.id != engine {
		s.counters.affinityViolations++
		s.recordEvent("AFFINITY-VIOLATION", id, fmt.Sprintf("bound=%s settled_on=%s", engineLabel(g.id), engineLabel(engine)), now)
	}
	m, seated := g.members[id]
	if seated {
		// C3/M1: remove the in-flight record; the counter follows the map
		// (derive-don't-accumulate). A missing reqID removes the OLDEST record
		// (defensive legacy path — never blind).
		removed := false
		removedMain := false
		var settledKeys []Hash
		if reqID != "" {
			// R2: a known id removes itself; an UNKNOWN non-empty id removes NOTHING
			// (the oldest-fallback would delete a different live request's record
			// after a late reap or a duplicate settle — the member would undercount
			// and evict mid-flight). Counted as a stray.
			if rec, had := m.inflightIDs[reqID]; had {
				delete(m.inflightIDs, reqID)
				removed, removedMain, settledKeys = true, rec.main, rec.keys
			} else {
				s.counters.straySettles++
			}
		} else if len(m.inflightIDs) > 0 {
			key, oldest := "", time.Time{}
			for rid, rec := range m.inflightIDs {
				if key == "" || rec.since.Before(oldest) {
					key, oldest = rid, rec.since
				}
			}
			if rec, had := m.inflightIDs[key]; had {
				delete(m.inflightIDs, key)
				removed, removedMain, settledKeys = true, rec.main, rec.keys
			}
		}
		// THE TIER SHADOW's SETTLE HOOK: a settled request's blocks are now STORED in
		// this engine's CPU tier (vLLM keeps them; the LRU evicts past capacity). The
		// request's OWN keys, not the member's whole set — a queued turn that has not
		// run yet is not in the tier and inserting it would make the shadow lie warm.
		g.tier.Insert(settledKeys)
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
		// S4/V1/R3a: the busy period closes only when a MAIN-chain request was
		// removed AND the main chain drained to zero — a sub-agent request settling
		// while the main chain is idle must not burn a quantum unit.
		if removed && removedMain && m.mainInflight == 0 {
			m.busyPeriods++
			m.lastCompletion = now
		}
	}
	if !seated {
		return
	}
	m.resTurns++ // A8: every settled turn counts
	// the settled request was the tracked queued head: clear the marker (the next
	// Pick re-establishes it if more heads remain — the TTL path cannot see stale
	// queue presence)
	m.waitingSince = time.Time{}
	// §6.3: an exiting member leaves when its in-flight requests drain
	if g.index.ExitingLocked(id) && m.inflight == 0 {
		s.exitLocked(g, m, "exiting-drain", now)
		return
	}
	if !natural {
		return
	}
	// SHADOW VALIDATION (piece 3): pair the prediction taken when this member was
	// SEATED as a return with the measured cache hit of the turn that seat was for.
	// One prediction, one measurement, then cleared — a second settle cannot inherit
	// the first prediction's claim. The engine's `prompt` and `cached` are the ground
	// truth (cached = the prefix vLLM served from its GPU cache AND its CPU tier), so
	// the pair is (predicted resident fraction, measured cached/prompt) with the
	// fairness id that makes the record joinable to the wire-level request.
	if m.predSet {
		m.predSet = false
		if promptTokens > 0 {
			measured := float64(cachedTokens) / float64(promptTokens)
			s.recordEventOn(Event{T: now, Kind: "SHADOW-VALIDATION", ID: id,
				Engine: engineLabel(g.id), Member: id,
				Detail: fmt.Sprintf("pred=%.4f measured=%.4f prompt=%d cached=%d fairness_id=%s",
					m.predResident, measured, promptTokens, cachedTokens, fairnessOrNone(m.fairnessID))})
			s.counters.shadowPairs++
			if s.onShadowValidation != nil {
				s.onShadowValidation(engineLabel(g.id), m.predResident, measured)
			}
		}
	}
	uncached := int64(promptTokens - cachedTokens)
	if uncached < 0 {
		uncached = 0
	}
	s.counters.resTurns = append(s.counters.resTurns, m.resTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(promptTokens)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// ---- rotation (§7): WHO leaves, and when ----

// silentLocked is the idle test with the C8 fix: silent means no activity AND no
// queued head (a waiting member wants to run).
func (s *Scheduler) silentLocked(m *member, now time.Time) bool {
	if !m.waitingSince.IsZero() {
		return false
	}
	return now.Sub(m.lastActivity) > time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// deriveLocked recomputes the derived aggregates from member records
// (derive-don't-accumulate): a leaked or missed increment anywhere cannot persist
// past the next decision point. Per engine.
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

// reapInflightLocked drops in-flight ids whose termination was never observed
// (C3): provisional ids (chosen at Pick, never reaching PreRequest) after 30s;
// confirmed ids after InflightReapSeconds (the harness request timeout is 3600s —
// anything older is a leak).
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
					// M3(a): the member's requests are all dead — release its cold
					// charge or the valve freezes the queue until the reap bound.
					s.releaseColdLocked(g, m.id)
				}
			}
		}
	}
}

// eligibleLocked asks the TENURE module whether a resident may be released: Q — the
// one rotation value — in COMPLETED BUSY PERIODS, or S of silence.
//
// Q is a number, not a mode. Q4 and Q8 are the same code with Config.Quantum 4 and 8;
// there is no ceiling, no per-member scaling and no pool split.
func (s *Scheduler) eligibleLocked(m *member, now time.Time) (string, bool) {
	return s.mod.tenure.Eligible(m, s.tenureCtxLocked(now))
}

// tenureCtxLocked is the read-only view the TENURE module gets (with the queue view's
// age, so "no queued head" is evidence rather than a guess).
func (s *Scheduler) tenureCtxLocked(now time.Time) TenureCtx {
	return TenureCtx{Now: now, IdleSeconds: s.cfg.IdleSeconds,
		QueueView: s.queueViewAt, QueueViewSet: !s.queueViewAt.IsZero()}
}

// pressureCtxLocked is the read-only view the PRESSURE module gets of ONE engine.
func (s *Scheduler) pressureCtxLocked(g *engineGroup, now time.Time) PressureCtx {
	tctx := s.tenureCtxLocked(now)
	return PressureCtx{
		Now:            now,
		Exiting:        g.index.ExitingLocked,
		RotationVictim: func() (string, *member) { return s.rotationVictimLocked(g, now) },
		Silent:         tctx.Silent,
		Headless:       tctx.Headless,
		QueueViewTrusted: !tctx.QueueView.IsZero() &&
			now.Sub(tctx.QueueView) <= time.Duration(s.cfg.IdleSeconds*float64(time.Second)),
	}
}

// rotationVictimLocked (§7 demand-driven exit) over ONE engine's members, in the
// order f0p ran: among the ELIGIBLE, prefer a member WITH a queued turn (it will
// use the seat again the moment it gets it), then earliest-seated, then id.
//
// The engine-idle rule is the liveness fallback below it: an engine with nothing in
// flight and nothing dispatchable, while a waiter is blocked, must free a seat or
// the queue stalls — and that rule scans ONLY THIS ENGINE, because an engine whose
// members are all in flight must not borrow another engine's idleness to justify an
// eviction.
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
	// engine-idle rule: no member in flight AND no member dispatchable (A1's
	// livelock guard) while a waiter is blocked
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

// olderCompletion breaks a tie between two equally-costly victims: a REAL last
// completion beats "never completed", then oldest completion first, then the id.
//
// The last clause is a CORRECTNESS fix, not a taste: f0p's version returned false
// for a member that had never completed a busy period, so among the (common) set of
// members that have not completed one yet the "first one in map iteration order" won
// — and Go randomises that. The archive build therefore produces DIFFERENT traces on
// the same scenario and config (measured: 3 divergences in 4 replays of the 2,600 s
// golden scenario with f0p's own config). Every victim rule below iterates a slice
// built from the member map, so a total order is what makes a replay reproducible at
// all; testdata/golden-f0p-total.txt is the archive build with EXACTLY this tie-break
// added, and the core reproduces it byte for byte.
func olderCompletion(a, b *member) bool {
	az, bz := a.lastCompletion.IsZero(), b.lastCompletion.IsZero()
	if az != bz {
		return bz // a real completion is the cheaper victim
	}
	if az {
		return a.id < b.id // TOTAL tie-break among members that never completed
	}
	return a.lastCompletion.Before(b.lastCompletion) ||
		(a.lastCompletion.Equal(b.lastCompletion) && a.id < b.id)
}

// budgetVictimLocked is MEMORY PRESSURE (§6 growth): the PRESSURE module picks the
// victim among ONE engine's members, in f0p's order (the budget valve, unchanged).
// Only memory pressure gets past the tenure rule's class (1): rotation itself never
// evicts a member that is not eligible.
func (s *Scheduler) budgetVictimLocked(g *engineGroup, now time.Time) (string, *member) {
	members := make([]*member, 0, len(g.members))
	for _, m := range g.members {
		members = append(members, m)
	}
	return s.mod.pressure.Victim(members, s.pressureCtxLocked(g, now))
}

// exitLocked unseats one member. The BINDING IS DROPPED HERE — the seat is the
// binding, so an exiting member keeps no placement claim: its next seat is chosen
// fresh, by the fit/most-free-budget rule, and may land on a different engine.
func (s *Scheduler) exitLocked(g *engineGroup, m *member, reason string, now time.Time) {
	s.releaseColdLocked(g, m.id) // S0-2: unseat releases the charge
	g.pendingTotal -= m.pendingBlocks
	m.pendingBlocks = 0
	g.index.UnseatLocked(m.id)
	delete(g.members, m.id)
	s.countExitLocked(g, reason)
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d fairness_id=%s", reason, m.resTurns, fairnessOrNone(m.fairnessID)), now)
	if s.onExit != nil {
		s.onExit(engineLabel(g.id), reason, m.resTurns)
	}
	delete(s.bindings, m.id)
	if s.onUnbind != nil {
		s.onUnbind(m.id)
	}
	// A7: a member with a queued head re-enters at its ORIGINAL waiting-since (FIFO
	// survives the exit) — back in the SHARED queue, with no engine attached.
	if !m.waitingSince.IsZero() {
		w := &waiter{id: m.id, since: m.waitingSince, sizeBytes: m.lastSizeBytes}
		s.setWaiterCostsLocked(w)
		s.waiters[m.id] = w
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

// countExitLocked books an exit (or an exiting-mark) on the engine AND in the
// global tally (Snapshot.Exits stays the whole-policy view).
func (s *Scheduler) countExitLocked(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

// seatLocked seats one waiting head on the engine the admission pass chose. It is
// the ONE place a binding is created: seat = pin, and the seat is the whole of it.
func (s *Scheduler) seatLocked(g *engineGroup, w *waiter, now time.Time) bool {
	// THE SHADOW-VALIDATION PREDICTION, taken BEFORE the seat is written into the index:
	// TrySeat puts the member's blocks in NE, after which every block of its own set
	// would read as resident and the prediction would be a tautology (1.0 by
	// construction). Only a RETURNING member gets one — a newcomer has no history to be
	// resident, and the metric is defined over seated returns.
	predResident, predOK := 0.0, false
	if !w.newcomer && g.tier.Enabled() {
		if _, resident, total := s.mod.sizer.TierCost(g.index, g.tier, w.id, w.sizeBytes); total > 0 && resident > 0 {
			predResident, predOK = float64(resident)/float64(total), true
		}
	}
	// seat in the index first (history-based cost; the estimate-based budget check
	// already ran in the admission pass — D13 reconciles the difference). Returns
	// false when TrySeat rejects (the caller must refresh and return — C4).
	if ok, _ := g.index.TrySeatLocked(w.id, g.budgetBlocks); !ok {
		return false
	}
	s.everSeated[w.id] = struct{}{}
	s.bindings[w.id] = g.id
	if s.onBind != nil {
		s.onBind(w.id, g.id) // the placement pin's mirror (filter.go)
	}
	delete(s.waiters, w.id)
	m := &member{
		id:           w.id,
		engine:       g.id,
		seatTime:     now,
		lastActivity: now,
		// lastSizeBytes is the estimate baseline at PREVIOUS dispatch (empty for a
		// newcomer): seatLocked must not pre-set it to this head's size, or the
		// first dispatch's delta would read zero.
		lastSizeBytes: g.index.LastSizeBytesLocked(w.id),
		// the head that triggered this seating is still queued (the serve happens
		// after the seat): inherit its presence so the engine-idle rule cannot evict
		// a member that is about to dispatch (A1's livelock class).
		waitingSince: w.since,
		fairnessID:   w.fairnessID,
		predResident: predResident,
		predSet:      predOK,
	}
	// B6: ALL entries carry a cold charge (tokens, R1), CAPPED at the engine's own
	// stagger cap — the valve's job is to serialise cold prefills inside the budget's
	// headroom, not to account the entry's whole footprint (the budget's pending
	// charge does that). Capping is also what makes the cap an invariant: an entry
	// bigger than the headroom charges the headroom, so an empty valve always admits
	// exactly one entry (R11 falls out instead of being an exception that let the
	// charge run 1.5x over its own cap for a quarter of the run).
	m.coldEntry = min(w.coldTokens, g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
	g.coldInFlight += m.coldEntry
	// the budget counts the admission estimate (blocks) until the exact keys land
	// at PreRequest
	m.pendingBlocks = w.coldTokens / int64(s.blockSize)
	m.awaitingEngine = true
	g.pendingTotal += m.pendingBlocks
	// C7: record ALL seat waits (newcomers AND re-entries — the last campaign's
	// re-entry waits were invisible); the onEntry hook keeps its newcomer semantics.
	kind := "re-entry"
	if w.newcomer {
		kind = "newcomer"
		wait := now.Sub(w.since).Seconds()
		s.counters.entryWaits = append(s.counters.entryWaits, wait)
		g.counters.entryWaits = append(g.counters.entryWaits, wait)
		if s.onEntry != nil {
			s.onEntry(engineLabel(g.id), wait)
		}
	} else {
		wait := now.Sub(w.since).Seconds()
		s.counters.reentryWaits = append(s.counters.reentryWaits, wait)
		g.counters.reentryWaits = append(g.counters.reentryWaits, wait)
		if s.onReentry != nil {
			s.onReentry(engineLabel(g.id), wait)
		}
	}
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f fairness_id=%s", kind, now.Sub(w.since).Seconds(), fairnessOrNone(w.fairnessID)), now)
	g.members[w.id] = m
	return true
}

// queueViewTrustedLocked reports whether the queue-head snapshot behind this decision
// pass is recent enough that "this member has no queued head" means anything at all.
func (s *Scheduler) queueViewTrustedLocked(now time.Time) bool {
	return !s.queueViewAt.IsZero() &&
		now.Sub(s.queueViewAt) <= time.Duration(s.cfg.IdleSeconds*float64(time.Second))
}

// queueEvidenceAgeSecondsLocked is how old the head list the queue view was built from
// is, on the DECISION clock (-1 when no head list has been observed yet — the
// absent-signal convention: a 0 would read as "fresh"). It grows whenever Picks stop
// (a closed pool gate is the normal cause), which is exactly the window in which a head
// can be queued while `waitingSince` says otherwise (the C8 race).
func (s *Scheduler) queueEvidenceAgeSecondsLocked() float64 {
	if s.headsObservedAt.IsZero() {
		return -1
	}
	return s.now.Sub(s.headsObservedAt).Seconds()
}

// noteQueueViewUntrustedLocked books one enforcement pass that ran without trustworthy
// head evidence. The value is CUMULATIVE and exported from the snapshot (one source of
// truth: the metrics layer converts it to a Prometheus counter by delta, so a hook
// nobody wired cannot leave the series at zero).
func (s *Scheduler) noteQueueViewUntrustedLocked() { s.counters.queueViewUntrusted++ }

// enforceBudgetLocked is MEMORY PRESSURE on the growth path: while the engine's
// accounted footprint (NE + pending estimates) is over its budget, take victims in
// the valve's order until it fits or nothing is evictable. NE counts EXITING members
// too — they still hold their blocks — so a pass makes progress only by unseating
// somebody, never by marking them.
func (s *Scheduler) enforceBudgetLocked(g *engineGroup, now time.Time) {
	if g.budgetBlocks <= 0 {
		return // no measured capacity yet for THIS engine: nothing to enforce
	}
	countedStale := false
	for int64(g.index.UniqueBlocksNELocked())+g.pendingTotal > int64(g.budgetBlocks) {
		if !countedStale && !s.queueViewTrustedLocked(now) {
			// ONE per engine-pass: this pass will choose victims with a queue view too old
			// to tell a busy member from an idle one, so the C8 guard drops the
			// head-presence classes. The gauge beside this counter is the clock itself;
			// both are the evidence that the guard is doing something (or that the
			// decision clock has run away from the queue syncs).
			s.noteQueueViewUntrustedLocked()
			countedStale = true
		}
		reason, victim := s.budgetVictimLocked(g, now)
		if victim == nil {
			return
		}
		if victim.inflight > 0 {
			if g.index.ExitingLocked(victim.id) {
				// Already on its way out. Its blocks stay charged until it drains, so
				// marking it again frees nothing: this pass is DONE (continuing would
				// re-pick the same member and spin under the scheduler's mutex).
				return
			}
			g.index.MarkExitingLocked(victim.id)
			s.countExitLocked(g, "exiting")
			// the series has to see it: the member is on its way out NOW (it drains, then
			// exits as "exiting-drain"), and counting it only in the snapshot left
			// llm_d_epp_group_exits_total{reason="exiting"} permanently at zero.
			if s.onExit != nil {
				s.onExit(engineLabel(g.id), "exiting", victim.resTurns)
			}
			continue
		}
		s.exitLocked(g, victim, reason, now)
	}
}

// ---- lifecycle (§8): the ~1s ticker ----

func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
	s.deriveLocked() // derive-don't-accumulate, every decision point
	s.reapInflightLocked(now)
	for _, g := range s.activeGroupsLocked() {
		s.settleNetLocked(g, now)
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
	// R2: NO TTL on waiters — a waiter exists because its head was present at the
	// last sync; rotation waits are minutes. Headless, unseated records are GC'd by
	// lastSeen (silence, not queue age).
	s.gcSilentRecordsLocked(now)
	// BUG 3 (the C8 race): Tick calls the admission loop WITHOUT a queue sync, so a
	// head that arrived since the last Pick left its member looking headless. Re-sync
	// from the LAST head snapshot (a candidate set is what Pick sees; it is refreshed on
	// every decision pass, and Tick's rotate uses it so the two paths agree), then seat.
	s.syncWaitersLocked(s.lastHeads, now)
	s.rotateLocked(now)
}

// settleNetLocked re-evaluates the safety net against the pushed snapshot so the
// fire COUNT and the fired TIME are booked on the decision clock (a net that opens
// and closes between two Picks is still a fire event).
func (s *Scheduler) settleNetLocked(g *engineGroup, now time.Time) {
	s.settleSafetyLocked(g, s.safetyFiredLocked(g), now)
}

// gcSilentRecordsLocked forgets headless, unseated records whose last activity is
// older than the TTL (R2: silence, not queue age — a waiter with a live head is
// NEVER forgotten, however long it waits). A forgotten record loses its
// "ever seated" mark with it, so a member that has been gone long enough arrives as
// a NEWCOMER again (which is what it is: nothing about its cache survived).
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
	_, ok := s.waiters[id]
	return ok
}

func (s *Scheduler) isNewcomerLocked(id string) bool {
	_, seen := s.everSeated[id]
	return !seen
}

// forgetEverSeatedLocked drops a record's "was served" mark.
func (s *Scheduler) forgetEverSeatedLocked(id string) { delete(s.everSeated, id) }

// observeMemberSourceLocked counts a dispatch by its member-key source and fires
// the once-per-source hook (the run's effective-member-source echo).
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
// Its members are exited (reason "engine-gone") — their seats die with it and
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
	// The waiter QUEUE is shared and survives a teardown: those heads can seat on
	// a survivor, which is the whole point of pooling the waits.
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
	Kind   string    `json:"kind"`   // SEAT | EXIT | EXITING | REBIND | AFFINITY-VIOLATION | SHADOW-VALIDATION
	ID     string    `json:"id"`     // the session id
	Engine string    `json:"engine"` // the engine label ("" single-mode group -> "single")
	Detail string    `json:"detail"` // reason=... turns=... / kind=... wait=... / pred=... measured=... fairness_id=...
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
	RefillShare    float64
	Served         int64
	FCFSServed     int64
	// Config echoes the effective knob values (auditable runs).
	Config Config
	// RecentEvents is the tail of the seat/exit ring.
	RecentEvents []Event
	// MemberSources counts dispatches by the member-key source that resolved them
	// ("branch" | "none") — the effective-member echo.
	MemberSources map[string]int64
	// Engines is the PER-ENGINE section. The top-level fields above remain the
	// whole-policy sums, so a single-engine dump is unchanged and a multi-engine
	// dump is scoreable.
	Engines map[string]EngineSnapshot `json:"engines,omitempty"`
	// AffinityViolations / Rebinds are the CORRECTNESS counters (a request landing
	// off its member's seat engine).
	AffinityViolations int64 `json:"affinityViolations"`
	Rebinds            int64 `json:"rebinds"`
	EngineGone         int64 `json:"enginesGone"`
	// Bindings is the number of LIVE seat bindings per engine (a member is bound
	// exactly while it is seated, so this equals that engine's seats; a binding to
	// an unknown engine — the B1 phantom — must be 0).
	Bindings map[string]int `json:"bindings,omitempty"`
	// HeterogeneousEngines lists engines whose num_gpu_blocks deviates >1% from the
	// first measured engine (the wiring alarm).
	HeterogeneousEngines []string `json:"heterogeneousEngines,omitempty"`
	// QueueViewAgeSeconds is how old the head EVIDENCE behind the queue view was on the
	// DECISION clock (a Tick re-applies the last observation, so this keeps measuring the
	// observation's age), and QueueViewUntrusted the number of budget-enforcement passes
	// that ran with the guard's own view too old to trust (−1 = no head list observed
	// yet). Together they are the C8 guard's state: a growing age is the window in which
	// a head can be queued unseen, and an untrusted-pass count means the enforcement
	// passes are re-syncing less often than S.
	QueueViewAgeSeconds float64 `json:"queueViewAgeSeconds"`
	QueueViewUntrusted  int64   `json:"queueViewUntrustedPasses"`
	// EventRingDropped is how many SEAT/EXIT events the bounded ring has wrapped past:
	// the dump's RecentEvents covers the LAST eventRingSize events, and this says how
	// much of the window lies before them.
	EventRingDropped int64 `json:"eventRingDropped"`
	// ShadowPairs counts the CPU-tier shadow's VALIDATION pairs (one per seated return
	// whose turn settled with a measured cache hit). 0 on an arm without a configured
	// tier — and on an arm WITH one it is the denominator of the shadow's AUC.
	ShadowPairs int64 `json:"shadowPairs"`
}

// EngineSnapshot is ONE engine's group state: seats, waiters (accounted), the
// budget, the exit mix, the safety net and the per-engine wait ladder.
type EngineSnapshot struct {
	ID            string `json:"id"`
	Seats         int    `json:"seats"`
	Waiters       int    `json:"waiters"`
	UniqueBlocks  int    `json:"uniqueBlocks"`
	NEBlocks      int    `json:"neBlocks"`
	BudgetBlocks  int    `json:"budgetBlocks"`
	NumBlocks     int    `json:"numBlocks"`
	CapacityOK    bool   `json:"capacityOk"`
	PendingBlocks int64  `json:"pendingBlocks"`
	ColdInFlight  int64  `json:"coldInFlightTokens"`
	// ColdCapTokens is the DERIVED cold-entry stagger cap (the budget's own
	// headroom, in tokens). A scored run divides coldInFlightTokens by THIS, not by
	// a constant from the arm's config: the cap moves with the engine.
	ColdCapTokens int64 `json:"coldCapTokens"`
	BoundMembers  int   `json:"boundMembers"`
	Served        int64 `json:"served"`
	FCFSServed    int64 `json:"fcfsServed"`
	ForcedSeats   int64 `json:"forcedSeats"`
	// Safety net: the live verdict, the fire EVENTS and the fired TIME, plus the
	// scraped inputs it reads (the queue depth is reported, never decisive).
	SafetyFiring       bool    `json:"safetyFiring"`
	SafetyFires        int64   `json:"safetyFires"`
	SafetyFiredSeconds float64 `json:"safetyFiredSeconds"`
	SafetySkips        int64   `json:"safetySkips"`
	SafetyKV           float64 `json:"safetyKv"`
	SafetyWaitingQueue int     `json:"safetyWaitingQueue"`
	// EntryWaitP50 / ReentryWaitP50 are THIS engine's median seat waits (B7: the
	// Prometheus histograms were per engine while the dump was whole-policy only,
	// so the per-engine wait ladder was unrecoverable from a lights-out dump).
	EntryWaitP50   float64          `json:"entryWaitP50"`
	ReentryWaitP50 float64          `json:"reentryWaitP50"`
	Exits          map[string]int64 `json:"exits,omitempty"`
	BlockSizeErr   string           `json:"blockSizeMismatch,omitempty"`
	// TierShadowBlocks / TierCapacityBlocks are the CPU-tier shadow's fill and its
	// configured capacity, in blocks (both 0 when no tier is configured). A shadow
	// pinned at capacity is the model's steady state; a shadow far below it means the
	// store traffic the tier was sized for never arrived.
	TierShadowBlocks   int `json:"tierShadowBlocks"`
	TierCapacityBlocks int `json:"tierCapacityBlocks"`
}

// SnapshotLite is Snapshot without the event ring and config echo — the metrics
// Refresh path (S5: Refresh runs every second and must not copy the 2048-event
// ring).
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
		EngineGone:          s.counters.engineGone,
		QueueViewAgeSeconds: s.queueEvidenceAgeSecondsLocked(),
		QueueViewUntrusted:  s.counters.queueViewUntrusted,
		EventRingDropped:    s.counters.eventsDropped,
		ShadowPairs:         s.counters.shadowPairs,
		Engines:             map[string]EngineSnapshot{}, Bindings: map[string]int{}}
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
	for _, engine := range s.bindings {
		snap.Bindings[engineLabel(engine)]++
	}
	var seats, uniq, ne int
	for _, id := range s.groupOrder {
		g := s.groups[id]
		label := engineLabel(id)
		st := s.safetyStateLocked(g, s.now)
		es := EngineSnapshot{ID: label, Seats: len(g.members),
			UniqueBlocks: g.index.UniqueBlocks(), NEBlocks: g.index.UniqueBlocksNE(),
			BudgetBlocks: g.budgetBlocks, NumBlocks: g.numBlocks,
			CapacityOK: g.numBlocks > 0, PendingBlocks: g.pendingTotal,
			ColdInFlight: g.coldInFlight, Served: g.counters.served,
			ColdCapTokens: g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize),
			FCFSServed:    g.counters.fcfsServed,
			ForcedSeats:   g.counters.forcedSeats,
			BoundMembers:  snap.Bindings[label],
			SafetyFiring:  st.Fired, SafetyFires: st.Fires,
			SafetyFiredSeconds: st.FiredSeconds, SafetySkips: g.counters.safetySkips,
			SafetyKV: st.KV, SafetyWaitingQueue: st.Waiting,
			EntryWaitP50:       p50(g.counters.entryWaits),
			ReentryWaitP50:     p50(g.counters.reentryWaits),
			TierShadowBlocks:   g.tier.Len(),
			TierCapacityBlocks: g.tier.Capacity()}
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
	if n := len(s.counters.resTurns); n > 0 {
		sum := 0
		for _, t := range s.counters.resTurns {
			sum += t
		}
		snap.MeanResTurns = float64(sum) / float64(n)
	}
	snap.EntryWaitP50 = p50(s.counters.entryWaits)
	snap.ReentryWaitP50 = p50(s.counters.reentryWaits)
	snap.RefillShare = s.counters.refillShare
	for id := range s.heterogeneousLocked(0.01) {
		snap.HeterogeneousEngines = append(snap.HeterogeneousEngines, engineLabel(id))
	}
	sort.Strings(snap.HeterogeneousEngines)
	return snap
}

// p50 is the median of a wait sample (0 when empty).
func p50(w []float64) float64 {
	if len(w) == 0 {
		return 0
	}
	cp := append([]float64(nil), w...)
	sort.Float64s(cp)
	return cp[len(cp)/2]
}

// Sessions returns the tracked-member-record count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, g := range s.groups {
		n += g.index.TrackedCount()
	}
	return n
}

// seatedIDs is the union of every engine's seated member ids. Used by the dumps,
// the golden trace and the equivalence tests.
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
