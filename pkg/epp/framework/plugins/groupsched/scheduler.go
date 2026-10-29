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
}

func defaultConfig() Config {
	return Config{
		GroupBudgetFraction: 0.95,
		Rho:                 4,
		NMax:                0, // -> 2ρ
		IdleSeconds:         20,
		TTLSeconds:          30,
		ColdEntryCapTokens:  131072,
		InFlightCap:         64,
		BlockSizeTokens:     64,
		RotationMode:        "ledger",
		QuantumBusyPeriods:  0, // -> 4
		ValveMode:           "on",
		VictimOrder:         "legacy",
		WindowMode:          "last4",
		GateMode:            "detector",
		PriorityHeader:      "off",
		InflightReapSeconds: 3700,
	}
}

func (c Config) nMax() int {
	if c.NMax > 0 {
		return c.NMax
	}
	return 2 * c.Rho
}

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
		return fmt.Errorf("rho must be >= 1, got %v", c.Rho)
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

// Head is one waiting queue head as seen at Pick: the session (agent
// identity), the request id, the enqueue time, and the wire size (the
// conservative-cost input).
type Head struct {
	Session   string
	ReqID     string
	Arrival   time.Time
	SizeBytes int64
}

// ---- per-session scheduling state (§3) ----

type member struct {
	id string
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
	// against the budget; converts to exact blocks at PreRequest
	// (Dispatched) and releases at exit. Without it a newcomer seats at
	// index-cost 0 and the budget understates until the exact keys land.
	pendingBlocks int64
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
	id       string
	since    time.Time // the OLDEST arrival (B5)
	newcomer bool
	// sizeBytes is the last seen head's wire size (the C4 refresh input).
	sizeBytes int64
	// coldTokens is the entry's cold-prefill estimate in TOKENS
	// (R1: one unit everywhere — blocks x blockSize).
	coldTokens int64
}

// Scheduler is the group scheduler state machine over a BlockIndex.
// The 80% dispatch gate stays in the stack's saturation detector.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// observers are the optional metric hooks (nil in tests).
	onExit  func(reason string, resTurns int)
	onEntry func(waitSeconds float64)
	onServe func()
	onEvent func(Event)

	index             *BlockIndex // attached at startup (block size from config)
	budgetBlocks      int         // 0 until num_gpu_blocks arrives (metrics)
	numBlocks         int
	blockSize         int
	blockSizeMismatch error

	members    map[string]*member
	waiters    map[string]*waiter
	events     []Event
	everSeated map[string]struct{}
	// lastSeen is the per-record activity time for the silence GC (R2):
	// updated at Dispatched and Settle; a headless, unseated record
	// older than TTL is forgotten.
	lastSeen map[string]time.Time

	coldInFlight int64
	// pendingTotal is the sum of seated members' pendingBlocks (the
	// budget's estimate component).
	pendingTotal int64
	// pickSeq synthesizes request ids for heads that lack one (M1).
	pickSeq int64
	// estCache/estVersion cache waiter admission estimates keyed on the
	// index's mutation version (M2: Picks run at ~1kHz; unchanged state
	// must not recompute ~150 × ~1k-block history scans).
	estCache   map[string]estEntry
	estVersion uint64

	counters struct {
		exits        map[string]int64
		entryWaits   []float64
		reentryWaits []float64
		// R1/R2 diagnostics: both must read 0 with the gateway (every
		// request carries a real id end to end).
		emptyReqIDs  int64
		straySettles int64
		// gateSkipped counts heads the size-aware gate turned away
		// (first-fit skips; the v2-freeze diagnostic).
		gateSkipped int64
		resTurns    []int
		refillShare float64
		uncachedTok int64
		totalTok    int64
		served      int64
		fcfsServed  int64
	}
}

// NewScheduler builds the state machine. The index attaches NOW (A5:
// block size from the token processor's config, known at startup).
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg = cfg.effective()
	idx, err := NewBlockIndex(cfg.BlockSizeTokens)
	if err != nil {
		return nil, err
	}
	idx.SetWindowMode(cfg.WindowMode)
	s := &Scheduler{cfg: cfg, index: idx, blockSize: cfg.BlockSizeTokens,
		members: map[string]*member{}, waiters: map[string]*waiter{},
		everSeated: map[string]struct{}{}, lastSeen: map[string]time.Time{},
		estCache: map[string]estEntry{}}
	s.counters.exits = map[string]int64{}
	return s, nil
}

// SetNumBlocks attaches the measured capacity (num_gpu_blocks from the
// endpoint metrics). A measured block size that differs from the config
// value fails loudly (recorded; HasCapacity stays until it matches).
func (s *Scheduler) SetNumBlocks(blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if blockSizeTokens > 0 && blockSizeTokens != s.blockSize {
		s.blockSizeMismatch = fmt.Errorf(
			"groupsched: measured CacheBlockSize %d != configured BlockSizeTokens %d (the token processor's value); refusing the capacity",
			blockSizeTokens, s.blockSize)
		return s.blockSizeMismatch
	}
	if s.numBlocks > 0 || numBlocks <= 0 {
		return nil // first capacity wins (single-pool v0)
	}
	s.numBlocks = numBlocks
	s.budgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	return nil
}

// HasCapacity reports whether the measured num_gpu_blocks has arrived.
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.numBlocks > 0
}

// BlockSizeMismatch reports the loud-failure state, if any.
func (s *Scheduler) BlockSizeMismatch() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.blockSizeMismatch
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
// heads: sync the waiter view, enforce the budget, rotate, then serve
// the oldest queued head of any non-exiting member under the in-flight
// cap. Returns the index into heads of the chosen request, or -1.
func (s *Scheduler) PickFromHeads(heads []Head, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return -1
	}
	s.deriveLocked() // derive-don't-accumulate: recomputed, never trusted across paths
	s.reapInflightLocked(now)
	s.syncWaitersLocked(heads, now)
	s.enforceBudgetLocked(now)
	s.rotateLocked(now)

	// member heads, OLDEST-FIRST. The size-aware dispatch gate (v2) is
	// FIRST-FIT: dispatch the first head that fits, SKIP the ones that
	// don't (they stay queued). The first deployed v2 checked only the
	// single oldest head and returned -1 when it didn't fit — with the
	// budget near-full the oldest head NEVER fit, nothing else could
	// dispatch, no busy period ever completed (no quantum-complete
	// victims), and the C8 fix kept the waiting members from being
	// idle-evicted: a self-sustaining head-of-line freeze (the 02:13Z
	// v2 incident; dump banked as v2-freeze-goroutines.txt).
	type cand struct {
		i int
		h Head
	}
	var cands []cand
	for i, h := range heads {
		if h.Session == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		m, seated := s.members[h.Session]
		if !seated || s.index.ExitingLocked(h.Session) {
			continue
		}
		if m.inflight >= s.cfg.InFlightCap {
			continue
		}
		cands = append(cands, cand{i, h})
	}
	sort.SliceStable(cands, func(x, y int) bool { return cands[x].h.Arrival.Before(cands[y].h.Arrival) })
	best := -1
	var bestHead Head
	for _, c := range cands {
		if s.cfg.GateMode == "size-aware" && !s.gateAllowsLocked(c.h) {
			s.counters.gateSkipped++ // diagnostic: heads the gate turned away
			continue
		}
		best, bestHead = c.i, c.h
		break
	}
	if best >= 0 {
		m := s.members[heads[best].Session]
		// M1: a dispatch is NEVER counted without an id — synthesize one
		// when the head lacks ReqID (the counter follows the map).
		rid := bestHead.ReqID
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
		s.counters.served++
		if s.onServe != nil {
			s.onServe()
		}
		return best
	}
	// R9: no member head is dispatchable — serve the oldest ANONYMOUS
	// head (FCFS pass-through, the siblings' behaviour; anonymous
	// traffic is never tracked, never seated, never starved).
	for i, h := range heads {
		if h.Session == "" {
			if s.cfg.GateMode == "size-aware" && !s.gateAllowsLocked(h) {
				continue
			}
			return i
		}
	}
	return -1
}

// gateAllowsLocked is the size-aware dispatch gate: current usage (NE
// + pending estimates) plus this request's new blocks must fit the
// budget. Computed from the EPP's OWN accounting — never the scraped
// kv_cache_usage (stale by a scrape interval; kept as a cross-check
// gauge only). A member's own admission estimate (pendingBlocks) is
// NOT double-counted: its new blocks are est minus what its pending
// already reserves.
func (s *Scheduler) gateAllowsLocked(h Head) bool {
	if s.budgetBlocks <= 0 {
		return true
	}
	est := s.estimateCostCachedLocked(h.Session, h.SizeBytes)
	if m, ok := s.members[h.Session]; ok && m.pendingBlocks > 0 {
		est = max(est-m.pendingBlocks, 0)
	}
	// M4: ALL seated blocks (exiting members' blocks stay pinned until
	// they drain) + pending estimates + this request's new blocks.
	all := int64(s.index.UniqueBlocksLocked()) + s.pendingTotal
	return all+est <= int64(s.budgetBlocks)
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

// FCFSServed counts pre-capacity dispatches (the gauge substrate).
func (s *Scheduler) FCFSServed() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
	return s.counters.fcfsServed
}

// syncWaitersLocked reconciles the waiter view with the queue-head
// snapshot: new sessions become waiters (estimates from history + wire
// size), vanished sessions leave, member waiting-since tracks the oldest
// queued head (the victim preference's input).
func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Session == "" {
			continue
		}
		present[h.Session] = h.Arrival
		if m, seated := s.members[h.Session]; seated {
			if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
				m.waitingSince = h.Arrival
			}
			m.lastActivity = now
			continue
		}
		w, ok := s.waiters[h.Session]
		if !ok {
			w = &waiter{id: h.Session, newcomer: s.isNewcomerLocked(h.Session)}
			s.waiters[h.Session] = w
		}
		// B5: keep the OLDEST arrival across fan-out re-arrivals
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
		w.sizeBytes = h.SizeBytes
		w.coldTokens = s.estimateCostCachedLocked(h.Session, h.SizeBytes) * int64(s.blockSize) // tokens (R1)
	}
	// vanished: members clear their waiting marker; waiters leave
	for id, m := range s.members {
		if _, ok := present[id]; !ok {
			m.waitingSince = time.Time{}
		}
	}
	for id := range s.waiters {
		if _, ok := present[id]; !ok {
			delete(s.waiters, id)
		}
	}
}

// estimateCostCachedLocked is estimateCostLocked behind the M2 cache:
// valid while the index version is unchanged and the head size matches.
func (s *Scheduler) estimateCostCachedLocked(id string, sizeBytes int64) int64 {
	v := s.index.VersionLocked()
	if s.estVersion != v {
		s.estCache = map[string]estEntry{}
		s.estVersion = v
	}
	if e, ok := s.estCache[id]; ok && e.sizeBytes == sizeBytes {
		return e.blocks
	}
	b := s.estimateCostLocked(id, sizeBytes)
	s.estCache[id] = estEntry{sizeBytes: sizeBytes, blocks: b}
	return b
}

// estimateCostLocked is the conservative admission cost of a QUEUED
// request (S0-1 b/c): the session's persisted history blocks not held by
// the non-exiting group, plus the wire-size DELTA over the last known
// body (bytes/4/blockSize, zero sharing assumed). A session with no
// history costs its full wire size the same way.
func (s *Scheduler) estimateCostLocked(id string, sizeBytes int64) int64 {
	hist := s.index.SetBlocks(id) // persisted across residencies (CP1)
	known := int64(0)
	for _, h := range hist {
		if !s.index.HasNELocked(h) {
			known++
		}
	}
	// the baseline is the session's last DISPATCHED body size (the index
	// record, updated at PreRequest) — never the current head's size, or
	// the second sight of the same head would estimate zero new blocks
	// (found by the integration test: w seated on the retry pick)
	lastBytes := s.index.LastSizeBytesLocked(id)
	delta := sizeBytes - lastBytes
	if delta < 0 {
		delta = 0
	}
	return known + delta/4/int64(s.blockSize)
}

// ---- PreRequest (post-dispatch): the exact tier-A correction ----

// Dispatched records a released request at PreRequest: the exact keys
// replace the estimate (D13 absorbs the error at the next enforcement),
// in-flight and the cold charge are taken, and the size baseline moves.
func (s *Scheduler) Dispatched(id, reqID string, keys []Hash, sizeBytes int64, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return
	}
	s.index.SetSizeLocked(id, sizeBytes) // the estimate baseline
	onMain := s.index.SettleDispatchedAtLocked(id, reqID, keys, now)
	if m, ok := s.members[id]; ok {
		m.lastActivity = now
		m.lastSizeBytes = sizeBytes
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
			s.pendingTotal -= m.pendingBlocks
			m.pendingBlocks = 0 // the exact keys are in NE now (D13 reconciles)
		}
		if !m.coldDone && m.coldEntry > 0 {
			// refine the charge to the exact token count (R1: one unit)
			s.coldInFlight = max(s.coldInFlight-m.coldEntry, 0)
			m.coldEntry = int64(len(keys)) * int64(s.blockSize)
			s.coldInFlight += m.coldEntry
		}
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry
// charge is released HERE (B3: not at end of decode), exactly once.
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseColdLocked(id)
}

func (s *Scheduler) releaseColdLocked(id string) {
	m, ok := s.members[id]
	if !ok || m.coldDone || m.coldEntry <= 0 {
		return
	}
	s.coldInFlight = max(s.coldInFlight-m.coldEntry, 0)
	m.coldEntry = 0
	m.coldDone = true
}

// ---- settle (ResponseBody at EndOfStream: the finalizer) ----

// Settle records a terminated dispatched request: the in-flight
// decrement for EVERY cause; n_max counts every settled turn (A8); the
// payback ledger only on natural completion.
func (s *Scheduler) Settle(id, reqID string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSeen[id] = now
	m, seated := s.members[id]
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
		s.releaseColdLocked(id) // S0-2: any termination releases the charge
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
	if s.index.ExitingLocked(id) && m.inflight == 0 {
		s.exitLocked(m, "exiting-drain", now)
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
// anywhere cannot persist past the next decision point.
func (s *Scheduler) deriveLocked() {
	var cold, pend int64
	for _, m := range s.members {
		cold += m.coldEntry
		pend += m.pendingBlocks
	}
	s.coldInFlight = cold
	s.pendingTotal = pend
}

// reapInflightLocked drops in-flight ids whose termination was never
// observed (C3): provisional ids (chosen at Pick, never reaching
// PreRequest) after 30s; confirmed ids after InflightReapSeconds (the
// harness request timeout is 3600s — anything older is a leak).
func (s *Scheduler) reapInflightLocked(now time.Time) {
	for _, m := range s.members {
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
				s.releaseColdLocked(m.id)
			}
		}
	}
}

func (s *Scheduler) rotateLocked(now time.Time) {
	if len(s.waiters) == 0 {
		return
	}
	for {
		head := s.entryHeadLocked()
		if head == nil {
			return
		}
		ne := int64(s.index.UniqueBlocksNELocked()) + s.pendingTotal
		estBlocks := head.coldTokens / int64(s.blockSize)
		if ne+estBlocks <= int64(s.budgetBlocks) && s.coldAllows(head) {
			if !s.seatLocked(head, now) {
				// C4 (always-on): TrySeat failed although the estimate
				// passed — refresh the head's estimate from the
				// authoritative index state and RETURN. Continuing with
				// the stale estimate spins this loop forever under s.mu
				// (the 13:00 deadlock class, reproduced by the golden
				// scenario at sim≈1114).
				head.coldTokens = s.estimateCostLocked(head.id, head.sizeBytes) * int64(s.blockSize)
				return
			}
			continue
		}
		if reason, victim := s.rotationVictimLocked(now); victim != nil {
			if victim.inflight > 0 {
				s.index.MarkExitingLocked(victim.id)
				s.counters.exits["exiting"]++
				return
			}
			s.exitLocked(victim, reason, now)
			continue
		}
		return
	}
}

// entryHeadLocked: newcomers first (D7; oldest within the class), then
// oldest waiting (A4/A7: re-entrants keep their original since).
func (s *Scheduler) entryHeadLocked() *waiter {
	var head *waiter
	for _, w := range s.waiters {
		if !w.newcomer {
			continue
		}
		if head == nil || w.since.Before(head.since) || (w.since.Equal(head.since) && w.id < head.id) {
			head = w
		}
	}
	if head != nil {
		return head
	}
	for _, w := range s.waiters {
		if head == nil || w.since.Before(head.since) || (w.since.Equal(head.since) && w.id < head.id) {
			head = w
		}
	}
	return head
}

// coldAllows staggers ALL cold entries (B6: re-entries too — WEKA
// re-entries are fully cold), in TOKENS (R1). R11's rule: one cold entry
// is ALWAYS allowed — a session bigger than the cap must be seatable
// when nothing cold is in flight, or newcomers-first + hold-the-line
// deadlocks the queue on it forever.
func (s *Scheduler) coldAllows(w *waiter) bool {
	if s.cfg.ValveMode == "off" {
		return true // the stagger moved to the engine (Phase-2 deployment)
	}
	return s.coldInFlight == 0 || s.coldInFlight+w.coldTokens <= s.cfg.ColdEntryCapTokens
}

// rotationVictimLocked (§7 demand-driven exit). LEGACY order (B4,
// C2-as-shipped): eligible members sorted by (has-waiting, seatTime) —
// inverted vs ALGORITHM §7, kept as the default. SPEC order
// (victimOrder "spec"): no-queued-turn members first, then oldest
// last completion. The engine-idle rule stays the liveness fallback.
func (s *Scheduler) rotationVictimLocked(now time.Time) (string, *member) {
	var best *member
	var bestReason string
	bestHasWaiting := false
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
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
	for _, m := range s.members {
		if m.inflight > 0 || !m.waitingSince.IsZero() {
			idle = false
			break
		}
	}
	if idle {
		for _, m := range s.members {
			if s.index.ExitingLocked(m.id) {
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

// budgetVictimLocked is the §6 growth order. LEGACY: eligible, then
// idle (nothing in flight), then the exiting class — earliest-seated
// EVEN MID-FLIGHT (the caller marks it exiting, which drops its NE
// contribution immediately; it leaves when its requests drain, §6.3).
// QUANTUM mode replaces the whole ladder with the BUILD-PLAN §1 order:
// (1) quantum-complete idle; (2) done or silent > S; (3) mid-quantum
// idle, oldest last completion first; (4) the exiting class,
// mid-flight. Budget enforcement only — rotation never evicts an
// un-eligible member.
func (s *Scheduler) budgetVictimLocked(now time.Time) (string, *member) {
	if reason, m := s.rotationVictimLocked(now); m != nil {
		return reason, m
	}
	if s.cfg.RotationMode == "quantum" {
		// class (2): done (no queued head) or silent > S
		var best *member
		for _, m := range s.members {
			if s.index.ExitingLocked(m.id) || m.inflight > 0 {
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
		for _, m := range s.members {
			if s.index.ExitingLocked(m.id) || m.inflight > 0 {
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
		for _, m := range s.members {
			if s.index.ExitingLocked(m.id) {
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
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) || m.inflight > 0 {
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
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
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

func (s *Scheduler) exitLocked(m *member, reason string, now time.Time) {
	s.releaseColdLocked(m.id) // S0-2: unseat releases the charge
	s.pendingTotal -= m.pendingBlocks
	m.pendingBlocks = 0
	s.index.UnseatLocked(m.id)
	delete(s.members, m.id)
	s.counters.exits[reason]++
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d", reason, m.resTurns), now)
	if s.onExit != nil {
		s.onExit(reason, m.resTurns)
	}
	// A7: a session with a queued head re-enters at its ORIGINAL
	// waiting-since (FIFO survives the exit)
	if !m.waitingSince.IsZero() {
		s.waiters[m.id] = &waiter{id: m.id, since: m.waitingSince,
			sizeBytes:  m.lastSizeBytes,
			coldTokens: s.estimateCostLocked(m.id, m.lastSizeBytes) * int64(s.blockSize)}
	}
}

func (s *Scheduler) seatLocked(w *waiter, now time.Time) bool {
	// seat in the index first (history-based cost; the estimate-based
	// budget check already ran in rotate — D13 reconciles the difference).
	// Returns false when TrySeat rejects (the caller must refresh and
	// return — C4).
	if ok, _ := s.index.TrySeatLocked(w.id, s.budgetBlocks); !ok {
		return false
	}
	s.everSeated[w.id] = struct{}{}
	s.waitersdelete(w.id)
	m := &member{
		id:            w.id,
		seatTime:      now,
		lastActivity:  now,
		lastSizeBytes: s.index.LastSizeBytesLocked(w.id),
		// the head that triggered this seating is still queued (the
		// serve happens after rotate): inherit its presence so the
		// engine-idle rule cannot evict a member that is about to
		// dispatch (A1's livelock class, entering via the fresh record's
		// zero waitingSince — found by TestNewcomerFirst)
		waitingSince: w.since,
	}
	// B6: ALL entries carry a cold charge (tokens, R1)
	m.coldEntry = w.coldTokens
	s.coldInFlight += m.coldEntry
	// the budget counts the admission estimate (blocks) until the exact
	// keys land at PreRequest
	m.pendingBlocks = w.coldTokens / int64(s.blockSize)
	s.pendingTotal += m.pendingBlocks
	// C7: record ALL seat waits (newcomers AND re-entries — the last
	// campaign's re-entry waits were invisible); the newcomer counter
	// and the onEntry hook keep their legacy semantics (golden trace).
	kind := "re-entry"
	if w.newcomer {
		kind = "newcomer"
		wait := now.Sub(w.since).Seconds()
		s.counters.entryWaits = append(s.counters.entryWaits, wait)
		if s.onEntry != nil {
			s.onEntry(wait)
		}
	} else {
		s.counters.reentryWaits = append(s.counters.reentryWaits, now.Sub(w.since).Seconds())
	}
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f", kind, now.Sub(w.since).Seconds()), now)
	s.members[w.id] = m
	return true
}

func (s *Scheduler) waitersdelete(id string) { delete(s.waiters, id) }

// ---- budget enforcement (§6, growth) ----

func (s *Scheduler) enforceBudgetLocked(now time.Time) {
	if s.budgetBlocks <= 0 {
		return // pre-capacity (A5): no budget yet
	}
	for int64(s.index.UniqueBlocksNELocked())+s.pendingTotal > int64(s.budgetBlocks) {
		reason, victim := s.budgetVictimLocked(now)
		if victim == nil {
			return
		}
		if victim.inflight > 0 {
			s.index.MarkExitingLocked(victim.id)
			s.counters.exits["exiting"]++
			continue
		}
		s.exitLocked(victim, reason, now)
	}
}

// ---- lifecycle (§8): the ~1s ticker ----

func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deriveLocked() // derive-don't-accumulate, every decision point
	s.reapInflightLocked(now)
	// the derived window's chain expiry (windowMode "derived"):
	// chains with no dispatch for > S leave the session's set.
	if s.cfg.WindowMode == "derived" {
		s.index.ExpireChainsLocked(time.Duration(s.cfg.IdleSeconds*float64(time.Second)), now)
	}
	s.enforceBudgetLocked(now)
	for id, m := range s.members {
		if now.Sub(m.lastActivity) > time.Duration(s.cfg.TTLSeconds*float64(time.Second)) &&
			m.inflight == 0 && m.waitingSince.IsZero() {
			s.exitLocked(m, "ttl", now)
			s.index.ForgetLocked(id)
			delete(s.everSeated, id) // B7: prune consistently with the waiter path
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
// with a live head is NEVER forgotten, however long it waits).
func (s *Scheduler) gcSilentRecordsLocked(now time.Time) {
	ttl := time.Duration(s.cfg.TTLSeconds * float64(time.Second))
	for id, seen := range s.lastSeen {
		if _, isMember := s.members[id]; isMember {
			continue
		}
		if _, isWaiter := s.waiters[id]; isWaiter {
			continue // has a live head (syncWaiters keeps waiters honest)
		}
		if now.Sub(seen) > ttl {
			s.index.ForgetLocked(id)
			delete(s.everSeated, id)
			delete(s.lastSeen, id)
		}
	}
}

func (s *Scheduler) isNewcomerLocked(id string) bool {
	_, seen := s.everSeated[id]
	return !seen
}

// ---- introspection ----

// ---- the seat/exit event log (BUILD-PLAN §1 instrumentation: spans
// cannot see seats — every future run needs this) ----

// Event is one seat/exit/exiting decision, per id, with reason and
// timestamp. Kept as a bounded ring (dumped in DumpState) and emitted
// to the onEvent hook (the plugin logs one line per event).
type Event struct {
	T      time.Time `json:"t"`
	Kind   string    `json:"kind"`   // SEAT | EXIT | EXITING
	ID     string    `json:"id"`     // the session id
	Detail string    `json:"detail"` // reason=... turns=... / kind=... wait=...
}

const eventRingSize = 2048

func (s *Scheduler) recordEvent(kind, id, detail string, now time.Time) {
	ev := Event{T: now, Kind: kind, ID: id, Detail: detail}
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
	m, ok := s.members[id]
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
	snap := Snapshot{Seats: len(s.members), Waiters: len(s.waiters),
		Exits: map[string]int64{}, BudgetBlocks: s.budgetBlocks,
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed}
	if full {
		snap.Config = s.cfg
		snap.RecentEvents = append([]Event(nil), s.events...)
	}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	if s.index != nil {
		snap.UniqueBlocks = s.index.UniqueBlocks()
		snap.NEBlocks = s.index.UniqueBlocksNE()
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
	return snap
}

// Sessions returns the tracked-session count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.index.TrackedCount()
}

// Seated reports membership (tests).
func (s *Scheduler) Seated(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.members[id]
	return ok
}
