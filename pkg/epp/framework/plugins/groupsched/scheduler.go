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
	"math"
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

	// BindingTTLSeconds bounds the member→engine HOMING independently of the
	// member RECORD's lifetime. The two must not share a clock: the record is a
	// residency (TTLSeconds of silence forgets it), while the binding models
	// where the member's BLOCKS are — and the placement authority (the
	// session-affinity filter) pins for 3600 s. Tying the binding to the record
	// GC made every silent-then-returning member re-home by least backlog while
	// the filter still pinned it elsewhere: measured on the first live multi-pod
	// stack (e1, 2026-10-30 19:00Z) at 11 % of dispatches off-binding in steady
	// state. 0 → max(20 x TTLSeconds, 3600).
	BindingTTLSeconds float64 `json:"bindingTtlSeconds"`

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

	// ---- F-arm knobs (single pod; every one defaults to today's behaviour) ----

	// QuantumMode (rotationMode "quantum" only):
	//
	//	"fixed"    — Q busy periods for everybody (Q̄). DEFAULT.
	//	"dyn-size" — F3: Q_i = max(1, round(Q̄ · blocks_i / median seated blocks)).
	//	             Q̄ is the anchor's own Q, so the AVERAGE scale matches and only
	//	             the size effect varies. Job sizes are bimodal, so one uniform Q
	//	             is a compromise; a big member's slice should be longer (fewer
	//	             cold re-prefills, and waste is proportional to chain size).
	QuantumMode string `json:"quantumMode"`
	// QuantumCeiling is the TENURE BOUND (completed busy periods) of every dynamic
	// rule (F3/F6′/F7/F9). 0 = off (today's behaviour). A dynamic rule without a
	// ceiling can starve; this is the fairness guarantee that lets the others run.
	QuantumCeiling int `json:"quantumCeiling"`
	// EntryOrder selects D1's head order:
	//
	//	"newcomer-first" — newcomers first, oldest within the class, then id.
	//	                   DEFAULT (today's behaviour, A4/A7).
	//	"hrrn"           — F5: priority (wait + s)/s with s the entry's own estimated
	//	                   PREFILL SECONDS (cold tokens ÷ the run-time MEASURED
	//	                   prefill rate). Small-first without starving big jobs: wait
	//	                   grows without bound, so every head eventually wins.
	//	"hrrn-bounded"   — F5′: the same ratio with s_eff = min(cold tokens, the live
	//	                   MEDIAN cold tokens of the CURRENT waiters), recomputed once
	//	                   per decision pass from the scheduler's own waiter pool.
	//	                   Parameter-free, and it is what bounds F5's far tail: a
	//	                   200K-token entry otherwise ages ~67x slower than a 3K one,
	//	                   so its wait was measured at ~an hour (B1b) instead of the
	//	                   ~10 min the median rate gives it. Small entries (below the
	//	                   median) keep F5's ordering exactly; entries at or above it
	//	                   age at the median rate, oldest first among themselves.
	EntryOrder string `json:"entryOrder"`
	// RotateWhenFree (F7): when true, a rotation-based eviction is allowed only
	// while the engine's PREFILL BACKLOG (dispatched, no first token yet, in
	// TOKENS) is under MaxNumBatchedTokens. When prefill is saturated the slices
	// lengthen instead — up to QuantumCeiling, which is why F7 always carries it.
	RotateWhenFree bool `json:"rotateWhenFree"`
	// MaxNumBatchedTokens is the engine's own per-step token budget: the
	// denominator of F7's gate. It MUST equal the engine's
	// --max-num-batched-tokens (the campaign stacks run 8192, from exp-ops
	// _defaults.env VLLM_MAX_NUM_BATCHED_TOKENS); it is a WIRING FACT, not a knob of
	// ours — the endpoint metrics do not carry it.
	MaxNumBatchedTokens int `json:"maxNumBatchedTokens"`
	// LongQuantum (F9, rotationMode "mlfq2"): Q_l, the long pool's slice length in
	// busy periods. 0 → 8 (the B1a winner sets the value).
	LongQuantum int `json:"longQuantum"`

	// ---- PAT knob (B2: the pause-aligned tenure) ----

	// EligibilityMode selects the exit-eligibility MODEL — WHO may be rotated. It is
	// orthogonal to VictimOrder (who goes FIRST among the eligible):
	//
	//	"off" — the existing models (rotationMode's busy-period quantum / mlfq2, or the
	//	        legacy ledger rule). DEFAULT, golden-pinned.
	//	"pat" — PAT, the PAUSE-ALIGNED TENURE. A member is rotation-eligible iff
	//	        (a) it has completed its FAIR SHARE (Q_fair = the quantum, 4 busy
	//	            periods by default) AND it is IDLE (no queued turn, nothing in
	//	            flight) AND its residue class is COLD — the parameter-free residue
	//	            model (residue.go) predicts under half of its chain still resident,
	//	            i.e. the PAUSE OUTLIVED THE CACHE, a fact derived from block
	//	            ALLOCATIONS (no clock, no threshold, no cache read); OR
	//	        (b) it reaches the safety CEILING (quantumCeiling; 8 busy periods — the
	//	            B1a measured-safe knee — when unset).
	//
	// WHY IT SHOULD DOMINATE A FIXED Q8: a long-pausing member returns COLD under BOTH
	// rules, so the cold tokens it costs are the same either way — but under Q8 it holds
	// its seat idle through the pause while a waiter queues. PAT rotates at the
	// CACHE-DEAD moment instead of at the 8th busy period (which may be WARM), so the
	// seat serves a waiter during the pause at ZERO cache cost. And nothing is released
	// BEFORE fair share: that is the fairness guarantee the removed F6′ lacked.
	//
	// FAIL-CLOSED PAIRS (validated at startup, never silently degraded): "pat" requires
	// rotationMode="quantum" (Q_fair IS the quantum — the same Q the anchor runs),
	// residue="tiebreak" (the COLD class is the model's, and the shadow is only built
	// when it is on) and victimOrder="residue" (F6″'s order is PAT's own victim order).
	EligibilityMode string `json:"eligibilityMode"`

	// ---- per-engine gating (both arms: the multi-pod gate coupling fix) ----

	// EngineGate applies the saturation gate PER ENGINE, at head selection:
	//
	//	"off"        — today's behaviour: the pool-level detector alone gates
	//	               dispatch, and admission never looks at an individual
	//	               engine's saturation. DEFAULT (golden-pinned).
	//	"per-engine" — a head is admitted only to an engine that is NOT saturated
	//	               (that engine's OWN kv < SatKvThreshold AND its waiting queue
	//	               < SatQueueThreshold). A saturated engine takes no new seats;
	//	               seats that free there still serve its own residents.
	//
	// WHY (measured): the pool detector aggregates over endpoints, so with one
	// engine at 0.95 beside one at 0.50 the pool reads 0.725 and NOTHING is gated
	// while an engine is saturated — and because the gate closes on the pool
	// average while one engine is starved, the under-filled engine keeps getting
	// FEWER seats (0.71:1): the gate never rebalances. Both arms of the E3′/E4
	// contrast carry this fix so the comparison is gate-clean.
	EngineGate string `json:"engineGate"`
	// SatKvThreshold / SatQueueThreshold are the per-engine saturation limits used
	// by EngineGate. They are the detector's own defaults (0.80 / 5), repeated
	// here because the policy must gate on the SAME numbers the detector does —
	// a different pair would make the two gates disagree.
	SatKvThreshold    float64 `json:"satKvThreshold"`
	SatQueueThreshold int     `json:"satQueueThreshold"`

	// ---- E4b knob (the parameter-free residue model) ----

	// Residue selects whether the seat-time engine choice uses the scheduler's own
	// RESIDUE MODEL (residue.go) as a TIE-BREAK among engines that already pass the
	// fit gate:
	//
	//	ResidueReconcile (json "residueReconcile", off|settle, default off) is the
	//	diagnostic knob: with "settle" the shadow is corrected from the measured
	//	cached_tokens at each settle (the reference's settle-corrected variant).
	//	Default "off" = open loop, because the validation's ablation found the two
	//	identical (0.920/0.964 vs 0.919/0.964) and open loop is the strict form of
	//	dictate-don't-observe. The measured hit ALWAYS feeds the calibration verdict
	//	and the actual-ratio series; it never moves state unless this knob says so.
	ResidueReconcile string `json:"residueReconcile"`
	//
	//	"off"       — fit → least relative backlog per seat → cheapest-victim.
	//	              DEFAULT, and what the E3 arm runs (the model is never built).
	//	"tiebreak"  — among FITTING engines, prefer the one where the model says
	//	              more of the member's chain is still resident; the backlog key
	//	              decides when the model has nothing to say (two cold engines
	//	              both read 0 — the model is NEVER a ranker of cold engines).
	//	              E4 sets this; the per-engine self-model resets (restart /
	//	              sustained mis-calibration) drop an engine to fit-only.
	Residue string `json:"residue"`

	// ---- E4a knob (the multi-pod binding model) ----

	// BindingMode selects WHAT the member→engine binding IS, and with it where
	// the pin comes from. It is the ONE variable between the E3 arm and the E4a
	// arm (FULL-RECORD-FOR-REVIEWER §9-11; PREREG-multipod-offload):
	//
	//	"member-stable" — the E3 design. The engine is chosen ONCE, at FIRST
	//	                 SIGHT of the member (least backlog per seat, round
	//	                 robin tie-break), and never changes while the member is
	//	                 active; waiters are PARTITIONED per engine (a waiter
	//	                 only queues at the engine it was homed to); the binding
	//	                 has its own TTL (BindingTTLSeconds) and OUTLIVES the
	//                 member record, so a silent-then-returning member still
	//                 finds its blocks; placement is pinned by the EXTERNAL
	//                 session-affinity-filter. THE DEFAULT (golden-pinned
	//                 byte-identically).
	//
	//	"seat"         — E4a: SEAT = PIN. The binding IS the member's CURRENT
	//	                 SEAT: created when the member is seated on an engine,
	//	                 dropped the moment it exits. No TTL, no pre-seat homing,
	//	                 no residue — an unseated member carries no placement
	//	                 opinion at all. Waiters live in ONE GLOBAL POOL (not
	//	                 partitioned by engine: whichever engine has room takes
	//	                 the global head), and the seat-time engine choice is
	//	                 fit → least relative backlog per seat → cheapest-victim
	//	                 engine (see seatEngineLocked — the E4b residue seam).
	//	                 Placement is pinned by THIS plugin's own scheduling
	//	                 Filter (filter.go), so the arm carries no external
	//	                 session-affinity-filter.
	BindingMode string `json:"bindingMode"`
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
		BindingMode:             "member-stable",
		Residue:                 "off",
		ResidueReconcile:        "off",
		QuantumMode:             "fixed",
		EntryOrder:              "newcomer-first",
		MaxNumBatchedTokens:     8192,
		EligibilityMode:         "off",
		EngineGate:              "off",
		SatKvThreshold:          0.80,
		SatQueueThreshold:       5,
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
	if c.BindingMode == "" {
		c.BindingMode = def.BindingMode
	}
	if c.Residue == "" {
		c.Residue = def.Residue
	}
	if c.ResidueReconcile == "" {
		c.ResidueReconcile = def.ResidueReconcile
	}
	if c.QuantumMode == "" {
		c.QuantumMode = def.QuantumMode
	}
	if c.EntryOrder == "" {
		c.EntryOrder = def.EntryOrder
	}
	if c.MaxNumBatchedTokens == 0 {
		c.MaxNumBatchedTokens = def.MaxNumBatchedTokens
	}
	if c.EngineGate == "" {
		c.EngineGate = def.EngineGate
	}
	if c.SatKvThreshold == 0 {
		c.SatKvThreshold = def.SatKvThreshold
	}
	if c.SatQueueThreshold == 0 {
		c.SatQueueThreshold = def.SatQueueThreshold
	}
	if c.EligibilityMode == "" {
		c.EligibilityMode = def.EligibilityMode
	}
	if c.BindingTTLSeconds == 0 {
		c.BindingTTLSeconds = 20 * c.TTLSeconds
		if c.BindingTTLSeconds < 3600 {
			c.BindingTTLSeconds = 3600
		}
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

// hrrnOrder reports whether the entry order is an HRRN variant (F5 or F5′). Both need
// the SAME per-dispatch prefill accounting, and treating them alike is what makes an
// F5-vs-F5′ comparison differ in the entry order alone: an F5′ arm must not also lose
// the PREFILL backlog bookkeeping that F7 and the F9 pools read.
func (c Config) hrrnOrder() bool {
	return c.EntryOrder == "hrrn" || c.EntryOrder == "hrrn-bounded"
}

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
	case "ledger", "quantum", "mlfq2":
	default:
		return fmt.Errorf("rotationMode: invalid value %q (ledger|quantum|mlfq2)", c.RotationMode)
	}
	switch c.QuantumMode {
	case "", "fixed", "dyn-size":
	default:
		return fmt.Errorf("quantumMode: invalid value %q (fixed|dyn-size)", c.QuantumMode)
	}
	switch c.EntryOrder {
	case "", "newcomer-first", "hrrn", "hrrn-bounded":
	default:
		return fmt.Errorf("entryOrder: invalid value %q (newcomer-first|hrrn|hrrn-bounded)", c.EntryOrder)
	}
	if c.QuantumCeiling < 0 {
		return fmt.Errorf("quantumCeiling must be >= 0 (0 = off), got %d", c.QuantumCeiling)
	}
	if c.MaxNumBatchedTokens < 0 {
		return fmt.Errorf("maxNumBatchedTokens must be >= 0, got %d", c.MaxNumBatchedTokens)
	}
	if c.VictimOrder == "residue" && !c.residueOn() {
		return fmt.Errorf("victimOrder=\"residue\" requires residue=\"tiebreak\" (F6\" reads the same per-engine shadow)")
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
	case "legacy", "spec", "residue":
	default:
		return fmt.Errorf("victimOrder: invalid value %q (legacy|spec|residue)", c.VictimOrder)
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
	if c.BindingTTLSeconds < 0 {
		return fmt.Errorf("bindingTtlSeconds must be >= 0 (0 = max(20 x ttlSeconds, 3600); got %v)", c.BindingTTLSeconds)
	}
	if c.CapacityLivenessSeconds < 0 {
		return fmt.Errorf("capacityLivenessSeconds must be >= 0 (0 = default; got %v)", c.CapacityLivenessSeconds)
	}
	switch c.EngineGroups {
	case "", "single", "per-engine":
	default:
		return fmt.Errorf("engineGroups: invalid value %q (single|per-engine)", c.EngineGroups)
	}
	switch c.BindingMode {
	case "", "member-stable", "seat":
	default:
		return fmt.Errorf("bindingMode: invalid value %q (member-stable|seat)", c.BindingMode)
	}
	switch c.ResidueReconcile {
	case "", "off", "settle":
	default:
		return fmt.Errorf("residueReconcile: invalid value %q (off|settle)", c.ResidueReconcile)
	}
	switch c.EngineGate {
	case "", "off", "per-engine":
	default:
		return fmt.Errorf("engineGate: invalid value %q (off|per-engine)", c.EngineGate)
	}
	if c.SatKvThreshold < 0 || c.SatKvThreshold > 1 {
		return fmt.Errorf("satKvThreshold must be in (0,1], got %v", c.SatKvThreshold)
	}
	if c.SatQueueThreshold < 0 {
		return fmt.Errorf("satQueueThreshold must be >= 0, got %d", c.SatQueueThreshold)
	}
	switch c.Residue {
	case "", "off", "tiebreak":
	default:
		return fmt.Errorf("residue: invalid value %q (off|tiebreak)", c.Residue)
	}
	switch c.EligibilityMode {
	case "", "off", "pat":
	default:
		return fmt.Errorf("eligibilityMode: invalid value %q (off|pat)", c.EligibilityMode)
	}
	if c.EligibilityMode == "pat" {
		// PAT reads its three inputs from knobs that already exist, and a silent
		// fallback would turn the arm into a differently-named Q-arm. All three pairs
		// are therefore startup failures.
		if c.RotationMode != "quantum" {
			return fmt.Errorf("eligibilityMode=\"pat\" requires rotationMode=\"quantum\" (Q_fair IS the quantum; got %q)", c.RotationMode)
		}
		if !c.residueOn() {
			return fmt.Errorf("eligibilityMode=\"pat\" requires residue=\"tiebreak\" (the COLD class is the residue model's verdict, and the shadow is only built when it is on)")
		}
		if c.VictimOrder != "residue" {
			return fmt.Errorf("eligibilityMode=\"pat\" requires victimOrder=\"residue\" (PAT's victim order among the eligible IS F6″'s; got %q)", c.VictimOrder)
		}
		if c.QuantumCeiling > 0 {
			// The ceiling is a CAP, and a cap below the fair share would release members
			// BEFORE their share — the one thing PAT guarantees never happens. (For the
			// per-member rules a low cap is meaningful: it only trims the members whose
			// own base is above it.)
			q := c.quantum()
			if q < 0 {
				return fmt.Errorf("eligibilityMode=\"pat\" needs a finite quantumBusyPeriods (Q_fair is the fair share; got %d)", q)
			}
			if c.QuantumCeiling < q {
				return fmt.Errorf("eligibilityMode=\"pat\" with quantumCeiling=%d below the fair share Q_fair=%d would release members before their share (the ceiling caps, it cannot undercut the base)", c.QuantumCeiling, q)
			}
		}
	}
	if c.BindingMode == "seat" && !c.perEngine() {
		// seat=pin is a MULTI-POD model: under engineGroups=single there is one
		// group, one binding target and nothing to pin, so the arm would measure
		// the single-engine behaviour while claiming the seat model.
		return fmt.Errorf("bindingMode=\"seat\" requires engineGroups=\"per-engine\" (the seat model only means something with more than one engine group; got %q)", c.EngineGroups)
	}
	return nil
}

// seatPin reports whether the binding follows the SEAT (E4a) rather than being
// chosen at first sight and kept (E3).
func (c Config) seatPin() bool { return c.BindingMode == "seat" }

// residueOn reports whether the residue model ranks the fitting engines (E4b).
func (c Config) residueOn() bool { return c.Residue == "tiebreak" }

// residueReconcile reports whether the residue shadow is corrected from the
// measured cached_tokens at settle. OFF by default: the shipped model is OPEN LOOP
// (dictate, don't observe) and the measured hit is a validation signal only.
func (c Config) residueReconcile() bool { return c.ResidueReconcile == "settle" }

// engineGating reports whether admission is gated per engine.
func (c Config) engineGating() bool { return c.EngineGate == "per-engine" }

// patOn reports whether PAT (the pause-aligned tenure) is the eligibility model.
func (c Config) patOn() bool { return c.EligibilityMode == "pat" }

// reapSeconds is the effective confirmed-id reap bound (0 → default).
func (c Config) reapSeconds() float64 {
	if c.InflightReapSeconds > 0 {
		return c.InflightReapSeconds
	}
	return 3700
}

// engineLoad is ONE engine's pushed saturation snapshot.
type engineLoad struct {
	kv      float64
	waiting int
	at      time.Time
}

// SetEngineLoad records one engine's saturation snapshot (called by the plugin from
// its metrics poll). A zero `at` clears the entry.
func (s *Scheduler) SetEngineLoad(engine string, kv float64, waiting int, at time.Time) {
	if !s.cfg.engineGating() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.load == nil {
		s.load = map[string]engineLoad{}
	}
	if at.IsZero() {
		delete(s.load, engine)
		return
	}
	s.load[engine] = engineLoad{kv: kv, waiting: waiting, at: at}
}

// EngineLoadStat is one engine's pushed saturation snapshot, exported for the
// metrics layer (the gate's inputs must be visible in the run artifacts or "the gate
// was closed X% of the time" is unfalsifiable — the banked arms left only a final
// snapshot behind).
type EngineLoadStat struct {
	KV      float64
	Waiting int
}

// EngineLoads snapshots the per-engine gate inputs.
func (s *Scheduler) EngineLoads() map[string]EngineLoadStat {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]EngineLoadStat, len(s.load))
	for k, v := range s.load {
		out[k] = EngineLoadStat{KV: v.kv, Waiting: v.waiting}
	}
	return out
}

// ForgetEngineLoadsExcept drops load snapshots for engines the EPP no longer sees.
func (s *Scheduler) ForgetEngineLoadsExcept(keep map[string]struct{}) {
	if !s.cfg.engineGating() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for k := range s.load {
		if _, ok := keep[k]; !ok {
			delete(s.load, k)
		}
	}
}

// saturatingLocked reports whether an engine is SATURATED by its own numbers: kv
// utilisation at or above the detector's threshold, or a waiting queue at or above
// its queue threshold. An engine with no snapshot is NOT saturated: absence of
// evidence must not silently stop admission on a healthy engine (the pool detector
// is the fail-closed layer, and capacity_missing already alarms on unknown engines).
func (s *Scheduler) saturatingLocked(g *engineGroup) bool {
	if g == nil || !s.cfg.engineGating() {
		return false
	}
	l, ok := s.load[g.id]
	if !ok || l.at.IsZero() {
		return false
	}
	// THE DETECTOR'S OWN FORMULA, de-averaged: the per-endpoint score is
	//
	//	saturation_e = max(waiting_e / queueThreshold, kv_e / kvThreshold)
	//
	// (utilization.Detector.Saturation's roofline, for ONE endpoint) and the engine
	// is saturated at >= 1. This is deliberately the SCRAPED kv, not the policy's
	// own NE/budget: they measure different things (kv_cache_usage is the blocks
	// held by RUNNING requests; NE is the seated set the budget tracks), and the
	// gate's job is to leave preemption headroom for running work.
	return math.Max(float64(l.waiting)/float64(s.cfg.SatQueueThreshold), l.kv/s.cfg.SatKvThreshold) >= 1
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
	// FairnessID is the request's flow-control identity — the id the GATEWAY and
	// the harness record on the wire (x_correlation_id / the client's thread id).
	// The member id is a DERIVED branch id and joins to nothing in the request
	// records (measured 0/768 matches), so every SEAT/EXIT event must carry this
	// too or a run's seat membership cannot be joined to its own requests.
	FairnessID string
}

// ---- per-session scheduling state (§3) ----

type member struct {
	id string
	// patReason is PAT's eligibility LATCH (eligibilityMode "pat" only): the reason
	// this member currently IS rotation-eligible under PAT, "" when it is not. It
	// exists so the two mechanism counters book MEMBERS BECOMING ELIGIBLE (a
	// transition) instead of one increment per decision pass over the same member.
	patReason string
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
	// prefillPending is this member's prefill work in TOKENS for its dispatches that
	// have not produced a first token yet (F7's backlog input; released at
	// StartOfStream, exactly like the cold charge).
	prefillPending int64
	// fairnessID is the last fairness id (queue key) this member was seen with —
	// the join key for its SEAT/EXIT events (see Head.FairnessID).
	fairnessID string
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
	// (R1: one unit everywhere — blocks x blockSize). Under seat=pin it is the
	// pool's BEST-CASE value (the minimum over engines) and serves the D1 / HRRN
	// service input; the ADMISSION cost is the per-engine vector below.
	coldTokens int64
	// coldTokensByEngine is the per-engine admission cost, FROZEN at sync time for
	// this decision pass (reviewer R2). Each entry comes from that engine's own
	// index, so an engine that does not hold the member's blocks is costed as cold
	// — which is the truth about that engine — instead of inheriting the pool's
	// best case.
	coldTokensByEngine map[string]int64
	// fairnessID rides with the head (see Head.FairnessID) so a SEAT event can name
	// the wire-level identity even before the member has a record.
	fairnessID string
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

	// residue is THIS engine's shadow of the free-block queue (E4b). nil unless the
	// arm turned the model on, so an off arm pays nothing at all.
	residue *residueEngine
	// resFlights holds the in-flight residue state per request id (the nodes the
	// request holds, freed at settle in reverse position order).
	resFlights map[string]residueHeld
	// residueReported latches the one-shot "this engine dropped to fit-only" log.
	residueReported bool
	// prefillRate is the MEASURED prefill rate (tokens/s) for this engine and
	// prefillSamples how many observations it rests on (F5's s, and nothing else).
	prefillRate    float64
	prefillSamples int
	// prefillBacklog is the engine's EPP-side prefill backlog in TOKENS: dispatched
	// requests with no first token yet (F7's gate). Derived per pass from the
	// members' pending prefill charges, like coldInFlight.
	prefillBacklog int64

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
		// engineGateSkipped counts dispatches the PER-ENGINE saturation gate held
		// back (G1: the pool gate applied to one engine).
		engineGateSkipped int64
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
		estCache: map[string]estEntry{}, resFlights: map[string]residueHeld{}}
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
	// onBind / onUnbind are the PLACEMENT-PIN hooks (E4a): they fire when a
	// member's engine binding is created or dropped, so the plugin's lock-light
	// Filter table can follow the seats without ever touching s.mu. nil in
	// tests and in member-stable mode (nothing to mirror there — the pin is
	// the external filter's).
	onBind   func(member, engine string)
	onUnbind func(member string)
	// onResiduePick / onResidueSettle / onResidueDisabled are the E4b
	// counterfactual export (nil-safe; nil in tests and when the model is off).
	// onResiduePick fires per seat-time engine choice with the predicted surviving
	// fraction and whether the model actually DECIDED it; onResidueSettle fires at
	// each settle with (predicted, actual) so the arm can score its own argmax
	// post-hoc against usage.cached_tokens.
	onResiduePick     func(engine string, predicted float64, decided bool)
	onResidueSettle   func(engine string, predicted, actual float64)
	onResidueDisabled func(engine string)
	// onValveBlockedEviction fires when a head would FIT an engine's budget but the
	// cold-entry valve refuses it, so the loop's only way forward is an eviction
	// (the C1 sizing signal; installed in BOTH binding models).
	onValveBlockedEviction func(engine string)
	// onEngineGateSkip fires when the per-engine saturation gate holds a dispatch
	// back (G1).
	onEngineGateSkip func(engine string)
	// onPatEligible fires when a member BECOMES rotation-eligible under PAT (a
	// transition, not a decision pass), carrying the rule that made it eligible — it
	// is how the mechanism stays countable in the artifacts (nil-safe; nil in tests
	// and in every non-PAT arm).
	onPatEligible func(engine, reason string)

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
	// bindingSeen is the homing's own last-activity clock (see
	// Config.BindingTTLSeconds): the binding outlives the member RECORD, so it
	// needs a lifetime of its own or every silent return re-homes off the
	// placement filter's pin. UNUSED under bindingMode=seat (there the binding
	// lives exactly as long as the seat).
	bindingSeen map[string]time.Time

	// gWaiters is the GLOBAL waiter pool (bindingMode=seat ONLY). In the
	// member-stable model a waiter queues at the engine it was homed to
	// (engineGroup.waiters — the partition is what makes "whichever engine
	// frees a seat takes ITS head" well defined). Seat=pin has no pre-seat
	// homing, so the pool cannot be partitioned: the waiting demand belongs to
	// no engine until an engine seats it, and each rotate pass hands the
	// global head to an engine that has room. nil outside seat mode (so a
	// stray read is a nil-map read, never a silent cross-model mix).
	gWaiters map[string]*waiter

	// load is the PER-ENGINE saturation view the plugin pushes up from the endpoint
	// metrics it already polls (kv utilisation and waiting queue depth). A pushed
	// snapshot, never a schedule: the policy never scrapes and never blocks. Keyed
	// by endpoint id and only allocated when the per-engine gate is on.
	load map[string]engineLoad

	// pickSeq synthesizes request ids for heads that lack one (M1).
	pickSeq int64
	// now is the virtual clock of the CURRENT decision pass (set at the top of every
	// public decision entry point). The ordering layer needs "how long has this head
	// waited" and must not call time.Now: a golden replay has to be reproducible.
	now time.Time
	// promoted is the MLFQ-2 class of a member (F9): true once it has outlived its
	// first busy period. It is MEMBER-scoped and survives rotation and exit — the
	// class is a property of the job, not of the seat.
	promoted map[string]bool
	// turnSeen / gapHistory are the per-member INTER-TURN GAP history (F4's
	// own-median pause rule, DYN-PAUSE): turnSeen is the last arrival seen for
	// the member, gapHistory the last gapHistoryLimit gaps between successive
	// arrivals. Member-scoped (never engine- or seat-scoped): the gap is a
	// property of the client's think time, and it must survive rotation, exit
	// and re-entry — that is exactly the population the pause rule is about.
	turnSeen   map[string]time.Time
	gapHistory map[string][]float64
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
		sessionRefs: map[string]int{}, bindings: map[string]string{},
		bindingSeen: map[string]time.Time{},
		turnSeen:    map[string]time.Time{}, gapHistory: map[string][]float64{},
		promoted: map[string]bool{}}
	if cfg.seatPin() {
		s.gWaiters = map[string]*waiter{}
	}
	s.counters.exits = map[string]int64{}
	s.counters.memberSources = map[string]int64{}
	if cfg.engineGating() {
		s.load = map[string]engineLoad{}
	}
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
	if s.cfg.residueOn() {
		// capacity is unknown until the engine's metrics arrive; SetNumBlocksFor
		// sizes the shadow and a capacity change voids it (a re-created pool).
		g.residue = newResidueEngine(0)
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

// waitersLocked is the waiter pool that engine g's admission loop serves:
//
//	bindingMode=member-stable — g's OWN map (the partition is structural: an
//	    unseated member was homed at first sight, so its head belongs to one
//	    engine and only that engine may seat it).
//	bindingMode=seat — the ONE GLOBAL pool, the same map for every engine. A
//	    waiter is not homed, so there is nothing for an engine to own; each
//	    rotate pass offers the global head to an engine that has room.
//
// A map is a reference type on purpose: callers mutate it in place (insert /
// delete) and both models share every call site. Returning nil is never valid
// in seat mode (NewScheduler allocates the pool), and in member-stable mode a
// nil g is a programming error the callers already guard.
func (s *Scheduler) waitersLocked(g *engineGroup) map[string]*waiter {
	if s.cfg.seatPin() {
		return s.gWaiters
	}
	if g == nil {
		return nil
	}
	return g.waiters
}

// waiterPoolSizeLocked is the total waiting-head count across every pool that
// is live in the current mode (used by the liveness guards that must ask "is
// ANYTHING waiting" without knowing the model).
func (s *Scheduler) waiterPoolSizeLocked() int {
	if s.cfg.seatPin() {
		return len(s.gWaiters)
	}
	n := 0
	for _, g := range s.groups {
		n += len(g.waiters)
	}
	return n
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
	if g.residue != nil {
		// the shadow's pool is the ENGINE's measured num_gpu_blocks (not the 95%
		// budget): the free queue is the whole KV pool.
		g.residue.setCapacity(numBlocks)
	}
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
	s.now = now
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
		if s.saturatingLocked(g) {
			// THE PER-ENGINE DISPATCH GATE (reviewer G1): a saturated engine holds its
			// own DISPATCHES too, not just its admissions. Gating only seats would let a
			// saturated engine's running KV grow past the threshold unbounded — the
			// preemption hazard the 80% gate exists to prevent (G90's failure mode) —
			// and it would not be the detector's semantics de-averaged: the detector
			// gates EVERY dispatch, so this must too. First-fit, like the size-aware
			// gate's skip: the head stays queued and another engine's heads dispatch.
			g.counters.engineGateSkipped++
			if s.onEngineGateSkip != nil {
				s.onEngineGateSkip(g.id)
			}
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
	// what the engine cannot serve yet). Under seat=pin the unseated demand is
	// the GLOBAL pool (a waiter belongs to no engine yet), so the numerator is
	// the same for every engine and the denominator — the seats it holds — is
	// what separates them: the engine with more seats per queued member is the
	// one that should take the next head.
	backlog := len(s.waitersLocked(g))
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
		if h.FairnessID != "" {
			if g := s.groupOfLocked(h.Member); g != nil {
				if m, seated := g.members[h.Member]; seated && m.fairnessID != h.FairnessID {
					m.fairnessID = h.FairnessID
				}
			}
		}
		// the F4 pause rule's input: the gap between successive TURNS of this
		// member, measured on the head's own enqueue time (the truthful
		// inter-arrival). It is a fact about the MEMBER, so it is recorded
		// before the seated/waiter split — a re-entering member's history must
		// survive its seat.
		s.noteTurnArrivalLocked(h.Member, h.Arrival)
		if g := s.groupOfLocked(h.Member); g != nil {
			if m, seated := g.members[h.Member]; seated {
				if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
					m.waitingSince = h.Arrival
				}
				m.lastActivity = now
				continue
			}
		}
		if s.cfg.seatPin() {
			// SEAT=PIN: there is no pre-seat homing, so the waiter joins the ONE
			// GLOBAL pool and its engine is decided at SEAT time (seatEngineLocked).
			w, ok := s.gWaiters[h.Member]
			if !ok {
				w = &waiter{id: h.Member, newcomer: s.isNewcomerLocked(h.Member)}
				s.gWaiters[h.Member] = w
			}
			w.session = sessionUnitOf(h.Session, h.Member)
			if w.since.IsZero() || h.Arrival.Before(w.since) {
				w.since = h.Arrival
			}
			w.sizeBytes = h.SizeBytes
			// per-engine admission costs, frozen for this pass (R2) + the pool
			// best case as the service input.
			s.setWaiterCostsLocked(w)
			continue
		}
		g := s.homeEngineLocked(h.Member, now)
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
		w.fairnessID = h.FairnessID
		// B5: keep the OLDEST arrival across fan-out re-arrivals
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
		w.sizeBytes = h.SizeBytes
		// tokens (R1); under member-stable the scalar IS the admission cost because
		// there is exactly one engine to cost against.
		w.coldTokens = s.estimateCostCachedLocked(g, h.Member, h.SizeBytes) * int64(s.blockSize)
		w.coldTokensByEngine = map[string]int64{g.id: w.coldTokens}
	}
	// vanished: members clear their waiting marker; waiters leave
	for _, g := range s.activeGroupsLocked() {
		for id, m := range g.members {
			if _, ok := present[id]; !ok {
				m.waitingSince = time.Time{}
			}
		}
	}
	for _, wp := range s.waiterPoolsLocked() {
		for id := range wp {
			if _, ok := present[id]; !ok {
				delete(wp, id)
			}
		}
	}
}

// waiterPoolsLocked is every live waiter pool in the current mode: the engine
// pools (member-stable) or the single global pool (seat=pin).
func (s *Scheduler) waiterPoolsLocked() []map[string]*waiter {
	if s.cfg.seatPin() {
		return []map[string]*waiter{s.gWaiters}
	}
	out := make([]map[string]*waiter, 0, len(s.groups))
	for _, g := range s.activeGroupsLocked() {
		out = append(out, g.waiters)
	}
	return out
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
func (s *Scheduler) homeEngineLocked(member string, now time.Time) *engineGroup {
	if s.cfg.seatPin() {
		// SEAT=PIN: there is NO pre-seat homing. An unseated member has no engine
		// (its head is ordinary demand in the global pool) and a seated one is
		// bound by its SEAT, which seatLocked/exitLocked create and drop.
		return nil
	}
	if id, ok := s.bindings[member]; ok {
		if g, ok := s.groups[id]; ok {
			s.bindingSeen[member] = now // the homing is alive (virtual clock: never time.Now — a
			// golden replay must be reproducible and a wall clock here breaks that)
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
	s.bindingSeen[member] = now
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
		s.bindingSeen[id] = now
		g.counters.homings++ // a binding created here is still a homing
		return g
	}
	if from.id == observed {
		s.bindingSeen[id] = now
		return from
	}
	to, err := s.groupForLocked(observed)
	if err != nil || to == nil {
		return from
	}
	m := from.members[id]
	if m == nil {
		s.bindings[id] = to.id
		s.bindingSeen[id] = now
		to.counters.homings++
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
	s.bindingSeen[id] = now
	to.counters.homings++
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
	// F7's prefill charge MUST be measured BEFORE the keys are settled into the
	// index: SettleDispatchedAtLocked records them, so afterwards the engine holds
	// every one of them and the "uncached" estimate would read ZERO for every
	// dispatch (measured: the gate never saw a single backlogged token).
	prefillUncached := int64(0)
	if s.cfg.RotateWhenFree || s.cfg.hrrnOrder() || s.cfg.RotationMode == "mlfq2" {
		prefillUncached = s.uncachedTokensLocked(g, keys)
	}
	onMain := g.index.SettleDispatchedAtLocked(id, reqID, keys, now)
	if s.cfg.RotateWhenFree || s.cfg.hrrnOrder() || s.cfg.RotationMode == "mlfq2" {
		// F7's backlog input and F5's service estimate: the request's PREFILL WORK is
		// its prompt tokens (the exact tier-A keys x block size), held against the
		// engine until the first token arrives.
		if m, ok := g.members[id]; ok {
			m.prefillPending = int64(len(keys)) * int64(s.blockSize)
			g.prefillBacklog += m.prefillPending
		}
	}
	if g.residue.usable() {
		// E4b: model this admission. The exact tier-A keys ARE the prompt blocks,
		// so the model is fed the truth about WHAT was asked for; only the hit/miss
		// split is a prediction (reconciled at settle against usage.cached_tokens).
		held := g.residue.dispatch(id, len(keys))
		if g.resFlights == nil {
			g.resFlights = map[string]residueHeld{}
		}
		g.resFlights[reqID] = held
	}
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
		if s.cfg.RotateWhenFree || s.cfg.hrrnOrder() || s.cfg.RotationMode == "mlfq2" {
			// F7's backlog input and F5/F9's service estimate: the request's PREFILL
			// WORK is its prompt tokens (the exact tier-A keys, block-quantised), held
			// against the engine until the first token arrives (ChunkAt releases it).
			g.prefillBacklog = max(g.prefillBacklog-m.prefillPending, 0)
			// ESTIMATED UNCACHED work against THIS engine (reviewer RF7): prompt tokens
			// would make the gate degenerate — one warm 50-100k-token turn at ~0.9 hit
			// exceeds an 8,192-token step budget 6-12x while its real prefill is a few
			// k, so the gate would sit closed and F7 would stop being load-adaptive.
			m.prefillPending = prefillUncached
			g.prefillBacklog += m.prefillPending
		}
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry charge is
// released HERE (B3: not at end of decode), exactly once. It carries no clock, so it
// cannot feed the prefill-RATE measurement (see ChunkAt).
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groupOfLocked(id); g != nil {
		s.releaseColdLocked(g, id)
		s.releasePrefillLocked(g, id)
	}
}

// ChunkAt is Chunk WITH the first-token clock: it is the only observation point for
// the prefill rate F5/F9 measure (time from dispatch to first token), so the live
// path calls this and a test that drives a virtual clock calls it too.
func (s *Scheduler) ChunkAt(id string, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOfLocked(id)
	if g == nil {
		return
	}
	s.releaseColdLocked(g, id)
	s.releasePrefillLocked(g, id)
}

// releasePrefillLocked ends a member's pending prefill at the FIRST TOKEN and
// releases the charge F7's gate reads (the rate measurement this used to feed is
// gone: HRRN's ordering is rate-free — see entryPriority).
func (s *Scheduler) releasePrefillLocked(g *engineGroup, id string) {
	m, ok := g.members[id]
	if !ok || m.prefillPending <= 0 {
		return
	}
	tokens := m.prefillPending
	m.prefillPending = 0
	g.prefillBacklog = max(g.prefillBacklog-tokens, 0)
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
	s.SettleOnUsage(id, reqID, engine, usage.Prompt, usage.Cached, 0, natural, now)
}

// SettleOnUsage is SettleOn with the completion token count, which the residue
// model needs (decode allocates output blocks exactly like prompt blocks — they
// are what a departed member leaves deepest in the free queue).
func (s *Scheduler) SettleOnUsage(id, reqID, engine string, promptTokens, cachedTokens, completionTokens int, natural bool, now time.Time) {
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
			// F9's ONLY promotion rule: a member that outlived its first busy period is
			// long-lived. Recorded globally so the class survives rotation and exit.
			if m.busyPeriods >= 1 {
				s.promoted[m.id] = true
			}
		}
	}
	if !seated {
		return
	}
	// E4b: reconcile the model against the engine's OWN outcome for this request
	// (usage.cached_tokens is the OUTCOME of our decisions, not a cache read).
	if g.residue.usable() {
		bs := int64(s.blockSize)
		hit := 0
		if cachedTokens > 0 {
			hit = int((int64(cachedTokens) + bs - 1) / bs)
		}
		pblk := 0
		if promptTokens > 0 {
			pblk = int((int64(promptTokens) + bs - 1) / bs)
		}
		oblk := 0
		if completionTokens > 0 {
			oblk = int((int64(completionTokens) + bs - 1) / bs)
		}
		held, had := g.resFlights[reqID]
		if !had {
			// no dispatch record (a settle we never saw dispatch: engine restart,
			// multi-EPP traffic): rebuild the request's footprint from the outcome.
			held = residueHeld{predHit: hit, full: pblk}
		}
		delete(g.resFlights, reqID)
		pred, act := g.residue.settle(s.cfg.residueReconcile(), id, held, hit, oblk)
		if s.onResidueSettle != nil {
			s.onResidueSettle(g.id, pred, act)
		}
		if g.residue.disabled && !g.residueReported {
			g.residueReported = true
			if s.onResidueDisabled != nil {
				s.onResidueDisabled(g.id)
			}
		}
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
	uncached := int64(promptTokens - cachedTokens)
	if uncached < 0 {
		uncached = 0
	}
	if !m.entryRecorded {
		m.entryCost = uncached // the entry turn's cold cost (measured; may be 0 when fully warm)
		m.entryRecorded = true
	} else {
		m.earned += int64(cachedTokens)
	}
	s.counters.resTurns = append(s.counters.resTurns, m.resTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(promptTokens)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// ---- rotation (§7) ----

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
		var cold, pend, pre int64
		for _, m := range g.members {
			cold += m.coldEntry
			pend += m.pendingBlocks
			pre += m.prefillPending
		}
		g.coldInFlight = cold
		g.pendingTotal = pend
		g.prefillBacklog = pre
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
	if s.cfg.seatPin() {
		// E4a: ONE global pool, one pool-wide admission loop whose per-head
		// engine choice is the seat-time rule (seatpin.go). The member-stable
		// model below stays per engine, where the pool really is partitioned.
		s.rotateSeatPoolLocked(now)
		return
	}
	groups := s.activeGroupsLocked()
	if s.waiterPoolSizeLocked() == 0 {
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
	if s.saturatingLocked(g) {
		// PER-ENGINE GATE: a saturated engine takes no new seats. Its own residents
		// keep dispatching (they are already admitted); only ADMISSION stops, which
		// is exactly what lets the sibling engine serve the queue instead.
		g.counters.gateSkipped++
		return
	}
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
		if ne+estBlocks <= int64(g.budgetBlocks) && !s.coldAllows(g, head) && s.onValveBlockedEviction != nil {
			// C1 sizing signal, identical in both binding models: the BUDGET would
			// have taken this head, but the cold-entry valve refused it, so the loop's
			// only way forward is an eviction.
			s.onValveBlockedEviction(g.id)
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
	if s.cfg.RotationMode == "mlfq2" {
		return s.mlfq2HeadLocked(g, seats)
	}
	waiters := s.waitersLocked(g)
	// F5′: ONE bound per decision pass, from the CURRENT waiter set — the same pool
	// this loop orders. 0 (any other entry order) leaves every key as it was.
	bound := s.entryColdBoundLocked(waiters)
	var head *waiter
	var headKey entryKey
	for _, w := range waiters {
		k := s.entryKeyBoundLocked(w, seats, bound)
		// the key is a TOTAL order (id breaks every tie), so the map's
		// iteration order cannot leak into the decision.
		if head == nil || k.less(headKey) {
			head, headKey = w, k
		}
	}
	return head
}

// mlfq2HeadLocked is F9's SEAT SPLIT: the short pool and the long pool each have a
// D1 head, and the seat goes to whichever head has the HIGHER HRRN response ratio
// (wait + s)/s. That is the load-adaptivity, and it is parameter-free: the ratio
// uses the measured prefill rate and the head's own cold tokens, and wait grows
// without bound, so neither pool can be starved by the other — the starvation
// argument is HRRN's, not a quota.
func (s *Scheduler) mlfq2HeadLocked(g *engineGroup, seats map[string]int) *waiter {
	var short, long *waiter
	var shortK, longK entryKey
	waiters := s.waitersLocked(g)
	// F5′ composes with F9 too: the bound is one per pass, over the pools' union (it
	// is 0 unless the entry order is hrrn-bounded, so plain mlfq2 is unaffected).
	bound := s.entryColdBoundLocked(waiters)
	for _, w := range waiters {
		k := s.entryKeyBoundLocked(w, seats, bound)
		if s.promoted[w.id] {
			if long == nil || k.less(longK) {
				long, longK = w, k
			}
			continue
		}
		if short == nil || k.less(shortK) {
			short, shortK = w, k
		}
	}
	switch {
	case short == nil:
		return long
	case long == nil:
		return short
	}
	now := s.nowLocked()
	shortP := entryPriority(now.Sub(short.since).Seconds(), short.coldTokens, s.blockSize)
	longP := entryPriority(now.Sub(long.since).Seconds(), long.coldTokens, s.blockSize)
	if shortP >= longP {
		return short
	}
	return long
}

// entryKey is the D1 ordering key. seats/isNew are 0-or-1 ranks (lower = first)
// so the comparison is plain lexicographic; hrrn is F5's response ratio, compared
// DESCENDING (higher priority first) after the session ranks, so the two layers
// compose instead of replacing each other.
type entryKey struct {
	seats int
	isNew int
	// hrrn is (wait + s)/s, or F5′'s min(cold, median)-bounded form of it, or 0 when
	// the ordering rule is not HRRN at all (in which case every key carries 0 and the
	// field is inert — that is what keeps the pinned newcomer-first order
	// byte-identical).
	hrrn  float64
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
	if k.hrrn != o.hrrn {
		return k.hrrn > o.hrrn // higher response ratio first
	}
	if !k.since.Equal(o.since) {
		return k.since.Before(o.since)
	}
	return k.id < o.id
}

func (s *Scheduler) entryKeyLocked(w *waiter, seats map[string]int) entryKey {
	return s.entryKeyBoundLocked(w, seats, 0)
}

// entryKeyBoundLocked builds the key under F5′'s pass bound (coldBound = the live
// median waiter size, 0 = unbounded). Direct callers get entryKeyLocked, which is the
// unbounded key: the bound is a property of a decision PASS over a waiter set, not of
// an individual waiter.
func (s *Scheduler) entryKeyBoundLocked(w *waiter, seats map[string]int, coldBound int64) entryKey {
	k := entryKey{since: w.since, id: w.id}
	switch s.cfg.EntryOrder {
	case "hrrn":
		// F5: wait / cold_tokens (the measured rate cancels — see entryPriority).
		k.hrrn = entryPriority(s.nowLocked().Sub(w.since).Seconds(), w.coldTokens, s.blockSize)
	case "hrrn-bounded":
		// F5′: wait / min(cold_tokens, median waiter size) — the far tail's fix.
		k.hrrn = boundedEntryPriority(s.nowLocked().Sub(w.since).Seconds(), w.coldTokens, coldBound, s.blockSize)
	}
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
	// F7: when the engine's PREFILL is already the bottleneck, a rotation's cold
	// re-prefill costs the whole pool — so hold, unless this member has hit the
	// tenure ceiling (the guarantee that stops holding from becoming starvation).
	gated := s.prefillSaturatedLocked(g)
	var best *member
	var bestReason string
	bestHasWaiting := false
	for _, m := range g.members {
		if g.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(g, m, now)
		if !ok {
			continue
		}
		if gated && !s.ceilingReachedLocked(g, m) {
			continue // F7: prefill saturated — lengthen slices, up to the ceiling
		}
		hasWaiting := !m.waitingSince.IsZero()
		var better bool
		if s.cfg.VictimOrder == "residue" {
			// F6″: the SAME eligible set (eligibility is untouched — nobody loses a
			// seat early), ordered by what the eviction COSTS: the coldest member
			// first. Class before raw ratio, so two cold members tie on class and the
			// spec tie-break (oldest last completion) decides instead of a noisy
			// 0.001-vs-0.002 comparison.
			better = best == nil || colderVictim(g, m, best)
		} else if s.cfg.VictimOrder == "spec" {
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
	// F7's gate closes the DEMAND-DRIVEN rotation above, never the engine-idle
	// liveness rule below: a completely idle engine with a blocked waiter is a stall,
	// and prefill that is not running cannot be saturated by our own choice.
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

// colderVictim is F6″'s order: cold (predicted surviving fraction below the warm/cold
// cut) before warm; then the RAW ratio, lower first; then oldest last completion; then
// id. A member whose engine has no usable model is treated as WARM (its cache is not
// known to be gone, so it is not the cheaper victim).
func colderVictim(g *engineGroup, a, b *member) bool {
	if g == nil || g.residue == nil {
		return olderCompletion(a, b)
	}
	ra, rb := g.residue.ratio(a.id), g.residue.ratio(b.id)
	ca, cb := ra < residueWarmFraction, rb < residueWarmFraction
	if g.residue.disabled {
		ca, cb = false, false // no verdict: fall through to the spec order
	}
	if ca != cb {
		return ca // cold goes first
	}
	if ca && cb && ra != rb {
		return ra < rb
	}
	return olderCompletion(a, b)
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
	s.recordEvent("EXIT", m.id, fmt.Sprintf("reason=%s turns=%d fairness_id=%s", reason, m.resTurns, fairnessOrNone(m.fairnessID)), now)
	if s.onExit != nil {
		s.onExit(engineLabel(g.id), reason, m.resTurns)
	}
	// A7: a session with a queued head re-enters at its ORIGINAL
	// waiting-since (FIFO survives the exit) — on the SAME engine (the
	// binding survives rotation: its cache is there).
	//
	// SEAT=PIN: the binding is dropped HERE (seat = pin: created at seat, dropped
	// at exit) and the re-entering head goes back to the GLOBAL pool — its next
	// seat is chosen fresh, by the fit/backlog rule, and may land on a different
	// engine. That is the model, not a leak: a member that left its seat holds no
	// placement claim until an engine seats it again.
	if s.cfg.seatPin() {
		delete(s.bindings, m.id)
		delete(s.bindingSeen, m.id)
		if s.onUnbind != nil {
			s.onUnbind(m.id)
		}
	}
	if !m.waitingSince.IsZero() {
		pool := s.waitersLocked(g)
		if pool != nil {
			w := &waiter{id: m.id, session: m.session, since: m.waitingSince, sizeBytes: m.lastSizeBytes}
			if s.cfg.seatPin() {
				s.setWaiterCostsLocked(w)
			} else {
				w.coldTokens = s.estimateCostLocked(g, m.id, m.lastSizeBytes) * int64(s.blockSize)
				w.coldTokensByEngine = map[string]int64{g.id: w.coldTokens}
			}
			pool[m.id] = w
		}
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

// countExitLocked books an exit (or an exiting-mark) on the engine AND in
// the global tally (Snapshot.Exits stays the whole-policy view).
func (s *Scheduler) countExitLocked(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

// setWaiterCostsLocked fills a waiting head's admission costs: the per-engine
// vector (each engine's OWN index estimate, frozen for the decision pass —
// reviewer R2) and, under seat=pin, the pool's best-case scalar (the minimum,
// used as the D1/HRRN service input, never as an admission cost).
func (s *Scheduler) setWaiterCostsLocked(w *waiter) {
	groups := s.activeGroupsLocked()
	if len(groups) == 0 {
		return
	}
	vec := make(map[string]int64, len(groups))
	best := int64(-1)
	for _, g := range groups {
		e := s.estimateCostCachedLocked(g, w.id, w.sizeBytes) * int64(s.blockSize)
		vec[g.id] = e
		if best < 0 || e < best {
			best = e
		}
	}
	if best < 0 {
		best = 0
	}
	w.coldTokensByEngine = vec
	w.coldTokens = best
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
	if s.onBind != nil {
		s.onBind(w.id, g.id) // the placement pin's mirror (seatpin/filter)
	}
	s.deleteWaiterLocked(g, w.id)
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
	s.recordEvent("SEAT", w.id, fmt.Sprintf("kind=%s wait=%.3f fairness_id=%s", kind, now.Sub(w.since).Seconds(), fairnessOrNone(w.fairnessID)), now)
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
	s.now = now
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
			s.forgetTurnHistoryLocked(id) // F4's own-median input dies with the record
			delete(s.lastSeen, id)
		}
	}
	// The HOMING has its own clock: it is forgotten only after
	// BindingTTLSeconds of silence, so a member that is evicted (or simply
	// quiet for longer than the 30 s record TTL) returns to the engine that
	// still holds its blocks — and to the engine the placement filter pinned.
	//
	// SEAT=PIN: there is no such clock. The binding lives exactly as long as the
	// seat, and exitLocked drops it, so a sweep here could only corrupt the
	// model (it would rewrite the pin of a member whose seat is live).
	if !s.cfg.seatPin() {
		bttl := time.Duration(s.cfg.BindingTTLSeconds * float64(time.Second))
		for id, seen := range s.bindingSeen {
			if now.Sub(seen) > bttl {
				delete(s.bindingSeen, id)
				delete(s.bindings, id)
				s.counters.rehomes++ // silent-member rehoming: the binding is released
			}
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
	for _, wp := range s.waiterPoolsLocked() {
		if _, ok := wp[id]; ok {
			return true
		}
	}
	return false
}

// deleteWaiterLocked removes a waiter from the pool it lives in (the engine map
// in member-stable mode, the one global pool under seat=pin).
func (s *Scheduler) deleteWaiterLocked(g *engineGroup, id string) {
	if pool := s.waitersLocked(g); pool != nil {
		delete(pool, id)
	}
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
	if !s.cfg.seatPin() {
		// The per-engine pools die with the engine (a waiter bound to it has no
		// other home in the member-stable model). Under seat=pin the pool is
		// GLOBAL and must SURVIVE a teardown: those heads can seat on a survivor.
		for id := range g.waiters {
			delete(g.waiters, id)
		}
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
	if s.cfg.seatPin() {
		// SEAT=PIN: the unseated demand lives in ONE global pool, so it belongs to
		// no engine (the per-engine Waiters above are 0 by construction). The
		// whole-policy number is still reported, so the run's liveness arithmetic
		// (waiting vs seats) reads the same in both models.
		snap.Waiters = len(s.gWaiters)
	}
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
