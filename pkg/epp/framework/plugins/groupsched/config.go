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

// THE PLUGIN'S PARAMETERS: what a deployment sets, what the decoder accepts, and what the startup
// echo prints back.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Config is the plugin's decoded parameters. One code path for one engine and for N: a single-pod
// stack is the N=1 case of the per-engine state, never a thread of its own.
type Config struct {
	// GroupBudgetFraction is the share of each engine's own measured GPU KV the group may use:
	// fraction × num_gpu_blocks. Without a CPU tier it sizes the SEATS; with one it sizes the turns
	// RUNNING on the GPU, and the seats are sized by the tier.
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`

	// TierCapacityTokens is each engine's CPU KV tier in tokens (the offloading connector's
	// cpu_bytes_to_use ÷ the model's KV bytes per token). 0 = no tier.
	TierCapacityTokens int64 `json:"tierCapacityTokens"`
	// TierBudgetFraction is the share of the tier the seats may hold.
	TierBudgetFraction float64 `json:"tierBudgetFraction"`

	// Quantum is Q — the fixed tenure, in COMPLETED busy periods, after which a resident becomes
	// rotation-eligible.
	Quantum int `json:"quantum"`

	// IdleSeconds is S: a resident with no queued turn and no activity for S is rotation-eligible
	// (it is not going to use its seat).
	IdleSeconds float64 `json:"idleSeconds"`
	// InFlightCap bounds how many of one thread's requests may be in flight at once.
	InFlightCap int `json:"inFlightCap"`
	// BlockSizeTokens is the engine's KV block size in tokens: the token processor's value, and it
	// must equal vLLM's --block-size.
	BlockSizeTokens int `json:"blockSizeTokens"`

	// InflightReapSeconds bounds the age of a tracked request before reconciliation drops it: the
	// belt for a termination that no callback reported (a record is written when a Pick releases
	// the turn and ended at its settle, its abandon, its thread's exit — so a record that
	// outlives this bound has no owner left).
	InflightReapSeconds float64 `json:"inflightReapSeconds"`

	// CPUResidencyProducer names an approx-prefix-cache-producer instance sized to the CPU tier
	// (autoTune false, lruCapacityPerServer = tierCapacityTokens / blockSizeTokens, both match caps
	// 0). With it, placement prefers the engine whose tier still holds the most of the head's prompt
	// (the fewest tokens to really prefill); "" = off, placement exactly as without it.
	CPUResidencyProducer string `json:"cpuResidencyProducer,omitempty"`

	// OwnTierIndex makes the group OWN its model of each engine's CPU tier: one block LRU per engine
	// (approximateprefix.TierLRU, capacity = tierCapacityTokens / blockSizeTokens), updated by the group
	// itself — at every dispatch (the turn's chain is touched, as vLLM's lookup touches it) and, with
	// roomGate, at every seat. It feeds placement exactly as cpuResidencyProducer does, and replaces it
	// (the two are exclusive). Needs a tier.
	OwnTierIndex bool `json:"ownTierIndex,omitempty"`
	// RoomGate admits a head only when the engine's tier can absorb what the head will WRITE without
	// evicting a seated thread's block: write (its prompt beyond the tier's resident prefix) +
	// roomReserveTokens <= room (free tier capacity + the run of non-seated blocks at the LRU's eviction
	// end). A seated head's whole chain is recorded in the index AT THE SEAT, so the next head sees the
	// room it leaves. It replaces the cold-entry valve. A room-held head may rotate ONE eligible member
	// out per pass, and only when that member's blocks would make room. Needs ownTierIndex.
	RoomGate bool `json:"roomGate,omitempty"`
	// RoomReserveTokens is held back from the room on every room test (0 = none).
	RoomReserveTokens int64 `json:"roomReserveTokens,omitempty"`
	// RoomHoldMaxSeconds is the room gate's optional last resort: a head the room has held this long
	// (measured from the first pass the room held it) is admitted anyway. 0 = OFF: a held head waits
	// until the room opens (a dispatch moves the tail blocker's chain off the tail, or the blocker turns
	// eligible and a room rotation moves it out; an empty group always seats).
	RoomHoldMaxSeconds float64 `json:"roomHoldMaxSeconds,omitempty"`
	// RoomDisplace (roomGate) counts the ELIGIBLE members' blocks (quantum-complete or idle, exactly as
	// the rotation reads them) as room too: the room walk stops only at the first block a NON-eligible
	// staying member holds. A head that fits is seated, and the eligible members whose blocks its
	// recorded write pushed out of the tier are rotated out at once (exit reason "displaced"): admit
	// first, then remove the members whose KV the newcomer displaced. The room-driven rotation is off.
	RoomDisplace bool `json:"roomDisplace,omitempty"`
	// QuantumDoneLast orders the turns INSIDE an engine's group while anyone waits outside (the shared
	// queue is not empty): the turns of members still within their quantum first (oldest first), the
	// turns of quantum-complete members only when no other member has a turn queued. With nobody waiting
	// the order is plain oldest-first, as without it. A held-back member's blocks are not refreshed, so
	// they sink to the tier's tail, where roomDisplace counts them as room.
	QuantumDoneLast bool `json:"quantumDoneLast,omitempty"`
	// QuantumLastTurn makes the quantum BINDING while anyone waits outside: at the first chunk of a
	// member's quantum-ending (or later) turn it is marked exiting — its prompt KV is in the GPU by then,
	// so its CPU-tier blocks no longer serve an in-group turn and count as room — and it leaves when the
	// turn ends (exit reason "quantum-last"). With nobody waiting nothing changes.
	QuantumLastTurn bool `json:"quantumLastTurn,omitempty"`
	// RoomVictims (roomGate) is the HEAD-DRIVEN binding quantum: when the queue's head is held for room,
	// quantum-complete members that are between turns leave the group for it — oldest dispatch first,
	// just enough that their blocks cover the head's write (counted per head, so one head never takes
	// more). They stop being served, so their blocks stop being refreshed and sink to the tail, where they
	// are room once the bottom blocker clears; then the head is admitted and the next head does the same.
	// One out per one in, at the admission's pace (exit reason "quantum-head").
	RoomVictims bool `json:"roomVictims,omitempty"`
	// FairRotation is ONE design for who leaves (fair.go). Eligible = quantum done (Q turns this stay, or
	// running the turn that completes them and past its first token) or idle. The BUDGET takes idle
	// members first, then quantum-done members by the LONGEST TIME IN THE GROUP since they joined; the
	// ROOM counts free + non-member + idle + removed blocks and, when a head is held, takes only the member
	// actually at the bottom, and only if it is eligible — otherwise the head waits. A removed member is
	// out at once (seat budget freed; an in-flight request finishes and counts on the GPU until it ends)
	// and queues behind the head. Needs roomGate + roomDisplace; exclusive with the older rotation knobs.
	FairRotation bool `json:"fairRotation,omitempty"`
	// CostFairRotation (fairRotation) changes ONE thing — the victim order among quantum-done members: the
	// highest SEAT TIME ÷ COST leaves first, cost = the blocks only it holds among staying members (what its
	// leaving frees and it will rebuild) + the staying members' blocks the head's write evicts from the tail
	// anyway − those it alone holds there (its leaving turns that thrash, or that room-blocking, into its own
	// loss). "How much you've had, per what sending you out costs." Idle members still go first. No constant:
	// every term is measured blocks and the ratio only ranks. Gated or not (roomGate) — the same formula.
	CostFairRotation bool `json:"costFairRotation,omitempty"`
	// ColdValveOff turns the cold-entry valve (the cold-start stagger) off without the room gate: admission is
	// the budget alone, as it already is under roomGate. The valve charged every token a head does not share
	// with a SEATED member as cold — a returner whose chain the CPU tier still holds included — and holding a
	// head only concentrates the waiting; vLLM itself schedules only what fits its KV and chunks prefills.
	ColdValveOff bool `json:"coldValveOff,omitempty"`
	// SizeWaitOrder is the order of the SEATED members' turns at the dispatch gate (about half of all waiting
	// happens there): a thread's FIRST turn first (newcomers first, inside the group too), then the highest
	// PROMPT TOKENS × TIME WAITED — big contexts go first so advanced sessions finish and the footprint shrinks,
	// and a waiting turn's priority keeps growing, so nobody starves. No constant: both are measured. Plain
	// oldest-first without it. Exclusive with quantumDoneLast (one in-group order).
	SizeWaitOrder bool `json:"sizeWaitOrder,omitempty"`
	// LeastRecentlyServed is THE STRICT RULE for the order of the SEATED members' turns at the dispatch gate
	// (protect.go): the member served longest ago (the router's own clock; the first turn of a stay first of
	// all; ties by arrival) is the candidate — unless its write would evict blocks of a protectable queued
	// member and serving that member first is a net save in blocks; then that member goes first. No constant,
	// no exchange rate; cache state decides only who is in a write's danger zone. Needs ownTierIndex. Exclusive
	// with the other in-group orders (sizeWaitOrder, quantumDoneLast).
	LeastRecentlyServed bool `json:"leastRecentlyServed,omitempty"`
	// NewcomersFirst orders the shared admission queue: true (the default) seats a thread's FIRST turn
	// before returning threads, then oldest first; false is plain oldest-first for every head.
	NewcomersFirst bool `json:"newcomersFirst"`
	// VictimOldestDispatch orders the rotation victims (eligibility unchanged: quantum-complete or
	// idle) by the oldest last dispatch — the oldest tier recency — instead of queued-turn-first then
	// earliest-seated.
	VictimOldestDispatch bool `json:"victimOldestDispatch,omitempty"`
}

// defaultConfig is what a config that omits every key gets: the shipped arm's own numbers.
func defaultConfig() Config {
	return Config{
		GroupBudgetFraction: 0.95,
		TierCapacityTokens:  0,
		TierBudgetFraction:  0.95,
		Quantum:             4,
		IdleSeconds:         20,
		InFlightCap:         64,
		BlockSizeTokens:     64,
		InflightReapSeconds: defaultInflightReapSeconds,
		RoomHoldMaxSeconds:  90,
		NewcomersFirst:      true,
	}
}

// validate is the fail-closed startup gate for the knobs: every rule here refuses a config that would
// otherwise run a policy nobody configured.
func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.TierCapacityTokens < 0 {
		return fmt.Errorf("tierCapacityTokens must be >= 0 (0 = no CPU tier), got %d", c.TierCapacityTokens)
	}
	if c.TierBudgetFraction <= 0 || c.TierBudgetFraction > 1 {
		return fmt.Errorf("tierBudgetFraction must be in (0,1], got %v", c.TierBudgetFraction)
	}
	if c.Quantum < 1 {
		return fmt.Errorf("quantum must be >= 1 (Q is the fixed tenure, in completed busy periods), got %d", c.Quantum)
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
	if c.CPUResidencyProducer != "" && c.TierCapacityTokens <= 0 {
		return fmt.Errorf("cpuResidencyProducer %q needs tierCapacityTokens: the residency index models each engine's CPU tier, and without a tier there is none to model", c.CPUResidencyProducer)
	}
	if c.OwnTierIndex && c.TierCapacityTokens <= 0 {
		return fmt.Errorf("ownTierIndex needs tierCapacityTokens: the index models each engine's CPU tier, and without a tier there is none to model")
	}
	if c.OwnTierIndex && c.CPUResidencyProducer != "" {
		return fmt.Errorf("ownTierIndex and cpuResidencyProducer %q are exclusive: both answer placement's residency read — keep one", c.CPUResidencyProducer)
	}
	if c.RoomGate && !c.OwnTierIndex {
		return fmt.Errorf("roomGate needs ownTierIndex: the gate records each seat in the group's own tier index, and reads the room from it")
	}
	if c.RoomReserveTokens < 0 {
		return fmt.Errorf("roomReserveTokens must be >= 0, got %d", c.RoomReserveTokens)
	}
	if c.FairRotation {
		if !c.OwnTierIndex {
			return fmt.Errorf("fairRotation needs ownTierIndex: its order and removal read the group's own tier index")
		}
		if c.RoomGate && !c.RoomDisplace {
			return fmt.Errorf("fairRotation with roomGate needs roomDisplace: its room counts idle members' blocks")
		}
		// ONE design, not a stack: the earlier experimental rotation switches cannot ride along
		if c.RoomVictims || c.QuantumLastTurn || c.QuantumDoneLast || c.RoomHoldMaxSeconds > 0 {
			return fmt.Errorf("fairRotation is exclusive with roomVictims, quantumLastTurn, quantumDoneLast and roomHoldMaxSeconds > 0")
		}
	}
	if c.LeastRecentlyServed && !c.OwnTierIndex {
		return fmt.Errorf("leastRecentlyServed needs ownTierIndex: its protection step reads each write's danger zone from the group's own tier index")
	}
	if n := btoi(c.SizeWaitOrder) + btoi(c.QuantumDoneLast) + btoi(c.LeastRecentlyServed); n > 1 {
		return fmt.Errorf("sizeWaitOrder, quantumDoneLast and leastRecentlyServed are in-group orders: choose one")
	}
	if c.CostFairRotation && !c.FairRotation {
		return fmt.Errorf("costFairRotation needs fairRotation: it is fair rotation's victim order")
	}
	if c.RoomVictims && !c.RoomGate {
		return fmt.Errorf("roomVictims needs roomGate: it removes members for a head the room gate holds")
	}
	if c.RoomDisplace && !c.RoomGate {
		return fmt.Errorf("roomDisplace needs roomGate: it changes what the room gate counts as room")
	}
	if c.RoomHoldMaxSeconds < 0 {
		return fmt.Errorf("roomHoldMaxSeconds must be >= 0 (0 = no last resort), got %v", c.RoomHoldMaxSeconds)
	}
	if c.InflightReapSeconds < 60 {
		return fmt.Errorf("inflightReapSeconds must be >= 60 (it is the belt over an unobserved termination, not a timeout: it must sit above the client's own request timeout; got %v)", c.InflightReapSeconds)
	}
	return nil
}

// defaultInflightReapSeconds is the reap bound a config that omits the key gets: above any client
// request timeout the deployment uses, with headroom, so the belt can only ever cut a leak.
const defaultInflightReapSeconds = 9300

// configEcho renders the config as sorted key=value pairs (the startup echo's form).
func configEcho(cfg Config) string {
	raw, err := json.Marshal(cfg)
	if err != nil {
		return ""
	}
	fields := map[string]any{}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return string(raw)
	}
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, fields[k]))
	}
	return strings.Join(parts, " ")
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
