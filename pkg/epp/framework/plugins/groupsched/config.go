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
	// GroupBudgetFraction sizes each engine's TIER-1 (MEMBERSHIP) budget and its TIER-2 (RUNNING) cap
	// from that engine's own measured KV capacity: the running cap is always
	// fraction × num_gpu_blocks, and the membership budget is that number or, under
	// groupBudgetSource = max, the CPU tier's block count when it is the larger one.
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`

	// GroupBudgetSource is which tier sizes the GROUP (membership): "gpu" (the default, and today's
	// policy: fraction × num_gpu_blocks) or "max" (max(fraction × num_gpu_blocks,
	// cpuTierTokens ÷ blockSizeTokens)). The RUNNING cap is fraction × num_gpu_blocks either way.
	GroupBudgetSource string `json:"groupBudgetSource"`
	// CpuTierTokens is the engine's CPU KV-offload tier in TOKENS — a deployment fact, configured
	// exactly like the block size (vLLM reports no capacity for it that this policy may read). 0 =
	// no CPU tier, and then the group is the GPU's, as today.
	CpuTierTokens int64 `json:"cpuTierTokens"`
	// MaxNumSeqs is the engine's vLLM --max-num-seqs: the dispatch gate's sequence half. 0 = no
	// sequence gate (the memory half still applies).
	MaxNumSeqs int `json:"maxNumSeqs"`

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
}

// defaultConfig is what a config that omits every key gets: the shipped arm's own numbers.
func defaultConfig() Config {
	return Config{
		GroupBudgetFraction: 0.95,
		GroupBudgetSource:   budgetSourceGPU,
		CpuTierTokens:       0,
		MaxNumSeqs:          0,
		Quantum:             4,
		IdleSeconds:         20,
		InFlightCap:         64,
		BlockSizeTokens:     64,
		InflightReapSeconds: defaultInflightReapSeconds,
	}
}

// validate is the fail-closed startup gate for the knobs: every rule here refuses a config that would
// otherwise run a policy nobody configured.
func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.GroupBudgetSource != budgetSourceGPU && c.GroupBudgetSource != budgetSourceMax {
		return fmt.Errorf("groupBudgetSource must be %q or %q, got %q", budgetSourceGPU, budgetSourceMax, c.GroupBudgetSource)
	}
	if c.CpuTierTokens < 0 {
		return fmt.Errorf("cpuTierTokens must be >= 0 (it is the engine's CPU KV-offload tier in tokens), got %d", c.CpuTierTokens)
	}
	if c.MaxNumSeqs < 0 {
		return fmt.Errorf("maxNumSeqs must be >= 0 (0 = no sequence gate; else vLLM's --max-num-seqs), got %d", c.MaxNumSeqs)
	}
	// FAIL CLOSED BOTH WAYS: a source that names a tier the config does not size, and a tier the
	// source does not use, are both a configuration whose effect nobody would see.
	if c.GroupBudgetSource == budgetSourceMax && c.CpuTierTokens <= 0 {
		return fmt.Errorf("groupBudgetSource %q needs cpuTierTokens > 0: the group would be sized by a CPU tier that is not configured", budgetSourceMax)
	}
	if c.GroupBudgetSource == budgetSourceGPU && c.CpuTierTokens > 0 {
		return fmt.Errorf("cpuTierTokens is set but groupBudgetSource is %q, so the CPU tier would be ignored: set groupBudgetSource %q, or cpuTierTokens 0 for the GPU-sized group", budgetSourceGPU, budgetSourceMax)
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
	if c.InflightReapSeconds < 60 {
		return fmt.Errorf("inflightReapSeconds must be >= 60 (it is the belt over an unobserved termination, not a timeout: it must sit above the client's own request timeout; got %v)", c.InflightReapSeconds)
	}
	return nil
}

// defaultInflightReapSeconds is the reap bound a config that omits the key gets: above any client
// request timeout the deployment uses, with headroom, so the belt can only ever cut a leak.
const defaultInflightReapSeconds = 9300

// The two GROUP BUDGET SOURCES — which tier sizes membership. The RUNNING cap does not depend on
// them: it is the GPU's, always (A1, amended).
const (
	// budgetSourceGPU: membership = fraction × num_gpu_blocks. The default, and exactly today's
	// policy: with no CPU tier configured, the two tiers coincide.
	budgetSourceGPU = "gpu"
	// budgetSourceMax: membership = max(fraction × num_gpu_blocks, cpuTierTokens ÷ blockSizeTokens).
	// Not additive: the CPU tier's copy CONTAINS the GPU's (write-through), so the pool is the
	// larger capacity, never the sum.
	budgetSourceMax = "max"
)

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
