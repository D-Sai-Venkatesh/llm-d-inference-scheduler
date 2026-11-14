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
	// GroupBudgetFraction sizes each engine's SEAT BUDGET from that engine's own measured KV
	// capacity: budgetBlocks = fraction × num_gpu_blocks.
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`

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

	// ---- the nca2 selector (nca2 spec §1) ----
	//
	// The nca2 keys decode as POINTERS: presence is the fact the fail-closed rule needs (an nca2
	// key without mode: nca2 refuses the startup), and a value that was never set must not be
	// distinguishable from a zero. They are omitted from the echo while nil, so a config that
	// names none of them echoes exactly the shipped parameter set.
	//
	// Mode selects the policy: "" (the absent value) is the shipped behaviour, "nca2" is the
	// CPU-tier admission of the nca2 spec. Any other value is refused.
	Mode string `json:"mode,omitempty"`
	// CPUTierTokens is C: the pod's CPU KV tier, in tokens.
	CPUTierTokens *int64 `json:"cpuTierTokens,omitempty"`
	// AdmitLine and TrustLine are the two fractions of C (the same number for now): the admission
	// line L_a and the prediction line L_t.
	AdmitLine *float64 `json:"admitLine,omitempty"`
	TrustLine *float64 `json:"trustLine,omitempty"`
	// MaxInflightSeconds bounds one request's in-flight age for the ACTIVITY test: an older record
	// is treated as leaked and stops holding its thread active.
	MaxInflightSeconds *float64 `json:"maxInflightSeconds,omitempty"`
	// LostRequestSeconds is the released-window liveness bound (the base's expireWindows /
	// windowLive): a request that never reached a pod is lost after this long.
	LostRequestSeconds *float64 `json:"lostRequestSeconds,omitempty"`
}

// defaultConfig is what a config that omits every key gets: the shipped arm's own numbers.
func defaultConfig() Config {
	return Config{
		GroupBudgetFraction: 0.95,
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
	// the selector's own gate: an unknown mode, or an nca2 key without mode: nca2, refuses the
	// startup here (the same call NewScheduler makes: one rule, one place)
	if _, err := c.resolveNCA2(); err != nil {
		return err
	}
	return nil
}

// defaultInflightReapSeconds is the reap bound a config that omits the key gets: above any client
// request timeout the deployment uses, with headroom, so the belt can only ever cut a leak.
const defaultInflightReapSeconds = 9300

// The policy selector: OFF is the shipped behaviour and the absent value, and "nca2" is the only
// mode that changes anything.
const (
	modeOff  = ""
	modeNCA2 = "nca2"
)

// The nca2 defaults: the numbers the spec names, so a deployment that sets only mode: nca2 and a
// tier size runs the policy the spec describes.
const (
	defaultCPUTierTokens      = 5592405
	defaultAdmitLine          = 0.95
	defaultTrustLine          = 0.95
	defaultMaxInflightSeconds = 1200
	defaultLostRequestSeconds = 300
)

// nca2KeyNames lists the parameters that only mean something in nca2, in the order the refusal
// names them: a config that carries one without mode: nca2 is refused rather than run.
func (c Config) nca2KeyNames() []string {
	var out []string
	if c.CPUTierTokens != nil {
		out = append(out, "cpuTierTokens")
	}
	if c.AdmitLine != nil {
		out = append(out, "admitLine")
	}
	if c.TrustLine != nil {
		out = append(out, "trustLine")
	}
	if c.MaxInflightSeconds != nil {
		out = append(out, "maxInflightSeconds")
	}
	if c.LostRequestSeconds != nil {
		out = append(out, "lostRequestSeconds")
	}
	return out
}

// resolveNCA2 is the ONE place the selector becomes policy: nil for the shipped behaviour, the
// resolved numbers for nca2, and an error for everything else (fail closed — an unknown mode, or
// an nca2 key that would otherwise sit in the config looking effective while the shipped policy
// runs underneath it).
func (c Config) resolveNCA2() (*nca2Policy, error) {
	keys := c.nca2KeyNames()
	if c.Mode != modeNCA2 {
		if c.Mode != modeOff {
			return nil, fmt.Errorf("unknown mode %q: the modes are %q (the absent value too: the shipped behaviour) and %q", c.Mode, modeOff, modeNCA2)
		}
		if len(keys) > 0 {
			return nil, fmt.Errorf("the nca2 parameters (%s) require mode: %q — without it the shipped behaviour runs and these would look effective while changing nothing", strings.Join(keys, ", "), modeNCA2)
		}
		return nil, nil
	}
	n := &nca2Policy{
		CPUTierTokens:      defaultCPUTierTokens,
		MaxInflightSeconds: defaultMaxInflightSeconds,
		LostRequestSeconds: defaultLostRequestSeconds,
	}
	admit, trust := defaultAdmitLine, defaultTrustLine
	if c.CPUTierTokens != nil {
		n.CPUTierTokens = *c.CPUTierTokens
	}
	if c.AdmitLine != nil {
		admit = *c.AdmitLine
	}
	if c.TrustLine != nil {
		trust = *c.TrustLine
	}
	if c.MaxInflightSeconds != nil {
		n.MaxInflightSeconds = *c.MaxInflightSeconds
	}
	if c.LostRequestSeconds != nil {
		n.LostRequestSeconds = *c.LostRequestSeconds
	}
	if n.CPUTierTokens <= 0 {
		return nil, fmt.Errorf("cpuTierTokens must be > 0 (it is the pod's CPU KV tier in tokens), got %d", n.CPUTierTokens)
	}
	if admit <= 0 || admit > 1 {
		return nil, fmt.Errorf("admitLine must be in (0,1] (a fraction of cpuTierTokens), got %v", admit)
	}
	if trust <= 0 || trust > 1 {
		return nil, fmt.Errorf("trustLine must be in (0,1] (a fraction of cpuTierTokens), got %v", trust)
	}
	if n.MaxInflightSeconds <= 0 {
		return nil, fmt.Errorf("maxInflightSeconds must be > 0 (it is the in-flight age after which a record is leaked), got %v", n.MaxInflightSeconds)
	}
	if n.LostRequestSeconds <= 0 {
		return nil, fmt.Errorf("lostRequestSeconds must be > 0 (it is the released-window liveness bound), got %v", n.LostRequestSeconds)
	}
	n.AdmitLineTokens = int64(admit * float64(n.CPUTierTokens))
	n.TrustLineTokens = int64(trust * float64(n.CPUTierTokens))
	return n, nil
}

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
