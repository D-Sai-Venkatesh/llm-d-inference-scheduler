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
	"errors"
	"fmt"
	"sort"
	"strings"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
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

	// ---- the door switches (session-layer contract C1): nca2-only presence keys ----
	//
	// DoorBackfill is Part A: one head collects seats on one pod, everyone else backfills into what is
	// already free. SessionAware is Part B: the door and the seat order rank by session, read from the
	// SessionHeader request header. Both off is v2 exactly (C7).
	DoorBackfill  *bool   `json:"doorBackfill,omitempty"`
	SessionAware  *bool   `json:"sessionAware,omitempty"`
	SessionHeader *string `json:"sessionHeader,omitempty"`
	// BackfillReclaim lets backfill release eligible seats on a pod that is not the collection pod and has
	// no returner waiting, for an outsider admitted in the same step (LOGIC Amendment 2, C8/BF4). It
	// requires doorBackfill; off is the door without it.
	BackfillReclaim *bool `json:"backfillReclaim,omitempty"`

	// ---- LOGIC Amendment 3 (C9/C10): nca2-only presence keys, with or without the door ----
	//
	// ValveCPUWarm makes the cold valve count the tokens of a returner's CPU-line entry on its home pod as
	// warm, exactly like the GPU-resident prefix (V1). NoQuantum releases seats by silence only: quantum
	// completion stops making a seat rotation-eligible (Q0).
	ValveCPUWarm *bool `json:"valveCPUWarm,omitempty"`
	NoQuantum    *bool `json:"noQuantum,omitempty"`
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
	// C2: the door switches are nca2 keys too
	if c.DoorBackfill != nil {
		out = append(out, "doorBackfill")
	}
	if c.SessionAware != nil {
		out = append(out, "sessionAware")
	}
	if c.SessionHeader != nil {
		out = append(out, "sessionHeader")
	}
	if c.BackfillReclaim != nil {
		out = append(out, "backfillReclaim")
	}
	if c.ValveCPUWarm != nil {
		out = append(out, "valveCPUWarm")
	}
	if c.NoQuantum != nil {
		out = append(out, "noQuantum")
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
	n.AdmitLine, n.TrustLine = admit, trust
	n.AdmitLineTokens = int64(admit * float64(n.CPUTierTokens))
	n.TrustLineTokens = int64(trust * float64(n.CPUTierTokens))
	door, err := c.resolveDoor()
	if err != nil {
		return nil, err
	}
	n.door = door
	n.valveCPUWarm = c.ValveCPUWarm != nil && *c.ValveCPUWarm
	n.noQuantum = c.NoQuantum != nil && *c.NoQuantum
	return n, nil
}

// forbiddenSessionHeaders are the headers a session header may not be (C4, LOGIC Amendment 1 Q6): the
// fairness header and its deprecated alias OVERRIDE the derived thread id as the flow-control queue key
// (the director's derivation), which would turn v2 into a session-keyed scheduler; the two wire ids are
// per-request, not per-session.
var forbiddenSessionHeaders = []string{metadata.FlowFairnessIDKey, metadata.OldFlowFairnessIDKey,
	reqcommon.RequestIDHeaderKey, "x-correlation-id"}

// resolveDoor is C3-C5 and C8: nil unless doorBackfill is true (C7: the v2 decision runs), Part B and
// backfillReclaim only with Part A, a usable header, and no key that would sit in the config doing nothing.
func (c Config) resolveDoor() (*nca2Door, error) {
	door := c.DoorBackfill != nil && *c.DoorBackfill
	aware := c.SessionAware != nil && *c.SessionAware
	reclaim := c.BackfillReclaim != nil && *c.BackfillReclaim
	if aware && !door {
		return nil, errors.New("sessionAware requires doorBackfill: true (the session layer orders the door that Part A builds)")
	}
	if reclaim && !door {
		return nil, errors.New("backfillReclaim requires doorBackfill: true (it is a second chance of the door's backfill)")
	}
	if c.SessionHeader != nil && !aware {
		return nil, errors.New("sessionHeader is set while sessionAware is not true: the header would be read by nothing")
	}
	if !door {
		return nil, nil //nolint:nilnil // nil IS the answer: the door is off and v2's decision runs (C7)
	}
	d := &nca2Door{sessionAware: aware, backfillReclaim: reclaim}
	if aware {
		header := ""
		if c.SessionHeader != nil {
			header = strings.ToLower(strings.TrimSpace(*c.SessionHeader))
		}
		if header == "" {
			return nil, errors.New("sessionAware requires a non-empty sessionHeader (the request header that carries the session id)")
		}
		for _, bad := range forbiddenSessionHeaders {
			if header == bad {
				return nil, fmt.Errorf("sessionHeader may not be %q: it is a per-request id or the flow-control fairness key, not a session id", bad)
			}
		}
		d.header = header
		d.sessions = newNCA2Sessions()
	}
	return d, nil
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
