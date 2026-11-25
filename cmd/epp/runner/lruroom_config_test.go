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

package runner

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
)

// lruRoomRunConfig is the LRU-ROOM arm's plugins config (epp-cm/t2l-q8-lru-plugins.yaml of the experiment
// repo; backticks in its comments replaced): no residency producer, the group owns its tier index and
// runs the room gate.
const lruRoomRunConfig = `
# epp-cm/t2l-q8-lru-plugins.yaml — the GSCH-CORE chain on the LRU-ROOM build (gsch-2level-lru).
# Derived from runs/onepod-quartet-90m/group-c96-mixed-4h/epp-plugins.yaml (the banked group arm).
#
# The chain audit is docs/gsch-core/README.md §1 of the build repo
# (builds/epp-otel-fc-core/docs/gsch-core). Only what can be shown to be INERT, or
# what the policy actually reads, survives:
#
#   branch-identity   WHO    — it produces the derived branch id = the queue key
#   token-producer    (datalayer) — PreRequest fails closed without it; the block
#                       keys come from it; its backend is the vLLM render sidecar
#   group-scheduler   WHO (Pick) + WHERE (its own Filter) + the pool gate (its own
#                       SaturationDetector) — all three answer from one state
#
# REMOVED, on purpose (each would be an input to a decision we cannot prove inert):
#   agent-identity           never decides when branch-identity is present
#   token-load-scorer        it ORDERS a pass-through (unseated) request
#   inflight-load-producer   feeds the stock detector's credit; our detector never reads it
#   utilization-detector     replaced in the saturationDetector slot by 'group'
#   session-affinity-filter  the core pins with its OWN Filter; two pins = two authorities
#
# DETECTOR IDENTITY (the loader's trap): flowControl.saturationDetector MUST name
# 'group'. The loader FABRICATES the stock utilization-detector when the slot names
# nothing, and it injects whatever occupies the slot into every profile as a Filter
# — so a core arm that forgot this line would run the stock gate AND lose the pin.
# The plugin echoes which state it is in at startup and announces a DETECTOR
# CONFLICT if a second SaturationDetector is configured.
#
# The profile lists no plugin: the loader injects the group Filter (it occupies the
# detector slot) and the default max-score picker. Filters run before scorers and
# the picker regardless of list order (scheduler_profile.go addPlugins sorts by
# interface), so the injected position is not a correctness question.
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
# Identity (BRANCH_PLUGIN=1): derives the fairness id from the request
# BODY before admission.
#   SHADOW arm - the client also sends x-llm-d-inference-fairness-id, so the
#   explicit id is already set at assembly and wins: this plugin derives,
#   resolves and counts, and must show thread_explicit_override_total == the
#   request count (anything else was not a shadow run).
#   LIVE arm (stage 1+) - no fairness header, so the derived id IS the queue key.
- type: thread-identity
  parameters:
    blockSizeTokens: 64
    forkOnBusy: true
    stickyBusyGuard: true
    retryRule: false
    logRequests: true
    logHeaders: false
    markerTtlSeconds: 9300
    tokenProducer:
      modelName: Qwen/Qwen3-30B-A3B-Thinking-2507
      vllm:
        url: http://localhost:8000
- type: token-producer
  parameters:
    modelName: Qwen/Qwen3-30B-A3B-Thinking-2507
    vllm:
      url: http://localhost:8000
# Empty for a core arm (AFFINITY_FILTER=0): the policy's own Filter pins the seat.

- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95
    quantum: 8
    tierCapacityTokens: 5592405
    tierBudgetFraction: 0.95
    # LRU-ROOM build (gsch-2level-lru): a seat must not evict a seated thread's CPU-tier blocks.
    # Changed vs the banked group arm: idleSeconds 5 -> 15; cpuResidencyProducer + its producer
    # REMOVED (the group owns its tier index); the six keys below are new.
    idleSeconds: 15

    inFlightCap: 64
    # the engine's KV block size (must match --block-size on the decode)
    blockSizeTokens: 64
    inflightReapSeconds: 9300
    # the group's OWN CPU-tier model: one 87,381-block LRU per engine (5592405/64), touched at every
    # dispatch, recorded at every seat; it is also placement's residency read
    ownTierIndex: true
    # a head seats only when write + reserve <= room (free + non-seated blocks at the LRU tail, up to
    # the first seated block); the cold-entry valve is OFF; a room-held head may rotate ONE eligible
    # member per pass out, only when that member's blocks make room; last resort after 90 s held
    roomGate: true
    roomReserveTokens: 0
    roomHoldMaxSeconds: 90
    # shared admission queue: plain oldest-first (no newcomer priority)
    newcomersFirst: false
    # rotation victim (eligibility unchanged: quantum-complete or idle): the oldest last dispatch
    victimOldestDispatch: true


schedulingProfiles:
- name: default
  plugins:


flowControl:
  defaultRequestTTL: 0s
  # THE BAND-LIMIT LINE (config-identity, identical in every arm of a wave):
  # unbounded by intent, not tuned — see configs/_defaults.env FLOW_MAX_BYTES.
  # The GLOBAL pair and the BAND's own pair are both set because the band default
  # (1 GB / 5000) is the bound that actually applies to a queued request.
  maxBytes: "9223372036854775807"
  maxRequests: "9223372036854775807"
  priorityBands:
  - priority: 0
    fairnessPolicyRef: group
    maxBytes: "9223372036854775807"
    maxRequests: "9223372036854775807"
  saturationDetector:
    pluginRef: group
`

// The arm config loads through the real registry and loader with the strict factory, the startup gate
// accepts it (no residency producer to find), and the dump carries the room ledger.
func TestLRURoomRunConfigLoads(t *testing.T) {
	loader.RegisterFeatureGate(flowcontrol.FeatureGate, false)
	NewRunner().registerInTreePlugins()
	raw, _, err := loader.LoadRawConfig([]byte(lruRoomRunConfig), logr.Discard())
	require.NoError(t, err)
	handle := fwkplugin.NewEppHandle(context.Background(), func() []k8stypes.NamespacedName { return nil },
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	_, err = loader.InstantiateAndConfigure(raw, handle, logr.Discard())
	require.NoError(t, err)
	require.Nil(t, handle.Plugin("cpu-residency"), "the arm has no residency producer")
	g, ok := handle.Plugin("group").(*groupsched.GroupScheduler)
	require.True(t, ok)
	require.NoError(t, g.RegisterDependencies(noopRegistrar{}))
	dump, err := g.DumpState()
	require.NoError(t, err)
	var snap map[string]any
	require.NoError(t, json.Unmarshal(dump, &snap))
	require.Contains(t, snap, "room", "the room gate's ledger is in the dump")
	require.Contains(t, snap, "residency", "the own tier index is placement's residency read")
}
