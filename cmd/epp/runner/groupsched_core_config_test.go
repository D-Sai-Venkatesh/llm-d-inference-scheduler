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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// corePluginConfig is the GSCH-CORE arm's plugin chain, exactly as the wave renders
// it (docs/gsch-core/README.md §1): branch-identity (the member) → token-producer (the
// block keys) → the group scheduler, which is the FAIRNESS POLICY, the FILTER (the
// seat pin) AND the SATURATION DETECTOR — one instance behind THREE references — and
// nothing else that can decide anything. The dual/triple reference is the shape that
// killed wave 1 live (the DAG validator rejected a consumer whose producer sat in a
// later layer; 75 green unit tests missed it because they never build the DAG), so it
// is built here through the SAME runner path the EPP uses at startup.
const corePluginConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: branch-identity
  parameters:
    blockSizeTokens: 64
    logRequests: true
- type: token-producer
  parameters:
    modelName: test-model
    vllm:
      url: http://localhost:8000
- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95
    quantum: 4
    idleSeconds: 20
    ttlSeconds: 30
    coldEntryCapTokens: 131072
    inFlightCap: 64
    blockSizeTokens: 64
    capacityLivenessSeconds: 120
    inflightReapSeconds: 3700
    sizing: "index"
    entryOrder: "newcomer-first"
    engineChoice: "most-free-budget"
    rotation: "quantum"
    victim: "valve"
    safety: "none"
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: group
flowControl:
  defaultRequestTTL: 0s
  priorityBands:
  - priority: 0
    fairnessPolicyRef: group
  saturationDetector:
    pluginRef: group
`

// TestGroupSchedulerDualReferenceStarts is the config-level DAG gate: one
// instance, referenced as both the fairness policy and a filter, and the EPP
// configuration phases complete.
func TestGroupSchedulerCoreChainStarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "core-plugins.yaml")
	require.NoError(t, os.WriteFile(path, []byte(corePluginConfig), 0o644))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = corePluginConfig
	opts.PoolName = "core-triple-reference"
	opts.AllowExperimentalPlugins = true

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err, "the E4a config must parse")

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	cfg, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err, "the E4a config must instantiate its plugins AND pass the plugin-DAG/ layer-order validation")

	// ONE INSTANCE: exactly one plugin carries the name both references use, and
	// it is the SAME object the priority band resolved.
	var found []fwkplugin.Plugin
	for _, p := range r.PluginHandle.GetAllPlugins() {
		if p.TypedName().Name == "group" {
			found = append(found, p)
		}
	}
	require.Len(t, found, 1, "the dual reference must resolve to ONE instance, not two (%d found)", len(found))
	named := r.PluginHandle.Plugin("group")
	require.NotNil(t, named)

	// both interfaces, one object — that is what makes the layer check pass and
	// the pin possible.
	_, isFilter := named.(fwksched.Filter)
	require.True(t, isFilter, "the group scheduler must implement the scheduling Filter")
	_, isPolicy := named.(fwkfc.FairnessPolicy)
	require.True(t, isPolicy, "the group scheduler must implement the FairnessPolicy")
	det, isDetector := named.(fwkfc.SaturationDetector)
	require.True(t, isDetector,
		"the group scheduler must implement the SaturationDetector: a core arm runs ONE decision plugin, and the pool gate is one of its three answers")
	// THE DETECTOR-IDENTITY GATE: the configured detector must be THIS plugin, and the
	// stock utilization-detector must not be in the config at all. (The loader
	// FABRICATES the stock detector when the slot names nothing — loader/defaults.go —
	// so "no error" is not enough: the identity has to be checked.)
	for _, p := range r.PluginHandle.GetAllPlugins() {
		require.NotEqual(t, "utilization-detector", p.TypedName().Type,
			"a core arm must not carry the stock roofline detector: exactly one detector answers the gate, and it is the group scheduler")
	}
	require.Equal(t, 0.0, det.Saturation(context.Background(), nil),
		"the default detector answer is NOT saturated: the seat budget is the admission control")

	require.NotNil(t, cfg)
	require.NotNil(t, cfg.FlowControlConfig, "flowControl must be configured")
	require.NotNil(t, cfg.FlowControlConfig.Registry, "the registry must be configured")
	// PRIORITY 0 is the band every campaign arm runs its traffic in, and it is
	// the one the arm configs override (flowControl.priorityBands). The
	// DefaultPriorityBand TEMPLATE stays the system default on purpose: it is the
	// template for newly-seen priorities, not the band priority-0 traffic lands in.
	band, ok := cfg.FlowControlConfig.Registry.PriorityBands[0]
	require.True(t, ok, "the priority-0 band must exist")
	require.NotNil(t, band.FairnessPolicy, "the priority-0 band must have resolved its fairness policy")
	require.Equal(t, "group", band.FairnessPolicy.TypedName().Name,
		"the priority-0 band must run the GROUP scheduler — this is the reference every campaign arm config writes, and a fallback to the system default here would silently run FCFS-ish fairness instead of the policy under test")

	// the DAG the EPP builds at startup: the group scheduler must appear in the
	// execution order EXACTLY ONCE (one instance behind two references), and the
	// order must be valid (no layer-order violation) — this is the check that
	// rejected wave 1's config live.
	order, err := datalayer.ValidateAndOrderDataDependencies(r.PluginHandle.GetAllPlugins())
	require.NoError(t, err, "the plugin DAG / layer order must be valid")
	var occurrences int
	for _, name := range order {
		if strings.Contains(name, "group") {
			occurrences++
		}
	}
	require.Equal(t, 1, occurrences, "the group scheduler must appear once in the execution order, got %v", order)
}
