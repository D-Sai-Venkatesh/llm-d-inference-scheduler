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
	"sync/atomic"
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
    executionBudget: "none"
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

// runnerStartedInProcess records that a full runner config has already been instantiated
// in THIS test binary. The plugins register their series into the plugin handle's
// registry, which is process-global, so a second full start in one process fails on a
// duplicate collector — a property of the harness (one EPP per process), not of a config.
// The offload arm's loader gate (below) therefore SKIPS when another test already started
// a runner, and is run on its own:
//
//	go test ./cmd/epp/runner/ -run TestGroupSchedulerCoreChainStartsWithTheCPUTier
//	go test ./cmd/epp/runner/ -run TestGroupSchedulerCoreChainStartsWithGpuPlusTierSizing
var runnerStartedInProcess atomic.Bool

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
	runnerStartedInProcess.Store(true)

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

// TestGroupSchedulerCoreChainStartsWithTheCPU Tier is the OFFLOAD arm's loader gate: the
// same chain the wave renders, with the tier selector (`entryOrder: "hrrn-tier"`) and the
// two deployment facts that size the tier shadow (cpuTierBytes, kvBlockBytes). It runs
// through the REAL startup path — YAML to strict decode to plugin construction to the
// DAG/layer-order validation — because the conformance test's strict decode is a mirror
// of that path, not the path itself. A key the loader rejects, or a value whose YAML type
// does not match the Go field, CrashLoops every arm carrying the block; this is where that
// is caught before a deploy.
//
// The pair of subtests matters as much as the happy path: the tier arm must START, and the
// key must be REJECTED when kvBlockBytes is missing for a configured tier (a tier with no
// block size is a silent zero-capacity shadow — a run that looks tier-aware and is not).
func TestGroupSchedulerCoreChainStartsWithTheCPUTier(t *testing.T) {
	if runnerStartedInProcess.Load() {
		t.Skip("a runner has already been instantiated in this process (plugin metrics are process-global): run this gate on its own with -run TestGroupSchedulerCoreChainStartsWithTheCPUTier")
	}
	tierCfg := strings.Replace(corePluginConfig, `entryOrder: "newcomer-first"`, `entryOrder: "hrrn-tier"`, 1)
	tierCfg = strings.Replace(tierCfg, "    inflightReapSeconds: 3700\n",
		"    inflightReapSeconds: 3700\n    cpuTierBytes: 549755813888\n    kvBlockBytes: 6291456\n", 1)
	require.NotEqual(t, corePluginConfig, tierCfg, "the fixture must actually carry the tier keys")

	start := func(t *testing.T, cfgText, pool string) error {
		t.Helper()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		opts := runserver.NewOptions()
		opts.ConfigText = cfgText
		opts.PoolName = pool
		opts.AllowExperimentalPlugins = true
		r := NewRunner()
		rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
		if err != nil {
			return err
		}
		ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
		_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
		return err
	}

	require.NoError(t, start(t, tierCfg, "core-tier"), "the offload arm's config must parse, decode and pass the plugin-DAG/layer-order validation")
	runnerStartedInProcess.Store(true)
	if !strings.Contains(tierCfg, "entryOrder: \"hrrn-tier\"") {
		t.Fatal("the fixture must keep the tier selector")
	}
}

// TestGroupSchedulerCoreChainStartsWithGpuPlusTierSizing is the TIER-SIZED budget arm's
// loader gate (docs/gsch-core/core-q4-offload-tierbudget.env): the same chain with
// `sizing: "gpu-plus-tier"` and the two tier facts it sizes the budget from. buildModules
// fails closed on an unknown module value, so a value the emitter can render and the plugin
// rejects is a CrashLoop on that arm only — the contract test (config_conformance_test.go)
// mirrors this path, and this runs it.
func TestGroupSchedulerCoreChainStartsWithGpuPlusTierSizing(t *testing.T) {
	if runnerStartedInProcess.Load() {
		t.Skip("a runner has already been instantiated in this process (plugin metrics are process-global): run this gate on its own with -run TestGroupSchedulerCoreChainStartsWithGpuPlusTierSizing")
	}
	tierCfg := strings.Replace(corePluginConfig, `sizing: "index"`, `sizing: "gpu-plus-tier"`, 1)
	tierCfg = strings.Replace(tierCfg, "    inflightReapSeconds: 3700\n",
		"    inflightReapSeconds: 3700\n    cpuTierBytes: 549755813888\n    kvBlockBytes: 6291456\n", 1)
	require.Contains(t, tierCfg, `sizing: "gpu-plus-tier"`, "the fixture must carry the sizing selector")
	require.Contains(t, tierCfg, "kvBlockBytes: 6291456", "the fixture must carry the tier facts the budget is sized from")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	opts := runserver.NewOptions()
	opts.ConfigText = tierCfg
	opts.PoolName = "core-tierbudget"
	opts.AllowExperimentalPlugins = true
	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err, "the tier-sized budget arm's config must parse, decode and pass the plugin-DAG/layer-order validation")
	runnerStartedInProcess.Store(true)
}

// TestGroupSchedulerCoreChainStartsWithTheExecutionBudget is the TWO-BUDGET arm's loader
// gate (docs/gsch-core/core-q4-offload-tierbudget-execbudget.env): the tier-sized seat
// budget AND the execution budget, i.e. the config the batch-2 pair runs on one side.
// `executionBudget` is a NEW key, so this is the path that must fail if the plugin's
// parameters stop decoding it — the emitter renders the value verbatim and the EPP decodes
// strictly, and buildModules fails closed on a value it does not know. Two checks in one:
// the arm starts, and the value survives the decode.
func TestGroupSchedulerCoreChainStartsWithTheExecutionBudget(t *testing.T) {
	if runnerStartedInProcess.Load() {
		t.Skip("a runner has already been instantiated in this process (plugin metrics are process-global): run this gate on its own with -run TestGroupSchedulerCoreChainStartsWithTheExecutionBudget")
	}
	cfg := strings.Replace(corePluginConfig, `sizing: "index"`, `sizing: "gpu-plus-tier"`, 1)
	cfg = strings.Replace(cfg, "    inflightReapSeconds: 3700\n",
		"    inflightReapSeconds: 3700\n    cpuTierBytes: 549755813888\n    kvBlockBytes: 6291456\n", 1)
	cfg = strings.Replace(cfg, `executionBudget: "none"`, `executionBudget: "execution-budget"`, 1)
	require.Contains(t, cfg, `executionBudget: "execution-budget"`, "the fixture must carry the dispatch selector")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	opts := runserver.NewOptions()
	opts.ConfigText = cfg
	opts.PoolName = "core-execbudget"
	opts.AllowExperimentalPlugins = true
	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err, "the two-budget arm's config must parse, decode and pass the plugin-DAG/layer-order validation")
	runnerStartedInProcess.Store(true)
}
