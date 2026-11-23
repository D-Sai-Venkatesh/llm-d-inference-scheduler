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
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
)

// cpuResidencyRunConfig is the T2L-q8 two-level arm's config plus the CPU-residency index: the
// exact shape the 2-replica run deploys (epp-cm/t2l-q8-cpuw-plugins.yaml).
const cpuResidencyRunConfig = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
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
- type: approx-prefix-cache-producer
  name: cpu-residency
  parameters:
    autoTune: false
    blockSizeTokens: 64
    lruCapacityPerServer: 87381
    maxPrefixTokensToMatch: 0
    maxPrefixBlocksToMatch: 0
    syncCrossReplicaState: false
- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95
    quantum: 8
    tierCapacityTokens: 5592405
    tierBudgetFraction: 0.95
    idleSeconds: 5
    inFlightCap: 64
    blockSizeTokens: 64
    inflightReapSeconds: 9300
    cpuResidencyProducer: cpu-residency
schedulingProfiles:
- name: default
  plugins:
flowControl:
  defaultRequestTTL: 0s
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

type noopRegistrar struct{}

func (noopRegistrar) Register(fwkdl.PendingRegistration) error { return nil }

// The run config loads through the real registry and loader, the producer is a request-control data
// producer (its Produce/PreRequest run on every request), and the group scheduler's startup gate finds
// it by name and switches CPU-residency placement on.
func TestCPUResidencyRunConfigLoads(t *testing.T) {
	loader.RegisterFeatureGate(flowcontrol.FeatureGate, false)
	NewRunner().registerInTreePlugins()
	raw, _, err := loader.LoadRawConfig([]byte(cpuResidencyRunConfig), logr.Discard())
	require.NoError(t, err)
	handle := fwkplugin.NewEppHandle(context.Background(), func() []k8stypes.NamespacedName { return nil },
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	_, err = loader.InstantiateAndConfigure(raw, handle, logr.Discard())
	require.NoError(t, err)

	prod := handle.Plugin("cpu-residency")
	require.NotNil(t, prod)
	_, isProducer := prod.(fwkrh.DataProducer)
	_, isPre := prod.(fwkrh.PreRequest)
	require.True(t, isProducer && isPre, "the index is fed by its own Produce + PreRequest")

	g, ok := handle.Plugin("group").(*groupsched.GroupScheduler)
	require.True(t, ok)
	require.NoError(t, g.RegisterDependencies(noopRegistrar{}), "the startup gate accepts the tier-sized index")
	dump, err := g.DumpState()
	require.NoError(t, err)
	var snap map[string]any
	require.NoError(t, json.Unmarshal(dump, &snap))
	require.Contains(t, snap, "residency", "residency placement is ON")
}
