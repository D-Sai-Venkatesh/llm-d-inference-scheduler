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

package datalayer

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
)

// TestGroupSchedulerLayerStaysFlowControl pins the LAYER of the group scheduler
// now that it also implements the scheduling Filter (E4a: seat=pin needs the
// policy itself to narrow placement to the member's seat engine).
//
// pluginToLayerExecutionOrder checks FairnessPolicy BEFORE Filter, so adding
// Filter must not move the plugin into SchedulingLayer. If it ever did, the
// DAG validator would reject the branch-identity data key it reads (measured
// live in wave 1, build 5e851b18: "producer branch-identity needs to be executed
// before consumer group/group-scheduler") and — worse — the policy would stop
// being constructed as the flow-control policy before admission runs.
func TestGroupSchedulerLayerStaysFlowControl(t *testing.T) {
	p, err := groupsched.Factory("group", nil, nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	if got := pluginToLayerExecutionOrder(p); got != FlowControlLayer {
		t.Fatalf("group scheduler layer = %d, want %d (FlowControlLayer): implementing the scheduling Filter must NOT move it out of the flow-control layer", got, FlowControlLayer)
	}
}
