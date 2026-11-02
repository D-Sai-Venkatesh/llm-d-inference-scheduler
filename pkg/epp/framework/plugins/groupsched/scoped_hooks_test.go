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

// THE SCOPED-HOOKS TEST (F13) — the test class whose absence let a silent breakage ship.
//
// The framework SCOPES the request a plugin receives at Filter, Scorer, PreRequest, the
// profile/scheduling points and every DataProducer (datalayer.ScopeInvocation /
// ScopeRequest). Under a scope an undeclared read RESOLVES AS ABSENT and an undeclared write
// is DROPPED (and fails PreRequest through violations.Write()). So a policy that reads its
// unit of work out of a request ATTRIBUTE sees "" — silently — and degrades to "every
// request anonymous, nothing seated, nothing starved, FCFS", which looks healthy in every
// outcome metric.
//
// That is exactly what shipped: branch identity PUBLISHES the branch at RequestHeader and the
// policy read it in every hook, while every existing test called the hooks with an UNSCOPED
// request — so nothing could fail. This file drives the hooks through the REAL scoping
// functions and fails on any violation, read or write, for every request shape a run can
// produce.
//
// It also pins where the branch comes from, because that is the invariant the fix rests on:
//
//	Pick         the flow controller's own queue item, UNSCOPED — the ONLY hook that reads the
//	             arrival attributes (branch identity wrote them at RequestHeader). The
//	             flow-control package applies no request scope (processor.go: fairnessP.Pick).
//	every other  the scheduler's OWN request record, resolved by request id — no request
//	hook         attribute is read there and none is written.

import (
	"context"
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	epmetadata "github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Extension-point labels for the scope. They only label the violation counter and the log;
// PreRequest uses the framework's own constant so it matches the director's exactly.
const (
	scopedFilterPoint   = "Filter"
	scopedResponsePoint = "ResponseBody"
	scopedAbandonPoint  = "RequestAbandoned"
	// scopedDefaultFairnessID is what the director sets when nothing named a flow: anonymous
	// traffic still gets a queue key, and it must NOT become a branch.
	scopedDefaultFairnessID = "default-flow"
	// scopedAgentSession is the agent identity the session header carries.
	scopedAgentSession = "sess-7"
)

// scopeViolations sums plugin_data_scope_violations_total for one plugin over BOTH access
// kinds. A rejected READ sets no error (it resolves as absent), so the counter is the only
// place a read violation lands — and a read violation is the defect this file exists for.
func scopeViolations(t *testing.T, pluginName string) float64 {
	t.Helper()
	eppmetrics.Register() // the violation counter is the read path's only signal
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	total := 0.0
	for _, family := range families {
		if family.GetName() != "llm_d_epp_plugin_data_scope_violations_total" {
			continue
		}
		for _, m := range family.GetMetric() {
			if labelValue(m, "plugin_name") != pluginName {
				continue
			}
			total += m.GetCounter().GetValue()
		}
	}
	return total
}

func labelValue(m *dto.Metric, name string) string {
	for _, l := range m.GetLabel() {
		if l.GetName() == name {
			return l.GetValue()
		}
	}
	return ""
}

// bareRequest is a request carrying nothing but what the wire provides: an id, a size, the
// tokens the render backend produced (the exact chain), and the client's headers. The
// identity plugins under test are the ones that write the attributes.
func bareRequest(id string, blocks uint64, headers map[string]string) *intRequest {
	size := blocks * 256 // the wire size a prompt of this length has at 4 bytes/token
	ir := &fwksched.InferenceRequest{RequestID: id, RequestSizeBytes: int(size), Headers: headers}
	ir.Body = &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		TokenizedRequest: &fwkrh.TokenizedRequest{
			Prompts: []fwkrh.PromptTokens{{TokenIDs: tokenRun(int(seedFor(id)%1000003)+1, int(blocks)*64)}},
		},
	}
	return &intRequest{id: id, size: size, infReq: ir}
}

// scopedCase is one request shape a run can produce.
type scopedCase struct {
	name string
	// build runs the REAL identity plugins on a fresh request, exactly where the director
	// runs them (RequestHeader, unscoped), and hands back the request the flow controller
	// would enqueue: its queue key is the request's flow identity.
	build func(t *testing.T, id string) *intRequest
	// anonymous marks the shape that must stay untracked.
	anonymous bool
}

// scopedCases are the shapes every arm must survive: branch identity (what the arms run), an
// explicit client fairness header (the operator override), no identity at all (pass-through),
// and agent identity as the ONLY source.
func scopedCases(t *testing.T) []scopedCase {
	t.Helper()
	bi := newBranchIdentity(t)
	ai, err := agentidentity.PluginFactory("agent", nil, nil)
	require.NoError(t, err)
	agent := ai.(*agentidentity.Plugin)
	// asIdentity finishes a request the way the director does after RequestHeader: a flow
	// identity always exists (branch → agent → default), and it is the queue key.
	asIdentity := func(req *intRequest) *intRequest {
		ir := req.InferenceRequest()
		if ir.FairnessID == "" {
			ir.FairnessID = scopedDefaultFairnessID
		}
		req.key = flowcontrol.FlowKey{ID: ir.FairnessID, Priority: 0}
		return req
	}
	return []scopedCase{
		{
			name: "branch-identity",
			build: func(t *testing.T, id string) *intRequest {
				req := bareRequest(id, 100, map[string]string{})
				require.NoError(t, bi.RequestHeader(context.Background(), req.InferenceRequest()))
				return asIdentity(req)
			},
		},
		{
			// THE EXPLICIT HEADER: the client's fairness id becomes the queue key while the
			// BRANCH stays branch identity's. The policy's unit is the branch — the wire
			// fairness id is a join key, never the identity — so the seat must follow the
			// branch even though the queue is named by the client.
			name: "explicit-fairness-header",
			build: func(t *testing.T, id string) *intRequest {
				req := bareRequest(id, 100, map[string]string{epmetadata.FlowFairnessIDKey: "client-tenant-7"})
				// the director reads the header into the request's flow identity BEFORE the
				// header processors run (director.go), so it is what the identity plugin sees
				req.InferenceRequest().FairnessID = "client-tenant-7"
				require.NoError(t, bi.RequestHeader(context.Background(), req.InferenceRequest()))
				require.Equal(t, "client-tenant-7", req.InferenceRequest().FairnessID,
					"an explicit header keeps the QUEUE KEY (the operator override)")
				minted, ok := fwksched.ReadRequestAttribute[string](req.InferenceRequest(), branchidentity.BranchIdentityKey)
				require.True(t, ok, "branch identity must publish its branch whatever the queue key is")
				require.NotEqual(t, "client-tenant-7", minted,
					"the BRANCH stays branch identity's: the two identities must not be confused")
				return asIdentity(req)
			},
		},
		{
			name:      "anonymous",
			anonymous: true,
			build: func(t *testing.T, id string) *intRequest {
				return asIdentity(bareRequest(id, 100, map[string]string{}))
			},
		},
		{
			// AGENT IDENTITY ONLY: no branch identity in the config. The identity gate
			// accepts it and the director derives the queue key from the agent id, so the
			// policy's unit must be the agent id too — otherwise every request would be
			// untracked while the startup gate reported a healthy wiring.
			name: "agent-identity-only",
			build: func(t *testing.T, id string) *intRequest {
				req := bareRequest(id, 100, map[string]string{agentidentity.ClaudeCodeSessionHeader: scopedAgentSession})
				require.NoError(t, agent.RequestHeader(context.Background(), req.InferenceRequest()))
				return asIdentity(req)
			},
		},
	}
}

// TestScopedHooksResolveTheBranchAndViolateNothing is THE F13 test: at N=1 and N=2, for every
// request shape, Pick (through the real flow controller, unscoped) releases the request and
// Filter / PreRequest / ResponseBody / Abandon — driven through the REAL scope — resolve its
// branch from the scheduler's record, take the right decision, and record NO violation.
func TestScopedHooksResolveTheBranchAndViolateNothing(t *testing.T) {
	for _, n := range []int{1, 2} {
		n := n
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			for _, tc := range scopedCases(t) {
				tc := tc
				t.Run(tc.name, func(t *testing.T) {
					gs := newScopedPlugin(t, n)
					fc, _ := newIntegrationHarness(t, gs)
					endpoints := scopedEndpoints(n)
					before := scopeViolations(t, "gs")

					req := tc.build(t, "s-1")

					// PICK — through the REAL flow controller, so the release happens exactly
					// where it happens in production (the flow-control path applies no scope).
					outcome, err := fc.EnqueueAndWait(context.Background(), req)
					require.NoError(t, err)
					require.Equal(t, fcTypes.QueueOutcomeDispatched, outcome, "the request must dispatch")
					branch, _ := gs.sched.RequestBranch(req.ID())

					if tc.anonymous {
						require.Empty(t, branch,
							"no identity source means no branch: the default flow key must not become a seated pseudo-branch")
					} else {
						require.NotEmpty(t, branch, "Pick must resolve the branch from the unscoped queue item")
					}
					if tc.name == "agent-identity-only" {
						require.Equal(t, scopedAgentSession, branch,
							"with no branch identity, the agent identity is the branch (as the director's queue key)")
					}

					// FILTER — ScopeInvocation, the framework's own call for a filter.
					scopedReq, scopedPods, violations := datalayer.ScopeInvocation(logr.Discard(),
						scopedFilterPoint, gs, req.InferenceRequest(), endpoints)
					require.NoError(t, violations.Write(), "the Filter wrote an undeclared attribute")
					kept := gs.Filter(context.Background(), scopedReq, scopedPods)
					if branch == "" {
						require.Len(t, kept, len(endpoints), "an untracked request carries no placement opinion")
					} else {
						require.Len(t, kept, 1, "a tracked request must be narrowed to its seat engine")
						require.Equal(t, endpointKey(kept[0]), gs.sched.bindingOf(branch),
							"the Filter must keep exactly the branch's seat engine")
					}

					// PRE-REQUEST — ScopeRequest, the director's own call.
					scopedReq, violations = datalayer.ScopeRequest(logr.Discard(),
						fwkrc.PreRequestExtensionPoint, gs, req.InferenceRequest())
					require.NoError(t, violations.Write(), "PreRequest wrote an undeclared attribute")
					require.NoError(t, gs.PreRequest(context.Background(), scopedReq, nil))
					if branch == "" {
						require.Equal(t, 0, gs.sched.TrackedRequests(), "nothing tracks an anonymous request")
					} else {
						require.Equal(t, 1, gs.sched.TrackedRequests(), "the released request is the branch's one record")
						require.Equal(t, 0, gs.sched.pendingForTest(branch),
							"the observed PreRequest closes the released window")
					}

					// RESPONSE BODY — scoped as well (the director does not scope this hook
					// today; the policy must be clean under the strictest reading anyway).
					for _, response := range []*fwkrc.Response{
						{StartOfStream: true},
						{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseNatural},
					} {
						scopedReq, violations = datalayer.ScopeRequest(logr.Discard(),
							scopedResponsePoint, gs, req.InferenceRequest())
						require.NoError(t, violations.Write(), "ResponseBody wrote an undeclared attribute")
						gs.ResponseBody(context.Background(), scopedReq, response, nil)
					}
					require.Equal(t, 0, gs.sched.TrackedRequests(), "the terminal callback ends the record")

					// ABANDON — a second request, released and never brought to a pod.
					req2 := tc.build(t, "s-2")
					outcome2, err2 := fc.EnqueueAndWait(context.Background(), req2)
					require.NoError(t, err2)
					require.Equal(t, fcTypes.QueueOutcomeDispatched, outcome2)
					scopedReq, violations = datalayer.ScopeRequest(logr.Discard(),
						scopedAbandonPoint, gs, req2.InferenceRequest())
					require.NoError(t, violations.Write(), "RequestAbandoned wrote an undeclared attribute")
					gs.RequestAbandoned(context.Background(), scopedReq)
					require.Equal(t, 0, gs.sched.TrackedRequests(),
						"the abandon ends the record it names (and invents none)")

					require.Equal(t, before, scopeViolations(t, "gs"),
						"a hook recorded a scope violation: an undeclared request attribute was read or written")
				})
			}
		})
	}
}

// TestBranchIdentityIsScopeClean drives the OTHER identity plugin we added through the same
// scoping: its PreRequest/ResponseBody reads must all be keys it Produces. (The reviewers
// reported it clean; pinning it here means a later read of somebody else's key fails.)
func TestBranchIdentityIsScopeClean(t *testing.T) {
	bi := newBranchIdentity(t)
	req := bareRequest("bi-1", 100, map[string]string{})
	require.NoError(t, bi.RequestHeader(context.Background(), req.InferenceRequest()))
	before := scopeViolations(t, bi.TypedName().Name)

	scopedReq, violations := datalayer.ScopeRequest(logr.Discard(), fwkrc.PreRequestExtensionPoint, bi, req.InferenceRequest())
	require.NoError(t, violations.Write(), "branch identity wrote an undeclared attribute at PreRequest")
	require.NoError(t, bi.PreRequest(context.Background(), scopedReq, nil))

	scopedReq, violations = datalayer.ScopeRequest(logr.Discard(), scopedResponsePoint, bi, req.InferenceRequest())
	require.NoError(t, violations.Write(), "branch identity wrote an undeclared attribute at ResponseBody")
	bi.ResponseBody(context.Background(), scopedReq, &fwkrc.Response{EndOfStream: true}, nil)

	require.Equal(t, before, scopeViolations(t, bi.TypedName().Name),
		"branch identity recorded a scope violation under a real scope")
}

// newScopedPlugin builds the plugin with n engines of measured capacity.
func newScopedPlugin(t *testing.T, n int) *GroupScheduler {
	t.Helper()
	plugin, err := Factory("gs", nil, nil)
	require.NoError(t, err)
	gs := plugin.(*GroupScheduler)
	for i := 0; i < n; i++ {
		require.NoError(t, gs.sched.SetNumBlocksFor(engineID(i), 64, 1000))
	}
	return gs
}

// scopedEndpoints are the Filter's candidates: one per engine the plugin was given, named the
// way the endpoint key names them.
func scopedEndpoints(n int) []fwksched.Endpoint {
	pods := make([]fwksched.Endpoint, 0, n)
	for i := 0; i < n; i++ {
		pods = append(pods, schedulerEndpoint(string(rune('a'+i))))
	}
	return pods
}

// bindingOf is the engine a branch is seated on ("" when unseated).
func (s *Scheduler) bindingOf(branch string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bindings[branch]
}

// pendingForTest is a branch's LIVE released-window count (0 once PreRequest closed it).
func (s *Scheduler) pendingForTest(branch string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groupOfLocked(branch); g != nil {
		if m := g.branches[branch]; m != nil {
			return m.pending
		}
	}
	return 0
}
