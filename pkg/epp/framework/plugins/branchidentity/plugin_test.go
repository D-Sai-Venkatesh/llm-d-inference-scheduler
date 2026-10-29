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

package branchidentity

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// filler produces n deterministic bytes' worth of content (the default chunk is
// 256 canonical bytes, so this controls how many blocks a body spans).
func filler(seed byte, n int) string { return strings.Repeat(string(seed), n) }

func user(seed byte, n int) fwkrh.Message {
	return fwkrh.Message{Role: "user", Content: fwkrh.Content{Raw: filler(seed, n)}}
}

func assistant(seed byte, n int) fwkrh.Message {
	return fwkrh.Message{Role: "assistant", Content: fwkrh.Content{Raw: filler(seed, n)}}
}

func chatReq(msgs ...fwkrh.Message) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: "req",
		Headers:   map[string]string{},
		Body:      &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: msgs}},
	}
}

func newTestPlugin(t *testing.T) *Plugin {
	t.Helper()
	p, err := Factory("branch", nil, testutils.NewTestHandle(t.Context()))
	require.NoError(t, err)
	return p.(*Plugin)
}

// branchOf runs the header hook and returns the derived id and the request's
// fairness id.
func branchOf(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) (branchID, fairnessID string) {
	t.Helper()
	require.NoError(t, p.RequestHeader(context.Background(), req))
	id, ok := fwksched.ReadRequestAttribute[string](req, BranchIdentityKey)
	if !ok {
		return "", req.FairnessID
	}
	return id, req.FairnessID
}

// dispatchAndComplete models a full turn: the request leaves the queue and its
// response completes (the director forces the EndOfStream call even on abort).
func dispatchAndComplete(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) {
	t.Helper()
	require.NoError(t, p.PreRequest(context.Background(), req, nil))
	p.ResponseBody(context.Background(), req, &fwkrc.Response{
		EndOfStream:      true,
		TerminationCause: fwkrc.TerminationCauseNatural,
	}, nil)
}

// TestRequestHeader_SetsTheQueueKey: the derived branch id is the request's
// fairness id, so it is set before the request enqueues (it IS the queue key).
func TestRequestHeader_SetsTheQueueKey(t *testing.T) {
	p := newTestPlugin(t)

	id, fairness := branchOf(t, p, chatReq(user('a', 600)))
	require.NotEmpty(t, id)
	assert.Equal(t, id, fairness, "the branch id must be the flow-control queue key")
	assert.Len(t, id, idHexLen)
	assert.Regexp(t, "^[0-9a-f]+$", id, "branch ids are opaque, short hex")
}

// TestRequestHeader_ExplicitFairnessHeaderWins: an operator/client override still
// wins over the derivation (the derivation is the default, not a mandate).
func TestRequestHeader_ExplicitFairnessHeaderWins(t *testing.T) {
	p := newTestPlugin(t)
	req := chatReq(user('a', 600))
	req.FairnessID = "client-thread-1"

	id, fairness := branchOf(t, p, req)
	assert.NotEmpty(t, id, "the branch is still tracked and published")
	assert.Equal(t, "client-thread-1", fairness)
}

// TestLifecycle_ContinuationIsTheSameID: the queue key changes only when the
// branch does.
func TestLifecycle_ContinuationIsTheSameID(t *testing.T) {
	p := newTestPlugin(t)

	turn1 := chatReq(user('a', 600))
	id1, _ := branchOf(t, p, turn1)
	dispatchAndComplete(t, p, turn1)

	turn2 := chatReq(user('a', 600), assistant('b', 400), user('c', 300))
	id2, fairness2 := branchOf(t, p, turn2)
	assert.Equal(t, id1, id2)
	assert.Equal(t, id1, fairness2)

	// Determinism: the same body derives the same chain and the same id.
	idAgain, _ := branchOf(t, p, chatReq(user('a', 600)))
	assert.Equal(t, id1, idAgain)
}

// TestLifecycle_ForkWhileBusy: a sibling that extends a BUSY branch's tip (a
// sub-agent inheriting the parent's context) gets its own queue key, so it
// dispatches in parallel instead of queueing behind its parent.
func TestLifecycle_ForkWhileBusy(t *testing.T) {
	p := newTestPlugin(t)

	parent := chatReq(user('a', 600))
	idParent, _ := branchOf(t, p, parent)
	require.NoError(t, p.PreRequest(context.Background(), parent, nil)) // in flight, no response yet

	child := chatReq(user('a', 600), user('z', 300))
	idChild, _ := branchOf(t, p, child)
	assert.NotEqual(t, idParent, idChild, "a sibling must not share the parent's queue")
	assert.NotEmpty(t, idChild)
}

// TestH1_ReplyGuardIsLoadBalancingTheRealCase (H1, load-bearing): a request that
// extends a busy tip and begins with the tip's assistant reply is a proven
// continuation, so a stale busy flag cannot fragment the conversation. Stage 0
// measured 42 of 48 fork candidates to be exactly this case.
func TestH1_ReplyGuard(t *testing.T) {
	p := newTestPlugin(t)

	parent := chatReq(user('a', 600))
	idParent, _ := branchOf(t, p, parent)
	require.NoError(t, p.PreRequest(context.Background(), parent, nil)) // never settles

	// The next turn begins with the parent's assistant reply.
	continued := chatReq(user('a', 600), assistant('b', 400), user('c', 300))
	idNext, _ := branchOf(t, p, continued)
	assert.Equal(t, idParent, idNext, "content proves completion; the guard must override the busy flag")
}

// TestSessionIsAnAttributeNotAQueueKey: the client session is published under its
// own key (stage 2 reads it) and never becomes the branch id.
func TestSessionIsAnAttributeNotAQueueKey(t *testing.T) {
	p := newTestPlugin(t)

	req := chatReq(user('a', 600))
	req.Headers["x-claude-code-session-id"] = "sess-1"
	id, _ := branchOf(t, p, req)
	session, ok := fwksched.ReadRequestAttribute[string](req, BranchSessionKey)
	require.True(t, ok)
	assert.Equal(t, "sess-1", session)
	assert.NotEqual(t, "sess-1", id, "the session must never be the queue key")

	// The same body in another session is another branch (no look-alike merge).
	other := chatReq(user('a', 600))
	other.Headers["x-claude-code-session-id"] = "sess-2"
	idOther, _ := branchOf(t, p, other)
	assert.NotEqual(t, id, idOther)
}

// TestToolCalls_DoNotMergeConversations: two agentic loops that differ only in
// their assistant tool-call arguments must not collapse onto one branch — the
// survey's false-merge gap (assistant turns with empty content).
//
// NOTE: unvalidated by the campaign corpus — the WEKA trace contains no tool
// blocks, so this path is exercised only by unit tests. Treat it as a
// live-deployment risk, not a measured behaviour.
func TestToolCalls_DoNotMergeConversations(t *testing.T) {
	p := newTestPlugin(t)
	call := func(args string) fwkrh.Message {
		var v any
		require.NoError(t, json.Unmarshal([]byte(`{"id":"call_1","type":"function","function":{"name":"read","arguments":`+args+`}}`), &v))
		return fwkrh.Message{Role: "assistant", ToolCalls: []any{v}}
	}
	a := chatReq(user('a', 600), call(`"{\"path\":\"/tmp\"}"`))
	b := chatReq(user('a', 600), call(`"{\"path\":\"/etc\"}"`))

	idA, _ := branchOf(t, p, a)
	idB, _ := branchOf(t, p, b)
	assert.NotEqual(t, idA, idB)

	// And the tool call is part of the branch's identity for its own continuations.
	dispatchAndComplete(t, p, a)
	idA2, _ := branchOf(t, p, chatReq(user('a', 600), call(`"{\"path\":\"/tmp\"}"`), user('c', 300)))
	assert.Equal(t, idA, idA2)
}

// TestUnflattenableBody_FallsBackSilently: a body with no canonical stream leaves
// the fairness id alone (the director falls back) and never fails the request.
func TestUnflattenableBody_FallsBackSilently(t *testing.T) {
	p := newTestPlugin(t)
	req := &fwksched.InferenceRequest{RequestID: "r", Headers: map[string]string{}, Body: &fwkrh.InferenceRequestBody{}}

	id, fairness := branchOf(t, p, req)
	assert.Empty(t, id)
	assert.Empty(t, fairness, "no derivation → the existing fairness-id fallbacks apply")
	_, ok := fwksched.ReadRequestAttribute[string](req, BranchSessionKey)
	assert.False(t, ok)
}

// TestDormantReturn_KeepsTheID is the stage-0 retention rule at the plugin level:
// hours of silence do not change a returning thread's branch (the sweep only
// enforces the table cap).
func TestDormantReturn_KeepsTheID(t *testing.T) {
	p := newTestPlugin(t)

	turn1 := chatReq(user('a', 600))
	id1, _ := branchOf(t, p, turn1)
	dispatchAndComplete(t, p, turn1)

	p.sweepOnce(time.Now().Add(2 * time.Hour))

	turn2 := chatReq(user('a', 600), assistant('b', 400), user('c', 300))
	id2, _ := branchOf(t, p, turn2)
	assert.Equal(t, id1, id2, "a dormant thread is not a new thread")
}

// TestDumpState reports the branch table.
func TestDumpState(t *testing.T) {
	p := newTestPlugin(t)
	req := chatReq(user('a', 600))
	branchOf(t, p, req)
	require.NoError(t, p.PreRequest(context.Background(), req, nil))

	raw, err := p.DumpState()
	require.NoError(t, err)
	var snap map[string]any
	require.NoError(t, json.Unmarshal(raw, &snap))
	assert.EqualValues(t, 1, snap["branches"])
	assert.EqualValues(t, 1, snap["inFlight"])
	rows, ok := snap["rows"].([]any)
	require.True(t, ok)
	require.Len(t, rows, 1)
}

// TestFactory_Config: defaults resolve, invalid values fail closed.
func TestFactory_Config(t *testing.T) {
	p := newTestPlugin(t)
	assert.Equal(t, defaultBlockSizeTokens, p.idx.blockSize)
	assert.True(t, p.forkOnBusy)
	assert.True(t, p.sticky)
	assert.Equal(t, defaultMaxBranches, p.idx.maxBranches)
	assert.Equal(t, []string{
		"x-claude-code-session-id", "x-session-affinity", "session-id", "session_id",
	}, p.sessions)

	for name, params := range map[string]string{
		"negative block size": `{"blockSizeTokens":-1}`,
		"zero sweep":          `{"sweepSeconds":-5}`,
		"negative cap":        `{"maxBranches":-1}`,
		"bad json":            `{`,
	} {
		dec := json.NewDecoder(strings.NewReader(params))
		_, err := Factory("bad", dec, nil)
		assert.Error(t, err, name)
	}
}

// TestFactory_MetricsRegistered: the derivation's counters and the table gauge
// are exported.
func TestFactory_MetricsRegistered(t *testing.T) {
	handle := testutils.NewTestHandle(t.Context())
	built, err := Factory("branch", nil, handle)
	require.NoError(t, err)
	p := built.(*Plugin)

	req := chatReq(user('a', 600))
	require.NoError(t, p.RequestHeader(context.Background(), req))
	require.NoError(t, p.PreRequest(context.Background(), req, nil))

	reg, ok := handle.Metrics().(*prometheus.Registry)
	require.True(t, ok)
	families, err := reg.Gather()
	require.NoError(t, err)
	names := make([]string, 0, len(families))
	for _, f := range families {
		names = append(names, f.GetName())
	}
	for _, want := range []string{
		"llm_d_epp_branch_new_total",
		"llm_d_epp_branch_forks_total",
		"llm_d_epp_branch_records",
		"llm_d_epp_branch_flatten_ns",
		"llm_d_epp_branch_sticky_busy_total",
	} {
		assert.Contains(t, names, want)
	}
}
