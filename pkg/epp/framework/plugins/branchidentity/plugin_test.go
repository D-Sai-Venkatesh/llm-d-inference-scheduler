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
	"fmt"
	"io"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
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

// reqSeq makes every test request a distinct request id, because the branch's
// outstanding set is keyed by request id (two hooks of one request share a marker;
// two requests never do).
var reqSeq atomic.Uint64

func chatReq(msgs ...fwkrh.Message) *fwksched.InferenceRequest {
	return chatReqID(fmt.Sprintf("req-%d", reqSeq.Add(1)), msgs...)
}

func chatReqID(id string, msgs ...fwkrh.Message) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
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

	// A re-send of the CURRENT tip (a retry) stays on the branch.
	idAgain, _ := branchOf(t, p, chatReq(user('a', 600), assistant('b', 400), user('c', 300)))
	assert.Equal(t, id1, idAgain, "a retry of the tip continues the branch")

	// A re-send of an EARLIER turn does not: the tip has moved past it, and an id
	// held by a live branch is never reused (it used to overwrite and ghost it).
	idOld, _ := branchOf(t, p, chatReq(user('a', 600)))
	assert.NotEqual(t, id1, idOld)
	assert.Contains(t, idsOf(p), id1, "the original branch is still live")
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

// idsOf lists the live branch ids (test helper).
func idsOf(p *Plugin) []string {
	p.idx.mu.Lock()
	defer p.idx.mu.Unlock()
	ids := make([]string, 0, len(p.idx.branches))
	for id := range p.idx.branches {
		ids = append(ids, id)
	}
	return ids
}

// TestFactory_Config: defaults resolve, invalid values fail closed.
func TestFactory_Config(t *testing.T) {
	p := newTestPlugin(t)
	assert.Equal(t, defaultBlockSizeTokens, p.idx.blockSize)
	assert.True(t, p.forkOnBusy)
	assert.True(t, p.sticky)
	assert.Equal(t, defaultMaxBranches, p.idx.maxBranches)
	assert.Equal(t, time.Duration(defaultMarkerTTLSeconds)*time.Second, p.markerTTL,
		"the marker TTL is the client request timeout, not the flow queue's 5-minute GC")
	assert.True(t, p.retry, "the S2 retry rule is on by default")

	// The deprecated arm name still parses (an older deployment key).
	dec := json.NewDecoder(strings.NewReader(`{"queuedTtlSeconds":900}`))
	built, err := Factory("legacy", dec, nil)
	require.NoError(t, err)
	assert.Equal(t, 900*time.Second, built.(*Plugin).markerTTL)
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
		"llm_d_epp_branch_resolve_ns",
		"llm_d_epp_branch_sticky_busy_total",
		"llm_d_epp_branch_retry_total",
		"llm_d_epp_branch_explicit_override_total",
	} {
		assert.Contains(t, names, want)
	}
}

// TestIdentity_ToolSetChurnMidThreadContinues (B4): the identity stream excludes
// the tool DEFINITIONS, so an agent that swaps its tool set between turns (claude_code
// does exactly this when it hands a sub-task to a `Task` tool) still continues its
// branch. Stage 0B measured 32 of 33 Exgentic mid-thread "splits" to be this class,
// and a per-step tool shortlist would make it unbounded — the unsafe direction.
func TestIdentity_ToolSetChurnMidThreadContinues(t *testing.T) {
	p := newTestPlugin(t)

	turn1 := chatReq(user('a', 600))
	turn1.Body.ChatCompletions.Tools = []any{map[string]any{"type": "function", "function": map[string]any{"name": "read"}}}
	id1, _ := branchOf(t, p, turn1)
	dispatchAndComplete(t, p, turn1)

	// Same conversation, a completely different tool set.
	turn2 := chatReq(user('a', 600), assistant('b', 400), user('c', 300))
	turn2.Body.ChatCompletions.Tools = []any{map[string]any{"type": "function", "function": map[string]any{"name": "Task"}}}
	id2, _ := branchOf(t, p, turn2)
	assert.Equal(t, id1, id2, "a tool-set change must cost a cache miss, never an identity change")

	// And with the tool set REMOVED entirely.
	turn3 := chatReq(user('a', 600), assistant('b', 400), user('c', 300), assistant('d', 100), user('e', 100))
	id3, _ := branchOf(t, p, turn3)
	assert.Equal(t, id1, id3)
}

// TestIdentity_DroppedThinkingContinues (B4): Anthropic clients only keep the LAST
// assistant turn's signed thinking — earlier turns drop theirs — so thinking blocks
// must not be part of the identity stream, or every turn would look like a new
// conversation.
func TestIdentity_DroppedThinkingContinues(t *testing.T) {
	p := newTestPlugin(t)
	filler600 := filler('a', 600)

	withThinking := &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{Raw: "be terse"},
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: filler600}},
			{Role: "assistant", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
				{Type: "thinking", Thinking: "let me think about this at length"},
				{Type: "text", Text: filler('b', 400)},
			}}},
		},
	}}
	turn1 := &fwksched.InferenceRequest{RequestID: "m1", Headers: map[string]string{}, Body: withThinking}
	id1, _ := branchOf(t, p, turn1)
	dispatchAndComplete(t, p, turn1)

	// The next turn drops the earlier assistant's thinking block, as the API does.
	next := &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{Raw: "be terse"},
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: filler600}},
			{Role: "assistant", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
				{Type: "text", Text: filler('b', 400)},
			}}},
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: filler('c', 300)}},
		},
	}}
	id2, _ := branchOf(t, p, &fwksched.InferenceRequest{RequestID: "m2", Headers: map[string]string{}, Body: next})
	assert.Equal(t, id1, id2, "dropped thinking must not fragment the conversation")
}

// TestExplicitFairnessOverrideIsCounted (B1): the derivation is silently bypassed
// when a client (or a harness flag) sets the fairness id, so the bypass is counted —
// it is how a run proves the stage-1 harness did not send one.
func TestExplicitFairnessOverrideIsCounted(t *testing.T) {
	p := newTestPlugin(t)
	req := chatReq(user('a', 600))
	req.FairnessID = "client-thread-1"
	branchOf(t, p, req)
	require.NotNil(t, p.metrics.overridden)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.overridden))

	plain := chatReq(user('b', 600))
	branchOf(t, p, plain)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.overridden), "no override, no count")
}

// TestRetryIsCounted (S2): a retry of a still-queued turn continues the branch and
// is visible as its own counter (stage 1 reads it to see how much of the fork rate
// was retries).
func TestRetryIsContainedAndCounted(t *testing.T) {
	p := newTestPlugin(t)

	turn1 := chatReqID("dup", user('a', 600))
	id1, _ := branchOf(t, p, turn1) // arrival only: still queued

	retry := chatReqID("dup-2", user('a', 600))
	id2, _ := branchOf(t, p, retry)
	assert.Equal(t, id1, id2, "a byte-identical duplicate of a queued turn is a retry")
	require.NotNil(t, p.metrics.retried)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.retried))
}

// TestLogRequests (the shadow/stage-1 instrument): with logRequests on, ONE JSON
// line per request is printed, carrying the scope (the client session header), the
// DERIVED id and the outcome — the per-request pair a shadow arm cannot get from a
// span, because there the span's fairness id is the client's explicit header. With
// the parameter off (the default) nothing is printed, so a production build is
// untouched.
func TestLogRequests(t *testing.T) {
	build := func(t *testing.T, logOn bool) *Plugin {
		t.Helper()
		p, err := Factory("branch", json.NewDecoder(strings.NewReader(
			`{"sessionHeaders":["x-session-scope"],"logRequests":`+fmt.Sprint(logOn)+`}`)),
			testutils.NewTestHandle(t.Context()))
		require.NoError(t, err)
		return p.(*Plugin)
	}

	capture := func(t *testing.T, fn func()) string {
		t.Helper()
		r, w, err := os.Pipe()
		require.NoError(t, err)
		orig := os.Stdout
		os.Stdout = w
		defer func() { os.Stdout = orig }()
		done := make(chan string, 1)
		go func() {
			b, _ := io.ReadAll(r)
			done <- string(b)
		}()
		fn()
		require.NoError(t, w.Close())
		return <-done
	}

	req := func() *fwksched.InferenceRequest {
		r := chatReq(user('a', 600))
		r.Headers["x-session-scope"] = "thread-7"
		return r
	}

	on := build(t, true)
	var id string
	out := capture(t, func() {
		id, _ = branchOf(t, on, req())
	})
	require.Contains(t, out, "branch-identity: request ")
	line := strings.TrimPrefix(strings.TrimSpace(out), "branch-identity: request ")
	var rec struct {
		Seq      uint64 `json:"seq"`
		Scope    string `json:"scope"`
		Derived  string `json:"derived"`
		Outcome  string `json:"outcome"`
		Explicit bool   `json:"explicit"`
		Blocks   int    `json:"blocks"`
	}
	require.NoError(t, json.Unmarshal([]byte(line), &rec), "the instrument line must be one JSON object: %q", line)
	assert.Equal(t, "thread-7", rec.Scope)
	assert.Equal(t, id, rec.Derived, "the logged derived id is the one the plugin resolved")
	assert.Equal(t, "new", rec.Outcome)
	assert.False(t, rec.Explicit)
	assert.EqualValues(t, 1, rec.Seq)

	off := build(t, false)
	out = capture(t, func() { branchOf(t, off, req()) })
	assert.Empty(t, out, "logRequests=false must print nothing")
}

// TestLogHeadersKeepResolveOutcomeIdentical (stage-1 P1): the logHeaders instrument
// copies the client's x-correlation-id / x-request-id into the per-request line so an
// UNSCOPED arm (no session header → empty scope) can still score the derived-vs-thread
// confusion matrix per request. It must be a pure OBSERVER: the same request sequence
// fed to a plugin with the instrument on and one with it off must resolve to the same
// derived ids, the same outcomes and the same fairness ids.
func TestLogHeadersKeepResolveOutcomeIdentical(t *testing.T) {
	build := func(t *testing.T, logHeaders bool) *Plugin {
		t.Helper()
		p, err := Factory("branch", json.NewDecoder(strings.NewReader(
			`{"logRequests":true,"logHeaders":`+fmt.Sprint(logHeaders)+`}`)),
			testutils.NewTestHandle(t.Context()))
		require.NoError(t, err)
		return p.(*Plugin)
	}

	// Three turns of one thread plus a sibling extending the same prefix: the
	// sequence exercises a "new", a "matched" and a fork decision.
	seq := func() []*fwksched.InferenceRequest {
		mk := func(turn int) *fwksched.InferenceRequest {
			msgs := []fwkrh.Message{user('a', 600)}
			for i := 1; i < turn; i++ {
				msgs = append(msgs, assistant('b', 300), user('c', 300))
			}
			r := chatReq(msgs...)
			r.Headers["x-correlation-id"] = "thread-42"
			r.Headers["x-request-id"] = fmt.Sprintf("req-%d", turn)
			return r
		}
		return []*fwksched.InferenceRequest{mk(1), mk(2), mk(3), mk(2)}
	}

	run := func(t *testing.T, logHeaders bool) (ids, fids []string, logged string) {
		t.Helper()
		r, w, err := os.Pipe()
		require.NoError(t, err)
		orig := os.Stdout
		os.Stdout = w
		defer func() { os.Stdout = orig }()
		done := make(chan string, 1)
		go func() {
			b, _ := io.ReadAll(r)
			done <- string(b)
		}()
		p := build(t, logHeaders)
		for _, req := range seq() {
			id, fid := branchOf(t, p, req)
			ids = append(ids, id)
			fids = append(fids, fid)
			// clear the arrival marker so the next turn sees a free tip
			p.idx.Settle(id, markerOf(req), p.now())
		}
		require.NoError(t, w.Close())
		return ids, fids, <-done
	}

	offIDs, offFIDs, offOut := run(t, false)
	onIDs, onFIDs, onOut := run(t, true)

	assert.Equal(t, offIDs, onIDs, "derived ids must not depend on the instrument")
	assert.Equal(t, offFIDs, onFIDs, "fairness ids must not depend on the instrument")
	// the derived id IS the fairness id in an arm with no explicit header
	for i := range onIDs {
		assert.Equal(t, onIDs[i], onFIDs[i], "unscoped run: derived id is the queue key")
	}
	assert.NotContains(t, offOut, "\"headers\"", "logHeaders=false must not copy headers")
	assert.Contains(t, onOut, `"x-correlation-id":"thread-42"`)
	assert.Contains(t, onOut, `"x-request-id":"req-1"`)
}

// TestLogHeaderNamesConfigured (WAVE-1 item 4): logHeaderNames EXTENDS the
// whitelist with a harness's own markers without touching anything else — the
// extras are only ever read by the log line, never by the identity, the scope or
// a matching decision.
func TestLogHeaderNamesConfigured(t *testing.T) {
	build := func(t *testing.T) *Plugin {
		t.Helper()
		p, err := Factory("branch", json.NewDecoder(strings.NewReader(
			`{"logRequests":true,"logHeaders":true,"logHeaderNames":["X-Turn-Id","x-correlation-id",""]}`)),
			testutils.NewTestHandle(t.Context()))
		require.NoError(t, err)
		return p.(*Plugin)
	}
	require.Equal(t, []string{"x-correlation-id", "x-request-id", "x-turn-id"},
		build(t).logHeadersList, "extras must be lowercased, de-duplicated and appended")

	p := build(t) // build BEFORE the capture: Factory echoes its config to stdout
	r, w, err := os.Pipe()
	require.NoError(t, err)
	orig := os.Stdout
	os.Stdout = w
	defer func() { os.Stdout = orig }()
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	req := chatReq(user('a', 600))
	req.Headers["x-correlation-id"] = "thread-9"
	req.Headers["x-turn-id"] = "turn-3"
	id, _ := branchOf(t, p, req)
	require.NoError(t, w.Close())
	out := <-done

	line := strings.TrimPrefix(strings.TrimSpace(out), "branch-identity: request ")
	var rec struct {
		Derived string            `json:"derived"`
		Headers map[string]string `json:"headers"`
	}
	require.NoError(t, json.Unmarshal([]byte(line), &rec))
	assert.Equal(t, id, rec.Derived)
	assert.Equal(t, "thread-9", rec.Headers["x-correlation-id"])
	assert.Equal(t, "turn-3", rec.Headers["x-turn-id"], "the configured extra must be copied")
}
