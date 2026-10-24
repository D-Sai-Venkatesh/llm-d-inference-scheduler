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

package ringwindow

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// testConfig uses round numbers: capacity 480 tokens, theta 0.8 → budget
// 384; body estimates at 4 bytes per token.
func testConfig() Config {
	cfg := defaultConfig()
	cfg.CapacityTokens = 480
	return cfg
}

func fwkTypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PluginType, Name: "test"}
}

func newTestPlugin(cfg Config) *RingWindow {
	mgr := newRingManager(cfg)
	return &RingWindow{
		typedName: fwkTypedName(),
		cfg:       cfg,
		mgr:       mgr,
		metrics:   newRingMetrics(mgr),
	}
}

func newRequest(sessionID string, sizeBytes int) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		RequestID:        "req-" + sessionID,
		RequestSizeBytes: sizeBytes,
	}
	if sessionID != "" {
		req.PutAttribute(agentidentity.AgentIdentityKey, sessionID)
	}
	return req
}

// makeQueue builds a one-item queue whose head carries the given session,
// enqueue age and queue-wait budget (the request's EffectiveTTL).
func makeQueue(session string, enqueueAgo, ttl time.Duration, bytes int) *fwkfcmocks.MockFlowQueueAccessor {
	req := fwkfcmocks.NewMockFlowControlRequest(uint64(bytes), "req-"+session, fwkfc.FlowKey{ID: session})
	req.InferenceRequestV = newRequest(session, bytes)
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: fwkfc.FlowKey{ID: session},
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     time.Now().Add(-enqueueAgo),
			EffectiveTTLV:    ttl,
			OriginalRequestV: req,
		},
	}
}

func bandOf(queues ...fwkfc.FlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(callback func(flow fwkfc.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !callback(q) {
					return
				}
			}
		},
	}
}

func pick(t *testing.T, r *RingWindow, queues ...fwkfc.FlowQueueAccessor) fwkfc.FlowQueueAccessor {
	t.Helper()
	got, err := r.Pick(context.Background(), bandOf(queues...))
	require.NoError(t, err)
	return got
}

func pickID(t *testing.T, r *RingWindow, queues ...fwkfc.FlowQueueAccessor) string {
	t.Helper()
	q := pick(t, r, queues...)
	if q == nil {
		return ""
	}
	return q.FlowKey().ID
}

func isMember(r *RingWindow, id string) bool {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	s, ok := r.mgr.sessions[id]
	return ok && s.member
}

func memberIDs(r *RingWindow) []string {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	var out []string
	for _, s := range r.mgr.membersLocked() {
		out = append(out, s.id)
	}
	return out
}

// setFinish backdates a session's last turn completion (the release order).
func setFinish(r *RingWindow, id string, ago time.Duration) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.sessions[id].lastFinish = time.Now().Add(-ago)
}

// setCommitted sets a session's measured footprint directly.
func setCommitted(r *RingWindow, id string, tokens int64) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.sessions[id].committedTokens = tokens
}

// noteMeanTurn records completed turns so the running mean is ŝ.
func noteMeanTurn(r *RingWindow, d time.Duration, n int64) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.turnDurSum += d * time.Duration(n)
	r.mgr.turnDurCount += n
}

// runTurn dispatches and completes one turn reporting totalTokens and a
// turn duration, the way PreRequest and ResponseBody would.
func runTurn(t *testing.T, r *RingWindow, id string, sizeBytes, totalTokens int, dur time.Duration) {
	t.Helper()
	req := newRequest(id, sizeBytes)
	start := time.Now().Add(-dur)
	req.PutAttribute(requestStartKey, start)
	req.PutAttribute(inflightEstimateKey, int64(sizeBytes/4))
	r.mgr.settleResponse(id, int64(sizeBytes/4), int64(totalTokens), dur, time.Now())
}

// resetScan points the round-counter cursor at the start of round 0, so
// tests assert exact sequences.
func resetScan(r *RingWindow) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.round = 0
	r.mgr.slot = 0
}

// hashOrder returns ids sorted by their placement hash (the plugin's
// first-placement ring order), so tests can compute expectations under
// the per-process salt.
func hashOrder(ids ...string) []string {
	out := append([]string(nil), ids...)
	sort.Slice(out, func(i, j int) bool { return placementHash(out[i]) < placementHash(out[j]) })
	return out
}

func counters(r *RingWindow) managerCounters {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	return r.mgr.counters
}

// joinWindow drives the cold-start join path: with no member waiting, each
// Pick admits the head outsider. One queue per call keeps the member from
// being "waiting" while the next session joins.
func joinWindow(t *testing.T, r *RingWindow, ids ...string) {
	t.Helper()
	for _, id := range ids {
		q := pick(t, r, makeQueue(id, time.Second, time.Minute, 280))
		require.NotNil(t, q, "cold-start join should dispatch %s", id)
		// The dispatched turn runs and completes: the in-flight estimate
		// is charged and settled, which also clears the admission
		// reservation, exactly as PreRequest and ResponseBody would.
		r.mgr.chargePreRequest(id, "pod", 70, time.Now())
		r.mgr.settleResponse(id, 70, 0, 0, time.Now())
	}
}

func TestColdStartJoinsInRingOrder(t *testing.T) {
	r := newTestPlugin(testConfig())
	// All four arrive at once; joins follow ring (hash) order, one per
	// Pick. A joined session's queue is not re-presented (its next turn
	// has not arrived yet), so each Pick admits the next head outsider.
	H := hashOrder("A", "B", "C", "D")
	remaining := []string{"A", "B", "C", "D"}
	qs := func(ids ...string) []fwkfc.FlowQueueAccessor {
		out := make([]fwkfc.FlowQueueAccessor, 0, len(ids))
		for _, id := range ids {
			out = append(out, makeQueue(id, 0, time.Minute, 280))
		}
		return out
	}
	for _, want := range H {
		got := pickID(t, r, qs(remaining...)...)
		require.Equal(t, want, got)
		for i, id := range remaining {
			if id == got {
				remaining = append(remaining[:i], remaining[i+1:]...)
				break
			}
		}
	}
	require.Equal(t, H, memberIDs(r))
}

func TestMemberServesBackAndForth(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	queues := func() []fwkfc.FlowQueueAccessor {
		return []fwkfc.FlowQueueAccessor{
			makeQueue("A", 0, time.Minute, 280),
			makeQueue("B", 0, time.Minute, 280),
			makeQueue("C", 0, time.Minute, 280),
			makeQueue("D", 0, time.Minute, 280),
		}
	}
	// The round-counter order: round r serves [r, r+1, …, r+L−1] then the
	// interior in reverse, over the members in ring (hash) order H0..H3.
	// Four rounds form the full period — a 24-serve cycle, not period-4.
	resetScan(r)
	H := hashOrder("A", "B", "C", "D")
	var got []string
	for i := 0; i < 24; i++ {
		got = append(got, pickID(t, r, queues()...))
	}
	require.Equal(t, []string{
		H[0], H[1], H[2], H[3], H[2], H[1], // round 0
		H[1], H[2], H[3], H[0], H[3], H[2], // round 1
		H[2], H[3], H[0], H[1], H[0], H[3], // round 2
		H[3], H[0], H[1], H[2], H[1], H[0], // round 3
	}, got)
	// Every member served exactly six times per period: equal service.
	counts := map[string]int{}
	for _, id := range got {
		counts[id]++
	}
	for _, id := range []string{"A", "B", "C", "D"} {
		require.Equal(t, 6, counts[id], "member %s should be served six times per period", id)
	}
}

func TestQuietMemberSkippedKeepsPlace(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	// C is in a tool call: A, B, D have turns waiting, C does not.
	resetScan(r)
	live := func() []fwkfc.FlowQueueAccessor {
		return []fwkfc.FlowQueueAccessor{
			makeQueue("A", 0, time.Minute, 280),
			makeQueue("B", 0, time.Minute, 280),
			makeQueue("D", 0, time.Minute, 280),
		}
	}
	var got []string
	for i := 0; i < 4; i++ {
		got = append(got, pickID(t, r, live()...))
	}
	for _, id := range got {
		require.NotEqual(t, "C", id, "quiet member must never be served")
	}
	// One round serves each live member at its slots (hash order, C quiet).
	want := hashOrder("A", "B", "D")
	require.Equal(t, want, got[:3], "round 0 serves the live members at their slots")
	// C returns: it must be served within one round (2L-2 = 6 serves).
	resetScan(r)
	all := append(live(), makeQueue("C", 0, time.Minute, 280))
	served := false
	for i := 0; i < 6; i++ {
		if pickID(t, r, all...) == "C" {
			served = true
			break
		}
	}
	require.True(t, served, "returning member served within one round")
}

func TestRotationOnOutsiderSlackExpiry(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80) // sum 320, budget 384
	}
	// Release order: C finished longest ago and is quiet; A and B have
	// turns waiting; D is quiet but newer than C.
	setFinish(r, "C", 90*time.Second)
	setFinish(r, "D", 30*time.Second)
	setFinish(r, "A", 10*time.Second)
	setFinish(r, "B", 5*time.Second)

	// E waited past its whole budget: slack is deeply negative.
	got := pickID(t, r,
		makeQueue("A", 0, time.Minute, 320),
		makeQueue("B", 0, time.Minute, 320),
		makeQueue("E", 120*time.Second, time.Minute, 320))
	require.Equal(t, "E", got, "expired outsider rotates the window and is dispatched first")
	require.True(t, isMember(r, "E"))
	require.False(t, isMember(r, "C"), "the oldest-released quiet member is demoted")
	require.True(t, isMember(r, "A"))
	require.True(t, isMember(r, "B"))
	require.True(t, isMember(r, "D"))
	require.Equal(t, int64(1), counters(r).rotations)
	require.Equal(t, int64(1), counters(r).demotions["rotate"])
}

func TestMembersNeverTriggerRotation(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	// Every member waited past its budget; no outsider exists.
	queues := []fwkfc.FlowQueueAccessor{
		makeQueue("A", 300*time.Second, time.Minute, 280),
		makeQueue("B", 200*time.Second, time.Minute, 280),
		makeQueue("C", 100*time.Second, time.Minute, 280),
		makeQueue("D", 50*time.Second, time.Minute, 280),
	}
	got := pickID(t, r, queues...)
	require.Equal(t, "A", got, "the most expired member recovers first")
	require.Equal(t, int64(0), counters(r).rotations, "member slacks never rotate the window")
	require.Equal(t, int64(1), counters(r).memberServes)
}

func TestElasticDPreventsRotationStorm(t *testing.T) {
	cfg := testConfig()
	r := newTestPlugin(cfg)
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	// Measured mean turn: 5 s. Ten waiting outsiders with a 1 s budget:
	// D_eff stretches to 50 s, so nobody's slack is expired.
	noteMeanTurn(r, 5*time.Second, 10)
	var queues []fwkfc.FlowQueueAccessor
	for _, id := range []string{"E", "F", "G", "H", "I", "J", "K", "L", "M", "N"} {
		queues = append(queues, makeQueue(id, 2*time.Second, time.Second, 320))
	}
	got := pickID(t, r, queues...)
	waiting := []string{"E", "F", "G", "H", "I", "J", "K", "L", "M", "N"}
	require.Contains(t, waiting, got, "the hash-head outsider joins through the normal path")
	require.Equal(t, int64(0), counters(r).rotations, "elastic D_eff must not rotate on a 1s budget with 10 waiting")
	require.True(t, isMember(r, got))
}

func TestElasticDFiresWhenTrulyExpired(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	noteMeanTurn(r, 5*time.Second, 10)
	// One waiting outsider, waited 10 s of a 1 s budget: D_eff = max(1, 5) = 5,
	// slack = -10 + 5 - 5 = -10 → rotation.
	got := pickID(t, r, makeQueue("E", 10*time.Second, time.Second, 320))
	require.Equal(t, "E", got)
	require.Equal(t, int64(1), counters(r).rotations)
}

func TestNeverIdleGuest(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 90) // sum 360, budget 384
	}
	setFinish(r, "C", 90*time.Second)
	setFinish(r, "D", 30*time.Second)
	// No member is waiting; F does not fit (360 + 80 > 384) but the GPU
	// must not idle.
	got := pickID(t, r, makeQueue("F", 0, time.Minute, 320))
	require.Equal(t, "F", got, "a non-fitting outsider runs as a guest when no member waits")
	require.Equal(t, int64(1), counters(r).guests)
	require.True(t, isMember(r, "F"), "the guest joins once demotion makes room")
	require.False(t, isMember(r, "C"), "the oldest-released quiet member was demoted for the guest")
}

func TestOversizedGuestClearsWindow(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	// F's estimate (500 tokens) exceeds the whole budget (384): no subset
	// of members can cover it.
	got := pickID(t, r, makeQueue("F", 0, time.Minute, 2000))
	require.Equal(t, "F", got, "an oversized guest still runs: the GPU must not idle")
	require.Equal(t, int64(1), counters(r).oversized)
	require.False(t, isMember(r, "F"), "the oversized guest does not join")
	require.Empty(t, memberIDs(r), "the window clears and refills by arrival")
}

func TestDemotionSkipsWaitingAndRunning(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	// A has a turn waiting (now a valid victim — but C is older); B has a
	// turn in flight (never a victim: its blocks are pinned); C and D are
	// quiet. C is the oldest release.
	r.mgr.mu.Lock()
	r.mgr.sessions["B"].inflightTokens = 80
	r.mgr.mu.Unlock()
	setFinish(r, "C", 90*time.Second)
	setFinish(r, "D", 30*time.Second)
	setFinish(r, "B", 10*time.Second)

	got := pickID(t, r,
		makeQueue("A", 0, time.Minute, 320),
		makeQueue("E", 120*time.Second, time.Minute, 320))
	require.Equal(t, "E", got)
	require.True(t, isMember(r, "A"), "a waiting member is never the demotion victim")
	require.True(t, isMember(r, "B"), "a running member is never the demotion victim")
	require.False(t, isMember(r, "C"), "the oldest quiet release is the victim")
	require.True(t, isMember(r, "D"))
}

func TestDWidthBinds(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	setCommitted(r, "A", 70)
	// Mean turn 10 s, D = 30 s → W_D = 30/(2·10) = 1: the budget would
	// allow five members of 70, but the D-width allows one.
	noteMeanTurn(r, 10*time.Second, 10)
	got := pickID(t, r, makeQueue("B", 0, 30*time.Second, 280))
	require.Equal(t, "B", got, "never idle: B runs even though the D-width is full")
	require.Equal(t, int64(1), counters(r).guests)
	require.LessOrEqual(t, len(memberIDs(r)), 1, "the D-width caps the window at one member")
	require.Equal(t, []string{"B"}, memberIDs(r), "B replaced A: the demotion loop respects the D-width")
}

func TestAnonymousPassesLast(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	setCommitted(r, "A", 70)
	// An identified outsider outranks anonymous traffic.
	got := pickID(t, r, makeQueue("", 0, time.Minute, 280), makeQueue("B", 0, time.Minute, 280))
	require.Equal(t, "B", got)
	// Anonymous alone passes through.
	got = pickID(t, r, makeQueue("", 0, time.Minute, 280))
	require.Equal(t, "", got)
	require.Equal(t, float64(1), testutil.ToFloat64(r.metrics.pickAnonymous))
}

func TestLedgerCommitsFootprint(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	runTurn(t, r, "A", 280, 313, 7*time.Second)
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	s := r.mgr.sessions["A"]
	require.Equal(t, int64(313), s.committedTokens, "usage.total_tokens of the last completed turn is the footprint")
	require.Equal(t, 7*time.Second, s.lastTurnDur)
	require.False(t, s.lastFinish.IsZero(), "the completion time is the release order")
	require.Equal(t, 7*time.Second, r.mgr.meanTurnDurLocked())
}

func TestSweepRejoinsAtTail(t *testing.T) {
	cfg := testConfig()
	cfg.EvictionTTLSeconds = 3600
	r := newTestPlugin(cfg)
	joinWindow(t, r, "A", "B")
	// A goes quiet long past the TTL and is swept.
	r.mgr.mu.Lock()
	r.mgr.sessions["A"].lastActivity = time.Now().Add(-2 * time.Hour)
	r.mgr.mu.Unlock()
	r.mgr.sweep(time.Now())
	require.False(t, isMember(r, "A"))
	// A returns and rejoins behind everyone.
	joinWindow(t, r, "C")
	joinWindow(t, r, "A")
	r.mgr.mu.Lock()
	seqA := r.mgr.sessions["A"].seq
	seqC := r.mgr.sessions["C"].seq
	r.mgr.mu.Unlock()
	require.Greater(t, seqA, seqC, "a swept session that returns rejoins at the ring tail")
}

func TestTurnHitRatioTelemetry(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	req := newRequest("A", 280)
	req.PutAttribute(inflightEstimateKey, int64(70))
	resp := &fwkrc.Response{
		EndOfStream: true,
		Usage: requesthandling.Usage{
			PromptTokens: 100, TotalTokens: 170,
			PromptTokenDetails: &requesthandling.PromptTokenDetails{CachedTokens: 95},
		},
	}
	r.ResponseBody(context.Background(), req, resp, nil)
	require.Equal(t, 1, testutil.CollectAndCount(r.metrics.turnHitRatio), "one hit-ratio observation")
}

// --- V0-REVIEW code-review regression tests ---

// TestTwoMemberWindowNeverStalls: the old cursor deadlocked a 2-member
// window at both edges and Pick returned nil forever (code review issue 4).
func TestTwoMemberWindowNeverStalls(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B")
	setCommitted(r, "A", 70)
	setCommitted(r, "B", 70)
	resetScan(r)
	queues := func() []fwkfc.FlowQueueAccessor {
		return []fwkfc.FlowQueueAccessor{
			makeQueue("A", 0, time.Minute, 280),
			makeQueue("B", 0, time.Minute, 280),
		}
	}
	var got []string
	for i := 0; i < 8; i++ {
		id := pickID(t, r, queues()...)
		require.NotEmpty(t, id, "a 2-member window must never stall (pick %d)", i)
		got = append(got, id)
	}
	counts := map[string]int{}
	for _, id := range got {
		counts[id]++
	}
	require.InDelta(t, 4, counts["A"], 1, "equal service in a 2-member window")
	require.InDelta(t, 4, counts["B"], 1, "equal service in a 2-member window")
}

// TestBusyRotationDemotesNotClears: in the busy steady state every member
// has a turn waiting; the rotation must demote the oldest-released member,
// not take the oversized path and clear the window (code review issue 5).
func TestBusyRotationDemotesNotClears(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	setFinish(r, "C", 90*time.Second)
	setFinish(r, "D", 30*time.Second)
	setFinish(r, "B", 10*time.Second)
	setFinish(r, "A", 5*time.Second)
	// Every member has a turn waiting when E's D-clock expires.
	got := pickID(t, r,
		makeQueue("A", 0, time.Minute, 320),
		makeQueue("B", 0, time.Minute, 320),
		makeQueue("C", 0, time.Minute, 320),
		makeQueue("D", 0, time.Minute, 320),
		makeQueue("E", 120*time.Second, time.Minute, 320))
	require.Equal(t, "E", got)
	require.Equal(t, int64(1), counters(r).rotations)
	require.Equal(t, int64(0), counters(r).oversized, "a busy rotation is not an oversized guest")
	require.Equal(t, int64(0), counters(r).noVictims)
	require.Equal(t, int64(0), counters(r).demotions["clear"], "the window must not clear")
	require.Equal(t, int64(1), counters(r).demotions["rotate"])
	require.Equal(t, hashOrder("A", "B", "D", "E"), memberIDs(r), "the oldest-released member is demoted, the rest stay")
}

// TestDemotionReSeqsVictim: a demoted session rejoins the outsider line at
// the tail, so it cannot be the head outsider ahead of later sessions
// (code review issue 6).
func TestDemotionReSeqsVictim(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	setFinish(r, "C", 90*time.Second)
	setFinish(r, "D", 30*time.Second)
	setFinish(r, "B", 10*time.Second)
	setFinish(r, "A", 5*time.Second)
	// E (expired) and F (fresh) both wait; the seat goes to the head by
	// ring order (hash), the D clock may have been triggered by E.
	head := hashOrder("E", "F")[0]
	got := pickID(t, r,
		makeQueue("E", 120*time.Second, time.Minute, 320),
		makeQueue("F", 60*time.Second, time.Minute, 280))
	require.Equal(t, head, got)
	require.False(t, isMember(r, "C"))

	// No member waits and nobody's D-clock has expired, so the head
	// outsider — the earliest seq — dispatches. C was just demoted
	// (re-seq'd behind F), so F is the head even though C has waited
	// longer in wall time.
	got = pickID(t, r,
		makeQueue("C", 40*time.Second, time.Minute, 320),
		makeQueue("F", 30*time.Second, time.Minute, 280))
	require.Equal(t, "F", got, "a just-demoted session rejoins behind everyone, regardless of wall-clock wait")
}

// TestC256Magnitudes: the elastic D_eff must include the measured
// concurrency (code review issue 3). With mean turn 70 s, 126 waiting and
// 77 in flight, the achievable round is ~115 s, not 8,820 s.
func TestC256Magnitudes(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	setCommitted(r, "A", 70000)
	noteMeanTurn(r, 70*time.Second, 100)
	// 77 sessions with a turn in flight: ĉ = 77.
	for i := 0; i < 77; i++ {
		r.mgr.chargePreRequest(fmt.Sprintf("run-%d", i), "pod", 1000, time.Now())
	}
	// 126 waiting turns: A (member) plus 125 outsiders.
	var queues []fwkfc.FlowQueueAccessor
	queues = append(queues, makeQueue("A", 0, 60*time.Second, 280000))
	for i := 0; i < 125; i++ {
		queues = append(queues, makeQueue(fmt.Sprintf("w-%d", i), 0, 60*time.Second, 280000))
	}
	got := pickID(t, r, queues...)
	require.NotEmpty(t, got)
	r.mgr.mu.Lock()
	dEff := r.mgr.lastDEff
	widthD := r.mgr.lastWidthD
	r.mgr.mu.Unlock()
	require.InDelta(t, 114, int(dEff.Seconds()), 3, "D_eff = mean*N/c = 70*126/77 ≈ 114.5 s, not 8820 s")
	// The elastic floor (114.5 s) exceeds the operator's D (60 s), so the
	// D-width is bypassed and the budget binds (V0-REVIEW verification
	// residuals 1+2): the width must NOT degenerate to N̂/2.7 = 46.
	require.Equal(t, 1<<30, widthD, "with the elastic floor active the D-width is bypassed; the budget binds")
}

// TestRecoveryLaxityRunsScan: raw least-slack recovery is round-robin
// (code review issue 2); with laxity relative to the scan, a generous
// budget leaves the scan in charge and a tight budget rescues only the
// member the scan cannot reach in time.
func TestRecoveryLaxityRunsScan(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	noteMeanTurn(r, 10*time.Second, 10)
	resetScan(r)
	// Generous budget: nobody's urgency is negative, the scan runs.
	queues := func() []fwkfc.FlowQueueAccessor {
		return []fwkfc.FlowQueueAccessor{
			makeQueue("A", 50*time.Second, 100*time.Second, 280),
			makeQueue("B", 40*time.Second, 100*time.Second, 280),
			makeQueue("C", 30*time.Second, 100*time.Second, 280),
			makeQueue("D", 20*time.Second, 100*time.Second, 280),
		}
	}
	H := hashOrder("A", "B", "C", "D")
	require.Equal(t, H[0], pickID(t, r, queues()...), "the scan serves in round-counter order")
	require.Equal(t, H[1], pickID(t, r, queues()...))
	require.Equal(t, float64(0), testutil.ToFloat64(r.metrics.pickRecoveries))

	// Tight budget (D = 25 s): only A — which the scan reaches next but
	// whose budget is already blown — is rescued; the rest scan.
	resetScan(r)
	tight := func(withA bool) []fwkfc.FlowQueueAccessor {
		qs := []fwkfc.FlowQueueAccessor{
			makeQueue("B", 40*time.Second, 25*time.Second, 280),
			makeQueue("C", 30*time.Second, 25*time.Second, 280),
			makeQueue("D", 20*time.Second, 25*time.Second, 280),
		}
		if withA {
			qs = append([]fwkfc.FlowQueueAccessor{makeQueue("A", 90*time.Second, 25*time.Second, 280)}, qs...)
		}
		return qs
	}
	require.Equal(t, "A", pickID(t, r, tight(true)...), "the member the scan cannot reach in time is rescued")
	require.Equal(t, "B", pickID(t, r, tight(false)...), "with A served, the scan resumes")
}

// TestNoVictimsKeepsWindow: when every member is running (nothing
// demotable), the admission runs as a non-member and the window stays
// intact — and is not counted as oversized (code review issue 7).
func TestNoVictimsKeepsWindow(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B")
	for _, id := range []string{"A", "B"} {
		setCommitted(r, id, 160)
		r.mgr.mu.Lock()
		r.mgr.sessions[id].inflightTokens = 160 // running: not demotable
		r.mgr.mu.Unlock()
	}
	noteMeanTurn(r, 5*time.Second, 10)
	// G does not fit (320 + 70 > 384) and no member can be demoted.
	got := pickID(t, r, makeQueue("G", 0, time.Minute, 280))
	require.Equal(t, "G", got, "never idle: G runs")
	require.Equal(t, int64(0), counters(r).oversized, "not oversized: G fits an empty window")
	require.Equal(t, int64(1), counters(r).noVictims)
	require.False(t, isMember(r, "G"), "G runs as a non-member")
	require.Equal(t, hashOrder("A", "B"), memberIDs(r), "the window stays intact")
}

// TestRotationGuardAtColdStart: with D = 0 and no measured turn duration,
// the rotation test is skipped and the first joins are counted as joins,
// not rotations (code review issue 8).
func TestRotationGuardAtColdStart(t *testing.T) {
	r := newTestPlugin(testConfig())
	require.Equal(t, hashOrder("A", "B")[0], pickID(t, r,
		makeQueue("A", 0, 0, 280),
		makeQueue("B", 0, 0, 280)))
	require.Equal(t, int64(0), counters(r).rotations, "cold start with D=0 must not count joins as rotations")
	require.Equal(t, float64(1), testutil.ToFloat64(r.metrics.pickJoins))
}

// TestShallowQueueWindowDoesNotCollapse: with the elastic floor active
// (D=0 or D below the floor), the D-width must not pinch the window to one
// member at shallow queue depths (V0-REVIEW verification residual 2).
func TestShallowQueueWindowDoesNotCollapse(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A")
	setCommitted(r, "A", 70)
	noteMeanTurn(r, 10*time.Second, 10)
	// N̂ = 1 (B waiting), ĉ = 1: the elastic floor is 10 s. With D = 0 the
	// width must be unbounded; with D = 5 s (below the floor) it must also
	// be unbounded (the floor already paces rotation); with D = 30 s (above
	// the floor) the width applies: 30·1/(2.7·10) = 1.
	r.mgr.mu.Lock()
	r.mgr.lastNWaiting = 1
	w0 := r.mgr.widthDLocked(0)
	w5 := r.mgr.widthDLocked(5 * time.Second)
	w30 := r.mgr.widthDLocked(30 * time.Second)
	r.mgr.mu.Unlock()
	require.Equal(t, 1<<30, w0, "D=0: no D-width")
	require.Equal(t, 1<<30, w5, "D below the elastic floor: the floor paces; no D-width")
	require.Equal(t, 1, w30, "D above the floor: the D-width applies")
}

// TestRoundClockOnePerRound: at D=0 the window advances exactly once per
// completed round — never per Pick.
func TestRoundClockOnePerRound(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	resetScan(r)
	queues := func() []fwkfc.FlowQueueAccessor {
		return []fwkfc.FlowQueueAccessor{
			makeQueue("A", 0, 0, 280), makeQueue("B", 0, 0, 280),
			makeQueue("C", 0, 0, 280), makeQueue("D", 0, 0, 280),
		}
	}
	// Before a round completes: an expired outsider waits (D=0), no rotation.
	got := pickID(t, r, append(queues(), makeQueue("E", 300*time.Second, 0, 280))...)
	require.NotEqual(t, "E", got, "no rotation before a round completes at D=0")
	require.Equal(t, int64(0), counters(r).rotations)

	// Serve one full round (2L-2 = 6 serves).
	H := hashOrder("A", "B", "C", "D")
	for i := 0; i < 6; i++ {
		pickID(t, r, queues()...)
	}
	// Now the round clock ticks: exactly one rotation for E.
	got = pickID(t, r, append(queues(), makeQueue("E", 300*time.Second, 0, 280))...)
	require.Equal(t, "E", got)
	require.Equal(t, int64(1), counters(r).rotations)
	_ = H
	// And not a second one until another round completes: the follow-up
	// pick may serve any waiting member (E included — it just joined),
	// but no further rotation may fire.
	pickID(t, r, makeQueue("E", 0, 0, 280))
	require.Equal(t, int64(1), counters(r).rotations, "one place per round, not per Pick")
}

// TestTransparentModeWhenPoolDies: when the idle pool cannot hold one mean
// footprint, serve oldest-first without membership churn.
func TestTransparentModeWhenPoolDies(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B")
	for _, id := range []string{"A", "B"} {
		setCommitted(r, id, 70000)
	}
	r.mgr.noteKVUsage(0.999) // idle pool = 480 tokens << 70k footprint
	got := pickID(t, r,
		makeQueue("A", 30*time.Second, time.Minute, 280),
		makeQueue("B", 5*time.Second, time.Minute, 280))
	require.Equal(t, "A", got, "transparent mode serves the oldest arrival")
	require.Equal(t, int64(0), counters(r).rotations)
	require.Equal(t, float64(1), testutil.ToFloat64(r.metrics.pickTransparent))
}

// TestHashCohortReentry: a demoted session (cohort >= 1) sorts behind every
// first-placement session (cohort 0), whatever their hashes.
func TestHashCohortReentry(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 80)
	}
	setFinish(r, "A", 90*time.Second)
	// A busy rotation demotes A (the oldest release).
	pickID(t, r,
		makeQueue("A", 0, time.Minute, 320),
		makeQueue("B", 0, time.Minute, 320),
		makeQueue("C", 0, time.Minute, 320),
		makeQueue("D", 0, time.Minute, 320),
		makeQueue("E", 120*time.Second, time.Minute, 320))
	require.False(t, isMember(r, "A"))
	r.mgr.mu.Lock()
	gen := r.mgr.sessions["A"].sortGen
	r.mgr.mu.Unlock()
	require.GreaterOrEqual(t, gen, int64(1), "demotion moves the session to a re-entry cohort")

	// A (demoted, cohort 1) and F (never seen, cohort 0) both wait: F
	// must be the head — cohort 0 beats cohort 1 regardless of hash order.
	got := pickID(t, r,
		makeQueue("A", 300*time.Second, time.Minute, 320),
		makeQueue("F", 5*time.Second, time.Minute, 280))
	require.Equal(t, "F", got, "a demoted session rejoins behind every first-placement session")
}
