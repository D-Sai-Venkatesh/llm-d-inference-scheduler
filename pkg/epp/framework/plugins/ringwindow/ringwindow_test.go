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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/prometheus/client_golang/prometheus/testutil"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
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
	// All four arrive at once; each Pick dispatches one and joins it.
	require.Equal(t, "A", pickID(t, r, makeQueue("A", 0, time.Minute, 280),
		makeQueue("B", 0, time.Minute, 280), makeQueue("C", 0, time.Minute, 280),
		makeQueue("D", 0, time.Minute, 280)))
	require.Equal(t, "B", pickID(t, r, makeQueue("B", 0, time.Minute, 280),
		makeQueue("C", 0, time.Minute, 280), makeQueue("D", 0, time.Minute, 280)))
	require.Equal(t, "C", pickID(t, r, makeQueue("C", 0, time.Minute, 280),
		makeQueue("D", 0, time.Minute, 280)))
	require.Equal(t, "D", pickID(t, r, makeQueue("D", 0, time.Minute, 280)))
	require.Equal(t, []string{"A", "B", "C", "D"}, memberIDs(r))
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
	// The joins re-anchored the cursor at D; the sweep continues backward
	// from there and wraps: C B A D, then repeats with a rotated turnaround.
	var got []string
	for i := 0; i < 8; i++ {
		got = append(got, pickID(t, r, queues()...))
	}
	require.Equal(t, []string{"C", "B", "A", "D", "C", "B", "A", "D"}, got)
	// Every member served exactly once per pass.
	counts := map[string]int{}
	for _, id := range got {
		counts[id]++
	}
	for _, id := range []string{"A", "B", "C", "D"} {
		require.Equal(t, 2, counts[id], "member %s should be served twice in 8 serves", id)
	}
}

func TestQuietMemberSkippedKeepsPlace(t *testing.T) {
	r := newTestPlugin(testConfig())
	joinWindow(t, r, "A", "B", "C", "D")
	for _, id := range []string{"A", "B", "C", "D"} {
		setCommitted(r, id, 70)
	}
	// C is in a tool call: A, B, D have turns waiting, C does not.
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
	// One pass over the ring serves each live member once; the fourth
	// serve is the start of the next pass.
	require.Equal(t, []string{"B", "A", "D"}, got[:3], "one pass serves each live member once")
	// C returns: it must be served within one pass.
	all := append(live(), makeQueue("C", 0, time.Minute, 280))
	served := false
	for i := 0; i < 4; i++ {
		if pickID(t, r, all...) == "C" {
			served = true
			break
		}
	}
	require.True(t, served, "returning member served within one pass")
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
	require.Equal(t, "E", got, "the head outsider joins through the normal path")
	require.Equal(t, int64(0), counters(r).rotations, "elastic D_eff must not rotate on a 1s budget with 10 waiting")
	require.True(t, isMember(r, "E"))
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
	// A has a turn waiting; B has a turn in flight (charged but not
	// settled); C and D are quiet. C is the oldest release.
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
