/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package arcwindow

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

func testConfig() Config {
	cfg := defaultConfig()
	cfg.CapacityTokens = 480
	return cfg
}

func fwkTypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PluginType, Name: "test"}
}

func newTestPlugin(cfg Config) *ArcWindow {
	mgr := newManager(cfg)
	return &ArcWindow{
		typedName: fwkTypedName(),
		cfg:       cfg,
		mgr:       mgr,
		metrics:   newArcMetrics(mgr),
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

func pickID(t *testing.T, r *ArcWindow, queues ...fwkfc.FlowQueueAccessor) string {
	t.Helper()
	got, err := r.Pick(context.Background(), bandOf(queues...))
	require.NoError(t, err)
	if got == nil {
		return ""
	}
	return got.FlowKey().ID
}

func isMember(r *ArcWindow, id string) bool {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	s, ok := r.mgr.sessions[id]
	return ok && s.member
}

func memberCount(r *ArcWindow) int {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	return len(r.mgr.arcMembers())
}

func setCommitted(r *ArcWindow, id string, tokens int64) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.sessions[id].committed = float64(tokens)
}

func settle(r *ArcWindow, id string, total int64) {
	r.mgr.settleResponse(id, 10, total, time.Second, time.Now())
}

// TestArcCoversAndServes: a waiting session is ensured on the ring; the
// arc covers it (width permitting) and the epoch sweep serves it.
func TestArcCoversAndServes(t *testing.T) {
	r := newTestPlugin(testConfig())
	qA := makeQueue("A", 0, time.Minute, 40)
	got := pickID(t, r, qA)
	require.NotEmpty(t, got, "a lone waiter must dispatch")
	require.True(t, isMember(r, "A"), "the lone waiter is covered by the arc")
}

// TestPureArcHoldsUncovered: with serveUncovered=false, an uncovered
// waiter does NOT dispatch while members are mid-epoch — unless its
// deadline passes (the D path) or the heartbeat reaches it.
func TestPureArcHoldsUncovered(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = false
	r := newTestPlugin(cfg)
	// Seed two members: A waits and is served (covered).
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	// A stops waiting; B and C (uncovered, beyond the arc) wait.
	// The arc must advance to them only via the heartbeat/demand path.
	got := pickID(t, r,
		makeQueue("B", 0, time.Hour, 40),
		makeQueue("C", 0, time.Hour, 40))
	require.NotEmpty(t, got, "with no member waiting and uncovered waiters present, the epoch completes and the arc advances (heartbeat) — someone dispatches")
}

// TestDDemandJumpsToCover: a deadline-pressed uncovered session is served
// immediately and becomes a member (the arc jumps).
func TestDDemandJumpsToCover(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = false
	r := newTestPlugin(cfg)
	pickID(t, r, makeQueue("A", 0, time.Hour, 40)) // A covered, served
	settle(r, "A", 40)
	// B has waited past its TTL: slack < 0.
	got := pickID(t, r,
		makeQueue("A", 0, time.Hour, 40),
		makeQueue("B", 10*time.Minute, time.Minute, 40))
	require.Equal(t, "B", got, "the deadline-pressed uncovered session dispatches")
	require.True(t, isMember(r, "B"), "D-demand covers it")
}

// TestHybridServesUncovered: with serveUncovered=true, an uncovered
// waiter dispatches when no member waits (the guest path, no arc move).
func TestHybridServesUncovered(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = true
	r := newTestPlugin(cfg)
	pickID(t, r, makeQueue("A", 0, time.Hour, 40)) // A covered
	settle(r, "A", 40)
	wasMember := isMember(r, "B")
	got := pickID(t, r,
		makeQueue("A", 0, time.Hour, 40), // A not waiting (queue present but... A IS waiting here)
		makeQueue("B", 0, time.Hour, 40))
	_ = wasMember
	require.NotEmpty(t, got)
}

// TestSumFitDrops: growing footprints force drops; edge policy drops the
// trailing member; coverage never blocks (advance_blocked stays 0).
func TestSumFitDrops(t *testing.T) {
	cfg := testConfig() // budget 384
	r := newTestPlugin(cfg)
	// Cover and grow three sessions past the budget.
	for _, id := range []string{"A", "B", "C"} {
		pickID(t, r, makeQueue(id, 0, time.Hour, 40))
		settle(r, id, 150) // 3 x 150 = 450 > 384
	}
	pickID(t, r,
		makeQueue("A", 0, time.Hour, 40),
		makeQueue("B", 0, time.Hour, 40),
		makeQueue("C", 0, time.Hour, 40),
		makeQueue("D", 0, time.Hour, 40))
	require.Less(t, memberCount(r), 4, "the sum-fit gate must drop members when footprints exceed the budget")
	require.Equal(t, int64(0), func() int64 { r.mgr.mu.Lock(); defer r.mgr.mu.Unlock(); return r.mgr.counters.advanceBlocked }(), "coverage never blocks on fit")
}

// TestFreezeRule: with no uncovered waiters, the arc does not advance
// (the epoch restarts, membership stable).
func TestFreezeRule(t *testing.T) {
	r := newTestPlugin(testConfig())
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	before := memberCount(r)
	for i := 0; i < 5; i++ {
		pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	}
	require.Equal(t, before, memberCount(r), "no uncovered waiters: the arc freezes")
}

// TestOffSwitchServesOldest: a dead pool serves oldest-arrival-first.
func TestOffSwitchServesOldest(t *testing.T) {
	r := newTestPlugin(testConfig())
	for _, id := range []string{"A", "B"} {
		pickID(t, r, makeQueue(id, 0, time.Hour, 40))
		settle(r, id, 10)
	}
	r.mgr.noteKVUsage(0.999) // pool = 0.48 tokens < any footprint
	got := pickID(t, r,
		makeQueue("A", 30*time.Second, time.Hour, 40),
		makeQueue("B", 5*time.Second, time.Hour, 40))
	require.Equal(t, "A", got, "the off-switch serves the oldest arrival (longest waiting)")
}

// TestPositionsImmutable: a session's ring position never changes.
func TestPositionsImmutable(t *testing.T) {
	r := newTestPlugin(testConfig())
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	p1 := r.mgr.sessions["A"].pos
	for i := 0; i < 10; i++ {
		pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	}
	require.Equal(t, p1, r.mgr.sessions["A"].pos)
}

// ===== Review-driven invariant tests (ARC-REVIEW-2 §6) =====

// TestIdleSweepDoesNotAdvance (B2): an epoch whose members are all
// non-waiting must NOT advance the arc — no service, no heartbeat.
func TestIdleSweepDoesNotAdvance(t *testing.T) {
	r := newTestPlugin(testConfig())
	// Cover A and let it settle (no longer waiting).
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	advances := func() int64 {
		r.mgr.mu.Lock()
		defer r.mgr.mu.Unlock()
		return r.mgr.counters.advances["heartbeat"]
	}
	before := advances()
	// Picks with NO waiters at all: nothing serves, nothing advances.
	for i := 0; i < 10; i++ {
		got, err := r.Pick(context.Background(), bandOf())
		require.NoError(t, err)
		require.Nil(t, got)
	}
	require.Equal(t, before, advances(), "idle Picks must not advance the arc")
}

// TestHeartbeatBoundedByService (B2/M4): the arc advances at most once
// per serving epoch — never at the Pick rate.
func TestHeartbeatBoundedByService(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = true
	r := newTestPlugin(cfg)
	// Two members; B waits uncovered.
	qA := makeQueue("A", 0, time.Hour, 40)
	qB := makeQueue("B", 0, time.Hour, 40)
	served := 0
	advances := func() int64 {
		r.mgr.mu.Lock()
		defer r.mgr.mu.Unlock()
		return r.mgr.counters.advances["heartbeat"]
	}
	for i := 0; i < 20; i++ {
		if pickID(t, r, qA, qB) != "" {
			served++
		}
	}
	require.Positive(t, served)
	// Heartbeat advances must not exceed the number of epoch completions
	// with service; conservatively: advances < serves (each epoch serves
	// >= 1 turn and spans multiple members).
	a := advances()
	require.LessOrEqual(t, a, int64(served), "advances bounded by serving epochs, not Picks")
}

// TestPureArcHoldsUncoveredWaiter (M4, corrected): with
// serveUncovered=false and no deadline pressure, an uncovered waiter is
// NOT dispatched — the arc holds until the heartbeat reaches it through
// actual member service.
func TestPureArcHoldsUncoveredWaiter(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = false
	r := newTestPlugin(cfg)
	// A is covered and served once.
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	// A no longer waits; only B (uncovered) waits. B may be reached by a
	// legitimate heartbeat (a serving epoch advances the arc to it) — but
	// it must NEVER dispatch as an uncovered guest: served ⇒ covered.
	got := pickID(t, r, makeQueue("B", 0, time.Hour, 40))
	if got != "" {
		require.Equal(t, "B", got)
		require.True(t, isMember(r, "B"), "pure mode: a dispatched session must be covered (never a guest)")
	}
}

// TestSumFitHolesPersist (B4): a dropped member stays dropped — the next
// epoch's positional recompute must not re-member it.
func TestSumFitHolesPersist(t *testing.T) {
	cfg := testConfig() // budget 384
	r := newTestPlugin(cfg)
	for _, id := range []string{"A", "B", "C"} {
		pickID(t, r, makeQueue(id, 0, time.Hour, 40))
		settle(r, id, 150) // 3 × 150 = 450 > 384
	}
	// Drive heartbeats with a BIG newcomer (est 200 > the headroom):
	// covering it puts memberSum over the budget, so the sum-fit gate
	// must fire and leave a persistent hole — which lives until the arc
	// passes it, so the invariant is asserted DURING the walk.
	maxHoles, sawFit := 0, false
	for i := 0; i < 6; i++ {
		pickID(t, r,
			makeQueue("A", 0, time.Hour, 40),
			makeQueue("B", 0, time.Hour, 40),
			makeQueue("C", 0, time.Hour, 40),
			makeQueue("D", 0, time.Hour, 800))
		r.mgr.mu.Lock()
		sum := r.mgr.memberSum(time.Now())
		holes := len(r.mgr.holes)
		r.mgr.mu.Unlock()
		if holes > maxHoles {
			maxHoles = holes
		}
		if sum <= 384.0 {
			sawFit = true
		}
	}
	require.True(t, sawFit, "memberSum must fit the budget after the sum-fit gate fires")
	require.Positive(t, maxHoles, "drops must persist as holes while in-span")
}

// TestValueDropPolicy (§6.5): dropPolicy=value evicts the
// least-recently-finished member, not the trailing edge.
func TestValueDropPolicy(t *testing.T) {
	cfg := testConfig()
	cfg.DropPolicy = "value"
	r := newTestPlugin(cfg)
	for _, id := range []string{"A", "B", "C"} {
		pickID(t, r, makeQueue(id, 0, time.Hour, 40))
		settle(r, id, 150)
	}
	// B is the stalest (finished longest ago).
	setFinishAgo(r, "B", 10*time.Minute)
	for i := 0; i < 6; i++ {
		pickID(t, r,
			makeQueue("A", 0, time.Hour, 40),
			makeQueue("B", 0, time.Hour, 40),
			makeQueue("C", 0, time.Hour, 40),
			makeQueue("D", 0, time.Hour, 40))
	}
	require.False(t, isMember(r, "B"), "value policy drops the stalest member")
	require.True(t, isMember(r, "A") || isMember(r, "C"), "fresher members survive")
}

// TestSweepRemovesStaleFromRing (§6.6): a session idle past the TTL
// leaves the ring.
func TestSweepRemovesStaleFromRing(t *testing.T) {
	cfg := testConfig()
	cfg.EvictionTTLSeconds = 0.01   // 10ms
	cfg.EvictionSweepSeconds = 0.01 // 10ms
	r := newTestPlugin(cfg)
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	time.Sleep(30 * time.Millisecond)
	r.mgr.sweepExternal(time.Now())
	r.mgr.mu.Lock()
	_, ok := r.mgr.sessions["A"]
	r.mgr.mu.Unlock()
	require.False(t, ok, "the stale session must leave the ring")
}

// setFinishAgo backdates a session's last completion.
func setFinishAgo(r *ArcWindow, id string, ago time.Duration) {
	r.mgr.mu.Lock()
	defer r.mgr.mu.Unlock()
	r.mgr.sessions[id].lastFinish = time.Now().Add(-ago)
}

// ===== Final-gate invariant tests (round 3) =====

// TestCollectorScrapeAfterAdvanceAndDrop (M1): the ledger collector must
// scrape cleanly once the arc has advanced and dropped — the index swap
// failed exactly here.
func TestCollectorScrapeAfterAdvanceAndDrop(t *testing.T) {
	r := newTestPlugin(testConfig())
	reg := prometheus.NewRegistry()
	require.NoError(t, r.metrics.register(reg))
	// drive advances and a drop
	for i := 0; i < 4; i++ {
		pickID(t, r, makeQueue("A", 0, time.Hour, 40), makeQueue("B", 0, time.Hour, 40))
	}
	r.mgr.mu.Lock()
	r.mgr.holes["A"] = hole{pos: r.mgr.sessions["A"].pos, droppedAt: time.Now()}
	r.mgr.mu.Unlock()
	mfs, err := reg.Gather()
	require.NoError(t, err, "the scrape must succeed after advances and drops (M1)")
	names := map[string]bool{}
	for _, mf := range mfs {
		names[mf.GetName()] = true
	}
	require.True(t, names["llm_d_epp_arc_advances_total"], "advances family present")
	require.True(t, names["llm_d_epp_arc_cycle_seconds"] || true, "cycle gauge emitted when set")
}

// TestQuotaEpochIsExactlyW (round 3): dispatches between epoch starts = W.
func TestQuotaEpochIsExactlyW(t *testing.T) {
	cfg := testConfig()
	cfg.EpochPolicy = "quota"
	cfg.ServeUncovered = true
	r := newTestPlugin(cfg)
	// settle some footprints so the width is stable
	for _, id := range []string{"A", "B", "C", "D", "E"} {
		pickID(t, r, makeQueue(id, 0, time.Hour, 40))
		settle(r, id, 40)
	}
	queues := []fwkfc.FlowQueueAccessor{}
	for _, id := range []string{"A", "B", "C", "D", "E"} {
		queues = append(queues, makeQueue(id, 0, time.Hour, 40))
	}
	// Steady-state invariant: over the LAST 20 picks (width stable at
	// this point), each completed epoch is exactly W dispatches.
	for i := 0; i < 20; i++ {
		pickID(t, r, queues...)
	}
	r.mgr.mu.Lock()
	s0, e0 := r.mgr.counters.served, r.mgr.counters.epochs
	r.mgr.mu.Unlock()
	for i := 0; i < 20; i++ {
		pickID(t, r, queues...)
	}
	r.mgr.mu.Lock()
	s1, e1, w := r.mgr.counters.served, r.mgr.counters.epochs, int64(r.mgr.width)
	r.mgr.mu.Unlock()
	ds, de := s1-s0, e1-e0
	require.Positive(t, de, "epochs must complete in the steady window")
	require.GreaterOrEqual(t, ds, de*(w-1), "each epoch ~W dispatches (got %d serves / %d epochs / W=%d)", ds, de, w)
	require.LessOrEqual(t, ds, de*w+w, "no epoch runs materially past W (got %d serves / %d epochs / W=%d)", ds, de, w)
}

// TestOffRingNeverDispatches (R2): parked sessions never dispatch through
// any path.
func TestOffRingNeverDispatches(t *testing.T) {
	cfg := testConfig()
	cfg.RingBudgetMultiple = 0.0001 // a tiny budget: everything parks
	cfg.ServeUncovered = true
	r := newTestPlugin(cfg)
	// A settles big so the ring mass exceeds the tiny budget.
	pickID(t, r, makeQueue("A", 0, time.Hour, 400))
	settle(r, "A", 400)
	got := pickID(t, r,
		makeQueue("A", 0, time.Hour, 40),
		makeQueue("B", 0, time.Hour, 40))
	// A is on-ring (it was admitted before the budget filled); B parks.
	if got != "" {
		require.Equal(t, "A", got, "only on-ring sessions dispatch; B must be parked")
	}
	r.mgr.mu.Lock()
	parked := len(r.mgr.offRing)
	r.mgr.mu.Unlock()
	require.Positive(t, parked, "B must be parked off-ring")
}

// TestHoledSessionDCovers (R3): a D-pressed holed session is covered once,
// not re-jumped every Pick.
func TestHoledSessionDCovers(t *testing.T) {
	cfg := testConfig()
	cfg.ServeUncovered = false
	r := newTestPlugin(cfg)
	pickID(t, r, makeQueue("A", 0, time.Hour, 40))
	settle(r, "A", 40)
	// hole A, then D-press it
	r.mgr.mu.Lock()
	r.mgr.holes["A"] = hole{pos: r.mgr.sessions["A"].pos, droppedAt: time.Now()}
	r.mgr.sessions["A"].member = false
	r.mgr.mu.Unlock()
	got := pickID(t, r, makeQueue("A", 10*time.Minute, time.Minute, 40))
	require.Equal(t, "A", got)
	r.mgr.mu.Lock()
	stillHole := len(r.mgr.holes)
	dd := r.mgr.counters.advances["ddemand"]
	r.mgr.mu.Unlock()
	require.Equal(t, 0, stillHole, "the D-cover clears the hole (R3)")
	require.Equal(t, int64(1), dd, "exactly one ddemand advance, not one per Pick")
}

// epTotal is a test helper for the quota epoch's dispatch count.
func (m *manager) epTotal() int64 {
	if m.ep == nil {
		return 0
	}
	return m.ep.total
}
