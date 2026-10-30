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

// WAVE-1 tests (PREREG-branch-identity §WAVE-1, BRANCH-IDENTITY-DESIGN
// §V2 BASE-MODE FRAMING): the member/session separation, the D1
// session-ordering layer (with its GOLDEN EQUALITY invariant), the
// no-starvation property, and the capacity metrics-poll fallback.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// ---- 1. THE GOLDEN EQUALITY (the pre-registered invariant) ----

// TestSessionOrderGoldenEquality is THE binding invariant of the session
// layer: with session := branch — i.e. no session attribute on any request,
// which is exactly base mode — every sessionOrder mode must produce a trace
// BYTE-IDENTICAL to today's (newcomer-first, oldest, id) order. The full
// golden scenario is replayed under each mode and compared to the default
// config's trace, so any drift in seats, dispatches, exits, exit reasons or
// gauges fails here.
//
// The companion assertions make the test non-vacuous:
//   - "off" is INSENSITIVE to the session attribute (a stage-1-style config
//     that sends a session header with the layer disabled must not move a
//     single decision);
//   - "newcomer" and "lsf" DO change the trace when sessions group several
//     chains (the layer is live, not a no-op).
func TestSessionOrderGoldenEquality(t *testing.T) {
	base := runGoldenScenario(t)

	for _, mode := range []string{"newcomer", "lsf"} {
		t.Run(mode+" reduces to today with session:=branch", func(t *testing.T) {
			cfg := defaultConfig()
			cfg.SessionOrder = mode
			got := runGoldenScenarioCfg(t, 2600, cfg, nil) // nil = no session attribute
			requireTraceEqual(t, base, got)
		})
	}

	// grouped sessions: chains s00..s45 fall into threes under one session
	grouped := func(id string) string {
		var n int
		if _, err := fmt.Sscanf(id, "s%02d", &n); err != nil {
			t.Fatalf("unexpected chain id %q: %v", id, err)
		}
		return fmt.Sprintf("g%02d", n/3)
	}

	t.Run("off ignores the session attribute", func(t *testing.T) {
		cfg := defaultConfig() // SessionOrder "off"
		got := runGoldenScenarioCfg(t, 2600, cfg, grouped)
		requireTraceEqual(t, base, got)
	})
	for _, mode := range []string{"newcomer", "lsf"} {
		t.Run(mode+" acts when sessions group chains", func(t *testing.T) {
			cfg := defaultConfig()
			cfg.SessionOrder = mode
			got := runGoldenScenarioCfg(t, 2600, cfg, grouped)
			require.NotEqual(t, base, got,
				"sessionOrder=%q changed nothing on grouped sessions — the layer is a no-op", mode)
		})
	}
}

// requireTraceEqual reports the first divergent line (a byte diff of a 2,600s
// trace is unreadable otherwise).
func requireTraceEqual(t *testing.T, want, got string) {
	t.Helper()
	if got == want {
		return
	}
	wl, gl := strings.Split(want, "\n"), strings.Split(got, "\n")
	for i := 0; i < len(wl) || i < len(gl); i++ {
		var a, b string
		if i < len(gl) {
			a = gl[i]
		}
		if i < len(wl) {
			b = wl[i]
		}
		if a != b {
			t.Fatalf("trace divergence at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
		}
	}
	t.Fatal("trace divergence (length)")
}

// ---- 2. THE LSF KEY LADDER ----

// entryOf returns the D1 entry head after syncing the given heads. The heads
// are deliberately too big to seat (2000 blocks vs the 950 budget), so the
// sync populates the waiter view and rotate can neither seat nor evict —
// the ordering decision is observed in isolation.
func entryOf(t *testing.T, s *Scheduler, heads []Head, now time.Time) string {
	t.Helper()
	s.PickFromHeads(heads, now) // sync the waiter view (no seat possible)
	s.mu.Lock()
	defer s.mu.Unlock()
	w := s.entryHeadLocked(nil)
	if w == nil {
		return ""
	}
	return w.id
}

// bigHead is a head that can never fit the 950-block budget (10 seeds per
// member keep the estimates distinct).
func bigHead(member, session string, ago time.Duration) Head {
	return Head{Member: member, Session: session, ReqID: "r-" + member,
		Arrival: time.Now().Add(-ago), SizeBytes: bytesFor(2000)}
}

// seatMember seats and settles one member (so it holds a SEAT but nothing
// in flight — the session-seat input of the LSF key).
func seatMember(t *testing.T, s *Scheduler, member, session string, now time.Time) {
	t.Helper()
	h := []Head{{Member: member, Session: session, ReqID: "r-" + member, Arrival: now, SizeBytes: bytesFor(10)}}
	require.Equal(t, 0, s.PickFromHeads(h, now), "%s must seat", member)
	s.Dispatched(member, "r-"+member, keys(10, Hash(len(member)*7+int(member[0]))), bytesFor(10), now)
	s.Chunk(member)
	s.Settle(member, "r-"+member, struct{ Prompt, Cached int }{Prompt: 10}, true, now)
}

// TestSessionOrderLSFLadder pins the pre-registered key — (session seats held
// ↑, session-is-new first, oldest arrival, id) — one rung at a time.
func TestSessionOrderLSFLadder(t *testing.T) {
	cfg := defaultConfig()
	cfg.SessionOrder = "lsf"
	s := newSchedCfg(t, cfg)
	now := time.Now()

	// session sA holds two seats, sB one.
	seatMember(t, s, "a1", "sA", now)
	seatMember(t, s, "a2", "sA", now)
	seatMember(t, s, "b1", "sB", now)

	// RUNG 1 — seats dominate age: sC holds none, so its newest branch goes
	// before sA's oldest branch.
	heads := []Head{
		bigHead("a3", "sA", 100*time.Second),
		bigHead("b2", "sB", 50*time.Second),
		bigHead("c1", "sC", time.Second),
	}
	require.Equal(t, "c1", entryOf(t, s, heads, now), "least-seats-first: sC (0 seats) must win")

	// RUNG 2 — session honours the layer: e1 is a served session that now
	// holds no seat (it paid out and rotated), d1 belongs to a session that
	// has never been seated. Equal seats ⇒ the NEW session goes first.
	seatMember(t, s, "e1", "sE", now)
	// e1 pays out: the entry cost is measured on the first settle, cached
	// credit on the second ⇒ ledger-eligible ("paid").
	h := []Head{{Member: "e1", Session: "sE", ReqID: "r-e1b", Arrival: now, SizeBytes: bytesFor(10)}}
	s.PickFromHeads(h, now)
	s.Dispatched("e1", "r-e1b", keys(10, 1), bytesFor(10), now)
	s.Chunk("e1")
	s.Settle("e1", "r-e1b", struct{ Prompt, Cached int }{Prompt: 10, Cached: 10}, true, now)
	// the blocked heads make rotation take the paid member out (and NOT
	// forget it: only the silence GC does that, so sE stays "served")
	heads = []Head{
		bigHead("e2", "sE", 100*time.Second), // served session, 0 seats now
		bigHead("d1", "sD", 2*time.Second),   // never-seated session
	}
	_ = entryOf(t, s, heads, now)
	s.mu.Lock()
	_, served := s.everSeatedSession["sE"]
	_, e2Seat := s.members["e1"]
	s.mu.Unlock()
	require.True(t, served, "sE must still count as served (its record keeps the session mark)")
	require.False(t, e2Seat, "e1 must have rotated out for this rung")
	require.Equal(t, "d1", entryOf(t, s, heads, now), "session-is-new must break the seat tie")

	// RUNG 3 — equal seats and equal session status ⇒ oldest arrival, then id.
	// sB and sA are both served and still hold seats, so compare two branches
	// of NEVER-seated sessions to isolate the age rung.
	heads = []Head{
		bigHead("n1", "sN1", 100*time.Second),
		bigHead("n2", "sN2", 2*time.Second),
	}
	require.Equal(t, "n1", entryOf(t, s, heads, now), "oldest arrival must break the tie")
	// RUNG 4 — identical arrivals: the id is the last tiebreak (a total order,
	// so map iteration can never leak into the decision).
	same := time.Now().Add(-5 * time.Second)
	heads = []Head{
		{Member: "z9", Session: "sZ9", ReqID: "rz9", Arrival: same, SizeBytes: bytesFor(2000)},
		{Member: "z1", Session: "sZ1", ReqID: "rz1", Arrival: same, SizeBytes: bytesFor(2000)},
	}
	require.Equal(t, "z1", entryOf(t, s, heads, now), "the id must break an exact tie")
}

// TestSessionOrderNewcomerMode: the newcomer CLASS moves to the session — a
// new branch of an already-served session queues FIFO behind a returning
// waiter instead of jumping it.
func TestSessionOrderNewcomerMode(t *testing.T) {
	cfg := defaultConfig()
	cfg.SessionOrder = "newcomer"
	s := newSchedCfg(t, cfg)
	now := time.Now()
	seatMember(t, s, "a1", "sA", now) // sA is served; a1 keeps a seat

	// a2 is a brand-new member but sA is served ⇒ it must NOT be a newcomer;
	// r's session is unknown to the layer, so r keeps its (>a1-seated) age.
	heads := []Head{
		bigHead("a2", "sA", 100*time.Second), // older, session served
		bigHead("r1", "sR", time.Second),     // newer, session never seated
	}
	require.Equal(t, "r1", entryOf(t, s, heads, now),
		"a served session's new branch must not jump a new session's entry")
}

// ---- 3. NO STARVATION (the pre-registered property) ----

// TestSessionOrderNoStarvation is the property test behind the pre-registration's
// "a waiter eventually seats under sustained arrivals (quantum rotation
// guarantees it)".
//
// The shape: session sH holds a permanent seat and keeps minting a NEW branch
// every second — each a fresh member, hence a newcomer under "off" — while W
// is a RETURNING waiter whose case predates all of them. Under "off" every
// seat that frees up goes to the freshest newcomer and W waits forever (the
// newcomer-jumping failure v1 and the ID arm both measured). Both session
// modes must seat W.
func TestSessionOrderNoStarvation(t *testing.T) {
	for _, mode := range []string{"off", "newcomer", "lsf"} {
		t.Run(mode, func(t *testing.T) {
			seatedAt := starvationRun(t, mode)
			if mode == "off" {
				require.Equal(t, -1, seatedAt,
					"the synthetic baseline must actually starve W — otherwise this test proves nothing")
				return
			}
			require.GreaterOrEqual(t, seatedAt, 0,
				"sessionOrder=%q starved the returning waiter for the whole window", mode)
			t.Logf("sessionOrder=%q seated the returning waiter at sim=%ds", mode, seatedAt)
		})
	}
}

// starvationRun drives the scenario and returns the sim second at which W got
// a seat, or -1.
func starvationRun(t *testing.T, mode string) int {
	t.Helper()
	cfg := defaultConfig()
	cfg.SessionOrder = mode
	s, err := NewScheduler(cfg)
	require.NoError(t, err)
	// budget 38 blocks: the permanent resident (10) + W (10) + one transient
	// (10) fill it, so entries compete for every seat that frees.
	require.NoError(t, s.SetNumBlocks(64, 40))

	t0 := time.Unix(1800000000, 0).UTC()
	// the permanent resident: dispatched once, never settled — it keeps its
	// seat (and sH's seat count ≥ 1) for the whole run.
	residentHead := func(now time.Time) []Head {
		return []Head{{Member: "res", Session: "sH", ReqID: "r-res", Arrival: now, SizeBytes: bytesFor(10)}}
	}
	if s.PickFromHeads(residentHead(t0), t0) != 0 {
		t.Fatalf("the resident must seat")
	}
	s.Dispatched("res", "r-res", keys(10, 99), bytesFor(10), t0)

	// W: served (so its session counts as served and it is a RETURNING
	// waiter), paid out, then rotated out into a waiter at its original
	// arrival — the shape exitLocked produces on a rotation.
	seatMember(t, s, "w1", "sW", t0)
	wh := []Head{{Member: "w1", Session: "sW", ReqID: "r-w1", Arrival: t0, SizeBytes: bytesFor(10)}}
	s.PickFromHeads(wh, t0)
	s.Dispatched("w1", "r-w1", keys(10, 7), bytesFor(10), t0)
	s.Chunk("w1")
	// the "paid" ledger entry: earned 100 against an entry cost of 10 clears
	// rho x entryCost, so W is rotation-eligible
	s.Settle("w1", "r-w1", struct{ Prompt, Cached int }{Prompt: 100, Cached: 100}, true, t0)
	// force the rotation: W's head is present (so the exit re-queues it at
	// t0) and an impossible entry blocks the seat.
	s.PickFromHeads([]Head{
		{Member: "w1", Session: "sW", ReqID: "r-w1", Arrival: t0, SizeBytes: bytesFor(10)},
		bigHead("blocker", "sBlk", time.Second),
	}, t0)
	if s.Seated("w1") {
		t.Fatalf("W must have been rotated out into a waiter for this scenario")
	}

	type flight struct {
		member, reqID string
		at            int
	}
	var flying []*flight

	const branchesPerSec = 3
	seatedAt := -1
	for sim := 1; sim <= 300 && seatedAt < 0; sim++ {
		now := t0.Add(time.Duration(sim) * time.Second)
		// settle whatever is due (fully warm ⇒ the ledger pays it out fast)
		still := flying[:0]
		for _, f := range flying {
			if f.at <= sim {
				s.Chunk(f.member)
				s.Settle(f.member, f.reqID, struct{ Prompt, Cached int }{Prompt: 10, Cached: 10}, true, now)
				continue
			}
			still = append(still, f)
		}
		flying = still

		// W's case is present from the start (the oldest arrival in the set)
		// and sH mints FAILURE_BRANCHES_PER_SEC new branches per second (fresh
		// members ⇒ newcomers under "off"), more than the freed seats can
		// absorb, so a newcomer backlog always exists.
		heads := []Head{
			{Member: "w1", Session: "sW", ReqID: fmt.Sprintf("rw-%d", sim), Arrival: t0, SizeBytes: bytesFor(10)},
		}
		for k := 0; k < branchesPerSec; k++ {
			id := fmt.Sprintf("h%d-%d", sim, k)
			heads = append(heads, Head{Member: id, Session: "sH", ReqID: "r-" + id,
				Arrival: now, SizeBytes: bytesFor(10)})
		}
		s.Tick(now)
		for round := 0; round < 3; round++ {
			i := s.PickFromHeads(heads, now)
			if i < 0 {
				break
			}
			member := heads[i].Member
			if member == "w1" {
				seatedAt = sim
				break
			}
			if s.Seated(member) {
				s.Dispatched(member, heads[i].ReqID, keys(10, Hash(sim)), bytesFor(10), now)
				flying = append(flying, &flight{member: member, reqID: heads[i].ReqID, at: sim + 1})
			}
		}
	}
	return seatedAt
}

// ---- 4. MEMBER-KEY / SESSION SEPARATION ----

// TestMemberKeyResolution pins the resolution order and — the standing rule —
// that nothing is ever WRITTEN into the agent-identity slot.
func TestMemberKeyResolution(t *testing.T) {
	build := func(t *testing.T, params string) *GroupScheduler {
		t.Helper()
		var dec *json.Decoder
		if params != "" {
			dec = json.NewDecoder(strings.NewReader(params))
		}
		p, err := Factory("gs", dec, nil)
		require.NoError(t, err)
		return p.(*GroupScheduler)
	}
	req := func(branch, session string) *fwksched.InferenceRequest {
		ir := &fwksched.InferenceRequest{RequestID: "r1"}
		if branch != "" {
			ir.PutAttribute(branchidentity.BranchIdentityKey, branch)
		}
		if session != "" {
			ir.PutAttribute(agentidentity.AgentIdentityKey, session)
		}
		return ir
	}

	for _, tc := range []struct {
		name    string
		params  string
		branch  string
		session string
		member  string
		src     string
		scope   string
	}{
		{"auto prefers the branch", `{"memberKey":"auto"}`, "br-1", "sess-1", "br-1", "branch", "sess-1"},
		{"auto falls back to the session", `{"memberKey":"auto"}`, "", "sess-1", "sess-1", "session", "sess-1"},
		{"auto with neither is anonymous (R9)", `{"memberKey":"auto"}`, "", "", "", "", ""},
		{"branch mode", `{"memberKey":"branch"}`, "br-1", "sess-1", "br-1", "branch", "sess-1"},
		{"branch mode without a branch is anonymous", `{"memberKey":"branch"}`, "", "sess-1", "", "", "sess-1"},
		{"session mode ignores the branch", `{"memberKey":"session"}`, "br-1", "sess-1", "sess-1", "session", "sess-1"},
		{"the default is auto", "", "br-1", "sess-1", "br-1", "branch", "sess-1"},
		{"the branch plugin's session copy is a fallback scope",
			`{"memberKey":"auto"}`, "br-1", "", "br-1", "branch", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gs := build(t, tc.params)
			ir := req(tc.branch, tc.session)
			member, scope, src := gs.memberOf(ir)
			require.Equal(t, tc.member, member)
			require.Equal(t, tc.src, src)
			require.Equal(t, tc.scope, scope)
			// the standing rule: the branch id never lands in the
			// agent-identity slot, and the session attribute keeps its value
			got, _ := fwksched.ReadRequestAttribute[string](ir, agentidentity.AgentIdentityKey)
			require.Equal(t, tc.session, got, "the agent-identity attribute must never be rewritten")
		})
	}

	t.Run("the branch plugin's session copy serves as the scope", func(t *testing.T) {
		gs := build(t, `{"memberKey":"auto"}`)
		ir := &fwksched.InferenceRequest{RequestID: "r1"}
		ir.PutAttribute(branchidentity.BranchIdentityKey, "br-1")
		ir.PutAttribute(branchidentity.BranchSessionKey, "sess-x")
		member, scope, src := gs.memberOf(ir)
		require.Equal(t, "br-1", member)
		require.Equal(t, "sess-x", scope)
		require.Equal(t, "branch", src)
	})
}

// TestMemberProducerGate: the configured member source must exist in the same
// config — otherwise every request is anonymous and Pick degrades to R9 FCFS
// (the Amendment-4 failure), silently.
func TestMemberProducerGate(t *testing.T) {
	build := func(t *testing.T, params string) *GroupScheduler {
		t.Helper()
		var dec *json.Decoder
		if params != "" {
			dec = json.NewDecoder(strings.NewReader(params))
		}
		p, err := Factory("gs", dec, testutils.NewTestHandle(t.Context()))
		require.NoError(t, err)
		return p.(*GroupScheduler)
	}
	branchPlugin := &stubPlugin{tn: fwkplugin.TypedName{Type: branchidentity.PluginType, Name: "bi"}}
	agentPlugin := &stubPlugin{tn: fwkplugin.TypedName{Type: agentidentity.PluginType, Name: "ai"}}

	t.Run("branch mode without the branch plugin fails", func(t *testing.T) {
		gs := build(t, `{"memberKey":"branch"}`)
		gs.handle.AddPlugin("ai", agentPlugin)
		require.Error(t, gs.validateMemberProducer())
	})
	t.Run("branch mode with it passes", func(t *testing.T) {
		gs := build(t, `{"memberKey":"branch"}`)
		gs.handle.AddPlugin("bi", branchPlugin)
		require.NoError(t, gs.validateMemberProducer())
	})
	t.Run("session mode without the agent-identity plugin fails", func(t *testing.T) {
		gs := build(t, `{"memberKey":"session"}`)
		gs.handle.AddPlugin("bi", branchPlugin)
		require.Error(t, gs.validateMemberProducer())
	})
	t.Run("session mode with it passes", func(t *testing.T) {
		gs := build(t, `{"memberKey":"session"}`)
		gs.handle.AddPlugin("ai", agentPlugin)
		require.NoError(t, gs.validateMemberProducer())
	})
	t.Run("auto needs at least one identity plugin", func(t *testing.T) {
		gs := build(t, "")
		require.Error(t, gs.validateMemberProducer())
		gs.handle.AddPlugin("ai", agentPlugin)
		require.NoError(t, gs.validateMemberProducer())
	})
}

// ---- 5. THE CAPACITY METRICS-POLL FALLBACK (+ startup liveness) ----

// TestCapacityPollFallback is the capacity race in one test: the endpoint's
// creation event carries NO metrics (the real ordering — the event fires
// before the first scrape), so the event path learns nothing; the datalayer
// then refreshes the metrics IN PLACE on the same endpoint object, and the
// ticker's poll picks them up.
func TestCapacityPollFallback(t *testing.T) {
	gs := newTestGroupScheduler(t, `{"capacityLivenessSeconds":120}`)
	// metrics collectors, so the liveness gauge is observable
	require.NoError(t, gs.metrics.register(prometheus.NewRegistry()))

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "decode-0"}}, nil)
	// the creation event: metrics exist but carry no cache geometry yet
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	require.False(t, gs.sched.HasCapacity(), "the creation event must not invent a capacity")

	// inside the liveness window nothing is flagged
	firstSeen := gs.firstSeen
	require.False(t, firstSeen.IsZero(), "the endpoint observation must be recorded")
	gs.pollCapacity(firstSeen.Add(30 * time.Second))
	require.False(t, gs.sched.HasCapacity())
	require.False(t, gs.capWarned, "no liveness warning inside capacityLivenessSeconds")
	require.Equal(t, 0.0, promtestutil.ToFloat64(gs.metrics.capacityMissing))

	// the scrape lands (the collector swaps the pointer on the SAME object)
	ep.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	gs.pollCapacity(firstSeen.Add(31 * time.Second))
	require.True(t, gs.sched.HasCapacity(), "the metrics poll must recover the capacity")
	require.Equal(t, 950, gs.sched.SnapshotLite().BudgetBlocks, "budget = 0.95 x num_gpu_blocks")
	require.Equal(t, "metrics-poll", gs.capFrom)
	require.Equal(t, 1.0, promtestutil.ToFloat64(gs.metrics.capacityLearned.WithLabelValues("metrics-poll")))
}

// TestCapacityLivenessAlarm: the budget stays unknown past the bound ⇒ the
// process says so loudly and raises the gauge (the failure was invisible in
// the outcome metrics: FCFS pass-through looks like a slow arm, not a bug).
func TestCapacityLivenessAlarm(t *testing.T) {
	gs := newTestGroupScheduler(t, `{"capacityLivenessSeconds":120}`)
	require.NoError(t, gs.metrics.register(prometheus.NewRegistry()))
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "decode-0"}}, nil)
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))

	gs.pollCapacity(gs.firstSeen.Add(121 * time.Second))
	require.True(t, gs.capWarned, "the liveness bound must fire the alarm")
	require.Equal(t, 1.0, promtestutil.ToFloat64(gs.metrics.capacityMissing))

	// the alarm is one-shot: the gauge stays up until the capacity arrives
	gs.pollCapacity(gs.firstSeen.Add(600 * time.Second))
	require.Equal(t, 1.0, promtestutil.ToFloat64(gs.metrics.capacityMissing))

	// a late metric clears it
	ep.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 512})
	gs.pollCapacity(gs.firstSeen.Add(601 * time.Second))
	require.True(t, gs.sched.HasCapacity())
	require.Equal(t, 0.0, promtestutil.ToFloat64(gs.metrics.capacityMissing))
}

// TestCapacityMismatchStaysLoud: a measured block size that disagrees with the
// token processor's is still refused on the poll path (no silent capacity).
func TestCapacityMismatchStaysLoud(t *testing.T) {
	gs := newTestGroupScheduler(t, "")
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "decode-0"}}, nil)
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	ep.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 128, CacheNumBlocks: 1000})
	gs.pollCapacity(time.Now())
	require.False(t, gs.sched.HasCapacity(), "a mismatched block size must not install a capacity")
	require.Error(t, gs.sched.BlockSizeMismatch())
}

// ---- 6. THE EVENT PATH STAYS (belt and braces) ----

// TestCapacityEventPathStillLearns: Extract keeps the original behaviour when
// the event DOES carry the geometry.
func TestCapacityEventPathStillLearns(t *testing.T) {
	gs := newTestGroupScheduler(t, "")
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "decode-0"}},
		&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 2000})
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
	require.True(t, gs.sched.HasCapacity())
	require.Equal(t, 1900, gs.sched.SnapshotLite().BudgetBlocks)
	require.Equal(t, "endpoint-event", gs.capFrom)
}

// ---- 7. +29% MEMBERS: do the per-member structures hold? ----

// TestPerMemberStructuresAtMoreMembers: base mode adds ~29% more members (reset
// splits) without removing any per-member bound, so the per-member and
// per-session structures are re-checked at that size: the derived-window chain
// cap (maxChains, per record), the per-member in-flight cap, the event ring and
// the record maps must all stay bounded, and the run must not freeze.
func TestPerMemberStructuresAtMoreMembers(t *testing.T) {
	for _, order := range []string{"off", "lsf"} {
		t.Run("sessionOrder="+order, func(t *testing.T) { checkPerMemberStructures(t, order) })
	}
}

func checkPerMemberStructures(t *testing.T, order string) {
	cfg := defaultConfig()
	cfg.WindowMode = "derived" // the mode whose per-record chain list is capped
	cfg.SessionOrder = order
	s := newSchedCfg(t, cfg)
	now := time.Now()

	const members = 190 // ~+29% over the golden scenario's 46... a 4x overshoot of the real delta
	const turnsPer = 12
	for turn := 0; turn < turnsPer; turn++ {
		for i := 0; i < members; i++ {
			id := fmt.Sprintf("m%03d", i)
			head := []Head{{Member: id, Session: id, ReqID: fmt.Sprintf("r%d", turn),
				Arrival: now, SizeBytes: bytesFor(10)}}
			if s.PickFromHeads(head, now) < 0 {
				continue // blocked by the budget (fine: it stays a waiter)
			}
			// a distinct fan-out chain per turn keeps the per-record chain
			// list growing (maxChains must cap it)
			chain := keys(40, Hash(turn*1000+i))
			s.Dispatched(id, fmt.Sprintf("r%d", turn), chain, bytesFor(10), now)
			s.Chunk(id)
			s.Settle(id, fmt.Sprintf("r%d", turn), struct{ Prompt, Cached int }{Prompt: 40, Cached: 30}, true, now)
		}
		s.Tick(now)
	}

	snap := s.Snapshot()
	require.LessOrEqual(t, len(snap.RecentEvents), eventRingSize, "the event ring must stay bounded")
	require.LessOrEqual(t, s.Sessions(), members, "the index must not accumulate phantom records")
	require.NotEmpty(t, snap.MemberSources, "the effective member source must be counted")
	s.mu.Lock()
	defer s.mu.Unlock()
	s.index.mu.Lock()
	for id, rec := range s.index.sessions {
		require.LessOrEqual(t, len(rec.chains), maxChains, "record %s exceeded the chain cap", id)
	}
	s.index.mu.Unlock()
	for id, m := range s.members {
		require.LessOrEqual(t, m.inflight, cfg.InFlightCap, "member %s exceeded the in-flight cap", id)
	}
	// the session mirror is exact: every everSeated record contributes exactly
	// one session ref, and every referenced session is marked served.
	sum := 0
	for sess, n := range s.sessionRefs {
		require.Positive(t, n, "session %s has a non-positive refcount", sess)
		require.Contains(t, s.everSeatedSession, sess, "session %s holds refs but is not marked served", sess)
		sum += n
	}
	require.Equal(t, len(s.everSeated), sum, "session refs must total the everSeated records")
	require.Equal(t, len(s.everSeated), len(s.recordSession), "recordSession must mirror everSeated")
}

// ---- 8. THE WIRING ON THE LIVE PATH ----

// TestIntegrationMemberIsBranchSessionIsAttribute drives a base-mode request
// (branch queue key + a session attribute) through the REAL flow controller and
// checks the split lands where the design says: the member/seat is the BRANCH,
// and the session attribute is carried alongside (readable by the session
// layer, never the member).
func TestIntegrationMemberIsBranchSessionIsAttribute(t *testing.T) {
	plugin, err := Factory("gs", nil, nil)
	require.NoError(t, err)
	gs := plugin.(*GroupScheduler)
	require.NoError(t, gs.sched.SetNumBlocks(64, 1000))

	fc, _ := newIntegrationHarness(t, plugin)

	// the request as the director hands it over in base mode: the fairness id
	// (queue key) is the derived branch; the agent-identity slot keeps the
	// client's session.
	ir := &fwksched.InferenceRequest{RequestID: "b-1", RequestSizeBytes: int(bytesFor(100))}
	ir.PutAttribute(branchidentity.BranchIdentityKey, "br-7")
	ir.PutAttribute(agentidentity.AgentIdentityKey, "sess-7")
	req := &intRequest{id: "b-1", key: flowcontrol.FlowKey{ID: "br-7"}, size: uint64(bytesFor(100)), infReq: ir}

	done := make(chan error, 1)
	go func() {
		_, err := fc.EnqueueAndWait(t.Context(), req)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err, "the base-mode request must dispatch")
	case <-time.After(5 * time.Second):
		t.Fatal("the base-mode request never dispatched")
	}

	require.True(t, gs.sched.Seated("br-7"), "the BRANCH must be the seated member")
	require.False(t, gs.sched.Seated("sess-7"), "the session must never be seated as a member")
	gs.sched.mu.Lock()
	sess := gs.sched.members["br-7"].session
	gs.sched.mu.Unlock()
	require.Equal(t, "sess-7", sess, "the member must carry the session ATTRIBUTE for the ordering layer")
	// and the attribute itself is untouched (nothing writes the branch there)
	got, _ := fwksched.ReadRequestAttribute[string](ir, agentidentity.AgentIdentityKey)
	require.Equal(t, "sess-7", got)
}

// ---- helpers ----

// newTestGroupScheduler builds the plugin WITHOUT a handle (no ticker, no
// metrics registration — the test drives pollCapacity directly).
func newTestGroupScheduler(t *testing.T, params string) *GroupScheduler {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(strings.NewReader(params))
	}
	p, err := Factory("gs", dec, nil)
	require.NoError(t, err)
	return p.(*GroupScheduler)
}
