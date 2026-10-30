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

// PLUGIN-LEVEL MULTI-POD TESTS: the parts of the split that live ABOVE the
// scheduler — endpoint attribution through PreRequest/ResponseBody, the
// per-endpoint capacity acquisition path, the affinity-violation metric, and
// the EventDelete teardown. The scheduler's own behaviour is covered in
// multipod_test.go.

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// mpHarness builds a per-engine plugin with two discovered endpoints, each
// carrying its own capacity.
// mpEngineKeys are the group ids the two harness endpoints resolve to (an
// endpoint id is <namespace>/<name>).
type mpEngineKeys struct{ A, B string }

func mpHarness(t *testing.T) (*GroupScheduler, *fwkdl.ModelServer, *fwkdl.ModelServer, mpEngineKeys) {
	t.Helper()
	p, err := Factory("gs", mpDecoder(t, `{"engineGroups":"per-engine"}`), nil)
	require.NoError(t, err)
	gs := p.(*GroupScheduler)
	require.NoError(t, gs.metrics.register(prometheus.NewRegistry()))

	a := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-a"}},
		&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	b := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-b"}},
		&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: a}))
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: b}))
	keys := mpEngineKeys{A: endpointKey(a), B: endpointKey(b)}
	require.True(t, gs.sched.HasCapacityFor(keys.A), "engine A key %q", keys.A)
	require.True(t, gs.sched.HasCapacityFor(keys.B), "engine B key %q", keys.B)
	return gs, a, b, keys
}

// mpRequest builds a request carrying a session attribute and a tokenized
// body (the tier-A producer's output — PreRequest fails closed without it).
func mpRequest(session, reqID string, blocks int) *fwksched.InferenceRequest {
	ids := make([]uint32, 0, blocks*64)
	for i := 0; i < blocks*64; i++ {
		ids = append(ids, uint32(i+1))
	}
	ir := &fwksched.InferenceRequest{RequestID: reqID, RequestSizeBytes: int(bytesFor(blocks))}
	ir.PutAttribute(agentidentity.AgentIdentityKey, session)
	ir.Body = &fwkrh.InferenceRequestBody{TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{ids})}
	return ir
}

// mpSchedEndpoint wraps a discovered endpoint as the scheduling-facing
// Endpoint the plugin reads back at PreRequest (same id, same metrics).
func mpSchedEndpoint(ms *fwkdl.ModelServer) fwksched.Endpoint {
	return fwksched.NewEndpoint(ms.GetMetadata(), ms.GetMetrics(), nil)
}

// mpResult builds a SchedulingResult whose picked endpoint is ep.
func mpResult(ep fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		},
	}
}

// TestPluginPerEndpointCapacityLiveness: the plugin learns each engine's
// capacity from ITS OWN metrics — a second engine must not inherit the
// first's, and the pre-capacity FCFS gate must stay closed only while NO
// engine has capacity.
func TestPluginPerEndpointCapacityLiveness(t *testing.T) {
	p, err := Factory("gs", mpDecoder(t, `{"engineGroups":"per-engine"}`), nil)
	require.NoError(t, err)
	gs := p.(*GroupScheduler)

	a := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-a"}}, nil)
	b := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-b"}}, nil)
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: a}))
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: b}))
	require.False(t, gs.sched.HasCapacity(), "neither engine has metrics yet")

	a.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	gs.pollCapacity(time.Now())
	ka, kb := endpointKey(a), endpointKey(b)
	require.True(t, gs.sched.HasCapacityFor(ka))
	require.False(t, gs.sched.HasCapacityFor(kb),
		"engine-b must NOT inherit engine-a's capacity (the single-pool 'first wins' bug)")

	// engine-b's own scrape arrives late and teaches ITS geometry
	b.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 800})
	gs.pollCapacity(time.Now())
	require.True(t, gs.sched.HasCapacityFor(kb))
	snap := gs.sched.SnapshotLite()
	require.Equal(t, 950, snap.Engines[ka].BudgetBlocks)
	require.Equal(t, 760, snap.Engines[kb].BudgetBlocks)
}

// TestPluginAffinityViolationThroughPreRequest: Pick charges the seat to the
// member's homed engine; PreRequest observes the engine the framework ACTUALLY
// picked and, when they disagree, the observed endpoint WINS (rebind), with
// the violation counted in the metric and the charge moved.
func TestPluginAffinityViolationThroughPreRequest(t *testing.T) {
	gs, a, b, keys := mpHarness(t)
	now := time.Now()

	// seat "sess-1" through Pick so it has a homing
	engine, i := gs.sched.PickFromHeadsEngine([]Head{{Member: "sess-1", ReqID: "r1",
		Arrival: now, SizeBytes: bytesFor(10)}}, now)
	require.GreaterOrEqual(t, i, 0)
	require.NotEmpty(t, engine)
	bound := gs.sched.bindingOf("sess-1")
	require.Equal(t, engine, bound)

	// the framework places the request on the OTHER engine
	other := a
	if bound == keys.A {
		other = b
	}
	req := mpRequest("sess-1", "r1", 10)
	require.NoError(t, gs.PreRequest(t.Context(), req, mpResult(mpSchedEndpoint(other))))

	require.Equal(t, endpointKey(other), gs.sched.bindingOf("sess-1"),
		"the OBSERVED endpoint must win (unseat-and-rebind)")
	require.EqualValues(t, 1, gs.sched.SnapshotLite().AffinityViolations)
	require.EqualValues(t, 1, gs.sched.SnapshotLite().Rebinds)
	from, to := engineLabel(bound), engineLabel(endpointKey(other))
	require.EqualValues(t, 1, promtestutil.ToFloat64(gs.metrics.affinityViolations.WithLabelValues(from, to)))
	// and the charge moved with it: the new engine holds the member's blocks
	moved := gs.sched.SnapshotLite().Engines[to]
	require.Greater(t, moved.NEBlocks, 0, "the exact keys must land on the engine that serves the request")
}

// TestPluginEventDeleteTearsDownEngine: a decode endpoint delete (the wave
// protocol rotates nodes every arm) must tear its group down — members exited,
// homing released — instead of leaving charges on a dead engine.
func TestPluginEventDeleteTearsDownEngine(t *testing.T) {
	gs, _, b, _ := mpHarness(t)
	now := time.Now()
	var heads []Head
	for _, id := range []string{"s1", "s2", "s3", "s4"} {
		heads = append(heads, Head{Member: id, ReqID: id + "-r", Arrival: now, SizeBytes: bytesFor(10)})
	}
	gs.sched.Tick(now)
	for len(heads) > 0 {
		engine, i := gs.sched.PickFromHeadsEngine(heads, now)
		require.GreaterOrEqual(t, i, 0)
		gs.sched.DispatchedOn(heads[i].Member, heads[i].ReqID, engine, keys(10, Hash(i+1)), bytesFor(10), now)
		heads = append(heads[:i], heads[i+1:]...)
	}
	require.Len(t, gs.sched.SnapshotLite().Engines, 2)

	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: b}))
	snap := gs.sched.SnapshotLite()
	require.NotContains(t, snap.Engines, endpointKey(b), "the torn-down engine must leave the group set")
	require.EqualValues(t, 1, snap.EngineGone)
	for member, engine := range gs.sched.bindingsCopy() {
		require.NotEqual(t, endpointKey(b), engine, "member %s is still bound to the dead engine", member)
	}
	require.EqualValues(t, 1, promtestutil.ToFloat64(gs.metrics.engineGone))
}

// TestPluginSettleCarriesEndpoint: ResponseBody's EndpointMetadata attributes
// the settle to the engine it ran on (F6) — an off-binding settle is counted,
// never silently folded into the binding's record.
func TestPluginSettleCarriesEndpoint(t *testing.T) {
	gs, a, b, keys := mpHarness(t)
	now := time.Now()
	engine, i := gs.sched.PickFromHeadsEngine([]Head{{Member: "sess-9", ReqID: "r9",
		Arrival: now, SizeBytes: bytesFor(10)}}, now)
	require.GreaterOrEqual(t, i, 0)
	req := mpRequest("sess-9", "r9", 10)
	onEngine := a
	if engine == keys.A {
		onEngine = b
	}
	// PreRequest on the OTHER engine rebinds the member there
	require.NoError(t, gs.PreRequest(t.Context(), req, mpResult(mpSchedEndpoint(onEngine))))
	before := gs.sched.SnapshotLite().AffinityViolations

	// the response reports the same endpoint: no further violation
	resp := &fwkrc.Response{EndOfStream: true, Usage: fwkrh.Usage{PromptTokens: 640, PromptTokenDetails: &fwkrh.PromptTokenDetails{CachedTokens: 600}}}
	gs.ResponseBody(t.Context(), req, resp, onEngine.GetMetadata())
	require.Equal(t, before, gs.sched.SnapshotLite().AffinityViolations,
		"a settle on the bound engine is not a violation")

	// a settle that contradicts the binding IS counted (the engine the request
	// really ran on cannot be reconciled silently)
	other := a
	if gs.sched.bindingOf("sess-9") == keys.A {
		other = b
	}
	req2 := mpRequest("sess-9", "r9b", 10)
	require.NoError(t, gs.PreRequest(t.Context(), req2, mpResult(mpSchedEndpoint(onEngine))))
	gs.ResponseBody(t.Context(), req2, resp, other.GetMetadata())
	require.Equal(t, before+1, gs.sched.SnapshotLite().AffinityViolations)
}

// mpDecoder builds the JSON decoder a Factory expects.
func mpDecoder(t *testing.T, params string) *json.Decoder {
	t.Helper()
	return json.NewDecoder(strings.NewReader(params))
}

// TestPluginPooledEndpointsAlarm is AUDIT N1: engineGroups defaults to "single",
// which fails OPEN — a second discovered endpoint silently pools independent
// engines into one budget/index. The state must be visible as a SERIES for every
// arm (0/absent when per-engine), because a startup log line scrolls away.
func TestPluginPooledEndpointsAlarm(t *testing.T) {
	p, err := Factory("gs", mpDecoder(t, `{}`), nil) // engineGroups defaults to single
	require.NoError(t, err)
	gs := p.(*GroupScheduler)
	require.NoError(t, gs.metrics.register(prometheus.NewRegistry()))

	a := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-a"}},
		&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: a}))
	require.EqualValues(t, 1, promtestutil.ToFloat64(gs.metrics.pooledEndpoints),
		"one endpoint pooled: expected, and the series must exist so the field is never ambiguous")
	b := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-b"}},
		&fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 1000})
	require.NoError(t, gs.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: b}))
	require.EqualValues(t, 2, promtestutil.ToFloat64(gs.metrics.pooledEndpoints),
		"two endpoints pooled under engineGroups=single — the fail-open state is now a series")
	require.Len(t, gs.sched.EngineGroupIDs(), 1, "and it really is ONE group")
	require.Equal(t, 950, gs.sched.SnapshotLite().BudgetBlocks,
		"the pooled budget is first-wins, i.e. engine-b's 1000 blocks are invisible")

	// per-engine mode reports 0 (no pooling at all)
	p2, err := Factory("gs", mpDecoder(t, `{"engineGroups":"per-engine"}`), nil)
	require.NoError(t, err)
	gs2 := p2.(*GroupScheduler)
	require.NoError(t, gs2.metrics.register(prometheus.NewRegistry()))
	require.NoError(t, gs2.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: a}))
	require.NoError(t, gs2.Extract(t.Context(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: b}))
	require.EqualValues(t, 0, promtestutil.ToFloat64(gs2.metrics.pooledEndpoints))
	require.Len(t, gs2.sched.EngineGroupIDs(), 2)
}
