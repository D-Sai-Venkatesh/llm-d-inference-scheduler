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

// PLUGIN-LEVEL TESTS: the framework-facing surface (detector, Filter, capacity
// learning, the live-metrics rule) plus the config/emitter guard.

import (
	"context"
	"testing"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	"k8s.io/apimachinery/pkg/types"
)

// ---- framework test doubles ----

// liveEndpoint is ONE datalayer endpoint (the type the DETECTOR receives). It is a
// ModelServer, i.e. the mutable object the datastore updates in place.
func liveEndpoint(id string, kv float64, waiting, blockSize, numBlocks int, at time.Time) *fwkdl.ModelServer {
	meta := &fwkdl.EndpointMetadata{Name: id, Address: id + ":8080",
		ID: types.NamespacedName{Namespace: "ns", Name: id}}
	return fwkdl.NewEndpoint(meta, &fwkdl.Metrics{
		KVCacheUsagePercent: kv, WaitingQueueSize: waiting,
		CacheBlockSize: blockSize, CacheNumBlocks: numBlocks, UpdateTime: at,
	})
}

// schedulerEndpoint is the SCHEDULING-layer endpoint (the type the Filter takes). NOTE:
// fwksched.NewEndpoint CLONES the metrics it is given — which is exactly the property
// that makes a remembered endpoint handle a FROZEN input (the E-wave latch), and why
// nothing in the core reads policy inputs from one.
func schedulerEndpoint(id string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: id, Address: id + ":8080",
		ID: types.NamespacedName{Namespace: "ns", Name: id}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, CacheBlockSize: 64, CacheNumBlocks: 1000}, nil)
}

// livePods is the detector's input: live datalayer endpoints.
func livePods(eps ...*fwkdl.ModelServer) []fwkdl.Endpoint {
	out := make([]fwkdl.Endpoint, 0, len(eps))
	for _, e := range eps {
		out = append(out, e)
	}
	return out
}

// podsOf is the Filter's input: scheduling-layer endpoints.
func podsOf(eps ...fwksched.Endpoint) []fwksched.Endpoint { return eps }

// requestFor builds an InferenceRequest carrying the derived-branch attribute (the
// core's identity input) — "" means anonymous.
func requestFor(member string) *fwksched.InferenceRequest {
	ir := &fwksched.InferenceRequest{RequestID: "req-" + member, RequestSizeBytes: 4096}
	if member != "" {
		ir.PutAttribute(branchidentity.BranchIdentityKey, member)
	}
	return ir
}

// newTestPlugin builds the plugin around a scheduler with n engines of `blocks`
// capacity each (no metrics registrar: unit mode).
func newTestPlugin(t *testing.T, cfg Config, n, blocks int) *GroupScheduler {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := s.SetNumBlocksFor(engineID(i), cfg.BlockSizeTokens, blocks); err != nil {
			t.Fatal(err)
		}
	}
	p := &GroupScheduler{typedName: fwkplugin.TypedName{Type: PluginType, Name: "gs"},
		cfg: cfg, sched: s, metrics: newGroupMetrics(s), pins: newPinTable(),
		endpoints: map[string]*endpointState{}, hetero: map[string]struct{}{}}
	// the pin table follows the seats, exactly as Factory wires it
	s.onBind = func(member, engine string) { p.pins.set(member, engine) }
	s.onUnbind = func(member string) { p.pins.del(member) }
	return p
}

// engineID is the engine id the core will see for the i-th test endpoint: the
// endpoint's own datalayer key (namespace/name) — the same value endpointKey yields.
func engineID(i int) string { return "ns/" + string(rune('a'+i)) }

// ---- the detector ----

// The DEFAULT detector answer is "not saturated": the seat budget is the admission
// control, so the flow controller's pool gate never closes on this policy.
func TestDetectorDefaultsToNotSaturated(t *testing.T) {
	for _, n := range []int{1, 2} {
		p := newTestPlugin(t, defaultConfig(), n, 1000)
		now := time.Now()
		eps := make([]*fwkdl.ModelServer, 0, n)
		for i := 0; i < n; i++ {
			eps = append(eps, liveEndpoint(engineID(i), 0.99, 500, 64, 1000, now))
		}
		if got := p.Saturation(context.Background(), livePods(eps...)); got != 0 {
			t.Fatalf("N=%d: the default detector must answer 0 (not saturated), got %v", n, got)
		}
		// it still SAMPLES what it read on the ticker (the instrumentation has to exist
		// even when nothing is gated: a run states what would have happened)
		p.pushEngineLoad(time.Now())
		if got := len(p.liveSamples()); got != n {
			t.Fatalf("N=%d: the detector must publish one sample per engine, got %d", n, got)
		}
	}
}

// The memory-only ALTERNATIVE: pool score = MIN over engines of kv/threshold; stale or
// missing metrics are OPEN (excluded); the threshold is the budget's own fraction.
func TestDetectorMemoryOnlyAlternative(t *testing.T) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction"
	p := newTestPlugin(t, cfg, 2, 1000)
	now := time.Now()
	// one engine at half the threshold, one at the threshold: MIN is the half one
	got := p.Saturation(context.Background(), livePods(
		liveEndpoint("a", cfg.GroupBudgetFraction/2, 0, 64, 1000, now),
		liveEndpoint("b", cfg.GroupBudgetFraction, 900, 64, 1000, now)))
	want := 0.5
	if got < want-1e-9 || got > want+1e-9 {
		t.Fatalf("the pool score must be the MIN over engines (%v), got %v", want, got)
	}
	// a STALE sample is excluded: the fresh half-filled engine decides, and a fully
	// stale pool reads 0 (OPEN)
	stale := now.Add(-time.Hour)
	if got := p.Saturation(context.Background(), livePods(
		liveEndpoint("a", cfg.GroupBudgetFraction/2, 0, 64, 1000, now),
		liveEndpoint("b", 1.0, 900, 64, 1000, stale))); got < want-1e-9 || got > want+1e-9 {
		t.Fatalf("a stale engine must be excluded (OPEN), got %v", got)
	}
	if got := p.Saturation(context.Background(), livePods(
		liveEndpoint("a", 1.0, 0, 64, 1000, stale))); got != 0 {
		t.Fatalf("an all-stale pool must read 0 (open), got %v", got)
	}
}

// THE CLONE REGRESSION (the latch class): the policy's engine-state input must change
// with NO dispatch. The detector reads the LIVE endpoint objects the framework passes
// in, so a metric that moves between two Saturation calls is seen — even though the
// endpoint object a plugin could have remembered at construction carries a frozen
// clone.
func TestLiveMetricsUpdateWithoutADispatch(t *testing.T) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction"
	p := newTestPlugin(t, cfg, 1, 1000)
	now := time.Now()

	// the FROZEN case, for contrast: the clone NewEndpoint took never moves
	frozen := &fwkdl.Metrics{KVCacheUsagePercent: 0.10, CacheBlockSize: 64, CacheNumBlocks: 1000, UpdateTime: now}
	frozenEp := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: "a",
		ID: types.NamespacedName{Namespace: "ns", Name: "a"}}, frozen, nil)
	frozen.KVCacheUsagePercent = 0.99 // mutating the ORIGINAL does not move the clone
	if got := frozenEp.GetMetrics().KVCacheUsagePercent; got != 0.10 {
		t.Fatalf("the framework's endpoint clone is expected to be frozen; got %v (the core's live-read rule assumes it)", got)
	}

	// the LIVE case: the datastore hands fresh endpoint objects; the verdict follows
	lo := p.Saturation(context.Background(), livePods(liveEndpoint("a", 0.10, 0, 64, 1000, time.Now())))
	if lo >= 1 {
		t.Fatalf("kv 0.10 must not saturate, got %v", lo)
	}
	hi := p.Saturation(context.Background(), livePods(liveEndpoint("a", 0.99, 0, 64, 1000, time.Now())))
	if hi < 1 {
		t.Fatalf("kv 0.99 at a 0.95 budget fraction MUST saturate, got %v", hi)
	}
	// ... and the sample the ticker pushes carries that change, with no Pick, no
	// dispatch and no settle in between
	p.pushEngineLoad(time.Now())
	st := p.sched.Snapshot().Engines[engineLabel(engineID(0))]
	if !st.SafetyFiring {
		t.Fatalf("the pushed sample must fire the net with no dispatch in between (kv=%v fires=%d)", st.SafetyKV, st.SafetyFires)
	}
	// ... and the fire CLEARS the same way, live
	p.Saturation(context.Background(), livePods(liveEndpoint("a", 0.20, 0, 64, 1000, time.Now())))
	p.pushEngineLoad(time.Now())
	if p.sched.Snapshot().Engines[engineLabel(engineID(0))].SafetyFiring {
		t.Fatal("the fire must clear from a live sample, again with no dispatch")
	}
}

// The detector CONFLICT guard: a second detector in the same config is announced
// (exactly one may answer the flow-control gate).
func TestDetectorIdentityGate(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 1, 1000)
	if got := p.otherDetectors(); len(got) != 0 {
		t.Fatalf("no handle → no other detectors, got %v", got)
	}
}

// Capacity rides the LIVE path (the event-time copy may carry an empty geometry), and
// the per-engine liveness alarm is per engine.
func TestCapacityLearnedFromLiveSamples(t *testing.T) {
	p := newTestPlugin(t, defaultConfig(), 0, 0)
	if p.sched.HasCapacity() {
		t.Fatal("no capacity yet")
	}
	p.Saturation(context.Background(), livePods(liveEndpoint("a", 0.1, 0, 64, 1000, time.Now()),
		liveEndpoint("b", 0.1, 0, 64, 2000, time.Now())))
	p.pushEngineLoad(time.Now())
	if !p.sched.HasCapacityFor(engineID(0)) || !p.sched.HasCapacityFor(engineID(1)) {
		t.Fatal("both engines' capacity must be learned from the live samples")
	}
	snap := p.sched.Snapshot()
	a, b := snap.Engines[engineLabel(engineID(0))], snap.Engines[engineLabel(engineID(1))]
	if a.BudgetBlocks != 950 || b.BudgetBlocks != 1900 {
		t.Fatalf("per-engine budgets wrong: a=%d b=%d", a.BudgetBlocks, b.BudgetBlocks)
	}
}

// BENCHMARK: the flow controller polls the detector at ~1 kHz, so Saturation must be a
// couple of loads and NO allocation on the default (noop) path.
//
//	go test -run xxx -bench BenchmarkSaturation -benchmem ./pkg/epp/framework/plugins/groupsched/
func BenchmarkSaturationNoop(b *testing.B) {
	cfg := defaultConfig()
	s, err := NewScheduler(cfg)
	if err != nil {
		b.Fatal(err)
	}
	p := &GroupScheduler{cfg: cfg, sched: s, metrics: newGroupMetrics(s), pins: newPinTable(),
		endpoints: map[string]*endpointState{}, hetero: map[string]struct{}{}}
	now := time.Now()
	eps := make([]fwkdl.Endpoint, 0, 2)
	for i := 0; i < 2; i++ {
		eps = append(eps, liveEndpoint(engineID(i), 0.5, 10, 64, 1000, now))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := p.Saturation(context.Background(), eps); got != 0 {
			b.Fatalf("noop detector must answer 0, got %v", got)
		}
	}
}

// The ALTERNATIVE path is rate-limited: at ~1 kHz only one recompute per refresh
// interval happens, so its cost is amortised to ~nothing per call.
func BenchmarkSaturationMemoryOnly(b *testing.B) {
	cfg := defaultConfig()
	cfg.Safety = "budget-fraction"
	s, err := NewScheduler(cfg)
	if err != nil {
		b.Fatal(err)
	}
	p := &GroupScheduler{cfg: cfg, sched: s, metrics: newGroupMetrics(s), pins: newPinTable(),
		endpoints: map[string]*endpointState{}, hetero: map[string]struct{}{}}
	now := time.Now()
	eps := make([]fwkdl.Endpoint, 0, 2)
	for i := 0; i < 2; i++ {
		eps = append(eps, liveEndpoint(engineID(i), 0.5, 10, 64, 1000, now))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Saturation(context.Background(), eps)
	}
}
