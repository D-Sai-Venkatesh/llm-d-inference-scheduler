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

// The E4a placement Filter's gates. Each one is a property the E4 CONFIGS depend
// on, so each is asserted directly rather than inferred from a run:
//
//  1. NEVER EMPTY — a Filter that returns no candidates fails the request. When
//     the member's seat engine is not among the candidates the Filter must pass
//     everything through and let observed-endpoint-wins repair the binding.
//  2. PASS-THROUGH for unseated and anonymous traffic (seat=pin's defining
//     property: no seat, no placement opinion).
//  3. LOCK-LIGHT — the Filter must not take s.mu. Pick holds it at ~1 kHz; a
//     Filter that waited on it would serialise dispatch behind the decision loop.
//     Proven by holding s.mu and requiring Filter to return anyway.
//  4. BINDING MODE — inert outside seat=pin (member-stable's pin is external).

import (
	"context"
	"testing"
	"time"

	types "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestFilterPinsTheSeatEngine(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	r.pins.set("m1", endpointKey(b))
	got := r.Filter(context.Background(), mpRequest("m1", "r1", 2), []fwksched.Endpoint{a, b})
	if len(got) != 1 {
		t.Fatalf("Filter returned %d candidate(s), want 1 (the seat engine)", len(got))
	}
	if endpointKey(got[0]) != endpointKey(b) {
		t.Fatalf("Filter kept %q, want %q", endpointKey(got[0]), endpointKey(b))
	}
}

// TestFilterNeverEmpty is the hard requirement: the seat engine is NOT among the
// candidates (the EPP stopped offering it, or the picker's view is narrower), so
// the Filter must hand back the full candidate set — a request that fails here is
// strictly worse than a request that lands off-binding and is repaired.
func TestFilterNeverEmpty(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	r.pins.set("m1", "engine-that-does-not-exist")
	got := r.Filter(context.Background(), mpRequest("m1", "r1", 2), []fwksched.Endpoint{a, b})
	if len(got) != 2 {
		t.Fatalf("Filter returned %d candidate(s) with the seat engine absent, want all 2 (never empty, never fail the request)", len(got))
	}
}

// TestFilterPassThroughs covers both pass-through classes.
func TestFilterPassThroughs(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	pods := []fwksched.Endpoint{a, b}
	// unseated: no pin at all
	if got := r.Filter(context.Background(), mpRequest("m-unseated", "r1", 2), pods); len(got) != 2 {
		t.Fatalf("unseated member: Filter returned %d, want all 2", len(got))
	}
	// anonymous: no member id resolves (no identity attribute at all)
	if got := r.Filter(context.Background(), &fwksched.InferenceRequest{RequestID: "anon"}, pods); len(got) != 2 {
		t.Fatalf("anonymous request: Filter returned %d, want all 2", len(got))
	}
	// single candidate: nothing to narrow
	r.pins.set("m1", endpointKey(a))
	if got := r.Filter(context.Background(), mpRequest("m1", "r1", 2), pods[:1]); len(got) != 1 {
		t.Fatalf("single candidate: Filter returned %d, want 1", len(got))
	}
}

// TestFilterInertInMemberStable: the E3 arm keeps the EXTERNAL pin, so this
// plugin must not narrow anything there (one pin, one authority).
func TestFilterInertInMemberStable(t *testing.T) {
	r := newFilterTestScheduler(t, func(cfg *Config) { cfg.BindingMode = "member-stable" })
	a, b := epA(), epB()
	r.pins.set("m1", endpointKey(b))
	if got := r.Filter(context.Background(), mpRequest("m1", "r1", 2), []fwksched.Endpoint{a, b}); len(got) != 2 {
		t.Fatalf("member-stable: Filter narrowed to %d candidate(s), want all 2", len(got))
	}
}

// TestFilterTakesNoSchedulerLock is the lock-light gate. The scheduler's decision
// mutex is held (exactly as Pick holds it for the whole decision) and the Filter
// must still complete: the pin table has its own RWMutex and the Filter never
// touches s.mu.
func TestFilterTakesNoSchedulerLock(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	r.pins.set("m1", endpointKey(b))
	pods := []fwksched.Endpoint{a, b}
	request := mpRequest("m1", "r1", 2)

	r.sched.mu.Lock() // Pick's lock, held for the whole decision
	done := make(chan int, 1)
	go func() {
		done <- len(r.Filter(context.Background(), request, pods))
	}()
	select {
	case n := <-done:
		r.sched.mu.Unlock()
		if n != 1 {
			t.Fatalf("Filter returned %d candidate(s), want 1", n)
		}
	case <-time.After(2 * time.Second):
		r.sched.mu.Unlock()
		t.Fatal("Filter blocked on s.mu: it must use the pin table's own RWMutex, never the scheduler's decision mutex (Pick holds s.mu at ~1 kHz)")
	}
}

// TestPinTableFollowsTheSeat: the pin mirror is written by the scheduler's
// seat/unseat hooks, so it tracks the SEAT (not the member-stable homing).
func TestPinTableFollowsTheSeat(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	if err := r.sched.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	heads := []Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(40)}}
	engine, i := r.sched.PickFromHeadsEngine(heads, now)
	if i < 0 {
		t.Fatal("the newcomer was not seated")
	}
	if got, ok := r.pins.get("m1"); !ok || got != engine {
		t.Fatalf("pin table = (%q,%v), want (%q,true) — the mirror must follow the seat", got, ok, engine)
	}
	// exit ⇒ the pin is gone (seat = pin)
	r.sched.DispatchedOn("m1", "r1", engine, keys(10, Hash(3)), bytesFor(40), now)
	r.sched.SettleOn("m1", "r1", engine, struct{ Prompt, Cached int }{Prompt: 10, Cached: 10}, true, now.Add(time.Second))
	r.sched.Tick(now.Add(10 * time.Hour))
	if _, ok := r.pins.get("m1"); ok {
		t.Fatal("the pin survived the exit; seat=pin drops it at unseat")
	}
}

// ---- helpers ----

func epA() fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-a"}}, nil, nil)
}

func epB() fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "engine-b"}}, nil, nil)
}

func newFilterTestScheduler(t *testing.T, tweak func(*Config)) *GroupScheduler {
	t.Helper()
	cfg := e4aCfg()
	// the member is the SESSION here so mpRequest's attribute resolves a member
	// id without a branch-identity plugin.
	cfg.MemberKey = "session"
	if tweak != nil {
		tweak(&cfg)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r := &GroupScheduler{typedName: fwkplugin.TypedName{Type: PluginType, Name: "gs"}, cfg: s.cfg, sched: s, pins: newPinTable()}
	r.metrics = newGroupMetrics(s)
	s.onBind = func(member, engine string) { r.pins.set(member, engine) }
	s.onUnbind = func(member string) { r.pins.del(member) }
	return r
}

// TestFilterPinsTheFirstDispatchAfterSeating pins the ordering fact that makes
// the E4 liveness gate meaningful: seating happens in Pick BEFORE the dispatch, so
// the first request a member ever sends is already narrowed to its seat engine.
// A "passthrough-unseated" rate that is not ~0 therefore means the policy is NOT
// seating (the arm is not running the model at all), not that the pin lags.
func TestFilterPinsTheFirstDispatchAfterSeating(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	if err := r.sched.SetNumBlocksFor(endpointKey(a), 64, 1000); err != nil {
		t.Fatal(err)
	}
	if err := r.sched.SetNumBlocksFor(endpointKey(b), 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	// the head is seated by the Pick pass, then dispatched.
	if _, i := r.sched.PickFromHeadsEngine([]Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(40)}}, now); i < 0 {
		t.Fatal("the newcomer was not seated")
	}
	got := r.Filter(context.Background(), mpRequest("m1", "r1", 2), []fwksched.Endpoint{a, b})
	if len(got) != 1 {
		t.Fatalf("the FIRST dispatch after seating returned %d candidate(s), want 1 (already pinned)", len(got))
	}
}

// TestFilterKeepsPinWhileExiting: a member MARKED EXITING (its in-flight requests
// are still draining) keeps its pin until the actual exit — otherwise the tail of
// its own residency would be thrown off-binding, cold, and counted as an affinity
// violation.
func TestFilterKeepsPinWhileExiting(t *testing.T) {
	r := newFilterTestScheduler(t, nil)
	a, b := epA(), epB()
	if err := r.sched.SetNumBlocksFor(endpointKey(a), 64, 1000); err != nil {
		t.Fatal(err)
	}
	if err := r.sched.SetNumBlocksFor(endpointKey(b), 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	heads := []Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(40)}}
	engine, i := r.sched.PickFromHeadsEngine(heads, now)
	if i < 0 {
		t.Fatal("the newcomer was not seated")
	}
	r.sched.DispatchedOn("m1", "r1", engine, keys(20, Hash(5)), bytesFor(40), now)
	// mark it EXITING with the request still in flight
	r.sched.mu.Lock()
	g := r.sched.groups[engine]
	g.index.MarkExitingLocked("m1")
	r.sched.mu.Unlock()
	if got, ok := r.pins.get("m1"); !ok || got != engine {
		t.Fatalf("pin while exiting = (%q,%v), want (%q,true) — the pin must survive until the actual exit", got, ok, engine)
	}
	if got := r.Filter(context.Background(), mpRequest("m1", "r2", 2), []fwksched.Endpoint{a, b}); len(got) != 1 {
		t.Fatalf("an exiting member's dispatch returned %d candidate(s), want 1 (still pinned)", len(got))
	}
	// the drain completes ⇒ the pin goes with it
	r.sched.SettleOn("m1", "r1", engine, struct{ Prompt, Cached int }{Prompt: 10, Cached: 10}, true, now.Add(time.Second))
	if _, ok := r.pins.get("m1"); ok {
		t.Fatal("the pin survived the drain-induced exit")
	}
}
