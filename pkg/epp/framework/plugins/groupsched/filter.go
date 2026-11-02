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

// THE GROUP SCHEDULER'S OWN PLACEMENT FILTER.
//
// The framework splits "who may serve this request" (Filters, scheduling layer)
// from "which queue to serve next" (the FairnessPolicy, flow-control layer). The
// pin is a fact the policy already holds: a branch IS seated on exactly one engine,
// so the policy narrows the candidate set to that engine itself. Two consequences
// the model depends on:
//
//   - the configs carry NO external session-affinity-filter (one pin, one
//     authority: two authorities is a second variable, and the earlier arm showed
//     they can disagree once per branch at birth);
//   - the narrowing reads the scheduler's OWN seat state, so it cannot be silently
//     no-op: if the pin table is empty the policy is not seating anybody, which
//     every other liveness gate already reports.
//
// LOCK DISCIPLINE (hard requirement): Pick runs at ~1 kHz holding s.mu (the
// scheduler's decision mutex). The Filter runs on every dispatch in the
// scheduling layer. If the Filter took s.mu, every Pick would contend with every
// dispatch for the same lock and the 1 ms decision loop would queue behind the
// request path. So the Filter NEVER touches s.mu: the scheduler mirrors each
// seat/unseat into pinTable under its own RWMutex, and the Filter takes one
// RLock for one map read. pinTable's critical sections are a single map
// read/write.

import (
	"context"
	"sync"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// pinTable is the lock-light mirror of the scheduler's branch→engine SEAT
// binding. Writers are the scheduler's hooks (seat / exit / rebind), which run
// under s.mu; the reader is the Filter, which must not.
type pinTable struct {
	mu sync.RWMutex
	m  map[string]string
}

func newPinTable() *pinTable { return &pinTable{m: map[string]string{}} }

func (p *pinTable) set(branch, engine string) {
	if p == nil || branch == "" {
		return
	}
	p.mu.Lock()
	p.m[branch] = engine
	p.mu.Unlock()
}

func (p *pinTable) del(branch string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	delete(p.m, branch)
	p.mu.Unlock()
}

func (p *pinTable) get(branch string) (string, bool) {
	if p == nil {
		return "", false
	}
	p.mu.RLock()
	e, ok := p.m[branch]
	p.mu.RUnlock()
	return e, ok
}

func (p *pinTable) size() int {
	if p == nil {
		return 0
	}
	p.mu.RLock()
	n := len(p.m)
	p.mu.RUnlock()
	return n
}

// Filter narrows the candidate endpoints to the branch's CURRENT SEAT ENGINE.
// Contract, in order:
//
//  1. ≤1 candidate → pass through (nothing to narrow; still counted).
//  2. ANONYMOUS request (no branch resolved) → pass through freely (the policy never
//     tracks, seats or constrains anonymous traffic).
//  3. UNSEATED branch → pass through freely. This is the model's defining
//     property: an unseated branch carries NO placement opinion, so its head is
//     placed by the ordinary scorer and the engine it lands on becomes its seat
//     (observed-endpoint-wins, PreRequest). The pin is created at SEAT time
//     (seatLocked → onBind), and the seat happens in Pick BEFORE the request
//     dispatches, so a branch's FIRST dispatch after being seated is ALREADY
//     pinned — "passthrough-unseated" covers only heads dispatched while unseated
//     (the anonymous and pre-capacity classes), which on a healthy run is ≈ 0.
//  4. SEATED branch → return ONLY the seat engine.
//
// NEVER RETURNS AN EMPTY LIST. If the seat engine is not among the candidates
// (the EPP no longer offers it, or the picker's own view is narrower), the
// Filter passes EVERY candidate through instead: an empty list makes the profile
// fail the request, while a pass-through lets the dispatch land somewhere real —
// PreRequest observes the endpoint, rebinds (the observed endpoint wins) and
// counts an affinity violation, so the event is both survived and measured. A
// silent request failure would be the worst possible failure mode here.
func (r *GroupScheduler) Filter(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	if len(pods) <= 1 {
		return pods
	}
	branch := r.branchOf(request)
	if branch == "" {
		return pods
	}
	engine, bound := r.pins.get(branch)
	if !bound || engine == "" {
		return pods
	}
	out := make([]fwksched.Endpoint, 0, 1)
	for _, p := range pods {
		if p != nil && endpointKey(p) == engine {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		// the seat engine is not a candidate: pass through and let the
		// observed endpoint win (PreRequest rebinds + counts the violation).
		r.logger.V(logutil.DEBUG).Info("group-scheduler: FILTER PASS-THROUGH — the seat engine is not a candidate; passing every candidate through (the observed endpoint wins the rebind)",
			"branch", branch, "seatedOn", engineLabel(engine), "candidates", len(pods))
		return pods
	}
	return out
}
