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

// THE GROUP SCHEDULER'S OWN PLACEMENT FILTER (E4a, bindingMode "seat").
//
// The framework splits "who may serve this request" (Filters, scheduling layer)
// from "which queue to serve next" (the FairnessPolicy, flow-control layer). In
// the E3 arm the pin lived entirely OUTSIDE the policy — session-affinity-filter,
// an external Filter keyed on the branch identity, with its own TTL, its own
// attribute plumbing and its own silent-no-op failure mode (the pilot-2 class: a
// producer-name mismatch makes every request pass through and the arm measures
// nothing). Under seat=pin the pin is a FACT the policy already holds: the member
// IS seated on exactly one engine, so the policy can narrow the candidate set to
// that engine itself. Two consequences the arm depends on:
//
//   - the E4 configs carry NO session-affinity-filter (one pin, one authority:
//     two authorities is a second variable, and the E3 arm already showed they
//     can disagree once per member at birth);
//   - the narrowing reads the scheduler's OWN seat state, so it cannot be
//     silently no-op: if the pin table is empty the arm is not seating anybody,
//     which every other liveness gate already reports.
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
	"fmt"
	"sync"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// filterDecision labels llm_d_epp_group_filter_decisions_total. Every dispatch
// through the Filter books exactly one, so the run artifact states what the pin
// actually did (an arm whose decisions are all "passthrough-unseated" is not a
// pinned arm).
const (
	filterPinned        = "pinned"                       // narrowed to the member's seat engine
	filterNotCandidate  = "engine-not-a-candidate"       // the seat engine is not offered → pass ALL through
	filterUnseated      = "passthrough-unseated"         // no seat yet: the member has no placement opinion
	filterAnonymous     = "passthrough-anonymous"        // R9: never constrained
	filterSingle        = "passthrough-single-candidate" // nothing to narrow
	filterModelDisabled = "passthrough-binding-mode"     // member-stable: the pin is not ours
)

// pinTable is the lock-light mirror of the scheduler's member→engine SEAT
// binding. Writers are the scheduler's hooks (seat / exit / rebind), which run
// under s.mu; the reader is the Filter, which must not.
type pinTable struct {
	mu sync.RWMutex
	m  map[string]string
}

func newPinTable() *pinTable { return &pinTable{m: map[string]string{}} }

func (p *pinTable) set(member, engine string) {
	if p == nil || member == "" {
		return
	}
	p.mu.Lock()
	p.m[member] = engine
	p.mu.Unlock()
}

func (p *pinTable) del(member string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	delete(p.m, member)
	p.mu.Unlock()
}

func (p *pinTable) get(member string) (string, bool) {
	if p == nil {
		return "", false
	}
	p.mu.RLock()
	e, ok := p.m[member]
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

// Filter narrows the candidate endpoints to the member's CURRENT SEAT ENGINE
// (bindingMode=seat). Contract, in order:
//
//  1. NOT SEAT MODE → pass everything through. In member-stable mode (E3 and
//     every legacy arm) placement is pinned by the external
//     session-affinity-filter, and a second pin here would be a second variable.
//  2. ≤1 candidate → pass through (nothing to narrow; still counted).
//  3. ANONYMOUS request (no member resolved) → pass through freely (R9: the
//     policy never tracks, seats or constrains anonymous traffic).
//  4. UNSEATED member → pass through freely. This is the seat=pin model's
//     defining property: an unseated member carries NO placement opinion, so its
//     head is placed by the ordinary scorer and the engine it lands on becomes
//     its seat (observed-endpoint-wins, PreRequest).
//  5. SEATED member → return ONLY the seat engine.
//
// NEVER RETURNS AN EMPTY LIST. If the seat engine is not among the candidates
// (the EPP no longer offers it, or the picker's own view is narrower), the
// Filter passes EVERY candidate through instead: an empty list makes the profile
// fail the request, while a pass-through lets the dispatch land somewhere real —
// PreRequest observes the endpoint, rebinds (the observed endpoint wins) and
// counts an affinity violation, so the event is both survived and measured. A
// silent request failure would be the worst possible failure mode here.
func (r *GroupScheduler) Filter(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	if !r.cfg.seatPin() {
		r.metrics.ObserveFilterDecision(filterModelDisabled)
		return pods
	}
	if len(pods) <= 1 {
		r.metrics.ObserveFilterDecision(filterSingle)
		return pods
	}
	member, _, _ := r.memberOf(request)
	if member == "" {
		r.metrics.ObserveFilterDecision(filterAnonymous)
		return pods
	}
	engine, bound := r.pins.get(member)
	if !bound || engine == "" {
		r.metrics.ObserveFilterDecision(filterUnseated)
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
		r.metrics.ObserveFilterDecision(filterNotCandidate)
		fmt.Printf("group-scheduler: FILTER PASS-THROUGH member=%s seated-on=%s is not among the %d candidate endpoint(s) — passing all candidates through (the observed endpoint wins and the violation is counted)\n",
			member, engineLabel(engine), len(pods))
		return pods
	}
	r.metrics.ObserveFilterDecision(filterPinned)
	return out
}
