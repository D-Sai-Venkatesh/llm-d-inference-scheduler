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

import (
	"context"
	"sync"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// pinTable is the lock-light mirror of the scheduler's own state that the request path reaches
// without ever taking the scheduler's decision mutex (see filter.go's lock discipline).
type pinTable struct {
	mu sync.RWMutex
	m  map[string]string
	// req is request id → branch: present from the Pick that released the request to the moment
	// its record ends (settle, abandon, reap, or its branch's exit).
	req map[string]string
}

func newPinTable() *pinTable {
	return &pinTable{m: map[string]string{}, req: map[string]string{}}
}

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

// setRequest records that reqID belongs to branch (the Pick that released it).
func (p *pinTable) setRequest(reqID, branch string) {
	if p == nil || reqID == "" || branch == "" {
		return
	}
	p.mu.Lock()
	p.req[reqID] = branch
	p.mu.Unlock()
}

// delRequest forgets a request (its record ended).
func (p *pinTable) delRequest(reqID string) {
	if p == nil || reqID == "" {
		return
	}
	p.mu.Lock()
	delete(p.req, reqID)
	p.mu.Unlock()
}

// branchOfRequest is the branch a request id belongs to ("" when nothing tracks it: anonymous
// traffic, a pre-capacity FCFS dispatch, a request whose branch has left).
func (p *pinTable) branchOfRequest(reqID string) string {
	if p == nil || reqID == "" {
		return ""
	}
	p.mu.RLock()
	b := p.req[reqID]
	p.mu.RUnlock()
	return b
}

// requestCount is the mirror's request count (the leak test's input).
func (p *pinTable) requestCount() int {
	if p == nil {
		return 0
	}
	p.mu.RLock()
	n := len(p.req)
	p.mu.RUnlock()
	return n
}

// Filter narrows the candidate endpoints to the branch's CURRENT SEAT ENGINE.
func (r *GroupScheduler) Filter(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	if len(pods) <= 1 {
		return pods
	}
	if request == nil {
		return pods
	}
	branch := r.pins.branchOfRequest(request.RequestID)
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
		// the seat engine is not a candidate: pass through and let the observed endpoint win
		// (PreRequest rebinds + counts the violation).
		r.logger.V(logutil.DEBUG).Info("group-scheduler: FILTER PASS-THROUGH — the seat engine is not a candidate; passing every candidate through (the observed endpoint wins the rebind)",
			"branch", branch, "seatedOn", engineLabel(engine), "candidates", len(pods))
		return pods
	}
	return out
}
