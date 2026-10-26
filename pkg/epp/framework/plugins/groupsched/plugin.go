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

// Package groupsched implements the group scheduler (ALGORITHM-group.md):
// a working set of sessions whose unique prefix blocks fit ≤95% of the
// GPU cache, members dispatched immediately (oldest waiting request
// first, in-flight cap 64), rotation by payback (a member leaves when it
// has earned ρ× its measured entry cost, went silent past T_idle, or hit
// the turn cap — and only when a waiter needs the room), newcomers
// first at entry, hold-the-line, cold-entry staggering, and no deadlines
// anywhere. Block accounting is tier A (CP2b): chained token-block keys
// from the token-producer's TokenizedRequest at the measured block size.
//
// The flow-control gate stays exactly as deployed: the stack's
// saturation detector decides how much runs (the 80% pinned-KV gate);
// this policy only decides who.
package groupsched

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "group-scheduler"

var (
	_ fwkrc.PreRequest            = &GroupScheduler{}
	_ fwkrc.ResponseBodyProcessor = &GroupScheduler{}
	_ fwkplugin.ConsumerPlugin    = &GroupScheduler{}
	_ fwkplugin.StateDumper       = &GroupScheduler{}
	_ fwkdl.Registrant            = &GroupScheduler{}
	_ fwkdl.EndpointExtractor     = &GroupScheduler{}
	_ fwkfc.FairnessPolicy        = &GroupScheduler{}
)

// GroupScheduler is a single named instance shared by every hookup, so
// all of them read and write the same group state.
type GroupScheduler struct {
	typedName fwkplugin.TypedName
	cfg       Config

	sched   *Scheduler
	metrics *groupMetrics
}

// Factory builds a GroupScheduler from raw plugin parameters. It FAILS
// CLOSED at startup if the token-producer is not configured: without
// the TokenizedRequest there is no tier-A input, and tier B failed the
// CP2 trust measurements — no silent fallback (the CP3 decision line).
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}
	r := &GroupScheduler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:       cfg,
		sched:     NewScheduler(cfg),
	}
	r.metrics = newGroupMetrics(r.sched)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		go r.runTicker(handle.Context())
	}
	return r, nil
}

// runTicker runs the lifecycle tick (T_idle release, TTL silence,
// rotation) every second until ctx is cancelled.
func (r *GroupScheduler) runTicker(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.sched.Tick(time.Now())
		}
	}
}

func (r *GroupScheduler) TypedName() fwkplugin.TypedName { return r.typedName }

// Consumes declares agent identity AND the tokenized request as required
// inputs: configuration loading fails when either provider is missing.
// The tokenized-request requirement is the fail-closed gate (no tier-B
// fallback; CP2's M3 verdict).
func (r *GroupScheduler) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			agentidentity.AgentIdentityKey: "",
			tokenizer.TokenizedPromptDataKey: nil,
		},
	}
}

// Produces declares the request attributes PreRequest writes for its
// own later hooks.
func (r *GroupScheduler) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		requestKeysKey: []Hash(nil),
	}
}

// requestKeysKey stashes the request's block keys between PreRequest
// (arrival) and the settle/finalizer path.
var requestKeysKey = fwkplugin.NewDataKey("group-keys", PluginType)

// sessionID returns the session identifier published by the
// agent-identity plugin (the sibling plugins' pattern; anonymous
// traffic is not tracked and passes through last).
func sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return id
}

// RegisterDependencies subscribes the plugin to endpoint lifecycle
// events (the capacity source: cache_config_info per M4).
func (r *GroupScheduler) RegisterDependencies(reg fwkdl.Registrar) error {
	return reg.Register(fwkdl.PendingRegistration{
		Owner:      r.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  r,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract learns an endpoint's block size and block count when it joins
// the pool — the measured capacity the budget is denominated in (M4:
// block size from cache_config_info; 64 on the campaign stacks).
func (r *GroupScheduler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type != fwkdl.EventAddOrUpdate || event.Endpoint == nil {
		return nil
	}
	if m := event.Endpoint.GetMetrics(); m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return r.sched.SetCapacity(m.CacheBlockSize, m.CacheNumBlocks)
	}
	return nil
}

// blockKeysFor derives the chained token-block keys from the request's
// TokenizedRequest (tier A). Returns nil when the tokenized request is
// absent — the caller treats that as a wiring fault, never tier B.
func (r *GroupScheduler) blockKeysFor(request *fwksched.InferenceRequest) ([]Hash, string) {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil, ""
	}
	tp := request.Body.TokenizedRequest
	if len(tp.Prompts) == 0 {
		return nil, ""
	}
	// single-prompt requests (chat) are the v0 scope; multi-prompt
	// completions hash each prompt and concatenate (the sibling's
	// per-prompt pattern)
	var all []uint32
	for _, p := range tp.Prompts {
		all = append(all, p.TokenIDs...)
	}
	bs := 0
	if latestMetrics != nil {
		bs = latestMetrics.CacheBlockSize
	}
	return r.sched.HashTokens(all, tp.CacheSalt, bs), request.RequestID
}

// latestMetrics carries the freshest endpoint metrics for the block
// size (refreshed at PreRequest; the Extract path sets the index).
var latestMetrics *fwkdl.Metrics

// PreRequest records the request's arrival: the block keys enter the
// session's waiting set, non-members become waiters, and the keys are
// stashed for the settle path.
func (r *GroupScheduler) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	id := sessionID(request)
	if id == "" {
		return nil
	}
	if endpoint := pickedEndpoint(schedulingResult); endpoint != nil {
		if m := endpoint.GetMetrics(); m != nil && m.Updated() {
			latestMetrics = m
			if m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
				_ = r.sched.SetCapacity(m.CacheBlockSize, m.CacheNumBlocks)
			}
		}
	}
	keys, reqID := r.blockKeysFor(request)
	if keys == nil {
		// fail closed: the tokenized request is required (Consumes
		// enforces it at config load; this guards the runtime path)
		return errors.New("group-scheduler: no TokenizedRequest on the request (token-producer required; no tier-B fallback)")
	}
	request.PutAttribute(requestKeysKey, keys)
	r.sched.Arrive(id, reqID, keys, time.Now())
	return nil
}

// ResponseBody settles a dispatched request: the payback ledger from
// the usage report, the in-flight decrement, and the block settle. The
// EndOfStream call is the finalizer for EVERY termination cause
// (natural, client-disconnect, evicted, error) — the in-flight
// decrement happens regardless; the payback ledger only on natural.
func (r *GroupScheduler) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := sessionID(request)
	if id == "" {
		return
	}
	keys, _ := fwksched.ReadRequestAttribute[[]Hash](request, requestKeysKey)
	reqID := request.RequestID
	usage := struct{ Prompt, Cached int }{
		Prompt: response.Usage.PromptTokens,
	}
	if response.Usage.PromptTokenDetails != nil {
		usage.Cached = response.Usage.PromptTokenDetails.CachedTokens
	}
	r.sched.Settle(id, reqID, keys, usage, response.TerminationCause == fwkrc.TerminationCauseNatural, time.Now())
}

// Pick is the dispatch decision (§5): budget enforcement and rotation
// run eagerly, then the oldest waiting request of any non-exiting
// member under the in-flight cap dispatches. Non-members never run
// unseated (D10).
func (r *GroupScheduler) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	reqID, _, ok := r.sched.Pick(time.Now())
	if !ok {
		return nil, nil //nolint:nilnil
	}
	// find the queue whose head matches the chosen request
	var chosen fwkfc.FlowQueueAccessor
	band.IterateQueues(func(q fwkfc.FlowQueueAccessor) bool {
		if q != nil && q.Len() > 0 {
			if h := q.Peek(); h != nil {
				if h.OriginalRequest() != nil && h.OriginalRequest().InferenceRequest() != nil &&
					h.OriginalRequest().InferenceRequest().RequestID == reqID {
					chosen = q
					return false
				}
			}
		}
		return true
	})
	if chosen == nil {
		// the request vanished (cancelled between pick and here):
		// release the in-flight charge the Pick took
		r.sched.AbortPick(reqID)
		return nil, nil //nolint:nilnil
	}
	return chosen, nil
}

// DumpState reports the group snapshot (§13).
func (r *GroupScheduler) DumpState() (json.RawMessage, error) {
	return json.Marshal(r.sched.Snapshot())
}

// NewState satisfies the FairnessPolicy contract.
func (r *GroupScheduler) NewState(_ context.Context) any { return nil }

// AbortPick releases a Pick's in-flight charge when the chosen request
// is no longer dispatchable (the vanished-queue guard).
func (s *Scheduler) AbortPick(reqID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, m := range s.members {
		if s.index != nil && s.index.OldestWaiting(m.id) == reqID {
			m.inflight = max(m.inflight-1, 0)
			return
		}
	}
}

func pickedEndpoint(schedulingResult *fwksched.SchedulingResult) fwksched.Endpoint {
	if schedulingResult == nil {
		return nil
	}
	result, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 {
		return nil
	}
	return result.TargetEndpoints[0]
}
