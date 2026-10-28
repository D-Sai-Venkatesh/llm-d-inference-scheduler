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

// Package groupsched implements the group scheduler (ALGORITHM-group.md).
// See scheduler.go's header for the algorithm summary.
//
// Hook wiring (the director's order, S0-1's reality): a request blocks in
// flow control at admission → the fairness Pick (HERE) sees it as a
// queue head → the chosen queue is released → the data producers run
// (the tokenizer sets TokenizedRequest) → PreRequest (HERE) corrects the
// estimate with the exact tier-A keys → the engine → ResponseBody
// (HERE) settles. The waiting set therefore comes from the queues at
// Pick time; queued-request admission cost = persisted history blocks +
// a conservative wire-size delta; the exact keys land at PreRequest and
// D13's growth enforcement absorbs the estimate error.
package groupsched

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
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

// GroupScheduler is a single named instance shared by every hookup.
type GroupScheduler struct {
	typedName fwkplugin.TypedName
	cfg       Config

	sched   *Scheduler
	metrics *groupMetrics
	handle  fwkplugin.Handle

	fcfsOnce sync.Once
}

// Factory builds a GroupScheduler. FAILS CLOSED without the
// token-producer (Consumes) and on an invalid config; the index attaches
// at startup with the config's block size (the token processor's value).
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	sched, err := NewScheduler(cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}

	r := &GroupScheduler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:       cfg,
		sched:     sched,
		// the handle is held for the RegisterDependencies-time gate: plugin
		// construction order is NOT the config order (the runner may build
		// the fairness policy first), so the token-producer check cannot
		// run at Factory time — RegisterDependencies runs after ALL
		// plugins are constructed and added.
		handle: handle,
	}
	r.metrics = newGroupMetrics(r.sched)
	// BUILD-PLAN §1(ii): echo EVERY knob at startup — each arm's config
	// is verifiable afterwards from the run artifacts (also carried in
	// DumpState/Snapshot.Config).
	if echo, err := json.Marshal(cfg); err == nil {
		fmt.Printf("group-scheduler: effective config %s\n", echo)
	}
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		r.sched.onExit = r.metrics.ObserveExit
		r.sched.onEntry = r.metrics.ObserveEntry
		r.sched.onServe = r.metrics.ObserveServed
		// the per-id SEAT/EXIT event log (one debug line each; the
		// campaign's spans could not see seat membership).
		r.sched.onEvent = func(ev Event) {
			fmt.Printf("group-scheduler event t=%s kind=%s id=%s %s\n",
				ev.T.UTC().Format(time.RFC3339Nano), ev.Kind, ev.ID, ev.Detail)
		}
		go r.runTicker(handle.Context())
	}
	return r, nil
}

func (r *GroupScheduler) runTicker(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.sched.Tick(time.Now())
			r.metrics.Refresh()
		}
	}
}

func (r *GroupScheduler) TypedName() fwkplugin.TypedName { return r.typedName }

// Consumes requires agent identity. The tokenizer's output is enforced
// at RUNTIME instead: the data layer forbids a FairnessPolicy (flow
// control layer) from declaring a DataProducer's key in Consumes (the
// layer order is request-control before flow-control), so the fail-closed
// gate is PreRequest's error on a missing TokenizedRequest — without the
// token-producer EVERY request fails loudly at dispatch; nothing ever
// falls back to tier B (CP2's M3 verdict).
func (r *GroupScheduler) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			agentidentity.AgentIdentityKey: "",
		},
	}
}

// Produces declares the request attributes PreRequest writes.
func (r *GroupScheduler) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		requestKeysKey: []Hash(nil),
	}
}

var requestKeysKey = fwkplugin.NewDataKey("group-keys", PluginType)

func sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return id
}

// validateTokenProducer is the gate body (testable without a registrar).
func (r *GroupScheduler) validateTokenProducer() error {
	if r.handle == nil {
		return nil // unit/test mode
	}
	for _, p := range r.handle.GetAllPlugins() {
		if p != nil && p.TypedName().Type == tokenizer.PluginType {
			return nil
		}
	}
	return fmt.Errorf(
		"the '%s' plugin requires a token-producer (tier-A block keys; tier B failed the CP2 trust measurements and there is no fallback) — add the token-producer with the vllm render backend",
		PluginType)
}

// RegisterDependencies runs after every plugin is constructed — the
// STARTUP fail-closed gate lives here: a missing token-producer is a
// full-outage misconfiguration (every request would error at PreRequest,
// the backstop), burning an arm before anyone noticed. The producer's
// BACKEND is not inspectable from outside the plugin — an estimate-backend
// producer still passes this gate (it produces tokens; tier-A quality is
// the config author's responsibility, guarded by the config comment).
func (r *GroupScheduler) RegisterDependencies(reg fwkdl.Registrar) error {
	if err := r.validateTokenProducer(); err != nil {
		return err
	}
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

// Extract learns num_gpu_blocks (and checks the block size matches the
// config — a mismatch fails loudly).
func (r *GroupScheduler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type != fwkdl.EventAddOrUpdate || event.Endpoint == nil {
		return nil
	}
	if m := event.Endpoint.GetMetrics(); m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return r.sched.SetNumBlocks(m.CacheBlockSize, m.CacheNumBlocks)
	}
	return nil
}

// Pick is the dispatch decision (§5). The waiting set is the queue-head
// snapshot (S0-1); pre-capacity it degrades to FCFS pass-through (A5).
func (r *GroupScheduler) Pick(_ context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()
	var queues []fwkfc.FlowQueueAccessor
	var heads []Head
	band.IterateQueues(func(q fwkfc.FlowQueueAccessor) bool {
		if q == nil || q.Len() == 0 {
			return true
		}
		h := q.Peek()
		if h == nil {
			return true
		}
		session, reqID, size := "", "", int64(0)
		if req := h.OriginalRequest(); req != nil {
			if ir := req.InferenceRequest(); ir != nil {
				session = sessionID(ir)
				reqID = ir.RequestID
				size = int64(ir.RequestSizeBytes)
				if size <= 0 {
					size = int64(req.ByteSize())
				}
			}
		}
		queues = append(queues, q)
		heads = append(heads, Head{Session: session, ReqID: reqID, Arrival: h.EnqueueTime(), SizeBytes: size})
		return true
	})
	if len(heads) == 0 {
		return nil, nil //nolint:nilnil
	}
	// A5: before the measured capacity arrives, FCFS pass-through (once
	// per process log; the gauge counts it)
	if !r.sched.HasCapacity() {
		r.fcfsOnce.Do(func() {
			fmt.Println("group-scheduler: num_gpu_blocks not yet measured; serving FCFS until the endpoint metrics arrive")
		})
		r.metrics.ObserveFCFS()
		if i := FCFSPick(heads); i >= 0 {
			return queues[i], nil
		}
		return nil, nil //nolint:nilnil
	}
	if i := r.sched.PickFromHeads(heads, now); i >= 0 {
		return queues[i], nil
	}
	return nil, nil //nolint:nilnil
}

// NewState satisfies the FairnessPolicy contract.
func (r *GroupScheduler) NewState(_ context.Context) any { return nil }

// PreRequest runs after the request leaves the queue (dispatch): the
// exact tier-A keys from the TokenizedRequest correct the estimate
// (S0-1 d), the in-flight charge was already taken at Pick.
func (r *GroupScheduler) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	id := sessionID(request)
	if id == "" {
		return nil
	}
	if endpoint := pickedEndpoint(schedulingResult); endpoint != nil {
		if m := endpoint.GetMetrics(); m != nil && m.Updated() && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
			if err := r.sched.SetNumBlocks(m.CacheBlockSize, m.CacheNumBlocks); err != nil {
				return err // the loud mismatch failure
			}
		}
	}
	// V3/R6: the engine-side stagger — RESIDENT turns get priority over
	// cold ENTRY prefills in vLLM's waiting loop (Phase-2 deployments
	// with --scheduling-policy priority; lower value = higher
	// priority). The split is entry-vs-resident, not seated-vs-not:
	// seating happens in Pick BEFORE PreRequest, so a seated test would
	// mark every tracked request high. EntryTurn marks this residency's
	// first dispatch (the entry) low; every later turn is high.
	if r.cfg.PriorityHeader == "engine" && request.Headers != nil {
		if r.sched.EntryTurn(id) {
			request.Headers["X-Vllm-Priority"] = "10"
		} else {
			request.Headers["X-Vllm-Priority"] = "0"
		}
	}
	keys := r.blockKeysFor(request)
	if keys == nil {
		return errors.New("group-scheduler: no TokenizedRequest on the request (token-producer required; no tier-B fallback)")
	}
	request.PutAttribute(requestKeysKey, keys)
	size := int64(request.RequestSizeBytes)
	r.sched.Dispatched(id, request.RequestID, keys, size, time.Now())
	return nil
}

// blockKeysFor derives the chained token-block keys (tier A).
func (r *GroupScheduler) blockKeysFor(request *fwksched.InferenceRequest) []Hash {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	tp := request.Body.TokenizedRequest
	if len(tp.Prompts) == 0 {
		return nil
	}
	var all []uint32
	for _, p := range tp.Prompts {
		all = append(all, p.TokenIDs...)
	}
	return r.sched.HashTokens(all, tp.CacheSalt, r.cfg.BlockSizeTokens)
}

// ResponseBody: the first chunk is prefill complete (the cold-entry
// release, B3); EndOfStream is the finalizer for every termination
// cause (in-flight decrement always; the payback ledger natural-only).
func (r *GroupScheduler) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	id := sessionID(request)
	if id == "" {
		return
	}
	if response.StartOfStream {
		r.sched.Chunk(id)
	}
	if !response.EndOfStream {
		return
	}
	usage := struct{ Prompt, Cached int }{Prompt: response.Usage.PromptTokens}
	if response.Usage.PromptTokenDetails != nil {
		usage.Cached = response.Usage.PromptTokenDetails.CachedTokens
	}
	r.sched.Settle(id, request.RequestID, usage, response.TerminationCause == fwkrc.TerminationCauseNatural, time.Now())
}

// DumpState reports the group snapshot (§13).
func (r *GroupScheduler) DumpState() (json.RawMessage, error) {
	return json.Marshal(r.sched.Snapshot())
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
