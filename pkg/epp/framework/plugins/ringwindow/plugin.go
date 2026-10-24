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

// Package ringwindow implements the ring/window session scheduler for a
// single serving pool: a member set sized by a measured KV-footprint ledger
// (idle members included — the part of the cache the utilization gate cannot
// see), served in back-and-forth order so the engine's LRU-by-release-time
// eviction aligns with the schedule, rotating only when a waiting outsider's
// D-clock expires, and never holding work back.
//
// The design and its decisions (D1–D10) are specified in the project's
// V0-PLAN; this is the first build. The flow-control gate stays exactly as
// deployed: it decides how much runs, this policy only decides who.
package ringwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

var (
	_ fwkrc.PreRequest            = &RingWindow{}
	_ fwkrc.ResponseBodyProcessor = &RingWindow{}
	_ fwkplugin.ConsumerPlugin    = &RingWindow{}
	_ fwkplugin.StateDumper       = &RingWindow{}
	_ fwkdl.Registrant            = &RingWindow{}
	_ fwkdl.EndpointExtractor     = &RingWindow{}
	_ fwkfc.FairnessPolicy        = &RingWindow{}
)

// RingWindow is a single named instance shared by every hookup, so all of
// them read and write the same ledger, ring and scan state.
type RingWindow struct {
	typedName fwkplugin.TypedName
	cfg       Config

	mgr     *ringManager
	metrics *ringMetrics
}

// Factory builds a RingWindow from raw plugin parameters, registers its
// metrics and runs the ledger-hygiene sweep for the plugin's lifetime.
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

	r := &RingWindow{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:       cfg,
		mgr:       newRingManager(cfg),
	}
	r.metrics = newRingMetrics(r.mgr)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		sweepInterval := time.Duration(cfg.EvictionSweepSeconds * float64(time.Second))
		go r.runSweep(handle.Context(), sweepInterval)
	}
	return r, nil
}

// runSweep runs the ledger-hygiene sweep every interval until ctx is
// cancelled. It never decides who runs; it only drops the bookkeeping of
// sessions that ended.
func (r *RingWindow) runSweep(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.mgr.sweep(time.Now())
		}
	}
}

func (r *RingWindow) TypedName() fwkplugin.TypedName { return r.typedName }

// Consumes declares agent identity as a required input, so configuration
// loading fails when no identity provider is enabled: without per-session
// ids the ring has no members to track.
func (r *RingWindow) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// Produces declares the request attributes PreRequest writes for the
// plugin's own later hooks; the data layer drops undeclared writes and
// fails the request.
func (r *RingWindow) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		requestStartKey:     time.Time{},
		inflightEstimateKey: int64(0),
	}
}

// sessionID returns the session identifier published by the agent-identity
// plugin, or "" for requests carrying none. Anonymous traffic is not
// tracked; it passes through the policy last.
func sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return id
}

// headSessionID returns the session of a queue head, read the same way as
// in PreRequest, or "" for anonymous traffic. A queue is named by the
// fairness ID, which can group several sessions; the head's own session id
// is what the ledger tracks.
func headSessionID(head fwkfc.QueueItemAccessor) string {
	req := head.OriginalRequest()
	if req == nil {
		return ""
	}
	return sessionID(req.InferenceRequest())
}

// fwkpluginNewDataKey is a small indirection so accounting.go can declare
// its keys without importing the plugin interface twice under different
// aliases.
func fwkpluginNewDataKey(name, pluginType string) fwkplugin.DataKey {
	return fwkplugin.NewDataKey(name, pluginType)
}

// RegisterDependencies subscribes the plugin to endpoint lifecycle events.
func (r *RingWindow) RegisterDependencies(reg fwkdl.Registrar) error {
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

// Extract learns an endpoint's capacity when it joins the pool. The
// single-pool v0 keeps serving with the last capacity seen; capacity is
// refreshed at PreRequest anyway.
func (r *RingWindow) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type != fwkdl.EventAddOrUpdate || event.Endpoint == nil {
		return nil
	}
	if md := event.Endpoint.GetMetadata(); md != nil {
		if m := event.Endpoint.GetMetrics(); m != nil {
			r.mgr.noteCapacity(md.ID.String(), endpointCapacity(m, r.cfg.CapacityTokens))
		}
	}
	return nil
}

// stateDump is the snapshot returned by DumpState. Session ids come from
// client headers, so they are omitted from the dump; the ring order is
// shown by sequence number.
type stateDump struct {
	Members      int              `json:"members"`
	MemberSum    float64          `json:"memberSumTokens"`
	Budget       float64          `json:"budgetTokens"`
	Capacity     float64          `json:"capacityTokens"`
	MeanTurnSec  float64          `json:"meanTurnSeconds"`
	Rotations    int64            `json:"rotations"`
	Guests       int64            `json:"guests"`
	Oversized    int64            `json:"oversizedGuests"`
	Demotions    map[string]int64 `json:"demotions"`
	Sessions     []sessionView    `json:"sessions"`
}

// DumpState reports the window, the counters and the per-session ledger.
func (r *RingWindow) DumpState() (json.RawMessage, error) {
	m := r.mgr
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	dump := stateDump{
		Members:     len(m.membersLocked()),
		MemberSum:   m.memberSumLocked(now),
		Budget:      m.budgetTokensLocked(),
		Capacity:    m.capacity,
		MeanTurnSec: m.meanTurnDurLocked().Seconds(),
		Rotations:   m.counters.rotations,
		Guests:      m.counters.guests,
		Oversized:   m.counters.oversized,
		Demotions:   m.counters.demotions,
		Sessions:    m.sessionViewsLocked(),
	}
	return json.Marshal(dump)
}
