/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package arcwindow

import (
	"context"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// requestStartKey is the request attribute under which PreRequest stashes
// the dispatch time (the per-session turn-duration input).
var requestStartKey = fwkpluginNewDataKey("turn-start", PluginType)

// inflightEstimateKey is the request attribute under which PreRequest
// stashes the token estimate it charged, so ResponseBody removes exactly
// what was added.
var inflightEstimateKey = fwkpluginNewDataKey("inflight-estimate", PluginType)

// noteKVUsage records the latest engine KV utilization (locked).
func (m *manager) noteKVUsage(v float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if v >= 0 && v <= 1 {
		m.kvUsage = v
	}
}

// estimateTokens converts a request body size in bytes to a token
// estimate (bytes/4; the same heuristic as the sibling plugin).
func estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 1
	}
	t := int64(sizeBytes) / 4
	if t < 1 {
		t = 1
	}
	return t
}

// PreRequest charges the request's estimated prompt tokens to its
// session's in-flight footprint, refreshes the engine's KV reading and
// the capacity, and stashes the turn's start time and estimate.
func (r *ArcWindow) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	id := sessionID(request)
	if id == "" {
		return nil
	}
	endpoint := pickedEndpoint(schedulingResult)
	if endpoint == nil {
		return nil
	}
	md := endpoint.GetMetadata()
	if md == nil {
		return nil
	}
	m := endpoint.GetMetrics()
	capacity := endpointCapacity(m, r.cfg.CapacityTokens)
	if m != nil && m.Updated() {
		r.mgr.noteKVUsage(m.KVCacheUsagePercent)
	}

	now := time.Now()
	estimate := max(estimateTokens(request.RequestSizeBytes), 1)
	r.mgr.noteCapacity(md.ID.String(), capacity)
	r.mgr.chargePreRequest(id, estimate, now)

	request.PutAttribute(requestStartKey, now)
	request.PutAttribute(inflightEstimateKey, estimate)
	return nil
}

// ResponseBody settles a completed turn: it removes the turn's in-flight
// estimate, commits the measured footprint (usage.total_tokens), records
// the turn duration and the completion timestamp (the drop-order input).
func (r *ArcWindow) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := sessionID(request)
	if id == "" {
		return
	}
	estimate, ok := fwksched.ReadRequestAttribute[int64](request, inflightEstimateKey)
	if !ok {
		return
	}
	start, hasStart := fwksched.ReadRequestAttribute[time.Time](request, requestStartKey)
	turnDur := time.Duration(0)
	if hasStart {
		turnDur = time.Since(start)
	}

	r.mgr.settleResponse(id, estimate, int64(response.Usage.TotalTokens), turnDur, time.Now())

	// Per-turn hit telemetry, when the engine reports cached tokens.
	if details := response.Usage.PromptTokenDetails; details != nil && response.Usage.PromptTokens > 0 {
		r.metrics.turnHitRatio.Observe(float64(details.CachedTokens) / float64(response.Usage.PromptTokens))
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

// endpointCapacity returns a pod's KV cache capacity in tokens.
func endpointCapacity(m *fwkdl.Metrics, fallback int64) float64 {
	if m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return float64(m.CacheBlockSize) * float64(m.CacheNumBlocks)
	}
	return float64(fallback)
}
