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

package ringwindow

import (
	"context"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// requestStartKey is the request attribute under which PreRequest stashes
// the dispatch time, so ResponseBody can measure the turn's wall duration —
// the per-session service prediction (decision D10).
var requestStartKey = fwkpluginNewDataKey("turn-start", PluginType)

// inflightEstimateKey is the request attribute under which PreRequest stashes
// the token estimate it charged to the session, so ResponseBody removes
// exactly what was added.
var inflightEstimateKey = fwkpluginNewDataKey("inflight-estimate", PluginType)

// PreRequest charges the request's estimated prompt tokens to its session's
// in-flight total, binds the session to the picked endpoint and learns the
// pool capacity from it. A session can have several turns in flight at once,
// so each turn charges and later removes only its own estimate.
func (r *RingWindow) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
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
	metrics := endpoint.GetMetrics()
	capacity := endpointCapacity(metrics, r.cfg.CapacityTokens)
	if metrics != nil && metrics.Updated() {
		r.mgr.noteKVUsage(metrics.KVCacheUsagePercent)
	}

	now := time.Now()
	estimate := max(estimateTokens(request.RequestSizeBytes), 1)
	r.mgr.noteCapacity(md.ID.String(), capacity)
	r.mgr.chargePreRequest(id, md.ID.String(), estimate, now)

	request.PutAttribute(requestStartKey, now)
	request.PutAttribute(inflightEstimateKey, estimate)
	return nil
}

// ResponseBody settles a completed turn: it removes the turn's in-flight
// estimate, commits the measured footprint (usage.total_tokens of the last
// completed turn — decision D2), records the turn duration (D10) and the
// release order (the completion timestamp that drives demotion order).
func (r *RingWindow) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
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

	// Per-turn hit telemetry, when the engine reports cached tokens
	// (requires enable_prompt_tokens_details).
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
func endpointCapacity(m *datalayer.Metrics, fallback int64) float64 {
	if m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return float64(m.CacheBlockSize) * float64(m.CacheNumBlocks)
	}
	return float64(fallback)
}
