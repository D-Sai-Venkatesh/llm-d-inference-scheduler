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

package tokenizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	mmobs "github.com/llm-d/llm-d-router/pkg/epp/framework/common/observability/multimodal"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	rcplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// Producer is the token producer as a LIBRARY rather than a plugin: exactly one
// render of a request body, written to Body.TokenizedRequest.
//
// One render per request is the property the whole arrangement rests on: the
// plugin runs it at dispatch (its data-producer phase) and a plugin that needs
// tokens EARLIER — branch identity, whose queue key must exist before admission —
// runs it at arrival. Whichever runs second finds TokenizedRequest already
// populated and returns on the skip path below, and every consumer (the prefix
// scorers, the identity) then reads one and the same tokenization.
//
// The render is EXPENSIVE (a /render round trip: ~120ms, 1.42µs/token, 91% of
// EPP processing on the campaign's prompts), which is why it is a single shared
// call rather than a per-consumer one.
type Producer struct {
	backend           tokenInputProducer
	backendName       string
	owner             plugin.TypedName
	endpointDiscovery *endpointDiscoveryHandler
}

// NewProducer builds the backend described by a token-producer parameter block,
// in the token-producer plugin's own parameter language, and starts its warmup.
// owner names the plugin that runs it (span attributes, endpoint events).
func NewProducer(ctx context.Context, owner plugin.TypedName, rawConfig json.RawMessage) (*Producer, error) {
	config := tokenizerPluginConfig{}
	if len(rawConfig) > 0 {
		if err := json.Unmarshal(rawConfig, &config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: %w", PluginType, err)
	}
	return newProducer(ctx, owner, &config)
}

// Tokenize renders the request's tokens onto its body and returns the backend's
// error. A body that already carries tokens is left alone, so the second caller
// of a request pays nothing.
//
// The tokenize span opens below the already-tokenized skip path, so it is
// emitted only when the backend is actually invoked.
func (pr *Producer) Tokenize(ctx context.Context, request *scheduling.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return errors.New("request body is nil")
	}
	if request.Body.RenderRequest {
		return nil
	}
	if request.Body.TokenizedRequest != nil {
		// A parser (e.g. vLLM gRPC) may pre-populate tokens without a salt;
		// ensure cache-salt isolation still applies on the skip path.
		if request.Body.TokenizedRequest.CacheSalt == "" {
			request.Body.TokenizedRequest.CacheSalt = CacheSaltFromBody(request.Body)
		}
		return nil
	}

	ctx = withMMMetadata(ctx, parseMMMetadataHeaders(request.Headers))
	if auth, ok := metadata.GetLowerCaseHeaderValue(request.Headers, "authorization"); ok {
		ctx = withAuthHeader(ctx, auth)
	}

	ctx, span := tracing.Tracer(rcplugins.TracerScope).Start(ctx, "tokenize",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()
	// On the default (tracing-disabled) path Start returns a non-recording span;
	// skip attribute construction, which walks the request's multimodal features.
	tracingActive := span.IsRecording()
	if tracingActive {
		attrs := []attribute.KeyValue{
			semconv.LLMDEPPTokenProducerBackend(pr.backendName),
		}
		if request.TargetModel != "" {
			attrs = append(attrs, semconv.GenAIRequestModel(request.TargetModel))
		}
		if clientRequestID := request.Headers[reqcommon.RequestIDHeaderKey]; clientRequestID != "" {
			attrs = append(attrs, semconv.GenAIRequestID(clientRequestID))
		}
		span.SetAttributes(attrs...)
	}

	tp, err := pr.backend.produce(ctx, request.Body)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return err
	}
	if tp == nil || tp.TokenCount() == 0 {
		if tracingActive {
			span.SetAttributes(semconv.LLMDEPPTokenProducerResult(resultSkippedNoTokens))
		}
		return nil
	}
	tp.CacheSalt = CacheSaltFromBody(request.Body)
	request.Body.TokenizedRequest = tp

	if tracingActive {
		span.SetAttributes(append(mmobs.SpanAttributes(request),
			semconv.LLMDEPPTokenProducerTokenCount(tp.TokenCount()),
		)...)
	}
	return nil
}

// BackendName identifies the configured backend (BackendVLLM or BackendEstimate).
func (pr *Producer) BackendName() string { return pr.backendName }

// Timeout is the backend's own render timeout, zero when it manages none (the
// estimate backend, which is in-memory).
func (pr *Producer) Timeout() time.Duration {
	if pr.backend == nil {
		return 0
	}
	if ta, ok := pr.backend.(timeoutAware); ok {
		return ta.produceTimeout()
	}
	return 0
}

// RegisterDependencies wires a discovery-backed renderer to endpoint lifecycle
// events. A fixed-URL backend registers nothing.
func (pr *Producer) RegisterDependencies(r datalayer.Registrar) error {
	if pr.endpointDiscovery == nil {
		return nil
	}
	return r.Register(datalayer.PendingRegistration{
		Owner:         pr.owner,
		SourceType:    sourcenotifications.EndpointNotificationSourceType,
		Extractor:     pr.endpointDiscovery,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// newProducer constructs the configured backend around an already-validated
// config: vllm /render (selected by 'vllm' or 'modelName'), or estimate
// byte-packing (the default when no backend is set).
func newProducer(ctx context.Context, owner plugin.TypedName, config *tokenizerPluginConfig) (*Producer, error) {
	var backend tokenInputProducer
	var backendName string
	var endpointPicker *discoveredEndpointPicker
	switch {
	case config.VLLM != nil || config.ModelName != "":
		cfg := config.VLLM
		if cfg == nil {
			cfg = &vllmConfig{}
		}
		renderer, err := newVLLMHTTPRenderer(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
		renderer.pluginName = owner.Name
		registerRenderMetrics()
		legacyMessages, err := configureLegacyMessages(ctx, owner.Name, cfg.MessagesRenderMode)
		if err != nil {
			return nil, err
		}
		legacyResponses, err := configureLegacyResponses(ctx, owner.Name, cfg.ResponsesRenderMode)
		if err != nil {
			return nil, err
		}
		backend = renderBackend{
			tk: renderer, modelName: config.ModelName,
			legacyMessages: legacyMessages, legacyResponses: legacyResponses,
			warmupAuth: vllmWarmupAuthHeader(),
		}
		backendName = BackendVLLM
		endpointPicker, _ = renderer.endpointPicker.(*discoveredEndpointPicker)
		if endpointPicker != nil && endpointPicker.config.DiscoverModelLimits {
			go endpointPicker.watchModelLimits(ctx, renderer.client, config.ModelName)
		}
	default:
		backend = estimateBackend{img: newImageEstimator(config.Estimate), vid: newVideoEstimator(config.Estimate), aud: newAudioEstimator(config.Estimate)}
		backendName = BackendEstimate
	}

	pr := &Producer{backend: backend, backendName: backendName, owner: owner}
	if endpointPicker != nil {
		pr.endpointDiscovery = newEndpointDiscoveryHandler(owner, endpointPicker)
	}
	if w, ok := backend.(warmer); ok {
		go w.warmup(ctx)
	}
	return pr, nil
}
