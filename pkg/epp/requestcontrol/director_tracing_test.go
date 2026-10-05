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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestRecordPrefixPrediction(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(context.Background(), "request_orchestration")

	warm := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: "pod-a"}, nil, fwkdl.NewAttributes())
	warm.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(3, 5, 64))
	cold := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: "pod-b"}, nil, fwkdl.NewAttributes())
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{"default": {
			TargetEndpoints:  []fwksched.Endpoint{warm},
			ScoredCandidates: []fwksched.ScoredEndpoint{{Endpoint: warm, Score: 1}, {Endpoint: cold, Score: 0.5}},
		}},
	}
	recordPrefixPrediction(span, result)
	span.End()

	got := map[attribute.Key]attribute.Value{}
	for _, kv := range recorder.Ended()[0].Attributes() {
		got[kv.Key] = kv.Value
	}
	require.Equal(t, "pod-a", got["llm_d.epp.target_endpoint"].AsString())
	require.Equal(t, int64(192), got["llm_d.epp.prefix.predicted_cached_tokens"].AsInt64())
	require.Equal(t, int64(320), got["llm_d.epp.prefix.indexed_tokens"].AsInt64())
	require.Equal(t, attrprefix.PrefixCacheMatchInfoDataKey.String(), got["llm_d.epp.prefix.key"].AsString())
	require.Equal(t, []string{"pod-a=192", "pod-b=0"}, got["llm_d.epp.prefix.candidates_predicted_cached_tokens"].AsStringSlice())
}

func TestRecordPrefixPredictionNonRecording(t *testing.T) {
	// A non-recording span (tracing off) must be a no-op, including a nil result.
	_, span := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample())).Tracer("test").Start(context.Background(), "x")
	recordPrefixPrediction(span, nil)
	recordPrefixPrediction(span, &fwksched.SchedulingResult{})
}
