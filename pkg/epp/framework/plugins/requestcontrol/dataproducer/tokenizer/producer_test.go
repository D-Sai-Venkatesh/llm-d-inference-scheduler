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
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvcache/tokenization"
)

// chatBody is a minimal chat body whose Payload the mock renderer ignores.
func chatBody() *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
		Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Raw: "hello"}}},
	}}
}

// TestProducer_TokenizeIsIdempotent pins the property the whole one-render
// arrangement rests on: the SECOND caller of a request pays nothing. Branch
// identity renders at arrival; the plugin's Produce runs later at dispatch and
// must find the tokens already there.
func TestProducer_TokenizeIsIdempotent(t *testing.T) {
	var renders atomic.Int64
	pr := &Producer{backendName: BackendVLLM, backend: renderBackend{tk: &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			renders.Add(1)
			return []uint32{1, 2, 3, 4}, nil, nil
		},
	}}}

	request := &scheduling.InferenceRequest{RequestID: "r1", Body: chatBody()}
	require.NoError(t, pr.Tokenize(context.Background(), request))
	require.Equal(t, int64(1), renders.Load())
	require.NotNil(t, request.Body.TokenizedRequest)
	first := request.Body.TokenizedRequest

	// The dispatch-time caller gets the same TokenizedRequest back, unrendered.
	require.NoError(t, pr.Tokenize(context.Background(), request))
	assert.Equal(t, int64(1), renders.Load(), "the second caller must not render again")
	assert.Same(t, first, request.Body.TokenizedRequest)
}

// TestProducer_PrePopulatedTokensAreKept: a parser may hand the request tokens
// (vLLM gRPC) — the producer keeps them and only fills the missing cache salt.
func TestProducer_PrePopulatedTokensAreKept(t *testing.T) {
	var renders atomic.Int64
	pr := &Producer{backendName: BackendVLLM, backend: renderBackend{tk: &mockTokenizer{
		renderChatFunc: func(fwkrh.RequestPayload) ([]uint32, *tokenization.MultiModalFeatures, error) {
			renders.Add(1)
			return []uint32{9}, nil, nil
		},
	}}}

	body := chatBody()
	body.ChatCompletions.CacheSalt = "salt-7"
	body.TokenizedRequest = &fwkrh.TokenizedRequest{
		Prompts: []fwkrh.PromptTokens{{TokenIDs: []uint32{1, 2, 3}}},
	}
	request := &scheduling.InferenceRequest{RequestID: "r2", Body: body}

	require.NoError(t, pr.Tokenize(context.Background(), request))
	assert.Zero(t, renders.Load(), "pre-populated tokens are never re-rendered")
	assert.Equal(t, []uint32{1, 2, 3}, request.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Equal(t, "salt-7", request.Body.TokenizedRequest.CacheSalt, "the skip path still applies the salt")
}

// TestNewProducer_ValidatesItsConfig: the library entry point fails closed on a
// config no backend can serve.
func TestNewProducer_ValidatesItsConfig(t *testing.T) {
	_, err := NewProducer(context.Background(), plugin.TypedName{Type: PluginType, Name: "lib"},
		[]byte(`{"estimate":{},"modelName":"m"}`))
	require.Error(t, err)

	_, err = NewProducer(context.Background(), plugin.TypedName{Type: PluginType, Name: "lib"}, nil)
	require.NoError(t, err, "an empty config selects the estimate backend")
}
