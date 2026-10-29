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
	"encoding/json"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func flattenBody(msgs ...fwkrh.Message) *fwkrh.InferenceRequestBody {
	return &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: msgs}}
}

func flattenMsg(role, content string) fwkrh.Message {
	return fwkrh.Message{Role: role, Content: fwkrh.Content{Raw: content}}
}

// TestFlattenCanonicalBytes_RolePlusContent: a tool-call-free chat body flattens
// to exactly role-tag + content per message, nothing else. This is the
// byte-identical contract for bodies without tool calls (adding ToolCalls to the
// flattener must not perturb them).
func TestFlattenCanonicalBytes_RolePlusContent(t *testing.T) {
	raw, err := FlattenCanonicalBytes(flattenBody(
		flattenMsg("user", "hello"),
		flattenMsg("assistant", "hi there"),
	))
	require.NoError(t, err)
	assert.Equal(t, "userhelloassistanthi there", string(raw))
}

// TestFlattenCanonicalBytes_DropsPerRequestFields: two bodies of the same
// conversation that differ only in per-request fields flatten identically —
// otherwise every turn's first chunk could shift and the chain would not be
// prefix-stable.
func TestFlattenCanonicalBytes_DropsPerRequestFields(t *testing.T) {
	base, err := FlattenCanonicalBytes(flattenBody(flattenMsg("user", "hello")))
	require.NoError(t, err)
	other := flattenBody(flattenMsg("user", "hello"))
	other.ChatCompletions.ChatTemplateKWArgs = map[string]any{"enable_thinking": true}
	other.ChatCompletions.ContinueFinalMessage = true
	raw, err := FlattenCanonicalBytes(other)
	require.NoError(t, err)
	assert.Equal(t, string(base), string(raw))
}

// TestFlattenCanonicalBytes_PrefixStableAcrossTurns is the property branch
// identity depends on: turn N's canonical stream is a byte prefix of turn N+1's
// once the conversation only grows.
func TestFlattenCanonicalBytes_PrefixStableAcrossTurns(t *testing.T) {
	turn1, err := FlattenCanonicalBytes(flattenBody(flattenMsg("user", "explain the ring window")))
	require.NoError(t, err)
	turn2, err := FlattenCanonicalBytes(flattenBody(
		flattenMsg("user", "explain the ring window"),
		flattenMsg("assistant", "it is a residency ring"),
		flattenMsg("user", "and the second one?"),
	))
	require.NoError(t, err)
	require.Greater(t, len(turn2), len(turn1))
	assert.Equal(t, string(turn1), string(turn2[:len(turn1)]), "turn 1 must be a byte prefix of turn 2")
}

// TestFlattenCanonicalBytes_ToolCallsAreContent is the survey's false-merge
// gap: an assistant turn that only calls a tool has empty content, so two
// different tool calls (identical arguments aside) would flatten to the same
// bytes and merge two conversations. The arguments must reach the stream.
func TestFlattenCanonicalBytes_ToolCallsAreContent(t *testing.T) {
	call := func(args string) *fwkrh.InferenceRequestBody {
		var call any
		require.NoError(t, json.Unmarshal([]byte(`{"id":"call_1","type":"function","function":{"name":"run","arguments":`+args+`}}`), &call))
		return flattenBody(
			flattenMsg("user", "list the files"),
			fwkrh.Message{Role: "assistant", ToolCalls: []any{call}},
		)
	}
	a, err := FlattenCanonicalBytes(call(`"{\"path\":\"/tmp\"}"`))
	require.NoError(t, err)
	b, err := FlattenCanonicalBytes(call(`"{\"path\":\"/etc\"}"`))
	require.NoError(t, err)

	assert.NotEqual(t, string(a), string(b), "tool-call arguments must reach the canonical stream")
	assert.True(t, strings.HasPrefix(string(a), "userlist the filesassistant"),
		"tool calls are appended after role+content, got %q", string(a))
	// Deterministic: the same body flattens to the same bytes.
	again, err := FlattenCanonicalBytes(call(`"{\"path\":\"/tmp\"}"`))
	require.NoError(t, err)
	assert.Equal(t, string(a), string(again))
}

// TestFlattenCanonicalBytes_AnthropicPrefixStable: the /v1/messages path (tools,
// top-level system, role-tagged messages) is prefix-stable too.
func TestFlattenCanonicalBytes_AnthropicPrefixStable(t *testing.T) {
	turn1 := &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
		System:   fwkrh.AnthropicContent{Raw: "be terse"},
		Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hello"}}},
	}}
	turn2 := &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{Raw: "be terse"},
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hello"}},
			{Role: "assistant", Content: fwkrh.AnthropicContent{Raw: "hi"}},
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "again"}},
		},
	}}
	a, err := FlattenCanonicalBytes(turn1)
	require.NoError(t, err)
	b, err := FlattenCanonicalBytes(turn2)
	require.NoError(t, err)
	require.Greater(t, len(b), len(a))
	assert.Equal(t, string(a), string(b[:len(a)]))
}

// TestFlattenCanonicalBytes_NoStream: body shapes with nothing to flatten report
// the sentinel error, so callers fall back instead of hashing an empty stream.
func TestFlattenCanonicalBytes_NoStream(t *testing.T) {
	for name, body := range map[string]*fwkrh.InferenceRequestBody{
		"nil":       nil,
		"empty":     {},
		"emptyChat": {ChatCompletions: &fwkrh.ChatCompletionsRequest{}},
	} {
		_, err := FlattenCanonicalBytes(body)
		assert.ErrorIs(t, err, ErrNoCanonicalStream, name)
	}
}

// TestFlattenCanonicalBytes_PrefixProperty_Randomized is the property the
// campaign corpus cannot exercise (the WEKA trace has zero tool blocks, so the
// ToolCalls path is unvalidated in production): for RANDOMIZED conversations that
// mix string content, multi-part content, tool calls and a tools definition, the
// canonical stream of any message prefix must be a byte prefix of the canonical
// stream of the whole conversation. A violation would mint a false new branch on
// every turn of such a conversation (the unsafe direction of the pre-registration).
func TestFlattenCanonicalBytes_PrefixProperty_Randomized(t *testing.T) {
	const (
		conversations = 200
		maxMessages   = 8
	)
	rng := rand.New(rand.NewSource(42))

	roles := []string{"system", "user", "assistant", "tool", ""}
	texts := []string{
		"", "ok", "read the file", `{"path":"/tmp/a.go"}`,
		strings.Repeat("padding ", 40), // crosses several 256-byte chunks
		"ünïcode ✓",
	}

	randomMessage := func() fwkrh.Message {
		msg := fwkrh.Message{Role: roles[rng.Intn(len(roles))]}
		// String content, parts content, or both shapes with the same text —
		// clients switch between them across turns.
		switch rng.Intn(3) {
		case 0:
			msg.Content = fwkrh.Content{Raw: texts[rng.Intn(len(texts))]}
		case 1:
			parts := make([]fwkrh.ContentBlock, rng.Intn(3)+1)
			for i := range parts {
				parts[i] = fwkrh.ContentBlock{Type: "text", Text: texts[rng.Intn(len(texts))]}
			}
			msg.Content = fwkrh.Content{Structured: parts}
		case 2:
			msg.Content = fwkrh.Content{Raw: texts[rng.Intn(len(texts))]}
			msg.Content.Structured = []fwkrh.ContentBlock{{Type: "text", Text: "ignored when raw is set"}}
		}
		if rng.Intn(3) == 0 {
			// Assistant tool call with random arguments (the false-merge gap).
			var call any
			err := json.Unmarshal([]byte(`{"id":"call_`+strconv.Itoa(rng.Intn(5))+`","type":"function","function":{"name":"run","arguments":"{\"n\":`+strconv.Itoa(rng.Intn(100))+`}"}}`), &call)
			require.NoError(t, err)
			msg.ToolCalls = []any{call}
		}
		return msg
	}

	for c := 0; c < conversations; c++ {
		count := rng.Intn(maxMessages) + 1
		msgs := make([]fwkrh.Message, count)
		for i := range msgs {
			msgs[i] = randomMessage()
		}
		tools := []any(nil)
		if rng.Intn(2) == 0 {
			tools = []any{map[string]any{"type": "function", "function": map[string]any{"name": "run", "parameters": map[string]any{"type": "object"}}}}
		}

		whole, err := FlattenCanonicalBytes(&fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: msgs, Tools: tools},
		})
		require.NoError(t, err, "conversation %d", c)

		for j := 0; j <= count; j++ {
			prefixBody := &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: msgs[:j], Tools: tools},
			}
			prefix, err := FlattenCanonicalBytes(prefixBody)
			if err != nil {
				// A prefix with nothing in it yet has no stream at all: it is
				// trivially a prefix of anything.
				require.ErrorIs(t, err, ErrNoCanonicalStream, "conversation %d prefix %d", c, j)
				prefix = nil
			}
			require.LessOrEqual(t, len(prefix), len(whole), "conversation %d prefix %d is longer than the whole", c, j)
			assert.Equal(t, string(whole[:len(prefix)]), string(prefix),
				"conversation %d: flattening must be append-only at message boundary %d", c, j)
		}
	}
}
