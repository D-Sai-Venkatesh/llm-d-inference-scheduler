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
	"encoding/binary"
	"errors"
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// ErrNoCanonicalStream reports a body shape the canonical flattener does not
// cover (there is nothing to derive a prefix from).
var ErrNoCanonicalStream = errors.New("tokenizer: no canonical byte stream for this request body")

// flattenMode selects which parts of a body are IDENTITY-bearing. The two modes
// differ on exactly the two things measured to break the prefix property of a
// conversation under real clients, and they differ deliberately:
//
//   - tools (tool DEFINITIONS, the top-level `tools` block): excluded from the
//     identity stream. An agent that changes its tool set mid-thread (claude_code
//     switches the main agent's tools for the sub-agent's `Task` tool) breaks the
//     byte prefix at the tools block while the entire message history is intact —
//     a tool-set change must cost a cache miss, not an identity change. Measured:
//     stage 0B §5.3, 32 of 33 Exgentic mid-thread "splits" are this class, and a
//     per-step tool shortlist would make it unbounded.
//   - thinking (Anthropic `thinking` blocks): excluded from the identity stream.
//     Clients legitimately drop earlier turns' reasoning (the API only keeps the
//     last assistant turn's signed thinking), so including it would false-new a
//     conversation on the very next turn.
//
// The canonical (sizing) mode keeps both: the estimate backend's byte stream
// mirrors what the engine will prefill, so it must include everything the model
// reads. Identity is a different question with a different answer — hence two
// functions, not one.
type flattenMode struct {
	tools    bool
	thinking bool
}

var (
	// canonicalMode is the sizing view: everything the engine reads.
	canonicalMode = flattenMode{tools: true, thinking: true}
	// identityMode is the prefix-key view: message content only.
	identityMode = flattenMode{tools: false, thinking: false}
)

// FlattenCanonicalBytes returns the canonical byte stream of a request body:
// the stable rendering that prefix identity (branch identity, conversation
// prefix keys) can be chained over.
//
// The stream is the protocol's shared scaffolding — [tools], system prompt —
// followed by each message as role-tagged content, in wire order. Per-request
// fields (max_tokens, temperature, stream, metadata, request ids) are absent by
// construction, and JSON is re-marshalled with sorted map keys, so two turns of
// an append-only conversation produce streams where the earlier one is a BYTE
// PREFIX of the later one. That property is the whole point: chunking and
// chaining such a stream from its own start yields prefix-stable hashes, while
// hashing raw request bytes does not (tier-B's defect: a per-request field
// shifts every chunk boundary → every turn looks like a new conversation).
//
// Identity must not depend on an operator's size heuristics, so the multimodal
// placeholders use the package's DEFAULT estimators rather than a configured
// backend's; a placeholder is derived from the asset's content hash, so it is
// stable across turns either way. The output is a pure function of the body.
//
// This is the SIZING view: tools and thinking blocks are part of it. For the
// identity view (branch identity's queue key) use FlattenIdentityBytes.
//
// Bodies without a text/chat shape (none of chat, messages, completions,
// embeddings, responses, conversations, generate) return ErrNoCanonicalStream.
func FlattenCanonicalBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	return flattenBytes(body, canonicalMode)
}

// FlattenIdentityBytes returns the byte stream that BRANCH IDENTITY is derived
// from: the conversation as the client means it, with the two things that
// legitimately churn mid-thread left out (see flattenMode).
//
//   - chat completions: role + content + tool_calls per message, no top-level
//     tools block.
//   - Anthropic messages: the system prompt + role/content per message, no tools
//     block, no thinking blocks. Tool calls and tool results are content (they
//     are what the model said and read), so they stay.
//
// Tool-set equality is deliberately NOT part of the decision: a set-equality
// gate would make a mid-thread tool change a NON-match, which is the very
// false-new-branch class it is meant to remove. (The tool-set hash is available
// to a caller as a tie-break between otherwise identical candidates — never as a
// gate.)
//
// Like FlattenCanonicalBytes, the output is a pure function of the body and is
// append-only over an append-only conversation: turn N is a byte prefix of turn
// N+1.
func FlattenIdentityBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	return flattenBytes(body, identityMode)
}

// flattenBytes is the shared body-shape dispatch behind the two views.
func flattenBytes(body *fwkrh.InferenceRequestBody, mode flattenMode) ([]byte, error) {
	if body == nil {
		return nil, ErrNoCanonicalStream
	}
	b := estimateBackend{}
	switch {
	case body.ChatCompletions != nil:
		raw, _ := b.chatCompletionsBytes(body.ChatCompletions, mmMetadata{}, mode)
		if len(raw) == 0 {
			return nil, ErrNoCanonicalStream
		}
		return raw, nil
	case body.Messages != nil:
		raw, _ := b.messagesBytes(body.Messages, mode)
		if len(raw) == 0 {
			return nil, ErrNoCanonicalStream
		}
		return raw, nil
	case body.Generate != nil:
		if len(body.Generate.TokenIDs) == 0 {
			return nil, ErrNoCanonicalStream
		}
		return packTokenBytes(body.Generate.TokenIDs), nil
	}
	raw, err := estimateBytesMode(body, mode)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrNoCanonicalStream, err)
	}
	if len(raw) == 0 {
		return nil, ErrNoCanonicalStream
	}
	return raw, nil
}

// packTokenBytes renders pre-tokenized input as the little-endian byte stream
// of its token IDs, so a generate/completions token-ID body flattens into the
// same shape as a text body (a byte prefix relation between two such bodies is
// then a token prefix relation).
func packTokenBytes(tokens []uint32) []byte {
	out := make([]byte, 0, len(tokens)*bytesPerToken)
	var buf [4]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint32(buf[:], t)
		out = append(out, buf[:]...)
	}
	return out
}
