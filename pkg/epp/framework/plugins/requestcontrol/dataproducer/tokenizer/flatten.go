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
// Bodies without a text/chat shape (none of chat, messages, completions,
// embeddings, responses, conversations, generate) return ErrNoCanonicalStream.
func FlattenCanonicalBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	if body == nil {
		return nil, ErrNoCanonicalStream
	}
	b := estimateBackend{}
	switch {
	case body.ChatCompletions != nil:
		raw, _ := b.chatCompletionsBytes(body.ChatCompletions, mmMetadata{})
		if len(raw) == 0 {
			return nil, ErrNoCanonicalStream
		}
		return raw, nil
	case body.Messages != nil:
		raw, _ := b.messagesBytes(body.Messages)
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
	raw, err := estimateBytes(body)
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
