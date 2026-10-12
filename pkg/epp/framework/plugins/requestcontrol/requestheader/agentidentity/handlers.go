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

package agentidentity

import (
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// compile-time checks that every handler implements the interface.
var (
	_ HarnessHandler = canonicalHandler{}
	_ HarnessHandler = customHandler{}
	_ HarnessHandler = claudeCodeHandler{}
	_ HarnessHandler = openCodeHandler{}
	_ HarnessHandler = codexHandler{}
)

// headerValue returns the trimmed value of one header, empty when absent.
func headerValue(headers map[string]string, name string) string {
	return strings.TrimSpace(headers[name])
}

// canonicalHandler resolves identity from the llm-d canonical headers, which
// any harness or gateway can send to bypass harness-specific detection.
type canonicalHandler struct{}

func (canonicalHandler) Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	if request == nil {
		return AgentIdentity{}, false
	}
	headers := request.Headers
	id := AgentIdentity{
		Session: headerValue(headers, CanonicalSessionHeader),
		Thread:  headerValue(headers, CanonicalThreadHeader),
		Parent:  headerValue(headers, CanonicalParentHeader),
	}
	if id.Session == "" && id.Thread == "" && id.Parent == "" {
		return AgentIdentity{}, false
	}
	// A canonical thread id with no parent and no session names its own
	// root: the thread is the program.
	if id.Session == "" && id.Thread != "" && id.Parent == "" {
		id.Session = id.Thread
	}
	return id, true
}

// customHandler resolves identity from the operator's additional session
// headers (Parameters.AdditionalSessionHeaders), so an agent llm-d does
// not know, or an upstream rename, can be supported without a code change.
// It contributes its match without stopping the chain: the resolver lets it
// win the session field, and the recognizing harness handler still fills
// thread and parent.
type customHandler struct {
	sessionHeaders []string
}

func (h customHandler) Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	if request == nil {
		return AgentIdentity{}, false
	}
	for _, header := range h.sessionHeaders {
		if v := headerValue(request.Headers, header); v != "" {
			return AgentIdentity{Session: v}, true
		}
	}
	return AgentIdentity{}, false
}

// claudeCodeHandler resolves identity from Claude Code's headers. The main
// thread sends only the session id; a sub-agent adds its agent id, and a
// nested one also the parent's agent id.
type claudeCodeHandler struct{}

func (claudeCodeHandler) Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	if request == nil {
		return AgentIdentity{}, false
	}
	headers := request.Headers
	id := AgentIdentity{
		Session: headerValue(headers, ClaudeCodeSessionHeader),
		Thread:  headerValue(headers, ClaudeCodeAgentHeader),
		Parent:  headerValue(headers, ClaudeCodeParentHeader),
	}
	if id.Session == "" && id.Thread == "" && id.Parent == "" {
		return AgentIdentity{}, false
	}
	// A sub-agent spawned by the main thread sends its agent id but no
	// parent header; its parent is the main thread, whose id is the
	// session id.
	if id.Thread != "" && id.Parent == "" && id.Session != "" {
		id.Parent = id.Session
	}
	return id, true
}

// openCodeHandler resolves identity from OpenCode's headers. It reads only
// the session affinity id, which names one thread rather than a program
// root; thread, parent and root derivation are deferred pending
// anomalyco/opencode#54881, so OpenCode keeps today's behaviour.
type openCodeHandler struct{}

func (openCodeHandler) Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	if request == nil {
		return AgentIdentity{}, false
	}
	session := headerValue(request.Headers, OpenCodeSessionHeader)
	if session == "" {
		return AgentIdentity{}, false
	}
	return AgentIdentity{Session: session}, true
}

// codexHandler resolves identity from Codex's headers. Every thread sends
// thread-id, a sub-agent adds the parent thread's id, and the session-id
// header carries the root thread's id.
type codexHandler struct{}

func (codexHandler) Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	if request == nil {
		return AgentIdentity{}, false
	}
	headers := request.Headers
	id := AgentIdentity{
		Session: headerValue(headers, CodexSessionHeader),
		Thread:  headerValue(headers, CodexThreadHeader),
		Parent:  headerValue(headers, CodexParentHeader),
	}
	if id.Session == "" {
		id.Session = headerValue(headers, CodexSessionHeaderLegacy)
	}
	if id.Session == "" && id.Thread == "" && id.Parent == "" {
		return AgentIdentity{}, false
	}
	return id, true
}
