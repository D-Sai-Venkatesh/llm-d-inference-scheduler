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

// Package agentidentity provides a RequestHeaderProcessor plugin that resolves
// agent identity — the session a request belongs to, the thread that sent it,
// and the thread that spawned that one — from canonical llm-d headers and
// harness-specific headers, and stores them as request attributes for use by
// other subsystems (flow control, scheduling, KV cache, etc.).
package agentidentity

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// AgentIdentityKey is the request-attribute key under which this plugin
// publishes the resolved agent identity. Downstream consumers such as the
// Director read it via scheduling.ReadRequestAttribute to derive the
// FairnessID.
// The key carries no producer name: configuration names this attribute bare
// ("attribute: agent-identity"), and any plugin able to resolve an agent
// identity is an acceptable source.
var AgentIdentityKey = plugin.NewDataKey("agent-identity", "")

// ThreadIDKey is the request-attribute key for the thread that sent the
// request: the main conversation or one sub-agent. It equals the session id
// when the harness sends no thread id of its own.
var ThreadIDKey = plugin.NewDataKey("agent-thread-id", "")

// ParentThreadIDKey is the request-attribute key for the thread that spawned
// the sending thread. Empty marks a root thread; there is no reserved value.
var ParentThreadIDKey = plugin.NewDataKey("agent-parent-thread-id", "")

const (
	PluginType = "agent-identity"

	// Claude Code: the session header carries the program root on every
	// request; a sub-agent sends its agent id, a nested one also the parent's.
	ClaudeCodeSessionHeader = "x-claude-code-session-id"
	ClaudeCodeAgentHeader   = "x-claude-code-agent-id"
	ClaudeCodeParentHeader  = "x-claude-code-parent-agent-id"

	// Codex: every thread sends thread-id, and a sub-agent adds the parent
	// thread's id. The session-id header carries the root thread's id.
	CodexSessionHeader = "session-id"
	// CodexSessionHeaderLegacy is the underscored form used by Codex 0.130.x;
	// kept as a fallback for the brief window before users upgrade.
	CodexSessionHeaderLegacy = "session_id"
	CodexThreadHeader        = "thread-id"
	CodexParentHeader        = "x-codex-parent-thread-id"

	// OpenCode: the affinity id names one thread, not a root; no other
	// OpenCode header is read (see README, Limitations).
	OpenCodeSessionHeader = "x-session-affinity"

	// Canonical llm-d identity headers: any harness or gateway can send the
	// identity directly, without harness-specific detection. A canonical
	// header wins over the harness-specific value for its field.
	CanonicalSessionHeader = "x-llm-d-session-id"
	CanonicalThreadHeader  = "x-llm-d-thread-id"
	CanonicalParentHeader  = "x-llm-d-parent-thread-id"
)

// AgentIdentity is the identity one source resolved from a request's
// headers: the session (program root), the thread that sent the request,
// and the thread that spawned it. An empty Parent marks a root thread;
// there is no reserved value.
type AgentIdentity struct {
	Session string
	Thread  string
	Parent  string
}

// HarnessHandler resolves the identity one source can see on a request: a
// harness, the canonical llm-d headers, or an operator's custom headers.
// Identify returns the fields the source resolved and whether the source
// recognized the request at all; the resolution chain decides which
// recognized source is used. The whole request is passed, not just its
// headers, so a handler may use more of it than the headers one day.
type HarnessHandler interface {
	Identify(request *scheduling.InferenceRequest) (AgentIdentity, bool)
}

// Parameters is the user-facing plugin configuration block. See the package
// README for a configmap example showing how the operator's extra session
// headers are checked ahead of the built-in harness handlers.
type Parameters struct {
	// AdditionalSessionHeaders is checked ahead of the built-in harness
	// handlers. Order is preserved; the first non-empty match wins.
	AdditionalSessionHeaders []string `json:"additionalSessionHeaders,omitempty"`
}

// PluginFactory is the factory function for the agent identity plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("agent-identity: failed to decode parameters: %w", err)
		}
	}

	// Request headers are stored with lowercased keys (see
	// handlers.HandleRequestHeaders), so the operator's extras must match.
	extras := make([]string, 0, len(params.AdditionalSessionHeaders))
	for _, h := range params.AdditionalSessionHeaders {
		extras = append(extras, strings.ToLower(h))
	}

	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		handlers: []HarnessHandler{
			canonicalHandler{},
			customHandler{sessionHeaders: extras},
			claudeCodeHandler{},
			openCodeHandler{},
			codexHandler{},
		},
	}, nil
}

// compile-time interface assertions
var (
	_ requestcontrol.RequestHeaderProcessor = &Plugin{}
	_ plugin.ProducerPlugin                 = &Plugin{}
)

// Plugin resolves agent identity from provider-specific headers and stores it
// as request attributes for use by other subsystems.
type Plugin struct {
	typedName plugin.TypedName
	// handlers is the resolution chain, in order: the canonical llm-d
	// headers, the operator's custom session headers, then the harness
	// handlers. The first handler that recognizes a request owns it.
	handlers []HarnessHandler
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the identity request attributes written by this
// request-header processor.
func (p *Plugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{
		AgentIdentityKey:  "",
		ThreadIDKey:       "",
		ParentThreadIDKey: "",
	}
}

// resolve returns the identity from the first handler that recognizes the
// request, in chain order. A recognized identity without a session is not an
// identity: the canonical handler owns the request when any of its headers
// is present, so a partial canonical stamp suppresses harness detection. A
// thread still empty defaults to the session, because a harness that sends
// no thread id has exactly one thread per session.
func (p *Plugin) resolve(request *scheduling.InferenceRequest) (AgentIdentity, bool) {
	for _, h := range p.handlers {
		if id, ok := h.Identify(request); ok {
			if id.Session == "" {
				return AgentIdentity{}, false
			}
			if id.Thread == "" {
				id.Thread = id.Session
			}
			return id, true
		}
	}
	return AgentIdentity{}, false
}

func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	id, ok := p.resolve(request)
	if !ok {
		return nil
	}
	request.PutAttribute(AgentIdentityKey, id.Session)
	request.PutAttribute(ThreadIDKey, id.Thread)
	request.PutAttribute(ParentThreadIDKey, id.Parent)
	return nil
}
