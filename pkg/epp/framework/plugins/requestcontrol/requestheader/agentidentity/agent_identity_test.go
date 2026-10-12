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
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newDefaultPlugin builds a Plugin with no parameters — the built-in defaults.
// All RequestHeader tests below use this so they exercise the same code
// path as production-default configs.
func newDefaultPlugin(t *testing.T) *Plugin {
	t.Helper()
	pi, err := PluginFactory("test", nil, nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	return pi.(*Plugin)
}

func TestRequestHeader(t *testing.T) {
	p := newDefaultPlugin(t)

	tests := []struct {
		name          string
		headers       map[string]string
		body          *fwkrh.InferenceRequestBody
		wantIdentity  string
		wantThread    string
		wantParent    string
		wantAttrFound bool
	}{
		// Compatibility cases: agent-identity must keep the value the plugin
		// produced before thread and parent existed.
		{
			name:          "claude code session header",
			headers:       map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantIdentity:  "session-abc",
			wantThread:    "session-abc",
			wantAttrFound: true,
		},
		{
			name:          "opencode session header",
			headers:       map[string]string{OpenCodeSessionHeader: "oc-session-1"},
			wantIdentity:  "oc-session-1",
			wantThread:    "oc-session-1",
			wantAttrFound: true,
		},
		{
			name:          "codex session header (hyphenated, >= 0.131.0)",
			headers:       map[string]string{CodexSessionHeader: "codex-session-1"},
			wantIdentity:  "codex-session-1",
			wantThread:    "codex-session-1",
			wantAttrFound: true,
		},
		{
			name:          "codex legacy session header (underscored, <= 0.130.x)",
			headers:       map[string]string{CodexSessionHeaderLegacy: "codex-legacy-1"},
			wantIdentity:  "codex-legacy-1",
			wantThread:    "codex-legacy-1",
			wantAttrFound: true,
		},
		{
			name: "priority order: codex hyphenated wins over legacy underscored",
			headers: map[string]string{
				CodexSessionHeader:       "codex-new",
				CodexSessionHeaderLegacy: "codex-old",
			},
			wantIdentity:  "codex-new",
			wantThread:    "codex-new",
			wantAttrFound: true,
		},
		{
			name: "priority order: claude code wins over opencode",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantIdentity:  "session-abc",
			wantThread:    "session-abc",
			wantAttrFound: true,
		},
		{
			name: "priority order: opencode wins over codex",
			headers: map[string]string{
				OpenCodeSessionHeader: "oc-session-1",
				CodexSessionHeader:    "codex-session-1",
			},
			wantIdentity:  "oc-session-1",
			wantThread:    "oc-session-1",
			wantAttrFound: true,
		},
		{
			name:          "previous_response_id in body is ignored",
			headers:       map[string]string{},
			body:          &fwkrh.InferenceRequestBody{Payload: fwkrh.PayloadMap{"previous_response_id": "resp-456"}},
			wantAttrFound: false,
		},
		{
			name: "whitespace-only header falls through to next priority header",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "   ",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantIdentity:  "oc-session-1",
			wantThread:    "oc-session-1",
			wantAttrFound: true,
		},
		{
			name: "whitespace-only header with no fallback leaves no attribute",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "   ",
			},
			wantAttrFound: false,
		},
		{
			name: "leading and trailing whitespace is trimmed",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "  session-abc  ",
			},
			wantIdentity:  "session-abc",
			wantThread:    "session-abc",
			wantAttrFound: true,
		},
		{
			name:          "nil body does not panic",
			headers:       map[string]string{},
			body:          nil,
			wantAttrFound: false,
		},
		{
			name:          "no matching headers leaves no attribute",
			headers:       map[string]string{"x-unrelated": "value"},
			wantAttrFound: false,
		},
		{
			name:          "empty headers leaves no attribute",
			headers:       map[string]string{},
			wantAttrFound: false,
		},
		{
			name:          "nil headers does not panic",
			headers:       nil,
			wantAttrFound: false,
		},
		// Thread and parent extraction per harness.
		{
			name: "claude code depth-1 sub-agent: agent id, no parent header",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				ClaudeCodeAgentHeader:   "agent-1",
			},
			wantIdentity:  "session-abc",
			wantThread:    "agent-1",
			wantParent:    "session-abc",
			wantAttrFound: true,
		},
		{
			name: "claude code nested sub-agent: agent and parent headers",
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				ClaudeCodeAgentHeader:   "agent-2",
				ClaudeCodeParentHeader:  "agent-1",
			},
			wantIdentity:  "session-abc",
			wantThread:    "agent-2",
			wantParent:    "agent-1",
			wantAttrFound: true,
		},
		{
			name: "codex main thread: thread id, no parent",
			headers: map[string]string{
				CodexSessionHeader: "codex-root",
				CodexThreadHeader:  "thread-1",
			},
			wantIdentity:  "codex-root",
			wantThread:    "thread-1",
			wantAttrFound: true,
		},
		{
			name: "codex sub-agent: parent thread id",
			headers: map[string]string{
				CodexSessionHeader: "codex-root",
				CodexThreadHeader:  "thread-2",
				CodexParentHeader:  "thread-1",
			},
			wantIdentity:  "codex-root",
			wantThread:    "thread-2",
			wantParent:    "thread-1",
			wantAttrFound: true,
		},
		// Canonical llm-d headers own the request.
		{
			name: "canonical session wins over claude code session",
			headers: map[string]string{
				CanonicalSessionHeader:  "llmd-session",
				ClaudeCodeSessionHeader: "session-abc",
			},
			wantIdentity:  "llmd-session",
			wantThread:    "llmd-session",
			wantAttrFound: true,
		},
		{
			name: "canonical all three fields",
			headers: map[string]string{
				CanonicalSessionHeader: "llmd-session",
				CanonicalThreadHeader:  "llmd-thread",
				CanonicalParentHeader:  "llmd-parent",
			},
			wantIdentity:  "llmd-session",
			wantThread:    "llmd-thread",
			wantParent:    "llmd-parent",
			wantAttrFound: true,
		},
		{
			name: "canonical thread with no parent is its own root; harness headers ignored",
			headers: map[string]string{
				CanonicalThreadHeader:   "llmd-thread",
				ClaudeCodeSessionHeader: "session-abc",
			},
			wantIdentity:  "llmd-thread",
			wantThread:    "llmd-thread",
			wantAttrFound: true,
		},
		{
			name: "canonical thread with parent but no session leaves no attribute",
			headers: map[string]string{
				CanonicalThreadHeader: "llmd-thread",
				CanonicalParentHeader: "llmd-parent",
			},
			wantAttrFound: false,
		},
		{
			name:          "canonical parent alone leaves no attribute",
			headers:       map[string]string{CanonicalParentHeader: "llmd-parent"},
			wantAttrFound: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				Headers: tt.headers,
				Body:    tt.body,
			}
			err := p.RequestHeader(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey)
			if ok != tt.wantAttrFound {
				t.Fatalf("agent-identity attribute found = %v, want %v", ok, tt.wantAttrFound)
			}
			if !tt.wantAttrFound {
				return
			}
			if got != tt.wantIdentity {
				t.Errorf("agent identity = %q, want %q", got, tt.wantIdentity)
			}
			thread, ok := scheduling.ReadRequestAttribute[string](req, ThreadIDKey)
			if !ok {
				t.Fatal("agent-thread-id attribute not found")
			}
			if thread != tt.wantThread {
				t.Errorf("thread id = %q, want %q", thread, tt.wantThread)
			}
			parent, ok := scheduling.ReadRequestAttribute[string](req, ParentThreadIDKey)
			if !ok {
				t.Fatal("agent-parent-thread-id attribute not found")
			}
			if parent != tt.wantParent {
				t.Errorf("parent thread id = %q, want %q", parent, tt.wantParent)
			}
			// FairnessID must never be set by the plugin.
			if req.FairnessID != "" {
				t.Errorf("FairnessID = %q, want empty (plugin must not set FairnessID)", req.FairnessID)
			}
		})
	}
}

func TestPluginFactory(t *testing.T) {
	t.Run("default chain order", func(t *testing.T) {
		pi, err := PluginFactory("test", nil, nil)
		if err != nil {
			t.Fatalf("PluginFactory: %v", err)
		}
		handlers := pi.(*Plugin).handlers
		want := []string{
			"agentidentity.canonicalHandler", "agentidentity.customHandler", "agentidentity.claudeCodeHandler", "agentidentity.openCodeHandler", "agentidentity.codexHandler",
		}
		if len(handlers) != len(want) {
			t.Fatalf("chain has %d handlers, want %d", len(handlers), len(want))
		}
		// The chain order is the resolution order: canonical llm-d headers,
		// the operator's custom session headers, then the harness handlers.
		for i, name := range want {
			if got := fmt.Sprintf("%T", handlers[i]); got != name {
				t.Errorf("handlers[%d] = %s, want %s", i, got, name)
			}
		}
	})

	t.Run("additional session headers are lowercased into the custom handler", func(t *testing.T) {
		pi, err := PluginFactory("test",
			fwkplugin.StrictDecoder(json.RawMessage(`{"additionalSessionHeaders":["X-Tenant-ID","x-user-session"]}`)), nil)
		if err != nil {
			t.Fatalf("PluginFactory: %v", err)
		}
		custom, ok := pi.(*Plugin).handlers[1].(customHandler)
		if !ok {
			t.Fatal("handlers[1] is not customHandler")
		}
		want := []string{"x-tenant-id", "x-user-session"}
		if !reflect.DeepEqual(custom.sessionHeaders, want) {
			t.Errorf("custom session headers = %v, want %v", custom.sessionHeaders, want)
		}
	})

	t.Run("malformed json", func(t *testing.T) {
		if _, err := PluginFactory("test",
			fwkplugin.StrictDecoder(json.RawMessage(`{"additionalSessionHeaders": not-json}`)), nil); err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

// TestRequestHeader_CustomHeader proves end-to-end that a header added
// via additionalSessionHeaders is honored at request time.
func TestRequestHeader_CustomHeader(t *testing.T) {
	pi, err := PluginFactory("test",
		fwkplugin.StrictDecoder(json.RawMessage(`{"additionalSessionHeaders":["x-tenant-id"]}`)), nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{"x-tenant-id": "tenant-42"},
	}
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	got, ok := scheduling.ReadRequestAttribute[string](req, AgentIdentityKey)
	if !ok {
		t.Fatal("agent-identity attribute not found")
	}
	if got != "tenant-42" {
		t.Errorf("agent identity = %q, want %q", got, "tenant-42")
	}

	// And it wins over a default-bucket header (because the custom handler
	// sits ahead of the harness handlers in the chain).
	req2 := &scheduling.InferenceRequest{
		Headers: map[string]string{
			"x-tenant-id":           "tenant-42",
			ClaudeCodeSessionHeader: "claude-session",
		},
	}
	if err := p.RequestHeader(context.Background(), req2); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
	got2, ok2 := scheduling.ReadRequestAttribute[string](req2, AgentIdentityKey)
	if !ok2 {
		t.Fatal("agent-identity attribute not found")
	}
	if got2 != "tenant-42" {
		t.Errorf("agent identity = %q, want %q (custom should win)", got2, "tenant-42")
	}
}

// TestAgentIdentityKeyMatchesDocumentedConfig pins the published key to the
// name session-affinity sources use in configuration ("attribute:
// agent-identity"). A drift makes the session-affinity fallback silently
// resolve nothing.
func TestAgentIdentityKeyMatchesDocumentedConfig(t *testing.T) {
	const documented = "agent-identity"
	if got := fwkplugin.NewDataKey(documented, ""); got != AgentIdentityKey {
		t.Errorf("config %q resolves to %s, want %s", documented, got, AgentIdentityKey)
	}
}

func TestProduces(t *testing.T) {
	t.Parallel()

	p := &Plugin{}
	produced := p.Produces()
	for _, key := range []fwkplugin.DataKey{AgentIdentityKey, ThreadIDKey, ParentThreadIDKey} {
		value, ok := produced[key]
		if !ok {
			t.Fatalf("Produces() does not declare %s", key)
		}
		if _, ok := value.(string); !ok {
			t.Fatalf("Produces()[%s] has type %T, want string", key, value)
		}
	}
}
