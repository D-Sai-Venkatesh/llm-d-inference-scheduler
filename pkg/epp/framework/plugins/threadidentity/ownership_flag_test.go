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

package threadidentity

import (

	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// fakeSibling is a configurable sibling plugin for the ownership-check tests:
// it can declare itself a producer and/or consumer of the flow-key attribute.
type fakeSibling struct {
	name      string
	produces  bool
	consumes  bool
	lastError error
}

func (f *fakeSibling) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake", Name: f.name}
}
func (f *fakeSibling) RegisterDependencies(r datalayer.Registrar) error { return nil }
func (f *fakeSibling) Produces() map[fwkplugin.DataKey]any {
	if !f.produces {
		return nil
	}
	return map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""}
}
func (f *fakeSibling) Consumes() fwkplugin.DataDependencies {
	if !f.consumes {
		return fwkplugin.DataDependencies{}
	}
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""}}
}

// buildForOwnership builds the plugin with the given allowKeyConsumers setting
// and a handle already carrying the siblings.
func buildForOwnership(t *testing.T, allowConsumers bool, siblings ...*fakeSibling) *Plugin {
	t.Helper()
	params := `{"tokenProducer":{"modelName":"test-model","vllm":{"url":"http://127.0.0.1:1"}}`
	if allowConsumers {
		params += `,"allowKeyConsumers":true`
	}
	params += `}`
	handle := testutils.NewTestHandle(t.Context())
	adder := handle.(interface {
		AddPlugin(string, plugin.Plugin)
	})
	for _, s := range siblings {
		adder.AddPlugin(s.name, s)
	}
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(params)), handle)
	require.NoError(t, err)
	return built.(*Plugin)
}

// TestOwnershipCheck_DefaultRefusesConsumers: without the flag, a consumer sibling
// (thunder-agent's shape) is refused — the unchanged, fail-closed default.
func TestOwnershipCheck_DefaultRefusesConsumers(t *testing.T) {
	p := buildForOwnership(t, false, &fakeSibling{name: "thunder", consumes: true})
	err := p.RegisterDependencies(nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "consumes it")
}

// TestOwnershipCheck_FlagAllowsConsumersButStillRefusesProducers: with
// allowKeyConsumers the consumer sibling is permitted, while a second WRITER of
// the flow key is still refused — the producer half is never relaxed.
func TestOwnershipCheck_FlagAllowsConsumersButStillRefusesProducers(t *testing.T) {
	p := buildForOwnership(t, true, &fakeSibling{name: "thunder", consumes: true})
	require.NoError(t, p.RegisterDependencies(nil), "a declared consumer must be allowed under the flag")

	p2 := buildForOwnership(t, true, &fakeSibling{name: "other-writer", produces: true})
	err := p2.RegisterDependencies(nil)
	require.Error(t, err, "a second producer must be refused even under the flag")
	require.Contains(t, err.Error(), "produces it")
}

// TestOwnershipCheck_NoSiblingsUnchanged: with no siblings the check passes with
// or without the flag — the flag alone changes nothing.
func TestOwnershipCheck_NoSiblingsUnchanged(t *testing.T) {
	for _, flag := range []bool{false, true} {
		p := buildForOwnership(t, flag)
		require.NoError(t, p.RegisterDependencies(nil), "no siblings, flag=%v", flag)
	}
}

// TestOwnershipCheck_FlagUnknownByDefault: the rendered default config (no
// allowKeyConsumers key) decodes to false — the field is strictly opt-in.
func TestOwnershipCheck_FlagUnknownByDefault(t *testing.T) {
	p := buildForOwnership(t, false, &fakeSibling{name: "thunder", consumes: true})
	require.False(t, p.allowKeyConsumers)
}
