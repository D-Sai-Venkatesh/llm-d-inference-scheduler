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

package groupsched

import (
	"strings"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// stubPlugin is a minimal plugin carrying only a TypedName (the startup
// gate inspects nothing else).
type stubPlugin struct{ tn fwkplugin.TypedName }

func (s *stubPlugin) TypedName() fwkplugin.TypedName { return s.tn }

// TestFactoryRequiresTokenProducer: the startup fail-closed gate lives at
// RegisterDependencies (construction order is not config order; the gate
// runs after ALL plugins are built). It fails without a token-producer
// sibling and passes with one.
func TestFactoryRequiresTokenProducer(t *testing.T) {
	ctx := t.Context()

	build := func(t *testing.T) *GroupScheduler {
		t.Helper()
		handle := testutils.NewTestHandle(ctx)
		p, err := Factory("gs", nil, handle)
		if err != nil {
			t.Fatalf("Factory must succeed (the gate is not at construction): %v", err)
		}
		return p.(*GroupScheduler)
	}

	t.Run("without a token-producer", func(t *testing.T) {
		gs := build(t)
		err := gs.validateTokenProducer()
		if err == nil || !strings.Contains(err.Error(), "token-producer") {
			t.Fatalf("the gate must fail without a token-producer, got: %v", err)
		}
	})
	t.Run("with a token-producer", func(t *testing.T) {
		gs := build(t)
		gs.handle.AddPlugin("tp", &stubPlugin{tn: fwkplugin.TypedName{Type: tokenizer.PluginType, Name: "tp"}})
		if err := gs.validateTokenProducer(); err != nil {
			t.Fatalf("the gate must pass with a token-producer sibling: %v", err)
		}
	})
	t.Run("nil handle skips the gate (unit/test mode)", func(t *testing.T) {
		p, err := Factory("gs", nil, nil)
		if err != nil {
			t.Fatalf("Factory with nil handle: %v", err)
		}
		if err := p.(*GroupScheduler).validateTokenProducer(); err != nil {
			t.Fatalf("nil handle must skip the gate: %v", err)
		}
	})
}
