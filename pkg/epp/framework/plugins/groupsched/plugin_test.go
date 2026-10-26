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

// TestFactoryRequiresTokenProducer: construction fails without a
// token-producer sibling on the handle (the startup fail-closed gate) and
// succeeds with one.
func TestFactoryRequiresTokenProducer(t *testing.T) {
	ctx := t.Context()

	t.Run("without a token-producer", func(t *testing.T) {
		handle := testutils.NewTestHandle(ctx)
		_, err := Factory("gs", nil, handle)
		if err == nil || !strings.Contains(err.Error(), "token-producer") {
			t.Fatalf("construction must fail without a token-producer, got: %v", err)
		}
	})
	t.Run("with a token-producer", func(t *testing.T) {
		handle := testutils.NewTestHandle(ctx)
		handle.AddPlugin("tp", &stubPlugin{tn: fwkplugin.TypedName{Type: tokenizer.PluginType, Name: "tp"}})
		if _, err := Factory("gs", nil, handle); err != nil {
			t.Fatalf("construction must succeed with a token-producer sibling: %v", err)
		}
	})
	t.Run("nil handle skips the gate (unit/test mode)", func(t *testing.T) {
		if _, err := Factory("gs", nil, nil); err != nil {
			t.Fatalf("nil handle must skip the startup gate: %v", err)
		}
	})
}
