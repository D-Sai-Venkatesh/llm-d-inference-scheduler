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

package runner

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestGroupSchedulerArmConfigLoads is the CP5c dry run: the prepared
// groupsched arm config (token-producer with the vllm render backend +
// group-scheduler with the §9 defaults) parses AND instantiates its
// plugins through both configuration phases — the same gates the stable
// configs promise. No cluster interaction.
func TestGroupSchedulerArmConfigLoads(t *testing.T) {
	path := "/Users/sai/.sai-ka-knowledge-bank/explore/ring-window-v2/cluster-prep/groupsched-plugins.yaml"
	configText, err := os.ReadFile(path)
	require.NoError(t, err, "the prepared config must exist")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = string(configText)
	opts.PoolName = "groupsched-dry-run"
	// groupsched is Alpha (like the siblings): experimental plugins allowed
	opts.AllowExperimentalPlugins = true

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err, "the groupsched arm config must parse")

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err, "the groupsched arm config must instantiate its plugins (token-producer + group-scheduler + the Consumes gate)")
}
