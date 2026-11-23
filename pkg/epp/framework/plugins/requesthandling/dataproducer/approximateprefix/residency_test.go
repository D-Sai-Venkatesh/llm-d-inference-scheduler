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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
)

// The residency read answers per endpoint with the leading run the index holds, never refreshes
// recency, and reports the sizing a reader must validate.
func TestResidencyReader(t *testing.T) {
	cfg := config{AutoTune: false, BlockSizeTokens: 64, LRUCapacityPerServer: 4,
		MaxPrefixTokensToMatch: 0, MaxPrefixBlocksToMatch: 0}
	p, err := newDataProducer(context.Background(), "cpu-residency", cfg, testHandle())
	require.NoError(t, err)

	var r ResidencyReader = p
	a := k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"}
	b := k8stypes.NamespacedName{Namespace: "ns", Name: "pod-b"}
	keys := []blockHash{11, 12, 13}
	p.indexerInst.Add(keys, server{ServerID: ServerID(a), NumOfGPUBlocks: 4})

	require.Equal(t, []int{3, 0}, r.ResidentPrefixBlocks(keys, []string{a.String(), b.String()}))
	require.Equal(t, []int{2}, r.ResidentPrefixBlocks([]blockHash{11, 12, 99}, []string{a.String()}),
		"the run stops at the first block the endpoint does not hold")
	require.Equal(t, []int{0, 0}, r.ResidentPrefixBlocks(nil, []string{a.String(), b.String()}))

	// a READ must not keep an entry alive. LRU (capacity 4) after the next Add: 31, 11, 12, 13
	// (13 oldest). Reading the first prompt must not move it ahead of 31, so the following Add evicts
	// 13 — had the read refreshed it, 31 would have been the victim.
	p.indexerInst.Add([]blockHash{31}, server{ServerID: ServerID(a), NumOfGPUBlocks: 4})
	_ = r.ResidentPrefixBlocks(keys, []string{a.String()})
	p.indexerInst.Add([]blockHash{41}, server{ServerID: ServerID(a), NumOfGPUBlocks: 4})
	require.Equal(t, []int{2}, r.ResidentPrefixBlocks(keys, []string{a.String()}),
		"reads must not refresh recency: block 13 was the oldest entry and is gone")
	require.Equal(t, []int{1}, r.ResidentPrefixBlocks([]blockHash{31}, []string{a.String()}))

	got := r.ResidencyConfig()
	require.Equal(t, ResidencyIndexConfig{AutoTune: false, BlockSizeTokens: 64, CapacityBlocks: 4, UnlimitedPrefix: true}, got)

	capped, err := newDataProducer(context.Background(), "capped", config{AutoTune: false, BlockSizeTokens: 64,
		LRUCapacityPerServer: 4, MaxPrefixTokensToMatch: 131072}, testHandle())
	require.NoError(t, err)
	require.False(t, capped.ResidencyConfig().UnlimitedPrefix, "a token cap records only a prompt's leading blocks")
}

// The reader is reachable through the plugin handle by name, typed by the exported interface.
func TestResidencyReaderThroughHandle(t *testing.T) {
	h := testHandle()
	p, err := newDataProducer(context.Background(), "cpu-residency", config{AutoTune: false, BlockSizeTokens: 64,
		LRUCapacityPerServer: 8}, h)
	require.NoError(t, err)
	h.AddPlugin("cpu-residency", p)
	r, err := plugin.PluginByType[ResidencyReader](h, "cpu-residency")
	require.NoError(t, err)
	require.Equal(t, 8, r.ResidencyConfig().CapacityBlocks)
}

// The keys a reader asks with are the chain keys thread identity publishes:
// prefixhash.GetBlockHashesWithPromptTokens over the request's TokenizedRequest at the same block size
// with an effectively unbounded cap (thread identity's maxChainBlocks). An uncapped residency index
// records exactly those keys at PreRequest on the endpoint the request was sent to, so a read with
// them finds the whole prompt there and nothing elsewhere.
func TestResidencyKeysAreTheThreadChainKeys(t *testing.T) {
	cfg := config{AutoTune: false, BlockSizeTokens: 64, LRUCapacityPerServer: 1 << 16,
		MaxPrefixTokensToMatch: 0, MaxPrefixBlocksToMatch: 0}
	p, err := newDataProducer(context.Background(), "cpu-residency", cfg, testHandle())
	require.NoError(t, err)

	a := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "dasari", Name: "decode-a-rank-0"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	b := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "dasari", Name: "decode-b-rank-0"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	tokens := make([]uint32, 64*2500+17) // 2,500 full blocks and a partial one: past the default 2,048-block cap, as DeepSWE prompts are
	for i := range tokens {
		tokens[i] = uint32(i*7 + 3)
	}
	req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m", Body: tokenizedBody(tokens)}
	require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{a, b}))
	require.NoError(t, p.PreRequest(context.Background(), req, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{a}}},
	}))
	p.wg.Wait()

	chains, promptTokens := prefixhash.GetBlockHashesWithPromptTokens(context.Background(), req, 64, 1<<20)
	require.Len(t, chains, 1)
	require.Equal(t, len(tokens), promptTokens[0])
	got := p.ResidentPrefixBlocks(chains[0], []string{a.GetMetadata().ID.String(), b.GetMetadata().ID.String()})
	require.Equal(t, []int{len(chains[0]), 0}, got, "the endpoint the request went to holds the whole chain; the other none")
}
