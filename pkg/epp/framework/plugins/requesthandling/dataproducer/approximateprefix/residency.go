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
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
)

// ResidencyReader is the READ-ONLY view of this producer's per-endpoint index that another plugin
// reaches through the plugin handle (handle.Plugin(name)): how many LEADING blocks of a prompt's
// chained block keys the producer believes each endpoint still holds.
//
// The index records a prompt's blocks at PreRequest on the endpoint the request was sent to and
// never records output tokens, which is what an engine's CPU offload tier holds too (it stores the
// prompt blocks the engine computed). Sized to that tier (autoTune false, lruCapacityPerServer =
// tier tokens / blockSizeTokens) the index is an estimate of what is still on each endpoint's tier.
//
// A read never refreshes recency (it uses Contains only), so a reader cannot keep an entry alive.
type ResidencyReader interface {
	plugin.Plugin
	// ResidentPrefixBlocks returns, positionally per endpoint id ("namespace/name", the endpoint
	// metadata ID's String()), the number of leading blocks of keys the index holds for it.
	ResidentPrefixBlocks(keys []prefixhash.BlockHash, endpoints []string) []int
	// ResidencyConfig is the index's sizing, so a reader can refuse an index that does not model
	// what it needs (an autoTuned index is sized to the GPU, not the tier).
	ResidencyConfig() ResidencyIndexConfig
}

// ResidencyIndexConfig is the sizing a ResidencyReader answers with.
type ResidencyIndexConfig struct {
	// AutoTune reports that the block size and capacity follow the endpoint's metrics (GPU blocks).
	AutoTune bool
	// BlockSizeTokens is the block size the index hashes at (after the minimum clamp).
	BlockSizeTokens int
	// CapacityBlocks is the per-endpoint LRU capacity in blocks (lruCapacityPerServer).
	CapacityBlocks int
	// UnlimitedPrefix reports that whole prompts are hashed and recorded (both match caps 0). A
	// capped index records only a prompt's leading blocks and under-counts the tier's occupancy.
	UnlimitedPrefix bool
}

var _ ResidencyReader = (*dataProducer)(nil)

// ResidentPrefixBlocks implements ResidencyReader.
func (p *dataProducer) ResidentPrefixBlocks(keys []prefixhash.BlockHash, endpoints []string) []int {
	if p.indexerInst == nil || len(keys) == 0 {
		return make([]int, len(endpoints))
	}
	candidates := make([]ServerID, len(endpoints))
	for i, e := range endpoints {
		candidates[i] = serverIDFromString(e)
	}
	return p.indexerInst.MatchLongestPrefix(keys, candidates)
}

// ResidencyConfig implements ResidencyReader.
func (p *dataProducer) ResidencyConfig() ResidencyIndexConfig {
	blockSize := max(p.config.BlockSizeTokens, minBlockSizeTokens)
	return ResidencyIndexConfig{
		AutoTune:        p.config.AutoTune,
		BlockSizeTokens: blockSize,
		CapacityBlocks:  p.config.LRUCapacityPerServer,
		UnlimitedPrefix: p.resolveMaxBlocks(blockSize) == unlimitedPrefixBlocks,
	}
}

// serverIDFromString inverts ServerID.String() ("namespace/name"; a bare name has no namespace).
func serverIDFromString(id string) ServerID {
	if ns, name, ok := strings.Cut(id, "/"); ok {
		return ServerID{Namespace: ns, Name: name}
	}
	return ServerID{Name: id}
}
