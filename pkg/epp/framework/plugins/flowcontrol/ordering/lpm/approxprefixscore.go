package lpm

import (
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
	dataprodprefixhash "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	reqheaderprefixhash "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/prefixhash"
)

const (
	ApproxPrefixScoringStrategyType = "approx-prefix-scoring-strategy"
)

type ApproxPrefixScoringStrategyParameters struct {
	ApproxPrefixCacheProducerName string `json:"approxPrefixCacheProducerName,omitempty"`
}

func (p *ApproxPrefixScoringStrategyParameters) setDefaults() {
	if p.ApproxPrefixCacheProducerName == "" {
		p.ApproxPrefixCacheProducerName = approximateprefix.ApproxPrefixCachePluginType
	}
}

// NewApproxPrefixScoringStrategy resolves the named approximateprefix producer via handle and
// wraps its indexer. raw is the strategy's own "parameters" sub-object from config, decoded here
// rather than by the caller since only this constructor knows its shape.
func NewApproxPrefixScoringStrategy(raw json.RawMessage, handle plugin.Handle) (*ApproxPrefixScoringStrategy, error) {
	var params ApproxPrefixScoringStrategyParameters
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, fmt.Errorf("approx-prefix-scoring-strategy: failed to decode parameters: %w", err)
		}
	}
	params.setDefaults()

	producer, err := plugin.PluginByType[*approximateprefix.DataProducer](handle, params.ApproxPrefixCacheProducerName)
	if err != nil {
		return nil, fmt.Errorf("approx-prefix-scoring-strategy: resolving %q: %w", params.ApproxPrefixCacheProducerName, err)
	}

	return &ApproxPrefixScoringStrategy{
		indexer: producer.Indexer(),
	}, nil
}

type ApproxPrefixScoringStrategy struct {
	indexer approximateprefix.IndexerInterface
}

// Score returns the longest prefix-match depth, in blocks, across every server the shared
// indexer knows about. It reads the block hashes prefixhash-producer already computed
// pre-admission rather than recomputing them, and stops at the first block no server has --
// the same greedy walk approximateprefix.matchLongestPrefix performs internally.
func (s *ApproxPrefixScoringStrategy) Score(item flowcontrol.QueueItemAccessor) float64 {
	req := item.OriginalRequest().InferenceRequest()

	perPromptHashes, ok := scheduling.ReadRequestAttribute[[][]dataprodprefixhash.BlockHash](req, reqheaderprefixhash.PrefixHashKey)
	if !ok || len(perPromptHashes) == 0 {
		return 0
	}

	var depth int
	for _, hash := range perPromptHashes[0] {
		if len(s.indexer.Get(hash)) == 0 {
			break
		}
		depth++
	}
	return float64(depth)
}