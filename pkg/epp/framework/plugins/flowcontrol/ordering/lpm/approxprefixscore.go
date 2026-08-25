package lpm

import (
	"encoding/json"
	"errors"
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
	// ApproxPrefixCacheProducerName is the approximateprefix instance to resolve via handle. There
	// is no type-based default: handle.AddPlugin registers plugins under their config-supplied
	// name verbatim, including the empty string when a name is omitted, so a type-keyed default
	// here would only ever resolve a producer instance that happens to be named after its own
	// type. Required; NewApproxPrefixScoringStrategy fails loud if unset.
	ApproxPrefixCacheProducerName string `json:"approxPrefixCacheProducerName,omitempty"`
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
	if params.ApproxPrefixCacheProducerName == "" {
		return nil, errors.New("approx-prefix-scoring-strategy: approxPrefixCacheProducerName is required")
	}

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

// Score returns the longest prefix-match depth, in blocks, that any single server holds across
// every prompt in the request. It reads the block hashes prefixhash-producer already computed
// pre-admission rather than recomputing them.
//
// Per prompt, it tracks each server's running match count rather than counting "some server has
// this block": a block that only server B holds does not extend a match server A is building, so
// per-server counts can diverge starting at the first block where their cached sets differ. The
// walk still stops at the first block no server has at all, matching
// approximateprefix.matchLongestPrefix, but the depth reported is the max over per-server counts,
// not the count of blocks that had at least one cache hit somewhere.
func (s *ApproxPrefixScoringStrategy) Score(item flowcontrol.QueueItemAccessor) float64 {
	req := item.OriginalRequest().InferenceRequest()

	perPromptHashes, ok := scheduling.ReadRequestAttribute[[][]dataprodprefixhash.BlockHash](req, reqheaderprefixhash.PrefixHashKey)
	if !ok || len(perPromptHashes) == 0 {
		return 0
	}

	var maxDepth int
	for _, hashes := range perPromptHashes {
		if depth := s.matchDepth(hashes); depth > maxDepth {
			maxDepth = depth
		}
	}
	return float64(maxDepth)
}

// matchDepth returns the longest prefix-match depth, in blocks, that any single server in the
// indexer holds for hashes. The walk stops at the first block no server has at all, then reports
// the largest per-server running count reached up to that point.
func (s *ApproxPrefixScoringStrategy) matchDepth(hashes []dataprodprefixhash.BlockHash) int {
	counts := make(map[approximateprefix.ServerID]int)
	maxDepth := 0
	for _, hash := range hashes {
		servers := s.indexer.Get(hash)
		if len(servers) == 0 {
			break
		}
		for server := range servers {
			counts[server]++
			if counts[server] > maxDepth {
				maxDepth = counts[server]
			}
		}
	}
	return maxDepth
}
