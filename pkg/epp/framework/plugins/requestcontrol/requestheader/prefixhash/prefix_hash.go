package prefixhash

import (
	"fmt"
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

const (
	PluginType = "prefix-hash"
	DefaultBlockSizeTokens = 16
	DefaultMaxPrefixBlocks = 2048
)

var PrefixHashKey = plugin.NewDataKey("prefix-hash", "")

var _ requestcontrol.RequestHeaderProcessor = &Plugin{}

type Parameters struct {
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	MaxPrefixBlocks int `json:"maxPrefixBlocks,omitempty"`
}

func (p *Parameters) setDefaults() {
	if p.BlockSizeTokens == 0 {
		p.BlockSizeTokens = DefaultBlockSizeTokens
	}

	if p.MaxPrefixBlocks == 0 {
		p.MaxPrefixBlocks = DefaultMaxPrefixBlocks
	}
}

func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("prefix-hash: failed to decode parameters: %w", err)
		}
	}

	params.setDefaults()

	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		blockSizeTokens: params.BlockSizeTokens,
		maxPrefixBlocks: params.MaxPrefixBlocks,
	}, nil
}


type Plugin struct {
	typedName       plugin.TypedName
	blockSizeTokens int
	maxPrefixBlocks int
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) RequestHeader(ctx context.Context, request *scheduling.InferenceRequest) error {
	request.PutAttribute(PrefixHashKey, prefixhash.GetBlockHashes(ctx, request, p.blockSizeTokens, p.maxPrefixBlocks))
	return nil
}



