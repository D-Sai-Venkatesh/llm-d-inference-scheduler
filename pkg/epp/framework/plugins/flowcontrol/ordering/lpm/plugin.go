package lpm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	PluginType = "least-prefix-plugin"
	DefaultGenerationIntervalSeconds = 1
)

type Parameters struct {
	GenerationIntervalSeconds int                         `json:"generationIntervalSeconds,omitempty"`
	Strategies                []ScoringStrategyParameters `json:"strategies,omitempty"`

	// PrefixCacheProducerName is the approximateprefix instance the approx-prefix-scoring-strategy
	// entry in Strategies reads from. It lives here, as a plain top-level string, rather than
	// inside that entry's own Parameters (json.RawMessage), because findPluginDependencies only
	// reflects into direct or nested-struct fields -- not slice elements or json.RawMessage -- so
	// buildPluginDAG can only see the dependency this way. StrategyFactory passes it down to
	// NewApproxPrefixScoringStrategy instead of that constructor decoding its own copy.
	PrefixCacheProducerName string `json:"prefixCacheProducerName,omitempty" pluginRef:""`
}

func (p *Parameters) setDefaults() {
	if p.GenerationIntervalSeconds == 0 {
		p.GenerationIntervalSeconds = DefaultGenerationIntervalSeconds
	}
}

// StrategyFactory constructs the ScoringStrategy named by params.Type, dispatching to that
// strategy's own constructor, and pairs it with params.Weight for Plugin.Score's sum.
func StrategyFactory(params ScoringStrategyParameters, handle plugin.Handle) (ScoringStrategyWithWeights, error) {
	switch params.Type {
	case ApproxPrefixScoringStrategyType:
		strat, err := NewApproxPrefixScoringStrategy(params.Parameters, handle)
		if err != nil {
			return ScoringStrategyWithWeights{}, fmt.Errorf("least-prefix-plugin: strategy %q: %w", params.Type, err)
		}
		return ScoringStrategyWithWeights{Type: params.Type, Weight: params.Weight, Strategy: strat}, nil
	default:
		return ScoringStrategyWithWeights{}, fmt.Errorf("least-prefix-plugin: unknown strategy type %q", params.Type)
	}
}

// ConfigParser decodes this plugin's raw config into Parameters. Registered via
// RegisterWithPluginDependencies so buildPluginDAG can reflect over the result for
// PrefixCacheProducerName before any plugin is constructed. PluginFactory calls this same
// function rather than decoding independently, so there is one decode, shared.
func ConfigParser(rawParameters *json.Decoder, _ plugin.Handle) (any, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("least-prefix-plugin: failed to decode parameters: %w", err)
		}
	}
	params.setDefaults()
	return params, nil
}

func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("least-prefix-plugin: plugin handle is required")
	}

	tmpParams, err := ConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	params := tmpParams.(Parameters)

	strategies := make([]ScoringStrategyWithWeights, 0, len(params.Strategies))
	for _, sp := range params.Strategies {
		if sp.Type == ApproxPrefixScoringStrategyType {
			fields := map[string]any{}
			if len(sp.Parameters) > 0 {
				if err := json.Unmarshal(sp.Parameters, &fields); err != nil {
					return nil, fmt.Errorf("least-prefix-plugin: failed to decode strategy parameters: %w", err)
				}
			}
			fields["approxPrefixCacheProducerName"] = params.PrefixCacheProducerName

			raw, err := json.Marshal(fields)
			if err != nil {
				return nil, fmt.Errorf("least-prefix-plugin: %w", err)
			}
			sp.Parameters = raw
		}
		sw, err := StrategyFactory(sp, handle)
		if err != nil {
			return nil, err
		}
		strategies = append(strategies, sw)
	}

	p := &Plugin{
		typedName:         plugin.TypedName{Type: PluginType, Name: name},
		scoringStrategies: strategies,
	}
	go p.runGenerationTicker(handle.Context(), params.GenerationIntervalSeconds)

	return p, nil
}

var _ flowcontrol.ScoringOrderingPolicy = &Plugin{}

type Plugin struct {
	typedName         plugin.TypedName
	generation        atomic.Uint64
	scoringStrategies []ScoringStrategyWithWeights
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) Less(a, b flowcontrol.QueueItemAccessor) bool {
	return flowcontrol.CompareByScore(p, a, b)
}

// Score sums every active strategy's score, weighted per its configured weight.
func (p *Plugin) Score(item flowcontrol.QueueItemAccessor) float64 {
	var total float64
	for _, sw := range p.scoringStrategies {
		total += float64(sw.Weight) * sw.Strategy.Score(item)
	}
	return total
}

func (p *Plugin) runGenerationTicker(ctx context.Context, intervalSeconds int) {
	ticker := time.NewTicker(time.Duration(intervalSeconds) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <- ctx.Done():
			return
		case <- ticker.C:
			p.generation.Add(1)
		}
	}
}

func (p *Plugin) Generation() uint64 {
	return p.generation.Load()
}
