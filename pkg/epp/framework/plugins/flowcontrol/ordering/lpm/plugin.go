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
	GenerationIntervalSeconds int `json:"generationIntervalSeconds,omitempty"`
}

func (p *Parameters) setDefaults() {
	if p.GenerationIntervalSeconds == 0 {
		p.GenerationIntervalSeconds = DefaultGenerationIntervalSeconds
	}
}

func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("least-prefix-plugin: failed to decode parameters: %w", err)
		}
	}
	params.setDefaults()

	p :=  &Plugin{
		typedName:       plugin.TypedName{Type: PluginType, Name: name},
	}

	if handle == nil {
		return nil, errors.New("least-prefix-plugin: plugin handle is required")
	}
	go p.runGenerationTicker(handle.Context(), params.GenerationIntervalSeconds)

	return p, nil
}

var _ flowcontrol.ScoringOrderingPolicy = &Plugin{}

type Plugin struct {
	typedName plugin.TypedName
	generation atomic.Uint64
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) Less(a, b flowcontrol.QueueItemAccessor) bool {
	return true
}

func (p *Plugin) Score(item flowcontrol.QueueItemAccessor) float64 {
	return 0
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
