package lpm

import (
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

type ScoringStrategyParameters struct {
	Type       string          `json:"type,omitempty"`
	Weight     int             `json:"weight,omitempty"`
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

type ScoringStrategyWithWeights struct {
	Type     string
	Weight   int
	Strategy ScoringStrategy
}

type ScoringStrategy interface {
	Score(item flowcontrol.QueueItemAccessor) float64
}
