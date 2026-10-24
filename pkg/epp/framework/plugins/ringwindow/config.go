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

package ringwindow

import "fmt"

// Config holds the configuration for the ring-window plugin.
type Config struct {
	// CapacityTokens is the KV cache capacity in tokens used while no
	// endpoint has reported cache_config_info through the datalayer.
	CapacityTokens int64 `json:"capacityTokens"`
	// Theta is the membership budget as a fraction of KV capacity:
	// Σ member footprints (idle members included) ≤ θ·C. It is the ring's
	// own operator parameter — plugin configurations are private in llm-d,
	// so the gate's KV threshold cannot be read from here. Default 0.8
	// mirrors the gate's default; the operator keeps the two in sync.
	Theta float64 `json:"theta"`
	// RotatingTurnaround serves the window in back-and-forth order with the
	// turnaround position rotating one place per reversal, which spreads
	// the overflow loss off the p10 population. false = plain
	// back-and-forth (the turnaround members carry the loss), kept so the
	// experiment can measure the trade.
	RotatingTurnaround bool `json:"rotatingTurnaround"`
	// EvictionTTLSeconds is ledger hygiene only: how long a session with no
	// in-flight request and no activity is kept before its bookkeeping is
	// dropped. It never decides who runs. A swept session that returns
	// rejoins the ring at the tail.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
	// EvictionSweepSeconds is how often the hygiene sweep runs.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds"`
}

// PluginType is the plugin type registered with the framework.
const PluginType = "ring-window"

func defaultConfig() Config {
	return Config{
		CapacityTokens:      4194304,
		Theta:               0.8,
		RotatingTurnaround:  true,
		EvictionTTLSeconds:  3600,
		EvictionSweepSeconds: 10,
	}
}

func (c Config) validate() error {
	if c.CapacityTokens <= 0 {
		return fmt.Errorf("capacityTokens must be > 0, got %d", c.CapacityTokens)
	}
	if c.Theta <= 0 || c.Theta > 1 {
		return fmt.Errorf("theta must be in (0, 1], got %v", c.Theta)
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	return nil
}
