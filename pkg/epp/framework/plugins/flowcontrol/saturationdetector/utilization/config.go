/*
Copyright 2025 The Kubernetes Authors.

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

package utilization

import (
	"errors"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

// Default configuration values
const (
	// DefaultQueueDepthThreshold is the default waiting queue size threshold.
	DefaultQueueDepthThreshold int = 5
	// DefaultKVCacheUtilThreshold is the default KV cache utilization (0.0 to 1.0) threshold.
	DefaultKVCacheUtilThreshold float64 = 0.8
	// DefaultMetricsStalenessThreshold defines how old metrics can be before they are considered
	// stale.
	DefaultMetricsStalenessThreshold time.Duration = 200 * time.Millisecond
	// defaultHeadroom is the default burst allowance (0%).
	defaultHeadroom float64 = 0.0
)

// StalenessPolicy selects how endpoints with missing or stale metrics contribute to pool
// saturation in Saturation().
type StalenessPolicy string

const (
	// StalenessSaturated scores endpoints with missing or stale metrics as fully saturated
	// (1.0). A fleet-wide metrics collection failure therefore halts dispatch entirely rather than
	// admitting blind.
	StalenessSaturated StalenessPolicy = "saturated"
	// StalenessIgnore excludes endpoints with missing or stale metrics from the saturation
	// average. When every candidate endpoint is stale, the pool scores 0.0 and dispatch continues,
	// consistent with the Filter's fail-open fallback. Deployments choose this posture when the
	// availability cost of a metrics collection outage outweighs the risk of driving stale,
	// possibly overloaded endpoints.
	StalenessIgnore StalenessPolicy = "ignore"
)

// apiConfig represents the external configuration schema for the utilization detector.
// It dictates how the plugin calculates pool-level saturation (to trigger backpressure) and
// endpoint-level limits (to filter out overloaded candidates during routing).
//
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// QueueDepthThreshold defines the saturation threshold for an endpoint's waiting queue depth.
	//
	// This limit serves as the "ideal" queue capacity for a single endpoint.
	// When combined with the KV Cache threshold, it defines PoolSaturation. When the pool
	// approaches or exceeds this aggregate capacity, the Flow Controller triggers backpressure
	// to buffer new traffic until capacity becomes available.
	//
	// Defaults to 5 if unset.
	QueueDepthThreshold *int `json:"queueDepthThreshold,omitempty"`

	// KVCacheUtilThreshold defines the saturation threshold for an endpoint's KV cache memory
	// utilization, expressed as a fraction in [0.0, 1.0].
	//
	// This limit serves as the "ideal" memory capacity for a single endpoint. The detector uses
	// a roofline model, taking the maximum of either the QueueDepth ratio or the KVCache ratio to
	// determine the endpoint's true saturation score.
	//
	// Defaults to 0.8 (80%) if unset.
	KVCacheUtilThreshold *float64 `json:"kvCacheUtilThreshold,omitempty"`

	// MetricsStalenessThreshold defines how old an endpoint's metrics can be before they are
	// considered stale. How stale endpoints affect pool saturation is controlled by
	// StalenessPolicy.
	//
	// WARNING: In continuous batching architectures, the KV cache grows rapidly during the decode
	// phase. A high staleness threshold combined with a high KV cache utilization limit can lead to
	// routing traffic to an endpoint whose memory is silently expanding within the telemetry blind
	// spot.
	//
	// Defaults to 200ms if unset.
	MetricsStalenessThreshold *metav1.Duration `json:"metricsStalenessThreshold,omitempty"`

	// StalenessPolicy selects how endpoints with missing or stale metrics contribute to pool
	// saturation: "saturated" scores them as fully saturated, and "ignore" excludes them from
	// the saturation average.
	//
	// Defaults to "saturated" if unset.
	StalenessPolicy *StalenessPolicy `json:"stalenessPolicy,omitempty"`

	// Headroom defines the allowed burst capacity above the ideal thresholds, expressed as a
	// multiplier (e.g., 0.2 for 20%).
	//
	// This parameter decouples pool-level backpressure from individual endpoint routing. While
	// PoolSaturation uses the strict ideal capacity to manage overall pool load, the Filter logic
	// evaluates relaxed limits (Threshold * (1 + Headroom)) to determine if a specific endpoint can
	// accept more work.
	//
	// Example: QueueDepthThreshold=5, Headroom=0.2.
	//
	//   - Backpressure: An endpoint with 6 queued requests is considered 120% full, pushing pool
	//     averages up.
	//   - Routing: The endpoint's hard filter limit is 6. The router can still send it requests
	//     (e.g., to satisfy high affinity) until it hits 6.
	//
	// Defaults to 0.0 (no burst allowed) if unset.
	Headroom *float64 `json:"headroom,omitempty"`

	// InFlightLoadProducerName selects which inflight-load-producer's InFlightLoad
	// attribute supplies the in-flight request count used for scrape-lag
	// compensation (see Detector.Saturation). Empty selects the default producer.
	// When no such producer is configured the compensation is skipped and the
	// detector relies solely on the scraped WaitingQueueSize.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`

	// Aggregation selects how the per-endpoint saturation scores are combined into
	// the POOL score the flow-control gate uses:
	//
	//	"mean" — the arithmetic mean over the candidate endpoints. DEFAULT, and the
	//	         behaviour every banked run was produced with.
	//	"min"  — the MINIMUM over the endpoints, i.e. the pool is gated only when
	//	         EVERY engine is saturated.
	//
	// The mean is what couples the gate to pool-level load in a multi-engine stack:
	// one engine at 0.95 beside one at 0.50 reads 0.725, so NOTHING is gated while
	// an engine is saturated (PREREG-multipod-offload audit N2). It also never
	// rebalances: the gate closes on the pool average while one engine is starved,
	// so the under-filled engine keeps getting fewer seats. "min" decouples them at
	// the cost of gating later.
	//
	// Staleness interacts with the aggregation by design: an endpoint whose metrics
	// are stale/absent is EXCLUDED from the min (it can neither open nor close the
	// gate) and its 1.0 keeps counting in the mean; if no endpoint has fresh metrics
	// the pool scores 1.0 either way (fail-closed preserved).
	//
	// Single-engine stacks are byte-identical under both values.
	Aggregation *string `json:"aggregation,omitempty"`
}

// Aggregation selects how per-endpoint saturation scores become the pool score.
type Aggregation string

const (
	// AggregationMean averages the per-endpoint scores (the legacy behaviour).
	AggregationMean Aggregation = "mean"
	// AggregationMin takes the minimum, so the pool gates only when every endpoint
	// is saturated.
	AggregationMin Aggregation = "min"
)

// Config is the internal, fully-validated configuration used by the detector.
type Config struct {
	Aggregation               Aggregation
	QueueDepthThreshold       int
	KVCacheUtilThreshold      float64
	MetricsStalenessThreshold time.Duration
	Headroom                  float64
	StalenessPolicy           StalenessPolicy
	InFlightLoadProducerName  string
}

// buildConfig applies the configuration lifecycle (defaulting and validation) and translates the
// external schema into the internal domain model.
// The provided apiConfig is copied to prevent mutation side-effects.
func buildConfig(apiCfg *apiConfig) (*Config, error) {
	var safeCfg apiConfig
	if apiCfg != nil {
		safeCfg = *apiCfg
	}

	applyDefaults(&safeCfg)

	if err := validateConfig(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid utilization detector configuration: %w", err)
	}

	aggregation := AggregationMean
	if safeCfg.Aggregation != nil {
		aggregation = Aggregation(*safeCfg.Aggregation)
	}
	return &Config{
		Aggregation:               aggregation,
		QueueDepthThreshold:       *safeCfg.QueueDepthThreshold,
		KVCacheUtilThreshold:      *safeCfg.KVCacheUtilThreshold,
		MetricsStalenessThreshold: safeCfg.MetricsStalenessThreshold.Duration,
		Headroom:                  *safeCfg.Headroom,
		StalenessPolicy:           *safeCfg.StalenessPolicy,
		InFlightLoadProducerName:  safeCfg.InFlightLoadProducerName,
	}, nil
}

// applyDefaults populates unset fields in the external configuration with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.QueueDepthThreshold == nil {
		cfg.QueueDepthThreshold = ptr.To(DefaultQueueDepthThreshold)
	}
	if cfg.KVCacheUtilThreshold == nil {
		cfg.KVCacheUtilThreshold = ptr.To(DefaultKVCacheUtilThreshold)
	}
	if cfg.MetricsStalenessThreshold == nil {
		cfg.MetricsStalenessThreshold = &metav1.Duration{Duration: DefaultMetricsStalenessThreshold}
	}
	if cfg.Headroom == nil {
		cfg.Headroom = ptr.To(defaultHeadroom)
	}
	if cfg.StalenessPolicy == nil {
		cfg.StalenessPolicy = ptr.To(StalenessSaturated)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures rather than failing on the first error.
func validateConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.QueueDepthThreshold != nil && *cfg.QueueDepthThreshold <= 0 {
		errs = append(errs, fmt.Errorf("queueDepthThreshold must be strictly positive, got %d", *cfg.QueueDepthThreshold))
	}
	if cfg.KVCacheUtilThreshold != nil && (*cfg.KVCacheUtilThreshold <= 0.0 || *cfg.KVCacheUtilThreshold > 1.0) {
		errs = append(errs, fmt.Errorf("kvCacheUtilThreshold must be in (0.0, 1.0], got %f", *cfg.KVCacheUtilThreshold))
	}
	if cfg.MetricsStalenessThreshold != nil && cfg.MetricsStalenessThreshold.Duration <= 0 {
		errs = append(errs, fmt.Errorf("metricsStalenessThreshold must be strictly positive, got %v",
			cfg.MetricsStalenessThreshold.Duration))
	}
	if cfg.Headroom != nil && *cfg.Headroom < 0.0 {
		errs = append(errs, fmt.Errorf("headroom must be a non-negative value, got %f", *cfg.Headroom))
	}
	if cfg.Aggregation != nil && *cfg.Aggregation != string(AggregationMean) && *cfg.Aggregation != string(AggregationMin) {
		errs = append(errs, fmt.Errorf("aggregation must be %q or %q, got %q",
			AggregationMean, AggregationMin, *cfg.Aggregation))
	}
	if cfg.StalenessPolicy != nil && *cfg.StalenessPolicy != StalenessSaturated &&
		*cfg.StalenessPolicy != StalenessIgnore {
		errs = append(errs, fmt.Errorf("stalenessPolicy must be %q or %q, got %q",
			StalenessSaturated, StalenessIgnore, *cfg.StalenessPolicy))
	}

	return errors.Join(errs...)
}
