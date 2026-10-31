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

package groupsched

// THE SAFETY DETECTOR PLUGIN — the CORE'S OWN SaturationDetector.
//
// It occupies the FRAMEWORK'S detector slot for every core arm (the stock
// utilization-detector is a roofline gate at 0.80 kv / 5 waiting, and FCFS arms keep
// exactly that one, which is the comparison's point).
//
// Two implementations, selected by the `safety` value — the SAME two the scheduler's
// module bundle uses (modules.go):
//
//	"none"            (DEFAULT) the seat budget is the whole control: the pool score
//	                  is 0.0, i.e. this detector NEVER closes the dispatch gate. It
//	                  still exports its inputs and its verdict, so a run states what
//	                  would have fired.
//	"budget-fraction" the memory-only detector, OFF by default until its input is
//	                  verified as IN-USE work.
//
// The memory-only implementation has three properties:
//
//  1. MEMORY ONLY. No queue term: the parked queue depth is REPORTED, never
//     decisive (the e4 defect's second half was a fixed count of 5 closing the gate
//     on a 4-seat engine at 3 % KV).
//  2. THE THRESHOLD IS THE SEAT BUDGET'S OWN FRACTION, read from the same config
//     value the budget is sized with (groupBudgetFraction). There is no second
//     constant anywhere in the path.
//  3. PER ENGINE, MIN OVER ENGINES FOR THE POOL. The pool score is the MINIMUM over
//     the engines with usable metrics, so the flow controller stops dispatching only
//     when EVERY engine is at its budget fraction — the stock detector's default is
//     a MEAN, which hides one saturated engine behind an idle sibling (exactly the
//     measured coupling that made the wave's gate research necessary).
//
// STALE → OPEN. An endpoint whose metrics are missing or older than a small multiple
// of the framework's own refresh interval contributes NO score, and a pool with no
// usable engine reads 0.0 (dispatch continues). The seat budget is the primary
// admission control; a detector that failed closed on a scrape gap would add a stall
// mode of its own — and this removes the stock detector's fail-closed 200 ms time
// constant from the path entirely.
//
// The SAME module (SafetyDetector / BudgetFraction in modules.go) is what the
// scheduler's own per-engine holds use, so the pool-level gate and the per-engine
// gate can never disagree about what "fired" means (the D3 fix).
//
// INSTRUMENTATION (item 4 of the gate instrumentation gap): the plugin exports what
// it READ and what it DECIDED, per engine — the value (kv), the score, the fired
// verdict, whether the sample was stale, and the pool score it handed the flow
// controller. A 15 s scrape of EPP metrics therefore carries the detector's actual
// input and state, not just its consequences.

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// SafetyDetectorPluginType is the detector plugin's registered type.
const SafetyDetectorPluginType = "group-safety-detector"

// staleMultiple is how many framework refresh intervals a sample may be behind
// before it counts as stale. It is a WIRING FACT (the poll cadence the handle
// reports), not a policy knob: 3 × the refresh interval is "one scrape may be lost,
// two in a row means the path is broken".
const staleMultiple = 3

// safetyDetectorConfig is the plugin's parameter block. ONE number: the budget
// fraction — the same value the group scheduler's seat budget is sized with.
type safetyDetectorConfig struct {
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`
	// Safety selects the implementation: "none" (default; never closes) or
	// "budget-fraction" (the memory-only detector).
	Safety string `json:"safety"`
}

// SafetyDetectorPlugin is the core's saturation detector.
type SafetyDetectorPlugin struct {
	typedName fwkplugin.TypedName
	det       SafetyDetector
	refresh   time.Duration

	mu   sync.Mutex
	last map[string]engineSample

	// per-engine instrumentation (nil-safe until registered)
	kvGauge      *prometheus.GaugeVec
	scoreGauge   *prometheus.GaugeVec
	firedGauge   *prometheus.GaugeVec
	staleGauge   *prometheus.GaugeVec
	waitingGauge *prometheus.GaugeVec
	poolScore    prometheus.Gauge
	evaluations  prometheus.Counter
}

// engineSample is the last verdict for one engine: what the detector READ (kv, the
// waiting depth — reporting only — and the measured cache geometry) and what it
// DECIDED. It is the only sanctioned metric view for the rest of the core: the
// framework hands the detector the LIVE datastore endpoints on every Saturation call,
// while a plugin's remembered endpoint handle carries a metrics CLONE taken at
// construction (fwksched.NewEndpoint → metrics.Clone()). Reading the clone is the
// latch class that froze the E-wave's gate: the clone only refreshed when a dispatch
// re-remembered the endpoint, so a closed gate could never see its input change.
type engineSample struct {
	kv      float64
	waiting int
	blockSz int
	numBlk  int
	score   float64
	fired   bool
	stale   bool
	at      time.Time
}

var (
	_ flowcontrol.SaturationDetector = &SafetyDetectorPlugin{}
	_ fwkplugin.Plugin               = &SafetyDetectorPlugin{}
)

// SafetyDetectorFactory builds the plugin. It fails CLOSED on a missing/absurd
// threshold: a detector whose threshold is 0 would fire permanently and stall the
// stack.
func SafetyDetectorFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := safetyDetectorConfig{GroupBudgetFraction: defaultConfig().GroupBudgetFraction}
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", SafetyDetectorPluginType, err)
		}
	}
	if cfg.GroupBudgetFraction <= 0 || cfg.GroupBudgetFraction > 1 {
		return nil, fmt.Errorf("%s: groupBudgetFraction must be in (0,1], got %v (it must equal the group scheduler's seat-budget fraction: one value, two readers)",
			SafetyDetectorPluginType, cfg.GroupBudgetFraction)
	}
	mods, err := buildModules(Config{GroupBudgetFraction: cfg.GroupBudgetFraction, BlockSizeTokens: 64,
		Quantum: 4, IdleSeconds: 20, TTLSeconds: 30, ColdEntryCapTokens: 1, InFlightCap: 1,
		Safety: cfg.Safety})
	if err != nil {
		return nil, err
	}
	p := &SafetyDetectorPlugin{
		typedName: fwkplugin.TypedName{Type: SafetyDetectorPluginType, Name: name},
		det:       mods.safety,
		last:      map[string]engineSample{},
	}
	if handle != nil {
		if r := handle.RefreshMetricsInterval(); r > 0 {
			p.refresh = r
		} else {
			p.refresh = 50 * time.Millisecond
		}
		if reg := handle.Metrics(); reg != nil {
			if err := p.register(reg); err != nil {
				return nil, err
			}
		}
	}
	if p.det.Name() == "none" {
		fmt.Printf("group-scheduler: safety detector: NONE — the seat budget is the whole admission control (this detector never closes the dispatch gate; the framework's stock 0.80 kv / 5-waiting roofline is NOT in this arm). The inputs it would have judged are still exported: llm_d_epp_safety_detector_{kv,score,fired,stale,waiting_queue} per engine + _pool_score. Run-level safety: preemptions <= the FCFS arm's.\n")
	} else {
		fmt.Printf("group-scheduler: safety detector: MEMORY ONLY per engine — fires at measured kv >= %v (the seat budget's own fraction; EQUAL to groupBudgetFraction, no queue term); pool score = MIN over engines with fresh metrics; stale or missing metrics = OPEN (the seat budget is the primary control). Series: llm_d_epp_safety_detector_{kv,score,fired,stale,waiting_queue} per engine + _pool_score.\n",
			p.det.Threshold())
	}
	return p, nil
}

func (p *SafetyDetectorPlugin) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *SafetyDetectorPlugin) register(reg prometheus.Registerer) error {
	p.kvGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "kv",
		Help: "the kv_cache_usage value the safety detector actually READ for this engine (its only decision input)"}, engineLabels)
	p.scoreGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "score",
		Help: "kv / threshold for this engine (1.0 = firing)"}, engineLabels)
	p.firedGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "fired",
		Help: "1 while this engine is at or over the seat budget's fraction"}, engineLabels)
	p.staleGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "stale",
		Help: "1 when this engine's metrics were too old to use (excluded, i.e. OPEN)"}, engineLabels)
	p.waitingGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "waiting_queue",
		Help: "the engine's scraped vLLM waiting queue depth (REPORTING only: the detector does not read it)"}, engineLabels)
	p.poolScore = prometheus.NewGauge(prometheus.GaugeOpts{Subsystem: "llm_d_epp_safety_detector", Name: "pool_score",
		Help: "the pool saturation the detector returned to the flow controller (MIN over fresh engines; 0 = open)"})
	p.evaluations = prometheus.NewCounter(prometheus.CounterOpts{Subsystem: "llm_d_epp_safety_detector", Name: "evaluations_total",
		Help: "Saturation evaluations"})
	for _, c := range []prometheus.Collector{p.kvGauge, p.scoreGauge, p.firedGauge, p.staleGauge,
		p.waitingGauge, p.poolScore, p.evaluations} {
		if err := reg.Register(c); err != nil {
			if _, dup := err.(prometheus.AlreadyRegisteredError); dup {
				fmt.Printf("group-scheduler: WARNING safety detector metrics already registered in this process — this instance will not emit its own series\n")
				continue
			}
			return err
		}
	}
	return nil
}

// staleAfter is the freshness bound, derived from the framework's own refresh
// interval (no policy constant).
func (p *SafetyDetectorPlugin) staleAfter() time.Duration {
	r := p.refresh
	if r <= 0 {
		r = 50 * time.Millisecond
	}
	return staleMultiple * r
}

// Saturation is the pool signal: MIN over the engines with fresh metrics, 0.0 when
// nothing is usable (OPEN).
func (p *SafetyDetectorPlugin) Saturation(_ context.Context, candidates []datalayer.Endpoint) float64 {
	now := time.Now()
	pool := -1.0
	p.mu.Lock()
	for _, e := range candidates {
		if e == nil {
			continue
		}
		engine := endpointKey(e)
		m := e.GetMetrics()
		s := engineSample{at: now, stale: true}
		if m != nil {
			s.kv = m.KVCacheUsagePercent
			s.waiting = m.WaitingQueueSize
			s.blockSz = m.CacheBlockSize
			s.numBlk = m.CacheNumBlocks
			s.stale = now.Sub(m.UpdateTime) > p.staleAfter()
		}
		if !s.stale {
			// the SAME definition the scheduler's per-engine net uses
			s.score = p.det.Score(s.kv, true)
			s.fired = p.det.Fired(s.kv, true)
			if pool < 0 || s.score < pool {
				pool = s.score
			}
		}
		p.last[engine] = s
		if p.kvGauge != nil {
			p.kvGauge.WithLabelValues(engineLabel(engine)).Set(s.kv)
			p.scoreGauge.WithLabelValues(engineLabel(engine)).Set(s.score)
			fired, stale := 0.0, 0.0
			if s.fired {
				fired = 1
			}
			if s.stale {
				stale = 1
			}
			p.firedGauge.WithLabelValues(engineLabel(engine)).Set(fired)
			p.staleGauge.WithLabelValues(engineLabel(engine)).Set(stale)
			p.waitingGauge.WithLabelValues(engineLabel(engine)).Set(float64(s.waiting))
		}
	}
	p.mu.Unlock()
	if pool < 0 {
		pool = 0 // no usable engine: OPEN (see the file header)
	}
	if p.poolScore != nil {
		p.poolScore.Set(pool)
		p.evaluations.Inc()
	}
	return pool
}

// Samples returns the last verdict per engine (the run artifacts / a dump).
func (p *SafetyDetectorPlugin) Samples() map[string]engineSample {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[string]engineSample, len(p.last))
	for k, v := range p.last {
		out[k] = v
	}
	return out
}
