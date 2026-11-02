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

// Package groupsched implements the group scheduler: one shared queue of branches, a seat
// budget per engine, and the rules in SCHEDULER-REDESIGN.md.
//
// Hook wiring: a request blocks in flow control at admission → the fairness Pick (here)
// sees it as a queue head, with its tokens already rendered by branch identity at arrival →
// the chosen queue is released → PreRequest (here) records the in-flight request and
// reconciles the engine the framework actually picked → the engine → ResponseBody (here)
// settles. So the waiting set and every cost come from the queues at Pick time, and every
// cost is the request's own exact prefill chain.
package groupsched

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "group-scheduler"

var (
	_ fwkrc.PreRequest            = &GroupScheduler{}
	_ fwkrc.ResponseBodyProcessor = &GroupScheduler{}
	_ fwkplugin.ConsumerPlugin    = &GroupScheduler{}
	_ fwkplugin.StateDumper       = &GroupScheduler{}
	_ fwkdl.Registrant            = &GroupScheduler{}
	_ fwkdl.EndpointExtractor     = &GroupScheduler{}
	_ fwkfc.FairnessPolicy        = &GroupScheduler{}
	_ fwkfc.SaturationDetector    = &GroupScheduler{}
	_ fwksched.Filter             = &GroupScheduler{}
)

// GroupScheduler is a single named instance shared by every hookup: the flow-control
// FAIRNESS POLICY, the scheduling FILTER, and the SATURATION DETECTOR — one plugin,
// one state, so the three answers cannot disagree.
//
//	policy    Pick: which queued request dispatches next (seated branches only)
//	filter    Filter: which engine may serve it (the branch's seat)
//	detector  Saturation: is the POOL saturated (the flow controller's dispatch gate)
//
// The detector answers 0 (NOT saturated) from an atomic read and never takes the
// scheduler's mutex (the flow controller polls it about once per millisecond). That is not
// a placeholder: the SEAT BUDGET is the admission control, and this policy only ever
// releases requests of branches it has seated, so a pool-level gate has nothing left to
// decide.
type GroupScheduler struct {
	typedName fwkplugin.TypedName
	cfg       Config

	sched   *Scheduler
	metrics *groupMetrics
	handle  fwkplugin.Handle
	// logger is the plugin's logger: startup state at the default level (a run's arm
	// and its wiring must be readable from the artifacts), per-request detail at V(DEBUG)
	// — off unless the operator asks for it.
	logger logr.Logger

	fcfsOnce sync.Once

	// ---- capacity acquisition (the capacity race) ----
	//
	// num_gpu_blocks arrives on the endpoint's METRICS, and an endpoint's
	// creation event fires BEFORE the first scrape has populated them, so
	// the event path alone can miss the capacity for the whole process
	// lifetime (measured twice: capacity was only recovered by a metadata
	// nudge). The plugin therefore keeps the endpoint HANDLES it has seen
	// and re-reads their metrics on the ticker: the datalayer refreshes a
	// ModelServer's metrics IN PLACE (atomic pointer swap inside the same
	// endpoint object), so a stored handle is a live poll target.
	//
	// MULTI-POD: capacity, the liveness deadline and the acquisition audit
	// are PER ENDPOINT. "First capacity wins" was a single-pool assumption;
	// an engine whose metrics never arrive must not look healthy just
	// because its sibling learned its own num_gpu_blocks.
	capMu     sync.Mutex
	endpoints map[string]*endpointState // endpoint id -> handle + acquisition state
	hetero    map[string]struct{}       // engines already reported as capacity-heterogeneous
	// ---- the placement pin (the seat binding mirror) ----
	//
	// candidates is the LIVE endpoint set the framework last offered to Saturation
	// (stored only when it changes). It is the only metrics path in the core: a
	// remembered endpoint handle carries a construction-time CLONE
	// (fwksched.NewEndpoint → metrics.Clone()), so the capacity retry reads THIS set.
	candidates atomic.Pointer[[]fwkdl.Endpoint]

	// pins mirrors the scheduler's branch→engine SEAT binding AND its request id →
	// branch mapping, so the Filter and the scheduler can each answer "which branch is
	// this request?" WITHOUT taking s.mu (see filter.go's lock discipline). The
	// scheduler pushes every seat/unseat/rebind and every release/forget through
	// onBind/onUnbind/onRelease/onForget; nothing here infers anything.
	pins *pinTable

	// chains memoizes each QUEUED request's exact prefill chain, keyed by request id and
	// pruned to the ids the last Pick saw: Pick runs at ~1 kHz over the same requests, and
	// the chain is a hash of the whole prompt. Our own state, never a request attribute
	// (a scoped hook may not write one — see chainsFor).
	chainsMu sync.Mutex
	chains   map[string][]Chain
}

// endpointState is one discovered endpoint's capacity-acquisition state.
type endpointState struct {
	// NOTE: no metrics handle is stored. A remembered handle's metrics are a
	// construction-time CLONE (fwksched.NewEndpoint → metrics.Clone()), so every read
	// of one is a frozen input; the live view is the detector plugin's sample set
	// (liveSource).
	// firstSeen is this endpoint's first observation (its OWN liveness
	// deadline starts here, not the process's).
	firstSeen time.Time
	// warned is the one-shot per-endpoint liveness warning.
	warned bool
	// via records how this engine's capacity was learned (audit).
	via string
}

// Factory builds a GroupScheduler. FAILS CLOSED without the
// token-producer (Consumes) and on an invalid config; the index attaches
// at startup with the config's block size (the token processor's value).
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	sched, err := NewScheduler(cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}

	logCtx := context.Background()
	if handle != nil {
		logCtx = handle.Context()
	}
	r := &GroupScheduler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		logger:    log.FromContext(logCtx).WithName(PluginType + "/" + name),
		cfg:       cfg,
		sched:     sched,
		endpoints: map[string]*endpointState{},
		hetero:    map[string]struct{}{},
		pins:      newPinTable(),
		chains:    map[string][]Chain{},
		// the handle is held for the RegisterDependencies-time gate: plugin
		// construction order is NOT the config order (the runner may build
		// the fairness policy first), so the token-producer check cannot
		// run at Factory time — RegisterDependencies runs after ALL
		// plugins are constructed and added.
		handle: handle,
	}
	r.metrics = newGroupMetrics(r.sched)
	// The effective config is echoed at the DEFAULT level, always: a run's arm must be
	// verifiable from its own log (it is also carried in DumpState/Snapshot.Config). It is
	// rendered as sorted key=value pairs, NOT as JSON: a logger escapes quotes inside a
	// message, which no tool can parse afterwards.
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: effective config " + configEcho(cfg))
	r.logPolicy()

	// The observation hooks are installed UNCONDITIONALLY: the metrics they
	// feed are nil-safe, and a unit-mode run (no handle) must record the same
	// accounting as a live one — the affinity-violation and engine-gone
	// counters are CORRECTNESS signals, not decoration.
	r.sched.onExit = func(engine, reason string, _ int) { r.metrics.ObserveExit(engine, reason) }
	r.sched.onEntry = r.metrics.ObserveEntry
	r.sched.onServe = r.metrics.ObserveServed
	r.sched.onRehome = func(branch, from, to string) {
		r.metrics.ObserveAffinityViolation(engineLabel(from), engineLabel(to))
		r.logger.V(logutil.DEFAULT).Info("group-scheduler: AFFINITY VIOLATION — a request landed off its seat; rebound to the observed endpoint",
			"branch", branch, "charged", engineLabel(from), "observed", engineLabel(to))
	}
	// The pin table follows the SEATS: one map write per seat, one delete per exit.
	r.sched.onBind = func(branch, engine string) { r.pins.set(branch, engine) }
	r.sched.onUnbind = func(branch string) { r.pins.del(branch) }
	// The pin table also follows the REQUESTS: a release names its branch, and the record's
	// end forgets it. This is what makes the Filter able to resolve a scoped request to its
	// branch by request id alone.
	r.sched.onRelease = func(branch, reqID string) { r.pins.setRequest(reqID, branch) }
	r.sched.onForget = func(reqID string) { r.pins.delRequest(reqID) }
	r.sched.onColdValveHeld = func(engine string) {
		r.metrics.ObserveColdValveHeld(engineLabel(engine))
	}
	r.sched.onEngineGone = func(engine string, branches int) {
		r.metrics.ObserveEngineGone()
		r.logger.V(logutil.DEFAULT).Info("group-scheduler: ENGINE GONE — branches exited and the group dropped",
			"engine", engineLabel(engine), "branches", branches)
	}
	if handle != nil {
		// THE DETECTOR-IDENTITY GATE. The loader FABRICATES the stock utilization
		// detector when the config names none (loader/defaults.go ensureSaturationDetector),
		// so a core arm can silently run the stock 0.80 kv / 5-waiting roofline gate
		// instead of this plugin — the arm would then measure someone else's policy.
		// Exactly one detector may exist, and on a core arm it must be THIS plugin
		// (flowControl.saturationDetector.pluginRef: <this plugin's name>). Any other
		// detector instance in the same config is announced as a conflict.
		if others := r.otherDetectors(); len(others) > 0 {
			r.logger.V(logutil.DEFAULT).Info(
				"group-scheduler: WARNING DETECTOR CONFLICT — another plugin answers the flow-control saturation gate; point flowControl.saturationDetector.pluginRef at this one",
				"others", others)
		} else {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: this plugin answers the flow-control saturation gate; it answers NOT saturated (the seat budget is the admission control)")
		}
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		// the per-id SEAT/EXIT event stream, at V(DEBUG): off by default, because one
		// line per seat and per exit is measurement detail, not startup state. The detail
		// carries what a run's forensics need without the event ring — branch id, wire id,
		// engine, reason, waiting-since, turns served — and the engine is also its own
		// field, never spliced into the detail.
		r.sched.onEvent = func(ev Event) {
			if !r.logger.V(logutil.DEBUG).Enabled() {
				return
			}
			r.logger.V(logutil.DEBUG).Info("group-scheduler event",
				"t", ev.T.UTC().Format(time.RFC3339Nano), "kind", ev.Kind, "id", ev.ID,
				"detail", ev.Detail, "engine", ev.Engine, "branch", ev.Branch)
		}
		go r.runTicker(handle.Context())
	}
	return r, nil
}

func (r *GroupScheduler) runTicker(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			r.pollCapacity(now)
			r.pushEngineLoad(now)
			r.sched.Tick(now)
			r.metrics.Refresh()
		}
	}
}

func (r *GroupScheduler) TypedName() fwkplugin.TypedName { return r.typedName }

// Consumes declares the ONE request attribute the policy reads under a scope: agent
// identity, Optional, because a config may run without it.
//
// The BRANCH keys are deliberately NOT declared, and that is a startup constraint, not a
// preference: the DAG's layer validator rejects a dependency whose PRODUCER sits in a later
// layer than its consumer, and branch identity implements PreRequest (request-control layer)
// while this policy is a FairnessPolicy (flow-control layer 0) — declaring its key makes the
// EPP refuse to start with "invalid plugin layer execution order". The layer model is the
// thing to fix upstream (a draft note in README §5; never opened by an agent).
//
// AND THAT IS FINE, because this policy does not read the branch from a request: the only
// hook that reads the arrival attributes is Pick, which receives the flow controller's own
// queue item UNSCOPED (processor.go: fairnessP.Pick — the flow-control package applies no
// request scope). Every other hook is handed a SCOPED request (ScopeInvocation in
// scheduler_profile.go, ScopeRequest in requestcontrol/director.go) where an undeclared read
// RESOLVES AS ABSENT, so a branch id read that way would silently be "" — an unseated,
// unstarved, FCFS-pass-through scheduler that looks healthy in every outcome metric. Those
// hooks resolve the branch from the scheduler's own record of the release instead
// (Scheduler.branchOfRequestLocked), by request id.
//
// The key is optional at the data layer; validateBranchProducer is the fail-closed startup
// check instead (an identity plugin must EXIST in the same config).
func (r *GroupScheduler) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			agentidentity.AgentIdentityKey: "",
		},
	}
}

// Produces declares no request attribute, and that is now the whole truth: the plugin
// publishes NOTHING onto the request. Its own bookkeeping (the per-request chains, the
// released record) lives in its own state, so no scoped hook can lose a write here and no
// scoped read can come back empty.
func (r *GroupScheduler) Produces() map[fwkplugin.DataKey]any { return nil }

// maxChainBlocks bounds a prompt's chain. It is effectively unbounded: a cap here would
// silently stop counting a long prompt's tail, which is exactly the occupancy the budget
// exists to bound.
const maxChainBlocks = 1 << 20

// chainsFor is the request's EXACT prefill chains: one per prompt, from the tokens the
// token producer rendered at arrival. Each prompt is hashed independently because the
// engine runs it as its own sequence, and its last block is partial — so its key is
// private to that exact token sequence, which is why a new turn is never free.
//
// THE MEMO IS OURS, NOT THE REQUEST'S. Pick runs at ~1 kHz over the same queued request, so
// re-hashing a prompt per pass would be absurd; the memo is keyed by request id and pruned
// to the ids this pass saw, so it is bounded by the queue depth. It used to be a request
// attribute — an undeclared WRITE, which at Pick (unscoped) happened to survive and at
// PreRequest (scoped) would have been DROPPED and reported as a violation, failing the
// request (director.go: violations.Write() fails PreRequest).
func (r *GroupScheduler) chainsFor(ctx context.Context, reqID string, request *fwksched.InferenceRequest) []Chain {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	r.chainsMu.Lock()
	cached, ok := r.chains[reqID]
	r.chainsMu.Unlock()
	if ok {
		return cached
	}
	hashed, promptTokens := prefixhash.GetBlockHashesWithPromptTokens(ctx, request, r.cfg.BlockSizeTokens, maxChainBlocks)
	chains := make([]Chain, 0, len(hashed))
	for i, prompt := range hashed {
		keys := make([]Hash, len(prompt))
		for j, h := range prompt {
			keys[j] = Hash(h)
		}
		chains = append(chains, Chain{Keys: keys, Tokens: promptTokens[i]})
	}
	if reqID != "" {
		r.chainsMu.Lock()
		r.chains[reqID] = chains
		r.chainsMu.Unlock()
	}
	return chains
}

// pruneChains drops every memo entry this pass did not see: a request that left the queue
// (dispatched, cancelled, expired) needs no chain any more, and at Pick its chain has
// already been handed to the release record that carries it from here on.
func (r *GroupScheduler) pruneChains(seen map[string]struct{}) {
	r.chainsMu.Lock()
	for id := range r.chains {
		if _, keep := seen[id]; !keep {
			delete(r.chains, id)
		}
	}
	r.chainsMu.Unlock()
}

// readID is the string attribute at key ("" when it is absent).
func readID(request *fwksched.InferenceRequest, key fwkplugin.DataKey) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, key)
	return id
}

// otherDetectors lists the OTHER plugins in this config that implement the
// framework's SaturationDetector (the detector-identity gate's input).
func (r *GroupScheduler) otherDetectors() []string {
	if r.handle == nil {
		return nil
	}
	var out []string
	for _, p := range r.handle.GetAllPlugins() {
		if p == nil || p.TypedName().Type == PluginType {
			continue
		}
		if _, ok := p.(fwkfc.SaturationDetector); ok {
			out = append(out, p.TypedName().String())
		}
	}
	sort.Strings(out)
	return out
}

// branchOf is the request's branch id — the queue key, read from branch identity (or, in a
// config with no branch identity, from agent identity: see below). A request with neither is
// ANONYMOUS: it is never seated and never starved (FCFS pass-through). The branch is the
// identity, and a session header is an attribute, never a key: nothing here falls back to it.
//
// PICK ONLY. This is the one hook whose request is the flow controller's own queue item,
// unscoped (processor.go: fairnessP.Pick(ctx, flowGroup) — the flow-control package applies
// no request scope), so the arrival attributes are readable exactly here. Every other hook
// is handed a scoped copy of the request and resolves the branch from the scheduler's
// record of the release (Scheduler.branchOfRequestLocked) instead.
func (r *GroupScheduler) branchOf(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if id := readID(request, branchidentity.BranchIdentityKey); id != "" {
		return id
	}
	// AGENT-IDENTITY-ONLY CONFIG: the other source the identity gate accepts. The director
	// derives the flow queue key from the same order (branch → agent → default), so with no
	// branch identity the agent id IS the queue key, and the policy's unit has to be the same
	// or every request would be anonymous while validateBranchProducer reported a healthy
	// wiring — the exact failure mode the gate exists to prevent. The key is declared
	// Optional in Consumes(), so this read is inside the plugin's own contract.
	return readID(request, agentidentity.AgentIdentityKey)
}

// isReturner reports whether the request is NOT its branch's first turn — the
// fairness class the entry order sorts on. It is branch identity's arrival record,
// not a history the scheduler keeps: a branch's first turn is exactly a request the
// identity minted as new or forked. A request with no arrival record (identity not
// in the config, or a render failure) is a NEWCOMER: nothing is known about its
// history, and the safe direction is to give it the seat ahead of incumbents.
//
// PICK ONLY, for the same reason as branchOf — and note the failure mode if that ever
// stops holding: under a scope this read resolves as absent, so every request would
// classify as a NEWCOMER. That is the safe direction by construction (a newcomer is never
// demoted), which is why the class is phrased as the exception.
func isReturner(request *fwksched.InferenceRequest) bool {
	if request == nil {
		return false
	}
	rec, ok := fwksched.ReadRequestAttribute[branchidentity.BranchArrival](request, branchidentity.BranchArrivalKey)
	if !ok {
		return false
	}
	return rec.Turn > 1
}

// validateTokenProducer is the gate body (testable without a registrar).
func (r *GroupScheduler) validateTokenProducer() error {
	if r.handle == nil {
		return nil // unit/test mode
	}
	for _, p := range r.handle.GetAllPlugins() {
		if p != nil && p.TypedName().Type == tokenizer.PluginType {
			return nil
		}
	}
	return fmt.Errorf(
		"the '%s' plugin requires a token-producer (tier-A block keys; tier B failed the CP2 trust measurements and there is no fallback) — add the token-producer with the vllm render backend",
		PluginType)
}

// validateBranchProducer is the fail-closed branch-source gate: the identity plugin that
// produces the branch attribute must be in the same config. Without it every request is
// anonymous and Pick degrades to FCFS pass-through — a failure that looks healthy in every
// outcome metric, so it fails at startup instead.
func (r *GroupScheduler) validateBranchProducer() error {
	if r.handle == nil {
		return nil // unit/test mode
	}
	b, s := false, false
	for _, p := range r.handle.GetAllPlugins() {
		if p == nil {
			continue
		}
		switch p.TypedName().Type {
		case branchidentity.PluginType:
			b = true
		case agentidentity.PluginType:
			s = true
		}
	}
	if !b && !s {
		return fmt.Errorf("the identity source needs at least one identity plugin ('%s' or '%s'); with neither, every request is anonymous and the group policy degrades to R9 FCFS",
			branchidentity.PluginType, agentidentity.PluginType)
	}
	if b && !s {
		r.logger.V(logutil.DEFAULT).Info("group-scheduler: no session-identity plugin in the config — the derived branch is the branch key on every request")
	}
	return nil
}

// RegisterDependencies runs after every plugin is constructed — the
// STARTUP fail-closed gate lives here: a missing token-producer is a
// full-outage misconfiguration (every request would error at PreRequest,
// the backstop), burning an arm before anyone noticed. The producer's
// BACKEND is not inspectable from outside the plugin — an estimate-backend
// producer still passes this gate (it produces tokens; tier-A quality is
// the config author's responsibility, guarded by the config comment).
func (r *GroupScheduler) RegisterDependencies(reg fwkdl.Registrar) error {
	if err := r.validateTokenProducer(); err != nil {
		return err
	}
	if err := r.validateBranchProducer(); err != nil {
		return err
	}
	return reg.Register(fwkdl.PendingRegistration{
		Owner:      r.typedName,
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  r,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract learns num_gpu_blocks (and checks the block size matches the
// config — a mismatch fails loudly) from the endpoint lifecycle EVENTS, and
// remembers the endpoint so the ticker can retry its capacity acquisition (the
// event carries the metrics before the first scrape has populated them — the
// capacity race).
//
// The engine SET is DISCOVERED here, never configured: an add-or-update
// creates (or refreshes) that endpoint's engine group, and a delete tears it
// down. Node rotation exercises the scale-down path regularly, and it must not leave
// charges on a dead engine.
func (r *GroupScheduler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil {
		return nil
	}
	switch event.Type {
	case fwkdl.EventDelete:
		r.forgetEndpoint(event.Endpoint)
		return nil
	case fwkdl.EventAddOrUpdate:
		r.rememberEndpoint(event.Endpoint)
		return r.learnCapacity(endpointKey(event.Endpoint), event.Endpoint.GetMetrics(), "endpoint-event")
	}
	return nil
}

// learnCapacity takes ONE engine's measured capacity from a metrics snapshot,
// once per engine. Shared by every acquisition path (endpoint event, metrics
// poll, PreRequest) so the block-size match check and the "first capacity
// wins" rule are identical everywhere — and per engine, never pooled.
func (r *GroupScheduler) learnCapacity(engine string, m *fwkdl.Metrics, via string) error {
	if m == nil || m.CacheBlockSize <= 0 || m.CacheNumBlocks <= 0 {
		return nil
	}
	if r.sched.HasCapacityFor(engine) {
		return nil
	}
	if err := r.sched.SetNumBlocksFor(engine, m.CacheBlockSize, m.CacheNumBlocks); err != nil {
		// EVERY mismatched engine is named, not just the one that failed first: a single
		// engine's misconfigured block size must not hide behind another's.
		for bad, berr := range r.sched.BlockSizeMismatches() {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: BLOCK SIZE MISMATCH — this engine's measured block size disagrees with the configured one, so its budget would be in the wrong unit; fix --block-size or blockSizeTokens",
				"engine", bad, "err", berr.Error())
		}
		return err
	}
	if !r.sched.HasCapacityFor(engine) {
		return nil
	}
	r.capMu.Lock()
	wait := time.Duration(0)
	if st, ok := r.endpoints[engine]; ok && !st.firstSeen.IsZero() {
		wait = time.Since(st.firstSeen)
		st.via = via
	}
	r.capMu.Unlock()
	label := r.sched.GroupLabel(engine)
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: capacity learned",
		"via", via, "engine", label, "afterSeconds", wait.Seconds(),
		"cacheBlockSize", m.CacheBlockSize, "numGpuBlocks", m.CacheNumBlocks,
		"budgetBlocks", int(r.cfg.GroupBudgetFraction*float64(m.CacheNumBlocks)))
	r.metrics.ObserveCapacityLearned(via, label)
	r.metrics.SetCapacityMissing(label, false)
	r.reportHeterogeneousCapacity()
	return nil
}

// reportHeterogeneousCapacity fails LOUDLY when a newly measured engine's
// num_gpu_blocks deviates from the first engine's by more than 1%: per-engine
// budgets are only sound while the engines are interchangeable, and a
// heterogeneous pool must never silently inherit another engine's budget
// (log + gauge; the policy keeps running on each engine's own measurement).
func (r *GroupScheduler) reportHeterogeneousCapacity() {
	hetero := r.sched.HeterogeneousEngines(0.01)
	if len(hetero) == 0 {
		return
	}
	r.capMu.Lock()
	var fresh []string
	for id, blocks := range hetero {
		if _, seen := r.hetero[id]; seen {
			continue
		}
		r.hetero[id] = struct{}{}
		fresh = append(fresh, fmt.Sprintf("%s=%d", r.sched.GroupLabel(id), blocks))
	}
	r.capMu.Unlock()
	if len(fresh) == 0 {
		return
	}
	sort.Strings(fresh)
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: CAPACITY HETEROGENEOUS — an engine's num_gpu_blocks deviates >1% from the first measured one; each budget still uses its own engine's value, but the pool is not homogeneous",
		"engines", strings.Join(fresh, ", "))
}

// metricsSource is the narrow poll target: anything whose metrics can be
// re-read. Both the datalayer endpoint (whose metrics are swapped in place by
// the collector) and the scheduler's endpoint snapshot satisfy it; polling a
// snapshot is harmless (it is simply never fresher).
type metricsSource interface {
	GetMetrics() *fwkdl.Metrics
	GetMetadata() *fwkdl.EndpointMetadata
}

func (r *GroupScheduler) rememberEndpoint(ep metricsSource) {
	if ep == nil {
		return
	}
	key := endpointKey(ep)
	r.capMu.Lock()
	defer r.capMu.Unlock()
	if r.endpoints == nil {
		r.endpoints = map[string]*endpointState{}
	}
	st, ok := r.endpoints[key]
	if !ok {
		st = &endpointState{firstSeen: time.Now()}
		r.endpoints[key] = st
	}
	_ = ep // the handle is NOT stored: see endpointState's note (live read only)
}

func (r *GroupScheduler) forgetEndpoint(ep metricsSource) {
	if ep == nil {
		return
	}
	key := endpointKey(ep)
	r.capMu.Lock()
	delete(r.endpoints, key)
	delete(r.hetero, key)
	r.capMu.Unlock()
	// tear the engine's group down: its branches are exited, their homing
	// released, its budget/index dropped (leaving charges on a dead engine
	// poisons every later budget decision on a scale-down).
	r.sched.RemoveEngine(key, time.Now())
	r.metrics.SetCapacityMissing(r.sched.GroupLabel(key), false)
}

func endpointKey(ep metricsSource) string {
	if ep == nil {
		return ""
	}
	if md := ep.GetMetadata(); md != nil && md.ID.Name != "" {
		return md.ID.String()
	}
	return fmt.Sprint(ep)
}

// pushEngineLoad learns CAPACITY from the LIVE per-engine sample set — never from a
// remembered endpoint (a remembered handle's metrics are a construction-time CLONE,
// which is why nothing in this core reads one).
func (r *GroupScheduler) pushEngineLoad(now time.Time) {
	samples := r.sampleCandidates(now)
	if samples == nil {
		return // the framework has not offered a candidate set yet
	}
	for engine, sm := range samples {
		// The event-time copy may carry an empty geometry forever (the capacity race),
		// and the live value is the same metric, so the retry rides this tick.
		if sm.blockSz > 0 && sm.numBlk > 0 && !r.sched.HasCapacityFor(engine) {
			if err := r.learnCapacity(engine, &fwkdl.Metrics{CacheBlockSize: sm.blockSz,
				CacheNumBlocks: sm.numBlk}, "live-detector"); err != nil {
				r.logger.V(logutil.DEFAULT).Info("group-scheduler: CAPACITY REJECTED from the live detector",
					"engine", r.sched.GroupLabel(engine), "err", err.Error())
			}
		}
	}
	_ = now
}

// pollCapacity is the STARTUP LIVENESS check: once a decode endpoint has been observed,
// its capacity must arrive within capacityLivenessSeconds, else the FCFS pass-through is
// being served silently — log loudly and raise the missing gauge.
//
// It reads NO metrics: the capacity RETRY rides the live detector samples
// (pushEngineLoad, same tick). A remembered endpoint handle carries a
// construction-time metrics CLONE, so a poll over clones could never see the geometry
// arrive late — the read that used to live here was itself the capacity-race bug's shape.
func (r *GroupScheduler) pollCapacity(now time.Time) {
	r.capMu.Lock()
	type pollTarget struct {
		key       string
		firstSeen time.Time
		warned    bool
	}
	targets := make([]pollTarget, 0, len(r.endpoints))
	for key, st := range r.endpoints {
		if st == nil {
			continue
		}
		targets = append(targets, pollTarget{key: key, firstSeen: st.firstSeen, warned: st.warned})
	}
	r.capMu.Unlock()

	// sort for a deterministic log order (the map is not)
	sort.Slice(targets, func(i, j int) bool { return targets[i].key < targets[j].key })

	for _, t := range targets {
		if r.sched.HasCapacityFor(t.key) {
			continue
		}
		// PER-ENGINE liveness: this endpoint's own clock, one warning each.
		if t.firstSeen.IsZero() || t.warned || r.cfg.CapacityLivenessSeconds <= 0 {
			continue
		}
		if now.Sub(t.firstSeen) < time.Duration(r.cfg.CapacityLivenessSeconds*float64(time.Second)) {
			continue
		}
		r.capMu.Lock()
		if st, ok := r.endpoints[t.key]; ok {
			st.warned = true
		}
		r.capMu.Unlock()
		r.metrics.SetCapacityMissing(r.sched.GroupLabel(t.key), true)
		r.logger.V(logutil.DEFAULT).Info("group-scheduler: CAPACITY MISSING — this engine's num_gpu_blocks is still unknown, so it seats nothing while the policy looks healthy; check that its metrics carry the cache geometry",
			"engine", engineLabel(t.key), "afterSeconds", now.Sub(t.firstSeen).Seconds(), "enginesTracked", len(targets))
	}
}

// Saturation answers the flow controller's pool gate: 0.0 (NOT saturated), always. The
// seat budget is the admission control — this policy only ever releases requests of
// branches it has seated, so a pool-level gate has nothing left to decide.
//
// It still stores the LIVE candidate set on the way (a no-op unless the set changed):
// that set is the ticker's capacity-retry input, and this hook is the only place the
// framework hands the plugin live endpoint objects. HOT PATH (≈1 kHz): one interface
// compare and one atomic store when nothing changed, no mutex, no allocation.
func (r *GroupScheduler) Saturation(_ context.Context, candidates []fwkdl.Endpoint) float64 {
	r.storeCandidates(candidates)
	return 0
}

// storeCandidates keeps the LIVE endpoint set (the datastore mutates these objects in
// place, so a stored slice is a live view, unlike a plugin's remembered endpoint clone).
// It is a no-op when the set is unchanged — one compare per call, no allocation.
func (r *GroupScheduler) storeCandidates(cands []fwkdl.Endpoint) bool {
	if prev := r.candidates.Load(); prev != nil && sameEndpoints(*prev, cands) {
		return false
	}
	cp := append([]fwkdl.Endpoint(nil), cands...)
	r.candidates.Store(&cp)
	return true
}

// sameEndpoints compares two candidate sets by identity (the interface values hold
// pointers to the datastore's endpoint objects).
func sameEndpoints(a, b []fwkdl.Endpoint) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// sampleCandidates reads every candidate's LIVE metrics (the ticker calls this at 1 Hz
// for the capacity path and the series).
func (r *GroupScheduler) sampleCandidates(now time.Time) map[string]engineSample {
	if p := r.candidates.Load(); p != nil {
		return r.readSamples(*p, now)
	}
	return nil
}

// readSamples reads the live endpoint metrics into an immutable per-engine map.
func (r *GroupScheduler) readSamples(candidates []fwkdl.Endpoint, now time.Time) map[string]engineSample {
	out := make(map[string]engineSample, len(candidates))
	for _, e := range candidates {
		if e == nil {
			continue
		}
		engine := endpointKey(e)
		s := engineSample{at: now}
		if m := e.GetMetrics(); m != nil {
			s.blockSz = m.CacheBlockSize
			s.numBlk = m.CacheNumBlocks
		}
		out[engine] = s
	}
	return out
}

// Pick is the dispatch decision. The waiting set is the queue-head
// snapshot; pre-capacity it degrades to FCFS pass-through.
//
// THIS IS THE ONLY HOOK THAT READS THE REQUEST'S ATTRIBUTES. The flow controller hands it
// its own queue items (processor.go: fairnessP.Pick(ctx, flowGroup)), and the flow-control
// package applies no request scope — so the branch id and the arrival record branch identity
// wrote at RequestHeader are readable here, and nowhere else. Every later hook is handed a
// SCOPED request (Filter/Scorer via scheduler_profile.go, PreRequest via director.go,
// producers via plugin_executor.go) and resolves its branch from the record this Pick
// releases, by request id.
func (r *GroupScheduler) Pick(ctx context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()
	var queues []fwkfc.FlowQueueAccessor
	var heads []Head
	seen := map[string]struct{}{}
	band.IterateQueues(func(q fwkfc.FlowQueueAccessor) bool {
		if q == nil || q.Len() == 0 {
			return true
		}
		h := q.Peek()
		if h == nil {
			return true
		}
		branch := ""
		reqID, fid := "", ""
		returner := false
		var chains []Chain
		if req := h.OriginalRequest(); req != nil {
			if ir := req.InferenceRequest(); ir != nil {
				branch = r.branchOf(ir)
				reqID = ir.RequestID
				fid = ir.FairnessID // the wire-level identity (the queue key)
				returner = isReturner(ir)
				chains = r.chainsFor(ctx, reqID, ir)
				if reqID != "" {
					seen[reqID] = struct{}{}
				}
			}
		}
		queues = append(queues, q)
		heads = append(heads, Head{Branch: branch, Returner: returner,
			FairnessID: fid, ReqID: reqID, Arrival: h.EnqueueTime(), Chains: chains,
		})
		return true
	})
	// the memo is bounded by the queue: an id this pass did not see has left it
	r.pruneChains(seen)
	if len(heads) == 0 {
		return nil, nil //nolint:nilnil
	}
	// before the measured capacity arrives: FCFS pass-through (once
	// per process log; the gauge counts it)
	if !r.sched.HasCapacity() {
		r.fcfsOnce.Do(func() {
			r.logger.V(logutil.DEFAULT).Info("group-scheduler: num_gpu_blocks not yet measured; serving FCFS until the endpoint metrics arrive")
		})
		r.sched.FCFSServed()
		if i := FCFSPick(heads); i >= 0 {
			return queues[i], nil
		}
		return nil, nil //nolint:nilnil
	}
	if i := r.sched.PickFromHeads(heads, now); i >= 0 {
		return queues[i], nil
	}
	return nil, nil //nolint:nilnil
}

// NewState satisfies the FairnessPolicy contract.
func (r *GroupScheduler) NewState(_ context.Context) any { return nil }

// Produce implements DataProducer ONLY to receive RequestAbandoned (see the upstream draft
// note: the framework has the right event, but exposes it to data producers only). It
// produces nothing — the scheduler reads the request, it publishes no data for anyone.
//
// RequestAbandoned is what covers a request the flow controller released but that never
// reached PreRequest (the pinned engine vanished, a screener or an admitter refused it,
// scheduling failed): without it such a request would leave its branch's released window
// open forever.
func (r *GroupScheduler) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// RequestAbandoned ends the record of a request that left the request path without
// reaching PreRequest. It is safe when no record exists. The branch is the record's — this
// hook's request is the framework's (unscoped today), but the scheduler's own record is the
// one source of truth everywhere, so the branch is never read off the request at all.
func (r *GroupScheduler) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	r.sched.Abandon("", request.RequestID)
}

// PreRequest runs after the request leaves the queue (dispatch). It records the in-flight
// request and reconciles the engine the framework ACTUALLY picked with the one the seat
// assumed; nothing is charged here, because the seat charged the exact chains.
//
// THE BRANCH IS NOT READ FROM THE REQUEST. This hook's request is SCOPED
// (datalayer.ScopeRequest, director.go), so an attribute read the plugin did not declare
// resolves as absent and an undeclared write is DROPPED and turns into a failed PreRequest
// (violations.Write()). The branch — and the chain the release charged — come from the
// record the Pick wrote, resolved by request id.
func (r *GroupScheduler) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	// capacity first (belt-and-braces, third acquisition path): it must not
	// depend on the request being trackable — anonymous traffic used to
	// return before ever teaching the budget.
	engine := ""
	if endpoint := pickedEndpoint(schedulingResult); endpoint != nil {
		r.rememberEndpoint(endpoint)
		engine = endpointKey(endpoint)
		if err := r.learnCapacity(engine, endpoint.GetMetrics(), "pre-request"); err != nil {
			return err // the loud mismatch failure
		}
	}
	if request == nil {
		return nil
	}
	// The engine here is AUTHORITATIVE: the seat was placed on the branch's engine at
	// Pick, and the framework's actual choice moves it if it disagrees. The chain is the
	// one the release recorded, so a branch that holds no seat at all is seated by the
	// observation with exactly the cost its head was charged.
	r.sched.DispatchedOnUsage("", request.RequestID, engine, nil, time.Now())
	return nil
}

// ResponseBody: the first chunk is prefill complete (the cold-entry charge is released
// there); EndOfStream is the finalizer for every termination cause (the in-flight decrement
// always, the payback ledger only for a natural one). Both are branch-level facts, and the
// branch is the record's, by request id (this hook is unscoped today; the record is used
// regardless so one source of truth covers every hook).
func (r *GroupScheduler) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, target *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	if response.StartOfStream {
		r.sched.ChunkForRequest(request.RequestID, time.Now())
	}
	if !response.EndOfStream {
		return
	}
	promptTokens := response.Usage.PromptTokens
	cachedTokens := 0
	if response.Usage.PromptTokenDetails != nil {
		cachedTokens = response.Usage.PromptTokenDetails.CachedTokens
	}
	completionTokens := response.Usage.CompletionTokens
	// The settle's engine comes from the response's own endpoint metadata: a settle with no
	// endpoint would fold this engine's keys into another's record. An empty metadata id
	// keeps the binding.
	engine := ""
	if target != nil && target.ID.String() != "" {
		engine = target.ID.String()
	}
	r.sched.SettleOnUsage("", request.RequestID, engine, promptTokens, cachedTokens, completionTokens,
		response.TerminationCause == fwkrc.TerminationCauseNatural, time.Now())
}

// DumpState reports the group snapshot.
func (r *GroupScheduler) DumpState() (json.RawMessage, error) {
	return json.Marshal(r.sched.Snapshot())
}

func pickedEndpoint(schedulingResult *fwksched.SchedulingResult) fwksched.Endpoint {
	if schedulingResult == nil {
		return nil
	}
	result, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 {
		return nil
	}
	return result.TargetEndpoints[0]
}

// configEcho renders the config as sorted key=value pairs (the startup echo's form).
func configEcho(cfg Config) string {
	raw, err := json.Marshal(cfg)
	if err != nil {
		return ""
	}
	fields := map[string]any{}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return string(raw)
	}
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, fields[k]))
	}
	return strings.Join(parts, " ")
}

// logPolicy states what the wired policy DOES, in words, at the default level: a run's
// artifact then says what ran without decoding knob names. One line per module.
func (r *GroupScheduler) logPolicy() {
	cfg := r.cfg
	r.logger.V(logutil.DEFAULT).Info("group-scheduler: policy",
		"identity", "derived branch (the branch is the queue key)",
		"sizing", "seat budget = groupBudgetFraction x num_gpu_blocks, footprint from the engine's own block index",
		"admission", "one shared queue, newcomer-first then oldest; most free budget where the head fits, tie -> lowest engine id",
		"rotation", fmt.Sprintf("fixed quantum Q=%d completed busy periods, or silence > %vs", cfg.quantum(), cfg.IdleSeconds),
		"safety", "none (the seat budget is the admission control)",
		"dispatch", "none (an admitted turn runs)")
}
