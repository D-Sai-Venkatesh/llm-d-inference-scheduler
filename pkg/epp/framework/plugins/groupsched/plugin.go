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

// Package groupsched implements the group scheduler (ALGORITHM-group.md).
// See scheduler.go's header for the algorithm summary.
//
// Hook wiring (the director's order, S0-1's reality): a request blocks in
// flow control at admission → the fairness Pick (HERE) sees it as a
// queue head → the chosen queue is released → the data producers run
// (the tokenizer sets TokenizedRequest) → PreRequest (HERE) corrects the
// estimate with the exact tier-A keys → the engine → ResponseBody
// (HERE) settles. The waiting set therefore comes from the queues at
// Pick time; queued-request admission cost = persisted history blocks +
// a conservative wire-size delta; the exact keys land at PreRequest and
// D13's growth enforcement absorbs the estimate error.
package groupsched

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
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
//	policy    Pick: which queued request dispatches next (seated members only)
//	filter    Filter: which engine may serve it (the member's seat)
//	detector  Saturation: is the POOL saturated (the flow controller's dispatch gate)
//
// The detector answers from an ATOMIC snapshot and never takes the scheduler's mutex
// (the flow controller polls it about once per millisecond). Its DEFAULT answer is
// "not saturated": the SEAT BUDGET is the admission control, and this policy only ever
// releases requests of members it has seated, so a pool-level gate has nothing left to
// decide. The memory-only backstop (live datastore metrics, threshold = the budget
// fraction, stale → open) is the alternative safety module and is OFF by default.
type GroupScheduler struct {
	typedName fwkplugin.TypedName
	cfg       Config

	sched   *Scheduler
	metrics *groupMetrics
	handle  fwkplugin.Handle

	fcfsOnce sync.Once

	// ---- capacity acquisition (WAVE-1 item 1: the capacity race) ----
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
	// detectorScore is the pool saturation last handed to the flow controller, as
	// math.Float64bits (read at ~1 kHz by the detector, written once per Saturation).
	detectorScore atomic.Uint64
	// samples is the LIVE per-engine metric view, published copy-on-write by the ticker
	// and used for the capacity path and the series. It is the only metrics path in the
	// core: a remembered endpoint handle carries a construction-time CLONE
	// (fwksched.NewEndpoint → metrics.Clone()), so a gate fed from one freezes the
	// moment dispatches stop — the E-wave latch.
	samples atomic.Pointer[map[string]engineSample]
	// candidates is the LIVE endpoint set the framework last offered to Saturation
	// (stored only when it changes).
	candidates atomic.Pointer[[]fwkdl.Endpoint]
	// scoreAt is when the pool score was last recomputed (UnixNano).
	scoreAt atomic.Int64

	// pins mirrors the scheduler's member→engine SEAT binding so the Filter can
	// read it WITHOUT taking s.mu (see filter.go's lock discipline). The
	// scheduler pushes every seat/unseat/rebind through onBind/onUnbind; nothing
	// here infers anything.
	pins *pinTable
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

	r := &GroupScheduler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:       cfg,
		sched:     sched,
		endpoints: map[string]*endpointState{},
		hetero:    map[string]struct{}{},
		pins:      newPinTable(),
		// the handle is held for the RegisterDependencies-time gate: plugin
		// construction order is NOT the config order (the runner may build
		// the fairness policy first), so the token-producer check cannot
		// run at Factory time — RegisterDependencies runs after ALL
		// plugins are constructed and added.
		handle: handle,
	}
	r.metrics = newGroupMetrics(r.sched)
	// BUILD-PLAN §1(ii): echo EVERY knob at startup — each arm's config
	// is verifiable afterwards from the run artifacts (also carried in
	// DumpState/Snapshot.Config).
	if echo, err := json.Marshal(cfg); err == nil {
		fmt.Printf("group-scheduler: effective config %s\n", echo)
	}
	// The policy has no MODES left, so the echo states the six things it does in
	// words — a run's artifacts then say what ran without decoding knob names.
	fmt.Printf("group-scheduler: CORE policy — identity: derived branch (member = queue key; nothing is written into the session slot)\n")
	fmt.Printf("group-scheduler: CORE policy — sizing: per-engine seat budget = %v x that engine's measured num_gpu_blocks; footprint from the engine's own block index (dedup + cold-token estimate)\n", cfg.GroupBudgetFraction)
	fmt.Printf("group-scheduler: CORE policy — admission: ONE shared waiter queue, newcomer-first then oldest; the head goes to the engine with the MOST FREE BUDGET where it fits (tie -> lowest engine id); the binding lasts only while seated\n")
	fmt.Printf("group-scheduler: CORE policy — rotation: fixed quantum Q=%d completed busy periods (plus silence > %vs); memory pressure: the budget valve\n", cfg.quantum(), cfg.IdleSeconds)
	if cfg.Safety == "none" {
		fmt.Printf("group-scheduler: CORE policy — safety: NONE (the seat budget IS the admission control; run-level safety is preemptions <= the FCFS arm's)\n")
	} else {
		fmt.Printf("group-scheduler: CORE policy — safety: MEMORY-ONLY net per engine, fires at measured kv >= %v (the budget's own fraction; no queue term), holds that engine's own dispatches and new seats, never influences placement\n", cfg.GroupBudgetFraction)
	}
	// The observation hooks are installed UNCONDITIONALLY: the metrics they
	// feed are nil-safe, and a unit-mode run (no handle) must record the same
	// accounting as a live one — the affinity-violation and engine-gone
	// counters are CORRECTNESS signals, not decoration.
	r.sched.onExit = r.metrics.ObserveExit
	r.sched.onEntry = r.metrics.ObserveEntry
	r.sched.onServe = r.metrics.ObserveServed
	r.sched.onRehome = func(member, from, to string) {
		r.metrics.ObserveAffinityViolation(engineLabel(from), engineLabel(to))
		fmt.Printf("group-scheduler: AFFINITY VIOLATION member=%s charged=%s observed=%s — rebound (the observed endpoint wins)\n",
			member, engineLabel(from), engineLabel(to))
	}
	// The pin table follows the SEATS: one map write per seat, one delete per exit.
	r.sched.onBind = func(member, engine string) { r.pins.set(member, engine) }
	r.sched.onUnbind = func(member string) { r.pins.del(member) }
	r.sched.onColdValveHeld = func(engine string) {
		r.metrics.ObserveColdValveHeld(engineLabel(engine))
	}
	r.sched.onEngineGone = func(engine string, members int) {
		r.metrics.ObserveEngineGone()
		fmt.Printf("group-scheduler: ENGINE GONE %s — exited %d member(s) and dropped the group; their homing is released for re-placement\n",
			engineLabel(engine), members)
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
			fmt.Printf("group-scheduler: WARNING DETECTOR CONFLICT — %v is also configured as a saturation detector. Exactly one detector may answer the flow-control gate, and on a CORE arm it must be this plugin: set flowControl.saturationDetector.pluginRef to this plugin's name. If the stock detector is the one answering, this arm is running the 0.80 kv / 5-waiting roofline, not the core.\n", others)
		} else {
			fmt.Printf("group-scheduler: detector: THIS plugin answers the flow-control saturation gate (default answer: NOT saturated — the seat budget is the admission control; safety=%s).\n", r.cfg.Safety)
		}
		if reg := handle.Metrics(); reg != nil {
			if err := r.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		r.sched.onMemberSource = func(source string) {
			fmt.Printf("group-scheduler: member source in effect at dispatch: %s\n", source)
		}
		// the per-id SEAT/EXIT event log (one debug line each; the
		// campaign's spans could not see seat membership).
		//
		// The engine label is appended to the DETAIL, never inserted before
		// it: the pre-multi-pod reducer regex
		// (`t=<ts> kind=<KIND> id=<id> <detail>`) and the detail keys
		// (kind=/wait=/reason=) keep matching unchanged, so a wave-1 parser
		// reads a multi-pod log without modification.
		r.sched.onEvent = func(ev Event) {
			fmt.Printf("group-scheduler event t=%s kind=%s id=%s %s engine=%s member=%s\n",
				ev.T.UTC().Format(time.RFC3339Nano), ev.Kind, ev.ID, ev.Detail,
				ev.Engine, ev.Member)
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
			r.metrics.Refresh(r.detectorScore.Load())
		}
	}
}

// memberKeyRule states the resolution rule of a memberKey value in words (the
// startup echo).
func memberKeyRule(memberKey string) string {
	switch memberKey {
	case "branch":
		return "base mode: the derived branch is member AND queue key"
	case "session":
		return "the agent-identity session attribute is the member"
	default:
		return "branch when present, else the session, else anonymous (R9)"
	}
}

func (r *GroupScheduler) TypedName() fwkplugin.TypedName { return r.typedName }

// Consumes declares the attribute keys the policy READS. Only the agent-identity
// key is declared here; the BRANCH keys are read the same way but are NOT
// declared, and that is deliberate (WAVE-1 build fix, 2026-10-30):
//
//	the data layer's DAG validator rejects a dependency whose PRODUCER sits in a
//	later layer than its CONSUMER, with the layer of a plugin derived from the
//	interfaces it implements (FlowControlLayer=0 < RequestControlLayer=1). The
//	group scheduler is a FairnessPolicy (layer 0); branch-identity implements
//	PreRequest (it clears the busy flag at dispatch) and is therefore layer 1, so
//	declaring branchidentity.BranchIdentityKey here made the EPP refuse to start:
//	"invalid plugin layer execution order: producer branch-identity/branch-identity
//	needs to be executed before consumer group/group-scheduler" (measured live,
//	wave-1 build 5e851b18, 2026-10-30T03:54Z, the three branch arms only; the
//	group-scheduler unit tests construct the policy directly and never build the
//	plugin DAG, which is why 75 green tests missed it).
//
//	Nothing is lost by omitting them: the attributes are produced by
//	branch-identity's RequestHeader hook, which the request path always runs
//	before admission and therefore before this policy; stage 1 ran exactly this
//	way (its Consumes listed the agent-identity key only) and the values were
//	present on every request. agentidentity.AgentIdentityKey stays because its
//	plugin implements only RequestHeader (layer -1) and the declaration is
//	harmless. The real fix belongs upstream in the layer model (flow control listed
//	before request control while the code comments assume the opposite order) —
//	filed as a draft note for the human, never opened by an agent.
//
// Both identity keys are OPTIONAL at the data layer because which one is the member
// depends on Config.MemberKey (and on which identity plugin the arm deployed); the
// fail-closed check is validateMemberProducer at startup, which requires the
// producer of the configured member source to EXIST. Making the keys required
// unconditionally would have made base mode (no session header, branch
// identity only) a startup failure.
//
// The tokenizer's output is enforced at RUNTIME instead: the data layer
// forbids a FairnessPolicy (flow control layer) from declaring a
// DataProducer's key in Consumes (the layer order is request-control before
// flow-control), so the fail-closed gate is PreRequest's error on a missing
// TokenizedRequest — without the token-producer EVERY request fails loudly at
// dispatch; nothing ever falls back to tier B (CP2's M3 verdict).
func (r *GroupScheduler) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			agentidentity.AgentIdentityKey: "",
		},
	}
}

// Produces declares the request attributes PreRequest writes.
func (r *GroupScheduler) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		requestKeysKey: []Hash(nil),
	}
}

var requestKeysKey = fwkplugin.NewDataKey("group-keys", PluginType)

// readID is the non-empty string attribute at key ("" when absent).
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

// memberOf resolves the request's MEMBER id and the source that produced it:
//
//	branch  -> the derived branch id (branch-identity; the BASE mode)
//	session -> the agent-identity session attribute, when the branch plugin is not
//	           in the config (the queue key is then the session, as it always was)
//	(none)  -> nothing resolvable: anonymous traffic, R9 FCFS pass-through
//
// It is a PURE READ: the branch id is never written into the agent-identity slot
// (that slot keeps meaning "the client's session" for the rest of the stack).
func (r *GroupScheduler) memberOf(request *fwksched.InferenceRequest) (member, source string) {
	if request == nil {
		return "", ""
	}
	// reads are lazy: the hot Pick path pays at most two attribute loads (the
	// branch, and the session only when the branch is absent). Attribute loads are a
	// sync.Map read.
	if branch := readID(request, branchidentity.BranchIdentityKey); branch != "" {
		return branch, "branch"
	}
	if session := r.sessionAttr(request); session != "" {
		return session, "session"
	}
	return "", ""
}

// sessionAttr is the request's SESSION attribute: the system's agent-identity
// session first (the session the rest of the stack sees), else the branch
// plugin's copy of the same client header. Read-only.
func (r *GroupScheduler) sessionAttr(request *fwksched.InferenceRequest) string {
	if s := readID(request, agentidentity.AgentIdentityKey); s != "" {
		return s
	}
	return readID(request, branchidentity.BranchSessionKey)
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

// validateMemberProducer is the fail-closed MEMBER-SOURCE gate: the configured
// memberKey needs its identity plugin in the same config. Without it every
// request is anonymous and Pick degrades to R9 FCFS pass-through — the worst
// arm of the campaign, and a failure that is silent in the outcome metrics
// (Amendment 4 measured it twice), so it must fail at startup.
func (r *GroupScheduler) validateMemberProducer() error {
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
		fmt.Printf("group-scheduler: no '%s' plugin in the config — the SESSION attribute is absent, so the derived branch is the member key on every request (base mode)\n", agentidentity.PluginType)
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
	if err := r.validateMemberProducer(); err != nil {
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
// remembers the endpoint HANDLE so the ticker can poll its metrics (the
// event carries them before the first scrape has populated them — the
// capacity race).
//
// The engine SET is DISCOVERED here, never configured: an add-or-update
// creates (or refreshes) that endpoint's engine group, and a delete tears it
// down. The campaign's wave protocol rotates nodes every arm, so the
// scale-down path is exercised constantly and must not leave charges on a
// dead engine.
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
		return err // the loud mismatch failure
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
	fmt.Printf("group-scheduler: capacity learned via %s engine=%s after %.1fs: cacheBlockSize=%d numGpuBlocks=%d budgetBlocks=%d (that engine's 95%%)\n",
		via, label, wait.Seconds(), m.CacheBlockSize, m.CacheNumBlocks,
		int(r.cfg.GroupBudgetFraction*float64(m.CacheNumBlocks)))
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
	fmt.Printf("group-scheduler: CAPACITY HETEROGENEOUS — engine(s) %s deviate >1%% from the first measured engine. Per-engine budgets are taking each engine's OWN num_gpu_blocks, but a heterogeneous pool means the stack is not the homogeneous replica set the experiment assumes (a half-ready or differently-configured engine).\n",
		strings.Join(fresh, ", "))
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
	// tear the engine's group down: its members are exited, their homing
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

// pushEngineLoad feeds the safety module and the capacity path from the LIVE
// per-engine sample set — never from a remembered endpoint (see liveSource: a
// remembered handle's metrics are a construction-time CLONE, which is how the E-wave's
// gate latched closed; nothing in this core reads one).
//
// A vanished endpoint's snapshot is cleared, because a stale verdict on an engine the
// EPP no longer sees would keep it idle forever.
func (r *GroupScheduler) pushEngineLoad(now time.Time) {
	samples := r.sampleCandidates(now)
	if samples == nil {
		return // the framework has not offered a candidate set yet
	}
	r.publishSamples(samples)
	if r.metrics != nil {
		for engine, sm := range samples {
			if sm.stale {
				r.metrics.ObserveStaleSample(engine)
			}
		}
	}
	withLoad := r.sched.mod.safety.Name() != "none"
	seen := make(map[string]struct{}, len(samples))
	for engine, sm := range samples {
		seen[engine] = struct{}{}
		// CAPACITY always rides the live path: the event-time copy may carry an empty
		// geometry forever (the capacity race), and the live value is the same metric.
		if sm.blockSz > 0 && sm.numBlk > 0 && !r.sched.HasCapacityFor(engine) {
			if err := r.learnCapacity(engine, &fwkdl.Metrics{CacheBlockSize: sm.blockSz,
				CacheNumBlocks: sm.numBlk}, "live-detector"); err != nil {
				fmt.Printf("group-scheduler: CAPACITY REJECTED from the live detector engine=%s: %v\n", r.sched.GroupLabel(engine), err)
			}
		}
		if withLoad {
			// the sample's OWN timestamp, never `now`: if the upstream froze, the age is
			// visible instead of being papered over by the ticker's clock (the latch).
			r.sched.SetEngineLoad(engine, sm.kv, sm.waiting, sm.at)
		}
	}
	if withLoad {
		r.sched.ForgetEngineLoadsExcept(seen)
	}
	_ = now
}

// pollCapacity is the STARTUP LIVENESS check (WAVE-1 item 1): once a decode endpoint
// has been observed, its capacity must arrive within capacityLivenessSeconds, else
// the FCFS pass-through (A5) is being served silently — log loudly and raise the
// missing gauge.
//
// It reads NO metrics: the capacity RETRY rides the live detector samples
// (pushEngineLoad, same tick). A remembered endpoint handle carries a
// construction-time metrics CLONE, so a poll over clones could never see the geometry
// arrive late — the read that used to live here was itself the capacity-race bug's
// shape (and, for the gate, the E-wave latch).
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
		fmt.Printf("group-scheduler: CAPACITY MISSING engine=%s — its num_gpu_blocks is still unknown %.0fs after the endpoint was observed (%d endpoint(s) tracked): that engine seats nothing while the policy looks healthy, and any dispatch it takes is FCFS pass-through (A5). Check that the endpoint metrics carry cache block size/num blocks (--refresh-metrics-interval) and that the flow controller names this plugin as its saturationDetector (the live metrics path).\n",
			engineLabel(t.key), now.Sub(t.firstSeen).Seconds(), len(targets))
	}
}

// Saturation answers the flow controller's pool gate. HOT PATH (≈1 kHz): it must never
// take the scheduler's mutex, so the default answer is a single atomic load.
//
//	safety = "none" (default)            → 0.0, always: seats are the admission control
//	safety = "budget-fraction"           → MIN over engines of kv / groupBudgetFraction
//	                                       (1.0 = at the budget fraction), computed from
//	                                       the LIVE candidates the framework passes in;
//	                                       an engine whose metrics are missing or older
//	                                       than staleMultiple refresh intervals is
//	                                       EXCLUDED (stale → open).
//
// The per-engine samples are published for the ticker (which pushes them into the
// scheduler for the per-engine holds and exports them as series), so a metric that
// changes with NO dispatch is still seen — the latch class has a regression test.
func (r *GroupScheduler) Saturation(_ context.Context, candidates []fwkdl.Endpoint) float64 {
	// HOT PATH. Two things must never happen here: taking the scheduler's mutex, and
	// allocating per call (~1 kHz). So:
	//   1. the candidate SET is stored only when it CHANGES (an interface-value
	//      compare, no allocation) — the ticker samples from it;
	//   2. when the safety module is a noop (the default) the answer is a constant 0,
	//      with no sampling at all;
	//   3. with the memory-only module on, the score is recomputed at most once per
	//      framework refresh interval and otherwise read from the atomic — the metric
	//      it reads is scraped at that cadence anyway, so a faster recompute would be
	//      re-reading the same numbers.
	changed := r.storeCandidates(candidates)
	if r.sched.mod.safety.Name() == "none" {
		r.detectorScore.Store(math.Float64bits(0))
		return 0
	}
	now := time.Now()
	if !changed && now.Sub(time.Unix(0, r.scoreAt.Load())) < r.staleAfter() {
		// nothing new to look at: the atomic answer stands (the metric it reads is
		// scraped at this cadence anyway)
		return math.Float64frombits(r.detectorScore.Load())
	}
	r.scoreAt.Store(now.UnixNano())
	pool := r.poolScore(r.sampleCandidates(now))
	r.detectorScore.Store(math.Float64bits(pool))
	return pool
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

// poolScore is MIN over the fresh engines of kv/threshold (the flow controller's pool
// gate); 0 when nothing is usable (OPEN).
func (r *GroupScheduler) poolScore(samples map[string]engineSample) float64 {
	pool := -1.0
	for _, s := range samples {
		if s.stale {
			continue
		}
		score := r.sched.mod.safety.Score(s.kv, true)
		if pool < 0 || score < pool {
			pool = score
		}
	}
	if pool < 0 {
		return 0
	}
	return pool
}

// readSamples reads the live endpoint metrics into an immutable per-engine map.
func (r *GroupScheduler) readSamples(candidates []fwkdl.Endpoint, now time.Time) map[string]engineSample {
	out := make(map[string]engineSample, len(candidates))
	for _, e := range candidates {
		if e == nil {
			continue
		}
		engine := endpointKey(e)
		s := engineSample{at: now, stale: true}
		if m := e.GetMetrics(); m != nil {
			s.kv = m.KVCacheUsagePercent
			s.waiting = m.WaitingQueueSize
			s.blockSz = m.CacheBlockSize
			s.numBlk = m.CacheNumBlocks
			s.stale = now.Sub(m.UpdateTime) > r.staleAfter()
		}
		s.fired = !s.stale && r.sched.mod.safety.Fired(s.kv, true)
		s.score = 0
		if !s.stale {
			s.score = r.sched.mod.safety.Score(s.kv, true)
		}
		out[engine] = s
	}
	return out
}

func (r *GroupScheduler) publishSamples(m map[string]engineSample) { r.samples.Store(&m) }

// liveSamples is the published snapshot (nil-safe).
func (r *GroupScheduler) liveSamples() map[string]engineSample {
	if p := r.samples.Load(); p != nil {
		return *p
	}
	return nil
}

// staleAfter is the freshness bound for the LIVE input: a few framework refresh
// intervals. It is a wiring fact (the poll cadence), not a policy constant.
func (r *GroupScheduler) staleAfter() time.Duration {
	base := time.Duration(0)
	if r.handle != nil {
		base = r.handle.RefreshMetricsInterval()
	}
	if base <= 0 {
		base = 50 * time.Millisecond
	}
	return staleMultiple * base
}

// Pick is the dispatch decision (§5). The waiting set is the queue-head
// snapshot (S0-1); pre-capacity it degrades to FCFS pass-through (A5).
func (r *GroupScheduler) Pick(_ context.Context, band fwkfc.PriorityBandAccessor) (fwkfc.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	now := time.Now()
	var queues []fwkfc.FlowQueueAccessor
	var heads []Head
	band.IterateQueues(func(q fwkfc.FlowQueueAccessor) bool {
		if q == nil || q.Len() == 0 {
			return true
		}
		h := q.Peek()
		if h == nil {
			return true
		}
		member, source := "", ""
		reqID, size, fid := "", int64(0), ""
		if req := h.OriginalRequest(); req != nil {
			if ir := req.InferenceRequest(); ir != nil {
				member, source = r.memberOf(ir)
				reqID = ir.RequestID
				fid = ir.FairnessID // the wire-level identity (the queue key)
				size = int64(ir.RequestSizeBytes)
				if size <= 0 {
					size = int64(req.ByteSize())
				}
			}
		}
		queues = append(queues, q)
		heads = append(heads, Head{Member: member, Source: source,
			FairnessID: fid, ReqID: reqID, Arrival: h.EnqueueTime(), SizeBytes: size})
		return true
	})
	if len(heads) == 0 {
		return nil, nil //nolint:nilnil
	}
	// A5: before the measured capacity arrives, FCFS pass-through (once
	// per process log; the gauge counts it)
	if !r.sched.HasCapacity() {
		r.fcfsOnce.Do(func() {
			fmt.Println("group-scheduler: num_gpu_blocks not yet measured; serving FCFS until the endpoint metrics arrive")
		})
		_, label := r.sched.FCFSServed()
		r.metrics.ObserveFCFS(label)
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

// PreRequest runs after the request leaves the queue (dispatch): the
// exact tier-A keys from the TokenizedRequest correct the estimate
// (S0-1 d), the in-flight charge was already taken at Pick.
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
	id, _ := r.memberOf(request)
	if id == "" {
		return nil
	}
	keys := r.blockKeysFor(request)
	if keys == nil {
		return errors.New("group-scheduler: no TokenizedRequest on the request (token-producer required; no tier-B fallback)")
	}
	request.PutAttribute(requestKeysKey, keys)
	size := int64(request.RequestSizeBytes)
	// The engine here is AUTHORITATIVE (the pendingEngine idiom): Pick
	// charged the seat to the member's homed engine, and the framework's
	// actual choice moves that charge if it disagrees.
	r.sched.DispatchedOn(id, request.RequestID, engine, keys, size, time.Now())
	return nil
}

// blockKeysFor derives the chained token-block keys (tier A).
func (r *GroupScheduler) blockKeysFor(request *fwksched.InferenceRequest) []Hash {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	tp := request.Body.TokenizedRequest
	if len(tp.Prompts) == 0 {
		return nil
	}
	var all []uint32
	for _, p := range tp.Prompts {
		all = append(all, p.TokenIDs...)
	}
	return r.sched.HashTokens(all, tp.CacheSalt, r.cfg.BlockSizeTokens)
}

// ResponseBody: the first chunk is prefill complete (the cold-entry
// release, B3); EndOfStream is the finalizer for every termination
// cause (in-flight decrement always; the payback ledger natural-only).
func (r *GroupScheduler) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, target *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	id, _ := r.memberOf(request)
	if id == "" {
		return
	}
	if response.StartOfStream {
		r.sched.ChunkAt(id, time.Now())
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
	// The settle's engine comes from the response's own endpoint metadata
	// (F6: Settle without an endpoint silently folds B's keys into A's
	// record). An empty metadata id keeps the binding.
	engine := ""
	if target != nil && target.ID.String() != "" {
		engine = target.ID.String()
	}
	r.sched.SettleOnUsage(id, request.RequestID, engine, promptTokens, cachedTokens, completionTokens,
		response.TerminationCause == fwkrc.TerminationCauseNatural, time.Now())
}

// DumpState reports the group snapshot (§13).
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
