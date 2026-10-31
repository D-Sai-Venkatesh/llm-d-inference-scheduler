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
	_ fwksched.Filter             = &GroupScheduler{}
)

// GroupScheduler is a single named instance shared by every hookup.
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
	// pooledCount is the number of endpoints the pooled (engineGroups=single)
	// group is summing, kept as an ATOMIC so the hot path costs one load and a
	// no-change swap; pooledOnce LATCHES the AUDIT-N1 warning so it is emitted
	// exactly once without a lock-guarded comparison per request.
	pooledCount atomic.Int64
	pooledOnce  sync.Once

	// ---- the placement pin (E4a, bindingMode=seat) ----
	//
	// pins mirrors the scheduler's member→engine SEAT binding so the Filter can
	// read it WITHOUT taking s.mu (see filter.go's lock discipline). The
	// scheduler pushes every seat/unseat/rebind through onBind/onUnbind; nothing
	// here infers anything.
	pins *pinTable
}

// endpointState is one discovered endpoint's capacity-acquisition state.
type endpointState struct {
	ep metricsSource
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
	// WAVE-1 item 2: the member-key mode is the one knob whose EFFECT is
	// invisible in the config text ("auto" resolves per request), so state
	// the resolution rule in words at startup and log the first resolution
	// at runtime.
	fmt.Printf("group-scheduler: member key source: %s (%s); session ordering: %s\n",
		cfg.MemberKey, memberKeyRule(cfg.MemberKey), cfg.SessionOrder)
	// The engine scope is the one knob whose EFFECT is a discovered set, so
	// state it in words: a run whose config says per-engine but whose EPP
	// sees one endpoint is numerically a single-engine run.
	if cfg.perEngine() {
		fmt.Printf("group-scheduler: engine groups: PER-ENGINE — one fairness group per DISCOVERED endpoint (own index, own 95%% budget, own cold-entry cap, own rotation loop); identity stays global; members are homed at first seat and never migrate while active. Groups are created on endpoint add and torn down on endpoint delete.\n")
	} else {
		fmt.Printf("group-scheduler: engine groups: SINGLE — one group sums every endpoint into one pool (correct only for one engine; the multi-pod arms set engineGroups=per-engine)\n")
	}
	// F-arms: echo each knob in words when it is NOT at its default, so a run's
	// artifacts state which rotation policy actually ran.
	fmt.Printf("group-scheduler: rotation policy: mode=%s quantum=%d quantumMode=%s ceiling=%d entryOrder=%s victimOrder=%s rotateWhenFree=%v maxNumBatchedTokens=%d longQuantum=%d\n",
		cfg.RotationMode, cfg.quantum(), cfg.QuantumMode, cfg.QuantumCeiling, cfg.EntryOrder,
		cfg.VictimOrder, cfg.RotateWhenFree, cfg.MaxNumBatchedTokens, cfg.LongQuantum)
	if cfg.residueOn() {
		fmt.Printf("group-scheduler: residue model: TIEBREAK — a per-engine shadow of vLLM's free-block queue (allocate from the head, free tail-block-first to the tail, live blocks never evictable) predicts the surviving PREFIX of a returning member's chain and breaks ties among FIT-PASSING engines only (never a ranker of two cold engines). Reconciled at settle against usage.cached_tokens; parameter-free (no clock, no threshold, no cache read).\n")
	} else {
		fmt.Printf("group-scheduler: residue model: OFF — the seat-time engine choice is fit → least relative backlog per seat → cheapest-victim engine.\n")
	}
	// E4a: the binding model is the ONE variable between the E3 and E4a arms, so
	// the run artifact must state it in words as well as in the config echo.
	if cfg.seatPin() {
		fmt.Printf("group-scheduler: binding mode: SEAT = PIN — the member→engine binding IS the current seat (created at seat, dropped at exit; no TTL, no pre-seat homing); waiters are ONE GLOBAL POOL and the seat engine is chosen by fit → least relative backlog per seat → cheapest-victim; placement is pinned by THIS plugin's scheduling Filter (no external session-affinity-filter).\n")
	} else {
		fmt.Printf("group-scheduler: binding mode: MEMBER-STABLE — the engine is chosen ONCE at first sight and kept (bindingTtlSeconds=%.0f); waiters are partitioned per engine; placement is pinned externally (session-affinity-filter) or not at all.\n", cfg.BindingTTLSeconds)
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
	// E4a: the pin table follows the SEATS. Installed in BOTH binding models —
	// the Filter ignores it outside seat mode, and a table that is filled in
	// member-stable mode costs one map write per seat.
	r.sched.onBind = func(member, engine string) { r.pins.set(member, engine) }
	r.sched.onUnbind = func(member string) { r.pins.del(member) }
	// E4b: the counterfactual export. Every seat-time engine choice reports the
	// predicted surviving fraction and whether the MODEL decided it; every settle
	// reports (predicted, actual) so the arm can score its own argmax post-hoc
	// against usage.cached_tokens. An engine that drops to fit-only says so once.
	r.sched.onResiduePick = func(engine string, predicted float64, decided bool) {
		r.metrics.ObserveResiduePredicted(engineLabel(engine), predicted, decided)
	}
	r.sched.onResidueSettle = func(engine string, predicted, actual float64) {
		r.metrics.ObserveResidueSettled(engineLabel(engine), predicted, actual)
	}
	r.sched.onEngineGateSkip = func(engine string) {
		r.metrics.ObserveEngineGateSkip(engineLabel(engine))
	}
	r.sched.onValveBlockedEviction = func(engine string) {
		r.metrics.ObserveValveBlockedEviction(engineLabel(engine))
	}
	r.sched.onResidueDisabled = func(engine string) {
		r.metrics.SetResidueDisabled(engineLabel(engine))
		fmt.Printf("group-scheduler: RESIDUE MODEL DISABLED on engine=%s — sustained prediction error above %.2f over >=%d reconciled requests. That engine is now ranked by fit → backlog only (never by residue). Cause is almost always a state the model cannot see: an engine restart, a second EPP replica, or traffic outside this EPP.\n",
			engineLabel(engine), residueErrLimit, residueMinCalibration)
	}
	r.sched.onEngineGone = func(engine string, members int) {
		r.metrics.ObserveEngineGone()
		fmt.Printf("group-scheduler: ENGINE GONE %s — exited %d member(s) and dropped the group; their homing is released for re-placement\n",
			engineLabel(engine), members)
	}
	if handle != nil {
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
			r.metrics.Refresh()
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

// memberOf resolves the request's MEMBER id, its SESSION attribute, and which
// source produced the member (Config.MemberKey):
//
//	branch  -> the derived branch id (branch-identity)
//	session -> the agent-identity session attribute (pre-WAVE-1 behaviour)
//	(none)  -> nothing resolvable: anonymous traffic, R9 FCFS pass-through
//
// It is a PURE READ. The branch id is never written into the agent-identity
// slot (in session mode that slot must hold the real session), and the session
// attribute is never derived from the branch here — the session layer applies
// its own "session := member when absent" rule, in ONE place, in the
// scheduler.
func (r *GroupScheduler) memberOf(request *fwksched.InferenceRequest) (member, session, source string) {
	if request == nil {
		return "", "", ""
	}
	// reads are lazy: the hot Pick path pays at most two attribute loads (the
	// branch, and the session only when it is the member source). At most two
	// per head is the ceiling — attribute loads are a sync.Map read.
	branch := readID(request, branchidentity.BranchIdentityKey)
	session = r.sessionAttr(request)
	switch r.cfg.MemberKey {
	case "branch":
		if branch != "" {
			return branch, session, "branch"
		}
		return "", session, ""
	case "session":
		if session != "" {
			return session, session, "session"
		}
		return "", session, ""
	default: // "auto": branch when present, else the session, else R9
		if branch != "" {
			return branch, session, "branch"
		}
		if session != "" {
			return session, session, "session"
		}
		return "", "", ""
	}
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
	switch r.cfg.MemberKey {
	case "branch":
		if !b {
			return fmt.Errorf("memberKey=\"branch\" requires the '%s' plugin in the same config (the member id comes from its derived branch)", branchidentity.PluginType)
		}
	case "session":
		if !s {
			return fmt.Errorf("memberKey=\"session\" requires the '%s' plugin in the same config (the member id comes from the session attribute)", agentidentity.PluginType)
		}
	default:
		if !b && !s {
			return fmt.Errorf("memberKey=%q needs at least one identity plugin ('%s' or '%s'); with neither, every request is anonymous and the group policy degrades to R9 FCFS",
				"auto", branchidentity.PluginType, agentidentity.PluginType)
		}
	}
	if b && !s {
		fmt.Printf("group-scheduler: no '%s' plugin in the config — the SESSION attribute is absent, so the session layer runs with session := member (base mode)\n", agentidentity.PluginType)
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
	st.ep = ep // the handle is refreshed: the datalayer swaps metrics in place
	n := int64(len(r.endpoints))
	pooled := !r.cfg.perEngine()
	want := int64(0)
	if pooled {
		want = n
	}
	// One atomic swap: the gauge is touched only when the pool size CHANGES, so
	// rememberEndpoint (called on every PreRequest) costs a load + a swap.
	if r.pooledCount.Swap(want) != want {
		r.metrics.SetPooledEndpoints(int(want))
	}
	if pooled && n > 1 {
		// AUDIT N1: the knob's default fails OPEN, so a multi-endpoint pool is
		// announced ONCE (sync.Once — latched, never re-tested per request) and
		// reported forever as llm_d_epp_group_pooled_endpoints.
		r.pooledOnce.Do(func() {
			fmt.Printf("group-scheduler: WARNING engineGroups=single with %d discovered endpoints — the policy is POOLING independent engines into ONE group: one BlockIndex (a block on A counted as if B held it), one budget, one seat set, and the capacity comes from whichever engine reported first. That is correct only for a single-engine stack: set engineGroups=per-engine for a multi-engine stack. llm_d_epp_group_pooled_endpoints reports the pool size.\n", n)
		})
	}
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

// pushEngineLoad feeds the PER-ENGINE saturation gate (engineGate=per-engine) from
// the endpoint metrics the plugin already polls. The policy never scrapes and never
// blocks; it receives one snapshot per engine per tick and applies the DETECTOR'S
// OWN FORMULA per engine (max(waiting/queueThreshold, kv/kvThreshold) >= 1).
//
// A vanished endpoint's snapshot is cleared, because a stale "saturated" flag on an
// engine the EPP no longer sees would keep a live sibling's gate closed forever.
func (r *GroupScheduler) pushEngineLoad(now time.Time) {
	if !r.cfg.engineGating() {
		return
	}
	r.capMu.Lock()
	type target struct {
		key string
		ep  metricsSource
	}
	targets := make([]target, 0, len(r.endpoints))
	for key, st := range r.endpoints {
		if st == nil || st.ep == nil {
			continue
		}
		targets = append(targets, target{key: key, ep: st.ep})
	}
	r.capMu.Unlock()
	sort.Slice(targets, func(i, j int) bool { return targets[i].key < targets[j].key })
	seen := make(map[string]struct{}, len(targets))
	for _, t := range targets {
		m := t.ep.GetMetrics()
		if m == nil {
			continue
		}
		seen[t.key] = struct{}{}
		r.sched.SetEngineLoad(t.key, m.KVCacheUsagePercent, m.WaitingQueueSize, now)
	}
	r.sched.ForgetEngineLoadsExcept(seen)
}

// pollCapacity is the METRICS-POLL FALLBACK (WAVE-1 item 1), run on the
// ticker while the capacity is unknown: re-read every remembered endpoint's
// metrics (the datalayer refreshes them in place on the same object) and take
// the first one that carries the cache geometry. It also carries the STARTUP
// LIVENESS check: once a decode endpoint has been observed, the capacity must
// arrive within capacityLivenessSeconds, else the FCFS pass-through (A5) is
// being served silently — log loudly and raise the missing gauge.
func (r *GroupScheduler) pollCapacity(now time.Time) {
	r.capMu.Lock()
	type pollTarget struct {
		key       string
		ep        metricsSource
		firstSeen time.Time
		warned    bool
	}
	targets := make([]pollTarget, 0, len(r.endpoints))
	for key, st := range r.endpoints {
		if st == nil || st.ep == nil {
			continue
		}
		targets = append(targets, pollTarget{key: key, ep: st.ep, firstSeen: st.firstSeen, warned: st.warned})
	}
	r.capMu.Unlock()

	// sort for a deterministic log/poll order (the map is not)
	sort.Slice(targets, func(i, j int) bool { return targets[i].key < targets[j].key })

	for _, t := range targets {
		if r.sched.HasCapacityFor(t.key) {
			continue
		}
		if err := r.learnCapacity(t.key, t.ep.GetMetrics(), "metrics-poll"); err != nil {
			// a block-size mismatch is loud and permanent: keep it in the
			// log, don't spin the ticker on it
			if !t.warned {
				fmt.Printf("group-scheduler: CAPACITY REJECTED from metrics-poll engine=%s: %v\n", r.sched.GroupLabel(t.key), err)
			}
		}
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
		fmt.Printf("group-scheduler: CAPACITY MISSING engine=%s — its num_gpu_blocks is still unknown %.0fs after the endpoint was observed (%d endpoint(s) polled): that engine seats nothing while the policy looks healthy, and any dispatch it takes is FCFS pass-through (A5). Check that the endpoint metrics carry cache block size/num blocks (--refresh-metrics-interval), and that the EPP was rolled AFTER decode readiness.\n",
			engineLabel(t.key), now.Sub(t.firstSeen).Seconds(), len(targets))
	}
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
		member, scope, source := "", "", ""
		reqID, size, fid := "", int64(0), ""
		if req := h.OriginalRequest(); req != nil {
			if ir := req.InferenceRequest(); ir != nil {
				member, scope, source = r.memberOf(ir)
				reqID = ir.RequestID
				fid = ir.FairnessID // the wire-level identity (the queue key)
				size = int64(ir.RequestSizeBytes)
				if size <= 0 {
					size = int64(req.ByteSize())
				}
			}
		}
		queues = append(queues, q)
		heads = append(heads, Head{Member: member, Session: scope, Source: source,
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
	id, _, _ := r.memberOf(request)
	if id == "" {
		return nil
	}
	// V3/R6: the engine-side stagger — RESIDENT turns get priority over
	// cold ENTRY prefills in vLLM's waiting loop (Phase-2 deployments
	// with --scheduling-policy priority; lower value = higher
	// priority). The split is entry-vs-resident, not seated-vs-not:
	// seating happens in Pick BEFORE PreRequest, so a seated test would
	// mark every tracked request high. EntryTurn marks this residency's
	// first dispatch (the entry) low; every later turn is high.
	if r.cfg.PriorityHeader == "engine" && request.Headers != nil {
		if r.sched.EntryTurn(id) {
			request.Headers["X-Vllm-Priority"] = "10"
		} else {
			request.Headers["X-Vllm-Priority"] = "0"
		}
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
	id, _, _ := r.memberOf(request)
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
