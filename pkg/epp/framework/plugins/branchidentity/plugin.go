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

package branchidentity

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "branch-identity"

// BranchIdentityKey is the request-attribute key carrying the DERIVED branch id.
// The director uses it as the fairness id (the flow-control queue key) when the
// request carried no explicit fairness header. Like agent-identity, the key
// carries no producer name: any plugin able to derive a branch is an acceptable
// source.
var BranchIdentityKey = fwkplugin.NewDataKey("branch-identity", "")

// BranchSessionKey is the request-attribute key carrying the client's session
// id, when it sent one. It is deliberately SEPARATE from the branch id: the
// session is an attribute for the session fairness layer, the branch is the
// queue key (design: "the client session header is kept as a SEPARATE attribute
// for stage 2 — neither overwrites the other").
var BranchSessionKey = fwkplugin.NewDataKey("branch-session", "")

// branchMarkerKey is the request-attribute key carrying the request's internal
// marker id — the handle all three hooks use to move this request between its
// branch's pending and in-flight sets. It equals the request id when the client
// supplied one and a per-request synthetic id otherwise, so a body-less request
// id cannot collapse two arrivals into one marker.
var branchMarkerKey = fwkplugin.NewDataKey("branch-marker", "")

const (
	// defaultBlockSizeTokens is the identity chunk length in pseudo-tokens
	// (4 canonical bytes each, so 256-byte chunks — the survey's convention).
	// Independent of the engine's KV block size: identity only needs a stable
	// chunking of the identity stream.
	defaultBlockSizeTokens = 64
	// defaultMarkerTTLSeconds bounds how long an arrival/in-flight marker keeps
	// its branch outstanding when the request never reports an end of stream.
	// That happens only to requests REJECTED before dispatch (admission
	// TTL/capacity): the response hook is forced on the abort path for every
	// request that picked a pod, but a rejected request never gets one.
	//
	// The horizon is the CLIENT's request timeout (the harness runs 3600s), NOT
	// the flow-control queue's 5-minute idle GC: those are different clocks, and
	// on A0 16% of requests waited more than 600s for admission, so a 600s marker
	// TTL would expire markers of requests that are still legitimately queued and
	// hand their branch to a concurrent sibling as if it were free. Pruning an
	// arrival errs toward CONTINUATION, the safe direction, which is why the TTL
	// is generous rather than tight.
	defaultMarkerTTLSeconds = 3600
	// defaultSweepSeconds is the cap-enforcement tick.
	defaultSweepSeconds = 60
	// defaultMaxBranches is the table cap (the ONLY eviction path; there is no
	// time-based GC — see EnforceCap). Generous: a record is ~200 bytes, and
	// stage 0 measured ~9.6k threads per 90-minute window.
	defaultMaxBranches = 100000
)

// defaultSessionHeaders are the client session headers read for scoping, in
// priority order: the same provider headers the agent-identity plugin resolves,
// so the session ATTRIBUTE stage 2 needs is the session the rest of the system
// already sees.
var defaultSessionHeaders = []string{
	agentidentity.ClaudeCodeSessionHeader,
	agentidentity.OpenCodeSessionHeader,
	agentidentity.CodexSessionHeader,
	agentidentity.CodexSessionHeaderLegacy,
}

// Parameters is the plugin's configuration block.
type Parameters struct {
	// BlockSizeTokens is the identity chunk length in pseudo-tokens (4 bytes
	// each). Zero selects the default (64 → 256-byte chunks).
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// MarkerTTLSeconds bounds how long an outstanding marker (arrival or in
	// flight) that never settles keeps its branch busy. It must be at least the
	// client's request timeout — a marker that expires while its request is still
	// queued makes a live branch look free. There is no idle/branch TTL: branches
	// are retained (stage 0 measured a p99.9 inter-turn gap of 3,036s, so any
	// silence timeout would mint false new branches when dormant threads return).
	MarkerTTLSeconds float64 `json:"markerTtlSeconds,omitempty"`
	// QueuedTTLSeconds is the deprecated name of MarkerTTLSeconds, accepted so an
	// older arm's config still parses.
	QueuedTTLSeconds float64 `json:"queuedTtlSeconds,omitempty"`
	// SweepSeconds is the cap-enforcement tick.
	SweepSeconds float64 `json:"sweepSeconds,omitempty"`
	// MaxBranches caps the branch table: the only eviction path, idlest
	// non-outstanding branches first. Zero selects the default.
	MaxBranches int `json:"maxBranches,omitempty"`
	// SessionHeaders are client session headers prepended to the built-in list.
	SessionHeaders []string `json:"sessionHeaders,omitempty"`
	// Salt is folded into the block-hash chain. Empty (the default) keeps ids
	// reproducible across instances.
	Salt string `json:"salt,omitempty"`
	// ForkOnBusy is the continuation rule's busy half: extend a busy tip → fork
	// a new branch. Default true (the design's rule). False makes every match a
	// continuation — the v2c failure mode, kept only as a measurement arm.
	ForkOnBusy *bool `json:"forkOnBusy,omitempty"`
	// StickyBusyGuard is the H1 guard: a request that extends a tip AND begins
	// with the tip's assistant reply is a proven continuation even when the
	// branch looks busy. Default true.
	StickyBusyGuard *bool `json:"stickyBusyGuard,omitempty"`
	// RetryRule is S2: a request byte-identical to a tip whose branch has nothing
	// IN FLIGHT (only arrival markers) is a client retry, not a sibling, and
	// continues the branch. Default true.
	RetryRule *bool `json:"retryRule,omitempty"`
	// LogRequests emits one JSON line per request on stdout:
	//
	//	branch-identity: request {"seq":…,"scope":…,"derived":…,"outcome":…}
	//
	// Default FALSE — it is an INSTRUMENT, not a feature, and it exists because in
	// shadow mode the derived id reaches no span: the span's fairness id is the
	// client's explicit header (that is what keeps the arm an anchor), and the
	// EPP's request id does not join to a harness's records. A run that needs the
	// per-request derived-vs-thread confusion matrix turns this on and joins these
	// lines to its records by (scope, arrival order). It changes no state and no
	// decision; with it off the plugin is byte-for-byte the same code path.
	LogRequests bool `json:"logRequests,omitempty"`
	// LogHeaders adds the client's own identity headers to that line:
	//
	//	"headers":{"x-correlation-id":"…","x-request-id":"…"}
	//
	// It exists for the UNSCOPED stage-1 arm, where no session header is sent and
	// `scope` is therefore empty: the census (StateDumper) cannot attribute a
	// branch to a client thread there, but the harness's thread id still travels
	// on a NON-fairness header, so copying it makes the per-request
	// derived-vs-thread confusion matrix exact. A FIXED whitelist of two headers,
	// read-only: it is gated on LogRequests and touches neither the identity nor
	// any decision (`TestLogHeadersKeepResolveOutcomeIdentical` pins that).
	LogHeaders bool `json:"logHeaders,omitempty"`
}

// loggedHeaderNames is the logHeaders whitelist: the client's correlation id (the
// thread ground truth a harness reports in its records) and its request id.
var loggedHeaderNames = []string{"x-correlation-id", "x-request-id"}

// maxLoggedHeaderLen truncates a copied header value so one malformed client
// cannot blow up a log line.
const maxLoggedHeaderLen = 128

// validate resolves defaults in place and rejects impossible values.
func (p *Parameters) validate() error {
	if p.BlockSizeTokens == 0 {
		p.BlockSizeTokens = defaultBlockSizeTokens
	}
	if p.BlockSizeTokens < 0 {
		return fmt.Errorf("blockSizeTokens must be > 0, got %d", p.BlockSizeTokens)
	}
	if p.MarkerTTLSeconds == 0 {
		p.MarkerTTLSeconds = p.QueuedTTLSeconds
	}
	if p.MarkerTTLSeconds == 0 {
		p.MarkerTTLSeconds = defaultMarkerTTLSeconds
	}
	if p.MarkerTTLSeconds < 0 {
		return fmt.Errorf("markerTtlSeconds must be >= 0, got %v", p.MarkerTTLSeconds)
	}
	if p.SweepSeconds == 0 {
		p.SweepSeconds = defaultSweepSeconds
	}
	if p.SweepSeconds <= 0 {
		return fmt.Errorf("sweepSeconds must be > 0, got %v", p.SweepSeconds)
	}
	if p.MaxBranches == 0 {
		p.MaxBranches = defaultMaxBranches
	}
	if p.MaxBranches < 0 {
		return fmt.Errorf("maxBranches must be > 0, got %d", p.MaxBranches)
	}
	return nil
}

var (
	_ fwkrc.RequestHeaderProcessor = &Plugin{}
	_ fwkrc.PreRequest             = &Plugin{}
	_ fwkrc.ResponseBodyProcessor  = &Plugin{}
	_ fwkplugin.ProducerPlugin     = &Plugin{}
	_ fwkplugin.StateDumper        = &Plugin{}
)

// Plugin derives branch identity from the request body and owns the branch
// table. One instance serves every hook (the groupsched pattern): the
// busy/chain state is a single mutex-guarded struct rather than a cross-plugin
// channel, and the framework registers a multi-interface plugin at every
// interface it satisfies.
type Plugin struct {
	typedName fwkplugin.TypedName

	idx     *branchIndex
	metrics *branchMetrics

	markerTTL   time.Duration
	sweep       time.Duration
	forkOnBusy  bool
	sticky      bool
	retry       bool
	logRequests bool
	logHeaders  bool
	sessions    []string
	// now is the clock seam: the replay validation (stage 0c) drives recorded
	// arrival/dispatch/settle times through the real hooks, so every time
	// observation the table makes must come from here rather than time.Now.
	now func() time.Time

	anonSeq atomic.Uint64
	// logSeq orders the LogRequests instrument's lines (arrival order, so a scoring
	// pass can align a scope's lines with its records' turns without a request-id
	// join).
	logSeq atomic.Uint64
}

// Factory builds the branch-identity plugin.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", PluginType, err)
	}

	p := &Plugin{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		markerTTL:   time.Duration(params.MarkerTTLSeconds * float64(time.Second)),
		sweep:       time.Duration(params.SweepSeconds * float64(time.Second)),
		forkOnBusy:  params.ForkOnBusy == nil || *params.ForkOnBusy,
		sticky:      params.StickyBusyGuard == nil || *params.StickyBusyGuard,
		retry:       params.RetryRule == nil || *params.RetryRule,
		logRequests: params.LogRequests,
		logHeaders:  params.LogRequests && params.LogHeaders,
		sessions:    sessionHeaders(params.SessionHeaders),
		now:         time.Now,
	}
	p.idx = newBranchIndex(params.BlockSizeTokens, params.Salt, p.markerTTL, params.MaxBranches)
	p.metrics = newBranchMetrics()

	// The effective config is echoed at startup so a run's arm is verifiable
	// from its artifacts (the groupsched convention).
	if echo, err := json.Marshal(params); err == nil {
		fmt.Printf("%s: effective config %s\n", PluginType, echo)
	}

	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := p.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", PluginType, err)
			}
		}
		if p.idx.maxBranches > 0 {
			go p.runSweeper(handle.Context())
		}
	}
	return p, nil
}

// TypedName returns the type and name of the plugin.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces declares the attributes written at the request-header hook.
func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		BranchIdentityKey: "",
		BranchSessionKey:  "",
		branchMarkerKey:   "",
	}
}

// RequestHeader derives the branch id before admission control and makes it the
// request's fairness id — the flow-control queue key must exist before the
// request enqueues, and this hook runs after body parsing but before Admit.
//
// It also publishes the branch (and the session, when the client sent one) as
// request attributes: the director's fairness-id fallback and the stage-2 session
// policy read them, and keeping the session under its own key is what lets the
// policy tell a sibling from a newcomer without making the session a queue key.
//
// It never fails the request: a body shape the flattener does not cover simply
// leaves the attribute unset, and the fairness id falls back to the explicit
// header, the agent identity, or the default id.
func (p *Plugin) RequestHeader(_ context.Context, request *fwksched.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return nil
	}
	// An explicit x-llm-d-inference-fairness-id header wins (operators keep a
	// manual override). It is counted, because a harness that sets it turns the
	// whole derivation into a no-op without any other visible signal — the
	// stage-1 harness fix (no --session-header=x-llm-d-inference-fairness-id) is
	// deployment config, and this counter is how a run proves it was applied.
	explicit := request.FairnessID != ""

	start := p.now()
	identity, err := tokenizer.FlattenIdentityBytes(request.Body)
	if err != nil {
		p.metrics.observeUnflattened()
		return nil
	}
	chain := p.idx.Chain(identity)
	roles := messageRoles(request.Body)
	scope := p.sessionID(request)
	marker := p.markerID(request)
	flattened := p.now()

	res := p.idx.Resolve(p.now(), Arrival{
		Stream: identity, Chain: chain, Scope: scope, Roles: roles, Marker: marker,
	}, p.forkOnBusy, p.sticky, p.retry)

	p.metrics.observeFlatten(flattened.Sub(start))
	p.metrics.observeResolveLatency(res.Elapsed)
	p.metrics.observeResolve(res)

	request.PutAttribute(BranchIdentityKey, res.ID)
	request.PutAttribute(branchMarkerKey, marker)
	if explicit {
		p.metrics.observeExplicitOverride()
	} else {
		// Otherwise the derived branch IS the queue key.
		request.FairnessID = res.ID
	}
	if scope != "" {
		request.PutAttribute(BranchSessionKey, scope)
	}
	p.logRequest(request, res, scope, explicit, len(chain), len(identity))
	return nil
}

// logRequest emits the per-request instrument line when Parameters.LogRequests is
// on (off by default). One write per line, in arrival order, carrying exactly what
// a scoring pass needs and nothing that requires a join into a harness's private
// id space: the client SCOPE (the session header, which for a shadow arm is the
// thread id), the DERIVED branch, the outcome, and the identity size.
func (p *Plugin) logRequest(request *fwksched.InferenceRequest, res ResolveResult, scope string, explicit bool, blocks, bytes int) {
	if !p.logRequests {
		return
	}
	line, err := json.Marshal(struct {
		Seq      uint64            `json:"seq"`
		TS       string            `json:"ts"`
		Scope    string            `json:"scope"`
		Derived  string            `json:"derived"`
		Outcome  string            `json:"outcome"`
		Retry    bool              `json:"retry,omitempty"`
		Sticky   bool              `json:"sticky,omitempty"`
		Explicit bool              `json:"explicit"`
		Blocks   int               `json:"blocks"`
		Bytes    int               `json:"bytes"`
		Marker   string            `json:"marker"`
		Headers  map[string]string `json:"headers,omitempty"`
	}{
		Seq: p.logSeq.Add(1), TS: p.now().UTC().Format(time.RFC3339Nano), Scope: scope,
		Derived: res.ID, Outcome: string(res.Outcome), Retry: res.Retry, Sticky: res.StickyBusy,
		Explicit: explicit, Blocks: blocks, Bytes: bytes, Marker: markerOf(request),
		Headers: p.loggedHeaders(request),
	})
	if err != nil {
		return
	}
	fmt.Printf("branch-identity: request %s\n", line)
}

// loggedHeaders copies the whitelisted client headers from the request (lowercase
// keys, as the handler stores them; a case-insensitive scan covers a client that
// framed them otherwise). Pure read — no attribute, no id, no decision.
func (p *Plugin) loggedHeaders(request *fwksched.InferenceRequest) map[string]string {
	if !p.logHeaders || request == nil || len(request.Headers) == 0 {
		return nil
	}
	out := make(map[string]string, len(loggedHeaderNames))
	for _, name := range loggedHeaderNames {
		v := request.Headers[name]
		if v == "" {
			for k, cand := range request.Headers {
				if strings.EqualFold(k, name) {
					v = cand
					break
				}
			}
		}
		if v == "" {
			continue
		}
		if len(v) > maxLoggedHeaderLen {
			v = v[:maxLoggedHeaderLen]
		}
		out[name] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// PreRequest records the DISPATCH transition (admitted → on its way to an
// engine), moving this request's marker from its branch's pending set to its
// in-flight set. The branch is already busy from ARRIVAL — RequestHeader marks
// it — so nothing about identity depends on this hook; it exists so the
// outstanding set names WHICH requests are running. Marking busy only here would
// leave the queueing interval unprotected, and under load that is most of the
// busy period.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, BranchIdentityKey); ok && id != "" {
		p.idx.Dispatch(id, markerOf(request), p.now())
	}
	return nil
}

// ResponseBody settles the branch on the end of the response lifecycle, removing
// this request's marker from its branch's in-flight set. The director forces this
// call on abort/disconnect for every dispatched request, so the busy flag cannot
// wedge.
func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, BranchIdentityKey); ok && id != "" {
		p.idx.Settle(id, markerOf(request), p.now())
	}
}

// DumpState reports the branch table.
func (p *Plugin) DumpState() (json.RawMessage, error) {
	now := time.Now()
	snap := p.idx.Snapshot(now)
	p.metrics.setRecords(snap.Branches)
	return json.Marshal(snap)
}

// sessionID returns the client's session id from the configured header list.
func (p *Plugin) sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	for _, header := range p.sessions {
		if id := strings.TrimSpace(request.Headers[header]); id != "" {
			return id
		}
	}
	return ""
}

func (p *Plugin) runSweeper(ctx context.Context) {
	ticker := time.NewTicker(p.sweep)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.sweepOnce(time.Now())
		}
	}
}

// sweepOnce enforces the table cap and refreshes the branch_records gauge. There
// is no idle GC (see branchIndex.EnforceCap): a dormant thread that returns must
// find its branch, or it is minted a fresh id and re-enters as a newcomer.
func (p *Plugin) sweepOnce(now time.Time) {
	if removed := p.idx.EnforceCap(now); removed > 0 {
		p.metrics.addEvicted(removed)
	}
	p.metrics.setRecords(p.idx.Len())
}

// markerID returns the request's marker handle for the branch's outstanding set:
// its request id when the client/framework supplied one, a per-request synthetic
// id otherwise. The value is published as a request attribute so every hook of
// the same request agrees on it.
func (p *Plugin) markerID(request *fwksched.InferenceRequest) string {
	if id := strings.TrimSpace(request.RequestID); id != "" {
		return id
	}
	return fmt.Sprintf("anon-%d", p.anonSeq.Add(1))
}

// markerOf reads the marker an earlier hook published, falling back to the
// request id (and, if even that is absent, to a constant so the operations stay
// no-ops rather than corrupting another request's marker).
func markerOf(request *fwksched.InferenceRequest) string {
	if marker, ok := fwksched.ReadRequestAttribute[string](request, branchMarkerKey); ok && marker != "" {
		return marker
	}
	return request.RequestID
}

// sessionHeaders returns the operator's extras followed by the defaults, lower-
// cased (request headers are stored lowercased).
func sessionHeaders(extras []string) []string {
	out := make([]string, 0, len(extras)+len(defaultSessionHeaders))
	for _, h := range extras {
		out = append(out, strings.ToLower(h))
	}
	return append(out, defaultSessionHeaders...)
}

// messageRoles returns the request's message roles in wire order. Body shapes
// without a message array return nil: the sticky-busy guard needs message
// structure and simply cannot fire for them.
func messageRoles(body *fwkrh.InferenceRequestBody) []string {
	if body == nil {
		return nil
	}
	switch {
	case body.ChatCompletions != nil:
		msgs := body.ChatCompletions.Messages
		if len(msgs) == 0 {
			return nil
		}
		roles := make([]string, len(msgs))
		for i, m := range msgs {
			roles[i] = m.Role
		}
		return roles
	case body.Messages != nil:
		msgs := body.Messages.Messages
		if len(msgs) == 0 {
			return nil
		}
		roles := make([]string, len(msgs))
		for i, m := range msgs {
			roles[i] = m.Role
		}
		return roles
	}
	return nil
}
