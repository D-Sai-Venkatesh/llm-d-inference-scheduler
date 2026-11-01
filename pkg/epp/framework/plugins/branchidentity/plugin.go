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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// PluginType is the canonical type name used to register the plugin.
const PluginType = "branch-identity"

// BranchIdentityKey is the request-attribute key carrying the DERIVED branch id.
// The director uses it as the fairness id (the flow-control queue key) when the
// request carried no explicit fairness header.
var BranchIdentityKey = fwkplugin.NewDataKey("branch-identity", "")

// BranchSessionKey is the request-attribute key carrying the client's session id,
// when it sent one. It is deliberately SEPARATE from the branch id: the session is
// an attribute for the session fairness layer, the branch is the queue key.
var BranchSessionKey = fwkplugin.NewDataKey("branch-session", "")

// BranchArrivalKey is the request-attribute key carrying the ARRIVAL RECORD: which
// of the three outcomes this request was (continued / forked / new) and its turn
// number on its branch. It is a pure OUTPUT — the scheduler reads it to tell a
// branch's first turn from a continuation (a newcomer from an incumbent); nothing
// here reads it back, so publishing it cannot change the derivation.
var BranchArrivalKey = fwkplugin.NewDataKey("branch-arrival", "")

// BranchArrival is the published arrival record.
type BranchArrival struct {
	// Outcome is the arrival outcome: OutcomeMatched (continued its branch),
	// OutcomeForked (started a sibling of a busy tip) or OutcomeNew.
	Outcome Outcome `json:"outcome"`
	// Turn is the arrival's 1-based turn number on its branch: 1 for a branch's
	// first request, n for the n-th continuation.
	Turn int `json:"turn"`
}

// branchMarkerKey is the request-attribute key carrying the request's marker —
// the handle all three hooks use to move this request between its branch's
// pending and in-flight sets. It equals the request id when the client supplied
// one and a per-request synthetic id otherwise.
var branchMarkerKey = fwkplugin.NewDataKey("branch-marker", "")

const (
	// defaultBlockSizeTokens is the identity block length — the prefix scorer's
	// B, one shared knob, not an identity-specific size.
	defaultBlockSizeTokens = 64
	// defaultMarkerTTLSeconds bounds how long an outstanding marker keeps its
	// branch busy when the request never reports an end of stream (a request
	// rejected in admission control never gets one). It is the CLIENT's request
	// timeout, not the flow queue's 5-minute idle GC: 16% of a0's requests waited
	// over 600s for admission, so a 600s horizon would expire markers of requests
	// that are still legitimately queued. Pruning errs toward continuation.
	defaultMarkerTTLSeconds = 3600
	// defaultSweepSeconds is the cap-enforcement tick.
	defaultSweepSeconds = 60
	// defaultMaxBranches is the table cap (the only eviction path).
	defaultMaxBranches = 100000
	// defaultRenderTimeoutSeconds bounds the ARRIVAL-time render. Identity carries
	// its own bound rather than inheriting the producer's: the producer's timeout
	// protects a dispatch phase that may fall back to a default, while a request
	// must not enter the fairness path without an identity for longer than this.
	defaultRenderTimeoutSeconds = 10
)

// defaultSessionHeaders are the client session headers read for scoping, in
// priority order: the provider headers the agent-identity plugin resolves, so the
// session attribute is the session the rest of the system already sees.
var defaultSessionHeaders = []string{
	agentidentity.ClaudeCodeSessionHeader,
	agentidentity.OpenCodeSessionHeader,
	agentidentity.CodexSessionHeader,
	agentidentity.CodexSessionHeaderLegacy,
}

// Parameters is the plugin's configuration block.
type Parameters struct {
	// BlockSizeTokens is the identity block length in tokens — the prefix
	// scorer's block size. Zero selects the default (64).
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// TokenProducer configures the render backend that supplies the request's
	// real tokens, in the token-producer plugin's own parameter language. The
	// identity is a token-space key, so an arm wants the render backend here; the
	// estimate backend is accepted (tests, and the byte-space fallback) and is
	// announced on startup.
	TokenProducer json.RawMessage `json:"tokenProducer,omitempty"`
	// RenderTimeoutSeconds bounds the arrival-time render. Zero selects the
	// default (10s).
	RenderTimeoutSeconds float64 `json:"renderTimeoutSeconds,omitempty"`
	// MarkerTTLSeconds bounds how long an outstanding marker (arrival or in
	// flight) that never settles keeps its branch busy. It must be at least the
	// client's request timeout. There is no branch TTL: branches are retained (no
	// idle GC — a dormant client that returns must find its branch).
	MarkerTTLSeconds float64 `json:"markerTtlSeconds,omitempty"`
	// QueuedTTLSeconds is the deprecated name of MarkerTTLSeconds, accepted so an
	// older arm's config still parses.
	QueuedTTLSeconds float64 `json:"queuedTtlSeconds,omitempty"`
	// SweepSeconds is the cap-enforcement tick.
	SweepSeconds float64 `json:"sweepSeconds,omitempty"`
	// MaxBranches caps the branch table: the only eviction path, idlest
	// non-outstanding branches first.
	MaxBranches int `json:"maxBranches,omitempty"`
	// SessionHeaders are client session headers prepended to the built-in list.
	SessionHeaders []string `json:"sessionHeaders,omitempty"`
	// ForkOnBusy is the continuation rule's busy half: extend a busy tip → fork a
	// new branch. Default true (the design's rule). False makes every match a
	// continuation — a measurement arm only.
	ForkOnBusy *bool `json:"forkOnBusy,omitempty"`
	// StickyBusyGuard is H1: a request that extends a tip AND begins with the
	// tip's assistant reply is a proven continuation even when the branch looks
	// busy. Default true.
	StickyBusyGuard *bool `json:"stickyBusyGuard,omitempty"`
	// RetryRule is S2: a request identical to a tip whose branch has nothing in
	// flight (only arrival markers) is a client retry, not a sibling. Default true.
	RetryRule *bool `json:"retryRule,omitempty"`
	// LogRequests emits one JSON line per request on stdout:
	//
	//	branch-identity: request {"seq":…,"scope":…,"derived":…,"outcome":…}
	//
	// Default FALSE — it is an INSTRUMENT for scoring a shadow arm whose derived
	// id reaches no span. It changes no state and no decision.
	LogRequests bool `json:"logRequests,omitempty"`
	// LogHeaders adds the client's own identity headers to that line, for an
	// UNSCOPED arm where no session header is sent and the census cannot
	// attribute a branch to a client thread.
	LogHeaders bool `json:"logHeaders,omitempty"`
	// LogHeaderNames extends the logHeaders whitelist. A pure observer: no header
	// here is read by the identity, the scope or any decision.
	LogHeaderNames []string `json:"logHeaderNames,omitempty"`
}

// loggedHeaderNames is the built-in logHeaders whitelist: the client's
// correlation id (a harness's thread ground truth) and its request id.
var loggedHeaderNames = []string{"x-correlation-id", "x-request-id"}

// loggedHeaderList returns the effective whitelist: built-ins plus extras,
// lowercased and de-duplicated (request headers are stored lowercased).
func loggedHeaderList(extra []string) []string {
	out := append([]string(nil), loggedHeaderNames...)
	seen := make(map[string]struct{}, len(out)+len(extra))
	for _, n := range out {
		seen[n] = struct{}{}
	}
	for _, n := range extra {
		n = strings.ToLower(strings.TrimSpace(n))
		if n == "" {
			continue
		}
		if _, dup := seen[n]; dup {
			continue
		}
		seen[n] = struct{}{}
		out = append(out, n)
	}
	return out
}

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
	if p.RenderTimeoutSeconds == 0 {
		p.RenderTimeoutSeconds = defaultRenderTimeoutSeconds
	}
	if p.RenderTimeoutSeconds < 0 {
		return fmt.Errorf("renderTimeoutSeconds must be > 0, got %v", p.RenderTimeoutSeconds)
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
	_ fwkdl.Registrant             = &Plugin{}
)

// Plugin derives branch identity from the request body and owns the branch table.
// One instance serves every hook: the busy/tip state is a single mutex-guarded
// struct rather than a cross-plugin channel.
type Plugin struct {
	typedName fwkplugin.TypedName

	idx     *branchIndex
	metrics *branchMetrics
	// producer renders the request's real tokens (the token producer, as a
	// library); tokenize is the call site, so a test can substitute a
	// deterministic renderer.
	producer      *tokenizer.Producer
	tokenize      func(ctx context.Context, request *fwksched.InferenceRequest) error
	blockSize     int
	renderTimeout time.Duration

	markerTTL      time.Duration
	sweep          time.Duration
	forkOnBusy     bool
	sticky         bool
	retry          bool
	logRequests    bool
	logHeaders     bool
	logHeadersList []string
	sessions       []string
	// now is the clock seam: the stage-0c replay drives recorded arrival /
	// dispatch / settle times through the real hooks.
	now func() time.Time

	anonSeq atomic.Uint64
	// logSeq orders the LogRequests instrument's lines in arrival order.
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

	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	typedName := fwkplugin.TypedName{Type: PluginType, Name: name}
	producer, err := tokenizer.NewProducer(ctx, typedName, params.TokenProducer)
	if err != nil {
		return nil, fmt.Errorf("failed to build the token producer of the '%s' plugin: %w", PluginType, err)
	}

	p := &Plugin{
		typedName:      typedName,
		producer:       producer,
		tokenize:       producer.Tokenize,
		blockSize:      params.BlockSizeTokens,
		renderTimeout:  time.Duration(params.RenderTimeoutSeconds * float64(time.Second)),
		markerTTL:      time.Duration(params.MarkerTTLSeconds * float64(time.Second)),
		sweep:          time.Duration(params.SweepSeconds * float64(time.Second)),
		forkOnBusy:     params.ForkOnBusy == nil || *params.ForkOnBusy,
		sticky:         params.StickyBusyGuard == nil || *params.StickyBusyGuard,
		retry:          params.RetryRule == nil || *params.RetryRule,
		logRequests:    params.LogRequests,
		logHeaders:     params.LogRequests && params.LogHeaders,
		logHeadersList: loggedHeaderList(params.LogHeaderNames),
		sessions:       sessionHeaders(params.SessionHeaders),
		now:            time.Now,
	}
	p.idx = newBranchIndex(params.BlockSizeTokens, p.markerTTL, params.MaxBranches)
	p.metrics = newBranchMetrics()

	// The effective config is echoed at startup so a run's arm is verifiable from
	// its artifacts.
	if echo, err := json.Marshal(params); err == nil {
		fmt.Printf("%s: effective config %s\n", PluginType, echo)
	}
	if producer.BackendName() != tokenizer.BackendVLLM {
		fmt.Printf("%s: WARNING token producer backend is %q — identity is a token-space key, so an arm must render (modelName/vllm)\n",
			PluginType, producer.BackendName())
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
		BranchArrivalKey:  BranchArrival{},
		branchMarkerKey:   "",
	}
}

// RegisterDependencies forwards the render backend's endpoint discovery, so a
// discovery-backed renderer sees the endpoint lifecycle like the token-producer
// plugin's does.
func (p *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	return p.producer.RegisterDependencies(r)
}

// RequestHeader derives the branch id before admission control and makes it the
// request's fairness id — the flow-control queue key must exist before the
// request enqueues, and this hook runs after body parsing but before Admit.
//
// It renders the request's tokens HERE, on arrival, so the dispatch-time token
// producer finds them already present and skips: one render per request, shared
// by identity and every prefix scorer.
//
// It also publishes the branch, the session (when the client sent one) and the
// arrival record. It never fails the request: a request the render cannot
// describe gets its own branch (counted) and the request proceeds.
func (p *Plugin) RequestHeader(ctx context.Context, request *fwksched.InferenceRequest) error {
	if request == nil || request.Body == nil {
		return nil
	}
	// An explicit x-llm-d-inference-fairness-id header wins (operators keep a
	// manual override). It is counted, because a harness that sets it turns the
	// whole derivation into a no-op with no other visible signal.
	explicit := request.FairnessID != ""

	c, ok := p.chainFor(ctx, request)
	roles := messageRoles(request.Body)
	scope := p.sessionID(request)
	marker := p.markerID(request)

	res := p.idx.Resolve(p.now(), Arrival{Chain: c, Scope: scope, Roles: roles, Marker: marker},
		p.forkOnBusy, p.sticky, p.retry)
	p.metrics.observeResolve(res)
	if !ok {
		// A render failure leaves the request without an identity: it gets its own
		// branch (never a continuation) and the counter names the failure.
		p.metrics.observeRenderFailure()
	}

	request.PutAttribute(BranchIdentityKey, res.ID)
	request.PutAttribute(BranchArrivalKey, BranchArrival{Outcome: res.Outcome, Turn: res.Turn})
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
	p.logRequest(request, res, scope, explicit, c)
	return nil
}

// chainFor renders the request's tokens and hashes them into its identity chain.
// ok is false when the request has no identity: a render error or timeout, or a
// body that renders to no tokens.
func (p *Plugin) chainFor(ctx context.Context, request *fwksched.InferenceRequest) (chain, bool) {
	renderCtx, cancel := context.WithTimeout(ctx, p.renderTimeout)
	defer cancel()

	start := p.now()
	err := p.tokenize(renderCtx, request)
	p.metrics.observeRender(p.now().Sub(start))
	if err != nil {
		return chain{}, false
	}

	start = p.now()
	c := identityChain(renderCtx, request, p.blockSize)
	p.metrics.observeChain(p.now().Sub(start))
	if len(c.blocks) == 0 {
		return chain{}, false
	}
	if prompts := tokenizedPrompts(request); prompts > 1 {
		p.metrics.addMultiPrompt(prompts)
	}
	return c, true
}

// identityChain hashes a tokenized request into its identity chain, in the prefix
// scorer's own block-hash space (model + cache salt seeded, chained). ALL prompts
// count: their chains are concatenated in order.
func identityChain(ctx context.Context, request *fwksched.InferenceRequest, blockSize int) chain {
	hashes, promptTokens := prefixhash.GetBlockHashesWithPromptTokens(ctx, request, blockSize, maxChainBlocks)
	if len(hashes) == 0 {
		return chain{}
	}
	c := chain{blocks: make([]prefixhash.BlockHash, 0, len(hashes))}
	for _, prompt := range hashes {
		c.blocks = append(c.blocks, prompt...)
	}
	for _, n := range promptTokens {
		c.tokens += n
	}
	c.fullBlocks = len(c.blocks)
	if last := promptTokens[len(promptTokens)-1]; last%blockSize != 0 {
		// The last prompt ends mid-block: that block is the tip's partial block.
		c.fullBlocks--
	}
	return c
}

// tokenizedPrompts is the number of prompts the render produced.
func tokenizedPrompts(request *fwksched.InferenceRequest) int {
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return 0
	}
	return len(request.Body.TokenizedRequest.Prompts)
}

// logRequest emits the per-request instrument line when Parameters.LogRequests is
// on (off by default): one write per line, in arrival order, carrying the client
// scope, the derived branch, the outcome and the identity size.
func (p *Plugin) logRequest(request *fwksched.InferenceRequest, res ResolveResult, scope string, explicit bool, c chain) {
	if !p.logRequests {
		return
	}
	line, err := json.Marshal(struct {
		Seq      uint64            `json:"seq"`
		TS       string            `json:"ts"`
		Scope    string            `json:"scope"`
		Derived  string            `json:"derived"`
		Outcome  string            `json:"outcome"`
		Turn     int               `json:"turn"`
		Retry    bool              `json:"retry,omitempty"`
		Sticky   bool              `json:"sticky,omitempty"`
		Explicit bool              `json:"explicit"`
		Blocks   int               `json:"blocks"`
		Tokens   int               `json:"tokens"`
		Marker   string            `json:"marker"`
		Headers  map[string]string `json:"headers,omitempty"`
	}{
		Seq: p.logSeq.Add(1), TS: p.now().UTC().Format(time.RFC3339Nano), Scope: scope,
		Derived: res.ID, Outcome: string(res.Outcome), Turn: res.Turn, Retry: res.Retry, Sticky: res.StickyBusy,
		Explicit: explicit, Blocks: c.fullBlocks, Tokens: c.tokens, Marker: markerOf(request),
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
	out := make(map[string]string, len(p.logHeadersList))
	for _, name := range p.logHeadersList {
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

// PreRequest records the DISPATCH transition, moving this request's marker from
// its branch's pending set to its in-flight set. The branch is already busy from
// ARRIVAL, so nothing about identity depends on this hook; it names WHICH requests
// are running.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if request == nil {
		return nil
	}
	if id, ok := fwksched.ReadRequestAttribute[string](request, BranchIdentityKey); ok && id != "" {
		p.idx.Dispatch(id, markerOf(request), p.now())
	}
	return nil
}

// ResponseBody settles the branch at the end of the response lifecycle. The
// director forces this call on abort/disconnect for every dispatched request, so
// the busy flag cannot wedge.
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
	snap := p.idx.Snapshot(time.Now())
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

// sweepOnce enforces the table cap and refreshes the records gauge. There is no
// idle GC (see branchIndex.EnforceCap): a dormant client that returns must find
// its branch, or it is minted a fresh id and re-enters as a newcomer.
func (p *Plugin) sweepOnce(now time.Time) {
	if removed := p.idx.EnforceCap(now); removed > 0 {
		p.metrics.addEvicted(removed)
	}
	p.metrics.setRecords(p.idx.Len())
}

// markerID returns the request's marker handle for the branch's outstanding set:
// its request id when the client or framework supplied one, a per-request
// synthetic id otherwise.
func (p *Plugin) markerID(request *fwksched.InferenceRequest) string {
	if id := strings.TrimSpace(request.RequestID); id != "" {
		return id
	}
	return fmt.Sprintf("anon-%d", p.anonSeq.Add(1))
}

// markerOf reads the marker an earlier hook published, falling back to the
// request id so the operations stay no-ops rather than corrupting another
// request's marker.
func markerOf(request *fwksched.InferenceRequest) string {
	if marker, ok := fwksched.ReadRequestAttribute[string](request, branchMarkerKey); ok && marker != "" {
		return marker
	}
	return request.RequestID
}

// sessionHeaders returns the operator's extras followed by the defaults,
// lower-cased (request headers are stored lowercased).
func sessionHeaders(extras []string) []string {
	out := make([]string, 0, len(extras)+len(defaultSessionHeaders))
	for _, h := range extras {
		out = append(out, strings.ToLower(h))
	}
	return append(out, defaultSessionHeaders...)
}

// messageRoles returns the request's message roles in wire order. Body shapes
// without a message array return nil: H1 needs message structure and cannot fire
// for them.
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
