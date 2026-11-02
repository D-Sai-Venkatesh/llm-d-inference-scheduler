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

// THE PRODUCTION-SHAPED ARRIVAL FIXTURE.
//
// The tests that measure the ENTRY ORDER need a queue-shaped like production's, and
// the cheap fixture (a hand-written Head per branch, re-stamped every pass) is not
// that shape. It has two fictions that change the order itself:
//
//   - arrivals stamped with the wall clock while the decision clock is the fixture's
//     virtual one, so every head arrived "in the future" and a wait was negative;
//   - the between-cycle enforcement pass run with an EMPTY head list, which reads as
//     "every waiter vanished": the shared queue was deleted and re-created with new
//     stamps, so there was no order to be fair about.
//
// This fixture keeps the shape production has:
//   - a head exists from its ARRIVAL until it is DISPATCHED, then it is gone until the
//     branch's next turn arrives (a seated branch between turns has no head);
//   - an unserved branch keeps its head, unchanged, however long it waits (TTL 0);
//   - the fairness class is the ARRIVAL RECORD branch identity publishes — turn 1 for a
//     branch's first request, n for its n-th — read by the same function the plugin uses
//     (arrivalIsNewcomer), so the class is not restated here;
//   - every instant is the fixture's own clock.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	// promptBytes is the system prompt the fixture gives every branch: at four bytes
	// per token that is ~300 blocks of 64 tokens, the seat the budget arithmetic here
	// uses (three seats inside a 950-block budget).
	promptBytes = 76_700
	// turnTokens is roughly what each further turn appends, so a long residency grows
	// its chain by a couple of blocks and does not perturb the budget shape.
	turnTokens = 8
	// thinkGap is the fixture's inter-turn gap: a seated branch is headless for this
	// long between turns, which is what makes its waiting stamp mean something.
	thinkGap = 200 * time.Millisecond
	// roundStep is one decision pass's time step. It is SHORTER than the think gap on
	// purpose: a served branch stays headless for several passes, so the dispatch pass
	// visits the other engines instead of re-serving the branchiest one forever.
	roundStep = thinkGap / 10
	// prefillTime is when a dispatched turn's first token arrives (the cold charge is
	// released there), and serviceTime when it ends: a turn OCCUPIES its branch, which
	// is what keeps a resident from looking idle between passes.
	prefillTime = 40 * time.Millisecond
	serviceTime = 120 * time.Millisecond
)

// renderTokenIDs is the render backend's tokenizer: one token per FOUR bytes, with the
// token's id derived from its bytes, so a longer prompt EXTENDS the stream (the property
// branch identity's chain relies on).
func renderTokenIDs(s string) []uint32 {
	out := make([]uint32, 0, len(s)/4+1)
	for i := 0; i < len(s); i += 4 {
		end := min(i+4, len(s))
		h := uint32(2166136261)
		for j := i; j < end; j++ {
			h = (h ^ uint32(s[j])) * 16777619
		}
		out = append(out, 1000+h%1_000_000)
	}
	return out
}

// roleTokenID gives each role its own token, above the content range.
func roleTokenID(role string) uint32 {
	switch role {
	case "user":
		return 1001
	case "assistant":
		return 1002
	case "system":
		return 1003
	}
	return 1004
}

// newRenderBackend serves vLLM's /render route for the fixture: it reads the chat
// messages and answers with token ids. Every other route answers the same way, because
// the token producer's warm-up probes three of them.
func newRenderBackend(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var ids []uint32
		for _, m := range body.Messages {
			var text string
			_ = json.Unmarshal(m.Content, &text) // a non-string part renders as nothing
			ids = append(ids, roleTokenID(m.Role))
			ids = append(ids, renderTokenIDs(text)...)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": ids})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newBranchIdentity builds the REAL branch-identity plugin against the fixture's render
// backend. The fixture therefore gets genuine branch ids and genuine arrival records
// rather than a restatement of the rule under test.
func newBranchIdentity(t *testing.T) *branchidentity.Plugin {
	t.Helper()
	srv := newRenderBackend(t)
	params := fmt.Sprintf(`{"blockSizeTokens":64,"tokenProducer":{"modelName":"test-model","vllm":{"url":%q}}}`,
		srv.URL)
	built, err := branchidentity.Factory("branch", json.NewDecoder(strings.NewReader(params)),
		testutils.NewTestHandle(t.Context()))
	if err != nil {
		t.Fatalf("branch identity: %v", err)
	}
	return built.(*branchidentity.Plugin)
}

// queuedTurn is ONE arrival sitting in the queue: its request (branch identity has
// already run on it) and the head the plugin would build from it.
type queuedTurn struct {
	logical string
	req     *fwksched.InferenceRequest
	head    Head
	tokens  []uint32
}

// flyingTurn is a dispatched turn in flight: its prefill completes at chunkAt (the cold
// charge is released there) and the response ends at settleAt.
type flyingTurn struct {
	q        *queuedTurn
	chunkAt  time.Time
	settleAt time.Time
	chunked  bool
}

// workload is the arrival model: when each branch's next turn arrives, and what BI made
// of it when it did.
type workload struct {
	t      *testing.T
	fx     *engineSet
	bi     *branchidentity.Plugin
	due    map[string]time.Time   // when the branch's next turn arrives (zero = none scheduled)
	nxt    map[string]int         // the branch's next turn number
	queued map[string]*queuedTurn // the head waiting in the queue, by logical branch
	byID   map[string]string      // derived branch id -> logical branch
	bid    map[string]string      // logical branch -> derived branch id
	binds  map[string]int         // branch id -> how many times it was SEATED (onBind)
	served map[string]int         // logical branch -> turns the fixture completed
	bytes  map[string]int         // logical branch -> its system prompt's length
	flying map[string]*flyingTurn // logical branch -> its turn in flight
}

func newWorkload(t *testing.T, fx *engineSet) *workload {
	t.Helper()
	w := &workload{
		t: t, fx: fx, bi: newBranchIdentity(t),
		due: map[string]time.Time{}, nxt: map[string]int{},
		queued: map[string]*queuedTurn{}, byID: map[string]string{},
		bid: map[string]string{}, binds: map[string]int{},
		served: map[string]int{}, bytes: map[string]int{},
		flying: map[string]*flyingTurn{},
	}
	// the seat is the only event that says a branch HELD one: a seat taken and given
	// back inside one pass is still a seat, and a test that sampled between passes
	// would miss it
	fx.s.onBind = func(branch, _ string) { w.binds[branch]++ }
	return w
}

// everSeated reports whether the fixture's branch ever held a seat.
func (w *workload) everSeated(logical string) bool {
	return w.binds[w.bid[logical]] > 0
}

// arrive schedules a branch's next turn for instant at.
func (w *workload) arrive(logical string, at time.Time) {
	if _, ok := w.nxt[logical]; !ok {
		w.nxt[logical] = 1
	}
	w.due[logical] = at
}

// setPromptBytes overrides the branch's system prompt length (blocks × 256, the
// four-bytes-per-token convention the estimate and the renderer share).
func (w *workload) setPromptBytes(logical string, bytes int) { w.bytes[logical] = bytes }

// retire drops a branch's queued head and its schedule: from here on it holds its
// seat without ever asking to run again (the closed-engine filler).
func (w *workload) retire(logical string) {
	delete(w.queued, logical)
	delete(w.due, logical)
}

// chatRequest builds turn n of a branch: a fixed system prompt (unique to the branch,
// so two branches never share a chain) plus the turn's own messages.
func chatRequest(logical string, turn, promptLen int) *fwksched.InferenceRequest {
	system := logical + ":" + strings.Repeat(string(rune('a'+len(logical)%26)), promptLen)
	msgs := []any{map[string]any{"role": "system", "content": system}}
	for i := 1; i < turn; i++ {
		msgs = append(msgs,
			map[string]any{"role": "assistant", "content": "ok"},
			map[string]any{"role": "user", "content": fmt.Sprintf("turn %d %s", i, strings.Repeat("x", turnTokens))})
	}
	msgs = append(msgs, map[string]any{"role": "user", "content": fmt.Sprintf("go %d", turn)})
	return &fwksched.InferenceRequest{
		RequestID: fmt.Sprintf("%s-t%d", logical, turn),
		Headers:   map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{},
			Payload:         fwkrh.PayloadMap{"model": "test-model", "messages": msgs},
		},
	}
}

// materialize turns every branch whose turn is due into a queued head, running branch
// identity on it at ARRIVAL time exactly as the request path does.
func (w *workload) materialize(now time.Time) {
	logicals := make([]string, 0, len(w.due))
	for logical := range w.due {
		logicals = append(logicals, logical)
	}
	sort.Strings(logicals)
	for _, logical := range logicals {
		if now.Before(w.due[logical]) {
			continue
		}
		arrival := w.due[logical]
		delete(w.due, logical)
		turn := w.nxt[logical]
		promptLen := promptBytes
		if n, ok := w.bytes[logical]; ok {
			promptLen = n
		}
		req := chatRequest(logical, turn, promptLen)
		if err := w.bi.RequestHeader(context.Background(), req); err != nil {
			w.t.Fatalf("branch identity on %s turn %d: %v", logical, turn, err)
		}
		outcome, _ := fwksched.ReadRequestAttribute[branchidentity.BranchArrival](req, branchidentity.BranchArrivalKey)
		wantOutcome := branchidentity.OutcomeMatched
		if turn == 1 {
			wantOutcome = branchidentity.OutcomeNew
		}
		if outcome.Outcome != wantOutcome || outcome.Turn != turn {
			w.t.Fatalf("%s turn %d: identity said outcome=%v turn=%d, the arrival model is broken",
				logical, turn, outcome.Outcome, outcome.Turn)
		}
		payload, err := req.Body.WirePayload().(fwkrh.PayloadMap).Marshal()
		if err != nil {
			w.t.Fatalf("%s: marshal: %v", logical, err)
		}
		branch := readID(req, branchidentity.BranchIdentityKey)
		w.byID[branch] = logical
		w.bid[logical] = branch
		q := &queuedTurn{
			logical: logical,
			req:     req,
			tokens:  req.Body.TokenizedRequest.Prompts[0].TokenIDs,
			head: Head{
				Branch:     branch,
				ReqID:      req.RequestID,
				Arrival:    arrival,
				SizeBytes:  int64(len(payload)),
				FairnessID: req.FairnessID,
				// the class, from the arrival record the plugin reads (never restated)
				Returner: isReturner(req),
			},
		}
		w.queued[logical] = q
		w.nxt[logical] = turn + 1
	}
}

// heads is the queue-head snapshot as the plugin builds it: one head per branch whose
// turn has arrived and not been dispatched, in a stable order.
func (w *workload) heads(now time.Time) ([]Head, []string) {
	w.materialize(now)
	logicals := make([]string, 0, len(w.queued))
	for logical := range w.queued {
		logicals = append(logicals, logical)
	}
	sort.Strings(logicals)
	hs := make([]Head, 0, len(logicals))
	ids := make([]string, 0, len(logicals))
	for _, logical := range logicals {
		hs = append(hs, w.queued[logical].head)
		ids = append(ids, logical)
	}
	return hs, ids
}

// dispatchTurn puts one queued turn in flight: the exact chain from the request's own
// tokens at PreRequest, then the response timeline (first token, end of stream).
func (w *workload) dispatchTurn(logical string) {
	w.t.Helper()
	q, ok := w.queued[logical]
	if !ok {
		w.t.Fatalf("%s has no queued turn to dispatch", logical)
	}
	delete(w.queued, logical)
	branch := q.head.Branch
	keys := w.fx.s.HashTokens(q.tokens, q.req.Body.TokenizedRequest.CacheSalt, 64)
	w.fx.s.Dispatched(branch, q.req.RequestID, keys, q.head.SizeBytes, w.fx.clock)
	if err := w.bi.PreRequest(context.Background(), q.req, nil); err != nil {
		w.t.Fatalf("branch identity PreRequest: %v", err)
	}
	w.flying[logical] = &flyingTurn{q: q, chunkAt: w.fx.clock.Add(prefillTime), settleAt: w.fx.clock.Add(serviceTime)}
}

// settleDue completes every turn whose timeline reached the fixture's clock: the first
// chunk (prefill done, the cold charge released), then the end of stream, then the
// branch's next turn a think gap later. It runs BEFORE the pass's decision, so a Pick
// sees the state that pass's clock justifies.
func (w *workload) settleDue() {
	now := w.fx.clock
	logicals := make([]string, 0, len(w.flying))
	for logical := range w.flying {
		logicals = append(logicals, logical)
	}
	sort.Strings(logicals)
	for _, logical := range logicals {
		f := w.flying[logical]
		branch := f.q.head.Branch
		if !f.chunked && !now.Before(f.chunkAt) {
			w.fx.s.ChunkAt(branch, now)
			f.chunked = true
		}
		if now.Before(f.settleAt) {
			continue
		}
		w.fx.s.Settle(branch, f.q.req.RequestID,
			struct{ Prompt, Cached int }{len(f.q.tokens), 0}, true, now)
		w.bi.ResponseBody(context.Background(), f.q.req, &fwkrc.Response{EndOfStream: true}, &fwkdl.EndpointMetadata{})
		delete(w.flying, logical)
		w.served[logical]++
		w.arrive(logical, now.Add(thinkGap))
	}
}

// step is one decision pass: settle what is due, then the ~1s ticker. It takes NO head
// list and makes no decision (R5) — the Pick that follows carries the queue view.
func (w *workload) step(d time.Duration) {
	w.settleDue()
	w.fx.advance(d)
}

// advance moves the fixture's clock by d and runs the ~1s ticker. It takes NO head list
// and makes no decision: that is what a tick does (R5).
func (fx *engineSet) advance(d time.Duration) {
	fx.clock = fx.clock.Add(d)
	fx.s.Tick(fx.clock)
}

// seatStatic seats one branch of a chosen size on a NAMED engine and retires it: the
// fixture's way of shaping an engine (an anchor that leaves exactly the room a later
// group needs, or a filler that closes it).
func (w *workload) seatStatic(t *testing.T, logical, engine string, blocks int) {
	t.Helper()
	w.setPromptBytes(logical, blocks*256)
	w.arrive(logical, w.fx.clock.Add(-3*time.Second))
	for round := 0; round < 40 && !w.everSeated(logical); round++ {
		hs, ids := w.heads(w.fx.clock)
		if i := w.fx.pick(hs...); i >= 0 {
			w.dispatchTurn(ids[i])
		}
		w.step(roundStep)
	}
	if !w.everSeated(logical) {
		t.Fatalf("%s (%d blocks) could not be seated", logical, blocks)
	}
	if got := w.fx.s.bindings[w.bid[logical]]; got != engine {
		t.Fatalf("%s holds %q, wanted %q", logical, got, engine)
	}
	w.retire(logical)
}

// TestIsReturnerFromTheArrivalRecord pins the fairness class's SOURCE: branch
// identity's arrival record, whose turn number is 1 for a branch's first request and n
// for its n-th (branchidentity's own tests pin the number; this pins the mapping). A
// request with no record is a NEWCOMER — a failure to classify must never demote a
// branch behind incumbents.
func TestIsReturnerFromTheArrivalRecord(t *testing.T) {
	withRecord := func(turn int) *fwksched.InferenceRequest {
		req := &fwksched.InferenceRequest{RequestID: "r", Headers: map[string]string{}}
		req.PutAttribute(branchidentity.BranchArrivalKey, branchidentity.BranchArrival{
			Outcome: branchidentity.OutcomeMatched, Turn: turn})
		return req
	}
	if isReturner(withRecord(1)) {
		t.Fatal("turn 1 is a branch's first request: a newcomer, not a returner")
	}
	for _, turn := range []int{2, 3, 17} {
		if !isReturner(withRecord(turn)) {
			t.Fatalf("turn %d is a continuation: a returner", turn)
		}
	}
	if isReturner(&fwksched.InferenceRequest{RequestID: "r"}) {
		t.Fatal("no arrival record must default to NEWCOMER (the safe direction)")
	}
	if isReturner(nil) {
		t.Fatal("no request is a newcomer, not a returner")
	}
	// and the head the plugin builds carries it: a Head built without the field is a
	// newcomer by construction, which is the property the fixtures rely on
	var h Head
	if h.Returner {
		t.Fatal("the zero Head must be a newcomer")
	}
}
