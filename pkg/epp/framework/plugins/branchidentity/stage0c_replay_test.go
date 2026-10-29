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

// STAGE 0C — replay a campaign window through the SHIPPED plugin.
//
// The stage 0 / 0B validations scored a Python re-implementation. This drives the
// real thing: each request of a recorded window is parsed by the PRODUCTION parser
// (parsers/openai — the same InferenceRequestBody the EPP builds) and pushed
// through the plugin's own hooks at the recorded arrival / dispatch / settle
// timestamps, with the clock injected (p.now) and the session attribute fed from
// the harness's root id. The per-request derived ids are written out for scoring
// against ground truth (stage0c_score.py).
//
// It is skipped unless STAGE0C_WINDOW points at a prepared window:
//
//	STAGE0C_WINDOW=/tmp/stage0c/a0.jsonl STAGE0C_OUT=/tmp/stage0c/a0.go.jsonl \
//	  go test ./pkg/epp/framework/plugins/branchidentity/ -run TestStage0CReplay -v -timeout 3h
//
// Env:
//
//	STAGE0C_UNSCOPED=1   send no session header (the no-session-header floor, and
//	                     the S4 "unscoped" variant: what a corpus of look-alike
//	                     sessions costs when nothing scopes the index)
//	STAGE0C_RETRY_RULE=0 configure the arm with retryRule=false (the campaign
//	                     setting: both a0 firings of S2 turned out to be
//	                     CROSS-thread, i.e. no client retries in the corpus at all)
//	STAGE0C_SCOPE=thread scope the index by the request's own thread id
//	                     (x_correlation_id) instead of its session root — the shape
//	                     the shadow arm has, where the only session header the client
//	                     sends IS the thread id. Root is the stage-0c default.
//
// Nothing here touches the cluster, a GPU, or the network: it is a local replay of
// recorded bodies.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"
	"sort"
	"testing"
	"time"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"

	testutils "github.com/llm-d/llm-d-router/test/utils"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// stage0cScopeHeader is the header the replay uses as the client session id. It is
// NOT one of the built-in provider headers, so the plugin is configured to read it
// explicitly and no real-session semantics are assumed.
const stage0cScopeHeader = "x-session-scope"

// stage0cLine is one prepared request (stage0c_prep.py).
type stage0cLine struct {
	I       int             `json:"i"`
	Req     string          `json:"req"`
	Conv    string          `json:"conv"`
	Turn    int             `json:"turn"`
	GT      string          `json:"gt"`
	GTRoot  string          `json:"gt_root"`
	Kind    string          `json:"kind"`
	Reset   bool            `json:"reset"`
	StartNs int64           `json:"start_ns"`
	AckNs   int64           `json:"ack_ns"`
	EndNs   int64           `json:"end_ns"`
	InTok   int             `json:"in_tokens"`
	Body    json.RawMessage `json:"body"`
	Off     int64           `json:"-"`
	Len     int             `json:"-"`
}

// stage0cResult is one request's outcome, keyed by the window index.
//
// Outcome/Retry/Sticky come from the plugin's own counters, diffed around the one
// RequestHeader call, so a scoring pass can name WHICH requests a rule fired on
// (the S2 retry total alone says how many, never which).
type stage0cResult struct {
	I       int    `json:"i"`
	Req     string `json:"req"`
	ID      string `json:"id"`
	Parse   bool   `json:"parseError"`
	Outcome string `json:"outcome,omitempty"`
	Retry   bool   `json:"retry,omitempty"`
	Sticky  bool   `json:"sticky,omitempty"`
}

func TestStage0CReplay(t *testing.T) {
	window := os.Getenv("STAGE0C_WINDOW")
	if window == "" {
		t.Skip("STAGE0C_WINDOW not set: stage-0c replay is an offline validation harness")
	}
	out := os.Getenv("STAGE0C_OUT")
	if out == "" {
		out = window + ".go.jsonl"
	}
	unscoped := os.Getenv("STAGE0C_UNSCOPED") == "1"

	ctx := context.Background()
	f, err := os.Open(window)
	if err != nil {
		t.Fatalf("open window: %v", err)
	}
	defer f.Close()

	// Pass 1 — metadata + byte offsets only, so the window never has to fit in
	// memory (a WEKA window is ~850 MB of bodies).
	start := time.Now()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	var (
		lines  []*stage0cLine
		offset int64
	)
	for sc.Scan() {
		raw := sc.Bytes()
		l := &stage0cLine{Off: offset, Len: len(raw)}
		if err := json.Unmarshal(raw, l); err != nil {
			t.Fatalf("bad window line at offset %d: %v", offset, err)
		}
		lines = append(lines, l)
		offset += int64(len(raw)) + 1
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan window: %v", err)
	}
	t.Logf("[stage0c] window %s: %d requests indexed in %s (unscoped=%v)",
		window, len(lines), time.Since(start).Round(time.Millisecond), unscoped)

	// Events in wall-clock order: a settle precedes an arrival at the same instant
	// (the request that ended there must free its tip before the next one looks).
	const (
		evSettle = iota
		evArrive
		evDispatch
	)
	type event struct {
		ts   int64
		kind int
		idx  int
	}
	events := make([]event, 0, 3*len(lines))
	for idx, l := range lines {
		events = append(events,
			event{l.StartNs, evArrive, idx},
			event{l.AckNs, evDispatch, idx},
			event{l.EndNs, evSettle, idx})
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].ts != events[j].ts {
			return events[i].ts < events[j].ts
		}
		if events[i].kind != events[j].kind {
			return events[i].kind < events[j].kind
		}
		return events[i].idx < events[j].idx
	})

	// The plugin, configured as the stage-1 arm will be, except that the session
	// header is the replay's own scope header. STAGE0C_RETRY_RULE=0 reproduces the
	// CAMPAIGN setting (retryRule off — see the C3 labelling of the two a0 firings).
	handle := testutils.NewTestHandle(ctx)
	retryRule := "true"
	if os.Getenv("STAGE0C_RETRY_RULE") == "0" {
		retryRule = "false"
	}
	built, err := Factory("stage0c", json.NewDecoder(bytes.NewReader(
		[]byte(`{"sessionHeaders":["`+stage0cScopeHeader+`"],"retryRule":`+retryRule+`}`))), handle)
	if err != nil {
		t.Fatalf("build plugin: %v", err)
	}
	p := built.(*Plugin)
	var clock time.Time
	p.now = func() time.Time { return clock }

	parser := openai.NewOpenAIParser()
	reqs := make([]*fwksched.InferenceRequest, len(lines))
	results := make([]stage0cResult, len(lines))
	for i, l := range lines {
		results[i] = stage0cResult{I: l.I, Req: l.Req}
	}

	var parseErrors, noID int
	for _, ev := range events {
		l := lines[ev.idx]
		clock = time.Unix(0, ev.ts)

		switch ev.kind {
		case evArrive:
			// Read this request's body and parse it the way the EPP does.
			buf := make([]byte, l.Len)
			if _, err := f.ReadAt(buf, l.Off); err != nil {
				t.Fatalf("read body %d: %v", l.I, err)
			}
			var wire struct {
				Body json.RawMessage `json:"body"`
			}
			if err := json.Unmarshal(buf, &wire); err != nil {
				t.Fatalf("decode line %d: %v", l.I, err)
			}
			headers := map[string]string{":path": "/v1/chat/completions"}
			if !unscoped {
				if os.Getenv("STAGE0C_SCOPE") == "thread" {
					headers[stage0cScopeHeader] = l.GT
				} else {
					headers[stage0cScopeHeader] = l.GTRoot
				}
			}
			req := &fwksched.InferenceRequest{RequestID: l.Req, Headers: headers}
			parsed, err := parser.ParseRequest(ctx, wire.Body, headers)
			if err != nil {
				parseErrors++
				results[ev.idx].Parse = true
			} else {
				req.Body = parsed.Body
			}
			reqs[ev.idx] = req
			matched0, forked0, new0 := counterValue(p.metrics.matched), counterValue(p.metrics.forked), counterValue(p.metrics.fresh)
			sticky0, retry0 := counterValue(p.metrics.sticky), counterValue(p.metrics.retried)
			if err := p.RequestHeader(ctx, req); err != nil {
				t.Fatalf("RequestHeader %d: %v", l.I, err)
			}
			switch {
			case counterValue(p.metrics.matched) > matched0:
				results[ev.idx].Outcome = "matched"
			case counterValue(p.metrics.forked) > forked0:
				results[ev.idx].Outcome = "forked"
			case counterValue(p.metrics.fresh) > new0:
				results[ev.idx].Outcome = "new"
			default:
				results[ev.idx].Outcome = "unflattened"
			}
			results[ev.idx].Retry = counterValue(p.metrics.retried) > retry0
			results[ev.idx].Sticky = counterValue(p.metrics.sticky) > sticky0
			id, _ := fwksched.ReadRequestAttribute[string](req, BranchIdentityKey)
			if id == "" {
				noID++
			}
			results[ev.idx].ID = id

		case evDispatch:
			if req := reqs[ev.idx]; req != nil {
				if err := p.PreRequest(ctx, req, nil); err != nil {
					t.Fatalf("PreRequest %d: %v", l.I, err)
				}
			}
		case evSettle:
			if req := reqs[ev.idx]; req != nil {
				p.ResponseBody(ctx, req, &fwkrc.Response{
					EndOfStream:      true,
					TerminationCause: fwkrc.TerminationCauseNatural,
				}, nil)
				// Keep memory flat on a 13k-request window.
				req.Body = nil
				reqs[ev.idx] = nil
			}
		}
	}

	t.Logf("[stage0c] %d requests, %d branches, %d parse errors, %d no-id",
		len(lines), p.idx.Len(), parseErrors, noID)
	t.Logf("[stage0c] counters: matched=%v forked=%v new=%v sticky=%v retry=%v overridden=%v",
		counterValue(p.metrics.matched), counterValue(p.metrics.forked),
		counterValue(p.metrics.fresh), counterValue(p.metrics.sticky),
		counterValue(p.metrics.retried), counterValue(p.metrics.overridden))

	w, err := os.Create(out)
	if err != nil {
		t.Fatalf("create output: %v", err)
	}
	defer w.Close()
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i := range results {
		if err := enc.Encode(results[i]); err != nil {
			t.Fatalf("write result: %v", err)
		}
	}
	if err := bw.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	t.Logf("[stage0c] wrote %s", out)
}

// counterValue reads one counter (nil-safe: a plugin built without a handle has
// unregistered metrics).
func counterValue(c prometheus.Counter) float64 {
	if c == nil {
		return -1
	}
	return testutil.ToFloat64(c)
}
