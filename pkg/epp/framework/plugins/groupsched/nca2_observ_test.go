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

// THE OBSERVABILITY (nca2 spec §7): the nca2 series exist ONLY in nca2, the snapshot section is nil in
// OFF, a scrape never changes a decision (I9), and the span carries the client's own request id plus
// the class the primary metric's population is built from.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// seriesNames is every metric name a registry currently holds.
func seriesNames(t *testing.T, reg prometheus.Gatherer) map[string]struct{} {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	out := map[string]struct{}{}
	for _, f := range families {
		out[f.GetName()] = struct{}{}
	}
	return out
}

// nca2Series is the full expected nca2 surface: names AND labels are what a scraper stores, so the
// list is checked by name and the label sets are exercised by the gauge test below.
var nca2Series = []string{
	"llm_d_epp_nca2_admissions_total",
	"llm_d_epp_nca2_returner_seats_total",
	"llm_d_epp_nca2_decisions_total",
	"llm_d_epp_nca2_pushouts_total",
	"llm_d_epp_nca2_active_pushouts_total",
	"llm_d_epp_nca2_moves_total",
	"llm_d_epp_nca2_line_touch_tokens_total",
	"llm_d_epp_nca2_leaked_requests_total",
	"llm_d_epp_nca2_door_wait_seconds",
	"llm_d_epp_nca2_seat_wait_seconds",
	"llm_d_epp_nca2_room_tokens",
	"llm_d_epp_nca2_laa_position_tokens",
	"llm_d_epp_nca2_line_tokens",
	"llm_d_epp_nca2_admitted_tokens",
	"llm_d_epp_nca2_admitted_active_tokens",
	"llm_d_epp_nca2_admitted_threads",
	"llm_d_epp_nca2_active_threads",
	"llm_d_epp_nca2_warm_returners_waiting",
	"llm_d_epp_nca2_pending_heads",
	"llm_d_epp_nca2_seats_inflight",
	"llm_d_epp_nca2_seats_reclaimable",
	"llm_d_epp_nca2_idle_with_waiters",
	"llm_d_epp_nca2_priority_waiting",
	"llm_d_epp_nca2_newcomer_waiting",
	"llm_d_epp_nca2_head_blocked_seconds",
}

// TestNCA2SeriesAreRegisteredOnlyInNCA2: OFF's metric surface must not gain a single series.
func TestNCA2SeriesAreRegisteredOnlyInNCA2(t *testing.T) {
	build := func(t *testing.T, params string) (fwkplugin.Handle, map[string]struct{}) {
		handle := testutils.NewTestHandle(t.Context())
		built, err := Factory("group", json.NewDecoder(strings.NewReader(params)), handle)
		require.NoError(t, err)
		// ONE measured pod: the per-pod zero series and the gauges exist from its first scrape, exactly
		// as they do in a run (nothing is registered per pod at startup)
		require.NoError(t, built.(*GroupScheduler).sched.SetNumBlocksFor("p0", 64, 1000))
		return handle, seriesNames(t, handle.Metrics().(prometheus.Gatherer))
	}
	_, off := build(t, `{}`)
	for _, name := range nca2Series {
		if _, has := off[name]; has {
			t.Fatalf("OFF must not register %q", name)
		}
	}
	if _, has := off["llm_d_epp_group_dispatches_total"]; !has {
		t.Fatal("OFF must keep the shipped series")
	}

	_, on := build(t, `{"mode":"nca2","cpuTierTokens":5592405}`)
	for _, name := range nca2Series {
		if _, has := on[name]; !has {
			t.Fatalf("nca2 must register %q", name)
		}
	}
	if _, has := on["llm_d_epp_group_exits_total"]; !has {
		t.Fatal("nca2 keeps the shipped series (the exit reasons and the cold valve are scraped)")
	}
}

// TestSnapshotNCA2Section: the per-pod and global numbers a dashboard reads, from a state whose
// arithmetic is exact.
func TestSnapshotNCA2Section(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	f.turn("warm", 10) // admitted, 640 tokens at the head, then inactive
	// an outsider waits, and the pod carries the line
	f.s.mu.Lock()
	f.s.nca2.ever["outsider"] = true // a PRIORITY outsider
	f.s.waiters["outsider"] = &waiter{id: "outsider", since: f.now}
	f.s.waiters["newcomer"] = &waiter{id: "newcomer", since: f.now}
	snap := f.s.snapshot(f.now).NCA2 // the lock-held form: we hold s.mu here
	f.s.mu.Unlock()
	if snap == nil {
		t.Fatal("nca2 must report its own section")
	}
	if snap.PriorityWaiting != 1 || snap.NewcomerWaiting != 1 {
		t.Fatalf("the outside lists: %d priority, %d newcomer", snap.PriorityWaiting, snap.NewcomerWaiting)
	}
	p := snap.Engines["p0"]
	if p.LineTokens != 640 || p.AdmittedTokens != 640 || p.AdmittedThreads != 1 {
		t.Fatalf("the line: %+v", p)
	}
	if p.RoomTokens != lineAdmit {
		t.Fatalf("an INACTIVE admitted entry is space: room %d, want %d", p.RoomTokens, lineAdmit)
	}
	if p.LAA != "" || p.LAAPositionTokens != 0 {
		t.Fatalf("no admitted ACTIVE entry: no LAA (%q at %d)", p.LAA, p.LAAPositionTokens)
	}
	if p.SeatsInflight != 0 || p.ActiveThreads != 0 {
		t.Fatalf("nothing is running: %+v", p)
	}
	if p.IdleWithWaiters != 1 {
		t.Fatal("an idle pod with an outside head waiting is the state the gauge exists for")
	}
	// the JSON is the DumpState shape: a nil section in OFF, an object here
	raw, err := json.Marshal(f.s.Snapshot(f.now))
	require.NoError(t, err)
	require.Contains(t, string(raw), `"nca2":{`)
	off, err := NewScheduler(defaultConfig())
	require.NoError(t, err)
	raw, err = json.Marshal(off.Snapshot(f.now))
	require.NoError(t, err)
	require.NotContains(t, string(raw), `"nca2"`, "OFF's snapshot must not gain a section")
}

// TestTheGaugesReadTheSnapshot: a scrape emits the snapshot's numbers, one row per pod, and measures
// nothing (a second scrape is identical).
func TestTheGaugesReadTheSnapshot(t *testing.T) {
	f := newFixture(t, "p0")
	f.turn("warm", 10)
	gauges := newNCA2Gauges(&GroupScheduler{sched: f.s, now: func() time.Time { return f.now }})
	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(gauges))
	values := func() map[string]float64 {
		fams, err := reg.Gather()
		require.NoError(t, err)
		out := map[string]float64{}
		for _, fam := range fams {
			for _, m := range fam.GetMetric() {
				out[fam.GetName()+labelsOf(m)] = m.GetGauge().GetValue()
			}
		}
		return out
	}
	first := values()
	if got := first["llm_d_epp_nca2_line_tokens{pod=p0}"]; got != 640 {
		t.Fatalf("line_tokens{pod=p0} = %v, want 640 (all values: %v)", got, first)
	}
	if got := first["llm_d_epp_nca2_room_tokens{pod=p0}"]; got != lineAdmit {
		t.Fatalf("room_tokens = %v, want %d", got, lineAdmit)
	}
	if _, has := first["llm_d_epp_nca2_priority_waiting"]; !has {
		t.Fatalf("the global gauges must be emitted unlabelled: %v", first)
	}
	second := values()
	require.Equal(t, first, second, "a scrape must not change what the next one reports (I9)")
}

// labelsOf renders one metric's label set canonically, for the value map above.
func labelsOf(m *dto.Metric) string {
	if len(m.GetLabel()) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("{")
	for i, lp := range m.GetLabel() {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(lp.GetName() + "=" + lp.GetValue())
	}
	b.WriteString("}")
	return b.String()
}

// TestScrapesDoNotChangeDecisions: I9 — the same fixture, driven twice, once with a Snapshot (and a
// gauge collect) before every pass and once without, must produce the same decision stream.
func TestScrapesDoNotChangeDecisions(t *testing.T) {
	run := func(t *testing.T, scrape bool) []string {
		t.Helper()
		f := newFixture(t, "p0", "p1")
		var trace []string
		reg := prometheus.NewRegistry()
		require.NoError(t, reg.Register(newNCA2Gauges(&GroupScheduler{sched: f.s, now: func() time.Time { return f.now }})))
		for i := 0; i < 6; i++ {
			if scrape {
				snap, err := json.Marshal(f.s.Snapshot(f.now))
				require.NoError(t, err)
				trace = append(trace, string(snap))
				fams, err := reg.Gather()
				require.NoError(t, err)
				require.NotEmpty(t, fams)
			}
			id := string(rune('a' + i))
			trace = append(trace, string(rune('a'+i))+":"+itoa(f.step(f.hd(id, 10))))
			if i%2 == 1 {
				f.atPod(id, f.homeOf(id))
				f.s.FirstChunk("r-"+id, f.now)
				f.done(id, f.homeOf(id))
			}
		}
		return trace
	}
	with := run(t, true)
	without := run(t, false)
	// the snapshots are only in the scraped run's trace: compare the decision lines
	strip := func(lines []string) []string {
		out := make([]string, 0, len(lines))
		for _, l := range lines {
			if !strings.HasPrefix(l, "{") {
				out = append(out, l)
			}
		}
		return out
	}
	require.Equal(t, strip(without), strip(with), "a scrape must never change a decision")
}

// itoa is a tiny local integer formatter (a trace line, not a report).
func itoa(v int) string {
	if v < 0 {
		return "-" + itoa(-v)
	}
	if v < 10 {
		return string(rune('0' + v))
	}
	return itoa(v/10) + string(rune('0'+v%10))
}

// TestSpanCarriesTheClientRequestID: §7 — the nca2 attributes, the client's own x-request-id among
// them (the EXACT join with the client's records), land on the recording span at PreRequest.
func TestSpanCarriesTheClientRequestID(t *testing.T) {
	recorder := tracetestRecorder(t)
	ctx, span := tracerStart(t, "request_orchestration")
	obs := dispatchObservables{queueWait: 1500 * time.Millisecond, predictedCached: 64, promptTokens: 640,
		nca2: nca2Obs{valid: true, class: nca2ClassReturner, path: pathNone, homePod: "p0",
			dispatchPod: "p0", positionAtTouch: 960, predictedWarm: true, room: 2080,
			doorWait: 2500 * time.Millisecond, seatWait: 2500 * time.Millisecond,
			pushedOutBefore: true, clientRequestID: "x-req-1", threadID: "t1", wireID: "w1"}}
	annotateSpan(ctx, "p0", obs)
	span.End()
	ended := recorder.Ended()
	if len(ended) != 1 {
		t.Fatalf("one span, got %d", len(ended))
	}
	got := map[string]string{}
	for _, kv := range ended[0].Attributes() {
		got[string(kv.Key)] = kv.Value.Emit()
	}
	for key, want := range map[string]string{
		spanNCA2Class:           nca2ClassReturner,
		spanNCA2Path:            pathNone,
		spanNCA2HomePod:         "p0",
		spanNCA2DispatchPod:     "p0",
		spanNCA2PositionAtTouch: "960",
		spanNCA2PredictedWarm:   "true",
		spanNCA2RoomAtDecision:  "2080",
		spanNCA2DoorWaitMs:      "2500",
		spanNCA2SeatWaitMs:      "2500",
		spanNCA2PushedOutBefore: "true",
		spanNCA2ClientRequestID: "x-req-1",
		spanNCA2Thread:          "t1",
		spanNCA2WireID:          "w1",
	} {
		if got[key] != want {
			t.Fatalf("span attribute %s = %q, want %q (all: %v)", key, got[key], want, got)
		}
	}
}

// TestSpanHasNoNCA2AttributesWhenTheTurnIsNotNCA2s: OFF's span keeps exactly the shipped attributes.
func TestSpanHasNoNCA2AttributesWhenTheTurnIsNotNCA2s(t *testing.T) {
	recorder := tracetestRecorder(t)
	ctx, span := tracerStart(t, "request_orchestration")
	annotateSpan(ctx, "p0", dispatchObservables{})
	span.End()
	for _, kv := range recorder.Ended()[0].Attributes() {
		if strings.HasPrefix(string(kv.Key), "llm_d.nca2.") {
			t.Fatalf("an nca2 attribute on a non-nca2 turn: %s", kv.Key)
		}
	}
}

// tracetestRecorder installs an in-memory span recorder as the global tracer provider.
func tracetestRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	orig := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(orig) })
	return recorder
}

// tracerStart starts a recording span from the global provider.
func tracerStart(t *testing.T, name string) (context.Context, trace.Span) {
	t.Helper()
	return otel.Tracer("test").Start(context.Background(), name)
}
