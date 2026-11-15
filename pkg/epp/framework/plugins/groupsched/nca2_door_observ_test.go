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

// THE DOOR'S OBSERVABILITY (session-layer §9, O1-O9): the series exist only with the switch they
// describe, the counters move at the events they name, the gauges and the debug state read ONE snapshot,
// and the span carries the door path and the session labels on the production path.

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// doorSeries and sessionSeries are the door's and the session layer's series names.
var doorSeries = []string{
	"llm_d_epp_nca2_door_admissions_total",
	"llm_d_epp_nca2_head_changes_total",
	"llm_d_epp_nca2_collections_total",
	"llm_d_epp_nca2_backfill_refused_total",
	"llm_d_epp_nca2_collect_held_blocks",
	"llm_d_epp_nca2_collect_pod",
	"llm_d_epp_nca2_door_waiting",
}

var sessionSeries = []string{
	"llm_d_epp_nca2_session_header_total",
	"llm_d_epp_nca2_session_mismatch_total",
}

// TestO_DoorSeriesOnlyWithTheirSwitch: v2 (nca2 with both switches off) gains no series; Part A gains the
// door's; Part B also the session layer's.
func TestO_DoorSeriesOnlyWithTheirSwitch(t *testing.T) {
	build := func(t *testing.T, params string) map[string]struct{} {
		handle := testutils.NewTestHandle(t.Context())
		built, err := Factory("group", json.NewDecoder(strings.NewReader(params)), handle)
		require.NoError(t, err)
		require.NoError(t, built.(*GroupScheduler).sched.SetNumBlocksFor("p0", 64, 1000))
		return seriesNames(t, handle.Metrics().(prometheus.Gatherer))
	}
	has := func(names map[string]struct{}, list []string) (all, none bool) {
		all, none = true, true
		for _, n := range list {
			if _, ok := names[n]; ok {
				none = false
			} else {
				all = false
			}
		}
		return all, none
	}
	v2 := build(t, `{"mode":"nca2"}`)
	if _, none := has(v2, append(append([]string{}, doorSeries...), sessionSeries...)); !none {
		t.Fatal("U8/C7: nca2 with both switches off must not gain a door series")
	}
	a := build(t, `{"mode":"nca2","doorBackfill":true}`)
	if all, _ := has(a, doorSeries); !all {
		t.Fatalf("Part A must register the door series: %v", a)
	}
	if _, none := has(a, sessionSeries); !none {
		t.Fatal("Part A must not register the session series")
	}
	b := build(t, `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"x-llm-d-session-id"}`)
	if all, _ := has(b, append(append([]string{}, doorSeries...), sessionSeries...)); !all {
		t.Fatal("Part B must register the door and the session series")
	}
}

// withDoorMetrics wires real door metrics into a fixture and returns their registry.
func withDoorMetrics(t *testing.T, f *fixture) *prometheus.Registry {
	t.Helper()
	m := newNCA2Metrics()
	m.door = newDoorMetrics(f.s.nca2.door.sessionAware)
	reg := prometheus.NewRegistry()
	for _, c := range m.collectors() {
		require.NoError(t, reg.Register(c))
	}
	require.NoError(t, reg.Register(newNCA2Gauges(&GroupScheduler{sched: f.s, now: func() time.Time { return f.now }})))
	f.s.nca2m = m
	return reg
}

// metricValues is every series' value, keyed by name{labels}.
func metricValues(t *testing.T, reg prometheus.Gatherer) map[string]float64 {
	t.Helper()
	fams, err := reg.Gather()
	require.NoError(t, err)
	out := map[string]float64{}
	for _, fam := range fams {
		for _, m := range fam.GetMetric() {
			v := m.GetCounter().GetValue() + m.GetGauge().GetValue()
			out[fam.GetName()+labelsOf(m)] = v
		}
	}
	return out
}

// TestO1toO5_CountersAndGaugesOnTheDryRun: T-DR1's pass moves O1 (one backfill and one returner... the
// returner is v2's own counter), O2 (one collection start), O3 (p0 holds, p1 does not), O4 (the waiting
// classes) and O5 (the exhaustive sweep's refusals).
func TestO1toO5_CountersAndGaugesOnTheDryRun(t *testing.T) {
	f, heads := collectingFixture(t, true)
	reg := withDoorMetrics(t, f)
	f.pass(append(heads(), f.heavyAt("s", 15, 320, f.now), f.heavyAt("big2", 940, 320, f.now))...)
	v := metricValues(t, reg)
	for key, want := range map[string]float64{
		"llm_d_epp_nca2_door_admissions_total{class=newcomer,path=backfill}": 1,
		"llm_d_epp_nca2_door_admissions_total{class=newcomer,path=head_fit}": 0,
		"llm_d_epp_nca2_collections_total{event=start}":                      1,
		"llm_d_epp_nca2_collect_held_blocks{pod=p0}":                         400,
		"llm_d_epp_nca2_collect_held_blocks{pod=p1}":                         0,
		"llm_d_epp_nca2_collect_pod{pod=p0}":                                 1,
		"llm_d_epp_nca2_collect_pod{pod=p1}":                                 0,
		"llm_d_epp_nca2_door_waiting{class=newcomer}":                        2, // the head N and big2
		"llm_d_epp_nca2_door_waiting{class=priority}":                        0,
		"llm_d_epp_nca2_backfill_refused_total{reason=gpu}":                  1, // big2 fits no budget
	} {
		if got, ok := v[key]; !ok || got != want {
			t.Fatalf("%s = %v (present %v), want %v", key, got, ok, want)
		}
	}
	// the head is admitted later: O2 counts the head change and the completed collection
	f.s.FirstChunk("rq-a2", f.now)
	f.s.Settle("rq-a2", "p0", f.now)
	f.wait(10 * time.Second)
	f.pass(heads()...)
	v = metricValues(t, reg)
	if !f.admitted("N") || v["llm_d_epp_nca2_head_changes_total{reason=admitted}"] != 1 ||
		v["llm_d_epp_nca2_collections_total{event=completed}"] != 1 ||
		v["llm_d_epp_nca2_door_admissions_total{class=newcomer,path=head_collected}"] != 1 {
		t.Fatalf("O1/O2 on the head's admission: admitted %v, %v", f.admitted("N"), v)
	}
}

// TestO2_TakeoverAndReleaseEvents: O2/O8 — a takeover to another pod is a head change and a released
// collection, and the COLLECT events name them.
func TestO2_TakeoverAndReleaseEvents(t *testing.T) {
	f, heads := collectingFixture(t, true)
	reg := withDoorMetrics(t, f)
	var events []string
	f.s.onEvent = func(ev Event) {
		if ev.Kind == "COLLECT" || ev.Kind == "SEAT" {
			events = append(events, ev.Kind+" "+ev.ID+" "+ev.Detail)
		}
	}
	f.pass(heads()...)
	f.priorityOn("p1", "P", f.now.Add(-time.Hour))
	f.pass(append(heads(), f.heavyAt("P", 900, 320, f.now.Add(-time.Hour)))...)
	v := metricValues(t, reg)
	if v["llm_d_epp_nca2_head_changes_total{reason=takeover}"] != 1 || v["llm_d_epp_nca2_collections_total{event=released_takeover}"] != 1 {
		t.Fatalf("O2: %v", v)
	}
	joined := strings.Join(events, "\n")
	for _, want := range []string{"COLLECT N event=start pod=p0", "COLLECT N event=released_takeover pod=p0 held=400",
		"COLLECT P event=start pod=p1", "SEAT rB kind=", "path=returner"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("O8: the event stream lacks %q:\n%s", want, joined)
		}
	}
}

// TestO6_SessionCounters: O6 — one count per dispatched request, with or without the header, and the
// mismatch counter.
func TestO6_SessionCounters(t *testing.T) {
	f := newSessionFixture(t, "p0")
	reg := withDoorMetrics(t, f)
	f.pass(inSession(f.heavyAt("a", 5, 320, f.now), "S"))
	f.pass(f.heavyAt("b", 5, 320, f.now))
	other := inSession(f.heavyAt("a", 5, 320, f.now), "T")
	other.ReqID = "r-a2"
	f.atPod("a", "p0")
	f.done("a", "p0")
	f.pass(other)
	v := metricValues(t, reg)
	if v["llm_d_epp_nca2_session_header_total{present=true}"] != 2 || v["llm_d_epp_nca2_session_header_total{present=false}"] != 1 ||
		v["llm_d_epp_nca2_session_mismatch_total"] != 1 {
		t.Fatalf("O6: %v", v)
	}
}

// TestO7_SpanCarriesTheDoorPathAndTheSession: O7 — on the production path (Pass, AtPod, annotateSpan) a
// backfilled turn's span names its door path, its session group and the session id's hash.
func TestO7_SpanCarriesTheDoorPathAndTheSession(t *testing.T) {
	f := newSessionFixture(t, "p0")
	recorder := tracetestRecorder(t)
	ctx, span := tracerStart(t, "request_orchestration")
	h := inSession(f.heavyAt("a", 5, 320, f.now), "root-1")
	if i := f.pass(h); i != 0 {
		t.Fatalf("fixture: served %d", i)
	}
	annotateSpan(ctx, "p0", f.s.AtPod(h.ReqID, "p0", f.now))
	span.End()
	got := map[string]string{}
	for _, kv := range recorder.Ended()[0].Attributes() {
		got[string(kv.Key)] = kv.Value.String()
	}
	if got[spanNCA2DoorPath] != doorPathHeadFit || got[spanNCA2SessionGroup] != "C" || len(got[spanNCA2SessionIDHash]) != 10 {
		t.Fatalf("O7: %v", got)
	}
	// a seated thread's later turn: path seated, its (now inside) session group A
	f.done("a", "p0")
	ctx, span = tracerStart(t, "request_orchestration")
	h2 := inSession(f.heavyAt("a", 6, 384, f.now), "root-1")
	h2.ReqID = "r-a2"
	if i := f.pass(h2); i != 0 {
		t.Fatalf("fixture: served %d", i)
	}
	annotateSpan(ctx, "p0", f.s.AtPod(h2.ReqID, "p0", f.now))
	span.End()
	got = map[string]string{}
	for _, kv := range recorder.Ended()[1].Attributes() {
		got[string(kv.Key)] = kv.Value.String()
	}
	if got[spanNCA2DoorPath] != doorPathSeated || got[spanNCA2SessionGroup] != "A" {
		t.Fatalf("O7 on a seated turn: %v", got)
	}
}

// TestO7_NoDoorAttributesWhenTheDoorIsOff: C7/U8 — v2's span gains nothing.
func TestO7_NoDoorAttributesWhenTheDoorIsOff(t *testing.T) {
	f := newFixture(t, "p0")
	recorder := tracetestRecorder(t)
	ctx, span := tracerStart(t, "request_orchestration")
	h := f.hd("a", 5)
	f.step(h)
	annotateSpan(ctx, "p0", f.s.AtPod(h.ReqID, "p0", f.now))
	span.End()
	for _, kv := range recorder.Ended()[0].Attributes() {
		switch string(kv.Key) {
		case spanNCA2DoorPath, spanNCA2SessionGroup, spanNCA2SessionIDHash:
			t.Fatalf("a door attribute with the door off: %s", kv.Key)
		}
	}
}

// TestO9_DebugStateCarriesTheDoor: O9 — /debug/plugins/state's nca2 block names the head, its pod, the
// held blocks, the door's waiting counts and (Part B) the sessions inside; v2's block is unchanged.
func TestO9_DebugStateCarriesTheDoor(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	raw, err := json.Marshal(f.s.Snapshot(f.now))
	require.NoError(t, err)
	for _, want := range []string{`"headID":"N"`, `"collectPod":"p0"`, `"heldBlocks":400`, `"doorWaitingByGroup":{"newcomer":1,"priority":0}`,
		`"collectHeldBlocks":400`} {
		require.Contains(t, string(raw), want)
	}
	require.NotContains(t, string(raw), "sessionsInside", "Part A has no sessions")
	g := newSessionFixture(t, "p0")
	g.pass(inSession(g.heavyAt("a", 5, 320, g.now), "S"), inSession(g.heavyAt("b", 2000, 320, g.now), "T"))
	raw, err = json.Marshal(g.s.Snapshot(g.now))
	require.NoError(t, err)
	require.Contains(t, string(raw), `"sessionsInside":1`)
	require.Contains(t, string(raw), `"doorWaitingByGroup":{"A":0,"B":0,"C":1}`)
	v2 := newFixture(t, "p0")
	v2.step(v2.hd("a", 5))
	raw, err = json.Marshal(v2.s.Snapshot(v2.now))
	require.NoError(t, err)
	for _, key := range []string{"headID", "collectPod", "heldBlocks", "doorWaitingByGroup", "collectHeldBlocks", "sessionsInside"} {
		require.NotContains(t, string(raw), key, "v2's debug state must not gain %s", key)
	}
}

// TestO_AScrapeChangesNothing: I9 for the door — a snapshot and a gauge collect write nothing: the same
// fixture with and without scrapes takes the same decisions.
func TestO_AScrapeChangesNothing(t *testing.T) {
	run := func(scrape bool) []string {
		f := newSessionFixture(t, "p0", "p1")
		reg := withDoorMetrics(t, f)
		var out []string
		for i := 0; i < 8; i++ {
			if scrape {
				_ = metricValues(t, reg)
				_, _ = json.Marshal(f.s.Snapshot(f.now))
			}
			id := "t" + itoa(i)
			out = append(out, id+":"+itoa(f.pass(inSession(f.heavyAt(id, 100+50*i, 320, f.now), "S"+itoa(i%3)))))
		}
		return out
	}
	require.Equal(t, run(false), run(true))
}
