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

// LOGIC Amendment 3: the cold valve counts a returner's CPU-warm context as warm (valveCPUWarm, C9/V1),
// and seats are released by silence only (noQuantum, C10/Q0).

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// resolveParams decodes a parameters block strictly and resolves its nca2 policy.
func resolveParams(t *testing.T, params string) *nca2Policy {
	t.Helper()
	cfg := defaultConfig()
	dec := json.NewDecoder(strings.NewReader(params))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		t.Fatal(err)
	}
	n, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatalf("%s must resolve, got %v", params, err)
	}
	return n
}

// TestC9C10_KeysRequireNCA2: valveCPUWarm and noQuantum are nca2 keys: either one without mode: nca2
// refuses the startup, whatever its value (presence is the fact).
func TestC9C10_KeysRequireNCA2(t *testing.T) {
	for name, params := range map[string]string{
		"valveCPUWarm without mode":       `{"valveCPUWarm":true}`,
		"valveCPUWarm false without mode": `{"valveCPUWarm":false}`,
		"noQuantum without mode":          `{"noQuantum":true}`,
		"noQuantum false without mode":    `{"noQuantum":false}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", name)
			}
		})
	}
}

// TestC9C10_KeysResolve: each key resolves into the nca2 policy, with the door or without it; absent or
// false is off.
func TestC9C10_KeysResolve(t *testing.T) {
	if n := resolveParams(t, `{"mode":"nca2"}`); n.valveCPUWarm || n.noQuantum {
		t.Fatalf("absent keys are off: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","valveCPUWarm":false,"noQuantum":false}`); n.valveCPUWarm || n.noQuantum {
		t.Fatalf("false keys are off: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","valveCPUWarm":true}`); !n.valveCPUWarm || n.noQuantum || n.door != nil {
		t.Fatalf("C9 alone, v2's decision: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","noQuantum":true}`); n.valveCPUWarm || !n.noQuantum || n.door != nil {
		t.Fatalf("C10 alone, v2's decision: %+v", n)
	}
	n := resolveParams(t, `{"mode":"nca2","doorBackfill":true,"backfillReclaim":true,"valveCPUWarm":true,"noQuantum":true}`)
	if !n.valveCPUWarm || !n.noQuantum || n.door == nil || !n.door.backfillReclaim {
		t.Fatalf("both with the door: %+v", n)
	}
	if err := factoryWith(t, `{"mode":"nca2","valveCPUWarm":true,"noQuantum":true}`); err != nil {
		t.Fatalf("the factory accepts both keys in nca2: %v", err)
	}
}

// TestC9C10_Echo: the nca2 echo prints both resolved values, set or not; the effective-config echo carries
// them when set; OFF's echo names neither.
func TestC9C10_Echo(t *testing.T) {
	cfg := defaultConfig()
	cfg.Mode = modeNCA2
	n, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatal(err)
	}
	if got := n.amend3Echo(); got != "valveCPUWarm=false noQuantum=false" {
		t.Fatalf("both off: %q", got)
	}
	on := true
	cfg.ValveCPUWarm, cfg.NoQuantum = &on, &on
	if n, err = cfg.resolveNCA2(); err != nil {
		t.Fatal(err)
	}
	if got := n.amend3Echo(); got != "valveCPUWarm=true noQuantum=true" {
		t.Fatalf("both on: %q", got)
	}
	if echo := configEcho(cfg); !strings.Contains(echo, "valveCPUWarm=true") || !strings.Contains(echo, "noQuantum=true") {
		t.Fatalf("the effective-config echo carries the keys that were set: %s", echo)
	}
	for _, key := range []string{"valveCPUWarm", "noQuantum"} {
		if strings.Contains(configEcho(defaultConfig()), key) {
			t.Fatalf("the OFF echo must not name %q", key)
		}
	}
}

// valveFixture is one pod (cold cap 3200 tokens) in v2 or with the door, valveCPUWarm on or off.
func valveFixture(t *testing.T, door, warm bool, pods ...string) *fixture {
	t.Helper()
	tune := func(c *Config) {
		if warm {
			on := true
			c.ValveCPUWarm = &on
		}
	}
	if door {
		return newDoorFixture(t, tune, pods...)
	}
	return newV2Twin(t, tune, pods...)
}

// hotCharge gives the running seat "hot" an outstanding cold-entry charge of tokens on its pod (the caller
// does not hold the lock).
func (f *fixture) hotCharge(tokens int64) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g := f.s.groupOf("hot")
	m := g.threads["hot"]
	m.coldEntry, m.coldDone = tokens, false
	g.coldInFlight += tokens
}

// cpuWarmReturner makes id an admitted thread of pod, waiting with no seat, whose CPU-line entry holds
// entryTokens (its context before this turn).
func (f *fixture) cpuWarmReturner(pod, id string, entryTokens int64) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	f.s.nca2.homes[id], f.s.nca2.admitted[id], f.s.nca2.ever[id] = pod, true, true
	f.s.groups[pod].nca2.line = append(f.s.groups[pod].nca2.line, nca2Entry{thread: id, tokens: entryTokens})
}

// pushedOutWithEntry makes id a pushed-out thread whose old entry (entryTokens) is still on pod.
func (f *fixture) pushedOutWithEntry(pod, id string, entryTokens int64) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	f.s.nca2.homes[id], f.s.nca2.ever[id], f.s.nca2.pushed[id] = pod, true, true
	f.s.groups[pod].nca2.line = append(f.s.groups[pod].nca2.line, nca2Entry{thread: id, tokens: entryTokens})
}

// warmHead is a turn of 80 blocks and 5120 tokens, none of it resident on the GPU.
func (f *fixture) warmHead(id string) Head {
	return f.heavyAt(id, 80, 5120, f.now.Add(-time.Minute))
}

// coldOn reads a pod's outstanding cold charge and one seat's own.
func (f *fixture) coldOn(pod, id string) (pod64, seat int64) {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g := f.s.groups[pod]
	if m := g.threads[id]; m != nil {
		seat = m.coldEntry
	}
	return g.coldInFlight, seat
}

// TestVT1_CPUWarmReturnerPassesTheValve: V1 — a returner whose context (5000 of its 5120 tokens) is in its
// home pod's CPU line, with nothing on the GPU, is refused by the GPU-only valve while another cold
// prefill holds 3000 of the 3200-token cap, and is seated with valveCPUWarm (120 cold tokens). v2 and the
// door, each with its off-twin.
func TestVT1_CPUWarmReturnerPassesTheValve(t *testing.T) {
	for _, door := range []bool{false, true} {
		for _, warm := range []bool{false, true} {
			f := valveFixture(t, door, warm, "p0")
			f.running("p0", "hot", 100, 320)
			f.hotCharge(3000)
			f.cpuWarmReturner("p0", "r", 5000)
			served := f.pass(f.warmHead("r"))
			if warm && (served != 0 || !seated(f.s, "r")) {
				t.Fatalf("door %v: with valveCPUWarm the CPU-warm returner is seated (served %d)", door, served)
			}
			if !warm && (served != -1 || seated(f.s, "r")) {
				t.Fatalf("door %v: off-twin: the GPU-only valve refuses the returner (served %d)", door, served)
			}
		}
	}
}

// TestVT3_ColdInFlightIsChargedValveCold: V1/O10 — the seat's cold-entry charge is valveCold (120), not
// the GPU-cold tokens (5120 capped at 3200), and the excused tokens (5000) are counted per pod.
func TestVT3_ColdInFlightIsChargedValveCold(t *testing.T) {
	for _, door := range []bool{false, true} {
		f := valveFixture(t, door, true, "p0")
		m := newNCA2Metrics()
		m.cpuWarm = newCPUWarmCounter()
		reg := prometheus.NewRegistry()
		for _, c := range m.collectors() {
			require.NoError(t, reg.Register(c))
		}
		f.s.nca2m = m
		m.preInstantiatePod("p0")
		f.running("p0", "hot", 100, 320)
		f.hotCharge(3000)
		f.cpuWarmReturner("p0", "r", 5000)
		if f.pass(f.warmHead("r")) != 0 {
			t.Fatalf("door %v: fixture: r is seated", door)
		}
		pod, seat := f.coldOn("p0", "r")
		if seat != 120 || pod != 3120 {
			t.Fatalf("door %v: the seat is charged valveCold: seat %d (want 120), pod %d (want 3120)", door, seat, pod)
		}
		if got := metricValues(t, reg)["llm_d_epp_nca2_valve_cpu_warm_tokens_total{pod=p0}"]; got != 5000 {
			t.Fatalf("door %v: O10 counts the 5000 excused tokens, got %v", door, got)
		}
	}
}

// TestVT2_NoEntryIsFullyCold: V1 (as corrected) - warmth is the thread's own entry on the pod's CPU line:
// a newcomer, a rewritten context under a new thread id, and a thread on a pod that does not hold its
// entry are as cold as the GPU says; an admitted returner and a pushed-out thread whose entry is still in
// the line are warm on their pod.
func TestVT2_NoEntryIsFullyCold(t *testing.T) {
	f := valveFixture(t, false, true, "p0", "p1")
	f.cpuWarmReturner("p0", "r", 5000)
	f.pushedOutWithEntry("p0", "p", 5000)
	h := f.warmHead("x")
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g0, g1 := f.s.groups["p0"], f.s.groups["p1"]
	for _, c := range []struct {
		name string
		g    *engineGroup
		w    *waiter
		want int64
	}{
		{"newcomer", g0, &waiter{id: "n", chains: h.Chains}, 5120},
		{"new-id rewrite of r", g0, &waiter{id: "r-rewritten", chains: h.Chains}, 5120},
		{"returner off its entry's pod", g1, &waiter{id: "r", chains: h.Chains}, 5120},
		{"pushed out off its entry's pod", g1, &waiter{id: "p", chains: h.Chains}, 5120},
		{"returner", g0, &waiter{id: "r", chains: h.Chains}, 120},
		{"pushed out, entry in the line", g0, &waiter{id: "p", chains: h.Chains}, 120},
	} {
		if got := f.s.valveCold(c.g, c.w); got != c.want {
			t.Fatalf("%s: valveCold %d, want %d (GPU-cold %d)", c.name, got, c.want, c.g.index.ColdTokens(c.w.chains))
		}
	}
}

// pushedOutFixture is one pod with a 20000-token CPU tier and the admission line at 10000: a running seat
// holds 3000 of the 3200-token cold cap (it is not on the line, so the room is the whole admission line),
// and p was pushed out with its 1900-token entry at position 11700, past the admission line and inside
// the tier. With trim, another thread's touch moves the entry past the tier and the trim drops it.
func pushedOutFixture(t *testing.T, door, warm, trim bool) *fixture {
	t.Helper()
	tune := func(c *Config) {
		c.CPUTierTokens, c.AdmitLine = i64(20_000), f64(0.5)
		if warm {
			on := true
			c.ValveCPUWarm = &on
		}
	}
	s := newNCA2Tuned(t, func(c *Config) {
		if door {
			on := true
			c.DoorBackfill = &on
		}
		tune(c)
	}, []string{"p0"})
	f := &fixture{t: t, s: s, now: epoch}
	f.running("p0", "hot", 100, 320)
	f.hotCharge(3000)
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g := f.s.groups["p0"]
	g.nca2.line = nca2DropEntry(g.nca2.line, "hot")
	f.s.nca2.admitted["hot"] = false
	f.s.nca2.homes["f1"], f.s.nca2.homes["p"] = "p0", "p0"
	f.s.nca2.ever["p"], f.s.nca2.pushed["p"] = true, true
	g.nca2.line = append(g.nca2.line, nca2Entry{thread: "f1", tokens: 9800}, nca2Entry{thread: "p", tokens: 1900})
	if trim {
		f.s.nca2.homes["o"] = "p0"
		f.s.nca2LineSet(g, "o", 8500, f.now, pushoutTouch)
	}
	if _, has := f.s.nca2PositionOn(g, "p"); has == trim {
		t.Fatalf("fixture: p's entry on the line %v, want %v", has, !trim)
	}
	return f
}

// TestVT5_PushedOutEntryInTheLineIsWarm: V1 (as corrected) - a pushed-out thread whose entry is still in
// the line (past the admission line, inside the tier) is charged only its new tokens (2048 - 1900 = 148)
// and re-admitted past a valve the GPU-cold tokens would not pass; its twin whose entry was trimmed past
// the tier is fully cold and waits. v2 and the door; the off-switch waits too.
func TestVT5_PushedOutEntryInTheLineIsWarm(t *testing.T) {
	for _, door := range []bool{false, true} {
		for _, c := range []struct{ warm, trim, admitted bool }{
			{warm: true, trim: false, admitted: true},
			{warm: true, trim: true, admitted: false},
			{warm: false, trim: false, admitted: false},
		} {
			f := pushedOutFixture(t, door, c.warm, c.trim)
			f.pass(f.heavyAt("p", 32, 2048, f.now.Add(-time.Minute)))
			if f.admitted("p") != c.admitted {
				t.Fatalf("door %v warm %v trim %v: p admitted %v, want %v", door, c.warm, c.trim, f.admitted("p"), c.admitted)
			}
			if !c.admitted {
				continue
			}
			if pod, seat := f.coldOn("p0", "p"); seat != 148 || pod != 3148 {
				t.Fatalf("door %v: p is charged its new tokens: seat %d (want 148), pod %d (want 3148)", door, seat, pod)
			}
		}
	}
}

// TestVT4_HeadHeadroomUsesValveCold: V1 on the head-headroom term - the door's head is a pushed-out thread
// whose entry (its whole 1000-token prompt) is still on its collection pod: with valveCPUWarm it needs no
// valve headroom there, so a small cold outsider passes BF1's headroom check on that pod; the off-twin
// keeps the 1000-token headroom and refuses it (valve_head_headroom).
func TestVT4_HeadHeadroomUsesValveCold(t *testing.T) {
	for _, warm := range []bool{false, true} {
		f := valveFixture(t, true, warm, "p0")
		f.running("p0", "hot", 100, 320)
		f.hotCharge(2000)
		f.pushedOutWithEntry("p0", "N", 1000)
		head := f.heavyAt("N", 900, 1000, f.now)
		small := f.heavyAt("s", 5, 300, f.now)
		f.s.mu.Lock()
		g := f.s.groups["p0"]
		f.s.syncQueue([]Head{head, small}, f.now)
		d := f.s.nca2.door
		d.headID, d.collectPod = "N", "p0"
		g.nca2.reservedHead = "N"
		need := f.s.doorHeadColdNeed(g, f.s.waiters["N"])
		p := podState{g: g, room: f.s.nca2View(g).room(f.s.nca2.AdmitLineTokens)}
		reason, ok := f.s.doorBackfillFits(&p, f.s.waiters["s"], 300, g, need, map[fitKey]fitFacts{})
		f.s.mu.Unlock()
		if warm && (need != 0 || !ok) {
			t.Fatalf("valveCPUWarm: the head needs no headroom (%d) and s passes BF1 on p0 (ok %v, reason %d)", need, ok, reason)
		}
		if !warm && (need != 1000 || ok || reason != refusedValveHeadHeadroom) {
			t.Fatalf("off-twin: the head needs 1000 (%d) and s is refused for the headroom (ok %v, reason %d)", need, ok, reason)
		}
	}
}

// TestO10_CPUWarmSeriesOnlyWithItsSwitch: valve_cpu_warm_tokens_total exists only with valveCPUWarm, with
// one zero series per pod.
func TestO10_CPUWarmSeriesOnlyWithItsSwitch(t *testing.T) {
	build := func(t *testing.T, params string) map[string]float64 {
		handle := testutils.NewTestHandle(t.Context())
		built, err := Factory("group", json.NewDecoder(strings.NewReader(params)), handle)
		require.NoError(t, err)
		require.NoError(t, built.(*GroupScheduler).sched.SetNumBlocksFor("p0", 64, 1000))
		return metricValues(t, handle.Metrics().(prometheus.Gatherer))
	}
	series := "llm_d_epp_nca2_valve_cpu_warm_tokens_total{pod=p0}"
	for _, params := range []string{`{"mode":"nca2"}`, `{"mode":"nca2","doorBackfill":true,"noQuantum":true}`} {
		if _, ok := build(t, params)[series]; ok {
			t.Fatalf("%s must not carry %s", params, series)
		}
	}
	for _, params := range []string{`{"mode":"nca2","valveCPUWarm":true}`, `{"mode":"nca2","doorBackfill":true,"valveCPUWarm":true}`} {
		if v, ok := build(t, params)[series]; !ok || v != 0 {
			t.Fatalf("%s pre-instantiates %s at zero (present %v, %v)", params, series, ok, v)
		}
	}
}

// quantumFixture is one pod in v2 (quantum 2), noQuantum on or off: q has run two whole turns on its seat
// (two completed busy periods, its last settle a second ago), and a running seat leaves 5 blocks free.
func quantumFixture(t *testing.T, noQuantum bool) *fixture {
	t.Helper()
	f := newV2Twin(t, func(c *Config) {
		c.Quantum = 2
		if noQuantum {
			on := true
			c.NoQuantum = &on
		}
	}, "p0")
	f.turn("q", 10)
	f.turn("q", 10)
	f.running("p0", "big", 935, 320)
	return f
}

// exitsByReason is the scheduler's exits by reason, read under the lock.
func (f *fixture) exitsByReason(reason string) int64 {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.counters.exits[reason]
}

// TestQT1_PastItsQuantumIsNoVictimWithNoQuantum: Q0 — a seat past its quantum that is not silent (its next
// head is queued) is rotation-eligible by quantum completion without noQuantum, and released for an
// outsider that needs the space (v2 then seats it again as a returner from the surplus); with noQuantum it
// is not eligible and keeps its seat.
func TestQT1_PastItsQuantumIsNoVictimWithNoQuantum(t *testing.T) {
	for _, noQuantum := range []bool{false, true} {
		f := quantumFixture(t, noQuantum)
		f.s.mu.Lock()
		g := f.s.groups["p0"]
		m := g.threads["q"]
		if m == nil || m.busyPeriods != 2 || m.busy() {
			f.s.mu.Unlock()
			t.Fatalf("fixture: q is seated with two busy periods and nothing in flight")
		}
		m.waitingSince = f.now // its next head is queued
		reason, ok := f.s.eligible(m, f.now)
		_, victim := f.s.rotationVictim(g, f.now)
		f.s.mu.Unlock()
		if noQuantum && (ok || victim != nil) {
			t.Fatalf("noQuantum: q is not eligible (eligible %v %q, victim %v)", ok, reason, victim)
		}
		if !noQuantum && (!ok || reason != "quantum-complete" || victim == nil || victim.id != "q") {
			t.Fatalf("off-twin: q is the quantum-complete victim (eligible %v %q, victim %v)", ok, reason, victim)
		}
		f.pass(f.hd("q", 10), f.hd("n", 20))
		exits := f.exitsByReason("quantum-complete")
		if noQuantum && (exits != 0 || !seated(f.s, "q")) {
			t.Fatalf("noQuantum: q keeps its seat (quantum-complete exits %d)", exits)
		}
		if !noQuantum && exits != 1 {
			t.Fatalf("off-twin: q is released for n (quantum-complete exits %d)", exits)
		}
	}
}

// TestQT2_AnIdleSeatIsStillAVictim: Q0 — silence still releases a seat with noQuantum: q, past its quantum
// and silent for longer than idleSeconds, is the idle victim and is released for the outsider.
func TestQT2_AnIdleSeatIsStillAVictim(t *testing.T) {
	f := quantumFixture(t, true)
	f.wait(10 * time.Second)
	f.s.mu.Lock()
	reason, ok := f.s.eligible(f.s.groups["p0"].threads["q"], f.now)
	f.s.mu.Unlock()
	if !ok || reason != "idle" {
		t.Fatalf("a silent seat is eligible as idle with noQuantum, got %v %q", ok, reason)
	}
	f.pass(f.hd("n", 15))
	if f.exitsByReason("idle") != 1 || seated(f.s, "q") || !f.admitted("n") {
		t.Fatalf("q is released as idle and n admitted (idle exits %d, n admitted %v)", f.exitsByReason("idle"), f.admitted("n"))
	}
}
