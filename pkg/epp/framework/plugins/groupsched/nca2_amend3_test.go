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

// coldCharge gives a running seat an outstanding cold-entry charge of tokens (the caller does not hold
// the lock).
func (f *fixture) coldCharge(pod, id string, tokens int64) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g := f.s.groups[pod]
	m := g.threads[id]
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
			f.coldCharge("p0", "hot", 3000)
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
		f.coldCharge("p0", "hot", 3000)
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

// TestVT2_OutsidersStayFullyCold: V1 — warmth counts only for an admitted thread on its home pod: a
// newcomer, a rewritten context under a new thread id, a pushed-out thread with its old entry, and a
// returner on a pod that is not its home are all as cold as the GPU says; and a pushed-out head the
// GPU-only valve holds is still held with valveCPUWarm on.
func TestVT2_OutsidersStayFullyCold(t *testing.T) {
	f := valveFixture(t, false, true, "p0", "p1")
	f.cpuWarmReturner("p0", "r", 5000)
	f.pushedOutWithEntry("p0", "p", 5000)
	h := f.warmHead("x")
	f.s.mu.Lock()
	g0, g1 := f.s.groups["p0"], f.s.groups["p1"]
	for _, c := range []struct {
		name string
		g    *engineGroup
		w    *waiter
	}{
		{"newcomer", g0, &waiter{id: "n", chains: h.Chains}},
		{"new-id rewrite of r", g0, &waiter{id: "r-rewritten", chains: h.Chains}},
		{"pushed out", g0, &waiter{id: "p", chains: h.Chains}},
		{"returner off its home", g1, &waiter{id: "r", chains: h.Chains}},
	} {
		if got, want := f.s.valveCold(c.g, c.w), c.g.index.ColdTokens(c.w.chains); got != want {
			f.s.mu.Unlock()
			t.Fatalf("%s: valveCold %d, want the GPU-cold %d", c.name, got, want)
		}
	}
	if got := f.s.valveCold(g0, &waiter{id: "r", chains: h.Chains}); got != 120 {
		f.s.mu.Unlock()
		t.Fatalf("fixture: the returner on its home is warm: valveCold %d, want 120", got)
	}
	f.s.mu.Unlock()
	f.running("p0", "hot", 100, 320)
	f.coldCharge("p0", "hot", 3000)
	f.running("p1", "hot1", 100, 320)
	f.coldCharge("p1", "hot1", 3000)
	if served := f.pass(f.warmHead("p")); served != -1 || f.admitted("p") {
		t.Fatalf("the pushed-out head stays cold at the valve (served %d)", served)
	}
}

// TestVT4_HeadHeadroomUsesValveCold: V1 on the head-headroom term — the door's head is an outsider, so
// valveCold is its GPU-cold tokens: a pushed-out head whose old entry is on its collection pod still
// needs its full headroom there, and a small cold outsider that would eat it backfills on the other pod.
func TestVT4_HeadHeadroomUsesValveCold(t *testing.T) {
	f := valveFixture(t, true, true, "p0", "p1")
	f.s.mu.Lock()
	g := f.s.groups["p0"]
	w := &waiter{id: "cold", since: f.now, chains: []Chain{{Keys: keys(400, seedFor("cold")), Tokens: 320}}}
	f.s.seat(g, w, 2000, f.now)
	f.s.nca2.homes["cold"], f.s.nca2.admitted["cold"], f.s.nca2.ever["cold"] = "p0", true, true
	f.s.nca2LineSet(g, "cold", 320, f.now, pushoutRoom)
	f.s.release(g.threads["cold"], "rq-cold", w.chains, f.now, dispatchObservables{})
	f.s.mu.Unlock()
	f.s.AtPod("rq-cold", "p0", f.now)
	f.idleSeat("p0", "a1", 400)
	for i := 0; i < 2; i++ {
		f.running("p1", "b"+string(rune('0'+i)), 300, 320)
	}
	f.pushedOutWithEntry("p0", "N", 1000)
	f.wait(10 * time.Second)
	head := f.heavyAt("N", 900, 1000, f.now)
	f.pass(head)
	if h, pod := f.doorHead(); h != "N" || pod != "p0" {
		t.Fatalf("fixture: the head N collects on p0, got %q on %q", h, pod)
	}
	f.s.mu.Lock()
	n := f.s.waiters["N"]
	need := f.s.doorHeadColdNeed(f.s.groups["p0"], n)
	want := min(f.s.groups["p0"].index.ColdTokens(n.chains), f.s.groups["p0"].coldCapTokens(f.s.cfg.GroupBudgetFraction, f.s.blockSize))
	f.s.mu.Unlock()
	if need != want || need != 1000 {
		t.Fatalf("the head's headroom is min(valveCold, cap) = its GPU-cold %d, got %d", want, need)
	}
	f.pass(head, f.heavyAt("s", 5, 300, f.now))
	if f.homeOf("s") != "p1" {
		t.Fatalf("the small cold outsider backfills on p1, got %q", f.homeOf("s"))
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
