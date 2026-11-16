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

// LOGIC Amendment 4: a collecting head of a session that is or was inside keeps its turn (headKeepsTurn,
// C11): its config gate, its echo, the rule with its twins, its counter, and the door's property loops
// with it on.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// partBParams is the session layer's config, without headKeepsTurn.
const partBParams = `"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"x-llm-d-session-id"`

// TestC11_KeyRequiresTheSessionLayer: headKeepsTurn is an nca2 key (its presence without mode: nca2 refuses
// the startup), and true requires doorBackfill and sessionAware.
func TestC11_KeyRequiresTheSessionLayer(t *testing.T) {
	for name, params := range map[string]string{
		"without mode":           `{"headKeepsTurn":true}`,
		"false without mode":     `{"headKeepsTurn":false}`,
		"nca2 without the door":  `{"mode":"nca2","headKeepsTurn":true}`,
		"Part A without session": `{"mode":"nca2","doorBackfill":true,"headKeepsTurn":true}`,
		"sessions without door":  `{"mode":"nca2","sessionAware":true,"sessionHeader":"x-llm-d-session-id","headKeepsTurn":true}`,
		"Part A with reclaim":    `{"mode":"nca2","doorBackfill":true,"backfillReclaim":true,"headKeepsTurn":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", params)
			}
		})
	}
}

// TestC11_KeyResolves: true with the session layer resolves into the door; absent or false is off.
func TestC11_KeyResolves(t *testing.T) {
	if n := resolveParams(t, `{`+partBParams+`}`); n.door.headKeepsTurn {
		t.Fatal("absent is off")
	}
	if n := resolveParams(t, `{`+partBParams+`,"headKeepsTurn":false}`); n.door.headKeepsTurn {
		t.Fatal("false is off")
	}
	if n := resolveParams(t, `{"mode":"nca2","doorBackfill":true,"headKeepsTurn":false}`); n.door == nil || n.door.headKeepsTurn {
		t.Fatalf("false with Part A is off: %+v", n.door)
	}
	n := resolveParams(t, `{`+partBParams+`,"backfillReclaim":true,"valveCPUWarm":true,"noQuantum":true,"headKeepsTurn":true}`)
	if !n.door.headKeepsTurn || !n.door.sessionAware || !n.door.backfillReclaim || !n.valveCPUWarm || !n.noQuantum {
		t.Fatalf("with every other switch: %+v %+v", n, n.door)
	}
	if err := factoryWith(t, `{`+partBParams+`,"headKeepsTurn":true}`); err != nil {
		t.Fatalf("the factory accepts headKeepsTurn with the session layer: %v", err)
	}
}

// TestC11_Echo: the nca2 echo prints the resolved value, set or not; the effective-config echo carries it
// when set and the nca2 key list names it; OFF's echo does not.
func TestC11_Echo(t *testing.T) {
	cfg := defaultConfig()
	cfg.Mode = modeNCA2
	n, err := cfg.resolveNCA2()
	require.NoError(t, err)
	if got := n.amend4Echo(); got != "headKeepsTurn=false" {
		t.Fatalf("v2: %q", got)
	}
	on, header := true, "x-llm-d-session-id"
	cfg.DoorBackfill, cfg.SessionAware, cfg.SessionHeader = &on, &on, &header
	if n, err = cfg.resolveNCA2(); err != nil {
		t.Fatal(err)
	}
	if got := n.amend4Echo(); got != "headKeepsTurn=false" {
		t.Fatalf("Part B, absent: %q", got)
	}
	cfg.HeadKeepsTurn = &on
	if n, err = cfg.resolveNCA2(); err != nil {
		t.Fatal(err)
	}
	if got := n.amend4Echo(); got != "headKeepsTurn=true" {
		t.Fatalf("on: %q", got)
	}
	if echo := configEcho(cfg); !strings.Contains(echo, "headKeepsTurn=true") {
		t.Fatalf("the effective-config echo carries the key that was set: %s", echo)
	}
	if keys := strings.Join(cfg.nca2KeyNames(), ","); !strings.HasSuffix(keys, ",headKeepsTurn") {
		t.Fatalf("the nca2 key list names it: %s", keys)
	}
	if strings.Contains(configEcho(defaultConfig()), "headKeepsTurn") {
		t.Fatal("the OFF echo must not name headKeepsTurn")
	}
}

// TestO11_TakeoverSuppressedSeriesOnlyWithItsSwitch: door_takeover_suppressed_total exists only with
// headKeepsTurn, at zero.
func TestO11_TakeoverSuppressedSeriesOnlyWithItsSwitch(t *testing.T) {
	build := func(t *testing.T, params string) map[string]float64 {
		handle := testutils.NewTestHandle(t.Context())
		built, err := Factory("group", json.NewDecoder(strings.NewReader(params)), handle)
		require.NoError(t, err)
		require.NoError(t, built.(*GroupScheduler).sched.SetNumBlocksFor("p0", 64, 1000))
		return metricValues(t, handle.Metrics().(prometheus.Gatherer))
	}
	for _, params := range []string{`{"mode":"nca2"}`, `{"mode":"nca2","doorBackfill":true}`, `{` + partBParams + `}`,
		`{` + partBParams + `,"headKeepsTurn":false}`} {
		if _, ok := build(t, params)[suppressedSeries]; ok {
			t.Fatalf("%s must not carry %s", params, suppressedSeries)
		}
	}
	if v, ok := build(t, `{`+partBParams+`,"headKeepsTurn":true}`)[suppressedSeries]; !ok || v != 0 {
		t.Fatalf("headKeepsTurn pre-instantiates %s at zero (present %v, %v)", suppressedSeries, ok, v)
	}
}

// suppressedSeries is C11's counter.
const suppressedSeries = "llm_d_epp_nca2_door_takeover_suppressed_total"

// keepsTurnFixture is Part B on p0 and p1, headKeepsTurn on or off. p0 holds an idle seat a1 (400 blocks)
// and a2 (545, running, of session "in": that session is inside), 5 blocks free; p1 holds b1 (900, running),
// 50 free. The head H needs 900 blocks: of a session that was inside (group B: H was admitted and pushed
// out, its old entry is on p0) or of a new session (group C). With a prompt beyond every pod's CPU room it
// collects nowhere; otherwise it collects on p0 in the first pass, which the fixture runs.
func keepsTurnFixture(t *testing.T, keepsTurn bool, group string, tokens int) (*fixture, Head) {
	t.Helper()
	f := newDoorFixture(t, func(c *Config) {
		on, header := true, "x-llm-d-session-id"
		c.SessionAware, c.SessionHeader = &on, &header
		if keepsTurn {
			c.HeadKeepsTurn = &on
		}
	}, "p0", "p1")
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.running("p1", "b1", 900, 320)
	f.s.mu.Lock()
	f.s.sessionNote("a2", "in")
	f.s.mu.Unlock()
	if group == "B" {
		f.priorityOn("p0", "H", f.now.Add(-time.Hour))
	}
	f.wait(10 * time.Second) // a1 is silent past idleSeconds: reclaimable
	h := inSession(f.heavyAt("H", 900, tokens, f.now.Add(-time.Hour)), "sH")
	f.pass(h)
	f.s.mu.Lock()
	class := f.s.doorClassOf("H")
	f.s.mu.Unlock()
	if want := map[string]int{"B": 1, "C": 2}[group]; class != want {
		t.Fatalf("fixture: H is of group %s (class %d), got class %d", group, want, class)
	}
	return f, h
}

// insider is a turn of the inside session "in" that fits no pod's free budget now (group A).
func insider(f *fixture) Head {
	return inSession(f.heavyAt("c2", 600, 320, f.now), "in")
}

// TestC11_CollectingGroupBHeadKeepsItsTurn: C11 — a group-B head collecting on p0 and a group-A outsider
// that cannot backfill anywhere: with headKeepsTurn the head stays, its collection with it, and the counter
// moves once per pass (the pass below takes two decisions: a small outsider backfills on p1 first); the
// off-twin hands the head to the group-A outsider (H3), which keeps the blocks on the same pod.
func TestC11_CollectingGroupBHeadKeepsItsTurn(t *testing.T) {
	for _, keepsTurn := range []bool{true, false} {
		f, h := keepsTurnFixture(t, keepsTurn, "B", 320)
		if head, pod := f.doorHead(); head != "H" || pod != "p0" || f.held("p0") != 405 {
			t.Fatalf("keepsTurn=%v fixture: H collects on p0 (%q on %q, held %d)", keepsTurn, head, pod, f.held("p0"))
		}
		reg := withDoorMetrics(t, f)
		small := inSession(f.heavyAt("s", 20, 320, f.now), "sS")
		f.pass(h, insider(f), small)
		if !f.admitted("s") || f.homeOf("s") != "p1" {
			t.Fatalf("keepsTurn=%v fixture: the small outsider backfills on p1 in the same pass", keepsTurn)
		}
		v := metricValues(t, reg)
		head, pod := f.doorHead()
		if !keepsTurn {
			if head != "c2" || pod != "p0" || f.reservedHead("p0") != "c2" || v["llm_d_epp_nca2_head_changes_total{reason=takeover}"] != 1 {
				t.Fatalf("off: H3 hands the head to c2 with the blocks (%q on %q for %q)", head, pod, f.reservedHead("p0"))
			}
			if _, ok := v[suppressedSeries]; ok {
				t.Fatal("off: no C11 counter")
			}
			continue
		}
		if head != "H" || pod != "p0" || f.reservedHead("p0") != "H" || f.held("p0") != 405 {
			t.Fatalf("C11: H keeps the head and its collection (%q on %q for %q, held %d)", head, pod, f.reservedHead("p0"), f.held("p0"))
		}
		if v["llm_d_epp_nca2_head_changes_total{reason=takeover}"] != 0 || v[suppressedSeries] != 1 {
			t.Fatalf("C11: no takeover, one suppressed pass: takeovers %v, suppressed %v",
				v["llm_d_epp_nca2_head_changes_total{reason=takeover}"], v[suppressedSeries])
		}
		f.pass(h, insider(f))
		if head, _ := f.doorHead(); head != "H" || metricValues(t, reg)[suppressedSeries] != 2 {
			t.Fatalf("C11: the next pass keeps H too and counts again (%q, %v)", head, metricValues(t, reg)[suppressedSeries])
		}
		// a2's turn ends and its seat becomes reclaimable: H completes its collection and is admitted, and c2
		// is the next head (H1), admitted in the same pass or collecting
		f.s.FirstChunk("rq-a2", f.now)
		f.s.Settle("rq-a2", "p0", f.now)
		f.wait(10 * time.Second)
		f.pass(h, insider(f))
		if !f.admitted("H") || f.doorPathOf("H") != doorPathHeadCollected {
			t.Fatalf("C11: H is admitted on its collection (admitted %v, path %q)", f.admitted("H"), f.doorPathOf("H"))
		}
		if head, _ := f.doorHead(); head != "c2" && !f.admitted("c2") {
			t.Fatalf("H1: c2 is the next head, got %q", head)
		}
	}
}

// TestC11_GroupCHeadIsStillTakenOver: C11 leaves a new session's head as H3 has it: a group-A outsider
// that cannot backfill takes the head from a collecting group-C head, with headKeepsTurn on.
func TestC11_GroupCHeadIsStillTakenOver(t *testing.T) {
	f, h := keepsTurnFixture(t, true, "C", 320)
	if head, pod := f.doorHead(); head != "H" || pod != "p0" || f.held("p0") == 0 {
		t.Fatalf("fixture: the group-C head collects on p0 (%q on %q, held %d)", head, pod, f.held("p0"))
	}
	reg := withDoorMetrics(t, f)
	f.pass(h, insider(f))
	v := metricValues(t, reg)
	if head, pod := f.doorHead(); head != "c2" || pod != "p0" || f.reservedHead("p0") != "c2" {
		t.Fatalf("H3: c2 takes the head and the blocks (%q on %q for %q)", head, pod, f.reservedHead("p0"))
	}
	if v["llm_d_epp_nca2_head_changes_total{reason=takeover}"] != 1 || v[suppressedSeries] != 0 {
		t.Fatalf("one takeover, nothing suppressed: %v", v)
	}
}

// TestC11_HeadNotCollectingIsStillTakenOver: C11 leaves a head that has not started collecting as H3 has
// it: a group-B head whose prompt fits no pod's CPU room (no collection pod is chosen) is taken over by a
// group-A outsider that cannot backfill, with headKeepsTurn on.
func TestC11_HeadNotCollectingIsStillTakenOver(t *testing.T) {
	f, h := keepsTurnFixture(t, true, "B", 2_000_000)
	if head, pod := f.doorHead(); head != "H" || pod != "" || f.held("p0") != 0 {
		t.Fatalf("fixture: the group-B head collects nowhere (%q on %q, held %d)", head, pod, f.held("p0"))
	}
	reg := withDoorMetrics(t, f)
	f.pass(h, insider(f))
	v := metricValues(t, reg)
	if head, pod := f.doorHead(); head != "c2" || pod != "p0" {
		t.Fatalf("H3: c2 takes the head and collects on p0 (%q on %q)", head, pod)
	}
	if v["llm_d_epp_nca2_head_changes_total{reason=takeover}"] != 1 || v[suppressedSeries] != 0 {
		t.Fatalf("one takeover, nothing suppressed: %v", v)
	}
}

// TestTDR6_HeadKeepsTurnUnderInterleavings: T-DR6 in Part B with headKeepsTurn, backfillReclaim off and on,
// with and without valveCPUWarm and noQuantum, over the door loop's own population and seeds and CPU tiers
// and over amend4Mix: every invariant of the door's loop holds (I7 as C11 amends it, and no takeover
// from a collecting group-A/B head), C11 keeps a head, and every session finishes (liveness).
func TestTDR6_HeadKeepsTurnUnderInterleavings(t *testing.T) {
	keptTurns, collected, backfills, yields := 0, 0, 0, 0
	for _, seed := range []int64{1, 7, 99, 4242, 20261115} {
		for _, tier := range []int64{56_000, 72_000, 150_000} {
			for _, reclaim := range []bool{false, true} {
				for _, options := range []bool{false, true} {
					tune := func(c *Config) {
						on, header := true, "x-llm-d-session-id"
						c.SessionAware, c.SessionHeader, c.HeadKeepsTurn = &on, &header, &on
						if reclaim {
							c.BackfillReclaim = &on
						}
						if options {
							amend3Tune(true, true)(c)
						}
					}
					for _, mix := range []doorMix{{sessionSize: 4, bigEvery: 6, heavy: tier >= 150_000}, amend4Mix} {
						_, probe, took := doorLoopMixed(t, seed, tier, 3000, tune, mix, nil)
						where := fmt.Sprintf("seed %d tier %d reclaim %v options %v mix %+v", seed, tier, reclaim, options, mix)
						if len(probe.violations) > 0 {
							t.Fatalf("%s: %v", where, probe.violations)
						}
						if took >= 3000 {
							t.Fatalf("%s: liveness: sessions still unfinished after 3000 passes", where)
						}
						keptTurns += probe.keptTurns
						collected += probe.collected
						backfills += probe.backfills
						yields += probe.yields
					}
				}
			}
		}
	}
	t.Logf("C11 kept the head in %d decisions; %d heads admitted after collecting, %d backfills, %d group-C yields",
		keptTurns, collected, backfills, yields)
	if keptTurns == 0 || collected == 0 || backfills == 0 || yields == 0 {
		t.Fatalf("fixture coverage: %d kept turns, %d head collections, %d backfills, %d group-C yields",
			keptTurns, collected, backfills, yields)
	}
}
