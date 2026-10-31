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

// E4a — SEAT = PIN (bindingMode "seat"). The tests here pin the four claims the
// arm rests on:
//
//  1. N=1 GOLDEN EQUALITY: with one engine, seat=pin must reproduce the committed
//     single-engine golden byte for byte. The seat model changes WHERE the engine
//     comes from and WHERE the pin lives; with one engine there is nothing left to
//     choose, so the decision stream must not move.
//  2. THE TWO-ENGINE GOLDEN: re-entry seats on a DIFFERENT engine when fit and
//     backlog say so, and the binding FOLLOWS the seat (created at seat, dropped
//     at exit, re-created wherever the next seat lands).
//  3. THE SEAT-TIME CHOICE: fit → least relative backlog per seat →
//     cheapest-victim engine.
//  4. THE ONE-VARIABLE DIFF vs E3: the two arms' configs differ in exactly one
//     field (bindingMode), everything else identical.

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

var updateE4aGolden = flag.Bool("update-e4a-golden", false, "rewrite testdata/golden-e4a.txt from the current run")

// e4aCfg is the E4a arm's config: the multipod base plus seat=pin.
func e4aCfg() Config {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	cfg.ValveMode = "on"
	cfg.BindingMode = "seat"
	return cfg
}

// TestSeatPinSingleEngineGoldenEquality is the N=1 gate: seat=pin with ONE
// discovered engine reproduces the committed single-mode golden byte for byte
// (the same gate per-engine mode passes, extended to the new binding model).
func TestSeatPinSingleEngineGoldenEquality(t *testing.T) {
	cfg := e4aCfg() // per-engine + seat; the scenario discovers exactly one engine
	got := runGoldenScenarioCfg(t, 2600, cfg, nil)
	want, err := os.ReadFile(filepath.Join("testdata", "golden.txt"))
	if err != nil {
		t.Fatalf("golden file missing: %v", err)
	}
	if got != string(want) {
		gotL := strings.Split(got, "\n")
		wantL := strings.Split(string(want), "\n")
		for i := 0; i < len(gotL) || i < len(wantL); i++ {
			var a, b string
			if i < len(gotL) {
				a = gotL[i]
			}
			if i < len(wantL) {
				b = wantL[i]
			}
			if a != b {
				t.Fatalf("SEAT=PIN (1 engine) diverges from the single-engine golden at line %d:\n  seat:     %s\n  golden:   %s", i+1, a, b)
			}
		}
		t.Fatal("SEAT=PIN (1 engine) diverges from the single-engine golden (length)")
	}
}

// e4aMember is one scripted member in the E4a scenario.
type e4aMember struct {
	id       string
	chain    []Hash
	turn     int
	nextTurn int
}

func (m *e4aMember) blocks() int { return 8 + m.turn/3 }

// e4aScenario drives the SEAT=PIN scheduler against TWO engines of UNEQUAL size
// on the virtual clock. The asymmetry is the point: engine-a is small enough to
// saturate and rotate, engine-b is large, so a member whose seat on engine-a was
// evicted re-enters into a pool where the fit/backlog rule prefers engine-b —
// which is exactly the behaviour the arm must exhibit, and which member-stable
// (E3) cannot exhibit by construction (a homed member never moves while active,
// and a silent one returns to the same engine).
//
// Trace lines the golden pins:
//
//	SEAT   id seat=<engine> kind=<newcomer|re-entry>   the seat AND its engine
//	BOUND  id engine=<engine>                          bindingOf() == seat engine
//	RESEAT id prev=<engine> seat=<engine>              the SAME member seated on a
//	                                                   DIFFERENT engine than its
//	                                                   previous seat
//	EXIT   id reason=<r> engine=<e>                    the seat (and the binding) end
//	UNBOUND id                                         bindingOf() == "" after exit
func e4aScenario(t *testing.T, simEnd int) (string, Snapshot, *Scheduler) {
	t.Helper()
	s, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	// The two engines are the SAME size — the campaign's real shape (a
	// heterogeneous pool is an alarm: capacity_heterogeneous must read 0). The
	// seat-time rule therefore differentiates on mass (free budget) and on the
	// member's own queued demand, never on a capacity advantage; the RESEAT
	// behaviour itself is pinned by TestSeatPinReseatOnDifferentEngine, where it
	// can be engineered deterministically instead of waiting for the scenario to
	// produce it.
	if err := s.SetNumBlocksFor("engine-a", 64, 900); err != nil { // budget 855
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 900); err != nil { // budget 855
		t.Fatal(err)
	}

	var sb strings.Builder
	evs := map[string]string{} // member -> the engine of its LAST seat
	prevSeat := map[string]string{}
	s.onEvent = func(ev Event) {
		switch ev.Kind {
		case "SEAT":
			sb.WriteString(fmt.Sprintf("  SEAT id=%s engine=%s %s\n", ev.ID, ev.Engine, ev.Detail))
			// the seat=pin fact the arm exists to show: a member's NEXT seat may be
			// on a different engine than its previous one (member-stable cannot do
			// this — a homed member never moves while active, and a silent one
			// returns to the same engine).
			if prev := prevSeat[ev.ID]; prev != "" && prev != ev.Engine {
				sb.WriteString(fmt.Sprintf("  RESEAT id=%s prev=%s seat=%s\n", ev.ID, prev, ev.Engine))
			}
			prevSeat[ev.ID] = ev.Engine
		case "EXIT":
			sb.WriteString(fmt.Sprintf("  EXIT id=%s engine=%s %s\n", ev.ID, ev.Engine, ev.Detail))
		}
		evs[ev.ID] = ev.Engine
	}

	const nMembers = 12
	ms := make([]*e4aMember, 0, nMembers)
	for i := 0; i < nMembers; i++ {
		ms = append(ms, &e4aMember{id: fmt.Sprintf("m%02d", i),
			chain: keys(6, Hash(5000+i)), nextTurn: 5 + i})
	}
	type flight struct {
		m        *e4aMember
		reqID    string
		engine   string
		settleAt int
	}
	var flying []*flight
	reqSeq := map[string]int{}
	nextReqID := func(id string) string {
		reqSeq[id]++
		return fmt.Sprintf("%s-r%02d", id, reqSeq[id])
	}
	headsAt := func(sim int) []Head {
		var hs []Head
		for _, m := range ms {
			if m.nextTurn <= sim {
				hs = append(hs, Head{Member: m.id,
					ReqID:     fmt.Sprintf("%s-r%02d", m.id, reqSeq[m.id]+1),
					Arrival:   goldenT0.Add(time.Duration(m.nextTurn) * time.Second),
					SizeBytes: bytesFor(len(m.chain) + m.blocks())})
			}
		}
		return hs
	}

	for sim := 0; sim <= simEnd; sim++ {
		now := goldenT0.Add(time.Duration(sim) * time.Second)
		var still []*flight
		for _, f := range flying {
			if f.settleAt > sim {
				still = append(still, f)
				continue
			}
			s.Chunk(f.m.id)
			s.SettleOn(f.m.id, f.reqID, f.engine, struct{ Prompt, Cached int }{Prompt: 100, Cached: 96}, true, now)
		}
		flying = still

		s.Tick(now)

		for pick := 0; pick < 2; pick++ {
			hs := headsAt(sim)
			if len(hs) == 0 {
				break
			}
			engine, i := s.PickFromHeadsEngine(hs, now)
			if i < 0 {
				break
			}
			h := hs[i]
			var m *e4aMember
			for _, cand := range ms {
				if cand.id == h.Member {
					m = cand
				}
			}
			if m == nil {
				t.Fatalf("picked an unknown member %q", h.Member)
			}
			reqID := nextReqID(m.id)
			m.turn++
			m.chain = append(m.chain, keys(m.blocks(), Hash(9100+m.turn*17+len(m.chain)))...)
			// THE SEAT=PIN INVARIANT, checked on every dispatch: a dispatched
			// member is seated, and its binding IS the engine Pick served from.
			// (An unseated member has no binding at all — seat = pin.)
			if got := s.bindingOf(m.id); got != engine {
				t.Fatalf("t=%d member %s dispatched on %q but bound to %q (seat=pin violated)", sim, m.id, engine, got)
			}
			sb.WriteString(fmt.Sprintf("t=%d PICK id=%s engine=%s blocks=%d\n", sim, m.id, engine, len(m.chain)))
			sb.WriteString(fmt.Sprintf("  BOUND id=%s engine=%s\n", m.id, s.bindingOf(m.id)))
			s.DispatchedOn(m.id, reqID, engine, append([]Hash(nil), m.chain...), h.SizeBytes, now)
			flying = append(flying, &flight{m: m, reqID: reqID, engine: engine,
				settleAt: sim + 4 + (m.turn*3)%9})
			m.nextTurn = sim + 3 + (m.turn*7)%13
		}
	}

	snap := s.Snapshot()
	sb.WriteString(fmt.Sprintf("SNAPSHOT served=%d violations=%d rebinds=%d waiters=%d seats=%d\n",
		snap.Served, snap.AffinityViolations, snap.Rebinds, snap.Waiters, snap.Seats))
	labels := make([]string, 0, len(snap.Engines))
	for l := range snap.Engines {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	for _, l := range labels {
		es := snap.Engines[l]
		sb.WriteString(fmt.Sprintf("ENGINE %s seats=%d ne=%d budget=%d bound=%d served=%d\n",
			l, es.Seats, es.NEBlocks, es.BudgetBlocks, es.BoundMembers, es.Served))
		reasons := make([]string, 0, len(es.Exits))
		for r := range es.Exits {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			sb.WriteString(fmt.Sprintf("ENGINE-EXITS %s %s=%d\n", l, r, es.Exits[r]))
		}
	}
	return sb.String(), snap, s
}

// TestSeatPinMultiEngineGolden pins the two-engine seat=pin trace.
func TestSeatPinMultiEngineGolden(t *testing.T) {
	got, snap, _ := e4aScenario(t, 400)
	path := filepath.Join("testdata", "golden-e4a.txt")
	if *updateE4aGolden {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("E4a golden rewritten: %d bytes", len(got))
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("E4a golden missing (run with -update-e4a-golden): %v", err)
	}
	if got != string(want) {
		gotL := strings.Split(got, "\n")
		wantL := strings.Split(string(want), "\n")
		for i := 0; i < len(gotL) || i < len(wantL); i++ {
			var a, b string
			if i < len(gotL) {
				a = gotL[i]
			}
			if i < len(wantL) {
				b = wantL[i]
			}
			if a != b {
				t.Fatalf("the E4a trace diverges at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
			}
		}
		t.Fatal("the E4a trace diverges (length)")
	}
	// and the pin must hold: no member is charged to an engine it is not seated on.
	if snap.AffinityViolations != 0 {
		t.Fatalf("affinity violations = %d in a seat=pin run with no injected violation, want 0", snap.AffinityViolations)
	}
	// every engine that served should have served somebody (the split is real).
	if snap.Engines["engine-a"].Served == 0 || snap.Engines["engine-b"].Served == 0 {
		t.Fatalf("expected both engines to serve, got %+v", snap.Engines)
	}
}

// TestSeatPinBindingDroppedAtExit: seat = pin. The binding exists while the
// seat does and is gone the instant the member exits — no TTL, no residue.
func TestSeatPinBindingDroppedAtExit(t *testing.T) {
	s, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	heads := []Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(40)}}
	if _, i := s.PickFromHeadsEngine(heads, now); i < 0 {
		t.Fatal("the newcomer was not seated")
	}
	engine := s.bindingOf("m1")
	if engine == "" {
		t.Fatal("a seated member must be bound to its seat's engine")
	}
	s.DispatchedOn("m1", "r1", engine, keys(20, Hash(1)), bytesFor(40), now)
	// the member goes quiet long enough for the TTL class, and the engine has
	// pressure: an exit must drop the binding IMMEDIATELY (not after bindingTtl).
	s.SettleOn("m1", "r1", engine, struct{ Prompt, Cached int }{Prompt: 100, Cached: 100}, true, now.Add(time.Second))
	later := now.Add(10 * time.Hour)
	s.Tick(later)
	if got := s.bindingOf("m1"); got != "" {
		t.Fatalf("binding after exit = %q, want \"\" (seat=pin: the binding is dropped at exit)", got)
	}
	if n := len(s.bindingsCopy()); n != 0 {
		t.Fatalf("%d bindings survive, want 0", n)
	}
}

// TestSeatPinGlobalWaiterPool: unseated demand lives in ONE pool that no engine
// owns, so a head can be seated by EITHER engine (in member-stable it is
// partitioned at first sight and only the homed engine may seat it).
func TestSeatPinGlobalWaiterPool(t *testing.T) {
	s, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	heads := []Head{
		{Member: "w1", ReqID: "w1-r1", Arrival: now, SizeBytes: bytesFor(40)},
		{Member: "w2", ReqID: "w2-r1", Arrival: now.Add(time.Second), SizeBytes: bytesFor(40)},
	}
	// sync + rotate WITHOUT dispatching: both heads become waiters, in the pool.
	s.mu.Lock()
	s.deriveLocked()
	s.syncWaitersLocked(heads, now)
	pooled := len(s.gWaiters)
	perEngine := len(s.groups["engine-a"].waiters) + len(s.groups["engine-b"].waiters)
	s.mu.Unlock()
	if pooled != 2 {
		t.Fatalf("global pool size = %d, want 2", pooled)
	}
	if perEngine != 0 {
		t.Fatalf("per-engine waiter maps hold %d entries in seat=pin mode, want 0", perEngine)
	}
	// both members get seated and are pinned to whichever engine took them.
	s.mu.Lock()
	s.rotateLocked(now)
	s.mu.Unlock()
	if len(s.gWaiters) != 0 {
		t.Fatalf("pool still holds %d waiter(s) after rotate", len(s.gWaiters))
	}
	for _, id := range []string{"w1", "w2"} {
		if s.bindingOf(id) == "" {
			t.Fatalf("%s was seated but carries no binding", id)
		}
	}
}

// TestSeatPinSeatChoiceLadder is the seat-time rule, level by level:
//
//	fit → least own queued demand per seat → MORE free budget → fewer seats → id.
//
// The free-budget key is the reviewer's R1: a seat COUNT balances member count
// while mass is what the KV budget holds, and count-vs-mass is the measured
// asymmetry (E1 0.977/0.683 with symmetric logical splits).
func TestSeatPinSeatChoiceLadder(t *testing.T) {
	s, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 400); err != nil { // budget 380
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 4000); err != nil { // budget 3800
		t.Fatal(err)
	}
	now := goldenT0
	w := &waiter{id: "m1", since: now, sizeBytes: bytesFor(60), coldTokens: 60 * 64,
		coldTokensByEngine: map[string]int64{"engine-a": 60 * 64, "engine-b": 60 * 64}}

	// (a) FREE BUDGET decides first: both idle, engine-b has far more mass room.
	if c := s.seatEngineLocked(w, now); c == nil || !c.fits {
		t.Fatalf("expected a fitting candidate, got %+v", c)
	} else if c.g.id != "engine-b" {
		t.Fatalf("free-budget tie-break chose %q, want engine-b (3800 free vs 380)", c.g.id)
	}

	// (b) BACKLOG outranks free budget: give engine-b a resident with a queued head.
	s.mu.Lock()
	gb := s.groups["engine-b"]
	if ok, _ := gb.index.TrySeatLocked("b0", gb.budgetBlocks); !ok {
		s.mu.Unlock()
		t.Fatal("could not seat b0")
	}
	gb.members["b0"] = &member{id: "b0", engine: "engine-b", seatTime: now, lastActivity: now, waitingSince: now}
	c := s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c == nil || c.g.id != "engine-a" {
		t.Fatalf("a queued resident on engine-b must hand the seat to engine-a, got %+v", c)
	}

	// (c) ID ORDER when everything else ties.
	eq, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := eq.SetNumBlocksFor(id, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	if c := eq.seatEngineLocked(w, now); c == nil || c.g.id != "engine-a" {
		t.Fatalf("a total tie must break on engine id (engine-a), got %+v", c)
	}

	// (d) FIT PREFERENCE: a head too big for engine-a must skip it, whatever the
	// other keys say.
	big := &waiter{id: "big", since: now, sizeBytes: bytesFor(12000),
		coldTokensByEngine: map[string]int64{"engine-a": 2000 * 64, "engine-b": 2000 * 64}}
	s.mu.Lock()
	cb := s.seatEngineLocked(big, now)
	s.mu.Unlock()
	if cb == nil || cb.g.id != "engine-b" || !cb.fits {
		t.Fatalf("a head that only engine-b can hold must be placed there, got %+v", cb)
	}
}

// TestSeatPinSeatChoiceCheapestVictim is the rule's third level: when NOTHING
// fits, the choice falls to an engine that can evict an eligible incumbent.
func TestSeatPinSeatChoiceCheapestVictim(t *testing.T) {
	cfg := e4aCfg()
	cfg.RotationMode = "quantum"
	cfg.QuantumBusyPeriods = 4
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 400); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 400); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	s.mu.Lock()
	for _, id := range []string{"engine-a", "engine-b"} {
		g := s.groups[id]
		for i := 0; i < 3; i++ {
			mid := fmt.Sprintf("%s-m%d", id, i)
			if ok, _ := g.index.TrySeatLocked(mid, g.budgetBlocks); !ok {
				s.mu.Unlock()
				t.Fatalf("could not fill %s", id)
			}
			// quantum-complete: rotation-eligible by the tenure bound
			g.members[mid] = &member{id: mid, engine: id, seatTime: now, lastActivity: now,
				busyPeriods: 4, resTurns: 4}
		}
	}
	// a head NOTHING can hold: both engines are at budget and the head is huge
	w := &waiter{id: "huge", since: now, sizeBytes: bytesFor(40000), coldTokens: 40000 * 64}
	c := s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c == nil {
		t.Fatal("expected a cheapest-victim candidate, got nil")
	}
	if c.fits {
		t.Fatalf("expected no fitting engine for the oversized head, got %+v", c)
	}
	if !c.hasVictim {
		t.Fatalf("the chosen engine %q holds no rotation victim, so the head could never seat", c.g.id)
	}
}

// TestE4aVsE3OneVariableDiff is the arm-identity gate: E3's config and E4a's
// config differ in EXACTLY ONE field. Anything else that differs is an
// undeclared variable in a one-variable contrast — the failure class the wave-1
// audit found twice (B2's token-producer, B1's missing pinning).
func TestE4aVsE3OneVariableDiff(t *testing.T) {
	e3 := defaultConfig()
	e3.EngineGroups = "per-engine"
	e3.ValveMode = "on"
	e3.MemberKey = "auto"
	e3.BindingMode = "member-stable"

	e4a := e3
	e4a.BindingMode = "seat"

	a := reflect.ValueOf(e3)
	b := reflect.ValueOf(e4a)
	if a.NumField() != b.NumField() {
		t.Fatal("config struct field count changed under the diff")
	}
	var differing []string
	for i := 0; i < a.NumField(); i++ {
		name := a.Type().Field(i).Name
		if !reflect.DeepEqual(a.Field(i).Interface(), b.Field(i).Interface()) {
			differing = append(differing, name)
		}
	}
	if len(differing) != 1 || differing[0] != "BindingMode" {
		t.Fatalf("E3 vs E4a differ in %v, want exactly [BindingMode]", differing)
	}

	// the BEHAVIOURAL half of the one-variable claim: the same scenario really
	// does move. Member-stable partitions the waiters per engine and homes at
	// first sight; seat=pin keeps one pool and pins only at the seat.
	heads := []Head{{Member: "m1", ReqID: "r1", Arrival: goldenT0, SizeBytes: bytesFor(40)}}
	ms := func(cfg Config) (bindingsBeforeSeat int, pooled int, perEngine int) {
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
			t.Fatal(err)
		}
		if err := s.SetNumBlocksFor("engine-b", 64, 1000); err != nil {
			t.Fatal(err)
		}
		s.mu.Lock()
		s.syncWaitersLocked(heads, goldenT0)
		bindingsBeforeSeat = len(s.bindings)
		pooled = len(s.gWaiters)
		for _, g := range s.groups {
			perEngine += len(g.waiters)
		}
		s.mu.Unlock()
		return
	}
	if got, _, per := ms(e3); got != 1 || per != 1 {
		t.Fatalf("E3 (member-stable): %d binding(s) before any seat and %d partitioned waiter(s), want 1 and 1", got, per)
	}
	if got, pooled, per := ms(e4a); got != 0 || pooled != 1 || per != 0 {
		t.Fatalf("E4a (seat): %d binding(s) before any seat, %d pooled and %d partitioned waiter(s), want 0, 1, 0", got, pooled, per)
	}
}

// TestSeatPinReseatOnDifferentEngine is the seat=pin fact member-stable cannot
// produce: a member's NEXT seat may be on a different engine than its previous
// one, because the binding is the SEAT and nothing else. (In member-stable the
// engine is chosen at first sight and kept for bindingTtlSeconds, so a returning
// member lands where it was homed.)
func TestSeatPinReseatOnDifferentEngine(t *testing.T) {
	s, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 900); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 900); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	heads := func(since time.Time) []Head {
		return []Head{{Member: "m1", ReqID: "r", Arrival: since, SizeBytes: bytesFor(40)}}
	}
	// first seat: both engines are empty, the id order decides → engine-a
	if eng, i := s.PickFromHeadsEngine(heads(now), now); i < 0 || eng != "engine-a" {
		t.Fatalf("first seat engine = %q (i=%d), want engine-a", eng, i)
	}
	first := s.bindingOf("m1")
	s.DispatchedOn("m1", "r", first, keys(20, Hash(1)), bytesFor(40), now)
	s.SettleOn("m1", "r", first, struct{ Prompt, Cached int }{Prompt: 100, Cached: 90}, true, now.Add(time.Second))
	// burn engine-a's free budget so the seat-time rule must choose engine-b
	s.mu.Lock()
	ga := s.groups["engine-a"]
	for i := 0; i < 12; i++ {
		id := fmt.Sprintf("fill%d", i)
		if ok, _ := ga.index.TrySeatLocked(id, ga.budgetBlocks); !ok {
			s.mu.Unlock()
			t.Fatalf("could not fill engine-a at %d", i)
		}
		ga.members[id] = &member{id: id, engine: "engine-a", seatTime: now, lastActivity: now}
	}
	s.mu.Unlock()
	// m1 is still SEATED (its record has not expired) — force the exit the way the
	// policy does, then let it re-enter.
	s.mu.Lock()
	if m, ok := ga.members["m1"]; ok {
		m.waitingSince = now.Add(2 * time.Second) // it has a turn queued
		s.exitLocked(ga, m, "paid", now.Add(2*time.Second))
	}
	s.mu.Unlock()
	if got := s.bindingOf("m1"); got != "" {
		t.Fatalf("binding after exit = %q, want \"\"", got)
	}
	later := now.Add(3 * time.Second)
	eng, i := s.PickFromHeadsEngine(heads(now.Add(2*time.Second)), later)
	if i < 0 {
		t.Fatal("the re-entering member was not seated")
	}
	if eng != "engine-b" {
		t.Fatalf("re-entry seat engine = %q, want engine-b (engine-a exhausted its free budget; seat=pin means the re-entry is placed afresh)", eng)
	}
	if got := s.bindingOf("m1"); got != "engine-b" {
		t.Fatalf("binding after RE-SEAT = %q, want engine-b — the binding must follow the seat", got)
	}
}
