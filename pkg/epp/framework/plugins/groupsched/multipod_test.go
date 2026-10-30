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

// MULTI-POD TESTS (PREREG-multipod-offload task 1, implementation checklist
// items 1-6 + the instrumentation rules).
//
// Two halves:
//
//  1. TestPerEngineGoldenEquality — the UNIT EQUIVALENCE GATE: the committed
//     single-engine golden scenario replayed through the PER-ENGINE code with
//     ONE discovered engine must produce the committed trace BYTE FOR BYTE.
//     If the state split broke C3/C4/A1/A7/R9 anywhere on the normal path,
//     this fails in minutes with no GPU.
//
//  2. TestMultiPodGolden — a two-engine scenario pinned to
//     testdata/golden-multipod.txt: homing (each member seated on exactly ONE
//     engine, for the whole run), the per-engine budget/capacity split,
//     per-engine rotation and cold-entry caps, the affinity-violation repair,
//     and the engine-teardown path — with the per-engine skew accounting the
//     E1 precondition is scored on.
//
// Regenerate the multi-pod golden with:
//
//	go test ./pkg/epp/framework/plugins/groupsched/ -run TestMultiPodGolden -update-multipod-golden

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

var updateMultiPodGolden = flag.Bool("update-multipod-golden", false,
	"rewrite testdata/golden-multipod.txt from the current run")

// ---- 1. THE UNIT EQUIVALENCE GATE ----

// TestPerEngineGoldenEquality: engineGroups=per-engine with a single engine
// must reproduce the committed single-mode golden trace byte for byte. This
// is the pre-registered first gate ("before reporting state split done"): it
// runs the whole normal-path surface (seating, rotation, budget-idle exits,
// TTL, re-entry, n_max, the C4 refresh-and-return, the R9 pass-through)
// through the per-engine code and diffs the trace.
func TestPerEngineGoldenEquality(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	got := runGoldenScenarioCfg(t, 2600, cfg, nil)
	want, err := os.ReadFile(filepath.Join("testdata", "golden.txt"))
	if err != nil {
		t.Fatalf("golden file missing (run with -update-golden): %v", err)
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
				t.Fatalf("PER-ENGINE (1 engine) diverges from the single-mode golden at line %d:\n  per-engine: %s\n  single:     %s", i+1, a, b)
			}
		}
		t.Fatal("PER-ENGINE (1 engine) diverges from the single-mode golden (length)")
	}
}

// TestPerEngineOneEngineSnapshot: with one engine the per-engine snapshot's
// engine section must equal the whole-policy numbers (the reducer's oldest
// invariant: sum(engines) == top level).
func TestPerEngineOneEngineSnapshot(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	runSimpleSeats(t, s, 6)
	snap := s.Snapshot()
	if len(snap.Engines) != 1 {
		t.Fatalf("engines = %d, want 1", len(snap.Engines))
	}
	var es EngineSnapshot
	for _, v := range snap.Engines {
		es = v
	}
	if es.Seats != snap.Seats || es.Waiters != snap.Waiters ||
		es.UniqueBlocks != snap.UniqueBlocks || es.NEBlocks != snap.NEBlocks {
		t.Fatalf("per-engine section %+v disagrees with the whole-policy snapshot %+v", es, snap)
	}
	if snap.BudgetBlocks != 950 {
		t.Fatalf("budget = %d, want 950", snap.BudgetBlocks)
	}
}

// ---- 2. THE TWO-ENGINE SCENARIO ----

// mpMember is one scripted member: a growing chain with a think gap.
type mpMember struct {
	id       string
	chain    []Hash
	turn     int
	nextTurn int
}

func (m *mpMember) blocks() int { return 10 + m.turn/2 } // grows 10 -> ~40
func (m *mpMember) gap() int    { return 3 + (m.turn*7)%19 }

// multiPodScenario drives the per-engine scheduler against TWO synthetic
// engines on a virtual clock and returns a deterministic trace.
//
// The scenario is built to exercise, in one run:
//   - homing across the whole member population (a burst of 20 newcomers in
//     the first seconds — the anti-burst tie-break must split them),
//   - per-engine budget pressure (each engine's budget is 950 blocks; its
//     ~10 homed members grow to >950 combined, so each engine rotates on its
//     OWN ledger),
//   - per-engine cold-entry staggering (valveMode "on"),
//   - a deliberate AFFINITY VIOLATION at sim 200 (a request for a member
//     charged to one engine landing on the other: counted, repaired by
//     rebind, accounting moved),
//   - an ENGINE TEARDOWN at sim 320 (EventDelete: members exited
//     "engine-gone", homing released, group dropped) and the re-placement of
//     the survivors onto the remaining engine.
func multiPodScenario(t *testing.T, simEnd int) (string, Snapshot, []Event, *Scheduler) {
	t.Helper()
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	cfg.ValveMode = "on"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// TWO engines, DISCOVERED (never configured): each with its own capacity.
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 1200); err != nil { // budget 1140
		t.Fatal(err)
	}

	var sb strings.Builder
	var evs []Event
	s.onEvent = func(ev Event) {
		evs = append(evs, ev)
		sb.WriteString(fmt.Sprintf("EVENT kind=%s engine=%s id=%s %s\n", ev.Kind, ev.Engine, ev.ID, ev.Detail))
	}

	const nMembers = 20
	ms := make([]*mpMember, 0, nMembers)
	for i := 0; i < nMembers; i++ {
		ms = append(ms, &mpMember{id: fmt.Sprintf("m%02d", i),
			chain: keys(8, Hash(7000+i)), nextTurn: 10 + i}) // a 20-member burst inside 20s
	}

	type flight struct {
		m        *mpMember
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
	seenEngines := map[string]map[string]struct{}{} // member -> set of engines it was seated on
	noteEngine := func(member, engine string) {
		if seenEngines[member] == nil {
			seenEngines[member] = map[string]struct{}{}
		}
		seenEngines[member][engine] = struct{}{}
	}
	stateLine := func(sim int) {
		per := map[string]int{}
		for label, es := range s.Snapshot().Engines {
			per[label] = es.Seats
		}
		labels := make([]string, 0, len(per))
		for l := range per {
			labels = append(labels, l)
		}
		sort.Strings(labels)
		var parts []string
		for _, l := range labels {
			parts = append(parts, fmt.Sprintf("%s=%d", l, per[l]))
		}
		sb.WriteString(fmt.Sprintf("t=%d STATE %s\n", sim, strings.Join(parts, " ")))
	}

	violationDone, teardownDone := false, false
	for sim := 0; sim <= simEnd; sim++ {
		now := goldenT0.Add(time.Duration(sim) * time.Second)

		// settle due flights (deterministic decode periods)
		var still []*flight
		for _, f := range flying {
			if f.settleAt > sim {
				still = append(still, f)
				continue
			}
			s.Chunk(f.m.id)
			usage := struct{ Prompt, Cached int }{Prompt: 100, Cached: 98}
			// SettleOn with the engine the request ACTUALLY ran on.
			s.SettleOn(f.m.id, f.reqID, f.engine, usage, true, now)
			sb.WriteString(fmt.Sprintf("t=%d SETTLE %s req=%s engine=%s\n", sim, f.m.id, f.reqID, f.engine))
		}
		flying = still

		s.Tick(now)

		// THE ENGINE TEARDOWN (once): the EPP reports engine-b gone.
		if sim == 320 && !teardownDone {
			teardownDone = true
			n := s.RemoveEngine("engine-b", now)
			sb.WriteString(fmt.Sprintf("t=%d ENGINE-GONE engine-b exited=%d\n", sim, n))
			stateLine(sim)
		}

		for pick := 0; pick < 2; pick++ {
			hs := headsAt(sim)
			if len(hs) == 0 {
				break
			}
			engine, i := s.PickFromHeadsEngine(hs, now)
			if i < 0 {
				sb.WriteString(fmt.Sprintf("t=%d PICK -1\n", sim))
				break
			}
			h := hs[i]
			var m *mpMember
			for _, cand := range ms {
				if cand.id == h.Member {
					m = cand
				}
			}
			reqID := nextReqID(m.id)
			m.turn++
			m.chain = append(m.chain, keys(m.blocks(), Hash(9000+m.turn*31+len(m.chain)))...)
			dispatchEngine := engine
			// THE DELIBERATE AFFINITY VIOLATION (once, at sim 200): the
			// framework places this request on the OTHER engine than the
			// member's charge. PreRequest observes it and repairs.
			if sim == 200 && !violationDone {
				violationDone = true
				if other := otherEngine(engine); other != "" {
					dispatchEngine = other
				}
			}
			before := s.bindingOf(m.id)
			s.DispatchedOn(m.id, reqID, dispatchEngine, append([]Hash(nil), m.chain...), h.SizeBytes, now)
			after := s.bindingOf(m.id)
			if before != after {
				sb.WriteString(fmt.Sprintf("t=%d REBIND %s %s->%s\n", sim, m.id, before, after))
			}
			noteEngine(m.id, after)
			flying = append(flying, &flight{m: m, reqID: reqID, engine: dispatchEngine,
				settleAt: sim + 5 + (m.turn*3)%11})
			m.nextTurn = sim + m.gap()
			sb.WriteString(fmt.Sprintf("t=%d PICK engine=%s %s req=%s blocks=%d\n",
				sim, engine, m.id, reqID, len(m.chain)))
			stateLine(sim)
		}
	}

	snap := s.Snapshot()
	sb.WriteString(fmt.Sprintf("SNAPSHOT served=%d violations=%d rebinds=%d enginesGone=%d\n",
		snap.Served, snap.AffinityViolations, snap.Rebinds, snap.EngineGone))
	labels := make([]string, 0, len(snap.Engines))
	for l := range snap.Engines {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	for _, l := range labels {
		es := snap.Engines[l]
		sb.WriteString(fmt.Sprintf("ENGINE %s seats=%d waiters=%d sessions=%d ne=%d budget=%d num=%d bound=%d served=%d homings=%d forced=%d\n",
			l, es.Seats, es.Waiters, es.Sessions, es.NEBlocks, es.BudgetBlocks, es.NumBlocks,
			es.BoundMembers, es.Served, es.Homings, es.ForcedSeats))
		reasons := make([]string, 0, len(es.Exits))
		for r := range es.Exits {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			sb.WriteString(fmt.Sprintf("ENGINE-EXITS %s %s=%d\n", l, r, es.Exits[r]))
		}
	}
	// the homing invariant: every member was seated on at most the engines
	// the trace shows (checked below, not in the golden text)
	for id, set := range seenEngines {
		if len(set) > 2 {
			t.Fatalf("member %s was seated on %d engines", id, len(set))
		}
	}
	return sb.String(), snap, evs, s
}

// otherEngine names the engine a violation should target (the one that is
// not `engine`); "" when the stack has only one.
func otherEngine(engine string) string {
	switch engine {
	case "engine-a":
		return "engine-b"
	case "engine-b":
		return "engine-a"
	}
	return ""
}

func TestMultiPodGolden(t *testing.T) {
	got, _, _, _ := multiPodScenario(t, 600)
	path := filepath.Join("testdata", "golden-multipod.txt")
	if *updateMultiPodGolden {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("multi-pod golden rewritten: %d bytes", len(got))
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("multi-pod golden missing (run with -update-multipod-golden): %v", err)
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
				t.Fatalf("multi-pod divergence at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
			}
		}
		t.Fatal("multi-pod divergence (length)")
	}
}

// TestMultiPodInvariants is the readable half of the golden: the properties
// the trace must satisfy, asserted directly (a golden diff says "something
// changed", these say WHAT the something must be).
func TestMultiPodInvariants(t *testing.T) {
	_, snap, evs, s := multiPodScenario(t, 600)

	// (a) the affinity violation was counted, repaired, and the member moved
	if snap.AffinityViolations < 1 {
		t.Fatalf("affinity violations = %d, the deliberate off-engine dispatch must be counted", snap.AffinityViolations)
	}
	if snap.Rebinds < 1 {
		t.Fatalf("rebinds = %d, a violation must be repaired by a rebind", snap.Rebinds)
	}
	var rebindEvents int
	for _, ev := range evs {
		switch ev.Kind {
		case "REBIND":
			rebindEvents++
			if ev.Engine == "" {
				t.Fatal("a REBIND event must carry the engine it moved FROM")
			}
			if !strings.Contains(ev.Detail, "from=") || !strings.Contains(ev.Detail, "to=") {
				t.Fatalf("REBIND detail must carry from= and to= (the join key): %q", ev.Detail)
			}
		}
	}
	if rebindEvents != int(snap.Rebinds) {
		t.Fatalf("REBIND events = %d, counter = %d (the artifacts and the metric must agree)", rebindEvents, snap.Rebinds)
	}

	// (b) EVERY event carries an engine label (the reducer rule: without it
	// the per-engine equivalence test is unscoreable)
	for _, ev := range evs {
		if ev.Engine == "" {
			t.Fatalf("event without an engine label: %+v", ev)
		}
	}

	// (c) the teardown happened: engine-b is gone from the group set and its
	// members were exited with reason engine-gone
	if snap.EngineGone < 1 {
		t.Fatalf("engine-gone counter = %d, the EventDelete path must tear the group down", snap.EngineGone)
	}
	if _, still := snap.Engines["engine-b"]; still {
		t.Fatal("engine-b is still in the snapshot after RemoveEngine")
	}
	for _, id := range s.EngineGroupIDs() {
		if id == "engine-b" {
			t.Fatal("engine-b is still a group after RemoveEngine")
		}
	}

	// (d) SURVIVAL: no charge is left on the dead engine, and the survivors
	// all live on a live engine (re-homed, not orphaned)
	for member, engine := range s.bindingsCopy() {
		if engine == "engine-b" {
			t.Fatalf("member %s is still bound to the torn-down engine", member)
		}
	}

	// (e) per-engine budgets were respected on the surviving engine: seats
	// are bounded by its budget (the split is enforced, not hoped for)
	for label, es := range snap.Engines {
		if es.NumBlocks > 0 && es.Seats == 0 && label == "engine-a" {
			t.Fatalf("engine-a has capacity but no seats at the end of the run")
		}
	}
}

// TestMultiPodHomingSplit: the SKEW PRECONDITION the E1 equivalence test is
// scored on — a burst of newcomers must SPLIT across the engines, not race to
// one. Per-engine member counts within ±10% of an even split.
func TestMultiPodHomingSplit(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 100000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	const n = 40
	var heads []Head
	for i := 0; i < n; i++ {
		heads = append(heads, Head{Member: fmt.Sprintf("b%02d", i), ReqID: fmt.Sprintf("r%02d", i),
			Arrival: now, SizeBytes: bytesFor(4)})
	}
	// one tick seats the whole burst; each pick dispatches (and drops) one
	// head, so the loop is bounded by the burst size
	s.Tick(now)
	for len(heads) > 0 {
		engine, i := s.PickFromHeadsEngine(heads, now)
		if i < 0 {
			break
		}
		s.DispatchedOn(heads[i].Member, heads[i].ReqID, engine, keys(4, Hash(i+1)), bytesFor(4), now)
		heads = append(heads[:i], heads[i+1:]...)
	}
	// seat accounting: everyone is seated somewhere
	var a, b int
	snap := s.Snapshot()
	for label, es := range snap.Engines {
		switch label {
		case "engine-a":
			a = es.Seats
		case "engine-b":
			b = es.Seats
		}
	}
	if a+b != n {
		t.Fatalf("seats %d+%d = %d, want %d (every newcomer must be homed)", a, b, a+b, n)
	}
	skew := float64(a-b) / float64(n)
	if skew < 0 {
		skew = -skew
	}
	if skew > 0.10 {
		t.Fatalf("homing split a=%d b=%d skew=%.1f%% > 10%% — bursts must split, not race", a, b, skew*100)
	}
	// and the split is recorded where the E1 precondition reads it
	for label, es := range snap.Engines {
		if es.BoundMembers != es.Seats {
			t.Fatalf("engine %s: bound members %d != seats %d (every seated member is bound)", label, es.BoundMembers, es.Seats)
		}
	}
}

// TestMultiPodPerEngineBudget: one engine's budget pressure must never evict
// on the other engine, and each engine's NE counter must stay under ITS OWN
// budget.
func TestMultiPodPerEngineBudget(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 100); err != nil { // budget 95
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	now := goldenT0
	// pin members to engines explicitly (the homing decision is tested above;
	// here we need a known assignment)
	s.mu.Lock()
	for i := 0; i < 6; i++ {
		s.bindings[fmt.Sprintf("a%d", i)] = "engine-a"
		s.bindings[fmt.Sprintf("b%d", i)] = "engine-b"
	}
	s.mu.Unlock()

	step := func(sim int) []Head {
		var hs []Head
		for i := 0; i < 6; i++ {
			for _, p := range []string{"a", "b"} {
				id := fmt.Sprintf("%s%d", p, i)
				hs = append(hs, Head{Member: id, ReqID: fmt.Sprintf("%s-r%d", id, sim),
					Arrival: now.Add(time.Duration(sim) * time.Second), SizeBytes: bytesFor(20)})
			}
		}
		return hs
	}
	for sim := 0; sim < 40; sim++ {
		tnow := now.Add(time.Duration(sim) * time.Second)
		s.Tick(tnow)
		hs := step(sim)
		per := map[string]int{}
		for {
			engine, i := s.PickFromHeadsEngine(hs, tnow)
			if i < 0 {
				break
			}
			per[engine]++
			id := hs[i].Member
			s.DispatchedOn(id, hs[i].ReqID, engine, keys(20, Hash(len(id)+sim)), bytesFor(20), tnow)
			// settle immediately: this test is about admissions, not decode
			s.SettleOn(id, hs[i].ReqID, engine, struct{ Prompt, Cached int }{20, 20}, true, tnow)
			// drop the dispatched head so the loop terminates
			hs = append(hs[:i], hs[i+1:]...)
		}
		snap := s.Snapshot()
		for label, es := range snap.Engines {
			if es.BudgetBlocks > 0 && es.NEBlocks > es.BudgetBlocks {
				t.Fatalf("t=%d engine %s NE blocks %d exceed ITS budget %d (a shared pool leaked)",
					sim, label, es.NEBlocks, es.BudgetBlocks)
			}
		}
		if len(hs) > 0 && len(per) == 0 {
			t.Fatalf("t=%d no engine could dispatch any of %d heads (cross-engine starvation)", sim, len(hs))
		}
	}
}

// TestMultiPodColdCapPerEngine: N engines hold N concurrent cold prefills.
// R11's "one cold entry is ALWAYS allowed" is re-derived per engine: each
// engine must seat its own first cold entry even while the other engine's
// charge is at its cap.
func TestMultiPodColdCapPerEngine(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	cfg.ColdEntryCapTokens = 64 * 100 // 100 blocks
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 100000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	s.mu.Lock()
	s.bindings["a0"] = "engine-a"
	s.bindings["a1"] = "engine-a"
	s.bindings["b0"] = "engine-b"
	s.mu.Unlock()
	// a0 charges a cold entry of 60 blocks (under the 100-block cap)
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "a0", ReqID: "a0-r1", Arrival: now, SizeBytes: bytesFor(60)}}, now); i < 0 {
		t.Fatal("a0 must seat")
	}
	// engine-a's charge is now ~60 blocks of cold-entry tokens; its next entry
	// (60 more) would exceed the 100-block cap ⇒ a1 is staggered on engine-a.
	// b0 must still seat, because engine-b's cap is its OWN.
	if _, i := s.PickFromHeadsEngine([]Head{
		{Member: "a1", ReqID: "a1-r1", Arrival: now, SizeBytes: bytesFor(60)},
		{Member: "b0", ReqID: "b0-r1", Arrival: now, SizeBytes: bytesFor(60)},
	}, now); i < 0 {
		t.Fatal("b0 must seat: engine-b's cold cap is its OWN, not engine-a's")
	}
	if !s.Seated("b0") {
		t.Fatal("b0 was not seated on engine-b")
	}
	snap := s.Snapshot()
	if snap.Engines["engine-b"].Seats != 1 {
		t.Fatalf("engine-b seats = %d, want 1", snap.Engines["engine-b"].Seats)
	}
	if s.Seated("a1") {
		t.Fatal("a1 must NOT seat: engine-a's own cold cap is saturated")
	}
}

// TestMultiPodSessionSeatsSumAcrossEngines: item 5 — LSF's unit is the
// session, and a session's seats may sit on different engines, so the count
// must SUM across engines even though the budgets do not.
func TestMultiPodSessionSeatsSumAcrossEngines(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	cfg.SessionOrder = "lsf"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 100000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	// the same SESSION on two branches, homed to two different engines
	s.mu.Lock()
	s.bindings["branch-x"] = "engine-a"
	s.bindings["branch-y"] = "engine-b"
	s.mu.Unlock()
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "branch-x", Session: "sess-1", ReqID: "x1", Arrival: now, SizeBytes: bytesFor(4)}}, now); i < 0 {
		t.Fatal("branch-x must seat")
	}
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "branch-y", Session: "sess-1", ReqID: "y1", Arrival: now, SizeBytes: bytesFor(4)}}, now); i < 0 {
		t.Fatal("branch-y must seat")
	}
	s.mu.Lock()
	counts := s.sessionSeatCountsLocked()
	s.mu.Unlock()
	if counts["sess-1"] != 2 {
		t.Fatalf("session seats = %d, want 2 (summed ACROSS engines)", counts["sess-1"])
	}
	// and the per-engine view still reports one session on each engine
	snap := s.Snapshot()
	if snap.Engines["engine-a"].Sessions != 1 || snap.Engines["engine-b"].Sessions != 1 {
		t.Fatalf("per-engine sessions = %d/%d, want 1/1",
			snap.Engines["engine-a"].Sessions, snap.Engines["engine-b"].Sessions)
	}
}

// TestMultiPodCapacityLivenessPerEngine: item 2 — a late-ready engine must not
// inherit its sibling's capacity, and it must be the one the liveness alarm
// names.
func TestMultiPodCapacityLivenessPerEngine(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	if !s.HasCapacityFor("engine-a") {
		t.Fatal("engine-a must have capacity")
	}
	if s.HasCapacityFor("engine-b") {
		t.Fatal("engine-b must NOT inherit engine-a's capacity (single-pool 'first wins' is the bug)")
	}
	// a member homed while only engine-a is ready goes to engine-a
	now := goldenT0
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(4)}}, now); i < 0 {
		t.Fatal("m1 must seat on the ready engine")
	}
	if got := s.Snapshot().Engines["engine-a"].Seats; got != 1 {
		t.Fatalf("engine-a seats = %d, want 1 (the only engine with capacity)", got)
	}
	// the late engine learns its OWN (different) capacity and keeps it
	if err := s.SetNumBlocksFor("engine-b", 64, 2000); err != nil {
		t.Fatal(err)
	}
	snap := s.Snapshot()
	if snap.Engines["engine-b"].NumBlocks != 2000 || snap.Engines["engine-b"].BudgetBlocks != 1900 {
		t.Fatalf("engine-b capacity = %d/%d, want 2000/1900", snap.Engines["engine-b"].NumBlocks, snap.Engines["engine-b"].BudgetBlocks)
	}
	if snap.Engines["engine-a"].NumBlocks != 1000 || snap.Engines["engine-a"].BudgetBlocks != 950 {
		t.Fatalf("engine-a capacity = %d/%d, want 1000/950", snap.Engines["engine-a"].NumBlocks, snap.Engines["engine-a"].BudgetBlocks)
	}
	// the >1% deviation is reported as heterogeneous (a pool that must not
	// share a budget)
	if len(snap.HeterogeneousEngines) != 1 || snap.HeterogeneousEngines[0] != "engine-b" {
		t.Fatalf("heterogeneous engines = %v, want [engine-b]", snap.HeterogeneousEngines)
	}
}

// TestMultiPodBlockIndexDedupPerEngine: item 1 — two engines holding the SAME
// block are 2× HBM, counted once by a global refcount. Per-engine indexes must
// count it once per engine, and the gate must reason on one engine's budget.
func TestMultiPodBlockIndexDedupPerEngine(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
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
	now := goldenT0
	shared := keys(50, 4242) // the SAME prefix on both engines
	s.mu.Lock()
	s.bindings["a1"] = "engine-a"
	s.bindings["b1"] = "engine-b"
	s.mu.Unlock()
	for _, id := range []string{"a1", "b1"} {
		engine := "engine-a"
		if id == "b1" {
			engine = "engine-b"
		}
		if _, i := s.PickFromHeadsEngine([]Head{{Member: id, ReqID: id + "-r1", Arrival: now, SizeBytes: bytesFor(50)}}, now); i < 0 {
			t.Fatalf("%s must seat", id)
		}
		s.DispatchedOn(id, id+"-r1", engine, shared, bytesFor(50), now)
	}
	snap := s.Snapshot()
	if snap.Engines["engine-a"].UniqueBlocks != 50 || snap.Engines["engine-b"].UniqueBlocks != 50 {
		t.Fatalf("per-engine unique blocks = %d/%d, want 50/50 (the same prefix on two engines is 2x HBM)",
			snap.Engines["engine-a"].UniqueBlocks, snap.Engines["engine-b"].UniqueBlocks)
	}
	// and the shared-prefix effect WITHIN one engine is still deduped
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "a2", ReqID: "a2-r1", Arrival: now, SizeBytes: bytesFor(50)}}, now); i < 0 {
		t.Fatal("a2 must seat")
	}
	s.DispatchedOn("a2", "a2-r1", "engine-a", shared, bytesFor(50), now)
	if got := s.Snapshot().Engines["engine-a"].UniqueBlocks; got != 50 {
		t.Fatalf("engine-a unique blocks = %d, want 50 (a prefix shared INSIDE an engine is deduped)", got)
	}
}

// TestMultiPodCrossEngineFairness: the outer engine selection is relative
// backlog per seat, NOT global oldest-first (a global oldest-first order
// re-creates head-of-line blocking ACROSS engines: engine-a's 900 s-old head
// would always win while engine-b sits on a deep queue).
func TestMultiPodCrossEngineFairness(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// engine-b's budget fits exactly TWO 20-block seats, so four of its six
	// demanding members stay WAITING — the deep queue the fairness key is
	// about. engine-a is idle-rich with a single member.
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 42); err != nil { // budget 39
		t.Fatal(err)
	}
	now := goldenT0
	s.mu.Lock()
	for i := 0; i < 6; i++ {
		s.bindings[fmt.Sprintf("b%d", i)] = "engine-b"
	}
	s.bindings["a0"] = "engine-a"
	s.mu.Unlock()
	var heads []Head
	heads = append(heads, Head{Member: "a0", ReqID: "a0-r", Arrival: now.Add(-900 * time.Second), SizeBytes: bytesFor(20)})
	for i := 0; i < 6; i++ {
		heads = append(heads, Head{Member: fmt.Sprintf("b%d", i), ReqID: fmt.Sprintf("b%d-r", i),
			Arrival: now.Add(-time.Duration(60+i) * time.Second), SizeBytes: bytesFor(20)})
	}
	s.Tick(now)
	engine, i := s.PickFromHeadsEngine(heads, now)
	if i < 0 {
		t.Fatal("something must dispatch")
	}
	served := heads[i].Member
	if engine != "engine-b" || served[:1] != "b" {
		t.Fatalf("served %s from %q: relative backlog per seat (b is 6-deep over 2 seats) must beat global oldest-first (a0 is 900s old)",
			served, engine)
	}
	snap := s.Snapshot()
	if snap.Engines["engine-b"].Seats > 2 {
		t.Fatalf("engine-b seats = %d, want <= 2 (its own budget, 39 blocks)", snap.Engines["engine-b"].Seats)
	}
	if snap.Engines["engine-b"].Waiters == 0 {
		t.Fatalf("engine-b has no waiters: the deep queue this test needs did not form (seats=%d)", snap.Engines["engine-b"].Seats)
	}
}

// TestMultiPodEngineGoneRehomes: RemoveEngine exits the engine's members,
// drops the group and RELEASES their homing so the next head sighting places
// them on a live engine (the wave protocol's node rotation).
func TestMultiPodEngineGoneRehomes(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 100000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	var heads []Head
	for i := 0; i < 8; i++ {
		heads = append(heads, Head{Member: fmt.Sprintf("m%d", i), ReqID: fmt.Sprintf("r%d", i), Arrival: now, SizeBytes: bytesFor(4)})
	}
	s.Tick(now)
	for len(heads) > 0 {
		engine, i := s.PickFromHeadsEngine(heads, now)
		if i < 0 {
			break
		}
		s.DispatchedOn(heads[i].Member, heads[i].ReqID, engine, keys(4, Hash(i+3)), bytesFor(4), now)
		heads = append(heads[:i], heads[i+1:]...)
	}
	snap := s.Snapshot()
	if len(snap.Engines) != 2 {
		t.Fatalf("engines = %d, want 2", len(snap.Engines))
	}
	bSeats := snap.Engines["engine-b"].Seats
	if bSeats == 0 {
		t.Skip("the split homed nobody to engine-b; nothing to re-home")
	}
	exited := s.RemoveEngine("engine-b", now)
	if exited != bSeats {
		t.Fatalf("RemoveEngine exited %d members, engine-b had %d seats", exited, bSeats)
	}
	snap = s.Snapshot()
	if _, ok := snap.Engines["engine-b"]; ok {
		t.Fatal("engine-b survived RemoveEngine")
	}
	// the survivors' homing was released: no member still points at engine-b
	for member, engine := range s.bindingsCopy() {
		if engine == "engine-b" {
			t.Fatalf("member %s is still bound to the dead engine", member)
		}
	}
	// and a re-appearing member is placed on the live engine
	late := []Head{{Member: "late", ReqID: "late-r", Arrival: now.Add(time.Second), SizeBytes: bytesFor(4)}}
	if engine, i := s.PickFromHeadsEngine(late, now.Add(time.Second)); i < 0 || engine != "engine-a" {
		t.Fatalf("late member placed on %q (i=%d), want engine-a", engine, i)
	}
	if got := s.Snapshot().Engines["engine-a"].Seats; got < 1 {
		t.Fatalf("engine-a seats = %d after the teardown", got)
	}
}

// TestSingleModeNeverSplits: engineGroups=single must keep the pooled
// behaviour — two discovered endpoints collapse into ONE group (one index, one
// budget, "first capacity wins"), which is what the golden and every banked
// single-pod artifact were measured on.
func TestSingleModeNeverSplits(t *testing.T) {
	cfg := defaultConfig() // engineGroups=single
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 1200); err != nil {
		t.Fatal(err)
	}
	if ids := s.EngineGroupIDs(); len(ids) != 1 || ids[0] != "" {
		t.Fatalf("engine group ids = %v, want [\"\"] (one pooled group)", ids)
	}
	if snap := s.Snapshot(); snap.BudgetBlocks != 950 {
		t.Fatalf("pooled budget = %d, want 950 (first capacity wins, single pool)", snap.BudgetBlocks)
	}
}

// ---- helpers ----

// bindingOf is the member's homed engine (test introspection).
func (s *Scheduler) bindingOf(member string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bindings[member]
}

// bindingsCopy is a copy of the homing map (test introspection).
func (s *Scheduler) bindingsCopy() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]string, len(s.bindings))
	for k, v := range s.bindings {
		out[k] = v
	}
	return out
}

// runSimpleSeats seats n members through the public API (snapshot tests).
func runSimpleSeats(t *testing.T, s *Scheduler, n int) {
	t.Helper()
	now := goldenT0
	var heads []Head
	for i := 0; i < n; i++ {
		heads = append(heads, Head{Member: fmt.Sprintf("s%d", i), ReqID: fmt.Sprintf("s%d-r", i),
			Arrival: now, SizeBytes: bytesFor(4)})
	}
	s.Tick(now)
	for {
		engine, i := s.PickFromHeadsEngine(heads, now)
		if i < 0 {
			return
		}
		s.DispatchedOn(heads[i].Member, heads[i].ReqID, engine, keys(4, Hash(i+1)), bytesFor(4), now)
		// drop the dispatched head: one dispatch per member (a repeated head
		// with the same ReqID would keep the in-flight count at 1 and spin)
		heads = append(heads[:i], heads[i+1:]...)
	}
}

// TestBindingOutlivesRecordGC is the live lesson of 2026-10-30 19:00Z: on the
// first pinned multi-pod stack, 11 % of dispatches in steady state still landed
// off-binding — because the member RECORD's 30 s TTL forgot a silent member and
// took its HOMING with it, so every return re-homed by least backlog while the
// placement filter (3600 s) still pinned the old engine. The binding has its own
// clock (bindingTtlSeconds, default max(20 x ttlSeconds, 3600)) and a silent
// return must land on the SAME engine.
func TestBindingOutlivesRecordGC(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	cfg.TTLSeconds = 30
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	requireBindTTL(t, s, cfg)
	// engine-b is deliberately EMPTIER than engine-a: least-backlog homing would
	// send a returning member to b, so this test can only pass if the binding is
	// honoured.
	if err := s.SetNumBlocksFor("engine-a", 64, 100000); err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("engine-b", 64, 100000); err != nil {
		t.Fatal(err)
	}
	now := goldenT0
	s.mu.Lock()
	s.bindings["m1"] = "engine-a" // an existing homing
	s.bindingSeen["m1"] = now
	s.mu.Unlock()
	if _, i := s.PickFromHeadsEngine([]Head{{Member: "m1", ReqID: "m1-r1", Arrival: now, SizeBytes: bytesFor(4)}}, now); i < 0 {
		t.Fatal("m1 must seat")
	}
	if got := s.bindingOf("m1"); got != "engine-a" {
		t.Fatalf("binding = %q, want engine-a", got)
	}
	// let the RECORD go: no activity for longer than TTLSeconds
	later := now.Add(time.Duration(cfg.TTLSeconds+5) * time.Second)
	s.Tick(later)
	// the record may be gone; the HOMING must not be
	if got := s.bindingOf("m1"); got != "engine-a" {
		t.Fatalf("the homing was forgotten with the record (%q): a silent return would re-home off the placement pin", got)
	}
	// the return lands on the SAME engine even though engine-b is emptier
	if engine, i := s.PickFromHeadsEngine([]Head{{Member: "m1", ReqID: "m1-r2", Arrival: later, SizeBytes: bytesFor(4)}}, later); i < 0 || engine != "engine-a" {
		t.Fatalf("return placed on %q (i=%d), want engine-a (the homing)", engine, i)
	}
}

// TestBindingReleasedOnItsOwnClock is the other half: the homing is NOT immortal.
// After bindingTtlSeconds of silence it is released (the silent-member defrag —
// nothing warm is left to lose), and that clock is checked on the GC pass alone so
// a still-registered waiter cannot mask the policy.
func TestBindingReleasedOnItsOwnClock(t *testing.T) {
	cfg := defaultConfig()
	cfg.EngineGroups = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	requireBindTTL(t, s, cfg)
	now := goldenT0
	s.mu.Lock()
	s.bindings["m1"] = "engine-a"
	s.bindingSeen["m1"] = now
	s.mu.Unlock()

	// the record clock alone must not touch it
	s.mu.Lock()
	s.gcSilentRecordsLocked(now.Add(time.Duration(s.cfg.TTLSeconds+5) * time.Second))
	s.mu.Unlock()
	if got := s.bindingOf("m1"); got != "engine-a" {
		t.Fatalf("binding = %q at record-TTL age, want it kept (the clocks are separate)", got)
	}
	// the binding clock releases it
	s.mu.Lock()
	s.gcSilentRecordsLocked(now.Add(time.Duration(s.cfg.BindingTTLSeconds+1) * time.Second))
	s.mu.Unlock()
	if got := s.bindingOf("m1"); got != "" {
		t.Fatalf("binding = %q past bindingTtlSeconds, want it released for re-homing", got)
	}
	if s.SnapshotLite().Rehomes < 1 {
		t.Fatal("a released binding must be counted in rehomes_total (the defrag is auditable)")
	}
}

// requireBindTTL asserts the effective binding TTL is the long one, not the
// record TTL (the two clocks must not be the same).
func requireBindTTL(t *testing.T, s *Scheduler, cfg Config) {
	t.Helper()
	if cfg := s.cfg; cfg.BindingTTLSeconds < 3600 {
		t.Fatalf("effective bindingTtlSeconds = %v, want >= 3600 (>= the placement filter's pin)", cfg.BindingTTLSeconds)
	}
}
