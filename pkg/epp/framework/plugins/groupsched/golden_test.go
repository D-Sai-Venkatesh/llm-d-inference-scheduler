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

// The GOLDEN REPLAY TEST (BUILD-PLAN §1): the default config must
// reproduce the legacy decision path exactly. A deterministic scenario
// drives the Scheduler through the full public API on a virtual clock;
// every observable decision (pick results, seats, exits with reasons,
// seated-set transitions) is recorded as a trace. The committed golden
// file (testdata/golden.txt) was generated from the PRE-CHANGE code;
// any refactor must keep the default-config trace byte-identical.
//
// The scenario is NORMAL-PATH ONLY by design: the always-on liveness
// fixes (C3/C4/C8, derive-don't-accumulate) deliberately alter abnormal
// paths (cancellations without EOS, TrySeat failures, gate closures
// without Picks) and must not change this trace. Regenerate with:
//
//	go test ./pkg/epp/framework/plugins/groupsched/ -run TestGolden -update-golden

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

var updateGolden = flag.Bool("update-golden", false, "rewrite testdata/golden.txt from the current run")

// ---- the virtual clock ----

var goldenT0 = time.Unix(1793000000, 0).UTC()

// ---- the scenario model ----

// gsSession is one scripted chain: its turns arrive on a pattern, grow
// their context, and sometimes go silent (TTL) or re-enter late.
type gsSession struct {
	id       string
	chain    []Hash // the growing context (one block per turn-step chunk)
	turn     int
	kind     string // "main" | "single" | "reentry" | "canceller"
	nextTurn int    // sim second of the next arrival
}

// goldenScenario builds the scripted session set: 46 sessions over
// 2,600 sim seconds against a 1000-block capacity (budget 950), enough
// pressure for seating, rotation, budget-idle exits, TTL and re-entry.
func goldenScenario() []*gsSession {
	var ss []*gsSession
	for i := 0; i < 46; i++ {
		kind := "main"
		switch {
		case i%11 == 7:
			kind = "single" // one request then silence (TTL path)
		case i%13 == 5:
			kind = "reentry" // a few turns, long silence, returns
		case i%17 == 3:
			kind = "canceller" // one non-natural settle mid-run
		}
		ss = append(ss, &gsSession{
			id:       fmt.Sprintf("s%02d", i),
			chain:    keys(8, Hash(1000+i)),
			kind:     kind,
			nextTurn: 20 + 9*i, // staggered arrivals, 20s..425s
		})
	}
	return ss
}

// turnBlocks is the per-turn context growth (blocks): mains grow
// steadily; singles stay small.
func (g *gsSession) turnBlocks() int {
	switch g.kind {
	case "single":
		return 6
	default:
		return 10 + g.turn/2 // 10..~40 blocks over a long chain
	}
}

// thinkGap is the inter-turn gap pattern (seconds).
func (g *gsSession) thinkGap() int {
	switch g.kind {
	case "single":
		return 1 << 30 // never returns
	case "reentry":
		if g.turn >= 3 {
			return 900 // long silence, then re-enter via nextTurn below
		}
		return 6 + g.turn*7
	default:
		return 3 + (g.turn*5)%23 // 3..25s
	}
}

// ---- the driver ----

type goldenTrace struct {
	sb    strings.Builder
	seats map[string]bool
}

func (g *goldenTrace) line(format string, args ...any) {
	fmt.Fprintf(&g.sb, format+"\n", args...)
}

// observeState appends the seated set ONLY when it changes (keeps the
// golden small without losing decisions).
func (g *goldenTrace) observeState(s *Scheduler) {
	cur := s.seatedIDs()
	if len(cur) != len(g.seats) {
		goto diff
	}
	for id := range cur {
		if !g.seats[id] {
			goto diff
		}
	}
	return
diff:
	g.seats = cur
	ids := make([]string, 0, len(cur))
	for id := range cur {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	g.line("STATE seats=%d [%s]", len(ids), strings.Join(ids, ","))
}

// runGoldenScenario drives the scheduler and returns the trace.
func runGoldenScenario(t *testing.T) string {
	return runGoldenScenarioEnd(t, 2600)
}

// runGoldenScenarioCfg is the PARAMETRIZED driver used by the WAVE-1
// session-ordering equality test: cfg is the scheduler config and sessions
// maps a scripted chain's id to its SESSION attribute (nil = every head
// carries none, i.e. session := member — base mode).
func runGoldenScenarioCfg(t *testing.T, simEnd int, cfg Config) string {
	return runGoldenScenarioFull(t, simEnd, cfg)
}

// runGoldenScenarioEnd drives the scenario truncated at simEnd (the
// legacy-c57e92ed comparison uses 489 — the pre-C4 code stalls on this
// scenario shortly after ~500, AND at t=490 the always-on C8 fix
// deliberately diverges from legacy (a member with a queued head is no
// longer idle-evicted — the reviewer's predicted A0 caveat). The 489s
// prefix is byte-identical; the divergence point is documented, not
// hidden; S1).
func runGoldenScenarioEnd(t *testing.T, simEnd int) string {
	t.Helper()
	return runGoldenScenarioFull(t, simEnd, defaultConfig())
}

func runGoldenScenarioFull(t *testing.T, simEnd int, cfg Config) string {
	t.Helper()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// The scenario names its ONE engine: in engineGroups=single the key
	// collapses to the pooled group (the golden was generated that way), and
	// in engineGroups=per-engine it is a real discovered engine — which is
	// what makes TestPerEngineGoldenEquality a genuine unit-equivalence gate.
	if err := s.SetNumBlocksFor("engine-a", 64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	tr := &goldenTrace{seats: map[string]bool{}}

	// exit/entry observation: hook order relative to the op is part of
	// the trace (the hooks fire inside the op).
	var exitLog, entryLog []string
	// The hooks carry the ENGINE label in multi-engine mode; the trace text
	// is unchanged, so a per-engine replay of this scenario compares against
	// the same golden file line for line (PREREG-multipod-offload directive
	// 1: the unit-equivalence gate).
	s.onExit = func(_ string, reason string, resTurns int) {
		exitLog = append(exitLog, fmt.Sprintf("EXIT reason=%s turns=%d", reason, resTurns))
	}
	s.onEntry = func(_ string, wait float64) {
		entryLog = append(entryLog, fmt.Sprintf("SEAT wait=%.3f", wait))
	}

	ss := goldenScenario()
	byID := func(id string) *gsSession {
		for _, g := range ss {
			if g.id == id {
				return g
			}
		}
		return nil
	}

	// reqSeq numbers a session's requests (reqIDs must be unique).
	reqSeq := map[string]int{}
	nextReqID := func(id string) string {
		reqSeq[id]++
		return fmt.Sprintf("%s-r%02d", id, reqSeq[id])
	}

	// inFlight: reqID -> (session, arrivalSim, sizeBytes) — the requests
	// dispatched and not yet settled; settle after a decode period.
	type flight struct {
		sess     *gsSession
		reqID    string
		arr      int
		blocks   int
		settleAt int
		natural  bool
	}
	var flying []*flight

	headsAt := func(sim int) []Head {
		var hs []Head
		for _, g := range ss {
			if g.nextTurn <= sim {
				// the head is queued since nextTurn
				hs = append(hs, Head{
					Member:    g.id,
					ReqID:     fmt.Sprintf("%s-r%02d", g.id, reqSeq[g.id]+1),
					Arrival:   goldenT0.Add(time.Duration(g.nextTurn) * time.Second),
					SizeBytes: int64(len(g.chain)+g.turnBlocks()) * 4 * 64,
				})
			}
		}
		return hs
	}

	for sim := 0; sim <= simEnd; sim++ {
		now := goldenT0.Add(time.Duration(sim) * time.Second)

		// settle due flights (decode periods of 5..40s)
		var still []*flight
		for _, f := range flying {
			if f.settleAt <= sim {
				g := f.sess
				usage := struct{ Prompt, Cached int }{Prompt: f.blocks}
				if f.natural {
					usage.Cached = f.blocks - 2 // mostly warm
					if usage.Cached < 0 {
						usage.Cached = 0
					}
				}
				s.Chunk(g.id)
				s.Settle(g.id, f.reqID, usage, f.natural, goldenT0.Add(time.Duration(f.settleAt)*time.Second))
				tr.line("t=%d SETTLE %s req=%s natural=%v", f.settleAt, g.id, f.reqID, f.natural)
				for _, l := range exitLog {
					tr.line("t=%d %s", f.settleAt, l)
				}
				exitLog = nil
				tr.observeState(s)
			} else {
				still = append(still, f)
			}
		}
		flying = still

		// the 1s ticker (as the plugin runs)
		s.Tick(now)

		// dispatch attempts: up to 2 picks per sim second
		for pick := 0; pick < 2; pick++ {
			hs := headsAt(sim)
			if len(hs) == 0 {
				break
			}
			i := s.PickFromHeads(hs, now)
			if i < 0 {
				tr.line("t=%d PICK -1", sim)
				break
			}
			h := hs[i]
			g := byID(h.Member)
			reqID := nextReqID(g.id)
			// the dispatched request's context = chain + this turn's growth
			g.turn++
			newBlocks := g.turnBlocks()
			for _, k := range keys(newBlocks, Hash(1000+len(ss)+g.turn*31+int(Hash(g.id[1]))%97)) {
				g.chain = append(g.chain, k)
			}
			s.Dispatched(g.id, reqID, append([]Hash(nil), g.chain...), h.SizeBytes, now)
			// schedule the settle; cancellers fail their 3rd turn
			natural := true
			if g.kind == "canceller" && g.turn == 3 {
				natural = false
			}
			decode := 5 + (g.turn*7)%35
			flying = append(flying, &flight{sess: g, reqID: reqID, arr: sim,
				blocks: len(g.chain), settleAt: sim + decode, natural: natural})
			// the next arrival (or silence)
			gap := g.thinkGap()
			if g.kind == "reentry" && g.turn == 3 {
				g.nextTurn = sim + 900 // the long silence, then re-entry
			} else if gap < 1<<29 {
				g.nextTurn = sim + gap
			} else {
				g.nextTurn = 1 << 29 // silent forever (single-shot)
			}
			tr.line("t=%d PICK %d -> %s req=%s blocks=%d", sim, i, g.id, reqID, len(g.chain))
			for _, l := range entryLog {
				tr.line("t=%d %s", sim, l)
			}
			entryLog = nil
			tr.observeState(s)
		}
		// hook emissions from Tick (ttl/budget exits with no op line)
		for _, l := range exitLog {
			tr.line("t=%d %s", sim, l)
		}
		exitLog = nil
		for _, l := range entryLog {
			tr.line("t=%d %s", sim, l)
		}
		entryLog = nil
		tr.observeState(s)
	}

	// final gauges (derived, not per-event): exit counters and snapshot
	snap := s.Snapshot()
	tr.line("SNAPSHOT served=%d fcfs=%d refill=%.4f", snap.Served, snap.FCFSServed, snap.RefillShare)
	var reasons []string
	for r := range snap.Exits {
		reasons = append(reasons, r)
	}
	sort.Strings(reasons)
	for _, r := range reasons {
		tr.line("EXITS %s=%d", r, snap.Exits[r])
	}
	return tr.sb.String()
}

func TestGoldenReplay(t *testing.T) {
	got := runGoldenScenario(t)
	goldenPath := filepath.Join("testdata", "golden.txt")
	if *updateGolden {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("golden rewritten: %d bytes", len(got))
		return
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("golden file missing (run with -update-golden): %v", err)
	}
	if got != string(want) {
		// first divergence, for a readable failure
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
				t.Fatalf("golden divergence at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
			}
		}
		t.Fatal("golden divergence (length)")
	}
}

// TestGoldenDeterministic runs the scenario twice and requires identical
// traces (guards against residual map-order nondeterminism).
func TestGoldenDeterministic(t *testing.T) {
	a := runGoldenScenario(t)
	b := runGoldenScenario(t)
	if a != b {
		t.Fatal("scenario is not deterministic across runs")
	}
}

// TestGoldenLegacyC57e92ed closes the golden's circularity (S1): the
// committed testdata/golden-legacy.txt was generated from the RAW
// pre-change code (c57e92ed) on this scenario truncated at sim 500
// (the legacy TrySeat spin — C4 — stalls it shortly after). The current
// build's default config must reproduce that prefix byte-identically,
// pinning default == legacy on code that predates every change in this
// branch, not just on itself.
func TestGoldenLegacyC57e92ed(t *testing.T) {
	got := runGoldenScenarioEnd(t, 489)
	want, err := os.ReadFile(filepath.Join("testdata", "golden-legacy.txt"))
	if err != nil {
		t.Fatalf("legacy golden missing: %v", err)
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
				t.Fatalf("legacy golden divergence at line %d:\n  got:  %s\n  want: %s", i+1, a, b)
			}
		}
		t.Fatal("legacy golden divergence (length)")
	}
}

// TestGoldenHashesPinned closes the loop between "the test passes" and "the bytes
// that passed are the bytes an auditor can check" (audit item 4): testdata/
// SHA256SUMS pins the sha256 of every committed trace, and this test FAILS if a
// golden file changes without its hash being updated — a silent `-update-golden`
// regeneration can no longer ride along unnoticed.
func TestGoldenHashesPinned(t *testing.T) {
	sums, err := os.ReadFile(filepath.Join("testdata", "SHA256SUMS"))
	if err != nil {
		t.Fatalf("testdata/SHA256SUMS missing: %v", err)
	}
	want := map[string]string{}
	for _, line := range strings.Split(string(sums), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		want[f[1]] = f[0]
	}
	if len(want) < 3 {
		t.Fatalf("SHA256SUMS lists %d files, want at least the three trace files", len(want))
	}
	for name, h := range want {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		got := fmt.Sprintf("%x", sha256.Sum256(b))
		if got != h {
			t.Fatalf("%s: sha256 %s != pinned %s — if the change is intended, update both the trace and testdata/SHA256SUMS (and say why in the commit)", name, got, h)
		}
	}
}

// ---- GSCH-CORE: the f0p golden equality ----
//
// The core's baseline is f0p — the single-pod anchor every banked number is stated
// against. testdata/golden-f0p-core.txt was produced by the ARCHIVE build
// (dsai/groupsched-tier-aware @ dd5350b5) on THIS scenario with f0p's rendered
// config: rotationMode=quantum Q4, valve on, victimOrder legacy, windowMode last4,
// gateMode detector (no plugin-side dispatch gate), engineGroups=per-engine,
// bindingMode=member-stable, every F/PAT/residue knob at its default-off value.
//
// The core must reproduce it BYTE FOR BYTE at N=1. The ONE declared behavioural
// difference between the two builds is the safety module, and this comparison is
// exact because of the module's DEFAULT: NoopSafety (the seat budget is the whole
// admission control). f0p's plugin-side gate was OFF as well, so the two builds run
// with the SAME gate state — open, always. (Run the core with safety=budget-fraction
// and the trace MAY diverge wherever that net fires; that arm's divergence is the
// measured effect of the net, not a golden violation, and the wave's matched anchor
// arm measures it on the cluster.)
//
// What the trace proves: identical picks, identical seats, identical exits with
// reasons, identical seated-set transitions, identical served/refill totals, over
// 2,600 simulated seconds and 1,210 dispatches.
func TestCoreEqualsF0P(t *testing.T) {
	got := runGoldenScenario(t)
	want, err := os.ReadFile(filepath.Join("testdata", "golden-f0p-core.txt"))
	if err != nil {
		t.Fatalf("f0p golden missing: %v", err)
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
				t.Fatalf("core/f0p divergence at line %d:\n  core: %s\n  f0p:  %s", i+1, a, b)
			}
		}
		t.Fatal("core/f0p divergence (length)")
	}
}
