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
	cur := map[string]bool{}
	for id := range s.members {
		cur[id] = true
	}
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

// runGoldenScenarioEnd drives the scenario truncated at simEnd (the
// legacy-c57e92ed comparison uses 489 — the pre-C4 code stalls on this
// scenario shortly after ~500, AND at t=490 the always-on C8 fix
// deliberately diverges from legacy (a member with a queued head is no
// longer idle-evicted — the reviewer's predicted A0 caveat). The 489s
// prefix is byte-identical; the divergence point is documented, not
// hidden; S1).
func runGoldenScenarioEnd(t *testing.T, simEnd int) string {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocks(64, 1000); err != nil { // budget 950
		t.Fatal(err)
	}
	tr := &goldenTrace{seats: map[string]bool{}}

	// exit/entry observation: hook order relative to the op is part of
	// the trace (the hooks fire inside the op).
	var exitLog, entryLog []string
	s.onExit = func(reason string, resTurns int) {
		exitLog = append(exitLog, fmt.Sprintf("EXIT reason=%s turns=%d", reason, resTurns))
	}
	s.onEntry = func(wait float64) {
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
					Session:   g.id,
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
			g := byID(h.Session)
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
