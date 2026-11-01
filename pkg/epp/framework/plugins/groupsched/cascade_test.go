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

// THE ENFORCEMENT CASCADE, and the TWO VIEWS that end it.
//
// The charged drain (iii) put an exiting member's blocks on the budget's used side until
// it drains — correct, because the engine pins them. What it also did, silently, was make
// MARKING FREE: a pass that marked a mid-flight victim saw the same over-budget number
// afterwards, so enforceBudgetLocked marked the next in-flight member, and the next, on
// every pass — and passes are every Pick. Behind those marks a perfectly healthy engine
// churned its whole membership out from under one turn each.
//
// The fix keeps (iii) exactly as it is and asks the two questions separately:
//
//	ADMISSION  (freeBudgetLocked / TrySeat): exiting blocks are OCCUPIED. A head that
//	           would only fit if they were free still waits.        → tested below
//	ENFORCEMENT (overBudgetLocked):           exiting blocks are NOT A REASON TO EVICT
//	           MORE — they are already leaving.                    → tested below
//
// The numbers these tests pin, from the live batch-1 anchor (99a47617 vs 88e74f85, same
// config, c128) and from the saturated replay arm (golden-hrrn-tier):
//
//	                       live: no-fix → cascade        replay arm: cascade → fixed
//	mid-flight marks        87     → 2,520 (29x)          102   → 36
//	turns per residency      2.26  → 0.59                  —      —
//	quantum-complete exits   532   → 44                   13    → 34
//	dispatches             4,027   → 3,104 (−23%)        1,020 → 1,212 (+19%)
//	refill share (uncached) 31.7%  → 44.5%                 —      —
//
// Every test here runs at N=1 and N=2 (the twins rule).

import (
	"strings"
	"testing"
	"time"
)

// ---- ENFORCEMENT: one exiting member covering the overage ends the pass ----

// The engine is over budget by (member's blocks), and that member is ALREADY exiting:
// enforcement must mark NOBODY else. This is the cascade's first domino.
func TestEnforcementDoesNotMarkAnotherWhileOneIsLeaving(t *testing.T) {
	chargedPath(t, defaultConfig(), "no-cascade", func(t *testing.T, fx *engineSet, g *engineGroup) {
		// three members in flight on one engine; the engine's budget is exactly full
		fx.s.Chunk("gone") // its prefill completed: the cold valve is free for the third seat
		fx.seatOne(t, "m2", 250, 5*time.Second)
		markExitingInflight(t, fx, g, "r-gone", true) // the 300-block member leaves
		fx.s.mu.Lock()
		for _, id := range []string{"m2"} { // give the sibling a request too (mid-flight)
			m := g.members[id]
			m.inflightIDs = map[string]inflightRec{"r-" + id: {since: fx.clock, confirmed: true}}
			m.inflight = 1
		}
		// THE OVERAGE IS EXACTLY THE EXITING MEMBER'S OWN FOOTPRINT: 950 total seated
		// (400 + 300 + 250) against a budget of 100 more... make it precise:
		g.budgetBlocks = 700 // NE is 950: over by 250, and "gone" (300) already covers it
		fx.s.mu.Unlock()

		fx.s.mu.Lock()
		fx.s.enforceBudgetLocked(g, fx.clock)
		m2Exiting := g.index.ExitingLocked("m2")
		fx.s.mu.Unlock()
		if m2Exiting {
			t.Fatal("an exiting member already covers the overage: enforcement marked ANOTHER in-flight member (the cascade)")
		}
		if !g.index.Exiting("gone") {
			t.Fatal("the member on its way out must still be the one leaving")
		}
	})
}

// ... and when the overage is LARGER than what is already leaving, the pass takes ONE
// more victim — not the whole engine: after that mark the pass is done again.
func TestEnforcementTakesOneMoreWhenTheOverageExceedsTheRetirees(t *testing.T) {
	chargedPath(t, defaultConfig(), "one-more", func(t *testing.T, fx *engineSet, g *engineGroup) {
		fx.s.Chunk("gone")
		fx.seatOne(t, "m2", 250, 5*time.Second)
		markExitingInflight(t, fx, g, "r-gone", true)
		fx.s.mu.Lock()
		g.members["m2"].inflightIDs = map[string]inflightRec{"r-m2": {since: fx.clock, confirmed: true}}
		g.members["m2"].inflight = 1
		// NE 950, budget 300: over by 650; "gone" excuses 300, so 350 is still over —
		// more than m2's 250, so exactly ONE further mark can be justified... and after
		// that (250 excused, 100 left) the pass must stop: nobody else may be marked
		// while the two retirees alone carry the footprint.
		g.budgetBlocks = 300
		fx.s.mu.Unlock()

		fx.s.mu.Lock()
		fx.s.enforceBudgetLocked(g, fx.clock)
		marked := 0
		for id := range g.members {
			if g.index.ExitingLocked(id) {
				marked++
			}
		}
		fx.s.mu.Unlock()
		if marked != 2 {
			t.Fatalf("%d members are exiting, want exactly 2 (one retiree + the one mark the overage justifies)", marked)
		}
	})
}

// ---- ADMISSION: the same blocks are still OCCUPIED for a new seat ----

// The heart of the two views: a head that would fit only if the exiting member's blocks
// were free must WAIT — and then be admitted the moment that member actually drains.
func TestAdmissionRefusesWhatOnlyTheRetireesBlocksWouldMakeRoomFor(t *testing.T) {
	chargedPath(t, defaultConfig(), "admission-refuses", func(t *testing.T, fx *engineSet, g *engineGroup) {
		markExitingInflight(t, fx, g, "r-gone", true) // NE = 700 of a 950 budget
		fx.s.mu.Lock()
		// shrink the budget so that the newcomer only fits if "gone"'s 300 were free
		g.budgetBlocks = 650
		fx.s.mu.Unlock()

		h := head("newcomer", time.Second, bytesFor(100))
		if got := neOn(fx, g); got != 700 {
			t.Fatalf("premise: NE = %d", got)
		}
		fx.pick(h)
		if fx.s.Seated("newcomer") {
			t.Fatal("a newcomer was seated on room that only exists if the exiting member's blocks are free")
		}
		// the drain: now the room is real, and the same head is admitted
		fx.s.SettleOnUsage("gone", "r-gone", g.id, 300, 0, 0, false, fx.clock)
		fx.s.mu.Lock()
		g.budgetBlocks = 950
		fx.s.mu.Unlock()
		fx.pick(h)
		if !fx.s.Seated("newcomer") {
			t.Fatal("after the drain the room is real: the head must be admitted")
		}
	})
}

// ---- the livelock guard that (iii) added must still work ----

// An engine whose ONLY member is exiting and in flight, with a queue view too old to
// trust: the pass must return instead of re-picking the same victim forever (the guard
// the accounting fix added — the cascade fix must not remove the reason for it).
func TestExitingVictimDoesNotSpinEnforcementAfterTheCascadeFix(t *testing.T) {
	forEngines(t, defaultConfig(), func(t *testing.T, fx *engineSet) {
		fx.seatOne(t, "m", 900, 2*time.Second)
		m, g := fx.member(t, "m")
		fx.s.mu.Lock()
		m.inflightIDs = map[string]inflightRec{"r-m": {since: fx.clock, confirmed: true}}
		m.inflight = 1
		g.index.MarkExitingLocked("m")
		g.budgetBlocks = 100
		fx.s.mu.Unlock()
		fx.clock = fx.clock.Add(60 * time.Second)
		done := make(chan struct{})
		go func() {
			fx.s.Tick(fx.clock)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("Tick spun: an exiting in-flight victim was re-picked forever")
		}
		if !fx.s.Seated("m") {
			t.Fatal("the exiting member must still be seated (it drains, it is not evicted)")
		}
	})
}

// exitReasonOf pulls the reason out of a trace's EXIT line.
func exitReasonOf(line string) (string, bool) {
	i := strings.Index(line, "EXIT reason=")
	if i < 0 {
		return "", false
	}
	r := line[i+len("EXIT reason="):]
	if j := strings.IndexAny(r, " \t"); j >= 0 {
		r = r[:j]
	}
	return r, true
}

// ---- the replay differential: the EXIT COMPOSITION, not just the dispatch count ----

// TestExitCompositionIsNotACascade is the replay-side gate on the cascade: MID-FLIGHT
// MARKS PER DISPATCH. The cascade's signature is not "many exits" (a busy engine exits
// constantly) — it is marks that outnumber the work they are supposed to protect: the
// measured live arm ran 2,520 marks for 3,104 dispatches (81 %), against 87 for 4,027
// (2 %) on the arm it is compared with. In the replay, the arm that saturates (a CPU
// tier + hrrn-tier) went 102/1,020 (10.0 %) before the fix to 36/1,212 (3.0 %) after,
// while the default arm's trace did not move by a single byte.
//
// The live pre-run gate is the same number from the other side: the exiting-drain share
// of llm_d_epp_group_exits_total within ~2x of the no-fix arm's (~5 %, so <= ~10 %).
func TestExitCompositionIsNotACascade(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"default arm (f0p config)", defaultConfig()},
		{"saturated arm (tier + hrrn-tier)", func() Config {
			c := defaultConfig()
			c.EntryOrder = "hrrn-tier"
			c.KvBlockBytes = 4 * 64
			c.CpuTierBytes = 2000 * c.KvBlockBytes
			return c
		}()},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			trace := runGoldenScenarioCfg(t, 2600, tc.cfg)
			exits := map[string]int{}
			total, dispatches := 0, 0
			for _, line := range strings.Split(trace, "\n") {
				if reason, ok := exitReasonOf(line); ok {
					exits[reason]++
					total++
				}
				if strings.Contains(line, " PICK ") && !strings.Contains(line, "PICK -1") {
					dispatches++
				}
			}
			if total == 0 || dispatches == 0 {
				t.Fatalf("the scenario produced %d exits and %d dispatches", total, dispatches)
			}
			marks := exits["exiting-drain"]
			perDispatch := float64(marks) / float64(dispatches)
			t.Logf("%d exits (%.1f %% exiting-drain), %d dispatches, marks/dispatch %.1f %%",
				total, 100*float64(marks)/float64(total), dispatches, 100*perDispatch)
			if perDispatch > 0.05 {
				t.Fatalf("%d mid-flight marks for %d dispatches (%.1f %%): enforcement is marking members in bulk "+
					"(the cascade measured 81 %% live and 10 %% in this replay; the fixed arm runs 3 %%)",
					marks, dispatches, 100*perDispatch)
			}
		})
	}
}
