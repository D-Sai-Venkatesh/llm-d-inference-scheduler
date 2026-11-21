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

// A CPU-TIER GROUP: the seats sized by the tier, what runs sized by the GPU, first come first served
// inside the group. One mechanism per test, then the replay workload with a tier.

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// tierSched is engines of 1000 GPU blocks (GPU budget 0.95 x 1000 = 950) with a CPU tier of 4000
// blocks (seat budget 0.95 x 4000 = 3800).
func tierSched(t *testing.T, engines ...string) *Scheduler {
	t.Helper()
	cfg := defaultConfig()
	cfg.TierCapacityTokens = 4000 * 64
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(engines) == 0 {
		engines = []string{""}
	}
	for _, e := range engines {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

// headN is turn n of a thread, with its own request id.
func headN(id string, n int, ago time.Duration, c []Chain) Head {
	return Head{Thread: id, ReqID: fmt.Sprintf("%s#%d", id, n), Arrival: epoch.Add(-ago), Chains: c, Returner: n > 0}
}

// start dispatches one head and brings it to a pod with its prefill done: the turn is RUNNING.
func start(t *testing.T, s *Scheduler, h Head) {
	t.Helper()
	if i := s.Pass([]Head{h}, epoch); i != 0 {
		t.Fatalf("%s must be dispatched, got %d", h.ReqID, i)
	}
	s.AtPod(h.ReqID, "", epoch)
	s.FirstChunk(h.ReqID, epoch)
}

func engine(s *Scheduler, id string) EngineSnapshot { return s.Snapshot(epoch).Engines[id] }

func TestTierSizesTheSeatsAndTheGPUSizesWhatRuns(t *testing.T) {
	if e := engine(tierSched(t), ""); e.BudgetBlocks != 3800 || e.RunBudgetBlocks != 950 {
		t.Fatalf("seat budget 0.95 x 4000 = 3800, GPU budget 0.95 x 1000 = 950; got %d / %d", e.BudgetBlocks, e.RunBudgetBlocks)
	}
	if e := engine(newSched(t), ""); e.BudgetBlocks != 950 || e.RunBudgetBlocks != 950 {
		t.Fatalf("no tier: the GPU sizes the seats, 950 / 950; got %d / %d", e.BudgetBlocks, e.RunBudgetBlocks)
	}
	cfg := defaultConfig()
	cfg.TierCapacityTokens = 200 * 64 // a tier smaller than the GPU
	s, _ := NewScheduler(cfg)
	_ = s.SetNumBlocksFor("", 64, 1000)
	if e := engine(s, ""); e.BudgetBlocks != 950 {
		t.Fatalf("a tier never makes the group smaller than the GPU alone holds: got %d", e.BudgetBlocks)
	}
	cfg.TierBudgetFraction = 0
	if _, err := NewScheduler(cfg); err == nil {
		t.Fatal("tierBudgetFraction 0 must be refused")
	}
}

// TestASeatWaitsForTheGPU: three 300-block turns run (900 of 950). A fourth thread gets a seat — the
// tier has room — but its turn waits until a running one ends.
func TestASeatWaitsForTheGPU(t *testing.T) {
	s := tierSched(t)
	for _, id := range []string{"a", "b", "c"} {
		start(t, s, headN(id, 0, time.Second, chain(id, 300)))
	}
	if e := engine(s, ""); e.RunningBlocks != 900 || e.RunningThreads != 3 {
		t.Fatalf("900 blocks / 3 seats running, got %d / %d", e.RunningBlocks, e.RunningThreads)
	}
	d := headN("d", 0, time.Second, chain("d", 300))
	if i := s.Pass([]Head{d}, epoch); i != -1 {
		t.Fatalf("d must wait for the GPU (900 + 300 > 950), got %d", i)
	}
	if !seated(s, "d") {
		t.Fatal("d must hold a seat: the group is sized by the tier")
	}
	s.Settle("a#0", "", epoch)
	if i := s.Pass([]Head{d}, epoch); i != 0 {
		t.Fatalf("a ended (600 running): d must run, got %d", i)
	}
	if e := engine(s, ""); e.RunningBlocks != 900 || e.RunningThreads != 3 {
		t.Fatalf("b, c, d running: 900 / 3, got %d / %d", e.RunningBlocks, e.RunningThreads)
	}
}

// TestFirstComeFirstServedInsideTheGroup: a waiting head holds the younger heads behind it, even one
// small enough to fit.
func TestFirstComeFirstServedInsideTheGroup(t *testing.T) {
	s := tierSched(t)
	start(t, s, headN("small", 0, 30*time.Second, chain("small", 10)))
	s.Settle("small#0", "", epoch)
	for _, id := range []string{"a", "b", "c"} {
		start(t, s, headN(id, 0, time.Second, chain(id, 300)))
	}
	older := headN("d", 0, 5*time.Second, chain("d", 300))
	if i := s.Pass([]Head{older}, epoch); i != -1 {
		t.Fatal("d must wait for the GPU")
	}
	younger := headN("small", 1, time.Second, chain("small", 12)) // 900 + 2 new blocks would fit
	if i := s.Pass([]Head{older, younger}, epoch); i != -1 {
		t.Fatalf("the younger head must not overtake the older one, got %d", i)
	}
}

// TestNothingRunningRunsAnything: with nothing running, the oldest head runs whatever its size.
func TestNothingRunningRunsAnything(t *testing.T) {
	s := tierSched(t)
	start(t, s, headN("big", 0, time.Second, chain("big", 1200))) // above the 950-block GPU budget
	if e := engine(s, ""); e.RunningBlocks != 1200 {
		t.Fatalf("the big head runs alone, got %d running", e.RunningBlocks)
	}
	if i := s.Pass([]Head{headN("x", 0, time.Second, chain("x", 5))}, epoch); i != -1 {
		t.Fatalf("while it runs nothing else fits, got %d", i)
	}
}

// TestASharedPrefixRunsOnce: two seats sharing a 400-block prefix hold it once on the GPU.
func TestASharedPrefixRunsOnce(t *testing.T) {
	s := tierSched(t)
	start(t, s, headN("a", 0, time.Second, chainWithPrefix("sys", 400, "a", 300)))
	start(t, s, headN("b", 0, time.Second, chainWithPrefix("sys", 400, "b", 200)))
	if e := engine(s, ""); e.RunningBlocks != 900 {
		t.Fatalf("400 + 300 + 200 = 900 distinct blocks running, got %d", e.RunningBlocks)
	}
	s.Settle("a#0", "", epoch)
	s.Pass(nil, epoch)
	if e := engine(s, ""); e.RunningBlocks != 600 {
		t.Fatalf("a ended; b still runs the prefix: 600, got %d", e.RunningBlocks)
	}
}

// TestAnExpiredTurnStopsRunningAndALateOneRunsAgain: a released turn that never reaches a pod within S
// stops counting; when it does reach one later, it counts again.
func TestAnExpiredTurnStopsRunningAndALateOneRunsAgain(t *testing.T) {
	s := tierSched(t)
	h := headN("a", 0, time.Second, chain("a", 300))
	if s.Pass([]Head{h}, epoch) != 0 {
		t.Fatal("a must be released")
	}
	late := epoch.Add(time.Duration(s.cfg.IdleSeconds+5) * time.Second)
	s.Pass(nil, late)
	if e := engine(s, ""); e.RunningBlocks != 0 {
		t.Fatalf("an expired release runs nothing, got %d", e.RunningBlocks)
	}
	s.AtPod(h.ReqID, "", late)
	s.Pass(nil, late)
	if e := engine(s, ""); e.RunningBlocks != 300 {
		t.Fatalf("a late PreRequest runs: 300, got %d", e.RunningBlocks)
	}
	s.Settle(h.ReqID, "", late)
	s.Pass(nil, late)
	if e := engine(s, ""); e.RunningBlocks != 0 || e.RunningThreads != 0 {
		t.Fatalf("settled: nothing runs, got %d / %d", e.RunningBlocks, e.RunningThreads)
	}
}

// TestRunningFollowsTheObservedEngine: a turn the framework placed on another engine runs THERE.
func TestRunningFollowsTheObservedEngine(t *testing.T) {
	s := tierSched(t, "e0", "e1")
	h := headN("a", 0, time.Second, chain("a", 300))
	if s.Pass([]Head{h}, epoch) != 0 {
		t.Fatal("a must be released")
	}
	s.AtPod(h.ReqID, "e1", epoch)
	s.Pass(nil, epoch)
	if e0, e1 := engine(s, "e0"), engine(s, "e1"); e0.RunningBlocks != 0 || e1.RunningBlocks != 300 {
		t.Fatalf("the turn runs on e1: e0 %d, e1 %d", e0.RunningBlocks, e1.RunningBlocks)
	}
}

// TestWithoutATierNothingWaitsForTheGPU: one level — a seated thread's grown turn that takes what runs
// past the GPU budget is dispatched exactly as before (the seat budget's enforcement owns growth).
func TestWithoutATierNothingWaitsForTheGPU(t *testing.T) {
	small := defaultConfig()
	small.TierCapacityTokens = 200 * 64 // a tier too small to grow the group: seats = GPU budget
	for name, cfg := range map[string]Config{"no tier": defaultConfig(), "tier smaller than the GPU": small} {
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		_ = s.SetNumBlocksFor("", 64, 1000)
		start(t, s, headN("a", 0, 10*time.Second, chain("a", 450)))
		start(t, s, headN("b", 0, 9*time.Second, chain("b", 450)))
		s.Settle("b#0", "", epoch)
		if i := s.Pass([]Head{headN("b", 1, time.Second, chain("b", 510))}, epoch); i != 0 {
			t.Fatalf("%s: 450 + 510 > 950 running, still dispatched as today, got %d", name, i)
		}
		if name == "no tier" {
			if e := engine(s, ""); e.RunningBlocks != 0 || e.RunningThreads != 0 {
				t.Fatalf("no tier: the GPU side is not kept (one level is the PR's path), got %d / %d", e.RunningBlocks, e.RunningThreads)
			}
		}
	}
}

// TestRunSideFollowsTheSeatsKeys: a running seat's growth, replacement, exit mark and unseat move the
// RUN side exactly with it, and a shared key stays while any running seat holds it.
func TestRunSideFollowsTheSeatsKeys(t *testing.T) {
	idx := newIdx(t)
	mustSeat(t, idx, "a", chainWithPrefix("p", 10, "a", 10), 1000)
	mustSeat(t, idx, "b", chainWithPrefix("p", 10, "b", 5), 1000)
	idx.SetRunning("a", true)
	if idx.RunningBlocks() != 20 || idx.RunCost(chainWithPrefix("p", 10, "b", 5)) != 5 {
		t.Fatalf("a runs 20; b would add only its own 5, got %d / %d", idx.RunningBlocks(), idx.RunCost(chainWithPrefix("p", 10, "b", 5)))
	}
	idx.Grow("a", chainWithPrefix("p", 10, "a", 16)) // a's turn grows while it runs
	if idx.RunningBlocks() != 26 {
		t.Fatalf("growth while running: 26, got %d", idx.RunningBlocks())
	}
	idx.SetRunning("b", true)
	idx.Grow("a", chain("a2", 4)) // a's chain is replaced (a compaction): the shared prefix stays for b
	if idx.RunningBlocks() != 4+15 {
		t.Fatalf("a's 4 + b's 15 = 19, got %d", idx.RunningBlocks())
	}
	idx.MarkExiting("b")
	if idx.RunningBlocks() != 19 {
		t.Fatalf("an exit mark does not stop a running turn: 19, got %d", idx.RunningBlocks())
	}
	idx.Unseat("b")
	idx.SetRunning("a", false)
	if idx.RunningBlocks() != 0 {
		t.Fatalf("nothing runs, got %d", idx.RunningBlocks())
	}
}

// runSideExact checks the RUN side against the seats: exactly the union of the running seats' keys.
func runSideExact(s *Scheduler) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, g := range s.groups {
		union := map[Hash]struct{}{}
		for _, sb := range g.index.seats {
			if sb.running {
				for h := range sb.charged {
					union[h] = struct{}{}
				}
			}
		}
		if len(union) != g.index.RunningBlocks() {
			return fmt.Errorf("engine %q: run side %d blocks, running seats hold %d", id, g.index.RunningBlocks(), len(union))
		}
	}
	return nil
}

// TestTierReplay drives the replay workload with a tier. After every step the RUN side is exactly the
// running seats' keys; without requests placed off their seat, what runs never exceeds the GPU budget
// except one seat running alone; and the run gate, the seats beyond the GPU, and rotation all happen.
func TestTierReplay(t *testing.T) {
	for _, moves := range []bool{false, true} {
		for _, seed := range []int64{21, 22, 23, 24} {
			cfg := defaultConfig()
			cfg.TierCapacityTokens = 2000 * 64 // seats 1900 blocks against a 475-block GPU budget
			s, err := NewScheduler(cfg)
			if err != nil {
				t.Fatal(err)
			}
			held := prometheus.NewCounterVec(prometheus.CounterOpts{Name: "held"}, engineLabels)
			s.metrics = &groupMetrics{runGateHeld: held}
			var out strings.Builder
			var bad error
			beyondGPU := 0
			driveReplay(t, s, seed, 500, &out, replayOpts{noMoves: !moves}, func(step int) {
				if bad == nil {
					bad = runSideExact(s)
				}
				s.mu.Lock()
				defer s.mu.Unlock()
				for id, g := range s.groups {
					if g.index.UniqueBlocks() > g.runBudgetBlocks {
						beyondGPU++
					}
					if bad == nil && !moves && g.index.RunningSeats() > 1 && g.index.RunningBlocks() > g.runBudgetBlocks {
						bad = fmt.Errorf("step %d engine %q: %d blocks running over the %d-block GPU budget", step, id, g.index.RunningBlocks(), g.runBudgetBlocks)
					}
				}
			})
			if bad != nil {
				t.Fatalf("moves=%v seed %d: %v", moves, seed, bad)
			}
			gate := 0.0
			for _, e := range []string{"a", "b", "c"} {
				gate += testutil.ToFloat64(held.WithLabelValues(e))
			}
			snap := s.Snapshot(epoch)
			rotations := snap.Exits["quantum-complete"] + snap.Exits["idle"]
			if gate == 0 || beyondGPU == 0 || rotations == 0 {
				t.Fatalf("moves=%v seed %d: the workload must exercise the gate (%v passes held), seats beyond the GPU (%d engine-steps) and rotation (%d exits)", moves, seed, gate, beyondGPU, rotations)
			}
			t.Logf("moves=%v seed %d: gate held %v passes, seats beyond the GPU on %d engine-steps, %d rotations", moves, seed, gate, beyondGPU, rotations)
		}
	}
}
