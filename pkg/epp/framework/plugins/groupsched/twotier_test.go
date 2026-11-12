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

// THE TWO TIERS, one mechanism each: membership is max(GPU, CPU) and RUNNING is the GPU's, the gate
// that keeps the running set inside the GPU, head-of-line blocking MEASURED and not fixed, Rule O and
// the cold valve unchanged by the larger budget, and the running index cleaned by every way a request
// can end.

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/prometheus/client_golang/prometheus"
)

// tierSched is one engine at 64-token blocks with 1,000 measured blocks — a 950-block GPU share
// (0.95 x 1,000) — under whatever config the caller needs.
func tierSched(t *testing.T, mutate func(*Config)) *Scheduler {
	t.Helper()
	cfg := defaultConfig()
	if mutate != nil {
		mutate(&cfg)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("", 64, 1000); err != nil {
		t.Fatal(err)
	}
	return s
}

// cpuTierSched is the two-tier shape: a CPU tier of 1,900 blocks over the same 950-block GPU share, so
// MEMBERSHIP is 1,900 and the RUNNING cap stays 950.
func cpuTierSched(t *testing.T) *Scheduler {
	t.Helper()
	return tierSched(t, func(c *Config) {
		c.GroupBudgetSource = budgetSourceMax
		c.CpuTierTokens = 1900 * 64
	})
}

// turn is one queued head of a thread, with its own request id (a fan-out has several).
func turn(id, reqID string, ago time.Duration, blocks int) Head {
	return Head{Thread: id, ReqID: reqID, Arrival: epoch.Add(-ago), Chains: chain(id, blocks)}
}

// inFlight lands one dispatched request on its engine and completes its prefill: the cold-entry charge
// goes back to the engine's valve, so the NEXT member is not stagger-held behind it (the valve is the
// fixture's business, not the gate's).
func inFlight(t *testing.T, s *Scheduler, engine, reqID string) {
	t.Helper()
	s.AtPod(reqID, engine, epoch)
	s.FirstChunk(reqID, epoch)
}

// running is one engine's running side, read under the lock.
func running(s *Scheduler, engine string) (seqs, blocks, cap int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groups[engine]
	return g.running.Seqs(), g.running.Blocks(), g.runningCapBlocks
}

// charged is what one seated thread's record holds on its engine, in blocks.
func charged(s *Scheduler, engine, id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groups[engine].index.ChargedBlocks(id)
}

// exiting reports whether a thread has been marked on its way out.
func exiting(s *Scheduler, engine, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groups[engine].index.Exiting(id)
}

// liveRequests is how many tracked request records the scheduler still holds.
func liveRequests(s *Scheduler) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.reqs)
}

// TestMembershipIsLargerButRunningStaysGpuSized: the CPU tier sizes how many members may stay WARM
// (1,900 blocks: three 400-block members), the GPU sizes what RUNS (950: two of them), and a settle
// releases the gate so the waiting member runs next.
func TestMembershipIsLargerButRunningStaysGpuSized(t *testing.T) {
	s := cpuTierSched(t)
	for _, id := range []string{"a", "b"} {
		if i := s.Pass([]Head{head(id, 3*time.Second, 400)}, epoch); i != 0 {
			t.Fatalf("member %s must be seated and dispatched, got index %d", id, i)
		}
		inFlight(t, s, "", "r-"+id)
	}
	// the third member: SEATED (membership is the larger tier) but HELD at the gate (running is not)
	if i := s.Pass([]Head{head("c", 3*time.Second, 400)}, epoch); i != -1 {
		t.Fatalf("the third member must wait at the dispatch gate, got index %d", i)
	}
	if !seated(s, "c") {
		t.Fatal("membership must still take the third member: the CPU tier is the group's budget")
	}
	if seqs, blocks, cap := running(s, ""); seqs != 2 || blocks != 800 || cap != 950 {
		t.Fatalf("running must stay the GPU's: seqs=%d blocks=%d cap=%d, want 2/800/950", seqs, blocks, cap)
	}
	snap := s.Snapshot(epoch)
	if snap.Engines[""].BudgetBlocks != 1900 {
		t.Fatalf("membership budget: %d, want 1900 (max(GPU share, cpuTierTokens / blockSize))", snap.Engines[""].BudgetBlocks)
	}
	if snap.Seats != 3 {
		t.Fatalf("three members must be seated over a two-request GPU, got %d", snap.Seats)
	}
	// a settle frees a running slot, and the member that was waiting is dispatched on the next pass
	settle(t, s, "a")
	if seqs, blocks, _ := running(s, ""); seqs != 1 || blocks != 400 {
		t.Fatalf("one settle must give back exactly its own blocks: seqs=%d blocks=%d", seqs, blocks)
	}
	if i := s.Pass([]Head{head("c", 3*time.Second, 400)}, epoch); i != 0 {
		t.Fatalf("the waiting member must run once a slot is free, got index %d", i)
	}
}

// TestSeqCapHoldsTheHead: maxNumSeqs is a deployment fact (vLLM's --max-num-seqs) and holds the
// head whatever the blocks say. It is armed on its own: a GPU-sized group with a sequence cap is a
// legitimate configuration, and 0 means no sequence gate.
func TestSeqCapHoldsTheHead(t *testing.T) {
	s := tierSched(t, func(c *Config) { c.MaxNumSeqs = 1 })
	if i := s.Pass([]Head{head("a", 3*time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("the first member must dispatch, got %d", i)
	}
	inFlight(t, s, "", "r-a")
	if i := s.Pass([]Head{head("b", 2*time.Second, 100)}, epoch); i != -1 {
		t.Fatalf("maxNumSeqs 1 with one request in flight must hold the head, got %d", i)
	}
	settle(t, s, "a")
	if i := s.Pass([]Head{head("b", 2*time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("the settle must release the sequence gate, got %d", i)
	}
}

// TestHeadOfLineBlockingIsMeasuredNotFixed: v1 dispatches the per-engine OLDEST head and never skips
// ahead, so a big head that cannot fit blocks a small one that could. The test PINS the behaviour:
// the measurement is the point, the fix (backfill) is not in v1.
func TestHeadOfLineBlockingIsMeasuredNotFixed(t *testing.T) {
	s := cpuTierSched(t)
	// big's first turn is running: 700 blocks, so only 250 of the GPU's 950 are free
	if i := s.Pass([]Head{turn("big", "big-1", 20*time.Second, 700)}, epoch); i != 0 {
		t.Fatalf("big's first turn must dispatch, got %d", i)
	}
	inFlight(t, s, "", "big-1")
	// big's SECOND turn is older than the small member's head and its history has grown to 1,000
	// blocks: it is the engine's head, it does not fit (700 running + 300 new > 950), and the small
	// head behind it waits even though 700 + 100 would fit
	heads := []Head{turn("big", "big-2", 10*time.Second, 1000), turn("small", "small-1", time.Second, 100)}
	if i := s.Pass(heads, epoch); i != -1 {
		t.Fatalf("the big head must hold the engine (no skip-ahead in v1), got index %d", i)
	}
	// the slot frees, and the head — not the small one behind it — is what runs: the blocking is the
	// measured cost, and it is still there
	s.FirstChunk("big-1", epoch)
	s.Settle("big-1", "", fwkrequest.Usage{}, epoch)
	heads = []Head{turn("big", "big-2", 10*time.Second, 1000), turn("small", "small-1", time.Second, 100)}
	if i := s.Pass(heads, epoch); i != 0 || heads[i].ReqID != "big-2" {
		t.Fatalf("the oldest head must run first (FIFO, no backfill): index %d", i)
	}
}

// TestRuleOAtTheDispatchGate: an engine with NOTHING in flight runs one head whatever its size — the
// dispatch twin of Rule O. Without it a member whose chain exceeds the GPU share (a head Rule O
// itself seated) could never run, and its seat would be a stall.
func TestRuleOAtTheDispatchGate(t *testing.T) {
	s := cpuTierSched(t)
	// 1,500 blocks: over the 950-block GPU share, inside the 1,900-block membership
	if i := s.Pass([]Head{head("big", time.Second, 1500)}, epoch); i != 0 {
		t.Fatalf("an empty engine must run one head whatever its size, got %d", i)
	}
	if seqs, blocks, _ := running(s, ""); seqs != 1 || blocks != 1500 {
		t.Fatalf("the oversize head's own blocks are the running set: seqs=%d blocks=%d", seqs, blocks)
	}
}

// TestRuleOAndTheBudgetWithTheCpuTier: a head over the GPU share but inside the CPU tier is SEATED
// normally (no Rule O needed), and one over BOTH is admitted by Rule O on the empty engine, charging
// its full chain and keeping no-thrash protection.
func TestRuleOAndTheBudgetWithTheCpuTier(t *testing.T) {
	t.Run("inside the cpu tier: an ordinary seat", func(t *testing.T) {
		s := cpuTierSched(t)
		if i := s.Pass([]Head{head("m", time.Second, 1500)}, epoch); i != 0 {
			t.Fatalf("a 1,500-block head must be seated and dispatched, got %d", i)
		}
		if got := charged(s, "", "m"); got != 1500 {
			t.Fatalf("the seat charges the exact chain: %d blocks, want 1500", got)
		}
	})
	t.Run("over both: rule O, and the no-thrash clause holds", func(t *testing.T) {
		s := cpuTierSched(t)
		if i := s.Pass([]Head{head("huge", time.Second, 2000)}, epoch); i != 0 {
			t.Fatalf("rule O must admit an oversize head on an empty engine, got %d", i)
		}
		if got := charged(s, "", "huge"); got != 2000 {
			t.Fatalf("rule O charges the FULL chain: %d blocks, want 2000", got)
		}
		inFlight(t, s, "", "r-huge")
		// enforcement runs with the engine over its budget, and the sole oversize resident is not a
		// victim: evicting it frees nothing it could return to
		s.Pass(nil, epoch)
		if !seated(s, "huge") {
			t.Fatal("rule O's sole oversize resident must not be evicted (the no-thrash clause)")
		}
	})
}

// TestEnforceBudgetAtTheCpuBudget: memory pressure is enforced at the MEMBERSHIP budget, so growth
// past the GPU share is neither evicted nor blocked while it stays inside the CPU tier — and past the
// CPU tier the engine pays for it by evicting a resident.
func TestEnforceBudgetAtTheCpuBudget(t *testing.T) {
	s := cpuTierSched(t)
	// a second, cheaper resident: enforcement's own victim, so the growing member is never the engine's
	// SOLE resident (the no-thrash clause protects that one, and Rule O's test covers it)
	if i := s.Pass([]Head{head("filler", 5*time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("the filler must be seated and dispatched, got %d", i)
	}
	inFlight(t, s, "", "r-filler")
	settle(t, s, "filler")
	// a 1,000-block member: over the 950-block GPU share, inside the 1,900-block membership
	if i := s.Pass([]Head{head("m", 3*time.Second, 1000)}, epoch); i != 0 {
		t.Fatalf("the member must dispatch, got %d", i)
	}
	inFlight(t, s, "", "r-m")
	s.Settle("r-m", "", fwkrequest.Usage{}, epoch)
	s.Pass(nil, epoch) // enforcement, at the CPU budget
	if !seated(s, "m") || !seated(s, "filler") {
		t.Fatal("1,100 blocks of members must both stay resident: the CPU tier is the group's budget")
	}
	// the member grows past the CPU tier: 2,500 blocks on one engine
	if i := s.Pass([]Head{turn("m", "r-m", time.Second, 2500)}, epoch); i != 0 {
		t.Fatalf("the grown head must dispatch (an empty running set runs one head), got %d", i)
	}
	inFlight(t, s, "", "r-m")
	s.Pass(nil, epoch) // enforcement now sees 2,600 against 1,900
	if !seated(s, "m") {
		t.Fatal("the member in flight is not the victim: the cheaper resident pays")
	}
	if seated(s, "filler") {
		t.Fatal("growth past the CPU budget must be paid for: enforcement evicts a resident")
	}
}

// TestColdValveIsUnchangedByTheCpuTier: the cold-entry stagger cap is the GPU's own headroom, so a
// CPU-sized group does not enlarge it (the stagger serialises GPU prefill, not membership).
func TestColdValveIsUnchangedByTheCpuTier(t *testing.T) {
	gpu := tierSched(t, nil)
	cpu := cpuTierSched(t)
	gs, cs := gpu.Snapshot(epoch).Engines[""], cpu.Snapshot(epoch).Engines[""]
	if gs.ColdCapTokens != 3200 { // floor(0.05 x 1,000) x 64
		t.Fatalf("the GPU-sized group's cold cap: %d, want 3200", gs.ColdCapTokens)
	}
	if cs.ColdCapTokens != gs.ColdCapTokens {
		t.Fatalf("the cold valve must not follow the membership budget: %d with a CPU tier, %d without", cs.ColdCapTokens, gs.ColdCapTokens)
	}
	if cs.BudgetBlocks == gs.BudgetBlocks {
		t.Fatal("fixture: the CPU tier must actually enlarge the membership budget")
	}
}

// TestTheCpuTierNeverShrinksTheGroup: the source is a MAXIMUM, so a CPU tier smaller than the GPU's
// own share leaves the group exactly where it was — the tier can only ever add membership.
func TestTheCpuTierNeverShrinksTheGroup(t *testing.T) {
	s := tierSched(t, func(c *Config) {
		c.GroupBudgetSource, c.CpuTierTokens = budgetSourceMax, 100*64 // 100 blocks < the 950-block share
	})
	es := s.Snapshot(epoch).Engines[""]
	if es.BudgetBlocks != 950 || es.RunningCapBlocks != 950 {
		t.Fatalf("a smaller CPU tier must not shrink the group: budget=%d cap=%d, want 950/950", es.BudgetBlocks, es.RunningCapBlocks)
	}
}

// TestEngineFlapCleansTheRunningIndex: an engine that disappears takes its in-flight records with
// it, and the group that comes back is EMPTY — no running count, no blocks, nothing left to charge
// the next pass for.
func TestEngineFlapCleansTheRunningIndex(t *testing.T) {
	s := tierSched(t, func(c *Config) {
		c.GroupBudgetSource = budgetSourceMax
		c.CpuTierTokens = 1900 * 64
	})
	for _, e := range []string{"e0", "e1"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	if i := s.Pass([]Head{head("m", time.Second, 400)}, epoch); i != 0 {
		t.Fatalf("the member must dispatch, got %d", i)
	}
	inFlight(t, s, "e0", "r-m") // the seat-time choice is e0 (lowest id on a tie)
	if seqs, blocks, _ := running(s, "e0"); seqs != 1 || blocks != 400 {
		t.Fatalf("e0 must hold the in-flight request: seqs=%d blocks=%d", seqs, blocks)
	}
	if n := s.RemoveEngine("e0", epoch); n != 1 {
		t.Fatalf("the flap must exit the engine's thread, got %d", n)
	}
	if liveRequests(s) != 0 {
		t.Fatalf("the flap must sweep the thread's records, %d left", liveRequests(s))
	}
	if err := s.SetNumBlocksFor("e0", 64, 1000); err != nil {
		t.Fatal(err)
	}
	if seqs, blocks, _ := running(s, "e0"); seqs != 0 || blocks != 0 {
		t.Fatalf("the re-added engine starts empty: seqs=%d blocks=%d", seqs, blocks)
	}
	// and it counts exactly, once, from there
	if i := s.Pass([]Head{head("n", time.Second, 300)}, epoch); i != 0 {
		t.Fatalf("a fresh dispatch after the flap must be served, got %d", i)
	}
	inFlight(t, s, "e0", "r-n")
	if seqs, blocks, _ := running(s, "e0"); seqs != 1 || blocks != 300 {
		t.Fatalf("after the flap the count is exact: seqs=%d blocks=%d, want 1/300", seqs, blocks)
	}
}

// TestAbandonReapAndExpiredWindowCleanTheRunningIndex: every way a request can end gives its running
// slot back — an abandon, the reap, and a released request that never reached a pod at all.
func TestAbandonReapAndExpiredWindowCleanTheRunningIndex(t *testing.T) {
	s := cpuTierSched(t)
	t.Run("abandon", func(t *testing.T) {
		if i := s.Pass([]Head{head("a", time.Second, 400)}, epoch); i != 0 {
			t.Fatalf("must dispatch, got %d", i)
		}
		inFlight(t, s, "", "r-a")
		if seqs, _, _ := running(s, ""); seqs != 1 {
			t.Fatalf("the request is in flight: %d", seqs)
		}
		s.Abandon("r-a", epoch)
		if seqs, blocks, _ := running(s, ""); seqs != 0 || blocks != 0 {
			t.Fatalf("an abandon must give the slot back: seqs=%d blocks=%d", seqs, blocks)
		}
	})
	t.Run("reap", func(t *testing.T) {
		if i := s.Pass([]Head{head("b", time.Second, 400)}, epoch); i != 0 {
			t.Fatalf("must dispatch, got %d", i)
		}
		inFlight(t, s, "", "r-b")
		later := epoch.Add(time.Duration(s.cfg.InflightReapSeconds+1) * time.Second)
		s.Pass(nil, later)
		if seqs, blocks, _ := running(s, ""); seqs != 0 || blocks != 0 {
			t.Fatalf("the reap must give the slot back: seqs=%d blocks=%d", seqs, blocks)
		}
		if liveRequests(s) != 0 {
			t.Fatalf("the reap must close the record, %d left", liveRequests(s))
		}
	})
	t.Run("a released request that never reached a pod", func(t *testing.T) {
		if i := s.Pass([]Head{head("w", time.Second, 400)}, epoch); i != 0 {
			t.Fatalf("must dispatch, got %d", i)
		}
		// no AtPod: the framework never brought it to a pod, so its window is the only thing holding it
		if seqs, _, _ := running(s, ""); seqs != 1 {
			t.Fatalf("the released request is in flight until its window dies: %d", seqs)
		}
		s.Pass(nil, epoch.Add(21*time.Second)) // > idleSeconds (S = 20)
		if seqs, blocks, _ := running(s, ""); seqs != 0 || blocks != 0 {
			t.Fatalf("a dead window must give the slot back: seqs=%d blocks=%d", seqs, blocks)
		}
	})
}

// TestLatePreRequestReopensItsRunningSlot: a released request whose window outlived S gave its slot back
// (decision: a dead window is not in flight). If the request reaches its pod anyway it IS in flight, so the
// slot is re-opened UNGATED — over-counting a real request is safe, under-counting it is not — and the
// event is counted, because it means Pick->PreRequest took longer than S.
func TestLatePreRequestReopensItsRunningSlot(t *testing.T) {
	s := tierSched(t, func(c *Config) { c.MaxNumSeqs = 1 })
	if i := s.Pass([]Head{head("late", time.Second, 600)}, epoch); i != 0 {
		t.Fatalf("the head must dispatch, got %d", i)
	}
	// the pass at t+21 s: the window is dead, so the slot comes back
	later := epoch.Add(21 * time.Second)
	s.Pass(nil, later)
	if seqs, _, _ := running(s, ""); seqs != 0 {
		t.Fatalf("a dead window must not hold a slot: %d", seqs)
	}
	// ... and then the request arrives after all
	s.AtPod("r-late", "", later)
	if seqs, blocks, _ := running(s, ""); seqs != 1 || blocks != 600 {
		t.Fatalf("the late arrival must be charged again: seqs=%d blocks=%d, want 1/600", seqs, blocks)
	}
	s.mu.Lock()
	if got := s.counters.anomalies[anomalyLatePreRequest]; got != 1 {
		t.Fatalf("the late arrival must be counted once, got %d", got)
	}
	s.mu.Unlock()
	// and the seq gate holds the next head again: the re-opened charge is real
	if i := s.Pass([]Head{head("next", time.Second, 100)}, later); i != -1 {
		t.Fatalf("the re-opened charge must close the sequence gate, got %d", i)
	}
}

// TestLatePreRequestAfterTheSweepIsCounted: the residual v1 cannot fix — the thread's exit swept the record
// before the request arrived, so the request runs untracked. The sweep is where it is counted, because after
// that delete no hook can see it.
func TestLatePreRequestAfterTheSweepIsCounted(t *testing.T) {
	s := tierSched(t, func(c *Config) { c.MaxNumSeqs = 2 })
	if i := s.Pass([]Head{head("gone", time.Second, 100)}, epoch); i != 0 {
		t.Fatalf("the head must dispatch, got %d", i)
	}
	later := epoch.Add(21 * time.Second)
	s.Pass(nil, later) // the window dies
	if n := s.RemoveEngine("", later); n != 1 {
		t.Fatalf("the engine teardown must exit its thread, got %d", n)
	}
	s.mu.Lock()
	late := s.counters.anomalies[anomalyLatePreRequest]
	s.mu.Unlock()
	if late != 1 {
		t.Fatalf("the swept dead window must be counted once, got %d", late)
	}
}

// TestTheGateIsExactAtTheCap: the boundary itself. A head that brings the running set EXACTLY to the cap is
// dispatched; one block more is held (an off-by-one here would either waste the last slot or over-admit).
func TestTheGateIsExactAtTheCap(t *testing.T) {
	s := cpuTierSched(t) // running cap 950, membership 1,900
	if i := s.Pass([]Head{head("a", 3*time.Second, 700)}, epoch); i != 0 {
		t.Fatalf("the first head must dispatch, got %d", i)
	}
	inFlight(t, s, "", "r-a")
	// b's chain grows out of a's, so it adds exactly 250 blocks: 700 + 250 = 950 = the cap
	if i := s.Pass([]Head{turn("a", "r-a2", 2*time.Second, 950)}, epoch); i != 0 {
		t.Fatalf("a head that lands EXACTLY on the cap must be dispatched, got %d", i)
	}
	if seqs, blocks, cap := running(s, ""); seqs != 2 || blocks != 950 || cap != 950 {
		t.Fatalf("running must be exactly at the cap: seqs=%d blocks=%d cap=%d", seqs, blocks, cap)
	}
	// one more block anywhere is over
	if i := s.Pass([]Head{turn("b", "r-b", time.Second, 1)}, epoch); i != -1 {
		t.Fatalf("one block past the cap must be held, got %d", i)
	}
}

// TestRunningUnderflowIsCountedNotMasked: a double release cannot happen through the public entry points, so
// it is provoked at the index: the counter still clamps (the process must not die under the scheduler's mutex
// for an accounting bug) but the event is counted, never silently swallowed.
func TestRunningUnderflowIsCountedNotMasked(t *testing.T) {
	s := cpuTierSched(t)
	r := newRunningIndex()
	set := chargeSet(chain("x", 10))
	r.add(set)
	if !r.release(set) {
		t.Fatal("a matching release must report clean")
	}
	if r.release(set) {
		t.Fatal("a second release must report the underflow")
	}
	if r.Seqs() != 0 || r.Blocks() != 0 {
		t.Fatalf("the counters still clamp: seqs=%d blocks=%d", r.Seqs(), r.Blocks())
	}
	// the scheduler books it
	rec := &reqRec{chains: chain("x", 10), runSet: set, runningOn: s.groups[""]}
	s.mu.Lock()
	s.unchargeRunning(rec)
	got := s.counters.anomalies[anomalyRunningUnderflow]
	s.mu.Unlock()
	if got != 1 {
		t.Fatalf("the masked underflow must be counted once, got %d", got)
	}
}

// TestRunningIndexIsDedupedAndRefcounted: the running set counts BLOCKS, not charges — two requests
// of one thread share their prefix, and the shared keys stay while either of them is in flight.
func TestRunningIndexIsDedupedAndRefcounted(t *testing.T) {
	r := newRunningIndex()
	short, long := chargeSet(chain("m", 100)), chargeSet(chain("m", 150))
	r.add(short)
	if r.Seqs() != 1 || r.Blocks() != 100 {
		t.Fatalf("one request of 100 blocks: seqs=%d blocks=%d", r.Seqs(), r.Blocks())
	}
	if got := r.wouldAdd(chain("m", 150)); got != 50 {
		t.Fatalf("a grown turn adds only its new blocks: %d, want 50", got)
	}
	r.add(long)
	if r.Seqs() != 2 || r.Blocks() != 150 {
		t.Fatalf("the shared prefix counts once: seqs=%d blocks=%d, want 2/150", r.Seqs(), r.Blocks())
	}
	r.release(short)
	if r.Seqs() != 1 || r.Blocks() != 150 {
		t.Fatalf("the shared prefix stays while the longer request runs: seqs=%d blocks=%d", r.Seqs(), r.Blocks())
	}
	r.release(long)
	if r.Seqs() != 0 || r.Blocks() != 0 {
		t.Fatalf("nothing in flight, nothing held: seqs=%d blocks=%d", r.Seqs(), r.Blocks())
	}
}

// TestTwoTierConfigFailsClosed: every knob of the two-tier policy refuses a config that would run a
// policy nobody configured — including a CPU tier the source does not use.
func TestTwoTierConfigFailsClosed(t *testing.T) {
	ok := func(mutate func(*Config)) bool {
		cfg := defaultConfig()
		mutate(&cfg)
		_, err := NewScheduler(cfg)
		return err == nil
	}
	if !ok(func(c *Config) {}) {
		t.Fatal("the default config (gpu, no CPU tier, no sequence gate) must be accepted")
	}
	if !ok(func(c *Config) { c.GroupBudgetSource, c.CpuTierTokens = budgetSourceMax, 87_381*64 }) {
		t.Fatal("max with a CPU tier must be accepted")
	}
	if !ok(func(c *Config) { c.MaxNumSeqs = 128 }) {
		t.Fatal("a sequence cap on a GPU-sized group must be accepted")
	}
	for _, c := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"max without a cpu tier", func(c *Config) { c.GroupBudgetSource = budgetSourceMax }},
		{"max with a negative cpu tier", func(c *Config) { c.GroupBudgetSource, c.CpuTierTokens = budgetSourceMax, -1 }},
		{"a cpu tier the gpu source would ignore", func(c *Config) { c.CpuTierTokens = 121_600 }},
		{"an unknown source", func(c *Config) { c.GroupBudgetSource = "cpu" }},
		{"a negative sequence cap", func(c *Config) { c.MaxNumSeqs = -1 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			if ok(c.mutate) {
				t.Fatalf("%s must be refused at startup", c.name)
			}
		})
	}
}

// TestTwoTierSeriesAreExported: the five new series exist with their labels, the gate's held counter
// moves when a head waits, and the member-return hit records the response's own cached ÷ prompt for a
// member's later turn.
func TestTwoTierSeriesAreExported(t *testing.T) {
	const params = `{"groupBudgetSource":"max","cpuTierTokens":121600,"maxNumSeqs":8,"blockSizeTokens":64}`
	built, err := Factory("gs", json.NewDecoder(strings.NewReader(params)), nil)
	if err != nil {
		t.Fatal(err)
	}
	gs := built.(*GroupScheduler)
	reg := prometheus.NewRegistry()
	if err := gs.metrics.register(reg); err != nil {
		t.Fatal(err)
	}
	s := gs.sched
	if err := s.SetNumBlocksFor("e0", 64, 1000); err != nil {
		t.Fatal(err)
	}
	// one member, two turns: the first (a later turn of the thread, so a RETURNER) fills the GPU
	// share, the grown second is held at the gate meanwhile
	first := turn("m", "r-m", 3*time.Second, 900)
	first.Returner = true
	if i := s.Pass([]Head{first}, epoch); i != 0 {
		t.Fatalf("the first turn must dispatch, got %d", i)
	}
	s.AtPod("r-m", "e0", epoch)
	s.FirstChunk("r-m", epoch)
	second := Head{Thread: "m", ReqID: "r-m-2", Arrival: epoch.Add(-2 * time.Second), Chains: chain("m", 1200)}
	if i := s.Pass([]Head{second}, epoch); i != -1 {
		t.Fatalf("the second turn must be held at the gate, got %d", i)
	}
	// the settle of the first carries the response's own prompt-token details: 1,000 of 2,000 cached
	s.Settle("r-m", "e0", fwkrequest.Usage{PromptTokens: 2000,
		PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 1000}}, epoch)

	// a SECOND turn of the same member, this time NOT a seat entry: kind=member
	third := turn("m", "r-m-3", time.Second, 300)
	third.Returner = true
	if i := s.Pass([]Head{third}, epoch); i != 0 {
		t.Fatalf("the third turn must dispatch once the gate frees, got %d", i)
	}
	s.AtPod("r-m-3", "e0", epoch)
	s.FirstChunk("r-m-3", epoch)
	s.Settle("r-m-3", "e0", fwkrequest.Usage{PromptTokens: 400,
		PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 400}}, epoch)

	mfs, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	var gateHeld float64
	kinds := map[string][2]float64{} // kind -> {samples, sum}
	for _, mf := range mfs {
		found[mf.GetName()] = true
		switch mf.GetName() {
		case "llm_d_epp_group_dispatch_gate_held_total":
			for _, m := range mf.GetMetric() {
				for _, l := range m.GetLabel() {
					if l.GetName() == "reason" && l.GetValue() == gateReasonMemory {
						gateHeld += m.GetCounter().GetValue()
					}
				}
			}
		case "llm_d_epp_group_member_return_hit_ratio":
			for _, m := range mf.GetMetric() {
				kind := ""
				for _, l := range m.GetLabel() {
					if l.GetName() == "kind" {
						kind = l.GetValue()
					}
				}
				k := kinds[kind]
				k[0] += float64(m.GetHistogram().GetSampleCount())
				k[1] += m.GetHistogram().GetSampleSum()
				kinds[kind] = k
			}
		}
	}
	for _, want := range []string{
		"llm_d_epp_group_running_seqs", "llm_d_epp_group_running_blocks",
		"llm_d_epp_group_dispatch_gate_held_total", "llm_d_epp_group_budget_source",
		"llm_d_epp_group_member_return_hit_ratio",
	} {
		if !found[want] {
			t.Fatalf("%s must be exported", want)
		}
	}
	if gateHeld < 1 {
		t.Fatalf("the memory half of the gate must be counted, got %v", gateHeld)
	}
	// the FIRST settled turn was dispatched on a seat granted in that same pass: a RE-ENTRY (0.5 cached)
	// the THIRD ran on a residency that was already there: a MEMBER turn (1.0)
	if got := kinds[memberReturnReentry]; got[0] != 1 || got[1] != 0.5 {
		t.Fatalf("kind=reentry must carry the seat-entry turn: %v samples, sum %v (want 1 and 0.5)", got[0], got[1])
	}
	if got := kinds[memberReturnMember]; got[0] != 1 || got[1] != 1 {
		t.Fatalf("kind=member must carry the resident turn: %v samples, sum %v (want 1 and 1)", got[0], got[1])
	}
}
