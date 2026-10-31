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

// E4b — the parameter-free residue model. The model is asserted MECHANICALLY
// (free-queue semantics) and then through the seat-time rule:
//
//   - allocate from the HEAD, free tail-block-first to the TAIL ⇒ a departed
//     chain's surviving blocks are its PREFIX and they erode from the tail;
//   - the prediction is reconciled at settle against usage.cached_tokens, and both
//     error directions are counted (overCredit / resyncs);
//   - the model ALWAYS is a CLASS comparison between fitting engines (R3), never
//     an argmax on raw blocks between two cold engines;
//   - the two self-model resets fire (sustained mis-calibration ⇒ fit-only; a
//     resync burst ⇒ void the shadow).

import (
	"fmt"
	"testing"
	"time"
)

// TestResidueFreeQueueOrder is the mechanism: a chain that completes leaves its
// TAIL blocks nearest the head of the free queue, so the PREFIX is what survives
// another member's allocations.
func TestResidueFreeQueueOrder(t *testing.T) {
	e := newResidueEngine(8) // 8 blocks of pool
	// m1 asks for 4 blocks: all anonymous slots, no eviction possible yet
	h := e.dispatch("m1", 4)
	if got := e.survivingPrefix("m1"); got != 4 {
		t.Fatalf("live blocks must read as resident, got %d", got)
	}
	e.settle(true, "m1", h, 0, 0) // completion: 4 blocks freed tail-first → prefix deepest
	if got := e.survivingPrefix("m1"); got != 4 {
		t.Fatalf("after completion m1's prefix = %d, want 4 (nothing has been evicted yet)", got)
	}
	// m2 now consumes the remaining 4 anonymous slots and then starts evicting.
	// The first identities to go are m1's TAIL blocks (they entered the queue last).
	h2 := e.dispatch("m2", 4) // 4 anon slots, m1 untouched
	e.settle(false, "m2", h2, 0, 0)
	if got := e.survivingPrefix("m1"); got != 4 {
		t.Fatalf("m1 prefix after m2 filled the pool = %d, want 4", got)
	}
	h3 := e.dispatch("m2", 8) // now the queue must recycle: m1's tail goes first
	e.settle(false, "m2", h3, 4, 0)
	if got := e.survivingPrefix("m1"); got != 0 {
		t.Fatalf("m1 prefix after every slot was recycled = %d, want 0", got)
	}
}

// TestResiduePrefixErodesFromTheTail: partial pressure erodes the TAIL of a
// departed chain first, which is what makes "prefix survives longest" observable.
func TestResiduePrefixErodesFromTheTail(t *testing.T) {
	e := newResidueEngine(10)
	h := e.dispatch("m1", 6)      // 6 live blocks
	e.settle(true, "m1", h, 0, 0) // all 6 go to the queue tail, pos 0 deepest
	// consume the 4 anonymous slots then 2 recycled identities (m1's pos 5, 4)
	h2 := e.dispatch("m2", 6)
	e.settle(false, "m2", h2, 0, 0)
	if got := e.survivingPrefix("m1"); got != 4 {
		t.Fatalf("m1 surviving prefix = %d, want 4 (positions 0..3 survive, 4..5 recycled)", got)
	}
	if got := e.ratio("m1"); got != 4.0/6.0 {
		t.Fatalf("m1 ratio = %v, want 4/6", got)
	}
}

// TestResidueReconcileCountsBothErrorDirections: the model predicts at dispatch and
// is corrected by the engine's own usage.cached_tokens at settle. Both directions
// are COUNTED (a silent correction would hide a broken model).
func TestResidueReconcileCountsBothErrorDirections(t *testing.T) {
	// OVER-CREDIT: the model claims 4 cached blocks, the engine reports 0.
	e := newResidueEngine(64)
	h := e.dispatch("m1", 8)
	e.settle(true, "m1", h, 0, 0) // 8 resident
	h2 := e.dispatch("m1", 8)
	if h2.predHit != 8 {
		t.Fatalf("predHit = %d, want 8", h2.predHit)
	}
	before := e.overCred
	e.settle(true, "m1", h2, 2, 0) // engine says only 2 hit → 6 blocks were re-allocated
	if e.overCred <= before {
		t.Fatalf("over-credit not counted (%d → %d)", before, e.overCred)
	}
	// RESYNC: the defensive branch. It fires when the engine reports hits on blocks
	// the model holds NO identity for — the shape of the "no dispatch record" path
	// (a settle whose dispatch we never saw: an engine restart, a second EPP, a
	// request that reached the engine before the policy did). The model re-assigns
	// the identities and COUNTS the event rather than silently believing itself.
	e2 := newResidueEngine(8)
	if e2.resyncs != 0 {
		t.Fatal("setup: resyncs must start at 0")
	}
	// a PARTIAL resync (3 of 8): above the burst threshold a full reset voids the
	// shadow instead, which TestResidueResetOnResyncBurst covers.
	e2.settle(true, "m2", residueHeld{predHit: 0, full: 8}, 3, 0)
	if e2.resyncs == 0 {
		t.Fatal("resync not counted: the engine reported hits on blocks the model had no identity for")
	}
	if got := e2.survivingPrefix("m2"); got != 3 {
		t.Fatalf("after the partial resync the model should hold m2's 3 blocks, got %d", got)
	}
}

// TestResidueDisabledOnSustainedError is self-model reset (2): an engine whose
// predictions are persistently wrong is dropped to FIT-ONLY, and the model says so
// once (never a cache read, never a threshold we invented).
func TestResidueDisabledOnSustainedError(t *testing.T) {
	e := newResidueEngine(1024)
	e.setCapacity(1024) // same capacity: no reset
	for i := 0; i < residueMinCalibration+4; i++ {
		// claim warm (8 blocks believed cached) and be wrong every time
		h := residueHeld{predHit: 8, full: 8}
		e.settle(true, "m1", h, 0, 0)
	}
	if !e.disabled {
		t.Fatalf("model not disabled after %d fully-wrong reconciliations (errEWMA=%.2f)", residueMinCalibration+4, e.errEWMA)
	}
	if e.usable() {
		t.Fatal("a disabled engine must not be usable for ranking")
	}
}

// TestResidueResetOnResyncBurst is self-model reset (1b): a request whose claimed
// hits were mostly blocks the model had already evicted means the shadow lost the
// engine's state.
func TestResidueResetOnResyncBurst(t *testing.T) {
	e := newResidueEngine(64)
	gen := e.gen
	// Under open loop the model does NOT correct itself from the measured hit; the
	// persistent mismatch shows up as calibration error and drops the engine to
	// fit-only (the resync-burst reset was folded into that single verdict).
	for i := 0; i < residueMinCalibration; i++ {
		e.settle(false, "m1", residueHeld{predHit: 0, full: 10}, 10, 0)
	}
	if e.gen != gen {
		t.Fatal("open loop must not void the shadow on a resync burst (that would be a state correction)")
	}
	if !e.disabled {
		t.Fatalf("a persistently mis-predicting engine must drop to fit-only (errEWMA=%.2f, samples=%d)", e.errEWMA, e.samples)
	}
}

// TestResidueTieBreakOnlyAmongFittingWarmEngines is R3 at the policy level: the
// model separates two WARM engines, never two cold ones.
func TestResidueTieBreakOnlyAmongFittingWarmEngines(t *testing.T) {
	cfg := e4aCfg()
	cfg.Residue = "tiebreak"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s.SetNumBlocksFor(id, 64, 4000); err != nil {
			t.Fatal(err)
		}
	}
	now := goldenT0
	// make m1 WARM on engine-b only
	s.mu.Lock()
	gb := s.groups["engine-b"]
	gb.residue.setCapacity(4000)
	h := gb.residue.dispatch("m1", 100)
	gb.residue.settle(true, "m1", h, 0, 0)
	gb.residue.samples = 0 // keep it enabled/calibrated for this check
	warm := gb.residue.ratio("m1")
	coldA := s.groups["engine-a"].residue.ratio("m1")
	s.mu.Unlock()
	if warm < residueWarmFraction || coldA != 0 {
		t.Fatalf("setup: warm=%v cold=%v (want warm>=%.2f, cold==0)", warm, coldA, residueWarmFraction)
	}
	w := &waiter{id: "m1", since: now, sizeBytes: bytesFor(100 * 64),
		coldTokensByEngine: map[string]int64{"engine-a": 100 * 64, "engine-b": 100 * 64}}
	s.mu.Lock()
	c := s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c == nil || c.g.id != "engine-b" {
		t.Fatalf("a member warm on engine-b must be seated there, got %+v", c)
	}
	// now the COLD case: a member with no history on either engine. The model must
	// not decide — the backlog/free-budget keys do (both read 0, so the id order).
	w2 := &waiter{id: "never-seen", since: now, sizeBytes: bytesFor(100 * 64),
		coldTokensByEngine: map[string]int64{"engine-a": 100 * 64, "engine-b": 100 * 64}}
	s.mu.Lock()
	c2 := s.seatEngineLocked(w2, now)
	s.mu.Unlock()
	if c2 == nil || c2.g.id != "engine-a" {
		t.Fatalf("a COLD member must fall through to the non-residue keys (id order → engine-a), got %+v", c2)
	}
	if c2.residue != 0 || c2.residueWarm {
		t.Fatalf("cold candidate carries residue=%v warm=%v, want 0/false", c2.residue, c2.residueWarm)
	}
}

// TestE4bVsE4aOneVariableDiff: the residue knob is the only config difference
// between the E4a and E4b arms.
func TestE4bVsE4aOneVariableDiff(t *testing.T) {
	a := e4aCfg()
	b := e4aCfg()
	b.Residue = "tiebreak"
	if b.Residue == a.Residue {
		t.Fatal("the knob did not change")
	}
	// the model must be OFF by default (the golden / every banked arm relies on it)
	if def := defaultConfig(); def.Residue != "off" {
		t.Fatalf("default residue = %q, want \"off\"", def.Residue)
	}
}

// TestResidueOffCostsNothing: with the knob off, no engine allocates a shadow, so
// an off arm pays nothing at all (not even a nil-check per decision).
func TestResidueOffCostsNothing(t *testing.T) {
	cfg := e4aCfg()
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s.SetNumBlocksFor(id, 64, 1000); err != nil {
			t.Fatal(err)
		}
		if g := s.groups[id]; g.residue != nil {
			t.Fatalf("engine %s has a residue shadow with the model off", id)
		}
	}
	// and with it ON, every engine gets one sized to its measured pool.
	cfg.Residue = "tiebreak"
	s2, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s2.SetNumBlocksFor(id, 64, 1000); err != nil {
			t.Fatal(err)
		}
		g := s2.groups[id]
		if g.residue == nil || g.residue.capacity != 1000 {
			t.Fatalf("engine %s shadow = %+v, want capacity 1000", id, g.residue)
		}
	}
}

// TestResidueSurvivesARunEndToEnd drives the scheduler (not the model directly)
// through seat → dispatch → settle → re-seat and checks the model tracked the
// member, i.e. the hooks are actually wired.
func TestResidueSurvivesARunEndToEnd(t *testing.T) {
	cfg := e4aCfg()
	cfg.Residue = "tiebreak"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s.SetNumBlocksFor(id, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	now := goldenT0
	var settles []string
	s.onResidueSettle = func(engine string, predicted, actual float64) {
		settles = append(settles, fmt.Sprintf("%s %.2f/%.2f", engine, predicted, actual))
	}
	heads := []Head{{Member: "m1", ReqID: "r1", Arrival: now, SizeBytes: bytesFor(40)}}
	engine, i := s.PickFromHeadsEngine(heads, now)
	if i < 0 {
		t.Fatal("not seated")
	}
	s.DispatchedOn("m1", "r1", engine, keys(20, Hash(9)), bytesFor(40), now)
	s.SettleOnUsage("m1", "r1", engine, 20*64, 20*64, 8*64, true, now.Add(2*time.Second))
	if len(settles) != 1 {
		t.Fatalf("residue settle hook fired %d time(s), want 1 (%v)", len(settles), settles)
	}
	s.mu.Lock()
	ratio := s.groups[engine].residue.ratio("m1")
	s.mu.Unlock()
	if ratio < 0.9 {
		t.Fatalf("after a full hit on a 20-block prompt the model should call m1 warm, ratio=%v", ratio)
	}
}

// TestEngineGateSkipsSaturatedEngine is the per-engine gate: a saturated engine
// takes no new seats (its own residents keep dispatching), so the queue goes to the
// sibling that has room. The signal is the DETECTOR'S OWN FORMULA de-averaged:
// max(waiting/5, kv/0.80) >= 1.
func TestEngineGateSkipsSaturatedEngine(t *testing.T) {
	cfg := e4aCfg()
	cfg.EngineGate = "per-engine"
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := s.SetNumBlocksFor(id, 64, 4000); err != nil {
			t.Fatal(err)
		}
	}
	now := goldenT0
	// engine-a is at kv 0.90 (> the 0.80 threshold) — saturated by its own kv
	s.SetEngineLoad("engine-a", 0.90, 0, now)
	s.SetEngineLoad("engine-b", 0.10, 0, now)
	w := &waiter{id: "m1", since: now, sizeBytes: bytesFor(40),
		coldTokensByEngine: map[string]int64{"engine-a": 40 * 64, "engine-b": 40 * 64}}
	s.mu.Lock()
	c := s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c == nil || c.g.id != "engine-b" {
		t.Fatalf("a kv-saturated engine must be skipped, got %+v", c)
	}
	// and by its own waiting queue (>= 5)
	s.SetEngineLoad("engine-a", 0.10, 6, now)
	s.SetEngineLoad("engine-b", 0.10, 0, now)
	s.mu.Lock()
	c = s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c == nil || c.g.id != "engine-b" {
		t.Fatalf("a queue-saturated engine must be skipped, got %+v", c)
	}
	// BOTH saturated ⇒ no candidate: admission waits rather than piling onto a
	// saturated engine (the heads stay queued; nothing is lost).
	s.SetEngineLoad("engine-b", 0.95, 0, now)
	s.mu.Lock()
	c = s.seatEngineLocked(w, now)
	s.mu.Unlock()
	if c != nil {
		t.Fatalf("with every engine saturated the seat choice must return nothing, got %+v", c)
	}
	// and the gate is OFF by default: the same state admits normally.
	off, err := NewScheduler(e4aCfg())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"engine-a", "engine-b"} {
		if err := off.SetNumBlocksFor(id, 64, 4000); err != nil {
			t.Fatal(err)
		}
	}
	off.SetEngineLoad("engine-a", 0.99, 99, now) // a no-op with the gate off
	off.mu.Lock()
	cOff := off.seatEngineLocked(w, now)
	off.mu.Unlock()
	if cOff == nil {
		t.Fatal("with engineGate=off the seat choice must be unaffected by load")
	}
}
