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

// THE QUEUE-MODULE SUITE — bounded HRRN (`entryOrder: "hrrn-bounded"`, F5′).
//
// Every behavioural test runs at N=1 AND N=2 through forEngines. The queue is ONE
// shared map at both engine counts, so the twin rule is cheap here and worth keeping:
// the value is selected per config, never per topology, and a rule that only worked on
// the single-engine fixture would be a bug by definition.
//
// The unit-level tests install waiter sets DIRECTLY (the queue module's input IS that
// map) and read the module's answer through `mod.entry.Head` — no seat pass, no
// engine, no bookkeeping in the way. Every waiter is created as a NEWCOMER, which makes
// the newcomer class tie and leaves the response ratio deciding: that is the property
// under test. The class itself is pinned separately (TestHrrnBoundedKeepsTheArchiveClass).

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// hrrnSpec is one waiter of a scenario: how long it has been waiting and how many cold
// tokens its head must prefill.
type hrrnSpec struct {
	id   string
	wait time.Duration
	cold int64
}

// queue installs a waiter set on the fixture's shared queue, all in ONE newcomer class.
func (fx *engineSet) queue(specs ...hrrnSpec) {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	fx.s.waiters = make(map[string]*waiter, len(specs))
	for _, sp := range specs {
		fx.s.waiters[sp.id] = &waiter{
			id:         sp.id,
			newcomer:   true,
			since:      fx.clock.Add(-sp.wait),
			coldTokens: sp.cold,
		}
	}
}

// headOf is the QUEUE module's answer over the fixture's current waiter set.
func (fx *engineSet) headOf() *waiter {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	return fx.s.mod.entry.Head(fx.s.waiters, EntryCtx{Now: fx.clock})
}

// priorityOf is one waiter's bounded response ratio in the CURRENT pass (the module
// recomputes the pass's bound from the queue, exactly as the scheduler's pass does).
func (fx *engineSet) priorityOf(id string) float64 {
	fx.s.mu.Lock()
	defer fx.s.mu.Unlock()
	h := fx.s.mod.entry.(HrrnBounded)
	bound := lowerMedianColdTokens(fx.s.waiters, h.BlockSize)
	return h.priority(fx.s.waiters[id], bound, fx.clock)
}

// boundedConfig is the default config with the queue module switched to F5′.
func boundedConfig() Config {
	c := defaultConfig()
	c.EntryOrder = "hrrn-bounded"
	return c
}

// ---- (1) small-first survives INSIDE the small class ----

// TestHrrnBoundedSmallFreshBeatsBigFresh: among waiters of the SAME (short) wait, a
// small entry goes before a big one. The mechanism is the LOWER median: a waiter below
// it keeps its own size, so its ratio is strictly higher at an equal wait, while the big
// one is capped AT the median and therefore only ties with median-sized waiters.
//
// The population matters and is not incidental: with two waiters {3K, 200K} the lower
// median IS 3K, both ratios are 3K-denominated and they tie. The small class's
// small-first ordering is a property of a POPULATION with a small class in it, which is
// what a real queue looks like (B1b: nineteen 3K entries and one 200K one).
func TestHrrnBoundedSmallFreshBeatsBigFresh(t *testing.T) {
	forEngines(t, boundedConfig(), func(t *testing.T, fx *engineSet) {
		const cold = time.Duration(10) * time.Second
		fx.queue(
			hrrnSpec{"small", cold, 3_000},
			hrrnSpec{"m1", cold, 6_000},
			hrrnSpec{"m2", cold, 6_000},
			hrrnSpec{"big", cold, 200_000},
		)
		if got := lowerMedianColdTokens(fx.s.waiters, 64); got != 6_000 {
			t.Fatalf("pass bound = %d, want the LOWER median 6000 of {3000,6000,6000,200000}", got)
		}
		pSmall, pBig, pM1 := fx.priorityOf("small"), fx.priorityOf("big"), fx.priorityOf("m1")
		if !(pSmall > pBig) {
			t.Fatalf("at an equal wait the 3K entry (%v) must outrank the 200K one (%v): the small class is served first", pSmall, pBig)
		}
		if pBig != pM1 {
			t.Fatalf("the 200K entry (%v) must age exactly at the MEDIAN rate (%v): that is what the cap does", pBig, pM1)
		}
		if h := fx.headOf(); h == nil || h.id != "small" {
			t.Fatalf("head = %v, want small", h)
		}
	})
}

// ---- (2) the no-starvation intent: a big waiter DOES overtake, at the median rate ----

// TestHrrnBoundedBigOvertakesTheSmallClassOnTheMedianRate is the fix's intent, stated as
// the two takeover bounds it differs from:
//
//	uncapped HRRN: wait_big > ratio_small × cold_big = (600/3000) × 200,000 = 40,000 s
//	F5′ (bounded): wait_big > ratio_small × MEDIAN    = (600/3000) ×   6,000 =  1,200 s
//
// and it asserts the crossing lands where the MEDIAN puts it, not where the big entry's
// own size would (that is the measured B1b defect: 3,601 s max entry wait, 20 stalled
// sessions). Two big entries of DIFFERENT sizes cross at the SAME wait, which is the
// "equal slowdown above the median" property the cap is for.
func TestHrrnBoundedBigOvertakesTheSmallClassOnTheMedianRate(t *testing.T) {
	forEngines(t, boundedConfig(), func(t *testing.T, fx *engineSet) {
		const (
			smallWait      = 600 * time.Second
			smallCold      = int64(3_000)
			medianCold     = int64(6_000)
			bigCold        = int64(200_000)
			hugeCold       = int64(400_000)
			uncappedBounds = 40_000 * time.Second // ratio_small × cold_big
			boundedBounds  = 1_200 * time.Second  // ratio_small × median
		)
		smallClass := []hrrnSpec{
			{"s1", smallWait, smallCold},
			{"m1", smallWait, medianCold},
			{"m2", smallWait, medianCold},
		}
		// (a) below the crossing the small class is still served first: the takeover is
		// bounded, NOT immediate — a big entry never jumps a queue it has not aged in.
		fx.queue(append(append([]hrrnSpec{}, smallClass...),
			hrrnSpec{"big", boundedBounds - 1, bigCold})...)
		if h := fx.headOf(); h == nil || h.id == "big" {
			t.Fatalf("head = %v at %v, want the small class (the bounded crossing is %v)", h, boundedBounds-1, boundedBounds)
		}
		// (b) one instant past it, the big entry IS the head.
		fx.queue(append(append([]hrrnSpec{}, smallClass...),
			hrrnSpec{"big", boundedBounds + 1, bigCold})...)
		if h := fx.headOf(); h == nil || h.id != "big" {
			t.Fatalf("head = %v at %v, want big: bounded HRRN's crossing is %v", h, boundedBounds+1, boundedBounds)
		}
		// (c) ... and it is nowhere near the UNCAPPED crossing: the size gap is the
		// whole defect, and the cap deletes it.
		fx.queue(append(append([]hrrnSpec{}, smallClass...),
			hrrnSpec{"big", uncappedBounds - 1, bigCold})...)
		if h := fx.headOf(); h == nil || h.id != "big" {
			t.Fatalf("head = %v at %v: bounded HRRN must have served the big entry long before %v (its own-size crossing)", h, uncappedBounds-1, uncappedBounds)
		}
		// (d) EQUAL SLOWDOWN ABOVE THE MEDIAN: a 400K entry crosses at exactly the same
		// wait as the 200K one — both age at the median rate, so neither can buy extra
		// waiting time with size.
		fx.queue(append(append([]hrrnSpec{}, smallClass...),
			hrrnSpec{"huge", boundedBounds + 1, hugeCold})...)
		if h := fx.headOf(); h == nil || h.id != "huge" {
			t.Fatalf("head = %v, want huge: above the median the size no longer buys waiting time", h)
		}
		if got, want := fx.priorityOf("huge"), (boundedBounds+1).Seconds()/float64(medianCold); got != want {
			t.Fatalf("the 400K entry's priority = %v, want the same median rate (%v) the 200K one gets", got, want)
		}
	})
}

// ---- (3) the clamp, stated as "a 100K-token waiter ages like a median waiter" ----

// TestHrrnBoundedClampsA100KTokenWaiterToTheMedianRate pins the arithmetic of s_eff:
//
//	s_eff = min(cold tokens, the LOWER-median cold tokens), floored at ONE BLOCK
//
// so a 100K-token waiter's ratio is EXACTLY the ratio of a median-sized waiter with the
// same wait (and of any other waiter at or above the median), it is linear in the wait
// (twice the wait, twice the ratio — the prediction the takeover bound rests on), a
// waiter BELOW the median keeps its own size, and a lone waiter is its own median (the
// cap cannot bind, which is the no-op case).
func TestHrrnBoundedClampsA100KTokenWaiterToTheMedianRate(t *testing.T) {
	forEngines(t, boundedConfig(), func(t *testing.T, fx *engineSet) {
		// four 3K waiters put the lower median at 3K; 100K/150K are far above it
		median := []hrrnSpec{
			{"a", 0, 3_000}, {"b", 0, 3_000}, {"c", 0, 3_000}, {"d", 0, 3_000},
		}
		const wait = 100 * time.Second
		fx.queue(append(append([]hrrnSpec{}, median...),
			hrrnSpec{"warm", wait, 0},     // a fully warm head: floored at ONE BLOCK
			hrrnSpec{"tiny", wait, 1_000}, // below the median: keeps its own size
			hrrnSpec{"big100", wait, 100_000},
			hrrnSpec{"big150", wait, 150_000},
		)...)
		bound := lowerMedianColdTokens(fx.s.waiters, 64)
		if bound != 3_000 {
			t.Fatalf("pass bound = %d, want 3000 (the lower median of four 3K waiters)", bound)
		}
		wantMedianRate := wait.Seconds() / float64(bound)
		for _, id := range []string{"big100", "big150"} {
			if got := fx.priorityOf(id); got != wantMedianRate {
				t.Fatalf("%s priority = %v, want the MEDIAN rate %v — a big waiter must age exactly like the median one", id, got, wantMedianRate)
			}
		}
		if fx.priorityOf("big100") != fx.priorityOf("big150") {
			t.Fatal("the clamp must make every waiter at or above the median age at ONE rate")
		}
		// a waiter BELOW the median keeps its own (smaller) denominator, hence a HIGHER
		// ratio — the small class's ordering is untouched by the cap.
		if got, want := fx.priorityOf("tiny"), wait.Seconds()/1_000.0; got != want {
			t.Fatalf("tiny priority = %v, want its own size's rate %v", got, want)
		}
		// a fully warm head (0 cold tokens) is floored at ONE BLOCK, never zero: a zero
		// denominator would make its ratio infinite and collapse the median to 0
		// (silently restoring the uncapped rule).
		if got, want := fx.priorityOf("warm"), wait.Seconds()/64.0; got != want {
			t.Fatalf("warm priority = %v, want one block's rate %v", got, want)
		}
		// LINEAR IN THE WAIT at the clamped rate: this is the property the takeover bound
		// is derived from (nothing else in the ratio moves with time).
		before := fx.priorityOf("big100")
		fx.clock = fx.clock.Add(wait)
		if after, want := fx.priorityOf("big100"), 2*wantMedianRate; after != want {
			t.Fatalf("doubling the wait gave %v, want exactly 2x (%v)", after, want)
		}
		_ = before
		// a LONE waiter is its own median: the cap cannot bind, so the rule is uncapped
		// HRRN for it (the no-op case, and why an empty/single queue needs no special path).
		fx.queue(hrrnSpec{"solo", wait, 100_000})
		if got, want := fx.priorityOf("solo"), wait.Seconds()/100_000.0; got != want {
			t.Fatalf("a lone waiter's priority = %v, want its own size's rate %v (no cap binds)", got, want)
		}
		if h := fx.headOf(); h == nil || h.id != "solo" {
			t.Fatalf("head = %v, want solo", h)
		}
		// and an EMPTY queue has no head at all.
		fx.queue()
		if h := fx.headOf(); h != nil {
			t.Fatalf("an empty queue must have no head, got %v", h)
		}
	})
}

// ---- (4) determinism ----

// TestHrrnBoundedIsATotalOrder is the campaign's nondeterminism lesson made structural
// (the archive produced 3 different traces in 4 replays of the same scenario): the order
// must ignore map iteration. The same waiter set is installed in many different
// insertion orders and every one of them must answer with the same head — including the
// fully-tied case, where only the id can decide.
func TestHrrnBoundedIsATotalOrder(t *testing.T) {
	forEngines(t, boundedConfig(), func(t *testing.T, fx *engineSet) {
		set := []hrrnSpec{
			{"p", 0, 3_000}, {"q", 0, 3_000}, {"r", 0, 3_000}, {"s", 30 * time.Second, 3_000},
			{"t", 30 * time.Second, 3_000}, {"u", 900 * time.Second, 250_000},
			{"v", 900 * time.Second, 400_000}, {"w", 60 * time.Second, 12_000},
			{"x", 0, 64}, {"y", 1 * time.Second, 0},
		}
		want := ""
		for shift := 0; shift < len(set); shift++ {
			rotated := append(append([]hrrnSpec{}, set[shift:]...), set[:shift]...)
			fx.queue(rotated...)
			h := fx.headOf()
			if h == nil {
				t.Fatal("a non-empty queue must have a head")
			}
			if want == "" {
				want = h.id
				continue
			}
			if h.id != want {
				t.Fatalf("insertion order %d changed the head: %s != %s (map order must not leak into a decision)", shift, h.id, want)
			}
		}
		// the fully-tied case: same class, same wait, same size -> the id decides, always
		for shift := 0; shift < 4; shift++ {
			tied := []hrrnSpec{{"m3", 5 * time.Second, 3_000}, {"m1", 5 * time.Second, 3_000},
				{"m4", 5 * time.Second, 3_000}, {"m2", 5 * time.Second, 3_000}}
			tied = append(append([]hrrnSpec{}, tied[shift:]...), tied[:shift]...)
			fx.queue(tied...)
			if h := fx.headOf(); h == nil || h.id != "m1" {
				t.Fatalf("a fully tied set must resolve to the LOWEST id, got %v", h)
			}
		}
	})
}

// TestHrrnBoundedReplayIsDeterministic runs the FULL 2,600 s replay three times with the
// queue module switched to F5′ and requires identical traces. The unit test above pins
// the ordering rule; this pins the whole stack around it (seats, exits, accounting,
// placement) under the config the wave will actually run.
func TestHrrnBoundedReplayIsDeterministic(t *testing.T) {
	first := runGoldenScenarioCfg(t, 2600, boundedConfig())
	for i := 0; i < 2; i++ {
		if again := runGoldenScenarioCfg(t, 2600, boundedConfig()); again != first {
			t.Fatalf("the F5′ replay is not deterministic (run %d differs) — map order or a wall clock leaked into a decision", i+2)
		}
	}
	// ... and it is a DIFFERENT trace from the default arm's: the selector really selects
	// (a no-op "alternative" would measure the anchor twice).
	if first == runGoldenScenario(t) {
		t.Fatal("hrrn-bounded produced the newcomer-first trace: the module is not on the decision path")
	}
}

// ---- (5) the newcomer class: the archive's outermost key ----

// TestHrrnBoundedKeepsTheArchiveNewcomerClass pins a semantic fact that is easy to get
// wrong from the knob's name, and that the equivalence replay depends on: the archive's
// F5′ (dsai/groupsched-tier-aware @ dd5350b5, `entryKey.less`) compares its newcomer rank
// (isNew) BEFORE the response ratio, so the newcomer CLASS is the outer partition and
// the ratio orders heads INSIDE a class. Selecting this module therefore does not delete
// newcomer-first; it replaces age-ordering inside the classes with equal slowdown.
//
// The consequence is asserted in both directions, because it is a real scheduling
// statement and not an implementation detail: a fresh newcomer outranks a long-waiting
// RETURNING member (the class), while among newcomers the ratio wins over the wait
// (a 3K entry with a 10 s wait beats a 250K one that has waited longer — the newcomer
// burst cannot starve the small heads inside its own class).
func TestHrrnBoundedKeepsTheArchiveNewcomerClass(t *testing.T) {
	forEngines(t, boundedConfig(), func(t *testing.T, fx *engineSet) {
		fx.queue(
			hrrnSpec{"newcomer", 1 * time.Second, 200_000},
			hrrnSpec{"returning", 900 * time.Second, 3_000},
		)
		fx.s.mu.Lock()
		fx.s.waiters["returning"].newcomer = false
		fx.s.mu.Unlock()
		if h := fx.headOf(); h == nil || h.id != "newcomer" {
			t.Fatalf("head = %v, want the newcomer: the class is the OUTER key (the archive's isNew), so a returning member waits behind a newcomer however long its wait", h)
		}
		// inside the newcomer class the ratio decides, not the raw wait: a 3K head that has
		// waited 10 s (1/300) outranks a 250K one that has waited longer (15 s at the 6K
		// MEDIAN rate = 1/400), i.e. a newcomer burst cannot starve the small heads inside
		// its own class.
		fx.queue(
			hrrnSpec{"small", 10 * time.Second, 3_000},
			hrrnSpec{"med", 10 * time.Second, 6_000},
			hrrnSpec{"med2", 10 * time.Second, 6_000},
			hrrnSpec{"big", 15 * time.Second, 250_000},
		)
		if h := fx.headOf(); h == nil || h.id == "big" {
			t.Fatalf("head = %v, want a small-class newcomer: inside the class the response ratio beats the raw wait", h)
		}
	})
}

// ---- (6) EQUIVALENCE: the core's F5′ IS the archive's F5′ ----

// TestHrrnBoundedEqualsTheArchiveF5P is the module's GOLDEN: on the same 2,600 s replay,
// with ONE config value changed (entryOrder: newcomer-first -> hrrn-bounded) and NOTHING
// else, the core must reproduce the ARCHIVE build's trace byte for byte.
//
// How testdata/golden-f5p-archive.txt was produced (and why it is not circular):
//
//  1. the archive tree at `dsai/groupsched-tier-aware` @ dd5350b5 — the commit that
//     introduced F5′ there — in a scratch worktree;
//  2. the four DECLARED FIXES of README §2 applied to it (total victim tie-break, the
//     valve never evicts, class-3 unstarted/wanting last, the stale-view guard) — the
//     patch is in docs/gsch-core/golden-f5p-recipe.md;
//  3. VALIDATED before use: that patched archive regenerates testdata/
//     golden-f0p-bugfixes.txt BYTE FOR BYTE (so the patch is the same recipe the other
//     goldens came from, not a re-invention);
//  4. the archive's own golden driver (this file's scenario, verbatim) at f0p's rendered
//     config with entryOrder: "hrrn-bounded".
//
// The result: 6,089 trace lines, 1,664 dispatches, 2,600 simulated seconds, identical
// picks, seats, exits-with-reasons, seated-set transitions and totals. There is no
// normalisation of any kind — no divergence to document, which is also the evidence that
// the module carries the archive's newcomer-class outer key (dropping it moves decisions
// from the first minute).
func TestHrrnBoundedEqualsTheArchiveF5P(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "golden-f5p-archive.txt"))
	if err != nil {
		t.Fatalf("F5′ equivalence golden missing: %v", err)
	}
	got := runGoldenScenarioCfg(t, 2600, boundedConfig())
	if got == string(want) {
		return
	}
	gotL := strings.Split(got, "\n")
	wantL := strings.Split(string(want), "\n")
	diff := 0
	for i := 0; i < len(gotL) || i < len(wantL); i++ {
		var a, b string
		if i < len(gotL) {
			a = gotL[i]
		}
		if i < len(wantL) {
			b = wantL[i]
		}
		if a == b {
			continue
		}
		diff++
		if diff <= 5 {
			t.Errorf("divergence at line %d:\n  core:    %s\n  archive: %s", i+1, a, b)
		}
	}
	t.Fatalf("the core's F5′ differs from the archive build's in %d lines: the module must be the archive's rule, not an approximation of it", diff)
}
