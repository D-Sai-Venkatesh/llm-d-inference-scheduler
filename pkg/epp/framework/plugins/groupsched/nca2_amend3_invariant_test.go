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

// LOGIC Amendment 3 under the property loops: the door's T-DR6 family and v2's closed loops run with
// valveCPUWarm, noQuantum and both, and every invariant of the base loops still holds, every returner is
// eventually seated and every session finishes.

import (
	"fmt"
	"math/rand"
	"testing"
)

// amend3Switches are the switch combinations the loops run with.
var amend3Switches = []struct {
	name            string
	warm, noQuantum bool
}{
	{"valveCPUWarm", true, false},
	{"noQuantum", false, true},
	{"both", true, true},
}

// amend3Tune sets the two switches.
func amend3Tune(warm, noQuantum bool) func(*Config) {
	return func(c *Config) {
		on := true
		if warm {
			c.ValveCPUWarm = &on
		}
		if noQuantum {
			c.NoQuantum = &on
		}
	}
}

// TestTDR6_Amend3SwitchesUnderInterleavings: T-DR6 with each switch and both: Part A and Part B, with and
// without backfillReclaim, over the same seeds and CPU tiers: I2, I3 (after every pass), I4, I5, I6, I7,
// I8, I11 (inside every decision), I12 (after every pass); every session finishes and no returner is left
// waiting; with noQuantum no seat leaves as quantum-complete.
func TestTDR6_Amend3SwitchesUnderInterleavings(t *testing.T) {
	for _, sw := range amend3Switches {
		collected, backfills, reclaims, yields, i5, decisions := 0, 0, 0, 0, 0, 0
		for _, seed := range []int64{1, 7, 99, 4242, 20261115} {
			for _, tier := range []int64{56_000, 72_000, 150_000} {
				for _, sessions := range []bool{false, true} {
					for _, reclaim := range []bool{false, true} {
						tune := func(c *Config) {
							amend3Tune(sw.warm, sw.noQuantum)(c)
							if sessions {
								on, header := true, "x-llm-d-session-id"
								c.SessionAware, c.SessionHeader = &on, &header
							}
							if reclaim {
								on := true
								c.BackfillReclaim = &on
							}
						}
						where := fmt.Sprintf("%s seed %d tier %d sessions %v reclaim %v", sw.name, seed, tier, sessions, reclaim)
						s, probe, took := doorLoopTraced(t, seed, tier, 3000, tune, sessions, tier >= 150_000, nil)
						if len(probe.violations) > 0 {
							t.Fatalf("%s: %v", where, probe.violations)
						}
						if took >= 3000 {
							t.Fatalf("%s: liveness: sessions still unfinished after %d passes", where, took)
						}
						s.mu.Lock()
						left, quantumExits := len(s.waiters), s.counters.exits["quantum-complete"]
						s.mu.Unlock()
						if left != 0 {
							t.Fatalf("%s: liveness: %d threads still waiting at the end", where, left)
						}
						if sw.noQuantum && quantumExits != 0 {
							t.Fatalf("%s: Q0: %d quantum-complete exits with noQuantum", where, quantumExits)
						}
						collected += probe.collected
						backfills += probe.backfills
						reclaims += probe.reclaims
						yields += probe.yields
						i5 += probe.i5
						decisions += probe.decisions
					}
				}
			}
		}
		t.Logf("%s: %d decisions: %d heads admitted after collecting, %d backfills (%d while a collection stood, I5), %d backfill reclaims, %d group-C yields",
			sw.name, decisions, collected, backfills, i5, reclaims, yields)
		if collected == 0 || backfills == 0 || reclaims == 0 || i5 == 0 {
			t.Fatalf("%s: fixture coverage: %d collections, %d backfills, %d reclaims, %d I5 checks",
				sw.name, collected, backfills, reclaims, i5)
		}
	}
}

// TestI15_EachSwitchMovesTheFixture: the I15 fixture is sensitive to both switches (its digest moves with
// each), and with noQuantum it books no quantum-complete exit although seats past their quantum occur.
func TestI15_EachSwitchMovesTheFixture(t *testing.T) {
	for _, sw := range amend3Switches {
		got, _, covs := amend3Digest(t, amend3Tune(sw.warm, sw.noQuantum))
		if got == amend3IdentityDigest {
			t.Fatalf("%s: the fixture's decisions did not move", sw.name)
		}
		for name, cov := range covs {
			if sw.noQuantum && (cov.quantumExits != 0 || cov.quantumOnly == 0) {
				t.Fatalf("%s %s: quantum-complete exits %d (want 0) over %d seats past their quantum", sw.name, name, cov.quantumExits, cov.quantumOnly)
			}
			if !sw.noQuantum && cov.quantumExits == 0 {
				t.Fatalf("%s %s: the quantum still rotates without noQuantum", sw.name, name)
			}
		}
	}
}

// newSimTuned is v2's closed loop (newSimSeeded) with the switches set.
func newSimTuned(t *testing.T, seed int64, threads, blocks int, tier int64, tune func(*Config)) *sim {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		c.CPUTierTokens = i64(tier)
		tune(c)
	}, []string{"p0", "p1", "p2"})
	m := &sim{s: s, rng: rand.New(rand.NewSource(seed)), now: epoch, blocks: blocks}
	s.nca2.onPushout = func(reason string, active bool) {
		m.pushouts = append(m.pushouts, pushoutEvent{reason: reason, active: active})
	}
	for i := 0; i < threads; i++ {
		m.threads = append(m.threads, &simThread{id: fmt.Sprintf("s%03d", i), session: fmt.Sprintf("sess-%03d", i),
			turns: 3 + i%3, gap: i % 4})
	}
	return m
}

// simFingerprint is what a v2 loop did: its exits by reason and every session's worst head wait.
func simFingerprint(m *sim) (string, int64) {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	waits := make([]int, 0, len(m.threads))
	for _, th := range m.threads {
		waits = append(waits, th.maxWait)
	}
	return fmt.Sprint(m.s.counters.exits, waits), m.s.counters.exits["quantum-complete"]
}

// TestV2Amend3SwitchesUnderInterleavings: v2's own loops (the door off) with each switch and both: the CPU
// pressure loop (I1-I4, I7 work conservation, the push-out class rules) and a GPU-bound loop on the real
// tier (quantum 2, prompts of 6400 tokens against a 3200-token cold cap), each checked after every pass,
// and every session finishes. The GPU-bound loop's decisions move with each switch, and with noQuantum it
// books no quantum-complete exit while its off-twin books some.
func TestV2Amend3SwitchesUnderInterleavings(t *testing.T) {
	gpuBound := func(c *Config) { c.Quantum = 2 }
	for _, seed := range []int64{1, 7, 20261114, 99, 4242} {
		off := newSimTuned(t, seed, 40, 100, defaultCPUTierTokens, gpuBound)
		off.run(t, 2000)
		offPrint, offQuantum := simFingerprint(off)
		if offQuantum == 0 {
			t.Fatalf("seed %d: fixture: the GPU-bound loop rotates by quantum without noQuantum", seed)
		}
		offWorst := 0
		for _, th := range off.threads {
			offWorst = max(offWorst, th.maxWait)
		}
		for _, sw := range amend3Switches {
			tune := amend3Tune(sw.warm, sw.noQuantum)
			m := newSimTuned(t, seed, 40, 16, lineC, tune)
			m.run(t, 400)
			checkPushoutMix(t, m, seed)
			g := newSimTuned(t, seed, 40, 100, defaultCPUTierTokens, func(c *Config) { gpuBound(c); tune(c) })
			g.run(t, 2000)
			print, quantumExits := simFingerprint(g)
			_, pressureQuantum := simFingerprint(m)
			if sw.noQuantum && quantumExits+pressureQuantum != 0 {
				t.Fatalf("%s seed %d: Q0: %d quantum-complete exits with noQuantum", sw.name, seed, quantumExits+pressureQuantum)
			}
			if print == offPrint {
				t.Fatalf("%s seed %d: the GPU-bound loop did not move with the switch", sw.name, seed)
			}
			worst := 0
			for _, th := range g.threads {
				worst = max(worst, th.maxWait)
			}
			t.Logf("%s seed %d: I7 checked on %d + %d passes; GPU-bound worst head wait %d passes (off-twin %d)",
				sw.name, seed, m.checked, g.checked, worst, offWorst)
		}
	}
}
