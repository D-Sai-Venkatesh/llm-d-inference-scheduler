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

// THE DECISION STREAM BEFORE headKeepsTurn IS PINNED (session-layer LOGIC Amendment 4, I16).
//
// I15's fixture (v2, the door's Part A and Part B, all three with their policies' switches) and I14's door
// loops (Part A, and Part B over the tiers in which H10b's state never arises) run with valveCPUWarm and
// noQuantum on, and each digest is pinned to the value the same fixture produces at 5c61d6eb. With both
// options off the same fixtures are I15's and I14's own pins. Neither fixture reaches the state
// headKeepsTurn changes (a head of a session that is or was inside, collecting, while a strictly higher
// outsider that cannot backfill waits), so the door's Part B closed loop also runs over amend4Mix, a
// population that reaches it, and its decision stream is pinned the same way. The file names no symbol
// 5c61d6eb lacks.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"testing"
)

// The decision streams at 5c61d6eb with valveCPUWarm and noQuantum on, pinned.
const (
	amend4OptionsDigest      = "35b4e52a52654861d4d9dfb9ca160bebddf94ac0457ed893854bddcc6ed02f58"
	amend4DoorOptionsDigestA = "ccd3d57295fa5a21e54c6c0e53bd9a011987ac2a0387d39e50dce9258a92abf6"
	amend4DoorOptionsDigestB = "31752ada32dc2550e9497edadfea42506405c96aac95146a906cd0384f142a4d"
	amend4MixDigest          = "8e734108a3b5021552b1abaf814abe489f879b7fb69e7327d55aebf7e09e72a7"
)

// amend4Mix is a population whose Part B loop reaches headKeepsTurn's state: sessions of two threads and
// one thread in two big, so a session whose threads were all pushed out has a big head that collects
// while a big thread of an inside session cannot backfill.
var amend4Mix = doorMix{sessionSize: 2, bigEvery: 2, heavy: true}

// amend4MixTrace is the Part B closed loop over amend4Mix (seeds and tiers fixed), its recorded decision
// stream made canonical (each run of push-out EXITs of one trim is sorted, as canonicalEvents does), and the
// takeovers from a collecting head it saw.
func amend4MixTrace(t *testing.T, tune func(*Config)) ([]string, int) {
	t.Helper()
	var trace []string
	takeovers := 0
	for _, seed := range []int64{1, 7, 99, 4242, 20261115} {
		for _, tier := range []int64{56_000, 72_000} {
			var out []string
			_, probe, took := doorLoopMixed(t, seed, tier, 3000, func(c *Config) {
				on, header := true, "x-llm-d-session-id"
				c.SessionAware, c.SessionHeader = &on, &header
				if tune != nil {
					tune(c)
				}
			}, amend4Mix, &out)
			if len(probe.violations) > 0 || took >= 3000 {
				t.Fatalf("seed %d tier %d: %v, %d passes", seed, tier, probe.violations, took)
			}
			trace = append(trace, fmt.Sprintf("loop seed=%d tier=%d passes=%d", seed, tier, took))
			run := 0
			for _, line := range out {
				if strings.HasPrefix(line, "ev COLLECT ") && strings.Contains(line, "_takeover ") {
					takeovers++
				}
				trace = append(trace, line)
				if strings.HasPrefix(line, "ev EXIT ") && strings.Contains(line, "reason="+exitReasonPushout+" ") {
					run++
					sort.Strings(trace[len(trace)-run:])
					continue
				}
				run = 0
			}
		}
	}
	return trace, takeovers
}

// amend4MixStream is amend4MixTrace's digest under a tune.
func amend4MixStream(t *testing.T, tune func(*Config)) string {
	t.Helper()
	trace, takeovers := amend4MixTrace(t, tune)
	if takeovers == 0 {
		t.Fatal("fixture: the amend4Mix loop must take the head from a collecting head")
	}
	sum := sha256.Sum256([]byte(strings.Join(trace, "\n")))
	return hex.EncodeToString(sum[:])
}

// amend4I15 is I15's fixture under a tune.
func amend4I15(t *testing.T, tune func(*Config)) string {
	t.Helper()
	got, _, _ := amend3Digest(t, tune)
	return got
}

// amend4I14 is I14's door loops of one part under a tune.
func amend4I14(sessions bool, tiers [2]int64) func(*testing.T, func(*Config)) string {
	return func(t *testing.T, tune func(*Config)) string {
		t.Helper()
		got, _, _ := doorIdentityDigest(t, sessions, tiers, tune)
		return got
	}
}

// amend4Pins are the streams I16 holds to 5c61d6eb: with valveCPUWarm and noQuantum on (options), pinned
// here, and with both off, pinned by I15 and I14.
var amend4Pins = []struct {
	name    string
	digest  func(*testing.T, func(*Config)) string
	options bool
	want    string
}{
	{"I15 fixture", amend4I15, false, amend3IdentityDigest},
	{"I14 Part A", amend4I14(false, doorIdentityTiersA), false, doorIdentityDigestA},
	{"I14 Part B", amend4I14(true, doorIdentityTiersB), false, doorIdentityDigestB},
	{"I15 fixture, valveCPUWarm and noQuantum", amend4I15, true, amend4OptionsDigest},
	{"I14 Part A, valveCPUWarm and noQuantum", amend4I14(false, doorIdentityTiersA), true, amend4DoorOptionsDigestA},
	{"I14 Part B, valveCPUWarm and noQuantum", amend4I14(true, doorIdentityTiersB), true, amend4DoorOptionsDigestB},
	{"Part B over amend4Mix", amend4MixStream, false, amend4MixDigest},
}

// TestI16_DecisionsAre5c61d6eb holds every pinned stream to 5c61d6eb (I16): the streams with valveCPUWarm
// and noQuantum on, and I15's and I14's with them off.
func TestI16_DecisionsAre5c61d6eb(t *testing.T) {
	for _, p := range amend4Pins {
		tune := func(c *Config) {
			if p.options {
				amend3Tune(true, true)(c)
			}
		}
		if got := p.digest(t, tune); got != p.want {
			t.Errorf("%s: the decision stream is not 5c61d6eb's: digest %s, want %s", p.name, got, p.want)
		}
	}
}
