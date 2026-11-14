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

// OFF IS THE SHIPPED POLICY, BYTE FOR BYTE (nca2 spec I5).
//
// A fixture drives the pass through everything a decision touches — seating, growth, rotation by
// the quantum, the budget valve, an engine removal, a rebind, the cold valve, an anonymous
// pass-through, an abandon, a block-size mismatch — on a fixed clock, and the WHOLE observable
// decision stream is digested: the served index of every pass, the full Snapshot (seats, blocks,
// exits, anomalies, every per-engine number) and the SEAT/EXIT/REBIND event stream.
//
// The digest is pinned to the value the SAME fixture produces at fb57f538, the PR's base. It was
// measured by running this file there (the file names no nca2 symbol on purpose: it compiles
// against the base unchanged). Any commit that changes the shipped decision, a shipped counter, or
// the shape of the shipped Snapshot breaks this test — which is the point: the nca2 work is allowed
// to add a mode, never to move the one that ships.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// offIdentityDigest is the SHA-256 of the OFF decision stream at fb57f538 (the base), pinned.
const offIdentityDigest = "0034de971ad01ae1e6bbe1c0e34df198cdf601db839cc6505ea08f510bf55328"

// offTimeline is the fixture's own clock step: everything the policy measures is measured in the
// timestamps its passes are handed.
const offTimeline = 500 * time.Millisecond

// offTrace drives the fixture and returns the canonical decision stream, one line per fact.
func offTrace(t *testing.T) []string {
	t.Helper()
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	s.onEvent = func(ev Event) { events = append(events, ev) }
	trace := make([]string, 0, 256)
	now := epoch
	// a head that arrived `ago` before the pass, with its own blocks, and optionally a class
	head := func(id string, ago time.Duration, blocks int) Head {
		return Head{Thread: id, ReqID: "r-" + id, Arrival: now.Add(-ago),
			FairnessID: "f-" + id, WireID: "w-" + id, Chains: chain(id, blocks)}
	}
	pass := func(step int, heads []Head) int {
		now = now.Add(offTimeline)
		i := s.Pass(heads, now)
		snap, err := json.Marshal(s.Snapshot(now))
		if err != nil {
			t.Fatal(err)
		}
		trace = append(trace, fmt.Sprintf("pass %d t=%d served=%d", step, now.UnixNano(), i))
		trace = append(trace, "snap "+string(snap))
		for _, ev := range events {
			trace = append(trace, fmt.Sprintf("ev t=%s kind=%s id=%s engine=%s detail=%s",
				ev.T.UTC().Format(time.RFC3339Nano), ev.Kind, ev.ID, ev.Engine, ev.Detail))
		}
		events = nil
		return i
	}
	at := func(id, engine string) { s.AtPod("r-"+id, engine, now) }
	first := func(id string) { s.FirstChunk("r-"+id, now) }
	done := func(id, engine string) { s.Settle("r-"+id, engine, now) }

	// three measured engines, one 1000-block budget each (0.95 x 1000 = 950 blocks), and one whose
	// measured block size disagrees with the configured one (the mismatch is a recorded fact)
	for _, p := range []string{"p0", "p1", "p2"} {
		if err := s.SetNumBlocksFor(p, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.SetNumBlocksFor("p3", 32, 1000); err == nil {
		t.Fatal("a block-size disagreement must be recorded as a mismatch")
	}

	// 1. a newcomer is seated and served, then its presence is made real on the pod
	if i := pass(1, []Head{head("a", time.Second, 100)}); i != 0 {
		t.Fatalf("pass 1 must serve the only head, got %d", i)
	}
	at("a", "p0")
	at("a", "p0") // a REPEATED PreRequest: the shipped at-pod is a no-op, and the digest must say so
	first("a")
	done("a", "p0")

	// 2. two cold heads at once: the second meets the first's outstanding cold charge (the valve)
	pass(2, []Head{head("b", 2*time.Second, 200), head("c", 3*time.Second, 300)})
	at("b", "p1")
	first("b")
	done("b", "p1")

	// 3. a thread that returns (its class rides on the head) and a newcomer, both fitting once
	pass(3, []Head{{Thread: "a", ReqID: "r-a", Arrival: now.Add(-time.Second), Returner: true,
		FairnessID: "f-a", WireID: "w-a", Chains: chain("a", 100)}, head("d", 4*time.Second, 400)})
	at("a", "p0")
	at("a", "p0") // a REPEATED PreRequest: the shipped at-pod is a no-op, and the digest must say so
	first("a")
	done("a", "p0")

	// 4. growth: a's next turn is much longer, so the engine's used side outgrows its budget and the
	// valve has to take a victim
	pass(4, []Head{head("a", time.Second, 600)})
	at("a", "p0")
	pass(5, nil) // the growth lands and the valve runs on an empty queue

	// 5. fill the engines, then ask for one more seat than the budget holds
	pass(6, []Head{head("e", time.Second, 500), head("f", 2*time.Second, 500)})
	for _, id := range []string{"e", "f"} {
		at(id, "p1")
		first(id)
		done(id, "p1")
	}
	pass(7, []Head{head("g", time.Second, 700), head("h", 2*time.Second, 700), head("i", 3*time.Second, 700)})

	// 6. the quantum: a resident that completed Q busy periods is rotation-eligible when a head
	// needs its seat
	for turn := 0; turn < 5; turn++ {
		pass(10+turn, []Head{head("b", time.Second, 200)})
		at("b", "p1")
		first("b")
		done("b", "p1")
	}

	// 7. a rebind: the framework lands a seated thread on another engine (the observation wins)
	pass(20, []Head{head("b", time.Second, 200)})
	at("b", "p2")
	first("b")
	done("b", "p2")

	// 8. an anonymous head passes through unseated, and a head with no rendered tokens is an anomaly
	pass(21, []Head{{Thread: "", ReqID: "anon", Arrival: now.Add(-time.Second)}})
	pass(22, []Head{{Thread: "j", ReqID: "r-j", Arrival: now.Add(-time.Second)}})

	// 9. an abandon: a released request that leaves the path before reaching a pod
	pass(23, []Head{head("k", time.Second, 50)})
	s.Abandon("r-k", now)

	// 10. an engine disappears under its threads
	s.RemoveEngine("p2", now)
	pass(30, []Head{head("b", time.Second, 200), head("f", 2*time.Second, 500)})

	// 11. the end: one last snapshot with whatever is left (idle exits, anomalies, cleanups)
	now = now.Add(60 * time.Second)
	pass(31, nil)
	pass(32, nil)
	return trace
}

// TestOFFRepeatedAtPodIsTheBasesZeros: a request that already reached a pod and is handed to
// PreRequest AGAIN is the base's no-op: the shipped atPod returns the zero observables, so the span
// keeps whatever it had. OFF must do exactly that — the nca2 ride-back is gated behind the mode.
// (This file compiles against fb57f538 unchanged, so the assertion is the base's own behaviour.)
func TestOFFRepeatedAtPodIsTheBasesZeros(t *testing.T) {
	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("p0", 64, 1000); err != nil {
		t.Fatal(err)
	}
	now := epoch
	head := Head{Thread: "m", ReqID: "r-m", Arrival: now.Add(-time.Second), Chains: chain("m", 10)}
	if i := s.Pass([]Head{head}, now); i != 0 {
		t.Fatalf("the head must be seated and served, got %d", i)
	}
	first := s.AtPod("r-m", "p0", now)
	if first.promptTokens == 0 {
		t.Fatal("fixture: the first at-pod carries the turn's observables")
	}
	again := s.AtPod("r-m", "p0", now)
	if again.queueWait != 0 || again.predictedCached != 0 || again.promptTokens != 0 {
		t.Fatalf("OFF must return the base's zero observables on a repeated at-pod, got %+v", again)
	}
}

// TestOFFIdentityIsTheBase pins the shipped decision stream: OFF (no mode) must decide exactly what
// fb57f538 decides, with the same counters and the same Snapshot.
func TestOFFIdentityIsTheBase(t *testing.T) {
	trace := offTrace(t)
	sum := sha256.Sum256([]byte(strings.Join(trace, "\n")))
	got := hex.EncodeToString(sum[:])
	if got != offIdentityDigest {
		for i, line := range trace {
			if i == 40 {
				t.Logf("... %d more lines", len(trace)-40)
				break
			}
			t.Logf("%s", line)
		}
		t.Fatalf("the OFF decision stream is not the base's: digest %s, want %s", got, offIdentityDigest)
	}
}
