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

// nca2 IS v2, BYTE FOR BYTE, WHEN THE DOOR SWITCHES ARE ABSENT (session-layer contract C7/I1).
//
// A closed loop drives mode nca2 over three pods with mixed prompt sizes: small threads, big heads
// that have to collect seats across passes (A9), returners whose seats the idle rule released, threads
// pushed out of the CPU line (the PRIORITY class) and cold prefills that bind the valve. The whole
// observable decision stream is digested — the served index of every pass, the full Snapshot, the
// SEAT/EXIT/REBIND event stream, each pod's reservation, and the admission state — and the digest is
// pinned to the value the SAME fixture produces at ee8ea128 (gsch-nca-v2), the base of the door fix.
// The file names no symbol that base lacks, so it compiles there unchanged.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
	"time"
)

// nca2IdentityDigest is the SHA-256 of the nca2 decision stream at ee8ea128 (the base), pinned.
const nca2IdentityDigest = "4b90127540be881c709383bd261044acd8c9335e9ca06cd759ca2c467fef1d13"

// identityThread is one thread of the loop: a growing prompt, a number of turns and the gaps between
// them. A gap longer than idleSeconds lets the idle rule release the seat, so the next turn is a
// returner; a long-enough line of other threads pushes the entry out, so the next turn is PRIORITY.
type identityThread struct {
	id       string
	blocks   int
	grow     int
	turns    int
	gap      int
	req      string
	waiting  bool
	inflight bool
	settleIn int
	arrival  time.Time
}

// identityCoverage is what the fixture must have exercised for the digest to mean anything.
type identityCoverage struct {
	priorityWaiting, newcomerWaiting, returnerSeats, reservations, valveHeld, pushouts int
}

// nca2IdentityTrace drives one loop and returns its canonical decision stream, adding to the coverage.
func nca2IdentityTrace(t *testing.T, seed, tier int64, bigSpread int, cov *identityCoverage) []string {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		c.CPUTierTokens = i64(tier)
		c.Quantum = 2
	}, []string{"p0", "p1", "p2"})
	s.nca2.onPushout = func(string, bool) { cov.pushouts++ }
	var events []Event
	s.onEvent = func(ev Event) { events = append(events, ev) }
	rng := rand.New(rand.NewSource(seed))
	threads := make([]*identityThread, 0, 36)
	for i := 0; i < 36; i++ {
		th := &identityThread{id: fmt.Sprintf("t%02d", i), blocks: 5 + rng.Intn(40), grow: rng.Intn(8),
			turns: 3 + rng.Intn(4), gap: rng.Intn(3)}
		if i%6 == 0 {
			th.blocks = 300 + rng.Intn(bigSpread) // a big head: it has to collect seats across passes
		}
		threads = append(threads, th)
	}
	now := epoch
	seq := 0
	trace := make([]string, 0, 4096)
	homeOf := func(id string) string {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.nca2.homes[id]
	}
	for pass := 0; pass < 260; pass++ {
		now = now.Add(time.Second)
		if pass == 200 {
			// an engine disappears under its threads (S5): the line resets, the seats exit
			s.RemoveEngine("p2", now)
		}
		var heads []Head
		for _, th := range threads {
			if th.inflight || th.turns == 0 {
				continue
			}
			if !th.waiting {
				if th.gap > 0 {
					th.gap--
					continue
				}
				seq++
				th.req = fmt.Sprintf("%s-r%d", th.id, seq)
				th.arrival = now
				th.waiting = true
			}
			heads = append(heads, Head{Thread: th.id, ReqID: th.req, Arrival: th.arrival,
				FairnessID: "f-" + th.id, WireID: "w-" + th.req, ClientReqID: "c-" + th.req,
				Chains: chain(th.id, th.blocks)})
		}
		served := s.Pass(heads, now)
		snap, err := json.Marshal(s.Snapshot(now))
		if err != nil {
			t.Fatal(err)
		}
		trace = append(trace, fmt.Sprintf("pass %d t=%d heads=%d served=%d", pass, now.UnixNano(), len(heads), served))
		trace = append(trace, "snap "+string(snap))
		for _, ev := range events {
			trace = append(trace, fmt.Sprintf("ev t=%s kind=%s id=%s engine=%s detail=%s",
				ev.T.UTC().Format(time.RFC3339Nano), ev.Kind, ev.ID, ev.Engine, ev.Detail))
			if ev.Kind == "SEAT" {
				s.mu.Lock()
				if g := s.groupOf(ev.ID); g != nil && g.threads[ev.ID] != nil && g.threads[ev.ID].nca2Class == nca2ClassReturner {
					cov.returnerSeats++
				}
				s.mu.Unlock()
			}
		}
		events = nil
		trace = append(trace, identityState(s, cov))
		if served >= 0 {
			id := heads[served].Thread
			for _, th := range threads {
				if th.id == id {
					th.waiting, th.inflight = false, true
					th.settleIn = 1 + rng.Intn(3)
					s.AtPod(th.req, homeOf(th.id), now)
				}
			}
		}
		for _, th := range threads {
			if !th.inflight {
				continue
			}
			if th.settleIn > 0 {
				th.settleIn--
				continue
			}
			s.FirstChunk(th.req, now)
			s.Settle(th.req, homeOf(th.id), now)
			th.inflight, th.req = false, ""
			th.turns--
			th.blocks += th.grow
			th.gap = rng.Intn(9) // past idleSeconds (5) the seat is released: the next turn returns
		}
	}
	return trace
}

// identityState is the decision state a Snapshot does not carry, in a canonical form: each pod's
// A9 reservation, the admission state, the homes, and the outside lists' classes.
func identityState(s *Scheduler, cov *identityCoverage) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var b strings.Builder
	for _, id := range s.groupOrder {
		g := s.groups[id]
		if g.nca2 == nil {
			continue
		}
		fmt.Fprintf(&b, "res %s=%s/%d ", id, g.nca2.reservedHead, g.nca2.reservedBlocks)
		if g.nca2.reservedBlocks > 0 {
			cov.reservations++
		}
	}
	ids := make([]string, 0, len(s.nca2.homes))
	for id := range s.nca2.homes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		fmt.Fprintf(&b, "%s:%s:%t:%t ", id, s.nca2.homes[id], s.nca2.admitted[id], s.nca2.ever[id])
	}
	for id := range s.waiters {
		if s.nca2.admitted[id] {
			continue
		}
		if s.nca2.ever[id] {
			cov.priorityWaiting++
		} else {
			cov.newcomerWaiting++
		}
	}
	if head := s.nca2OutsideHead(); head != nil {
		fmt.Fprintf(&b, "head=%s", head.id)
		if _, held := s.coldValveHeldOn(head); held {
			cov.valveHeld++
		}
	}
	return "state " + b.String()
}

// TestNCA2IdentityIsTheBase pins the nca2 decision stream with both door switches absent: the door fix
// and the session layer must not move a single v2 decision, counter or snapshot field (C7/I1).
func TestNCA2IdentityIsTheBase(t *testing.T) {
	// two loops: a tight CPU tier (the line pushes threads out, so the PRIORITY class waits) and a
	// roomier one (big heads fit the room and collect seats, returners come back to their pods)
	var cov identityCoverage
	trace := nca2IdentityTrace(t, 20261115, 56_000, 200, &cov)
	trace = append(trace, nca2IdentityTrace(t, 20261115, 72_000, 250, &cov)...)
	// the fixture's own coverage first: a digest of a loop that never collected, never returned and
	// never held a head at the valve would pin nothing the door fix touches
	if cov.priorityWaiting == 0 || cov.newcomerWaiting == 0 || cov.returnerSeats == 0 ||
		cov.reservations == 0 || cov.valveHeld == 0 || cov.pushouts == 0 {
		t.Fatalf("fixture coverage: %+v (every class must have happened)", cov)
	}
	sum := sha256.Sum256([]byte(strings.Join(trace, "\n")))
	got := hex.EncodeToString(sum[:])
	t.Logf("coverage %+v, %d lines", cov, len(trace))
	if got != nca2IdentityDigest {
		t.Fatalf("the nca2 decision stream is not the base's: digest %s, want %s", got, nca2IdentityDigest)
	}
}
