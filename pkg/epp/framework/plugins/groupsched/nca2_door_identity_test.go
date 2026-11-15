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

// THE DOOR'S OWN DECISION STREAM IS PINNED (session-layer LOGIC Amendment 2, I14).
//
// Two closed loops drive the door (doorBackfill on) over three pods with mixed prompt sizes (heads that
// collect on one pod, backfills, returners, idle seat holders, an engine removal), followed by a fixed
// R-A1 scenario: a non-head outsider that backfill refuses everywhere while a pod other than the
// collection pod holds reclaimable seats and has no returner waiting (the state backfillReclaim changes).
// The observable decision stream (the served index of every pass, the Snapshot, the event stream, each
// pod's held blocks, the door's head and collection pod, the admission state) is digested and pinned to
// the value the same fixture produces at a2142ddc. With backfillReclaim absent the door must decide
// exactly as there.
//
// Part B is pinned over loops in which no new-session head holds blocks while a returner of its pod waits:
// H10b changes exactly that state whatever backfillReclaim says, so that state is not pinned.

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

// The door decision stream at a2142ddc, pinned.
const (
	doorIdentityDigestA = "9ca384611216e7376dea1d5191bc5189ba224160be6a826ae96f202cc2347f84"
	doorIdentityDigestB = "c768e7eb5e3718fd87236d06a11bbf622fa689c5499c84633acffb07e8f00579"
)

// doorIdentityCoverage is what the loop must have exercised for a digest to mean anything.
type doorIdentityCoverage struct {
	headFit, headCollected, backfills, returners, collectStarts int
	// rA1 counts backfill sweeps in which an outsider that BF1 refuses everywhere would fit a pod that is
	// not the collection pod and has no returner waiting by reclaiming its eligible seats.
	rA1 int
	// h10b counts decision boundaries at which a new-session head holds blocks on its collection pod while
	// a returner of that pod waits.
	h10b int
}

// returnersWaitingPure is whether any returner of the pod waits, read from the state without building the
// pass's lists (the probe must not move a decision).
func returnersWaitingPure(s *Scheduler, g *engineGroup) bool {
	return len(s.nca2WarmReturners(g)) > 0
}

// doorIdentityProbe counts the coverage through the test seam; it only reads.
func doorIdentityProbe(s *Scheduler, now *time.Time, cov *doorIdentityCoverage) {
	s.nca2.door.onStage = func(st doorStage) {
		d := s.nca2.door
		switch st.name {
		case doorStageStart, doorStageEnd:
			if d.sessionAware && d.collectPod != "" && s.doorIsOutsider(d.headID) && s.doorClassOf(d.headID) == 2 {
				if g := s.groups[d.collectPod]; g != nil && g.nca2 != nil && g.nca2.reservedBlocks > 0 && returnersWaitingPure(s, g) {
					cov.h10b++
				}
			}
		case doorStageBackfill:
			if d.collectPod == "" {
				return
			}
			ids := make([]string, 0, len(s.waiters))
			for id := range s.waiters {
				if id != d.headID && !s.nca2.admitted[id] {
					ids = append(ids, id)
				}
			}
			sort.Strings(ids)
			for _, id := range ids {
				w := s.waiters[id]
				if _, ok := backfillFitsAnywhere(s, w); ok {
					continue
				}
				for _, g := range s.activeGroups() {
					if g.id == d.collectPod || g.nca2 == nil || g.numBlocks <= 0 || returnersWaitingPure(s, g) {
						continue
					}
					if PromptTokens(w.chains) > s.nca2View(g).room(s.nca2.AdmitLineTokens) {
						continue
					}
					if s.ncaV2ReclaimableBlocks(g, *now) > 0 && s.ncaV2Fits(g, w, *now, map[fitKey]fitFacts{}) {
						cov.rA1++
						return
					}
				}
			}
		}
		if st.name == doorStageEnd {
			switch st.path {
			case doorPathHeadFit:
				cov.headFit++
			case doorPathHeadCollected:
				cov.headCollected++
			case doorPathBackfill:
				cov.backfills++
			case doorPathReturner:
				cov.returners++
			}
		}
	}
}

// canonicalEvents renders a pass's events. The push-outs of one admission's trim are exited in map order
// (v2's L4, unchanged), so each run of consecutive push-out EXITs is sorted; every other event keeps its
// place in the stream.
func canonicalEvents(events []Event) []string {
	out := make([]string, 0, len(events))
	run := 0
	for _, ev := range events {
		line := fmt.Sprintf("ev t=%s kind=%s id=%s engine=%s detail=%s",
			ev.T.UTC().Format(time.RFC3339Nano), ev.Kind, ev.ID, ev.Engine, ev.Detail)
		if ev.Kind == "EXIT" && strings.Contains(ev.Detail, "reason="+exitReasonPushout+" ") {
			out = append(out, line)
			run++
			sort.Strings(out[len(out)-run:])
			continue
		}
		run = 0
		out = append(out, line)
	}
	return out
}

// doorIdentityTrace drives one door loop and returns its canonical decision stream.
func doorIdentityTrace(t *testing.T, seed, tier int64, sessions bool, cov *doorIdentityCoverage) []string {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		on := true
		c.DoorBackfill = &on
		c.CPUTierTokens = i64(tier)
		c.Quantum = 2
		if sessions {
			header := "x-llm-d-session-id"
			c.SessionAware, c.SessionHeader = &on, &header
		}
	}, []string{"p0", "p1", "p2"})
	now := epoch
	doorIdentityProbe(s, &now, cov)
	var events []Event
	s.onEvent = func(ev Event) {
		events = append(events, ev)
		if ev.Kind == "COLLECT" && strings.Contains(ev.Detail, "event=start") {
			cov.collectStarts++
		}
	}
	rng := rand.New(rand.NewSource(seed))
	threads := make([]*identityThread, 0, 40)
	sessionOf := map[string]string{}
	for i := 0; i < 40; i++ {
		th := &identityThread{id: fmt.Sprintf("t%02d", i), blocks: 10 + rng.Intn(110), grow: rng.Intn(8),
			turns: 3 + rng.Intn(5), gap: rng.Intn(3)}
		if i%6 == 0 {
			th.blocks = 300 + rng.Intn(300)
		}
		threads = append(threads, th)
		if sessions {
			sessionOf[th.id] = fmt.Sprintf("S%d", i/4)
		}
	}
	seq := 0
	trace := make([]string, 0, 4096)
	homeOf := func(id string) string {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.nca2.homes[id]
	}
	for pass := 0; pass < 300; pass++ {
		now = now.Add(time.Second)
		if pass == 220 {
			s.RemoveEngine("p2", now) // E9 and S5 inside the pinned stream
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
				Chains: chain(th.id, th.blocks), Session: sessionOf[th.id]})
		}
		served := s.Pass(heads, now)
		snap, err := json.Marshal(s.Snapshot(now))
		if err != nil {
			t.Fatal(err)
		}
		trace = append(trace, fmt.Sprintf("pass %d t=%d heads=%d served=%d", pass, now.UnixNano(), len(heads), served))
		trace = append(trace, "snap "+string(snap))
		trace = append(trace, canonicalEvents(events)...)
		events = nil
		s.mu.Lock()
		trace = append(trace, fmt.Sprintf("door head=%s collect=%s", s.nca2.door.headID, s.nca2.door.collectPod))
		s.mu.Unlock()
		trace = append(trace, identityState(s, &identityCoverage{}))
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
			th.gap = rng.Intn(9) // past idleSeconds (5) the seat is reclaimable
			if th.id[len(th.id)-1]%4 == 1 {
				th.gap = 15 + rng.Intn(30) // a long pause: an idle seat holder with no turn queued
			}
		}
	}
	return trace
}

// doorRA1Trace is R-A1 made deterministic, inside the pinned stream: the head N collects on p0 (one
// running seat, one idle), p1's other seats are held by idle threads and no returner of p1 waits, and a
// small newcomer arrives every pass that p1's free budget cannot take as it is. Seat a2 on p0 ends its
// turn at pass 5, so the head is admitted inside the horizon.
func doorRA1Trace(t *testing.T, sessions bool, cov *doorIdentityCoverage) []string {
	t.Helper()
	f := newDoorFixture(t, func(c *Config) {
		if sessions {
			on, header := true, "x-llm-d-session-id"
			c.SessionAware, c.SessionHeader = &on, &header
		}
	}, "p0", "p1")
	doorIdentityProbe(f.s, &f.now, cov)
	var events []Event
	f.s.onEvent = func(ev Event) { events = append(events, ev) }
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.idleSeat("p1", "b1", 300)
	f.idleSeat("p1", "b2", 300)
	f.running("p1", "b3", 300, 320) // 50 blocks free on p1; p0 comes first in the pod order (the id)
	f.wait(10 * time.Second)
	heads := []Head{inSession(f.heavyAt("N", 900, 320, f.now.Add(-5*time.Second)), "sN")}
	var trace []string
	inflight := map[string]string{}
	for pass := 0; pass < 16; pass++ {
		if pass < 6 {
			heads = append(heads, inSession(f.heavyAt("s"+itoa(pass), 100, 320, f.now), "s"+itoa(pass)))
		}
		if pass == 5 {
			f.s.FirstChunk("rq-a2", f.now)
			f.s.Settle("rq-a2", "p0", f.now)
		}
		for id, req := range inflight {
			f.s.FirstChunk(req, f.now)
			f.s.Settle(req, f.homeOf(id), f.now)
			delete(inflight, id)
		}
		served := f.pass(heads...)
		snap, err := json.Marshal(f.s.Snapshot(f.now))
		if err != nil {
			t.Fatal(err)
		}
		trace = append(trace, fmt.Sprintf("ra1 pass %d served=%d", pass, served), "snap "+string(snap))
		trace = append(trace, canonicalEvents(events)...)
		events = nil
		head, pod := f.doorHead()
		trace = append(trace, fmt.Sprintf("door head=%s collect=%s", head, pod), identityState(f.s, &identityCoverage{}))
		if served >= 0 {
			h := heads[served]
			f.s.AtPod(h.ReqID, f.homeOf(h.Thread), f.now)
			inflight[h.Thread] = h.ReqID
			heads = append(heads[:served], heads[served+1:]...)
		}
	}
	return trace
}

// doorIdentityDigest is the digest of two closed loops (a tight and a roomier CPU tier) and the R-A1
// scenario, with their coverage.
func doorIdentityDigest(t *testing.T, sessions bool, tiers [2]int64) (string, doorIdentityCoverage, int) {
	t.Helper()
	var cov doorIdentityCoverage
	trace := doorIdentityTrace(t, 20261115, tiers[0], sessions, &cov)
	trace = append(trace, doorIdentityTrace(t, 7, tiers[1], sessions, &cov)...)
	trace = append(trace, doorRA1Trace(t, sessions, &cov)...)
	sum := sha256.Sum256([]byte(strings.Join(trace, "\n")))
	return hex.EncodeToString(sum[:]), cov, len(trace)
}

// TestI14_DoorPartAIsA2142ddc pins Part A's door decision stream (I14), over a loop that reaches R-A1.
func TestI14_DoorPartAIsA2142ddc(t *testing.T) {
	got, cov, lines := doorIdentityDigest(t, false, [2]int64{150_000, 300_000})
	t.Logf("coverage %+v, %d lines, digest %s", cov, lines, got)
	if cov.headFit == 0 || cov.headCollected == 0 || cov.backfills == 0 || cov.returners == 0 ||
		cov.collectStarts == 0 || cov.rA1 == 0 {
		t.Fatalf("fixture coverage: %+v (every path and R-A1 must have happened)", cov)
	}
	if got != doorIdentityDigestA {
		t.Fatalf("the door's Part A decision stream is not a2142ddc's: digest %s, want %s", got, doorIdentityDigestA)
	}
}

// TestI14_DoorPartBIsA2142ddc pins Part B's door decision stream (I14) over a loop in which H10b's state
// never arises.
func TestI14_DoorPartBIsA2142ddc(t *testing.T) {
	got, cov, lines := doorIdentityDigest(t, true, [2]int64{56_000, 72_000})
	t.Logf("coverage %+v, %d lines, digest %s", cov, lines, got)
	if cov.headCollected == 0 || cov.backfills == 0 || cov.returners == 0 || cov.collectStarts == 0 {
		t.Fatalf("fixture coverage: %+v", cov)
	}
	if cov.h10b != 0 {
		t.Fatalf("the Part B loop reaches H10b's state %d times: it cannot be pinned", cov.h10b)
	}
	if got != doorIdentityDigestB {
		t.Fatalf("the door's Part B decision stream is not a2142ddc's: digest %s, want %s", got, doorIdentityDigestB)
	}
}
