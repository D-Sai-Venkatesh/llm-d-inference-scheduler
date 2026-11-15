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

// THE DECISION STREAM BEFORE valveCPUWarm AND noQuantum IS PINNED (session-layer LOGIC Amendment 3, I15).
//
// Closed loops drive v2 (the door off), the door's Part A and the door's Part B, both with backfillReclaim,
// over three pods with mixed prompt sizes and quantum 2. The fixture reaches the two states the amendment
// changes: returners whose seats were released while they stayed admitted (their context is warm in the
// pod's CPU line) consulting the cold valve, which refuses them on their GPU-cold tokens while it would
// take them on what the CPU line does not hold; pushed-out outsiders whose entry is still in the line; and
// seats past their quantum that are not silent. The
// observable decision stream (the served index of every pass, the Snapshot, the event stream, the door's
// head and collection pod, each pod's held blocks, the admission state) is digested and pinned to the value
// the same fixture produces at e962778a. The file names no symbol e962778a lacks.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"
)

// amend3IdentityDigest is the decision stream at e962778a, pinned.
const amend3IdentityDigest = "ae1c1d2e2a9b1238cb76ed19b7882a6464208279111a1293d4d3b19156e7594c"

// The loop's three policies.
const (
	amend3V2 = iota
	amend3PartA
	amend3PartB
)

// amend3Coverage is what the loops must have exercised for the digest to mean anything.
type amend3Coverage struct {
	// warmExcused counts end-of-pass returners whose CPU-line entry on their home pod covers more of the
	// prompt than the GPU-resident prefix does; valveFlips the ones among them the valve refuses on the
	// GPU-cold tokens and would take on the tokens the CPU line does not hold. outsiderExcused and
	// outsiderFlips are the same for pushed-out outsiders whose entry is still in the line.
	warmExcused, valveFlips, outsiderExcused, outsiderFlips int
	// quantumExits counts seats released as quantum-complete; quantumOnly counts end-of-pass seats that
	// are eligible by their quantum alone (not busy, not silent).
	quantumExits, quantumOnly                         int
	returnerSeats, headCollected, backfills, reclaims int
}

// amend3Probe reads the end-of-pass state for the coverage; it only reads (the caller holds the lock).
func amend3Probe(s *Scheduler, now time.Time, cov *amend3Coverage) {
	for id, w := range s.waiters {
		g := s.groups[s.nca2.homes[id]]
		if g == nil || g.nca2 == nil || g.numBlocks <= 0 {
			continue
		}
		var entry int64
		found := false
		for _, e := range g.nca2.line {
			if e.thread == id {
				entry, found = e.tokens, true
				break
			}
		}
		if !found {
			continue
		}
		prompt := PromptTokens(w.chains)
		gpuCold := g.index.ColdTokens(w.chains)
		cpuCold := min(gpuCold, prompt-min(entry, prompt))
		if cpuCold >= gpuCold {
			continue
		}
		flip := !s.coldAllowsTokens(g, gpuCold) && s.coldAllowsTokens(g, cpuCold)
		switch {
		case s.nca2.admitted[id]:
			cov.warmExcused++
			if flip {
				cov.valveFlips++
			}
		default:
			cov.outsiderExcused++
			if flip {
				cov.outsiderFlips++
			}
		}
	}
	for _, g := range s.activeGroups() {
		for _, m := range g.threads {
			if g.index.Exiting(m.id) || m.busy() || s.silent(m, now) {
				continue
			}
			if m.busyPeriods >= s.cfg.Quantum {
				cov.quantumOnly++
			}
		}
	}
}

// amend3Trace drives one loop and returns its canonical decision stream, adding to the coverage. tune is
// applied last (nil for the pinned stream).
func amend3Trace(t *testing.T, seed, tier int64, policy int, tune func(*Config), cov *amend3Coverage) []string {
	t.Helper()
	sessions := policy == amend3PartB
	s := newNCA2Tuned(t, func(c *Config) {
		on := true
		c.CPUTierTokens = i64(tier)
		c.Quantum = 2
		if policy != amend3V2 {
			c.DoorBackfill, c.BackfillReclaim = &on, &on
		}
		if sessions {
			header := "x-llm-d-session-id"
			c.SessionAware, c.SessionHeader = &on, &header
		}
		if tune != nil {
			tune(c)
		}
	}, []string{"p0", "p1", "p2"})
	var events []Event
	// the event hook runs under the scheduler's lock
	s.onEvent = func(ev Event) {
		events = append(events, ev)
		if ev.Kind == "EXIT" && strings.Contains(ev.Detail, "reason=quantum-complete ") {
			cov.quantumExits++
		}
		if ev.Kind == "SEAT" {
			if s.nca2.admitted[ev.ID] {
				cov.returnerSeats++ // an admission's state is granted after its seat
			}
			switch {
			case strings.HasSuffix(ev.Detail, " path="+doorPathHeadCollected):
				cov.headCollected++
			case strings.HasSuffix(ev.Detail, " path="+doorPathBackfill):
				cov.backfills++
			case strings.HasSuffix(ev.Detail, " path="+doorPathBackfillReclaim):
				cov.reclaims++
			}
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
	now := epoch
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
		amend3Probe(s, now, cov)
		if s.nca2.door != nil {
			trace = append(trace, fmt.Sprintf("door head=%s collect=%s", s.nca2.door.headID, s.nca2.door.collectPod))
		}
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
			th.gap = rng.Intn(9) // past idleSeconds (5) the seat is reclaimable: the next turn returns
			if th.id[len(th.id)-1]%4 == 1 {
				th.gap = 15 + rng.Intn(30) // a long pause: an idle seat holder with no turn queued
			}
		}
	}
	return trace
}

// amend3Policies is the pinned fixture: each policy over a tight and a roomier CPU tier.
var amend3Policies = []struct {
	name   string
	policy int
	tiers  [2]int64
}{
	{"v2", amend3V2, [2]int64{72_000, 150_000}},
	{"partA", amend3PartA, [2]int64{72_000, 150_000}},
	{"partB", amend3PartB, [2]int64{72_000, 150_000}},
}

// amend3Digest is the digest of every policy's two loops, with the per-policy digests and coverage.
func amend3Digest(t *testing.T, tune func(*Config)) (string, map[string]string, map[string]amend3Coverage) {
	t.Helper()
	var all []string
	parts := map[string]string{}
	covs := map[string]amend3Coverage{}
	for _, p := range amend3Policies {
		var cov amend3Coverage
		trace := amend3Trace(t, 20261115, p.tiers[0], p.policy, tune, &cov)
		trace = append(trace, amend3Trace(t, 7, p.tiers[1], p.policy, tune, &cov)...)
		sum := sha256.Sum256([]byte(strings.Join(trace, "\n")))
		parts[p.name] = hex.EncodeToString(sum[:])
		covs[p.name] = cov
		all = append(all, trace...)
	}
	sum := sha256.Sum256([]byte(strings.Join(all, "\n")))
	return hex.EncodeToString(sum[:]), parts, covs
}

// TestI15_DecisionsAreE962778a pins v2's and the door's decision stream over a fixture that reaches the
// CPU-warm returner at the valve and the seat past its quantum (I15).
func TestI15_DecisionsAreE962778a(t *testing.T) {
	got, parts, covs := amend3Digest(t, nil)
	for _, p := range amend3Policies {
		cov := covs[p.name]
		t.Logf("%s: coverage %+v, digest %s", p.name, cov, parts[p.name])
		if cov.warmExcused == 0 || cov.valveFlips == 0 || cov.outsiderExcused == 0 || cov.quantumExits == 0 ||
			cov.quantumOnly == 0 || cov.returnerSeats == 0 {
			t.Fatalf("%s: fixture coverage %+v (CPU-warm returners at the valve and seats past their quantum must have happened)",
				p.name, cov)
		}
		if p.policy != amend3V2 && (cov.headCollected == 0 || cov.backfills == 0 || cov.reclaims == 0) {
			t.Fatalf("%s: door coverage %+v", p.name, cov)
		}
	}
	if got != amend3IdentityDigest {
		t.Fatalf("the decision stream is not e962778a's: digest %s, want %s", got, amend3IdentityDigest)
	}
}
