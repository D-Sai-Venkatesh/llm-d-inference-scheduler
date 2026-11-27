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

// THE REPLAY: a seeded workload driven through the exported entry points only, written to a trace —
// every pass's result, every forensic event, and a per-step snapshot. The trace's md5 is pinned: it
// was recorded at a7b13cc4 before this branch changed anything, so with no CPU tier the policy must
// produce the SAME trace, decision for decision. The workload is meant to be hard on a diff: three
// engines, thread churn, a shared prompt head, growing contexts, think times, abandoned and lost
// requests, requests the framework places on another engine, anonymous traffic, and one engine that
// leaves and comes back.

import (
	"crypto/md5"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

type replayThread struct {
	id       string
	ctx      int  // own blocks of the current prompt
	shared   bool // the prompt starts with the shared 12-block head
	turns    int
	maxTurns int
	readyAt  int // the step its next turn arrives
	queued   bool
	arrived  time.Time
	inFlight bool
}

type replayReq struct {
	id, thread      string
	podAt, chunkAt  int
	endAt           int
	abandon, lost   bool
	moved, atPod    bool
	chunked, landed bool
	engine          string
}

// replayOpts are the knobs a test may turn; the zero value is the pinned workload.
type replayOpts struct {
	noMoves bool     // no request is placed on another engine than its seat
	atPod   func(reqID, engine string) // called right after each AtPod (the PreRequest moment)
	engines []string // the engines (default a, b, c — engine c leaves at half time and comes back)
}

func (th *replayThread) chains() []Chain {
	if th.shared {
		return chainWithPrefix("shared-head", 12, th.id, th.ctx)
	}
	return chain(th.id, th.ctx)
}

// driveReplay runs steps one-second steps of the workload on s. onStep, when set, runs after each step
// and must not write to out.
func driveReplay(t *testing.T, s *Scheduler, seed int64, steps int, out *strings.Builder, opts replayOpts, onStep func(step int)) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	engines := []string{"a", "b", "c"}
	if len(opts.engines) > 0 {
		engines = opts.engines
	}
	hasC := false
	for _, e := range engines {
		hasC = hasC || e == "c"
	}
	for _, e := range engines {
		if err := s.SetNumBlocksFor(e, 64, 500); err != nil {
			t.Fatal(err)
		}
	}
	s.onEvent = func(ev Event) {
		fmt.Fprintf(out, "E %s %s %s %s\n", ev.Kind, ev.ID, ev.Engine, ev.Detail)
	}
	threads := map[string]*replayThread{}
	born := 0
	newThread := func(step int) {
		id := fmt.Sprintf("th%03d", born)
		born++
		threads[id] = &replayThread{id: id, ctx: 8 + rng.Intn(60), shared: rng.Intn(3) == 0,
			maxTurns: 2 + rng.Intn(14), readyAt: step + rng.Intn(6)}
	}
	for i := 0; i < 30; i++ {
		newThread(0)
	}
	reqs := map[string]*replayReq{}
	var anon []Head
	anonN := 0
	sortedKeys := func(m map[string]*replayThread) []string {
		ks := make([]string, 0, len(m))
		for k := range m {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		return ks
	}
	for step := 0; step < steps; step++ {
		now := epoch.Add(time.Duration(step) * time.Second)
		switch {
		case !hasC:
		case step == steps/2:
			fmt.Fprintf(out, "GONE c %d\n", s.RemoveEngine("c", now))
		case step == steps/2+20:
			if err := s.SetNumBlocksFor("c", 64, 500); err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(out, "BACK c\n")
		}
		// turns arrive
		for _, id := range sortedKeys(threads) {
			th := threads[id]
			if !th.queued && !th.inFlight && step >= th.readyAt && th.turns < th.maxTurns {
				th.queued, th.arrived = true, now
				th.turns++
				th.ctx += 2 + rng.Intn(7)
			}
		}
		if rng.Intn(8) == 0 {
			anonN++
			anon = append(anon, Head{ReqID: fmt.Sprintf("anon%d", anonN), Arrival: now})
		}
		// the requests' lives
		rids := make([]string, 0, len(reqs))
		for id := range reqs {
			rids = append(rids, id)
		}
		sort.Strings(rids)
		for _, rid := range rids {
			r := reqs[rid]
			th := threads[r.thread]
			done := func(think int) {
				if th != nil {
					th.inFlight = false
					th.readyAt = step + think
				}
				delete(reqs, rid)
			}
			switch {
			case !r.atPod && step >= r.podAt:
				if r.abandon {
					s.Abandon(rid, now)
					fmt.Fprintf(out, "A %s\n", rid)
					done(1)
					continue
				}
				eng, ok := s.SeatEngineOfRequest(rid)
				if !ok {
					eng = engines[rng.Intn(len(engines))]
				}
				if r.moved {
					eng = engines[(sort.SearchStrings(engines, eng)+1)%len(engines)]
				}
				r.engine, r.atPod = eng, true
				s.AtPod(rid, eng, now)
				if opts.atPod != nil {
					opts.atPod(rid, eng)
				}
			case r.atPod && !r.chunked && step >= r.chunkAt:
				s.FirstChunk(rid, now)
				r.chunked = true
			case r.chunked && !r.landed && step >= r.endAt:
				r.landed = true
				if r.lost {
					// never settles: only the reap could close it, far beyond this horizon
					if th != nil {
						th.inFlight = false
						th.readyAt = step + rng.Intn(15)
					}
					continue
				}
				s.Settle(rid, r.engine, now)
				done(rng.Intn(25))
			}
		}
		// churn: a finished thread is replaced
		for _, id := range sortedKeys(threads) {
			th := threads[id]
			if th.turns >= th.maxTurns && !th.queued && !th.inFlight {
				delete(threads, id)
				newThread(step + 1)
			}
		}
		// the passes
		for k := 0; k < 10; k++ {
			var heads []Head
			for _, id := range sortedKeys(threads) {
				th := threads[id]
				if th.queued {
					heads = append(heads, Head{Thread: th.id, ReqID: fmt.Sprintf("%s.%d", th.id, th.turns),
						Arrival: th.arrived, Chains: th.chains(), Returner: th.turns > 1,
						FairnessID: th.id, WireID: "w." + th.id})
				}
			}
			nThread := len(heads)
			heads = append(heads, anon...)
			if len(heads) == 0 {
				break
			}
			i := s.Pass(heads, now)
			fmt.Fprintf(out, "P %d %d", step, i)
			if i < 0 {
				fmt.Fprintln(out)
				break
			}
			h := heads[i]
			fmt.Fprintf(out, " %s %s\n", h.Thread, h.ReqID)
			if i >= nThread { // anonymous: served unseated
				anon = append(anon[:i-nThread], anon[i-nThread+1:]...)
				s.Settle(h.ReqID, engines[rng.Intn(len(engines))], now)
				continue
			}
			th := threads[h.Thread]
			th.queued, th.inFlight = false, true
			reqs[h.ReqID] = &replayReq{id: h.ReqID, thread: th.id, podAt: step,
				chunkAt: step + 1 + rng.Intn(3), endAt: step + 3 + rng.Intn(12),
				abandon: rng.Intn(60) == 0, lost: rng.Intn(120) == 0,
				moved: rng.Intn(50) == 0 && !opts.noMoves}
		}
		if onStep != nil {
			onStep(step)
		}
		snap := s.Snapshot(now)
		fmt.Fprintf(out, "S %d %d %d %d %d %d %d %d %d", step, snap.Seats, snap.NEBlocks, snap.BudgetBlocks,
			snap.WaitersShared, snap.Served, snap.FCFSServed, snap.AffinityViolations, snap.EngineGone)
		ids := make([]string, 0, len(snap.Engines))
		for id := range snap.Engines {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			e := snap.Engines[id]
			fmt.Fprintf(out, " [%s %d %d %d %d %d %d %d %v]", id, e.Seats, e.NEBlocks, e.BudgetBlocks, e.ColdInFlight,
				e.ExitingThreads, e.ExitingBlocks, e.Served, e.Exits)
		}
		fmt.Fprintf(out, " %v %v\n", snap.Exits, snap.Anomalies)
	}
}

// replayDigest is the md5 of the six-seed trace, recorded at a7b13cc4.
const replayDigest = "8b05b1a6999ab273a2782e2df77a419c"

// TestNoTierReplaysTheBaseCommit: with no CPU tier the policy's trace is the base commit's, byte for
// byte. GSCH_REPLAY_OUT keeps the trace for a diff.
func TestNoTierReplaysTheBaseCommit(t *testing.T) {
	var all strings.Builder
	for _, seed := range []int64{11, 12, 13, 14, 15, 16} {
		s, err := NewScheduler(defaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		driveReplay(t, s, seed, 500, &out, replayOpts{}, nil)
		fmt.Fprintf(&all, "# seed %d\n%s", seed, out.String())
	}
	if p := os.Getenv("GSCH_REPLAY_OUT"); p != "" {
		if err := os.WriteFile(p, []byte(all.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := fmt.Sprintf("%x", md5.Sum([]byte(all.String()))); got != replayDigest {
		t.Fatalf("the no-tier trace moved: md5 %s, recorded %s", got, replayDigest)
	}
}
