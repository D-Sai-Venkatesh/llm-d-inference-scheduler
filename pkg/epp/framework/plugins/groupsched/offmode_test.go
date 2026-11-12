package groupsched

// OFF-MODE EQUIVALENCE: a seeded, deterministic workload driven through the public entry points, whose
// every decision (a Pass result), every forensic event and a compact per-step state digest is hashed.
// The digest below was produced at fb57f538 (the commit this branch was cut from) by the independent
// verifier's copy of this harness: the two-tier policy with groupBudgetSource=gpu, cpuTierTokens=0 and
// maxNumSeqs=0 must be decision-IDENTICAL to fb57f538, not merely "gated".
//
// The workload is deliberately hostile to a diff: 3 engines, thread churn, shared prefixes, think times,
// abandons, lost records, rebinds, anonymous traffic, responses with and without prompt-token details,
// and one engine flap (RemoveEngine then re-add).

import (
	"crypto/md5"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

type simThread struct {
	id       string
	turn     int
	blocks   int
	shared   bool
	nextAt   int // step at which the next head arrives (if not queued / in flight)
	queued   bool
	arrival  time.Time
	inflight int
	maxTurns int
}

type simReq struct {
	id, thread            string
	atPodAt, fcAt, endAt  int
	engine                string
	prompt, cached        int
	abandon, lost, rebind bool
	stage                 int
}

func simChains(th *simThread) []Chain {
	if th.shared {
		return chainWithPrefix("sys", 8, th.id, th.blocks)
	}
	return chain(th.id, th.blocks)
}

func runEquivSim(t *testing.T, s *Scheduler, seed int64, steps int, out *strings.Builder) {
	rng := rand.New(rand.NewSource(seed))
	engines := []string{"e0", "e1", "e2"}
	for _, e := range engines {
		if err := s.SetNumBlocksFor(e, 64, 400); err != nil {
			t.Fatal(err)
		}
	}
	s.onEvent = func(ev Event) {
		fmt.Fprintf(out, "EV %s %s %s %s %s\n", ev.T.Format("150405"), ev.Kind, ev.ID, ev.Engine, ev.Detail)
	}
	threads := map[string]*simThread{}
	nextThread := 0
	spawn := func(step int) {
		id := fmt.Sprintf("t%03d", nextThread)
		nextThread++
		threads[id] = &simThread{id: id, blocks: 10 + rng.Intn(50), shared: rng.Intn(3) == 0,
			nextAt: step + rng.Intn(5), maxTurns: 3 + rng.Intn(12)}
	}
	for i := 0; i < 36; i++ {
		spawn(0)
	}
	reqs := map[string]*simReq{}
	anonN := 0
	var anon []Head
	for step := 0; step < steps; step++ {
		now := epoch.Add(time.Duration(step) * time.Second)
		// engine flap
		if step == steps/3 {
			n := s.RemoveEngine("e2", now)
			fmt.Fprintf(out, "FLAP-DOWN e2 %d\n", n)
		}
		if step == steps/3+15 {
			if err := s.SetNumBlocksFor("e2", 64, 400); err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(out, "FLAP-UP e2\n")
		}
		ids := make([]string, 0, len(threads))
		for id := range threads {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		// arrivals
		for _, id := range ids {
			th := threads[id]
			if !th.queued && th.inflight == 0 && step >= th.nextAt && th.turn < th.maxTurns {
				th.queued = true
				th.arrival = now
				th.turn++
				th.blocks += 1 + rng.Intn(6)
			}
		}
		if rng.Intn(10) == 0 {
			anonN++
			anon = append(anon, Head{ReqID: fmt.Sprintf("anon-%d", anonN), Arrival: now})
		}
		// request lifecycle events due this step (sorted)
		rids := make([]string, 0, len(reqs))
		for id := range reqs {
			rids = append(rids, id)
		}
		sort.Strings(rids)
		for _, rid := range rids {
			r := reqs[rid]
			th := threads[r.thread]
			switch {
			case r.stage == 0 && step >= r.atPodAt:
				if r.abandon {
					s.Abandon(rid, now)
					fmt.Fprintf(out, "ABANDON %s\n", rid)
					if th != nil {
						th.inflight--
						th.nextAt = step + 1
					}
					delete(reqs, rid)
					continue
				}
				eng, ok := s.SeatEngineOfRequest(rid)
				if !ok {
					eng = engines[rng.Intn(len(engines))]
				}
				if r.rebind {
					eng = engines[(indexOf(engines, eng)+1)%len(engines)]
				}
				r.engine = eng
				s.AtPod(rid, eng, now)
				r.stage = 1
			case r.stage == 1 && step >= r.fcAt:
				s.FirstChunk(rid, now)
				r.stage = 2
			case r.stage == 2 && step >= r.endAt:
				if r.lost {
					// never settles: the reap is the belt (out of this horizon)
					r.stage = 3
					if th != nil {
						th.inflight--
						th.nextAt = step + rng.Intn(20)
					}
					continue
				}
				u := fwkrequest.Usage{PromptTokens: r.prompt,
					PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: r.cached}}
				if rng.Intn(25) == 0 {
					u.PromptTokenDetails = nil
				}
				s.Settle(rid, r.engine, u, now)
				if th != nil {
					th.inflight--
					th.nextAt = step + rng.Intn(30)
				}
				delete(reqs, rid)
			}
		}
		// thread churn: finished threads leave, new ones arrive
		for _, id := range ids {
			th := threads[id]
			if th.turn >= th.maxTurns && !th.queued && th.inflight == 0 {
				delete(threads, id)
				spawn(step + 1)
			}
		}
		// the passes
		for k := 0; k < 12; k++ {
			ids = ids[:0]
			for id := range threads {
				ids = append(ids, id)
			}
			sort.Strings(ids)
			var heads []Head
			for _, id := range ids {
				th := threads[id]
				if !th.queued {
					continue
				}
				heads = append(heads, Head{Thread: th.id, ReqID: fmt.Sprintf("%s-%d", th.id, th.turn),
					Arrival: th.arrival, Chains: simChains(th), Returner: th.turn > 1,
					FairnessID: th.id, WireID: fmt.Sprintf("w-%s-%d", th.id, th.turn)})
			}
			heads = append(heads, anon...)
			if len(heads) == 0 {
				break
			}
			i := s.Pass(heads, now)
			if i < 0 {
				fmt.Fprintf(out, "PASS %d -1\n", step)
				break
			}
			h := heads[i]
			fmt.Fprintf(out, "PASS %d %d %s %s\n", step, i, h.Thread, h.ReqID)
			if h.Thread == "" {
				// anonymous: served; settle it as untracked traffic on a random engine
				anon = append(anon[:i-(len(heads)-len(anon))], anon[i-(len(heads)-len(anon))+1:]...)
				s.Settle(h.ReqID, engines[rng.Intn(len(engines))],
					fwkrequest.Usage{PromptTokens: 300, PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 100}}, now)
				continue
			}
			th := threads[h.Thread]
			th.queued = false
			th.inflight++
			tok := th.blocks * 64
			if th.shared {
				tok += 8 * 64
			}
			reqs[h.ReqID] = &simReq{id: h.ReqID, thread: th.id, atPodAt: step, fcAt: step + 1 + rng.Intn(3),
				endAt: step + 3 + rng.Intn(10), prompt: tok, cached: rng.Intn(tok + 1),
				abandon: rng.Intn(50) == 0, lost: rng.Intn(100) == 0, rebind: rng.Intn(40) == 0}
		}
		// digest
		snap := s.Snapshot(now)
		fmt.Fprintf(out, "SNAP %d seats=%d ne=%d budget=%d waiters=%d served=%d fcfs=%d aff=%d gone=%d",
			step, snap.Seats, snap.NEBlocks, snap.BudgetBlocks, snap.WaitersShared, snap.Served, snap.FCFSServed,
			snap.AffinityViolations, snap.EngineGone)
		en := make([]string, 0, len(snap.Engines))
		for k := range snap.Engines {
			en = append(en, k)
		}
		sort.Strings(en)
		for _, k := range en {
			e := snap.Engines[k]
			fmt.Fprintf(out, " [%s s=%d ne=%d b=%d cold=%d ex=%d exb=%d srv=%d exits=%v]", k, e.Seats, e.NEBlocks,
				e.BudgetBlocks, e.ColdInFlight, e.ExitingThreads, e.ExitingBlocks, e.Served, e.Exits)
		}
		fmt.Fprintf(out, " exits=%v anomalies=%v\n", snap.Exits, snap.Anomalies)
	}
}

// indexOf is the sim's engine scan.
func indexOf(xs []string, x string) int {
	for i, y := range xs {
		if y == x {
			return i
		}
	}
	return 0
}

// equivDigest is the md5 of the 8-seed run. It is PINNED: any change to a decision, an event or a
// per-step snapshot in OFF mode changes it, which is exactly what this test is for.
const equivDigest = "36d963861fc863f67ed8975cc0165900"

// TestOffModeIsDecisionIdenticalToTheBaseCommit pins OFF mode to the base commit's behaviour: the
// digest was taken at fb57f538 and must not move. Set GSCH_EQUIV_OUT to keep the trace itself.
func TestOffModeIsDecisionIdenticalToTheBaseCommit(t *testing.T) {
	var all strings.Builder
	for _, seed := range []int64{1, 2, 3, 4, 5, 6, 7, 8} {
		s, err := NewScheduler(defaultConfig()) // the two-tier gate is OFF by default
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		runEquivSim(t, s, seed, 600, &out)
		fmt.Fprintf(&all, "=== seed %d\n%s", seed, out.String())
	}
	if path := os.Getenv("GSCH_EQUIV_OUT"); path != "" {
		if err := os.WriteFile(path, []byte(all.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	sum := fmt.Sprintf("%x", md5.Sum([]byte(all.String())))
	if sum != equivDigest {
		t.Fatalf("OFF mode is not decision-identical to fb57f538: digest %s, want %s (set GSCH_EQUIV_OUT and diff against the base commit's trace)", sum, equivDigest)
	}
}
