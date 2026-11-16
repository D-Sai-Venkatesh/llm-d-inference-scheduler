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

// THE DOOR'S INVARIANTS (session-layer §10.1), checked on the STATE of a randomized closed loop rather
// than argued from the code: after every pass I2 (one pod holds, the collection pod), I3 (it holds for
// the head) and I12 (backfill was exhaustive); inside every decision, through the test seam, I4 (backfill
// releases nothing), I6 (a head that fits now releases nothing), I7 (the head is the first-ranked
// outsider, or a sticky head no first-ranked higher outsider could take over) and I11 (a head change
// leaves nothing held but a collection a takeover kept). The fit questions are re-asked here from the
// state, with fresh exact costs: the test does not trust the decision's own answers.

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"
)

// exitTotal is every seat RELEASE booked so far (the caller holds the lock). An admission's own L4
// push-out (reason pushout: the provisional entry moved an inactive seat holder past the line, U1) is the
// line's physics that v2's admission has too, not a seat released for anybody's budget, so I4/I6 do not
// count it.
func exitTotal(s *Scheduler) int64 {
	var n int64
	for reason, v := range s.counters.exits {
		if reason != exitReasonPushout {
			n += v
		}
	}
	return n
}

// headFitsNowAnywhere is fit-now(head, P) for some pod, from the state (the caller holds the lock).
func headFitsNowAnywhere(s *Scheduler, head *waiter) bool {
	f := PromptTokens(head.chains)
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 || g.nca2 == nil || f > s.nca2View(g).room(s.nca2.AdmitLineTokens) {
			continue
		}
		if !s.coldAllowsTokens(g, s.valveCold(g, head)) {
			continue
		}
		cost := int64(g.index.Cost(head.chains))
		if cost <= s.freeBudget(g) || (len(g.threads) == 0 && cost > int64(g.budgetBlocks)) {
			return true
		}
	}
	return false
}

// podExits is every pod's seat RELEASES so far, push-outs excluded as in exitTotal (the caller holds the lock).
func podExits(s *Scheduler) map[string]int64 {
	out := map[string]int64{}
	for _, g := range s.activeGroups() {
		for reason, v := range g.counters.exits {
			if reason != exitReasonPushout {
				out[g.id] += v
			}
		}
	}
	return out
}

// returnerWaitingAsTheDoorSees is BF4(b)/H10/H10b's question: the pass's returner list once it is built,
// and otherwise the list the door would build from the state now — read without building it, so the
// probe moves no decision.
func returnerWaitingAsTheDoorSees(s *Scheduler, g *engineGroup) bool {
	if s.nca2.warmBuilt {
		return s.doorReturnerWaitingOn(g)
	}
	return len(s.nca2WarmReturners(g)) > 0
}

// reclaimFitsAnywhere is BF4's checks for a non-head outsider on some pod, from the state (nothing is
// released, the costs are fresh).
func reclaimFitsAnywhere(s *Scheduler, w *waiter, now time.Time) (string, bool) {
	f := PromptTokens(w.chains)
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 || g.nca2 == nil || g.id == s.nca2.door.collectPod || len(g.threads) == 0 ||
			f > s.nca2View(g).room(s.nca2.AdmitLineTokens) || returnerWaitingAsTheDoorSees(s, g) {
			continue
		}
		if s.ncaV2Fits(g, w, now, map[fitKey]fitFacts{}) {
			return g.id, true
		}
	}
	return "", false
}

// canBackfill is "can backfill" as H3 and I12 mean it: BF1, and with backfillReclaim also BF4.
func canBackfill(s *Scheduler, w *waiter, now time.Time) (string, bool) {
	if pod, ok := backfillFitsAnywhere(s, w); ok {
		return pod, true
	}
	if s.nca2.door.backfillReclaim {
		return reclaimFitsAnywhere(s, w, now)
	}
	return "", false
}

// backfillFitsAnywhere is BF1's three checks for a non-head outsider on some pod, from the state.
func backfillFitsAnywhere(s *Scheduler, w *waiter) (string, bool) {
	d := s.nca2.door
	f := PromptTokens(w.chains)
	var headNeed int64
	head := s.waiters[d.headID]
	for _, g := range s.activeGroups() {
		if g.numBlocks <= 0 || g.nca2 == nil || f > s.nca2View(g).room(s.nca2.AdmitLineTokens) {
			continue
		}
		free := s.freeBudget(g)
		held := min(g.nca2.reservedBlocks, free)
		if w.id != g.nca2.reservedHead {
			free -= held
		}
		if int64(g.index.Cost(w.chains)) > free {
			continue
		}
		cold := s.valveCold(g, w)
		if !s.coldAllowsTokens(g, cold) {
			continue
		}
		if head != nil && g.id == d.collectPod {
			capTokens := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
			headNeed = min(s.valveCold(g, head), capTokens)
			if capTokens > 0 && g.coldInFlight+min(cold, capTokens)+headNeed > capTokens {
				continue
			}
		}
		return g.id, true
	}
	return "", false
}

// headHeadroom is whether the head's valve headroom is available on its collection pod (the caller holds
// the lock).
func headHeadroom(s *Scheduler, g *engineGroup) bool {
	head := s.waiters[s.nca2.door.headID]
	capTokens := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
	return head == nil || capTokens <= 0 || g.coldInFlight+min(s.valveCold(g, head), capTokens) <= capTokens
}

// checkHolding is I2 and I3 (the caller holds the lock).
func checkHolding(s *Scheduler) string {
	d := s.nca2.door
	holding := 0
	for _, g := range s.activeGroups() {
		if g.nca2 == nil || g.nca2.reservedBlocks <= 0 {
			continue
		}
		holding++
		if g.id != d.collectPod {
			return fmt.Sprintf("I2: %s holds %d blocks and the collection pod is %q", g.id, g.nca2.reservedBlocks, d.collectPod)
		}
	}
	if holding > 1 {
		return fmt.Sprintf("I2: %d pods hold blocks", holding)
	}
	if d.collectPod != "" {
		if g := s.groups[d.collectPod]; g == nil || g.nca2.reservedHead != d.headID {
			return fmt.Sprintf("I3: the collection pod %s holds for %q, the head is %q", d.collectPod, g.nca2.reservedHead, d.headID)
		}
	}
	return ""
}

// doorProbe installs the per-decision checks (I4 as amended, I6, I7, I8 extended, I11) on the seam and
// collects violations.
type doorProbe struct {
	s          *Scheduler
	now        *time.Time
	violations []string
	lastHead   string
	fitAtStart bool
	exitsStart int64
	exitsBF    int64
	// podStart / podBF are each pod's releases at the decision's start and at BF's start; collectBF and
	// returnerBF are the collection pod and the pods with a returner waiting when BF began (I4 amended).
	podStart, podBF map[string]int64
	collectBF       string
	returnerBF      map[string]bool
	decisions       int
	takeovers       int
	collected       int
	backfills       int
	reclaims        int
	yields          int

	// heldBF, freeBF and headroomBF are the collection pod's held blocks, free budget and whether the head's
	// valve headroom was available there when BF began; i5 counts the backfill admissions I5 was checked on
	// (a collection pod existed).
	heldBF, freeBF int64
	headroomBF     bool
	i5             int
}

func newDoorProbe(s *Scheduler, now *time.Time) *doorProbe {
	p := &doorProbe{s: s, now: now}
	s.nca2.door.onStage = p.stage
	return p
}

func (p *doorProbe) fail(format string, args ...any) {
	if len(p.violations) < 10 {
		p.violations = append(p.violations, fmt.Sprintf(format, args...))
	}
}

// stage runs under the scheduler's lock (the seam is called from inside the decision).
func (p *doorProbe) stage(st doorStage) {
	s, d := p.s, p.s.nca2.door
	switch st.name {
	case doorStageStart:
		p.decisions++
		p.exitsStart = exitTotal(s)
		p.podStart = podExits(s)
		if msg := checkHolding(s); msg != "" {
			p.fail("at a decision start: %s", msg)
		}
		head := s.waiters[st.head]
		p.fitAtStart = head != nil && headFitsNowAnywhere(s, head)
		first := s.doorFirstOutsider()
		// I7: the first-ranked outsider, or the sticky head that the first-ranked could not take over
		if head != nil && first != nil && first.w.id != st.head {
			if st.head != p.lastHead {
				p.fail("I7: a new head %s that is not the first-ranked outsider %s", st.head, first.w.id)
			} else if first.class < s.doorClassOf(st.head) {
				if _, ok := canBackfill(s, first.w, *p.now); !ok {
					p.fail("I7: %s (class %d) cannot backfill and did not take the head from %s (class %d)",
						first.w.id, first.class, st.head, s.doorClassOf(st.head))
				}
			}
		}
		// I11: a head change leaves nothing held, except a collection a takeover kept on its own pod
		if st.head != p.lastHead {
			if st.kept {
				p.takeovers++
			}
			for _, g := range s.activeGroups() {
				if g.nca2 == nil || g.nca2.reservedBlocks == 0 {
					continue
				}
				if !st.kept || g.id != d.collectPod {
					p.fail("I11: after the head changed %q -> %q, %s still holds %d", p.lastHead, st.head, g.id, g.nca2.reservedBlocks)
				}
			}
		}
	case doorStageBackfill:
		p.exitsBF = exitTotal(s)
		p.podBF = podExits(s)
		p.collectBF = d.collectPod
		if g := s.groups[d.collectPod]; g != nil && g.nca2 != nil {
			p.heldBF, p.freeBF, p.headroomBF = g.nca2.reservedBlocks, s.freeBudget(g), headHeadroom(s, g)
		}
		p.returnerBF = map[string]bool{}
		for _, g := range s.activeGroups() {
			p.returnerBF[g.id] = g.nca2 != nil && returnerWaitingAsTheDoorSees(s, g)
		}
	case doorStageEnd:
		exits := exitTotal(s)
		pods := podExits(s)
		// I8: a new session's head releases no seat on its pod while a returner of that pod waits, and
		// (LOGIC Amendment 2, H10b) holds no blocks there while one waits
		if d.sessionAware && d.collectPod != "" && s.doorIsOutsider(d.headID) && s.doorClassOf(d.headID) == 2 {
			g := s.groups[d.collectPod]
			if g != nil && returnerWaitingAsTheDoorSees(s, g) {
				if st.path != doorPathReturner && pods[g.id] != p.podStart[g.id] {
					p.fail("I8: the new-session head %s released %d seats while a returner of %s waits", d.headID, pods[g.id]-p.podStart[g.id], g.id)
				}
				if g.nca2.reservedBlocks > 0 {
					p.fail("I8 (H10b): the new-session head %s holds %d on %s while a returner of it waits", d.headID, g.nca2.reservedBlocks, g.id)
				}
			}
		}
		// I6: a head that fit some pod now released nothing in its decision
		if p.fitAtStart && exits != p.exitsStart {
			p.fail("I6: the head fit a pod now and %d seats were released", exits-p.exitsStart)
		}
		// I4 (LOGIC Amendment 2): backfill releases seats only through BF4 — never on the collection pod,
		// never on a pod whose returner waits, and only for the admission of this decision
		if p.exitsBF >= 0 && exits != p.exitsBF {
			if !d.backfillReclaim || st.path != doorPathBackfillReclaim {
				p.fail("I4: backfill released %d seats (path %q)", exits-p.exitsBF, st.path)
			}
			for id, n := range pods {
				if n == p.podBF[id] {
					continue
				}
				if id == p.collectBF || p.returnerBF[id] {
					p.fail("I4: backfill released %d seats on %s (collection pod %q, returner waiting %v)",
						n-p.podBF[id], id, p.collectBF, p.returnerBF[id])
				}
			}
		}
		// I5: a backfill admission leaves the head's held blocks, its budget and its valve headroom on the
		// collection pod as they were
		if (st.path == doorPathBackfill || st.path == doorPathBackfillReclaim) && p.collectBF != "" && p.collectBF == d.collectPod {
			if g := s.groups[p.collectBF]; g != nil && g.nca2 != nil {
				p.i5++
				if g.nca2.reservedBlocks != p.heldBF {
					p.fail("I5: backfill moved the held blocks on %s (%d -> %d)", g.id, p.heldBF, g.nca2.reservedBlocks)
				}
				if p.heldBF <= p.freeBF && g.nca2.reservedBlocks > s.freeBudget(g) {
					p.fail("I5: backfill spent the head's blocks on %s (held %d, free %d)", g.id, g.nca2.reservedBlocks, s.freeBudget(g))
				}
				if p.headroomBF && !headHeadroom(s, g) {
					p.fail("I5: backfill took the head's valve headroom on %s", g.id)
				}
			}
		}
		switch st.path {
		case doorPathHeadCollected:
			p.collected++
		case doorPathBackfill:
			p.backfills++
		case doorPathBackfillReclaim:
			p.reclaims++
		}
		p.exitsBF = -1
		p.lastHead = d.headID
	}
}

// doorLoopTraced is the closed loop: mixed sizes (small threads and big heads that must collect), growing
// prompts, pauses past idleSeconds, one served head per pass. It records the decision stream (served
// index, SEAT/EXIT/COLLECT events, each pod's held blocks, the head) when out is not nil.
// With sessions, thread i belongs to session S(i/4) (a root and three sub-agents). heavy is a mix that makes
// the GPU bind: larger threads, and one in four pausing long on its seat (an idle seat holder).
func doorLoopTraced(t *testing.T, seed, tier int64, passes int, tune func(*Config), sessions, heavy bool,
	out *[]string) (*Scheduler, *doorProbe, int) {
	t.Helper()
	mix := doorMix{bigEvery: 6, heavy: heavy}
	if sessions {
		mix.sessionSize = 4
	}
	return doorLoopMixed(t, seed, tier, passes, tune, mix, out)
}

// doorMix is a closed loop's population: sessionSize threads per session (0: no session header), one
// thread in bigEvery big, and heavy the mix that makes the GPU bind.
type doorMix struct {
	sessionSize, bigEvery int
	heavy                 bool
}

// doorLoopMixed is doorLoopTraced's loop over any population.
func doorLoopMixed(t *testing.T, seed, tier int64, passes int, tune func(*Config), mix doorMix,
	out *[]string) (*Scheduler, *doorProbe, int) {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		on := true
		c.DoorBackfill = &on
		c.CPUTierTokens = i64(tier)
		c.Quantum = 2
		if tune != nil {
			tune(c)
		}
	}, []string{"p0", "p1", "p2"})
	now := epoch
	probe := newDoorProbe(s, &now)
	s.onEvent = func(ev Event) {
		if ev.Kind == "COLLECT" && strings.Contains(ev.Detail, "event="+collectReleasedReturner) {
			probe.yields++
		}
		if out != nil {
			*out = append(*out, fmt.Sprintf("ev %s %s %s %s", ev.Kind, ev.ID, ev.Engine, ev.Detail))
		}
	}
	rng := rand.New(rand.NewSource(seed))
	threads := make([]*identityThread, 0, 36)
	sessionOf := map[string]string{}
	for i := 0; i < 36; i++ {
		th := &identityThread{id: fmt.Sprintf("t%02d", i), blocks: 5 + rng.Intn(40), grow: rng.Intn(8),
			turns: 3 + rng.Intn(4), gap: rng.Intn(3)}
		if i%mix.bigEvery == 0 {
			th.blocks = 300 + rng.Intn(250)
		}
		if mix.heavy {
			th.blocks = 10 + rng.Intn(110)
			if i%mix.bigEvery == 0 {
				th.blocks = 300 + rng.Intn(300)
			}
		}
		threads = append(threads, th)
		if mix.sessionSize > 0 {
			sessionOf[th.id] = fmt.Sprintf("S%d", i/mix.sessionSize)
		}
	}
	seq := 0
	homeOf := func(id string) string {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.nca2.homes[id]
	}
	for pass := 0; pass < passes; pass++ {
		now = now.Add(time.Second)
		var heads []Head
		alive := 0
		for _, th := range threads {
			if th.turns > 0 {
				alive++
			}
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
			heads = append(heads, Head{Thread: th.id, ReqID: th.req, Arrival: th.arrival, Chains: chain(th.id, th.blocks),
				Session: sessionOf[th.id]})
		}
		if alive == 0 {
			return s, probe, pass
		}
		served := s.Pass(heads, now)
		s.mu.Lock()
		if out != nil {
			var line strings.Builder
			fmt.Fprintf(&line, "pass %d served=%d head=%s collect=%s", pass, served, s.nca2.door.headID, s.nca2.door.collectPod)
			for _, g := range s.activeGroups() {
				fmt.Fprintf(&line, " %s=%d", g.id, g.nca2.reservedBlocks)
			}
			*out = append(*out, line.String())
		}
		if msg := checkHolding(s); msg != "" {
			probe.fail("pass %d: %s", pass, msg)
		}
		// I12: the pass ended with no outsider that backfill could still have admitted
		if snap := s.snapshot(now).NCA2; !snap.DecisionCapHit {
			for id, w := range s.waiters {
				if s.nca2.admitted[id] || id == s.nca2.door.headID {
					continue
				}
				if pod, ok := canBackfill(s, w, now); ok {
					probe.fail("I12 at pass %d: %s fits %s by backfill and was left outside", pass, id, pod)
				}
			}
		}
		s.mu.Unlock()
		if served >= 0 {
			for _, th := range threads {
				if th.id == heads[served].Thread {
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
			th.gap = rng.Intn(9)
			if mix.heavy && th.id[len(th.id)-1]%4 == 1 {
				th.gap = 15 + rng.Intn(30)
			}
		}
	}
	return s, probe, passes
}

// TestTDR6_DoorInvariantsUnderInterleavings: T-DR6 — random interleavings over several seeds and three CPU
// tiers (the roomiest one makes the GPU bind, so new-session heads yield to returners), Part A and Part B
// (I8 too), each with backfillReclaim off and on (LOGIC Amendment 2): no invariant breaks, the loop's own
// coverage includes collections, backfills, backfill reclaims and group-C yields, and every session
// finishes (liveness).
func TestTDR6_DoorInvariantsUnderInterleavings(t *testing.T) {
	collected, backfills, decisions, takeovers, reclaims, yields := 0, 0, 0, 0, 0, 0
	for _, seed := range []int64{1, 7, 99, 4242, 20261115} {
		for _, tier := range []int64{56_000, 72_000, 150_000} {
			for _, sessions := range []bool{false, true} {
				for _, reclaim := range []bool{false, true} {
					tune := func(c *Config) {
						if sessions {
							on, header := true, "x-llm-d-session-id"
							c.SessionAware, c.SessionHeader = &on, &header
						}
						if reclaim {
							on := true
							c.BackfillReclaim = &on
						}
					}
					_, probe, took := doorLoopTraced(t, seed, tier, 3000, tune, sessions, tier >= 150_000, nil)
					if len(probe.violations) > 0 {
						t.Fatalf("seed %d tier %d sessions %v reclaim %v: %v", seed, tier, sessions, reclaim, probe.violations)
					}
					if took >= 3000 {
						t.Fatalf("seed %d tier %d sessions %v reclaim %v: liveness: sessions still unfinished after %d passes",
							seed, tier, sessions, reclaim, took)
					}
					if !reclaim && probe.reclaims > 0 {
						t.Fatalf("seed %d: %d backfill reclaims with the switch off", seed, probe.reclaims)
					}
					collected += probe.collected
					backfills += probe.backfills
					decisions += probe.decisions
					takeovers += probe.takeovers
					reclaims += probe.reclaims
					yields += probe.yields
				}
			}
		}
	}
	t.Logf("%d decisions: %d heads admitted after collecting, %d backfills, %d backfill reclaims, %d group-C yields, %d takeovers that kept the blocks",
		decisions, collected, backfills, reclaims, yields, takeovers)
	if collected == 0 || backfills == 0 || reclaims == 0 || yields == 0 {
		t.Fatalf("fixture coverage: %d head collections, %d backfills, %d backfill reclaims, %d group-C yields",
			collected, backfills, reclaims, yields)
	}
}

// TestE10_ACutPassIsContinuedByTheNext: E10 — a pass that stops after one decision (the cap's effect)
// with outsiders that still fit leaves them to the next pass, which admits them all.
func TestE10_ACutPassIsContinuedByTheNext(t *testing.T) {
	f := newDoorFixture(t, nil, "p0", "p1")
	var heads []Head
	for i := 0; i < 6; i++ {
		heads = append(heads, f.heavyAt("s"+itoa(i), 20, 320, f.now))
	}
	f.now = f.now.Add(time.Second)
	f.s.mu.Lock()
	f.s.syncQueue(heads, f.now)
	f.s.nca2.passMemo, f.s.nca2.cand = map[fitKey]fitFacts{}, map[string]nca2Cand{}
	f.s.nca2.warm, f.s.nca2.warmBuilt = map[string][]*waiter{}, false
	f.s.doorBeginPass()
	if !f.s.ncaDoorDecideOnce(f.now) {
		f.s.mu.Unlock()
		t.Fatal("fixture: the first decision admits")
	}
	waiting := len(f.s.waiters)
	f.s.mu.Unlock()
	if waiting != 5 {
		t.Fatalf("fixture: the cut pass leaves 5 outsiders, got %d", waiting)
	}
	f.pass(heads...)
	for i := 0; i < 6; i++ {
		if !f.admitted("s" + itoa(i)) {
			t.Fatalf("E10: s%d still outside after the next pass", i)
		}
	}
}
