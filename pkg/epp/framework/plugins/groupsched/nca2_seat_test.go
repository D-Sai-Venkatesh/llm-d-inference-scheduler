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

// THE DECISION (A0–A7), one rule per test: the outside head first and only the head (A2.1/A3), the
// warm returner nearest the tail after it (A2.2), the pod order by in-flight then room (A5) with the
// pushed-out thread's old-pod preference (A4), the base fit with the RECLAIMABLE seats evicted for
// it (A2/S1) and the cold valve never evicting for (C9), and the room consumed at the decision (L0).

import (
	"sort"
	"strings"
	"testing"
	"time"
)

// fixture is one nca2 scheduler and its own clock: the policy measures what its passes are handed.
type fixture struct {
	t   *testing.T
	s   *Scheduler
	now time.Time
}

// newFixture is a scheduler in nca2 with the fixture tier: a 3040-token admission line and one
// 950-block budget per pod.
func newFixture(t *testing.T, pods ...string) *fixture {
	t.Helper()
	return &fixture{t: t, s: newNCA2(t, pods), now: epoch}
}

// hd is a head that arrived at this instant, with `blocks` blocks of its own content.
func (f *fixture) hd(id string, blocks int) Head {
	return Head{Thread: id, ReqID: "r-" + id, Arrival: f.now, FairnessID: "f-" + id, Chains: chain(id, blocks)}
}

// hdHeavy is a head with a large GPU footprint (blocks) and a small CPU footprint (tokens): the two
// spaces are different questions, and a test needs to move one without the other.
func (f *fixture) hdHeavy(id string, blocks, tokens int) Head {
	return Head{Thread: id, ReqID: "r-" + id, Arrival: f.now, FairnessID: "f-" + id,
		Chains: []Chain{{Keys: keys(blocks, seedFor(id)), Tokens: tokens}}}
}

// running seats the thread on pod and puts a REAL request of it in flight, with its CPU entry: the
// state a decision sees for a thread that is holding a seat while it runs. It is built from the
// scheduler's own parts (seat + release + the at-pod hook), so the derived counts the pass uses are
// the ones a run would have.
func (f *fixture) running(pod, id string, blocks int, tokens int64) {
	f.t.Helper()
	f.s.mu.Lock()
	g := f.s.groups[pod]
	w := &waiter{id: id, since: f.now, chains: []Chain{{Keys: keys(blocks, seedFor(id)), Tokens: int(tokens)}}}
	if !f.s.seat(g, w, 0, f.now) {
		f.s.mu.Unlock()
		f.t.Fatalf("fixture: %s must be seated on %s", id, pod)
	}
	m := g.threads[id]
	f.s.nca2.homes[id], f.s.nca2.admitted[id], f.s.nca2.ever[id] = pod, true, true
	f.s.nca2LineSet(g, id, tokens, f.now, pushoutRoom)
	req := "rq-" + id
	f.s.release(m, req, w.chains, f.now, dispatchObservables{promptTokens: tokens})
	f.s.mu.Unlock()
	f.s.AtPod(req, pod, f.now)
}

// step runs one pass, a second after the last one.
func (f *fixture) step(heads ...Head) int {
	f.t.Helper()
	f.now = f.now.Add(time.Second)
	return f.s.Pass(heads, f.now)
}

// atPod and done are the request's life, for the threads a test keeps running.
func (f *fixture) atPod(id, pod string) { f.s.AtPod("r-"+id, pod, f.now) }
func (f *fixture) done(id, pod string)  { f.s.Settle("r-"+id, pod, f.now) }

// turn runs one whole turn of a seated thread: dispatch, the pod, prefill complete, settle.
func (f *fixture) turn(id string, blocks int) {
	f.t.Helper()
	if i := f.step(f.hd(id, blocks)); i < 0 {
		f.t.Fatalf("the fixture expected %s to be served", id)
	}
	f.atPod(id, f.homeOf(id))
	f.s.FirstChunk("r-"+id, f.now)
	f.done(id, f.homeOf(id))
}

// wait advances the fixture clock.
func (f *fixture) wait(d time.Duration) { f.now = f.now.Add(d) }

// homeOf, admittedOn, classOf, entriesOfPod read the state under the lock.
func (f *fixture) homeOf(id string) string {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.nca2.homes[id]
}

func (f *fixture) admitted(id string) bool {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.nca2.admitted[id]
}

func (f *fixture) classOf(id string) string {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	if g := f.s.groupOf(id); g != nil {
		if m := g.threads[id]; m != nil {
			return m.nca2Class
		}
	}
	return ""
}

func (f *fixture) roomOn(pod string) int64 {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.nca2View(f.s.groups[pod]).room(f.s.nca2.AdmitLineTokens)
}

// TestAdmissionIsTheHeadAndSeatedAtOnce: A2.1 — the outside head is admitted on the pod the A5 order
// names, gets its home, L0's provisional entry at the head, and a seat in the same decision.
func TestAdmissionIsTheHeadAndSeatedAtOnce(t *testing.T) {
	f := newFixture(t, "p0")
	if i := f.step(f.hd("a", 10)); i != 0 {
		t.Fatalf("the admitted head must be served in the same pass, got %d", i)
	}
	if !f.admitted("a") || f.homeOf("a") != "p0" {
		t.Fatal("A2.1 admits on p0: home set and the admission granted")
	}
	if !seated(f.s, "a") {
		t.Fatal("the admission SEATS it at once")
	}
	if got := f.classOf("a"); got != nca2ClassAdmittedNow {
		t.Fatalf("the seat's class is admitted_now (it came from outside), got %q", got)
	}
	f.s.mu.Lock()
	pos, has := f.s.nca2PositionOn(f.s.groups["p0"], "a")
	f.s.mu.Unlock()
	if !has || pos != 640 {
		t.Fatalf("L0 puts the entry at the head with f = 640 tokens, got %d (present=%v)", pos, has)
	}
	// the room is consumed AT THE DECISION: 3040 - 640, and the admitted position is inside the line
	if got := f.roomOn("p0"); got != lineAdmit-640 {
		t.Fatalf("the room is L_a - the admitted position: %d, want %d", got, lineAdmit-640)
	}
	f.atPod("a", "p0")
	f.done("a", "p0")
	// a is INACTIVE now, so it is not the LAA and its 640 tokens are space: room is the whole line
	if got := f.roomOn("p0"); got != lineAdmit {
		t.Fatalf("an inactive admitted entry is space (room = L_a), got %d", got)
	}
}

// TestStrictHeadOfLineBlocksTheSmallerNewcomer: A3 — A2.1 considers ONLY the outside head, so a head
// too big for the room leaves the capacity unused rather than letting a smaller newcomer past it.
func TestStrictHeadOfLineBlocksTheSmallerNewcomer(t *testing.T) {
	f := newFixture(t, "p0")
	if i := f.step(f.hd("big", 48), f.hd("small", 5)); i != -1 {
		t.Fatalf("nothing may be served: the head does not fit the room, got %d", i)
	}
	if f.admitted("small") || seated(f.s, "small") {
		t.Fatal("no skip-ahead: the smaller newcomer behind the head must wait (A3)")
	}
	if f.admitted("big") {
		t.Fatal("a head larger than the room is never admitted")
	}
}

// TestWarmReturnerNearestTheTailIsSeated: A2.2 — the head does not fit the room, so the pod's own
// admitted waiting thread takes the seat, and it is the one nearest the tail (lowest position).
func TestWarmReturnerNearestTheTailIsSeated(t *testing.T) {
	f := newFixture(t, "p0")
	// two admitted threads, settled and then evicted by the (shipped) rotation: S4, they keep their
	// admission and their entries
	f.turn("warm", 10) // position 640 at the head
	f.turn("cool", 5)  // 320 at the head, warm at 960
	f.s.mu.Lock()
	for _, id := range []string{"warm", "cool"} {
		g := f.s.groups["p0"]
		f.s.exit(g, g.threads[id], "idle", f.now)
	}
	f.s.mu.Unlock()
	if !f.admitted("warm") || !f.admitted("cool") {
		t.Fatal("S4: an evicted seat holder keeps its admission")
	}
	// A3 puts a PRIORITY thread strictly before the newcomers, and every warm returner is priority:
	// so the head is an older priority thread of this pod whose own prompt cannot fit the room.
	f.s.mu.Lock()
	f.s.nca2.homes["big"] = "p0"
	f.s.nca2.ever["big"] = true
	f.s.mu.Unlock()
	// big (3072 tokens) cannot fit the room (L_a - 960 = 2080); warm (960) is nearer the tail than
	// cool (320), so warm takes the first freed seat
	head := Head{Thread: "big", ReqID: "r-big", Arrival: f.now.Add(-10 * time.Second), Chains: chain("big", 48)}
	wh := f.hd("warm", 10)
	wc := f.hd("cool", 5)
	var seats []string
	f.s.onEvent = func(ev Event) {
		if ev.Kind == "SEAT" {
			seats = append(seats, ev.ID)
		}
	}
	f.now = f.now.Add(time.Second)
	if i := f.s.Pass([]Head{head, wh, wc}, f.now); i < 0 {
		t.Fatal("A2.2 must seat a warm returner")
	}
	if len(seats) != 2 || seats[0] != "warm" || seats[1] != "cool" {
		t.Fatalf("the returner NEAREST THE TAIL (the deepest position) goes first: seat order %v", seats)
	}
	if !seated(f.s, "warm") || !seated(f.s, "cool") {
		t.Fatal("every fitting returner takes a seat (A1 repeats A2 until it seats nobody)")
	}
	f.s.mu.Lock()
	pos, _ := f.s.nca2PositionOn(f.s.groups["p0"], "warm")
	f.s.mu.Unlock()
	if pos != 960 {
		t.Fatalf("a returner seat does not touch the line (its entry moves at L1): position %d", pos)
	}
	if f.classOf("warm") != nca2ClassReturner {
		t.Fatalf("the seat's class is returner, got %q", f.classOf("warm"))
	}
	f.s.onEvent = nil
}

// TestReturnersOnlyIsA2_2sPathAndTheyAreNeverOutsiders: §2/A2.2 — a currently ADMITTED thread that
// waits without a seat is a warm returner, never an outside thread: with no outsider waiting, A2.2
// simply seats them all, the deepest position first.
func TestReturnersOnlyIsA2_2sPathAndTheyAreNeverOutsiders(t *testing.T) {
	f := newFixture(t, "p0")
	f.turn("warm", 10) // 640 at the head
	f.turn("cool", 5)  // 320 at the head, warm at 960
	f.s.mu.Lock()
	for _, id := range []string{"warm", "cool"} {
		g := f.s.groups["p0"]
		f.s.exit(g, g.threads[id], "idle", f.now) // unseated, the admission kept (S4)
	}
	head := f.s.nca2OutsideHead() // the CURRENT admission state decides the class
	f.s.mu.Unlock()
	if head != nil {
		t.Fatalf("an admitted waiting thread is not the outside head: %v", head.id)
	}
	var seats []string
	f.s.onEvent = func(ev Event) {
		if ev.Kind == "SEAT" {
			seats = append(seats, ev.ID)
		}
	}
	if i := f.step(f.hd("warm", 10), f.hd("cool", 5)); i < 0 {
		t.Fatal("A2.2 must seat the returners")
	}
	if len(seats) != 2 || seats[0] != "warm" || seats[1] != "cool" {
		t.Fatalf("deepest position first: %v", seats)
	}
	f.s.onEvent = nil
}

// TestOutsiderFirstThenTheReturner: A2's order — a fitting OUTSIDE head is admitted first (A2.1), and
// the pod's waiting warm returner takes the next decision (A2.2), in the same pass.
func TestOutsiderFirstThenTheReturner(t *testing.T) {
	f := newFixture(t, "p0")
	f.turn("warm", 5) // admitted, entry at 320, then evicted: a warm returner when its head arrives
	f.s.mu.Lock()
	g := f.s.groups["p0"]
	f.s.exit(g, g.threads["warm"], "idle", f.now)
	s := f.s
	s.mu.Unlock()
	var seats []string
	f.s.onEvent = func(ev Event) {
		if ev.Kind == "SEAT" {
			seats = append(seats, ev.ID+"@"+ev.Detail)
		}
	}
	// an outsider (5 blocks = 320 tokens) and the returner, both waiting
	if i := f.step(f.hd("new", 5), f.hd("warm", 5)); i < 0 {
		t.Fatal("a decision must be taken")
	}
	if len(seats) != 2 {
		t.Fatalf("two seats: the outsider (A2.1) and the returner (A2.2), got %v", seats)
	}
	if !strings.HasPrefix(seats[0], "new@") || !strings.HasPrefix(seats[1], "warm@") {
		t.Fatalf("A2.1 admits the outsider first, then A2.2 seats the returner: %v", seats)
	}
	if !f.admitted("new") || f.classOf("new") != nca2ClassAdmittedNow {
		t.Fatal("the outsider is admitted and seated (admitted_now)")
	}
	if f.classOf("warm") != nca2ClassReturner {
		t.Fatalf("the returner is seated as a returner, got %q", f.classOf("warm"))
	}
	f.s.onEvent = nil
}

// TestHeadIsTriedOverTheFleetBeforeAnyReturnerSeat: R2 — the outside head is tried on EVERY pod
// before A2.2 seats anybody: here it cannot fit its own (A4-preferred) pod but fits another, and the
// preferred pod's warm returner gets the NEXT repetition.
func TestHeadIsTriedOverTheFleetBeforeAnyReturnerSeat(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	// warm lives on p0, admitted, and its head is waiting (a warm returner)
	f.s.mu.Lock()
	f.s.nca2.homes["warm"] = "p0"
	f.s.nca2.admitted["warm"] = true
	f.s.nca2.ever["warm"] = true
	f.s.groups["p0"].nca2.line = []nca2Entry{{thread: "warm", tokens: 640}}
	// big was pushed out of p0 and still has an entry there: A4 prefers p0 for it
	f.s.nca2.homes["big"] = "p0"
	f.s.nca2.ever["big"] = true
	f.s.groups["p0"].nca2.line = append(f.s.groups["p0"].nca2.line, nca2Entry{thread: "big", tokens: 320})
	f.s.mu.Unlock()
	head := Head{Thread: "big", ReqID: "r-big", Arrival: f.now.Add(-10 * time.Second), Chains: chain("big", 40)}
	waitingHead := f.hd("warm", 10)
	f.now = f.now.Add(time.Second)
	if i := f.s.Pass([]Head{head, waitingHead}, f.now); i < 0 {
		t.Fatal("the pass must decide")
	}
	// p0's room is 2400 (warm's 640 occupies the line): 40 blocks = 2560 tokens cannot fit it
	if got := f.homeOf("big"); got != "p1" {
		t.Fatalf("the head must be admitted on the pod that can take it, got %q", got)
	}
	if !f.admitted("big") {
		t.Fatal("the head is admitted (A2.1 on p1)")
	}
	if !seated(f.s, "warm") || f.homeOf("warm") != "p0" {
		t.Fatal("the preferred pod's warm returner takes the next repetition (A2.2, on its home)")
	}
}

// TestPodOrderIsInFlightThenRoom: A5 — the pod whose GPU is the least busy goes first, then the pod
// with the most free CPU, then the id.
func TestPodOrderIsInFlightThenRoom(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	// p0 takes an admission and KEEPS a request at the pod: p0 is the busier GPU
	if i := f.step(f.hd("a", 5)); i != 0 {
		t.Fatalf("the first admission is served, got %d", i)
	}
	f.atPod("a", f.homeOf("a"))
	f.s.mu.Lock()
	home := f.s.nca2.homes["a"]
	rooms := map[string]int64{}
	for _, p := range []string{"p0", "p1"} {
		rooms[p] = f.s.nca2View(f.s.groups[p]).room(f.s.nca2.AdmitLineTokens)
	}
	f.s.mu.Unlock()
	other := "p1"
	if home == "p1" {
		other = "p0"
	}
	f.s.mu.Lock()
	order := f.s.ncaV2PodOrder(f.s.nca2OutsideHead())
	f.s.mu.Unlock()
	if len(order) == 0 || order[0].g.id != other {
		t.Fatalf("the pod with fewer requests in flight goes first: order starts %v (in flight on %s)", order[0].g.id, home)
	}
	// and the admission follows the order
	if i := f.step(f.hd("b", 5)); i != 0 {
		t.Fatalf("the second admission is served, got %d", i)
	}
	if got := f.homeOf("b"); got != other {
		t.Fatalf("the newcomer must land on the idler pod %q, got %q (rooms %v)", other, got, rooms)
	}
}

// TestTheInFlightKeyBeatsTheRoomKey: A5 — the pod with FEWER REQUESTS IN FLIGHT wins even when the
// other pod has more free CPU. Both numbers are live: the in-flight count is derived from the request
// records at the pass, the room from the admitted active entries.
func TestTheInFlightKeyBeatsTheRoomKey(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	// p0: ONE request in flight behind a 2000-token entry → room 1040
	f.running("p0", "w0", 5, 2000)
	// p1: TWO requests in flight behind 320-token entries → room 2400
	f.running("p1", "w1", 5, 320)
	f.running("p1", "w2", 5, 320)
	f.s.mu.Lock()
	f.s.nca2Derive(f.now)
	rooms := map[string]int64{}
	for _, p := range []string{"p0", "p1"} {
		rooms[p] = f.s.nca2View(f.s.groups[p]).room(f.s.nca2.AdmitLineTokens)
	}
	inflight := map[string]int{"p0": f.s.groups["p0"].nca2.inflight, "p1": f.s.groups["p1"].nca2.inflight}
	f.s.mu.Unlock()
	if !(inflight["p0"] < inflight["p1"] && rooms["p0"] < rooms["p1"]) {
		t.Fatalf("fixture: p0 must have fewer in flight and LESS room (in flight %v, rooms %v)", inflight, rooms)
	}
	f.wait(time.Second)
	if i := f.s.Pass([]Head{f.hd("new", 5)}, f.now); i != 0 {
		t.Fatalf("the newcomer must be admitted, got %d", i)
	}
	if got := f.homeOf("new"); got != "p0" {
		t.Fatalf("A5: the idler GPU wins over the larger room (in flight %v, rooms %v), got %q", inflight, rooms, got)
	}
}

// TestReclaimableSeatsAreEvictedForAFittingHead: A2/S1 — when the budget is the blocker, the pod's
// RECLAIMABLE seats (quantum-complete or silent past idleSeconds) are evicted until the head fits.
func TestReclaimableSeatsAreEvictedForAFittingHead(t *testing.T) {
	f := newFixture(t, "p0")
	// 900 of the 950-block budget, idle: silence past 5 s makes it reclaimable
	if i := f.step(f.hdHeavy("filler", 900, 100)); i != 0 {
		t.Fatalf("the filler must be seated, got %d", i)
	}
	f.atPod("filler", "p0")
	f.s.FirstChunk("r-filler", f.now)
	f.done("filler", "p0")
	f.wait(10 * time.Second)
	if i := f.step(f.hdHeavy("new", 400, 200)); i != 0 {
		t.Fatalf("the newcomer must be admitted over the reclaimable seat, got %d", i)
	}
	if !f.admitted("new") || !seated(f.s, "new") {
		t.Fatal("A2.1 admits and seats the head once the reclaimable seat is evicted")
	}
	if seated(f.s, "filler") {
		t.Fatal("the silent seat holder is the victim (S1: the base rotationVictim order)")
	}
	if !f.admitted("filler") {
		t.Fatal("S4: the evicted seat holder keeps its admission (it is a warm returner next time)")
	}
	if got := f.classOf("new"); got != nca2ClassAdmittedNow {
		t.Fatalf("the seat class: %q", got)
	}
}

// TestAnInFlightSeatIsNeverTheVictim: S1/C10 — a thread with a request in flight is not reclaimable,
// so a head that does not fit the free budget waits rather than evicting it.
func TestAnInFlightSeatIsNeverTheVictim(t *testing.T) {
	f := newFixture(t, "p0")
	if i := f.step(f.hdHeavy("busy", 900, 100)); i != 0 {
		t.Fatalf("the busy thread must be seated, got %d", i)
	}
	f.atPod("busy", "p0") // in flight, and NOT settled
	if i := f.step(f.hdHeavy("new", 400, 200)); i != -1 {
		t.Fatalf("nothing may be served while the only reclaimable seat is in flight, got %d", i)
	}
	if !seated(f.s, "busy") || f.admitted("new") {
		t.Fatal("an in-flight seat holder is never a victim")
	}
}

// TestTheColdValveNeverEvicts: A2/C9 — the cold-entry valve is not budget: a head it holds waits,
// and no seat is taken from anybody for it.
func TestTheColdValveNeverEvicts(t *testing.T) {
	// the stagger cap is the budget's own headroom: fraction 0.99 on 1000 blocks = 640 tokens
	s := newNCA2Tuned(t, func(c *Config) { c.GroupBudgetFraction = 0.99 }, []string{"p0"})
	f := &fixture{t: t, s: s, now: epoch}
	// a seat whose cold charge is outstanding (no first chunk yet) fills the pod's stagger cap
	if i := f.step(f.hd("cold", 20)); i != 0 {
		t.Fatalf("the first seat must be taken, got %d", i)
	}
	f.atPod("cold", "p0")
	if i := f.step(f.hd("held", 5)); i != -1 {
		t.Fatalf("the valve holds the second head on this pod, got %d", i)
	}
	if !seated(f.s, "cold") {
		t.Fatal("the valve must never evict: its charge belongs to a seated thread")
	}
	if f.admitted("held") {
		t.Fatal("a valve-held head is not admitted")
	}
}

// TestFitsIsPure: A2/R1 — the fit test answers a question and changes NOTHING. The state here makes
// BOTH blockers true at once: the budget needs a reclaim, and the cold valve refuses the head. A fit
// test that evicted first (the M5 break) would take the reclaimable seat for a candidate it then
// refuses.
func TestFitsIsPure(t *testing.T) {
	// fraction 0.99 on 1000 blocks: the stagger cap is 640 tokens
	s := newNCA2Tuned(t, func(c *Config) { c.GroupBudgetFraction = 0.99 }, []string{"p0"})
	f := &fixture{t: t, s: s, now: epoch}
	// a RECLAIMABLE seat: settled, then silent past idleSeconds (5 s)
	if i := f.step(f.hdHeavy("filler", 400, 100)); i != 0 {
		t.Fatalf("the filler must be seated, got %d", i)
	}
	f.atPod("filler", "p0")
	f.s.FirstChunk("r-filler", f.now)
	f.done("filler", "p0")
	f.wait(10 * time.Second)
	// a seat whose cold charge FILLS the valve, still in flight (so it is not reclaimable)
	if i := f.step(f.hdHeavy("cold", 400, 700)); i != 0 {
		t.Fatalf("the cold thread must be seated, got %d", i)
	}
	f.atPod("cold", "p0")
	s.mu.Lock()
	g := s.groups["p0"]
	cold := g.coldInFlight
	capTokens := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
	free := s.freeBudget(g)
	seatsBefore := len(g.threads)
	exitsBefore := len(s.counters.exits)
	reclaimable := s.ncaV2ReclaimableBlocks(g, f.now)
	s.mu.Unlock()
	// THE FIXTURE'S OWN FACTS FIRST: a mutated fits() cannot pollute what these assert
	if cold != capTokens || reclaimable == 0 || free >= 300 {
		t.Fatalf("fixture: valve full (%d/%d), a reclaimable seat (%d blocks), free %d < 300", cold, capTokens, reclaimable, free)
	}
	// a head that needs 300 blocks: more than the free budget, and the valve refuses it too
	head := &waiter{id: "big", since: f.now, chains: []Chain{{Keys: keys(300, seedFor("big")), Tokens: 320}}}
	s.mu.Lock()
	fits := s.ncaV2Fits(g, head, f.now, map[fitKey]fitFacts{})
	seatsAfter := len(g.threads)
	exitsAfter := len(s.counters.exits)
	s.mu.Unlock()
	if fits {
		t.Fatal("the valve refuses this head on this pod")
	}
	if seatsAfter != seatsBefore {
		t.Fatalf("fits() took a seat: %d before, %d after", seatsBefore, seatsAfter)
	}
	if exitsAfter != exitsBefore {
		t.Fatalf("fits() booked a departure: %v -> %v", exitsBefore, exitsAfter)
	}
	// and through the DECISION it is the same: nothing is admitted, nobody leaves
	if i := f.step(f.hdHeavy("big", 300, 320)); i != -1 {
		t.Fatalf("the pass must decide nothing, got %d", i)
	}
	if !seated(f.s, "filler") || !seated(f.s, "cold") {
		t.Fatal("no seat may be taken for a valve-held head")
	}
}

// TestTheFitTestIsTheExactAccounting: C1 (the re-check's disagreement). The CPU line's entry is a
// CLAIM about what the pod holds, and it can over-claim — a stale entry after a rebind, a trim, or a
// thread whose next prompt does not extend the previous one exactly. A fit test derived from that
// claim (keys − entry/blockSize) therefore UNDER-estimates the cost, and a decision that trusted it
// would evict seats for a candidate whose exact cost refuses the seat (the re-check's
// eviction-without-a-seat). The fit test is the block index's own accounting: a pod whose entry claims
// far more than the index holds must answer NO, whatever the line says.
func TestTheFitTestIsTheExactAccounting(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	s.mu.Lock()
	// the line claims 4,000 tokens of residence (62 blocks) for the thread ...
	s.groups["p0"].nca2.line = []nca2Entry{{thread: "t", tokens: 4000}}
	s.nca2.homes["t"] = "p0"
	s.nca2.ever["t"] = true
	// ... and the GPU has 30 free blocks of its 950-block budget, while the thread's 40-block chain is
	// NOT in the index: the exact cost is 40 blocks, which does not fit
	g := s.groups["p0"]
	// 920 blocks of somebody else's footprint, so free = 30
	if ok, _ := g.index.TrySeat("filler", []Chain{{Keys: keys(920, seedFor("filler")), Tokens: 1000}}, g.budgetBlocks); !ok {
		s.mu.Unlock()
		t.Fatal("fixture: the filler must take 920 blocks")
	}
	w := &waiter{id: "t", since: epoch, chains: []Chain{{Keys: keys(40, seedFor("t")), Tokens: 2560}}}
	free := s.freeBudget(g)
	exact := int64(g.index.Cost(w.chains))
	fits := s.ncaV2Fits(g, w, epoch, map[fitKey]fitFacts{})
	s.mu.Unlock()
	if free != 30 || exact != 40 {
		t.Fatalf("fixture: free %d blocks, exact cost %d (want 30 / 40)", free, exact)
	}
	if fits {
		t.Fatalf("the fit test must be the exact accounting: cost %d > free %d, and the line's claim "+
			"(entry 4000 tokens) must not be believed over the index", exact, free)
	}
	// and the decision agrees: nothing is seated, nobody is evicted for it
	f := &fixture{t: t, s: s, now: epoch}
	f.s.mu.Lock()
	exitsBefore := len(s.counters.exits)
	f.s.mu.Unlock()
	f.now = f.now.Add(time.Second)
	if i := f.step(f.hdHeavy("t", 40, 2560)); i != -1 {
		t.Fatalf("a candidate the exact cost refuses is not seated, got %d", i)
	}
	f.s.mu.Lock()
	exitsAfter := len(s.counters.exits)
	s.mu.Unlock()
	if exitsAfter != exitsBefore {
		t.Fatalf("no seat may be taken for a candidate the exact cost refuses: %v -> %v", exitsBefore, exitsAfter)
	}
}

// TestPushedOutThreadPrefersItsOldPod: A4 — a priority thread whose old entry still exists on its old
// pod is admitted THERE, when BOTH A5 keys prefer another pod (p0 is running a request and has the
// smaller room; p1 is idle).
func TestPushedOutThreadPrefersItsOldPod(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	f.running("p0", "warm", 5, 640) // p0: one request in flight, room 2400
	f.s.mu.Lock()
	// a was pushed out of p0 and its entry is still in p0's line: A4 prefers p0 for it
	f.s.nca2.homes["a"] = "p0"
	f.s.nca2.ever["a"] = true
	f.s.nca2.pushed["a"] = true
	f.s.groups["p0"].nca2.line = append(f.s.groups["p0"].nca2.line, nca2Entry{thread: "a", tokens: 320})
	f.s.waiters["a"] = &waiter{id: "a", since: f.now.Add(-2 * time.Minute)}
	f.s.nca2Derive(f.now)
	rooms := map[string]int64{}
	for _, p := range []string{"p0", "p1"} {
		rooms[p] = f.s.nca2View(f.s.groups[p]).room(f.s.nca2.AdmitLineTokens)
	}
	inflight := map[string]int{"p0": f.s.groups["p0"].nca2.inflight, "p1": f.s.groups["p1"].nca2.inflight}
	f.s.mu.Unlock()
	if !(inflight["p0"] > inflight["p1"] && rooms["p0"] < rooms["p1"]) {
		t.Fatalf("fixture: BOTH A5 keys must prefer p1 (in flight %v, rooms %v)", inflight, rooms)
	}
	f.wait(time.Second)
	if i := f.s.Pass([]Head{f.hd("a", 5)}, f.now); i != 0 {
		t.Fatalf("the priority thread must be admitted on its old pod, got %d", i)
	}
	if got := f.homeOf("a"); got != "p0" {
		t.Fatalf("A4: the old pod must win over the A5 order (p1 is idler and roomier), got %q", got)
	}
	if f.classOf("a") != nca2ClassAdmittedNow {
		t.Fatalf("the class of a re-admission is admitted_now, got %q", f.classOf("a"))
	}
}

// TestPushedOutActiveThreadLeavesTheGpuAndReturnsAsPriority: L4 + A6 + I3 — an ACTIVE admitted thread
// pushed past the line by a TOUCH (its sibling's prompt grew: the active threads alone exceeded the
// line) is marked exiting (no dispatch, the seat kept for the request in flight), exits when that
// request ends, and comes back as the PRIORITY class.
func TestPushedOutActiveThreadLeavesTheGpuAndReturnsAsPriority(t *testing.T) {
	f := newFixture(t, "p0")
	// a and b are both admitted and both running: 640 + 640 at the head, inside the line
	if i := f.step(f.hd("a", 10)); i != 0 {
		t.Fatalf("a must be admitted, got %d", i)
	}
	f.atPod("a", "p0")
	if i := f.step(f.hd("b", 10)); i != 0 {
		t.Fatalf("b must be admitted, got %d", i)
	}
	f.atPod("b", "p0")
	if !f.admitted("a") || !f.admitted("b") {
		t.Fatal("fixture: an admission never pushes an ACTIVE entry out (I2)")
	}
	// b's next turn is a much longer prompt: 2560 tokens at the head puts a at 3200 > L_a
	if i := f.step(f.hd("b", 40)); i != 0 {
		t.Fatalf("b's grown turn must be dispatched (it is seated), got %d", i)
	}
	f.atPod("b", "p0") // L1: the touch writes 2560 at the head
	if f.admitted("a") {
		t.Fatal("A6: the touch pushed the deepest active entry past the line")
	}
	if !f.admitted("b") || !seated(f.s, "b") {
		t.Fatal("the touching thread stays admitted and seated")
	}
	if !seated(f.s, "a") {
		t.Fatal("I3: an ACTIVE pushed-out thread keeps its seat for its in-flight request")
	}
	f.s.mu.Lock()
	exiting := f.s.groups["p0"].index.Exiting("a")
	f.s.mu.Unlock()
	if !exiting {
		t.Fatal("it is MARKED exiting: dispatch skips it and the drain releases it when the request ends")
	}
	// its request ends: the base drain exits it (nothing queued), and its next head is a waiter
	f.s.FirstChunk("r-a", f.now)
	f.done("a", "p0")
	f.step()
	if seated(f.s, "a") {
		t.Fatal("the drain must release the marked seat once nothing is in flight")
	}
	// b settles too: its entry stays, but an INACTIVE entry is space (room = the whole line)
	f.s.FirstChunk("r-b", f.now)
	f.done("b", "p0")
	f.step()
	if i := f.step(f.hd("a", 10)); i < 0 {
		t.Fatal("a's next head must be considered (A2.1: it is the priority class)")
	}
	if got := f.classOf("a"); got != nca2ClassAdmittedNow && got != "" {
		t.Fatalf("a serves as an ADMITTED thread, never as a seated one: class %q", got)
	}
	// and the class it came from is the priority list
	if !f.admitted("a") {
		t.Fatal("fixture: the room allows its re-admission")
	}
	f.s.mu.Lock()
	ever := f.s.nca2.ever["a"]
	f.s.mu.Unlock()
	if !ever {
		t.Fatal("a was admitted before: it is the PRIORITY class when outside")
	}
}

// budgetFixture builds A9's state: two 400-block seats in the 950-block budget — A idle (so its seat
// becomes reclaimable), B running (never a victim) — one outside head of 900 blocks (it needs BOTH
// seats), and the given warm returners waiting on p0.
func budgetFixture(t *testing.T, returners map[string]int) (*fixture, Head) {
	t.Helper()
	f := newFixture(t, "p0")
	f.running("p0", "A", 400, 320)
	f.s.Settle("rq-A", "p0", f.now)
	f.running("p0", "B", 400, 320)
	ids := make([]string, 0, len(returners))
	for id := range returners {
		ids = append(ids, id)
	}
	sort.Strings(ids) // deterministic setup order
	f.s.mu.Lock()
	for _, id := range ids {
		f.s.nca2.homes[id] = "p0"
		f.s.nca2.admitted[id] = true
		f.s.nca2.ever[id] = true
		f.s.groups["p0"].nca2.line = append(f.s.groups["p0"].nca2.line, nca2Entry{thread: id, tokens: 320})
		f.s.waiters[id] = &waiter{id: id, since: f.now.Add(-time.Minute)}
	}
	f.s.mu.Unlock()
	f.wait(10 * time.Second) // A is silent: reclaimable
	return f, f.hdHeavy("big", 900, 320)
}

// returnerHeads is one head per waiting returner, in a stable order (they must be offered every pass,
// or the queue drops them).
func returnerHeads(f *fixture, returners map[string]int) []Head {
	ids := make([]string, 0, len(returners))
	for id := range returners {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Head, 0, len(ids))
	for _, id := range ids {
		out = append(out, f.hdHeavy(id, returners[id], 320))
	}
	return out
}

// TestTheAdmissionBudgetGathersSeatsAcrossPasses: A9 — the X3 door freeze (verifier 2). A prompt bigger
// than one seat needs SEVERAL seats freed, and they become reclaimable one quantum at a time. The pod
// frees what it can for the outside head and HOLDS it (the head's admission budget), so a returner
// that would need the reserved space cannot take it in the same pass; the next pass finishes the job
// and admits the head.
func TestTheAdmissionBudgetGathersSeatsAcrossPasses(t *testing.T) {
	returners := map[string]int{"ret": 300}
	f, big := budgetFixture(t, returners)
	heads := append([]Head{big}, returnerHeads(f, returners)...)
	f.now = f.now.Add(time.Second)
	if i := f.s.Pass(heads, f.now); i != -1 {
		t.Fatalf("nothing can be served yet (the head needs two seats), got %d", i)
	}
	if seated(f.s, "big") {
		t.Fatal("one freed seat is not enough for a 900-block prompt")
	}
	if seated(f.s, "ret") {
		t.Fatal("A9: a returner must not take the space the head's budget is holding")
	}
	if seated(f.s, "A") {
		t.Fatal("A was the reclaimable seat the budget spent")
	}
	if !seated(f.s, "B") {
		t.Fatal("B has a request in flight: never a victim")
	}
	// the second seat becomes reclaimable: the reservation from the FIRST pass is still the head's
	f.s.FirstChunk("rq-B", f.now)
	f.s.Settle("rq-B", "p0", f.now)
	f.wait(10 * time.Second)
	f.now = f.now.Add(time.Second)
	if i := f.s.Pass(heads, f.now); i < 0 {
		t.Fatal("the head must be admitted once the budget has gathered both seats")
	}
	if !f.admitted("big") || !seated(f.s, "big") {
		t.Fatal("A9: the head is admitted after accumulating across passes")
	}
}

// TestAReturnerIsSeatedFromSurplusBeyondTheReservation: A9's OTHER half (the owner's clarification) —
// the reservation protects only the blocks the head still needs. The surplus beyond it is the pod's,
// so a returner that fits INSIDE the surplus is seated in the very pass that budgets for the head.
// A total ban would stall it for no benefit to the head.
func TestAReturnerIsSeatedFromSurplusBeyondTheReservation(t *testing.T) {
	returners := map[string]int{"small": 150, "too-big": 300}
	f, big := budgetFixture(t, returners)
	heads := append([]Head{big}, returnerHeads(f, returners)...)
	f.now = f.now.Add(time.Second)
	f.s.mu.Lock()
	free := f.s.freeBudget(f.s.groups["p0"])
	f.s.mu.Unlock()
	if free != 150 {
		t.Fatalf("fixture: the pod must have 150 free blocks before the budget acts, got %d", free)
	}
	if i := f.s.Pass(heads, f.now); i < 0 {
		t.Fatal("a returner must be seated from the surplus")
	}
	// after A's eviction free = 550 and the head still needs 350: the surplus is 200, so the 150-block
	// returner fits and the 300-block one does not
	if !seated(f.s, "small") {
		t.Fatal("A9: the surplus beyond the reservation belongs to the pod: a fitting returner is seated")
	}
	if seated(f.s, "too-big") {
		t.Fatal("a returner larger than the surplus may not eat the head's budget")
	}
	if seated(f.s, "big") {
		t.Fatal("the head still needs the next pass's seat")
	}
	f.s.mu.Lock()
	reserved := f.s.groups["p0"].nca2.reservedBlocks
	f.s.mu.Unlock()
	if reserved <= 0 {
		t.Fatal("the head's budget must still be outstanding for the next pass")
	}
}

// TestTheReservationIsDroppedWhenTheRoomShrinks: A9's third drop condition (the gate's MAJOR-1). The
// pod holds a reservation for the outside head; then a deep admitted entry's thread comes back, the
// LAA deepens, the room falls below the head's own f — and the hold must be released, or a returner
// that fits the freed space stays unseated for as long as that head is the head.
func TestTheReservationIsDroppedWhenTheRoomShrinks(t *testing.T) {
	returners := map[string]int{"ret": 300}
	f, big := budgetFixture(t, returners)
	heads := append([]Head{big}, returnerHeads(f, returners)...)
	f.now = f.now.Add(time.Second)
	if i := f.s.Pass(heads, f.now); i != -1 {
		t.Fatalf("pass 1 must decide nothing (the head needs two seats), got %d", i)
	}
	f.s.mu.Lock()
	reserved := f.s.groups["p0"].nca2.reservedBlocks
	f.s.mu.Unlock()
	if reserved <= 0 {
		t.Fatalf("fixture: the head's budget must be outstanding after pass 1, got %d", reserved)
	}
	if seated(f.s, "ret") {
		t.Fatal("fixture: the returner must be held out in pass 1")
	}
	// a DEEP admitted entry's thread returns: it is admitted and waiting, so it becomes the LAA and the
	// room falls below the head's f (320 tokens)
	f.s.mu.Lock()
	f.s.nca2.homes["deep"] = "p0"
	f.s.nca2.admitted["deep"] = true
	f.s.nca2.ever["deep"] = true
	// appended at the TAIL: deep becomes the LAA at position 2860 (room 180 < f 320) without pushing the
	// few shallow entries past the line
	f.s.groups["p0"].nca2.line = append(f.s.groups["p0"].nca2.line, nca2Entry{thread: "deep", tokens: 1900})
	f.s.waiters["deep"] = &waiter{id: "deep", since: f.now.Add(-time.Second)}
	f.s.mu.Unlock()
	deep := f.hdHeavy("deep", 5, 320)
	f.now = f.now.Add(time.Second)
	heads2 := append([]Head{big, deep}, returnerHeads(f, returners)...)
	if i := f.s.Pass(heads2, f.now); i < 0 {
		t.Fatal("the returner must be seated once the reservation is released")
	}
	f.s.mu.Lock()
	room := f.s.nca2View(f.s.groups["p0"]).room(f.s.nca2.AdmitLineTokens)
	after := f.s.groups["p0"].nca2.reservedBlocks
	f.s.mu.Unlock()
	if room >= 320 {
		t.Fatalf("fixture: the room must be below the head's f (320), got %d", room)
	}
	if after != 0 {
		t.Fatalf("A9: the reservation must be dropped when the head no longer fits the room, got %d", after)
	}
	if !seated(f.s, "ret") {
		t.Fatal("the freed space belongs to the pod's returners once the hold is dropped")
	}
}

// TestTheDecisionCapIsWaitersPlusFourPods: A1 — the cap's own arithmetic (a pure function so the rule
// is checkable) and the pass's report that it never exceeded it.
func TestTheDecisionCapIsWaitersPlusFourPods(t *testing.T) {
	for _, c := range []struct{ waiters, pods, want int }{
		{0, 1, 4}, {10, 3, 22}, {500, 3, 512},
	} {
		if got := ncaV2DecisionCap(c.waiters, c.pods); got != c.want {
			t.Fatalf("cap(%d waiters, %d pods) = %d, want %d (= waiters + 4 x pods)", c.waiters, c.pods, got, c.want)
		}
	}
	f := newFixture(t, "p0", "p1")
	var heads []Head
	for i := 0; i < 30; i++ {
		heads = append(heads, f.hdHeavy(string(rune('a'+i)), 20, 100))
	}
	f.step(heads...)
	snap := f.s.Snapshot(f.now).NCA2
	if snap.DecisionsThisPass > ncaV2DecisionCap(30, 2) {
		t.Fatalf("the pass took %d decisions, past its own cap %d", snap.DecisionsThisPass, ncaV2DecisionCap(30, 2))
	}
}

// TestTheSafetyCapBoundsThePass: A1 — a pass with more waiters than it can seat terminates, seats
// what fits, and leaves the rest in the global queue.
func TestTheSafetyCapBoundsThePass(t *testing.T) {
	f := newFixture(t, "p0")
	var heads []Head
	for i := 0; i < 40; i++ {
		heads = append(heads, f.hdHeavy(string(rune('a'+i%26))+string(rune('0'+i/26)), 100, 100))
	}
	served := f.step(heads...)
	if served < 0 {
		t.Fatal("at least one head must be admitted and served")
	}
	f.s.mu.Lock()
	waiting := len(f.s.waiters)
	f.s.mu.Unlock()
	if waiting == 0 {
		t.Fatal("fixture: 40 heads of 100 blocks cannot all fit a 950-block pod")
	}
	// a second pass must make progress on the rest (the queue is global and the pod re-ordered)
	if i := f.step(heads...); i < 0 {
		t.Fatal("the next pass must seat another head once the first one's seat is reclaimable")
	}
}
