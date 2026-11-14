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
	order := f.s.ncaV2PodOrder()
	f.s.mu.Unlock()
	if len(order) == 0 || order[0].id != other {
		t.Fatalf("the pod with fewer requests in flight goes first: order starts %v (in flight on %s)", order[0].id, home)
	}
	// and the admission follows the order
	if i := f.step(f.hd("b", 5)); i != 0 {
		t.Fatalf("the second admission is served, got %d", i)
	}
	if got := f.homeOf("b"); got != other {
		t.Fatalf("the newcomer must land on the idler pod %q, got %q (rooms %v)", other, got, rooms)
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

// TestPushedOutThreadPrefersItsOldPod: A4 — a priority thread whose old entry still exists on its old
// pod is admitted THERE, even when the A5 order puts another pod first.
func TestPushedOutThreadPrefersItsOldPod(t *testing.T) {
	f := newFixture(t, "p0", "p1")
	// a lives on p0 and is admitted; b follows it on p0 and is active
	if i := f.step(f.hd("a", 5)); i != 0 {
		t.Fatalf("a must be seated, got %d", i)
	}
	f.atPod("a", f.homeOf("a"))
	homeA := f.homeOf("a")
	f.s.mu.Lock()
	g := f.s.groups[homeA]
	f.s.exit(g, g.threads["a"], "idle", f.now) // unseated, still admitted (S4)
	f.s.nca2.admitted["a"] = false             // and pushed out: it is a PRIORITY thread now
	f.s.mu.Unlock()
	if f.homeOf("a") != homeA {
		t.Fatal("fixture: the home survives the push-out (A4's old pod)")
	}
	// the A5 order: give the OTHER pod more room and fewer in-flight requests
	idle := "p1"
	if homeA == "p1" {
		idle = "p0"
	}
	f.s.mu.Lock()
	f.s.groups[idle].nca2.inflight = 0
	f.s.groups[homeA].nca2.inflight = 3
	f.s.mu.Unlock()
	f.wait(time.Second)
	if i := f.s.Pass([]Head{f.hd("a", 5)}, f.now); i != 0 {
		t.Fatalf("the priority thread must be admitted on its old pod, got %d", i)
	}
	if got := f.homeOf("a"); got != homeA {
		t.Fatalf("A4: the old pod wins over the A5 order, got %q want %q", got, homeA)
	}
	if got := f.classOf("a"); got != nca2ClassAdmittedNow {
		t.Fatalf("the class of a re-admission is admitted_now, got %q", got)
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
