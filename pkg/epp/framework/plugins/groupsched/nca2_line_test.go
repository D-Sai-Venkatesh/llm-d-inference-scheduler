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

// THE CPU LINE: the order (a touch moves a thread to the head), the trim (past C), the geometry
// (positions, the LAA, the room), and the push-out classes (room / growth / overflow) with what
// each one does to the GPU layer. Plus §2's ACTIVITY, including the leak bound.

import (
	"testing"
	"time"
)

// i64 and f64 are the nca2 config's pointer keys (presence IS the fact the decoder keeps).
func i64(v int64) *int64     { return &v }
func f64(v float64) *float64 { return &v }

// nca2LineConfig is the fixture's tier: C = 3200 tokens (50 blocks of 64) with the admission line
// at 3040, so a thread's position is readable straight off its block count.
const (
	lineC     = 3200
	lineAdmit = 3040
)

// newNCA2 is a scheduler in nca2 with a small, exact CPU tier and one 1000-block engine per pod id.
func newNCA2(t testing.TB, pods []string) *Scheduler {
	t.Helper()
	return newNCA2Tuned(t, nil, pods)
}

// newNCA2Tuned is the same with one knob the test is about (the cold valve's cap, for instance).
func newNCA2Tuned(t testing.TB, tune func(*Config), pods []string) *Scheduler {
	t.Helper()
	cfg := defaultConfig()
	cfg.Mode = modeNCA2
	cfg.IdleSeconds = 5
	cfg.CPUTierTokens = i64(lineC)
	cfg.AdmitLine = f64(0.95)
	cfg.TrustLine = f64(0.95)
	if tune != nil {
		tune(&cfg)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pods {
		if err := s.SetNumBlocksFor(p, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

// admit is A2.1's state transition, taken directly for a test: the seat (the base's, unchanged),
// the home, the admission STATE and L0's provisional entry at the head of the pod's line.
func admit(t *testing.T, s *Scheduler, pod, id string, blocks int) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groups[pod]
	w := &waiter{id: id, since: epoch, chains: chain(id, blocks), fairnessID: "f-" + id}
	if !s.seat(g, w, g.index.ColdTokens(w.chains), epoch) {
		t.Fatalf("the fixture could not seat %s on %s", id, pod)
	}
	s.nca2.homes[id] = pod
	s.nca2.admitted[id] = true
	s.nca2.ever[id] = true
	s.nca2LineSet(g, id, int64(blocks*64), epoch, pushoutRoom)
	// the base's seat inherits the head's presence; a fixture thread is INACTIVE unless the test
	// says otherwise (that is the class the 5 s grace and the room are about)
	if m := g.threads[id]; m != nil {
		m.waitingSince = time.Time{}
	}
}

// lineOf is the pod's line as (thread, tokens) pairs, head first — what a reader sees in a debugger.
func lineOf(s *Scheduler, pod string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, e := range s.groups[pod].nca2.line {
		out = append(out, e.thread)
	}
	return out
}

// admittedNow is the admission STATE, read under the lock.
func admittedNow(s *Scheduler, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.nca2.admitted[id]
}

// waiting marks a thread as WAITING (a queued head) without touching the queue: enough for the
// activity test, which is what the decisions read.
func waiting(s *Scheduler, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if m := s.groupOf(id).threads[id]; m != nil {
		m.waitingSince = epoch
	}
}

// TestTouchIsTheHead: L1 — a request reaching the pod moves the thread's entry to the head with its
// own prompt tokens, and its position becomes its own size.
func TestTouchIsTheHead(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10) // 640 tokens
	admit(t, s, "p0", "b", 5)  // 320
	if got := lineOf(s, "p0"); len(got) != 2 || got[0] != "b" || got[1] != "a" {
		t.Fatalf("the newest admission is the head: %v", got)
	}
	s.mu.Lock()
	if pos, _ := s.nca2PositionOn(s.groups["p0"], "a"); pos != 960 {
		t.Fatalf("a's position is the sum from the head through it: %d, want 960", pos)
	}
	s.mu.Unlock()

	// a's next request reaches the pod with 8 blocks: it is now the head, position 512
	s.mu.Lock()
	s.nca2LineSet(s.groups["p0"], "a", 8*64, epoch.Add(time.Second), pushoutTouch)
	pos, ok := s.nca2PositionOn(s.groups["p0"], "a")
	line := lineOfLocked(s.groups["p0"])
	s.mu.Unlock()
	if !ok || pos != 512 {
		t.Fatalf("a touch writes the entry at the head: pos %d, want 512", pos)
	}
	if len(line) != 2 || line[0] != "a" || line[1] != "b" {
		t.Fatalf("the touched entry is the head and the thread has ONE entry: %v", line)
	}
}

// lineOfLocked reads a line's thread order under the caller's lock.
func lineOfLocked(g *engineGroup) []string {
	var out []string
	for _, e := range g.nca2.line {
		out = append(out, e.thread)
	}
	return out
}

// TestTrimIsPastTheTier: L2 — everything from the first entry past C is dropped, whole.
func TestTrimIsPastTheTier(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 20) // 1280 tokens
	admit(t, s, "p0", "b", 20) // 1280: position 2560
	admit(t, s, "p0", "c", 20) // 1280: a is now at 3840 > C
	if got := lineOf(s, "p0"); len(got) != 2 || got[0] != "c" || got[1] != "b" {
		t.Fatalf("everything past C must be dropped by the trim (a is at 3840): %v", got)
	}
	// an OVERSIZE entry (larger than the whole tier) has no place in the line at all
	s.mu.Lock()
	s.nca2LineSet(s.groups["p0"], "d", lineC+64, epoch, pushoutTouch)
	_, has := s.nca2PositionOn(s.groups["p0"], "d")
	s.mu.Unlock()
	if has {
		t.Fatal("an entry larger than the CPU tier cannot survive the trim")
	}
}

// TestRoomIsBelowTheLowestAdmittedActive: §2 — room is L_a minus the position of the admitted
// ACTIVE entry nearest the tail; an inactive admitted entry is space, and a line with no admitted
// active entry has the whole line as room.
func TestRoomIsBelowTheLowestAdmittedActive(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10) // 640
	admit(t, s, "p0", "b", 5)  // 320, a is now at 960
	waiting(s, "a")
	waiting(s, "b")
	s.mu.Lock()
	v := s.nca2View(s.groups["p0"])
	s.mu.Unlock()
	if v.laa != "a" || v.laaPosition != 960 {
		t.Fatalf("the LAA is the admitted active entry nearest the tail: %q at %d", v.laa, v.laaPosition)
	}
	if got := v.room(s.nca2.AdmitLineTokens); got != lineAdmit-960 {
		t.Fatalf("room = L_a - LAA position: %d, want %d", got, lineAdmit-960)
	}
	if v.admittedThreads != 2 || v.admittedTokens != 960 {
		t.Fatalf("admitted threads/tokens: %d / %d, want 2 / 960", v.admittedThreads, v.admittedTokens)
	}

	// a goes INACTIVE: b is now the lowest active entry, and a's 640 tokens below it are room
	s.mu.Lock()
	s.groups["p0"].threads["a"].waitingSince = time.Time{}
	delete(s.waiters, "a")
	v = s.nca2View(s.groups["p0"])
	s.mu.Unlock()
	if v.laa != "b" || v.laaPosition != 320 {
		t.Fatalf("with a inactive the LAA is b: %q at %d", v.laa, v.laaPosition)
	}
	if got := v.room(s.nca2.AdmitLineTokens); got != lineAdmit-320 {
		t.Fatalf("room with an inactive tail entry: %d, want %d", got, lineAdmit-320)
	}

	// nothing active: the whole line is room
	s.mu.Lock()
	s.groups["p0"].threads["b"].waitingSince = time.Time{}
	delete(s.waiters, "b")
	v = s.nca2View(s.groups["p0"])
	s.mu.Unlock()
	if got := v.room(s.nca2.AdmitLineTokens); got != s.nca2.AdmitLineTokens {
		t.Fatalf("no admitted active entry: room is the whole line, got %d", got)
	}
	if v.admittedActiveTokens != 0 || v.activeThreads != 0 {
		t.Fatalf("no active threads: %d / %d", v.admittedActiveTokens, v.activeThreads)
	}
}

// TestPushoutOfAnInactiveSeatHolderExitsIt: L4 with reason room (an admission's provisional entry):
// an INACTIVE admitted seat holder loses the admission AND the seat it is not using (exit pushout).
func TestPushoutOfAnInactiveSeatHolderExitsIt(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10) // 640, INACTIVE (no queued head, nothing in flight)
	admit(t, s, "p0", "b", 40) // 2560: a is now at 3200 > L_a
	if admittedNow(s, "a") {
		t.Fatal("a was pushed past the admission line: it is not admitted any more")
	}
	if seated(s, "a") {
		t.Fatal("an INACTIVE pushed-out seat holder must lose the seat it is not using (I3)")
	}
	if !admittedNow(s, "b") || !seated(s, "b") {
		t.Fatal("the admitting thread keeps its admission and its seat")
	}
	s.mu.Lock()
	exits := map[string]int64{}
	for k, v := range s.counters.exits {
		exits[k] = v
	}
	s.mu.Unlock()
	if exits[exitReasonPushout] != 1 {
		t.Fatalf("the exit must be booked under the pushout reason: %v", exits)
	}
}

// TestPushoutOfAnActiveSeatHolderKeepsTheSeat: L4 with an ACTIVE thread (a touch, so the reason is
// overflow): it keeps the seat for its in-flight request, MARKED exiting — the base's mid-flight
// machinery, which is what makes its next head go outside.
func TestPushoutOfAnActiveSeatHolderKeepsTheSeat(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10) // 640
	admit(t, s, "p0", "b", 30) // 1920 — a is at 2560, inside the line, admitted
	waiting(s, "a")            // a's next head is queued: ACTIVE
	var got []pushoutEvent
	s.mu.Lock()
	s.nca2.onPushout = func(reason string, active bool) { got = append(got, pushoutEvent{reason, active}) }
	s.nca2LineSet(s.groups["p0"], "b", 40*64, epoch.Add(time.Second), pushoutTouch) // a → 3200
	s.mu.Unlock()
	if admittedNow(s, "a") {
		t.Fatal("b's touch of its own size pushed a past the line: a is not admitted any more")
	}
	// A6/C13: an ACTIVE entry pushed by a touch is OVERFLOW, and that is the whole of A6
	if len(got) != 1 || got[0].reason != pushoutOverflow || !got[0].active {
		t.Fatalf("an active entry pushed by a touch is an overflow push-out, got %v", got)
	}
	if !seated(s, "a") {
		t.Fatal("an ACTIVE pushed-out thread keeps its seat for its in-flight request (I3)")
	}
	s.mu.Lock()
	exiting := s.groups["p0"].index.Exiting("a")
	s.mu.Unlock()
	if !exiting {
		t.Fatal("it is MARKED exiting: dispatch skips it and the drain releases it once nothing is in flight")
	}
	if admittedNow(s, "b") == false {
		t.Fatal("the touching thread stays admitted")
	}
}

// TestPushoutByATouchOfAnInactiveEntryIsGrowth: L4 with an INACTIVE pushed entry and a TOUCH as the
// cause: the reason is growth (another thread's touch moved it past the line), and because it is
// inactive it loses its seat at once (I3).
func TestPushoutByATouchOfAnInactiveEntryIsGrowth(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10) // 640, INACTIVE
	admit(t, s, "p0", "b", 30) // 1920: a is at 2560, inside the line
	var got []pushoutEvent
	s.mu.Lock()
	s.nca2.onPushout = func(reason string, active bool) { got = append(got, pushoutEvent{reason, active}) }
	s.nca2LineSet(s.groups["p0"], "b", 40*64, epoch.Add(time.Second), pushoutTouch) // a → 3200
	s.mu.Unlock()
	if admittedNow(s, "a") {
		t.Fatal("the touch pushed a past the line")
	}
	// A6/L4: an INACTIVE entry pushed by a TOUCH is GROWTH (the overflow class is the ACTIVE one)
	if len(got) != 1 || got[0].reason != pushoutGrowth || got[0].active {
		t.Fatalf("an inactive entry pushed by a touch is a growth push-out, got %v", got)
	}
	if seated(s, "a") {
		t.Fatal("I3: an INACTIVE pushed-out thread loses the seat it is not using")
	}
	if !admittedNow(s, "b") {
		t.Fatal("the touching thread stays admitted")
	}
}

// TestEngineRemovalResetsTheLineAndKeepsThePriorityClass: S5 — an engine removal resets that pod's CPU
// line, its threads LOSE their admission (booked as push-outs with reason restart), their homes are
// gone, and the priority class survives (they were admitted before).
func TestEngineRemovalResetsTheLineAndKeepsThePriorityClass(t *testing.T) {
	s := newNCA2(t, []string{"p0", "p1"})
	admit(t, s, "p0", "a", 10)
	admit(t, s, "p0", "b", 5)
	var got []pushoutEvent
	s.mu.Lock()
	s.nca2.onPushout = func(reason string, active bool) { got = append(got, pushoutEvent{reason, active}) }
	s.groups["p0"].threads["a"].waitingSince = epoch // ACTIVE: a restart push-out of an active thread
	s.mu.Unlock()
	s.RemoveEngine("p0", epoch)
	s.mu.Lock()
	gone := s.groups["p0"] == nil
	lineGone := true
	for _, g := range s.groups {
		if g.nca2 != nil && len(g.nca2.line) != 0 {
			lineGone = false
		}
	}
	aAdmitted, aEver, aHome := s.nca2.admitted["a"], s.nca2.ever["a"], s.nca2.homes["a"]
	s.mu.Unlock()
	if !gone {
		t.Fatal("the group must be torn down")
	}
	if !lineGone {
		t.Fatal("S5: the removed pod's line is reset (no other line carries its entries)")
	}
	if aAdmitted {
		t.Fatal("S5: its threads lose their admission")
	}
	if aHome != "" {
		t.Fatalf("S5: no home on a pod that no longer exists, got %q", aHome)
	}
	if !aEver {
		t.Fatal("S5: the thread WAS admitted, so it is the PRIORITY class when it comes back")
	}
	restart := 0
	for _, ev := range got {
		if ev.reason != pushoutRestart {
			t.Fatalf("S5 books a restart push-out, got %v", ev)
		}
		restart++
	}
	if restart != 2 {
		t.Fatalf("both admitted threads lose their admission, got %d restart push-outs", restart)
	}
}

// TestActivityAndTheLeakBound: §2 — a thread is ACTIVE while a head is queued or a request is in
// flight; a released request that never arrives stops counting at lostRequestSeconds, an at-pod one
// is LEAKED at maxInflightSeconds (counted once), and both stop holding the thread active.
func TestActivityAndTheLeakBound(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	admit(t, s, "p0", "a", 10)
	if s.nca2Active("a") {
		t.Fatal("a seat with no queued head and nothing in flight is INACTIVE")
	}
	waiting(s, "a")
	if !s.nca2Active("a") {
		t.Fatal("a queued head is activity")
	}
	s.mu.Lock()
	s.groups["p0"].threads["a"].waitingSince = time.Time{}
	delete(s.waiters, "a")
	// a request released and NEVER at a pod: in flight until lostRequestSeconds
	s.reqs["r-lost"] = &reqRec{thread: "a", released: epoch, counted: true}
	s.nca2Derive(epoch.Add(100 * time.Second))
	inFlight := s.nca2Active("a")
	s.nca2Derive(epoch.Add(400 * time.Second))
	lost := s.nca2Active("a")
	// an AT-POD request: in flight until maxInflightSeconds, then leaked (once)
	s.reqs["r-pod"] = &reqRec{thread: "a", released: epoch, pod: epoch, engine: "p0", counted: true}
	s.nca2Derive(epoch.Add(400 * time.Second))
	atPod := s.nca2Active("a")
	s.nca2Derive(epoch.Add(2 * defaultMaxInflightSeconds * time.Second))
	leaked := s.reqs["r-pod"].leaked
	active := s.nca2Active("a")
	s.mu.Unlock()
	if !inFlight {
		t.Fatal("a released request that has not completed is IN FLIGHT (a released turn counts)")
	}
	if lost {
		t.Fatal("a released request that never reached a pod stops counting at lostRequestSeconds")
	}
	if !atPod {
		t.Fatal("an at-pod request is in flight")
	}
	if !leaked || active {
		t.Fatalf("an at-pod request past maxInflightSeconds is leaked and stops holding its thread active (leaked=%v active=%v)", leaked, active)
	}
}

// TestWarmReturnersNearestTheTailFirst: A2.2's order — the admitted waiting threads of the pod, the
// LOWEST position (nearest the tail) first, and only that pod's own.
func TestWarmReturnersNearestTheTailFirst(t *testing.T) {
	s := newNCA2(t, []string{"p0", "p1"})
	s.mu.Lock()
	// a line with three admitted entries at known positions: head(320) → mid(960) → tail(1600)
	g := s.groups["p0"]
	for _, e := range []struct {
		id     string
		tokens int64
	}{{"tail", 640}, {"mid", 640}, {"head", 320}} {
		g.nca2.line = append([]nca2Entry{{thread: e.id, tokens: e.tokens}}, g.nca2.line...)
		s.nca2.homes[e.id] = "p0"
		s.nca2.admitted[e.id] = true
		s.nca2.ever[e.id] = true
		s.waiters[e.id] = &waiter{id: e.id, since: epoch}
	}
	// a different pod's admitted waiter, never a returner of p0
	s.groups["p1"].nca2.line = []nca2Entry{{thread: "other", tokens: 320}}
	s.nca2.homes["other"] = "p1"
	s.nca2.admitted["other"] = true
	s.waiters["other"] = &waiter{id: "other", since: epoch}
	if pos, _ := s.nca2PositionOn(g, "mid"); pos != 960 {
		t.Fatalf("fixture positions: mid at %d, want 960", pos)
	}
	got := s.nca2WarmReturners(g)
	s.mu.Unlock()
	if len(got) != 3 || got[0].id != "tail" || got[1].id != "mid" || got[2].id != "head" {
		t.Fatalf("the returner nearest the tail (the deepest position) goes first, and only this pod's: %v", ids(got))
	}
}

// ids is a waiter list's ids, for a readable failure.
func ids(ws []*waiter) []string {
	out := make([]string, 0, len(ws))
	for _, w := range ws {
		out = append(out, w.id)
	}
	return out
}

// TestOutsideHeadIsPriorityThenNewcomer: A3/P2 — the priority class strictly before the newcomers,
// the oldest arrival first within a class, and the id as the total tie-break.
func TestOutsideHeadIsPriorityThenNewcomer(t *testing.T) {
	s := newNCA2(t, []string{"p0"})
	s.mu.Lock()
	s.waiters["new-old"] = &waiter{id: "new-old", since: epoch.Add(-10 * time.Second)}
	s.waiters["new-young"] = &waiter{id: "new-young", since: epoch.Add(-1 * time.Second)}
	s.waiters["prio"] = &waiter{id: "prio", since: epoch.Add(-2 * time.Second)}
	s.nca2.ever["prio"] = true
	head := s.nca2OutsideHead()
	s.mu.Unlock()
	if head == nil || head.id != "prio" {
		t.Fatalf("a priority thread goes before every newcomer, however old: %v", head)
	}
	s.mu.Lock()
	s.waiters["prio2"] = &waiter{id: "prio2", since: epoch.Add(-5 * time.Second)}
	s.nca2.ever["prio2"] = true
	head = s.nca2OutsideHead()
	delete(s.waiters, "prio")
	delete(s.waiters, "prio2")
	s.nca2.ever["prio2"] = false
	head2 := s.nca2OutsideHead()
	s.mu.Unlock()
	if head == nil || head.id != "prio2" {
		t.Fatalf("the oldest priority thread is the head: %v", ids([]*waiter{head}))
	}
	if head2 == nil || head2.id != "new-old" {
		t.Fatalf("with no priority thread left the oldest newcomer is the head: %v", ids([]*waiter{head2}))
	}
}

// entriesOf counts the thread's entries across EVERY pod's line: the invariant is one, at most.
func entriesOf(s *Scheduler, id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, g := range s.groups {
		for _, e := range g.nca2.line {
			if e.thread == id {
				n++
			}
		}
	}
	return n
}

// TestOneEntryPerThreadAcrossPods: L0's merge — a re-admission (or a touch) on the pod the entry is
// already on MOVES it, never duplicates it, and a home move leaves ONE entry in the whole fleet.
func TestOneEntryPerThreadAcrossPods(t *testing.T) {
	s := newNCA2(t, []string{"p0", "p1"})
	admit(t, s, "p0", "a", 10)
	s.mu.Lock()
	// a re-admission at its old pod (A4's preference): a merge at the head, not a second entry
	s.nca2LineSet(s.groups["p0"], "a", 12*64, epoch.Add(time.Second), pushoutRoom)
	s.mu.Unlock()
	if got := entriesOf(s, "a"); got != 1 {
		t.Fatalf("a re-admission at the same pod must merge: %d entries", got)
	}
	s.mu.Lock()
	pos, _ := s.nca2PositionOn(s.groups["p0"], "a")
	s.mu.Unlock()
	if pos != 12*64 {
		t.Fatalf("the merged entry carries the new tokens: %d", pos)
	}
	// a move: still ONE entry, on the pod that served the request
	s.mu.Lock()
	s.nca2AtPod("a", "p1", 4*64, epoch.Add(2*time.Second))
	s.mu.Unlock()
	if got := entriesOf(s, "a"); got != 1 {
		t.Fatalf("a move must leave one entry across every pod: %d", got)
	}
	if got := lineOf(s, "p0"); len(got) != 0 {
		t.Fatalf("the old pod's line must no longer hold the thread: %v", got)
	}
}

// TestMoveTransfersTheAdmission: L3 — a request reaching another pod moves the entry and the home
// there; an admission TRANSFERS (it is never created by a move), and a move is counted.
func TestMoveTransfersTheAdmission(t *testing.T) {
	s := newNCA2(t, []string{"p0", "p1"})
	admit(t, s, "p0", "a", 10)
	s.mu.Lock()
	_, warm := s.nca2AtPod("a", "p1", 5*64, epoch.Add(time.Second))
	s.mu.Unlock()
	if warm {
		t.Fatal("the prediction is read BEFORE the write (a fresh entry is not warm)")
	}
	if got := lineOf(s, "p0"); len(got) != 0 {
		t.Fatalf("the entry left the pod it is not on: %v", got)
	}
	if got := lineOf(s, "p1"); len(got) != 1 || got[0] != "a" {
		t.Fatalf("the entry follows the data: %v", got)
	}
	s.mu.Lock()
	home, admitted := s.nca2.homes["a"], s.nca2.admitted["a"]
	pos, _ := s.nca2PositionOn(s.groups["p1"], "a")
	s.mu.Unlock()
	if home != "p1" || !admitted {
		t.Fatalf("the home follows the pod and the admission TRANSFERS: home=%q admitted=%v", home, admitted)
	}
	if pos != 320 {
		t.Fatalf("the entry at the new pod's head is the request's prompt: %d, want 320", pos)
	}
}
