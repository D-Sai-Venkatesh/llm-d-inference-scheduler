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

// THE DOOR FIX (session-layer contract Part A), one rule per test: the ranking (R1), the sticky head and
// its takeover (H1-H3), the head's fit-now on every pod before anything is released (H4/H5), ONE
// collection pod chosen once and kept while the CPU room holds (H6/H7), the collection itself (H9 with
// LOGIC Amendment 1 Q7's held rule), the returners (L2-1) and the backfill (BF1-BF3). The scenario tests
// are the review's dry runs (T-DR1..3, T-DR5), with v2 as the contrast where the contract asks for it.

import (
	"testing"
	"time"
)

// newDoorFixture is nca2 with doorBackfill on, a CPU tier large enough that only a test that wants the
// room to bind sees it bind, and one 950-block budget per pod (cold cap 3200 tokens).
func newDoorFixture(t *testing.T, tune func(*Config), pods ...string) *fixture {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		on := true
		c.DoorBackfill = &on
		c.CPUTierTokens = i64(1_000_000)
		if tune != nil {
			tune(c)
		}
	}, pods)
	return &fixture{t: t, s: s, now: epoch}
}

// newV2Twin is the same fixture with the switch off: the contrast the contract asks for.
func newV2Twin(t *testing.T, tune func(*Config), pods ...string) *fixture {
	t.Helper()
	s := newNCA2Tuned(t, func(c *Config) {
		c.CPUTierTokens = i64(1_000_000)
		if tune != nil {
			tune(c)
		}
	}, pods)
	return &fixture{t: t, s: s, now: epoch}
}

// idleSeat seats a thread that ran one turn and finished it: after idleSeconds of silence the rotation
// accepts it as a victim.
func (f *fixture) idleSeat(pod, id string, blocks int) {
	f.t.Helper()
	f.running(pod, id, blocks, 320)
	f.s.FirstChunk("rq-"+id, f.now)
	f.s.Settle("rq-"+id, pod, f.now)
}

// returnerOn makes an admitted thread of pod waiting without a seat (A2.2's warm returner). Its head must
// be offered every pass (returnerHead).
func (f *fixture) returnerOn(pod, id string) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	f.s.nca2.homes[id] = pod
	f.s.nca2.admitted[id] = true
	f.s.nca2.ever[id] = true
	f.s.groups[pod].nca2.line = append(f.s.groups[pod].nca2.line, nca2Entry{thread: id, tokens: 320})
	f.s.waiters[id] = &waiter{id: id, since: f.now.Add(-time.Minute)}
}

// priorityOn makes a pushed-out thread whose old entry is still on pod (A4 prefers that pod for it).
func (f *fixture) priorityOn(pod, id string, since time.Time) {
	f.t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	f.s.nca2.homes[id] = pod
	f.s.nca2.ever[id] = true
	f.s.nca2.pushed[id] = true
	f.s.groups[pod].nca2.line = append(f.s.groups[pod].nca2.line, nca2Entry{thread: id, tokens: 320})
	f.s.waiters[id] = &waiter{id: id, since: since}
}

// heavyAt is a head with a fixed arrival (the queue position survives the passes).
func (f *fixture) heavyAt(id string, blocks, tokens int, since time.Time) Head {
	h := f.hdHeavy(id, blocks, tokens)
	h.Arrival = since
	return h
}

// pass runs one pass a second after the last, with the given heads.
func (f *fixture) pass(heads ...Head) int {
	f.t.Helper()
	f.now = f.now.Add(time.Second)
	return f.s.Pass(heads, f.now)
}

func (f *fixture) held(pod string) int64 {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	if g := f.s.groups[pod]; g != nil && g.nca2 != nil {
		return g.nca2.reservedBlocks
	}
	return 0
}

func (f *fixture) reservedHead(pod string) string {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	if g := f.s.groups[pod]; g != nil && g.nca2 != nil {
		return g.nca2.reservedHead
	}
	return ""
}

func (f *fixture) doorHead() (head, pod string) {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.nca2.door.headID, f.s.nca2.door.collectPod
}

func (f *fixture) free(pod string) int64 {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.freeBudget(f.s.groups[pod])
}

// exitsOn is every departure booked on one pod so far.
func (f *fixture) exitsOn(pod string) int64 {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	var n int64
	if g := f.s.groups[pod]; g != nil {
		for _, v := range g.counters.exits {
			n += v
		}
	}
	return n
}

// doorPathOf is the door path of the decision that seated the thread: on its seat until its first
// dispatch consumes it, on that request's record afterwards.
func (f *fixture) doorPathOf(id string) string {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	if g := f.s.groupOf(id); g != nil {
		if m := g.threads[id]; m != nil && m.seatObs != nil {
			return m.seatObs.nca2.doorPath
		}
	}
	if rec := f.s.reqs["r-"+id]; rec != nil {
		return rec.nca2.doorPath
	}
	return ""
}

// TestR1_RankIsPriorityThenNewcomerThenSinceThenID: Part A's ranking is v2's outside order, as one
// sorted slice per pass (R3).
func TestR1_RankIsPriorityThenNewcomerThenSinceThenID(t *testing.T) {
	f := newDoorFixture(t, nil, "p0")
	f.s.mu.Lock()
	for id, c := range map[string]struct {
		ever bool
		ago  time.Duration
	}{
		"n-old": {false, 9 * time.Second}, "n-young": {false, time.Second},
		"p-young": {true, 2 * time.Second}, "p-old-b": {true, 5 * time.Second}, "p-old-a": {true, 5 * time.Second},
	} {
		f.s.nca2.ever[id] = c.ever
		f.s.waiters[id] = &waiter{id: id, since: f.now.Add(-c.ago)}
	}
	f.s.nca2.admitted["ret"], f.s.nca2.ever["ret"] = true, true
	f.s.waiters["ret"] = &waiter{id: "ret", since: f.now.Add(-time.Hour)} // a returner, never an outsider
	f.s.doorBeginPass()
	var got []string
	for _, c := range f.s.nca2.door.rank {
		got = append(got, c.w.id)
	}
	f.s.mu.Unlock()
	want := []string{"p-old-a", "p-old-b", "p-young", "n-old", "n-young"}
	if len(got) != len(want) {
		t.Fatalf("rank %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rank %v, want %v", got, want)
		}
	}
}

// collectingFixture is T-DR1's state (units of 50 blocks, budget 19 units per pod): pod A holds an idle
// seat a1 (8 units) and a running a2 (10.9), so 0.1 unit is free; pod B an idle b1 (8) and a running b2
// (8), 3 free. The head N needs 18 units: it fits no pod now. rA / rB are returners of A / B needing 6.
func collectingFixture(t *testing.T, door bool) (*fixture, func() []Head) {
	t.Helper()
	var f *fixture
	if door {
		f = newDoorFixture(t, nil, "p0", "p1")
	} else {
		f = newV2Twin(t, nil, "p0", "p1")
	}
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.idleSeat("p1", "b1", 400)
	f.running("p1", "b2", 400, 320)
	f.returnerOn("p0", "rA")
	f.returnerOn("p1", "rB")
	f.wait(10 * time.Second) // a1 and b1 are silent past idleSeconds: reclaimable
	since := f.now
	heads := func() []Head {
		return []Head{f.heavyAt("N", 900, 320, since.Add(-5*time.Second)),
			f.heavyAt("rA", 300, 320, since.Add(-time.Minute)), f.heavyAt("rB", 300, 320, since.Add(-time.Minute))}
	}
	return f, heads
}

// TestTDR1_HeadCollectsOnOnePodTheOtherPodKeepsSeating: DR1 — the head collects on ONE pod (A), B's
// returner is seated from B's own reclaimable seat, and a small newcomer backfills on B at once.
func TestTDR1_HeadCollectsOnOnePodTheOtherPodKeepsSeating(t *testing.T) {
	f, heads := collectingFixture(t, true)
	small := f.heavyAt("s", 15, 320, f.now)
	f.pass(append(heads(), small)...)
	head, pod := f.doorHead()
	if head != "N" || pod != "p0" {
		t.Fatalf("the head collects on ONE pod: head %q on %q", head, pod)
	}
	if f.held("p0") != 400 || f.held("p1") != 0 {
		t.Fatalf("H9/I2: held p0=%d p1=%d (want 400 on p0 only)", f.held("p0"), f.held("p1"))
	}
	if seated(f.s, "a1") || !seated(f.s, "a2") {
		t.Fatal("A's idle seat is released for the head, its running seat is not")
	}
	if !seated(f.s, "rB") {
		t.Fatal("L2-1: B's returner is seated (B is not the collection pod)")
	}
	if seated(f.s, "rA") {
		t.Fatal("A's returner may not eat the head's held blocks")
	}
	if !f.admitted("s") || f.homeOf("s") != "p1" {
		t.Fatalf("BF1: the small newcomer backfills on B now (home %q)", f.homeOf("s"))
	}
	if got := f.doorPathOf("s"); got != doorPathBackfill {
		t.Fatalf("the small newcomer's path: %q", got)
	}
}

// TestTDR1_V2TwinCollectsEverywhere is the contrast: v2 collects on EVERY CPU-fitting pod, so B's
// returner waits for the head too and the small newcomer waits behind it.
func TestTDR1_V2TwinCollectsEverywhere(t *testing.T) {
	f, heads := collectingFixture(t, false)
	f.pass(append(heads(), f.heavyAt("s", 15, 320, f.now))...)
	if seated(f.s, "b1") || f.held("p1") == 0 {
		t.Fatal("v2 releases B's idle seat for the head and holds it")
	}
	if seated(f.s, "rB") {
		t.Fatal("v2: B's returner waits behind the head's reservation")
	}
	if f.admitted("s") {
		t.Fatal("v2: strict head-of-line, the small newcomer waits")
	}
}

// TestH4_HeadFitNowOnAnyPodBeforeAnyRelease: H4/H5/I6 — the head is tried on EVERY pod without releasing
// anything first. p0 comes first in the pod order and could take it only by releasing its idle seat; p1
// takes it from free budget now. v2 (F5) releases on p0.
func TestH4_HeadFitNowOnAnyPodBeforeAnyRelease(t *testing.T) {
	for _, door := range []bool{true, false} {
		var f *fixture
		if door {
			f = newDoorFixture(t, nil, "p0", "p1")
		} else {
			f = newV2Twin(t, nil, "p0", "p1")
		}
		f.idleSeat("p0", "idle", 900)
		f.running("p1", "busy", 10, 320) // p1: one request in flight, so p0 is first in the A5 order
		f.wait(10 * time.Second)
		f.pass(f.heavyAt("N", 600, 320, f.now))
		if door {
			if f.homeOf("N") != "p1" || !seated(f.s, "idle") || f.exitsOn("p0") != 0 {
				t.Fatalf("door: the head enters p1 from free budget and nothing is released (home %q, p0 exits %d)",
					f.homeOf("N"), f.exitsOn("p0"))
			}
			if got := f.doorPathOf("N"); got != doorPathHeadFit {
				t.Fatalf("door path %q", got)
			}
		} else if f.homeOf("N") != "p0" || seated(f.s, "idle") {
			t.Fatalf("v2: the head releases p0's idle seat (F5), home %q", f.homeOf("N"))
		}
	}
}

// TestH1_HeadGoneReleasesTheCollection: H1/E1/I11 — the head leaves the queue mid-collection: every pod's
// held blocks are released and the next outsider becomes the head.
func TestH1_HeadGoneReleasesTheCollection(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	if f.held("p0") == 0 {
		t.Fatal("fixture: the head is collecting on p0")
	}
	// N is cancelled: its head is no longer offered; a newcomer M (too big to backfill) waits
	rest := heads()[1:]
	f.pass(append(rest, f.heavyAt("M", 940, 320, f.now))...)
	head, _ := f.doorHead()
	if head != "M" {
		t.Fatalf("H1: the next outsider is the head, got %q", head)
	}
	if f.reservedHead("p0") == "N" {
		t.Fatal("I11: nothing is held for a head that is gone")
	}
}

// TestH2_HeadIsStickyWithinItsClass: H2 — an outsider of the SAME class that now ranks first (it is
// older) does not take the head, and the collection stays.
func TestH2_HeadIsStickyWithinItsClass(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	held := f.held("p0")
	older := f.heavyAt("O", 940, 320, f.now.Add(-time.Hour)) // a newcomer older than N, too big to backfill
	f.pass(append(heads(), older)...)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" {
		t.Fatalf("H2: the head stays (%q on %q)", head, pod)
	}
	if f.held("p0") < held {
		t.Fatalf("the collection stays: held %d -> %d", held, f.held("p0"))
	}
}

// TestTDR3_TakeoverToAnotherPodReleases: H3/DR3 — a PRIORITY outsider that cannot backfill takes the
// head from a newcomer; its own pod (A4: its old entry is on p1) is not the collection pod, so p0's held
// blocks are released and p0's returner is seated again.
func TestTDR3_TakeoverToAnotherPodReleases(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	if f.held("p0") == 0 || seated(f.s, "rA") {
		t.Fatal("fixture: N collects on p0 and rA waits")
	}
	f.priorityOn("p1", "P", f.now.Add(-time.Hour))
	f.pass(append(heads(), f.heavyAt("P", 900, 320, f.now.Add(-time.Hour)))...)
	head, pod := f.doorHead()
	if head != "P" {
		t.Fatalf("H3: the priority outsider takes the head, got %q", head)
	}
	if pod != "p1" || f.held("p0") != 0 {
		t.Fatalf("H3: a different pod releases the collection (collect %q, p0 held %d)", pod, f.held("p0"))
	}
	if !seated(f.s, "rA") {
		t.Fatal("DR3: the released blocks belong to p0's returner again")
	}
}

// TestE3_TakeoverOnTheSamePodKeepsTheBlocks: H3/E3 — the taker's own pod is the collection pod: the held
// blocks are kept, now for the new head.
func TestE3_TakeoverOnTheSamePodKeepsTheBlocks(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	held := f.held("p0")
	f.priorityOn("p0", "P", f.now.Add(-time.Hour))
	f.pass(append(heads(), f.heavyAt("P", 900, 320, f.now.Add(-time.Hour)))...)
	head, pod := f.doorHead()
	if head != "P" || pod != "p0" {
		t.Fatalf("H3: P takes the head on the same pod (%q on %q)", head, pod)
	}
	if f.held("p0") < held || f.reservedHead("p0") != "P" {
		t.Fatalf("E3: the blocks are kept for P (held %d -> %d, for %q)", held, f.held("p0"), f.reservedHead("p0"))
	}
}

// TestH3_AHigherOutsiderThatCanBackfillDoesNotTakeOver: H3 (LOGIC Amendment 1 Q3) — a priority outsider
// that fits somewhere by backfill is admitted, and the newcomer head keeps collecting.
func TestH3_AHigherOutsiderThatCanBackfillDoesNotTakeOver(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	f.priorityOn("p1", "P", f.now.Add(-time.Hour))
	f.pass(append(heads(), f.heavyAt("P", 20, 320, f.now.Add(-time.Hour)))...)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" {
		t.Fatalf("the head stays (%q on %q)", head, pod)
	}
	if !f.admitted("P") || f.doorPathOf("P") != doorPathBackfill {
		t.Fatalf("P backfills (admitted %v, path %q)", f.admitted("P"), f.doorPathOf("P"))
	}
}

// TestH7_CollectionReleasedWhenTheRoomIsLost: H7/E2 — the collection pod stops passing the CPU check
// for the head: its collection is released and the next pod is chosen.
func TestH7_CollectionReleasedWhenTheRoomIsLost(t *testing.T) {
	f := newDoorFixture(t, func(c *Config) { c.CPUTierTokens = i64(20_000) }, "p0", "p1")
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.running("p1", "b2", 545, 320)
	f.running("p1", "b3", 5, 320) // p1 is busier: p0 comes first
	f.wait(10 * time.Second)
	head := f.heavyAt("N", 900, 3000, f.now)
	f.pass(head)
	if _, pod := f.doorHead(); pod != "p0" || f.held("p0") == 0 {
		t.Fatal("fixture: the head collects on p0")
	}
	// a deep admitted entry of p0 becomes ACTIVE (its head waits): the room falls below the head's 3000
	f.s.mu.Lock()
	f.s.nca2.homes["deep"], f.s.nca2.admitted["deep"], f.s.nca2.ever["deep"] = "p0", true, true
	f.s.groups["p0"].nca2.line = append(f.s.groups["p0"].nca2.line, nca2Entry{thread: "deep", tokens: 17_000})
	f.s.mu.Unlock()
	f.pass(head, f.heavyAt("deep", 2000, 320, f.now))
	if got := f.roomOn("p0"); got >= 3000 {
		t.Fatalf("fixture: the room on p0 must be below 3000, got %d", got)
	}
	if f.held("p0") != 0 {
		t.Fatalf("H7: the collection on p0 is released, held %d", f.held("p0"))
	}
	if _, pod := f.doorHead(); pod != "p1" {
		t.Fatalf("H7: the next pod is chosen, got %q", pod)
	}
}

// TestH7_ValveRefusalKeepsTheCollection: LOGIC Amendment 1 Q1 — the valve refusing the head on its
// collection pod is transient: the held blocks are kept and nothing is released in that decision.
func TestH7_ValveRefusalKeepsTheCollection(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	held := f.held("p0")
	exits := f.exitsOn("p0")
	// a cold prefill fills p0's valve: the head (320 cold tokens) no longer passes V there
	// the pass re-derives coldInFlight from the seats, so the fill is applied inside the decision
	f.s.mu.Lock()
	f.s.nca2.door.onStage = func(st doorStage) {
		if st.name == doorStageStart {
			f.s.groups["p0"].coldInFlight = f.s.groups["p0"].coldCapTokens(f.s.cfg.GroupBudgetFraction, f.s.blockSize)
		}
	}
	f.s.mu.Unlock()
	f.idleSeat("p0", "late", 3) // a seat that will be reclaimable: the valve must still stop the release
	f.wait(10 * time.Second)
	f.pass(heads()...)
	if _, pod := f.doorHead(); pod != "p0" || f.held("p0") != held {
		t.Fatalf("Q1: the collection stays on p0 with its blocks (pod %q, held %d -> %d)", pod, held, f.held("p0"))
	}
	if f.exitsOn("p0") != exits {
		t.Fatal("Q1: nothing is released while the valve refuses the head")
	}
}

// TestTDR2_ValveHeadroomSendsBackfillElsewhere: DR2/BF1/I5 — on the collection pod, a small cold outsider
// may not use the valve headroom the head needs; it backfills on the other pod.
func TestTDR2_ValveHeadroomSendsBackfillElsewhere(t *testing.T) {
	f := newDoorFixture(t, nil, "p0", "p1")
	// p0: a running seat whose 2000-token cold charge is outstanding, and an idle 400-block seat
	f.s.mu.Lock()
	g := f.s.groups["p0"]
	w := &waiter{id: "cold", since: f.now, chains: []Chain{{Keys: keys(400, seedFor("cold")), Tokens: 320}}}
	f.s.seat(g, w, 2000, f.now)
	f.s.nca2.homes["cold"], f.s.nca2.admitted["cold"], f.s.nca2.ever["cold"] = "p0", true, true
	f.s.nca2LineSet(g, "cold", 320, f.now, pushoutRoom)
	f.s.release(g.threads["cold"], "rq-cold", w.chains, f.now, dispatchObservables{})
	f.s.mu.Unlock()
	f.s.AtPod("rq-cold", "p0", f.now)
	f.idleSeat("p0", "a1", 400)
	for i := 0; i < 2; i++ { // p1 is busier (p0 first in every pod order) and cannot take the head
		f.running("p1", "b"+string(rune('0'+i)), 300, 320)
	}
	f.wait(10 * time.Second)
	head := f.heavyAt("N", 900, 1000, f.now) // 1000 cold tokens: 2000 + 1000 <= 3200
	f.pass(head)
	if _, pod := f.doorHead(); pod != "p0" {
		t.Fatal("fixture: the head collects on p0")
	}
	f.s.mu.Lock()
	cold, capTokens := f.s.groups["p0"].coldInFlight, f.s.groups["p0"].coldCapTokens(f.s.cfg.GroupBudgetFraction, f.s.blockSize)
	f.s.mu.Unlock()
	if cold+300 > capTokens || cold+300+1000 <= capTokens {
		t.Fatalf("fixture: the base valve allows the small one (%d+300 <= %d) and the headroom does not", cold, capTokens)
	}
	held := f.held("p0")
	f.pass(head, f.heavyAt("s", 5, 300, f.now))
	if f.homeOf("s") != "p1" {
		t.Fatalf("DR2: the small cold outsider backfills on p1, got %q", f.homeOf("s"))
	}
	if f.held("p0") != held {
		t.Fatalf("I5: the head's held blocks are untouched (%d -> %d)", held, f.held("p0"))
	}
}

// TestBF1_BackfillNeverReleasesASeat: BF1/I4 — an outsider that would fit only by reclaiming a seat is not
// admitted by backfill, and nothing is released for it.
func TestBF1_BackfillNeverReleasesASeat(t *testing.T) {
	f := newDoorFixture(t, nil, "p0")
	f.idleSeat("p0", "idle", 900) // reclaimable, 50 blocks free
	f.wait(10 * time.Second)
	f.priorityOn("p0", "H", f.now.Add(-time.Hour)) // the head: its prompt fits no room, it never collects
	head := f.heavyAt("H", 10, 2_000_000, f.now.Add(-time.Hour))
	f.pass(head, f.heavyAt("s", 100, 320, f.now))
	if f.admitted("s") || !seated(f.s, "idle") || f.exitsOn("p0") != 0 {
		t.Fatal("I4: backfill took a seat it could only have by a release")
	}
	f.pass(head, f.heavyAt("s2", 40, 320, f.now))
	if !f.admitted("s2") || f.doorPathOf("s2") != doorPathBackfill {
		t.Fatal("a small one that fits the free 50 blocks backfills")
	}
}

// TestBF1_BackfillNeverSpendsTheHeldBlocks: BF1/I5 (GPU half) — on the collection pod a backfill
// candidate sees only free − held.
func TestBF1_BackfillNeverSpendsTheHeldBlocks(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...) // p0: free 405, held 400
	f.pass(append(heads(), f.heavyAt("mid", 300, 320, f.now))...)
	if f.homeOf("mid") == "p0" {
		t.Fatal("I5: backfill spent the blocks held for the head")
	}
	if f.held("p0") != 400 {
		t.Fatalf("I5: held %d", f.held("p0"))
	}
}

// TestBF2_PrefilterIsSound: LOGIC Amendment 1 Q2 — a thread whose prompt is mostly a prefix the pod
// already holds costs only its own blocks; a size-based prefilter would skip it, the sound bound does not.
func TestBF2_PrefilterIsSound(t *testing.T) {
	f := newDoorFixture(t, nil, "p0")
	f.running("p0", "parent", 400, 320)
	f.running("p0", "filler", 540, 320) // 10 blocks free
	f.priorityOn("p0", "H", f.now.Add(-time.Hour))
	head := f.heavyAt("H", 10, 2_000_000, f.now.Add(-time.Hour))
	sub := Head{Thread: "sub", ReqID: "r-sub", Arrival: f.now, Chains: chainWithPrefix("parent", 400, "sub", 10)}
	f.pass(head, sub)
	if !f.admitted("sub") || f.doorPathOf("sub") != doorPathBackfill {
		t.Fatal("BF2: the sub-agent whose parent's prefix is resident costs 10 blocks and backfills")
	}
}

// TestBF3_NothingFitsEndsThePass: BF3 — no admission and no seat: the decision is false and the pass ends.
func TestBF3_NothingFitsEndsThePass(t *testing.T) {
	f := newDoorFixture(t, nil, "p0")
	f.running("p0", "busy", 940, 320)
	if i := f.pass(f.heavyAt("x", 100, 320, f.now), f.heavyAt("y", 100, 320, f.now)); i != -1 {
		t.Fatalf("nothing fits: nothing served, got %d", i)
	}
	if snap := f.s.Snapshot(f.now).NCA2; snap.DecisionsThisPass != 0 || snap.DecisionCapHit {
		t.Fatalf("BF3: the first decision ends the pass: %+v", snap)
	}
}

// TestE4_RuleOHeadOnEmptyPod: E4/U4 — an oversize head is admitted by an EMPTY pod (Rule O) at H4.
func TestE4_RuleOHeadOnEmptyPod(t *testing.T) {
	f := newDoorFixture(t, nil, "p0", "p1")
	f.running("p0", "busy", 10, 320)
	f.pass(f.heavyAt("huge", 1200, 320, f.now))
	if f.homeOf("huge") != "p1" || !seated(f.s, "huge") {
		t.Fatalf("Rule O: the empty pod admits the oversize head, home %q", f.homeOf("huge"))
	}
}

// TestE9_EngineRemovalReleasesTheCollection: E9 — the collection pod goes away: the collection is
// released with it and the head collects elsewhere.
func TestE9_EngineRemovalReleasesTheCollection(t *testing.T) {
	f, heads := collectingFixture(t, true)
	f.pass(heads()...)
	f.s.RemoveEngine("p0", f.now)
	if _, pod := f.doorHead(); pod != "" {
		t.Fatalf("E9: the collection pod is gone, got %q", pod)
	}
	f.pass(heads()...)
	if _, pod := f.doorHead(); pod != "p1" {
		t.Fatalf("E9: the head collects on the surviving pod, got %q", pod)
	}
}

// TestTDR5_HeldAndHeadBudgetNeverDecrease: DR5 and LOGIC Amendment 1 Q7 — a stream of small arrivals
// cannot take anything back from the collecting head: on its pod both the held blocks and the budget
// the head can spend only grow until it enters.
func TestTDR5_HeldAndHeadBudgetNeverDecrease(t *testing.T) {
	f := newDoorFixture(t, nil, "p0", "p1")
	sizes := []int{238, 238, 237, 237} // p0 exactly full
	for i, b := range sizes {
		f.running("p0", "s"+string(rune('0'+i)), b, 320)
	}
	for i := 0; i < 5; i++ { // p1: busier (p0 first), 450 blocks free, nothing reclaimable
		f.running("p1", "q"+string(rune('0'+i)), 100, 320)
	}
	since := f.now
	var heldPrev, freePrev int64
	entered := false
	for step := 0; step < 40 && !entered; step++ {
		if step%5 == 0 && step/5 < len(sizes) { // one of p0's seats finishes its turn every 5 passes
			id := "s" + string(rune('0'+step/5))
			f.s.FirstChunk("rq-"+id, f.now)
			f.s.Settle("rq-"+id, "p0", f.now)
		}
		small := f.heavyAt("x"+itoa(step), 2, 64, f.now)
		f.pass(f.heavyAt("N", 900, 320, since), small)
		if f.admitted("N") {
			entered = true
			break
		}
		held, free := f.held("p0"), f.free("p0")
		if held < heldPrev || free < freePrev {
			t.Fatalf("step %d: held %d -> %d, the head's budget %d -> %d: the head lost ground", step, heldPrev, held, freePrev, free)
		}
		heldPrev, freePrev = held, free
		if f.homeOf("x"+itoa(step)) == "p0" {
			t.Fatalf("step %d: a small arrival took budget on the collection pod", step)
		}
	}
	if !entered || f.homeOf("N") != "p0" {
		t.Fatalf("DR5: the head enters on its collection pod in bounded time (entered %v, home %q)", entered, f.homeOf("N"))
	}
}
