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

// LOGIC AMENDMENT 2, one rule per test. Backfill reclaim (BF4, I4 amended): an outsider that fits no pod's
// free budget may release eligible seats on a pod that is not the collection pod and has no returner
// waiting, and only for its own admission in the same step (T-R1..T-R4). The group-C yield (H10b): a new
// session's head holds nothing on its pod while a returner of that pod waits (T-R5).

import (
	"strings"
	"testing"
	"time"
)

// withReclaim turns backfillReclaim on in a door fixture's config.
func withReclaim(tune func(*Config)) func(*Config) {
	return func(c *Config) {
		on := true
		c.BackfillReclaim = &on
		if tune != nil {
			tune(c)
		}
	}
}

// rA1Fixture is R-A1: the priority head N (its old entry is on p0, so A4 puts p0 first for it) collects on
// p0, where one seat is running; every seat of p1 is held by an idle thread and no returner of p1 waits.
// The small newcomer s fits neither p1's free budget (50 blocks) nor p0's free budget less the held blocks.
func rA1Fixture(t *testing.T, reclaim bool) (*fixture, []Head) {
	t.Helper()
	var tune func(*Config)
	if reclaim {
		tune = withReclaim(nil)
	}
	f := newDoorFixture(t, tune, "p0", "p1")
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	for _, id := range []string{"b1", "b2", "b3"} {
		f.idleSeat("p1", id, 300)
	}
	f.priorityOn("p0", "N", f.now.Add(-time.Hour))
	f.wait(10 * time.Second)
	return f, []Head{f.heavyAt("N", 900, 320, f.now.Add(-time.Hour)), f.heavyAt("s", 100, 320, f.now)}
}

// TestTR1_BackfillReclaimsIdleSeatsOffTheCollectionPod: T-R1/BF4 — with the switch on, p1's eligible seats
// are released (one, in the rotationVictim order) and s is admitted on p1 by path backfill_reclaim; the
// head's collection on p0 is untouched.
func TestTR1_BackfillReclaimsIdleSeatsOffTheCollectionPod(t *testing.T) {
	f, heads := rA1Fixture(t, true)
	f.pass(heads...)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" || f.held("p0") != 405 {
		t.Fatalf("fixture: N collects on p0 (%q on %q, held %d)", head, pod, f.held("p0"))
	}
	if !f.admitted("s") || f.homeOf("s") != "p1" {
		t.Fatalf("BF4: s is admitted on p1 (admitted %v, home %q)", f.admitted("s"), f.homeOf("s"))
	}
	if got := f.doorPathOf("s"); got != doorPathBackfillReclaim {
		t.Fatalf("BF4: the door path is %q", got)
	}
	if f.exitsOn("p1") != 1 || seated(f.s, "b1") || !seated(f.s, "b2") || !seated(f.s, "b3") {
		t.Fatalf("BF4: exactly the first rotation victim is released on p1 (exits %d)", f.exitsOn("p1"))
	}
}

// TestTR1_OffTwinWaits is T-R1's contrast: with the switch off, s waits and p1 releases nothing.
func TestTR1_OffTwinWaits(t *testing.T) {
	f, heads := rA1Fixture(t, false)
	f.pass(heads...)
	if f.admitted("s") || f.exitsOn("p1") != 0 {
		t.Fatalf("off: s waits and p1 keeps its idle seats (admitted %v, p1 exits %d)", f.admitted("s"), f.exitsOn("p1"))
	}
}

// TestTR2_NoReclaimOnAPodWithAWaitingReturner: T-R2/BF4(b) — a returner of p1 is waiting (one that p1
// cannot seat even by reclaim): backfill releases nothing on p1, inside threads go first.
func TestTR2_NoReclaimOnAPodWithAWaitingReturner(t *testing.T) {
	f, heads := rA1Fixture(t, true)
	f.returnerOn("p1", "rB")
	f.pass(append(heads, f.heavyAt("rB", 2000, 320, f.now.Add(-time.Minute)))...)
	if f.admitted("s") || f.exitsOn("p1") != 0 {
		t.Fatalf("BF4(b): no release on p1 while its returner waits (admitted %v, p1 exits %d)", f.admitted("s"), f.exitsOn("p1"))
	}
}

// TestTR3_NeverOnTheCollectionPod: T-R3/BF4(a) — the collection pod still holds an eligible seat (the
// valve refuses the head there, so it releases nothing: Amendment 1 Q1), and s would fit there by
// reclaiming it: backfill does not, the freed budget there is the head's.
func TestTR3_NeverOnTheCollectionPod(t *testing.T) {
	f := newDoorFixture(t, withReclaim(nil), "p0")
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.wait(10 * time.Second)
	head := f.heavyAt("N", 900, 3000, f.now)
	f.pass(head)
	if _, pod := f.doorHead(); pod != "p0" || f.held("p0") == 0 {
		t.Fatal("fixture: N collects on p0")
	}
	f.idleSeat("p0", "late", 150)
	f.wait(10 * time.Second)
	// the valve has room for s (64 cold tokens) and none for the head (3000): the head releases nothing
	f.s.mu.Lock()
	f.s.nca2.door.onStage = func(st doorStage) {
		if st.name == doorStageStart {
			f.s.groups["p0"].coldInFlight = 500
		}
	}
	f.s.mu.Unlock()
	exits := f.exitsOn("p0")
	f.pass(head, f.heavyAt("s", 100, 64, f.now))
	if f.admitted("s") || !seated(f.s, "late") || f.exitsOn("p0") != exits {
		t.Fatalf("BF4(a): nothing is reclaimed on the collection pod (s admitted %v, late seated %v)", f.admitted("s"), seated(f.s, "late"))
	}
}

// TestTR4_NoReleaseWithoutTheAdmission: T-R4 — a candidate that does not fit p1 even with every eligible
// seat released there releases nothing.
func TestTR4_NoReleaseWithoutTheAdmission(t *testing.T) {
	f, heads := rA1Fixture(t, true)
	heads[1] = f.heavyAt("s", 960, 320, f.now) // p1: 50 free + 900 reclaimable < 960
	f.pass(heads...)
	if f.admitted("s") || f.exitsOn("p1") != 0 {
		t.Fatalf("T-R4: no release without the admission (admitted %v, p1 exits %d)", f.admitted("s"), f.exitsOn("p1"))
	}
}

// TestH3_ReclaimCountsAsBackfill: H3 with the switch on (LOGIC Amendment 2) — a priority outsider that
// fits nowhere as the budget is, but would be admitted by BF4, does not take the head: "can backfill"
// includes BF4. It is admitted by backfill reclaim and N keeps collecting. With the switch off it takes
// the head and is admitted after collecting on its own pod p1.
func TestH3_ReclaimCountsAsBackfill(t *testing.T) {
	for _, reclaim := range []bool{true, false} {
		var tune func(*Config)
		if reclaim {
			tune = withReclaim(nil)
		}
		f := newDoorFixture(t, tune, "p0", "p1")
		f.idleSeat("p0", "a1", 400)
		f.running("p0", "a2", 545, 320)
		for _, id := range []string{"b1", "b2", "b3"} {
			f.idleSeat("p1", id, 300)
		}
		f.running("p1", "b4", 5, 320) // p1 is busier: the newcomer head N's order starts at p0
		f.wait(10 * time.Second)
		n := f.heavyAt("N", 900, 320, f.now.Add(-time.Minute))
		f.pass(n)
		if head, pod := f.doorHead(); head != "N" || pod != "p0" || f.held("p0") != 405 {
			t.Fatalf("fixture: the newcomer N collects on p0 (%q on %q, held %d)", head, pod, f.held("p0"))
		}
		// P (600 blocks) fits no pod as it is (p0 410 free, p1 45), and p1 by reclaiming two idle seats
		f.priorityOn("p1", "P", f.now.Add(-time.Hour))
		f.pass(n, f.heavyAt("P", 600, 320, f.now.Add(-time.Hour)))
		head, pod := f.doorHead()
		if reclaim {
			if head != "N" || pod != "p0" || f.held("p0") != 405 || !f.admitted("P") || f.doorPathOf("P") != doorPathBackfillReclaim {
				t.Fatalf("reclaim on: P backfills by reclaim and N keeps collecting (head %q on %q, held %d, P admitted %v, path %q)",
					head, pod, f.held("p0"), f.admitted("P"), f.doorPathOf("P"))
			}
		} else if f.doorPathOf("P") != doorPathHeadCollected || f.homeOf("P") != "p1" {
			t.Fatalf("reclaim off: P takes the head and collects on p1 (path %q, home %q)", f.doorPathOf("P"), f.homeOf("P"))
		}
	}
}

// h10bFixture: the head N collects on p0 (one idle seat released, one running); in Part B N is a new
// session's head (group C), collecting through H10 because no returner of p0 waits yet.
func h10bFixture(t *testing.T, sessions bool) (*fixture, Head, *[]string) {
	t.Helper()
	var f *fixture
	if sessions {
		f = newSessionFixture(t, "p0")
	} else {
		f = newDoorFixture(t, nil, "p0")
	}
	var events []string
	f.s.onEvent = func(ev Event) {
		if ev.Kind == "COLLECT" {
			events = append(events, ev.ID+" "+ev.Detail)
		}
	}
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.wait(10 * time.Second)
	n := inSession(f.heavyAt("N", 900, 320, f.now), "new")
	f.pass(n)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" || f.held("p0") != 405 {
		t.Fatalf("fixture: N collects on p0 (%q on %q, held %d)", head, pod, f.held("p0"))
	}
	return f, n, &events
}

// TestTR5_NewSessionHeadYieldsToItsPodsReturner: T-R5/H10b — a returner of the collection pod starts
// waiting: the group-C head's held blocks are released (its collection pod and reservedHead are kept),
// and the yield is an O2 event. While the returner waits the head holds nothing (I8 extended); once it
// is gone, H10 collects again.
func TestTR5_NewSessionHeadYieldsToItsPodsReturner(t *testing.T) {
	f, n, events := h10bFixture(t, true)
	f.returnerOn("p0", "ret")
	ret := f.heavyAt("ret", 600, 320, f.now.Add(-time.Minute)) // 405 free, nothing reclaimable: it waits
	f.pass(n, ret)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" || f.reservedHead("p0") != "N" {
		t.Fatalf("H10b: the head and its collection pod are kept (%q on %q, reservedHead %q)", head, pod, f.reservedHead("p0"))
	}
	if f.held("p0") != 0 || seated(f.s, "ret") {
		t.Fatalf("H10b: nothing is held while p0's returner waits (held %d)", f.held("p0"))
	}
	if !strings.Contains(strings.Join(*events, "\n"), "N event=released_returner pod=p0 held=405") {
		t.Fatalf("O2/O8: the yield is a COLLECT event: %v", *events)
	}
	f.pass(n, ret) // still waiting: still nothing held
	if f.held("p0") != 0 {
		t.Fatalf("I8: held %d while the returner waits", f.held("p0"))
	}
	f.pass(n) // the returner's request is gone: H10 collects again
	if f.held("p0") != 405 || f.reservedHead("p0") != "N" {
		t.Fatalf("H10: the head collects again once no returner waits (held %d for %q)", f.held("p0"), f.reservedHead("p0"))
	}
}

// TestTR5_PartAHeadKeepsItsBlocks is the contrast: in Part A the head keeps head-first on its pod (D1).
func TestTR5_PartAHeadKeepsItsBlocks(t *testing.T) {
	f, n, events := h10bFixture(t, false)
	f.returnerOn("p0", "ret")
	f.pass(n, f.heavyAt("ret", 600, 320, f.now.Add(-time.Minute)))
	if f.held("p0") != 405 || strings.Contains(strings.Join(*events, "\n"), "released_returner") {
		t.Fatalf("Part A: the head keeps its held blocks (held %d)", f.held("p0"))
	}
}
