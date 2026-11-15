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

// THE SESSION LAYER (session-layer contract Part B), one rule per test: the session id recorded from the
// header (S3-S5), the per-pass session facts (ST1/ST2), the ranking by group (R2), the new-session head
// that waits for its pod's returners (H8/H10), the seat order that spreads seats over sessions (L2-2),
// and the collapse to Part A when no request carries the header (I10).

import (
	"strings"
	"testing"
	"time"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newSessionFixture is the door fixture with the session layer on.
func newSessionFixture(t *testing.T, pods ...string) *fixture {
	t.Helper()
	return newDoorFixture(t, func(c *Config) {
		on := true
		header := "x-llm-d-session-id"
		c.SessionAware, c.SessionHeader = &on, &header
	}, pods...)
}

// inSession is a head of the given session.
func inSession(h Head, session string) Head {
	h.Session = session
	return h
}

// sessionKeyOf reads a thread's session key under the lock.
func (f *fixture) sessionKeyOf(id string) sessKey {
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	return f.s.sessionKey(id)
}

// TestS4_FirstSessionValueWinsAndMismatchCounted: S4/E6 — the first value a thread carries is its
// session for good; a later different value is counted and ignored.
func TestS4_FirstSessionValueWinsAndMismatchCounted(t *testing.T) {
	f := newSessionFixture(t, "p0")
	f.pass(inSession(f.heavyAt("t", 5, 320, f.now), "sess-a"))
	f.atPod("t", "p0")
	f.done("t", "p0")
	h := inSession(f.heavyAt("t", 6, 384, f.now), "sess-b")
	h.ReqID = "r-t2"
	f.pass(h)
	if got := f.sessionKeyOf("t"); got != (sessKey{id: "sess-a"}) {
		t.Fatalf("S4: the first value is kept, got %+v", got)
	}
	f.s.mu.Lock()
	mismatches := f.s.nca2.door.sessions.mismatches
	f.s.mu.Unlock()
	if mismatches != 1 {
		t.Fatalf("E6: one dispatched request carried another value, mismatches %d", mismatches)
	}
}

// TestS5_NoHeaderIsItsOwnSession: S5 — a thread whose requests carry no header is its own session.
func TestS5_NoHeaderIsItsOwnSession(t *testing.T) {
	f := newSessionFixture(t, "p0")
	f.pass(f.heavyAt("lone", 5, 320, f.now))
	if got := f.sessionKeyOf("lone"); got != (sessKey{id: "lone", own: true}) {
		t.Fatalf("S5: got %+v", got)
	}
}

// TestS3_TheConfiguredHeaderIsRead: S3 — the request hook reads the configured header (lower-case), and
// nothing when the session layer is off.
func TestS3_TheConfiguredHeaderIsRead(t *testing.T) {
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-session-id": "root-1"}}
	if got := sessionOfRequest(req, "x-llm-d-session-id"); got != "root-1" {
		t.Fatalf("S3: got %q", got)
	}
	if got := sessionOfRequest(req, ""); got != "" {
		t.Fatalf("S3: no header configured reads nothing, got %q", got)
	}
}

// sessionState builds ST1's input: session S is inside (t1 admitted and running on p0, t2 an outsider,
// t3 a returner), session P was inside (p1 pushed out, waiting), session N is new (n1 waiting).
func sessionState(t *testing.T) *fixture {
	t.Helper()
	f := newSessionFixture(t, "p0", "p1")
	f.running("p0", "t1", 10, 320)
	f.returnerOn("p0", "t3")
	f.priorityOn("p1", "p1", f.now.Add(-time.Hour))
	f.s.mu.Lock()
	for id, sess := range map[string]string{"t1": "S", "t2": "S", "t3": "S", "p1": "P", "n1": "N"} {
		f.s.sessionNote(id, sess)
	}
	f.s.waiters["t2"] = &waiter{id: "t2", since: f.now}
	f.s.waiters["n1"] = &waiter{id: "n1", since: f.now.Add(-2 * time.Hour)}
	f.s.nca2Derive(f.now)
	f.s.sessionBeginPass()
	f.s.mu.Unlock()
	return f
}

// TestST1_SessionFactsAreDerivedPerPass: ST1 — admitted, everIn, active and seats per session, from the
// live state.
func TestST1_SessionFactsAreDerivedPerPass(t *testing.T) {
	f := sessionState(t)
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	sf := f.s.nca2.door.sessions.facts
	s := sf[sessKey{id: "S"}]
	if s.admitted != 2 || s.active != 3 || s.seats != 1 {
		t.Fatalf("S: admitted t1+t3, active t1 (in flight) + t2 + t3 (waiting), one seat: %+v", s)
	}
	if p := sf[sessKey{id: "P"}]; p.admitted != 0 || p.active != 1 || p.seats != 0 {
		t.Fatalf("P: %+v", p)
	}
	if !f.s.sessionEverIn(sessKey{id: "P"}) || f.s.sessionEverIn(sessKey{id: "N"}) {
		t.Fatal("everIn: P had a thread admitted, N never")
	}
}

// TestST2_GroupsAndR2Ranking: ST2/R2 — group A (inside) before B (was inside) before C (new), whatever the
// queue position, and v2's priority/newcomer split is not used inside the groups.
func TestST2_GroupsAndR2Ranking(t *testing.T) {
	f := sessionState(t)
	f.s.mu.Lock()
	f.s.doorBeginPass()
	got := make([]string, 0, len(f.s.nca2.door.rank))
	for _, c := range f.s.nca2.door.rank {
		got = append(got, c.w.id)
	}
	labels := []string{f.s.doorClassLabel("t2"), f.s.doorClassLabel("p1"), f.s.doorClassLabel("n1")}
	f.s.mu.Unlock()
	if strings.Join(got, ",") != "t2,p1,n1" {
		t.Fatalf("R2: A then B then C, got %v", got)
	}
	if strings.Join(labels, ",") != "A,B,C" {
		t.Fatalf("ST2: %v", labels)
	}
}

// TestE7_FewestActiveSessionFirst: R2/E7 — inside group A, the session with the fewest active threads
// goes first: one session's 87 waiting sub-agents rank behind a session with one; no cap.
func TestE7_FewestActiveSessionFirst(t *testing.T) {
	f := newSessionFixture(t, "p0")
	f.running("p0", "rootBig", 10, 320)
	f.running("p0", "rootSmall", 10, 320)
	f.s.mu.Lock()
	f.s.sessionNote("rootBig", "big")
	f.s.sessionNote("rootSmall", "small")
	for i := 0; i < 87; i++ {
		id := "sub" + itoa(i)
		f.s.sessionNote(id, "big")
		f.s.waiters[id] = &waiter{id: id, since: f.now.Add(-time.Hour + time.Duration(i)*time.Second)}
	}
	f.s.sessionNote("late", "small")
	f.s.waiters["late"] = &waiter{id: "late", since: f.now}
	f.s.nca2Derive(f.now)
	f.s.doorBeginPass()
	rank := f.s.nca2.door.rank
	f.s.mu.Unlock()
	if len(rank) != 88 || rank[0].w.id != "late" {
		t.Fatalf("E7: the session with fewer active threads first: %s first of %d", rank[0].w.id, len(rank))
	}
	if rank[1].w.id != "sub0" || rank[87].w.id != "sub86" {
		t.Fatalf("E7: the big session's sub-agents by queue position: %s .. %s", rank[1].w.id, rank[87].w.id)
	}
}

// TestH3_InsideSessionTakesTheHeadFromANewSession: H3 in Part B (the design's B2 example, step 3): a
// group-A thread that cannot backfill takes the head from a group-C head and keeps its collection when
// its own pod is the same.
func TestH3_InsideSessionTakesTheHeadFromANewSession(t *testing.T) {
	f := newSessionFixture(t, "p0")
	f.idleSeat("p0", "a1", 400)
	f.running("p0", "a2", 545, 320)
	f.s.mu.Lock()
	f.s.sessionNote("a2", "in")
	f.s.mu.Unlock()
	f.wait(10 * time.Second)
	n := inSession(f.heavyAt("N", 900, 320, f.now), "new")
	f.pass(n)
	if head, pod := f.doorHead(); head != "N" || pod != "p0" || f.held("p0") == 0 {
		t.Fatalf("fixture: the new session's head collects on p0 (%q on %q, held %d)", head, pod, f.held("p0"))
	}
	c2 := inSession(f.heavyAt("c2", 600, 320, f.now), "in")
	f.pass(n, c2)
	if head, pod := f.doorHead(); head != "c2" || pod != "p0" || f.reservedHead("p0") != "c2" {
		t.Fatalf("H3: the inside session's thread takes the head and the blocks (%q on %q for %q)", head, pod, f.reservedHead("p0"))
	}
}

// h10Fixture: a group-C head that has to collect on p0 (an idle reclaimable seat there), and a returner of
// p0 that needs `need` blocks.
func h10Fixture(t *testing.T, sessions bool, need int) (*fixture, []Head) {
	t.Helper()
	var f *fixture
	if sessions {
		f = newSessionFixture(t, "p0")
	} else {
		f = newDoorFixture(t, nil, "p0")
	}
	f.idleSeat("p0", "idle", 400)
	f.running("p0", "busy", 545, 320) // 5 free
	f.returnerOn("p0", "ret")
	f.wait(10 * time.Second)
	return f, []Head{inSession(f.heavyAt("N", 900, 320, f.now), "new"), f.heavyAt("ret", need, 320, f.now.Add(-time.Minute))}
}

// TestH10_NewSessionHeadNeverTakesItsPodsReturnersSeats: H8/H10/I8/T-DR4 — a returner of the collection
// pod that the pod can seat (by its own reclaim) is seated first; one that cannot fit keeps the new
// session's head from releasing anything at all.
func TestH10_NewSessionHeadNeverTakesItsPodsReturnersSeats(t *testing.T) {
	f, heads := h10Fixture(t, true, 300)
	f.pass(heads...)
	if !seated(f.s, "ret") || seated(f.s, "idle") {
		t.Fatal("H10: the returner reclaims the idle seat for itself first")
	}
	if f.held("p0") != 0 {
		t.Fatalf("I8: the new session's head took nothing a returner waited for, held %d", f.held("p0"))
	}
	// the returner cannot fit (it needs more than the pod can ever free now): nothing is released
	f, heads = h10Fixture(t, true, 2000)
	f.pass(heads...)
	if !seated(f.s, "idle") || f.exitsOn("p0") != 0 || f.held("p0") != 0 {
		t.Fatal("I8/T-DR4: while a returner of its pod waits, a new session's head releases no seat")
	}
}

// TestH10_PartAHeadCollectsFirst is the contrast: in Part A the head collects before the returners.
func TestH10_PartAHeadCollectsFirst(t *testing.T) {
	f, heads := h10Fixture(t, false, 300)
	f.pass(heads...)
	if seated(f.s, "idle") || f.held("p0") == 0 || seated(f.s, "ret") {
		t.Fatal("Part A: the head releases the idle seat and holds it; the returner waits")
	}
}

// TestL2_2_FewestSeatsFirstThenDeepest: L2-2/I9 (the design's B3 example) — pod p0 has one freed seat's
// worth of budget; r1 (session S1, which holds 3 seats; the deeper entry) and r2 (session S2, holding
// none) wait. Part B seats r2; Part A (v2's order) seats r1.
func TestL2_2_FewestSeatsFirstThenDeepest(t *testing.T) {
	for _, sessions := range []bool{true, false} {
		var f *fixture
		if sessions {
			f = newSessionFixture(t, "p0")
		} else {
			f = newDoorFixture(t, nil, "p0")
		}
		for i := 0; i < 3; i++ {
			f.running("p0", "s1-"+itoa(i), 200, 320)
		}
		f.running("p0", "other", 240, 320) // 110 free: room for one 100-block returner
		f.returnerOn("p0", "r2")           // shallower
		f.returnerOn("p0", "r1")           // deeper
		f.s.mu.Lock()
		for _, id := range []string{"s1-0", "s1-1", "s1-2", "r1"} {
			f.s.sessionNote(id, "S1")
		}
		f.s.sessionNote("r2", "S2")
		f.s.sessionNote("other", "S3")
		f.s.mu.Unlock()
		f.pass(f.heavyAt("r1", 100, 320, f.now.Add(-time.Minute)), f.heavyAt("r2", 100, 320, f.now.Add(-time.Minute)))
		want, other := "r2", "r1"
		if !sessions {
			want, other = "r1", "r2"
		}
		if !seated(f.s, want) || seated(f.s, other) {
			t.Fatalf("sessions=%v: %s must take the seat (r1 seated %v, r2 seated %v)", sessions, want, seated(f.s, "r1"), seated(f.s, "r2"))
		}
	}
}

// TestE8_TheHeadsGroupChangesWhileItIsHead: E8 — the head's session gets a thread admitted (it backfills)
// while the head collects: the next pass re-ranks it as group A, it stays the head (H2), and its
// collection no longer waits for returners.
func TestE8_TheHeadsGroupChangesWhileItIsHead(t *testing.T) {
	f, heads := h10Fixture(t, true, 2000)
	f.pass(heads...)
	if f.exitsOn("p0") != 0 {
		t.Fatal("fixture: the group-C head waits for the returner")
	}
	// a small thread of the head's session backfills: the session is inside now
	small := inSession(f.heavyAt("sib", 2, 64, f.now), "new")
	f.pass(append(heads, small)...)
	if !f.admitted("sib") {
		t.Fatal("fixture: the sibling backfills")
	}
	f.pass(append(heads, small)...)
	f.s.mu.Lock()
	class := f.s.doorClassOf("N")
	f.s.mu.Unlock()
	if head, _ := f.doorHead(); head != "N" || class != 0 {
		t.Fatalf("E8: re-ranked as group A and still the head (%q, class %d)", head, class)
	}
	if seated(f.s, "idle") || f.held("p0") == 0 {
		t.Fatal("E8: a group-A head collects before the returners (H8 no longer applies)")
	}
}

// TestI10_NoHeadersCollapsesToPartA: I10/E5 — with no session header anywhere every thread is its own
// session: R2's groups collapse to v2's classes (B = priority, C = newcomer; group A is empty because an
// outsider's one-thread session is not inside) and L2-2's key collapses to v2's (a returner's one-thread
// session holds no seat). The decision stream of a loop in which no returner waits is identical to Part
// A's; where a returner of a new head's collection pod waits, H10 is the one documented difference
// (TestH10_PartAHeadCollectsFirst vs TestH10_NewSessionHeadNeverTakesItsPodsReturnersSeats).
func TestI10_NoHeadersCollapsesToPartA(t *testing.T) {
	trace := func(sessions bool) []string {
		var out []string
		tune := func(c *Config) {
			c.Quantum = 1000
			c.IdleSeconds = 1000 // no seat is ever released by tenure or silence: no returner waits
			if sessions {
				on := true
				header := "x-llm-d-session-id"
				c.SessionAware, c.SessionHeader = &on, &header
			}
		}
		s, probe, _ := doorLoopTraced(t, 20261115, 72_000, 400, tune, false, &out)
		if len(probe.violations) > 0 {
			t.Fatalf("sessions=%v: %v", sessions, probe.violations)
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		return out
	}
	a, b := trace(false), trace(true)
	if len(a) != len(b) {
		t.Fatalf("I10: %d vs %d trace lines", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("I10: the streams differ at line %d:\nPart A: %s\nPart B: %s", i, a[i], b[i])
		}
	}
}
