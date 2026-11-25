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

// THE ROOM GATE and the group's OWN tier index: a head seats only when its engine's CPU tier can
// absorb its write without evicting a seated block; the seat records its chain; dispatch touches it.

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// roomSched is one engine of 1000 GPU blocks with a 2000-block CPU tier (seat budget 1900), the group's
// own tier index on, and (gate) the room gate.
func roomSched(t *testing.T, gate bool, mut ...func(*Config)) *Scheduler {
	t.Helper()
	cfg := defaultConfig()
	cfg.TierCapacityTokens = 2000 * 64
	cfg.OwnTierIndex = true
	cfg.RoomGate = gate
	for _, f := range mut {
		f(&cfg)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("", 64, 1000); err != nil {
		t.Fatal(err)
	}
	return s
}

// runTurn seats (if needed) and runs one turn of id to its end, at time at.
func runTurn(t *testing.T, s *Scheduler, h Head, at time.Time) {
	t.Helper()
	if i := s.Pass([]Head{h}, at); i != 0 {
		t.Fatalf("%s must dispatch, got %d", h.ReqID, i)
	}
	s.AtPod(h.ReqID, "", at)
	s.FirstChunk(h.ReqID, at)
	s.Settle(h.ReqID, "", at)
}

func tierOf(s *Scheduler) *engineGroup {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groups[""]
}

func evict(t *testing.T, s *Scheduler, id string) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groupOf(id)
	if g == nil {
		t.Fatalf("%s is not seated", id)
	}
	s.exit(g, g.threads[id], "test", s.now)
}

func TestRoomConfigIsValidated(t *testing.T) {
	cases := map[string]func(*Config){
		"ownTierIndex without a tier":    func(c *Config) { c.OwnTierIndex = true },
		"roomGate without ownTierIndex":  func(c *Config) { c.TierCapacityTokens = 64000; c.RoomGate = true },
		"ownTierIndex with the producer": func(c *Config) { c.TierCapacityTokens = 64000; c.OwnTierIndex = true; c.CPUResidencyProducer = "x" },
		"negative reserve":               func(c *Config) { c.RoomReserveTokens = -1 },
		"roomGate with a zero hold": func(c *Config) {
			c.TierCapacityTokens = 64000
			c.OwnTierIndex = true
			c.RoomGate = true
			c.RoomHoldMaxSeconds = 0
		},
	}
	for name, f := range cases {
		cfg := defaultConfig()
		f(&cfg)
		if _, err := NewScheduler(cfg); err == nil {
			t.Errorf("%s must be refused", name)
		}
	}
	if !defaultConfig().NewcomersFirst || defaultConfig().RoomGate || defaultConfig().OwnTierIndex {
		t.Fatal("the defaults must be today's policy: newcomers first, no own index, no room gate")
	}
}

// The own index follows the engine: a dispatch touches the chain, placement reads it.
func TestOwnTierIndexIsTouchedAtDispatch(t *testing.T) {
	s := roomSched(t, false)
	a := headN("a", 0, time.Second, chain("a", 300))
	runTurn(t, s, a, epoch)
	g := tierOf(s)
	if got := g.tier.ResidentPrefix(a.Chains[0].Keys); got != 300 {
		t.Fatalf("the dispatched chain must be in the index, got %d of 300", got)
	}
	snap := s.Snapshot(epoch)
	if snap.Room == nil || snap.Room.TierWrites[tierCauseDispatch] != 1 || snap.Room.TierWrites[tierCauseSeat] != 0 {
		t.Fatalf("one dispatch write and (gate off) no seat write, got %+v", snap.Room)
	}
	if e := snap.Engines[""]; e.TierIndexBlocks != 300 || e.TierIndexCapBlocks != 2000 {
		t.Fatalf("index 300 of 2000 blocks, got %d / %d", e.TierIndexBlocks, e.TierIndexCapBlocks)
	}
	if s.residency == nil {
		t.Fatal("the own index must be placement's residency read")
	}
}

// The valve is off under the gate: no cold charge is ever taken.
func TestRoomGateTurnsTheValveOff(t *testing.T) {
	s := roomSched(t, true)
	for _, id := range []string{"a", "b", "c"} {
		if i := s.Pass([]Head{headN(id, 0, time.Second, chain(id, 300))}, epoch); i != 0 {
			t.Fatalf("%s must dispatch, got %d", id, i)
		}
	}
	if e := engine(s, ""); e.ColdInFlight != 0 {
		t.Fatalf("no cold charge with the valve off, got %d", e.ColdInFlight)
	}
}

// The seat RECORDS the chain: the next head in the SAME pass sees the room it took. Here outside
// content sits ABOVE a seated member (it is newer), so the room is only the free capacity.
func TestSeatRecordIsTheLedger(t *testing.T) {
	s := roomSched(t, true)
	runTurn(t, s, headN("a", 0, time.Second, chain("a", 600)), epoch) // oldest, stays seated
	runTurn(t, s, headN("x", 0, time.Second, chain("x", 600)), epoch) // newer than a
	evict(t, s, "x")                                                  // x's blocks stay in the tier, ABOVE a
	// free = 2000 - 1200 = 800; the tail is a (seated): room = 800
	b := headN("b", 0, 2*time.Second, chain("b", 500))
	c := headN("c", 0, time.Second, chain("c", 400))
	s.Pass([]Head{b, c}, epoch)
	if !seated(s, "b") {
		t.Fatal("b (500 <= room 800) must seat")
	}
	if seated(s, "c") {
		t.Fatal("c must be HELD: b's seat recorded 500 blocks, room is now 300 < 400 — the same pass must see it")
	}
	if snap := s.Snapshot(epoch); snap.Room.Held != 1 || snap.Room.TierWrites[tierCauseSeat] != 3 {
		t.Fatalf("one hold, three seat records (a, x, b), got %+v", snap.Room)
	}
}

// A room-held head may rotate an eligible member out when that member's blocks make room.
func TestRoomHeldHeadRotatesAMemberThatMakesRoom(t *testing.T) {
	s := roomSched(t, true)
	runTurn(t, s, headN("a", 0, time.Second, chain("a", 600)), epoch)
	runTurn(t, s, headN("x", 0, time.Second, chain("x", 600)), epoch)
	evict(t, s, "x")
	// a is silent for > idleSeconds: rotation-eligible (idle); its blocks are the tail
	later := epoch.Add(time.Duration(s.cfg.IdleSeconds+1) * time.Second)
	b := headN("b", 0, time.Second, chain("b", 900)) // budget 600+900 <= 1900, room 800 < 900
	s.Pass([]Head{b}, later)
	if seated(s, "a") {
		t.Fatal("a (idle, at the tail) must be rotated out to make room")
	}
	if !seated(s, "b") {
		t.Fatal("with a gone the room is 2000 - 1200 + a's 600: b must seat in the same pass")
	}
	if snap := s.Snapshot(later); snap.Room.Rotations != 1 {
		t.Fatalf("one rotation for room, got %+v", snap.Room)
	}
}

// THE GUARD: a member whose blocks lie ABOVE a busy member at the tail makes no room — nobody leaves.
func TestRoomRotationGuardRefusesAUselessExit(t *testing.T) {
	s := roomSched(t, true, func(c *Config) { c.RoomReserveTokens = 700 * 64 })
	// d is RUNNING (in flight) and its blocks are the oldest: the tail blocker
	d := headN("d", 0, 3*time.Second, chain("d", 400))
	if i := s.Pass([]Head{d}, epoch); i != 0 {
		t.Fatal("d must dispatch")
	}
	s.AtPod(d.ReqID, "", epoch)
	runTurn(t, s, headN("e", 0, time.Second, chain("e", 400)), epoch) // newer than d; settles, will be idle
	later := epoch.Add(time.Duration(s.cfg.IdleSeconds+1) * time.Second)
	// budget 800 + 600 <= 1900; need 600 + reserve 700 = 1300 > free 1200, and d (staying) stops the walk
	s.Pass([]Head{headN("b", 0, time.Second, chain("b", 600))}, later)
	if !seated(s, "e") {
		t.Fatal("e's blocks are above the busy d at the tail: rotating e makes no room, so e must stay")
	}
	if seated(s, "b") {
		t.Fatal("b must stay held")
	}
	if snap := s.Snapshot(later); snap.Room.Rotations != 0 || snap.Room.Held != 1 {
		t.Fatalf("no rotation, one hold, got %+v", snap.Room)
	}
}

// The last resort: a head held roomHoldMaxSeconds seats anyway (and the forced seat is counted).
func TestRoomHoldMaxIsTheLastResort(t *testing.T) {
	s := roomSched(t, true, func(c *Config) { c.RoomHoldMaxSeconds = 30 })
	d := headN("d", 0, 3*time.Second, chain("d", 400))
	if i := s.Pass([]Head{d}, epoch); i != 0 {
		t.Fatal("d must dispatch")
	}
	s.AtPod(d.ReqID, "", epoch) // running, the tail blocker: no rotation can help
	runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), epoch)
	evict(t, s, "x")                                  // x's 500 blocks ABOVE d: free 1100, room 1100 (d stops the walk)
	b := headN("b", 0, time.Second, chain("b", 1200)) // budget 400 + 1200 <= 1900; room 1100 < 1200
	s.Pass([]Head{b}, epoch)
	if seated(s, "b") {
		t.Fatal("b must be held (room 1100 < 1200)")
	}
	s.Pass([]Head{b}, epoch.Add(29*time.Second))
	if seated(s, "b") {
		t.Fatal("29 s < roomHoldMaxSeconds: still held")
	}
	s.Pass([]Head{b}, epoch.Add(31*time.Second))
	if !seated(s, "b") {
		t.Fatal("held 31 s >= 30 s: the last resort seats it")
	}
	snap := s.Snapshot(epoch)
	if snap.Room.Forced != 1 || snap.Room.Held != 1 {
		t.Fatalf("one hold, one forced seat, got %+v", snap.Room)
	}
	// its record (1200 into 1100 free) evicted 100 blocks of d's (seated) tail: booked by cause
	if got := snap.Room.SeatedEvicted[tierCauseSeat]; got != 100 {
		t.Fatalf("the forced seat's record must have evicted 100 seated blocks of d, got %+v", snap.Room.SeatedEvicted)
	}
}

// The first dispatch after a seat that recorded its chain reports the read taken BEFORE the record.
func TestTierPredictionOfASeatRecordedTurn(t *testing.T) {
	s := roomSched(t, true)
	a := headN("a", 0, time.Second, chain("a", 300))
	s.Pass([]Head{a}, epoch)
	s.mu.Lock()
	m := s.groupOf("a").threads["a"]
	pending := m.seatTierPending
	s.mu.Unlock()
	if pending {
		t.Fatal("the seat's prediction must have been consumed by the dispatch in the same pass")
	}
	// a newcomer that holds nothing: its seat read 0, so its first turn predicted 0 — not 300
	if got := m.seatTierResident; got != 0 {
		t.Fatalf("a cold newcomer's pre-record resident is 0, got %d", got)
	}
}

// victimOldestDispatch: among eligible members the one whose chain the tier touched longest ago leaves.
func TestVictimOldestDispatch(t *testing.T) {
	s := roomSched(t, false, func(c *Config) { c.VictimOldestDispatch = true })
	runTurn(t, s, headN("old", 0, time.Second, chain("old", 100)), epoch)
	runTurn(t, s, headN("new", 0, time.Second, chain("new", 100)), epoch.Add(10*time.Second))
	later := epoch.Add(time.Duration(s.cfg.IdleSeconds+20) * time.Second) // both idle
	s.mu.Lock()
	_, v := s.rotationVictim(s.groups[""], later)
	s.mu.Unlock()
	if v == nil || v.id != "old" {
		t.Fatalf("the oldest dispatch must be the victim, got %v", v)
	}
}

// newcomersFirst false: plain oldest-first for every head.
func TestNewcomersFirstOff(t *testing.T) {
	s := roomSched(t, false, func(c *Config) { c.NewcomersFirst = false })
	s.mu.Lock()
	s.waiters["ret"] = &waiter{id: "ret", since: epoch.Add(-10 * time.Second), returner: true}
	s.waiters["new"] = &waiter{id: "new", since: epoch.Add(-time.Second)}
	h := s.headWaiter()
	s.mu.Unlock()
	if h.id != "ret" {
		t.Fatalf("oldest first regardless of class, got %s", h.id)
	}
	d := roomSched(t, false)
	d.mu.Lock()
	d.waiters["ret"] = &waiter{id: "ret", since: epoch.Add(-10 * time.Second), returner: true}
	d.waiters["new"] = &waiter{id: "new", since: epoch.Add(-time.Second)}
	h = d.headWaiter()
	d.mu.Unlock()
	if h.id != "new" {
		t.Fatalf("the default stays newcomers first, got %s", h.id)
	}
}

// The replay workload with the gate: it runs to the end (liveness), the run side stays exact, the gate
// fires, and the index's own count of SEATED blocks evicted is far below the same workload without it.
func TestRoomGateReplay(t *testing.T) {
	seatedEvicted := func(gate bool, seed int64, tierBlocks int64) (int64, Snapshot) {
		cfg := defaultConfig()
		cfg.TierCapacityTokens = tierBlocks * 64
		cfg.OwnTierIndex = true
		cfg.RoomGate = gate
		cfg.VictimOldestDispatch = gate
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		var bad error
		driveReplay(t, s, seed, 500, &out, replayOpts{noMoves: true}, func(step int) {
			if bad == nil {
				bad = runSideExact(s)
			}
		})
		if bad != nil {
			t.Fatalf("gate=%v seed %d: %v", gate, seed, bad)
		}
		snap := s.Snapshot(epoch)
		var n int64
		for _, v := range snap.Room.SeatedEvicted {
			n += v
		}
		return n, snap
	}
	held := int64(0)
	for _, tc := range []struct {
		seed int64
		tier int64
	}{{21, 2000}, {22, 2000}, {23, 1000}, {24, 1000}, {25, 700}, {26, 700}} {
		seed := tc.seed
		off, offSnap := seatedEvicted(false, seed, tc.tier)
		on, onSnap := seatedEvicted(true, seed, tc.tier)
		held += onSnap.Room.Held
		if onSnap.Room.TierWrites[tierCauseSeat] == 0 {
			t.Fatalf("seed %d: the gate must record seats", seed)
		}
		if onSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: the gate must not starve the workload: served %d vs %d", seed, onSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: seated blocks evicted %d (gate off) -> %d (gate on); held %d forced %d rotations %d; served %d -> %d",
			seed, tc.tier, off, on, onSnap.Room.Held, onSnap.Room.Forced, onSnap.Room.Rotations, offSnap.Served, onSnap.Served)
		if on > off {
			t.Errorf("seed %d: the gate must not evict MORE seated blocks than no gate (%d > %d)", seed, on, off)
		}
	}
	if held == 0 {
		t.Fatal("the small tiers must make the room gate hold somebody")
	}
}

// With newcomersFirst false a rotation serves the head that asked for it: the victim (quantum-complete,
// its next turn queued since BEFORE the head arrived) re-queues behind the head instead of re-seating
// at once with a fresh quantum. (Found by the verifier; reproduced with the gate on and off.)
func TestRotationVictimRequeuesBehindTheHead(t *testing.T) {
	for _, gate := range []bool{false, true} {
		s := roomSched(t, gate, func(c *Config) { c.NewcomersFirst = false; c.VictimOldestDispatch = true })
		at := func(sec int) time.Time { return epoch.Add(time.Duration(sec) * time.Second) }
		hd := func(id string, n int, arr time.Time, c []Chain) Head {
			return Head{Thread: id, ReqID: fmt.Sprintf("%s#%d", id, n), Arrival: arr, Chains: c, Returner: n > 0}
		}
		turn := func(h Head, now time.Time) {
			if i := s.Pass([]Head{h}, now); i != 0 {
				t.Fatalf("%s must dispatch, got %d", h.ReqID, i)
			}
			s.AtPod(h.ReqID, "", now)
			s.FirstChunk(h.ReqID, now)
			s.Settle(h.ReqID, "", now)
		}
		for i := 0; i < 8; i++ { // m completes its quantum (8 busy periods)
			turn(hd("m", i, at(i), chain("m", 300)), at(i))
		}
		turn(hd("f", 0, at(20), chain("f", 500)), at(20))
		r := hd("r", 0, at(21), chain("r", 800))
		if i := s.Pass([]Head{r}, at(21)); i != 0 {
			t.Fatal("r must dispatch")
		}
		s.AtPod(r.ReqID, "", at(21))
		s.FirstChunk(r.ReqID, at(21))             // r RUNNING: 800 of the 950-block run budget
		mh := hd("m", 8, at(20), chain("m", 300)) // m's next turn, queued at 20, GPU-held behind r
		h := hd("h", 0, at(24), chain("h", 400))  // the head: 1600 + 400 > the 1900-block seat budget
		s.Pass([]Head{mh, h}, at(25))
		if !seated(s, "h") {
			t.Fatalf("gate=%v: the head that triggered the rotation must be seated", gate)
		}
		if seated(s, "m") {
			t.Fatalf("gate=%v: the victim must wait behind the head, not re-seat with a fresh quantum", gate)
		}
	}
}
