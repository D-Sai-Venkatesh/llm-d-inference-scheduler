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
		"negative hold":                  func(c *Config) { c.RoomHoldMaxSeconds = -1 },
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

// roomHoldMaxSeconds 0 = NO forced seat: the same head stays held however long, and the room opens
// only the way the design opens it — here the tail blocker d finishes, idles past idleSeconds, and the
// held head's room rotation moves it out.
func TestRoomHoldZeroNeverForces(t *testing.T) {
	s := roomSched(t, true, func(c *Config) { c.RoomHoldMaxSeconds = 0 })
	d := headN("d", 0, 3*time.Second, chain("d", 400))
	if i := s.Pass([]Head{d}, epoch); i != 0 {
		t.Fatal("d must dispatch")
	}
	s.AtPod(d.ReqID, "", epoch) // running, the tail blocker: no rotation can help yet
	runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), epoch)
	evict(t, s, "x")
	b := headN("b", 0, time.Second, chain("b", 1200)) // budget fits; room 1100 < 1200
	for _, sec := range []int{0, 31, 91, 600} {
		s.Pass([]Head{b}, epoch.Add(time.Duration(sec)*time.Second))
		if seated(s, "b") {
			t.Fatalf("held %d s with roomHoldMaxSeconds 0: no forced seat", sec)
		}
	}
	// d finishes at 600 s and idles past idleSeconds: the held head rotates it out (its blocks are the
	// tail) and seats into the room it leaves — without any forced seat
	s.FirstChunk(d.ReqID, epoch.Add(600*time.Second))
	s.Settle(d.ReqID, "", epoch.Add(600*time.Second))
	later := epoch.Add(time.Duration(600+s.cfg.IdleSeconds+1) * time.Second)
	s.Pass([]Head{b}, later)
	s.Pass([]Head{b}, later.Add(time.Millisecond))
	if !seated(s, "b") {
		t.Fatal("once the blocker idles out, the room rotation must open the room and seat b")
	}
	snap := s.Snapshot(later)
	if snap.Room.Forced != 0 || snap.Room.Rotations < 1 {
		t.Fatalf("no forced seat, at least one room rotation, got %+v", snap.Room)
	}
	if got := snap.Room.SeatedEvicted[tierCauseSeat]; got != 0 {
		t.Fatalf("b's seat must not evict a seated block, got %+v", snap.Room.SeatedEvicted)
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
	seatedEvicted := func(gate bool, seed int64, tierBlocks int64, hold ...float64) (int64, Snapshot) {
		cfg := defaultConfig()
		cfg.TierCapacityTokens = tierBlocks * 64
		cfg.OwnTierIndex = true
		cfg.RoomGate = gate
		cfg.VictimOldestDispatch = gate
		if len(hold) > 0 {
			cfg.RoomHoldMaxSeconds = hold[0]
		}
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
	seatedEvictedDisplace := func(seed int64, tierBlocks int64, qdLast ...bool) (int64, Snapshot) {
		cfg := defaultConfig()
		cfg.QuantumDoneLast = len(qdLast) > 0 && qdLast[0]
		if len(qdLast) > 1 { // (false, true) = newcomers-first + quantumLastTurn, no qdLast
			cfg.QuantumLastTurn = qdLast[1]
			cfg.NewcomersFirst = true
		}
		if len(qdLast) > 2 { // (false, false, true) = newcomers-first + roomVictims
			cfg.RoomVictims = qdLast[2]
		}
		if len(qdLast) > 3 { // (false, false, false, true) = newcomers-first + fairRotation
			cfg.FairRotation = qdLast[3]
		}
		if len(qdLast) > 4 { // (…, true, true) = + costFairRotation
			cfg.CostFairRotation = qdLast[4]
		}
		cfg.TierCapacityTokens = tierBlocks * 64
		cfg.OwnTierIndex = true
		cfg.RoomGate = true
		cfg.VictimOldestDispatch = true
		cfg.RoomDisplace = true
		cfg.RoomHoldMaxSeconds = 0
		if len(qdLast) > 5 && qdLast[5] { // (…, true, true, true) = UNGATED cost-fair (the CPU-warm group's successor)
			cfg.RoomGate = false
			cfg.RoomDisplace = false
		}
		if len(qdLast) > 6 && qdLast[6] { // (…, true) = + coldValveOff
			cfg.ColdValveOff = true
		}
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
			t.Fatalf("displace seed %d: %v", seed, bad)
		}
		snap := s.Snapshot(epoch)
		var n int64
		for c, v := range snap.Room.SeatedEvicted {
			if c != tierCauseDisplaced {
				n += v
			}
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
		// roomHoldMaxSeconds 0: no forced seat ever, no stall (the room rotation and the dispatch touch
		// open the room), and no more seated loss than with the last resort
		nf, nfSnap := seatedEvicted(true, seed, tc.tier, 0)
		if nfSnap.Room.Forced != 0 {
			t.Fatalf("seed %d: roomHoldMaxSeconds 0 must never force a seat, got %d", seed, nfSnap.Room.Forced)
		}
		if nfSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: no-forced-seat must not starve the workload: served %d vs %d", seed, nfSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: no forced seat: seated blocks evicted %d; held %d rotations %d; served %d",
			seed, tc.tier, nf, nfSnap.Room.Held, nfSnap.Room.Rotations, nfSnap.Served)
		dn, dnSnap := seatedEvictedDisplace(seed, tc.tier)
		if dnSnap.Room.Forced != 0 || dnSnap.Room.Rotations != 0 {
			t.Fatalf("seed %d: displace mode has no forced seat and no room rotation, got %+v", seed, dnSnap.Room)
		}
		if dnSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: displace mode must not starve the workload: served %d vs %d", seed, dnSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: displace: staying blocks evicted %d (seat %d, dispatch protected %d / eligible %d), displaced %d; held %d; served %d",
			seed, tc.tier, dn, dnSnap.Room.SeatedEvicted[tierCauseSeat], dnSnap.Room.SeatedEvicted[tierCauseDispatch], dnSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			dnSnap.Room.SeatedEvicted[tierCauseDisplaced], dnSnap.Room.Held, dnSnap.Served)
		qn, qnSnap := seatedEvictedDisplace(seed, tc.tier, true)
		if qnSnap.Room.Forced != 0 || qnSnap.Room.Rotations != 0 || qnSnap.Room.SeatedEvicted[tierCauseSeat] != 0 {
			t.Fatalf("seed %d: displace+quantumDoneLast: no forced seat, no room rotation, no seat-caused protected eviction, got %+v", seed, qnSnap.Room)
		}
		if qnSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: displace+quantumDoneLast must not starve the workload: served %d vs %d", seed, qnSnap.Served, offSnap.Served)
		}
		ln, lnSnap := seatedEvictedDisplace(seed, tc.tier, false, true)
		if lnSnap.Room.Forced != 0 || lnSnap.Room.Rotations != 0 || lnSnap.Room.SeatedEvicted[tierCauseSeat] != 0 {
			t.Fatalf("seed %d: NF+quantumLastTurn: no forced seat, no room rotation, no seat-caused protected eviction, got %+v", seed, lnSnap.Room)
		}
		if lnSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: NF+quantumLastTurn must not starve the workload: served %d vs %d", seed, lnSnap.Served, offSnap.Served)
		}
		vn, vnSnap := seatedEvictedDisplace(seed, tc.tier, false, false, true)
		if vnSnap.Room.Forced != 0 || vnSnap.Room.Rotations != 0 || vnSnap.Room.SeatedEvicted[tierCauseSeat] != 0 {
			t.Fatalf("seed %d: NF+roomVictims: no forced seat, no room rotation, no seat-caused protected eviction, got %+v", seed, vnSnap.Room)
		}
		if vnSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: NF+roomVictims must not starve the workload: served %d vs %d", seed, vnSnap.Served, offSnap.Served)
		}
		nfn, nfSnap := seatedEvictedDisplace(seed, tc.tier, false, false, false)
		t.Logf("seed %d tier %d: NF only: staying evicted %d, displaced %d; held %d; served %d | NF+roomVictims: staying evicted %d (dispatch protected %d / eligible %d), displaced %d; held %d; victims %d; served %d",
			seed, tc.tier, nfn, nfSnap.Room.SeatedEvicted[tierCauseDisplaced], nfSnap.Room.Held, nfSnap.Served,
			vn, vnSnap.Room.SeatedEvicted[tierCauseDispatch], vnSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			vnSnap.Room.SeatedEvicted[tierCauseDisplaced], vnSnap.Room.Held, vnSnap.Room.HeadVictims, vnSnap.Served)
		fr, frSnap := seatedEvictedDisplace(seed, tc.tier, false, false, false, true)
		if frSnap.Room.Forced != 0 || frSnap.Room.Rotations != 0 || frSnap.Room.SeatedEvicted[tierCauseSeat] != 0 {
			t.Fatalf("seed %d: NF+fairRotation: no forced seat, no room rotation, no seat-caused protected eviction, got %+v", seed, frSnap.Room)
		}
		if frSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: NF+fairRotation must not starve the workload: served %d vs %d", seed, frSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: NF+fairRotation: staying evicted %d (dispatch protected %d / eligible %d), displaced %d; held %d; head victims %d; exits %v; served %d",
			seed, tc.tier, fr, frSnap.Room.SeatedEvicted[tierCauseDispatch], frSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			frSnap.Room.SeatedEvicted[tierCauseDisplaced], frSnap.Room.Held, frSnap.Room.HeadVictims, frSnap.Exits, frSnap.Served)
		cf, cfSnap := seatedEvictedDisplace(seed, tc.tier, false, false, false, true, true)
		if cfSnap.Room.Forced != 0 || cfSnap.Room.Rotations != 0 || cfSnap.Room.SeatedEvicted[tierCauseSeat] != 0 {
			t.Fatalf("seed %d: costFairRotation: no forced seat, no room rotation, no seat-caused protected eviction, got %+v", seed, cfSnap.Room)
		}
		if cfSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: costFairRotation must not starve the workload: served %d vs %d", seed, cfSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: costFairRotation: staying evicted %d (dispatch protected %d / eligible %d), displaced %d; held %d; exits %v; served %d",
			seed, tc.tier, cf, cfSnap.Room.SeatedEvicted[tierCauseDispatch], cfSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			cfSnap.Room.SeatedEvicted[tierCauseDisplaced], cfSnap.Room.Held, cfSnap.Exits, cfSnap.Served)
		ug, ugSnap := seatedEvictedDisplace(seed, tc.tier, false, false, false, true, true, true)
		if ugSnap.Room.Forced != 0 || ugSnap.Room.Held != 0 {
			t.Fatalf("seed %d: ungated cost-fair never holds or forces, got %+v", seed, ugSnap.Room)
		}
		if ugSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: ungated cost-fair must not starve the workload: served %d vs %d", seed, ugSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: UNGATED cost-fair: staying evicted %d (seat %d / dispatch %d) vs gate-off %d; tail saved %d; exits %v; served %d vs %d",
			seed, tc.tier, ug, ugSnap.Room.SeatedEvicted[tierCauseSeat], ugSnap.Room.SeatedEvicted[tierCauseDispatch], off,
			ugSnap.Room.FairTailSaved, ugSnap.Exits, ugSnap.Served, offSnap.Served)
		uv, uvSnap := seatedEvictedDisplace(seed, tc.tier, false, false, false, true, true, true, true)
		if uvSnap.Room.Forced != 0 || uvSnap.Room.Held != 0 {
			t.Fatalf("seed %d: ungated cost-fair valve-off never holds or forces, got %+v", seed, uvSnap.Room)
		}
		if uvSnap.Served < offSnap.Served/2 {
			t.Fatalf("seed %d: ungated cost-fair valve-off must not starve the workload: served %d vs %d", seed, uvSnap.Served, offSnap.Served)
		}
		t.Logf("seed %d tier %d: UNGATED cost-fair VALVE OFF: staying evicted %d (dispatch %d) vs valve on %d; tail saved %d; exits %v; served %d",
			seed, tc.tier, uv, uvSnap.Room.SeatedEvicted[tierCauseDispatch], ug, uvSnap.Room.FairTailSaved, uvSnap.Exits, uvSnap.Served)
		t.Logf("seed %d tier %d: NF+qLastTurn: staying blocks evicted %d (dispatch protected %d / eligible %d), displaced %d; held %d; marks %d; served %d",
			seed, tc.tier, ln, lnSnap.Room.SeatedEvicted[tierCauseDispatch], lnSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			lnSnap.Room.SeatedEvicted[tierCauseDisplaced], lnSnap.Room.Held, lnSnap.Room.QuantumLastMarks, lnSnap.Served)
		t.Logf("seed %d tier %d: displace+qdLast: staying blocks evicted %d (dispatch protected %d / eligible %d), displaced %d; held %d; deferred %d; served %d",
			seed, tc.tier, qn, qnSnap.Room.SeatedEvicted[tierCauseDispatch], qnSnap.Room.SeatedEvicted[tierCauseDispatch+"-eligible"],
			qnSnap.Room.SeatedEvicted[tierCauseDisplaced], qnSnap.Room.Held, qnSnap.Room.QuantumDoneDeferred, qnSnap.Served)
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

// The floor holds ACROSS passes: a head that needs more than one pass (here its seat needs more than m's
// exit, and no other member is eligible yet) still goes before the victim it displaced, and the victim's
// wait is still measured from its true arrival.
func TestRotationVictimFloorHoldsAcrossPasses(t *testing.T) {
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
		for i := 0; i < 8; i++ {
			turn(hd("m", i, at(i), chain("m", 300)), at(i))
		}
		turn(hd("f", 0, at(20), chain("f", 500)), at(20))
		r := hd("r", 0, at(21), chain("r", 800))
		if i := s.Pass([]Head{r}, at(21)); i != 0 {
			t.Fatal("r must dispatch")
		}
		s.AtPod(r.ReqID, "", at(21))
		s.FirstChunk(r.ReqID, at(21))
		mh := hd("m", 8, at(20), chain("m", 300))
		h := hd("h", 0, at(24), chain("h", 700)) // 1600 - 300 + 700 = 2000 > 1900: m's exit is not enough
		s.Pass([]Head{mh, h}, at(25))
		s.Pass([]Head{mh, h}, at(25).Add(time.Millisecond))
		if seated(s, "m") {
			t.Fatalf("gate=%v: the next pass must not re-seat the victim ahead of the head (the floor holds)", gate)
		}
		s.mu.Lock()
		w := s.waiters["m"]
		s.mu.Unlock()
		if w == nil || !w.since.Equal(at(20)) || !entryKey(w).After(at(24)) {
			t.Fatalf("gate=%v: the victim keeps its true arrival (20 s) and queues behind the head (24 s): %+v", gate, w)
		}
	}
}

// roomDisplace: an ELIGIBLE member's tail blocks count as room — the head is admitted at once and the
// member whose blocks its write pushed out is rotated out ("displaced"); a NON-eligible (running)
// member's blocks stop the walk and are never touched; before the member turns eligible the head waits.
func TestRoomDisplaceAdmitsThenRemovesTheDisplaced(t *testing.T) {
	s := roomSched(t, true, func(c *Config) {
		c.RoomDisplace = true
		c.RoomHoldMaxSeconds = 0
		c.NewcomersFirst = false
		c.VictimOldestDispatch = true
	})
	runTurn(t, s, headN("e", 0, time.Second, chain("e", 400)), epoch) // e: done, the OLDEST blocks
	r := headN("r", 0, time.Second, chain("r", 300))
	if i := s.Pass([]Head{r}, epoch.Add(time.Second)); i != 0 {
		t.Fatal("r must dispatch")
	}
	s.AtPod(r.ReqID, "", epoch.Add(time.Second)) // r: running — never eligible, its blocks are protected
	runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), epoch.Add(2*time.Second))
	evict(t, s, "x") // x's 500 blocks: non-member, the NEWEST
	// tail -> head: e 400 | r 300 | x 500 | free 800. h writes 1000 (budget 400+300+1000 <= 1900).
	h := headN("h", 0, time.Second, chain("h", 1000))
	early := epoch.Add(5 * time.Second) // e not idle yet: room = free 800 (e is protected) < 1000
	s.Pass([]Head{h}, early)
	if seated(s, "h") || !seated(s, "e") {
		t.Fatal("before e is eligible, h must wait and e must stay")
	}
	late := epoch.Add(time.Duration(s.cfg.IdleSeconds+2) * time.Second) // e idle: room = 800 + e's 400
	s.Pass([]Head{h}, late)
	if !seated(s, "h") {
		t.Fatal("e's tail blocks count as room once it is eligible: h must be admitted")
	}
	if seated(s, "e") {
		t.Fatal("e's blocks were pushed out by h's write: e must be rotated out (displaced)")
	}
	if !seated(s, "r") {
		t.Fatal("r is running: it must stay")
	}
	snap := s.Snapshot(late)
	if got := snap.Room.SeatedEvicted[tierCauseSeat]; got != 0 {
		t.Fatalf("no protected (non-eligible) block may be evicted by the seat, got %+v", snap.Room.SeatedEvicted)
	}
	if got := snap.Room.SeatedEvicted[tierCauseDisplaced]; got != 200 {
		t.Fatalf("h's 1000 into 800 free pushes out 200 of e's blocks, got %+v", snap.Room.SeatedEvicted)
	}
	if snap.Room.Rotations != 0 || snap.Room.Forced != 0 {
		t.Fatalf("no room rotation and no forced seat in displace mode, got %+v", snap.Room)
	}
	// the early hold: the walk ended at e's block (not yet eligible, not in flight) -> between-turns
	if snap.Room.Held != 1 || snap.Room.Blockers[blockerBetweenTurns] != 1 {
		t.Fatalf("one hold, blocked by a between-turns member, got held %d blockers %+v", snap.Room.Held, snap.Room.Blockers)
	}
}

// The blocker diagnostic names a RUNNING member at the tail (and its quantum's last turn apart).
func TestRoomBlockerNamesARunningTail(t *testing.T) {
	for _, last := range []bool{false, true} {
		s := roomSched(t, true, func(c *Config) { c.RoomDisplace = true; c.RoomHoldMaxSeconds = 0; c.Quantum = 8 })
		at := epoch
		turns := 0
		if last {
			turns = 7 // seven completed busy periods: the next one is the quantum's last
		}
		for i := 0; i < turns; i++ {
			runTurn(t, s, headN("d", i, time.Second, chain("d", 400)), at)
			at = at.Add(time.Second)
		}
		d := headN("d", turns, time.Second, chain("d", 400))
		if i := s.Pass([]Head{d}, at); i != 0 {
			t.Fatal("d must dispatch")
		}
		s.AtPod(d.ReqID, "", at) // in flight: the tail blocker
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), at.Add(time.Second))
		evict(t, s, "x")
		b := headN("b", 0, time.Second, chain("b", 1200)) // room 1100 < 1200
		s.Pass([]Head{b}, at.Add(2*time.Second))
		if seated(s, "b") {
			t.Fatal("b must be held")
		}
		want := blockerRunning
		if last {
			want = blockerRunningLast
		}
		if got := s.Snapshot(at).Room.Blockers[want]; got != 1 {
			t.Fatalf("last=%v: the blocker must read %s, got %+v", last, want, s.Snapshot(at).Room.Blockers)
		}
	}
}

// quantumDoneLast: while a head waits OUTSIDE, a member within its quantum is served ahead of an OLDER
// turn of a quantum-complete member; with nobody waiting (or the switch off) plain oldest-first.
func TestQuantumDoneLastServesTheQuantumFirst(t *testing.T) {
	for _, tc := range []struct {
		name       string
		on, waiter bool
		want       string
	}{{"on, someone waiting", true, true, "r"}, {"on, nobody waiting", true, false, "q"}, {"off, someone waiting", false, true, "q"}} {
		s := roomSched(t, true, func(c *Config) {
			c.RoomDisplace = true
			c.RoomHoldMaxSeconds = 0
			c.Quantum = 4
			c.QuantumDoneLast = tc.on
		})
		// d: running, the OLDEST blocks — the tail blocker that keeps z waiting outside
		d := headN("d", 0, time.Second, chain("d", 400))
		if i := s.Pass([]Head{d}, epoch); i != 0 {
			t.Fatal("d must dispatch")
		}
		s.AtPod(d.ReqID, "", epoch)
		at := epoch.Add(time.Second)
		for i := 0; i < 4; i++ { // q completes its quantum (4 busy periods)
			runTurn(t, s, headN("q", i, time.Second, chain("q", 300)), at)
			at = at.Add(time.Second)
		}
		runTurn(t, s, headN("r", 0, time.Second, chain("r", 300)), at) // r: one turn, within its quantum
		at = at.Add(time.Second)
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), at)
		evict(t, s, "x") // free 500
		at = at.Add(time.Second)
		qh := Head{Thread: "q", ReqID: "q#4", Arrival: at, Chains: chain("q", 300), Returner: true}                  // OLDER
		rh := Head{Thread: "r", ReqID: "r#1", Arrival: at.Add(time.Second), Chains: chain("r", 300), Returner: true} // younger
		heads := []Head{qh, rh}
		if tc.waiter {
			// z: budget fits (400+300+300+800 <= 1900), room 500 (d stops the walk) < 800 -> waits outside
			heads = append(heads, headN("z", 0, time.Second, chain("z", 800)))
		}
		i := s.Pass(heads, at.Add(2*time.Second))
		if i < 0 || heads[i].Thread != tc.want {
			got := "-"
			if i >= 0 {
				got = heads[i].Thread
			}
			t.Fatalf("%s: want %s served, got %s", tc.name, tc.want, got)
		}
		if tc.waiter && seated(s, "z") {
			t.Fatalf("%s: z must still be waiting outside", tc.name)
		}
		deferred := s.Snapshot(at).Room.QuantumDoneDeferred
		if (tc.want == "r") != (deferred == 1) {
			t.Fatalf("%s: deferred count %d", tc.name, deferred)
		}
	}
}

// quantumLastTurn: a member running its quantum-ending turn at the bottom of the tier holds a waiting
// head; at that turn's FIRST CHUNK it is marked exiting, its blocks stop being protected, the head is
// admitted into them, and the member leaves at the turn's end ("quantum-last"). Off: the head waits.
func TestQuantumLastTurnReleasesALastTurnBlocker(t *testing.T) {
	for _, on := range []bool{true, false} {
		s := roomSched(t, true, func(c *Config) {
			c.RoomDisplace = true
			c.RoomHoldMaxSeconds = 0
			c.Quantum = 4
			c.QuantumLastTurn = on
		})
		at := epoch
		for i := 0; i < 3; i++ { // m: three completed busy periods (Q-1)
			runTurn(t, s, headN("m", i, time.Second, chain("m", 400)), at)
			at = at.Add(time.Second)
		}
		mh := headN("m", 3, time.Second, chain("m", 400)) // m's 4th = quantum-ending turn
		if i := s.Pass([]Head{mh}, at); i != 0 {
			t.Fatal("m's last turn must dispatch")
		}
		s.AtPod(mh.ReqID, "", at) // running, NO first chunk yet
		at = at.Add(time.Second)
		runTurn(t, s, headN("r", 0, time.Second, chain("r", 300)), at) // r: mid-quantum, between turns, newer
		at = at.Add(time.Second)
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 500)), at)
		evict(t, s, "x") // bottom -> top: m 400 | r 300 | x 500 | free 800
		at = at.Add(time.Second)
		z := headN("z", 0, time.Second, chain("z", 1000)) // budget 400+300+1000 <= 1900; room 800 < 1000
		s.Pass([]Head{z}, at)
		if seated(s, "z") {
			t.Fatalf("on=%v: z must be held (m is running at the bottom)", on)
		}
		if got := s.Snapshot(at).Room.Blockers[blockerRunningLast]; got != 1 {
			t.Fatalf("on=%v: the hold must name a running-last blocker, got %+v", on, s.Snapshot(at).Room.Blockers)
		}
		at = at.Add(time.Second)
		s.FirstChunk(mh.ReqID, at) // m's prompt KV is in the GPU now
		s.mu.Lock()
		marked := s.groupOf("m").index.Exiting("m")
		s.mu.Unlock()
		if marked != on {
			t.Fatalf("on=%v: m marked exiting at its last turn's first chunk = %v", on, marked)
		}
		at = at.Add(time.Second)
		s.Pass([]Head{z}, at)
		if seated(s, "z") != on {
			t.Fatalf("on=%v: z admitted = %v (m's blocks are room only when m is marked)", on, seated(s, "z"))
		}
		s.Settle(mh.ReqID, "", at.Add(time.Second))
		snap := s.Snapshot(at.Add(time.Second))
		if on {
			if seated(s, "m") || snap.Exits["quantum-last"] != 1 || snap.Room.QuantumLastMarks != 1 {
				t.Fatalf("m must leave at its turn's end as quantum-last: seated %v exits %+v marks %d", seated(s, "m"), snap.Exits, snap.Room.QuantumLastMarks)
			}
			if !seated(s, "r") || snap.Room.SeatedEvicted[tierCauseSeat] != 0 {
				t.Fatalf("r (mid-quantum) must stay untouched: seated %v evicted %+v", seated(s, "r"), snap.Room.SeatedEvicted)
			}
		} else if !seated(s, "m") {
			t.Fatal("off: m stays seated after its turn (nobody needs its seat by budget)")
		}
	}
}

// quantumLastTurn marks nothing when nobody waits outside, and nothing before the quantum-ending turn.
func TestQuantumLastTurnNeedsAWaiterAndTheLastTurn(t *testing.T) {
	s := roomSched(t, true, func(c *Config) { c.RoomDisplace = true; c.Quantum = 4; c.QuantumLastTurn = true })
	at := epoch
	for i := 0; i < 4; i++ { // nobody waiting: even the 4th (last) turn is not marked
		h := headN("m", i, time.Second, chain("m", 100))
		if i := s.Pass([]Head{h}, at); i != 0 {
			t.Fatal("m must dispatch")
		}
		s.AtPod(h.ReqID, "", at)
		s.FirstChunk(h.ReqID, at)
		s.mu.Lock()
		marked := s.groupOf("m").index.Exiting("m")
		s.mu.Unlock()
		if marked {
			t.Fatalf("turn %d: no waiter, no mark", i)
		}
		s.Settle(h.ReqID, "", at)
		at = at.Add(time.Second)
	}
	if s.Snapshot(at).Room.QuantumLastMarks != 0 {
		t.Fatal("no marks without a waiter")
	}
}

// roomVictims (head-driven binding): a room-held head sends out quantum-complete members that are between
// turns — oldest dispatch first, only until their blocks cover its write — and never a running or a
// mid-quantum member; each victim queues BEHIND the head it left for (even under newcomers-first), so its
// small warm write cannot re-seat it ahead of the head. Off: nobody is sent out.
func TestRoomVictimsAreHeadDrivenAndBounded(t *testing.T) {
	for _, tc := range []struct {
		name   string
		on     bool
		need   int
		victim []string
	}{{"on, write 500", true, 500, []string{"q1", "q2"}}, {"on, write 260", true, 260, []string{"q1"}}, {"off", false, 500, nil}} {
		s := roomSched(t, true, func(c *Config) {
			c.RoomDisplace = true
			c.RoomHoldMaxSeconds = 0
			c.Quantum = 4
			c.NewcomersFirst = true
			c.RoomVictims = tc.on
		})
		at := epoch
		d := headN("d", 0, time.Second, chain("d", 400))
		if i := s.Pass([]Head{d}, at); i != 0 {
			t.Fatal("d must dispatch")
		}
		s.AtPod(d.ReqID, "", at) // running at the bottom: the room blocker
		at = at.Add(time.Second)
		for _, id := range []string{"q1", "q2"} { // two quantum-complete members, q1 the older
			for i := 0; i < 4; i++ {
				runTurn(t, s, headN(id, i, time.Second, chain(id, 300)), at)
				at = at.Add(time.Second)
			}
		}
		runTurn(t, s, headN("r", 0, time.Second, chain("r", 200)), at) // mid-quantum, between turns
		at = at.Add(time.Second)
		// fill the tier's free space with a non-member so the head is room-held (x fits the GPU beside d)
		runTurn(t, s, headN("x", 0, time.Second, chain("x", 550)), at)
		evict(t, s, "x") // free 2000-400-600-200-550 = 250
		at = at.Add(time.Second)
		// q1's next (5th) turn arrived BEFORE the head: without the floor it would sort ahead of the head
		q1h := Head{Thread: "q1", ReqID: "q1#4", Arrival: at, Chains: chain("q1", 300), Returner: true}
		z := Head{Thread: "z", ReqID: "z#1", Arrival: at.Add(time.Second), Chains: chain("z", tc.need), Returner: true}
		heads := []Head{z, q1h}
		// z (cold): budget 400+600+200+write <= 1900; room 250 (d stops the walk) < its write
		s.Pass(heads, at.Add(2*time.Second))
		snap := s.Snapshot(at)
		var gone []string
		for _, id := range []string{"q1", "q2", "r", "d"} {
			if !seated(s, id) {
				gone = append(gone, id)
			}
		}
		if fmt.Sprint(gone) != fmt.Sprint(tc.victim) && !(len(gone) == 0 && tc.victim == nil) {
			t.Fatalf("%s: sent out %v, want %v", tc.name, gone, tc.victim)
		}
		if got := snap.Exits["quantum-head"]; got != int64(len(tc.victim)) {
			t.Fatalf("%s: quantum-head exits %d, want %d", tc.name, got, len(tc.victim))
		}
		if tc.on {
			s.mu.Lock()
			w1, wz := s.waiters["q1"], s.waiters["z"]
			s.mu.Unlock()
			if w1 == nil || wz == nil || !entryKey(w1).After(entryKey(wz)) {
				t.Fatalf("%s: q1 must queue behind z (the head it left for): q1 %+v z %+v", tc.name, w1, wz)
			}
			// the next pass must not re-seat q1 ahead of z
			s.Pass(heads, at.Add(3*time.Second))
			if seated(s, "q1") && !seated(s, "z") {
				t.Fatalf("%s: q1 re-seated ahead of the head it left for", tc.name)
			}
		}
	}
}
