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

// THE STATE MACHINE'S NOUNS: the thread, the waiter, the request record, the engine group, and the
// Scheduler that owns them — plus the engine life cycle and the whole-policy Snapshot.
//
// THE LOCK CONVENTION, stated once, for every file in this package:
//   - EXPORTED methods take s.mu themselves and are safe to call from anywhere;
//   - UNEXPORTED methods ASSUME the caller already holds it — no name says so, because there is no
//     other kind of unexported method here;
//   - the BlockIndex has no lock of its own: it is called only under s.mu.

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/approximateprefix"
)

// ---- the queue-head snapshot (waiting state comes from the flow-control
// queues at Pick time, never from a hook) ----

// Head is one waiting queue head as seen at Pick: the thread id (the fairness unit AND the
// queue key), the request id, the arrival, and the request's exact prefill chains.
type Head struct {
	// Thread is the group policy's unit: the derived thread id, or "" for anonymous traffic
	// (never tracked, never seated).
	Thread  string
	ReqID   string
	Arrival time.Time
	// Chains is the request's EXACT prefill chains, computed once from the tokens the token
	// producer rendered at arrival (one chain per prompt).
	Chains []Chain
	// FairnessID is the request's flow-control identity — its QUEUE KEY.
	FairnessID string
	// WireID is the client's OWN id for this request, from its headers (x-correlation-id, else
	// x-request-id): the one field that joins a SEAT/EXIT line to the client's own request records,
	// whatever the queue key turned out to be.
	WireID string
	// Returner reports whether this request is NOT its thread's first turn — the fairness class,
	// from thread identity's arrival record.
	Returner bool
}

// ---- per-thread scheduling state ----

// thread is one seated residency: a record that exists exactly as long as the thread holds a
// seat on ONE engine: the seat IS the binding.
type thread struct {
	id string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// seatTurns counts settled turns in this residency (the residency histogram's unit).
	seatTurns int
	// inflight and pending are DERIVED each decision point from the request map (derive-
	// don't-accumulate): a leaked increment cannot persist past a reconciliation.
	inflight int
	pending  int
	// busyPeriods counts completed busy periods: the thread's in-flight count reaching zero and
	// staying there, which — one thread, one chain — is the whole of its work draining.
	busyPeriods int
	// lastCompletion is the last time the thread's in-flight count returned to zero.
	lastCompletion time.Time
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this thread's outstanding cold-entry charge in TOKENS against the stagger cap;
	// released EXACTLY once, at the first of: prefill complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// windows are this thread's released-but-not-yet-at-a-pod records, so the END of a turn can
	// expire its own dead windows without scanning the whole record map: a window stops counting
	// when it outlives S, which is a fact about TIME, not about whether a pass has run since.
	windows []*reqRec
	// seatObs is the observability of the turn that SEATED this thread, measured at the seat before
	// its own reservation: the first dispatch after the seat reports it instead of a post-insert
	// view (which would count the reservation as residency). Cleared by that dispatch.
	seatObs *dispatchObservables
	// waitingSince is the thread's oldest queued head: the wait its victims and its SEAT event
	// are measured from, kept OLDEST across fan-out re-arrivals.
	waitingSince time.Time
	// headChains / headReturner are the LAST queued head observed for this thread, from the queue
	// view of a decision pass.
	headChains   []Chain
	headReturner bool
	// exitingSince is when this thread was marked exiting (the mid-flight class): the age of a
	// retiree is the time between "nothing more will be admitted for it" and "its blocks are off
	// the budget", i.e. how long the engine pays for a seat it has already written off.
	exitingSince time.Time
	// exitReason is the reason a MARKED thread's drain books (empty = "exiting-drain")
	exitReason string
	// fairnessID is the last fairness id (queue key) this thread was seen with — the queue key
	// its SEAT/EXIT events name (see Head.FairnessID).
	fairnessID string
	// wireID is the last client request id this thread was seen with: the field a SEAT/EXIT line
	// joins a request record on (see Head.WireID).
	wireID string
	// lastDispatch is when this thread's chain was last touched in the engine's tier: its last
	// dispatch, or its seat (with roomGate the seat records the chain). victimOldestDispatch orders the
	// rotation victims by it.
	lastDispatch time.Time
	// lastServed is this thread's last DISPATCH in this stay — set only at dispatch (never at the seat), so it is
	// zero until the stay's first turn goes out. It is when vLLM last touched the chain (its lookup touches the
	// whole chain; the turn's new prompt blocks are stored right after; decode output never is), so "served
	// longest ago" is "closest to the tier's eviction end". leastRecentlyServed orders the seated turns by it.
	lastServed time.Time
	// seatTierResident is the tier-resident prefix (tokens) the group's own index read for this
	// thread's head at the seat, BEFORE the seat recorded it; seatTierPending marks that the first
	// dispatch after the seat must report it (a read after the seat's record would say "all resident").
	seatTierResident int64
	seatTierPending  bool
}

// busy reports whether the thread has work in progress: a request at a pod, or a LIVE released
// window (a turn that a Pick released and PreRequest has not yet brought to a pod).
func (m *thread) busy() bool { return m.inflight > 0 || m.pending > 0 }

// drainReason is the exit reason a marked thread's drain books.
func (m *thread) drainReason() string {
	if m.exitReason != "" {
		return m.exitReason
	}
	return "exiting-drain"
}

// noQueuedHead reports that this thread has nothing waiting in the queue. It is NOT the same
// question as "no turn waiting": a thread whose head a Pick has already released has no queued
// head and a turn on the way.
func (m *thread) noQueuedHead() bool { return m.waitingSince.IsZero() }

// reqRec is ONE tracked request's record, and the scheduler's only memory of a request that is
// not in a queue.
type reqRec struct {
	// thread is the thread that released the request (never empty: anonymous traffic is not
	// tracked).
	thread string
	// chains is the prefill chain the release was charged with.
	chains []Chain
	// engine is the engine the request landed on, once an observation named one.
	engine string
	// released is the Pick that released the request.
	released time.Time
	// pod is when the request reached a pod (PreRequest); zero while it is pending.
	pod time.Time
	// firstToken is set at the request's first chunk: its prefill is done (fairRotation removes an
	// in-flight member only after it, when the request no longer writes into the CPU tier).
	firstToken bool
	// counted is whether this record currently contributes to its thread's pending count. It is set
	// when the record is opened, and corrected by the SAME liveness test at every decrement, so
	// pending is always exactly "the records that counted at their last evaluation".
	counted bool
	// the OBSERVABLES of this turn, captured at dispatch for the span PreRequest may set. Nothing
	// in a decision reads them: the queue wait is dispatch − arrival, and the predicted cache is
	// the resident prefix read BEFORE the charge (after it, it would always be everything).
	queueWait       time.Duration
	predictedCached int64
	promptTokens    int64
}

// waiter is one head in the SHARED waiter queue.
type waiter struct {
	id    string
	since time.Time // the OLDEST arrival: a fan-out re-arrival does not reset it
	// returner is the fairness class: this thread's queued turn is NOT its first one.
	returner bool
	// chains is the head's exact prefill chain set.
	chains []Chain
	// fairnessID rides with the head (see Head.FairnessID) so a SEAT event can name the queue key
	// even before the thread has a record, and wireID rides with it for the join to the client's
	// own request record (see Head.WireID).
	fairnessID string
	wireID     string
	// floor is the queue-ORDER floor a rotation leaves on its victim (newcomersFirst false): the victim
	// queues no earlier than just behind the head that asked for its seat. since stays the TRUE arrival
	// (the wait metrics and the SEAT event read it); the queue orders by entryKey = max(since, floor).
	// syncQueue never touches it, and the waiter is dropped at its seat or when it vanishes.
	floor time.Time
	// claim / claimSince / claimVictims (claimOrder): the engine this waiter holds as its blocked head, when it
	// first claimed one, and the victims removed for it on its current claim (claims.go).
	claim        string
	claimSince   time.Time
	claimVictims int
	// roomHeldSince is the first pass the room gate held this head (zero while it is not held): the
	// roomHoldMaxSeconds last resort is measured from it, never from since (a returner's since is its
	// whole exile).
	roomHeldSince time.Time
	// victimBlocks (roomVictims) is the blocks of the members already sent out for THIS head: it takes
	// victims only while this is below its write, so one head never empties more than it needs.
	victimBlocks int
	// keySet caches the head's own chain keys for the room walk (rebuilt when keySetOf changes).
	keySet   map[Hash]struct{}
	keySetOf *Hash
	keySetN  int
}

// ownKeys is the head's chain keys as a set, cached on the waiter: the room walk skips them.
func (w *waiter) ownKeys() map[Hash]struct{} {
	var first *Hash
	n := 0
	for _, c := range w.chains {
		if len(c.Keys) > 0 && first == nil {
			first = &c.Keys[0]
		}
		n += len(c.Keys)
	}
	if w.keySet != nil && w.keySetOf == first && w.keySetN == n {
		return w.keySet
	}
	set := make(map[Hash]struct{}, n)
	for _, c := range w.chains {
		for _, k := range c.Keys {
			set[k] = struct{}{}
		}
	}
	w.keySet, w.keySetOf, w.keySetN = set, first, n
	return set
}

// engineGroup is everything that is a fact about ONE engine's HBM: its block index (dedup is per
// engine — a block on A is worthless to B), its budget, its measured capacity, its seats, its
// cold-entry charge and its counters.
type engineGroup struct {
	// id is the endpoint id the EPP discovered.
	id string

	index *BlockIndex // attached at creation (block size from config)
	// budgetBlocks is what this engine may SEAT, in blocks: groupBudgetFraction x THIS engine's
	// measured capacity, or with a CPU tier tierBudgetFraction x the tier. 0 until THIS engine's
	// num_gpu_blocks arrives.
	budgetBlocks int
	// runBudgetBlocks is what the seats with a turn RUNNING may hold on the GPU: groupBudgetFraction
	// x num_gpu_blocks. Without a tier it equals budgetBlocks.
	runBudgetBlocks int
	numBlocks       int
	// blockSizeMismatch is the loud per-engine wiring failure (a measured CacheBlockSize that
	// differs from the config's BlockSizeTokens).
	blockSizeMismatch error

	threads map[string]*thread
	// coldInFlight is this engine's outstanding cold-entry charge against its own stagger cap, so
	// N engines allow N concurrent cold prefills and no more.
	coldInFlight int64

	// tier is the group's OWN model of this engine's CPU tier (ownTierIndex): created with the
	// engine's capacity, touched at every dispatch and (roomGate) recorded at every seat. nil = off.
	tier *approximateprefix.TierLRU
	// incarnation names this group's life in its member notices (memberTierEviction): the engine's policy
	// restarts its epoch on a new one
	incarnation string
	// protectStreak / protectSince (leastRecentlyServed): protections in a row on this engine since its last
	// unprotected dispatch, and when the first of them happened — observed when the next candidate is served
	// (the strict rule has no bound; this measures what one would have capped).
	protectStreak int
	protectSince  time.Time
	// protectMemo is the engine's last protection decision and its inputs (protect.go)
	protectMemo *protectMemo
	// tierCause names what the tier index is being written for right now (seat / dispatch /
	// passthrough / rebind), so its eviction observer can book a seated block's eviction by cause.
	tierCause string
	// roomMemo caches the room gate's walks (head, victim, need) while neither the tier index nor the
	// seated set has changed: a room-held head is re-tested on every pass (~1 kHz).
	roomMemo    map[roomKey]int
	roomMemoGen [3]uint64
	// eligCount (roomDisplace) counts, per block, the ELIGIBLE staying members that hold it: a block is
	// protected iff it has a staying holder that is not eligible (blocksStay > eligCount). Cached for one
	// (pass time, tier gen, index gen).
	eligCount map[Hash]int
	eligKey   [3]uint64
	// displaced (roomDisplace) collects, while a seat's chain is being recorded, the staying blocks the
	// write evicted: their eligible holders are rotated out right after the record.
	displaced map[Hash]struct{}

	counters struct {
		served int64 // dispatches served on this engine
		exits  map[string]int64
	}
}

// coldCapTokens is THIS engine's cold-ENTRY stagger cap, in tokens. It is NOT a knob: it is the
// seat budget's own headroom — floor((1 - groupBudgetFraction) x num_gpu_blocks) blocks x
// blockSizeTokens — so the valve can only ever spend the margin the budget already reserved.
func (g *engineGroup) coldCapTokens(fraction float64, blockSize int) int64 {
	if g.numBlocks <= 0 || blockSize <= 0 {
		return 0
	}
	headroomBlocks := int64((1 - fraction) * float64(g.numBlocks))
	return headroomBlocks * int64(blockSize)
}

// newEngineGroup builds one engine's group: its block index, and the counters it will report.
func newEngineGroup(id string, blockSizeTokens int) (*engineGroup, error) {
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return nil, err
	}
	g := &engineGroup{id: id, index: idx,
		threads: map[string]*thread{}}
	g.counters.exits = map[string]int64{}
	return g, nil
}

// engineLabel is the metrics/event label value for an engine id.
func engineLabel(id string) string { return id }

// Scheduler is the group scheduler state machine over an engine set.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// plainLRS (tests only) turns the strict rule's protection off — leastRecentlyServed becomes the bare time
	// order, so the replay can compare the two on the same seeds. Never set by configuration.
	plainLRS bool
	// noProtectMemo (tests only) recomputes the protection decision on every pass (the memo's equivalence test).
	noProtectMemo bool
	// metrics is the Prometheus surface. Every observation is a DIRECT call on it, and every
	// method is nil-safe, so a bare Scheduler (the unit tests) records nothing without any
	// wiring.
	metrics *groupMetrics
	// logger carries the two lines a run's artifacts have to be able to point at: an affinity
	// violation and an engine disappearing under its threads. The zero Logger is safe.
	logger logr.Logger
	// onEvent is the ONLY observer: the forensic seat/exit/rebind stream. The plugin logs it at
	// V(DEBUG) and the tests collect it; nothing else watches the pass.
	onEvent func(Event)
	// groups holds one engineGroup per DISCOVERED engine, keyed by endpoint id.
	groups map[string]*engineGroup
	// groupOrder is the sorted key list (deterministic iteration: every cross-engine decision is
	// taken in id order, never map order).
	groupOrder []string

	blockSize int

	// bindings is the thread→engine association.
	bindings map[string]string

	// waiters is THE SHARED WAITER QUEUE: one entry per thread with a queued head and no seat.
	waiters map[string]*waiter

	// reqs is EVERY tracked request, keyed by request id: the record written when a Pick releases
	// a turn and dropped at its settle, its abandon, its reap, or its thread's exit.
	reqs map[string]*reqRec

	// now is the virtual clock of the CURRENT decision pass (set at the top of every public
	// decision entry point).
	now time.Time

	// residency, when set, answers per engine how many LEADING blocks of a chain the engine's CPU
	// tier still holds (the CPU-residency index, read through the plugin handle). It only ORDERS the
	// engines a head is placed on (fewest tokens to really prefill first); it never admits, evicts or
	// charges anything. nil = placement exactly as without it.
	residency func(keys []Hash, engines []string) []int

	counters struct {
		// anomalies counts the EVENTS THE POLICY CANNOT ACT ON, by kind (anomalyKinds).
		anomalies map[string]int64
		// affinityViolations counts requests that LANDED on an engine other than the one they were
		// seated on (CORRECTNESS: the observed endpoint wins, so each violation is repaired by
		// moving the seat; a nonzero steady rate means the placement path is not honouring the pin).
		affinityViolations int64
		engineGone         int64
		exits              map[string]int64
		served             int64
		fcfsServed         int64
		// placements counts seats by what the residency index said about the chosen engine
		// (placementOutcomes); seatResident / seatCold split the seated heads' prompt tokens into the
		// part the chosen engine's tier held and the part it had to prefill. Residency on only.
		placements   map[string]int64
		seatResident int64
		seatCold     int64
		// THE ROOM GATE's own proof that it fired (roomGate): heads it held (once per held head, not
		// per pass), heads its last resort admitted, members rotated out to make room, and index
		// writes by cause; seatedEvicted is the index's eviction of a SEATED (staying) block, by cause
		// — the in-group loss the gate exists to prevent.
		roomHeld      int64
		roomForced    int64
		roomRotations int64
		tierWrites    map[string]int64
		seatedEvicted map[string]int64
		// tierDemoted (memberTierEviction) counts blocks leavers' exits moved to the tier index's eviction end
		tierDemoted int64
		// roomBlocker counts, once per room-held head, what held it (the member state ending the walk)
		roomBlocker map[string]int64
		// quantumDoneDeferred counts dispatches where quantumDoneLast served a member within its quantum
		// ahead of an OLDER turn of a quantum-complete member
		quantumDoneDeferred int64
		// quantumLastMarks counts members quantumLastTurn marked at the first chunk of their last turn
		quantumLastMarks int64
		// headVictims counts members roomVictims sent out for a room-held head
		headVictims int64
		// fairRemovals counts members fair rotation took out, by reason (idle / fair-budget / fair-room)
		fairRemovals map[string]int64
		// fairTailSaved sums, over budget victims (costFairRotation), the staying blocks at the tail that their
		// leaving took off the head's write (thrash avoided ungated; room unblocked gated)
		fairTailSaved int64
		// sizeWaitOvertakes counts dispatches where the in-group order's head (sizeWaitOrder or
		// leastRecentlyServed) was not the oldest seated turn
		sizeWaitOvertakes int64
		// protected / protectedBlocks / protectDeclined (leastRecentlyServed): dispatches where a member at risk
		// was served before the candidate, the blocks that saved, and dispatches where members were at risk but
		// no protection was a net save
		protected, protectedBlocks, protectDeclined int64
		// protectEvals counts protection decisions actually computed (memo misses) — tests read it
		protectEvals int64
		// protectSkippedWaiting / protectSkippedBlocks: dispatches where a quantum-complete queued member had
		// blocks in the candidate's danger zone but was not protected because someone waits outside (the
		// simplification that "someone waits" is global across engines), and those blocks
		protectSkippedWaiting, protectSkippedBlocks int64
		// claims (claimOrder): new claims, waiters seated while an older waiter held a blocked claim in the same
		// pass, and victims removed for a waiter that was then seated on another engine (should stay ≈ 0)
		claims, seatedPastClaim, claimVictimsElsewhere int64
		// tokensGivenUp: prompt tokens the warmest engine held beyond the chosen engine, summed over seats
		// (greedy placement's cold cost)
		tokensGivenUp int64
	}
}

// roomKey is one memoised room walk.
type roomKey struct {
	head, victim string
	need, own    int
}

// tier write causes (index writes and seated-block evictions are booked by them).
const (
	tierCauseSeat        = "seat"
	tierCauseDispatch    = "dispatch"
	tierCausePassthrough = "passthrough"
	tierCauseRebind      = "rebind"
)

// tierCauseDisplaced books (roomDisplace) the blocks of ELIGIBLE members a seat's write pushed out —
// the room the design meant to use; their members are rotated out with exit reason "displaced".
const tierCauseDisplaced = "displaced"

var tierCauses = []string{tierCauseSeat, tierCauseDispatch, tierCausePassthrough, tierCauseRebind}

// recordTier writes a chain set into the engine's own tier index (ownTierIndex; a no-op otherwise),
// booking the write and any SEATED block it evicts under cause.
func (s *Scheduler) recordTier(g *engineGroup, chains []Chain, cause string) {
	if g == nil || g.tier == nil {
		return
	}
	g.tierCause = cause
	for _, c := range chains {
		g.tier.AddChain(c.Keys)
	}
	g.tierCause = ""
	s.counters.tierWrites[cause]++
	s.metrics.ObserveTierWrite(engineLabel(g.id), cause)
}

// ownResidency is the residency read of the group's OWN tier index: per engine id (positional), the
// leading blocks of keys that engine's index holds. It is placement's residency read with
// ownTierIndex on.
func (s *Scheduler) ownResidency(keys []Hash, engines []string) []int {
	out := make([]int, len(engines))
	for i, id := range engines {
		if g, ok := s.groups[id]; ok && g.tier != nil {
			out[i] = g.tier.ResidentPrefix(keys)
		}
	}
	return out
}

// SetResidency installs the CPU-residency read (nil removes it). Called once at startup, before any
// pass, by the plugin's sibling gate.
func (s *Scheduler) SetResidency(f func(keys []Hash, engines []string) []int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.residency = f
}

// NewScheduler builds the state machine.
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	s := &Scheduler{cfg: cfg, logger: logr.Discard(), blockSize: cfg.BlockSizeTokens,
		groups:   map[string]*engineGroup{},
		bindings: map[string]string{},
		reqs:     map[string]*reqRec{},
		waiters:  map[string]*waiter{}}
	s.counters.exits = map[string]int64{}
	s.counters.anomalies = map[string]int64{}
	s.counters.placements = map[string]int64{}
	s.counters.tierWrites = map[string]int64{}
	s.counters.seatedEvicted = map[string]int64{}
	s.counters.roomBlocker = map[string]int64{}
	s.counters.fairRemovals = map[string]int64{}
	if cfg.OwnTierIndex {
		// the group's own tier index is placement's residency read (it replaces cpuResidencyProducer)
		s.residency = s.ownResidency
	}
	return s, nil
}

// groupFor returns (creating) the group for an engine id.
func (s *Scheduler) groupFor(id string) (*engineGroup, error) {
	if g, ok := s.groups[id]; ok {
		return g, nil
	}
	g, err := newEngineGroup(id, s.blockSize)
	if err != nil {
		return nil, err
	}
	s.groups[id] = g
	s.rebuildGroupOrder()
	// this engine's zero-valued COUNTER series exist from its first scrape, whatever its capacity
	// turns out to be (a budgetless or block-size-mismatched group included)
	s.metrics.preInstantiateEngine(id)
	return g, nil
}

// rebuildGroupOrder re-derives the sorted engine id list every cross-engine decision iterates in.
func (s *Scheduler) rebuildGroupOrder() {
	ids := make([]string, 0, len(s.groups))
	for id := range s.groups {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	s.groupOrder = ids
}

// groupOf is the thread's SEAT engine (nil when the thread is unseated).
func (s *Scheduler) groupOf(thread string) *engineGroup {
	id, ok := s.bindings[thread]
	if !ok {
		return nil
	}
	return s.groups[id]
}

// activeGroups is the deterministic engine list every decision pass iterates: every
// discovered engine, in id order.
func (s *Scheduler) activeGroups() []*engineGroup {
	out := make([]*engineGroup, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		out = append(out, s.groups[id])
	}
	return out
}

// SetNumBlocksFor attaches ONE engine's measured capacity.
func (s *Scheduler) SetNumBlocksFor(engine string, blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, err := s.groupFor(engine)
	if err != nil {
		return err
	}
	if blockSizeTokens > 0 && blockSizeTokens != s.blockSize {
		g.blockSizeMismatch = fmt.Errorf(
			"groupsched[%s]: measured CacheBlockSize %d != configured BlockSizeTokens %d (the token processor's value); refusing the capacity",
			engineLabel(g.id), blockSizeTokens, s.blockSize)
		return g.blockSizeMismatch
	}
	if g.numBlocks > 0 || numBlocks <= 0 {
		return nil // first capacity wins, PER ENGINE
	}
	g.numBlocks = numBlocks
	// The GPU budget: the deployment's own statement about how much of this engine's KV one group
	// may hold. Without a tier it sizes the seats; with one it sizes what runs, and the tier sizes the
	// seats (never below what the GPU alone holds).
	g.runBudgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	g.budgetBlocks = g.runBudgetBlocks
	if s.cfg.TierCapacityTokens > 0 {
		tierBlocks := s.cfg.TierCapacityTokens / int64(s.blockSize)
		g.budgetBlocks = max(g.runBudgetBlocks, int(s.cfg.TierBudgetFraction*float64(tierBlocks)))
		if s.cfg.OwnTierIndex && g.tier == nil {
			g.tier = approximateprefix.NewTierLRU(int(tierBlocks))
			gg := g
			g.tier.OnEvict(func(k Hash) {
				if gg.index.Staying(k) && gg.displaced != nil {
					gg.displaced[k] = struct{}{} // classified (displaced vs seated) after the seat's record
					return
				}
				if gg.index.Staying(k) {
					cause := gg.tierCause
					if s.cfg.RoomDisplace && cause != tierCauseSeat {
						// roomDisplace: a block whose every staying holder is ELIGIBLE is a designated
						// leaver's, booked apart from a protected member's (the real in-group loss)
						if elig := s.eligibleKeyCount(gg); gg.index.blocksStay[k] <= elig[k] {
							cause += "-eligible"
						}
					}
					s.counters.seatedEvicted[cause]++
					s.metrics.ObserveSeatedEvicted(engineLabel(gg.id), cause)
				}
			})
			if s.cfg.MemberTierEviction {
				// a leaver's unshared blocks go to the eviction end — the engine's policy does the same move
				gg.index.onLeaveStay = func(keys []Hash) {
					if n := gg.tier.Demote(keys); n > 0 {
						s.counters.tierDemoted += int64(n)
						s.metrics.ObserveTierDemoted(engineLabel(gg.id), n)
					}
				}
			}
		}
	}
	return nil
}

// tiered reports whether the group has a CPU tier: the seats are sized by it, and what runs by the GPU.
func (s *Scheduler) tiered() bool { return s.cfg.TierCapacityTokens > 0 }

// BudgetsOf is one engine's seat and run budgets, in blocks (0 before its capacity arrives).
func (s *Scheduler) BudgetsOf(engine string) (seat, run int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g, ok := s.groups[engine]; ok {
		return g.budgetBlocks, g.runBudgetBlocks
	}
	return 0, 0
}

// HasCapacity reports whether ANY engine has a measured num_gpu_blocks (the pre-capacity FCFS
// pass-through's whole-stack condition).
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, g := range s.groups {
		if g.numBlocks > 0 {
			return true
		}
	}
	return false
}

// HasCapacityFor reports whether THAT engine's capacity has arrived. Until it has, the engine
// takes no seats: an engine without metrics would serve nothing while looking healthy.
func (s *Scheduler) HasCapacityFor(engine string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, ok := s.groups[engine]
	return ok && g.numBlocks > 0
}

// GroupLabel resolves an endpoint key to the label the metrics and events carry for it.
func (s *Scheduler) GroupLabel(engine string) string {
	return engineLabel(engine)
}

// BlockSizeMismatches is every mismatched engine, in id order.
func (s *Scheduler) BlockSizeMismatches() map[string]error {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]error{}
	for _, id := range s.groupOrder {
		if err := s.groups[id].blockSizeMismatch; err != nil {
			out[engineLabel(id)] = err
		}
	}
	return out
}

// FCFSServed counts a pre-capacity dispatch: it belongs to no engine (a pre-capacity stack has
// none to attribute it to), so it is only counted.
func (s *Scheduler) FCFSServed() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
}

// ---- the seat/exit event log (spans cannot see seats, so every run needs it) ----

// Event is one seat/exit/rebind decision, per id, with reason and timestamp.
type Event struct {
	T      time.Time `json:"t"`
	Kind   string    `json:"kind"`   // SEAT | EXIT | REBIND | AFFINITY-VIOLATION
	ID     string    `json:"id"`     // the thread id the decision was taken for
	Engine string    `json:"engine"` // the engine label the decision happened on
	Detail string    `json:"detail"` // reason=… turns=… / kind=… wait=… / from=… to=… / fairness_id=… wire_id=…
}

// recordEvent books one seat/exit event: the engine attribution plus the observer. The Event is
// the forensic stream (the adapter logs it at V(DEBUG)); nothing keeps a copy.
func (s *Scheduler) recordEvent(kind, id, detail string, now time.Time) {
	ev := Event{T: now, Kind: kind, ID: id, Detail: detail}
	if g := s.groupOf(id); g != nil {
		ev.Engine = engineLabel(g.id)
	}
	if s.onEvent != nil {
		s.onEvent(ev)
	}
}

// anomalyNoTokens opens the anomaly kinds: each names ONE event the policy cannot act on, and each is
// counted under its own name in anomalies_total. This one is the render failure — a Pick pass saw a
// thread head with no rendered tokens, so the head could not be seated or charged.
const (
	// anomalyNoTokens: a Pick pass saw a thread head with no rendered tokens, so it cannot be
	// seated or charged (the render failed; thread identity counts that too).
	anomalyNoTokens = "no_tokens_head"
	// anomalyNoRequestID: a Pick released a turn for a request with no id.
	anomalyNoRequestID  = "no_request_id"
	anomalyReaped       = "reaped_inflight"
	anomalyReapedWindow = "reaped_window"
)

// anomalyKinds is every kind, in a fixed order (the metric pre-instantiates one zero series
// per kind: a zero is a fact, an absent series is a question).
var anomalyKinds = []string{anomalyNoTokens, anomalyNoRequestID, anomalyReaped, anomalyReapedWindow}

// noteAnomaly books one anomaly and tells the observer.
func (s *Scheduler) noteAnomaly(kind string) {
	if s.counters.anomalies == nil {
		s.counters.anomalies = map[string]int64{}
	}
	s.counters.anomalies[kind]++
	s.metrics.ObserveAnomaly(kind)
}

// countExit books ONE departure on the engine AND in the global tally (Snapshot.Exits
// stays the whole-policy view).
func (s *Scheduler) countExit(g *engineGroup, reason string) {
	if g != nil {
		g.counters.exits[reason]++
	}
	s.counters.exits[reason]++
}

// Snapshot is the whole-policy view of the state machine: DumpState serialises it and the metrics
// refresh reads it, so it carries exactly what either of them reports.
type Snapshot struct {
	Seats int
	// NEBlocks is the USED side: every seated thread's distinct blocks, exiting threads INCLUDED
	// (the engine pins them until their requests drain).
	NEBlocks     int
	BudgetBlocks int
	// WaitersShared is the SHARED waiter queue's size — the whole policy's admission queue.
	WaitersShared int
	Exits         map[string]int64
	Served        int64
	FCFSServed    int64
	// Engines is the PER-ENGINE section.
	Engines map[string]EngineSnapshot `json:"engines,omitempty"`
	// AffinityViolations counts requests that LANDED off their thread's seat engine (the
	// correctness counter: each one is repaired by the observed endpoint winning).
	AffinityViolations int64 `json:"affinityViolations"`
	EngineGone         int64 `json:"enginesGone"`
	// Anomalies is the should-be-zero ledger, by kind (anomalyKinds): every entry is an event the
	// policy could not act on.
	Anomalies map[string]int64 `json:"anomalies,omitempty"`
	// Residency is the CPU-residency placement ledger (absent when the index is not configured).
	Residency *ResidencySnapshot `json:"residency,omitempty"`
	// Room is the room gate's ledger (absent without ownTierIndex).
	Room *RoomSnapshot `json:"room,omitempty"`
}

// ResidencySnapshot is what the CPU-residency index did to placement: seats by outcome, and the
// seated heads' prompt tokens split into tier-resident and to-prefill on the chosen engine.
type ResidencySnapshot struct {
	Placements         map[string]int64 `json:"placements"`
	SeatResidentTokens int64            `json:"seatResidentTokens"`
	SeatColdTokens     int64            `json:"seatColdTokens"`
}

// EngineSnapshot is ONE engine's group state: seats, the budget, the exit mix and the exiting
// ledger.
type EngineSnapshot struct {
	ID    string `json:"id"`
	Seats int    `json:"seats"`
	// NEBlocks is the USED side (exiting threads included): see Snapshot.NEBlocks.
	NEBlocks     int `json:"neBlocks"`
	BudgetBlocks int `json:"budgetBlocks"`
	NumBlocks    int `json:"numBlocks"`
	// The GPU side: its budget, and the seats with a turn running and the blocks they hold.
	RunBudgetBlocks int   `json:"runBudgetBlocks"`
	RunningBlocks   int   `json:"runningBlocks"`
	RunningThreads  int   `json:"runningThreads"`
	ColdInFlight    int64 `json:"coldInFlightTokens"`
	// ColdCapTokens is the DERIVED cold-entry stagger cap (the budget's own headroom, in tokens).
	ColdCapTokens int64 `json:"coldCapTokens"`
	// The EXITING ledger (the enforcement view's own inputs, per engine): threads on their way
	// out, the blocks of the used side they still hold, and how long the oldest one has been
	// leaving.
	ExitingThreads    int              `json:"exitingThreads"`
	ExitingBlocks     int              `json:"exitingBlocks"`
	ExitingOldestAgeS float64          `json:"exitingOldestAgeSeconds"`
	Served            int64            `json:"served"`
	Exits             map[string]int64 `json:"exits,omitempty"`
	BlockSizeErr      string           `json:"blockSizeMismatch,omitempty"`
	// TierIndexBlocks / TierIndexCapBlocks: the group's own tier index (ownTierIndex; 0 when off).
	TierIndexBlocks    int `json:"tierIndexBlocks,omitempty"`
	TierIndexCapBlocks int `json:"tierIndexCapBlocks,omitempty"`
}

// RoomSnapshot is the room gate's ledger (roomGate / ownTierIndex): heads held, last-resort seats,
// rotations for room, the index's writes and the seated blocks they evicted, by cause.
type RoomSnapshot struct {
	Held          int64            `json:"held"`
	Forced        int64            `json:"forced"`
	Rotations     int64            `json:"rotations"`
	TierWrites    map[string]int64 `json:"tierWrites"`
	SeatedEvicted map[string]int64 `json:"seatedEvicted"`
	// Blockers: once per room-held head, the state of the member whose block ended the room walk
	Blockers map[string]int64 `json:"blockers"`
	// QuantumDoneDeferred: dispatches where quantumDoneLast put a quantum-complete member's older turn back
	QuantumDoneDeferred int64 `json:"quantumDoneDeferred"`
	// QuantumLastMarks: members quantumLastTurn marked exiting at their last turn's first chunk
	QuantumLastMarks int64 `json:"quantumLastMarks"`
	// HeadVictims: members roomVictims sent out for a room-held head (exit reason "quantum-head")
	HeadVictims int64 `json:"headVictims"`
	// FairRemovals: members fair rotation took out, by reason
	FairRemovals map[string]int64 `json:"fairRemovals,omitempty"`
	// FairTailSaved: staying tail blocks the cost-fair victims' leaving took off heads' writes
	FairTailSaved int64 `json:"fairTailSaved,omitempty"`
	// SizeWaitOvertakes: dispatches whose head (sizeWaitOrder / leastRecentlyServed) overtook an older seated turn
	SizeWaitOvertakes int64 `json:"sizeWaitOvertakes,omitempty"`
	// Protected / ProtectedBlocks / ProtectDeclined: the strict rule's protections, the blocks they saved, and
	// the at-risk cases where no protection was a net save
	Protected       int64 `json:"protected,omitempty"`
	ProtectedBlocks int64 `json:"protectedBlocks,omitempty"`
	ProtectDeclined int64 `json:"protectDeclined,omitempty"`
	// ProtectSkippedWaiting / ProtectSkippedBlocks: see counters.protectSkippedWaiting
	ProtectSkippedWaiting int64 `json:"protectSkippedWaiting,omitempty"`
	ProtectSkippedBlocks  int64 `json:"protectSkippedBlocks,omitempty"`
	// Claims / SeatedPastClaim / ClaimVictimsElsewhere (claimOrder): see counters.claims
	Claims                int64 `json:"claims,omitempty"`
	SeatedPastClaim       int64 `json:"seatedPastClaim,omitempty"`
	ClaimVictimsElsewhere int64 `json:"claimVictimsElsewhere,omitempty"`
	// TokensGivenUp: see counters.tokensGivenUp
	TokensGivenUp int64 `json:"tokensGivenUp,omitempty"`
}

// Snapshot is the whole-policy view of the state machine: DumpState serialises it and the scrape
// collector reads it. It takes the caller's clock because one field in it — the oldest exiting
// thread's age — is a DURATION, and the scheduler's own clock only advances when the pass runs: an
// idle stack would otherwise report a frozen age, and that gauge is the drain warning.
func (s *Scheduler) Snapshot(now time.Time) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshot(now)
}

// snapshot is Snapshot under the caller's lock.
func (s *Scheduler) snapshot(now time.Time) Snapshot {
	snap := Snapshot{Exits: map[string]int64{},
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed,
		AffinityViolations: s.counters.affinityViolations,
		EngineGone:         s.counters.engineGone,
		Engines:            map[string]EngineSnapshot{}}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	if len(s.counters.anomalies) > 0 {
		snap.Anomalies = map[string]int64{}
		for k, v := range s.counters.anomalies {
			snap.Anomalies[k] = v
		}
	}
	var seats, ne int
	for _, id := range s.groupOrder {
		g := s.groups[id]
		label := engineLabel(id)
		exitingMembers, exitingOldest := 0, time.Duration(0)
		for id, m := range g.threads {
			if !g.index.Exiting(id) {
				continue
			}
			exitingMembers++
			if age := now.Sub(m.exitingSince); age > exitingOldest {
				exitingOldest = age
			}
		}
		es := EngineSnapshot{ID: label, Seats: len(g.threads),
			ExitingThreads: exitingMembers, ExitingBlocks: g.index.ExitingBlocks(),
			ExitingOldestAgeS: exitingOldest.Seconds(),
			NEBlocks:          g.index.UniqueBlocks(),
			BudgetBlocks:      g.budgetBlocks, NumBlocks: g.numBlocks,
			RunBudgetBlocks: g.runBudgetBlocks, RunningBlocks: g.index.RunningBlocks(),
			RunningThreads: g.index.RunningSeats(),
			ColdInFlight:   g.coldInFlight, Served: g.counters.served,
			ColdCapTokens: g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize),
		}
		if g.blockSizeMismatch != nil {
			es.BlockSizeErr = g.blockSizeMismatch.Error()
		}
		if g.tier != nil {
			es.TierIndexBlocks, es.TierIndexCapBlocks = g.tier.Len(), g.tier.Cap()
		}
		if len(g.counters.exits) > 0 {
			es.Exits = map[string]int64{}
			for k, v := range g.counters.exits {
				es.Exits[k] = v
			}
		}
		snap.Engines[label] = es
		seats += es.Seats
		ne += es.NEBlocks
		snap.BudgetBlocks += g.budgetBlocks
	}
	snap.Seats, snap.NEBlocks = seats, ne
	snap.WaitersShared = len(s.waiters)
	if s.residency != nil {
		rs := &ResidencySnapshot{Placements: map[string]int64{},
			SeatResidentTokens: s.counters.seatResident, SeatColdTokens: s.counters.seatCold}
		for _, k := range placementOutcomes {
			rs.Placements[k] = s.counters.placements[k]
		}
		snap.Residency = rs
	}
	if s.cfg.OwnTierIndex {
		room := &RoomSnapshot{Held: s.counters.roomHeld, Forced: s.counters.roomForced,
			Rotations: s.counters.roomRotations, TierWrites: map[string]int64{}, SeatedEvicted: map[string]int64{}}
		for _, c := range tierCauses {
			room.TierWrites[c] = s.counters.tierWrites[c]
			room.SeatedEvicted[c] = s.counters.seatedEvicted[c]
		}
		if s.cfg.RoomDisplace {
			room.SeatedEvicted[tierCauseDisplaced] = s.counters.seatedEvicted[tierCauseDisplaced]
			for _, c := range tierCauses {
				if c != tierCauseSeat {
					room.SeatedEvicted[c+"-eligible"] = s.counters.seatedEvicted[c+"-eligible"]
				}
			}
		}
		room.QuantumDoneDeferred = s.counters.quantumDoneDeferred
		room.QuantumLastMarks = s.counters.quantumLastMarks
		room.HeadVictims = s.counters.headVictims
		if s.cfg.FairRotation {
			room.FairRemovals = map[string]int64{}
			for k, v := range s.counters.fairRemovals {
				room.FairRemovals[k] = v
			}
			room.FairTailSaved = s.counters.fairTailSaved
			room.SizeWaitOvertakes = s.counters.sizeWaitOvertakes
			room.Protected = s.counters.protected
			room.ProtectedBlocks = s.counters.protectedBlocks
			room.ProtectDeclined = s.counters.protectDeclined
			room.ProtectSkippedWaiting = s.counters.protectSkippedWaiting
			room.ProtectSkippedBlocks = s.counters.protectSkippedBlocks
			room.Claims = s.counters.claims
			room.SeatedPastClaim = s.counters.seatedPastClaim
			room.ClaimVictimsElsewhere = s.counters.claimVictimsElsewhere
			room.TokensGivenUp = s.counters.tokensGivenUp
		}
		room.Blockers = map[string]int64{}
		for _, b := range blockerStates {
			room.Blockers[b] = s.counters.roomBlocker[b]
		}
		snap.Room = room
	}
	return snap
}
