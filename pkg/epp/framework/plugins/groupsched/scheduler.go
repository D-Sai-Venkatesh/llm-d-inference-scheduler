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

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ---- §9 parameters (the only knobs; spec defaults) ----

// Config is the group scheduler's parameter surface: ALGORITHM-group.md
// §9 plus BlockSizeTokens, a WIRING FACT (not a tuning knob): it must
// equal the token processor's block size so the index's unit matches the
// engine's. Measured capacity (num_gpu_blocks) arrives from the endpoint
// metrics; a mismatch with the measured CacheBlockSize fails loudly.
type Config struct {
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`
	Rho                 int     `json:"rho"`
	NMax                int     `json:"nMax"`
	IdleSeconds         float64 `json:"idleSeconds"`
	TTLSeconds          float64 `json:"ttlSeconds"`
	ColdEntryCapTokens  int64   `json:"coldEntryCapTokens"`
	InFlightCap         int     `json:"inFlightCap"`
	// BlockSizeTokens is the engine's KV block size in tokens (the token
	// processor's value; 64 on the campaign stacks, M4). Required > 0.
	BlockSizeTokens int `json:"blockSizeTokens"`
}

func defaultConfig() Config {
	return Config{
		GroupBudgetFraction: 0.95,
		Rho:                 4,
		NMax:                0, // -> 2ρ
		IdleSeconds:         20,
		TTLSeconds:          30,
		ColdEntryCapTokens:  131072,
		InFlightCap:         64,
		BlockSizeTokens:     64,
	}
}

func (c Config) nMax() int {
	if c.NMax > 0 {
		return c.NMax
	}
	return 2 * c.Rho
}

func (c Config) validate() error {
	if c.GroupBudgetFraction <= 0 || c.GroupBudgetFraction > 1 {
		return fmt.Errorf("groupBudgetFraction must be in (0,1], got %v", c.GroupBudgetFraction)
	}
	if c.Rho < 1 {
		return fmt.Errorf("rho must be >= 1, got %v", c.Rho)
	}
	if c.IdleSeconds <= 0 || c.TTLSeconds <= 0 {
		return fmt.Errorf("idleSeconds and ttlSeconds must be > 0")
	}
	if c.ColdEntryCapTokens <= 0 || c.InFlightCap <= 0 {
		return fmt.Errorf("coldEntryCapTokens and inFlightCap must be > 0")
	}
	if c.BlockSizeTokens <= 0 {
		return fmt.Errorf("blockSizeTokens is required (the token processor's block size; got %d)", c.BlockSizeTokens)
	}
	return nil
}

// ---- the queue-head snapshot (S0-1: waiting state comes from the
// flow-control queues at Pick time, not from PreRequest) ----

// Head is one waiting queue head as seen at Pick: the session (agent
// identity), the request id, the enqueue time, and the wire size (the
// conservative-cost input).
type Head struct {
	Session   string
	ReqID     string
	Arrival   time.Time
	SizeBytes int64
}

// ---- per-session scheduling state (§3) ----

type member struct {
	id string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// residency payback ledger (§7, tokens from usage reports)
	entryCost int64
	earned    int64
	resTurns  int
	// inflight counts dispatched, unsettled requests.
	inflight int
	// lastActivity is the last arrival, dispatch or completion.
	lastActivity time.Time
	// coldEntry is this member's outstanding cold-entry charge in TOKENS
	// against the stagger cap; released EXACTLY once (S0-2) at the first
	// of: prefill complete, any termination, unseat.
	coldEntry int64
	coldDone  bool
	// entryRecorded marks that the entry turn's cost has been measured
	// (R4: a fully warm entry has uncached 0, which the old
	// entryCost==0&&earned==0 test misread as "still the entry turn").
	entryRecorded bool
	// lastSizeBytes is the last known wire size (the estimate delta's
	// baseline; updated at PreRequest with the exact size).
	lastSizeBytes int64
	// pendingBlocks is the seat-time admission ESTIMATE still counted
	// against the budget; converts to exact blocks at PreRequest
	// (Dispatched) and releases at exit. Without it a newcomer seats at
	// index-cost 0 and the budget understates until the exact keys land.
	pendingBlocks int64
	// waitingSince is the member's oldest queued head (B5: kept oldest).
	waitingSince time.Time
}

type waiter struct {
	id        string
	since     time.Time // the OLDEST arrival (B5)
	newcomer  bool
	// coldTokens is the entry's cold-prefill estimate in TOKENS
	// (R1: one unit everywhere — blocks x blockSize).
	coldTokens int64
}

// Scheduler is the group scheduler state machine over a BlockIndex.
// The 80% dispatch gate stays in the stack's saturation detector.
type Scheduler struct {
	mu  sync.Mutex
	cfg Config
	// observers are the optional metric hooks (nil in tests).
	onExit  func(reason string, resTurns int)
	onEntry func(waitSeconds float64)
	onServe func()

	index        *BlockIndex // attached at startup (block size from config)
	budgetBlocks int         // 0 until num_gpu_blocks arrives (metrics)
	numBlocks    int
	blockSize    int
	blockSizeMismatch error

	members   map[string]*member
	waiters   map[string]*waiter
	everSeated map[string]struct{}
	// lastSeen is the per-record activity time for the silence GC (R2):
	// updated at Dispatched and Settle; a headless, unseated record
	// older than TTL is forgotten.
	lastSeen map[string]time.Time

	coldInFlight int64
	// pendingTotal is the sum of seated members' pendingBlocks (the
	// budget's estimate component).
	pendingTotal int64

	counters struct {
		exits      map[string]int64
		entryWaits []float64
		resTurns   []int
		refillShare float64
		uncachedTok int64
		totalTok    int64
		served      int64
		fcfsServed  int64
	}
}

// NewScheduler builds the state machine. The index attaches NOW (A5:
// block size from the token processor's config, known at startup).
func NewScheduler(cfg Config) (*Scheduler, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	idx, err := NewBlockIndex(cfg.BlockSizeTokens)
	if err != nil {
		return nil, err
	}
	s := &Scheduler{cfg: cfg, index: idx, blockSize: cfg.BlockSizeTokens,
		members: map[string]*member{}, waiters: map[string]*waiter{},
		everSeated: map[string]struct{}{}, lastSeen: map[string]time.Time{}}
	s.counters.exits = map[string]int64{}
	return s, nil
}

// SetNumBlocks attaches the measured capacity (num_gpu_blocks from the
// endpoint metrics). A measured block size that differs from the config
// value fails loudly (recorded; HasCapacity stays until it matches).
func (s *Scheduler) SetNumBlocks(blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if blockSizeTokens > 0 && blockSizeTokens != s.blockSize {
		s.blockSizeMismatch = fmt.Errorf(
			"groupsched: measured CacheBlockSize %d != configured BlockSizeTokens %d (the token processor's value); refusing the capacity",
			blockSizeTokens, s.blockSize)
		return s.blockSizeMismatch
	}
	if s.numBlocks > 0 || numBlocks <= 0 {
		return nil // first capacity wins (single-pool v0)
	}
	s.numBlocks = numBlocks
	s.budgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	return nil
}

// HasCapacity reports whether the measured num_gpu_blocks has arrived.
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.numBlocks > 0
}

// BlockSizeMismatch reports the loud-failure state, if any.
func (s *Scheduler) BlockSizeMismatch() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.blockSizeMismatch
}

// HashTokens computes the chained token-block keys — the tier-A input
// (CP2b): blockSizeTokens-token blocks, each key chained from the
// previous, the cache salt folded into the first block.
func (s *Scheduler) HashTokens(tokens []uint32, cacheSalt string, blockSizeTokens int) []Hash {
	if blockSizeTokens <= 0 {
		blockSizeTokens = s.blockSize
	}
	if blockSizeTokens <= 0 || len(tokens) == 0 {
		return nil
	}
	n := len(tokens) / blockSizeTokens
	out := make([]Hash, 0, n)
	var parent uint64
	salt := sha256.Sum256([]byte(cacheSalt))
	saltSeed := binary.LittleEndian.Uint64(salt[:8])
	for i := 0; i < n; i++ {
		h := sha256.New()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], saltSeed^parent)
		h.Write(b[:])
		for j := i * blockSizeTokens; j < (i+1)*blockSizeTokens; j++ {
			binary.LittleEndian.PutUint32(b[:4], tokens[j])
			h.Write(b[:4])
		}
		parent = binary.LittleEndian.Uint64(h.Sum(nil)[:8])
		out = append(out, Hash(parent))
	}
	return out
}

// ---- Pick (§5): the waiting set is the queue-head snapshot ----

// PickFromHeads runs one dispatch decision against the current queue
// heads: sync the waiter view, enforce the budget, rotate, then serve
// the oldest queued head of any non-exiting member under the in-flight
// cap. Returns the index into heads of the chosen request, or -1.
func (s *Scheduler) PickFromHeads(heads []Head, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return -1
	}
	s.syncWaitersLocked(heads, now)
	s.enforceBudgetLocked(now)
	s.rotateLocked(now)

	best := -1
	var bestArrival time.Time
	for i, h := range heads {
		if h.Session == "" {
			continue // anonymous traffic is not tracked (passes through last)
		}
		m, seated := s.members[h.Session]
		if !seated || s.index.ExitingLocked(h.Session) {
			continue
		}
		if m.inflight >= s.cfg.InFlightCap {
			continue
		}
		if best < 0 || h.Arrival.Before(bestArrival) {
			best, bestArrival = i, h.Arrival
		}
	}
	if best >= 0 {
		s.members[heads[best].Session].inflight++
		s.members[heads[best].Session].lastActivity = now
		s.counters.served++
		if s.onServe != nil {
			s.onServe()
		}
		return best
	}
	// R9: no member head is dispatchable — serve the oldest ANONYMOUS
	// head (FCFS pass-through, the siblings' behaviour; anonymous
	// traffic is never tracked, never seated, never starved).
	for i, h := range heads {
		if h.Session == "" {
			return i
		}
	}
	return -1
}

// FCFSPick returns the oldest head (the pre-capacity pass-through, A5).
func FCFSPick(heads []Head) int {
	best := -1
	var bestArrival time.Time
	for i, h := range heads {
		if best < 0 || h.Arrival.Before(bestArrival) {
			best, bestArrival = i, h.Arrival
		}
	}
	return best
}

// FCFSServed counts pre-capacity dispatches (the gauge substrate).
func (s *Scheduler) FCFSServed() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counters.fcfsServed++
	return s.counters.fcfsServed
}

// syncWaitersLocked reconciles the waiter view with the queue-head
// snapshot: new sessions become waiters (estimates from history + wire
// size), vanished sessions leave, member waiting-since tracks the oldest
// queued head (the victim preference's input).
func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
	present := map[string]time.Time{}
	for _, h := range heads {
		if h.Session == "" {
			continue
		}
		present[h.Session] = h.Arrival
		if m, seated := s.members[h.Session]; seated {
			if m.waitingSince.IsZero() || h.Arrival.Before(m.waitingSince) {
				m.waitingSince = h.Arrival
			}
			m.lastActivity = now
			continue
		}
		w, ok := s.waiters[h.Session]
		if !ok {
			w = &waiter{id: h.Session, newcomer: s.isNewcomerLocked(h.Session)}
			s.waiters[h.Session] = w
		}
		// B5: keep the OLDEST arrival across fan-out re-arrivals
		if w.since.IsZero() || h.Arrival.Before(w.since) {
			w.since = h.Arrival
		}
		w.coldTokens = s.estimateCostLocked(h.Session, h.SizeBytes) * int64(s.blockSize) // tokens (R1)
	}
	// vanished: members clear their waiting marker; waiters leave
	for id, m := range s.members {
		if _, ok := present[id]; !ok {
			m.waitingSince = time.Time{}
		}
	}
	for id := range s.waiters {
		if _, ok := present[id]; !ok {
			delete(s.waiters, id)
		}
	}
}

// estimateCostLocked is the conservative admission cost of a QUEUED
// request (S0-1 b/c): the session's persisted history blocks not held by
// the non-exiting group, plus the wire-size DELTA over the last known
// body (bytes/4/blockSize, zero sharing assumed). A session with no
// history costs its full wire size the same way.
func (s *Scheduler) estimateCostLocked(id string, sizeBytes int64) int64 {
	hist := s.index.SetBlocks(id) // persisted across residencies (CP1)
	known := int64(0)
	for _, h := range hist {
		if !s.index.HasNELocked(h) {
			known++
		}
	}
	// the baseline is the session's last DISPATCHED body size (the index
	// record, updated at PreRequest) — never the current head's size, or
	// the second sight of the same head would estimate zero new blocks
	// (found by the integration test: w seated on the retry pick)
	lastBytes := s.index.LastSizeBytesLocked(id)
	delta := sizeBytes - lastBytes
	if delta < 0 {
		delta = 0
	}
	return known + delta/4/int64(s.blockSize)
}

// ---- PreRequest (post-dispatch): the exact tier-A correction ----

// Dispatched records a released request at PreRequest: the exact keys
// replace the estimate (D13 absorbs the error at the next enforcement),
// in-flight and the cold charge are taken, and the size baseline moves.
func (s *Scheduler) Dispatched(id, reqID string, keys []Hash, sizeBytes int64, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return
	}
	s.index.SetSizeLocked(id, sizeBytes) // the estimate baseline
	s.index.SettleDispatchedLocked(id, reqID, keys)
	if m, ok := s.members[id]; ok {
		m.lastActivity = now
		m.lastSizeBytes = sizeBytes
		if m.pendingBlocks > 0 {
			s.pendingTotal -= m.pendingBlocks
			m.pendingBlocks = 0 // the exact keys are in NE now (D13 reconciles)
		}
		if !m.coldDone && m.coldEntry > 0 {
			// refine the charge to the exact token count (R1: one unit)
			s.coldInFlight = max(s.coldInFlight-m.coldEntry, 0)
			m.coldEntry = int64(len(keys)) * int64(s.blockSize)
			s.coldInFlight += m.coldEntry
		}
	}
}

// Chunk is the first response chunk: prefill complete — the cold-entry
// charge is released HERE (B3: not at end of decode), exactly once.
func (s *Scheduler) Chunk(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseColdLocked(id)
}

func (s *Scheduler) releaseColdLocked(id string) {
	m, ok := s.members[id]
	if !ok || m.coldDone || m.coldEntry <= 0 {
		return
	}
	s.coldInFlight = max(s.coldInFlight-m.coldEntry, 0)
	m.coldEntry = 0
	m.coldDone = true
}

// ---- settle (ResponseBody at EndOfStream: the finalizer) ----

// Settle records a terminated dispatched request: the in-flight
// decrement for EVERY cause; n_max counts every settled turn (A8); the
// payback ledger only on natural completion.
func (s *Scheduler) Settle(id string, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSeen[id] = now
	m, seated := s.members[id]
	if seated {
		m.inflight = max(m.inflight-1, 0)
		m.lastActivity = now
		s.releaseColdLocked(id) // S0-2: any termination releases the charge
	}
	if !seated {
		return
	}
	m.resTurns++ // A8: every settled turn counts toward n_max
	// the settled request was the tracked queued head: clear the marker
	// (the next Pick re-establishes it if more heads remain — the TTL
	// path cannot see stale queue presence)
	m.waitingSince = time.Time{}
	// §6.3: an exiting member leaves when its in-flight requests drain
	if s.index.ExitingLocked(id) && m.inflight == 0 {
		s.exitLocked(m, "exiting-drain", now)
		return
	}
	if !natural {
		return
	}
	uncached := int64(usage.Prompt - usage.Cached)
	if uncached < 0 {
		uncached = 0
	}
	if !m.entryRecorded {
		m.entryCost = uncached // the entry turn's cold cost (measured; may be 0 when fully warm)
		m.entryRecorded = true
	} else {
		m.earned += int64(usage.Cached)
	}
	s.counters.resTurns = append(s.counters.resTurns, m.resTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(usage.Prompt)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// ---- rotation (§7) ----

func (s *Scheduler) eligibleLocked(m *member, now time.Time) (string, bool) {
	if m.inflight > 0 {
		return "", false
	}
	if m.resTurns >= 1 && m.earned >= int64(s.cfg.Rho)*max(m.entryCost, 1) {
		return "paid", true
	}
	if now.Sub(m.lastActivity) > time.Duration(s.cfg.IdleSeconds*float64(time.Second)) {
		return "idle", true
	}
	if m.resTurns >= s.cfg.nMax() {
		return "nmax", true
	}
	return "", false
}

func (s *Scheduler) rotateLocked(now time.Time) {
	if len(s.waiters) == 0 {
		return
	}
	for {
		head := s.entryHeadLocked()
		if head == nil {
			return
		}
		ne := int64(s.index.UniqueBlocksNELocked()) + s.pendingTotal
		estBlocks := head.coldTokens / int64(s.blockSize)
		if ne+estBlocks <= int64(s.budgetBlocks) && s.coldAllows(head) {
			s.seatLocked(head, now)
			continue
		}
		if reason, victim := s.rotationVictimLocked(now); victim != nil {
			if victim.inflight > 0 {
				s.index.MarkExitingLocked(victim.id)
				s.counters.exits["exiting"]++
				return
			}
			s.exitLocked(victim, reason, now)
			continue
		}
		return
	}
}

// entryHeadLocked: newcomers first (D7; oldest within the class), then
// oldest waiting (A4/A7: re-entrants keep their original since).
func (s *Scheduler) entryHeadLocked() *waiter {
	var head *waiter
	for _, w := range s.waiters {
		if !w.newcomer {
			continue
		}
		if head == nil || w.since.Before(head.since) || (w.since.Equal(head.since) && w.id < head.id) {
			head = w
		}
	}
	if head != nil {
		return head
	}
	for _, w := range s.waiters {
		if head == nil || w.since.Before(head.since) || (w.since.Equal(head.since) && w.id < head.id) {
			head = w
		}
	}
	return head
}

// coldAllows staggers ALL cold entries (B6: re-entries too — WEKA
// re-entries are fully cold), in TOKENS (R1). R11's rule: one cold entry
// is ALWAYS allowed — a session bigger than the cap must be seatable
// when nothing cold is in flight, or newcomers-first + hold-the-line
// deadlocks the queue on it forever.
func (s *Scheduler) coldAllows(w *waiter) bool {
	return s.coldInFlight == 0 || s.coldInFlight+w.coldTokens <= s.cfg.ColdEntryCapTokens
}

// rotationVictimLocked (§7 demand-driven exit, B4: no-waiting is a
// PREFERENCE not an exclusion — paid members with queued next turns
// must remain evictable): eligible members, sorted by (has-waiting,
// seatTime); the engine-idle rule as the liveness fallback.
func (s *Scheduler) rotationVictimLocked(now time.Time) (string, *member) {
	var best *member
	var bestReason string
	bestHasWaiting := false
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(m, now)
		if !ok {
			continue
		}
		hasWaiting := !m.waitingSince.IsZero()
		if best == nil ||
			(hasWaiting && !bestHasWaiting) ||
			(hasWaiting == bestHasWaiting && m.seatTime.Before(best.seatTime)) {
			best, bestReason, bestHasWaiting = m, reason, hasWaiting
		}
	}
	if best != nil {
		return bestReason, best
	}
	// engine-idle rule: no member in flight AND no member dispatchable
	// (A1's livelock guard) while a waiter is blocked
	idle := true
	for _, m := range s.members {
		if m.inflight > 0 || !m.waitingSince.IsZero() {
			idle = false
			break
		}
	}
	if idle {
		for _, m := range s.members {
			if s.index.ExitingLocked(m.id) {
				continue
			}
			if best == nil || m.lastActivity.Before(best.lastActivity) {
				best, bestReason = m, "engine-idle"
			}
		}
	}
	return bestReason, best
}

// budgetVictimLocked is the §6 growth order: eligible, then idle
// (nothing in flight), then the exiting class — the earliest-seated
// member EVEN MID-FLIGHT (the caller marks it exiting, which drops its
// NE contribution immediately; it leaves when its requests drain, §6.3).
// Budget enforcement only — rotation never evicts an un-eligible member.
func (s *Scheduler) budgetVictimLocked(now time.Time) (string, *member) {
	if reason, m := s.rotationVictimLocked(now); m != nil {
		return reason, m
	}
	var best *member
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) || m.inflight > 0 {
			continue
		}
		if best == nil || m.seatTime.Before(best.seatTime) {
			best = m
		}
	}
	if best != nil {
		return "budget-idle", best
	}
	// the exiting class: earliest-seated regardless of inflight
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
			continue
		}
		if best == nil || m.seatTime.Before(best.seatTime) {
			best = m
		}
	}
	if best != nil {
		return "budget-exiting", best
	}
	return "", nil
}

func (s *Scheduler) exitLocked(m *member, reason string, now time.Time) {
	s.releaseColdLocked(m.id) // S0-2: unseat releases the charge
	s.pendingTotal -= m.pendingBlocks
	m.pendingBlocks = 0
	s.index.UnseatLocked(m.id)
	delete(s.members, m.id)
	s.counters.exits[reason]++
	if s.onExit != nil {
		s.onExit(reason, m.resTurns)
	}
	// A7: a session with a queued head re-enters at its ORIGINAL
	// waiting-since (FIFO survives the exit)
	if !m.waitingSince.IsZero() {
		s.waiters[m.id] = &waiter{id: m.id, since: m.waitingSince,
			coldTokens: s.estimateCostLocked(m.id, m.lastSizeBytes) * int64(s.blockSize)}
	}
}

func (s *Scheduler) seatLocked(w *waiter, now time.Time) {
	// seat in the index first (history-based cost; the estimate-based
	// budget check already ran in rotate — D13 reconciles the difference)
	if ok, _ := s.index.TrySeatLocked(w.id, s.budgetBlocks); !ok {
		return
	}
	s.everSeated[w.id] = struct{}{}
	s.waitersdelete(w.id)
	m := &member{
		id:        w.id,
		seatTime:  now,
		lastActivity: now,
		lastSizeBytes: s.index.LastSizeBytesLocked(w.id),
		// the head that triggered this seating is still queued (the
		// serve happens after rotate): inherit its presence so the
		// engine-idle rule cannot evict a member that is about to
		// dispatch (A1's livelock class, entering via the fresh record's
		// zero waitingSince — found by TestNewcomerFirst)
		waitingSince: w.since,
	}
	// B6: ALL entries carry a cold charge (tokens, R1)
	m.coldEntry = w.coldTokens
	s.coldInFlight += m.coldEntry
	// the budget counts the admission estimate (blocks) until the exact
	// keys land at PreRequest
	m.pendingBlocks = w.coldTokens / int64(s.blockSize)
	s.pendingTotal += m.pendingBlocks
	if w.newcomer {
		wait := now.Sub(w.since).Seconds()
		s.counters.entryWaits = append(s.counters.entryWaits, wait)
		if s.onEntry != nil {
			s.onEntry(wait)
		}
	}
	s.members[w.id] = m
}

func (s *Scheduler) waitersdelete(id string) { delete(s.waiters, id) }

// ---- budget enforcement (§6, growth) ----

func (s *Scheduler) enforceBudgetLocked(now time.Time) {
	if s.budgetBlocks <= 0 {
		return // pre-capacity (A5): no budget yet
	}
	for int64(s.index.UniqueBlocksNELocked())+s.pendingTotal > int64(s.budgetBlocks) {
		reason, victim := s.budgetVictimLocked(now)
		if victim == nil {
			return
		}
		if victim.inflight > 0 {
			s.index.MarkExitingLocked(victim.id)
			s.counters.exits["exiting"]++
			continue
		}
		s.exitLocked(victim, reason, now)
	}
}

// ---- lifecycle (§8): the ~1s ticker ----

func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enforceBudgetLocked(now)
	for id, m := range s.members {
		if now.Sub(m.lastActivity) > time.Duration(s.cfg.TTLSeconds*float64(time.Second)) &&
			m.inflight == 0 && m.waitingSince.IsZero() {
			s.exitLocked(m, "ttl", now)
			s.index.ForgetLocked(id)
			delete(s.everSeated, id) // B7: prune consistently with the waiter path
		}
	}
	// R2: NO TTL on waiters — a waiter exists because its head was
	// present at the last sync; rotation waits are minutes. Headless,
	// unseated records are GC'd by lastSeen (silence, not queue age).
	s.gcSilentRecordsLocked(now)
	s.rotateLocked(now)
}

// gcSilentRecordsLocked forgets headless, unseated records whose last
// activity is older than the TTL (R2: silence, not queue age — a waiter
// with a live head is NEVER forgotten, however long it waits).
func (s *Scheduler) gcSilentRecordsLocked(now time.Time) {
	ttl := time.Duration(s.cfg.TTLSeconds * float64(time.Second))
	for id, seen := range s.lastSeen {
		if _, isMember := s.members[id]; isMember {
			continue
		}
		if _, isWaiter := s.waiters[id]; isWaiter {
			continue // has a live head (syncWaiters keeps waiters honest)
		}
		if now.Sub(seen) > ttl {
			s.index.ForgetLocked(id)
			delete(s.everSeated, id)
			delete(s.lastSeen, id)
		}
	}
}

func (s *Scheduler) isNewcomerLocked(id string) bool {
	_, seen := s.everSeated[id]
	return !seen
}

// ---- introspection ----

type Snapshot struct {
	Seats        int
	UniqueBlocks int
	NEBlocks     int
	BudgetBlocks int
	Waiters      int
	Exits        map[string]int64
	MeanResTurns float64
	EntryWaitP50 float64
	RefillShare  float64
	Served       int64
	FCFSServed   int64
}

func (s *Scheduler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := Snapshot{Seats: len(s.members), Waiters: len(s.waiters),
		Exits: map[string]int64{}, BudgetBlocks: s.budgetBlocks,
		Served: s.counters.served, FCFSServed: s.counters.fcfsServed}
	for k, v := range s.counters.exits {
		snap.Exits[k] = v
	}
	if s.index != nil {
		snap.UniqueBlocks = s.index.UniqueBlocks()
		snap.NEBlocks = s.index.UniqueBlocksNE()
	}
	if n := len(s.counters.resTurns); n > 0 {
		sum := 0
		for _, t := range s.counters.resTurns {
			sum += t
		}
		snap.MeanResTurns = float64(sum) / float64(n)
	}
	if n := len(s.counters.entryWaits); n > 0 {
		w := append([]float64(nil), s.counters.entryWaits...)
		sort.Float64s(w)
		snap.EntryWaitP50 = w[n/2]
	}
	snap.RefillShare = s.counters.refillShare
	return snap
}

// Sessions returns the tracked-session count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.index.TrackedCount()
}

// Seated reports membership (tests).
func (s *Scheduler) Seated(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.members[id]
	return ok
}
