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

// Config is the group scheduler's parameter surface: exactly ALGORITHM-
// group.md §9. No other knobs exist; capacity and block size are
// measured (cache_config_info), never configured.
type Config struct {
	// GroupBudgetFraction is the cache share the group's unique blocks
	// may hold (θ_g). Default 0.95.
	GroupBudgetFraction float64 `json:"groupBudgetFraction"`
	// Rho is the payback multiple: a member may leave once its earned
	// cached tokens ≥ ρ × its measured entry cost. Default 4.
	Rho int `json:"rho"`
	// NMax is the residency turn cap before an unpaid member may leave
	// (n_max = 2ρ). 0 means derive from Rho.
	NMax int `json:"nMax"`
	// IdleSeconds is T_idle: a member silent longer than this gives up
	// its seat. Default 20 (> the trace's 15s gap cap).
	IdleSeconds float64 `json:"idleSeconds"`
	// TTLSeconds is the silence before a session leaves the system
	// (Unseat then Forget). Default 30 (≥ the p99 think gap).
	TTLSeconds float64 `json:"ttlSeconds"`
	// ColdEntryCapTokens staggers in-flight cold-entry prefill
	// (~one large session). Default 131072.
	ColdEntryCapTokens int64 `json:"coldEntryCapTokens"`
	// InFlightCap is the per-session in-flight bound. Default 64.
	InFlightCap int `json:"inFlightCap"`
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
	return nil
}

// ---- per-session scheduling state (§3, on top of the block index) ----

// member is the scheduling state of a seated session: the payback
// residency ledger and the seat bookkeeping. Block sets live in the
// BlockIndex's session records.
type member struct {
	id string
	// seatTime orders exits: earliest-seated leaves first.
	seatTime time.Time
	// residency payback ledger (§7, in tokens from usage reports)
	entryCost int64 // uncached tokens of the entry turn (measured)
	earned    int64 // cached tokens accumulated this residency
	resTurns  int   // settled dispatches this residency
	// inflight counts dispatched, unsettled requests.
	inflight int
	// lastActivity is the last request arrival or completion.
	lastActivity time.Time
	// coldEntry is the admission-time prefill estimate still counted
	// against the stagger cap; settled to the measured entry cost.
	coldEntry int64
	// waited tracks arrival of the current waiting head (entry order).
	waitingSince time.Time
	// newcomer is cleared at first seat.
	newcomer bool
}

// waiter is a non-member with a waiting request.
type waiter struct {
	id          string
	since       time.Time
	newcomer    bool
	coldEstimate int64 // blocks × block size (the stagger input)
}

// Scheduler is the group scheduler state machine over a BlockIndex.
// The 80% dispatch gate is NOT here: it stays exactly as deployed (the
// stack's saturation detector decides how much runs; this policy only
// decides who — the same split as the sibling plugins).
type Scheduler struct {
	mu sync.Mutex
	cfg Config
	// observers are the optional metric hooks (nil in tests).
	onExit  func(reason string, resTurns int)
	onEntry func(waitSeconds float64)
	onServe func()

	// index is nil until the first measured capacity arrives (M4: block
	// size from cache_config_info). Before that the scheduler serves
	// oldest-waiting members without budgeting — see the CP3 report's
	// ambiguity list.
	index        *BlockIndex
	budgetBlocks int // 0.95 × num_gpu_blocks
	blockSize    int

	members map[string]*member
	waiters map[string]*waiter
	// everSeated tracks sessions that have held a seat at least once
	// (the newcomer flag's other side — instance-scoped, so restarts and
	// tests are isolated).
	everSeated map[string]struct{}

	// coldInFlight is the running cold-entry prefill estimate.
	coldInFlight int64

	// metrics counters (§13)
	counters struct {
		exits       map[string]int64 // by reason: paid / idle / nmax / budget / engineIdle
		entryWaits  []float64        // seconds, newcomer seat waits
		resTurns    []int
		refillShare float64          // uncached / total prompt tokens at settles
		uncachedTok int64
		totalTok    int64
		served      int64
	}
}

// NewScheduler builds the state machine; the index attaches at the first
// measured capacity (SetCapacity).
func NewScheduler(cfg Config) *Scheduler {
	if err := cfg.validate(); err != nil {
		panic(err) // the plugin Factory surfaces this as a startup error
	}
	s := &Scheduler{cfg: cfg, members: map[string]*member{}, waiters: map[string]*waiter{},
		everSeated: map[string]struct{}{}}
	s.counters.exits = map[string]int64{}
	return s
}

// SetCapacity attaches the block index from measured engine facts. Once
// set, the budget is live. Called from the endpoint metrics path.
func (s *Scheduler) SetCapacity(blockSizeTokens, numBlocks int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index != nil {
		return nil // single-pool v0: first capacity wins (the sibling's pattern)
	}
	idx, err := NewBlockIndex(blockSizeTokens)
	if err != nil {
		return err
	}
	s.index = idx
	s.blockSize = blockSizeTokens
	s.budgetBlocks = int(s.cfg.GroupBudgetFraction * float64(numBlocks))
	return nil
}

// HasCapacity reports whether the measured engine facts have arrived.
func (s *Scheduler) HasCapacity() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.index != nil
}

// HashTokens computes the chained token-block keys for a request — the
// tier-A input (CP2b): 64-token blocks (the measured block size), each
// key chained from the previous, the cache salt folded into the first
// block. Structure-compatible with kvblock's ComputeBlockKeysFromTokens;
// no dependency on the indexer.
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

// ---- arrival (PreRequest) ----

// Arrive records a waiting request: the block keys enter the session's
// set (seated or not), and a non-member becomes a waiter.
func (s *Scheduler) Arrive(id string, reqID string, keys []Hash, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return // pre-capacity: tracked from the first measured facts
	}
	s.index.UpdateWaitingLocked(id, reqID, keys)
	if m, ok := s.members[id]; ok {
		m.lastActivity = now
		return
	}
	w, ok := s.waiters[id]
	if !ok {
		w = &waiter{id: id, newcomer: s.isNewcomerLocked(id)}
		s.waiters[id] = w
	}
	w.since = now
	w.coldEstimate = int64(len(keys)) * int64(s.blockSize)
}

// isNewcomerLocked reports whether the session has never held a seat.
// The block index's records persist across residencies (CP1), so
// "never seated" is scheduler state, not index state.
func (s *Scheduler) isNewcomerLocked(id string) bool {
	_, seen := s.everSeated[id]
	return !seen
}

// ---- dispatch (§5) ----

// Pick returns the request to dispatch next: the oldest waiting request
// of any non-exiting member under the in-flight cap. Non-members never
// run unseated (D10). Before budget enforcement and rotation run
// (demand-driven, eager — D3).
func (s *Scheduler) Pick(now time.Time) (reqID string, session string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return "", "", false
	}
	s.enforceBudgetLocked(now)
	s.rotateLocked(now)

	var best *member
	var bestSince time.Time
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
			continue // exiting members' new requests are held (§6.3)
		}
		if m.inflight >= s.cfg.InFlightCap {
			continue
		}
		if w := s.index.WaitingSince(m.id); !w.IsZero() && (best == nil || w.Before(bestSince)) {
			best, bestSince = m, w
		}
	}
	if best == nil {
		return "", "", false
	}
	reqID = s.index.OldestWaiting(best.id)
	if reqID == "" {
		return "", "", false
	}
	best.inflight++
	best.lastActivity = now
	s.counters.served++
	if s.onServe != nil {
		s.onServe()
	}
	return reqID, best.id, true
}

// ---- settle (ResponseBody: the payback ledger, §7) ----

// Settle records a completed (or terminated) dispatched request: the
// payback ledger from the usage report, the in-flight decrement (the
// finalizer runs for every termination cause), and the block-settle.
func (s *Scheduler) Settle(id, reqID string, keys []Hash, usage struct{ Prompt, Cached int }, natural bool, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return
	}
	m, seated := s.members[id]
	if seated {
		m.inflight = max(m.inflight-1, 0)
		m.lastActivity = now
	}
	s.index.SettleDispatchedLocked(id, reqID, keys)
	if !seated || !natural {
		return
	}
	// payback ledger (tokens from the engine's own report)
	uncached := int64(usage.Prompt - usage.Cached)
	if uncached < 0 {
		uncached = 0
	}
	if m.resTurns == 0 {
		m.entryCost = uncached // the entry turn's cold cost (measured)
		m.coldEntry = 0        // stagger settled to the actual
		s.coldInFlight = max(s.coldInFlight-uncached, 0)
	} else {
		m.earned += int64(usage.Cached)
	}
	m.resTurns++
	s.counters.resTurns = append(s.counters.resTurns, m.resTurns)
	s.counters.uncachedTok += uncached
	s.counters.totalTok += int64(usage.Prompt)
	if s.counters.totalTok > 0 {
		s.counters.refillShare = float64(s.counters.uncachedTok) / float64(s.counters.totalTok)
	}
}

// Cancel drops a waiting request (disconnect/eviction of a queued
// request; the finalizer path).
func (s *Scheduler) Cancel(id, reqID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return
	}
	_ = s.index.CancelLocked(id, reqID)
	s.pruneWaiterLocked(id)
}

// ---- rotation (§7): exit by eligibility, entry by order ----

// eligibleLocked reports whether a member may leave: nothing in flight
// AND (paid back | idle past T_idle | past n_max turns).
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

// rotateLocked seats waiters (demand-driven: only when a waiter could
// fit, possibly after an eligible member leaves) — D3, checked eagerly
// on every dispatch decision.
func (s *Scheduler) rotateLocked(now time.Time) {
	if len(s.waiters) == 0 || s.index == nil {
		return
	}
	for {
		head := s.entryHeadLocked()
		if head == nil {
			return
		}
		cost := s.index.FitBlocksLocked(s.index.SetBlocks(head.id))
		if s.index.UniqueBlocksNELocked()+cost <= s.budgetBlocks && s.coldAllows(head) {
			s.seatLocked(head, now)
			continue // seat the next head too while room remains
		}
		// the head does not fit: demand-driven exit via an ELIGIBLE
		// member (D9), then retry the head once; hold the line — nobody
		// behind the head jumps ahead
		if reason, victim := s.rotationVictimLocked(now); victim != nil {
			if victim.inflight > 0 {
				s.index.MarkExitingLocked(victim.id)
				s.counters.exits["exiting"]++
				return // drains on its own; the head waits
			}
			s.exitLocked(victim, reason, now)
			continue
		}
		return
	}
}

// entryHeadLocked is the next session to seat: newcomers first (D7),
// then the oldest waiting request; hold the line — if the head does not
// fit, nobody behind it jumps ahead (no skipping in this function; the
// budget check happens at the caller).
func (s *Scheduler) entryHeadLocked() *waiter {
	var head *waiter
	// newcomers first, oldest arrival among them
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

// coldAllows enforces the cold-entry stagger: in-flight cold prefill
// (admission estimates, settled at entry-turn completion) stays under
// the cap.
func (s *Scheduler) coldAllows(w *waiter) bool {
	if !w.newcomer {
		return true // re-entry: its blocks are partly warm
	}
	return s.coldInFlight+w.coldEstimate <= s.cfg.ColdEntryCapTokens
}

// rotationVictimLocked picks the member to evict for the head waiter
// (§7's demand-driven exit): ONLY eligible members (paid back, silent
// past T_idle, or past n_max — D9: no eviction of members that have not
// gotten their value), nothing waiting, earliest-seated first. The
// engine-idle rule (nothing running anywhere, a waiter blocked) is the
// liveness fallback.
func (s *Scheduler) rotationVictimLocked(now time.Time) (string, *member) {
	var best *member
	var bestReason string
	for _, m := range s.members {
		if s.index.ExitingLocked(m.id) {
			continue
		}
		reason, ok := s.eligibleLocked(m, now)
		if !ok {
			continue
		}
		if !s.index.WaitingSince(m.id).IsZero() {
			continue // prefer members with nothing waiting (§7 order)
		}
		if best == nil || m.seatTime.Before(best.seatTime) {
			best, bestReason = m, reason
		}
	}
	if best != nil {
		return bestReason, best
	}
	// engine-idle rule: nothing in flight anywhere, no member has a
	// waiting request (the engine would truly idle), and a waiter is
	// blocked -> the longest-idle member leaves. Without the no-waiting
	// condition this livelocks: evicting a member whose request is about
	// to dispatch, to seat a waiter that cannot co-fit, just swaps
	// membership forever (found by TestSmokeNonMembersNeverRunUnseated).
	idle := true
	for _, m := range s.members {
		if m.inflight > 0 || !s.index.WaitingSince(m.id).IsZero() {
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

// budgetVictimLocked is the §6 growth order: eligible members, then
// idle members (nothing in flight), then exiting (drain). Used ONLY by
// budget re-enforcement — rotation never evicts an un-eligible member
// for a waiter's room.
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
	return "", nil
}

// exitLocked removes a member (real exit; the exiting flag is only for
// paid members still in flight — §7's drain path). A session with a
// waiting request re-enters as a waiter (re-entry, not newcomer); a
// session with nothing waiting simply leaves.
func (s *Scheduler) exitLocked(m *member, reason string, now time.Time) {
	s.index.UnseatLocked(m.id)
	delete(s.members, m.id)
	s.counters.exits[reason]++
	if s.onExit != nil {
		s.onExit(reason, m.resTurns)
	}
	if since := s.index.WaitingSince(m.id); !since.IsZero() {
		s.waiters[m.id] = &waiter{id: m.id, since: since, coldEstimate: 0}
	}
}

// seatLocked seats a waiter (TrySeat under the budget already checked).
func (s *Scheduler) seatLocked(w *waiter, now time.Time) {
	ok, _ := s.index.TrySeatLocked(w.id, s.budgetBlocks)
	if !ok {
		return
	}
	wait := now.Sub(w.since).Seconds()
	delete(s.waiters, w.id)
	if w.newcomer {
		s.counters.entryWaits = append(s.counters.entryWaits, wait)
		if s.onEntry != nil {
			s.onEntry(wait)
		}
	}
	s.everSeated[w.id] = struct{}{}
	s.members[w.id] = &member{
		id:           w.id,
		seatTime:     now,
		lastActivity: now,
		newcomer:     w.newcomer,
		coldEntry:    w.coldEstimate,
	}
	if w.newcomer {
		s.coldInFlight += w.coldEstimate
	}
}

// ---- budget enforcement (§6, growth) ----

// enforceBudgetLocked re-checks the budget as members grow: when the
// non-exiting unique blocks exceed 0.95 × num_gpu_blocks, members leave
// in the D13 order: eligible, then idle, then exiting (held, drain).
func (s *Scheduler) enforceBudgetLocked(now time.Time) {
	if s.index == nil || s.budgetBlocks <= 0 {
		return
	}
	for s.index.UniqueBlocksNELocked() > s.budgetBlocks {
		reason, victim := s.budgetVictimLocked(now)
		if victim == nil {
			// nobody leaves: the overflow is transient (growth settles
			// or the engine-idle rule fires on the next rotation)
			return
		}
		if victim.inflight > 0 {
			// paid/still-in-flight members are marked exiting: their new
			// requests are held and they leave on drain (§6.3)
			s.index.MarkExitingLocked(victim.id)
			s.counters.exits["exiting"]++
			continue
		}
		s.exitLocked(victim, reason, now)
	}
}

// ---- lifecycle (§8): the ~1s ticker ----

// Tick runs the periodic work: T_idle seat release, TTL silence
// (Unseat then Forget — the CP1 condition), and rotation.
func (s *Scheduler) Tick(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return
	}
	// D13 between dispatches: growth is re-enforced on the ticker as
	// well as on every dispatch decision (a growing member can overflow
	// the budget while nothing dispatches).
	s.enforceBudgetLocked(now)
	// T_idle: a silent member with nothing waiting or in flight gives up
	// its seat (handled by eligibility at rotation; here we only release
	// when there are waiters — demand-driven. Absolute idleness is the
	// TTL path below.)
	// TTL: silence with nothing waiting or in flight -> leave the system
	for id, m := range s.members {
		if now.Sub(m.lastActivity) > time.Duration(s.cfg.TTLSeconds*float64(time.Second)) &&
			m.inflight == 0 && s.index.WaitingSince(id).IsZero() {
			s.exitLocked(m, "ttl", now)
			s.index.ForgetLocked(id)
		}
	}
	for id, w := range s.waiters {
		if now.Sub(w.since) > time.Duration(s.cfg.TTLSeconds*float64(time.Second)) &&
			s.index.WaitingSince(id).IsZero() {
			delete(s.waiters, id)
			s.index.ForgetLocked(id)
			delete(s.everSeated, id)
		}
	}
	s.rotateLocked(now)
}

// pruneWaiterLocked drops a waiter with no waiting requests left.
func (s *Scheduler) pruneWaiterLocked(id string) {
	if s.index == nil || !s.index.WaitingSince(id).IsZero() {
		return
	}
	if _, still := s.waiters[id]; still {
		delete(s.waiters, id)
	}
}

// ---- introspection for metrics and tests ----

// Snapshot returns the legible state (§13 metrics substrate).
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
}

func (s *Scheduler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := Snapshot{Seats: len(s.members), Waiters: len(s.waiters),
		Exits: map[string]int64{}, BudgetBlocks: s.budgetBlocks}
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
	snap.Served = s.counters.served
	return snap
}

// Sessions returns the tracked-session count (the record-leak test).
func (s *Scheduler) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index == nil {
		return 0
	}
	return s.index.TrackedCount()
}

var _ = sort.Float64s
