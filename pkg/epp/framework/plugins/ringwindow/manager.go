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

package ringwindow

import (
	"sort"
	"sync"
	"time"
)

// reservationTTL bounds an admission reservation whose session never reached
// PreRequest, such as a request cancelled between dispatch and binding. It is
// ledger consistency machinery (copied from thunderagent), not a scheduling
// constant: it never decides who runs, only guards against double-charging
// room between Pick and PreRequest.
const reservationTTL = 5 * time.Second

// gapFactorPerServe is the worst gap between a member's turns, in serves per
// member, under the round-counter serve order — 2.7, measured on the block
// model (V0-REVIEW code review, issue 1: max gap 27 serves at W=10). It is a
// structural property of the order, not a tuned number.
const gapFactorPerServe = 2.7

// session is one agent trajectory on the ring, identified by its
// agent-identity session id. All fields are guarded by ringManager.mu.
type session struct {
	id string
	// endpointID is the pod the session last ran on; used only to learn
	// capacity in the single-pool v0.
	endpointID string
	// committedTokens is usage.total_tokens of the last completed turn:
	// the measured footprint of decision D2.
	committedTokens int64
	// inflightTokens is the sum of the estimates of the session's turns in
	// flight.
	inflightTokens int64
	// A turn Pick dispatched but PreRequest has not yet charged counts
	// reservedTokens until reservedUntil, so the next Pick cycle does not
	// admit into the same room twice.
	reservedTokens float64
	reservedUntil  time.Time
	// lastFinish is when the session's last turn completed: the release
	// order. vLLM evicts the oldest-released blocks first, so the demotion
	// victim is the member with the oldest lastFinish.
	lastFinish time.Time
	// lastTurnDur is the wall time of the session's last completed turn:
	// the per-session service prediction of decision D10.
	lastTurnDur time.Duration
	turnCount   int64
	// seq is the ring position: assigned on first join and re-assigned on
	// demotion and on rejoin after a sweep, so a demoted session rejoins
	// the outsider line at the tail rather than at the head.
	seq int64
	// member labels the session as inside the window. Labels govern
	// scheduling only; they never pin or protect blocks.
	member       bool
	lastActivity time.Time
}

// effectiveInflight is the in-flight estimate including a live reservation.
func (s *session) effectiveInflight(now time.Time) float64 {
	f := float64(s.inflightTokens)
	if now.Before(s.reservedUntil) {
		f += s.reservedTokens
	}
	return f
}

// size is the session's KV footprint in tokens: the committed total or the
// in-flight estimate, whichever is larger (each turn resends the whole
// history).
func (s *session) size(now time.Time) float64 {
	if f := s.effectiveInflight(now); f > float64(s.committedTokens) {
		return f
	}
	return float64(s.committedTokens)
}

// dispatchable reports whether the session's blocks are free to be evicted:
// no turn in flight and no live reservation. Demotion candidates skip
// sessions that are not dispatchable — the engine cannot evict their pinned
// blocks, so demoting them would write a label the cache contradicts.
func (s *session) dispatchable(now time.Time) bool {
	return s.effectiveInflight(now) == 0
}

// ringManager is the ledger, ring, window and scan state shared by all of the
// ring-window plugin's hooks. v0 assumes a single serving pool (single
// replica): one ring, one budget. Capacity is learned from the endpoints
// seen at PreRequest; the pool capacity is the most recently seen one.
type ringManager struct {
	mu sync.Mutex

	sessions map[string]*session
	nextSeq  int64

	// capacity is the pool's KV capacity in tokens (single-pool v0);
	// capacityFromConfig records that it is still the configured fallback
	// rather than a reported cache_config_info.
	capacity           float64
	capacityFromConfig bool
	// endpoints caches the last-seen capacity per endpoint id.
	endpoints map[string]float64

	// Scan state (V0-REVIEW code review, issue 1): the member order is a
	// pure function of a round counter. Round r serves the member
	// positions [r, r+1, …, r+L−1] (ring order, wrapping) and then the
	// interior positions [r+L−2 … r+1] in reverse — a rotating start with
	// no endpoint re-serve. This keeps the engine's next LRU victim among
	// the members the cursor reaches last (the alignment D1 exists for),
	// spreads the turnaround burden off p10, and every member is served
	// equally over L rounds. slot is the next index to consider in the
	// current round's sequence; it advances only on serves, never on
	// skips.
	round int64
	slot  int

	// turnDurSum/turnDurCount maintain the running mean turn duration
	// since plugin start (no window, no constant): the ŝ used by the
	// elastic D_eff and the D-width.
	turnDurSum   time.Duration
	turnDurCount int64

	// Telemetry state, sampled by the metrics collector.
	lastNWaiting int
	lastDEff     time.Duration
	lastWidthD   int
	lastBinding  string
	lastRotation time.Time

	// serveHistory is the recent serve order (debug; last 2L serves).
	serveHistory []string

	// rotationIntervals is the recent time between rotations (debug).
	rotationIntervals []time.Duration

	cfg Config

	// counters for metrics and state dumps.
	counters managerCounters
}

type managerCounters struct {
	rotations    int64
	guests       int64
	oversized    int64
	noVictims    int64
	memberServes int64
	demotions    map[string]int64
}

func newRingManager(cfg Config) *ringManager {
	return &ringManager{
		sessions:  make(map[string]*session),
		endpoints: make(map[string]float64),
		// The configured capacity is the fallback budget until an
		// endpoint reports its real cache_config_info (thunderagent's
		// pattern): without it the first dispatches would see a budget
		// of zero and take the oversized-guest path. The fallback must
		// be asserted against the pod's real capacity (it can be 3–6×
		// off at TP2/TP4).
		capacity:           float64(cfg.CapacityTokens),
		capacityFromConfig: true,
		cfg:                cfg,
		counters:           managerCounters{demotions: make(map[string]int64)},
	}
}

// ensureSession returns the ledger entry for id, creating it at the ring tail
// on first sight (or on rejoin after a sweep removed it).
func (m *ringManager) ensureSessionLocked(id string) *session {
	s, ok := m.sessions[id]
	if !ok {
		m.nextSeq++
		s = &session{id: id, seq: m.nextSeq, lastActivity: time.Now()}
		m.sessions[id] = s
	}
	return s
}

// members returns the member set in ring (seq) order.
func (m *ringManager) membersLocked() []*session {
	var out []*session
	for _, s := range m.sessions {
		if s.member {
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].seq < out[j].seq })
	return out
}

// memberSum is Σ member footprints, idle members included (decision D3).
func (m *ringManager) memberSumLocked(now time.Time) float64 {
	var sum float64
	for _, s := range m.membersLocked() {
		sum += s.size(now)
	}
	return sum
}

// budgetTokens is θ·C.
func (m *ringManager) budgetTokensLocked() float64 {
	return m.cfg.Theta * m.capacity
}

// meanTurnDur is the running mean completed-turn duration; 0 before the
// first response.
func (m *ringManager) meanTurnDurLocked() time.Duration {
	if m.turnDurCount == 0 {
		return 0
	}
	return m.turnDurSum / time.Duration(m.turnDurCount)
}

// inflightCountLocked is ĉ: the measured concurrency — sessions with a turn
// in flight (or a live reservation). It divides the elastic D_eff and the
// D-width, because with ĉ turns running concurrently one serve completes
// every ŝ/ĉ, not every ŝ.
func (m *ringManager) inflightCountLocked() int {
	now := time.Now()
	n := 0
	for _, s := range m.sessions {
		if s.effectiveInflight(now) > 0 {
			n++
		}
	}
	return n
}

// widthDLocked is the D-width of the window: a member's worst gap under the
// round-counter order is ~gapFactorPerServe serves per member, each taking
// ŝ/ĉ wall time, so W ≤ d·ĉ/(gapFactorPerServe·ŝ). The width applies only
// when the operator's D is the binding term of the deadline: when the
// elastic floor is active (ŝ·N̂/ĉ > d, including D=0), the floor already
// paces rotation at one pass per round and capping the width on top of it
// would double-count — the D-width degenerates to N̂/2.7 with no deadline
// information, and at shallow queues it collapses the window to one member,
// discarding locality exactly where it is free (V0-REVIEW verification
// round, residuals 1+2). With the floor active the budget binds.
func (m *ringManager) widthDLocked(d time.Duration) int {
	ŝ := m.meanTurnDurLocked()
	if ŝ <= 0 || d <= 0 {
		return 1 << 30
	}
	ĉ := m.inflightCountLocked()
	if ĉ < 1 {
		ĉ = 1
	}
	if stretch := ŝ * time.Duration(m.lastNWaiting) / time.Duration(ĉ); stretch > d {
		return 1 << 30
	}
	w := int(float64(d) * float64(ĉ) / (gapFactorPerServe * float64(ŝ)))
	if w < 1 {
		w = 1
	}
	return w
}

// fitsLocked reports whether a session of est tokens can join the window
// under both widths: the budget width (Σ + est ≤ θ·C) and the D-width
// (member count + 1 ≤ W_D, operator's raw D). It records which width binds,
// for telemetry.
func (m *ringManager) fitsLocked(est float64, d time.Duration, now time.Time) bool {
	if m.memberSumLocked(now)+est > m.budgetTokensLocked() {
		m.lastBinding = "budget"
		return false
	}
	if len(m.membersLocked())+1 > m.widthDLocked(d) {
		m.lastBinding = "d"
		return false
	}
	return true
}

// demotionVictims returns the demotion candidates — members whose turn is
// not in flight (a running member's blocks are pinned; the engine cannot
// evict them, so demoting it would write a label the cache contradicts) —
// ordered by release time, oldest first: the order in which the engine
// evicts (decision D7's completion-order primary). A member with a turn
// waiting IS a candidate: its blocks age in the free queue while it queues,
// and with re-seq on demotion the ping-pong the old skip guarded against
// cannot happen (V0-REVIEW code review, issues 5 and 6).
func (m *ringManager) demotionVictimsLocked(now time.Time) []*session {
	var victims []*session
	for _, s := range m.membersLocked() {
		if !s.dispatchable(now) {
			continue
		}
		victims = append(victims, s)
	}
	sort.Slice(victims, func(i, j int) bool {
		return victims[i].lastFinish.Before(victims[j].lastFinish)
	})
	return victims
}

// demoteLocked removes a member label and re-seqs the session to the ring
// tail, so a demoted session rejoins the outsider line behind everyone.
func (m *ringManager) demoteLocked(s *session, reason string) {
	s.member = false
	m.nextSeq++
	s.seq = m.nextSeq
	m.counters.demotions[reason]++
}

// demoteUntilFit demotes members in eviction order until a session of est
// tokens fits both widths, or no demotable member remains. reason labels the
// demotion counter. Returns whether it fits now.
func (m *ringManager) demoteUntilFitLocked(est float64, d time.Duration, reason string, now time.Time) bool {
	for !m.fitsLocked(est, d, now) {
		victims := m.demotionVictimsLocked(now)
		if len(victims) == 0 {
			return m.fitsLocked(est, d, now)
		}
		m.demoteLocked(victims[0], reason)
	}
	return true
}

// admitWithDemotion is the join path shared by rotation (Pick step 1) and
// the never-idle guest (Pick step 3): demote until the candidate fits, then
// join it and re-anchor the scan just past it. Three outcomes:
//   - joined: the candidate is a member;
//   - oversized: the candidate alone exceeds the whole budget — it runs as
//     a non-member and the window clears to rebuild by arrival (decision D6);
//   - noVictims: every member is running (nothing demotable) — the candidate
//     runs as a non-member and the window stays intact (V0-REVIEW code
//     review, issue 7: this is not the oversized case and must not be
//     counted as it).
func (m *ringManager) admitWithDemotion(id string, est float64, d time.Duration, reason string, now time.Time) (joined, oversized, noVictims bool) {
	s := m.ensureSessionLocked(id)
	if !m.demoteUntilFitLocked(est, d, reason, now) {
		if est > m.budgetTokensLocked() {
			for _, mem := range m.membersLocked() {
				m.demoteLocked(mem, "clear")
			}
			m.counters.oversized++
			return false, true, false
		}
		m.counters.noVictims++
		return false, false, true
	}
	s.member = true
	m.reanchorToLocked(id)
	return true, false, false
}

// rotate is Pick step 1: an outsider's D-clock has expired, so the window
// rotates one place — the outsider joins (demoting in eviction order until
// the budget covers it) and is dispatched first in the new window. Members
// never trigger rotation (decision D4).
func (m *ringManager) rotate(id string, est float64, d time.Duration, now time.Time) (joined, oversized, noVictims bool) {
	m.counters.rotations++
	if !m.lastRotation.IsZero() {
		m.rotationIntervals = append(m.rotationIntervals, now.Sub(m.lastRotation))
		if len(m.rotationIntervals) > 64 {
			m.rotationIntervals = m.rotationIntervals[1:]
		}
	}
	m.lastRotation = now
	return m.admitWithDemotion(id, est, d, "rotate", now)
}

// guest is Pick step 3's non-fitting path: the GPU must not idle while no
// member waits, so the head outsider runs even though it does not fit
// (decision D6, the weak sense).
func (m *ringManager) guest(id string, est float64, d time.Duration, now time.Time) (joined, oversized, noVictims bool) {
	m.counters.guests++
	return m.admitWithDemotion(id, est, d, "guest", now)
}

// joinQuiet admits the head outsider when it fits without demoting anyone.
func (m *ringManager) joinQuiet(id string, est float64, d time.Duration, now time.Time) bool {
	if !m.fitsLocked(est, d, now) {
		return false
	}
	s := m.ensureSessionLocked(id)
	s.member = true
	m.reanchorToLocked(id)
	return true
}

// roundSeqFor is the serve sequence of one round: the member positions in
// ring order starting at round mod L, then the interior positions in
// reverse. L + (L−2) slots; for L < 3 the backward phase is empty. The
// double-serve at each round boundary (…C B | B C D…) is intentional: it is
// what makes the counts equal over L rounds, and the second serve is a free
// hit because that member's blocks are the youngest in the queue.
func roundSeqFor(round int64, L int) []int {
	r := int(round % int64(L))
	if r < 0 {
		r += L
	}
	seq := make([]int, 0, 2*L-2)
	for k := 0; k < L; k++ {
		seq = append(seq, (r+k)%L)
	}
	for k := L - 2; k >= 1; k-- {
		seq = append(seq, (r+k)%L)
	}
	return seq
}

// reanchorToLocked moves the scan to just past the given member's slot in
// the current round (used when a joiner was just dispatched).
func (m *ringManager) reanchorToLocked(id string) {
	members := m.membersLocked()
	L := len(members)
	if L <= 1 {
		return
	}
	pos := -1
	for i, s := range members {
		if s.id == id {
			pos = i
			break
		}
	}
	if pos < 0 {
		return
	}
	seq := roundSeqFor(m.round, L)
	for i, p := range seq {
		if p == pos {
			m.slot = i + 1
			if m.slot >= len(seq) {
				m.round++
				m.slot = 0
			}
			return
		}
	}
}

// pickNextMemberLocked returns the next member to serve in round-counter
// order. Quiet members (no turn waiting) are skipped without consuming a
// serve — they keep their place and get their slot in a later round — and
// the round advances only on a serve, never on a skip.
func (m *ringManager) pickNextMemberLocked(waiting map[string]bool) *session {
	members := m.membersLocked()
	L := len(members)
	if L == 0 {
		return nil
	}
	if L == 1 {
		if waiting[members[0].id] {
			m.recordServeLocked(members[0].id)
			return members[0]
		}
		return nil
	}
	seq := roundSeqFor(m.round, L)
	if m.slot >= len(seq) {
		m.round++
		m.slot = 0
		seq = roundSeqFor(m.round, L)
	}
	for i := m.slot; i < len(seq); i++ {
		cand := members[seq[i]]
		if waiting[cand.id] {
			m.slot = i + 1
			if m.slot >= len(seq) {
				m.round++
				m.slot = 0
			}
			m.recordServeLocked(cand.id)
			return cand
		}
	}
	// No waiter in the remainder of this round: look into the next round.
	// Serving there commits the round advance; finding nobody mutates
	// nothing (the caller already knows some member waits — it must be
	// before m.slot in this round, and its slot comes round again).
	next := roundSeqFor(m.round+1, L)
	for i := 0; i < len(next); i++ {
		cand := members[next[i]]
		if waiting[cand.id] {
			m.round++
			m.slot = i + 1
			if m.slot >= len(next) {
				m.round++
				m.slot = 0
			}
			m.recordServeLocked(cand.id)
			return cand
		}
	}
	return nil
}

// scanDistanceLocked is how many serves the cursor needs to reach the
// member, counting from the current slot (crossing into the next round if
// needed). It feeds the member-recovery laxity: a member is rescued only
// when the scan cannot reach it inside its budget. Returns -1 if the member
// is not in the member set.
func (m *ringManager) scanDistanceLocked(id string) int {
	members := m.membersLocked()
	L := len(members)
	if L == 0 {
		return -1
	}
	pos := -1
	for i, s := range members {
		if s.id == id {
			pos = i
			break
		}
	}
	if pos < 0 {
		return -1
	}
	if L == 1 {
		return 0
	}
	seq := roundSeqFor(m.round, L)
	slot := m.slot
	if slot >= len(seq) {
		slot = len(seq) // distances wrap through the next round anyway
	}
	for i := slot; i < len(seq); i++ {
		if seq[i] == pos {
			return i - slot
		}
	}
	next := roundSeqFor(m.round+1, L)
	for i := 0; i < len(next); i++ {
		if next[i] == pos {
			return (len(seq) - slot) + i
		}
	}
	return -1
}

// recordServeLocked appends to the serve history (debug telemetry).
func (m *ringManager) recordServeLocked(id string) {
	m.serveHistory = append(m.serveHistory, id)
	if len(m.serveHistory) > 64 {
		m.serveHistory = m.serveHistory[1:]
	}
	m.counters.memberServes++
}

// headOutsider returns the waiting outsider earliest in ring order. Because
// demotion re-seqs, this is "least recently an outsider", not arrival order
// — a just-demoted session rejoins behind everyone.
func (m *ringManager) headOutsiderLocked(waitingOutsiders map[string]bool) *session {
	var best *session
	for id := range waitingOutsiders {
		s := m.sessions[id]
		if s == nil {
			continue
		}
		if best == nil || s.seq < best.seq {
			best = s
		}
	}
	return best
}

// noteCapacity records an endpoint's KV capacity; the pool capacity is the
// most recently seen value (single-pool v0).
func (m *ringManager) noteCapacity(endpointID string, capacity float64) {
	if capacity <= 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.endpoints[endpointID] = capacity
	m.capacity = capacity
	m.capacityFromConfig = false
}

// chargePreRequest is the PreRequest hook body: bind the session, charge the
// in-flight estimate, clear any reservation.
func (m *ringManager) chargePreRequest(id, endpointID string, estimate int64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.ensureSessionLocked(id)
	s.endpointID = endpointID
	s.inflightTokens += estimate
	s.turnCount++
	s.lastActivity = now
	s.reservedUntil = time.Time{}
}

// reserveLocked marks a just-picked candidate so the next Pick cycle does
// not admit into the same room twice before PreRequest charges it. The
// caller must hold m.mu (Pick does).
func (m *ringManager) reserveLocked(id string, tokens float64, now time.Time) {
	if s := m.sessions[id]; s != nil {
		s.reservedTokens = tokens
		s.reservedUntil = now.Add(reservationTTL)
	}
}

// settleResponse is the ResponseBody hook body: remove the turn's in-flight
// estimate, commit the measured footprint, record the turn duration and the
// release order.
func (m *ringManager) settleResponse(id string, estimate int64, totalTokens int64, turnDur time.Duration, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	s.inflightTokens = max(s.inflightTokens-estimate, 0)
	if totalTokens > 0 {
		s.committedTokens = totalTokens
	} else if estimate > s.committedTokens {
		s.committedTokens = estimate
	}
	if turnDur > 0 {
		s.lastTurnDur = turnDur
		m.turnDurSum += turnDur
		m.turnDurCount++
	}
	s.lastFinish = now
	s.lastActivity = now
}

// sweep drops sessions idle past the TTL with nothing in flight: ledger
// hygiene for ended sessions, not a scheduling rule (the leave rule is
// demotion by eviction order; the sweep only reclaims bookkeeping).
func (m *ringManager) sweep(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ttl := time.Duration(m.cfg.EvictionTTLSeconds * float64(time.Second))
	for id, s := range m.sessions {
		if !s.dispatchable(now) {
			continue
		}
		if now.Sub(s.lastActivity) <= ttl {
			continue
		}
		if s.member {
			s.member = false
			m.counters.demotions["sweep"]++
		}
		delete(m.sessions, id)
	}
}

// snapshot is the state reported by metrics and DumpState.
type managerSnapshot struct {
	members            int
	waiting            int
	inflight           int
	memberSum          float64
	budget             float64
	capacity           float64
	capacityFromConfig bool
	widthD             int
	binding            string
	meanTurnDur        time.Duration
	dEff               time.Duration
	counters           managerCounters
	rotationIntervals  []time.Duration
}

func (m *ringManager) snapshot(now time.Time) managerSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	snap := managerSnapshot{
		members:            len(m.membersLocked()),
		waiting:            m.lastNWaiting,
		inflight:           m.inflightCountLocked(),
		memberSum:          m.memberSumLocked(now),
		budget:             m.budgetTokensLocked(),
		capacity:           m.capacity,
		capacityFromConfig: m.capacityFromConfig,
		widthD:             m.lastWidthD,
		binding:            m.lastBinding,
		meanTurnDur:        m.meanTurnDurLocked(),
		dEff:               m.lastDEff,
		rotationIntervals:  append([]time.Duration(nil), m.rotationIntervals...),
		counters: managerCounters{
			rotations:    m.counters.rotations,
			guests:       m.counters.guests,
			oversized:    m.counters.oversized,
			noVictims:    m.counters.noVictims,
			memberServes: m.counters.memberServes,
			demotions:    make(map[string]int64, len(m.counters.demotions)),
		},
	}
	for k, v := range m.counters.demotions {
		snap.counters.demotions[k] = v
	}
	return snap
}

// sessionView is a session's public state, for tests and dumps.
type sessionView struct {
	ID       string  `json:"id"`
	Member   bool    `json:"member"`
	Seq      int64   `json:"seq"`
	Tokens   int64   `json:"tokens"`
	Turns    int64   `json:"turns"`
	LastTurn float64 `json:"lastTurnSeconds"`
}

func (m *ringManager) sessionViewsLocked() []sessionView {
	var out []sessionView
	for _, s := range m.sessions {
		out = append(out, sessionView{
			ID: s.id, Member: s.member, Seq: s.seq,
			Tokens: s.committedTokens, Turns: s.turnCount,
			LastTurn: s.lastTurnDur.Seconds(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out
}
