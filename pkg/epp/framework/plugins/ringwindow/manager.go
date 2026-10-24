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
	turnCount  int64
	// seq is the ring position: assigned on first join and again on rejoin
	// after a sweep. The ring is a queue discipline, not arrival order.
	seq int64
	// member labels the session as inside the window. Labels govern
	// scheduling only; they never pin or protect blocks.
	member      bool
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
// no turn in flight and no live reservation. Demotion candidates and the
// sweep skip sessions that are not dispatchable; the engine cannot evict
// their blocks anyway.
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

	// capacity is the pool's KV capacity in tokens (single-pool v0).
	capacity float64
	// endpoints caches the last-seen capacity per endpoint id.
	endpoints map[string]float64

	// Scan state: the window is served back-and-forth over the member set
	// in ring (seq) order. lastServed anchors the cursor by session id, so
	// membership changes never orphan it. origin rotates the turnaround
	// position by one on each reversal (decision D1); with
	// rotatingTurnstation == false the origin never advances and the order
	// is plain back-and-forth.
	lastServed string
	dir        int
	origin     int

	// turnDurSum/turnDurCount maintain the running mean turn duration
	// since plugin start (no window, no constant): the ŝ used by the
	// elastic D_eff and the D-width.
	turnDurSum   time.Duration
	turnDurCount int64

	cfg Config

	// counters for metrics and state dumps.
	counters managerCounters
}

type managerCounters struct {
	rotations    int64
	guests       int64
	oversized    int64
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
		// of zero and take the oversized-guest path.
		capacity: float64(cfg.CapacityTokens),
		cfg:      cfg,
		dir:      1,
		counters: managerCounters{demotions: make(map[string]int64)},
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

// widthD is the D-width of the window: a member's worst gap under
// back-and-forth is ~2W·ŝ, so W ≤ D/(2ŝ). Returns a large value while ŝ is
// unknown (before the first turn completes).
func (m *ringManager) widthDLocked(dEff time.Duration) int {
	ŝ := m.meanTurnDurLocked()
	if ŝ <= 0 {
		return 1 << 30
	}
	w := int(dEff / (2 * ŝ))
	if w < 1 {
		w = 1
	}
	return w
}

// fitsLocked reports whether a session of est tokens can join the window
// under both widths: the budget width (Σ + est ≤ θ·C) and the D-width
// (member count + 1 ≤ W_D).
func (m *ringManager) fitsLocked(est float64, dEff time.Duration, now time.Time) bool {
	if m.memberSumLocked(now)+est > m.budgetTokensLocked() {
		return false
	}
	return len(m.membersLocked())+1 <= m.widthDLocked(dEff)
}

// demotionVictims returns the demotion candidates: members whose turn is
// neither waiting nor in flight (a waiting or running member's blocks are
// about to be touched or are pinned, and demoting it would ping-pong the
// label), ordered by release time, oldest first — the order in which the
// engine evicts (decision D7's completion-order primary).
func (m *ringManager) demotionVictimsLocked(waiting map[string]bool, now time.Time) []*session {
	var victims []*session
	for _, s := range m.membersLocked() {
		if waiting[s.id] || !s.dispatchable(now) {
			continue
		}
		victims = append(victims, s)
	}
	sort.Slice(victims, func(i, j int) bool {
		return victims[i].lastFinish.Before(victims[j].lastFinish)
	})
	return victims
}

// demoteUntilFit demotes members in eviction order until a session of est
// tokens fits both widths, or no demotable member remains. reason labels the
// demotion counter. Returns whether it fits now.
func (m *ringManager) demoteUntilFitLocked(est float64, dEff time.Duration, waiting map[string]bool, reason string, now time.Time) bool {
	for !m.fitsLocked(est, dEff, now) {
		victims := m.demotionVictimsLocked(waiting, now)
		if len(victims) == 0 {
			return m.fitsLocked(est, dEff, now)
		}
		victims[0].member = false
		m.counters.demotions[reason]++
	}
	return true
}

// admitWithDemotion is the join path shared by rotation (Pick step 1) and
// the never-idle guest (Pick step 3): demote until the candidate fits, then
// join it and re-anchor the scan cursor to it. An oversized candidate that
// no subset of members can cover runs as a non-member and the window clears
// to rebuild by arrival (decision D6).
func (m *ringManager) admitWithDemotion(id string, est float64, dEff time.Duration, waiting map[string]bool, reason string, now time.Time) (joined, oversized bool) {
	s := m.ensureSessionLocked(id)
	if fits := m.demoteUntilFitLocked(est, dEff, waiting, reason, now); fits {
		s.member = true
		m.lastServed = id
		return true, false
	}
	// Oversized: clear the window, serve as a non-member, let it refill.
	for _, mem := range m.membersLocked() {
		mem.member = false
		m.counters.demotions["clear"]++
	}
	m.counters.oversized++
	m.lastServed = ""
	return false, true
}

// clearWindowLocked drops every member label (oversized-guest fallback).
func (m *ringManager) clearWindowLocked() {
	for _, s := range m.membersLocked() {
		s.member = false
		m.counters.demotions["clear"]++
	}
}

// rotate is Pick step 1: an outsider's D-clock has expired, so the window
// rotates one place — the outsider joins (demoting in eviction order until
// the budget covers it) and is dispatched first in the new window. Members
// never trigger rotation (decision D4).
func (m *ringManager) rotate(id string, est float64, dEff time.Duration, waiting map[string]bool, now time.Time) (joined, oversized bool) {
	m.counters.rotations++
	return m.admitWithDemotion(id, est, dEff, waiting, "rotate", now)
}

// guest is Pick step 3's non-fitting path: the GPU must not idle while no
// member waits, so the head outsider runs even though it does not fit
// (decision D6, the weak sense).
func (m *ringManager) guest(id string, est float64, dEff time.Duration, waiting map[string]bool, now time.Time) (joined, oversized bool) {
	m.counters.guests++
	return m.admitWithDemotion(id, est, dEff, waiting, "guest", now)
}

// joinQuiet admits the head outsider when it fits without demoting anyone.
func (m *ringManager) joinQuiet(id string, est float64, dEff time.Duration, now time.Time) bool {
	if !m.fitsLocked(est, dEff, now) {
		return false
	}
	s := m.ensureSessionLocked(id)
	s.member = true
	m.lastServed = id
	return true
}

// scanView is the member list rotated by the origin: the back-and-forth
// sweep runs over this view, and the origin advancing on each reversal is
// what rotates the turnaround position.
func (m *ringManager) scanViewLocked() []*session {
	members := m.membersLocked()
	if len(members) == 0 {
		return nil
	}
	o := m.origin % len(members)
	if o < 0 {
		o += len(members)
	}
	return append(members[o:], members[:o]...)
}

// pickNextMember returns the next member to serve in back-and-forth order:
// starting after the last-served member, walking in the scan direction,
// skipping members with no turn waiting (a skipped member keeps its place —
// it is served in a later pass, not dropped). Quiet members never advance
// the anchor. Returns nil when no waiting member exists.
func (m *ringManager) pickNextMemberLocked(waiting map[string]bool) *session {
	view := m.scanViewLocked()
	L := len(view)
	if L == 0 {
		return nil
	}
	if L == 1 {
		// A single member cannot be walked past: serve it when it waits.
		if waiting[view[0].id] {
			return view[0]
		}
		return nil
	}
	idx := 0
	if m.lastServed != "" {
		for i, s := range view {
			if s.id == m.lastServed {
				idx = i
				break
			}
		}
	}
	// Bound: two full passes cover every position in both directions.
	for step := 0; step < 2*L+2; step++ {
		probe := idx + m.dir
		if probe < 0 || probe >= L {
			// Edge: reverse and rotate the turnaround one place.
			m.dir = -m.dir
			if m.cfg.RotatingTurnaround {
				m.origin++
			}
			view = m.scanViewLocked()
			L = len(view)
			if L == 0 {
				return nil
			}
			idx = 0
			if m.lastServed != "" {
				for i, s := range view {
					if s.id == m.lastServed {
						idx = i
						break
					}
				}
			}
			continue
		}
		cand := view[probe]
		if waiting[cand.id] {
			return cand
		}
		idx = probe
	}
	return nil
}

// serveMember records a member dispatch: the anchor moves to it and the
// member-serve counter ticks.
func (m *ringManager) serveMemberLocked(s *session) {
	m.lastServed = s.id
	m.counters.memberServes++
}

// headOutsider returns the waiting outsider earliest in ring order.
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
	members     int
	waiting     int
	memberSum   float64
	budget      float64
	capacity    float64
	widthD      int
	meanTurnDur time.Duration
	counters    managerCounters
}

func (m *ringManager) snapshot(now time.Time) managerSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	snap := managerSnapshot{
		members:     len(m.membersLocked()),
		memberSum:   m.memberSumLocked(now),
		budget:      m.budgetTokensLocked(),
		capacity:    m.capacity,
		meanTurnDur: m.meanTurnDurLocked(),
		counters: managerCounters{
			rotations:    m.counters.rotations,
			guests:       m.counters.guests,
			oversized:    m.counters.oversized,
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
