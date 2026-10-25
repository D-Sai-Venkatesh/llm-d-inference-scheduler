// Package arcwindow implements the sliding-arc session scheduler: every
// live session holds an immutable position on a hash ring, and a contiguous
// ARC of W positions — the window — slides over it. Membership is whoever
// the arc currently covers; the wait bound is one arc cycle, made
// structural by the D-demand path. Design: ARC-PLAN.md (Revision 1/1a/1b).
package arcwindow

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// PluginType is the plugin type registered with the framework.
const PluginType = "arc-window"

// Config configures the arc-window plugin.
type Config struct {
	// CapacityTokens is the KV cache capacity in tokens used while no
	// endpoint has reported cache_config_info through the datalayer.
	CapacityTokens int64 `json:"capacityTokens"`
	// Theta bounds the arc's token budget: Σ member footprints ≤ θ·C.
	Theta float64 `json:"theta"`
	// ServeUncovered selects the work-conservation fork (A5):
	// false (the pure arc) — only arc members dispatch; uncovered
	//   sessions wait for the arc (cache protection by construction);
	// true (the hybrid) — when no member waits, the next uncovered
	//   session (ring order from the trailing edge) dispatches without
	//   moving the arc (v0-ring's guest path).
	ServeUncovered bool `json:"serveUncovered"`
	// DropPolicy selects the shrink/growth victim (A6-bis):
	// "edge" (primary) — the trailing-edge member, positional;
	// "value" — the least-recently-finished member anywhere in the arc
	//   (the user's rearrangement idea: warm members are implicitly
	//   pulled away from the edge).
	DropPolicy string `json:"dropPolicy"`
	// HashRootID hashes the flow's root id (when the fairness id carries
	// one — a "_"/"/" separator) instead of the flow id, co-locating a
	// session's subagent flows on the ring. Off for the race
	// (pre-registered caveat: WEKA's aiperf ids are opaque UUIDs).
	HashRootID bool `json:"hashRootId"`
	// Salt seeds the placement hash so arms are comparable under the
	// same salt and reproducible across runs (default fixed).
	Salt string `json:"salt"`
	// EpochPolicy selects the arc's completion rule: "lap" (default, the
	// cursor sweep) or "quota" (exactly W dispatches; unserved-first,
	// then least-dispatched selection).
	EpochPolicy string `json:"epochPolicy"`
	// RingBudgetMultiple caps the ring's total footprint at x·C (the
	// population tier): sessions beyond wait off-ring, admitted
	// oldest-first when mass frees. 0 (default) = unbounded.
	RingBudgetMultiple float64 `json:"ringBudgetMultiple"`
	// EvictionTTLSeconds is ledger hygiene: how long a session with no
	// activity stays on the ring. The arc's N paces over live sessions,
	// so a tight TTL keeps the cycle honest (5 min per Revision 1b).
	EvictionTTLSeconds   float64 `json:"evictionTtlSeconds"`
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds"`
}

func defaultConfig() Config {
	return Config{
		CapacityTokens:       4194304,
		Theta:                0.8,
		ServeUncovered:       false,
		DropPolicy:           "edge",
		HashRootID:           false,
		Salt:                 "arcwindow.v1",
		EpochPolicy:          "lap",
		RingBudgetMultiple:   0,
		EvictionTTLSeconds:   300,
		EvictionSweepSeconds: 10,
	}
}

func (c Config) validate() error {
	if c.CapacityTokens <= 0 {
		return fmt.Errorf("capacityTokens must be > 0, got %d", c.CapacityTokens)
	}
	if c.Theta <= 0 || c.Theta > 1 {
		return fmt.Errorf("theta must be in (0, 1], got %v", c.Theta)
	}
	if c.DropPolicy != "edge" && c.DropPolicy != "value" {
		return fmt.Errorf("dropPolicy must be \"edge\" or \"value\", got %v", c.DropPolicy)
	}
	if c.EpochPolicy != "lap" && c.EpochPolicy != "quota" {
		return fmt.Errorf("epochPolicy must be \"lap\" or \"quota\", got %v", c.EpochPolicy)
	}
	if c.RingBudgetMultiple < 0 {
		return fmt.Errorf("ringBudgetMultiple must be >= 0 (0 = unbounded), got %v", c.RingBudgetMultiple)
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	return nil
}

// positionKey derives the hash input from the fairness id: when
// hashRootId is set and the id carries a root separator, the root prefix
// is used, co-locating a session's subagent flows on the ring.
func positionKey(id string, hashRoot bool) string {
	if hashRoot {
		if i := strings.IndexAny(id, "_/"); i > 0 {
			return id[:i]
		}
	}
	return id
}

// position is a session's immutable ring position: a well-mixing hash
// (splitmix64 over FNV-1a) of its key under the configured salt — never
// a plain string hash (sequential ids would recreate the ramp, M5).
func position(key, salt string) uint64 {
	h := uint64(14695981039346656037)
	s := key + "\x00" + salt
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	h += 0x9e3779b97f4a7c15
	h ^= h >> 30
	h *= 0xbf58476d1ce4e5b9
	h ^= h >> 27
	h *= 0x94d049bb133111eb
	h ^= h >> 31
	return h
}

// session is the arc ledger entry.
type session struct {
	id         string
	pos        uint64
	member     bool // inside the arc
	committed  float64
	inflight   float64
	chargedAt  time.Time // the last in-flight charge (M7: aging)
	lastFinish time.Time
	lastSeen   time.Time
	turns      int64
	lastTurn   time.Duration
}

// ringEntry is one live position on the big ring.
type ringEntry struct {
	pos uint64
	id  string
}

// epoch is one service cycle of the arc: the member set is fixed at start;
// the sweep serves waiting epoch-members in ring order (trailing →
// leading); the epoch completes when every epoch member that waited during
// it has been served (served-epoch semantics, Revision 1 fix 1/5).
type epoch struct {
	members []string        // ids in ring order, trailing → leading
	cursor  int             // next member index to consider
	waited  map[string]bool // members that waited during this epoch

	// quota mode: per-member dispatch counts this epoch, and the total.
	dispatches map[string]int64
	total      int64
}

// manager is the arc's state: the ring, the arc, the epoch, the ledger.
type manager struct {
	mu       sync.Mutex
	cfg      Config
	sessions map[string]*session
	ring     []ringEntry // sorted by pos

	// The arc: the leading edge position and the current width target.
	leadPos uint64
	width   int             // live member count target
	holes   map[string]hole // persistent drop record (B4)

	// The current epoch (nil between epochs).
	ep *epoch

	// lastAdvanceAt feeds arc_cycle_seconds (the heartbeat's period).
	lastAdvanceAt time.Time
	lastCycle     float64

	// offRing tracks sessions waiting outside the ring's token budget
	// (ringBudgetMultiple > 0 only): id -> park record (arrival + the
	// footprint estimate that parked it — the admission test uses THIS,
	// not the session's yet-empty committed mass, R1).
	offRing map[string]offRingEntry

	// kvUsage is the latest engine KV utilization (-1 = unseen).
	kvUsage            float64
	capacity           float64
	capacityFromConfig bool

	counters struct {
		advances           map[string]int64 // heartbeat | ddemand (arc motion only)
		membershipGrowth   int64            // membership transitions (churn, not arc motion)
		drops              map[string]int64 // edge | value
		immediateRedrop    int64
		coveredNeverServed int64
		advanceBlocked     int64
		epochs             int64
		served             int64
		uncoveredServed    int64
		ringAdmissions     int64
		offRingParked      int64
		deferred           int64 // A5-true guests skipped... (not used; kept for symmetry)
	}
	swept int64

	lastSweep time.Time
}

func newManager(cfg Config) *manager {
	return &manager{
		cfg:                cfg,
		sessions:           make(map[string]*session),
		ring:               make([]ringEntry, 0, 128),
		width:              1,
		holes:              make(map[string]hole),
		offRing:            make(map[string]offRingEntry),
		kvUsage:            -1,
		capacity:           float64(cfg.CapacityTokens),
		capacityFromConfig: true,
		counters: struct {
			advances           map[string]int64
			membershipGrowth   int64
			drops              map[string]int64
			immediateRedrop    int64
			coveredNeverServed int64
			advanceBlocked     int64
			epochs             int64
			served             int64
			uncoveredServed    int64
			ringAdmissions     int64
			offRingParked      int64
			deferred           int64
		}{
			advances: map[string]int64{},
			drops:    map[string]int64{},
		},
		lastSweep: time.Now(),
	}
}

// ensureSession returns the ledger entry, inserting the session on the
// ring at its immutable position on first sight.
func (m *manager) ensureSession(id string, now time.Time) *session {
	if s, ok := m.sessions[id]; ok {
		s.lastSeen = now
		return s
	}
	s := &session{id: id, pos: position(positionKey(id, m.cfg.HashRootID), m.cfg.Salt), lastSeen: now, lastFinish: now}
	m.sessions[id] = s
	i := sort.Search(len(m.ring), func(i int) bool { return m.ring[i].pos >= s.pos })
	m.ring = append(m.ring, ringEntry{})
	copy(m.ring[i+1:], m.ring[i:])
	m.ring[i] = ringEntry{pos: s.pos, id: s.id}
	return s
}

// removeSession drops a session from the ledger and the ring (hygiene).
func (m *manager) removeSession(id string) {
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	delete(m.sessions, id)
	i := sort.Search(len(m.ring), func(i int) bool { return m.ring[i].pos >= s.pos })
	if i < len(m.ring) && m.ring[i].id == id {
		m.ring = append(m.ring[:i], m.ring[i+1:]...)
	}
}

// holes is the persistent drop record: a dropped session's position stays
// a hole until the arc's trailing edge passes it (B4: drops must not be
// erased by the next positional recompute).
type hole struct {
	pos         uint64
	droppedAt   time.Time
	turnsAtDrop int64 // the session's dispatch count when dropped (the vacuous-coverage detector)
}

// arcMembers returns the live sessions the arc currently covers, in ring
// order trailing → leading. The arc is the W non-hole live sessions at
// positions ≤ leadPos (walking backward, wrapping). Holes are skipped and
// do not count toward W.
func (m *manager) arcMembers() []*session {
	var out []*session
	n := len(m.ring)
	if n == 0 {
		return out
	}
	// index of leadPos (or where it would insert)
	i := sort.Search(n, func(i int) bool { return m.ring[i].pos >= m.leadPos })
	if i < n && m.ring[i].pos == m.leadPos {
		// exact
	} else if i > 0 {
		i-- // snap down to the live member at or below leadPos
	} else {
		i = n - 1 // wrap: the highest position leads
	}
	if i >= n {
		i = n - 1
	}
	for k := 0; k < n && len(out) < m.width; k++ {
		j := (i - k + n) % n
		s := m.sessions[m.ring[j].id]
		if s == nil {
			continue
		}
		if _, holed := m.holes[s.id]; holed {
			continue
		}
		out = append(out, s)
	}
	// out is leading-first; reverse to trailing → leading.
	for l, r := 0, len(out)-1; l < r; l, r = l+1, r-1 {
		out[l], out[r] = out[r], out[l]
	}
	return out
}

// clearPassedHoles removes holes whose positions are now behind the arc's
// trailing edge (the leading edge having advanced past them).
func (m *manager) clearPassedHoles(members []*session) {
	if len(members) == 0 {
		return
	}
	trailingPos := members[0].pos
	for id, h := range m.holes {
		if !m.posInArcSpan(h.pos, trailingPos) {
			// M1's detector: the session left the arc without ever being
			// dispatched while covered (vacuous coverage).
			if s, ok := m.sessions[id]; ok && s.turns == h.turnsAtDrop {
				m.counters.coveredNeverServed++
			}
			delete(m.holes, id)
		}
	}
}

// posInArcSpan reports whether pos lies within the arc's positional span
// (trailingPos, leadPos], forward direction, wrapping.
func (m *manager) posInArcSpan(pos, trailingPos uint64) bool {
	if m.leadPos == trailingPos {
		return true
	}
	if m.leadPos > trailingPos {
		return pos > trailingPos && pos <= m.leadPos
	}
	return pos > trailingPos || pos <= m.leadPos // wraps
}

// dropHole records a persistent hole for a dropped member (B4).
func (m *manager) dropHole(s *session, now time.Time) {
	m.holes[s.id] = hole{pos: s.pos, droppedAt: now, turnsAtDrop: s.turns}
	s.member = false
}

// nextBeyond returns the live session at the smallest position strictly
// greater than leadPos (wrapping) — the session the next advance covers.
func (m *manager) nextBeyond() *session {
	n := len(m.ring)
	if n == 0 {
		return nil
	}
	i := sort.Search(n, func(i int) bool { return m.ring[i].pos > m.leadPos })
	if i == n {
		i = 0
	}
	return m.sessions[m.ring[i].id]
}

// setMembers marks exactly the arc's sessions as members.
func (m *manager) setMembers(list []*session, now time.Time) {
	inArc := make(map[string]bool, len(list))
	for _, s := range list {
		inArc[s.id] = true
	}
	for id, s := range m.sessions {
		was, nowIs := s.member, inArc[id]
		if nowIs && !was {
			m.counters.membershipGrowth++
		} else if !nowIs && was {
			m.counters.drops[m.cfg.DropPolicy]++
		}
		s.member = nowIs
	}
}

// memberSum is Σ member footprints (committed + inflight).
func (m *manager) memberSum(now time.Time) float64 {
	var sum float64
	for _, s := range m.sessions {
		if s.member {
			sum += s.committed + s.inflight
		}
	}
	return sum
}

// idlePool is the measured (1 - KV) * C; -1 when KV is unknown.
func (m *manager) idlePool() float64 {
	if m.kvUsage < 0 || m.capacity <= 0 {
		return -1
	}
	return (1 - m.kvUsage) * m.capacity
}

// budget is θ·C.
func (m *manager) budget() float64 { return m.cfg.Theta * m.capacity }

// meanFootprint is the measured mean committed footprint of live sessions.
func (m *manager) meanFootprint() float64 {
	var sum float64
	var n int
	for _, s := range m.sessions {
		if s.committed > 0 {
			sum += s.committed
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// updateWidth recomputes the width target from the measured pool and the
// mean footprint (the v0-ring calibration, Revision 1 fix 3: W is a
// TARGET; the sum-fit gate enforces the real bound).
func (m *manager) updateWidth(now time.Time) {
	f := m.meanFootprint()
	if f <= 0 {
		return
	}
	w := m.budget() / f
	if p := m.idlePool(); p >= 0 {
		if pw := p / f; pw < w {
			w = pw
		}
	}
	if w < 1 {
		w = 1
	}
	m.width = int(math.Ceil(w))
}

// sweepLocked removes sessions idle beyond the TTL (hygiene; the arc's N
// paces over live sessions, so the TTL keeps the cycle honest). The caller
// holds mu.
func (m *manager) sweepLocked(now time.Time) {
	if now.Sub(m.lastSweep) < time.Duration(m.cfg.EvictionSweepSeconds*float64(time.Second)) {
		return
	}
	m.lastSweep = now
	ttl := time.Duration(m.cfg.EvictionTTLSeconds * float64(time.Second))
	was := len(m.sessions)
	for id, s := range m.sessions {
		// M7: an in-flight charge older than the hygiene TTL never
		// settles (client cancel, EvictedTTL, upstream error) — age it
		// out so the footprint cannot leak forever.
		if s.inflight > 0 && !s.chargedAt.IsZero() && now.Sub(s.chargedAt) > ttl {
			s.inflight = 0
		}
		if s.inflight <= 0 && now.Sub(s.lastSeen) > ttl {
			m.removeSession(id)
			m.swept++
		}
	}
	if len(m.sessions) < was && m.ep != nil {
		// The epoch's member list may reference swept sessions. Filter
		// while adjusting the cursor for removals BEFORE it (M9): a
		// removal ahead of the cursor shifts indices down; a removal
		// behind it means that member was already served or skipped.
		keep := m.ep.members[:0]
		removedBefore := 0
		for i, id := range m.ep.members {
			if _, ok := m.sessions[id]; ok {
				keep = append(keep, id)
			} else if i < m.ep.cursor {
				removedBefore++
			}
		}
		m.ep.members = keep
		// N2: removals before the cursor shift indices down by their count.
		m.ep.cursor -= removedBefore
		if m.ep.cursor < 0 {
			m.ep.cursor = 0
		}
		if m.ep.cursor > len(keep) {
			m.ep.cursor = len(keep)
		}
	}
}

// startEpoch snapshots the current arc as the epoch's member list
// (trailing → leading) and resets the cursor.
func (m *manager) startEpoch(now time.Time) {
	list := m.arcMembers()
	m.setMembers(list, now)
	ids := make([]string, len(list))
	for i, s := range list {
		ids[i] = s.id
	}
	// Q1: the dispatch counts carry across epochs (the long-run deficit —
	// tier 2's "least-dispatched" must mean over the arc's tenure, or the
	// same members win every epoch by ring order).
	var carry map[string]int64
	if m.ep != nil && m.ep.dispatches != nil {
		carry = m.ep.dispatches
	}
	m.ep = &epoch{members: ids, waited: map[string]bool{}, dispatches: carry}
	if m.ep.dispatches == nil {
		m.ep.dispatches = map[string]int64{}
	}
	m.counters.epochs++
}

// advance moves the arc one position: the leading edge takes in the next
// live session beyond it; the member set recomputes. The sum-fit gate
// drops victims (edge or value policy) — coverage NEVER blocks (C8).
func (m *manager) advance(reason string, now time.Time) {
	next := m.nextBeyond()
	if next == nil {
		return
	}
	delete(m.holes, next.id) // N1: covering clears any hole at this position
	if !m.lastAdvanceAt.IsZero() {
		m.lastCycle = now.Sub(m.lastAdvanceAt).Seconds()
	}
	m.lastAdvanceAt = now
	m.leadPos = next.pos
	m.counters.advances[reason]++
	m.refitArc(now)
	m.clearPassedHoles(m.arcMembers())
}

// coverTo advances the arc until id is a member (D-demand path), then
// starts a fresh epoch with the cursor at id (re-anchor, Revision 1 fix 2).
func (m *manager) coverTo(id string, now time.Time) {
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	if !s.member {
		// R3: clear any sum-fit hole for this session — or every Pick
		// re-enters coverTo on it (a ddemand advance per dispatch, the
		// arc pinned backwards on the hole).
		delete(m.holes, id)
		m.leadPos = s.pos
		m.counters.advances["ddemand"]++
		m.refitArc(now)
		m.clearPassedHoles(m.arcMembers())
	}
	m.startEpoch(now)
	// re-anchor the cursor at id
	for i, mid := range m.ep.members {
		if mid == id {
			m.ep.cursor = i
			break
		}
	}
}

// refitArc enforces the sum-fit gate: while memberSum exceeds the budget
// or the pool, drop victims (edge: the trailing member; value: the
// least-recently-finished member anywhere in the arc). Drops create
// holes the sweep skips; the arc's count recovers on later advances.
func (m *manager) refitArc(now time.Time) {
	list := m.arcMembers()
	m.setMembers(list, now)
	for {
		limit := m.budget()
		if p := m.idlePool(); p >= 0 && p < limit {
			limit = p
		}
		if m.memberSum(now) <= limit || len(list) <= 1 {
			break
		}
		// pick the victim
		var victim *session
		if m.cfg.DropPolicy == "value" {
			for _, s := range list {
				if s.inflight > 0 {
					continue // never drop mid-turn
				}
				if victim == nil || s.lastFinish.Before(victim.lastFinish) {
					victim = s
				}
			}
		}
		if victim == nil {
			// edge policy (or all mid-turn): the trailing member
			for _, s := range list {
				if s.inflight <= 0 {
					victim = s
					break
				}
			}
		}
		if victim == nil {
			m.counters.advanceBlocked++ // C8's detector: over budget, no demotable member
			break
		}
		m.dropHole(victim, now)
		m.counters.drops[m.cfg.DropPolicy]++
		m.counters.immediateRedrop++ // refined by the caller when re-covered
		// rebuild the list without the victim
		nl := list[:0]
		for _, s := range list {
			if s.id != victim.id {
				nl = append(nl, s)
			}
		}
		list = nl
	}
}

// stateDump is the snapshot returned by DumpState.
type stateDump struct {
	EpochPolicy        string           `json:"epochPolicy"`
	RingBudgetMultiple float64          `json:"ringBudgetMultiple"`
	OffRing            int              `json:"offRing"`
	RingAdmissions     int64            `json:"ringAdmissions"`
	Width              int              `json:"width"`
	Members            int              `json:"members"`
	MemberSum          float64          `json:"memberSumTokens"`
	Budget             float64          `json:"budgetTokens"`
	Pool               float64          `json:"idlePoolTokens"`
	RingSize           int              `json:"ringSize"`
	DemandLive         int              `json:"demandLive"`
	Epochs             int64            `json:"epochs"`
	Advances           map[string]int64 `json:"advances"`
	Drops              map[string]int64 `json:"drops"`
	ServeUncovered     bool             `json:"serveUncovered"`
	DropPolicy         string           `json:"dropPolicy"`
}

// DumpState reports the arc, the counters and the ledger summary.
func (m *manager) dump() json.RawMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	d := stateDump{
		EpochPolicy: m.cfg.EpochPolicy, RingBudgetMultiple: m.cfg.RingBudgetMultiple,
		OffRing: len(m.offRing), RingAdmissions: m.counters.ringAdmissions,
		Width: m.width, Members: len(m.arcMembers()), MemberSum: m.memberSum(time.Now()),
		Budget: m.budget(), Pool: m.idlePool(), RingSize: len(m.sessions),
		Epochs: m.counters.epochs, Advances: m.counters.advances, Drops: m.counters.drops,
		ServeUncovered: m.cfg.ServeUncovered, DropPolicy: m.cfg.DropPolicy,
	}
	for _, s := range m.sessions {
		if time.Since(s.lastSeen) < 10*time.Minute {
			d.DemandLive++
		}
	}
	out, _ := json.Marshal(d)
	return out
}

// noteCapacity records an endpoint's reported KV capacity (locked).
func (m *manager) noteCapacity(_ string, capacity float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if capacity > 0 {
		m.capacity = capacity
		m.capacityFromConfig = false
	}
}

// chargePreRequest adds the turn's estimate to the session's in-flight
// footprint (first sight inserts it on the ring at its immutable position).
func (m *manager) chargePreRequest(id string, estimate int64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.ensureSession(id, now)
	s.inflight += float64(estimate)
	s.chargedAt = now
	s.turns++
}

// settleResponse settles a completed turn: in-flight estimate removed,
// committed footprint set from the usage total, turn duration and
// completion time recorded.
func (m *manager) settleResponse(id string, estimate, total int64, turnDur time.Duration, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return
	}
	s.inflight = max(s.inflight-float64(estimate), 0)
	if total > 0 {
		s.committed = float64(total)
	}
	s.lastFinish = now
	s.lastSeen = now
	if turnDur > 0 {
		s.lastTurn = turnDur
	}
}

// sweepExternal runs the hygiene sweep from the plugin's ticker
// goroutine (takes the lock; Pick uses sweepLocked instead).
func (m *manager) sweepExternal(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sweepLocked(now)
}

// arcSnapshot is the scrape-time view for the ledger collector.
type arcSnapshot struct {
	cycleSeconds       float64
	members            int
	offRing            int
	ringAdmissions     int64
	offRingParked      int64
	ringBudget         float64
	epochPolicy        string
	ringBudgetMultiple float64
	memberSum          float64
	width              int
	pool               float64
	budget             float64
	ringSize           int
	demandLive         int
	holes              int
	epochs             int64
	served             int64
	membershipGrowth   int64
	uncoveredServed    int64
	immediateRedrop    int64
	coveredNeverServed int64
	advanceBlocked     int64
	swept              int64
	advances           map[string]int64
	drops              map[string]int64
}

// arcSnapshot collects the ledger state under the lock (B5: every gauge
// and counter observable at scrape time).
func (m *manager) arcSnapshot() arcSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	s := arcSnapshot{
		cycleSeconds: m.lastCycle,
		offRing:      len(m.offRing), ringAdmissions: m.counters.ringAdmissions,
		offRingParked: m.counters.offRingParked, ringBudget: m.ringBudget(),
		epochPolicy: m.cfg.EpochPolicy, ringBudgetMultiple: m.cfg.RingBudgetMultiple,
		width: m.width, pool: m.idlePool(), budget: m.budget(),
		ringSize: len(m.sessions), holes: len(m.holes),
		epochs: m.counters.epochs, served: m.counters.served,
		membershipGrowth:   m.counters.membershipGrowth,
		uncoveredServed:    m.counters.uncoveredServed,
		immediateRedrop:    m.counters.immediateRedrop,
		coveredNeverServed: m.counters.coveredNeverServed,
		advanceBlocked:     m.counters.advanceBlocked,
		swept:              m.swept,
		advances:           map[string]int64{}, drops: map[string]int64{},
	}
	for r, v := range m.counters.advances {
		s.advances[r] = v
	}
	for p, v := range m.counters.drops {
		s.drops[p] = v
	}
	s.members = len(m.arcMembers())
	s.memberSum = m.memberSum(now)
	ttl := time.Duration(m.cfg.EvictionTTLSeconds * float64(time.Second))
	for _, sess := range m.sessions {
		if now.Sub(sess.lastSeen) < ttl {
			s.demandLive++
		}
	}
	return s
}

// ringMass is the total footprint of all live ring sessions.
func (m *manager) ringMass() float64 {
	var sum float64
	for _, s := range m.sessions {
		sum += s.committed + s.inflight
	}
	return sum
}

// ringBudget is x·C (0 = unbounded).
func (m *manager) ringBudget() float64 {
	if m.cfg.RingBudgetMultiple <= 0 {
		return 0
	}
	return m.cfg.RingBudgetMultiple * m.capacity
}

// offRingEntry is the park record: when the session arrived off-ring and
// the footprint estimate that parked it.
type offRingEntry struct {
	arrival time.Time
	est     float64
}

// admitOffRing admits the oldest-waiting off-ring session if the ring
// budget allows (R1: the test uses the PARKED ESTIMATE; the session's own
// mass is not in ringMass because it is off-ring).
func (m *manager) admitOffRing(waiting map[string]bool, now time.Time) {
	budget := m.ringBudget()
	if budget <= 0 || len(m.offRing) == 0 {
		return
	}
	var oldestID string
	var oldest time.Time
	for id, e := range m.offRing {
		if !waiting[id] {
			continue
		}
		if oldestID == "" || e.arrival.Before(oldest) {
			oldestID, oldest = id, e.arrival
		}
	}
	if oldestID == "" {
		return
	}
	est := m.offRing[oldestID].est
	m.ensureSession(oldestID, now)
	if est <= budget-m.ringMass() {
		delete(m.offRing, oldestID)
		m.counters.ringAdmissions++
	}
}

// maybeOffRing parks a first-seen session outside the ring budget (R1:
// park once, storing the estimate; the counter counts sessions not events).
func (m *manager) maybeOffRing(id string, est float64, now time.Time) bool {
	budget := m.ringBudget()
	if budget <= 0 {
		return false
	}
	if _, parked := m.offRing[id]; parked {
		return true
	}
	if m.ringMass()+est <= budget {
		return false
	}
	m.offRing[id] = offRingEntry{arrival: now, est: est}
	m.counters.offRingParked++
	return true
}

// maybeOffRingMass is maybeOffRing with the ring mass precomputed by the
// caller (the O(N²) hot-path fix: one mass computation per Pick).
func (m *manager) maybeOffRingMass(id string, est float64, now time.Time, mass float64) bool {
	budget := m.ringBudget()
	if budget <= 0 {
		return false
	}
	if _, parked := m.offRing[id]; parked {
		return true
	}
	if mass+est <= budget {
		return false
	}
	m.offRing[id] = offRingEntry{arrival: now, est: est}
	m.counters.offRingParked++
	return true
}
