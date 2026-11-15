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

// THE SESSION LAYER (session-layer contract Part B, `sessionAware`): the session id is used for ORDER
// only (P5) — who is considered first at the door (R2), who gets a freed seat (L2-2), and when a new
// session's head may collect (H8/H10). Nothing here changes what is charged, counted or pushed (P1), and
// the flow-control queues stay per THREAD: the session header is not the fairness header (C4, S2b).
//
// The thread → session mapping is an identity fact recorded once (S4); everything a decision reads about
// a session is DERIVED once per pass from the live state (ST1), so there is no session table to keep
// consistent.

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// The session keys: a header value and a thread's own (no-header) session can never collide.
const (
	sessionKeyHeader = "s:"
	sessionKeyThread = "t:"
)

// sessionsLogged is S7's startup read-back: the first distinct session ids are logged.
const sessionsLogged = 5

// nca2Sessions is Part B's state.
type nca2Sessions struct {
	// sessionOf is S4: the first session value a thread's requests carried. A thread with none is its
	// own session (S5). Its lifetime is s.nca2.ever's (LOGIC Amendment 1 Q4: v2 never forgets a thread).
	sessionOf map[string]string
	// members is the reverse index (session value → its threads), for everIn: a session whose threads
	// are all outside is group B only if one of them was ever admitted.
	members map[string][]string
	// distinct counts the session values seen (the first sessionsLogged are logged).
	distinct int
	// mismatches counts dispatched requests whose session value differs from their thread's (S4/E6).
	mismatches int64

	// facts is ST1, derived at the top of every pass from the live sets.
	facts map[string]sessFacts
}

// sessFacts is ST1 for one session.
type sessFacts struct {
	// admitted, active and seats are the session's admitted, active and seated threads (seats cluster-wide).
	admitted, active, seats int
	// everIn / everKnown memoise everIn for the pass (it is only needed for sessions with nothing admitted).
	everIn, everKnown bool
}

// sessionOfRequest is S3: the configured header's value on the request ("" with no header configured).
func sessionOfRequest(request *fwksched.InferenceRequest, header string) string {
	if request == nil || header == "" {
		return ""
	}
	v, _ := metadata.GetLowerCaseHeaderValue(request.Headers, header)
	return v
}

// sessionsOn is Part B's state, nil unless the session layer runs.
func (s *Scheduler) sessionsOn() *nca2Sessions {
	if s.nca2 == nil || s.nca2.door == nil || !s.nca2.door.sessionAware {
		return nil
	}
	d := s.nca2.door
	if d.sessions == nil {
		d.sessions = &nca2Sessions{sessionOf: map[string]string{}, members: map[string][]string{},
			facts: map[string]sessFacts{}}
	}
	return d.sessions
}

// sessionNote is S4: a thread's first session value is recorded; a later different value is ignored here
// (the dispatch counts it once per request).
func (s *Scheduler) sessionNote(thread, value string) {
	ss := s.sessionsOn()
	if ss == nil || thread == "" || value == "" {
		return
	}
	if _, known := ss.sessionOf[thread]; known {
		return
	}
	ss.sessionOf[thread] = value
	if len(ss.members[value]) == 0 {
		ss.distinct++
		if ss.distinct <= sessionsLogged {
			s.logger.V(logutil.DEFAULT).Info("group-scheduler: session id seen (session layer read-back)",
				"session", value, "thread", thread, "n", ss.distinct)
		}
	}
	ss.members[value] = append(ss.members[value], thread)
}

// sessionObserveDispatch is S7/O6 for one dispatched request: whether it carried the header, and whether
// its value disagreed with its thread's session.
func (s *Scheduler) sessionObserveDispatch(h Head) {
	ss := s.sessionsOn()
	if ss == nil {
		return
	}
	s.nca2m.ObserveSessionHeader(h.Session != "")
	if h.Session == "" {
		return
	}
	if v, known := ss.sessionOf[h.Thread]; known && v != h.Session {
		ss.mismatches++
		s.nca2m.ObserveSessionMismatch()
	}
}

// sessionKey is a thread's session: its header value, or the thread itself (S5).
func (s *Scheduler) sessionKey(thread string) string {
	if ss := s.sessionsOn(); ss != nil {
		if v, ok := ss.sessionOf[thread]; ok {
			return sessionKeyHeader + v
		}
	}
	return sessionKeyThread + thread
}

// sessionBeginPass is ST1: every session's admitted, active and seated threads, from the BOUNDED live
// sets — the line entries (an admitted thread has exactly one, on its home), the waiters, the in-flight
// records and the seats — so the cost of a pass does not grow with the history of threads.
func (s *Scheduler) sessionBeginPass() {
	ss := s.sessionsOn()
	if ss == nil {
		return
	}
	clear(ss.facts)
	bump := func(thread string, fn func(*sessFacts)) {
		key := s.sessionKey(thread)
		f := ss.facts[key]
		fn(&f)
		ss.facts[key] = f
	}
	for _, g := range s.groups {
		if g.nca2 == nil {
			continue
		}
		for _, e := range g.nca2.line {
			if s.nca2.admitted[e.thread] && s.nca2.homes[e.thread] == g.id {
				bump(e.thread, func(f *sessFacts) { f.admitted++ })
			}
		}
		for id := range g.threads {
			if !g.index.Exiting(id) {
				bump(id, func(f *sessFacts) { f.seats++ })
			}
		}
	}
	// active = waiting (door or seat) or in flight, each thread once
	for id := range s.waiters {
		bump(id, func(f *sessFacts) { f.active++ })
	}
	for id, n := range s.nca2.inf {
		if _, waiting := s.waiters[id]; n > 0 && !waiting {
			bump(id, func(f *sessFacts) { f.active++ })
		}
	}
	for _, g := range s.groups {
		for id, m := range g.threads {
			if _, waiting := s.waiters[id]; waiting || s.nca2.inf[id] > 0 || m.noQueuedHead() {
				continue
			}
			bump(id, func(f *sessFacts) { f.active++ })
		}
	}
}

// sessionEverIn is ST1's everIn: any thread of the session was ever admitted (memoised for the pass).
func (s *Scheduler) sessionEverIn(key string) bool {
	ss := s.sessionsOn()
	f := ss.facts[key]
	if f.everKnown {
		return f.everIn
	}
	if thread, own := strings.CutPrefix(key, sessionKeyThread); own {
		f.everIn = s.nca2.ever[thread]
	} else {
		value, _ := strings.CutPrefix(key, sessionKeyHeader)
		for _, t := range ss.members[value] {
			if s.nca2.ever[t] {
				f.everIn = true
				break
			}
		}
	}
	f.everKnown = true
	ss.facts[key] = f
	return f.everIn
}

// sessionGroup is ST2: 0 = A (inside), 1 = B (was inside), 2 = C (new).
func (s *Scheduler) sessionGroup(thread string) int {
	key := s.sessionKey(thread)
	if s.sessionsOn().facts[key].admitted > 0 {
		return 0
	}
	if s.sessionEverIn(key) {
		return 1
	}
	return 2
}

// sessionRankKey is R2's key for one outsider: its session's group, and in group A the session's active
// threads (the fewest first, D18).
func (s *Scheduler) sessionRankKey(id string) (group, active int) {
	group = s.sessionGroup(id)
	if group == 0 {
		active = s.sessionsOn().facts[s.sessionKey(id)].active
	}
	return group, active
}

// sessionSeatOrder is L2-2's order of one pod's returners. Part A keeps v2's (the deepest position first,
// then the arrival, then the id); Part B puts the session holding the FEWEST seats cluster-wide first
// (D25) and keeps v2's order as the tie-break — a stable sort of v2's total order.
func (s *Scheduler) sessionSeatOrder(warm []*waiter) []*waiter {
	ss := s.sessionsOn()
	if ss == nil {
		return warm
	}
	sort.SliceStable(warm, func(i, j int) bool {
		return ss.facts[s.sessionKey(warm[i].id)].seats < ss.facts[s.sessionKey(warm[j].id)].seats
	})
	return warm
}

// doorSessionLabels is O7's session group (A | B | C, "none" in Part A) and the 10-hex hash of the session
// id ("" in Part A), for a thread at the decision ("" and "" when the door is off).
func (s *Scheduler) doorSessionLabels(id string) (group, hash string) {
	if s.nca2 == nil || s.nca2.door == nil {
		return "", ""
	}
	if s.sessionsOn() == nil {
		return "none", ""
	}
	group = [...]string{"A", "B", "C"}[s.sessionGroup(id)]
	key := s.sessionKey(id)
	sum := sha256.Sum256([]byte(key[len(sessionKeyHeader):])) // both prefixes have the same length
	return group, hex.EncodeToString(sum[:])[:10]
}
