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

// THE SAFETY NET (one definition, two readers: this package's per-engine holds and
// the framework's detector plugin in detector.go, which is the SAME module in the
// framework's detector slot).
//
// The SEATS are the admission control: an engine's seat budget (its own KV
// capacity × GroupBudgetFraction) is what bounds its memory, and a freed seat
// takes the next waiter from the shared queue. The net below is NOT a second
// admission policy — it is the backstop for the one thing the budget cannot see:
// a request's footprint GROWING past its admission estimate while it runs.
//
// Three properties, and each one is a deletion:
//
//  1. MEMORY ONLY. The old per-engine gate was
//     max(waiting/queueThreshold, kv/kvThreshold) >= 1 with queueThreshold=5 —
//     a raw request COUNT, not capacity-relative. On the E-wave that closed the
//     gate on a 4-seat engine at 3 % KV (5 waiting requests) and locked the
//     engine out of every new seat (the e4 defect's second half). The queue term
//     is therefore GONE — deleted from the code, not set to a big number.
//  2. THE THRESHOLD IS THE BUDGET'S OWN FRACTION. The budget already assumes an
//     engine may run to GroupBudgetFraction of its KV capacity; the net fires when
//     the MEASURED kv utilisation reaches exactly that fraction. No new constant,
//     and a swept budget fraction moves both together.
//  2b. STALE / ABSENT METRICS ARE OPEN. The budget is the primary control; a net
//     that failed closed on a scrape gap would add a stall mode (and the stock
//     detector's fail-closed staleness constant with it).
//  3. IT NEVER INFLUENCES PLACEMENT. While an engine's net is closed it holds
//     its OWN dispatches and takes no NEW seats; it is not ranked, penalised or
//     skipped when another engine is chosen. Excluded-from-candidates is a gate,
//     not a placement rule: the second a fire clears, the engine competes for
//     the head on exactly the same key as everybody else.
//
// The FIRE is counted (events) and TIMED (seconds fired) per engine, because a
// safety net nobody can see is a safety net nobody can trust — and the numbers
// are also the liveness gate of a wave ("fires <= 1 %").
//
// ONE DEFINITION, TWO READERS (the D3 fix): safetyFiredLocked is the decision; the
// metric series reads the same function through SafetyState. The old metric
// re-derived closure from the literals `st.KV >= 0.80 || st.Waiting >= 5` and
// would have reported a gate that was not the gate the moment a knob was swept.

import (
	"sort"
	"time"
)

// engineLoad is ONE engine's pushed utilisation snapshot (the plugin's metrics
// poll writes it; the policy never scrapes and never blocks).
type engineLoad struct {
	kv      float64
	waiting int
	at      time.Time
}

// SetEngineLoad records one engine's scrape. A zero `at` clears the entry (a
// vanished engine must not keep a stale verdict).
func (s *Scheduler) SetEngineLoad(engine string, kv float64, waiting int, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.load == nil {
		s.load = map[string]engineLoad{}
	}
	g := s.groups[engine]
	if at.IsZero() {
		delete(s.load, engine)
		if g != nil {
			s.settleSafetyLocked(g, false, at) // clear a latched fire
		}
		return
	}
	s.load[engine] = engineLoad{kv: kv, waiting: waiting, at: at}
	if g != nil {
		s.settleSafetyLocked(g, s.safetyFiredLocked(g), at)
	}
}

// ForgetEngineLoadsExcept drops snapshots for engines the EPP no longer sees.
func (s *Scheduler) ForgetEngineLoadsExcept(keep map[string]struct{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k := range s.load {
		if _, ok := keep[k]; !ok {
			delete(s.load, k)
		}
	}
}

// safetyFiredLocked is THE definition of the net, and the only one: the SAFETY
// module's verdict on this engine's pushed scrape.
//
// An engine with no snapshot (or a stale one) is NOT firing: the seat budget is the
// primary control, so a detector that failed closed would only add a stall mode of
// its own — and it would remove the framework's fail-closed time constant from the
// path. That is the declared trade: the net is a HEADROOM guard, not the admission
// authority.
func (s *Scheduler) safetyFiredLocked(g *engineGroup) bool {
	if g == nil {
		return false
	}
	l := s.load[g.id]
	have := !l.at.IsZero()
	return s.mod.safety.Fired(l.kv, have)
}

// settleSafetyLocked books a fire TRANSITION (a fire event) and the time spent
// fired. Deterministic by construction: it reads the decision pass's virtual
// clock, never time.Now.
func (s *Scheduler) settleSafetyLocked(g *engineGroup, fired bool, now time.Time) {
	if fired == g.safetyFiredOn {
		return
	}
	g.safetyFiredOn = fired
	if fired {
		g.counters.safetyFires++
		g.safetySince = now
		return
	}
	if !g.safetySince.IsZero() && now.After(g.safetySince) {
		g.counters.safetyFiredSeconds += now.Sub(g.safetySince).Seconds()
	}
	g.safetySince = time.Time{}
}

// SafetyState is one engine's net state, exported for the metrics layer (the 15 s
// per-engine series the wave's BALANCE gate reads). It answers from the SAME
// definition the decision path uses.
type SafetyState struct {
	Engine string
	KV     float64
	// Waiting is the engine's scraped waiting-queue depth. It is REPORTING only:
	// the queue term is not part of the net any more (see the file header).
	Waiting int
	// Fired is the live verdict; Fires counts fire EVENTS (transitions);
	// FiredSeconds is the accumulated time spent fired (including the open
	// interval).
	Fired        bool
	Fires        int64
	FiredSeconds float64
}

// safetyStateLocked is one engine's net state as of now (the fired-seconds
// include the open interval).
func (s *Scheduler) safetyStateLocked(g *engineGroup, now time.Time) SafetyState {
	st := SafetyState{Engine: g.id, Fires: g.counters.safetyFires,
		FiredSeconds: g.counters.safetyFiredSeconds}
	if l, ok := s.load[g.id]; ok && !l.at.IsZero() {
		st.KV, st.Waiting = l.kv, l.waiting
	}
	st.Fired = s.safetyFiredLocked(g)
	if st.Fired && !g.safetySince.IsZero() && now.After(g.safetySince) {
		st.FiredSeconds += now.Sub(g.safetySince).Seconds()
	}
	return st
}

// SafetyStates snapshots every engine's net state (sorted by engine id) — the
// metrics layer's ONE reader of the definition above (the D3 fix).
func (s *Scheduler) SafetyStates() []SafetyState {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]SafetyState, 0, len(s.groupOrder))
	for _, id := range s.groupOrder {
		if g := s.groups[id]; g != nil {
			out = append(out, s.safetyStateLocked(g, s.now))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Engine < out[j].Engine })
	return out
}
