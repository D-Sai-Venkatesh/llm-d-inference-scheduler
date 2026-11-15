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
// only — who is considered first at the door (R2), and who gets a freed seat (L2-2).

// nca2Sessions is Part B's state.
type nca2Sessions struct{}

// sessionBeginPass derives the pass's session facts (ST1); nothing in Part A.
func (s *Scheduler) sessionBeginPass() {}

// sessionRankKey is R2's key for one outsider.
func (s *Scheduler) sessionRankKey(id string) (group, active int) {
	if s.nca2.ever[id] {
		return 1, 0
	}
	return 2, 0
}

// sessionSeatOrder is L2-2's order of one pod's returners: v2's (deepest first) in Part A.
func (s *Scheduler) sessionSeatOrder(warm []*waiter) []*waiter { return warm }

// doorSessionLabels is O7's session group and session-id hash for a thread ("" when the door is off).
func (s *Scheduler) doorSessionLabels(id string) (group, hash string) {
	if s.nca2 == nil || s.nca2.door == nil {
		return "", ""
	}
	return "none", ""
}
