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

// THE OBSERVED ENGINE WINS. This test was written for the reviewable set and has no counterpart in
// the tree the rest of this suite came from, so it lives here on its own: the framework picks the
// endpoint, and an observation on another engine MOVES the thread — with its blocks — instead of
// letting the accounting drift from the engine that is really serving it.

import (
	"testing"
	"time"
)

func TestRebindFollowsTheObservedEngine(t *testing.T) {
	now := time.Unix(1793000000, 0).UTC()
	h := Head{
		Thread: "m", ReqID: "r-m", Arrival: now,
		Chains: []Chain{{Keys: keys(100, seedFor("m")), Tokens: 100 * 64}},
	}

	s, err := NewScheduler(defaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []string{"e0", "e1"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}
	if i := s.Pass([]Head{h}, now); i != 0 {
		t.Fatalf("the thread must be seated, got %d", i)
	}

	s.mu.Lock()
	charged := s.groupOf("m").id
	s.mu.Unlock()
	if charged != "e0" {
		t.Fatalf("fixture: the seat-time choice must be e0, got %q", charged)
	}

	s.AtPod("r-m", "e1", now) // the request lands on the OTHER engine

	s.mu.Lock()
	defer s.mu.Unlock()
	if got := s.groupOf("m").id; got != "e1" {
		t.Fatalf("the observed engine must win: bound to %q, want e1", got)
	}
	if got := s.groups[charged].index.ChargedBlocks("m"); got != 0 {
		t.Fatalf("the blocks must leave the engine that is not serving the thread, %d left", got)
	}
	if got := s.groups["e1"].index.ChargedBlocks("m"); got != 100 {
		t.Fatalf("the serving engine must carry the thread's 100 blocks, got %d", got)
	}
	if s.counters.affinityViolations != 1 {
		t.Fatalf("one observation, one violation: %d", s.counters.affinityViolations)
	}
}
