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
	"fmt"
	"strings"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestMemberTierEvictionNeedsOwnTierIndex(t *testing.T) {
	cfg := armCConfig(2000, 8, "compute")
	cfg.MemberTierEviction = true
	if _, err := NewScheduler(cfg); err != nil {
		t.Fatalf("memberTierEviction on arm C's line must be accepted: %v", err)
	}
	cfg.OwnTierIndex = false
	cfg.LeastRecentlyServed, cfg.FairRotation, cfg.CostFairRotation, cfg.ClaimOrder = false, false, false, ""
	if _, err := NewScheduler(cfg); err == nil || !strings.Contains(err.Error(), "memberTierEviction needs ownTierIndex") {
		t.Fatalf("memberTierEviction without ownTierIndex must be refused, got %v", err)
	}
}

func TestLeaveStayReportsUnsharedKeysInChainOrder(t *testing.T) {
	b, _ := NewBlockIndex(64)
	var got [][]Hash
	b.onLeaveStay = func(keys []Hash) { got = append(got, append([]Hash(nil), keys...)) }
	a := chainWithPrefix("head", 2, "A", 3)
	bb := chainWithPrefix("head", 2, "B", 2)
	b.insert("A", a)
	b.insert("B", bb)
	b.MarkExiting("A")
	if len(got) != 1 || fmt.Sprint(got[0]) != fmt.Sprint(a[0].Keys[2:]) {
		t.Fatalf("A's exit must report its own keys in chain order (the shared head stays with B): %v", got)
	}
	b.Unseat("A")
	if len(got) != 1 {
		t.Fatal("an exiting thread's unseat must not report again")
	}
	b.Unseat("B") // B leaves without a mark: now the head has no staying holder either
	if len(got) != 2 || fmt.Sprint(got[1]) != fmt.Sprint(bb[0].Keys) {
		t.Fatalf("B's unseat must report all its keys, the head included, in chain order: %v", got)
	}
}

func TestMemberNoticeListsTheStayingThreads(t *testing.T) {
	cfg := armCConfig(2000, 8, "compute")
	cfg.MemberTierEviction = true
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetNumBlocksFor("a", 64, 500); err != nil {
		t.Fatal(err)
	}
	g := s.groups["a"]
	g.index.insert("t2", chain("t2", 3))
	g.index.insert("t1", chain("t1", 3))
	g.index.insert("t3", chain("t3", 3))
	g.index.MarkExiting("t3")
	s.reqs["r1"], s.reqs["r3"] = &reqRec{thread: "t1"}, &reqRec{thread: "t3"}
	n, ok := s.MemberNotice("a", "r1")
	if !ok || fmt.Sprint(n.Members) != "[t1 t2]" || n.Thread != "t1" || n.Source == "" {
		t.Fatalf("notice for a staying member = %+v ok=%v", n, ok)
	}
	if n3, _ := s.MemberNotice("a", "r3"); n3.Thread != "" {
		t.Fatal("an exiting thread must not be named (the policy records only members)")
	}
	if nu, _ := s.MemberNotice("a", "no-such-request"); nu.Thread != "" || len(nu.Members) != 2 {
		t.Fatal("an untracked request carries the list, unnamed")
	}
	if n2, _ := s.MemberNotice("a", "r1"); n2.Source != n.Source {
		t.Fatal("one engine group keeps one source")
	}
	off, _ := NewScheduler(armCConfig(2000, 8, "compute"))
	_ = off.SetNumBlocksFor("a", 64, 500)
	if _, ok := off.MemberNotice("a", "r1"); ok {
		t.Fatal("with memberTierEviction off no notice is produced")
	}
	// the body keeps every other kv_transfer_params field
	req := &fwksched.InferenceRequest{RequestID: "r", Headers: map[string]string{},
		Body: &fwkrequest.InferenceRequestBody{Payload: fwkrequest.PayloadMap{
			"model": "m", reqcommon.FieldKVTransferParams: map[string]any{"max_offload_tokens": 7}}}}
	if !writeMemberNotice(req, n) || !req.Body.Mutated {
		t.Fatal("the notice must be written and the body marked mutated")
	}
	params := req.Body.Payload.(fwkrequest.PayloadMap)[reqcommon.FieldKVTransferParams].(map[string]any)
	if params["max_offload_tokens"] != 7 {
		t.Fatal("other kv_transfer_params fields must be kept")
	}
	gm := params[memberNoticeKey].(map[string]any)
	if fmt.Sprint(gm["members"]) != "[t1 t2]" || gm["thread"] != "t1" || gm["epoch"] != n.Epoch {
		t.Fatalf("notice body = %v", gm)
	}
}

func TestMemberTierFairCostSavesUpToTheLeaversBlocks(t *testing.T) {
	for _, on := range []bool{false, true} {
		cfg := armCConfig(2000, 8, "compute")
		cfg.MemberTierEviction = on
		s, _ := NewScheduler(cfg)
		_ = s.SetNumBlocksFor("a", 64, 500)
		g := s.groups["a"]
		m := chain("m", 4)
		g.index.insert("m", m)
		// the head's write reaches 6 staying blocks, one of them m's
		ft := fairTail{bottom: map[Hash]struct{}{m[0].Keys[3]: {}}, t0: 6}
		cost, saved := s.fairCost(g, &thread{id: "m"}, ft)
		if !on && (saved != 1 || cost != 4+6-1) {
			t.Fatalf("LRU: saved = m's blocks already in the zone (1), cost = excl+t0-saved (9); got %d %d", saved, cost)
		}
		if on && (saved != 4 || cost != 6) {
			t.Fatalf("member eviction: m's 4 blocks all go to the eviction end, cost = max(excl, t0) = 6; got saved %d cost %d", saved, cost)
		}
	}
}

// the replay: with memberTierEviction the group's own tier index evicts fewer SEATED blocks, and it demotes
// leavers' blocks; off, nothing is demoted.
func TestMemberTierReplayEvictsFewerSeatedBlocks(t *testing.T) {
	run := func(seed, tier int64, on bool) (seated, demoted int64) {
		cfg := armCConfig(tier, 8, "compute")
		cfg.MemberTierEviction = on
		s, err := NewScheduler(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		driveReplay(t, s, seed, 500, &out, replayOpts{noMoves: true, engines: []string{"a"}}, func(int) {})
		for _, v := range s.counters.seatedEvicted {
			seated += v
		}
		return seated, s.counters.tierDemoted
	}
	var offSum, onSum int64
	for _, tier := range []int64{700, 1000} {
		for seed := int64(21); seed <= 24; seed++ {
			so, do := run(seed, tier, false)
			sn, dn := run(seed, tier, true)
			if do != 0 {
				t.Fatalf("off must demote nothing (got %d)", do)
			}
			if dn == 0 {
				t.Fatalf("tier %d seed %d: on must demote leavers' blocks", tier, seed)
			}
			offSum += so
			onSum += sn
			t.Logf("tier %d seed %d: seated evictions off %d on %d, demoted %d", tier, seed, so, sn, dn)
		}
	}
	if onSum >= offSum {
		t.Fatalf("membership eviction must evict fewer seated blocks: off %d on %d", offSum, onSum)
	}
}

// THE REAL PATH (member2-void1's defect): at the PreRequest moment the notice must name the dispatched turn's
// thread from the scheduler's own request record, for (nearly) every tracked dispatch — and only staying members.
func TestMemberNoticeNamesTheDispatchedThread(t *testing.T) {
	cfg := armCConfig(1000, 8, "compute")
	cfg.MemberTierEviction = true
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var tracked, named int
	opts := replayOpts{noMoves: true, engines: []string{"a"}, atPod: func(rid, eng string) {
		n, ok := s.MemberNotice(eng, rid)
		if !ok {
			t.Fatalf("no notice for %s on %s", rid, eng)
		}
		rec := s.reqs[rid]
		if rec == nil {
			return // anonymous traffic: unnamed by design
		}
		tracked++
		if n.Thread == "" {
			return
		}
		named++
		if n.Thread != rec.thread {
			t.Fatalf("named %q, record says %q", n.Thread, rec.thread)
		}
		found := false
		for _, m := range n.Members {
			found = found || m == n.Thread
		}
		if !found {
			t.Fatalf("named thread %q is not in the member list", n.Thread)
		}
	}}
	var out strings.Builder
	driveReplay(t, s, 22, 500, &out, opts, func(int) {})
	t.Logf("tracked dispatches %d, named %d", tracked, named)
	if tracked == 0 || named*10 < tracked*9 {
		t.Fatalf("the notice must name the thread of (nearly) every tracked dispatch: %d of %d", named, tracked)
	}
}
