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

// MEMBERSHIP-AWARE CPU-TIER EVICTION (memberTierEviction): the engine's cache policy
// (vllm-plugin/gsch_member_policy.py) moves a departed member's chunks to its LRU's eviction end. It learns
// who the members are from every request this group sends it, in kv_transfer_params["gsch_member"]:
//
//	{"thread": <this request's thread, when it is a staying member here>, "members": [<staying threads here>],
//	 "epoch": <the engine's block-index generation: grows with every seat/exit/growth>,
//	 "source": <this engine group's incarnation: a new group (or EPP) restarts the epoch>}
//
// The full list rides on every request, so a lost or reordered request costs nothing: the policy keeps the
// newest list it has seen per source.

import (
	"crypto/rand"
	"encoding/hex"
	"sort"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

// memberNoticeKey is the kv_transfer_params field the engine's policy reads (gsch_member_policy.PARAMS_KEY).
const memberNoticeKey = "gsch_member"

// MemberNotice is what one request tells the engine's cache policy.
type MemberNotice struct {
	Thread  string
	Members []string
	Epoch   uint64
	Source  string
}

// newIncarnation names one engine group's life (the notice's source).
func newIncarnation() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "x"
	}
	return hex.EncodeToString(b)
}

// StayingIDs is the seated threads that are not exiting, sorted.
func (b *BlockIndex) StayingIDs() []string {
	out := make([]string, 0, len(b.seats))
	for id, s := range b.seats {
		if !s.exiting {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// MemberNotice is the notice for a request sent to engine on behalf of thread (thread may be "": an
// untracked request still carries the list). ok is false when memberTierEviction is off or the engine has
// no group with a tier index.
func (s *Scheduler) MemberNotice(engine, thread string) (MemberNotice, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.cfg.MemberTierEviction {
		return MemberNotice{}, false
	}
	g, ok := s.groups[engine]
	if !ok || g.tier == nil {
		return MemberNotice{}, false
	}
	if g.incarnation == "" {
		g.incarnation = newIncarnation()
	}
	n := MemberNotice{Members: g.index.StayingIDs(), Epoch: g.index.Gen(), Source: g.incarnation}
	if sb, seated := g.index.seats[thread]; seated && !sb.exiting {
		n.Thread = thread
	}
	return n, true
}

// writeMemberNotice puts n into the request body's kv_transfer_params (other fields kept); it reports whether
// the body changed. Paths whose engine forbids unknown fields (/v1/systemone, /v1/decisions) are skipped,
// as selectivekv does.
func writeMemberNotice(request *fwksched.InferenceRequest, n MemberNotice) bool {
	if request == nil || request.Body == nil || request.Body.Payload == nil || request.Body.Generate != nil {
		return false
	}
	if path := fwkrequest.GetRequestPath(request.Headers); fwkrequest.MatchPathSuffix(path, fwkrequest.SystemOnePathSuffix) ||
		fwkrequest.MatchPathSuffix(path, fwkrequest.DecisionsPathSuffix) {
		return false
	}
	payload, ok := request.Body.Payload.(fwkrequest.PayloadMap)
	if !ok {
		return false
	}
	params, decoded := parserutil.ObjectField(payload, reqcommon.FieldKVTransferParams)
	if !decoded || params == nil {
		params = map[string]any{}
	}
	members := make([]any, len(n.Members))
	for i, m := range n.Members {
		members[i] = m
	}
	notice := map[string]any{"members": members, "epoch": n.Epoch, "source": n.Source}
	if n.Thread != "" {
		notice["thread"] = n.Thread
	}
	params[memberNoticeKey] = notice
	payload[reqcommon.FieldKVTransferParams] = params
	request.Body.Mutated = true
	return true
}
