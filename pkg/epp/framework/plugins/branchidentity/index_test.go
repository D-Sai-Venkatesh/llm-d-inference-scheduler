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

package branchidentity

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stream builds a synthetic pseudo-token stream of n whole blocks: every token is
// seed-derived, so two streams with the same seed share a prefix and differ only
// after their common length (an append-only conversation).
func stream(blocks int, seed uint32) []uint32 {
	tokens := make([]uint32, blocks*defaultBlockSizeTokens)
	for i := range tokens {
		tokens[i] = seed*uint32(defaultBlockSizeTokens*1000) + uint32(i)
	}
	return tokens
}

// chainOf is the chain a request would have whose canonical stream ends after n
// whole blocks of tokens.
func chainOf(t *testing.T, tokens []uint32, blocks int) []Hash {
	t.Helper()
	require.LessOrEqual(t, blocks*defaultBlockSizeTokens, len(tokens))
	return ChainedBlockHashes(tokens[:blocks*defaultBlockSizeTokens], "", defaultBlockSizeTokens)
}

func newTestIndex() *branchIndex {
	return newBranchIndex(defaultBlockSizeTokens, "", time.Minute, 1000)
}

// turn is one complete sequential turn: arrival, dispatch, settle — the shape of
// a conversation that waits for each reply.
func turn(ix *branchIndex, at time.Time, chain []Hash, scope string, roles []string) ResolveResult {
	res := ix.Resolve(at, chain, 0, scope, roles, true, true)
	ix.Dispatch(res.ID, at)
	ix.Settle(res.ID, at)
	return res
}

// TestChainedBlockHashes_DropsPartialTail: only whole blocks are chained, so a
// continuation whose last block is still being written still matches the tip.
func TestChainedBlockHashes_DropsPartialTail(t *testing.T) {
	full := stream(3, 1)
	partial := append(append([]uint32{}, full...), make([]uint32, defaultBlockSizeTokens/2)...)
	assert.Len(t, ChainedBlockHashes(partial, "", defaultBlockSizeTokens), 3)
	assert.Len(t, ChainedBlockHashes(full, "", defaultBlockSizeTokens), 3)
	assert.Equal(t, ChainedBlockHashes(full, "", defaultBlockSizeTokens),
		ChainedBlockHashes(partial, "", defaultBlockSizeTokens),
		"a partial trailing block must not change the chain")
}

// TestChainedBlockHashes_Deterministic: the same stream and salt always produce
// the same chain (the id-stability requirement), and the chain is genuinely
// chained — the same block at a different position hashes differently.
func TestChainedBlockHashes_Deterministic(t *testing.T) {
	tokens := stream(4, 7)
	assert.Equal(t, ChainedBlockHashes(tokens, "salt", 64), ChainedBlockHashes(tokens, "salt", 64))
	assert.NotEqual(t, ChainedBlockHashes(tokens, "salt", 64), ChainedBlockHashes(tokens, "other", 64))

	shifted := append([]uint32{99}, tokens[:len(tokens)-1]...)
	assert.NotEqual(t, ChainedBlockHashes(tokens, "salt", 64)[0], ChainedBlockHashes(shifted, "salt", 64)[0])
}

// TestResolve_Continuation: growing the conversation continues the SAME branch,
// and an identical retry does too.
func TestResolve_Continuation(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(8, 1)

	first := turn(ix, now, chainOf(t, tokens, 3), "", []string{"user"})
	require.Equal(t, OutcomeNew, first.Outcome)

	grown := turn(ix, now, chainOf(t, tokens, 5), "", []string{"user", "assistant", "user"})
	assert.Equal(t, OutcomeMatched, grown.Outcome)
	assert.Equal(t, first.ID, grown.ID, "a continuation keeps the branch id")
	assert.Equal(t, 3, grown.MatchedBlocks)
	assert.False(t, grown.StickyBusy, "an idle tip needs no content proof")

	retry := ix.Resolve(now, chainOf(t, tokens, 5), 0, "", []string{"user", "assistant", "user"}, true, true)
	assert.Equal(t, OutcomeMatched, retry.Outcome)
	assert.Equal(t, first.ID, retry.ID)
	assert.Len(t, ix.branches, 1)
}

// TestResolve_ForkIfBusy: extending a BUSY tip forks a new branch (a sub-agent
// inheriting the parent's context), and the fork's own continuations stay on the
// fork.
func TestResolve_ForkIfBusy(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(12, 2)

	parent := ix.Resolve(now, chainOf(t, tokens, 3), 0, "", []string{"user"}, true, true)
	require.True(t, ix.Dispatch(parent.ID, now), "the parent turn is dispatched (busy)")

	// The child inherits the parent's context and appends its own instruction: no
	// assistant message past the tip, so nothing proves the parent completed.
	fork := ix.Resolve(now, chainOf(t, tokens, 5), 0, "",
		[]string{"user", "user", "user", "user"}, true, true)
	assert.Equal(t, OutcomeForked, fork.Outcome)
	assert.NotEqual(t, parent.ID, fork.ID, "a fork gets its own queue key")
	assert.Equal(t, 3, fork.MatchedBlocks)

	// The child dispatches and completes; a request extending the FORK's tip
	// continues the fork (the longest tip prefix wins over the parent's tip).
	ix.Dispatch(fork.ID, now)
	ix.Settle(fork.ID, now)
	forkTurn := turn(ix, now, chainOf(t, tokens, 6), "", nil)
	assert.Equal(t, OutcomeMatched, forkTurn.Outcome)
	assert.Equal(t, fork.ID, forkTurn.ID)

	// A chain shorter than every tip matches nothing, so it is a new branch.
	shorter := ix.Resolve(now, chainOf(t, tokens, 2), 0, "", nil, true, true)
	assert.Equal(t, OutcomeNew, shorter.Outcome)
	assert.NotEqual(t, parent.ID, shorter.ID)
}

// TestResolve_SamePromptConcurrentIsASibling: two IDENTICAL prompts in flight at
// once are two parallel siblings, the one case content cannot separate — so the
// busy flag forks them. A repeat of a FINISHED turn continues (a retry).
func TestResolve_SamePromptConcurrentIsASibling(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(5, 3)
	same := chainOf(t, tokens, 4)

	first := ix.Resolve(now, same, 0, "", nil, true, true)
	ix.Dispatch(first.ID, now)

	concurrent := ix.Resolve(now, same, 0, "", nil, true, true)
	assert.Equal(t, OutcomeForked, concurrent.Outcome, "a same-prompt duplicate in flight gets its own seat")
	assert.NotEqual(t, first.ID, concurrent.ID)

	// Once the first turn completed, an identical body is a retry: same branch.
	ix.Settle(first.ID, now)
	ix.Settle(concurrent.ID, now)
	retry := ix.Resolve(now, same, 0, "", nil, true, true)
	assert.Equal(t, OutcomeMatched, retry.Outcome)
	_, ok := ix.branches[retry.ID]
	assert.True(t, ok)
}

// TestResolve_StickyBusyGuard (H1): a request that extends a busy tip AND begins
// with the tip's assistant reply contains that reply, which an in-flight request
// cannot have — proven completion, so it continues the branch.
func TestResolve_StickyBusyGuard(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 4)
	tipRoles := []string{"system", "user", "tool"}

	first := ix.Resolve(now, chainOf(t, tokens, 3), 0, "", tipRoles, true, true)
	ix.Dispatch(first.ID, now) // never settles: the busy flag is stale

	// First message past the tip is the assistant reply → proven continuation.
	withReply := ix.Resolve(now, chainOf(t, tokens, 5), 0, "",
		append(append([]string{}, tipRoles...), "assistant", "tool"), true, true)
	assert.Equal(t, OutcomeMatched, withReply.Outcome)
	assert.Equal(t, first.ID, withReply.ID)
	assert.True(t, withReply.StickyBusy, "the guard must report that it overrode the busy flag")
}

// TestResolve_StickyBusyGuard_NotProven: the guard must not fire for a sibling
// (nothing of the tip's reply in the request), and is configurable — the arm that
// measures how much of the fork rate is stale-busy noise.
func TestResolve_StickyBusyGuard_NotProven(t *testing.T) {
	tokens := stream(6, 5)
	now := time.Now()
	tipRoles := []string{"system", "user", "tool"}
	siblingRoles := append(append([]string{}, tipRoles...), "user", "user")

	live := newTestIndex()
	first := live.Resolve(now, chainOf(t, tokens, 3), 0, "", tipRoles, true, true)
	live.Dispatch(first.ID, now)
	sibling := live.Resolve(now, chainOf(t, tokens, 5), 0, "", siblingRoles, true, true)
	assert.Equal(t, OutcomeForked, sibling.Outcome)
	assert.NotEqual(t, first.ID, sibling.ID)

	off := newTestIndex()
	second := off.Resolve(now, chainOf(t, tokens, 3), 0, "", tipRoles, true, true)
	off.Dispatch(second.ID, now)
	withReply := off.Resolve(now, chainOf(t, tokens, 5), 0, "",
		append(append([]string{}, tipRoles...), "assistant", "tool"), true, false)
	assert.Equal(t, OutcomeForked, withReply.Outcome, "with the guard off, a busy tip always forks")
	assert.NotEqual(t, second.ID, withReply.ID)
}

// TestResolve_ForkOnBusyDisabled: with the knob off every match continues (the
// v2c failure mode, kept as a measurement arm only).
func TestResolve_ForkOnBusyDisabled(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 6)

	first := ix.Resolve(now, chainOf(t, tokens, 3), 0, "", nil, true, true)
	ix.Dispatch(first.ID, now)

	shared := ix.Resolve(now, chainOf(t, tokens, 5), 0, "", nil, false, true)
	assert.Equal(t, OutcomeMatched, shared.Outcome)
	assert.Equal(t, first.ID, shared.ID)
}

// TestResolve_SessionScope: tips are scoped by client session when the request
// carries one, so look-alike traces of two sessions do not merge; an unscoped tip
// matches any session (the design's accepted floor).
func TestResolve_SessionScope(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 7)

	a := turn(ix, now, chainOf(t, tokens, 3), "sess-a", nil)
	b := turn(ix, now, chainOf(t, tokens, 3), "sess-b", nil)
	assert.NotEqual(t, a.ID, b.ID, "same body, different session → different branch")

	againA := turn(ix, now, chainOf(t, tokens, 4), "sess-a", nil)
	assert.Equal(t, a.ID, againA.ID, "the same session continues its own branch")

	// An unscoped request may continue a scoped tip (no session header to scope by).
	unscoped := turn(ix, now, chainOf(t, tokens, 5), "", nil)
	assert.Equal(t, OutcomeMatched, unscoped.Outcome)

	// An unscoped tip is visible to a scoped request too — and two instances
	// derive the same id for the same content (ids need no registry to agree).
	ix2 := newTestIndex()
	open := turn(ix2, now, chainOf(t, tokens, 3), "", nil)
	scoped := turn(ix2, now, chainOf(t, tokens, 4), "sess-c", nil)
	assert.Equal(t, open.ID, scoped.ID)
}

// TestResolve_NoChainUsesStreamHash: a body shorter than one block has no chain,
// so its id comes from the whole-stream hash — distinct short conversations stay
// distinct, an identical retry reuses the id.
func TestResolve_NoChainUsesStreamHash(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()

	shortA := ix.Resolve(now, nil, StreamHash([]byte("hi")), "", nil, true, true)
	shortB := ix.Resolve(now, nil, StreamHash([]byte("yo")), "", nil, true, true)
	assert.NotEqual(t, shortA.ID, shortB.ID)

	retry := ix.Resolve(now, nil, StreamHash([]byte("hi")), "", nil, true, true)
	assert.Equal(t, shortA.ID, retry.ID)
}

// TestNoTimeGC_DormantBranchKeepsItsID is the stage-0 retention rule: branches
// are never dropped for being silent. The measured p99.9 inter-turn idle gap is
// 3,036 seconds, so a thread that comes back after a long tool/human wait must
// land on its own branch — a silence GC would mint it a false NEW id and
// re-introduce newcomer churn.
func TestNoTimeGC_DormantBranchKeepsItsID(t *testing.T) {
	ix := newTestIndex()
	base := time.Now()
	tokens := stream(9, 8)

	first := turn(ix, base, chainOf(t, tokens, 3), "", nil)
	// Two hours later — far past any plausible T_gc — the sweep removes nothing.
	assert.Equal(t, 0, ix.EnforceCap(base.Add(2*time.Hour)))
	again := turn(ix, base.Add(2*time.Hour), chainOf(t, tokens, 5), "", nil)
	assert.Equal(t, OutcomeMatched, again.Outcome, "a dormant thread is not a new thread")
	assert.Equal(t, first.ID, again.ID)
}

// TestOutstandingBranchIsNeverEvicted: neither the cap nor the sweep removes a
// branch with anything in flight or queued (a dropped branch would abort the
// busy bookkeeping and mint a false new id on the next turn).
func TestOutstandingBranchIsNeverEvicted(t *testing.T) {
	ix := newBranchIndex(defaultBlockSizeTokens, "", 5*time.Minute, 1)
	base := time.Now()
	tokens := stream(9, 8)

	busy := ix.Resolve(base, chainOf(t, tokens, 3), 0, "", nil, true, true)
	ix.Dispatch(busy.ID, base)
	queued := ix.Resolve(base, chainOf(t, tokens, 4), 0, "", nil, true, true) // arrival only

	assert.Equal(t, 0, ix.EnforceCap(base.Add(time.Minute)), "over the cap, but everything is outstanding")
	ix.Settle(busy.ID, base)

	// Once nothing is outstanding, the cap collects, oldest first.
	assert.Equal(t, 1, ix.EnforceCap(base.Add(time.Minute)))
	assert.Equal(t, 1, ix.Len())
	assert.Len(t, ix.byTip, 1, "eviction must keep the tip index in sync")
	_ = queued
}

// TestCap_MaxBranches: the hard cap evicts the idlest non-outstanding
// branches and never a busy one.
func TestCap_MaxBranches(t *testing.T) {
	ix := newBranchIndex(defaultBlockSizeTokens, "", time.Minute, 3)
	base := time.Now()

	var ids []string
	for i := 1; i <= 5; i++ {
		at := base.Add(time.Duration(i) * time.Second)
		// A distinct stream per branch: unrelated conversations.
		res := turn(ix, at, chainOf(t, stream(2, uint32(i)), 2), "", nil)
		ids = append(ids, res.ID)
	}
	busy := ix.Resolve(base.Add(6*time.Second), chainOf(t, stream(2, 99), 2), 0, "", nil, true, true)
	ix.Dispatch(busy.ID, base.Add(6*time.Second))

	ix.EnforceCap(base.Add(7 * time.Second))
	assert.LessOrEqual(t, ix.Len(), 4, "the cap must bound the table (one over, for the busy branch)")
	_, kept := ix.branches[ids[4]]
	assert.True(t, kept, "the newest branch survives")
	_, keptBusy := ix.branches[busy.ID]
	assert.True(t, keptBusy, "a busy branch is never evicted by the cap")
	_, keptOldest := ix.branches[ids[0]]
	assert.False(t, keptOldest, "the idlest branch is evicted first")
}

// TestSnapshot_Bounded: the debug dump reports the table shape and stays small.
func TestSnapshot_Bounded(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	res := ix.Resolve(now, chainOf(t, stream(4, 1), 3), 0, "s1", nil, true, true)
	ix.Dispatch(res.ID, now)

	snap := ix.Snapshot(now)
	require.Equal(t, 1, snap.Branches)
	assert.Equal(t, 1, snap.Busy)
	assert.Equal(t, 1, snap.InFlight)
	require.Len(t, snap.Rows, 1)
	assert.Equal(t, res.ID, snap.Rows[0].ID)
	assert.Equal(t, "s1", snap.Rows[0].Scope)
	assert.Equal(t, 3, snap.Rows[0].Blocks)

	big := newBranchIndex(defaultBlockSizeTokens, "", time.Minute, 100000)
	for i := 0; i < maxSnapshotRows+20; i++ {
		big.Resolve(now, chainOf(t, stream(2, uint32(i+1)), 2), 0, "", nil, true, true)
	}
	assert.LessOrEqual(t, len(big.Snapshot(now).Rows), maxSnapshotRows)
}

// TestResolve_Concurrent: the table is safe under concurrent arrivals. Run with
// -race to make this meaningful.
func TestResolve_Concurrent(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 11)
	chains := make([][]Hash, 8)
	for i := range chains {
		chains[i] = chainOf(t, tokens, 3)
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res := ix.Resolve(now, chains[i%len(chains)], 0, "", nil, true, true)
			ix.Dispatch(res.ID, now)
			ix.Settle(res.ID, now)
			_ = ix.Len()
			_ = ix.Snapshot(now)
		}(i)
	}
	wg.Wait()
	// Every request had the same content, so there is exactly one branch ROOT:
	// concurrent arrivals of an outstanding tip fork, but every id is derived
	// from that one conversation.
	assert.GreaterOrEqual(t, ix.Len(), 1)
}
