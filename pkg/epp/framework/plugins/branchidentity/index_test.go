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
	"encoding/binary"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const bs = defaultBlockSizeTokens

// stream builds a synthetic pseudo-token stream of n whole blocks: every token is
// seed-derived, so two streams with the same seed share a prefix and differ only
// after their common length (an append-only conversation).
func stream(blocks int, seed uint32) []uint32 {
	tokens := make([]uint32, blocks*bs)
	for i := range tokens {
		tokens[i] = seed*uint32(bs*1000) + uint32(i)
	}
	return tokens
}

// grown returns the first n tokens of base followed by n-new distinct tokens: the
// sub-chunk counterpart of stream (an append-only conversation whose turns are
// shorter than one block).
func grown(base []uint32, n int, seed uint32) []uint32 {
	if n <= len(base) {
		return append([]uint32{}, base[:n]...)
	}
	out := append([]uint32{}, base...)
	for i := len(base); i < n; i++ {
		out = append(out, seed*uint32(bs*1000)+uint32(i))
	}
	return out
}

func newTestIndex() *branchIndex {
	return newBranchIndex(bs, "", time.Minute, 1000)
}

// bytesOf renders a pseudo-token stream as the canonical bytes it was packed
// from (4 little-endian bytes per token, no padding).
func bytesOf(tokens []uint32) []byte {
	out := make([]byte, len(tokens)*bytesPerPseudoToken)
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(out[i*bytesPerPseudoToken:], t)
	}
	return out
}

// arr builds one arrival over the given token stream.
func arr(tokens []uint32, scope string, roles []string, marker string) Arrival {
	return arrBytes(bytesOf(tokens), scope, roles, marker)
}

// arrBytes builds one arrival over a raw identity stream.
func arrBytes(stream []byte, scope string, roles []string, marker string) Arrival {
	return Arrival{
		Stream: stream,
		Chain:  ChainStream(stream, "", bs),
		Scope:  scope,
		Roles:  roles,
		Marker: marker,
	}
}

// turn is one complete sequential turn: arrival, dispatch, settle — the shape of
// a conversation that waits for each reply.
func turn(ix *branchIndex, at time.Time, a Arrival) ResolveResult {
	res := ix.Resolve(at, a, true, true, true)
	ix.Dispatch(res.ID, a.Marker, at)
	ix.Settle(res.ID, a.Marker, at)
	return res
}

// requireInvariants asserts the tip indexes hold exactly the live records, once
// each — the property the id-mint overwrite used to break.
func requireInvariants(t *testing.T, ix *branchIndex) {
	t.Helper()
	assert.Empty(t, ix.checkIndexInvariants())
}

// TestChainedBlockHashes_DropsPartialTail: only whole blocks are chained, so a
// continuation whose last block is still being written still matches the tip.
func TestChainedBlockHashes_DropsPartialTail(t *testing.T) {
	full := stream(3, 1)
	partial := append(append([]uint32{}, full...), make([]uint32, bs/2)...)
	assert.Len(t, ChainedBlockHashes(partial, "", bs), 3)
	assert.Len(t, ChainedBlockHashes(full, "", bs), 3)
	assert.Equal(t, ChainedBlockHashes(full, "", bs), ChainedBlockHashes(partial, "", bs),
		"a partial trailing block must not change the chain")
}

// TestChainedBlockHashes_Deterministic: the same stream and salt always produce
// the same chain (the id-stability requirement), and the chain is genuinely
// chained — the same block at a different position hashes differently.
func TestChainedBlockHashes_Deterministic(t *testing.T) {
	tokens := stream(4, 7)
	assert.Equal(t, ChainedBlockHashes(tokens, "salt", bs), ChainedBlockHashes(tokens, "salt", bs))
	assert.NotEqual(t, ChainedBlockHashes(tokens, "salt", bs), ChainedBlockHashes(tokens, "other", bs))

	shifted := append([]uint32{99}, tokens[:len(tokens)-1]...)
	assert.NotEqual(t, ChainedBlockHashes(tokens, "salt", bs)[0], ChainedBlockHashes(shifted, "salt", bs)[0])
}

// TestResolve_Continuation: growing the conversation continues the SAME branch,
// and an identical retry does too.
func TestResolve_Continuation(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(8, 1)

	first := turn(ix, now, arr(tokens[:3*bs], "", []string{"user"}, "r1"))
	require.Equal(t, OutcomeNew, first.Outcome)
	assert.Empty(t, ix.checkIndexInvariants())

	grown3 := turn(ix, now, arr(tokens[:5*bs], "", []string{"user", "assistant", "user"}, "r2"))
	assert.Equal(t, OutcomeMatched, grown3.Outcome)
	assert.Equal(t, first.ID, grown3.ID, "a continuation keeps the branch id")
	assert.Equal(t, 3, grown3.MatchedBlocks)
	assert.False(t, grown3.StickyBusy, "an idle tip needs no content proof")

	retry := ix.Resolve(now, arr(tokens[:5*bs], "", []string{"user", "assistant", "user"}, "r3"), true, true, true)
	assert.Equal(t, OutcomeMatched, retry.Outcome)
	assert.Equal(t, first.ID, retry.ID)
	assert.Len(t, ix.branches, 1)
	requireInvariants(t, ix)
}

// TestResolve_TailExact is the tip's tail check: two requests whose full blocks
// are identical but whose TAIL (the partial block) diverges are different
// conversations at that point, so the arriving one must not be merged into the
// tip. Without the tail check the chained block hash alone matches and the
// request silently joins someone else's queue.
func TestResolve_TailExact(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	base := stream(4, 21)

	tip := append(append([]uint32{}, base[:3*bs]...), 7, 7, 7, 7, 7)
	first := turn(ix, now, arr(tip, "", nil, "a"))
	require.Equal(t, OutcomeNew, first.Outcome)

	// Same three full blocks, different tail: NOT a continuation.
	divergent := append(append([]uint32{}, base[:3*bs]...), 9, 9, 9, 9, 9, 9, 9)
	res := ix.Resolve(now, arr(divergent, "", nil, "b"), true, true, true)
	assert.Equal(t, OutcomeNew, res.Outcome, "a divergent tail is a different branch")
	assert.NotEqual(t, first.ID, res.ID)

	// The true continuation contains the tip's whole stream, tail included.
	continued := append(append([]uint32{}, tip...), 1234, 5678)
	cont := ix.Resolve(now, arr(continued, "", nil, "c"), true, true, true)
	assert.Equal(t, OutcomeMatched, cont.Outcome)
	assert.Equal(t, first.ID, cont.ID)
	requireInvariants(t, ix)
}

// TestResolve_TailExactUnalignedBytes: the tail is hashed over BYTES. A stream's
// length is generally not a multiple of 4, so its last pseudo-token is
// zero-padded; two streams that share that region byte for byte must still match.
// Hashing the padded token would make every unaligned turn a false NEW branch —
// the case a real chat body hits on almost every turn.
func TestResolve_TailExactUnalignedBytes(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()

	// 600 + 4 bytes of message framing: 604 bytes = 1 block (512 B) + a 92-byte tail.
	base := make([]byte, 604)
	for i := range base {
		base[i] = byte(i % 251)
	}
	first := turn(ix, now, arrBytes(base, "", []string{"user"}, "a"))
	require.Equal(t, OutcomeNew, first.Outcome)

	// Append one more (odd-length) message: the old tail bytes are unchanged.
	continued := append(append([]byte{}, base...), []byte("assistantreply!")...)
	res := turn(ix, now, arrBytes(continued, "", []string{"user", "assistant", "user"}, "b"))
	assert.Equal(t, OutcomeMatched, res.Outcome)
	assert.Equal(t, first.ID, res.ID)

	// A stream that shares the 92-byte tail region only up to a byte is not a
	// continuation.
	diverging := append([]byte{}, base[:len(base)-1]...)
	diverging = append(diverging, base[len(base)-1]^0xff)
	diverging = append(diverging, []byte("more")...)
	other := ix.Resolve(now, arrBytes(diverging, "", nil, "c"), true, true, true)
	assert.Equal(t, OutcomeNew, other.Outcome)
	requireInvariants(t, ix)
}

// TestResolve_SubChunkContinuation (B3): a conversation whose turns are all
// shorter than one block has no chain at all, so it used to be unmatchable —
// every turn was a false NEW branch. Sub-chunk tips are indexed by (length,
// tail hash) and matched like any other tail.
func TestResolve_SubChunkContinuation(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	base := grown(nil, 40, 4)

	first := turn(ix, now, arr(base, "", []string{"user"}, "s1"))
	require.Equal(t, OutcomeNew, first.Outcome)
	assert.Equal(t, 0, first.Blocks, "the stream is shorter than one block")

	second := turn(ix, now, arr(grown(base, 55, 5), "", []string{"user", "assistant", "user"}, "s2"))
	assert.Equal(t, OutcomeMatched, second.Outcome, "a sub-chunk continuation must continue, not re-mint")
	assert.Equal(t, first.ID, second.ID)

	// A turn that finally crosses a block boundary continues the same branch.
	long := append(grown(base, 55, 5), stream(1, 77)...)
	third := turn(ix, now, arr(long, "", nil, "s3"))
	assert.Equal(t, OutcomeMatched, third.Outcome)
	assert.Equal(t, first.ID, third.ID)

	// ... and so does a turn that extends that one.
	fourth := turn(ix, now, arr(append(append([]uint32{}, long...), 4242), "", nil, "s4"))
	assert.Equal(t, OutcomeMatched, fourth.Outcome)
	assert.Equal(t, first.ID, fourth.ID)
	requireInvariants(t, ix)
}

// TestResolve_SubChunkFallsThroughToShorterTip: a sub-chunk tip that does NOT
// match must not block a genuinely matching shorter one — the scan walks candidate
// lengths longest-first and falls through on a mismatch.
func TestResolve_SubChunkFallsThroughToShorterTip(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()

	head := grown(nil, 20, 31)
	longer := append(append([]uint32{}, head...), grown(nil, 10, 99)...) // 30 tokens, same head

	firstLong := turn(ix, now, arr(longer, "", nil, "a"))
	// A SHORTER stream is not a continuation of a longer tip, so it is its own
	// sub-chunk branch — and the two now share a 20-token head.
	firstShort := turn(ix, now, arr(head, "", nil, "b"))
	require.NotEqual(t, firstLong.ID, firstShort.ID)

	// 26 tokens: too short to contain the 30-token tip, long enough to contain the
	// 20-token one, and it diverges right after that head.
	diverging := append(append([]uint32{}, head...), grown(nil, 6, 55)...)
	res := turn(ix, now, arr(diverging, "", nil, "c"))
	assert.Equal(t, OutcomeMatched, res.Outcome, "the shorter tip still matches")
	assert.Equal(t, firstShort.ID, res.ID)

	// A request whose head matches no sub-chunk tip is a new branch.
	stranger := turn(ix, now, arr(append(grown(nil, 20, 555), 1, 2, 3), "", nil, "d"))
	assert.Equal(t, OutcomeNew, stranger.Outcome)
	requireInvariants(t, ix)
}

// TestResolve_ForkIfBusy: extending a BUSY tip forks a new branch (a sub-agent
// inheriting the parent's context), and the fork's own continuations stay on the
// fork.
func TestResolve_ForkIfBusy(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(12, 2)

	parent := ix.Resolve(now, arr(tokens[:3*bs], "", []string{"user"}, "p"), true, true, true)
	require.True(t, ix.Dispatch(parent.ID, "p", now), "the parent turn is dispatched (busy)")

	// The child inherits the parent's context and appends its own instruction: no
	// assistant message past the tip, so nothing proves the parent completed.
	fork := ix.Resolve(now, arr(tokens[:5*bs], "", []string{"user", "user", "user", "user"}, "c"), true, true, true)
	assert.Equal(t, OutcomeForked, fork.Outcome)
	assert.NotEqual(t, parent.ID, fork.ID, "a fork gets its own queue key")
	assert.Equal(t, 3, fork.MatchedBlocks)

	// The child dispatches and completes; a request extending the FORK's tip
	// continues the fork (the longest tip prefix wins over the parent's tip).
	ix.Dispatch(fork.ID, "c", now)
	ix.Settle(fork.ID, "c", now)
	forkTurn := turn(ix, now, arr(tokens[:6*bs], "", nil, "f2"))
	assert.Equal(t, OutcomeMatched, forkTurn.Outcome)
	assert.Equal(t, fork.ID, forkTurn.ID)

	// A stream shorter than every tip matches nothing, so it is a new branch.
	shorter := ix.Resolve(now, arr(tokens[:2*bs], "", nil, "s"), true, true, true)
	assert.Equal(t, OutcomeNew, shorter.Outcome)
	assert.NotEqual(t, parent.ID, shorter.ID)
	requireInvariants(t, ix)
}

// TestResolve_SamePromptConcurrentIsASibling: two IDENTICAL prompts in flight at
// once are two parallel siblings, the one case content cannot separate — so the
// busy flag forks them. A repeat of a FINISHED turn continues (a retry).
func TestResolve_SamePromptConcurrentIsASibling(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(5, 3)[:4*bs]

	first := ix.Resolve(now, arr(tokens, "", nil, "a"), true, true, true)
	ix.Dispatch(first.ID, "a", now)

	concurrent := ix.Resolve(now, arr(tokens, "", nil, "b"), true, true, true)
	assert.Equal(t, OutcomeForked, concurrent.Outcome, "a same-prompt duplicate in flight gets its own seat")
	assert.NotEqual(t, first.ID, concurrent.ID)

	// Once both turns completed, an identical body is a retry: same branch.
	ix.Settle(first.ID, "a", now)
	ix.Settle(concurrent.ID, "b", now)
	retry := ix.Resolve(now, arr(tokens, "", nil, "c"), true, true, true)
	assert.Equal(t, OutcomeMatched, retry.Outcome)
	_, ok := ix.branches[retry.ID]
	assert.True(t, ok)
	requireInvariants(t, ix)
}

// TestResolve_RetryOfStillQueuedTurn (S2): a byte-identical duplicate of a tip
// that has NOTHING IN FLIGHT — only an arrival marker, so the request is still
// sitting in the admission queue — is a client retry, not a parallel sibling, and
// continues the branch. The error asymmetry decides it: a false fork re-fragments
// a conversation into a newcomer id, while a false continuation only shares a
// queue with itself.
func TestResolve_RetryOfStillQueuedTurn(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(4, 12)[:3*bs]

	first := ix.Resolve(now, arr(tokens, "", nil, "orig"), true, true, true)
	require.Equal(t, OutcomeNew, first.Outcome)

	retry := ix.Resolve(now, arr(tokens, "", nil, "retry"), true, true, true)
	assert.Equal(t, OutcomeMatched, retry.Outcome, "a duplicate of a still-queued turn is a retry")
	assert.Equal(t, first.ID, retry.ID)
	assert.True(t, retry.Retry)

	// With the rule disabled the same arrival forks (the conservative arm).
	off := ix.Resolve(now, arr(tokens, "", nil, "retry2"), true, true, false)
	assert.Equal(t, OutcomeForked, off.Outcome)

	// Once a copy is genuinely in flight, a duplicate is a parallel sibling.
	live := newTestIndex()
	inFlight := live.Resolve(now, arr(tokens, "", nil, "x"), true, true, true)
	live.Dispatch(inFlight.ID, "x", now)
	dup := live.Resolve(now, arr(tokens, "", nil, "y"), true, true, true)
	assert.Equal(t, OutcomeForked, dup.Outcome, "an identical prompt in flight is a sibling, not a retry")
	assert.NotEqual(t, inFlight.ID, dup.ID)

	requireInvariants(t, ix)
}

// TestResolve_StickyBusyGuard (H1): a request that extends a busy tip AND begins
// with the tip's assistant reply contains that reply, which an in-flight request
// cannot have — proven completion, so it continues the branch.
func TestResolve_StickyBusyGuard(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 4)
	tipRoles := []string{"system", "user", "tool"}

	first := ix.Resolve(now, arr(tokens[:3*bs], "", tipRoles, "a"), true, true, true)
	ix.Dispatch(first.ID, "a", now) // never settles: the busy flag is stale

	// First message past the tip is the assistant reply → proven continuation.
	withReply := ix.Resolve(now, arr(tokens[:5*bs], "",
		append(append([]string{}, tipRoles...), "assistant", "tool"), "b"), true, true, true)
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
	first := live.Resolve(now, arr(tokens[:3*bs], "", tipRoles, "a"), true, true, true)
	live.Dispatch(first.ID, "a", now)
	sibling := live.Resolve(now, arr(tokens[:5*bs], "", siblingRoles, "b"), true, true, true)
	assert.Equal(t, OutcomeForked, sibling.Outcome)
	assert.NotEqual(t, first.ID, sibling.ID)

	off := newTestIndex()
	second := off.Resolve(now, arr(tokens[:3*bs], "", tipRoles, "a"), true, false, true)
	off.Dispatch(second.ID, "a", now)
	withReply := off.Resolve(now, arr(tokens[:5*bs], "",
		append(append([]string{}, tipRoles...), "assistant", "tool"), "b"), true, false, true)
	assert.Equal(t, OutcomeForked, withReply.Outcome, "with the guard off, a busy tip always forks")
	assert.NotEqual(t, second.ID, withReply.ID)
}

// TestResolve_ForkOnBusyDisabled: with the knob off every match continues (the
// v2c failure mode, kept as a measurement arm only).
func TestResolve_ForkOnBusyDisabled(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 6)

	first := ix.Resolve(now, arr(tokens[:3*bs], "", nil, "a"), true, true, true)
	ix.Dispatch(first.ID, "a", now)

	shared := ix.Resolve(now, arr(tokens[:5*bs], "", nil, "b"), false, true, true)
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

	a := turn(ix, now, arr(tokens[:3*bs], "sess-a", nil, "a"))
	b := turn(ix, now, arr(tokens[:3*bs], "sess-b", nil, "b"))
	assert.NotEqual(t, a.ID, b.ID, "same body, different session → different branch")

	againA := turn(ix, now, arr(tokens[:4*bs], "sess-a", nil, "a2"))
	assert.Equal(t, a.ID, againA.ID, "the same session continues its own branch")

	// An unscoped request may continue a scoped tip (no session header to scope by).
	unscoped := turn(ix, now, arr(tokens[:5*bs], "", nil, "u"))
	assert.Equal(t, OutcomeMatched, unscoped.Outcome)

	// An unscoped tip is visible to a scoped request too — and two instances
	// derive the same id for the same content (ids need no registry to agree).
	ix2 := newTestIndex()
	open := turn(ix2, now, arr(tokens[:3*bs], "", nil, "a"))
	scoped := turn(ix2, now, arr(tokens[:4*bs], "sess-c", nil, "b"))
	assert.Equal(t, open.ID, scoped.ID)
}

// TestResolve_NoChainUsesStreamHash: a body shorter than one block still gets a
// deterministic id from its whole-stream hash — distinct short conversations stay
// distinct, and an identical retry lands on the same branch instead of minting a
// second one (S2, at the sub-chunk shape).
func TestResolve_NoChainUsesStreamHash(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()

	shortA := turn(ix, now, arr([]uint32{1, 2, 3, 4, 5}, "", nil, "a"))
	shortB := turn(ix, now, arr([]uint32{9, 9, 9}, "", nil, "b"))
	assert.NotEqual(t, shortA.ID, shortB.ID)

	retry := ix.Resolve(now, arr([]uint32{1, 2, 3, 4, 5}, "", nil, "c"), true, true, true)
	assert.Equal(t, shortA.ID, retry.ID)
	assert.Equal(t, OutcomeMatched, retry.Outcome)
	requireInvariants(t, ix)
}

// TestMint_NeverReusesALiveID (B2): re-sending a branch's FIRST prompt after the
// branch has advanced mints the same seed (same tip commitment, same ordinal). The
// old mint then OVERWROTE the live record with the new one, leaving the old record
// ghosted — still reachable from the tip index, no longer in the table — so it kept
// matching, advancing and (in the cap's eyes) not existing. The re-sent prompt must
// get a distinct id and leave the original branch untouched.
func TestMint_NeverReusesALiveID(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(9, 41)
	firstPrompt := tokens[:3*bs]

	a := turn(ix, now, arr(firstPrompt, "", nil, "a1"))
	require.Equal(t, OutcomeNew, a.Outcome)
	advanced := turn(ix, now, arr(tokens[:6*bs], "", nil, "a2"))
	require.Equal(t, a.ID, advanced.ID, "the branch advanced")
	before := *ix.branches[a.ID]

	resent := ix.Resolve(now, arr(firstPrompt, "", nil, "a3"), true, true, true)
	assert.NotEqual(t, a.ID, resent.ID, "a re-minted seed must not reuse a live branch's id")

	live, ok := ix.branches[a.ID]
	require.True(t, ok, "the original branch is still live")
	assert.Equal(t, before.idSeed, live.idSeed)
	assert.Equal(t, before.blocks, live.blocks, "the original tip was not reset")
	assert.Equal(t, before.tip, live.tip)
	assert.Equal(t, before.turns, live.turns, "the re-sent prompt did not advance the original branch")
	assert.Len(t, ix.branches, 2, "two distinct branches, no overwrite")
	requireInvariants(t, ix)
}

// TestMarkers_ArePerRequestID (S1): the outstanding set names exactly which
// requests are unfinished, so settling ONE request leaves its siblings' markers in
// place — the old FIFO-counter model shifted the oldest arrival whatever actually
// settled, which could free a branch that still had work queued (or hold one that
// had none).
func TestMarkers_ArePerRequestID(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(4, 51)[:3*bs]

	// Three arrivals on the same branch, none dispatched: a queued turn plus two
	// byte-identical retries of it (S2 keeps all three on the branch).
	first := ix.Resolve(now, arr(tokens, "", nil, "r1"), true, true, true)
	second := ix.Resolve(now, arr(tokens, "", nil, "r2"), true, true, true)
	third := ix.Resolve(now, arr(tokens, "", nil, "r3"), true, true, true)
	require.Equal(t, first.ID, second.ID)
	require.Equal(t, first.ID, third.ID, "retries of a still-queued turn continue the branch")

	rec := ix.branches[first.ID]
	require.Len(t, rec.pending, 3)
	require.Empty(t, rec.inflight)

	// Dispatch the middle one: only ITS marker moves.
	require.True(t, ix.Dispatch(first.ID, "r2", now))
	assert.Len(t, rec.pending, 2)
	assert.Contains(t, rec.pending, "r1")
	assert.Contains(t, rec.pending, "r3")
	assert.NotContains(t, rec.pending, "r2")
	assert.Contains(t, rec.inflight, "r2")

	// Settle it: the branch is still outstanding for r1/r3.
	require.True(t, ix.Settle(first.ID, "r2", now))
	assert.Empty(t, rec.inflight)
	assert.True(t, rec.outstanding(), "two queued requests still own the branch")

	// Settling an unknown marker removes nothing.
	require.True(t, ix.Settle(first.ID, "nope", now))
	assert.Len(t, rec.pending, 2)

	// Once everything settles the branch is free again.
	ix.Dispatch(first.ID, "r1", now)
	ix.Settle(first.ID, "r1", now)
	ix.Dispatch(first.ID, "r3", now)
	ix.Settle(first.ID, "r3", now)
	assert.False(t, rec.outstanding())
}

// TestMarkers_ExpireOnTheRequestTimeout: a marker whose request never reports an
// end of stream must not wedge the branch forever, and the horizon is the client
// request timeout (the harness's 3600s), not the flow queue's 5-minute GC — 16% of
// A0's requests waited longer than 600s for admission.
func TestMarkers_ExpireOnTheRequestTimeout(t *testing.T) {
	ix := newBranchIndex(bs, "", defaultMarkerTTLSeconds*time.Second, 1000)
	base := time.Now()
	tokens := stream(6, 61)

	res := ix.Resolve(base, arr(tokens[:3*bs], "", []string{"user"}, "r1"), true, true, true)
	rec := ix.branches[res.ID]
	ix.Dispatch(res.ID, "r1", base) // in flight, never settles
	require.True(t, rec.outstanding())

	// Ten minutes later the marker is still live (a 600s TTL would have expired it).
	at10m := base.Add(10 * time.Minute)
	ix.mu.Lock()
	rec.pruneMarkers(at10m, ix.markerTTL)
	ix.mu.Unlock()
	assert.True(t, rec.outstanding(), "a 600s TTL would free a branch whose request is still queued")

	// Past the request timeout it is pruned, and the next turn continues.
	late := base.Add(defaultMarkerTTLSeconds*time.Second + time.Minute)
	next := ix.Resolve(late, arr(tokens[:5*bs], "", []string{"user", "user", "assistant", "tool", "user"}, "r2"), true, true, true)
	assert.Equal(t, OutcomeMatched, next.Outcome)
	assert.Equal(t, res.ID, next.ID)
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

	first := turn(ix, base, arr(tokens[:3*bs], "", nil, "a"))
	// Two hours later — far past any plausible T_gc — the sweep removes nothing.
	assert.Equal(t, 0, ix.EnforceCap(base.Add(2*time.Hour)))
	again := turn(ix, base.Add(2*time.Hour), arr(tokens[:5*bs], "", nil, "b"))
	assert.Equal(t, OutcomeMatched, again.Outcome, "a dormant thread is not a new thread")
	assert.Equal(t, first.ID, again.ID)
}

// TestOutstandingBranchIsNeverEvicted: neither the cap nor the sweep removes a
// branch with anything in flight or pending (a dropped branch would abort the
// busy bookkeeping and mint a false new id on the next turn).
func TestOutstandingBranchIsNeverEvicted(t *testing.T) {
	ix := newBranchIndex(bs, "", 5*time.Minute, 1)
	base := time.Now()
	tokens := stream(9, 8)

	busy := ix.Resolve(base, arr(tokens[:3*bs], "", nil, "a"), true, true, true)
	ix.Dispatch(busy.ID, "a", base)
	queued := ix.Resolve(base, arr(tokens[:4*bs], "", nil, "b"), true, true, true) // arrival only

	assert.Equal(t, 0, ix.EnforceCap(base.Add(time.Minute)), "over the cap, but everything is outstanding")
	ix.Settle(busy.ID, "a", base)

	// Once nothing is outstanding, the cap collects, oldest first.
	assert.Equal(t, 1, ix.EnforceCap(base.Add(time.Minute)))
	assert.Equal(t, 1, ix.Len())
	assert.Len(t, ix.byTip, 1, "eviction must keep the tip index in sync")
	_ = queued
	requireInvariants(t, ix)
}

// TestCap_MaxBranches: the hard cap evicts the idlest non-outstanding
// branches and never a busy one.
func TestCap_MaxBranches(t *testing.T) {
	ix := newBranchIndex(bs, "", time.Minute, 3)
	base := time.Now()

	var ids []string
	for i := 1; i <= 5; i++ {
		at := base.Add(time.Duration(i) * time.Second)
		// A distinct stream per branch: unrelated conversations.
		res := turn(ix, at, arr(stream(2, uint32(i)), "", nil, fmt.Sprintf("r%d", i)))
		ids = append(ids, res.ID)
	}
	busy := ix.Resolve(base.Add(6*time.Second), arr(stream(2, 99), "", nil, "busy"), true, true, true)
	ix.Dispatch(busy.ID, "busy", base.Add(6*time.Second))

	ix.EnforceCap(base.Add(7 * time.Second))
	assert.LessOrEqual(t, ix.Len(), 4, "the cap must bound the table (one over, for the busy branch)")
	_, kept := ix.branches[ids[4]]
	assert.True(t, kept, "the newest branch survives")
	_, keptBusy := ix.branches[busy.ID]
	assert.True(t, keptBusy, "a busy branch is never evicted by the cap")
	_, keptOldest := ix.branches[ids[0]]
	assert.False(t, keptOldest, "the idlest branch is evicted first")
	requireInvariants(t, ix)
}

// TestCap_SubChunkBranchesAreEvictable: sub-chunk tips live in the second index,
// so the cap must unindex them there too (a stale (length, hash) entry would keep
// matching an evicted branch).
func TestCap_SubChunkBranchesAreEvictable(t *testing.T) {
	ix := newBranchIndex(bs, "", time.Minute, 1)
	base := time.Now()

	a := turn(ix, base, arr([]uint32{5, 6, 7, 8}, "", nil, "a"))
	b := turn(ix, base.Add(time.Second), arr([]uint32{11, 12, 13}, "", nil, "b"))
	require.NotEqual(t, a.ID, b.ID)
	assert.Len(t, ix.byShort, 2)

	assert.Equal(t, 1, ix.EnforceCap(base.Add(2*time.Second)))
	assert.Len(t, ix.byShort, 1)
	_, lengthStillIndexed := ix.shortLens[4]
	assert.False(t, lengthStillIndexed, "the evicted tip's length leaves the scan bound")

	// The surviving branch still matches; the evicted one is a new branch again.
	survivor := turn(ix, base.Add(3*time.Second), arr([]uint32{11, 12, 13, 14}, "", nil, "c"))
	assert.Equal(t, b.ID, survivor.ID)
	gone := turn(ix, base.Add(4*time.Second), arr([]uint32{5, 6, 7, 8, 9}, "", nil, "d"))
	assert.NotEqual(t, a.ID, gone.ID)
	requireInvariants(t, ix)
}

// TestSnapshot_Bounded: the debug dump reports the table shape and stays small.
func TestSnapshot_Bounded(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	res := ix.Resolve(now, arr(stream(4, 1)[:3*bs], "s1", nil, "a"), true, true, true)
	ix.Dispatch(res.ID, "a", now)

	snap := ix.Snapshot(now)
	require.Equal(t, 1, snap.Branches)
	assert.Equal(t, 1, snap.Busy)
	assert.Equal(t, 1, snap.InFlight)
	require.Len(t, snap.Rows, 1)
	assert.Equal(t, res.ID, snap.Rows[0].ID)
	assert.Equal(t, "s1", snap.Rows[0].Scope)
	assert.Equal(t, 3, snap.Rows[0].Blocks)

	big := newBranchIndex(bs, "", time.Minute, 100000)
	for i := 0; i < maxSnapshotRows+20; i++ {
		big.Resolve(now, arr(stream(2, uint32(i+1)), "", nil, fmt.Sprintf("r%d", i)), true, true, true)
	}
	assert.LessOrEqual(t, len(big.Snapshot(now).Rows), maxSnapshotRows)
}

// TestResolve_Concurrent: the table is safe under concurrent arrivals, and the
// tip indexes never hold anything but the live records — the byTip ⊆ branches
// invariant, which a minted-id overwrite used to break silently. Run with -race to
// make this meaningful.
func TestResolve_Concurrent(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(6, 11)
	streams := [][]uint32{
		tokens[:3*bs], tokens[:4*bs], tokens[:5*bs],
		grown(nil, 20, 3), grown(nil, 30, 3), grown(nil, 40, 4),
		stream(2, 71), stream(3, 72),
	}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s := streams[i%len(streams)]
			marker := fmt.Sprintf("req-%d", i)
			res := ix.Resolve(now, arr(s, "", nil, marker), true, true, true)
			ix.Dispatch(res.ID, marker, now)
			ix.Settle(res.ID, marker, now)
			_ = ix.Len()
			_ = ix.Snapshot(now)
		}(i)
	}
	wg.Wait()

	assert.Empty(t, ix.checkIndexInvariants(), "every indexed record must be the live record for its id")
	for _, rec := range ix.branches {
		assert.False(t, rec.outstanding(), "every request settled")
	}
}

// TestResolve_ConcurrentForkStorm drives the mint path hard: many goroutines
// resolving the SAME stream while a busy parent holds the tip, so forks with
// colliding seeds and ordinals race for ids. No two live branches may share an id,
// and no live record may be left unindexed.
func TestResolve_ConcurrentForkStorm(t *testing.T) {
	ix := newTestIndex()
	now := time.Now()
	tokens := stream(4, 91)

	parent := ix.Resolve(now, arr(tokens[2*bs:3*bs], "", nil, "parent"), true, true, true)
	ix.Dispatch(parent.ID, "parent", now) // busy forever

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Same content, same ordinal source: the id mint must bump, not reuse.
			ix.Resolve(now, arr(tokens[2*bs:4*bs], "", nil, fmt.Sprintf("f%d", i)), true, true, true)
		}(i)
	}
	wg.Wait()
	assert.Empty(t, ix.checkIndexInvariants())
	assert.GreaterOrEqual(t, ix.Len(), 2)
}
