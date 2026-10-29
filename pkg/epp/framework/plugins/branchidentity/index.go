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

// Package branchidentity derives the flow-control queue key from the request
// body: a request-control plugin that flattens the conversation into an
// identity byte stream, chains it into block hashes, and matches it against the
// live branch table before admission control, so the fairness id
// (FairnessID → the flow queue key) is a DERIVED, stable branch id rather than
// a client-supplied thread id.
//
// The identity model (BRANCH-IDENTITY-DESIGN.md):
//
//   - Queue key = BRANCH. Sibling requests of one agent session (sub-agents
//     inheriting the parent's context) are separate branches, so they dispatch
//     in parallel instead of serialising behind each other in one queue.
//   - SESSION is an attribute, never a queue key. The plugin reads the client's
//     session header (when present) and publishes it under its own key for the
//     stage-2 session fairness layer; it does not overwrite the branch id.
//   - Identity is FLATTENING, not tokens: chat templates can rewrite history in
//     token space while the conversation continues (thinking models drop past
//     reasoning, some stamp dates), and clients drop earlier turns' thinking and
//     swap tool sets mid-thread. The tokenizer's IDENTITY stream is the stable
//     view (tokenizer.FlattenIdentityBytes).
//   - A tip is TAIL-EXACT: (blocks, tip hash, tail length, tail hash) describes
//     the tip request's ENTIRE stream, not just its last full block. A request
//     continues the branch only if the tip's whole stream is a byte prefix of
//     its own — the full blocks are implied by the chained hash, the tail is
//     verified explicitly. Tips shorter than one block (a tiny first turn) are
//     indexed by (tail length, tail hash), so they are matchable too: without
//     that, every sub-chunk conversation read as a false NEW branch.
//   - Continuation: R continues B iff B's tip is a prefix of R's stream AND B
//     has nothing outstanding; otherwise R FORKS a new branch. Two overrides:
//     the sticky-busy guard (H1: R extends B's tip and the first message past
//     the tip is an assistant message, so R contains B's reply — an in-flight
//     request cannot have it; proven completion), and the retry rule (S2: R is
//     byte-IDENTICAL to B's tip and B has nothing in flight but its arrival
//     marker — a duplicate of a still-queued request is a client retry; the
//     error asymmetry favours continuation).
//   - BUSY IS MARKED AT ARRIVAL (RequestHeader), not at dispatch: the queued
//     interval is most of a busy period under load (v2c's median wait was 22s),
//     and a branch that looks free while its turn sits in the admission queue
//     would hand a sibling the parent's queue — the head-of-line blocking this
//     design exists to remove. The flag is cleared at end of stream, or expires
//     with its marker when a request is rejected before dispatch.
//   - Markers are per REQUEST ID (a set of pending and a set of in-flight
//     request ids per branch), never FIFO counters: a branch's outstanding count
//     must be exactly "which requests of mine have not finished", so a
//     cancellation, a rejection and a retry each remove or add exactly one
//     entry. The marker TTL is the harness request timeout (3600s), not the
//     flow queue's 5-minute idle GC — those are different clocks, and 16% of A0's
//     requests waited longer than 600s for admission.
//   - The TIP moves at arrival too, so a fork cannot hide: a sibling of the old
//     prefix extends an interior node of the moved tip and gets its own branch
//     structurally, and the busy flag only has to guard the one case content
//     cannot separate — two IDENTICAL prompts in flight at once, which are two
//     parallel siblings and therefore two branches.
//   - Retention is content-addressed, not time-based: branches are never GC'd by
//     silence (stage 0 measured a p99.9 inter-turn idle gap of 3,036s; any T_gc
//     would false-new-branch a dormant thread when it returns). Only a count cap
//     evicts, idlest first, and never a branch with anything outstanding.
//
// Error asymmetry (PREREG-branch-identity.md): a missed continuation (a false
// NEW branch) is the unsafe direction — it re-fragments a conversation into
// fresh fairness ids and re-introduces newcomer churn — while a false merge
// only shares a queue between two continuations of the same conversation. Every
// rule below resolves in favour of continuation. Stage 0 confirmed the asymmetry
// is real and that the H1 guard, not a busy grace window, is what rescues it: 42
// of 48 fork candidates were continuations the reply guard recovered.
package branchidentity

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Hash is one chained block key.
type Hash uint64

const (
	// bytesPerPseudoToken mirrors the tokenizer's estimate backend: canonical
	// bytes are read 4 at a time as pseudo-token IDs, so a block of N tokens
	// covers 4N canonical bytes.
	bytesPerPseudoToken = 4
	// idHexLen keeps branch ids short (survey §3.4): a branch id can become a
	// metric label value, and fairness-id label cardinality is capped
	// (--fairness-id-metric-label-limit).
	idHexLen = 10
	// maxPendingPerBranch bounds one branch's set of arrival markers that have
	// not dispatched yet. A leak (a request admitted then rejected before
	// PreRequest, a client that vanishes) can then never grow without bound or
	// wedge the branch.
	maxPendingPerBranch = 64
	// maxSnapshotRows bounds the StateDumper table.
	maxSnapshotRows = 200
	// maxMintAttempts bounds the ordinal bumps used to avoid reusing a live
	// branch id; past it a process-unique nonce mint is used instead (see
	// mintLocked).
	maxMintAttempts = 1 << 12
	// tailSeed starts the chained tail hash.
	tailSeed = 0xcbf29ce484222325
	// tailPrime is the tail chain's multiplier (FNV-1a's 64-bit prime).
	tailPrime = 1099511628211
)

// mintFallbackSeq makes a minted id unique within this process when the ordinal
// space is exhausted (see mintLocked).
var mintFallbackSeq atomic.Uint64

// Outcome labels how Resolve identified a request.
type Outcome string

const (
	// OutcomeMatched: the request continued an existing branch (same id as the
	// branch's previous turn).
	OutcomeMatched Outcome = "matched"
	// OutcomeForked: the request extended a BUSY branch's tip, so it started a
	// new branch (a parallel sibling — a sub-agent inheriting the parent's
	// context).
	OutcomeForked Outcome = "forked"
	// OutcomeNew: nothing in the table matched (or the body has no chain), so
	// the request started an unrelated branch.
	OutcomeNew Outcome = "new"
)

// ChainedBlockHashes computes the chained block keys of a pseudo-token stream:
// blockSize-token blocks, each key chained from the previous one, the salt
// folded into the chain's seed. Lifted from groupsched's Scheduler.HashTokens
// (same hashing, same tail rule) so there is one chaining implementation for
// both the sizing path and the identity path.
//
// The trailing PARTIAL block is not in the chain (n = len(tokens)/blockSize): its
// content changes the moment the conversation grows, so counting it would make
// every continuation look divergent. It is not ignored either — the branch table
// keeps its hash as the tip's TAIL, and a continuation must contain it byte for
// byte (see branchRecord.tipPrefixOf).
func ChainedBlockHashes(tokens []uint32, salt string, blockSize int) []Hash {
	if blockSize <= 0 || len(tokens) == 0 {
		return nil
	}
	n := len(tokens) / blockSize
	out := make([]Hash, 0, n)
	var parent uint64
	s := sha256.Sum256([]byte(salt))
	saltSeed := binary.LittleEndian.Uint64(s[:8])
	for i := 0; i < n; i++ {
		h := sha256.New()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], saltSeed^parent)
		h.Write(b[:])
		for j := i * blockSize; j < (i+1)*blockSize; j++ {
			binary.LittleEndian.PutUint32(b[:4], tokens[j])
			h.Write(b[:4])
		}
		parent = binary.LittleEndian.Uint64(h.Sum(nil)[:8])
		out = append(out, Hash(parent))
	}
	return out
}

// tailChain hashes a run of canonical BYTES as a chain: h_0 = tailSeed,
// h_i = mix(h_{i-1}, byte_i).
//
// Bytes, not pseudo-tokens: a stream's length is generally not a multiple of 4,
// so its last token is zero-padded, and two streams that share that region in
// bytes would hash differently if the padding were part of the digest. The tail is
// the one hash that has to be exact at BYTE granularity — a block boundary is
// always 4-byte aligned, so the chained block hashes are unaffected.
func tailChain(b []byte) Hash {
	h := uint64(tailSeed)
	for _, c := range b {
		h = (h ^ uint64(c)) * tailPrime
	}
	return Hash(h)
}

// packPseudoTokens packs canonical bytes into little-endian uint32 pseudo-token
// IDs (zero-padded tail). Mirrors the tokenizer's packBytes so the identity path
// and the sizing path chunk the same body at the same boundaries.
//
// The padding is internal: the caller's stream is never modified, because the
// tip's TAIL is hashed over the raw bytes (a padded last token would make two
// streams that share those bytes hash differently).
func packPseudoTokens(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	out := make([]uint32, (len(raw)+bytesPerPseudoToken-1)/bytesPerPseudoToken)
	for i := range out {
		var w [bytesPerPseudoToken]byte
		end := (i + 1) * bytesPerPseudoToken
		if end > len(raw) {
			end = len(raw)
		}
		copy(w[:], raw[i*bytesPerPseudoToken:end])
		out[i] = binary.LittleEndian.Uint32(w[:])
	}
	return out
}

// shortKey indexes a tip that is shorter than one block: its whole stream is the
// tail, so (length, chained hash) identifies it. Such tips have no block hash to
// look up, which is exactly why the block-keyed index cannot find them.
type shortKey struct {
	tailLen  int
	tailHash Hash
}

// branchRecord is one live branch: its tip and its liveness.
//
// The tip describes the ENTIRE stream of the branch's latest request:
//
//	blocks   — number of full blocks in its chain (0 for a sub-chunk request)
//	tip      — the chained hash of the last full block (a chained hash commits to
//	           every block before it, so this plus blocks identifies that prefix)
//	tailLen  — pseudo-tokens past the last full block
//	tailHash — tailChain over exactly those tokens (the whole stream when
//	           blocks == 0)
//
// A request continues this branch only if its own stream CONTAINS all of that as
// a byte prefix: the chained block hash proves the full blocks, tailHash proves
// the tail. Without the tail check a request that shares only the full blocks
// (same first 256 bytes, different continuation) would be merged into the
// branch; without the blocks==0 index a sub-chunk conversation could never be
// continued at all.
type branchRecord struct {
	id string
	// idSeed is the hash preimage the id was minted from (tip commitment, fork
	// ordinal, scope): it identifies the branch's creation, so a genuine id
	// collision is distinguishable from the same branch being minted again.
	idSeed Hash
	scope  string // client session id at creation, "" when the client sent none

	blocks   int
	tip      Hash
	tailLen  int // bytes past the last full block
	tailHash Hash

	// msgCount is the message count of the tip request. The sticky-busy guard
	// needs it: the message at that index of a later request is the first one
	// past the tip.
	msgCount int

	// pending holds the arrival markers of this branch's requests that have not
	// dispatched yet (they are in admission control); inflight those that have.
	// Both are keyed by request id: a branch's outstanding set must name exactly
	// which of its requests are unfinished, so a cancellation, a rejection or a
	// retry adds or removes exactly one entry (a FIFO counter cannot say that).
	pending  map[string]time.Time
	inflight map[string]time.Time

	lastSeen time.Time
	created  time.Time
	turns    int
	forks    int
}

// idle reports the branch's silence (no arrival, dispatch or settle).
func (r *branchRecord) idle(now time.Time) time.Duration { return now.Sub(r.lastSeen) }

// streamLen is the tip request's identity stream length in bytes.
func (r *branchRecord) streamLen(blockBytes int) int { return r.blocks*blockBytes + r.tailLen }

// markPending records one arrival marker, keeping the branch alive.
func (r *branchRecord) markPending(marker string, now time.Time) {
	if r.pending == nil {
		r.pending = make(map[string]time.Time, 2)
	}
	if _, dup := r.pending[marker]; !dup && len(r.pending) >= maxPendingPerBranch {
		// Bound the set: a runaway client cannot grow one branch's marker set
		// without limit. The branch stays outstanding through the existing
		// markers, so a lost marker errs toward continuation.
		return
	}
	r.pending[marker] = now
}

// markDispatched moves a marker from pending to in flight.
func (r *branchRecord) markDispatched(marker string, now time.Time) {
	if r.inflight == nil {
		r.inflight = make(map[string]time.Time, 2)
	}
	delete(r.pending, marker)
	r.inflight[marker] = now
}

// markSettled removes a marker from in flight (response complete or aborted).
func (r *branchRecord) markSettled(marker string) {
	delete(r.inflight, marker)
}

// pruneMarkers drops markers older than ttl: a request that never reached
// PreRequest (denied in admission control, cancelled) or never reported an end
// of stream must not keep its branch outstanding forever. Pruning errs toward
// continuation, the safe direction.
func (r *branchRecord) pruneMarkers(now time.Time, ttl time.Duration) {
	if ttl <= 0 {
		return
	}
	for _, set := range []map[string]time.Time{r.pending, r.inflight} {
		for marker, at := range set {
			if now.Sub(at) > ttl {
				delete(set, marker)
			}
		}
	}
}

// outstanding reports whether the branch has anything in flight or pending — the
// continuation rule's "busy" predicate.
func (r *branchRecord) outstanding() bool {
	return len(r.inflight) > 0 || len(r.pending) > 0
}

// tipPrefixOf reports whether this branch's whole tip stream is a byte prefix of
// stream, i.e. whether the request that produced stream can be a continuation of
// this branch. The full blocks are implied by the chained hash the caller matched
// on; the tail is verified here, byte for byte.
func (r *branchRecord) tipPrefixOf(stream []byte, blockBytes int) bool {
	start := r.blocks * blockBytes
	if len(stream) < start+r.tailLen {
		return false
	}
	if r.tailLen == 0 {
		return true
	}
	return tailChain(stream[start:start+r.tailLen]) == r.tailHash
}

// isExact reports whether stream is byte-identical to the branch's tip stream.
func (r *branchRecord) isExact(stream []byte, blockBytes int) bool {
	return len(stream) == r.streamLen(blockBytes) && r.tipPrefixOf(stream, blockBytes)
}

// Arrival is one request's identity inputs.
type Arrival struct {
	// Stream is the request's identity stream (the bytes
	// tokenizer.FlattenIdentityBytes produced).
	Stream []byte
	// Chain is ChainedBlockHashes(packPseudoTokens(Stream), salt, blockSize),
	// computed by the caller so the hashing stays off the index lock.
	Chain []Hash
	// Scope is the client session id, "" when the client sent none.
	Scope string
	// Roles is the request's message roles in wire order (nil for body shapes
	// without messages — the sticky-busy guard then cannot fire).
	Roles []string
	// Marker is the request's id: the handle that later removes this arrival
	// from the branch's outstanding set.
	Marker string
}

// ResolveResult is the outcome of one arrival.
type ResolveResult struct {
	// ID is the branch id (the value the director uses as the fairness id).
	ID string
	// Outcome is matched / forked / new.
	Outcome Outcome
	// Blocks is the request's own chain length (full blocks).
	Blocks int
	// MatchedBlocks is the matched tip's length (0 for a new branch); for a
	// fork it is the parent tip's length.
	MatchedBlocks int
	// StickyBusy is set when the H1 guard overrode the busy flag.
	StickyBusy bool
	// Retry is set when the S2 rule treated a byte-identical duplicate of a
	// still-queued tip as a continuation (a client retry, not a sibling).
	Retry bool
	// Elapsed is the time the resolve spent inside the table lock.
	Elapsed time.Duration
}

// branchIndex is the branch table: branch id → record, plus the tip lookup
// (last block hash → candidate branches, and (length, hash) → sub-chunk tips).
// Safe for concurrent use.
type branchIndex struct {
	mu sync.Mutex

	blockSize   int
	salt        string
	markerTTL   time.Duration
	maxBranches int

	branches map[string]*branchRecord
	// byTip indexes tips with at least one full block by their last block hash.
	byTip map[Hash][]*branchRecord
	// byShort indexes tips shorter than one block by (length, chained hash).
	byShort map[shortKey][]*branchRecord
	// shortLens is the set of tail lengths live in byShort (the scan bound).
	shortLens map[int]struct{}
}

func newBranchIndex(blockSize int, salt string, markerTTL time.Duration, maxBranches int) *branchIndex {
	if blockSize <= 0 {
		blockSize = defaultBlockSizeTokens
	}
	return &branchIndex{
		blockSize:   blockSize,
		salt:        salt,
		markerTTL:   markerTTL,
		maxBranches: maxBranches,
		branches:    make(map[string]*branchRecord),
		byTip:       make(map[Hash][]*branchRecord),
		byShort:     make(map[shortKey][]*branchRecord),
		shortLens:   make(map[int]struct{}),
	}
}

// blockBytes is the byte length of one identity block.
func (ix *branchIndex) blockBytes() int { return ix.blockSize * bytesPerPseudoToken }

// Chain hashes an identity byte stream into the request's block chain.
//
// The block count is floor(len(stream) / blockBytes): a block counts only when the
// stream COMPLETES it. Reading the count off the packed tokens instead would count
// a partially-filled block (the last token is zero-padded), so two streams whose
// tails share their bytes would hash a different number of blocks — the exact
// false-new failure the tail check exists to prevent.
func (ix *branchIndex) Chain(stream []byte) []Hash {
	return ChainStream(stream, ix.salt, ix.blockSize)
}

// ChainStream is Chain without an index: the pure function the tests and the
// replay harness both need.
func ChainStream(stream []byte, salt string, blockSize int) []Hash {
	if blockSize <= 0 {
		blockSize = defaultBlockSizeTokens
	}
	blockBytes := blockSize * bytesPerPseudoToken
	blocks := len(stream) / blockBytes
	if blocks == 0 {
		return nil
	}
	return ChainedBlockHashes(packPseudoTokens(stream[:blocks*blockBytes]), salt, blockSize)
}

// Resolve identifies the branch for an arriving request and records the
// arrival.
//
// The decision, in order:
//  1. longest branch tip that is a TAIL-EXACT prefix of the request's stream, in
//     the same session scope (a tip from another session never matches; an
//     unscoped tip matches any session).
//  2. tip not outstanding → continue (a repeat of a finished turn).
//  3. sticky-busy guard (when enabled): the first message past the tip is an
//     assistant message → the request contains the tip's reply → continue.
//  4. retry rule (when enabled): the request is byte-identical to the tip and the
//     tip has nothing IN FLIGHT (only arrival markers) → continue.
//  5. tip outstanding and unproven (unless forkOnBusy is off) → fork: an
//     identical prompt in flight is a parallel sibling, an extension is context
//     inheritance. Both need their own queue.
//  6. no matching tip → new branch.
func (ix *branchIndex) Resolve(now time.Time, a Arrival, forkOnBusy, stickyGuard, retryRule bool) ResolveResult {
	// Chaining is the expensive part, so a caller may pass the chain precomputed;
	// deriving it here only costs the caller who forgot (and stays correct).
	chain := a.Chain
	if chain == nil && len(a.Stream) > 0 {
		chain = ix.Chain(a.Stream)
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()
	start := time.Now()

	rec, matched, exact := ix.matchLocked(now, a.Stream, chain, a.Scope)
	if rec == nil {
		fresh := ix.mintLocked(now, nil, a, chain)
		return ResolveResult{ID: fresh.id, Outcome: OutcomeNew, Blocks: len(chain), Elapsed: time.Since(start)}
	}

	// Continuation needs a FREE tip: a tip with a request in flight or pending
	// belongs to that request (an identical prompt arriving meanwhile is a
	// parallel sibling, not a continuation), unless content proves completion
	// (H1) or a still-queued duplicate is a retry (S2), or the fork rule is
	// configured off.
	cont := !rec.outstanding()
	sticky, retry := false, false
	if !cont {
		switch {
		case stickyGuard && startsWithReply(rec.msgCount, a.Roles):
			sticky, cont = true, true
		case retryRule && exact && len(rec.inflight) == 0:
			// Byte-identical to the tip, and nothing of the branch has actually
			// started: this is a client retry of a request still sitting in the
			// admission queue, not a parallel sibling. (Two identical prompts
			// genuinely in flight both reach PreRequest and fork there.)
			retry, cont = true, true
		default:
			cont = !forkOnBusy
		}
	}

	if cont {
		ix.advanceLocked(now, rec, a, chain)
		return ResolveResult{
			ID: rec.id, Outcome: OutcomeMatched, Blocks: len(chain),
			MatchedBlocks: matched, StickyBusy: sticky, Retry: retry, Elapsed: time.Since(start),
		}
	}

	fork := ix.mintLocked(now, rec, a, chain)
	rec.forks++
	return ResolveResult{
		ID: fork.id, Outcome: OutcomeForked, Blocks: len(chain), MatchedBlocks: matched,
		Elapsed: time.Since(start),
	}
}

// matchLocked returns the branch whose tip stream is the LONGEST tail-exact
// prefix of the request's stream, plus that prefix's length in full blocks and
// whether the match is byte-identical to the whole request. Scoped tips only
// match their own session; unscoped tips match any (the floor the design accepts
// when the client sends no session header).
//
// Candidates are tried longest first: block-indexed tips descending from the
// request's own chain length, then sub-chunk tips descending by length — and a
// candidate whose tail does not match falls through to the next shorter tip,
// which is how a request that shares only a prefix of a tip's tail still finds
// its (shorter) true predecessor.
func (ix *branchIndex) matchLocked(now time.Time, stream []byte, chain []Hash, scope string) (*branchRecord, int, bool) {
	for i := len(chain) - 1; i >= 0; i-- {
		var best *branchRecord
		bestOutstanding := true
		for _, cand := range ix.byTip[chain[i]] {
			if cand.blocks != i+1 {
				// Same last hash at a different length can only be a hash
				// collision; it is not a prefix relation, so skip it.
				continue
			}
			if !sameScope(cand.scope, scope) || !cand.tipPrefixOf(stream, ix.blockBytes()) {
				continue
			}
			cand.pruneMarkers(now, ix.markerTTL)
			out := cand.outstanding()
			switch {
			case best == nil:
				best, bestOutstanding = cand, out
			case bestOutstanding && !out:
				// A free tip is the better continuation target than a busy one
				// with identical content (an exact duplicate of the tip).
				best, bestOutstanding = cand, out
			case bestOutstanding == out && cand.lastSeen.After(best.lastSeen):
				best = cand
			}
		}
		if best != nil {
			return best, i + 1, best.isExact(stream, ix.blockBytes())
		}
	}

	if rec := ix.matchShortLocked(now, stream, scope); rec != nil {
		return rec, 0, rec.isExact(stream, ix.blockBytes())
	}
	return nil, 0, false
}

// matchShortLocked matches tips shorter than one block (a stream with no full
// block has no hash to look up in byTip, so it is indexed by length + hash
// instead). Every live sub-chunk length is tried, longest first: each needs one
// hash of the request's own prefix of that many bytes.
func (ix *branchIndex) matchShortLocked(now time.Time, stream []byte, scope string) *branchRecord {
	if len(ix.byShort) == 0 || len(stream) == 0 {
		return nil
	}
	maxLen := ix.blockBytes() - 1
	if maxLen > len(stream) {
		maxLen = len(stream)
	}
	if maxLen <= 0 {
		return nil
	}
	for _, length := range ix.shortLensDescending(maxLen) {
		hash := tailChain(stream[:length])
		var best *branchRecord
		bestOutstanding := true
		for _, cand := range ix.byShort[shortKey{tailLen: length, tailHash: hash}] {
			if cand.tailLen != length || !sameScope(cand.scope, scope) || !cand.tipPrefixOf(stream, ix.blockBytes()) {
				continue
			}
			cand.pruneMarkers(now, ix.markerTTL)
			out := cand.outstanding()
			switch {
			case best == nil:
				best, bestOutstanding = cand, out
			case bestOutstanding && !out:
				best, bestOutstanding = cand, out
			case bestOutstanding == out && cand.lastSeen.After(best.lastSeen):
				best = cand
			}
		}
		if best != nil {
			return best
		}
	}
	return nil
}

// shortLensDescending returns the live sub-chunk tail lengths up to maxLen,
// longest first (the longest matching tip wins).
func (ix *branchIndex) shortLensDescending(maxLen int) []int {
	if len(ix.shortLens) == 0 {
		return nil
	}
	lens := make([]int, 0, len(ix.shortLens))
	for length := range ix.shortLens {
		if length <= maxLen {
			lens = append(lens, length)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(lens)))
	return lens
}

// advanceLocked moves the branch's tip to the arriving request's stream and
// records the arrival. The tip moves at ARRIVAL (design §5): the request is the
// branch's newest context whether or not it has dispatched yet, and a later
// request extending it must see it.
func (ix *branchIndex) advanceLocked(now time.Time, rec *branchRecord, a Arrival, chain []Hash) {
	ix.setTipLocked(rec, a.Stream, chain)
	rec.msgCount = len(a.Roles)
	rec.lastSeen = now
	rec.turns++
	rec.markPending(a.Marker, now)
}

// setTipLocked re-points the record at a new tip stream, keeping both tip
// indexes in sync.
func (ix *branchIndex) setTipLocked(rec *branchRecord, stream []byte, chain []Hash) {
	blocks := len(chain)
	tip := Hash(0)
	if blocks > 0 {
		tip = chain[blocks-1]
	}
	start := blocks * ix.blockBytes()
	if start > len(stream) {
		start = len(stream)
	}
	tailLen := len(stream) - start
	tailHash := tailChain(stream[start:])
	if rec.blocks == blocks && rec.tip == tip && rec.tailLen == tailLen && rec.tailHash == tailHash {
		return
	}
	ix.unindexLocked(rec)
	rec.blocks, rec.tip, rec.tailLen, rec.tailHash = blocks, tip, tailLen, tailHash
	ix.indexLocked(rec)
}

// indexLocked inserts the record into the tip index that its shape belongs to.
func (ix *branchIndex) indexLocked(rec *branchRecord) {
	if rec.blocks > 0 {
		ix.byTip[rec.tip] = append(ix.byTip[rec.tip], rec)
		return
	}
	key := shortKey{tailLen: rec.tailLen, tailHash: rec.tailHash}
	ix.byShort[key] = append(ix.byShort[key], rec)
	ix.shortLens[rec.tailLen] = struct{}{}
}

// unindexLocked removes the record from the tip index (GC, re-tip).
func (ix *branchIndex) unindexLocked(rec *branchRecord) {
	if rec.blocks > 0 {
		cands := ix.byTip[rec.tip]
		for i, c := range cands {
			if c == rec {
				cands = append(cands[:i], cands[i+1:]...)
				break
			}
		}
		if len(cands) == 0 {
			delete(ix.byTip, rec.tip)
			return
		}
		ix.byTip[rec.tip] = cands
		return
	}
	if rec.tailLen == 0 && rec.tailHash == 0 {
		return // never indexed (a record that has not been given a stream yet)
	}
	key := shortKey{tailLen: rec.tailLen, tailHash: rec.tailHash}
	cands := ix.byShort[key]
	for i, c := range cands {
		if c == rec {
			cands = append(cands[:i], cands[i+1:]...)
			break
		}
	}
	if len(cands) == 0 {
		delete(ix.byShort, key)
		ix.refreshShortLens(key.tailLen)
		return
	}
	ix.byShort[key] = cands
}

// refreshShortLens drops a tail length from the scan bound once no sub-chunk tip
// uses it.
func (ix *branchIndex) refreshShortLens(length int) {
	for key := range ix.byShort {
		if key.tailLen == length {
			return
		}
	}
	delete(ix.shortLens, length)
}

// mintLocked creates a branch. Its id is derived from the creating stream's tip
// commitment (plus a fork ordinal, so a fork of the same content gets its own
// id): the same conversation yields the same id on any instance, without a
// registry to consult. The id does not change as the branch continues —
// continuations keep the record's id, only the tip moves.
//
// An id the table already holds belongs to a LIVE branch, which is never
// overwritten: overwriting it used to leave the old record GHOSTED — still in the
// tip index, no longer in the table — so it kept matching and advancing requests
// (corrupting the branch counters) and was invisible to the cap (never evicted,
// tip index growing without bound). What looks like the same branch being minted
// twice (a client re-sending a first turn after its branch advanced, a
// look-alike fork with the same ordinal) therefore bumps the ordinal until the id
// is free, and gets a distinct branch. Stage-0c/1 telemetry did not depend on
// cross-arrival id reuse: the id is the queue key, and two different moments in a
// conversation are two different queue states.
func (ix *branchIndex) mintLocked(now time.Time, parent *branchRecord, a Arrival, chain []Hash) *branchRecord {
	tip := Hash(0)
	if len(chain) > 0 {
		tip = chain[len(chain)-1]
	} else {
		// A stream shorter than one block has no chain: its whole-stream tail
		// hash is its commitment.
		tip = tailChain(a.Stream)
	}
	ordinal := 0
	if parent != nil {
		ordinal = parent.forks + 1
	}
	seed := mintSeed(tip, ordinal, a.Scope)
	id := branchID(seed)
	for attempt := 0; ; attempt++ {
		if _, clash := ix.branches[id]; !clash {
			break
		}
		if attempt < maxMintAttempts {
			ordinal++
			seed = mintSeed(tip, ordinal, a.Scope)
		} else {
			// Every ordinal in the window is taken by a live id (astronomically
			// unlikely): salt with a process-unique nonce rather than reuse.
			seed = mintSeed(tip, ordinal, a.Scope) ^ Hash(mintFallbackSeq.Add(1))
		}
		id = branchID(seed)
	}

	rec := &branchRecord{
		id:       id,
		idSeed:   seed,
		scope:    a.Scope,
		msgCount: len(a.Roles),
		created:  now,
		lastSeen: now,
		turns:    1,
	}
	if parent != nil && rec.scope == "" {
		// A fork inherits its parent's scope when the request carried none:
		// siblings belong to the same session.
		rec.scope = parent.scope
	}
	ix.branches[id] = rec
	ix.setTipLocked(rec, a.Stream, chain)
	rec.markPending(a.Marker, now)
	return rec
}

// Dispatch marks one request as dispatched (in flight) for the branch, moving its
// arrival marker out of the pending set. An unknown id is ignored (the branch may
// have been evicted while the request was queued).
func (ix *branchIndex) Dispatch(id, marker string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.branches[id]
	if !ok || rec == nil {
		return false
	}
	rec.pruneMarkers(now, ix.markerTTL)
	rec.markDispatched(marker, now)
	rec.lastSeen = now
	return true
}

// Settle ends one dispatched request on the branch (response complete or
// aborted). The tip stays where the request left it; only liveness changes.
func (ix *branchIndex) Settle(id, marker string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.branches[id]
	if !ok || rec == nil {
		return false
	}
	rec.markSettled(marker)
	rec.lastSeen = now
	return true
}

// EnforceCap bounds the table when it grows past maxBranches, evicting the
// IDLEST non-outstanding branches first.
//
// There is deliberately NO time-based (silence) GC. Stage 0 measured a p99.9
// inter-turn idle gap of 3,036 seconds on the campaign corpus: any plausible T_gc
// would drop dormant threads that then return, and a premature drop is a false
// NEW branch — the unsafe direction (PREREG's hard <1% bar) — because a returning
// thread would be minted a fresh fairness id and re-enter as a newcomer. Branch
// records are content-addressed and tiny, so they are retained; the flow-control
// registry already GCs the idle QUEUES. The count cap is the only eviction path,
// and it never removes a branch with anything in flight or pending.
func (ix *branchIndex) EnforceCap(now time.Time) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	removed := 0
	for _, rec := range ix.branches {
		rec.pruneMarkers(now, ix.markerTTL)
	}
	if ix.maxBranches <= 0 || len(ix.branches) <= ix.maxBranches {
		return 0
	}
	victims := make([]*branchRecord, 0, len(ix.branches)-ix.maxBranches)
	for _, rec := range ix.branches {
		if !rec.outstanding() {
			victims = append(victims, rec)
		}
	}
	sort.Slice(victims, func(i, j int) bool { return victims[i].lastSeen.Before(victims[j].lastSeen) })
	for i := 0; i < len(victims) && len(ix.branches) > ix.maxBranches; i++ {
		ix.unindexLocked(victims[i])
		delete(ix.branches, victims[i].id)
		removed++
	}
	return removed
}

// Len is the live branch count (the branch_records gauge).
func (ix *branchIndex) Len() int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return len(ix.branches)
}

// checkIndexInvariants reports every record reachable from a tip index that is
// not the live record for its id. A violation means a record was overwritten
// (ghosted) or unindexed inconsistently; the concurrency test asserts it is
// empty.
func (ix *branchIndex) checkIndexInvariants() []string {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	var problems []string
	seen := make(map[*branchRecord]int, len(ix.branches))
	for hash, cands := range ix.byTip {
		for _, cand := range cands {
			seen[cand]++
			if cand.blocks == 0 || ix.branches[cand.id] != cand {
				problems = append(problems, "byTip record not live: "+cand.id)
			}
			if cand.blocks > 0 && cand.tip != hash {
				problems = append(problems, "byTip record under the wrong hash: "+cand.id)
			}
		}
	}
	for key, cands := range ix.byShort {
		for _, cand := range cands {
			seen[cand]++
			if cand.blocks != 0 || ix.branches[cand.id] != cand {
				problems = append(problems, "byShort record not live: "+cand.id)
			}
			if cand.tailLen != key.tailLen || cand.tailHash != key.tailHash {
				problems = append(problems, "byShort record under the wrong key: "+cand.id)
			}
		}
	}
	for _, rec := range ix.branches {
		if seen[rec] != 1 {
			problems = append(problems, "live record indexed "+itoa(seen[rec])+" times: "+rec.id)
		}
	}
	return problems
}

// itoa is a tiny helper so the invariant messages do not pull in strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// branchTable is the StateDumper payload.
type branchTable struct {
	Branches  int         `json:"branches"`
	Busy      int         `json:"busy"`
	Pending   int         `json:"pending"`
	InFlight  int         `json:"inFlight"`
	BlockSize int         `json:"blockSizeTokens"`
	Rows      []branchRow `json:"rows"`
}

type branchRow struct {
	ID          string  `json:"id"`
	Scope       string  `json:"scope,omitempty"`
	Blocks      int     `json:"blocks"`
	TailLen     int     `json:"tailLen"`
	MsgCount    int     `json:"msgCount"`
	InFlight    int     `json:"inflight"`
	Pending     int     `json:"pending"`
	Forks       int     `json:"forks"`
	Turns       int     `json:"turns"`
	IdleSeconds float64 `json:"idleSeconds"`
	AgeSeconds  float64 `json:"ageSeconds"`
}

// Snapshot returns the branch table (bounded, newest first) for DumpState.
func (ix *branchIndex) Snapshot(now time.Time) branchTable {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	snap := branchTable{Branches: len(ix.branches), BlockSize: ix.blockSize}
	rows := make([]branchRow, 0, len(ix.branches))
	for _, rec := range ix.branches {
		rec.pruneMarkers(now, ix.markerTTL)
		if rec.outstanding() {
			snap.Busy++
		}
		snap.InFlight += len(rec.inflight)
		snap.Pending += len(rec.pending)
		rows = append(rows, branchRow{
			ID:          rec.id,
			Scope:       rec.scope,
			Blocks:      rec.blocks,
			TailLen:     rec.tailLen,
			MsgCount:    rec.msgCount,
			InFlight:    len(rec.inflight),
			Pending:     len(rec.pending),
			Forks:       rec.forks,
			Turns:       rec.turns,
			IdleSeconds: rec.idle(now).Seconds(),
			AgeSeconds:  now.Sub(rec.created).Seconds(),
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].IdleSeconds != rows[j].IdleSeconds {
			return rows[i].IdleSeconds < rows[j].IdleSeconds
		}
		return rows[i].ID < rows[j].ID
	})
	if len(rows) > maxSnapshotRows {
		rows = rows[:maxSnapshotRows]
	}
	snap.Rows = rows
	return snap
}

// startsWithReply implements the H1 sticky-busy guard's content test: the
// message at the tip's message count is an assistant message, i.e. the request
// begins with the reply to the tip's request. A concurrent fork cannot contain
// the reply of a request that is still in flight, so the content proves the tip
// completed even when the busy flag is stale.
//
// The test is sound only because tips are TAIL-EXACT: the tip's entire stream is
// a prefix of the request's, so message index msgCount in the request IS the
// first message past the tip. With the old block-only tips a request could share
// the full blocks but diverge inside the tail, and the role at that index
// belonged to a different message.
//
// (Structural caveat, measured not fixed — design H2: a framework that spawns a
// child AFTER the parent's assistant tool-call turn puts that assistant message
// into the child's history too, where it looks like a continuation.)
func startsWithReply(tipMsgCount int, roles []string) bool {
	if tipMsgCount <= 0 || tipMsgCount >= len(roles) {
		return false
	}
	return roles[tipMsgCount] == "assistant"
}

// sameScope reports whether a branch tip is visible to a request in scope. An
// absent scope on either side matches (no session header → no scoping, the
// design's accepted floor); two different sessions never match.
func sameScope(branchScope, requestScope string) bool {
	return branchScope == "" || requestScope == "" || branchScope == requestScope
}

// branchID mints the opaque branch id: the first 10 hex chars of a hash over the
// mint seed.
func branchID(seed Hash) string {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(seed))
	sum := sha256.Sum256(b[:])
	return hex.EncodeToString(sum[:])[:idHexLen]
}

// mintSeed hashes the branch id's inputs: the creating stream's tip commitment,
// the fork ordinal (siblings of the same content get distinct ids) and the
// session scope (two clients sending look-alike bodies stay distinct even when
// their streams coincide).
func mintSeed(tip Hash, ordinal int, scope string) Hash {
	h := sha256.New()
	var b [16]byte
	binary.LittleEndian.PutUint64(b[:8], uint64(tip))
	binary.LittleEndian.PutUint64(b[8:], uint64(ordinal)) //#nosec G115 -- ordinal is a bounded counter
	h.Write(b[:])
	h.Write([]byte(scope))
	return Hash(binary.LittleEndian.Uint64(h.Sum(nil)[:8]))
}
