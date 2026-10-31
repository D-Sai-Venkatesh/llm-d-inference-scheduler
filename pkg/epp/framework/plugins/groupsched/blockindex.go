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

// Package groupsched implements the group scheduler (ALGORITHM-group.md):
// a working set of sessions whose unique prefix blocks fit a budget of the
// GPU cache, members dispatched immediately, rotation by payback.
//
// This file is the block accounting (decision D4, tier B — the reviewed
// build, applying all six fixes from the adversarial review of the first
// draft):
//
//   - CHAINED hashes: h_i = H(salt ‖ h_{i−1} ‖ chunk_i). The engine's
//     prefix cache is chained — a block is reusable only if every token
//     before it matches — so identical chunk content after divergent
//     prefixes must NOT count as shared, and repeated chunks within one
//     body must not be deduplicated. Chaining makes the hash sequence a
//     position-sensitive fingerprint of the body's prefix structure.
//   - ONE UNIT end to end: the chunk length is derived from the MEASURED
//     engine block size (block_size tokens × ~4 bytes/token), and the
//     index FAILS CLOSED until the caller provides it (NewBlockIndex
//     rejects block_size ≤ 0). The campaign engine runs 64-token blocks
//     (40,183 × 64 = 2.57M) — a hardcoded default silently admitting 4×
//     the cache was the first draft's S0-3.
//   - PER-SESSION EXITING STATE: the non-exiting (budget) counter is the
//     union over seated, non-exiting members — maintained as an
//     invariant, not approximated by refcount tricks. MarkExiting is
//     idempotent; updates on an exiting member never touch the budget
//     counter; Unseat releases from the budget counter only what the
//     session actually contributed.
//   - ATOMIC TrySeat: fit and seat in one critical section (no TOCTOU
//     between FitBlocks and seating).
//   - O(Δ) COUNTER DIFFS: every mutation computes the set delta and
//     applies only the delta — no containsHash scans under the lock.
//   - NO PHANTOM BLOCKS: an empty body hashes to zero chunks. The
//     trailing partial chunk counts as one block (conservative).
//
// Remaining tier-B approximations, accepted and documented (see the
// review's measurement plan before relying on them in production):
//   - chunk boundaries are body bytes, not engine tokens: bytes/token
//     varies by content (CJK ~3 B/char, base64 ~2.5–3 B/token, image
//     URLs severe undercounts — multimodal bodies must be excluded by
//     the caller);
//   - the router sees the raw messages JSON, not the engine's templated
//     prompt; per-request fields serialized before the messages shift
//     chunk alignment across turns (overcount, safe direction);
//   - FNV-64a is salted per index (crafted cross-tenant collisions made
//     non-trivial), not cryptographic.
package groupsched

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"sync"
	"time"
)

// Errors returned by the index. All are programming or wiring faults on
// the caller's side except ErrBudgetUnknown.
var (
	ErrBlockSizeUnknown = errors.New("groupsched: engine block size not set (fail closed)")
	ErrNotTracked       = errors.New("groupsched: session not tracked")
	ErrAlreadySeated    = errors.New("groupsched: session already seated")
)

// DefaultDispatchWindow is the number of recent dispatched requests whose
// blocks make up a session's set (D4: covers sub-agent chains that
// interleave with the main chain; for one growing chain the union of the
// window equals the latest prompt, which is what the engine holds).
const DefaultDispatchWindow = 4

// minChunkBytes floors the derived chunk length (tiny block sizes must
// not produce pathological chunk counts).
const minChunkBytes = 64

// Hash is a chained chunk hash.
type Hash uint64

// BlockIndex is the group-level unique-block accounting: a reference
// counter over the blocks of seated members, plus the same over seated,
// non-exiting members (the budget counter — a draining member's blocks
// do not block admissions).
//
// Session records persist across residencies (Unseat keeps the record so
// a re-seated multi-chain session counts its full set; Forget drops the
// record when the session leaves the system).
//
// Safe for concurrent use.
type BlockIndex struct {
	mu sync.Mutex

	salt            Hash
	blockSizeTokens int
	chunkBytes      int
	// version bumps on EVERY mutation (the scheduler's estimate cache
	// key — M2: Picks run at ~1kHz; unchanged state must not recompute).
	version uint64

	blocks   map[Hash]int // refcount over seated members' sets
	blocksNE map[Hash]int // refcount over seated, non-exiting members

	sessions map[string]*sessionBlocks
	seated   map[string]struct{}
}

// sessionBlocks is one member's rolling block set: the union of its waiting
// requests and its last-window dispatched requests (the fixed 4-request window:
// one entry per live prefix chain, expiring after S, was tried and retired), plus
// the seating state.
type sessionBlocks struct {
	id            string
	window        int
	waiting       map[string][]Hash    // reqID -> chained hashes
	waitSince     map[string]time.Time // reqID -> arrival (the entry-order input)
	lastSizeBytes int64                // last-known wire size (the estimate baseline)
	dispHist      [][]Hash
	dispSet       map[Hash]struct{}
	waitSet       map[Hash]struct{}
	exiting       bool
}

// NewBlockIndex returns an index for an engine whose KV blocks hold
// blockSizeTokens tokens (from cache_config_info / the endpoint metrics).
// It fails closed on blockSizeTokens <= 0: the unit must be measured,
// never assumed.
func NewBlockIndex(blockSizeTokens int) (*BlockIndex, error) {
	if blockSizeTokens <= 0 {
		return nil, fmt.Errorf("%w: got %d", ErrBlockSizeUnknown, blockSizeTokens)
	}
	var saltBytes [8]byte
	_, _ = rand.Read(saltBytes[:])
	return &BlockIndex{
		salt:            Hash(binary.LittleEndian.Uint64(saltBytes[:])),
		blockSizeTokens: blockSizeTokens,
		chunkBytes:      chunkBytesFor(blockSizeTokens),
		blocks:          make(map[Hash]int),
		blocksNE:        make(map[Hash]int),
		sessions:        make(map[string]*sessionBlocks),
		seated:          make(map[string]struct{}),
	}, nil
}

// chunkBytesFor converts the measured block size (tokens) to a chunk
// length (bytes) at ~4 bytes/token, floored.
func chunkBytesFor(blockSizeTokens int) int {
	b := blockSizeTokens * 4
	if b < minChunkBytes {
		b = minChunkBytes
	}
	return b
}

// ChunkBytes is the chunk length in use (blockSizeTokens × ~4).
func (b *BlockIndex) ChunkBytes() int { return b.chunkBytes }

// BlockSizeTokens is the measured engine block size in tokens.
func (b *BlockIndex) BlockSizeTokens() int { return b.blockSizeTokens }

// HashBody chunks a serialized request body and returns the CHAINED hash
// of each chunk: h_i = H(salt ‖ h_{i−1} ‖ chunk_i). Position-sensitive:
// identical content after divergent prefixes hashes differently, and
// repeated chunks within a body are NOT deduplicated (the engine's
// prefix cache is chained the same way). An empty body returns zero
// chunks (no phantom shared block). The trailing partial chunk counts as
// one block (conservative).
func (b *BlockIndex) HashBody(body []byte) []Hash {
	b.mu.Lock()
	salt, cb := b.salt, b.chunkBytes
	b.mu.Unlock()
	return hashBody(body, cb, salt)
}

func hashBody(body []byte, chunkBytes int, salt Hash) []Hash {
	if len(body) == 0 {
		return nil
	}
	n := (len(body) + chunkBytes - 1) / chunkBytes
	out := make([]Hash, 0, n)
	h := fnv.New64a()
	var prev Hash
	for i := 0; i < len(body); i += chunkBytes {
		end := i + chunkBytes
		if end > len(body) {
			end = len(body)
		}
		h.Reset()
		var s [8]byte
		binary.LittleEndian.PutUint64(s[:], uint64(salt))
		h.Write(s[:])
		binary.LittleEndian.PutUint64(s[:], uint64(prev))
		h.Write(s[:])
		h.Write(body[i:end])
		prev = Hash(h.Sum64())
		out = append(out, prev)
	}
	return out
}

// ---- session records ----

// Track creates (or returns) a session record. Idempotent; the record
// persists across residencies until Forget.
func (b *BlockIndex) Track(id string, window int) *sessionBlocks {
	b.mu.Lock()
	defer b.mu.Unlock()
	if s, ok := b.sessions[id]; ok {
		return s
	}
	if window <= 0 {
		window = DefaultDispatchWindow
	}
	s := &sessionBlocks{id: id, window: window,
		waiting: make(map[string][]Hash), waitSince: make(map[string]time.Time),
		dispSet: make(map[Hash]struct{}), waitSet: make(map[Hash]struct{})}
	b.sessions[id] = s
	return s
}

// Forget drops a session's record entirely (the session left the
// system). A seated session is unseated first.
func (b *BlockIndex) Forget(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return
	}
	if _, isSeated := b.seated[id]; isSeated {
		b.releaseLocked(s, !s.exiting)
		delete(b.seated, id)
	}
	delete(b.sessions, id)
}

// set returns the session's full block set (waiting ∪ dispatch window).
// The returned map must not be mutated by callers.
func (s *sessionBlocks) set() map[Hash]struct{} {
	if len(s.waitSet) == 0 {
		return s.dispSet
	}
	out := make(map[Hash]struct{}, len(s.dispSet)+len(s.waitSet))
	for h := range s.dispSet {
		out[h] = struct{}{}
	}
	for h := range s.waitSet {
		out[h] = struct{}{}
	}
	return out
}

// ---- seating ----

// TrySeat seats a session atomically against a budget in BLOCKS: the
// session's current set (its waiting requests — a newcomer's entry
// request must be recorded via UpdateWaiting first) is costed against
// the non-exiting group, and the seat happens only if
// UniqueBlocksNE + cost <= budgetBlocks. Returns (ok, cost).
//
// The waiting request is NOT settled into the dispatch window here; the
// scheduler settles it at dispatch (SettleDispatched) — the first
// draft's pre-settle caused a double window push.
func (b *BlockIndex) TrySeat(id string, budgetBlocks int) (bool, int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		s = b.TrackLocked(id, 0)
	}
	if _, isSeated := b.seated[id]; isSeated {
		return true, 0 // already a member
	}
	cost := 0
	for h := range s.set() {
		if _, held := b.blocksNE[h]; !held {
			cost++
		}
	}
	if len(b.blocksNE)+cost > budgetBlocks {
		return false, cost
	}
	b.version++ // R4: the seated write
	b.seated[id] = struct{}{}
	s.exiting = false // a fresh residency is non-exiting
	set := s.set()
	for h := range set {
		b.blocks[h]++
		b.blocksNE[h]++
	}
	return true, cost
}

// TrackLocked is Track without its own lock.
func (b *BlockIndex) TrackLocked(id string, window int) *sessionBlocks {
	if s, ok := b.sessions[id]; ok {
		return s
	}
	b.version++ // R4: record creation changes SetBlocks
	if window <= 0 {
		window = DefaultDispatchWindow
	}
	s := &sessionBlocks{id: id, window: window,
		waiting: make(map[string][]Hash), waitSince: make(map[string]time.Time),
		dispSet: make(map[Hash]struct{}), waitSet: make(map[Hash]struct{})}
	b.sessions[id] = s
	return s
}

// Unseat removes a session from the group (its record persists; its
// blocks are released from both counters — the budget counter only for
// what it still contributes there, which is nothing if it was exiting).
func (b *BlockIndex) Unseat(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return
	}
	if _, isSeated := b.seated[id]; !isSeated {
		return
	}
	b.releaseLocked(s, !s.exiting)
	delete(b.seated, id)
}

// releaseLocked removes s's set from the all-seated counter and (if
// countNE) from the budget counter. Refcount underflow is a programming
// error and panics (the first draft's `c > 0` guards masked real bugs).
func (b *BlockIndex) releaseLocked(s *sessionBlocks, countNE bool) {
	b.version++ // R4: every mutation path bumps at its primitive
	for h := range s.set() {
		b.dec(b.blocks, h)
		if countNE {
			b.dec(b.blocksNE, h)
		}
	}
}

func (b *BlockIndex) dec(m map[Hash]int, h Hash) {
	c, ok := m[h]
	if !ok || c <= 0 {
		panic(fmt.Sprintf("groupsched: refcount underflow for hash %d", h))
	}
	if c == 1 {
		delete(m, h)
	} else {
		m[h] = c - 1
	}
}

// MarkExiting moves a member's blocks out of the budget (non-exiting)
// counter while it remains seated and served. Idempotent.
func (b *BlockIndex) MarkExiting(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok || s.exiting {
		return // idempotent
	}
	if _, isSeated := b.seated[id]; !isSeated {
		return // an unseated session is not exiting
	}
	b.version++ // R4: the exiting write
	s.exiting = true
	for h := range s.set() {
		b.dec(b.blocksNE, h)
	}
}

// ---- waiting / dispatch bookkeeping ----

// UpdateWaiting ingests a newly arrived request body for a session
// (tracked, seated or not), keyed by the request ID the caller uses at
// dispatch — the scheduler never has to keep the body, and a rewritten
// body is simply re-registered under its reqID. If seated, the session's
// NEW blocks are counted into the group (the budget counter only while
// non-exiting) — the growth re-check substrate (D13).
func (b *BlockIndex) UpdateWaiting(id, reqID string, body []byte) error {
	hashes := b.HashBody(body)
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.TrackLocked(id, 0) // first sight creates the record
	old := s.set()
	if _, ok := s.waiting[reqID]; ok {
		// a retry rewrote the body under the same reqID: drop the old
		// entry (and its waitSet contribution) before registering the new
		delete(s.waiting, reqID)
		s.rebuildWaitSet()
	}
	s.waiting[reqID] = hashes
	for _, h := range hashes {
		s.waitSet[h] = struct{}{}
	}
	b.applyDeltaLocked(s, old)
	return nil
}

// SettleDispatched moves a request from waiting to the dispatch window,
// matched by the request ID registered at UpdateWaiting. A reqID that
// never waited is dispatched directly (its hashes are computed from the
// body the caller passes). Window slides release blocks that left the
// session's set.
func (b *BlockIndex) SettleDispatched(id, reqID string, body []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.TrackLocked(id, 0) // first sight creates the record
	old := s.set()
	hashes, ok := s.waiting[reqID]
	if !ok {
		// never waited (dispatched directly): hash the body now
		hashes = b.HashBodyLocked(body)
	} else {
		delete(s.waiting, reqID)
		s.rebuildWaitSet()
	}
	s.pushDispatched(hashes)
	b.applyDeltaLocked(s, old)
	return nil
}

// HashBodyLocked is HashBody under an already-held lock.
func (b *BlockIndex) HashBodyLocked(body []byte) []Hash {
	return hashBody(body, b.chunkBytes, b.salt)
}

// Cancel drops a waiting request (client disconnect, queue eviction),
// matched by request ID. O(Δ) counter maintenance.
func (b *BlockIndex) Cancel(id, reqID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return ErrNotTracked
	}
	if _, ok := s.waiting[reqID]; !ok {
		return fmt.Errorf("groupsched: no waiting request %s for %s", reqID, id)
	}
	old := s.set()
	delete(s.waiting, reqID)
	s.rebuildWaitSet()
	b.applyDeltaLocked(s, old)
	return nil
}

// applyDeltaLocked applies the counter diff between the session's
// previous set and its current set: increments for new blocks, decrements
// for gone blocks. The budget (non-exiting) counter participates only
// while the session is seated and non-exiting.
func (b *BlockIndex) applyDeltaLocked(s *sessionBlocks, old map[Hash]struct{}) {
	b.version++ // R4: bumps even when unseated (SetBlocks feeds estimates)
	_, isSeated := b.seated[s.id]
	if !isSeated {
		return // unseated sessions' records are private
	}
	countNE := !s.exiting
	cur := s.set()
	for h := range cur {
		if _, had := old[h]; !had {
			b.blocks[h]++
			if countNE {
				b.blocksNE[h]++
			}
		}
	}
	for h := range old {
		if _, has := cur[h]; !has {
			b.dec(b.blocks, h)
			if countNE {
				b.dec(b.blocksNE, h)
			}
		}
	}
}

func (s *sessionBlocks) pushDispatched(hashes []Hash) {
	s.dispHist = append(s.dispHist, hashes)
	if len(s.dispHist) > s.window {
		s.dispHist = s.dispHist[len(s.dispHist)-s.window:]
	}
	s.rebuildDispSet()
}

func (s *sessionBlocks) rebuildDispSet() {
	s.dispSet = make(map[Hash]struct{})
	for _, w := range s.dispHist {
		for _, h := range w {
			s.dispSet[h] = struct{}{}
		}
	}
}

func (s *sessionBlocks) rebuildWaitSet() {
	s.waitSet = make(map[Hash]struct{})
	for _, w := range s.waiting {
		for _, h := range w {
			s.waitSet[h] = struct{}{}
		}
	}
}

// ---- queries ----

// FitBlocks returns how many of the candidate's blocks are NOT already
// held by the non-exiting group: the admission cost estimate. With
// chained hashes the candidate's hashes are position-sensitive, so only
// true prefix sharing discounts.
func (b *BlockIndex) FitBlocks(hashes []Hash) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := 0
	for _, h := range hashes {
		if _, held := b.blocksNE[h]; !held {
			n++
		}
	}
	return n
}

// UniqueBlocks is the group's size in unique blocks (all seated members).
func (b *BlockIndex) UniqueBlocks() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.blocks)
}

// UniqueBlocksNE is the same over seated, non-exiting members — the
// budget's enforcement counter (D6/D13).
func (b *BlockIndex) UniqueBlocksNE() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.blocksNE)
}

// Seated reports whether the session is currently a group member.
func (b *BlockIndex) Seated(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, ok := b.seated[id]
	return ok
}

// Exiting reports the member's exiting state.
func (b *BlockIndex) Exiting(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	return ok && s.exiting
}

// EstimateTokens converts a block count to tokens at the measured block
// size (one unit end to end).
func (b *BlockIndex) EstimateTokens(blocks int) int64 {
	return int64(blocks) * int64(b.blockSizeTokens)
}

// checkInvariantLocked recomputes both counters from the seated sets and
// panics on divergence. Used by tests (the review's invariant checker).
func (b *BlockIndex) checkInvariantLocked() {
	want := make(map[Hash]int)
	wantNE := make(map[Hash]int)
	for id := range b.seated {
		s := b.sessions[id]
		for h := range s.set() {
			want[h]++
			if !s.exiting {
				wantNE[h]++
			}
		}
	}
	if !sameCounts(want, b.blocks) {
		panic(fmt.Sprintf("invariant: blocks %v != recomputed %v", b.blocks, want))
	}
	if !sameCounts(wantNE, b.blocksNE) {
		panic(fmt.Sprintf("invariant: blocksNE %v != recomputed %v", b.blocksNE, wantNE))
	}
}

func sameCounts(a, c map[Hash]int) bool {
	if len(a) != len(c) {
		return false
	}
	for h, n := range a {
		if c[h] != n {
			return false
		}
	}
	return true
}

func sameHashes(a, b []Hash) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---- the scheduler-facing core: Locked variants and waiting-order queries ----
// The scheduler holds its own lock and calls these; the public API above
// wraps the same logic for standalone use and tests.

// UpdateWaitingLocked is UpdateWaiting under an already-held lock, with
// the arrival time recorded for the entry-order queries.
func (b *BlockIndex) UpdateWaitingLocked(id, reqID string, hashes []Hash) {
	s := b.TrackLocked(id, 0)
	old := s.set()
	if _, ok := s.waiting[reqID]; ok {
		delete(s.waiting, reqID)
		s.rebuildWaitSet()
	}
	s.waiting[reqID] = hashes
	s.waitSince[reqID] = time.Now()
	for _, h := range hashes {
		s.waitSet[h] = struct{}{}
	}
	b.applyDeltaLocked(s, old)
}

// SettleDispatchedAtLocked is SettleDispatchedLocked with an explicit
// time (the derived window's chain timestamps — deterministic tests).
// Returns whether the dispatch landed on the session's MAIN chain —
// the largest chain of the derived window (V1: the quantum counts busy
// periods on the main chain; under root identity a busy root with
// overlapping sub-agent threads never fully drains, which would make it
// de-facto Q=∞). In the fixed-window mode every dispatch is main.
func (b *BlockIndex) SettleDispatchedAtLocked(id, reqID string, hashes []Hash, now time.Time) bool {
	s := b.TrackLocked(id, 0)
	old := s.set()
	if _, ok := s.waiting[reqID]; ok {
		delete(s.waiting, reqID)
		delete(s.waitSince, reqID)
		s.rebuildWaitSet()
	}
	main := true
	if hashes != nil {
		s.pushDispatched(hashes)
	}
	b.applyDeltaLocked(s, old)
	return main
}

// SettleDispatchedLocked is SettleDispatched under an already-held lock.
func (b *BlockIndex) SettleDispatchedLocked(id, reqID string, hashes []Hash) {
	s := b.TrackLocked(id, 0)
	old := s.set()
	if _, ok := s.waiting[reqID]; ok {
		delete(s.waiting, reqID)
		delete(s.waitSince, reqID)
		s.rebuildWaitSet()
	}
	if hashes != nil {
		s.pushDispatched(hashes)
	}
	b.applyDeltaLocked(s, old)
}

// CancelLocked is Cancel under an already-held lock.
func (b *BlockIndex) CancelLocked(id, reqID string) error {
	s, ok := b.sessions[id]
	if !ok {
		return ErrNotTracked
	}
	if _, ok := s.waiting[reqID]; !ok {
		return fmt.Errorf("groupsched: no waiting request %s for %s", reqID, id)
	}
	old := s.set()
	delete(s.waiting, reqID)
	delete(s.waitSince, reqID)
	s.rebuildWaitSet()
	b.applyDeltaLocked(s, old)
	return nil
}

// WaitingSince returns the arrival time of the session's OLDEST waiting
// request (zero if none) — the "oldest waiting request first" input.
func (b *BlockIndex) WaitingSince(id string) time.Time {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.waitingSinceLocked(id)
}

func (b *BlockIndex) waitingSinceLocked(id string) time.Time {
	s, ok := b.sessions[id]
	if !ok {
		return time.Time{}
	}
	var oldest time.Time
	for _, t := range s.waitSince {
		if oldest.IsZero() || t.Before(oldest) {
			oldest = t
		}
	}
	return oldest
}

// OldestWaiting returns the reqID of the session's oldest waiting
// request, or "".
func (b *BlockIndex) OldestWaiting(id string) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return ""
	}
	var bestID string
	var best time.Time
	for rid, t := range s.waitSince {
		if best.IsZero() || t.Before(best) || (t.Equal(best) && rid < bestID) {
			best, bestID = t, rid
		}
	}
	return bestID
}

// ExitingLocked reports the exiting state under an already-held lock.
func (b *BlockIndex) ExitingLocked(id string) bool {
	s, ok := b.sessions[id]
	return ok && s.exiting
}

// MarkExitingLocked is MarkExiting under an already-held lock.
func (b *BlockIndex) MarkExitingLocked(id string) {
	s, ok := b.sessions[id]
	if !ok || s.exiting {
		return
	}
	if _, isSeated := b.seated[id]; !isSeated {
		return
	}
	b.version++ // R4: the exiting write
	s.exiting = true
	for h := range s.set() {
		b.dec(b.blocksNE, h)
	}
}

// UnseatLocked is Unseat under an already-held lock.
func (b *BlockIndex) UnseatLocked(id string) {
	s, ok := b.sessions[id]
	if !ok {
		return
	}
	if _, isSeated := b.seated[id]; !isSeated {
		return
	}
	b.releaseLocked(s, !s.exiting)
	delete(b.seated, id)
}

// ForgetLocked is Forget under an already-held lock.
func (b *BlockIndex) ForgetLocked(id string) {
	s, ok := b.sessions[id]
	if !ok {
		return
	}
	if _, isSeated := b.seated[id]; isSeated {
		b.releaseLocked(s, !s.exiting)
		delete(b.seated, id)
	}
	delete(b.sessions, id)
}

// TrySeatLocked is TrySeat under an already-held lock.
func (b *BlockIndex) TrySeatLocked(id string, budgetBlocks int) (bool, int) {
	s := b.TrackLocked(id, 0)
	if _, isSeated := b.seated[id]; isSeated {
		return true, 0
	}
	cost := 0
	for h := range s.set() {
		if _, held := b.blocksNE[h]; !held {
			cost++
		}
	}
	if len(b.blocksNE)+cost > budgetBlocks {
		return false, cost
	}
	b.version++ // R4: the seated write
	b.seated[id] = struct{}{}
	s.exiting = false
	set := s.set()
	for h := range set {
		b.blocks[h]++
		b.blocksNE[h]++
	}
	return true, cost
}

// SeatForceLocked seats a session WITHOUT the budget check: the block set
// is taken as present on this engine (an affinity reconciliation observed
// it there — the engine, not our estimate, is authoritative about what it
// holds). The next enforce pass evicts to fit; an over-budget counter is
// recoverable, a member absent from the engine serving it is not.
func (b *BlockIndex) SeatForceLocked(id string) {
	s := b.TrackLocked(id, 0)
	if _, isSeated := b.seated[id]; isSeated {
		return
	}
	b.version++ // R4: the seated write
	b.seated[id] = struct{}{}
	s.exiting = false
	for h := range s.set() {
		b.blocks[h]++
		b.blocksNE[h]++
	}
}

// FitBlocksLocked is FitBlocks under an already-held lock.
func (b *BlockIndex) FitBlocksLocked(hashes []Hash) int {
	n := 0
	for _, h := range hashes {
		if _, held := b.blocksNE[h]; !held {
			n++
		}
	}
	return n
}

// UniqueBlocksNELocked is UniqueBlocksNE under an already-held lock.
func (b *BlockIndex) UniqueBlocksNELocked() int { return len(b.blocksNE) }

// UniqueBlocksLocked is UniqueBlocks under an already-held lock (ALL
// seated members, including exiting — their blocks stay pinned until
// they drain; the size-aware gate's usage term, M4).
func (b *BlockIndex) UniqueBlocksLocked() int { return len(b.blocks) }

// VersionLocked is the mutation counter under an already-held lock (the
// estimate-cache key).
func (b *BlockIndex) VersionLocked() uint64 { return b.version }

// SetBlocks returns the session's full block set (the fit input).
func (b *BlockIndex) SetBlocks(id string) []Hash {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return nil
	}
	out := make([]Hash, 0, len(s.dispSet)+len(s.waitSet))
	for h := range s.set() {
		out = append(out, h)
	}
	return out
}

// TrackedCount is the number of session records (the leak test's input).
func (b *BlockIndex) TrackedCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.sessions)
}

// LastSizeBytesLocked returns the session's last dispatched body size
// (the estimate delta's baseline; 0 when never dispatched).
func (b *BlockIndex) LastSizeBytesLocked(id string) int64 {
	s, ok := b.sessions[id]
	if !ok {
		return 0
	}
	return s.lastSizeBytes
}

// HasNELocked reports whether a hash is held by the non-exiting group
// (the estimate path's membership test).
func (b *BlockIndex) HasNELocked(h Hash) bool {
	_, ok := b.blocksNE[h]
	return ok
}

// SetSizeLocked records the session's last-known wire size (the estimate
// delta's baseline). Stored on the session record so it persists across
// residencies with the block set.
func (b *BlockIndex) SetSizeLocked(id string, sizeBytes int64) int64 {
	b.version++ // R4: the lastSizeBytes write feeds the estimate delta
	s := b.TrackLocked(id, 0)
	old := s.lastSizeBytes
	s.lastSizeBytes = sizeBytes
	return old
}
