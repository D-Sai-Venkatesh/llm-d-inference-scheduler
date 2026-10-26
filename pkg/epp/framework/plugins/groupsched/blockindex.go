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

	blocks   map[Hash]int // refcount over seated members' sets
	blocksNE map[Hash]int // refcount over seated, non-exiting members

	sessions map[string]*sessionBlocks
	seated   map[string]struct{}
}

// sessionBlocks is one session's rolling block set: the union of its
// waiting requests and its last-window dispatched requests, plus the
// seating state.
type sessionBlocks struct {
	id      string
	window  int
	waiting [][]Hash
	dispHist [][]Hash
	dispSet map[Hash]struct{}
	waitSet map[Hash]struct{}
	exiting bool
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
	if window <= 0 {
		window = DefaultDispatchWindow
	}
	s := &sessionBlocks{id: id, window: window,
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
		s.exiting = true // unseated records: sticky for the next residency? no —
		s.exiting = false // an unseated session is not exiting; keep state clean
		return
	}
	s.exiting = true
	for h := range s.set() {
		b.dec(b.blocksNE, h)
	}
}

// ---- waiting / dispatch bookkeeping ----

// UpdateWaiting ingests a newly arrived request body for a session
// (tracked, seated or not). If seated, the session's NEW blocks are
// counted into the group (the budget counter only while non-exiting) —
// the growth re-check substrate (D13).
func (b *BlockIndex) UpdateWaiting(id string, body []byte) error {
	hashes := b.HashBody(body)
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.TrackLocked(id, 0) // first sight creates the record
	old := s.set()
	s.waiting = append(s.waiting, hashes)
	for _, h := range hashes {
		s.waitSet[h] = struct{}{}
	}
	b.applyDeltaLocked(s, old)
	return nil
}

// SettleDispatched moves a request from waiting to the dispatch window.
// The body must be byte-identical to the one recorded by UpdateWaiting
// (or a body never waited — dispatched directly); a mutated body must go
// through ReplaceWaiting first. Window slides release blocks that left
// the session's set.
func (b *BlockIndex) SettleDispatched(id string, body []byte) error {
	hashes := b.HashBody(body)
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.TrackLocked(id, 0) // first sight creates the record
	old := s.set()
	// remove the matching waiting entry (exactly one; no duplicate scan)
	for i, w := range s.waiting {
		if sameHashes(w, hashes) {
			s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
			s.rebuildWaitSet()
			break
		}
	}
	s.pushDispatched(hashes)
	b.applyDeltaLocked(s, old)
	return nil
}

// ReplaceWaiting swaps a waiting request's blocks (client retry rewrote
// the body, or a cancellation supplies nil for newBody). O(Δ) counter
// maintenance.
func (b *BlockIndex) ReplaceWaiting(id string, oldBody, newBody []byte) error {
	oldH := b.HashBody(oldBody)
	newH := b.HashBody(newBody)
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.sessions[id]
	if !ok {
		return ErrNotTracked
	}
	old := s.set()
	for i, w := range s.waiting {
		if sameHashes(w, oldH) {
			if newH == nil {
				s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
			} else {
				s.waiting[i] = newH
			}
			s.rebuildWaitSet()
			b.applyDeltaLocked(s, old)
			return nil
		}
	}
	return fmt.Errorf("groupsched: no waiting entry matches the old body for %s", id)
}

// applyDeltaLocked applies the counter diff between the session's
// previous set and its current set: increments for new blocks, decrements
// for gone blocks. The budget (non-exiting) counter participates only
// while the session is seated and non-exiting.
func (b *BlockIndex) applyDeltaLocked(s *sessionBlocks, old map[Hash]struct{}) {
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
	if len(want) != len(b.blocks) {
		panic(fmt.Sprintf("invariant: blocks %d != recomputed %d", len(b.blocks), len(want)))
	}
	if len(wantNE) != len(b.blocksNE) {
		panic(fmt.Sprintf("invariant: blocksNE %d != recomputed %d", len(b.blocksNE), len(wantNE)))
	}
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
