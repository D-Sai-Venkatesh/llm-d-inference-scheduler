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
// This file is the block accounting (decision D4, tier B): session size and
// group commonality are measured in UNIQUE BLOCKS, estimated by chunking
// the serialized request body into fixed-size chunks (~block_size x ~4
// bytes/token) and hashing each chunk. Shared text — the system prompt and
// tool definitions that begin nearly every request body — produces
// identical chunks across sessions and is therefore counted once, which is
// how the engine itself stores it. The estimate is anchored to truth by
// the engine's own per-turn reports (usage.prompt_tokens and
// cached_tokens), which already dedupe: the payback rule needs no block
// knowledge at all, and the admission fit test only needs this estimate
// to be conservative-ish (D13 re-enforces on growth; the 80% dispatch
// gate is the hard backstop).
//
// Known approximation limits, accepted deliberately (tier B of the design):
//   - chunk boundaries are byte offsets in the request body, not token
//     boundaries in the engine's templated prompt. A shared prefix whose
//     length is not a multiple of the chunk size loses its final partial
//     chunk to misalignment. The dominant shared prefix (system prompt)
//     starts at body offset 0, so it aligns exactly; only mid-prompt
//     sharing between divergent sessions can be undercounted.
//   - the router sees the raw request body (the messages array), not the
//     engine's templated prompt. Commonality structure is preserved (the
//     shared fields serialize first); absolute block counts are estimates.
//   - blocks are counted, not sized: a chunk stands for one engine block
//     (block_size tokens) because the chunk size is derived from the
//     measured block size when available.
package groupsched

import (
	"encoding/binary"
	"hash/fnv"
	"sync"
)

// DefaultChunkBytes is the chunk size used when the engine's block size is
// unknown: 1024 bytes ~ 256 tokens x ~4 bytes/token, the deployment's
// block size (cache_config_info) on the campaign stacks.
const DefaultChunkBytes = 1024

// Hash is a chunk hash. FNV-64a: deterministic, dependency-free, fast
// enough (hundreds of MB/s) that hashing a 250k-token (~1 MB) prompt costs
// ~1 ms.
type Hash uint64

// HashBody chunks a serialized request body into fixed-size chunks and
// returns the hash of each full chunk plus the trailing partial chunk.
// The returned slice has len(body)/chunkBytes + 1 entries (at least one,
// for the empty body). Order is body order, so a shared prefix yields a
// shared hash prefix.
func HashBody(body []byte, chunkBytes int) []Hash {
	if chunkBytes <= 0 {
		chunkBytes = DefaultChunkBytes
	}
	n := len(body)/chunkBytes + 1
	out := make([]Hash, 0, n)
	h := fnv.New64a()
	for i := 0; i < len(body); i += chunkBytes {
		end := i + chunkBytes
		if end > len(body) {
			end = len(body)
		}
		h.Reset()
		h.Write(body[i:end])
		out = append(out, Hash(h.Sum64()))
	}
	if len(body) == 0 {
		h.Reset()
		out = append(out, Hash(h.Sum64()))
	}
	return out
}

// HashBodyString is HashBody for string bodies.
func HashBodyString(body string, chunkBytes int) []Hash {
	return HashBody([]byte(body), chunkBytes)
}

// sessionBlocks is one session's rolling block set: the union of the
// blocks of its waiting requests and its last dispatchWindow dispatched
// requests (D4: the window covers sub-agent chains that interleave with
// the main chain; for one growing chain it is simply the latest prompt).
type sessionBlocks struct {
	id       string
	window   int       // dispatch window (4 per the design)
	waiting  [][]Hash  // block hashes of requests waiting (not yet settled)
	dispHist [][]Hash  // block hashes of the last `window` dispatched requests
	dispSet  map[Hash]struct{} // union of dispHist (derived, kept for O(1) membership)
	waitSet  map[Hash]struct{} // union of waiting (derived)
}

func newSessionBlocks(id string, window int) *sessionBlocks {
	if window <= 0 {
		window = 4
	}
	return &sessionBlocks{
		id:      id,
		window:  window,
		dispSet: make(map[Hash]struct{}),
		waitSet: make(map[Hash]struct{}),
	}
}

// set returns the session's full block set (waiting + recent dispatched).
func (s *sessionBlocks) set() map[Hash]struct{} {
	// fast path: nothing waiting
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

// addWaiting records the blocks of a newly arrived request.
func (s *sessionBlocks) addWaiting(hashes []Hash) {
	s.waiting = append(s.waiting, hashes)
	for _, h := range hashes {
		s.waitSet[h] = struct{}{}
	}
}

// settleWaiting moves a request from waiting to dispatched (called at
// dispatch). If hashes is nil the oldest waiting entry is settled.
func (s *sessionBlocks) settleWaiting(hashes []Hash) {
	if hashes == nil && len(s.waiting) > 0 {
		hashes = s.waiting[0]
		s.waiting = s.waiting[1:]
	}
	if hashes == nil {
		return
	}
	// drop from waiting bookkeeping (the request is no longer waiting)
	for i, w := range s.waiting {
		if sameHashes(w, hashes) {
			s.waiting = append(s.waiting[:i], s.waiting[i+1:]...)
			break
		}
	}
	s.rebuildWaitSet()
	s.pushDispatched(hashes)
}

// pushDispatched records a dispatched request's blocks into the window.
func (s *sessionBlocks) pushDispatched(hashes []Hash) {
	s.dispHist = append(s.dispHist, hashes)
	if len(s.dispHist) > s.window {
		s.dispHist = s.dispHist[len(s.dispHist)-s.window:]
	}
	s.rebuildDispSet()
}

// rebuildDispSet recomputes the union over the window (cheap: window<=4).
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

// BlockIndex is the group-level unique-block accounting: a reference
// counter over the blocks of seated members, plus the same over seated,
// non-exiting members (the design's two counters — the budget is enforced
// on the non-exiting set so a draining member's blocks do not block
// admissions).
//
// It is safe for concurrent use.
type BlockIndex struct {
	mu sync.Mutex
	// blocks counts, per hash, how many seated members hold it.
	blocks map[Hash]int
	// blocksNE is the same over non-exiting seated members.
	blocksNE map[Hash]int
	// seated is the per-session block sets of seated members.
	seated map[string]*sessionBlocks
}

// NewBlockIndex returns an empty index.
func NewBlockIndex() *BlockIndex {
	return &BlockIndex{
		blocks:   make(map[Hash]int),
		blocksNE: make(map[Hash]int),
		seated:   make(map[string]*sessionBlocks),
	}
}

// Seat adds a session to the group, counting its unique blocks. The
// session's set is its waiting requests plus its recent dispatched
// window; for a newcomer that is just its waiting request. Returns the
// number of NEW blocks the session added (its admission cost estimate).
func (b *BlockIndex) Seat(id string, window int) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.seated[id]; ok {
		return 0 // already seated
	}
	s := newSessionBlocks(id, window)
	b.seated[id] = s
	return 0
}

// SeatWithSet seats a session carrying a precomputed block set (its
// waiting request's hashes) and returns the number of new blocks added.
func (b *BlockIndex) SeatWithSet(id string, hashes []Hash, window int) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.seated[id]; ok {
		return 0
	}
	s := newSessionBlocks(id, window)
	b.seated[id] = s
	s.addWaiting(hashes)
	s.settleWaiting(hashes)
	added := 0
	for h := range s.set() {
		b.blocks[h]++
		b.blocksNE[h]++
		if b.blocks[h] == 1 {
			added++
		}
	}
	return added
}

// Unseat removes a session; its blocks are released (refcount--), and the
// unique counts fall accordingly.
func (b *BlockIndex) Unseat(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.seated[id]
	if !ok {
		return
	}
	for h := range s.set() {
		if c := b.blocks[h]; c > 0 {
			if c == 1 {
				delete(b.blocks, h)
			} else {
				b.blocks[h] = c - 1
			}
		}
		if c := b.blocksNE[h]; c > 0 {
			if c == 1 {
				delete(b.blocksNE, h)
			} else {
				b.blocksNE[h] = c - 1
			}
		}
	}
	delete(b.seated, id)
}

// MarkExiting moves a session's blocks out of the non-exiting counter
// (the member is draining; its blocks no longer count against the
// admission budget) while it remains seated and served.
func (b *BlockIndex) MarkExiting(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.seated[id]
	if !ok {
		return
	}
	for h := range s.set() {
		if c := b.blocksNE[h]; c > 0 {
			if c == 1 {
				delete(b.blocksNE, h)
			} else {
				b.blocksNE[h] = c - 1
			}
		}
	}
}

// FitBlocks returns how many of the candidate session's blocks are NOT
// already held by the (non-exiting) group: the admission cost estimate.
// A session sharing the common prefix is cheaper, exactly as the engine
// stores it.
func (b *BlockIndex) FitBlocks(hashes []Hash) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := 0
	seen := make(map[Hash]struct{}, len(hashes))
	for _, h := range hashes {
		if _, dup := seen[h]; dup {
			continue // within-session duplicates count once
		}
		seen[h] = struct{}{}
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

// UpdateWaiting ingests a newly arrived request body for a session:
// hashed, added to its waiting set, and — if the session is seated —
// counted into the group (growth is re-checked by the caller via
// UniqueBlocksNE, per D13).
func (b *BlockIndex) UpdateWaiting(id string, body []byte, chunkBytes, window int) {
	hashes := HashBody(body, chunkBytes)
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.seated[id]
	if !ok {
		return // non-member: hashed at admission time, not tracked here
	}
	before := len(s.set())
	s.addWaiting(hashes)
	after := s.set()
	for h := range after {
		if _, old := s.dispSet[h]; !old && !containsHash(s.waiting[:len(s.waiting)-1], h) {
			// new to the session's set: count into the group
			_ = before
			b.blocks[h]++
			b.blocksNE[h]++
		}
	}
}

// SettleDispatched moves a request from waiting to the dispatched window
// for a seated session, refreshing the group counters for blocks that
// were waiting-only and are now dispatched (the set union does not
// change; only the window bookkeeping does).
func (b *BlockIndex) SettleDispatched(id string, body []byte, chunkBytes int) {
	hashes := HashBody(body, chunkBytes)
	b.mu.Lock()
	defer b.mu.Unlock()
	s, ok := b.seated[id]
	if !ok {
		return
	}
	setBefore := s.set()
	s.settleWaiting(hashes)
	setAfter := s.set()
	// blocks may leave the set if the window slid past them
	for h := range setBefore {
		if _, still := setAfter[h]; !still {
			if c := b.blocks[h]; c > 0 {
				if c == 1 {
					delete(b.blocks, h)
				} else {
					b.blocks[h] = c - 1
				}
			}
			if c := b.blocksNE[h]; c > 0 {
				if c == 1 {
					delete(b.blocksNE, h)
				} else {
					b.blocksNE[h] = c - 1
				}
			}
		}
	}
	for h := range setAfter {
		if _, had := setBefore[h]; !had {
			b.blocks[h]++
			b.blocksNE[h]++
		}
	}
}

// EstimateTokens converts a block count to an approximate token count
// (block count x block size), for reporting against the token budget.
func EstimateTokens(blocks, blockSizeTokens int) int64 {
	if blockSizeTokens <= 0 {
		blockSizeTokens = 256
	}
	return int64(blocks) * int64(blockSizeTokens)
}

// BytesToBlocksHint is the inverse rough conversion (bytes -> blocks),
// used to size the chunk length from the measured block size: ~4
// bytes per token x block size tokens per block.
func BytesToBlocksHint(blockSizeTokens int) int {
	if blockSizeTokens <= 0 {
		return DefaultChunkBytes
	}
	b := blockSizeTokens * 4
	if b < 64 {
		b = 64
	}
	return b
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

func containsHash(list [][]Hash, h Hash) bool {
	for _, l := range list {
		for _, x := range l {
			if x == h {
				return true
			}
		}
	}
	return false
}

// itob avoids linter warnings about unused imports in some build tags.
var _ = binary.BigEndian
