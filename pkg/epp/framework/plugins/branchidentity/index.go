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
// body: a request-control plugin that flattens the conversation into a
// canonical byte stream, chains it into block hashes, and matches it against
// the live branch table before admission control, so the fairness id
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
//     reasoning, some stamp dates). The tokenizer's canonical stream is the
//     stable view.
//   - Only branch TIPS are indexed (a chained hash commits to every block
//     before it, so the tip's last block hash plus the block count identifies
//     the whole prefix). Extending an interior node is a new branch by
//     definition.
//   - Continuation: R continues B iff B's tip is a prefix of R's chain AND B
//     has nothing outstanding; otherwise R FORKS a new branch. The one override
//     is the sticky-busy guard (H1): if R extends B's tip and the first message
//     past the tip is an assistant message, R contains B's reply, which an
//     in-flight request cannot have — proven completion, continue B.
//   - BUSY IS MARKED AT ARRIVAL (RequestHeader), not at dispatch: the queued
//     interval is most of a busy period under load (v2c's median wait was 22s),
//     and a branch that looks free while its turn sits in the admission queue
//     would hand a sibling the parent's queue — the head-of-line blocking this
//     design exists to remove. The flag is cleared at end of stream, or expires
//     with the arrival marker when a request is rejected before dispatch.
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
	// maxQueuedPerBranch bounds one branch's backlog of matches that have not
	// dispatched yet. A leak (a request admitted then rejected before
	// PreRequest) can then never grow without bound or wedge the branch.
	maxQueuedPerBranch = 64
	// maxSnapshotRows bounds the StateDumper table.
	maxSnapshotRows = 200
)

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
// The trailing PARTIAL block is DROPPED (n = len(tokens)/blockSize): its
// content changes the moment the conversation grows, so counting it would make
// every continuation look divergent — the false-new-branch direction. A stream
// shorter than one block therefore has no chain at all (see Resolve).
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

// packPseudoTokens packs canonical bytes into little-endian uint32 pseudo-token
// IDs (zero-padded tail). Mirrors the tokenizer's packBytes so the identity path
// and the sizing path chunk the same body at the same boundaries.
func packPseudoTokens(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	if r := len(raw) % bytesPerPseudoToken; r != 0 {
		raw = append(raw, make([]byte, bytesPerPseudoToken-r)...)
	}
	out := make([]uint32, len(raw)/bytesPerPseudoToken)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[i*bytesPerPseudoToken:])
	}
	return out
}

// branchRecord is one live branch: its tip and its liveness.
//
// The tip is (blocks, tip): the number of FULL blocks in the tip request's
// chain and the chained hash of the last of them. Because every chained hash
// commits to all blocks before it, that pair identifies the whole prefix — the
// index needs no chains (design §5, "only tips are stored").
type branchRecord struct {
	id string
	// idSeed is the hash preimage the id was minted from (tip commitment, fork
	// ordinal, scope): it identifies the branch's creation, so a genuine id
	// collision is distinguishable from the same branch being minted again.
	idSeed Hash
	scope  string // client session id at creation, "" when the client sent none
	blocks int
	tip    Hash
	// msgCount is the message count of the tip request. The sticky-busy guard
	// needs it: the message at that index of a later request is the first one
	// past the tip.
	msgCount int

	inflight int
	// queued holds the arrival times of requests that matched this branch and
	// have not dispatched yet (they are in admission control). A queued arrival
	// keeps the branch alive and counts as outstanding for the continuation
	// rule.
	queued []time.Time

	lastSeen time.Time
	created  time.Time
	turns    int
	forks    int
}

// idle reports the branch's silence (no arrival, dispatch or settle).
func (r *branchRecord) idle(now time.Time) time.Duration { return now.Sub(r.lastSeen) }

// pruneQueued drops queued arrivals older than ttl: a request that never
// reached PreRequest (denied in admission control, cancelled) must not keep its
// branch outstanding forever.
func (r *branchRecord) pruneQueued(now time.Time, ttl time.Duration) {
	if len(r.queued) == 0 {
		return
	}
	kept := r.queued[:0]
	for _, t := range r.queued {
		if ttl <= 0 || now.Sub(t) <= ttl {
			kept = append(kept, t)
		}
	}
	r.queued = kept
}

// outstanding reports whether the branch has anything in flight or queued — the
// continuation rule's "busy" predicate (design: "nothing in flight or queued").
func (r *branchRecord) outstanding() bool {
	return r.inflight > 0 || len(r.queued) > 0
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
}

// branchIndex is the branch table: branch id → record, plus the tip lookup
// (last block hash → candidate branches). Safe for concurrent use.
type branchIndex struct {
	mu sync.Mutex

	blockSize   int
	salt        string
	queuedTTL   time.Duration
	maxBranches int

	branches map[string]*branchRecord
	byTip    map[Hash][]*branchRecord
}

func newBranchIndex(blockSize int, salt string, queuedTTL time.Duration, maxBranches int) *branchIndex {
	if blockSize <= 0 {
		blockSize = defaultBlockSizeTokens
	}
	return &branchIndex{
		blockSize:   blockSize,
		salt:        salt,
		queuedTTL:   queuedTTL,
		maxBranches: maxBranches,
		branches:    make(map[string]*branchRecord),
		byTip:       make(map[Hash][]*branchRecord),
	}
}

// Chain hashes a canonical byte stream into the request's block chain.
func (ix *branchIndex) Chain(canonical []byte) []Hash {
	tokens := packPseudoTokens(canonical)
	return ChainedBlockHashes(tokens, ix.salt, ix.blockSize)
}

// Resolve identifies the branch for an arriving request and records the
// arrival. roles is the request's message roles in order (empty for body shapes
// without messages — the sticky-busy guard then cannot fire). bodyKey is the
// hash of the whole canonical stream; it only matters when the stream is shorter
// than one block and therefore has no chain (see mintLocked).
//
// The decision, in order:
//  1. longest branch tip that is a prefix of the request's chain, in the same
//     session scope (a tip from another session never matches; an unscoped tip
//     matches any session).
//  2. tip not outstanding → continue (a repeat of a finished turn).
//  3. sticky-busy guard (when enabled): the first message past the tip is an
//     assistant message → the request contains the tip's reply → continue.
//  4. tip outstanding and unproven (unless forkOnBusy is off) → fork: an
//     identical prompt is a parallel sibling, an extension is context
//     inheritance. Both need their own queue.
//  5. no matching tip → new branch.
func (ix *branchIndex) Resolve(now time.Time, chain []Hash, bodyKey Hash, scope string, roles []string, forkOnBusy, stickyGuard bool) ResolveResult {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	rec, matched := ix.matchLocked(now, chain, scope)
	if rec == nil {
		fresh := ix.mintLocked(now, nil, chain, bodyKey, scope, len(roles))
		return ResolveResult{ID: fresh.id, Outcome: OutcomeNew, Blocks: len(chain)}
	}

	// Continuation needs a FREE tip: a tip with a request in flight or queued
	// belongs to that request (an identical prompt arriving meanwhile is a
	// parallel sibling, not a continuation), unless content proves completion
	// (H1) or the fork rule is configured off.
	cont := !rec.outstanding()
	sticky := false
	if !cont {
		if stickyGuard {
			sticky = startsWithReply(rec.msgCount, roles)
		}
		cont = sticky || !forkOnBusy
	}

	if cont {
		ix.advanceLocked(now, rec, chain, len(roles))
		return ResolveResult{
			ID: rec.id, Outcome: OutcomeMatched, Blocks: len(chain),
			MatchedBlocks: matched, StickyBusy: sticky,
		}
	}

	fork := ix.mintLocked(now, rec, chain, bodyKey, scope, len(roles))
	rec.forks++
	return ResolveResult{
		ID: fork.id, Outcome: OutcomeForked, Blocks: len(chain), MatchedBlocks: matched,
	}
}

// matchLocked returns the branch whose tip is the LONGEST prefix of chain, or
// nil. Scoped tips only match their own session; unscoped tips match any (the
// floor the design accepts when the client sends no session header).
func (ix *branchIndex) matchLocked(now time.Time, chain []Hash, scope string) (*branchRecord, int) {
	for i := len(chain) - 1; i >= 0; i-- {
		var best *branchRecord
		bestOutstanding := true
		for _, cand := range ix.byTip[chain[i]] {
			if cand.blocks != i+1 {
				// Same last hash at a different length can only be a hash
				// collision; it is not a prefix relation, so skip it.
				continue
			}
			if !sameScope(cand.scope, scope) {
				continue
			}
			cand.pruneQueued(now, ix.queuedTTL)
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
			return best, i + 1
		}
	}
	return nil, 0
}

// advanceLocked moves the branch's tip to the arriving request's chain and
// records the arrival. The tip moves at ARRIVAL (design §5): the request is the
// branch's newest context whether or not it has dispatched yet, and a later
// request extending it must see it.
func (ix *branchIndex) advanceLocked(now time.Time, rec *branchRecord, chain []Hash, msgCount int) {
	ix.setTipLocked(rec, chain)
	rec.msgCount = msgCount
	rec.lastSeen = now
	rec.turns++
	if len(rec.queued) < maxQueuedPerBranch {
		rec.queued = append(rec.queued, now)
	}
}

// setTipLocked re-points the tip index at the record's new chain.
func (ix *branchIndex) setTipLocked(rec *branchRecord, chain []Hash) {
	newBlocks, newTip := len(chain), Hash(0)
	if newBlocks > 0 {
		newTip = chain[newBlocks-1]
	}
	if rec.blocks == newBlocks && rec.tip == newTip {
		return
	}
	ix.unindexLocked(rec)
	rec.blocks, rec.tip = newBlocks, newTip
	if rec.blocks > 0 {
		ix.byTip[rec.tip] = append(ix.byTip[rec.tip], rec)
	}
}

// unindexLocked removes the record from the tip table (GC, re-tip).
func (ix *branchIndex) unindexLocked(rec *branchRecord) {
	if rec.blocks == 0 {
		return
	}
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
}

// mintLocked creates a branch, ids derived from the creating chain's tip
// commitment (plus a fork ordinal, so a fork of the same content gets its own
// id): same conversation → same id, on any instance, without a registry to
// consult. The id does not change as the branch continues — continuations keep
// the record's id, only the tip moves.
//
// A stream shorter than one block (a tiny first turn) has no chain: its id then
// comes from the whole-stream hash, so distinct short conversations stay
// distinct and an identical retry reuses the id. Such a branch is never
// MATCHABLE (a tip needs at least one full block), so its continuations report
// as new — the id they get is still the right one.
func (ix *branchIndex) mintLocked(now time.Time, parent *branchRecord, chain []Hash, bodyKey Hash, scope string, msgCount int) *branchRecord {
	tip := bodyKey
	if len(chain) > 0 {
		tip = chain[len(chain)-1]
	}
	ordinal := 0
	if parent != nil {
		ordinal = parent.forks + 1
	}
	seed := mintSeed(tip, ordinal, scope)
	id := branchID(seed)
	// Two distinct chains never share a tip commitment short of a hash
	// collision; this loop only guards the 40-bit id prefix.
	for i := 0; i < 1<<10; i++ {
		existing, clash := ix.branches[id]
		if !clash || existing == nil || existing.idSeed == seed {
			break
		}
		ordinal++
		seed = mintSeed(tip, ordinal, scope)
		id = branchID(seed)
	}
	rec := &branchRecord{
		id:       id,
		idSeed:   seed,
		scope:    scope,
		msgCount: msgCount,
		created:  now,
		lastSeen: now,
		turns:    1,
		queued:   []time.Time{now},
	}
	if parent != nil && rec.scope == "" {
		// A fork inherits its parent's scope when the request carried none:
		// siblings belong to the same session.
		rec.scope = parent.scope
	}
	ix.branches[id] = rec
	ix.setTipLocked(rec, chain)
	return rec
}

// Dispatch marks one request as dispatched (in flight) for the branch and
// consumes its queued arrival. Idempotent-ish: an unknown id is ignored (the
// branch may have been GC'd while the request was queued).
func (ix *branchIndex) Dispatch(id string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.branches[id]
	if !ok || rec == nil {
		return false
	}
	rec.inflight++
	rec.lastSeen = now
	rec.pruneQueued(now, ix.queuedTTL)
	if len(rec.queued) > 0 {
		rec.queued = rec.queued[1:]
	}
	return true
}

// Settle ends one dispatched request on the branch (response complete or
// aborted). The tip stays where the request left it; only liveness changes.
func (ix *branchIndex) Settle(id string, now time.Time) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	rec, ok := ix.branches[id]
	if !ok || rec == nil {
		return false
	}
	if rec.inflight > 0 {
		rec.inflight--
	}
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
// and it never removes a branch with anything in flight or queued.
func (ix *branchIndex) EnforceCap(now time.Time) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	removed := 0
	for _, rec := range ix.branches {
		rec.pruneQueued(now, ix.queuedTTL)
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

// branchTable is the StateDumper payload.
type branchTable struct {
	Branches  int         `json:"branches"`
	Busy      int         `json:"busy"`
	Queued    int         `json:"queued"`
	InFlight  int         `json:"inFlight"`
	BlockSize int         `json:"blockSizeTokens"`
	Rows      []branchRow `json:"rows"`
}

type branchRow struct {
	ID          string  `json:"id"`
	Scope       string  `json:"scope,omitempty"`
	Blocks      int     `json:"blocks"`
	MsgCount    int     `json:"msgCount"`
	InFlight    int     `json:"inflight"`
	Queued      int     `json:"queued"`
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
		rec.pruneQueued(now, ix.queuedTTL)
		if rec.inflight > 0 {
			snap.Busy++
		}
		if len(rec.queued) > 0 {
			snap.Queued++
		}
		snap.InFlight += rec.inflight
		rows = append(rows, branchRow{
			ID:          rec.id,
			Scope:       rec.scope,
			Blocks:      rec.blocks,
			MsgCount:    rec.msgCount,
			InFlight:    rec.inflight,
			Queued:      len(rec.queued),
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

// StreamHash is the whole-stream content hash used to derive an id for a body
// too short to have a block chain.
func StreamHash(canonical []byte) Hash {
	sum := sha256.Sum256(canonical)
	return Hash(binary.LittleEndian.Uint64(sum[:8]))
}

// branchID mints the opaque branch id: the first 10 hex chars of a hash over the
// mint seed.
func branchID(seed Hash) string {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(seed))
	sum := sha256.Sum256(b[:])
	return hex.EncodeToString(sum[:])[:idHexLen]
}

// mintSeed hashes the branch id's inputs: the creating chain's tip commitment,
// the fork ordinal (siblings of the same content get distinct ids) and the
// session scope (two clients sending look-alike bodies stay distinct even when
// their chains coincide).
func mintSeed(tip Hash, ordinal int, scope string) Hash {
	h := sha256.New()
	var b [16]byte
	binary.LittleEndian.PutUint64(b[:8], uint64(tip))
	binary.LittleEndian.PutUint64(b[8:], uint64(ordinal)) //#nosec G115 -- ordinal is a bounded counter
	h.Write(b[:])
	h.Write([]byte(scope))
	return Hash(binary.LittleEndian.Uint64(h.Sum(nil)[:8]))
}
