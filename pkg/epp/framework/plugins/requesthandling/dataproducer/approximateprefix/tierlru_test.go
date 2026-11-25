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

package approximateprefix

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
)

func chainOf(base, n int) []prefixhash.BlockHash {
	out := make([]prefixhash.BlockHash, n)
	for i := range out {
		out[i] = prefixhash.BlockHash(uint64(base*1000 + i + 1))
	}
	return out
}

// The exported LRU behaves exactly like the indexer's per-pod LRU: tail-first insertion, a held chain is
// a contiguous run from block 0, and the tail goes first.
func TestTierLRUMatchesIndexerOrder(t *testing.T) {
	tl := NewTierLRU(10)
	idx := newHashLRU(10)
	a, b := chainOf(1, 6), chainOf(2, 6)
	for _, c := range [][]prefixhash.BlockHash{a, b} {
		tl.AddChain(c)
		for i := len(c) - 1; i >= 0; i-- {
			idx.Add(c[i])
		}
	}
	require.Equal(t, 10, tl.Len())
	// a's two TAIL blocks were the oldest: both copies dropped exactly those
	require.Equal(t, runLength(idx, a), tl.ResidentPrefix(a))
	require.Equal(t, 4, tl.ResidentPrefix(a))
	require.Equal(t, 6, tl.ResidentPrefix(b))
}

func TestTierLRURefreshAndGen(t *testing.T) {
	tl := NewTierLRU(8)
	a, b := chainOf(1, 4), chainOf(2, 4)
	tl.AddChain(a)
	tl.AddChain(b)
	g := tl.Gen()
	tl.AddChain(a) // refresh only: membership unchanged, a moves to the fresh end
	require.Greater(t, tl.Gen(), g)
	require.Equal(t, 8, tl.Len())
	tl.AddChain(chainOf(3, 2)) // evicts the two oldest = b's tail
	require.Equal(t, 4, tl.ResidentPrefix(a))
	require.Equal(t, 2, tl.ResidentPrefix(b))
}

func TestTierLRUOnEvictNamesEveryDroppedKey(t *testing.T) {
	tl := NewTierLRU(4)
	var dropped []prefixhash.BlockHash
	tl.OnEvict(func(k prefixhash.BlockHash) { dropped = append(dropped, k) })
	a := chainOf(1, 4)
	tl.AddChain(a)
	require.Empty(t, dropped)
	tl.AddChain(chainOf(2, 3))
	// tail-first eviction of a: its last three blocks, deepest first
	require.Equal(t, []prefixhash.BlockHash{a[3], a[2], a[1]}, dropped)
	tl.AddChain(a[:1]) // a held key: refresh, nothing dropped
	require.Len(t, dropped, 3)
}

// TailRoom counts free capacity, then the tail run up to the first protected block, skipping (not
// counting, not stopping at) the inserting chain's own blocks.
func TestTierLRUTailRoom(t *testing.T) {
	tl := NewTierLRU(12)
	out, seated, own := chainOf(1, 3), chainOf(2, 3), chainOf(3, 2)
	tl.AddChain(out)    // oldest
	tl.AddChain(own)    // the head's own leftovers
	tl.AddChain(seated) // newest
	is := func(set []prefixhash.BlockHash) func(prefixhash.BlockHash) bool {
		m := map[prefixhash.BlockHash]bool{}
		for _, k := range set {
			m[k] = true
		}
		return func(k prefixhash.BlockHash) bool { return m[k] }
	}
	// free 4 + out's 3 (own skipped) then stop at seated
	require.Equal(t, 7, tl.TailRoom(100, is(own), is(seated)))
	// early exit: the walk stops once need is reached
	require.GreaterOrEqual(t, tl.TailRoom(5, is(own), is(seated)), 5)
	// nothing protected, nothing skipped: free + everything
	require.Equal(t, 12, tl.TailRoom(100, nil, nil))
	// a protected block AT the tail: only the free capacity
	require.Equal(t, 4, tl.TailRoom(100, nil, is(out)))
}

// Extending a held chain evicts only OTHER content: the chain's surviving blocks are refreshed before the
// inserts, so the observer never names them.
func TestTierLRUExtendEvictsOnlyOthers(t *testing.T) {
	tl := NewTierLRU(10)
	c := chainOf(1, 8)
	o := chainOf(2, 5)
	tl.AddChain(c[:5]) // oldest
	tl.AddChain(o)     // newer
	var dropped []prefixhash.BlockHash
	tl.OnEvict(func(k prefixhash.BlockHash) { dropped = append(dropped, k) })
	tl.AddChain(c) // the chain grows by 3: three OTHER blocks go, tail-first
	require.Equal(t, []prefixhash.BlockHash{o[4], o[3], o[2]}, dropped)
	require.Equal(t, 8, tl.ResidentPrefix(c))
	require.Equal(t, 2, tl.ResidentPrefix(o))
}

// A key the inserting chain shares with a protected chain is skipped, never a stop.
func TestTierLRUTailRoomSkipBeforeStop(t *testing.T) {
	tl := NewTierLRU(6)
	shared := chainOf(1, 2)
	tl.AddChain(shared) // oldest: shared by the head and a seated chain
	tl.AddChain(chainOf(2, 2))
	is := func(set []prefixhash.BlockHash) func(prefixhash.BlockHash) bool {
		m := map[prefixhash.BlockHash]bool{}
		for _, k := range set {
			m[k] = true
		}
		return func(k prefixhash.BlockHash) bool { return m[k] }
	}
	// free 2 + walk: shared (own: skipped, not a stop) then the 2 outside blocks
	require.Equal(t, 4, tl.TailRoom(100, is(shared), is(shared)))
}
