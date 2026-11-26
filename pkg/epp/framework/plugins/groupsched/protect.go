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

// THE STRICT RULE (leastRecentlyServed) — the order of the seated members' queued turns at the dispatch gate:
//
//  1. Candidate X = the queued member served longest ago (the router's last dispatch of it in this stay; not
//     yet served in this stay first of all; ties by arrival). The time order is round robin by last service,
//     and since vLLM touches a chain only when its turn is dispatched, it is also "closest to eviction first".
//  2. Protect: X's write (its prompt beyond the tier-resident prefix: the cold part + the new context) evicts
//     the least-recently-touched blocks of the tier — X's DANGER ZONE. If a PROTECTABLE queued member Y has
//     blocks there, serve Y first when that is a NET SAVE: a > b, where
//     a = Y's blocks in X's danger zone (what Y loses if X goes first), and
//     b = queued members' blocks in Y's own danger zone that X's write would spare (what serving Y first
//     newly costs — in practice X's own tail, which sits at the bottom because X is the stalest).
//     Several at risk: checked in served-longest-ago order, the first that passes is served.
//  3. Otherwise serve X.
//
// PROTECTABLE: a queued member inside its quantum; while nobody waits outside, every queued member (a
// quantum-complete member then stays, and its cache is worth keeping; with someone waiting it leaves soon and
// its chain is lost during the exile anyway). Idle members have no queued turn and running ones are not
// queued: neither is ever protected. Blocks are compared with blocks — no constant, no exchange rate; cache
// state decides only who is in a danger zone, never who is seated. Re-decided at every dispatch; no bound (a
// protected member is dispatched, so it is not queued, and its refreshed chain needs about one survival age to
// sink back to the bottom).

// headKeySet is a turn's chain keys as a set.
func headKeySet(h Head) map[Hash]struct{} {
	n := 0
	for _, c := range h.Chains {
		n += len(c.Keys)
	}
	set := make(map[Hash]struct{}, n)
	for _, c := range h.Chains {
		for _, k := range c.Keys {
			set[k] = struct{}{}
		}
	}
	return set
}

// dangerZone is the blocks a dispatched turn's write would evict from g's tier index: as many as the turn
// writes (its prompt beyond the tier-resident prefix, in blocks) beyond the free capacity, from the eviction
// end, the turn's own blocks skipped. nil when the write fits the free room (or without a tier index).
func (s *Scheduler) dangerZone(g *engineGroup, h Head) map[Hash]struct{} {
	if g.tier == nil || s.blockSize <= 0 {
		return nil
	}
	resident := s.residentTokens(h.Chains, []*engineGroup{g})
	cold := physCold(PromptTokens(h.Chains), resident, 0)
	bs := int64(s.blockSize)
	n := int((cold+bs-1)/bs) - (g.tier.Cap() - g.tier.Len())
	if n <= 0 {
		return nil
	}
	own := headKeySet(h)
	keys := g.tier.TailKeys(n, func(k Hash) bool { _, ok := own[k]; return ok })
	dz := make(map[Hash]struct{}, len(keys))
	for _, k := range keys {
		dz[k] = struct{}{}
	}
	return dz
}

// protectFirst is the strict rule's step 2 for candidate x over the other queued turns (rest, in the time
// order). It returns the index in rest of the member to serve first (-1 = serve x), that member's protected
// blocks (a), and whether some member was at risk but no protection was a net save (declined).
func (s *Scheduler) protectFirst(g *engineGroup, x Head, rest []Head) (int, int, bool) {
	dzX := s.dangerZone(g, x)
	if len(dzX) == 0 {
		return -1, 0, false
	}
	waiting := len(s.waiters) > 0
	var queued map[Hash]struct{} // every queued turn's keys (x's included), built on first need
	declined := false
	for j, y := range rest {
		if waiting && s.quantumDone(g, y.Thread) {
			continue // about to leave: its chain is lost during the exile anyway
		}
		a := 0
		for _, c := range y.Chains {
			for _, k := range c.Keys {
				if _, ok := dzX[k]; ok {
					a++
				}
			}
		}
		if a == 0 {
			continue
		}
		if queued == nil {
			queued = headKeySet(x)
			for _, r := range rest {
				for _, c := range r.Chains {
					for _, k := range c.Keys {
						queued[k] = struct{}{}
					}
				}
			}
		}
		b := 0
		for k := range s.dangerZone(g, y) {
			if _, lostAnyway := dzX[k]; lostAnyway {
				continue // X's write would evict it too: serving Y first changes only when, not whether
			}
			if _, q := queued[k]; q {
				b++
			}
		}
		if a > b {
			return j, a, false
		}
		declined = true
	}
	return -1, 0, declined
}
