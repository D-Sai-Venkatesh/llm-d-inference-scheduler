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

// THE CPU-RESIDENCY INDEX ORDERS PLACEMENT AND NOTHING ELSE: a head goes to the engine whose tier
// still holds the most of its prompt when it fits there, the eviction today's rule makes anyway is
// made where the head is warm, and with the index off every decision is today's.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/approximateprefix"
)

// residentOn is a residency stub: the threads listed hold their whole chain on that engine only.
func residentOn(where map[string]string) func(keys []Hash, engines []string) []int {
	owner := map[Hash]string{}
	for id, e := range where {
		for _, c := range chain(id, 4000) {
			for _, k := range c.Keys {
				owner[k] = e
			}
		}
	}
	return func(keys []Hash, engines []string) []int {
		out := make([]int, len(engines))
		for i, e := range engines {
			n := 0
			for _, k := range keys {
				if owner[k] != e {
					break
				}
				n++
			}
			out[i] = n
		}
		return out
	}
}

// twoEngines seats filler-a (100 blocks) on "a" and filler-b (500 blocks) on "b", so "a" has the
// more free budget and today's rule places the next head there.
func twoEngines(t *testing.T) *Scheduler {
	t.Helper()
	s := tierSched(t, "a", "b")
	for _, f := range []struct {
		id     string
		blocks int
	}{{"filler-a", 100}, {"filler-b", 500}} {
		h := headN(f.id, 0, time.Second, chain(f.id, f.blocks))
		start(t, s, h)
		s.Settle(h.ReqID, "", epoch)
	}
	require.Equal(t, 1, engine(s, "a").Seats)
	require.Equal(t, 1, engine(s, "b").Seats)
	return s
}

func seatEngineOf(s *Scheduler, id string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bindings[id]
}

// Off: the returner goes where today's rule puts it (the most free budget after the seat).
// On, with its prompt still on "b": it goes to "b", which also fits.
func TestResidencyPlacesOnTheWarmEngine(t *testing.T) {
	off := twoEngines(t)
	ret := headN("ret", 3, 2*time.Second, chain("ret", 200))
	off.Pass([]Head{ret}, epoch)
	require.Equal(t, "a", seatEngineOf(off, "ret"), "without the index: most free budget after the seat")
	require.Nil(t, off.Snapshot(epoch).Residency, "no residency ledger without the index")

	on := twoEngines(t)
	on.SetResidency(residentOn(map[string]string{"ret": "b"}))
	on.Pass([]Head{ret}, epoch)
	require.Equal(t, "b", seatEngineOf(on, "ret"), "with the index: the engine whose tier still holds the prompt")
	r := on.Snapshot(epoch).Residency
	require.NotNil(t, r)
	require.Equal(t, int64(1), r.Placements["warmest"])
	require.Equal(t, int64(200*64), r.SeatResidentTokens)
	require.Equal(t, int64(0), r.SeatColdTokens)

	// cold everywhere (a newcomer, nothing resident): today's rule decides
	on.Pass([]Head{headN("new", 0, time.Second, chain("new", 50))}, epoch)
	require.Equal(t, "a", seatEngineOf(on, "new"))
	require.Equal(t, int64(1), on.Snapshot(epoch).Residency.Placements["cold-everywhere"])
}

// The index never admits what does not fit: a warm engine without room loses the head to the
// engine that has room (no wait, no eviction for it), and the seat is booked "not-warmest".
func TestResidencyNeverOverridesTheFit(t *testing.T) {
	s := tierSched(t, "a", "b")
	big := headN("big", 0, time.Second, chain("big", 3700)) // fills b's 3,800-block seat budget
	s.SetResidency(residentOn(map[string]string{"big": "b", "ret": "b"}))
	start(t, s, big)
	s.Settle(big.ReqID, "", epoch)
	require.Equal(t, "b", seatEngineOf(s, "big"))

	// "ret" is warm on b, but b cannot take 200 more blocks and "big" is mid-quantum, active: no victim
	ret := headN("ret", 1, time.Second, chain("ret", 200))
	bigNext := headN("big", 1, time.Second, chain("big", 3700))
	s.Pass([]Head{ret, bigNext}, epoch)
	require.Equal(t, "a", seatEngineOf(s, "ret"), "the head goes where it fits; the index only orders fitting engines")
	require.Equal(t, "b", seatEngineOf(s, "big"), "nobody was evicted for the warm engine")
	require.Equal(t, int64(1), s.Snapshot(epoch).Residency.Placements["not-warmest"])
}

// When the head fits nowhere, today's rule evicts on ONE engine anyway; the index only chooses which
// (the one where the head is warm), never whether or whom.
func TestChooseEngineEvictsWhereTheHeadIsWarm(t *testing.T) {
	victim := func() bool { return true }
	cands := []EngineCandidate{
		{Engine: "a", CostBlocks: 300, FreeBudget: 200, Budget: 3800, HasVictim: victim},
		{Engine: "b", CostBlocks: 300, FreeBudget: 50, Budget: 3800, HasVictim: victim},
	}
	require.Equal(t, 0, chooseEngine(cands), "index off: the most free budget")
	cands[0].PhysColdTokens, cands[1].PhysColdTokens = 300*64, 0
	require.Equal(t, 1, chooseEngine(cands), "index on: the engine where the head is warm")

	noVictim := func() bool { return false }
	cands[1].HasVictim = noVictim
	require.Equal(t, 0, chooseEngine(cands), "a warm engine with no eligible victim is never forced")
}

// Among fitting engines the order is fewest tokens to really prefill, then today's order; equal
// PhysColdTokens (index off) is today's order exactly.
func TestBetterFitOrder(t *testing.T) {
	a := EngineCandidate{Engine: "a", FreeBudget: 1000, CostBlocks: 100}
	b := EngineCandidate{Engine: "b", FreeBudget: 500, CostBlocks: 100}
	require.True(t, betterFit(a, b))
	b.PhysColdTokens, a.PhysColdTokens = 0, 6400
	require.True(t, betterFit(b, a))
}

// The startup gate: a named producer must exist, be an approx-prefix producer, and model the CPU tier.
func TestResidencyProducerGate(t *testing.T) {
	build := func(t *testing.T, producerParams string) (*GroupScheduler, error) {
		t.Helper()
		h := fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
		if producerParams != "" {
			p, err := approximateprefix.ApproxPrefixCacheFactory("cpu-residency", json.NewDecoder(strings.NewReader(producerParams)), h)
			require.NoError(t, err)
			h.AddPlugin("cpu-residency", p)
		}
		raw, err := Factory("group", json.NewDecoder(strings.NewReader(
			`{"tierCapacityTokens":256000,"cpuResidencyProducer":"cpu-residency"}`)), h)
		require.NoError(t, err)
		r := raw.(*GroupScheduler)
		return r, r.validateResidencyProducer()
	}
	want := `"lruCapacityPerServer":4000`
	good := `{"autoTune":false,"blockSizeTokens":64,` + want + `,"maxPrefixTokensToMatch":0,"maxPrefixBlocksToMatch":0}`
	r, err := build(t, good)
	require.NoError(t, err)
	require.NotNil(t, r.sched.residency, "a valid producer is installed")

	for name, params := range map[string]string{
		"missing":     "",
		"autoTune":    `{"autoTune":true,"blockSizeTokens":64,` + want + `,"maxPrefixTokensToMatch":0,"maxPrefixBlocksToMatch":0}`,
		"GPU-sized":   `{"autoTune":false,"blockSizeTokens":64,"lruCapacityPerServer":1000,"maxPrefixTokensToMatch":0,"maxPrefixBlocksToMatch":0}`,
		"capped":      `{"autoTune":false,"blockSizeTokens":64,` + want + `}`,
		"wrong block": `{"autoTune":false,"blockSizeTokens":128,"lruCapacityPerServer":2000,"maxPrefixTokensToMatch":0,"maxPrefixBlocksToMatch":0}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := build(t, params)
			require.Error(t, err)
		})
	}

	_, err = Factory("group", json.NewDecoder(strings.NewReader(`{"cpuResidencyProducer":"cpu-residency"}`)), nil)
	require.Error(t, err, "the index models the CPU tier: without tierCapacityTokens it is refused")
}
