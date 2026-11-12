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

// THE RUNNING INDEX UNDER CONCURRENCY: the pass, the request path and a scrape all at once. The lock
// convention says every unexported running.go function is reached from a locked exported entry point, and
// this is the test that would catch it if that stopped being true — `-race` needs two goroutines to say
// anything at all, and before this test the package had none.

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/prometheus/client_golang/prometheus"
)

// TestRunningIndexUnderConcurrentHooks drives one scheduler from three goroutines — Pick at "1 kHz", the
// request path (PreRequest / first chunk / settle, with abandons and off-seat landings), and a scrape loop
// that also flaps an engine — then asserts the index still equals the records it claims to count.
func TestRunningIndexUnderConcurrentHooks(t *testing.T) {
	const params = `{"groupBudgetSource":"max","cpuTierTokens":121600,"maxNumSeqs":4,"blockSizeTokens":64}`
	built, err := Factory("gs", json.NewDecoder(strings.NewReader(params)), nil)
	if err != nil {
		t.Fatal(err)
	}
	gs := built.(*GroupScheduler)
	reg := prometheus.NewRegistry()
	if err := gs.metrics.register(reg); err != nil {
		t.Fatal(err)
	}
	s := gs.sched
	for _, e := range []string{"e0", "e1"} {
		if err := s.SetNumBlocksFor(e, 64, 1000); err != nil {
			t.Fatal(err)
		}
	}

	released := make(chan string, 10000)
	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() { // the flow-control processor: one Pass per cycle, one dispatch per Pass
		defer wg.Done()
		defer close(stop)
		turn := map[string]int{}
		for k := 0; k < 4000; k++ {
			var heads []Head
			for j := 0; j < 12; j++ {
				id := fmt.Sprintf("t%02d", (k+j)%24)
				heads = append(heads, Head{Thread: id, ReqID: fmt.Sprintf("%s-%d", id, turn[id]),
					Arrival: epoch.Add(time.Duration(k) * time.Millisecond),
					Chains:  chain(id, 50+((k+j)%24)*20), Returner: turn[id] > 0})
			}
			if i := s.Pass(heads, epoch.Add(time.Duration(k)*time.Millisecond)); i >= 0 {
				turn[heads[i].Thread]++
				released <- heads[i].ReqID
			}
		}
	}()

	wg.Add(1)
	go func() { // the request path: PreRequest, first chunk, settle — an abandon every 17th, an off-seat
		defer wg.Done() // landing every 9th (e1 is a spine apart, so it rebinds and MOVES the running charge)
		n := 0
		for {
			select {
			case rid := <-released:
				n++
				now := epoch.Add(time.Duration(n) * time.Millisecond)
				if n%17 == 0 {
					s.Abandon(rid, now)
					continue
				}
				eng := "e0"
				if n%9 == 0 {
					eng = "e1"
				}
				s.AtPod(rid, eng, now)
				s.FirstChunk(rid, now)
				s.Settle(rid, eng, fwkrequest.Usage{PromptTokens: 10,
					PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 5}}, now)
			case <-stop:
				return
			}
		}
	}()

	wg.Add(1)
	go func() { // the scrape, the state dump, and one engine flap under both
		defer wg.Done()
		for k := 0; ; k++ {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := reg.Gather(); err != nil {
				t.Errorf("scrape: %v", err)
				return
			}
			if _, err := gs.DumpState(); err != nil {
				t.Errorf("dump: %v", err)
				return
			}
			if k == 50 {
				s.RemoveEngine("e1", epoch)
				if err := s.SetNumBlocksFor("e1", 64, 1000); err != nil {
					t.Errorf("re-add: %v", err)
					return
				}
			}
		}
	}()
	wg.Wait()

	// THE INVARIANT, read with every writer stopped: the index is exactly the records' own claim. A
	// double charge, a lost release or a charge left on a dead group all show up here.
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range s.groupOrder {
		g := s.groups[id]
		seqs, recs := 0, 0
		want := map[Hash]int{}
		for _, rec := range s.reqs {
			if rec.runningOn != g {
				continue
			}
			recs++
			seqs++
			for h := range rec.runSet {
				want[h]++
			}
		}
		if g.running.Seqs() != recs {
			t.Fatalf("%s: index says %d in flight, %d records claim it", id, g.running.Seqs(), recs)
		}
		if g.running.Blocks() != len(want) {
			t.Fatalf("%s: index holds %d distinct blocks, the records claim %d", id, g.running.Blocks(), len(want))
		}
		for h, c := range want {
			if g.running.blocks[h] != c {
				t.Fatalf("%s: refcount for %d is %d, records claim %d", id, h, g.running.blocks[h], c)
			}
		}
	}
}
