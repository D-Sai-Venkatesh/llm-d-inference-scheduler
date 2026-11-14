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

// THE COST (I8): one heavy pass — 3 pods with 250-entry CPU lines, 500 waiters of which 60 are warm
// returners — so the per-decision bound O(pods × (line + warm returners)) can be read off a number
// rather than argued from the code. The fixture is rebuilt outside the timed region.

import (
	"fmt"
	"testing"
	"time"
)

// benchFixture is the heavy state: three pods whose lines are full, a 500-deep outside queue and 60
// warm returners spread over the pods. blocked makes the ONE sweep the pass's whole cost (the steady
// state a ~1 kHz Pick lives in): the room cannot take the head and no seat is reclaimable.
func benchFixture(b *testing.B, blocked bool) (*Scheduler, []Head) {
	b.Helper()
	tier := int64(defaultCPUTierTokens)
	if blocked {
		// a line of 250 entries of 640 tokens fills this tier, so the head has no room anywhere
		tier = 200_000
	}
	s := newNCA2Tuned(b, func(c *Config) { c.CPUTierTokens = i64(tier) },
		[]string{"p0", "p1", "p2"})
	if blocked {
		for _, pod := range []string{"p0", "p1", "p2"} {
			g := s.groups[pod]
			for i := 0; i < 3; i++ {
				id := fmt.Sprintf("%s-seat%d", pod, i)
				chains := chain(id, 300) // 900 blocks of the 950-block budget: nothing is reclaimable
				s.seat(g, &waiter{id: id, since: epoch, chains: chains}, 0, epoch)
			}
		}
	}
	now := epoch
	s.mu.Lock()
	defer s.mu.Unlock()
	// 250 entries per pod, the real tier: positions up to ~160k tokens, far inside L_a
	for _, pod := range []string{"p0", "p1", "p2"} {
		g := s.groups[pod]
		for i := 0; i < 250; i++ {
			id := fmt.Sprintf("%s-t%d", pod, i)
			g.nca2.line = append(g.nca2.line, nca2Entry{thread: id, tokens: 640, lastTouch: now})
			// the first 20 per pod are ADMITTED and waiting: the warm returners (60 in all)
			if i < 20 {
				s.nca2.homes[id] = pod
				s.nca2.admitted[id] = true
				s.nca2.ever[id] = true
				s.nca2.inf[id] = 1 // active, so the room is measured below the deepest of them
			}
		}
	}
	// 440 outsiders of 50 blocks each, plus the 60 returners' heads
	heads := make([]Head, 0, 500)
	for i := 0; i < 440; i++ {
		id := fmt.Sprintf("new%d", i)
		blocks := 50
		if blocked && i == 0 {
			blocks = 5000 // the outside head: it fits no pod's room and no pod's budget
		}
		heads = append(heads, Head{Thread: id, ReqID: "r-" + id, Arrival: now.Add(-time.Duration(i) * time.Millisecond),
			Chains: chain(id, blocks)})
	}
	for _, pod := range []string{"p0", "p1", "p2"} {
		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("%s-t%d", pod, i)
			heads = append(heads, Head{Thread: id, ReqID: "r-" + id, Arrival: now.Add(-time.Duration(i) * time.Millisecond),
				Chains: chain(id, 10)})
		}
	}
	return s, heads
}

// benchRealistic is the c192-fidelity state the adversarial review measures: prompts of ~130k tokens
// (2,031 blocks of 64), a pod whose CPU line holds ~41 of them, an 8,000-block GPU budget, 60 warm
// returners (their entries ARE their prompts, so they are warm) and 300 outsiders whose prompts are
// bigger than one seat — the case the strict head-of-line has to survive.
func benchRealistic(b *testing.B, blocked bool) (*Scheduler, []Head) {
	b.Helper()
	const (
		blocks = 2031 // ~130k tokens
		seats  = 8000 // num_gpu_blocks; the 0.95 budget is 7,600 blocks
		tier   = defaultCPUTierTokens
	)
	s := newNCA2Tuned(b, func(c *Config) {
		c.CPUTierTokens = i64(tier)
		c.Quantum = 8
		c.IdleSeconds = 5
	}, []string{"p0", "p1", "p2"})
	for _, pod := range []string{"p0", "p1", "p2"} {
		if err := s.SetNumBlocksFor(pod, 64, seats); err != nil {
			b.Fatal(err)
		}
	}
	now := epoch
	s.mu.Lock()
	for _, pod := range []string{"p0", "p1", "p2"} {
		g := s.groups[pod]
		// 41 entries: the line as the CPU tier really fills at this prompt size
		for i := 0; i < 41; i++ {
			id := fmt.Sprintf("%s-t%d", pod, i)
			g.nca2.line = append(g.nca2.line, nca2Entry{thread: id, tokens: blocks * 64, lastTouch: now})
			if i < 20 { // 60 warm returners in the fleet: admitted, waiting, no seat
				s.nca2.homes[id] = pod
				s.nca2.admitted[id] = true
				s.nca2.ever[id] = true
				s.nca2.inf[id] = 1
			}
		}
	}
	s.mu.Unlock()
	heads := make([]Head, 0, 360)
	for i := 0; i < 300; i++ {
		id := fmt.Sprintf("out%d", i)
		heads = append(heads, Head{Thread: id, ReqID: "r-" + id,
			Arrival: now.Add(-time.Duration(i) * time.Millisecond), Chains: chain(id, blocks)})
	}
	for _, pod := range []string{"p0", "p1", "p2"} {
		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("%s-t%d", pod, i)
			heads = append(heads, Head{Thread: id, ReqID: "r-" + id,
				Arrival: now.Add(-time.Duration(1000+i) * time.Millisecond), Chains: chain(id, blocks)})
		}
	}
	return s, heads
}

// BenchmarkNCA2PassRealistic is one pass at c192 fidelity: 300 outsiders of ~130k tokens, 60 warm
// returners whose entries are their prompts, 41-entry lines.
func BenchmarkNCA2PassRealistic(b *testing.B) {
	b.ReportAllocs()
	var total, worst time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		s, heads := benchRealistic(b, false)
		b.StartTimer()
		start := time.Now()
		s.Pass(heads, epoch.Add(time.Second))
		d := time.Since(start)
		total += d
		if d > worst {
			worst = d
		}
	}
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N), "ns/pass")
	b.ReportMetric(float64(worst.Nanoseconds())/1e6, "worst-ms")
}

// BenchmarkNCA2Pass is one nca2 pass over the heavy state (an admission burst: ~57 admissions, then
// the sweep that ends the pass).
func BenchmarkNCA2Pass(b *testing.B) {
	b.ReportAllocs()
	var total time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		s, heads := benchFixture(b, false)
		b.StartTimer()
		start := time.Now()
		s.Pass(heads, epoch.Add(time.Second))
		total += time.Since(start)
	}
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N), "ns/pass")
}

// BenchmarkNCA2PassBlocked is the steady state: the outside head fits no pod, every returner is
// blocked, and the pass is ONE sweep of three 250-entry lines and the shared queue.
func BenchmarkNCA2PassBlocked(b *testing.B) {
	b.ReportAllocs()
	var total time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		s, heads := benchFixture(b, true)
		b.StartTimer()
		start := time.Now()
		s.Pass(heads, epoch.Add(time.Second))
		total += time.Since(start)
	}
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N), "ns/pass")
}

// BenchmarkBASEPass is the same load through the SHIPPED policy (3 pods, the same 500 heads): the
// reference the nca2 numbers are read against, so "the decision costs X" is a difference and not an
// absolute.
func BenchmarkBASEPass(b *testing.B) {
	b.ReportAllocs()
	var total time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		s, err := NewScheduler(defaultConfig())
		if err != nil {
			b.Fatal(err)
		}
		for _, p := range []string{"p0", "p1", "p2"} {
			if err := s.SetNumBlocksFor(p, 64, 1000); err != nil {
				b.Fatal(err)
			}
		}
		heads := make([]Head, 0, 500)
		for j := 0; j < 500; j++ {
			id := fmt.Sprintf("new%d", j)
			heads = append(heads, Head{Thread: id, ReqID: "r-" + id,
				Arrival: epoch.Add(-time.Duration(j) * time.Millisecond), Chains: chain(id, 50)})
		}
		b.StartTimer()
		start := time.Now()
		s.Pass(heads, epoch.Add(time.Second))
		total += time.Since(start)
	}
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N), "ns/pass")
}
