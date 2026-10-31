# How `testdata/golden-f5p-archive.txt` was generated

*The F5′ equivalence golden (`TestHrrnBoundedEqualsTheArchiveF5P`). Written so the trace
can be REBUILT and re-checked, not just trusted: every step below was run to produce the
committed bytes, and step 3 is the check that makes the recipe auditable — the same
patched archive regenerates the existing `golden-f0p-bugfixes.txt` byte for byte, so the
patch used here is the one the other goldens came from.*

**Two patch layers.** Layer A (`#0`-`#4`) is the original four-fix patch and is unchanged.
Layer B (`#5`-`#7`, §3) is the ACCOUNTING BATCH: three more declared fixes, applied on top
of Layer A. Both fix-bearing goldens were regenerated through BOTH layers, and the archive
reproduces them byte for byte — so the traces the core is pinned to are still traces an
independent implementation produced, not the core's own output.

Current pins: `golden-f0p-bugfixes.txt` 4,709 lines / 989 dispatches ·
`golden-f5p-archive.txt` 5,267 lines / 1,299 dispatches.

## The steps

1. **The archive tree at the F5′ commit**, in a scratch worktree (never in the archive's
   own checkout, which the campaign was using):

   ```bash
   cd ~/Desktop/llm-d/workload-aware/builds/epp-otel-fc
   git worktree add /tmp/f5p-arch dd5350b5     # "F5' (bounded HRRN ...)"
   ```

2. **Apply Layer A — the four fixes `#0`-`#4`** (`README.md` §2 / `docs/gsch-core/README.md`
   §4) to that tree. The archive is one monolithic scheduler, so the fixes are not the
   core's module code: the patch below is the minimal archive-side form of each. It is one
   file (`pkg/epp/framework/plugins/groupsched/scheduler.go`, +95/−18). Layer B (§3, the
   accounting batch) applies on top of it.

3. **Validate the patch before using it**: run the archive's own golden driver at f0p's
   rendered config and compare with the committed `golden-f0p-bugfixes.txt`. Measured:
   **identical, 0 differing lines**. A patch that does not pass this check is not this
   recipe.

4. **Generate the F5′ trace**: the same patched tree, the same driver, one value changed —
   `entryOrder: "hrrn-bounded"` — written to `testdata/golden-f5p-archive.txt`. Measured:
   6,089 lines, 1,664 dispatches, deterministic across runs.

   `f0pCfg()` is f0p's rendered config in archive terms — the archive's defaults (valve
   `on`, `victimOrder` legacy, `windowMode` last4, `gateMode` detector, member-stable
   binding, every F/PAT/residue knob off) with the wave's rotation and topology:

   ```go
   c := defaultConfig()
   c.RotationMode = "quantum"
   c.QuantumBusyPeriods = 4        // Q4
   c.EngineGroups = "per-engine"
   c.EntryOrder = "hrrn-bounded"   // the ONE value this golden exists for
   trace := runGoldenScenarioCfg(t, 2600, c, nil)   // the archive's own driver
   ```

5. **Compare**: `TestHrrnBoundedEqualsTheArchiveF5P` replays the core with
   `entryOrder: "hrrn-bounded"` and requires byte equality — **no normalisation of any
   kind**. Measured: identical, 0 differing lines.

## The archive-side patch (step 2)

Each hunk is tagged with the fix it is (`#0`–`#4` in the README's table). Apply with
`git apply` in the worktree.

```diff
--- a/pkg/epp/framework/plugins/groupsched/scheduler.go
+++ b/pkg/epp/framework/plugins/groupsched/scheduler.go
@@ Scheduler struct @@
 	gWaiters map[string]*waiter
+
+	// queueViewAt is when the queue-head snapshot was last reconciled
+	// (syncWaitersLocked) and lastHeads is that snapshot (FIX 3): Tick re-syncs from it
+	// so a head that arrived since the last Pick cannot leave its member looking
+	// headless, and "no queued head" is only evidence while the view is fresh.
+	queueViewAt time.Time
+	lastHeads   []Head
 
@@ syncWaitersLocked @@
 func (s *Scheduler) syncWaitersLocked(heads []Head, now time.Time) {
+	s.queueViewAt = now
+	s.lastHeads = heads
 	present := map[string]time.Time{}

@@ silentLocked (FIX 3) @@
 func (s *Scheduler) silentLocked(m *member, now time.Time) bool {
-	if !m.waitingSince.IsZero() {
+	if !s.headlessLocked(m, now) {
 		return false
 	}
 	return now.Sub(m.lastActivity) > time.Duration(s.cfg.IdleSeconds*float64(time.Second))
 }
+
+// headlessLocked reports "this member has no queued head" as EVIDENCE rather than as a
+// guess (FIX 3, the C8 race): a queue view older than S cannot tell a busy member from
+// an idle one, so the ABSENCE is not evidence.
+func (s *Scheduler) headlessLocked(m *member, now time.Time) bool {
+	if !m.waitingSince.IsZero() {
+		return false
+	}
+	return s.queueViewTrustedLocked(now)
+}
+
+func (s *Scheduler) queueViewTrustedLocked(now time.Time) bool {
+	if s.queueViewAt.IsZero() {
+		return false
+	}
+	return now.Sub(s.queueViewAt) <= time.Duration(s.cfg.IdleSeconds*float64(time.Second))
+}

@@ rotateEngineLocked (FIX 1 + FIX 4's dead-code half) @@
-		if ne+estBlocks <= int64(g.budgetBlocks) && !s.coldAllows(g, head) && s.onValveBlockedEviction != nil {
-			s.onValveBlockedEviction(g.id)
+		if ne+estBlocks <= int64(g.budgetBlocks) {
+			// the BUDGET would have taken this head; only the cold-entry valve is holding
+			// it. Eviction frees budget the head does not need and never opens the valve,
+			// so the head WAITS for the prefill slot.
+			if s.onValveBlockedEviction != nil {
+				s.onValveBlockedEviction(g.id)
+			}
+			*seatsP = seats
+			return
 		}
 		if reason, victim := s.rotationVictimLocked(g, now); victim != nil {
-			if victim.inflight > 0 {
-				g.index.MarkExitingLocked(victim.id)
-				s.countExitLocked(g, "exiting")
-				*seatsP = seats
-				return
-			}
+			// (this branch was UNREACHABLE: the tenure rule refuses in-flight members)
 			s.exitLocked(g, victim, reason, now)

@@ olderCompletion (FIX 0) @@
 func olderCompletion(a, b *member) bool {
-	if a.lastCompletion.IsZero() {
-		return false
+	az, bz := a.lastCompletion.IsZero(), b.lastCompletion.IsZero()
+	if az != bz {
+		return bz // a real completion is the cheaper victim
 	}
-	if b.lastCompletion.IsZero() {
-		return true
+	if az {
+		return a.id < b.id // TOTAL tie-break among members that never completed
 	}
 	return a.lastCompletion.Before(b.lastCompletion) ||
 		(a.lastCompletion.Equal(b.lastCompletion) && a.id < b.id)
 }
+
+// class3Less is the mid-quantum victim order (FIX 2): UNSTARTED or WANTING members sort
+// last, then the oldest last completion, then the id.
+func class3Less(a, b *member) bool {
+	aRaw := a.resTurns == 0 || !a.waitingSince.IsZero()
+	bRaw := b.resTurns == 0 || !b.waitingSince.IsZero()
+	if aRaw != bRaw {
+		return !aRaw
+	}
+	return olderCompletion(a, b)
+}

@@ budgetVictimLocked, class 2 + the untrusted-view guard (FIX 3) @@
-			if m.waitingSince.IsZero() || s.silentLocked(m, now) {
+			if s.headlessLocked(m, now) || s.silentLocked(m, now) {
@@
 		if best != nil {
 			return "budget-noqueue", best
 		}
+		if !s.queueViewTrustedLocked(now) {
+			// no trustworthy view of who has a queued head: only members already on their
+			// way out can be named; the next pass (at most one tick away) decides.
+			for _, m := range g.members {
+				if !g.index.ExitingLocked(m.id) {
+					continue
+				}
+				if best == nil || olderCompletion(m, best) {
+					best = m
+				}
+			}
+			if best != nil {
+				return "budget-exiting", best
+			}
+			return "", nil
+		}

@@ budgetVictimLocked, class 3 (FIX 2) @@
-			if best == nil || olderCompletion(m, best) {
+			if best == nil || class3Less(m, best) {
 				best = m
 			}

@@ enforceBudgetLocked, mid-flight victim (FIX 4) @@
 		if victim.inflight > 0 {
 			g.index.MarkExitingLocked(victim.id)
 			s.countExitLocked(g, "exiting")
+			// the member is on its way out NOW, so the event series has to see it
+			// (llm_d_epp_group_exits_total{reason="exiting"} was permanently 0).
+			if s.onExit != nil {
+				s.onExit(engineLabel(g.id), "exiting", victim.resTurns)
+			}
 			continue
 		}

@@ Tick (FIX 3) @@
 	s.gcSilentRecordsLocked(now)
+	// Tick calls the admission loop WITHOUT a queue sync, so a head that arrived since
+	// the last Pick left its member looking headless. Re-sync from the LAST snapshot.
+	s.syncWaitersLocked(s.lastHeads, now)
 	s.rotateLocked(now)
```

## 3. Layer B — the accounting batch (`#5`-`#7`)

Applied to the Layer-A tree with `git apply`. Measured on that tree, one fix at a time
(`git checkout` back to Layer A between them), on the same 2,600 s replay:

| # | fix | lines changed vs Layer A | served |
|---|---|---|---|
| — | Layer A (the previous golden) | — | 1,231 |
| 5 | an exiting member's blocks stay on the used side until it drains | **6,378** | 1,049 |
| 6 | the cold-entry cap derived from the budget's headroom and enforced | **5,907** | 1,185 |
| 7 | token-exact block counting (the trailing partial block) | **0** | 1,231 |
| 5+6+7 | the build | **5,948** | 989 |

Fix 7 is 0 on this replay by construction: the driver feeds `Dispatched` a key slice
directly (`keys(n, seed)`), so `HashTokens` is never on the path, and every wire size is a
whole number of blocks, so `floor` and `ceil` agree. Its effect is measured where it lives —
the token-level replay `TestSeatIndexBlockCountReadings` (package README §2a), where every
request with a partial tail gains exactly the block the engine allocates.

```diff
--- a/pkg/epp/framework/plugins/groupsched/scheduler.go
+++ b/pkg/epp/framework/plugins/groupsched/scheduler.go
@@ Config @@
-	ColdEntryCapTokens  int64   `json:"coldEntryCapTokens"`
 	InFlightCap         int     `json:"inFlightCap"`
@@ defaultConfig @@
-		ColdEntryCapTokens:      131072,
@@ validate @@
-	if c.ColdEntryCapTokens <= 0 || c.InFlightCap <= 0 {
-		return fmt.Errorf("coldEntryCapTokens and inFlightCap must be > 0")
+	if c.InFlightCap <= 0 {
+		return fmt.Errorf("inFlightCap must be > 0")
 	}
@@ HashTokens (FIX 7) @@
-	n := len(tokens) / blockSizeTokens
+	// FIX 7: the TRAILING PARTIAL BLOCK IS COUNTED (the engine allocates it).
+	n := (len(tokens) + blockSizeTokens - 1) / blockSizeTokens
@@ HashTokens loop (FIX 7) @@
-		for j := i * blockSizeTokens; j < (i+1)*blockSizeTokens; j++ {
+		end := (i + 1) * blockSizeTokens
+		if end > len(tokens) {
+			end = len(tokens)
+		}
+		for j := i * blockSizeTokens; j < end; j++ {
@@ estimateCostLocked (FIX 7) @@
-	return known + delta/4/int64(s.blockSize)
+	// FIX 7: rounded UP — a partial block still occupies a block.
+	blockBytes := 4 * int64(s.blockSize)
+	return known + (delta+blockBytes-1)/blockBytes
@@ DispatchedOn cold refinement (FIX 6) @@
-			m.coldEntry = int64(len(keys)) * int64(s.blockSize)
+			m.coldEntry = min(int64(len(keys))*int64(s.blockSize),
+				g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
@@ coldAllows (FIX 6) @@
+// coldCapTokens is THIS engine's cold-entry stagger cap: the seat budget's
+// headroom, floor((1 - fraction) x num_gpu_blocks) blocks x blockSize.
+func (g *engineGroup) coldCapTokens(fraction float64, blockSize int) int64 {
+	if g.numBlocks <= 0 || blockSize <= 0 {
+		return 0
+	}
+	return int64((1-fraction)*float64(g.numBlocks)) * int64(blockSize)
+}
+
 func (s *Scheduler) coldAllows(g *engineGroup, w *waiter) bool {
 	if s.cfg.ValveMode == "off" {
 		return true
 	}
-	return g.coldInFlight == 0 || g.coldInFlight+w.coldTokens <= s.cfg.ColdEntryCapTokens
+	return s.coldAllowsTokens(g, w.coldTokens)
+}
+
+// coldAllowsTokens: the charge a seat registers is min(cost, cap), so this is
+// "is the engine still inside its headroom after this entry?".
+func (s *Scheduler) coldAllowsTokens(g *engineGroup, tokens int64) bool {
+	cap := g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize)
+	if cap <= 0 {
+		return true
+	}
+	return g.coldInFlight+min(tokens, cap) <= cap
 }
@@ seatLocked (FIX 6) @@
-	// B6: ALL entries carry a cold charge (tokens, R1)
-	m.coldEntry = w.coldTokens
+	// B6/FIX 6: ALL entries carry a cold charge, CAPPED at this engine's cap.
+	m.coldEntry = min(w.coldTokens, g.coldCapTokens(s.cfg.GroupBudgetFraction, s.blockSize))
 	g.coldInFlight += m.coldEntry
@@ enforceBudgetLocked (FIX 5) @@
 		if victim.inflight > 0 {
+			if g.index.ExitingLocked(victim.id) {
+				// FIX 5: the mark frees nothing now, so the pass is done
+				return
+			}
 			g.index.MarkExitingLocked(victim.id)
 			s.countExitLocked(g, "exiting")
```

```diff
--- a/pkg/epp/framework/plugins/groupsched/blockindex.go
+++ b/pkg/epp/framework/plugins/groupsched/blockindex.go
@@ releaseLocked (FIX 5) @@
-// releaseLocked removes s's set from the all-seated counter and (if
-// countNE) from the budget counter. ...
-func (b *BlockIndex) releaseLocked(s *sessionBlocks, countNE bool) {
+// releaseLocked removes s's set from both counters (FIX 5).
+func (b *BlockIndex) releaseLocked(s *sessionBlocks) {
 	b.version++
 	for h := range s.set() {
 		b.dec(b.blocks, h)
-		if countNE {
-			b.dec(b.blocksNE, h)
-		}
+		b.dec(b.blocksNE, h)
 	}
 }
@@ MarkExiting (FIX 5) @@
 func (b *BlockIndex) MarkExiting(id string) {
 	b.mu.Lock()
 	defer b.mu.Unlock()
+	b.markExitingLocked(id)
+}
+
+func (b *BlockIndex) markExitingLocked(id string) {
 	s, ok := b.sessions[id]
 	if !ok || s.exiting { return }
 	if _, isSeated := b.seated[id]; !isSeated { return }
-	b.version++                                  // R4: the exiting write
 	s.exiting = true
-	for h := range s.set() {
-		b.dec(b.blocksNE, h)
-	}
 }
@@ applyDeltaLocked (FIX 5) @@
-	countNE := !s.exiting
+	countNE := true // FIX 5: an exiting member's growth is pinned like anyone else's
@@ checkInvariantLocked (FIX 5) @@
 			want[h]++
-			if !s.exiting {
-				wantNE[h]++
-			}
+			wantNE[h]++ // FIX 5: exiting members are charged until they drain
@@ MarkExitingLocked (FIX 5) @@
-func (b *BlockIndex) MarkExitingLocked(id string) { ...the same body... }
+func (b *BlockIndex) MarkExitingLocked(id string) { b.markExitingLocked(id) }
```

The archive's own test files reference the removed `Config.ColdEntryCapTokens`; those four
lines are deleted in the worktree only so the package COMPILES (`knobs_test.go`,
`cp4_test.go`, `multipod_test.go`, `scheduler_test.go`). The archive-side unit suite is not
maintained here — the golden DRIVER is what the recipe uses, and it is unchanged.

## 4. Regenerating the traces (steps)

```bash
cd ~/Desktop/llm-d/workload-aware/builds/epp-otel-fc
git worktree add /tmp/fixarch dd5350b5
cd /tmp/fixarch
# 1. Layer A (the §2 patch below), then check it reproduces the OLD golden:
#    trace == golden-f0p-bugfixes.txt as committed before the accounting batch
# 2. Layer B (§3 above), then
cat > pkg/epp/framework/plugins/groupsched/zzgen_test.go <<'EOF'
package groupsched
import ("os";"testing")
func TestZZGenTrace(t *testing.T) {
	out := os.Getenv("TRACE_OUT")
	if out == "" { t.Skip("no TRACE_OUT") }
	c := defaultConfig()
	c.RotationMode = "quantum"; c.QuantumBusyPeriods = 4; c.EngineGroups = "per-engine"
	if os.Getenv("F5P") == "1" { c.EntryOrder = "hrrn-bounded" }
	if err := os.WriteFile(out, []byte(runGoldenScenarioCfg(t, 2600, c, nil)), 0o644); err != nil { t.Fatal(err) }
}
EOF
TRACE_OUT=/tmp/g-f0p.txt go test ./pkg/epp/framework/plugins/groupsched/ -run TestZZGenTrace
F5P=1 TRACE_OUT=/tmp/g-f5p.txt go test ./pkg/epp/framework/plugins/groupsched/ -run TestZZGenTrace
# 3. copy into testdata/, update SHA256SUMS, run the two equality tests
```

Measured: both traces are deterministic across runs (0 differing lines), and the core
reproduces both **byte for byte** on the first try (`TestCoreEqualsF0P`,
`TestHrrnBoundedEqualsTheArchiveF5P`) — which is the point of the recipe: the fixes, not
the core's implementation, decide the trace.

## What the replay shows (a trace check, not a cluster claim)

Same replay, all seven fixes, one value changed (measured from the two committed traces):

| arm | served | exits `budget-idle` / `budget-noqueue` / `engine-idle` / `exiting` | max entry wait |
|---|---|---|---|
| `newcomer-first` (`golden-f0p-bugfixes.txt`) | 989 | 17 / 11 / 1 / 103 | 20.0 s |
| `hrrn-bounded` (`golden-f5p-archive.txt`) | 1,299 | 13 / 5 / 0 / 118 | 26.0 s |

Two things worth noting, neither of them a result: the ratio order serves 31 % more
requests on this scenario, and the `engine-idle` evictions (the livelock fallback that
fires when an engine has nothing in flight and nothing dispatchable while a waiter is
blocked) disappear entirely under F5′ — with a ratio order the queue is never blocked by
one big head. (Before the accounting batch these were 1,231 / 1,664 served — +35 %; the
fixes cost this replay's small engine more throughput than the ratio order gains, and net
of both it is still F5′ ahead.) This scenario is NOT the B1b load shape (its waits are seconds, not
minutes), so it says nothing about the far tail: **B5/B6 on the cluster measure that**
(c64 and c128, core-Q4 ‖ core-F5′-Q4). What the replay proves is narrower and exact: the
core's rule IS the archive's rule, decision for decision.
