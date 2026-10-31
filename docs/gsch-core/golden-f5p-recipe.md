# How `testdata/golden-f5p-archive.txt` was generated

*The F5′ equivalence golden (`TestHrrnBoundedEqualsTheArchiveF5P`). Written so the trace
can be REBUILT and re-checked, not just trusted: every step below was run to produce the
committed bytes, and step 3 is the check that makes the recipe auditable — the same
patched archive regenerates the existing `golden-f0p-bugfixes.txt` byte for byte, so the
patch used here is the one the other goldens came from.*

## The steps

1. **The archive tree at the F5′ commit**, in a scratch worktree (never in the archive's
   own checkout, which the campaign was using):

   ```bash
   cd ~/Desktop/llm-d/workload-aware/builds/epp-otel-fc
   git worktree add /tmp/f5p-arch dd5350b5     # "F5' (bounded HRRN ...)"
   ```

2. **Apply the four declared fixes** (`README.md` §2 / `docs/gsch-core/README.md` §4) to
   that tree. The archive is one monolithic scheduler, so the fixes are not the core's
   module code: the patch below is the minimal archive-side form of each. It is one file
   (`pkg/epp/framework/plugins/groupsched/scheduler.go`, +95/−18).

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

## What the replay shows (a trace check, not a cluster claim)

Same replay, same four fixes, one value changed (measured from the two committed traces):

| arm | served | exits `budget-idle` / `budget-noqueue` / `engine-idle` / `exiting` | max entry wait |
|---|---|---|---|
| `newcomer-first` (`golden-f0p-bugfixes.txt`) | 1,231 | 41 / 20 / 21 / 62 | 20.0 s |
| `hrrn-bounded` (`golden-f5p-archive.txt`) | 1,664 | 35 / 61 / 0 / 141 | 17.0 s |

Two things worth noting, neither of them a result: the ratio order serves 35 % more
requests on this scenario, and the `engine-idle` evictions (the livelock fallback that
fires when an engine has nothing in flight and nothing dispatchable while a waiter is
blocked) disappear entirely under F5′ — with a ratio order the queue is never blocked by
one big head. This scenario is NOT the B1b load shape (its waits are seconds, not
minutes), so it says nothing about the far tail: **B5/B6 on the cluster measure that**
(c64 and c128, core-Q4 ‖ core-F5′-Q4). What the replay proves is narrower and exact: the
core's rule IS the archive's rule, decision for decision.
