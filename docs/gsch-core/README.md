# GSCH-CORE — the chain, the emitter patch and the wave configs

*Deliverable of the GSCH-CORE subtraction (`dsai/gsch-core`). The exp-ops tree was in
use by a running campaign when this was written, so the emitter PATCH below is handed
over as text rather than applied underneath the wave.*

## 1. Chain audit — every plugin in a core arm's rendered config

The core arm's plugin list is deliberately SHORT: only what can be shown to be inert, or
what the policy actually reads, survives. "Who dispatches" is the FairnessPolicy's
answer; "where" is the Filter's. Anything that can change either one is either ours or
gone.

| plugin | layer / role | can it influence WHO or WHERE? | verdict |
|---|---|---|---|
| `branch-identity` | RequestHeader (datalayer producer) | **WHO**: it produces the derived branch id, which the policy reads as the member (queue key) | **KEEP** — it IS the identity module. Its key is read but NOT declared in `Consumes()` (the layer-order constraint: a producer in a later layer than the consumer makes the DAG validator refuse startup — the 946f43e3 class). |
| `group-scheduler` (ours) | FlowControl FairnessPolicy **+ scheduling Filter + saturationDetector** | **WHO** (Pick) and **WHERE** (Filter) and the pool gate (Saturation) — all three answer from the same state | **KEEP** — the core. |
| `token-producer` | datalayer producer (TokenizedRequest) | No: it only publishes token counts. But `PreRequest` FAILS the request without it (fail-closed, no tier-B) and the block keys come from it | **KEEP** — the one producer groupsched consumes. Its backend must be the vllm render sidecar (tier-A). |
| `agent-identity` | RequestHeader | Only through the policy's member fallback (session attribute when no branch attribute exists). With branch-identity present it never decides | **DROP** from core arms (kept in code as the fallback for non-branch deployments). |
| `token-load-scorer` | scheduling Scorer | **Not inert in general**: it orders candidates by a metrics-derived util score. With the Filter pinning a seated member to ONE candidate it cannot change the choice — but for a pass-through (unseated / anonymous) it DOES order | **DROP**: a core arm should not have a placement input we cannot prove inert. |
| `inflight-load-producer` | datalayer producer (InFlightLoad) | Feeds the STOCK detector's scrape-lag credit. Our detector does not read it | **DROP**. |
| `utilization-detector` | saturationDetector (+ auto-injected Filter by the loader) | **The pool dispatch gate** at `max(waiting/5, kv/0.80)`, mean-aggregated over endpoints. Also a Filter that blocks "saturated" pods | **DROP from the plugin list; replaced in the detector slot** by the group-scheduler itself (`flowControl.saturationDetector.pluginRef: group`). The loader injects whichever plugin occupies the slot into the profiles as a Filter, so the pin still happens — by our Filter. |
| `session-affinity-filter` (AFFINITY_BLOCK) | scheduling Filter | **WHERE** — the E3 arm's external pin | **DROP**: the core pins with its own Filter. Two pins is two authorities, and they can disagree once per member at birth (the measured E3 caveat). |
| `max-score-picker` (fabricated default) | scheduling Picker | With one candidate it returns that candidate; with several it takes the highest score | **KEEP (fabricated)**: after our Filter a seated dispatch has exactly one candidate, and a pass-through has no member to place. If a future module ever leaves >1 candidate, this picker decides — that is why the scorers above are dropped instead. |
| `single` profile handler (fabricated default) | profile handling | No: a fixed profile choice | **KEEP (fabricated)**. |

Net effect: **branch-identity → group-scheduler (policy · Filter · picker-input ·
detector) → max-score-picker(1 candidate) → single handler.** Nothing else can decide
anything, and no stock piece reads a metric into a decision except the memory-only
backstop, which is OFF.

### DETECTOR IDENTITY (the loader's trap)

`pkg/epp/config/loader/defaults.go:ensureSaturationDetector` FABRICATES the stock
`utilization-detector` when `flowControl.saturationDetector` names nothing, and injects
whatever occupies the slot as a Filter into every profile. So:

* a core arm must set `flowControl.saturationDetector.pluginRef: group`;
* the plugin echoes at startup which of the two states it is in, and announces a
  DETECTOR CONFLICT if another SaturationDetector is configured in the same process
  (exactly one may answer the gate);
* `Saturation` answers from an **atomic** (the flow controller polls it at ~1 kHz and
  the answer must never take the scheduler mutex); the default answer is 0.0 = not
  saturated, because the SEAT BUDGET is the admission control.

## 2. The emitter patch (exp-ops/lib/common.sh + templates/plugins.yaml.tmpl)

Replace the five knob BLOCKS (V2 / WAVE1 / MULTIPOD / FARM / PAT) with ONE, and the
base block with the core's:

```diff
--- a/templates/plugins.yaml.tmpl
+++ b/templates/plugins.yaml.tmpl
@@ plugins:
- - type: agent-identity
-   parameters:
-     additionalSessionHeaders: [@@SESSION_HEADERS_YAML@@]
  @@BLOCK:BRANCH_IDENTITY_BLOCK@@
  - type: token-producer
    parameters:
      modelName: @@BASE_MODEL@@
      vllm:
        url: http://localhost:8000
- - type: token-load-scorer
- - type: inflight-load-producer
- @@BLOCK:AFFINITY_BLOCK@@
- - type: utilization-detector
- @@BLOCK:GATE_PARAMS@@
+ @@BLOCK:AFFINITY_BLOCK@@          # empty for core arms (AFFINITY_FILTER=0)
  - type: group-scheduler
    name: group
    parameters:
      groupBudgetFraction: @@GROUP_BUDGET_FRACTION@@
-     rho: @@RHO@@
-     nMax: @@NMAX@@
+     quantum: @@CORE_QUANTUM@@
      idleSeconds: @@IDLE_SECONDS@@
      ttlSeconds: @@TTL_SECONDS@@
      # the cold-entry cap is DERIVED per engine (the budget's headroom): no key
      inFlightCap: @@INFLIGHT_CAP@@
      blockSizeTokens: @@BLOCK_SIZE_TOKENS@@
- @@BLOCK:V2_KNOBS_BLOCK@@
- @@BLOCK:WAVE1_KNOBS_BLOCK@@
- @@BLOCK:MULTIPOD_KNOBS_BLOCK@@
- @@BLOCK:FARM_KNOBS_BLOCK@@@@PAT_KNOBS_BLOCK@@
+     capacityLivenessSeconds: @@CAPACITY_LIVENESS_SECONDS@@
+     inflightReapSeconds: @@INFLIGHT_REAP_SECONDS@@
+ @@BLOCK:CORE_MODULE_BLOCK@@
 ...
  flowControl:
    saturationDetector:
-     pluginRef: utilization-detector
+     pluginRef: @@CORE_DETECTOR@@   # "group" on a core arm; the stock id on FCFS arms
```

```bash
# exp-ops/lib/common.sh — ONE block replaces V2/WAVE1/MULTIPOD/FARM/PAT.
# The EPP decodes this plugin's parameters with unknown fields REJECTED, so an
# emitted key that Config does not have is a CrashLoop on every arm (the
# residueRelease lesson). Build-side contract: TestConfigEmitterConformance.
if [ "${CHAIN:-}" = "groupsched-core" ]; then
  # CORE_ENTRY_ORDER is "newcomer-first" (default) or "hrrn-bounded" (F5', see §4d):
  # ONE value, and buildModules fails closed on anything else (a typo must never
  # silently measure the anchor twice).
  printf -v CORE_MODULE_BLOCK '    sizing: "%s"\n    entryOrder: "%s"\n    engineChoice: "%s"\n    rotation: "%s"\n    victim: "%s"\n    safety: "%s"\n    executionBudget: "%s"' \
    "${CORE_SIZING:-index}" "${CORE_ENTRY_ORDER:-newcomer-first}" \
    "${CORE_ENGINE_CHOICE:-most-free-budget}" "${CORE_ROTATION:-quantum}" \
    "${CORE_VICTIM:-valve}" "${CORE_SAFETY:-none}" "${CORE_EXECUTION_BUDGET:-none}"
else
  CORE_MODULE_BLOCK=""   # legacy/B2a arms render exactly what they render today
fi
export CORE_MODULE_BLOCK
```

`validate_exp_config` (common.sh) gains the core-only checks:

* `ENGINES>1` is fine (the core is per-engine) — the old "single mode refuses ENGINES>1"
  guard disappears with the pooled mode;
* `CORE_SAFETY=budget-fraction` is refused unless the arm is explicitly a backstop arm
  (it is not part of the wave);
* `CORE_DETECTOR` must be `group` on a core arm and `utilization-detector` on FCFS;
* `AFFINITY_FILTER=0` and `BRANCH_PLUGIN=1` are mandatory on a core arm.

## 3. The collector (the 15 s series)

The wave's BALANCE gate reads the per-engine series every 15 s. A core arm must scrape:

| series | meaning |
|---|---|
| `llm_d_epp_group_seats{engine}` | seats per engine |
| `llm_d_epp_group_waiters_shared` | the ONE shared queue's depth |
| `llm_d_epp_group_budget_blocks{engine}` / `num_gpu_blocks{engine}` / `ne_blocks{engine}` | sizing + the budget's used side |
| `llm_d_epp_group_exits_total{engine,reason}` | exits by reason (`budget-*`, `quantum-complete`, `idle`, `ttl`, `engine-idle`, `exiting*`) |
| `llm_d_epp_group_dispatches_total{engine}` | served per engine |
| `llm_d_epp_group_entry_wait_seconds{engine}` / `reentry_wait_seconds{engine}` | the wait ladder |
| `llm_d_epp_group_detector_pool_score` | what the detector handed the flow controller (0 = open, the default) |
| `llm_d_epp_group_engine_kv_usage{engine}` / `engine_waiting_queue{engine}` | the detector's raw input (REPORTING only on a `none` arm) |
| `llm_d_epp_group_safety_fired{engine}` / `safety_fires_total{engine}` / `safety_fired_seconds_total{engine}` | the backstop's state (all zero on a `none` arm) |
| `llm_d_epp_group_detector_stale_samples_total{engine}` | samples excluded as stale (proves the live read is fresh) |
| `llm_d_epp_group_queue_view_age_seconds` | how old the queue-head EVIDENCE behind the view is, on the decision clock (−1 = nothing observed yet). It grows exactly while Picks stop (a closed pool gate), i.e. the window in which a head can be queued while `waitingSince` says otherwise — the C8 race's size, which is invisible in every other series |
| `llm_d_epp_group_queue_view_untrusted_passes_total` | budget-enforcement passes that ran with that view older than `idleSeconds`, so the C8 guard skipped the head-presence victim classes. 0 on a healthy stack; nonzero means the enforcement passes are re-syncing less often than S |
| `llm_d_epp_group_filter_decisions_total{decision}` | `pinned` vs `passthrough-*` (a run whose decisions are all passthrough is NOT pinned) |
| `llm_d_epp_group_cold_cap_tokens{engine}` | the DERIVED cold-entry stagger cap (the budget's headroom, in tokens). Scorers divide `cold_inflight_tokens` by THIS; the cap moves with the engine and must never be copied out of an arm's config |
| `llm_d_epp_group_cold_valve_held_total{engine}` | heads the valve held back (the budget would have admitted them; they WAIT). Non-zero is the stagger's measured cost; a value under `engine="single"` on a multi-engine stack was a labelling defect fixed in 8397e9aa |
| `llm_d_epp_group_running_blocks{engine}` / `execution_budget_blocks{engine}` | THE EXECUTION BUDGET's two sides (§4g): the engine's RUNNING set (the dedup-exact union of its in-flight requests' chain blocks, from the EPP's own accounting) and the budget it is tested against (`fraction x num_gpu_blocks`). Read them as a ratio — that ratio IS the two-budget arm's mechanism, and `execution_budget_blocks == 0` means no dispatch gate is wired (every arm whose selector is `none`; `running_blocks` is then 0 by construction) |
| `llm_d_epp_group_dispatch_holds_total{engine}` / `dispatch_held_turns_total{engine}` / `dispatch_held_seconds{engine}` | the gate's cost, two ways: decision PASSES that held at least one turn, TURNS held (counted once, at the first denial) and how long each held turn waited before it dispatched |
| `llm_d_epp_group_cold_valve_held_gated_total{engine}` / `gate_charge_hold_seconds{engine}` | THE ONE INTERACTION between the two budgets, measured: cold-entry valve holds whose outstanding charge belonged to a turn the execution budget was HOLDING, and the age of the oldest such charge right now (0 = no admission is blocked by the interaction; a value that only grows IS an admission stall, and the refinement is to suspend a held turn's charge while it is held) |
| `llm_d_epp_group_entry_wait_seconds_count{engine}` | the ADMISSION rate (seats taken per engine): the histogram's own count. Compare it against the tier-inclusive arm's — that is the number the interaction above would move |
| `llm_d_epp_group_event_ring_dropped_total` | SEAT/EXIT events the bounded ring has wrapped past — how much of the window lies BEFORE the dump's `RecentEvents` tail. The stdout event stream is the durable copy; this measures what the dump lost |

The gate line to read on a `none` arm: `detector_pool_score == 0` for the whole window
(the pool gate never closed) and `safety_fired == 0` everywhere (the backstop was off).

## 4. Declared differences vs f0p (each measured separately on the replay)

The core's golden is the ARCHIVE build with f0p's config plus the eight fixes below; it
is reproduced byte for byte (`TestCoreEqualsF0P`). Per-fix counts are on the same 2,600 s
replay, against `golden-f0p-total.txt` (the archive with only tie-break #0):

| # | fix | effect on the replay |
|---|---|---|
| 0 | total "never completed" victim tie-break (member id) | the baseline's own correction: f0p read map iteration order and produced **3 different traces in 4 replays**. |
| 1 | the cold-entry valve never evicts (a valve-held head waits) | **0 lines** at the shipped cap; at a binding cap, f0p took **20 evictions for nothing** (3,220 valve-blocked passes) and served 1,201 vs the core's 1,253 (**+4.3 %**). **Replay: 0 at the shipped cap, 20 at a tiny cap. Live: measured by the anchor** — f0p logged 913,162 valve-blocked passes live; those evictions look like ordinary exits in the banked artifacts, so the live effect is UNKNOWN (bounded by the rotation exits, ~40-45 % of exits). |
| 2 | class-3 victims: unstarted/wanting members sort LAST | **0 lines** (this replay's class-3 set is entirely unstarted); the fix targets f0p's measured 53/54 zero-turn victims. |
| 3 | a stale queue view never justifies an eviction (the C8 race) + Tick re-syncs from the last head snapshot | **14 lines**, +9 class-3 exits, served unchanged (1,231). |
| 4 | mid-flight exits reach the exit hook; dead rotation in-flight branch deleted | **62 extra `EXIT reason=exiting` lines**, 0 decisions changed (accounting only). |

The ACCOUNTING BATCH (fixes 5-7, 2026-11-01) is measured the same way, each fix ALONE on
top of fixes 0-4 (the tree it applies to); the counts are lines differing from the
5,033-line 0-4 trace, and both fix-bearing goldens were regenerated by the recipe:

| # | fix | effect on the replay |
|---|---|---|
| 5 | **an exiting member's blocks stay on the budget's used side until it drains** | **6,378 lines; served 1,231 -> 1,049**. The wave measured the excluded set at **14-18 % of the pool** (p50, per engine) — the budget believed it had that much free while the engine was still pinning it. Also closes a latent enforcement livelock (re-picking an already-exiting in-flight victim freed nothing). |
| 6 | **the cold-entry cap is DERIVED (the budget's headroom) and ENFORCED** | **5,907 lines; served 1,231 -> 1,185**. The cap is `floor((1 - fraction) x num_gpu_blocks) x blockSizeTokens` per engine, the charge is capped at it, and R11's "one always allowed" falls out of the arithmetic. The replay divergence is the replay's SHAPE (a 1,000-block engine gives a 50-block cap where the constant was 2,048); at the campaign shape the derived cap is 2,009 blocks = 128,576 tokens, i.e. the constant it replaces. |
| 7 | **token-exact block counting: the trailing partial block is counted** | **0 lines here** (the replay's driver feeds keys directly and its sizes are whole blocks). Measured on the token-level A/B replay instead: every request with a partial tail gains its block, and the seat index is otherwise token-exact — **Reading B** (`TestSeatIndexBlockCountReadings`, package README §2a). |

All three together: 4,709 lines, served 989 (-19.7 % on this replay's 950-block-budget
engine). The throughput cost is the replay's: on the campaign engine the same fixes put
the corrected occupancy into the preemption zone instead of shrinking a pool, which is what
the wave's own confirmation arms measure (preemptions vs FCFS on the fixed core).

The ENFORCEMENT-CASCADE BATCH (fix 8 and its safety half, 2026-11-01 late) is measured
against the 4,709-line 5-7 tree, and both fix-bearing goldens were regenerated by the
recipe again (Layer C):

| # | fix | effect on the replay |
|---|---|---|
| 8 | **two views of one accounting: an exiting member's blocks are OCCUPIED for admission, and NOT A REASON TO EVICT MORE for enforcement** | **(iii) made marking free.** A pass that marked a mid-flight victim saw the same over-budget number, so it marked the next in-flight member, and the next, on every pass — and passes are every Pick. Live, on the batch-1 anchor (99a47617 vs 88e74f85, identical config, c128): **2,520 mid-flight marks against 87** (29x), bursts of up to 46 members marked in one second, **turns per residency 2.26 -> 0.59**, the rotation exits displaced (quantum-complete 532 -> 44, idle 298 -> 14), re-entry waits to a 220 s p50, the cache destruction behind them (refill share 31.7 % -> 44.5 %): **served −23 %, TTFT p50 2.6 s -> 95 s** (the fix's own success shows in the same arm: **preemptions 11 -> 0**). Replay: **mid-flight marks 103 -> 48** on the default arm, 118 -> 37 on F5′, 102 -> 36 on the tier arm; **rotation exits restored** (quantum-complete 11 -> 25 / 21 -> 54 / 13 -> 34); **dispatches 989 -> 1,182 / 1,299 -> 1,500 / 1,020 -> 1,212 (+15-20 %)**. The pass's victim CLASSES are unchanged — what changes is that a pass is done once the overage is entirely made of members already leaving. |
| 8b | **the release point is reconciled at every decision point** (not only at the settle that happens to reach zero in flight) | **0 lines on every replay arm** — it fires only where a termination was never observed. It closes the leak the audit went looking for: with (iii), a member that became exiting and reached zero in flight WITHOUT a settle kept its blocks charged forever (the reap drops an unobserved termination; a member with a queued head is refused by rotation, by the valve AND by the TTL's idle test). The engine would then look permanently full, `TrySeat` would refuse every seat and marking frees nothing: that engine stops serving, silently. Four unit paths prove it (cancel-with-no-settle, error-before-first-chunk, headless TTL, exiting-with-a-queued-head) — each one FAILS without this half. |

New series (per engine, for the pre-run gate on the next digest):
`llm_d_epp_group_exiting_members`, `llm_d_epp_group_exiting_blocks` (the part of
`ne_blocks` already leaving), `llm_d_epp_group_exiting_oldest_age_seconds` (a value that
only GROWS is a drain that is not happening). Gate: the exiting-drain share of
`exits_total` within ~2x of 88e74f85's ~5 % (so <= ~10 %), and marks-per-dispatch <= 5 %.

## 4a0. THE DIGEST (what the wave runs)

    EPP_IMAGE_TAG=groupsched-88e74f85
    EPP_IMAGE_DIGEST=sha256:16b02e17ed3e2d9022e9a37f94245e3783b538a776223855ba4b4b9287f20925   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:2f62059465efc17efe546890a3ed5576c75e523a33bc7c85858bf85496d88de2

Built from dsai/gsch-core HEAD `88e74f85fb49b9f1a0e23af2d3e6970cdd2c0668` with a clean
tree, `docker buildx --platform linux/amd64` and the Makefile's arg set (TARGETOS,
TARGETARCH, COMMIT_SHA, BUILD_REF), pushed directly so the index carries the build
attestation. Pull-back verification: the linux/amd64 manifest unpacks to an x86-64
statically-linked `/app/epp` whose strings carry
`version.CommitSHA=88e74f85fb49b9f1a0e23af2d3e6970cdd2c0668` and
`version.BuildRef=dsai/gsch-core`, plus the core's own markers
(`CORE policy`, `detector_pool_score`, `cold_valve_held_total`, `waiters_shared`,
`most-free-budget`) and NONE of the removed series
(`pat_eligible_*`, `residue_predicted_ratio`, `engine_gate_closed`, `pooled_endpoints`).

## 4a1. THE SECOND DIGEST — the F5′ build (what B5/B6 run)

    EPP_IMAGE_TAG=groupsched-8397e9aa
    EPP_IMAGE_DIGEST=sha256:6aff12ac8608506d2f53740bda705d0b753f2993991023324466d4179100c846   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:aa0a911d3670337822b55e5c1ed67696cd159bf28cb51c3d5001d697dc904a1c
    attestation manifest  sha256:1bab5f1959b1265a49bdff0220249771631f035c9e70ca5418cf88375fbf3bf8 (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `8397e9aac9471cd67df547dd992a41524d0944d4` with a clean tree,
by the same recipe as §4a0 (`docker buildx build --platform linux/amd64`, `--build-arg
TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core LDFLAGS="-s -w"`,
`--push`). Pull-back verification, same checks as §4a0 plus the markers this build adds: the
linux/amd64 manifest unpacks to a 75.6 MB x86-64 statically-linked `/app/epp` whose build
settings carry `CommitSHA=8397e9aac9471cd67df547dd992a41524d0944d4` and
`BuildRef=dsai/gsch-core`, whose strings carry `hrrn-bounded`, `queue_view_age_seconds` and
`queue_view_untrusted_passes_total` alongside `CORE policy`, `detector_pool_score`,
`cold_valve_held_total`, `waiters_shared`, `most-free-budget`, `newcomer-first`, and NONE of
the removed series (`pat_eligible_*`, `residue_predicted_ratio`, `engine_gate_closed`,
`pooled_endpoints`, `mlfq2`, `rotateWhenFree`).

It contains three commits on top of the wave's build: the F5′ module (`fd542c8d`), its
equivalence golden and docs (`197c658e`), and the two observability fixes (`8397e9aa` —
valve-held attribution, the C8 gauge/counter). An intermediate image of the module commit
alone (`groupsched-197c658e`) was built and is SUPERSEDED: no arm used it, and it lacks the
observability pair the B5/B6 pre-flight reads.

## 4a2. THE THIRD DIGEST — the ACCOUNTING BUG-FIX build (2026-11-01)

    EPP_IMAGE_TAG=groupsched-33e038da
    EPP_IMAGE_DIGEST=sha256:cd73a1a41c3665dd01b06759838197a71418aca4508e08a486ccac367a6509fa   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:46483357da8dba17836a5ae66664db82e61ef22cc418d9e024fc40f24647c8e2
    attestation manifest  sha256:b17d04a1853054583a9de649e8ad7bb4dbe56d24a8b34369f0dffa4cd0945010 (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `33e038da3b016fb745e40824fbd4498b6e9eb9ee` with a clean tree,
by the same recipe as §4a0/§4a1 (`docker buildx build --platform linux/amd64`, `--build-arg
TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core LDFLAGS="-s -w"`,
`--push`). Pull-back verification: the linux/amd64 manifest unpacks to a 75,587,746-byte
x86-64 statically-linked stripped `/app/epp` whose build settings carry
`CommitSHA=33e038da3b016fb745e40824fbd4498b6e9eb9ee` and `BuildRef=dsai/gsch-core`; strings
carry `CORE policy`, `detector_pool_score`, `cold_valve_held_total`, `waiters_shared`,
`most-free-budget`, `newcomer-first`, `hrrn-bounded`, `queue_view_age_seconds`,
`queue_view_untrusted_passes_total` and the two series this build adds
(`cold_cap_tokens`, `event_ring_dropped_total`); strings carry NONE of the removed
series (`pat_eligible_*`, `residue_predicted_ratio`, `engine_gate_closed`,
`pooled_endpoints`, `mlfq2`, `rotateWhenFree`) and no `coldEntryCapTokens` at all — the
config key is gone, not merely defaulted.

What it contains on top of §4a1 (one commit): the three accounting fixes and their
goldens — (iii) an exiting member's blocks stay on the used side until it drains,
(E) the cold-entry cap derived from the budget's headroom and enforced, (C) the
trailing partial block counted. Both fix-bearing goldens were regenerated through the
archive-patch recipe, so the new traces are archive-generated, not self-referential
(package README §2/§2a, recipe §3). Divergence from the previous build is DECLARED:
6,378 / 5,907 / 0 lines per fix alone on the 2,600 s replay, 989 dispatches vs 1,231.

**First live use (ruled): same-digest ANCHORS for the offload solution — core-Q4 with
offload OFF and ON at the load the reviewer picks.** The OFF anchor against
`88e74f85`'s OFF cell quantifies the fixes' live effect (cross-digest, flagged as such):
preemptions, throughput, and the pinned-set invariant (reconstructed pinned set <= pool at
p90 per engine). Nothing launches on this digest before the reviewer has read
OFFLOAD-SIMPLE c64.

**WHICH BUILD EACH ARM RUNS** (ruled 2026-11-01 — one wave, one digest):

| arms | build |
|---|---|
| the MP wave (`core-q4.env`, `core-q8.env`, `core-q4-anchor-n1.env`) | **§4a0** `groupsched-88e74f85` — that is where the wave is defined and where B1 ran. Do NOT re-pin these files (§4a1 exists for a different pair). |
| B5/B6 (`core-q4-8397e9aa.env` ‖ `core-f5p-q4.env`) | **§4a1** `groupsched-8397e9aa`, both arms, same hour — the pair carries its own same-digest Q4 anchor, so the entry order is the only difference between them. |
| the offload anchors and every arm after the accounting build | **§4a2** `groupsched-33e038da` — the fixed core. One digest per wave still holds: the offload cells are its own same-digest OFF/ON pair. |

The new build is decision-identical at `entryOrder: "newcomer-first"` (golden byte-identical
at both commits), so nothing about the comparison depends on which build an arm runs; the
split is about PROVENANCE — a mid-wave switch would blur what the wave measured even when
the decisions are provably the same.

**The two builds are decision-identical for every arm that does not select F5′.**
`88e74f85 → 8397e9aa` changes one production file's behaviour (`modules.go`: the new
`HrrnBounded` + the `EntryCtx` argument); everything else added is inert at
`entryOrder: "newcomer-first"` — a label fix and two new series. Evidence:
`testdata/golden-f0p-bugfixes.txt` is reproduced byte for byte at BOTH commits
(`TestCoreEqualsF0P` was run at each), and the F5′ equivalence golden
(`TestHrrnBoundedEqualsTheArchiveF5P`) is byte-identical too. The wave's configs therefore
STAY on §4a0 and the B5/B6 pair carries this digest on both of its arms.

## 4a3. THE FOURTH DIGEST — the OFFLOAD SOLUTION build (2026-11-02)

    EPP_IMAGE_TAG=groupsched-99a47617
    EPP_IMAGE_DIGEST=sha256:1163763c0afbe97e116bbe29d71a830105c4c73c16f7811304920d769aac3c4b   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:6207d362256d1708cc7f97673fb81d1c83f0449dea4023d3508eac62844522ca
    attestation manifest  sha256:d3e59a6d3785e897015e8ae8d9b416fc001c42213ee136168c1d17cf9b98ffbe (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `99a4761746e73f21fc3c89f45a9ffbc7fae761b1` with a clean tree,
by the same recipe as §4a0/§4a1/§4a2 (`docker buildx build --platform linux/amd64`,
`--build-arg TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core
LDFLAGS="-s -w"`, `--push`). Pull-back verification: the linux/amd64 manifest unpacks to a
75,612,322-byte x86-64 statically-linked stripped `/app/epp` (go1.26.8) whose build settings
carry `CommitSHA=99a4761746e73f21fc3c89f45a9ffbc7fae761b1`, `BuildRef=dsai/gsch-core`,
`CGO_ENABLED=0`, `GOARCH=amd64`; strings carry everything §4a2 carries (`CORE policy`,
`detector_pool_score`, `cold_valve_held_total`, `waiters_shared`, `most-free-budget`,
`newcomer-first`, `hrrn-bounded`, `queue_view_age_seconds`,
`queue_view_untrusted_passes_total`, `cold_cap_tokens`, `event_ring_dropped_total`) PLUS the
four things this build adds: `hrrn-tier`, `cpuTierBytes`, `kvBlockBytes`,
`tier_shadow_blocks`, `tier_shadow_capacity_blocks`, `tier_shadow_pairs_total` and
`SHADOW-VALIDATION`; and NONE of the removed series (`pat_eligible_*`,
`residue_predicted_ratio`, `engine_gate_closed`, `pooled_endpoints`, `mlfq2`,
`rotateWhenFree`), no `coldEntryCapTokens`, no `residueRelease`.

What it contains on top of §4a2 (four commits): the TIER SHADOW, `entryOrder: "hrrn-tier"`
and the SHADOW-VALIDATION metric (`1f6a0ac5`), the three offload arm configs + this
document's §4e + the loader gate (`cd600a66`), the gating of the tier-aware cost to the
module that reads it + the shadow-neutrality pin (`251cdc40`), and pairing COLD returns
(`99a47617`). DECISION EQUIVALENCE with §4a2 is a test, not a claim: with `cpuTierBytes: 0`
the new value reproduces `testdata/golden-f5p-archive.txt` byte for byte, and with a tier
CONFIGURED the default and F5′ arms reproduce their own goldens byte for byte
(`TestTierConfiguredIsShadowNeutralForTheOtherArms`) — so an OFF/ON pair on this digest
differs by the DEPLOYMENT, not by the shadow. The arm's own trace is
`testdata/golden-hrrn-tier.txt` (4,774 lines, 1,020 dispatches).

## 4a4. THE FIFTH DIGEST — the ENFORCEMENT-CASCADE build (2026-11-01 late)

    EPP_IMAGE_TAG=groupsched-b9540fb9
    EPP_IMAGE_DIGEST=sha256:8e9a90ac85cc479f28426f535a174b2b5f429bab365c7889c013011a625d825b   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:c032527af73c36e287f88205baa0702055cb20706ade58c4ae3a216f5caf44ad
    attestation manifest  sha256:122f35ae05a0afddf40439bddaf2e4c4a2c4f390790a221678bba37c522c5590 (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `b9540fb972a52a55a8cec6f70e390b4c25c63d12` with a clean tree,
by the same recipe as §4a0-§4a3 (`docker buildx build --platform linux/amd64`,
`--build-arg TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core
LDFLAGS="-s -w"`, `-f Dockerfile.epp`, `--push`). Pull-back verification: the linux/amd64
manifest unpacks to a 75,624,610-byte x86-64 statically-linked stripped `/app/epp` whose build
settings carry `CommitSHA=b9540fb972a52a55a8cec6f70e390b4c25c63d12`, `BuildRef=dsai/gsch-core`,
`CGO_ENABLED=0`, `GOARCH=amd64`, `GOOS=linux`; strings carry everything §4a3 carries (`CORE
policy`, `detector_pool_score`, `cold_valve_held_total`, `waiters_shared`, `most-free-budget`,
`newcomer-first`, `hrrn-bounded`, `hrrn-tier`, `cpuTierBytes`, `kvBlockBytes`,
`tier_shadow_blocks`, `tier_shadow_capacity_blocks`, `tier_shadow_pairs_total`,
`SHADOW-VALIDATION`, `cold_cap_tokens`, `event_ring_dropped_total`, `gpu-plus-tier`) PLUS the
three series this build adds — `exiting_members`, `exiting_blocks`,
`exiting_oldest_age_seconds` (wire names `llm_d_epp_group_exiting_*`) — and NONE of the
removed series (`pat_eligible_*`, `residue_predicted_ratio`, `engine_gate_closed`,
`pooled_endpoints`, `mlfq2`, `rotateWhenFree`), no `coldEntryCapTokens`, no `residueRelease`.
(Strings read with `grep -a` or `strings -a`: BSD `strings` without `-a` skips Go's rodata and
reports every marker absent — the first pass of this verification did exactly that.)

What it contains on top of §4a3 (ONE commit, `b9540fb9`, on top of the tier-budget head
`d962c5f3`): fix 8 (TWO VIEWS of one accounting — an exiting member's blocks are OCCUPIED for
admission and NOT A REASON TO EVICT MORE for enforcement) and fix 8b (the release point
reconciled at every decision point; safety, 0 golden lines), plus the three gauges above.
This is ONE digest for the whole night: the accounting batch, the tier shadow, hrrn-tier,
gpu-plus-tier and the cascade fix. Every default stays golden-neutral per pair — each arm
still differs from its anchor by exactly its one declared line — and the wave pins ALL of
tonight's arms to it in one operation (the batch-1 re-run pair, the four tb-* arms, and the
core-q4 OFF arm).

Declared difference (§4 rows 8/8b, recipe §4): f0p 4,709 -> 4,871 lines / 989 -> 1,182
dispatches, F5' 5,267 -> 5,465 / 1,299 -> 1,500, tier arm 1,020 -> 1,212; mid-flight marks
103 -> 48 / 118 -> 37 / 102 -> 36 and quantum-complete exits restored 11 -> 25 / 21 -> 54 /
13 -> 34, i.e. the cascade (2,520 marks for 3,104 dispatches, 29x the no-fix arm) is gone.

**PRE-RUN GATE (new, from the three series + the exit mix):** marks-per-dispatch <= 5 % and
the `exiting-drain` share of `exits_total` inside the replay's own band (~24 %, NOT <= 10 % —
10 % is below a healthy replay), and `exiting_oldest_age_seconds` must not climb monotonically
per engine. Baselines: no-fix 2.2 % marks/dispatch and 4.9 % exiting-drain share; cascade
81 % / 85 %; fixed replay 3.0 % / 24 %.

## 4a5. THE SIXTH DIGEST — the EXECUTION BUDGET (2026-11-01, one digest for everything new)

    EPP_IMAGE_TAG=groupsched-17a10e8c
    EPP_IMAGE_DIGEST=sha256:f71edae25dca386ce5ea499bede3d435274e6b7f323fd56d06a3e4665b92db92   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:303d16eb313379a006315a4d277c98e10efb967f44730bb7e285b2230a8f48c6
    attestation manifest  sha256:af57df8d39ca906ccf1d70f99388145c36c28b10ef70d8f52ff39df4544ffd09 (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `17a10e8cd1b75e955d98f3819fb7f6e389853b01` with a clean tree,
by the same recipe as §4a0-§4a4 (`docker buildx build --platform linux/amd64`,
`--build-arg TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core
LDFLAGS="-s -w"`, `-f Dockerfile.epp`, `--push`). Pull-back verification: the linux/amd64
manifest unpacks to a 75,649,186-byte x86-64 statically-linked stripped `/app/epp`
(sha256 `ed183aadba40cb661f636a9049c5d01c556b13f252f6b2b181a6fa339c2ed1dc`) whose build
settings carry `CommitSHA=17a10e8cd1b75e955d98f3819fb7f6e389853b01`,
`BuildRef=dsai/gsch-core`, `CGO_ENABLED=0`, `GOARCH=amd64`, `GOOS=linux`; strings carry
everything §4a4 carries (incl. `gpu-plus-tier`, `hrrn-tier`, `cold_cap_tokens`,
`exiting_*`) PLUS this build's own markers — `execution-budget`, `executionBudget`,
`running_blocks`, `execution_budget_blocks`, `dispatch_holds_total`,
`dispatch_held_turns_total`, `dispatch_held_seconds`, `cold_valve_held_gated_total`,
`gate_charge_hold_seconds` — and NONE of the removed series (`pat_eligible_*`,
`residue_predicted_ratio`, `engine_gate_closed`, `pooled_endpoints`, `mlfq2`,
`rotateWhenFree`), no `coldEntryCapTokens`, no `residueRelease`.

**ONE DIGEST FROM HERE:** everything new runs on it. It carries TWO commits on top of §4a4
(`b9540fb9`): `005eb1dc` (the EXECUTION BUDGET — the DISPATCH module, §4g) and `17a10e8c`
(the interaction MEASURED: held-turn count and duration, the valve share attributable to a
held turn's charge, and the charge's age). Both are default-OFF: the selector's default is
`none`, so a **default-config arm is behaviourally identical to §4a4** — the goldens are
byte-identical (`TestCoreEqualsF0P`, `TestGoldenHashesPinned`, testdata untouched, and
`TestExecutionBudgetOnChangesNothingWhenItNeverBinds` proves the module moves nothing where
the running set always fits). The only thing a default arm gains is the new series, which
read 0.

New series (per engine): `running_blocks`, `execution_budget_blocks` (0 = no gate wired),
`dispatch_holds_total`, `dispatch_held_turns_total`, `dispatch_held_seconds`,
`cold_valve_held_gated_total`, `gate_charge_hold_seconds`.

**PRE-RUN GATE for the two-budget arm** (c64, `core-q4-offload-tierinclusive ‖
core-q4-offload-tierinclusive-execbudget`, same digest): `running_blocks /
execution_budget_blocks <= 1` for the whole window on the exec arm (the gate working);
`cold_valve_held_gated_total` and `gate_charge_hold_seconds` reported, and the admission
rate (`entry_wait_seconds_count`) within ~2x of the tier-inclusive arm's — outside that, the
interaction is what starved admissions and the refinement is §4g's (suspend a held turn's
charge while it is held).

## 4a6. THE SEVENTH DIGEST — the TIER-INCLUSIVE correction (2026-11-02)

    EPP_IMAGE_TAG=groupsched-e78152da
    EPP_IMAGE_DIGEST=sha256:4c44cf26d51017a49fae1376f3bc24fad4db268c9d51d5cc7e419cb81c27f639   (OCI index, keeps the attestation)
    EPP_IMAGE_DIGEST_AMD64=sha256:8362525787f987f3ebd5b3927953c4751b319f2c4571daa7a4a697eecafbdc5a
    attestation manifest  sha256:d57c5fa4e825f3035c6027abd1d33b7315c8cb26c0a2b4342aa39ec50de10850 (unknown/unknown,
                          vnd.docker.reference.digest = the amd64 manifest)

Built from dsai/gsch-core HEAD `e78152da2796b2ef5cfc9a03e12034e567958495` with a CLEAN tree,
by the same recipe as §4a0-§4a5 (`docker buildx build --platform linux/amd64`,
`--build-arg TARGETOS=linux TARGETARCH=amd64 COMMIT_SHA=<head> BUILD_REF=dsai/gsch-core
LDFLAGS="-s -w"`, `-f Dockerfile.epp`, `--push`). Pull-back verification: the linux/amd64
manifest unpacks to a 75,649,186-byte x86-64 statically-linked stripped `/app/epp`
(sha256 `5232f014abafe715af2ae40f6e851f7ab170534f30da870aacf2e767d713cd89`) whose build
settings carry `CommitSHA=e78152da2796b2ef5cfc9a03e12034e567958495`,
`BuildRef=dsai/gsch-core`, `CGO_ENABLED=0`, `GOARCH=amd64`, `GOOS=linux`; strings carry
everything §4a5 carries (`execution-budget`, `hrrn-tier`, `cold_cap_tokens`, `exiting_*`,
`dispatch_held_*`, …) PLUS this build's own marker — **`tier-inclusive`** — and **NONE of the
retired `gpu-plus-tier`** (nor the additive phrase "whole reachable working set"): the summed
value is gone from the code, not just from the configs. Same absent set as §4a5
(`pat_eligible_*`, `residue_predicted_ratio`, `engine_gate_closed`, `pooled_endpoints`,
`mlfq2`, `rotateWhenFree`, `coldEntryCapTokens`, `residueRelease`).

**What it carries:** ONE commit on top of §4a5 (`17a10e8c`) — the tier-inclusive correction
(§4f): `budgetBlocks = floor(fraction x max(num_gpu_blocks, tierBlocks))`, and the additive
`sizing: "gpu-plus-tier"` retired (`buildModules` fails closed on it).
**DEFAULT-NEUTRAL, decision-identical for every arm that does not select the value:** the
default replay is byte-identical (`TestCoreEqualsF0P`, `TestGoldenHashesPinned`, testdata
untouched), and the arms whose `CORE_SIZING` is `index` (`core-q4-offload`, `core-hrrn-tier-q4`,
`core-f5p-q4-offload`, and every non-offload arm) only gain the renamed strings.

**ONE DIGEST FOR THE WHOLE OFFLOAD PROGRAM (the pin operation, verified):** every arm still
on the superseded batch-2 build `groupsched-99a47617` was re-pinned to this one in ONE
operation — `core-q4-offload.env` (the INDEX arm of the c64 triple: its old pin
**overwritten**), `core-q4-offload-tierinclusive.env`, `core-q4-offload-tierinclusive-execbudget.env`,
and the anchor pair `core-hrrn-tier-q4.env` + `core-f5p-q4-offload.env`. All ten shipped
`docs/gsch-core/*.env` files were then checked AGAINST THE REGISTRY in one pass: each
declared `(tag, index, amd64)` triple equals what `docker buildx imagetools inspect` reports
for its tag — five on `groupsched-e78152da`, and the ARCHIVAL arms untouched on their own
digests by their own headers (`core-q4.env` / `core-q8.env` / `core-q4-anchor-n1.env` on
`groupsched-88e74f85`, `core-q4-8397e9aa.env` / `core-f5p-q4.env` on `groupsched-8397e9aa`).
A new way to get a Q4 no-offload cell on THIS build is a digest-named copy (the
`core-q4-8397e9aa.env` convention), never a re-pin of a wave record.

## 4b. Pre-flight gates (every core arm, before its window counts)

* `llm_d_epp_group_detector_pool_score == 0` for the WHOLE window (the pool gate never
  closed) and `safety_fired{engine} == 0` everywhere on a `safety: "none"` arm.
* capacity learned within 120 s on EVERY engine: `num_gpu_blocks{engine} > 0` and
  `capacity_missing{engine} == 0` on the first scrape after the window starts, and
  `capacity_learned_total{via="live-detector"}` present per engine. (The flow
  controller calls the detector on a TICKER, not only when work arrives —
  `pkg/epp/flowcontrol/controller/internal/processor.go:270-271`, and even for an empty
  pool at :486-488 — so the live capacity path runs while the stack is idle. The
  endpoint-event and PreRequest paths remain as fast first attempts.)
* pre-capacity pass-through is counted and must be ~0 over the window:
  `llm_d_epp_group_fcfs_dispatches_total{engine="nocapacity"}`.
* `llm_d_epp_group_filter_decisions_total{decision="pinned"}` ≫ `passthrough-*` (a run
  whose decisions are all passthrough is NOT pinned).
* `llm_d_epp_group_cold_valve_held_total{engine}` is REPORTING: on a cap-binding arm it
  is the stagger's waiting cost, and it must never be accompanied by budget exits it
  caused. **Its `engine` label is the engine the head WAITS for** (the one whose budget
  would have admitted it while its valve refused it, most-free-budget first) — it read
  `engine="single"` on a two-engine stack until 2026-11-01, because the admission loop
  passed an empty engine id.
* `llm_d_epp_group_queue_view_age_seconds` ≤ `idleSeconds` and
  `queue_view_untrusted_passes_total` = 0 for the window: the C8 guard ("a stale queue
  view never justifies an eviction") then never had to fire. A growing age with the
  ticker running means the queue view is not being refreshed; a nonzero pass count means
  victim choices are being made without head evidence — report it, do not chase it.
* run-level safety: preemptions ≤ the FCFS arm's (hard fail if above).
* **TTL exits are reported PER ARM at c32 and c64** (`llm_d_epp_group_exits_total{reason="ttl"}`):
  the silence TTL is inert at c128 (1-6 exits per arm) and the open question is whether
  it matters at low load, where seats are not scarce and a member could sit idle longer.

### Instrumentation rules (apply to the collector and the reducers, not the build)

* Every safety/gate gauge emits a REAL 0/1 in every mode (a `none` arm exports
  `safety_fired = 0` and `detector_pool_score = 0`; an absent series is a question, a
  zero is a fact). Where a series cannot exist, the arm config carries an explicit
  "instrumented" flag rather than relying on a missing value.
* The 15 s poller writes BLANK or `-1` for an absent series — NEVER 0 (a 0 is a
  measurement; a gap is not).
* Reducers filter the shared otel stream BY POD (two arms run concurrently; the stream
  is not per-arm).

## 4c. Hot path

The flow controller polls the detector at ~1 kHz. Measured on this build:
`BenchmarkSaturationNoop` = 7.5 ns/op, 0 B/op, 0 allocs/op;
`BenchmarkSaturationMemoryOnly` = 43 ns/op, 0 B/op, 0 allocs/op (the score is
recomputed at most once per refresh interval, otherwise read from an atomic). The wave's
report carries the EPP arrival-path p99 against B2a f0p's.

## 4d. The F5′ arm (bounded HRRN) — the ONE value that changes

The queue module has a second implementation, selected by `CORE_ENTRY_ORDER=hrrn-bounded`
(rendered as `entryOrder: "hrrn-bounded"` in `CORE_MODULE_BLOCK`; the emitter's block for
that arm is the one in `config_conformance_test.go`). Nothing else in an arm moves: the
F5′-vs-Q4 pair is the same build, the same parameters, one selector value apart —
`core-f5p-q4.env` is `core-q4-8397e9aa.env` (the B5/B6 same-digest Q4 anchor) with
that one line changed, and B5/B6 run `{core-q4-8397e9aa, core-f5p-q4} × {c64, c128}`
same-hour on the §4a1 digest.

The rule is the archive's dd5350b5 F5′: priority = wait ÷ `min(this waiter's cold tokens,
the LOWER-median cold tokens of the current waiters)`, every size floored at one block,
the median computed once per decision pass from the scheduler's own waiter accounting. The
newcomer class stays the outer key (the archive's `isNew`), so inside a class the ratio
replaces age ordering — a big entry ages at the MEDIAN rate, which bounds its takeover
wait at `ratio × median` instead of `ratio × its own size` (the B1b far tail: max entry
wait 3,601 s, 20 stalled sessions).

**What the arm is and is not** (the phrasing that keeps the B5/B6 read honest): F5′ does
NOT let a returning main thread catch up with newcomers — the newcomer class is still
served first, and a long-waiting returner still waits behind every newcomer. What bounded
HRRN changes is INSIDE a class: a big branch's wait is bounded at `ratio × median` instead
of `ratio × its own size`, so small branches pass big ones in minutes rather than in
hours. Read the per-class tables, not the aggregate: an aggregate improvement with the
returner class unchanged is the expected shape, not a contradiction.

Evidence that the core's rule IS the archive's rule: `TestHrrnBoundedEqualsTheArchiveF5P`
reproduces the archive build's F5′ trace byte for byte on the 2,600 s replay (6,089 lines,
1,664 dispatches, no normalisation). Recipe: `docs/gsch-core/golden-f5p-recipe.md`.

## 4e. The offload solution — `entryOrder: "hrrn-tier"` + the TIER SHADOW (2026-11-02)

The queue module's THIRD value, and the only one that reads anything to do with vLLM's CPU
offload tier. The whole design is one page in
`~/.sai-ka-knowledge-bank/explore/ring-window-v2/PREREG-multipod-offload.md`
(§"THE OFFLOAD SOLUTION"); the reasoning is `tier-design-data/OFFLOAD-FIRST-PRINCIPLES.md`.
Three default-OFF pieces, and the `entryOrder` selector is unchanged as a shape — a new
value, not a new knob:

| piece | what it is | where |
|---|---|---|
| 1. TIER SHADOW | per engine, an LRU of block keys mimicking vLLM's CPU tier: capacity `cpuTierBytes ÷ kvBlockBytes` blocks, touch to MRU on dispatch, insert on settle, evict the tail past capacity. Reads NO cache state — the same dictate-don't-observe discipline as the GPU residue model | `tiershadow.go` |
| 2. `hrrn-tier` | F5′ EXACTLY (newcomer class outer, bounded HRRN inside, LOWER-median clamp, the total tie order) with ONE input changed: a waiter's cost `s` = tokens resident in NEITHER the engine's GPU index NOR its tier shadow | `modules.go` (`HrrnTier`, `priorityWith`, `lowerMedianTokens`) |
| 3. SHADOW VALIDATION | per seated RETURN: the predicted resident fraction (taken at seat, before the seat is written into the index) vs the measured `cached/prompt` at the turn's settle, emitted as a `SHADOW-VALIDATION` event with the `fairness_id` and as `llm_d_epp_group_tier_shadow_pairs_total{engine,pred_bin,meas_bin}` (the 11×11 matrix an AUC is computed from) | `scheduler.go`, `metrics.go` |

**Two config values, both DEPLOYMENT FACTS** (they describe the engine, like
`groupBudgetFraction` describes HBM): `cpuTierBytes` (vLLM's `cpu_bytes_to_use`) and
`kvBlockBytes` (one KV block's bytes on this model+TP). Capacity is the division; a
bytes-per-block constant would have been a magic number, so both are given. `0/0` = no
tier: the shadow is nil, nothing is modelled, and `hrrn-tier` **degenerates to F5′**
(`TestHrrnTierNoTierEqualsF5P`: byte-identical to `testdata/golden-f5p-archive.txt`).

**The tier adds no seats.** The tier-aware cost is a PRIORITY input only — the seat
budget, the cold-entry valve and placement keep the GPU-only cost, because vLLM runs only
from HBM and an admission discounted by a restorable block would overbook the engine the
moment the restores land (`TestHrrnTierDoesNotDiscountTheSeatBudget`). That is a property of
the tier-aware COST; §4f is the one value that grows the BUDGET itself, and it keeps this
rule for the per-head cost.

**What the effect is, from the c128 order cut:** a returner with a live CPU copy has a
DEADLINE — its restored share falls 90 % → 30 % → 14 % → 6.4 % across the <1, 1-3, 3-5,
5-10 min bins, and the scheduler's own entry wait for a seat-loss return is 328-420 s, so
the queue is what pushes returners over the cliff. Under `hrrn-tier` such a returner costs
~its new turn's tokens and is seated inside the bin; a COLD returner keeps
`min(full, median)` (F5′'s bound unchanged) and therefore ages, overtaking at the COST
RATIO × the warm wait. Newcomers are untouched: the class key is still the outer
partition. Frozen gates vs core-Q4 ON at c128: returns away 1-10 min restored share
+≥30 pts, matched-work output +5-12 %, newcomer TTFT p90 ≤1.2× anchor, max entry wait
≤1.5× anchor, real starved = 0, **shadow AUC ≥ 0.8**.

**Emitter patch — REQUIRED before any offload arm deploys** (the exp-ops tree was live
when this landed, so it is handed over as text, exactly as §2 was):

```diff
--- a/templates/plugins-core.yaml.tmpl
+++ b/templates/plugins-core.yaml.tmpl
@@
     capacityLivenessSeconds: @@CAPACITY_LIVENESS_SECONDS@@
     inflightReapSeconds: @@INFLIGHT_REAP_SECONDS@@
+    # the CPU offload tier's two deployment facts (0/0 = no tier). The tier SHADOW's
+    # capacity is cpuTierBytes / kvBlockBytes; they only order the queue, they never
+    # admit (see docs/gsch-core/README.md §4e).
+    cpuTierBytes: @@CPU_TIER_BYTES@@
+    kvBlockBytes: @@KV_BLOCK_BYTES@@
 @@BLOCK:CORE_MODULE_BLOCK@@

--- a/lib/common.sh
+++ b/lib/common.sh
@@
   CORE_QUANTUM="${CORE_QUANTUM:-4}"
   CORE_DETECTOR="${CORE_DETECTOR:-group}"
+  # The CPU tier the DECODE is configured with is the same fact the scheduler must model:
+  # default it from the offload deployment so the two can never disagree.
+  CPU_TIER_BYTES="${CPU_TIER_BYTES:-${DECODE_TIER_BYTES:-0}}"
+  KV_BLOCK_BYTES="${KV_BLOCK_BYTES:-0}"
+  export CPU_TIER_BYTES KV_BLOCK_BYTES
   export CORE_QUANTUM CORE_DETECTOR
```

Arms (this directory): `core-q4-offload.env` (default queue, offload ON — the accounting
fix's live confirmation AND the offload anchor) · `core-f5p-q4-offload.env` (tier-blind
comparator; its settings diff to the arm below is the ONE line
`CORE_ENTRY_ORDER=hrrn-bounded`) · `core-hrrn-tier-q4.env` (the arm) ·
`core-q4-offload-tierinclusive.env` (§4f: the same arm with `sizing: tier-inclusive`). Each
keeps `OFFLOAD=0` cells on the same digest for the OFF/ON pair. All five arms here are
pinned to `groupsched-e78152da`, the tier-inclusive build (§4a6, the one re-pin operation).

## 4f. The tier-inclusive seat budget — `sizing: "tier-inclusive"` (2026-11-02)

The SIZING module's SECOND value, and the one that prices the offload deployment's real
capacity:

```
budgetBlocks = floor(groupBudgetFraction x max(num_gpu_blocks, tierBlocks))
```

`tierBlocks` is the CPU tier SHADOW's own capacity (§4e), so the value adds no constant:
the same two deployment facts (`cpuTierBytes`, `kvBlockBytes`) that size the shadow also
size the capacity the budget is a fraction of. On the campaign deployment (40,183 GPU
blocks, a 512 GiB tier = 87,381 blocks = 2.17x the GPU pool) the budget goes from 38,173
blocks to **83,011**.

**THE CORRECTION THIS SECTION CARRIES (the tier-inclusive fix).** This value shipped first
as `sizing: "gpu-plus-tier"`, which sized the budget from the **SUM** of the two capacities.
That is wrong for this deployment: vLLM's native CPU offload is **write-through at compute
time** — every chunk is stored to the tier as it is computed, and nothing is stored on GPU
eviction — so the tier holds a copy of essentially everything HBM holds, and a block on the
GPU is by construction also in the tier. The hierarchy is **inclusive**: the engine's unique
capacity is `max(GPU, tier)`, not the sum. The summed rule gave 121,185 blocks (3.02x the
GPU) where 83,011 exist — **~46 % of that budget was blocks it had counted twice** — and
part of batch 2's overbooking is that number. At HEAD the additive value is **retired**, not
kept beside the new one: a selector that prices capacity the deployment does not have is a
footgun for the next arm that copies a config, and `buildModules` fails closed on it
(`TestGpuPlusTierValueIsRetired`).

**The archive, intact.** Batch 2's arm and its measurement stay here and in git history:
`core-q4-offload-tierbudget.env` with `CORE_SIZING=gpu-plus-tier`, at commit `17a10e8c` /
digest `groupsched-17a10e8c` (`git show 17a10e8c:docs/gsch-core/core-q4-offload-tierbudget.env`).
The result is §4g's *Why, measured* paragraph, which now names both numbers: the arm ran a
3.02x seat budget and the deployment's unique capacity is 2.17x.

**Why the budget and not the order.** The budget sizes the working set an engine may hold,
and what it reserves against is a MISS: with no tier a miss is a full prefill, with one it
is ~1-4 % of a prefill. A seated member whose chain has left HBM is therefore still cheap
to serve, and the engine's reachable capacity is the larger of the two tiers — the CPU tier
at 2.17x the GPU pool here. Sizing from GPU alone reserves for the expensive miss the
deployment no longer pays, and the c64 cells show the cost: stock FCFS gained ~2x from
offload (FCFS-ON 1.390 req/s) where the GPU-sized core gained 1.267.

**What does not move**, both pinned by test:

* the per-head COST stays GPU-only (`TestTierInclusiveCostIsTheGPUOnlyCost`): a head is
  admitted on what it will occupy in HBM, the same rule `hrrn-tier`'s tier-aware cost
  obeys — restorable blocks price priority, never seats;
* the cold-entry valve's CAP stays the GPU headroom, `floor((1 - fraction) x num_gpu_blocks)
  x blockSizeTokens` = 128,576 tokens here (`TestTierInclusiveValveCapStaysGPUHeadroom`): the
  valve staggers GPU allocation bursts, and a tier cannot absorb one.

**`tier = 0` is `sizing: "index"`** arithmetically and on the state machine
(`TestTierInclusiveZeroTierEqualsGPU`), so an arm can carry the value in both deployments and
an OFF cell is not a second policy. **A tier SMALLER than the GPU is the GPU**
(`TestTierInclusiveSmallerTierIsTheGPU`): the tier only ever COVERS HBM, so below the GPU's
capacity it adds no reachable block. The default value stays `"index"`, and the default's
budget does not move when a tier is configured (`TestTierInclusiveDefaultIsUnchanged`); the
golden replay (`TestGolden`) runs on that default, byte for byte.

The arm: `core-q4-offload-tierinclusive.env` is `core-q4-offload.env` with ONE setting
changed (`CORE_SIZING=tier-inclusive`). Its contrast is a same-digest pair, so the batch is
`core-q4-offload ‖ core-q4-offload-tierinclusive` at c64 and the same pair at c128. The
emitter needs no new key: `CORE_SIZING` is already rendered into `CORE_MODULE_BLOCK`, and
`buildModules` accepts the value (conformance test
`TestModuleValuesMatchTheEmitter`, loader gate
`TestGroupSchedulerCoreChainStartsWithTierInclusiveSizing`).

## 4g. THE EXECUTION BUDGET — `executionBudget: "execution-budget"` (2026-11-02)

The DISPATCH module: the second of the TWO BUDGETS that §4f needed and did not have.

```
SEATING  (residency) = floor(fraction x max(num_gpu_blocks, tierBlocks))  §4f, unchanged
DISPATCH (execution) = floor(fraction x  num_gpu_blocks)                  this value
```

**Why, measured.** Batch 2 at c64 (the then-named `tierbudget` arm vs the same-digest
q4-offload): the tier-sized seat budget widened the working set and then let that many
members be seated — and seated members RUN. (Batch 2 ran the ADDITIVE rule: 121,185 blocks,
3.02x the GPU pool. The deployment's unique capacity is 2.17x, so ~46 % of those seats were
blocks that live in HBM AND in the tier at once; §4f is the correction, and the seats this
section reasons about are now 83,011.) A running request's KV needs HBM blocks, which the tier
does not provide, so vLLM queued and preempted what the EPP had just dispatched: **43
preemptions vs 0, TTFT p50 0.9 s -> 17.8 s, newcomer TTFT p90 9.9 s -> 142 s**, with
throughput up 10.8 % (the recompute waste behind it). The tier widens where IDLE chains live,
not how many requests may run at once — and the correction does not remove the reason for this
module: at 83,011 seated blocks against a 38,173-block dispatch budget, seats still outnumber
what HBM can run by 2.17x, so the gate still binds.

**The rule.** A seated member's turn dispatches only if the engine's RUNNING SET — the
dedup-exact union of its in-flight requests' chain blocks — plus what the turn would ADD to it
fits the dispatch budget. Otherwise the turn waits at the EPP, FIFO by turn arrival among the
dispatch-eligible. LIVENESS: one turn is always allowed when nothing is in flight on the
engine. The waiting moved from vLLM's preempting queue into the EPP's ordered one, where it is
measurable (`dispatch_holds_total`).

**Where the numbers come from** — no constant, no metric:

| term | source |
|---|---|
| running set | `BlockIndex.blocksRun`: a refcount over the chain blocks of the requests the scheduler itself dispatched, maintained where an in-flight record changes (dispatch, settle, reap, rebind, exit) and re-derived from those records once per tick |
| what the turn adds | the sizing module's own footprint estimate (its chain + the wire-size growth — the same number the seat budget is charged with), minus the blocks the running set already holds |
| budget | `fraction x num_gpu_blocks`, the deployment's own statement applied to the execution quantity |

**Why a NEW seam and not the SAFETY module.** `SafetyDetector` answers a different question
from a different input class: it is defined over ENDPOINT METRICS (`kv`, `have`), it has two
readers (the per-engine holds and the framework's pool Saturation signal), and it is OFF by
default precisely because a metric gate that misreads its input is worse than no gate. Folding
an accounting gate in would have given one interface two unrelated inputs and two unrelated
readers. The gate is its own module — one question (`may this turn RUN now?`), one default
(`none`), one config value — and the safety module's default is untouched.

**Why it cannot resurrect the deleted per-engine gate.** That gate read a construction-time
metrics CLONE (it froze when dispatch stopped, and latched) and a raw request COUNT
(`waiting >= 5`). This one reads the EPP's own block accounting, per engine, by construction:
no scrape, no staleness rule, no clone, and nothing pooled across engines. It also never
touches placement or the seats — it excludes one turn for one pass.

**What does not move** (each pinned by test): the seat budget, the per-head cost, the
cold-entry valve, the queue order, placement, rotation. The default value is `none`, the
default config's replay is byte-identical (`TestExecutionBudgetDefaultIsGoldenNeutral`), and
enabling the module where the running set always fits changes nothing in the whole 2,600 s
replay (`TestExecutionBudgetOnChangesNothingWhenItNeverBinds`). With the selector at `none`
NOTHING is maintained — no key scan per dispatch, no per-tick rebuild, no cache — so a
comparator arm's CPU is identical to the build before this module existed.

**ONE interaction, stated AND MEASURED:** a HELD turn keeps its seat-time cold-entry charge
(it will prefill when it runs), and the derived valve admits one outstanding prefill per
engine — so a held turn occupies that slot while it waits. It is released the moment the turn
dispatches (which the FIFO rule and the draining running set guarantee). The interaction is
reportable from the digest, not inferred afterwards:

| series | what it says |
|---|---|
| `dispatch_held_turns_total{engine}`, `dispatch_held_seconds{engine}` | how many turns the execution budget held, and for how long each waited before it ran (a clock started at the first denial and booked at dispatch) |
| `cold_valve_held_gated_total{engine}` | how much of the cold-entry valve's blocking is attributable to a HELD turn's charge — the interaction's size |
| `gate_charge_hold_seconds{engine}` | the age of the oldest such charge right now; a value that only grows = admissions are stalled by this and nothing else |
| `entry_wait_seconds_count{engine}` | the ADMISSION rate (seats taken), the number to compare against the tier-inclusive arm |

If those say admissions starve, the refinement is: SUSPEND a held turn's cold charge while it
is held and re-charge it at dispatch. None of this changes a decision, so every arm's trace is
untouched by it.

The arm: `core-q4-offload-tierinclusive-execbudget.env` is `core-q4-offload-tierinclusive.env`
with ONE setting changed (`CORE_EXECUTION_BUDGET=execution-budget`), so the pair is
`core-q4-offload-tierinclusive ‖ core-q4-offload-tierinclusive-execbudget` on ONE digest at
c64.
The emitter needs ONE new key in `CORE_MODULE_BLOCK` (§2's printf) and ONE new enum case
(`CORE_EXECUTION_BUDGET=none|execution-budget`); build-side contract
`TestConfigEmitterConformance` + `TestModuleValuesMatchTheEmitter`, loader gate
`TestGroupSchedulerCoreChainStartsWithTheExecutionBudget`.

## 5. The wave (for the record)

`2 pods × {FCFS, core-Q4, core-Q8} × {c64, c128, c32}`, 45 min each, batched
c64 → c128 → c32, plus **core-Q4 at N=1, c128, 45 min** as the matched anchor against
B2a f0p's first 45 minutes. Configs: `core-q4.env`, `core-q8.env`,
`core-q4-anchor-n1.env` (this directory).

Then **B5/B6**: `{core-q4-8397e9aa, core-f5p-q4} × {c64, c128}` on the §4a1 digest, each
pair same-hour — the entry-order question, one selector apart, on a digest of its own (the
wave above keeps §4a0). Scored against the amended
matched-work predictions (matched tokens +2-6 %, entry wait −10-15 %, newcomer p90 ≤
anchor/3, per-thread goodput p10 +25-40 % at c128, starved = 0, max EW ≤ 1.5×), with
tok/s and records reported but labelled "includes mix", and the path-flip and
re-entry-TTFT observables read in both directions.
