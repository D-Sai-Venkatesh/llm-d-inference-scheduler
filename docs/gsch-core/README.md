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
      coldEntryCapTokens: @@COLD_ENTRY_CAP_TOKENS@@
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
  printf -v CORE_MODULE_BLOCK '    sizing: "%s"\n    entryOrder: "%s"\n    engineChoice: "%s"\n    rotation: "%s"\n    victim: "%s"\n    safety: "%s"' \
    "${CORE_SIZING:-index}" "${CORE_ENTRY_ORDER:-newcomer-first}" \
    "${CORE_ENGINE_CHOICE:-most-free-budget}" "${CORE_ROTATION:-quantum}" \
    "${CORE_VICTIM:-valve}" "${CORE_SAFETY:-none}"
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
| `llm_d_epp_group_filter_decisions_total{decision}` | `pinned` vs `passthrough-*` (a run whose decisions are all passthrough is NOT pinned) |

The gate line to read on a `none` arm: `detector_pool_score == 0` for the whole window
(the pool gate never closed) and `safety_fired == 0` everywhere (the backstop was off).

## 4. Declared differences vs f0p (each measured separately on the replay)

The core's golden is the ARCHIVE build with f0p's config plus the four fixes below; it
is reproduced byte for byte (`TestCoreEqualsF0P`). Per-fix counts are on the same 2,600 s
replay, against `golden-f0p-total.txt` (the archive with only tie-break #0):

| # | fix | effect on the replay |
|---|---|---|
| 0 | total "never completed" victim tie-break (member id) | the baseline's own correction: f0p read map iteration order and produced **3 different traces in 4 replays**. |
| 1 | the cold-entry valve never evicts (a valve-held head waits) | **0 lines** at the shipped cap; at a binding cap, f0p took **20 evictions for nothing** (3,220 valve-blocked passes) and served 1,201 vs the core's 1,253 (**+4.3 %**). **Replay: 0 at the shipped cap, 20 at a tiny cap. Live: measured by the anchor** — f0p logged 913,162 valve-blocked passes live; those evictions look like ordinary exits in the banked artifacts, so the live effect is UNKNOWN (bounded by the rotation exits, ~40-45 % of exits). |
| 2 | class-3 victims: unstarted/wanting members sort LAST | **0 lines** (this replay's class-3 set is entirely unstarted); the fix targets f0p's measured 53/54 zero-turn victims. |
| 3 | a stale queue view never justifies an eviction (the C8 race) + Tick re-syncs from the last head snapshot | **14 lines**, +9 class-3 exits, served unchanged (1,231). |
| 4 | mid-flight exits reach the exit hook; dead rotation in-flight branch deleted | **62 extra `EXIT reason=exiting` lines**, 0 decisions changed (accounting only). |

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
  caused.
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
F5′-vs-Q4 pair is the same build, the same parameters, one selector value apart.

The rule is the archive's dd5350b5 F5′: priority = wait ÷ `min(this waiter's cold tokens,
the LOWER-median cold tokens of the current waiters)`, every size floored at one block,
the median computed once per decision pass from the scheduler's own waiter accounting. The
newcomer class stays the outer key (the archive's `isNew`), so inside a class the ratio
replaces age ordering — a big entry ages at the MEDIAN rate, which bounds its takeover
wait at `ratio × median` instead of `ratio × its own size` (the B1b far tail: max entry
wait 3,601 s, 20 stalled sessions).

Evidence that the core's rule IS the archive's rule: `TestHrrnBoundedEqualsTheArchiveF5P`
reproduces the archive build's F5′ trace byte for byte on the 2,600 s replay (6,089 lines,
1,664 dispatches, no normalisation). Recipe: `docs/gsch-core/golden-f5p-recipe.md`.

## 5. The wave (for the record)

`2 pods × {FCFS, core-Q4, core-Q8} × {c64, c128, c32}`, 45 min each, batched
c64 → c128 → c32, plus **core-Q4 at N=1, c128, 45 min** as the matched anchor against
B2a f0p's first 45 minutes. Configs: `core-q4.env`, `core-q8.env`,
`core-q4-anchor-n1.env` (this directory).
