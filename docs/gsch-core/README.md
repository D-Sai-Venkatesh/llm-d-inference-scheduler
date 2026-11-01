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
| `llm_d_epp_group_queue_view_age_seconds` | how old the queue-head EVIDENCE behind the view is, on the decision clock (−1 = nothing observed yet). It grows exactly while Picks stop (a closed pool gate), i.e. the window in which a head can be queued while `waitingSince` says otherwise — the C8 race's size, which is invisible in every other series |
| `llm_d_epp_group_queue_view_untrusted_passes_total` | budget-enforcement passes that ran with that view older than `idleSeconds`, so the C8 guard skipped the head-presence victim classes. 0 on a healthy stack; nonzero means the enforcement passes are re-syncing less often than S |
| `llm_d_epp_group_filter_decisions_total{decision}` | `pinned` vs `passthrough-*` (a run whose decisions are all passthrough is NOT pinned) |
| `llm_d_epp_group_cold_cap_tokens{engine}` | the DERIVED cold-entry stagger cap (the budget's headroom, in tokens). Scorers divide `cold_inflight_tokens` by THIS; the cap moves with the engine and must never be copied out of an arm's config |
| `llm_d_epp_group_cold_valve_held_total{engine}` | heads the valve held back (the budget would have admitted them; they WAIT). Non-zero is the stagger's measured cost; a value under `engine="single"` on a multi-engine stack was a labelling defect fixed in 8397e9aa |
| `llm_d_epp_group_event_ring_dropped_total` | SEAT/EXIT events the bounded ring has wrapped past — how much of the window lies BEFORE the dump's `RecentEvents` tail. The stdout event stream is the durable copy; this measures what the dump lost |

The gate line to read on a `none` arm: `detector_pool_score == 0` for the whole window
(the pool gate never closed) and `safety_fired == 0` everywhere (the backstop was off).

## 4. Declared differences vs f0p (each measured separately on the replay)

The core's golden is the ARCHIVE build with f0p's config plus the seven fixes below; it
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
`core-q4-offload-tierbudget.env` (§4f: the same arm with `sizing: gpu-plus-tier`). Each
keeps `OFFLOAD=0` cells on the same digest for the OFF/ON pair; `EPP_IMAGE_*` stays a
PLACEHOLDER until the reviewer has read the diff and a digest is built.

## 4f. The tier-sized seat budget — `sizing: "gpu-plus-tier"` (2026-11-02)

The SIZING module's SECOND value, and the one that prices the offload deployment's real
capacity:

```
budgetBlocks = floor(groupBudgetFraction x (num_gpu_blocks + tierBlocks))
```

`tierBlocks` is the CPU tier SHADOW's own capacity (§4e), so the value adds no constant:
the same two deployment facts (`cpuTierBytes`, `kvBlockBytes`) that size the shadow also
size the budget it is added to. On the campaign deployment the budget goes from 38,173
blocks to 121,186 (GPU 40,183 + tier 87,381, x 0.95).

**Why the budget and not the order.** The budget sizes the working set an engine may hold,
and what it reserves against is a MISS: with no tier a miss is a full prefill, with one it
is ~1-4 % of a prefill. A seated member whose chain has left HBM is therefore still cheap
to serve, and the engine's reachable capacity is GPU + tier (3.17x the GPU pool). Sizing
from GPU alone reserves for the expensive miss the deployment no longer pays, and the c64
cells show the cost: stock FCFS gained ~2x from offload (FCFS-ON 1.390 req/s) where the
GPU-sized core gained 1.267.

**What does not move**, both pinned by test:

* the per-head COST stays GPU-only (`TestGpuPlusTierCostIsTheGPUOnlyCost`): a head is
  admitted on what it will occupy in HBM, the same rule `hrrn-tier`'s tier-aware cost
  obeys — restorable blocks price priority, never seats;
* the cold-entry valve's CAP stays the GPU headroom, `floor((1 - fraction) x num_gpu_blocks)
  x blockSizeTokens` = 128,576 tokens here (`TestGpuPlusTierValveCapStaysGPUHeadroom`): the
  valve staggers GPU allocation bursts, and a tier cannot absorb one.

**`tier = 0` is `sizing: "index"`** arithmetically and on the state machine
(`TestGpuPlusTierZeroTierEqualsGPU`), so an arm can carry the value in both deployments and
an OFF cell is not a second policy. The default value stays `"index"`, and the default's
budget does not move when a tier is configured (`TestGpuPlusTierDefaultIsUnchanged`); the
golden replay (`TestGolden`) runs on that default, byte for byte.

The arm: `core-q4-offload-tierbudget.env` is `core-q4-offload.env` with ONE setting changed
(`CORE_SIZING=gpu-plus-tier`). Its contrast is a same-digest pair, so the batch is
`core-q4-offload ‖ core-q4-offload-tierbudget` at c64 and the same pair at c128. The
emitter needs no new key: `CORE_SIZING` is already rendered into `CORE_MODULE_BLOCK`, and
`buildModules` accepts the value (conformance test
`TestModuleValuesMatchTheEmitter`, loader gate
`TestGroupSchedulerCoreChainStartsWithGpuPlusTierSizing`).


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
