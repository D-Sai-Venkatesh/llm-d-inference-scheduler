# gsch_member_policy — membership-aware CPU-tier eviction for vLLM (out-of-tree cache policy)

`GroupMemberLRU` is vLLM's own `LRUCachePolicy` plus one rule: when the group scheduler (EPP `groupsched`,
`memberTierEviction: true`) reports that a thread LEFT this engine's group, the thread's chunks that no
staying member holds move to the LRU's eviction end. They are evicted only when a write needs room, and a
staying member's chunks go only once every departed chunk is gone. Requests without the notice change
nothing: the policy is then exactly LRU.

The notice rides on every request the EPP sends, in `kv_transfer_params["gsch_member"]`:
`{"thread", "members", "epoch", "source"}` (see `pkg/epp/framework/plugins/groupsched/member_notice.go`).

## Deploy (vLLM v0.30.0, OffloadingConnector)
1. ConfigMap with the module: `oc -n dasari create configmap gsch-member-policy --from-file=gsch_member_policy.py`
2. Engine pod: mount it (e.g. at `/opt/gsch-plugin`, read-only) and add `/opt/gsch-plugin` to `PYTHONPATH`.
3. `--kv-transfer-config`: in `kv_connector_extra_config` replace `"eviction_policy":"lru"` with
   `"eviction_policy":"GroupMemberLRU","cache_policy_module_path":"gsch_member_policy"` (keep `cpu_bytes_to_use`).
4. EPP: the arm's groupsched config adds `memberTierEviction: true` (needs `ownTierIndex: true`).

## Verify
- vLLM log at start: `Loading out-of-tree cache policy 'GroupMemberLRU' from 'gsch_member_policy'`.
- vLLM log every ~60 s under load: `gsch_member_policy: {'notices': …, 'lists_accepted': …, 'departed': …, 'demoted': …}`
  (`notices` ≈ requests, `departed` ≈ the EPP's exits on that engine, `demoted` > 0).
- EPP: `llm_d_epp_group_member_notices_total{engine,thread}` ≈ dispatches; `tier_demoted_blocks_total` > 0;
  `tier_seated_evictions_total` (the group's model of in-group loss) ≈ 0.

## Tests
`python3 -m unittest -v test_gsch_member_policy` — against vLLM's real `policies/base.py` + `lru.py`
(`VLLM_SRC`, default `/tmp/vllm-v0.30.0`); only the torch-importing `kv_offload/base.py` is stubbed.

# gsch_fs_tier — a bounded, parallel GPFS/filesystem tier (out-of-tree SecondaryTierManager)

`GschFsTier` sits behind the CPU tier (vLLM `TieringOffloadingSpec`): every chunk stored to the CPU tier is also written to
disk (write-through, async); a returner whose chain left the CPU tier is promoted back GPFS → CPU → GPU instead of recomputed.
It is vLLM's own `FileSystemTierManager` (layout, atomic temp+rename writes, thread pool) plus: parallel slices per job, an
in-memory index (no disk access on the scheduler thread), an LRU capacity cap with background deletes, write backpressure
(a slow disk never pins CPU slots), a per-load deadline (cancels slices not yet started), a free-space floor (5 %), one live
tier per root_dir (POSIX lock), and a fresh run directory per start (old ones swept at ≤ 500 files/s).
`GroupMemberLRU` additionally protects a lookup scan's HITs from that request's own promotions until its touch.

## Deploy (with the member policy)
kv_connector_extra_config:
  {"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":549755813888,
   "eviction_policy":"GroupMemberLRU","cache_policy_module_path":"gsch_member_policy",
   "secondary_tiers":[{"type":"GschFsTier","module_path":"gsch_fs_tier","root_dir":"/kvtier/<slot>",
      "max_bytes":900000000000,"n_load_slices":8,"n_store_slices":4,"load_deadline_s":10,"expected_row_bytes":6291456}]}
- both modules importable by EVERY vLLM process (PYTHONPATH at the container level: the metrics setup imports the tier too)
- ONE root_dir per engine; the volume must hold the sum of the caps plus headroom
- start command: `rm -f /dev/shm/vllm_offload_*.mmap` first (a container restart would otherwise leave 512 GiB in /dev/shm)
- expect a slower start: the tiering spec populates the whole CPU region at start-up
Log line every 60 s: `gsch_fs_tier: {stats} index=… writing=… pending_del=… jobs=… low_space=… load_s <by chain size>`.

## Tests
`./run_tests.sh` (each suite in its own process; needs a Python with torch CPU + vLLM's pure-Python deps and VLLM_SRC =
the v0.30.0 tree): member policy (stubbed base), the tier (real vLLM modules, incl. end-to-end through
TieringOffloadingManager), and an independent verifier's suite (incl. vLLM's C fs_io fast path when a compiler is present).
