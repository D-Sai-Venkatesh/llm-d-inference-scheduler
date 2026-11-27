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
