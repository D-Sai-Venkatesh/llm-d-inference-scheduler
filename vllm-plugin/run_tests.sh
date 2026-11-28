#!/usr/bin/env bash
# Each suite in its own process: test_gsch_member_policy stubs vllm.v1.kv_offload.base (torch-free), the fs-tier suites load
# vLLM's REAL modules (need a Python with torch CPU + vLLM's pure-Python deps, e.g. /tmp/vllm-test-venv). VLLM_SRC = v0.30.0 tree.
set -u; cd "$(dirname "$0")"; PY=${PY:-/tmp/vllm-test-venv/bin/python}; rc=0
for t in test_gsch_member_policy test_gsch_fs_tier test_gsch_fs_tier_verify; do
  out=$("$PY" -m unittest "$t" 2>&1); last=$(echo "$out" | grep -E "^(OK|FAILED)" | tail -1); ran=$(echo "$out" | grep -E "^Ran " | tail -1)
  echo "$t: $ran — $last"; case "$last" in OK*) ;; *) rc=1; echo "$out" | grep -E "^(FAIL|ERROR):|Error|assert" | head -20;; esac
done
exit $rc
