# Retired arms

These env files are the configs of measurement arms whose MODULE no longer exists in the
build. They are kept verbatim as provenance — the numbers they produced are cited in
`docs/gsch-core/README.md` and in the notes they were run for — but they are moved out of
`docs/gsch-core/` so the shipped-configs guards
(`TestShippedCoreEnvFilesUseKnownModuleValues`) keep meaning what they say: every `*.env`
beside the README names values THIS build accepts, and an arm that named a removed value
would CrashLoop at startup instead of measuring something.

A retired file is therefore a RECORD, not a runnable arm. To reproduce one, check out the
build digest named in the README section that cites it.

| file | module it needed | why it retired |
|---|---|---|
| `core-hrrn-tier-q4.env` | `entryOrder: "hrrn-tier"` (+ the CPU-tier shadow) | the tier-aware queue module — the scheduler trim (2026-11-02) keeps one implementation per module: sizing=index, entryOrder=newcomer-first. The measurement stands (README §4e). |
