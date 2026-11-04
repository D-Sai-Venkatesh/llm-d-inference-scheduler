# Branches — the group scheduler work

Repo: `github.com/D-Sai-Venkatesh/llm-d-inference-scheduler` (remote `dsai`)

| branch | commit | what it holds | use it for |
|---|---|---|---|
| **`group-scheduler-working-branch`** | `c9c6a5ea` | **the working branch.** Same code as the backup commit `c7291bb7`, with the inline comments trimmed to one sentence each (2,258 comment lines removed; token-stream hashes identical, build + `go vet` + `go test -race` green, goldens pass). | **All work from now on.** |
| `group-scheduler-working-branch-backup` | `c7291bb7` | **the backup.** The code as it was before the comment trim — full inline reasoning intact (the R4 two-refcount note, rule 4's deletion, the cascade proof, the historical lessons). This is the commit the campaign's runs were built from: image `groupsched-5d8cb80d`. | Reference only. Read it if a trimmed comment is ambiguous, or if you want the original prose back. |

## What did NOT change between the two

The trim touched **comments only**. Verified, not assumed:

- comments and whitespace stripped from every `.go` file → **28/28 files byte-identical** across the two commits
- per-file token-stream SHA-256 identical before and after
- `go build`, `go vet`, `gofmt -l`, `go test -race` (groupsched + branchidentity), and the behavior goldens all green

One cosmetic exception, declared: `rule_o_test.go` was **not gofmt-clean at `c7291bb7`** (a test body was over-indented by one tab), so `gofmt` re-indented it during the trim. Whitespace only.

## Run provenance

The campaign's measured runs used the image built from **`5d8cb80d`**, which is in this branch's history (two observability-only commits below `c7291bb7`). Neither branch above changes behavior relative to it.

## Not to be confused with

- `dsai/gsch-core` (`65ef5a57`) — the pre-rebase reference tree.
- `checkpoint/explained-and-verified-2026-11-02` (`65ef5a57`) — the pre-rebase checkpoint.
