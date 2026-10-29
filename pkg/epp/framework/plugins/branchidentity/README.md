# Branch Identity

**Type:** `branch-identity`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.PreRequest`,
`requestcontrol.ResponseBodyProcessor`, `plugin.ProducerPlugin`, `plugin.StateDumper`

Derives the **flow-control queue key from the request body**: the plugin flattens the
conversation into a canonical byte stream, chains it into block hashes, matches it against the
live branch table before admission control, and makes the matched (or newly minted) **branch id**
the request's `FairnessID`.

## Why the queue key is a BRANCH, not a session

The fairness id *is* the flow-control queue key: one id = one FIFO queue, and the scheduler only
ever sees queue heads. Keying by session therefore serialises a session's siblings behind each
other by construction — measured as TTFT p50 ×6.1 and a −13.5pt main-chain cache-hit collapse
(`v2c-analysis.md`). A branch is the unit that can actually dispatch in parallel: the parent turn
and each sub-agent that inherits its context get their own queue.

The **session** is still available — as an attribute (`branch-session`), never as a queue key.
That is what the stage-2 session fairness layer reads to tell a newcomer from a sibling.

Nothing in the scheduler changes: the policy seats whatever id it is given.

## How identity is derived

1. **Canonical flattening** (`tokenizer.FlattenCanonicalBytes`): `[tools]` + each message as
   `role` + content (plus assistant tool-call arguments), in wire order, with no per-request
   fields. Append-only conversations therefore produce streams where turn N is a byte prefix of
   turn N+1 — independent of JSON key order, `max_tokens`, `stream`, request ids, and of the chat
   template's token-space rewriting (thinking models drop past reasoning between turns, so token
   identity would false-new-branch on every turn).
2. **Chained block hashes**: the stream is packed 4 bytes per pseudo-token and hashed in
   `blockSizeTokens`-token blocks, each key chained from the previous (`ChainedBlockHashes`,
   lifted from groupsched with the same tail rule). The trailing **partial block is dropped**: its
   content changes as soon as the conversation grows.
3. **Tip matching**: the index stores only each branch's tip — the block count `n` and the
   chained hash of block `n`. A chained hash commits to every block before it, so one position
   check proves the whole prefix: O(1) memory per branch, ~one map lookup per chunk of a new
   request. The **longest** tip that is a prefix of the request's chain wins.
4. **Continue or fork**:
   - tip free (nothing in flight or queued) → **continue** (same id, same queue);
   - tip outstanding → **fork** a new branch. The usual case is a sub-agent inheriting the
     parent's context; the other is two identical prompts in flight at once, which content cannot
     separate and which are genuinely two parallel siblings;
   - **H1 reply guard**: a request that extends an outstanding tip AND begins with the tip's
     assistant reply contains that reply, which an in-flight request cannot have — proven
     completion, continue. Stage 0 measured 42 of 48 fork candidates to be exactly this case, so
     the guard — not a busy grace window — is what keeps the fork rate honest (the guard is
     `stickyBusyGuard`, on by default).
5. **Opaque stable ids**: 10 hex chars derived from the creating tip, the fork ordinal and the
   session scope. The same conversation derives the same id on any instance with the same salt,
   and a continuation keeps it for the branch's whole life.

`x-llm-d-inference-fairness-id` still wins when present (an explicit override), and a silently
derived id never fails a request: a body shape the flattener does not cover leaves the attribute
unset and the existing fallbacks apply.

## Hook timing

| hook | what it does | why here |
|---|---|---|
| `RequestHeader` (pre-admission) | flatten → chain → resolve → sets `FairnessID`; publishes `branch-identity` + `branch-session` | the queue key must exist **before** the request enqueues; the parsed body is available and nothing network-bound may run here |
| `PreRequest` (post-scheduling) | records the dispatch transition (in-flight counter) | **not** where busy is set: busy is set at arrival, because under load the queueing interval is most of a busy period (v2c median wait 22s) |
| `ResponseBody(EndOfStream)` | clears the branch's outstanding mark | forced by the server on abort/disconnect/panic for every request that picked a pod |

## Retention

Branches are **never GC'd by silence**. Stage 0 measured a p99.9 inter-turn idle gap of 3,036
seconds: any plausible `T_gc` drops dormant threads that then return, and a dropped branch is a
**false new branch** — a fresh fairness id, i.e. a newcomer, which is the unsafe direction of the
pre-registration. Branch records are content-addressed and tiny, the flow-control registry already
GCs idle *queues*, and the only eviction path is the count cap (`maxBranches`, idlest
non-outstanding branches first).

An *arrival* marker (a request that has not reached dispatch) expires after `queuedTtlSeconds`:
that is the only way a rejected-before-dispatch request can be observed (the response hook is
forced only once a pod is picked). Pruning an arrival errs toward continuation.

## Configuration

```yaml
plugins:
  - name: branch-identity
    type: branch-identity
    parameters:
      blockSizeTokens: 64        # 256-byte chunks (4 bytes per pseudo-token)
      queuedTtlSeconds: 600      # arrival-marker horizon (rejected-before-dispatch)
      sweepSeconds: 60           # cap enforcement tick
      maxBranches: 100000        # the only eviction path; ~200 B per record
      # sessionHeaders: [x-my-session]   # prepended to the built-in provider list
      # salt: ""                          # folded into the chain; empty = reproducible ids
      # forkOnBusy: true                  # false = every match continues (measurement arm only)
      # stickyBusyGuard: true             # H1 reply guard
```

Built-in session headers (priority order): `x-claude-code-session-id`, `x-session-affinity`,
`session-id`, `session_id` — the same list `agent-identity` resolves, so the session the policy
sees is the session the rest of the system sees.

## Metrics

| metric | meaning |
|---|---|
| `llm_d_epp_branch_matched_total` | continuations (same id as the branch's previous turn) |
| `llm_d_epp_branch_forks_total` | forks (extension of an outstanding tip) |
| `llm_d_epp_branch_new_total` | no tip matched: an unrelated branch |
| `llm_d_epp_branch_sticky_busy_total` | continuations proven by the H1 reply guard despite a busy tip |
| `llm_d_epp_branch_records` | live branches |
| `llm_d_epp_branch_flatten_ns` | flattening + chaining cost per request (pre-admission, untimed path) |
| `llm_d_epp_branch_unflattened_total` | bodies with no canonical stream (fairness id falls back) |
| `llm_d_epp_branch_evicted_total` | cap evictions |

`DumpState` returns the branch table (block count, message count, in-flight/queued, forks, turns,
idle/age; newest 200 rows).

Keep ids short: they can become metric label values, and fairness-id label cardinality is capped
(`--fairness-id-metric-label-limit` / `BoundFairnessID`).

## Known limits

- **State is per EPP instance.** Behind N replicas, the same conversation landing on another EPP
  re-derives its branch from content (same id, different table), which is fine for identity but
  not for the *outstanding* marks. The stage-1 campaign runs a single EPP.
- **Bodies shorter than one block** (a tiny first turn, < 256 canonical bytes) have no chain: their
  id comes from the whole-stream hash, so distinct short conversations stay distinct, but they are
  never *matchable* — their continuations derive the same id only if the bound body is identical.
- **A fork's first new message that is an assistant message** (a framework that spawns a child
  *after* the parent's assistant tool-call turn puts that message in the child's history) is
  indistinguishable from an H1 continuation by construction. The design labels this H2 and says
  measure, don't fix.
- **Tool-call flattening is UNVALIDATED by the campaign corpus** — the WEKA trace contains zero
  tool blocks, so the path is covered only by unit tests and the randomized prefix property test.
  Treat it as a **live-deployment risk** until a corpus with tool calls (Exgentic) confirms it.
- Without a client session header, look-alike traces of different sessions may share a branch
  (the accepted floor: A0-equivalent with derived branches).
