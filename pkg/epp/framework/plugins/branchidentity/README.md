# Branch Identity

**Type:** `branch-identity`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.PreRequest`,
`requestcontrol.ResponseBodyProcessor`, `plugin.ProducerPlugin`, `plugin.StateDumper`

Derives the **flow-control queue key from the request body**: the plugin flattens the
conversation into an identity byte stream, chains it into block hashes, matches it against the
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

1. **Identity flattening** (`tokenizer.FlattenIdentityBytes`, a *separate* view from the sizing
   path's `FlattenCanonicalBytes`): each message as `role` + content + assistant tool-call
   arguments, in wire order, with no per-request fields, **no tool-DEFINITION block** and **no
   thinking blocks**. Append-only conversations therefore produce streams where turn N is a byte
   prefix of turn N+1 — independent of JSON key order, `max_tokens`, `stream`, request ids, of the
   chat template's token-space rewriting, of the client dropping earlier turns' thinking (the
   Anthropic API keeps only the last assistant turn's signed thinking), and of a mid-thread tool-set
   change (claude_code hands a sub-task to a `Task` tool and the set changes under it).
   Tool-*set* equality is deliberately **not** a gate: a gate would turn exactly those legitimate
   churns into non-matches, i.e. the false-new-branch class it was meant to remove.
2. **Chained block hashes**: the stream is packed 4 bytes per pseudo-token and hashed in
   `blockSizeTokens`-token blocks, each key chained from the previous (`ChainedBlockHashes`,
   lifted from groupsched with the same tail rule). The block count is
   `floor(len(stream) / blockBytes)` — a block counts only when the stream completes it — and the
   trailing partial block is not chained: its content changes as soon as the conversation grows.
3. **Tail-exact tips**: the index stores each branch's tip as `(blocks, h_blocks, tailLen,
   tailHash)` — the number of full blocks, the chained hash of the last of them, and the byte
   length + hash of everything past it. A chained hash commits to every block before it, so the
   full blocks are proven by one lookup while the **tail is verified explicitly, byte for byte**:
   a request continues the branch only if the tip's *entire* stream is a byte prefix of its own.
   The **longest** matching tip wins, and a tip whose tail does not match falls through to the next
   shorter candidate. Tips **shorter than one block** have no block hash to look up, so they are
   indexed by `(tailLen, tailHash)` and matched the same way — without that, every sub-chunk
   conversation (a turn under one block) read as a false NEW branch on every turn.
4. **Continue or fork**:
   - tip free (nothing in flight or pending) → **continue** (same id, same queue);
   - **H1 reply guard**: a request that extends an outstanding tip AND begins with the tip's
     assistant reply contains that reply, which an in-flight request cannot have — proven
     completion, continue. Stage 0 measured 42 of 48 fork candidates to be exactly this case, so
     the guard — not a busy grace window — is what keeps the fork rate honest (`stickyBusyGuard`,
     on by default). The guard needs tail-exact tips to be sound: only then is message index
     `msgCount` in the request the first message past the tip.
   - **S2 retry rule**: a request **byte-identical** to the tip while the branch has nothing IN
     FLIGHT (only arrival markers) is a client retry of a still-queued turn, not a sibling →
     continue (`retryRule`, on by default; counted as `retry_total`). Two identical prompts that
     are genuinely both in flight reach `PreRequest` and fork there;
   - tip outstanding → **fork** a new branch. The usual case is a sub-agent inheriting the
     parent's context.
5. **Opaque stable ids**: 10 hex chars derived from the creating stream's tip commitment, the fork
   ordinal and the session scope. The same conversation derives the same id on any instance with
   the same salt, and a continuation keeps it for the branch's whole life. An id held by a **live**
   branch is never reused: a re-minted seed (a client re-sending an earlier turn after the branch
   advanced, a look-alike fork) bumps the ordinal and gets a distinct branch. Overwriting would
   leave the old record *ghosted* — still matched by the tip index, no longer in the table — so it
   would keep advancing requests and would never be evictable.

`x-llm-d-inference-fairness-id` still wins when present (an explicit override) and is counted
(`explicit_override_total`), and a silently derived id never fails a request: a body shape the
flattener does not cover leaves the attribute unset and the existing fallbacks apply.

**Stage-1 deployment note (the harness, not the code).** The derivation is a no-op for any run
whose harness sends `x-llm-d-inference-fairness-id`: aiperf's
`--session-header=x-llm-d-inference-fairness-id` (and `--session-header-source=root`) turn every
request into an "explicit override" and the arm then measures A0 with extra steps. Stage 1 must run
with **no fairness-id session header** (`--session-header-source=root` alone is fine: the session id
lands in the branch's separate `branch-session` attribute). Check
`llm_d_epp_branch_explicit_override_total == 0` in the run's scrapes before reading any result.

## Hook timing

| hook | what it does | why here |
|---|---|---|
| `RequestHeader` (pre-admission) | flatten → chain → resolve → sets `FairnessID`; publishes `branch-identity` + `branch-session` | the queue key must exist **before** the request enqueues; the parsed body is available and nothing network-bound may run here |
| `PreRequest` (post-scheduling) | moves the request's marker from the branch's `pending` set to its `inflight` set | **not** where busy is set: busy is set at arrival, because under load the queueing interval is most of a busy period (v2c median wait 22s) |
| `ResponseBody(EndOfStream)` | removes the request's marker from the branch's `inflight` set | forced by the server on abort/disconnect/panic for every request that picked a pod |

**Markers are per request id, not counters.** Each branch holds a `pending` and an `inflight` *set*
of request ids (`x-request-id`, or a per-request synthetic id when the client sent none, published
as the `branch-marker` attribute so all three hooks agree). The outstanding predicate is "my sets
are non-empty", so a cancellation, a rejection, a retry and a settle each add or remove exactly one
entry — a FIFO counter cannot say *which* request ended, and shifting it on the wrong one either
frees a branch that still has work queued or holds one that has none.

## Retention

Branches are **never GC'd by silence**. Stage 0 measured a p99.9 inter-turn idle gap of 3,036
seconds: any plausible `T_gc` drops dormant threads that then return, and a dropped branch is a
**false new branch** — a fresh fairness id, i.e. a newcomer, which is the unsafe direction of the
pre-registration. Branch records are content-addressed and tiny, the flow-control registry already
GCs idle *queues*, and the only eviction path is the count cap (`maxBranches`, idlest
non-outstanding branches first).

**Markers expire after `markerTtlSeconds` (default 3600s — the client's request timeout), not after
the flow queue's 5-minute GC.** Those are different clocks: on A0 16% of requests waited more than
600s for admission, so a 600s horizon would expire the markers of requests that are still
legitimately queued and hand their branch to a concurrent sibling as if it were free. Expiry is the
only way a rejected-before-dispatch request (or a client that vanished mid-stream) is ever observed
— the response hook is forced only once a pod is picked — and pruning errs toward continuation.

## Configuration

```yaml
plugins:
  - name: branch-identity
    type: branch-identity
    parameters:
      blockSizeTokens: 64        # 256-byte chunks (4 bytes per pseudo-token)
      markerTtlSeconds: 3600     # marker horizon = the client request timeout (NOT the queue GC)
      sweepSeconds: 60           # cap enforcement tick
      maxBranches: 100000        # the only eviction path; ~200 B per record
      # sessionHeaders: [x-my-session]   # prepended to the built-in provider list
      # salt: ""                          # folded into the chain; empty = reproducible ids
      # forkOnBusy: true                  # false = every match continues (measurement arm only)
      # stickyBusyGuard: true             # H1 reply guard
      # retryRule: true                   # S2: an exact duplicate of a still-queued tip is a retry
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
| `llm_d_epp_branch_retry_total` | S2 continuations: byte-identical duplicates of a still-queued tip |
| `llm_d_epp_branch_explicit_override_total` | requests that carried an explicit fairness id (a run of this plugin must keep it at 0) |
| `llm_d_epp_branch_records` | live branches |
| `llm_d_epp_branch_flatten_ns` | identity flattening + chaining cost per request (pre-admission, untimed path) |
| `llm_d_epp_branch_resolve_ns` | resolve cost per request (inside the table lock: match + mint) |
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
- **Bodies shorter than one block** (a tiny first turn, < 256 canonical bytes) have no chain: they
  are matched by `(tailLen, tailHash)` instead, so their continuations do continue them. The
  sub-chunk scan costs one hash per live sub-chunk length (bounded by the block size in bytes) and
  only runs while such a tip exists.
- **A fork's first new message that is an assistant message** (a framework that spawns a child
  *after* the parent's assistant tool-call turn puts that message in the child's history) is
  indistinguishable from an H1 continuation by construction. The design labels this H2 and says
  measure, don't fix.
- **Tool-call flattening is UNVALIDATED by the WEKA corpus** (zero tool blocks there) but measured
  on Exgentic: 67.5% of its requests carry `tool_calls`, and the amended matcher scores 0.00%
  content-continuous false-new on them (stage 0B §5). Stage 0C re-scores the SHIPPED Go code.
- Without a client session header, look-alike traces of different sessions may share a branch
  (the accepted floor: A0-equivalent with derived branches).
