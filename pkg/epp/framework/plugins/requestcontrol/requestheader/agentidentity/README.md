# Agent Identity

**Type:** `agent-identity`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `plugin.ProducerPlugin`

Resolves an agent's identity — the session (program) a request belongs to, the thread that sent it, and the thread that spawned that one — from canonical llm-d headers and harness-specific HTTP headers, and stores them as request attributes for use by other subsystems. The Director then derives `FairnessID` from the session attribute when no explicit fairness header is present, so every turn of an agent session lands in the same flow-control fairness queue.

## What It Does

The plugin runs after request assembly and before admission control. It resolves three fields — session, thread, parent — and stores them as request attributes: `agent-identity` (the session, unchanged from before), `agent-thread-id`, and `agent-parent-thread-id`. An empty parent marks a root thread; a thread id the harness does not send defaults to the session id, so a harness without thread ids has exactly one thread per session.

The plugin stores the identity as request attributes only — it does not set `FairnessID`. The Director reads the `"agent-identity"` attribute and derives `FairnessID` from it when no explicit `x-llm-d-inference-fairness-id` header is present. This separation means the agent identity is available as a reliable signal to other subsystems (scheduling, KV cache control, etc.) without being conflated with flow-control identity.

## How It Works

Resolution runs a fixed chain of handlers (the `HarnessHandler` interface, one `Identify` per source); the first that recognizes the request owns it:

1. **llm-d canonical** — `x-llm-d-session-id`, `x-llm-d-thread-id`, `x-llm-d-parent-thread-id`. Any of these present and the canonical source owns the whole identity, skipping harness detection: a gateway that stamps llm-d headers takes responsibility for the identity. A canonical thread id with no parent and no session names its own root.
2. **custom** — the operator's `additionalSessionHeaders`. A match wins the session, ahead of every harness.
3. **Claude Code** — `x-claude-code-session-id` (the program root), `x-claude-code-agent-id` (sub-agent thread; absent on the main thread), `x-claude-code-parent-agent-id` (nested sub-agents only). An agent id with no parent header marks a depth-1 sub-agent whose parent is the main thread, so the parent is the session id.
4. **OpenCode** — `x-session-affinity` as the session only. The affinity id names one thread, not a program root; thread, parent and root derivation are deferred (see Limitations).
5. **Codex** — `session-id` (root thread's id; `session_id` legacy fallback), `thread-id`, `x-codex-parent-thread-id`.

If nothing recognizes the request, no attributes are stored — the Director falls through to `metadata.DefaultFairnessID`, so the request is still admitted, just into the shared default queue.

After the plugin returns, the Director checks:
1. If `x-llm-d-inference-fairness-id` header is set → use as FairnessID (explicit always wins).
2. Else if `"agent-identity"` attribute exists → copy to FairnessID.
3. Else → `"default-flow"`.

The plugin is stateless and safe under concurrent use.

## Inputs Consumed

- `scheduling.InferenceRequest.Headers` — read-only lookup of the headers above (canonical, custom, and per-harness). Keys are expected lowercase (Envoy normalizes inbound headers).

## Outputs Produced

- `scheduling.InferenceRequest` attribute `"agent-identity"` (`string`) — the session; set when a source recognizes the request.
- `scheduling.InferenceRequest` attribute `"agent-thread-id"` (`string`) — the thread that sent the request; equals the session when the harness sends no thread id.
- `scheduling.InferenceRequest` attribute `"agent-parent-thread-id"` (`string`) — the spawning thread; empty on a root thread.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: agent-identity` entry to enable; the runner discovers it as a `RequestHeaderProcessor` and wires it in.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `additionalSessionHeaders` | `[]string` | No | `[]` | Extra session header names, checked ahead of every harness handler. Order is preserved; the first non-empty match wins. Use this to support a new agent, or to track an upstream rename, without a code change. |

### Examples

Default configuration — no parameters, only the built-in headers are checked:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: agent-identity
```

With additional headers — checked before the built-in defaults (header names are arbitrary; substitute whatever the agent actually emits):

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: agent-identity
    parameters:
      additionalSessionHeaders:
        - x-my-agent-session
        - x-another-agent-id
```

### Per-agent client setup

The plugin only reads headers — getting them onto the wire is the agent's job. Each supported agent has different requirements.

#### Claude Code — **LiteLLM is required**

Claude Code speaks Anthropic's Messages API. llm-d's gateway exposes the OpenAI chat-completions wire format, so a translator is required in the path. LiteLLM works:

```yaml
# LiteLLM proxy config (pass to `litellm --config <path>`)
model_list:
  - model_name: <client-facing-model-name>
    litellm_params:
      model: hosted_vllm/<upstream-model-name>
      api_base: http://<llmd-gateway>/v1

general_settings:
  forward_client_headers_to_llm_api: true
```

`forward_client_headers_to_llm_api: true` is **required** — without it LiteLLM strips `x-claude-code-session-id` (and every other `x-*` header) on the way to the upstream, and the plugin sees nothing.

Then point Claude Code at LiteLLM and launch it. Use a settings file (rather than env vars) so inherited user-level settings, OAuth credentials, or keychain entries cannot override the configuration:

```json
// Claude Code settings file (any path, e.g. /tmp/claude-llmd-settings.json)
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://<litellm-host>",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "<client-facing-model-name>"
  }
}
```

```bash
claude --bare --settings <path-to-settings.json> --setting-sources ""
```

`--bare` disables OAuth and keychain reads; `--setting-sources ""` disables loading any other settings file. Together they ensure only the file passed via `--settings` is used.

`<client-facing-model-name>` must match the `model_name` declared in the LiteLLM `model_list` above. `ANTHROPIC_AUTH_TOKEN` is required by Claude Code but its value is unused when LiteLLM has no `master_key` set — any non-empty string works. Claude Code emits `x-claude-code-session-id` automatically on every outbound request — no further client config needed.

#### OpenCode — **No LiteLLM required**

OpenCode uses Vercel's AI SDK with `@ai-sdk/openai-compatible` and speaks OpenAI chat-completions natively, so it talks to the llm-d gateway directly.

```json
// ~/.config/opencode/opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llmd-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llmd-local",
      "options": {
        "baseURL": "http://<llmd-gateway>/v1",
        "apiKey": "dummy"
      },
      "models": {
        "<upstream-model-name>": { "name": "<display-name>" }
      }
    }
  }
}
```

OpenCode emits `x-session-affinity` automatically on every outbound request.

#### Codex — **No LiteLLM required**

Codex emits a session header automatically on every outbound request. Current builds use the hyphenated `session-id` (no `x-` prefix); older builds use the underscored `session_id` form, which the plugin still recognizes as a fallback.

## Limitations

- **Default-queue fall-through is silent.** Requests from agents that don't match any of the configured headers land in the default fairness queue without any indication. This is by design (the plugin is non-fatal), but operators should not assume the absence of errors means every client is being identified.
- **OpenCode identity is session-only.** `x-session-affinity` names one thread, not a program root, so an OpenCode sub-agent counts as a session of its own and `agent-thread-id` equals its session id. Thread, parent and root derivation are deferred pending [anomalyco/opencode#54881](https://github.com/anomalyco/opencode/issues/54881), which asks OpenCode for a root session header; a root header would make program-level grouping a one-line change here.
- **Forks and compaction are not read.** The `forked_from` and `compaction` markers exist on the wire (Claude Code hint headers, Codex `x-codex-turn-metadata`) but are a later extension.
- **Codex `previous_response_id` is not used.** It references the prior turn's response, not the chain root, so keying on it would shard one conversation across many queues. Correctly folding it back to the root requires a `ResponseBody` hook recording `response.id → root` mappings, which this plugin does not implement.

## Related Documentation
- Claude Code session header (official): <https://code.claude.com/docs/en/llm-gateway> — the `X-Claude-Code-Session-Id` row in "Request headers Claude Code includes."
- OpenCode session header (Cloudflare announcement, documents the `x-session-affinity` contract): <https://blog.cloudflare.com/workers-ai-large-models/>
- Codex session header (Codex CLI source — `build_session_headers` inserts `session_id` as an HTTP header on every outbound request; OpenAI does not document this in the public docs): <https://github.com/openai/codex/blob/d2e18246c96e8b440f9d97135356d37f3f3b4d63/codex-rs/codex-api/src/requests/headers.rs>
