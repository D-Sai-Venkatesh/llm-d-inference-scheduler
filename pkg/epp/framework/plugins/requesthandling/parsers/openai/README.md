# OpenAI Parser Plugin

**Type:** `openai-parser`

Parses HTTP/H2C requests and responses in the OpenAI API format. 

> [!NOTE]
> This plugin is enabled by default if no other parser is specified in `EndpointPickerConfig`. You do not need to explicitly declare it in your configuration.

Supports all standard OpenAI-compatible endpoints: completions, chat/completions, conversations, responses, embeddings, images/generations, and images/edits. 
The fields parsed out vary by endpoint: the request's input content (prompt, messages, or input), the streaming mode, and token usage from responses that report it. 
The images/edits endpoint accepts multipart/form-data.

**Parameters:** None.

## Tolerated wire shapes

One non-spec shape is decoded rather than rejected: `tools` delivered as a JSON
**string** that itself holds the tool array (some agent frameworks, and trace
replays built from OTel `gen_ai.tool.definitions`, serialize the definition list
as text). Strict decoding used to fail the whole request over it, which dropped
requests the engine could have served — inference-perf's Exgentic replay hit this
on 76 % of its requests. Only a string that decodes to a JSON **array** is
accepted; any other value keeps the original error. The repair is confined to the
**parsed** view (`InferenceRequestBody`): the forwarded body is the client's raw
bytes, so the EPP never rewrites what it could not parse.

> [!NOTE]
> Priority propagation is controlled by a single global switch,
> `requestHandler.propagatePriority` (bool, default: `false`), not per parser. When
> enabled, the EPP injects the resolved request priority into the outgoing request
> body's `priority` field (client-supplied `priority` is replaced so backend-native
> priority scheduling is governed by EPP). When disabled, the request body is
> forwarded unchanged.

---

## Related Documentation
- [Parsers Index](../README.md)
