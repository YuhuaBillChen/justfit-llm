# Hard thinking budget

The JustFit candidate provides a native hard cap. The frontend validates
`thinking_budget`, passes its token count and the tokenizer's `</think>` ID to the
native engine, and limits speculative commits at the exact boundary. The close
token then enters the ordinary decode path, including KV and recurrent state;
answer generation continues within the request's overall output limit.

For example, on the direct JustFit endpoint:

```json
{
  "model": "incoai/Qwen3.8-27B-Splash",
  "messages": [{"role": "user", "content": "Compute 17 times 19."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_budget": 1536,
  "max_completion_tokens": 2048
}
```

An omitted or zero budget preserves unlimited thinking within the overall output
limit. A positive integer requires thinking enabled and an output limit greater
than `thinking_budget + 1`, reserving the close token and at least one answer
token. Invalid values return HTTP 400. The budget excludes the injected close
itself. A naturally emitted close stops budget enforcement early. Budgets are
per request and apply to ordinary, streamed, structured and tool-call generation.
They do not guarantee answer correctness or enough answer space for every task.
Raw text completions reject a positive budget because they do not establish a
thinking template.

Responses requests carry the same top-level `thinking_budget`. Messages requests
map `thinking.budget_tokens` onto it. An external broker may apply model aliases or reasoning policies before
forwarding a request. This example targets the direct engine endpoint.

## Paired upgrade and checks

This change uses native protocol **9**, with a 94-byte fixed request payload.
Upgrade the Python server and native executable together. An older executable
cannot be paired with the new frontend. Metal kernels and their interfaces are
unchanged. The host executable build identity changes. Persistent cache directory
names depend on the model, layouts and cache format, so a host-only update can
reuse the directory. Preserve its data and validate actual cold restore with the
new executable rather than assuming a fresh directory or automatic qualification.

Run the packaged Python tests with the pinned dependencies and a C++20 compiler:

```bash
python -m unittest discover -s tests -v
```

On an isolated Apple server built from this candidate, run:

```bash
python scripts/check_thinking_budget.py \
  --base-url http://127.0.0.1:YOUR_PORT \
  --output local-state/thinking-budget-check
```

The checker requires a genuine long-thinking positive control, exact small caps
across speculative boundaries, the 1,536-token cap, streaming, JSON,
tool calls, concurrent distinct caps and HTTP validation. It preserves responses
and fails on a missing answer. Run it on the target device; Python unit tests alone do not verify native execution.

For a compatible external node broker, use its public reasoning policy rather
than assuming the direct-engine field survives translation:

```bash
python scripts/check_node_budget.py \
  --base-url http://YOUR_NODE:18011 \
  --output local-state/node-thinking-budget-check
```

This checks `reasoning.max_tokens` at 1/8/31/32, the public bounded-thinking
alias at 1536, streaming, tools and concurrent distinct budgets. The unlimited
control uses the broker's `reasoning_effort: max` policy. The check consumes
model capacity and should run while the node is drained from product traffic.
