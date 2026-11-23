# Using JustFit with other clients

`justfit serve` (default `http://127.0.0.1:8000`) answers four API families on
the same port:

| API | Endpoints | Typical clients |
|---|---|---|
| OpenAI Chat Completions | `/v1/chat/completions`, `/v1/completions`, `/v1/models` | Open WebUI, Continue, Cline, Jan, most SDKs |
| OpenAI Responses | `/v1/responses` | Codex CLI, the OpenAI Agents SDK |
| Anthropic Messages | `/v1/messages`, `/v1/messages/count_tokens` | Claude Code, the Anthropic SDK |
| Ollama | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` | the `ollama` CLI and apps that only offer an "Ollama" provider |

The served model is `mlx-community/Qwen3.8-27B-4bit` unless you chose another
one; `GET /v1/models` (and `GET /api/tags`) lists every accepted name.

## Thinking on or off: `:instant`

The model thinks before it answers by default. Every model name also has an
`:instant` variant, for example `mlx-community/Qwen3.8-27B-4bit:instant`, that
answers without thinking. Use it in clients that let you pick a model but not
send reasoning settings (Ollama apps, an "Instant" model in Open WebUI).
An explicit setting in the request still wins: `reasoning_effort`,
`chat_template_kwargs.enable_thinking`, Responses `reasoning.effort`,
Anthropic `thinking`, or Ollama `think`.

## Coding agents: `justfit launch`

Start the server, then in another terminal:

```bash
justfit launch claude [-- CLAUDE_ARGS...]
justfit launch codex  [-- CODEX_ARGS...]
```

`launch` asks the running server for its model name and context window and
starts the agent pointed at it. Claude Code gets environment variables
(`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` and the
per-tier model variables); Codex gets `-c` overrides that add a `justfit`
model provider using the Responses API, set `model_context_window`, and turn
off web search. Neither agent's configuration files are changed. Options:
`--url URL` for a server on another port or host, `--model NAME`, and
`--dry-run` to print the command instead of running it. If the server runs
with `--api-key`, export `JUSTFIT_API_KEY` before `launch`.

Both agents send long system prompts (Claude Code about 12K tokens). The
first request after the server starts pays for that prefill; later requests
reuse the cached prefix. On an M4 Pro, fixing a one-line bug took 106 s the
first time and 16 s after that with Claude Code, and 75 s with Codex.

The agents report the model as unknown (`unrecognized_model`, "model metadata
not found"); these are warnings only.

## Ollama clients

Point the client at the JustFit address instead of `http://127.0.0.1:11434`,
or start JustFit on Ollama's port with `justfit serve --port 11434` (stop
Ollama first). For the CLI:

```bash
OLLAMA_HOST=http://127.0.0.1:8000 ollama run mlx-community/Qwen3.8-27B-4bit
```

`:latest` tags are accepted. With the chat page off (`--no-webui`), `GET /` answers `Ollama is running`, the heartbeat Ollama clients check. Mapped request fields: `messages` (including
`images` and tool calls/results), `prompt`/`system`/`images` for generate,
`tools`, `format` (`"json"` or a JSON schema), `think` (boolean or
`low|medium|high`), `stream` (NDJSON, on by default) and the `options`
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `num_predict`,
`repeat_penalty`, `presence_penalty`, `frequency_penalty`. Other options
(`num_ctx`, `num_gpu`, …) are ignored: the server's own limits apply.
Not offered: model management (`pull`, `create`, `copy`, `delete`, `push`),
embeddings, raw `generate` and `suffix`.

## Open WebUI and other OpenAI-compatible apps

Add an OpenAI connection with base URL `http://127.0.0.1:8000/v1` and any API
key (or the `--api-key` value). Thinking arrives as `reasoning_content`, which
Open WebUI shows as a collapsible "thinking" block. For fast answers without
thinking, pick the `:instant` model; to make no-thinking the server-wide
default instead, start it with `--default-reasoning-effort none`.

## Hosted tools (web search, code interpreter, …)

Some clients include tools that the provider is expected to run on its own
servers: OpenAI's `web_search`, `file_search`, `code_interpreter`,
`image_generation`, `computer_use_preview` and remote `mcp`, and Anthropic's
`web_search_*`, `web_fetch_*` and `code_execution_*` server tools. A local
server cannot run them.

- Default (`--tool-policy ignore`): such tools are left out of the prompt, so
  the model never sees or calls them, and the response carries
  `x-justfit-ignored-tools: <types>`; the server logs each type once.
- `--tool-policy strict` (or `JUSTFIT_TOOL_POLICY=strict`): the request fails
  with HTTP 400 naming the tool.

Tools the client runs itself are always passed to the model: OpenAI
`function` tools, Responses `namespace` tools (flattened to
`namespace__name`) and Anthropic custom tools. Any other tool type (for
example Responses `custom` or `local_shell`) is rejected with an error that
names the type, never silently dropped.

## Checking a setup

`tools/client-smoke.sh [URL]` checks every API above against a running server
and, when installed, runs a small bug-fix task with Claude Code and Codex
through `justfit launch` and a prompt through the `ollama` CLI. It prints one
PASS/FAIL/SKIP line per check.
