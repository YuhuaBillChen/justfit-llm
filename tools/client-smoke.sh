#!/usr/bin/env bash
# Client compatibility smoke test for a running JustFit server.
#
#   tools/client-smoke.sh [URL]          default http://127.0.0.1:${JUSTFIT_PORT:-8000}
#
# Protocol checks always run (curl + python3). Agent checks run when the agent
# is installed: Claude Code and Codex through `justfit launch`, Ollama through
# the ollama CLI. Each agent fixes a one-line bug in a scratch directory.
# Prints one PASS/FAIL/SKIP line per check and exits non-zero on any FAIL.
set -u
URL="${1:-http://127.0.0.1:${JUSTFIT_PORT:-8000}}"
URL="${URL%/}"
JUSTFIT="${JUSTFIT:-justfit}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-600}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/justfit-smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
failures=0

report() { printf '%-5s %-34s %s\n' "$1" "$2" "${3:-}"; [ "$1" = FAIL ] && failures=$((failures + 1)); return 0; }
post() { curl -sS -m "$AGENT_TIMEOUT" -D "$WORK/headers" -H 'content-type: application/json' -d "$2" "$URL$1"; }
check() { # name, python expression over `r` (parsed JSON) and `h` (headers text), response text
  local name="$1" expr="$2" body="$3"
  if R="$body" H="$(cat "$WORK/headers" 2>/dev/null)" python3 -c "
import json, os, sys
raw = os.environ['R']; h = os.environ['H'].lower()
try: r = json.loads(raw)
except Exception: r = [json.loads(l) for l in raw.splitlines() if l.strip()]
sys.exit(0 if ($expr) else 1)" 2>/dev/null; then report PASS "$name"; else report FAIL "$name" "$(printf '%s' "$body" | head -c 200)"; fi
}

models="$(curl -sS -m 10 "$URL/v1/models")" || { report FAIL "server reachable" "$URL"; exit 1; }
MODEL="$(printf '%s' "$models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
report PASS "server reachable" "$MODEL"

# Request bodies are printf templates in single quotes: macOS /bin/bash 3.2
# mangles escaped quotes inside "$( ... )".
body() { printf "$1" "$MODEL"; }

# OpenAI chat completions
check "openai chat" "r['choices'][0]['message'].get('content')" \
  "$(post /v1/chat/completions "$(body '{"model":"%s","messages":[{"role":"user","content":"Say ok."}],"max_tokens":32,"reasoning_effort":"none"}')")"

# :instant answers without thinking when the request says nothing about it
check "instant model skips thinking" "r['choices'][0]['message'].get('content') and not r['choices'][0]['message'].get('reasoning_content')" \
  "$(post /v1/chat/completions "$(body '{"model":"%s:instant","messages":[{"role":"user","content":"Say ok."}],"max_tokens":64}')")"
check "default model thinks" "r['choices'][0]['message'].get('reasoning_content')" \
  "$(post /v1/chat/completions "$(body '{"model":"%s","messages":[{"role":"user","content":"Say ok."}],"max_tokens":400}')")"

# OpenAI Responses with a hosted tool: dropped and disclosed, not a 400
check "responses + hosted web_search" "r.get('status') in ('completed','incomplete') and 'x-justfit-ignored-tools: web_search' in h" \
  "$(post /v1/responses "$(body '{"model":"%s","input":"Say ok.","max_output_tokens":32,"reasoning":{"effort":"none"},"tools":[{"type":"web_search"},{"type":"function","name":"noop","parameters":{"type":"object","properties":{}}}]}')")"

# Unknown client-side tool types are still rejected, by name
check "responses rejects unknown tool type" "'custom' in json.dumps(r) and 'error' in r" \
  "$(post /v1/responses "$(body '{"model":"%s","input":"hi","tools":[{"type":"custom","name":"x"}]}')")"

# Anthropic Messages with a server tool
check "anthropic + server web_search" "r.get('type')=='message' and 'x-justfit-ignored-tools: web_search_20250305' in h" \
  "$(post /v1/messages "$(body '{"model":"%s","max_tokens":32,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Say ok."}],"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":1}]}')")"

# Ollama API
check "ollama version" "'version' in r" "$(curl -sS -m 10 "$URL/api/version")"
check "ollama tags" "any(m['name']=='$MODEL' for m in r['models'])" "$(curl -sS -m 10 "$URL/api/tags")"
check "ollama show" "'completion' in r['capabilities']" "$(post /api/show "$(body '{"model":"%s"}')")"
check "ollama chat (non-stream)" "r['done'] and r['message']['content']" \
  "$(post /api/chat "$(body '{"model":"%s:latest","stream":false,"think":false,"messages":[{"role":"user","content":"Say ok."}],"options":{"num_predict":32}}')")"
check "ollama chat (stream)" "isinstance(r,list) and r[-1]['done'] and ''.join(x.get('message',{}).get('content','') for x in r)" \
  "$(post /api/chat "$(body '{"model":"%s","think":false,"messages":[{"role":"user","content":"Count to three."}],"options":{"num_predict":32}}')")"
check "ollama generate" "r['done'] and r['response']" \
  "$(post /api/generate "$(body '{"model":"%s","stream":false,"think":false,"prompt":"Say ok.","options":{"num_predict":32}}')")"
check "ollama tool call" "r['message'].get('tool_calls') and r['message']['tool_calls'][0]['function']['name']=='get_weather' and isinstance(r['message']['tool_calls'][0]['function']['arguments'],dict)" \
  "$(post /api/chat "$(body '{"model":"%s","stream":false,"think":false,"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')")"

# Agents: fix add() in calc.py, then run it.
TASK='calc.py has a bug. Read it, fix it, then run python3 -c "from calc import add; print(add(2,3))" and tell me the output.'
agent() { # name, command...
  local name="$1"; shift
  local dir="$WORK/$name"; mkdir -p "$dir"; printf 'def add(a, b):\n    return a - b\n' > "$dir/calc.py"
  local start=$SECONDS
  (cd "$dir" && timeout "$AGENT_TIMEOUT" "$@" < /dev/null > "$dir/out.txt" 2>&1)
  if grep -q 'return a + b' "$dir/calc.py"; then report PASS "$name" "$((SECONDS - start)) s"
  else report FAIL "$name" "$(tail -c 300 "$dir/out.txt" | tr '\n' ' ')"; fi
}
if command -v claude >/dev/null; then
  agent "claude code (justfit launch)" "$JUSTFIT" launch claude --url "$URL" -- -p --permission-mode bypassPermissions "$TASK"
else report SKIP "claude code (justfit launch)" "claude not installed"; fi
if command -v codex >/dev/null; then
  agent "codex (justfit launch)" "$JUSTFIT" launch codex --url "$URL" -- exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "$TASK"
  # Codex's own defaults (web search on): the hosted tool must be dropped, not fail the request.
  agent "codex (default web_search on)" codex -c 'model_provider="justfit"' -c 'model_providers.justfit.name="JustFit"' \
    -c "model_providers.justfit.base_url=\"$URL/v1\"" -c 'model_providers.justfit.wire_api="responses"' \
    -c "model=\"$MODEL\"" exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "$TASK"
else report SKIP "codex (justfit launch)" "codex not installed"; fi
if command -v ollama >/dev/null; then
  out="$(OLLAMA_HOST="$URL" timeout "$AGENT_TIMEOUT" ollama run "$MODEL:instant" 'Reply with the single word: ready' 2>&1)"
  if printf '%s' "$out" | grep -qi ready; then report PASS "ollama cli run"; else report FAIL "ollama cli run" "$(printf '%s' "$out" | tail -c 200)"; fi
else report SKIP "ollama cli run" "ollama not installed"; fi

exit $((failures > 0))
