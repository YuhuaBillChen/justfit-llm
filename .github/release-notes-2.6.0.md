# JustFit 2.6.0

Native (Rust, C++/Metal, Swift) serving of Qwen3.8-27B with vision on 24 GB
Apple silicon Macs; no Python at inference time.

    python3 -m venv ~/.justfit && ~/.justfit/bin/pip install justfit-llm==2.6.0
    ~/.justfit/bin/justfit serve

## What's new since 2.3.1
- All native: `justfit` is a native executable and nothing at inference time
  runs Python. pip only places the files; `justfit serve` sets up the runtime,
  model and ANE programs on first start, with no manifest flags.
- Serves the original `mlx-community/Qwen3.8-27B-4bit` checkpoint with the
  DFlash2 draft, instead of the `incoai/Qwen3.8-27B-Splash` model pack.
- Draft-tree decode: the draft proposes a small tree and the model verifies
  it in one pass; on for greedy requests on all three presets and for sampled
  requests on M5 Pro and M6. SPEED-Bench with greedy decoding (512 output
  tokens), against verifying a single draft chain: median per-task decode
  gain +4-14% with 1K prompts and +4-7% with 16K prompts, depending on the
  chip; time to first token within 4%.
- Memory safety under full load: a request in service never takes the host's
  memory reserve.
- Fewer SSD writes: decode checkpoints stay in memory; a long prompt's
  recovery point is written only when memory runs short, at most once per 16K
  prompt tokens.
- Multi-turn reuse: a finished request keeps its exact state for the next turn
  that extends it.
- `return_token_ids` (token-in/token-out), crash-request quarantine
  (`400 request_quarantined`), and a sandbox that `justfit` applies to the
  engine and native workers (no outbound network).
- ANE vision programs for the chip are found and downloaded automatically on
  first use from Hugging Face (pinned, hash-checked index); the wheel is
  about 29 MB. Precompiled for macOS 27.2.1 (26B131); other macOS builds
  compile them once locally (30 minutes to over an hour).

Binaries are signed with a Developer ID; they are not notarized.

## 16K speed against the published v2 numbers
SPEED-Bench, same eight 16K tasks, 512 output tokens, greedy, median:

| Chip | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| M4 Pro | 133.9 (−4.6%) | 48.3 (+3.8%) |
| M5 Pro | 511.5 (+6.7%) | 59.5 (+7.3%) |
| M6 | 457.5 (+1.9%) | 38.5 (+6.6%) |

2.6.0 measured on macOS 27.2.1 with the shipped presets; protocol and the
other systems in docs/RESULTS.md.

## Validated
macOS 27.2.1 (26B131) on M4 Pro (24 GB), M5 Pro (24 GB) and M6 (32 GB, 8 GB
locked for the capacity test):
- Fresh wheel install on all three: ANE programs downloaded from Hugging Face,
  90 of 90 identical to the published ones, no local compile.
- Two simultaneous requests of 128K input (prefix restored from cache) + 32K
  output each, at least 327,680 tokens resident, no pause: M5 Pro and M6.

## Known limits
- Two 160K requests at once on a 24 GB MacBook are at the edge of what macOS
  lets a process wire. On an M4 Pro MacBook, of five runs one completed, two
  paused a request near the end, and in two (display on) macOS stopped the
  server. With the display on, macOS 27 wires GPU memory again on every
  screen redraw. Let the display sleep and use a still wallpaper for this
  workload. Single requests up to 256K are not affected.
- With greedy decoding, tree verification almost always gives the same text
  as single-chain verification; in the SPEED-Bench runs one of sixteen tasks
  differed on one of three chips.
- Draft trees do not yet apply to grammar-constrained (JSON schema) requests.
