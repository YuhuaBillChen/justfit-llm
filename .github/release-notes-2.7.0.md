# JustFit 2.7.0

Native serving of Qwen3.8-27B with vision on Apple silicon Macs: now also on
16 GB Macs, with less memory on M5 Pro and M6, a faster first token on M4 Pro,
and more requests on draft trees.

    python3 -m venv ~/.justfit && ~/.justfit/bin/pip install justfit-llm==2.7.0
    ~/.justfit/bin/justfit serve

## What's new since 2.6.0
- 16 GB Macs: `justfit serve --node m6-16` serves Bonsai 2 27B
  (`prism-ml/Ternary-Bonsai-2-27B-gguf`, PTQ1_0) from Hugging Face in a
  12.1 GiB budget, up to 221,169 tokens of context, with vision.
- GDN state compression based on LeapQuant (INT8 + rank-4 recurrent state,
  shared window keys), with draft-tree verification kept fast. Two-request
  peak memory is about 490–500 MiB lower than 2.6.0 on M5 Pro and M6.
- TQ4 KV cache: the first page and the most recent pages stay in BF16 and the
  rest store norm-corrected 4-bit codes. In our offline measurement on
  captured attention inputs, attention output error against BF16 dropped from
  7.8% to 2.9%.
- Draft trees now also serve requests with repetition, presence or frequency
  penalties, and JSON-schema constrained output. Decode speed on such
  requests (greedy, against verifying a single chain): +4–18% on M4 Pro to M6,
  +48% with Bonsai 2 on 16 GB.
- M4 Pro: the text prefill FFN split on the Neural Engine is back on (2.6.0
  left it off): time to first token on a 16K prompt 412 s → 294 s (−29%).
- Clients: Ollama API, `:instant` model names (answer without thinking),
  hosted tools such as web search are dropped and disclosed in `/v1/responses`
  instead of rejected, and `justfit launch claude|codex`.
- First download: large files download as 16 MiB ranges over eight parallel
  connections, each resumable, instead of one stream (the Hub slows each
  connection to a few MB/s after its first ~64 MB; in our test a 3.8 GB file
  took about 100 s instead of about 15 minutes). An
  interrupted download is no longer reported as a missing `config.json`;
  network errors say what failed, are retried for about five minutes, and
  finished bytes are kept (#1).
- ANE vision: on macOS builds without precompiled programs, images run on the
  GPU; `--ane-compile` opts in to compiling them on this Mac (#2).

Binaries are signed with a Developer ID; they are not notarized.

## Measured against 2.6.0
Same Macs, same session, shipped presets, macOS 27.2.1:

| | M4 Pro | M5 Pro | M6 (24 GB test) |
| --- | ---: | ---: | ---: |
| Two-request peak memory | +238 MiB¹ | −492 MiB | −500 MiB |
| 16K time to first token | −29% | ±2% | +5% |
| 16 held-out MATH problems | 13 → 13 | 13 → 14 | 13 → 13 |

¹ M4 Pro runs prefill FFN layers on the Neural Engine again in 2.7;
that costs about 730 MiB and buys the −29% time to first token. Without it,
GDN compression alone would be −492 MiB on M4 Pro too.

Two-request decode speed varies with the request (−15% to +21% across the
pairs we ran); single-request decode is unchanged or faster.

## Validated
macOS 27.2.1 (26B131) on M4 Pro (24 GB), M5 Pro (24 GB) and M6 (32 GB, 8 GB
locked to act as a 24 GB Mac); 16 GB measured on M6 with 16 GB locked:
- Fresh wheel install: ANE programs 90 of 90 identical to the published ones,
  no local compile; text and images correct (M4 Pro, M5 Pro, M6).
- Two simultaneous requests of 128K input (prefix restored from cache) + 32K
  output each, 327,680 tokens resident, no pause: M4 Pro, M5 Pro, M6.
- 16 GB first start from an empty Mac (no workspace, no cached model):
  `justfit serve --node m6-16` downloaded Bonsai 2, its vision projector and
  the draft (10.7 GB) and was ready in about 5 minutes; text and image
  correct, 221,169 tokens of context.
- 16 GB: a 212,607-token needle with 4,096 output tokens, alone and with two
  chats and an image served meanwhile, under a 3 GB foreground load.

## Known limits
- With GDN compression, greedy draft-tree decoding no longer gives
  byte-identical text to single-chain decoding; answers on our math set match
  or improve.
- Draft trees still fall back to a single chain when three or more requests
  decode at once, and on M4 Pro sampled requests use a single chain.
- Two 160K requests at once on a 24 GB MacBook remain close to what macOS
  lets a process wire; the M4 Pro MacBook passed this test with 2.7.0
  (it failed in some 2.6.0 runs).
