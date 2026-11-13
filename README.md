<div align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="docs/assets/justfit-banner-mobile.svg">
    <img src="docs/assets/justfit-banner.svg" alt="JustFit — long-context LLM serving on Apple silicon" width="100%">
  </picture>

  [![Version](https://img.shields.io/badge/version-2.6.0-087f78)](CHANGELOG.md)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-246a96)](LICENSE)
  [![Apple silicon](https://img.shields.io/badge/platform-Apple%20silicon-526b81)](#install)
</div>

# JustFit 2.6.0

JustFit serves a 27B language-and-vision model (Qwen3.8-27B) on Apple silicon
Macs with 24 GB of unified memory. The `justfit` command and everything it
starts are native (Rust, C++/Metal, Swift); no Python runs at inference time.

## Install

```bash
python3 -m venv ~/.justfit && ~/.justfit/bin/pip install justfit-llm==2.6.0
~/.justfit/bin/justfit doctor
~/.justfit/bin/justfit serve
```

pip only places the files: `justfit` is a native executable. The first
`justfit serve` installs the bundled runtime, downloads the original
`mlx-community/Qwen3.8-27B-4bit` checkpoint and its DFlash2 draft from Hugging
Face, and downloads the ANE vision programs for this chip (about 2.8 GB) from a
pinned, hash-checked index. Later starts reuse them.

Requirements: macOS 27.0 or later on Apple silicon; presets for M4 Pro, M5 Pro
and M6 (`--node m4|m5|m6`, chosen from the chip; at least 24 GB of memory).
About 22 GB of disk for the model, draft and ANE programs, plus the SSD cache
(64 GB quota by default; `--max-cache-disk` changes it, 0 turns it off).
`--no-ane-vision` uses GPU vision and skips the ANE download. Precompiled ANE
programs are published for macOS 27.2.1 (build 26B131); on other macOS builds
the first `justfit serve` compiles them on this Mac, which took 30 minutes to
over an hour in our tests, once (vision runs on the GPU meanwhile).

## What changed since 2.3.1

- All native: the `justfit` command is a native executable and nothing at
  inference time runs Python. pip only places the files; no runtime manifest
  or ANE manifest flags are needed, and `justfit serve` sets up everything on
  first start.
- Original checkpoint: serves `mlx-community/Qwen3.8-27B-4bit` with the
  DFlash2 draft, both downloaded from Hugging Face on first start, instead of
  the `incoai/Qwen3.8-27B-Splash` model pack.
- Draft-tree decode: the draft proposes a small tree of continuations and the
  model checks the whole tree in one pass. It applies to greedy requests on
  all three presets and to sampled requests on M5 Pro and M6, not to
  grammar-constrained (JSON schema) requests. On SPEED-Bench with greedy
  decoding (512 output tokens) decode is 4-14% faster than verifying a single
  draft chain with 1K-token prompts, and 4-7% faster with 16K-token prompts,
  depending on the chip.
- Memory safety: a request in service never takes the host's memory reserve,
  so the system keeps room under full load.
- Fewer SSD writes: progress checkpoints taken while decoding stay in memory;
  the long-prompt recovery point is written only when memory runs short, at
  most once per 16K prompt tokens.
- Multi-turn reuse: a finished request keeps its exact state for the next turn
  that extends it, so agent and chat turns prefill only the new tail.
- `return_token_ids` (vLLM/SGLang-compatible token-in/token-out), crash-request
  quarantine (`400 request_quarantined`), and a sandbox applied by `justfit`
  itself to the engine and native workers (no outbound network).
- ANE vision programs for the chip are found and downloaded automatically on
  first use.

The binaries are signed with a Developer ID; they are not notarized.

With greedy decoding, tree verification almost always gives the same text as
verifying a single draft chain: in the SPEED-Bench runs above, every task
matched on two chips and one of sixteen tasks differed on the third. Sampled
requests keep the same distribution. A multi-turn request that resumes a kept
state can likewise differ slightly from a full recompute.

## Two 160K requests on a MacBook

Two 160K requests at once need almost all the memory a 24 GB Mac can wire
(keep non-swappable). macOS stops the process that holds the most wired memory
when the system-wide total reaches a kernel limit, so what else is wired
decides whether both requests fit.

On Macs without a display attached (our M5 Pro and M6 test machines) this
workload completed in every run of this release. On a 24 GB M4 Pro MacBook
with macOS 27.2.1 it is at the edge: of five runs, one completed, in two
JustFit paused one request near the end to stay within memory, and in two,
both with the display on, macOS stopped the server. With the display on,
macOS 27 wires GPU memory again each time the screen redraws; we measured
about 0.45 GB for a few seconds every minute from the lock screen clock, and
about 0.7 GB held by a moving (Aerial) wallpaper.

For this workload on a MacBook, let the display sleep, use a still wallpaper
and leave GPU-heavy apps closed; that removes the redraw spikes, though a
request may still be paused near the end. A single request up to 256K is not
affected, and shorter pairs leave more room.

## Source, results and paper

This repository is the complete 2.6.0 source. `package-manifest.json` is the
published wheel's file list with SHA-256 pins; every source file here matches
it (this README adds this section to the packaged one). `justfit build`
rebuilds the native runtime from it (Rust/Cargo and Apple's Metal tools
required).
JustFit derives from Splash 1.3.0; see LICENSE, NOTICE and ORIGIN.json.

- 16K speed (SPEED-Bench, eight tasks, 512 output tokens), 2.6.0 against the
  published v2 measurements: prefill 133.9 / 511.5 / 457.5 tok/s
  (−4.6% / +6.7% / +1.9%) and decode 48.3 / 59.5 / 38.5 tok/s
  (+3.8% / +7.3% / +6.6%) on M4 Pro / M5 Pro / M6;
  [protocol and all rows](docs/RESULTS.md).
- [Paper: JustFit v2](paper/README.md)
- [Changelog](CHANGELOG.md)
- 2.3.x (Python package and its documentation) remains at tag
  [`v2.3.1`](https://github.com/YuhuaBillChen/justfit-llm/tree/v2.3.1).

`justfit --help` and `justfit serve --help` list all commands and options.
