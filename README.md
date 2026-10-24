<div align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="docs/assets/justfit-banner-mobile.svg">
    <img src="docs/assets/justfit-banner.svg" alt="JustFit — long-context LLM serving on Apple silicon" width="100%">
  </picture>

  [![Version](https://img.shields.io/badge/version-2.3.0-087f78)](CHANGELOG.md)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-246a96)](LICENSE)
  [![Apple silicon](https://img.shields.io/badge/platform-Apple%20silicon-526b81)](#supported-setup)

  **[Get started](#get-started) · [Results](#measured-results) · [Documentation](#documentation) · [简体中文](README.zh-CN.md)**
</div>

JustFit is a local inference system for serving a **27B language-and-vision model on memory-constrained Macs**. It coordinates compressed context, a resident speculative draft, concurrent requests and CPU/GPU/ANE work under one unified-memory budget.

**[2.3.0 is available on GitHub Releases](https://github.com/YuhuaBillChen/justfit-llm/releases/tag/v2.3.0)** with source, Python packages, an Apple-notarized native runtime and six ANE bundles. Install from [PyPI](https://pypi.org/project/justfit-llm/2.3.0/) using the instructions below.

## Why JustFit

- **Keep long histories alongside the draft.** Warm-prefix tests reach a 320K aggregate page capacity across two live requests while retaining the speculative draft. This is a tested B2 bound, not a 320K single-request or cold-prefill guarantee.
- **Make concurrent work useful.** Bound transient workspaces, share compressed pages and preserve committed state across scheduling and cold restarts.
- **Put idle accelerators to work.** Use CPU-assisted prefill on M4 Pro, GPU A8 paths on M5 Pro/M6, and qualified ANE work when execution phase and memory allow it. Vision offload aims to leave GPU capacity for text; it does not guarantee lower image latency.
- **Serve through a familiar API.** OpenAI-compatible chat, images, streaming, structured output and native-enforced thinking budgets.

## Measured results

<a href="docs/assets/system-comparison.png"><img src="docs/assets/system-comparison.png" alt="Four-system comparison on M4 Pro, M5 Pro and M6: tested retained context, 16K prefill and 16K decode rates" width="100%"></a>

**Capacity and speed are separate experiments.** Solid bars show tested B1 bounds; hatching extends them to the B2 total. Both JustFit generations reach a 320K page capacity using their respective warm-prefix protocols; v2 retains its draft. Speed uses eight fixed SPEED-Bench 16K tasks with 512 forced output tokens. Weight formats, decoders and profiles differ. M4 Pro/M5 Pro have physical 24 GiB; M6 has physical 32 GiB, with 8 GiB locked only for the v1/v2 B2 capacity tests. M6 speed runs use physical 32 GiB.

At 16K, v2's median decode rate is **2.0–2.5× the v1 rate** across these configurations. Relative to upstream Splash 1.3.0, median prefill improves **9–46%**, while singleton decode declines **2–9%**. These whole-system results are not isolated mechanism gains or a chip-generation ranking.

[Protocols, exact rates and limitations →](docs/RESULTS.md)

## Get started

### Supported setup

- **Apple silicon and macOS 27.0+.** Reference configurations: M4 Pro, M5 Pro and M6, with at least 24 GiB. Other chip variants are not individually qualified.
- **Python 3.12 recommended**; the package declares support for Python 3.12–3.14.
- The prebuilt installation below requires no local compilation. Building from source separately requires a **compatible Xcode/CoreAI SDK and Metal compiler toolchain**.
- Internet access and storage for the pinned model pack: approximately **17.4 GB**, plus dependencies, build outputs and caches. The model repository is publicly accessible without a Hugging Face token at the checked revision.

Installing the Python package on Linux does not enable native inference there. The standalone GPU engine's macOS 26.4 source target is separate from this complete macOS 27/CoreAI setup.

### Install the reviewed release

Create a fresh environment and install the pinned release from PyPI:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install justfit-llm==2.3.0
justfit verify
justfit doctor
justfit serve \
  --runtime-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/runtime-manifest.json \
  --runtime-sha256 f759236b3f78766bd41d47c75b399d6bca0f56ba9d6b39dd83c705f74206aa3c \
  --ane-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/ane-index.json \
  --ane-manifest-sha256 fd1b71cfa9d4e785741dc2e0ba4d43f2a4dd556a62c0a61f37838b7f9e3ec99d
```

First run with these options downloads and verifies the prebuilt runtime, matching ANE assets and pinned model, then serves on **`http://127.0.0.1:8000`**. ANE bundles require an exact chip and macOS build `26A434` match; on other builds, omit the two ANE options to use GPU vision. Later runs of `justfit serve` reuse installed components. Ctrl+C stops the foreground server.

A first run without runtime options or an installed runtime uses the source-build path and requires developer tools. [Quickstart](QUICKSTART.md) covers source builds, ports and models.

### Send a request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "incoai/Qwen3.8-27B-Splash",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "max_tokens": 64,
    "reasoning_effort": "none"
  }'
```

The model ID retains the upstream runtime pack's name; the engine and CLI are JustFit. The default model is pinned to revision `9d27070b71f7142c6b6025f03ac011d70a73cb48`.

## How it works

| Mechanism | Responsibility |
| --- | --- |
| **KVExec** | Consume compressed history with bounded reconstruction and rotated-domain attention. |
| **PhaseSwap** | Control component and workspace lifetimes while the speculative draft stays resident. |
| **StateTrans** | Preserve committed KV, recurrent and draft state across scheduling and persistence. |
| **CrossFire** | Coordinate CPU, GPU and ANE work using phase ownership and memory admission. |

### ANE behavior

| Reference preset | Language prefill | ANE vision eligibility |
| --- | --- | --- |
| M4 Pro / `m4` | CPU-assisted GPU | Text prefill or decode |
| M5 Pro / `m5` | GPU A8 | Text prefill or decode |
| M6 / `m6` | GPU A8 + ANE FFN | Decode with no active text prefill |

Vision eligibility requires qualified 4K/8K assets, a matching chip and OS build, and memory admission. **4K/8K count image patches, not text tokens or screen pixels.** Idle requests, unsupported inputs and unavailable resources use GPU. M6 reserves ANE for language prefill before admitting vision work. [ANE setup and download interface](docs/ANE_DOWNLOADS.md).

## Documentation

| I want to… | Start here |
| --- | --- |
| Install, build or choose a port/model | [Quickstart](QUICKSTART.md) · [Python installation](docs/PYPI.md) |
| Understand the measurements | [Results and protocols](docs/RESULTS.md) · [Preview validation](docs/PREVIEW_VALIDATION.json) |
| Change device settings | [Configuration](docs/CONFIGURATION.md) · [ANE policy](ANE_POLICY.md) |
| Prepare ANE assets or a prebuilt runtime | [ANE downloads](docs/ANE_DOWNLOADS.md) · [Offline provisioning](docs/ANE_PROVISIONING.md) · [Runtime distribution](docs/RUNTIME_DISTRIBUTION.md) |
| Deploy a managed service | [Deployment and rollback](docs/DEPLOYMENT.md) · [Broker integration](service/broker-integration/README.md) |
| Control reasoning output | [Thinking budget](docs/THINKING_BUDGET.md) |
| Report a problem or contribute | [Issues](https://github.com/YuhuaBillChen/justfit-llm/issues) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) |

The default endpoint is local. Network deployment needs appropriate authentication and TLS; see [security scope](SECURITY.md). Native runtime sources are in `source/`, ANE tooling in `vision-tools/`, and optional managed-service components in `service/`.

## Paper and citation

**JustFit v2: Long-Context Local LLM Serving with Just-in-Time State Management** — Yuhua Chen, 2026. **[Read the preprint (PDF)](https://github.com/YuhuaBillChen/justfit-llm/blob/paper-v2-2026-10-24-r2/paper/2026-10-24-r2/JustFit-v2.pdf)** · [Paper and citation files](paper/README.md). This dated version remains available as the manuscript evolves; an arXiv link will be added when public.

**Full-set AIME 2026:** on M5 Pro, the same four-bit target with TQ4 KV scores **86/90 (95.6%) with A8 off** and **85/90 (94.4%) with A8 on**, covering all 30 questions at three seeds. Each arm has three capped generations included in the denominator. This tests A8 within the quantized configuration.

If JustFit supports your research, please cite:

```bibtex
@unpublished{chen2026justfitv2,
  author = {Chen, Yuhua},
  title = {{JustFit v2}: Long-Context Local LLM Serving with Just-in-Time State Management},
  year = {2026},
  month = oct,
  note = {Preprint, version 2026-10-24-r2, attribution clarification},
  url = {https://github.com/YuhuaBillChen/justfit-llm/blob/paper-v2-2026-10-24-r2/paper/2026-10-24-r2/JustFit-v2.pdf}
}
```

[Machine-readable citation](CITATION.cff) · [JustFit v1 paper](https://arxiv.org/abs/2609.17475)

## Acknowledgments and license

JustFit's native runtime is derived from **[Splash 1.3.0](https://github.com/incoai/splash)**. Splash already provides language ANE FFN, paged KV tiering and composite-state persistence; JustFit extends their integration, retention and resource coordination. Upstream authors, licenses and implementation credits are preserved in [NOTICE](NOTICE) and [third-party notices](source/THIRD_PARTY_NOTICES). Model weights and derived assets retain their own licenses.

Code is available under [Apache 2.0](LICENSE). JustFit maintainers handle this project's support; please direct JustFit issues here rather than to upstream maintainers.
