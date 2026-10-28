# JustFit 2.3.1

A fix release for long-context image requests. Language, vision, scheduling and
memory behaviour are otherwise unchanged from 2.3.0.

## Fix

At long context, an image request could fail with `capacity_exhausted`, or wait on
host memory, although its image and its context each fit: the image's vision tower,
encoder scratch and pixel buffers stayed resident while the engine admitted the
prompt's whole KV target. When no other request holds a lane, images are now encoded
right after their admission and the vision resources are released before KV
admission. A failed early encode falls back to encoding in the prefill chunk as
before. `JUSTFIT_VISION_ENCODE_FIRST=0` restores the previous order.

On an M5 Pro, a recorded agent follow-up with about 215K tokens of history and a
1672×941 image now completes (time to first token 21 s with the history cached)
instead of failing admission.

## Install

Use Apple silicon, macOS 27.0+ and native arm64 Python 3.12 in a fresh environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install justfit-llm==2.3.1
justfit serve \
  --runtime-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.1/runtime-manifest.json \
  --runtime-sha256 2522e5766a48a2048b73d217f1a2aa8107ff6147f7bfb7195f5d75eecbd6e529 \
  --ane-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/ane-index.json \
  --ane-manifest-sha256 fd1b71cfa9d4e785741dc2e0ba4d43f2a4dd556a62c0a61f37838b7f9e3ec99d
```

The model pack is a separate pinned download, approximately 17.4 GB.
After first setup, `justfit serve` reuses the installed runtime, model and ANE profile.
The default endpoint is `http://127.0.0.1:8000`.

The first-run command above explicitly selects verified prebuilt downloads.
Without these options and without an existing runtime receipt, `justfit serve`
uses the source-build path and requires compatible developer tools.

## Assets and validation

- One source ZIP, Python wheel and sdist contain the three reference presets.
- One macOS arm64 runtime contains the engine, two Metal libraries and two helpers.
  Its executables have Developer ID signatures and Apple Accepted notarization.
  Only the engine is rebuilt; the Metal libraries and helpers are byte-identical to 2.3.0.
- The six ANE bundles of [2.3.0](https://github.com/YuhuaBillChen/justfit-llm/releases/tag/v2.3.0)
  apply unchanged; the command above uses their 2.3.0 index.
- Exact source/wheel/sdist/runtime installation and text/image/stream/thinking-budget/
  cold-restart checks passed on all three reference devices (M4 Pro, M5 Pro, M6).
- SHA256SUMS binds all artifacts. Source tests: 113; service tests: 15.

The wheels contain no weights or native binaries. No production service is registered
by installation. Developer ID signatures identify YUHUA CHEN, Team ID 68M2VQ7A96.

The software is Apache-2.0 with preserved Splash 1.3.0 and third-party attribution.
The evolving [JustFit v2 preprint](https://github.com/YuhuaBillChen/justfit-llm/tree/main/paper)
has a separate manuscript version history. Please cite it when using JustFit in research.
