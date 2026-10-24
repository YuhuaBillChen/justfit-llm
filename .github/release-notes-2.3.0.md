# JustFit 2.3.0

Local 27B language-and-vision serving for Apple silicon, with compressed context,
a resident speculative draft, native thinking budgets and coordinated CPU/GPU/ANE work.

## Install

Use Apple silicon, macOS 27.0+ and native arm64 Python 3.12 in a fresh environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ./justfit_llm-2.3.0-py3-none-any.whl
justfit serve \
  --runtime-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/runtime-manifest.json \
  --runtime-sha256 f759236b3f78766bd41d47c75b399d6bca0f56ba9d6b39dd83c705f74206aa3c \
  --ane-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/ane-index.json \
  --ane-manifest-sha256 fd1b71cfa9d4e785741dc2e0ba4d43f2a4dd556a62c0a61f37838b7f9e3ec99d
```

Download the wheel from this release before running the local-file installation.
The model pack is a separate pinned download, approximately 17.4 GB.
After first setup, `justfit serve` reuses the installed runtime, model and ANE profile.
The default endpoint is `http://127.0.0.1:8000`.

The first-run command above explicitly selects verified prebuilt downloads.
Without these options and without an existing runtime receipt, `justfit serve`
uses the source-build path and requires compatible developer tools.

## Assets and validation

- One source ZIP, Python wheel and sdist contain the three reference presets.
- One macOS arm64 runtime contains the engine, two Metal libraries and two helpers.
  Its three executables have Developer ID signatures and Apple Accepted notarization.
- Six ANE ZIPs supply 4K/8K image-patch tiers for M4 Pro, M5 Pro and M6.
  They require the exact chip name and macOS build 26A434; other OS builds require
  separately qualified caches or GPU vision.
- Exact source/wheel/sdist/runtime installation and text/image/stream/thinking-budget/
  cold-restart checks passed on all three reference devices. Both ANE tiers passed
  cached full-tower output comparison and actual routing checks on each device.
- SHA256SUMS binds all artifacts. Source tests: 113; service tests: 15;
  installation lifecycle: 30 checks per platform qualification.

The wheels contain no weights or native binaries. No production service is registered
by installation. Device-specific ANE assets retain upstream weight licensing and notices.
Developer ID signatures identify YUHUA CHEN, Team ID 68M2VQ7A96.

The software is Apache-2.0 with preserved Splash 1.3.0 and third-party attribution.
The evolving [JustFit v2 preprint](https://github.com/YuhuaBillChen/justfit-llm/tree/main/paper)
has a separate manuscript version history. Please cite it when using JustFit in research.
