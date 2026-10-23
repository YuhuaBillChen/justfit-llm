# Python distribution and acceptance

This candidate provides a source-bearing Python wheel and sdist, version
`2.3.0rc24`. It installs the frontend dependencies and the `justfit` / `justfit-llm` commands.
It does not ship a precompiled native executable, Metal libraries, model weights
or machine-specific ANE caches. Installation does not compile code, download
models, register services or modify existing services.

## Install the local candidate

Use Python 3.12–3.14 in a new virtual environment without
`--system-site-packages`. Do not reuse a model-export or research environment. Until publication, install the
reviewed local wheel; `pip install justfit-llm` from the public index is not yet
an accepted path and might refer to an unrelated or unavailable project.

## One command

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install /absolute/path/justfit_llm-2.3.0rc24-py3-none-any.whl
.venv/bin/justfit serve
```

On first run `serve` creates a workspace under
`~/Library/Application Support/JustFit/<version>`, builds the native runtime with
the Metal toolchain it finds (install one with
`xcodebuild -downloadComponent MetalToolchain` if none is found), downloads and
verifies `incoai/Qwen3.8-27B-Splash` at the qualified revision (about 16 GB),
picks the `m4`, `m5` or `m6` preset from the chip and memory size, and listens on
`127.0.0.1:8000`. Later runs verify workspace sources and compiled-output receipts, then reuse the build and pinned model. A failed integrity check prevents execution. Every step can
be overridden with `--project`, `--metal-bin`, `--model`, `--node` and `--port`.
Automatic selection supports M4, M5 and M6 with at least 24 GiB. Other chips or smaller machines require an explicit `--node`; that override does not establish compatibility.

The source-bearing distribution requires a compatible Xcode/CoreAI SDK as well as the Metal toolchain. It does not download or install developer tools. Missing tools stop setup before model download. It does not include prebuilt executables or machine-specific ANE caches; vision uses GPU until a local ANE profile is provisioned. Runtime-pack `--revision` accepts a full commit SHA, never a moving branch.

## Prebuilt runtime

A separately qualified runtime ZIP can be installed with `--runtime-bundle` and
`--runtime-sha256`, or fetched from an HTTPS `--runtime-manifest` with its pinned
SHA256. The CLI verifies the package identity, architecture, OS floor, archive
members and every output before use, and skips local compilation.
See [runtime distribution](RUNTIME_DISTRIBUTION.md). No public default runtime
URL is configured yet; the no-argument path still builds locally.

## Step by step

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install /absolute/path/justfit_llm-2.3.0rc24-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/justfit-llm --version
.venv/bin/justfit-llm verify
.venv/bin/justfit-llm init ./justfit-workspace
.venv/bin/justfit-llm doctor --metal-bin /absolute/path/to/Metal.xctoolchain/usr/bin
.venv/bin/justfit-llm build --project ./justfit-workspace --metal-bin /absolute/path/to/Metal.xctoolchain/usr/bin
.venv/bin/justfit-llm serve --project ./justfit-workspace --node m5 --model /absolute/path/to/model --port 8000 --dry-run
.venv/bin/justfit-llm serve --project ./justfit-workspace --node m5 --model /absolute/path/to/model --port 8000
```

The wheel is portable Python tooling and source data; the native runtime requires
Apple silicon/macOS 27.0+/Xcode/CoreAI/Metal. `doctor` rejects unsupported native
platforms. Python package installation on Linux does not qualify native execution.
`init` requires a new destination and verifies all source hashes. Reinstalling
or uninstalling the wheel preserves extracted workspaces, models and caches.
`serve` uses the interpreter that installed the CLI, with no hidden source venv.
The child server clears inherited `PYTHONPATH`/`PYTHONHOME` and disables user
site-packages. Keep export dependencies such as PyTorch in a separate environment.
Choose an appropriate node reference/memory budget, not just a matching name.
For models, image assets, API smoke tests and managed production guards follow
[Quickstart](../QUICKSTART.md) and [ANE provisioning](ANE_PROVISIONING.md).

## Build and distribution gates

From a verified source project, create an isolated packaging environment:

```bash
python3.12 -m venv .packaging-venv
.packaging-venv/bin/python -m pip install build twine
.packaging-venv/bin/python -m build --outdir /absolute/path/outside/project/dist
.packaging-venv/bin/python -m twine check /absolute/path/outside/project/dist/*
```

The build copies only source files pinned by `package-manifest.json`; changes
require regeneration via `package_release.py`. Both wheel and sdist must pass
fresh-environment installation, dependency checks, entrypoint/integrity/workspace
checks, uninstall/reinstall, and offline wheel rebuild from sdist.
The exact installed candidate must additionally pass a clean Mac build and
text/image/stream/hard-budget smoke before public installability is claimed.
Do not reuse a running production deployment as proof of the new installation.

TestPyPI upload requires a separately authorized target/account; no upload is
performed by the build. After a reviewed TestPyPI upload, download the exact
version with `pip download --no-deps --index-url https://test.pypi.org/simple/`
and compare the artifact hash before installing it with dependencies resolved
from the normal trusted index. This avoids mixing dependency indexes.
Formal PyPI upload and GitHub/HF publication are separate explicit actions.
Paper and website remain placeholders pending coordinated release.

