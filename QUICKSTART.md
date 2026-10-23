# Quickstart

Run all commands from this directory. These commands build and run JustFit from
this archive; no global installation or service registration is performed.

## Installed wheel: one command

After installing the reviewed wheel in a clean Python environment:

```bash
justfit serve
```

The first run creates a versioned workspace, builds with an installed compatible
Xcode/CoreAI SDK and Metal toolchain, downloads and verifies the pinned runtime
pack, and starts on `127.0.0.1:8000`. Later runs verify and reuse the build and
model. Missing developer tools stop setup with installation instructions.
This source distribution has no prebuilt engine; local ANE vision assets remain
optional and separately provisioned. See [Python installation](docs/PYPI.md)
for toolchain prerequisites, hardware selection, overrides and limitations.

## Verified prebuilt runtime

To skip compilation, provide a matching runtime ZIP and its published SHA256:

```bash
justfit serve --runtime-bundle /absolute/path/runtime.zip --runtime-sha256 <published-sha256>
```

An HTTPS download manifest is also supported with `--runtime-manifest` and its
pinned `--runtime-sha256`. See [runtime distribution](docs/RUNTIME_DISTRIBUTION.md).
No public default runtime URL is configured in this candidate.

## Source archive: step by step

## 1. Verify the download

Check the archive against its adjacent `SHA256SUMS` before extraction. After
extraction, verify every packaged file:

```bash
python3 validate_package.py
```

A checksum verifies integrity against the supplied manifest, not publisher identity.

## 2. Install dependencies and build on a Mac

Use Apple silicon, macOS 27.0 or later, Python 3.12–3.14 and Xcode with Metal 4.0
and the compatible CoreAI SDK. The complete build includes the ANE Swift bridge.
An installed Metal toolchain may be outside `xcrun`'s normal lookup path.

```bash
make -C source install-environment
source/.venv/bin/python -m pip install -r service/requirements.txt
source/.venv/bin/python -m pip check
python3 build_release.py --metal-bin /absolute/path/to/Metal.xctoolchain/usr/bin
source/.venv/bin/python -m unittest discover -s tests -v
source/.venv/bin/python -m unittest discover -s service -p 'test_*.py' -v
```

Replace the Metal path with the directory containing both `metal` and `metallib`.
`build-receipt.json` records actual compiler and output hashes. It does not certify
a new build's runtime behavior. Compiler versions can produce different binary
hashes; validate the binaries built on your target machine. See
[validation](docs/VALIDATION.md) before a production cutover with another toolchain.

## 3. Prepare a model

Use a separately provisioned compatible model directory containing `target/`,
`draft/` and `tokenizer/`, such as a compatible Qwen3.8-27B package.
Keep its exact weight revisions and hashes with your deployment receipt.
The archive does not redistribute weights. The source installer is available for
new model preparation; inspect its actual arguments before choosing a download:

```bash
source/.venv/bin/python source/install/models.py --help
```

For the required runtime pack, use the pinned download and full-verification
commands in [model acquisition](docs/ANE_PROVISIONING.md#prerequisites-and-model-acquisition).

New model revisions require their own validation; existing receipts do not
apply automatically to arbitrary downloaded models.

## 4. Review and start a local server

Choose the preset matching the intended node configuration. `m4`, `m5`, `m6` are
configuration preset names. They do not auto-detect the chip or its memory size.
The packaged presets specify memory/context limits; review them
in [configuration](docs/CONFIGURATION.md) before running on a different machine.
Choose an unused local port (8000 is only an example):

```bash
python3 scripts/run_local.py --node m5 --model /absolute/path/to/model --port 8000 --dry-run
python3 scripts/run_local.py --node m5 --model /absolute/path/to/model --port 8000
```

The wrapper uses `source/.venv`, the locally built native binary and the selected
preset. It keeps the server in the foreground on localhost. Ctrl+C stops it.
Caches and ANE logs go under this project's `local-state/<node>/` unless
`--state-dir` is supplied. This development entry does not provide managed disk
or log guards. Use the [managed deployment workflow](docs/DEPLOYMENT.md) for production.

By default ANE **vision** is disabled because graphs/cache profiles are not shipped.
M6's separate language ANE configuration remains enabled by its preset. To use
ANE vision on a new Mac, first follow [asset provisioning](docs/ANE_PROVISIONING.md).
For already provisioned, pinned local ANE vision assets, add:

```bash
python3 scripts/run_local.py --node m5 --model /absolute/path/to/model --port 8000 --vision-profile /absolute/path/to/profiles/tiered.json
```

Do not copy another machine's cache or mark an untested profile qualified. Missing
assets or incorrect pins can reject startup; a runtime memory admission denial
uses the qualified GPU fallback behavior.

## 5. Check the API

In another terminal, using the port selected above:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"incoai/Qwen3.8-27B-Splash","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"reasoning_effort":"none"}'
```

Health alone does not verify image routing, cancellation, cache restore or rollback.
The final deployment gates are listed in [DEPLOYMENT.md](docs/DEPLOYMENT.md).

Before production use, review [hard thinking budgets](docs/THINKING_BUDGET.md)
and upgrade the Python frontend and native executable together (protocol 9).
Validate the exact installed frontend/native pair before enabling the service.

If routing through a compatible external node broker, follow the optional
[broker integration](service/broker-integration/README.md) before relying on
full image-bearing conversation history. The direct engine example above does
not use that external broker.

## Python distribution candidate

For the local wheel and `justfit-llm init/build/serve` workflow, see
[Python installation](docs/PYPI.md). This installs dependencies but requires an
explicit Mac native build. Public PyPI/TestPyPI installation and clean Mac
installation acceptance remain separately tracked.
