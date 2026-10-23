# Provision ANE vision on a new Mac

This workflow exports the 4K/8K graph recipe from local model weights,
compiles on the destination Mac, checks cached placement, creates an isolated
probe profile, and qualifies it before creating a deployment profile. It uses
only files in this project and a separately installed model pack. It does not
download another machine's CoreAI cache or compile during an inference request.

The recipe is fixed: 27 blocks in nine groups of three, FP16, query tile 1024,
block normalization conditioning q8, merger normalization q64, PV range shift
1024, tanh GELU in blocks and erf GELU in the merger. `vision_graph.py` and
`vision_weights.py` implement the graph/packed-weight arithmetic;
`provision.py` supplies the portable entry point. The older `graph.py` is a
historical research utility, not this installer's import.

## Prerequisites and model acquisition

Build the project and install the source environment as in [Quickstart](../QUICKSTART.md).
The exporter uses CoreAI's default macOS 27 asset target; use macOS 27 or later
and a compatible Apple CoreAI SDK for this ANE workflow. The native GPU path has
its separate macOS 26.4 floor. The reference compilation
environment used coreai-torch 0.4.3, coreai-core 1.0.0b3, torch 2.14.1, numpy
2.4.6 and Pillow 12.3.0. That coreai-torch version warns that its vendor-validated
torch ceiling is 2.13.0; 2.14.1 is our measured recipe, not a vendor support claim.
A different compiler/OS/device needs its own acceptance.
Keep a laptop on AC with its lid open throughout compilation and validation.
Reserve a maintenance window: export/compilation can take hours, consumes memory
and disk, and must not run alongside production inference or another compiler.

Model weights are not bundled. The required public runtime pack is
[`incoai/Qwen3.8-27B-Splash` revision 9d27070b71f7142c6b6025f03ac011d70a73cb48](https://huggingface.co/incoai/Qwen3.8-27B-Splash/tree/9d27070b71f7142c6b6025f03ac011d70a73cb48).
Retain its license and installer receipt. This workflow expects that **converted runtime pack**,
not raw Hugging Face safetensors: `target/`, `draft/`, `tokenizer/config.json` and
`vision/model.bin`. The reference model pack is Qwen3.8-27B; its vision file
SHA256 is `8973858e75ec3464f626d5f9111dbceda58f3bac5737ea5693767686003c65df`.
Other weights are not covered by that acceptance. Check your local file with:

```bash
shasum -a 256 /absolute/path/to/model/vision/model.bin
```

From the project directory, set paths for **this machine** (spaces are supported):

```bash
PROJECT="$PWD"
MODEL="/absolute/path/to/converted/model"
ASSETS="$PROJECT/local-state/ane-assets-v1"
NODE="m4"                    # choose the actual m4/m5/m6 reference configuration
```

If the runtime pack is not installed, download and verify that exact revision:

```bash
REVISION=9d27070b71f7142c6b6025f03ac011d70a73cb48
"$PROJECT/source/.venv/bin/python" "$PROJECT/source/install/models.py" \
  --models "$PROJECT/local-state/models" --model incoai/Qwen3.8-27B-Splash \
  --revision "$REVISION" prepare
MODEL="$("$PROJECT/source/.venv/bin/python" "$PROJECT/source/install/models.py" \
  --models "$PROJECT/local-state/models" --model incoai/Qwen3.8-27B-Splash \
  --revision "$REVISION" link)"
"$PROJECT/source/.venv/bin/python" "$PROJECT/source/install/models.py" \
  --models "$PROJECT/local-state/models" --model incoai/Qwen3.8-27B-Splash \
  --revision "$REVISION" verify --full
```

The pinned upstream manifest declares the vision hash above and a 930250752-byte
vision file. The model download is separate from this source archive. A newer
upstream assembly in safetensors/GGUF form is not automatically a replacement for
this packed vision format.

## 1. Install an offline export environment

Use Python 3.12. Keep export dependencies separate: the frontend pins numpy 2.5,
whereas coreai-torch 0.4.3 requires numpy below 2.5.

```bash
python3.12 -m venv "$PROJECT/local-state/ane-export-venv"
EXPORT_PY="$PROJECT/local-state/ane-export-venv/bin/python"
"$EXPORT_PY" -m pip install -r "$PROJECT/vision-tools/requirements-export.txt"
"$EXPORT_PY" -m pip check
"$EXPORT_PY" -m pip freeze > "$PROJECT/local-state/ane-export-environment.txt"
```

The serving worker needs numpy and the two local dylibs built by
`build_release.py`; it does not import torch. `source/.venv/bin/python` is the
worker interpreter in the examples. Do not resolve that venv symlink to a base
interpreter when writing a profile.

## 2. Export and compile offline

Use a new asset directory. Export refuses to overwrite an existing group. Each
command is a separate process, releasing exporter allocations between groups.
Compilation is explicit; its receipt requires actual full ANE/no GPU placement.

```bash
set -e
for capacity in 4096 8192; do
  for group in 0 1 2 3 4 5 6 7 8; do
    "$EXPORT_PY" "$PROJECT/vision-tools/provision.py" export \
      --root "$ASSETS" --model "$MODEL" --capacity "$capacity" --group "$group"
    "$EXPORT_PY" "$PROJECT/vision-tools/provision.py" compile \
      --root "$ASSETS" --capacity "$capacity" --group "$group"
  done
  "$EXPORT_PY" "$PROJECT/vision-tools/provision.py" catalog \
    --root "$ASSETS" --capacity "$capacity"
  "$PROJECT/source/.venv/bin/python" "$PROJECT/vision-tools/provision.py" check \
    --root "$ASSETS" --capacity "$capacity"
done
```

Stop on any failed command; do not run the next stage after a failure. Run this
loop from a shell with `set -e`, or check each exit code explicitly. A completed
group can be retained: resume at the first missing group rather than re-exporting
successful groups. Run `catalog` only when all nine groups for that tier exist.
An existing catalog is immutable; preserve it and use a new versioned asset root
when rebuilding. A compile receipt alone is not numerical or task qualification.

Outputs per tier are `assets/group-00` through `group-08`, `positions.npy`, export
and compile receipts, and `catalog.json`. The catalog records the destination OS
build, graph identity, compiled file metadata and placement manifest hashes.
The CoreAI specialization itself lives in this user's system cache. Raw catalogs contain local paths and must not be copied between users/Macs.
Use [precompiled asset installation](ANE_DOWNLOADS.md) to import a qualified
graph/cache bundle and regenerate the destination records. An OS update, changed cache or
changed asset requires offline rebuilding/rechecking and new qualification.

## 3. Create a probe profile and launch an isolated server

```bash
"$PROJECT/source/.venv/bin/python" "$PROJECT/vision-tools/provision.py" probe-profile \
  --root "$ASSETS" --model "$MODEL" --node "$NODE" \
  --worker-python "$PROJECT/source/.venv/bin/python"
```

This writes `probe/tiered.json` and two pinned leaves marked
`unqualified_probe_only`. They request the intended routing policy so the finite
checks can exercise it; **they are not production qualification**. Run the
candidate on an unused localhost port during the maintenance window:

```bash
PORT=8000                    # choose an unused local port
"$PROJECT/source/.venv/bin/python" "$PROJECT/scripts/run_local.py" \
  --node "$NODE" --model "$MODEL" --port "$PORT" \
  --state-dir "$PROJECT/local-state/ane-probe" \
  --vision-profile "$ASSETS/probe/tiered.json"
```

For `m6`, language ANE FFN is a separate native MIL/ANE path, enabled by the M6
preset. Its first startup can compile and calibrate FFN programs, recording them
under `justfit-ane` in macOS's per-user Darwin cache directory; later starts reuse
that cache. This is separate from the CoreAI vision specializations above. Wait
for native startup to finish before testing. `/status` must show the actual
`ane_ffn` outcome; a healthy GPU fallback is not evidence that FFN offload is on.
The routing checker requires fresh `split_commands` during each M6 prefill probe,
then verifies that vision stays on GPU during that prefill. Do not copy another
Mac's remembered FFN calibration or clear its cache during service.

In a second terminal, repeat the same path variables and run:

```bash
"$PROJECT/source/.venv/bin/python" "$PROJECT/scripts/check_ane_routes.py" \
  --base-url "http://127.0.0.1:$PORT" --node "$NODE" --model "$MODEL" \
  --assets-root "$ASSETS" --meter-dir "$PROJECT/local-state/ane-probe/ane-meter" \
  --output "$PROJECT/local-state/ane-acceptance-v1"
```

This uses the packaged synthetic 800×640 and 1920×1088 OCR fixtures and checks idle GPU,
two real decoding text lanes plus actual 4K/8K ANE, all nine groups fully placed
on ANE without GPU, and text-prefill policy. M4/M5 may offload during text prefill;
M6 must keep vision on GPU while language ANE prefill is active. A result that only
falls back to GPU does not pass the decode check. It also checks Metal health,
request failures, released payload reservations and kernel-reported physical plus
neural worker memory against 1127 MiB. It saves responses/status/samples and only
writes `qualification.json` after these checks pass. Review them before adoption.

These are bounded functional checks, not broad vision quality or throughput
equivalence. Validate your real screenshots and workloads against GPU/reference
answers too. Source tests cover timeout/cancellation/reaping; production deployment
still needs the [deployment checks](DEPLOYMENT.md), including disconnect recovery
and cold restore through the final service entry. Do not enable M6 vision during
text prefill just because an offline graph load succeeds.

## 4. Create and use the qualified local profile

Stop the isolated server with Ctrl+C before changing its profile. Then:

```bash
"$PROJECT/source/.venv/bin/python" "$PROJECT/vision-tools/provision.py" profile \
  --root "$ASSETS" --model "$MODEL" --node "$NODE" \
  --worker-python "$PROJECT/source/.venv/bin/python" \
  --qualification "$PROJECT/local-state/ane-acceptance-v1/qualification.json"
```

The generator checks the receipt's node, weight hash, exact asset/position/catalog
and worker/helper identities, worker memory bound and phase decisions. It writes
`qualified/tiered.json` plus two immutable leaves; all worker/helper/dylib/asset/
position/catalog/receipt files are pinned. This is an operator-reviewed receipt,
not a cryptographic attestation of device testing. Do not invent a passing receipt
or reuse one after changing assets or dependencies. Use the qualified profile with
`run_local.py --vision-profile "$ASSETS/qualified/tiered.json"`, then follow the
managed deployment workflow. Keep the previous service entry and assets available
until the new final-entry checks pass.

## Failure handling

- Missing SDK/import: fix the separate environment or toolchain; do not alter a
  production venv while a model or worker is using it.
- Missing/stale/ambiguous specialization: return to offline compilation/checking.
  Serving is cached-only and must not silently spend hours compiling a request.
- Memory denial: retain GPU fallback; do not increase the accepted worker bound
  or remove host reserve just to make a check pass.
- Image too large: 4K/8K are **raw patch capacities**, not display pixels or text
  context. Larger/unsupported images use GPU with their geometry preserved.
- Failed route, OCR, memory or placement check: keep the raw failure and GPU mode;
  do not promote the probe profile. Correct the cause and use a new evidence directory.
