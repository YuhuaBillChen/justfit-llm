# Precompiled ANE vision downloads

JustFit uses separate precompiled assets for the 4K and 8K image-patch tiers.
They include the converted graph, compiled ANE program and position table; they
are separate from the Python wheel, native runtime and language-model download.
The reference devices are M4 Pro, M5 Pro and M6. Asset identity includes the exact
chip name and macOS build, not just arm64. Use a compatible asset set; a different
OS build needs new qualification or the explicit offline compilation workflow.

## Install and serve without an ANE compiler

After downloading the reviewed native runtime and matching tier archives, run:

```bash
justfit serve \
  --runtime-bundle /path/to/JustFit-runtime.zip --runtime-sha256 <runtime-sha256> \
  --ane-bundle /path/to/JustFit-ANE-4K.zip --ane-sha256 <4k-sha256> \
  --ane-bundle /path/to/JustFit-ANE-8K.zip --ane-sha256 <8k-sha256>
```

Filenames and checksums above illustrate the interface; use the actual matching
assets and independently published hashes. The CLI installs the runtime, obtains
the pinned model, verifies both vision bundles, validates cached ANE loads, creates
a local profile, and starts the server. Later `justfit serve` runs reuse that
profile. Use `justfit serve --no-ane-vision` to select GPU vision explicitly.
Python 3.12 is the reference interpreter used for qualification.

A publisher can provide an HTTPS index for automatic device/tier selection:

```bash
justfit serve --ane-manifest <https-index-url> --ane-manifest-sha256 <index-sha256>
```

Supply the native runtime options as well when it has not already been installed.
The index selects both tiers for the detected chip and OS build. This document
does not claim that a public download URL is live.

For an existing built workspace, `justfit ane install` accepts the same ANE options
and `--project`, `--model`, and `--node`. Installing one tier is allowed, but both
4K and 8K must be present before the automatic two-tier profile is enabled.

## Integrity and behavior

Every archive and member has a SHA256 binding. Imports reject traversal, links,
unknown files, wrong chip/OS/weights/worker source or helper ABI and excessive archive sizes. Compiled
cache directories are published atomically and exclusively. Existing different
cache files are preserved and cause an error. The installer does not invalidate
or clear shared caches, execute archive scripts, import Torch or compile graphs.
It rebuilds user-specific path records and verifies actual full-ANE/no-GPU cached
loads before enabling a profile. An OS update or stale cache fails explicitly;
reinstall a compatible asset set or use offline provisioning.

Graph compatibility binds worker/helper source and ABI, so changing only a native
code signature does not require recompiling the model. Native binary hashes remain
verified by the runtime receipt, and local profiles pin the actual installed helpers.

The profile carries reference-device route/memory qualification and local cached
load validation. It is not a new per-user throughput benchmark. Runtime memory
admission and GPU fallback still apply. M6 reserves language ANE during text
prefill; vision ANE eligibility there is decode-only. Unsupported image shapes
and unavailable resources use the GPU according to the existing routing policy.

Converted vision weights retain their upstream Apache-2.0 license and attribution
in each bundle. Publisher signing/notarization of the executable and dylibs is a
separate part of native runtime distribution.

## Build a reference bundle

After completing both-tier routing qualification on the reference device:

```bash
python scripts/package_ane.py --assets-root <qualified-assets-root> \
  --capacity 4096 --node m4 --model <model-directory> \
  --qualification <qualification.json> --output <new-output.zip>
```

Repeat for 8192. The builder binds current worker/helper hashes and the actual
compiled files, scans payloads for private identifiers and retains provenance.
Keep the original qualification evidence; publish only the neutral asset archives
and their checksums, not internal reports or machine-specific catalogs.
