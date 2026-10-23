# Native runtime distribution

One source project and Python distribution contain all three reference presets.
The native runtime is a separate macOS arm64 asset. A single asset may serve
multiple chips only after actual compatibility checks; do not label an untested
machine compatible because its architecture matches. Models remain separate
pinned downloads. Device-specific CoreAI caches are not runtime distribution files.

## Use a reviewed asset

```bash
justfit serve --runtime-bundle /absolute/path/runtime.zip --runtime-sha256 <published-sha256>
```

Or pass an HTTPS download manifest and its independently published SHA256:

```bash
justfit serve --runtime-manifest https://example.org/release/runtime-manifest.json --runtime-sha256 <published-manifest-sha256>
```

The URL above illustrates the interface; it is not a download location.
The manifest binds the software version, source manifest, architecture, minimum
macOS, archive SHA256 and five output hashes. The CLI checks both the download
manifest and the ZIP's metadata, rejects unexpected/duplicate/nonregular entries,
and verifies every output before installation. It never executes an installer
from an archive. HTTPS is required even after redirects. Checksums require a
trusted publication channel; they do not establish publisher identity themselves.

This candidate has **no configured public default native download**. With no
explicit asset and no valid prior build receipt, `justfit serve` still builds
from source using compatible Xcode/CoreAI and Metal tools. Explicit asset failures
stop setup rather than silently falling back to compilation. Keep ANE vision
provisioning separate; these five files do not install compiled vision graphs.

## Build a runtime asset

After building and verifying this exact source package on a Mac:

```bash
python scripts/package_runtime.py --output /absolute/path/outside/project/runtime.zip
```

The builder copies only the native executable, two Metal libraries and two
helper dylibs. It normalizes helper install names and checks native dependencies
for local paths. The asset is bound to the complete package manifest, so changing
source or docs requires a new matching asset. The current asset floor is macOS
27.0 because the full helper set includes CoreAI; this is deliberately narrower
than the separately documented GPU source floor. Qualification on one OS does
not qualify every future OS release.

Default signing is ad-hoc **for local qualification only**. Public native assets
should use the publisher's Developer ID and a reviewed signing/notarization
workflow. `--sign-identity` selects a signing identity; the script does not upload
to Apple or claim notarization. Do not advertise ad-hoc test assets as notarized.

A hosted download manifest extends the ZIP's `runtime.json` with:

```json
{
  "url": "https://example.org/release/runtime.zip",
  "archive_sha256": "<sha256-of-zip>"
}
```

Retain the other metadata fields unchanged. Publish its SHA256 through the
trusted release channel and give that value to `--runtime-sha256`. No signing
credentials, local build paths or private deployment receipts belong in assets.

## Qualification and publication

Use the same exact source wheel and native ZIP on each supported configuration.
Install in a fresh Python environment, verify integrity and dependencies, start
without a compiler, check text/image/stream/thinking-budget behavior and the
configuration-specific path, then restart and check build reuse. Source-build
acceptance and prebuilt-install acceptance are distinct receipts.

Only after those gates should a publisher configure an official release URL,
freeze version/contact/project metadata, test the real TestPyPI download and
publish to PyPI/GitHub. The release must state the model license separately and
retain all third-party notices. Installation must not register production services.
