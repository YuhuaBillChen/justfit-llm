# Changelog

## 2.3.1

- Fix long-context image requests that could not be admitted: an image's vision
  tower, encoder scratch and pixel buffers were held while the engine admitted the
  prompt's whole KV target, so a large image at about 215K context failed with
  `capacity_exhausted` (or waited on host memory) although each part fits on its
  own. When no other request holds a lane, images are now encoded right after
  their admission and the vision resources are released before KV admission; a
  failed early encode falls back to encoding in the prefill chunk as before.
  `JUSTFIT_VISION_ENCODE_FIRST=0` restores the previous order.

## 2.3.0

- Promote the qualified preview to a stable-version release candidate.
- Distribute the native engine and helpers with Developer ID signatures; verify
  Apple notarization and exact installed artifacts before publication.

- Support a separately pinned local or HTTPS-manifest native runtime, validating
  source identity, platform, archive paths and output hashes before installation.
- Add a relocatable native asset builder; ad-hoc assets are for local qualification
  and do not claim Developer ID signing, notarization or public hosting.

- Align all three reference node presets with the tested 22,015 MiB native
  memory budget; document 21,631 MiB as a conservative local alternative.

- Release persistent-state staging memory while retaining immutable buffers
  through asynchronous writes; preserve direct-to-disk recovery when snapshot
  memory admission is denied.
- Isolate serving imports from inherited Python paths and user site-packages.
- Include a native regression covering state IO ownership, cancellation and
  memory-constrained disk recovery.
- Add citation metadata for the JustFit v2 manuscript.
- `justfit serve` with no arguments creates the workspace, finds the Metal
  toolchain and builds once, downloads the qualified model revision, selects the
  preset from chip and memory, and defaults to port 8000. Adds a `justfit` command alias.

- Use neutral source-distribution owner/group metadata.
- Document installation, configuration and validation for standalone users.

## 2.3 features

- Source-bearing wheel/sdist and explicit workspace, build and serve commands.
- Native per-request thinking budgets, including speculative boundaries.
- Conditional 4K/8K ANE vision routing with device-specific phase eligibility.
- CPU/GPU execution references and language-ANE prefill integration.
- Cancellation propagation, worker lifecycle and resource-admission checks.
- Portable ANE export, offline compilation and local profile qualification tools.
- Managed service disk/log protection and restart port checks.
- Optional external broker history-preservation and health-probe locking adapters.
- Management Host/Origin protection, optional bearer key, bounded uploads and
  updated certificate/image-parser dependencies.
- Synthetic image fixtures with frozen expected answers.

See [validation](docs/VALIDATION.md) for test procedures and limitations.
The dated JustFit v2 preprint is linked from the README and citation metadata.

### ANE installation integrity

Revalidate installed graph assets and compiled cache contents against SHA256 before reuse or native prewarm. Missing, changed, extra, linked or special files are refused without overwriting user caches.
