# Configuration

`config/nodes/{m4,m5,m6}.json` provides engine argument templates and
scalar environment settings. It is a reference consumed by `scripts/run_local.py`,
not a complete managed-service release manifest. Asset paths are resolved locally.

The launcher replaces `{release}`, `{model}`, `{port}`, `{cache}`, sets the local
verify16 path and removes inherited JustFit/DASH/MLX/APC configuration to
avoid accidentally mixing an old configuration into the selected preset.
It sets the cache and ANE meter paths explicitly. Source virtualenv and model
assets must exist before an actual launch; dry-run does not load or download them.

The presets set a 262144-token context ceiling, 128 MiB host
reserve, TQ4 format, 128 GiB disk-cache ceiling and per-node memory limit. A ceiling
is neither preallocated space nor a promise that every concurrent workload fits.
M4 CPU prefill uses fraction .16 / 8 threads; M5/M6 disable CPU prefill.
M5/M6 enable both A8 flags; M4 disables them. A8 is GPU activation quantization,
not a synonym for ANE.

## ANE vision assets

A tiered profile (`schema: 3`, `kind: tiered-vision`) references SHA256-pinned leaf
profiles for raw patch capacities 4096 and 8192. Leaves use `gpu_relief` and
`phase_qualification: {prefill: true, decode: true}` on M4/M5, or
`{prefill: false, decode: true}` on M6. See [policy](../ANE_POLICY.md).
Each leaf must point to its worker Python, metered worker, graph assets, position
table, cache catalog, vision-weight hash and complete `files` hash map. The standard recipe uses 9 groups and a 1181745152-byte native worker lease. Payload accounting is
separate. Runtime loading is cached-only and does not compile missing graphs.

The source workers and bridge are packaged. Model-specific graphs, catalog and
CoreAI cache are not. Follow [new-machine ANE provisioning](ANE_PROVISIONING.md)
to export the fixed recipe from weights, compile offline, check cached placement,
run the finite routing checks, and create a pinned local profile. The portable
entry is `vision-tools/provision.py`; the historical `graph.py` probe retains
research provenance. Each installation must use its own pinned local assets.
Changes to paths or helper binaries require repinning and actual route/placement validation.

4K/8K denotes patches before the vision merge. Typical 1920×1080 input rounds to
1920×1088, or 8160 raw patches, and fits 8K. 2560×1440 is 14400 and defaults to GPU.
Images are not forcibly shrunk to 1080p just to enter the ANE route.
