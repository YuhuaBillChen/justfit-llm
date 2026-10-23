# Implementation

JustFit includes a native runtime, Metal kernels, a Python frontend and grouped
ANE vision workers. It derives from Splash 1.3.0; see [provenance](../ORIGIN.json)
for upstream identity and attribution. Local model and compiled graph assets
are provisioned separately.

## Build

On Apple silicon with the matching macOS CoreAI SDK and an installed Apple Metal
toolchain (the compiler may be outside xcrun's search path):

```
cd source
make install-environment
cd ..
python3 build_release.py --metal-bin /path/to/Metal.xctoolchain/usr/bin
source/.venv/bin/python -m unittest discover -s tests -v
```

The script builds the host, primary Metal library, eight-kernel 16-row verify
library, CoreAI Swift bridge, and physical/neural memory-attribution library.
It records exact commands, output hashes and compiler hashes in build-receipt.json.
No frozen binary or shader is substituted for a clean source build.

Managed service dependencies are in service/requirements.txt. Inference dependencies
are pinned in source/install/requirements.txt. Run service tests using an environment
containing both sets. `source/dev/native.mk` retains upstream development targets;
the pinned upstream GitHub archive itself omits dev/tests, including engine_cache_test.cpp.
Those unavailable upstream tests are not claimed as passing. The included JustFit
loader, packaging and service tests have independent invocations above/below.

```
python3 -m unittest discover -s service -p 'test_*.py' -v
```

## Assets and routing

Weights, tokenizers, compiled vision graphs and OS-specific CoreAI cache are not
redistributed. Provision each node's separately licensed assets, verify their
hashes, and create its cache catalog offline. Profiles pin the assets, catalog,
worker tools and native helpers. A runtime request never compiles missing graphs.
4K and 8K use an explicit `gpu_relief` policy with per-tier boolean
`phase_qualification` for `prefill` and `decode`. GPU-only text prefill can
offload vision when qualified. With a configured language ANE FFN split, any
active text prefill keeps ANE priority and vision uses GPU; decode-only work can
offload vision. The route is checked again after worker and memory waits. Native
vision ownership still pauses/resumes language FFN if a new prefill starts after
admission. Idle, queue-only, unknown-status, multi-image, failed-admission and
unsupported-size requests use GPU. Each tier admits one image per offload.
The historical `prefill_overlap` policy is retained for old profiles.
For `gpu_relief`, transient startup lease denial defers prewarming instead of
failing the whole service; requests still require a fresh native memory lease
and cached-only placement validation. Asset/pin validation errors still fail startup. A forced calibration frontend is not
part of the serving package. 14.4K remains optional and defaults to GPU.

HTTP disconnects propagate through the managed proxy and are monitored during
synchronous vision preparation. Cancellation reaps the owned worker before
releasing its native memory lease; ordinary text submissions add no monitor thread.

The serving worker imports adjacent coreai_bridge.py and uses adjacent
libcoreai_bridge.dylib, worker_meter.py and attributed_memory.dylib. graph.py
preserves historical exporter provenance; its research/export paths are not a
supported offline graph-generation installer.

## Operations

The managed entry verifies release pins before loading, enforces a 20 GiB disk
floor with 24 GiB reopening hysteresis, caps each child log to 32 MiB plus four
backups, and stops admission at the 128 MiB ANE diagnostic ceiling. APC's 128 GiB
limit is a maximum, not preallocated space. Guards only stop their owned engine
process group and never delete caches. Deployment must preserve the prior launch
configuration and check actual health, routing, restart restoration and rollback.

Preserve source/LICENSE, source/THIRD_PARTY_NOTICES and inline attribution.
