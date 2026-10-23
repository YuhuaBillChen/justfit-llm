# ANE vision eligibility

The release policy is `gpu_relief`, with per-node, per-capacity acceptance.
4K/8K refer to raw patch capacities, not text context lengths. Select the smallest
ready graph that fits the preserved preprocessed geometry.

| Current native scheduler state | Language ANE FFN state | Eligible phase |
| --- | --- | --- |
| Any text prefill, with or without decoding lanes | off | qualified prefill |
| Any text prefill, with or without decoding lanes | split/stopped | GPU fallback |
| Decoding, no text prefill | known off/split/stopped | qualified decode |
| Idle, only queued, unknown status | any | GPU fallback |

Each leaf explicitly sets `phase_qualification: {"prefill": false, "decode": false}`
until its hardware/service acceptance supports the corresponding phase. These
booleans express an operator-reviewed deployment contract, not a speed estimate.
There is no automatic promotion from a standalone encoder benchmark.

A second waiting image or a multi-image request uses the existing native route.
All external workers retain fixed native memory leases, host-reserve admission,
bounded output payloads, cached-only graph loading and actual placement checks.
No request compiles an asset. A transient prewarm admission denial defers the
prewarm; it never bypasses runtime admission. Invalid assets or hashes fail load.

Eligibility is rechecked after serialization and memory waits. Native ownership
is authoritative across subsequent phase changes: language FFN jobs drain before
vision ownership is granted, FFN split remains paused for the lease, and resumes
on release. Cancellation/failure reaps the worker before releasing its lease.

Evaluate text decode continuity and matched text output/throughput alongside image
latency, prefill progress, memory and recovery. A slower isolated ANE encoder can
still reduce interruption of the busy GPU. A single cohort completion difference
is not proof of accelerator contention or of every lane slowing down.
