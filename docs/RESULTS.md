# Results and measurement protocols

These are whole-system observations from the JustFit v2 research record, not an all-workload speed guarantee. The manuscript remains under author review. This page publishes the figure inputs and the scope needed to interpret them.

## Fixed-output 16K speed

Eight code tasks are fixed in source order from the low-entropy block of [NVIDIA SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench), `throughput_16k/test`, before inspecting outputs. Each request generates 512 tokens with greedy sampling, medium thinking, EOS/custom stops disabled and zero prompt-cache hits. Actual inputs span 17,053–18,253 tokens. Values below are medians over the task cohort.

PP is input tokens divided by client TTFT. TG uses the 511 inter-token intervals after the first generated token. Fixed output length equalizes token work, not trajectories or task quality. Preserved historical mlx-vlm/v1 use MXFP4 and MTP; upstream Splash/v2 use packed Q4 and DFlash. The historical mlx-vlm snapshot is not a claim about current upstream performance.

| Device | System | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | ---: | ---: |
| M4 Pro | mlx-vlm (historical) | 123.1 | 13.3 |
| M4 Pro | JustFit v1 | 120.9 | 19.4 |
| M4 Pro | Splash 1.3.0 | 128.6 | 51.0 |
| M4 Pro | JustFit v2 | 140.4 | 46.5 |
| M5 Pro | mlx-vlm (historical) | 345.6 | 14.5 |
| M5 Pro | JustFit v1 | 338.2 | 22.5 |
| M5 Pro | Splash 1.3.0 | 403.7 | 57.6 |
| M5 Pro | JustFit v2 | 479.6 | 55.5 |
| M6 | mlx-vlm (historical) | 277.1 | 8.2 |
| M6 | JustFit v1 | 273.5 | 18.1 |
| M6 | Splash 1.3.0 | 307.2 | 36.9 |
| M6 | JustFit v2 | 449.1 | 36.1 |

The displayed v2 cohort is refreshed against the preceding runtime using alternating paired tasks: 24 paired outputs agree across the three devices and median PP/TG changes are within 1%. These speed checks use a 21,631 MiB native allocation setting. A one-prompt check at 22,015 MiB is not a replacement speed distribution. M4 Pro's upstream Splash arm uses a 1 GiB host reserve after startup failures at the original default; M5 Pro/M6 retain 2 GiB upstream reserves. Early trials partly overlapped dependency/asset preparation. These are not controlled background-I/O or single-factor ablations.

## Retained-context bounds

K means 1,024 positions. A tested bound establishes successful execution at that point, not a searched maximum or long-context answer quality.

| System | B1 tested positions | B2 aggregate page capacity | Meaning |
| --- | ---: | ---: | --- |
| mlx-vlm (historical) | 32K | — | Preserved baseline's tested bound; not a theoretical maximum. |
| JustFit v1 | 256K | 320K | Ordered warm prefixes; concurrent autoregressive decoding. |
| Splash 1.3.0 | 73,721 (about 72K) | — | Recorded bound under that configuration; not a universal hardware limit. |
| JustFit v2 | 256K | 320K | Warm prefixes; resident speculative draft during concurrent decoding. |

V1 B2 uses `(160K + 16K) + (128K + 16K)` with a 21,000 MiB process-group guard; its final simultaneous logical sample is 327,677 positions, while page high-water reaches 327,680. V2 B2 uses `(128K + 32K) × 2`, a 22,015 MiB native budget and demand-allocated persistence staging. Initialization, guards and output shapes differ. Neither is a cold 320K or single-request 320K claim. Historical B1 and stock/mlx tests retain their own protocols.

M4 Pro and M5 Pro have physical 24 GiB. M6 has physical 32 GiB; only the final v1/v2 B2 capacity tests hold an independently checked 8 GiB locked reservation. That approximates available RAM, including compression/swap in the remainder; it does not reproduce all behavior of a physical 24 GiB machine. M6's speed runs use physical 32 GiB. Device-specific A8/CPU/ANE/lookup profiles also differ, so the rows do not rank chip generations.

## What the comparison supports

- Both JustFit generations retain a tested 320K B2 page capacity; v2 does so with its speculative draft resident.
- At 16K, v2 median decode is approximately 2.0–2.5× v1 across these complete stacks. Their different weights/decoders prevent attributing that ratio to any individual feature.
- Relative to Splash, v2 median PP changes by +9.2%, +18.8% and +46.2%; TG changes by −8.8%, −3.6% and −2.2%, for M4 Pro, M5 Pro and M6 respectively. These compare cohort medians, not medians of per-task percentage changes.
- Quality subsets, historical development progression and final installation checks answer separate questions. No broad quality-equivalence claim follows from equal generated lengths or passing installation tests.

[Figure input CSV](system-comparison.csv) · [Preview installation validation](PREVIEW_VALIDATION.json) · [Configuration](CONFIGURATION.md)
