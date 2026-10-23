// Modified for JustFit: product identity and integration.
#pragma once

// C/Metal ABI constants shared by host and shader compilation. Startup
// static assertions check the corresponding model and operator contracts.
#define JUSTFIT_DRAFT_QUERY_ROWS 8u
#define JUSTFIT_DRAFT_PROPOSAL_TOKENS 7u
#ifndef JUSTFIT_TARGET_VERIFY_ROWS
#define JUSTFIT_TARGET_VERIFY_ROWS 8u
#endif
#define JUSTFIT_VERIFY_PROPOSALS (JUSTFIT_TARGET_VERIFY_ROWS - 1u)
#define JUSTFIT_MAXIMUM_CONTEXT_TOKENS 262144u
#define JUSTFIT_SPECULATIVE_SCRATCH_TOKENS                                  \
  15u
#define JUSTFIT_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (JUSTFIT_MAXIMUM_CONTEXT_TOKENS + JUSTFIT_SPECULATIVE_SCRATCH_TOKENS)
#define JUSTFIT_MAXIMUM_BATCH_WIDTH 4u
#define JUSTFIT_PREFILL_TOKEN_BUDGET 2048u
#define JUSTFIT_DRAFT_SLIDING_WINDOW 2048u
#define JUSTFIT_TARGET_KV_BLOCK_TOKENS 32u
// Rows per KV head (and per query group) of one lane's verify chunk staging:
// one KV block, which holds the lane's JUSTFIT_TARGET_VERIFY_ROWS rows.
#define JUSTFIT_VERIFY_CHUNK_STRIDE JUSTFIT_TARGET_KV_BLOCK_TOKENS
#define JUSTFIT_PREFILL_ATTENTION_TILE_ROWS 8u
#define JUSTFIT_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
// Draft attention deals the live ring tiles of one (lane, KV head)
// round-robin to this many groups, the last of which also attends the eight
// current rows. Fixed rather than derived from the GPU so the combine order,
// and with it the rounding, is the same on every machine and lane count.
#define JUSTFIT_DRAFT_ATTENTION_SPLITS 4u
#define JUSTFIT_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define JUSTFIT_TARGET_SAMPLING_SHARDS 16u
// Threads of each group that selects a sampled row over the whole
// vocabulary (decode_sample_vocabulary*); a bracket of at most this many
// tokens is ordered in threadgroup memory, one token per thread.
#define JUSTFIT_TARGET_VOCABULARY_THREADS 1024u
// Groups that share such a row's draw, each over its own slice of the
// vocabulary, so the eight rows of a lane spread over the GPU's cores.
#define JUSTFIT_TARGET_VOCABULARY_GROUPS 8u
// The ranges of the vocabulary such a row's draw sums: one per simdgroup of
// the row's groups.
#define JUSTFIT_TARGET_VOCABULARY_RANGES                                    \
  (JUSTFIT_TARGET_VOCABULARY_GROUPS * (JUSTFIT_TARGET_VOCABULARY_THREADS / 32u))
#define JUSTFIT_DRAFT_SAMPLING_SHARDS 8u
// Candidates the draft selector keeps per proposal position; acceptance and
// the sampled draw read them.
#define JUSTFIT_DRAFT_CANDIDATES 16u
// The rank of the draft selector's codebooks.
#define JUSTFIT_DRAFT_SELECTOR_RANK 256u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define JUSTFIT_GDN_SCAN_STATE_ROWS 16u
#define JUSTFIT_GDN_SCAN_THREADS 128u
// The epsilon of every RMS norm the kernels compute, which a target's and a
// draft's configurations must state. Unsuffixed, so the host compares a
// configuration's double with it exactly and the kernels take it as a float.
#define JUSTFIT_RMS_EPSILON 1e-6
// Plain norms of at most JUSTFIT_STAGED_NORM_ROWS rows of at most
// JUSTFIT_STAGED_NORM_WIDTH columns run norm_rms_staged, whose 1024-thread
// groups hold a row in threadgroup memory: the region where it measured
// faster than norm_rms (shared/normalization.metal), which covers every
// decode norm of a 2048-wide model and its short prefill chunks.
#define JUSTFIT_STAGED_NORM_WIDTH 2048u
#define JUSTFIT_STAGED_NORM_ROWS 64u
#define JUSTFIT_STAGED_NORM_THREADS 1024u
