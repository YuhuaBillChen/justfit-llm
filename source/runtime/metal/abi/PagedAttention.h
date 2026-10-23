// Modified for JustFit: product identity and integration.
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

// stride: rows of one KV head's (and one query group's) chunk staging;
// equals the store's chunk_stride.
struct FullPrefillParams {
  uint32_t tokens;
  uint32_t stride;
};

static_assert(sizeof(FullPrefillParams) == 8,
              "Full attention prefill parameters are 8 bytes on both sides");

// Each lane holds JUSTFIT_TARGET_VERIFY_ROWS rows in a chunk staging of
// JUSTFIT_VERIFY_CHUNK_STRIDE rows.
struct FullDecodeBatchParams {
  uint32_t lanes;
};

static_assert(sizeof(FullDecodeBatchParams) == 4,
              "Full attention verify parameters are 4 bytes on both sides");

// One full-attention layer's paged KV. Every current row is written directly
// into its final page slot before attention. Prefill and verify both read
// all visible history from the same paged representation. Decode accepts rows
// only by advancing committed_tokens; the next command overwrites rejected
// slots. Page tables hold one JustFitKvPage per logical page, and kv places the
// layer in the pool's extents; the host sets it for each layer it encodes.
struct JustFitChunkedPrefillParams {
  uint32_t committed_tokens;
  uint32_t chunk_tokens;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  JustFitKvLayer kv;
};

static_assert(sizeof(JustFitChunkedPrefillParams) == 24,
              "Chunked store parameters are 24 bytes on both sides");

// Prefill divides each query tile's visible Page32 history into balanced
// splits. The same count and partition rule are used by split and reduce.
struct JustFitPrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  JustFitKvLayer kv;
  uint32_t split_count;
};

static_assert(sizeof(JustFitPrefillAttentionParams) == 28,
              "Prefill attention parameters are 28 bytes on both sides");

// A verify lane attends all its JUSTFIT_TARGET_VERIFY_ROWS rows, which its
// queries and output hold in JUSTFIT_VERIFY_CHUNK_STRIDE rows per KV head.
struct JustFitVerifyAttentionParams {
  uint32_t committed_tokens;
  uint32_t page_table_entries;
  JustFitKvLayer kv;
  // Filled from the plan: this lane's history-scaled split count and the
  // plan-wide slot stride that every lane's partials use.
  uint32_t split_count;
  uint32_t slot_splits;
};

static_assert(sizeof(JustFitVerifyAttentionParams) == 24,
              "Verify attention parameters are 24 bytes on both sides");

struct JustFitTQ4StagedPrefillParams {
  uint32_t committed_tokens, rows, chunk_stride, visible_tokens;
  JustFitKvLayer kv;
  uint32_t kv_head;
};
static_assert(sizeof(JustFitTQ4StagedPrefillParams)==28);
