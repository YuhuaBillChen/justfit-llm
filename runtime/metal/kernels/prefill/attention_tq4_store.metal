#include "metal/kernels/common/tq4_store_row.h"

kernel void prefill_attention_tq4_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device uchar *model_pages [[buffer(2)]],
    device const uint *page_table [[buffer(3)]],
    constant SplashTQ4StoreParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constant SplashChunkedPrefillParams &chunk = params.chunk;
  const uint rows = chunk.chunk_tokens * SplashTQ4KVHeads;
  if (!splash_chunk_contract_valid(chunk) ||
      params.layer_index >= SplashTQ4AttentionLayers || group >= 2 * rows ||
      thread_index >= SplashTQ4HeadDimension)
    return;

  const bool value_tensor = group >= rows;
  const uint row = value_tensor ? group - rows : group;
  const uint chunk_token = row % chunk.chunk_tokens;
  const uint head = row / chunk.chunk_tokens;
  threadgroup float vector[SplashTQ4HeadDimension];
  threadgroup float partial_sums[8];
  threadgroup float stored_norm;
  splash_tq4_store_row<4>(
      chunk_keys, chunk_values, model_pages, page_table, params, vector,
      partial_sums, stored_norm, value_tensor, head, chunk_token, thread_index,
      simd_lane, simd_group);
}
