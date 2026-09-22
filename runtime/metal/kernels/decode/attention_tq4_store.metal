#include "metal/kernels/common/tq4_store_row.h"

kernel void verify_attention_tq4_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device uchar *model_pages [[buffer(2)]],
    device const uint *page_table0 [[buffer(3)]],
    device const uint *page_table1 [[buffer(4)]],
    device const uint *page_table2 [[buffer(5)]],
    device const uint *page_table3 [[buffer(6)]],
    constant SplashTQ4StoreParams *params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * SplashTQ4KVHeads;
  const uint lane = group / GroupsPerLane;
  const uint local_group = group % GroupsPerLane;
  constant SplashTQ4StoreParams &lane_params = params[lane];
  constant SplashChunkedPrefillParams &chunk = lane_params.chunk;
  if (!splash_chunk_contract_valid(chunk) || chunk.chunk_tokens != Rows ||
      lane_params.layer_index >= SplashTQ4AttentionLayers ||
      lane_params.reserved0 || lane_params.reserved1 || lane_params.reserved2 ||
      thread_index >= SplashTQ4HeadDimension)
    return;
  device const uint *page_table =
      lane == 0 ? page_table0
                : (lane == 1 ? page_table1
                             : (lane == 2 ? page_table2 : page_table3));
  const ulong lane_stride = ulong(SplashTQ4KVHeads) * chunk.chunk_stride *
                            SplashTQ4HeadDimension;
  chunk_keys += lane * lane_stride;
  chunk_values += lane * lane_stride;
  const uint tensor_rows = Rows * SplashTQ4KVHeads;
  const bool value_tensor = local_group >= tensor_rows;
  const uint row = value_tensor ? local_group - tensor_rows : local_group;
  const uint chunk_token = row % Rows;
  const uint head = row / Rows;
  threadgroup float vector[SplashTQ4HeadDimension];
  threadgroup float partial_sums[8];
  threadgroup float stored_norm;
  splash_tq4_store_row<4>(
      chunk_keys, chunk_values, model_pages, page_table, lane_params, vector,
      partial_sums, stored_norm, value_tensor, head, chunk_token, thread_index,
      simd_lane, simd_group);
}
