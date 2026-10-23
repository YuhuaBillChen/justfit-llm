// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/tq4_store_row.h"

kernel void verify_attention_tq4_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const JustFitKvPage *page_table0 [[buffer(2)]],
    device const JustFitKvPage *page_table1 [[buffer(3)]],
    device const JustFitKvPage *page_table2 [[buffer(4)]],
    device const JustFitKvPage *page_table3 [[buffer(5)]],
    constant JustFitChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * JustFitTQ4KVHeads;
  const uint lane = group / GroupsPerLane;
  const uint local_group = group % GroupsPerLane;
  constant JustFitChunkedPrefillParams &chunk = params[lane];
  if (!justfit_chunk_contract_valid(chunk) || chunk.chunk_tokens != Rows ||
      thread_index >= JustFitTQ4HeadDimension)
    return;
  device const JustFitKvPage *page_table =
      lane == 0 ? page_table0
                : (lane == 1 ? page_table1
                             : (lane == 2 ? page_table2 : page_table3));
  const ulong lane_stride = ulong(JustFitTQ4KVHeads) * chunk.chunk_stride *
                            JustFitTQ4HeadDimension;
  chunk_keys += lane * lane_stride;
  chunk_values += lane * lane_stride;
  const uint tensor_rows = Rows * JustFitTQ4KVHeads;
  const bool value_tensor = local_group >= tensor_rows;
  const uint row = value_tensor ? local_group - tensor_rows : local_group;
  const uint chunk_token = row % Rows;
  const uint head = row / Rows;
  threadgroup float vector[JustFitTQ4HeadDimension];
  threadgroup float partial_sums[8];
  threadgroup float stored_norm;
  justfit_tq4_store_row<4>(
      chunk_keys, chunk_values, page_table, chunk, vector,
      partial_sums, stored_norm, value_tensor, head, chunk_token, thread_index,
      simd_lane, simd_group);
}
