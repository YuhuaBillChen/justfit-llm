// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_store_row.h"

template <uint KVHeads, typename CacheElement>
inline void justfit_store_verify_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const JustFitKvPage *page_table0, device const JustFitKvPage *page_table1,
    device const JustFitKvPage *page_table2, device const JustFitKvPage *page_table3,
    constant JustFitChunkedPrefillParams *params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  constexpr uint Rows = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * KVHeads;
  uint batch = group / GroupsPerLane;
  uint local_group = group % GroupsPerLane;
  constant JustFitChunkedPrefillParams &lane_params = params[batch];
  if (!justfit_chunk_contract_valid(lane_params) ||
      lane_params.chunk_tokens != Rows ||
      lane_params.chunk_stride != JUSTFIT_VERIFY_CHUNK_STRIDE ||
      thread_index >= JustFitKvHeadDimension)
    return;
  device const JustFitKvPage *page_table =
      JUSTFIT_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
  constexpr ulong lane_tensor_stride =
      ulong(KVHeads) * JUSTFIT_VERIFY_CHUNK_STRIDE * JustFitKvHeadDimension;
  chunk_keys += batch * lane_tensor_stride;
  chunk_values += batch * lane_tensor_stride;

  uint rows = Rows * KVHeads;
  bool value_tensor = local_group >= rows;
  uint row = value_tensor ? local_group - rows : local_group;
  uint chunk_token = row % Rows;
  uint head = row / Rows;
  justfit_store_kv_row<KVHeads, CacheElement>(
      chunk_keys, chunk_values, page_table, lane_params, maxima, value_tensor,
      head, chunk_token, thread_index, simd_lane, simd_group);
}

#define PAGED_VERIFY_STORE(Name, Heads, CacheElement)                          \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const JustFitKvPage *page_table0 [[buffer(2)]],                    \
      device const JustFitKvPage *page_table1 [[buffer(3)]],                    \
      device const JustFitKvPage *page_table2 [[buffer(4)]],                    \
      device const JustFitKvPage *page_table3 [[buffer(5)]],                    \
      constant JustFitChunkedPrefillParams *params [[buffer(6)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    JUSTFIT_STORE_MAXIMA_##CacheElement                                         \
    justfit_store_verify_phase<Heads, CacheElement>(                            \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2,       \
        page_table3, params, maxima, group, thread_index, simd_lane,           \
        simd_group);                                                           \
  }
PAGED_VERIFY_STORE(verify_attention_q8_store, 4, int8_t)
PAGED_VERIFY_STORE(verify_attention_q8_store_kv2_g8, 2, int8_t)
PAGED_VERIFY_STORE(verify_attention_bf16_store, 4, bfloat)
PAGED_VERIFY_STORE(verify_attention_bf16_store_kv2_g8, 2, bfloat)
#undef PAGED_VERIFY_STORE
