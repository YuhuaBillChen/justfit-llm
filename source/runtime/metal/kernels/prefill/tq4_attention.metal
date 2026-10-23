// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/tq4_attention_tile.h"

kernel void prefill_attention_tq4_mpp_split(
    device bfloat *queries [[buffer(0)]],
    device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]],
    device const JustFitKvPage *page_table [[buffer(3)]],
    constant JustFitPrefillAttentionParams &attention [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint Group = 6;
  constexpr uint Rows = JUSTFIT_PREFILL_ATTENTION_TILE_ROWS;
  constexpr uint M = Group * Rows;
  constexpr uint N = JustFitTQ4PageTokens;
  alignas(16) threadgroup bfloat page_values[N * JustFitTQ4HeadDimension];
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;

  const uint kv_head = group.x;
  const uint tile = group.y;
  const uint split = group.z;
  const uint tile_start = tile * Rows;
  if (!justfit_prefill_attention_contract_valid(attention) ||
      kv_head >= JustFitTQ4KVHeads ||
      split >= attention.split_count || tile_start >= attention.rows)
    return;
  const uint active_rows = min(Rows, attention.rows - tile_start);
  const ulong tile_offset =
      (ulong(kv_head) * attention.chunk_stride + tile_start) * Group *
      JustFitTQ4HeadDimension;
  const ulong slot =
      (ulong(tile) * JustFitTQ4KVHeads + kv_head) * attention.split_count +
      split;
  justfit_tq4_attention_direct_tile<Group, Rows>(
      queries + tile_offset, page_table, attention.kv,
      kv_head, attention.committed_tokens + tile_start, active_rows,
      attention.split_count, split, partials, statistics, slot, page_values,
      scores, probabilities, row_max, row_sum, previous_scale, &rescale,
      thread_index, simd_group, simd_lane);
}

