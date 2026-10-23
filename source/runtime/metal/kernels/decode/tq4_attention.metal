// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/tq4_attention_tile.h"
kernel void verify_attention_tq4_mpp_split(
    device bfloat *queries [[buffer(0)]],
    device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]],
    device const JustFitKvPage *page_table0 [[buffer(3)]],
    device const JustFitKvPage *page_table1 [[buffer(4)]],
    device const JustFitKvPage *page_table2 [[buffer(5)]],
    device const JustFitKvPage *page_table3 [[buffer(6)]],
    constant JustFitVerifyAttentionParams *params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint Group = 6;
  constexpr uint Rows = 8;
  static_assert(JUSTFIT_TARGET_VERIFY_ROWS == 8 || JUSTFIT_TARGET_VERIFY_ROWS == 16);
  constexpr uint Tiles = JUSTFIT_TARGET_VERIFY_ROWS / Rows;
  constexpr uint M = Group * Rows;
  constexpr uint N = JustFitTQ4PageTokens;
  alignas(16) threadgroup bfloat page_values[N * JustFitTQ4HeadDimension];
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;

  const uint kv_head = group.x / Tiles;
  const uint tile = group.x % Tiles;
  const uint split = group.y;
  const uint lane = group.z;
  constant JustFitVerifyAttentionParams &attention = params[lane];
  if (!justfit_verify_attention_contract_valid(attention) ||
      kv_head >= JustFitTQ4KVHeads || split >= attention.split_count)
    return;
  device const JustFitKvPage *page_table =
      lane == 0 ? page_table0
                : (lane == 1 ? page_table1
                             : (lane == 2 ? page_table2 : page_table3));
  const ulong query_stride =
      ulong(JUSTFIT_VERIFY_CHUNK_STRIDE) * Group * JustFitTQ4HeadDimension;
  device bfloat *tile_queries =
      queries + (ulong(lane) * JustFitTQ4KVHeads + kv_head) * query_stride + tile * Rows * Group * JustFitTQ4HeadDimension;
  const ulong slot =
      ((ulong(lane) * JustFitTQ4KVHeads + kv_head) * Tiles + tile) * attention.slot_splits +
      split;
  justfit_tq4_attention_direct_tile<Group, Rows>(
      tile_queries, page_table, attention.kv, kv_head,
      attention.committed_tokens + tile * Rows, min(Rows, JUSTFIT_TARGET_VERIFY_ROWS - tile * Rows),
      attention.split_count, split, partials, statistics, slot, page_values,
      scores, probabilities, row_max, row_sum, previous_scale, &rescale,
      thread_index, simd_group, simd_lane);
}


// Both widths retain eight-row attention tiles and their original split order.
kernel void verify_attention_tq4_reduce_tiled(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant JustFitVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Rows=8, Group=6, Tiles=JUSTFIT_TARGET_VERIFY_ROWS/Rows;
  const uint head=group.x/Tiles,tile=group.x%Tiles,lane=group.z;
  constant JustFitVerifyAttentionParams &p=params[lane];
  if(!justfit_verify_attention_contract_valid(p) || head>=4 || group.y>=Rows*Group || thread_index>=256) return;
  threadgroup float weights[JustFitVerifyMaximumSplits];
  threadgroup float group_values[8];
  justfit_attention_reduce_row<Group,Rows>(partials,statistics,
    output+((ulong(lane)*4+head)*JUSTFIT_VERIFY_CHUNK_STRIDE+tile*Rows)*Group*256,
    p.committed_tokens+tile*Rows,Rows,p.split_count,
    ((ulong(lane)*4+head)*Tiles+tile)*p.slot_splits,group.y,thread_index,weights,group_values);
}
