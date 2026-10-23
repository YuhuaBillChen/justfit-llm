// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (justfit_paged_attention_tile) over eight query rows of one KV head.

// Prefill split: KV heads vary first, then query tiles, then history splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement>
inline void justfit_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const JustFitKvPage *page_table,
    constant JustFitPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index) {
  constexpr uint D = JustFitKvHeadDimension;
  uint kv_head = group.x;
  uint split = group.z;
  uint tile = group.y;
  uint tile_start = tile * JustFitPrefillTileRows;
  if (!justfit_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || split >= params.split_count ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(JustFitPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  ulong slot = (ulong(tile) * KVHeads + kv_head) * params.split_count + split;
  justfit_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              JUSTFIT_PREFILL_ATTENTION_TILE_ROWS, CacheElement>(
      queries + tile_offset, page_table, params.kv, kv_head,
      params.committed_tokens + tile_start, active_rows, params.split_count, split,
      partials, statistics, slot, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index);
}

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void justfit_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant JustFitPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values) {
  constexpr ushort M = JUSTFIT_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = JustFitKvHeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * JustFitPrefillTileRows;
  if (!justfit_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(JustFitPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  justfit_attention_reduce_row<QueryHeadsPerKVHead,
                              JUSTFIT_PREFILL_ATTENTION_TILE_ROWS>(
      partials, statistics, output + tile_offset,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index, weights, group_values);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const JustFitKvPage *page_table [[buffer(3)]],                     \
      constant JustFitPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * JUSTFIT_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = JustFitKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    justfit_prefill_attention_split_phase<Heads, Group, CacheElement>(          \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale, group,      \
        thread_index);                                                         \
  }

PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, int8_t)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, bfloat)
#undef PAGED_PREFILL_SPLIT

#define PAGED_PREFILL_REDUCE(Name, Heads, Group)                               \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant JustFitPrefillAttentionParams &params [[buffer(3)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[JustFitPrefillMaximumSplits];                     \
    threadgroup float group_values[8];                                         \
    justfit_prefill_attention_reduce_phase<Heads, Group>(                       \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values);                                                         \
  }
PAGED_PREFILL_REDUCE(prefill_attention_reduce, 4, 6)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g8, 2, 8)
#undef PAGED_PREFILL_REDUCE
