#include "metal/kernels/common/tq4_attention_tile.h"

kernel void prefill_attention_tq4_mpp_split(
    device bfloat *queries [[buffer(0)]],
    device const uchar *model_pages [[buffer(1)]],
    device float *partials [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device const uint *page_table [[buffer(4)]],
    constant SplashTQ4PrefillAttentionParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint Group = 6;
  constexpr uint Rows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  constexpr uint M = Group * Rows;
  constexpr uint N = SplashTQ4PageTokens;
  alignas(16) threadgroup bfloat page_values[N * SplashTQ4HeadDimension];
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;

  constant SplashQ8PrefillAttentionParams &attention = params.attention;
  const uint kv_head = group.x;
  const uint tile = group.y;
  const uint split = group.z;
  const uint tile_start = tile * Rows;
  if (!splash_q8_prefill_attention_contract_valid(attention) ||
      params.layer_index >= SplashTQ4AttentionLayers || params.reserved0 ||
      params.reserved1 || params.reserved2 || kv_head >= SplashTQ4KVHeads ||
      split >= attention.split_count || tile_start >= attention.rows)
    return;
  const uint active_rows = min(Rows, attention.rows - tile_start);
  const ulong tile_offset =
      (ulong(kv_head) * attention.chunk_stride + tile_start) * Group *
      SplashTQ4HeadDimension;
  const ulong slot =
      (ulong(tile) * SplashTQ4KVHeads + kv_head) * attention.split_count +
      split;
  splash_tq4_attention_direct_tile<Group, Rows>(
      queries + tile_offset, model_pages, page_table, params.layer_index,
      kv_head, attention.committed_tokens + tile_start, active_rows,
      attention.split_count, split, partials, statistics, slot, page_values,
      scores, probabilities, row_max, row_sum, previous_scale, &rescale,
      thread_index, simd_group, simd_lane);
}

kernel void prefill_attention_tq4_mpp_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashTQ4PrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Group = 6;
  constexpr uint Rows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  const uint kv_head = group.x;
  const uint fused_row = group.y;
  const uint tile = group.z;
  const uint tile_start = tile * Rows;
  constant SplashQ8PrefillAttentionParams &attention = params.attention;
  if (!splash_q8_prefill_attention_contract_valid(attention) ||
      kv_head >= SplashTQ4KVHeads || fused_row >= Rows * Group ||
      thread_index >= SplashTQ4HeadDimension || tile_start >= attention.rows)
    return;
  const uint active_rows = min(Rows, attention.rows - tile_start);
  if (fused_row / Group >= active_rows)
    return;
  const ulong tile_offset =
      (ulong(kv_head) * attention.chunk_stride + tile_start) * Group *
      SplashTQ4HeadDimension;
  splash_q8_attention_reduce_row<Group, Rows>(
      partials, statistics, output + tile_offset,
      attention.committed_tokens + tile_start, active_rows,
      attention.split_count,
      (ulong(tile) * SplashTQ4KVHeads + kv_head) * attention.split_count,
      fused_row, thread_index);
}

// Experimental prefill-only staging path. One KV head is materialized at a
// time so the persistent cache remains TQ4 and staging stays bounded to two
// BF16 [tokens, D] planes. The host orders stage -> attention before reusing
// the planes for the next head.
kernel void prefill_attention_tq4_stage_head(
    device const uchar *model_pages [[buffer(0)]],
    device bfloat *staged_keys [[buffer(1)]],
    device bfloat *staged_values [[buffer(2)]],
    device const uint *page_table [[buffer(3)]],
    constant SplashTQ4StagedPrefillParams &params [[buffer(4)]],
    uint page [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint N = SplashTQ4PageTokens;
  constexpr uint D = SplashTQ4HeadDimension;
  alignas(16) threadgroup bfloat decoded[N * D];
  const uint pages = splash_attention_pages(params.visible_tokens);
  if (page >= pages || params.layer_index >= SplashTQ4AttentionLayers ||
      params.kv_head >= SplashTQ4KVHeads || params.reserved0 ||
      params.reserved1)
    return;
  const uint physical = page_table[page];
  device const uchar *layer_page =
      splash_tq4_layer_page(model_pages, physical, params.layer_index);
  splash_tq4_inverse_page<false>(layer_page, params.kv_head, decoded,
                                 simd_group, simd_lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = page * N;
  for (uint index = thread_index; index < N * D; index += 256) {
    const uint token = index / D;
    if (start + token < params.visible_tokens)
      staged_keys[ulong(start + token) * D + index % D] = decoded[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  splash_tq4_inverse_page<true>(layer_page, params.kv_head, decoded,
                                simd_group, simd_lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_index; index < N * D; index += 256) {
    const uint token = index / D;
    const uint dimension = index % D;
    if (start + token < params.visible_tokens)
      staged_values[ulong(start + token) * D + dimension] =
          decoded[dimension * N + token];
  }
}

kernel void prefill_attention_tq4_staged_head(
    device bfloat *queries [[buffer(0)]],
    device bfloat *staged_keys [[buffer(1)]],
    device bfloat *staged_values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant SplashTQ4StagedPrefillParams &params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr ushort Rows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  constexpr ushort Group = 6;
  constexpr ushort M = Rows * Group;
  constexpr ushort N = 64;
  constexpr ushort D = SplashTQ4HeadDimension;
  constexpr ushort RowsPerSimdgroup = M / 8;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float scales[M];
  threadgroup float denominators[M];
  const uint tile_start = tile * Rows;
  if (tile_start >= params.rows || params.kv_head >= SplashTQ4KVHeads ||
      params.visible_tokens != params.committed_tokens + params.rows)
    return;
  const uint active_rows = min(uint(Rows), params.rows - tile_start);
  const ulong query_offset =
      (ulong(params.kv_head) * params.chunk_stride + tile_start) * Group * D;
  device bfloat *q = queries + query_offset;
  device bfloat *target = output + query_offset;
  auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto q0 = qt.slice<D, M>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto probability_tensor = tensor(probabilities, dextents<int, 2>{N, M},
                                   array<int, 2>{1, N});
  auto probability_slice = probability_tensor.slice<N, M>(0, 0);
  auto score_tensor = tensor(scores, dextents<int, 2>{N, M},
                             array<int, 2>{1, N});
  auto first_values = tensor(staged_values, dextents<int, 2>{D, N},
                             array<int, 2>{1, D});
  auto first_value_slice = first_values.slice<D, N>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(probability_slice), decltype(first_value_slice), float>();
  const bool running_full =
      uint(running.get_capacity()) * 256u == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  float running_max[RowsPerSimdgroup];
  float running_sum[RowsPerSimdgroup];
#pragma unroll
  for (ushort row = 0; row < RowsPerSimdgroup; ++row) {
    running_max[row] = -INFINITY;
    running_sum[row] = 0.0f;
  }
  const uint key_tiles = (params.visible_tokens + N - 1) / N;
  for (uint key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const uint key_start = key_tile * N;
    auto keys = tensor(staged_keys + ulong(key_start) * D,
                       dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto key_slice = keys.slice<D, N>(0, 0);
    auto tile_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(key_slice), float>();
    qk.run(q0, key_slice, tile_scores);
    tile_scores.store(score_tensor.slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float exponent[RowsPerSimdgroup][2];
    float local_scale[RowsPerSimdgroup];
#pragma unroll
    for (ushort local_row = 0; local_row < RowsPerSimdgroup; ++local_row) {
      const uint fused_row = simd_group * RowsPerSimdgroup + local_row;
      const uint query_row = fused_row / Group;
      const uint causal_end = params.committed_tokens +
                              min(query_row, active_rows - 1) + 1;
      float value0 = key_start + simd_lane < min(params.visible_tokens, causal_end)
                         ? scores[fused_row * N + simd_lane] * 0.0625f
                         : -INFINITY;
      float value1 = key_start + simd_lane + 32 <
                             min(params.visible_tokens, causal_end)
                         ? scores[fused_row * N + simd_lane + 32] * 0.0625f
                         : -INFINITY;
      const float tile_max = simd_max(max(value0, value1));
      const float next_max = max(running_max[local_row], tile_max);
      local_scale[local_row] = running_max[local_row] == -INFINITY
                                   ? 0.0f
                                   : fast::exp(running_max[local_row] - next_max);
      exponent[local_row][0] = value0 == -INFINITY ? 0.0f
                                                    : fast::exp(value0 - next_max);
      exponent[local_row][1] = value1 == -INFINITY ? 0.0f
                                                    : fast::exp(value1 - next_max);
      const float tile_sum =
          simd_sum(exponent[local_row][0] + exponent[local_row][1]);
      running_sum[local_row] =
          running_sum[local_row] * local_scale[local_row] + tile_sum;
      running_max[local_row] = next_max;
      if (simd_lane == 0) {
        scales[fused_row] = local_scale[local_row];
        denominators[fused_row] = running_sum[local_row];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort index = 0; index < running.get_capacity(); ++index) {
      if (!running_full && !running.is_valid_element(index))
        continue;
      const auto coordinates = running.get_multidimensional_index(index);
      running[index] *= scales[coordinates[1]];
    }
#pragma unroll
    for (ushort local_row = 0; local_row < RowsPerSimdgroup; ++local_row) {
      const uint fused_row = simd_group * RowsPerSimdgroup + local_row;
      probabilities[fused_row * N + simd_lane] =
          bfloat(exponent[local_row][0]);
      probabilities[fused_row * N + simd_lane + 32] =
          bfloat(exponent[local_row][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto values = tensor(staged_values + ulong(key_start) * D,
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto value_slice = values.slice<D, N>(0, 0);
    pv.run(probability_slice, value_slice, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Normalize by the per-row online-softmax denominator before the final
  // distributed store.
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (!running_full && !running.is_valid_element(index))
      continue;
    const auto coordinates = running.get_multidimensional_index(index);
    const uint row = coordinates[1];
    const uint dimension = coordinates[0];
    target[ulong(row) * D + dimension] =
        bfloat(running[index] / denominators[row]);
  }
}
