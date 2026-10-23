// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/tq4_attention_tile.h"
// Experimental prefill-only staging path. One KV head is materialized at a
// time so the persistent cache remains TQ4 and staging stays bounded to two
// BF16 [tokens, D] planes. The host orders stage -> attention before reusing
// the planes for the next head.
kernel void prefill_attention_tq4_stage_head(
    device bfloat *staged_keys [[buffer(0)]],
    device bfloat *staged_values [[buffer(1)]],
    device const JustFitKvPage *page_table [[buffer(2)]],
    constant JustFitTQ4StagedPrefillParams &params [[buffer(3)]],
    uint page [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint N = JustFitTQ4PageTokens;
  constexpr uint D = JustFitTQ4HeadDimension;
  alignas(16) threadgroup bfloat decoded[N * D];
  const uint pages = justfit_attention_pages(params.visible_tokens);
  if (page >= pages || params.kv_head >= JustFitTQ4KVHeads)
    return;
  const JustFitTQ4PageView layer_page = justfit_tq4_page(page_table[page], params.kv);
  justfit_tq4_inverse_page<false>(layer_page, params.kv_head, decoded,
                                 simd_group, simd_lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = page * N;
  for (uint index = thread_index; index < N * D; index += 256) {
    const uint token = index / D;
    if (start + token < params.visible_tokens)
      staged_keys[ulong(start + token) * D + index % D] = decoded[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  justfit_tq4_inverse_page<true>(layer_page, params.kv_head, decoded,
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
    constant JustFitTQ4StagedPrefillParams &params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr ushort Rows = JUSTFIT_PREFILL_ATTENTION_TILE_ROWS;
  constexpr ushort Group = 6;
  constexpr ushort M = Rows * Group;
  constexpr ushort N = 64;
  constexpr ushort D = JustFitTQ4HeadDimension;
  constexpr ushort RowsPerSimdgroup = M / 8;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float scales[M];
  threadgroup float denominators[M];
  const uint tile_start = tile * Rows;
  if (tile_start >= params.rows || params.kv_head >= JustFitTQ4KVHeads ||
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
  // Keys past the tile's last causal position are fully masked: they leave the
  // running max, sum and output unchanged, so the loop stops at that edge.
  const uint causal_last = min(params.visible_tokens,
                               params.committed_tokens + tile_start + active_rows);
  const uint key_tiles = (causal_last + N - 1) / N;
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
      // Absolute chunk position: the tile's first row plus the row in it.
      const uint causal_end = params.committed_tokens + tile_start +
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


// Experimental upstream-inspired compact softmax; runtime opt-in.
kernel void prefill_attention_tq4_staged_compact(
    device bfloat *queries [[buffer(0)]],
    device bfloat *staged_keys [[buffer(1)]],
    device bfloat *staged_values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant JustFitTQ4StagedPrefillParams &params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ushort Rows = JUSTFIT_PREFILL_ATTENTION_TILE_ROWS;
  constexpr ushort Group = 6;
  constexpr ushort M = Rows * Group;
  constexpr ushort N = 64;
  constexpr ushort D = JustFitTQ4HeadDimension;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float scales[M];
  threadgroup float denominators[M];
  threadgroup float row_max[M];
  threadgroup atomic_uint rescale;
  const uint tile_start = tile * Rows;
  if (tile_start >= params.rows || params.kv_head >= JustFitTQ4KVHeads ||
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
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    denominators[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Keys past the tile's last causal position are fully masked: they leave the
  // running max, sum and output unchanged, so the loop stops at that edge.
  const uint causal_last = min(params.visible_tokens,
                               params.committed_tokens + tile_start + active_rows);
  const uint key_tiles = (causal_last + N - 1) / N;
  for (uint key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const uint key_start = key_tile * N;
    auto keys = tensor(staged_keys + ulong(key_start) * D,
                       dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto key_slice = keys.slice<D, N>(0, 0);
    auto tile_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(key_slice), float>();
    qk.run(q0, key_slice, tile_scores);
    tile_scores.store(score_tensor.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Four neighboring lanes own sixteen consecutive keys each. As in the
    // upstream Page32 softmax, two xor shuffles reduce each row. N64 retains
    // the staged kernel's existing matrix tile and memory footprint.
    if (thread_index < uint(M) * 4) {
      const uint row = thread_index / 4;
      const uint column = (thread_index % 4) * 16;
      const uint limit = min(params.visible_tokens, params.committed_tokens +
          tile_start + min(row / Group, active_rows - 1) + 1);
      threadgroup const float4 *src = reinterpret_cast<threadgroup const float4 *>(
          scores + row * N + column);
      float score[16];
      float maximum = -INFINITY;
#pragma unroll
      for (uint v = 0; v < 4; ++v) {
        const float4 packed = src[v] * 0.0625f;
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
          const uint i = v * 4 + j;
          score[i] = key_start + column + i < limit ? packed[j] : -INFINITY;
          maximum = max(maximum, score[i]);
        }
      }
      maximum = max(maximum, simd_shuffle_xor(maximum, 1));
      maximum = max(maximum, simd_shuffle_xor(maximum, 2));
      const float previous = row_max[row];
      const float next = max(previous, maximum);
      float probability[16];
      float sum = 0.0f;
#pragma unroll
      for (uint i = 0; i < 16; ++i) {
        probability[i] = key_start + column + i < limit
            ? fast::exp(score[i] - next) : 0.0f;
        sum += probability[i];
      }
      sum += simd_shuffle_xor(sum, 1);
      sum += simd_shuffle_xor(sum, 2);
      if (column == 0) {
        const float scale = next == -INFINITY || next == previous
            ? 1.0f : fast::exp(previous - next);
        scales[row] = scale;
        denominators[row] = denominators[row] * scale + sum;
        row_max[row] = next;
        if (scale != 1.0f)
          atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
      }
      threadgroup bfloat4 *dst = reinterpret_cast<threadgroup bfloat4 *>(
          probabilities + row * N + column);
#pragma unroll
      for (uint v = 0; v < 4; ++v)
        dst[v] = bfloat4(float4(probability[v*4], probability[v*4+1],
                                probability[v*4+2], probability[v*4+3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index)) continue;
        const auto coordinates = running.get_multidimensional_index(index);
        running[index] *= scales[coordinates[1]];
      }
    }
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

