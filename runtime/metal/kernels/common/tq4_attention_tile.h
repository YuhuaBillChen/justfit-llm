#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/q8_attention_tile.h"
#include "metal/kernels/common/tq4_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Inverse one packed Page32 tensor into a 16-KiB threadgroup tile. Eight
// SIMD-groups own four token vectors each. The 256-point RHT is H32 across
// lanes times H8 within each lane, so no per-stage threadgroup barrier is
// required. Keys leave token-major [N,D]; values leave dimension-major [D,N]
// for the two MPP operands.
template <bool ValueTensor>
inline void splash_tq4_inverse_page(device const uchar *layer_page,
                                    uint kv_head,
                                    threadgroup bfloat *page_values,
                                    uint simd_group, uint simd_lane) {
  constexpr uint D = SplashTQ4HeadDimension;
  constexpr uint N = SplashTQ4PageTokens;
  constexpr uint Local = D / 32;
  device const uint *indices =
      splash_tq4_indices(layer_page, ValueTensor);
  device const half *norms = splash_tq4_norms(layer_page, ValueTensor);
  for (uint token = simd_group; token < N; token += 8) {
    const ulong vector_index = splash_tq4_vector_index(kv_head, token);
    float value[Local];
#pragma unroll
    for (uint i = 0; i < Local; ++i) {
      const uint dimension = simd_lane * Local + i;
      value[i] = splash_tq4_value(indices, vector_index, dimension);
    }
    // H8 over the lane-local contiguous dimensions.
#pragma unroll
    for (uint half_span = 1; half_span < Local; half_span <<= 1) {
#pragma unroll
      for (uint base = 0; base < Local; base += 2 * half_span) {
#pragma unroll
        for (uint offset = 0; offset < half_span; ++offset) {
          const float a = value[base + offset];
          const float b = value[base + offset + half_span];
          value[base + offset] = a + b;
          value[base + offset + half_span] = a - b;
        }
      }
    }
    // H32 across lanes for each local dimension.
#pragma unroll
    for (uint mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
      for (uint i = 0; i < Local; ++i) {
        const float partner = simd_shuffle_xor(value[i], mask);
        value[i] = (simd_lane & mask) ? partner - value[i]
                                      : value[i] + partner;
      }
    }
    const float norm = float(norms[vector_index]) * (1.0f / 16.0f);
#pragma unroll
    for (uint i = 0; i < Local; ++i) {
      const uint dimension = simd_lane * Local + i;
      const bfloat decoded =
          bfloat(value[i] * norm * splash_tq4_sign(ValueTensor, dimension));
      if constexpr (ValueTensor)
        page_values[dimension * N + token] = decoded;
      else
        page_values[token * D + dimension] = decoded;
    }
  }
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_tq4_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint token_start, uint visible_tokens, uint committed_tokens,
    uint active_rows, uint thread_index) {
  constexpr uint N = SplashTQ4PageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / QueryHeadsPerKVHead;
  const uint causal_end =
      committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint limit = min(visible_tokens, causal_end);
  const uint token = token_start + column;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                   column);
  const float4 low_scores = scores4[0] * 0.0625f;
  const float4 high_scores = scores4[1] * 0.0625f;
  float score[TokensPerLane]{low_scores.x, low_scores.y, low_scores.z,
                             low_scores.w, high_scores.x, high_scores.y,
                             high_scores.z, high_scores.w};
  float local_max = -INFINITY;
#pragma unroll
  for (uint i = 0; i < TokensPerLane; ++i) {
    score[i] = token + i < limit ? score[i] : -INFINITY;
    local_max = max(local_max, score[i]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float old_max = row_max[fused_row];
  const float next_max = max(old_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint i = 0; i < TokensPerLane; ++i) {
    probability[i] =
        token + i < limit ? fast::exp(score[i] - next_max) : 0.0f;
    local_sum += probability[i];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == old_max
                            ? 1.0f
                            : fast::exp(old_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  threadgroup bfloat4 *target =
      reinterpret_cast<threadgroup bfloat4 *>(probabilities + fused_row * N +
                                              column);
  target[0] = bfloat4(float4(probability[0], probability[1], probability[2],
                            probability[3]));
  target[1] = bfloat4(float4(probability[4], probability[5], probability[6],
                            probability[7]));
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_tq4_attention_direct_tile(
    device bfloat *tile_queries, device const uchar *model_pages,
    device const uint *page_table, uint layer_index, uint kv_head,
    uint committed_tokens, uint active_rows, uint splits, uint split,
    device float *partials, device float *statistics, ulong slot,
    threadgroup bfloat *page_values, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint thread_index, uint simd_group,
    uint simd_lane) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashTQ4PageTokens;
  constexpr ushort D = SplashTQ4HeadDimension;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint pages = splash_attention_pages(visible_tokens);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  const uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto queries =
      tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto score_tensor =
      tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto probability_tensor = tensor(probabilities, dextents<int, 2>{N, M},
                                   array<int, 2>{1, N});
  auto probability_slice = probability_tensor.slice<N, M>(0, 0);
  auto query_slice = queries.slice<D, M>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto dummy_values = tensor(page_values, dextents<int, 2>{N, D},
                             array<int, 2>{1, N});
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(probability_slice), decltype(dummy_values), float>();
  const bool running_full =
      uint(running.get_capacity()) * 256u == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;

  for (uint page = page_begin; page < page_end; ++page) {
    const uint physical = page_table[page];
    device const uchar *layer_page =
        splash_tq4_layer_page(model_pages, physical, layer_index);
    splash_tq4_inverse_page<false>(layer_page, kv_head, page_values,
                                   simd_group, simd_lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto keys = tensor(page_values, dextents<int, 2>{D, N},
                       array<int, 2>{1, D});
    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(query_slice), decltype(keys), float>();
    auto key_slice = keys.slice<D, N>(0, 0);
    qk.run(query_slice, key_slice, page_scores);
    page_scores.store(score_tensor.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_tq4_page_softmax<QueryHeadsPerKVHead, RowsPerTile>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        page * N, visible_tokens, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        const auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    splash_tq4_inverse_page<true>(layer_page, kv_head, page_values,
                                  simd_group, simd_lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto values = tensor(page_values, dextents<int, 2>{N, D},
                         array<int, 2>{1, N});
    auto value_slice = values.slice<N, D>(0, 0);
    pv.run(probability_slice, value_slice, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D,
                       dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}
