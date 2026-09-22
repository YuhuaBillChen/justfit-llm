#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/tq4_attention_tile.h"
#include "metal/kernels/common/tq4_paging.h"
#include <metal_stdlib>

using namespace metal;

// Qwen3.8-27B target specialization. Queries and outputs retain Splash's
// [lane][KV head][row][GQA head][D] placement. The first kernel rotates the
// queries into the TurboQuant key basis and applies 1/sqrt(D). The attention
// kernel consumes packed Q4 pages without materializing BF16 history, reduces
// its 32 token stripes, and performs the inverse value rotation before the
// result leaves the threadgroup.
constant uint SplashTQ4QueryHeadsPerKVHead = 6;
constant uint SplashTQ4QueryHeads =
    SplashTQ4KVHeads * SplashTQ4QueryHeadsPerKVHead;

inline ulong splash_tq4_query_offset(uint lane, uint kv_head, uint row,
                                     uint gqa_head, uint chunk_stride,
                                     uint dimension) {
  return (((ulong(lane) * SplashTQ4KVHeads + kv_head) * chunk_stride + row) *
              SplashTQ4QueryHeadsPerKVHead +
          gqa_head) *
             SplashTQ4HeadDimension +
         dimension;
}

kernel void verify_attention_tq4_rotate_queries(
    device const bfloat *queries [[buffer(0)]],
    device bfloat *rotated [[buffer(1)]],
    constant SplashTQ4StoreParams *params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
  const uint lane_vectors = SplashTQ4QueryHeads * SPLASH_TARGET_VERIFY_ROWS;
  const uint lane = group / lane_vectors;
  const uint vector = group - lane * lane_vectors;
  const uint query_head = vector / SPLASH_TARGET_VERIFY_ROWS;
  const uint row = vector - query_head * SPLASH_TARGET_VERIFY_ROWS;
  const uint kv_head = query_head / SplashTQ4QueryHeadsPerKVHead;
  const uint gqa_head = query_head % SplashTQ4QueryHeadsPerKVHead;
  constant SplashChunkedPrefillParams &chunk = params[lane].chunk;
  if (row >= chunk.chunk_tokens || dimension >= SplashTQ4HeadDimension)
    return;

  threadgroup float values[SplashTQ4HeadDimension];
  const ulong offset = splash_tq4_query_offset(
      lane, kv_head, row, gqa_head, chunk.chunk_stride, dimension);
  values[dimension] = float(queries[offset]) *
                      splash_tq4_sign(false, dimension);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint half_span = 1; half_span < SplashTQ4HeadDimension;
       half_span <<= 1) {
    if ((dimension & half_span) == 0) {
      const float a = values[dimension];
      const float b = values[dimension + half_span];
      values[dimension] = a + b;
      values[dimension + half_span] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Normalized Hadamard (1/16) and attention scale 1/sqrt(256) (1/16).
  rotated[offset] = bfloat(values[dimension] * (1.0f / 256.0f));
}

kernel void verify_attention_tq4(
    device const bfloat *rotated_queries [[buffer(0)]],
    device const uchar *model_pages [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    device const uint *page_table0 [[buffer(3)]],
    device const uint *page_table1 [[buffer(4)]],
    device const uint *page_table2 [[buffer(5)]],
    device const uint *page_table3 [[buffer(6)]],
    constant SplashTQ4StoreParams *params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint Stripes = 32;
  constexpr uint ElementsPerLane = SplashTQ4HeadDimension / 32;
  const uint query_head = group.x;
  const uint lane = group.y;
  if (query_head >= SplashTQ4QueryHeads)
    return;
  constant SplashTQ4StoreParams &lane_params = params[lane];
  constant SplashChunkedPrefillParams &chunk = lane_params.chunk;
  if (!chunk.chunk_tokens || chunk.chunk_tokens > SPLASH_TARGET_VERIFY_ROWS ||
      chunk.chunk_stride < chunk.chunk_tokens)
    return;
  device const uint *page_table =
      lane == 0 ? page_table0
                : (lane == 1 ? page_table1
                             : (lane == 2 ? page_table2 : page_table3));
  const uint token_count = chunk.committed_tokens + chunk.chunk_tokens;
  const uint logical_pages =
      (token_count + SplashTQ4PageTokens - 1) / SplashTQ4PageTokens;
  const uint kv_head = query_head / SplashTQ4QueryHeadsPerKVHead;
  const uint gqa_head = query_head % SplashTQ4QueryHeadsPerKVHead;

  float query[SPLASH_TARGET_VERIFY_ROWS][ElementsPerLane];
  float accumulator[SPLASH_TARGET_VERIFY_ROWS][ElementsPerLane];
  float maximum[SPLASH_TARGET_VERIFY_ROWS];
  float denominator[SPLASH_TARGET_VERIFY_ROWS];
#pragma unroll
  for (uint row = 0; row < SPLASH_TARGET_VERIFY_ROWS; ++row) {
#pragma unroll
    for (uint i = 0; i < ElementsPerLane; ++i) {
      const uint dimension = simd_lane * ElementsPerLane + i;
      const ulong offset = splash_tq4_query_offset(
          lane, kv_head, row, gqa_head, chunk.chunk_stride, dimension);
      query[row][i] =
          row < chunk.chunk_tokens ? float(rotated_queries[offset]) : 0.0f;
      accumulator[row][i] = 0.0f;
    }
    maximum[row] = -INFINITY;
    denominator[row] = 0.0f;
  }

  for (uint logical_page = 0; logical_page < logical_pages; ++logical_page) {
    const uint physical_page = page_table[logical_page];
    if (physical_page >= chunk.physical_page_count)
      continue;
    device const uchar *layer_page = splash_tq4_layer_page(
        model_pages, physical_page, lane_params.layer_index);
    device const uint *key_indices = splash_tq4_indices(layer_page, false);
    device const uint *value_indices = splash_tq4_indices(layer_page, true);
    device const half *key_norms = splash_tq4_norms(layer_page, false);
    device const half *value_norms = splash_tq4_norms(layer_page, true);
    const uint token_base = logical_page * SplashTQ4PageTokens;
    for (uint page_token = simd_group;
         page_token < SplashTQ4PageTokens && token_base + page_token < token_count;
         page_token += Stripes) {
      const uint token = token_base + page_token;
      const ulong vector_index = splash_tq4_vector_index(kv_head, page_token);
      const float key_norm = float(key_norms[vector_index]);
      const float value_norm = float(value_norms[vector_index]);
      float key[ElementsPerLane];
      float value[ElementsPerLane];
#pragma unroll
      for (uint i = 0; i < ElementsPerLane; ++i) {
        const uint dimension = simd_lane * ElementsPerLane + i;
        key[i] = splash_tq4_value(key_indices, vector_index, dimension) *
                 key_norm;
        value[i] = splash_tq4_value(value_indices, vector_index, dimension) *
                   value_norm;
      }
#pragma unroll
      for (uint row = 0; row < SPLASH_TARGET_VERIFY_ROWS; ++row) {
        if (row >= chunk.chunk_tokens ||
            token > chunk.committed_tokens + row)
          continue;
        float score = 0.0f;
#pragma unroll
        for (uint i = 0; i < ElementsPerLane; ++i)
          score += query[row][i] * key[i];
        score = simd_sum(score);
        const float next_maximum = max(maximum[row], score);
        const float old_factor = fast::exp(maximum[row] - next_maximum);
        const float probability = fast::exp(score - next_maximum);
        maximum[row] = next_maximum;
        denominator[row] = denominator[row] * old_factor + probability;
#pragma unroll
        for (uint i = 0; i < ElementsPerLane; ++i)
          accumulator[row][i] = accumulator[row][i] * old_factor +
                                probability * value[i];
      }
    }
  }

  threadgroup float stripe_max[Stripes];
  threadgroup float stripe_sum[Stripes];
  threadgroup float transpose[32 * 32];
  threadgroup float inverse[SplashTQ4HeadDimension];
#pragma unroll
  for (uint row = 0; row < SPLASH_TARGET_VERIFY_ROWS; ++row) {
    if (simd_lane == 0) {
      stripe_max[simd_group] = maximum[row];
      stripe_sum[simd_group] = denominator[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float local_max = stripe_max[simd_lane];
    const float global_max = simd_max(local_max);
    const float stripe_factor =
        isfinite(global_max) ? fast::exp(local_max - global_max) : 0.0f;
    const float total_sum = simd_sum(stripe_sum[simd_lane] * stripe_factor);
    const float my_factor =
        isfinite(global_max) ? fast::exp(maximum[row] - global_max) : 0.0f;
#pragma unroll
    for (uint i = 0; i < ElementsPerLane; ++i) {
      transpose[simd_lane * 32 + simd_group] =
          accumulator[row][i] * my_factor;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float part = transpose[simd_group * 32 + simd_lane];
      const float reduced = simd_sum(part);
      if (simd_lane == 0)
        inverse[simd_group * ElementsPerLane + i] =
            total_sum > 0.0f ? reduced / total_sum : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The first 256 threads cooperatively apply normalized inverse RHT.
    for (uint half_span = 1; half_span < SplashTQ4HeadDimension;
         half_span <<= 1) {
      if (thread_index < SplashTQ4HeadDimension &&
          (thread_index & half_span) == 0) {
        const float a = inverse[thread_index];
        const float b = inverse[thread_index + half_span];
        inverse[thread_index] = a + b;
        inverse[thread_index + half_span] = a - b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index < SplashTQ4HeadDimension && row < chunk.chunk_tokens) {
      const ulong offset = splash_tq4_query_offset(
          lane, kv_head, row, gqa_head, chunk.chunk_stride, thread_index);
      output[offset] = bfloat(inverse[thread_index] * (1.0f / 16.0f) *
                              splash_tq4_sign(true, thread_index));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void verify_attention_tq4_mpp_split(
    device bfloat *queries [[buffer(0)]],
    device const uchar *model_pages [[buffer(1)]],
    device float *partials [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device const uint *page_table0 [[buffer(4)]],
    device const uint *page_table1 [[buffer(5)]],
    device const uint *page_table2 [[buffer(6)]],
    device const uint *page_table3 [[buffer(7)]],
    constant SplashTQ4VerifyAttentionParams *params [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint Group = SplashTQ4QueryHeadsPerKVHead;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint M = Group * Rows;
  constexpr uint N = SplashTQ4PageTokens;
  alignas(16) threadgroup bfloat page_values[N * SplashTQ4HeadDimension];
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;

  const uint kv_head = group.x;
  const uint split = group.y;
  const uint lane = group.z;
  constant SplashTQ4VerifyAttentionParams &lane_params = params[lane];
  constant SplashQ8VerifyAttentionParams &attention = lane_params.attention;
  if (!splash_q8_verify_attention_contract_valid(attention) ||
      lane_params.layer_index >= SplashTQ4AttentionLayers ||
      lane_params.reserved0 || lane_params.reserved1 || lane_params.reserved2 ||
      kv_head >= SplashTQ4KVHeads || split >= attention.split_count)
    return;
  device const uint *page_table =
      lane == 0 ? page_table0
                : (lane == 1 ? page_table1
                             : (lane == 2 ? page_table2 : page_table3));
  const ulong query_stride =
      ulong(attention.chunk_stride) * Group * SplashTQ4HeadDimension;
  device bfloat *tile_queries =
      queries + (ulong(lane) * SplashTQ4KVHeads + kv_head) * query_stride;
  const ulong slot =
      (ulong(lane) * SplashTQ4KVHeads + kv_head) * attention.slot_splits +
      split;
  splash_tq4_attention_direct_tile<Group, Rows>(
      tile_queries, model_pages, page_table, lane_params.layer_index, kv_head,
      attention.committed_tokens, attention.active_rows,
      attention.split_count, split, partials, statistics, slot, page_values,
      scores, probabilities, row_max, row_sum, previous_scale, &rescale,
      thread_index, simd_group, simd_lane);
}

kernel void verify_attention_tq4_mpp_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashTQ4VerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Group = SplashTQ4QueryHeadsPerKVHead;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  const uint kv_head = group.x;
  const uint fused_row = group.y;
  const uint lane = group.z;
  constant SplashQ8VerifyAttentionParams &attention = params[lane].attention;
  if (!splash_q8_verify_attention_contract_valid(attention) ||
      kv_head >= SplashTQ4KVHeads || fused_row >= Group * Rows ||
      thread_index >= SplashTQ4HeadDimension)
    return;
  const ulong query_stride =
      ulong(attention.chunk_stride) * Group * SplashTQ4HeadDimension;
  splash_q8_attention_reduce_row<Group, Rows>(
      partials, statistics,
      output + (ulong(lane) * SplashTQ4KVHeads + kv_head) * query_stride,
      attention.committed_tokens, attention.active_rows,
      attention.split_count,
      (ulong(lane) * SplashTQ4KVHeads + kv_head) * attention.slot_splits,
      fused_row, thread_index);
}
