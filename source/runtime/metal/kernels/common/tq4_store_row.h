// Modified for JustFit: product identity and integration.
#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/paged_store_row.h"
#include "metal/kernels/common/tq4_paging.h"
#include <metal_stdlib>

using namespace metal;

template <uint KVHeads>
inline void justfit_tq4_store_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const JustFitKvPage *page_table,
    constant JustFitChunkedPrefillParams &chunk, threadgroup float *vector,
    threadgroup float *partial_sums, threadgroup float &stored_norm,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  const uint logical_token = chunk.committed_tokens + chunk_token;
  const JustFitKvPage entry = page_table[logical_token / JustFitTQ4PageTokens];
  const uint page_token = logical_token % JustFitTQ4PageTokens;
  const ulong source_index =
      value_tensor
          ? justfit_current_value_index(chunk.chunk_stride, head, chunk_token,
                                       dimension)
          : justfit_current_key_index(chunk.chunk_stride, head, chunk_token,
                                     dimension);
  const float input =
      float(value_tensor ? chunk_values[source_index] : chunk_keys[source_index]);

  const float local_sum = simd_sum(input * input);
  if (simd_lane == 0)
    partial_sums[simd_group] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float group_sum = simd_lane < 8 ? partial_sums[simd_lane] : 0.0f;
    const float total_sum = simd_sum(group_sum);
    if (simd_lane == 0)
      stored_norm = sqrt(total_sum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float norm = stored_norm;
  vector[dimension] =
      (norm > 0.0f ? input / norm : 0.0f) *
      justfit_tq4_sign(value_tensor, dimension);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint half_span = 1; half_span < JustFitTQ4HeadDimension;
       half_span <<= 1) {
    if ((dimension & half_span) == 0) {
      const float a = vector[dimension];
      const float b = vector[dimension + half_span];
      vector[dimension] = a + b;
      vector[dimension + half_span] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const auto layer_page = justfit_tq4_page(entry, chunk.kv);
  const ulong vector_index = justfit_tq4_vector_index(head, page_token);
  if (dimension < JustFitTQ4PackedWords) {
    uint word = 0;
    for (uint nibble = 0; nibble < 8; ++nibble) {
      const uint d = dimension * 8 + nibble;
      const float rotated = vector[d] * (1.0f / 16.0f);
      uint code = 0;
      for (uint threshold = 0; threshold < 15; ++threshold)
        code += rotated > JustFitTQ4Midpoints[threshold];
      word |= code << (nibble * 4);
    }
    justfit_tq4_indices(layer_page, value_tensor)
        [vector_index * JustFitTQ4PackedWords + dimension] = word;
  }
  if (dimension == 0)
    justfit_tq4_norms(layer_page, value_tensor)[vector_index] = half(norm);
}
