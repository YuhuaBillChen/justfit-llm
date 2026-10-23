// Modified for JustFit: product identity and integration.
#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

constant uint JustFitChunkMaximumRows = JUSTFIT_PREFILL_TOKEN_BUDGET;
constant uint JustFitChunkMaximumPhysicalTokens =
    JUSTFIT_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
justfit_chunk_contract_valid(constant JustFitChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         JustFitKvPageTokens - 1) /
                        JustFitKvPageTokens;
  return params.committed_tokens <= JustFitChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= JustFitChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             JustFitChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= JustFitChunkMaximumRows &&
         params.chunk_stride % JUSTFIT_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.kv.extent_pages > 0;
}

inline ulong justfit_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * JustFitKvHeadDimension + dimension;
}

inline ulong justfit_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * JustFitKvHeadDimension + dimension) * stride + token;
}

// The scratch of a store entry's scale reduction, by CacheElement: eight
// simdgroup maxima for INT8; BF16 copies the source bits and has none.
#define JUSTFIT_STORE_MAXIMA_int8_t threadgroup float maxima[8];
#define JUSTFIT_STORE_MAXIMA_bfloat threadgroup float *maxima = nullptr;

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits. Slots are
// addressed inside the head's slab of the page: the page index functions at
// page zero and head zero.
template <uint KVHeads, typename CacheElement>
__attribute__((always_inline)) inline void justfit_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const JustFitKvPage *page_table,
    constant JustFitChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const JustFitKvPage page = page_table[logical_token / JustFitKvPageTokens];
  const JustFitKvPageTensors<CacheElement> slab =
      JustFitKvAddressing<KVHeads, CacheElement>(params.kv, head).page(page);
  uint page_token = logical_token % JustFitKvPageTokens;
  ulong source_index =
      value_tensor ? justfit_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : justfit_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      slab.values[justfit_kv_value_index<KVHeads>(0, 0, page_token, dimension)] = source;
    else
      slab.keys[justfit_kv_key_index<KVHeads>(0, 0, page_token, dimension)] = source;
    return;
  } else {

    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = 4; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 127.0f / maximum)), -127, 127);

    if (value_tensor) {
      slab.values[justfit_kv_value_index<KVHeads>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.value_scales[justfit_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      slab.keys[justfit_kv_key_index<KVHeads>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[justfit_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
}
