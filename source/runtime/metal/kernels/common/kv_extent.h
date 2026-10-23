// Modified for JustFit: product identity and integration.
#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// Kernels reach a KV page through its entry (abi/KvExtent.h): the GPU address
// of the extent that holds it, with the page's index in the low bits. Every
// kernel that touches KV pages forms the pointers to one KV head's slab of
// each of a page's tensors in one layer the same way, with
// JustFitKvAddressing: justfit_kv_offset places the layer's regions in the
// extent, whose 64-bit offsets it adds to the extent's address, and the page
// index functions of kv_paging.h place the page and the head in each region.

inline uint justfit_kv_page_index(JustFitKvPage page) {
  return uint(page) & JUSTFIT_KV_PAGE_INDEX_MASK;
}

inline device uchar *justfit_kv_extent(JustFitKvPage page, uint index) {
  return reinterpret_cast<device uchar *>(page - index);
}

// One page's bytes of keys (or values) and of their scales in one layer.
template <uint KVHeads, typename CacheElement> struct JustFitKvPageBytes {
  static constexpr constant bool Quantized = is_same<CacheElement, int8_t>::value;
  static constexpr constant uint Data =
      KVHeads * JustFitKvPageTokens * JustFitKvHeadDimension * sizeof(CacheElement);
  static constexpr constant uint Scale =
      Quantized ? KVHeads * JustFitKvPageTokens * sizeof(float) : 0;
};

// One KV head's slab of each tensor of a page; BF16 has no scales.
template <typename CacheElement> struct JustFitKvPageTensors {
  device CacheElement *keys;
  device CacheElement *values;
  device float *key_scales;
  device float *value_scales;
};

template <uint KVHeads, typename CacheElement> struct JustFitKvAddressing {
  using Bytes = JustFitKvPageBytes<KVHeads, CacheElement>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  JustFitKvAddressing(JustFitKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(justfit_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                           0, JUSTFIT_KV_KEY_SCALES, 0)),
        values_offset(justfit_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                       0, JUSTFIT_KV_VALUES, 0)),
        value_scales_offset(justfit_kv_offset(kv.extent_pages, Bytes::Data,
                                             Bytes::Scale, 0,
                                             JUSTFIT_KV_VALUE_SCALES, 0)) {}

  JustFitKvPageTensors<CacheElement> page(JustFitKvPage entry) const {
    const uint index = justfit_kv_page_index(entry);
    device uchar *region = justfit_kv_extent(entry, index) + layer_offset;
    JustFitKvPageTensors<CacheElement> tensors{};
    tensors.keys = reinterpret_cast<device CacheElement *>(region) +
                   justfit_kv_key_index<KVHeads>(index, head, 0, 0);
    tensors.values = reinterpret_cast<device CacheElement *>(region + values_offset) +
                     justfit_kv_value_index<KVHeads>(index, head, 0, 0);
    if constexpr (Bytes::Quantized) {
      const ulong scale_index = justfit_q8_scale_index<KVHeads>(index, head, 0);
      tensors.key_scales =
          reinterpret_cast<device float *>(region + key_scales_offset) + scale_index;
      tensors.value_scales =
          reinterpret_cast<device float *>(region + value_scales_offset) + scale_index;
    }
    return tensors;
  }
};
