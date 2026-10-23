// Modified for JustFit: product identity and integration.
#pragma once

// Where KV pages live, shared by the host and the kernels. A pool's pages sit
// in extents: ordinary shared Metal buffers that all hold the same number of
// pages, allocated when the pool grows; an empty one is released by the next
// reclaim pass between commands. Kernels reach them only through the page
// entries of a request's table, never through a bound buffer.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

// One KV page as kernels address it: the GPU address of the extent that holds
// it, 16 KiB-aligned, with the page's index in the extent in the low bits.
typedef uint64_t JustFitKvPage;
#define JUSTFIT_KV_PAGE_INDEX_BITS 14u
#define JUSTFIT_KV_PAGE_INDEX_MASK ((1u << JUSTFIT_KV_PAGE_INDEX_BITS) - 1u)

// The dimension of every KV head a page holds.
#define JUSTFIT_KV_HEAD_DIMENSION 256u

// The tensors of a layer, in the order they sit in its region of an extent.
// BF16 has no scales: their bytes are zero.
#define JUSTFIT_KV_KEYS 0u
#define JUSTFIT_KV_KEY_SCALES 1u
#define JUSTFIT_KV_VALUES 2u
#define JUSTFIT_KV_VALUE_SCALES 3u

// The bytes one page holds of one tensor of one layer: data_bytes for keys
// and values, scale_bytes for their scales.
inline uint32_t justfit_kv_page_bytes(uint32_t data_bytes, uint32_t scale_bytes,
                                     uint32_t tensor) {
  return tensor % 2 ? scale_bytes : data_bytes;
}

// Where one element of one KV head sits in a page's slab of a tensor, in
// elements, for `token` of the page's JUSTFIT_TARGET_KV_BLOCK_TOKENS: keys
// token-major, values dimension-major, and one scale per (head, token) for
// either of them.
inline uint64_t justfit_kv_key_element(uint32_t head, uint32_t token,
                                      uint32_t dimension) {
  return (uint64_t(head) * JUSTFIT_TARGET_KV_BLOCK_TOKENS + token) *
             JUSTFIT_KV_HEAD_DIMENSION +
         dimension;
}

inline uint64_t justfit_kv_value_element(uint32_t head, uint32_t token,
                                        uint32_t dimension) {
  return (uint64_t(head) * JUSTFIT_KV_HEAD_DIMENSION + dimension) *
             JUSTFIT_TARGET_KV_BLOCK_TOKENS +
         token;
}

inline uint64_t justfit_kv_scale_element(uint32_t head, uint32_t token) {
  return uint64_t(head) * JUSTFIT_TARGET_KV_BLOCK_TOKENS + token;
}

// Where one tensor of one page of one layer sits in its extent, in bytes.
// Every extent of a pool holds extent_pages pages. Per attention layer a
// region holds the keys of every page, then their key scales, values and
// value scales.
inline uint64_t justfit_kv_offset(uint32_t extent_pages, uint32_t data_bytes,
                                 uint32_t scale_bytes, uint32_t layer,
                                 uint32_t tensor, uint32_t index) {
  const uint64_t region =
      uint64_t(layer) * extent_pages * 2 * (uint64_t(data_bytes) + scale_bytes);
  const uint64_t before =
      (tensor + 1) / 2 * uint64_t(data_bytes) + tensor / 2 * uint64_t(scale_bytes);
  return region + extent_pages * before +
         uint64_t(index) * justfit_kv_page_bytes(data_bytes, scale_bytes, tensor);
}

// Where one attention layer sits in every extent of a pool: its region
// begins `offset` bytes into each extent.
struct JustFitKvLayer {
  uint32_t extent_pages;
  uint32_t offset;
};

static_assert(sizeof(JustFitKvLayer) == 8, "KV layer placement is 8 bytes on both sides");

// The placement of `layer` in a pool of extents of extent_pages pages.
inline JustFitKvLayer justfit_kv_layer(uint32_t extent_pages, uint32_t data_bytes,
                                     uint32_t scale_bytes, uint32_t layer) {
  return {extent_pages, uint32_t(justfit_kv_offset(extent_pages, data_bytes, scale_bytes,
                                                  layer, JUSTFIT_KV_KEYS, 0))};
}

// The entry of the page at `index` of the extent whose GPU address is
// extent_address.
inline JustFitKvPage justfit_kv_page_entry(uint64_t extent_address, uint32_t index) {
  return extent_address | index;
}
