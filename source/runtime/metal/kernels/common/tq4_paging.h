// Modified for JustFit: product identity and integration.
#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/kernels/common/kv_extent.h"
#include <metal_stdlib>

using namespace metal;

// JustFit TurboQuant-Q4 ABI.  Each D=256 vector stores 32 uint32 words of
// packed nibbles and one fp16 L2 norm.  Both K and V are token-major inside a
// Page32 block.  The codebook and randomized-Hadamard signs exactly match the
// MLX JustFit codec (D=256, key seed 0, value seed 1).
constant uint JustFitTQ4PageTokens = JUSTFIT_TARGET_KV_BLOCK_TOKENS;
constant uint JustFitTQ4HeadDimension = 256;
constant uint JustFitTQ4PackedWords = 32;
constant uint JustFitTQ4KVHeads = 4;
constant uint JustFitTQ4AttentionLayers = 16;
constant uint JustFitTQ4IndicesBytesPerLayerPage = 16384;
constant uint JustFitTQ4NormBytesPerLayerPage = 256;
constant uint JustFitTQ4BytesPerLayerPage = 33280;
constant uint JustFitTQ4BytesPerModelPage = 532480;

constant float JustFitTQ4Codebook[16] = {
    -0.168522835f, -0.127540648f, -0.0996119156f, -0.0771940202f,
    -0.0578124076f, -0.0402186327f, -0.0237036943f, -0.00783268455f,
     0.00783268549f, 0.0237036962f, 0.0402186289f, 0.0578124076f,
     0.0771940276f, 0.0996119007f, 0.127540648f, 0.168522835f};

constant float JustFitTQ4Midpoints[15] = {
    -0.148031741f, -0.113576278f, -0.0884029716f, -0.0675032139f,
    -0.049015522f, -0.0319611654f, -0.015768189f, 4.65661287e-10f,
     0.0157681908f, 0.0319611616f, 0.0490155183f, 0.0675032139f,
     0.0884029642f, 0.113576278f, 0.148031741f};

constant uint JustFitTQ4KeySignBits[8] = {
    0xe05422cdu, 0xf23e567du, 0xed27523du, 0x5a9acdcau,
    0x22d0370du, 0x10b3a0b8u, 0xf87f105fu, 0x3efbe5ceu};
constant uint JustFitTQ4ValueSignBits[8] = {
    0xa45beb87u, 0xac6b118eu, 0xeafcb960u, 0x7c1e2f3bu,
    0xeac834c2u, 0xac17ef56u, 0xd86514b1u, 0xe2c2bde9u};

inline float justfit_tq4_sign(bool value_tensor, uint dimension) {
  uint bits = value_tensor ? JustFitTQ4ValueSignBits[dimension >> 5]
                           : JustFitTQ4KeySignBits[dimension >> 5];
  return ((bits >> (dimension & 31u)) & 1u) ? 1.0f : -1.0f;
}

inline ulong justfit_tq4_vector_index(uint head, uint token) {
  return ulong(head) * JustFitTQ4PageTokens + token;
}

inline uint justfit_tq4_extract(device const uint *packed, ulong vector_index,
                               uint dimension) {
  uint word = packed[vector_index * JustFitTQ4PackedWords + (dimension >> 3)];
  return (word >> ((dimension & 7u) * 4u)) & 15u;
}

inline float justfit_tq4_value(device const uint *packed, ulong vector_index,
                              uint dimension) {
  return JustFitTQ4Codebook[
      justfit_tq4_extract(packed, vector_index, dimension)];
}

// Extent addressing replaces the old contiguous model-page binding.
struct JustFitTQ4PageView {
  device uint *keys;
  device uint *values;
  device half *key_norms;
  device half *value_norms;
};
inline JustFitTQ4PageView justfit_tq4_page(JustFitKvPage entry, JustFitKvLayer kv) {
  uint index = justfit_kv_page_index(entry);
  device uchar *region = justfit_kv_extent(entry,index) + kv.offset;
  constexpr uint data = JustFitTQ4IndicesBytesPerLayerPage;
  constexpr uint norm = JustFitTQ4NormBytesPerLayerPage;
  return {
    reinterpret_cast<device uint *>(region+justfit_kv_offset(kv.extent_pages,data,norm,0,JUSTFIT_KV_KEYS,index)),
    reinterpret_cast<device uint *>(region+justfit_kv_offset(kv.extent_pages,data,norm,0,JUSTFIT_KV_VALUES,index)),
    reinterpret_cast<device half *>(region+justfit_kv_offset(kv.extent_pages,data,norm,0,JUSTFIT_KV_KEY_SCALES,index)),
    reinterpret_cast<device half *>(region+justfit_kv_offset(kv.extent_pages,data,norm,0,JUSTFIT_KV_VALUE_SCALES,index))};
}
inline device uint *justfit_tq4_indices(JustFitTQ4PageView page, bool value) {
  return value ? page.values : page.keys;
}
inline device half *justfit_tq4_norms(JustFitTQ4PageView page, bool value) {
  return value ? page.value_norms : page.key_norms;
}
