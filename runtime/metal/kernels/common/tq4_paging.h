#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include <metal_stdlib>

using namespace metal;

// JustFit TurboQuant-Q4 ABI.  Each D=256 vector stores 32 uint32 words of
// packed nibbles and one fp16 L2 norm.  Both K and V are token-major inside a
// Page32 block.  The codebook and randomized-Hadamard signs exactly match the
// MLX JustFit codec (D=256, key seed 0, value seed 1).
constant uint SplashTQ4PageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant uint SplashTQ4HeadDimension = 256;
constant uint SplashTQ4PackedWords = 32;
constant uint SplashTQ4KVHeads = 4;
constant uint SplashTQ4AttentionLayers = 16;
constant uint SplashTQ4IndicesBytesPerLayerPage = 16384;
constant uint SplashTQ4NormBytesPerLayerPage = 256;
constant uint SplashTQ4BytesPerLayerPage = 33280;
constant uint SplashTQ4BytesPerModelPage = 532480;

constant float SplashTQ4Codebook[16] = {
    -0.168522835f, -0.127540648f, -0.0996119156f, -0.0771940202f,
    -0.0578124076f, -0.0402186327f, -0.0237036943f, -0.00783268455f,
     0.00783268549f, 0.0237036962f, 0.0402186289f, 0.0578124076f,
     0.0771940276f, 0.0996119007f, 0.127540648f, 0.168522835f};

constant float SplashTQ4Midpoints[15] = {
    -0.148031741f, -0.113576278f, -0.0884029716f, -0.0675032139f,
    -0.049015522f, -0.0319611654f, -0.015768189f, 4.65661287e-10f,
     0.0157681908f, 0.0319611616f, 0.0490155183f, 0.0675032139f,
     0.0884029642f, 0.113576278f, 0.148031741f};

constant uint SplashTQ4KeySignBits[8] = {
    0xe05422cdu, 0xf23e567du, 0xed27523du, 0x5a9acdcau,
    0x22d0370du, 0x10b3a0b8u, 0xf87f105fu, 0x3efbe5ceu};
constant uint SplashTQ4ValueSignBits[8] = {
    0xa45beb87u, 0xac6b118eu, 0xeafcb960u, 0x7c1e2f3bu,
    0xeac834c2u, 0xac17ef56u, 0xd86514b1u, 0xe2c2bde9u};

inline float splash_tq4_sign(bool value_tensor, uint dimension) {
  uint bits = value_tensor ? SplashTQ4ValueSignBits[dimension >> 5]
                           : SplashTQ4KeySignBits[dimension >> 5];
  return ((bits >> (dimension & 31u)) & 1u) ? 1.0f : -1.0f;
}

inline device uchar *splash_tq4_layer_page(device uchar *model_pages,
                                            uint page, uint layer) {
  return model_pages + ulong(page) * SplashTQ4BytesPerModelPage +
         ulong(layer) * SplashTQ4BytesPerLayerPage;
}

inline device const uchar *
splash_tq4_layer_page(device const uchar *model_pages, uint page, uint layer) {
  return model_pages + ulong(page) * SplashTQ4BytesPerModelPage +
         ulong(layer) * SplashTQ4BytesPerLayerPage;
}

inline ulong splash_tq4_vector_index(uint head, uint token) {
  return ulong(head) * SplashTQ4PageTokens + token;
}

inline uint splash_tq4_extract(device const uint *packed, ulong vector_index,
                               uint dimension) {
  uint word = packed[vector_index * SplashTQ4PackedWords + (dimension >> 3)];
  return (word >> ((dimension & 7u) * 4u)) & 15u;
}

inline float splash_tq4_value(device const uint *packed, ulong vector_index,
                              uint dimension) {
  return SplashTQ4Codebook[
      splash_tq4_extract(packed, vector_index, dimension)];
}

inline device uint *splash_tq4_indices(device uchar *layer_page,
                                       bool value_tensor) {
  uint offset = value_tensor ? SplashTQ4IndicesBytesPerLayerPage +
                                   SplashTQ4NormBytesPerLayerPage
                             : 0;
  return reinterpret_cast<device uint *>(layer_page + offset);
}

inline device const uint *splash_tq4_indices(device const uchar *layer_page,
                                             bool value_tensor) {
  uint offset = value_tensor ? SplashTQ4IndicesBytesPerLayerPage +
                                   SplashTQ4NormBytesPerLayerPage
                             : 0;
  return reinterpret_cast<device const uint *>(layer_page + offset);
}

inline device half *splash_tq4_norms(device uchar *layer_page,
                                     bool value_tensor) {
  uint offset = value_tensor
                    ? 2 * SplashTQ4IndicesBytesPerLayerPage +
                          SplashTQ4NormBytesPerLayerPage
                    : SplashTQ4IndicesBytesPerLayerPage;
  return reinterpret_cast<device half *>(layer_page + offset);
}

inline device const half *splash_tq4_norms(device const uchar *layer_page,
                                           bool value_tensor) {
  uint offset = value_tensor
                    ? 2 * SplashTQ4IndicesBytesPerLayerPage +
                          SplashTQ4NormBytesPerLayerPage
                    : SplashTQ4IndicesBytesPerLayerPage;
  return reinterpret_cast<device const half *>(layer_page + offset);
}
