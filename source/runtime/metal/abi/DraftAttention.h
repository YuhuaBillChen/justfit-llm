// Modified for JustFit: product identity and integration.
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The draft attention the kernels are compiled for, the only one
// ops::DraftAttention accepts: 32 query heads over 8 KV heads of 128
// dimensions, so a q|k|v row holds 6144 values and an attention row 4096.
// The attention core runs a KV head's four query heads over the eight query
// rows together, as 32 attention rows.
#define JUSTFIT_DRAFT_QUERY_HEADS 32u
#define JUSTFIT_DRAFT_KV_HEADS 8u
#define JUSTFIT_DRAFT_HEAD_DIMENSION 128u
#define JUSTFIT_DRAFT_ATTENTION_WIDTH                                           \
  (JUSTFIT_DRAFT_QUERY_HEADS * JUSTFIT_DRAFT_HEAD_DIMENSION)
#define JUSTFIT_DRAFT_QKV_WIDTH                                                 \
  (JUSTFIT_DRAFT_ATTENTION_WIDTH +                                              \
   2u * JUSTFIT_DRAFT_KV_HEADS * JUSTFIT_DRAFT_HEAD_DIMENSION)
#define JUSTFIT_DRAFT_ATTENTION_ROWS                                            \
  (JUSTFIT_DRAFT_QUERY_HEADS / JUSTFIT_DRAFT_KV_HEADS * JUSTFIT_DRAFT_QUERY_ROWS)

// The dynamic convolutions of a draft layer (draft_conv), each run in two
// stages, before its attention or MLP and after it with the residual: a
// stage weighs a channel's row and the row before it, its two taps, each by
// a base weight plus the dynamic weight of the channel's group of 16
// channels. The draft projects every row's dynamic weights, for each stage,
// tap and group of a `hidden`-wide layer.
#define JUSTFIT_DRAFT_CONVOLUTION_STAGES 2u
#define JUSTFIT_DRAFT_CONVOLUTION_TAPS 2u
#define JUSTFIT_DRAFT_CONVOLUTION_GROUP 16u
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden) {
  return JUSTFIT_DRAFT_CONVOLUTION_STAGES * JUSTFIT_DRAFT_CONVOLUTION_TAPS *
         (hidden / JUSTFIT_DRAFT_CONVOLUTION_GROUP);
}

// The batched draft kernels' grids cover exactly the dispatch's lanes:
// per-lane arrays hold those lanes, and entries past them are zero and unread.

struct DraftConvBatchParams {
  uint32_t finish;
};

static_assert(sizeof(DraftConvBatchParams) == 4,
              "Draft convolution parameters are 4 bytes on both sides");

// The attention core's split count (JUSTFIT_DRAFT_ATTENTION_SPLITS) and its
// rings' slots per KV head (JUSTFIT_DRAFT_SLIDING_WINDOW) are compiled in.
// value_stride, the values ring's stride between head dimensions, is the
// window too (the op passes nothing else) but stays a run-time value: with it
// compiled into the value tiles, the split kernel returns wrong rows at random
// on an M5 Max, and reports no error, when MTL_SHADER_VALIDATION instruments
// threadgroup memory, tensors and resource usage together; with any of the
// three off, without validation, or on an M3 Max, its output is bitwise that
// of the run-time stride.
struct DraftAttentionBatchParams {
  uint32_t value_stride;
  uint32_t lanes;
  uint32_t cache_length[JUSTFIT_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) == 24,
              "Draft attention parameters are 24 bytes on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t start_position;
};

static_assert(sizeof(DraftContextParams) == 8,
              "Draft context prefill parameters are 8 bytes on both sides");

struct DraftContextBatchParams {
  uint32_t start_position[JUSTFIT_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) == 16,
              "Draft context commit parameters are 16 bytes on both sides");
