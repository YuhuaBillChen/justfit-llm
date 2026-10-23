// Modified for JustFit: product identity and integration.
#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/tq4_attention_tile.h"
#include "metal/kernels/common/tq4_paging.h"
#include <metal_stdlib>

using namespace metal;

// Qwen3.8-27B target specialization. Queries and outputs retain JustFit's
// [lane][KV head][row][GQA head][D] placement. The first kernel rotates the
// queries into the TurboQuant key basis and applies 1/sqrt(D). The attention
// kernel consumes packed Q4 pages without materializing BF16 history, reduces
// its 32 token stripes, and performs the inverse value rotation before the
// result leaves the threadgroup.
constant uint JustFitTQ4QueryHeadsPerKVHead = 6;
constant uint JustFitTQ4QueryHeads =
    JustFitTQ4KVHeads * JustFitTQ4QueryHeadsPerKVHead;

inline ulong justfit_tq4_query_offset(uint lane, uint kv_head, uint row,
                                     uint gqa_head, uint chunk_stride,
                                     uint dimension) {
  return (((ulong(lane) * JustFitTQ4KVHeads + kv_head) * chunk_stride + row) *
              JustFitTQ4QueryHeadsPerKVHead +
          gqa_head) *
             JustFitTQ4HeadDimension +
         dimension;
}


template <uint QueryHeadsPerKVHead, uint RowsPerTile, typename T>
inline void vtq4_rot_softmax(
    threadgroup const float *scores, threadgroup T *probabilities,
    threadgroup const float *key_norms, threadgroup const float *value_norms,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint token_start, uint visible_tokens, uint committed_tokens,
    uint active_rows, uint thread_index) {
  constexpr uint N = JustFitTQ4PageTokens;
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
  float score[TokensPerLane];
  float local_max = -INFINITY;
#pragma unroll
  for (uint i = 0; i < TokensPerLane; ++i) {
    const float raw = scores[fused_row * N + column + i] * key_norms[column + i];
    score[i] = token + i < limit ? raw : -INFINITY;
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
#pragma unroll
  for (uint i = 0; i < TokensPerLane; ++i)
    probabilities[fused_row * N + column + i] =
        T(probability[i] * value_norms[column + i]);
}

// Codebook-only page decode. 1024 packed words per (page, head); each word
// holds eight consecutive dimensions of one token. Keys token-major [N,D],
// values dimension-major [D,N], matching the production MPP operands.
template <bool ValueTensor, typename T, bool Coalesced>
inline void vtq4_codebook_page(JustFitTQ4PageView layer_page, uint kv_head,
                               threadgroup T *page_values,
                               threadgroup float *norms_out,
                               uint thread_index) {
  constexpr uint D = JustFitTQ4HeadDimension;
  constexpr uint N = JustFitTQ4PageTokens;
  device const uint *words = justfit_tq4_indices(layer_page, ValueTensor) +
                             ulong(kv_head) * N * JustFitTQ4PackedWords;
  for (uint w = thread_index; w < N * JustFitTQ4PackedWords; w += 256) {
    // Value writes are dimension-major. Neighboring lanes own neighboring
    // tokens so each decoded dimension leaves as a contiguous shared store.
    const uint token = ValueTensor && Coalesced ? w % N : w / JustFitTQ4PackedWords;
    const uint d0 = (ValueTensor && Coalesced ? w / N : w % JustFitTQ4PackedWords) * 8;
    const uint word = words[token * JustFitTQ4PackedWords + d0 / 8];
#pragma unroll
    for (uint i = 0; i < 8; ++i) {
      const T c = T(JustFitTQ4Codebook[(word >> (4 * i)) & 15u]);
      if constexpr (ValueTensor)
        page_values[(d0 + i) * N + token] = c;
      else
        page_values[token * D + d0 + i] = c;
    }
  }
  if (thread_index < N)
    norms_out[thread_index] = float(
        justfit_tq4_norms(layer_page, ValueTensor)[justfit_tq4_vector_index(
            kv_head, thread_index)]);
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile, typename T, bool Coalesced>
inline void vtq4_rot_tile(
    device T *rotated_queries,
    device const JustFitKvPage *page_table, JustFitKvLayer kv, uint kv_head,
    uint committed_tokens, uint active_rows, uint splits, uint split,
    device float *partials, device float *statistics, ulong slot,
    threadgroup T *page_values, threadgroup float *scores,
    threadgroup T *probabilities, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup float *key_norms, threadgroup float *value_norms,
    threadgroup atomic_uint *rescale, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = JustFitTQ4PageTokens;
  constexpr ushort D = JustFitTQ4HeadDimension;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint pages = justfit_attention_pages(visible_tokens);
  const uint per_split = justfit_attention_pages_per_split(pages, splits);
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
      tensor(rotated_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto score_tensor =
      tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto probability_tensor = tensor(probabilities, dextents<int, 2>{N, M},
                                   array<int, 2>{1, N});
  auto probability_slice = probability_tensor.template slice<N, M>(0, 0);
  auto query_slice = queries.template slice<D, M>(0, 0);
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
    const JustFitTQ4PageView layer_page = justfit_tq4_page(page_table[page], kv);
    vtq4_codebook_page<false, T, Coalesced>(layer_page, kv_head, page_values, key_norms,
                              thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto keys = tensor(page_values, dextents<int, 2>{D, N},
                       array<int, 2>{1, D});
    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(query_slice), decltype(keys), float>();
    auto key_slice = keys.template slice<D, N>(0, 0);
    qk.run(query_slice, key_slice, page_scores);
    page_scores.store(score_tensor.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    if (thread_index < N)
      value_norms[thread_index] = float(
          justfit_tq4_norms(layer_page, true)[justfit_tq4_vector_index(
              kv_head, thread_index)]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vtq4_rot_softmax<QueryHeadsPerKVHead, RowsPerTile, T>(
        scores, probabilities, key_norms, value_norms, row_max, row_sum,
        previous_scale, rescale, page * N, visible_tokens, committed_tokens,
        active_rows, thread_index);
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
    vtq4_codebook_page<true, T, Coalesced>(layer_page, kv_head, page_values, value_norms,
                             thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto values = tensor(page_values, dextents<int, 2>{N, D},
                         array<int, 2>{1, N});
    auto value_slice = values.template slice<N, D>(0, 0);
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


#define VTQ4_ROT_SPLIT(T, SUFFIX, Coalesced) \
kernel void verify_attention_tq4_rot_split##SUFFIX( \
    device T *rotated_queries [[buffer(0)]], \
    device float *partials [[buffer(1)]], \
    device float *statistics [[buffer(2)]], \
    device const JustFitKvPage *page_table0 [[buffer(3)]], \
    device const JustFitKvPage *page_table1 [[buffer(4)]], \
    device const JustFitKvPage *page_table2 [[buffer(5)]], \
    device const JustFitKvPage *page_table3 [[buffer(6)]], \
    constant JustFitVerifyAttentionParams *params [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]]) { \
  constexpr uint Group = JustFitTQ4QueryHeadsPerKVHead; \
  constexpr uint Rows = 8; \
  constexpr uint Tiles = JUSTFIT_TARGET_VERIFY_ROWS / 8; \
  constexpr uint M = Group * Rows; \
  constexpr uint N = JustFitTQ4PageTokens; \
  alignas(16) threadgroup T page_values[N * JustFitTQ4HeadDimension]; \
  alignas(16) threadgroup float scores[M * N]; \
  alignas(16) threadgroup T probabilities[M * N]; \
  threadgroup float row_max[M]; \
  threadgroup float row_sum[M]; \
  threadgroup float previous_scale[M]; \
  threadgroup float key_norms[N]; \
  threadgroup float value_norms[N]; \
  threadgroup atomic_uint rescale; \
 \
  const uint kv_head = group.x / Tiles; \
  const uint tile = group.x % Tiles; \
  const uint split = group.y; \
  const uint lane = group.z; \
  constant JustFitVerifyAttentionParams &lane_params = params[lane]; \
  constant JustFitVerifyAttentionParams &attention = lane_params; \
  if (!justfit_verify_attention_contract_valid(attention) || \
      kv_head >= JustFitTQ4KVHeads || split >= attention.split_count) \
    return; \
  device const JustFitKvPage *page_table = \
      lane == 0 ? page_table0 \
                : (lane == 1 ? page_table1 \
                             : (lane == 2 ? page_table2 : page_table3)); \
  const ulong query_stride = \
      ulong(JUSTFIT_VERIFY_CHUNK_STRIDE) * Group * JustFitTQ4HeadDimension; \
  const ulong slot = \
      ((ulong(lane) * JustFitTQ4KVHeads + kv_head) * Tiles + tile) * attention.slot_splits + \
      split; \
  vtq4_rot_tile<Group, Rows, T, Coalesced>( \
      rotated_queries + (ulong(lane) * JustFitTQ4KVHeads + kv_head) * query_stride + tile * Rows * Group * JustFitTQ4HeadDimension, \
      page_table, lane_params.kv, kv_head, \
      attention.committed_tokens + tile * Rows, min(Rows, JUSTFIT_TARGET_VERIFY_ROWS - tile * Rows), \
      attention.split_count, split, partials, statistics, slot, page_values, \
      scores, probabilities, row_max, row_sum, previous_scale, key_norms, \
      value_norms, &rescale, thread_index); \
}

VTQ4_ROT_SPLIT(half, _rows8, false)
VTQ4_ROT_SPLIT(half, _rows8_coalesced, true)
#undef VTQ4_ROT_SPLIT



kernel void verify_attention_tq4_rot_reduce_rows8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant JustFitVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Group = JustFitTQ4QueryHeadsPerKVHead;
  constexpr uint Rows = 8;
  constexpr uint Tiles = JUSTFIT_TARGET_VERIFY_ROWS / 8;
  constexpr uint M = Group * Rows;
  constexpr uint D = JustFitTQ4HeadDimension;
  const uint kv_head = group.x / Tiles;
  const uint tile = group.x % Tiles;
  const uint fused_row = group.y;
  const uint lane = group.z;
  constant JustFitVerifyAttentionParams &attention = params[lane];
  if (!justfit_verify_attention_contract_valid(attention) ||
      kv_head >= JustFitTQ4KVHeads || fused_row >= M || thread_index >= D)
    return;
  const ulong head_slot =
      ((ulong(lane) * JustFitTQ4KVHeads + kv_head) * Tiles + tile) * attention.slot_splits;
  const uint query_row = tile * Rows + fused_row / Group;
  threadgroup float values[D];
  float value = 0.0f;
  if (query_row < JUSTFIT_TARGET_VERIFY_ROWS) {
    const uint pages =
        justfit_attention_pages(attention.committed_tokens + tile * Rows + min(Rows, JUSTFIT_TARGET_VERIFY_ROWS - tile * Rows));
    const uint per_split =
        justfit_attention_pages_per_split(pages, attention.split_count);
    const uint written = (pages + per_split - 1) / per_split;
    float maximum = -INFINITY;
    for (uint split = 0; split < written; ++split)
      maximum = max(maximum,
                    statistics[((head_slot + split) * M + fused_row) * 2]);
    float numerator = 0.0f, denominator = 0.0f;
    for (uint split = 0; split < written; ++split) {
      const ulong stat = ((head_slot + split) * M + fused_row) * 2;
      const float weight = fast::exp(statistics[stat] - maximum);
      numerator +=
          weight * partials[((head_slot + split) * M + fused_row) * D +
                            thread_index];
      denominator += weight * statistics[stat + 1];
    }
    value = denominator > 0.0f ? numerator / denominator : 0.0f;
  }
  values[thread_index] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint half_span = 1; half_span < D; half_span <<= 1) {
    if ((thread_index & half_span) == 0) {
      const float a = values[thread_index];
      const float b = values[thread_index + half_span];
      values[thread_index] = a + b;
      values[thread_index + half_span] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong query_stride = ulong(JUSTFIT_VERIFY_CHUNK_STRIDE) * Group * D;
  output[(ulong(lane) * JustFitTQ4KVHeads + kv_head) * query_stride +
         (tile * Rows * Group + fused_row) * D + thread_index] =
      bfloat(values[thread_index] * (1.0f / 16.0f) *
             justfit_tq4_sign(true, thread_index));
}


kernel void verify_attention_tq4_rotate_queries_half_rows8(
    device const bfloat *queries [[buffer(0)]],
    device half *rotated [[buffer(1)]],
    constant JustFitChunkedPrefillParams *params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
  const uint lane_vectors = JustFitTQ4QueryHeads * JUSTFIT_TARGET_VERIFY_ROWS;
  const uint lane = group / lane_vectors;
  const uint vector = group - lane * lane_vectors;
  const uint query_head = vector / JUSTFIT_TARGET_VERIFY_ROWS;
  const uint row = vector - query_head * JUSTFIT_TARGET_VERIFY_ROWS;
  const uint kv_head = query_head / JustFitTQ4QueryHeadsPerKVHead;
  const uint gqa_head = query_head % JustFitTQ4QueryHeadsPerKVHead;
  constant JustFitChunkedPrefillParams &chunk = params[lane];
  if (row >= chunk.chunk_tokens || dimension >= JustFitTQ4HeadDimension)
    return;
  threadgroup float values[JustFitTQ4HeadDimension];
  const ulong offset = justfit_tq4_query_offset(
      lane, kv_head, row, gqa_head, chunk.chunk_stride, dimension);
  values[dimension] = float(queries[offset]) * justfit_tq4_sign(false, dimension);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint half_span = 1; half_span < JustFitTQ4HeadDimension; half_span <<= 1) {
    if ((dimension & half_span) == 0) {
      const float a = values[dimension];
      const float b = values[dimension + half_span];
      values[dimension] = a + b;
      values[dimension + half_span] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  rotated[offset] = half(values[dimension] * (1.0f / 256.0f));
}


