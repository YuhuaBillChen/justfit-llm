// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output (fp32 from
// the _a_f32 instances, the logits).
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

// ---------------- prefill tiles: a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) skip their matmuls and stores, so a chunk costs its rows rounded up to RowsPerSG rather than
// to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile). Full tiles run this
// loop too; a branch-free copy for them is no faster (M3 Max, M5 Max).
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, GgufEpilogue Ep = EpNone>
inline void gguf_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint input_size, uint output_origin, uint rows, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride, uint out_offset, device bfloat *aux = nullptr) {
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto acc = staged_accumulator<RowsPerSG, TileN, KS>(rows_input, input_size, stage);
  gguf_zero(acc);
  gguf_staged_steps<F, RowsPerSG, TileN, KS, Simdgroups * 32>(rows_input, w0, w1, meta, input_size, output_origin, stage, tl,
                                                              simd_group * 32 + simd_lane, 0, input_size / KS, owns_rows,
                                                              acc);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = gguf_epilogue<Ep>(acc[i], aux, o);
  }
}

// The two stages of a prefill tile.
constant constexpr uint kPrefillStages = 2 * GGUF_TILE_COLUMNS * GGUF_PREFILL_STEP;

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows (these kernels and the fused ones): a 3-lane step
// runs the 32-row tile over the storage of four lanes (LinearPlan::storageRows) and the padding lane's rows are
// computed and discarded. Rows are independent, so every active row is the bits of any other tile height
// (gguf-projection checks it); the 32-row tile at three lanes costs what it costs at four, 3-15% less than a 16-row
// plus an 8-row matmul per stage (DRAM-cold, 16- and 20-core M5 Pro).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do. The destination's type Out is bf16, or fp32 for the plain epilogue's
// logits (a_f32, ops::Projection::destination).
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per,
                                          (group.y + 1) * per, acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// The staged decode kernels: grid (column tiles, K partitions), two simdgroups.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[(F::Kind == QuantCodebook || quant_has_table<F>::value) ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE(F, f, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode<F, R, Ep, Out>;
#define GGUF_DECODE_ROWS(F, f, ep, Ep, Out) \
  GGUF_DECODE(F, f, 8, ep, Ep, Out) GGUF_DECODE(F, f, 16, ep, Ep, Out) GGUF_DECODE(F, f, 32, ep, Ep, Out)
#define GGUF_DECODE_FORMAT(F, f)                                                                                  \
  GGUF_DECODE_ROWS(F, f, a, EpNone, bfloat) GGUF_DECODE_ROWS(F, f, a_f32, EpNone, float)                          \
  GGUF_DECODE_ROWS(F, f, r, EpResidual, bfloat) GGUF_DECODE_ROWS(F, f, g, EpUpWithGate, bfloat)
QUANT_FORMATS(GGUF_DECODE_FORMAT)
#undef GGUF_DECODE_FORMAT
#undef GGUF_DECODE_ROWS
#undef GGUF_DECODE

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits. The fused and prefill kernels stay plain kernels: as
// template instantiations (weak_odr) the compiler infers fewer parameter attributes and inlines differently.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_FUSED(R)                                                                                       \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], GGUF_SEGMENT(1, w0a, w1a, ma),          \
                                     GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc),                \
                                     device bfloat *output [[buffer(10)]],                                        \
                                     device coherent(device) float *partials [[buffer(11)]],                      \
                                     device atomic_uint *counters [[buffer(12)]],                                 \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]],                            \
                                     uint2 group [[threadgroup_position_in_grid]],                                \
                                     uint simd_lane [[thread_index_in_simdgroup]],                                \
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                        \
    threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries]; threadgroup uint arrival; \
    const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;                        \
    const uint s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;                                                       \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                         \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                         \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                          \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / GGUF_STAGED_STEP / p.splits; \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin; \
    threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;                                            \
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);                                         \
    gguf_zero(acc);                                                                                               \
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, \
                                            simd_group * 32 + simd_lane, simd_lane, group.y * per, (group.y + 1) * per, acc); \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,       \
                       p.out_stride, column0, simd_group * 32 + simd_lane, &arrival,                              \
                       [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
GGUF_DECODE_FUSED(8) GGUF_DECODE_FUSED(16) GGUF_DECODE_FUSED(32)
#undef GGUF_DECODE_FUSED

// ---------------- int8-activation decode of rotated projections (ops::LinearConfig::decodeInt8) in the formats whose
// codes are int8 exactly (quant_decode_int8_format: PQ2_0 and PTQ1_0, codes q standing for q - 1 with one d per
// 128-input block), from Splash 1.6.1 (engine 767c646) on these staged decode tiles. gguf_rotate_int8 writes the
// rotated input as int8 rows with one scale s per row and rotation block (metal/abi/Gguf.h). The tiles keep the
// staged decode tiles' grid, K splits, split reductions (gguf_store_sums), epilogues, parameters and bindings; per step
// each lane stages its column's 32 codes as int8 q - 1 in {-1, 0, 1, 2} in the order of the int8 rows
// (gguf_decode_int8_position), and one int8 matmul2d accumulates in int32 over a block's four steps (|acc| <= 2 * 128
// * 127, exact); the block's last step folds it into the fp32 sums as acc * d * s. Each K partition holds whole
// blocks (the host's LinearPlan checks it). The tiles hold 16 or 32 rows: the 8-row int8 matmul takes about as long
// as the 16-row one.

// A group's 2-bit codes as PQ2_0's plane0 words: word w holds chunks 2w, 2w + 1, slot s of chunk c at bit 16 (c & 1)
// + 2 s. PTQ1_0 decodes each chunk through the pair table and packs its spread codes (quant_spread2) back.
template <class F>
inline uint2 gguf_int8_words(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl) {
  if constexpr (F::Id == GGUF_FMT_PQ20) {
    return w.a;
  } else {
    static_assert(F::Id == GGUF_FMT_PTQ10, "the int8 stage takes PQ2_0 and PTQ1_0");
    uint2 words = 0;
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
      const uint q = F::chunk(w, c, meta, j, tl);
      words[c >> 1] |= ((q & 0x3333u) | ((q >> 14) & 0xCCCCu)) << (16 * (c & 1));
    }
    return words;
  }
}
// Stages `step` of the lane's column as int8: its 32 codes q as q - 1 ((q + 0x7F) ^ 0x80 a byte: bytes 0x7F..0x82
// before the xor, no carries) at the lane's 32 bytes of `stage`, and at a block's last step the block's d at d[lane].
template <class F>
inline void gguf_int8_stage(typename F::Payload w, typename F::Meta meta, uint step, threadgroup half2 *tl,
                            threadgroup char *stage, threadgroup float *d, uint lane) {
  static_assert(F::Kind == QuantLinear && F::Zero == 1 && F::MetaGroups == 4,
                "the int8 stage takes two-bit codes of value d (q - 1), one d per 128 inputs");
  const uint2 words = gguf_int8_words<F>(w, meta, step % F::MetaGroups, tl);
  threadgroup uint *dst = (threadgroup uint *)(stage + lane * GGUF_STAGED_STEP);
#pragma unroll
  for (ushort e = 0; e < 4; ++e) {
    dst[e] = (((words.x >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
    dst[4 + e] = (((words.y >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
  }
  if (step % F::MetaGroups == F::MetaGroups - 1) d[lane] = F::coef(meta, 0).s.x;
}
// A block's int32 sums into the fp32 sums, sums += acc * d[column] * s[row][block], acc reset. The two cooperative
// tensors are destinations of one matmul2d, element i the same (row, column) in both.
template <class Acc, class Sums>
inline void gguf_int8_fold(thread Acc &acc, thread Sums &sums, threadgroup const float *d, device const float *s,
                           uint blocks, uint block) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      const auto index = acc.get_multidimensional_index(i);
      sums[i] = fma(float(acc[i]), d[index[0]] * s[uint(index[1]) * blocks + block], float(sums[i]));
    }
    acc[i] = 0;
  }
}
// The int8 rows of a Rows-row tile (input [Rows][K] int8, then their scales) and a stage of 32 columns, as matmul2d
// operands.
template <ushort Rows> inline auto gguf_int8_rows(device char *input, uint input_size) {
  return tensor<device int8_t, dextents<int, 2>, tensor_inline>((device int8_t *)input,
                                                                dextents<int, 2>{int(input_size), Rows},
                                                                array<int, 2>{1, int(input_size)});
}
inline auto gguf_int8_operand(threadgroup char *stage) {
  return tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>(
             (threadgroup int8_t *)stage, dextents<int, 2>{GGUF_STAGED_STEP, GGUF_STAGED_COLUMNS},
             array<int, 2>{1, GGUF_STAGED_STEP})
      .template slice<GGUF_STAGED_STEP, GGUF_STAGED_COLUMNS>(0, 0);
}
template <ushort Rows>
using GgufInt8Matmul = matmul2d<matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate),
                                execution_simdgroups<1>>;
// The destination of a Rows-row int8 tile, of int (a block's sums) or float (the folded sums), not initialized, as
// staged_accumulator's.
template <ushort Rows, class T>
inline auto gguf_int8_destination(device char *input, uint input_size, threadgroup char *stage) {
  GgufInt8Matmul<Rows> op;
  auto a0 = gguf_int8_rows<Rows>(input, input_size).template slice<GGUF_STAGED_STEP, Rows>(0, 0);
  auto b0 = gguf_int8_operand(stage);
  return op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), T>();
}

// The int8 decode tile loop of one simdgroup without the store, as staged_accumulate's: its GGUF_STAGED_COLUMNS
// columns from `origin`, lane l column origin + l, over steps [step_begin, step_end) of K, both on block boundaries,
// folded into `sums`; `stages` holds two stages, d a block's d of each column, tl PTQ1_0's pair table.
template <class F, ushort Rows, class Sums>
inline void gguf_int8_accumulate(device char *input, device uchar *w0, device uchar *w1, device uchar *meta,
                                 uint input_size, uint origin, threadgroup char *stages, threadgroup float *d,
                                 threadgroup half2 *tl, uint lane, uint step_begin, uint step_end, thread Sums &sums) {
  constexpr ushort Stage = GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP;
  auto rows = gguf_int8_rows<Rows>(input, input_size);
  GgufInt8Matmul<Rows> op;
  auto b0 = gguf_int8_operand(stages), b1 = gguf_int8_operand(stages + Stage);
  auto acc = gguf_int8_destination<Rows, int>(input, input_size, stages);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  device const float *scales = (device const float *)(input + ulong(Rows) * input_size);
  const uint groups = input_size / 32, units = groups / F::MetaGroups, blocks = input_size / GGUF_ROTATION_BLOCK;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + lane;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  // Every partition holds at least one block, so step_begin is a step of K.
  typename F::Payload packed =
      F::load(tw0 + ulong(step_begin) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(step_begin) * QUANT_TILE_ROWS * F::P1);
  typename F::Meta hdr = F::loadMeta(tmeta + ulong(step_begin / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  for (uint step = step_begin; step < step_end; ++step) {
    gguf_int8_stage<F>(packed, hdr, step, tl, stages + (step & 1) * Stage, d, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < step_end) {
      const uint next = step + 1;
      packed = F::load(tw0 + ulong(next) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(next) * QUANT_TILE_ROWS * F::P1);
      if (next % F::MetaGroups == 0)
        hdr = F::loadMeta(tmeta + ulong(next / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
    }
    auto a = rows.template slice<GGUF_STAGED_STEP, Rows>(step * GGUF_STAGED_STEP, 0);
    if (step & 1) op.run(a, b1, acc); else op.run(a, b0, acc);
    if (step % F::MetaGroups == F::MetaGroups - 1)
      gguf_int8_fold(acc, sums, d, scales, blocks, step * GGUF_STAGED_STEP / GGUF_ROTATION_BLOCK);
  }
}
// The int8 decode kernels, as gguf_decode's: grid (column tiles, K partitions), two simdgroups. Their stores take the
// parameters they need by value, as upstream's: under shader validation a load through the constant parameters'
// address that a store's closure held read as 0 in 1 run of 30 there.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_int8(device char *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                             device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                             device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                             device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[quant_has_table<F>::value ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup char stages[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup float d[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  auto sums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[simd_group]);
  gguf_zero(sums);
  gguf_int8_accumulate<F, Rows>(input, w0, w1, meta, p.input_size, origin, stages[simd_group], d[simd_group], tl,
                                simd_lane, group.y * per, (group.y + 1) * per, sums);
  const uint stride = p.out_stride;
  gguf_store_sums<Rows>(sums, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x,
                        stride, column0, simd_group * 32 + simd_lane, &arrival, [=](uint row, uint column, float v) {
    const ulong o = ulong(row) * stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// A fused projection's segments in one dispatch, as gguf_decode_fused_m<rows> runs them, every segment of the
// kernel's format: the threadgroup's tile picks its segment.
template <class F, ushort Rows>
kernel void gguf_decode_int8_fused(device char *input [[buffer(0)]], GGUF_SEGMENT(1, w0a, w1a, ma),
                                   GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc),
                                   device bfloat *output [[buffer(10)]],
                                   device coherent(device) float *partials [[buffer(11)]],
                                   device atomic_uint *counters [[buffer(12)]],
                                   constant GgufDecodeFusedParams &p [[buffer(13)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[quant_has_table<F>::value ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup char stages[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup float d[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;
  const uint s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;
  device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;
  device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;
  device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;
  const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / GGUF_STAGED_STEP / p.splits;
  const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin;
  auto sums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[simd_group]);
  gguf_zero(sums);
  gguf_int8_accumulate<F, Rows>(input, w0, w1, meta, p.input_size, origin, stages[simd_group], d[simd_group], tl,
                                simd_lane, group.y * per, (group.y + 1) * per, sums);
  const uint stride = p.out_stride;
  gguf_store_sums<Rows>(sums, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,
                        stride, column0, simd_group * 32 + simd_lane, &arrival,
                        [=](uint row, uint column, float v) { output[ulong(row) * stride + column0 + column] = bfloat(v); });
}
template <class Out>
using GgufDecodeInt8Kernel = void(device char *, device uchar *, device uchar *, device uchar *, device Out *,
                                  device coherent(device) float *, device atomic_uint *, device bfloat *,
                                  constant GgufDecodeParams &, uint2, uint, uint);
using GgufDecodeInt8FusedKernel = void(device char *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device bfloat *, device coherent(device) float *, device atomic_uint *,
                                       constant GgufDecodeFusedParams &, uint2, uint, uint);
#define GGUF_DECODE_INT8(F, f, R, ep, Ep, Out)                                                                     \
  template [[host_name("gguf_decode_int8_" #f "_m" #R "_" #ep)]] kernel GgufDecodeInt8Kernel<Out>                  \
      gguf_decode_int8<F, R, Ep, Out>;
#define GGUF_DECODE_INT8_ROWS(F, f, R)                                                                             \
  GGUF_DECODE_INT8(F, f, R, a, EpNone, bfloat) GGUF_DECODE_INT8(F, f, R, r, EpResidual, bfloat)                    \
  GGUF_DECODE_INT8(F, f, R, g, EpUpWithGate, bfloat) GGUF_DECODE_INT8(F, f, R, a_f32, EpNone, float)               \
  template [[host_name("gguf_decode_int8_" #f "_fused_m" #R)]] kernel GgufDecodeInt8FusedKernel                    \
      gguf_decode_int8_fused<F, R>;
// The formats the int8 tiles take (quant_decode_int8_format).
#define GGUF_DECODE_INT8_FORMAT(F, f) GGUF_DECODE_INT8_ROWS(F, f, 16) GGUF_DECODE_INT8_ROWS(F, f, 32)
GGUF_DECODE_INT8_FORMAT(FmtPQ20, pq20)
GGUF_DECODE_INT8_FORMAT(FmtPTQ10, ptq10)
#undef GGUF_DECODE_INT8_FORMAT
#undef GGUF_DECODE_INT8_ROWS
#undef GGUF_DECODE_INT8
#undef GGUF_SEGMENT

// The prefill kernels: grid (GGUF_PREFILL_ROWS-row tiles of the chunk, column tiles of the segment), four
// simdgroups; the residual and gate kernels read aux at buffer 5, the plain one binds none.
#define GGUF_PREFILL_BUFFERS                                                                                       \
  device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],             \
      device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]]
#define GGUF_PREFILL_THREAD                                                                                        \
  uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define GGUF_PREFILL_TABLES(F)                                                                                     \
  threadgroup half2 tl[(F::Kind == QuantCodebook || quant_has_table<F>::value) ? kQuantPairTableEntries : 1];                                     \
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_PREFILL_THREADS);                                      \
  threadgroup half stage[kPrefillStages]
#define GGUF_PREFILL(F, f)                                                                                         \
  kernel void gguf_prefill_##f##_a(GGUF_PREFILL_BUFFERS, constant GgufPrefillParams &p [[buffer(5)]],            \
                                   GGUF_PREFILL_THREAD) {                                                          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP>(input + ulong(first) * p.input_size, w0, w1, meta,                        \
                                         output + ulong(first) * p.out_stride, p.input_size,                       \
                                         group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,      \
                                         p.out_stride, p.out_offset);                                              \
  }
#define GGUF_PREFILL_EPILOGUE(F, f, ep, Ep)                                                                        \
  kernel void gguf_prefill_##f##_##ep(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],                    \
                                      constant GgufPrefillParams &p [[buffer(6)]], GGUF_PREFILL_THREAD) {          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, Ep>(input + ulong(first) * p.input_size, w0, w1, meta,                    \
                                             output + ulong(first) * p.out_stride, p.input_size,                   \
                                             group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,  \
                                             p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);       \
  }
#define GGUF_PREFILL_FORMAT(F, f) \
  GGUF_PREFILL(F, f) GGUF_PREFILL_EPILOGUE(F, f, r, EpResidual) GGUF_PREFILL_EPILOGUE(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_PREFILL_FORMAT)
#undef GGUF_PREFILL_FORMAT

// The residual kernels over a view of the leading inputs of wider weight rows, for the Neural Engine FFN split's down
// projection, one per format. Each runs the tile of its residual kernel on the planes advanced past the groups and
// meta units the view leaves unread in the rows of the plane tiles before its column tile, which the tile then
// addresses as rows of input_size inputs.
#define GGUF_PREFILL_LEADING_INPUTS(F, f)                                                                          \
  kernel void gguf_prefill_##f##_r_leading_inputs(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],          \
                                                  constant GgufPrefillLeadingParams &leading [[buffer(6)]],        \
                                                  GGUF_PREFILL_THREAD) {                                           \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    constant GgufPrefillParams &p = leading.prefill;                                                               \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    const ulong before = ulong(group.y * GGUF_TILE_COLUMNS / QUANT_TILE_ROWS) * QUANT_TILE_ROWS;                   \
    const uint groups = p.input_size / 32, plane_groups = leading.plane_input_size / 32;                           \
    const ulong unread = before * (plane_groups - groups),                                                         \
                unread_units = before * (plane_groups / F::MetaGroups - groups / F::MetaGroups);                   \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, \
                      EpResidual>(input + ulong(first) * p.input_size, w0 + unread * F::P0, w1 + unread * F::P1,   \
                                  meta + unread_units * F::MetaBytes, output + ulong(first) * p.out_stride,        \
                                  p.input_size, group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane,           \
                                  simd_group, p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);      \
  }
QUANT_FORMATS(GGUF_PREFILL_LEADING_INPUTS)
#undef GGUF_PREFILL_LEADING_INPUTS
#undef GGUF_PREFILL_EPILOGUE
#undef GGUF_PREFILL
#undef GGUF_PREFILL_TABLES
#undef GGUF_PREFILL_THREAD
#undef GGUF_PREFILL_BUFFERS
