// Modified for JustFit: ternary-weight A8 prefill.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/abi/KernelABI.h"
#include "metal/abi/Linear.h"
#include "metal/kernels/common/quant_formats.h"

// W2A8 prefill projection for Apple10 tensor units over Prism's ternary GGUF
// image tensors (PQ2_0, PTQ1_0): int8 activations times the weights' 2-bit
// codes q in {0, 1, 2}, int8 x int8 -> int32, then w = d (q - 1) applied as the
// affine correction of decode/linear_q4_a8.metal, s = d, b = -d per group of
// 64 inputs (d holds for 128):
//   y += a[m,g] * (d[g,n] * P - d[g,n] * R[m,g]).
// Tiling, fragment order and the activation quantizer are linear_q4_a8's
// (verify_linear_q4a8_quantize stores each 16-input run in the order
// 8*(t>>1) + 2j + (t&1) that its codes take here); only the weight codes and
// d come from the GGUF planes (QuantFormat.h tiles of QUANT_TILE_ROWS rows).

constant constexpr ushort T2A8Columns = 128;
enum class T2A8Epilogue { None, Residual, UpSilu };

// Each K group of 64 inputs, the threadgroup decodes its 128 columns' codes
// into threadgroup memory first, one column per thread, two GGUF groups of 32
// each (coalesced along the plane tiles; PTQ1_0 through the pair table), and
// each column's d; the simdgroups then read their fragments from it.
template <class F, T2A8Epilogue Epilogue>
inline void t2a8_tile(device const char *codes, device const float *scale, device const float *code_sums,
                      device uchar *plane0, device uchar *plane1, device uchar *meta,
                      device const bfloat *auxiliary, device bfloat *output, constant Q4A8Params &params,
                      uint2 group, uint simd_group, uint lane,
                      threadgroup uint *wtile,   // [column][128 codes], 4 per uint
                      threadgroup float *dtile, threadgroup half2 *tl) {
  const uint tid = simd_group * 32 + lane;
  quant_pair_table<F>(tl, tid, 256);

  constexpr auto accumulate = matmul2d_descriptor(
      16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto assign = matmul2d_descriptor(
      16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<accumulate, execution_simdgroup> op;
  matmul2d<assign, execution_simdgroup> op_set;
  auto left = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto right = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(left), decltype(right), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(left), decltype(right), int32_t>();

  const uint K = params.input_size, N = params.output_size, rows = params.rows;
  const uint groups = K / 64, groups32 = K / 32;
  const auto origin = left.get_multidimensional_index(ushort(0));
  const uint cx = origin[0], cy = origin[1], run = cx >> 2;
  // Eight simdgroups: four column quarters by two halves of the 64 rows.
  const uint row_base = group.y * 64 + (simd_group >> 2) * 32;
  const uint tile_base = group.x * T2A8Columns;
  const uint col_local = (simd_group & 3) * 32;
  const uint col_base = tile_base + col_local;
  const uint m_base = row_base + cy;

  float result[2][16];
#pragma clang loop unroll(full)
  for (ushort i = 0; i < 2; ++i)
#pragma clang loop unroll(full)
    for (ushort e = 0; e < 16; ++e) result[i][e] = 0.0f;

  for (uint g2 = 0; g2 < groups / 2; ++g2) {
    // Stage 128 inputs: 128 columns x 4 GGUF groups, two per thread.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort u = 0; u < 2; ++u) {
      const uint item = tid + 256 * u, column = item & 127, local = item >> 7, g32 = g2 * 4 + local;
      const uint n_decode = tile_base + column;
      const ulong unit = quant_tile_index(n_decode, g32, groups32);
      const typename F::Payload w = F::load(plane0 + unit * F::P0, plane1 + unit * F::P1);
      const typename F::Meta header =
          F::loadMeta(meta + quant_tile_index(n_decode, g32 / F::MetaGroups, groups32 / F::MetaGroups) * F::MetaBytes);
      const ushort j = ushort(g32 % F::MetaGroups);
      if (local == 0) dtile[column] = F::coef(header, j).s.x;
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 4; ++c) {
        typename F::Chunk q;
        if constexpr (quant_meta_chunk<F>::value) q = F::chunk(w, c, header, j, tl);
        else q = F::chunk(w, c);
        const uint4 pairs = F::codes(q);
#pragma clang loop unroll(full)
        for (ushort h = 0; h < 2; ++h) {
          const uint a = h ? pairs.z : pairs.x, b = h ? pairs.w : pairs.y;
          wtile[column * 32 + local * 8 + h * 4 + c] =
              (a & 0xFFFFu) | (a >> 16) << 8 | (b & 0xFFFFu) << 16 | (b >> 16) << 24;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort sub = 0; sub < 2; ++sub) {
    const uint g = g2 * 2 + sub;
#pragma clang loop unroll(full)
    for (ushort t = 0; t < 4; ++t) {
#pragma clang loop unroll(full)
      for (ushort q = 0; q < 4; ++q) {
        const ushort base = (q >> 1) * 8 + (q & 1) * 4, s = t >> 1, parity = t & 1;
        const uint column = col_local + cy + (q & 1) * 8 + (q >> 1) * 16;
        const uint2 pair = *reinterpret_cast<threadgroup const uint2 *>(&wtile[column * 32 + sub * 16 + run * 4 + 2 * s]);
        const char4 lo = as_type<char4>(pair.x), hi = as_type<char4>(pair.y);
        right[base + 0] = parity ? lo.y : lo.x; right[base + 1] = parity ? lo.w : lo.z;
        right[base + 2] = parity ? hi.y : hi.x; right[base + 3] = parity ? hi.w : hi.z;
      }
#pragma clang loop unroll(full)
      for (ushort half_rows = 0; half_rows < 2; ++half_rows) {
#pragma clang loop unroll(full)
        for (ushort r = 0; r < 2; ++r) {
          const uint m = m_base + half_rows * 16 + r * 8;
          const char4 quad = m < rows ? as_type<char4>(*reinterpret_cast<device const uint *>(
              codes + ulong(m) * K + g * 64 + run * 16 + t * 4)) : char4(0);
          left[r * 4 + 0] = quad.x; left[r * 4 + 1] = quad.y;
          left[r * 4 + 2] = quad.z; left[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (half_rows == 0) op_set.run(left, right, acc0);
          else op_set.run(left, right, acc1);
        } else {
          if (half_rows == 0) op.run(left, right, acc0);
          else op.run(left, right, acc1);
        }
      }
    }
    float a[2][2], sum[2][2];
#pragma clang loop unroll(full)
    for (ushort i = 0; i < 2; ++i)
#pragma clang loop unroll(full)
      for (ushort r = 0; r < 2; ++r) {
        const ulong index = ulong(g) * rows + m_base + i * 16 + r * 8;
        const bool valid = m_base + i * 16 + r * 8 < rows;
        a[i][r] = valid ? scale[index] : 0.0f;
        sum[i][r] = valid ? code_sums[index] : 0.0f;
      }
#pragma clang loop unroll(full)
    for (ushort e = 0; e < 16; ++e) {
      const ushort r = (e & 7) >> 2;
      const float dw = dtile[col_local + (e >> 3) * 16 + cx + (e & 3)];
      result[0][e] = fma(a[0][r] * dw, float(acc0[e]) - sum[0][r], result[0][e]);
      result[1][e] = fma(a[1][r] * dw, float(acc1[e]) - sum[1][r], result[1][e]);
    }
  }
  }

#pragma clang loop unroll(full)
  for (ushort i = 0; i < 2; ++i)
#pragma clang loop unroll(full)
    for (ushort e = 0; e < 16; ++e) {
      const uint m = row_base + i * 16 + cy + ((e & 7) >> 2) * 8;
      if (m >= rows) continue;
      const uint n = col_base + (e >> 3) * 16 + cx + (e & 3);
      const ulong index = ulong(m) * N + n;
      float value = float(bfloat(result[i][e]));
      if constexpr (Epilogue == T2A8Epilogue::UpSilu) {
        const float gate = float(auxiliary[index]);
        value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
      } else if constexpr (Epilogue == T2A8Epilogue::Residual) {
        value += float(auxiliary[index]);
      }
      output[index] = bfloat(value);
    }
}

#define T2A8_BUFFERS                                                                                         \
  device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],                          \
      device const float *code_sums [[buffer(2)]], device uchar *plane0 [[buffer(3)]],                      \
      device uchar *plane1 [[buffer(4)]], device uchar *meta [[buffer(5)]]
#define T2A8_THREADS                                                                                         \
  uint2 group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],          \
      uint lane [[thread_index_in_simdgroup]]
#define T2A8_SHARED(F)                                                                                       \
  threadgroup uint wtile[T2A8Columns * 32];                                                                  \
  threadgroup float dtile[T2A8Columns];                                                                      \
  threadgroup half2 tl[quant_has_table<F>::value ? kQuantPairTableEntries : 1];
#define T2A8_ENTRIES(F, f)                                                                                   \
  kernel void prefill_linear_t2a8_##f(T2A8_BUFFERS, device bfloat *output [[buffer(6)]],                     \
                                      constant Q4A8Params &params [[buffer(7)]], T2A8_THREADS) {             \
    T2A8_SHARED(F)                                                                                           \
    t2a8_tile<F, T2A8Epilogue::None>(codes, scale, code_sums, plane0, plane1, meta, output, output, params,  \
                                     group, simd_group, lane, wtile, dtile, tl);                                                     \
  }                                                                                                          \
  kernel void prefill_linear_t2a8_residual_##f(T2A8_BUFFERS, device const bfloat *residual [[buffer(6)]],    \
                                               device bfloat *output [[buffer(7)]],                          \
                                               constant Q4A8Params &params [[buffer(8)]], T2A8_THREADS) {    \
    T2A8_SHARED(F)                                                                                           \
    t2a8_tile<F, T2A8Epilogue::Residual>(codes, scale, code_sums, plane0, plane1, meta, residual, output,    \
                                         params, group, simd_group, lane, wtile, dtile, tl);                                         \
  }                                                                                                          \
  kernel void prefill_linear_t2a8_up_silu_##f(T2A8_BUFFERS, device const bfloat *gate [[buffer(6)]],         \
                                              device bfloat *output [[buffer(7)]],                           \
                                              constant Q4A8Params &params [[buffer(8)]], T2A8_THREADS) {     \
    T2A8_SHARED(F)                                                                                           \
    t2a8_tile<F, T2A8Epilogue::UpSilu>(codes, scale, code_sums, plane0, plane1, meta, gate, output, params,  \
                                       group, simd_group, lane, wtile, dtile, tl);                                                   \
  }
T2A8_ENTRIES(FmtPQ20, pq20)
T2A8_ENTRIES(FmtPTQ10, ptq10)
