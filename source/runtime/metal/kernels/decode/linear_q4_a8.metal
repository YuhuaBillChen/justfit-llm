#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/abi/Linear.h"

// W4A8 target verification projection for Apple10 tensor units: int8 activations times
// the packed Q4 codes, read in place, int8 x int8 -> int32.
//
// Structure follows oMLX v0.7.0 qwen35_oq_a8_nax.metal (Apache-2.0): 16x32x16
// ops with cooperative-tensor operands; unsigned codes go into the int8
// fragment unchanged; inside each GS64 group a lane owns one 16-code run and
// feeds it in the "step-transposed" order 8*(t>>1) + 2j + (t&1), so its
// weights cost one uint2 load per row and group. The quantizer stores each
// activation run in the same order. The four K=16 steps of a group accumulate
// in int32 before the affine correction
//   y += a[m,g] * (s[g,n] * P + b[g,n] * R[m,g]),  R = sum of the row's codes,
// with a per-(row, group) activation scale a.
//
// Codes are [N/256][G][256][8 words], scales/biases [N/256][G][256] (bf16).
// Activations: codes [rows][K] (int8, permuted per run), scale and code sums
// [G][rows] (float). A threadgroup is four simdgroups covering 16 or 32 rows x 128
// columns; each simdgroup computes 32 x 32 as two 16-row accumulators.

constant constexpr ushort A8Columns = 128;

enum class A8Epilogue { None, Residual, UpSilu };

template <A8Epilogue Epilogue, ushort Halves>
inline void q4a8_tile(device const char *codes, device const float *scale,
                      device const float *code_sums, device const uint *weights,
                      device const bfloat *scales, device const bfloat *biases,
                      device const bfloat *auxiliary, device bfloat *output,
                      constant Q4A8Params &params, uint2 group, uint simd_group) {
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
  const uint groups = K / 64;
  // Fragment origin of this lane: column (K) 0/4/8/12, row 0..7.
  const auto origin = left.get_multidimensional_index(ushort(0));
  const uint cx = origin[0], cy = origin[1], run = cx >> 2;
  const uint row_base = group.y * (Halves * 16);
  const uint col_base = group.x * A8Columns + simd_group * 32;
  const uint m_base = row_base + cy;

  float result[Halves][16];
#pragma clang loop unroll(full)
  for (ushort i = 0; i < Halves; ++i)
#pragma clang loop unroll(full)
    for (ushort e = 0; e < 16; ++e) result[i][e] = 0.0f;

  // Weight rows of this lane: n_lane + 8*(q&1) + 16*(q>>1); one 256-column
  // storage tile holds all of them (col_base % 32 == 0).
  const uint n_lane = col_base + cy;
  device const uint *lane_weights =
      weights + (ulong(n_lane / 256) * groups * 256 + n_lane % 256) * 8;
  const uint n_run = col_base + cx;
  const ulong parameter_base = ulong(n_run / 256) * groups * 256 + n_run % 256;

  for (uint g = 0; g < groups; ++g) {
    uint2 packed[4];
#pragma clang loop unroll(full)
    for (ushort q = 0; q < 4; ++q) {
      device const uint *row = lane_weights +
          (ulong((q & 1) * 8 + (q >> 1) * 16) + ulong(g) * 256) * 8;
      packed[q] = reinterpret_cast<device const uint2 *>(row)[run];
    }
#pragma clang loop unroll(full)
    for (ushort t = 0; t < 4; ++t) {
#pragma clang loop unroll(full)
      for (ushort q = 0; q < 4; ++q) {
        const ushort base = (q >> 1) * 8 + (q & 1) * 4;
        const uint word = packed[q][t >> 1];
        const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
        right[base + 0] = quad.x; right[base + 1] = quad.y;
        right[base + 2] = quad.z; right[base + 3] = quad.w;
      }
#pragma clang loop unroll(full)
      for (ushort half_rows = 0; half_rows < Halves; ++half_rows) {
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
    const ulong parameter = parameter_base + ulong(g) * 256;
    vec<bfloat, 4> s[2], b[2];
#pragma clang loop unroll(full)
    for (ushort h = 0; h < 2; ++h) {
      s[h] = vec<bfloat, 4>(*reinterpret_cast<device const packed_bfloat4 *>(scales + parameter + h * 16));
      b[h] = vec<bfloat, 4>(*reinterpret_cast<device const packed_bfloat4 *>(biases + parameter + h * 16));
    }
    float a[2][2], sum[2][2];
#pragma clang loop unroll(full)
    for (ushort i = 0; i < Halves; ++i)
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
      const float sw = float(s[e >> 3][e & 3]), bw = float(b[e >> 3][e & 3]);
      result[0][e] = fma(a[0][r], fma(sw, float(acc0[e]), bw * sum[0][r]), result[0][e]);
      if constexpr (Halves == 2) result[1][e] = fma(a[1][r], fma(sw, float(acc1[e]), bw * sum[1][r]), result[1][e]);
    }
  }

  // BF16 boundary before residual / gate-up epilogues.
#pragma clang loop unroll(full)
  for (ushort i = 0; i < Halves; ++i)
#pragma clang loop unroll(full)
    for (ushort e = 0; e < 16; ++e) {
      const uint m = row_base + i * 16 + cy + ((e & 7) >> 2) * 8;
      if (m >= rows) continue;
      const uint n = col_base + (e >> 3) * 16 + cx + (e & 3);
      const ulong index = ulong(m) * N + n;
      float value = float(bfloat(result[i][e]));
      if constexpr (Epilogue == A8Epilogue::UpSilu) {
        const float gate = float(auxiliary[index]);
        value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
      } else if constexpr (Epilogue == A8Epilogue::Residual) {
        value += float(auxiliary[index]);
      }
      output[index] = bfloat(value);
    }
}

kernel void verify_linear_q4a8_m16(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device bfloat *output [[buffer(6)]], constant Q4A8Params &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::None, 1>(codes, scale, code_sums, weights, scales, biases,
                              output, output, params, group, simd_group);
}

kernel void verify_linear_q4a8_residual_m16(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device const bfloat *residual [[buffer(6)]], device bfloat *output [[buffer(7)]],
    constant Q4A8Params &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::Residual, 1>(codes, scale, code_sums, weights, scales, biases,
                                  residual, output, params, group, simd_group);
}

kernel void verify_linear_q4a8_up_silu_m16(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device const bfloat *gate [[buffer(6)]], device bfloat *output [[buffer(7)]],
    constant Q4A8Params &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::UpSilu, 1>(codes, scale, code_sums, weights, scales, biases,
                                gate, output, params, group, simd_group);
}

kernel void verify_linear_q4a8_m32(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device bfloat *output [[buffer(6)]], constant Q4A8Params &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::None, 2>(codes, scale, code_sums, weights, scales, biases,
                              output, output, params, group, simd_group);
}

kernel void verify_linear_q4a8_residual_m32(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device const bfloat *residual [[buffer(6)]], device bfloat *output [[buffer(7)]],
    constant Q4A8Params &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::Residual, 2>(codes, scale, code_sums, weights, scales, biases,
                                  residual, output, params, group, simd_group);
}

kernel void verify_linear_q4a8_up_silu_m32(
    device const char *codes [[buffer(0)]], device const float *scale [[buffer(1)]],
    device const float *code_sums [[buffer(2)]], device const uint *weights [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]], device const bfloat *biases [[buffer(5)]],
    device const bfloat *gate [[buffer(6)]], device bfloat *output [[buffer(7)]],
    constant Q4A8Params &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  q4a8_tile<A8Epilogue::UpSilu, 2>(codes, scale, code_sums, weights, scales, biases,
                                gate, output, params, group, simd_group);
}

// One simdgroup per (row, GS64 group): symmetric int8 with scale max|x|/127,
// codes stored in the GEMM's per-run order, and the integer code sum.
kernel void verify_linear_q4a8_quantize(
    device const bfloat *input [[buffer(0)]], device char *codes [[buffer(1)]],
    device float *scale [[buffer(2)]], device float *code_sums [[buffer(3)]],
    constant Q4A8Params &params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint K = params.input_size, groups = K / 64;
  const uint task = tile * 8 + simd_group;
  const uint row = task / groups, g = task % groups;
  const ulong origin = ulong(row) * K + g * 64;
  const float x0 = float(input[origin + lane]);
  const float x1 = float(input[origin + lane + 32]);
  const float maximum = simd_max(max(fabs(x0), fabs(x1)));
  const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
  const int q0 = int(rint(x0 * inverse)), q1 = int(rint(x1 * inverse));
  // Offset o inside a 16-code run is fed at step t = 2*(o>>3) + (o&1),
  // position j = (o>>1)&3, i.e. stored at 4t + j.
  const auto stored = [](uint o) {
    return (o & ~15u) + 4 * ((((o & 15) >> 3) << 1) | (o & 1)) + ((o >> 1) & 3);
  };
  codes[origin + stored(lane)] = char(q0);
  codes[origin + stored(lane + 32)] = char(q1);
  const int total = simd_sum(q0 + q1);
  if (lane == 0) {
    scale[ulong(g) * params.rows + row] = maximum / 127.0f;
    code_sums[ulong(g) * params.rows + row] = float(total);
  }
}

