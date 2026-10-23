// Modified for JustFit: product identity and integration.
#pragma once

#include <cstddef>
#include <cstdint>

namespace justfit::ops {

// Research path (DASH_HETERO_PREFILL=1): the CPU's SME units compute the
// trailing columns [firstColumn, outputSize) of a Q4 prefill projection
// while the GPU computes the rest. Same storage as the GPU kernels: bf16
// input [rows][K], codes [N/256][G][256][32 bytes], bf16 scales/biases
// [N/256][G][256], bf16 output [rows][N]. Weights are dequantized to bf16
// and accumulated in fp32; the epilogue rounds like the GPU kernels.
enum class HostEpilogue : uint32_t { None, Residual, UpSilu };

struct HostLinearTask final {
  const uint16_t *input = nullptr;
  // GPU-produced per64 input sums in [rowTile][group][32] order.
  const float *inputSums = nullptr;
  const uint8_t *weights = nullptr;
  const uint16_t *scales = nullptr;
  const uint16_t *biases = nullptr;
  uint16_t *output = nullptr;
  // Residual (Residual) or gate (UpSilu), [rows][N].
  const uint16_t *auxiliary = nullptr;
  // UpSilu: per-64-column output sums in the GPU layout
  // [rows/32][N/64][32] for the following projection.
  float *outputSums = nullptr;
  // Dead arena storage for packed activations; bounds the row chunk.
  uint16_t *scratch = nullptr;
  uint64_t scratchBytes = 0;
  uint32_t rows = 0;  // multiple of 32
  uint32_t inputSize = 0;
  uint32_t weightInputSize = 0; // physical K stride for a leading-input view
  uint32_t outputSize = 0;
  uint32_t firstColumn = 0;  // multiple of 64
  HostEpilogue epilogue = HostEpilogue::None;
};

// Blocks until the slice is complete. Uses a process-wide worker pool
// (DASH_HETERO_THREADS, default 8).
void runHostLinear(const HostLinearTask &task);

} // namespace justfit::ops
