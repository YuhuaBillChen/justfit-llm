// Modified for JustFit: product identity and integration.
#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <utility>
#include <stdexcept>

namespace justfit::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  requireBytes(weights.buffer, weights.bytes(width), "norm weight");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  if (layout != LinearInput::Plain) {
    requireTableScratch(scratch, layout, width, rows);
    graph.add(normKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= JUSTFIT_STAGED_NORM_ROWS && width <= JUSTFIT_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(normKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {JUSTFIT_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    const NormWeights &weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  if (weight.float32 || !rows || !width || width % 64)
    throw std::invalid_argument("the Q4-sum norm takes bf16 weights and whole 64-input groups");
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  // The sums are [32-row tile][64-input group][row of the tile]: the last
  // row's sum of the last group ends them.
  const uint64_t groups = width / 64, last = rows - 1;
  requireBytes(sums, ((last / 32 * groups + groups - 1) * 32 + last % 32 + 1) * sizeof(float), "norm sums");
  graph.add(normKernel("prefill_norm_rms_sums32", weight, width),
            {std::move(input), weight.buffer, std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace justfit::ops
