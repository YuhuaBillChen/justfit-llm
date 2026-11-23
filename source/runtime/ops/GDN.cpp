// Modified for JustFit: product identity and integration.
#include "ops/VerifyShape.hpp"
#include "ops/GDN.hpp"
#include "model/StateLayout.hpp"

#include <cstring>

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace justfit::ops {
namespace {
using model::compactGdnEnabled;
metal::MetalBackend *compactBackend = nullptr;
metal::MetalBuffer compactInput, compactOutput, compactSeed;

std::string compactVerifyName(const std::string &dense, bool tree) {
  constexpr std::string_view prefix = "verify_gdn_fused";
  if (!dense.starts_with(prefix))
    throw std::invalid_argument("unsupported compact GDN projection");
  const std::string suffix = dense.substr(prefix.size());
  if (tree)
    return (activeVerifyRows() == 16 ? "compact_gdn_tree16_verify" : "compact_gdn_tree_verify") + suffix;
  if (activeVerifyRows() == 16) return "compact_gdn_tree16_chain_verify" + suffix;
  return "compact_gdn_fused" + suffix;
}

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 8);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

uint64_t valueWidth(const GdnShape &shape) { return uint64_t{shape.valueHeads} * shape.headDimension; }
// One layer's carried convolution rows and fp32 recurrent state, a head
// dimension square per value head.
uint64_t carriedBytes(const GdnShape &shape) {
  return uint64_t{JUSTFIT_GDN_CONVOLUTION_TAPS - 1} * shape.convolutionDimension * 2;
}
uint64_t recurrentBytes(const GdnShape &shape) {
  if (compactGdnEnabled())
    return uint64_t{shape.valueHeads} * JUSTFIT_COMPACT_HEAD_BYTES + uint64_t{shape.valueHeads / 3} * 4096u;
  return valueWidth(shape) * shape.headDimension * sizeof(float);
}

// The bytes of `rows` rows of the packed projection the kernels read: each
// row's q, k and v convolution inputs, z, then beta and alpha per value head.
uint64_t packedBytes(const GdnShape &shape, uint64_t rows) {
  return rowBytes(rows, shape.packedWidth, shape.convolutionDimension + valueWidth(shape) + 2 * shape.valueHeads, 2);
}

// The convolution weights of every channel and a_scale and dt_bias of every
// value head.
void requireMixerWeights(const GdnShape &shape, const metal::MetalBuffer &convolution,
                         const metal::MetalBuffer &decay, const metal::MetalBuffer &timeBias) {
  requireBytes(convolution, uint64_t{shape.convolutionDimension} * JUSTFIT_GDN_CONVOLUTION_TAPS * 2,
               "GDN convolution weight");
  requireBytes(decay, uint64_t{shape.valueHeads} * sizeof(float), "GDN decay weight");
  requireBytes(timeBias, uint64_t{shape.valueHeads} * 2, "GDN time bias");
}

// The fp32 decay and bf16 beta gates of `rows` rows of value heads.
void requireGates(const GdnShape &shape, uint64_t rows, const metal::MetalBuffer &decay,
                  const metal::MetalBuffer &beta) {
  requireBytes(decay, rows * shape.valueHeads * sizeof(float), "GDN decay");
  requireBytes(beta, rows * shape.valueHeads * 2, "GDN beta");
}

// Each running lane's current and next state cells reach the end of layer
// `layer`'s state: its carried rows from layer x convolutionLayerBytes, its
// recurrent state from convolutionStateBytes + layer x recurrentLayerBytes.
void requireStates(const GdnShape &shape, GdnStateStrides strides, uint32_t layer, uint32_t lanes,
                   std::span<const metal::MetalBuffer> current, std::span<const metal::MetalBuffer> next) {
  const uint64_t bytes =
      std::max(uint64_t{layer} * strides.convolutionLayerBytes + carriedBytes(shape),
               strides.convolutionStateBytes + uint64_t{layer} * strides.recurrentLayerBytes + recurrentBytes(shape));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(current[lane], bytes, "GDN current state");
    requireBytes(next[lane], uint64_t{layer}*strides.convolutionLayerBytes+carriedBytes(shape), "GDN commit convolution");
  }
}

} // namespace

void GDN::setCompactResources(metal::MetalBackend &backend, metal::MetalBuffer input,
                              metal::MetalBuffer output, metal::MetalBuffer seed) {
  compactBackend = &backend;
  compactInput = std::move(input);
  compactOutput = std::move(output);
  compactSeed = std::move(seed);
}

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens, GdnHeadOrder order) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const std::string gate = normKernel(kernelName(kernel, "prefill_gdn_gate", "prefill_gdn_gate_vh32"),
                                      buffers.mixerNorm, shape.headDimension);
  const uint64_t keyRows = uint64_t{tokens} * shape.keyHeads * shape.headDimension * 2;
  const uint64_t valueRows = tokens * valueWidth(shape) * 2;
  requireBytes(buffers.packed, packedBytes(shape, tokens), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.convolutionIn, carriedBytes(shape), "GDN convolution state");
  requireBytes(buffers.convolutionOut, carriedBytes(shape), "GDN next convolution state");
  requireBytes(buffers.queries, keyRows, "GDN query");
  requireBytes(buffers.keys, keyRows, "GDN key");
  requireBytes(buffers.values, valueRows, "GDN value");
  requireGates(shape, tokens, buffers.decay, buffers.beta);
  requireBytes(buffers.recurrentIn, recurrentBytes(shape), "GDN recurrent state");
  requireBytes(buffers.recurrentOut, recurrentBytes(shape), "GDN next recurrent state");
  requireBytes(buffers.recurrentRows, valueRows, "GDN recurrent row");
  requireBytes(buffers.hidden, valueRows, "GDN hidden");
  const auto residentOutput = buffers.recurrentOut;
  if (compactGdnEnabled()) {
    graph.add("compact_unfold_layer", {buffers.recurrentIn, compactInput},
              CompactLayerParams{shape.valueHeads, 0}, {shape.valueHeads, 1, 1}, {128, 1, 1});
    buffers.recurrentIn = compactInput;
    buffers.recurrentOut = compactOutput;
  }
  const GDNPrefillParams params{tokens};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            params, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            params,
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 JUSTFIT_GDN_SCAN_STATE_ROWS,
             1, 1},
            {JUSTFIT_GDN_SCAN_THREADS, 1, 1});
  graph.add(gate,
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm.buffer,
             buffers.hidden},
            GDNGatePrefillParams{order == GdnHeadOrder::Tiled},
            {uint64_t{tokens} * shape.valueHeads, 1, 1}, {128, 1, 1});
  if (compactGdnEnabled())
    graph.add("compact_pack_layer", {compactOutput, compactSeed, residentOutput},
              CompactLayerParams{shape.valueHeads, 0}, {shape.valueHeads, 1, 1}, {128, 1, 1});
  const uint32_t words=static_cast<uint32_t>(carriedBytes(shape)/4);
  graph.add("gdn_prefill_carry_copy",{buffers.convolutionOut,buffers.convolutionIn},words,{(words+255u)/256u,1,1},{256,1,1});
}

PreparedInput GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                             GdnShape shape, uint32_t lanes, uint32_t layer,
                             GdnStateStrides state, GdnHeadOrder order, LinearInput input) {
  if (!lanes || lanes > JUSTFIT_MAXIMUM_BATCH_WIDTH || !state.valid())
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  // Lane l's rows of the packed, mixed, gate and hidden rows are rows
  // [8 l, 8 l + 8).
  const uint64_t rows = uint64_t{lanes} * activeVerifyRows();
  requireBytes(buffers.packed, packedBytes(shape, rows), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.hidden, rows * valueWidth(shape) * 2, "GDN hidden");
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.valueHeads * shape.headDimension,
                        lanes * activeVerifyRows());
  bindings.reserve(prepare ? 19 : 17);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  requireStates(shape, state, layer, lanes, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.mixerNorm.buffer,
                   buffers.hidden});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  const GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                                    layer,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  const std::string name = std::string("verify_gdn_fused") + tableSuffix(input) + kernelName(kernel, "", "_vh32");
  std::string selected = normKernel(name, buffers.mixerNorm, shape.headDimension);
  if (compactGdnEnabled()) selected = compactVerifyName(selected, false);
  graph.add(std::move(selected), std::move(bindings), params,
            {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

void GDN::addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                    GdnShape shape, uint32_t layers, uint32_t lanes,
                    GdnStateStrides state, bool tree) {
  if (!layers || !lanes || lanes > JUSTFIT_MAXIMUM_BATCH_WIDTH ||
      !state.valid())
    throw std::invalid_argument("invalid GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  // The decoded rows of lane l in layer y start at row (y x the maximum batch
  // width + l) x 8 of the packed, mixed and gate rows. A lane replays at most
  // the 7 rows it retains (all 8 leave the decoded state) and carries its
  // last three retained inputs, within those rows.
  const uint64_t rows =
      uint64_t{layers - 1} * 32 + uint64_t{lanes - 1} * activeVerifyRows() +
      activeVerifyRows() - 1;
  requireBytes(buffers.packed, rowBytes(rows, shape.packedWidth, shape.convolutionDimension, 2), "GDN packed");
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.retainedCounts, uint64_t{lanes} * sizeof(uint32_t), "GDN retained counts");
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  requireStates(shape, state, layers - 1, lanes, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  const GDNBatchCommitParams params{state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  if (tree && shape.valueHeads != 48)
    throw std::invalid_argument("tree GDN commit supports 48 value heads");
  const bool compact = compactGdnEnabled();
  const bool wide = activeVerifyRows() == 16;
  const bool wideChain = compact && wide && !tree;
  std::vector<metal::MetalBuffer> continuation;
  if (wideChain) continuation = bindings;
  if (compact && tree && !wide) {
    requireBytes(buffers.acceptedPath, uint64_t{lanes} * JUSTFIT_TREE_MAXIMUM_DEPTH * sizeof(uint32_t),
                 "compact GDN accepted path");
    bindings.push_back(buffers.acceptedPath);
  }
  const std::string selected = compact
      ? (tree ? (wide ? "compact_gdn_tree16_commit" : "compact_gdn_tree_commit")
              : (wide ? "compact_gdn_tree16_chain_commit0" : "compact_gdn_commit"))
      : (tree ? "verify_gdn_commit_tree"
              : kernelName(kernel, "verify_gdn_commit", "verify_gdn_commit_vh32"));
  graph.add(selected, std::move(bindings), params, {shape.valueHeads, layers, lanes});
  std::vector<metal::MetalBuffer> carryBindings;
  appendLaneBindings(carryBindings,buffers.currentStates,buffers.nextStates);
  graph.add(compact ? "compact_finish_commit" : "gdn_finish_commit",
            std::move(carryBindings), params, {(state.convolutionStateBytes/4+255)/256,lanes,1}, {256,1,1});
  if (!compact) return;
  constexpr uint32_t layersPerFlush = 4;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (uint32_t layer = 0; layer < layers; layer += layersPerFlush) {
      const uint32_t chunk = std::min(layersPerFlush, layers - layer);
      auto cell = compactBackend->view(buffers.currentStates[lane],
          state.convolutionStateBytes + uint64_t{layer} * state.recurrentLayerBytes,
          uint64_t{chunk} * state.recurrentLayerBytes);
      bool needsRefresh = false;
      for (uint32_t index = 0; index < chunk; ++index) {
        uint32_t count = 0;
        std::memcpy(&count, static_cast<const std::byte *>(cell.contents()) +
                    uint64_t{index} * state.recurrentLayerBytes + JUSTFIT_COMPACT_COUNT, sizeof(count));
        needsRefresh |= count + activeVerifyRows() > 8;
      }
      if (!needsRefresh) continue;
      const uint32_t heads = shape.valueHeads * chunk;
      for (const char *name : {"lq_flush_prepare_early", "lq_flush_all_hoisted"})
        graph.add(name, {cell, compactOutput, cell, compactSeed}, CompactLayerParams{heads, 1},
                  {heads, 1, 1}, {128, 1, 1});
    }
  }
  if (wideChain)
    graph.add("compact_gdn_tree16_chain_commit1", std::move(continuation), params,
              {shape.valueHeads, layers, lanes});
}

bool GDN::hasDecodeTree(GdnShape shape, LinearInput input, bool normFloat32) noexcept {
  const std::string name = std::string("verify_gdn_fused") + tableSuffix(input) +
                           kernelName(kernelShape(shape), "", "_vh32") + (normFloat32 ? "_f32" : "");
  return name == "verify_gdn_fused" || name == "verify_gdn_fused_table64" ||
         name == "verify_gdn_fused_f32" || name == "verify_gdn_fused_table16_f32";
}

PreparedInput GDN::addDecodeTree(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                                 GdnShape shape, uint32_t lanes, uint32_t layer,
                                 GdnStateStrides state, GdnHeadOrder order,
                                 LinearInput input, const JustFitTreeBatch &tree) {
  // addDecode's checks and bindings, then the tree entry with the trees.
  metal::CommandGraph probe;
  addDecode(probe, buffers, shape, lanes, layer, state, order, input);
  const auto &dispatch = probe.dispatches().front();
  std::string name = dispatch.pipelineName;
  if (name.starts_with("wide16::")) name = name.substr(8);  // push() adds it back
  if (compactGdnEnabled()) {
    const std::string prefix = activeVerifyRows() == 16
        ? "compact_gdn_tree16_chain_verify" : "compact_gdn_fused";
    name = compactVerifyName("verify_gdn_fused" + name.substr(prefix.size()), true);
  } else if (name == "verify_gdn_fused_table64") name = "verify_gdn_fused_tree_table64";
  else if (name == "verify_gdn_fused") name = "verify_gdn_fused_tree";
  else if (name == "verify_gdn_fused_table16_f32") name = "verify_gdn_fused_tree_table16_f32";
  else if (name == "verify_gdn_fused_f32") name = "verify_gdn_fused_tree_f32";
  else throw std::invalid_argument("no tree GDN verify for " + name);
  if (lanes > 2) throw std::invalid_argument("tree GDN verify takes at most two lanes");
  GDNTreeDecodeParams params{};
  std::memcpy(&params.base, dispatch.bytes.front().data, sizeof(params.base));
  params.tree = tree;
  std::vector<metal::MetalBuffer> bindings;
  for (const auto &binding : dispatch.buffers) bindings.push_back(binding.buffer);
  graph.add(std::move(name), std::move(bindings), params, dispatch.threadgroups,
            dispatch.threadsPerThreadgroup);
  if (input == LinearInput::Plain) return {};
  return {buffers.hidden, input};
}

} // namespace justfit::ops
