#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

enum class KernelLayout : uint8_t { Kv4Group6, Kv2Group8 };

bool sameGrid(metal::DispatchSize a, metal::DispatchSize b) noexcept {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// One thread per gate element, across rows x query heads x head dimension.
uint64_t gateGroups(uint64_t rows, uint32_t queryHeads, uint32_t headDimension) {
  const uint64_t elements = rows * queryHeads * headDimension;
  return (elements + metal::CommandGraph::kDefaultThreads - 1) /
         metal::CommandGraph::kDefaultThreads;
}

KernelLayout storageKernelLayout(kv::Q8Layout layout) {
  if (!layout.valid() || layout.headDimension != 256) {
    throw std::invalid_argument("no Q8 store kernel for layout");
  }
  if (layout.kvHeads == 4)
    return KernelLayout::Kv4Group6;
  if (layout.kvHeads == 2)
    return KernelLayout::Kv2Group8;
  throw std::invalid_argument("no Q8 store kernel for layout");
}

KernelLayout attentionKernelLayout(uint32_t queryHeads, kv::Q8Layout layout) {
  // Select a compiled GQA variant so kernels need no geometry branches.
  const KernelLayout result = storageKernelLayout(layout);
  if ((result == KernelLayout::Kv4Group6 && queryHeads == 24) ||
      (result == KernelLayout::Kv2Group8 && queryHeads == 16))
    return result;
  throw std::invalid_argument("no paged-attention kernel for layout");
}

void requireTQ4Layout(kv::TQ4Layout layout) {
  if (!layout.valid() || layout.attentionLayers != 16 ||
      layout.kvHeads != 4 || layout.headDimension != 256)
    throw std::invalid_argument("no TQ4 attention kernel for layout");
}

std::string_view pipeline(KernelLayout layout, std::string_view kv4Group6,
                          std::string_view kv2Group8) noexcept {
  return layout == KernelLayout::Kv4Group6 ? kv4Group6 : kv2Group8;
}

AttentionWorkspace attentionWorkspace(uint64_t rows, uint32_t headDimension) {
  return {rows * headDimension * sizeof(float), rows * 2 * sizeof(float)};
}

// Verify scratch is sized once for the maximum split count of every lane, so
// a lane's history-scaled partition never needs a reallocation.
AttentionWorkspace verifyWorkspaceBound(uint32_t lanes, uint32_t queryHeads,
                                        kv::Q8Layout layout) {
  return attentionWorkspace(uint64_t{lanes} * kv::kQ8VerifyMaximumRows *
                                kv::kQ8VerifyMaximumSplits * queryHeads,
                            layout.headDimension);
}

void requireScalePlacement(AttentionScalePlacement placement) {
  if (placement != AttentionScalePlacement::Softmax &&
      placement != AttentionScalePlacement::Cooperative)
    throw std::invalid_argument("invalid attention scale placement");
}

uint32_t prefillSplits(uint32_t tiles, PrefillAttentionConfig configuration) {
  requireScalePlacement(configuration.scalePlacement);
  switch (configuration.splitMultiplier) {
  case PrefillSplitMultiplier::One:
  case PrefillSplitMultiplier::Two:
    break;
  default:
    throw std::invalid_argument("invalid prefill attention configuration");
  }
  constexpr uint32_t maximum = SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS;
  const uint32_t baseline = std::clamp(maximum / tiles, 1U, maximum);
  return std::min(maximum,
                  baseline * static_cast<uint32_t>(configuration.splitMultiplier));
}

uint32_t verifySplits(VerifyAttentionConfig configuration) {
  requireScalePlacement(configuration.scalePlacement);
  switch (configuration.splitCount) {
  case VerifySplitCount::One:
  case VerifySplitCount::Eight:
  case VerifySplitCount::Sixteen:
  case VerifySplitCount::ThirtyTwo:
    return static_cast<uint32_t>(configuration.splitCount);
  }
  throw std::invalid_argument("invalid verify attention configuration");
}

std::string_view verifySplitPipeline(KernelLayout layout,
                                     VerifyAttentionConfig configuration) noexcept {
  const bool cooperative =
      configuration.scalePlacement == AttentionScalePlacement::Cooperative;
  return pipeline(layout,
      cooperative ? "verify_attention_q8_split_cooperative_scale"
                  : "verify_attention_q8_split",
      cooperative ? "verify_attention_q8_split_cooperative_scale_kv2_g8"
                  : "verify_attention_q8_split_kv2_g8");
}

} // namespace

bool PrefillAttentionPlan::sameExecutionAs(const PrefillAttentionPlan &other) const noexcept {
  return rows == other.rows && historyTokens == other.historyTokens &&
      splits == other.splits &&
      workspace.partialsBytes == other.workspace.partialsBytes &&
      workspace.statisticsBytes == other.workspace.statisticsBytes &&
      splitPipeline == other.splitPipeline && reducePipeline == other.reducePipeline &&
      sameGrid(splitGroups, other.splitGroups) && sameGrid(reduceGroups, other.reduceGroups);
}

bool VerifyAttentionPlan::sameExecutionAs(const VerifyAttentionPlan &other) const noexcept {
  return lanes == other.lanes && laneSplits == other.laneSplits && splits == other.splits &&
      workspace.partialsBytes == other.workspace.partialsBytes &&
      workspace.statisticsBytes == other.workspace.statisticsBytes &&
      splitPipeline == other.splitPipeline && reducePipeline == other.reducePipeline &&
      sameGrid(splitGroups, other.splitGroups) && sameGrid(reduceGroups, other.reduceGroups) &&
      storePipeline_ == other.storePipeline_ && sameGrid(storeGroups_, other.storeGroups_) &&
      sameGrid(storeThreads_, other.storeThreads_);
}

std::span<const PrefillAttentionConfig>
PagedAttention::prefillCandidates() noexcept {
  static constexpr std::array configurations{
      PrefillAttentionConfig{},
      PrefillAttentionConfig{PrefillSplitMultiplier::Two},
      PrefillAttentionConfig{PrefillSplitMultiplier::One, AttentionScalePlacement::Cooperative},
      PrefillAttentionConfig{PrefillSplitMultiplier::Two, AttentionScalePlacement::Cooperative}};
  return configurations;
}

std::span<const VerifyAttentionConfig>
PagedAttention::verifyCandidates() noexcept {
  // Keep the default first, then every split/scale combination.
  static constexpr auto configurations = [] {
    constexpr std::array splits{VerifySplitCount::ThirtyTwo, VerifySplitCount::Sixteen,
                                VerifySplitCount::Eight, VerifySplitCount::One};
    std::array<VerifyAttentionConfig, 8> result{};
    size_t index = 0;
    for (auto placement :
         {AttentionScalePlacement::Softmax, AttentionScalePlacement::Cooperative})
      for (auto count : splits)
        result[index++] = {count, placement};
    return result;
  }();
  return configurations;
}

PrefillAttentionPlan PagedAttention::prefillPlan(
    uint32_t rows, uint32_t queryHeads, kv::Q8Layout layout,
    uint32_t historyTokens, PrefillAttentionConfig configuration) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!rows || rows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  if (uint64_t{historyTokens} + rows > kv::kMaximumPhysicalTokens)
    throw std::invalid_argument("prefill attention history exceeds physical context");
  const uint32_t tiles = kv::prefillAttentionTiles(rows);
  const uint32_t splits = prefillSplits(tiles, configuration);
  const uint32_t fusedRows =
      kv::kQ8PrefillAttentionTileRows * (queryHeads / layout.kvHeads);
  const bool cooperative = configuration.scalePlacement == AttentionScalePlacement::Cooperative;
  return {configuration, rows, historyTokens, splits,
          attentionWorkspace(uint64_t{tiles} * splits * layout.kvHeads * fusedRows,
                             layout.headDimension),
          pipeline(kernel,
                   cooperative ? "prefill_attention_q8_split_cooperative_scale"
                               : "prefill_attention_q8_split",
                   cooperative ? "prefill_attention_q8_split_cooperative_scale_kv2_g8"
                               : "prefill_attention_q8_split_kv2_g8"),
          pipeline(kernel, "prefill_attention_q8_reduce",
                   "prefill_attention_q8_reduce_kv2_g8"),
          {layout.kvHeads, tiles, splits}, {layout.kvHeads, fusedRows, tiles}};
}

VerifyAttentionPlan PagedAttention::verifyPlan(
    uint32_t lanes, uint32_t queryHeads, kv::Q8Layout layout,
    std::span<const uint32_t> historyTokens,
    VerifyAttentionConfig configuration) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  if (historyTokens.size() != lanes &&
      historyTokens.size() != SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid verify attention history vector");
  const uint32_t base = verifySplits(configuration);
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> laneSplits{};
  uint32_t splits = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (uint64_t{historyTokens[lane]} + kv::kQ8VerifyMaximumRows >
        kv::kMaximumPhysicalTokens)
      throw std::invalid_argument(
          "verify attention history exceeds physical context");
    laneSplits[lane] = kv::q8VerifyAttentionSplits(
        base, historyTokens[lane], kv::kQ8VerifyMaximumRows);
    splits = std::max(splits, laneSplits[lane]);
  }
  return {configuration, lanes, laneSplits, splits,
          verifyWorkspaceBound(lanes, queryHeads, layout),
          verifySplitPipeline(kernel, configuration),
          pipeline(kernel, "verify_attention_q8_reduce",
                   "verify_attention_q8_reduce_kv2_g8"),
          {layout.kvHeads, splits, lanes},
          {layout.kvHeads,
           kv::kQ8VerifyMaximumRows * (queryHeads / layout.kvHeads), lanes},
          pipeline(kernel, "verify_attention_q8_store",
                   "verify_attention_q8_store_kv2_g8"),
          {uint64_t{lanes} * 2 * kv::kQ8VerifyMaximumRows * layout.kvHeads, 1, 1},
          {layout.headDimension, 1, 1}};
}

AttentionWorkspace PagedAttention::prefillWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Q8Layout layout,
    PrefillAttentionConfig configuration) {
  (void)attentionKernelLayout(queryHeads, layout);
  if (!maximumRows || maximumRows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  // Allocation-time bound for every shorter sequence. Encoding resolves its
  // own exact plan in constant time; no bound scan occurs on the hot path.
  uint64_t slots = 0;
  for (uint32_t tiles = 1; tiles <= kv::prefillAttentionTiles(maximumRows); ++tiles)
    slots = std::max(slots, uint64_t{tiles} * prefillSplits(tiles, configuration));
  return attentionWorkspace(slots * kv::kQ8PrefillAttentionTileRows * queryHeads,
                             layout.headDimension);
}

AttentionWorkspace PagedAttention::verifyWorkspace(
    uint32_t lanes, uint32_t queryHeads, kv::Q8Layout layout,
    VerifyAttentionConfig configuration) {
  (void)attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  (void)verifySplits(configuration);
  return verifyWorkspaceBound(lanes, queryHeads, layout);
}

void PagedAttention::addPrefillProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t tokens, uint32_t cacheStride,
    uint32_t rowStride, uint32_t queryHeads, kv::Q8Layout layout) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!tokens || !cacheStride || !rowStride)
    throw std::invalid_argument("invalid paged prefill projection geometry");
  const FullPrefillParams params{tokens, cacheStride, rowStride};
  graph.add(std::string(pipeline(kernel, "prefill_attention_qkv",
                                 "prefill_attention_qkv_kv2_g8")),
            {std::move(packed), std::move(queryNorm), std::move(keyNorm),
             std::move(ropeCos), std::move(ropeSin), std::move(queries),
             std::move(chunkKeys), std::move(chunkValues)},
            params,
            {uint64_t{tokens} * (queryHeads + layout.kvHeads), 1, 1});
}

void PagedAttention::addPrefillGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden, uint32_t tokens,
    uint32_t cacheStride, uint32_t rowStride, uint32_t queryHeads,
    kv::Q8Layout layout) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!tokens || !cacheStride || !rowStride)
    throw std::invalid_argument("invalid paged prefill gate geometry");
  const FullPrefillParams params{tokens, cacheStride, rowStride};
  graph.add(std::string(pipeline(kernel, "prefill_attention_gate",
                                 "prefill_attention_gate_kv2_g8")),
            {std::move(packed), std::move(attention), std::move(hidden)}, params,
            {gateGroups(tokens, queryHeads, layout.headDimension), 1, 1});
}

void PagedAttention::addVerifyProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t rowsPerLane,
    uint32_t cacheStride, uint32_t rowStride, uint32_t queryHeads,
    kv::Q8Layout layout, uint32_t lanes) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!rowsPerLane || !cacheStride || !rowStride || !lanes ||
      lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify projection geometry");
  const FullDecodeBatchParams params{rowsPerLane, cacheStride, rowStride,
                                     lanes};
  graph.add(std::string(pipeline(kernel, "verify_attention_qkv",
                                 "verify_attention_qkv_kv2_g8")),
            {std::move(packed), std::move(queryNorm), std::move(keyNorm),
             std::move(ropeCos), std::move(ropeSin), std::move(queries),
             std::move(chunkKeys), std::move(chunkValues)},
            params,
            {uint64_t{rowsPerLane} * (queryHeads + layout.kvHeads), lanes, 1});
}

void PagedAttention::addVerifyGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden,
    uint32_t rowsPerLane, uint32_t cacheStride, uint32_t rowStride,
    uint32_t queryHeads, kv::Q8Layout layout, uint32_t lanes) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!rowsPerLane || !cacheStride || !rowStride || !lanes ||
      lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify gate geometry");
  const FullDecodeBatchParams params{rowsPerLane, cacheStride, rowStride,
                                     lanes};
  graph.add(std::string(pipeline(kernel, "verify_attention_gate",
                                 "verify_attention_gate_kv2_g8")),
            {std::move(packed), std::move(attention), std::move(hidden)}, params,
            {gateGroups(uint64_t{rowsPerLane} * lanes, queryHeads,
                        layout.headDimension),
             1, 1});
}

kv::Q8ChunkedPrefillParams PagedAttention::prefillParams(
    uint64_t logicalPosition, uint32_t chunkTokens, uint32_t chunkStride,
    std::span<const uint32_t> pageTable, uint32_t physicalPageCount) {
  if (logicalPosition > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Q8 logical position exceeds kernel ABI");
  kv::Q8ChunkedPrefillParams params{
      static_cast<uint32_t>(logicalPosition), chunkTokens, chunkStride,
      static_cast<uint32_t>(pageTable.size()), physicalPageCount, 0, 0, 0};
  const std::string_view error = kv::chunkedPrefillValidationError(params);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  if (!kv::chunkedPrefillPageTableInRange(params, pageTable))
    throw std::invalid_argument("invalid Q8 chunk page table");
  return params;
}

void PagedAttention::addPrefillStore(
    metal::CommandGraph &graph, const kv::Q8LayerStorage &layer,
    metal::MetalBuffer chunkKeys, metal::MetalBuffer chunkValues,
    metal::MetalBuffer pageTable,
    const kv::Q8ChunkedPrefillParams &params, kv::Q8Layout layout) {
  const KernelLayout kernel = storageKernelLayout(layout);
  graph.add(std::string(pipeline(kernel, "prefill_attention_q8_store",
                                 "prefill_attention_q8_store_kv2_g8")),
            {std::move(chunkKeys), std::move(chunkValues), layer.keyData,
             layer.keyScales, layer.valueData, layer.valueScales,
             std::move(pageTable)},
            params, {uint64_t{2} * params.chunk_tokens * layout.kvHeads, 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefillStore(
    metal::CommandGraph &graph, const kv::TQ4LayerStorage &layer,
    metal::MetalBuffer chunkKeys, metal::MetalBuffer chunkValues,
    metal::MetalBuffer pageTable, const kv::Q8ChunkedPrefillParams &params,
    kv::TQ4Layout layout) {
  requireTQ4Layout(layout);
  if (layer.layerIndex >= layout.attentionLayers)
    throw std::invalid_argument("TQ4 layer index is outside layout");
  const kv::TQ4StoreParams tq4{params, layer.layerIndex, 0, 0, 0};
  graph.add("prefill_attention_tq4_store",
            {std::move(chunkKeys), std::move(chunkValues), layer.modelPages,
             std::move(pageTable)},
            tq4, {uint64_t{2} * params.chunk_tokens * layout.kvHeads, 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefill(
    metal::CommandGraph &graph, const kv::Q8LayerStorage &layer,
    metal::MetalBuffer queries, metal::MetalBuffer output,
    metal::MetalBuffer partials, metal::MetalBuffer statistics,
    metal::MetalBuffer pageTable, const kv::Q8ChunkedPrefillParams &chunk,
    const PrefillAttentionPlan &plan) {
  if (chunk.chunk_tokens != plan.rows ||
      chunk.committed_tokens != plan.historyTokens)
    throw std::invalid_argument("prefill attention rows or history do not match plan");
  const std::string_view error = kv::chunkedPrefillValidationError(chunk);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  if (partials.sizeBytes() < plan.workspace.partialsBytes ||
      statistics.sizeBytes() < plan.workspace.statisticsBytes) {
    throw std::invalid_argument(
        "prefill attention scratch is smaller than its bound");
  }
  const kv::Q8PrefillAttentionParams params{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, chunk.physical_page_count, plan.splits, 0, 0};
  graph.add(std::string(plan.splitPipeline),
            {std::move(queries), layer.keyData, layer.keyScales, layer.valueData,
             layer.valueScales, partials, statistics, std::move(pageTable)},
            params, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline),
            {std::move(partials), std::move(statistics), std::move(output)},
            params, plan.reduceGroups);
}

void PagedAttention::addStagedPrefill(
    metal::CommandGraph &graph, const kv::TQ4LayerStorage &layer,
    metal::MetalBuffer queries, metal::MetalBuffer output,
    metal::MetalBuffer stagedKeys, metal::MetalBuffer stagedValues,
    metal::MetalBuffer pageTable, const kv::Q8ChunkedPrefillParams &chunk,
    kv::TQ4Layout layout) {
  requireTQ4Layout(layout);
  const std::string_view error = kv::chunkedPrefillValidationError(chunk);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  if (layer.layerIndex >= layout.attentionLayers)
    throw std::invalid_argument("TQ4 staged prefill layer is outside layout");
  const uint32_t visible = chunk.committed_tokens + chunk.chunk_tokens;
  const uint64_t stagedBytes = uint64_t{visible} * layout.headDimension *
                               sizeof(uint16_t);
  if (stagedKeys.sizeBytes() < stagedBytes ||
      stagedValues.sizeBytes() < stagedBytes)
    throw std::invalid_argument("TQ4 staged prefill planes are too small");
  const uint32_t pages = (visible + kv::kPageTokens - 1) / kv::kPageTokens;
  const uint32_t tiles = kv::prefillAttentionTiles(chunk.chunk_tokens);
  for (uint32_t head = 0; head < layout.kvHeads; ++head) {
    const kv::TQ4StagedPrefillParams params{
        chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
        visible, layer.layerIndex, head, 0, 0};
    if (head == 0) {
      graph.add("prefill_attention_tq4_stage_head",
                {layer.modelPages, stagedKeys, stagedValues, pageTable}, params,
                {pages, 1, 1}, {256, 1, 1});
    } else {
      graph.addAfterBarrier(
          "prefill_attention_tq4_stage_head",
          {layer.modelPages, stagedKeys, stagedValues, pageTable}, params,
          {pages, 1, 1}, {256, 1, 1});
    }
    graph.addAfterBarrier("prefill_attention_tq4_staged_head",
                          {queries, stagedKeys, stagedValues, output}, params,
                          {tiles, 1, 1}, {256, 1, 1});
  }
}

void PagedAttention::addPrefill(
    metal::CommandGraph &graph, const kv::TQ4LayerStorage &layer,
    metal::MetalBuffer queries, metal::MetalBuffer output,
    metal::MetalBuffer partials, metal::MetalBuffer statistics,
    metal::MetalBuffer pageTable, const kv::Q8ChunkedPrefillParams &chunk,
    const PrefillAttentionPlan &plan) {
  if (chunk.chunk_tokens != plan.rows ||
      chunk.committed_tokens != plan.historyTokens)
    throw std::invalid_argument("TQ4 prefill rows or history do not match plan");
  const std::string_view error = kv::chunkedPrefillValidationError(chunk);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  if (partials.sizeBytes() < plan.workspace.partialsBytes ||
      statistics.sizeBytes() < plan.workspace.statisticsBytes)
    throw std::invalid_argument("TQ4 prefill scratch is smaller than its bound");
  const kv::TQ4PrefillAttentionParams params{
      {chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
       chunk.page_table_entries, chunk.physical_page_count, plan.splits, 0, 0},
      layer.layerIndex, 0, 0, 0};
  graph.add("prefill_attention_tq4_mpp_split",
            {std::move(queries), layer.modelPages, partials, statistics,
             std::move(pageTable)},
            params, plan.splitGroups, {256, 1, 1});
  graph.add("prefill_attention_tq4_mpp_reduce",
            {std::move(partials), std::move(statistics), std::move(output)},
            params, plan.reduceGroups, {256, 1, 1});
}

void PagedAttention::addVerify(
    metal::CommandGraph &graph, const kv::Q8LayerStorage &layer,
    PagedVerifyBuffers buffers,
    std::span<const kv::Q8ChunkedPrefillParams> storeParams,
    std::span<const kv::Q8VerifyAttentionParams> attentionParams,
    const VerifyAttentionPlan &plan) {
  const uint32_t lanes = plan.lanes;
  constexpr uint32_t maximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  if (storeParams.size() != maximumLanes ||
      attentionParams.size() != maximumLanes ||
      buffers.pageTables.size() != maximumLanes) {
    throw std::invalid_argument("invalid paged verify batch");
  }
  if (buffers.partials.sizeBytes() < plan.workspace.partialsBytes ||
      buffers.statistics.sizeBytes() < plan.workspace.statisticsBytes) {
    throw std::invalid_argument(
        "verify attention scratch is smaller than its bound");
  }
  std::array<kv::Q8ChunkedPrefillParams, maximumLanes> stores{};
  std::array<kv::Q8VerifyAttentionParams, maximumLanes> attention{};
  std::copy(storeParams.begin(), storeParams.end(), stores.begin());
  std::copy(attentionParams.begin(), attentionParams.end(), attention.begin());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto storeError = kv::chunkedPrefillValidationError(stores[lane]);
    const auto attentionError =
        kv::q8VerifyAttentionValidationError(attention[lane]);
    if (!storeError.empty() || !attentionError.empty() ||
        stores[lane].chunk_tokens != kv::kQ8VerifyMaximumRows ||
        stores[lane].committed_tokens != attention[lane].committed_tokens ||
        stores[lane].chunk_stride != attention[lane].chunk_stride ||
        stores[lane].page_table_entries != attention[lane].page_table_entries ||
        stores[lane].physical_page_count !=
            attention[lane].physical_page_count ||
        stores[lane].chunk_stride != stores[0].chunk_stride)
      throw std::invalid_argument("inconsistent paged verify lane parameters");
    // The plan scaled each lane's split count from this same committed
    // history; every lane's partials use the plan-wide slot stride.
    attention[lane].split_count = plan.laneSplits[lane];
    attention[lane].slot_splits = plan.splits;
    if (attention[lane].split_count !=
        kv::q8VerifyAttentionSplits(verifySplits(plan.configuration),
                                    attention[lane].committed_tokens,
                                    kv::kQ8VerifyMaximumRows))
      throw std::invalid_argument("paged verify lane history does not match plan");
  }
  std::vector<metal::MetalBuffer> storeBuffers{
      buffers.chunkKeys, buffers.chunkValues, layer.keyData, layer.keyScales,
      layer.valueData, layer.valueScales};
  storeBuffers.insert(storeBuffers.end(), buffers.pageTables.begin(),
                      buffers.pageTables.end());
  graph.add(std::string(plan.storePipeline_), std::move(storeBuffers), stores,
            plan.storeGroups_, plan.storeThreads_);

  std::vector<metal::MetalBuffer> attentionBuffers{
      buffers.queries, layer.keyData, layer.keyScales, layer.valueData,
      layer.valueScales, buffers.partials, buffers.statistics};
  attentionBuffers.insert(attentionBuffers.end(), buffers.pageTables.begin(),
                          buffers.pageTables.end());
  graph.add(std::string(plan.splitPipeline), std::move(attentionBuffers),
            attention, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline),
            {buffers.partials, buffers.statistics, buffers.output},
            attention, plan.reduceGroups);
}

void PagedAttention::addVerify(
    metal::CommandGraph &graph, const kv::TQ4LayerStorage &layer,
    PagedVerifyBuffers buffers,
    std::span<const kv::Q8ChunkedPrefillParams> storeParams,
    std::span<const kv::Q8VerifyAttentionParams> attentionParams,
    const VerifyAttentionPlan &plan) {
  const uint32_t lanes = plan.lanes;
  constexpr uint32_t maximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  if (storeParams.size() != maximumLanes ||
      attentionParams.size() != maximumLanes ||
      buffers.pageTables.size() != maximumLanes)
    throw std::invalid_argument("invalid TQ4 paged verify batch");
  if (buffers.partials.sizeBytes() < plan.workspace.partialsBytes ||
      buffers.statistics.sizeBytes() < plan.workspace.statisticsBytes)
    throw std::invalid_argument("TQ4 verify scratch is smaller than its bound");
  std::array<kv::TQ4StoreParams, maximumLanes> stores{};
  std::array<kv::TQ4VerifyAttentionParams, maximumLanes> attention{};
  for (uint32_t lane = 0; lane < maximumLanes; ++lane) {
    stores[lane] = {storeParams[lane], layer.layerIndex, 0, 0, 0};
    kv::Q8VerifyAttentionParams resolved = attentionParams[lane];
    if (lane < lanes) {
      const auto storeError = kv::chunkedPrefillValidationError(storeParams[lane]);
      const auto attentionError =
          kv::q8VerifyAttentionValidationError(attentionParams[lane]);
      if (!storeError.empty() || !attentionError.empty() ||
          storeParams[lane].chunk_tokens != kv::kQ8VerifyMaximumRows ||
          storeParams[lane].committed_tokens !=
              attentionParams[lane].committed_tokens ||
          storeParams[lane].chunk_stride !=
              attentionParams[lane].chunk_stride ||
          storeParams[lane].page_table_entries !=
              attentionParams[lane].page_table_entries ||
          storeParams[lane].physical_page_count !=
              attentionParams[lane].physical_page_count ||
          storeParams[lane].chunk_stride != storeParams[0].chunk_stride)
        throw std::invalid_argument("inconsistent TQ4 verify lane parameters");
      resolved.split_count = plan.laneSplits[lane];
      resolved.slot_splits = plan.splits;
    }
    attention[lane] = {resolved, layer.layerIndex, 0, 0, 0};
  }
  std::vector<metal::MetalBuffer> storeBuffers{
      buffers.chunkKeys, buffers.chunkValues, layer.modelPages};
  storeBuffers.insert(storeBuffers.end(), buffers.pageTables.begin(),
                      buffers.pageTables.end());
  graph.add("verify_attention_tq4_store", std::move(storeBuffers), stores,
            {uint64_t{lanes} * 2 * kv::kQ8VerifyMaximumRows * 4, 1, 1},
            {256, 1, 1});
  std::vector<metal::MetalBuffer> splitBuffers{
      buffers.queries, layer.modelPages, buffers.partials, buffers.statistics};
  splitBuffers.insert(splitBuffers.end(), buffers.pageTables.begin(),
                      buffers.pageTables.end());
  graph.add("verify_attention_tq4_mpp_split", std::move(splitBuffers), attention,
            plan.splitGroups, {256, 1, 1});
  graph.add("verify_attention_tq4_mpp_reduce",
            {buffers.partials, buffers.statistics, buffers.output}, attention,
            plan.reduceGroups, {256, 1, 1});
}

} // namespace splash::ops
