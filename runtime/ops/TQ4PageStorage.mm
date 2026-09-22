#include "ops/TQ4PageStorage.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::kv {

static_assert(kSparseMappingAlignmentBytes ==
                  metal::MetalBackend::kPlacementSparsePageBytes,
              "KV sparse mapping alignment must equal the Metal placement page");

namespace {

uint64_t checkedMultiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    throw std::overflow_error("TQ4 page storage size overflow");
  return left * right;
}

} // namespace

TQ4PageStorage::TQ4PageStorage(metal::MetalBackend &backend,
                               metal::AllocationAdmission admitAllocation,
                               TQ4Layout layout, uint32_t pageCount)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), pageCount_(pageCount), layers_(layout.attentionLayers) {
  if (!admitAllocation_)
    throw std::invalid_argument(
        "TQ4 page storage requires physical allocation admission");
  if (!layout_.valid())
    throw std::invalid_argument("TQ4 page storage layout is invalid");
  if (!pageCount || pageCount % sparseMappingBatchPages())
    throw std::invalid_argument("TQ4 page pool is not sparse-mapping aligned");

  const StorageByteCounts bytes = layout_.storageByteCounts(pageCount);
  modelPages_ = backend_.allocatePlacementSparseBuffer(
      bytes.total, kSparseMappingAlignmentBytes, "tq4-kv-model-pages");
  for (uint32_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex) {
    layers_[layerIndex] = TQ4LayerStorage{modelPages_, layerIndex};
  }
  if (declaredBytes() != bytes.total)
    throw std::logic_error("TQ4 storage accounting mismatch");

  for (uint32_t first = 0; first < pageCount_; first += backingExtentPages()) {
    const uint32_t count =
        std::min(backingExtentPages(), pageCount_ - first);
    if (count % sparseMappingBatchPages())
      throw std::logic_error("TQ4 backing extent is not tile aligned");
    extents_.push_back(Extent{first, count, std::nullopt});
  }
  if (auto result = ensureResident(0); !result) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate initial TQ4 backing extent: ") +
            metal::allocationFailureName(result.failure),
        result.failure);
  }
}

TQ4PageStorage::~TQ4PageStorage() {
  for (Extent &extent : extents_) {
    if (!backend_.healthy())
      break;
    if (!extent.heap)
      continue;
    try {
      backend_.unmapSparse(mappingsFor(extent), std::move(*extent.heap));
    } catch (...) {
    }
    extent.heap.reset();
  }
  if (!backend_.healthy())
    return;
  try {
    backend_.drainSparseUnmaps();
  } catch (...) {
  }
}

uint64_t TQ4PageStorage::declaredBytes() const noexcept {
  return uint64_t{pageCount_} * layout_.bytesPerModelPage();
}

uint64_t TQ4PageStorage::actualAllocatedBytes() const noexcept {
  return residentBackingBytes_;
}

uint32_t TQ4PageStorage::residentPages() const noexcept {
  return residentPages_;
}

size_t TQ4PageStorage::extentIndex(uint32_t page) const {
  if (page >= pageCount_)
    throw std::out_of_range("invalid TQ4 page id");
  return page / backingExtentPages();
}

uint32_t TQ4PageStorage::extentFirstPage(uint32_t page) const {
  return extents_.at(extentIndex(page)).firstPage;
}

uint32_t TQ4PageStorage::extentPageCount(uint32_t page) const {
  return extents_.at(extentIndex(page)).pageCount;
}

bool TQ4PageStorage::isResident(uint32_t page) const {
  return extents_.at(extentIndex(page)).heap.has_value();
}

std::vector<metal::SparseMapping>
TQ4PageStorage::mappingsFor(const Extent &extent) const {
  std::vector<metal::SparseMapping> mappings;
  const uint64_t expected =
      checkedMultiply(extent.pageCount, layout_.bytesPerModelPage());
  mappings.push_back(
      {modelPages_,
       checkedMultiply(extent.firstPage, layout_.bytesPerModelPage()), expected,
       0});
  return mappings;
}

metal::AllocationResult TQ4PageStorage::ensureResident(uint32_t page) {
  Extent &extent = extents_.at(extentIndex(page));
  if (extent.heap)
    return true;
  const uint64_t bytes =
      checkedMultiply(extent.pageCount, layout_.bytesPerModelPage());
  try {
    return admitAllocation_(bytes, [&] {
      std::optional<metal::SparseHeap> heap;
      heap.emplace(backend_.allocatePlacementHeap(
          bytes, kSparseMappingAlignmentBytes,
          "tq4-kv-extent-" + std::to_string(extent.firstPage)));
      if (heap->sizeBytes() != bytes)
        throw std::logic_error(
            "placement heap size differs from admitted TQ4 extent bytes");
      backend_.mapSparse(*heap, mappingsFor(extent));
      residentBackingBytes_ += heap->sizeBytes();
      residentPages_ += extent.pageCount;
      extent.heap = std::move(heap);
    });
  } catch (const metal::MetalAllocationError &error) {
    return error.failure();
  }
}

bool TQ4PageStorage::releaseBackingForPage(uint32_t page) {
  Extent &extent = extents_.at(extentIndex(page));
  if (!extent.heap)
    return false;
  const uint64_t heapBytes = extent.heap->sizeBytes();
  if (residentBackingBytes_ < heapBytes || residentPages_ < extent.pageCount)
    throw std::logic_error("TQ4 resident accounting underflowed");
  backend_.unmapSparse(mappingsFor(extent), std::move(*extent.heap));
  residentBackingBytes_ -= heapBytes;
  residentPages_ -= extent.pageCount;
  extent.heap.reset();
  return true;
}

bool TQ4PageStorage::releaseReady() const noexcept {
  return !backend_.sparseUnmapPending();
}

void TQ4PageStorage::awaitRelease() { backend_.drainSparseUnmaps(); }

const TQ4LayerStorage &TQ4PageStorage::layer(uint32_t index) const {
  if (index >= layers_.size())
    throw std::out_of_range("invalid attention layer index");
  return layers_[index];
}

} // namespace splash::kv
