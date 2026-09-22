#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace splash::kv {

// Placement-sparse backing for TurboQuant-Q4 KV.  Logical cache blocks stay
// Page32; physical mappings are admitted in the layout's 64-KiB-aligned
// extents.  Codebooks and deterministic RHT signs are codec metadata and are
// not duplicated per page.
class TQ4PageStorage final : public Backing {
public:
  TQ4PageStorage(metal::MetalBackend &backend,
                 metal::AllocationAdmission admitAllocation,
                 TQ4Layout layout, uint32_t pageCount);
  ~TQ4PageStorage() override;

  TQ4PageStorage(const TQ4PageStorage &) = delete;
  TQ4PageStorage &operator=(const TQ4PageStorage &) = delete;

  [[nodiscard]] uint32_t pageCount() const noexcept override {
    return pageCount_;
  }
  [[nodiscard]] uint64_t bytesPerPage() const noexcept override {
    return layout_.bytesPerModelPage();
  }
  [[nodiscard]] TQ4Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t sparseMappingBatchPages() const noexcept {
    return layout_.sparseMappingBatchPages();
  }
  [[nodiscard]] uint32_t backingExtentPages() const noexcept {
    return layout_.backingExtentPages();
  }
  [[nodiscard]] uint64_t declaredBytes() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] uint32_t residentPages() const noexcept;
  [[nodiscard]] bool isResident(uint32_t page) const override;
  [[nodiscard]] metal::AllocationResult ensureResident(uint32_t page) override;
  [[nodiscard]] bool releaseBackingForPage(uint32_t page) override;
  [[nodiscard]] bool releaseReady() const noexcept override;
  void awaitRelease() override;
  [[nodiscard]] uint32_t extentFirstPage(uint32_t page) const override;
  [[nodiscard]] uint32_t extentPageCount(uint32_t page) const override;
  [[nodiscard]] const TQ4LayerStorage &layer(uint32_t index) const;

private:
  struct Extent {
    uint32_t firstPage = 0;
    uint32_t pageCount = 0;
    std::optional<metal::SparseHeap> heap;
  };

  [[nodiscard]] size_t extentIndex(uint32_t page) const;
  [[nodiscard]] std::vector<metal::SparseMapping>
  mappingsFor(const Extent &extent) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  TQ4Layout layout_;
  uint32_t pageCount_ = 0;
  metal::MetalBuffer modelPages_;
  std::vector<TQ4LayerStorage> layers_;
  std::vector<Extent> extents_;
  uint64_t residentBackingBytes_ = 0;
  uint32_t residentPages_ = 0;
};

} // namespace splash::kv
