#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace splash::kv {

// Physical backing for the engine's page pool. Implementations provide Metal
// storage or deterministic test storage.
class Backing {
public:
  virtual ~Backing() = default;
  [[nodiscard]] virtual uint32_t pageCount() const noexcept = 0;
  [[nodiscard]] virtual uint64_t bytesPerPage() const noexcept = 0;
  [[nodiscard]] virtual bool isResident(uint32_t page) const = 0;
  [[nodiscard]] virtual metal::AllocationResult ensureResident(uint32_t page) = 0;
  [[nodiscard]] virtual bool releaseBackingForPage(uint32_t page) = 0;
  [[nodiscard]] virtual uint32_t extentFirstPage(uint32_t page) const = 0;
  [[nodiscard]] virtual uint32_t extentPageCount(uint32_t page) const = 0;
  // Physical release is asynchronous and paced: while a previous release is
  // still being torn down by the kernel, callers keep the next empty extent
  // resident instead of queueing more unmap work. Test backings are always
  // ready; awaitRelease() blocks only at startup and shutdown.
  [[nodiscard]] virtual bool releaseReady() const noexcept { return true; }
  virtual void awaitRelease() {}
};

struct Q8LayerStorage final {
  metal::MetalBuffer keyData;
  metal::MetalBuffer keyScales;
  metal::MetalBuffer valueData;
  metal::MetalBuffer valueScales;
};

// TurboQuant Q4 uses one fp16 L2 norm and 256 packed four-bit codebook
// indices per (token, KV head), independently for keys and values.  Keeping
// the physical cache block at Page32 preserves Splash's prefix and scheduler
// granularity; the codec does not require the 256-token allocation pages used
// by the MLX implementation.
struct TQ4LayerStorage final {
  metal::MetalBuffer modelPages;
  uint32_t layerIndex = 0;
};

// Shared cache format and execution limits; model dimensions live in Q8Layout.
inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
inline constexpr uint32_t kMaximumLogicalTokens =
    SPLASH_MAXIMUM_CONTEXT_TOKENS;
inline constexpr uint32_t kMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;
inline constexpr int32_t kQuantizedMinimum = -127;
inline constexpr int32_t kQuantizedMaximum = 127;
inline constexpr uint64_t kSparseMappingAlignmentBytes = 64 * 1024;
inline constexpr uint64_t kAllocationExtentTargetBytes =
    SPLASH_ALLOCATION_EXTENT_TARGET_BYTES;

struct StorageByteCounts final {
  uint64_t keyData = 0;
  uint64_t keyScales = 0;
  uint64_t valueData = 0;
  uint64_t valueScales = 0;
  uint64_t total = 0;
};

namespace detail {

[[nodiscard]] constexpr uint64_t gcd(uint64_t left, uint64_t right) noexcept {
  while (right) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

[[nodiscard]] constexpr uint64_t lcm(uint64_t left, uint64_t right) noexcept {
  return left && right ? left / gcd(left, right) * right : 0;
}

[[nodiscard]] constexpr uint64_t pagesForAlignedMapping(
    uint64_t bytesPerPage) noexcept {
  return bytesPerPage
             ? kSparseMappingAlignmentBytes /
                   gcd(kSparseMappingAlignmentBytes, bytesPerPage)
             : 0;
}

} // namespace detail

// Physical KV geometry: Page32 with per-(token, head) symmetric INT8 scaling.
// Layer and head counts vary by target.
struct Q8Layout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension;
  }
  [[nodiscard]] constexpr uint32_t elementsPerScale() const noexcept {
    return headDimension;
  }
  [[nodiscard]] constexpr uint64_t elementsPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * headDimension;
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads;
  }
  // Keys and values share one data and one scale geometry per layer page.
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPage() const noexcept {
    return elementsPerLayerPage() * sizeof(int8_t);
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPage() const noexcept {
    return scalesPerTensorLayerPage() * sizeof(float);
  }
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return 2 * (dataBytesPerLayerPage() + scaleBytesPerLayerPage());
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }

  // Metal sparse mappings must begin and end on 64-KiB tile boundaries. The
  // scale buffers are the tightest constraint, so a 4-head model maps 128
  // pages at a time while a 2-head model maps 256. This is physical
  // allocation geometry; prefix matching remains Page32 in both cases.
  [[nodiscard]] constexpr uint32_t sparseMappingBatchPages() const noexcept {
    return static_cast<uint32_t>(detail::lcm(
        detail::pagesForAlignedMapping(dataBytesPerLayerPage()),
        detail::pagesForAlignedMapping(scaleBytesPerLayerPage())));
  }

  [[nodiscard]] constexpr uint64_t sparseMappingBatchBytes() const noexcept {
    return uint64_t{sparseMappingBatchPages()} * bytesPerModelPage();
  }

  [[nodiscard]] constexpr uint32_t backingExtentPages() const noexcept {
    const uint64_t unit = sparseMappingBatchBytes();
    if (!unit)
      return 0;
    return static_cast<uint32_t>(
        ((kAllocationExtentTargetBytes + unit - 1) / unit) *
        sparseMappingBatchPages());
  }

  [[nodiscard]] constexpr StorageByteCounts
  storageByteCounts(uint64_t pageCount) const noexcept {
    const uint64_t data = pageCount * attentionLayers * dataBytesPerLayerPage();
    const uint64_t scale =
        pageCount * attentionLayers * scaleBytesPerLayerPage();
    return {data, scale, data, scale, pageCount * bytesPerModelPage()};
  }

  bool operator==(const Q8Layout &) const = default;
};

struct TQ4Layout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension == 256;
  }
  [[nodiscard]] constexpr uint32_t bitsPerElement() const noexcept {
    return 4;
  }
  [[nodiscard]] constexpr uint32_t packedWordsPerVector() const noexcept {
    return headDimension * bitsPerElement() / 32;
  }
  [[nodiscard]] constexpr uint64_t indexBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * packedWordsPerVector() *
           sizeof(uint32_t);
  }
  [[nodiscard]] constexpr uint64_t normBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * sizeof(uint16_t);
  }
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return 2 * (indexBytesPerLayerPage() + normBytesPerLayerPage());
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint32_t sparseMappingBatchPages() const noexcept {
    return static_cast<uint32_t>(
        detail::pagesForAlignedMapping(bytesPerModelPage()));
  }
  [[nodiscard]] constexpr uint64_t sparseMappingBatchBytes() const noexcept {
    return uint64_t{sparseMappingBatchPages()} * bytesPerModelPage();
  }
  [[nodiscard]] constexpr uint32_t backingExtentPages() const noexcept {
    const uint64_t unit = sparseMappingBatchBytes();
    if (!unit)
      return 0;
    return static_cast<uint32_t>(
        ((kAllocationExtentTargetBytes + unit - 1) / unit) *
        sparseMappingBatchPages());
  }
  [[nodiscard]] constexpr StorageByteCounts
  storageByteCounts(uint64_t pageCount) const noexcept {
    const uint64_t indices =
        pageCount * attentionLayers * indexBytesPerLayerPage();
    const uint64_t norms =
        pageCount * attentionLayers * normBytesPerLayerPage();
    return {indices, norms, indices, norms,
            pageCount * bytesPerModelPage()};
  }

  bool operator==(const TQ4Layout &) const = default;
};

enum class Quantization : uint32_t { SymmetricInt8 = 1 };
enum class ScaleType : uint32_t { Float32 = 2 };
enum class KeyLayout : uint32_t { TokenMajor = 1 };
enum class ValueLayout : uint32_t { DimensionMajor = 1 };

// Stable metadata for rejecting incompatible cached blocks before any block is
// read. modelArtifactSha256 is the digest of the exact packed target artifact
// set; geometry/layout fields remain explicit so format changes cannot alias.
struct alignas(8) Q8LayoutGuard final {
  uint32_t quantization = 0;
  uint32_t scaleType = 0;
  uint32_t keyLayout = 0;
  uint32_t valueLayout = 0;
  uint32_t pageTokens = 0;
  uint32_t elementsPerScale = 0;
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  int32_t quantizedMinimum = 0;
  int32_t quantizedMaximum = 0;
  uint64_t bytesPerLayerPage = 0;
  uint64_t bytesPerModelPage = 0;
  std::array<uint8_t, 32> modelArtifactSha256{};
};

static_assert(sizeof(Q8LayoutGuard) == 96);
static_assert(std::is_standard_layout_v<Q8LayoutGuard>);
static_assert(std::is_trivially_copyable_v<Q8LayoutGuard>);

[[nodiscard]] inline Q8LayoutGuard makeQ8LayoutGuard(
    Q8Layout layout,
    const std::array<uint8_t, 32> &modelArtifactSha256) {
  Q8LayoutGuard result;
  result.quantization = uint32_t(Quantization::SymmetricInt8);
  result.scaleType = uint32_t(ScaleType::Float32);
  result.keyLayout = uint32_t(KeyLayout::TokenMajor);
  result.valueLayout = uint32_t(ValueLayout::DimensionMajor);
  result.pageTokens = kPageTokens;
  result.elementsPerScale = layout.elementsPerScale();
  result.attentionLayers = layout.attentionLayers;
  result.kvHeads = layout.kvHeads;
  result.headDimension = layout.headDimension;
  result.quantizedMinimum = kQuantizedMinimum;
  result.quantizedMaximum = kQuantizedMaximum;
  result.bytesPerLayerPage = layout.bytesPerLayerPage();
  result.bytesPerModelPage = layout.bytesPerModelPage();
  result.modelArtifactSha256 = modelArtifactSha256;
  return result;
}

} // namespace splash::kv
