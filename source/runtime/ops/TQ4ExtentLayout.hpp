// Modified for JustFit: product identity and integration.
#pragma once
#include "ops/PagedKv.hpp"
namespace justfit::kv {
// JustFit TQ4 codec: 256 four-bit codebook indices and one FP16 norm
// per head/token, independently for K and V. The extent plane layout uses
// the inherited address table. This does not enable TQ4 runtime dispatch.
struct TQ4ExtentLayout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension == 256;
  }
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * headDimension / 2;
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * sizeof(uint16_t);
  }
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return 2 * (dataBytesPerLayerPage() + scaleBytesPerLayerPage());
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint32_t extentAlignmentPages() const noexcept {
    if (!valid()) return 0;
    return static_cast<uint32_t>(detail::lcm(
      detail::pagesForAlignedRegion(dataBytesPerLayerPage()),
      detail::pagesForAlignedRegion(scaleBytesPerLayerPage())));
  }
  [[nodiscard]] constexpr uint32_t minimumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units =
        std::max<uint64_t>(1, (kAllocationExtentTargetBytes / 2 + unit - 1) / unit);
    return static_cast<uint32_t>(units * extentAlignmentPages());
  }
  [[nodiscard]] constexpr uint32_t maximumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units = kAllocationExtentTargetBytes * 3 / 2 / unit;
    return std::max(static_cast<uint32_t>(units * extentAlignmentPages()),
                    minimumExtentPages());
  }

  // The extent size of a pool that holds `pages` pages: all of a pool's
  // extents hold the same number of pages, because kernels find a layer's
  // region from that number. Of the sizes above that fit the pool, the one
  // that leaves the fewest pages over, and the one nearest the target on a
  // tie. Zero when the pool holds fewer pages than the smallest size.
  [[nodiscard]] constexpr uint32_t extentPagesFor(uint64_t pages) const noexcept {
    const uint32_t step = extentAlignmentPages();
    const uint64_t smallest = minimumExtentPages();
    if (!smallest || pages < smallest)
      return 0;
    const uint64_t largest = std::min<uint64_t>(maximumExtentPages(), pages);
    uint64_t best = 0, bestLeft = 0, bestDistance = 0;
    for (uint64_t extent = smallest; extent <= largest; extent += step) {
      const uint64_t bytes = extent * bytesPerModelPage();
      const uint64_t distance = bytes > kAllocationExtentTargetBytes
                                    ? bytes - kAllocationExtentTargetBytes
                                    : kAllocationExtentTargetBytes - bytes;
      if (!best || pages % extent < bestLeft ||
          (pages % extent == bestLeft && distance < bestDistance)) {
        best = extent;
        bestLeft = pages % extent;
        bestDistance = distance;
      }
    }
    return static_cast<uint32_t>(best);
  }

  bool operator==(const TQ4ExtentLayout &) const = default;
};
} // namespace justfit::kv
