// Modified for JustFit: product identity and integration.
#pragma once

#include "Checked.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include "metal/abi/CompactGDN.h"
#include "metal/abi/Tq4Window.h"

namespace justfit::model {

inline bool compactGdnEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("JUSTFIT_COMPACT_GDN");
    return value && std::string_view(value) == "1";
  }();
  return enabled;
}

inline bool tq4WindowEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("JUSTFIT_TQ4_WINDOW");
    return value && std::string_view(value) == "1";
  }();
  return enabled;
}

inline uint32_t tq4NormCorrection() noexcept {
  static const uint32_t mode = [] {
    const char *value = std::getenv("JUSTFIT_TQ4_NC");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10) & 3u) : 0u;
  }();
  return mode;
}

// The taps of the GDN short convolution the kernels compute; a state cell
// keeps the inputs of all taps but the current token's.
inline constexpr uint32_t kGdnConvolutionTaps = JUSTFIT_GDN_CONVOLUTION_TAPS;

// Physical state geometry is supplied by the paired target and draft models.
// The engine sees only opaque CompositeState handles and byte accounting.
struct GdnStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && convolutionHistory && convolutionChannels &&
           recurrentGroups && recurrentRows && recurrentColumns;
  }
  [[nodiscard]] constexpr uint64_t convolutionLayerBytes() const noexcept {
    return alignUp(uint64_t{convolutionHistory} * convolutionChannels *
                   bfloat16Bytes);
  }
  [[nodiscard]] constexpr uint64_t convolutionBytes() const noexcept {
    return uint64_t{layers} * convolutionLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t recurrentLayerBytes() const noexcept {
    if (!std::is_constant_evaluated() && compactGdnEnabled())
      return alignUp(uint64_t{recurrentGroups} * JUSTFIT_COMPACT_HEAD_BYTES +
                     uint64_t{recurrentGroups / 3} * 4096u);
    return alignUp(uint64_t{recurrentGroups} * recurrentRows *
                   recurrentColumns * sizeof(float));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t tq4WindowOffset() const noexcept {
    return convolutionBytes() + recurrentBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    if (!std::is_constant_evaluated() && tq4WindowEnabled())
      return tq4WindowOffset() + JUSTFIT_TQ4_WINDOW_BYTES;
    return tq4WindowOffset();
  }

  bool operator==(const GdnStateLayout &) const = default;
};

// One ring of JUSTFIT_DRAFT_SLIDING_WINDOW slots per KV head for the keys and
// one for the values of every draft layer.
struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && kvHeads && headDimension;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes() const noexcept {
    return uint64_t{kvHeads} * JUSTFIT_DRAFT_SLIDING_WINDOW * headDimension *
           bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    return uint64_t{layers} * 2 * tensorBytes();
  }

  bool operator==(const DraftStateLayout &) const = default;
};

struct CompositeStateLayout final {
  // The GDN cells a lane holds, with one draft ring.
  static constexpr uint32_t kLaneGdnCells = 1;

  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.valid();
  }
  [[nodiscard]] constexpr uint64_t commitScratchBytes() const noexcept {
    if (!std::is_constant_evaluated() && compactGdnEnabled())
      return target.convolutionBytes() + uint64_t{target.layers} * target.recurrentGroups * 8 * 128 * sizeof(float);
    return target.convolutionBytes();
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return target.cellBytes() + commitScratchBytes() + draft.ringBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cellBytes() + draft.ringBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace justfit::model
