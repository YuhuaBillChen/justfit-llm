// Modified for JustFit: preserve a released calibration across runtime budgets.
#pragma once
#include "ops/AneFfnCalibration.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace justfit::engine {
struct CompatibleAneCalibration {
  ops::ane_ffn::Calibration calibration;
  std::string sourceKey;
};
// Exact released M6 native identity from its qualified 2.3.1 receipt. Layer
// shapes, weight layouts, device and full OS identity remain part of the key.
inline constexpr std::string_view kReleasedM6CalibrationDigest =
    "a82317114d1d72af55aefc87086ef7977b1edcc9bfbc9a3109c3568c62f1648e";
inline std::string releasedM6CalibrationBuild() {
  // Keep a legacy lookup key distinct from this binary's own embedded src-ID.
  return std::string("src-") + std::string(kReleasedM6CalibrationDigest);
}
inline std::optional<CompatibleAneCalibration> releasedM6Calibration(
    std::string_view context, std::string_view device, uint32_t maximumUnits,
    uint32_t totalUnits, uint32_t minimumRows, uint32_t maximumRows,
    const std::function<std::optional<std::string>(const std::string &)> &recall) {
  if (!(device == "Apple M6" || device.starts_with("Apple M6 ")) ||
      maximumUnits == 0 || maximumUnits > totalUnits || totalUnits > 4096)
    return std::nullopt;
  const std::string prefix = std::string(context) + " build " +
                             releasedM6CalibrationBuild() + " most ";
  // Prefer the same budget, then the nearest larger/smaller released budget.
  // A recalled split still has to fit TODAY's maximum; a larger old budget
  // never authorizes allocating more memory in this process.
  for (uint32_t distance = 0; distance < totalUnits; ++distance) {
    for (bool larger : {true, false}) {
      if (!larger && distance == 0) continue;
      const int64_t oldMaximum = int64_t(maximumUnits) + (larger ? int64_t(distance) : -int64_t(distance));
      if (oldMaximum < 1 || oldMaximum > totalUnits) continue;
      const std::string key = prefix + std::to_string(oldMaximum);
      const auto line = recall(key);
      const auto calibration = line ? ops::ane_ffn::Calibration::parse(*line) : std::nullopt;
      // Negative timing results are budget-specific; never migrate them.
      if (calibration && calibration->aneUnits && calibration->aneUnits <= maximumUnits &&
          calibration->minimumRows >= minimumRows && calibration->minimumRows <= maximumRows)
        return CompatibleAneCalibration{*calibration, key};
    }
  }
  return std::nullopt;
}
} // namespace justfit::engine
