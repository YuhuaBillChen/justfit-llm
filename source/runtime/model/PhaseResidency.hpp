// Modified for JustFit: product identity and integration.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string_view>

namespace justfit::model {
inline constexpr std::string_view kVisionWeightComponent = "vision/model.bin";

inline bool visionPhaseEnabled() noexcept {
  const char *value = std::getenv("JUSTFIT_VISION_PHASE");
  return value && std::strcmp(value, "1") == 0;
}
} // namespace justfit::model
