// Modified for JustFit: product identity and integration.
#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
namespace justfit::model {
// Wire rows are unpadded; the GPU embedding buffer holds padded rows.
inline void copyVisionRows(std::span<const uint8_t> payload, std::span<uint8_t> buffer) {
  if (payload.size() > buffer.size()) throw std::invalid_argument("vision rows exceed embedding buffer");
  std::copy(payload.begin(), payload.end(), buffer.begin());
  std::fill(buffer.begin() + payload.size(), buffer.end(), uint8_t{0});
}
}
