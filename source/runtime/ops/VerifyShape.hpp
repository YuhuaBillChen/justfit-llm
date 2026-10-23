// Modified for JustFit: product identity and integration.
#pragma once
#include <cstdint>
#include <stdexcept>
namespace justfit::ops {
inline thread_local uint32_t currentVerifyRows = 8;
inline uint32_t activeVerifyRows() noexcept { return currentVerifyRows; }
class VerifyShapeScope {
 uint32_t prior_;
public:
 explicit VerifyShapeScope(uint32_t rows):prior_(currentVerifyRows) {
  if(rows!=8 && rows!=16) throw std::invalid_argument("verify shape must be 8 or 16");
  currentVerifyRows=rows;
 }
 ~VerifyShapeScope(){currentVerifyRows=prior_;}
 VerifyShapeScope(const VerifyShapeScope&)=delete;
 VerifyShapeScope& operator=(const VerifyShapeScope&)=delete;
};
}
