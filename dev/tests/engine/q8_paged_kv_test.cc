#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace splash::kv;

namespace {

void testByteAccounting() {
  static_assert(kBytesPerModelPage == 1'064'960);
  StorageByteCounts one = storageByteCounts(1);
  assert(one.keyData == 512 * 1024);
  assert(one.keyScales == 8 * 1024);
  assert(one.valueData == 512 * 1024);
  assert(one.valueScales == 8 * 1024);
  assert(one.total == 1'064'960);
  StorageByteCounts production = storageByteCounts(4608);
  assert(production.total == 4'907'335'680ULL); // 147456 tokens.

  constexpr TQ4Layout tq4{16, 4, 256};
  static_assert(tq4.valid());
  static_assert(tq4.packedWordsPerVector() == 32);
  static_assert(tq4.indexBytesPerLayerPage() == 16'384);
  static_assert(tq4.normBytesPerLayerPage() == 256);
  static_assert(tq4.bytesPerModelPage() == 532'480);
  static_assert(tq4.bytesPerModelPage() / kPageTokens == 16'640);
  static_assert(tq4.sparseMappingBatchPages() == 8);
  static_assert(tq4.backingExtentPages() == 256);
  static_assert(tq4.storageByteCounts(8'192).total ==
                4'362'076'160ULL); // 262,144 tokens = 4,160 MiB.
}

void testLayouts() {
  assert(keyDataIndex(0, 0, 1) == keyDataIndex(0, 0, 0) + 1);
  assert(keyDataIndex(0, 1, 0) == keyDataIndex(0, 0, 0) + kHeadDimension);
  assert(valueDataIndex(0, 1, 0) == valueDataIndex(0, 0, 0) + 1);
  assert(valueDataIndex(0, 0, 1) == valueDataIndex(0, 0, 0) + kPageTokens);
  assert(keyDataIndex(kKvHeads - 1, kPageTokens - 1, kHeadDimension - 1) ==
         kElementsPerLayerPage - 1);
  assert(valueDataIndex(kKvHeads - 1, kPageTokens - 1, kHeadDimension - 1) ==
         kElementsPerLayerPage - 1);
}

void testBFloat16() {
  for (float value : {0.0f, -0.0f, 1.0f, -3.5f, 1.0e-20f, 65504.0f}) {
    float roundTrip = bfloat16ToFloat(floatToBFloat16(value));
    if (value == 0.0f) {
      assert(roundTrip == value);
    } else {
      assert(std::abs(roundTrip - value) <= std::abs(value) / 128.0f);
    }
  }
  assert(std::isinf(bfloat16ToFloat(
      floatToBFloat16(std::numeric_limits<float>::infinity()))));
  assert(std::isnan(bfloat16ToFloat(
      floatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

void testLayoutGuard() {
  std::array<uint8_t, 32> digest{};
  for (uint32_t i = 0; i < digest.size(); ++i)
    digest[i] = uint8_t(i);
  Q8LayoutGuard first = makeQ8LayoutGuard(kOracleQ8Layout, digest);
  Q8LayoutGuard second = makeQ8LayoutGuard(kOracleQ8Layout, digest);
  assert(matchesQ8Layout(first, kOracleQ8Layout));
  assert(first.modelArtifactSha256 == second.modelArtifactSha256);
  second.elementsPerScale = 32;
  assert(!isValidQ8LayoutGuard(second));
  second = first;
  ++second.modelArtifactSha256[0];
  assert(isValidQ8LayoutGuard(second));
  assert(first.modelArtifactSha256 != second.modelArtifactSha256);
}

void testQuantization(uint32_t validTokens) {
  uint64_t inputElements = uint64_t{validTokens} * kKvHeads * kHeadDimension;
  std::vector<float> keys(inputElements);
  std::vector<float> values(inputElements);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t index = logicalIndex(token, head, dimension);
        keys[index] =
            2.5f * std::sin(float(token * 17 + head * 31 + dimension) * 0.013f);
        values[index] =
            1.75f *
            std::cos(float(token * 29 + head * 11 + dimension) * 0.017f);
      }
    }
  }
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(keys, values, validTokens, *page);
  std::vector<float> decodedKeys(kElementsPerLayerPage);
  std::vector<float> decodedValues(kElementsPerLayerPage);
  dequantizeLayerPage(*page, validTokens, decodedKeys, decodedValues);

  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t logical = logicalIndex(token, head, dimension);
        float keyScale = page->keyScales[keyScaleIndex(head, token)];
        float valueScale = page->valueScales[valueScaleIndex(head, token)];
        assert(std::abs(decodedKeys[logical] - keys[logical]) <=
               keyScale * 0.51f + 1.0e-7f);
        assert(std::abs(decodedValues[logical] - values[logical]) <=
               valueScale * 0.51f + 1.0e-7f);
        assert(decodedKeys[logical] ==
               dequantizeKey(*page, head, token, dimension));
        assert(decodedValues[logical] ==
               dequantizeValue(*page, head, token, dimension));
      }
    }
  }
  for (uint32_t token = validTokens; token < kPageTokens; ++token) {
    uint64_t logical = logicalIndex(token, 0, 0);
    assert(decodedKeys[logical] == 0.0f);
    assert(decodedValues[logical] == 0.0f);
  }
}

void testZeroAndInvalidInputs() {
  constexpr uint32_t tokens = 1;
  std::vector<float> values(uint64_t{tokens} * kKvHeads * kHeadDimension);
  auto page = std::make_unique<Q8LayerPage>();
  quantizeLayerPage(values, values, tokens, *page);
  assert(std::all_of(page->keys.begin(), page->keys.end(),
                     [](int8_t value) { return value == 0; }));
  assert(std::all_of(page->values.begin(), page->values.end(),
                     [](int8_t value) { return value == 0; }));
  values[0] = std::numeric_limits<float>::quiet_NaN();
  bool rejected = false;
  try {
    quantizeLayerPage(values, values, tokens, *page);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

} // namespace

int main() {
  testByteAccounting();
  testLayouts();
  testBFloat16();
  testLayoutGuard();
  testQuantization(kPageTokens);
  testQuantization(17);
  testZeroAndInvalidInputs();
  std::cout << "q8_paged_kv_test: ok\n";
}
