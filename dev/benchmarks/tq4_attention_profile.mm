#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/abi/PagedAttention.h"
#include "ops/PagedKv.hpp"
#include "dev/tests/engine/Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace splash::kv;

namespace {

id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal allocation failed for " +
                             std::to_string(bytes) + " bytes");
  return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString *name) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run(const char *libraryPath) {
  constexpr TQ4Layout layout{16, 4, 256};
  constexpr uint32_t layer = 3;
  constexpr uint32_t stride = 32;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithURL:[NSURL fileURLWithPath:
                                   [NSString stringWithUTF8String:libraryPath]]
                   error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  const auto rotate =
      pipeline(device, library, @"verify_attention_tq4_rotate_queries");
  const auto attention = pipeline(device, library, @"verify_attention_tq4");
  const auto mppSplit =
      pipeline(device, library, @"verify_attention_tq4_mpp_split");
  const auto mppReduce =
      pipeline(device, library, @"verify_attention_tq4_mpp_reduce");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  constexpr uint64_t queryElements = uint64_t{4} * stride * 6 * 256;
  id<MTLBuffer> queries = buffer(device, queryElements * sizeof(uint16_t));
  id<MTLBuffer> rotated = buffer(device, queryElements * sizeof(uint16_t));
  id<MTLBuffer> output = buffer(device, queryElements * sizeof(uint16_t));
  auto *queryBits = static_cast<uint16_t *>(queries.contents);
  for (uint64_t i = 0; i < queryElements; ++i)
    queryBits[i] = floatToBFloat16(float(int(i % 37) - 18) / 23.0f);

  for (uint32_t tokens : {32768u, 131072u, 262144u}) {
    const uint32_t pages = (tokens + 31) / 32;
    id<MTLBuffer> modelPages =
        buffer(device, uint64_t{pages} * layout.bytesPerModelPage());
    id<MTLBuffer> table = buffer(device, uint64_t{pages} * sizeof(uint32_t));
    auto *tableData = static_cast<uint32_t *>(table.contents);
    auto *base = static_cast<uint8_t *>(modelPages.contents);
    for (uint32_t page = 0; page < pages; ++page) {
      tableData[page] = page;
      auto *layerPage = base + uint64_t{page} * layout.bytesPerModelPage() +
                        uint64_t{layer} * layout.bytesPerLayerPage();
      auto *keyNorms = reinterpret_cast<_Float16 *>(
          layerPage + layout.indexBytesPerLayerPage());
      auto *valueNorms = reinterpret_cast<_Float16 *>(
          layerPage + 2 * layout.indexBytesPerLayerPage() +
          layout.normBytesPerLayerPage());
      std::fill_n(keyNorms, 4 * 32, _Float16(1.0f));
      std::fill_n(valueNorms, 4 * 32, _Float16(1.0f));
    }
    std::array<SplashTQ4StoreParams, 4> params{};
    params[0] = {{tokens - SPLASH_TARGET_VERIFY_ROWS,
                  SPLASH_TARGET_VERIFY_ROWS, stride, pages, pages, 0, 0, 0},
                 layer, 0, 0, 0};
    id<MTLBuffer> paramBuffer = buffer(device, sizeof(params));
    std::memcpy(paramBuffer.contents, params.data(), sizeof(params));

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:rotate];
    [encoder setBuffer:queries offset:0 atIndex:0];
    [encoder setBuffer:rotated offset:0 atIndex:1];
    [encoder setBuffer:paramBuffer offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                       24 * SPLASH_TARGET_VERIFY_ROWS, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    std::vector<double> samples;
    for (uint32_t sample = 0; sample < 12; ++sample) {
      command = [queue commandBuffer];
      encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:attention];
      [encoder setBuffer:rotated offset:0 atIndex:0];
      [encoder setBuffer:modelPages offset:0 atIndex:1];
      [encoder setBuffer:output offset:0 atIndex:2];
      for (NSUInteger index = 3; index <= 6; ++index)
        [encoder setBuffer:table offset:0 atIndex:index];
      [encoder setBuffer:paramBuffer offset:0 atIndex:7];
      [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      if (sample >= 2)
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0);
    }
    const double scalarLayerMs = median(samples);
    constexpr uint32_t splits = 32;
    constexpr uint64_t fusedRows =
        SPLASH_TARGET_VERIFY_ROWS * uint64_t{6};
    id<MTLBuffer> partials = buffer(
        device, uint64_t{4} * splits * fusedRows * 256 * sizeof(float));
    id<MTLBuffer> statistics = buffer(
        device, uint64_t{4} * splits * fusedRows * 2 * sizeof(float));
    std::array<SplashTQ4VerifyAttentionParams, 4> mppParams{};
    mppParams[0] = {{tokens - SPLASH_TARGET_VERIFY_ROWS,
                     SPLASH_TARGET_VERIFY_ROWS, stride, pages, pages, splits,
                     splits, 0},
                    layer, 0, 0, 0};
    id<MTLBuffer> mppParamsBuffer = buffer(device, sizeof(mppParams));
    std::memcpy(mppParamsBuffer.contents, mppParams.data(), sizeof(mppParams));
    samples.clear();
    for (uint32_t sample = 0; sample < 12; ++sample) {
      command = [queue commandBuffer];
      encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:mppSplit];
      [encoder setBuffer:queries offset:0 atIndex:0];
      [encoder setBuffer:modelPages offset:0 atIndex:1];
      [encoder setBuffer:partials offset:0 atIndex:2];
      [encoder setBuffer:statistics offset:0 atIndex:3];
      for (NSUInteger index = 4; index <= 7; ++index)
        [encoder setBuffer:table offset:0 atIndex:index];
      [encoder setBuffer:mppParamsBuffer offset:0 atIndex:8];
      [encoder dispatchThreadgroups:MTLSizeMake(4, splits, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder setComputePipelineState:mppReduce];
      [encoder setBuffer:partials offset:0 atIndex:0];
      [encoder setBuffer:statistics offset:0 atIndex:1];
      [encoder setBuffer:output offset:0 atIndex:2];
      [encoder setBuffer:mppParamsBuffer offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(4, fusedRows, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      if (sample >= 2)
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0);
    }
    const double mppLayerMs = median(samples);
    std::cout << "tokens=" << tokens << " pages=" << pages
              << " tq4_pool_mib="
              << double(uint64_t{pages} * layout.bytesPerModelPage()) /
                     (1024.0 * 1024.0)
              << " scalar_layer_ms=" << scalarLayerMs
              << " mpp_layer_ms=" << mppLayerMs
              << " mpp_projected_16_layer_ms=" << mppLayerMs * 16.0
              << " mpp_speedup=" << scalarLayerMs / mppLayerMs << '\n';
  }
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tq4_attention_profile METALLIB\n";
    return 2;
  }
  try {
    run(argv[1]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "tq4 attention profile failed: " << error.what() << '\n';
    return 1;
  }
}
