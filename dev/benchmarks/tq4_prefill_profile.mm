#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/abi/PagedAttention.h"
#include "ops/PagedKv.hpp"
#include "dev/tests/engine/Q8PageFormatReference.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::kv;

namespace {
struct StagedParams {
  uint32_t committedTokens;
  uint32_t rows;
  uint32_t chunkStride;
  uint32_t visibleTokens;
  uint32_t layerIndex;
  uint32_t kvHead;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(StagedParams) == 32);
id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal allocation failed");
  return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
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

void run(const char *libraryPath, uint32_t splits) {
  constexpr TQ4Layout layout{16, 4, 256};
  constexpr uint32_t layer = 3;
  constexpr uint32_t rows = SPLASH_PREFILL_TOKEN_BUDGET;
  constexpr uint32_t fusedRows = SPLASH_PREFILL_ATTENTION_TILE_ROWS * 6;
  constexpr uint32_t tiles =
      (rows + SPLASH_PREFILL_ATTENTION_TILE_ROWS - 1) /
      SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  if (!splits || splits > SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS)
    throw std::invalid_argument("invalid split count");

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithURL:[NSURL fileURLWithPath:
                                   [NSString stringWithUTF8String:libraryPath]]
                   error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  auto split = makePipeline(device, library,
                            @"prefill_attention_tq4_mpp_split");
  auto reduce = makePipeline(device, library,
                             @"prefill_attention_tq4_mpp_reduce");
  auto stage = makePipeline(device, library,
                            @"prefill_attention_tq4_stage_head");
  auto stagedAttention = makePipeline(
      device, library, @"prefill_attention_tq4_staged_head");
  id<MTLCommandQueue> queue = [device newCommandQueue];

  const uint64_t queryElements = uint64_t{4} * rows * 6 * 256;
  id<MTLBuffer> queries = makeBuffer(device, queryElements * sizeof(uint16_t));
  id<MTLBuffer> output = makeBuffer(device, queryElements * sizeof(uint16_t));
  auto *queryData = static_cast<uint16_t *>(queries.contents);
  for (uint64_t index = 0; index < queryElements; ++index)
    queryData[index] = floatToBFloat16(
        float(int32_t((index * 17 + index / 31) % 257) - 128) / 512.0f);

  for (uint32_t tokens : {32768u, 131072u, 262144u}) {
    const uint32_t pages = (tokens + 31) / 32;
    id<MTLBuffer> modelPages =
        makeBuffer(device, uint64_t{pages} * layout.bytesPerModelPage());
    id<MTLBuffer> table = makeBuffer(device, uint64_t{pages} * sizeof(uint32_t));
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
      std::memset(layerPage, 0x11, layout.indexBytesPerLayerPage());
      std::memset(layerPage + layout.indexBytesPerLayerPage() +
                                  layout.normBytesPerLayerPage(),
                  0x22, layout.indexBytesPerLayerPage());
    }
    id<MTLBuffer> partials = makeBuffer(
        device, uint64_t{tiles} * 4 * splits * fusedRows * 256 * sizeof(float));
    id<MTLBuffer> statistics = makeBuffer(
        device, uint64_t{tiles} * 4 * splits * fusedRows * 2 * sizeof(float));
    const SplashTQ4PrefillAttentionParams params{
        {tokens - rows, rows, rows, pages, pages, splits, 0, 0},
        layer, 0, 0, 0};
    id<MTLBuffer> stagedKeys = makeBuffer(
        device, uint64_t{tokens} * 256 * sizeof(uint16_t));
    id<MTLBuffer> stagedValues = makeBuffer(
        device, uint64_t{tokens} * 256 * sizeof(uint16_t));

    std::vector<double> samples;
    for (uint32_t sample = 0; sample < 5; ++sample) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:split];
      [encoder setBuffer:queries offset:0 atIndex:0];
      [encoder setBuffer:modelPages offset:0 atIndex:1];
      [encoder setBuffer:partials offset:0 atIndex:2];
      [encoder setBuffer:statistics offset:0 atIndex:3];
      [encoder setBuffer:table offset:0 atIndex:4];
      [encoder setBytes:&params length:sizeof(params) atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(4, tiles, splits)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder setComputePipelineState:reduce];
      [encoder setBuffer:partials offset:0 atIndex:0];
      [encoder setBuffer:statistics offset:0 atIndex:1];
      [encoder setBuffer:output offset:0 atIndex:2];
      [encoder setBytes:&params length:sizeof(params) atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(4, fusedRows, tiles)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      if (sample)
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0);
    }
    std::cout << "tile_rows=" << SPLASH_PREFILL_ATTENTION_TILE_ROWS
              << " splits=" << splits
              << " visible_tokens=" << tokens << " tiles=" << tiles
              << " median_gpu_ms=" << median(samples) << '\n';
    std::vector<uint16_t> baseline(queryElements);
    std::memcpy(baseline.data(), output.contents, output.length);

    samples.clear();
    for (uint32_t sample = 0; sample < 4; ++sample) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      for (uint32_t head = 0; head < 4; ++head) {
        const StagedParams stagedParams{tokens - rows, rows, rows, tokens,
                                        layer, head, 0, 0};
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:stage];
        [encoder setBuffer:modelPages offset:0 atIndex:0];
        [encoder setBuffer:stagedKeys offset:0 atIndex:1];
        [encoder setBuffer:stagedValues offset:0 atIndex:2];
        [encoder setBuffer:table offset:0 atIndex:3];
        [encoder setBytes:&stagedParams length:sizeof(stagedParams) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(pages, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:stagedAttention];
        [encoder setBuffer:queries offset:0 atIndex:0];
        [encoder setBuffer:stagedKeys offset:0 atIndex:1];
        [encoder setBuffer:stagedValues offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&stagedParams length:sizeof(stagedParams) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(tiles, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
      if (sample)
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0);
    }
    std::cout << "tile_rows=" << SPLASH_PREFILL_ATTENTION_TILE_ROWS
              << " staged_n=64 visible_tokens=" << tokens
              << " tiles=" << tiles
              << " median_gpu_ms=" << median(samples) << '\n';
    auto *candidate = static_cast<const uint16_t *>(output.contents);
    double absoluteSum = 0.0;
    float maximumAbsolute = 0.0f;
    uint64_t nonfinite = 0;
    for (uint64_t index = 0; index < queryElements; ++index) {
      const float expected = bfloat16ToFloat(baseline[index]);
      const float actual = bfloat16ToFloat(candidate[index]);
      if (!std::isfinite(expected) || !std::isfinite(actual)) {
        ++nonfinite;
        continue;
      }
      const float difference = std::abs(expected - actual);
      maximumAbsolute = std::max(maximumAbsolute, difference);
      absoluteSum += difference;
    }
    std::cout << "comparison visible_tokens=" << tokens
              << " max_abs=" << maximumAbsolute
              << " mean_abs=" << absoluteSum / double(queryElements)
              << " nonfinite=" << nonfinite << '\n';
  }
}
} // namespace

int main(int argc, const char **argv) {
  if (argc != 3) {
    std::cerr << "usage: tq4_prefill_profile METALLIB SPLITS\n";
    return 2;
  }
  try {
    run(argv[1], static_cast<uint32_t>(std::stoul(argv[2])));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "tq4 prefill profile failed: " << error.what() << '\n';
    return 1;
  }
}
