#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/abi/PagedAttention.h"
#include "ops/PagedKv.hpp"
#include "Q8PageFormatReference.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::kv;

namespace {

constexpr std::array<uint32_t, 32> expectedKeys{
    0x58b74d20, 0xe9347388, 0x3991723b, 0xbb4467b5, 0xc1a48c17,
    0x58b52984, 0x476b727b, 0x82799d8d, 0xb9a84b47, 0x8c6b569c,
    0x5672677b, 0x55485671, 0xbab76489, 0x4857da4e, 0xaa5aa999,
    0x78d9628f, 0x483a5e2b, 0x542dcaa8, 0x2445263d, 0x6c788793,
    0x6cad488a, 0x65469baa, 0xa4a99a72, 0x2732aeb1, 0x39a27a63,
    0x97449a9b, 0x37aabec6, 0x64382a24, 0x8945d955, 0x3d2c5396,
    0x3e816852, 0xa1b45792};
constexpr std::array<uint32_t, 32> expectedValues{
    0x88e6649e, 0xfe4f3a27, 0xaa814c79, 0xa029a3a3, 0x7cb68594,
    0x4b86aa73, 0xa4c69316, 0xa874d727, 0xd5859da8, 0x323a7a33,
    0x7572e6b7, 0x45b86657, 0x567944bb, 0x86981a92, 0x37aa6e42,
    0x78948824, 0x1399595d, 0x59b3573d, 0x2a67a6d9, 0x5878a578,
    0x6882b498, 0xbd678dac, 0xa67ad724, 0x756121a6, 0x81779443,
    0x3693a398, 0xa6337379, 0xb50b9849, 0x586ec5e2, 0x78866726,
    0xdcc7b4c4, 0x5626f786};

id<MTLBuffer> makeBuffer(id<MTLDevice> device, NSUInteger bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString *name) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function)
    throw std::runtime_error("missing Metal function " +
                             std::string(name.UTF8String));
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

float tq4Codebook(uint32_t code) {
  static constexpr std::array<float, 16> values{
      -0.168522835f, -0.127540648f, -0.0996119156f, -0.0771940202f,
      -0.0578124076f, -0.0402186327f, -0.0237036943f, -0.00783268455f,
       0.00783268549f, 0.0237036962f, 0.0402186289f, 0.0578124076f,
       0.0771940276f, 0.0996119007f, 0.127540648f, 0.168522835f};
  return values.at(code);
}

float tq4Sign(bool valueTensor, uint32_t dimension) {
  static constexpr std::array<uint32_t, 8> keyBits{
      0xe05422cdu, 0xf23e567du, 0xed27523du, 0x5a9acdcau,
      0x22d0370du, 0x10b3a0b8u, 0xf87f105fu, 0x3efbe5ceu};
  static constexpr std::array<uint32_t, 8> valueBits{
      0xa45beb87u, 0xac6b118eu, 0xeafcb960u, 0x7c1e2f3bu,
      0xeac834c2u, 0xac17ef56u, 0xd86514b1u, 0xe2c2bde9u};
  const auto &bits = valueTensor ? valueBits : keyBits;
  return ((bits[dimension >> 5] >> (dimension & 31u)) & 1u) ? 1.0f
                                                               : -1.0f;
}

std::array<float, 256> decodedVector(const uint32_t *words, float norm,
                                     bool valueTensor) {
  std::array<float, 256> result{};
  for (uint32_t dimension = 0; dimension < result.size(); ++dimension) {
    const uint32_t word = words[dimension >> 3];
    result[dimension] =
        tq4Codebook((word >> ((dimension & 7u) * 4u)) & 15u);
  }
  for (uint32_t halfSpan = 1; halfSpan < result.size(); halfSpan <<= 1) {
    for (uint32_t base = 0; base < result.size(); base += 2 * halfSpan) {
      for (uint32_t offset = 0; offset < halfSpan; ++offset) {
        const float a = result[base + offset];
        const float b = result[base + offset + halfSpan];
        result[base + offset] = a + b;
        result[base + offset + halfSpan] = a - b;
      }
    }
  }
  for (uint32_t dimension = 0; dimension < result.size(); ++dimension)
    result[dimension] *=
        (norm / 16.0f) * tq4Sign(valueTensor, dimension);
  return result;
}

void run(const char *libraryPath) {
  constexpr uint32_t stride = 32;
  constexpr uint32_t layer = 3;
  constexpr TQ4Layout layout{16, 4, 256};
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLComputePipelineState> storePipeline =
      pipeline(device, library, @"prefill_attention_tq4_store");

  const uint64_t tensorElements = uint64_t{4} * stride * 256;
  id<MTLBuffer> keys = makeBuffer(device, tensorElements * sizeof(uint16_t));
  id<MTLBuffer> values = makeBuffer(device, tensorElements * sizeof(uint16_t));
  auto *keyBits = static_cast<uint16_t *>(keys.contents);
  auto *valueBits = static_cast<uint16_t *>(values.contents);
  for (uint32_t d = 0; d < 256; ++d) {
    keyBits[d] = floatToBFloat16(float(int((d * 17) % 251) - 125) / 37.0f);
    valueBits[d * stride] =
        floatToBFloat16(float(int((d * 29) % 257) - 128) / 41.0f);
    keyBits[256 + d] =
        floatToBFloat16(float(int((d * 23) % 263) - 131) / 43.0f);
    valueBits[d * stride + 1] =
        floatToBFloat16(float(int((d * 31) % 269) - 134) / 47.0f);
  }
  id<MTLBuffer> pages = makeBuffer(device, layout.bytesPerModelPage());
  id<MTLBuffer> table = makeBuffer(device, sizeof(uint32_t));
  *static_cast<uint32_t *>(table.contents) = 0;
  SplashTQ4StoreParams params{{0, 2, stride, 1, 1, 0, 0, 0}, layer, 0, 0, 0};

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:storePipeline];
  [encoder setBuffer:keys offset:0 atIndex:0];
  [encoder setBuffer:values offset:0 atIndex:1];
  [encoder setBuffer:pages offset:0 atIndex:2];
  [encoder setBuffer:table offset:0 atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);

  const auto *base = static_cast<const uint8_t *>(pages.contents) +
                     layer * layout.bytesPerLayerPage();
  const auto *keyWords = reinterpret_cast<const uint32_t *>(base);
  const auto *keyNorms = reinterpret_cast<const _Float16 *>(
      base + layout.indexBytesPerLayerPage());
  const auto *valueWords = reinterpret_cast<const uint32_t *>(
      base + layout.indexBytesPerLayerPage() + layout.normBytesPerLayerPage());
  const auto *valueNorms = reinterpret_cast<const _Float16 *>(
      base + 2 * layout.indexBytesPerLayerPage() +
      layout.normBytesPerLayerPage());
  for (uint32_t word = 0; word < 32; ++word) {
    if (keyWords[word] != expectedKeys[word] ||
        valueWords[word] != expectedValues[word])
      throw std::runtime_error("TQ4 packed indices differ from MLX oracle at word " +
                               std::to_string(word));
  }
  if (std::abs(float(keyNorms[0]) - float(_Float16(31.539794921875f))) > 0 ||
      std::abs(float(valueNorms[0]) - float(_Float16(28.908180236816406f))) > 0)
    throw std::runtime_error("TQ4 fp16 norms differ from MLX oracle");

  // The first causal row sees one token, so its softmax is exactly one. This
  // isolates the packed value consumer and inverse RHT from score tolerance.
  constexpr uint64_t queryElements = uint64_t{4} * stride * 6 * 256;
  id<MTLBuffer> queries = makeBuffer(device, queryElements * sizeof(uint16_t));
  id<MTLBuffer> rotated = makeBuffer(device, queryElements * sizeof(uint16_t));
  id<MTLBuffer> output = makeBuffer(device, queryElements * sizeof(uint16_t));
  auto *queryBits = static_cast<uint16_t *>(queries.contents);
  for (uint64_t index = 0; index < queryElements; ++index)
    queryBits[index] = floatToBFloat16(float(int(index % 31) - 15) / 19.0f);
  std::array<SplashTQ4StoreParams, 4> laneParams{};
  laneParams[0] = params;
  id<MTLBuffer> paramsBuffer = makeBuffer(device, sizeof(laneParams));
  std::memcpy(paramsBuffer.contents, laneParams.data(), sizeof(laneParams));
  id<MTLComputePipelineState> rotatePipeline =
      pipeline(device, library, @"verify_attention_tq4_rotate_queries");
  id<MTLComputePipelineState> attentionPipeline =
      pipeline(device, library, @"verify_attention_tq4");

  command = [queue commandBuffer];
  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:rotatePipeline];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:rotated offset:0 atIndex:1];
  [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
  [encoder dispatchThreadgroups:MTLSizeMake(24 * SPLASH_TARGET_VERIFY_ROWS, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:attentionPipeline];
  [encoder setBuffer:rotated offset:0 atIndex:0];
  [encoder setBuffer:pages offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  for (NSUInteger index = 3; index <= 6; ++index)
    [encoder setBuffer:table offset:0 atIndex:index];
  [encoder setBuffer:paramsBuffer offset:0 atIndex:7];
  [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);

  const auto expected = decodedVector(valueWords, float(valueNorms[0]), true);
  const auto *outputBits = static_cast<const uint16_t *>(output.contents);
  float maximumError = 0.0f;
  for (uint32_t dimension = 0; dimension < 256; ++dimension) {
    const float observed = bfloat16ToFloat(outputBits[dimension]);
    maximumError = std::max(maximumError,
                            std::abs(observed - expected[dimension]));
  }
  if (maximumError > 0.02f)
    throw std::runtime_error("TQ4 one-token attention mismatch: max abs " +
                             std::to_string(maximumError));

  // Row one sees both tokens. Reconstruct the stored TQ4 vectors on the CPU,
  // then compare the causal two-token softmax against the page-native kernel.
  const auto key0 = decodedVector(keyWords, float(keyNorms[0]), false);
  const auto key1 = decodedVector(keyWords + 32, float(keyNorms[1]), false);
  const auto value0 = decodedVector(valueWords, float(valueNorms[0]), true);
  const auto value1 = decodedVector(valueWords + 32, float(valueNorms[1]), true);
  constexpr uint64_t rowOne = uint64_t{6} * 256;
  float score0 = 0.0f;
  float score1 = 0.0f;
  for (uint32_t dimension = 0; dimension < 256; ++dimension) {
    const float query = bfloat16ToFloat(queryBits[rowOne + dimension]);
    score0 += query * key0[dimension] * (1.0f / 16.0f);
    score1 += query * key1[dimension] * (1.0f / 16.0f);
  }
  const float maximumScore = std::max(score0, score1);
  const float weight0 = std::exp(score0 - maximumScore);
  const float weight1 = std::exp(score1 - maximumScore);
  const float weightSum = weight0 + weight1;
  float rowOneError = 0.0f;
  for (uint32_t dimension = 0; dimension < 256; ++dimension) {
    const float expectedValue =
        (weight0 * value0[dimension] + weight1 * value1[dimension]) /
        weightSum;
    const float observed = bfloat16ToFloat(outputBits[rowOne + dimension]);
    rowOneError = std::max(rowOneError, std::abs(observed - expectedValue));
  }
  if (rowOneError > 0.03f)
    throw std::runtime_error("TQ4 causal attention mismatch: max abs " +
                             std::to_string(rowOneError));

  // The MPP path decodes one packed page into 16 KiB of threadgroup memory,
  // then shares it across all eight causal rows and six GQA heads.
  std::array<SplashTQ4VerifyAttentionParams, 4> attentionParams{};
  attentionParams[0] = {
      {0, 2, stride, 1, 1, 1, 1, 0}, layer, 0, 0, 0};
  id<MTLBuffer> attentionParamsBuffer =
      makeBuffer(device, sizeof(attentionParams));
  std::memcpy(attentionParamsBuffer.contents, attentionParams.data(),
              sizeof(attentionParams));
  constexpr uint64_t fusedRows =
      SPLASH_TARGET_VERIFY_ROWS * uint64_t{6};
  id<MTLBuffer> partials =
      makeBuffer(device, 4 * fusedRows * 256 * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, 4 * fusedRows * 2 * sizeof(float));
  id<MTLBuffer> mppOutput =
      makeBuffer(device, queryElements * sizeof(uint16_t));
  const auto mppSplit =
      pipeline(device, library, @"verify_attention_tq4_mpp_split");
  const auto mppReduce =
      pipeline(device, library, @"verify_attention_tq4_mpp_reduce");
  command = [queue commandBuffer];
  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:mppSplit];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:pages offset:0 atIndex:1];
  [encoder setBuffer:partials offset:0 atIndex:2];
  [encoder setBuffer:statistics offset:0 atIndex:3];
  for (NSUInteger index = 4; index <= 7; ++index)
    [encoder setBuffer:table offset:0 atIndex:index];
  [encoder setBuffer:attentionParamsBuffer offset:0 atIndex:8];
  [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:mppReduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:mppOutput offset:0 atIndex:2];
  [encoder setBuffer:attentionParamsBuffer offset:0 atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(4, fusedRows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  const auto *mppBits = static_cast<const uint16_t *>(mppOutput.contents);
  float mppError = 0.0f;
  for (uint32_t dimension = 0; dimension < 256; ++dimension) {
    const float expectedValue =
        (weight0 * value0[dimension] + weight1 * value1[dimension]) /
        weightSum;
    const float observed = bfloat16ToFloat(mppBits[rowOne + dimension]);
    mppError = std::max(mppError, std::abs(observed - expectedValue));
  }
  if (mppError > 0.03f)
    throw std::runtime_error("TQ4 MPP attention mismatch: max abs " +
                             std::to_string(mppError));

  // Prefill uses a different host ABI and dispatch geometry from verify. Keep
  // a numerical gate here so a correct decode kernel cannot conceal a broken
  // full-window construction path.
  SplashTQ4PrefillAttentionParams prefillParams{
      {0, 2, stride, 1, 1, 1, 0, 0}, layer, 0, 0, 0};
  id<MTLBuffer> prefillPartials =
      makeBuffer(device, 4 * fusedRows * 256 * sizeof(float));
  id<MTLBuffer> prefillStatistics =
      makeBuffer(device, 4 * fusedRows * 2 * sizeof(float));
  id<MTLBuffer> prefillOutput =
      makeBuffer(device, queryElements * sizeof(uint16_t));
  const auto prefillSplit =
      pipeline(device, library, @"prefill_attention_tq4_mpp_split");
  const auto prefillReduce =
      pipeline(device, library, @"prefill_attention_tq4_mpp_reduce");
  command = [queue commandBuffer];
  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:prefillSplit];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:pages offset:0 atIndex:1];
  [encoder setBuffer:prefillPartials offset:0 atIndex:2];
  [encoder setBuffer:prefillStatistics offset:0 atIndex:3];
  [encoder setBuffer:table offset:0 atIndex:4];
  [encoder setBytes:&prefillParams length:sizeof(prefillParams) atIndex:5];
  [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:prefillReduce];
  [encoder setBuffer:prefillPartials offset:0 atIndex:0];
  [encoder setBuffer:prefillStatistics offset:0 atIndex:1];
  [encoder setBuffer:prefillOutput offset:0 atIndex:2];
  [encoder setBytes:&prefillParams length:sizeof(prefillParams) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(4, fusedRows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  const auto *prefillBits =
      static_cast<const uint16_t *>(prefillOutput.contents);
  float prefillError = 0.0f;
  for (uint32_t dimension = 0; dimension < 256; ++dimension) {
    const float expectedValue =
        (weight0 * value0[dimension] + weight1 * value1[dimension]) /
        weightSum;
    const float observed = bfloat16ToFloat(prefillBits[rowOne + dimension]);
    prefillError =
        std::max(prefillError, std::abs(observed - expectedValue));
  }
  if (prefillError > 0.03f)
    throw std::runtime_error("TQ4 prefill attention mismatch: max abs " +
                             std::to_string(prefillError));

  // The production fast path materializes one KV head, consumes it, then
  // reuses the same planes.  Exercise the exact stage/barrier/attention
  // ordering and require bit-identical BF16 output to the page-native path.
  id<MTLBuffer> stagedKeys = makeBuffer(device, 2 * 256 * sizeof(uint16_t));
  id<MTLBuffer> stagedValues = makeBuffer(device, 2 * 256 * sizeof(uint16_t));
  id<MTLBuffer> stagedOutput =
      makeBuffer(device, queryElements * sizeof(uint16_t));
  const auto stageHead =
      pipeline(device, library, @"prefill_attention_tq4_stage_head");
  const auto stagedHead =
      pipeline(device, library, @"prefill_attention_tq4_staged_head");
  command = [queue commandBuffer];
  encoder = [command computeCommandEncoder];
  for (uint32_t head = 0; head < layout.kvHeads; ++head) {
    SplashTQ4StagedPrefillParams stagedParams{
        0, 2, stride, 2, layer, head, 0, 0};
    if (head)
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:stageHead];
    [encoder setBuffer:pages offset:0 atIndex:0];
    [encoder setBuffer:stagedKeys offset:0 atIndex:1];
    [encoder setBuffer:stagedValues offset:0 atIndex:2];
    [encoder setBuffer:table offset:0 atIndex:3];
    [encoder setBytes:&stagedParams length:sizeof(stagedParams) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:stagedHead];
    [encoder setBuffer:queries offset:0 atIndex:0];
    [encoder setBuffer:stagedKeys offset:0 atIndex:1];
    [encoder setBuffer:stagedValues offset:0 atIndex:2];
    [encoder setBuffer:stagedOutput offset:0 atIndex:3];
    [encoder setBytes:&stagedParams length:sizeof(stagedParams) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  const auto *stagedBits =
      static_cast<const uint16_t *>(stagedOutput.contents);
  for (uint32_t head = 0; head < layout.kvHeads; ++head) {
    for (uint32_t row = 0; row < 2; ++row) {
      const uint64_t begin = (uint64_t{head} * stride + row) * 6 * 256;
      for (uint32_t index = 0; index < 6 * 256; ++index) {
        if (stagedBits[begin + index] != prefillBits[begin + index])
          throw std::runtime_error(
              "TQ4 staged prefill differs from page-native BF16 output");
      }
    }
  }
  std::cout << "tq4_store_metal_test: exact MLX fixture match\n";
  std::cout << "tq4_attention_metal_test: page-native value max_abs="
            << maximumError << " causal_two_token_max_abs=" << rowOneError
            << '\n';
  std::cout << "tq4_mpp_attention_test: causal_two_token_max_abs="
            << mppError << '\n';
  std::cout << "tq4_prefill_attention_test: causal_two_token_max_abs="
            << prefillError << '\n';
  std::cout << "tq4_staged_prefill_test: exact page-native BF16 match\n";
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tq4_store_metal_test METALLIB\n";
    return 2;
  }
  try {
    run(argv[1]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "tq4 store test failed: " << error.what() << '\n';
    return 1;
  }
}
