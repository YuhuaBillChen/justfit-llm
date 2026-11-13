// Modified for JustFit: preserve the original MLX model's FP32 decay transform.
#include "model/AffinePreparation.hpp"
#include "model/WeightStore.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace justfit::model::affine {
namespace {
struct DecayKernel {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
};
const DecayKernel &decayKernel() {
  static const DecayKernel kernel = [] {
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw WeightStoreError("Metal device unavailable for source GDN decay");
      // This is the operation used by MLX's Exp functor. Neither host exp(double)
      // nor Metal fast::exp reproduces the 2.3.1 packed model's FP32 values.
      NSString *source = @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "kernel void justfit_source_decay(device const float *x [[buffer(0)]],"
          "device float *y [[buffer(1)]], uint i [[thread_position_in_grid]]) {"
          "y[i] = -metal::precise::exp(x[i]);}\n";
      MTLCompileOptions *options = [MTLCompileOptions new];
      options.mathMode = MTLMathModeFast;
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
      if (!library) throw WeightStoreError("cannot compile source GDN decay operation");
      id<MTLFunction> function = [library newFunctionWithName:@"justfit_source_decay"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!pipeline || !queue) throw WeightStoreError("cannot create source GDN decay pipeline");
      return DecayKernel{device, queue, pipeline};
    }
  }();
  return kernel;
}
}

void writeAffineDecay(std::span<const float> logarithms, std::span<float> destination) {
  if (logarithms.size() != destination.size() || logarithms.size() > std::numeric_limits<uint32_t>::max())
    throw WeightStoreError("invalid source GDN decay dimensions");
  if (logarithms.empty()) return;
  @autoreleasepool {
    const auto &kernel = decayKernel();
    const size_t bytes = logarithms.size_bytes();
    id<MTLBuffer> input = [kernel.device newBufferWithBytes:logarithms.data()
                                                  length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [kernel.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [kernel.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!input || !output || !command || !encoder)
      throw WeightStoreError("cannot allocate source GDN decay operation");
    [encoder setComputePipelineState:kernel.pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(logarithms.size(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, kernel.pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw WeightStoreError("source GDN decay operation failed");
    std::memcpy(destination.data(), output.contents, bytes);
  }
}
}
