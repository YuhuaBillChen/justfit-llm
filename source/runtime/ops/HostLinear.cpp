// Modified for JustFit: product identity and integration.
#include "HostLinear.hpp"

#include <arm_neon.h>
#include <arm_sme.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// File scope: clang rejects these keyword attributes inside a namespace.
// 32x32 fp32 block per 32-row block of the packed chunk.
__attribute__((target("+sme2"))) __arm_locally_streaming __arm_new("za") static void
justfit_host_multiply_panel(const float *packed, const float *panel, uint32_t inputSize,
                          uint32_t rowBlocks, float *blocks, uint32_t packedRowStride = 0) {
  const svbool_t p32 = svptrue_b32();
  for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
    svzero_za();
    const float *a = packed + size_t{rb} * (packedRowStride ? packedRowStride : inputSize) * 32;
    const float *b = panel;
    for (uint32_t k = 0; k < inputSize; ++k) {
      const svfloat32_t a0 = svld1_f32(p32, a), a1 = svld1_f32(p32, a + 16);
      const svfloat32_t b0 = svld1_f32(p32, b), b1 = svld1_f32(p32, b + 16);
      svmopa_za32_f32_m(0, p32, p32, a0, b0);
      svmopa_za32_f32_m(1, p32, p32, a0, b1);
      svmopa_za32_f32_m(2, p32, p32, a1, b0);
      svmopa_za32_f32_m(3, p32, p32, a1, b1);
      a += 32;
      b += 32;
    }
    float *out = blocks + size_t{rb} * 32 * 32;
    for (uint32_t r = 0; r < 16; ++r) {
      svst1_hor_za32(0, r, p32, out + r * 32);
      svst1_hor_za32(1, r, p32, out + r * 32 + 16);
      svst1_hor_za32(2, r, p32, out + (r + 16) * 32);
      svst1_hor_za32(3, r, p32, out + (r + 16) * 32 + 16);
    }
  }
}

// Preserve per64 SME Q4-code dot and group-affine rounding; stay in streaming
// mode across groups and apply scale/bias to 16 columns at once.
__attribute__((target("+sme2"))) __arm_locally_streaming __arm_new("za") static void
justfit_host_grouped_panel(const float *packed, const float *panel,
                          const float *scales, const float *biases,
                          const float *inputSums, uint32_t inputSize,
                          uint32_t rowBlocks, uint32_t row0, float *blocks) {
  const svbool_t p32 = svptrue_b32();
  const char *option=std::getenv("JUSTFIT_CPU_GPU_FUSION_ORDER");
  const bool fusion=option && std::strcmp(option,"1")==0;
  const uint32_t groups = inputSize / 64;
  alignas(64) float partial[32 * 32];
  for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
    float *out = blocks + size_t{rb} * 32 * 32;
    for (uint32_t r = 0; r < 32; ++r) {
      svst1_f32(p32, out + r * 32, svdup_n_f32(0.0f));
      svst1_f32(p32, out + r * 32 + 16, svdup_n_f32(0.0f));
    }
    for (uint32_t g = 0; g < groups; ++g) {
      svzero_za();
      const float *a = packed + (size_t{rb} * inputSize + g * 64) * 32;
      const float *b = panel + size_t{g} * 64 * 32;
      for (uint32_t k = 0; k < 64; ++k) {
        const svfloat32_t a0 = svld1_f32(p32, a), a1 = svld1_f32(p32, a + 16);
        const svfloat32_t b0 = svld1_f32(p32, b), b1 = svld1_f32(p32, b + 16);
        svmopa_za32_f32_m(0, p32, p32, a0, b0);
        svmopa_za32_f32_m(1, p32, p32, a0, b1);
        svmopa_za32_f32_m(2, p32, p32, a1, b0);
        svmopa_za32_f32_m(3, p32, p32, a1, b1);
        a += 32; b += 32;
      }
      for (uint32_t r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, p32, partial + r * 32);
        svst1_hor_za32(1, r, p32, partial + r * 32 + 16);
        svst1_hor_za32(2, r, p32, partial + (r + 16) * 32);
        svst1_hor_za32(3, r, p32, partial + (r + 16) * 32 + 16);
      }
      const svfloat32_t s0 = svld1_f32(p32, scales + g * 32);
      const svfloat32_t s1 = svld1_f32(p32, scales + g * 32 + 16);
      const svfloat32_t b0 = svld1_f32(p32, biases + g * 32);
      const svfloat32_t b1 = svld1_f32(p32, biases + g * 32 + 16);
      for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t row = row0 + rb * 32 + r;
        const float sum = inputSums[(size_t{row / 32} * groups + g) * 32 + row % 32];
        const svfloat32_t p0 = svld1_f32(p32, partial + r * 32);
        const svfloat32_t p1 = svld1_f32(p32, partial + r * 32 + 16);
        if(fusion) {
          const svfloat32_t next0=svmla_f32_m(p32,svld1_f32(p32,out+r*32),p0,s0);
          const svfloat32_t next1=svmla_f32_m(p32,svld1_f32(p32,out+r*32+16),p1,s1);
          svst1_f32(p32,out+r*32,svmla_n_f32_m(p32,next0,b0,sum));
          svst1_f32(p32,out+r*32+16,svmla_n_f32_m(p32,next1,b1,sum));
        } else {
        const svfloat32_t v0 = svmla_f32_m(p32, svmul_n_f32_m(p32, b0, sum), p0, s0);
        const svfloat32_t v1 = svmla_f32_m(p32, svmul_n_f32_m(p32, b1, sum), p1, s1);
        svst1_f32(p32, out + r * 32, svadd_f32_m(p32, svld1_f32(p32, out + r * 32), v0));
        svst1_f32(p32, out + r * 32 + 16, svadd_f32_m(p32, svld1_f32(p32, out + r * 32 + 16), v1));
        }
      }
    }
  }
}

// Reduce FP32 dequantized weights independently per64, then add each group
// in FP32 order. Weight-panel packing amortizes the affine operation.
__attribute__((target("+sme2"))) __arm_locally_streaming __arm_new("za") static void
justfit_host_weighted_panel(const float *packed, const float *panel,
                          const float *scales, const float *biases,
                          const float *inputSums, uint32_t inputSize,
                          uint32_t rowBlocks, uint32_t row0, float *blocks) {
  const svbool_t p32 = svptrue_b32();
  (void)scales;(void)biases;(void)inputSums;(void)row0;
  const uint32_t groups = inputSize / 64;
  alignas(64) float partial[32 * 32];
  for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
    float *out = blocks + size_t{rb} * 32 * 32;
    for (uint32_t r = 0; r < 32; ++r) {
      svst1_f32(p32, out + r * 32, svdup_n_f32(0.0f));
      svst1_f32(p32, out + r * 32 + 16, svdup_n_f32(0.0f));
    }
    for (uint32_t g = 0; g < groups; ++g) {
      svzero_za();
      const float *a = packed + (size_t{rb} * inputSize + g * 64) * 32;
      const float *b = panel + size_t{g} * 64 * 32;
      for (uint32_t k = 0; k < 64; ++k) {
        const svfloat32_t a0 = svld1_f32(p32, a), a1 = svld1_f32(p32, a + 16);
        const svfloat32_t b0 = svld1_f32(p32, b), b1 = svld1_f32(p32, b + 16);
        svmopa_za32_f32_m(0, p32, p32, a0, b0);
        svmopa_za32_f32_m(1, p32, p32, a0, b1);
        svmopa_za32_f32_m(2, p32, p32, a1, b0);
        svmopa_za32_f32_m(3, p32, p32, a1, b1);
        a += 32; b += 32;
      }
      for (uint32_t r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, p32, partial + r * 32);
        svst1_hor_za32(1, r, p32, partial + r * 32 + 16);
        svst1_hor_za32(2, r, p32, partial + (r + 16) * 32);
        svst1_hor_za32(3, r, p32, partial + (r + 16) * 32 + 16);
      }
      for(uint32_t r=0;r<32;++r){
        const svfloat32_t p0=svld1_f32(p32,partial+r*32),p1=svld1_f32(p32,partial+r*32+16);
        svst1_f32(p32,out+r*32,svadd_f32_m(p32,svld1_f32(p32,out+r*32),p0));
        svst1_f32(p32,out+r*32+16,svadd_f32_m(p32,svld1_f32(p32,out+r*32+16),p1));
      }
    }
  }
}

__attribute__((target("+sme2"))) __arm_locally_streaming __arm_new("za") static void
justfit_host_weighted_za_panel(const float *packed, const float *panel,
                          const float *scales, const float *biases,
                          const float *inputSums, uint32_t inputSize,
                          uint32_t rowBlocks, uint32_t row0, float *blocks) {
  const svbool_t p32 = svptrue_b32();
  (void)scales;(void)biases;(void)inputSums;(void)row0;
  const uint32_t groups = inputSize / 64;
  for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
    float *out = blocks + size_t{rb} * 32 * 32;
    for (uint32_t r = 0; r < 32; ++r) {
      svst1_f32(p32, out + r * 32, svdup_n_f32(0.0f));
      svst1_f32(p32, out + r * 32 + 16, svdup_n_f32(0.0f));
    }
    for (uint32_t g = 0; g < groups; ++g) {
      svzero_za();
      const float *a = packed + (size_t{rb} * inputSize + g * 64) * 32;
      const float *b = panel + size_t{g} * 64 * 32;
      for (uint32_t k = 0; k < 64; ++k) {
        const svfloat32_t a0 = svld1_f32(p32, a), a1 = svld1_f32(p32, a + 16);
        const svfloat32_t b0 = svld1_f32(p32, b), b1 = svld1_f32(p32, b + 16);
        svmopa_za32_f32_m(0, p32, p32, a0, b0);
        svmopa_za32_f32_m(1, p32, p32, a0, b1);
        svmopa_za32_f32_m(2, p32, p32, a1, b0);
        svmopa_za32_f32_m(3, p32, p32, a1, b1);
        a += 32; b += 32;
      }
      for(uint32_t r=0;r<32;++r){
        svfloat32_t p0,p1;
        if(r<16){p0=svread_hor_za32_f32_m(svdup_n_f32(0),p32,0,r);p1=svread_hor_za32_f32_m(svdup_n_f32(0),p32,1,r);}
        else{p0=svread_hor_za32_f32_m(svdup_n_f32(0),p32,2,r-16);p1=svread_hor_za32_f32_m(svdup_n_f32(0),p32,3,r-16);}
        svst1_f32(p32,out+r*32,svadd_f32_m(p32,svld1_f32(p32,out+r*32),p0));
        svst1_f32(p32,out+r*32+16,svadd_f32_m(p32,svld1_f32(p32,out+r*32+16),p1));
      }
    }
  }
}

namespace justfit::ops {
namespace {

constexpr uint32_t kBlockColumns = 32;
constexpr uint32_t kBlockRows = 32;
constexpr uint32_t kMaximumChunkRows = 2048;  // bounded by the scratch plane

inline float fromBf16(uint16_t value) {
  uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, 4);
  return result;
}
inline uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return uint16_t(bits >> 16);
}

// Bounded FP32 weight panel [K][32], with no intermediate BF16 rounding.
void dequantizePanel(const HostLinearTask &t, uint32_t n0, float *panel) {
  const uint32_t groups = t.inputSize / 64;
  const uint32_t planeGroups = (t.weightInputSize ? t.weightInputSize : t.inputSize) / 64;
  for (uint32_t c = 0; c < kBlockColumns; ++c) {
    const uint32_t n = n0 + c, tile = n / 256, column = n % 256;
    for (uint32_t g = 0; g < groups; ++g) {
      const size_t parameter = (size_t{tile} * planeGroups + g) * 256 + column;
      const float32x4_t scale = vdupq_n_f32(fromBf16(t.scales[parameter]));
      const float32x4_t bias = vdupq_n_f32(fromBf16(t.biases[parameter]));
      const uint8_t *codes = t.weights + parameter * 32;
      for (uint32_t j = 0; j < 32; j += 8) {
        const uint8x8_t raw = vld1_u8(codes + j);
        const uint8x8x2_t zipped =
            vzip_u8(vand_u8(raw, vdup_n_u8(15)), vshr_n_u8(raw, 4));
        const uint8x16_t u = vcombine_u8(zipped.val[0], zipped.val[1]);
        const uint16x8_t u0 = vmovl_u8(vget_low_u8(u)), u1 = vmovl_u8(vget_high_u8(u));
        const float32x4_t f[4] = {vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))),
                                  vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0))),
                                  vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))),
                                  vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1)))};
        for (uint32_t q = 0; q < 4; ++q) {
          const float32x4_t w = vfmaq_f32(bias, f[q], scale);
          for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = g * 64 + 2 * j + q * 4 + e;
            panel[size_t{k} * kBlockColumns + c] = w[e];
          }
        }
      }
    }
  }
}

// Same FP32 affine FMA as the baseline, four columns transposed into contiguous stores.
void dequantizePanelVector(const HostLinearTask &t, uint32_t n0, float *panel) {
  const uint32_t groups = t.inputSize / 64;
  const uint32_t planeGroups = (t.weightInputSize ? t.weightInputSize : t.inputSize) / 64;
  for (uint32_t c = 0; c < kBlockColumns; c += 4) {
    for (uint32_t g = 0; g < groups; ++g) {
      const uint8_t *codes[4];
      float32x4_t scales[4], biases[4];
      for (uint32_t col = 0; col < 4; ++col) {
        const uint32_t n = n0 + c + col;
        const size_t parameter = (size_t{n / 256} * planeGroups + g) * 256 + n % 256;
        codes[col] = t.weights + parameter * 32;
        scales[col] = vdupq_n_f32(fromBf16(t.scales[parameter]));
        biases[col] = vdupq_n_f32(fromBf16(t.biases[parameter]));
      }
      for (uint32_t j = 0; j < 32; j += 8) {
        float32x4_t values[4][4];
        for (uint32_t col = 0; col < 4; ++col) {
          const uint8x8_t raw = vld1_u8(codes[col] + j);
          const uint8x8x2_t zipped = vzip_u8(vand_u8(raw, vdup_n_u8(15)), vshr_n_u8(raw, 4));
          const uint8x16_t u = vcombine_u8(zipped.val[0], zipped.val[1]);
          const uint16x8_t u0 = vmovl_u8(vget_low_u8(u)), u1 = vmovl_u8(vget_high_u8(u));
          const float32x4_t f[4] = {
              vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))),
              vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0))),
              vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))),
              vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1)))};
          for (uint32_t q = 0; q < 4; ++q)
            values[col][q] = vfmaq_f32(biases[col], f[q], scales[col]);
        }
        for (uint32_t q = 0; q < 4; ++q) {
          const float32x4x2_t a = vtrnq_f32(values[0][q], values[1][q]);
          const float32x4x2_t b = vtrnq_f32(values[2][q], values[3][q]);
          const float32x4_t rows[4] = {
              vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0])),
              vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1])),
              vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])),
              vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1]))};
          for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = g * 64 + 2 * j + q * 4 + e;
            vst1q_f32(panel + size_t{k} * kBlockColumns + c, rows[e]);
          }
        }
      }
    }
  }
}

void codePanel(const HostLinearTask &t, uint32_t n0, float *panel, float *packedScales, float *packedBiases) {
  const uint32_t planeGroups = (t.weightInputSize ? t.weightInputSize : t.inputSize) / 64;
  for (uint32_t c = 0; c < kBlockColumns; ++c) {
    const uint32_t n = n0 + c, tile = n / 256, column = n % 256;
    for (uint32_t g = 0; g < t.inputSize / 64; ++g) {
      const size_t parameter = (size_t{tile} * planeGroups + g) * 256 + column;
      packedScales[size_t{g}*32+c]=fromBf16(t.scales[parameter]);
      packedBiases[size_t{g}*32+c]=fromBf16(t.biases[parameter]);
      const uint8_t *codes = t.weights + parameter * 32;
      for (uint32_t j = 0; j < 32; j += 8) {
        const uint8x8_t raw = vld1_u8(codes + j);
        const uint8x8x2_t zipped =
            vzip_u8(vand_u8(raw, vdup_n_u8(15)), vshr_n_u8(raw, 4));
        const uint8x16_t u = vcombine_u8(zipped.val[0], zipped.val[1]);
        const uint16x8_t u0 = vmovl_u8(vget_low_u8(u)), u1 = vmovl_u8(vget_high_u8(u));
        const float32x4_t f[4] = {vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))),
                                  vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0))),
                                  vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))),
                                  vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1)))};
        for (uint32_t q = 0; q < 4; ++q) {
          const float32x4_t w = f[q];
          for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = g * 64 + 2 * j + q * 4 + e;
            panel[size_t{k} * kBlockColumns + c] = w[e];
          }
        }
      }
    }
  }
}

void finishBlock(const HostLinearTask &t, const float *block, uint32_t row0, uint32_t n0) {
  const char *deferOption=std::getenv("JUSTFIT_CPU_GPU_UP_EPILOGUE");
  const bool deferUp=deferOption && std::strcmp(deferOption,"1")==0;
  for (uint32_t r = 0; r < kBlockRows; ++r) {
    const size_t base = size_t{row0 + r} * t.outputSize + n0;
    for (uint32_t c = 0; c < kBlockColumns; ++c) {
      float value = fromBf16(toBf16(block[r * 32 + c]));
      if (t.epilogue == HostEpilogue::UpSilu && !deferUp) {
        const float gate = fromBf16(t.auxiliary[base + c]);
        value = gate / (1.0f + std::exp2(-1.44269504089f * gate)) * value;
      } else if (t.epilogue == HostEpilogue::Residual) {
        value += fromBf16(t.auxiliary[base + c]);
      }
      t.output[base + c] = toBf16(value);
    }
  }
}

class Pool final {
public:
  explicit Pool(uint32_t threads) {
    for (uint32_t i = 0; i < threads; ++i)
      workers_.emplace_back([this, i] { loop(i); });
  }
  ~Pool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto &worker : workers_) worker.join();
  }
  [[nodiscard]] uint32_t size() const noexcept { return uint32_t(workers_.size()); }
  // Runs job(worker) on every worker and returns when all have finished.
  void run(const std::function<void(uint32_t)> &job) {
    std::unique_lock lock(mutex_);
    job_ = &job;
    pending_ = size();
    ++generation_;
    wake_.notify_all();
    done_.wait(lock, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

private:
  void loop(uint32_t index) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(uint32_t)> *job;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job = job_;
      }
      (*job)(index);
      std::lock_guard lock(mutex_);
      if (--pending_ == 0) done_.notify_one();
    }
  }
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(uint32_t)> *job_ = nullptr;
  uint32_t pending_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

Pool &pool() {
  static Pool instance([] {
    const char *value = std::getenv("JUSTFIT_CPU_THREADS");
    const int threads = value ? std::atoi(value) : 8;
    return uint32_t(std::clamp(threads, 1, 16));
  }());
  return instance;
}

// Barrier for one run() of the pool.
class Barrier final {
public:
  explicit Barrier(uint32_t count) : count_(count) {}
  void arrive() {
    std::unique_lock lock(mutex_);
    const uint64_t phase = phase_;
    if (++arrived_ == count_) {
      arrived_ = 0;
      ++phase_;
      wake_.notify_all();
    } else {
      wake_.wait(lock, [&] { return phase_ != phase; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable wake_;
  uint32_t count_, arrived_ = 0;
  uint64_t phase_ = 0;
};

} // namespace

void runHostLinear(const HostLinearTask &t) {
  if (!t.rows || t.rows % kBlockRows || t.firstColumn % 64 ||
      t.firstColumn >= t.outputSize || t.inputSize % 64)
    throw std::invalid_argument("invalid host linear slice");
  const char *option=std::getenv("JUSTFIT_CPU_GROUP64");
  const bool grouped=option && std::strcmp(option,"1")==0;
  if(grouped && !t.inputSums) throw std::invalid_argument("Grouped CPU needs GPU input sums");
  const char *weightedOption=std::getenv("JUSTFIT_CPU_GROUP64_WEIGHTED");
  const bool weighted=grouped && weightedOption && std::strcmp(weightedOption,"1")==0;
  const char *sumOrder=std::getenv("JUSTFIT_CPU_GPU_SUM_ORDER");
  const bool adjacentSums=sumOrder && std::strcmp(sumOrder,"1")==0;
  const char *deferOption=std::getenv("JUSTFIT_CPU_GPU_UP_EPILOGUE");
  const bool deferUp=deferOption && std::strcmp(deferOption,"1")==0;
  const uint64_t rowBytes = uint64_t{t.inputSize} * sizeof(float);
  uint32_t chunk = uint32_t(std::min<uint64_t>(kMaximumChunkRows, t.scratchBytes / rowBytes));
  chunk = std::min(chunk - chunk % kBlockRows, t.rows);
  if (!chunk) throw std::invalid_argument("host linear scratch is too small");
  Pool &workers = pool();
  const uint32_t threads = workers.size();
  // Columns are assigned in 64-column pairs of blocks so one worker owns each
  // output-sum group.
  const uint32_t pairs = (t.outputSize - t.firstColumn) / 64;
  Barrier barrier(threads);
  workers.run([&](uint32_t worker) {
    std::vector<float> panel(size_t{t.inputSize} * kBlockColumns);
    std::vector<float> blocks(size_t{chunk} * kBlockColumns);
    std::vector<float> packedScales(grouped && !weighted ? size_t{t.inputSize/64} * 32 : 0);
    std::vector<float> packedBiases(packedScales.size());
    const uint32_t pairBegin = pairs * worker / threads;
    const uint32_t pairEnd = pairs * (worker + 1) / threads;
    for (uint32_t row0 = 0; row0 < t.rows; row0 += chunk) {
      const uint32_t rows = std::min(chunk, t.rows - row0);
      // Pack exact BF16-to-FP32 values as [rows/32][K][32].
      const uint32_t rowBlocks = rows / kBlockRows;
      for (uint32_t rb = worker; rb < rowBlocks; rb += threads) {
        for (uint32_t r = 0; r < kBlockRows; ++r) {
          const uint16_t *source =
              t.input + size_t{row0 + rb * kBlockRows + r} * t.inputSize;
          float *destination =
              reinterpret_cast<float *>(t.scratch) + size_t{rb} * t.inputSize * 32 + r;
          for (uint32_t k = 0; k < t.inputSize; ++k)
            destination[size_t{k} * 32] = fromBf16(source[k]);
        }
      }
      barrier.arrive();
      for (uint32_t pair = pairBegin; pair < pairEnd; ++pair) {
        for (uint32_t half = 0; half < 2; ++half) {
          const uint32_t n0 = t.firstColumn + pair * 64 + half * 32;
          if (!grouped) {
            dequantizePanel(t, n0, panel.data());
            justfit_host_multiply_panel(reinterpret_cast<const float *>(t.scratch), panel.data(), t.inputSize, rowBlocks, blocks.data());
          } else {
            if(weighted){
              const char *vectorPanel=std::getenv("JUSTFIT_CPU_VECTOR_PANEL");
              if(vectorPanel && std::strcmp(vectorPanel,"1")==0) dequantizePanelVector(t,n0,panel.data());
              else dequantizePanel(t,n0,panel.data());
              const char *direct=std::getenv("JUSTFIT_CPU_DIRECT_ZA");
              if(direct && std::strcmp(direct,"1")==0)
                justfit_host_weighted_za_panel(reinterpret_cast<const float *>(t.scratch),panel.data(),
                  nullptr,nullptr,t.inputSums,t.inputSize,rowBlocks,row0,blocks.data());
              else
              justfit_host_weighted_panel(reinterpret_cast<const float *>(t.scratch),panel.data(),
                  nullptr,nullptr,t.inputSums,t.inputSize,rowBlocks,row0,blocks.data());
            }else{
              codePanel(t,n0,panel.data(),packedScales.data(),packedBiases.data());
              justfit_host_grouped_panel(reinterpret_cast<const float *>(t.scratch),panel.data(),
                  packedScales.data(),packedBiases.data(),t.inputSums,t.inputSize,rowBlocks,row0,blocks.data());
            }
          }
          for (uint32_t rb = 0; rb < rowBlocks; ++rb)
            finishBlock(t, blocks.data() + size_t{rb} * kBlockRows * kBlockColumns,
                        row0 + rb * kBlockRows, n0);
        }
        if (t.epilogue == HostEpilogue::UpSilu && t.outputSums && !deferUp) {
          const uint32_t group = (t.firstColumn + pair * 64) / 64;
          const uint32_t outputGroups = t.outputSize / 64;
          for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t row = row0 + r;
            const uint16_t *values = t.output + size_t{row} * t.outputSize + group * 64;
            float sum = 0.0f;
            if (!grouped) {
              for(uint32_t c=0;c<64;++c)sum+=fromBf16(values[c]);
            } else {
              float lanes[32];for(uint32_t c=0;c<32;++c)lanes[c]=fromBf16(values[c])+fromBf16(values[c+32]);
              if(adjacentSums) {
                for(uint32_t step=1;step<32;step*=2)for(uint32_t c=0;c<32;c+=2*step)lanes[c]+=lanes[c+step];
              } else {
              for(uint32_t step=16;step;step/=2)for(uint32_t c=0;c<step;++c)lanes[c]+=lanes[c+step];
              }
              sum=lanes[0];
            }
            t.outputSums[(size_t{row / 32} * outputGroups + group) * 32 + row % 32] = sum;
          }
        }
      }
      // The next chunk repacks the shared scratch.
      barrier.arrive();
    }
  });
}

} // namespace justfit::ops
