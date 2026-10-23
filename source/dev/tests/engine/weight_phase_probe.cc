// Modified for JustFit: product identity and integration.
#include "model/WeightImages.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  assert(argc == 2);
  using namespace justfit;
  metal::MetalBackend backend(argv[1]);
  model::WeightImages images(backend, "phase-probe");
  bool failVision = false;
  auto plan = [&](std::string name, uint8_t value, bool vision) {
    return model::ImagePlan{std::move(name), "PHASETST", 0, 0, 32768,
      [&, value, vision](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
        std::memset(bytes.data(), value, bytes.size());
        auto header = model::weightFileHeader("PHASETST", 0, 0);
        std::memcpy(bytes.data(), header.data(), header.size());
        if (vision && failVision) throw std::runtime_error("injected partial write");
      }};
  };
  auto target = images.load(plan("target/model.bin", 17, false));
  auto targetView = target.section(16384, "body"); target.finish();
  auto vision = images.load(plan("vision/model.bin", 29, true));
  auto visionView = vision.section(16384, "body"); vision.finish();
  const uint64_t loadedBytes = backend.memoryStats().allocatedBytes;
  auto check = [](const metal::MetalBuffer &buffer, uint8_t value) {
    const auto *data = static_cast<const uint8_t *>(buffer.contents());
    assert(data);
    for (uint64_t i = 0; i < buffer.sizeBytes(); ++i) assert(data[i] == value);
  };
  check(targetView, 17); check(visionView, 29);
  assert(images.releaseComponent("vision/model.bin") == 32768);
  assert(images.releaseComponent("vision/model.bin") == 0);
  assert(!images.released() && !images.componentResident("vision/model.bin"));
  assert(!visionView.contents()); check(targetView, 17);
  assert(backend.memoryStats().allocatedBytes == loadedBytes - 32768);

  // Waking text from global idle preserves the previously released vision.
  images.release();
  bool refused = false;
  try { images.restoreComponent("vision/model.bin"); }
  catch (const std::logic_error &) { refused = true; }
  assert(refused && !targetView.contents() && !visionView.contents());
  while (!images.restore()) {}
  check(targetView, 17); assert(!visionView.contents());

  failVision = true; refused = false;
  try { images.restoreComponent("vision/model.bin"); }
  catch (const std::runtime_error &) { refused = true; }
  assert(refused && !visionView.contents());
  assert(!images.componentResident("vision/model.bin"));
  assert(backend.memoryStats().allocatedBytes == loadedBytes - 32768);
  failVision = false;
  images.restoreComponent("vision/model.bin");
  check(visionView, 29); check(targetView, 17);
  assert(backend.memoryStats().allocatedBytes == loadedBytes);
  images.release();
  while (!images.restore()) {}
  check(targetView, 17); check(visionView, 29);
  refused = false;
  try { images.releaseComponent("absent/model.bin"); }
  catch (const std::out_of_range &) { refused = true; }
  assert(refused);
  std::cout << "weight phase release/restore, stable views, idle wake and failed-write rollback PASS\n";
}
