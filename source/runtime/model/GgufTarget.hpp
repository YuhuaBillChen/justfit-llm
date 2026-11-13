// Modified for JustFit: product identity and integration.
#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/WeightImages.hpp"

namespace justfit::model {

// One GGUF or all parts of a split GGUF in validated filename order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once.
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::vector<std::filesystem::path> &paths,
                   const QwenTargetDimensions &geometry);
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // The GGUF and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::vector<std::filesystem::path> &paths) : source(paths) {}
    OpenGguf source;
    std::vector<gguf::Image> images; // layers, head, embedding
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  std::optional<GgufRotation> rotation_;
};

} // namespace justfit::model
