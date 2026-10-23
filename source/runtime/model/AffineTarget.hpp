// Modified for JustFit: product identity and integration.
#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"
#include "model/HostEmbedding.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace justfit::model {

struct Qwen3_8Layout;
struct Qwen3_6MoeLayout;

namespace affine {
struct PlannedCheckpoint;
}

// Writes a native MLX affine checkpoint's target into images laid out as a
// JustFit package's target files, which QwenTargetLoader reads as affine
// weights (AffineTargetFormat). The checkpoint is planned once; each image is
// written into memory when it is opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_8Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
  ~AffineTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  [[nodiscard]] std::shared_ptr<HostEmbedding> hostEmbedding();
private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, head, embedding
};

// Every planned image of a layout, its sections at their offsets: the layers,
// the head, the embedding.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout);

} // namespace justfit::model
