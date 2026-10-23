// Modified for JustFit: product identity and integration.
#include "ops/VerifyShape.hpp"
#pragma once

// Shared Qwen prefill/decode scratch layouts and allocation owners.

#include "model/DFlashDraft.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenTarget.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace justfit::model {

inline constexpr uint32_t kLaneCount = ExecutionLimits::maximumBatchWidth;
inline constexpr uint32_t kDecodeRows = ExecutionLimits::targetVerifyRows;
// Per-lane sampling uniforms handed to the sampler each cycle, in the
// layout of metal/abi/Sampling.h.
inline constexpr uint32_t kSamplingUniformCount = JUSTFIT_SAMPLING_UNIFORMS;
inline constexpr uint32_t kDraftProposalTokens = ExecutionLimits::draftProposalTokens;
inline constexpr uint32_t kPrefillRows = ExecutionLimits::prefillTokenBudget;
inline constexpr uint32_t kTileRows = kv::kPageTokens;
inline constexpr uint32_t kRaggedAttentionRows =
    kPrefillRows + kLaneCount * (kTileRows - 1);
inline constexpr uint32_t kDraftCacheStride = ExecutionLimits::draftContextTokens;
inline constexpr uint32_t kMaximumPageTableEntries =
    (kv::kMaximumPhysicalTokens + kv::kPageTokens - 1) / kv::kPageTokens;

struct RuntimeGeometry final {
  QwenTargetGeometry target;
  DFlashDraftLayout draft;

  [[nodiscard]] static RuntimeGeometry from(const LoadedModel &model,
                                            kv::Format format) {
    RuntimeGeometry result;
    result.target = std::visit(
        [](const auto &weights) { return qwenTargetGeometry(weights); },
        model.target);
    result.target.kvLayout = model.targetKvLayout(format);
    result.draft = model.draft.layout;
    if (!result.target.valid() || !result.draft.stateLayout().valid() ||
        result.target.hiddenSize != result.draft.hiddenSize ||
        result.target.vocabularySize != result.draft.vocabularySize ||
        result.target.capturedHiddenSize() != result.draft.targetHiddenSize) {
      throw std::invalid_argument("invalid model runtime geometry");
    }
    return result;
  }

  [[nodiscard]] uint32_t draftRotaryPairs() const noexcept {
    return draft.attentionHeadDimension / 2;
  }
  [[nodiscard]] uint32_t maskWords() const noexcept {
    return maskWordsPerToken(target.vocabularySize);
  }
  [[nodiscard]] uint32_t projectionSumsWidth() const noexcept {
    uint32_t maximumInput = std::max(
        {target.hiddenSize, target.attentionWidth,
         target.capturedHiddenSize(), target.ffnScratchWidth(),
         draft.targetHiddenSize, draft.hiddenSize,
         draft.intermediateSize});
    return (maximumInput + kQ4GroupElements - 1) / kQ4GroupElements;
  }
};

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept {
  return elements * sizeof(T);
}

enum class PrefillTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  Captured,
  GdnPacked,
  GdnQueries,
  GdnKeys,
  GdnValues,
  GdnDecay,
  GdnBeta,
  Recurrent,
  GdnHidden,
  GdnOutput,
  GateIntermediate,
  Intermediate,
  FullPacked,
  FullQueries,
  FullAttention,
  AttentionPartials,
  AttentionStatistics,
  AttentionHidden,
  AttentionOutput,
  ProjectionSums,
  DownProjectionSums,
  TargetPositions,
  DraftPositions,
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextKv,
  DraftRopeCos,
  DraftRopeSin,
  ChunkKeys,
  ChunkValues,
  TQ4StagedKeys,
  TQ4StagedValues,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  LinearPartials,
  LinearCounters,
  // The rotated input of a rotated projection (ops::LinearScratch::rotated).
  LinearRotated,
  A8Codes,
  A8Scale,
  A8Sums,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// The arena tensor of MoE scratch field `field` (ops::kMoeScratchFields).
template <class Tensor>
constexpr Tensor moeScratchTensor(size_t field) noexcept {
  return static_cast<Tensor>(static_cast<uint32_t>(Tensor::MoeScratch) + field);
}

// Scratch lives only within its mixer or FFN phase. Keep the ablation explicit
// while the new upstream ANE/graph paths undergo qualification.
inline bool prefillAliasEnabled(kv::Format format) {
  const char *value = std::getenv("JUSTFIT_PREFILL_ALIAS");
  return format == kv::Format::TQ4 && value && std::strcmp(value, "1") == 0;
}

inline bool stagedTQ4Enabled(kv::Format format) {
  const char *value = std::getenv("JUSTFIT_STAGED_PREFILL");
  return format==kv::Format::TQ4 && value && std::strcmp(value,"1")==0;
}
struct PrefillArenaLayout final {
  std::array<uint64_t, prefillTensorCount> offsets{};
  uint64_t bytes = 0;
};

inline PrefillArenaLayout prefillArenaLayout(
    const std::array<uint64_t, prefillTensorCount> &sizes, bool alias) {
  using T = PrefillTensor;
  const char *phaseOption = std::getenv("JUSTFIT_PREFILL_PHASE_REUSE");
  const bool phaseReuse = alias && phaseOption && std::strcmp(phaseOption, "1") == 0;
  // Each pair executes in separate phases. Mixer residuals remain separate
  // from FFN scratch because the down projection still reads the residual.
  constexpr std::array pairs{
      std::array{T::GdnQueries, T::FullQueries},
      std::array{T::GdnKeys, T::ChunkKeys},
      std::array{T::GdnValues, T::FullAttention},
      std::array{T::Recurrent, T::AttentionHidden},
      std::array{T::GdnOutput, T::AttentionOutput},
      std::array{T::GdnPacked, T::GateIntermediate},
      std::array{T::GdnHidden, T::Intermediate},
      std::array{T::TQ4StagedKeys,T::GateIntermediate},
      std::array{T::TQ4StagedValues,T::Intermediate},
      // Draft context runs after all target layers. Captured target states
      // and the final target hidden rows remain independently allocated.
      std::array{T::ContextProjected, T::GateIntermediate},
      std::array{T::ContextHidden, T::Intermediate},
      std::array{T::ContextKv, T::FullPacked},
      // Input normalization is consumed before recurrence/attention gating.
      std::array{T::Normalized, T::AttentionHidden},
      // All sequences finish KV attention before the output projection.
      std::array{T::ChunkValues, T::AttentionOutput}};
  const auto activePair = [&](const auto &pair) {
    return phaseReuse || (pair[0] != T::Normalized && pair[0] != T::ChunkValues);
  };
  PrefillArenaLayout result;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto tensor = static_cast<T>(i);
    if (alias && std::any_of(pairs.begin(), pairs.end(),
                            [&](const auto &pair) { return activePair(pair) && tensor == pair[0]; }))
      continue;
    result.offsets[i] = result.bytes;
    uint64_t bytes = sizes[i];
    if (alias)
      for (const auto &pair : pairs)
        if (activePair(pair) && tensor == pair[1]) {
          const auto other = static_cast<size_t>(pair[0]);
          result.offsets[other] = result.bytes;
          bytes = std::max(bytes, sizes[other]);
        }
    result.bytes = checkedAdd(result.bytes, alignUp(bytes), "prefill arena");
  }
  return result;
}

// Sizes depend on the geometry and the device's operator plans.
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators,
                   uint32_t stagingTokens = 262144);
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators,
                                           uint32_t stagingTokens = 262144);

class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators, uint32_t stagingTokens = 262144)
      : stagingTokens_(stagingTokens),
        bytes_(plannedPrefillBytes(geometry, operators, stagingTokens)) {
    const auto sizes = prefillTensorBytes(geometry, operators, stagingTokens);
    base_ = backend.allocateBuffer(bytes_, metal::BufferStorage::Shared,
                                   "qwen-shared-prefill");
    const auto layout = prefillArenaLayout(sizes, prefillAliasEnabled(geometry.target.kvLayout.format));
    for (uint32_t index = 0; index < sizes.size(); ++index)
      if (sizes[index])
        tensors_[index] = backend.view(base_, layout.offsets[index], sizes[index]);
    if (layout.bytes != bytes_)
      throw std::logic_error("prefill arena mismatch");
    auto *target = static_cast<float *>(
        get(PrefillTensor::TargetInverseFrequencies).contents());
    auto *draft = static_cast<float *>(
        get(PrefillTensor::DraftInverseFrequencies).contents());
    if (!target || !draft)
      throw std::logic_error("RoPE frequencies are not CPU-visible");
    for (uint32_t dim = 0; dim < geometry.target.rotaryPairs; ++dim) {
      target[dim] =
          std::pow(geometry.target.rotaryTheta,
                   -static_cast<float>(dim) / geometry.target.rotaryPairs);
    }
    for (uint32_t dim = 0; dim < geometry.draftRotaryPairs(); ++dim) {
      draft[dim] = std::pow(geometry.draft.rotaryTheta,
                            -static_cast<float>(dim) / geometry.draftRotaryPairs());
    }
    // Split projections return their counters to zero; they start there.
    if (const metal::MetalBuffer counters = get(PrefillTensor::LinearCounters))
      std::memset(counters.contents(), 0, counters.sizeBytes());
  }

  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor) const {
    return tensors_[static_cast<uint32_t>(tensor)];
  }
  [[nodiscard]] ops::MoeScratch moeScratch() const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          get(moeScratchTensor<PrefillTensor>(field));
    return scratch;
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] uint32_t stagingTokens() const noexcept { return stagingTokens_; }

private:
  uint32_t stagingTokens_ = 262144;
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  uint64_t bytes_ = 0;
};

enum class DecodeTensor : uint32_t {
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  GdnHidden,
  GdnOutput,
  Intermediate,
  FullPacked,
  FullQueries,
  AttentionPartials,
  AttentionStatistics,
  FullAttention,
  AttentionHidden,
  AttentionOutput,
  Positions,
  DraftPositions,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextKv,
  CapturedTargetHidden,
  DraftQueryKeys,
  DraftQueryValues,
  DraftRopeCos,
  DraftRopeSin,
  FinalHidden,
  Logits,
  ArgmaxValues,
  ArgmaxIndices,
  TargetPartialMasses,
  TargetVocabularyRows,
  TargetVocabularyRanges,
  TargetVocabularyArrivals,
  SamplingUniforms,
  ConstraintMasks,
  OutputTokens,
  RetainedCount,
  AcceptedCount,
  DraftInputTokens,
  DraftHidden0,
  DraftHidden1,
  DraftNormalized,
  DraftDynamic,
  DraftConvolved,
  DraftProposalQkv,
  DraftAttention,
  DraftProjected,
  DraftResidual,
  DraftIntermediate,
  DraftFinalHidden,
  SelectorHidden,
  Candidates,
  Unary,
  TopPartialIds,
  TopPartialValues,
  ProposalProbs,
  ProposedTokens,
  PageTable,
  // Indexed by state lane, like PageTable: a penalized request's penalty
  // words (ops::Sampling::rebuildPenaltyWords).
  PenaltyState,
  VerifyPackedBase,
  VerifyMixedBase,
  VerifyDecayBase,
  VerifyBetaBase,
  ChunkKeysBase,
  ChunkValuesBase,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  VerifyProposals,
  VerifyCandidates,
  VerifyProbabilities,
  Count,
};

constexpr uint32_t decodeTensorCount =
    static_cast<uint32_t>(DecodeTensor::Count);

constexpr bool isGdnLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::VerifyPackedBase ||
         tensor == DecodeTensor::VerifyMixedBase ||
         tensor == DecodeTensor::VerifyDecayBase ||
         tensor == DecodeTensor::VerifyBetaBase;
}

constexpr bool isAttentionLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::ChunkKeysBase ||
         tensor == DecodeTensor::ChunkValuesBase;
}

constexpr bool isLayerMajorTensor(DecodeTensor tensor) noexcept {
  return isGdnLayerTensor(tensor) || isAttentionLayerTensor(tensor);
}

[[nodiscard]] std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators);
// A decode tensor is one M32 allocation.  B1/B2/B3/B4 are prefixes
// containing 8/16/24/32 rows. Lanes are never separated by arena-alignment
// holes; only whole tensor boundaries are aligned.
[[nodiscard]] uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                                            const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                                          const ops::ExecutionPlans &operators);

inline bool targetRowTensor(DecodeTensor tensor) {
 switch(tensor) {
 case DecodeTensor::Hidden0:
 case DecodeTensor::Hidden1:
 case DecodeTensor::InputTokens:
 case DecodeTensor::Normalized:
 case DecodeTensor::GdnHidden:
 case DecodeTensor::GdnOutput:
 case DecodeTensor::Intermediate:
 case DecodeTensor::FullPacked:
 case DecodeTensor::AttentionPartials:
 case DecodeTensor::AttentionStatistics:
 case DecodeTensor::AttentionHidden:
 case DecodeTensor::AttentionOutput:
 case DecodeTensor::Positions:
 case DecodeTensor::DraftPositions:
 case DecodeTensor::RopeCos:
 case DecodeTensor::RopeSin:
 case DecodeTensor::ContextProjected:
 case DecodeTensor::ContextHidden:
 case DecodeTensor::ContextKv:
 case DecodeTensor::CapturedTargetHidden:
 case DecodeTensor::DraftRopeCos:
 case DecodeTensor::DraftRopeSin:
 case DecodeTensor::FinalHidden:
 case DecodeTensor::Logits:
 case DecodeTensor::ArgmaxValues:
 case DecodeTensor::ArgmaxIndices:
 case DecodeTensor::TargetPartialMasses:
 case DecodeTensor::TargetVocabularyRows:
 case DecodeTensor::TargetVocabularyRanges:
 case DecodeTensor::TargetVocabularyArrivals:
 case DecodeTensor::OutputTokens:
 return true;
 default:return false;
 }
}
class DecodeArena final {
public:
  DecodeArena(metal::MetalBackend &backend, RuntimeGeometry geometry,
               const ops::ExecutionPlans &operators)
      : backend_(backend), geometry_(std::move(geometry)) {
    const uint64_t baseBytes = decodeArenaBaseBytes(geometry_, operators);
    auto sizes = decodeTensorBytes(geometry_, operators);
    base_ = backend_.allocateBuffer(baseBytes, metal::BufferStorage::Shared,
                                    "qwen-shared-decode");
    uint64_t cursor = 0;
    for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
      uint64_t stride = sizes[tensor];
      offsets_[tensor] = cursor;
      sizes_[tensor] = sizes[tensor];
      const DecodeTensor kind = static_cast<DecodeTensor>(tensor);
      if (sizes[tensor] && !isLayerMajorTensor(kind)) {
        for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
          tensors_[lane][tensor] = backend_.view(
              base_, cursor + uint64_t{lane} * stride, sizes[tensor]);
        }
      }
      cursor +=
          alignUp(checkedMultiply(stride, kLaneCount, "decode tensor"));
    }
    if (cursor != baseBytes)
      throw std::logic_error("decode arena mismatch");
    // Sampled rows' draws return their arrival counts to zero; they start
    // there.
    const metal::MetalBuffer arrivals =
        batchSlice(DecodeTensor::TargetVocabularyArrivals, kLaneCount);
    std::memset(arrivals.contents(), 0, arrivals.sizeBytes());

    const uint64_t denseScratchBytes = gateScratchBytes(geometry_, operators);
    if (denseScratchBytes) {
      gateScratch_ = backend_.allocateBuffer(
          denseScratchBytes, metal::BufferStorage::Private, "qwen-gate-scratch");
    }
    // Each field exists only when some plan uses it (split-only plans have
    // partials and counters but no activation table).
    const auto linearSize = linearScratchSize(geometry_, operators);
    const auto allocate = [&](uint64_t bytes, metal::BufferStorage storage, const char *label) {
      return bytes ? backend_.allocateBuffer(bytes, storage, label) : metal::MetalBuffer{};
    };
    linearScratch_.input = allocate(linearSize.input, metal::BufferStorage::Private, "q4-input");
    linearScratch_.sums = allocate(linearSize.sums, metal::BufferStorage::Private, "q4-sums");
    linearScratch_.partials =
        allocate(linearSize.partials, metal::BufferStorage::Private, "q4-partials");
    linearScratch_.counters =
        allocate(linearSize.counters, metal::BufferStorage::Shared, "q4-counters");
    linearScratch_.rotated = allocate(linearSize.rotated, metal::BufferStorage::Private, "linear-rotated");
    linearScratch_.a8Codes = allocate(linearSize.a8Codes, metal::BufferStorage::Private, "verify-a8-codes");
    linearScratch_.a8Scale = allocate(linearSize.a8Scale, metal::BufferStorage::Private, "verify-a8-scales");
    linearScratch_.a8Sums = allocate(linearSize.a8Sums, metal::BufferStorage::Private, "verify-a8-sums");
    if (linearSize.counters)
      std::memset(linearScratch_.counters.contents(), 0, linearSize.counters);
    bytes_ = plannedDecodeBytes(geometry_, operators);
  }

  uint64_t stride(DecodeTensor tensor) const {
    const auto bytes = sizes_[static_cast<uint32_t>(tensor)];
    if (tensor == DecodeTensor::VerifyProposals) return (ops::activeVerifyRows()-1)*sizeof(uint32_t);
    if (tensor == DecodeTensor::VerifyCandidates || tensor == DecodeTensor::VerifyProbabilities)
      return (ops::activeVerifyRows()-1)*JUSTFIT_DRAFT_CANDIDATES*sizeof(uint32_t);
    return targetRowTensor(tensor) ? bytes/8*ops::activeVerifyRows() : bytes;
  }
  [[nodiscard]] metal::MetalBuffer get(uint32_t lane, DecodeTensor tensor) const {
    if (lane >= kLaneCount)
      throw std::out_of_range("invalid decode lane");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    const auto index=static_cast<uint32_t>(tensor);
    const auto bytes=stride(tensor);
    if ((lane+1)*bytes > sizes_[index]*kLaneCount) throw std::out_of_range("packed row budget exceeded");
    return bytes ? backend_.view(base_,offsets_[index]+lane*bytes,bytes) : metal::MetalBuffer{};
  }

  [[nodiscard]] metal::MetalBuffer batchSlice(DecodeTensor tensor, uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid decode batch width");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    const uint32_t index = static_cast<uint32_t>(tensor);
    // Model-specific scratch is represented by an empty buffer.  The
    // selected target graph consumes either dense-FFN or MoE tensors, never
    // both, so no dummy allocation is needed for the inactive operator.
    if (!sizes_[index])
      return {};
    const uint64_t bytes=lanes*stride(tensor);
    if (bytes>sizes_[index]*kLaneCount) throw std::out_of_range("packed row budget exceeded");
    return backend_.view(base_, offsets_[index], bytes);
  }

  [[nodiscard]] ops::MoeScratch moeScratch(uint32_t lanes) const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          batchSlice(moeScratchTensor<DecodeTensor>(field), lanes);
    return scratch;
  }

  [[nodiscard]] ops::LinearScratch linearScratch() const { return linearScratch_; }
  static ops::LinearScratchSize linearScratchSize(const RuntimeGeometry &geometry,
                                                 const ops::ExecutionPlans &operators);

  [[nodiscard]] metal::MetalBuffer gateScratch() const { return gateScratch_; }

  static uint64_t gateScratchBytes(const RuntimeGeometry &geometry,
                                  const ops::ExecutionPlans &operators) {
    // One private gate buffer is reused serially by the target dense FFN (when
    // present) and the always-dense DFlash draft. Sparse target FFNs use their
    // own route-major arena tensors, but must not remove the draft's scratch.
    const uint64_t draft = operators.gateUpWorkspace(
        {geometry.draft.intermediateSize, geometry.draft.hiddenSize, ops::WeightLayout::Affine64});
    uint64_t target = 0;
    for (const auto &p : geometry.target.gateUpProjections)
      target = std::max(target, operators.gateUpWorkspace(p));
    return std::max(target, draft);
  }

  [[nodiscard]] metal::MetalBuffer gdnBatchSlice(DecodeTensor base, uint32_t gdnLayer,
                                          uint32_t lanes) const {
    const uint32_t layers = geometry_.target.stateLayout.layers;
    if (!lanes || lanes > kLaneCount || gdnLayer >= layers ||
        !isGdnLayerTensor(base))
      throw std::out_of_range("invalid batched GDN layer");
    return layerBatchSlice(base, layers, gdnLayer, lanes);
  }

  [[nodiscard]] metal::MetalBuffer gdnStorage(DecodeTensor base) const {
    if (!isGdnLayerTensor(base))
      throw std::invalid_argument("tensor is not GDN replay scratch");
    const uint32_t index = static_cast<uint32_t>(base);
    return backend_.view(base_, offsets_[index], sizes_[index] * kLaneCount);
  }

  [[nodiscard]] metal::MetalBuffer attentionBatchSlice(DecodeTensor base,
                                                uint32_t attentionLayer,
                                                uint32_t lanes) const {
    const uint32_t layers = geometry_.target.kvLayout.attentionLayers;
    if (!lanes || lanes > kLaneCount || attentionLayer >= layers ||
        !isAttentionLayerTensor(base)) {
      throw std::out_of_range("invalid batched attention layer");
    }
    return layerBatchSlice(base, layers, attentionLayer, lanes);
  }

  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  // A layer-major tensor holds one stride per lane for each of its `layers`
  // layers, layer by layer; the planner sized it as layers x stride, so the
  // stride is recovered here, not supplied.
  [[nodiscard]] metal::MetalBuffer layerBatchSlice(DecodeTensor base, uint32_t layers,
                                                   uint32_t layer, uint32_t lanes) const {
    const uint32_t index = static_cast<uint32_t>(base);
    const uint64_t stride = sizes_[index] / layers;
    return backend_.view(base_, offsets_[index] + uint64_t{layer} * kLaneCount * stride,
                         uint64_t{lanes} * stride * (isGdnLayerTensor(base) ? ops::activeVerifyRows() : 8) / 8);
  }

  metal::MetalBackend &backend_;
  RuntimeGeometry geometry_;
  metal::MetalBuffer base_;
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> tensors_{};
  std::array<uint64_t, decodeTensorCount> offsets_{};
  std::array<uint64_t, decodeTensorCount> sizes_{};
  metal::MetalBuffer gateScratch_;
  ops::LinearScratch linearScratch_;
  uint64_t bytes_ = 0;
};

} // namespace justfit::model
