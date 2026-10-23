// Modified for JustFit: product identity and integration.
#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace justfit::ops {

class Embedding final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const EmbeddingWeights &table, metal::MetalBuffer output,
                  uint32_t rows);
  // A verify step's input tokens, JUSTFIT_TARGET_VERIFY_ROWS per lane: the
  // lane's anchor, row 0 of its draft input rows, then the draft's
  // JUSTFIT_DRAFT_PROPOSAL_TOKENS proposals, each clamped into the
  // vocabulary.
  static void addVerifyInput(metal::CommandGraph &graph,
                             metal::MetalBuffer draftInputTokens,
                             metal::MetalBuffer proposedTokens,
                             metal::MetalBuffer verifyInputTokens,
                             uint32_t vocabulary, uint32_t lanes);
};

} // namespace justfit::ops
