// Modified for JustFit: product identity and integration.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace justfit::model {

// The model may verify eight candidates at once, but only committed tokens
// count toward the thinking budget. The close token is the next anchor, so it
// enters KV and GDN state through the ordinary decode path.
struct ThinkingBudgetState final {
  uint32_t budget = 0;
  uint32_t endToken = 0;
  bool closed = false;

  [[nodiscard]] uint32_t maximumRetained(uint32_t generated,
                                          uint32_t remaining,
                                          uint32_t decodeRows) const {
    uint32_t limit = std::min(remaining, decodeRows);
    if (budget && !closed) {
      const uint32_t thoughtRemaining = generated < budget ? budget - generated : 1;
      limit = std::min(limit, thoughtRemaining);
    }
    return limit;
  }

  [[nodiscard]] uint32_t nextAnchor(std::span<const uint32_t> committed,
                                     uint32_t generated,
                                     uint32_t sampledAnchor) {
    if (budget && !closed) {
      closed = std::find(committed.begin(), committed.end(), endToken) !=
               committed.end();
      if (!closed && generated >= budget)
        return endToken;
    }
    return sampledAnchor;
  }
};

} // namespace justfit::model
