#include "model/ThinkingBudget.hpp"
#include <array>
#include <cassert>

int main() {
  using justfit::model::ThinkingBudgetState;
  ThinkingBudgetState unlimited;
  assert(unlimited.maximumRetained(100, 64, 16) == 16);
  assert(unlimited.nextAnchor({}, 100, 42) == 42);
  ThinkingBudgetState budget{3, 99, false};
  assert(budget.maximumRetained(0, 64, 16) == 3);
  assert(budget.maximumRetained(2, 62, 16) == 1);
  const std::array<uint32_t, 3> thoughts{1, 2, 3};
  assert(budget.nextAnchor(thoughts, 3, 42) == 99);
  // The forced close is an ordinary next anchor, not an out-of-band token.
  assert(!budget.closed);
  assert(budget.maximumRetained(3, 61, 16) == 1);
  const std::array<uint32_t, 1> close{99};
  assert(budget.nextAnchor(close, 4, 42) == 42);
  assert(budget.closed);
  assert(budget.maximumRetained(4, 60, 16) == 16);
  ThinkingBudgetState natural{1536, 99, false};
  const std::array<uint32_t, 2> early{1, 99};
  assert(natural.nextAnchor(early, 2, 42) == 42);
  assert(natural.maximumRetained(1600, 20, 16) == 16);
  ThinkingBudgetState other{1, 99, false};
  assert(other.maximumRetained(0, 64, 16) == 1);
  assert(!other.closed);
  const auto resumed = natural;
  assert(resumed.closed);
}
