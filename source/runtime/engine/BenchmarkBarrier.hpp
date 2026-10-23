// Modified for JustFit: product identity and integration.
#pragma once
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
namespace justfit::engine {
// Isolated benchmark gate; unset in product operation.
inline bool benchmarkDecodeHeld() {
  const char *path = std::getenv("DASH_TEST_DECODE_HOLD_FILE");
  if (!path || !*path) return false;
  if (::access(path, F_OK) == 0) return true;
  if (errno == ENOENT) return false;
  throw std::runtime_error("cannot inspect benchmark decode barrier");
}
}
