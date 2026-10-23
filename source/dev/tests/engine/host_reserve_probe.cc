// Modified for JustFit: product identity and integration.
#include "engine/NativeArguments.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  assert(argc == 2);
  auto parse = [&](std::vector<std::string> tail) {
    std::vector<std::string> words{"justfit", "serve-native", argv[1], "auto", "auto"};
    words.insert(words.end(), tail.begin(), tail.end());
    std::vector<const char *> pointers;
    for (const auto &word : words) pointers.push_back(word.c_str());
    return justfit::engine::parseNativeArguments(pointers.size(), pointers.data());
  };
  assert(parse({}).hostReserveBytes == 0);
  assert(parse({"--host-reserve", "auto"}).hostReserveBytes == 0);
  assert(parse({"--host-reserve", "134217728"}).hostReserveBytes == 134217728);
  for (const char *bad : {"0", "-1", "+1", "1x", "", "18446744073709551616"}) {
    bool refused = false;
    try { static_cast<void>(parse({"--host-reserve", bad})); }
    catch (const justfit::engine::UsageError &) { refused = true; }
    assert(refused);
  }
  bool refused = false;
  try { static_cast<void>(parse({"--host-reserve"})); }
  catch (const justfit::engine::UsageError &) { refused = true; }
  assert(refused);
  std::cout << "host reserve native auto/explicit/malformed/overflow checks PASS\n";
}
