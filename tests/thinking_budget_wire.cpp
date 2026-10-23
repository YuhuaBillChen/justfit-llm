#include "engine/Protocol.hpp"
#include <iostream>
#include <iterator>
#include <vector>

int main() {
  using namespace justfit::protocol;
  const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(std::cin), {}};
  FrameParser parser({});
  auto parsed = parser.consume(bytes);
  if (parsed.issue || !parsed.frame || parsed.consumedBytes != bytes.size()) return 2;
  auto decoded = decodeFrame(std::move(*parsed.frame), {});
  if (!decoded || !std::holds_alternative<RequestFrame>(*decoded.value)) return 3;
  const auto &request = std::get<RequestFrame>(*decoded.value);
  const auto view = justfit::engine::EngineRequest{
      .thinkingBudgetTokens = request.thinkingBudgetTokens,
      .thinkingEndTokenId = request.thinkingEndTokenId}.modelView();
  std::cout << view.thinkingBudgetTokens << ' ' << view.thinkingEndTokenId << ' '
            << request.logicalMaxOutputTokens << ' ' << request.promptTokens.back() << ' '
            << request.imageEmbeddingWidth;
}
