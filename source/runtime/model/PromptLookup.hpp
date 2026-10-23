// Modified for JustFit: product identity and integration.
#pragma once
// Research-only CPU reference of HyperQwen's longest/recent suffix and fusion.
// Only committed history + current verified anchor may be passed here.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace justfit::model {
struct LookupProposal {
  std::array<uint32_t,15> tokens;
  uint32_t match = 0, valid = 0, replaced = 0;
};
inline LookupProposal lookupProposal(std::span<const uint32_t> history,
    std::span<const uint32_t> draft, uint32_t nmin=6, uint32_t nmax=12,
    uint32_t strong=6, uint32_t agreement=0) {
  if (draft.size()!=7 || !nmin || nmax<nmin)
    throw std::invalid_argument("invalid lookup geometry");
  LookupProposal out;
  out.tokens.fill(std::numeric_limits<uint32_t>::max());
  std::copy(draft.begin(),draft.end(),out.tokens.begin());
  if (history.size()<nmin+2) return out;
  size_t bestEnd=0, last=history.size()-1;
  for (size_t end=nmin-1;end<last;++end) {
    uint32_t match=0;
    while (match<nmax && match<=end && history[end-match]==history[last-match]) ++match;
    if (match>=nmin && match>=out.match) { out.match=match;bestEnd=end; }
  }
  if (!out.match) return out;
  out.valid=std::min<size_t>(15,last-bestEnd);
  uint32_t agree=0;
  while (agree<std::min(7u,out.valid) && draft[agree]==history[bestEnd+1+agree]) ++agree;
  bool head=out.match>=strong || (out.match>=nmin && agree>=agreement);
  bool tail=out.match>=nmin && (head || agree==7);
  for (uint32_t i=0;i<out.valid;++i) {
    if ((i<7 && head) || (i>=7 && tail)) {
      out.tokens[i]=history[bestEnd+1+i];++out.replaced;
    }
  }
  return out;
}
} // namespace justfit::model
