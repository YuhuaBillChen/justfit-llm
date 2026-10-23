// Modified for JustFit: product identity and integration.
#pragma once
#include <cstdint>
#include <algorithm>
#include <span>
#include <stdexcept>
namespace justfit::model {
// Keep the learned seven-token head. Lookup may extend only a matching head;
// repetitive field names must not replace a correct draft with stale values.
inline bool retainDraftHead(std::span<uint32_t> proposal,
                            std::span<const uint32_t> draft) {
 if(proposal.size()!=15 || draft.size()!=7)
  throw std::invalid_argument("invalid adaptive proposal shape");
 bool consistent=std::equal(draft.begin(),draft.end(),proposal.begin());
 std::copy(draft.begin(),draft.end(),proposal.begin());
 if(!consistent) std::fill(proposal.begin()+7,proposal.end(),UINT32_MAX);
 return consistent && proposal[7]!=UINT32_MAX;
}
// Signals from the preceding completed step are already on the CPU: no
// extra device synchronization. This retains upstream's two-observation
// entry / singleton sticky3 policy, without its asynchronous flag-copy lag.
struct LookupController {
 bool want=false, previousWant=false;
 uint32_t sticky=0;
 uint32_t choose(bool singleton=true) {
  bool wide=false;
  if(want && previousWant){wide=true;sticky=singleton?3:0;}
  else if(singleton && sticky){wide=true;--sticky;}
  else sticky=0;
  previousWant=want;
  return wide?16:8;
 }
 void observe(bool fillableTail,uint32_t emitted){want=fillableTail && emitted>=8;}
};
inline uint32_t lookupBatchRows(uint32_t lanes, bool enabled, bool anyWide) {
 if(!lanes || lanes>4) throw std::invalid_argument("invalid lookup batch width");
 return enabled && lanes<=2 && anyWide ? 16u : 8u;
}

}
