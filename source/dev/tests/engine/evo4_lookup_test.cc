// Modified for JustFit: product identity and integration.
#include "model/LookupController.hpp"
#include "model/PromptLookup.hpp"
#include <array>
#include <cassert>
#include <iostream>
using namespace justfit::model;
int main(){
 for(uint32_t lanes=1;lanes<=4;++lanes) {
  assert(lookupBatchRows(lanes,true,true)==(lanes<=2?16:8));
  assert(lookupBatchRows(lanes,true,false)==8);
  assert(lookupBatchRows(lanes,false,true)==8);
 }
 LookupController a,b;
 assert(a.choose(false)==8);a.observe(true,8);assert(a.choose(false)==8);a.observe(true,8);assert(a.choose(false)==16);
 assert(b.choose(false)==8);a.observe(false,8);assert(a.choose(false)==8);
 LookupController single;single.observe(true,8);assert(single.choose()==8);assert(single.choose()==16);single.observe(false,8);
 for(int i=0;i<3;++i) { assert(single.choose()==16); }
 assert(single.choose()==8);
 std::array<uint32_t,7> draft{20,21,22,23,24,25,26};
 std::vector<uint32_t> history{10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,90,10,11,12,13,14,15};
 auto proposal=lookupProposal(history,draft);assert(retainDraftHead(proposal.tokens,draft));assert(proposal.tokens[7]==27);assert(proposal.tokens[14]==34);
 draft[3]=999;proposal=lookupProposal(history,draft);assert(!retainDraftHead(proposal.tokens,draft));assert(proposal.tokens[3]==999);
 for(int i=7;i<15;++i)assert(proposal.tokens[i]==UINT32_MAX);
 std::cout<<"PASS controller entry/exit/sticky and learned-head consistency\n";
}
