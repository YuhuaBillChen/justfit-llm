// Modified for JustFit: product identity and integration.
#include "engine/MemoryPlan.hpp"
#include <cassert>
#include <iostream>
int main(){using P=justfit::engine::EngineMemoryPolicy;constexpr uint64_t G=1024ULL*1024*1024;assert(P::hardBudgetBytes(22*G,0)==21*G);assert(P::hardBudgetBytes(22*G,20*G)==20*G);assert(P::hardBudgetBytes(22*G,22*G)==22*G);assert(P::hardBudgetBytes(22*G,30*G)==22*G);assert(P::hardBudgetBytes(0,1)==0);std::cout<<"explicit bounded budget PASS\n";}
