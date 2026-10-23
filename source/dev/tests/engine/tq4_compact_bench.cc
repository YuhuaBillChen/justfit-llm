// Modified for JustFit: product identity and integration.
#include "ops/PagedAttention.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace justfit;
static uint16_t bf(float x){auto b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
int main(int argc,char**argv){
 assert(argc==2);metal::MetalBackend backend(argv[1]);
 auto alloc=[&](uint64_t bytes){return backend.allocateBuffer(bytes,metal::BufferStorage::Shared,"compact-softmax-probe");};
 for(uint32_t committed : {0u, 4096u, 32768u, 65536u}) {
  const uint32_t rows=committed?256:104,stride=(rows+31)/32*32,visible=committed+rows,padded=(visible+63)/64*64;
  auto keys=alloc(uint64_t(padded)*256*2),values=alloc(uint64_t(padded)*256*2);
  auto queries=alloc(uint64_t(stride)*6*256*2),out=alloc(queries.sizeBytes());
  auto*k=static_cast<uint16_t*>(keys.contents());auto*v=static_cast<uint16_t*>(values.contents());auto*q=static_cast<uint16_t*>(queries.contents());
  uint32_t seed=1234567;
  auto sample=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%20001)-10000)/10000.f;};
  for(uint64_t i=0;i<uint64_t(padded)*256;++i){k[i]=bf(sample()*3.f);v[i]=bf(sample());}
  for(uint64_t i=0;i<queries.sizeBytes()/2;++i)q[i]=bf(sample()*3.f);
  JustFitTQ4StagedPrefillParams p{committed,rows,stride,visible,{},0};
  std::vector<uint16_t> reference;std::vector<double> times[2];float maxerr=0;double rel=0;
  for(int repetition=-1;repetition<5;++repetition) for(int j=0;j<2;++j){
   // Alternate order after warmup to reduce order bias.
   const int arm=repetition>=0 ? (j+(repetition%2))%2 : j;
   metal::CommandGraph graph;
   graph.add(arm?"prefill_attention_tq4_staged_compact":"prefill_attention_tq4_staged_head",{queries,keys,values,out},p,{(rows+7)/8,1,1},{256,1,1});
   const auto start=std::chrono::steady_clock::now();
   (void)backend.submitCommandAsync(graph.command()).wait();
   const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   auto *o=static_cast<uint16_t*>(out.contents());
   if(repetition==-1 && arm==0)reference.assign(o,o+uint64_t(rows)*6*256);
   if(repetition==-1 && arm==1){double diff=0,den=0;
    for(size_t i=0;i<reference.size();++i){float x=fp(o[i]),r=fp(reference[i]);assert(std::isfinite(x));maxerr=std::max(maxerr,std::abs(x-r));diff+=double(x-r)*(x-r);den+=double(r)*r;}
    rel=std::sqrt(diff/den);assert(maxerr<.02f && rel<.005);
   }
   if(repetition>=0)times[arm].push_back(ms);
  }
  for(auto& t:times)std::sort(t.begin(),t.end());
  std::cout<<"committed="<<committed<<" rows="<<rows<<" max_abs="<<maxerr<<" relative_l2="<<rel<<" base_ms="<<times[0][2]<<" compact_ms="<<times[1][2]<<" ratio="<<times[0][2]/times[1][2]<<std::endl;
 }
}
