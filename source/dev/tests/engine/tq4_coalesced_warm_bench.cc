// Modified for JustFit: product identity and integration.
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
using namespace justfit;
static uint16_t bf(float x){auto b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
int main(int argc,char**argv){
 assert(argc==2);metal::MetalBackend backend(argv[1]);
 setenv("JUSTFIT_ROTATED_VERIFY","1",1);unsetenv("JUSTFIT_DISPATCH_PROBE");
 kv::Layout layout{1,4,256,kv::Format::TQ4};const auto extent=layout.extentPagesFor(8192);
 const uint32_t usedPages=(65536+8+31)/32,pageCount=(usedPages+extent-1)/extent*extent;
 kv::PageStorage storage(backend,[](uint64_t,const auto&a){a();return metal::AllocationResult{};},layout,pageCount,extent);
 for(uint32_t e=0;e<pageCount/extent;++e)assert(storage.allocateExtent(e));
 uint32_t seed=123457;auto rnd=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};
 for(uint32_t p=0;p<usedPages;++p){auto spans=storage.spans(p);
  for(int i:{0,2}){auto*words=reinterpret_cast<uint32_t*>(spans[i].data());for(size_t j=0;j<spans[i].size()/4;++j)words[j]=rnd();}
  for(int i:{1,3}){auto*norms=reinterpret_cast<uint16_t*>(spans[i].data());for(size_t j=0;j<spans[i].size()/2;++j)norms[j]=std::bit_cast<uint16_t>(_Float16(1.f+float(rnd()%100)/100.f));}
 }
 auto alloc=[&](uint64_t bytes){return backend.allocateBuffer(bytes,metal::BufferStorage::Shared,"tq4-split-probe");};
 auto table=alloc(usedPages*sizeof(JustFitKvPage));std::vector<uint32_t> pages(usedPages);std::iota(pages.begin(),pages.end(),0);storage.writeEntries(pages,0,table);
 auto keys=alloc(4*32*256*2),values=alloc(4*32*256*2),queries=alloc(4*32*6*256*2),output=alloc(queries.sizeBytes());
 std::memset(keys.contents(),0,keys.sizeBytes());std::memset(values.contents(),0,values.sizeBytes());
 auto*q=static_cast<uint16_t*>(queries.contents());for(size_t i=0;i<queries.sizeBytes()/2;++i)q[i]=bf(float(int(rnd()%20001)-10000)/3000.f);
 std::array<metal::MetalBuffer,4> tables{table,table,table,table};
 for(uint32_t history:{4096u,32768u,65536u}){
  unsetenv("JUSTFIT_TQ4_SPLITS");auto plan=ops::PagedAttention::verifyPlan(1,24,layout,std::span(&history,1));
  auto partials=alloc(plan.workspace.partialsBytes),stats=alloc(plan.workspace.statisticsBytes);
  std::array<double,2> maxerr{},rel{};std::array<std::vector<double>,2> times;
  std::vector<uint16_t> reference;
  for(int rep=-1;rep<7;++rep)for(int j=0;j<2;++j){
   int arm=rep<0?j:(j+rep)%2;
   const char*options[]={nullptr,nullptr};
   setenv("JUSTFIT_VERIFY_COALESCED",arm?"1":"0",1);
   if(options[arm])setenv("JUSTFIT_TQ4_SPLITS",options[arm],1);else unsetenv("JUSTFIT_TQ4_SPLITS");
   auto selected=ops::PagedAttention::verifyPlan(1,24,layout,std::span(&history,1));
   auto chunk=ops::PagedAttention::verifyParams(history,usedPages);
   metal::CommandGraph graph;
   for(int batch=0;batch<16;++batch) ops::PagedAttention::addVerify(graph,storage.layers()[0],{keys,values,queries,partials,stats,output,tables},std::span(&chunk,1),selected);
   const auto timing=backend.submitCommandAsync(graph.command()).wait();
   auto*out=static_cast<uint16_t*>(output.contents());
   if(rep==-1&&arm==0)reference.assign(out,out+output.sizeBytes()/2);
   if(rep==-1){double den=0,diff=0;
    for(int h=0;h<4;++h)for(int row=0;row<8;++row)for(int g=0;g<6;++g)for(int d=0;d<256;++d){auto i=((h*32+row)*6+g)*256+d;double x=fp(out[i]),r=fp(reference[i]);assert(std::isfinite(x));maxerr[arm]=std::max(maxerr[arm],std::abs(x-r));diff+=(x-r)*(x-r);den+=r*r;}
    rel[arm]=std::sqrt(diff/den);assert(rel[arm]<.005 && maxerr[arm]<.001);
   }else times[arm].push_back(timing.gpuSeconds*1000/16);
  }
  for(int arm=0;arm<2;++arm){std::sort(times[arm].begin(),times[arm].end());std::cout<<"history="<<history<<" splits="<<plan.splits<<" arm="<<arm<<" gpu_ms="<<times[arm][3]<<" max_abs="<<maxerr[arm]<<" relative_l2="<<rel[arm]<<std::endl;}

 }
}
