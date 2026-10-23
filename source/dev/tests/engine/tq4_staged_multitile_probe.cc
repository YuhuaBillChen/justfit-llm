// Modified for JustFit: product identity and integration.
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace justfit;
static uint16_t bf(float x){auto b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
static constexpr float cb[16]={-.168522835f,-.127540648f,-.0996119156f,-.0771940202f,-.0578124076f,-.0402186327f,-.0237036943f,-.00783268455f,.00783268549f,.0237036962f,.0402186289f,.0578124076f,.0771940276f,.0996119007f,.127540648f,.168522835f};
static constexpr uint32_t signs[2][8]={{0xe05422cd,0xf23e567d,0xed27523d,0x5a9acdca,0x22d0370d,0x10b3a0b8,0xf87f105f,0x3efbe5ce},{0xa45beb87,0xac6b118e,0xeafcb960,0x7c1e2f3b,0xeac834c2,0xac17ef56,0xd86514b1,0xe2c2bde9}};
static float sign(int v,int d){return (signs[v][d/32]>>(d%32))&1 ? 1.f:-1.f;}
int main(int argc,char**argv){
 assert(argc==2);metal::MetalBackend backend(argv[1]);
 kv::Layout layout{16,4,256,kv::Format::TQ4};auto extent=layout.extentPagesFor(8192);
 kv::PageStorage storage(backend,[](uint64_t,const auto&a){a();return metal::AllocationResult{};},layout,extent*2,extent);
 assert(storage.allocateExtent(0));assert(storage.allocateExtent(1));
 auto alloc=[&](uint64_t bytes){return backend.allocateBuffer(bytes,metal::BufferStorage::Shared,"tq4-attention-probe");};
 auto table=alloc(4*sizeof(JustFitKvPage));uint32_t pages[]={extent-2,extent-1,extent,extent+1};storage.writeEntries(pages,0,table);
 auto keys=alloc(4*128*256*2),values=alloc(4*128*256*2);
 auto*k=static_cast<uint16_t*>(keys.contents());auto*v=static_cast<uint16_t*>(values.contents());
 std::memset(k,0,keys.sizeBytes());std::memset(v,0,values.sizeBytes());
 auto sample=[](int tensor,int h,int t,int d){return bf(std::sin(float(1+tensor*199+h*17+t*31+d*13)) * (.05f+.03f*(t%7)));};
 for(int h=0;h<4;++h)for(int t=0;t<104;++t)for(int d=0;d<256;++d){k[(h*128+t)*256+d]=sample(0,h,t,d);v[(h*256+d)*128+t]=sample(1,h,t,d);}
 auto chunk=ops::PagedAttention::prefillParams(0,104,128,4);
 metal::CommandGraph store;ops::PagedAttention::addPrefillStore(store,storage.layers()[15],keys,values,table,chunk,layout);
 (void)backend.submitCommandAsync(store.command()).wait();
 // Independent scalar encoder reference on dense BF16 vectors with varied norms.
 // Compare all codes and half norms after real GPU stores across an extent boundary.
 std::vector<float> decoded(2*4*104*256);
 for(int tensor=0;tensor<2;++tensor)for(int h=0;h<4;++h)for(int t=0;t<104;++t){
  auto spans=storage.spans(pages[t/32]);auto indices=reinterpret_cast<const uint32_t*>(spans[15*4+tensor*2].data());
  auto norms=reinterpret_cast<const uint16_t*>(spans[15*4+tensor*2+1].data());
  float values[256],transformed[256],norm=0;
  for(int d=0;d<256;++d){float x=fp(sample(tensor,h,t,d));norm+=x*x;transformed[d]=x*sign(tensor,d);}
  norm=std::sqrt(norm);_Float16 halfNorm=norm;float storedNorm=float(std::bit_cast<_Float16>(norms[h*32+t%32]));assert(storedNorm==float(halfNorm));
  for(int step=1;step<256;step*=2)for(int base=0;base<256;base+=step*2)for(int j=0;j<step;++j){float a=transformed[base+j],b=transformed[base+j+step];transformed[base+j]=a+b;transformed[base+j+step]=a-b;}
  for(int d=0;d<256;++d){float x=transformed[d]/(16*norm);int expected=0;for(int j=0;j<15;++j)expected+=x>(cb[j]+cb[j+1])/2;int code=(indices[(h*32+t%32)*32+d/8]>>(4*(d%8)))&15;
   assert(code==expected);values[d]=cb[code]*storedNorm;}
  for(int step=1;step<256;step*=2)for(int base=0;base<256;base+=step*2)for(int j=0;j<step;++j){float a=values[base+j],b=values[base+j+step];values[base+j]=a+b;values[base+j+step]=a-b;}
  for(int d=0;d<256;++d)decoded[((tensor*4+h)*104+t)*256+d]=fp(bf(values[d]*sign(tensor,d)/16));
 }
 auto queries=alloc(4*128*6*256*2),output=alloc(4*128*6*256*2);auto*q=static_cast<uint16_t*>(queries.contents());std::memset(q,0,queries.sizeBytes());
 for(int h=0;h<4;++h)for(int row=0;row<104;++row)for(int g=0;g<6;++g)for(int d=0;d<256;++d)q[((h*128+row)*6+g)*256+d]=bf(std::cos(float(h*3+row*7+g*11+d*5)));
 auto params=ops::PagedAttention::prefillParams(0,104,128,4);auto plan=ops::PagedAttention::prefillPlan(104,24,layout);
 auto partial=alloc(plan.workspace.partialsBytes),stats=alloc(plan.workspace.statisticsBytes);
 metal::CommandGraph graph;ops::PagedAttention::addPrefill(graph,storage.layers()[15],queries,output,partial,stats,table,params,plan);(void)backend.submitCommandAsync(graph.command()).wait();
 auto*out=static_cast<uint16_t*>(output.contents());float maxerr=0;
 for(int h=0;h<4;++h)for(int row=0;row<104;++row)for(int g=0;g<6;++g){
  float weights[104],sum=0;
  for(int t=0;t<=row;++t){float dot=0;for(int d=0;d<256;++d)dot+=decoded[(h*104+t)*256+d]*fp(q[((h*128+row)*6+g)*256+d]);weights[t]=std::exp(dot/16);sum+=weights[t];}
  for(int d=0;d<256;++d){float expected=0;for(int t=0;t<=row;++t)expected+=weights[t]/sum*decoded[((4+h)*104+t)*256+d];float actual=fp(out[((h*128+row)*6+g)*256+d]);assert(std::isfinite(actual));maxerr=std::max(maxerr,std::abs(expected-actual));}
 }
 std::cout<<"CPU-codec-exact; prefill causal cross-extent max_abs_error="<<maxerr<<"\n";assert(maxerr<.001f);
 std::vector<uint16_t> prefill(out,out+output.sizeBytes()/2);
 auto sk=alloc(128*256*2),sv=alloc(128*256*2);
 std::memset(sk.contents(),0,sk.sizeBytes());std::memset(sv.contents(),0,sv.sizeBytes());
 metal::CommandGraph staged;
 ops::PagedAttention::addStagedPrefill(staged,storage.layers()[15],queries,output,sk,sv,table,params,layout);
 (void)backend.submitCommandAsync(staged.command()).wait();
 float delta=0;
 for(int h=0;h<4;++h)for(int row=0;row<104;++row)for(int g=0;g<6;++g)for(int d=0;d<256;++d){size_t i=((h*128+row)*6+g)*256+d;float got=fp(out[i]);assert(std::isfinite(got));delta=std::max(delta,std::abs(got-fp(prefill[i])));}
 std::cout<<"staged40rows fivecausaltiles cross-extent versus CPU-checked direct max_abs="<<delta<<"\n";assert(delta<.001f);
}
