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
 auto table=alloc(2*sizeof(JustFitKvPage));uint32_t pages[]={extent-1,extent};storage.writeEntries(pages,0,table);
 auto keys=alloc(4*64*256*2),values=alloc(4*64*256*2);
 auto*k=static_cast<uint16_t*>(keys.contents());auto*v=static_cast<uint16_t*>(values.contents());
 std::memset(k,0,keys.sizeBytes());std::memset(v,0,values.sizeBytes());
 for(int h=0;h<4;++h)for(int t=0;t<40;++t){k[(h*64+t)*256+(t+h*3)%256]=bf(1);v[(h*256+(t*3+h)%256)*64+t]=bf(1);}
 auto chunk=ops::PagedAttention::prefillParams(0,40,64,2);
 metal::CommandGraph store;ops::PagedAttention::addPrefillStore(store,storage.layers()[15],keys,values,table,chunk,layout);
 (void)backend.submitCommandAsync(store.command()).wait();
 // Independent scalar encoder reference: orthonormal Hadamard of unit basis.
 // Compare all codes and half norms after real GPU stores across an extent boundary.
 std::vector<float> decoded(2*4*40*256);
 for(int tensor=0;tensor<2;++tensor)for(int h=0;h<4;++h)for(int t=0;t<40;++t){
  auto spans=storage.spans(pages[t/32]);auto indices=reinterpret_cast<const uint32_t*>(spans[15*4+tensor*2].data());
  auto norms=reinterpret_cast<const uint16_t*>(spans[15*4+tensor*2+1].data());assert(norms[h*32+t%32]==0x3c00);
  int axis=tensor?(t*3+h)%256:(t+h*3)%256;float values[256];
  for(int d=0;d<256;++d){float transformed=((std::popcount(unsigned(d&axis))%2)?-1.f:1.f)*sign(tensor,axis)/16;
   int expected=0;for(int j=0;j<15;++j)expected+=transformed>(cb[j]+cb[j+1])/2;int code=(indices[(h*32+t%32)*32+d/8]>>(4*(d%8)))&15;
   assert(code==expected);values[d]=cb[code];}
  for(int step=1;step<256;step*=2)for(int base=0;base<256;base+=step*2)for(int j=0;j<step;++j){float a=values[base+j],b=values[base+j+step];values[base+j]=a+b;values[base+j+step]=a-b;}
  for(int d=0;d<256;++d)decoded[((tensor*4+h)*40+t)*256+d]=fp(bf(values[d]*sign(tensor,d)/16));
 }
 auto queries=alloc(4*32*6*256*2),output=alloc(4*32*6*256*2);auto*q=static_cast<uint16_t*>(queries.contents());std::memset(q,0,queries.sizeBytes());
 for(int h=0;h<4;++h)for(int row=0;row<8;++row)for(int g=0;g<6;++g)q[((h*32+row)*6+g)*256+(row*5+g+h)%256]=bf(1);
 auto params=ops::PagedAttention::prefillParams(32,8,32,2);auto plan=ops::PagedAttention::prefillPlan(8,24,layout);
 auto partial=alloc(plan.workspace.partialsBytes),stats=alloc(plan.workspace.statisticsBytes);
 metal::CommandGraph graph;ops::PagedAttention::addPrefill(graph,storage.layers()[15],queries,output,partial,stats,table,params,plan);(void)backend.submitCommandAsync(graph.command()).wait();
 auto*out=static_cast<uint16_t*>(output.contents());float maxerr=0;
 for(int h=0;h<4;++h)for(int row=0;row<8;++row)for(int g=0;g<6;++g){
  int axis=(row*5+g+h)%256;float weights[40],sum=0;
  for(int t=0;t<=32+row;++t){weights[t]=std::exp(decoded[(h*40+t)*256+axis]/16);sum+=weights[t];}
  for(int d=0;d<256;++d){float expected=0;for(int t=0;t<=32+row;++t)expected+=weights[t]/sum*decoded[((4+h)*40+t)*256+d];float actual=fp(out[((h*32+row)*6+g)*256+d]);assert(std::isfinite(actual));maxerr=std::max(maxerr,std::abs(expected-actual));}
 }
 std::cout<<"CPU-codec-exact; prefill causal cross-extent max_abs_error="<<maxerr<<"\n";assert(maxerr<.001f);
 std::vector<uint16_t> prefill(out,out+output.sizeBytes()/2);
 for(uint32_t lanes : {1u,4u}) {
  auto vk=alloc(lanes*4*32*256*2),vv=alloc(lanes*4*32*256*2),vq=alloc(lanes*queries.sizeBytes()),vo=alloc(lanes*output.sizeBytes());
  std::memset(vk.contents(),0,vk.sizeBytes());std::memset(vv.contents(),0,vv.sizeBytes());
  std::array<metal::MetalBuffer,4> tables{table,table,table,table};
  std::vector<kv::ChunkedPrefillParams> chunks;
  std::vector<uint32_t> histories(lanes,32);
  for(uint32_t lane=0;lane<lanes;++lane) {
   std::memcpy(static_cast<std::byte*>(vq.contents())+lane*queries.sizeBytes(),queries.contents(),queries.sizeBytes());
   if(lane) {
    tables[lane]=alloc(2*sizeof(JustFitKvPage));uint32_t ps[]={extent+lane*2,extent+lane*2+1};storage.writeEntries(ps,0,tables[lane]);
    metal::CommandGraph fill;ops::PagedAttention::addPrefillStore(fill,storage.layers()[15],keys,values,tables[lane],chunk,layout);(void)backend.submitCommandAsync(fill.command()).wait();
   }
   auto kk=static_cast<uint16_t*>(vk.contents())+lane*4*32*256;
   auto vval=static_cast<uint16_t*>(vv.contents())+lane*4*32*256;
   for(int h=0;h<4;++h)for(int r=0;r<8;++r) {int t=32+r;kk[(h*32+r)*256+(t+h*3)%256]=bf(1);vval[(h*256+(t*3+h)%256)*32+r]=bf(1);}
   chunks.push_back(ops::PagedAttention::verifyParams(32,2));
  }
  auto vp=ops::PagedAttention::verifyPlan(lanes,24,layout,histories);
  auto partials=alloc(vp.workspace.partialsBytes),statistics=alloc(vp.workspace.statisticsBytes);
  metal::CommandGraph verify;ops::PagedAttention::addVerify(verify,storage.layers()[15],{vk,vv,vq,partials,statistics,vo,tables},chunks,vp);(void)backend.submitCommandAsync(verify.command()).wait();
  float error=0;auto result=static_cast<uint16_t*>(vo.contents());
  for(uint32_t lane=0;lane<lanes;++lane)for(int h=0;h<4;++h)for(int r=0;r<8;++r)for(int g=0;g<6;++g)for(int d=0;d<256;++d) {
   auto i=((h*32+r)*6+g)*256+d;float got=fp(result[lane*output.sizeBytes()/2+i]);assert(std::isfinite(got));error=std::max(error,std::abs(got-fp(prefill[i])));
  }
  std::cout<<"verify lanes="<<lanes<<" vs prefill max_abs_error="<<error<<"\n";assert(error<.001f);
 }

}
