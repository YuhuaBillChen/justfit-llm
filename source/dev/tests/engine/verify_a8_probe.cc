// Modified for JustFit: product identity and integration.
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Linear.h"
#include <bit>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace justfit;
static uint16_t bf(float x){auto b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
int main(int argc,char**argv){
 assert(argc==2);metal::MetalBackend backend(argv[1]);
 assert(backend.capabilities().appleGpuFamily>=10);
 auto alloc=[&](size_t n){return backend.allocateBuffer(n,metal::BufferStorage::Shared,"A8-oracle");};
 const unsigned K=256,N=256,G=K/64;
 auto weights=alloc(N*K/2),scales=alloc(N*G*2),biases=alloc(N*G*2);
 auto wp=(uint32_t*)weights.contents();auto sp=(uint16_t*)scales.contents();auto bp=(uint16_t*)biases.contents();
 for(unsigned g=0;g<G;++g)for(unsigned n=0;n<N;++n){sp[g*N+n]=bf(.005f+.001f*(n%3));bp[g*N+n]=bf(-.04f+.003f*(g%3));for(unsigned word=0;word<8;++word){uint32_t v=0;for(unsigned d=0;d<8;++d)v|=((n*3+g*7+word*8+d)%16)<<(d*4);wp[(g*N+n)*8+word]=v;}}
 for(unsigned rows:{8u,16u,24u,32u,64u,256u}){
  auto input=alloc(rows*K*2),codes=alloc(rows*K),scale=alloc(rows*G*4),sums=alloc(rows*G*4),aux=alloc(rows*N*2),output=alloc(rows*N*2);
  auto ip=(uint16_t*)input.contents();auto ap=(uint16_t*)aux.contents();
  for(unsigned r=0;r<rows;++r){for(unsigned k=0;k<K;++k)ip[r*K+k]=bf(r==0?0:std::sin(float(r*17+k*7))*.3f);for(unsigned n=0;n<N;++n)ap[r*N+n]=bf(std::cos(float(r+n))*.2f);}
  std::vector<int> expectedCodes(rows*K);std::vector<float> expectedScales(rows*G),expectedSums(rows*G);
  for(unsigned r=0;r<rows;++r)for(unsigned g=0;g<G;++g){float a=0;for(unsigned d=0;d<64;++d)a=std::max(a,std::abs(fp(ip[r*K+g*64+d])));int sum=0;for(unsigned d=0;d<64;++d){int q=std::lrint(a?fp(ip[r*K+g*64+d])*127/a:0);expectedCodes[r*K+g*64+d]=q;sum+=q;}expectedScales[g*rows+r]=a/127;expectedSums[g*rows+r]=sum;}
  Q4A8Params params{N,K,rows};
  for(const std::string epilogue:{"","_residual","_up_silu"}){
   metal::CommandGraph graph;
   graph.add("verify_linear_q4a8_quantize",{input,codes,scale,sums},params,{rows*G/8,1,1},{256,1,1});
   std::vector<metal::MetalBuffer> buffers{codes,scale,sums,weights,scales,biases};if(!epilogue.empty())buffers.push_back(aux);buffers.push_back(output);
   unsigned tile=rows<=16?16:32;
   graph.add("verify_linear_q4a8"+epilogue+(tile==16?"_m16":"_m32"),buffers,params,{N/128,(rows+tile-1)/tile,1},{128,1,1});
   (void)backend.submitCommandAsync(graph.command()).wait();
   auto cp=(int8_t*)codes.contents();auto ss=(float*)scale.contents();auto sm=(float*)sums.contents();auto op=(uint16_t*)output.contents();
   for(unsigned r=0;r<rows;++r)for(unsigned k=0;k<K;++k){unsigned d=k%64;unsigned stored=(d&~15u)+4*((((d&15)>>3)<<1)|(d&1))+((d>>1)&3);assert(cp[r*K+(k/64)*64+stored]==expectedCodes[r*K+k]);}
   for(unsigned i=0;i<rows*G;++i){assert(std::abs(ss[i]-expectedScales[i])<1e-8);assert(sm[i]==expectedSums[i]);}
   double mse=0,energy=0;float maxerr=0;
   for(unsigned r=0;r<rows;++r)for(unsigned n=0;n<N;++n){float value=0;for(unsigned g=0;g<G;++g){int dot=0;for(unsigned d=0;d<64;++d){int w=(wp[(g*N+n)*8+d/8]>>(4*(d%8)))&15;dot+=w*expectedCodes[r*K+g*64+d];}value+=expectedScales[g*rows+r]*(fp(sp[g*N+n])*dot+fp(bp[g*N+n])*expectedSums[g*rows+r]);}
    if(epilogue=="_residual")value+=fp(ap[r*N+n]);if(epilogue=="_up_silu"){float gate=fp(ap[r*N+n]);value*=gate/(1+std::exp(-gate));}
    float got=fp(op[r*N+n]);assert(std::isfinite(got));float e=got-value;maxerr=std::max(maxerr,std::abs(e));mse+=e*e;energy+=value*value;
   }
   double rel=std::sqrt(mse/(energy+1e-20));std::cout<<"rows="<<rows<<" epilogue="<<epilogue<<" exact_quantizer PASS relativeL2="<<rel<<" max_abs="<<maxerr<<"\n";assert(rel<.005 && maxerr<.005f);
  }
 }
}
