// Modified for JustFit: product identity and integration.
#include "ops/HostLinear.hpp"
#include <bit>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace justfit::ops;
static float from(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
static uint16_t bf(float x){uint32_t b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
int main(){
 for(uint32_t physicalK:{64u,128u}) for(auto ep:{HostEpilogue::None,HostEpilogue::Residual,HostEpilogue::UpSilu}) {
  constexpr uint32_t rows=256,k=64,n=512,first=256;
  const uint32_t pg=physicalK/64;
  std::vector<uint16_t> input(rows*k),scales(n*pg,bf(1.f/64)),bias(n*pg,bf(-7.f/64)),out(rows*n,0x1234),aux(rows*n),scratch(64*k);
  std::vector<uint8_t> codes(n*pg*32);std::vector<float> sums(rows*n/64,-999);
  for(size_t i=0;i<scales.size();++i){scales[i]=bf(float(i*7%9+1)/64);bias[i]=bf(float(int(i*5%11)-7)/64);}
  for(size_t i=0;i<input.size();++i)input[i]=bf(float(int(i*7%17)-8)/32);
  for(size_t i=0;i<aux.size();++i)aux[i]=bf(float(int(i%13)-6)/8);
  for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t((i*13+3)%256);
  HostLinearTask t; t.input=input.data();t.weights=codes.data();t.scales=scales.data();t.biases=bias.data();t.output=out.data();t.auxiliary=aux.data();t.outputSums=sums.data();t.scratch=scratch.data();t.scratchBytes=scratch.size()*2;t.rows=rows;t.inputSize=k;t.weightInputSize=physicalK;t.outputSize=n;t.firstColumn=first;t.epilogue=ep;
  runHostLinear(t);
  for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<n;++c){
   if(c<first){assert(out[r*n+c]==0x1234);continue;}
   double dot=0;
   for(uint32_t j=0;j<k;++j){
    size_t p=(size_t(c/256)*pg+j/64)*256+c%256;
    uint8_t raw=codes[p*32+(j%64)/2];uint32_t q=(raw>>((j&1)*4))&15;
    double w=from(bf(float(q)*from(scales[p])+from(bias[p])));
    dot+=double(from(input[r*k+j]))*w;
   }
   float v=from(bf(float(dot)));
   if(ep==HostEpilogue::Residual)v+=from(aux[r*n+c]);
   if(ep==HostEpilogue::UpSilu){float g=from(aux[r*n+c]);v=g/(1.f+std::exp2(-1.44269504089f*g))*v;}
   assert(out[r*n+c]==bf(v));
  }
  if(ep==HostEpilogue::UpSilu)for(uint32_t r=0;r<rows;++r)for(uint32_t g=first/64;g<n/64;++g){
   float sum=0;for(uint32_t c=0;c<64;++c)sum+=from(out[r*n+g*64+c]);
   assert(sums[(size_t(r/32)*(n/64)+g)*32+r%32]==sum);
  }
 }
 std::cout<<"PASS SME vs independent scalar reference, leading-input stride, epilogues, sums and bounded chunks\n";
}
