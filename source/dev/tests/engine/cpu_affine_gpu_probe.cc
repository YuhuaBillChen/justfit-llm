// Modified for JustFit: product identity and integration.
#include "ops/HostLinear.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Linear.h"
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace justfit;
static uint16_t bf(float x){uint32_t u=std::bit_cast<uint32_t>(x);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static float fp(uint16_t u){return std::bit_cast<float>(uint32_t(u)<<16);}
int main(int argc,char**argv){
 assert(argc==3);
 metal::MetalBackend backend(argv[1]);
 constexpr uint32_t rows=256,k=5120,n=512,groups=k/64;
 auto alloc=[&](uint64_t bytes){return backend.allocateBuffer(bytes,metal::BufferStorage::Shared,"cpu-affine-audit");};
 for(uint32_t layer:{0u,16u,32u,48u,63u}){
  std::ifstream file(std::string(argv[2])+"/target/layer-"+std::to_string(layer)+".bin",std::ios::binary);
  assert(file);char header[16];file.read(header,16);assert(std::memcmp(header,"MDFL0006",8)==0);
  uint32_t index,full;std::memcpy(&index,header+8,4);std::memcpy(&full,header+12,4);assert(index==layer && full<=1);
  const uint32_t sourceN=full?14336:16640;
  auto input=alloc(rows*k*2),weights=alloc(n*k/2),scales=alloc(n*k/32),biases=alloc(n*k/32);
  auto output=alloc(rows*n*2),cpu=alloc(rows*n*2),sums=alloc(rows*groups*4),scratch=alloc(rows*k*2);
  auto read=[&](uint64_t offset,metal::MetalBuffer dest){file.seekg(offset);file.read(static_cast<char*>(dest.contents()),dest.sizeBytes());assert(file);};
  constexpr uint64_t start=32768;
  read(start+uint64_t(sourceN-n)*k/2,weights);
  read(start+uint64_t(sourceN)*k/2+uint64_t(sourceN-n)*k/32,scales);
  read(start+uint64_t(sourceN)*k/2+uint64_t(sourceN)*k/32+uint64_t(sourceN-n)*k/32,biases);
  auto* x=static_cast<uint16_t*>(input.contents());
  for(uint32_t r=0;r<rows;++r)for(uint32_t j=0;j<k;++j)x[size_t(r)*k+j]=bf(std::sin(float(j+1)*.173f+float(r)*.031f));
  metal::CommandGraph graph;
  graph.add("prefill_linear_q4_sums32",{input,sums},k,{rows/32,1,1},{256,1,1});
  graph.add("prefill_linear_q4_n128",{input,weights,scales,biases,output,sums},Q4Params{n,k},{rows/32,n/128,1},{256,1,1});
  (void)backend.submitCommandAsync(graph.command()).wait();
  ops::HostLinearTask task;task.input=x;task.weights=static_cast<uint8_t*>(weights.contents());
  task.scales=static_cast<uint16_t*>(scales.contents());task.biases=static_cast<uint16_t*>(biases.contents());
  task.output=static_cast<uint16_t*>(cpu.contents());
  task.rows=rows;task.inputSize=k;task.outputSize=n;task.firstColumn=256;task.scratch=static_cast<uint16_t*>(scratch.contents());task.scratchBytes=scratch.sizeBytes();
  auto begin=std::chrono::steady_clock::now();ops::runHostLinear(task);
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  auto* gpu=static_cast<uint16_t*>(output.contents());auto* host=task.output;
  double error=0,norm=0;uint32_t mismatches=0;
  for(uint32_t r=0;r<rows;++r)for(uint32_t c=256;c<n;++c){float a=fp(gpu[size_t(r)*n+c]),b=fp(host[size_t(r)*n+c]);assert(std::isfinite(b));error+=(a-b)*(a-b);norm+=a*a;mismatches+=gpu[size_t(r)*n+c]!=host[size_t(r)*n+c];}
  double gpuRef=0,cpuRef=0,refNorm=0;
  for(uint32_t r=0;r<8;++r)for(uint32_t c=256;c<264;++c){
   double dot=0;for(uint32_t j=0;j<k;++j){size_t p=(size_t(c/256)*groups+j/64)*256+c%256;uint8_t raw=task.weights[p*32+(j%64)/2];uint32_t q=(raw>>((j%2)*4))&15;dot+=double(fp(x[size_t(r)*k+j]))*(double(q)*fp(task.scales[p])+fp(task.biases[p]));}
   double a=fp(gpu[size_t(r)*n+c])-dot,b=fp(host[size_t(r)*n+c])-dot;gpuRef+=a*a;cpuRef+=b*b;refNorm+=dot*dot;
  }
  std::cout<<"layer="<<layer<<" gpu_cpu_relative_rms="<<std::sqrt(error/norm)<<" bf16_mismatches="<<mismatches<<"/65536 gpu_fp64_rms="<<std::sqrt(gpuRef/refNorm)<<" cpu_fp64_rms="<<std::sqrt(cpuRef/refNorm)<<" cpu_ms="<<ms<<std::endl;
  assert(std::sqrt(error/norm)<.0003);
  assert(std::sqrt(gpuRef/refNorm)<.003);
  assert(std::sqrt(cpuRef/refNorm)<.003);
 }
}
