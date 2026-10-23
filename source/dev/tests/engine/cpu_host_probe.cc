// Modified for JustFit: product identity and integration.
#include "metal/CommandGraph.hpp"
#include "ops/Normalization.hpp"
#include <cassert>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
using namespace justfit;
int main(int argc,char **argv) {
  assert(argc==2);
  for(bool failure:{false,true}) {
    metal::MetalBackend backend(argv[1]);
    auto alloc=[&](size_t n,uint16_t v){auto b=backend.allocateBuffer(n*2,metal::BufferStorage::Shared,"cpu-host-probe");std::fill_n(static_cast<uint16_t*>(b.contents()),n,v);return b;};
    auto input=alloc(32*64,0x4000),producer=alloc(32*64,0),cpu=alloc(32*64,0),gpu=alloc(32*64,0),out=alloc(32*64,0),weights=alloc(64,0x3f80);
    metal::CommandGraph graph;
    ops::Normalization::addRms(graph,input,{weights},producer,64,32);
    std::atomic<bool> ran=false;
    graph.addHostTask([=,&ran]{
      assert(static_cast<uint16_t*>(producer.contents())[0]==0x3f80);
      if(failure) throw std::runtime_error("intentional host failure");
      std::fill_n(static_cast<uint16_t*>(cpu.contents()),32*64,0x4040);ran=true;
    });
    ops::Normalization::addRms(graph,input,{weights},gpu,64,32);
    graph.addHostJoin();
    ops::Normalization::addRms(graph,cpu,{weights},out,64,32);
    bool threw=false;
    try{(void)backend.submitCommandAsync(graph.command()).wait();}
    catch(const metal::MetalBackendError &){threw=true;}
    assert(threw==failure);
    if(!failure) {assert(ran && backend.healthy());assert(static_cast<uint16_t*>(out.contents())[0]==0x3f80);}
    else assert(!backend.healthy());
  }
  std::cout<<"PASS CPU producer ordering, join and failure propagation\n";
}
