// Modified for JustFit: product identity and integration.
#include "model/HostEmbedding.hpp"
#include "model/QwenTargetFiles.hpp"
#include "model/QwenHybridLayout.hpp"
#include "ops/Embedding.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
using namespace justfit;
int main(int argc,char **argv){
 assert(argc==3);metal::MetalBackend backend(argv[1]);
 const uint32_t vocabulary=4,hidden=5120;
 const uint64_t elems=uint64_t(vocabulary)*hidden,w=model::alignWeightOffset(16),s=model::alignWeightOffset(w+elems/2),b=model::alignWeightOffset(s+elems/32),n=model::alignWeightOffset(b+elems/32);
 std::vector<uint8_t> data(n,0);auto header=model::weightFileHeader(model::kEmbeddingMagic,vocabulary,hidden);std::copy(header.begin(),header.end(),data.begin());
 for(uint64_t i=0;i<elems/2;++i)data[w+i]=uint8_t(i*37+i/53);
 auto *scales=reinterpret_cast<uint16_t*>(data.data()+s),*biases=reinterpret_cast<uint16_t*>(data.data()+b);
 for(uint64_t i=0;i<elems/64;++i){scales[i]=uint16_t(0x3d00+(i%63)*3);biases[i]=uint16_t(0xbd00+(i%47)*5);}
 model::ImagePlan plan{"target/embedding.bin",std::string(model::kEmbeddingMagic),vocabulary,hidden,n,[&](std::span<uint8_t> destination,const metal::MetalBuffer&){std::copy_n(data.begin(),destination.size(),destination.begin());}};
 const auto baseline=backend.memoryStats().allocatedBytes;
 model::HostEmbedding memory(plan,"numeric-reference");
 std::ofstream file(argv[2],std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size());file.close();
 model::HostEmbedding mapping(plan,"numeric-reference",argv[2]);assert(backend.memoryStats().allocatedBytes==baseline);
 model::WeightImages images(backend,"numeric-reference");auto weights=images.load(plan);auto table=model::readAffineEmbedding(weights,vocabulary,hidden,"embedding");weights.finish();
 const std::array<uint32_t,7> ids{0,3,1,2,3,UINT32_MAX,4};
 auto tokens=backend.allocateBuffer(ids.size()*4,metal::BufferStorage::Shared,"test-tokens");std::memcpy(tokens.contents(),ids.data(),ids.size()*4);
 auto out=backend.allocateBuffer(ids.size()*hidden*2,metal::BufferStorage::Shared,"test-embedding");
 metal::CommandGraph graph;ops::Embedding::add(graph,tokens,table,out,ids.size());static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
 std::vector<uint16_t> cpu(ids.size()*hidden),mapped(cpu.size());memory.gather(ids,cpu.data());mapping.gather(ids,mapped.data());
 assert(cpu==mapped);assert(std::memcmp(cpu.data(),out.contents(),cpu.size()*2)==0);
 auto invalid=plan;invalid.bytes-=1;bool rejected=false;try{model::HostEmbedding bad(invalid,"invalid");}catch(const model::WeightStoreError&){rejected=true;}assert(rejected);
 std::cout<<"host heap/mapping vs Metal exact BF16 parity; row stride, bounds and malformed size PASS\n";
}
