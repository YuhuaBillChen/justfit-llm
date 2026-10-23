// Modified for JustFit: product identity and integration.
#pragma once
#include "model/WeightImages.hpp"
#include "model/WeightSource.hpp"
#include "model/Bfloat16.hpp"
#include <sys/mman.h>
#include <bit>
#include <cstring>
#include <memory>
namespace justfit::model {
// CPU-only affine table. Package data stays a pageable read-only mapping;
// an upstream affine checkpoint writes the same image into CPU memory.
class HostEmbedding final {
public:
  HostEmbedding(ImagePlan plan, std::string identity, const std::filesystem::path &path = {}) {
    if (!path.empty()) {
      source_=std::make_shared<WeightSource>(path);
      if(source_->bytes()!=plan.bytes) throw WeightStoreError("host embedding size changed");
      void *p=mmap(nullptr,plan.bytes,PROT_READ,MAP_PRIVATE,source_->descriptor(),0);
      if(p==MAP_FAILED) throw WeightStoreError("host embedding mapping failed");
      owner_=std::shared_ptr<void>(p,[n=plan.bytes](void *v){munmap(v,n);});
      bytes_=static_cast<const uint8_t*>(p);
    } else {
      auto data=std::make_shared<std::vector<uint8_t>>(plan.bytes);
      plan.write(*data,{}); bytes_=data->data(); owner_=data;
    }
    vocabulary_=plan.layer; hidden_=plan.type;
    const auto header=weightFileHeader(plan.magic,vocabulary_,hidden_);
    if(plan.bytes%kWeightFileAlignment || plan.bytes<header.size() || std::memcmp(bytes_,header.data(),header.size()) || !hidden_ || hidden_%64 || !vocabulary_)
      throw WeightStoreError("invalid host embedding header");
    const uint64_t elements=uint64_t(vocabulary_)*hidden_;
    const uint64_t packed=elements/2,params=elements/32;
    const uint64_t w=alignWeightOffset(16),s=alignWeightOffset(w+packed),b=alignWeightOffset(s+params);
    if(b>plan.bytes || params>plan.bytes-b || alignWeightOffset(b+params)!=plan.bytes)
      throw WeightStoreError("truncated host embedding table");
    weights_=bytes_+w;scales_=reinterpret_cast<const uint16_t*>(bytes_+s);biases_=reinterpret_cast<const uint16_t*>(bytes_+b);
    record_={plan.component,plan.magic,plan.layer,plan.type,plan.bytes,std::move(identity)};
  }
  const WeightFileRecord &record() const noexcept {return record_;}
  void gather(std::span<const uint32_t> tokens,uint16_t *output) const {
    if(source_)source_->checkUnchanged();
    const uint32_t groups=hidden_/64;
    for(uint32_t raw:tokens) {
      uint64_t token=raw<vocabulary_?raw:0;
      const uint8_t *packed=weights_+token*(hidden_/2);
      for(uint32_t g=0;g<groups;++g) {
        float scale=std::bit_cast<float>(uint32_t(scales_[token*groups+g])<<16);
        float bias=std::bit_cast<float>(uint32_t(biases_[token*groups+g])<<16);
        for(uint32_t pair=0;pair<32;++pair) {
          uint8_t q=packed[g*32+pair];
          for(uint32_t half=0;half<2;++half) {
            float value=float(half?q>>4:q&15)*scale+bias;
            uint32_t bits=std::bit_cast<uint32_t>(value);
            output[g*64+pair*2+half]=uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
          }
        }
      }
      output+=hidden_;
    }
  }
private:
  std::shared_ptr<WeightSource> source_;
  std::shared_ptr<void> owner_;
  const uint8_t *bytes_=nullptr,*weights_=nullptr;
  const uint16_t *scales_=nullptr,*biases_=nullptr;
  uint32_t vocabulary_=0,hidden_=0;
  WeightFileRecord record_;
};
}
