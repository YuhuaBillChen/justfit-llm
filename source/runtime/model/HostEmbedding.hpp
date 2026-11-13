// Modified for JustFit: product identity and integration.
#pragma once
#include "model/WeightImages.hpp"
#include "model/WeightSource.hpp"
#include "model/Bfloat16.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <memory>
namespace justfit::model {
// CPU-only affine table. Package data stays a pageable read-only mapping;
// an upstream affine checkpoint maps its original tensors without a second table.
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
  // The owner retains the checkpoint's open files. Each tensor remains in its
  // original safetensors shard; mapping clean file pages makes them reclaimable.
  HostEmbedding(const ImagePlan &plan, std::string identity,
                std::shared_ptr<const void> owner, std::array<SourceTensor,3> tensors)
      : checkpoint_(std::move(owner)), tensors_(std::move(tensors)) {
    vocabulary_=plan.layer;hidden_=plan.type;
    if(!checkpoint_ || !vocabulary_ || !hidden_ || hidden_%64)
      throw WeightStoreError("invalid source host embedding layout");
    const uint64_t elements=uint64_t(vocabulary_)*hidden_;
    const std::array<uint64_t,3> sizes{elements/2,elements/32,elements/32};
    for(size_t i=0;i<tensors_.size();++i) {
      const auto &tensor=tensors_[i];
      const std::vector<uint64_t> shape{vocabulary_,i?hidden_/64:hidden_/8};
      if(!tensor.file || tensor.dtype!=(i?"BF16":"U32") || tensor.shape!=shape || tensor.bytes!=sizes[i])
        throw WeightStoreError("source host embedding tensor type or shape changed");
      views_[i]=mapTensor(tensor,mappings_[i]);
    }
    weights_=views_[0];scales_=reinterpret_cast<const uint16_t*>(views_[1]);biases_=reinterpret_cast<const uint16_t*>(views_[2]);
    record_={plan.component,plan.magic,plan.layer,plan.type,plan.bytes,std::move(identity)};
  }
  const WeightFileRecord &record() const noexcept {return record_;}
  void gather(std::span<const uint32_t> tokens,uint16_t *output) const {
    if(source_)source_->checkUnchanged();
    for(size_t i=0;i<tensors_.size();++i)
      if(tensors_[i].file && (i==0 || tensors_[i].file!=tensors_[i-1].file))tensors_[i].file->checkUnchanged();
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
  static const uint8_t *mapTensor(const SourceTensor &tensor,std::shared_ptr<void> &owner) {
    const auto &file=*tensor.file;file.checkUnchanged();
    if(file.dataOffset()>file.bytes() || tensor.offset>file.bytes()-file.dataOffset())
      throw WeightStoreError("source host embedding offset is out of bounds");
    const uint64_t offset=file.dataOffset()+tensor.offset;
    const long pageSize=sysconf(_SC_PAGESIZE);
    if(pageSize<=0 || offset%alignof(uint16_t) || tensor.bytes>file.bytes()-offset)
      throw WeightStoreError("source host embedding range is out of bounds");
    const uint64_t start=offset-offset%uint64_t(pageSize),delta=offset-start;
    if(start>uint64_t(std::numeric_limits<off_t>::max()) || tensor.bytes>std::numeric_limits<size_t>::max()-delta)
      throw WeightStoreError("source host embedding mapping is too large");
    const size_t bytes=size_t(tensor.bytes+delta);
    void *mapping=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,file.descriptor(),off_t(start));
    if(mapping==MAP_FAILED)throw WeightStoreError("source host embedding mapping failed");
    owner=std::shared_ptr<void>(mapping,[bytes](void *p){munmap(p,bytes);});
    return static_cast<const uint8_t*>(mapping)+delta;
  }
  std::shared_ptr<WeightSource> source_;
  std::shared_ptr<void> owner_;
  std::shared_ptr<const void> checkpoint_;
  std::array<SourceTensor,3> tensors_{};
  std::array<std::shared_ptr<void>,3> mappings_{};
  std::array<const uint8_t*,3> views_{};
  const uint8_t *bytes_=nullptr,*weights_=nullptr;
  const uint16_t *scales_=nullptr,*biases_=nullptr;
  uint32_t vocabulary_=0,hidden_=0;
  WeightFileRecord record_;
};
}
