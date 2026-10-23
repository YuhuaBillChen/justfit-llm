// Modified for JustFit: product identity and integration.
#include "ops/TQ4ExtentLayout.hpp"
#include "metal/abi/KvExtent.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
using namespace justfit::kv;
int main() {
  assert((!TQ4ExtentLayout{16,4,128}.valid()));
  assert((!TQ4ExtentLayout{0,4,256}.valid()));
  for (uint32_t heads : {2u, 4u}) {
    TQ4ExtentLayout layout{16,heads,256};
    assert(layout.valid());
    assert(layout.bytesPerModelPage() == 16ull*32*heads*260);
    Layout q8{16,heads,256,Format::Int8};
    assert(layout.bytesPerModelPage()*2 == q8.bytesPerModelPage());
    const auto pages=layout.extentPagesFor(8192);
    assert(pages && pages%layout.extentAlignmentPages()==0);
    assert(layout.extentPagesFor(layout.minimumExtentPages()-1)==0);
    auto data=uint32_t(layout.dataBytesPerLayerPage());
    auto norm=uint32_t(layout.scaleBytesPerLayerPage());
    const auto bytes=pages*layout.bytesPerModelPage();
    std::vector<std::byte> memory(2*bytes+2,std::byte{0x7f});
    uint64_t copied=0;
    for(uint32_t extent=0;extent<2;++extent)
      for(uint32_t layer=0;layer<16;++layer)
        for(uint32_t tensor=0;tensor<4;++tensor) {
          assert(justfit_kv_offset(pages,data,norm,layer,tensor,0)%65536==0);
          auto size=justfit_kv_page_bytes(data,norm,tensor);
          for(uint32_t page : {0u,pages-1}) {
            auto offset=justfit_kv_offset(pages,data,norm,layer,tensor,page);
            assert(offset+size<=bytes);
            std::vector<std::byte> original(size);
            for(size_t i=0;i<size;++i) original[i]=std::byte((i+layer*7+tensor*13+page+extent*53)%251);
            std::memcpy(memory.data()+1+extent*bytes+offset,original.data(),size);
            copied+=size;
          }
        }
    // Verify after every plane is written to detect overlapping regions.
    for(uint32_t extent=0;extent<2;++extent)
      for(uint32_t layer=0;layer<16;++layer)
        for(uint32_t tensor=0;tensor<4;++tensor)
          for(uint32_t page : {0u,pages-1}) {
            auto offset=justfit_kv_offset(pages,data,norm,layer,tensor,page);
            auto size=justfit_kv_page_bytes(data,norm,tensor);
            for(size_t i=0;i<size;++i)
              assert(memory[1+extent*bytes+offset+i]==std::byte((i+layer*7+tensor*13+page+extent*53)%251));
          }
    assert(memory.front()==std::byte{0x7f} && memory.back()==std::byte{0x7f});
    assert(copied==4*layout.bytesPerModelPage());
    std::cout << "heads=" << heads << " extent_pages=" << pages << " extent_bytes=" << bytes << " PASS\n";
  }
}
