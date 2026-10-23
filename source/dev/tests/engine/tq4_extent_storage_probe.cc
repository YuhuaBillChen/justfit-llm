// Modified for JustFit: product identity and integration.
#include "ops/TQ4ExtentStorage.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
int main(int argc, char** argv) {
  assert(argc==2);
  using namespace justfit;
  metal::MetalBackend backend(argv[1]);
  kv::TQ4ExtentLayout layout{16,4,256};
  auto n=layout.extentPagesFor(8192);
  kv::TQ4ExtentStorage storage(backend,[](uint64_t,const auto& allocate){allocate();return metal::AllocationResult{};},layout,n*2,n);
  assert(storage.allocateExtent(0));assert(storage.allocateExtent(1));
  auto source=storage.spans(n-1);auto dest=storage.spans(n);
  for(size_t i=0;i<source.size();++i) std::memset(source[i].data(),int(i+1),source[i].size());
  kv::PageCopy copy{n-1,n};storage.copyPages(std::span(&copy,1));
  for(size_t i=0;i<source.size();++i)assert(std::memcmp(source[i].data(),dest[i].data(),source[i].size())==0);
  auto table=backend.allocateBuffer(2*sizeof(JustFitKvPage),metal::BufferStorage::Shared,"tq4-probe-table");
  uint32_t pages[]={n-1,n};storage.writeEntries(pages,0,table);
  auto* entries=static_cast<JustFitKvPage*>(table.contents());
  assert((entries[0]&JUSTFIT_KV_PAGE_INDEX_MASK)==n-1);
  assert((entries[1]&JUSTFIT_KV_PAGE_INDEX_MASK)==0);
  assert((entries[0]&~uint64_t(JUSTFIT_KV_PAGE_INDEX_MASK))!=(entries[1]&~uint64_t(JUSTFIT_KV_PAGE_INDEX_MASK)));
  storage.releaseExtent(0);storage.releaseExtent(1);assert(storage.actualAllocatedBytes()==0);
  std::cout<<"TQ4 Metal allocation/copy/page-table/release PASS; no attention dispatch\n";
}
