// Modified for JustFit: product identity and integration.
#include "model/QwenState.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>
using namespace justfit;
void require(bool x,const char *s){if(!x)throw std::runtime_error(s);}
void fill(model::QwenStateStorage &s,unsigned lane,unsigned char v){
 auto paint=[&](const metal::MetalBuffer &b){std::memset(b.contents(),v,b.sizeBytes());};
 paint(s.current(lane).stateBase);for(auto &d:s.draft(lane)){paint(d.keys);paint(d.values);}
}
void check(model::QwenStateStorage &s,unsigned lane,unsigned char v){
 auto verify=[&](const metal::MetalBuffer &b){auto p=static_cast<const unsigned char *>(b.contents());require(std::all_of(p,p+b.sizeBytes(),[&](unsigned char x){return x==v;}),"disk state bytes changed while live lane mutated");};
 verify(s.current(lane).stateBase);for(auto &d:s.draft(lane)){verify(d.keys);verify(d.values);}
}
struct WorkerGate {
 std::promise<void> entered,release;
 std::shared_future<void> exit=release.get_future().share();
 bool released=false;
 void open(){if(!released){released=true;release.set_value();}}
 ~WorkerGate(){open();}
};
int main(int argc,char **argv){
 try {
  require(argc==2 || argc==3,"need original metallib [full]");metal::MetalBackend backend(argv[1]);
  const model::CompositeStateLayout layout=argc==3 ? model::CompositeStateLayout{{48,3,10240,48,128,128},{5,8,128}} : model::CompositeStateLayout{{2,3,16,2,8,8},{2,2,16}};
  bool admit=true;
  metal::AllocationAdmission admission=[&](uint64_t,const std::function<void()> &f)->metal::AllocationResult {if(!admit)return metal::AllocationFailure::EngineBudget;f();return {};};
  const auto bytes=model::SlotFile::slotBytesFor(layout.cachedBytes());
  auto quota=std::make_shared<model::DiskBudget>(bytes*4);
  auto file=std::make_shared<model::SlotFile>(bytes,quota);
  uint64_t before=backend.memoryStats().allocatedBytes;
  model::QwenStateStorage s(backend,admission,layout,file);
  require(backend.memoryStats().allocatedBytes==before && s.stagingBytes()==0,"persistent storage eagerly reserves full state staging");
  require(bool(s.tryActivateLane(0,1)) && bool(s.tryActivateLane(1,2)),"activate lanes");
  s.clearForColdStart(0);s.clearForColdStart(1);s.updateLengths(0,{32,0,32});
  fill(s,0,0x35);auto source=s.snapshot(0);require(bool(source),"snapshot");
  auto gate=std::make_shared<WorkerGate>();auto entered=gate->entered.get_future();
  auto blocked=file->synchronize([gate]{gate->entered.set_value();gate->exit.wait_for(std::chrono::seconds(10));});
  require(entered.wait_for(std::chrono::seconds(5))==std::future_status::ready,"worker gate not entered");
  std::weak_ptr<const model::QwenCompositeState> weak=source;
  auto ticket=source->offload([]{});require(bool(ticket),"offload rejected");source.reset();
  // The caller may drop the source immediately. A distinct live snapshot
  // must not reuse and overwrite its buffers before the disk worker reads.
  require(!weak.expired(),"offload did not retain immutable source until IO finished");
  fill(s,0,0x72);auto second=s.snapshot(0);require(bool(second),"second snapshot");
  gate->open();require(ticket->finish(),"offload failed");
  auto disk=ticket->state();auto restore=s.beginRestore(1,*disk,true,[]{},[]{});
  require(bool(restore) && restore->finish(),"disk restore failed");check(s,1,0x35);
  second.reset();ticket.reset();require(weak.expired(),"source retained after ticket completion");
  // Snapshot-to-disk copies live state; a subsequent lane mutation must not
  // change its pending write. It uses admitted snapshot memory, no global buffer.
  fill(s,0,0x28);auto live=s.snapshotToDisk(0,[]{});require(bool(live),"live snapshot rejected");fill(s,0,0x91);
  require(live->finish(),"live snapshot write failed");auto liveDisk=live->state();
  auto read=s.beginRestore(1,*liveDisk,true,[]{},[]{});require(bool(read)&&read->finish(),"live snapshot restore failed");check(s,1,0x28);
  // Cancelling a queued write drains IO before releasing its immutable source.
  auto cancelSource=s.snapshot(0);require(bool(cancelSource),"cancel snapshot");
  std::weak_ptr<const model::QwenCompositeState> cancelWeak=cancelSource;
  auto cancelGate=std::make_shared<WorkerGate>();auto cancelEntered=cancelGate->entered.get_future();
  auto cancelBlocked=file->synchronize([cancelGate]{cancelGate->entered.set_value();cancelGate->exit.wait_for(std::chrono::seconds(10));});
  require(cancelEntered.wait_for(std::chrono::seconds(5))==std::future_status::ready,"cancel worker gate");
  auto cancelled=cancelSource->offload([]{});require(bool(cancelled),"cancel offload rejected");cancelSource.reset();
  const bool pinned=!cancelWeak.expired();
  std::thread unblock([cancelGate]{std::this_thread::sleep_for(std::chrono::milliseconds(50));cancelGate->open();});
  cancelled.reset();unblock.join();require(pinned&&cancelWeak.expired(),"cancel did not drain and release source");
  // Persistence uses the same safe immutable-source path.
  auto persistSource=s.snapshot(0);require(bool(persistSource),"persist snapshot");
  auto persisted=persistSource->persist([]{});require(bool(persisted),"persist rejected");persistSource.reset();
  require(persisted->finish(),"persist write failed");persisted.reset();
  std::vector<std::shared_ptr<model::SlotFile::Slot>> held;
  while(auto slot=file->acquire())held.push_back(std::move(slot));
  const auto allocated=backend.memoryStats().allocatedBytes;require(!s.snapshotToDisk(0,[]{}),"full quota admitted snapshot");
  require(backend.memoryStats().allocatedBytes==allocated,"full quota allocated a needless snapshot");held.clear();
  while(s.releaseOneIdle(false)){}
  admit=false;require(!s.snapshot(0),"denied memory admitted snapshot");
  const auto fallbackAllocated=backend.memoryStats().allocatedBytes;
  const auto fallbackStart=std::chrono::steady_clock::now();
  auto fallback=s.snapshotToDisk(0,[]{});require(bool(fallback)&&fallback->ready()&&fallback->finish(),"zero-extra-RAM direct recovery failed");
  require(backend.memoryStats().allocatedBytes==fallbackAllocated,"direct recovery allocated state RAM");
  const auto fallbackMillis=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fallbackStart).count();
  std::cout<<"payload_bytes="<<layout.cachedBytes()<<" direct_fallback_ms="<<fallbackMillis<<" extra_state_allocation_bytes="<<(backend.memoryStats().allocatedBytes-fallbackAllocated)<<"\n";
  fill(s,0,0x43);auto fallbackDisk=fallback->state();
  auto fallbackRead=s.beginRestore(1,*fallbackDisk,true,[]{},[]{});
  require(bool(fallbackRead)&&fallbackRead->finish(),"fallback restore failed");check(s,1,0x91);
  const auto used=quota->usedBytes();fallbackDisk.reset();fallbackRead.reset();fallback.reset();
  auto drained=file->synchronize([]{});require(drained->wait(),"quota release drain");
  require(quota->usedBytes()<used,"direct recovery leaked disk quota");
  admit=true;s.releaseLane(0,1);s.releaseLane(1,2);
  std::cout<<"PASS: zero eager staging; pending IO source lifetime; full byte restore; live mutation isolation; quota/admission refusal; zero-extra-RAM direct recovery; cancel drain; persist\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
