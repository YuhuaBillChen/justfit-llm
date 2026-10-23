// Modified for JustFit: product identity and integration.
#include "engine/Scheduler.hpp"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace justfit::engine;
using justfit::WorkKind;
int main() {
 const std::string path="/tmp/justfit-barrier-"+std::to_string(getpid());
 std::ofstream(path)<<"hold";
 setenv("DASH_TEST_DECODE_HOLD_FILE",path.c_str(),1);
 Scheduler s(0);
 s.submit({.id=1,.prefillTokens=16,.deadlineMilliseconds=100000});s.resourcesReady(1,0);
 auto p=*s.next({});s.commit(p,{});s.complete(p,std::array<StepResult,1>{{{1,16,false}}},1,false);
 assert(!s.next({})); // Old native baseline decodes here: deliberate before-failure.
 s.observePrefill(2048,4096);
 s.submit({.id=2,.prefillTokens=8192,.deadlineMilliseconds=100000});s.resourcesReady(2,0);
 p=*s.next({});assert(p.kind==WorkKind::Prefill && p.prefillParts==1 && p.items[0].tokenCount==2048);
 while(s.snapshot().prefilling) {
  p=*s.next({});s.commit(p,{});s.complete(p,std::array<StepResult,1>{{{2,p.items[0].tokenCount,false}}},1,false);
 }
 assert(s.snapshot().decoding==2 && !s.next({}));
 assert(unlink(path.c_str())==0);
 auto d=*s.next({});assert(d.kind==WorkKind::Decode && d.items.size()==2);
 unsetenv("DASH_TEST_DECODE_HOLD_FILE");
 std::cout<<"held B2 prefill, zero decode, release B2 PASS\n";
}
