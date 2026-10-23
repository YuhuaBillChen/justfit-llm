// Modified for JustFit: product identity and integration.
#include "metal/HostWork.hpp"
#include <cassert>
#include <thread>
#include <stdexcept>
using justfit::metal::HostWork;
int main() {
  HostWork work(2); int notifications=0;
  work.afterDone([&]{++notifications;});
  work.complete(); assert(!work.done() && notifications==0);
  work.fail("slice failed"); work.complete();
  assert(work.done() && notifications==1 && work.failure()=="slice failed");
  work.afterDone([&]{++notifications;}); assert(notifications==2);
  HostWork race(1); std::thread worker([&]{race.complete();});
  race.afterDone([&]{++notifications;}); worker.join(); assert(notifications==3);
}
