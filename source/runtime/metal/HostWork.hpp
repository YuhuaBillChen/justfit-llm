// Modified for JustFit: product identity and integration.
#pragma once
#include <functional>
#include <mutex>
#include <string>
#include <utility>
namespace justfit::metal {
// A completed GPU command must not retire CPU-visible allocations while a
// failed command's CPU work is still draining. All reports precede retirement.
class HostWork final {
public:
  explicit HostWork(size_t count):pending_(count),total_(count){}
  bool done() const {std::lock_guard lock(mutex_);return pending_==0;}
  size_t total() const noexcept {return total_;}
  std::string failure() const {std::lock_guard lock(mutex_);return error_;}
  void fail(std::string error){std::lock_guard lock(mutex_);if(error_.empty())error_=std::move(error);}
  void afterDone(std::function<void()> callback) {
    {std::lock_guard lock(mutex_);if(pending_){callback_=std::move(callback);return;}}
    callback();
  }
  void complete() {
    std::function<void()> callback;
    {std::lock_guard lock(mutex_);if(!pending_)return;if(--pending_==0)callback=std::move(callback_);}
    if(callback)callback();
  }
private:
  mutable std::mutex mutex_;
  size_t pending_,total_;
  std::string error_;
  std::function<void()> callback_;
};
}
