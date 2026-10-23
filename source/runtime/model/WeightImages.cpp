// Modified for JustFit: product identity and integration.
#include "model/WeightImages.hpp"
#include "model/WeightSource.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace justfit::model {

WeightFile WeightImages::load(ImagePlan image) {
  if (released_) throw std::logic_error("weights load while their memory is released");
  metal::MetalBuffer buffer = backend_->allocateBuffer(image.bytes, metal::BufferStorage::Shared, image.component);
  image.write(contentsOf(buffer), buffer);
  WeightFile file(*backend_, buffer, image.component, image.magic, image.layer, image.type, contentIdentity_);
  images_.push_back({std::move(image.component), std::move(buffer), std::move(image.write)});
  return file;
}

void WeightImages::release() {
  if (released_) throw std::logic_error("weights are already released");
  for (Image &image : images_) {
    image.restoreOnWake = image.resident;
    if (image.resident) {
      backend_->releaseMemory(image.buffer);
      image.resident = false;
    }
  }
  released_ = true;
  restored_ = 0;
}

bool WeightImages::restore() {
  if (!released_) throw std::logic_error("weights are not released");
  while (restored_ < images_.size() && !images_[restored_].restoreOnWake)
    ++restored_;
  if (restored_ < images_.size()) {
    Image &image = images_[restored_];
    restoreImage(image);
    ++restored_;
  }
  while (restored_ < images_.size() && !images_[restored_].restoreOnWake)
    ++restored_;
  released_ = restored_ < images_.size();
  return !released_;
}

const WeightImages::Image &WeightImages::component(std::string_view name) const {
  for (const Image &image : images_)
    if (image.component == name) return image;
  throw std::out_of_range("unknown weight component: " + std::string(name));
}

WeightImages::Image &WeightImages::component(std::string_view name) {
  return const_cast<Image &>(std::as_const(*this).component(name));
}

bool WeightImages::componentResident(std::string_view name) const {
  return component(name).resident;
}

uint64_t WeightImages::componentBytes(std::string_view name) const {
  return component(name).buffer.sizeBytes();
}

uint64_t WeightImages::releaseComponent(std::string_view name) {
  if (released_) throw std::logic_error("component release during global idle");
  Image &image = component(name);
  if (!image.resident) return 0;
  backend_->releaseMemory(image.buffer);
  image.resident = false;
  return image.buffer.sizeBytes();
}

void WeightImages::restoreImage(Image &image) {
  backend_->restoreMemory(image.buffer);
  try {
    image.write(contentsOf(image.buffer), image.buffer);
  } catch (...) {
    // A partial write never becomes a usable component. Existing views retain
    // their identity and cannot bind until a later successful restore.
    backend_->releaseMemory(image.buffer);
    throw;
  }
  image.resident = true;
}

void WeightImages::restoreComponent(std::string_view name) {
  if (released_) throw std::logic_error("component restore during global idle");
  Image &image = component(name);
  if (!image.resident) restoreImage(image);
}

std::vector<WeightImages::Contents> WeightImages::contents() const {
  std::vector<Contents> result;
  for (const Image &image : images_) {
    const auto *bytes = static_cast<const uint8_t *>(image.buffer.contents());
    result.push_back({image.component, {bytes, bytes ? image.buffer.sizeBytes() : 0}});
  }
  return result;
}

ImagePlan packageImage(const std::filesystem::path &path, std::string component, std::string_view magic,
                       uint32_t layer, uint32_t type) {
  auto source = std::make_shared<WeightSource>(path);
  if (!source->bytes()) throw WeightStoreError("package file is empty: " + path.string());
  return {std::move(component), std::string(magic), layer, type, source->bytes(),
          [source](std::span<uint8_t> destination, const metal::MetalBuffer &) {
            parallelFor((destination.size() + kLoadStepBytes - 1) / kLoadStepBytes, [&](size_t index, unsigned) {
              const uint64_t at = index * kLoadStepBytes;
              source->readData(at, destination.subspan(at, std::min<uint64_t>(kLoadStepBytes, destination.size() - at)));
            });
            source->checkUnchanged();
          }};
}

unsigned loadThreads() noexcept { return std::max(1u, std::thread::hardware_concurrency()); }

void parallelFor(size_t count, const std::function<void(size_t index, unsigned thread)> &task) {
  struct Work {
    Work(const std::function<void(size_t, unsigned)> &task, size_t count) : task(task), count(count) {}
    const std::function<void(size_t, unsigned)> &task;
    const size_t count;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex failureMutex;
    std::exception_ptr failure;
  } work(task, count);
  // Each worker takes indices until none is left; libdispatch runs the
  // workers on its own threads.
  dispatch_apply_f(std::min<size_t>(count, loadThreads()), DISPATCH_APPLY_AUTO, &work, [](void *context, size_t thread) {
    auto &work = *static_cast<Work *>(context);
    for (size_t index; !work.failed.load(std::memory_order_relaxed) && (index = work.next++) < work.count;) {
      try {
        work.task(index, static_cast<unsigned>(thread));
      } catch (...) {
        std::lock_guard lock(work.failureMutex);
        if (!work.failure) work.failure = std::current_exception();
        work.failed = true;
      }
    }
  });
  if (work.failure) std::rethrow_exception(work.failure);
}

void zeroUnwritten(std::span<uint8_t> image, std::vector<std::pair<uint64_t, uint64_t>> extents) {
  std::sort(extents.begin(), extents.end());
  uint64_t written = 0;
  for (const auto &[offset, bytes] : extents) {
    if (offset > image.size() || bytes > image.size() - offset)
      throw std::out_of_range("weight image extent is out of bounds");
    if (offset > written) std::memset(image.data() + written, 0, offset - written);
    written = std::max(written, offset + bytes);
  }
  std::memset(image.data() + written, 0, image.size() - written);
}

} // namespace justfit::model
