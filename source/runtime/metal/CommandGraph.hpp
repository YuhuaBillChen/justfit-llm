// Modified for JustFit: product identity and integration.
#pragma once

#include "MetalBackend.hpp"
#include "ops/VerifyShape.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace justfit::metal {

// An ordered dispatch list for one command, with the event steps that order
// it against another agent (EventStep); MetalBackend splits a command into
// Metal command buffers at its event signals. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payloads_.back().data(), sizeof(Params)});
  }

  // Event steps after the dispatches added so far (EventStep): signal once
  // all earlier work has completed, or hold all later work until the event
  // reaches value.
  void signal(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Signal);
  }
  void wait(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Wait);
  }

  // The backend converts these markers to a CPU-specific shared event.
  void addHostTask(std::function<void()> task) {
    events_.push_back({dispatches_.size(), {}, 0, EventStep::Kind::HostStart,
                      std::make_shared<const std::function<void()>>(std::move(task))});
  }
  void addHostJoin() {events_.push_back({dispatches_.size(), {}, 0, EventStep::Kind::HostJoin});}

  void markLayerBoundary() {layerEnds_.push_back(dispatches_.size());}
  [[nodiscard]] const std::vector<size_t> &layerEnds() const noexcept {return layerEnds_;}
  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  // The dispatches alone; command() has the event steps too.
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }
  [[nodiscard]] Command command() const noexcept {
    return {dispatches_, events_};
  }

private:
  ComputeDispatch &push(std::string pipeline, std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch dispatch;
    if (ops::activeVerifyRows() == 16 &&
        (pipeline.starts_with("verify_attention_") || pipeline.starts_with("verify_gdn_") ||
         pipeline.starts_with("decode_sample_") || pipeline == "decode_accept_dflash" ||
         pipeline == "draft_context_kv_commit")) pipeline = "wide16::" + pipeline;
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    dispatches_.push_back(std::move(dispatch));
    return dispatches_.back();
  }

  void step(SharedEvent event, uint64_t value, EventStep::Kind kind) {
    events_.push_back({dispatches_.size(), std::move(event), value, kind});
  }

  std::vector<size_t> layerEnds_;
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
  std::vector<EventStep> events_;
};

} // namespace justfit::metal
