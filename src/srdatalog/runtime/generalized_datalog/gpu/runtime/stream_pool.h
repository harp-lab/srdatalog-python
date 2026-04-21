/**
 * @file stream_pool.h
 * @brief CUDA stream pool for parallel pipeline execution.
 *
 * @details Manages a set of CUDA streams for launching independent rules
 * on separate streams within a moParallelGroup. Streams are lazily created
 * and reused across iterations.
 */

#pragma once

#include "../gpu_api.h"

#include <vector>

namespace SRDatalog::GPU {

struct StreamPool {
  std::vector<GPU_STREAM_T> streams;
  std::vector<GPU_EVENT_T> events;

  /// Ensure at least n streams (and matching events) are available
  void ensure(int n) {
    while (static_cast<int>(streams.size()) < n) {
      GPU_STREAM_T s;
      GPU_STREAM_CREATE(&s);
      streams.push_back(s);
      GPU_EVENT_T e;
      GPU_EVENT_CREATE(&e);
      events.push_back(e);
    }
  }

  /// Get stream i (modular if i >= size)
  GPU_STREAM_T get(int i) {
    if (streams.empty()) {
      ensure(1);
    }
    return streams[i % streams.size()];
  }

  /// Record an event on stream i (call after materialize)
  void record_event(int i) {
    GPU_EVENT_RECORD(events[i], streams[i]);
  }

  /// Make target_stream wait for stream i's recorded event
  void wait_event(int i, GPU_STREAM_T target_stream = 0) {
    GPU_STREAM_WAIT_EVENT(target_stream, events[i], 0);
  }

  /// Synchronize all managed streams
  void sync_all() {
    for (auto& s : streams) {
      GPU_STREAM_SYNCHRONIZE(s);
    }
  }

  ~StreamPool() {
    for (auto& e : events) {
      GPU_EVENT_DESTROY(e);
    }
    for (auto& s : streams) {
      GPU_STREAM_DESTROY(s);
    }
    events.clear();
    streams.clear();
  }

  // Non-copyable, movable
  StreamPool() = default;
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;
  StreamPool(StreamPool&& other) noexcept
      : streams(std::move(other.streams)), events(std::move(other.events)) {}
  StreamPool& operator=(StreamPool&& other) noexcept {
    if (this != &other) {
      for (auto& e : events)
        GPU_EVENT_DESTROY(e);
      for (auto& s : streams)
        GPU_STREAM_DESTROY(s);
      streams = std::move(other.streams);
      events = std::move(other.events);
    }
    return *this;
  }
};

}  // namespace SRDatalog::GPU
