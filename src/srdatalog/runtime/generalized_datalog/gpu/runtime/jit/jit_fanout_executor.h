#pragma once

/**
 * @file jit_fanout_executor.h
 * @brief Fan-out work-stealing infrastructure for JIT-compiled Cartesian product rules
 *
 * Provides:
 * - FanOutTask: generic 2-source Cartesian product slice (extensible to N sources)
 * - FanOutTaskQueue: device ring buffer with atomic push/pop
 * - RootTaskQueue: GlobalTaskQueue for root-key work stealing
 * - FanOutExecuteJit: host-side orchestrator for the 3-phase fan-out pipeline
 *
 * The fan-out pipeline replaces the standard count+scan+materialize with:
 *   explore → allocate → fan-out materialize
 *
 * The explore kernel traverses the join tree (same as count) but at the
 * Cartesian join leaf, instead of counting flat indices, it:
 * 1. atomicAdds to global_total to get the pre-assigned output offset
 * 2. Splits the Cartesian product into FanOutTask chunks
 *
 * The fan-out materialize kernel pops FanOutTasks and writes directly
 * to pre-assigned contiguous output positions — zero atomics during writes.
 */

#include "gpu/gpu_api.h"
#include <cooperative_groups.h>
#include <cstdint>

namespace SRDatalog::GPU::JIT::FanOut {

namespace cg = cooperative_groups;

// =============================================================================
// Configuration
// =============================================================================

/// Number of flat-index iterations per FanOutTask chunk
static constexpr uint32_t FANOUT_CHUNK_SIZE = 4096;

/// Default queue capacity (512K tasks)
static constexpr uint32_t DEFAULT_FANOUT_QUEUE_CAPACITY = 512 * 1024;

// =============================================================================
// FanOutTask — self-contained Cartesian product slice (N-source)
// =============================================================================

/// Maximum number of sources in a CartesianJoin
static constexpr uint32_t MAX_FANOUT_SOURCES = 4;

/**
 * @brief A slice of an N-source Cartesian product with pre-assigned output position.
 *
 * During the explore phase, each Cartesian product leaf is decomposed into
 * one or more FanOutTasks. Each task carries the handle positions and degrees
 * needed to materialize its slice — no re-traversal of the join tree needed.
 *
 * Sources correspond to the N relations participating in the
 * CartesianJoin (moCartesianJoin.ctSources[0..N-1]).
 */
struct FanOutTask {
  uint32_t source_begin[MAX_FANOUT_SOURCES];   ///< Handle begin per source
  uint32_t source_degree[MAX_FANOUT_SOURCES];  ///< Degree per source
  uint32_t num_sources;                        ///< Number of active sources (2..MAX_FANOUT_SOURCES)
  uint32_t output_offset;                      ///< Pre-assigned output position (from atomicAdd)
  uint32_t flat_begin;                         ///< Start index in [0, total) flat space
  uint32_t flat_end;                           ///< End index in [0, total) flat space
};

// =============================================================================
// FanOutTaskQueue — device ring buffer with atomic push/pop
// =============================================================================

struct FanOutTaskQueue {
  FanOutTask* tasks;  ///< Device-allocated task array
  uint32_t capacity;  ///< Max tasks
  uint32_t head;      ///< Atomic pull counter
  uint32_t tail;      ///< Atomic push counter

  /// @brief Push a task (device only, from explore kernel)
  __device__ bool push(const FanOutTask& task) {
    uint32_t slot = atomicAdd(&tail, 1);
    if (slot < capacity) {
      tasks[slot] = task;
      return true;
    }
    return false;
  }

  /// @brief Pop a task (device only, from materialize kernel, leader thread)
  __device__ bool pop(FanOutTask& task) {
    uint32_t idx = atomicAdd(&head, 1);
    uint32_t current_tail = atomicAdd(&tail, 0);
    if (idx < current_tail) {
      task = tasks[idx];
      return true;
    }
    return false;
  }
};

// =============================================================================
// WCOJTask — mid-level donated join traversal task
// =============================================================================
//
// When a warp encounters a wide intermediate join node (e.g., h_RLIF has
// degree=500 at the sig level), it splits the range and donates the second
// half to this queue. The donated task carries all parent handle ranges
// so the receiver can resume the join traversal from the donated sub-range.

static constexpr uint32_t MAX_WCOJ_HANDLES = 8;
static constexpr uint32_t WCOJ_SPLIT_THRESHOLD = 64;
static constexpr uint32_t DEFAULT_WCOJ_QUEUE_CAPACITY = 256 * 1024;

struct WCOJTask {
  /// Handle ranges narrowed by parent levels (only [0..num_handles) are valid)
  uint32_t handle_begin[MAX_WCOJ_HANDLES];
  uint32_t handle_end[MAX_WCOJ_HANDLES];
  uint32_t num_handles;  ///< Number of valid handle ranges

  /// Which handle index to split on (the one that was too wide)
  uint32_t split_handle_idx;
  /// Sub-range within split_handle to process
  uint32_t split_begin;
  uint32_t split_end;
};

// =============================================================================
// WCOJTaskQueue — global device queue for mid-level work stealing
// =============================================================================

struct WCOJTaskQueue {
  WCOJTask* tasks;    ///< Device-allocated task array
  uint32_t capacity;  ///< Max tasks
  uint32_t head;      ///< Atomic pull counter
  uint32_t tail;      ///< Atomic push counter

  __device__ bool push(const WCOJTask& task) {
    uint32_t slot = atomicAdd(&tail, 1);
    if (slot < capacity) {
      tasks[slot] = task;
      return true;
    }
    return false;
  }

  __device__ bool pop(WCOJTask& task) {
    uint32_t idx = atomicAdd(&head, 1);
    uint32_t current_tail = atomicAdd(&tail, 0);
    if (idx < current_tail) {
      task = tasks[idx];
      return true;
    }
    return false;
  }
};

// =============================================================================
// RootTaskQueue — work-stealing queue for root-key ranges
// =============================================================================

struct RootTaskQueue {
  static constexpr uint32_t MAX_TASKS = 1024 * 64;

  struct Task {
    uint32_t begin;
    uint32_t end;
  };

  Task tasks[MAX_TASKS];
  uint32_t head;  ///< Atomic pull counter
  uint32_t tail;  ///< Atomic push counter

  /// @brief Push a task (host or device)
  __host__ __device__ bool push(const Task& task) {
    uint32_t slot;
#ifdef __CUDA_ARCH__
    slot = atomicAdd(&tail, 1);
#else
    slot = tail++;
#endif
    if (slot < MAX_TASKS) {
      tasks[slot] = task;
      return true;
    }
    return false;
  }

  /// @brief Pop a task (device only, leader thread)
  __device__ bool pop(Task& task) {
    uint32_t idx = atomicAdd(&head, 1);
    uint32_t current_tail = atomicAdd(&tail, 0);
    if (idx < current_tail) {
      task = tasks[idx];
      return true;
    }
    return false;
  }

  /// @brief Reset queue for reuse
  __host__ void reset() {
    head = 0;
    tail = 0;
  }
};

// =============================================================================
// Host-side helpers
// =============================================================================

/// @brief Initialize root-key task queue with per-key tasks
inline void init_root_key_tasks(RootTaskQueue& queue, uint32_t num_unique_root_keys,
                                uint32_t chunk_size = 1) {
  queue.reset();
  for (uint32_t i = 0; i < num_unique_root_keys; i += chunk_size) {
    RootTaskQueue::Task task;
    task.begin = i;
    task.end = (i + chunk_size < num_unique_root_keys) ? i + chunk_size : num_unique_root_keys;
    queue.push(task);
  }
}

// =============================================================================
// FanOutExecuteJit — host-side orchestrator for the fan-out pipeline
// =============================================================================

/**
 * @brief Execute a JIT rule using the fan-out work-stealing pipeline.
 *
 * This is called from the JIT orchestrator for rules with epUseFanOut=true.
 * The JitRunner must provide:
 * - setup(db, iter) → LaunchParams
 * - launch_explore(params, stream) → total_count
 * - launch_fanout_materialize(params, total_count, stream)
 *
 * @tparam JitRunner  The generated JitRunner_<RuleName> struct
 */
template <typename JitRunner>
struct FanOutExecuteJit {
  using ValueType = typename JitRunner::ValueType;
  using IterType = typename JitRunner::IterType;

  static void run(typename JitRunner::DeviceDBType& db, IterType& iter, GPU_STREAM_T stream) {
    // Phase 0: Setup views and root-key unique values
    auto params = JitRunner::setup(db, iter);

    // Phase 1: Explore — traverse join tree + push FanOutTasks
    auto total = JitRunner::launch_explore(params, stream);
    GPU_STREAM_SYNCHRONIZE(stream);

    if (total == 0) {
      JitRunner::cleanup(params);
      return;
    }

    // Phase 2: Resize destination relation(s)
    JitRunner::resize_outputs(params, total, db, stream);
    GPU_STREAM_SYNCHRONIZE(stream);

    // Phase 3: Fan-out materialize — pop tasks, flat-index iterate, direct writes
    JitRunner::launch_fanout_materialize(params, stream);
    GPU_STREAM_SYNCHRONIZE(stream);

    // Cleanup
    JitRunner::cleanup(params);
  }
};

}  // namespace SRDatalog::GPU::JIT::FanOut
