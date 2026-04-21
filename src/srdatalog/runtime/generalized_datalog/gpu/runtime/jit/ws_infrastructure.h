// Work-Stealing Infrastructure for WCOJ JIT Kernels
//
// Extracted from the VPT_LoadField prototype (ws_wcoj_kernel.cuh).
// Provides reusable building blocks for work-stealing in generated kernels:
//   - WCOJTask: resumable task descriptor carrying handle state + split level
//   - WCOJTaskQueue: lock-free MPMC queue for task donation/stealing
//   - find_value_aligned_split: value-boundary-aware handle bisection
//   - ChunkedOutputContext: batched atomic writes for materialize
//
// Usage: JIT-generated kernels #include this header and use these primitives
// directly. The pragma `work_stealing: true` triggers generation of split
// logic and steal loops around the standard join body.

#pragma once

#include "gpu/search.h"  // for group_exp_upper_bound
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace SRDatalog::GPU::JIT::WS {

// ============================================================================
// Constants (tunable)
// ============================================================================
static constexpr uint32_t WCOJ_MAX_HANDLES = 8;  // max handles per task (generous for deep joins)
static constexpr uint32_t WCOJ_MAX_BOUND_VARS =
    4;  // max bound variables carried across join levels
static constexpr uint32_t WCOJ_SPLIT_THRESHOLD = 256;
static constexpr uint32_t CUM_WORK_THRESHOLD = 4096;
static constexpr uint32_t CARTESIAN_SPLIT_THRESHOLD = 32768;
static constexpr uint32_t WCOJ_QUEUE_CAPACITY = 1 << 20;  // 1M task slots

// ============================================================================
// WCOJTask: carries enough state to resume a join from a split point
// ============================================================================
// The split_level field encodes which join level to resume from.
// Level 0 = unused, 1..N-1 = ColumnJoin levels, N = CartesianJoin (flat range).
//
// For ColumnJoin tasks (split_level 1..N-1):
//   - handles[0..num_handles-1] carry the frozen state of all relevant handles
//   - split_handle_idx identifies which handle was split
//   - split_begin/split_end define the donated sub-range
//
// For CartesianJoin tasks (split_level = N):
//   - handles carry the leaf handles involved in the Cartesian product
//   - split_begin/split_end define the flat index sub-range [start, end)
struct WCOJTask {
  uint32_t handle_begin[WCOJ_MAX_HANDLES];
  uint32_t handle_end[WCOJ_MAX_HANDLES];
  uint32_t handle_depth[WCOJ_MAX_HANDLES];
  uint32_t handle_view_slot[WCOJ_MAX_HANDLES];  // view slot index for each handle (multi-segment
                                                // support)
  uint32_t num_handles;

  uint32_t split_handle_idx;
  uint32_t split_begin;
  uint32_t split_end;
  uint32_t split_depth;

  uint32_t split_level;

  // Variables bound at earlier join levels, needed by downstream emit/filter.
  // E.g., for L3 Cartesian tasks, bound_vars carries values resolved at L2.
  uint32_t bound_vars[WCOJ_MAX_BOUND_VARS];
  uint32_t num_bound_vars;
};

// ============================================================================
// WCOJRangeBoard: shared-cursor work ranges for ColumnJoin steal-by-atomicAdd
// ============================================================================
// Producer pushes a range [begin, end) with chunk_size T.
// Consumer atomicAdds cursor by T to grab [cursor, cursor+T).
// No re-push needed — O(1) per steal, no CAS contention.
static constexpr uint32_t WCOJ_RANGE_BOARD_CAPACITY = 4096;

struct WCOJRange {
  // Immutable after publish (read by all consumers)
  WCOJTask task;       // full task context (handles, split_level, etc.)
  uint32_t range_end;  // end of the range

  // Atomic cursor — consumers grab chunks by atomicAdd
  cuda::atomic<uint32_t, cuda::thread_scope_device> cursor;
};

struct WCOJRangeBoard {
  WCOJRange* ranges;
  uint32_t capacity;
  cuda::atomic<uint32_t, cuda::thread_scope_device> count{0};

  /// Producer: publish a range. Returns slot index, or UINT32_MAX if full.
  __device__ uint32_t publish(const WCOJTask& task_template, uint32_t range_begin,
                              uint32_t range_end) {
    uint32_t slot = count.fetch_add(1, cuda::memory_order_relaxed);
    if (slot >= capacity)
      return UINT32_MAX;
    ranges[slot].task = task_template;
    ranges[slot].range_end = range_end;
    // Initialize cursor to range_begin — first consumer starts here
    ranges[slot].cursor.store(range_begin, cuda::memory_order_release);
    return slot;
  }

  /// Consumer: try to grab a chunk of size T from range at slot.
  /// Returns true if work was grabbed (out_begin/out_end set).
  /// Returns false if range is exhausted.
  __device__ bool grab(uint32_t slot, uint32_t chunk_size, uint32_t& out_begin, uint32_t& out_end) {
    uint32_t my_begin = ranges[slot].cursor.fetch_add(chunk_size, cuda::memory_order_relaxed);
    if (my_begin >= ranges[slot].range_end)
      return false;
    out_begin = my_begin;
    out_end = my_begin + chunk_size;
    if (out_end > ranges[slot].range_end)
      out_end = ranges[slot].range_end;
    return true;
  }

  __host__ __device__ uint32_t size() const {
    return count.load(cuda::memory_order_relaxed);
  }
};

// ============================================================================
// WCOJTaskQueue: lock-free MPMC queue with per-slot ready flags
// ============================================================================
// Push: atomic tail increment → write data → set ready flag (release)
// Pop: CAS head → spin on ready flag (acquire) → read data
// Simple and proven to work correctly on GPU with high throughput.
struct WCOJTaskQueue {
  WCOJTask* tasks;
  cuda::atomic<uint32_t, cuda::thread_scope_device>* ready_flags;
  uint32_t capacity;
  cuda::atomic<uint32_t, cuda::thread_scope_device> head{0};
  cuda::atomic<uint32_t, cuda::thread_scope_device> tail{0};
  cuda::atomic<uint32_t, cuda::thread_scope_device> active_producers{0};

  __device__ bool push(const WCOJTask& task) {
    // CAS loop to avoid leaking tail on overflow.
    // Old code did fetch_add(1) then checked capacity — on overflow, tail was
    // incremented but no ready_flag was set, causing pop() to deadlock.
    uint32_t slot = tail.load(cuda::memory_order_relaxed);
    while (true) {
      if (slot >= capacity)
        return false;  // full — don't touch tail
      if (tail.compare_exchange_weak(slot, slot + 1, cuda::memory_order_relaxed,
                                     cuda::memory_order_relaxed))
        break;  // won slot
      // CAS failed, slot reloaded — retry
    }
    tasks[slot] = task;
    // Ensure task data is visible before marking as ready
    ready_flags[slot].store(1, cuda::memory_order_release);
    return true;
  }

  __device__ bool pop(WCOJTask& task) {
    uint32_t current_head = head.load(cuda::memory_order_relaxed);
    while (true) {
      uint32_t current_tail = tail.load(cuda::memory_order_acquire);
      if (current_head >= current_tail) {
        return false;  // queue empty
      }
      if (head.compare_exchange_weak(current_head, current_head + 1, cuda::memory_order_relaxed,
                                     cuda::memory_order_relaxed)) {
        // Won this slot — wait for data to be ready
        while (ready_flags[current_head].load(cuda::memory_order_acquire) == 0) {
        }
        task = tasks[current_head];
        return true;
      }
      // CAS failed — current_head was reloaded, retry
    }
  }

  __host__ __device__ uint32_t size() const {
    uint32_t h = head.load(cuda::memory_order_relaxed);
    uint32_t t = tail.load(cuda::memory_order_relaxed);
    return (t > h) ? (t - h) : 0;
  }
};

// ============================================================================
// find_value_aligned_split: bisect a handle at a value boundary
// ============================================================================
// Given a handle [begin, end) at a certain depth in a view, find a value-aligned
// split point near the midpoint. Returns the split position (upper_bound of
// the midpoint value), or 0 if the split is degenerate (all same value).
template <typename ViewType, typename HandleType, typename Tile>
__device__ uint32_t find_value_aligned_split(Tile& tile, const HandleType& handle,
                                             const ViewType& view) {
  uint32_t degree = handle.degree();
  if (degree <= 1)
    return 0;

  uint32_t raw_mid = handle.begin() + degree / 2;

  // Read the value at midpoint
  auto mid_val = view.get_value(handle.depth(), raw_mid);

  // Find upper_bound(mid_val) in [raw_mid, end) to snap to value boundary
  const auto* col = view.col_data() + handle.depth() * view.stride_;
  uint32_t remaining = handle.end() - raw_mid;
  uint32_t ub = SRDatalog::GPU::group_exp_upper_bound<typename ViewType::ValueType, uint32_t>(
      col + raw_mid, remaining, mid_val, tile);
  uint32_t split_pos = raw_mid + ub;

  // Guard: split must leave work on both sides
  if (split_pos <= handle.begin() || split_pos >= handle.end()) {
    return 0;  // degenerate — all values the same, can't split
  }
  return split_pos;
}

// ============================================================================
// ChunkedOutputContext: batched atomic writes for non-deterministic WS materialize
// ============================================================================
// Each thread buffers tuples locally. On buffer full, atomicAdds to a global
// write position counter and flushes the buffer to pre-allocated output arrays.
// Handles non-deterministic steal ordering: any warp can write anywhere.
//
// Arity is the number of output columns per tuple.
// ChunkSize controls the buffer size per thread (default 64).
template <typename ValueType, int Arity, int ChunkSize = 64>
struct ChunkedOutputContext {
  ValueType* global_data;      // SoA output buffer base
  uint32_t* global_write_pos;  // Global atomic write position counter
  uint32_t output_stride;      // Stride for SoA layout
  uint32_t old_size;           // Base offset (existing data in output)

  ValueType local_buffer[ChunkSize][Arity];
  uint32_t local_count;

  __device__ ChunkedOutputContext(ValueType* data, uint32_t* write_pos, uint32_t stride,
                                  uint32_t base_offset)
      : global_data(data), global_write_pos(write_pos), output_stride(stride),
        old_size(base_offset), local_count(0) {}

  // Emit a single tuple (variadic-safe for any arity)
  template <typename... Args>
  __device__ void emit_direct(Args... args) {
    static_assert(sizeof...(Args) == Arity, "emit_direct: wrong number of columns");
    ValueType vals[] = {static_cast<ValueType>(args)...};
    for (int j = 0; j < Arity; ++j) {
      local_buffer[local_count][j] = vals[j];
    }
    local_count++;
    if (local_count == ChunkSize) {
      flush();
    }
  }

  // Warp-cooperative coalesced emit for Cartesian flat loops.
  // All threads in the tile call this together. Each thread provides its own values.
  // valid=true means this thread has a real tuple to write.
  // One atomicAdd per warp, coalesced stores: thread k writes to base+k.
  template <typename Tile, typename... Args>
  __device__ void emit_warp_coalesced(Tile tile, bool valid, Args... args) {
    static_assert(sizeof...(Args) == Arity, "emit_warp_coalesced: wrong number of columns");
    // Count valid threads in warp
    uint32_t ballot = tile.ballot(valid);
    uint32_t active_count = __popc(ballot);
    if (active_count == 0)
      return;
    // Compute this thread's position within the valid set
    uint32_t my_mask = (1u << tile.thread_rank()) - 1u;
    uint32_t my_offset = __popc(ballot & my_mask);
    // Thread 0 allocates for the whole warp
    uint32_t base;
    if (tile.thread_rank() == 0) {
      base = atomicAdd(global_write_pos, active_count);
    }
    base = tile.shfl(base, 0) + old_size;
    // Coalesced write: each valid thread writes to base + its offset
    if (valid) {
      ValueType vals[] = {static_cast<ValueType>(args)...};
      for (int j = 0; j < Arity; ++j) {
        global_data[j * output_stride + base + my_offset] = vals[j];
      }
    }
  }

  __device__ void flush() {
    if (local_count == 0)
      return;
    uint32_t write_idx = atomicAdd(global_write_pos, local_count);
    write_idx += old_size;
    for (uint32_t i = 0; i < local_count; ++i) {
      for (int j = 0; j < Arity; ++j) {
        global_data[j * output_stride + write_idx + i] = local_buffer[i][j];
      }
    }
    local_count = 0;
  }
};

// ============================================================================
// SpeculativeOutputContext: fused count+materialize with overflow detection
// ============================================================================
template <typename ValueType, int Arity, int ChunkSize = 16>
struct SpeculativeOutputContext {
  ValueType* global_data;
  uint32_t* global_write_pos;
  uint32_t* overflow_flag;
  uint32_t output_stride;
  uint32_t old_size;
  uint32_t capacity;

  ValueType local_buffer[ChunkSize][Arity];
  uint32_t local_count;

  __device__ SpeculativeOutputContext(ValueType* data, uint32_t* write_pos, uint32_t* overflow,
                                      uint32_t stride, uint32_t base_offset, uint32_t cap)
      : global_data(data), global_write_pos(write_pos), overflow_flag(overflow),
        output_stride(stride), old_size(base_offset), capacity(cap), local_count(0) {}

  template <typename... Args>
  __device__ void emit_direct(Args... args) {
    static_assert(sizeof...(Args) == Arity, "emit_direct: wrong number of columns");
    if (*overflow_flag)
      return;
    ValueType vals[] = {static_cast<ValueType>(args)...};
    for (int j = 0; j < Arity; ++j) {
      local_buffer[local_count][j] = vals[j];
    }
    local_count++;
    if (local_count == ChunkSize) {
      flush();
    }
  }

  __device__ void emit_direct() {
    local_count++;
    if (local_count == ChunkSize) {
      flush();
    }
  }

  __device__ void flush() {
    if (local_count == 0)
      return;
    uint32_t write_idx = atomicAdd(global_write_pos, local_count);
    if (write_idx + local_count > capacity) {
      atomicExch(overflow_flag, 1u);
      local_count = 0;
      return;
    }
    write_idx += old_size;
    for (uint32_t i = 0; i < local_count; ++i) {
      for (int j = 0; j < Arity; ++j) {
        global_data[j * output_stride + write_idx + i] = local_buffer[i][j];
      }
    }
    local_count = 0;
  }

  __device__ uint32_t count() const {
    return local_count;
  }
};

// ============================================================================
// WarpCoalescedOutputContext: zero-buffer warp-cooperative output for WS kernels
// ============================================================================
// Unlike ChunkedOutputContext, this has NO local_buffer — saves ~32 registers.
// All writes go through emit_warp_coalesced() which does one atomicAdd per warp
// and coalesced stores directly to global memory.
template <typename ValueType, int Arity>
struct WarpCoalescedOutputContext {
  ValueType* global_data;      // SoA output buffer base
  uint32_t* global_write_pos;  // Global atomic write position counter
  uint32_t output_stride;      // Stride for SoA layout
  uint32_t old_size;           // Base offset (existing data in output)

  __device__ WarpCoalescedOutputContext(ValueType* data, uint32_t* write_pos, uint32_t stride,
                                        uint32_t base_offset)
      : global_data(data), global_write_pos(write_pos), output_stride(stride),
        old_size(base_offset) {}

  // Warp-cooperative coalesced emit: one atomicAdd per warp, coalesced stores.
  template <typename Tile, typename... Args>
  __device__ void emit_warp_coalesced(Tile tile, bool valid, Args... args) {
    static_assert(sizeof...(Args) == Arity, "emit_warp_coalesced: wrong number of columns");
    uint32_t ballot = tile.ballot(valid);
    uint32_t active_count = __popc(ballot);
    if (active_count == 0)
      return;
    uint32_t my_mask = (1u << tile.thread_rank()) - 1u;
    uint32_t my_offset = __popc(ballot & my_mask);
    uint32_t base;
    if (tile.thread_rank() == 0) {
      base = atomicAdd(global_write_pos, active_count);
    }
    base = tile.shfl(base, 0) + old_size;
    if (valid) {
      ValueType vals[] = {static_cast<ValueType>(args)...};
      for (int j = 0; j < Arity; ++j) {
        global_data[j * output_stride + base + my_offset] = vals[j];
      }
    }
  }

  // No-op flush (no buffer to flush)
  __device__ void flush() {}
};

// ============================================================================
// TiledCartesianHelper: pre-load Cartesian source columns into shared memory,
// then iterate from smem. Reduces global reads from D0*D1 to D0+D1 per column.
//
// Shared memory layout: each warp gets its own tile region.
// Total smem needed: WarpsPerBlock * 2 * TileSize * sizeof(ValueType)
//   e.g. 8 warps * 2 * 128 * 4B = 8KB
//
// Usage in kernel (manual patch for benchmarking):
//   // At kernel scope, declare shared memory:
//   constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;
//   constexpr int kCartTileSize = 128;
//   __shared__ ValueType s_cart_tiles[kWarpsPerBlock][2][kCartTileSize];
//   uint32_t warp_in_block = threadIdx.x / kGroupSize;
//
//   // Replace Cartesian loop with:
//   TiledCartesianHelper<ValueType, kCartTileSize>::run(
//       tile, view0, col0, h0.begin(), D0,
//              view1, col1, h1.begin(), D1,
//       s_cart_tiles[warp_in_block][0],
//       s_cart_tiles[warp_in_block][1],
//       [&](ValueType val0, ValueType val1) {
//         // filter + emit here
//       });
// ============================================================================
template <typename ValueType, int TileSize = 128>
struct TiledCartesianHelper {

  // 2-source Cartesian with shared-memory pre-load.
  // Callback: f(val_from_src0, val_from_src1) called for each valid pair.
  template <typename Tile, typename ViewType, typename Func>
  __device__ static void run(
      Tile tile, const ViewType& view0, int col0, uint32_t begin0, uint32_t deg0,
      const ViewType& view1, int col1, uint32_t begin1, uint32_t deg1,
      ValueType* __restrict__ smem0,  // per-warp shared mem tile for source 0
      ValueType* __restrict__ smem1,  // per-warp shared mem tile for source 1
      Func f) {

    uint32_t lane = tile.thread_rank();
    uint32_t gs = tile.size();

    // Process in tiles of TileSize for each source
    for (uint32_t t0_base = 0; t0_base < deg0; t0_base += TileSize) {
      uint32_t t0_end = min(t0_base + (uint32_t)TileSize, deg0);
      uint32_t t0_len = t0_end - t0_base;

      // Cooperatively load source 0 tile into shared memory (coalesced)
      for (uint32_t i = lane; i < t0_len; i += gs) {
        smem0[i] = view0.get_value(col0, begin0 + t0_base + i);
      }

      for (uint32_t t1_base = 0; t1_base < deg1; t1_base += TileSize) {
        uint32_t t1_end = min(t1_base + (uint32_t)TileSize, deg1);
        uint32_t t1_len = t1_end - t1_base;

        // Cooperatively load source 1 tile into shared memory (coalesced)
        for (uint32_t i = lane; i < t1_len; i += gs) {
          smem1[i] = view1.get_value(col1, begin1 + t1_base + i);
        }
        tile.sync();  // ensure all loads visible

        // Iterate Cartesian product from shared memory - zero global reads
        uint32_t total = t0_len * t1_len;
        for (uint32_t flat = lane; flat < total; flat += gs) {
          uint32_t i0 = flat / t1_len;
          uint32_t i1 = flat % t1_len;
          f(smem0[i0], smem1[i1]);
        }
        tile.sync();  // protect smem before next tile load
      }
    }
  }
};

// ============================================================================
// TiledCartesianWriter: tiled reads + ATOMIC-FREE coalesced writes for baseline
// materialize kernels. Uses the known per-warp write offset from the count
// phase's prefix scan (thread_offsets[lane0]) instead of atomicAdd.
//
// warp_write_base: broadcast of thread_offsets[lane0] — the warp's exclusive offset
// warp_local_count: mutable register tracking cumulative writes by this warp
//
// Shared memory layout per warp: same as TiledCartesianHelper (2 × TileSize values)
//
// Usage in kernel:
//   uint32_t warp_write_base = tile.shfl(thread_offset, 0);
//   uint32_t warp_local_count = 0;
//   TiledCartesianWriter<ValueType, kCartTileSize>::run(
//       tile, view0, col0, h0.begin(), D0, view1, col1, h1.begin(), D1,
//       s_cart[warp_in_block][0], s_cart[warp_in_block][1],
//       warp_write_base, warp_local_count, old_size,
//       [&](ValueType v0, ValueType v1) { return filter; },
//       [&](ValueType v0, ValueType v1, uint32_t pos) { write cols at pos; });
// ============================================================================
template <typename ValueType, int TileSize = 128>
struct TiledCartesianWriter {

  template <typename Tile, typename ViewType, typename FilterFunc, typename WriteFunc>
  __device__ static void run(Tile tile, const ViewType& view0, int col0, uint32_t begin0,
                             uint32_t deg0, const ViewType& view1, int col1, uint32_t begin1,
                             uint32_t deg1, ValueType* __restrict__ smem0,
                             ValueType* __restrict__ smem1, uint32_t warp_write_base,
                             uint32_t& warp_local_count, uint32_t old_size, FilterFunc filter,
                             WriteFunc write) {

    uint32_t lane = tile.thread_rank();
    uint32_t gs = tile.size();

    for (uint32_t t0_base = 0; t0_base < deg0; t0_base += TileSize) {
      uint32_t t0_end = min(t0_base + (uint32_t)TileSize, deg0);
      uint32_t t0_len = t0_end - t0_base;

      for (uint32_t i = lane; i < t0_len; i += gs) {
        smem0[i] = view0.get_value(col0, begin0 + t0_base + i);
      }

      for (uint32_t t1_base = 0; t1_base < deg1; t1_base += TileSize) {
        uint32_t t1_end = min(t1_base + (uint32_t)TileSize, deg1);
        uint32_t t1_len = t1_end - t1_base;

        for (uint32_t i = lane; i < t1_len; i += gs) {
          smem1[i] = view1.get_value(col1, begin1 + t1_base + i);
        }
        tile.sync();

        uint32_t total = t0_len * t1_len;
        for (uint32_t batch = 0; batch < total; batch += gs) {
          uint32_t flat = batch + lane;
          bool valid = flat < total;

          ValueType v0{}, v1{};
          if (valid) {
            uint32_t i0 = flat / t1_len;
            uint32_t i1 = flat % t1_len;
            v0 = smem0[i0];
            v1 = smem1[i1];
            valid = filter(v0, v1);
          }

          // Atomic-free warp-coalesced write using known base offset
          uint32_t ballot = tile.ballot(valid);
          uint32_t active_count = __popc(ballot);
          if (active_count > 0) {
            uint32_t my_mask = (1u << lane) - 1u;
            uint32_t my_offset = __popc(ballot & my_mask);
            if (valid) {
              write(v0, v1, old_size + warp_write_base + warp_local_count + my_offset);
            }
            warp_local_count += active_count;
          }
        }
        tile.sync();
      }
    }
  }

  // Convenience: no filter
  template <typename Tile, typename ViewType, typename WriteFunc>
  __device__ static void run_no_filter(Tile tile, const ViewType& view0, int col0, uint32_t begin0,
                                       uint32_t deg0, const ViewType& view1, int col1,
                                       uint32_t begin1, uint32_t deg1,
                                       ValueType* __restrict__ smem0, ValueType* __restrict__ smem1,
                                       uint32_t warp_write_base, uint32_t& warp_local_count,
                                       uint32_t old_size, WriteFunc write) {
    run(
        tile, view0, col0, begin0, deg0, view1, col1, begin1, deg1, smem0, smem1, warp_write_base,
        warp_local_count, old_size, [](ValueType, ValueType) { return true; }, write);
  }
};

// ============================================================================
// TiledCartesianEmitter: tiled reads + warp-coalesced writes (atomic variant)
// for WS or speculative paths where per-warp offsets are not known.
//
// Uses atomic global_write_pos (like ChunkedOutputContext) since per-thread
// write_base_ offsets can't be coalesced.
//
// Shared memory layout per warp:
//   smem_src0[TileSize]  - source 0 column values
//   smem_src1[TileSize]  - source 1 column values
//   (output goes directly to global via ballot+coalesced store, no smem buffer)
//
// Usage in kernel:
//   constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;
//   constexpr int kCartTileSize = 128;
//   __shared__ ValueType s_cart[kWarpsPerBlock][2][kCartTileSize];
//   uint32_t warp_in_block = threadIdx.x / kGroupSize;
//
//   TiledCartesianEmitter<ValueType, Arity, kCartTileSize>::run(
//       tile, view0, col0, h0.begin(), D0,
//              view1, col1, h1.begin(), D1,
//       s_cart[warp_in_block][0], s_cart[warp_in_block][1],
//       global_write_pos, output_data, output_stride, old_size,
//       [&](ValueType v0, ValueType v1) -> bool {
//           return filter_condition;  // return true to emit
//       },
//       [&](ValueType v0, ValueType v1, uint32_t out_pos) {
//           // write columns to out_pos
//           output_data[0 * stride + out_pos] = v0;
//           output_data[1 * stride + out_pos] = v1;
//       });
// ============================================================================
template <typename ValueType, int Arity, int TileSize = 128>
struct TiledCartesianEmitter {

  // FilterFunc: (v0, v1) -> bool (true = emit this pair)
  // WriteFunc: (v0, v1, output_position) -> void (write columns at position)
  template <typename Tile, typename ViewType, typename FilterFunc, typename WriteFunc>
  __device__ static void run(Tile tile, const ViewType& view0, int col0, uint32_t begin0,
                             uint32_t deg0, const ViewType& view1, int col1, uint32_t begin1,
                             uint32_t deg1, ValueType* __restrict__ smem0,
                             ValueType* __restrict__ smem1, uint32_t* __restrict__ global_write_pos,
                             ValueType* __restrict__ output_data, uint32_t output_stride,
                             uint32_t old_size, FilterFunc filter, WriteFunc write) {

    uint32_t lane = tile.thread_rank();
    uint32_t gs = tile.size();

    for (uint32_t t0_base = 0; t0_base < deg0; t0_base += TileSize) {
      uint32_t t0_end = min(t0_base + (uint32_t)TileSize, deg0);
      uint32_t t0_len = t0_end - t0_base;

      // Coalesced load source 0
      for (uint32_t i = lane; i < t0_len; i += gs) {
        smem0[i] = view0.get_value(col0, begin0 + t0_base + i);
      }

      for (uint32_t t1_base = 0; t1_base < deg1; t1_base += TileSize) {
        uint32_t t1_end = min(t1_base + (uint32_t)TileSize, deg1);
        uint32_t t1_len = t1_end - t1_base;

        // Coalesced load source 1
        for (uint32_t i = lane; i < t1_len; i += gs) {
          smem1[i] = view1.get_value(col1, begin1 + t1_base + i);
        }
        tile.sync();

        // Iterate Cartesian from smem, emit with warp-coalesced writes
        uint32_t total = t0_len * t1_len;
        for (uint32_t batch = 0; batch < total; batch += gs) {
          uint32_t flat = batch + lane;
          bool valid = flat < total;

          ValueType v0{}, v1{};
          if (valid) {
            uint32_t i0 = flat / t1_len;
            uint32_t i1 = flat % t1_len;
            v0 = smem0[i0];
            v1 = smem1[i1];
            valid = filter(v0, v1);
          }

          // Warp-coalesced write: one atomicAdd per warp
          uint32_t ballot = tile.ballot(valid);
          uint32_t active_count = __popc(ballot);
          if (active_count > 0) {
            uint32_t my_mask = (1u << lane) - 1u;
            uint32_t my_offset = __popc(ballot & my_mask);
            uint32_t base;
            if (lane == 0) {
              base = atomicAdd(global_write_pos, active_count);
            }
            base = tile.shfl(base, 0) + old_size;
            if (valid) {
              write(v0, v1, base + my_offset);
            }
          }
        }
        tile.sync();
      }
    }
  }

  // Convenience: no filter (emit all pairs)
  template <typename Tile, typename ViewType, typename WriteFunc>
  __device__ static void run_no_filter(Tile tile, const ViewType& view0, int col0, uint32_t begin0,
                                       uint32_t deg0, const ViewType& view1, int col1,
                                       uint32_t begin1, uint32_t deg1,
                                       ValueType* __restrict__ smem0, ValueType* __restrict__ smem1,
                                       uint32_t* __restrict__ global_write_pos,
                                       ValueType* __restrict__ output_data, uint32_t output_stride,
                                       uint32_t old_size, WriteFunc write) {
    run(
        tile, view0, col0, begin0, deg0, view1, col1, begin1, deg1, smem0, smem1, global_write_pos,
        output_data, output_stride, old_size, [](ValueType, ValueType) { return true; }, write);
  }
};

// ============================================================================
// Host-side helpers: allocate/free/reset queue on device
// All allocations go through RMM pool (rmm::mr::get_current_device_resource)
// to avoid OOM from raw cudaMalloc when RMM has reserved most GPU memory.
// ============================================================================

#ifndef __CUDA_ARCH__
#include <rmm/mr/device/per_device_resource.hpp>
#endif

inline void* ws_rmm_alloc(std::size_t bytes, GPU_STREAM_T stream) {
  return rmm::mr::get_current_device_resource()->allocate(bytes, rmm::cuda_stream_view{stream});
}

inline void ws_rmm_free(void* ptr, std::size_t bytes, GPU_STREAM_T stream) {
  rmm::mr::get_current_device_resource()->deallocate(ptr, bytes, rmm::cuda_stream_view{stream});
}

inline WCOJTaskQueue* allocate_task_queue(GPU_STREAM_T stream = 0,
                                          uint32_t capacity = WCOJ_QUEUE_CAPACITY) {
  auto* d_queue = static_cast<WCOJTaskQueue*>(ws_rmm_alloc(sizeof(WCOJTaskQueue), stream));

  auto* d_tasks = static_cast<WCOJTask*>(ws_rmm_alloc(capacity * sizeof(WCOJTask), stream));

  using FlagType = cuda::atomic<uint32_t, cuda::thread_scope_device>;
  auto* d_flags = static_cast<FlagType*>(ws_rmm_alloc(capacity * sizeof(FlagType), stream));

  // Zero-init ready flags (async on stream)
  cudaMemsetAsync(d_flags, 0, capacity * sizeof(FlagType), stream);

  // Build host-side queue struct and copy to device (async)
  WCOJTaskQueue h_queue;
  memset(&h_queue, 0, sizeof(WCOJTaskQueue));
  h_queue.tasks = d_tasks;
  h_queue.ready_flags = d_flags;
  h_queue.capacity = capacity;
  // active_producers is zero-initialized by memset, will be set at kernel launch time
  GPU_MEMCPY_ASYNC(d_queue, &h_queue, sizeof(WCOJTaskQueue), GPU_HOST_TO_DEVICE, stream);

  return d_queue;
}

inline void free_task_queue(WCOJTaskQueue* d_queue, GPU_STREAM_T stream = 0) {
  if (!d_queue)
    return;
  // D2H copy with pageable memory blocks until complete (implicit sync)
  WCOJTaskQueue h_queue;
  GPU_MEMCPY_ASYNC(&h_queue, d_queue, sizeof(WCOJTaskQueue), GPU_DEVICE_TO_HOST, stream);
  using FlagType = cuda::atomic<uint32_t, cuda::thread_scope_device>;
  ws_rmm_free(h_queue.tasks, h_queue.capacity * sizeof(WCOJTask), stream);
  ws_rmm_free(h_queue.ready_flags, h_queue.capacity * sizeof(FlagType), stream);
  ws_rmm_free(d_queue, sizeof(WCOJTaskQueue), stream);
}

inline void reset_task_queue(WCOJTaskQueue* d_queue, uint32_t capacity = WCOJ_QUEUE_CAPACITY,
                             GPU_STREAM_T stream = 0) {
  // Download the queue struct to get the device pointers (tasks, ready_flags)
  WCOJTaskQueue h_queue;
  GPU_MEMCPY_ASYNC(&h_queue, d_queue, sizeof(WCOJTaskQueue), GPU_DEVICE_TO_HOST, stream);
  // Must synchronize before reading h_queue — the D2H is async!
  cudaStreamSynchronize(stream);

  using FlagType = cuda::atomic<uint32_t, cuda::thread_scope_device>;
  cudaMemsetAsync(h_queue.ready_flags, 0, capacity * sizeof(FlagType), stream);
  h_queue.head.store(0, cuda::memory_order_relaxed);
  h_queue.tail.store(0, cuda::memory_order_relaxed);
  h_queue.active_producers.store(0, cuda::memory_order_relaxed);
  GPU_MEMCPY_ASYNC(d_queue, &h_queue, sizeof(WCOJTaskQueue), GPU_HOST_TO_DEVICE, stream);
}

// ============================================================================
// Range Board allocation/reset
// ============================================================================

inline WCOJRangeBoard* allocate_range_board(GPU_STREAM_T stream = 0,
                                            uint32_t capacity = WCOJ_RANGE_BOARD_CAPACITY) {
  auto* d_board = static_cast<WCOJRangeBoard*>(ws_rmm_alloc(sizeof(WCOJRangeBoard), stream));
  auto* d_ranges = static_cast<WCOJRange*>(ws_rmm_alloc(capacity * sizeof(WCOJRange), stream));

  WCOJRangeBoard h_board;
  memset(&h_board, 0, sizeof(WCOJRangeBoard));
  h_board.ranges = d_ranges;
  h_board.capacity = capacity;
  GPU_MEMCPY_ASYNC(d_board, &h_board, sizeof(WCOJRangeBoard), GPU_HOST_TO_DEVICE, stream);

  return d_board;
}

inline void reset_range_board(WCOJRangeBoard* d_board, GPU_STREAM_T stream = 0) {
  WCOJRangeBoard h_board;
  GPU_MEMCPY_ASYNC(&h_board, d_board, sizeof(WCOJRangeBoard), GPU_DEVICE_TO_HOST, stream);
  cudaStreamSynchronize(stream);
  h_board.count.store(0, cuda::memory_order_relaxed);
  GPU_MEMCPY_ASYNC(d_board, &h_board, sizeof(WCOJRangeBoard), GPU_HOST_TO_DEVICE, stream);
}

// ============================================================================
// Diagnostics: queue state readback after kernel
// ============================================================================
inline WCOJTaskQueue* g_ws_diagnostic_queue = nullptr;

inline void set_diagnostic_queue(WCOJTaskQueue* q) {
  g_ws_diagnostic_queue = q;
}

struct QueueStats {
  uint32_t tasks_pushed;    // tail value
  uint32_t tasks_consumed;  // head value
};

inline QueueStats read_queue_stats(GPU_STREAM_T stream = 0) {
  if (!g_ws_diagnostic_queue)
    return {0, 0};
  WCOJTaskQueue h_queue;
  GPU_MEMCPY(&h_queue, g_ws_diagnostic_queue, sizeof(WCOJTaskQueue), GPU_DEVICE_TO_HOST);
  return {h_queue.tail.load(cuda::memory_order_relaxed),
          h_queue.head.load(cuda::memory_order_relaxed)};
}

}  // namespace SRDatalog::GPU::JIT::WS
