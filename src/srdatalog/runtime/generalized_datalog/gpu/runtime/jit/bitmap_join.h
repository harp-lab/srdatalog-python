#pragma once

#include "gpu/device_array.h"

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace SRDatalog::GPU::bitmap {

struct Columns {
  uint64_t rows;
  const uint32_t* first;
  const uint32_t* second;
};

struct Keys {
  uint64_t size;
  const uint32_t* data;
};

// Borrowed columns are paired (join, destination) or (join, value).
// Each points segment is sorted by join. Each heap cache is sorted and unique;
// their union must contain every consumed value, but may be a safe superset.
// Heap-cache boundaries are independent of points FULL/head boundaries.
struct Input {
  Columns assign;
  Columns assign_head;
  Columns points_full;
  Columns points_head;
  Keys heaps_full;
  Keys heaps_head;
};

struct Output {
  uint32_t* value;
  uint32_t* destination;
};

namespace detail {

inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kWarpSize = 32;
inline constexpr unsigned kWarps = kThreads / kWarpSize;
inline constexpr uint64_t kRankMapBytes = 64ull * 1024 * 1024;

inline void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("bitmap join: ") + stage + ": " +
                             cudaGetErrorString(status));
  }
}

// Declare after all scratch arrays so exceptional exits finish GPU work before
// their destructors release storage. Normal exits synchronize with error checks.
struct SynchronizeOnError {
  int exceptions = std::uncaught_exceptions();
  ~SynchronizeOnError() {
    if (std::uncaught_exceptions() > exceptions) cudaStreamSynchronize(nullptr);
  }
};

template <class T>
inline void require_addressable(uint64_t size, const char* name) {
  // DeviceArray rounds byte counts up to uint32 storage; Thrust differences are
  // signed. Check before either multiplication or pointer arithmetic occurs.
  constexpr uint64_t limit =
      std::min<uint64_t>(std::numeric_limits<size_t>::max() - sizeof(uint32_t) + 1,
                         std::numeric_limits<std::ptrdiff_t>::max());
  if (size > limit / sizeof(T)) {
    throw std::overflow_error(std::string("bitmap join: ") + name +
                              " exceeds addressable size");
  }
}

inline uint64_t checked_add(uint64_t first, uint64_t second, const char* name) {
  if (first > std::numeric_limits<uint64_t>::max() - second) {
    throw std::overflow_error(std::string("bitmap join: ") + name + " overflows uint64");
  }
  return first + second;
}

inline void require_columns(const Columns& columns, const char* name) {
  require_addressable<uint32_t>(columns.rows, name);
  if (columns.rows != 0 && (columns.first == nullptr || columns.second == nullptr)) {
    throw std::invalid_argument(std::string("bitmap join: missing ") + name + " columns");
  }
}

inline void require_keys(const Keys& keys) {
  require_addressable<uint32_t>(keys.size, "heap cache");
  if (keys.size != 0 && keys.data == nullptr) {
    throw std::invalid_argument("bitmap join: missing heap cache data");
  }
}

inline unsigned blocks_for(uint64_t size) {
  return static_cast<unsigned>(std::min<uint64_t>(size / kThreads + (size % kThreads != 0),
                                                 65535));
}

template <class = void>
__global__ void reverse_assign(Columns full, Columns head, uint64_t edges, uint64_t* reversed) {
  for (uint64_t edge = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       edge < edges; edge += uint64_t(gridDim.x) * blockDim.x) {
    const Columns segment = edge < full.rows ? full : head;
    const uint64_t row = edge < full.rows ? edge : edge - full.rows;
    reversed[edge] = (uint64_t(segment.second[row]) << 32) | segment.first[row];
  }
}

struct DestinationStart {
  const uint64_t* reversed;
  __host__ __device__ bool operator()(uint64_t edge) const {
    return edge == 0 || (reversed[edge] >> 32) != (reversed[edge - 1] >> 32);
  }
};

template <class = void>
__global__ void finish_destinations(const uint64_t* reversed, uint64_t edges,
                                    uint32_t destinations, unsigned segments,
                                    uint32_t* ids, uint64_t* offsets) {
  for (uint64_t destination = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       destination <= destinations; destination += uint64_t(gridDim.x) * blockDim.x) {
    if (destination == destinations) {
      offsets[destination] = edges * segments;
    } else {
      const uint64_t edge = offsets[destination];
      ids[destination] = static_cast<uint32_t>(reversed[edge] >> 32);
      offsets[destination] = edge * segments;
    }
  }
}

__device__ inline uint64_t source_bound(const uint32_t* variables, uint64_t rows,
                                       uint32_t source, bool upper) {
  uint64_t low = 0;
  uint64_t high = rows;
  while (low < high) {
    const uint64_t middle = low + (high - low) / 2;
    const uint32_t value = variables[middle];
    if (value < source || (upper && value == source)) low = middle + 1;
    else high = middle;
  }
  return low;
}

template <class = void>
__global__ void build_point_ranges(const uint64_t* reversed, uint64_t edges,
                                   Columns full, Columns head, unsigned segments,
                                   uint64_t* begin, uint64_t* end) {
  for (uint64_t edge = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       edge < edges; edge += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t source = static_cast<uint32_t>(reversed[edge]);
    uint64_t slot = edge * segments;
    if (full.rows != 0) {
      begin[slot] = source_bound(full.first, full.rows, source, false);
      end[slot] = source_bound(full.first, full.rows, source, true);
      ++slot;
    }
    if (head.rows != 0) {
      begin[slot] = full.rows + source_bound(head.first, head.rows, source, false);
      end[slot] = full.rows + source_bound(head.first, head.rows, source, true);
    }
  }
}

template <class = void>
__global__ void build_heap_ranks(const uint32_t* ids, uint32_t heaps, uint32_t* ranks) {
  for (uint64_t rank = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       rank < heaps; rank += uint64_t(gridDim.x) * blockDim.x) {
    // No sentinel: every consumed value belongs to the cache union.
    ranks[ids[rank]] = static_cast<uint32_t>(rank);
  }
}

struct BitmapInput {
  uint32_t heaps;
  const uint32_t* destination_ids;
  const uint32_t* heap_ids;
  const uint32_t* heap_ranks;
  const uint64_t* edge_offsets;
  const uint64_t* point_begin;
  const uint64_t* point_end;
  const uint32_t* points_full;
  const uint32_t* points_head;
  uint64_t full_rows;
};

// One block owns one destination. The count and emit passes construct exactly
// the same shared bitmap; output is unique, but not globally value-sorted.
template <bool Emit>
__global__ void bitmap_kernel(BitmapInput input, uint32_t* counts,
                              const uint64_t* output_offsets, Output output) {
  extern __shared__ uint32_t bitmap[];
  __shared__ uint32_t warp_totals[kWarps];
  const uint32_t destination = blockIdx.x;
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread % kWarpSize;
  const unsigned warp = thread / kWarpSize;
  const uint64_t words = (uint64_t(input.heaps) + 31) / 32;
  for (uint64_t word = thread; word < words; word += kThreads) bitmap[word] = 0;
  __syncthreads();

  const uint64_t edge_begin = input.edge_offsets[destination];
  const uint64_t edge_end = input.edge_offsets[uint64_t(destination) + 1];
  for (uint64_t edge = edge_begin + warp; edge < edge_end; edge += kWarps) {
    const uint64_t begin = input.point_begin[edge];
    const uint64_t end = input.point_end[edge];
    for (uint64_t point = begin + lane; point < end; point += kWarpSize) {
      const uint32_t original = point < input.full_rows
          ? input.points_full[point] : input.points_head[point - input.full_rows];
      uint32_t heap;
      if (input.heap_ranks != nullptr) {
        heap = input.heap_ranks[original];
      } else {
        uint32_t low = 0;
        uint32_t high = input.heaps;
        while (low < high) {
          const uint32_t middle = low + (high - low) / 2;
          if (input.heap_ids[middle] < original) low = middle + 1;
          else high = middle;
        }
        heap = low;
      }
      const uint32_t mask = uint32_t(1) << (heap % 32);
      cuda::atomic_ref<uint32_t, cuda::thread_scope_block> word(bitmap[heap / 32]);
      // Racing reads must also be atomic. Monotone bit setting lets an observed
      // set bit bypass the shared-memory read-modify-write operation.
      if ((word.load(cuda::memory_order_relaxed) & mask) == 0) {
        word.fetch_or(mask, cuda::memory_order_relaxed);
      }
    }
  }
  __syncthreads();

  uint32_t local_count = 0;
  for (uint64_t word = thread; word < words; word += kThreads) {
    local_count += __popc(bitmap[word]);
  }
  uint32_t inclusive = local_count;
  for (unsigned distance = 1; distance < kWarpSize; distance *= 2) {
    const uint32_t preceding = __shfl_up_sync(0xffffffffu, inclusive, distance);
    if (lane >= distance) inclusive += preceding;
  }
  if (lane == kWarpSize - 1) warp_totals[warp] = inclusive;
  __syncthreads();

  if constexpr (Emit) {
    uint32_t prefix = inclusive - local_count;
    for (unsigned preceding_warp = 0; preceding_warp < warp; ++preceding_warp) {
      prefix += warp_totals[preceding_warp];
    }
    if (local_count == 0) return;
    uint64_t offset = output_offsets[destination] + prefix;
    const uint32_t destination_id = input.destination_ids[destination];
    for (uint64_t word = thread; word < words; word += kThreads) {
      uint32_t bits = bitmap[word];
      while (bits != 0) {
        const unsigned bit = unsigned(__ffs(static_cast<int>(bits)) - 1);
        const uint64_t heap = word * 32 + bit;
        output.value[offset] = input.heap_ids[heap];
        output.destination[offset] = destination_id;
        ++offset;
        bits &= bits - 1;
      }
    }
  } else if (thread == 0) {
    uint32_t total = 0;
    for (unsigned preceding_warp = 0; preceding_warp < kWarps; ++preceding_warp) {
      total += warp_totals[preceding_warp];
    }
    counts[destination] = total;
  }
}

template <bool Emit>
inline size_t configure_bitmap(uint32_t heaps, int device) {
  const size_t shared_bytes = size_t((uint64_t(heaps) + 31) / 32) * sizeof(uint32_t);
  cudaFuncAttributes attributes{};
  check_cuda(cudaFuncGetAttributes(&attributes, bitmap_kernel<Emit>), "inspect bitmap kernel");
  int default_shared = 0;
  check_cuda(cudaDeviceGetAttribute(&default_shared, cudaDevAttrMaxSharedMemoryPerBlock, device),
             "get default shared-memory limit");
  if (shared_bytes + attributes.sharedSizeBytes > size_t(default_shared)) {
    int optin_shared = 0;
    check_cuda(cudaDeviceGetAttribute(&optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
               "get opt-in shared-memory limit");
    if (shared_bytes + attributes.sharedSizeBytes > size_t(optin_shared) ||
        shared_bytes > size_t(std::numeric_limits<int>::max())) {
      throw std::runtime_error("bitmap join: exact value bitmap exceeds device shared-memory limit");
    }
    check_cuda(cudaFuncSetAttribute(bitmap_kernel<Emit>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_bytes)), "configure exact bitmap capacity");
  }
  return shared_bytes;
}

struct WidenCount {
  __host__ __device__ uint64_t operator()(uint32_t count) const { return count; }
};

}  // namespace detail

// Computes the exact set C(value, destination). All uint32 IDs are valid, including
// UINT32_MAX. The dictionary must fit one block's shared memory; unsupported sizes
// fail rather than falling back to approximate deduplication. No GPU scratch is
// retained. Allocate is called exactly once for nonzero output, with a uint64 row
// count, and returns pointers to the appended NEW columns in (value, destination)
// order. It must use the default stream; borrowed inputs remain valid through emit.
template <class Allocate>
void execute(const Input& spec, Allocate&& allocate) {
  using namespace detail;
  if ((spec.assign.rows == 0 && spec.assign_head.rows == 0) ||
      (spec.points_full.rows == 0 && spec.points_head.rows == 0)) return;
  require_columns(spec.assign, "assign FULL");
  require_columns(spec.assign_head, "assign head");
  require_columns(spec.points_full, "points FULL");
  require_columns(spec.points_head, "points head");
  require_keys(spec.heaps_full);
  require_keys(spec.heaps_head);
  const uint64_t edges = checked_add(spec.assign.rows, spec.assign_head.rows, "assign rows");
  checked_add(spec.points_full.rows, spec.points_head.rows, "points segment offsets");
  const uint64_t heap_capacity = checked_add(spec.heaps_full.size, spec.heaps_head.size,
                                            "heap cache union");
  if (heap_capacity == 0) {
    throw std::invalid_argument("bitmap join: nonempty points have no heap caches");
  }
  require_addressable<uint32_t>(heap_capacity, "heap cache union");
  const unsigned segments = unsigned(spec.points_full.rows != 0) + unsigned(spec.points_head.rows != 0);
  require_addressable<uint64_t>(edges, "reversed assign");
  const uint64_t ranges = segments == 2 ? checked_add(edges, edges, "source ranges") : edges;
  require_addressable<uint64_t>(ranges, "source ranges");

  const cudaStream_t stream = nullptr;
  auto policy = rmm::exec_policy(stream);
  DeviceArray<uint32_t> heap_union(0, stream);
  DeviceArray<uint32_t> heap_ranks(0, stream);
  DeviceArray<uint64_t> reversed(0, stream);
  DeviceArray<uint32_t> destination_ids(0, stream);
  DeviceArray<uint64_t> edge_offsets(0, stream);
  DeviceArray<uint64_t> point_begin(0, stream);
  DeviceArray<uint64_t> point_end(0, stream);
  DeviceArray<uint32_t> counts(0, stream);
  DeviceArray<uint64_t> output_offsets(0, stream);
  const SynchronizeOnError synchronize_on_error;

  // A union cannot be smaller than either unique cache. Reject an impossible
  // dictionary before allocating its union; exact kernel static usage is checked
  // below once the union's cardinality is known.
  int device = 0;
  int default_shared = 0;
  int optin_shared = 0;
  check_cuda(cudaGetDevice(&device), "get device");
  check_cuda(cudaDeviceGetAttribute(&default_shared, cudaDevAttrMaxSharedMemoryPerBlock, device),
             "get default shared-memory limit");
  check_cuda(cudaDeviceGetAttribute(&optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
             "get opt-in shared-memory limit");
  const uint64_t minimum_heaps = std::max(spec.heaps_full.size, spec.heaps_head.size);
  if ((minimum_heaps / 32 + (minimum_heaps % 32 != 0)) >
      uint64_t(std::max(default_shared, optin_shared)) / sizeof(uint32_t)) {
    throw std::runtime_error("bitmap join: exact value bitmap exceeds device shared-memory limit");
  }

  const uint32_t* heap_ids;
  uint64_t heap_count;
  if (spec.heaps_full.size == 0) {
    heap_ids = spec.heaps_head.data;
    heap_count = spec.heaps_head.size;
  } else if (spec.heaps_head.size == 0) {
    heap_ids = spec.heaps_full.data;
    heap_count = spec.heaps_full.size;
  } else {
    heap_union = DeviceArray<uint32_t>(heap_capacity, stream);
    const auto end = thrust::set_union(policy,
        spec.heaps_full.data, spec.heaps_full.data + spec.heaps_full.size,
        spec.heaps_head.data, spec.heaps_head.data + spec.heaps_head.size, heap_union.data());
    check_cuda(cudaGetLastError(), "union heap caches");
    heap_ids = heap_union.data();
    heap_count = end - heap_ids;
  }
  if (heap_count == 0 || heap_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("bitmap join: heap cache union exceeds nonzero uint32 rank capacity");
  }
  const uint32_t heaps = static_cast<uint32_t>(heap_count);
  const size_t shared_bytes = configure_bitmap<false>(heaps, device);
  configure_bitmap<true>(heaps, device);

  uint32_t maximum_heap = 0;
  check_cuda(cudaMemcpyAsync(&maximum_heap, heap_ids + heap_count - 1, sizeof(maximum_heap),
                             cudaMemcpyDeviceToHost, stream), "read maximum value ID");
  check_cuda(cudaStreamSynchronize(stream), "finish heap preprocessing");
  // Widen before adding: UINT32_MAX selects binary search, never wraps to zero.
  const uint64_t rank_entries = uint64_t(maximum_heap) + 1;
  if (rank_entries <= kRankMapBytes / sizeof(uint32_t)) {
    heap_ranks = DeviceArray<uint32_t>(rank_entries, stream);
    build_heap_ranks<><<<blocks_for(heap_count), kThreads, 0, stream>>>(heap_ids, heaps, heap_ranks.data());
    check_cuda(cudaGetLastError(), "build value rank map");
  }

  reversed = DeviceArray<uint64_t>(edges, stream);
  reverse_assign<><<<blocks_for(edges), kThreads, 0, stream>>>(
      spec.assign, spec.assign_head, edges, reversed.data());
  check_cuda(cudaGetLastError(), "reverse assign segments");
  thrust::sort(policy, reversed.data(), reversed.data() + edges);
  check_cuda(cudaGetLastError(), "sort reversed assign");
  const auto first_edge = thrust::make_counting_iterator<uint64_t>(0);
  const DestinationStart starts{reversed.data()};
  const uint64_t destination_count = thrust::count_if(policy, first_edge, first_edge + edges, starts);
  check_cuda(cudaGetLastError(), "count destinations");
  int max_grid_x = 0;
  check_cuda(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device), "get grid limit");
  if (destination_count > uint64_t(max_grid_x)) {
    throw std::runtime_error("bitmap join: destination count exceeds device grid capacity");
  }
  const uint32_t destinations = static_cast<uint32_t>(destination_count);
  const uint64_t offset_count = destination_count + 1;
  require_addressable<uint64_t>(offset_count, "destination offsets");
  destination_ids = DeviceArray<uint32_t>(destinations, stream);
  edge_offsets = DeviceArray<uint64_t>(offset_count, stream);
  thrust::copy_if(policy, first_edge, first_edge + edges, edge_offsets.data(), starts);
  check_cuda(cudaGetLastError(), "extract destination boundaries");
  finish_destinations<><<<blocks_for(offset_count), kThreads, 0, stream>>>(
      reversed.data(), edges, destinations, segments, destination_ids.data(), edge_offsets.data());
  check_cuda(cudaGetLastError(), "build destination edge offsets");
  point_begin = DeviceArray<uint64_t>(ranges, stream);
  point_end = DeviceArray<uint64_t>(ranges, stream);
  build_point_ranges<><<<blocks_for(edges), kThreads, 0, stream>>>(
      reversed.data(), edges, spec.points_full, spec.points_head, segments,
      point_begin.data(), point_end.data());
  check_cuda(cudaGetLastError(), "build source ranges");
  check_cuda(cudaStreamSynchronize(stream), "finish source preprocessing");
  reversed = DeviceArray<uint64_t>(0, stream);

  const BitmapInput input{heaps, destination_ids.data(), heap_ids,
      heap_ranks.empty() ? nullptr : heap_ranks.data(), edge_offsets.data(),
      point_begin.data(), point_end.data(), spec.points_full.second,
      spec.points_head.second, spec.points_full.rows};
  output_offsets = DeviceArray<uint64_t>(offset_count, stream);
  counts = DeviceArray<uint32_t>(destinations, stream);
  bitmap_kernel<false><<<destinations, kThreads, shared_bytes, stream>>>(
      input, counts.data(), nullptr, Output{nullptr, nullptr});
  check_cuda(cudaGetLastError(), "count exact destination bitmaps");
  check_cuda(cudaMemsetAsync(output_offsets.data(), 0, sizeof(uint64_t), stream), "initialize output offset");
  const auto first_count = thrust::make_transform_iterator(counts.data(), WidenCount{});
  // Each count is at most heaps, and both dimensions are uint32, so the widened
  // scan cannot overflow uint64 even before the output addressability check.
  thrust::inclusive_scan(policy, first_count, first_count + destinations, output_offsets.data() + 1,
                         thrust::plus<uint64_t>());
  check_cuda(cudaGetLastError(), "scan exact output counts");
  uint64_t output_rows = 0;
  check_cuda(cudaMemcpyAsync(&output_rows, output_offsets.data() + destinations, sizeof(output_rows),
                             cudaMemcpyDeviceToHost, stream), "read exact output total");
  check_cuda(cudaStreamSynchronize(stream), "finish exact output count");
  counts = DeviceArray<uint32_t>(0, stream);
  if (output_rows == 0) return;
  require_addressable<uint32_t>(output_rows, "output columns");
  const Output output = std::forward<Allocate>(allocate)(output_rows);
  if (output.value == nullptr || output.destination == nullptr) {
    throw std::runtime_error("bitmap join: output allocator returned null columns");
  }
  bitmap_kernel<true><<<destinations, kThreads, shared_bytes, stream>>>(
      input, nullptr, output_offsets.data(), output);
  check_cuda(cudaGetLastError(), "emit exact destination bitmaps");
  check_cuda(cudaStreamSynchronize(stream), "finish bitmap emission");
}

}  // namespace SRDatalog::GPU::bitmap
