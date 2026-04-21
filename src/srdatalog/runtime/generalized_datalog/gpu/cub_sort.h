/**
 * @file cub_sort.h
 * @brief Stream-aware lexicographic sort for SoA relations using CUB
 *
 * Matches Thrust's zip_sort_columns performance while being explicitly
 * stream-async (takes cudaStream_t parameter).
 *
 * == Why not just use thrust::sort with rmm::exec_policy(stream)? ==
 * Thrust internally allocates temp buffers via RMM and calls CUB DeviceMergeSort.
 * With stream-ordered RMM pools (CUDA 11.2+), this should be stream-safe.
 * However, the user reports that Thrust forces implicit synchronization
 * when multiple sorts run on different streams in practice.
 *
 * == Algorithm ==
 *
 * For arity-2: Pack SoA → uint64 → cub::DeviceRadixSort::SortKeys → unpack
 *   (Radix sort on 64-bit scalar is O(n), faster than merge sort)
 *
 * For arity-N (N >= 3): Pack SoA → AoS struct → cub::DeviceMergeSort::SortKeys
 *   with lexicographic comparator → unpack
 *   (Same algorithm Thrust uses internally for zip_iterator)
 *
 * Both paths: pack → single CUB call → unpack. Minimal data movement.
 */

#pragma once

#include "gpu/device_array.h"
#include "gpu/gpu_api.h"
#include <cassert>
#include <cstdint>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>

namespace SRDatalog::GPU {

// =============================================================================
// Helper kernels — all stream-aware, no implicit sync
// =============================================================================

/// Pack two SoA columns → uint64: key = (col[order[0]] << 32) | col[order[1]]
__global__ void cub_sort_pack_u64(const uint32_t* __restrict__ col0,
                                  const uint32_t* __restrict__ col1, uint64_t* __restrict__ dst,
                                  uint32_t n, int order0, int order1) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // order0/order1 are 0 or 1, selecting which column is high/low
    const uint32_t* hi = (order0 == 0) ? col0 : col1;
    const uint32_t* lo = (order1 == 0) ? col0 : col1;
    dst[i] = (static_cast<uint64_t>(hi[i]) << 32) | static_cast<uint64_t>(lo[i]);
  }
}

/// Unpack uint64 → two SoA columns
__global__ void cub_sort_unpack_u64(const uint64_t* __restrict__ src, uint32_t* __restrict__ col0,
                                    uint32_t* __restrict__ col1, uint32_t n, int order0,
                                    int order1) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t hi = static_cast<uint32_t>(src[i] >> 32);
    uint32_t lo = static_cast<uint32_t>(src[i] & 0xFFFFFFFFULL);
    uint32_t* dst_hi = (order0 == 0) ? col0 : col1;
    uint32_t* dst_lo = (order1 == 0) ? col0 : col1;
    dst_hi[i] = hi;
    dst_lo[i] = lo;
  }
}

// =============================================================================
// AoS tuple for N-column merge sort (N >= 3)
// =============================================================================

/// Compile-time sized tuple — exactly N×4 bytes, no padding waste.
template <int N>
struct __align__(4) SortTuple {
  uint32_t cols[N];
};

template <int N>
struct SortTupleCompare {
  __host__ __device__ bool operator()(const SortTuple<N>& a, const SortTuple<N>& b) const {
#pragma unroll
    for (int c = 0; c < N; ++c) {
      if (a.cols[c] < b.cols[c])
        return true;
      if (a.cols[c] > b.cols[c])
        return false;
    }
    return false;
  }
};

template <int N>
__global__ void cub_sort_pack_aos(const uint32_t* const* __restrict__ columns,
                                  const int* __restrict__ order, SortTuple<N>* __restrict__ dst,
                                  uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    SortTuple<N> t;
#pragma unroll
    for (int c = 0; c < N; ++c)
      t.cols[c] = columns[order[c]][i];
    dst[i] = t;
  }
}

template <int N>
__global__ void cub_sort_unpack_aos(const SortTuple<N>* __restrict__ src,
                                    uint32_t* const* __restrict__ columns,
                                    const int* __restrict__ order, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    SortTuple<N> t = src[i];
#pragma unroll
    for (int c = 0; c < N; ++c)
      columns[order[c]][i] = t.cols[c];
  }
}

// =============================================================================
// Temp storage — pre-allocatable, reusable (must be before mergesort helper)
// =============================================================================

struct CubSortStorage {
  DeviceArray<uint8_t> temp;          // CUB workspace
  DeviceArray<uint64_t> keys64;       // For arity-2 packed keys
  DeviceArray<uint8_t> tuple_buf;     // For arity-3+ packed tuples (type-erased)
  DeviceArray<uint32_t*> d_col_ptrs;  // Device copy of column pointers
  DeviceArray<int> d_order;           // Device copy of sort order

  void ensure_arity2(uint32_t n, size_t temp_bytes) {
    if (temp.size() < temp_bytes)
      temp.resize(temp_bytes);
    size_t need = 2 * static_cast<size_t>(n);
    if (keys64.size() < need)
      keys64.resize(need);
  }

  void ensure_general(uint32_t n, size_t temp_bytes, int num_cols) {
    if (temp.size() < temp_bytes)
      temp.resize(temp_bytes);
    if (d_col_ptrs.size() < static_cast<size_t>(num_cols))
      d_col_ptrs.resize(num_cols);
    if (d_order.size() < static_cast<size_t>(num_cols))
      d_order.resize(num_cols);
  }
};

/// Dispatch helper: run merge sort for specific arity N
template <int N>
inline void cub_lex_sort_mergesort(uint32_t* const* columns, uint32_t n, const int* order,
                                   GPU_STREAM_T stream, CubSortStorage& storage) {
  constexpr int kBS = 256;
  int nb = (n + kBS - 1) / kBS;

  size_t temp_bytes = 0;
  SortTupleCompare<N> cmp;
  cub::DeviceMergeSort::SortKeys(nullptr, temp_bytes, (SortTuple<N>*)nullptr, static_cast<int>(n),
                                 cmp, stream);
  storage.ensure_general(n, temp_bytes, N);

  GPU_MEMCPY_ASYNC(storage.d_col_ptrs.data(), columns, N * sizeof(uint32_t*), GPU_HOST_TO_DEVICE,
                   stream);
  GPU_MEMCPY_ASYNC(storage.d_order.data(), order, N * sizeof(int), GPU_HOST_TO_DEVICE, stream);

  // Ensure tuples buffer is sized for SortTuple<N>
  size_t tuple_bytes = static_cast<size_t>(n) * sizeof(SortTuple<N>);
  if (storage.tuple_buf.size() < tuple_bytes)
    storage.tuple_buf.resize(tuple_bytes);
  auto* tuples = reinterpret_cast<SortTuple<N>*>(storage.tuple_buf.data());

  cub_sort_pack_aos<N>
      <<<nb, kBS, 0, stream>>>(storage.d_col_ptrs.data(), storage.d_order.data(), tuples, n);

  cub::DeviceMergeSort::SortKeys(storage.temp.data(), temp_bytes, tuples, static_cast<int>(n), cmp,
                                 stream);

  cub_sort_unpack_aos<N>
      <<<nb, kBS, 0, stream>>>(tuples, storage.d_col_ptrs.data(), storage.d_order.data(), n);
}

// =============================================================================
// Main sort entry point — arbitrary arity
// =============================================================================

inline void cub_lex_sort(uint32_t* const* columns, int num_cols, uint32_t n, const int* order,
                         GPU_STREAM_T stream, CubSortStorage& storage) {
  if (n <= 1 || num_cols == 0)
    return;

  constexpr int kBS = 256;
  int nb = (n + kBS - 1) / kBS;

  if (num_cols <= 2) {
    // === Arity 1-2: Pack into uint64, radix sort (O(n), fastest possible) ===
    int o0, o1;
    uint32_t* c0 = columns[0];
    uint32_t* c1 = (num_cols == 2) ? columns[1] : columns[0];
    if (num_cols == 2) {
      o0 = order[0];
      o1 = order[1];
    } else {
      o0 = 0;
      o1 = 0;  // arity-1: both point to same column
    }

    // Query temp size (first call only — CUB returns same value for same n)
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (uint64_t*)nullptr, (uint64_t*)nullptr,
                                   static_cast<int>(n), 0, 64, stream);
    storage.ensure_arity2(n, temp_bytes);

    // Pack SoA → uint64
    uint64_t* keys_in = storage.keys64.data();
    uint64_t* keys_out = storage.keys64.data() + n;

    cub_sort_pack_u64<<<nb, kBS, 0, stream>>>(c0, c1, keys_in, n, o0, o1);

    // Radix sort (single CUB call, fully stream-async)
    cub::DeviceRadixSort::SortKeys(storage.temp.data(), temp_bytes, keys_in, keys_out,
                                   static_cast<int>(n), 0, 64, stream);

    // Unpack uint64 → SoA
    cub_sort_unpack_u64<<<nb, kBS, 0, stream>>>(keys_out, c0, c1, n, o0, o1);

  } else {
    // === Arity 3+: Pack into right-sized AoS, merge sort ===
    // Static dispatch to avoid 32-byte SortTuple waste for small arities
    switch (num_cols) {
      case 3:
        cub_lex_sort_mergesort<3>(columns, n, order, stream, storage);
        break;
      case 4:
        cub_lex_sort_mergesort<4>(columns, n, order, stream, storage);
        break;
      case 5:
        cub_lex_sort_mergesort<5>(columns, n, order, stream, storage);
        break;
      case 6:
        cub_lex_sort_mergesort<6>(columns, n, order, stream, storage);
        break;
      case 7:
        cub_lex_sort_mergesort<7>(columns, n, order, stream, storage);
        break;
      case 8:
        cub_lex_sort_mergesort<8>(columns, n, order, stream, storage);
        break;
      default:
        assert(false && "Arity > 8 not supported");
        break;
    }
  }
}

/// Convenience overload with std::vector
inline void cub_lex_sort(std::vector<uint32_t*>& columns, uint32_t n, const std::vector<int>& order,
                         GPU_STREAM_T stream, CubSortStorage& storage) {
  cub_lex_sort(columns.data(), static_cast<int>(columns.size()), n, order.data(), stream, storage);
}

}  // namespace SRDatalog::GPU
