#pragma once
/**
 * @file tiny_sort_dedup.h
 * @brief Fused sort+dedup kernel for tiny inputs (<= 1024 tuples).
 *
 * Two-pass CUB BlockRadixSort for full lexicographic order on arity 3-4.
 * CUB BlockScan for parallel prefix sum on uniqueness flags.
 * ~2-5x faster than thrust sort+dedup pipeline for <1024 tuples.
 */

#include <cstdint>
#include <cub/cub.cuh>

namespace SRDatalog::GPU::JIT {

static constexpr int kTinySortBlock = 256;
static constexpr int kTinySortItemsPerThread = 4;
static constexpr int kTinySortMaxItems = kTinySortBlock * kTinySortItemsPerThread;  // 1024

template <typename ValueType, int Arity>
__device__ __forceinline__ bool tuples_equal(const ValueType* a, uint32_t ai, uint32_t as,
                                             const ValueType* b, uint32_t bi, uint32_t bs) {
#pragma unroll
  for (int c = 0; c < Arity; c++)
    if (a[c * as + ai] != b[c * bs + bi])
      return false;
  return true;
}

/// Fused sort + self-dedup kernel for SoA tuples.
/// Two-pass radix sort for full lexicographic order, CUB BlockScan for compaction.
///
/// Launch as: <<<1, kTinySortBlock>>>
/// Input must have <= kTinySortMaxItems (1024) tuples.
template <typename ValueType, int Arity>
__global__ void tiny_sort_dedup_kernel(
    const ValueType* __restrict__ in_data, uint32_t in_count, uint32_t in_stride,
    const int* __restrict__ col_order,  // column permutation (e.g., {2,1,0,3})
    ValueType* __restrict__ out_data, uint32_t* __restrict__ out_count, uint32_t out_stride) {

  using BlockSortT =
      cub::BlockRadixSort<uint64_t, kTinySortBlock, kTinySortItemsPerThread, uint32_t>;
  using BlockScanT = cub::BlockScan<uint32_t, kTinySortBlock, cub::BLOCK_SCAN_RAKING, 1, 1>;

  __shared__ union {
    typename BlockSortT::TempStorage sort;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  __shared__ ValueType s_sorted[Arity * kTinySortMaxItems];

  uint32_t tid = threadIdx.x;
  uint64_t my_keys[kTinySortItemsPerThread];
  uint32_t my_vals[kTinySortItemsPerThread];

  // Init row indices
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    uint32_t idx = tid * kTinySortItemsPerThread + i;
    my_vals[i] = (idx < in_count) ? idx : 0xFFFFFFFFu;
  }

  // === SORT: Two-pass stable radix sort ===
  // Pass 1: sort by low-order columns (col2, col3) if arity > 2
  if constexpr (Arity > 2) {
#pragma unroll
    for (int i = 0; i < kTinySortItemsPerThread; i++) {
      uint32_t orig = my_vals[i];
      if (orig < in_count) {
        auto c2 = in_data[col_order[2] * in_stride + orig];
        auto c3 = (Arity >= 4) ? in_data[col_order[3] * in_stride + orig] : ValueType(0);
        my_keys[i] = ((uint64_t)c2 << 32) | c3;
      } else {
        my_keys[i] = 0xFFFFFFFFFFFFFFFFull;
      }
    }
    BlockSortT(temp_storage.sort).Sort(my_keys, my_vals);
    __syncthreads();
  }

  // Pass 2: sort by high-order columns (col0, col1)
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    uint32_t orig = my_vals[i];
    if (orig < in_count) {
      auto c0 = in_data[col_order[0] * in_stride + orig];
      auto c1 = (Arity >= 2) ? in_data[col_order[1] * in_stride + orig] : ValueType(0);
      my_keys[i] = ((uint64_t)c0 << 32) | c1;
    } else {
      my_keys[i] = 0xFFFFFFFFFFFFFFFFull;
    }
  }
  BlockSortT(temp_storage.sort).Sort(my_keys, my_vals);
  __syncthreads();

  // === GATHER sorted tuples into shared memory ===
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    uint32_t slot = tid * kTinySortItemsPerThread + i;
    uint32_t orig = my_vals[i];
    if (orig < in_count) {
#pragma unroll
      for (int c = 0; c < Arity; c++)
        s_sorted[c * kTinySortMaxItems + slot] = in_data[col_order[c] * in_stride + orig];
    } else {
#pragma unroll
      for (int c = 0; c < Arity; c++)
        s_sorted[c * kTinySortMaxItems + slot] = ~ValueType(0);
    }
  }
  __syncthreads();

  // === DEDUP: compute unique flags + parallel prefix scan + scatter ===
  // Each thread computes flags for its kTinySortItemsPerThread slots,
  // then CUB BlockScan computes exclusive prefix sum in parallel.

  uint32_t my_flags[kTinySortItemsPerThread];
  uint32_t my_positions[kTinySortItemsPerThread];

#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    uint32_t slot = tid * kTinySortItemsPerThread + i;
    if (slot < in_count) {
      if (slot == 0) {
        my_flags[i] = 1;
      } else {
        my_flags[i] = tuples_equal<ValueType, Arity>(s_sorted, slot, kTinySortMaxItems, s_sorted,
                                                     slot - 1, kTinySortMaxItems)
                          ? 0
                          : 1;
      }
    } else {
      my_flags[i] = 0;
    }
  }

  // Parallel prefix scan: each thread scans its items sequentially,
  // then CUB BlockScan combines across threads.
  // Thread-local exclusive scan first:
  uint32_t thread_prefix = 0;
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    my_positions[i] = thread_prefix;
    thread_prefix += my_flags[i];
  }
  // thread_prefix = total unique items in this thread's chunk
  // Now scan thread_prefix across threads to get global offsets
  uint32_t thread_offset;
  uint32_t block_total;
  BlockScanT(temp_storage.scan).ExclusiveSum(thread_prefix, thread_offset, block_total);
  __syncthreads();

  // Add thread_offset to each position
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++)
    my_positions[i] += thread_offset;

  // Write total count
  if (tid == 0)
    *out_count = block_total;

  // === SCATTER unique elements at scanned positions ===
#pragma unroll
  for (int i = 0; i < kTinySortItemsPerThread; i++) {
    uint32_t slot = tid * kTinySortItemsPerThread + i;
    if (slot >= in_count)
      continue;
    if (my_flags[i]) {
      uint32_t pos = my_positions[i];
#pragma unroll
      for (int c = 0; c < Arity; c++)
        out_data[c * out_stride + pos] = s_sorted[c * kTinySortMaxItems + slot];
    }
  }
}

/// Threshold below which tiny_sort_dedup_kernel is used instead of thrust.
static constexpr uint32_t kTinySortThreshold = 1024;

}  // namespace SRDatalog::GPU::JIT
