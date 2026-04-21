/**
 * @file materialized_join.h
 * @brief Runtime helpers for materialized binary join with row ID pairing
 *
 * Provides device-callable functions for the materialized join pattern:
 * 1. probe_count_matches - Count matches per input key (for load balancing)
 * 2. probe_materialize_pairs - Materialize (left_rowid, right_rowid) pairs
 * 3. gather_column - Gather column values using row IDs
 *
 * These helpers use merge-path load balancing for unbalanced join outputs.
 */

#pragma once

#include <cstdint>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>

namespace SRDatalog::GPU::JIT {

// ============================================================================
// Probe Count Matches
// ============================================================================

/**
 * @brief Device kernel to count matches for each input key
 *
 * For load balancing: counts how many matches each input row produces.
 * Output sizes are used for prefix sum to compute output offsets.
 * Uses binary search on col_data_ (first column of sorted index).
 *
 * @tparam KeyT Key type (typically int32_t)
 * @tparam ViewT Index view type with col_data_ and num_rows_
 */
template <typename KeyT, typename ViewT>
__global__ void probe_count_matches_kernel(const uint32_t* __restrict__ input_rowids,
                                           const KeyT* __restrict__ input_keys, const ViewT view,
                                           uint32_t* __restrict__ counts, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  KeyT key = input_keys[tid];

  // Binary search on first column (col_data_) to find range
  const KeyT* col0 = view.col_data_;
  uint32_t num_rows = view.num_rows_;

  // Find lower bound (first key >= search key)
  uint32_t lo = 0, hi = num_rows;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (col0[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint32_t range_begin = lo;

  // Find upper bound (first key > search key)
  hi = num_rows;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (col0[mid] <= key)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint32_t range_end = lo;

  counts[tid] = range_end - range_begin;
}

/**
 * @brief Host wrapper for probe count matches
 */
template <typename KeyT, typename ViewT>
void probe_count_matches(const thrust::device_vector<uint32_t>& input_rowids,
                         const thrust::device_vector<KeyT>& input_keys, const ViewT& view,
                         thrust::device_vector<uint32_t>& counts, cudaStream_t stream = 0) {
  uint32_t n = static_cast<uint32_t>(input_rowids.size());
  if (n == 0)
    return;

  constexpr int kBlockSize = 256;
  int numBlocks = (n + kBlockSize - 1) / kBlockSize;

  probe_count_matches_kernel<<<numBlocks, kBlockSize, 0, stream>>>(
      thrust::raw_pointer_cast(input_rowids.data()), thrust::raw_pointer_cast(input_keys.data()),
      view, thrust::raw_pointer_cast(counts.data()), n);
}

// ============================================================================
// Probe Materialize Pairs
// ============================================================================

/**
 * @brief Device kernel to materialize (left_rowid, right_rowid) pairs
 *
 * Each thread handles one input, writing to its output range [offset, offset+count).
 * Uses pre-computed offsets from exclusive_scan(counts).
 * Uses binary search on col_data_ for range finding.
 *
 * @tparam KeyT Key type
 * @tparam ViewT Index view type with col_data_ and num_rows_
 */
template <typename KeyT, typename ViewT>
__global__ void probe_materialize_pairs_kernel(const uint32_t* __restrict__ input_rowids,
                                               const KeyT* __restrict__ input_keys,
                                               const ViewT view,
                                               const uint32_t* __restrict__ offsets,
                                               uint32_t* __restrict__ out_left,
                                               uint32_t* __restrict__ out_right, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  uint32_t left_rowid = input_rowids[tid];
  KeyT key = input_keys[tid];

  // Binary search on first column to find range
  const KeyT* col0 = view.col_data_;
  uint32_t num_rows = view.num_rows_;

  // Find lower bound
  uint32_t lo = 0, hi = num_rows;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (col0[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint32_t range_begin = lo;

  // Find upper bound
  hi = num_rows;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (col0[mid] <= key)
      lo = mid + 1;
    else
      hi = mid;
  }
  uint32_t range_end = lo;

  uint32_t out_offset = offsets[tid];
  for (uint32_t i = range_begin; i < range_end; ++i) {
    out_left[out_offset] = left_rowid;
    out_right[out_offset] = i;  // Right side row ID
    ++out_offset;
  }
}

/**
 * @brief Host wrapper for probe materialize pairs
 */
template <typename KeyT, typename ViewT>
void probe_materialize_pairs(const thrust::device_vector<uint32_t>& input_rowids,
                             const thrust::device_vector<KeyT>& input_keys, const ViewT& view,
                             const thrust::device_vector<uint32_t>& offsets,
                             thrust::device_vector<uint32_t>& out_left,
                             thrust::device_vector<uint32_t>& out_right, cudaStream_t stream = 0) {
  uint32_t n = static_cast<uint32_t>(input_rowids.size());
  if (n == 0)
    return;

  constexpr int kBlockSize = 256;
  int numBlocks = (n + kBlockSize - 1) / kBlockSize;

  probe_materialize_pairs_kernel<<<numBlocks, kBlockSize, 0, stream>>>(
      thrust::raw_pointer_cast(input_rowids.data()), thrust::raw_pointer_cast(input_keys.data()),
      view, thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(out_left.data()),
      thrust::raw_pointer_cast(out_right.data()), n);
}

// ============================================================================
// Gather Column
// ============================================================================

/**
 * @brief Device kernel to gather column values using row IDs
 *
 * Simple gather operation: output[i] = column[rowids[i]]
 *
 * @tparam T Column value type
 */
template <typename T>
__global__ void gather_column_kernel(const uint32_t* __restrict__ rowids,
                                     const T* __restrict__ column_data, T* __restrict__ output,
                                     uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  output[tid] = column_data[rowids[tid]];
}

/**
 * @brief Host wrapper for gather column
 *
 * Uses Thrust gather for simplicity, which handles edge cases well.
 */
template <typename T>
void gather_column(const thrust::device_vector<uint32_t>& rowids,
                   const thrust::device_vector<T>& column_data, thrust::device_vector<T>& output,
                   cudaStream_t stream = 0) {
  auto policy = thrust::cuda::par.on(stream);
  thrust::gather(policy, rowids.begin(), rowids.end(), column_data.begin(), output.begin());
}

/**
 * @brief Overload using raw pointers for JIT codegen
 */
template <typename T>
void gather_column_raw(const uint32_t* rowids, const T* column_data, T* output, uint32_t n,
                       cudaStream_t stream = 0) {
  if (n == 0)
    return;

  constexpr int kBlockSize = 256;
  int numBlocks = (n + kBlockSize - 1) / kBlockSize;

  gather_column_kernel<<<numBlocks, kBlockSize, 0, stream>>>(rowids, column_data, output, n);
}

// ============================================================================
// Merge-Path Load Balancing Utilities
// ============================================================================

/**
 * @brief Compute output offsets using exclusive scan
 *
 * After counting matches, compute where each input's output starts.
 * Returns total output size.
 */
inline uint32_t compute_output_offsets(thrust::device_vector<uint32_t>& counts,
                                       thrust::device_vector<uint32_t>& offsets,
                                       cudaStream_t stream = 0) {
  auto policy = thrust::cuda::par.on(stream);
  uint32_t n = static_cast<uint32_t>(counts.size());

  // Exclusive scan to get output offsets
  thrust::exclusive_scan(policy, counts.begin(), counts.end(), offsets.begin(), 0u);

  // Total output size = last offset + last count
  uint32_t last_offset, last_count;
  cudaMemcpyAsync(&last_offset, thrust::raw_pointer_cast(offsets.data()) + n - 1, sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&last_count, thrust::raw_pointer_cast(counts.data()) + n - 1, sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return last_offset + last_count;
}

}  // namespace SRDatalog::GPU::JIT
