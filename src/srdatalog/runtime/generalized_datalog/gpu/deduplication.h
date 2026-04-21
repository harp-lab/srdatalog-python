/**
 * @file deduplication.h
 * @brief GPU-based deduplication on N-dimensional array.
 */
#pragma once

#include "../semiring.h"
#include "./device_sorted_array_index.h"
#include "./macro.h"
#include "./nd_device_array.h"
#ifdef USE_ROCm
// Define THRUST_IGNORE_CUB_VERSION_CHECK before any Thrust headers
// to avoid CUB version compatibility errors on ROCm
#define THRUST_IGNORE_CUB_VERSION_CHECK
#endif
// NVTX profiling support
#if defined(USE_ROCm)
// nvtx3 is not available on ROCm - provide stubs
#ifndef NVTX_STUB_DEFINED
#define NVTX_STUB_DEFINED
#define nvtxRangePushA(name) ((void)0)
#define nvtxRangePop() ((void)0)
namespace nvtx3 {
struct scoped_range {
  scoped_range(const char*) {}
};
}  // namespace nvtx3
#endif
#else
// Always include nvtx3 on CUDA builds
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>
#endif
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <vector>

namespace SRDatalog::GPU {

/**
 * @brief Deduplicate rows, aggregate provenance, and build root CSR in fused pass.
 *
 * @details This function combines three operations that share common computation:
 * 1. Deduplicate rows (all columns compared)
 * 2. Aggregate provenance using SR::add for duplicate rows
 * 3. Build CSR index for root level (first column only)
 *
 * ## Algorithm Flow
 * ```
 * Input (sorted):
 *   rows: [(A,X) (A,X) (A,Y) (B,X) (B,X)]
 *   prov: [  1     2     1     3     1  ]
 *
 * Step 1-2: Mark heads + positions (full row comparison)
 *   flags: [1 0 1 1 0]
 *   pos:   [0 0 1 2 2]
 *   num_unique = 3
 *
 * Step 3: Aggregate provenance
 *   agg_prov: [3, 1, 4]  (1+2, 1, 3+1)
 *
 * Step 4: Compact columns
 *   rows: [(A,X) (A,Y) (B,X)]
 *   prov: [  3     1     4  ]
 *
 * Step 5: Build CSR on first column (after compaction)
 *   csr_flags: [1 0 1]  (A != _, A == A, B != A)
 *   csr_values: [A, B]
 *   csr_meta: [(0,2), (2,3)]  (A spans [0,2), B spans [2,3))
 * ```
 *
 * ## Buffer Usage
 *
 * | Buffer       | Size       | Lifetime     | Used For                    | Reuse Potential |
 * |--------------|------------|--------------|-----------------------------|-----------------|
 * | head_flags   | num_rows   | Steps 1-4    | Mark unique rows (all cols) | (1)             |
 * | output_pos   | num_rows   | Steps 2-4    | Compaction positions        | (1)             |
 * | segment_ids  | num_rows   | Step 3       | Segment IDs for reduce      | (2)             |
 * | agg_prov     | num_unique | Steps 3-end  | Aggregated provenance       | None (output)   |
 * | csr_flags    | num_unique | Step 5       | Mark unique in first col    | (3)             |
 * | csr_pos      | num_unique | Step 5       | CSR segment positions       | (3)             |
 *
 * ## Future Buffer Reuse Opportunities
 *
 * (1) head_flags + output_pos: Could be combined by packing flag into position high bit.
 *     NOT IMPLEMENTED: Clarity > micro-optimization.
 *
 * (2) segment_ids: Could reuse output_pos after compaction since segment_ids =
 *     inclusive_scan(head_flags) and output_pos = exclusive_scan(head_flags).
 *     They differ by 1: segment_ids[i] = output_pos[i] + head_flags[i]
 *     NOT IMPLEMENTED: Extra arithmetic vs. one more scan. Profile first.
 *
 * (3) csr_flags + csr_pos: Operate on compacted data (num_unique elements).
 *     Could reuse head_flags/output_pos buffers since num_unique <= num_rows.
 *     NOT IMPLEMENTED: Keep separate for clarity and easier debugging.
 *
 * @tparam SR Semiring type for provenance aggregation
 * @tparam ValueType Column element type
 * @tparam N Number of columns (arity)
 *
 * @param[in,out] array       Sorted NDDeviceArray, compacted in-place
 * @param[in,out] provenance  Provenance values, aggregated and resized
 * @param[out]    root_unique_values  Unique values in first column (for partitioning only, not
 * correctness)
 *
 * @pre array must be sorted lexicographically
 * @pre provenance.size() == array.num_rows()
 * @post array.num_rows() == provenance.size() == number of unique rows
 * @post root_unique_values.size() == unique values in col 0
 *
 * @note This function uses array.column_ptr<I>() directly (not through a view) to avoid
 *       dangling pointers. If resize() reallocates, any pointers captured from a view would
 *       become invalid. Getting fresh pointers from the array ensures correctness.
 */
template <Semiring SR, typename ValueType, std::size_t N>
void deduplicate_aggregate_and_unique(NDDeviceArray<ValueType, N>& array,
                                      DeviceArray<semiring_value_t<SR>>& provenance,
                                      DeviceArray<ValueType>& root_unique_values);

/// @brief Single-thread kernel: dedup + compute root_unique in one launch for small arrays.
/// @details Avoids thrust overhead (rmm alloc + stream sync) for tiny sorted arrays.
///          Runs a single thread that walks the sorted data and compacts in-place.
template <typename ValueType, std::size_t N>
__global__ void small_dedup_kernel(ValueType* __restrict__ data, uint32_t stride, uint32_t num_rows,
                                   ValueType* __restrict__ root_unique,
                                   uint32_t* __restrict__ out_num_unique,
                                   uint32_t* __restrict__ out_num_root_unique) {
  // Single-thread kernel for small arrays
  uint32_t write = 0;
  uint32_t root_write = 0;

  // First row is always kept
  // (no copy needed — data[0] stays in place)
  root_unique[0] = data[0];  // first column value
  write = 1;
  root_write = 1;

  for (uint32_t i = 1; i < num_rows; ++i) {
    // Compare row i vs row (write-1) across all columns
    bool same = true;
#pragma unroll
    for (std::size_t col = 0; col < N; ++col) {
      if (data[(col * stride) + i] != data[(col * stride) + (write - 1)]) {
        same = false;
        break;
      }
    }
    if (!same) {
      // Compact: copy row i to position write
      if (write != i) {
#pragma unroll
        for (std::size_t col = 0; col < N; ++col) {
          data[(col * stride) + write] = data[(col * stride) + i];
        }
      }
      // Check if first-column value changed for root_unique
      if (data[write] != data[write - 1]) {
        root_unique[root_write] = data[write];
        root_write++;
      }
      write++;
    }
  }
  *out_num_unique = write;
  *out_num_root_unique = root_write;
}

/// @brief NoProvenance optimized overload - uses thrust::unique directly
/// @details When SR is NoProvenance, we can skip head_flags + scan + gather pattern
///          and use thrust::unique directly on zipped columns, which is more efficient.
template <Semiring SR, typename ValueType, std::size_t N>
requires(!has_provenance_v<SR>) void deduplicate_aggregate_and_unique(
    NDDeviceArray<ValueType, N>& array, [[maybe_unused]] std::monostate& provenance,
    DeviceArray<ValueType>& root_unique_values) {
  //==========================================================================
  // Trivial cases
  //==========================================================================
  if (array.num_rows() == 0) {
    root_unique_values.reset();
    return;
  }

  if (array.num_rows() == 1) {
    root_unique_values.resize(1);
    thrust::copy_n(rmm::exec_policy{}, array.template column_ptr<0>(), 1,
                   root_unique_values.begin());
    return;
  }

  const uint32_t num_rows = static_cast<uint32_t>(array.num_rows());

  //==========================================================================
  // Small-array fast path: single-thread kernel dedup + root_unique in one launch.
  // Avoids thrust's rmm alloc + stream sync overhead (~25us per thrust call).
  // Cost: 1 kernel launch (~3us) + 1 cudaMemcpy for 2 uint32s (~15us) = ~18us
  // vs thrust path: 2 kernel launches + 2 stream syncs = ~55us
  //==========================================================================
  static constexpr uint32_t kSmallDedupThreshold = 1024;
  if (num_rows <= kSmallDedupThreshold) {
    // Static pinned host memory for output counts — avoids per-call alloc/free overhead.
    // Kernel writes counts directly to CPU-visible mapped memory.
    static uint32_t* h_counts = [] {
      uint32_t* p = nullptr;
      cudaHostAlloc(&p, 2 * sizeof(uint32_t), cudaHostAllocMapped);
      return p;
    }();
    static uint32_t* d_counts = [] {
      uint32_t* p = nullptr;
      cudaHostGetDevicePointer(&p, h_counts, 0);
      return p;
    }();

    // Allocate root_unique to max possible size
    root_unique_values.resize(num_rows);

    // Single kernel: dedup in-place + build root_unique + write counts to pinned host
    small_dedup_kernel<ValueType, N><<<1, 1>>>(array.data(), static_cast<uint32_t>(array.stride()),
                                               num_rows, root_unique_values.data(), d_counts,
                                               d_counts + 1);
    cudaDeviceSynchronize();

    array.resize(h_counts[0]);
    root_unique_values.resize(h_counts[1]);
    return;
  }

  //==========================================================================
  // NoProvenance fast path: Use thrust::unique on zipped columns
  // This avoids: head_flags tabulate + exclusive_scan + reduce + copy_if + gather
  // Instead: single thrust::unique pass + unique_copy for root values
  //==========================================================================

  // Create zip iterator over all columns
  auto keys_begin = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return thrust::make_zip_iterator(thrust::make_tuple(array.template column_ptr<I>()...));
  }
  (std::make_index_sequence<N>());

  auto keys_end = keys_begin + num_rows;

  // thrust::unique operates in-place on sorted data
  nvtxRangePushA("Dedup_Unique_NoProvenance");
  auto new_end = thrust::unique(rmm::exec_policy{}, keys_begin, keys_end);
  nvtxRangePop();  // Dedup_Unique_NoProvenance

  const uint32_t num_unique = new_end - keys_begin;
  array.resize(num_unique);

  // Build root_unique_values from first column using unique_copy
  generate_unique(array, root_unique_values);
}

/// @brief Main implementation of deduplicate_aggregate_and_unique
template <Semiring SR, typename ValueType, std::size_t N>
void deduplicate_aggregate_and_unique(NDDeviceArray<ValueType, N>& array,
                                      DeviceArray<semiring_value_t<SR>>& provenance,
                                      DeviceArray<ValueType>& root_unique_values) {
  //==========================================================================
  // Trivial cases
  //==========================================================================
  if (array.num_rows() == 0) {
    root_unique_values.reset();
    return;
  }

  if (array.num_rows() == 1) {
    // Single element: no deduplication needed, directly build CSR
    // Short-circuit: just copy the single value
    root_unique_values.resize(1);
    thrust::copy_n(rmm::exec_policy{}, array.template column_ptr<0>(), 1,
                   root_unique_values.begin());
    return;
  }

  const uint32_t num_rows = static_cast<uint32_t>(array.num_rows());

  // Critical: Get column pointers directly from array (not from view) to avoid dangling pointers.
  // If resize() reallocates memory, any pointers captured from a view become invalid.
  // Using array.column_ptr<I>() ensures we always get fresh pointers to the current storage.

  //==========================================================================
  // Key iterator for full-row comparison
  //==========================================================================
  auto keys = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return thrust::make_zip_iterator(thrust::make_tuple(array.template column_ptr<I>()...));
  }
  (std::make_index_sequence<N>());

  //==========================================================================
  // Step 1: Mark segment heads (full row comparison)
  // Buffer: head_flags[num_rows] - uint32_t
  // head_flags[i] = 1 if row[i] != row[i-1], else 0
  //==========================================================================
  DeviceArray<uint32_t> head_flags(num_rows);

  nvtxRangePushA("Dedup_Tabulate");
  thrust::tabulate(rmm::exec_policy{}, head_flags.begin(), head_flags.end(),
                   [keys] __device__(uint32_t idx) -> uint32_t {
                     if (idx == 0)
                       return 1;
                     return (keys[idx] != keys[idx - 1]) ? 1 : 0;
                   });
  nvtxRangePop();  // Dedup_Tabulate

  //==========================================================================
  // Step 2: Compute output positions via exclusive scan
  // Buffer: output_pos[num_rows] - uint32_t
  // output_pos[i] = number of unique rows before index i
  // Invariant: output_pos[i] <= i (always)
  //==========================================================================
  DeviceArray<uint32_t> output_pos(num_rows);
  // Use rmm::exec_policy() for proper memory allocation through RMM pool
  thrust::exclusive_scan(rmm::exec_policy{}, head_flags.begin(), head_flags.end(),
                         output_pos.begin(), uint32_t{0}, thrust::plus<uint32_t>());

  // Get unique count by summing flags (count of 1s in head_flags)
  const uint32_t num_unique =
      thrust::reduce(rmm::exec_policy{}, head_flags.begin(), head_flags.end(), uint32_t{0},
                     thrust::plus<uint32_t>());

  //==========================================================================
  // Step 3: Aggregate provenance by segment (only if has provenance)
  // Buffer: segment_ids[num_rows] - uint32_t (could reuse, see notes above)
  // Buffer: agg_prov[num_unique] - SR::value_type
  // segment_ids[i] = 1-indexed segment ID for row i (count of segment heads up to i)
  //==========================================================================
  // For NoProvenance, we skip segment_ids and provenance aggregation entirely
  [[maybe_unused]] DeviceArray<semiring_value_t<SR>> agg_prov;
  if constexpr (has_provenance_v<SR>) {
    DeviceArray<uint32_t> segment_ids(num_rows);

    // Count segment heads using arithmetic plus (not boolean OR - we need numeric IDs for grouping)
    // bool implicitly converts: true→1, false→0, so this counts how many segment heads seen so far
    // Use rmm::exec_policy() for proper memory allocation through RMM pool
    thrust::inclusive_scan(rmm::exec_policy{}, head_flags.begin(), head_flags.end(),
                           segment_ids.begin(), thrust::plus<uint32_t>());

    agg_prov.resize(num_unique);

    // CRITICAL: Ensure provenance size matches array size before reduce_by_key
    // If provenance is too small, resize and fill with semiring identity to prevent
    // out-of-bounds access in reduce_by_key which would cause incorrect aggregation
    if (provenance.size() != num_rows) {
      if (provenance.size() < num_rows) {
        const auto old_size = provenance.size();
        provenance.resize(num_rows);
        // Fill new elements with semiring identity (one)
        if (old_size > 0) {
          thrust::fill(rmm::exec_policy{}, provenance.begin() + old_size, provenance.end(),
                       sr_one<SR>());
        } else {
          thrust::fill(rmm::exec_policy{}, provenance.begin(), provenance.end(), sr_one<SR>());
        }
      } else {
        // Provenance is larger than array - this is a serious error
        throw std::runtime_error("deduplicate_aggregate_and_unique: Provenance size (" +
                                 std::to_string(provenance.size()) + ") > array rows (" +
                                 std::to_string(num_rows) + "). This indicates a logic error.");
      }
    }

    {
      nvtx3::scoped_range r("deduplicate_aggregate_and_build_csr::reduce_by_key");
      // Use rmm::exec_policy() for proper memory allocation through RMM pool
      thrust::reduce_by_key(
          rmm::exec_policy{}, segment_ids.begin(), segment_ids.end(), provenance.begin(),
#ifdef USE_ROCm
          rocprim::make_discard_iterator(), agg_prov.begin(), thrust::equal_to<uint32_t>(),
#else
          thrust::make_discard_iterator(), agg_prov.begin(), thrust::equal_to<uint32_t>(),
#endif
          [] __device__(semiring_value_t<SR> a, semiring_value_t<SR> b) { return SR::add(a, b); });
    }
  }

  //==========================================================================
  // Step 4: Compact columns using gather
  // 1. Compute gather indices (indices of unique elements)
  // 2. Gather each column into temp buffer, then copy back
  // This is robust against race conditions and ensures correct compaction
  //==========================================================================
  DeviceArray<uint32_t> gather_indices(num_unique);
  nvtxRangePushA("Dedup_CopyIf");
  thrust::copy_if(rmm::exec_policy{}, thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(num_rows), head_flags.begin(),
                  gather_indices.begin(), [] __device__(uint32_t flag) { return flag == 1; });
  nvtxRangePop();  // Dedup_CopyIf

  // Use safe compaction with temporary buffer and gather
  // 1. Allocate temp buffer
  DeviceArray<ValueType> temp_col(num_unique);
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    auto compact = [&](auto dim_const) {
      constexpr std::size_t dim = decltype(dim_const)::value;
      // Get fresh column pointer directly from array to ensure correct stride
      ValueType* col = array.template column_ptr<dim>();

      // 2. Gather to temp buffer
      nvtxRangePushA("Dedup_Gather");
      thrust::gather(rmm::exec_policy{}, gather_indices.begin(), gather_indices.end(),
                     thrust::device_ptr<ValueType>(col), temp_col.begin());
      nvtxRangePop();  // Dedup_Gather

      // 3. Copy back to original column - use GPU_MEMCPY for better performance
      nvtxRangePushA("Dedup_Copy_Back");
      GPU_MEMCPY(col, temp_col.data(), num_unique * sizeof(ValueType), GPU_DEVICE_TO_DEVICE);
      nvtxRangePop();  // Dedup_Copy_Back
    };
    (compact(std::integral_constant<std::size_t, I>{}), ...);
  }
  (std::make_index_sequence<N>());

  array.resize(num_unique);
  // Resize and copy aggregated provenance (only if has provenance)
  if constexpr (has_provenance_v<SR>) {
    provenance.resize(num_unique);
    nvtxRangePushA("Dedup_Copy_Provenance");
    // Use GPU_MEMCPY for better performance (DMA vs kernel launch)
    GPU_MEMCPY(provenance.data(), agg_prov.data(), num_unique * sizeof(semiring_value_t<SR>),
               GPU_DEVICE_TO_DEVICE);
    nvtxRangePop();  // Dedup_Copy_Provenance
  }

  //==========================================================================
  // Step 5: Build CSR on first column (compacted data)
  // Use the separated build_csr_from_deduplicated_array function
  // Note: After resize(), we must ensure any cached pointers are refreshed.
  // build_csr_from_deduplicated_array will get a fresh column_ptr<0>() from the array,
  // so it should be safe.
  //==========================================================================
  generate_unique(array, root_unique_values);
}

/**
 * @brief Build root CSR from sorted, deduplicated data (first column only)
 *
 * @details This function builds CSR index on the first column of sorted, deduplicated data.
 * It identifies unique values in the first column and builds (begin, end) pairs for each.
 *
 * This function uses array.column_ptr<0>() directly (not through a view) to avoid dangling
 * pointers. If the array was recently resized, using a view could result in stale pointers.
 *
 * @param array Sorted, deduplicated NDDeviceArray (compacted data)
 * @param[out] root_unique_values Unique values in first column
 *
 * @pre array is sorted lexicographically
 * @pre array is deduplicated (no duplicate rows)
 * @post root_unique_values.size() == unique values in column 0
 */
template <typename ValueType, std::size_t N>
void generate_unique(const NDDeviceArray<ValueType, N>& array,
                     DeviceArray<ValueType>& root_unique_values) {
  const uint32_t num_rows = static_cast<uint32_t>(array.num_rows());

  if (num_rows == 0) {
    root_unique_values.reset();
    return;
  }

  if (num_rows == 1) {
    // Single element: directly build CSR
    root_unique_values.resize(1);
    thrust::copy_n(rmm::exec_policy{}, array.template column_ptr<0>(), 1,
                   root_unique_values.begin());
    return;
  }

  // Get column pointer directly (fresh pointer, not from view) to ensure correct stride
  const ValueType* first_col = array.template column_ptr<0>();

  // Step 1: Unique copy using iterator_unique (or reduce_by_key with discard)
  // We only need the unique keys, not the counts.
  // However, thrust::unique_copy is for contiguous duplicates.
  //
  // Option A: Use unique_copy with start/end of array.column_ptr<0>()
  // Option B: Keep reduce_by_key but discard values.
  //
  // reduce_by_key is optimized and we already have it. Let's just discard the counts.

  // Resize to maximum possible size (all unique) first
  root_unique_values.resize(num_rows);

  // Use unique_copy to extract unique keys
  // This is semantically equivalent to reduce_by_key on keys with discard on values,
  // but potentially more robust or optimized for this specific case.
  auto result_end = thrust::unique_copy(rmm::exec_policy{}, first_col, first_col + num_rows,
                                        root_unique_values.begin());

  const uint32_t num_csr = result_end - root_unique_values.begin();

  // Resize outputs to actual number of unique keys
  root_unique_values.resize(num_csr);
}

}  // namespace SRDatalog::GPU