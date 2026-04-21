/**
 * @file thrust_sort.h
 * @brief GPU version of sort function in sort.h
 * @details This file is a GPU version of the sort function in sort.h. It uses Thrust to perform the
 * sorting. See https://github.com/NVIDIA/cccl/blob/main/thrust/examples/lexicographical_sort.cu for
 * the original implementation.
 */

#pragma once

#include "device_array.h"
#include "thrust/gather.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"
#include <cstddef>
#include <ranges>
#include <rmm/exec_policy.hpp>
#include <span>
#include <thrust/execution_policy.h>
#include <tuple>
#include <utility>

namespace SRDatalog::GPU {

/**
 * @brief Context structure to hold reusable buffers for sorting operations
 *
 * This context is slightly simpler than Highway's SortContext because we don't need to
 * manually push the index into the context, sort_by_key works well on GPU.
 *
 * The context maintains device-side buffers that are reused across multiple
 * sorting operations to avoid repeated allocations.
 */
template <typename I = uint32_t>
struct SortContext {
  DeviceArray<I> permutation;  ///< Permutation buffer (indices)

  /**
   * @brief Resize permutation buffer to accommodate n elements
   * @param n Number of elements to sort
   */
  void resize_permutation(size_t n) {
    permutation.resize(n);
  }

  /**
   * @brief Get pointer to sorted indices after sorting
   * @param _n Just for match what's in sort.h (unused)
   * @return Pointer to sorted indices on device
   */
  const I* get_sorted_indices([[maybe_unused]] size_t _n) const {
    return permutation.data();
  }

  /**
   * @brief Get temporary buffer for keys of type KeyType
   * @tparam KeyType Key type
   * @param n Number of elements needed
   * @return Reference to DeviceArray<KeyType> with at least n elements
   *
   */
  template <typename KeyType>
  DeviceArray<KeyType>& get_temp_keys(size_t n) {
    // Each KeyType gets its own thread-local buffer via template instantiation
    // This is type-safe, simple, and efficient - no type erasure needed
    static thread_local DeviceArray<KeyType> temp;
    if (temp.size() < n) {
      temp.resize(n);
    }
    return temp;
  }
};

/**
 * @brief Update permutation by sorting keys
 * @tparam K Key type
 * @tparam I Index type
 * @param keys Pointer to key data on device
 * @param n Number of elements
 * @param permutation Pointer to permutation indices on device
 * @param ctx SortContext to manage temporary buffers
 */
template <typename K, typename I = uint32_t>
void update_permutation(K* keys, size_t n, I* permutation, SortContext<I>& ctx) {
  // Get temporary storage for keys from context
  auto& temp = ctx.template get_temp_keys<K>(n);

  // Permute the keys with the current reordering
  thrust::gather(thrust::device, thrust::device_ptr<I>(permutation),
                 thrust::device_ptr<I>(permutation) + n, thrust::device_ptr<K>(keys), temp.begin());

  // Stable sort the permuted keys and update the permutation
  // Use rmm::exec_policy() for proper memory allocation through RMM pool
  thrust::stable_sort_by_key(rmm::exec_policy{}, temp.begin(), temp.begin() + n,
                             thrust::device_ptr<I>(permutation));
}

/**
 * @brief Apply permutation to keys
 * @tparam K Key type
 * @tparam I Index type
 * @param keys Pointer to key data on device (will be permuted in-place)
 * @param n Number of elements
 * @param permutation Pointer to permutation indices on device
 * @param ctx SortContext to manage temporary buffers
 */
template <typename K, typename I = uint32_t>
void apply_permutation(K* keys, size_t n, I* permutation, SortContext<I>& ctx) {
  // Get temporary storage for keys from context
  auto& temp = ctx.template get_temp_keys<K>(n);

  // Copy keys to temporary buffer
  thrust::copy(thrust::device, thrust::device_ptr<K>(keys), thrust::device_ptr<K>(keys) + n,
               temp.begin());

  // Permute the keys
  thrust::gather(thrust::device, thrust::device_ptr<I>(permutation),
                 thrust::device_ptr<I>(permutation) + n, temp.begin(), thrust::device_ptr<K>(keys));
}

/**
 * @brief Stable lexicographic sort implementation (internal) with context - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort_impl in sort.h
 * for detailed documentation.
 *
 * @tparam I Index type (default: uint32_t)
 * @tparam Ts... Column data types
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority
 * @param ctx Reference to SortContext for buffer reuse and result storage
 * @param reorder_data If true, reorder the column data in place using sorted indices
 * @param cols... Pointers to column data arrays (one per column type in Ts...)
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort_impl(size_t n, std::span<const int> order, SortContext<I>& ctx,
                          bool reorder_data, Ts*... cols) {
  if (n == 0) {
    return;
  }

  ctx.resize_permutation(n);

  // Initialize permutation to [0, 1, 2, ..., n-1]
  I* permutation = const_cast<I*>(ctx.permutation.data());
  thrust::sequence(thrust::device, thrust::device_ptr<I>(permutation),
                   thrust::device_ptr<I>(permutation) + n);

  // Store column pointers in a tuple for indexed access
  std::tuple<Ts*...> col_tuple(cols...);
  constexpr size_t num_cols = sizeof...(Ts);

  // Sort from least significant column to most significant column
  // Process order in reverse (least significant first)
  for (int col_idx : std::ranges::reverse_view(order)) {
    if (col_idx < 0 || static_cast<size_t>(col_idx) >= num_cols) {
      continue;  // Skip invalid column indices
    }

    // Extract the column pointer by index and call update_permutation
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((void)((col_idx == static_cast<int>(Is) &&
               (update_permutation(std::get<Is>(col_tuple), n, permutation, ctx), true))),
       ...);
    }(std::make_index_sequence<num_cols>{});
  }

  // Optionally apply permutation to columns (for in-place reordering)
  if (reorder_data) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (apply_permutation(std::get<Is>(col_tuple), n, permutation, ctx), ...);
    }(std::make_index_sequence<num_cols>{});
  }
}

/**
 * @brief Stable lexicographic sort implementation (internal) with output array - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort_impl in sort.h
 * for detailed documentation.
 *
 * @tparam I Index type (default: uint32_t)
 * @tparam Ts... Column data types
 * @param ids_out Output array of sorted indices (must have space for n elements)
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority
 * @param reorder_data If true, reorder the column data in place using sorted indices
 * @param cols... Pointers to column arrays
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort_impl(I* ids_out, size_t n, std::span<const int> order, bool reorder_data,
                          Ts*... cols) {
  SortContext<I> ctx;
  stable_lex_sort_impl<I>(n, order, ctx, reorder_data, cols...);
  thrust::copy(thrust::device, thrust::device_ptr<I>(ctx.permutation.data()),
               thrust::device_ptr<I>(ctx.permutation.data()) + n, thrust::device_ptr<I>(ids_out));
}

/**
 * @brief Stable lexicographic sort (public API) with context - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort in sort.h
 * for detailed documentation.
 *
 * @tparam I Index type for output (default: uint32_t)
 * @tparam Ts... Types of the column arrays
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority
 * @param ctx Reference to SortContext for buffer reuse and result storage
 * @param reorder_data If true, reorder the column data in place using sorted indices
 * @param cols... Pointers to column data arrays (one per column)
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort(size_t n, std::span<const int> order, SortContext<I>& ctx, bool reorder_data,
                     Ts*... cols) {
  stable_lex_sort_impl<I>(n, order, ctx, reorder_data, cols...);
}

/**
 * @brief Stable lexicographic sort (public API) with context - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort in sort.h
 * for detailed documentation. This overload defaults reorder_data to false.
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort(size_t n, std::span<const int> order, SortContext<I>& ctx, Ts*... cols) {
  stable_lex_sort<I>(n, order, ctx, false, cols...);
}

/**
 * @brief Stable lexicographic sort (public API) without context - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort in sort.h
 * for detailed documentation.
 *
 * @tparam I Index type for output (default: uint32_t)
 * @tparam Ts... Types of the column arrays
 * @param ids_out Output array of sorted row indices (must have space for n elements)
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority
 * @param reorder_data If true, reorder the column data in place using sorted indices
 * @param cols... Pointers to column data arrays (one per column)
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort(I* ids_out, size_t n, std::span<const int> order, bool reorder_data,
                     Ts*... cols) {
  stable_lex_sort_impl<I>(ids_out, n, order, reorder_data, cols...);
}

/**
 * @brief Stable lexicographic sort (public API) without context - GPU version
 *
 * GPU-accelerated version using Thrust. See hwy_lexsort::stable_lex_sort in sort.h
 * for detailed documentation. This overload defaults reorder_data to false.
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort(I* ids_out, size_t n, std::span<const int> order, Ts*... cols) {
  stable_lex_sort<I>(ids_out, n, order, false, cols...);
}

/**
 * @brief In-place unstable sort (public API) - GPU version
 *
 * GPU-accelerated version using Thrust::sort. See hwy_lexsort::unstable_sort in sort.h
 * for detailed documentation.
 *
 * @tparam T Element type (must be supported by Thrust sort)
 * @param data Pointer to array to sort on device (modified in-place)
 * @param n Number of elements to sort
 */
template <typename T>
void unstable_sort(T* data, size_t n) {
  if (data == nullptr || n == 0) {
    return;
  }
  // Use rmm::exec_policy() for proper memory allocation through RMM pool
  thrust::sort(rmm::exec_policy{}, thrust::device_ptr<T>(data), thrust::device_ptr<T>(data) + n);
}

/**
 * @brief Sort SoA columns in place using zip iterator (single-pass merge sort)
 *
 * Uses thrust::sort on a zip_iterator over all column pointers. This performs
 * lexicographic comparison across all columns in a single pass, avoiding the
 * multi-pass radix sort + gather overhead of stable_lex_sort.
 *
 * Benchmark results (100M tuples, 10 runs):
 *   Arity 2: Zip 364ms vs Lex-Gather 406ms (10% faster)
 *   Arity 3: Zip 558ms vs Lex-Gather 680ms (18% faster)
 *   Arity 4: Zip 749ms vs Lex-Gather 1016ms (36% faster)
 *
 * @tparam Ts... Column element types (must support operator<)
 * @param n Number of rows to sort
 * @param cols... Pointers to column data arrays on device (modified in-place)
 */
template <typename... Ts>
void zip_sort_columns(size_t n, Ts*... cols) {
  if (n <= 1)
    return;
  auto keys = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<Ts>(cols)...));
  thrust::sort(rmm::exec_policy{}, keys, keys + n);
}

}  // namespace SRDatalog::GPU
