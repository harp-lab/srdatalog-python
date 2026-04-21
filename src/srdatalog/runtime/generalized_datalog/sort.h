/**
 * @file sort.h
 * @brief High-performance lexicographic sorting using Highway SIMD library
 *
 * This header provides two main sorting functions:
 * - stable_lex_sort: Performs stable lexicographic sorting on multiple columns
 * - unstable_sort: Performs in-place unstable sorting on a single array
 *
 * The implementation uses Highway's VQSort for vectorized sorting operations,
 * automatically dispatching to the best available SIMD instruction set (AVX2,
 * AVX-512, NEON, etc.) at compile time.
 *
 * @note This file uses Highway's target-specific compilation model. The SIMD
 *       implementation is compiled separately for each target architecture.
 */

#pragma once

// Highway headers are included at global scope in relation_col.h
// to avoid namespace issues. If this header is included independently,
// include Highway headers here as well.
#ifndef HWY_HIGHWAY_H
#include <hwy/aligned_allocator.h>
#include <hwy/contrib/sort/vqsort.h>
#include <hwy/highway.h>
#endif

#include "system.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

/**
 * @namespace hwy_lexsort
 * @brief Namespace for Highway-based lexicographic sorting utilities
 */
namespace hwy_lexsort {

// Helper to get the highway aligned memory resource pointer once
// This avoids repeated function calls in the SortContext constructor
// Use SRDatalog::memory_resource to match the namespace where it's defined
inline SRDatalog::memory_resource* get_highway_mem_res() {
  static SRDatalog::memory_resource* res = &SRDatalog::highway_aligned_memory_resource();
  return res;
}

// Helper to get memory resource for SortContext (resource or default)
inline SRDatalog::memory_resource* get_sort_mem_res(SRDatalog::memory_resource* resource) {
  return (resource != nullptr) ? resource : get_highway_mem_res();
}

/**
 * @brief Context structure to hold reusable buffers for sorting operations
 *
 * This structure maintains aligned buffers that are reused across multiple
 * sorting operations to avoid repeated allocations. It is defined outside
 * HWY_NAMESPACE so it is visible to both the main code and all SIMD target
 * implementations.
 */
struct SortContext {
  ::SRDatalog::AlignedVector<uint32_t> idx_buf_a;  ///< First index permutation buffer
  ::SRDatalog::AlignedVector<uint32_t>
      idx_buf_b;  ///< Second index permutation buffer (for swapping)
  ::SRDatalog::AlignedVector<uint8_t>
      packed_buffer;  ///< Buffer for packed composite keys (uint64_t or hwy::uint128_t)
  ::SRDatalog::AlignedVector<uint8_t> key_buffer;  ///< Buffer for extracted key values

  std::vector<std::size_t> sorted_order;

  /**
   * @brief Constructor that initializes all vectors with a memory resource
   * @param resource Optional memory resource (defaults to highway aligned memory resource)
   * @details If no resource is provided, uses highway_aligned_memory_resource() for SIMD alignment.
   *          If a resource is provided, uses that resource for all buffers.
   */
  explicit SortContext(SRDatalog::memory_resource* resource = nullptr)
      : idx_buf_a(get_sort_mem_res(resource)), idx_buf_b(get_sort_mem_res(resource)),
        packed_buffer(get_sort_mem_res(resource)), key_buffer(get_sort_mem_res(resource)) {}

  /**
   * @brief Resizes all buffers to accommodate n elements
   *
   * Ensures all buffers are large enough for sorting n elements. The packed
   * buffer is sized to accommodate the largest composite key format (hwy::uint128_t
   * = 16 bytes).
   *
   * Uses exponential growth (doubling) to reduce frequent reallocations.
   * Pre-caches a reasonable initial size to avoid small initial allocations.
   *
   * @param n Number of elements to sort
   * @param max_col_size Maximum size of a column element in bytes
   */
  void resize(size_t n, size_t max_col_size) {
    // Pre-cache a reasonable initial size to avoid many small allocations
    constexpr size_t initial_size = static_cast<size_t>(64) * 1024;  // 64K elements

    // Helper to resize with exponential growth (double when exceeded)
    auto resize_with_growth = [&](auto& vec, size_t required_size, size_t initial) {
      if (vec.size() < required_size) {
        // If vector is empty or very small, use initial or required_size (whichever is larger)
        size_t new_size = (vec.size() == 0)
                              ? std::max(initial, required_size)
                              : std::max(required_size, vec.size() * 2);  // Double the size
        vec.resize(new_size);
      }
    };

    resize_with_growth(idx_buf_a, n, initial_size);
    resize_with_growth(idx_buf_b, n, initial_size);

    // Packed buffer needs to accommodate uint64_t (8 bytes) or hwy::uint128_t (16 bytes)
    // Use the larger size (hwy::uint128_t = 16 bytes) to be safe for all cases
    constexpr size_t max_packed_size = 16;  // sizeof(hwy::uint128_t)
    constexpr size_t initial_packed_size = static_cast<size_t>(initial_size) * max_packed_size;
    resize_with_growth(packed_buffer, n * max_packed_size, initial_packed_size);

    // Key buffer size depends on max_col_size, so use a reasonable initial estimate
    constexpr size_t estimated_col_size = 8;  // Assume average 8 bytes per column element
    constexpr size_t initial_key_size = static_cast<size_t>(initial_size) * estimated_col_size;
    resize_with_growth(key_buffer, n * max_col_size, initial_key_size);
  }

  /**
   * @brief Get the sorted indices after a sort operation
   *
   * After calling stable_lex_sort with this context, the sorted indices are
   * stored in idx_buf_a. This method provides access to them.
   *
   * @tparam I Index type (default: uint32_t)
   * @param n Number of elements (must match the size used in the sort)
   * @return Pointer to the sorted indices array
   */
  template <typename I = uint32_t>
  const I* get_sorted_indices([[maybe_unused]] size_t n) const {
    return reinterpret_cast<const I*>(idx_buf_a.data());
  }

  /**
   * @brief Get the sorted indices after a sort operation (non-const version)
   *
   * After calling stable_lex_sort with this context, the sorted indices are
   * stored in idx_buf_a. This method provides access to them.
   *
   * @tparam I Index type (default: uint32_t)
   * @param n Number of elements (must match the size used in the sort)
   * @return Pointer to the sorted indices array
   */
  template <typename I = uint32_t>
  I* get_sorted_indices([[maybe_unused]] size_t n) {
    return reinterpret_cast<I*>(idx_buf_a.data());
  }

  template <std::size_t... Os>
  void update_sorted_order(std::index_sequence<Os...>) {
    sorted_order.resize(idx_buf_a.size());
    ((sorted_order[Os] = idx_buf_a[Os]), ...);
  }
};

}  // namespace hwy_lexsort

HWY_BEFORE_NAMESPACE();

namespace hwy_lexsort {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Converts a value to unsigned representation for radix sorting
 *
 * This function transforms signed integers and floating-point numbers into
 * unsigned integers in a way that preserves sort order. For signed integers,
 * it flips the sign bit. For floating-point numbers, it handles the sign bit
 * and ensures positive values sort before negative values.
 *
 * @tparam T The input type (signed integer, unsigned integer, or floating point)
 * @param v The value to convert
 * @return Unsigned representation that preserves sort order
 */
template <typename T>
HWY_INLINE auto to_unsigned(T v) {
  if constexpr (std::is_floating_point_v<T>) {
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    U b;
    std::memcpy(&b, &v, sizeof(T));
    U m = -static_cast<U>(b >> (sizeof(T) * 8 - 1)) | (U{1} << (sizeof(T) * 8 - 1));
    return b ^ m;
  } else if constexpr (std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    return static_cast<U>(v) ^ (U{1} << (sizeof(T) * 8 - 1));
  } else {
    return v;
  }
}

/**
 * @brief Selects the appropriate packed integer type for stable sorting
 *
 * To ensure stable sorting, we pack the key and index into a single integer
 * where the key is the MSB (most significant bits) and the index is the LSB
 * (least significant bits). This way, when keys are equal, the index acts as
 * a stable tie-breaker.
 *
 * - If both K and I are <= 32 bits: uint64_t (Key in upper 32 bits, Index in lower 32 bits)
 * - Else if both K and I are <= 64 bits: hwy::uint128_t (Key in upper 64 bits, Index in lower 64
 * bits)
 * - Otherwise: void (unsupported)
 *
 * @tparam K The key type
 * @tparam I The index/value type
 */
template <typename K, typename I>
struct PackedType {
  /**
   * @brief The selected packed type (uint64_t, hwy::uint128_t, or void if unsupported)
   *
   * - If both K and I are <= 32 bits: uint64_t
   * - Else if both K and I are <= 64 bits: hwy::uint128_t
   * - Otherwise: void (unsupported, will trigger static_assert)
   */
  using type =
      std::conditional_t<(sizeof(K) <= 4 && sizeof(I) <= 4), uint64_t,
                         std::conditional_t<(sizeof(K) <= 8 && sizeof(I) <= 8), hwy::uint128_t,
                                            // For larger types, we'd need a different approach
                                            // This should not happen in practice for stable sort
                                            void>>;
};

/**
 * @brief Vectorized gather operation using SIMD instructions
 *
 * Gathers elements from a source array using indices, storing them in a
 * destination array. Uses Highway's SIMD gather instructions when possible
 * (for 32-bit or 64-bit types), falling back to scalar code otherwise.
 *
 * @tparam T The element type to gather
 * @tparam I The index type
 * @param src Source array to gather from
 * @param idx Array of indices into src
 * @param dst Destination array to store gathered elements
 * @param n Number of elements to gather
 */
template <typename T, typename I>
HWY_INLINE void gather(const T* HWY_RESTRICT src, const I* HWY_RESTRICT idx,
                       std::remove_const_t<T>* HWY_RESTRICT dst, size_t n) {
  using V = std::remove_const_t<T>;
  constexpr bool is_32 = (sizeof(V) == 4 && sizeof(I) == 4);
  constexpr bool is_64 = (sizeof(V) == 8 && sizeof(I) == 8);
  if constexpr (is_32 || is_64) {
    using HwyT = std::conditional_t<is_32, int32_t, int64_t>;
    const hn::ScalableTag<HwyT> d;
    const hn::ScalableTag<HwyT> di;
    size_t lanes = hn::Lanes(d);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      auto idx_vec = hn::Load(di, reinterpret_cast<const HwyT*>(idx + i));

      auto val_vec = hn::GatherIndex(d, reinterpret_cast<const HwyT*>(src), idx_vec);

      hn::Store(val_vec, d, reinterpret_cast<HwyT*>(dst + i));
    }
    for (; i < n; ++i)
      dst[i] = src[idx[i]];
  } else {
    for (size_t i = 0; i < n; ++i)
      dst[i] = src[idx[i]];
  }
}

/**
 * @brief Least Significant Digit (LSD) radix sort engine
 *
 * Implements a single step of LSD radix sort for lexicographic sorting.
 * This is used internally by stable_lex_sort to sort by each column in
 * reverse order (least significant column first).
 *
 * @tparam I The index type (typically uint32_t)
 */
template <typename I>
struct LsdEngine {
  /**
   * @brief Performs one sorting step on a column
   *
   * This function:
   * 1. Gathers column values using the current permutation
   * 2. Packs them with indices into K32V32 or K64V64 format
   * 3. Sorts using VQSort
   * 4. Updates the permutation based on the sorted order
   *
   * @tparam K The key/column element type
   * @param col_data Pointer to the column data
   * @param current_perm Current permutation indices
   * @param next_perm Output permutation indices (will be updated)
   * @param n Number of elements
   * @param ctx Sort context with reusable buffers
   */
  template <typename K>
  static void sort_step(const K* col_data, const I* current_perm, I* next_perm, size_t n,
                        SortContext& ctx) {
    using Packed = typename PackedType<K, I>::type;

    // Ensure we have a valid packed type (uint64_t or hwy::uint128_t)
    static_assert(!std::is_same_v<Packed, void>, "Key and index types must fit in uint64_t (both "
                                                 "<= 32 bits) or hwy::uint128_t (both <= 64 bits)");

    // Ensure buffers are large enough
    size_t keys_bytes_needed = n * sizeof(K);
    size_t packed_bytes_needed = n * sizeof(Packed);
    if (ctx.key_buffer.size() < keys_bytes_needed) {
      ctx.key_buffer.resize(keys_bytes_needed);
    }
    if (ctx.packed_buffer.size() < packed_bytes_needed) {
      ctx.packed_buffer.resize(packed_bytes_needed);
    }

    auto* keys = reinterpret_cast<K*>(ctx.key_buffer.data());
    auto* packed = reinterpret_cast<Packed*>(ctx.packed_buffer.data());

    gather(col_data, current_perm, keys, n);

    // Pack (Key, Index) into a single integer where Key is MSB and Index is LSB
    // This ensures stability: when keys are equal, indices act as tie-breakers
    for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same_v<Packed, uint64_t>) {
        // Case: 32-bit Key + 32-bit Index
        // Pack as: Key (upper 32 bits) | Index (lower 32 bits)
        uint64_t key_part = static_cast<uint64_t>(to_unsigned(keys[i]));
        uint64_t idx_part = static_cast<uint64_t>(i);
        packed[i] = (key_part << 32) | idx_part;
      } else if constexpr (std::is_same_v<Packed, hwy::uint128_t>) {
        // Case: 64-bit Key + 64-bit Index
        // VQSort sorts uint128_t lexicographically. On Little Endian systems,
        // we need Key in the high 64 bits (MSB) and Index in the low 64 bits (LSB).
        // Use memcpy for portability - layout: [Index (8 bytes)][Key (8 bytes)]
        // because on Little Endian, the last byte is the MSB
        uint64_t key_part = static_cast<uint64_t>(to_unsigned(keys[i]));
        uint64_t idx_part = static_cast<uint64_t>(i);
        std::memcpy(reinterpret_cast<uint8_t*>(&packed[i]), &idx_part, 8);      // Low 64 (LSB)
        std::memcpy(reinterpret_cast<uint8_t*>(&packed[i]) + 8, &key_part, 8);  // High 64 (MSB)
      }
    }

    // VQSort now treats the whole Key+Index combination as one number
    // Since Index is in the LSB, it acts as the stable tie-breaker
    hwy::VQSort(packed, n, hwy::SortAscending());

    // Extract the index (LSB) from the sorted numbers
    for (size_t i = 0; i < n; ++i) {
      I old_rank;
      if constexpr (std::is_same_v<Packed, uint64_t>) {
        // Extract lower 32 bits (the index)
        old_rank = static_cast<I>(packed[i] & 0xFFFFFFFFULL);
      } else {
        // Extract low 64 bits (the index) using memcpy for portability
        uint64_t low_part = 0;
        std::memcpy(&low_part, reinterpret_cast<const uint8_t*>(&packed[i]), 8);
        old_rank = static_cast<I>(low_part);
      }
      next_perm[i] = current_perm[old_rank];
    }
  }
};

// --- Unstable Sort Implementation (simple wrapper around VQSort) ---

/**
 * @brief In-place unstable sort implementation using VQSort
 *
 * VQSort natively supports:
 * - Basic types: uint16/32/64, int16/32/64, float, double, float16_t
 * - Key-Value pairs: K32V32 (32-bit key + 32-bit value), K64V64 (64-bit key + 64-bit value)
 *   These allow sorting by key while keeping value attached (VQSort handles the packing internally)
 * - 128-bit types: hwy::uint128_t (must use this specific type, not arbitrary 128-bit types)
 *
 * @tparam T The element type (must be a type supported by VQSort)
 * @param data Pointer to the array to sort (modified in-place)
 * @param n Number of elements to sort
 */
template <typename T>
void unstable_sort_impl(T* data, size_t n) {
  if (n == 0)
    return;

  // Simple wrapper: sort in-place using VQSort
  // VQSort will handle type checking at compile time - it only accepts supported types
  hwy::VQSort(data, n, hwy::SortAscending());
}

// --- Target Specific Implementation ---

/**
 * @brief Stable lexicographic sort implementation (target-specific) with context
 *
 * Performs a stable lexicographic sort on multiple columns using LSD radix sort.
 * The sort order is specified by the `order` parameter, which is a span of column
 * indices indicating the order in which columns should be considered
 * (e.g., {1, 0} means sort by column 1 first, then column 0).
 *
 * The algorithm sorts from least significant column to most significant column,
 * ensuring stability (equal elements maintain their relative order).
 *
 * The sorted indices are stored in ctx.idx_buf_a after the sort completes.
 * Use ctx.get_sorted_indices<I>(n) to access them.
 *
 * @tparam I Index type (default: uint32_t)
 * @tparam Ts... Column data types
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority (e.g., {1, 0} means sort by col 1,
 * then col 0)
 * @param ctx Reference to SortContext for buffer reuse and result storage
 * @param reorder_data If true, reorder the column data in place using sorted indices
 * @param cols... Pointers to column arrays (one per column type in Ts...)
 *
 * @example
 * // Reuse context for multiple sorts
 * SortContext ctx;
 * std::vector<int> order = {1, 0};
 * stable_lex_sort_impl(100, order, ctx, false, col0, col1);  // Only sort indices
 * const uint32_t* sorted = ctx.get_sorted_indices<uint32_t>(100);
 *
 * // Or reorder data in place:
 * stable_lex_sort_impl(100, order, ctx, true, col0, col1);  // Sort and reorder data
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort_impl(size_t n, const int* order, size_t order_size, SortContext& ctx,
                          bool reorder_data, Ts*... cols) {
  if (n == 0)
    return;

  // // Validate order indices
  // constexpr size_t num_cols = sizeof...(Ts);
  // for (size_t j = 0; j < order_size; ++j) {
  //   int col_idx = order[j];
  //   if (col_idx < 0 || static_cast<size_t>(col_idx) >= num_cols) {
  //     throw std::invalid_argument("Column order index out of range");
  //   }
  // }

  size_t max_col_size = std::max({sizeof(Ts)...});
  ctx.resize(n, max_col_size);

  I* p_in = reinterpret_cast<I*>(ctx.idx_buf_a.data());
  I* p_out = reinterpret_cast<I*>(ctx.idx_buf_b.data());

  for (size_t i = 0; i < n; ++i)
    p_in[i] = static_cast<I>(i);

  auto col_tuple = std::make_tuple(cols...);

  // Process columns in reverse order (LSD radix sort)
  for (int i = static_cast<int>(order_size) - 1; i >= 0; --i) {
    int col_idx = order[i];
    size_t current = 0;

    std::apply(
        [&](auto... args) {
          ((current++ == static_cast<size_t>(col_idx)
                ? LsdEngine<I>::sort_step(args, p_in, p_out, n, ctx)
                : void()),
           ...);
        },
        col_tuple);

    std::swap(p_in, p_out);
  }
  // Result is now in p_in, which points to either idx_buf_a or idx_buf_b
  // If p_in != idx_buf_a.data(), we need to copy it back
  if (p_in != reinterpret_cast<I*>(ctx.idx_buf_a.data())) {
    std::memcpy(ctx.idx_buf_a.data(), p_in, n * sizeof(I));
  }

  // If reorder_data is true, reorder all columns in place using sorted indices
  if (reorder_data) {
    const I* sorted_indices = ctx.get_sorted_indices<I>(n);

    // Pre-allocate buffer to maximum column size to avoid repeated resizing
    // and ensure we have enough space for all columns
    size_t max_col_size = std::max({sizeof(Ts)...});
    size_t max_temp_size = n * max_col_size;
    if (ctx.key_buffer.size() < max_temp_size) {
      ctx.key_buffer.resize(max_temp_size);
    }

    // Reorder each column using gather into a temporary buffer, then copy back
    // Note: When reorder_data=true, columns must be non-const (enforced by removing const from
    // types)
    std::apply(
        [&](auto*... col_ptrs) {
          // For each column, gather into temp buffer then copy back
          (
              [&]<typename ColType>(ColType* col_ptr) {
                // Remove const from ColType if present (reorder_data=true requires non-const)
                using NonConstColType = std::remove_const_t<ColType>;
                NonConstColType* non_const_ptr = const_cast<NonConstColType*>(col_ptr);

                // Use pre-allocated temp buffer (already sized for largest column)
                size_t temp_size = n * sizeof(NonConstColType);
                NonConstColType* temp = reinterpret_cast<NonConstColType*>(ctx.key_buffer.data());
                // Gather reordered data into temp buffer
                gather<NonConstColType, I>(col_ptr, sorted_indices, temp, n);
                // Copy back to original column
                std::memcpy(non_const_ptr, temp, temp_size);
              }(col_ptrs),
              ...);
        },
        col_tuple);
  }
}

/**
 * @brief Stable lexicographic sort implementation (target-specific) with output array
 *
 * Overload that writes results to an external array. Creates a temporary context internally.
 * This maintains backward compatibility with the original API.
 *
 * @tparam I Index type (default: uint32_t)
 * @tparam Ts... Column data types
 * @param ids_out Output array of sorted indices (must have space for n elements)
 * @param n Number of rows to sort
 * @param order Span of column indices specifying sort priority
 * @param reorder_data If true, reorder the column data in place using sorted indices (default:
 * false)
 * @param cols... Pointers to column arrays
 *
 * @example
 * // Sort by column 1 first, then column 0
 * uint32_t indices[100];
 * int32_t col0[100];
 * float col1[100];
 * std::vector<int> order = {1, 0};
 * stable_lex_sort_impl(indices, 100, order, false, col0, col1);  // Only sort indices
 * // Or reorder data in place:
 * stable_lex_sort_impl(indices, 100, order, true, col0, col1);  // Sort and reorder data
 */
template <typename I = uint32_t, typename... Ts>
void stable_lex_sort_impl(I* ids_out, size_t n, const int* order, size_t order_size,
                          bool reorder_data, Ts*... cols) {
  SortContext ctx;
  stable_lex_sort_impl<I>(n, order, order_size, ctx, reorder_data, cols...);
  std::memcpy(ids_out, ctx.get_sorted_indices<I>(n), n * sizeof(I));
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_lexsort
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy_lexsort {

/**
 * @brief Stable lexicographic sort (public API) with context
 *
 * Performs a stable lexicographic sort on multiple columns. This function
 * automatically dispatches to the best available SIMD implementation (AVX2,
 * AVX-512, NEON, etc.) at compile time.
 *
 * The sort is stable, meaning that elements with equal keys maintain their
 * relative order from the input. The sorted indices are stored in the context
 * and can be accessed via ctx.get_sorted_indices<I>(n).
 *
 * @tparam I Index type for output (default: uint32_t)
 * @tparam Ts... Types of the column arrays
 * @param n Number of rows to sort
 * @param order Pointer to array of column indices specifying sort priority (int type)
 *              (e.g., {1, 0} means sort by column 1 first, then column 0)
 * @param order_size Number of elements in the order array
 * @param ctx Reference to SortContext for buffer reuse and result storage
 * @param reorder_data If true, reorder the column data in place using sorted indices (default:
 * false)
 * @param cols... Pointers to column data arrays (one per column)
 *
 * @note The number of column pointers must match the number of types in Ts...
 * @note Column order indices must be valid (0 <= index < number of columns)
 * @note Reusing a SortContext across multiple sorts can improve performance by avoiding
 *       repeated buffer allocations.
 * @note The sorted indices are stored in ctx.idx_buf_a after the sort completes.
 * @note When reorder_data is true, the column data is reordered in place, so accessing
 *       data via sorted indices is no longer necessary (data is already sorted).
 *
 * @example
 * // Reuse context for multiple sorts (better performance):
 * hwy_lexsort::SortContext ctx;
 * std::vector<int> order1 = {1, 0};
 * stable_lex_sort(100, order1.data(), order1.size(), ctx, false, column0, column1);  // Only sort
 * indices const uint32_t* sorted1 = ctx.get_sorted_indices<uint32_t>(100);
 *
 * // Or reorder data in place:
 * stable_lex_sort(100, order1.data(), order1.size(), ctx, true, column0, column1);  // Sort and
 * reorder data
 * // Now column0 and column1 are already sorted, no need to use sorted indices
 */
template <typename I = uint32_t, typename... Ts>
HWY_DLLEXPORT void stable_lex_sort(size_t n, const int* order, size_t order_size, SortContext& ctx,
                                   bool reorder_data, Ts*... cols) {
  // This correctly dispatches to AVX2/AVX512/NEON implementation
  HWY_STATIC_DISPATCH(stable_lex_sort_impl)<I>(n, order, order_size, ctx, reorder_data, cols...);
}

/**
 * @brief Stable lexicographic sort (public API) with context (backward compatibility overload)
 *
 * Overload without reorder_data parameter (defaults to false - only sort indices).
 */
template <typename I = uint32_t, typename... Ts>
HWY_DLLEXPORT void stable_lex_sort(size_t n, const int* order, size_t order_size, SortContext& ctx,
                                   Ts*... cols) {
  stable_lex_sort<I>(n, order, order_size, ctx, false, cols...);
}

/**
 * @brief Stable lexicographic sort (public API) without context
 *
 * Performs a stable lexicographic sort on multiple columns. This function
 * automatically dispatches to the best available SIMD implementation (AVX2,
 * AVX-512, NEON, etc.) at compile time.
 *
 * The sort is stable, meaning that elements with equal keys maintain their
 * relative order from the input. This overload creates a temporary context
 * internally for backward compatibility.
 *
 * @tparam I Index type for output (default: uint32_t)
 * @tparam Ts... Types of the column arrays
 * @param ids_out Output array of sorted row indices (must have space for n elements)
 * @param n Number of rows to sort
 * @param order Pointer to array of column indices specifying sort priority (int type)
 *              (e.g., {1, 0} means sort by column 1 first, then column 0)
 * @param order_size Number of elements in the order array
 * @param reorder_data If true, reorder the column data in place using sorted indices (default:
 * false)
 * @param cols... Pointers to column data arrays (one per column)
 *
 * @note The number of column pointers must match the number of types in Ts...
 * @note Column order indices must be valid (0 <= index < number of columns)
 * @note When reorder_data is true, the column data is reordered in place, so accessing
 *       data via sorted indices is no longer necessary (data is already sorted).
 *
 * @example
 * // Sort 100 rows by column 1 (ascending), then column 0 (ascending)
 * uint32_t sorted_indices[100];
 * int32_t column0[100];
 * float column1[100];
 * std::vector<int> order = {1, 0};
 * stable_lex_sort(sorted_indices, 100, order.data(), order.size(), false, column0, column1);  //
 * Only sort indices
 *
 * // Access sorted data:
 * for (size_t i = 0; i < 100; ++i) {
 *   size_t idx = sorted_indices[i];
 *   // column0[idx] and column1[idx] are the i-th row in sorted order
 * }
 *
 * // Or reorder data in place:
 * stable_lex_sort(sorted_indices, 100, order.data(), order.size(), true, column0, column1);  //
 * Sort and reorder data
 * // Now column0 and column1 are already sorted, no need to use sorted indices
 */
template <typename I = uint32_t, typename... Ts>
HWY_DLLEXPORT void stable_lex_sort(I* ids_out, size_t n, const int* order, size_t order_size,
                                   bool reorder_data, Ts*... cols) {
  // This correctly dispatches to AVX2/AVX512/NEON implementation
  HWY_STATIC_DISPATCH(stable_lex_sort_impl)<I>(ids_out, n, order, order_size, reorder_data,
                                               cols...);
}

/**
 * @brief Stable lexicographic sort (public API) without context (backward compatibility overload)
 *
 * Overload without reorder_data parameter (defaults to false - only sort indices).
 */
template <typename I = uint32_t, typename... Ts>
HWY_DLLEXPORT void stable_lex_sort(I* ids_out, size_t n, const int* order, size_t order_size,
                                   Ts*... cols) {
  stable_lex_sort<I>(ids_out, n, order, order_size, false, cols...);
}

/**
 * @brief In-place unstable sort (public API)
 *
 * Performs an in-place unstable sort on an array using Highway's VQSort.
 * This function automatically dispatches to the best available SIMD
 * implementation at compile time.
 *
 * Supported types:
 * - Integers: uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t
 * - Floating point: float, double, float16_t
 * - Key-value pairs: hwy::K32V32, hwy::K64V64
 * - 128-bit: hwy::uint128_t
 *
 * @tparam T Element type (must be supported by VQSort)
 * @param data Pointer to array to sort (modified in-place)
 * @param n Number of elements to sort
 *
 * @note This is an unstable sort - equal elements may be reordered
 * @note Only supports types up to 128 bits
 *
 * @example
 * // Sort an array of integers
 * int32_t data[1000];
 * // ... fill data ...
 * unstable_sort(data, 1000);
 * // data is now sorted in ascending order
 */
template <typename T>
HWY_DLLEXPORT void unstable_sort(T* data, size_t n) {
  // This correctly dispatches to AVX2/AVX512/NEON implementation
  HWY_STATIC_DISPATCH(unstable_sort_impl)(data, n);
}

}  // namespace hwy_lexsort
#endif