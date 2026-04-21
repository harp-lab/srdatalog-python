#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "column.h"
#include "index.h"
#include "semiring.h"
#include "sort.h"
#include "tmp.h"

namespace SRDatalog {

/**
 * @brief Sorted Array Index (Read-Only B-Tree / Flat Index).
 *
 * @details This index stores the indexed columns in a contiguous, sorted layout.
 * Unlike HashTrieIndex, it does not use a pointer-based tree structure.
 * Instead, "nodes" are represented by ranges [start, end) over the sorted arrays.
 *
 * @section tradeoffs Trade-offs vs HashTrieIndex
 * - **Pros:**
 * - Extreme memory locality (linear scan friendly).
 * - Lower memory overhead (no node objects, hash maps, or pointers).
 * - Supports Range Queries (e.g., x > 5) natively via binary search.
 * - **Cons:**
 * - Point lookups are O(log N) instead of O(1).
 * - `values()` (enumerating children) returns a range containing DUPLICATES.
 * Algorithms must handle this (e.g., using skip-scanning or upper_bound).
 *
 * @tparam SR The semiring type
 * @tparam AttrTuple The attribute tuple type
 * @tparam ValueType The type for encoded values (default: uint32_t for memory efficiency)
 * @tparam RowIdType The type for row IDs (default: uint32_t, must be able to represent all row
 * indices)
 */
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType = uint32_t,
          typename RowIdType = uint32_t>
class SortedArrayIndex {
  struct Impl;

 public:
  using RowId = RowIdType;
  using RowRange = std::span<const RowId>;
  using ValueRange = std::span<const ValueType>;
  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  /**
   * @brief Virtual Cursor for the Sorted Array.
   * @details Does not point to a physical node object. Instead, represents
   * a range `[begin, end)` within the materialized sorted columns at a specific `depth`.
   */
  class NodeHandle {
   public:
    NodeHandle() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool is_leaf() const noexcept;
    [[nodiscard]] uint32_t depth() const noexcept;

    /**
     * @brief Returns a view of unique values for this range.
     * @details For SortedArrayIndex, this deduplicates consecutive duplicate values.
     * Since the array is sorted, duplicates are contiguous, so this efficiently
     * skips to the next unique value. This ensures joins iterate over unique values
     * only, matching the behavior of HashTrieIndex.
     * @return A range view containing only unique values
     */
    [[nodiscard]] auto values() const;

    [[nodiscard]] RowRange rows() const;

    /**
     * @brief Binary Search for the next key.
     * @details Uses std::equal_range (O(log N)) to find the sub-range.
     */
    NodeHandle prefix(ValueType next_key) const;

    template <std::size_t N>
    NodeHandle prefix(const EncodedKeyPrefix<N>& suffix) const;

    [[nodiscard]] bool contains_value(ValueType key) const noexcept;

    // Degree is the size of the range (includes duplicates)
    [[nodiscard]] std::size_t degree() const noexcept;

    // Count of unique values (excludes consecutive duplicates)
    // This is the size of the view returned by values()
    [[nodiscard]] std::size_t unique_count() const noexcept;

   private:
    template <Semiring SR2, ColumnElementTuple AttrTuple2, typename ValueType2, typename RowIdType2>
    friend class SortedArrayIndex;

    const Impl* impl_{nullptr};
    std::size_t begin_{0};  // Start index in the arrays
    std::size_t end_{0};    // End index (exclusive)
    uint32_t depth_{0};     // Current column index in spec

    NodeHandle(const Impl* impl, std::size_t begin, std::size_t end, uint32_t depth) noexcept
        : impl_(impl), begin_(begin), end_(end), depth_(depth) {}
  };

  static_assert(CNodeHandle<NodeHandle>, "SortedArrayIndex::NodeHandle must satisfy CNodeHandle");

  explicit SortedArrayIndex(memory_resource* resource);
  ~SortedArrayIndex();

  SortedArrayIndex(const SortedArrayIndex&) = delete;
  SortedArrayIndex& operator=(const SortedArrayIndex&) = delete;
  SortedArrayIndex(SortedArrayIndex&&) noexcept = default;
  SortedArrayIndex& operator=(SortedArrayIndex&&) noexcept = default;

  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const ValueType>, arity>& encoded_rows,
                          memory_resource* resource = nullptr);

  [[nodiscard]] NodeHandle root() const;

  template <std::size_t N>
  NodeHandle prefix_lookup(const EncodedKeyPrefix<N>& k) const;

  [[nodiscard]] std::size_t bytes_used() const;
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }
  /// @brief Get the number of rows processed (before deduplication).
  /// @return Total number of relation rows that have been processed into this index
  /// @details This tracks how many rows from the relation have been indexed,
  ///          before deduplication. Used to determine if index needs rebuilding.
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return rows_processed_;
  }
  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }
  void clear() noexcept;

  // Deep clone
  void clone_from(const SortedArrayIndex& other, memory_resource* resource = nullptr);

  /// @brief Verify that the index is correctly sorted
  /// @details Checks that all rows are in lexicographic sorted order by the indexed columns.
  ///          This is a critical invariant that must be maintained for correct operation.
  /// @return true if sorted, false otherwise
  /// @throws std::runtime_error if not sorted (in debug builds) or returns false (in release)
  [[nodiscard]] bool verify_sorted() const;

  /// @brief Merge another SortedArrayIndex into this index
  /// @details Merges the index structure from another index into this index,
  ///          applying a row ID offset to the merged postings.
  /// @param other The source SortedArrayIndex to merge from
  /// @param row_id_offset Offset to add to row IDs from the source index
  void merge(const SortedArrayIndex& other, std::size_t row_id_offset);

  /**
   * @brief LFTJ Intersection (galloping intersection, "seek").
   * @details Returns a View that scans the first handle and probes the rest.
   */
  template <typename FirstHandle, typename... OtherHandles>
  static auto intersect(const FirstHandle& first, const OtherHandles&... others);

 private:
  std::unique_ptr<Impl> p_;        // Unique ptr is fine here
  std::size_t size_{0};            // Number of unique keys (after deduplication)
  std::size_t rows_processed_{0};  // Number of rows processed (before deduplication)
};

template <Semiring SR, ColumnElementTuple AttrTuple>
inline constexpr bool is_valid_sorted_index_v = IndexLike<SortedArrayIndex<SR, AttrTuple>>;

}  // namespace SRDatalog

#include "sorted_array_index.ipp"