/**
 * @file device_sorted_array_index.h
 * @brief GPU-based Sorted Array Index (Read-Only B-Tree / Flat Index) for GPU kernels
 *
 * @details This index stores the indexed columns in a contiguous, sorted layout on GPU device
 * memory. Unlike the CPU version, this uses device pointers and POD-style Handle structures that
 * can be safely copied to and used within GPU kernels.
 *
 * @section gpu_design GPU Design Considerations
 * - **POD Handle**: NodeHandle is a Plain Old Data structure containing only device pointers and
 *   indices, making it safe to copy to GPU and use in kernels
 * - **Device Memory**: All data is stored in DeviceArray containers using RMM (RAPIDS Memory
 * Manager) allocators
 * - **Kernel-Compatible**: Handle can be passed to GPU kernels via kernel launch parameters
 * - **Coalesced Access**: Data layout optimized for GPU memory coalescing patterns
 *
 * @section tradeoffs Trade-offs vs CPU SortedArrayIndex
 * - **Pros:**
 * - GPU-accelerated search and traversal operations
 * - Parallel processing of large datasets
 * - Memory coalescing optimizations for GPU
 * - **Cons:**
 * - Requires GPU memory transfers for host-device data movement
 * - Handle must be POD (no host pointers or complex types)
 * - Kernel launch overhead for small operations
 *
 * @tparam SR The semiring type
 * @tparam AttrTuple The attribute tuple type
 * @tparam ValueType The type for encoded values (default: uint32_t for memory efficiency)
 * @tparam RowIdType The type for row IDs (default: uint32_t)
 */

#pragma once

#include "../index.h"
#include "../semiring.h"
#include "./macro.h"
#include "./sort.h"
#include "device_array.h"
#include "nd_device_array.h"
#include <array>
#include <cstddef>
#include <rmm/device_uvector.hpp>
#include <span>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <tuple>
#include <utility>

namespace SRDatalog::GPU {

// Forward declaration - use the one from parent namespace
using IndexSpec = SRDatalog::IndexSpec;

template <typename ValueType, typename RowIdType, std::size_t N, typename Group, typename Accessor>
class IntersectView;

// Forward declaration of NodeHandle for use in NodeView typedef
template <Semiring SR, typename ValueTypeParam = uint32_t, typename RowIdType = uint32_t>
class NodeHandle;

/**
 * @brief Immutable view of the index structure (pointers only).
 *
 * @details Contains all the invariant pointers needed to traverse the index.
 * Passed to kernel as a single structure, shared by all threads.
 */
template <Semiring SR, typename ValueTypeParam = uint32_t>
struct NodeView {
  // Type alias for associated handle type (for JIT codegen)
  using NodeHandle = SRDatalog::GPU::NodeHandle<SR, ValueTypeParam>;
  using ValueType = ValueTypeParam;

  // Device pointers (order matches aggregate initialization in view())
  const ValueTypeParam* col_data_{nullptr};  // Start of flat column-major array
  semiring_value_t<SR>* provenance_ptr_{
      nullptr};  // Provenance/semiring values, not const provenance is mutable

  // Metadata
  uint32_t arity_{0};
  uint32_t stride_{0};    // Distance between columns (usually num_rows)
  uint32_t num_rows_{0};  // Actual number of valid rows

  // Access accessor for unified Scan interface
  [[nodiscard]] GPU_HD ValueTypeParam get_value(uint32_t depth,
                                                uint32_t sorted_idx) const noexcept {
    return col_data_[depth * stride_ + sorted_idx];
  }

  // Access provenance for unified Scan interface
  [[nodiscard]] GPU_HD semiring_value_t<SR> get_provenance(uint32_t sorted_idx) const noexcept {
    if constexpr (has_provenance_v<SR>) {
      return provenance_ptr_[sorted_idx];
    } else {
      return {};  // Return monostate for NoProvenance
    }
  }

  // Helper accessors for backward compatibility
  [[nodiscard]] GPU_HD const ValueTypeParam* col_data() const noexcept {
    return col_data_;
  }
};

/**
 * @brief POD-style Cursor for GPU kernel usage (lightweight state)
 *
 * @details NodeHandle (Cursor) contains only the mutable state needed for traversal:
 * - Indices (begin, end)
 * - Current depth
 * - NO pointers (accessed via NodeView)
 *
 * Size: 3 * 4 bytes = 12 bytes (vs 56 bytes previously)
 */
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
class NodeHandle {
 public:
  using View = NodeView<SR, ValueTypeParam>;

  /// @brief Default constructor creates an invalid handle
  NodeHandle() = default;

  /// @brief Constructor for creating a handle with state
  GPU_HD NodeHandle(RowIdType begin, RowIdType end, uint32_t depth)
      : begin_(begin), end_(end), depth_(depth) {}

  /// @brief Check if handle is valid
  [[nodiscard]] GPU_HD bool valid() const noexcept {
    return begin_ < end_;
  }

  /// @brief Check if this is a leaf node (depth equals index arity)
  [[nodiscard]] GPU_HD bool is_leaf(const View& view) const noexcept {
    // printf("DEBUG: is_leaf. handle=%p, view=%p. depth=%u, arity=%u\n", this, &view, depth_,
    // view.arity_);
    return valid() && depth_ == view.arity_;
  }

  /// @brief Get current depth in the index
  [[nodiscard]] GPU_HD uint32_t depth() const noexcept {
    return depth_;
  }

  /// @brief Get the degree (size of the range)
  [[nodiscard]] GPU_HD std::size_t degree() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - begin_) : 0;
  }

  // NOTE: values() and all_prov() removed in favor of indexed accessors
  // Use get_value_at() and get_provenance_at() instead

  /// @brief Get span of provenance values (only valid at leaf)
  [[nodiscard]] GPU_HD cuda_std::span<const semiring_value_t<SR>> rows(
      const View& view) const noexcept {
    if (!is_leaf(view)) {
      return cuda_std::span<const semiring_value_t<SR>>{};
    }
    return cuda_std::span<const semiring_value_t<SR>>(view.provenance_ptr_ + begin_,
                                                      static_cast<std::size_t>(end_ - begin_));
  }

  /// @brief Number of unique values (same as degree for sorted data)
  [[nodiscard]] GPU_HD std::size_t unique_count() const noexcept {
    return degree();
  }

  /// @brief Binary search for a key with configurable group size
  template <typename Group>
  [[nodiscard]] __device__ NodeHandle prefix(ValueTypeParam key, Group tile,
                                             const View& view) const;

  [[nodiscard]] __device__ NodeHandle prefix_seq(ValueTypeParam key, const View& view) const;

  /// @brief One-side search: only lower_bound, returns position for use with child_range()
  /// @details This enables the pattern: prefix_lower_only() + child_range() = 1 search instead of 2
  ///          Returns tuple(position, found) where position is the lower_bound offset and
  ///          found indicates if the key exists at that position.
  /// @param key The key to search for
  /// @param tile Cooperative group for parallel search
  /// @param view Index view
  /// @return tuple(lower_bound position relative to begin_, true if key found)
  template <typename Group>
  [[nodiscard]] __device__ thrust::pair<RowIdType, bool> prefix_lower_only(ValueTypeParam key,
                                                                           Group tile,
                                                                           const View& view) const;

  // @brief Get provenance value at leaf level
  template <int TILE_SIZE = 32>
  [[nodiscard]] __device__ semiring_value_t<SR> prov(const View& view) const noexcept;

  /// @brief Check if key exists (uses group-cooperative search)
  /// @param tile Cooperative group for parallel search
  template <typename Group>
  [[nodiscard]] __device__ bool contains_value(ValueTypeParam key, const View& view,
                                               Group tile) const noexcept;

  /// @brief Binary join optimization: Single lower_bound search for probe relations
  /// @details Returns a handle starting from lower_bound to END of current range.
  ///          Unlike prefix() which computes both lower and upper bounds (2 searches),
  ///          this only does lower_bound (1 search). The caller detects end of matching
  ///          values by comparing values during iteration (value != key means stop).
  ///          This saves ~50% of search work for binary join probes.
  /// @param key Key to search for
  /// @param tile Cooperative group for parallel search
  /// @param view Index view
  /// @return NodeHandle from lower_bound to end of current range (caller checks values)
  template <typename Group>
  [[nodiscard]] __device__ NodeHandle prefix_lower_bound(ValueTypeParam key, Group tile,
                                                         const View& view) const;

  /// @brief Sequential version of prefix_lower_bound for Cartesian loops
  [[nodiscard]] __device__ NodeHandle prefix_lower_bound_seq(ValueTypeParam key,
                                                             const View& view) const;

  /// @brief Get begin index
  [[nodiscard]] GPU_HD RowIdType begin() const noexcept {
    return begin_;
  }

  /// @brief Get end index
  [[nodiscard]] GPU_HD RowIdType end() const noexcept {
    return end_;
  }

  /// @brief Create invalid handle
  [[nodiscard]] GPU_HD static NodeHandle invalid() noexcept {
    return NodeHandle{0, 0, 0};
  }

  /// @brief Row position of current handle on data array
  [[nodiscard]] __device__ RowIdType row_position() const noexcept;
  /// @brief Column position of current handle on data array
  [[nodiscard]] __device__ RowIdType column_position() const noexcept;
  /// @brief get a offset with respect to the data array
  template <int COLUMN>
  [[nodiscard]] __device__ RowIdType offset(RowIdType row, const View& view) const noexcept;

  /// @brief Unified value access (Direct)
  /// @param view The handle's view
  /// @param idx The index within the range [0, degree)
  [[nodiscard]] GPU_HD ValueTypeParam get_value_at(const View& view,
                                                   std::size_t idx) const noexcept {
    // Access column at current depth, offset by begin_ + idx
    return view.col_data_[depth_ * view.stride_ + begin_ + idx];
  }

  /// @brief Unified provenance access (Direct)
  [[nodiscard]] GPU_HD semiring_value_t<SR> get_provenance_at(const View& view,
                                                              std::size_t idx) const noexcept {
    if constexpr (has_provenance_v<SR>) {
      return view.provenance_ptr_[begin_ + idx];
    } else {
      return {};  // Return monostate for NoProvenance
    }
  }

  /// @brief Get values as a span (for WCOJ intersection)
  /// @details Returns a span over the column data for this handle's range.
  ///          This is still valid for DeviceSortedArrayIndex since it uses contiguous storage.
  ///          For LSM indices, use get_value_at() instead.
  [[nodiscard]] __device__ cuda_std::span<const ValueTypeParam> values(
      const View& view) const noexcept {
    const ValueTypeParam* col_ptr = view.col_data_ + (depth_ * view.stride_);
    return cuda_std::span<const ValueTypeParam>(col_ptr + begin_, end_ - begin_);
  }

  /// @brief Get iterators for intersection (Indexed Accessor API)
  [[nodiscard]] __device__ auto iterators(const View& view) const noexcept {
    return values(view);
  }

  /// @brief Get provenance values as a span (for WCOJ intersection)
  [[nodiscard]] __device__ cuda_std::span<const semiring_value_t<SR>> all_prov(
      const View& view) const noexcept {
    if constexpr (has_provenance_v<SR>) {
      return cuda_std::span<const semiring_value_t<SR>>(view.provenance_ptr_ + begin_,
                                                        end_ - begin_);
    } else {
      return {};  // Empty span for NoProvenance
    }
  }

  /// @brief Get provenance pointer (for CDeviceHandle concept)
  /// @details Returns the provenance pointer from the view, or nullptr if invalid
  [[nodiscard]] GPU_HD const semiring_value_t<SR>* provenance_ptr(const View& view) const noexcept {
    if constexpr (has_provenance_v<SR>) {
      if (!valid()) {
        return nullptr;
      }
      return view.provenance_ptr_;
    } else {
      return nullptr;  // No provenance pointer for NoProvenance
    }
  }

  /// @brief Create child handle at next depth (public for direct access in intersection)
  [[nodiscard]] GPU_HD NodeHandle child(RowIdType new_begin, RowIdType new_end) const noexcept {
    return NodeHandle{new_begin, new_end, static_cast<uint32_t>(depth_ + 1)};
  }

  /// @brief Create child handle at a specific position (for intersection)
  /// @details When iterating intersection results, we get a position index within the current
  /// range.
  ///          This creates a single-element child at that specific position.
  [[nodiscard]] GPU_HD NodeHandle child(RowIdType position) const noexcept {
    // Create a single-element child at the given position
    return NodeHandle{begin_ + position, begin_ + position + 1, static_cast<uint32_t>(depth_ + 1)};
  }

  /// @brief Create child handle with proper range bounds (LFTJ optimization)
  /// @details When intersection finds value V at position P, we know the lower bound
  ///          but need to find the upper bound to create a properly bounded child.
  ///          This finds upper_bound(V) in [begin+P, end) and creates [begin+P, upper_bound).
  /// @param position The position within the current range where value was found
  /// @param key The value at that position (used for upper_bound search)
  /// @param tile Cooperative group for parallel upper_bound search
  /// @param view The view for column access
  template <typename Group>
  [[nodiscard]] __device__ NodeHandle child_range(RowIdType position, ValueTypeParam key,
                                                  Group tile, const View& view) const noexcept;

 private:
  // POD indices (use RowIdType for consistency with class template parameter)
  RowIdType begin_{0};
  RowIdType end_{0};
  uint32_t depth_{0};
};

/**
 * @brief GPU Sorted Array Index
 *
 * @details GPU-accelerated version of SortedArrayIndex that stores data on device memory
 * and provides POD-style Handle structures for use in GPU kernels.
 */
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam = uint32_t,
          typename RowIdType = uint32_t>
class DeviceSortedArrayIndex {
  struct Impl;

 public:
  // Expose template parameters as public type aliases
  using ValueType = ValueTypeParam;
  using RowId = RowIdType;
  using RowRange = cuda_std::span<const RowId>;
  using ValueRange = cuda_std::span<const ValueTypeParam>;
  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  /**
   * @brief POD-style Handle for GPU kernel usage (fatty pointer, need slim down in future)
   *
   * @details NodeHandle is a Plain Old Data structure containing only:
   * - Device pointers (safe to copy to GPU)
   * - Indices and offsets (POD types)
   * - No host pointers or complex types
   *
   * This design allows the Handle to be:
   * - Copied to GPU memory via kernel launch parameters
   * - Used directly in GPU kernels
   * - Stored in device memory for persistent queries
   *
   * @warning Handle Lifetime: Handles contain raw pointers to device memory that may be
   *          invalidated when the index is modified. The following operations invalidate
   *          all existing handles:
   *          - merge(): Rebuilds  may reallocate memory
   *          - build_from_encoded_device(): Rebuilds the entire index
   *          - clear(): Deallocates all memory
   *
   * @warning Always call root() AFTER operations that modify the index to get a fresh handle.
   *          Do NOT reuse handles across merge/build/clear operations, as this will result
   *          in dangling pointers and undefined behavior.
   *
   * @note All device pointers must remain valid for the lifetime of the Handle usage.
   *       Since C++ doesn't have borrow checking, it's the caller's responsibility to
   *       ensure handles are not used after invalidation.
   */
  using NodeHandle = SRDatalog::GPU::NodeHandle<SR, ValueType, RowId>;
  using NodeView = SRDatalog::GPU::NodeView<SR, ValueType>;

  // Forward declaration
  template <Semiring SR2, ColumnElementTuple AttrTuple2,
            template <Semiring, ColumnElementTuple> class IndexType2>
  class Relation;

  /// @brief Default constructor
  explicit DeviceSortedArrayIndex() = default;

  /// @brief Destructor
  ~DeviceSortedArrayIndex() = default;

  // Non-copyable (to avoid double deallocation of device memory)
  DeviceSortedArrayIndex(const DeviceSortedArrayIndex&) = delete;
  DeviceSortedArrayIndex& operator=(const DeviceSortedArrayIndex&) = delete;

  // Movable
  DeviceSortedArrayIndex(DeviceSortedArrayIndex&&) noexcept = default;
  DeviceSortedArrayIndex& operator=(DeviceSortedArrayIndex&&) noexcept = default;

  /// @brief Build index from encoded row data
  /// @param spec Index specification (which columns to index)
  /// @param encoded_rows Array of spans containing encoded column data
  /// @note This transfers data to GPU and builds the sorted index structure
  /// @note Uses std::span for host-side parameter (caller passes std::span from host)
  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const ValueTypeParam>, arity>& encoded_rows);

  /// @brief Build index from encoded row data on device
  /// @param spec Index specification (which columns to index)
  /// @param encoded_cols Array of NDDeviceArray containing encoded column data
  /// @param provenance Array of DeviceArray containing provenance data
  /// @note This builds the sorted index structure on device — always copies columns.
  /// @warning This operation invalidates all existing NodeHandles. Call root() after
  ///          build_from_encoded_device() to get a fresh handle.
  void build_from_encoded_device(const IndexSpec& spec,
                                 NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                                 DeviceArray<semiring_value_t<SR>>& provenance);

  /// @brief Build index from encoded device data (NoProvenance overload)
  /// @details For NoProvenance semiring, provenance is std::monostate - ignored
  void build_from_encoded_device(const IndexSpec& spec,
                                 NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                                 [[maybe_unused]] std::monostate& provenance) {
    DeviceArray<semiring_value_t<SR>> dummy_prov;
    build_from_encoded_device(spec, encoded_cols, dummy_prov);
  }

  /// @brief Build index by taking ownership of encoded columns (zero-copy for identity spec).
  /// @details For identity-spec, swaps columns (zero-copy). For non-identity, copies+reorders.
  ///          After this call, encoded_cols may be empty (swapped into index).
  ///          Sets rows_processed_ = 0 so is_dirty() returns false with cleared intern cols.
  ///          Caller should clear intern cols after this call.
  void build_take_ownership(const IndexSpec& spec,
                            NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                            DeviceArray<semiring_value_t<SR>>& provenance);

  /// @brief Build index taking ownership (NoProvenance overload)
  void build_take_ownership(const IndexSpec& spec,
                            NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                            [[maybe_unused]] std::monostate& provenance) {
    DeviceArray<semiring_value_t<SR>> dummy_prov;
    build_take_ownership(spec, encoded_cols, dummy_prov);
  }

  /// @brief Build index from another index's column data (index-to-index).
  /// @details Copies columns from an existing index, remapping positions according to
  ///          source_spec → target_spec mapping, then sorts and deduplicates.
  ///          This avoids the roundtrip through intern cols.
  /// @param source_spec The column spec of the source index (defines column layout in source)
  /// @param target_spec The column spec of this (target) index to build
  /// @param source The source index to read column data from (must be non-empty)
  void build_from_index(const IndexSpec& source_spec, const IndexSpec& target_spec,
                        const DeviceSortedArrayIndex& source);

  /// @brief Get root handle (entry point to the index)
  /// @return NodeHandle pointing to the root of the index
  /// @note The returned handle contains device pointers and can be copied to GPU
  /// @warning This handle becomes invalid if merge(), build_from_encoded_device(), or
  ///          clear() is called. Always call root() again after these operations.
  [[nodiscard]] NodeHandle root() const;

  /// @brief Get the view of this index
  [[nodiscard]] NodeView view() const;

  /// @brief Prefix lookup with encoded key
  /// @deprecated This function is not implemented for GPU DeviceSortedArrayIndex.
  /// Use device-based prefix lookup methods instead.
  /// @tparam N Size of the key prefix
  /// @param k Encoded key prefix
  /// @return NodeHandle matching the key prefix (always returns invalid handle)
  template <std::size_t N>
  [[deprecated("Not implemented for GPU DeviceSortedArrayIndex. Use device-based methods "
               "instead.")]] [[nodiscard]] NodeHandle
  prefix_lookup(const EncodedKeyPrefix<N>& k) const;

  /// @brief Get the size of the index (number of unique keys after deduplication)
  [[nodiscard]] std::size_t size() const noexcept {
    if (!p_) {
      return 0;
    }
    return p_->cols.num_rows();
  }

  /// @brief Get the number of relation rows indexed (input size before deduplication)
  /// @details Used for dirty checking: compares with relation.interned_size() to detect when new
  /// rows were added. After deduplication, size() < rows_processed() because duplicates are
  /// removed.
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return rows_processed_;
  }

  /// @brief Update rows_processed to match the relation size after reconstruct
  /// @details After reconstruct_from_index(), the relation's interned_size() matches the index
  /// size. This method updates rows_processed_ to match, preventing is_dirty() from incorrectly
  /// thinking the index needs rebuilding.
  void update_rows_processed(std::size_t new_value) noexcept {
    rows_processed_ = new_value;
  }

  /// @brief Check if index is empty
  /// @details Uses actual column data (size()) instead of rows_processed_ because
  /// the zero-copy swap sets rows_processed_=0 to satisfy is_dirty() invariant
  /// while p_->cols still contains valid data.
  [[nodiscard]] bool empty() const noexcept {
    return size() == 0;
  }

  /// @brief Get memory usage in bytes
  [[nodiscard]] std::size_t bytes_used() const;

  /// @brief Clear the index (deallocates device memory)
  /// @warning This operation invalidates all existing NodeHandles. After clear(), the index
  ///          is empty and root() will return an invalid handle until rebuild.
  void clear() noexcept;

  /// @brief Deep clone from another index
  /// @param other Source index to clone from
  void clone_from(const DeviceSortedArrayIndex& other);

  /// @brief Merge another index into this one
  /// @param other Source index to merge from
  /// @param row_id_offset Offset to add to row IDs from the source index
  /// @warning This operation invalidates all existing NodeHandles. Call root() again
  ///          after merge() to get a fresh handle pointing to the merged index.
  void merge(const DeviceSortedArrayIndex& other, std::size_t row_id_offset);

  /// @brief Compute set difference (this - full_idx) and update full_idx in-place
  /// @param full_idx The 'full' index to subtract from 'this' (newt)
  /// @param delta_idx Output index to store the Delta (Newt - Full).
  /// @note Side Effect: Updates 'full_idx' provenance in-place for matching keys!
  void set_difference_update(DeviceSortedArrayIndex& full_idx, DeviceSortedArrayIndex& delta_idx);

  /// @brief Fused set difference: (this - full_idx - head_idx) → delta_idx
  /// @details Single-kernel anti-join against two sorted arrays (FULL + HEAD).
  ///          Used by Device2LevelIndex to avoid 2-step diff with intermediate temp.
  ///          Only available for NoProvenance semiring.
  void set_difference_update_dual(DeviceSortedArrayIndex& full_idx,
                                  DeviceSortedArrayIndex& head_idx,
                                  DeviceSortedArrayIndex& delta_idx);

  /// @brief Reconstruct interned columns and provenance from this index
  /// @details Only works on device relations (`DeviceRelationPolicy`), and this is very ad-hoc and
  /// need relation must use one single NDDeviceArray for all columns. and DeviceArray for
  /// provenance.
  /// @param spec Index specification (defines column order)
  /// @param output_cols Output interned columns in relation order [arity]
  /// @param output_prov Output provenance array
  void reconstruct_to_relation(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& output_cols,
                               DeviceArray<semiring_value_t<SR>>& output_prov) const;

  /// @brief Reconstruct relation from index data (NoProvenance overload)
  /// @details For NoProvenance, provenance is std::monostate - only reconstruct columns
  void reconstruct_to_relation(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& output_cols,
                               [[maybe_unused]] std::monostate& output_prov) const {
    DeviceArray<semiring_value_t<SR>> dummy_prov;
    reconstruct_to_relation(spec, output_cols, dummy_prov);
  }

  /**
   * @brief LFTJ Intersection (galloping intersection, "seek")
   * @details Returns a View that scans the first span and probes the rest
   * @note GPU version
   * @param tile Cooperative group (can be warp, sub-warp, or even sequential)
   * @param spans Spans of values to intersect
   */
  template <typename Group, typename... Spans>
  __device__ static auto intersect(Group tile, Spans... spans);

  [[nodiscard]] thrust::device_ptr<semiring_value_t<SR>> provenance_ptr() const noexcept {
    if constexpr (has_provenance_v<SR>) {
      if (!p_) {
        return thrust::device_ptr<semiring_value_t<SR>>{};
      }
      return p_->provenance.device_ptr();
    } else {
      return thrust::device_ptr<semiring_value_t<SR>>{};
    }
  }

  // void set_provenance_ptr(thrust::device_ptr<SR> provenance_ptr) noexcept {
  //   provenance_ptr_ = provenance_ptr;
  // }

  [[nodiscard]] auto& data() const {
    if (!p_) {
      throw std::runtime_error("DeviceSortedArrayIndex::data() called on uninitialized index");
    }
    return p_->cols;
  }

  /// @brief Get the number of unique values in the first column (for partitioning work)
  /// @return Number of unique values in root_unique_values, or 0 if empty
  [[nodiscard]] std::size_t num_unique_root_values() const noexcept {
    if (!p_) {
      return 0;
    }
    return p_->root_unique_values.size();
  }

  [[nodiscard]] auto& root_unique_values() const {
    if (!p_) {
      throw std::runtime_error(
          "DeviceSortedArrayIndex::root_unique_values() called on uninitialized index");
    }
    return p_->root_unique_values;
  }

  /// @brief Debug helper to print index contents
  void print_debug() const {
    if (!p_) {
      printf("DeviceSortedArrayIndex: Uninitialized\n");
      return;
    }
    // Delegate to NDDeviceArray's print_first
    // Note: print_first uses std::cout, so we flush it
    p_->cols.print_first(100, std::cout);
    std::cout.flush();
  }

 private:
  struct Impl {
    // Device arrays for column data (one per indexed column)
    // Uses NDDeviceArray for efficient column-oriented storage
    NDDeviceArray<ValueTypeParam, arity> cols;

    // Unique values in first column (for partitioning work only, not for correctness)
    DeviceArray<ValueTypeParam> root_unique_values;

    // Conditional device array for provenance/semiring values
    // Uses [[no_unique_address]] to eliminate storage when NoProvenance is used
    [[no_unique_address]] std::conditional_t<has_provenance_v<SR>,
                                             DeviceArray<semiring_value_t<SR>>, std::monostate>
        provenance;

    std::size_t index_arity{0};

    void clear() {
      // Clear non-owning views first to avoid confusion
      cols.clear();
      if constexpr (has_provenance_v<SR>) {
        provenance.clear();
      }
      root_unique_values.clear();
    }
  };

  std::unique_ptr<Impl> p_;
  std::size_t rows_processed_{0};  // Input size (before deduplication) for dirty checking: tracks
                                   // how many relation rows were indexed

  void sort_with_permutation(std::size_t num_rows, std::span<const int> sort_order,
                             SortContext<RowIdType>& ctx);

  void gather_by_permutation(std::span<const RowIdType> permutation,
                             DeviceArray<semiring_value_t<SR>>& provenance);
};

}  // namespace SRDatalog::GPU

// Include implementation
#include "device_sorted_array_index.ipp"
