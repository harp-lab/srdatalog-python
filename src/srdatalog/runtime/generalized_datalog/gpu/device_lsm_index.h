/**
 * @file device_lsm_index.h
 * @brief GPU-based LSM Index (Indirect Index) for GPU kernels
 *
 * @details This index stores a permutation (RowId array) that provides a sorted view
 * over the underlying relation data, without modifying the relation data itself.
 * This corresponds to the "Index" part of "Index-Sorted Array" (ISA).
 */

#pragma once

#include "../index.h"
#include "../semiring.h"
#include "./macro.h"
#include "./sort.h"
#include "device_array.h"
#include "gpu/shim/thrust/cccl_compat.h"
#include "nd_device_array.h"
#include <array>
#include <cstddef>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <span>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/system/cuda/execution_policy.h>
#include <tuple>

namespace SRDatalog::GPU {

// Forward declaration
using IndexSpec = SRDatalog::IndexSpec;

// Forward declaration of IntersectView (defined in .ipp)
template <typename ValueType, typename RowIdType, std::size_t N, typename Group>
class IntersectView;

/**
 * @brief Immutable view of the LSM index (pointers only).
 */
template <Semiring SR, typename ValueTypeParam = uint32_t>
struct LsmNodeView {
  // Indirect Pointers
  const uint32_t* sorted_indices_{nullptr};  // Mapping: Sorted Position -> Row ID
  const ValueTypeParam* col_data_{nullptr};  // Relation Data (Column Major)
  semiring_value_t<SR>* provenance_ptr_{nullptr};

  // Metadata
  uint32_t arity_{0};
  uint32_t stride_{0};    // Stride for col_data (number of rows in relation)
  uint32_t num_rows_{0};  // Valid number of indexed elements

  // Access via Indirection
  [[nodiscard]] GPU_HD ValueTypeParam get_value(uint32_t depth,
                                                uint32_t sorted_idx) const noexcept {
    // 1. Get actual row ID from permutation
    uint32_t row_id = sorted_indices_[sorted_idx];
    // 2. Fetch value from column data
    return col_data_[depth * stride_ + row_id];
  }

  // Access provenance via Indirection
  [[nodiscard]] GPU_HD semiring_value_t<SR> get_provenance(uint32_t sorted_idx) const noexcept {
    uint32_t row_id = sorted_indices_[sorted_idx];
    return provenance_ptr_[row_id];
  }
};

/**
 * @brief NodeHandle for LSM Index (Indirect Access)
 */
template <Semiring SR, typename ValueTypeParam = uint32_t, typename RowIdType = uint32_t>
class LsmNodeHandle {
 public:
  using View = LsmNodeView<SR, ValueTypeParam>;

  GPU_HD LsmNodeHandle() = default;
  GPU_HD LsmNodeHandle(RowIdType begin, RowIdType end, uint32_t depth)
      : begin_(begin), end_(end), depth_(depth) {}

  [[nodiscard]] GPU_HD bool valid() const noexcept {
    return begin_ < end_;
  }

  [[nodiscard]] GPU_HD bool is_leaf(const View& view) const noexcept {
    return valid() && depth_ == view.arity_;
  }

  [[nodiscard]] GPU_HD uint32_t depth() const noexcept {
    return depth_;
  }

  [[nodiscard]] GPU_HD std::size_t degree() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - begin_) : 0;
  }

  [[nodiscard]] GPU_HD std::size_t unique_count() const noexcept {
    return degree();
  }

  // Important: values() returns a span... but for indirect access, we can't easily return
  // a contiguous span of *values* because values are scattered in memory!
  // BUT the Concept requires values().
  //
  // Solution: For now, we might not meet the full Concept if it strictly requires contiguous
  // memory. HOWEVER, SRDatalog's `values()` is mostly used for binary search or iteration. If we
  // return the *sorted indices*, the caller must know to dereference them. OR, we assume
  // LsmNodeHandle is primarily used via specialized kernels (like we planned) that don't call
  // generic `values()`.
  //
  // Let's implement `prefix` logic manually here.

  // Custom prefix search with indirection
  template <typename Group>
  [[nodiscard]] __device__ LsmNodeHandle prefix(ValueTypeParam key, Group tile,
                                                const View& view) const;

  [[nodiscard]] GPU_HD static LsmNodeHandle invalid() noexcept {
    return LsmNodeHandle{0, 0, 0};
  }

  // Accessors for range (Indices in sorted_indices array)
  [[nodiscard]] GPU_HD RowIdType begin() const noexcept {
    return begin_;
  }
  [[nodiscard]] GPU_HD RowIdType end() const noexcept {
    return end_;
  }

 private:
  RowIdType begin_{0};
  RowIdType end_{0};
  uint32_t depth_{0};
};

/**
 * @brief Device LSM Index
 */
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam = uint32_t,
          typename RowIdType = uint32_t>
class DeviceLsmIndex {
 public:
  using ValueType = ValueTypeParam;
  using RowId = RowIdType;
  using RowRange = cuda_std::span<const RowId>;
  // ValueRange issue: we don't have contiguous values.
  // We satisfy the Concept loosely or specialized check.

  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  using NodeHandle = SRDatalog::GPU::LsmNodeHandle<SR, ValueType, RowId>;
  using NodeView = SRDatalog::GPU::LsmNodeView<SR, ValueType>;

  DeviceLsmIndex() = default;
  ~DeviceLsmIndex() = default;

  DeviceLsmIndex(const DeviceLsmIndex&) = delete;
  DeviceLsmIndex& operator=(const DeviceLsmIndex&) = delete;
  DeviceLsmIndex(DeviceLsmIndex&&) noexcept = default;
  DeviceLsmIndex& operator=(DeviceLsmIndex&&) noexcept = default;

  // Build from device data (Relation's data)
  // Logic:
  // 1. Create a sequence 0..N
  // 2. Sort the sequence based on comparison of rows in `encoded_cols`
  void build_from_encoded_device(const IndexSpec& spec,
                                 NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                                 DeviceArray<semiring_value_t<SR>>& provenance);

  [[nodiscard]] NodeHandle root() const;
  [[nodiscard]] NodeView view() const;

  [[nodiscard]] std::size_t size() const noexcept {
    return num_rows_;
  }
  [[nodiscard]] bool empty() const noexcept {
    return num_rows_ == 0;
  }

  void clear() noexcept;

  // Placeholder for compaction
  void compact(NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
               DeviceArray<semiring_value_t<SR>>& provenance);

  /**
   * @brief Merges other index into this one using linear O(N+M) set_union (equivalent to merge for
   * disjoint ranges).
   *
   * @details This function performs a logical merge of sorted indices.
   * - It allocates new storage for `sorted_indices_`.
   * - It copies `other.sorted_indices_` and adds `row_id_offset`.
   * - It merges them with `this->sorted_indices_` using a custom comparator.
   *
   * @param other Index to merge into this one.
   * @param row_id_offset Offset to add to `other`'s RowIDs. This MUST map `other`'s 0-based IDs
   *                      to their new position in `this`'s data storage.
   *
   * @warning **Memory Safety Contract**:
   * The `DeviceLsmIndex` observes data via `data_ptr_`, but does NOT own it.
   * After merging, `this` index contains RowIDs referring to data that was originally indexed by
   * `other`.
   * **The caller must ensure that `other`'s underlying data remains valid and accessible via
   * `this->data_ptr_`.**
   *
   * Common Runtime Patterns:
   * 1. **Append-Bind-Merge**: Append `other`'s data to `this`'s storage (reallocating if needed),
   * call `bind_data()` to update `this->data_ptr_` to the new buffer, then call `merge()`.
   * 2. **Stable Buffer Handoff**: If `data_ptr_` points to a stable segmented array (simulated via
   * checking bounds or unified memory), the runtime must *preserve* `other`'s data buffer (do not
   * free it in `other.clear()`) and simply link it to the segmented storage, so `this->data_ptr_`
   * can access it.
   */
  void merge(const DeviceLsmIndex& other, std::size_t row_id_offset);
  void bind_data(NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
                 DeviceArray<semiring_value_t<SR>>& provenance);
  [[nodiscard]] std::size_t bytes_used() const;

 private:
  // We own the PERMUTATION
  DeviceArray<RowIdType> sorted_indices_;

  // We keep non-owning reference to the data (View) for constructing the View
  const ValueType* data_ptr_ = nullptr;
  semiring_value_t<SR>* prov_ptr_ = nullptr;
  std::size_t stride_ = 0;
  std::size_t num_rows_ = 0;

  // Internal build helpers
  void sort_indices(const IndexSpec& spec, NDDeviceArray<ValueTypeParam, arity>& cols);
};

}  // namespace SRDatalog::GPU

#include "device_lsm_index.ipp"
