#pragma once
/**
 * @file sorted_array_index.ipp
 * @brief Implementation of GPU Sorted Array Index
 */

#include "./device_sorted_array_index.h"
// #include "logging.h"
#include "macro.h"
#ifdef USE_ROCm
// Include CCCL compatibility shim before thrust headers
#include "gpu/shim/thrust/cccl_compat.h"
// Include CUDA error constants shim before thrust/system/cuda headers
#include "gpu/shim/thrust/system/cuda/cuda_error_constants.h"
#endif
#include <cassert>
#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <cstdio>
#include <numeric>
// NVTX profiling support
#if defined(USE_ROCm)
// nvtx3 is NVIDIA-specific profiling - stub out on ROCm
#define nvtxRangePushA(name) ((void)0)
#define nvtxRangePop() ((void)0)
#else
// Always include nvtx3 on CUDA builds
#include <nvtx3/nvToolsExt.h>
#endif
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <vector>

#include "./deduplication.h"
#include "./search.h"
#include "./tiling.h"
#include "runtime/custom_set_op.h"

namespace SRDatalog::GPU {

// function to get sort context in current thread
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
static SortContext<ValueType>& get_sort_context() {
  static thread_local SortContext<ValueType> ctx;
  return ctx;
}

// Template version with configurable group size
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
template <typename Group>
__device__ NodeHandle<SR, ValueTypeParam, RowIdType>
NodeHandle<SR, ValueTypeParam, RowIdType>::prefix(ValueTypeParam key, Group tile,
                                                  const View& view) const {
  if (!valid() || is_leaf(view)) {
    return invalid();
  }

  // Deeper levels: search in column
  // Get column base pointer, then offset by begin_ (matching values() implementation)
  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + begin_;
  RowIdType n = end_ - begin_;

  // Use group_equal_range for efficient range finding
  auto [lower, upper] = group_equal_range<ValueTypeParam, RowIdType>(col, n, key, tile);

  if (lower == upper) {
    return invalid();
  }

  return child(begin_ + lower, begin_ + upper);
}

// Sequential (non-cooperative) prefix search - for Cartesian loops
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
__device__ NodeHandle<SR, ValueTypeParam, RowIdType>
NodeHandle<SR, ValueTypeParam, RowIdType>::prefix_seq(ValueTypeParam key, const View& view) const {
  if (!valid() || is_leaf(view)) {
    return invalid();
  }

  // Get column base pointer, then offset by begin_
  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + begin_;
  RowIdType n = end_ - begin_;

  // Use sequential equal_range (no cooperative primitives)
  auto [lower, upper] = seq_equal_range<ValueTypeParam, RowIdType>(col, n, key);

  if (lower == upper) {
    return invalid();
  }

  return child(begin_ + lower, begin_ + upper);
}

// One-side search: only lower_bound, returns position for use with child_range()
// Pattern: prefix_lower_only() + child_range() = 1 search each instead of 2 in prefix()
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
template <typename Group>
__device__ thrust::pair<RowIdType, bool>
NodeHandle<SR, ValueTypeParam, RowIdType>::prefix_lower_only(ValueTypeParam key, Group tile,
                                                             const View& view) const {
  if (!valid() || is_leaf(view)) {
    return {0, false};
  }

  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + begin_;
  RowIdType n = end_ - begin_;

  // Only do lower_bound search
  RowIdType lower = group_exp_lower_bound<ValueTypeParam, RowIdType>(col, n, key, tile);

  // Check if key actually exists at this position
  bool found = (lower < n) && (col[lower] == key);

  return {lower, found};
}

// LFTJ optimization: Create child with proper range bounds from known lower bound position
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
template <typename Group>
__device__ NodeHandle<SR, ValueTypeParam, RowIdType>
NodeHandle<SR, ValueTypeParam, RowIdType>::child_range(RowIdType position, ValueTypeParam key,
                                                       Group tile,
                                                       const View& view) const noexcept {
  RowIdType new_begin = begin_ + position;

  // Search for upper bound in remaining range [new_begin, end_)
  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + new_begin;
  RowIdType remaining = end_ - new_begin;

  RowIdType upper_offset =
      group_exp_upper_bound<ValueTypeParam, RowIdType>(col, remaining, key, tile);
  RowIdType new_end = new_begin + upper_offset;

  return child(new_begin, new_end);
}

template <Semiring SR, typename ValueType, typename RowIdType>
__device__ RowIdType NodeHandle<SR, ValueType, RowIdType>::row_position() const noexcept {
  return begin_;
}

template <Semiring SR, typename ValueType, typename RowIdType>
__device__ RowIdType NodeHandle<SR, ValueType, RowIdType>::column_position() const noexcept {
  return depth_;
}

template <Semiring SR, typename ValueType, typename RowIdType>
template <int COLUMN>
__device__ RowIdType NodeHandle<SR, ValueType, RowIdType>::offset(RowIdType row,
                                                                  const View& view) const noexcept {
  return COLUMN * view.stride_ + row;
}

template <Semiring SR, typename ValueType, typename RowIdType>
template <int TILE_SIZE>
__device__ semiring_value_t<SR> NodeHandle<SR, ValueType, RowIdType>::prov(
    const View& view) const noexcept {
  if (end_ == begin_)
    return sr_zero<SR>();

  if (end_ - begin_ == 1) {
    return view.provenance_ptr_[begin_];  // Fast path: single row
  }
  return tile_semiring_reduce<SR, TILE_SIZE>(view.provenance_ptr_, begin_, end_);
}

template <Semiring SR, typename ValueType, typename RowIdType>
template <typename Group>
__device__ bool NodeHandle<SR, ValueType, RowIdType>::contains_value(ValueType key,
                                                                     const View& view,
                                                                     Group tile) const noexcept {
  if (!valid() || is_leaf(view)) {
    return false;
  }

  // Get column base pointer, then offset by begin_ (matching values() implementation)
  const ValueType* col_base = view.col_data() + depth_ * view.stride_;
  const ValueType* col = col_base + begin_;
  RowIdType n = end_ - begin_;
  return group_contains<ValueType, RowIdType>(col, n, key, tile);
}

// Binary join optimization: Single lower_bound search (saves ~50% search work)
// Returns handle from lower_bound to end of current range.
// Caller detects end of matching values by comparing during iteration.
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
template <typename Group>
__device__ NodeHandle<SR, ValueTypeParam, RowIdType>
NodeHandle<SR, ValueTypeParam, RowIdType>::prefix_lower_bound(ValueTypeParam key, Group tile,
                                                              const View& view) const {
  if (!valid() || is_leaf(view)) {
    return invalid();
  }

  // Get column base pointer, then offset by begin_
  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + begin_;
  RowIdType n = end_ - begin_;

  // Single lower_bound search - NO upper_bound!
  RowIdType lower = group_lower_bound<ValueTypeParam, RowIdType>(col, n, key, tile);

  // Check if key exists (lower bound points to it)
  if (lower >= n || col[lower] != key) {
    return invalid();
  }

  // Return handle from lower_bound to END of current range
  // The caller will detect end of matching values during iteration by checking value != key
  return child(begin_ + lower, end_);
}

// Sequential version for Cartesian loops
template <Semiring SR, typename ValueTypeParam, typename RowIdType>
__device__ NodeHandle<SR, ValueTypeParam, RowIdType>
NodeHandle<SR, ValueTypeParam, RowIdType>::prefix_lower_bound_seq(ValueTypeParam key,
                                                                  const View& view) const {
  if (!valid() || is_leaf(view)) {
    return invalid();
  }

  // Get column base pointer, then offset by begin_
  const ValueTypeParam* col_base = view.col_data() + depth_ * view.stride_;
  const ValueTypeParam* col = col_base + begin_;
  RowIdType n = end_ - begin_;

  // Single sequential lower_bound search - NO upper_bound!
  RowIdType lower = seq_lower_bound<ValueTypeParam, RowIdType>(col, n, key);

  // Check if key exists
  if (lower >= n || col[lower] != key) {
    return invalid();
  }

  // Return handle from lower_bound to END - caller checks values during iteration
  return child(begin_ + lower, end_);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_encoded_device(
    const IndexSpec& spec, NDDeviceArray<ValueType, arity>& encoded_cols,
    DeviceArray<semiring_value_t<SR>>& provenance) {
  const std::size_t num_rows = encoded_cols.num_rows();
  if (num_rows == 0 || spec.cols.empty()) {
    clear();
    return;
  }

  // Initialize impl
  if (!p_) {
    p_ = std::make_unique<Impl>();
  } else {
    p_->clear();
  }

  p_->index_arity = spec.cols.size();
  rows_processed_ = num_rows;

  // Step 1: Copy and reorder columns per spec
  const std::size_t index_arity = p_->index_arity;

  nvtxRangePushA("Index_Copy_Columns");
  {
    p_->cols.resize(num_rows);
    for (std::size_t idx_pos = 0; idx_pos < index_arity; ++idx_pos) {
      const std::size_t src_col = spec.cols[idx_pos];
      thrust::copy_n(rmm::exec_policy{},
                     thrust::device_ptr<const ValueType>(encoded_cols.column_ptr(src_col)),
                     num_rows, thrust::device_ptr<ValueType>(p_->cols.column_ptr(idx_pos)));
    }
  }
  nvtxRangePop();  // Index_Copy_Columns

  // Step 2: Copy provenance parameter into member (only if semiring has provenance)
  if constexpr (has_provenance_v<SR>) {
    p_->provenance.resize(num_rows);
    nvtxRangePushA("Index_Copy_Provenance");
    if (provenance.size() == num_rows) {
      // Copy from parameter if provided - use GPU_MEMCPY for better performance
      GPU_MEMCPY(p_->provenance.data(), provenance.data(), num_rows * sizeof(semiring_value_t<SR>),
                 GPU_DEVICE_TO_DEVICE);
    } else {
      // Initialize to SR::one() if parameter is empty or different size
      // thrust::fill is needed here (can't use memset for non-trivial types)
      thrust::fill(thrust::device, p_->provenance.begin(), p_->provenance.end(), sr_one<SR>());
    }
    nvtxRangePop();  // Index_Copy_Provenance
  }

  // Step 3: Sort lexicographically
  // Build sort order [0, 1, 2, ..., index_arity-1]
  std::vector<int> sort_order(p_->index_arity);
  std::iota(sort_order.begin(), sort_order.end(), 0);

  // Get thread-local sort context
  // Note: The template parameter is the index type (for permutation), not the value type
  // Value types are inferred from column pointers passed to stable_lex_sort
  // Use RowIdType for consistency with the class template parameter
  static thread_local SortContext<RowIdType> sort_ctx;

  // nvtxRangePushA("Index_Sort");
  sort_with_permutation(num_rows, sort_order, sort_ctx);
  // nvtxRangePop();  // Index_Sort

  // Gather provenance by permutation (only if has provenance)
  if constexpr (has_provenance_v<SR>) {
    // Convert DeviceArray to span for gather_by_permutation
    nvtxRangePushA("Index_Gather_Provenance");
    gather_by_permutation(
        std::span<const RowIdType>(sort_ctx.permutation.data(), sort_ctx.permutation.size()),
        p_->provenance);
    nvtxRangePop();  // Index_Gather_Provenance
  }

  // Step 4: Deduplicate rows, aggregate provenance, and build root unique values
  nvtxRangePushA("Index_Deduplicate");
  deduplicate_aggregate_and_unique<SR>(p_->cols, p_->provenance, p_->root_unique_values);
  nvtxRangePop();  // Index_Deduplicate

  // Note: rows_processed_ tracks input size (num_rows) for dirty checking.
  // The actual index size after deduplication is p_->cols.num_rows().
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::build_take_ownership(
    const IndexSpec& spec, NDDeviceArray<ValueType, arity>& encoded_cols,
    DeviceArray<semiring_value_t<SR>>& provenance) {
  const std::size_t num_rows = encoded_cols.num_rows();
  if (num_rows == 0 || spec.cols.empty()) {
    clear();
    return;
  }

  // Initialize impl
  if (!p_) {
    p_ = std::make_unique<Impl>();
  } else {
    p_->clear();
  }

  p_->index_arity = spec.cols.size();
  const std::size_t index_arity = p_->index_arity;

  // Step 1: Transfer columns — zero-copy swap for identity spec, copy otherwise
  bool is_identity_spec = (index_arity == arity);
  if (is_identity_spec) {
    for (std::size_t i = 0; i < index_arity; ++i) {
      if (spec.cols[i] != static_cast<int>(i)) {
        is_identity_spec = false;
        break;
      }
    }
  }

  nvtxRangePushA("Index_TakeOwnership_Columns");
  if (is_identity_spec) {
    // Identity spec: swap ownership — zero-copy transfer.
    p_->cols.swap(encoded_cols);
  } else {
    // Non-identity: copy and reorder columns per spec (same stream as sort)
    p_->cols.resize(num_rows);
    for (std::size_t idx_pos = 0; idx_pos < index_arity; ++idx_pos) {
      const std::size_t src_col = spec.cols[idx_pos];
      thrust::copy_n(rmm::exec_policy{},
                     thrust::device_ptr<const ValueType>(encoded_cols.column_ptr(src_col)),
                     num_rows, thrust::device_ptr<ValueType>(p_->cols.column_ptr(idx_pos)));
    }
  }
  nvtxRangePop();  // Index_TakeOwnership_Columns

  // Step 2: Copy provenance
  if constexpr (has_provenance_v<SR>) {
    p_->provenance.resize(num_rows);
    nvtxRangePushA("Index_Copy_Provenance");
    if (provenance.size() == num_rows) {
      GPU_MEMCPY(p_->provenance.data(), provenance.data(), num_rows * sizeof(semiring_value_t<SR>),
                 GPU_DEVICE_TO_DEVICE);
    } else {
      thrust::fill(thrust::device, p_->provenance.begin(), p_->provenance.end(), sr_one<SR>());
    }
    nvtxRangePop();
  }

  // Step 3: Sort
  std::vector<int> sort_order(p_->index_arity);
  std::iota(sort_order.begin(), sort_order.end(), 0);
  static thread_local SortContext<RowIdType> sort_ctx;
  sort_with_permutation(num_rows, sort_order, sort_ctx);

  // Gather provenance by permutation
  if constexpr (has_provenance_v<SR>) {
    nvtxRangePushA("Index_Gather_Provenance");
    gather_by_permutation(
        std::span<const RowIdType>(sort_ctx.permutation.data(), sort_ctx.permutation.size()),
        p_->provenance);
    nvtxRangePop();
  }

  // Step 4: Deduplicate
  nvtxRangePushA("Index_Deduplicate");
  deduplicate_aggregate_and_unique<SR>(p_->cols, p_->provenance, p_->root_unique_values);
  nvtxRangePop();

  // Mark intern cols as consumed — caller should clear them.
  // is_dirty() will see interned_size(0) == rows_processed_(0) → false.
  rows_processed_ = 0;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_index(
    const IndexSpec& source_spec, const IndexSpec& target_spec,
    const DeviceSortedArrayIndex& source) {
  // Source must have data
  if (!source.p_ || source.p_->cols.num_rows() == 0) {
    clear();
    return;
  }

  const std::size_t num_rows = source.p_->cols.num_rows();
  const std::size_t target_arity = target_spec.cols.size();

  // Initialize impl
  if (!p_) {
    p_ = std::make_unique<Impl>();
  } else {
    p_->clear();
  }

  p_->index_arity = target_arity;

  // Step 1: Copy columns from source with remapping.
  // Source's p_->cols stores columns in source_spec order:
  //   source position i holds original column source_spec.cols[i].
  // Target needs columns in target_spec order:
  //   target position j needs original column target_spec.cols[j].
  // Remap: for each target position j, find source position i where
  //   source_spec.cols[i] == target_spec.cols[j].
  nvtxRangePushA("Index_BuildFromIndex_Columns");
  {
    p_->cols.resize(num_rows);

    for (std::size_t target_pos = 0; target_pos < target_arity; ++target_pos) {
      const int target_col = target_spec.cols[target_pos];
      // Find this column in the source spec
      std::size_t source_pos = 0;
      bool found = false;
      for (std::size_t i = 0; i < source_spec.cols.size(); ++i) {
        if (source_spec.cols[i] == target_col) {
          source_pos = i;
          found = true;
          break;
        }
      }
      assert(found && "target column not found in source spec");

      thrust::copy_n(rmm::exec_policy{},
                     thrust::device_ptr<const ValueType>(source.p_->cols.column_ptr(source_pos)),
                     num_rows, thrust::device_ptr<ValueType>(p_->cols.column_ptr(target_pos)));
    }
  }
  nvtxRangePop();  // Index_BuildFromIndex_Columns

  // Step 2: Copy provenance from source
  if constexpr (has_provenance_v<SR>) {
    p_->provenance.resize(num_rows);
    nvtxRangePushA("Index_Copy_Provenance");
    if (source.p_->provenance.size() == num_rows) {
      GPU_MEMCPY(p_->provenance.data(), source.p_->provenance.data(),
                 num_rows * sizeof(semiring_value_t<SR>), GPU_DEVICE_TO_DEVICE);
    } else {
      thrust::fill(thrust::device, p_->provenance.begin(), p_->provenance.end(), sr_one<SR>());
    }
    nvtxRangePop();
  }

  // Step 3: Sort lexicographically
  std::vector<int> sort_order(p_->index_arity);
  std::iota(sort_order.begin(), sort_order.end(), 0);
  static thread_local SortContext<RowIdType> sort_ctx;
  sort_with_permutation(num_rows, sort_order, sort_ctx);

  // Gather provenance by permutation
  if constexpr (has_provenance_v<SR>) {
    nvtxRangePushA("Index_Gather_Provenance");
    gather_by_permutation(
        std::span<const RowIdType>(sort_ctx.permutation.data(), sort_ctx.permutation.size()),
        p_->provenance);
    nvtxRangePop();
  }

  // Step 4: Deduplicate
  nvtxRangePushA("Index_Deduplicate");
  deduplicate_aggregate_and_unique<SR>(p_->cols, p_->provenance, p_->root_unique_values);
  nvtxRangePop();

  // Not from intern cols — set rows_processed_ = 0.
  rows_processed_ = 0;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_encoded(
    const IndexSpec& spec, const std::array<std::span<const ValueType>, arity>& encoded_rows) {
  std::cout << "build_from_encoded not implemented for GPU DeviceSortedArrayIndex, use device "
               "build instead"
            << std::endl;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::sort_with_permutation(
    std::size_t num_rows, std::span<const int> sort_order, SortContext<RowIdType>& ctx) {

  // For NoProvenance with full-arity identity sort order:
  // Use zip sort directly (single-pass merge sort, ~18% faster at arity 3).
  // This avoids multi-pass radix sort + gather overhead entirely.
  // Only need permutation-based sort when provenance requires reordering.
  if constexpr (!has_provenance_v<SR>) {
    if (sort_order.size() == arity) {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        zip_sort_columns(num_rows, p_->cols.template column_ptr<Is>()...);
      }
      (std::make_index_sequence<arity>());
      return;
    }
  }

  // Fallback: multi-pass radix sort with permutation (needed for provenance gather)
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    stable_lex_sort<RowIdType>(num_rows, sort_order, ctx,
                               true,  // reorder_data = true
                               p_->cols.template column_ptr<Is>()...);
  }
  (std::make_index_sequence<arity>());
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::gather_by_permutation(
    std::span<const RowIdType> permutation, DeviceArray<semiring_value_t<SR>>& provenance) {
  thrust::gather(rmm::exec_policy{}, permutation.begin(), permutation.end(), provenance.begin(),
                 thrust::device_ptr<semiring_value_t<SR>>(provenance.data()));
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
typename DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle
DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::root() const {
  if (empty() || !p_) {
    return NodeHandle{};
  }
  // Use actual index size (after deduplication) for handle range
  return NodeHandle{static_cast<RowIdType>(0), static_cast<RowIdType>(p_->cols.num_rows()),
                    static_cast<uint32_t>(0)};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
typename DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeView
DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::view() const {
  if (empty() || !p_) {
    return NodeView{};
  }
  auto cols_view = p_->cols.view();
  // Provenance pointer: nullptr if NoProvenance, actual data otherwise
  semiring_value_t<SR>* prov_ptr = nullptr;
  if constexpr (has_provenance_v<SR>) {
    prov_ptr = p_->provenance.data();
  }
  NodeView result{cols_view.raw(), prov_ptr, static_cast<uint32_t>(p_->index_arity),
                  static_cast<uint32_t>(cols_view.get_stride()),
                  static_cast<uint32_t>(cols_view.num_rows)};
  return result;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <std::size_t N>
[[deprecated("Not implemented for GPU DeviceSortedArrayIndex. Use device-based methods instead.")]]
typename DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle
DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::prefix_lookup(
    const EncodedKeyPrefix<N>& k) const {
  // This function is deprecated and not implemented for GPU DeviceSortedArrayIndex.
  // Use device-based prefix lookup methods instead.
  std::cout << "prefix_lookup is deprecated and not implemented for GPU DeviceSortedArrayIndex. "
            << "Use device-based prefix lookup methods instead." << std::endl;
  return NodeHandle{};  // Always returns invalid handle
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
std::size_t DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::bytes_used() const {
  if (!p_) {
    return 0;
  }

  std::size_t b = sizeof(Impl);
  for (std::size_t i = 0; i < p_->index_arity; ++i) {
    b += p_->cols.num_rows() * sizeof(ValueType);
  }
  b += p_->provenance.size() * sizeof(semiring_value_t<SR>);
  b += p_->root_unique_values.size() * sizeof(ValueType);
  return b;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::clear() noexcept {
  if (p_) {
    p_->clear();
  }
  rows_processed_ = 0;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::clone_from(
    const DeviceSortedArrayIndex& other) {
  if (other.empty() || !other.p_) {
    clear();
    return;
  }

  if (!p_) {
    p_ = std::make_unique<Impl>();
  }
  // Always clear existing data to ensure clean state before cloning
  // This resets all arrays to size 0, but keeps capacity for efficiency
  p_->clear();

  p_->index_arity = other.p_->index_arity;
  rows_processed_ = other.rows_processed_;

  // Clone column arrays using NDDeviceArray
  // Note: arity is the compile-time number of columns in the tuple
  // We need to copy all arity columns, not just index_arity indexed ones
  p_->cols.resize(other.p_->cols.num_rows());
  for (std::size_t i = 0; i < arity; ++i) {
    thrust::copy(rmm::exec_policy{}, other.p_->cols.column_ptr(i),
                 other.p_->cols.column_ptr(i) + other.p_->cols.num_rows(), p_->cols.column_ptr(i));
  }

  // Clone provenance (only if semiring has provenance)
  if constexpr (has_provenance_v<SR>) {
    p_->provenance.resize(other.p_->provenance.size());
    if (other.p_->provenance.size() > 0) {
      thrust::copy(rmm::exec_policy{}, other.p_->provenance.begin(), other.p_->provenance.end(),
                   p_->provenance.begin());
    }
  }

  // Clone root unique values
  p_->root_unique_values.resize(other.p_->root_unique_values.size());
  if (other.p_->root_unique_values.size() > 0) {
    thrust::copy(rmm::exec_policy{}, other.p_->root_unique_values.begin(),
                 other.p_->root_unique_values.end(), p_->root_unique_values.begin());
  }
}

// =============================================================================
// Fast check kernel for root_unique_values: are all new values already present?
// If yes, skip thrust::set_union entirely (the common case in tail iterations).
// =============================================================================

/// @brief Single-thread kernel: binary search each new value in sorted arr.
///        Writes 1 if any value is missing (set_union needed), 0 if all present (skip).
template <typename ValueType>
__global__ void check_all_present_kernel(const ValueType* __restrict__ arr, uint32_t arr_size,
                                         const ValueType* __restrict__ new_values,
                                         uint32_t new_count, uint32_t* __restrict__ need_union) {
  for (uint32_t ni = 0; ni < new_count; ++ni) {
    ValueType val = new_values[ni];
    uint32_t lo = 0;
    uint32_t hi = arr_size;
    while (lo < hi) {
      uint32_t mid = lo + ((hi - lo) / 2);
      if (arr[mid] < val) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo >= arr_size || arr[lo] != val) {
      *need_union = 1;
      return;
    }
  }
  *need_union = 0;
}

/// Max delta root_unique_values size to use the fast check path.
static constexpr std::size_t kCheckPresentMaxNew = 32;

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::merge(
    const DeviceSortedArrayIndex& other, [[maybe_unused]] std::size_t row_id_offset) {
  if (other.empty()) {
    return;
  }
  if (empty()) {
    clone_from(other);
    return;
  }

  // Verify CUDA device is still valid before merge
  int current_device = -1;
  GPU_ERROR_T err = GPU_GET_DEVICE(&current_device);
  if (err != GPU_SUCCESS) {
    throw std::runtime_error("merge: CUDA device context lost. Error: " +
                             std::string(GPU_GET_ERROR_STRING(err)));
  }

  // Synchronize device to ensure all previous operations are complete
  err = GPU_DEVICE_SYNCHRONIZE();
  if (err != GPU_SUCCESS) {
    throw std::runtime_error("merge: Device synchronization failed before merge. Error: " +
                             std::string(GPU_GET_ERROR_STRING(err)));
  }

  const auto full_rows = p_->cols.num_rows();
  const auto delta_rows = other.p_->cols.num_rows();
  const auto total_rows = full_rows + delta_rows;

  auto make_key_iter = [](const auto& arr) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
    }
    (std::make_index_sequence<arity>());
  };

  // For merged_cols, we need mutable view since we're writing to it
  // Get column pointers directly from array (not through view) to ensure correct stride
  auto make_mutable_key_iter = [](auto& arr) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
    }
    (std::make_index_sequence<arity>());
  };
  // Check if arrays are sorted before merge (merge_by_key requires sorted inputs)
  auto full_keys = make_key_iter(p_->cols);
  auto other_keys = make_key_iter(other.p_->cols);

  // Allocate temporary arrays for merge - RMM pool memory resource handles efficient reuse
  // The pool allocator automatically reuses freed memory blocks, so we don't need manual buffer
  // management
  NDDeviceArray<ValueType, arity> merged_cols;
  merged_cols.resize(total_rows);

  const std::size_t max_unique =
      p_->root_unique_values.size() + other.p_->root_unique_values.size();

  // Verify allocations succeeded
  err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    throw std::runtime_error("merge: CUDA error during memory allocation. Error: " +
                             std::string(GPU_GET_ERROR_STRING(err)));
  }

  auto merged_keys = make_mutable_key_iter(merged_cols);

  if constexpr (has_provenance_v<SR>) {
    DeviceArray<semiring_value_t<SR>> merged_prov;
    merged_prov.resize(total_rows);
    {
      nvtx3::scoped_range r("merge_by_key");
      // Use rmm::exec_policy() for proper memory allocation through RMM pool
      thrust::merge_by_key(rmm::exec_policy{}, full_keys, full_keys + full_rows, other_keys,
                           other_keys + delta_rows, p_->provenance.begin(),
                           other.p_->provenance.begin(), merged_keys, merged_prov.begin());
    }
    // Swap merged data into member variables
    p_->cols.swap(merged_cols);
    p_->provenance.swap(merged_prov);

    // Free temporary memory
    merged_cols.reset();
    merged_prov.reset();
  } else {
    // No provenance - use simpler merge (keys only)
    {
      nvtx3::scoped_range r("merge");
      thrust::merge(rmm::exec_policy{}, full_keys, full_keys + full_rows, other_keys,
                    other_keys + delta_rows, merged_keys);
    }
    // Swap merged columns
    p_->cols.swap(merged_cols);
    merged_cols.reset();
  }

  // Check for errors after merge_by_key
  err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    throw std::runtime_error("merge: CUDA error after merge_by_key. Error: " +
                             std::string(GPU_GET_ERROR_STRING(err)));
  }

  // 3. Update Root Unique Values
  {
    nvtx3::scoped_range r("unique_values_union");
    const auto old_unique_size = p_->root_unique_values.size();
    const auto new_unique_size = other.p_->root_unique_values.size();

    if (old_unique_size == 0 && new_unique_size == 0) {
      // Nothing to do
    } else if (old_unique_size == 0) {
      // Full was empty — just take delta's root values
      p_->root_unique_values.resize(new_unique_size);
      GPU_MEMCPY(p_->root_unique_values.data(), other.p_->root_unique_values.data(),
                 new_unique_size * sizeof(ValueType), GPU_DEVICE_TO_DEVICE);
    } else if (new_unique_size == 0) {
      // Delta was empty — nothing to merge
    } else {
      // Large inputs: use thrust::set_union
      DeviceArray<ValueType> merged_unique;
      merged_unique.resize(max_unique);
      auto end_it =
          thrust::set_union(rmm::exec_policy{}, p_->root_unique_values.begin(),
                            p_->root_unique_values.end(), other.p_->root_unique_values.begin(),
                            other.p_->root_unique_values.end(), merged_unique.begin());
      auto final_unique_size = end_it - merged_unique.begin();
      p_->root_unique_values.swap(merged_unique);
      p_->root_unique_values.resize(final_unique_size);
    }
  }

  // Note: rows_processed_ tracks cumulative input size for dirty checking
  // The actual index size after merge is p_->cols.num_rows()
  rows_processed_ += other.rows_processed_;
}

// -----------------------------------------------------------------------------
// MergeMatchOp for Set Difference with In-Place Update
// -----------------------------------------------------------------------------

template <Semiring SR>
struct MergeMatchOp {
  using ValueType = typename SR::value_type;
  bool emit_matches;

  __device__ __host__ explicit MergeMatchOp(bool emit) : emit_matches(emit) {}

  __device__ ::cuda::std::pair<bool, ValueType> operator()(const auto& /*key*/, ValueType val_new,
                                                           ValueType val_full) {
    ValueType merged = SR::add(val_new, val_full);
    // Explicitly update full via side-effect if value changes
    if (merged != val_full) {
      return {true, merged};
    }
    return {emit_matches, merged};
  }
};

// =============================================================================
// Binary-search-based set difference: O(|NEW| * log|FULL|) instead of O(|NEW| + |FULL|)
// Used when |NEW| << |FULL| (e.g., late fixpoint iterations where delta is small
// but FULL has grown large). Each thread binary-searches one NEW element in FULL.
// =============================================================================

/// @brief Lexicographic comparison of a tuple at position `idx` in column-major layout
///        against values from `new_cols` at position `new_idx`.
///        Returns -1, 0, or 1 for less, equal, greater.
template <typename ValueType, std::size_t Arity>
__device__ int lex_compare_cols(const ValueType* full_data, uint32_t full_stride, uint32_t full_idx,
                                const ValueType* new_data, uint32_t new_stride, uint32_t new_idx) {
#pragma unroll
  for (std::size_t col = 0; col < Arity; ++col) {
    ValueType f = full_data[col * full_stride + full_idx];
    ValueType n = new_data[col * new_stride + new_idx];
    if (f < n)
      return -1;
    if (f > n)
      return 1;
  }
  return 0;
}

/// @brief Binary search for a NEW tuple in FULL. Returns true if found (exact match).
template <typename ValueType, std::size_t Arity>
__device__ bool binary_search_in_full(const ValueType* full_data, uint32_t full_stride,
                                      uint32_t full_size, const ValueType* new_data,
                                      uint32_t new_stride, uint32_t new_idx) {
  uint32_t lo = 0, hi = full_size;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    int cmp = lex_compare_cols<ValueType, Arity>(full_data, full_stride, mid, new_data, new_stride,
                                                 new_idx);
    if (cmp < 0) {
      lo = mid + 1;
    } else if (cmp > 0) {
      hi = mid;
    } else {
      return true;  // exact match found
    }
  }
  return false;
}

/// @brief Kernel: for each NEW element, probe into FULL via binary search.
///        Writes 1 to flags[i] if NEW[i] is NOT in FULL (i.e., it's a delta tuple).
template <typename ValueType, std::size_t Arity>
__global__ void set_diff_probe_kernel(const ValueType* __restrict__ new_data, uint32_t new_stride,
                                      uint32_t new_size, const ValueType* __restrict__ full_data,
                                      uint32_t full_stride, uint32_t full_size,
                                      uint32_t* __restrict__ flags) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= new_size)
    return;
  flags[tid] = binary_search_in_full<ValueType, Arity>(full_data, full_stride, full_size, new_data,
                                                       new_stride, tid)
                   ? 0u
                   : 1u;
}

/// @brief Fused anti-join kernel: probe NEW against two sorted arrays (FULL and HEAD),
///        and directly compact survivors using an atomic output counter.
///        Eliminates separate reduce + copy_if + unique_copy passes.
/// @note Output is NOT sorted — caller must sort if needed (for small deltas, cheap).
template <typename ValueType, std::size_t Arity>
__global__ void set_diff_probe_compact_dual_kernel(
    const ValueType* __restrict__ new_data, uint32_t new_stride, uint32_t new_size,
    const ValueType* __restrict__ full_data, uint32_t full_stride, uint32_t full_size,
    const ValueType* __restrict__ head_data, uint32_t head_stride, uint32_t head_size,
    ValueType* __restrict__ out_data, uint32_t out_stride, uint32_t* __restrict__ out_count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= new_size)
    return;

  // Probe FULL
  if (binary_search_in_full<ValueType, Arity>(full_data, full_stride, full_size, new_data,
                                              new_stride, tid))
    return;

  // Probe HEAD
  if (head_size > 0 && binary_search_in_full<ValueType, Arity>(head_data, head_stride, head_size,
                                                               new_data, new_stride, tid))
    return;

  // Survivor: atomically claim output slot and scatter
  uint32_t pos = atomicAdd(out_count, 1U);
#pragma unroll
  for (std::size_t col = 0; col < Arity; ++col) {
    out_data[(col * out_stride) + pos] = new_data[(col * new_stride) + tid];
  }
}

/// @brief Threshold ratio: use binary-search set_diff when |FULL| / |NEW| > this value.
///        At ratio 8, binary search does 8*N*log2(F) comparisons vs merge-scan's N+F reads.
///        For N=300K, F=250M: binary = 300K*28 = 8.4M ops, merge = 250M reads. ~30x better.
static constexpr double kSearchSetDiffRatio = 8.0;

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::set_difference_update(
    DeviceSortedArrayIndex& full_idx, DeviceSortedArrayIndex& delta_idx) {
  if (empty()) {
    // If Newt is empty, Delta is empty. Full is unchanged.
    delta_idx.clear();
    return;
  }

  // Ensure delta_idx is initialized
  if (!delta_idx.p_) {
    delta_idx.p_ = std::make_unique<Impl>();
  } else {
    delta_idx.p_->clear();
  }

  if (full_idx.empty()) {
    // If Full is empty, Delta is Newt. Full remains empty (caller must merge Delta into Full
    // later). Just swap new and delta.
    std::swap(delta_idx, *this);
    return;
  }

  // --- Adaptive: use binary-search probe when |FULL| >> |NEW| ---
  const std::size_t new_size = p_->cols.num_rows();
  const std::size_t full_size = full_idx.p_->cols.num_rows();

  if constexpr (!has_provenance_v<SR>) {
    if (full_size > kSearchSetDiffRatio * new_size) {
      // Fused anti-join: single kernel probes FULL + atomic compacts survivors.
      // Replaces the old 4-step pipeline (probe → reduce → copy_if → unique_copy).
      nvtxRangePushA(":set_diff_search");

      // Allocate output (max = new_size) + atomic counter
      NDDeviceArray<ValueType, arity> out_cols;
      out_cols.resize(new_size);
      DeviceArray<uint32_t> d_count(1);
      cudaMemsetAsync(d_count.data(), 0, sizeof(uint32_t));

      // Single kernel: probe FULL (head_size=0 means no HEAD), compact via atomic
      {
        constexpr uint32_t kBlock = 256;
        uint32_t grid = (static_cast<uint32_t>(new_size) + kBlock - 1) / kBlock;
        set_diff_probe_compact_dual_kernel<ValueType, arity><<<grid, kBlock>>>(
            p_->cols.data(), static_cast<uint32_t>(p_->cols.stride()),
            static_cast<uint32_t>(new_size), full_idx.p_->cols.data(),
            static_cast<uint32_t>(full_idx.p_->cols.stride()), static_cast<uint32_t>(full_size),
            nullptr, 0, 0,  // no HEAD segment
            out_cols.data(), static_cast<uint32_t>(out_cols.stride()), d_count.data());
      }

      // Read back count
      uint32_t diff_sz = 0;
      cudaMemcpy(&diff_sz, d_count.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);

      if (diff_sz > 0) {
        out_cols.resize(diff_sz);

        // Sort the atomic-scattered output (cheap for small deltas)
        auto make_key_iter = [](auto& arr) {
          return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
          }
          (std::make_index_sequence<arity>());
        };
        auto keys = make_key_iter(out_cols);
        thrust::sort(rmm::exec_policy{}, keys, keys + diff_sz);

        delta_idx.p_->cols.swap(out_cols);

        // Compute root unique values for delta
        delta_idx.p_->root_unique_values.resize(diff_sz);
        auto end_unique =
            thrust::unique_copy(rmm::exec_policy{}, delta_idx.p_->cols.template column_ptr<0>(),
                                delta_idx.p_->cols.template column_ptr<0>() + diff_sz,
                                delta_idx.p_->root_unique_values.begin());
        delta_idx.p_->root_unique_values.resize(end_unique -
                                                delta_idx.p_->root_unique_values.begin());
      }

      delta_idx.p_->index_arity = p_->index_arity;
      delta_idx.rows_processed_ = diff_sz;

      nvtxRangePop();
      return;
    }
  }

  // --- Fallback: merge-based thrust::set_difference (original path) ---

  // Setup Iterators
  auto make_key_iter = [](const auto& arr) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
    }
    (std::make_index_sequence<arity>());
  };

  auto make_mutable_key_iter = [](auto& arr) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
    }
    (std::make_index_sequence<arity>());
  };

  auto newt_keys = make_key_iter(p_->cols);
  auto full_keys = make_key_iter(full_idx.p_->cols);

  // Allocate Delta buffers (max size = newt size)
  NDDeviceArray<ValueType, arity> delta_cols;
  delta_cols.resize(p_->cols.num_rows());

  auto delta_keys_out = make_mutable_key_iter(delta_cols);

  std::size_t diff_size = 0;

  if constexpr (has_provenance_v<SR>) {
    DeviceArray<semiring_value_t<SR>> delta_prov;
    delta_prov.resize(p_->cols.num_rows());

    // Perform Custom Set Difference with provenance
    MergeMatchOp<SR> op(false);  // Emit only if changed (diff or merge-update)

    // Use rmm::exec_policy() for proper memory allocation through RMM pool
    auto result_end = thrust::cuda_cub::set_difference_by_key_with_match_custom(
        rmm::exec_policy{}, newt_keys, newt_keys + p_->cols.num_rows(), full_keys,
        full_keys + full_idx.p_->cols.num_rows(), p_->provenance.begin(),
        full_idx.p_->provenance.begin(),  // Mutable Full Values for in-place update!
        delta_keys_out, delta_prov.begin(),
        thrust::less<typename decltype(newt_keys)::value_type>(),  // Zip comparison
        op);

    diff_size = result_end.first - delta_keys_out;
    delta_cols.resize(diff_size);
    delta_prov.resize(diff_size);

    delta_idx.p_->cols.swap(delta_cols);
    delta_idx.p_->provenance.swap(delta_prov);
  } else {
    // No provenance - use simple set_difference (keys only)
    auto result_end =
        thrust::set_difference(rmm::exec_policy{}, newt_keys, newt_keys + p_->cols.num_rows(),
                               full_keys, full_keys + full_idx.p_->cols.num_rows(), delta_keys_out);

    diff_size = result_end - delta_keys_out;
    delta_cols.resize(diff_size);

    delta_idx.p_->cols.swap(delta_cols);
  }

  delta_idx.p_->index_arity = p_->index_arity;
  delta_idx.rows_processed_ = diff_size;  // approximate

  // Compute unique values for Delta (Step 4 of TODO)
  // Since Delta is sorted, unique_copy on first column (keys are Zip)
  // Wait, root_unique uses the *first* column of the index (if we are building a trie index)
  // Actually deduplicate_aggregate_and_unique does this. But we already have unique keys.
  // Just gathering the first column root values should be enough.
  // But `root_unique_values` is typically used for partitioning.
  // Let's perform a simple `unique_copy` on the first column of output.
  {
    if (diff_size > 0) {
      delta_idx.p_->root_unique_values.resize(diff_size);
      // Use rmm::exec_policy() for proper memory allocation through RMM pool
      auto end_unique =
          thrust::unique_copy(rmm::exec_policy{}, delta_idx.p_->cols.template column_ptr<0>(),
                              delta_idx.p_->cols.template column_ptr<0>() + diff_size,
                              delta_idx.p_->root_unique_values.begin());
      delta_idx.p_->root_unique_values.resize(end_unique -
                                              delta_idx.p_->root_unique_values.begin());
    }
  }
}

/// @brief Fused set difference against two sorted arrays: (this - full_idx - head_idx) → delta_idx
/// @details Single kernel probes both FULL and HEAD, atomic-compacts survivors.
///          Output is then sorted (cheap for small deltas). Only for NoProvenance.
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::set_difference_update_dual(
    DeviceSortedArrayIndex& full_idx, DeviceSortedArrayIndex& head_idx,
    DeviceSortedArrayIndex& delta_idx) {
  if (empty()) {
    delta_idx.clear();
    return;
  }

  if (!delta_idx.p_) {
    delta_idx.p_ = std::make_unique<Impl>();
  } else {
    delta_idx.p_->clear();
  }

  if (full_idx.empty() && head_idx.empty()) {
    std::swap(delta_idx, *this);
    return;
  }

  static_assert(!has_provenance_v<SR>, "set_difference_update_dual only supports NoProvenance");

  const auto new_size = static_cast<uint32_t>(p_->cols.num_rows());
  const auto full_size = static_cast<uint32_t>(full_idx.empty() ? 0 : full_idx.p_->cols.num_rows());
  const auto head_size = static_cast<uint32_t>(head_idx.empty() ? 0 : head_idx.p_->cols.num_rows());

  nvtxRangePushA(":set_diff_dual");

  // Allocate output (max = new_size) + atomic counter
  NDDeviceArray<ValueType, arity> out_cols;
  out_cols.resize(new_size);
  DeviceArray<uint32_t> d_count(1);
  cudaMemsetAsync(d_count.data(), 0, sizeof(uint32_t));

  // Single fused kernel: probe FULL + HEAD, compact survivors via atomic
  {
    constexpr uint32_t kBlock = 256;
    uint32_t grid = (new_size + kBlock - 1) / kBlock;
    set_diff_probe_compact_dual_kernel<ValueType, arity><<<grid, kBlock>>>(
        p_->cols.data(), static_cast<uint32_t>(p_->cols.stride()), new_size,
        full_size > 0 ? full_idx.p_->cols.data() : nullptr,
        full_size > 0 ? static_cast<uint32_t>(full_idx.p_->cols.stride()) : 0, full_size,
        head_size > 0 ? head_idx.p_->cols.data() : nullptr,
        head_size > 0 ? static_cast<uint32_t>(head_idx.p_->cols.stride()) : 0, head_size,
        out_cols.data(), static_cast<uint32_t>(out_cols.stride()), d_count.data());
  }

  // Read back count
  uint32_t diff_sz = 0;
  cudaMemcpy(&diff_sz, d_count.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (diff_sz == 0) {
    delta_idx.p_->clear();
    nvtxRangePop();
    return;
  }

  out_cols.resize(diff_sz);

  // Output is unsorted (atomic scatter) — sort it.
  // For small deltas (which is the common case in the long tail), this is very cheap.
  // Use thrust::sort with zip iterator on the column-major layout.
  {
    auto make_key_iter = [](auto& arr) {
      return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return thrust::make_zip_iterator(thrust::make_tuple(arr.template column_ptr<I>()...));
      }
      (std::make_index_sequence<arity>());
    };
    auto keys = make_key_iter(out_cols);
    thrust::sort(rmm::exec_policy{}, keys, keys + diff_sz);
  }

  delta_idx.p_->cols.swap(out_cols);
  delta_idx.p_->index_arity = p_->index_arity;
  delta_idx.rows_processed_ = diff_sz;

  // Compute root unique values for delta
  if (diff_sz > 0) {
    delta_idx.p_->root_unique_values.resize(diff_sz);
    auto end_unique =
        thrust::unique_copy(rmm::exec_policy{}, delta_idx.p_->cols.template column_ptr<0>(),
                            delta_idx.p_->cols.template column_ptr<0>() + diff_sz,
                            delta_idx.p_->root_unique_values.begin());
    delta_idx.p_->root_unique_values.resize(end_unique - delta_idx.p_->root_unique_values.begin());
  }

  nvtxRangePop();
}

/// @brief Find max value across array (device, single-thread)
template <typename T, std::size_t N>
__device__ T array_max(const T (&arr)[N]) {
  T result = arr[0];
#pragma unroll
  for (std::size_t i = 1; i < N; ++i) {
    result = (arr[i] > result) ? arr[i] : result;
  }
  return result;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void DeviceSortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::reconstruct_to_relation(
    const IndexSpec& spec, NDDeviceArray<ValueType, arity>& output_cols,
    DeviceArray<semiring_value_t<SR>>& output_prov) const {
  if (empty() || !p_) {
    return;
  }

  const std::size_t num_rows = p_->cols.num_rows();

  // Resize output arrays
  output_cols.resize(num_rows);
  output_prov.resize(num_rows);

  // Build inverse mapping: for each relation column, find its position in spec.cols
  // col_map[rel_col] = position in spec.cols where rel_col appears
  std::vector<std::size_t> col_map(arity);
  for (std::size_t rel_col = 0; rel_col < arity; ++rel_col) {
    // Find where rel_col appears in spec.cols
    auto it = std::ranges::find(spec.cols, rel_col);
    if (it == spec.cols.end()) {
      throw std::runtime_error("reconstruct_to_relation: IndexSpec must contain all columns");
    }
    col_map[rel_col] = std::distance(spec.cols.begin(), it);
  }

  // Copy columns with reordering: output_cols[rel_col] = p_->cols[col_map[rel_col]]
  for (std::size_t rel_col = 0; rel_col < arity; ++rel_col) {
    const std::size_t idx_col = col_map[rel_col];
    thrust::copy(rmm::exec_policy{}, p_->cols.column_ptr(idx_col),
                 p_->cols.column_ptr(idx_col) + num_rows, output_cols.column_ptr(rel_col));
  }

  // Copy provenance (only if semiring has provenance)
  if constexpr (has_provenance_v<SR>) {
    thrust::copy(rmm::exec_policy{}, p_->provenance.begin(), p_->provenance.end(),
                 output_prov.begin());
  }
}

template <typename ValueType, typename RowIdType, std::size_t N, typename Group, typename Accessor>
class IntersectView {
 public:
  struct Sentinel {};

  struct Iterator {
   private:
    // Store pointers to parent's arrays instead of copying
    const Accessor* accessors_;
    const RowIdType* end_;
    RowIdType pos_[N];
    bool exhausted_;
    Group tile_;

   public:
    __device__ Iterator(const Accessor* accessors, const RowIdType* sizes, Group tile)
        : accessors_(accessors), end_(sizes), exhausted_(false), tile_(tile) {

#pragma unroll
      for (std::size_t i = 0; i < N; ++i) {
        pos_[i] = 0;

        if (pos_[i] >= end_[i]) {
          exhausted_ = true;
        }
      }

      if (!exhausted_) {
        advance_to_intersection();
      }
    }

    __device__ bool valid() const {
      return !exhausted_;
    }

    __device__ ValueType value() const {
      return accessors_[0][pos_[0]];
    }

    __device__ void next() {
      // Sorted-array spans have duplicate values at non-leaf depth levels.
      // After ++pos, all sources may still be at the previous value, and
      // advance_to_intersection would re-report it (it only checks all_equal).
      // Fix: use upper_bound on source 0 to guarantee it moves past the
      // current value. advance_to_intersection then seeks the others forward.
      RowIdType remaining = end_[0] - pos_[0];
      RowIdType offset = group_exp_upper_bound<ValueType, RowIdType>(accessors_[0] + pos_[0],
                                                                     remaining, value(), tile_);
      pos_[0] += offset;
      if (pos_[0] >= end_[0]) {
        exhausted_ = true;
        return;
      }
#pragma unroll
      for (std::size_t i = 1; i < N; ++i) {
        ++pos_[i];
        if (pos_[i] >= end_[i]) {
          exhausted_ = true;
          return;
        }
      }
      advance_to_intersection();
    }

    // for-range support on iterator

    __device__ Iterator operator++() {
      next();
      return *this;
    }

    __device__ Iterator operator++(int) {
      Iterator tmp = *this;
      next();
      return tmp;
    }

    __device__ bool operator==(const Sentinel&) const {
      return exhausted_;
    }

    __device__ bool operator!=(const Sentinel&) const {
      return !exhausted_;
    }

    __device__ ValueType operator*() const {
      return value();
    }

    __device__ ValueType operator->() const {
      return value();
    }

    /// @brief Get positions for each source (for avoiding binary search in handle update)
    /// @details Returns the current position in each source's values span.
    ///          For source handle i, the global position is handle.begin_ + positions()[i]
    __device__ auto positions() const {
      std::array<RowIdType, N> result;
#pragma unroll
      for (std::size_t i = 0; i < N; ++i) {
        result[i] = pos_[i];
      }
      return result;
    }

   private:
    __device__ void advance_to_intersection() {
      uint32_t iter_count = 0;
      while (true) {
        // Find max value across all ranges
        ValueType vals[N];
#pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
          vals[i] = accessors_[i][pos_[i]];
        }

        ValueType max_val = array_max(vals);

        // Seek all ranges to >= max_val using warp-cooperative search
        bool all_equal = true;

#pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
          if (vals[i] < max_val) {
            all_equal = false;

            // Group-cooperative exponential lower bound on remaining range
            // Uses linear scan for ≤256 elements, exponential for larger
            RowIdType remaining = end_[i] - pos_[i];
            RowIdType offset = group_exp_lower_bound<ValueType, RowIdType>(
                accessors_[i] + pos_[i], remaining, max_val, tile_);
            pos_[i] += offset;

            if (pos_[i] >= end_[i]) {
              exhausted_ = true;
              return;
            }
          }
        }

        if (all_equal) {
          return;  // Found intersection
        }

        iter_count++;
        // if (iter_count > 1000) {  // Safety check to avoid infinite loop
        //   exhausted_ = true;
        //   return;
        // }
      }
    }
  };

 private:
  // do we need this or better to have only one data ptr compute from stride?
  Accessor accessors_[N];
  RowIdType sizes_[N];
  Group tile_;

 public:
  __device__ IntersectView(const Accessor* accessors, const RowIdType* sizes, Group tile)
      : tile_(tile) {

#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      accessors_[i] = accessors[i];
      sizes_[i] = sizes[i];
    }
  }

  __device__ Iterator begin() const {
    return Iterator(accessors_, sizes_, tile_);
  }

  __device__ Sentinel end() const {
    return Sentinel{};
  }

  __device__ RowIdType count() const {
    RowIdType cnt = 0;
    for (auto it = begin(); it.valid(); it.next()) {
      ++cnt;
    }
    return cnt;
  }
};

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
template <typename Group, typename... Spans>
__device__ auto DeviceSortedArrayIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::intersect(
    Group tile, Spans... spans) {

  constexpr std::size_t N = sizeof...(Spans);
  using ValueType = ValueTypeParam;
  using FirstSpan = std::tuple_element_t<0, std::tuple<Spans...>>;
  using Accessor = decltype((*static_cast<FirstSpan*>(nullptr)).begin());

  Accessor accessors[N] = {(spans.begin())...};  // Use .begin() (iterator/accessor)
  RowIdType sizes[N] = {static_cast<RowIdType>(spans.size())...};

  return IntersectView<ValueType, RowIdType, N, Group, Accessor>(accessors, sizes, tile);
}

}  // namespace SRDatalog::GPU
