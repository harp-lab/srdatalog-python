
#pragma once

#include "device_lsm_index.h"

// Implementation details for DeviceLsmIndex

namespace SRDatalog::GPU {

//
// LsmNodeHandle Implementation
//

template <Semiring SR, typename ValueTypeParam, typename RowIdType>
template <typename Group>
__device__ LsmNodeHandle<SR, ValueTypeParam, RowIdType>
LsmNodeHandle<SR, ValueTypeParam, RowIdType>::prefix(ValueTypeParam key, Group tile,
                                                     const View& view) const {
  if (!valid() || is_leaf(view)) {
    return LsmNodeHandle::invalid();
  }

  // Binary Search on Indirect Values
  // Range [begin_, end_)
  // Comparisons: view.get_value(depth_, mid) vs key

  RowIdType left = begin_;
  RowIdType right = end_;
  RowIdType count = right - left;

  while (count > 0) {
    RowIdType step = count / 2;
    RowIdType mid = left + step;

    // Indirect Access
    ValueTypeParam mid_val = view.get_value(depth_, mid);

    if (mid_val < key) {
      left = mid + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  RowIdType lower_bound = left;

  // Find upper bound
  // We can assume we just need to return the range where value == key
  // Upper search:
  left = lower_bound;
  right = end_;
  count = right - left;

  while (count > 0) {
    RowIdType step = count / 2;
    RowIdType mid = left + step;

    ValueTypeParam mid_val = view.get_value(depth_, mid);

    if (mid_val <= key) {
      left = mid + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  RowIdType upper_bound = left;

  if (lower_bound >= upper_bound) {
    return LsmNodeHandle::invalid();
  }

  return LsmNodeHandle(lower_bound, upper_bound, depth_ + 1);
}

//
// DeviceLsmIndex Implementation
//

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
void DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::build_from_encoded_device(
    const IndexSpec& spec, NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
    DeviceArray<semiring_value_t<SR>>& provenance) {

  clear();

  std::size_t num_tuples = encoded_cols.num_rows();
  if (num_tuples == 0)
    return;

  num_rows_ = num_tuples;
  stride_ = num_tuples;

  // Store pointers for View construction
  data_ptr_ = encoded_cols.data();
  prov_ptr_ = provenance.data();

  // Resize indices array
  sorted_indices_.resize(num_tuples);

  // Initialize with 0..N-1
  thrust::sequence(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end());

  // Indirect Sort
  auto cols_ptr = encoded_cols.data();
  auto stride = stride_;

  auto comp = [cols_ptr, stride, num_cols = arity] __host__ __device__(RowIdType a, RowIdType b) {
    for (int i = 0; i < num_cols; ++i) {
      ValueTypeParam va = cols_ptr[i * stride + a];
      ValueTypeParam vb = cols_ptr[i * stride + b];
      if (va < vb)
        return true;
      if (va > vb)
        return false;
    }
    return false;
  };

  thrust::sort(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end(), comp);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
auto DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::root() const -> NodeHandle {
  if (empty())
    return NodeHandle::invalid();
  return NodeHandle(0, static_cast<RowIdType>(num_rows_), 0);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
auto DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::view() const -> NodeView {
  NodeView v;
  v.sorted_indices_ = sorted_indices_.data();
  v.col_data_ = data_ptr_;
  v.provenance_ptr_ = prov_ptr_;
  v.arity_ = static_cast<uint32_t>(arity);
  v.stride_ = static_cast<uint32_t>(stride_);
  v.num_rows_ = static_cast<uint32_t>(num_rows_);
  return v;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
void DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::clear() noexcept {
  sorted_indices_.clear();
  num_rows_ = 0;
  data_ptr_ = nullptr;
  prov_ptr_ = nullptr;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
std::size_t DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::bytes_used() const {
  return sorted_indices_.size() * sizeof(RowIdType);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
void DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::compact(
    NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
    DeviceArray<semiring_value_t<SR>>& provenance) {
  if (empty())
    return;

  // Compaction logic (same as before)
  {
    DeviceArray<semiring_value_t<SR>> new_prov(provenance.size());
    thrust::gather(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end(),
                   provenance.begin(), new_prov.begin());
    provenance = std::move(new_prov);
    prov_ptr_ = provenance.data();
  }

  {
    NDDeviceArray<ValueTypeParam, arity> new_cols(encoded_cols.num_rows());
    auto raw_src = encoded_cols.data();
    auto raw_dst = new_cols.data();
    uint32_t stride = static_cast<uint32_t>(encoded_cols.num_rows());

    for (size_t i = 0; i < arity; ++i) {
      const ValueTypeParam* src_col = raw_src + i * stride;
      ValueTypeParam* dst_col = raw_dst + i * stride;

      thrust::gather(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end(),
                     thrust::device_ptr<const ValueTypeParam>(src_col),
                     thrust::device_ptr<ValueTypeParam>(dst_col));
    }
    encoded_cols = std::move(new_cols);
    data_ptr_ = encoded_cols.data();
  }

  thrust::sequence(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end());
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
void DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::bind_data(
    NDDeviceArray<ValueTypeParam, arity>& encoded_cols,
    DeviceArray<semiring_value_t<SR>>& provenance) {
  // Just update the pointers
  data_ptr_ = encoded_cols.data();
  prov_ptr_ = provenance.data();
  stride_ = encoded_cols.num_rows();
  // num_rows_ (of the index) might be different from stride if data array is larger than index
  // coverage? Usually stride == capacity or current size. We assume stride matches what the getters
  // expect.
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam, typename RowIdType>
void DeviceLsmIndex<SR, AttrTuple, ValueTypeParam, RowIdType>::merge(const DeviceLsmIndex& other,
                                                                     std::size_t row_id_offset) {
  if (other.empty())
    return;
  if (empty()) {
    // Copy other's indices, shift them by offset
    sorted_indices_.resize(other.size());
    thrust::transform(rmm::exec_policy{}, other.sorted_indices_.begin(),
                      other.sorted_indices_.end(), sorted_indices_.begin(),
                      [off = row_id_offset] __device__(RowIdType x) { return x + off; });
    num_rows_ = other.size();
    // Assume data_ptr_ is already set correctly by caller
    return;
  }

  // Linear Merge Implementation
  // Complexity: O(N+M)

  std::size_t new_size = num_rows_ + other.size();
  DeviceArray<RowIdType> new_indices(new_size);

  // Comparator
  auto cols_ptr = data_ptr_;
  auto stride = stride_;
  auto comp = [cols_ptr, stride, num_cols = arity] __host__ __device__(RowIdType a, RowIdType b) {
    for (int i = 0; i < num_cols; ++i) {
      ValueTypeParam va = cols_ptr[i * stride + a];
      ValueTypeParam vb = cols_ptr[i * stride + b];
      if (va < vb)
        return true;
      if (va > vb)
        return false;
    }
    return false;
  };

  // Transform iterator for other's indices to add offset
  auto transform_op = [off = static_cast<RowIdType>(row_id_offset)] __host__ __device__(
                          RowIdType x) { return x + off; };

  auto other_begin = thrust::make_transform_iterator(other.sorted_indices_.begin(), transform_op);
  auto other_end = thrust::make_transform_iterator(other.sorted_indices_.end(), transform_op);

  // Merge
  // Use 7-arg version with explicit policy and comparator
  // 1. Merge sorted indices
  // Use thrust::merge with rmm::exec_policy
  thrust::merge(rmm::exec_policy{}, sorted_indices_.begin(), sorted_indices_.end(), other_begin,
                other_end, new_indices.begin(), comp);

  // Swap
  sorted_indices_ = std::move(new_indices);
  num_rows_ = new_size;
}

}  // namespace SRDatalog::GPU
