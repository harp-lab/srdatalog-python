#pragma once

#include "search.h"
#include "sorted_array_index.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <ranges>
#include <vector>

#include "leapfrog_view.h"  // For LeapfrogView
#include "logging.h"
#include "skip.h"
#include "soa_merge.h"
namespace SRDatalog {

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
struct SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::Impl {
  using VecSizeT = Vector<ValueType>;
  using VecRowId = Vector<RowIdType>;

  memory_resource* resource_;

  // The materialized columns defined in IndexSpec.
  // stored_cols[0] corresponds to IndexSpec::cols[0], etc.
  Vector<VecSizeT> stored_cols;

  // The row IDs permuted to match the sort order of stored_cols.
  VecRowId postings;

  std::size_t index_arity{0};

  explicit Impl(memory_resource* r) : resource_(r), stored_cols(r), postings(r) {}

  void clear() {
    stored_cols.clear();
    postings.clear();
  }

  [[nodiscard]] std::size_t bytes_used() const {
    std::size_t b = sizeof(*this) + (postings.capacity() * sizeof(typename VecRowId::value_type));
    for (const auto& c : stored_cols) {
      b += sizeof(c) + (c.capacity() * sizeof(ValueType));
    }
    return b;
  }
};

// Thread-local merge buffers shared across all indexes in the same thread
template <typename ValueType, typename RowIdType>
struct ThreadLocalMergeBuffers {
  using VecSizeT = Vector<ValueType>;
  using VecRowId = Vector<RowIdType>;

  Vector<VecSizeT> merge_cols;
  VecRowId merge_postings;
  memory_resource* resource_;

  ThreadLocalMergeBuffers()
      : merge_cols(default_memory_resource()), merge_postings(default_memory_resource()),
        resource_(default_memory_resource()) {}

  // Ensure merge buffers are large enough, using exponential growth
  void ensure_buffers(memory_resource* res, std::size_t num_cols, std::size_t total_size) {
    // Update resource if needed
    if (resource_ != res) {
      resource_ = res;
      // Recreate buffers with new resource if resource changed
      // (This is rare, so we don't optimize for it)
      merge_cols = Vector<VecSizeT>(res);
      merge_postings = VecRowId(res);
    }

    // Resize merge_cols if needed
    if (merge_cols.size() < num_cols) {
      merge_cols.resize(num_cols);
      for (std::size_t c = 0; c < num_cols; ++c) {
        if (merge_cols[c].size() == 0) {
          merge_cols[c] = VecSizeT(resource_);
        }
      }
    }

    // Resize each column buffer with exponential growth
    // Use uninitialized resize since we'll immediately overwrite with merge data
    for (std::size_t c = 0; c < num_cols; ++c) {
      auto& col = merge_cols[c];
      if (col.size() < total_size) {
        std::size_t new_size = (col.size() == 0)
                                   ? std::max(static_cast<std::size_t>(64 * 1024), total_size)
                                   : std::max(total_size, col.size() * 2);
        col.resize(new_size, default_init);
      }
    }

    // Resize postings buffer with exponential growth
    // Use uninitialized resize since we'll immediately overwrite with merge data
    if (merge_postings.size() < total_size) {
      std::size_t new_size = (merge_postings.size() == 0)
                                 ? std::max(static_cast<std::size_t>(64 * 1024), total_size)
                                 : std::max(total_size, merge_postings.size() * 2);
      merge_postings.resize(new_size, default_init);
    }
  }
};

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::SortedArrayIndex(
    memory_resource* resource)
    : p_(std::make_unique<Impl>((resource != nullptr) ? resource : default_memory_resource())) {}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::~SortedArrayIndex() = default;

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline void SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::clear() noexcept {
  if (p_)
    p_->clear();
  size_ = 0;
  rows_processed_ = 0;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::bytes_used() const {
  // Return 0 if index is empty (no data stored)
  if (!p_ || size_ == 0)
    return 0;
  return p_->bytes_used();
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::root() const -> NodeHandle {
  if (!p_ || size_ == 0)
    return NodeHandle{};
  return NodeHandle{p_.get(), 0, size_, 0};
}

namespace detail {
template <size_t K, typename ValueType>
void dispatch_stable_sort(size_t, std::span<const int>, hwy_lexsort::SortContext&,
                          const Vector<Vector<ValueType>>&) {}

template <size_t K, typename ValueType>
void dispatch_stable_sort_impl(size_t n, std::span<const int> order, hwy_lexsort::SortContext& ctx,
                               Vector<Vector<ValueType>>& cols) {
  if (cols.size() == K) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      hwy_lexsort::stable_lex_sort(n, order.data(), order.size(), ctx, true, cols[Is].data()...);
    }(std::make_index_sequence<K>{});
  } else {
    if constexpr (K > 1) {
      dispatch_stable_sort_impl<K - 1, ValueType>(n, order, ctx, cols);
    }
  }
}

}  // namespace detail
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline void SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_encoded(
    const IndexSpec& spec, const std::array<std::span<const ValueType>, arity>& encoded_rows,
    memory_resource* resource) {

  std::size_t num_rows = encoded_rows[0].size();
  if (num_rows == 0)
    return;

  // Determine resource
  memory_resource* res = nullptr;
  if (resource != nullptr) {
    res = resource;
  } else if (p_ != nullptr) {
    res = p_->resource_;
  } else {
    res = default_memory_resource();
  }

  // Reset or Init Pimpl
  if (!p_ || p_->resource_ != res)
    p_ = std::make_unique<Impl>(res);
  else {
    p_->clear();
  }

  p_->index_arity = spec.cols.size();
  size_ = num_rows;
  rows_processed_ = num_rows;  // Track rows processed before deduplication

  p_->stored_cols.clear();
  p_->stored_cols.reserve(spec.cols.size());

  for (int col_idx : spec.cols) {
    typename Impl::VecSizeT vec(res);
    vec.resize(num_rows);
    // static assert value type in Impl::VecSizeT is same as ValueType
    std::memcpy(vec.data(), encoded_rows[col_idx].data(), num_rows * sizeof(ValueType));
    p_->stored_cols.push_back(std::move(vec));
  }

  Vector<int> sort_order(res);
  sort_order.reserve(spec.cols.size());
  for (size_t i = 0; i < spec.cols.size(); ++i) {
    sort_order.push_back(static_cast<int>(i));
  }

  // Use thread-local SortContext with the same memory resource as the index
  static thread_local hwy_lexsort::SortContext tls_ctx(res);
  hwy_lexsort::SortContext* ctx_ptr = &tls_ctx;

  detail::dispatch_stable_sort_impl<arity, ValueType>(num_rows, sort_order, *ctx_ptr,
                                                      p_->stored_cols);

  const uint32_t* sorted_indices = ctx_ptr->get_sorted_indices<uint32_t>(num_rows);
  p_->postings = Vector<RowId>(res);
  p_->postings.resize(num_rows);
  if constexpr (std::is_same_v<RowIdType, uint32_t>) {
    std::memcpy(p_->postings.data(), sorted_indices, num_rows * sizeof(RowId));
  } else {
    for (std::size_t i = 0; i < num_rows; ++i) {
      p_->postings[i] = static_cast<RowIdType>(sorted_indices[i]);
    }
  }

  if (num_rows > 1) {
    std::size_t write_idx = 0;
    std::size_t num_cols = p_->stored_cols.size();

    Vector<ValueType*> col_ptrs(num_cols);
    for (size_t c = 0; c < num_cols; ++c) {
      col_ptrs[c] = p_->stored_cols[c].data();
    }
    RowId* post_ptr = p_->postings.data();

    for (std::size_t read_idx = 1; read_idx < num_rows; ++read_idx) {
      bool equal = true;
      for (std::size_t c = 0; c < num_cols; ++c) {
        if (col_ptrs[c][read_idx] != col_ptrs[c][write_idx]) {
          equal = false;
          break;
        }
      }

      if (!equal) {
        ++write_idx;
        if (write_idx != read_idx) {
          for (std::size_t c = 0; c < num_cols; ++c) {
            col_ptrs[c][write_idx] = col_ptrs[c][read_idx];
          }
          post_ptr[write_idx] = post_ptr[read_idx];
        }
      }
    }

    std::size_t unique_count = write_idx + 1;
    if (unique_count < num_rows) {
      for (auto& col : p_->stored_cols)
        col.resize(unique_count);
      p_->postings.resize(unique_count);
      size_ = unique_count;
    }
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::merge(const SortedArrayIndex& other,
                                                                  std::size_t row_id_offset) {
  // 1. Fast path: If other is empty, do nothing.
  if (other.empty())
    return;

  // 2. Fast path: If this is empty, just clone other (handling offset).
  if (empty()) {
    clone_from(other);
    if (row_id_offset > 0 && p_) {
      for (auto& rid : p_->postings) {
        if constexpr (std::is_same_v<RowIdType, std::size_t>)
          rid += row_id_offset;
        else
          rid = static_cast<RowIdType>(static_cast<std::size_t>(rid) + row_id_offset);
      }
    }
    return;
  }

  const std::size_t old_size = size_;
  const std::size_t other_size = other.size_;

  // Use the generic SoA merge function
  size_ = soa_inplace_merge<ValueType, RowIdType>(p_->stored_cols, p_->postings,
                                                  other.p_->stored_cols, other.p_->postings,
                                                  old_size, other_size, row_id_offset);
  rows_processed_ += other.rows_processed_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::valid()
    const noexcept {
  return impl_ && begin_ < end_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::is_leaf()
    const noexcept {
  return valid() && depth_ == impl_->index_arity;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline uint32_t SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::depth()
    const noexcept {
  return depth_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::values() const {
  ValueRange raw_range;
  if (valid() && !is_leaf()) {
    const auto& col = impl_->stored_cols[depth_];
    raw_range = ValueRange{col.data() + begin_, end_ - begin_};
  } else {
    raw_range = ValueRange{static_cast<const ValueType*>(nullptr), 0};
  }

  // Return a deduplicated range view using skip_duplicates
  // This view iterates over unique values by skipping consecutive duplicates
  return SRDatalog::DedupRange<ValueType>{raw_range.data(), raw_range.data() + raw_range.size()};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::rows() const
    -> RowRange {
  if (!is_leaf())
    return RowRange{};
  return RowRange{impl_->postings.data() + begin_, end_ - begin_};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::prefix(
    ValueType next_key) const -> NodeHandle {
  if (!valid() || is_leaf())
    return NodeHandle{};

  const auto& col = impl_->stored_cols[depth_];
  const auto* data_ptr = col.data();
  std::size_t len = end_ - begin_;
  const auto* sub_start = data_ptr + begin_;

  // Use Highway-accelerated search for supported types
  std::size_t lb_idx;
  if constexpr (std::is_integral_v<ValueType> &&
                (sizeof(ValueType) == 4 || sizeof(ValueType) == 8)) {
    // For small arrays, branchless binary search is preferred over SIMD
    // to avoid setup overhead and ensure correctness on edge cases.
    if (len <= 16) {
      lb_idx = SRDatalog::search::branchless_lower_bound(sub_start, len, next_key);
    } else {
      lb_idx = SRDatalog::search::simd_lower_bound(sub_start, len, next_key);
    }
  } else {
    lb_idx = SRDatalog::search::branchless_lower_bound(sub_start, len, next_key);
  }
  // std::size_t lb_idx = SRDatalog::search::adaptive_lower_bound(sub_start, len, next_key);

  if (lb_idx == len || sub_start[lb_idx] != next_key) {
    return NodeHandle{};
  }

  // For upper bound, also use SIMD if available
  std::size_t ub_idx = len;
  if constexpr (std::is_integral_v<ValueType>) {
    if (next_key < std::numeric_limits<ValueType>::max()) {
      std::size_t rem_len = len - lb_idx;
      if constexpr (sizeof(ValueType) == 4 || sizeof(ValueType) == 8) {
        std::size_t rel_ub =
            SRDatalog::search::simd_lower_bound(sub_start + lb_idx, rem_len, next_key + 1);
        ub_idx = lb_idx + rel_ub;
      } else {
        std::size_t rel_ub =
            SRDatalog::search::branchless_lower_bound(sub_start + lb_idx, rem_len, next_key + 1);
        ub_idx = lb_idx + rel_ub;
      }
    }
  } else {
    for (std::size_t i = lb_idx; i < len; ++i) {
      if (sub_start[i] > next_key) {
        ub_idx = i;
        break;
      }
    }
  }

  return NodeHandle{impl_, begin_ + lb_idx, begin_ + ub_idx, depth_ + 1};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <std::size_t N>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::prefix(
    const EncodedKeyPrefix<N>& suffix) const -> NodeHandle {
  if (!valid())
    return NodeHandle{};
  NodeHandle cursor = *this;
  for (std::size_t i = 0; i < N; ++i) {
    cursor = cursor.prefix(static_cast<ValueType>(suffix[i]));
    if (!cursor.valid())
      break;
  }
  return cursor;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <std::size_t N>
inline auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::prefix_lookup(
    const EncodedKeyPrefix<N>& k) const -> NodeHandle {
  return root().prefix(k);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::contains_value(
    ValueType key) const noexcept {
  if (!valid() || is_leaf())
    return false;
  const auto& col = impl_->stored_cols[depth_];
  const auto* data_ptr = col.data();
  std::size_t len = end_ - begin_;
  std::size_t idx = SRDatalog::search::branchless_lower_bound(data_ptr + begin_, len, key);
  return idx < len && data_ptr[begin_ + idx] == key;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::degree()
    const noexcept {
  if (!valid())
    return 0;
  return end_ - begin_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::unique_count()
    const noexcept {
  if (!valid() || is_leaf())
    return 0;

  const auto& col = impl_->stored_cols[depth_];
  const auto* data_ptr = col.data();
  std::size_t len = end_ - begin_;

  if (len == 0)
    return 0;

  std::size_t count = 1;
  for (std::size_t i = 1; i < len; ++i) {
    if (data_ptr[begin_ + i] != data_ptr[begin_ + i - 1]) {
      ++count;
    }
  }
  return count;
}

// Clone
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::clone_from(
    const SortedArrayIndex& other, memory_resource* resource) {
  if (!other.p_) {
    clear();
    return;
  }

  memory_resource* res = nullptr;
  if (resource != nullptr) {
    res = resource;
  } else if (p_ != nullptr) {
    res = p_->resource_;
  } else {
    res = default_memory_resource();
  }
  p_ = std::make_unique<Impl>(res);
  p_->index_arity = other.p_->index_arity;
  size_ = other.size_;
  rows_processed_ = other.rows_processed_;

  p_->postings = Vector<RowId>(res);
  p_->postings.assign(other.p_->postings.begin(), other.p_->postings.end());

  p_->stored_cols.reserve(other.p_->stored_cols.size());
  for (const auto& col : other.p_->stored_cols) {
    typename Impl::VecSizeT new_col(res);
    new_col.assign(col.begin(), col.end());
    p_->stored_cols.push_back(std::move(new_col));
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
bool SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::verify_sorted() const {
  if (!p_ || size_ <= 1) {
    return true;
  }

  std::size_t num_cols = p_->stored_cols.size();
  if (num_cols == 0) {
    return true;
  }

  for (std::size_t i = 1; i < size_; ++i) {
    int cmp = 0;
    for (std::size_t c = 0; c < num_cols; ++c) {
      ValueType val_prev = p_->stored_cols[c][i - 1];
      ValueType val_curr = p_->stored_cols[c][i];
      if (val_prev < val_curr) {
        cmp = -1;
        break;
      }
      if (val_prev > val_curr) {
        cmp = 1;
        break;
      }
    }
    if (cmp > 0) {
      return false;
    }
  }
  return true;
}

// LeapfrogView is now defined in leapfrog_view.h for reuse

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <typename FirstHandle, typename... OtherHandles>
auto SortedArrayIndex<SR, AttrTuple, ValueType, RowIdType>::intersect(
    const FirstHandle& first, const OtherHandles&... others) {
  // If only one handle, just return its values (no intersection needed)
  if constexpr (sizeof...(OtherHandles) == 0) {
    return first.values();
  } else {
    // Multiple handles: use LeapfrogView for efficient intersection
    // Explicitly specify the iterator types to avoid deduction issues
    using IteratorType = typename decltype(first.values())::Iterator;
    return LeapfrogView<IteratorType, typename decltype(others.values())::Iterator...>(
        first.values().begin(), others.values().begin()..., first.values().end(),
        others.values().end()...);
  }
}
}  // namespace SRDatalog