#pragma once
#include "eytzinger.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "leapfrog_view.h"  // For LeapfrogView
#include "search.h"         // SRDatalog::search
#include "soa_merge.h"      // soa_inplace_merge
#include "sort.h"           // hwy_lexsort
#include "system.h"         // highway_aligned_memory_resource

namespace SRDatalog {

namespace {
// Local helper to dispatch stable sort based on number of columns
template <size_t K, typename ValueType>
void dispatch_stable_sort_impl(size_t n, std::span<const int> order, hwy_lexsort::SortContext& ctx,
                               Vector<Vector<ValueType>>& cols) {
  if (cols.size() == K) {
    // Safety check: ensure all columns have data
    for (size_t i = 0; i < K; ++i) {
      if (cols[i].data() == nullptr || cols[i].size() < n) {
        return;  // Invalid column data
      }
    }
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      hwy_lexsort::stable_lex_sort(n, order.data(), order.size(), ctx, true, cols[Is].data()...);
    }(std::make_index_sequence<K>{});
  } else {
    if constexpr (K > 1) {
      dispatch_stable_sort_impl<K - 1, ValueType>(n, order, ctx, cols);
    }
  }
}
}  // anonymous namespace

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline void ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_encoded(
    const IndexSpec& spec, const std::array<std::span<const ValueType>, arity>& encoded_rows,
    memory_resource* resource) {

  std::size_t num_rows = encoded_rows[0].size();
  if (num_rows == 0)
    return;

  // 1. Resource Management (Same as SortedArrayIndex)
  memory_resource* res = nullptr;
  if (resource != nullptr) {
    res = resource;
  } else if (p_ != nullptr) {
    res = p_->resource;
  } else {
    res = default_memory_resource();
  }

  if (!p_ || p_->resource != res)
    p_ = std::make_unique<Impl>(res);
  else
    p_->clear();

  // 2. Load Data into SoA (Structure of Arrays)
  // We resize columns based on the IndexSpec
  // Note: ImplicitBTree relies on spec.cols[0] being the primary sort key
  // Use highway_aligned_memory_resource for column vectors to ensure SIMD alignment
  // Note: We need to use the same resource for the entire lifetime of the vectors
  // to avoid reallocation issues during resize operations
  memory_resource* aligned_res = &highway_aligned_memory_resource();
  p_->cols.clear();
  p_->cols.reserve(spec.cols.size());
  for (size_t i = 0; i < spec.cols.size(); ++i) {
    // Allocate aligned vectors using highway aligned memory resource
    typename Impl::VecSizeT vec(aligned_res);
    vec.resize(num_rows);
    if (vec.data() == nullptr || vec.size() != num_rows) {
      return;  // Safety check: vector allocation failed
    }

    // Memcpy data from input encoded_rows
    // We map spec.cols[i] (logical column) to the physical storage
    if (encoded_rows[spec.cols[i]].data() == nullptr) {
      return;  // Safety check: source data is null
    }
    std::memcpy(vec.data(), encoded_rows[spec.cols[i]].data(), num_rows * sizeof(ValueType));

    p_->cols.push_back(std::move(vec));
  }

  // Verify all columns are properly initialized
  if (p_->cols.size() != spec.cols.size()) {
    return;  // Safety check: column count mismatch
  }
  for (size_t i = 0; i < p_->cols.size(); ++i) {
    if (p_->cols[i].data() == nullptr || p_->cols[i].size() != num_rows) {
      return;  // Safety check: column data invalid
    }
  }

  // 3. Prepare Postings (Row IDs 0..N)
  p_->postings.resize(num_rows);
  std::iota(p_->postings.begin(), p_->postings.end(), static_cast<RowId>(0));

  // Track rows processed before deduplication (for incremental builds)
  p_->rows_processed_ = num_rows;

  Vector<int> sort_order(res);
  sort_order.resize(spec.cols.size());
  // create the order vector: {0, 1, 2...} because p_->cols is already arranged by priority
  std::iota(sort_order.begin(), sort_order.end(), 0);

  // Thread-local sort context to avoid re-allocating buffers
  // Use highway_aligned_memory_resource for SortContext to ensure SIMD alignment
  static thread_local hwy_lexsort::SortContext tls_ctx(&highway_aligned_memory_resource());
  hwy_lexsort::SortContext* ctx_ptr = &tls_ctx;

  // Use the same dispatch pattern as sorted_array_index.ipp
  // This handles the case where spec.cols.size() may be less than arity
  // We need to ensure p_->cols.size() matches spec.cols.size() before dispatching
  if (p_->cols.size() != spec.cols.size()) {
    return;  // Safety check: columns should match spec
  }
  dispatch_stable_sort_impl<arity, ValueType>(num_rows, sort_order, *ctx_ptr, p_->cols);

  // 5. Apply Permutation to Postings
  // hwy_lexsort with reorder_data=true sorted the *columns*, but not the *postings*.
  // We need to fetch the permutation indices and apply them to p_->postings.
  const uint32_t* perm = ctx_ptr->get_sorted_indices<uint32_t>(num_rows);

  // Apply permutation to postings using a temp buffer
  // (Doing this manually because hwy_lexsort doesn't know about our postings vector)
  {
    Vector<RowId> tmp_postings(p_->resource);
    tmp_postings.resize(num_rows);
    // This loop is memory-bound; compiler should vectorize nicely
    for (size_t i = 0; i < num_rows; ++i) {
      tmp_postings[i] = p_->postings[perm[i]];
    }
    p_->postings = std::move(tmp_postings);
  }

  // 6. Deduplication (Compaction)
  // We merge identical rows and keep only unique tuples.
  if (num_rows > 1) {
    size_t write_idx = 0;
    size_t num_cols = p_->cols.size();

    // Cache pointers for speed
    std::vector<ValueType*> col_ptrs(num_cols);
    for (size_t c = 0; c < num_cols; ++c)
      col_ptrs[c] = p_->cols[c].data();
    RowId* post_ptr = p_->postings.data();

    for (size_t read_idx = 1; read_idx < num_rows; ++read_idx) {
      bool equal = true;
      // Check all columns for equality
      for (size_t c = 0; c < num_cols; ++c) {
        if (col_ptrs[c][read_idx] != col_ptrs[c][write_idx]) {
          equal = false;
          break;
        }
      }

      if (!equal) {
        ++write_idx;
        if (write_idx != read_idx) {
          // Move data to fill the gap
          for (size_t c = 0; c < num_cols; ++c) {
            col_ptrs[c][write_idx] = col_ptrs[c][read_idx];
          }
          post_ptr[write_idx] = post_ptr[read_idx];
        }
      }
    }

    // Resize if we shrank
    size_t unique_count = write_idx + 1;
    if (unique_count < num_rows) {
      for (auto& col : p_->cols)
        col.resize(unique_count);
      p_->postings.resize(unique_count);
      num_rows = unique_count;
    }
  }

  p_->size = num_rows;
  p_->index_arity = spec.cols.size();

  // 7. Build Eytzinger Index (The Implicit B-Tree Step)
  // Now that data is unique and sorted, we gather the block separators.
  if (num_rows > 0 && !p_->cols.empty()) {
    Vector<ValueType> separators(p_->resource);
    size_t num_blocks = (num_rows + BlockSize - 1) / BlockSize;
    separators.reserve(num_blocks);

    const auto& key_col = p_->cols[0];  // Primary Sort Key
    if (key_col.size() < num_rows) {
      return;  // Safety check: columns should match num_rows
    }

    // Gather the last element of every block
    for (size_t i = 0; i < num_blocks - 1; ++i) {
      size_t idx = ((i + 1) * BlockSize) - 1;
      if (idx < key_col.size()) {
        separators.push_back(key_col[idx]);
      }
    }

    // Build the accelerator
    p_->eytzinger_idx.build(separators);
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline typename ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle
ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::prefix(ValueType key) const {
  if (!valid() || is_leaf()) {
    return {};
  }

  if (!impl_ || depth_ >= impl_->cols.size()) {
    return {};
  }
  const auto& col = impl_->cols[depth_];
  const ValueType* col_ptr = col.data();
  if (col_ptr == nullptr) {
    return {};
  }

  // Initialize search bounds to the current node's range
  size_t lb = begin_;
  size_t current_end = end_;

  // only build eytzinger index for the root level now
  // TODO: Fix Eytzinger block search logic - currently disabled due to incorrect block narrowing
  // The Eytzinger search is an optimization to narrow the search range, but the current
  // implementation has issues with boundary cases. The search works correctly without
  // Eytzinger narrowing, so this is a performance optimization that can be fixed later.
  /*
  if (depth_ == 0 && impl_ && !impl_->eytzinger_idx.data.empty()) {
    size_t block = impl_->eytzinger_idx.search(key);
    size_t max_block = (impl_->size + BlockSize - 1) / BlockSize;
    if (block >= max_block) {
      block = max_block > 0 ? max_block - 1 : 0;
    }
    size_t block_start = block * BlockSize;
    size_t block_end = std::min(block_start + BlockSize, impl_->size);

    lb = std::max(begin_, block_start);
    current_end = std::min(end_, block_end);

    if (lb >= current_end) {
      return {};
    }
  }
  */

  // Perform a SIMD-accelerated search (should always fall into SIMD calls, check ASM code) within
  // the narrowed range.
  size_t search_len = current_end - lb;
  if (search_len == 0) {
    return {};
  }
  size_t offset = SRDatalog::search::adaptive_lower_bound(col_ptr + lb, search_len, key);
  lb += offset;

  // Check if we actually found the key
  if (lb >= end_ || col_ptr[lb] != key) {
    return {};
  }

  size_t ub = lb + 1;

  // Check the immediate next value.
  if (ub < end_ && col_ptr[ub] == key) {
    // Key repeats. Find the end of the run using exponential/binary search.
    size_t remaining = end_ - ub;
    size_t ub_offset;

    if (key < std::numeric_limits<ValueType>::max()) {
      ub_offset = SRDatalog::search::adaptive_lower_bound(col_ptr + ub, remaining, key + 1);
    } else {
      // Edge case: Key is MAX_INT, unlikely
      ub_offset = remaining;
    }
    ub += ub_offset;
  }

  return NodeHandle(impl_, lb, ub, depth_ + 1);
}

// --- Clone implementation ---
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::clone_from(
    const ImplicitBTreeIndex& other, memory_resource* resource) {
  if (!other.p_) {
    clear();
    return;
  }

  memory_resource* res = nullptr;
  if (resource != nullptr) {
    res = resource;
  } else if (p_ != nullptr) {
    res = p_->resource;
  } else {
    res = default_memory_resource();
  }
  p_ = std::make_unique<Impl>(res);
  p_->index_arity = other.p_->index_arity;
  p_->size = other.p_->size;
  p_->rows_processed_ = other.p_->rows_processed_;

  p_->postings = Vector<RowId>(res);
  p_->postings.assign(other.p_->postings.begin(), other.p_->postings.end());

  p_->cols.reserve(other.p_->cols.size());
  for (const auto& col : other.p_->cols) {
    typename Impl::VecSizeT new_col(res);
    new_col.assign(col.begin(), col.end());
    p_->cols.push_back(std::move(new_col));
  }

  // Copy Eytzinger index data
  p_->eytzinger_idx.data = Vector<ValueType>(res);
  p_->eytzinger_idx.data.assign(other.p_->eytzinger_idx.data.begin(),
                                other.p_->eytzinger_idx.data.end());
}

// --- Merge implementation ---
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::merge(const ImplicitBTreeIndex& other,
                                                                    std::size_t row_id_offset) {
  // 1. Fast path: If other is empty, do nothing.
  if (other.empty())
    return;

  // 2. Fast path: If this is empty, just clone other (handling offset).
  if (empty()) {
    clone_from(other);
    if (row_id_offset > 0 && p_) {
      for (auto& rid : p_->postings) {
        if constexpr (std::is_same_v<RowIdType, std::size_t>) {
          rid += row_id_offset;
        } else {
          rid = static_cast<RowIdType>(static_cast<std::size_t>(rid) + row_id_offset);
        }
      }
    }
    return;
  }

  const std::size_t old_size = p_->size;
  const std::size_t other_size = other.p_->size;

  // 3. In-place merge using soa_inplace_merge
  // Now that soa_inplace_merge uses Vector, we can pass p_->cols directly
  // Perform the merge directly - no conversion needed!
  p_->size = soa_inplace_merge<ValueType, RowIdType>(p_->cols, p_->postings, other.p_->cols,
                                                     other.p_->postings, old_size, other_size,
                                                     row_id_offset);
  p_->rows_processed_ += other.p_->rows_processed_;

  // 4. Rebuild Eytzinger index after merge
  if (p_->size > 0 && !p_->cols.empty()) {
    Vector<ValueType> separators(p_->resource);
    size_t num_blocks = (p_->size + BlockSize - 1) / BlockSize;
    separators.reserve(num_blocks);

    const auto& key_col = p_->cols[0];  // Primary Sort Key
    if (key_col.size() < p_->size) {
      return;  // Safety check: columns should match size
    }

    // Gather the last element of every block
    for (size_t i = 0; i < num_blocks - 1; ++i) {
      size_t idx = ((i + 1) * BlockSize) - 1;
      if (idx < key_col.size()) {
        separators.push_back(key_col[idx]);
      }
    }

    // Build the accelerator
    p_->eytzinger_idx.build(separators);
  }
}

// Implementation of static intersect method using LeapfrogView
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <typename FirstHandle, typename... OtherHandles>
auto ImplicitBTreeIndex<SR, AttrTuple, ValueType, RowIdType>::intersect(
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
