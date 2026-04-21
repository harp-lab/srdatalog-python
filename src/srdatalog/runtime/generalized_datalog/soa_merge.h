#pragma once

#include <cstddef>
#include <cstring>
#include <limits>

#include "system.h"
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace SRDatalog {

/**
 * @brief In-place merge of two sorted Structure-of-Arrays (SoA) data structures.
 *
 * @details This function performs an optimal in-place merge of two sorted SoA structures.
 * The merge strategy:
 * 1. Resize destination vectors to accommodate merged result
 * 2. Move existing data from [0, old_size-1] to [other_size, total_size-1]
 * 3. Merge forward: read from moved data [other_size, ...] and other [0, ...],
 *    write to [0, ...]
 * This avoids overwriting unread data and is optimal for in-place merge.
 *
 * @tparam ValueType The value type stored in columns
 * @tparam RowIdType The row ID type
 * @tparam ColumnContainer Container type for columns (must support .data(), .resize(), .size())
 * @tparam PostingContainer Container type for postings (must support .data(), .resize())
 *
 * @param this_cols Vector of column containers for "this" (destination) data
 * @param this_postings Postings container for "this" data
 * @param other_cols Vector of column containers for "other" (source) data
 * @param other_postings Postings container for "other" data
 * @param old_size Current size of "this" data (before merge)
 * @param other_size Size of "other" data
 * @param row_id_offset Offset to add to row IDs from "other" when merging
 *
 * @return The final merged size (total_size = old_size + other_size)
 */
template <typename ValueType, typename RowIdType, typename ColumnContainer,
          typename PostingContainer>
std::size_t soa_inplace_merge(Vector<ColumnContainer>& this_cols, PostingContainer& this_postings,
                              const Vector<ColumnContainer>& other_cols,
                              const PostingContainer& other_postings, std::size_t old_size,
                              std::size_t other_size, std::size_t row_id_offset) {
  const std::size_t total_size = old_size + other_size;
  const std::size_t num_cols = this_cols.size();

  // Resize vectors to accommodate merged result without zero-initialization
  // We'll immediately overwrite the new elements, so uninitialized resize avoids page faults
  for (size_t c = 0; c < num_cols; ++c) {
    this_cols[c].resize(total_size, default_init);
    optimize_vector_memory(this_cols[c].data(), total_size * sizeof(ValueType));
  }
  this_postings.resize(total_size, default_init);
  optimize_vector_memory(this_postings.data(), total_size * sizeof(RowIdType));

  // Prefault newly allocated memory pages without initialization
  // We only need to prefault the new memory region (old_size to total_size)
  // Use madvise to hint the OS to prefault pages, avoiding the need to touch memory
  if (other_size > 0) {
#ifdef __linux__
    for (size_t c = 0; c < num_cols; ++c) {
      void* new_mem_start = this_cols[c].data() + old_size;
      size_t new_bytes = other_size * sizeof(ValueType);
      // Align to page boundary
      uintptr_t addr = reinterpret_cast<uintptr_t>(new_mem_start);
      uintptr_t aligned_addr = addr & ~(4095ULL);
      size_t aligned_size = (addr - aligned_addr) + new_bytes;
      // Hint OS to prefault these pages
      madvise(reinterpret_cast<void*>(aligned_addr), aligned_size, MADV_WILLNEED);
    }
    // Also prefault postings
    void* new_post_start = this_postings.data() + old_size;
    size_t new_post_bytes = other_size * sizeof(RowIdType);
    uintptr_t post_addr = reinterpret_cast<uintptr_t>(new_post_start);
    uintptr_t aligned_post_addr = post_addr & ~(4095ULL);
    size_t aligned_post_size = (post_addr - aligned_post_addr) + new_post_bytes;
    madvise(reinterpret_cast<void*>(aligned_post_addr), aligned_post_size, MADV_WILLNEED);
#endif
  }

  // Move existing data to the end to make room for merge
  // Move from [0, old_size-1] to [other_size, total_size-1] backwards to avoid overwriting
  if (old_size > 0) {
    for (size_t c = 0; c < num_cols; ++c) {
      std::memmove(this_cols[c].data() + other_size,  // dest
                   this_cols[c].data(),               // src
                   old_size * sizeof(ValueType)       // size
      );
    }
    std::memmove(this_postings.data() + other_size,  // dest
                 this_postings.data(),               // src
                 old_size * sizeof(RowIdType)        // size
    );
  }

  // Setup pointers for fast raw access
  Vector<ValueType*> this_ptrs(num_cols, default_memory_resource());
  // Points to moved data at offset other_size
  Vector<const ValueType*> other_ptrs(num_cols, default_memory_resource());
  // Points to beginning where we write
  Vector<ValueType*> dst_ptrs(num_cols, default_memory_resource());

  for (size_t c = 0; c < num_cols; ++c) {
    this_ptrs[c] = this_cols[c].data() + other_size;  // Moved data starts here
    other_ptrs[c] = other_cols[c].data();
    dst_ptrs[c] = this_cols[c].data();  // Write destination
  }

  RowIdType* this_postings_ptr = this_postings.data() + other_size;  // Moved postings
  const RowIdType* other_postings_ptr = other_postings.data();
  RowIdType* dst_postings_ptr = this_postings.data();  // Write destination

  // In-place merge: merge forward from the beginning
  // Read from this[other_size + i] and other[j], write to this[k]
  std::size_t i = 0;  // index into moved this data (at offset other_size)
  std::size_t j = 0;  // index into other
  std::size_t k = 0;  // write position

  auto apply_row_id_offset = [&](RowIdType rid) -> RowIdType {
    if constexpr (std::is_same_v<RowIdType, std::size_t>) {
      return rid + row_id_offset;
    } else {
      std::size_t sum = static_cast<std::size_t>(rid) + row_id_offset;
      // Use assert instead of exception for performance-critical path
      // assert(sum <= static_cast<std::size_t>(std::numeric_limits<RowIdType>::max()) &&
      //        "row_id_offset too large for RowIdType");
      return static_cast<RowIdType>(sum);
    }
  };

  auto compare_at_indices = [&](std::size_t idx_a, std::size_t idx_b) -> int {
    for (size_t c = 0; c < num_cols; ++c) {
      ValueType val_a = this_ptrs[c][idx_a];
      ValueType val_b = other_ptrs[c][idx_b];
      // Single branch instead of two
      if (val_a != val_b) {
        return (val_a < val_b) ? -1 : 1;
      }
    }
    return 0;
  };

  while (i < old_size && j < other_size) {
    int cmp = compare_at_indices(i, j);
    if (cmp < 0) {
      // this[other_size + i] < other[j], write this[other_size + i]
      for (size_t c = 0; c < num_cols; ++c) {
        dst_ptrs[c][k] = this_ptrs[c][i];
      }
      dst_postings_ptr[k] = this_postings_ptr[i];
      ++i;
      ++k;
    } else if (cmp > 0) {
      // this[other_size + i] > other[j], write other[j]
      for (size_t c = 0; c < num_cols; ++c) {
        dst_ptrs[c][k] = other_ptrs[c][j];
      }
      dst_postings_ptr[k] = apply_row_id_offset(other_postings_ptr[j]);
      ++j;
      ++k;
    } else {
      // Equal: write both (this first, then other, to match original order)
      for (size_t c = 0; c < num_cols; ++c) {
        dst_ptrs[c][k] = this_ptrs[c][i];
      }
      dst_postings_ptr[k] = this_postings_ptr[i];
      ++i;
      ++k;
      for (size_t c = 0; c < num_cols; ++c) {
        dst_ptrs[c][k] = other_ptrs[c][j];
      }
      dst_postings_ptr[k] = apply_row_id_offset(other_postings_ptr[j]);
      ++j;
      ++k;
    }
  }

  // Copy remaining elements from this
  if (i < old_size) {
    size_t remaining = old_size - i;
    // Bulk copy all columns at once
    for (size_t c = 0; c < num_cols; ++c) {
      std::memcpy(&dst_ptrs[c][k], &this_ptrs[c][i], remaining * sizeof(ValueType));
    }
    std::memcpy(&dst_postings_ptr[k], &this_postings_ptr[i], remaining * sizeof(RowIdType));
    k += remaining;
  }

  // Copy remaining elements from other
  if (j < other_size) {
    size_t remaining = other_size - j;
    for (size_t c = 0; c < num_cols; ++c) {
      std::memcpy(&dst_ptrs[c][k], &other_ptrs[c][j], remaining * sizeof(ValueType));
    }
    // Need to apply offset to postings
    if (row_id_offset == 0) {
      std::memcpy(&dst_postings_ptr[k], &other_postings_ptr[j], remaining * sizeof(RowIdType));
    } else {
      for (size_t idx = 0; idx < remaining; ++idx) {
        dst_postings_ptr[k + idx] = apply_row_id_offset(other_postings_ptr[j + idx]);
      }
    }
    k += remaining;
  }

  return total_size;
}

}  // namespace SRDatalog
