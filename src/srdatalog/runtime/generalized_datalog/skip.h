/**
 * @file skip.h
 * @brief This file provide the SIMD operation for skip duplicate values in a sorted array during
 * search.
 */

#pragma once

#include "search.h"
#include <algorithm>  // For std::lower_bound
#include <hwy/highway.h>
#include <iterator>
#include <ranges>

// Highway Boilerplate: establishes the namespace for the specific architecture
HWY_BEFORE_NAMESPACE();
namespace hwy_impl {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
const T* skip_duplicates_hwy(const T* ptr, const T* end) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);

  // 1. Snapshot the value we are skipping
  const T target_val = *ptr;
  const auto v_target = hn::Set(d, target_val);

  // 2. SIMD Loop: Consume chunks of size N
  // We use <= end because we need a full vector's worth of data to load safely.
  // (If you have a padded buffer, you can be more aggressive here).
  while (ptr + N <= end) {
    // LoadU handles unaligned addresses (ptr increments by N, so alignment drifts)
    const auto v_data = hn::LoadU(d, ptr);

    // Compare: result is a mask where TRUE means "is a duplicate"
    const auto mask = hn::Eq(v_data, v_target);

    // Optimization: If the whole vector matches, skip it all.
    if (hn::AllTrue(d, mask)) {
      ptr += N;
      continue;
    }

    // 3. Mismatch Found inside this chunk
    // We want the first index where they are NOT equal.
    // Invert the mask: TRUE means "is DIFFERENT"
    const auto not_mask = hn::Not(mask);

    // FindFirstTrue returns the index (0..N-1) of the first set bit
    const size_t offset = hn::FindFirstTrue(d, not_mask);
    return ptr + offset;
  }

  // 4. Tail Cleanup (Scalar)
  // Handle the last few elements that didn't fit in a SIMD vector
  while (ptr < end && *ptr == target_val) {
    ptr++;
  }

  return ptr;
}

}  // namespace hwy_impl
HWY_AFTER_NAMESPACE();

// Public Wrapper
// This dispatches to the best instruction set selected at compile/runtime
namespace SRDatalog {
template <typename T>
const T* skip_duplicates(const T* ptr, const T* end) {
  return hwy_impl::skip_duplicates_hwy(ptr, end);
}

/**
 * @brief A range view that deduplicates consecutive duplicate values in a sorted array.
 * @details This range view uses skip_duplicates to efficiently skip over consecutive
 *          duplicate values, providing a view of only unique values. The underlying array
 *          must be sorted for correct behavior.
 * @tparam T The value type
 */
template <typename T>
struct DedupRange {
  const T* begin_ptr;
  const T* end_ptr;

  struct Iterator {
    const T* current;
    const T* end;

    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    Iterator() : current(nullptr), end(nullptr) {}
    Iterator(const T* ptr, const T* e) : current(ptr), end(e) {}

    reference operator*() const {
      return *current;
    }
    pointer operator->() const {
      return current;
    }

    Iterator& operator++() {
      if (current < end) {
        current = skip_duplicates(current, end);
      }
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    /**
     * @brief Seek to the first position >= target value.
     * @details Uses binary search to efficiently find the target value in the sorted range.
     *          This is required for LeapfrogView to work efficiently.
     *          After seeking, the iterator points to the first element >= target,
     *          or to end() if no such element exists.
     */
    void seek(const T& target) {
      if (current == nullptr || end == nullptr || current >= end) {
        return;  // Already at end or invalid
      }
      // Use adaptive_lower_bound to find the first position >= target
      // adaptive_lower_bound returns an offset (std::size_t), not a pointer
      std::size_t offset = SRDatalog::search::adaptive_lower_bound(
          current, static_cast<std::size_t>(end - current), target);
      const T* found = current + offset;
      if (found < end) {
        current = found;
        // After binary search, we might be in the middle of duplicates
        // Skip back to the beginning of this value's duplicates if needed
        // (Actually, lower_bound already gives us the first >= target, so we're good)
      } else {
        current = end;  // Target not found, move to end
      }
    }

    bool operator==(const Iterator& other) const {
      return current == other.current;
    }

    bool operator!=(const Iterator& other) const {
      return !(*this == other);
    }
  };

  Iterator begin() const {
    if (begin_ptr >= end_ptr) {
      return Iterator(end_ptr, end_ptr);
    }
    return Iterator(begin_ptr, end_ptr);
  }

  Iterator end() const {
    return Iterator(end_ptr, end_ptr);
  }
};

}  // namespace SRDatalog