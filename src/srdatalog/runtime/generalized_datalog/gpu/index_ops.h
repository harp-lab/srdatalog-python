#pragma once

/**
 * @file index_ops.h
 * @brief Customization Point Objects (CPOs) for index operations
 *
 * @details Provides extensible dispatch via ADL:
 * - index_ops::merge_index      - merges delta into full index
 * - index_ops::set_difference   - computes delta = new - full
 *
 * To customize for a new index type, define the corresponding `_impl` free
 * function in your index's namespace. ADL will find it automatically.
 *
 * @see index_concepts.h for the full BYODS contract
 */

#include "index_concepts.h"
#include <cstddef>

namespace SRDatalog::GPU::index_ops {

/**
 * @brief CPO for merging delta index into full index
 *
 * @details ADL dispatch: define `merge_index_impl(full, delta, full_rel, delta_rel)`
 * in your index's namespace to customize. Default: calls `full.merge(delta, full.size())`
 *
 * Example (Device2LevelIndex): merges into HEAD segment with compaction policy.
 */
inline constexpr struct merge_index_fn {
  template <typename Idx, typename FullRel, typename DeltaRel>
  void operator()(Idx& full, Idx& delta, FullRel& full_rel, DeltaRel& delta_rel) const {
    if constexpr (requires { merge_index_impl(full, delta, full_rel, delta_rel); }) {
      merge_index_impl(full, delta, full_rel, delta_rel);
    } else {
      // Default: simple merge using index's own merge method
      std::size_t offset = full.size();
      full.merge(delta, offset);
    }
  }
} merge_index{};

/**
 * @brief CPO for set difference: delta = new - full
 *
 * @details ADL dispatch: define `set_difference_impl(new_idx, full_idx, delta_idx)`
 * in your index's namespace to customize. Default: calls
 * `new_idx.set_difference_update(full_idx, delta_idx)`
 *
 * Example (Device2LevelIndex): 2-step diff against FULL then HEAD segments.
 */
inline constexpr struct set_difference_fn {
  template <typename Idx>
  void operator()(Idx& new_idx, Idx& full_idx, Idx& delta_idx) const {
    if constexpr (requires { set_difference_impl(new_idx, full_idx, delta_idx); }) {
      set_difference_impl(new_idx, full_idx, delta_idx);
    } else {
      // Default: direct method call
      new_idx.set_difference_update(full_idx, delta_idx);
    }
  }
} set_difference{};

}  // namespace SRDatalog::GPU::index_ops
