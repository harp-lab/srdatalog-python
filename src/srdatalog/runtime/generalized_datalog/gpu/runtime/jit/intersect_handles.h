/**
 * @file intersect_handles.h
 * @brief Runtime helper for JIT kernel multi-source intersection
 *
 * Provides intersect_handles() function used by JIT-generated kernels
 * for LFTJ-style multi-source variable binding.
 */

#pragma once

#include "../../device_sorted_array_index.h"

namespace SRDatalog::GPU::JIT {

/**
 * @brief Intersect multiple handle spans for multi-source variable binding
 *
 * This is a device-callable helper used by JIT-generated kernels.
 * It wraps DeviceSortedArrayIndex::intersect() to provide a uniform interface.
 *
 * Usage in JIT kernel:
 *   auto intersect = intersect_handles(tile, h1.iterators(view1), h2.iterators(view2));
 *   for (auto it = intersect.begin(); it.valid(); it.next()) {
 *     auto x = it.value();
 *     auto positions = it.positions();
 *     // ...
 *   }
 *
 * @tparam Group Cooperative group type (e.g., cg::thread_block_tile<32>)
 * @tparam Spans Variadic span types from handle.iterators()
 * @param tile The cooperative group for parallel search
 * @param spans The spans to intersect
 * @return IntersectView iterator over intersection results
 */
template <typename Group, typename... Spans>
__device__ auto intersect_handles(Group tile, Spans... spans) {
  // Call DeviceSortedArrayIndex's static intersect method
  // The method is a static template - AttrTuple doesn't matter for the static intersect method
  // since intersect only cares about the span value types
  return DeviceSortedArrayIndex<NoProvenance, std::tuple<uint32_t>, uint32_t, uint32_t>::intersect(
      tile, spans...);
}

}  // namespace SRDatalog::GPU::JIT
