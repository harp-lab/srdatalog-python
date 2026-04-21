#pragma once

/**
 * @file deduplication.h
 * @brief Deduplication logic for filtering against full relations
 */

#include "../device_sorted_array_index.h"
#include "../semiring.h"
#include "./store.h"

namespace SRDatalog::GPU {

/**
 * @brief Check if tuple should be emitted (not duplicate in full relation)
 */
template <typename Schema, typename Terms, typename VarPosMap, typename ValueType,
          std::size_t NumVars, typename HandleType, Semiring SR, typename IndexSpec>
struct FilterAgainstFull {

  template <typename Group>
  __device__ static bool should_emit(const state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                     HandleType full_handle,
                                     const typename HandleType::View& full_view,
                                     semiring_value_t<SR> new_prov, Group tile) {

    // Navigate through index using prefix lookups of IndexSpec column order
    // IndexSpec::column_indexes_type is integer_sequence<int, Col1, Col0...>
    // These int indices refer to the position in Terms tuple (which matches Schema order)
    using ColSeq = typename IndexSpec::column_indexes_type;

    HandleType current = full_handle;

    // Helper to short-circuit: if we become invalid, stops further prefix calls
    auto traverse = [&]<int... Is>(std::integer_sequence<int, Is...>) {
      ((current.valid()
            ? (current = current.prefix(vars.template get<std::tuple_element_t<Is, Terms>>(), tile,
                                        full_view),
               0)
            : 0),
       ...);
    };

    traverse(ColSeq{});

    if (!current.valid()) {
      return true;  // Tuple doesn't exist -> emit
    }

    // Tuple exists in FULL - should NOT emit (it's a duplicate)
    if constexpr (std::is_same_v<SR, BooleanSR>) {
      return false;  // Boolean: exists means skip
    } else {
      // Check if new provenance improves result
      auto existing_prov = current.template prov<Group::size()>(full_view);
      auto combined = sr_add<SR>(existing_prov, new_prov);
      return combined != existing_prov;
    }
  }
};

}  // namespace SRDatalog::GPU
