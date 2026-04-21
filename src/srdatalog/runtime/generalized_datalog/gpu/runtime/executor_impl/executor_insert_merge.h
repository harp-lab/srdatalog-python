#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for InsertFromRelation: Merges one relation into another.
 */
template <typename Schema, std::size_t FromVer, std::size_t ToVer, typename IndexSpec,
          typename... TailOps, typename DB>
struct GPUFixpointExecutor<
    std::tuple<mir::InsertFromRelation<Schema, FromVer, ToVer, IndexSpec>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    // In iteration 0, redirect DELTA_VER reads to FULL_VER
    auto& from = [&]() -> auto& {
      if constexpr (FromVer == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
      }
      return get_relation_by_schema<Schema, FromVer>(db);
    }();

    auto& to = get_relation_by_schema<Schema, ToVer>(db);

    using ColSeq = typename IndexSpec::column_indexes_type;
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    auto& from_idx = from.ensure_index(runtime_spec);

    if (iteration == 0) {
      to.ensure_index(runtime_spec);
    } else {
      try {
        auto& test_idx = to.get_index(runtime_spec);
        (void)test_idx;
      } catch (...) {
        to.ensure_index(runtime_spec);
      }
    }
    auto& to_idx = to.get_index(runtime_spec);

    std::size_t row_offset = to_idx.size();
    to_idx.merge(from_idx, row_offset);

    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
