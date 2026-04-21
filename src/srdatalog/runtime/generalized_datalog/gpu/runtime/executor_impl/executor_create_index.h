#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for CreateIndex: Creates a new index for a relation.
 */
template <mir::CIndexSpec IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::CreateIndex<IndexSpecT>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using Schema = typename IndexSpecT::schema_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    constexpr std::size_t ver = IndexSpecT::kVersion;

    // In iteration 0, DELTA_VER relations are redirected to FULL_VER
    auto& rel = [&]() -> auto& {
      if constexpr (ver == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
      }
      return get_relation_by_schema<Schema, ver>(db);
    }();

    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    rel.ensure_index(runtime_spec);

    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
