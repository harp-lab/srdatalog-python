#pragma once

#include "../gpu_fixpoint_executor_common.h"
#include "gpu/index_ops.h"

namespace SRDatalog::GPU {

/**
 * @brief MergeIndex executor - uses CPO dispatch for extensibility
 *
 * @details Merges DELTA index into FULL index using index_ops::merge_index CPO.
 * Different index types can customize behavior via ADL.
 */
template <typename IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::MergeIndex<IndexSpecT>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using Schema = typename IndexSpecT::schema_type;
    static_assert(IndexSpecT::kVersion == FULL_VER, "MergeIndex must be for FULL_VER index");

    auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(db);
    auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(db);

    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    // Ensure indexes exist
    full_rel.ensure_index(runtime_spec, false);
    auto& full_idx = full_rel.get_index(runtime_spec);

    // Delta index was just built by compute_delta_fn — use get_index() directly.
    auto& delta_idx = delta_rel.get_index(runtime_spec);

    // Merge using CPO - dispatches via ADL
    if (!delta_idx.empty()) {
      index_ops::merge_index(full_idx, delta_idx, full_rel, delta_rel);
    }

    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
