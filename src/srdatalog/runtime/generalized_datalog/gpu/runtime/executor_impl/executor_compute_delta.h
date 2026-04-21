#pragma once

#include "../gpu_fixpoint_executor_common.h"
#include "gpu/index_ops.h"

namespace SRDatalog::GPU {

/**
 * @brief ComputeDelta executor - uses CPO dispatch for extensibility
 */
template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
          typename IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<
    std::tuple<mir::ComputeDelta<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& new_rel = get_relation_by_schema<Schema, FromVer>(db);
    auto& full_rel = get_relation_by_schema<Schema, ToVer>(db);
    auto& delta_rel = get_relation_by_schema<Schema, DeltaVer>(db);

    delta_rel.clear();

    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    // NEW_VER index was just built by rebuild_index — use get_index() directly.
    auto& new_idx = new_rel.get_index(runtime_spec);

    full_rel.ensure_index(runtime_spec, false);
    auto& full_idx = full_rel.get_index(runtime_spec);

    delta_rel.ensure_index(runtime_spec);
    auto& delta_idx = delta_rel.get_index(runtime_spec);

    new_idx.set_difference_update(full_idx, delta_idx);

    using IndexType = std::remove_cvref_t<decltype(delta_idx)>;

    // For owning indices (DSAI), skip reconstruct — merge() uses p_->cols directly.

    new_rel.clear();

    bool has_new_tuples = !delta_idx.empty();

    bool tail_result =
        GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);

    return has_new_tuples || tail_result;
  }
};

/**
 * @brief ComputeDeltaAndMerge executor - uses CPO dispatch
 */
template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
          typename IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<
    std::tuple<mir::ComputeDeltaAndMerge<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>, TailOps...>,
    DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& new_rel = get_relation_by_schema<Schema, FromVer>(db);
    auto& full_rel = get_relation_by_schema<Schema, ToVer>(db);
    auto& delta_rel = get_relation_by_schema<Schema, DeltaVer>(db);

    delta_rel.clear();

    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    // NEW_VER index was just built by rebuild_index — use get_index() directly.
    auto& new_idx = new_rel.get_index(runtime_spec);

    full_rel.ensure_index(runtime_spec, false);
    auto& full_idx = full_rel.get_index(runtime_spec);

    delta_rel.ensure_index(runtime_spec);
    auto& delta_idx = delta_rel.get_index(runtime_spec);

    new_idx.set_difference_update(full_idx, delta_idx);

    if (!delta_idx.empty()) {
      index_ops::merge_index(full_idx, delta_idx, full_rel, delta_rel);
    }

    if constexpr (has_provenance_v<typename Schema::semiring_type>) {
      delta_idx.reconstruct_to_relation(runtime_spec, delta_rel.unsafe_interned_columns(),
                                        delta_rel.provenance());
    }

    new_rel.clear();

    bool has_new_tuples = !delta_idx.empty();
    bool tail_result =
        GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);

    return has_new_tuples || tail_result;
  }
};

}  // namespace SRDatalog::GPU
