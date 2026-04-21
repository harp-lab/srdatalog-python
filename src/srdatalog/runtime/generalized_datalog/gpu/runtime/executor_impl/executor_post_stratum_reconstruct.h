#pragma once

#include "../gpu_fixpoint_executor_common.h"
#include "mir_plan.h"  // For mir::PostStratumReconstructInternCols

namespace SRDatalog::GPU {

/**
 * @brief PostStratumReconstructInternCols executor
 *
 * @details Reconstructs intern column values FROM the canonical index TO column storage.
 * This MUST only be called after fixpoint plan completes, NOT inside fixpoint loop.
 *
 * The operation copies values from the sorted index back to the relation's linear
 * column storage (intern columns), ensuring that downstream strata have consistent
 * data regardless of which index they use.
 */
template <typename IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<
    std::tuple<mir::PostStratumReconstructInternCols<IndexSpecT>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using Schema = typename IndexSpecT::schema_type;

    auto& rel = get_relation_by_schema<Schema, FULL_VER>(db);

    // Extract runtime index spec from compile-time IndexSpecT
    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    // Ensure the canonical index exists and is built
    rel.ensure_index(runtime_spec, false);
    auto& idx = rel.get_index(runtime_spec);

    // Resize intern columns to match index size
    std::size_t idx_size = idx.size();
    rel.resize_interned_columns(idx_size);
    if constexpr (has_provenance_v<typename Schema::semiring_type>) {
      rel.provenance().resize(idx_size);
    }

    // Copy values from index to intern columns
    // This reconstructs the linear storage from the authoritative index state
    idx.reconstruct_to_relation(runtime_spec, rel.unsafe_interned_columns(), rel.provenance());

    // Continue with tail operations
    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
