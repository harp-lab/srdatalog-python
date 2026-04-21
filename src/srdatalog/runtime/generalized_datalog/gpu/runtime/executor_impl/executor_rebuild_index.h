#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

template <mir::CIndexSpec IndexSpecT, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::RebuildIndex<IndexSpecT>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using Schema = typename IndexSpecT::schema_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    constexpr std::size_t ver = IndexSpecT::kVersion;

    auto& rel = get_relation_by_schema<Schema, ver>(db);
    auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    bool new_tuples = false;
    nvtxRangePushA("RebuildIndex");
    if constexpr (ver == NEW_VER) {
      // One-shot ownership transfer: intern cols → index (zero-copy for identity spec).
      auto& new_idx = rel.build_index_take_ownership(runtime_spec);
      std::size_t root_degree = new_idx.root().degree();
      new_tuples = root_degree > 0;
    } else if constexpr (ver == FULL_VER) {
      rel.ensure_index(runtime_spec, false);
    } else {
      // DELTA_VER: build non-canonical index from canonical (index-to-index).
      rel.rebuild_index_from_existing(runtime_spec);
    }
    nvtxRangePop();  // RebuildIndex

    auto res = GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
    return res || new_tuples;
  }
};

}  // namespace SRDatalog::GPU
