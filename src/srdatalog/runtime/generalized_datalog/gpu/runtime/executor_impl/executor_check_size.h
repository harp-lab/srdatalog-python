#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

template <typename Schema, std::size_t Ver, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::CheckSize<Schema, Ver>, TailOps...>, DB> {
  // Helper predicate to find InsertFromRelation or MergeIndex for this schema
  template <typename Op>
  struct IsInsertForSchema {
    static constexpr bool value = []() {
      if constexpr (mir::is_insert_from_relation_v<Op>) {
        return std::is_same_v<typename Op::schema_type, Schema>;
      } else if constexpr (mir::is_compute_delta_and_merge_v<Op>) {
        return std::is_same_v<typename Op::schema_type, Schema>;
      } else if constexpr (mir::is_compute_delta_v<Op>) {
        return std::is_same_v<typename Op::schema_type, Schema>;
      } else if constexpr (mir::is_merge_index_v<Op>) {
        // MergeIndex uses IndexSpecT which contains Schema
        using IndexSpec = typename Op::index_spec_type;
        return std::is_same_v<typename IndexSpec::schema_type, Schema>;
      } else {
        return false;
      }
    }();
  };

  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& r = get_relation_by_schema<Schema, Ver>(db);

    // Get canonical index spec from the InsertFromRelation operation for this schema
    using TailOpsTuple = std::tuple<TailOps...>;
    using FoundInsert = mp::mp_find_if<TailOpsTuple, IsInsertForSchema>;
    static_assert(FoundInsert::value < sizeof...(TailOps),
                  "CheckSize: InsertFromRelation for this schema must exist in TailOps");

    using InsertOp = mp::mp_at_c<TailOpsTuple, FoundInsert::value>;
    using IndexSpecT = typename InsertOp::index_spec_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto runtime_canonical_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(ColSeq{});

    // Force rebuild to ensure index reflects relation data
    r.ensure_index(runtime_canonical_spec, true);

    std::size_t canonical_size = 0;
    try {
      auto& canonical_idx = r.get_index(runtime_canonical_spec);
      canonical_size = canonical_idx.root().degree();
    } catch (...) {
      canonical_size = r.size();
    }

    // Always continue to remaining operations
    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
