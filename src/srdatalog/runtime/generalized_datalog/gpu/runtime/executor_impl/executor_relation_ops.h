#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

// SwapRelations
template <typename Schema, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::SwapRelations<Schema>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& r1 = get_relation_by_schema<Schema, DELTA_VER>(db);
    auto& r2 = get_relation_by_schema<Schema, NEW_VER>(db);

    using std::swap;
    swap(r1, r2);

    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

// ClearRelation
template <typename Schema, std::size_t Ver, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::ClearRelation<Schema, Ver>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& r = get_relation_by_schema<Schema, Ver>(db);
    r.clear();

    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

// SetVersion
template <typename Schema, std::size_t Ver, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::SetVersion<Schema, Ver>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    auto& r = get_relation_by_schema<Schema, Ver>(db);
    r.set_version(Ver);
    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
