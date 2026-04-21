#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

// Specialization for Nested Fixpoint Plan
template <typename Instructions, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::FixpointPlan<Instructions>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using InnerExecutor = GPUFixpointExecutor<Instructions, DB>;

    bool any_change = false;
    for (std::size_t i = 0; i < max_output; ++i) {
      bool has_new = InnerExecutor::execute(db, i, max_output);

      if (has_new)
        any_change = true;

      if (!has_new && i > 0) {
        break;
      }
    }

    bool tail_result =
        GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);

    return any_change || tail_result;
  }
};

}  // namespace SRDatalog::GPU
