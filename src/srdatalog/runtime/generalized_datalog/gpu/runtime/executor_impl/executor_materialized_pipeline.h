#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for MaterializedExecutePipeline: Host-side Thrust executor
 *
 * @details MaterializedExecutePipeline uses a JitRunner struct with a static execute()
 * method that runs entirely on the host using Thrust primitives. No CUDA kernels are
 * launched directly; instead, Thrust operations handle all GPU work.
 *
 * The JitRunner implements:
 * 1. Probe phase: count matches, compute offsets, materialize row ID pairs
 * 2. Gather phase: extract columns from source views using row IDs
 * 3. Insert phase: copy gathered data to destination relation
 *
 * @tparam JitRunner The JitRunner struct (e.g., JitRunner_CGE_Materialized)
 */
template <typename JitRunner, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::MaterializedExecutePipeline<JitRunner>, TailOps...>,
                           DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    // Call JitRunner::execute() directly - it's a host-side function
    // that uses Thrust primitives internally
    JitRunner::execute(db, static_cast<uint32_t>(iteration));

    auto result =
        GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
    GPU_DEVICE_SYNCHRONIZE();
    return result;
  }
};

}  // namespace SRDatalog::GPU
