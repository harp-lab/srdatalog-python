#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for JitExecutePipeline: Calls JitRunner::execute()
 *
 * @details JitRunner is a complete struct with __global__ kernels and execute()
 * defined in the JIT batch file. No template instantiation needed at call site.
 *
 * @tparam JitRunner The complete JitRunner struct (e.g., JitRunner_TCRec)
 * @tparam Pipeline Pipeline type containing source/dest specs
 */
template <typename JitRunner, typename Pipeline, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::JitExecutePipeline<JitRunner, Pipeline>, TailOps...>,
                           DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using GlobalIndexSpecs = typename Pipeline::source_specs_type;

    constexpr std::size_t kNumHandles = std::tuple_size_v<GlobalIndexSpecs>;

    if constexpr (kNumHandles == 0) {
      return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
    } else {
      // Call JitRunner::execute() directly - it's a non-template function
      // defined in the JIT batch file with complete __global__ kernels
      JitRunner::execute(db, static_cast<uint32_t>(iteration));

      auto result =
          GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
      GPU_DEVICE_SYNCHRONIZE();
      return result;
    }
  }
};

}  // namespace SRDatalog::GPU
