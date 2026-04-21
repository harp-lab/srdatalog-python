#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include "gpu/macro.h"    // For GPU_DEFAULT_GROUP_SIZE
#include <iostream>
// NVTX profiling support
#if defined(USE_ROCm)
// nvtx3 is not available on ROCm - provide stubs
#ifndef NVTX_STUB_DEFINED
#define NVTX_STUB_DEFINED
#define nvtxRangePushA(name) ((void)0)
#define nvtxRangePop() ((void)0)
namespace nvtx3 {
struct scoped_range {
  scoped_range(const char*) {}
};
}  // namespace nvtx3
#endif
#else
// Always include nvtx3 on CUDA builds
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>
#endif
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../../logging.h"
#include "../../mir.h"
#include "../../query.h"  // For get_relation_by_schema
#include "../../runtime/executor.h"
#include "../../runtime/state.h"
#include "../../type_name.h"
#include "../device_sorted_array_index.h"
#include "runtime.h"

#include "./custom_set_op.h"

namespace SRDatalog::GPU {
namespace mp = boost::mp11;

/**
 * @brief GPU Fixpoint Executor template.
 *
 * @details Executes GPU fixpoint plan operations on a SemiNaiveDatabase.
 * Unlike the previous design that used a RelationsBundle tuple, this directly
 * takes the database and uses get_relation_by_schema<Schema, Ver>(db) for access.
 *
 * @tparam Plan The tuple of MIR instructions to execute
 * @tparam DB The SemiNaiveDatabase type
 */
template <typename Plan, typename DB>
struct GPUFixpointExecutor;

// Base case: empty plan
template <typename DB>
struct GPUFixpointExecutor<std::tuple<>, DB> {
  static bool execute(DB&, std::size_t, std::size_t) {
    // Done - all instructions processed
    return false;
  }
};

}  // namespace SRDatalog::GPU
