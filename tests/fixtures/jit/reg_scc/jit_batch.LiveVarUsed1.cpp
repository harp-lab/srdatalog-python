// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cooperative_groups.h>
#include <cstdint>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

// =============================================================
// JIT-Generated Kernel Functor: LiveVarUsed1
// Handles: 2
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_LiveVarUsed1 {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template <typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(Tile& tile, const Views* views,
                             const ValueType* __restrict__ root_unique_values,
                             uint32_t num_unique_root_keys, uint32_t num_root_keys,
                             uint32_t warp_id, uint32_t num_warps, Output& output) const {
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER = views[0];
    auto view_RegDefUseBlockLastDef_0_2_1_FULL_VER = views[1];

    // Root Scan: RegDefUseUsedInBlock binding blk, eaUsed, mvar, index
    // MIR: (scan :rel RegDefUseUsedInBlock :vars (blk eaUsed mvar index) :handle 0)
    auto root_handle_1 = HandleType(0, view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER.num_rows_, 0);
    if (!root_handle_1.valid())
      return;
    uint32_t degree_2 = root_handle_1.degree();

    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t idx_3 = warp_id; idx_3 < degree_2; idx_3 += num_warps) {
      auto blk = view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER.get_value(0, idx_3);
      auto eaUsed = view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER.get_value(1, idx_3);
      auto mvar = view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER.get_value(2, idx_3);
      auto index = view_RegDefUseUsedInBlock_0_1_2_3_FULL_VER.get_value(3, idx_3);
      // Negation: NOT EXISTS in RegDefUseBlockLastDef
      // MIR: (negation :rel RegDefUseBlockLastDef :prefix (eaUsed mvar) :handle 1)
      auto h_RegDefUseBlockLastDef_neg_1_1 =
          HandleType(0, view_RegDefUseBlockLastDef_0_2_1_FULL_VER.num_rows_, 0)
              .prefix(eaUsed, tile, view_RegDefUseBlockLastDef_0_2_1_FULL_VER)
              .prefix(mvar, tile, view_RegDefUseBlockLastDef_0_2_1_FULL_VER);
      if (!h_RegDefUseBlockLastDef_neg_1_1.valid()) {
        // Emit: RegDefUseLiveVarUsed(blk, mvar, eaUsed, index)
        if (tile.thread_rank() == 0)
          output.emit_direct(blk, mvar, eaUsed, index);
      }
    }
  }
};

// End of JIT batch file
