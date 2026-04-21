// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cstdint>
#include <cooperative_groups.h>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

// =============================================================
// JIT-Generated Kernel Functor: Knows2Load
// Handles: 1
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Knows2Load {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template<typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(
      Tile& tile,
      const Views* views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,
      uint32_t warp_id,
      uint32_t num_warps,
      Output& output
  ) const {
        using ViewType = std::remove_cvref_t<decltype(views[0])>;
        using HandleType = ViewType::NodeHandle;

        // View declarations (deduplicated by spec, 1 unique views)
        auto view_KnowsInput2_0_1_FULL_VER = views[0];

        // Root Scan: KnowsInput2 binding x, y
        // MIR: (scan :rel KnowsInput2 :vars (x y) :handle 0)
        auto root_handle_1 = HandleType(0, view_KnowsInput2_0_1_FULL_VER.num_rows_, 0);
        if (!root_handle_1.valid()) return;
        uint32_t degree_2 = root_handle_1.degree();

        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t idx_3 = warp_id; idx_3 < degree_2; idx_3 += num_warps) {
          auto x = view_KnowsInput2_0_1_FULL_VER.get_value(0, idx_3);
          auto y = view_KnowsInput2_0_1_FULL_VER.get_value(1, idx_3);
        // Emit: Knows2(x, y)
        if (tile.thread_rank() == 0) output.emit_direct(x, y);
        }
  }
};

// End of JIT batch file
