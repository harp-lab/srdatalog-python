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
// JIT-Generated Kernel Functor: ole_universal
// Handles: 2
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_ole_universal {
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

        // View declarations (deduplicated by spec, 2 unique views)
        auto view_cfg_node_0_FULL_VER = views[0];
        auto view_universal_region_0_FULL_VER = views[1];

        // Root CartesianJoin: bind point, origin from 2 source(s)
        // MIR: (cartesian-join :vars (point origin) :sources ((cfg_node :handle 0) (universal_region :handle 1) ))
        auto h_cfg_node_0_1 = HandleType(0, view_cfg_node_0_FULL_VER.num_rows_, 0);
        auto h_universal_region_1_3 = HandleType(0, view_universal_region_0_FULL_VER.num_rows_, 0);

        if (!h_cfg_node_0_1.valid() || !h_universal_region_1_3.valid()) return;

        uint32_t degree_2 = h_cfg_node_0_1.degree();
        uint32_t degree_4 = h_universal_region_1_3.degree();
        uint32_t total_5 = degree_2 * degree_4;
        if (total_5 == 0) return;

        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t flat_idx_6 = warp_id; flat_idx_6 < total_5; flat_idx_6 += num_warps) {
          uint32_t idx0_7 = flat_idx_6 / degree_4;
          uint32_t idx1_8 = flat_idx_6 % degree_4;

          auto point = h_cfg_node_0_1.get_value_at(view_cfg_node_0_FULL_VER, idx0_7);
          auto origin = h_universal_region_1_3.get_value_at(view_universal_region_0_FULL_VER, idx1_8);

        // Emit: origin_live_on_entry(origin, point)
        output.emit_direct(origin, point);
        }
  }
};

// End of JIT batch file
