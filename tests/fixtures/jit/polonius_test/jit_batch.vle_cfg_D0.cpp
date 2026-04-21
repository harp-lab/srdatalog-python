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
// JIT-Generated Kernel Functor: vle_cfg_D0
// Handles: 5
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_vle_cfg_D0 {
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

        // View declarations (deduplicated by spec, 3 unique views)
        auto view_var_live_on_entry_1_0_DELTA_VER = views[0];
        auto view_cfg_edge_1_0_FULL_VER = views[1];
        auto view_var_defined_at_0_1_FULL_VER = views[4];

        // Root ColumnJoin (multi-source intersection): bind 'point2' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var point2 :sources ((var_live_on_entry :handle 0) (cfg_edge :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_var_live_on_entry_1_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_var_live_on_entry_1_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_var_live_on_entry_1_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_var_live_on_entry_1_0_DELTA_VER.num_rows_;
          auto h_var_live_on_entry_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_var_live_on_entry_1_0_DELTA_VER);
          if (!h_var_live_on_entry_0_root.valid()) continue;
          auto h_cfg_edge_1_root = HandleType(0, view_cfg_edge_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_cfg_edge_1_0_FULL_VER);
          if (!h_cfg_edge_1_root.valid()) continue;
          auto point2 = root_val_2;
        // Nested CartesianJoin: bind vr, point1 from 2 source(s)
        // MIR: (cartesian-join :vars (vr point1) :sources ((var_live_on_entry :handle 2 :prefix (point2)) (cfg_edge :handle 3 :prefix (point2)) ))
        uint32_t lane_2 = tile.thread_rank();
        uint32_t group_size_3 = tile.size();

        auto h_var_live_on_entry_2_5 = h_var_live_on_entry_0_root;  // reusing narrowed handle
        auto h_cfg_edge_3_7 = h_cfg_edge_1_root;  // reusing narrowed handle

        if (!h_var_live_on_entry_2_5.valid() || !h_cfg_edge_3_7.valid()) continue;

        uint32_t degree_4 = h_var_live_on_entry_2_5.degree();
        uint32_t degree_6 = h_cfg_edge_3_7.degree();
        uint32_t total_8 = degree_4 * degree_6;
        if (total_8 == 0) continue;

        for (uint32_t flat_idx_9 = lane_2; flat_idx_9 < total_8; flat_idx_9 += group_size_3) {
          const bool major_is_1_12 = (degree_6 >= degree_4);
          uint32_t idx0_10, idx1_11;
          if (major_is_1_12) {
            idx0_10 = flat_idx_9 / degree_6;
            idx1_11 = flat_idx_9 % degree_6;
          } else {
            idx1_11 = flat_idx_9 / degree_4;
            idx0_10 = flat_idx_9 % degree_4;
          }

          auto vr = view_var_live_on_entry_1_0_DELTA_VER.get_value(1, h_var_live_on_entry_2_5.begin() + idx0_10);
          auto point1 = view_cfg_edge_1_0_FULL_VER.get_value(1, h_cfg_edge_3_7.begin() + idx1_11);

        // Negation: NOT EXISTS in var_defined_at
        // MIR: (negation :rel var_defined_at :prefix (vr point1) :handle 4)
        auto h_var_defined_at_neg_4_1 = HandleType(0, view_var_defined_at_0_1_FULL_VER.num_rows_, 0).prefix_seq(vr, view_var_defined_at_0_1_FULL_VER).prefix_seq(point1, view_var_defined_at_0_1_FULL_VER);
        if (!h_var_defined_at_neg_4_1.valid()) {
        // Emit: var_live_on_entry(vr, point1)
        output.emit_direct(vr, point1);
        }
        }
        }
  }
};

// End of JIT batch file
