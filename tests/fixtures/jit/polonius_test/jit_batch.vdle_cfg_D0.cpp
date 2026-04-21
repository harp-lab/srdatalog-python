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
// JIT-Generated Kernel Functor: vdle_cfg_D0
// Handles: 7
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_vdle_cfg_D0 {
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

        // View declarations (deduplicated by spec, 4 unique views)
        auto view_var_drop_live_on_entry_0_1_DELTA_VER = views[0];
        auto view_var_maybe_partly_initialized_on_exit_0_1_FULL_VER = views[1];
        auto view_cfg_edge_1_0_FULL_VER = views[3];
        auto view_var_defined_at_0_1_FULL_VER = views[6];

        // Root ColumnJoin (multi-source intersection): bind 'v' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var v :sources ((var_drop_live_on_entry :handle 0) (var_maybe_partly_initialized_on_exit :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_var_drop_live_on_entry_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_var_drop_live_on_entry_0_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_var_drop_live_on_entry_0_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_var_drop_live_on_entry_0_1_DELTA_VER.num_rows_;
          auto h_var_drop_live_on_entry_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_var_drop_live_on_entry_0_1_DELTA_VER);
          if (!h_var_drop_live_on_entry_0_root.valid()) continue;
          auto h_var_maybe_partly_initialized_on_exit_1_root = HandleType(0, view_var_maybe_partly_initialized_on_exit_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_var_maybe_partly_initialized_on_exit_0_1_FULL_VER);
          if (!h_var_maybe_partly_initialized_on_exit_1_root.valid()) continue;
          auto v = root_val_2;
        // Nested ColumnJoin (intersection): bind 'tgt' from 2 sources
        // MIR: (column-join :var tgt :sources ((var_drop_live_on_entry :handle 2 :prefix (v)) (cfg_edge :handle 3 :prefix ()) ))
        auto h_var_drop_live_on_entry_2_6 = h_var_drop_live_on_entry_0_root;
        auto h_cfg_edge_3_7 = HandleType(0, view_cfg_edge_1_0_FULL_VER.num_rows_, 0);
        auto intersect_8 = intersect_handles(tile, h_var_drop_live_on_entry_2_6.iterators(view_var_drop_live_on_entry_0_1_DELTA_VER), h_cfg_edge_3_7.iterators(view_cfg_edge_1_0_FULL_VER));
        for (auto it_9 = intersect_8.begin(); it_9.valid(); it_9.next()) {
          auto tgt = it_9.value();
          auto positions = it_9.positions();
          auto ch_var_drop_live_on_entry_2_tgt = h_var_drop_live_on_entry_2_6.child_range(positions[0], tgt, tile, view_var_drop_live_on_entry_0_1_DELTA_VER);
          auto ch_cfg_edge_3_tgt = h_cfg_edge_3_7.child_range(positions[1], tgt, tile, view_cfg_edge_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'src' from 2 sources
        // MIR: (column-join :var src :sources ((cfg_edge :handle 4 :prefix (tgt)) (var_maybe_partly_initialized_on_exit :handle 5 :prefix (v)) ))
        auto h_cfg_edge_4_2 = ch_cfg_edge_3_tgt;
        auto h_var_maybe_partly_initialized_on_exit_5_3 = h_var_maybe_partly_initialized_on_exit_1_root;
        auto intersect_4 = intersect_handles(tile, h_cfg_edge_4_2.iterators(view_cfg_edge_1_0_FULL_VER), h_var_maybe_partly_initialized_on_exit_5_3.iterators(view_var_maybe_partly_initialized_on_exit_0_1_FULL_VER));
        for (auto it_5 = intersect_4.begin(); it_5.valid(); it_5.next()) {
          auto src = it_5.value();
          auto positions = it_5.positions();
          auto ch_cfg_edge_4_src = h_cfg_edge_4_2.child_range(positions[0], src, tile, view_cfg_edge_1_0_FULL_VER);
          auto ch_var_maybe_partly_initialized_on_exit_5_src = h_var_maybe_partly_initialized_on_exit_5_3.child_range(positions[1], src, tile, view_var_maybe_partly_initialized_on_exit_0_1_FULL_VER);
        // Negation: NOT EXISTS in var_defined_at
        // MIR: (negation :rel var_defined_at :prefix (v src) :handle 6)
        auto h_var_defined_at_neg_6_1 = HandleType(0, view_var_defined_at_0_1_FULL_VER.num_rows_, 0).prefix(v, tile, view_var_defined_at_0_1_FULL_VER).prefix(src, tile, view_var_defined_at_0_1_FULL_VER);
        if (!h_var_defined_at_neg_6_1.valid()) {
        // Emit: var_drop_live_on_entry(v, src)
        if (tile.thread_rank() == 0) output.emit_direct(v, src);
        }
        }
        }
        }
  }
};

// End of JIT batch file
