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
// JIT-Generated Kernel Functor: vdle_dropped
// Handles: 4
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_vdle_dropped {
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
        auto view_var_dropped_at_0_1_FULL_VER = views[0];
        auto view_var_maybe_partly_initialized_on_entry_0_1_FULL_VER = views[1];

        // Root ColumnJoin (multi-source intersection): bind 'vr' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var vr :sources ((var_dropped_at :handle 0) (var_maybe_partly_initialized_on_entry :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_var_dropped_at_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_var_dropped_at_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_var_dropped_at_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_var_dropped_at_0_1_FULL_VER.num_rows_;
          auto h_var_dropped_at_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_var_dropped_at_0_1_FULL_VER);
          if (!h_var_dropped_at_0_root.valid()) continue;
          auto h_var_maybe_partly_initialized_on_entry_1_root = HandleType(0, view_var_maybe_partly_initialized_on_entry_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_var_maybe_partly_initialized_on_entry_0_1_FULL_VER);
          if (!h_var_maybe_partly_initialized_on_entry_1_root.valid()) continue;
          auto vr = root_val_2;
        // Nested ColumnJoin (intersection): bind 'point' from 2 sources
        // MIR: (column-join :var point :sources ((var_dropped_at :handle 2 :prefix (vr)) (var_maybe_partly_initialized_on_entry :handle 3 :prefix (vr)) ))
        auto h_var_dropped_at_2_1 = h_var_dropped_at_0_root;
        auto h_var_maybe_partly_initialized_on_entry_3_2 = h_var_maybe_partly_initialized_on_entry_1_root;
        auto intersect_3 = intersect_handles(tile, h_var_dropped_at_2_1.iterators(view_var_dropped_at_0_1_FULL_VER), h_var_maybe_partly_initialized_on_entry_3_2.iterators(view_var_maybe_partly_initialized_on_entry_0_1_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto point = it_4.value();
          auto positions = it_4.positions();
          auto ch_var_dropped_at_2_point = h_var_dropped_at_2_1.child_range(positions[0], point, tile, view_var_dropped_at_0_1_FULL_VER);
          auto ch_var_maybe_partly_initialized_on_entry_3_point = h_var_maybe_partly_initialized_on_entry_3_2.child_range(positions[1], point, tile, view_var_maybe_partly_initialized_on_entry_0_1_FULL_VER);
        // Emit: var_drop_live_on_entry(vr, point)
        if (tile.thread_rank() == 0) output.emit_direct(vr, point);
        }
        }
  }
};

// End of JIT batch file
