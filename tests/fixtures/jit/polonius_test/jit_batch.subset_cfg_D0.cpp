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
// JIT-Generated Kernel Functor: subset_cfg_D0
// Handles: 9
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_subset_cfg_D0 {
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
        auto view_subset_2_0_1_DELTA_VER = views[0];
        auto view_cfg_edge_0_1_FULL_VER = views[1];
        auto view_origin_live_on_entry_1_0_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'point1' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var point1 :sources ((subset :handle 0) (cfg_edge :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_subset_2_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_subset_2_0_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_subset_2_0_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_2_0_1_DELTA_VER.num_rows_;
          auto h_subset_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_2_0_1_DELTA_VER);
          if (!h_subset_0_root.valid()) continue;
          auto h_cfg_edge_1_root = HandleType(0, view_cfg_edge_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_cfg_edge_0_1_FULL_VER);
          if (!h_cfg_edge_1_root.valid()) continue;
          auto point1 = root_val_2;
        // Nested ColumnJoin (intersection): bind 'point2' from 3 sources
        // MIR: (column-join :var point2 :sources ((cfg_edge :handle 2 :prefix (point1)) (origin_live_on_entry :handle 3 :prefix ()) (origin_live_on_entry :handle 4 :prefix ()) ))
        auto h_cfg_edge_2_9 = h_cfg_edge_1_root;
        auto h_origin_live_on_entry_3_10 = HandleType(0, view_origin_live_on_entry_1_0_FULL_VER.num_rows_, 0);
        auto h_origin_live_on_entry_4_11 = HandleType(0, view_origin_live_on_entry_1_0_FULL_VER.num_rows_, 0);
        auto intersect_12 = intersect_handles(tile, h_cfg_edge_2_9.iterators(view_cfg_edge_0_1_FULL_VER), h_origin_live_on_entry_3_10.iterators(view_origin_live_on_entry_1_0_FULL_VER), h_origin_live_on_entry_4_11.iterators(view_origin_live_on_entry_1_0_FULL_VER));
        for (auto it_13 = intersect_12.begin(); it_13.valid(); it_13.next()) {
          auto point2 = it_13.value();
          auto positions = it_13.positions();
          auto ch_cfg_edge_2_point2 = h_cfg_edge_2_9.child_range(positions[0], point2, tile, view_cfg_edge_0_1_FULL_VER);
          auto ch_origin_live_on_entry_3_point2 = h_origin_live_on_entry_3_10.child_range(positions[1], point2, tile, view_origin_live_on_entry_1_0_FULL_VER);
          auto ch_origin_live_on_entry_4_point2 = h_origin_live_on_entry_4_11.child_range(positions[2], point2, tile, view_origin_live_on_entry_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'origin1' from 2 sources
        // MIR: (column-join :var origin1 :sources ((subset :handle 5 :prefix (point1)) (origin_live_on_entry :handle 6 :prefix (point2)) ))
        auto h_subset_5_5 = h_subset_0_root;
        auto h_origin_live_on_entry_6_6 = ch_origin_live_on_entry_4_point2;
        auto intersect_7 = intersect_handles(tile, h_subset_5_5.iterators(view_subset_2_0_1_DELTA_VER), h_origin_live_on_entry_6_6.iterators(view_origin_live_on_entry_1_0_FULL_VER));
        for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
          auto origin1 = it_8.value();
          auto positions = it_8.positions();
          auto ch_subset_5_origin1 = h_subset_5_5.child_range(positions[0], origin1, tile, view_subset_2_0_1_DELTA_VER);
          auto ch_origin_live_on_entry_6_origin1 = h_origin_live_on_entry_6_6.child_range(positions[1], origin1, tile, view_origin_live_on_entry_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'origin2' from 2 sources
        // MIR: (column-join :var origin2 :sources ((subset :handle 7 :prefix (point1 origin1)) (origin_live_on_entry :handle 8 :prefix (point2)) ))
        auto h_subset_7_1 = ch_subset_5_origin1;
        auto h_origin_live_on_entry_8_2 = ch_origin_live_on_entry_4_point2;
        auto intersect_3 = intersect_handles(tile, h_subset_7_1.iterators(view_subset_2_0_1_DELTA_VER), h_origin_live_on_entry_8_2.iterators(view_origin_live_on_entry_1_0_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto origin2 = it_4.value();
          auto positions = it_4.positions();
          auto ch_subset_7_origin2 = h_subset_7_1.child_range(positions[0], origin2, tile, view_subset_2_0_1_DELTA_VER);
          auto ch_origin_live_on_entry_8_origin2 = h_origin_live_on_entry_8_2.child_range(positions[1], origin2, tile, view_origin_live_on_entry_1_0_FULL_VER);
        // Emit: subset(origin1, origin2, point2)
        if (tile.thread_rank() == 0) output.emit_direct(origin1, origin2, point2);
        }
        }
        }
        }
  }
};

// End of JIT batch file
