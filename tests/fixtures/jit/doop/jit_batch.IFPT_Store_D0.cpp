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
// JIT-Generated Kernel Functor: IFPT_Store_D0
// Handles: 9
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_IFPT_Store_D0 {
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
        auto view_Reachable_0_DELTA_VER = views[0];
        auto view_StoreInstanceField_3_0_1_2_FULL_VER = views[1];
        auto view_VarPointsTo_1_0_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'inmeth' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var inmeth :sources ((Reachable :handle 0) (StoreInstanceField :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Reachable_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Reachable_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          auto h_Reachable_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Reachable_0_DELTA_VER);
          if (!h_Reachable_0_root.valid()) continue;
          auto h_StoreInstanceField_1_root = HandleType(0, view_StoreInstanceField_3_0_1_2_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StoreInstanceField_3_0_1_2_FULL_VER);
          if (!h_StoreInstanceField_1_root.valid()) continue;
          auto inmeth = root_val_2;
        // Nested ColumnJoin (intersection): bind 'frm' from 2 sources
        // MIR: (column-join :var frm :sources ((StoreInstanceField :handle 2 :prefix (inmeth)) (VarPointsTo :handle 3 :prefix ()) ))
        auto h_StoreInstanceField_2_18 = h_StoreInstanceField_1_root;
        auto h_VarPointsTo_3_19 = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0);
        auto intersect_20 = intersect_handles(tile, h_StoreInstanceField_2_18.iterators(view_StoreInstanceField_3_0_1_2_FULL_VER), h_VarPointsTo_3_19.iterators(view_VarPointsTo_1_0_FULL_VER));
        for (auto it_21 = intersect_20.begin(); it_21.valid(); it_21.next()) {
          auto frm = it_21.value();
          auto positions = it_21.positions();
          auto ch_StoreInstanceField_2_frm = h_StoreInstanceField_2_18.child_range(positions[0], frm, tile, view_StoreInstanceField_3_0_1_2_FULL_VER);
          auto ch_VarPointsTo_3_frm = h_VarPointsTo_3_19.child_range(positions[1], frm, tile, view_VarPointsTo_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'base' from 2 sources
        // MIR: (column-join :var base :sources ((StoreInstanceField :handle 4 :prefix (inmeth frm)) (VarPointsTo :handle 5 :prefix ()) ))
        auto h_StoreInstanceField_4_14 = ch_StoreInstanceField_2_frm;
        auto h_VarPointsTo_5_15 = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0);
        auto intersect_16 = intersect_handles(tile, h_StoreInstanceField_4_14.iterators(view_StoreInstanceField_3_0_1_2_FULL_VER), h_VarPointsTo_5_15.iterators(view_VarPointsTo_1_0_FULL_VER));
        for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
          auto base = it_17.value();
          auto positions = it_17.positions();
          auto ch_StoreInstanceField_4_base = h_StoreInstanceField_4_14.child_range(positions[0], base, tile, view_StoreInstanceField_3_0_1_2_FULL_VER);
          auto ch_VarPointsTo_5_base = h_VarPointsTo_5_15.child_range(positions[1], base, tile, view_VarPointsTo_1_0_FULL_VER);
        // Nested CartesianJoin: bind fld, heap, baseheap from 3 source(s)
        // MIR: (cartesian-join :vars (fld heap baseheap) :sources ((StoreInstanceField :handle 6 :prefix (inmeth frm base)) (VarPointsTo :handle 7 :prefix (frm)) (VarPointsTo :handle 8 :prefix (base)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_StoreInstanceField_6_4 = ch_StoreInstanceField_4_base;  // reusing narrowed handle
        auto h_VarPointsTo_7_6 = ch_VarPointsTo_3_frm;  // reusing narrowed handle
        auto h_VarPointsTo_8_8 = ch_VarPointsTo_5_base;  // reusing narrowed handle

        if (!h_StoreInstanceField_6_4.valid() || !h_VarPointsTo_7_6.valid() || !h_VarPointsTo_8_8.valid()) continue;

        uint32_t degree_3 = h_StoreInstanceField_6_4.degree();
        uint32_t degree_5 = h_VarPointsTo_7_6.degree();
        uint32_t degree_7 = h_VarPointsTo_8_8.degree();
        uint32_t total_9 = degree_3 * degree_5 * degree_7;
        if (total_9 == 0) continue;

        for (uint32_t flat_idx_10 = lane_1; flat_idx_10 < total_9; flat_idx_10 += group_size_2) {
          uint32_t remaining = flat_idx_10;
          uint32_t idx2_13 = remaining % degree_7;
          remaining /= degree_7;
          uint32_t idx1_12 = remaining % degree_5;
          remaining /= degree_5;
          uint32_t idx0_11 = remaining % degree_3;

          auto fld = view_StoreInstanceField_3_0_1_2_FULL_VER.get_value(3, h_StoreInstanceField_6_4.begin() + idx0_11);
          auto heap = view_VarPointsTo_1_0_FULL_VER.get_value(1, h_VarPointsTo_7_6.begin() + idx1_12);
          auto baseheap = view_VarPointsTo_1_0_FULL_VER.get_value(1, h_VarPointsTo_8_8.begin() + idx2_13);

        // Emit: InstanceFieldPointsTo(heap, fld, baseheap)
        output.emit_direct(heap, fld, baseheap);
        }
        }
        }
        }
  }
};

// End of JIT batch file
