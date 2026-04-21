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
// JIT-Generated Kernel Functor: VPT_LoadArray_D0
// Handles: 12
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_VPT_LoadArray_D0 {
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

        // View declarations (deduplicated by spec, 6 unique views)
        auto view_Reachable_0_DELTA_VER = views[0];
        auto view_LoadArrayIndex_2_0_1_FULL_VER = views[1];
        auto view_VarPointsTo_1_0_FULL_VER = views[3];
        auto view_Var_Type_0_1_FULL_VER = views[5];
        auto view_ArrayIndexPointsTo_0_1_FULL_VER = views[7];
        auto view_ArrayTypeCompat_0_1_FULL_VER = views[8];

        // Root ColumnJoin (multi-source intersection): bind 'inmeth' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var inmeth :sources ((Reachable :handle 0) (LoadArrayIndex :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Reachable_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Reachable_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          auto h_Reachable_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Reachable_0_DELTA_VER);
          if (!h_Reachable_0_root.valid()) continue;
          auto h_LoadArrayIndex_1_root = HandleType(0, view_LoadArrayIndex_2_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_LoadArrayIndex_2_0_1_FULL_VER);
          if (!h_LoadArrayIndex_1_root.valid()) continue;
          auto inmeth = root_val_2;
        // Nested ColumnJoin (intersection): bind 'base' from 2 sources
        // MIR: (column-join :var base :sources ((LoadArrayIndex :handle 2 :prefix (inmeth)) (VarPointsTo :handle 3 :prefix ()) ))
        auto h_LoadArrayIndex_2_21 = h_LoadArrayIndex_1_root;
        auto h_VarPointsTo_3_22 = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0);
        auto intersect_23 = intersect_handles(tile, h_LoadArrayIndex_2_21.iterators(view_LoadArrayIndex_2_0_1_FULL_VER), h_VarPointsTo_3_22.iterators(view_VarPointsTo_1_0_FULL_VER));
        for (auto it_24 = intersect_23.begin(); it_24.valid(); it_24.next()) {
          auto base = it_24.value();
          auto positions = it_24.positions();
          auto ch_LoadArrayIndex_2_base = h_LoadArrayIndex_2_21.child_range(positions[0], base, tile, view_LoadArrayIndex_2_0_1_FULL_VER);
          auto ch_VarPointsTo_3_base = h_VarPointsTo_3_22.child_range(positions[1], base, tile, view_VarPointsTo_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'to' from 2 sources
        // MIR: (column-join :var to :sources ((LoadArrayIndex :handle 4 :prefix (inmeth base)) (Var_Type :handle 5 :prefix ()) ))
        auto h_LoadArrayIndex_4_17 = ch_LoadArrayIndex_2_base;
        auto h_Var_Type_5_18 = HandleType(0, view_Var_Type_0_1_FULL_VER.num_rows_, 0);
        auto intersect_19 = intersect_handles(tile, h_LoadArrayIndex_4_17.iterators(view_LoadArrayIndex_2_0_1_FULL_VER), h_Var_Type_5_18.iterators(view_Var_Type_0_1_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto to = it_20.value();
          auto positions = it_20.positions();
          auto ch_LoadArrayIndex_4_to = h_LoadArrayIndex_4_17.child_range(positions[0], to, tile, view_LoadArrayIndex_2_0_1_FULL_VER);
          auto ch_Var_Type_5_to = h_Var_Type_5_18.child_range(positions[1], to, tile, view_Var_Type_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'baseheap' from 3 sources
        // MIR: (column-join :var baseheap :sources ((VarPointsTo :handle 6 :prefix (base)) (ArrayIndexPointsTo :handle 7 :prefix ()) (ArrayTypeCompat :handle 8 :prefix ()) ))
        auto h_VarPointsTo_6_12 = ch_VarPointsTo_3_base;
        auto h_ArrayIndexPointsTo_7_13 = HandleType(0, view_ArrayIndexPointsTo_0_1_FULL_VER.num_rows_, 0);
        auto h_ArrayTypeCompat_8_14 = HandleType(0, view_ArrayTypeCompat_0_1_FULL_VER.num_rows_, 0);
        auto intersect_15 = intersect_handles(tile, h_VarPointsTo_6_12.iterators(view_VarPointsTo_1_0_FULL_VER), h_ArrayIndexPointsTo_7_13.iterators(view_ArrayIndexPointsTo_0_1_FULL_VER), h_ArrayTypeCompat_8_14.iterators(view_ArrayTypeCompat_0_1_FULL_VER));
        for (auto it_16 = intersect_15.begin(); it_16.valid(); it_16.next()) {
          auto baseheap = it_16.value();
          auto positions = it_16.positions();
          auto ch_VarPointsTo_6_baseheap = h_VarPointsTo_6_12.child_range(positions[0], baseheap, tile, view_VarPointsTo_1_0_FULL_VER);
          auto ch_ArrayIndexPointsTo_7_baseheap = h_ArrayIndexPointsTo_7_13.child_range(positions[1], baseheap, tile, view_ArrayIndexPointsTo_0_1_FULL_VER);
          auto ch_ArrayTypeCompat_8_baseheap = h_ArrayTypeCompat_8_14.child_range(positions[2], baseheap, tile, view_ArrayTypeCompat_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'vtype' from 2 sources
        // MIR: (column-join :var vtype :sources ((Var_Type :handle 9 :prefix (to)) (ArrayTypeCompat :handle 10 :prefix (baseheap)) ))
        auto h_Var_Type_9_8 = ch_Var_Type_5_to;
        auto h_ArrayTypeCompat_10_9 = ch_ArrayTypeCompat_8_baseheap;
        auto intersect_10 = intersect_handles(tile, h_Var_Type_9_8.iterators(view_Var_Type_0_1_FULL_VER), h_ArrayTypeCompat_10_9.iterators(view_ArrayTypeCompat_0_1_FULL_VER));
        for (auto it_11 = intersect_10.begin(); it_11.valid(); it_11.next()) {
          auto vtype = it_11.value();
          auto positions = it_11.positions();
          auto ch_Var_Type_9_vtype = h_Var_Type_9_8.child_range(positions[0], vtype, tile, view_Var_Type_0_1_FULL_VER);
          auto ch_ArrayTypeCompat_10_vtype = h_ArrayTypeCompat_10_9.child_range(positions[1], vtype, tile, view_ArrayTypeCompat_0_1_FULL_VER);
        // Nested CartesianJoin: bind heap from 1 source(s)
        // MIR: (cartesian-join :vars (heap) :sources ((ArrayIndexPointsTo :handle 11 :prefix (baseheap)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_ArrayIndexPointsTo_11_4 = ch_ArrayIndexPointsTo_7_baseheap;  // reusing narrowed handle

        if (!h_ArrayIndexPointsTo_11_4.valid()) continue;

        uint32_t degree_3 = h_ArrayIndexPointsTo_11_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto heap = view_ArrayIndexPointsTo_0_1_FULL_VER.get_value(1, h_ArrayIndexPointsTo_11_4.begin() + idx0_7);

        // Emit: VarPointsTo(heap, to)
        output.emit_direct(heap, to);
        }
        }
        }
        }
        }
        }
  }
};

// End of JIT batch file
