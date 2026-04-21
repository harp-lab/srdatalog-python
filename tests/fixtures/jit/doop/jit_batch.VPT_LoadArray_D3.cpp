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
// JIT-Generated Kernel Functor: VPT_LoadArray_D3
// Handles: 12
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_VPT_LoadArray_D3 {
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
        auto view_ArrayIndexPointsTo_0_1_DELTA_VER = views[0];
        auto view_VarPointsTo_0_1_FULL_VER = views[1];
        auto view_ArrayTypeCompat_0_1_FULL_VER = views[2];
        auto view_LoadArrayIndex_0_1_2_FULL_VER = views[4];
        auto view_Var_Type_0_1_FULL_VER = views[6];
        auto view_Reachable_0_FULL_VER = views[8];

        // Root ColumnJoin (multi-source intersection): bind 'baseheap' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var baseheap :sources ((ArrayIndexPointsTo :handle 0) (VarPointsTo :handle 1) (ArrayTypeCompat :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_ArrayIndexPointsTo_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_ArrayIndexPointsTo_0_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_ArrayIndexPointsTo_0_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_ArrayIndexPointsTo_0_1_DELTA_VER.num_rows_;
          auto h_ArrayIndexPointsTo_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_ArrayIndexPointsTo_0_1_DELTA_VER);
          if (!h_ArrayIndexPointsTo_0_root.valid()) continue;
          auto h_VarPointsTo_1_root = HandleType(0, view_VarPointsTo_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_VarPointsTo_0_1_FULL_VER);
          if (!h_VarPointsTo_1_root.valid()) continue;
          auto h_ArrayTypeCompat_2_root = HandleType(0, view_ArrayTypeCompat_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ArrayTypeCompat_0_1_FULL_VER);
          if (!h_ArrayTypeCompat_2_root.valid()) continue;
          auto baseheap = root_val_2;
        // Nested ColumnJoin (intersection): bind 'base' from 2 sources
        // MIR: (column-join :var base :sources ((VarPointsTo :handle 3 :prefix (baseheap)) (LoadArrayIndex :handle 4 :prefix ()) ))
        auto h_VarPointsTo_3_20 = h_VarPointsTo_1_root;
        auto h_LoadArrayIndex_4_21 = HandleType(0, view_LoadArrayIndex_0_1_2_FULL_VER.num_rows_, 0);
        auto intersect_22 = intersect_handles(tile, h_VarPointsTo_3_20.iterators(view_VarPointsTo_0_1_FULL_VER), h_LoadArrayIndex_4_21.iterators(view_LoadArrayIndex_0_1_2_FULL_VER));
        for (auto it_23 = intersect_22.begin(); it_23.valid(); it_23.next()) {
          auto base = it_23.value();
          auto positions = it_23.positions();
          auto ch_VarPointsTo_3_base = h_VarPointsTo_3_20.child_range(positions[0], base, tile, view_VarPointsTo_0_1_FULL_VER);
          auto ch_LoadArrayIndex_4_base = h_LoadArrayIndex_4_21.child_range(positions[1], base, tile, view_LoadArrayIndex_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'to' from 2 sources
        // MIR: (column-join :var to :sources ((LoadArrayIndex :handle 5 :prefix (base)) (Var_Type :handle 6 :prefix ()) ))
        auto h_LoadArrayIndex_5_16 = ch_LoadArrayIndex_4_base;
        auto h_Var_Type_6_17 = HandleType(0, view_Var_Type_0_1_FULL_VER.num_rows_, 0);
        auto intersect_18 = intersect_handles(tile, h_LoadArrayIndex_5_16.iterators(view_LoadArrayIndex_0_1_2_FULL_VER), h_Var_Type_6_17.iterators(view_Var_Type_0_1_FULL_VER));
        for (auto it_19 = intersect_18.begin(); it_19.valid(); it_19.next()) {
          auto to = it_19.value();
          auto positions = it_19.positions();
          auto ch_LoadArrayIndex_5_to = h_LoadArrayIndex_5_16.child_range(positions[0], to, tile, view_LoadArrayIndex_0_1_2_FULL_VER);
          auto ch_Var_Type_6_to = h_Var_Type_6_17.child_range(positions[1], to, tile, view_Var_Type_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'inmeth' from 2 sources
        // MIR: (column-join :var inmeth :sources ((LoadArrayIndex :handle 7 :prefix (base to)) (Reachable :handle 8 :prefix ()) ))
        auto h_LoadArrayIndex_7_12 = ch_LoadArrayIndex_5_to;
        auto h_Reachable_8_13 = HandleType(0, view_Reachable_0_FULL_VER.num_rows_, 0);
        auto intersect_14 = intersect_handles(tile, h_LoadArrayIndex_7_12.iterators(view_LoadArrayIndex_0_1_2_FULL_VER), h_Reachable_8_13.iterators(view_Reachable_0_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto inmeth = it_15.value();
          auto positions = it_15.positions();
          auto ch_LoadArrayIndex_7_inmeth = h_LoadArrayIndex_7_12.child_range(positions[0], inmeth, tile, view_LoadArrayIndex_0_1_2_FULL_VER);
          auto ch_Reachable_8_inmeth = h_Reachable_8_13.child_range(positions[1], inmeth, tile, view_Reachable_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'vtype' from 2 sources
        // MIR: (column-join :var vtype :sources ((Var_Type :handle 9 :prefix (to)) (ArrayTypeCompat :handle 10 :prefix (baseheap)) ))
        auto h_Var_Type_9_8 = ch_Var_Type_6_to;
        auto h_ArrayTypeCompat_10_9 = h_ArrayTypeCompat_2_root;
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

        auto h_ArrayIndexPointsTo_11_4 = h_ArrayIndexPointsTo_0_root;  // reusing narrowed handle

        if (!h_ArrayIndexPointsTo_11_4.valid()) continue;

        uint32_t degree_3 = h_ArrayIndexPointsTo_11_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto heap = view_ArrayIndexPointsTo_0_1_DELTA_VER.get_value(1, h_ArrayIndexPointsTo_11_4.begin() + idx0_7);

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
