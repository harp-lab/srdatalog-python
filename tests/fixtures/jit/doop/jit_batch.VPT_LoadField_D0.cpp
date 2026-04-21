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
// JIT-Generated Kernel Functor: VPT_LoadField_D0
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_VPT_LoadField_D0 {
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
        auto view_ReachableLoadInstanceField_0_1_2_DELTA_VER = views[0];
        auto view_VarPointsTo_1_0_FULL_VER = views[1];
        auto view_InstanceFieldPointsTo_1_2_0_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'base' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var base :sources ((ReachableLoadInstanceField :handle 0) (VarPointsTo :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_ReachableLoadInstanceField_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_ReachableLoadInstanceField_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_ReachableLoadInstanceField_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_ReachableLoadInstanceField_0_1_2_DELTA_VER.num_rows_;
          auto h_ReachableLoadInstanceField_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_ReachableLoadInstanceField_0_1_2_DELTA_VER);
          if (!h_ReachableLoadInstanceField_0_root.valid()) continue;
          auto h_VarPointsTo_1_root = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_VarPointsTo_1_0_FULL_VER);
          if (!h_VarPointsTo_1_root.valid()) continue;
          auto base = root_val_2;
        // Nested ColumnJoin (intersection): bind 'sig' from 2 sources
        // MIR: (column-join :var sig :sources ((ReachableLoadInstanceField :handle 2 :prefix (base)) (InstanceFieldPointsTo :handle 3 :prefix ()) ))
        auto h_ReachableLoadInstanceField_2_16 = h_ReachableLoadInstanceField_0_root;
        auto h_InstanceFieldPointsTo_3_17 = HandleType(0, view_InstanceFieldPointsTo_1_2_0_FULL_VER.num_rows_, 0);
        auto intersect_18 = intersect_handles(tile, h_ReachableLoadInstanceField_2_16.iterators(view_ReachableLoadInstanceField_0_1_2_DELTA_VER), h_InstanceFieldPointsTo_3_17.iterators(view_InstanceFieldPointsTo_1_2_0_FULL_VER));
        for (auto it_19 = intersect_18.begin(); it_19.valid(); it_19.next()) {
          auto sig = it_19.value();
          auto positions = it_19.positions();
          auto ch_ReachableLoadInstanceField_2_sig = h_ReachableLoadInstanceField_2_16.child_range(positions[0], sig, tile, view_ReachableLoadInstanceField_0_1_2_DELTA_VER);
          auto ch_InstanceFieldPointsTo_3_sig = h_InstanceFieldPointsTo_3_17.child_range(positions[1], sig, tile, view_InstanceFieldPointsTo_1_2_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'baseheap' from 2 sources
        // MIR: (column-join :var baseheap :sources ((VarPointsTo :handle 4 :prefix (base)) (InstanceFieldPointsTo :handle 5 :prefix (sig)) ))
        auto h_VarPointsTo_4_12 = h_VarPointsTo_1_root;
        auto h_InstanceFieldPointsTo_5_13 = ch_InstanceFieldPointsTo_3_sig;
        auto intersect_14 = intersect_handles(tile, h_VarPointsTo_4_12.iterators(view_VarPointsTo_1_0_FULL_VER), h_InstanceFieldPointsTo_5_13.iterators(view_InstanceFieldPointsTo_1_2_0_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto baseheap = it_15.value();
          auto positions = it_15.positions();
          auto ch_VarPointsTo_4_baseheap = h_VarPointsTo_4_12.child_range(positions[0], baseheap, tile, view_VarPointsTo_1_0_FULL_VER);
          auto ch_InstanceFieldPointsTo_5_baseheap = h_InstanceFieldPointsTo_5_13.child_range(positions[1], baseheap, tile, view_InstanceFieldPointsTo_1_2_0_FULL_VER);
        // Nested CartesianJoin: bind to, heap from 2 source(s)
        // MIR: (cartesian-join :vars (to heap) :sources ((ReachableLoadInstanceField :handle 6 :prefix (base sig)) (InstanceFieldPointsTo :handle 7 :prefix (sig baseheap)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_ReachableLoadInstanceField_6_4 = ch_ReachableLoadInstanceField_2_sig;  // reusing narrowed handle
        auto h_InstanceFieldPointsTo_7_6 = ch_InstanceFieldPointsTo_5_baseheap;  // reusing narrowed handle

        if (!h_ReachableLoadInstanceField_6_4.valid() || !h_InstanceFieldPointsTo_7_6.valid()) continue;

        uint32_t degree_3 = h_ReachableLoadInstanceField_6_4.degree();
        uint32_t degree_5 = h_InstanceFieldPointsTo_7_6.degree();
        uint32_t total_7 = degree_3 * degree_5;
        if (total_7 == 0) continue;

        for (uint32_t flat_idx_8 = lane_1; flat_idx_8 < total_7; flat_idx_8 += group_size_2) {
          const bool major_is_1_11 = (degree_5 >= degree_3);
          uint32_t idx0_9, idx1_10;
          if (major_is_1_11) {
            idx0_9 = flat_idx_8 / degree_5;
            idx1_10 = flat_idx_8 % degree_5;
          } else {
            idx1_10 = flat_idx_8 / degree_3;
            idx0_9 = flat_idx_8 % degree_3;
          }

          auto to = view_ReachableLoadInstanceField_0_1_2_DELTA_VER.get_value(2, h_ReachableLoadInstanceField_6_4.begin() + idx0_9);
          auto heap = view_InstanceFieldPointsTo_1_2_0_FULL_VER.get_value(2, h_InstanceFieldPointsTo_7_6.begin() + idx1_10);

        // Emit: VarPointsTo(heap, to)
        output.emit_direct(heap, to);
        }
        }
        }
        }
  }
};

// End of JIT batch file
