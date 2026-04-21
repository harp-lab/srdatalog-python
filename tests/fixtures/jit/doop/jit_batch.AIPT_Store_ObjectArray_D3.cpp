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
// JIT-Generated Kernel Functor: AIPT_Store_ObjectArray_D3
// Handles: 7
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_AIPT_Store_ObjectArray_D3 {
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
        auto view_VarPointsTo_1_0_DELTA_VER = views[0];
        auto view_ReachableSortedIndex_0_1_FULL_VER = views[1];
        auto view_VarPointsTo_1_0_FULL_VER = views[3];
        auto view_IsObjectArrayHeap_0_FULL_VER = views[5];

        // Root ColumnJoin (multi-source intersection): bind 'frm' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var frm :sources ((VarPointsTo :handle 0) (ReachableSortedIndex :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_VarPointsTo_1_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_VarPointsTo_1_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_VarPointsTo_1_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_VarPointsTo_1_0_DELTA_VER.num_rows_;
          auto h_VarPointsTo_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_VarPointsTo_1_0_DELTA_VER);
          if (!h_VarPointsTo_0_root.valid()) continue;
          auto h_ReachableSortedIndex_1_root = HandleType(0, view_ReachableSortedIndex_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ReachableSortedIndex_0_1_FULL_VER);
          if (!h_ReachableSortedIndex_1_root.valid()) continue;
          auto frm = root_val_2;
        // Nested ColumnJoin (intersection): bind 'base' from 2 sources
        // MIR: (column-join :var base :sources ((ReachableSortedIndex :handle 2 :prefix (frm)) (VarPointsTo :handle 3 :prefix ()) ))
        auto h_ReachableSortedIndex_2_12 = h_ReachableSortedIndex_1_root;
        auto h_VarPointsTo_3_13 = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0);
        auto intersect_14 = intersect_handles(tile, h_ReachableSortedIndex_2_12.iterators(view_ReachableSortedIndex_0_1_FULL_VER), h_VarPointsTo_3_13.iterators(view_VarPointsTo_1_0_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto base = it_15.value();
          auto positions = it_15.positions();
          auto ch_ReachableSortedIndex_2_base = h_ReachableSortedIndex_2_12.child_range(positions[0], base, tile, view_ReachableSortedIndex_0_1_FULL_VER);
          auto ch_VarPointsTo_3_base = h_VarPointsTo_3_13.child_range(positions[1], base, tile, view_VarPointsTo_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'baseheap' from 2 sources
        // MIR: (column-join :var baseheap :sources ((VarPointsTo :handle 4 :prefix (base)) (IsObjectArrayHeap :handle 5 :prefix ()) ))
        auto h_VarPointsTo_4_8 = ch_VarPointsTo_3_base;
        auto h_IsObjectArrayHeap_5_9 = HandleType(0, view_IsObjectArrayHeap_0_FULL_VER.num_rows_, 0);
        auto intersect_10 = intersect_handles(tile, h_VarPointsTo_4_8.iterators(view_VarPointsTo_1_0_FULL_VER), h_IsObjectArrayHeap_5_9.iterators(view_IsObjectArrayHeap_0_FULL_VER));
        for (auto it_11 = intersect_10.begin(); it_11.valid(); it_11.next()) {
          auto baseheap = it_11.value();
          auto positions = it_11.positions();
          auto ch_VarPointsTo_4_baseheap = h_VarPointsTo_4_8.child_range(positions[0], baseheap, tile, view_VarPointsTo_1_0_FULL_VER);
          auto ch_IsObjectArrayHeap_5_baseheap = h_IsObjectArrayHeap_5_9.child_range(positions[1], baseheap, tile, view_IsObjectArrayHeap_0_FULL_VER);
        // Nested CartesianJoin: bind heap from 1 source(s)
        // MIR: (cartesian-join :vars (heap) :sources ((VarPointsTo :handle 6 :prefix (frm)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_VarPointsTo_6_4 = h_VarPointsTo_0_root;  // reusing narrowed handle

        if (!h_VarPointsTo_6_4.valid()) continue;

        uint32_t degree_3 = h_VarPointsTo_6_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto heap = view_VarPointsTo_1_0_DELTA_VER.get_value(1, h_VarPointsTo_6_4.begin() + idx0_7);

        // Emit: ArrayIndexPointsTo(baseheap, heap)
        output.emit_direct(baseheap, heap);
        }
        }
        }
        }
  }
};

// End of JIT batch file
