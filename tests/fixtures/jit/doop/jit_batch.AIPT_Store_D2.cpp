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
// JIT-Generated Kernel Functor: AIPT_Store_D2
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_AIPT_Store_D2 {
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
        auto view_VarPointsTo_0_1_DELTA_VER = views[0];
        auto view_HeapAllocSuperType_0_1_FULL_VER = views[1];
        auto view_ReachableSortedIndex_0_1_FULL_VER = views[3];
        auto view_VarPointsTo_1_0_FULL_VER = views[5];

        // Root ColumnJoin (multi-source intersection): bind 'heap' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var heap :sources ((VarPointsTo :handle 0) (HeapAllocSuperType :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_VarPointsTo_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_VarPointsTo_0_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_VarPointsTo_0_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_VarPointsTo_0_1_DELTA_VER.num_rows_;
          auto h_VarPointsTo_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_VarPointsTo_0_1_DELTA_VER);
          if (!h_VarPointsTo_0_root.valid()) continue;
          auto h_HeapAllocSuperType_1_root = HandleType(0, view_HeapAllocSuperType_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_HeapAllocSuperType_0_1_FULL_VER);
          if (!h_HeapAllocSuperType_1_root.valid()) continue;
          auto heap = root_val_2;
        // Nested ColumnJoin (intersection): bind 'frm' from 2 sources
        // MIR: (column-join :var frm :sources ((VarPointsTo :handle 2 :prefix (heap)) (ReachableSortedIndex :handle 3 :prefix ()) ))
        auto h_VarPointsTo_2_9 = h_VarPointsTo_0_root;
        auto h_ReachableSortedIndex_3_10 = HandleType(0, view_ReachableSortedIndex_0_1_FULL_VER.num_rows_, 0);
        auto intersect_11 = intersect_handles(tile, h_VarPointsTo_2_9.iterators(view_VarPointsTo_0_1_DELTA_VER), h_ReachableSortedIndex_3_10.iterators(view_ReachableSortedIndex_0_1_FULL_VER));
        for (auto it_12 = intersect_11.begin(); it_12.valid(); it_12.next()) {
          auto frm = it_12.value();
          auto positions = it_12.positions();
          auto ch_VarPointsTo_2_frm = h_VarPointsTo_2_9.child_range(positions[0], frm, tile, view_VarPointsTo_0_1_DELTA_VER);
          auto ch_ReachableSortedIndex_3_frm = h_ReachableSortedIndex_3_10.child_range(positions[1], frm, tile, view_ReachableSortedIndex_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'base' from 2 sources
        // MIR: (column-join :var base :sources ((ReachableSortedIndex :handle 4 :prefix (frm)) (VarPointsTo :handle 5 :prefix ()) ))
        auto h_ReachableSortedIndex_4_5 = ch_ReachableSortedIndex_3_frm;
        auto h_VarPointsTo_5_6 = HandleType(0, view_VarPointsTo_1_0_FULL_VER.num_rows_, 0);
        auto intersect_7 = intersect_handles(tile, h_ReachableSortedIndex_4_5.iterators(view_ReachableSortedIndex_0_1_FULL_VER), h_VarPointsTo_5_6.iterators(view_VarPointsTo_1_0_FULL_VER));
        for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
          auto base = it_8.value();
          auto positions = it_8.positions();
          auto ch_ReachableSortedIndex_4_base = h_ReachableSortedIndex_4_5.child_range(positions[0], base, tile, view_ReachableSortedIndex_0_1_FULL_VER);
          auto ch_VarPointsTo_5_base = h_VarPointsTo_5_6.child_range(positions[1], base, tile, view_VarPointsTo_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'baseheap' from 2 sources
        // MIR: (column-join :var baseheap :sources ((VarPointsTo :handle 6 :prefix (base)) (HeapAllocSuperType :handle 7 :prefix (heap)) ))
        auto h_VarPointsTo_6_1 = ch_VarPointsTo_5_base;
        auto h_HeapAllocSuperType_7_2 = h_HeapAllocSuperType_1_root;
        auto intersect_3 = intersect_handles(tile, h_VarPointsTo_6_1.iterators(view_VarPointsTo_1_0_FULL_VER), h_HeapAllocSuperType_7_2.iterators(view_HeapAllocSuperType_0_1_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto baseheap = it_4.value();
          auto positions = it_4.positions();
          auto ch_VarPointsTo_6_baseheap = h_VarPointsTo_6_1.child_range(positions[0], baseheap, tile, view_VarPointsTo_1_0_FULL_VER);
          auto ch_HeapAllocSuperType_7_baseheap = h_HeapAllocSuperType_7_2.child_range(positions[1], baseheap, tile, view_HeapAllocSuperType_0_1_FULL_VER);
        // Emit: ArrayIndexPointsTo(baseheap, heap)
        if (tile.thread_rank() == 0) output.emit_direct(baseheap, heap);
        }
        }
        }
        }
  }
};

// End of JIT batch file
