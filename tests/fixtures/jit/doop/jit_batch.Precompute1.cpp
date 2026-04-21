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
// JIT-Generated Kernel Functor: Precompute1
// Handles: 7
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Precompute1 {
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
        auto view_MethodLookup_2_3_0_1_FULL_VER = views[0];
        auto view_HeapAllocation_Type_1_0_FULL_VER = views[1];
        auto view_ThisVar_0_1_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'heaptype' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var heaptype :sources ((MethodLookup :handle 0) (HeapAllocation_Type :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_MethodLookup_2_3_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_MethodLookup_2_3_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_MethodLookup_2_3_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_MethodLookup_2_3_0_1_FULL_VER.num_rows_;
          auto h_MethodLookup_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_MethodLookup_2_3_0_1_FULL_VER);
          if (!h_MethodLookup_0_root.valid()) continue;
          auto h_HeapAllocation_Type_1_root = HandleType(0, view_HeapAllocation_Type_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_HeapAllocation_Type_1_0_FULL_VER);
          if (!h_HeapAllocation_Type_1_root.valid()) continue;
          auto heaptype = root_val_2;
        // Nested ColumnJoin (intersection): bind 'toMeth' from 2 sources
        // MIR: (column-join :var toMeth :sources ((MethodLookup :handle 2 :prefix (heaptype)) (ThisVar :handle 3 :prefix ()) ))
        auto h_MethodLookup_2_14 = h_MethodLookup_0_root;
        auto h_ThisVar_3_15 = HandleType(0, view_ThisVar_0_1_FULL_VER.num_rows_, 0);
        auto intersect_16 = intersect_handles(tile, h_MethodLookup_2_14.iterators(view_MethodLookup_2_3_0_1_FULL_VER), h_ThisVar_3_15.iterators(view_ThisVar_0_1_FULL_VER));
        for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
          auto toMeth = it_17.value();
          auto positions = it_17.positions();
          auto ch_MethodLookup_2_toMeth = h_MethodLookup_2_14.child_range(positions[0], toMeth, tile, view_MethodLookup_2_3_0_1_FULL_VER);
          auto ch_ThisVar_3_toMeth = h_ThisVar_3_15.child_range(positions[1], toMeth, tile, view_ThisVar_0_1_FULL_VER);
        // Nested CartesianJoin: bind simplename, descriptor, heap, thisP from 3 source(s)
        // MIR: (cartesian-join :vars (simplename descriptor heap thisP) :sources ((MethodLookup :handle 4 :prefix (heaptype toMeth)) (HeapAllocation_Type :handle 5 :prefix (heaptype)) (ThisVar :handle 6 :prefix (toMeth)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_MethodLookup_4_4 = ch_MethodLookup_2_toMeth;  // reusing narrowed handle
        auto h_HeapAllocation_Type_5_6 = h_HeapAllocation_Type_1_root;  // reusing narrowed handle
        auto h_ThisVar_6_8 = ch_ThisVar_3_toMeth;  // reusing narrowed handle

        if (!h_MethodLookup_4_4.valid() || !h_HeapAllocation_Type_5_6.valid() || !h_ThisVar_6_8.valid()) continue;

        uint32_t degree_3 = h_MethodLookup_4_4.degree();
        uint32_t degree_5 = h_HeapAllocation_Type_5_6.degree();
        uint32_t degree_7 = h_ThisVar_6_8.degree();
        uint32_t total_9 = degree_3 * degree_5 * degree_7;
        if (total_9 == 0) continue;

        for (uint32_t flat_idx_10 = lane_1; flat_idx_10 < total_9; flat_idx_10 += group_size_2) {
          uint32_t remaining = flat_idx_10;
          uint32_t idx2_13 = remaining % degree_7;
          remaining /= degree_7;
          uint32_t idx1_12 = remaining % degree_5;
          remaining /= degree_5;
          uint32_t idx0_11 = remaining % degree_3;

          auto simplename = view_MethodLookup_2_3_0_1_FULL_VER.get_value(2, h_MethodLookup_4_4.begin() + idx0_11);
          auto descriptor = view_MethodLookup_2_3_0_1_FULL_VER.get_value(3, h_MethodLookup_4_4.begin() + idx0_11);
          auto heap = view_HeapAllocation_Type_1_0_FULL_VER.get_value(1, h_HeapAllocation_Type_5_6.begin() + idx1_12);
          auto thisP = view_ThisVar_0_1_FULL_VER.get_value(1, h_ThisVar_6_8.begin() + idx2_13);

        // Emit: HeapHelper(simplename, descriptor, toMeth, thisP, heap)
        output.emit_direct(simplename, descriptor, toMeth, thisP, heap);
        }
        }
        }
  }
};

// End of JIT batch file
