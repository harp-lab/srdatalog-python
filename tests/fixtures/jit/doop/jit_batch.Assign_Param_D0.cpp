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
// JIT-Generated Kernel Functor: Assign_Param_D0
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Assign_Param_D0 {
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
        auto view_CallGraphEdge_0_1_DELTA_VER = views[0];
        auto view_ActualParam_1_0_2_FULL_VER = views[1];
        auto view_FormalParam_1_0_2_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'invocation' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var invocation :sources ((CallGraphEdge :handle 0) (ActualParam :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_CallGraphEdge_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_CallGraphEdge_0_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_CallGraphEdge_0_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_CallGraphEdge_0_1_DELTA_VER.num_rows_;
          auto h_CallGraphEdge_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_CallGraphEdge_0_1_DELTA_VER);
          if (!h_CallGraphEdge_0_root.valid()) continue;
          auto h_ActualParam_1_root = HandleType(0, view_ActualParam_1_0_2_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ActualParam_1_0_2_FULL_VER);
          if (!h_ActualParam_1_root.valid()) continue;
          auto invocation = root_val_2;
        // Nested ColumnJoin (intersection): bind 'meth' from 2 sources
        // MIR: (column-join :var meth :sources ((CallGraphEdge :handle 2 :prefix (invocation)) (FormalParam :handle 3 :prefix ()) ))
        auto h_CallGraphEdge_2_16 = h_CallGraphEdge_0_root;
        auto h_FormalParam_3_17 = HandleType(0, view_FormalParam_1_0_2_FULL_VER.num_rows_, 0);
        auto intersect_18 = intersect_handles(tile, h_CallGraphEdge_2_16.iterators(view_CallGraphEdge_0_1_DELTA_VER), h_FormalParam_3_17.iterators(view_FormalParam_1_0_2_FULL_VER));
        for (auto it_19 = intersect_18.begin(); it_19.valid(); it_19.next()) {
          auto meth = it_19.value();
          auto positions = it_19.positions();
          auto ch_CallGraphEdge_2_meth = h_CallGraphEdge_2_16.child_range(positions[0], meth, tile, view_CallGraphEdge_0_1_DELTA_VER);
          auto ch_FormalParam_3_meth = h_FormalParam_3_17.child_range(positions[1], meth, tile, view_FormalParam_1_0_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'idx' from 2 sources
        // MIR: (column-join :var idx :sources ((FormalParam :handle 4 :prefix (meth)) (ActualParam :handle 5 :prefix (invocation)) ))
        auto h_FormalParam_4_12 = ch_FormalParam_3_meth;
        auto h_ActualParam_5_13 = h_ActualParam_1_root;
        auto intersect_14 = intersect_handles(tile, h_FormalParam_4_12.iterators(view_FormalParam_1_0_2_FULL_VER), h_ActualParam_5_13.iterators(view_ActualParam_1_0_2_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto idx = it_15.value();
          auto positions = it_15.positions();
          auto ch_FormalParam_4_idx = h_FormalParam_4_12.child_range(positions[0], idx, tile, view_FormalParam_1_0_2_FULL_VER);
          auto ch_ActualParam_5_idx = h_ActualParam_5_13.child_range(positions[1], idx, tile, view_ActualParam_1_0_2_FULL_VER);
        // Nested CartesianJoin: bind formal, actual from 2 source(s)
        // MIR: (cartesian-join :vars (formal actual) :sources ((FormalParam :handle 6 :prefix (meth idx)) (ActualParam :handle 7 :prefix (invocation idx)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_FormalParam_6_4 = ch_FormalParam_4_idx;  // reusing narrowed handle
        auto h_ActualParam_7_6 = ch_ActualParam_5_idx;  // reusing narrowed handle

        if (!h_FormalParam_6_4.valid() || !h_ActualParam_7_6.valid()) continue;

        uint32_t degree_3 = h_FormalParam_6_4.degree();
        uint32_t degree_5 = h_ActualParam_7_6.degree();
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

          auto formal = view_FormalParam_1_0_2_FULL_VER.get_value(2, h_FormalParam_6_4.begin() + idx0_9);
          auto actual = view_ActualParam_1_0_2_FULL_VER.get_value(2, h_ActualParam_7_6.begin() + idx1_10);

        // Emit: Assign(actual, formal)
        output.emit_direct(actual, formal);
        }
        }
        }
        }
  }
};

// End of JIT batch file
