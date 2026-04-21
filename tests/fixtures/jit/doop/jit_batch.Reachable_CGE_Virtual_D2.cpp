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
// JIT-Generated Kernel Functor: Reachable_CGE_Virtual_D2
// Handles: 11
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Reachable_CGE_Virtual_D2 {
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
        auto view_VirtualMethodInvocation_1_0_2_3_FULL_VER = views[1];
        auto view_HeapHelperNoThis_3_0_1_2_FULL_VER = views[3];
        auto view_ReachableInstruction_0_FULL_VER = views[5];

        // Root ColumnJoin (multi-source intersection): bind 'base' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var base :sources ((VarPointsTo :handle 0) (VirtualMethodInvocation :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_VarPointsTo_1_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_VarPointsTo_1_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_VarPointsTo_1_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_VarPointsTo_1_0_DELTA_VER.num_rows_;
          auto h_VarPointsTo_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_VarPointsTo_1_0_DELTA_VER);
          if (!h_VarPointsTo_0_root.valid()) continue;
          auto h_VirtualMethodInvocation_1_root = HandleType(0, view_VirtualMethodInvocation_1_0_2_3_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_VirtualMethodInvocation_1_0_2_3_FULL_VER);
          if (!h_VirtualMethodInvocation_1_root.valid()) continue;
          auto base = root_val_2;
        // Nested ColumnJoin (intersection): bind 'heap' from 2 sources
        // MIR: (column-join :var heap :sources ((VarPointsTo :handle 2 :prefix (base)) (HeapHelperNoThis :handle 3 :prefix ()) ))
        auto h_VarPointsTo_2_20 = h_VarPointsTo_0_root;
        auto h_HeapHelperNoThis_3_21 = HandleType(0, view_HeapHelperNoThis_3_0_1_2_FULL_VER.num_rows_, 0);
        auto intersect_22 = intersect_handles(tile, h_VarPointsTo_2_20.iterators(view_VarPointsTo_1_0_DELTA_VER), h_HeapHelperNoThis_3_21.iterators(view_HeapHelperNoThis_3_0_1_2_FULL_VER));
        for (auto it_23 = intersect_22.begin(); it_23.valid(); it_23.next()) {
          auto heap = it_23.value();
          auto positions = it_23.positions();
          auto ch_VarPointsTo_2_heap = h_VarPointsTo_2_20.child_range(positions[0], heap, tile, view_VarPointsTo_1_0_DELTA_VER);
          auto ch_HeapHelperNoThis_3_heap = h_HeapHelperNoThis_3_21.child_range(positions[1], heap, tile, view_HeapHelperNoThis_3_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'invocation' from 2 sources
        // MIR: (column-join :var invocation :sources ((VirtualMethodInvocation :handle 4 :prefix (base)) (ReachableInstruction :handle 5 :prefix ()) ))
        auto h_VirtualMethodInvocation_4_16 = h_VirtualMethodInvocation_1_root;
        auto h_ReachableInstruction_5_17 = HandleType(0, view_ReachableInstruction_0_FULL_VER.num_rows_, 0);
        auto intersect_18 = intersect_handles(tile, h_VirtualMethodInvocation_4_16.iterators(view_VirtualMethodInvocation_1_0_2_3_FULL_VER), h_ReachableInstruction_5_17.iterators(view_ReachableInstruction_0_FULL_VER));
        for (auto it_19 = intersect_18.begin(); it_19.valid(); it_19.next()) {
          auto invocation = it_19.value();
          auto positions = it_19.positions();
          auto ch_VirtualMethodInvocation_4_invocation = h_VirtualMethodInvocation_4_16.child_range(positions[0], invocation, tile, view_VirtualMethodInvocation_1_0_2_3_FULL_VER);
          auto ch_ReachableInstruction_5_invocation = h_ReachableInstruction_5_17.child_range(positions[1], invocation, tile, view_ReachableInstruction_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'simplename' from 2 sources
        // MIR: (column-join :var simplename :sources ((VirtualMethodInvocation :handle 6 :prefix (base invocation)) (HeapHelperNoThis :handle 7 :prefix (heap)) ))
        auto h_VirtualMethodInvocation_6_12 = ch_VirtualMethodInvocation_4_invocation;
        auto h_HeapHelperNoThis_7_13 = ch_HeapHelperNoThis_3_heap;
        auto intersect_14 = intersect_handles(tile, h_VirtualMethodInvocation_6_12.iterators(view_VirtualMethodInvocation_1_0_2_3_FULL_VER), h_HeapHelperNoThis_7_13.iterators(view_HeapHelperNoThis_3_0_1_2_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto simplename = it_15.value();
          auto positions = it_15.positions();
          auto ch_VirtualMethodInvocation_6_simplename = h_VirtualMethodInvocation_6_12.child_range(positions[0], simplename, tile, view_VirtualMethodInvocation_1_0_2_3_FULL_VER);
          auto ch_HeapHelperNoThis_7_simplename = h_HeapHelperNoThis_7_13.child_range(positions[1], simplename, tile, view_HeapHelperNoThis_3_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'descriptor' from 2 sources
        // MIR: (column-join :var descriptor :sources ((VirtualMethodInvocation :handle 8 :prefix (base invocation simplename)) (HeapHelperNoThis :handle 9 :prefix (heap simplename)) ))
        auto h_VirtualMethodInvocation_8_8 = ch_VirtualMethodInvocation_6_simplename;
        auto h_HeapHelperNoThis_9_9 = ch_HeapHelperNoThis_7_simplename;
        auto intersect_10 = intersect_handles(tile, h_VirtualMethodInvocation_8_8.iterators(view_VirtualMethodInvocation_1_0_2_3_FULL_VER), h_HeapHelperNoThis_9_9.iterators(view_HeapHelperNoThis_3_0_1_2_FULL_VER));
        for (auto it_11 = intersect_10.begin(); it_11.valid(); it_11.next()) {
          auto descriptor = it_11.value();
          auto positions = it_11.positions();
          auto ch_VirtualMethodInvocation_8_descriptor = h_VirtualMethodInvocation_8_8.child_range(positions[0], descriptor, tile, view_VirtualMethodInvocation_1_0_2_3_FULL_VER);
          auto ch_HeapHelperNoThis_9_descriptor = h_HeapHelperNoThis_9_9.child_range(positions[1], descriptor, tile, view_HeapHelperNoThis_3_0_1_2_FULL_VER);
        // Nested CartesianJoin: bind toMeth from 1 source(s)
        // MIR: (cartesian-join :vars (toMeth) :sources ((HeapHelperNoThis :handle 10 :prefix (heap simplename descriptor)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_HeapHelperNoThis_10_4 = ch_HeapHelperNoThis_9_descriptor;  // reusing narrowed handle

        if (!h_HeapHelperNoThis_10_4.valid()) continue;

        uint32_t degree_3 = h_HeapHelperNoThis_10_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto toMeth = view_HeapHelperNoThis_3_0_1_2_FULL_VER.get_value(3, h_HeapHelperNoThis_10_4.begin() + idx0_7);

        // Emit: Reachable(toMeth)
        output.emit_direct(toMeth);
        // Emit: CallGraphEdge(invocation, toMeth)
        output.emit_direct(invocation, toMeth);
        }
        }
        }
        }
        }
        }
  }
};

// End of JIT batch file
