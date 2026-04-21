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
// JIT-Generated Kernel Functor: InitClass_StaticInvoke_D0
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_InitClass_StaticInvoke_D0 {
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

        // View declarations (deduplicated by spec, 5 unique views)
        auto view_Reachable_0_DELTA_VER = views[0];
        auto view_Instruction_Method_1_0_FULL_VER = views[1];
        auto view_MethodInvocation_Method_0_1_FULL_VER = views[3];
        auto view_isStaticMethodInvocation_Insn_0_FULL_VER = views[4];
        auto view_Method_DeclaringType_0_1_FULL_VER = views[6];

        // Root ColumnJoin (multi-source intersection): bind 'inmeth' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var inmeth :sources ((Reachable :handle 0) (Instruction_Method :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Reachable_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Reachable_0_DELTA_VER.num_rows_) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Reachable_0_DELTA_VER.num_rows_;
          auto h_Reachable_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Reachable_0_DELTA_VER);
          if (!h_Reachable_0_root.valid()) continue;
          auto h_Instruction_Method_1_root = HandleType(0, view_Instruction_Method_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Instruction_Method_1_0_FULL_VER);
          if (!h_Instruction_Method_1_root.valid()) continue;
          auto inmeth = root_val_2;
        // Nested ColumnJoin (intersection): bind 'invocation' from 3 sources
        // MIR: (column-join :var invocation :sources ((Instruction_Method :handle 2 :prefix (inmeth)) (MethodInvocation_Method :handle 3 :prefix ()) (isStaticMethodInvocation_Insn :handle 4 :prefix ()) ))
        auto h_Instruction_Method_2_12 = h_Instruction_Method_1_root;
        auto h_MethodInvocation_Method_3_13 = HandleType(0, view_MethodInvocation_Method_0_1_FULL_VER.num_rows_, 0);
        auto h_isStaticMethodInvocation_Insn_4_14 = HandleType(0, view_isStaticMethodInvocation_Insn_0_FULL_VER.num_rows_, 0);
        auto intersect_15 = intersect_handles(tile, h_Instruction_Method_2_12.iterators(view_Instruction_Method_1_0_FULL_VER), h_MethodInvocation_Method_3_13.iterators(view_MethodInvocation_Method_0_1_FULL_VER), h_isStaticMethodInvocation_Insn_4_14.iterators(view_isStaticMethodInvocation_Insn_0_FULL_VER));
        for (auto it_16 = intersect_15.begin(); it_16.valid(); it_16.next()) {
          auto invocation = it_16.value();
          auto positions = it_16.positions();
          auto ch_Instruction_Method_2_invocation = h_Instruction_Method_2_12.child_range(positions[0], invocation, tile, view_Instruction_Method_1_0_FULL_VER);
          auto ch_MethodInvocation_Method_3_invocation = h_MethodInvocation_Method_3_13.child_range(positions[1], invocation, tile, view_MethodInvocation_Method_0_1_FULL_VER);
          auto ch_isStaticMethodInvocation_Insn_4_invocation = h_isStaticMethodInvocation_Insn_4_14.child_range(positions[2], invocation, tile, view_isStaticMethodInvocation_Insn_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'sig' from 2 sources
        // MIR: (column-join :var sig :sources ((MethodInvocation_Method :handle 5 :prefix (invocation)) (Method_DeclaringType :handle 6 :prefix ()) ))
        auto h_MethodInvocation_Method_5_8 = ch_MethodInvocation_Method_3_invocation;
        auto h_Method_DeclaringType_6_9 = HandleType(0, view_Method_DeclaringType_0_1_FULL_VER.num_rows_, 0);
        auto intersect_10 = intersect_handles(tile, h_MethodInvocation_Method_5_8.iterators(view_MethodInvocation_Method_0_1_FULL_VER), h_Method_DeclaringType_6_9.iterators(view_Method_DeclaringType_0_1_FULL_VER));
        for (auto it_11 = intersect_10.begin(); it_11.valid(); it_11.next()) {
          auto sig = it_11.value();
          auto positions = it_11.positions();
          auto ch_MethodInvocation_Method_5_sig = h_MethodInvocation_Method_5_8.child_range(positions[0], sig, tile, view_MethodInvocation_Method_0_1_FULL_VER);
          auto ch_Method_DeclaringType_6_sig = h_Method_DeclaringType_6_9.child_range(positions[1], sig, tile, view_Method_DeclaringType_0_1_FULL_VER);
        // Nested CartesianJoin: bind classD from 1 source(s)
        // MIR: (cartesian-join :vars (classD) :sources ((Method_DeclaringType :handle 7 :prefix (sig)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_Method_DeclaringType_7_4 = ch_Method_DeclaringType_6_sig;  // reusing narrowed handle

        if (!h_Method_DeclaringType_7_4.valid()) continue;

        uint32_t degree_3 = h_Method_DeclaringType_7_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto classD = view_Method_DeclaringType_0_1_FULL_VER.get_value(1, h_Method_DeclaringType_7_4.begin() + idx0_7);

        // Emit: InitializedClass(classD)
        output.emit_direct(classD);
        }
        }
        }
        }
  }
};

// End of JIT batch file
