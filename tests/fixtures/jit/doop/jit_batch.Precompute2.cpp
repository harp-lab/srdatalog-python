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
// JIT-Generated Kernel Functor: Precompute2
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Precompute2 {
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
        auto view_VirtualMethodInvocation_Base_0_1_FULL_VER = views[0];
        auto view_VirtualMethodInvocation_SimpleName_0_1_FULL_VER = views[1];
        auto view_VirtualMethodInvocation_Descriptor_0_1_FULL_VER = views[2];

        // Root ColumnJoin (multi-source intersection): bind 'invocation' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var invocation :sources ((VirtualMethodInvocation_Base :handle 0) (VirtualMethodInvocation_SimpleName :handle 1) (VirtualMethodInvocation_Descriptor :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_VirtualMethodInvocation_Base_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_VirtualMethodInvocation_Base_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_VirtualMethodInvocation_Base_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_VirtualMethodInvocation_Base_0_1_FULL_VER.num_rows_;
          auto h_VirtualMethodInvocation_Base_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_VirtualMethodInvocation_Base_0_1_FULL_VER);
          if (!h_VirtualMethodInvocation_Base_0_root.valid()) continue;
          auto h_VirtualMethodInvocation_SimpleName_1_root = HandleType(0, view_VirtualMethodInvocation_SimpleName_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_VirtualMethodInvocation_SimpleName_0_1_FULL_VER);
          if (!h_VirtualMethodInvocation_SimpleName_1_root.valid()) continue;
          auto h_VirtualMethodInvocation_Descriptor_2_root = HandleType(0, view_VirtualMethodInvocation_Descriptor_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_VirtualMethodInvocation_Descriptor_0_1_FULL_VER);
          if (!h_VirtualMethodInvocation_Descriptor_2_root.valid()) continue;
          auto invocation = root_val_2;
        // Nested CartesianJoin: bind base, simplename, descriptor from 3 source(s)
        // MIR: (cartesian-join :vars (base simplename descriptor) :sources ((VirtualMethodInvocation_Base :handle 3 :prefix (invocation)) (VirtualMethodInvocation_SimpleName :handle 4 :prefix (invocation)) (VirtualMethodInvocation_Descriptor :handle 5 :prefix (invocation)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_VirtualMethodInvocation_Base_3_4 = h_VirtualMethodInvocation_Base_0_root;  // reusing narrowed handle
        auto h_VirtualMethodInvocation_SimpleName_4_6 = h_VirtualMethodInvocation_SimpleName_1_root;  // reusing narrowed handle
        auto h_VirtualMethodInvocation_Descriptor_5_8 = h_VirtualMethodInvocation_Descriptor_2_root;  // reusing narrowed handle

        if (!h_VirtualMethodInvocation_Base_3_4.valid() || !h_VirtualMethodInvocation_SimpleName_4_6.valid() || !h_VirtualMethodInvocation_Descriptor_5_8.valid()) continue;

        uint32_t degree_3 = h_VirtualMethodInvocation_Base_3_4.degree();
        uint32_t degree_5 = h_VirtualMethodInvocation_SimpleName_4_6.degree();
        uint32_t degree_7 = h_VirtualMethodInvocation_Descriptor_5_8.degree();
        uint32_t total_9 = degree_3 * degree_5 * degree_7;
        if (total_9 == 0) continue;

        for (uint32_t flat_idx_10 = lane_1; flat_idx_10 < total_9; flat_idx_10 += group_size_2) {
          uint32_t remaining = flat_idx_10;
          uint32_t idx2_13 = remaining % degree_7;
          remaining /= degree_7;
          uint32_t idx1_12 = remaining % degree_5;
          remaining /= degree_5;
          uint32_t idx0_11 = remaining % degree_3;

          auto base = view_VirtualMethodInvocation_Base_0_1_FULL_VER.get_value(1, h_VirtualMethodInvocation_Base_3_4.begin() + idx0_11);
          auto simplename = view_VirtualMethodInvocation_SimpleName_0_1_FULL_VER.get_value(1, h_VirtualMethodInvocation_SimpleName_4_6.begin() + idx1_12);
          auto descriptor = view_VirtualMethodInvocation_Descriptor_0_1_FULL_VER.get_value(1, h_VirtualMethodInvocation_Descriptor_5_8.begin() + idx2_13);

        // Emit: VirtualMethodInvocation(invocation, base, simplename, descriptor)
        output.emit_direct(invocation, base, simplename, descriptor);
        }
        }
  }
};

// End of JIT batch file
