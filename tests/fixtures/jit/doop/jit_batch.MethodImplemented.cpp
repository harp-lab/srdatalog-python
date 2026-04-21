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
// JIT-Generated Kernel Functor: MethodImplemented
// Handles: 7
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_MethodImplemented {
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
        auto view_Method_SimpleName_0_1_FULL_VER = views[0];
        auto view_Method_Descriptor_0_1_FULL_VER = views[1];
        auto view_Method_DeclaringType_0_1_FULL_VER = views[2];
        auto view_Method_Modifier_0_1_FULL_VER = views[6];

        // Root ColumnJoin (multi-source intersection): bind 'meth' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var meth :sources ((Method_SimpleName :handle 0) (Method_Descriptor :handle 1) (Method_DeclaringType :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Method_SimpleName_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Method_SimpleName_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_Method_SimpleName_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Method_SimpleName_0_1_FULL_VER.num_rows_;
          auto h_Method_SimpleName_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Method_SimpleName_0_1_FULL_VER);
          if (!h_Method_SimpleName_0_root.valid()) continue;
          auto h_Method_Descriptor_1_root = HandleType(0, view_Method_Descriptor_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Method_Descriptor_0_1_FULL_VER);
          if (!h_Method_Descriptor_1_root.valid()) continue;
          auto h_Method_DeclaringType_2_root = HandleType(0, view_Method_DeclaringType_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Method_DeclaringType_0_1_FULL_VER);
          if (!h_Method_DeclaringType_2_root.valid()) continue;
          auto meth = root_val_2;
        // Nested CartesianJoin: bind simplename, descriptor, mtype from 3 source(s)
        // MIR: (cartesian-join :vars (simplename descriptor mtype) :sources ((Method_SimpleName :handle 3 :prefix (meth)) (Method_Descriptor :handle 4 :prefix (meth)) (Method_DeclaringType :handle 5 :prefix (meth)) ))
        uint32_t lane_2 = tile.thread_rank();
        uint32_t group_size_3 = tile.size();

        auto h_Method_SimpleName_3_5 = h_Method_SimpleName_0_root;  // reusing narrowed handle
        auto h_Method_Descriptor_4_7 = h_Method_Descriptor_1_root;  // reusing narrowed handle
        auto h_Method_DeclaringType_5_9 = h_Method_DeclaringType_2_root;  // reusing narrowed handle

        if (!h_Method_SimpleName_3_5.valid() || !h_Method_Descriptor_4_7.valid() || !h_Method_DeclaringType_5_9.valid()) continue;

        uint32_t degree_4 = h_Method_SimpleName_3_5.degree();
        uint32_t degree_6 = h_Method_Descriptor_4_7.degree();
        uint32_t degree_8 = h_Method_DeclaringType_5_9.degree();
        uint32_t total_10 = degree_4 * degree_6 * degree_8;
        if (total_10 == 0) continue;

        // Pre-narrow negation handle for Method_Modifier (pre-Cartesian vars: meth)
        auto h_Method_Modifier_neg_pre_const_11 = HandleType(0, view_Method_Modifier_0_1_FULL_VER.num_rows_, 0).prefix(1001, tile, view_Method_Modifier_0_1_FULL_VER);
        auto h_Method_Modifier_neg_pre_1 = h_Method_Modifier_neg_pre_const_11.prefix(meth, tile, view_Method_Modifier_0_1_FULL_VER);

        for (uint32_t flat_idx_12 = lane_2; flat_idx_12 < total_10; flat_idx_12 += group_size_3) {
          uint32_t remaining = flat_idx_12;
          uint32_t idx2_15 = remaining % degree_8;
          remaining /= degree_8;
          uint32_t idx1_14 = remaining % degree_6;
          remaining /= degree_6;
          uint32_t idx0_13 = remaining % degree_4;

          auto simplename = view_Method_SimpleName_0_1_FULL_VER.get_value(1, h_Method_SimpleName_3_5.begin() + idx0_13);
          auto descriptor = view_Method_Descriptor_0_1_FULL_VER.get_value(1, h_Method_Descriptor_4_7.begin() + idx1_14);
          auto mtype = view_Method_DeclaringType_0_1_FULL_VER.get_value(1, h_Method_DeclaringType_5_9.begin() + idx2_15);

        // Negation: NOT EXISTS in Method_Modifier
        // MIR: (negation :rel Method_Modifier :prefix (meth) :handle 6)
        // Using pre-narrowed handle (pre-Cartesian vars: meth)
        if (!h_Method_Modifier_neg_pre_1.valid()) {
        // Emit: MethodImplemented(simplename, descriptor, mtype, meth)
        output.emit_direct(simplename, descriptor, mtype, meth);
        }
        }
        }
  }
};

// End of JIT batch file
