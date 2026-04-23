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
// JIT-Generated Kernel Functor: MainMethodDecl_Base
// Handles: 11
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_MainMethodDecl_Base {
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
        auto view_MainClass_0_FULL_VER = views[0];
        auto view_Method_DeclaringType_1_0_FULL_VER = views[1];
        auto view_Method_SimpleName_0_1_FULL_VER = views[3];
        auto view_Method_Descriptor_0_1_FULL_VER = views[4];
        auto view_Method_Modifier_1_0_FULL_VER = views[5];

        // Root ColumnJoin (multi-source intersection): bind 'mtype' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var mtype :sources ((MainClass :handle 0) (Method_DeclaringType :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_MainClass_0_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_MainClass_0_FULL_VER.num_rows_) ? hint_hi_4 : view_MainClass_0_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_MainClass_0_FULL_VER.num_rows_;
          auto h_MainClass_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_MainClass_0_FULL_VER);
          if (!h_MainClass_0_root.valid()) continue;
          auto h_Method_DeclaringType_1_root = HandleType(0, view_Method_DeclaringType_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Method_DeclaringType_1_0_FULL_VER);
          if (!h_Method_DeclaringType_1_root.valid()) continue;
          auto mtype = root_val_2;
        // Nested ColumnJoin (intersection): bind 'meth' from 5 sources
        // MIR: (column-join :var meth :sources ((Method_DeclaringType :handle 2 :prefix (mtype)) (Method_SimpleName :handle 3 :prefix ()) (Method_Descriptor :handle 4 :prefix ()) (Method_Modifier :handle 5 :prefix ()) (Method_Modifier :handle 6 :prefix ()) ))
        auto h_Method_DeclaringType_2_17 = h_Method_DeclaringType_1_root;
        auto h_Method_SimpleName_3_18 = HandleType(0, view_Method_SimpleName_0_1_FULL_VER.num_rows_, 0);
        auto h_Method_Descriptor_4_19 = HandleType(0, view_Method_Descriptor_0_1_FULL_VER.num_rows_, 0);
        auto h_Method_Modifier_5_20 = HandleType(0, view_Method_Modifier_1_0_FULL_VER.num_rows_, 0);
        auto h_Method_Modifier_6_21 = HandleType(0, view_Method_Modifier_1_0_FULL_VER.num_rows_, 0);
        auto intersect_22 = intersect_handles(tile, h_Method_DeclaringType_2_17.iterators(view_Method_DeclaringType_1_0_FULL_VER), h_Method_SimpleName_3_18.iterators(view_Method_SimpleName_0_1_FULL_VER), h_Method_Descriptor_4_19.iterators(view_Method_Descriptor_0_1_FULL_VER), h_Method_Modifier_5_20.iterators(view_Method_Modifier_1_0_FULL_VER), h_Method_Modifier_6_21.iterators(view_Method_Modifier_1_0_FULL_VER));
        for (auto it_23 = intersect_22.begin(); it_23.valid(); it_23.next()) {
          auto meth = it_23.value();
          auto positions = it_23.positions();
          auto ch_Method_DeclaringType_2_meth = h_Method_DeclaringType_2_17.child_range(positions[0], meth, tile, view_Method_DeclaringType_1_0_FULL_VER);
          auto ch_Method_SimpleName_3_meth = h_Method_SimpleName_3_18.child_range(positions[1], meth, tile, view_Method_SimpleName_0_1_FULL_VER);
          auto ch_Method_Descriptor_4_meth = h_Method_Descriptor_4_19.child_range(positions[2], meth, tile, view_Method_Descriptor_0_1_FULL_VER);
          auto ch_Method_Modifier_5_meth = h_Method_Modifier_5_20.child_range(positions[3], meth, tile, view_Method_Modifier_1_0_FULL_VER);
          auto ch_Method_Modifier_6_meth = h_Method_Modifier_6_21.child_range(positions[4], meth, tile, view_Method_Modifier_1_0_FULL_VER);
        // Nested CartesianJoin: bind _c0, _c1, _c2, _c3 from 4 source(s)
        // MIR: (cartesian-join :vars (_c0 _c1 _c2 _c3) :sources ((Method_SimpleName :handle 7 :prefix (meth)) (Method_Descriptor :handle 8 :prefix (meth)) (Method_Modifier :handle 9 :prefix (meth)) (Method_Modifier :handle 10 :prefix (meth)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_Method_SimpleName_7_4 = ch_Method_SimpleName_3_meth;  // reusing narrowed handle
        auto h_Method_Descriptor_8_6 = ch_Method_Descriptor_4_meth;  // reusing narrowed handle
        auto h_Method_Modifier_9_8 = ch_Method_Modifier_6_meth;  // reusing narrowed handle
        auto h_Method_Modifier_10_10 = ch_Method_Modifier_6_meth;  // reusing narrowed handle

        if (!h_Method_SimpleName_7_4.valid() || !h_Method_Descriptor_8_6.valid() || !h_Method_Modifier_9_8.valid() || !h_Method_Modifier_10_10.valid()) continue;

        uint32_t degree_3 = h_Method_SimpleName_7_4.degree();
        uint32_t degree_5 = h_Method_Descriptor_8_6.degree();
        uint32_t degree_7 = h_Method_Modifier_9_8.degree();
        uint32_t degree_9 = h_Method_Modifier_10_10.degree();
        uint32_t total_11 = degree_3 * degree_5 * degree_7 * degree_9;
        if (total_11 == 0) continue;

        for (uint32_t flat_idx_12 = lane_1; flat_idx_12 < total_11; flat_idx_12 += group_size_2) {
          uint32_t remaining = flat_idx_12;
          uint32_t idx3_16 = remaining % degree_9;
          remaining /= degree_9;
          uint32_t idx2_15 = remaining % degree_7;
          remaining /= degree_7;
          uint32_t idx1_14 = remaining % degree_5;
          remaining /= degree_5;
          uint32_t idx0_13 = remaining % degree_3;

          auto _c0 = view_Method_SimpleName_0_1_FULL_VER.get_value(1, h_Method_SimpleName_7_4.begin() + idx0_13);
          auto _c1 = view_Method_Descriptor_0_1_FULL_VER.get_value(1, h_Method_Descriptor_8_6.begin() + idx1_14);
          auto _c2 = view_Method_Modifier_1_0_FULL_VER.get_value(1, h_Method_Modifier_9_8.begin() + idx2_15);
          auto _c3 = view_Method_Modifier_1_0_FULL_VER.get_value(1, h_Method_Modifier_10_10.begin() + idx3_16);

        if (_c0 == 2979023) {
        if (_c1 == 3018506) {
        if (_c2 == 976234) {
        if (_c3 == 909718) {
        if (meth != 590319 && meth != 1138805 && meth != 926489) {
        // Emit: MainMethodDeclaration(meth)
        output.emit_direct(meth);
        }
        }
        }
        }
        }
        }
        }
        }
  }
};

// End of JIT batch file
