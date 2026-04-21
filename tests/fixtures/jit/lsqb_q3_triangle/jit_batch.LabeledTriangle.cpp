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
// JIT-Generated Kernel Functor: LabeledTriangle
// Handles: 18
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_LabeledTriangle {
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
        auto view_Knows_0_1_FULL_VER = views[0];
        auto view_IsLocatedIn_0_1_FULL_VER = views[1];
        auto view_Knows_1_0_FULL_VER = views[2];
        auto view_IsPartOf_0_1_FULL_VER = views[10];

        // Root ColumnJoin (multi-source intersection): bind 'p1' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var p1 :sources ((Knows :handle 0) (IsLocatedIn :handle 1) (Knows :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Knows_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Knows_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_Knows_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Knows_0_1_FULL_VER.num_rows_;
          auto h_Knows_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Knows_0_1_FULL_VER);
          if (!h_Knows_0_root.valid()) continue;
          auto h_IsLocatedIn_1_root = HandleType(0, view_IsLocatedIn_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_IsLocatedIn_0_1_FULL_VER);
          if (!h_IsLocatedIn_1_root.valid()) continue;
          auto h_Knows_2_root = HandleType(0, view_Knows_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Knows_1_0_FULL_VER);
          if (!h_Knows_2_root.valid()) continue;
          auto p1 = root_val_2;
        // Nested ColumnJoin (intersection): bind 'p2' from 3 sources
        // MIR: (column-join :var p2 :sources ((Knows :handle 3 :prefix (p1)) (Knows :handle 4 :prefix ()) (IsLocatedIn :handle 5 :prefix ()) ))
        auto h_Knows_3_23 = h_Knows_0_root;
        auto h_Knows_4_24 = HandleType(0, view_Knows_0_1_FULL_VER.num_rows_, 0);
        auto h_IsLocatedIn_5_25 = HandleType(0, view_IsLocatedIn_0_1_FULL_VER.num_rows_, 0);
        auto intersect_26 = intersect_handles(tile, h_Knows_3_23.iterators(view_Knows_0_1_FULL_VER), h_Knows_4_24.iterators(view_Knows_0_1_FULL_VER), h_IsLocatedIn_5_25.iterators(view_IsLocatedIn_0_1_FULL_VER));
        for (auto it_27 = intersect_26.begin(); it_27.valid(); it_27.next()) {
          auto p2 = it_27.value();
          auto positions = it_27.positions();
          auto ch_Knows_3_p2 = h_Knows_3_23.child_range(positions[0], p2, tile, view_Knows_0_1_FULL_VER);
          auto ch_Knows_4_p2 = h_Knows_4_24.child_range(positions[1], p2, tile, view_Knows_0_1_FULL_VER);
          auto ch_IsLocatedIn_5_p2 = h_IsLocatedIn_5_25.child_range(positions[2], p2, tile, view_IsLocatedIn_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'p3' from 3 sources
        // MIR: (column-join :var p3 :sources ((Knows :handle 6 :prefix (p2)) (IsLocatedIn :handle 7 :prefix ()) (Knows :handle 8 :prefix (p1)) ))
        auto h_Knows_6_18 = ch_Knows_4_p2;
        auto h_IsLocatedIn_7_19 = HandleType(0, view_IsLocatedIn_0_1_FULL_VER.num_rows_, 0);
        auto h_Knows_8_20 = h_Knows_2_root;
        auto intersect_21 = intersect_handles(tile, h_Knows_6_18.iterators(view_Knows_0_1_FULL_VER), h_IsLocatedIn_7_19.iterators(view_IsLocatedIn_0_1_FULL_VER), h_Knows_8_20.iterators(view_Knows_1_0_FULL_VER));
        for (auto it_22 = intersect_21.begin(); it_22.valid(); it_22.next()) {
          auto p3 = it_22.value();
          auto positions = it_22.positions();
          auto ch_Knows_6_p3 = h_Knows_6_18.child_range(positions[0], p3, tile, view_Knows_0_1_FULL_VER);
          auto ch_IsLocatedIn_7_p3 = h_IsLocatedIn_7_19.child_range(positions[1], p3, tile, view_IsLocatedIn_0_1_FULL_VER);
          auto ch_Knows_8_p3 = h_Knows_8_20.child_range(positions[2], p3, tile, view_Knows_1_0_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'c1' from 2 sources
        // MIR: (column-join :var c1 :sources ((IsLocatedIn :handle 9 :prefix (p1)) (IsPartOf :handle 10 :prefix ()) ))
        auto h_IsLocatedIn_9_14 = h_IsLocatedIn_1_root;
        auto h_IsPartOf_10_15 = HandleType(0, view_IsPartOf_0_1_FULL_VER.num_rows_, 0);
        auto intersect_16 = intersect_handles(tile, h_IsLocatedIn_9_14.iterators(view_IsLocatedIn_0_1_FULL_VER), h_IsPartOf_10_15.iterators(view_IsPartOf_0_1_FULL_VER));
        for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
          auto c1 = it_17.value();
          auto positions = it_17.positions();
          auto ch_IsLocatedIn_9_c1 = h_IsLocatedIn_9_14.child_range(positions[0], c1, tile, view_IsLocatedIn_0_1_FULL_VER);
          auto ch_IsPartOf_10_c1 = h_IsPartOf_10_15.child_range(positions[1], c1, tile, view_IsPartOf_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'c2' from 2 sources
        // MIR: (column-join :var c2 :sources ((IsLocatedIn :handle 11 :prefix (p2)) (IsPartOf :handle 12 :prefix ()) ))
        auto h_IsLocatedIn_11_10 = ch_IsLocatedIn_5_p2;
        auto h_IsPartOf_12_11 = HandleType(0, view_IsPartOf_0_1_FULL_VER.num_rows_, 0);
        auto intersect_12 = intersect_handles(tile, h_IsLocatedIn_11_10.iterators(view_IsLocatedIn_0_1_FULL_VER), h_IsPartOf_12_11.iterators(view_IsPartOf_0_1_FULL_VER));
        for (auto it_13 = intersect_12.begin(); it_13.valid(); it_13.next()) {
          auto c2 = it_13.value();
          auto positions = it_13.positions();
          auto ch_IsLocatedIn_11_c2 = h_IsLocatedIn_11_10.child_range(positions[0], c2, tile, view_IsLocatedIn_0_1_FULL_VER);
          auto ch_IsPartOf_12_c2 = h_IsPartOf_12_11.child_range(positions[1], c2, tile, view_IsPartOf_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'c3' from 2 sources
        // MIR: (column-join :var c3 :sources ((IsLocatedIn :handle 13 :prefix (p3)) (IsPartOf :handle 14 :prefix ()) ))
        auto h_IsLocatedIn_13_6 = ch_IsLocatedIn_7_p3;
        auto h_IsPartOf_14_7 = HandleType(0, view_IsPartOf_0_1_FULL_VER.num_rows_, 0);
        auto intersect_8 = intersect_handles(tile, h_IsLocatedIn_13_6.iterators(view_IsLocatedIn_0_1_FULL_VER), h_IsPartOf_14_7.iterators(view_IsPartOf_0_1_FULL_VER));
        for (auto it_9 = intersect_8.begin(); it_9.valid(); it_9.next()) {
          auto c3 = it_9.value();
          auto positions = it_9.positions();
          auto ch_IsLocatedIn_13_c3 = h_IsLocatedIn_13_6.child_range(positions[0], c3, tile, view_IsLocatedIn_0_1_FULL_VER);
          auto ch_IsPartOf_14_c3 = h_IsPartOf_14_7.child_range(positions[1], c3, tile, view_IsPartOf_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'co' from 3 sources
        // MIR: (column-join :var co :sources ((IsPartOf :handle 15 :prefix (c1)) (IsPartOf :handle 16 :prefix (c2)) (IsPartOf :handle 17 :prefix (c3)) ))
        auto h_IsPartOf_15_1 = ch_IsPartOf_10_c1;
        auto h_IsPartOf_16_2 = ch_IsPartOf_12_c2;
        auto h_IsPartOf_17_3 = ch_IsPartOf_14_c3;
        auto intersect_4 = intersect_handles(tile, h_IsPartOf_15_1.iterators(view_IsPartOf_0_1_FULL_VER), h_IsPartOf_16_2.iterators(view_IsPartOf_0_1_FULL_VER), h_IsPartOf_17_3.iterators(view_IsPartOf_0_1_FULL_VER));
        for (auto it_5 = intersect_4.begin(); it_5.valid(); it_5.next()) {
          auto co = it_5.value();
          auto positions = it_5.positions();
          auto ch_IsPartOf_15_co = h_IsPartOf_15_1.child_range(positions[0], co, tile, view_IsPartOf_0_1_FULL_VER);
          auto ch_IsPartOf_16_co = h_IsPartOf_16_2.child_range(positions[1], co, tile, view_IsPartOf_0_1_FULL_VER);
          auto ch_IsPartOf_17_co = h_IsPartOf_17_3.child_range(positions[2], co, tile, view_IsPartOf_0_1_FULL_VER);
        // Emit: Triangle(p1, p2, p3)
        if (tile.thread_rank() == 0) output.emit_direct(p1, p2, p3);
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
