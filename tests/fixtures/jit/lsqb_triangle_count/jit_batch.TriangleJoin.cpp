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
// JIT-Generated Kernel Functor: TriangleJoin
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_TriangleJoin {
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

        // View declarations (deduplicated by spec, 1 unique views)
        auto view_Knows_0_1_FULL_VER = views[0];

        // Root ColumnJoin (multi-source intersection): bind 'a' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var a :sources ((Knows :handle 0) (Knows :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_Knows_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_Knows_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_Knows_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Knows_0_1_FULL_VER.num_rows_;
          auto h_Knows_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Knows_0_1_FULL_VER);
          if (!h_Knows_0_root.valid()) continue;
          auto h_Knows_1_root = HandleType(0, view_Knows_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Knows_0_1_FULL_VER);
          if (!h_Knows_1_root.valid()) continue;
          auto a = root_val_2;
        // Nested ColumnJoin (intersection): bind 'b' from 2 sources
        // MIR: (column-join :var b :sources ((Knows :handle 2 :prefix (a)) (Knows :handle 3 :prefix ()) ))
        auto h_Knows_2_5 = h_Knows_1_root;
        auto h_Knows_3_6 = HandleType(0, view_Knows_0_1_FULL_VER.num_rows_, 0);
        auto intersect_7 = intersect_handles(tile, h_Knows_2_5.iterators(view_Knows_0_1_FULL_VER), h_Knows_3_6.iterators(view_Knows_0_1_FULL_VER));
        for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
          auto b = it_8.value();
          auto positions = it_8.positions();
          auto ch_Knows_2_b = h_Knows_2_5.child_range(positions[0], b, tile, view_Knows_0_1_FULL_VER);
          auto ch_Knows_3_b = h_Knows_3_6.child_range(positions[1], b, tile, view_Knows_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'c' from 2 sources
        // MIR: (column-join :var c :sources ((Knows :handle 4 :prefix (b)) (Knows :handle 5 :prefix (a)) ))
        auto h_Knows_4_1 = ch_Knows_3_b;
        auto h_Knows_5_2 = h_Knows_1_root;
        auto intersect_3 = intersect_handles(tile, h_Knows_4_1.iterators(view_Knows_0_1_FULL_VER), h_Knows_5_2.iterators(view_Knows_0_1_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto c = it_4.value();
          auto positions = it_4.positions();
          auto ch_Knows_4_c = h_Knows_4_1.child_range(positions[0], c, tile, view_Knows_0_1_FULL_VER);
          auto ch_Knows_5_c = h_Knows_5_2.child_range(positions[1], c, tile, view_Knows_0_1_FULL_VER);
        // Emit: Triangle(a, b, c)
        if (tile.thread_rank() == 0) output.emit_direct(a, b, c);
        }
        }
        }
  }
};

// End of JIT batch file
