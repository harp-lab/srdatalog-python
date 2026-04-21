// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cooperative_groups.h>
#include <cstdint>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

// =============================================================
// JIT-Generated Kernel Functor: Triangle
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Triangle {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template <typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(Tile& tile, const Views* views,
                             const ValueType* __restrict__ root_unique_values,
                             uint32_t num_unique_root_keys, uint32_t num_root_keys,
                             uint32_t warp_id, uint32_t num_warps, Output& output) const {
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_Edge_0_1_FULL_VER = views[0];
    auto view_Edge_1_0_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'x' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var x :sources ((Edge :handle 0) (Edge :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Edge_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Edge_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Edge_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Edge_0_1_FULL_VER.num_rows_;
      auto h_Edge_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Edge_0_1_FULL_VER);
      if (!h_Edge_0_root.valid())
        continue;
      auto h_Edge_1_root = HandleType(0, view_Edge_1_0_FULL_VER.num_rows_, 0)
                               .prefix(root_val_2, tile, view_Edge_1_0_FULL_VER);
      if (!h_Edge_1_root.valid())
        continue;
      auto x = root_val_2;
      // Nested ColumnJoin (intersection): bind 'y' from 2 sources
      // MIR: (column-join :var y :sources ((Edge :handle 2 :prefix (x)) (Edge :handle 3 :prefix ())
      // ))
      auto h_Edge_2_5 = h_Edge_0_root;
      auto h_Edge_3_6 = HandleType(0, view_Edge_0_1_FULL_VER.num_rows_, 0);
      auto intersect_7 = intersect_handles(tile, h_Edge_2_5.iterators(view_Edge_0_1_FULL_VER),
                                           h_Edge_3_6.iterators(view_Edge_0_1_FULL_VER));
      for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
        auto y = it_8.value();
        auto positions = it_8.positions();
        auto ch_Edge_2_y = h_Edge_2_5.child_range(positions[0], y, tile, view_Edge_0_1_FULL_VER);
        auto ch_Edge_3_y = h_Edge_3_6.child_range(positions[1], y, tile, view_Edge_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'z' from 2 sources
        // MIR: (column-join :var z :sources ((Edge :handle 4 :prefix (y)) (Edge :handle 5 :prefix
        // (x)) ))
        auto h_Edge_4_1 = ch_Edge_3_y;
        auto h_Edge_5_2 = h_Edge_1_root;
        auto intersect_3 = intersect_handles(tile, h_Edge_4_1.iterators(view_Edge_0_1_FULL_VER),
                                             h_Edge_5_2.iterators(view_Edge_1_0_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto z = it_4.value();
          auto positions = it_4.positions();
          auto ch_Edge_4_z = h_Edge_4_1.child_range(positions[0], z, tile, view_Edge_0_1_FULL_VER);
          auto ch_Edge_5_z = h_Edge_5_2.child_range(positions[1], z, tile, view_Edge_1_0_FULL_VER);
          // Emit: Z(x, y, z)
          if (tile.thread_rank() == 0)
            output.emit_direct(x, y, z);
        }
      }
    }
  }
};

// End of JIT batch file
