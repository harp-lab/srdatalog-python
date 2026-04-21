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
// JIT-Generated Kernel Functor: SGRec_D1
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_SGRec_D1 {
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
    auto view_Sg_0_1_DELTA_VER = views[0];
    auto view_Arc_0_1_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'p' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var p :sources ((Sg :handle 0) (Arc :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Sg_0_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Sg_0_1_DELTA_VER.num_rows_) ? hint_hi_4
                                                                 : view_Sg_0_1_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Sg_0_1_DELTA_VER.num_rows_;
      auto h_Sg_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Sg_0_1_DELTA_VER);
      if (!h_Sg_0_root.valid())
        continue;
      auto h_Arc_1_root = HandleType(0, view_Arc_0_1_FULL_VER.num_rows_, 0)
                              .prefix(root_val_2, tile, view_Arc_0_1_FULL_VER);
      if (!h_Arc_1_root.valid())
        continue;
      auto p = root_val_2;
      // Nested ColumnJoin (intersection): bind 'q' from 2 sources
      // MIR: (column-join :var q :sources ((Sg :handle 2 :prefix (p)) (Arc :handle 3 :prefix ()) ))
      auto h_Sg_2_12 = h_Sg_0_root;
      auto h_Arc_3_13 = HandleType(0, view_Arc_0_1_FULL_VER.num_rows_, 0);
      auto intersect_14 = intersect_handles(tile, h_Sg_2_12.iterators(view_Sg_0_1_DELTA_VER),
                                            h_Arc_3_13.iterators(view_Arc_0_1_FULL_VER));
      for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
        auto q = it_15.value();
        auto positions = it_15.positions();
        auto ch_Sg_2_q = h_Sg_2_12.child_range(positions[0], q, tile, view_Sg_0_1_DELTA_VER);
        auto ch_Arc_3_q = h_Arc_3_13.child_range(positions[1], q, tile, view_Arc_0_1_FULL_VER);
        // Nested CartesianJoin: bind x, y from 2 source(s)
        // MIR: (cartesian-join :vars (x y) :sources ((Arc :handle 4 :prefix (p)) (Arc :handle 5
        // :prefix (q)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_Arc_4_4 = h_Arc_1_root;  // reusing narrowed handle
        auto h_Arc_5_6 = ch_Arc_3_q;    // reusing narrowed handle

        if (!h_Arc_4_4.valid() || !h_Arc_5_6.valid())
          continue;

        uint32_t degree_3 = h_Arc_4_4.degree();
        uint32_t degree_5 = h_Arc_5_6.degree();
        uint32_t total_7 = degree_3 * degree_5;
        if (total_7 == 0)
          continue;

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

          auto x = view_Arc_0_1_FULL_VER.get_value(1, h_Arc_4_4.begin() + idx0_9);
          auto y = view_Arc_0_1_FULL_VER.get_value(1, h_Arc_5_6.begin() + idx1_10);

          // Emit: Sg(x, y)
          output.emit_direct(x, y);
        }
      }
    }
  }
};

// End of JIT batch file
