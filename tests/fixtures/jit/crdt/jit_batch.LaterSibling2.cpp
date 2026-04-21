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
// JIT-Generated Kernel Functor: LaterSibling2
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_LaterSibling2 {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template <typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(Tile& tile, const Views* views,
                             const ValueType* __restrict__ root_unique_values,
                             uint32_t num_unique_root_keys, uint32_t num_root_keys,
                             uint32_t warp_id, uint32_t num_warps, Output& output) const {
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 1 unique views)
    auto view_Sibling_0_1_2_3_FULL_VER = views[0];

    // Root ColumnJoin (multi-source intersection): bind 'Ctr1' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var Ctr1 :sources ((Sibling :handle 0) (Sibling :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_Sibling_0_1_2_3_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Sibling_0_1_2_3_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Sibling_0_1_2_3_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Sibling_0_1_2_3_FULL_VER.num_rows_;
      auto h_Sibling_0_root = HandleType(hint_lo_3, hint_hi_4, 0)
                                  .prefix(root_val_2, tile, view_Sibling_0_1_2_3_FULL_VER);
      if (!h_Sibling_0_root.valid())
        continue;
      auto h_Sibling_1_root = HandleType(0, view_Sibling_0_1_2_3_FULL_VER.num_rows_, 0)
                                  .prefix(root_val_2, tile, view_Sibling_0_1_2_3_FULL_VER);
      if (!h_Sibling_1_root.valid())
        continue;
      auto Ctr1 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'N1' from 2 sources
      // MIR: (column-join :var N1 :sources ((Sibling :handle 2 :prefix (Ctr1)) (Sibling :handle 3
      // :prefix (Ctr1)) ))
      auto h_Sibling_2_12 = h_Sibling_1_root;
      auto h_Sibling_3_13 = h_Sibling_1_root;
      auto intersect_14 =
          intersect_handles(tile, h_Sibling_2_12.iterators(view_Sibling_0_1_2_3_FULL_VER),
                            h_Sibling_3_13.iterators(view_Sibling_0_1_2_3_FULL_VER));
      for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
        auto N1 = it_15.value();
        auto positions = it_15.positions();
        auto ch_Sibling_2_N1 =
            h_Sibling_2_12.child_range(positions[0], N1, tile, view_Sibling_0_1_2_3_FULL_VER);
        auto ch_Sibling_3_N1 =
            h_Sibling_3_13.child_range(positions[1], N1, tile, view_Sibling_0_1_2_3_FULL_VER);
        // Nested CartesianJoin: bind Ctr2, N2, Ctr3, N3 from 2 source(s)
        // MIR: (cartesian-join :vars (Ctr2 N2 Ctr3 N3) :sources ((Sibling :handle 4 :prefix (Ctr1
        // N1)) (Sibling :handle 5 :prefix (Ctr1 N1)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_Sibling_4_4 = ch_Sibling_3_N1;  // reusing narrowed handle
        auto h_Sibling_5_6 = ch_Sibling_3_N1;  // reusing narrowed handle

        if (!h_Sibling_4_4.valid() || !h_Sibling_5_6.valid())
          continue;

        uint32_t degree_3 = h_Sibling_4_4.degree();
        uint32_t degree_5 = h_Sibling_5_6.degree();
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

          auto Ctr2 = view_Sibling_0_1_2_3_FULL_VER.get_value(2, h_Sibling_4_4.begin() + idx0_9);
          auto N2 = view_Sibling_0_1_2_3_FULL_VER.get_value(3, h_Sibling_4_4.begin() + idx0_9);
          auto Ctr3 = view_Sibling_0_1_2_3_FULL_VER.get_value(2, h_Sibling_5_6.begin() + idx1_10);
          auto N3 = view_Sibling_0_1_2_3_FULL_VER.get_value(3, h_Sibling_5_6.begin() + idx1_10);

          if ((Ctr1 * 10 + N1) > (Ctr2 * 10 + N2)) {
            if ((Ctr2 * 10 + N2) > (Ctr3 * 10 + N3)) {
              // Emit: LaterSibling2(Ctr1, N1, Ctr3, N3)
              output.emit_direct(Ctr1, N1, Ctr3, N3);
            }
          }
        }
      }
    }
  }
};

// End of JIT batch file
