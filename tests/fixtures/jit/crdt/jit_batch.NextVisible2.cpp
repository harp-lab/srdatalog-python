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
// JIT-Generated Kernel Functor: NextVisible2
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_NextVisible2 {
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
    auto view_HasValue_0_1_FULL_VER = views[0];
    auto view_SkipBlank_0_1_2_3_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'PrevCtr' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var PrevCtr :sources ((HasValue :handle 0) (SkipBlank :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_HasValue_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_HasValue_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_HasValue_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_HasValue_0_1_FULL_VER.num_rows_;
      auto h_HasValue_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_HasValue_0_1_FULL_VER);
      if (!h_HasValue_0_root.valid())
        continue;
      auto h_SkipBlank_1_root = HandleType(0, view_SkipBlank_0_1_2_3_FULL_VER.num_rows_, 0)
                                    .prefix(root_val_2, tile, view_SkipBlank_0_1_2_3_FULL_VER);
      if (!h_SkipBlank_1_root.valid())
        continue;
      auto PrevCtr = root_val_2;
      // Nested ColumnJoin (intersection): bind 'PrevN' from 2 sources
      // MIR: (column-join :var PrevN :sources ((HasValue :handle 2 :prefix (PrevCtr)) (SkipBlank
      // :handle 3 :prefix (PrevCtr)) ))
      auto h_HasValue_2_9 = h_HasValue_0_root;
      auto h_SkipBlank_3_10 = h_SkipBlank_1_root;
      auto intersect_11 =
          intersect_handles(tile, h_HasValue_2_9.iterators(view_HasValue_0_1_FULL_VER),
                            h_SkipBlank_3_10.iterators(view_SkipBlank_0_1_2_3_FULL_VER));
      for (auto it_12 = intersect_11.begin(); it_12.valid(); it_12.next()) {
        auto PrevN = it_12.value();
        auto positions = it_12.positions();
        auto ch_HasValue_2_PrevN =
            h_HasValue_2_9.child_range(positions[0], PrevN, tile, view_HasValue_0_1_FULL_VER);
        auto ch_SkipBlank_3_PrevN = h_SkipBlank_3_10.child_range(positions[1], PrevN, tile,
                                                                 view_SkipBlank_0_1_2_3_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'NextCtr' from 2 sources
        // MIR: (column-join :var NextCtr :sources ((SkipBlank :handle 4 :prefix (PrevCtr PrevN))
        // (HasValue :handle 5 :prefix ()) ))
        auto h_SkipBlank_4_5 = ch_SkipBlank_3_PrevN;
        auto h_HasValue_5_6 = HandleType(0, view_HasValue_0_1_FULL_VER.num_rows_, 0);
        auto intersect_7 =
            intersect_handles(tile, h_SkipBlank_4_5.iterators(view_SkipBlank_0_1_2_3_FULL_VER),
                              h_HasValue_5_6.iterators(view_HasValue_0_1_FULL_VER));
        for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
          auto NextCtr = it_8.value();
          auto positions = it_8.positions();
          auto ch_SkipBlank_4_NextCtr = h_SkipBlank_4_5.child_range(
              positions[0], NextCtr, tile, view_SkipBlank_0_1_2_3_FULL_VER);
          auto ch_HasValue_5_NextCtr =
              h_HasValue_5_6.child_range(positions[1], NextCtr, tile, view_HasValue_0_1_FULL_VER);
          // Nested ColumnJoin (intersection): bind 'NextN' from 2 sources
          // MIR: (column-join :var NextN :sources ((SkipBlank :handle 6 :prefix (PrevCtr PrevN
          // NextCtr)) (HasValue :handle 7 :prefix (NextCtr)) ))
          auto h_SkipBlank_6_1 = ch_SkipBlank_4_NextCtr;
          auto h_HasValue_7_2 = ch_HasValue_5_NextCtr;
          auto intersect_3 =
              intersect_handles(tile, h_SkipBlank_6_1.iterators(view_SkipBlank_0_1_2_3_FULL_VER),
                                h_HasValue_7_2.iterators(view_HasValue_0_1_FULL_VER));
          for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
            auto NextN = it_4.value();
            auto positions = it_4.positions();
            auto ch_SkipBlank_6_NextN = h_SkipBlank_6_1.child_range(
                positions[0], NextN, tile, view_SkipBlank_0_1_2_3_FULL_VER);
            auto ch_HasValue_7_NextN =
                h_HasValue_7_2.child_range(positions[1], NextN, tile, view_HasValue_0_1_FULL_VER);
            // Emit: NextVisible(PrevCtr, PrevN, NextCtr, NextN)
            if (tile.thread_rank() == 0)
              output.emit_direct(PrevCtr, PrevN, NextCtr, NextN);
          }
        }
      }
    }
  }
};

// End of JIT batch file
