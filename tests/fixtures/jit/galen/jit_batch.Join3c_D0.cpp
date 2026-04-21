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
// JIT-Generated Kernel Functor: Join3c_D0
// Handles: 9
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Join3c_D0 {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template <typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(Tile& tile, const Views* views,
                             const ValueType* __restrict__ root_unique_values,
                             uint32_t num_unique_root_keys, uint32_t num_root_keys,
                             uint32_t warp_id, uint32_t num_warps, Output& output) const {
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 3 unique views)
    auto view_OutQ_2_1_0_DELTA_VER = views[0];
    auto view_OutQ_0_1_2_FULL_VER = views[1];
    auto view_RInput_0_1_2_FULL_VER = views[3];

    // Root ColumnJoin (multi-source intersection): bind 'z' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var z :sources ((OutQ :handle 0) (OutQ :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_OutQ_2_1_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_OutQ_2_1_0_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_OutQ_2_1_0_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_OutQ_2_1_0_DELTA_VER.num_rows_;
      auto h_OutQ_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_OutQ_2_1_0_DELTA_VER);
      if (!h_OutQ_0_root.valid())
        continue;
      auto h_OutQ_1_root = HandleType(0, view_OutQ_0_1_2_FULL_VER.num_rows_, 0)
                               .prefix(root_val_2, tile, view_OutQ_0_1_2_FULL_VER);
      if (!h_OutQ_1_root.valid())
        continue;
      auto z = root_val_2;
      // Nested ColumnJoin (intersection): bind 'y' from 2 sources
      // MIR: (column-join :var y :sources ((OutQ :handle 2 :prefix (z)) (RInput :handle 3 :prefix
      // ()) ))
      auto h_OutQ_2_18 = h_OutQ_0_root;
      auto h_RInput_3_19 = HandleType(0, view_RInput_0_1_2_FULL_VER.num_rows_, 0);
      auto intersect_20 = intersect_handles(tile, h_OutQ_2_18.iterators(view_OutQ_2_1_0_DELTA_VER),
                                            h_RInput_3_19.iterators(view_RInput_0_1_2_FULL_VER));
      for (auto it_21 = intersect_20.begin(); it_21.valid(); it_21.next()) {
        auto y = it_21.value();
        auto positions = it_21.positions();
        auto ch_OutQ_2_y =
            h_OutQ_2_18.child_range(positions[0], y, tile, view_OutQ_2_1_0_DELTA_VER);
        auto ch_RInput_3_y =
            h_RInput_3_19.child_range(positions[1], y, tile, view_RInput_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'u' from 2 sources
        // MIR: (column-join :var u :sources ((RInput :handle 4 :prefix (y)) (OutQ :handle 5 :prefix
        // (z)) ))
        auto h_RInput_4_14 = ch_RInput_3_y;
        auto h_OutQ_5_15 = h_OutQ_1_root;
        auto intersect_16 =
            intersect_handles(tile, h_RInput_4_14.iterators(view_RInput_0_1_2_FULL_VER),
                              h_OutQ_5_15.iterators(view_OutQ_0_1_2_FULL_VER));
        for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
          auto u = it_17.value();
          auto positions = it_17.positions();
          auto ch_RInput_4_u =
              h_RInput_4_14.child_range(positions[0], u, tile, view_RInput_0_1_2_FULL_VER);
          auto ch_OutQ_5_u =
              h_OutQ_5_15.child_range(positions[1], u, tile, view_OutQ_0_1_2_FULL_VER);
          // Nested CartesianJoin: bind x, e, o from 3 source(s)
          // MIR: (cartesian-join :vars (x e o) :sources ((OutQ :handle 6 :prefix (z y)) (RInput
          // :handle 7 :prefix (y u)) (OutQ :handle 8 :prefix (z u)) ))
          uint32_t lane_1 = tile.thread_rank();
          uint32_t group_size_2 = tile.size();

          auto h_OutQ_6_4 = ch_OutQ_2_y;      // reusing narrowed handle
          auto h_RInput_7_6 = ch_RInput_4_u;  // reusing narrowed handle
          auto h_OutQ_8_8 = ch_OutQ_5_u;      // reusing narrowed handle

          if (!h_OutQ_6_4.valid() || !h_RInput_7_6.valid() || !h_OutQ_8_8.valid())
            continue;

          uint32_t degree_3 = h_OutQ_6_4.degree();
          uint32_t degree_5 = h_RInput_7_6.degree();
          uint32_t degree_7 = h_OutQ_8_8.degree();
          uint32_t total_9 = degree_3 * degree_5 * degree_7;
          if (total_9 == 0)
            continue;

          for (uint32_t flat_idx_10 = lane_1; flat_idx_10 < total_9; flat_idx_10 += group_size_2) {
            uint32_t remaining = flat_idx_10;
            uint32_t idx2_13 = remaining % degree_7;
            remaining /= degree_7;
            uint32_t idx1_12 = remaining % degree_5;
            remaining /= degree_5;
            uint32_t idx0_11 = remaining % degree_3;

            auto x = view_OutQ_2_1_0_DELTA_VER.get_value(2, h_OutQ_6_4.begin() + idx0_11);
            auto e = view_RInput_0_1_2_FULL_VER.get_value(2, h_RInput_7_6.begin() + idx1_12);
            auto o = view_OutQ_0_1_2_FULL_VER.get_value(2, h_OutQ_8_8.begin() + idx2_13);

            // Emit: OutQ(x, e, o)
            output.emit_direct(x, e, o);
          }
        }
      }
    }
  }
};

// End of JIT batch file
