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
// JIT-Generated Kernel Functor: RegDefUsed3_D0
// Handles: 9
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_RegDefUsed3_D0 {
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
    auto view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER = views[0];
    auto view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[1];
    auto view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER = views[4];

    // Root ColumnJoin (multi-source intersection): bind 'eaUsed' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var eaUsed :sources ((RegDefUseLiveVarAtPriorUsed :handle 0)
    // (RegDefUseDefUsed :handle 1) )) WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER.num_rows_ -
                           (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER.num_rows_;
      auto h_RegDefUseLiveVarAtPriorUsed_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0)
              .prefix(root_val_2, tile, view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER);
      if (!h_RegDefUseLiveVarAtPriorUsed_0_root.valid())
        continue;
      auto h_RegDefUseDefUsed_1_root =
          HandleType(0, view_RegDefUseDefUsed_2_1_0_3_FULL_VER.num_rows_, 0)
              .prefix(root_val_2, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
      if (!h_RegDefUseDefUsed_1_root.valid())
        continue;
      auto eaUsed = root_val_2;
      // Nested ColumnJoin (intersection): bind 'mvar' from 3 sources
      // MIR: (column-join :var mvar :sources ((RegDefUseLiveVarAtPriorUsed :handle 2 :prefix
      // (eaUsed)) (RegDefUseDefUsed :handle 3 :prefix (eaUsed)) (RegDefUseLiveVarUsed :handle 4
      // :prefix ()) ))
      auto h_RegDefUseLiveVarAtPriorUsed_2_16 = h_RegDefUseLiveVarAtPriorUsed_0_root;
      auto h_RegDefUseDefUsed_3_17 = h_RegDefUseDefUsed_1_root;
      auto h_RegDefUseLiveVarUsed_4_18 =
          HandleType(0, view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER.num_rows_, 0);
      auto intersect_19 = intersect_handles(
          tile,
          h_RegDefUseLiveVarAtPriorUsed_2_16.iterators(
              view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER),
          h_RegDefUseDefUsed_3_17.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER),
          h_RegDefUseLiveVarUsed_4_18.iterators(view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER));
      for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
        auto mvar = it_20.value();
        auto positions = it_20.positions();
        auto ch_RegDefUseLiveVarAtPriorUsed_2_mvar = h_RegDefUseLiveVarAtPriorUsed_2_16.child_range(
            positions[0], mvar, tile, view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER);
        auto ch_RegDefUseDefUsed_3_mvar = h_RegDefUseDefUsed_3_17.child_range(
            positions[1], mvar, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        auto ch_RegDefUseLiveVarUsed_4_mvar = h_RegDefUseLiveVarUsed_4_18.child_range(
            positions[2], mvar, tile, view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'nxtBlk' from 2 sources
        // MIR: (column-join :var nxtBlk :sources ((RegDefUseLiveVarAtPriorUsed :handle 5 :prefix
        // (eaUsed mvar)) (RegDefUseLiveVarUsed :handle 6 :prefix (mvar)) ))
        auto h_RegDefUseLiveVarAtPriorUsed_5_12 = ch_RegDefUseLiveVarAtPriorUsed_2_mvar;
        auto h_RegDefUseLiveVarUsed_6_13 = ch_RegDefUseLiveVarUsed_4_mvar;
        auto intersect_14 = intersect_handles(
            tile,
            h_RegDefUseLiveVarAtPriorUsed_5_12.iterators(
                view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER),
            h_RegDefUseLiveVarUsed_6_13.iterators(view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto nxtBlk = it_15.value();
          auto positions = it_15.positions();
          auto ch_RegDefUseLiveVarAtPriorUsed_5_nxtBlk =
              h_RegDefUseLiveVarAtPriorUsed_5_12.child_range(
                  positions[0], nxtBlk, tile, view_RegDefUseLiveVarAtPriorUsed_0_2_1_DELTA_VER);
          auto ch_RegDefUseLiveVarUsed_6_nxtBlk = h_RegDefUseLiveVarUsed_6_13.child_range(
              positions[1], nxtBlk, tile, view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER);
          // Nested CartesianJoin: bind eaDef, nextEaUsed, nextIndex, _gen1 from 2 source(s)
          // MIR: (cartesian-join :vars (eaDef nextEaUsed nextIndex _gen1) :sources
          // ((RegDefUseDefUsed :handle 7 :prefix (eaUsed mvar)) (RegDefUseLiveVarUsed :handle 8
          // :prefix (mvar nxtBlk)) ))
          uint32_t lane_1 = tile.thread_rank();
          uint32_t group_size_2 = tile.size();

          auto h_RegDefUseDefUsed_7_4 = ch_RegDefUseDefUsed_3_mvar;  // reusing narrowed handle
          auto h_RegDefUseLiveVarUsed_8_6 =
              ch_RegDefUseLiveVarUsed_6_nxtBlk;  // reusing narrowed handle

          if (!h_RegDefUseDefUsed_7_4.valid() || !h_RegDefUseLiveVarUsed_8_6.valid())
            continue;

          uint32_t degree_3 = h_RegDefUseDefUsed_7_4.degree();
          uint32_t degree_5 = h_RegDefUseLiveVarUsed_8_6.degree();
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

            auto eaDef = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(
                2, h_RegDefUseDefUsed_7_4.begin() + idx0_9);
            auto _gen1 = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(
                3, h_RegDefUseDefUsed_7_4.begin() + idx0_9);
            auto nextEaUsed = view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER.get_value(
                2, h_RegDefUseLiveVarUsed_8_6.begin() + idx1_10);
            auto nextIndex = view_RegDefUseLiveVarUsed_1_0_2_3_FULL_VER.get_value(
                3, h_RegDefUseLiveVarUsed_8_6.begin() + idx1_10);

            // Emit: RegDefUseDefUsed(eaDef, mvar, nextEaUsed, nextIndex)
            output.emit_direct(eaDef, mvar, nextEaUsed, nextIndex);
          }
        }
      }
    }
  }
};

// End of JIT batch file
