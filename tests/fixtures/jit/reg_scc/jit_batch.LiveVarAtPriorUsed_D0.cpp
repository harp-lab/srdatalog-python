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
// JIT-Generated Kernel Functor: LiveVarAtPriorUsed_D0
// Handles: 7
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_LiveVarAtPriorUsed_D0 {
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
    auto view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER = views[0];
    auto view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER = views[1];
    auto view_RegDefUseDefinedInBlock_0_1_FULL_VER = views[6];

    // Root ColumnJoin (multi-source intersection): bind 'blk' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var blk :sources ((RegDefUseLiveVarAtBlockEnd :handle 0)
    // (RegDefUseUsedInBlock :handle 1) )) WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER.num_rows_ -
                           (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER.num_rows_;
      auto h_RegDefUseLiveVarAtBlockEnd_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0)
              .prefix(root_val_2, tile, view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER);
      if (!h_RegDefUseLiveVarAtBlockEnd_0_root.valid())
        continue;
      auto h_RegDefUseUsedInBlock_1_root =
          HandleType(0, view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER.num_rows_, 0)
              .prefix(root_val_2, tile, view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER);
      if (!h_RegDefUseUsedInBlock_1_root.valid())
        continue;
      auto blk = root_val_2;
      // Nested ColumnJoin (intersection): bind 'mvar' from 2 sources
      // MIR: (column-join :var mvar :sources ((RegDefUseLiveVarAtBlockEnd :handle 2 :prefix (blk))
      // (RegDefUseUsedInBlock :handle 3 :prefix (blk)) ))
      auto h_RegDefUseLiveVarAtBlockEnd_2_13 = h_RegDefUseLiveVarAtBlockEnd_0_root;
      auto h_RegDefUseUsedInBlock_3_14 = h_RegDefUseUsedInBlock_1_root;
      auto intersect_15 = intersect_handles(
          tile,
          h_RegDefUseLiveVarAtBlockEnd_2_13.iterators(
              view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER),
          h_RegDefUseUsedInBlock_3_14.iterators(view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER));
      for (auto it_16 = intersect_15.begin(); it_16.valid(); it_16.next()) {
        auto mvar = it_16.value();
        auto positions = it_16.positions();
        auto ch_RegDefUseLiveVarAtBlockEnd_2_mvar = h_RegDefUseLiveVarAtBlockEnd_2_13.child_range(
            positions[0], mvar, tile, view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER);
        auto ch_RegDefUseUsedInBlock_3_mvar = h_RegDefUseUsedInBlock_3_14.child_range(
            positions[1], mvar, tile, view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER);
        // Nested CartesianJoin: bind blockUsed, eaUsed, _gen9 from 2 source(s)
        // MIR: (cartesian-join :vars (blockUsed eaUsed _gen9) :sources ((RegDefUseLiveVarAtBlockEnd
        // :handle 4 :prefix (blk mvar)) (RegDefUseUsedInBlock :handle 5 :prefix (blk mvar)) ))
        uint32_t lane_2 = tile.thread_rank();
        uint32_t group_size_3 = tile.size();

        auto h_RegDefUseLiveVarAtBlockEnd_4_5 =
            ch_RegDefUseLiveVarAtBlockEnd_2_mvar;  // reusing narrowed handle
        auto h_RegDefUseUsedInBlock_5_7 =
            ch_RegDefUseUsedInBlock_3_mvar;  // reusing narrowed handle

        if (!h_RegDefUseLiveVarAtBlockEnd_4_5.valid() || !h_RegDefUseUsedInBlock_5_7.valid())
          continue;

        uint32_t degree_4 = h_RegDefUseLiveVarAtBlockEnd_4_5.degree();
        uint32_t degree_6 = h_RegDefUseUsedInBlock_5_7.degree();
        uint32_t total_8 = degree_4 * degree_6;
        if (total_8 == 0)
          continue;

        // Pre-narrow negation handle for RegDefUseDefinedInBlock (pre-Cartesian vars: blk, mvar)
        auto h_RegDefUseDefinedInBlock_neg_pre_1 =
            HandleType(0, view_RegDefUseDefinedInBlock_0_1_FULL_VER.num_rows_, 0)
                .prefix(blk, tile, view_RegDefUseDefinedInBlock_0_1_FULL_VER)
                .prefix(mvar, tile, view_RegDefUseDefinedInBlock_0_1_FULL_VER);

        for (uint32_t flat_idx_9 = lane_2; flat_idx_9 < total_8; flat_idx_9 += group_size_3) {
          const bool major_is_1_12 = (degree_6 >= degree_4);
          uint32_t idx0_10, idx1_11;
          if (major_is_1_12) {
            idx0_10 = flat_idx_9 / degree_6;
            idx1_11 = flat_idx_9 % degree_6;
          } else {
            idx1_11 = flat_idx_9 / degree_4;
            idx0_10 = flat_idx_9 % degree_4;
          }

          auto blockUsed = view_RegDefUseLiveVarAtBlockEnd_0_2_1_DELTA_VER.get_value(
              2, h_RegDefUseLiveVarAtBlockEnd_4_5.begin() + idx0_10);
          auto eaUsed = view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER.get_value(
              2, h_RegDefUseUsedInBlock_5_7.begin() + idx1_11);
          auto _gen9 = view_RegDefUseUsedInBlock_0_2_1_3_FULL_VER.get_value(
              3, h_RegDefUseUsedInBlock_5_7.begin() + idx1_11);

          // Negation: NOT EXISTS in RegDefUseDefinedInBlock
          // MIR: (negation :rel RegDefUseDefinedInBlock :prefix (blk mvar) :handle 6)
          // Using pre-narrowed handle (pre-Cartesian vars: blk, mvar)
          if (!h_RegDefUseDefinedInBlock_neg_pre_1.valid()) {
            // Emit: RegDefUseLiveVarAtPriorUsed(eaUsed, blockUsed, mvar)
            output.emit_direct(eaUsed, blockUsed, mvar);
          }
        }
      }
    }
  }
};

// End of JIT batch file
