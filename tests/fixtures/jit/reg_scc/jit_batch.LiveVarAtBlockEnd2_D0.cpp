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
// JIT-Generated Kernel Functor: LiveVarAtBlockEnd2_D0
// Handles: 5
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_LiveVarAtBlockEnd2_D0 {
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
    auto view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER = views[0];
    auto view_BlockNext_2_0_1_FULL_VER = views[1];
    auto view_RegDefUseRefInBlock_0_1_FULL_VER = views[4];

    // Root ColumnJoin (multi-source intersection): bind 'blk' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var blk :sources ((RegDefUseLiveVarAtBlockEnd :handle 0) (BlockNext
    // :handle 1) )) WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.num_rows_ -
                           (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3)
                      ? hint_hi_4
                      : view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.num_rows_;
      auto h_RegDefUseLiveVarAtBlockEnd_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0)
              .prefix(root_val_2, tile, view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER);
      if (!h_RegDefUseLiveVarAtBlockEnd_0_root.valid())
        continue;
      auto h_BlockNext_1_root = HandleType(0, view_BlockNext_2_0_1_FULL_VER.num_rows_, 0)
                                    .prefix(root_val_2, tile, view_BlockNext_2_0_1_FULL_VER);
      if (!h_BlockNext_1_root.valid())
        continue;
      auto blk = root_val_2;
      // Nested CartesianJoin: bind blockUsed, mvar, prevBlock, _gen13 from 2 source(s)
      // MIR: (cartesian-join :vars (blockUsed mvar prevBlock _gen13) :sources
      // ((RegDefUseLiveVarAtBlockEnd :handle 2 :prefix (blk)) (BlockNext :handle 3 :prefix (blk))
      // ))
      uint32_t lane_3 = tile.thread_rank();
      uint32_t group_size_4 = tile.size();

      auto h_RegDefUseLiveVarAtBlockEnd_2_6 =
          h_RegDefUseLiveVarAtBlockEnd_0_root;    // reusing narrowed handle
      auto h_BlockNext_3_8 = h_BlockNext_1_root;  // reusing narrowed handle

      if (!h_RegDefUseLiveVarAtBlockEnd_2_6.valid() || !h_BlockNext_3_8.valid())
        continue;

      uint32_t degree_5 = h_RegDefUseLiveVarAtBlockEnd_2_6.degree();
      uint32_t degree_7 = h_BlockNext_3_8.degree();
      uint32_t total_9 = degree_5 * degree_7;
      if (total_9 == 0)
        continue;

      // Pre-narrow negation handle for RegDefUseRefInBlock (pre-Cartesian vars: blk)
      auto h_RegDefUseRefInBlock_neg_pre_1 =
          HandleType(0, view_RegDefUseRefInBlock_0_1_FULL_VER.num_rows_, 0)
              .prefix(blk, tile, view_RegDefUseRefInBlock_0_1_FULL_VER);

      for (uint32_t flat_idx_10 = lane_3; flat_idx_10 < total_9; flat_idx_10 += group_size_4) {
        const bool major_is_1_13 = (degree_7 >= degree_5);
        uint32_t idx0_11, idx1_12;
        if (major_is_1_13) {
          idx0_11 = flat_idx_10 / degree_7;
          idx1_12 = flat_idx_10 % degree_7;
        } else {
          idx1_12 = flat_idx_10 / degree_5;
          idx0_11 = flat_idx_10 % degree_5;
        }

        auto blockUsed = view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.get_value(
            1, h_RegDefUseLiveVarAtBlockEnd_2_6.begin() + idx0_11);
        auto mvar = view_RegDefUseLiveVarAtBlockEnd_0_1_2_DELTA_VER.get_value(
            2, h_RegDefUseLiveVarAtBlockEnd_2_6.begin() + idx0_11);
        auto prevBlock =
            view_BlockNext_2_0_1_FULL_VER.get_value(1, h_BlockNext_3_8.begin() + idx1_12);
        auto _gen13 = view_BlockNext_2_0_1_FULL_VER.get_value(2, h_BlockNext_3_8.begin() + idx1_12);

        // Negation: NOT EXISTS in RegDefUseRefInBlock
        // MIR: (negation :rel RegDefUseRefInBlock :prefix (blk mvar) :handle 4)
        // Using pre-narrowed handle (pre-Cartesian vars: blk)
        auto h_RegDefUseRefInBlock_neg_4_2 =
            h_RegDefUseRefInBlock_neg_pre_1.prefix_seq(mvar, view_RegDefUseRefInBlock_0_1_FULL_VER);
        if (!h_RegDefUseRefInBlock_neg_4_2.valid()) {
          // Emit: RegDefUseLiveVarAtBlockEnd(prevBlock, blockUsed, mvar)
          output.emit_direct(prevBlock, blockUsed, mvar);
        }
      }
    }
  }
};

// End of JIT batch file
