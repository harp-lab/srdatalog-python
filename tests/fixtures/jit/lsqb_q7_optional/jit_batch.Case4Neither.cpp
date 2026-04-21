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
// JIT-Generated Kernel Functor: Case4Neither
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Case4Neither {
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;

  template <typename Tile, typename Views, typename ValueType, typename Output>
  __device__ void operator()(Tile& tile, const Views* views,
                             const ValueType* __restrict__ root_unique_values,
                             uint32_t num_unique_root_keys, uint32_t num_root_keys,
                             uint32_t warp_id, uint32_t num_warps, Output& output) const {
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 4 unique views)
    auto view_HasTag_0_1_FULL_VER = views[0];
    auto view_HasCreator_0_1_FULL_VER = views[1];
    auto view_HasReply_0_FULL_VER = views[4];
    auto view_HasLiker_0_FULL_VER = views[5];

    // Root ColumnJoin (multi-source intersection): bind 'm' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var m :sources ((HasTag :handle 0) (HasCreator :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_HasTag_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_HasTag_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_HasTag_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_HasTag_0_1_FULL_VER.num_rows_;
      auto h_HasTag_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_HasTag_0_1_FULL_VER);
      if (!h_HasTag_0_root.valid())
        continue;
      auto h_HasCreator_1_root = HandleType(0, view_HasCreator_0_1_FULL_VER.num_rows_, 0)
                                     .prefix(root_val_2, tile, view_HasCreator_0_1_FULL_VER);
      if (!h_HasCreator_1_root.valid())
        continue;
      auto m = root_val_2;
      // Nested CartesianJoin: bind t, c from 2 source(s)
      // MIR: (cartesian-join :vars (t c) :sources ((HasTag :handle 2 :prefix (m)) (HasCreator
      // :handle 3 :prefix (m)) ))
      uint32_t lane_3 = tile.thread_rank();
      uint32_t group_size_4 = tile.size();

      auto h_HasTag_2_6 = h_HasTag_0_root;          // reusing narrowed handle
      auto h_HasCreator_3_8 = h_HasCreator_1_root;  // reusing narrowed handle

      if (!h_HasTag_2_6.valid() || !h_HasCreator_3_8.valid())
        continue;

      uint32_t degree_5 = h_HasTag_2_6.degree();
      uint32_t degree_7 = h_HasCreator_3_8.degree();
      uint32_t total_9 = degree_5 * degree_7;
      if (total_9 == 0)
        continue;

      // Pre-narrow negation handle for HasReply (pre-Cartesian vars: m)
      auto h_HasReply_neg_pre_1 = HandleType(0, view_HasReply_0_FULL_VER.num_rows_, 0)
                                      .prefix(m, tile, view_HasReply_0_FULL_VER);

      // Pre-narrow negation handle for HasLiker (pre-Cartesian vars: m)
      auto h_HasLiker_neg_pre_2 = HandleType(0, view_HasLiker_0_FULL_VER.num_rows_, 0)
                                      .prefix(m, tile, view_HasLiker_0_FULL_VER);

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

        auto t = view_HasTag_0_1_FULL_VER.get_value(1, h_HasTag_2_6.begin() + idx0_11);
        auto c = view_HasCreator_0_1_FULL_VER.get_value(1, h_HasCreator_3_8.begin() + idx1_12);

        // Negation: NOT EXISTS in HasReply
        // MIR: (negation :rel HasReply :prefix (m) :handle 4)
        // Using pre-narrowed handle (pre-Cartesian vars: m)
        if (!h_HasReply_neg_pre_1.valid()) {
          // Negation: NOT EXISTS in HasLiker
          // MIR: (negation :rel HasLiker :prefix (m) :handle 5)
          // Using pre-narrowed handle (pre-Cartesian vars: m)
          if (!h_HasLiker_neg_pre_2.valid()) {
            // Emit: Case4(m, t, c)
            output.emit_direct(m, t, c);
          }
        }
      }
    }
  }
};

// End of JIT batch file
