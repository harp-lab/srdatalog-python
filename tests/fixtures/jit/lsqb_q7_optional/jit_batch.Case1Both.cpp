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
// JIT-Generated Kernel Functor: Case1Both
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_Case1Both {
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

        // View declarations (deduplicated by spec, 4 unique views)
        auto view_HasTag_0_1_FULL_VER = views[0];
        auto view_ReplyOf_1_0_FULL_VER = views[1];
        auto view_Likes_1_0_FULL_VER = views[2];
        auto view_HasCreator_0_1_FULL_VER = views[3];

        // Root ColumnJoin (multi-source intersection): bind 'm' from 4 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var m :sources ((HasTag :handle 0) (ReplyOf :handle 1) (Likes :handle 2) (HasCreator :handle 3) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_HasTag_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_HasTag_0_1_FULL_VER.num_rows_) ? hint_hi_4 : view_HasTag_0_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_HasTag_0_1_FULL_VER.num_rows_;
          auto h_HasTag_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_HasTag_0_1_FULL_VER);
          if (!h_HasTag_0_root.valid()) continue;
          auto h_ReplyOf_1_root = HandleType(0, view_ReplyOf_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ReplyOf_1_0_FULL_VER);
          if (!h_ReplyOf_1_root.valid()) continue;
          auto h_Likes_2_root = HandleType(0, view_Likes_1_0_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_Likes_1_0_FULL_VER);
          if (!h_Likes_2_root.valid()) continue;
          auto h_HasCreator_3_root = HandleType(0, view_HasCreator_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_HasCreator_0_1_FULL_VER);
          if (!h_HasCreator_3_root.valid()) continue;
          auto m = root_val_2;
        // Nested CartesianJoin: bind comment, liker, t, c from 4 source(s)
        // MIR: (cartesian-join :vars (comment liker t c) :sources ((HasTag :handle 4 :prefix (m)) (ReplyOf :handle 5 :prefix (m)) (Likes :handle 6 :prefix (m)) (HasCreator :handle 7 :prefix (m)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_HasTag_4_4 = h_HasTag_0_root;  // reusing narrowed handle
        auto h_ReplyOf_5_6 = h_ReplyOf_1_root;  // reusing narrowed handle
        auto h_Likes_6_8 = h_Likes_2_root;  // reusing narrowed handle
        auto h_HasCreator_7_10 = h_HasCreator_3_root;  // reusing narrowed handle

        if (!h_HasTag_4_4.valid() || !h_ReplyOf_5_6.valid() || !h_Likes_6_8.valid() || !h_HasCreator_7_10.valid()) continue;

        uint32_t degree_3 = h_HasTag_4_4.degree();
        uint32_t degree_5 = h_ReplyOf_5_6.degree();
        uint32_t degree_7 = h_Likes_6_8.degree();
        uint32_t degree_9 = h_HasCreator_7_10.degree();
        uint32_t total_11 = degree_3 * degree_5 * degree_7 * degree_9;
        if (total_11 == 0) continue;

        for (uint32_t flat_idx_12 = lane_1; flat_idx_12 < total_11; flat_idx_12 += group_size_2) {
          uint32_t remaining = flat_idx_12;
          uint32_t idx3_16 = remaining % degree_9;
          remaining /= degree_9;
          uint32_t idx2_15 = remaining % degree_7;
          remaining /= degree_7;
          uint32_t idx1_14 = remaining % degree_5;
          remaining /= degree_5;
          uint32_t idx0_13 = remaining % degree_3;

          auto t = view_HasTag_0_1_FULL_VER.get_value(1, h_HasTag_4_4.begin() + idx0_13);
          auto comment = view_ReplyOf_1_0_FULL_VER.get_value(1, h_ReplyOf_5_6.begin() + idx1_14);
          auto liker = view_Likes_1_0_FULL_VER.get_value(1, h_Likes_6_8.begin() + idx2_15);
          auto c = view_HasCreator_0_1_FULL_VER.get_value(1, h_HasCreator_7_10.begin() + idx3_16);

        // Emit: Case1(m, t, c, comment, liker)
        output.emit_direct(m, t, c, comment, liker);
        }
        }
  }
};

// End of JIT batch file
