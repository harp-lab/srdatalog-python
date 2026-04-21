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
// JIT-Generated Kernel Functor: subset_error_rule
// Handles: 6
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_subset_error_rule {
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
    auto view_subset_0_1_2_FULL_VER = views[0];
    auto view_placeholder_origin_0_FULL_VER = views[1];
    auto view_known_placeholder_subset_0_1_FULL_VER = views[5];

    // Root ColumnJoin (multi-source intersection): bind 'origin1' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var origin1 :sources ((subset :handle 0) (placeholder_origin :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_subset_0_1_2_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_subset_0_1_2_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_subset_0_1_2_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_0_1_2_FULL_VER.num_rows_;
      auto h_subset_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_0_1_2_FULL_VER);
      if (!h_subset_0_root.valid())
        continue;
      auto h_placeholder_origin_1_root =
          HandleType(0, view_placeholder_origin_0_FULL_VER.num_rows_, 0)
              .prefix(root_val_2, tile, view_placeholder_origin_0_FULL_VER);
      if (!h_placeholder_origin_1_root.valid())
        continue;
      auto origin1 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'origin2' from 2 sources
      // MIR: (column-join :var origin2 :sources ((subset :handle 2 :prefix (origin1))
      // (placeholder_origin :handle 3 :prefix ()) ))
      auto h_subset_2_9 = h_subset_0_root;
      auto h_placeholder_origin_3_10 =
          HandleType(0, view_placeholder_origin_0_FULL_VER.num_rows_, 0);
      auto intersect_11 = intersect_handles(
          tile, h_subset_2_9.iterators(view_subset_0_1_2_FULL_VER),
          h_placeholder_origin_3_10.iterators(view_placeholder_origin_0_FULL_VER));
      for (auto it_12 = intersect_11.begin(); it_12.valid(); it_12.next()) {
        auto origin2 = it_12.value();
        auto positions = it_12.positions();
        auto ch_subset_2_origin2 =
            h_subset_2_9.child_range(positions[0], origin2, tile, view_subset_0_1_2_FULL_VER);
        auto ch_placeholder_origin_3_origin2 = h_placeholder_origin_3_10.child_range(
            positions[1], origin2, tile, view_placeholder_origin_0_FULL_VER);
        // Nested CartesianJoin: bind point from 1 source(s)
        // MIR: (cartesian-join :vars (point) :sources ((subset :handle 4 :prefix (origin1 origin2))
        // ))
        uint32_t lane_2 = tile.thread_rank();
        uint32_t group_size_3 = tile.size();

        auto h_subset_4_5 = ch_subset_2_origin2;  // reusing narrowed handle

        if (!h_subset_4_5.valid())
          continue;

        uint32_t degree_4 = h_subset_4_5.degree();
        uint32_t total_6 = degree_4;
        if (total_6 == 0)
          continue;

        // Pre-narrow negation handle for known_placeholder_subset (pre-Cartesian vars: origin1,
        // origin2)
        auto h_known_placeholder_subset_neg_pre_1 =
            HandleType(0, view_known_placeholder_subset_0_1_FULL_VER.num_rows_, 0)
                .prefix(origin1, tile, view_known_placeholder_subset_0_1_FULL_VER)
                .prefix(origin2, tile, view_known_placeholder_subset_0_1_FULL_VER);

        for (uint32_t flat_idx_7 = lane_2; flat_idx_7 < total_6; flat_idx_7 += group_size_3) {
          uint32_t idx0_8 = flat_idx_7;

          auto point = view_subset_0_1_2_FULL_VER.get_value(2, h_subset_4_5.begin() + idx0_8);

          // Negation: NOT EXISTS in known_placeholder_subset
          // MIR: (negation :rel known_placeholder_subset :prefix (origin1 origin2) :handle 5)
          // Using pre-narrowed handle (pre-Cartesian vars: origin1, origin2)
          if (!h_known_placeholder_subset_neg_pre_1.valid()) {
            if (origin1 != origin2) {
              // Emit: subset_error(origin1, origin2, point)
              output.emit_direct(origin1, origin2, point);
            }
          }
        }
      }
    }
  }
};

// End of JIT batch file
