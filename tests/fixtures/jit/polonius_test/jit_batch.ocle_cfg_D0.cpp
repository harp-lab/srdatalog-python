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
// JIT-Generated Kernel Functor: ocle_cfg_D0
// Handles: 8
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_ocle_cfg_D0 {
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
        auto view_origin_contains_loan_on_entry_0_2_1_DELTA_VER = views[0];
        auto view_origin_live_on_entry_0_1_FULL_VER = views[1];
        auto view_cfg_edge_0_1_FULL_VER = views[3];
        auto view_loan_killed_at_1_0_FULL_VER = views[7];

        // Root ColumnJoin (multi-source intersection): bind 'origin' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var origin :sources ((origin_contains_loan_on_entry :handle 0) (origin_live_on_entry :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_origin_contains_loan_on_entry_0_2_1_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_origin_contains_loan_on_entry_0_2_1_DELTA_VER.num_rows_) ? hint_hi_4 : view_origin_contains_loan_on_entry_0_2_1_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_origin_contains_loan_on_entry_0_2_1_DELTA_VER.num_rows_;
          auto h_origin_contains_loan_on_entry_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_origin_contains_loan_on_entry_0_2_1_DELTA_VER);
          if (!h_origin_contains_loan_on_entry_0_root.valid()) continue;
          auto h_origin_live_on_entry_1_root = HandleType(0, view_origin_live_on_entry_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_origin_live_on_entry_0_1_FULL_VER);
          if (!h_origin_live_on_entry_1_root.valid()) continue;
          auto origin = root_val_2;
        // Nested ColumnJoin (intersection): bind 'point1' from 2 sources
        // MIR: (column-join :var point1 :sources ((origin_contains_loan_on_entry :handle 2 :prefix (origin)) (cfg_edge :handle 3 :prefix ()) ))
        auto h_origin_contains_loan_on_entry_2_14 = h_origin_contains_loan_on_entry_0_root;
        auto h_cfg_edge_3_15 = HandleType(0, view_cfg_edge_0_1_FULL_VER.num_rows_, 0);
        auto intersect_16 = intersect_handles(tile, h_origin_contains_loan_on_entry_2_14.iterators(view_origin_contains_loan_on_entry_0_2_1_DELTA_VER), h_cfg_edge_3_15.iterators(view_cfg_edge_0_1_FULL_VER));
        for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
          auto point1 = it_17.value();
          auto positions = it_17.positions();
          auto ch_origin_contains_loan_on_entry_2_point1 = h_origin_contains_loan_on_entry_2_14.child_range(positions[0], point1, tile, view_origin_contains_loan_on_entry_0_2_1_DELTA_VER);
          auto ch_cfg_edge_3_point1 = h_cfg_edge_3_15.child_range(positions[1], point1, tile, view_cfg_edge_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'point2' from 2 sources
        // MIR: (column-join :var point2 :sources ((cfg_edge :handle 4 :prefix (point1)) (origin_live_on_entry :handle 5 :prefix (origin)) ))
        auto h_cfg_edge_4_10 = ch_cfg_edge_3_point1;
        auto h_origin_live_on_entry_5_11 = h_origin_live_on_entry_1_root;
        auto intersect_12 = intersect_handles(tile, h_cfg_edge_4_10.iterators(view_cfg_edge_0_1_FULL_VER), h_origin_live_on_entry_5_11.iterators(view_origin_live_on_entry_0_1_FULL_VER));
        for (auto it_13 = intersect_12.begin(); it_13.valid(); it_13.next()) {
          auto point2 = it_13.value();
          auto positions = it_13.positions();
          auto ch_cfg_edge_4_point2 = h_cfg_edge_4_10.child_range(positions[0], point2, tile, view_cfg_edge_0_1_FULL_VER);
          auto ch_origin_live_on_entry_5_point2 = h_origin_live_on_entry_5_11.child_range(positions[1], point2, tile, view_origin_live_on_entry_0_1_FULL_VER);
        // Nested CartesianJoin: bind loan from 1 source(s)
        // MIR: (cartesian-join :vars (loan) :sources ((origin_contains_loan_on_entry :handle 6 :prefix (origin point1)) ))
        uint32_t lane_3 = tile.thread_rank();
        uint32_t group_size_4 = tile.size();

        auto h_origin_contains_loan_on_entry_6_6 = ch_origin_contains_loan_on_entry_2_point1;  // reusing narrowed handle

        if (!h_origin_contains_loan_on_entry_6_6.valid()) continue;

        uint32_t degree_5 = h_origin_contains_loan_on_entry_6_6.degree();
        uint32_t total_7 = degree_5;
        if (total_7 == 0) continue;

        // Pre-narrow negation handle for loan_killed_at (pre-Cartesian vars: point1)
        auto h_loan_killed_at_neg_pre_1 = HandleType(0, view_loan_killed_at_1_0_FULL_VER.num_rows_, 0).prefix(point1, tile, view_loan_killed_at_1_0_FULL_VER);

        for (uint32_t flat_idx_8 = lane_3; flat_idx_8 < total_7; flat_idx_8 += group_size_4) {
          uint32_t idx0_9 = flat_idx_8;

          auto loan = view_origin_contains_loan_on_entry_0_2_1_DELTA_VER.get_value(2, h_origin_contains_loan_on_entry_6_6.begin() + idx0_9);

        // Negation: NOT EXISTS in loan_killed_at
        // MIR: (negation :rel loan_killed_at :prefix (point1 loan) :handle 7)
        // Using pre-narrowed handle (pre-Cartesian vars: point1)
        auto h_loan_killed_at_neg_7_2 = h_loan_killed_at_neg_pre_1.prefix_seq(loan, view_loan_killed_at_1_0_FULL_VER);
        if (!h_loan_killed_at_neg_7_2.valid()) {
        // Emit: origin_contains_loan_on_entry(origin, loan, point2)
        output.emit_direct(origin, loan, point2);
        }
        }
        }
        }
        }
  }
};

// End of JIT batch file
