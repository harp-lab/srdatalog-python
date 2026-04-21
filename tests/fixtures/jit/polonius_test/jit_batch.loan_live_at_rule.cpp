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
// JIT-Generated Kernel Functor: loan_live_at_rule
// Handles: 5
// =============================================================

// WARP MODE: 32 threads share 1 row, cooperative search
struct Kernel_loan_live_at_rule {
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

        // View declarations (deduplicated by spec, 2 unique views)
        auto view_origin_contains_loan_on_entry_0_2_1_FULL_VER = views[0];
        auto view_origin_live_on_entry_0_1_FULL_VER = views[1];

        // Root ColumnJoin (multi-source intersection): bind 'origin' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var origin :sources ((origin_contains_loan_on_entry :handle 0) (origin_live_on_entry :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_origin_contains_loan_on_entry_0_2_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_origin_contains_loan_on_entry_0_2_1_FULL_VER.num_rows_) ? hint_hi_4 : view_origin_contains_loan_on_entry_0_2_1_FULL_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_origin_contains_loan_on_entry_0_2_1_FULL_VER.num_rows_;
          auto h_origin_contains_loan_on_entry_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_origin_contains_loan_on_entry_0_2_1_FULL_VER);
          if (!h_origin_contains_loan_on_entry_0_root.valid()) continue;
          auto h_origin_live_on_entry_1_root = HandleType(0, view_origin_live_on_entry_0_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_origin_live_on_entry_0_1_FULL_VER);
          if (!h_origin_live_on_entry_1_root.valid()) continue;
          auto origin = root_val_2;
        // Nested ColumnJoin (intersection): bind 'point' from 2 sources
        // MIR: (column-join :var point :sources ((origin_contains_loan_on_entry :handle 2 :prefix (origin)) (origin_live_on_entry :handle 3 :prefix (origin)) ))
        auto h_origin_contains_loan_on_entry_2_8 = h_origin_contains_loan_on_entry_0_root;
        auto h_origin_live_on_entry_3_9 = h_origin_live_on_entry_1_root;
        auto intersect_10 = intersect_handles(tile, h_origin_contains_loan_on_entry_2_8.iterators(view_origin_contains_loan_on_entry_0_2_1_FULL_VER), h_origin_live_on_entry_3_9.iterators(view_origin_live_on_entry_0_1_FULL_VER));
        for (auto it_11 = intersect_10.begin(); it_11.valid(); it_11.next()) {
          auto point = it_11.value();
          auto positions = it_11.positions();
          auto ch_origin_contains_loan_on_entry_2_point = h_origin_contains_loan_on_entry_2_8.child_range(positions[0], point, tile, view_origin_contains_loan_on_entry_0_2_1_FULL_VER);
          auto ch_origin_live_on_entry_3_point = h_origin_live_on_entry_3_9.child_range(positions[1], point, tile, view_origin_live_on_entry_0_1_FULL_VER);
        // Nested CartesianJoin: bind loan from 1 source(s)
        // MIR: (cartesian-join :vars (loan) :sources ((origin_contains_loan_on_entry :handle 4 :prefix (origin point)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_origin_contains_loan_on_entry_4_4 = ch_origin_contains_loan_on_entry_2_point;  // reusing narrowed handle

        if (!h_origin_contains_loan_on_entry_4_4.valid()) continue;

        uint32_t degree_3 = h_origin_contains_loan_on_entry_4_4.degree();
        uint32_t total_5 = degree_3;
        if (total_5 == 0) continue;

        for (uint32_t flat_idx_6 = lane_1; flat_idx_6 < total_5; flat_idx_6 += group_size_2) {
          uint32_t idx0_7 = flat_idx_6;

          auto loan = view_origin_contains_loan_on_entry_0_2_1_FULL_VER.get_value(2, h_origin_contains_loan_on_entry_4_4.begin() + idx0_7);

        // Emit: loan_live_at(loan, point)
        output.emit_direct(loan, point);
        }
        }
        }
  }
};

// End of JIT batch file
