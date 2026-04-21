// =============================================================
// JIT-Generated Complete Runner: subset_trans_D0
// Fully concrete - NO C++ templates/metaprogramming
// =============================================================

struct JitRunner_subset_trans_D0 {
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = PoloniusRules_DB_DeviceDB;
  using FirstSchema = subset;
  using DestSchema = subset;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType =
      std::decay_t<decltype(get_relation_by_schema<FirstSchema, DELTA_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 3;
  static constexpr std::size_t OutputArity = OutputArity_0;  // Legacy alias
  static constexpr std::size_t NumSources = 2;

  // Non-template kernel_count (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_count(const ViewType* __restrict__ views,
                   const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
                   uint32_t num_root_keys, uint32_t* __restrict__ thread_counts) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];
    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);
    if (threadIdx.x < NumSources) {
      s_views[threadIdx.x] = views[threadIdx.x];
    }
    __syncthreads();
    views = s_views;  // redirect to shared memory copy
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)

    using OutputCtx = SRDatalog::GPU::OutputContext<ValueType, SR, true, Layout, OutputArity_0>;
    OutputCtx output_ctx{nullptr, nullptr, 0, 0};

    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_subset_1_2_0_DELTA_VER = views[0];
    auto view_subset_base_0_2_1_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'origin2' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var origin2 :sources ((subset :handle 0) (subset_base :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_subset_1_2_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_subset_1_2_0_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_subset_1_2_0_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_1_2_0_DELTA_VER.num_rows_;
      auto h_subset_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_1_2_0_DELTA_VER);
      if (!h_subset_0_root.valid())
        continue;
      auto h_subset_base_1_root = HandleType(0, view_subset_base_0_2_1_FULL_VER.num_rows_, 0)
                                      .prefix(root_val_2, tile, view_subset_base_0_2_1_FULL_VER);
      if (!h_subset_base_1_root.valid())
        continue;
      auto origin2 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'point' from 2 sources
      // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2)) (subset_base
      // :handle 3 :prefix (origin2)) ))
      auto h_subset_2_12 = h_subset_0_root;
      auto h_subset_base_3_13 = h_subset_base_1_root;
      auto intersect_14 =
          intersect_handles(tile, h_subset_2_12.iterators(view_subset_1_2_0_DELTA_VER),
                            h_subset_base_3_13.iterators(view_subset_base_0_2_1_FULL_VER));
      for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
        auto point = it_15.value();
        auto positions = it_15.positions();
        auto ch_subset_2_point =
            h_subset_2_12.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
        auto ch_subset_base_3_point = h_subset_base_3_13.child_range(
            positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
        // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
        // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix (origin2
        // point)) (subset_base :handle 5 :prefix (origin2 point)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_subset_4_4 = ch_subset_2_point;            // reusing narrowed handle
        auto h_subset_base_5_6 = ch_subset_base_3_point;  // reusing narrowed handle

        if (!h_subset_4_4.valid() || !h_subset_base_5_6.valid())
          continue;

        uint32_t degree_3 = h_subset_4_4.degree();
        uint32_t degree_5 = h_subset_base_5_6.degree();
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

          auto origin1 = view_subset_1_2_0_DELTA_VER.get_value(2, h_subset_4_4.begin() + idx0_9);
          auto origin3 =
              view_subset_base_0_2_1_FULL_VER.get_value(2, h_subset_base_5_6.begin() + idx1_10);

          if (origin1 != origin3) {
            // Emit: subset(origin1, origin3, point)
            output_ctx.emit_direct();
          }
        }
      }
    }
    thread_counts[thread_id] = output_ctx.count();
  }

  // Non-template kernel_materialize (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_materialize(const ViewType* __restrict__ views,
                         const ValueType* __restrict__ root_unique_values,
                         uint32_t num_unique_root_keys, uint32_t num_root_keys,
                         const uint32_t* __restrict__ thread_offsets,
                         ValueType* __restrict__ output_data_0,
                         semiring_value_t<SR>* __restrict__ output_prov_0,
                         std::size_t output_stride_0, uint32_t old_size_0) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];
    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);
    if (threadIdx.x < NumSources) {
      s_views[threadIdx.x] = views[threadIdx.x];
    }
    __syncthreads();
    views = s_views;
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)
    uint32_t thread_offset = thread_offsets[thread_id];

    using OutputCtx_0 = SRDatalog::GPU::OutputContext<ValueType, SR, false, Layout, OutputArity_0>;
    OutputCtx_0 output_ctx_0{output_data_0, output_prov_0, output_stride_0,
                             old_size_0 + thread_offset};

    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_subset_1_2_0_DELTA_VER = views[0];
    auto view_subset_base_0_2_1_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'origin2' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var origin2 :sources ((subset :handle 0) (subset_base :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_subset_1_2_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_subset_1_2_0_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_subset_1_2_0_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_1_2_0_DELTA_VER.num_rows_;
      auto h_subset_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_1_2_0_DELTA_VER);
      if (!h_subset_0_root.valid())
        continue;
      auto h_subset_base_1_root = HandleType(0, view_subset_base_0_2_1_FULL_VER.num_rows_, 0)
                                      .prefix(root_val_2, tile, view_subset_base_0_2_1_FULL_VER);
      if (!h_subset_base_1_root.valid())
        continue;
      auto origin2 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'point' from 2 sources
      // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2)) (subset_base
      // :handle 3 :prefix (origin2)) ))
      auto h_subset_2_12 = h_subset_0_root;
      auto h_subset_base_3_13 = h_subset_base_1_root;
      auto intersect_14 =
          intersect_handles(tile, h_subset_2_12.iterators(view_subset_1_2_0_DELTA_VER),
                            h_subset_base_3_13.iterators(view_subset_base_0_2_1_FULL_VER));
      for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
        auto point = it_15.value();
        auto positions = it_15.positions();
        auto ch_subset_2_point =
            h_subset_2_12.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
        auto ch_subset_base_3_point = h_subset_base_3_13.child_range(
            positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
        // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
        // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix (origin2
        // point)) (subset_base :handle 5 :prefix (origin2 point)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_subset_4_4 = ch_subset_2_point;            // reusing narrowed handle
        auto h_subset_base_5_6 = ch_subset_base_3_point;  // reusing narrowed handle

        if (!h_subset_4_4.valid() || !h_subset_base_5_6.valid())
          continue;

        uint32_t degree_3 = h_subset_4_4.degree();
        uint32_t degree_5 = h_subset_base_5_6.degree();
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

          auto origin1 = view_subset_1_2_0_DELTA_VER.get_value(2, h_subset_4_4.begin() + idx0_9);
          auto origin3 =
              view_subset_base_0_2_1_FULL_VER.get_value(2, h_subset_base_5_6.begin() + idx1_10);

          if (origin1 != origin3) {
            // Emit: subset(origin1, origin3, point)
            output_ctx_0.emit_direct(origin1, origin3, point);
          }
        }
      }
    }
  }

  // WS Count Kernel (phase 1 + steal loop, atomic total)
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_count_ws(const ViewType* __restrict__ views,
                      const ValueType* __restrict__ root_unique_values,
                      uint32_t num_unique_root_keys, uint32_t num_root_keys,
                      uint32_t* __restrict__ global_count,
                      SRDatalog::GPU::JIT::WS::WCOJTaskQueue* __restrict__ wcoj_queue,
                      SRDatalog::GPU::JIT::WS::WCOJRangeBoard* __restrict__ wcoj_range_board) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;

    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;
    namespace ws = SRDatalog::GPU::JIT::WS;

    if (tile.thread_rank() == 0) {
      wcoj_queue->active_producers.fetch_add(1, cuda::memory_order_relaxed);
    }

    uint32_t local_count = 0;

    // Phase 1: Main traversal with inline WS splitting (count mode)
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_subset_1_2_0_DELTA_VER = views[0];
    auto view_subset_base_0_2_1_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'origin2' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var origin2 :sources ((subset :handle 0) (subset_base :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_subset_1_2_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_subset_1_2_0_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_subset_1_2_0_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_1_2_0_DELTA_VER.num_rows_;
      auto h_subset_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_1_2_0_DELTA_VER);
      if (!h_subset_0_root.valid())
        continue;
      auto h_subset_base_1_root = HandleType(0, view_subset_base_0_2_1_FULL_VER.num_rows_, 0)
                                      .prefix(root_val_2, tile, view_subset_base_0_2_1_FULL_VER);
      if (!h_subset_base_1_root.valid())
        continue;
      auto origin2 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'point' from 2 sources
      // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2)) (subset_base
      // :handle 3 :prefix (origin2)) ))
      auto h_subset_2_12 = h_subset_0_root;
      auto h_subset_base_3_13 = h_subset_base_1_root;
      // WS: bulk-donate CJ chunks to task queue
      if (h_subset_2_12.degree() > ws::WCOJ_SPLIT_THRESHOLD) {
        uint32_t _ws_total_deg = h_subset_2_12.degree();
        uint32_t _ws_K = (_ws_total_deg - 1) / ws::WCOJ_SPLIT_THRESHOLD;
        if (_ws_K > 0) {
          uint32_t _ws_base_slot = 0;
          if (tile.thread_rank() == 0) {
            if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_K <= wcoj_queue->capacity)
              _ws_base_slot = wcoj_queue->tail.fetch_add(_ws_K, cuda::memory_order_relaxed);
            else
              _ws_K = 0;
          }
          _ws_base_slot = tile.shfl(_ws_base_slot, 0);
          _ws_K = tile.shfl(_ws_K, 0);
          for (uint32_t _ws_ti = tile.thread_rank(); _ws_ti < _ws_K; _ws_ti += tile.size()) {
            uint32_t _ws_slot = _ws_base_slot + _ws_ti;
            uint32_t _ws_cb = h_subset_2_12.begin() + (_ws_ti + 1) * ws::WCOJ_SPLIT_THRESHOLD;
            uint32_t _ws_ce = _ws_cb + ws::WCOJ_SPLIT_THRESHOLD;
            if (_ws_ce > h_subset_2_12.end())
              _ws_ce = h_subset_2_12.end();
            ws::WCOJTask& task = wcoj_queue->tasks[_ws_slot];
            task.split_level = 2;
            task.num_handles = 4;
            task.handle_begin[0] = _ws_cb;
            task.handle_end[0] = _ws_ce;
            task.handle_depth[0] = h_subset_2_12.depth();
            task.handle_view_slot[0] = 0;
            task.handle_begin[1] = h_subset_base_3_13.begin();
            task.handle_end[1] = h_subset_base_3_13.end();
            task.handle_depth[1] = h_subset_base_3_13.depth();
            task.handle_view_slot[1] = 1;
            task.handle_begin[2] = h_subset_0_root.begin();
            task.handle_end[2] = h_subset_0_root.end();
            task.handle_depth[2] = h_subset_0_root.depth();
            task.handle_view_slot[2] = 0;
            task.handle_begin[3] = h_subset_base_1_root.begin();
            task.handle_end[3] = h_subset_base_1_root.end();
            task.handle_depth[3] = h_subset_base_1_root.depth();
            task.handle_view_slot[3] = 1;
            task.split_handle_idx = 0;
            task.split_depth = h_subset_2_12.depth();
            task.num_bound_vars = 0;
            wcoj_queue->ready_flags[_ws_slot].store(1, cuda::memory_order_release);
          }
          h_subset_2_12 =
              HandleType(h_subset_2_12.begin(), h_subset_2_12.begin() + ws::WCOJ_SPLIT_THRESHOLD,
                         h_subset_2_12.depth());
        }
      }
      auto intersect_14 =
          intersect_handles(tile, h_subset_2_12.iterators(view_subset_1_2_0_DELTA_VER),
                            h_subset_base_3_13.iterators(view_subset_base_0_2_1_FULL_VER));
      uint32_t ws_cumulative_tuples = 0;
      constexpr uint32_t WS_CUM_WORK_THRESHOLD = 4096;
      for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
        auto point = it_15.value();
        auto positions = it_15.positions();
        auto ch_subset_2_point =
            h_subset_2_12.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
        auto ch_subset_base_3_point = h_subset_base_3_13.child_range(
            positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
        // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
        // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix (origin2
        // point)) (subset_base :handle 5 :prefix (origin2 point)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_subset_4_4 = ch_subset_2_point;            // reusing narrowed handle
        auto h_subset_base_5_6 = ch_subset_base_3_point;  // reusing narrowed handle

        if (!h_subset_4_4.valid() || !h_subset_base_5_6.valid())
          continue;

        uint32_t degree_3 = h_subset_4_4.degree();
        uint32_t degree_5 = h_subset_base_5_6.degree();
        uint32_t total_7 = degree_3 * degree_5;
        if (total_7 == 0)
          continue;

        // WS: chunk-donate all Cartesian work above threshold
        if (total_7 > ws::CARTESIAN_SPLIT_THRESHOLD) {
          uint32_t _ws_cart_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
          uint32_t _ws_cart_K = (total_7 - 1) / _ws_cart_chunk;
          if (_ws_cart_K > 0) {
            uint32_t _ws_cart_base = 0;
            if (tile.thread_rank() == 0) {
              if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_cart_K <=
                  wcoj_queue->capacity)
                _ws_cart_base = wcoj_queue->tail.fetch_add(_ws_cart_K, cuda::memory_order_relaxed);
              else
                _ws_cart_K = 0;
            }
            _ws_cart_base = tile.shfl(_ws_cart_base, 0);
            _ws_cart_K = tile.shfl(_ws_cart_K, 0);
            for (uint32_t _ws_ci = tile.thread_rank(); _ws_ci < _ws_cart_K; _ws_ci += tile.size()) {
              uint32_t _ws_cart_slot = _ws_cart_base + _ws_ci;
              ws::WCOJTask& task = wcoj_queue->tasks[_ws_cart_slot];
              task.split_level = 3;
              task.num_handles = 2;
              task.handle_begin[0] = h_subset_4_4.begin();
              task.handle_end[0] = h_subset_4_4.end();
              task.handle_depth[0] = h_subset_4_4.depth();
              task.handle_view_slot[0] = 0;
              task.handle_begin[1] = h_subset_base_5_6.begin();
              task.handle_end[1] = h_subset_base_5_6.end();
              task.handle_depth[1] = h_subset_base_5_6.depth();
              task.handle_view_slot[1] = 1;
              task.split_handle_idx = 0xFF;
              uint32_t _ws_cs = (_ws_ci + 1) * _ws_cart_chunk;
              uint32_t _ws_ce = _ws_cs + _ws_cart_chunk;
              if (_ws_ce > total_7)
                _ws_ce = total_7;
              task.split_begin = _ws_cs;
              task.split_end = _ws_ce;
              task.split_depth = 0;
              task.num_bound_vars = 1;
              task.bound_vars[0] = point;
              wcoj_queue->ready_flags[_ws_cart_slot].store(1, cuda::memory_order_release);
            }
            total_7 = _ws_cart_chunk;
          }
        }

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

          auto origin1 = view_subset_1_2_0_DELTA_VER.get_value(2, h_subset_4_4.begin() + idx0_9);
          auto origin3 =
              view_subset_base_0_2_1_FULL_VER.get_value(2, h_subset_base_5_6.begin() + idx1_10);

          if (origin1 != origin3) {
            // Emit: subset(origin1, origin3, point)
            local_count++;
          }
        }
      }
    }
    if (tile.thread_rank() == 0) {
      wcoj_queue->active_producers.fetch_sub(1, cuda::memory_order_release);
    }

    // Phase 2: Steal from task queue
    while (true) {
      ws::WCOJTask ws_task;
      bool has_work = false;
      if (tile.thread_rank() == 0)
        has_work = wcoj_queue->pop(ws_task);
      has_work = tile.shfl(has_work, 0);
      if (has_work) {
        for (int i = 0; i < ws::WCOJ_MAX_HANDLES; ++i) {
          ws_task.handle_begin[i] = tile.shfl(ws_task.handle_begin[i], 0);
          ws_task.handle_end[i] = tile.shfl(ws_task.handle_end[i], 0);
          ws_task.handle_depth[i] = tile.shfl(ws_task.handle_depth[i], 0);
          ws_task.handle_view_slot[i] = tile.shfl(ws_task.handle_view_slot[i], 0);
        }
        ws_task.split_begin = tile.shfl(ws_task.split_begin, 0);
        ws_task.split_end = tile.shfl(ws_task.split_end, 0);
        ws_task.split_depth = tile.shfl(ws_task.split_depth, 0);
        ws_task.split_level = tile.shfl(ws_task.split_level, 0);
        ws_task.split_handle_idx = tile.shfl(ws_task.split_handle_idx, 0);
        for (int i = 0; i < ws::WCOJ_MAX_BOUND_VARS; ++i)
          ws_task.bound_vars[i] = tile.shfl(ws_task.bound_vars[i], 0);

        if (ws_task.split_handle_idx == 0) {
          {
            auto h_subset_2_2 =
                HandleType(ws_task.handle_begin[0], ws_task.handle_end[0], ws_task.handle_depth[0]);
            auto h_subset_base_3_2 =
                HandleType(ws_task.handle_begin[1], ws_task.handle_end[1], ws_task.handle_depth[1]);
            auto h_subset_0_root =
                HandleType(ws_task.handle_begin[2], ws_task.handle_end[2], ws_task.handle_depth[2]);
            auto h_subset_base_1_root =
                HandleType(ws_task.handle_begin[3], ws_task.handle_end[3], ws_task.handle_depth[3]);
            auto view_subset_1_2_0_DELTA_VER = views[ws_task.handle_view_slot[0]];
            auto view_subset_base_0_2_1_FULL_VER = views[ws_task.handle_view_slot[1]];
            auto origin2 = view_subset_1_2_0_DELTA_VER.get_value(0, h_subset_2_2.begin());
            // Nested ColumnJoin (intersection): bind 'point' from 2 sources
            // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2))
            // (subset_base :handle 3 :prefix (origin2)) ))
            auto h_subset_2_29 = h_subset_2_2;
            auto h_subset_base_3_30 = h_subset_base_3_2;
            // WS: bulk-donate CJ chunks to task queue
            if (h_subset_2_29.degree() > ws::WCOJ_SPLIT_THRESHOLD) {
              uint32_t _ws_total_deg = h_subset_2_29.degree();
              uint32_t _ws_K = (_ws_total_deg - 1) / ws::WCOJ_SPLIT_THRESHOLD;
              if (_ws_K > 0) {
                uint32_t _ws_base_slot = 0;
                if (tile.thread_rank() == 0) {
                  if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_K <=
                      wcoj_queue->capacity)
                    _ws_base_slot = wcoj_queue->tail.fetch_add(_ws_K, cuda::memory_order_relaxed);
                  else
                    _ws_K = 0;
                }
                _ws_base_slot = tile.shfl(_ws_base_slot, 0);
                _ws_K = tile.shfl(_ws_K, 0);
                for (uint32_t _ws_ti = tile.thread_rank(); _ws_ti < _ws_K; _ws_ti += tile.size()) {
                  uint32_t _ws_slot = _ws_base_slot + _ws_ti;
                  uint32_t _ws_cb = h_subset_2_29.begin() + (_ws_ti + 1) * ws::WCOJ_SPLIT_THRESHOLD;
                  uint32_t _ws_ce = _ws_cb + ws::WCOJ_SPLIT_THRESHOLD;
                  if (_ws_ce > h_subset_2_29.end())
                    _ws_ce = h_subset_2_29.end();
                  ws::WCOJTask& task = wcoj_queue->tasks[_ws_slot];
                  task.split_level = 2;
                  task.num_handles = 4;
                  task.handle_begin[0] = _ws_cb;
                  task.handle_end[0] = _ws_ce;
                  task.handle_depth[0] = h_subset_2_29.depth();
                  task.handle_view_slot[0] = 0;
                  task.handle_begin[1] = h_subset_base_3_30.begin();
                  task.handle_end[1] = h_subset_base_3_30.end();
                  task.handle_depth[1] = h_subset_base_3_30.depth();
                  task.handle_view_slot[1] = 1;
                  task.handle_begin[2] = h_subset_0_root.begin();
                  task.handle_end[2] = h_subset_0_root.end();
                  task.handle_depth[2] = h_subset_0_root.depth();
                  task.handle_view_slot[2] = ws_task.handle_view_slot[2];
                  task.handle_begin[3] = h_subset_base_1_root.begin();
                  task.handle_end[3] = h_subset_base_1_root.end();
                  task.handle_depth[3] = h_subset_base_1_root.depth();
                  task.handle_view_slot[3] = ws_task.handle_view_slot[3];
                  task.split_handle_idx = 0;
                  task.split_depth = h_subset_2_29.depth();
                  task.num_bound_vars = 0;
                  wcoj_queue->ready_flags[_ws_slot].store(1, cuda::memory_order_release);
                }
                h_subset_2_29 = HandleType(h_subset_2_29.begin(),
                                           h_subset_2_29.begin() + ws::WCOJ_SPLIT_THRESHOLD,
                                           h_subset_2_29.depth());
              }
            }
            auto intersect_31 =
                intersect_handles(tile, h_subset_2_29.iterators(view_subset_1_2_0_DELTA_VER),
                                  h_subset_base_3_30.iterators(view_subset_base_0_2_1_FULL_VER));
            uint32_t ws_cumulative_tuples = 0;
            constexpr uint32_t WS_CUM_WORK_THRESHOLD = 4096;
            for (auto it_32 = intersect_31.begin(); it_32.valid(); it_32.next()) {
              auto point = it_32.value();
              auto positions = it_32.positions();
              auto ch_subset_2_point =
                  h_subset_2_29.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
              auto ch_subset_base_3_point = h_subset_base_3_30.child_range(
                  positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
              // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
              // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix
              // (origin2 point)) (subset_base :handle 5 :prefix (origin2 point)) ))
              uint32_t lane_18 = tile.thread_rank();
              uint32_t group_size_19 = tile.size();

              auto h_subset_4_21 = ch_subset_2_point;            // reusing narrowed handle
              auto h_subset_base_5_23 = ch_subset_base_3_point;  // reusing narrowed handle

              if (!h_subset_4_21.valid() || !h_subset_base_5_23.valid())
                continue;

              uint32_t degree_20 = h_subset_4_21.degree();
              uint32_t degree_22 = h_subset_base_5_23.degree();
              uint32_t total_24 = degree_20 * degree_22;
              if (total_24 == 0)
                continue;

              // WS: chunk-donate all Cartesian work above threshold
              if (total_24 > ws::CARTESIAN_SPLIT_THRESHOLD) {
                uint32_t _ws_cart_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
                uint32_t _ws_cart_K = (total_24 - 1) / _ws_cart_chunk;
                if (_ws_cart_K > 0) {
                  uint32_t _ws_cart_base = 0;
                  if (tile.thread_rank() == 0) {
                    if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_cart_K <=
                        wcoj_queue->capacity)
                      _ws_cart_base =
                          wcoj_queue->tail.fetch_add(_ws_cart_K, cuda::memory_order_relaxed);
                    else
                      _ws_cart_K = 0;
                  }
                  _ws_cart_base = tile.shfl(_ws_cart_base, 0);
                  _ws_cart_K = tile.shfl(_ws_cart_K, 0);
                  for (uint32_t _ws_ci = tile.thread_rank(); _ws_ci < _ws_cart_K;
                       _ws_ci += tile.size()) {
                    uint32_t _ws_cart_slot = _ws_cart_base + _ws_ci;
                    ws::WCOJTask& task = wcoj_queue->tasks[_ws_cart_slot];
                    task.split_level = 3;
                    task.num_handles = 2;
                    task.handle_begin[0] = h_subset_4_21.begin();
                    task.handle_end[0] = h_subset_4_21.end();
                    task.handle_depth[0] = h_subset_4_21.depth();
                    task.handle_view_slot[0] = 0;
                    task.handle_begin[1] = h_subset_base_5_23.begin();
                    task.handle_end[1] = h_subset_base_5_23.end();
                    task.handle_depth[1] = h_subset_base_5_23.depth();
                    task.handle_view_slot[1] = 1;
                    task.split_handle_idx = 0xFF;
                    uint32_t _ws_cs = (_ws_ci + 1) * _ws_cart_chunk;
                    uint32_t _ws_ce = _ws_cs + _ws_cart_chunk;
                    if (_ws_ce > total_24)
                      _ws_ce = total_24;
                    task.split_begin = _ws_cs;
                    task.split_end = _ws_ce;
                    task.split_depth = 0;
                    task.num_bound_vars = 1;
                    task.bound_vars[0] = point;
                    wcoj_queue->ready_flags[_ws_cart_slot].store(1, cuda::memory_order_release);
                  }
                  total_24 = _ws_cart_chunk;
                }
              }

              for (uint32_t flat_idx_25 = lane_18; flat_idx_25 < total_24;
                   flat_idx_25 += group_size_19) {
                const bool major_is_1_28 = (degree_22 >= degree_20);
                uint32_t idx0_26, idx1_27;
                if (major_is_1_28) {
                  idx0_26 = flat_idx_25 / degree_22;
                  idx1_27 = flat_idx_25 % degree_22;
                } else {
                  idx1_27 = flat_idx_25 / degree_20;
                  idx0_26 = flat_idx_25 % degree_20;
                }

                auto origin1 =
                    view_subset_1_2_0_DELTA_VER.get_value(2, h_subset_4_21.begin() + idx0_26);
                auto origin3 = view_subset_base_0_2_1_FULL_VER.get_value(
                    2, h_subset_base_5_23.begin() + idx1_27);

                if (origin1 != origin3) {
                  // Emit: subset(origin1, origin3, point)
                  local_count++;
                }
              }
            }
          }
        } else if (ws_task.split_handle_idx == 0xFF) {
          {
            auto ws_ct_h0 =
                HandleType(ws_task.handle_begin[0], ws_task.handle_end[0], ws_task.handle_depth[0]);
            auto ws_ct_h1 =
                HandleType(ws_task.handle_begin[1], ws_task.handle_end[1], ws_task.handle_depth[1]);
            auto view_subset_1_2_0_DELTA_VER = views[ws_task.handle_view_slot[0]];
            auto view_subset_base_0_2_1_FULL_VER = views[ws_task.handle_view_slot[1]];
            auto point = static_cast<ValueType>(ws_task.bound_vars[0]);
            uint32_t ws_ct_flat_start = ws_task.split_begin;
            uint32_t ws_ct_flat_end = ws_task.split_end;
            uint32_t ws_ct_deg0 = ws_ct_h0.degree();
            uint32_t ws_ct_deg1 = ws_ct_h1.degree();

            if ((ws_ct_flat_end - ws_ct_flat_start) > ws::CARTESIAN_SPLIT_THRESHOLD) {
              uint32_t ws_ct_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
              if (tile.thread_rank() == 0) {
                for (uint32_t ws_ct_s = ws_ct_flat_start + ws_ct_chunk; ws_ct_s < ws_ct_flat_end;
                     ws_ct_s += ws_ct_chunk) {
                  ws::WCOJTask subtask;
                  subtask.split_level = 3;
                  subtask.num_handles = 2;
                  subtask.handle_begin[0] = ws_ct_h0.begin();
                  subtask.handle_end[0] = ws_ct_h0.end();
                  subtask.handle_depth[0] = ws_ct_h0.depth();
                  subtask.handle_view_slot[0] = ws_task.handle_view_slot[0];
                  subtask.handle_begin[1] = ws_ct_h1.begin();
                  subtask.handle_end[1] = ws_ct_h1.end();
                  subtask.handle_depth[1] = ws_ct_h1.depth();
                  subtask.handle_view_slot[1] = ws_task.handle_view_slot[1];
                  subtask.split_handle_idx = 0xFF;
                  subtask.split_begin = ws_ct_s;
                  uint32_t ws_ct_e = ws_ct_s + ws_ct_chunk;
                  if (ws_ct_e > ws_ct_flat_end)
                    ws_ct_e = ws_ct_flat_end;
                  subtask.split_end = ws_ct_e;
                  subtask.split_depth = 0;
                  subtask.num_bound_vars = ws_task.num_bound_vars;
                  subtask.bound_vars[0] = ws_task.bound_vars[0];
                  wcoj_queue->push(subtask);
                }
              }
              ws_ct_flat_end = ws_ct_flat_start + ws_ct_chunk;
            }

            uint32_t ws_ct_lane = tile.thread_rank();
            uint32_t ws_ct_gs = tile.size();
            for (uint32_t ws_ct_flat = ws_ct_flat_start + ws_ct_lane; ws_ct_flat < ws_ct_flat_end;
                 ws_ct_flat += ws_ct_gs) {
              const bool ws_ct_major = (ws_ct_deg1 >= ws_ct_deg0);
              uint32_t ws_ct_idx0, ws_ct_idx1;
              if (ws_ct_major) {
                ws_ct_idx0 = ws_ct_flat / ws_ct_deg1;
                ws_ct_idx1 = ws_ct_flat % ws_ct_deg1;
              } else {
                ws_ct_idx1 = ws_ct_flat / ws_ct_deg0;
                ws_ct_idx0 = ws_ct_flat % ws_ct_deg0;
              }
              auto origin1 =
                  view_subset_1_2_0_DELTA_VER.get_value(2, ws_ct_h0.begin() + ws_ct_idx0);
              auto origin3 =
                  view_subset_base_0_2_1_FULL_VER.get_value(2, ws_ct_h1.begin() + ws_ct_idx1);
              if (origin1 != origin3) {
                local_count++;
              }
            }
          }
        }
      } else {
        __threadfence();
        uint32_t active = wcoj_queue->active_producers.load(cuda::memory_order_acquire);
        if (active == 0 && wcoj_queue->size() == 0)
          break;
      }
    }

    for (int _ws_offset = kGroupSize / 2; _ws_offset > 0; _ws_offset /= 2) {
      local_count += tile.shfl_down(local_count, _ws_offset);
    }
    if (local_count > 0 && tile.thread_rank() == 0) {
      atomicAdd(global_count, local_count);
    }
  }

  // WS Materialize Kernel (inline pipeline + steal loop)
  static __global__ void __launch_bounds__(kBlockSize) kernel_materialize_ws(
      const ViewType* __restrict__ views, const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys, uint32_t num_root_keys, ValueType* __restrict__ output_data_0,
      semiring_value_t<SR>* __restrict__ output_prov_0, std::size_t output_stride_0,
      uint32_t old_size_0, uint32_t* __restrict__ global_write_pos,
      SRDatalog::GPU::JIT::WS::WCOJTaskQueue* __restrict__ wcoj_queue,
      SRDatalog::GPU::JIT::WS::WCOJRangeBoard* __restrict__ wcoj_range_board) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;

    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;
    namespace ws = SRDatalog::GPU::JIT::WS;

    if (tile.thread_rank() == 0) {
      wcoj_queue->active_producers.fetch_add(1, cuda::memory_order_relaxed);
    }

    SRDatalog::GPU::JIT::WS::ChunkedOutputContext<ValueType, 3, 16> output_ctx_0{
        output_data_0, global_write_pos, static_cast<uint32_t>(output_stride_0), old_size_0};

    // Phase 1: Main traversal with inline WS splitting
    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_subset_1_2_0_DELTA_VER = views[0];
    auto view_subset_base_0_2_1_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'origin2' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var origin2 :sources ((subset :handle 0) (subset_base :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 =
          view_subset_1_2_0_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_subset_1_2_0_DELTA_VER.num_rows_)
                      ? hint_hi_4
                      : view_subset_1_2_0_DELTA_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_subset_1_2_0_DELTA_VER.num_rows_;
      auto h_subset_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_subset_1_2_0_DELTA_VER);
      if (!h_subset_0_root.valid())
        continue;
      auto h_subset_base_1_root = HandleType(0, view_subset_base_0_2_1_FULL_VER.num_rows_, 0)
                                      .prefix(root_val_2, tile, view_subset_base_0_2_1_FULL_VER);
      if (!h_subset_base_1_root.valid())
        continue;
      auto origin2 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'point' from 2 sources
      // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2)) (subset_base
      // :handle 3 :prefix (origin2)) ))
      auto h_subset_2_14 = h_subset_0_root;
      auto h_subset_base_3_15 = h_subset_base_1_root;
      // WS: bulk-donate CJ chunks to task queue
      if (h_subset_2_14.degree() > ws::WCOJ_SPLIT_THRESHOLD) {
        uint32_t _ws_total_deg = h_subset_2_14.degree();
        uint32_t _ws_K = (_ws_total_deg - 1) / ws::WCOJ_SPLIT_THRESHOLD;
        if (_ws_K > 0) {
          uint32_t _ws_base_slot = 0;
          if (tile.thread_rank() == 0) {
            if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_K <= wcoj_queue->capacity)
              _ws_base_slot = wcoj_queue->tail.fetch_add(_ws_K, cuda::memory_order_relaxed);
            else
              _ws_K = 0;
          }
          _ws_base_slot = tile.shfl(_ws_base_slot, 0);
          _ws_K = tile.shfl(_ws_K, 0);
          for (uint32_t _ws_ti = tile.thread_rank(); _ws_ti < _ws_K; _ws_ti += tile.size()) {
            uint32_t _ws_slot = _ws_base_slot + _ws_ti;
            uint32_t _ws_cb = h_subset_2_14.begin() + (_ws_ti + 1) * ws::WCOJ_SPLIT_THRESHOLD;
            uint32_t _ws_ce = _ws_cb + ws::WCOJ_SPLIT_THRESHOLD;
            if (_ws_ce > h_subset_2_14.end())
              _ws_ce = h_subset_2_14.end();
            ws::WCOJTask& task = wcoj_queue->tasks[_ws_slot];
            task.split_level = 2;
            task.num_handles = 4;
            task.handle_begin[0] = _ws_cb;
            task.handle_end[0] = _ws_ce;
            task.handle_depth[0] = h_subset_2_14.depth();
            task.handle_view_slot[0] = 0;
            task.handle_begin[1] = h_subset_base_3_15.begin();
            task.handle_end[1] = h_subset_base_3_15.end();
            task.handle_depth[1] = h_subset_base_3_15.depth();
            task.handle_view_slot[1] = 1;
            task.handle_begin[2] = h_subset_0_root.begin();
            task.handle_end[2] = h_subset_0_root.end();
            task.handle_depth[2] = h_subset_0_root.depth();
            task.handle_view_slot[2] = 0;
            task.handle_begin[3] = h_subset_base_1_root.begin();
            task.handle_end[3] = h_subset_base_1_root.end();
            task.handle_depth[3] = h_subset_base_1_root.depth();
            task.handle_view_slot[3] = 1;
            task.split_handle_idx = 0;
            task.split_depth = h_subset_2_14.depth();
            task.num_bound_vars = 0;
            wcoj_queue->ready_flags[_ws_slot].store(1, cuda::memory_order_release);
          }
          h_subset_2_14 =
              HandleType(h_subset_2_14.begin(), h_subset_2_14.begin() + ws::WCOJ_SPLIT_THRESHOLD,
                         h_subset_2_14.depth());
        }
      }
      auto intersect_16 =
          intersect_handles(tile, h_subset_2_14.iterators(view_subset_1_2_0_DELTA_VER),
                            h_subset_base_3_15.iterators(view_subset_base_0_2_1_FULL_VER));
      uint32_t ws_cumulative_tuples = 0;
      constexpr uint32_t WS_CUM_WORK_THRESHOLD = 4096;
      for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
        auto point = it_17.value();
        auto positions = it_17.positions();
        auto ch_subset_2_point =
            h_subset_2_14.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
        auto ch_subset_base_3_point = h_subset_base_3_15.child_range(
            positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
        // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
        // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix (origin2
        // point)) (subset_base :handle 5 :prefix (origin2 point)) ))
        uint32_t lane_2 = tile.thread_rank();
        uint32_t group_size_3 = tile.size();

        auto h_subset_4_5 = ch_subset_2_point;            // reusing narrowed handle
        auto h_subset_base_5_7 = ch_subset_base_3_point;  // reusing narrowed handle

        if (!h_subset_4_5.valid() || !h_subset_base_5_7.valid())
          continue;

        uint32_t degree_4 = h_subset_4_5.degree();
        uint32_t degree_6 = h_subset_base_5_7.degree();
        uint32_t total_8 = degree_4 * degree_6;
        if (total_8 == 0)
          continue;

        // WS: chunk-donate all Cartesian work above threshold
        if (total_8 > ws::CARTESIAN_SPLIT_THRESHOLD) {
          uint32_t _ws_cart_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
          uint32_t _ws_cart_K = (total_8 - 1) / _ws_cart_chunk;
          if (_ws_cart_K > 0) {
            uint32_t _ws_cart_base = 0;
            if (tile.thread_rank() == 0) {
              if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_cart_K <=
                  wcoj_queue->capacity)
                _ws_cart_base = wcoj_queue->tail.fetch_add(_ws_cart_K, cuda::memory_order_relaxed);
              else
                _ws_cart_K = 0;
            }
            _ws_cart_base = tile.shfl(_ws_cart_base, 0);
            _ws_cart_K = tile.shfl(_ws_cart_K, 0);
            for (uint32_t _ws_ci = tile.thread_rank(); _ws_ci < _ws_cart_K; _ws_ci += tile.size()) {
              uint32_t _ws_cart_slot = _ws_cart_base + _ws_ci;
              ws::WCOJTask& task = wcoj_queue->tasks[_ws_cart_slot];
              task.split_level = 3;
              task.num_handles = 2;
              task.handle_begin[0] = h_subset_4_5.begin();
              task.handle_end[0] = h_subset_4_5.end();
              task.handle_depth[0] = h_subset_4_5.depth();
              task.handle_view_slot[0] = 0;
              task.handle_begin[1] = h_subset_base_5_7.begin();
              task.handle_end[1] = h_subset_base_5_7.end();
              task.handle_depth[1] = h_subset_base_5_7.depth();
              task.handle_view_slot[1] = 1;
              task.split_handle_idx = 0xFF;
              uint32_t _ws_cs = (_ws_ci + 1) * _ws_cart_chunk;
              uint32_t _ws_ce = _ws_cs + _ws_cart_chunk;
              if (_ws_ce > total_8)
                _ws_ce = total_8;
              task.split_begin = _ws_cs;
              task.split_end = _ws_ce;
              task.split_depth = 0;
              task.num_bound_vars = 1;
              task.bound_vars[0] = point;
              wcoj_queue->ready_flags[_ws_cart_slot].store(1, cuda::memory_order_release);
            }
            total_8 = _ws_cart_chunk;
          }
        }

        for (uint32_t ws_batch_10 = 0; ws_batch_10 < total_8; ws_batch_10 += group_size_3) {
          uint32_t flat_idx_9 = ws_batch_10 + lane_2;
          bool ws_valid_1 = flat_idx_9 < total_8;
          const bool major_is_1_13 = (degree_6 >= degree_4);
          uint32_t idx0_11, idx1_12;
          if (major_is_1_13) {
            idx0_11 = flat_idx_9 / degree_6;
            idx1_12 = flat_idx_9 % degree_6;
          } else {
            idx1_12 = flat_idx_9 / degree_4;
            idx0_11 = flat_idx_9 % degree_4;
          }

          auto origin1 =
              ws_valid_1 ? view_subset_1_2_0_DELTA_VER.get_value(2, h_subset_4_5.begin() + idx0_11)
                         : ValueType{0};
          auto origin3 = ws_valid_1 ? view_subset_base_0_2_1_FULL_VER.get_value(
                                          2, h_subset_base_5_7.begin() + idx1_12)
                                    : ValueType{0};
          ws_valid_1 = ws_valid_1 && (origin1 != origin3);
          // Emit: subset(origin1, origin3, point)
          output_ctx_0.emit_warp_coalesced(tile, ws_valid_1, origin1, origin3, point);
        }
      }
    }
    if (tile.thread_rank() == 0) {
      wcoj_queue->active_producers.fetch_sub(1, cuda::memory_order_release);
    }

    // Phase 2: Steal from task queue
    while (true) {
      ws::WCOJTask ws_task;
      bool has_work = false;
      if (tile.thread_rank() == 0)
        has_work = wcoj_queue->pop(ws_task);
      has_work = tile.shfl(has_work, 0);
      if (has_work) {
        for (int i = 0; i < ws::WCOJ_MAX_HANDLES; ++i) {
          ws_task.handle_begin[i] = tile.shfl(ws_task.handle_begin[i], 0);
          ws_task.handle_end[i] = tile.shfl(ws_task.handle_end[i], 0);
          ws_task.handle_depth[i] = tile.shfl(ws_task.handle_depth[i], 0);
          ws_task.handle_view_slot[i] = tile.shfl(ws_task.handle_view_slot[i], 0);
        }
        ws_task.split_begin = tile.shfl(ws_task.split_begin, 0);
        ws_task.split_end = tile.shfl(ws_task.split_end, 0);
        ws_task.split_depth = tile.shfl(ws_task.split_depth, 0);
        ws_task.split_level = tile.shfl(ws_task.split_level, 0);
        ws_task.split_handle_idx = tile.shfl(ws_task.split_handle_idx, 0);
        for (int i = 0; i < ws::WCOJ_MAX_BOUND_VARS; ++i)
          ws_task.bound_vars[i] = tile.shfl(ws_task.bound_vars[i], 0);

        if (ws_task.split_handle_idx == 0) {
          {
            auto h_subset_2_2 =
                HandleType(ws_task.handle_begin[0], ws_task.handle_end[0], ws_task.handle_depth[0]);
            auto h_subset_base_3_2 =
                HandleType(ws_task.handle_begin[1], ws_task.handle_end[1], ws_task.handle_depth[1]);
            auto h_subset_0_root =
                HandleType(ws_task.handle_begin[2], ws_task.handle_end[2], ws_task.handle_depth[2]);
            auto h_subset_base_1_root =
                HandleType(ws_task.handle_begin[3], ws_task.handle_end[3], ws_task.handle_depth[3]);
            auto view_subset_1_2_0_DELTA_VER = views[ws_task.handle_view_slot[0]];
            auto view_subset_base_0_2_1_FULL_VER = views[ws_task.handle_view_slot[1]];
            auto origin2 = view_subset_1_2_0_DELTA_VER.get_value(0, h_subset_2_2.begin());
            // Nested ColumnJoin (intersection): bind 'point' from 2 sources
            // MIR: (column-join :var point :sources ((subset :handle 2 :prefix (origin2))
            // (subset_base :handle 3 :prefix (origin2)) ))
            auto h_subset_2_14 = h_subset_2_2;
            auto h_subset_base_3_15 = h_subset_base_3_2;
            // WS: bulk-donate CJ chunks to task queue
            if (h_subset_2_14.degree() > ws::WCOJ_SPLIT_THRESHOLD) {
              uint32_t _ws_total_deg = h_subset_2_14.degree();
              uint32_t _ws_K = (_ws_total_deg - 1) / ws::WCOJ_SPLIT_THRESHOLD;
              if (_ws_K > 0) {
                uint32_t _ws_base_slot = 0;
                if (tile.thread_rank() == 0) {
                  if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_K <=
                      wcoj_queue->capacity)
                    _ws_base_slot = wcoj_queue->tail.fetch_add(_ws_K, cuda::memory_order_relaxed);
                  else
                    _ws_K = 0;
                }
                _ws_base_slot = tile.shfl(_ws_base_slot, 0);
                _ws_K = tile.shfl(_ws_K, 0);
                for (uint32_t _ws_ti = tile.thread_rank(); _ws_ti < _ws_K; _ws_ti += tile.size()) {
                  uint32_t _ws_slot = _ws_base_slot + _ws_ti;
                  uint32_t _ws_cb = h_subset_2_14.begin() + (_ws_ti + 1) * ws::WCOJ_SPLIT_THRESHOLD;
                  uint32_t _ws_ce = _ws_cb + ws::WCOJ_SPLIT_THRESHOLD;
                  if (_ws_ce > h_subset_2_14.end())
                    _ws_ce = h_subset_2_14.end();
                  ws::WCOJTask& task = wcoj_queue->tasks[_ws_slot];
                  task.split_level = 2;
                  task.num_handles = 4;
                  task.handle_begin[0] = _ws_cb;
                  task.handle_end[0] = _ws_ce;
                  task.handle_depth[0] = h_subset_2_14.depth();
                  task.handle_view_slot[0] = 0;
                  task.handle_begin[1] = h_subset_base_3_15.begin();
                  task.handle_end[1] = h_subset_base_3_15.end();
                  task.handle_depth[1] = h_subset_base_3_15.depth();
                  task.handle_view_slot[1] = 1;
                  task.handle_begin[2] = h_subset_0_root.begin();
                  task.handle_end[2] = h_subset_0_root.end();
                  task.handle_depth[2] = h_subset_0_root.depth();
                  task.handle_view_slot[2] = ws_task.handle_view_slot[2];
                  task.handle_begin[3] = h_subset_base_1_root.begin();
                  task.handle_end[3] = h_subset_base_1_root.end();
                  task.handle_depth[3] = h_subset_base_1_root.depth();
                  task.handle_view_slot[3] = ws_task.handle_view_slot[3];
                  task.split_handle_idx = 0;
                  task.split_depth = h_subset_2_14.depth();
                  task.num_bound_vars = 0;
                  wcoj_queue->ready_flags[_ws_slot].store(1, cuda::memory_order_release);
                }
                h_subset_2_14 = HandleType(h_subset_2_14.begin(),
                                           h_subset_2_14.begin() + ws::WCOJ_SPLIT_THRESHOLD,
                                           h_subset_2_14.depth());
              }
            }
            auto intersect_16 =
                intersect_handles(tile, h_subset_2_14.iterators(view_subset_1_2_0_DELTA_VER),
                                  h_subset_base_3_15.iterators(view_subset_base_0_2_1_FULL_VER));
            uint32_t ws_cumulative_tuples = 0;
            constexpr uint32_t WS_CUM_WORK_THRESHOLD = 4096;
            for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
              auto point = it_17.value();
              auto positions = it_17.positions();
              auto ch_subset_2_point =
                  h_subset_2_14.child_range(positions[0], point, tile, view_subset_1_2_0_DELTA_VER);
              auto ch_subset_base_3_point = h_subset_base_3_15.child_range(
                  positions[1], point, tile, view_subset_base_0_2_1_FULL_VER);
              // Nested CartesianJoin: bind origin1, origin3 from 2 source(s)
              // MIR: (cartesian-join :vars (origin1 origin3) :sources ((subset :handle 4 :prefix
              // (origin2 point)) (subset_base :handle 5 :prefix (origin2 point)) ))
              uint32_t lane_2 = tile.thread_rank();
              uint32_t group_size_3 = tile.size();

              auto h_subset_4_5 = ch_subset_2_point;            // reusing narrowed handle
              auto h_subset_base_5_7 = ch_subset_base_3_point;  // reusing narrowed handle

              if (!h_subset_4_5.valid() || !h_subset_base_5_7.valid())
                continue;

              uint32_t degree_4 = h_subset_4_5.degree();
              uint32_t degree_6 = h_subset_base_5_7.degree();
              uint32_t total_8 = degree_4 * degree_6;
              if (total_8 == 0)
                continue;

              // WS: chunk-donate all Cartesian work above threshold
              if (total_8 > ws::CARTESIAN_SPLIT_THRESHOLD) {
                uint32_t _ws_cart_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
                uint32_t _ws_cart_K = (total_8 - 1) / _ws_cart_chunk;
                if (_ws_cart_K > 0) {
                  uint32_t _ws_cart_base = 0;
                  if (tile.thread_rank() == 0) {
                    if (wcoj_queue->tail.load(cuda::memory_order_relaxed) + _ws_cart_K <=
                        wcoj_queue->capacity)
                      _ws_cart_base =
                          wcoj_queue->tail.fetch_add(_ws_cart_K, cuda::memory_order_relaxed);
                    else
                      _ws_cart_K = 0;
                  }
                  _ws_cart_base = tile.shfl(_ws_cart_base, 0);
                  _ws_cart_K = tile.shfl(_ws_cart_K, 0);
                  for (uint32_t _ws_ci = tile.thread_rank(); _ws_ci < _ws_cart_K;
                       _ws_ci += tile.size()) {
                    uint32_t _ws_cart_slot = _ws_cart_base + _ws_ci;
                    ws::WCOJTask& task = wcoj_queue->tasks[_ws_cart_slot];
                    task.split_level = 3;
                    task.num_handles = 2;
                    task.handle_begin[0] = h_subset_4_5.begin();
                    task.handle_end[0] = h_subset_4_5.end();
                    task.handle_depth[0] = h_subset_4_5.depth();
                    task.handle_view_slot[0] = 0;
                    task.handle_begin[1] = h_subset_base_5_7.begin();
                    task.handle_end[1] = h_subset_base_5_7.end();
                    task.handle_depth[1] = h_subset_base_5_7.depth();
                    task.handle_view_slot[1] = 1;
                    task.split_handle_idx = 0xFF;
                    uint32_t _ws_cs = (_ws_ci + 1) * _ws_cart_chunk;
                    uint32_t _ws_ce = _ws_cs + _ws_cart_chunk;
                    if (_ws_ce > total_8)
                      _ws_ce = total_8;
                    task.split_begin = _ws_cs;
                    task.split_end = _ws_ce;
                    task.split_depth = 0;
                    task.num_bound_vars = 1;
                    task.bound_vars[0] = point;
                    wcoj_queue->ready_flags[_ws_cart_slot].store(1, cuda::memory_order_release);
                  }
                  total_8 = _ws_cart_chunk;
                }
              }

              for (uint32_t ws_batch_10 = 0; ws_batch_10 < total_8; ws_batch_10 += group_size_3) {
                uint32_t flat_idx_9 = ws_batch_10 + lane_2;
                bool ws_valid_1 = flat_idx_9 < total_8;
                const bool major_is_1_13 = (degree_6 >= degree_4);
                uint32_t idx0_11, idx1_12;
                if (major_is_1_13) {
                  idx0_11 = flat_idx_9 / degree_6;
                  idx1_12 = flat_idx_9 % degree_6;
                } else {
                  idx1_12 = flat_idx_9 / degree_4;
                  idx0_11 = flat_idx_9 % degree_4;
                }

                auto origin1 = ws_valid_1 ? view_subset_1_2_0_DELTA_VER.get_value(
                                                2, h_subset_4_5.begin() + idx0_11)
                                          : ValueType{0};
                auto origin3 = ws_valid_1 ? view_subset_base_0_2_1_FULL_VER.get_value(
                                                2, h_subset_base_5_7.begin() + idx1_12)
                                          : ValueType{0};
                ws_valid_1 = ws_valid_1 && (origin1 != origin3);
                // Emit: subset(origin1, origin3, point)
                output_ctx_0.emit_warp_coalesced(tile, ws_valid_1, origin1, origin3, point);
              }
            }
          }
        } else if (ws_task.split_handle_idx == 0xFF) {
          {
            auto ws_ct_h0 =
                HandleType(ws_task.handle_begin[0], ws_task.handle_end[0], ws_task.handle_depth[0]);
            auto ws_ct_h1 =
                HandleType(ws_task.handle_begin[1], ws_task.handle_end[1], ws_task.handle_depth[1]);
            auto view_subset_1_2_0_DELTA_VER = views[ws_task.handle_view_slot[0]];
            auto view_subset_base_0_2_1_FULL_VER = views[ws_task.handle_view_slot[1]];
            auto point = static_cast<ValueType>(ws_task.bound_vars[0]);
            uint32_t ws_ct_flat_start = ws_task.split_begin;
            uint32_t ws_ct_flat_end = ws_task.split_end;
            uint32_t ws_ct_deg0 = ws_ct_h0.degree();
            uint32_t ws_ct_deg1 = ws_ct_h1.degree();

            // Chunk-donate stolen Cartesian range if above threshold
            if ((ws_ct_flat_end - ws_ct_flat_start) > ws::CARTESIAN_SPLIT_THRESHOLD) {
              uint32_t ws_ct_chunk = ws::CARTESIAN_SPLIT_THRESHOLD;
              if (tile.thread_rank() == 0) {
                for (uint32_t ws_ct_s = ws_ct_flat_start + ws_ct_chunk; ws_ct_s < ws_ct_flat_end;
                     ws_ct_s += ws_ct_chunk) {
                  ws::WCOJTask subtask;
                  subtask.split_level = 3;
                  subtask.num_handles = 2;
                  subtask.handle_begin[0] = ws_ct_h0.begin();
                  subtask.handle_end[0] = ws_ct_h0.end();
                  subtask.handle_depth[0] = ws_ct_h0.depth();
                  subtask.handle_view_slot[0] = ws_task.handle_view_slot[0];
                  subtask.handle_begin[1] = ws_ct_h1.begin();
                  subtask.handle_end[1] = ws_ct_h1.end();
                  subtask.handle_depth[1] = ws_ct_h1.depth();
                  subtask.handle_view_slot[1] = ws_task.handle_view_slot[1];
                  subtask.split_handle_idx = 0xFF;
                  subtask.split_begin = ws_ct_s;
                  uint32_t ws_ct_e = ws_ct_s + ws_ct_chunk;
                  if (ws_ct_e > ws_ct_flat_end)
                    ws_ct_e = ws_ct_flat_end;
                  subtask.split_end = ws_ct_e;
                  subtask.split_depth = 0;
                  subtask.num_bound_vars = ws_task.num_bound_vars;
                  subtask.bound_vars[0] = ws_task.bound_vars[0];
                  wcoj_queue->push(subtask);
                }
              }
              ws_ct_flat_end = ws_ct_flat_start + ws_ct_chunk;
            }

            uint32_t ws_ct_lane = tile.thread_rank();
            uint32_t ws_ct_gs = tile.size();
            for (uint32_t ws_ct_flat = ws_ct_flat_start + ws_ct_lane; ws_ct_flat < ws_ct_flat_end;
                 ws_ct_flat += ws_ct_gs) {
              const bool ws_ct_major = (ws_ct_deg1 >= ws_ct_deg0);
              uint32_t ws_ct_idx0, ws_ct_idx1;
              if (ws_ct_major) {
                ws_ct_idx0 = ws_ct_flat / ws_ct_deg1;
                ws_ct_idx1 = ws_ct_flat % ws_ct_deg1;
              } else {
                ws_ct_idx1 = ws_ct_flat / ws_ct_deg0;
                ws_ct_idx0 = ws_ct_flat % ws_ct_deg0;
              }
              auto origin1 =
                  view_subset_1_2_0_DELTA_VER.get_value(2, ws_ct_h0.begin() + ws_ct_idx0);
              auto origin3 =
                  view_subset_base_0_2_1_FULL_VER.get_value(2, ws_ct_h1.begin() + ws_ct_idx1);
              if (origin1 != origin3) {
                output_ctx_0.emit_direct(origin1, origin3, point);
              }
            }
          }
        }
      } else {
        __threadfence();
        uint32_t active = wcoj_queue->active_producers.load(cuda::memory_order_acquire);
        if (active == 0 && wcoj_queue->size() == 0)
          break;
      }
    }

    output_ctx_0.flush();
  }

  // State carried between decomposed phases
  struct LaunchParams {
    std::vector<ViewType> views_vec;
    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};
    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};
    uint32_t* thread_counts_ptr = nullptr;
    uint32_t num_unique_root_keys = 0;
    uint32_t num_root_keys = 0;
    const ValueType* root_unique_values_ptr = nullptr;
    const ValueType* head_root_unique_values_ptr = nullptr;
    uint32_t num_full_unique_root_keys = 0;
    uint32_t num_head_unique_root_keys = 0;
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t old_size_0 = 0;
    bool used_ws = false;  // Adaptive: true if WS kernel was used this iteration
  };

  // Phase-decomposed methods for stream-parallel execution
  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count,
                                 GPU_STREAM_T stream = 0);

  // Non-template execute - calls kernels directly
  static void execute(DB& db, uint32_t iteration);
};

// Phase 1: Setup views and compute grid config
JitRunner_subset_trans_D0::LaunchParams JitRunner_subset_trans_D0::setup(DB& db, uint32_t iteration,
                                                                         GPU_STREAM_T stream) {
  LaunchParams p;
  p.views_vec.reserve(NumSources);

  // Source 0: subset version DELTA_VER
  {
    auto& rel_0 = (iteration == 0) ? get_relation_by_schema<subset, FULL_VER>(db)
                                   : get_relation_by_schema<subset, DELTA_VER>(db);
    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{1, 2, 0}}, false);
    p.views_vec.push_back(idx_0.view());
  }

  // Source 1: subset_base version FULL_VER
  {
    auto& rel_1 = get_relation_by_schema<subset_base, FULL_VER>(db);
    auto& idx_1 = rel_1.ensure_index(SRDatalog::IndexSpec{{0, 2, 1}}, false);
    p.views_vec.push_back(idx_1.view());
  }

  // First source for root keys
  auto& first_rel = (iteration == 0) ? get_relation_by_schema<subset, FULL_VER>(db)
                                     : get_relation_by_schema<subset, DELTA_VER>(db);
  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{{1, 2, 0}});
  p.num_root_keys = first_idx.root().degree();
  p.num_unique_root_keys = static_cast<uint32_t>(first_idx.num_unique_root_values());
  p.root_unique_values_ptr =
      (p.num_unique_root_keys > 0) ? first_idx.root_unique_values().data() : nullptr;
  p.num_full_unique_root_keys = p.num_unique_root_keys;

  // Copy views to device using provided stream (NOT stream 0)
  p.d_views = SRDatalog::GPU::DeviceArray<ViewType>(p.views_vec.size());
  GPU_MEMCPY_ASYNC(p.d_views.data(), p.views_vec.data(), p.views_vec.size() * sizeof(ViewType),
                   GPU_HOST_TO_DEVICE, stream);

  int num_sms = 0;
  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);
  // WCOJ: unique-key-based iteration
  p.num_threads = ((p.num_unique_root_keys + kGroupSize - 1) / kGroupSize) * kGroupSize;
  p.num_threads = std::max(p.num_threads, static_cast<uint32_t>(kBlockSize));
  p.num_blocks = (p.num_threads + kBlockSize - 1) / kBlockSize;
  p.num_blocks = std::max(p.num_blocks, static_cast<uint32_t>(num_sms) * 8);
  p.num_threads = p.num_blocks * kBlockSize;

  p.thread_counts = SRDatalog::GPU::DeviceArray<uint32_t>(p.num_threads + 1);
  cudaMemsetAsync(p.thread_counts.data(), 0, (p.num_threads + 1) * sizeof(uint32_t), stream);
  p.thread_counts_ptr = p.thread_counts.data();
  return p;
}

void JitRunner_subset_trans_D0::launch_count(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0)
    return;
  if (p.num_unique_root_keys == 0) {
    cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream);
    return;
  }
  constexpr uint32_t kAdaptiveWSThreshold = 10000;
  if (p.num_root_keys < kAdaptiveWSThreshold) {
    // Small delta: use non-WS kernel (lower overhead)
    p.used_ws = false;
    kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>(
        p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
        p.thread_counts_ptr);
  } else {
    // Large delta: use WS kernel (handles skew)
    p.used_ws = true;
    static uint32_t* s_ws_global_count = nullptr;
    static SRDatalog::GPU::JIT::WS::WCOJTaskQueue* s_ws_count_queue = nullptr;
    static SRDatalog::GPU::JIT::WS::WCOJRangeBoard* s_ws_count_range_board = nullptr;
    if (s_ws_global_count == nullptr) {
      s_ws_global_count =
          static_cast<uint32_t*>(SRDatalog::GPU::JIT::WS::ws_rmm_alloc(sizeof(uint32_t), stream));
    }
    cudaMemsetAsync(s_ws_global_count, 0, sizeof(uint32_t), stream);
    if (s_ws_count_queue == nullptr) {
      s_ws_count_queue = SRDatalog::GPU::JIT::WS::allocate_task_queue(stream);
    } else {
      SRDatalog::GPU::JIT::WS::reset_task_queue(
          s_ws_count_queue, SRDatalog::GPU::JIT::WS::WCOJ_QUEUE_CAPACITY, stream);
    }
    if (s_ws_count_range_board == nullptr) {
      s_ws_count_range_board = SRDatalog::GPU::JIT::WS::allocate_range_board(stream);
    } else {
      SRDatalog::GPU::JIT::WS::reset_range_board(s_ws_count_range_board, stream);
    }
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel_count_ws, kBlockSize,
                                                  0);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int ws_num_blocks = std::min((int)p.num_blocks, max_blocks_per_sm * num_sms);
    kernel_count_ws<<<ws_num_blocks, kBlockSize, 0, stream>>>(
        p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
        s_ws_global_count, s_ws_count_queue, s_ws_count_range_board);
    p.thread_counts_ptr = s_ws_global_count;
  }
}

// Phase 3: Prefix scan + readback total + resize destinations
uint32_t JitRunner_subset_trans_D0::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  uint32_t total_count = 0;
  if (p.used_ws) {
    // WS path: read global atomic counter (no prefix scan needed)
    GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr, sizeof(uint32_t), GPU_DEVICE_TO_HOST,
                     stream);
    GPU_STREAM_SYNCHRONIZE(stream);
  } else {
    // Non-WS path: standard prefix scan
    thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                           p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                           thrust::plus<uint32_t>());
    GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
                     GPU_DEVICE_TO_HOST, stream);
    GPU_STREAM_SYNCHRONIZE(stream);
  }
  if (total_count == 0)
    return 0;

  auto& dest_rel_0 = get_relation_by_schema<subset, NEW_VER>(db);
  p.old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  dest_rel_0.resize_interned_columns(p.old_size_0 + total_count, stream);
  return total_count;
}

// Phase 3a: Prefix scan only (async, no sync)
void JitRunner_subset_trans_D0::scan_only(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0)
    return;
  if (p.used_ws) {
    // WS: no prefix scan needed (global atomic counter)
  } else {
    thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                           p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                           thrust::plus<uint32_t>());
  }
}

// Phase 3b: Read total count (call after device sync)
uint32_t JitRunner_subset_trans_D0::read_total(LaunchParams& p) {
  if (p.num_threads == 0)
    return 0;
  uint32_t total_count = 0;
  if (p.used_ws) {
    // WS: read from global atomic counter
    GPU_MEMCPY(&total_count, p.thread_counts_ptr, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  } else {
    GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
               GPU_DEVICE_TO_HOST);
  }
  return total_count;
}

// Phase 4 (Adaptive WS): Launch WS or non-WS materialize kernel
void JitRunner_subset_trans_D0::launch_materialize(DB& db, LaunchParams& p, uint32_t total_count,
                                                   GPU_STREAM_T stream) {
  if (total_count == 0)
    return;
  using ProvPtrType = semiring_value_t<SR>*;
  ProvPtrType prov_ptr = nullptr;

  auto& dest_rel_0 = get_relation_by_schema<subset, NEW_VER>(db);
  uint32_t old_size_0 = p.old_size_0;

  if (p.used_ws) {
    static uint32_t* s_global_write_pos = nullptr;
    static SRDatalog::GPU::JIT::WS::WCOJTaskQueue* s_wcoj_queue = nullptr;
    static SRDatalog::GPU::JIT::WS::WCOJRangeBoard* s_wcoj_range_board = nullptr;
    if (s_global_write_pos == nullptr) {
      s_global_write_pos =
          static_cast<uint32_t*>(SRDatalog::GPU::JIT::WS::ws_rmm_alloc(sizeof(uint32_t), stream));
    }
    cudaMemsetAsync(s_global_write_pos, 0, sizeof(uint32_t), stream);
    if (s_wcoj_queue == nullptr) {
      s_wcoj_queue = SRDatalog::GPU::JIT::WS::allocate_task_queue(stream);
    } else {
      SRDatalog::GPU::JIT::WS::reset_task_queue(
          s_wcoj_queue, SRDatalog::GPU::JIT::WS::WCOJ_QUEUE_CAPACITY, stream);
    }
    SRDatalog::GPU::JIT::WS::set_diagnostic_queue(s_wcoj_queue);
    if (s_wcoj_range_board == nullptr) {
      s_wcoj_range_board = SRDatalog::GPU::JIT::WS::allocate_range_board(stream);
    } else {
      SRDatalog::GPU::JIT::WS::reset_range_board(s_wcoj_range_board, stream);
    }
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel_materialize_ws,
                                                  kBlockSize, 0);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    uint32_t ws_num_blocks = std::min(
        p.num_blocks, static_cast<uint32_t>(max_blocks_per_sm) * static_cast<uint32_t>(num_sms));
    kernel_materialize_ws<<<ws_num_blocks, kBlockSize, 0, stream>>>(
        p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
        dest_rel_0.template interned_column<0>(), prov_ptr, dest_rel_0.interned_stride(),
        old_size_0, s_global_write_pos, s_wcoj_queue, s_wcoj_range_board);
  } else {
    kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(
        p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
        p.thread_counts_ptr, dest_rel_0.template interned_column<0>(), prov_ptr,
        dest_rel_0.interned_stride(), old_size_0);
  }
}

// Execute definition - calls decomposed phases sequentially
void JitRunner_subset_trans_D0::execute(DB& db, uint32_t iteration) {
  nvtxRangePushA("subset_trans_D0");

  auto p = setup(db, iteration);
  launch_count(p, 0);
  uint32_t total_count = scan_and_resize(db, p, 0);
  if (total_count == 0) {
    nvtxRangePop();
    return;
  }

  launch_materialize(db, p, total_count, 0);
  GPU_STREAM_SYNCHRONIZE(0);
  nvtxRangePop();
}
