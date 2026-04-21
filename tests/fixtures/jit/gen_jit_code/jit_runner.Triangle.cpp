// =============================================================
// JIT-Generated Complete Runner: Triangle
// Fully concrete - NO C++ templates/metaprogramming
// =============================================================

struct JitRunner_Triangle {
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = TriangleRules_DB_DeviceDB;
  using FirstSchema = Edge;
  using DestSchema = Z;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType =
      std::decay_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(std::declval<DB&>()))>;
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
    auto view_Edge_0_1_FULL_VER = views[0];
    auto view_Edge_1_0_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'x' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var x :sources ((Edge :handle 0) (Edge :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Edge_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Edge_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Edge_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Edge_0_1_FULL_VER.num_rows_;
      auto h_Edge_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Edge_0_1_FULL_VER);
      if (!h_Edge_0_root.valid())
        continue;
      auto h_Edge_1_root = HandleType(0, view_Edge_1_0_FULL_VER.num_rows_, 0)
                               .prefix(root_val_2, tile, view_Edge_1_0_FULL_VER);
      if (!h_Edge_1_root.valid())
        continue;
      auto x = root_val_2;
      // Nested ColumnJoin (intersection): bind 'y' from 2 sources
      // MIR: (column-join :var y :sources ((Edge :handle 2 :prefix (x)) (Edge :handle 3 :prefix ())
      // ))
      auto h_Edge_2_5 = h_Edge_0_root;
      auto h_Edge_3_6 = HandleType(0, view_Edge_0_1_FULL_VER.num_rows_, 0);
      auto intersect_7 = intersect_handles(tile, h_Edge_2_5.iterators(view_Edge_0_1_FULL_VER),
                                           h_Edge_3_6.iterators(view_Edge_0_1_FULL_VER));
      for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
        auto y = it_8.value();
        auto positions = it_8.positions();
        auto ch_Edge_2_y = h_Edge_2_5.child_range(positions[0], y, tile, view_Edge_0_1_FULL_VER);
        auto ch_Edge_3_y = h_Edge_3_6.child_range(positions[1], y, tile, view_Edge_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'z' from 2 sources
        // MIR: (column-join :var z :sources ((Edge :handle 4 :prefix (y)) (Edge :handle 5 :prefix
        // (x)) ))
        auto h_Edge_4_1 = ch_Edge_3_y;
        auto h_Edge_5_2 = h_Edge_1_root;
        auto intersect_3 = intersect_handles(tile, h_Edge_4_1.iterators(view_Edge_0_1_FULL_VER),
                                             h_Edge_5_2.iterators(view_Edge_1_0_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto z = it_4.value();
          auto positions = it_4.positions();
          auto ch_Edge_4_z = h_Edge_4_1.child_range(positions[0], z, tile, view_Edge_0_1_FULL_VER);
          auto ch_Edge_5_z = h_Edge_5_2.child_range(positions[1], z, tile, view_Edge_1_0_FULL_VER);
          // Emit: Z(x, y, z)
          if (tile.thread_rank() == 0)
            output_ctx.emit_direct();
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
    auto view_Edge_0_1_FULL_VER = views[0];
    auto view_Edge_1_0_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'x' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var x :sources ((Edge :handle 0) (Edge :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Edge_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Edge_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Edge_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Edge_0_1_FULL_VER.num_rows_;
      auto h_Edge_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Edge_0_1_FULL_VER);
      if (!h_Edge_0_root.valid())
        continue;
      auto h_Edge_1_root = HandleType(0, view_Edge_1_0_FULL_VER.num_rows_, 0)
                               .prefix(root_val_2, tile, view_Edge_1_0_FULL_VER);
      if (!h_Edge_1_root.valid())
        continue;
      auto x = root_val_2;
      // Nested ColumnJoin (intersection): bind 'y' from 2 sources
      // MIR: (column-join :var y :sources ((Edge :handle 2 :prefix (x)) (Edge :handle 3 :prefix ())
      // ))
      auto h_Edge_2_5 = h_Edge_0_root;
      auto h_Edge_3_6 = HandleType(0, view_Edge_0_1_FULL_VER.num_rows_, 0);
      auto intersect_7 = intersect_handles(tile, h_Edge_2_5.iterators(view_Edge_0_1_FULL_VER),
                                           h_Edge_3_6.iterators(view_Edge_0_1_FULL_VER));
      for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
        auto y = it_8.value();
        auto positions = it_8.positions();
        auto ch_Edge_2_y = h_Edge_2_5.child_range(positions[0], y, tile, view_Edge_0_1_FULL_VER);
        auto ch_Edge_3_y = h_Edge_3_6.child_range(positions[1], y, tile, view_Edge_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'z' from 2 sources
        // MIR: (column-join :var z :sources ((Edge :handle 4 :prefix (y)) (Edge :handle 5 :prefix
        // (x)) ))
        auto h_Edge_4_1 = ch_Edge_3_y;
        auto h_Edge_5_2 = h_Edge_1_root;
        auto intersect_3 = intersect_handles(tile, h_Edge_4_1.iterators(view_Edge_0_1_FULL_VER),
                                             h_Edge_5_2.iterators(view_Edge_1_0_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto z = it_4.value();
          auto positions = it_4.positions();
          auto ch_Edge_4_z = h_Edge_4_1.child_range(positions[0], z, tile, view_Edge_0_1_FULL_VER);
          auto ch_Edge_5_z = h_Edge_5_2.child_range(positions[1], z, tile, view_Edge_1_0_FULL_VER);
          // Emit: Z(x, y, z)
          if (tile.thread_rank() == 0)
            output_ctx_0.emit_direct(x, y, z);
        }
      }
    }
  }

  // Fused kernel: single-pass join with atomic output (tail mode)
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_fused(const ViewType* __restrict__ views,
                   const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
                   uint32_t num_root_keys, ValueType* __restrict__ output_data_0,
                   std::size_t output_stride_0, uint32_t old_size_0,
                   uint32_t* __restrict__ atomic_write_pos_0, uint32_t capacity,
                   uint32_t* __restrict__ overflow_flag) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);
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
    uint32_t num_threads = num_warps;

    using SpecCtx_0 =
        SRDatalog::GPU::JIT::WS::SpeculativeOutputContext<ValueType, OutputArity_0, 16>;
    SpecCtx_0 output_ctx_0{output_data_0, atomic_write_pos_0,
                           overflow_flag, static_cast<uint32_t>(output_stride_0),
                           old_size_0,    capacity};

    using ViewType = std::remove_cvref_t<decltype(views[0])>;
    using HandleType = ViewType::NodeHandle;

    // View declarations (deduplicated by spec, 2 unique views)
    auto view_Edge_0_1_FULL_VER = views[0];
    auto view_Edge_1_0_FULL_VER = views[1];

    // Root ColumnJoin (multi-source intersection): bind 'x' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var x :sources ((Edge :handle 0) (Edge :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Edge_0_1_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Edge_0_1_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Edge_0_1_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Edge_0_1_FULL_VER.num_rows_;
      auto h_Edge_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Edge_0_1_FULL_VER);
      if (!h_Edge_0_root.valid())
        continue;
      auto h_Edge_1_root = HandleType(0, view_Edge_1_0_FULL_VER.num_rows_, 0)
                               .prefix(root_val_2, tile, view_Edge_1_0_FULL_VER);
      if (!h_Edge_1_root.valid())
        continue;
      auto x = root_val_2;
      // Nested ColumnJoin (intersection): bind 'y' from 2 sources
      // MIR: (column-join :var y :sources ((Edge :handle 2 :prefix (x)) (Edge :handle 3 :prefix ())
      // ))
      auto h_Edge_2_5 = h_Edge_0_root;
      auto h_Edge_3_6 = HandleType(0, view_Edge_0_1_FULL_VER.num_rows_, 0);
      auto intersect_7 = intersect_handles(tile, h_Edge_2_5.iterators(view_Edge_0_1_FULL_VER),
                                           h_Edge_3_6.iterators(view_Edge_0_1_FULL_VER));
      for (auto it_8 = intersect_7.begin(); it_8.valid(); it_8.next()) {
        auto y = it_8.value();
        auto positions = it_8.positions();
        auto ch_Edge_2_y = h_Edge_2_5.child_range(positions[0], y, tile, view_Edge_0_1_FULL_VER);
        auto ch_Edge_3_y = h_Edge_3_6.child_range(positions[1], y, tile, view_Edge_0_1_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'z' from 2 sources
        // MIR: (column-join :var z :sources ((Edge :handle 4 :prefix (y)) (Edge :handle 5 :prefix
        // (x)) ))
        auto h_Edge_4_1 = ch_Edge_3_y;
        auto h_Edge_5_2 = h_Edge_1_root;
        auto intersect_3 = intersect_handles(tile, h_Edge_4_1.iterators(view_Edge_0_1_FULL_VER),
                                             h_Edge_5_2.iterators(view_Edge_1_0_FULL_VER));
        for (auto it_4 = intersect_3.begin(); it_4.valid(); it_4.next()) {
          auto z = it_4.value();
          auto positions = it_4.positions();
          auto ch_Edge_4_z = h_Edge_4_1.child_range(positions[0], z, tile, view_Edge_0_1_FULL_VER);
          auto ch_Edge_5_z = h_Edge_5_2.child_range(positions[1], z, tile, view_Edge_1_0_FULL_VER);
          // Emit: Z(x, y, z)
          if (tile.thread_rank() == 0)
            output_ctx_0.emit_direct(x, y, z);
        }
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
    // Fused execution state (set by orchestrator)
    uint32_t* fused_wp_ptr_0 = nullptr;
    uint32_t* fused_of_ptr = nullptr;
    uint32_t fused_capacity = 0;
    uint32_t fused_h_wp_0 = 0;
    bool fused_overflow = false;
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
  static void execute_fused(DB& db, uint32_t iteration);
  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void read_fused_result(LaunchParams& p);
  static inline uint32_t prev_fused_total_ = 4096;
};

// Phase 1: Setup views and compute grid config
JitRunner_Triangle::LaunchParams JitRunner_Triangle::setup(DB& db, uint32_t iteration,
                                                           GPU_STREAM_T stream) {
  LaunchParams p;
  p.views_vec.reserve(NumSources);

  // Source 0: Edge version FULL_VER
  {
    auto& rel_0 = get_relation_by_schema<Edge, FULL_VER>(db);
    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{0, 1}}, false);
    p.views_vec.push_back(idx_0.view());
  }

  // Source 1: Edge version FULL_VER
  {
    auto& rel_1 = get_relation_by_schema<Edge, FULL_VER>(db);
    auto& idx_1 = rel_1.ensure_index(SRDatalog::IndexSpec{{1, 0}}, false);
    p.views_vec.push_back(idx_1.view());
  }

  // First source for root keys
  auto& first_rel = get_relation_by_schema<Edge, FULL_VER>(db);
  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{{0, 1}});
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

void JitRunner_Triangle::launch_count(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0)
    return;
  if (p.num_unique_root_keys == 0) {
    cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream);
    return;
  }
  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>(p.d_views.data(), p.root_unique_values_ptr,
                                                        p.num_unique_root_keys, p.num_root_keys,
                                                        p.thread_counts_ptr);
}

// Phase 3: Prefix scan + readback total + resize destinations
uint32_t JitRunner_Triangle::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                         p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                         thrust::plus<uint32_t>());
  uint32_t total_count = 0;
  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
                   GPU_DEVICE_TO_HOST, stream);
  GPU_STREAM_SYNCHRONIZE(stream);
  if (total_count == 0)
    return 0;

  auto& dest_rel_0 = get_relation_by_schema<Z, NEW_VER>(db);
  p.old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  dest_rel_0.resize_interned_columns(p.old_size_0 + total_count, stream);
  return total_count;
}

// Phase 3a: Prefix scan only (async, no sync)
void JitRunner_Triangle::scan_only(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0)
    return;
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                         p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                         thrust::plus<uint32_t>());
}

// Phase 3b: Read total count (call after device sync)
uint32_t JitRunner_Triangle::read_total(LaunchParams& p) {
  if (p.num_threads == 0)
    return 0;
  uint32_t total_count = 0;
  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
             GPU_DEVICE_TO_HOST);
  return total_count;
}

// Phase 4: Launch materialize kernel on given stream (no sync)
void JitRunner_Triangle::launch_materialize(DB& db, LaunchParams& p, uint32_t total_count,
                                            GPU_STREAM_T stream) {
  using ProvPtrType = semiring_value_t<SR>*;
  ProvPtrType prov_ptr = nullptr;

  auto& dest_rel_0 = get_relation_by_schema<Z, NEW_VER>(db);
  uint32_t old_size_0 = p.old_size_0;
  kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      p.thread_counts_ptr, dest_rel_0.template interned_column<0>(), prov_ptr,
      dest_rel_0.interned_stride(), old_size_0);
}

// launch_fused: launch fused kernel on given stream (no sync)
void JitRunner_Triangle::launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_unique_root_keys == 0)
    return;

  auto& dest_rel_0 = get_relation_by_schema<Z, NEW_VER>(db);
  kernel_fused<<<p.num_blocks, kBlockSize, 0, stream>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      dest_rel_0.template interned_column<0>(), dest_rel_0.interned_stride(), p.old_size_0,
      p.fused_wp_ptr_0, p.fused_capacity, p.fused_of_ptr);
}

// read_fused_result: readback fused write counts (call after device sync)
void JitRunner_Triangle::read_fused_result(LaunchParams& p) {
  GPU_MEMCPY(&p.fused_h_wp_0, p.fused_wp_ptr_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  uint32_t h_of = 0;
  GPU_MEMCPY(&h_of, p.fused_of_ptr, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  p.fused_overflow = (h_of != 0);
}

// Execute definition - calls decomposed phases sequentially
void JitRunner_Triangle::execute(DB& db, uint32_t iteration) {
  nvtxRangePushA("Triangle");

  auto p = setup(db, iteration);
  launch_count(p, 0);
  uint32_t total_count = scan_and_resize(db, p, 0);
  if (total_count == 0) {
    nvtxRangePop();
    return;
  }

  launch_materialize(db, p, total_count, 0);
  nvtxRangePop();
}

// Tail-mode fused execution: single kernel, no count/scan phase
void JitRunner_Triangle::execute_fused(DB& db, uint32_t iteration) {
  auto p = setup(db, iteration);
  if (p.num_unique_root_keys == 0)
    return;

  auto& dest_rel_0 = get_relation_by_schema<Z, NEW_VER>(db);
  uint32_t old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  uint32_t capacity = prev_fused_total_;
  dest_rel_0.resize_interned_columns(old_size_0 + capacity);

  static SRDatalog::GPU::DeviceArray<uint32_t> s_wp_0(1);
  cudaMemsetAsync(s_wp_0.data(), 0, sizeof(uint32_t), 0);
  static SRDatalog::GPU::DeviceArray<uint32_t> s_of(1);
  cudaMemsetAsync(s_of.data(), 0, sizeof(uint32_t), 0);
  kernel_fused<<<p.num_blocks, kBlockSize>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      dest_rel_0.template interned_column<0>(), dest_rel_0.interned_stride(), old_size_0,
      s_wp_0.data(), capacity, s_of.data());
  GPU_DEVICE_SYNCHRONIZE();
  uint32_t h_of = 0;
  uint32_t h_wp_0 = 0;
  GPU_MEMCPY(&h_wp_0, s_wp_0.data(), sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  GPU_MEMCPY(&h_of, s_of.data(), sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  if (!h_of) {
    dest_rel_0.resize_interned_columns(old_size_0 + h_wp_0);
    uint32_t max_wp = 0;
    max_wp = std::max(max_wp, h_wp_0);
    prev_fused_total_ = std::max(max_wp, 1u) * 4;
  } else {
    dest_rel_0.resize_interned_columns(old_size_0);
    uint32_t max_wp = 0;
    max_wp = std::max(max_wp, h_wp_0);
    prev_fused_total_ = max_wp * 4;
    execute(db, iteration);
  }
}
