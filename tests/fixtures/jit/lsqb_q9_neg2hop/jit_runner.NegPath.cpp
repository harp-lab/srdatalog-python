// =============================================================
// JIT-Generated Complete Runner: NegPath
// Fully concrete - NO C++ templates/metaprogramming
// =============================================================

struct JitRunner_NegPath {
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = LSQB_Q9_Rules_DB_DeviceDB;
  using FirstSchema = Knows;
  using DestSchema = Path;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType =
      std::decay_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 4;
  static constexpr std::size_t OutputArity = OutputArity_0;  // Legacy alias
  static constexpr std::size_t NumSources = 3;

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

    // View declarations (deduplicated by spec, 3 unique views)
    auto view_Knows_1_0_FULL_VER = views[0];
    auto view_Knows_0_1_FULL_VER = views[1];
    auto view_HasInterest_0_1_FULL_VER = views[2];

    // Root ColumnJoin (multi-source intersection): bind 'p2' from 2 sources
    // Uses root_unique_values + prefix() pattern (like TMP)
    // MIR: (column-join :var p2 :sources ((Knows :handle 0) (Knows :handle 1) ))
    // WARP MODE: 32 threads cooperatively handle one row
    for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
      auto root_val_2 = root_unique_values[y_idx_1];

      uint32_t hint_lo_3 = y_idx_1;
      uint32_t hint_hi_4 = view_Knows_1_0_FULL_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
      hint_hi_4 = (hint_hi_4 <= view_Knows_1_0_FULL_VER.num_rows_)
                      ? hint_hi_4
                      : view_Knows_1_0_FULL_VER.num_rows_;
      hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_Knows_1_0_FULL_VER.num_rows_;
      auto h_Knows_0_root =
          HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_Knows_1_0_FULL_VER);
      if (!h_Knows_0_root.valid())
        continue;
      auto h_Knows_1_root = HandleType(0, view_Knows_0_1_FULL_VER.num_rows_, 0)
                                .prefix(root_val_2, tile, view_Knows_0_1_FULL_VER);
      if (!h_Knows_1_root.valid())
        continue;
      auto p2 = root_val_2;
      // Nested ColumnJoin (intersection): bind 'p3' from 2 sources
      // MIR: (column-join :var p3 :sources ((Knows :handle 2 :prefix (p2)) (HasInterest :handle 3
      // :prefix ()) ))
      auto h_Knows_2_14 = h_Knows_1_root;
      auto h_HasInterest_3_15 = HandleType(0, view_HasInterest_0_1_FULL_VER.num_rows_, 0);
      auto intersect_16 =
          intersect_handles(tile, h_Knows_2_14.iterators(view_Knows_0_1_FULL_VER),
                            h_HasInterest_3_15.iterators(view_HasInterest_0_1_FULL_VER));
      for (auto it_17 = intersect_16.begin(); it_17.valid(); it_17.next()) {
        auto p3 = it_17.value();
        auto positions = it_17.positions();
        auto ch_Knows_2_p3 =
            h_Knows_2_14.child_range(positions[0], p3, tile, view_Knows_0_1_FULL_VER);
        auto ch_HasInterest_3_p3 =
            h_HasInterest_3_15.child_range(positions[1], p3, tile, view_HasInterest_0_1_FULL_VER);
        // Nested CartesianJoin: bind p1, t from 2 source(s)
        // MIR: (cartesian-join :vars (p1 t) :sources ((Knows :handle 4 :prefix (p2)) (HasInterest
        // :handle 5 :prefix (p3)) ))
        uint32_t lane_3 = tile.thread_rank();
        uint32_t group_size_4 = tile.size();

        auto h_Knows_4_6 = h_Knows_0_root;             // reusing narrowed handle
        auto h_HasInterest_5_8 = ch_HasInterest_3_p3;  // reusing narrowed handle

        if (!h_Knows_4_6.valid() || !h_HasInterest_5_8.valid())
          continue;

        uint32_t degree_5 = h_Knows_4_6.degree();
        uint32_t degree_7 = h_HasInterest_5_8.degree();
        uint32_t total_9 = degree_5 * degree_7;
        if (total_9 == 0)
          continue;

        // Pre-narrow negation handle for Knows (pre-Cartesian vars: p3)
        auto h_Knows_neg_pre_1 = HandleType(0, view_Knows_1_0_FULL_VER.num_rows_, 0)
                                     .prefix(p3, tile, view_Knows_1_0_FULL_VER);

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

          auto p1 = view_Knows_1_0_FULL_VER.get_value(1, h_Knows_4_6.begin() + idx0_11);
          auto t = view_HasInterest_0_1_FULL_VER.get_value(1, h_HasInterest_5_8.begin() + idx1_12);

          // Negation: NOT EXISTS in Knows
          // MIR: (negation :rel Knows :prefix (p3 p1) :handle 6)
          // Using pre-narrowed handle (pre-Cartesian vars: p3)
          auto h_Knows_neg_6_2 = h_Knows_neg_pre_1.prefix_seq(p1, view_Knows_1_0_FULL_VER);
          if (!h_Knows_neg_6_2.valid()) {
            if (p1 != p3) {
              // Emit: Path(p1, p2, p3, t)
              output_ctx.emit_direct();
            }
          }
        }
      }
    }
    thread_counts[thread_id] = output_ctx.count();
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
  };

  // Phase-decomposed methods for stream-parallel execution
  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  // Non-template execute - calls kernels directly
  static void execute(DB& db, uint32_t iteration);
};

// Phase 1: Setup views and compute grid config
JitRunner_NegPath::LaunchParams JitRunner_NegPath::setup(DB& db, uint32_t iteration,
                                                         GPU_STREAM_T stream) {
  LaunchParams p;
  p.views_vec.reserve(NumSources);

  // Source 0: Knows version FULL_VER
  {
    auto& rel_0 = get_relation_by_schema<Knows, FULL_VER>(db);
    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{1, 0}}, false);
    p.views_vec.push_back(idx_0.view());
  }

  // Source 1: Knows version FULL_VER
  {
    auto& rel_1 = get_relation_by_schema<Knows, FULL_VER>(db);
    auto& idx_1 = rel_1.ensure_index(SRDatalog::IndexSpec{{0, 1}}, false);
    p.views_vec.push_back(idx_1.view());
  }

  // Source 3: HasInterest version FULL_VER
  {
    auto& rel_3 = get_relation_by_schema<HasInterest, FULL_VER>(db);
    auto& idx_3 = rel_3.ensure_index(SRDatalog::IndexSpec{{0, 1}}, false);
    p.views_vec.push_back(idx_3.view());
  }

  // First source for root keys
  auto& first_rel = get_relation_by_schema<Knows, FULL_VER>(db);
  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{{1, 0}});
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

void JitRunner_NegPath::launch_count(LaunchParams& p, GPU_STREAM_T stream) {
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
uint32_t JitRunner_NegPath::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                         p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                         thrust::plus<uint32_t>());
  uint32_t total_count = 0;
  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
                   GPU_DEVICE_TO_HOST, stream);
  GPU_STREAM_SYNCHRONIZE(stream);
  if (total_count == 0)
    return 0;

  auto& dest_rel_0 = get_relation_by_schema<Path, NEW_VER>(db);
  p.old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  dest_rel_0.resize_interned_columns(p.old_size_0 + total_count, stream);
  return total_count;
}

// Phase 3a: Prefix scan only (async, no sync)
void JitRunner_NegPath::scan_only(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0)
    return;
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr,
                         p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0,
                         thrust::plus<uint32_t>());
}

// Phase 3b: Read total count (call after device sync)
uint32_t JitRunner_NegPath::read_total(LaunchParams& p) {
  if (p.num_threads == 0)
    return 0;
  uint32_t total_count = 0;
  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t),
             GPU_DEVICE_TO_HOST);
  return total_count;
}

// Execute definition - calls decomposed phases sequentially
void JitRunner_NegPath::execute(DB& db, uint32_t iteration) {
  nvtxRangePushA("NegPath");

  auto p = setup(db, iteration);
  launch_count(p, 0);
  scan_only(p, 0);
  GPU_STREAM_SYNCHRONIZE(0);
  uint32_t total_count = read_total(p);
  if (total_count == 0) {
    nvtxRangePop();
    return;
  }

  nvtxRangePop();
}
