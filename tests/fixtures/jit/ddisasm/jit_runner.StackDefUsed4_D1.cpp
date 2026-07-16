// =============================================================
// JIT-Generated Complete Runner: StackDefUsed4_D1
// Fully concrete - NO C++ templates/metaprogramming
// =============================================================

struct JitRunner_StackDefUsed4_D1 {
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = DdisasmPlan_DB_DeviceDB;
  using FirstSchema = StackDefUseDefUsed;
  using DestSchema = StackDefUseDefUsed;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, DELTA_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 6;
  static constexpr std::size_t OutputArity = OutputArity_0; // Legacy alias
  static constexpr std::size_t NumSources = 3;

  // GPU dedup hash table: full 64-bit hash + separate thread_id array
  struct DedupTable {
    unsigned long long* hash_slots; // full 64-bit hash per slot
    uint32_t* tid_slots;            // winner thread_id per slot
    uint32_t capacity;              // must be power of 2

    __device__ __forceinline__ unsigned long long compute_hash(
        uint32_t v0,
        uint32_t v1,
        uint32_t v2,
        uint32_t v3,
        uint32_t v4,
        uint32_t v5)
    {
      uint64_t h = 14695981039346656037ULL;
      h ^= (uint64_t)v0; h *= 1099511628211ULL;
      h ^= (uint64_t)v1; h *= 1099511628211ULL;
      h ^= (uint64_t)v2; h *= 1099511628211ULL;
      h ^= (uint64_t)v3; h *= 1099511628211ULL;
      h ^= (uint64_t)v4; h *= 1099511628211ULL;
      h ^= (uint64_t)v5; h *= 1099511628211ULL;
      return h | 1ULL; // ensure non-zero
    }

    __device__ __forceinline__ bool try_insert(
        uint32_t thread_id,
        uint32_t v0,
        uint32_t v1,
        uint32_t v2,
        uint32_t v3,
        uint32_t v4,
        uint32_t v5)
    {
      unsigned long long h = compute_hash(v0, v1, v2, v3, v4, v5);
      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);
      for (uint32_t p = 0; p < 128; p++) {
        uint32_t s = (base + p) & (capacity - 1);
        unsigned long long old = atomicCAS(&hash_slots[s], 0ULL, h);
        if (old == 0ULL) { tid_slots[s] = thread_id; return true; } // claimed
        if (old == h) return false; // same hash = duplicate
        // old != h: collision with different tuple -> probe next
      }
      return true; // probe overflow -> emit (safe)
    }

    __device__ __forceinline__ bool check_winner(
        uint32_t thread_id,
        uint32_t v0,
        uint32_t v1,
        uint32_t v2,
        uint32_t v3,
        uint32_t v4,
        uint32_t v5)
    {
      unsigned long long h = compute_hash(v0, v1, v2, v3, v4, v5);
      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);
      for (uint32_t p = 0; p < 128; p++) {
        uint32_t s = (base + p) & (capacity - 1);
        unsigned long long stored = hash_slots[s];
        if (stored == h) return tid_slots[s] == thread_id; // found: am I winner?
        if (stored == 0ULL) return true; // not found -> probe overflow, emit
        // different hash -> probe next (collision resolution)
      }
      return true; // probe overflow -> emit
    }
  };

  // Non-template kernel_count (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize) kernel_count(
      const ViewType* __restrict__ views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,
      DedupTable dedup_table,
      uint32_t* __restrict__ thread_counts) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];
    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);
    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }
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
        auto view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER = views[0];
        auto view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER = views[1];
        auto view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER = views[2];

        // Root ColumnJoin (multi-source intersection): bind 'eaUsed' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var eaUsed :sources ((StackDefUseDefUsed :handle 0) (StackDefUseLiveVarAtPriorUsed :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_;
          auto h_StackDefUseDefUsed_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          if (!h_StackDefUseDefUsed_0_root.valid()) continue;
          auto h_StackDefUseLiveVarAtPriorUsed_1_root = HandleType(0, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          if (!h_StackDefUseLiveVarAtPriorUsed_1_root.valid()) continue;
          auto eaUsed = root_val_2;
        // Nested ColumnJoin (intersection): bind 'varr' from 3 sources
        // MIR: (column-join :var varr :sources ((StackDefUseDefUsed :handle 2 :prefix (eaUsed)) (StackDefUseLiveVarAtPriorUsed :handle 3 :prefix (eaUsed)) (StackDefUseLiveVarUsedEDB :handle 4 :prefix ()) ))
        auto h_StackDefUseDefUsed_2_21 = h_StackDefUseDefUsed_0_root;
        auto h_StackDefUseLiveVarAtPriorUsed_3_22 = h_StackDefUseLiveVarAtPriorUsed_1_root;
        auto h_StackDefUseLiveVarUsedEDB_4_23 = HandleType(0, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.num_rows_, 0);
        auto intersect_24 = intersect_handles(tile, h_StackDefUseDefUsed_2_21.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER), h_StackDefUseLiveVarAtPriorUsed_3_22.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_4_23.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_25 = intersect_24.begin(); it_25.valid(); it_25.next()) {
          auto varr = it_25.value();
          auto positions = it_25.positions();
          auto ch_StackDefUseDefUsed_2_varr = h_StackDefUseDefUsed_2_21.child_range(positions[0], varr, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          auto ch_StackDefUseLiveVarAtPriorUsed_3_varr = h_StackDefUseLiveVarAtPriorUsed_3_22.child_range(positions[1], varr, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_4_varr = h_StackDefUseLiveVarUsedEDB_4_23.child_range(positions[2], varr, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'varp' from 3 sources
        // MIR: (column-join :var varp :sources ((StackDefUseDefUsed :handle 5 :prefix (eaUsed varr)) (StackDefUseLiveVarAtPriorUsed :handle 6 :prefix (eaUsed varr)) (StackDefUseLiveVarUsedEDB :handle 7 :prefix (varr)) ))
        auto h_StackDefUseDefUsed_5_16 = ch_StackDefUseDefUsed_2_varr;
        auto h_StackDefUseLiveVarAtPriorUsed_6_17 = ch_StackDefUseLiveVarAtPriorUsed_3_varr;
        auto h_StackDefUseLiveVarUsedEDB_7_18 = ch_StackDefUseLiveVarUsedEDB_4_varr;
        auto intersect_19 = intersect_handles(tile, h_StackDefUseDefUsed_5_16.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER), h_StackDefUseLiveVarAtPriorUsed_6_17.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_7_18.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto varp = it_20.value();
          auto positions = it_20.positions();
          auto ch_StackDefUseDefUsed_5_varp = h_StackDefUseDefUsed_5_16.child_range(positions[0], varp, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          auto ch_StackDefUseLiveVarAtPriorUsed_6_varp = h_StackDefUseLiveVarAtPriorUsed_6_17.child_range(positions[1], varp, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_7_varp = h_StackDefUseLiveVarUsedEDB_7_18.child_range(positions[2], varp, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'nxtBlk' from 2 sources
        // MIR: (column-join :var nxtBlk :sources ((StackDefUseLiveVarAtPriorUsed :handle 8 :prefix (eaUsed varr varp)) (StackDefUseLiveVarUsedEDB :handle 9 :prefix (varr varp)) ))
        auto h_StackDefUseLiveVarAtPriorUsed_8_12 = ch_StackDefUseLiveVarAtPriorUsed_6_varp;
        auto h_StackDefUseLiveVarUsedEDB_9_13 = ch_StackDefUseLiveVarUsedEDB_7_varp;
        auto intersect_14 = intersect_handles(tile, h_StackDefUseLiveVarAtPriorUsed_8_12.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_9_13.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto nxtBlk = it_15.value();
          auto positions = it_15.positions();
          auto ch_StackDefUseLiveVarAtPriorUsed_8_nxtBlk = h_StackDefUseLiveVarAtPriorUsed_8_12.child_range(positions[0], nxtBlk, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_9_nxtBlk = h_StackDefUseLiveVarUsedEDB_9_13.child_range(positions[1], nxtBlk, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested CartesianJoin: bind eaDef, varDefr, varDefp, varUsedr, varUsedp, nextEaUsed, _gen32, _gen33 from 2 source(s)
        // MIR: (cartesian-join :vars (eaDef varDefr varDefp varUsedr varUsedp nextEaUsed _gen32 _gen33) :sources ((StackDefUseDefUsed :handle 10 :prefix (eaUsed varr varp)) (StackDefUseLiveVarUsedEDB :handle 11 :prefix (varr varp nxtBlk)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_StackDefUseDefUsed_10_4 = ch_StackDefUseDefUsed_5_varp;  // reusing narrowed handle
        auto h_StackDefUseLiveVarUsedEDB_11_6 = ch_StackDefUseLiveVarUsedEDB_9_nxtBlk;  // reusing narrowed handle

        if (!h_StackDefUseDefUsed_10_4.valid() || !h_StackDefUseLiveVarUsedEDB_11_6.valid()) continue;

        uint32_t degree_3 = h_StackDefUseDefUsed_10_4.degree();
        uint32_t degree_5 = h_StackDefUseLiveVarUsedEDB_11_6.degree();
        uint32_t total_7 = degree_3 * degree_5;
        if (total_7 == 0) continue;

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

          auto eaDef = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(3, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varDefr = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(4, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varDefp = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(5, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varUsedr = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(3, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto varUsedp = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(4, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto nextEaUsed = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(5, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);

        // Emit: StackDefUseDefUsed(eaDef, varDefr, varDefp, nextEaUsed, varUsedr, varUsedp)
        { bool _p = dedup_table.try_insert(thread_id, eaDef, varDefr, varDefp, nextEaUsed, varUsedr, varUsedp);
          if (_p) {
        output_ctx.emit_direct();
        } }
        }
        }
        }
        }
        }
    thread_counts[thread_id] = output_ctx.count();
  }

  // Non-template kernel_materialize (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize) kernel_materialize(
      const ViewType* __restrict__ views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,
      uint32_t* __restrict__ atomic_write_pos,
      DedupTable dedup_table,
      ValueType* __restrict__ output_data_0,
      semiring_value_t<SR>* __restrict__ output_prov_0,
      std::size_t output_stride_0,
      uint32_t old_size_0) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];
    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);
    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }
    __syncthreads();
    views = s_views;
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)
    // Dedup materialize: use atomicAdd for write position (no thread_offset)

    // Dedup: atomic write context for dest 0
    ValueType* __restrict__ out_data_0 = output_data_0;
    std::size_t out_stride_0 = output_stride_0;
    uint32_t out_base_0 = old_size_0;

        using ViewType = std::remove_cvref_t<decltype(views[0])>;
        using HandleType = ViewType::NodeHandle;

        // View declarations (deduplicated by spec, 3 unique views)
        auto view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER = views[0];
        auto view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER = views[1];
        auto view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER = views[2];

        // Root ColumnJoin (multi-source intersection): bind 'eaUsed' from 2 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var eaUsed :sources ((StackDefUseDefUsed :handle 0) (StackDefUseLiveVarAtPriorUsed :handle 1) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.num_rows_;
          auto h_StackDefUseDefUsed_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          if (!h_StackDefUseDefUsed_0_root.valid()) continue;
          auto h_StackDefUseLiveVarAtPriorUsed_1_root = HandleType(0, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          if (!h_StackDefUseLiveVarAtPriorUsed_1_root.valid()) continue;
          auto eaUsed = root_val_2;
        // Nested ColumnJoin (intersection): bind 'varr' from 3 sources
        // MIR: (column-join :var varr :sources ((StackDefUseDefUsed :handle 2 :prefix (eaUsed)) (StackDefUseLiveVarAtPriorUsed :handle 3 :prefix (eaUsed)) (StackDefUseLiveVarUsedEDB :handle 4 :prefix ()) ))
        auto h_StackDefUseDefUsed_2_21 = h_StackDefUseDefUsed_0_root;
        auto h_StackDefUseLiveVarAtPriorUsed_3_22 = h_StackDefUseLiveVarAtPriorUsed_1_root;
        auto h_StackDefUseLiveVarUsedEDB_4_23 = HandleType(0, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.num_rows_, 0);
        auto intersect_24 = intersect_handles(tile, h_StackDefUseDefUsed_2_21.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER), h_StackDefUseLiveVarAtPriorUsed_3_22.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_4_23.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_25 = intersect_24.begin(); it_25.valid(); it_25.next()) {
          auto varr = it_25.value();
          auto positions = it_25.positions();
          auto ch_StackDefUseDefUsed_2_varr = h_StackDefUseDefUsed_2_21.child_range(positions[0], varr, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          auto ch_StackDefUseLiveVarAtPriorUsed_3_varr = h_StackDefUseLiveVarAtPriorUsed_3_22.child_range(positions[1], varr, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_4_varr = h_StackDefUseLiveVarUsedEDB_4_23.child_range(positions[2], varr, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'varp' from 3 sources
        // MIR: (column-join :var varp :sources ((StackDefUseDefUsed :handle 5 :prefix (eaUsed varr)) (StackDefUseLiveVarAtPriorUsed :handle 6 :prefix (eaUsed varr)) (StackDefUseLiveVarUsedEDB :handle 7 :prefix (varr)) ))
        auto h_StackDefUseDefUsed_5_16 = ch_StackDefUseDefUsed_2_varr;
        auto h_StackDefUseLiveVarAtPriorUsed_6_17 = ch_StackDefUseLiveVarAtPriorUsed_3_varr;
        auto h_StackDefUseLiveVarUsedEDB_7_18 = ch_StackDefUseLiveVarUsedEDB_4_varr;
        auto intersect_19 = intersect_handles(tile, h_StackDefUseDefUsed_5_16.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER), h_StackDefUseLiveVarAtPriorUsed_6_17.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_7_18.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto varp = it_20.value();
          auto positions = it_20.positions();
          auto ch_StackDefUseDefUsed_5_varp = h_StackDefUseDefUsed_5_16.child_range(positions[0], varp, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER);
          auto ch_StackDefUseLiveVarAtPriorUsed_6_varp = h_StackDefUseLiveVarAtPriorUsed_6_17.child_range(positions[1], varp, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_7_varp = h_StackDefUseLiveVarUsedEDB_7_18.child_range(positions[2], varp, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'nxtBlk' from 2 sources
        // MIR: (column-join :var nxtBlk :sources ((StackDefUseLiveVarAtPriorUsed :handle 8 :prefix (eaUsed varr varp)) (StackDefUseLiveVarUsedEDB :handle 9 :prefix (varr varp)) ))
        auto h_StackDefUseLiveVarAtPriorUsed_8_12 = ch_StackDefUseLiveVarAtPriorUsed_6_varp;
        auto h_StackDefUseLiveVarUsedEDB_9_13 = ch_StackDefUseLiveVarUsedEDB_7_varp;
        auto intersect_14 = intersect_handles(tile, h_StackDefUseLiveVarAtPriorUsed_8_12.iterators(view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER), h_StackDefUseLiveVarUsedEDB_9_13.iterators(view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER));
        for (auto it_15 = intersect_14.begin(); it_15.valid(); it_15.next()) {
          auto nxtBlk = it_15.value();
          auto positions = it_15.positions();
          auto ch_StackDefUseLiveVarAtPriorUsed_8_nxtBlk = h_StackDefUseLiveVarAtPriorUsed_8_12.child_range(positions[0], nxtBlk, tile, view_StackDefUseLiveVarAtPriorUsed_0_2_3_1_FULL_VER);
          auto ch_StackDefUseLiveVarUsedEDB_9_nxtBlk = h_StackDefUseLiveVarUsedEDB_9_13.child_range(positions[1], nxtBlk, tile, view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER);
        // Nested CartesianJoin: bind eaDef, varDefr, varDefp, varUsedr, varUsedp, nextEaUsed, _gen32, _gen33 from 2 source(s)
        // MIR: (cartesian-join :vars (eaDef varDefr varDefp varUsedr varUsedp nextEaUsed _gen32 _gen33) :sources ((StackDefUseDefUsed :handle 10 :prefix (eaUsed varr varp)) (StackDefUseLiveVarUsedEDB :handle 11 :prefix (varr varp nxtBlk)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_StackDefUseDefUsed_10_4 = ch_StackDefUseDefUsed_5_varp;  // reusing narrowed handle
        auto h_StackDefUseLiveVarUsedEDB_11_6 = ch_StackDefUseLiveVarUsedEDB_9_nxtBlk;  // reusing narrowed handle

        if (!h_StackDefUseDefUsed_10_4.valid() || !h_StackDefUseLiveVarUsedEDB_11_6.valid()) continue;

        uint32_t degree_3 = h_StackDefUseDefUsed_10_4.degree();
        uint32_t degree_5 = h_StackDefUseLiveVarUsedEDB_11_6.degree();
        uint32_t total_7 = degree_3 * degree_5;
        if (total_7 == 0) continue;

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

          auto eaDef = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(3, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varDefr = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(4, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varDefp = view_StackDefUseDefUsed_3_4_5_0_1_2_DELTA_VER.get_value(5, h_StackDefUseDefUsed_10_4.begin() + idx0_9);
          auto varUsedr = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(3, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto varUsedp = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(4, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto nextEaUsed = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(5, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto _gen32 = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(6, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);
          auto _gen33 = view_StackDefUseLiveVarUsedEDB_1_2_0_3_4_5_6_7_FULL_VER.get_value(7, h_StackDefUseLiveVarUsedEDB_11_6.begin() + idx1_10);

        // Emit: StackDefUseDefUsed(eaDef, varDefr, varDefp, nextEaUsed, varUsedr, varUsedp)
        { bool _p = dedup_table.try_insert(thread_id, eaDef, varDefr, varDefp, nextEaUsed, varUsedr, varUsedp);
          if (_p) {
        {
          uint32_t pos = atomicAdd(atomic_write_pos, 1u);
          out_data_0[(pos + out_base_0) + 0 * out_stride_0] = eaDef;
          out_data_0[(pos + out_base_0) + 1 * out_stride_0] = varDefr;
          out_data_0[(pos + out_base_0) + 2 * out_stride_0] = varDefp;
          out_data_0[(pos + out_base_0) + 3 * out_stride_0] = nextEaUsed;
          out_data_0[(pos + out_base_0) + 4 * out_stride_0] = varUsedr;
          out_data_0[(pos + out_base_0) + 5 * out_stride_0] = varUsedp;
        }
        } }
        }
        }
        }
        }
        }
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
    // Dedup hash table for in-kernel duplicate elimination
    SRDatalog::GPU::DeviceArray<unsigned long long> dedup_hash_arr{0};
    SRDatalog::GPU::DeviceArray<uint32_t> dedup_tid_arr{0};
    DedupTable dedup_table{};
    SRDatalog::GPU::DeviceArray<uint32_t> atomic_write_pos{0};
    uint32_t* atomic_write_pos_ptr = nullptr;
  };

  // Phase-decomposed methods for stream-parallel execution
  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  // Non-template execute - calls kernels directly
  static void execute(DB& db, uint32_t iteration);
};

// Phase 1: Setup views and compute grid config
JitRunner_StackDefUsed4_D1::LaunchParams JitRunner_StackDefUsed4_D1::setup(DB& db, uint32_t iteration, GPU_STREAM_T stream) {
  LaunchParams p;
  p.views_vec.reserve(NumSources);

  // Source 0: StackDefUseDefUsed version DELTA_VER
  {
    auto& rel_0 = (iteration == 0) ? get_relation_by_schema<StackDefUseDefUsed, FULL_VER>(db) : get_relation_by_schema<StackDefUseDefUsed, DELTA_VER>(db);
    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{3, 4, 5, 0, 1, 2}}, false);
    p.views_vec.push_back(idx_0.view());
  }

  // Source 1: StackDefUseLiveVarAtPriorUsed version FULL_VER
  {
    auto& rel_1 = get_relation_by_schema<StackDefUseLiveVarAtPriorUsed, FULL_VER>(db);
    auto& idx_1 = rel_1.ensure_index(SRDatalog::IndexSpec{{0, 2, 3, 1}}, false);
    p.views_vec.push_back(idx_1.view());
  }

  // Source 4: StackDefUseLiveVarUsedEDB version FULL_VER
  {
    auto& rel_4 = get_relation_by_schema<StackDefUseLiveVarUsedEDB, FULL_VER>(db);
    auto& idx_4 = rel_4.ensure_index(SRDatalog::IndexSpec{{1, 2, 0, 3, 4, 5, 6, 7}}, false);
    p.views_vec.push_back(idx_4.view());
  }

  // First source for root keys
  auto& first_rel = (iteration == 0) ? get_relation_by_schema<StackDefUseDefUsed, FULL_VER>(db) : get_relation_by_schema<StackDefUseDefUsed, DELTA_VER>(db);
  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{{3, 4, 5, 0, 1, 2}});
  p.num_root_keys = first_idx.root().degree();
  p.num_unique_root_keys = static_cast<uint32_t>(first_idx.num_unique_root_values());
  p.root_unique_values_ptr = (p.num_unique_root_keys > 0) ? first_idx.root_unique_values().data() : nullptr;
  p.num_full_unique_root_keys = p.num_unique_root_keys;

  // Copy views to device using provided stream (NOT stream 0)
  p.d_views = SRDatalog::GPU::DeviceArray<ViewType>(p.views_vec.size());
  GPU_MEMCPY_ASYNC(p.d_views.data(), p.views_vec.data(), p.views_vec.size() * sizeof(ViewType), GPU_HOST_TO_DEVICE, stream);

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
  // Dedup hash table: sized relative to FULL relation
  {
    auto& _dedup_full_rel = get_relation_by_schema<StackDefUseDefUsed, FULL_VER>(db);
    std::size_t full_size = std::max(_dedup_full_rel.size(), static_cast<std::size_t>(p.num_root_keys));
    std::size_t cap64 = std::max(full_size * 8, static_cast<std::size_t>(1u << 20));
    cap64 = std::min(cap64, static_cast<std::size_t>(1u << 30)); // cap at 1B entries (~12GB)
    uint32_t cap = static_cast<uint32_t>(cap64);
    // Round up to power of 2
    cap--; cap |= cap>>1; cap |= cap>>2; cap |= cap>>4; cap |= cap>>8; cap |= cap>>16; cap++;
    p.dedup_hash_arr = SRDatalog::GPU::DeviceArray<unsigned long long>(cap);
    p.dedup_tid_arr = SRDatalog::GPU::DeviceArray<uint32_t>(cap);
    cudaMemsetAsync(p.dedup_hash_arr.data(), 0, cap * sizeof(unsigned long long), stream);
    cudaMemsetAsync(p.dedup_tid_arr.data(), 0, cap * sizeof(uint32_t), stream);
    p.dedup_table.hash_slots = reinterpret_cast<unsigned long long*>(p.dedup_hash_arr.data());
    p.dedup_table.tid_slots = p.dedup_tid_arr.data();
    p.dedup_table.capacity = cap;
    p.atomic_write_pos = SRDatalog::GPU::DeviceArray<uint32_t>(1);
    cudaMemsetAsync(p.atomic_write_pos.data(), 0, sizeof(uint32_t), stream);
    p.atomic_write_pos_ptr = p.atomic_write_pos.data();
  }
  return p;
}

void JitRunner_StackDefUsed4_D1::launch_count(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0) return;
  if (p.num_unique_root_keys == 0) { cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream); return; }
  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>(p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys, p.dedup_table, p.thread_counts_ptr);
}

// Phase 3: Prefix scan + readback total + resize destinations
uint32_t JitRunner_StackDefUsed4_D1::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0, thrust::plus<uint32_t>());
  uint32_t total_count = 0;
  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t), GPU_DEVICE_TO_HOST, stream);
  GPU_STREAM_SYNCHRONIZE(stream);
  if (total_count == 0) return 0;

  auto& dest_rel_0 = get_relation_by_schema<StackDefUseDefUsed, NEW_VER>(db);
  p.old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  dest_rel_0.resize_interned_columns(p.old_size_0 + total_count, stream);
  return total_count;
}

// Phase 3a: Prefix scan only (async, no sync)
void JitRunner_StackDefUsed4_D1::scan_only(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0) return;
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0, thrust::plus<uint32_t>());
}

// Phase 3b: Read total count (call after device sync)
uint32_t JitRunner_StackDefUsed4_D1::read_total(LaunchParams& p) {
  if (p.num_threads == 0) return 0;
  uint32_t total_count = 0;
  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  return total_count;
}

// Phase 4: Launch materialize kernel on given stream (no sync)
void JitRunner_StackDefUsed4_D1::launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream) {
  using ProvPtrType = semiring_value_t<SR>*;
  ProvPtrType prov_ptr = nullptr;

  auto& dest_rel_0 = get_relation_by_schema<StackDefUseDefUsed, NEW_VER>(db);
  uint32_t old_size_0 = p.old_size_0;
  kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      p.atomic_write_pos_ptr,
      p.dedup_table,
      dest_rel_0.template interned_column<0>(), prov_ptr, dest_rel_0.interned_stride(), old_size_0);
}

// Execute definition - calls decomposed phases sequentially
void JitRunner_StackDefUsed4_D1::execute(DB& db, uint32_t iteration) {
  nvtxRangePushA("StackDefUsed4_D1");

  auto p = setup(db, iteration);
  launch_count(p, 0);
  uint32_t total_count = scan_and_resize(db, p, 0);
  if (total_count == 0) { nvtxRangePop(); return; }
  // Clear for materialize: fresh table ensures identical dedup decisions
  cudaMemsetAsync(p.dedup_hash_arr.data(), 0, p.dedup_table.capacity * sizeof(unsigned long long), 0);
  cudaMemsetAsync(p.atomic_write_pos_ptr, 0, sizeof(uint32_t), 0);
  launch_materialize(db, p, total_count, 0);
  nvtxRangePop();
}
