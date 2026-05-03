// =============================================================
// JIT-Generated Complete Runner: DefUsedAddr4_D0
// Fully concrete - NO C++ templates/metaprogramming
// =============================================================

struct JitRunner_DefUsedAddr4_D0 {
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = DdisasmPlan_DB_DeviceDB;
  using FirstSchema = DefUsedForAddress;
  using DestSchema = DefUsedForAddress;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, DELTA_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 3;
  static constexpr std::size_t OutputArity = OutputArity_0; // Legacy alias
  static constexpr std::size_t NumSources = 6;

  // Non-template kernel_count (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize) kernel_count(
      const ViewType* __restrict__ views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,
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

        // View declarations (deduplicated by spec, 5 unique views)
        auto view_DefUsedForAddress_0_1_2_DELTA_VER = views[0];
        auto view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER = views[1];
        auto view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER = views[2];
        auto view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER = views[3];
        auto view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4];

        // Root ColumnJoin (multi-source intersection): bind 'eaLoad' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var eaLoad :sources ((DefUsedForAddress :handle 0) (ArchMemoryAccess :handle 1) (StackDefUseDefUsed :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          auto h_DefUsedForAddress_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          if (!h_DefUsedForAddress_0_root.valid()) continue;
          auto h_ArchMemoryAccess_1_root = HandleType(0, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          if (!h_ArchMemoryAccess_1_root.valid()) continue;
          auto h_StackDefUseDefUsed_2_root = HandleType(0, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          if (!h_StackDefUseDefUsed_2_root.valid()) continue;
          auto eaLoad = root_val_2;
        // Nested ColumnJoin (intersection): bind 'reg2' from 2 sources
        // MIR: (column-join :var reg2 :sources ((DefUsedForAddress :handle 3 :prefix (eaLoad)) (ArchMemoryAccess :handle 4 :prefix (eaLoad)) ))
        auto h_DefUsedForAddress_3_42 = h_DefUsedForAddress_0_root;
        auto h_ArchMemoryAccess_4_43 = h_ArchMemoryAccess_1_root;
        auto intersect_44 = intersect_handles(tile, h_DefUsedForAddress_3_42.iterators(view_DefUsedForAddress_0_1_2_DELTA_VER), h_ArchMemoryAccess_4_43.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER));
        for (auto it_45 = intersect_44.begin(); it_45.valid(); it_45.next()) {
          auto reg2 = it_45.value();
          auto positions = it_45.positions();
          auto ch_DefUsedForAddress_3_reg2 = h_DefUsedForAddress_3_42.child_range(positions[0], reg2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          auto ch_ArchMemoryAccess_4_reg2 = h_ArchMemoryAccess_4_43.child_range(positions[1], reg2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseLoad' from 2 sources
        // MIR: (column-join :var regBaseLoad :sources ((ArchMemoryAccess :handle 5 :prefix (eaLoad reg2)) (StackDefUseDefUsed :handle 6 :prefix (eaLoad)) ))
        auto h_ArchMemoryAccess_5_38 = ch_ArchMemoryAccess_4_reg2;
        auto h_StackDefUseDefUsed_6_39 = h_StackDefUseDefUsed_2_root;
        auto intersect_40 = intersect_handles(tile, h_ArchMemoryAccess_5_38.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_6_39.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_41 = intersect_40.begin(); it_41.valid(); it_41.next()) {
          auto regBaseLoad = it_41.value();
          auto positions = it_41.positions();
          auto ch_ArchMemoryAccess_5_regBaseLoad = h_ArchMemoryAccess_5_38.child_range(positions[0], regBaseLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_6_regBaseLoad = h_StackDefUseDefUsed_6_39.child_range(positions[1], regBaseLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosLoad' from 2 sources
        // MIR: (column-join :var stackPosLoad :sources ((ArchMemoryAccess :handle 7 :prefix (eaLoad reg2 regBaseLoad)) (StackDefUseDefUsed :handle 8 :prefix (eaLoad regBaseLoad)) ))
        auto h_ArchMemoryAccess_7_34 = ch_ArchMemoryAccess_5_regBaseLoad;
        auto h_StackDefUseDefUsed_8_35 = ch_StackDefUseDefUsed_6_regBaseLoad;
        auto intersect_36 = intersect_handles(tile, h_ArchMemoryAccess_7_34.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_8_35.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_37 = intersect_36.begin(); it_37.valid(); it_37.next()) {
          auto stackPosLoad = it_37.value();
          auto positions = it_37.positions();
          auto ch_ArchMemoryAccess_7_stackPosLoad = h_ArchMemoryAccess_7_34.child_range(positions[0], stackPosLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_8_stackPosLoad = h_StackDefUseDefUsed_8_35.child_range(positions[1], stackPosLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'eaSt' from 3 sources
        // MIR: (column-join :var eaSt :sources ((StackDefUseDefUsed :handle 9 :prefix (eaLoad regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 10 :prefix ()) (RegDefUseDefUsed :handle 11 :prefix ()) ))
        for (int _nseg_2 = 0; _nseg_2 < 2; _nseg_2++) {
          view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4 + _nseg_2];
          auto h_StackDefUseDefUsed_9_29 = ch_StackDefUseDefUsed_8_stackPosLoad;
          auto h_ArchMemoryAccess_10_30 = HandleType(0, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.num_rows_, 0);
          auto h_RegDefUseDefUsed_11_31 = HandleType(0, view_RegDefUseDefUsed_2_1_0_3_FULL_VER.num_rows_, 0);
          auto intersect_32 = intersect_handles(tile, h_StackDefUseDefUsed_9_29.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_10_30.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_11_31.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
          for (auto it_33 = intersect_32.begin(); it_33.valid(); it_33.next()) {
          auto eaSt = it_33.value();
          auto positions = it_33.positions();
          auto ch_StackDefUseDefUsed_9_eaSt = h_StackDefUseDefUsed_9_29.child_range(positions[0], eaSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_10_eaSt = h_ArchMemoryAccess_10_30.child_range(positions[1], eaSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_11_eaSt = h_RegDefUseDefUsed_11_31.child_range(positions[2], eaSt, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseSt' from 2 sources
        // MIR: (column-join :var regBaseSt :sources ((StackDefUseDefUsed :handle 12 :prefix (eaLoad regBaseLoad stackPosLoad eaSt)) (ArchMemoryAccess :handle 13 :prefix (eaSt)) ))
        auto h_StackDefUseDefUsed_12_25 = ch_StackDefUseDefUsed_9_eaSt;
        auto h_ArchMemoryAccess_13_26 = ch_ArchMemoryAccess_10_eaSt;
        auto intersect_27 = intersect_handles(tile, h_StackDefUseDefUsed_12_25.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_13_26.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_28 = intersect_27.begin(); it_28.valid(); it_28.next()) {
          auto regBaseSt = it_28.value();
          auto positions = it_28.positions();
          auto ch_StackDefUseDefUsed_12_regBaseSt = h_StackDefUseDefUsed_12_25.child_range(positions[0], regBaseSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_13_regBaseSt = h_ArchMemoryAccess_13_26.child_range(positions[1], regBaseSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosSt' from 2 sources
        // MIR: (column-join :var stackPosSt :sources ((StackDefUseDefUsed :handle 14 :prefix (eaLoad regBaseLoad stackPosLoad eaSt regBaseSt)) (ArchMemoryAccess :handle 15 :prefix (eaSt regBaseSt)) ))
        auto h_StackDefUseDefUsed_14_21 = ch_StackDefUseDefUsed_12_regBaseSt;
        auto h_ArchMemoryAccess_15_22 = ch_ArchMemoryAccess_13_regBaseSt;
        auto intersect_23 = intersect_handles(tile, h_StackDefUseDefUsed_14_21.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_15_22.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_24 = intersect_23.begin(); it_24.valid(); it_24.next()) {
          auto stackPosSt = it_24.value();
          auto positions = it_24.positions();
          auto ch_StackDefUseDefUsed_14_stackPosSt = h_StackDefUseDefUsed_14_21.child_range(positions[0], stackPosSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_15_stackPosSt = h_ArchMemoryAccess_15_22.child_range(positions[1], stackPosSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'reg1' from 2 sources
        // MIR: (column-join :var reg1 :sources ((ArchMemoryAccess :handle 16 :prefix (eaSt regBaseSt stackPosSt)) (RegDefUseDefUsed :handle 17 :prefix (eaSt)) ))
        auto h_ArchMemoryAccess_16_17 = ch_ArchMemoryAccess_15_stackPosSt;
        auto h_RegDefUseDefUsed_17_18 = ch_RegDefUseDefUsed_11_eaSt;
        auto intersect_19 = intersect_handles(tile, h_ArchMemoryAccess_16_17.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_17_18.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto reg1 = it_20.value();
          auto positions = it_20.positions();
          auto ch_ArchMemoryAccess_16_reg1 = h_ArchMemoryAccess_16_17.child_range(positions[0], reg1, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_17_reg1 = h_RegDefUseDefUsed_17_18.child_range(positions[1], reg1, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested CartesianJoin: bind mtype, _c1, _c2, _c3, _c4, eaDef, _gen26 from 4 source(s)
        // MIR: (cartesian-join :vars (mtype _c1 _c2 _c3 _c4 eaDef _gen26) :sources ((DefUsedForAddress :handle 18 :prefix (eaLoad reg2)) (ArchMemoryAccess :handle 19 :prefix (eaLoad reg2 regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 20 :prefix (eaSt regBaseSt stackPosSt reg1)) (RegDefUseDefUsed :handle 21 :prefix (eaSt reg1)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_DefUsedForAddress_18_4 = ch_DefUsedForAddress_3_reg2;  // reusing narrowed handle
        auto h_ArchMemoryAccess_19_6 = ch_ArchMemoryAccess_7_stackPosLoad;  // reusing narrowed handle
        auto h_ArchMemoryAccess_20_8 = ch_ArchMemoryAccess_16_reg1;  // reusing narrowed handle
        auto h_RegDefUseDefUsed_21_10 = ch_RegDefUseDefUsed_17_reg1;  // reusing narrowed handle

        if (!h_DefUsedForAddress_18_4.valid() || !h_ArchMemoryAccess_19_6.valid() || !h_ArchMemoryAccess_20_8.valid() || !h_RegDefUseDefUsed_21_10.valid()) continue;

        uint32_t degree_3 = h_DefUsedForAddress_18_4.degree();
        uint32_t degree_5 = h_ArchMemoryAccess_19_6.degree();
        uint32_t degree_7 = h_ArchMemoryAccess_20_8.degree();
        uint32_t degree_9 = h_RegDefUseDefUsed_21_10.degree();
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

          auto mtype = view_DefUsedForAddress_0_1_2_DELTA_VER.get_value(2, h_DefUsedForAddress_18_4.begin() + idx0_13);
          auto _c1 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c2 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c3 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto _c4 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto eaDef = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(2, h_RegDefUseDefUsed_21_10.begin() + idx3_16);

        if (_c1 == 0 && _c2 == 4) {
        if (_c3 == 2309374 && _c4 == 4) {
        // Emit: DefUsedForAddress(eaDef, reg1, mtype)
        output_ctx.emit_direct();
        }
        }
        }
        }
        }
        }
          }
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
      const uint32_t* __restrict__ thread_offsets,
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
    uint32_t thread_offset = thread_offsets[thread_id];

    using OutputCtx_0 = SRDatalog::GPU::OutputContext<ValueType, SR, false, Layout, OutputArity_0>;
    OutputCtx_0 output_ctx_0{output_data_0, output_prov_0, output_stride_0, old_size_0 + thread_offset};

        using ViewType = std::remove_cvref_t<decltype(views[0])>;
        using HandleType = ViewType::NodeHandle;

        // View declarations (deduplicated by spec, 5 unique views)
        auto view_DefUsedForAddress_0_1_2_DELTA_VER = views[0];
        auto view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER = views[1];
        auto view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER = views[2];
        auto view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER = views[3];
        auto view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4];

        // Root ColumnJoin (multi-source intersection): bind 'eaLoad' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var eaLoad :sources ((DefUsedForAddress :handle 0) (ArchMemoryAccess :handle 1) (StackDefUseDefUsed :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          auto h_DefUsedForAddress_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          if (!h_DefUsedForAddress_0_root.valid()) continue;
          auto h_ArchMemoryAccess_1_root = HandleType(0, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          if (!h_ArchMemoryAccess_1_root.valid()) continue;
          auto h_StackDefUseDefUsed_2_root = HandleType(0, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          if (!h_StackDefUseDefUsed_2_root.valid()) continue;
          auto eaLoad = root_val_2;
        // Nested ColumnJoin (intersection): bind 'reg2' from 2 sources
        // MIR: (column-join :var reg2 :sources ((DefUsedForAddress :handle 3 :prefix (eaLoad)) (ArchMemoryAccess :handle 4 :prefix (eaLoad)) ))
        auto h_DefUsedForAddress_3_42 = h_DefUsedForAddress_0_root;
        auto h_ArchMemoryAccess_4_43 = h_ArchMemoryAccess_1_root;
        auto intersect_44 = intersect_handles(tile, h_DefUsedForAddress_3_42.iterators(view_DefUsedForAddress_0_1_2_DELTA_VER), h_ArchMemoryAccess_4_43.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER));
        for (auto it_45 = intersect_44.begin(); it_45.valid(); it_45.next()) {
          auto reg2 = it_45.value();
          auto positions = it_45.positions();
          auto ch_DefUsedForAddress_3_reg2 = h_DefUsedForAddress_3_42.child_range(positions[0], reg2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          auto ch_ArchMemoryAccess_4_reg2 = h_ArchMemoryAccess_4_43.child_range(positions[1], reg2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseLoad' from 2 sources
        // MIR: (column-join :var regBaseLoad :sources ((ArchMemoryAccess :handle 5 :prefix (eaLoad reg2)) (StackDefUseDefUsed :handle 6 :prefix (eaLoad)) ))
        auto h_ArchMemoryAccess_5_38 = ch_ArchMemoryAccess_4_reg2;
        auto h_StackDefUseDefUsed_6_39 = h_StackDefUseDefUsed_2_root;
        auto intersect_40 = intersect_handles(tile, h_ArchMemoryAccess_5_38.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_6_39.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_41 = intersect_40.begin(); it_41.valid(); it_41.next()) {
          auto regBaseLoad = it_41.value();
          auto positions = it_41.positions();
          auto ch_ArchMemoryAccess_5_regBaseLoad = h_ArchMemoryAccess_5_38.child_range(positions[0], regBaseLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_6_regBaseLoad = h_StackDefUseDefUsed_6_39.child_range(positions[1], regBaseLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosLoad' from 2 sources
        // MIR: (column-join :var stackPosLoad :sources ((ArchMemoryAccess :handle 7 :prefix (eaLoad reg2 regBaseLoad)) (StackDefUseDefUsed :handle 8 :prefix (eaLoad regBaseLoad)) ))
        auto h_ArchMemoryAccess_7_34 = ch_ArchMemoryAccess_5_regBaseLoad;
        auto h_StackDefUseDefUsed_8_35 = ch_StackDefUseDefUsed_6_regBaseLoad;
        auto intersect_36 = intersect_handles(tile, h_ArchMemoryAccess_7_34.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_8_35.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_37 = intersect_36.begin(); it_37.valid(); it_37.next()) {
          auto stackPosLoad = it_37.value();
          auto positions = it_37.positions();
          auto ch_ArchMemoryAccess_7_stackPosLoad = h_ArchMemoryAccess_7_34.child_range(positions[0], stackPosLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_8_stackPosLoad = h_StackDefUseDefUsed_8_35.child_range(positions[1], stackPosLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'eaSt' from 3 sources
        // MIR: (column-join :var eaSt :sources ((StackDefUseDefUsed :handle 9 :prefix (eaLoad regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 10 :prefix ()) (RegDefUseDefUsed :handle 11 :prefix ()) ))
        for (int _nseg_2 = 0; _nseg_2 < 2; _nseg_2++) {
          view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4 + _nseg_2];
          auto h_StackDefUseDefUsed_9_29 = ch_StackDefUseDefUsed_8_stackPosLoad;
          auto h_ArchMemoryAccess_10_30 = HandleType(0, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.num_rows_, 0);
          auto h_RegDefUseDefUsed_11_31 = HandleType(0, view_RegDefUseDefUsed_2_1_0_3_FULL_VER.num_rows_, 0);
          auto intersect_32 = intersect_handles(tile, h_StackDefUseDefUsed_9_29.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_10_30.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_11_31.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
          for (auto it_33 = intersect_32.begin(); it_33.valid(); it_33.next()) {
          auto eaSt = it_33.value();
          auto positions = it_33.positions();
          auto ch_StackDefUseDefUsed_9_eaSt = h_StackDefUseDefUsed_9_29.child_range(positions[0], eaSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_10_eaSt = h_ArchMemoryAccess_10_30.child_range(positions[1], eaSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_11_eaSt = h_RegDefUseDefUsed_11_31.child_range(positions[2], eaSt, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseSt' from 2 sources
        // MIR: (column-join :var regBaseSt :sources ((StackDefUseDefUsed :handle 12 :prefix (eaLoad regBaseLoad stackPosLoad eaSt)) (ArchMemoryAccess :handle 13 :prefix (eaSt)) ))
        auto h_StackDefUseDefUsed_12_25 = ch_StackDefUseDefUsed_9_eaSt;
        auto h_ArchMemoryAccess_13_26 = ch_ArchMemoryAccess_10_eaSt;
        auto intersect_27 = intersect_handles(tile, h_StackDefUseDefUsed_12_25.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_13_26.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_28 = intersect_27.begin(); it_28.valid(); it_28.next()) {
          auto regBaseSt = it_28.value();
          auto positions = it_28.positions();
          auto ch_StackDefUseDefUsed_12_regBaseSt = h_StackDefUseDefUsed_12_25.child_range(positions[0], regBaseSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_13_regBaseSt = h_ArchMemoryAccess_13_26.child_range(positions[1], regBaseSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosSt' from 2 sources
        // MIR: (column-join :var stackPosSt :sources ((StackDefUseDefUsed :handle 14 :prefix (eaLoad regBaseLoad stackPosLoad eaSt regBaseSt)) (ArchMemoryAccess :handle 15 :prefix (eaSt regBaseSt)) ))
        auto h_StackDefUseDefUsed_14_21 = ch_StackDefUseDefUsed_12_regBaseSt;
        auto h_ArchMemoryAccess_15_22 = ch_ArchMemoryAccess_13_regBaseSt;
        auto intersect_23 = intersect_handles(tile, h_StackDefUseDefUsed_14_21.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_15_22.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_24 = intersect_23.begin(); it_24.valid(); it_24.next()) {
          auto stackPosSt = it_24.value();
          auto positions = it_24.positions();
          auto ch_StackDefUseDefUsed_14_stackPosSt = h_StackDefUseDefUsed_14_21.child_range(positions[0], stackPosSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_15_stackPosSt = h_ArchMemoryAccess_15_22.child_range(positions[1], stackPosSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'reg1' from 2 sources
        // MIR: (column-join :var reg1 :sources ((ArchMemoryAccess :handle 16 :prefix (eaSt regBaseSt stackPosSt)) (RegDefUseDefUsed :handle 17 :prefix (eaSt)) ))
        auto h_ArchMemoryAccess_16_17 = ch_ArchMemoryAccess_15_stackPosSt;
        auto h_RegDefUseDefUsed_17_18 = ch_RegDefUseDefUsed_11_eaSt;
        auto intersect_19 = intersect_handles(tile, h_ArchMemoryAccess_16_17.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_17_18.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto reg1 = it_20.value();
          auto positions = it_20.positions();
          auto ch_ArchMemoryAccess_16_reg1 = h_ArchMemoryAccess_16_17.child_range(positions[0], reg1, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_17_reg1 = h_RegDefUseDefUsed_17_18.child_range(positions[1], reg1, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested CartesianJoin: bind mtype, _c1, _c2, _c3, _c4, eaDef, _gen26 from 4 source(s)
        // MIR: (cartesian-join :vars (mtype _c1 _c2 _c3 _c4 eaDef _gen26) :sources ((DefUsedForAddress :handle 18 :prefix (eaLoad reg2)) (ArchMemoryAccess :handle 19 :prefix (eaLoad reg2 regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 20 :prefix (eaSt regBaseSt stackPosSt reg1)) (RegDefUseDefUsed :handle 21 :prefix (eaSt reg1)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_DefUsedForAddress_18_4 = ch_DefUsedForAddress_3_reg2;  // reusing narrowed handle
        auto h_ArchMemoryAccess_19_6 = ch_ArchMemoryAccess_7_stackPosLoad;  // reusing narrowed handle
        auto h_ArchMemoryAccess_20_8 = ch_ArchMemoryAccess_16_reg1;  // reusing narrowed handle
        auto h_RegDefUseDefUsed_21_10 = ch_RegDefUseDefUsed_17_reg1;  // reusing narrowed handle

        if (!h_DefUsedForAddress_18_4.valid() || !h_ArchMemoryAccess_19_6.valid() || !h_ArchMemoryAccess_20_8.valid() || !h_RegDefUseDefUsed_21_10.valid()) continue;

        uint32_t degree_3 = h_DefUsedForAddress_18_4.degree();
        uint32_t degree_5 = h_ArchMemoryAccess_19_6.degree();
        uint32_t degree_7 = h_ArchMemoryAccess_20_8.degree();
        uint32_t degree_9 = h_RegDefUseDefUsed_21_10.degree();
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

          auto mtype = view_DefUsedForAddress_0_1_2_DELTA_VER.get_value(2, h_DefUsedForAddress_18_4.begin() + idx0_13);
          auto _c1 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c2 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c3 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto _c4 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto eaDef = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(2, h_RegDefUseDefUsed_21_10.begin() + idx3_16);
          auto _gen26 = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(3, h_RegDefUseDefUsed_21_10.begin() + idx3_16);

        if (_c1 == 0 && _c2 == 4) {
        if (_c3 == 2309374 && _c4 == 4) {
        // Emit: DefUsedForAddress(eaDef, reg1, mtype)
        output_ctx_0.emit_direct(eaDef, reg1, mtype);
        }
        }
        }
        }
        }
        }
          }
        }
        }
        }
        }
        }
  }

  // Fused kernel: single-pass join with atomic output (tail mode)
  static __global__ void __launch_bounds__(kBlockSize) kernel_fused(
      const ViewType* __restrict__ views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,
      ValueType* __restrict__ output_data_0,
      std::size_t output_stride_0,
      uint32_t old_size_0,
      uint32_t* __restrict__ atomic_write_pos_0,
      uint32_t capacity,
      uint32_t* __restrict__ overflow_flag) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);
    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];
    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);
    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }
    __syncthreads();
    views = s_views;
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;

    using SpecCtx_0 = SRDatalog::GPU::JIT::WS::SpeculativeOutputContext<ValueType, OutputArity_0, 16>;
    SpecCtx_0 output_ctx_0{output_data_0, atomic_write_pos_0, overflow_flag,
                         static_cast<uint32_t>(output_stride_0), old_size_0, capacity};

        using ViewType = std::remove_cvref_t<decltype(views[0])>;
        using HandleType = ViewType::NodeHandle;

        // View declarations (deduplicated by spec, 5 unique views)
        auto view_DefUsedForAddress_0_1_2_DELTA_VER = views[0];
        auto view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER = views[1];
        auto view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER = views[2];
        auto view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER = views[3];
        auto view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4];

        // Root ColumnJoin (multi-source intersection): bind 'eaLoad' from 3 sources
        // Uses root_unique_values + prefix() pattern (like TMP)
        // MIR: (column-join :var eaLoad :sources ((DefUsedForAddress :handle 0) (ArchMemoryAccess :handle 1) (StackDefUseDefUsed :handle 2) ))
        // WARP MODE: 32 threads cooperatively handle one row
        for (uint32_t y_idx_1 = warp_id; y_idx_1 < num_unique_root_keys; y_idx_1 += num_warps) {
          auto root_val_2 = root_unique_values[y_idx_1];

          uint32_t hint_lo_3 = y_idx_1;
          uint32_t hint_hi_4 = view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_ - (num_unique_root_keys - y_idx_1 - 1);
          hint_hi_4 = (hint_hi_4 <= view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          hint_hi_4 = (hint_hi_4 > hint_lo_3) ? hint_hi_4 : view_DefUsedForAddress_0_1_2_DELTA_VER.num_rows_;
          auto h_DefUsedForAddress_0_root = HandleType(hint_lo_3, hint_hi_4, 0).prefix(root_val_2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          if (!h_DefUsedForAddress_0_root.valid()) continue;
          auto h_ArchMemoryAccess_1_root = HandleType(0, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          if (!h_ArchMemoryAccess_1_root.valid()) continue;
          auto h_StackDefUseDefUsed_2_root = HandleType(0, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER.num_rows_, 0).prefix(root_val_2, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          if (!h_StackDefUseDefUsed_2_root.valid()) continue;
          auto eaLoad = root_val_2;
        // Nested ColumnJoin (intersection): bind 'reg2' from 2 sources
        // MIR: (column-join :var reg2 :sources ((DefUsedForAddress :handle 3 :prefix (eaLoad)) (ArchMemoryAccess :handle 4 :prefix (eaLoad)) ))
        auto h_DefUsedForAddress_3_42 = h_DefUsedForAddress_0_root;
        auto h_ArchMemoryAccess_4_43 = h_ArchMemoryAccess_1_root;
        auto intersect_44 = intersect_handles(tile, h_DefUsedForAddress_3_42.iterators(view_DefUsedForAddress_0_1_2_DELTA_VER), h_ArchMemoryAccess_4_43.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER));
        for (auto it_45 = intersect_44.begin(); it_45.valid(); it_45.next()) {
          auto reg2 = it_45.value();
          auto positions = it_45.positions();
          auto ch_DefUsedForAddress_3_reg2 = h_DefUsedForAddress_3_42.child_range(positions[0], reg2, tile, view_DefUsedForAddress_0_1_2_DELTA_VER);
          auto ch_ArchMemoryAccess_4_reg2 = h_ArchMemoryAccess_4_43.child_range(positions[1], reg2, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseLoad' from 2 sources
        // MIR: (column-join :var regBaseLoad :sources ((ArchMemoryAccess :handle 5 :prefix (eaLoad reg2)) (StackDefUseDefUsed :handle 6 :prefix (eaLoad)) ))
        auto h_ArchMemoryAccess_5_38 = ch_ArchMemoryAccess_4_reg2;
        auto h_StackDefUseDefUsed_6_39 = h_StackDefUseDefUsed_2_root;
        auto intersect_40 = intersect_handles(tile, h_ArchMemoryAccess_5_38.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_6_39.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_41 = intersect_40.begin(); it_41.valid(); it_41.next()) {
          auto regBaseLoad = it_41.value();
          auto positions = it_41.positions();
          auto ch_ArchMemoryAccess_5_regBaseLoad = h_ArchMemoryAccess_5_38.child_range(positions[0], regBaseLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_6_regBaseLoad = h_StackDefUseDefUsed_6_39.child_range(positions[1], regBaseLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosLoad' from 2 sources
        // MIR: (column-join :var stackPosLoad :sources ((ArchMemoryAccess :handle 7 :prefix (eaLoad reg2 regBaseLoad)) (StackDefUseDefUsed :handle 8 :prefix (eaLoad regBaseLoad)) ))
        auto h_ArchMemoryAccess_7_34 = ch_ArchMemoryAccess_5_regBaseLoad;
        auto h_StackDefUseDefUsed_8_35 = ch_StackDefUseDefUsed_6_regBaseLoad;
        auto intersect_36 = intersect_handles(tile, h_ArchMemoryAccess_7_34.iterators(view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER), h_StackDefUseDefUsed_8_35.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER));
        for (auto it_37 = intersect_36.begin(); it_37.valid(); it_37.next()) {
          auto stackPosLoad = it_37.value();
          auto positions = it_37.positions();
          auto ch_ArchMemoryAccess_7_stackPosLoad = h_ArchMemoryAccess_7_34.child_range(positions[0], stackPosLoad, tile, view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER);
          auto ch_StackDefUseDefUsed_8_stackPosLoad = h_StackDefUseDefUsed_8_35.child_range(positions[1], stackPosLoad, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'eaSt' from 3 sources
        // MIR: (column-join :var eaSt :sources ((StackDefUseDefUsed :handle 9 :prefix (eaLoad regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 10 :prefix ()) (RegDefUseDefUsed :handle 11 :prefix ()) ))
        for (int _nseg_2 = 0; _nseg_2 < 2; _nseg_2++) {
          view_RegDefUseDefUsed_2_1_0_3_FULL_VER = views[4 + _nseg_2];
          auto h_StackDefUseDefUsed_9_29 = ch_StackDefUseDefUsed_8_stackPosLoad;
          auto h_ArchMemoryAccess_10_30 = HandleType(0, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.num_rows_, 0);
          auto h_RegDefUseDefUsed_11_31 = HandleType(0, view_RegDefUseDefUsed_2_1_0_3_FULL_VER.num_rows_, 0);
          auto intersect_32 = intersect_handles(tile, h_StackDefUseDefUsed_9_29.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_10_30.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_11_31.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
          for (auto it_33 = intersect_32.begin(); it_33.valid(); it_33.next()) {
          auto eaSt = it_33.value();
          auto positions = it_33.positions();
          auto ch_StackDefUseDefUsed_9_eaSt = h_StackDefUseDefUsed_9_29.child_range(positions[0], eaSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_10_eaSt = h_ArchMemoryAccess_10_30.child_range(positions[1], eaSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_11_eaSt = h_RegDefUseDefUsed_11_31.child_range(positions[2], eaSt, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'regBaseSt' from 2 sources
        // MIR: (column-join :var regBaseSt :sources ((StackDefUseDefUsed :handle 12 :prefix (eaLoad regBaseLoad stackPosLoad eaSt)) (ArchMemoryAccess :handle 13 :prefix (eaSt)) ))
        auto h_StackDefUseDefUsed_12_25 = ch_StackDefUseDefUsed_9_eaSt;
        auto h_ArchMemoryAccess_13_26 = ch_ArchMemoryAccess_10_eaSt;
        auto intersect_27 = intersect_handles(tile, h_StackDefUseDefUsed_12_25.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_13_26.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_28 = intersect_27.begin(); it_28.valid(); it_28.next()) {
          auto regBaseSt = it_28.value();
          auto positions = it_28.positions();
          auto ch_StackDefUseDefUsed_12_regBaseSt = h_StackDefUseDefUsed_12_25.child_range(positions[0], regBaseSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_13_regBaseSt = h_ArchMemoryAccess_13_26.child_range(positions[1], regBaseSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'stackPosSt' from 2 sources
        // MIR: (column-join :var stackPosSt :sources ((StackDefUseDefUsed :handle 14 :prefix (eaLoad regBaseLoad stackPosLoad eaSt regBaseSt)) (ArchMemoryAccess :handle 15 :prefix (eaSt regBaseSt)) ))
        auto h_StackDefUseDefUsed_14_21 = ch_StackDefUseDefUsed_12_regBaseSt;
        auto h_ArchMemoryAccess_15_22 = ch_ArchMemoryAccess_13_regBaseSt;
        auto intersect_23 = intersect_handles(tile, h_StackDefUseDefUsed_14_21.iterators(view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER), h_ArchMemoryAccess_15_22.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER));
        for (auto it_24 = intersect_23.begin(); it_24.valid(); it_24.next()) {
          auto stackPosSt = it_24.value();
          auto positions = it_24.positions();
          auto ch_StackDefUseDefUsed_14_stackPosSt = h_StackDefUseDefUsed_14_21.child_range(positions[0], stackPosSt, tile, view_StackDefUseDefUsed_3_4_5_0_1_2_FULL_VER);
          auto ch_ArchMemoryAccess_15_stackPosSt = h_ArchMemoryAccess_15_22.child_range(positions[1], stackPosSt, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
        // Nested ColumnJoin (intersection): bind 'reg1' from 2 sources
        // MIR: (column-join :var reg1 :sources ((ArchMemoryAccess :handle 16 :prefix (eaSt regBaseSt stackPosSt)) (RegDefUseDefUsed :handle 17 :prefix (eaSt)) ))
        auto h_ArchMemoryAccess_16_17 = ch_ArchMemoryAccess_15_stackPosSt;
        auto h_RegDefUseDefUsed_17_18 = ch_RegDefUseDefUsed_11_eaSt;
        auto intersect_19 = intersect_handles(tile, h_ArchMemoryAccess_16_17.iterators(view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER), h_RegDefUseDefUsed_17_18.iterators(view_RegDefUseDefUsed_2_1_0_3_FULL_VER));
        for (auto it_20 = intersect_19.begin(); it_20.valid(); it_20.next()) {
          auto reg1 = it_20.value();
          auto positions = it_20.positions();
          auto ch_ArchMemoryAccess_16_reg1 = h_ArchMemoryAccess_16_17.child_range(positions[0], reg1, tile, view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER);
          auto ch_RegDefUseDefUsed_17_reg1 = h_RegDefUseDefUsed_17_18.child_range(positions[1], reg1, tile, view_RegDefUseDefUsed_2_1_0_3_FULL_VER);
        // Nested CartesianJoin: bind mtype, _c1, _c2, _c3, _c4, eaDef, _gen26 from 4 source(s)
        // MIR: (cartesian-join :vars (mtype _c1 _c2 _c3 _c4 eaDef _gen26) :sources ((DefUsedForAddress :handle 18 :prefix (eaLoad reg2)) (ArchMemoryAccess :handle 19 :prefix (eaLoad reg2 regBaseLoad stackPosLoad)) (ArchMemoryAccess :handle 20 :prefix (eaSt regBaseSt stackPosSt reg1)) (RegDefUseDefUsed :handle 21 :prefix (eaSt reg1)) ))
        uint32_t lane_1 = tile.thread_rank();
        uint32_t group_size_2 = tile.size();

        auto h_DefUsedForAddress_18_4 = ch_DefUsedForAddress_3_reg2;  // reusing narrowed handle
        auto h_ArchMemoryAccess_19_6 = ch_ArchMemoryAccess_7_stackPosLoad;  // reusing narrowed handle
        auto h_ArchMemoryAccess_20_8 = ch_ArchMemoryAccess_16_reg1;  // reusing narrowed handle
        auto h_RegDefUseDefUsed_21_10 = ch_RegDefUseDefUsed_17_reg1;  // reusing narrowed handle

        if (!h_DefUsedForAddress_18_4.valid() || !h_ArchMemoryAccess_19_6.valid() || !h_ArchMemoryAccess_20_8.valid() || !h_RegDefUseDefUsed_21_10.valid()) continue;

        uint32_t degree_3 = h_DefUsedForAddress_18_4.degree();
        uint32_t degree_5 = h_ArchMemoryAccess_19_6.degree();
        uint32_t degree_7 = h_ArchMemoryAccess_20_8.degree();
        uint32_t degree_9 = h_RegDefUseDefUsed_21_10.degree();
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

          auto mtype = view_DefUsedForAddress_0_1_2_DELTA_VER.get_value(2, h_DefUsedForAddress_18_4.begin() + idx0_13);
          auto _c1 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c2 = view_ArchMemoryAccess_1_2_3_5_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_19_6.begin() + idx1_14);
          auto _c3 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(4, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto _c4 = view_ArchMemoryAccess_1_3_5_2_0_4_FULL_VER.get_value(5, h_ArchMemoryAccess_20_8.begin() + idx2_15);
          auto eaDef = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(2, h_RegDefUseDefUsed_21_10.begin() + idx3_16);
          auto _gen26 = view_RegDefUseDefUsed_2_1_0_3_FULL_VER.get_value(3, h_RegDefUseDefUsed_21_10.begin() + idx3_16);

        if (_c1 == 0 && _c2 == 4) {
        if (_c3 == 2309374 && _c4 == 4) {
        // Emit: DefUsedForAddress(eaDef, reg1, mtype)
        output_ctx_0.emit_direct(eaDef, reg1, mtype);
        }
        }
        }
        }
        }
        }
          }
        }
        }
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
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  // Non-template execute - calls kernels directly
  static void execute(DB& db, uint32_t iteration);
  static void execute_fused(DB& db, uint32_t iteration);
  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void read_fused_result(LaunchParams& p);
  static inline uint32_t prev_fused_total_ = 4096;
};

// Phase 1: Setup views and compute grid config
JitRunner_DefUsedAddr4_D0::LaunchParams JitRunner_DefUsedAddr4_D0::setup(DB& db, uint32_t iteration, GPU_STREAM_T stream) {
  LaunchParams p;
  p.views_vec.reserve(NumSources);

  // Source 0: DefUsedForAddress version DELTA_VER
  {
    auto& rel_0 = (iteration == 0) ? get_relation_by_schema<DefUsedForAddress, FULL_VER>(db) : get_relation_by_schema<DefUsedForAddress, DELTA_VER>(db);
    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{0, 1, 2}}, false);
    p.views_vec.push_back(idx_0.view());
  }

  // Source 1: ArchMemoryAccess version FULL_VER
  {
    auto& rel_1 = get_relation_by_schema<ArchMemoryAccess, FULL_VER>(db);
    auto& idx_1 = rel_1.ensure_index(SRDatalog::IndexSpec{{1, 2, 3, 5, 0, 4}}, false);
    p.views_vec.push_back(idx_1.view());
  }

  // Source 2: StackDefUseDefUsed version FULL_VER
  {
    auto& rel_2 = get_relation_by_schema<StackDefUseDefUsed, FULL_VER>(db);
    auto& idx_2 = rel_2.ensure_index(SRDatalog::IndexSpec{{3, 4, 5, 0, 1, 2}}, false);
    p.views_vec.push_back(idx_2.view());
  }

  // Source 10: ArchMemoryAccess version FULL_VER
  {
    auto& rel_10 = get_relation_by_schema<ArchMemoryAccess, FULL_VER>(db);
    auto& idx_10 = rel_10.ensure_index(SRDatalog::IndexSpec{{1, 3, 5, 2, 0, 4}}, false);
    p.views_vec.push_back(idx_10.view());
  }

  // Source 11: RegDefUseDefUsed version FULL_VER
  {
    auto& rel_11 = get_relation_by_schema<RegDefUseDefUsed, FULL_VER>(db);
    auto& idx_11 = rel_11.ensure_index(SRDatalog::IndexSpec{{2, 1, 0, 3}}, false);
    p.views_vec.push_back(idx_11.full_view());
    p.views_vec.push_back(idx_11.head_view());
  }

  // First source for root keys
  auto& first_rel = (iteration == 0) ? get_relation_by_schema<DefUsedForAddress, FULL_VER>(db) : get_relation_by_schema<DefUsedForAddress, DELTA_VER>(db);
  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{{0, 1, 2}});
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
  return p;
}

void JitRunner_DefUsedAddr4_D0::launch_count(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0) return;
  if (p.num_unique_root_keys == 0) { cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream); return; }
  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>(p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys, p.thread_counts_ptr);
}

// Phase 3: Prefix scan + readback total + resize destinations
uint32_t JitRunner_DefUsedAddr4_D0::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0, thrust::plus<uint32_t>());
  uint32_t total_count = 0;
  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t), GPU_DEVICE_TO_HOST, stream);
  GPU_STREAM_SYNCHRONIZE(stream);
  if (total_count == 0) return 0;

  auto& dest_rel_0 = get_relation_by_schema<DefUsedForAddress, NEW_VER>(db);
  p.old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  dest_rel_0.resize_interned_columns(p.old_size_0 + total_count, stream);
  return total_count;
}

// Phase 3a: Prefix scan only (async, no sync)
void JitRunner_DefUsedAddr4_D0::scan_only(LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_threads == 0) return;
  thrust::exclusive_scan(rmm::exec_policy(stream), p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, p.thread_counts_ptr, 0, thrust::plus<uint32_t>());
}

// Phase 3b: Read total count (call after device sync)
uint32_t JitRunner_DefUsedAddr4_D0::read_total(LaunchParams& p) {
  if (p.num_threads == 0) return 0;
  uint32_t total_count = 0;
  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  return total_count;
}

// Phase 4: Launch materialize kernel on given stream (no sync)
void JitRunner_DefUsedAddr4_D0::launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream) {
  using ProvPtrType = semiring_value_t<SR>*;
  ProvPtrType prov_ptr = nullptr;

  auto& dest_rel_0 = get_relation_by_schema<DefUsedForAddress, NEW_VER>(db);
  uint32_t old_size_0 = p.old_size_0;
  kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      p.thread_counts_ptr,
      dest_rel_0.template interned_column<0>(), prov_ptr, dest_rel_0.interned_stride(), old_size_0);
}

// launch_fused: launch fused kernel on given stream (no sync)
void JitRunner_DefUsedAddr4_D0::launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream) {
  if (p.num_unique_root_keys == 0) return;

  auto& dest_rel_0 = get_relation_by_schema<DefUsedForAddress, NEW_VER>(db);
  kernel_fused<<<p.num_blocks, kBlockSize, 0, stream>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      dest_rel_0.template interned_column<0>(), dest_rel_0.interned_stride(), p.old_size_0, p.fused_wp_ptr_0,
      p.fused_capacity, p.fused_of_ptr);
}

// read_fused_result: readback fused write counts (call after device sync)
void JitRunner_DefUsedAddr4_D0::read_fused_result(LaunchParams& p) {
  GPU_MEMCPY(&p.fused_h_wp_0, p.fused_wp_ptr_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  uint32_t h_of = 0;
  GPU_MEMCPY(&h_of, p.fused_of_ptr, sizeof(uint32_t), GPU_DEVICE_TO_HOST);
  p.fused_overflow = (h_of != 0);
}

// Execute definition - calls decomposed phases sequentially
void JitRunner_DefUsedAddr4_D0::execute(DB& db, uint32_t iteration) {
  nvtxRangePushA("DefUsedAddr4_D0");

  auto p = setup(db, iteration);
  launch_count(p, 0);
  uint32_t total_count = scan_and_resize(db, p, 0);
  if (total_count == 0) { nvtxRangePop(); return; }

  launch_materialize(db, p, total_count, 0);
  nvtxRangePop();
}

// Tail-mode fused execution: single kernel, no count/scan phase
void JitRunner_DefUsedAddr4_D0::execute_fused(DB& db, uint32_t iteration) {
  auto p = setup(db, iteration);
  if (p.num_unique_root_keys == 0) return;

  auto& dest_rel_0 = get_relation_by_schema<DefUsedForAddress, NEW_VER>(db);
  uint32_t old_size_0 = static_cast<uint32_t>(dest_rel_0.size());
  uint32_t capacity = prev_fused_total_;
  dest_rel_0.resize_interned_columns(old_size_0 + capacity);

  static SRDatalog::GPU::DeviceArray<uint32_t> s_wp_0(1);
  cudaMemsetAsync(s_wp_0.data(), 0, sizeof(uint32_t), 0);
  static SRDatalog::GPU::DeviceArray<uint32_t> s_of(1);
  cudaMemsetAsync(s_of.data(), 0, sizeof(uint32_t), 0);
  kernel_fused<<<p.num_blocks, kBlockSize>>>(
      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,
      dest_rel_0.template interned_column<0>(), dest_rel_0.interned_stride(), old_size_0, s_wp_0.data(),
      capacity, s_of.data());
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
