template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using Assign_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using Deref_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(2);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<AssignInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<DerefInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_LoadAssign::execute_fused(db, 0);
    JitRunner_LoadDeref::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_LoadAssign::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_LoadDeref::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_LoadAssign::launch_count(p_0, _stream_pool.get(0));
      JitRunner_LoadDeref::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_LoadAssign::scan_only(p_0, _stream_pool.get(0));
      JitRunner_LoadDeref::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_LoadAssign::read_total(p_0);
      uint32_t total_1 = JitRunner_LoadDeref::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize Assign
        uint32_t sum_Assign = total_0;
        auto& dest_Assign = get_relation_by_schema<Assign, NEW_VER>(db);
        uint32_t base_offset_Assign = static_cast<uint32_t>(dest_Assign.size());
        if (sum_Assign > 0)
          dest_Assign.resize_interned_columns(base_offset_Assign + sum_Assign);
        uint32_t running_offset_Assign = base_offset_Assign;
        p_0.old_size_0 = running_offset_Assign;
        running_offset_Assign += total_0;
      }
      {  // Resize Deref
        uint32_t sum_Deref = total_1;
        auto& dest_Deref = get_relation_by_schema<Deref, NEW_VER>(db);
        uint32_t base_offset_Deref = static_cast<uint32_t>(dest_Deref.size());
        if (sum_Deref > 0)
          dest_Deref.resize_interned_columns(base_offset_Deref + sum_Deref);
        uint32_t running_offset_Deref = base_offset_Deref;
        p_1.old_size_0 = running_offset_Deref;
        running_offset_Deref += total_1;
      }

      if (total_0 > 0)
        JitRunner_LoadAssign::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_LoadDeref::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Assign, NEW_VER, Assign_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Assign, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Deref, NEW_VER, Deref_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Deref, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using ValueFlow_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using MemoryAlias_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(5);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_VF_Assign::execute_fused(db, 0);
    JitRunner_VF_Refl1::execute_fused(db, 0);
    JitRunner_VF_Refl2::execute_fused(db, 0);
    JitRunner_MA_Refl1::execute_fused(db, 0);
    JitRunner_MA_Refl2::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 5 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_VF_Assign::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_VF_Refl1::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_VF_Refl2::setup(db, 0, _stream_pool.get(2));
      auto p_3 = JitRunner_MA_Refl1::setup(db, 0, _stream_pool.get(3));
      auto p_4 = JitRunner_MA_Refl2::setup(db, 0, _stream_pool.get(4));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_MemoryAlias_size = p_3.num_threads + p_4.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_MemoryAlias(shared_buf_MemoryAlias_size);
      uint32_t off_MemoryAlias_0 = 0;
      p_3.thread_counts_ptr = shared_buf_MemoryAlias.data() + off_MemoryAlias_0;
      uint32_t off_MemoryAlias_1 = off_MemoryAlias_0 + p_3.num_threads;
      p_4.thread_counts_ptr = shared_buf_MemoryAlias.data() + off_MemoryAlias_1;
      uint32_t shared_buf_ValueFlow_size = p_0.num_threads + p_1.num_threads + p_2.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_ValueFlow(shared_buf_ValueFlow_size);
      uint32_t off_ValueFlow_0 = 0;
      p_0.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_0;
      uint32_t off_ValueFlow_1 = off_ValueFlow_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_1;
      uint32_t off_ValueFlow_2 = off_ValueFlow_1 + p_1.num_threads;
      p_2.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_2;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_VF_Assign::launch_count(p_0, _stream_pool.get(0));
      JitRunner_VF_Refl1::launch_count(p_1, _stream_pool.get(1));
      JitRunner_VF_Refl2::launch_count(p_2, _stream_pool.get(2));
      JitRunner_MA_Refl1::launch_count(p_3, _stream_pool.get(3));
      JitRunner_MA_Refl2::launch_count(p_4, _stream_pool.get(4));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(3)), shared_buf_MemoryAlias.data(),
                             shared_buf_MemoryAlias.data() + shared_buf_MemoryAlias_size,
                             shared_buf_MemoryAlias.data(), 0, thrust::plus<uint32_t>());
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_ValueFlow.data(),
                             shared_buf_ValueFlow.data() + shared_buf_ValueFlow_size,
                             shared_buf_ValueFlow.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      // Read boundary values for MemoryAlias (2 rules)
      uint32_t bnd_MemoryAlias[3];
      GPU_MEMCPY_ASYNC(&bnd_MemoryAlias[0], shared_buf_MemoryAlias.data() + off_MemoryAlias_0,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_MemoryAlias[1], shared_buf_MemoryAlias.data() + off_MemoryAlias_1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_MemoryAlias[2],
                       shared_buf_MemoryAlias.data() + shared_buf_MemoryAlias_size - 1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_3 = bnd_MemoryAlias[1] - bnd_MemoryAlias[0];
      uint32_t total_4 = bnd_MemoryAlias[2] - bnd_MemoryAlias[1];
      // Read boundary values for ValueFlow (3 rules)
      uint32_t bnd_ValueFlow[4];
      GPU_MEMCPY_ASYNC(&bnd_ValueFlow[0], shared_buf_ValueFlow.data() + off_ValueFlow_0,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_ValueFlow[1], shared_buf_ValueFlow.data() + off_ValueFlow_1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_ValueFlow[2], shared_buf_ValueFlow.data() + off_ValueFlow_2,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_ValueFlow[3],
                       shared_buf_ValueFlow.data() + shared_buf_ValueFlow_size - 1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_ValueFlow[1] - bnd_ValueFlow[0];
      uint32_t total_1 = bnd_ValueFlow[2] - bnd_ValueFlow[1];
      uint32_t total_2 = bnd_ValueFlow[3] - bnd_ValueFlow[2];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize MemoryAlias
        uint32_t sum_MemoryAlias = total_3 + total_4;
        auto& dest_MemoryAlias = get_relation_by_schema<MemoryAlias, NEW_VER>(db);
        uint32_t base_offset_MemoryAlias = static_cast<uint32_t>(dest_MemoryAlias.size());
        if (sum_MemoryAlias > 0)
          dest_MemoryAlias.resize_interned_columns(base_offset_MemoryAlias + sum_MemoryAlias);
        p_3.old_size_0 = base_offset_MemoryAlias;
        p_4.old_size_0 = base_offset_MemoryAlias;
      }
      {  // Resize ValueFlow
        uint32_t sum_ValueFlow = total_0 + total_1 + total_2;
        auto& dest_ValueFlow = get_relation_by_schema<ValueFlow, NEW_VER>(db);
        uint32_t base_offset_ValueFlow = static_cast<uint32_t>(dest_ValueFlow.size());
        if (sum_ValueFlow > 0)
          dest_ValueFlow.resize_interned_columns(base_offset_ValueFlow + sum_ValueFlow);
        p_0.old_size_0 = base_offset_ValueFlow;
        p_1.old_size_0 = base_offset_ValueFlow;
        p_2.old_size_0 = base_offset_ValueFlow;
      }

      if (total_0 > 0)
        JitRunner_VF_Assign::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_VF_Refl1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0)
        JitRunner_VF_Refl2::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      if (total_3 > 0)
        JitRunner_MA_Refl1::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
      if (total_4 > 0)
        JitRunner_MA_Refl2::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      _stream_pool.record_event(3);
      _stream_pool.record_event(4);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<ValueFlow, NEW_VER, ValueFlow_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueFlow, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<ValueFlow, NEW_VER, ValueFlow_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueFlow, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<ValueFlow, NEW_VER, ValueFlow_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueFlow, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<MemoryAlias, NEW_VER, MemoryAlias_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<MemoryAlias, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<MemoryAlias, NEW_VER, MemoryAlias_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<MemoryAlias, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_4(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_5(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_6(DB& db, std::size_t max_iterations) {
  using MemoryAlias_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using ValueAlias_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using ValueFlow_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Deref, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<MemoryAlias, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 1}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      {
        auto& delta_rel = get_relation_by_schema<ValueAlias, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 1}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      {
        auto& delta_rel = get_relation_by_schema<ValueFlow, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 1}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<MemoryAlias, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " MemoryAlias=" << _di.root().degree();
          } catch (...) {
            std::cerr << " MemoryAlias=?";
          }
        }
        {
          auto& _dr = get_relation_by_schema<ValueAlias, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " ValueAlias=" << _di.root().degree();
          } catch (...) {
            std::cerr << " ValueAlias=?";
          }
        }
        {
          auto& _dr = get_relation_by_schema<ValueFlow, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " ValueFlow=" << _di.root().degree();
          } catch (...) {
            std::cerr << " ValueFlow=?";
          }
        }
        std::cerr << std::endl;
      } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
        std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
      }
      if (total_new_facts == 0)
        break;
      _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
    }

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(9);

    if (_tail_mode) {
      JitRunner_VF_Trans_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VF_Trans_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VF_Assign_MA_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VA_VF_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VA_VF_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VA_VF_MA_VF_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VA_VF_MA_VF_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_VA_VF_MA_VF_D2::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_MA_Deref_VA_D1::execute_fused(db, static_cast<uint32_t>(iter));
    } else {
      // === ParallelGroup (stream-parallel, 9 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 =
            JitRunner_VF_Trans_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 =
            JitRunner_VF_Trans_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));
        auto p_2 =
            JitRunner_VF_Assign_MA_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(2));
        auto p_3 = JitRunner_VA_VF_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(3));
        auto p_4 = JitRunner_VA_VF_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(4));
        auto p_5 =
            JitRunner_VA_VF_MA_VF_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(5));
        auto p_6 =
            JitRunner_VA_VF_MA_VF_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(6));
        auto p_7 =
            JitRunner_VA_VF_MA_VF_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(7));
        auto p_8 =
            JitRunner_MA_Deref_VA_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(8));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_ValueAlias_size = p_3.num_threads + p_4.num_threads + p_5.num_threads +
                                              p_6.num_threads + p_7.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_ValueAlias(shared_buf_ValueAlias_size);
        uint32_t off_ValueAlias_0 = 0;
        p_3.thread_counts_ptr = shared_buf_ValueAlias.data() + off_ValueAlias_0;
        uint32_t off_ValueAlias_1 = off_ValueAlias_0 + p_3.num_threads;
        p_4.thread_counts_ptr = shared_buf_ValueAlias.data() + off_ValueAlias_1;
        uint32_t off_ValueAlias_2 = off_ValueAlias_1 + p_4.num_threads;
        p_5.thread_counts_ptr = shared_buf_ValueAlias.data() + off_ValueAlias_2;
        uint32_t off_ValueAlias_3 = off_ValueAlias_2 + p_5.num_threads;
        p_6.thread_counts_ptr = shared_buf_ValueAlias.data() + off_ValueAlias_3;
        uint32_t off_ValueAlias_4 = off_ValueAlias_3 + p_6.num_threads;
        p_7.thread_counts_ptr = shared_buf_ValueAlias.data() + off_ValueAlias_4;
        uint32_t shared_buf_ValueFlow_size =
            p_0.num_threads + p_1.num_threads + p_2.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_ValueFlow(shared_buf_ValueFlow_size);
        uint32_t off_ValueFlow_0 = 0;
        p_0.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_0;
        uint32_t off_ValueFlow_1 = off_ValueFlow_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_1;
        uint32_t off_ValueFlow_2 = off_ValueFlow_1 + p_1.num_threads;
        p_2.thread_counts_ptr = shared_buf_ValueFlow.data() + off_ValueFlow_2;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_VF_Trans_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_VF_Trans_D1::launch_count(p_1, _stream_pool.get(1));
        JitRunner_VF_Assign_MA_D1::launch_count(p_2, _stream_pool.get(2));
        JitRunner_VA_VF_D0::launch_count(p_3, _stream_pool.get(3));
        JitRunner_VA_VF_D1::launch_count(p_4, _stream_pool.get(4));
        JitRunner_VA_VF_MA_VF_D0::launch_count(p_5, _stream_pool.get(5));
        JitRunner_VA_VF_MA_VF_D1::launch_count(p_6, _stream_pool.get(6));
        JitRunner_VA_VF_MA_VF_D2::launch_count(p_7, _stream_pool.get(7));
        JitRunner_MA_Deref_VA_D1::launch_count(p_8, _stream_pool.get(8));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        JitRunner_MA_Deref_VA_D1::scan_only(p_8, _stream_pool.get(8));
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(3)), shared_buf_ValueAlias.data(),
                               shared_buf_ValueAlias.data() + shared_buf_ValueAlias_size,
                               shared_buf_ValueAlias.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_ValueFlow.data(),
                               shared_buf_ValueFlow.data() + shared_buf_ValueFlow_size,
                               shared_buf_ValueFlow.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        uint32_t total_8 = JitRunner_MA_Deref_VA_D1::read_total(p_8);
        // Read boundary values for ValueAlias (5 rules)
        uint32_t bnd_ValueAlias[6];
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[0], shared_buf_ValueAlias.data() + off_ValueAlias_0,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[1], shared_buf_ValueAlias.data() + off_ValueAlias_1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[2], shared_buf_ValueAlias.data() + off_ValueAlias_2,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[3], shared_buf_ValueAlias.data() + off_ValueAlias_3,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[4], shared_buf_ValueAlias.data() + off_ValueAlias_4,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueAlias[5],
                         shared_buf_ValueAlias.data() + shared_buf_ValueAlias_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_3 = bnd_ValueAlias[1] - bnd_ValueAlias[0];
        uint32_t total_4 = bnd_ValueAlias[2] - bnd_ValueAlias[1];
        uint32_t total_5 = bnd_ValueAlias[3] - bnd_ValueAlias[2];
        uint32_t total_6 = bnd_ValueAlias[4] - bnd_ValueAlias[3];
        uint32_t total_7 = bnd_ValueAlias[5] - bnd_ValueAlias[4];
        // Read boundary values for ValueFlow (3 rules)
        uint32_t bnd_ValueFlow[4];
        GPU_MEMCPY_ASYNC(&bnd_ValueFlow[0], shared_buf_ValueFlow.data() + off_ValueFlow_0,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueFlow[1], shared_buf_ValueFlow.data() + off_ValueFlow_1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueFlow[2], shared_buf_ValueFlow.data() + off_ValueFlow_2,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ValueFlow[3],
                         shared_buf_ValueFlow.data() + shared_buf_ValueFlow_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_ValueFlow[1] - bnd_ValueFlow[0];
        uint32_t total_1 = bnd_ValueFlow[2] - bnd_ValueFlow[1];
        uint32_t total_2 = bnd_ValueFlow[3] - bnd_ValueFlow[2];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize MemoryAlias
          uint32_t sum_MemoryAlias = total_8;
          auto& dest_MemoryAlias = get_relation_by_schema<MemoryAlias, NEW_VER>(db);
          uint32_t base_offset_MemoryAlias = static_cast<uint32_t>(dest_MemoryAlias.size());
          if (sum_MemoryAlias > 0)
            dest_MemoryAlias.resize_interned_columns(base_offset_MemoryAlias + sum_MemoryAlias);
          uint32_t running_offset_MemoryAlias = base_offset_MemoryAlias;
          p_8.old_size_0 = running_offset_MemoryAlias;
          running_offset_MemoryAlias += total_8;
        }
        {  // Resize ValueAlias
          uint32_t sum_ValueAlias = total_3 + total_4 + total_5 + total_6 + total_7;
          auto& dest_ValueAlias = get_relation_by_schema<ValueAlias, NEW_VER>(db);
          uint32_t base_offset_ValueAlias = static_cast<uint32_t>(dest_ValueAlias.size());
          if (sum_ValueAlias > 0)
            dest_ValueAlias.resize_interned_columns(base_offset_ValueAlias + sum_ValueAlias);
          p_3.old_size_0 = base_offset_ValueAlias;
          p_4.old_size_0 = base_offset_ValueAlias;
          p_5.old_size_0 = base_offset_ValueAlias;
          p_6.old_size_0 = base_offset_ValueAlias;
          p_7.old_size_0 = base_offset_ValueAlias;
        }
        {  // Resize ValueFlow
          uint32_t sum_ValueFlow = total_0 + total_1 + total_2;
          auto& dest_ValueFlow = get_relation_by_schema<ValueFlow, NEW_VER>(db);
          uint32_t base_offset_ValueFlow = static_cast<uint32_t>(dest_ValueFlow.size());
          if (sum_ValueFlow > 0)
            dest_ValueFlow.resize_interned_columns(base_offset_ValueFlow + sum_ValueFlow);
          p_0.old_size_0 = base_offset_ValueFlow;
          p_1.old_size_0 = base_offset_ValueFlow;
          p_2.old_size_0 = base_offset_ValueFlow;
        }

        if (total_0 > 0)
          JitRunner_VF_Trans_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_VF_Trans_D1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        if (total_2 > 0)
          JitRunner_VF_Assign_MA_D1::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
        if (total_3 > 0)
          JitRunner_VA_VF_D0::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
        if (total_4 > 0)
          JitRunner_VA_VF_D1::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
        if (total_5 > 0)
          JitRunner_VA_VF_MA_VF_D0::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
        if (total_6 > 0)
          JitRunner_VA_VF_MA_VF_D1::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
        if (total_7 > 0)
          JitRunner_VA_VF_MA_VF_D2::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
        if (total_8 > 0)
          JitRunner_MA_Deref_VA_D1::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        _stream_pool.record_event(2);
        _stream_pool.record_event(3);
        _stream_pool.record_event(4);
        _stream_pool.record_event(5);
        _stream_pool.record_event(6);
        _stream_pool.record_event(7);
        _stream_pool.record_event(8);
        nvtxRangePop();  // join_pipeline
      }
    }
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MemoryAlias, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MemoryAlias, NEW_VER, MemoryAlias_canonical_spec_t>(
        db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MemoryAlias, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueAlias, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ValueAlias, NEW_VER, ValueAlias_canonical_spec_t>(
        db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueAlias, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueFlow, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ValueFlow, NEW_VER, ValueFlow_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ValueFlow, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_7(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<MemoryAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_8(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ValueAlias, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_9(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ValueFlow, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
