template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using PointsTo_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<AddressOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_Base::execute_fused(db, 0);
  else
    JitRunner_Base::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<PointsTo, NEW_VER, PointsTo_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<PointsTo, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  using PointsTo_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Load, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Load, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Load, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Load, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Store, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Store, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Store, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Store, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<PointsTo, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<PointsTo, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " PointsTo=" << _di.root().degree();
          } catch (...) {
            std::cerr << " PointsTo=?";
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
    _stream_pool.ensure(5);

    if (_tail_mode) {
      JitRunner_Assign_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Load_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Load_D2::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Store_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Store_D2::execute_fused(db, static_cast<uint32_t>(iter));
    } else {
      // === ParallelGroup (stream-parallel, 5 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_Assign_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_Load_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));
        auto p_2 = JitRunner_Load_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(2));
        auto p_3 = JitRunner_Store_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(3));
        auto p_4 = JitRunner_Store_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(4));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_PointsTo_size = p_0.num_threads + p_1.num_threads + p_2.num_threads +
                                            p_3.num_threads + p_4.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_PointsTo(shared_buf_PointsTo_size);
        uint32_t off_PointsTo_0 = 0;
        p_0.thread_counts_ptr = shared_buf_PointsTo.data() + off_PointsTo_0;
        uint32_t off_PointsTo_1 = off_PointsTo_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_PointsTo.data() + off_PointsTo_1;
        uint32_t off_PointsTo_2 = off_PointsTo_1 + p_1.num_threads;
        p_2.thread_counts_ptr = shared_buf_PointsTo.data() + off_PointsTo_2;
        uint32_t off_PointsTo_3 = off_PointsTo_2 + p_2.num_threads;
        p_3.thread_counts_ptr = shared_buf_PointsTo.data() + off_PointsTo_3;
        uint32_t off_PointsTo_4 = off_PointsTo_3 + p_3.num_threads;
        p_4.thread_counts_ptr = shared_buf_PointsTo.data() + off_PointsTo_4;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_Assign_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_Load_D0::launch_count(p_1, _stream_pool.get(1));
        JitRunner_Load_D2::launch_count(p_2, _stream_pool.get(2));
        JitRunner_Store_D0::launch_count(p_3, _stream_pool.get(3));
        JitRunner_Store_D2::launch_count(p_4, _stream_pool.get(4));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_PointsTo.data(),
                               shared_buf_PointsTo.data() + shared_buf_PointsTo_size,
                               shared_buf_PointsTo.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for PointsTo (5 rules)
        uint32_t bnd_PointsTo[6];
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[0], shared_buf_PointsTo.data() + off_PointsTo_0,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[1], shared_buf_PointsTo.data() + off_PointsTo_1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[2], shared_buf_PointsTo.data() + off_PointsTo_2,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[3], shared_buf_PointsTo.data() + off_PointsTo_3,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[4], shared_buf_PointsTo.data() + off_PointsTo_4,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_PointsTo[5],
                         shared_buf_PointsTo.data() + shared_buf_PointsTo_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_PointsTo[1] - bnd_PointsTo[0];
        uint32_t total_1 = bnd_PointsTo[2] - bnd_PointsTo[1];
        uint32_t total_2 = bnd_PointsTo[3] - bnd_PointsTo[2];
        uint32_t total_3 = bnd_PointsTo[4] - bnd_PointsTo[3];
        uint32_t total_4 = bnd_PointsTo[5] - bnd_PointsTo[4];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize PointsTo
          uint32_t sum_PointsTo = total_0 + total_1 + total_2 + total_3 + total_4;
          auto& dest_PointsTo = get_relation_by_schema<PointsTo, NEW_VER>(db);
          uint32_t base_offset_PointsTo = static_cast<uint32_t>(dest_PointsTo.size());
          if (sum_PointsTo > 0)
            dest_PointsTo.resize_interned_columns(base_offset_PointsTo + sum_PointsTo);
          p_0.old_size_0 = base_offset_PointsTo;
          p_1.old_size_0 = base_offset_PointsTo;
          p_2.old_size_0 = base_offset_PointsTo;
          p_3.old_size_0 = base_offset_PointsTo;
          p_4.old_size_0 = base_offset_PointsTo;
        }

        if (total_0 > 0)
          JitRunner_Assign_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_Load_D0::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        if (total_2 > 0)
          JitRunner_Load_D2::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
        if (total_3 > 0)
          JitRunner_Store_D0::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
        if (total_4 > 0)
          JitRunner_Store_D2::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        _stream_pool.record_event(2);
        _stream_pool.record_event(3);
        _stream_pool.record_event(4);
        nvtxRangePop();  // join_pipeline
      }
    }
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<PointsTo, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<PointsTo, NEW_VER, PointsTo_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<PointsTo, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
