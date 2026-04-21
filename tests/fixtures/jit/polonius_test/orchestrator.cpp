template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);

  using cfg_node_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(2);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_cfg_node_from_edge_src::execute_fused(db, 0);
    JitRunner_cfg_node_from_edge_dst::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_cfg_node_from_edge_src::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_cfg_node_from_edge_dst::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_cfg_node_size = p_0.num_threads + p_1.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_cfg_node(shared_buf_cfg_node_size);
      uint32_t off_cfg_node_0 = 0;
      p_0.thread_counts_ptr = shared_buf_cfg_node.data() + off_cfg_node_0;
      uint32_t off_cfg_node_1 = off_cfg_node_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_cfg_node.data() + off_cfg_node_1;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_cfg_node_from_edge_src::launch_count(p_0, _stream_pool.get(0));
      JitRunner_cfg_node_from_edge_dst::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_cfg_node.data(),
                             shared_buf_cfg_node.data() + shared_buf_cfg_node_size,
                             shared_buf_cfg_node.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      // Read boundary values for cfg_node (2 rules)
      uint32_t bnd_cfg_node[3];
      GPU_MEMCPY_ASYNC(&bnd_cfg_node[0], shared_buf_cfg_node.data() + off_cfg_node_0,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_cfg_node[1], shared_buf_cfg_node.data() + off_cfg_node_1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_cfg_node[2], shared_buf_cfg_node.data() + shared_buf_cfg_node_size - 1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_cfg_node[1] - bnd_cfg_node[0];
      uint32_t total_1 = bnd_cfg_node[2] - bnd_cfg_node[1];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize cfg_node
        uint32_t sum_cfg_node = total_0 + total_1;
        auto& dest_cfg_node = get_relation_by_schema<cfg_node, NEW_VER>(db);
        uint32_t base_offset_cfg_node = static_cast<uint32_t>(dest_cfg_node.size());
        if (sum_cfg_node > 0)
          dest_cfg_node.resize_interned_columns(base_offset_cfg_node + sum_cfg_node);
        p_0.old_size_0 = base_offset_cfg_node;
        p_1.old_size_0 = base_offset_cfg_node;
      }

      if (total_0 > 0)
        JitRunner_cfg_node_from_edge_src::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_cfg_node_from_edge_dst::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<cfg_node, NEW_VER, cfg_node_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<cfg_node, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<cfg_node, NEW_VER, cfg_node_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<cfg_node, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using ancestor_path_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<child_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_ancestor_path_base::execute_fused(db, 0);
  else
    JitRunner_ancestor_path_base::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<ancestor_path, NEW_VER,
                                             ancestor_path_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<ancestor_path, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_4(DB& db, std::size_t max_iterations) {
  using ancestor_path_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<child_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<child_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<ancestor_path, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<ancestor_path, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " ancestor_path=" << _di.root().degree();
          } catch (...) {
            std::cerr << " ancestor_path=?";
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

    if (_tail_mode)
      JitRunner_ancestor_path_trans_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_ancestor_path_trans_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ancestor_path, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ancestor_path, NEW_VER,
                                               ancestor_path_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ancestor_path, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_5(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_6(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>>(
      db, 0);

  using path_begins_with_var_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_is_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_pbwv_base::execute_fused(db, 0);
  else
    JitRunner_pbwv_base::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>>(
      db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_begins_with_var, NEW_VER,
                                             path_begins_with_var_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_begins_with_var, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_7(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_8(DB& db, std::size_t max_iterations) {
  using path_begins_with_var_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_begins_with_var, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_begins_with_var, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>>(
      db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_begins_with_var, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<path_begins_with_var, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_begins_with_var=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_begins_with_var=?";
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

    if (_tail_mode)
      JitRunner_pbwv_ancestor_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_pbwv_ancestor_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_begins_with_var, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_begins_with_var, NEW_VER,
                                               path_begins_with_var_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>,
                                   FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>,
                                   DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_begins_with_var, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
}
template <typename DB>
static void step_9(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_10(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db,
                                                                                               0);

  using path_assigned_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_assigned_at_base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_path_assigned_at_base_rule::execute_fused(db, 0);
  else
    JitRunner_path_assigned_at_base_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_assigned_at, NEW_VER,
                                             path_assigned_at_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_assigned_at, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 1, 0>, DELTA_VER>>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_11(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_12(DB& db, std::size_t max_iterations) {
  using path_assigned_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db,
                                                                                               0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_assigned_at, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<path_assigned_at, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_assigned_at=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_assigned_at=?";
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

    if (_tail_mode)
      JitRunner_path_assigned_at_ancestor_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_path_assigned_at_ancestor_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(
        db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_assigned_at, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_assigned_at, NEW_VER,
                                               path_assigned_at_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(
        db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_assigned_at, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(
        db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_13(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_14(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using path_moved_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at_base, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  if (_tail_mode)
    JitRunner_path_moved_at_base_rule::execute_fused(db, 0);
  else
    JitRunner_path_moved_at_base_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_moved_at, NEW_VER,
                                             path_moved_at_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_moved_at, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_15(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_16(DB& db, std::size_t max_iterations) {
  using path_moved_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_moved_at, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<path_moved_at, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_moved_at=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_moved_at=?";
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

    if (_tail_mode)
      JitRunner_path_moved_at_ancestor_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_path_moved_at_ancestor_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_moved_at, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_moved_at, NEW_VER,
                                               path_moved_at_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_moved_at, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_17(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_18(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

  using path_maybe_initialized_on_exit_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_assigned_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  if (_tail_mode)
    JitRunner_pmioe_assigned::execute_fused(db, 0);
  else
    JitRunner_pmioe_assigned::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_maybe_initialized_on_exit, NEW_VER,
                                             path_maybe_initialized_on_exit_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_initialized_on_exit, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>,
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_19(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_20(DB& db, std::size_t max_iterations) {
  using path_maybe_initialized_on_exit_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_maybe_initialized_on_exit, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{1, 0}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<path_maybe_initialized_on_exit, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{1, 0}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_maybe_initialized_on_exit=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_maybe_initialized_on_exit=?";
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

    if (_tail_mode)
      JitRunner_pmioe_cfg_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_pmioe_cfg_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_initialized_on_exit, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_maybe_initialized_on_exit, NEW_VER,
                                               path_maybe_initialized_on_exit_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                   NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                   FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>,
                                   DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_initialized_on_exit, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
}
template <typename DB>
static void step_21(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_22(DB& db, std::size_t max_iterations) {
  // ERROR: unexpected plan kind for non-recursive step: moRebuildIndex
}
template <typename DB>
static void step_23(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

  using var_maybe_partly_initialized_on_exit_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 1, 0>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_begins_with_var, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  if (_tail_mode)
    JitRunner_vmpioe_from_path::execute_fused(db, 0);
  else
    JitRunner_vmpioe_from_path::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<var_maybe_partly_initialized_on_exit, NEW_VER,
                                             var_maybe_partly_initialized_on_exit_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<var_maybe_partly_initialized_on_exit, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 1, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_exit,
                                 std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_24(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_25(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using var_maybe_partly_initialized_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_entry,
                                 std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_vmpie_from_exit::execute_fused(db, 0);
  else
    JitRunner_vmpie_from_exit::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<
      var_maybe_partly_initialized_on_entry, NEW_VER,
      var_maybe_partly_initialized_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_entry,
                                 std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_entry,
                                 std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<var_maybe_partly_initialized_on_entry,
                                 std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<var_maybe_partly_initialized_on_entry, NEW_VER>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_26(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_27(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using var_drop_live_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_dropped_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_dropped_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_vdle_dropped::execute_fused(db, 0);
  else
    JitRunner_vdle_dropped::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<var_drop_live_on_entry, NEW_VER,
                                             var_drop_live_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>,
      SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<var_drop_live_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_28(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_29(DB& db, std::size_t max_iterations) {
  using var_drop_live_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_maybe_partly_initialized_on_exit, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<var_drop_live_on_entry, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<var_drop_live_on_entry, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " var_drop_live_on_entry=" << _di.root().degree();
          } catch (...) {
            std::cerr << " var_drop_live_on_entry=?";
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

    if (_tail_mode)
      JitRunner_vdle_cfg_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_vdle_cfg_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<var_drop_live_on_entry, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<var_drop_live_on_entry, NEW_VER,
                                               var_drop_live_on_entry_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                   NEW_VER>,
        SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                   FULL_VER>,
        SRDatalog::mir::IndexSpecT<var_drop_live_on_entry, std::integer_sequence<int, 0, 1>,
                                   DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<var_drop_live_on_entry, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_30(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_31(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(db,
                                                                                                0);

  using var_live_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_used_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_vle_used::execute_fused(db, 0);
  else
    JitRunner_vle_used::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<var_live_on_entry, NEW_VER,
                                             var_live_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<var_live_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 0, 1>, DELTA_VER>>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_32(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_33(DB& db, std::size_t max_iterations) {
  using var_live_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(db,
                                                                                                0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<var_live_on_entry, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{1, 0}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<var_live_on_entry, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{1, 0}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " var_live_on_entry=" << _di.root().degree();
          } catch (...) {
            std::cerr << " var_live_on_entry=?";
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

    if (_tail_mode)
      JitRunner_vle_cfg_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_vle_cfg_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
        db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<var_live_on_entry, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<var_live_on_entry, NEW_VER,
                                               var_live_on_entry_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>>(
        db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<var_live_on_entry, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
        db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
}
template <typename DB>
static void step_34(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_35(DB& db, std::size_t max_iterations) {
  // ERROR: unexpected plan kind for non-recursive step: moRebuildIndex
}
template <typename DB>
static void step_36(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db, 0);

  using origin_live_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(3);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_node, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<universal_region, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      drop_of_var_derefs_origin, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      var_drop_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      drop_of_var_derefs_origin, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      use_of_var_derefs_origin, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<var_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      use_of_var_derefs_origin, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_ole_universal::execute_fused(db, 0);
    JitRunner_ole_drop::execute_fused(db, 0);
    JitRunner_ole_use::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_ole_universal::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_ole_drop::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_ole_use::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_origin_live_on_entry_size =
          p_0.num_threads + p_1.num_threads + p_2.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_origin_live_on_entry(
          shared_buf_origin_live_on_entry_size);
      uint32_t off_origin_live_on_entry_0 = 0;
      p_0.thread_counts_ptr = shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_0;
      uint32_t off_origin_live_on_entry_1 = off_origin_live_on_entry_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_1;
      uint32_t off_origin_live_on_entry_2 = off_origin_live_on_entry_1 + p_1.num_threads;
      p_2.thread_counts_ptr = shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_2;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_ole_universal::launch_count(p_0, _stream_pool.get(0));
      JitRunner_ole_drop::launch_count(p_1, _stream_pool.get(1));
      JitRunner_ole_use::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      thrust::exclusive_scan(
          rmm::exec_policy(_stream_pool.get(0)), shared_buf_origin_live_on_entry.data(),
          shared_buf_origin_live_on_entry.data() + shared_buf_origin_live_on_entry_size,
          shared_buf_origin_live_on_entry.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      // Read boundary values for origin_live_on_entry (3 rules)
      uint32_t bnd_origin_live_on_entry[4];
      GPU_MEMCPY_ASYNC(&bnd_origin_live_on_entry[0],
                       shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_0,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_origin_live_on_entry[1],
                       shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_1,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_origin_live_on_entry[2],
                       shared_buf_origin_live_on_entry.data() + off_origin_live_on_entry_2,
                       sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(
          &bnd_origin_live_on_entry[3],
          shared_buf_origin_live_on_entry.data() + shared_buf_origin_live_on_entry_size - 1,
          sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_origin_live_on_entry[1] - bnd_origin_live_on_entry[0];
      uint32_t total_1 = bnd_origin_live_on_entry[2] - bnd_origin_live_on_entry[1];
      uint32_t total_2 = bnd_origin_live_on_entry[3] - bnd_origin_live_on_entry[2];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize origin_live_on_entry
        uint32_t sum_origin_live_on_entry = total_0 + total_1 + total_2;
        auto& dest_origin_live_on_entry = get_relation_by_schema<origin_live_on_entry, NEW_VER>(db);
        uint32_t base_offset_origin_live_on_entry =
            static_cast<uint32_t>(dest_origin_live_on_entry.size());
        if (sum_origin_live_on_entry > 0)
          dest_origin_live_on_entry.resize_interned_columns(base_offset_origin_live_on_entry +
                                                            sum_origin_live_on_entry);
        p_0.old_size_0 = base_offset_origin_live_on_entry;
        p_1.old_size_0 = base_offset_origin_live_on_entry;
        p_2.old_size_0 = base_offset_origin_live_on_entry;
      }

      if (total_0 > 0)
        JitRunner_ole_universal::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_ole_drop::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0)
        JitRunner_ole_use::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db);
  SRDatalog::GPU::mir_helpers::check_size_fn<origin_live_on_entry, NEW_VER,
                                             origin_live_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_live_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db);
  SRDatalog::GPU::mir_helpers::check_size_fn<origin_live_on_entry, NEW_VER,
                                             origin_live_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_live_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>>(
      db);
  SRDatalog::GPU::mir_helpers::check_size_fn<origin_live_on_entry, NEW_VER,
                                             origin_live_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_live_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_37(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_38(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db, 0);

  using subset_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset_base, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db,
                                                                                              0);
  if (_tail_mode)
    JitRunner_subset_base_rule::execute_fused(db, 0);
  else
    JitRunner_subset_base_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<subset, NEW_VER, subset_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<subset, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_39(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_40(DB& db, std::size_t max_iterations) {
  using subset_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset_base, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset_base, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset_base, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db,
                                                                                              0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 1, 0>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<subset, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{1, 2, 0}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<subset, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{1, 2, 0}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " subset=" << _di.root().degree();
          } catch (...) {
            std::cerr << " subset=?";
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
    _stream_pool.ensure(2);

    if (_tail_mode && false) {
    } else {
      // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 =
            JitRunner_subset_trans_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 =
            JitRunner_subset_cfg_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));

        // Phase 2a: Allocate shared count buffers per dest

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_subset_trans_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_subset_cfg_D0::launch_count(p_1, _stream_pool.get(1));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        JitRunner_subset_cfg_D0::scan_only(p_1, _stream_pool.get(1));
        JitRunner_subset_trans_D0::scan_only(p_0, _stream_pool.get(0));

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        uint32_t total_1 = JitRunner_subset_cfg_D0::read_total(p_1);
        uint32_t total_0 = JitRunner_subset_trans_D0::read_total(p_0);

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize subset
          uint32_t sum_subset = total_0 + total_1;
          auto& dest_subset = get_relation_by_schema<subset, NEW_VER>(db);
          uint32_t base_offset_subset = static_cast<uint32_t>(dest_subset.size());
          if (sum_subset > 0)
            dest_subset.resize_interned_columns(base_offset_subset + sum_subset);
          uint32_t running_offset_subset = base_offset_subset;
          p_0.old_size_0 = running_offset_subset;
          running_offset_subset += total_0;
          p_1.old_size_0 = running_offset_subset;
          running_offset_subset += total_1;
        }

        if (total_0 > 0)
          JitRunner_subset_trans_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_subset_cfg_D0::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        nvtxRangePop();  // join_pipeline
      }
    }
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<subset, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<subset, NEW_VER, subset_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<subset, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 2, 0, 1>, DELTA_VER>>(db);
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
}
template <typename DB>
static void step_41(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_42(DB& db, std::size_t max_iterations) {
  // ERROR: unexpected plan kind for non-recursive step: moRebuildIndex
}
template <typename DB>
static void step_43(DB& db, std::size_t max_iterations) {
  // ERROR: unexpected plan kind for non-recursive step: moRebuildIndex
}
template <typename DB>
static void step_44(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db, 0);

  using origin_contains_loan_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>,
                                 FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_issued_at, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db,
                                                                                                 0);
  if (_tail_mode)
    JitRunner_ocle_issued::execute_fused(db, 0);
  else
    JitRunner_ocle_issued::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<origin_contains_loan_on_entry, NEW_VER,
                                             origin_contains_loan_on_entry_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>,
                                 NEW_VER>,
      SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>,
                                 FULL_VER>,
      SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_contains_loan_on_entry, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_45(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_46(DB& db, std::size_t max_iterations) {
  using origin_contains_loan_on_entry_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>,
                                 FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<origin_contains_loan_on_entry, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 2, 1}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<origin_contains_loan_on_entry, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 2, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " origin_contains_loan_on_entry=" << _di.root().degree();
          } catch (...) {
            std::cerr << " origin_contains_loan_on_entry=?";
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
    _stream_pool.ensure(2);

    if (_tail_mode) {
      JitRunner_ocle_subset_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_ocle_cfg_D0::execute_fused(db, static_cast<uint32_t>(iter));
    } else {
      // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 =
            JitRunner_ocle_subset_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 =
            JitRunner_ocle_cfg_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_origin_contains_loan_on_entry_size =
            p_0.num_threads + p_1.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_origin_contains_loan_on_entry(
            shared_buf_origin_contains_loan_on_entry_size);
        uint32_t off_origin_contains_loan_on_entry_0 = 0;
        p_0.thread_counts_ptr =
            shared_buf_origin_contains_loan_on_entry.data() + off_origin_contains_loan_on_entry_0;
        uint32_t off_origin_contains_loan_on_entry_1 =
            off_origin_contains_loan_on_entry_0 + p_0.num_threads;
        p_1.thread_counts_ptr =
            shared_buf_origin_contains_loan_on_entry.data() + off_origin_contains_loan_on_entry_1;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_ocle_subset_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_ocle_cfg_D0::launch_count(p_1, _stream_pool.get(1));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(
            rmm::exec_policy(_stream_pool.get(0)), shared_buf_origin_contains_loan_on_entry.data(),
            shared_buf_origin_contains_loan_on_entry.data() +
                shared_buf_origin_contains_loan_on_entry_size,
            shared_buf_origin_contains_loan_on_entry.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for origin_contains_loan_on_entry (2 rules)
        uint32_t bnd_origin_contains_loan_on_entry[3];
        GPU_MEMCPY_ASYNC(
            &bnd_origin_contains_loan_on_entry[0],
            shared_buf_origin_contains_loan_on_entry.data() + off_origin_contains_loan_on_entry_0,
            sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(
            &bnd_origin_contains_loan_on_entry[1],
            shared_buf_origin_contains_loan_on_entry.data() + off_origin_contains_loan_on_entry_1,
            sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_origin_contains_loan_on_entry[2],
                         shared_buf_origin_contains_loan_on_entry.data() +
                             shared_buf_origin_contains_loan_on_entry_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 =
            bnd_origin_contains_loan_on_entry[1] - bnd_origin_contains_loan_on_entry[0];
        uint32_t total_1 =
            bnd_origin_contains_loan_on_entry[2] - bnd_origin_contains_loan_on_entry[1];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize origin_contains_loan_on_entry
          uint32_t sum_origin_contains_loan_on_entry = total_0 + total_1;
          auto& dest_origin_contains_loan_on_entry =
              get_relation_by_schema<origin_contains_loan_on_entry, NEW_VER>(db);
          uint32_t base_offset_origin_contains_loan_on_entry =
              static_cast<uint32_t>(dest_origin_contains_loan_on_entry.size());
          if (sum_origin_contains_loan_on_entry > 0)
            dest_origin_contains_loan_on_entry.resize_interned_columns(
                base_offset_origin_contains_loan_on_entry + sum_origin_contains_loan_on_entry);
          p_0.old_size_0 = base_offset_origin_contains_loan_on_entry;
          p_1.old_size_0 = base_offset_origin_contains_loan_on_entry;
        }

        if (total_0 > 0)
          JitRunner_ocle_subset_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_ocle_cfg_D0::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        nvtxRangePop();  // join_pipeline
      }
    }
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_contains_loan_on_entry, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<origin_contains_loan_on_entry, NEW_VER,
                                               origin_contains_loan_on_entry_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry,
                                   std::integer_sequence<int, 0, 2, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry,
                                   std::integer_sequence<int, 0, 2, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<origin_contains_loan_on_entry,
                                   std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<origin_contains_loan_on_entry, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_47(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_48(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, NEW_VER>>(db,
                                                                                              0);

  using placeholder_origin_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<universal_region, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_placeholder_origin_rule::execute_fused(db, 0);
  else
    JitRunner_placeholder_origin_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<placeholder_origin, NEW_VER,
                                             placeholder_origin_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<placeholder_origin, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_49(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_50(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using known_placeholder_subset_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset_input, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_kps_seed::execute_fused(db, 0);
  else
    JitRunner_kps_seed::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<known_placeholder_subset, NEW_VER,
                                             known_placeholder_subset_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 NEW_VER>,
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>,
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<known_placeholder_subset, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 DELTA_VER>,
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_51(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_52(DB& db, std::size_t max_iterations) {
  using known_placeholder_subset_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                 FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<known_placeholder_subset, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<known_placeholder_subset, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " known_placeholder_subset=" << _di.root().degree();
          } catch (...) {
            std::cerr << " known_placeholder_subset=?";
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
    _stream_pool.ensure(2);

    if (_tail_mode) {
      JitRunner_kps_transitive_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_kps_transitive_D1::execute_fused(db, static_cast<uint32_t>(iter));
    } else {
      // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_kps_transitive_D0::setup(db, static_cast<uint32_t>(iter),
                                                      _stream_pool.get(0));
        auto p_1 = JitRunner_kps_transitive_D1::setup(db, static_cast<uint32_t>(iter),
                                                      _stream_pool.get(1));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_known_placeholder_subset_size = p_0.num_threads + p_1.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_known_placeholder_subset(
            shared_buf_known_placeholder_subset_size);
        uint32_t off_known_placeholder_subset_0 = 0;
        p_0.thread_counts_ptr =
            shared_buf_known_placeholder_subset.data() + off_known_placeholder_subset_0;
        uint32_t off_known_placeholder_subset_1 = off_known_placeholder_subset_0 + p_0.num_threads;
        p_1.thread_counts_ptr =
            shared_buf_known_placeholder_subset.data() + off_known_placeholder_subset_1;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_kps_transitive_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_kps_transitive_D1::launch_count(p_1, _stream_pool.get(1));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(
            rmm::exec_policy(_stream_pool.get(0)), shared_buf_known_placeholder_subset.data(),
            shared_buf_known_placeholder_subset.data() + shared_buf_known_placeholder_subset_size,
            shared_buf_known_placeholder_subset.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for known_placeholder_subset (2 rules)
        uint32_t bnd_known_placeholder_subset[3];
        GPU_MEMCPY_ASYNC(
            &bnd_known_placeholder_subset[0],
            shared_buf_known_placeholder_subset.data() + off_known_placeholder_subset_0,
            sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(
            &bnd_known_placeholder_subset[1],
            shared_buf_known_placeholder_subset.data() + off_known_placeholder_subset_1,
            sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_known_placeholder_subset[2],
                         shared_buf_known_placeholder_subset.data() +
                             shared_buf_known_placeholder_subset_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_known_placeholder_subset[1] - bnd_known_placeholder_subset[0];
        uint32_t total_1 = bnd_known_placeholder_subset[2] - bnd_known_placeholder_subset[1];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize known_placeholder_subset
          uint32_t sum_known_placeholder_subset = total_0 + total_1;
          auto& dest_known_placeholder_subset =
              get_relation_by_schema<known_placeholder_subset, NEW_VER>(db);
          uint32_t base_offset_known_placeholder_subset =
              static_cast<uint32_t>(dest_known_placeholder_subset.size());
          if (sum_known_placeholder_subset > 0)
            dest_known_placeholder_subset.resize_interned_columns(
                base_offset_known_placeholder_subset + sum_known_placeholder_subset);
          p_0.old_size_0 = base_offset_known_placeholder_subset;
          p_1.old_size_0 = base_offset_known_placeholder_subset;
        }

        if (total_0 > 0)
          JitRunner_kps_transitive_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_kps_transitive_D1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        nvtxRangePop();  // join_pipeline
      }
    }
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        known_placeholder_subset, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<known_placeholder_subset, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<known_placeholder_subset, NEW_VER,
                                               known_placeholder_subset_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                   NEW_VER>,
        SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                   FULL_VER>,
        SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                   DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<known_placeholder_subset, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 0, 1>,
                                   DELTA_VER>,
        SRDatalog::mir::IndexSpecT<known_placeholder_subset, std::integer_sequence<int, 1, 0>,
                                   DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        known_placeholder_subset, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_53(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_54(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using loan_live_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<origin_live_on_entry, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      origin_contains_loan_on_entry, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_loan_live_at_rule::execute_fused(db, 0);
  else
    JitRunner_loan_live_at_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<loan_live_at, NEW_VER, loan_live_at_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<loan_live_at, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_55(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_56(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db,
                                                                                              0);

  using errors_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using subset_error_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(2);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_invalidated_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_invalidated_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(
      db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<loan_live_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>>(db,
                                                                                               0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<placeholder_origin, std::integer_sequence<int, 0>, FULL_VER>>(db,
                                                                                               0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<subset, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      known_placeholder_subset, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_errors_rule::execute_fused(db, 0);
    JitRunner_subset_error_rule::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_errors_rule::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_subset_error_rule::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_errors_rule::launch_count(p_0, _stream_pool.get(0));
      JitRunner_subset_error_rule::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_errors_rule::scan_only(p_0, _stream_pool.get(0));
      JitRunner_subset_error_rule::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_errors_rule::read_total(p_0);
      uint32_t total_1 = JitRunner_subset_error_rule::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize errors
        uint32_t sum_errors = total_0;
        auto& dest_errors = get_relation_by_schema<errors, NEW_VER>(db);
        uint32_t base_offset_errors = static_cast<uint32_t>(dest_errors.size());
        if (sum_errors > 0)
          dest_errors.resize_interned_columns(base_offset_errors + sum_errors);
        uint32_t running_offset_errors = base_offset_errors;
        p_0.old_size_0 = running_offset_errors;
        running_offset_errors += total_0;
      }
      {  // Resize subset_error
        uint32_t sum_subset_error = total_1;
        auto& dest_subset_error = get_relation_by_schema<subset_error, NEW_VER>(db);
        uint32_t base_offset_subset_error = static_cast<uint32_t>(dest_subset_error.size());
        if (sum_subset_error > 0)
          dest_subset_error.resize_interned_columns(base_offset_subset_error + sum_subset_error);
        uint32_t running_offset_subset_error = base_offset_subset_error;
        p_1.old_size_0 = running_offset_subset_error;
        running_offset_subset_error += total_1;
      }

      if (total_0 > 0)
        JitRunner_errors_rule::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_subset_error_rule::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<errors, NEW_VER, errors_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<errors, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<subset_error, NEW_VER, subset_error_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<subset_error, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_57(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<errors, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_58(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<subset_error, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_59(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db,
                                                                                               0);

  using path_accessed_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_accessed_at_base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_path_accessed_at_base_rule::execute_fused(db, 0);
  else
    JitRunner_path_accessed_at_base_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_accessed_at, NEW_VER,
                                             path_accessed_at_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(
      db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_accessed_at, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_60(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_61(DB& db, std::size_t max_iterations) {
  using path_accessed_at_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db,
                                                                                                 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<ancestor_path, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(db,
                                                                                               0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_accessed_at, DELTA_VER>(db);
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
          auto& _dr = get_relation_by_schema<path_accessed_at, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_accessed_at=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_accessed_at=?";
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

    if (_tail_mode)
      JitRunner_path_accessed_at_ancestor_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_path_accessed_at_ancestor_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>>(
        db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_accessed_at, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_accessed_at, NEW_VER,
                                               path_accessed_at_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, DELTA_VER>>(
        db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_accessed_at, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(
        db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
}
template <typename DB>
static void step_62(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<path_accessed_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_63(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

  using path_maybe_uninitialized_on_exit_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<path_moved_at, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_pmuoe_moved::execute_fused(db, 0);
  else
    JitRunner_pmuoe_moved::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<path_maybe_uninitialized_on_exit, NEW_VER,
                                             path_maybe_uninitialized_on_exit_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 NEW_VER>,
      SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>,
      SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_uninitialized_on_exit, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_64(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_65(DB& db, std::size_t max_iterations) {
  using path_maybe_uninitialized_on_exit_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>,
                                 FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<path_maybe_uninitialized_on_exit, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{1, 0}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<path_maybe_uninitialized_on_exit, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{1, 0}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " path_maybe_uninitialized_on_exit=" << _di.root().degree();
          } catch (...) {
            std::cerr << " path_maybe_uninitialized_on_exit=?";
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

    if (_tail_mode)
      JitRunner_pmuoe_cfg_D0::execute_fused(db, static_cast<uint32_t>(iter));
    else
      JitRunner_pmuoe_cfg_D0::execute(db, static_cast<uint32_t>(iter));
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<
        path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_uninitialized_on_exit, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<path_maybe_uninitialized_on_exit, NEW_VER,
                                               path_maybe_uninitialized_on_exit_canonical_spec_t>(
        db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit,
                                   std::integer_sequence<int, 1, 0>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit,
                                   std::integer_sequence<int, 1, 0>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<path_maybe_uninitialized_on_exit,
                                   std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<path_maybe_uninitialized_on_exit, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<
        path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
}
template <typename DB>
static void step_66(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_67(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using move_error_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<
      path_maybe_uninitialized_on_exit, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<cfg_edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_move_error_rule::execute_fused(db, 0);
  else
    JitRunner_move_error_rule::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<move_error, NEW_VER, move_error_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<move_error, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_68(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<move_error, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
