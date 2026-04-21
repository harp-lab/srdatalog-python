  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using Insert_canonical_spec_t = SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>;
    using Remove_canonical_spec_t = SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InsertInput, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RemoveInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_Insert::execute_fused(db, 0);
      JitRunner_Remove::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_Insert::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Remove::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_Insert::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Remove::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_Insert::scan_only(p_0, _stream_pool.get(0));
      JitRunner_Remove::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_Insert::read_total(p_0);
      uint32_t total_1 = JitRunner_Remove::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize Insert
        uint32_t sum_Insert = total_0;
        auto& dest_Insert = get_relation_by_schema<Insert, NEW_VER>(db);
        uint32_t base_offset_Insert = static_cast<uint32_t>(dest_Insert.size());
        if (sum_Insert > 0) dest_Insert.resize_interned_columns(base_offset_Insert + sum_Insert);
        uint32_t running_offset_Insert = base_offset_Insert;
        p_0.old_size_0 = running_offset_Insert;
        running_offset_Insert += total_0;
      }
      {  // Resize Remove
        uint32_t sum_Remove = total_1;
        auto& dest_Remove = get_relation_by_schema<Remove, NEW_VER>(db);
        uint32_t base_offset_Remove = static_cast<uint32_t>(dest_Remove.size());
        if (sum_Remove > 0) dest_Remove.resize_interned_columns(base_offset_Remove + sum_Remove);
        uint32_t running_offset_Remove = base_offset_Remove;
        p_1.old_size_0 = running_offset_Remove;
        running_offset_Remove += total_1;
      }

      if (total_0 > 0) JitRunner_Insert::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_Remove::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Insert, NEW_VER, Insert_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Insert, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Remove, NEW_VER, Remove_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Remove, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);

    using LaterChild_canonical_spec_t = SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_LaterChild::execute_fused(db, 0); else JitRunner_LaterChild::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<LaterChild, NEW_VER, LaterChild_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<LaterChild, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_4(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_5(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

    using FirstChild_canonical_spec_t = SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using Sibling_canonical_spec_t = SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterChild, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_FirstChild::execute_fused(db, 0);
      JitRunner_Sibling::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_FirstChild::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Sibling::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_FirstChild::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Sibling::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_FirstChild::scan_only(p_0, _stream_pool.get(0));
      JitRunner_Sibling::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_FirstChild::read_total(p_0);
      uint32_t total_1 = JitRunner_Sibling::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize FirstChild
        uint32_t sum_FirstChild = total_0;
        auto& dest_FirstChild = get_relation_by_schema<FirstChild, NEW_VER>(db);
        uint32_t base_offset_FirstChild = static_cast<uint32_t>(dest_FirstChild.size());
        if (sum_FirstChild > 0) dest_FirstChild.resize_interned_columns(base_offset_FirstChild + sum_FirstChild);
        uint32_t running_offset_FirstChild = base_offset_FirstChild;
        p_0.old_size_0 = running_offset_FirstChild;
        running_offset_FirstChild += total_0;
      }
      {  // Resize Sibling
        uint32_t sum_Sibling = total_1;
        auto& dest_Sibling = get_relation_by_schema<Sibling, NEW_VER>(db);
        uint32_t base_offset_Sibling = static_cast<uint32_t>(dest_Sibling.size());
        if (sum_Sibling > 0) dest_Sibling.resize_interned_columns(base_offset_Sibling + sum_Sibling);
        uint32_t running_offset_Sibling = base_offset_Sibling;
        p_1.old_size_0 = running_offset_Sibling;
        running_offset_Sibling += total_1;
      }

      if (total_0 > 0) JitRunner_FirstChild::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_Sibling::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<FirstChild, NEW_VER, FirstChild_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<FirstChild, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Sibling, NEW_VER, Sibling_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Sibling, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_6(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_7(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_8(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

    using LaterSibling_canonical_spec_t = SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using LaterSibling2_canonical_spec_t = SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_LaterSibling::execute_fused(db, 0);
      JitRunner_LaterSibling2::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_LaterSibling::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_LaterSibling2::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_LaterSibling::launch_count(p_0, _stream_pool.get(0));
      JitRunner_LaterSibling2::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_LaterSibling::scan_only(p_0, _stream_pool.get(0));
      JitRunner_LaterSibling2::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_LaterSibling::read_total(p_0);
      uint32_t total_1 = JitRunner_LaterSibling2::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize LaterSibling
        uint32_t sum_LaterSibling = total_0;
        auto& dest_LaterSibling = get_relation_by_schema<LaterSibling, NEW_VER>(db);
        uint32_t base_offset_LaterSibling = static_cast<uint32_t>(dest_LaterSibling.size());
        if (sum_LaterSibling > 0) dest_LaterSibling.resize_interned_columns(base_offset_LaterSibling + sum_LaterSibling);
        uint32_t running_offset_LaterSibling = base_offset_LaterSibling;
        p_0.old_size_0 = running_offset_LaterSibling;
        running_offset_LaterSibling += total_0;
      }
      {  // Resize LaterSibling2
        uint32_t sum_LaterSibling2 = total_1;
        auto& dest_LaterSibling2 = get_relation_by_schema<LaterSibling2, NEW_VER>(db);
        uint32_t base_offset_LaterSibling2 = static_cast<uint32_t>(dest_LaterSibling2.size());
        if (sum_LaterSibling2 > 0) dest_LaterSibling2.resize_interned_columns(base_offset_LaterSibling2 + sum_LaterSibling2);
        uint32_t running_offset_LaterSibling2 = base_offset_LaterSibling2;
        p_1.old_size_0 = running_offset_LaterSibling2;
        running_offset_LaterSibling2 += total_1;
      }

      if (total_0 > 0) JitRunner_LaterSibling::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_LaterSibling2::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<LaterSibling, NEW_VER, LaterSibling_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<LaterSibling, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<LaterSibling2, NEW_VER, LaterSibling2_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<LaterSibling2, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_9(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_10(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_11(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using NextSibling_canonical_spec_t = SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using HasNextSibling_canonical_spec_t = SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LaterSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_NextSibling::execute_fused(db, 0);
      JitRunner_HasNextSibling::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_NextSibling::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_HasNextSibling::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_NextSibling::launch_count(p_0, _stream_pool.get(0));
      JitRunner_HasNextSibling::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_HasNextSibling::scan_only(p_1, _stream_pool.get(1));
      JitRunner_NextSibling::scan_only(p_0, _stream_pool.get(0));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_1 = JitRunner_HasNextSibling::read_total(p_1);
      uint32_t total_0 = JitRunner_NextSibling::read_total(p_0);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize HasNextSibling
        uint32_t sum_HasNextSibling = total_1;
        auto& dest_HasNextSibling = get_relation_by_schema<HasNextSibling, NEW_VER>(db);
        uint32_t base_offset_HasNextSibling = static_cast<uint32_t>(dest_HasNextSibling.size());
        if (sum_HasNextSibling > 0) dest_HasNextSibling.resize_interned_columns(base_offset_HasNextSibling + sum_HasNextSibling);
        uint32_t running_offset_HasNextSibling = base_offset_HasNextSibling;
        p_1.old_size_0 = running_offset_HasNextSibling;
        running_offset_HasNextSibling += total_1;
      }
      {  // Resize NextSibling
        uint32_t sum_NextSibling = total_0;
        auto& dest_NextSibling = get_relation_by_schema<NextSibling, NEW_VER>(db);
        uint32_t base_offset_NextSibling = static_cast<uint32_t>(dest_NextSibling.size());
        if (sum_NextSibling > 0) dest_NextSibling.resize_interned_columns(base_offset_NextSibling + sum_NextSibling);
        uint32_t running_offset_NextSibling = base_offset_NextSibling;
        p_0.old_size_0 = running_offset_NextSibling;
        running_offset_NextSibling += total_0;
      }

      if (total_0 > 0) JitRunner_NextSibling::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_HasNextSibling::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextSibling, NEW_VER, NextSibling_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextSibling, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HasNextSibling, NEW_VER, HasNextSibling_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HasNextSibling, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_12(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_13(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_14(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);

    using NotSiblingInsert_canonical_spec_t = SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasNextSibling, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_NotSiblingInsert::execute_fused(db, 0); else JitRunner_NotSiblingInsert::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NotSiblingInsert, NEW_VER, NotSiblingInsert_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NotSiblingInsert, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_15(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_16(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

    using NextSiblingAnc_canonical_spec_t = SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSibling, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_NextSiblingAnc1::execute_fused(db, 0); else JitRunner_NextSiblingAnc1::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextSiblingAnc, NEW_VER, NextSiblingAnc_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextSiblingAnc, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_17(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_18(DB& db, std::size_t max_iterations) {
    using NextSiblingAnc_canonical_spec_t = SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotSiblingInsert, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<NextSiblingAnc, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1, 2, 3}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<NextSiblingAnc, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1, 2, 3}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " NextSiblingAnc=" << _di.root().degree(); } catch (...) { std::cerr << " NextSiblingAnc=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_NextSiblingAnc2_D0::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_NextSiblingAnc2_D0::execute(db, static_cast<uint32_t>(iter));
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<NextSiblingAnc, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<NextSiblingAnc, NEW_VER, NextSiblingAnc_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<NextSiblingAnc, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_19(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_20(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using HasChild_canonical_spec_t = SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_HasChild::execute_fused(db, 0); else JitRunner_HasChild::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HasChild, NEW_VER, HasChild_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HasChild, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_21(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_22(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>>(db, 0);

    using NextElem_canonical_spec_t = SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using Assign_canonical_spec_t = SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(3);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<FirstChild, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextSiblingAnc, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasChild, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Insert, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_NextElem1::execute_fused(db, 0);
      JitRunner_NextElem2::execute_fused(db, 0);
      JitRunner_Assign::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_NextElem1::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_NextElem2::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_Assign::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_NextElem_size = p_0.num_threads + p_1.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_NextElem(shared_buf_NextElem_size);
      uint32_t off_NextElem_0 = 0;
      p_0.thread_counts_ptr = shared_buf_NextElem.data() + off_NextElem_0;
      uint32_t off_NextElem_1 = off_NextElem_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_NextElem.data() + off_NextElem_1;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_NextElem1::launch_count(p_0, _stream_pool.get(0));
      JitRunner_NextElem2::launch_count(p_1, _stream_pool.get(1));
      JitRunner_Assign::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_Assign::scan_only(p_2, _stream_pool.get(2));
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_NextElem.data(), shared_buf_NextElem.data() + shared_buf_NextElem_size, shared_buf_NextElem.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_2 = JitRunner_Assign::read_total(p_2);
      // Read boundary values for NextElem (2 rules)
      uint32_t bnd_NextElem[3];
      GPU_MEMCPY_ASYNC(&bnd_NextElem[0], shared_buf_NextElem.data() + off_NextElem_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_NextElem[1], shared_buf_NextElem.data() + off_NextElem_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_NextElem[2], shared_buf_NextElem.data() + shared_buf_NextElem_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_NextElem[1] - bnd_NextElem[0];
      uint32_t total_1 = bnd_NextElem[2] - bnd_NextElem[1];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize Assign
        uint32_t sum_Assign = total_2;
        auto& dest_Assign = get_relation_by_schema<Assign, NEW_VER>(db);
        uint32_t base_offset_Assign = static_cast<uint32_t>(dest_Assign.size());
        if (sum_Assign > 0) dest_Assign.resize_interned_columns(base_offset_Assign + sum_Assign);
        uint32_t running_offset_Assign = base_offset_Assign;
        p_2.old_size_0 = running_offset_Assign;
        running_offset_Assign += total_2;
      }
      {  // Resize NextElem
        uint32_t sum_NextElem = total_0 + total_1;
        auto& dest_NextElem = get_relation_by_schema<NextElem, NEW_VER>(db);
        uint32_t base_offset_NextElem = static_cast<uint32_t>(dest_NextElem.size());
        if (sum_NextElem > 0) dest_NextElem.resize_interned_columns(base_offset_NextElem + sum_NextElem);
        p_0.old_size_0 = base_offset_NextElem;
        p_1.old_size_0 = base_offset_NextElem;
      }

      if (total_0 > 0) JitRunner_NextElem1::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_NextElem2::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_Assign::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextElem, NEW_VER, NextElem_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextElem, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextElem, NEW_VER, NextElem_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextElem, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Assign, NEW_VER, Assign_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>, SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>, SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Assign, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_23(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_24(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_25(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

    using CurrentValue_canonical_spec_t = SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Remove, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_CurrentValue::execute_fused(db, 0); else JitRunner_CurrentValue::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<CurrentValue, NEW_VER, CurrentValue_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<CurrentValue, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_26(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_27(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using HasValue_canonical_spec_t = SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_HasValue::execute_fused(db, 0); else JitRunner_HasValue::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HasValue, NEW_VER, HasValue_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HasValue, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_28(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_29(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);

    using NotHasValueNextElem_canonical_spec_t = SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_NotHasValueNextElem::execute_fused(db, 0); else JitRunner_NotHasValueNextElem::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NotHasValueNextElem, NEW_VER, NotHasValueNextElem_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NotHasValueNextElem, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_30(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_31(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

    using SkipBlank_canonical_spec_t = SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextElem, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_SkipBlank1::execute_fused(db, 0); else JitRunner_SkipBlank1::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SkipBlank, NEW_VER, SkipBlank_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SkipBlank, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_32(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_33(DB& db, std::size_t max_iterations) {
    using SkipBlank_canonical_spec_t = SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NotHasValueNextElem, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<SkipBlank, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1, 2, 3}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<SkipBlank, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1, 2, 3}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " SkipBlank=" << _di.root().degree(); } catch (...) { std::cerr << " SkipBlank=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_SkipBlank2_D0::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_SkipBlank2_D0::execute(db, static_cast<uint32_t>(iter));
      {
    auto& sb_new = get_relation_by_schema<SkipBlank, DELTA_VER>(db);
    auto& sb_new_idx = sb_new.get_index({{0,1,2,3}});
    auto& sb_full = get_relation_by_schema<SkipBlank, FULL_VER>(db);
    auto& sb_full_idx = sb_full.get_index({{0,1,2,3}});
    std::cout << "  SkipBlank delta: " << sb_new_idx.root().degree() 
              << ", full: " << sb_full_idx.root().degree() << std::endl;
    
      }
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<SkipBlank, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<SkipBlank, NEW_VER, SkipBlank_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<SkipBlank, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_34(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_35(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db, 0);

    using NextVisible_canonical_spec_t = SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SkipBlank, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_NextVisible1::execute_fused(db, 0);
      JitRunner_NextVisible2::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_NextVisible1::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_NextVisible2::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_NextVisible_size = p_0.num_threads + p_1.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_NextVisible(shared_buf_NextVisible_size);
      uint32_t off_NextVisible_0 = 0;
      p_0.thread_counts_ptr = shared_buf_NextVisible.data() + off_NextVisible_0;
      uint32_t off_NextVisible_1 = off_NextVisible_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_NextVisible.data() + off_NextVisible_1;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_NextVisible1::launch_count(p_0, _stream_pool.get(0));
      JitRunner_NextVisible2::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_NextVisible.data(), shared_buf_NextVisible.data() + shared_buf_NextVisible_size, shared_buf_NextVisible.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      // Read boundary values for NextVisible (2 rules)
      uint32_t bnd_NextVisible[3];
      GPU_MEMCPY_ASYNC(&bnd_NextVisible[0], shared_buf_NextVisible.data() + off_NextVisible_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_NextVisible[1], shared_buf_NextVisible.data() + off_NextVisible_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_NextVisible[2], shared_buf_NextVisible.data() + shared_buf_NextVisible_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_NextVisible[1] - bnd_NextVisible[0];
      uint32_t total_1 = bnd_NextVisible[2] - bnd_NextVisible[1];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize NextVisible
        uint32_t sum_NextVisible = total_0 + total_1;
        auto& dest_NextVisible = get_relation_by_schema<NextVisible, NEW_VER>(db);
        uint32_t base_offset_NextVisible = static_cast<uint32_t>(dest_NextVisible.size());
        if (sum_NextVisible > 0) dest_NextVisible.resize_interned_columns(base_offset_NextVisible + sum_NextVisible);
        p_0.old_size_0 = base_offset_NextVisible;
        p_1.old_size_0 = base_offset_NextVisible;
      }

      if (total_0 > 0) JitRunner_NextVisible1::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_NextVisible2::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextVisible, NEW_VER, NextVisible_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextVisible, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<NextVisible, NEW_VER, NextVisible_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<NextVisible, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_36(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_37(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

    using Result_canonical_spec_t = SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<NextVisible, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CurrentValue, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_Result::execute_fused(db, 0); else JitRunner_Result::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Result, NEW_VER, Result_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Result, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_38(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Result, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
