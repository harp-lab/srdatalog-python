  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>>(db, 0);

    using HasReply_canonical_spec_t = SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>;
    using HasLiker_canonical_spec_t = SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>;
    using Case1_canonical_spec_t = SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(3);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReplyOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Likes, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReplyOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Likes, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReplyOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Likes, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_MarkHasReply::execute_fused(db, 0);
      JitRunner_MarkHasLiker::execute_fused(db, 0);
      JitRunner_Case1Both::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_MarkHasReply::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_MarkHasLiker::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_Case1Both::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_MarkHasReply::launch_count(p_0, _stream_pool.get(0));
      JitRunner_MarkHasLiker::launch_count(p_1, _stream_pool.get(1));
      JitRunner_Case1Both::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_Case1Both::scan_only(p_2, _stream_pool.get(2));
      JitRunner_MarkHasLiker::scan_only(p_1, _stream_pool.get(1));
      JitRunner_MarkHasReply::scan_only(p_0, _stream_pool.get(0));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_2 = JitRunner_Case1Both::read_total(p_2);
      uint32_t total_1 = JitRunner_MarkHasLiker::read_total(p_1);
      uint32_t total_0 = JitRunner_MarkHasReply::read_total(p_0);

      std::cout << " >>>>>>>>>>>>>>>>> Case1Both count: " << total_2 << std::endl;
      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      // skip resize for count_only dest Case1
      p_2.old_size_0 = 0;
      {  // Resize HasLiker
        uint32_t sum_HasLiker = total_1;
        auto& dest_HasLiker = get_relation_by_schema<HasLiker, NEW_VER>(db);
        uint32_t base_offset_HasLiker = static_cast<uint32_t>(dest_HasLiker.size());
        if (sum_HasLiker > 0) dest_HasLiker.resize_interned_columns(base_offset_HasLiker + sum_HasLiker);
        uint32_t running_offset_HasLiker = base_offset_HasLiker;
        p_1.old_size_0 = running_offset_HasLiker;
        running_offset_HasLiker += total_1;
      }
      {  // Resize HasReply
        uint32_t sum_HasReply = total_0;
        auto& dest_HasReply = get_relation_by_schema<HasReply, NEW_VER>(db);
        uint32_t base_offset_HasReply = static_cast<uint32_t>(dest_HasReply.size());
        if (sum_HasReply > 0) dest_HasReply.resize_interned_columns(base_offset_HasReply + sum_HasReply);
        uint32_t running_offset_HasReply = base_offset_HasReply;
        p_0.old_size_0 = running_offset_HasReply;
        running_offset_HasReply += total_0;
      }

      if (total_0 > 0) JitRunner_MarkHasReply::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_MarkHasLiker::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      // skip materialize for count_only rule Case1Both
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HasReply, NEW_VER, HasReply_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HasReply, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HasLiker, NEW_VER, HasLiker_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HasLiker, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Case1, NEW_VER, Case1_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, NEW_VER>, SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>, SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Case1, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Case1, std::integer_sequence<int, 0, 1, 2, 3, 4>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_4(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

    using Case2_canonical_spec_t = SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using Case3_canonical_spec_t = SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using Case4_canonical_spec_t = SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(3);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReplyOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReplyOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Likes, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Likes, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasTag, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasCreator, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasReply, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HasLiker, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_Case2ReplyOnly::execute_fused(db, 0);
      JitRunner_Case3LikeOnly::execute_fused(db, 0);
      JitRunner_Case4Neither::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_Case2ReplyOnly::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Case3LikeOnly::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_Case4Neither::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_Case2ReplyOnly::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Case3LikeOnly::launch_count(p_1, _stream_pool.get(1));
      JitRunner_Case4Neither::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_Case2ReplyOnly::scan_only(p_0, _stream_pool.get(0));
      JitRunner_Case3LikeOnly::scan_only(p_1, _stream_pool.get(1));
      JitRunner_Case4Neither::scan_only(p_2, _stream_pool.get(2));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_Case2ReplyOnly::read_total(p_0);
      uint32_t total_1 = JitRunner_Case3LikeOnly::read_total(p_1);
      uint32_t total_2 = JitRunner_Case4Neither::read_total(p_2);

      std::cout << " >>>>>>>>>>>>>>>>> Case2ReplyOnly count: " << total_0 << std::endl;
      std::cout << " >>>>>>>>>>>>>>>>> Case3LikeOnly count: " << total_1 << std::endl;
      std::cout << " >>>>>>>>>>>>>>>>> Case4Neither count: " << total_2 << std::endl;
      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      // skip resize for count_only dest Case2
      p_0.old_size_0 = 0;
      // skip resize for count_only dest Case3
      p_1.old_size_0 = 0;
      // skip resize for count_only dest Case4
      p_2.old_size_0 = 0;

      // skip materialize for count_only rule Case2ReplyOnly
      // skip materialize for count_only rule Case3LikeOnly
      // skip materialize for count_only rule Case4Neither
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Case2, NEW_VER, Case2_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Case2, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Case3, NEW_VER, Case3_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Case3, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Case4, NEW_VER, Case4_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Case4, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_5(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Case2, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_6(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Case3, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_7(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Case4, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
