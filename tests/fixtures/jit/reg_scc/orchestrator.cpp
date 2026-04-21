  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

    using RegDefUseDefUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using RegDefUseLiveVarUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsed, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsed, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsed, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsedInBlock, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_RegDefUsed1::execute_fused(db, 0);
      JitRunner_LiveVarUsed1::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_RegDefUsed1::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_LiveVarUsed1::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_RegDefUsed1::launch_count(p_0, _stream_pool.get(0));
      JitRunner_LiveVarUsed1::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_RegDefUsed1::scan_only(p_0, _stream_pool.get(0));
      JitRunner_LiveVarUsed1::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_RegDefUsed1::read_total(p_0);
      uint32_t total_1 = JitRunner_LiveVarUsed1::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize RegDefUseDefUsed
        uint32_t sum_RegDefUseDefUsed = total_0;
        auto& dest_RegDefUseDefUsed = get_relation_by_schema<RegDefUseDefUsed, NEW_VER>(db);
        uint32_t base_offset_RegDefUseDefUsed = static_cast<uint32_t>(dest_RegDefUseDefUsed.size());
        if (sum_RegDefUseDefUsed > 0) dest_RegDefUseDefUsed.resize_interned_columns(base_offset_RegDefUseDefUsed + sum_RegDefUseDefUsed);
        uint32_t running_offset_RegDefUseDefUsed = base_offset_RegDefUseDefUsed;
        p_0.old_size_0 = running_offset_RegDefUseDefUsed;
        running_offset_RegDefUseDefUsed += total_0;
      }
      {  // Resize RegDefUseLiveVarUsed
        uint32_t sum_RegDefUseLiveVarUsed = total_1;
        auto& dest_RegDefUseLiveVarUsed = get_relation_by_schema<RegDefUseLiveVarUsed, NEW_VER>(db);
        uint32_t base_offset_RegDefUseLiveVarUsed = static_cast<uint32_t>(dest_RegDefUseLiveVarUsed.size());
        if (sum_RegDefUseLiveVarUsed > 0) dest_RegDefUseLiveVarUsed.resize_interned_columns(base_offset_RegDefUseLiveVarUsed + sum_RegDefUseLiveVarUsed);
        uint32_t running_offset_RegDefUseLiveVarUsed = base_offset_RegDefUseLiveVarUsed;
        p_1.old_size_0 = running_offset_RegDefUseLiveVarUsed;
        running_offset_RegDefUseLiveVarUsed += total_1;
      }

      if (total_0 > 0) JitRunner_RegDefUsed1::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_LiveVarUsed1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseDefUsed, NEW_VER, RegDefUseDefUsed_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseDefUsed, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseLiveVarUsed, NEW_VER, RegDefUseLiveVarUsed_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarUsed, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    using RegDefUseDefUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>;
    using RegDefUseLiveVarAtBlockEnd_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>;
    using RegDefUseLiveVarAtPriorUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>;
    using RegDefUseLiveVarUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using RegDefUseReturnValUsed_canonical_spec_t = SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarDef, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnBlockEnd, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnBlockEnd, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnBlockEnd, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseBlockLastDef, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectCall, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArchReturnReg, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectCall, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnBlockEnd, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnBlockEnd, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsedInBlock, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsedInBlock, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseUsedInBlock, std::integer_sequence<int, 0, 2, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<BlockNext, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<BlockNext, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<BlockNext, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<BlockNext, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<RegDefUseDefUsed, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{2, 1, 0, 3}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<RegDefUseLiveVarAtBlockEnd, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{1, 2, 0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<RegDefUseLiveVarAtPriorUsed, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 2, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<RegDefUseLiveVarUsed, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1, 2, 3}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<RegDefUseReturnValUsed, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{1, 0, 2, 3, 4}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<RegDefUseDefUsed, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{2, 1, 0, 3}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " RegDefUseDefUsed=" << _di.root().degree(); } catch (...) { std::cerr << " RegDefUseDefUsed=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<RegDefUseLiveVarAtBlockEnd, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{1, 2, 0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " RegDefUseLiveVarAtBlockEnd=" << _di.root().degree(); } catch (...) { std::cerr << " RegDefUseLiveVarAtBlockEnd=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<RegDefUseLiveVarAtPriorUsed, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 2, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " RegDefUseLiveVarAtPriorUsed=" << _di.root().degree(); } catch (...) { std::cerr << " RegDefUseLiveVarAtPriorUsed=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<RegDefUseLiveVarUsed, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1, 2, 3}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " RegDefUseLiveVarUsed=" << _di.root().degree(); } catch (...) { std::cerr << " RegDefUseLiveVarUsed=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<RegDefUseReturnValUsed, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{1, 0, 2, 3, 4}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " RegDefUseReturnValUsed=" << _di.root().degree(); } catch (...) { std::cerr << " RegDefUseReturnValUsed=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      static SRDatalog::GPU::StreamPool _stream_pool;
      _stream_pool.ensure(11);

      if (_tail_mode) {
        JitRunner_RegDefUsed2_D0::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RegDefUsed2_D2::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RegDefUsed3_D0::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RegDefUsed3_D1::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RegDefUsed3_D2::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RegDefUsed4_D0::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_RetValUsed_D1::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_LiveVarUsed2_D2::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_LiveVarAtPriorUsed_D0::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_LiveVarAtBlockEnd1_D1::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_LiveVarAtBlockEnd2_D0::execute_fused(db, static_cast<uint32_t>(iter));
      } else {
      // === ParallelGroup (stream-parallel, 11 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_RegDefUsed2_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_RegDefUsed2_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));
        auto p_2 = JitRunner_RegDefUsed3_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(2));
        auto p_3 = JitRunner_RegDefUsed3_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(3));
        auto p_4 = JitRunner_RegDefUsed3_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(4));
        auto p_5 = JitRunner_RegDefUsed4_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(5));
        auto p_6 = JitRunner_RetValUsed_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(6));
        auto p_7 = JitRunner_LiveVarUsed2_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(7));
        auto p_8 = JitRunner_LiveVarAtPriorUsed_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(8));
        auto p_9 = JitRunner_LiveVarAtBlockEnd1_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(9));
        auto p_10 = JitRunner_LiveVarAtBlockEnd2_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(10));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_RegDefUseDefUsed_size = p_0.num_threads + p_1.num_threads + p_2.num_threads + p_3.num_threads + p_4.num_threads + p_5.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_RegDefUseDefUsed(shared_buf_RegDefUseDefUsed_size);
        uint32_t off_RegDefUseDefUsed_0 = 0;
        p_0.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_0;
        uint32_t off_RegDefUseDefUsed_1 = off_RegDefUseDefUsed_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_1;
        uint32_t off_RegDefUseDefUsed_2 = off_RegDefUseDefUsed_1 + p_1.num_threads;
        p_2.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_2;
        uint32_t off_RegDefUseDefUsed_3 = off_RegDefUseDefUsed_2 + p_2.num_threads;
        p_3.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_3;
        uint32_t off_RegDefUseDefUsed_4 = off_RegDefUseDefUsed_3 + p_3.num_threads;
        p_4.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_4;
        uint32_t off_RegDefUseDefUsed_5 = off_RegDefUseDefUsed_4 + p_4.num_threads;
        p_5.thread_counts_ptr = shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_5;
        uint32_t shared_buf_RegDefUseLiveVarAtBlockEnd_size = p_9.num_threads + p_10.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_RegDefUseLiveVarAtBlockEnd(shared_buf_RegDefUseLiveVarAtBlockEnd_size);
        uint32_t off_RegDefUseLiveVarAtBlockEnd_0 = 0;
        p_9.thread_counts_ptr = shared_buf_RegDefUseLiveVarAtBlockEnd.data() + off_RegDefUseLiveVarAtBlockEnd_0;
        uint32_t off_RegDefUseLiveVarAtBlockEnd_1 = off_RegDefUseLiveVarAtBlockEnd_0 + p_9.num_threads;
        p_10.thread_counts_ptr = shared_buf_RegDefUseLiveVarAtBlockEnd.data() + off_RegDefUseLiveVarAtBlockEnd_1;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_RegDefUsed2_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_RegDefUsed2_D2::launch_count(p_1, _stream_pool.get(1));
        JitRunner_RegDefUsed3_D0::launch_count(p_2, _stream_pool.get(2));
        JitRunner_RegDefUsed3_D1::launch_count(p_3, _stream_pool.get(3));
        JitRunner_RegDefUsed3_D2::launch_count(p_4, _stream_pool.get(4));
        JitRunner_RegDefUsed4_D0::launch_count(p_5, _stream_pool.get(5));
        JitRunner_RetValUsed_D1::launch_count(p_6, _stream_pool.get(6));
        JitRunner_LiveVarUsed2_D2::launch_count(p_7, _stream_pool.get(7));
        JitRunner_LiveVarAtPriorUsed_D0::launch_count(p_8, _stream_pool.get(8));
        JitRunner_LiveVarAtBlockEnd1_D1::launch_count(p_9, _stream_pool.get(9));
        JitRunner_LiveVarAtBlockEnd2_D0::launch_count(p_10, _stream_pool.get(10));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_RegDefUseDefUsed.data(), shared_buf_RegDefUseDefUsed.data() + shared_buf_RegDefUseDefUsed_size, shared_buf_RegDefUseDefUsed.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(9)), shared_buf_RegDefUseLiveVarAtBlockEnd.data(), shared_buf_RegDefUseLiveVarAtBlockEnd.data() + shared_buf_RegDefUseLiveVarAtBlockEnd_size, shared_buf_RegDefUseLiveVarAtBlockEnd.data(), 0, thrust::plus<uint32_t>());
        JitRunner_LiveVarAtPriorUsed_D0::scan_only(p_8, _stream_pool.get(8));
        JitRunner_LiveVarUsed2_D2::scan_only(p_7, _stream_pool.get(7));
        JitRunner_RetValUsed_D1::scan_only(p_6, _stream_pool.get(6));

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for RegDefUseDefUsed (6 rules)
        uint32_t bnd_RegDefUseDefUsed[7];
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[0], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[1], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[2], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[3], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[4], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_4, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[5], shared_buf_RegDefUseDefUsed.data() + off_RegDefUseDefUsed_5, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseDefUsed[6], shared_buf_RegDefUseDefUsed.data() + shared_buf_RegDefUseDefUsed_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_RegDefUseDefUsed[1] - bnd_RegDefUseDefUsed[0];
        uint32_t total_1 = bnd_RegDefUseDefUsed[2] - bnd_RegDefUseDefUsed[1];
        uint32_t total_2 = bnd_RegDefUseDefUsed[3] - bnd_RegDefUseDefUsed[2];
        uint32_t total_3 = bnd_RegDefUseDefUsed[4] - bnd_RegDefUseDefUsed[3];
        uint32_t total_4 = bnd_RegDefUseDefUsed[5] - bnd_RegDefUseDefUsed[4];
        uint32_t total_5 = bnd_RegDefUseDefUsed[6] - bnd_RegDefUseDefUsed[5];
        // Read boundary values for RegDefUseLiveVarAtBlockEnd (2 rules)
        uint32_t bnd_RegDefUseLiveVarAtBlockEnd[3];
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseLiveVarAtBlockEnd[0], shared_buf_RegDefUseLiveVarAtBlockEnd.data() + off_RegDefUseLiveVarAtBlockEnd_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseLiveVarAtBlockEnd[1], shared_buf_RegDefUseLiveVarAtBlockEnd.data() + off_RegDefUseLiveVarAtBlockEnd_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_RegDefUseLiveVarAtBlockEnd[2], shared_buf_RegDefUseLiveVarAtBlockEnd.data() + shared_buf_RegDefUseLiveVarAtBlockEnd_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_9 = bnd_RegDefUseLiveVarAtBlockEnd[1] - bnd_RegDefUseLiveVarAtBlockEnd[0];
        uint32_t total_10 = bnd_RegDefUseLiveVarAtBlockEnd[2] - bnd_RegDefUseLiveVarAtBlockEnd[1];
        uint32_t total_8 = JitRunner_LiveVarAtPriorUsed_D0::read_total(p_8);
        uint32_t total_7 = JitRunner_LiveVarUsed2_D2::read_total(p_7);
        uint32_t total_6 = JitRunner_RetValUsed_D1::read_total(p_6);

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize RegDefUseDefUsed
          uint32_t sum_RegDefUseDefUsed = total_0 + total_1 + total_2 + total_3 + total_4 + total_5;
          auto& dest_RegDefUseDefUsed = get_relation_by_schema<RegDefUseDefUsed, NEW_VER>(db);
          uint32_t base_offset_RegDefUseDefUsed = static_cast<uint32_t>(dest_RegDefUseDefUsed.size());
          if (sum_RegDefUseDefUsed > 0) dest_RegDefUseDefUsed.resize_interned_columns(base_offset_RegDefUseDefUsed + sum_RegDefUseDefUsed);
          p_0.old_size_0 = base_offset_RegDefUseDefUsed;
          p_1.old_size_0 = base_offset_RegDefUseDefUsed;
          p_2.old_size_0 = base_offset_RegDefUseDefUsed;
          p_3.old_size_0 = base_offset_RegDefUseDefUsed;
          p_4.old_size_0 = base_offset_RegDefUseDefUsed;
          p_5.old_size_0 = base_offset_RegDefUseDefUsed;
        }
        {  // Resize RegDefUseLiveVarAtBlockEnd
          uint32_t sum_RegDefUseLiveVarAtBlockEnd = total_9 + total_10;
          auto& dest_RegDefUseLiveVarAtBlockEnd = get_relation_by_schema<RegDefUseLiveVarAtBlockEnd, NEW_VER>(db);
          uint32_t base_offset_RegDefUseLiveVarAtBlockEnd = static_cast<uint32_t>(dest_RegDefUseLiveVarAtBlockEnd.size());
          if (sum_RegDefUseLiveVarAtBlockEnd > 0) dest_RegDefUseLiveVarAtBlockEnd.resize_interned_columns(base_offset_RegDefUseLiveVarAtBlockEnd + sum_RegDefUseLiveVarAtBlockEnd);
          p_9.old_size_0 = base_offset_RegDefUseLiveVarAtBlockEnd;
          p_10.old_size_0 = base_offset_RegDefUseLiveVarAtBlockEnd;
        }
        {  // Resize RegDefUseLiveVarAtPriorUsed
          uint32_t sum_RegDefUseLiveVarAtPriorUsed = total_8;
          auto& dest_RegDefUseLiveVarAtPriorUsed = get_relation_by_schema<RegDefUseLiveVarAtPriorUsed, NEW_VER>(db);
          uint32_t base_offset_RegDefUseLiveVarAtPriorUsed = static_cast<uint32_t>(dest_RegDefUseLiveVarAtPriorUsed.size());
          if (sum_RegDefUseLiveVarAtPriorUsed > 0) dest_RegDefUseLiveVarAtPriorUsed.resize_interned_columns(base_offset_RegDefUseLiveVarAtPriorUsed + sum_RegDefUseLiveVarAtPriorUsed);
          uint32_t running_offset_RegDefUseLiveVarAtPriorUsed = base_offset_RegDefUseLiveVarAtPriorUsed;
          p_8.old_size_0 = running_offset_RegDefUseLiveVarAtPriorUsed;
          running_offset_RegDefUseLiveVarAtPriorUsed += total_8;
        }
        {  // Resize RegDefUseLiveVarUsed
          uint32_t sum_RegDefUseLiveVarUsed = total_7;
          auto& dest_RegDefUseLiveVarUsed = get_relation_by_schema<RegDefUseLiveVarUsed, NEW_VER>(db);
          uint32_t base_offset_RegDefUseLiveVarUsed = static_cast<uint32_t>(dest_RegDefUseLiveVarUsed.size());
          if (sum_RegDefUseLiveVarUsed > 0) dest_RegDefUseLiveVarUsed.resize_interned_columns(base_offset_RegDefUseLiveVarUsed + sum_RegDefUseLiveVarUsed);
          uint32_t running_offset_RegDefUseLiveVarUsed = base_offset_RegDefUseLiveVarUsed;
          p_7.old_size_0 = running_offset_RegDefUseLiveVarUsed;
          running_offset_RegDefUseLiveVarUsed += total_7;
        }
        {  // Resize RegDefUseReturnValUsed
          uint32_t sum_RegDefUseReturnValUsed = total_6;
          auto& dest_RegDefUseReturnValUsed = get_relation_by_schema<RegDefUseReturnValUsed, NEW_VER>(db);
          uint32_t base_offset_RegDefUseReturnValUsed = static_cast<uint32_t>(dest_RegDefUseReturnValUsed.size());
          if (sum_RegDefUseReturnValUsed > 0) dest_RegDefUseReturnValUsed.resize_interned_columns(base_offset_RegDefUseReturnValUsed + sum_RegDefUseReturnValUsed);
          uint32_t running_offset_RegDefUseReturnValUsed = base_offset_RegDefUseReturnValUsed;
          p_6.old_size_0 = running_offset_RegDefUseReturnValUsed;
          running_offset_RegDefUseReturnValUsed += total_6;
        }

        if (total_0 > 0) JitRunner_RegDefUsed2_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0) JitRunner_RegDefUsed2_D2::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        if (total_2 > 0) JitRunner_RegDefUsed3_D0::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
        if (total_3 > 0) JitRunner_RegDefUsed3_D1::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
        if (total_4 > 0) JitRunner_RegDefUsed3_D2::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
        if (total_5 > 0) JitRunner_RegDefUsed4_D0::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
        if (total_6 > 0) JitRunner_RetValUsed_D1::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
        if (total_7 > 0) JitRunner_LiveVarUsed2_D2::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
        if (total_8 > 0) JitRunner_LiveVarAtPriorUsed_D0::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
        if (total_9 > 0) JitRunner_LiveVarAtBlockEnd1_D1::launch_materialize(db, p_9, total_9, _stream_pool.get(9));
        if (total_10 > 0) JitRunner_LiveVarAtBlockEnd2_D0::launch_materialize(db, p_10, total_10, _stream_pool.get(10));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        _stream_pool.record_event(2);
        _stream_pool.record_event(3);
        _stream_pool.record_event(4);
        _stream_pool.record_event(5);
        _stream_pool.record_event(6);
        _stream_pool.record_event(7);
        _stream_pool.record_event(8);
        _stream_pool.record_event(9);
        _stream_pool.record_event(10);
        nvtxRangePop();  // join_pipeline
      }
      }
      GPU_DEVICE_SYNCHRONIZE(); // sync all materialize streams
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseDefUsed, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseDefUsed, NEW_VER, RegDefUseDefUsed_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseDefUsed, NEW_VER>(db);
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 1, 2, 0, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarAtBlockEnd, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseLiveVarAtBlockEnd, NEW_VER, RegDefUseLiveVarAtBlockEnd_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarAtBlockEnd, NEW_VER>(db);
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarAtPriorUsed, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseLiveVarAtPriorUsed, NEW_VER, RegDefUseLiveVarAtPriorUsed_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarAtPriorUsed, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarUsed, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseLiveVarUsed, NEW_VER, RegDefUseLiveVarUsed_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseLiveVarUsed, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseReturnValUsed, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<RegDefUseReturnValUsed, NEW_VER, RegDefUseReturnValUsed_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, NEW_VER>, SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>, SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<RegDefUseReturnValUsed, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, DELTA_VER>, SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 2, 0, 3, 4>, DELTA_VER>>(db);
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_4(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseDefUsed, std::integer_sequence<int, 2, 1, 0, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_5(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtBlockEnd, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_6(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarAtPriorUsed, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_7(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseLiveVarUsed, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_8(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<RegDefUseReturnValUsed, std::integer_sequence<int, 1, 0, 2, 3, 4>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
