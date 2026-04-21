template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using Knows_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>;
  using Knows2_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using HasInterest_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(3);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<KnowsInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<KnowsInput2, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterestInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  if (_tail_mode) {
    JitRunner_KnowsLoad::execute_fused(db, 0);
    JitRunner_Knows2Load::execute_fused(db, 0);
    JitRunner_InterestLoad::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_KnowsLoad::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Knows2Load::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_InterestLoad::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_KnowsLoad::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Knows2Load::launch_count(p_1, _stream_pool.get(1));
      JitRunner_InterestLoad::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_InterestLoad::scan_only(p_2, _stream_pool.get(2));
      JitRunner_KnowsLoad::scan_only(p_0, _stream_pool.get(0));
      JitRunner_Knows2Load::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_2 = JitRunner_InterestLoad::read_total(p_2);
      uint32_t total_0 = JitRunner_KnowsLoad::read_total(p_0);
      uint32_t total_1 = JitRunner_Knows2Load::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize HasInterest
        uint32_t sum_HasInterest = total_2;
        auto& dest_HasInterest = get_relation_by_schema<HasInterest, NEW_VER>(db);
        uint32_t base_offset_HasInterest = static_cast<uint32_t>(dest_HasInterest.size());
        if (sum_HasInterest > 0)
          dest_HasInterest.resize_interned_columns(base_offset_HasInterest + sum_HasInterest);
        uint32_t running_offset_HasInterest = base_offset_HasInterest;
        p_2.old_size_0 = running_offset_HasInterest;
        running_offset_HasInterest += total_2;
      }
      {  // Resize Knows
        uint32_t sum_Knows = total_0;
        auto& dest_Knows = get_relation_by_schema<Knows, NEW_VER>(db);
        uint32_t base_offset_Knows = static_cast<uint32_t>(dest_Knows.size());
        if (sum_Knows > 0)
          dest_Knows.resize_interned_columns(base_offset_Knows + sum_Knows);
        uint32_t running_offset_Knows = base_offset_Knows;
        p_0.old_size_0 = running_offset_Knows;
        running_offset_Knows += total_0;
      }
      {  // Resize Knows2
        uint32_t sum_Knows2 = total_1;
        auto& dest_Knows2 = get_relation_by_schema<Knows2, NEW_VER>(db);
        uint32_t base_offset_Knows2 = static_cast<uint32_t>(dest_Knows2.size());
        if (sum_Knows2 > 0)
          dest_Knows2.resize_interned_columns(base_offset_Knows2 + sum_Knows2);
        uint32_t running_offset_Knows2 = base_offset_Knows2;
        p_1.old_size_0 = running_offset_Knows2;
        running_offset_Knows2 += total_1;
      }

      if (total_0 > 0)
        JitRunner_KnowsLoad::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_Knows2Load::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0)
        JitRunner_InterestLoad::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Knows, NEW_VER, Knows_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Knows, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Knows2, NEW_VER, Knows2_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Knows2, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<HasInterest, NEW_VER, HasInterest_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<HasInterest, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_4(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);

  using PathBG_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows2, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<HasInterest, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  JitRunner_TwoHopBG::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<PathBG, NEW_VER, PathBG_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<PathBG, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_5(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<PathBG, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
