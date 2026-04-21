template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using Knows_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using IsLocatedIn_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using IsPartOf_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(3);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<KnowsInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedInInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db,
                                                                                                0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOfInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_KnowsLoad::execute_fused(db, 0);
    JitRunner_LocLoad::execute_fused(db, 0);
    JitRunner_PartOfLoad::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_KnowsLoad::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_LocLoad::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_PartOfLoad::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_KnowsLoad::launch_count(p_0, _stream_pool.get(0));
      JitRunner_LocLoad::launch_count(p_1, _stream_pool.get(1));
      JitRunner_PartOfLoad::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_LocLoad::scan_only(p_1, _stream_pool.get(1));
      JitRunner_PartOfLoad::scan_only(p_2, _stream_pool.get(2));
      JitRunner_KnowsLoad::scan_only(p_0, _stream_pool.get(0));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_1 = JitRunner_LocLoad::read_total(p_1);
      uint32_t total_2 = JitRunner_PartOfLoad::read_total(p_2);
      uint32_t total_0 = JitRunner_KnowsLoad::read_total(p_0);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize IsLocatedIn
        uint32_t sum_IsLocatedIn = total_1;
        auto& dest_IsLocatedIn = get_relation_by_schema<IsLocatedIn, NEW_VER>(db);
        uint32_t base_offset_IsLocatedIn = static_cast<uint32_t>(dest_IsLocatedIn.size());
        if (sum_IsLocatedIn > 0)
          dest_IsLocatedIn.resize_interned_columns(base_offset_IsLocatedIn + sum_IsLocatedIn);
        uint32_t running_offset_IsLocatedIn = base_offset_IsLocatedIn;
        p_1.old_size_0 = running_offset_IsLocatedIn;
        running_offset_IsLocatedIn += total_1;
      }
      {  // Resize IsPartOf
        uint32_t sum_IsPartOf = total_2;
        auto& dest_IsPartOf = get_relation_by_schema<IsPartOf, NEW_VER>(db);
        uint32_t base_offset_IsPartOf = static_cast<uint32_t>(dest_IsPartOf.size());
        if (sum_IsPartOf > 0)
          dest_IsPartOf.resize_interned_columns(base_offset_IsPartOf + sum_IsPartOf);
        uint32_t running_offset_IsPartOf = base_offset_IsPartOf;
        p_2.old_size_0 = running_offset_IsPartOf;
        running_offset_IsPartOf += total_2;
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

      if (total_0 > 0)
        JitRunner_KnowsLoad::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_LocLoad::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0)
        JitRunner_PartOfLoad::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Knows, NEW_VER, Knows_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Knows, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<IsLocatedIn, NEW_VER, IsLocatedIn_canonical_spec_t>(
      db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<IsLocatedIn, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<IsPartOf, NEW_VER, IsPartOf_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<IsPartOf, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_4(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

  using Triangle_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsLocatedIn, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<IsPartOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_LabeledTriangle::execute_fused(db, 0);
  else
    JitRunner_LabeledTriangle::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Triangle, NEW_VER, Triangle_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Triangle, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_5(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
