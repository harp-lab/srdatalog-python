template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

  using OutP_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using OutQ_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  bool _tail_mode = false;

  static SRDatalog::GPU::StreamPool _stream_pool;
  _stream_pool.ensure(2);

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<PInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<QInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  if (_tail_mode) {
    JitRunner_LoadP::execute_fused(db, 0);
    JitRunner_LoadQ::execute_fused(db, 0);
  } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_LoadP::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_LoadQ::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_LoadP::launch_count(p_0, _stream_pool.get(0));
      JitRunner_LoadQ::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_LoadP::scan_only(p_0, _stream_pool.get(0));
      JitRunner_LoadQ::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_LoadP::read_total(p_0);
      uint32_t total_1 = JitRunner_LoadQ::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize OutP
        uint32_t sum_OutP = total_0;
        auto& dest_OutP = get_relation_by_schema<OutP, NEW_VER>(db);
        uint32_t base_offset_OutP = static_cast<uint32_t>(dest_OutP.size());
        if (sum_OutP > 0)
          dest_OutP.resize_interned_columns(base_offset_OutP + sum_OutP);
        uint32_t running_offset_OutP = base_offset_OutP;
        p_0.old_size_0 = running_offset_OutP;
        running_offset_OutP += total_0;
      }
      {  // Resize OutQ
        uint32_t sum_OutQ = total_1;
        auto& dest_OutQ = get_relation_by_schema<OutQ, NEW_VER>(db);
        uint32_t base_offset_OutQ = static_cast<uint32_t>(dest_OutQ.size());
        if (sum_OutQ > 0)
          dest_OutQ.resize_interned_columns(base_offset_OutQ + sum_OutQ);
        uint32_t running_offset_OutQ = base_offset_OutQ;
        p_1.old_size_0 = running_offset_OutQ;
        running_offset_OutQ += total_1;
      }

      if (total_0 > 0)
        JitRunner_LoadP::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0)
        JitRunner_LoadQ::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
  }
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<OutP, NEW_VER, OutP_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<OutP, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<OutQ, NEW_VER, OutQ_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<OutQ, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>,
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_3(DB& db, std::size_t max_iterations) {
  using OutP_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using OutQ_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;

  // Build all required indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<UInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<CInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<SInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<SInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<RInput, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

  bool _tail_mode = false;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    if (iter > 0) {
      std::size_t total_new_facts = 0;
      {
        auto& delta_rel = get_relation_by_schema<OutP, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 1}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      {
        auto& delta_rel = get_relation_by_schema<OutQ, DELTA_VER>(db);
        SRDatalog::IndexSpec spec{{0, 1, 2}};
        try {
          auto& delta_idx = delta_rel.get_index(spec);
          total_new_facts += delta_idx.root().degree();
        } catch (...) {
        }
      }
      if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
        std::cerr << "[iter " << iter << "]";
        {
          auto& _dr = get_relation_by_schema<OutP, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " OutP=" << _di.root().degree();
          } catch (...) {
            std::cerr << " OutP=?";
          }
        }
        {
          auto& _dr = get_relation_by_schema<OutQ, DELTA_VER>(db);
          SRDatalog::IndexSpec _sp{{0, 1, 2}};
          try {
            auto& _di = _dr.get_index(_sp);
            std::cerr << " OutQ=" << _di.root().degree();
          } catch (...) {
            std::cerr << " OutQ=?";
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
    _stream_pool.ensure(11);

    if (_tail_mode) {
      JitRunner_TC_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_TC_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_PropQ_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_PropQ_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3a_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3a_D2::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3b_D1::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3b_D2::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_PropQS_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3c_D0::execute_fused(db, static_cast<uint32_t>(iter));
      JitRunner_Join3c_D2::execute_fused(db, static_cast<uint32_t>(iter));
    } else {
      // === ParallelGroup (stream-parallel, 11 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_TC_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_TC_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));
        auto p_2 = JitRunner_PropQ_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(2));
        auto p_3 = JitRunner_PropQ_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(3));
        auto p_4 = JitRunner_Join3a_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(4));
        auto p_5 = JitRunner_Join3a_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(5));
        auto p_6 = JitRunner_Join3b_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(6));
        auto p_7 = JitRunner_Join3b_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(7));
        auto p_8 = JitRunner_PropQS_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(8));
        auto p_9 = JitRunner_Join3c_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(9));
        auto p_10 =
            JitRunner_Join3c_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(10));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_OutP_size = p_0.num_threads + p_1.num_threads + p_4.num_threads +
                                        p_5.num_threads + p_6.num_threads + p_7.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_OutP(shared_buf_OutP_size);
        uint32_t off_OutP_0 = 0;
        p_0.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_0;
        uint32_t off_OutP_1 = off_OutP_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_1;
        uint32_t off_OutP_2 = off_OutP_1 + p_1.num_threads;
        p_4.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_2;
        uint32_t off_OutP_3 = off_OutP_2 + p_4.num_threads;
        p_5.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_3;
        uint32_t off_OutP_4 = off_OutP_3 + p_5.num_threads;
        p_6.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_4;
        uint32_t off_OutP_5 = off_OutP_4 + p_6.num_threads;
        p_7.thread_counts_ptr = shared_buf_OutP.data() + off_OutP_5;
        uint32_t shared_buf_OutQ_size = p_2.num_threads + p_3.num_threads + p_8.num_threads +
                                        p_9.num_threads + p_10.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_OutQ(shared_buf_OutQ_size);
        uint32_t off_OutQ_0 = 0;
        p_2.thread_counts_ptr = shared_buf_OutQ.data() + off_OutQ_0;
        uint32_t off_OutQ_1 = off_OutQ_0 + p_2.num_threads;
        p_3.thread_counts_ptr = shared_buf_OutQ.data() + off_OutQ_1;
        uint32_t off_OutQ_2 = off_OutQ_1 + p_3.num_threads;
        p_8.thread_counts_ptr = shared_buf_OutQ.data() + off_OutQ_2;
        uint32_t off_OutQ_3 = off_OutQ_2 + p_8.num_threads;
        p_9.thread_counts_ptr = shared_buf_OutQ.data() + off_OutQ_3;
        uint32_t off_OutQ_4 = off_OutQ_3 + p_9.num_threads;
        p_10.thread_counts_ptr = shared_buf_OutQ.data() + off_OutQ_4;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_TC_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_TC_D1::launch_count(p_1, _stream_pool.get(1));
        JitRunner_PropQ_D0::launch_count(p_2, _stream_pool.get(2));
        JitRunner_PropQ_D1::launch_count(p_3, _stream_pool.get(3));
        JitRunner_Join3a_D0::launch_count(p_4, _stream_pool.get(4));
        JitRunner_Join3a_D2::launch_count(p_5, _stream_pool.get(5));
        JitRunner_Join3b_D1::launch_count(p_6, _stream_pool.get(6));
        JitRunner_Join3b_D2::launch_count(p_7, _stream_pool.get(7));
        JitRunner_PropQS_D0::launch_count(p_8, _stream_pool.get(8));
        JitRunner_Join3c_D0::launch_count(p_9, _stream_pool.get(9));
        JitRunner_Join3c_D2::launch_count(p_10, _stream_pool.get(10));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_OutP.data(),
                               shared_buf_OutP.data() + shared_buf_OutP_size,
                               shared_buf_OutP.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(2)), shared_buf_OutQ.data(),
                               shared_buf_OutQ.data() + shared_buf_OutQ_size,
                               shared_buf_OutQ.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for OutP (6 rules)
        uint32_t bnd_OutP[7];
        GPU_MEMCPY_ASYNC(&bnd_OutP[0], shared_buf_OutP.data() + off_OutP_0, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[1], shared_buf_OutP.data() + off_OutP_1, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[2], shared_buf_OutP.data() + off_OutP_2, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[3], shared_buf_OutP.data() + off_OutP_3, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[4], shared_buf_OutP.data() + off_OutP_4, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[5], shared_buf_OutP.data() + off_OutP_5, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutP[6], shared_buf_OutP.data() + shared_buf_OutP_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_OutP[1] - bnd_OutP[0];
        uint32_t total_1 = bnd_OutP[2] - bnd_OutP[1];
        uint32_t total_4 = bnd_OutP[3] - bnd_OutP[2];
        uint32_t total_5 = bnd_OutP[4] - bnd_OutP[3];
        uint32_t total_6 = bnd_OutP[5] - bnd_OutP[4];
        uint32_t total_7 = bnd_OutP[6] - bnd_OutP[5];
        // Read boundary values for OutQ (5 rules)
        uint32_t bnd_OutQ[6];
        GPU_MEMCPY_ASYNC(&bnd_OutQ[0], shared_buf_OutQ.data() + off_OutQ_0, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutQ[1], shared_buf_OutQ.data() + off_OutQ_1, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutQ[2], shared_buf_OutQ.data() + off_OutQ_2, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutQ[3], shared_buf_OutQ.data() + off_OutQ_3, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutQ[4], shared_buf_OutQ.data() + off_OutQ_4, sizeof(uint32_t),
                         GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_OutQ[5], shared_buf_OutQ.data() + shared_buf_OutQ_size - 1,
                         sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_2 = bnd_OutQ[1] - bnd_OutQ[0];
        uint32_t total_3 = bnd_OutQ[2] - bnd_OutQ[1];
        uint32_t total_8 = bnd_OutQ[3] - bnd_OutQ[2];
        uint32_t total_9 = bnd_OutQ[4] - bnd_OutQ[3];
        uint32_t total_10 = bnd_OutQ[5] - bnd_OutQ[4];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize OutP
          uint32_t sum_OutP = total_0 + total_1 + total_4 + total_5 + total_6 + total_7;
          auto& dest_OutP = get_relation_by_schema<OutP, NEW_VER>(db);
          uint32_t base_offset_OutP = static_cast<uint32_t>(dest_OutP.size());
          if (sum_OutP > 0)
            dest_OutP.resize_interned_columns(base_offset_OutP + sum_OutP);
          p_0.old_size_0 = base_offset_OutP;
          p_1.old_size_0 = base_offset_OutP;
          p_4.old_size_0 = base_offset_OutP;
          p_5.old_size_0 = base_offset_OutP;
          p_6.old_size_0 = base_offset_OutP;
          p_7.old_size_0 = base_offset_OutP;
        }
        {  // Resize OutQ
          uint32_t sum_OutQ = total_2 + total_3 + total_8 + total_9 + total_10;
          auto& dest_OutQ = get_relation_by_schema<OutQ, NEW_VER>(db);
          uint32_t base_offset_OutQ = static_cast<uint32_t>(dest_OutQ.size());
          if (sum_OutQ > 0)
            dest_OutQ.resize_interned_columns(base_offset_OutQ + sum_OutQ);
          p_2.old_size_0 = base_offset_OutQ;
          p_3.old_size_0 = base_offset_OutQ;
          p_8.old_size_0 = base_offset_OutQ;
          p_9.old_size_0 = base_offset_OutQ;
          p_10.old_size_0 = base_offset_OutQ;
        }

        if (total_0 > 0)
          JitRunner_TC_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0)
          JitRunner_TC_D1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        if (total_2 > 0)
          JitRunner_PropQ_D0::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
        if (total_3 > 0)
          JitRunner_PropQ_D1::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
        if (total_4 > 0)
          JitRunner_Join3a_D0::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
        if (total_5 > 0)
          JitRunner_Join3a_D2::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
        if (total_6 > 0)
          JitRunner_Join3b_D1::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
        if (total_7 > 0)
          JitRunner_Join3b_D2::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
        if (total_8 > 0)
          JitRunner_PropQS_D0::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
        if (total_9 > 0)
          JitRunner_Join3c_D0::launch_materialize(db, p_9, total_9, _stream_pool.get(9));
        if (total_10 > 0)
          JitRunner_Join3c_D2::launch_materialize(db, p_10, total_10, _stream_pool.get(10));
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
    GPU_DEVICE_SYNCHRONIZE();  // sync all materialize streams
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<OutP, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<OutP, NEW_VER, OutP_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<OutP, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<OutQ, DELTA_VER>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<OutQ, NEW_VER, OutQ_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, NEW_VER>,
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>,
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<OutQ, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 1, 0, 2>, DELTA_VER>>(db);
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>,
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<
        SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  GPU_DEVICE_SYNCHRONIZE();

  // Reconstruct intern columns from canonical index
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
}
template <typename DB>
static void step_4(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutP, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_5(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<OutQ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
