template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

  using Knows_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<KnowsInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_KnowsLoad::execute_fused(db, 0);
  else
    JitRunner_KnowsLoad::execute(db, 0);
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
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
template <typename DB>
static void step_2(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

  using Triangle_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Knows, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  // Count-only query mode
  {
    auto p = JitRunner_TriangleJoin::setup(db, 0, 0);
    JitRunner_TriangleJoin::launch_count(p, 0);
    GPU_DEVICE_SYNCHRONIZE();
    uint64_t total =
        thrust::reduce(thrust::device, p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads,
                       uint64_t(0), thrust::plus<uint64_t>());
    std::cout << " >>>>>>>>>>>>>>>>> TriangleJoin count: " << total << std::endl;
  }
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
static void step_3(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Triangle, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
