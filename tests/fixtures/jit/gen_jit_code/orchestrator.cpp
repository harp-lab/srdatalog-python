template <typename DB>
static void step_0(DB& db, std::size_t max_iterations) {
  // Build output indexes
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

  using Z_canonical_spec_t =
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  bool _tail_mode = false;

  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
  mir_helpers::create_index_fn<
      SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
  if (_tail_mode)
    JitRunner_Triangle::execute_fused(db, 0);
  else
    JitRunner_Triangle::execute(db, 0);
  SRDatalog::GPU::mir_helpers::rebuild_index_fn<
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
  SRDatalog::GPU::mir_helpers::check_size_fn<Z, NEW_VER, Z_canonical_spec_t>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::compute_delta_index_fn<
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, NEW_VER>,
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, FULL_VER>,
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
  nvtxRangePop();  // merge
  SRDatalog::GPU::mir_helpers::clear_relation_fn<Z, NEW_VER>(db);
  nvtxRangePushA("merge");
  SRDatalog::GPU::mir_helpers::merge_index_fn<
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  nvtxRangePop();  // merge
}
template <typename DB>
static void step_1(DB& db, std::size_t max_iterations) {
  mir_helpers::reconstruct_fn<
      SRDatalog::mir::IndexSpecT<Z, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
  GPU_DEVICE_SYNCHRONIZE();
}
