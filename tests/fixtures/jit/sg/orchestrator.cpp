  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using Sg_canonical_spec_t = SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_SGBase::execute_fused(db, 0); else JitRunner_SGBase::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Sg, NEW_VER, Sg_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Sg, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    using Sg_canonical_spec_t = SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Arc, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<Sg, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<Sg, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Sg=" << _di.root().degree(); } catch (...) { std::cerr << " Sg=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_SGRec_D1::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_SGRec_D1::execute(db, static_cast<uint32_t>(iter));
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Sg, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Sg, NEW_VER, Sg_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Sg, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Sg, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
