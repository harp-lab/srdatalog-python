  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using isType_canonical_spec_t = SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>;
    using isReferenceType_canonical_spec_t = SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>;
    using isClassType_IDB_canonical_spec_t = SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>;
    using isArrayType_IDB_canonical_spec_t = SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>;
    using isInterfaceType_IDB_canonical_spec_t = SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>;
    using DirectSubclass_canonical_spec_t = SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(12);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InterfaceType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ApplicationClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InterfaceType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ApplicationClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InterfaceType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_isType_Class::execute_fused(db, 0);
      JitRunner_isType_Array::execute_fused(db, 0);
      JitRunner_isType_Interface::execute_fused(db, 0);
      JitRunner_isType_AppClass::execute_fused(db, 0);
      JitRunner_isRefType_Class::execute_fused(db, 0);
      JitRunner_isRefType_Array::execute_fused(db, 0);
      JitRunner_isRefType_Interface::execute_fused(db, 0);
      JitRunner_isRefType_AppClass::execute_fused(db, 0);
      JitRunner_isClassType::execute_fused(db, 0);
      JitRunner_isArrayType::execute_fused(db, 0);
      JitRunner_isInterfaceType::execute_fused(db, 0);
      JitRunner_DirectSubclass::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 12 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_isType_Class::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_isType_Array::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_isType_Interface::setup(db, 0, _stream_pool.get(2));
      auto p_3 = JitRunner_isType_AppClass::setup(db, 0, _stream_pool.get(3));
      auto p_4 = JitRunner_isRefType_Class::setup(db, 0, _stream_pool.get(4));
      auto p_5 = JitRunner_isRefType_Array::setup(db, 0, _stream_pool.get(5));
      auto p_6 = JitRunner_isRefType_Interface::setup(db, 0, _stream_pool.get(6));
      auto p_7 = JitRunner_isRefType_AppClass::setup(db, 0, _stream_pool.get(7));
      auto p_8 = JitRunner_isClassType::setup(db, 0, _stream_pool.get(8));
      auto p_9 = JitRunner_isArrayType::setup(db, 0, _stream_pool.get(9));
      auto p_10 = JitRunner_isInterfaceType::setup(db, 0, _stream_pool.get(10));
      auto p_11 = JitRunner_DirectSubclass::setup(db, 0, _stream_pool.get(11));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_isReferenceType_size = p_4.num_threads + p_5.num_threads + p_6.num_threads + p_7.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_isReferenceType(shared_buf_isReferenceType_size);
      uint32_t off_isReferenceType_0 = 0;
      p_4.thread_counts_ptr = shared_buf_isReferenceType.data() + off_isReferenceType_0;
      uint32_t off_isReferenceType_1 = off_isReferenceType_0 + p_4.num_threads;
      p_5.thread_counts_ptr = shared_buf_isReferenceType.data() + off_isReferenceType_1;
      uint32_t off_isReferenceType_2 = off_isReferenceType_1 + p_5.num_threads;
      p_6.thread_counts_ptr = shared_buf_isReferenceType.data() + off_isReferenceType_2;
      uint32_t off_isReferenceType_3 = off_isReferenceType_2 + p_6.num_threads;
      p_7.thread_counts_ptr = shared_buf_isReferenceType.data() + off_isReferenceType_3;
      uint32_t shared_buf_isType_size = p_0.num_threads + p_1.num_threads + p_2.num_threads + p_3.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_isType(shared_buf_isType_size);
      uint32_t off_isType_0 = 0;
      p_0.thread_counts_ptr = shared_buf_isType.data() + off_isType_0;
      uint32_t off_isType_1 = off_isType_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_isType.data() + off_isType_1;
      uint32_t off_isType_2 = off_isType_1 + p_1.num_threads;
      p_2.thread_counts_ptr = shared_buf_isType.data() + off_isType_2;
      uint32_t off_isType_3 = off_isType_2 + p_2.num_threads;
      p_3.thread_counts_ptr = shared_buf_isType.data() + off_isType_3;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_isType_Class::launch_count(p_0, _stream_pool.get(0));
      JitRunner_isType_Array::launch_count(p_1, _stream_pool.get(1));
      JitRunner_isType_Interface::launch_count(p_2, _stream_pool.get(2));
      JitRunner_isType_AppClass::launch_count(p_3, _stream_pool.get(3));
      JitRunner_isRefType_Class::launch_count(p_4, _stream_pool.get(4));
      JitRunner_isRefType_Array::launch_count(p_5, _stream_pool.get(5));
      JitRunner_isRefType_Interface::launch_count(p_6, _stream_pool.get(6));
      JitRunner_isRefType_AppClass::launch_count(p_7, _stream_pool.get(7));
      JitRunner_isClassType::launch_count(p_8, _stream_pool.get(8));
      JitRunner_isArrayType::launch_count(p_9, _stream_pool.get(9));
      JitRunner_isInterfaceType::launch_count(p_10, _stream_pool.get(10));
      JitRunner_DirectSubclass::launch_count(p_11, _stream_pool.get(11));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_DirectSubclass::scan_only(p_11, _stream_pool.get(11));
      JitRunner_isArrayType::scan_only(p_9, _stream_pool.get(9));
      JitRunner_isClassType::scan_only(p_8, _stream_pool.get(8));
      JitRunner_isInterfaceType::scan_only(p_10, _stream_pool.get(10));
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(4)), shared_buf_isReferenceType.data(), shared_buf_isReferenceType.data() + shared_buf_isReferenceType_size, shared_buf_isReferenceType.data(), 0, thrust::plus<uint32_t>());
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_isType.data(), shared_buf_isType.data() + shared_buf_isType_size, shared_buf_isType.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_11 = JitRunner_DirectSubclass::read_total(p_11);
      uint32_t total_9 = JitRunner_isArrayType::read_total(p_9);
      uint32_t total_8 = JitRunner_isClassType::read_total(p_8);
      uint32_t total_10 = JitRunner_isInterfaceType::read_total(p_10);
      // Read boundary values for isReferenceType (4 rules)
      uint32_t bnd_isReferenceType[5];
      GPU_MEMCPY_ASYNC(&bnd_isReferenceType[0], shared_buf_isReferenceType.data() + off_isReferenceType_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isReferenceType[1], shared_buf_isReferenceType.data() + off_isReferenceType_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isReferenceType[2], shared_buf_isReferenceType.data() + off_isReferenceType_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isReferenceType[3], shared_buf_isReferenceType.data() + off_isReferenceType_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isReferenceType[4], shared_buf_isReferenceType.data() + shared_buf_isReferenceType_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_4 = bnd_isReferenceType[1] - bnd_isReferenceType[0];
      uint32_t total_5 = bnd_isReferenceType[2] - bnd_isReferenceType[1];
      uint32_t total_6 = bnd_isReferenceType[3] - bnd_isReferenceType[2];
      uint32_t total_7 = bnd_isReferenceType[4] - bnd_isReferenceType[3];
      // Read boundary values for isType (4 rules)
      uint32_t bnd_isType[5];
      GPU_MEMCPY_ASYNC(&bnd_isType[0], shared_buf_isType.data() + off_isType_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isType[1], shared_buf_isType.data() + off_isType_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isType[2], shared_buf_isType.data() + off_isType_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isType[3], shared_buf_isType.data() + off_isType_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_isType[4], shared_buf_isType.data() + shared_buf_isType_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_isType[1] - bnd_isType[0];
      uint32_t total_1 = bnd_isType[2] - bnd_isType[1];
      uint32_t total_2 = bnd_isType[3] - bnd_isType[2];
      uint32_t total_3 = bnd_isType[4] - bnd_isType[3];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize DirectSubclass
        uint32_t sum_DirectSubclass = total_11;
        auto& dest_DirectSubclass = get_relation_by_schema<DirectSubclass, NEW_VER>(db);
        uint32_t base_offset_DirectSubclass = static_cast<uint32_t>(dest_DirectSubclass.size());
        if (sum_DirectSubclass > 0) dest_DirectSubclass.resize_interned_columns(base_offset_DirectSubclass + sum_DirectSubclass);
        uint32_t running_offset_DirectSubclass = base_offset_DirectSubclass;
        p_11.old_size_0 = running_offset_DirectSubclass;
        running_offset_DirectSubclass += total_11;
      }
      {  // Resize isArrayType_IDB
        uint32_t sum_isArrayType_IDB = total_9;
        auto& dest_isArrayType_IDB = get_relation_by_schema<isArrayType_IDB, NEW_VER>(db);
        uint32_t base_offset_isArrayType_IDB = static_cast<uint32_t>(dest_isArrayType_IDB.size());
        if (sum_isArrayType_IDB > 0) dest_isArrayType_IDB.resize_interned_columns(base_offset_isArrayType_IDB + sum_isArrayType_IDB);
        uint32_t running_offset_isArrayType_IDB = base_offset_isArrayType_IDB;
        p_9.old_size_0 = running_offset_isArrayType_IDB;
        running_offset_isArrayType_IDB += total_9;
      }
      {  // Resize isClassType_IDB
        uint32_t sum_isClassType_IDB = total_8;
        auto& dest_isClassType_IDB = get_relation_by_schema<isClassType_IDB, NEW_VER>(db);
        uint32_t base_offset_isClassType_IDB = static_cast<uint32_t>(dest_isClassType_IDB.size());
        if (sum_isClassType_IDB > 0) dest_isClassType_IDB.resize_interned_columns(base_offset_isClassType_IDB + sum_isClassType_IDB);
        uint32_t running_offset_isClassType_IDB = base_offset_isClassType_IDB;
        p_8.old_size_0 = running_offset_isClassType_IDB;
        running_offset_isClassType_IDB += total_8;
      }
      {  // Resize isInterfaceType_IDB
        uint32_t sum_isInterfaceType_IDB = total_10;
        auto& dest_isInterfaceType_IDB = get_relation_by_schema<isInterfaceType_IDB, NEW_VER>(db);
        uint32_t base_offset_isInterfaceType_IDB = static_cast<uint32_t>(dest_isInterfaceType_IDB.size());
        if (sum_isInterfaceType_IDB > 0) dest_isInterfaceType_IDB.resize_interned_columns(base_offset_isInterfaceType_IDB + sum_isInterfaceType_IDB);
        uint32_t running_offset_isInterfaceType_IDB = base_offset_isInterfaceType_IDB;
        p_10.old_size_0 = running_offset_isInterfaceType_IDB;
        running_offset_isInterfaceType_IDB += total_10;
      }
      {  // Resize isReferenceType
        uint32_t sum_isReferenceType = total_4 + total_5 + total_6 + total_7;
        auto& dest_isReferenceType = get_relation_by_schema<isReferenceType, NEW_VER>(db);
        uint32_t base_offset_isReferenceType = static_cast<uint32_t>(dest_isReferenceType.size());
        if (sum_isReferenceType > 0) dest_isReferenceType.resize_interned_columns(base_offset_isReferenceType + sum_isReferenceType);
        p_4.old_size_0 = base_offset_isReferenceType;
        p_5.old_size_0 = base_offset_isReferenceType;
        p_6.old_size_0 = base_offset_isReferenceType;
        p_7.old_size_0 = base_offset_isReferenceType;
      }
      {  // Resize isType
        uint32_t sum_isType = total_0 + total_1 + total_2 + total_3;
        auto& dest_isType = get_relation_by_schema<isType, NEW_VER>(db);
        uint32_t base_offset_isType = static_cast<uint32_t>(dest_isType.size());
        if (sum_isType > 0) dest_isType.resize_interned_columns(base_offset_isType + sum_isType);
        p_0.old_size_0 = base_offset_isType;
        p_1.old_size_0 = base_offset_isType;
        p_2.old_size_0 = base_offset_isType;
        p_3.old_size_0 = base_offset_isType;
      }

      if (total_0 > 0) JitRunner_isType_Class::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_isType_Array::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_isType_Interface::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      if (total_3 > 0) JitRunner_isType_AppClass::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
      if (total_4 > 0) JitRunner_isRefType_Class::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
      if (total_5 > 0) JitRunner_isRefType_Array::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
      if (total_6 > 0) JitRunner_isRefType_Interface::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
      if (total_7 > 0) JitRunner_isRefType_AppClass::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
      if (total_8 > 0) JitRunner_isClassType::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
      if (total_9 > 0) JitRunner_isArrayType::launch_materialize(db, p_9, total_9, _stream_pool.get(9));
      if (total_10 > 0) JitRunner_isInterfaceType::launch_materialize(db, p_10, total_10, _stream_pool.get(10));
      if (total_11 > 0) JitRunner_DirectSubclass::launch_materialize(db, p_11, total_11, _stream_pool.get(11));
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
      _stream_pool.record_event(11);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isType, NEW_VER, isType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isType, NEW_VER, isType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isType, NEW_VER, isType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isType, NEW_VER, isType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isReferenceType, NEW_VER, isReferenceType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isReferenceType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isReferenceType, NEW_VER, isReferenceType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isReferenceType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isReferenceType, NEW_VER, isReferenceType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isReferenceType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isReferenceType, NEW_VER, isReferenceType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isReferenceType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isClassType_IDB, NEW_VER, isClassType_IDB_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isClassType_IDB, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isArrayType_IDB, NEW_VER, isArrayType_IDB_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isArrayType_IDB, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<isInterfaceType_IDB, NEW_VER, isInterfaceType_IDB_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<isInterfaceType_IDB, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<DirectSubclass, NEW_VER, DirectSubclass_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<DirectSubclass, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_4(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_5(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_6(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_7(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

    using Superinterface_canonical_spec_t = SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_Superinterface_Base::execute_fused(db, 0); else JitRunner_Superinterface_Base::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Superinterface, NEW_VER, Superinterface_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Superinterface, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_8(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_9(DB& db, std::size_t max_iterations) {
    using Superinterface_canonical_spec_t = SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<Superinterface, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{1, 0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<Superinterface, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{1, 0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Superinterface=" << _di.root().degree(); } catch (...) { std::cerr << " Superinterface=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      static SRDatalog::GPU::StreamPool _stream_pool;
      _stream_pool.ensure(2);

      if (_tail_mode) {
        JitRunner_Superinterface_Trans_D0::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_Superinterface_Inherit_D0::execute_fused(db, static_cast<uint32_t>(iter));
      } else {
      // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_Superinterface_Trans_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_Superinterface_Inherit_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_Superinterface_size = p_0.num_threads + p_1.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_Superinterface(shared_buf_Superinterface_size);
        uint32_t off_Superinterface_0 = 0;
        p_0.thread_counts_ptr = shared_buf_Superinterface.data() + off_Superinterface_0;
        uint32_t off_Superinterface_1 = off_Superinterface_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_Superinterface.data() + off_Superinterface_1;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_Superinterface_Trans_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_Superinterface_Inherit_D0::launch_count(p_1, _stream_pool.get(1));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_Superinterface.data(), shared_buf_Superinterface.data() + shared_buf_Superinterface_size, shared_buf_Superinterface.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for Superinterface (2 rules)
        uint32_t bnd_Superinterface[3];
        GPU_MEMCPY_ASYNC(&bnd_Superinterface[0], shared_buf_Superinterface.data() + off_Superinterface_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_Superinterface[1], shared_buf_Superinterface.data() + off_Superinterface_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_Superinterface[2], shared_buf_Superinterface.data() + shared_buf_Superinterface_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_Superinterface[1] - bnd_Superinterface[0];
        uint32_t total_1 = bnd_Superinterface[2] - bnd_Superinterface[1];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize Superinterface
          uint32_t sum_Superinterface = total_0 + total_1;
          auto& dest_Superinterface = get_relation_by_schema<Superinterface, NEW_VER>(db);
          uint32_t base_offset_Superinterface = static_cast<uint32_t>(dest_Superinterface.size());
          if (sum_Superinterface > 0) dest_Superinterface.resize_interned_columns(base_offset_Superinterface + sum_Superinterface);
          p_0.old_size_0 = base_offset_Superinterface;
          p_1.old_size_0 = base_offset_Superinterface;
        }

        if (total_0 > 0) JitRunner_Superinterface_Trans_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0) JitRunner_Superinterface_Inherit_D0::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        nvtxRangePop();  // join_pipeline
      }
      }
      GPU_DEVICE_SYNCHRONIZE(); // sync all materialize streams
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Superinterface, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Superinterface, NEW_VER, Superinterface_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Superinterface, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_10(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_11(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using MethodImplemented_canonical_spec_t = SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using MainMethodDeclaration_canonical_spec_t = SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>;
    using MethodInvocation_Base_canonical_spec_t = SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(4);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Modifier, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MainClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Modifier, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Modifier, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Modifier, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_Modifier, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SpecialMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_MethodImplemented::execute_fused(db, 0);
      JitRunner_MainMethodDecl_Base::execute_fused(db, 0);
      JitRunner_MIBase_Virtual::execute_fused(db, 0);
      JitRunner_MIBase_Special::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 4 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_MethodImplemented::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_MainMethodDecl_Base::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_MIBase_Virtual::setup(db, 0, _stream_pool.get(2));
      auto p_3 = JitRunner_MIBase_Special::setup(db, 0, _stream_pool.get(3));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_MethodInvocation_Base_size = p_2.num_threads + p_3.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_MethodInvocation_Base(shared_buf_MethodInvocation_Base_size);
      uint32_t off_MethodInvocation_Base_0 = 0;
      p_2.thread_counts_ptr = shared_buf_MethodInvocation_Base.data() + off_MethodInvocation_Base_0;
      uint32_t off_MethodInvocation_Base_1 = off_MethodInvocation_Base_0 + p_2.num_threads;
      p_3.thread_counts_ptr = shared_buf_MethodInvocation_Base.data() + off_MethodInvocation_Base_1;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_MethodImplemented::launch_count(p_0, _stream_pool.get(0));
      JitRunner_MainMethodDecl_Base::launch_count(p_1, _stream_pool.get(1));
      JitRunner_MIBase_Virtual::launch_count(p_2, _stream_pool.get(2));
      JitRunner_MIBase_Special::launch_count(p_3, _stream_pool.get(3));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_MainMethodDecl_Base::scan_only(p_1, _stream_pool.get(1));
      JitRunner_MethodImplemented::scan_only(p_0, _stream_pool.get(0));
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(2)), shared_buf_MethodInvocation_Base.data(), shared_buf_MethodInvocation_Base.data() + shared_buf_MethodInvocation_Base_size, shared_buf_MethodInvocation_Base.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_1 = JitRunner_MainMethodDecl_Base::read_total(p_1);
      uint32_t total_0 = JitRunner_MethodImplemented::read_total(p_0);
      // Read boundary values for MethodInvocation_Base (2 rules)
      uint32_t bnd_MethodInvocation_Base[3];
      GPU_MEMCPY_ASYNC(&bnd_MethodInvocation_Base[0], shared_buf_MethodInvocation_Base.data() + off_MethodInvocation_Base_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_MethodInvocation_Base[1], shared_buf_MethodInvocation_Base.data() + off_MethodInvocation_Base_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_MethodInvocation_Base[2], shared_buf_MethodInvocation_Base.data() + shared_buf_MethodInvocation_Base_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_2 = bnd_MethodInvocation_Base[1] - bnd_MethodInvocation_Base[0];
      uint32_t total_3 = bnd_MethodInvocation_Base[2] - bnd_MethodInvocation_Base[1];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize MainMethodDeclaration
        uint32_t sum_MainMethodDeclaration = total_1;
        auto& dest_MainMethodDeclaration = get_relation_by_schema<MainMethodDeclaration, NEW_VER>(db);
        uint32_t base_offset_MainMethodDeclaration = static_cast<uint32_t>(dest_MainMethodDeclaration.size());
        if (sum_MainMethodDeclaration > 0) dest_MainMethodDeclaration.resize_interned_columns(base_offset_MainMethodDeclaration + sum_MainMethodDeclaration);
        uint32_t running_offset_MainMethodDeclaration = base_offset_MainMethodDeclaration;
        p_1.old_size_0 = running_offset_MainMethodDeclaration;
        running_offset_MainMethodDeclaration += total_1;
      }
      {  // Resize MethodImplemented
        uint32_t sum_MethodImplemented = total_0;
        auto& dest_MethodImplemented = get_relation_by_schema<MethodImplemented, NEW_VER>(db);
        uint32_t base_offset_MethodImplemented = static_cast<uint32_t>(dest_MethodImplemented.size());
        if (sum_MethodImplemented > 0) dest_MethodImplemented.resize_interned_columns(base_offset_MethodImplemented + sum_MethodImplemented);
        uint32_t running_offset_MethodImplemented = base_offset_MethodImplemented;
        p_0.old_size_0 = running_offset_MethodImplemented;
        running_offset_MethodImplemented += total_0;
      }
      {  // Resize MethodInvocation_Base
        uint32_t sum_MethodInvocation_Base = total_2 + total_3;
        auto& dest_MethodInvocation_Base = get_relation_by_schema<MethodInvocation_Base, NEW_VER>(db);
        uint32_t base_offset_MethodInvocation_Base = static_cast<uint32_t>(dest_MethodInvocation_Base.size());
        if (sum_MethodInvocation_Base > 0) dest_MethodInvocation_Base.resize_interned_columns(base_offset_MethodInvocation_Base + sum_MethodInvocation_Base);
        p_2.old_size_0 = base_offset_MethodInvocation_Base;
        p_3.old_size_0 = base_offset_MethodInvocation_Base;
      }

      if (total_0 > 0) JitRunner_MethodImplemented::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_MainMethodDecl_Base::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_MIBase_Virtual::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      if (total_3 > 0) JitRunner_MIBase_Special::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      _stream_pool.record_event(3);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MethodImplemented, NEW_VER, MethodImplemented_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodImplemented, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MainMethodDeclaration, NEW_VER, MainMethodDeclaration_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MainMethodDeclaration, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MethodInvocation_Base, NEW_VER, MethodInvocation_Base_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodInvocation_Base, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MethodInvocation_Base, NEW_VER, MethodInvocation_Base_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodInvocation_Base, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_12(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_13(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_14(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_15(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using Subclass_canonical_spec_t = SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_Subclass_Base::execute_fused(db, 0); else JitRunner_Subclass_Base::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Subclass, NEW_VER, Subclass_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Subclass, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_16(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_17(DB& db, std::size_t max_iterations) {
    using Subclass_canonical_spec_t = SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSubclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<Subclass, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<Subclass, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Subclass=" << _di.root().degree(); } catch (...) { std::cerr << " Subclass=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_Subclass_Trans_D0::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_Subclass_Trans_D0::execute(db, static_cast<uint32_t>(iter));
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Subclass, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Subclass, NEW_VER, Subclass_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Subclass, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_18(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_19(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using Superclass_canonical_spec_t = SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_Superclass::execute_fused(db, 0); else JitRunner_Superclass::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Superclass, NEW_VER, Superclass_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Superclass, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_20(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Superclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_21(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using SubtypeOf_canonical_spec_t = SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(10);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Subclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isClassType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Superinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isInterfaceType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isArrayType_IDB, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_SubtypeOf_Refl_Class::execute_fused(db, 0);
      JitRunner_SubtypeOf_Refl_Type::execute_fused(db, 0);
      JitRunner_SubtypeOf_Refl_Interface::execute_fused(db, 0);
      JitRunner_SubtypeOf_Subclass::execute_fused(db, 0);
      JitRunner_SubtypeOf_Class_Iface::execute_fused(db, 0);
      JitRunner_SubtypeOf_Iface_Iface::execute_fused(db, 0);
      JitRunner_SubtypeOf_Iface_Object::execute_fused(db, 0);
      JitRunner_SubtypeOf_Array_Object::execute_fused(db, 0);
      JitRunner_SubtypeOf_Array_Cloneable::execute_fused(db, 0);
      JitRunner_SubtypeOf_Array_Serializable::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 10 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_SubtypeOf_Refl_Class::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_SubtypeOf_Refl_Type::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_SubtypeOf_Refl_Interface::setup(db, 0, _stream_pool.get(2));
      auto p_3 = JitRunner_SubtypeOf_Subclass::setup(db, 0, _stream_pool.get(3));
      auto p_4 = JitRunner_SubtypeOf_Class_Iface::setup(db, 0, _stream_pool.get(4));
      auto p_5 = JitRunner_SubtypeOf_Iface_Iface::setup(db, 0, _stream_pool.get(5));
      auto p_6 = JitRunner_SubtypeOf_Iface_Object::setup(db, 0, _stream_pool.get(6));
      auto p_7 = JitRunner_SubtypeOf_Array_Object::setup(db, 0, _stream_pool.get(7));
      auto p_8 = JitRunner_SubtypeOf_Array_Cloneable::setup(db, 0, _stream_pool.get(8));
      auto p_9 = JitRunner_SubtypeOf_Array_Serializable::setup(db, 0, _stream_pool.get(9));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_SubtypeOf_size = p_0.num_threads + p_1.num_threads + p_2.num_threads + p_3.num_threads + p_4.num_threads + p_5.num_threads + p_6.num_threads + p_7.num_threads + p_8.num_threads + p_9.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_SubtypeOf(shared_buf_SubtypeOf_size);
      uint32_t off_SubtypeOf_0 = 0;
      p_0.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_0;
      uint32_t off_SubtypeOf_1 = off_SubtypeOf_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_1;
      uint32_t off_SubtypeOf_2 = off_SubtypeOf_1 + p_1.num_threads;
      p_2.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_2;
      uint32_t off_SubtypeOf_3 = off_SubtypeOf_2 + p_2.num_threads;
      p_3.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_3;
      uint32_t off_SubtypeOf_4 = off_SubtypeOf_3 + p_3.num_threads;
      p_4.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_4;
      uint32_t off_SubtypeOf_5 = off_SubtypeOf_4 + p_4.num_threads;
      p_5.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_5;
      uint32_t off_SubtypeOf_6 = off_SubtypeOf_5 + p_5.num_threads;
      p_6.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_6;
      uint32_t off_SubtypeOf_7 = off_SubtypeOf_6 + p_6.num_threads;
      p_7.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_7;
      uint32_t off_SubtypeOf_8 = off_SubtypeOf_7 + p_7.num_threads;
      p_8.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_8;
      uint32_t off_SubtypeOf_9 = off_SubtypeOf_8 + p_8.num_threads;
      p_9.thread_counts_ptr = shared_buf_SubtypeOf.data() + off_SubtypeOf_9;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_SubtypeOf_Refl_Class::launch_count(p_0, _stream_pool.get(0));
      JitRunner_SubtypeOf_Refl_Type::launch_count(p_1, _stream_pool.get(1));
      JitRunner_SubtypeOf_Refl_Interface::launch_count(p_2, _stream_pool.get(2));
      JitRunner_SubtypeOf_Subclass::launch_count(p_3, _stream_pool.get(3));
      JitRunner_SubtypeOf_Class_Iface::launch_count(p_4, _stream_pool.get(4));
      JitRunner_SubtypeOf_Iface_Iface::launch_count(p_5, _stream_pool.get(5));
      JitRunner_SubtypeOf_Iface_Object::launch_count(p_6, _stream_pool.get(6));
      JitRunner_SubtypeOf_Array_Object::launch_count(p_7, _stream_pool.get(7));
      JitRunner_SubtypeOf_Array_Cloneable::launch_count(p_8, _stream_pool.get(8));
      JitRunner_SubtypeOf_Array_Serializable::launch_count(p_9, _stream_pool.get(9));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_SubtypeOf.data(), shared_buf_SubtypeOf.data() + shared_buf_SubtypeOf_size, shared_buf_SubtypeOf.data(), 0, thrust::plus<uint32_t>());

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      // Read boundary values for SubtypeOf (10 rules)
      uint32_t bnd_SubtypeOf[11];
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[0], shared_buf_SubtypeOf.data() + off_SubtypeOf_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[1], shared_buf_SubtypeOf.data() + off_SubtypeOf_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[2], shared_buf_SubtypeOf.data() + off_SubtypeOf_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[3], shared_buf_SubtypeOf.data() + off_SubtypeOf_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[4], shared_buf_SubtypeOf.data() + off_SubtypeOf_4, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[5], shared_buf_SubtypeOf.data() + off_SubtypeOf_5, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[6], shared_buf_SubtypeOf.data() + off_SubtypeOf_6, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[7], shared_buf_SubtypeOf.data() + off_SubtypeOf_7, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[8], shared_buf_SubtypeOf.data() + off_SubtypeOf_8, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[9], shared_buf_SubtypeOf.data() + off_SubtypeOf_9, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_SubtypeOf[10], shared_buf_SubtypeOf.data() + shared_buf_SubtypeOf_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_SubtypeOf[1] - bnd_SubtypeOf[0];
      uint32_t total_1 = bnd_SubtypeOf[2] - bnd_SubtypeOf[1];
      uint32_t total_2 = bnd_SubtypeOf[3] - bnd_SubtypeOf[2];
      uint32_t total_3 = bnd_SubtypeOf[4] - bnd_SubtypeOf[3];
      uint32_t total_4 = bnd_SubtypeOf[5] - bnd_SubtypeOf[4];
      uint32_t total_5 = bnd_SubtypeOf[6] - bnd_SubtypeOf[5];
      uint32_t total_6 = bnd_SubtypeOf[7] - bnd_SubtypeOf[6];
      uint32_t total_7 = bnd_SubtypeOf[8] - bnd_SubtypeOf[7];
      uint32_t total_8 = bnd_SubtypeOf[9] - bnd_SubtypeOf[8];
      uint32_t total_9 = bnd_SubtypeOf[10] - bnd_SubtypeOf[9];

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize SubtypeOf
        uint32_t sum_SubtypeOf = total_0 + total_1 + total_2 + total_3 + total_4 + total_5 + total_6 + total_7 + total_8 + total_9;
        auto& dest_SubtypeOf = get_relation_by_schema<SubtypeOf, NEW_VER>(db);
        uint32_t base_offset_SubtypeOf = static_cast<uint32_t>(dest_SubtypeOf.size());
        if (sum_SubtypeOf > 0) dest_SubtypeOf.resize_interned_columns(base_offset_SubtypeOf + sum_SubtypeOf);
        p_0.old_size_0 = base_offset_SubtypeOf;
        p_1.old_size_0 = base_offset_SubtypeOf;
        p_2.old_size_0 = base_offset_SubtypeOf;
        p_3.old_size_0 = base_offset_SubtypeOf;
        p_4.old_size_0 = base_offset_SubtypeOf;
        p_5.old_size_0 = base_offset_SubtypeOf;
        p_6.old_size_0 = base_offset_SubtypeOf;
        p_7.old_size_0 = base_offset_SubtypeOf;
        p_8.old_size_0 = base_offset_SubtypeOf;
        p_9.old_size_0 = base_offset_SubtypeOf;
      }

      if (total_0 > 0) JitRunner_SubtypeOf_Refl_Class::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_SubtypeOf_Refl_Type::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_SubtypeOf_Refl_Interface::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      if (total_3 > 0) JitRunner_SubtypeOf_Subclass::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
      if (total_4 > 0) JitRunner_SubtypeOf_Class_Iface::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
      if (total_5 > 0) JitRunner_SubtypeOf_Iface_Iface::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
      if (total_6 > 0) JitRunner_SubtypeOf_Iface_Object::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
      if (total_7 > 0) JitRunner_SubtypeOf_Array_Object::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
      if (total_8 > 0) JitRunner_SubtypeOf_Array_Cloneable::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
      if (total_9 > 0) JitRunner_SubtypeOf_Array_Serializable::launch_materialize(db, p_9, total_9, _stream_pool.get(9));
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
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_22(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_23(DB& db, std::size_t max_iterations) {
    using SubtypeOf_canonical_spec_t = SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isReferenceType, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<SubtypeOf, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<SubtypeOf, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " SubtypeOf=" << _di.root().degree(); } catch (...) { std::cerr << " SubtypeOf=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_SubtypeOf_Array_D0::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_SubtypeOf_Array_D0::execute(db, static_cast<uint32_t>(iter));
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOf, NEW_VER, SubtypeOf_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOf, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_24(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_25(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using SupertypeOf_canonical_spec_t = SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using SubtypeOfDifferent_canonical_spec_t = SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_SupertypeOf::execute_fused(db, 0);
      JitRunner_SubtypeOfDiff_Base::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_SupertypeOf::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_SubtypeOfDiff_Base::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_SupertypeOf::launch_count(p_0, _stream_pool.get(0));
      JitRunner_SubtypeOfDiff_Base::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_SubtypeOfDiff_Base::scan_only(p_1, _stream_pool.get(1));
      JitRunner_SupertypeOf::scan_only(p_0, _stream_pool.get(0));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_1 = JitRunner_SubtypeOfDiff_Base::read_total(p_1);
      uint32_t total_0 = JitRunner_SupertypeOf::read_total(p_0);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize SubtypeOfDifferent
        uint32_t sum_SubtypeOfDifferent = total_1;
        auto& dest_SubtypeOfDifferent = get_relation_by_schema<SubtypeOfDifferent, NEW_VER>(db);
        uint32_t base_offset_SubtypeOfDifferent = static_cast<uint32_t>(dest_SubtypeOfDifferent.size());
        if (sum_SubtypeOfDifferent > 0) dest_SubtypeOfDifferent.resize_interned_columns(base_offset_SubtypeOfDifferent + sum_SubtypeOfDifferent);
        uint32_t running_offset_SubtypeOfDifferent = base_offset_SubtypeOfDifferent;
        p_1.old_size_0 = running_offset_SubtypeOfDifferent;
        running_offset_SubtypeOfDifferent += total_1;
      }
      {  // Resize SupertypeOf
        uint32_t sum_SupertypeOf = total_0;
        auto& dest_SupertypeOf = get_relation_by_schema<SupertypeOf, NEW_VER>(db);
        uint32_t base_offset_SupertypeOf = static_cast<uint32_t>(dest_SupertypeOf.size());
        if (sum_SupertypeOf > 0) dest_SupertypeOf.resize_interned_columns(base_offset_SupertypeOf + sum_SupertypeOf);
        uint32_t running_offset_SupertypeOf = base_offset_SupertypeOf;
        p_0.old_size_0 = running_offset_SupertypeOf;
        running_offset_SupertypeOf += total_0;
      }

      if (total_0 > 0) JitRunner_SupertypeOf::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_SubtypeOfDiff_Base::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SupertypeOf, NEW_VER, SupertypeOf_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SupertypeOf, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<SubtypeOfDifferent, NEW_VER, SubtypeOfDifferent_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<SubtypeOfDifferent, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_26(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_27(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<SubtypeOfDifferent, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_28(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>>(db, 0);

    using MethodLookup_canonical_spec_t = SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_MethodLookup_Base::execute_fused(db, 0); else JitRunner_MethodLookup_Base::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<MethodLookup, NEW_VER, MethodLookup_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodLookup, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_29(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_30(DB& db, std::size_t max_iterations) {
    using MethodLookup_canonical_spec_t = SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<MethodLookup, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{2, 0, 1, 3}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<MethodLookup, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{2, 0, 1, 3}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " MethodLookup=" << _di.root().degree(); } catch (...) { std::cerr << " MethodLookup=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      static SRDatalog::GPU::StreamPool _stream_pool;
      _stream_pool.ensure(2);

      if (_tail_mode) {
        JitRunner_MethodLookup_Super_D1::execute_fused(db, static_cast<uint32_t>(iter));
        JitRunner_MethodLookup_Iface_D1::execute_fused(db, static_cast<uint32_t>(iter));
      } else {
      // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_MethodLookup_Super_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_MethodLookup_Iface_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_MethodLookup_size = p_0.num_threads + p_1.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_MethodLookup(shared_buf_MethodLookup_size);
        uint32_t off_MethodLookup_0 = 0;
        p_0.thread_counts_ptr = shared_buf_MethodLookup.data() + off_MethodLookup_0;
        uint32_t off_MethodLookup_1 = off_MethodLookup_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_MethodLookup.data() + off_MethodLookup_1;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_MethodLookup_Super_D1::launch_count(p_0, _stream_pool.get(0));
        JitRunner_MethodLookup_Iface_D1::launch_count(p_1, _stream_pool.get(1));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_MethodLookup.data(), shared_buf_MethodLookup.data() + shared_buf_MethodLookup_size, shared_buf_MethodLookup.data(), 0, thrust::plus<uint32_t>());

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for MethodLookup (2 rules)
        uint32_t bnd_MethodLookup[3];
        GPU_MEMCPY_ASYNC(&bnd_MethodLookup[0], shared_buf_MethodLookup.data() + off_MethodLookup_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_MethodLookup[1], shared_buf_MethodLookup.data() + off_MethodLookup_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_MethodLookup[2], shared_buf_MethodLookup.data() + shared_buf_MethodLookup_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_MethodLookup[1] - bnd_MethodLookup[0];
        uint32_t total_1 = bnd_MethodLookup[2] - bnd_MethodLookup[1];

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize MethodLookup
          uint32_t sum_MethodLookup = total_0 + total_1;
          auto& dest_MethodLookup = get_relation_by_schema<MethodLookup, NEW_VER>(db);
          uint32_t base_offset_MethodLookup = static_cast<uint32_t>(dest_MethodLookup.size());
          if (sum_MethodLookup > 0) dest_MethodLookup.resize_interned_columns(base_offset_MethodLookup + sum_MethodLookup);
          p_0.old_size_0 = base_offset_MethodLookup;
          p_1.old_size_0 = base_offset_MethodLookup;
        }

        if (total_0 > 0) JitRunner_MethodLookup_Super_D1::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0) JitRunner_MethodLookup_Iface_D1::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        _stream_pool.record_event(0);
        _stream_pool.record_event(1);
        nvtxRangePop();  // join_pipeline
      }
      }
      GPU_DEVICE_SYNCHRONIZE(); // sync all materialize streams
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodLookup, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<MethodLookup, NEW_VER, MethodLookup_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<MethodLookup, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_31(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 0, 1, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_32(DB& db, std::size_t max_iterations) {
    // ERROR: unexpected plan kind for non-recursive step: moRebuildIndex
  }
  template <typename DB>
  static void step_33(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);

    using ClassInitializer_canonical_spec_t = SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using IsCastableToString_canonical_spec_t = SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, FULL_VER>;
    using IsStringHeap_canonical_spec_t = SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(3);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodImplemented, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_ClassInitializer::execute_fused(db, 0);
      JitRunner_IsCastableToString_rule::execute_fused(db, 0);
      JitRunner_IsStringHeap_rule::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 3 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_ClassInitializer::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_IsCastableToString_rule::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_IsStringHeap_rule::setup(db, 0, _stream_pool.get(2));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_ClassInitializer::launch_count(p_0, _stream_pool.get(0));
      JitRunner_IsCastableToString_rule::launch_count(p_1, _stream_pool.get(1));
      JitRunner_IsStringHeap_rule::launch_count(p_2, _stream_pool.get(2));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_ClassInitializer::scan_only(p_0, _stream_pool.get(0));
      JitRunner_IsCastableToString_rule::scan_only(p_1, _stream_pool.get(1));
      JitRunner_IsStringHeap_rule::scan_only(p_2, _stream_pool.get(2));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_ClassInitializer::read_total(p_0);
      uint32_t total_1 = JitRunner_IsCastableToString_rule::read_total(p_1);
      uint32_t total_2 = JitRunner_IsStringHeap_rule::read_total(p_2);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize ClassInitializer
        uint32_t sum_ClassInitializer = total_0;
        auto& dest_ClassInitializer = get_relation_by_schema<ClassInitializer, NEW_VER>(db);
        uint32_t base_offset_ClassInitializer = static_cast<uint32_t>(dest_ClassInitializer.size());
        if (sum_ClassInitializer > 0) dest_ClassInitializer.resize_interned_columns(base_offset_ClassInitializer + sum_ClassInitializer);
        uint32_t running_offset_ClassInitializer = base_offset_ClassInitializer;
        p_0.old_size_0 = running_offset_ClassInitializer;
        running_offset_ClassInitializer += total_0;
      }
      {  // Resize IsCastableToString
        uint32_t sum_IsCastableToString = total_1;
        auto& dest_IsCastableToString = get_relation_by_schema<IsCastableToString, NEW_VER>(db);
        uint32_t base_offset_IsCastableToString = static_cast<uint32_t>(dest_IsCastableToString.size());
        if (sum_IsCastableToString > 0) dest_IsCastableToString.resize_interned_columns(base_offset_IsCastableToString + sum_IsCastableToString);
        uint32_t running_offset_IsCastableToString = base_offset_IsCastableToString;
        p_1.old_size_0 = running_offset_IsCastableToString;
        running_offset_IsCastableToString += total_1;
      }
      {  // Resize IsStringHeap
        uint32_t sum_IsStringHeap = total_2;
        auto& dest_IsStringHeap = get_relation_by_schema<IsStringHeap, NEW_VER>(db);
        uint32_t base_offset_IsStringHeap = static_cast<uint32_t>(dest_IsStringHeap.size());
        if (sum_IsStringHeap > 0) dest_IsStringHeap.resize_interned_columns(base_offset_IsStringHeap + sum_IsStringHeap);
        uint32_t running_offset_IsStringHeap = base_offset_IsStringHeap;
        p_2.old_size_0 = running_offset_IsStringHeap;
        running_offset_IsStringHeap += total_2;
      }

      if (total_0 > 0) JitRunner_ClassInitializer::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_IsCastableToString_rule::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_IsStringHeap_rule::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ClassInitializer, NEW_VER, ClassInitializer_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ClassInitializer, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<IsCastableToString, NEW_VER, IsCastableToString_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<IsCastableToString, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<IsStringHeap, NEW_VER, IsStringHeap_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<IsStringHeap, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_34(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_35(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_36(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_37(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using CastTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>;
    using HeapHelper_canonical_spec_t = SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>;
    using HeapHelperNoThis_canonical_spec_t = SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>;
    using VirtualMethodInvocation_canonical_spec_t = SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>;
    using HeapAllocSuperType_canonical_spec_t = SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using IsObjectArrayHeap_canonical_spec_t = SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>;
    using ArrayTypeCompat_canonical_spec_t = SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(8);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignCast, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignCast, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignCast, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsCastableToString, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignCast, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsStringHeap, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodLookup, std::integer_sequence<int, 2, 3, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_SimpleName, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation_Descriptor, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ComponentType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SupertypeOf, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    if (_tail_mode && false) {
    } else {
    // === ParallelGroup (stream-parallel, 8 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_Precompute0::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Precompute0_String::setup(db, 0, _stream_pool.get(1));
      auto p_2 = JitRunner_Precompute1::setup(db, 0, _stream_pool.get(2));
      auto p_3 = JitRunner_Precompute1b::setup(db, 0, _stream_pool.get(3));
      auto p_4 = JitRunner_Precompute2::setup(db, 0, _stream_pool.get(4));
      auto p_5 = JitRunner_HeapAllocHelper::setup(db, 0, _stream_pool.get(5));
      auto p_6 = JitRunner_IsObjectArrayHeap_rule::setup(db, 0, _stream_pool.get(6));
      auto p_7 = JitRunner_ArrayTypeCompatPrecompute::setup(db, 0, _stream_pool.get(7));

      // Phase 2a: Allocate shared count buffers per dest
      uint32_t shared_buf_CastTo_size = p_0.num_threads + p_1.num_threads + 1;
      SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_CastTo(shared_buf_CastTo_size);
      uint32_t off_CastTo_0 = 0;
      p_0.thread_counts_ptr = shared_buf_CastTo.data() + off_CastTo_0;
      uint32_t off_CastTo_1 = off_CastTo_0 + p_0.num_threads;
      p_1.thread_counts_ptr = shared_buf_CastTo.data() + off_CastTo_1;

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_Precompute0::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Precompute0_String::launch_count(p_1, _stream_pool.get(1));
      JitRunner_Precompute1::launch_count(p_2, _stream_pool.get(2));
      JitRunner_Precompute1b::launch_count(p_3, _stream_pool.get(3));
      JitRunner_Precompute2::launch_count(p_4, _stream_pool.get(4));
      JitRunner_HeapAllocHelper::launch_count(p_5, _stream_pool.get(5));
      JitRunner_IsObjectArrayHeap_rule::launch_count(p_6, _stream_pool.get(6));
      JitRunner_ArrayTypeCompatPrecompute::launch_count(p_7, _stream_pool.get(7));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_ArrayTypeCompatPrecompute::scan_only(p_7, _stream_pool.get(7));
      thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_CastTo.data(), shared_buf_CastTo.data() + shared_buf_CastTo_size, shared_buf_CastTo.data(), 0, thrust::plus<uint32_t>());
      JitRunner_HeapAllocHelper::scan_only(p_5, _stream_pool.get(5));
      JitRunner_Precompute1::scan_only(p_2, _stream_pool.get(2));
      JitRunner_Precompute1b::scan_only(p_3, _stream_pool.get(3));
      JitRunner_IsObjectArrayHeap_rule::scan_only(p_6, _stream_pool.get(6));
      JitRunner_Precompute2::scan_only(p_4, _stream_pool.get(4));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_7 = JitRunner_ArrayTypeCompatPrecompute::read_total(p_7);
      // Read boundary values for CastTo (2 rules)
      uint32_t bnd_CastTo[3];
      GPU_MEMCPY_ASYNC(&bnd_CastTo[0], shared_buf_CastTo.data() + off_CastTo_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_CastTo[1], shared_buf_CastTo.data() + off_CastTo_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      GPU_MEMCPY_ASYNC(&bnd_CastTo[2], shared_buf_CastTo.data() + shared_buf_CastTo_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
      uint32_t total_0 = bnd_CastTo[1] - bnd_CastTo[0];
      uint32_t total_1 = bnd_CastTo[2] - bnd_CastTo[1];
      uint32_t total_5 = JitRunner_HeapAllocHelper::read_total(p_5);
      uint32_t total_2 = JitRunner_Precompute1::read_total(p_2);
      uint32_t total_3 = JitRunner_Precompute1b::read_total(p_3);
      uint32_t total_6 = JitRunner_IsObjectArrayHeap_rule::read_total(p_6);
      uint32_t total_4 = JitRunner_Precompute2::read_total(p_4);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize ArrayTypeCompat
        uint32_t sum_ArrayTypeCompat = total_7;
        auto& dest_ArrayTypeCompat = get_relation_by_schema<ArrayTypeCompat, NEW_VER>(db);
        uint32_t base_offset_ArrayTypeCompat = static_cast<uint32_t>(dest_ArrayTypeCompat.size());
        if (sum_ArrayTypeCompat > 0) dest_ArrayTypeCompat.resize_interned_columns(base_offset_ArrayTypeCompat + sum_ArrayTypeCompat);
        uint32_t running_offset_ArrayTypeCompat = base_offset_ArrayTypeCompat;
        p_7.old_size_0 = running_offset_ArrayTypeCompat;
        running_offset_ArrayTypeCompat += total_7;
      }
      {  // Resize CastTo
        uint32_t sum_CastTo = total_0 + total_1;
        auto& dest_CastTo = get_relation_by_schema<CastTo, NEW_VER>(db);
        uint32_t base_offset_CastTo = static_cast<uint32_t>(dest_CastTo.size());
        if (sum_CastTo > 0) dest_CastTo.resize_interned_columns(base_offset_CastTo + sum_CastTo);
        p_0.old_size_0 = base_offset_CastTo;
        p_1.old_size_0 = base_offset_CastTo;
      }
      {  // Resize HeapAllocSuperType
        uint32_t sum_HeapAllocSuperType = total_5;
        auto& dest_HeapAllocSuperType = get_relation_by_schema<HeapAllocSuperType, NEW_VER>(db);
        uint32_t base_offset_HeapAllocSuperType = static_cast<uint32_t>(dest_HeapAllocSuperType.size());
        if (sum_HeapAllocSuperType > 0) dest_HeapAllocSuperType.resize_interned_columns(base_offset_HeapAllocSuperType + sum_HeapAllocSuperType);
        uint32_t running_offset_HeapAllocSuperType = base_offset_HeapAllocSuperType;
        p_5.old_size_0 = running_offset_HeapAllocSuperType;
        running_offset_HeapAllocSuperType += total_5;
      }
      {  // Resize HeapHelper
        uint32_t sum_HeapHelper = total_2;
        auto& dest_HeapHelper = get_relation_by_schema<HeapHelper, NEW_VER>(db);
        uint32_t base_offset_HeapHelper = static_cast<uint32_t>(dest_HeapHelper.size());
        if (sum_HeapHelper > 0) dest_HeapHelper.resize_interned_columns(base_offset_HeapHelper + sum_HeapHelper);
        uint32_t running_offset_HeapHelper = base_offset_HeapHelper;
        p_2.old_size_0 = running_offset_HeapHelper;
        running_offset_HeapHelper += total_2;
      }
      {  // Resize HeapHelperNoThis
        uint32_t sum_HeapHelperNoThis = total_3;
        auto& dest_HeapHelperNoThis = get_relation_by_schema<HeapHelperNoThis, NEW_VER>(db);
        uint32_t base_offset_HeapHelperNoThis = static_cast<uint32_t>(dest_HeapHelperNoThis.size());
        if (sum_HeapHelperNoThis > 0) dest_HeapHelperNoThis.resize_interned_columns(base_offset_HeapHelperNoThis + sum_HeapHelperNoThis);
        uint32_t running_offset_HeapHelperNoThis = base_offset_HeapHelperNoThis;
        p_3.old_size_0 = running_offset_HeapHelperNoThis;
        running_offset_HeapHelperNoThis += total_3;
      }
      {  // Resize IsObjectArrayHeap
        uint32_t sum_IsObjectArrayHeap = total_6;
        auto& dest_IsObjectArrayHeap = get_relation_by_schema<IsObjectArrayHeap, NEW_VER>(db);
        uint32_t base_offset_IsObjectArrayHeap = static_cast<uint32_t>(dest_IsObjectArrayHeap.size());
        if (sum_IsObjectArrayHeap > 0) dest_IsObjectArrayHeap.resize_interned_columns(base_offset_IsObjectArrayHeap + sum_IsObjectArrayHeap);
        uint32_t running_offset_IsObjectArrayHeap = base_offset_IsObjectArrayHeap;
        p_6.old_size_0 = running_offset_IsObjectArrayHeap;
        running_offset_IsObjectArrayHeap += total_6;
      }
      {  // Resize VirtualMethodInvocation
        uint32_t sum_VirtualMethodInvocation = total_4;
        auto& dest_VirtualMethodInvocation = get_relation_by_schema<VirtualMethodInvocation, NEW_VER>(db);
        uint32_t base_offset_VirtualMethodInvocation = static_cast<uint32_t>(dest_VirtualMethodInvocation.size());
        if (sum_VirtualMethodInvocation > 0) dest_VirtualMethodInvocation.resize_interned_columns(base_offset_VirtualMethodInvocation + sum_VirtualMethodInvocation);
        uint32_t running_offset_VirtualMethodInvocation = base_offset_VirtualMethodInvocation;
        p_4.old_size_0 = running_offset_VirtualMethodInvocation;
        running_offset_VirtualMethodInvocation += total_4;
      }

      if (total_0 > 0) JitRunner_Precompute0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_Precompute0_String::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      if (total_2 > 0) JitRunner_Precompute1::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
      if (total_3 > 0) JitRunner_Precompute1b::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
      if (total_4 > 0) JitRunner_Precompute2::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
      if (total_5 > 0) JitRunner_HeapAllocHelper::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
      if (total_6 > 0) JitRunner_IsObjectArrayHeap_rule::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
      if (total_7 > 0) JitRunner_ArrayTypeCompatPrecompute::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      _stream_pool.record_event(2);
      _stream_pool.record_event(3);
      _stream_pool.record_event(4);
      _stream_pool.record_event(5);
      _stream_pool.record_event(6);
      _stream_pool.record_event(7);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<CastTo, NEW_VER, CastTo_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<CastTo, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<CastTo, NEW_VER, CastTo_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<CastTo, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HeapHelper, NEW_VER, HeapHelper_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HeapHelper, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HeapHelperNoThis, NEW_VER, HeapHelperNoThis_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HeapHelperNoThis, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, DELTA_VER>, SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<VirtualMethodInvocation, NEW_VER, VirtualMethodInvocation_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, NEW_VER>, SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>, SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<VirtualMethodInvocation, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, DELTA_VER>, SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<HeapAllocSuperType, NEW_VER, HeapAllocSuperType_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<HeapAllocSuperType, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<IsObjectArrayHeap, NEW_VER, IsObjectArrayHeap_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<IsObjectArrayHeap, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ArrayTypeCompat, NEW_VER, ArrayTypeCompat_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ArrayTypeCompat, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_38(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_39(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_40(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_41(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_42(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_43(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_44(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_45(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);

    using InitializedClass_canonical_spec_t = SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>;
    using Reachable_canonical_spec_t = SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>;
    bool _tail_mode = false;

    static SRDatalog::GPU::StreamPool _stream_pool;
    _stream_pool.ensure(2);

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MainMethodDeclaration, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    if (_tail_mode) {
      JitRunner_InitClass_Main::execute_fused(db, 0);
      JitRunner_Reachable_Main::execute_fused(db, 0);
    } else {
    // === ParallelGroup (stream-parallel, 2 rules, shared dests) ===
    {
      nvtxRangePushA("join_pipeline");
      // Phase 1: Setup all rules
      auto p_0 = JitRunner_InitClass_Main::setup(db, 0, _stream_pool.get(0));
      auto p_1 = JitRunner_Reachable_Main::setup(db, 0, _stream_pool.get(1));

      // Phase 2a: Allocate shared count buffers per dest

      // Phase 2b: Launch count kernels (parallel streams)
      JitRunner_InitClass_Main::launch_count(p_0, _stream_pool.get(0));
      JitRunner_Reachable_Main::launch_count(p_1, _stream_pool.get(1));
      _stream_pool.sync_all();

      // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
      JitRunner_InitClass_Main::scan_only(p_0, _stream_pool.get(0));
      JitRunner_Reachable_Main::scan_only(p_1, _stream_pool.get(1));

      // Phase 3b: Single sync + readback all totals
      GPU_DEVICE_SYNCHRONIZE();
      uint32_t total_0 = JitRunner_InitClass_Main::read_total(p_0);
      uint32_t total_1 = JitRunner_Reachable_Main::read_total(p_1);

      // Phase 3c: Resize once per unique dest + assign per-rule offsets
      {  // Resize InitializedClass
        uint32_t sum_InitializedClass = total_0;
        auto& dest_InitializedClass = get_relation_by_schema<InitializedClass, NEW_VER>(db);
        uint32_t base_offset_InitializedClass = static_cast<uint32_t>(dest_InitializedClass.size());
        if (sum_InitializedClass > 0) dest_InitializedClass.resize_interned_columns(base_offset_InitializedClass + sum_InitializedClass);
        uint32_t running_offset_InitializedClass = base_offset_InitializedClass;
        p_0.old_size_0 = running_offset_InitializedClass;
        running_offset_InitializedClass += total_0;
      }
      {  // Resize Reachable
        uint32_t sum_Reachable = total_1;
        auto& dest_Reachable = get_relation_by_schema<Reachable, NEW_VER>(db);
        uint32_t base_offset_Reachable = static_cast<uint32_t>(dest_Reachable.size());
        if (sum_Reachable > 0) dest_Reachable.resize_interned_columns(base_offset_Reachable + sum_Reachable);
        uint32_t running_offset_Reachable = base_offset_Reachable;
        p_1.old_size_0 = running_offset_Reachable;
        running_offset_Reachable += total_1;
      }

      if (total_0 > 0) JitRunner_InitClass_Main::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
      if (total_1 > 0) JitRunner_Reachable_Main::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
      _stream_pool.record_event(0);
      _stream_pool.record_event(1);
      nvtxRangePop();  // join_pipeline
    }
    }
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<InitializedClass, NEW_VER, InitializedClass_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<InitializedClass, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Reachable, NEW_VER, Reachable_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Reachable, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_46(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_47(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_48(DB& db, std::size_t max_iterations) {
    using ArrayIndexPointsTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using Assign_canonical_spec_t = SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using CallGraphEdge_canonical_spec_t = SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using InitializedClass_canonical_spec_t = SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>;
    using InstanceFieldPointsTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>;
    using Reachable_canonical_spec_t = SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>;
    using ReachableInstruction_canonical_spec_t = SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>;
    using ReachableLoadInstanceField_canonical_spec_t = SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
    using ReachableSortedIndex_canonical_spec_t = SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>;
    using StaticFieldPointsTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>;
    using VarPointsTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperclass, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<DirectSuperinterface, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignHeapAllocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignHeapAllocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignHeapAllocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocation_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Instruction_Method, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Instruction_Method, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<isStaticMethodInvocation_Insn, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Method_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Field_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Field_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Field_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Field_DeclaringType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ClassInitializer, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticMethodInvocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticMethodInvocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ActualParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<FormalParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<FormalParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ActualParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<FormalParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ActualParam, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignReturnValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReturnVar, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReturnVar, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignReturnValue, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignHeapAllocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignHeapAllocation, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AssignLocal, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 2, 0, 3, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CastTo, std::integer_sequence<int, 3, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<LoadArrayIndex, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Var_Type, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayTypeCompat, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Instruction_Method, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Instruction_Method, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 0, 1, 4, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelper, std::integer_sequence<int, 4, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 0, 1, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 0, 1, 3, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VirtualMethodInvocation, std::integer_sequence<int, 1, 0, 2, 3>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapHelperNoThis, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SpecialMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SpecialMethodInvocation_Base, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SpecialMethodInvocation_Base, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SpecialMethodInvocation_Base, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<MethodInvocation_Method, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ThisVar, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 3, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 0, 3, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 1, 3, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 1, 3, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 1, 3, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreInstanceField, std::integer_sequence<int, 1, 3, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreStaticField, std::integer_sequence<int, 0, 2, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreArrayIndex, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StoreArrayIndex, std::integer_sequence<int, 2, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<IsObjectArrayHeap, std::integer_sequence<int, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<HeapAllocSuperType, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<ArrayIndexPointsTo, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<Assign, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<CallGraphEdge, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<InitializedClass, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<InstanceFieldPointsTo, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{2, 1, 0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<Reachable, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<ReachableInstruction, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<ReachableLoadInstanceField, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1, 2}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<ReachableSortedIndex, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<StaticFieldPointsTo, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{1, 0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        {
          auto& delta_rel = get_relation_by_schema<VarPointsTo, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{0, 1}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<ArrayIndexPointsTo, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " ArrayIndexPointsTo=" << _di.root().degree(); } catch (...) { std::cerr << " ArrayIndexPointsTo=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<Assign, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Assign=" << _di.root().degree(); } catch (...) { std::cerr << " Assign=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<CallGraphEdge, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " CallGraphEdge=" << _di.root().degree(); } catch (...) { std::cerr << " CallGraphEdge=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<InitializedClass, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " InitializedClass=" << _di.root().degree(); } catch (...) { std::cerr << " InitializedClass=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<InstanceFieldPointsTo, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{2, 1, 0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " InstanceFieldPointsTo=" << _di.root().degree(); } catch (...) { std::cerr << " InstanceFieldPointsTo=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<Reachable, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Reachable=" << _di.root().degree(); } catch (...) { std::cerr << " Reachable=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<ReachableInstruction, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " ReachableInstruction=" << _di.root().degree(); } catch (...) { std::cerr << " ReachableInstruction=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<ReachableLoadInstanceField, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1, 2}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " ReachableLoadInstanceField=" << _di.root().degree(); } catch (...) { std::cerr << " ReachableLoadInstanceField=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<ReachableSortedIndex, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " ReachableSortedIndex=" << _di.root().degree(); } catch (...) { std::cerr << " ReachableSortedIndex=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<StaticFieldPointsTo, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{1, 0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " StaticFieldPointsTo=" << _di.root().degree(); } catch (...) { std::cerr << " StaticFieldPointsTo=?"; }
          }
          {
            auto& _dr = get_relation_by_schema<VarPointsTo, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{0, 1}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " VarPointsTo=" << _di.root().degree(); } catch (...) { std::cerr << " VarPointsTo=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      static SRDatalog::GPU::StreamPool _stream_pool;
      _stream_pool.ensure(45);

      if (_tail_mode && false) {
      } else {
      // === ParallelGroup (stream-parallel, 45 rules, shared dests) ===
      {
        nvtxRangePushA("join_pipeline");
        // Phase 1: Setup all rules
        auto p_0 = JitRunner_InitClass_Super_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(0));
        auto p_1 = JitRunner_InitClass_Iface_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(1));
        auto p_2 = JitRunner_InitClass_Heap_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(2));
        auto p_3 = JitRunner_InitClass_StaticInvoke_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(3));
        auto p_4 = JitRunner_InitClass_StoreStatic_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(4));
        auto p_5 = JitRunner_InitClass_LoadStatic_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(5));
        auto p_6 = JitRunner_Reachable_Clinit_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(6));
        auto p_7 = JitRunner_Static_MultiHead_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(7));
        auto p_8 = JitRunner_Assign_Param_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(8));
        auto p_9 = JitRunner_Assign_Return_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(9));
        auto p_10 = JitRunner_VPT_HeapAlloc_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(10));
        auto p_11 = JitRunner_VPT_Assign_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(11));
        auto p_12 = JitRunner_VPT_Assign_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(12));
        auto p_13 = JitRunner_VPT_Local_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(13));
        auto p_14 = JitRunner_VPT_Local_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(14));
        auto p_15 = JitRunner_VPT_Cast_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(15));
        auto p_16 = JitRunner_VPT_Cast_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(16));
        auto p_17 = JitRunner_ReachableLoadInstanceField_rule_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(17));
        auto p_18 = JitRunner_VPT_LoadField_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(18));
        auto p_19 = JitRunner_VPT_LoadField_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(19));
        auto p_20 = JitRunner_VPT_LoadField_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(20));
        auto p_21 = JitRunner_VPT_LoadStatic_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(21));
        auto p_22 = JitRunner_VPT_LoadStatic_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(22));
        auto p_23 = JitRunner_VPT_LoadArray_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(23));
        auto p_24 = JitRunner_VPT_LoadArray_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(24));
        auto p_25 = JitRunner_VPT_LoadArray_D3::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(25));
        auto p_26 = JitRunner_ReachableInstructionHelper_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(26));
        auto p_27 = JitRunner_VirtualDispatch_MultiHead_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(27));
        auto p_28 = JitRunner_VirtualDispatch_MultiHead_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(28));
        auto p_29 = JitRunner_Reachable_CGE_Virtual_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(29));
        auto p_30 = JitRunner_Reachable_CGE_Virtual_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(30));
        auto p_31 = JitRunner_SpecialDispatch_MultiHead_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(31));
        auto p_32 = JitRunner_SpecialDispatch_MultiHead_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(32));
        auto p_33 = JitRunner_IFPT_Store_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(33));
        auto p_34 = JitRunner_IFPT_Store_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(34));
        auto p_35 = JitRunner_IFPT_Store_D3::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(35));
        auto p_36 = JitRunner_SFPT_Store_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(36));
        auto p_37 = JitRunner_SFPT_Store_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(37));
        auto p_38 = JitRunner_ReachableSortedIndex_rule_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(38));
        auto p_39 = JitRunner_AIPT_Store_ObjectArray_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(39));
        auto p_40 = JitRunner_AIPT_Store_ObjectArray_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(40));
        auto p_41 = JitRunner_AIPT_Store_ObjectArray_D3::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(41));
        auto p_42 = JitRunner_AIPT_Store_D0::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(42));
        auto p_43 = JitRunner_AIPT_Store_D1::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(43));
        auto p_44 = JitRunner_AIPT_Store_D2::setup(db, static_cast<uint32_t>(iter), _stream_pool.get(44));

        // Phase 2a: Allocate shared count buffers per dest
        uint32_t shared_buf_ArrayIndexPointsTo_size = p_39.num_threads + p_40.num_threads + p_41.num_threads + p_42.num_threads + p_43.num_threads + p_44.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_ArrayIndexPointsTo(shared_buf_ArrayIndexPointsTo_size);
        uint32_t off_ArrayIndexPointsTo_0 = 0;
        p_39.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_0;
        uint32_t off_ArrayIndexPointsTo_1 = off_ArrayIndexPointsTo_0 + p_39.num_threads;
        p_40.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_1;
        uint32_t off_ArrayIndexPointsTo_2 = off_ArrayIndexPointsTo_1 + p_40.num_threads;
        p_41.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_2;
        uint32_t off_ArrayIndexPointsTo_3 = off_ArrayIndexPointsTo_2 + p_41.num_threads;
        p_42.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_3;
        uint32_t off_ArrayIndexPointsTo_4 = off_ArrayIndexPointsTo_3 + p_42.num_threads;
        p_43.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_4;
        uint32_t off_ArrayIndexPointsTo_5 = off_ArrayIndexPointsTo_4 + p_43.num_threads;
        p_44.thread_counts_ptr = shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_5;
        uint32_t shared_buf_Assign_size = p_8.num_threads + p_9.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_Assign(shared_buf_Assign_size);
        uint32_t off_Assign_0 = 0;
        p_8.thread_counts_ptr = shared_buf_Assign.data() + off_Assign_0;
        uint32_t off_Assign_1 = off_Assign_0 + p_8.num_threads;
        p_9.thread_counts_ptr = shared_buf_Assign.data() + off_Assign_1;
        uint32_t shared_buf_InitializedClass_size = p_0.num_threads + p_1.num_threads + p_2.num_threads + p_3.num_threads + p_4.num_threads + p_5.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_InitializedClass(shared_buf_InitializedClass_size);
        uint32_t off_InitializedClass_0 = 0;
        p_0.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_0;
        uint32_t off_InitializedClass_1 = off_InitializedClass_0 + p_0.num_threads;
        p_1.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_1;
        uint32_t off_InitializedClass_2 = off_InitializedClass_1 + p_1.num_threads;
        p_2.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_2;
        uint32_t off_InitializedClass_3 = off_InitializedClass_2 + p_2.num_threads;
        p_3.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_3;
        uint32_t off_InitializedClass_4 = off_InitializedClass_3 + p_3.num_threads;
        p_4.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_4;
        uint32_t off_InitializedClass_5 = off_InitializedClass_4 + p_4.num_threads;
        p_5.thread_counts_ptr = shared_buf_InitializedClass.data() + off_InitializedClass_5;
        uint32_t shared_buf_InstanceFieldPointsTo_size = p_33.num_threads + p_34.num_threads + p_35.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_InstanceFieldPointsTo(shared_buf_InstanceFieldPointsTo_size);
        uint32_t off_InstanceFieldPointsTo_0 = 0;
        p_33.thread_counts_ptr = shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_0;
        uint32_t off_InstanceFieldPointsTo_1 = off_InstanceFieldPointsTo_0 + p_33.num_threads;
        p_34.thread_counts_ptr = shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_1;
        uint32_t off_InstanceFieldPointsTo_2 = off_InstanceFieldPointsTo_1 + p_34.num_threads;
        p_35.thread_counts_ptr = shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_2;
        uint32_t shared_buf_StaticFieldPointsTo_size = p_36.num_threads + p_37.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_StaticFieldPointsTo(shared_buf_StaticFieldPointsTo_size);
        uint32_t off_StaticFieldPointsTo_0 = 0;
        p_36.thread_counts_ptr = shared_buf_StaticFieldPointsTo.data() + off_StaticFieldPointsTo_0;
        uint32_t off_StaticFieldPointsTo_1 = off_StaticFieldPointsTo_0 + p_36.num_threads;
        p_37.thread_counts_ptr = shared_buf_StaticFieldPointsTo.data() + off_StaticFieldPointsTo_1;
        uint32_t shared_buf_VarPointsTo_size = p_10.num_threads + p_11.num_threads + p_12.num_threads + p_13.num_threads + p_14.num_threads + p_15.num_threads + p_16.num_threads + p_18.num_threads + p_19.num_threads + p_20.num_threads + p_21.num_threads + p_22.num_threads + p_23.num_threads + p_24.num_threads + p_25.num_threads + 1;
        SRDatalog::GPU::DeviceArray<uint32_t> shared_buf_VarPointsTo(shared_buf_VarPointsTo_size);
        uint32_t off_VarPointsTo_0 = 0;
        p_10.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_0;
        uint32_t off_VarPointsTo_1 = off_VarPointsTo_0 + p_10.num_threads;
        p_11.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_1;
        uint32_t off_VarPointsTo_2 = off_VarPointsTo_1 + p_11.num_threads;
        p_12.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_2;
        uint32_t off_VarPointsTo_3 = off_VarPointsTo_2 + p_12.num_threads;
        p_13.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_3;
        uint32_t off_VarPointsTo_4 = off_VarPointsTo_3 + p_13.num_threads;
        p_14.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_4;
        uint32_t off_VarPointsTo_5 = off_VarPointsTo_4 + p_14.num_threads;
        p_15.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_5;
        uint32_t off_VarPointsTo_6 = off_VarPointsTo_5 + p_15.num_threads;
        p_16.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_6;
        uint32_t off_VarPointsTo_7 = off_VarPointsTo_6 + p_16.num_threads;
        p_18.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_7;
        uint32_t off_VarPointsTo_8 = off_VarPointsTo_7 + p_18.num_threads;
        p_19.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_8;
        uint32_t off_VarPointsTo_9 = off_VarPointsTo_8 + p_19.num_threads;
        p_20.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_9;
        uint32_t off_VarPointsTo_10 = off_VarPointsTo_9 + p_20.num_threads;
        p_21.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_10;
        uint32_t off_VarPointsTo_11 = off_VarPointsTo_10 + p_21.num_threads;
        p_22.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_11;
        uint32_t off_VarPointsTo_12 = off_VarPointsTo_11 + p_22.num_threads;
        p_23.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_12;
        uint32_t off_VarPointsTo_13 = off_VarPointsTo_12 + p_23.num_threads;
        p_24.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_13;
        uint32_t off_VarPointsTo_14 = off_VarPointsTo_13 + p_24.num_threads;
        p_25.thread_counts_ptr = shared_buf_VarPointsTo.data() + off_VarPointsTo_14;

        // Phase 2b: Launch count kernels (parallel streams)
        JitRunner_InitClass_Super_D0::launch_count(p_0, _stream_pool.get(0));
        JitRunner_InitClass_Iface_D0::launch_count(p_1, _stream_pool.get(1));
        JitRunner_InitClass_Heap_D0::launch_count(p_2, _stream_pool.get(2));
        JitRunner_InitClass_StaticInvoke_D0::launch_count(p_3, _stream_pool.get(3));
        JitRunner_InitClass_StoreStatic_D0::launch_count(p_4, _stream_pool.get(4));
        JitRunner_InitClass_LoadStatic_D0::launch_count(p_5, _stream_pool.get(5));
        JitRunner_Reachable_Clinit_D0::launch_count(p_6, _stream_pool.get(6));
        JitRunner_Static_MultiHead_D0::launch_count(p_7, _stream_pool.get(7));
        JitRunner_Assign_Param_D0::launch_count(p_8, _stream_pool.get(8));
        JitRunner_Assign_Return_D0::launch_count(p_9, _stream_pool.get(9));
        JitRunner_VPT_HeapAlloc_D1::launch_count(p_10, _stream_pool.get(10));
        JitRunner_VPT_Assign_D0::launch_count(p_11, _stream_pool.get(11));
        JitRunner_VPT_Assign_D1::launch_count(p_12, _stream_pool.get(12));
        JitRunner_VPT_Local_D0::launch_count(p_13, _stream_pool.get(13));
        JitRunner_VPT_Local_D2::launch_count(p_14, _stream_pool.get(14));
        JitRunner_VPT_Cast_D0::launch_count(p_15, _stream_pool.get(15));
        JitRunner_VPT_Cast_D2::launch_count(p_16, _stream_pool.get(16));
        JitRunner_ReachableLoadInstanceField_rule_D0::launch_count(p_17, _stream_pool.get(17));
        JitRunner_VPT_LoadField_D0::launch_count(p_18, _stream_pool.get(18));
        JitRunner_VPT_LoadField_D1::launch_count(p_19, _stream_pool.get(19));
        JitRunner_VPT_LoadField_D2::launch_count(p_20, _stream_pool.get(20));
        JitRunner_VPT_LoadStatic_D0::launch_count(p_21, _stream_pool.get(21));
        JitRunner_VPT_LoadStatic_D2::launch_count(p_22, _stream_pool.get(22));
        JitRunner_VPT_LoadArray_D0::launch_count(p_23, _stream_pool.get(23));
        JitRunner_VPT_LoadArray_D2::launch_count(p_24, _stream_pool.get(24));
        JitRunner_VPT_LoadArray_D3::launch_count(p_25, _stream_pool.get(25));
        JitRunner_ReachableInstructionHelper_D0::launch_count(p_26, _stream_pool.get(26));
        JitRunner_VirtualDispatch_MultiHead_D0::launch_count(p_27, _stream_pool.get(27));
        JitRunner_VirtualDispatch_MultiHead_D2::launch_count(p_28, _stream_pool.get(28));
        JitRunner_Reachable_CGE_Virtual_D0::launch_count(p_29, _stream_pool.get(29));
        JitRunner_Reachable_CGE_Virtual_D2::launch_count(p_30, _stream_pool.get(30));
        JitRunner_SpecialDispatch_MultiHead_D0::launch_count(p_31, _stream_pool.get(31));
        JitRunner_SpecialDispatch_MultiHead_D2::launch_count(p_32, _stream_pool.get(32));
        JitRunner_IFPT_Store_D0::launch_count(p_33, _stream_pool.get(33));
        JitRunner_IFPT_Store_D2::launch_count(p_34, _stream_pool.get(34));
        JitRunner_IFPT_Store_D3::launch_count(p_35, _stream_pool.get(35));
        JitRunner_SFPT_Store_D0::launch_count(p_36, _stream_pool.get(36));
        JitRunner_SFPT_Store_D2::launch_count(p_37, _stream_pool.get(37));
        JitRunner_ReachableSortedIndex_rule_D0::launch_count(p_38, _stream_pool.get(38));
        JitRunner_AIPT_Store_ObjectArray_D0::launch_count(p_39, _stream_pool.get(39));
        JitRunner_AIPT_Store_ObjectArray_D1::launch_count(p_40, _stream_pool.get(40));
        JitRunner_AIPT_Store_ObjectArray_D3::launch_count(p_41, _stream_pool.get(41));
        JitRunner_AIPT_Store_D0::launch_count(p_42, _stream_pool.get(42));
        JitRunner_AIPT_Store_D1::launch_count(p_43, _stream_pool.get(43));
        JitRunner_AIPT_Store_D2::launch_count(p_44, _stream_pool.get(44));
        _stream_pool.sync_all();

        // Phase 3a: Scan shared buffers (one per dest) + multi-head scans
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(39)), shared_buf_ArrayIndexPointsTo.data(), shared_buf_ArrayIndexPointsTo.data() + shared_buf_ArrayIndexPointsTo_size, shared_buf_ArrayIndexPointsTo.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(8)), shared_buf_Assign.data(), shared_buf_Assign.data() + shared_buf_Assign_size, shared_buf_Assign.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(0)), shared_buf_InitializedClass.data(), shared_buf_InitializedClass.data() + shared_buf_InitializedClass_size, shared_buf_InitializedClass.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(33)), shared_buf_InstanceFieldPointsTo.data(), shared_buf_InstanceFieldPointsTo.data() + shared_buf_InstanceFieldPointsTo_size, shared_buf_InstanceFieldPointsTo.data(), 0, thrust::plus<uint32_t>());
        JitRunner_Reachable_Clinit_D0::scan_only(p_6, _stream_pool.get(6));
        JitRunner_ReachableInstructionHelper_D0::scan_only(p_26, _stream_pool.get(26));
        JitRunner_ReachableLoadInstanceField_rule_D0::scan_only(p_17, _stream_pool.get(17));
        JitRunner_ReachableSortedIndex_rule_D0::scan_only(p_38, _stream_pool.get(38));
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(36)), shared_buf_StaticFieldPointsTo.data(), shared_buf_StaticFieldPointsTo.data() + shared_buf_StaticFieldPointsTo_size, shared_buf_StaticFieldPointsTo.data(), 0, thrust::plus<uint32_t>());
        thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get(10)), shared_buf_VarPointsTo.data(), shared_buf_VarPointsTo.data() + shared_buf_VarPointsTo_size, shared_buf_VarPointsTo.data(), 0, thrust::plus<uint32_t>());
        JitRunner_Static_MultiHead_D0::scan_only(p_7, _stream_pool.get(7));
        JitRunner_VirtualDispatch_MultiHead_D0::scan_only(p_27, _stream_pool.get(27));
        JitRunner_VirtualDispatch_MultiHead_D2::scan_only(p_28, _stream_pool.get(28));
        JitRunner_Reachable_CGE_Virtual_D0::scan_only(p_29, _stream_pool.get(29));
        JitRunner_Reachable_CGE_Virtual_D2::scan_only(p_30, _stream_pool.get(30));
        JitRunner_SpecialDispatch_MultiHead_D0::scan_only(p_31, _stream_pool.get(31));
        JitRunner_SpecialDispatch_MultiHead_D2::scan_only(p_32, _stream_pool.get(32));

        // Phase 3b: Single sync + readback all totals
        GPU_DEVICE_SYNCHRONIZE();
        // Read boundary values for ArrayIndexPointsTo (6 rules)
        uint32_t bnd_ArrayIndexPointsTo[7];
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[0], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[1], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[2], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[3], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[4], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_4, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[5], shared_buf_ArrayIndexPointsTo.data() + off_ArrayIndexPointsTo_5, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_ArrayIndexPointsTo[6], shared_buf_ArrayIndexPointsTo.data() + shared_buf_ArrayIndexPointsTo_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_39 = bnd_ArrayIndexPointsTo[1] - bnd_ArrayIndexPointsTo[0];
        uint32_t total_40 = bnd_ArrayIndexPointsTo[2] - bnd_ArrayIndexPointsTo[1];
        uint32_t total_41 = bnd_ArrayIndexPointsTo[3] - bnd_ArrayIndexPointsTo[2];
        uint32_t total_42 = bnd_ArrayIndexPointsTo[4] - bnd_ArrayIndexPointsTo[3];
        uint32_t total_43 = bnd_ArrayIndexPointsTo[5] - bnd_ArrayIndexPointsTo[4];
        uint32_t total_44 = bnd_ArrayIndexPointsTo[6] - bnd_ArrayIndexPointsTo[5];
        // Read boundary values for Assign (2 rules)
        uint32_t bnd_Assign[3];
        GPU_MEMCPY_ASYNC(&bnd_Assign[0], shared_buf_Assign.data() + off_Assign_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_Assign[1], shared_buf_Assign.data() + off_Assign_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_Assign[2], shared_buf_Assign.data() + shared_buf_Assign_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_8 = bnd_Assign[1] - bnd_Assign[0];
        uint32_t total_9 = bnd_Assign[2] - bnd_Assign[1];
        // Read boundary values for InitializedClass (6 rules)
        uint32_t bnd_InitializedClass[7];
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[0], shared_buf_InitializedClass.data() + off_InitializedClass_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[1], shared_buf_InitializedClass.data() + off_InitializedClass_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[2], shared_buf_InitializedClass.data() + off_InitializedClass_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[3], shared_buf_InitializedClass.data() + off_InitializedClass_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[4], shared_buf_InitializedClass.data() + off_InitializedClass_4, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[5], shared_buf_InitializedClass.data() + off_InitializedClass_5, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InitializedClass[6], shared_buf_InitializedClass.data() + shared_buf_InitializedClass_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_0 = bnd_InitializedClass[1] - bnd_InitializedClass[0];
        uint32_t total_1 = bnd_InitializedClass[2] - bnd_InitializedClass[1];
        uint32_t total_2 = bnd_InitializedClass[3] - bnd_InitializedClass[2];
        uint32_t total_3 = bnd_InitializedClass[4] - bnd_InitializedClass[3];
        uint32_t total_4 = bnd_InitializedClass[5] - bnd_InitializedClass[4];
        uint32_t total_5 = bnd_InitializedClass[6] - bnd_InitializedClass[5];
        // Read boundary values for InstanceFieldPointsTo (3 rules)
        uint32_t bnd_InstanceFieldPointsTo[4];
        GPU_MEMCPY_ASYNC(&bnd_InstanceFieldPointsTo[0], shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InstanceFieldPointsTo[1], shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InstanceFieldPointsTo[2], shared_buf_InstanceFieldPointsTo.data() + off_InstanceFieldPointsTo_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_InstanceFieldPointsTo[3], shared_buf_InstanceFieldPointsTo.data() + shared_buf_InstanceFieldPointsTo_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_33 = bnd_InstanceFieldPointsTo[1] - bnd_InstanceFieldPointsTo[0];
        uint32_t total_34 = bnd_InstanceFieldPointsTo[2] - bnd_InstanceFieldPointsTo[1];
        uint32_t total_35 = bnd_InstanceFieldPointsTo[3] - bnd_InstanceFieldPointsTo[2];
        uint32_t total_6 = JitRunner_Reachable_Clinit_D0::read_total(p_6);
        uint32_t total_26 = JitRunner_ReachableInstructionHelper_D0::read_total(p_26);
        uint32_t total_17 = JitRunner_ReachableLoadInstanceField_rule_D0::read_total(p_17);
        uint32_t total_38 = JitRunner_ReachableSortedIndex_rule_D0::read_total(p_38);
        // Read boundary values for StaticFieldPointsTo (2 rules)
        uint32_t bnd_StaticFieldPointsTo[3];
        GPU_MEMCPY_ASYNC(&bnd_StaticFieldPointsTo[0], shared_buf_StaticFieldPointsTo.data() + off_StaticFieldPointsTo_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_StaticFieldPointsTo[1], shared_buf_StaticFieldPointsTo.data() + off_StaticFieldPointsTo_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_StaticFieldPointsTo[2], shared_buf_StaticFieldPointsTo.data() + shared_buf_StaticFieldPointsTo_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_36 = bnd_StaticFieldPointsTo[1] - bnd_StaticFieldPointsTo[0];
        uint32_t total_37 = bnd_StaticFieldPointsTo[2] - bnd_StaticFieldPointsTo[1];
        // Read boundary values for VarPointsTo (15 rules)
        uint32_t bnd_VarPointsTo[16];
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[0], shared_buf_VarPointsTo.data() + off_VarPointsTo_0, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[1], shared_buf_VarPointsTo.data() + off_VarPointsTo_1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[2], shared_buf_VarPointsTo.data() + off_VarPointsTo_2, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[3], shared_buf_VarPointsTo.data() + off_VarPointsTo_3, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[4], shared_buf_VarPointsTo.data() + off_VarPointsTo_4, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[5], shared_buf_VarPointsTo.data() + off_VarPointsTo_5, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[6], shared_buf_VarPointsTo.data() + off_VarPointsTo_6, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[7], shared_buf_VarPointsTo.data() + off_VarPointsTo_7, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[8], shared_buf_VarPointsTo.data() + off_VarPointsTo_8, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[9], shared_buf_VarPointsTo.data() + off_VarPointsTo_9, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[10], shared_buf_VarPointsTo.data() + off_VarPointsTo_10, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[11], shared_buf_VarPointsTo.data() + off_VarPointsTo_11, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[12], shared_buf_VarPointsTo.data() + off_VarPointsTo_12, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[13], shared_buf_VarPointsTo.data() + off_VarPointsTo_13, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[14], shared_buf_VarPointsTo.data() + off_VarPointsTo_14, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        GPU_MEMCPY_ASYNC(&bnd_VarPointsTo[15], shared_buf_VarPointsTo.data() + shared_buf_VarPointsTo_size - 1, sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);
        uint32_t total_10 = bnd_VarPointsTo[1] - bnd_VarPointsTo[0];
        uint32_t total_11 = bnd_VarPointsTo[2] - bnd_VarPointsTo[1];
        uint32_t total_12 = bnd_VarPointsTo[3] - bnd_VarPointsTo[2];
        uint32_t total_13 = bnd_VarPointsTo[4] - bnd_VarPointsTo[3];
        uint32_t total_14 = bnd_VarPointsTo[5] - bnd_VarPointsTo[4];
        uint32_t total_15 = bnd_VarPointsTo[6] - bnd_VarPointsTo[5];
        uint32_t total_16 = bnd_VarPointsTo[7] - bnd_VarPointsTo[6];
        uint32_t total_18 = bnd_VarPointsTo[8] - bnd_VarPointsTo[7];
        uint32_t total_19 = bnd_VarPointsTo[9] - bnd_VarPointsTo[8];
        uint32_t total_20 = bnd_VarPointsTo[10] - bnd_VarPointsTo[9];
        uint32_t total_21 = bnd_VarPointsTo[11] - bnd_VarPointsTo[10];
        uint32_t total_22 = bnd_VarPointsTo[12] - bnd_VarPointsTo[11];
        uint32_t total_23 = bnd_VarPointsTo[13] - bnd_VarPointsTo[12];
        uint32_t total_24 = bnd_VarPointsTo[14] - bnd_VarPointsTo[13];
        uint32_t total_25 = bnd_VarPointsTo[15] - bnd_VarPointsTo[14];
        uint32_t total_7 = JitRunner_Static_MultiHead_D0::read_total(p_7);
        uint32_t total_27 = JitRunner_VirtualDispatch_MultiHead_D0::read_total(p_27);
        uint32_t total_28 = JitRunner_VirtualDispatch_MultiHead_D2::read_total(p_28);
        uint32_t total_29 = JitRunner_Reachable_CGE_Virtual_D0::read_total(p_29);
        uint32_t total_30 = JitRunner_Reachable_CGE_Virtual_D2::read_total(p_30);
        uint32_t total_31 = JitRunner_SpecialDispatch_MultiHead_D0::read_total(p_31);
        uint32_t total_32 = JitRunner_SpecialDispatch_MultiHead_D2::read_total(p_32);

        // Phase 3c: Resize once per unique dest + assign per-rule offsets
        {  // Resize ArrayIndexPointsTo
          uint32_t sum_ArrayIndexPointsTo = total_39 + total_40 + total_41 + total_42 + total_43 + total_44;
          auto& dest_ArrayIndexPointsTo = get_relation_by_schema<ArrayIndexPointsTo, NEW_VER>(db);
          uint32_t base_offset_ArrayIndexPointsTo = static_cast<uint32_t>(dest_ArrayIndexPointsTo.size());
          if (sum_ArrayIndexPointsTo > 0) dest_ArrayIndexPointsTo.resize_interned_columns(base_offset_ArrayIndexPointsTo + sum_ArrayIndexPointsTo);
          p_39.old_size_0 = base_offset_ArrayIndexPointsTo;
          p_40.old_size_0 = base_offset_ArrayIndexPointsTo;
          p_41.old_size_0 = base_offset_ArrayIndexPointsTo;
          p_42.old_size_0 = base_offset_ArrayIndexPointsTo;
          p_43.old_size_0 = base_offset_ArrayIndexPointsTo;
          p_44.old_size_0 = base_offset_ArrayIndexPointsTo;
        }
        {  // Resize Assign
          uint32_t sum_Assign = total_8 + total_9;
          auto& dest_Assign = get_relation_by_schema<Assign, NEW_VER>(db);
          uint32_t base_offset_Assign = static_cast<uint32_t>(dest_Assign.size());
          if (sum_Assign > 0) dest_Assign.resize_interned_columns(base_offset_Assign + sum_Assign);
          p_8.old_size_0 = base_offset_Assign;
          p_9.old_size_0 = base_offset_Assign;
        }
        {  // Resize CallGraphEdge
          uint32_t sum_CallGraphEdge = total_7 + total_27 + total_28 + total_29 + total_30 + total_31 + total_32;
          auto& dest_CallGraphEdge = get_relation_by_schema<CallGraphEdge, NEW_VER>(db);
          uint32_t base_offset_CallGraphEdge = static_cast<uint32_t>(dest_CallGraphEdge.size());
          if (sum_CallGraphEdge > 0) dest_CallGraphEdge.resize_interned_columns(base_offset_CallGraphEdge + sum_CallGraphEdge);
          uint32_t running_offset_CallGraphEdge = base_offset_CallGraphEdge;
          p_7.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_7;
          p_27.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_27;
          p_28.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_28;
          p_29.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_29;
          p_30.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_30;
          p_31.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_31;
          p_32.old_size_1 = running_offset_CallGraphEdge;
          running_offset_CallGraphEdge += total_32;
        }
        {  // Resize InitializedClass
          uint32_t sum_InitializedClass = total_0 + total_1 + total_2 + total_3 + total_4 + total_5;
          auto& dest_InitializedClass = get_relation_by_schema<InitializedClass, NEW_VER>(db);
          uint32_t base_offset_InitializedClass = static_cast<uint32_t>(dest_InitializedClass.size());
          if (sum_InitializedClass > 0) dest_InitializedClass.resize_interned_columns(base_offset_InitializedClass + sum_InitializedClass);
          p_0.old_size_0 = base_offset_InitializedClass;
          p_1.old_size_0 = base_offset_InitializedClass;
          p_2.old_size_0 = base_offset_InitializedClass;
          p_3.old_size_0 = base_offset_InitializedClass;
          p_4.old_size_0 = base_offset_InitializedClass;
          p_5.old_size_0 = base_offset_InitializedClass;
        }
        {  // Resize InstanceFieldPointsTo
          uint32_t sum_InstanceFieldPointsTo = total_33 + total_34 + total_35;
          auto& dest_InstanceFieldPointsTo = get_relation_by_schema<InstanceFieldPointsTo, NEW_VER>(db);
          uint32_t base_offset_InstanceFieldPointsTo = static_cast<uint32_t>(dest_InstanceFieldPointsTo.size());
          if (sum_InstanceFieldPointsTo > 0) dest_InstanceFieldPointsTo.resize_interned_columns(base_offset_InstanceFieldPointsTo + sum_InstanceFieldPointsTo);
          p_33.old_size_0 = base_offset_InstanceFieldPointsTo;
          p_34.old_size_0 = base_offset_InstanceFieldPointsTo;
          p_35.old_size_0 = base_offset_InstanceFieldPointsTo;
        }
        {  // Resize Reachable
          uint32_t sum_Reachable = total_6 + total_7 + total_27 + total_28 + total_29 + total_30 + total_31 + total_32;
          auto& dest_Reachable = get_relation_by_schema<Reachable, NEW_VER>(db);
          uint32_t base_offset_Reachable = static_cast<uint32_t>(dest_Reachable.size());
          if (sum_Reachable > 0) dest_Reachable.resize_interned_columns(base_offset_Reachable + sum_Reachable);
          uint32_t running_offset_Reachable = base_offset_Reachable;
          p_6.old_size_0 = running_offset_Reachable;
          running_offset_Reachable += total_6;
          p_7.old_size_0 = running_offset_Reachable;
          running_offset_Reachable += total_7;
          p_27.old_size_2 = running_offset_Reachable;
          running_offset_Reachable += total_27;
          p_28.old_size_2 = running_offset_Reachable;
          running_offset_Reachable += total_28;
          p_29.old_size_0 = running_offset_Reachable;
          running_offset_Reachable += total_29;
          p_30.old_size_0 = running_offset_Reachable;
          running_offset_Reachable += total_30;
          p_31.old_size_2 = running_offset_Reachable;
          running_offset_Reachable += total_31;
          p_32.old_size_2 = running_offset_Reachable;
          running_offset_Reachable += total_32;
        }
        {  // Resize ReachableInstruction
          uint32_t sum_ReachableInstruction = total_26;
          auto& dest_ReachableInstruction = get_relation_by_schema<ReachableInstruction, NEW_VER>(db);
          uint32_t base_offset_ReachableInstruction = static_cast<uint32_t>(dest_ReachableInstruction.size());
          if (sum_ReachableInstruction > 0) dest_ReachableInstruction.resize_interned_columns(base_offset_ReachableInstruction + sum_ReachableInstruction);
          uint32_t running_offset_ReachableInstruction = base_offset_ReachableInstruction;
          p_26.old_size_0 = running_offset_ReachableInstruction;
          running_offset_ReachableInstruction += total_26;
        }
        {  // Resize ReachableLoadInstanceField
          uint32_t sum_ReachableLoadInstanceField = total_17;
          auto& dest_ReachableLoadInstanceField = get_relation_by_schema<ReachableLoadInstanceField, NEW_VER>(db);
          uint32_t base_offset_ReachableLoadInstanceField = static_cast<uint32_t>(dest_ReachableLoadInstanceField.size());
          if (sum_ReachableLoadInstanceField > 0) dest_ReachableLoadInstanceField.resize_interned_columns(base_offset_ReachableLoadInstanceField + sum_ReachableLoadInstanceField);
          uint32_t running_offset_ReachableLoadInstanceField = base_offset_ReachableLoadInstanceField;
          p_17.old_size_0 = running_offset_ReachableLoadInstanceField;
          running_offset_ReachableLoadInstanceField += total_17;
        }
        {  // Resize ReachableSortedIndex
          uint32_t sum_ReachableSortedIndex = total_38;
          auto& dest_ReachableSortedIndex = get_relation_by_schema<ReachableSortedIndex, NEW_VER>(db);
          uint32_t base_offset_ReachableSortedIndex = static_cast<uint32_t>(dest_ReachableSortedIndex.size());
          if (sum_ReachableSortedIndex > 0) dest_ReachableSortedIndex.resize_interned_columns(base_offset_ReachableSortedIndex + sum_ReachableSortedIndex);
          uint32_t running_offset_ReachableSortedIndex = base_offset_ReachableSortedIndex;
          p_38.old_size_0 = running_offset_ReachableSortedIndex;
          running_offset_ReachableSortedIndex += total_38;
        }
        {  // Resize StaticFieldPointsTo
          uint32_t sum_StaticFieldPointsTo = total_36 + total_37;
          auto& dest_StaticFieldPointsTo = get_relation_by_schema<StaticFieldPointsTo, NEW_VER>(db);
          uint32_t base_offset_StaticFieldPointsTo = static_cast<uint32_t>(dest_StaticFieldPointsTo.size());
          if (sum_StaticFieldPointsTo > 0) dest_StaticFieldPointsTo.resize_interned_columns(base_offset_StaticFieldPointsTo + sum_StaticFieldPointsTo);
          p_36.old_size_0 = base_offset_StaticFieldPointsTo;
          p_37.old_size_0 = base_offset_StaticFieldPointsTo;
        }
        {  // Resize VarPointsTo
          uint32_t sum_VarPointsTo = total_10 + total_11 + total_12 + total_13 + total_14 + total_15 + total_16 + total_18 + total_19 + total_20 + total_21 + total_22 + total_23 + total_24 + total_25 + total_27 + total_28 + total_31 + total_32;
          auto& dest_VarPointsTo = get_relation_by_schema<VarPointsTo, NEW_VER>(db);
          uint32_t base_offset_VarPointsTo = static_cast<uint32_t>(dest_VarPointsTo.size());
          if (sum_VarPointsTo > 0) dest_VarPointsTo.resize_interned_columns(base_offset_VarPointsTo + sum_VarPointsTo);
          p_10.old_size_0 = base_offset_VarPointsTo;
          p_11.old_size_0 = base_offset_VarPointsTo;
          p_12.old_size_0 = base_offset_VarPointsTo;
          p_13.old_size_0 = base_offset_VarPointsTo;
          p_14.old_size_0 = base_offset_VarPointsTo;
          p_15.old_size_0 = base_offset_VarPointsTo;
          p_16.old_size_0 = base_offset_VarPointsTo;
          p_18.old_size_0 = base_offset_VarPointsTo;
          p_19.old_size_0 = base_offset_VarPointsTo;
          p_20.old_size_0 = base_offset_VarPointsTo;
          p_21.old_size_0 = base_offset_VarPointsTo;
          p_22.old_size_0 = base_offset_VarPointsTo;
          p_23.old_size_0 = base_offset_VarPointsTo;
          p_24.old_size_0 = base_offset_VarPointsTo;
          p_25.old_size_0 = base_offset_VarPointsTo;
          uint32_t running_offset_VarPointsTo = base_offset_VarPointsTo + total_10 + total_11 + total_12 + total_13 + total_14 + total_15 + total_16 + total_18 + total_19 + total_20 + total_21 + total_22 + total_23 + total_24 + total_25;
          p_27.old_size_0 = running_offset_VarPointsTo;
          running_offset_VarPointsTo += total_27;
          p_28.old_size_0 = running_offset_VarPointsTo;
          running_offset_VarPointsTo += total_28;
          p_31.old_size_0 = running_offset_VarPointsTo;
          running_offset_VarPointsTo += total_31;
          p_32.old_size_0 = running_offset_VarPointsTo;
          running_offset_VarPointsTo += total_32;
        }

        if (total_0 > 0) JitRunner_InitClass_Super_D0::launch_materialize(db, p_0, total_0, _stream_pool.get(0));
        if (total_1 > 0) JitRunner_InitClass_Iface_D0::launch_materialize(db, p_1, total_1, _stream_pool.get(1));
        if (total_2 > 0) JitRunner_InitClass_Heap_D0::launch_materialize(db, p_2, total_2, _stream_pool.get(2));
        if (total_3 > 0) JitRunner_InitClass_StaticInvoke_D0::launch_materialize(db, p_3, total_3, _stream_pool.get(3));
        if (total_4 > 0) JitRunner_InitClass_StoreStatic_D0::launch_materialize(db, p_4, total_4, _stream_pool.get(4));
        if (total_5 > 0) JitRunner_InitClass_LoadStatic_D0::launch_materialize(db, p_5, total_5, _stream_pool.get(5));
        if (total_6 > 0) JitRunner_Reachable_Clinit_D0::launch_materialize(db, p_6, total_6, _stream_pool.get(6));
        if (total_7 > 0) JitRunner_Static_MultiHead_D0::launch_materialize(db, p_7, total_7, _stream_pool.get(7));
        if (total_8 > 0) JitRunner_Assign_Param_D0::launch_materialize(db, p_8, total_8, _stream_pool.get(8));
        if (total_9 > 0) JitRunner_Assign_Return_D0::launch_materialize(db, p_9, total_9, _stream_pool.get(9));
        if (total_10 > 0) JitRunner_VPT_HeapAlloc_D1::launch_materialize(db, p_10, total_10, _stream_pool.get(10));
        if (total_11 > 0) JitRunner_VPT_Assign_D0::launch_materialize(db, p_11, total_11, _stream_pool.get(11));
        if (total_12 > 0) JitRunner_VPT_Assign_D1::launch_materialize(db, p_12, total_12, _stream_pool.get(12));
        if (total_13 > 0) JitRunner_VPT_Local_D0::launch_materialize(db, p_13, total_13, _stream_pool.get(13));
        if (total_14 > 0) JitRunner_VPT_Local_D2::launch_materialize(db, p_14, total_14, _stream_pool.get(14));
        if (total_15 > 0) JitRunner_VPT_Cast_D0::launch_materialize(db, p_15, total_15, _stream_pool.get(15));
        if (total_16 > 0) JitRunner_VPT_Cast_D2::launch_materialize(db, p_16, total_16, _stream_pool.get(16));
        if (total_17 > 0) JitRunner_ReachableLoadInstanceField_rule_D0::launch_materialize(db, p_17, total_17, _stream_pool.get(17));
        if (total_18 > 0) JitRunner_VPT_LoadField_D0::launch_materialize(db, p_18, total_18, _stream_pool.get(18));
        if (total_19 > 0) JitRunner_VPT_LoadField_D1::launch_materialize(db, p_19, total_19, _stream_pool.get(19));
        if (total_20 > 0) JitRunner_VPT_LoadField_D2::launch_materialize(db, p_20, total_20, _stream_pool.get(20));
        if (total_21 > 0) JitRunner_VPT_LoadStatic_D0::launch_materialize(db, p_21, total_21, _stream_pool.get(21));
        if (total_22 > 0) JitRunner_VPT_LoadStatic_D2::launch_materialize(db, p_22, total_22, _stream_pool.get(22));
        if (total_23 > 0) JitRunner_VPT_LoadArray_D0::launch_materialize(db, p_23, total_23, _stream_pool.get(23));
        if (total_24 > 0) JitRunner_VPT_LoadArray_D2::launch_materialize(db, p_24, total_24, _stream_pool.get(24));
        if (total_25 > 0) JitRunner_VPT_LoadArray_D3::launch_materialize(db, p_25, total_25, _stream_pool.get(25));
        if (total_26 > 0) JitRunner_ReachableInstructionHelper_D0::launch_materialize(db, p_26, total_26, _stream_pool.get(26));
        if (total_27 > 0) JitRunner_VirtualDispatch_MultiHead_D0::launch_materialize(db, p_27, total_27, _stream_pool.get(27));
        if (total_28 > 0) JitRunner_VirtualDispatch_MultiHead_D2::launch_materialize(db, p_28, total_28, _stream_pool.get(28));
        if (total_29 > 0) JitRunner_Reachable_CGE_Virtual_D0::launch_materialize(db, p_29, total_29, _stream_pool.get(29));
        if (total_30 > 0) JitRunner_Reachable_CGE_Virtual_D2::launch_materialize(db, p_30, total_30, _stream_pool.get(30));
        if (total_31 > 0) JitRunner_SpecialDispatch_MultiHead_D0::launch_materialize(db, p_31, total_31, _stream_pool.get(31));
        if (total_32 > 0) JitRunner_SpecialDispatch_MultiHead_D2::launch_materialize(db, p_32, total_32, _stream_pool.get(32));
        if (total_33 > 0) JitRunner_IFPT_Store_D0::launch_materialize(db, p_33, total_33, _stream_pool.get(33));
        if (total_34 > 0) JitRunner_IFPT_Store_D2::launch_materialize(db, p_34, total_34, _stream_pool.get(34));
        if (total_35 > 0) JitRunner_IFPT_Store_D3::launch_materialize(db, p_35, total_35, _stream_pool.get(35));
        if (total_36 > 0) JitRunner_SFPT_Store_D0::launch_materialize(db, p_36, total_36, _stream_pool.get(36));
        if (total_37 > 0) JitRunner_SFPT_Store_D2::launch_materialize(db, p_37, total_37, _stream_pool.get(37));
        if (total_38 > 0) JitRunner_ReachableSortedIndex_rule_D0::launch_materialize(db, p_38, total_38, _stream_pool.get(38));
        if (total_39 > 0) JitRunner_AIPT_Store_ObjectArray_D0::launch_materialize(db, p_39, total_39, _stream_pool.get(39));
        if (total_40 > 0) JitRunner_AIPT_Store_ObjectArray_D1::launch_materialize(db, p_40, total_40, _stream_pool.get(40));
        if (total_41 > 0) JitRunner_AIPT_Store_ObjectArray_D3::launch_materialize(db, p_41, total_41, _stream_pool.get(41));
        if (total_42 > 0) JitRunner_AIPT_Store_D0::launch_materialize(db, p_42, total_42, _stream_pool.get(42));
        if (total_43 > 0) JitRunner_AIPT_Store_D1::launch_materialize(db, p_43, total_43, _stream_pool.get(43));
        if (total_44 > 0) JitRunner_AIPT_Store_D2::launch_materialize(db, p_44, total_44, _stream_pool.get(44));
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
        _stream_pool.record_event(11);
        _stream_pool.record_event(12);
        _stream_pool.record_event(13);
        _stream_pool.record_event(14);
        _stream_pool.record_event(15);
        _stream_pool.record_event(16);
        _stream_pool.record_event(17);
        _stream_pool.record_event(18);
        _stream_pool.record_event(19);
        _stream_pool.record_event(20);
        _stream_pool.record_event(21);
        _stream_pool.record_event(22);
        _stream_pool.record_event(23);
        _stream_pool.record_event(24);
        _stream_pool.record_event(25);
        _stream_pool.record_event(26);
        _stream_pool.record_event(27);
        _stream_pool.record_event(28);
        _stream_pool.record_event(29);
        _stream_pool.record_event(30);
        _stream_pool.record_event(31);
        _stream_pool.record_event(32);
        _stream_pool.record_event(33);
        _stream_pool.record_event(34);
        _stream_pool.record_event(35);
        _stream_pool.record_event(36);
        _stream_pool.record_event(37);
        _stream_pool.record_event(38);
        _stream_pool.record_event(39);
        _stream_pool.record_event(40);
        _stream_pool.record_event(41);
        _stream_pool.record_event(42);
        _stream_pool.record_event(43);
        _stream_pool.record_event(44);
        nvtxRangePop();  // join_pipeline
      }
      }
      GPU_DEVICE_SYNCHRONIZE(); // sync all materialize streams
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ArrayIndexPointsTo, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<ArrayIndexPointsTo, NEW_VER, ArrayIndexPointsTo_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ArrayIndexPointsTo, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Assign, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Assign, NEW_VER, Assign_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Assign, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<CallGraphEdge, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<CallGraphEdge, NEW_VER, CallGraphEdge_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<CallGraphEdge, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<InitializedClass, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<InitializedClass, NEW_VER, InitializedClass_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<InitializedClass, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<InstanceFieldPointsTo, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<InstanceFieldPointsTo, NEW_VER, InstanceFieldPointsTo_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<InstanceFieldPointsTo, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, DELTA_VER>, SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 1, 2, 0>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 1, 2, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Reachable, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Reachable, NEW_VER, Reachable_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Reachable, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableInstruction, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<ReachableInstruction, NEW_VER, ReachableInstruction_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableInstruction, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableLoadInstanceField, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<ReachableLoadInstanceField, NEW_VER, ReachableLoadInstanceField_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableLoadInstanceField, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>, SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 1, 0, 2>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableSortedIndex, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<ReachableSortedIndex, NEW_VER, ReachableSortedIndex_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<ReachableSortedIndex, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<StaticFieldPointsTo, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<StaticFieldPointsTo, NEW_VER, StaticFieldPointsTo_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<StaticFieldPointsTo, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<VarPointsTo, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<VarPointsTo, NEW_VER, VarPointsTo_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<VarPointsTo, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>, SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_49(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ArrayIndexPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_50(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_51(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<CallGraphEdge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_52(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<InitializedClass, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_53(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<InstanceFieldPointsTo, std::integer_sequence<int, 2, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_54(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Reachable, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_55(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableInstruction, std::integer_sequence<int, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_56(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableLoadInstanceField, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_57(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ReachableSortedIndex, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_58(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<StaticFieldPointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_59(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<VarPointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
