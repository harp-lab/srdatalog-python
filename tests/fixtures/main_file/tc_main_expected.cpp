using TCPlan_DB = AST::Database<Edge, ArcInput, Path>;
using namespace SRDatalog::mir::dsl;
// Device DB type alias (matches batch files)
using TCPlan_DB_Blueprint = SRDatalog::AST::Database<Edge, ArcInput, Path>;
using TCPlan_DB_DeviceDB = SRDatalog::AST::SemiNaiveDatabase<TCPlan_DB_Blueprint, SRDatalog::GPU::DeviceRelationType>;

#include "gpu/runtime/gpu_mir_helpers.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"
#include "gpu/runtime/stream_pool.h"
using namespace SRDatalog::GPU;

// Forward declaration - defined in JIT batch file
// See: /home/stargazermiao/.cache/nim/jit/TCPlan_DB_2C07/jit_batch_*.cpp
struct JitRunner_EdgeLoad {
  using DB = TCPlan_DB_DeviceDB;
  using FirstSchema = ArcInput;
  using DestSchema = Edge;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 2;
  static constexpr std::size_t OutputArity = OutputArity_0;
  static constexpr std::size_t NumSources = 1;

  struct LaunchParams {
    std::vector<ViewType> views_vec;
    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};
    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};
    uint32_t* thread_counts_ptr = nullptr;
    uint32_t num_unique_root_keys = 0;
    uint32_t num_root_keys = 0;
    const ValueType* root_unique_values_ptr = nullptr;
    const ValueType* head_root_unique_values_ptr = nullptr;
    uint32_t num_full_unique_root_keys = 0;
    uint32_t num_head_unique_root_keys = 0;
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t old_size_0 = 0;
    // Fused execution state (set by orchestrator)
    uint32_t* fused_wp_ptr_0 = nullptr;
    uint32_t* fused_of_ptr = nullptr;
    uint32_t fused_capacity = 0;
    uint32_t fused_h_wp_0 = 0;
    bool fused_overflow = false;
  };

  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  static void execute(DB& db, uint32_t iteration);
  static void execute_fused(DB& db, uint32_t iteration);
  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void read_fused_result(LaunchParams& p);
  static inline uint32_t prev_fused_total_ = 4096;
};


// Forward declaration - defined in JIT batch file
// See: /home/stargazermiao/.cache/nim/jit/TCPlan_DB_2C07/jit_batch_*.cpp
struct JitRunner_TCBase {
  using DB = TCPlan_DB_DeviceDB;
  using FirstSchema = Edge;
  using DestSchema = Path;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 2;
  static constexpr std::size_t OutputArity = OutputArity_0;
  static constexpr std::size_t NumSources = 1;

  struct LaunchParams {
    std::vector<ViewType> views_vec;
    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};
    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};
    uint32_t* thread_counts_ptr = nullptr;
    uint32_t num_unique_root_keys = 0;
    uint32_t num_root_keys = 0;
    const ValueType* root_unique_values_ptr = nullptr;
    const ValueType* head_root_unique_values_ptr = nullptr;
    uint32_t num_full_unique_root_keys = 0;
    uint32_t num_head_unique_root_keys = 0;
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t old_size_0 = 0;
    // Fused execution state (set by orchestrator)
    uint32_t* fused_wp_ptr_0 = nullptr;
    uint32_t* fused_of_ptr = nullptr;
    uint32_t fused_capacity = 0;
    uint32_t fused_h_wp_0 = 0;
    bool fused_overflow = false;
  };

  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  static void execute(DB& db, uint32_t iteration);
  static void execute_fused(DB& db, uint32_t iteration);
  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void read_fused_result(LaunchParams& p);
  static inline uint32_t prev_fused_total_ = 4096;
};


// Forward declaration - defined in JIT batch file
// See: /home/stargazermiao/.cache/nim/jit/TCPlan_DB_2C07/jit_batch_*.cpp
struct JitRunner_TCRec_D0 {
  using DB = TCPlan_DB_DeviceDB;
  using FirstSchema = Path;
  using DestSchema = Path;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, DELTA_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 2;
  static constexpr std::size_t OutputArity = OutputArity_0;
  static constexpr std::size_t NumSources = 2;

  struct LaunchParams {
    std::vector<ViewType> views_vec;
    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};
    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};
    uint32_t* thread_counts_ptr = nullptr;
    uint32_t num_unique_root_keys = 0;
    uint32_t num_root_keys = 0;
    const ValueType* root_unique_values_ptr = nullptr;
    const ValueType* head_root_unique_values_ptr = nullptr;
    uint32_t num_full_unique_root_keys = 0;
    uint32_t num_head_unique_root_keys = 0;
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;
    uint32_t old_size_0 = 0;
    // Fused execution state (set by orchestrator)
    uint32_t* fused_wp_ptr_0 = nullptr;
    uint32_t* fused_of_ptr = nullptr;
    uint32_t fused_capacity = 0;
    uint32_t fused_h_wp_0 = 0;
    bool fused_overflow = false;
  };

  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  static void execute(DB& db, uint32_t iteration);
  static void execute_fused(DB& db, uint32_t iteration);
  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void read_fused_result(LaunchParams& p);
  static inline uint32_t prev_fused_total_ = 4096;
};


namespace TCPlan_Plans {
}

struct TCPlan_Runner {
  using DB = TCPlan_DB;

  template <typename DB>
  static void load_data(DB& db, std::string root_dir) {
    SRDatalog::load_from_file<ArcInput>(db, root_dir + "/Arc.csv");
  }

  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, NEW_VER>>(db, 0);

    using Edge_canonical_spec_t = SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ArcInput, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_EdgeLoad::execute_fused(db, 0); else JitRunner_EdgeLoad::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Edge, NEW_VER, Edge_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, NEW_VER>, SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>, SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Edge, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_2(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);

    using Path_canonical_spec_t = SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_TCBase::execute_fused(db, 0); else JitRunner_TCBase::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<Path, NEW_VER, Path_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<Path, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_3(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void step_4(DB& db, std::size_t max_iterations) {
    using Path_canonical_spec_t = SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>;

    // Build all required indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Edge, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);


    bool _tail_mode = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter > 0) {
        std::size_t total_new_facts = 0;
        {
          auto& delta_rel = get_relation_by_schema<Path, DELTA_VER>(db);
          SRDatalog::IndexSpec spec{{1, 0}};
          try {
            auto& delta_idx = delta_rel.get_index(spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {}
        }
        if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {
          std::cerr << "[iter " << iter << "]";
          {
            auto& _dr = get_relation_by_schema<Path, DELTA_VER>(db);
            SRDatalog::IndexSpec _sp{{1, 0}};
            try { auto& _di = _dr.get_index(_sp); std::cerr << " Path=" << _di.root().degree(); } catch (...) { std::cerr << " Path=?"; }
          }
          std::cerr << std::endl;
        } else if (std::getenv("SRDATALOG_PRINT_DELTA")) {
          std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;
        }
        if (total_new_facts == 0) break;
        _tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) && (total_new_facts < 1000);
      }

      if (_tail_mode) JitRunner_TCRec_D0::execute_fused(db, static_cast<uint32_t>(iter)); else JitRunner_TCRec_D0::execute(db, static_cast<uint32_t>(iter));
      SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Path, DELTA_VER>(db);
      SRDatalog::GPU::mir_helpers::check_size_fn<Path, NEW_VER, Path_canonical_spec_t>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
      nvtxRangePop();  // merge
      SRDatalog::GPU::mir_helpers::clear_relation_fn<Path, NEW_VER>(db);
      nvtxRangePushA("merge");
      SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
      nvtxRangePop();  // merge
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Reconstruct intern columns from canonical index
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
  }
  template <typename DB>
  static void step_5(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<Path, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void run(DB& db, std::size_t max_iterations = std::numeric_limits<int>::max()) {
    auto step_0_start = std::chrono::high_resolution_clock::now();
    step_0(db, max_iterations);
    auto step_0_end = std::chrono::high_resolution_clock::now();
    auto step_0_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_0_end - step_0_start);
    std::cout << "[Step 0 (simple)] " << "Relations: Edge" << " completed in " << step_0_duration.count() << " ms" << std::endl;
    auto step_1_start = std::chrono::high_resolution_clock::now();
    step_1(db, max_iterations);
    auto step_1_end = std::chrono::high_resolution_clock::now();
    auto step_1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_1_end - step_1_start);
    std::cout << "[Step 1 (simple)] " << " completed in " << step_1_duration.count() << " ms" << std::endl;
    auto step_2_start = std::chrono::high_resolution_clock::now();
    step_2(db, max_iterations);
    auto step_2_end = std::chrono::high_resolution_clock::now();
    auto step_2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_2_end - step_2_start);
    std::cout << "[Step 2 (simple)] " << "Relations: Path" << " completed in " << step_2_duration.count() << " ms" << std::endl;
    auto step_3_start = std::chrono::high_resolution_clock::now();
    step_3(db, max_iterations);
    auto step_3_end = std::chrono::high_resolution_clock::now();
    auto step_3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_3_end - step_3_start);
    std::cout << "[Step 3 (simple)] " << " completed in " << step_3_duration.count() << " ms" << std::endl;
    auto step_4_start = std::chrono::high_resolution_clock::now();
    step_4(db, max_iterations);
    auto step_4_end = std::chrono::high_resolution_clock::now();
    auto step_4_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_4_end - step_4_start);
    std::cout << "[Step 4 (recursive)] " << "Relations: Path" << " completed in " << step_4_duration.count() << " ms" << std::endl;
    auto step_5_start = std::chrono::high_resolution_clock::now();
    step_5(db, max_iterations);
    auto step_5_end = std::chrono::high_resolution_clock::now();
    auto step_5_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_5_end - step_5_start);
    std::cout << "[Step 5 (simple)] " << " completed in " << step_5_duration.count() << " ms" << std::endl;
    {
      SRDatalog::IndexSpec canonical_idx{1, 0};
      auto& rel = get_relation_by_schema<Path, FULL_VER>(db);
      if (rel.has_index(canonical_idx)) {
        auto& idx = rel.get_index(canonical_idx);
        std::cout << " >>>>>>>>>>>>>>>>> Path : " << idx.root().degree() << std::endl;
      } else {
        std::cout << " >>>>>>>>>>>>>>>>> Path : [Index Missing]" << std::endl;
      }
    }
  }
};

// ======== JIT File-Based Compilation ========
// JIT kernels in 1 batch files
