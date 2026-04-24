using TrianglePlan_DB = AST::Database<ZRel, RRel, SRel, TRel>;
using namespace SRDatalog::mir::dsl;
// Device DB type alias (matches batch files)
using TrianglePlan_DB_Blueprint = SRDatalog::AST::Database<ZRel, RRel, SRel, TRel>;
using TrianglePlan_DB_DeviceDB = SRDatalog::AST::SemiNaiveDatabase<TrianglePlan_DB_Blueprint, SRDatalog::GPU::DeviceRelationType>;

#include "gpu/runtime/gpu_mir_helpers.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"
#include "gpu/runtime/stream_pool.h"
using namespace SRDatalog::GPU;

// Forward declaration - defined in JIT batch file
// See: /home/stargazermiao/.cache/nim/jit/TrianglePlan_DB_FC06/jit_batch_*.cpp
struct JitRunner_Triangle {
  using DB = TrianglePlan_DB_DeviceDB;
  using FirstSchema = RRel;
  using DestSchema = ZRel;
  using SR = NoProvenance;
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  static constexpr std::size_t OutputArity_0 = 3;
  static constexpr std::size_t OutputArity = OutputArity_0;
  static constexpr std::size_t NumSources = 3;

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


namespace TrianglePlan_Plans {
}

struct TrianglePlan_Runner {
  using DB = TrianglePlan_DB;

  template <typename DB>
  static void load_data(DB& db, std::string root_dir) {
  }

  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    // Build output indexes
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db, 0);

    using ZRel_canonical_spec_t = SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
    bool _tail_mode = false;

    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RRel, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<RRel, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0, 2>, FULL_VER>>(db, 0);
    if (_tail_mode) JitRunner_Triangle::execute_fused(db, 0); else JitRunner_Triangle::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<ZRel, NEW_VER, ZRel_canonical_spec_t>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, NEW_VER>, SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>, SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, DELTA_VER>>(db);
    nvtxRangePop();  // merge
    SRDatalog::GPU::mir_helpers::clear_relation_fn<ZRel, NEW_VER>(db);
    nvtxRangePushA("merge");
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    nvtxRangePop();  // merge
  }
  template <typename DB>
  static void step_1(DB& db, std::size_t max_iterations) {
    mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);
    GPU_DEVICE_SYNCHRONIZE();
  }
  template <typename DB>
  static void run(DB& db, std::size_t max_iterations = std::numeric_limits<int>::max()) {
    auto step_0_start = std::chrono::high_resolution_clock::now();
    step_0(db, max_iterations);
    auto step_0_end = std::chrono::high_resolution_clock::now();
    auto step_0_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_0_end - step_0_start);
    std::cout << "[Step 0 (simple)] " << "Relations: ZRel" << " completed in " << step_0_duration.count() << " ms" << std::endl;
    auto step_1_start = std::chrono::high_resolution_clock::now();
    step_1(db, max_iterations);
    auto step_1_end = std::chrono::high_resolution_clock::now();
    auto step_1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_1_end - step_1_start);
    std::cout << "[Step 1 (simple)] " << " completed in " << step_1_duration.count() << " ms" << std::endl;
  }
};

// ======== JIT File-Based Compilation ========
// JIT kernels in 1 batch files
