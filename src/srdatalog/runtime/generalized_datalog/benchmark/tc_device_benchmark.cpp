
#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gpu/init.h"
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include "gpu/runtime/query.h"

namespace sd = SRDatalog;
namespace ast = sd::AST;

using ast::database;
using ast::fixpoint;
using ast::non_iterative;
using ast::rel;
using ast::relation;
using ast::SemiNaiveDatabase;
using ast::operator""_v;
using ast::Literals::operator""_s;

using ::BooleanSR;
using sd::execute_query;
using sd::get_relation_by_schema;
using sd::load_from_file;
using sd::TestUtil::find_project_root;

using SR = BooleanSR;

// -----------------------------------------------------------------------------
// 1. Definition
// -----------------------------------------------------------------------------

using EdgeSchema = relation<decltype("Edge"_s), SR, int, int>;
using PathSchema = relation<decltype("Path"_s), SR, int, int>;
using TCBlueprint = database<EdgeSchema, PathSchema>;

// Variables
constexpr auto x = "x"_v;
constexpr auto y = "y"_v;
constexpr auto z = "z"_v;

// Relation accessors
constexpr auto edge = rel<EdgeSchema>;
constexpr auto path = rel<PathSchema>;

// Base rule: Path(x, y) :- Edge(x, y)
const auto tc_base = path.newt(x, y) <<= edge.full(x, y);

// Transitive rule: Path(x, z) :- Path(x, y), Edge(y, z)
const auto tc_step = (path.newt(x, z) <<= (path.delta(x, y), edge.full(y, z))) | plan(y, x, z);

using TransitiveBase = decltype(non_iterative(tc_base));
using TransitiveFixpoint = decltype(fixpoint(tc_step));

// -----------------------------------------------------------------------------
// 2. GPU Query Executor
// -----------------------------------------------------------------------------

using Executor = SRDatalog::GPU::GPUQueryExecutor<TransitiveFixpoint>;

// -----------------------------------------------------------------------------
// 3. Execution
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <edge_file>" << std::endl;
    return 1;
  }

  std::string file_path = argv[1];
  if (!std::filesystem::exists(file_path)) {
    std::filesystem::path project_root = find_project_root();
    file_path = (project_root / "misc" / file_path).string();
  }

  if (!std::filesystem::exists(file_path)) {
    std::cerr << "File not found: " << file_path << std::endl;
    return 1;
  }

  SRDatalog::GPU::init_cuda();

  // 1. Initialize Host Data
  SemiNaiveDatabase<TCBlueprint> host_db;

  // Load Edge(Full)
  try {
    load_from_file<EdgeSchema>(host_db, file_path);
  } catch (const std::exception& e) {
    std::cerr << "Error loading file: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Loaded " << get_relation_by_schema<EdgeSchema, FULL_VER>(host_db).size()
            << " edges." << std::endl;

  // Execute Base Rule (Non-Iterative) on CPU to prime Path(x,y) :- Edge(x,y)
  execute_query<TransitiveBase>(host_db);

  // Init Path(Delta) with Path(Full) for the first iteration of Fixpoint
  get_relation_by_schema<PathSchema, DELTA_VER>(host_db).concat(
      get_relation_by_schema<PathSchema, FULL_VER>(host_db));

  // 2. Prepare (H2D)
  auto device_db = Executor::prepare(host_db);

  std::cout << "Device DB Initialized." << std::endl;

  // 3. Execute Kernel
  std::cout << "Executing Fixpoint..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  using Plan = typename Executor::Plan;

  constexpr size_t MAX_ITERS = 1000;
  Executor::execute_kernel(device_db, MAX_ITERS);
  using eplan_ = typename Executor::Plan;
  auto full_size =
      get_relation_by_schema<PathSchema, FULL_VER>(device_db).get_index({{1, 0}}).root().degree();
  std::cout << "Full size: " << full_size << std::endl;

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms."
            << std::endl;

  return 0;
}
