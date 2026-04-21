#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include "type_name.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gpu/init.h"
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include "gpu/runtime/query.h"
#include "mir_printer.h"

namespace sd = SRDatalog;
namespace ast = sd::AST;

using ast::database;
using ast::fixpoint;
using ast::if_;
using ast::non_iterative;
using ast::rel;
using ast::relation;
using ast::SemiNaiveDatabase;
using ast::operator""_v;
using ast::Literals::operator""_s;

using sd::get_relation_by_schema;
using sd::load_from_file;
using sd::TestUtil::find_project_root;

using SR = ::BooleanSR;

// -----------------------------------------------------------------------------
// 1. Definition
// -----------------------------------------------------------------------------

using ArcSchema = relation<decltype("Arc"_s), SR, int, int>;
using SGSchema = relation<decltype("SG"_s), SR, int, int>;
using SGBlueprint = database<ArcSchema, SGSchema>;

// Variables
constexpr auto x_ = "x"_v;
constexpr auto y_ = "y"_v;
constexpr auto p_ = "p"_v;
constexpr auto q_ = "q"_v;

// Relation accessors
constexpr auto arc = rel<ArcSchema>;
constexpr auto sg = rel<SGSchema>;

// sg(x, y) :- arc(p, x), arc(p, y)
const auto sg_base_rule =
    (sg.full(x_, y_) <<= (arc.full(p_, x_), arc.full(p_, y_),
                          if_<[](int x, int y) -> bool { return x != y; }>(x_, y_))) |
    plan(p_, x_, y_);

// sg(x, y) :- arc(p, x), sg(p, q), arc(q, y)
const auto sg_fixpoint_rule =
    (sg.newt(x_, y_) <<= (sg.delta(p_, q_), arc.full(p_, x_), arc.full(q_, y_))) |
    plan(p_, q_, x_, y_);

using SGBase = decltype(non_iterative(sg_base_rule));
using SGBaseF = decltype(fixpoint(sg_base_rule));
using SGFixpoint = decltype(fixpoint(sg_fixpoint_rule));

// -----------------------------------------------------------------------------
// 3. Execution
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << (argv[0] ? argv[0] : "sg_device_benchmark") << " <arc_file>"
              << std::endl;
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
  SemiNaiveDatabase<SGBlueprint> host_db;

  // Load Arc(Full)
  try {
    load_from_file<ArcSchema>(host_db, file_path);
  } catch (const std::exception& e) {
    std::cerr << "Error loading file: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Loaded " << get_relation_by_schema<ArcSchema, FULL_VER>(host_db).size() << " arcs."
            << std::endl;

  // 2. Prepare (H2D)
  auto device_db = SRDatalog::GPU::copy_host_to_device(host_db);

  // Get canonical index size from device before reflect (unique count)
  auto& sg_full_device = SRDatalog::get_relation_by_schema<SGSchema, FULL_VER>(device_db);

  // Execute Base Rule on GPU to initialize SG
  SRDatalog::GPU::inspect<
      SRDatalog::GPU::GPUQueryExecutor<decltype(fixpoint(sg_base_rule))>::Plan>();
  SRDatalog::GPU::execute_gpu_query<SGBase>(device_db);

  std::cout << "Device DB Initialized." << std::endl;
  std::cout << "Full Index Size: " << sg_full_device.get_index({{0, 1}}).root().degree()
            << std::endl;

  // 3. Execute Kernel
  std::cout << "Executing SG Fixpoint..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  SRDatalog::GPU::execute_gpu_query<SGFixpoint>(device_db);
  SRDatalog::GPU::inspect<SRDatalog::GPU::GPUQueryExecutor<SGFixpoint>::Plan>();

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms."
            << std::endl;

  return 0;
}
