// Benchmark using google's microbenchmark library
// Reference : https:  // github.com/google/benchmark/blob/main/docs/user_guide.md

#include "ast.h"
#include "query.h"  // For get_relation_by_schema
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <benchmark/benchmark.h>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
using Arena = boost::container::pmr::monotonic_buffer_resource;

// Define schemas for Transitive Closure
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
auto tc_base = path.newt(x, y) <<= edge.full(x, y);

// Transitive rule: Path(x, z) :- Path(x, y), Edge(y, z)
auto tc_step = path.newt(x, z) <<= (path.delta(x, y), edge.full(y, z));

using TransitiveBase = decltype(non_iterative(tc_base));
using TransitiveFixpoint = decltype(fixpoint(tc_step));

// List of graph files to benchmark (relative to misc/ directory)
// You can modify this list to add/remove graph files
static const std::vector<std::string> graph_files = {
    "edges.txt", "edges_1.txt",
    // Add more graph files here as needed
    // "soc-Epinions1.txt",
    // "edges_w.txt",
};

// Benchmark Transitive Closure computation from file
static void BM_TransitiveClosure(benchmark::State& state) {
  // Get file index from benchmark argument
  int file_index = state.range(0);

  if (file_index < 0 || file_index >= static_cast<int>(graph_files.size())) {
    state.SkipWithError("Invalid file index");
    return;
  }

  std::string file_name = graph_files[file_index];

  // Find project root and construct full path
  std::filesystem::path project_root = find_project_root();
  std::filesystem::path file_path = project_root / "misc" / file_name;

  // Check if file exists
  if (!std::filesystem::exists(file_path)) {
    std::cerr << "Warning: Graph file not found: " << file_path << std::endl;
    std::cerr << "Skipping benchmark for this file." << std::endl;
    state.SkipWithError("File not found");
    return;
  }

  std::string file_path_string = file_path.string();

  for (auto _ : state) {
    // Create fresh arenas for each iteration
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;

    SemiNaiveDatabase<TCBlueprint> db(&arena_full, &arena_delta, &arena_newt);

    // Load edges from file (space-separated format: "u v")
    try {
      load_from_file<EdgeSchema>(db, file_path_string);
    } catch (const std::exception& e) {
      state.SkipWithError(e.what());
      return;
    }

    // Execute transitive closure
    execute_query<TransitiveBase>(db);
    execute_query<TransitiveFixpoint>(db);

    // Get result size to ensure computation is not optimized away
    auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
    auto& edge_full = get_relation_by_schema<EdgeSchema, FULL_VER>(db);
    benchmark::DoNotOptimize(path_full.size());

    // Set counters
    state.counters["input_edges"] = edge_full.size();
    state.counters["output_paths"] = path_full.size();
  }

  // Set file name in benchmark label
  state.SetLabel(file_name);
}

// Register benchmarks for each graph file
// Each file gets its own benchmark instance
void RegisterGraphFileBenchmarks() {
  for (size_t i = 0; i < graph_files.size(); ++i) {
    std::string benchmark_name = "BM_TransitiveClosure/" + graph_files[i];
    benchmark::RegisterBenchmark(benchmark_name.c_str(), BM_TransitiveClosure)
        ->Arg(static_cast<int>(i))
        ->Unit(benchmark::kMillisecond);
  }
}

// Custom main to register file-based benchmarks
int main(int argc, char** argv) {
  // Register benchmarks for graph files
  RegisterGraphFileBenchmarks();

  // Run benchmarks
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
