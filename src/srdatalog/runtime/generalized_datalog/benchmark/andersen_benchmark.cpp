// Benchmark using google's microbenchmark library
#include "andersen_def.h"
#include "query.h"
#include "runtime.h"
#include "test_util.h"
#include <benchmark/benchmark.h>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace SRDatalog;
using namespace SRDatalog::AST;

using sd::execute_query;
using sd::get_relation_by_schema;
using sd::load_from_file;
using sd::TestUtil::find_project_root;
using Arena = boost::container::pmr::monotonic_buffer_resource;

// List of dataset directories to benchmark (relative to misc/ directory)
static const std::vector<std::string> dataset_dirs = {
    // "andersen_medium",
    // "andersen_large",
    "other/FlowLog-VLDB-Reproduction/dataset/medium",
    // "other/FlowLog-VLDB-Reproduction/dataset/large",
};

// Benchmark Andersen's Points-To Analysis computation from CSV files
static void BM_AndersenPointsTo(benchmark::State& state) {
  // Get dataset index from benchmark argument
  int dataset_index = state.range(0);

  if (dataset_index < 0 || dataset_index >= static_cast<int>(dataset_dirs.size())) {
    state.SkipWithError("Invalid dataset index");
    return;
  }

  std::string dataset_dir = dataset_dirs[dataset_index];

  // Find project root and construct full paths
  std::filesystem::path project_root = find_project_root();
  std::filesystem::path dataset_path = project_root / "misc" / dataset_dir;

  std::filesystem::path addressof_file = dataset_path / "addressOf.csv";
  std::filesystem::path assign_file = dataset_path / "assign.csv";
  std::filesystem::path load_file = dataset_path / "load.csv";
  std::filesystem::path store_file = dataset_path / "store.csv";

  // Check if files exist
  if (!std::filesystem::exists(addressof_file) || !std::filesystem::exists(assign_file) ||
      !std::filesystem::exists(load_file) || !std::filesystem::exists(store_file)) {
    std::cerr << "Warning: Dataset files not found in: " << dataset_path << std::endl;
    state.SkipWithError("Dataset files not found");
    return;
  }

  for (auto _ : state) {
    // Create fresh arenas for each iteration
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;

    SemiNaiveDatabase<AndersenBlueprint> db(&arena_full, &arena_delta, &arena_newt);

    // Load input relations from CSV files
    try {
      load_from_file<AddressOfSchema>(db, addressof_file.string());
      load_from_file<AssignSchema>(db, assign_file.string());
      load_from_file<LoadSchema>(db, load_file.string());
      load_from_file<StoreSchema>(db, store_file.string());
    } catch (const std::exception& e) {
      state.SkipWithError(e.what());
      return;
    }

    // Execute Andersen's analysis
    execute_query<AndersenBase>(db);
    execute_query<AndersenFixpoint>(db);

    // Get result size to ensure computation is not optimized away
    auto& pointsto_full = get_relation_by_schema<PointsToSchema, FULL_VER>(db);
    auto& addressof_full = get_relation_by_schema<AddressOfSchema, FULL_VER>(db);
    auto& assign_full = get_relation_by_schema<AssignSchema, FULL_VER>(db);
    auto& load_full = get_relation_by_schema<LoadSchema, FULL_VER>(db);
    auto& store_full = get_relation_by_schema<StoreSchema, FULL_VER>(db);

    benchmark::DoNotOptimize(pointsto_full.size());

    // Set counters
    state.counters["input_addressof"] = addressof_full.size();
    state.counters["input_assign"] = assign_full.size();
    state.counters["input_load"] = load_full.size();
    state.counters["input_store"] = store_full.size();
    state.counters["output_pointsto"] = pointsto_full.size();
  }

  // Set dataset name in benchmark label
  state.SetLabel(dataset_dir);
}

// Register benchmarks
void RegisterDatasetBenchmarks() {
  for (size_t i = 0; i < dataset_dirs.size(); ++i) {
    std::string benchmark_name = "BM_AndersenPointsTo/" + dataset_dirs[i];
    benchmark::RegisterBenchmark(benchmark_name.c_str(), BM_AndersenPointsTo)
        ->Arg(static_cast<int>(i))
        ->Unit(benchmark::kMillisecond);
  }
}

int main(int argc, char** argv) {
  RegisterDatasetBenchmarks();
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
