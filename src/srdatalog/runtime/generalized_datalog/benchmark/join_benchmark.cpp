
#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <benchmark/benchmark.h>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// GPU Support
#ifdef SR_ENABLE_GPU
#include "gpu/init.h"
#include "gpu/runtime/query.h"
#endif

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

// ============================================================================
// Schemas
// ============================================================================
using EdgeSchema = relation<decltype("Edge"_s), SR, int, int>;
using R3Schema = relation<decltype("R3"_s), SR, int, int, int>;

// Output schemas (arity varies)
using TriangleResult = relation<decltype("TriRes"_s), SR, int, int, int>;
using Chain3Result = relation<decltype("Chain3Res"_s), SR, int, int, int, int>;
using StarResult = relation<decltype("StarRes"_s), SR, int, int, int, int>;

// BenchmarkDB removed from here to allow redefinition later or I should have just updated it here.
// I will remove it here.

// Variables
using x_ = ast::Var<decltype("x"_s)>;
using y_ = ast::Var<decltype("y"_s)>;
using z_ = ast::Var<decltype("z"_s)>;
using w_ = ast::Var<decltype("w"_s)>;
using a_ = ast::Var<decltype("a"_s)>;

// ============================================================================
// Queries
// ============================================================================

// 1. Triangle: TriRes(x, y, z) :- Edge(x, y), Edge(y, z), Edge(z, x)
// Join Plan: Edge(x, y) -> Edge(y, z) -> Edge(z, x)
// Variables: x, y, z. Order: x, y, z?
// Optimal order often starts with smallest relation or most selective. Here all same.
// (x, y, z) -> Edge(x, y) ok, Edge(y, z) ok, Edge(z, x) ok.
using TriangleRule = ast::Rule<
    std::tuple<ast::Clause<TriangleResult, NEW_VER, x_, y_, z_>>,
    std::tuple<ast::Clause<EdgeSchema, FULL_VER, x_, y_>, ast::Clause<EdgeSchema, FULL_VER, y_, z_>,
               ast::Clause<EdgeSchema, FULL_VER, z_, x_>>,
    ast::JoinPlan<std::tuple<x_, y_, z_>>>;
using TriangleQuery = ast::NonIterativeRuleSets<TriangleRule>;

// 2. Star (3 branches): StarRes(x, y, z, w) :- Edge(x, y), Edge(x, z), Edge(x, w)
using StarRule = ast::Rule<
    std::tuple<ast::Clause<StarResult, NEW_VER, x_, y_, z_, w_>>,
    std::tuple<ast::Clause<EdgeSchema, FULL_VER, x_, y_>, ast::Clause<EdgeSchema, FULL_VER, x_, z_>,
               ast::Clause<EdgeSchema, FULL_VER, x_, w_>>,
    ast::JoinPlan<std::tuple<x_, y_, z_, w_>>>;
using StarQuery = ast::NonIterativeRuleSets<StarRule>;

// 3. 3-Arity Chain: Chain3Res(x, y, z, w) :- R3(x, y, z), R3(y, z, w)
// Join on (y, z)
using Chain3Rule = ast::Rule<std::tuple<ast::Clause<Chain3Result, NEW_VER, x_, y_, z_, w_>>,
                             std::tuple<ast::Clause<R3Schema, FULL_VER, x_, y_, z_>,
                                        ast::Clause<R3Schema, FULL_VER, y_, z_, w_>>,
                             ast::JoinPlan<std::tuple<y_, z_, x_, w_>>>;
using Chain3Query = ast::NonIterativeRuleSets<Chain3Rule>;

// Copy Query: Result(x, y) :- Edge(x, y)
using CopyResult = relation<decltype("CopyRes"_s), SR, int, int>;
using CopyRule = ast::Rule<std::tuple<ast::Clause<CopyResult, NEW_VER, x_, y_>>,
                           std::tuple<ast::Clause<EdgeSchema, FULL_VER, x_, y_>>,
                           ast::JoinPlan<std::tuple<x_, y_>>>;
using CopyQuery = ast::NonIterativeRuleSets<CopyRule>;

using BenchmarkDB =
    database<EdgeSchema, R3Schema, TriangleResult, Chain3Result, StarResult, CopyResult>;

// ============================================================================
// Benchmarks
// ============================================================================

std::string get_file_path(const std::string& filename) {
  std::filesystem::path project_root = find_project_root();
  return (project_root / "misc" / filename).string();
}

static void BM_Triangle_CPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<EdgeSchema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> db(&arena_full, &arena_delta, &arena_newt);
    // Copy EDB from source
    get_relation_by_schema<EdgeSchema, FULL_VER>(db).concat(
        get_relation_by_schema<EdgeSchema, FULL_VER>(source_db));
    state.ResumeTiming();

    execute_query<TriangleQuery>(db);

    auto& res = get_relation_by_schema<TriangleResult, FULL_VER>(db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}

static void BM_Star_CPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<EdgeSchema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> db(&arena_full, &arena_delta, &arena_newt);
    // Copy EDB from source
    get_relation_by_schema<EdgeSchema, FULL_VER>(db).concat(
        get_relation_by_schema<EdgeSchema, FULL_VER>(source_db));
    state.ResumeTiming();

    execute_query<StarQuery>(db);

    auto& res = get_relation_by_schema<StarResult, FULL_VER>(db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}

static void BM_Chain3_CPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_3arity_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<R3Schema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> db(&arena_full, &arena_delta, &arena_newt);
    // Copy EDB from source
    auto& rel = get_relation_by_schema<R3Schema, FULL_VER>(db);
    rel.concat(get_relation_by_schema<R3Schema, FULL_VER>(source_db));
    rel.ensure_index({{0, 1, 2}});
    state.ResumeTiming();

    execute_query<Chain3Query>(db);

    auto& res = get_relation_by_schema<Chain3Result, FULL_VER>(db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}

static void BM_Debug_Small(benchmark::State& state) {
  std::string file_path = get_file_path("debug_edges.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  // state.PauseTiming();
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;
  SemiNaiveDatabase<BenchmarkDB> db(&arena_full, &arena_delta, &arena_newt);
  load_from_file<EdgeSchema>(db, file_path);
  // state.ResumeTiming();

  for (auto _ : state) {
    state.PauseTiming();
    // Clear only result relations (keep Edge intact) to reset state between iterations
    get_relation_by_schema<TriangleResult, NEW_VER>(db).clear();
    get_relation_by_schema<TriangleResult, FULL_VER>(db).clear();
    get_relation_by_schema<StarResult, NEW_VER>(db).clear();
    get_relation_by_schema<StarResult, FULL_VER>(db).clear();
    get_relation_by_schema<CopyResult, NEW_VER>(db).clear();
    get_relation_by_schema<CopyResult, FULL_VER>(db).clear();
    state.ResumeTiming();

    // Only time the join operations
    execute_query<TriangleQuery>(db);
    auto& tri = get_relation_by_schema<TriangleResult, FULL_VER>(db);
    state.counters["Triangle"] = tri.size();

    execute_query<StarQuery>(db);
    auto& star = get_relation_by_schema<StarResult, FULL_VER>(db);
    state.counters["Star"] = star.size();

    execute_query<CopyQuery>(db);
    auto& copy = get_relation_by_schema<CopyResult, FULL_VER>(db);
    state.counters["Copy"] = copy.size();
  }
}

BENCHMARK(BM_Debug_Small)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Triangle_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Star_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Chain3_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();

#ifdef SR_ENABLE_GPU
static void BM_Triangle_GPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }
  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<EdgeSchema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> host_db(&arena_full, &arena_delta, &arena_newt);
    // Copy from source
    get_relation_by_schema<EdgeSchema, FULL_VER>(host_db).concat(
        get_relation_by_schema<EdgeSchema, FULL_VER>(source_db));

    // Prepare GPU data (H2D copy) - excluded from timing
    using Executor = sd::GPU::GPUQueryExecutor<ast::Fixpoint<TriangleRule>>;
    auto bundle = Executor::prepare(host_db);
    // Pre-build index on GPU
    std::get<0>(bundle.full).ensure_index({{0, 1}});
    std::get<0>(bundle.full).ensure_index({{1, 0}});
    state.ResumeTiming();

    // Execute Kernel - included in timing
    Executor::execute_kernel<BenchmarkDB>(bundle);

    // Reflect results back (D2H copy) - excluded from timing
    state.PauseTiming();
    Executor::reflect(host_db, bundle);
    state.ResumeTiming();

    // Verify results by copying back
    auto& res = get_relation_by_schema<TriangleResult, FULL_VER>(host_db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}
BENCHMARK(BM_Triangle_GPU)->Unit(benchmark::kMillisecond)->UseRealTime();

static void BM_Star_GPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<EdgeSchema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> host_db(&arena_full, &arena_delta, &arena_newt);
    get_relation_by_schema<EdgeSchema, FULL_VER>(host_db).concat(
        get_relation_by_schema<EdgeSchema, FULL_VER>(source_db));

    using Executor = sd::GPU::GPUQueryExecutor<ast::Fixpoint<StarRule>>;
    auto bundle = Executor::prepare(host_db);
    // Pre-build index on GPU
    std::get<0>(bundle.full).ensure_index({{0, 1}});
    std::get<0>(bundle.full).ensure_index({{1, 0}});
    state.ResumeTiming();

    Executor::execute_kernel<BenchmarkDB>(bundle);

    state.PauseTiming();
    Executor::reflect(host_db, bundle);
    state.ResumeTiming();

    auto& res = get_relation_by_schema<StarResult, FULL_VER>(host_db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}
BENCHMARK(BM_Star_GPU)->Unit(benchmark::kMillisecond)->UseRealTime();

static void BM_Chain3_GPU(benchmark::State& state) {
  std::string file_path = get_file_path("large_relation_3arity_1M.txt");
  if (!std::filesystem::exists(file_path)) {
    state.SkipWithError("File not found");
    return;
  }

  Arena source_arena_full;
  Arena source_arena_delta;
  Arena source_arena_newt;
  SemiNaiveDatabase<BenchmarkDB> source_db(&source_arena_full, &source_arena_delta,
                                           &source_arena_newt);
  load_from_file<R3Schema>(source_db, file_path);

  for (auto _ : state) {
    state.PauseTiming();
    Arena arena_full;
    Arena arena_delta;
    Arena arena_newt;
    SemiNaiveDatabase<BenchmarkDB> host_db(&arena_full, &arena_delta, &arena_newt);
    get_relation_by_schema<R3Schema, FULL_VER>(host_db).concat(
        get_relation_by_schema<R3Schema, FULL_VER>(source_db));

    using Executor = sd::GPU::GPUQueryExecutor<ast::Fixpoint<Chain3Rule>>;
    auto bundle = Executor::prepare(host_db);
    // Pre-build index on GPU (R3 is second schema, index 1)
    std::get<1>(bundle.full).ensure_index({{0, 1, 2}});
    state.ResumeTiming();

    Executor::execute_kernel<BenchmarkDB>(bundle);

    state.PauseTiming();
    Executor::reflect(host_db, bundle);
    state.ResumeTiming();

    auto& res = get_relation_by_schema<Chain3Result, FULL_VER>(host_db);
    benchmark::DoNotOptimize(res.size());
    state.counters["output"] = res.size();
  }
}
BENCHMARK(BM_Chain3_GPU)->Unit(benchmark::kMillisecond)->UseRealTime();
#endif

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);

  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
