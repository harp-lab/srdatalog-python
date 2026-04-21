// Integration test: Transitive Closure with Device2LevelIndex on GPU
#define BOOST_TEST_MODULE device_2level_index_tc_test
#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <set>

// GPU includes
#include "gpu/init.h"
#include "gpu/runtime/query.h"

using SRDatalog::AST::Literals::operator""_s;

namespace {
using SRDatalog::get_relation_by_schema;
using SRDatalog::GPU::execute_gpu_query;
using SRDatalog::GPU::init_cuda;

namespace AST = SRDatalog::AST;
using AST::SemiNaiveDatabase;
using SR = BooleanSR;
using Arena = boost::container::pmr::monotonic_buffer_resource;

// ── Schemas with Device2LevelIndex ──────────────────────────────────────

// Edge uses default index (HashTrieIndex → DeviceSortedArrayIndex on GPU)
using EdgeSchema = AST::RelationSchema<decltype("Edge"_s), SR, std::tuple<int, int>>;

// Path uses Device2LevelIndex — this is the index under test
using PathSchema = AST::RelationSchema<decltype("Path"_s), SR, std::tuple<int, int>,
                                       SRDatalog::GPU::Device2LevelIndex>;

using TCBlueprint = AST::Database<EdgeSchema, PathSchema>;

// ── Variables ───────────────────────────────────────────────────────────

using x_ = AST::Var<decltype("x"_s)>;
using y_ = AST::Var<decltype("y"_s)>;
using z_ = AST::Var<decltype("z"_s)>;

// ── Rules ───────────────────────────────────────────────────────────────

// Base rule: Path(x, y) :- Edge(x, y)
using BaseRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, y_>>,
                           std::tuple<AST::Clause<EdgeSchema, FULL_VER, x_, y_>>,
                           AST::JoinPlan<std::tuple<x_, y_>>>;

// Transitive rule: Path(x, z) :- Path(x, y), Edge(y, z)
using TransitiveRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, z_>>,
                                 std::tuple<AST::Clause<PathSchema, FULL_VER, x_, y_>,
                                            AST::Clause<EdgeSchema, FULL_VER, y_, z_>>,
                                 AST::JoinPlan<std::tuple<y_, z_, x_>>>;

using TCFixpoint = AST::Fixpoint<BaseRule, TransitiveRule>;

// ── Reference schemas with default index (for comparison) ───────────────

using PathSchemaRef = AST::RelationSchema<decltype("PathRef"_s), SR, std::tuple<int, int>>;
using TCBlueprintRef = AST::Database<EdgeSchema, PathSchemaRef>;

using BaseRuleRef = AST::Rule<std::tuple<AST::Clause<PathSchemaRef, NEW_VER, x_, y_>>,
                              std::tuple<AST::Clause<EdgeSchema, FULL_VER, x_, y_>>,
                              AST::JoinPlan<std::tuple<x_, y_>>>;

using TransitiveRuleRef = AST::Rule<std::tuple<AST::Clause<PathSchemaRef, NEW_VER, x_, z_>>,
                                    std::tuple<AST::Clause<PathSchemaRef, FULL_VER, x_, y_>,
                                               AST::Clause<EdgeSchema, FULL_VER, y_, z_>>,
                                    AST::JoinPlan<std::tuple<y_, z_, x_>>>;

using TCFixpointRef = AST::Fixpoint<BaseRuleRef, TransitiveRuleRef>;

// ── Helpers ─────────────────────────────────────────────────────────────

std::set<std::pair<int, int>> collect_paths(const auto& rel) {
  std::set<std::pair<int, int>> paths;
  for (std::size_t i = 0; i < rel.size(); ++i) {
    paths.insert({rel.template column<0>()[i], rel.template column<1>()[i]});
  }
  return paths;
}

BOOST_AUTO_TEST_SUITE(device_2level_tc_suite)

// ── Test 1: Basic TC with small graph ───────────────────────────────────

BOOST_AUTO_TEST_CASE(test_2level_tc_basic) {
  init_cuda();

  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  // Simple graph: 1→2→3, 2→4
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 1, 2);
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 2, 3);
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 2, 4);

  execute_gpu_query<TCFixpoint>(db);

  auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
  auto paths = collect_paths(path_full);

  std::cout << "2-Level TC paths (" << paths.size() << "):" << std::endl;
  for (auto& [a, b] : paths) {
    std::cout << "  (" << a << ", " << b << ")" << std::endl;
  }

  // Direct edges
  BOOST_CHECK(paths.count({1, 2}));
  BOOST_CHECK(paths.count({2, 3}));
  BOOST_CHECK(paths.count({2, 4}));

  // Transitive edges
  BOOST_CHECK(paths.count({1, 3}));  // 1→2→3
  BOOST_CHECK(paths.count({1, 4}));  // 1→2→4

  BOOST_CHECK_EQUAL(paths.size(), 5);
}

// ── Test 2: Longer chain ────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(test_2level_tc_chain) {
  init_cuda();

  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  // Linear chain: 1→2→3→4→5
  for (int i = 1; i <= 4; ++i) {
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), i, i + 1);
  }

  execute_gpu_query<TCFixpoint>(db);

  auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
  auto paths = collect_paths(path_full);

  // Expected: 4 direct + 3 + 2 + 1 = 10 transitive paths
  // (1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)
  BOOST_CHECK_EQUAL(paths.size(), 10);

  // Check all transitive paths exist
  for (int i = 1; i <= 4; ++i) {
    for (int j = i + 1; j <= 5; ++j) {
      BOOST_CHECK_MESSAGE(paths.count({i, j}), "Missing path (" << i << ", " << j << ")");
    }
  }
}

// ── Test 3: Compare with default DSAI index ─────────────────────────────

BOOST_AUTO_TEST_CASE(test_2level_vs_dsai) {
  init_cuda();

  // Build a more complex graph
  auto build_edges = [](auto& db) {
    // Diamond + chain: 1→2, 1→3, 2→4, 3→4, 4→5→6
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 1, 2);
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 1, 3);
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 2, 4);
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 3, 4);
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 4, 5);
    SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 5, 6);
  };

  // Run with Device2LevelIndex
  Arena arena1;
  SemiNaiveDatabase<TCBlueprint> db1(&arena1, &arena1, &arena1);
  build_edges(db1);
  execute_gpu_query<TCFixpoint>(db1);
  auto paths_2level = collect_paths(get_relation_by_schema<PathSchema, FULL_VER>(db1));

  // Run with default DeviceSortedArrayIndex
  Arena arena2;
  SemiNaiveDatabase<TCBlueprintRef> db2(&arena2, &arena2, &arena2);
  build_edges(db2);
  execute_gpu_query<TCFixpointRef>(db2);
  auto paths_dsai = collect_paths(get_relation_by_schema<PathSchemaRef, FULL_VER>(db2));

  std::cout << "2-Level paths: " << paths_2level.size() << ", DSAI paths: " << paths_dsai.size()
            << std::endl;

  // Both should produce identical results
  BOOST_CHECK_EQUAL(paths_2level.size(), paths_dsai.size());
  BOOST_CHECK(paths_2level == paths_dsai);
}

// ── Test 4: Cycle handling ──────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(test_2level_tc_cycle) {
  init_cuda();

  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  // Cycle: 1→2→3→1
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 1, 2);
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 2, 3);
  SRDatalog::add_fact<EdgeSchema>(db, SR::one(), 3, 1);

  execute_gpu_query<TCFixpoint>(db);

  auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
  auto paths = collect_paths(path_full);

  // In a 3-cycle, every node reaches every other node (including self)
  // Expected: 3*3 = 9 paths
  BOOST_CHECK_EQUAL(paths.size(), 9);

  for (int i = 1; i <= 3; ++i) {
    for (int j = 1; j <= 3; ++j) {
      BOOST_CHECK_MESSAGE(paths.count({i, j}), "Missing path (" << i << ", " << j << ")");
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
