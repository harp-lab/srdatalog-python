// Unit test for transitive closure computation on GPU
#define BOOST_TEST_MODULE gpu_query_test
#include "ast.h"
#include "query.h"  // For get_relation_by_schema
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <set>

// GPU includes
#include "gpu/init.h"
#include "gpu/runtime/query.h"

using SRDatalog::AST::Literals::operator""_s;

namespace {
using SRDatalog::execute_query;
using SRDatalog::get_relation_by_schema;
using SRDatalog::load_from_file;
using SRDatalog::GPU::execute_gpu_query;
using SRDatalog::GPU::init_cuda;
using SRDatalog::TestUtil::find_project_root;

namespace AST = SRDatalog::AST;
using AST::SemiNaiveDatabase;
using SR = BooleanSR;
using Arena = boost::container::pmr::monotonic_buffer_resource;

// Define schemas
using EdgeSchema = AST::RelationSchema<decltype("Edge"_s), SR, std::tuple<int, int>>;
using PathSchema = AST::RelationSchema<decltype("Path"_s), SR, std::tuple<int, int>>;
using TCBlueprint = AST::Database<EdgeSchema, PathSchema>;

// Define variables
using x_ = AST::Var<decltype("x"_s)>;
using y_ = AST::Var<decltype("y"_s)>;
using z_ = AST::Var<decltype("z"_s)>;

// Base rule: Path(x, y) :- Edge(x, y)
// Uses FULL_VER for body - compiler handles semi-naive transformation
using BaseRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, y_>>,
                           std::tuple<AST::Clause<EdgeSchema, FULL_VER, x_, y_>>,
                           AST::JoinPlan<std::tuple<x_, y_>>>;

// Transitive rule: Path(x, z) :- Path(x, y), Edge(y, z)
// Uses FULL_VER for body - compiler handles semi-naive transformation
using TransitiveRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, z_>>,
                                 std::tuple<AST::Clause<PathSchema, FULL_VER, x_, y_>,
                                            AST::Clause<EdgeSchema, FULL_VER, y_, z_>>,
                                 AST::JoinPlan<std::tuple<y_, z_, x_>>>;

// Single Fixpoint with both rules - matches Nim pattern
using TCFixpoint = AST::Fixpoint<BaseRule, TransitiveRule>;

// Helper to collect all path pairs from a relation
std::set<std::pair<int, int>> collect_paths(const auto& path_rel) {
  std::set<std::pair<int, int>> paths;
  for (std::size_t i = 0; i < path_rel.size(); ++i) {
    paths.insert({path_rel.template column<0>()[i], path_rel.template column<1>()[i]});
  }
  return paths;
}

BOOST_AUTO_TEST_SUITE(gpu_query_test_suite)

BOOST_AUTO_TEST_CASE(test_gpu_tc_basic) {
  init_cuda();

  // Test basic transitive closure computation
  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  // Load test data
  // Create a simple graph: 1->2->3, 2->4
  SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 1, 2);
  SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 2, 3);
  SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 2, 4);

  auto& edge_full = get_relation_by_schema<EdgeSchema, FULL_VER>(db);
  BOOST_REQUIRE_GT(edge_full.size(), 0);

  // Execute TC fixpoint on GPU - single call handles everything
  SRDatalog::GPU::execute_gpu_query<TCFixpoint>(db);

  // Verify transitive closure properties
  auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
  auto paths = collect_paths(path_full);

  std::cout << "Path relation content (" << path_full.size() << " rows):" << std::endl;
  for (std::size_t i = 0; i < path_full.size(); ++i) {
    std::cout << "(" << path_full.template column<0>()[i] << ", "
              << path_full.template column<1>()[i] << ")" << std::endl;
  }

  // Direct edges
  BOOST_CHECK(paths.count({1, 2}));
  BOOST_CHECK(paths.count({2, 3}));
  BOOST_CHECK(paths.count({2, 4}));

  // Transitive edges
  BOOST_CHECK(paths.count({1, 3}));  // 1->2, 2->3
  BOOST_CHECK(paths.count({1, 4}));  // 1->2, 2->4
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
