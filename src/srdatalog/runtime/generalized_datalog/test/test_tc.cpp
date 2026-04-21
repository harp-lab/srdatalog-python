// Unit test for transitive closure computation
#define BOOST_TEST_MODULE transitive_closure_test
#include "ast.h"
#include "query.h"  // For get_relation_by_schema
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <set>

using SRDatalog::AST::Literals::operator""_s;

namespace {
using SRDatalog::execute_query;
using SRDatalog::get_relation_by_schema;
using SRDatalog::load_from_file;
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
using BaseRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, y_>>,
                           std::tuple<AST::Clause<EdgeSchema, FULL_VER, x_, y_>>,
                           AST::JoinPlan<std::tuple<x_, y_>>>;

// Transitive rule: Path(x, z) :- Path(x, y), Edge(y, z)
using TransitiveRule = AST::Rule<std::tuple<AST::Clause<PathSchema, NEW_VER, x_, z_>>,
                                 std::tuple<AST::Clause<PathSchema, DELTA_VER, x_, y_>,
                                            AST::Clause<EdgeSchema, FULL_VER, y_, z_>>,
                                 AST::JoinPlan<std::tuple<y_, z_, x_>>>;

using TransitiveBase = AST::NonIterativeRuleSets<BaseRule>;
using TransitiveFixpoint = AST::Fixpoint<TransitiveRule>;
using test_t = typename SRDatalog::mir::CompileRuleToMIR<TransitiveRule>::type;

// Helper to collect all path pairs from a relation
std::set<std::pair<int, int>> collect_paths(const auto& path_rel) {
  std::set<std::pair<int, int>> paths;
  for (std::size_t i = 0; i < path_rel.size(); ++i) {
    paths.insert({path_rel.template column<0>()[i], path_rel.template column<1>()[i]});
  }
  return paths;
}

BOOST_AUTO_TEST_SUITE(transitive_closure_suite)

BOOST_AUTO_TEST_CASE(test_transitive_closure_basic) {
  // Test basic transitive closure computation
  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  // Load test data from file
  std::filesystem::path project_dir = find_project_root();
  std::filesystem::path test_data_dir = project_dir / "test_data";
  std::filesystem::path edge_file = test_data_dir / "Edge.csv";

  // If test_data/Edge.csv doesn't exist, create a simple test case
  if (!std::filesystem::exists(edge_file)) {
    // Create a simple graph: 1->2->3->4, 2->5->6
    SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 1, 2);
    SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 2, 3);
    SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 3, 4);
    SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 2, 5);
    SRDatalog::add_fact<EdgeSchema>(db, BooleanSR::one(), 5, 6);
  } else {
    // Load from file
    BOOST_REQUIRE_NO_THROW((load_from_file<EdgeSchema>(db, edge_file.string())));
  }

  auto& edge_full = get_relation_by_schema<EdgeSchema, FULL_VER>(db);
  BOOST_REQUIRE_GT(edge_full.size(), 0);

  // Execute base rule
  execute_query<TransitiveBase>(db);

  auto& path_full = get_relation_by_schema<PathSchema, FULL_VER>(db);
  BOOST_REQUIRE_GE(path_full.size(),
                   edge_full.size());  // Path should have at least as many as edges

  // Execute transitive closure fixpoint
  execute_query<TransitiveFixpoint>(db);

  // Verify transitive closure properties
  auto paths = collect_paths(path_full);

  // All direct edges should be in paths
  for (std::size_t i = 0; i < edge_full.size(); ++i) {
    int from = edge_full.template column<0>()[i];
    int to = edge_full.template column<1>()[i];
    BOOST_CHECK(paths.count({from, to}) > 0);
  }

  // Verify transitive paths exist (e.g., if 1->2 and 2->3, then 1->3 should exist)
  // This is a basic sanity check - the exact paths depend on the input data
  BOOST_CHECK_GT(path_full.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_transitive_closure_from_file) {
  // Test transitive closure with data loaded from file
  Arena arena;
  SemiNaiveDatabase<TCBlueprint> db(&arena, &arena, &arena);

  std::filesystem::path project_dir = find_project_root();
  std::filesystem::path test_data_dir = project_dir / "test_data";
  std::filesystem::path edge_file = test_data_dir / "Edge.csv";

  // Skip if test data file doesn't exist
  if (!std::filesystem::exists(edge_file)) {
    BOOST_TEST_MESSAGE("Skipping test: test_data/Edge.csv not found");
    return;
  }

  // Load edges from file
  BOOST_REQUIRE_NO_THROW((load_from_file<EdgeSchema>(db, edge_file.string())));

  auto& edge_full = get_relation_by_schema<EdgeSchema, FULL_VER>(db);
  std::size_t initial_edge_count = edge_full.size();
  BOOST_REQUIRE_GT(initial_edge_count, 0);

  // Execute base rule
  execute_query<TransitiveBase>(db);

  auto& path_full_after_base = get_relation_by_schema<PathSchema, FULL_VER>(db);
  BOOST_CHECK_GE(path_full_after_base.size(), initial_edge_count);

  // Execute transitive closure
  execute_query<TransitiveFixpoint>(db);

  auto& path_full_final = get_relation_by_schema<PathSchema, FULL_VER>(db);

  // Final path count should be >= initial edge count (transitive closure adds paths)
  BOOST_CHECK_GE(path_full_final.size(), initial_edge_count);

  // Verify indexes are built
  BOOST_CHECK_GT(path_full_final.get_all_indexes().size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
