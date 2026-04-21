/**
 * @file test_self_join.cpp
 * @brief Unit test for self-joins (e.g. Triangle on single Edge relation)
 */

#define BOOST_TEST_MODULE SelfJoinTest
#include <boost/test/included/unit_test.hpp>

#include "ast.h"

#include "query.h"
#include "runtime.h"
#include "semiring.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

BOOST_AUTO_TEST_CASE(test_triangle_self_join) {
  std::cout << "=== Starting test_triangle_self_join ===" << std::endl;
  using SR = BooleanSR;
  using Arena = boost::container::pmr::monotonic_buffer_resource;
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;

  using EdgeSchema = AST::RelationSchema<decltype("Edge"_s), SR, std::tuple<int, int>>;
  using TriangleResult = AST::RelationSchema<decltype("TriRes"_s), SR, std::tuple<int, int, int>>;
  using DBBlueprint = AST::Database<EdgeSchema, TriangleResult>;

  using x_ = AST::Var<decltype("x"_s)>;
  using y_ = AST::Var<decltype("y"_s)>;
  using z_ = AST::Var<decltype("z"_s)>;

  // Triangle: TriRes(x, y, z) :- Edge(x, y), Edge(y, z), Edge(z, x)
  using TriangleRule = AST::Rule<std::tuple<AST::Clause<TriangleResult, NEW_VER, x_, y_, z_>>,
                                 std::tuple<AST::Clause<EdgeSchema, FULL_VER, x_, y_>,
                                            AST::Clause<EdgeSchema, FULL_VER, y_, z_>,
                                            AST::Clause<EdgeSchema, FULL_VER, z_, x_>>,
                                 AST::JoinPlan<std::tuple<x_, y_, z_>>>;

  SemiNaiveDatabase<DBBlueprint> db(&arena_full, &arena_delta, &arena_newt);
  using rule_set = AST::NonIterativeRuleSets<TriangleRule>;
  using executor = ExecuteRulesForNonIterativeRuleSets<rule_set>;

  // Create a triangle: 1->2->3->1
  add_fact<EdgeSchema>(db, BooleanSR::one(), 1, 2);
  add_fact<EdgeSchema>(db, BooleanSR::one(), 2, 3);
  add_fact<EdgeSchema>(db, BooleanSR::one(), 3, 1);
  // Add some noise
  add_fact<EdgeSchema>(db, BooleanSR::one(), 1, 4);

  executor{}(db);

  auto& res = get_relation_by_schema<TriangleResult, FULL_VER>(db);
  std::cout << "Result size: " << res.size() << std::endl;

  // Should find (1, 2, 3), (2, 3, 1), and (3, 1, 2) because it captures directed 3-cycles
  std::array<std::size_t, 3> t1 = {1, 2, 3};
  std::array<std::size_t, 3> t2 = {2, 3, 1};
  std::array<std::size_t, 3> t3 = {3, 1, 2};

  bool f1 = is_in_relation<TriangleResult, FULL_VER>(db, t1);
  bool f2 = is_in_relation<TriangleResult, FULL_VER>(db, t2);
  bool f3 = is_in_relation<TriangleResult, FULL_VER>(db, t3);

  BOOST_CHECK_EQUAL(res.size(), 3);
  BOOST_CHECK(f1);
  BOOST_CHECK(f2);
  BOOST_CHECK(f3);
}
