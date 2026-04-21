/**
 * @file test_tmp.cpp
 * @author Yihao Sun (ysun67@syr.edu)
 * @brief test the TMP based query compiler and executor
 * @version 0.1
 * @date 2025-11-04
 */

#define BOOST_TEST_MODULE TMPQueryCompilerExecutor
#include <boost/test/included/unit_test.hpp>

#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

BOOST_AUTO_TEST_CASE(test_tmp_query_compiler_executor_triangle_join) {
  std::cout << "=== Starting test_tmp_query_compiler_executor_triangle_join ===" << std::endl;
  using SR = BooleanSR;
  using Arena = boost::container::pmr::monotonic_buffer_resource;
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;

  using RelationR = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
  using RelationS = AST::RelationSchema<decltype("S"_s), SR, std::tuple<int, int, int>>;
  using RelationT = AST::RelationSchema<decltype("T"_s), SR, std::tuple<int, int, int>>;
  using RelationZ = AST::RelationSchema<decltype("Z"_s), SR, std::tuple<int, int, int>>;
  using TriangleDBBlueprint = AST::Database<RelationR, RelationS, RelationT, RelationZ>;

  using x_ = AST::Var<decltype("x"_s)>;
  using y_ = AST::Var<decltype("y"_s)>;
  using z_ = AST::Var<decltype("z"_s)>;
  using h_ = AST::Var<decltype("h"_s)>;
  using f_ = AST::Var<decltype("f"_s)>;

  using rule1 = AST::Rule<
      std::tuple<AST::Clause<RelationZ, NEW_VER, x_, y_, z_>>,  // Head: Z(x,y,z)
                                                                // with NEW_VER
      std::tuple<  // Body: R(x,y), S(y,z,h), T(z,x,f) with FULL_VER (default)
          AST::Clause<RelationR, FULL_VER, x_, y_>, AST::Clause<RelationS, FULL_VER, y_, z_, h_>,
          AST::Clause<RelationT, FULL_VER, z_, x_, f_>>,
      // join plan, the order of variables intersected in the body of the rule
      AST::JoinPlan<std::tuple<x_, y_, z_>>>;

  SemiNaiveDatabase<TriangleDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);
  using rule_set = AST::NonIterativeRuleSets<rule1>;
  using executor = ExecuteRulesForNonIterativeRuleSets<rule_set>;

  // Add facts to relation R
  add_fact<RelationR>(db, BooleanSR::one(), 1, 10);
  add_fact<RelationR>(db, BooleanSR::one(), 2, 10);
  add_fact<RelationR>(db, BooleanSR::one(), 3, 10);
  add_fact<RelationR>(db, BooleanSR::one(), 4, 10);
  add_fact<RelationR>(db, BooleanSR::one(), 5, 20);
  add_fact<RelationR>(db, BooleanSR::one(), 6, 30);
  add_fact<RelationR>(db, BooleanSR::one(), 7, 40);
  add_fact<RelationR>(db, BooleanSR::one(), 8, 50);

  // Add facts to relation S
  add_fact<RelationS>(db, BooleanSR::one(), 10, 7, 5);
  add_fact<RelationS>(db, BooleanSR::one(), 10, 8, 9);
  add_fact<RelationS>(db, BooleanSR::one(), 10, 9, 4);
  add_fact<RelationS>(db, BooleanSR::one(), 10, 100, 3);
  add_fact<RelationS>(db, BooleanSR::one(), 20, 7, 6);
  add_fact<RelationS>(db, BooleanSR::one(), 30, 6, 12);
  add_fact<RelationS>(db, BooleanSR::one(), 40, 7, 57);
  add_fact<RelationS>(db, BooleanSR::one(), 50, 7, 34);

  // Add facts to relation T
  add_fact<RelationT>(db, BooleanSR::one(), 7, 1, 4);
  add_fact<RelationT>(db, BooleanSR::one(), 8, 2, 3);
  add_fact<RelationT>(db, BooleanSR::one(), 9, 3, 3);
  add_fact<RelationT>(db, BooleanSR::one(), 7, 5, 4);
  add_fact<RelationT>(db, BooleanSR::one(), 6, 6, 2);
  add_fact<RelationT>(db, BooleanSR::one(), 7, 7, 6);
  add_fact<RelationT>(db, BooleanSR::one(), 7, 8, 2);
  add_fact<RelationT>(db, BooleanSR::one(), 123, 4, 1);

  // Expected results after join:
  //   Triangle
  //   ===============
  //   1,10,7
  //   2,10,8
  //   3,10,9
  //   5,20,7
  //   6,30,6
  //   7,40,7
  //   8,50,7
  //   ===============

  // Execute the rule set (this handles index building and execution)
  executor{}(db);
  // For non-iterative rules, data is written to FULL_VER, not NEW_VER
  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  rel_z.ensure_index(rel_z.get_default_index());
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Relation Z size: " << rel_z.size() << std::endl;

  // Verify that the expected tuple (1, 10, 7) is in the result relation
  std::array<std::size_t, 3> tuple1 = {1, 10, 7};
  bool in_relation_1_10_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_REQUIRE(in_relation_1_10_7);
  std::array<std::size_t, 3> tuple2 = {2, 10, 8};
  bool in_relation_2_10_8 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_REQUIRE(in_relation_2_10_8);
  std::array<std::size_t, 3> tuple3 = {3, 10, 9};
  bool in_relation_3_10_9 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_REQUIRE(in_relation_3_10_9);
  // Optionally print the relation for debugging
  // print_relation<RelationZ, NEW_VER>(db, std::cout, 20);
}

// In test_if_clause_modulo_filter::test_method

BOOST_AUTO_TEST_CASE(test_if_clause_modulo_filter) {
  std::cout << "=== Starting test_if_clause_modulo_filter ===" << std::endl;
  using SR = BooleanSR;
  using Arena = boost::container::pmr::monotonic_buffer_resource;
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;

  using RelationR = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
  using RelationS = AST::RelationSchema<decltype("S"_s), SR, std::tuple<int, int, int>>;
  using RelationT = AST::RelationSchema<decltype("T"_s), SR, std::tuple<int, int, int>>;
  using RelationZ = AST::RelationSchema<decltype("Z"_s), SR, std::tuple<int, int, int>>;
  using TriangleDBBlueprint = AST::Database<RelationR, RelationS, RelationT, RelationZ>;

  using x_ = AST::Var<decltype("x"_s)>;
  using y_ = AST::Var<decltype("y"_s)>;
  using z_ = AST::Var<decltype("z"_s)>;
  using h_ = AST::Var<decltype("h"_s)>;
  using f_ = AST::Var<decltype("f"_s)>;

  using rule_if_test = AST::Rule<
      std::tuple<AST::Clause<RelationZ, NEW_VER, x_, y_, z_>>,
      std::tuple<AST::Clause<RelationR, FULL_VER, x_, y_>,
                 AST::Clause<RelationS, FULL_VER, y_, z_, h_>,
                 AST::Clause<RelationT, FULL_VER, z_, x_, f_>,
                 // Define a filter: x % 2 == 0 (only even values of x)
                 AST::IfClause<AST::CppExpr<std::tuple<x_>, [](int x) { return x % 2 == 0; }>>>,
      AST::JoinPlan<std::tuple<x_, y_, z_>>>;

  SemiNaiveDatabase<TriangleDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);
  using rule_set_if_test = AST::NonIterativeRuleSets<rule_if_test>;
  using executor_if_test = ExecuteRulesForNonIterativeRuleSets<rule_set_if_test>;

  // Add data
  add_fact<RelationR>(db, BooleanSR::one(), 2, 10);  // even
  add_fact<RelationR>(db, BooleanSR::one(), 3, 10);  // odd
  add_fact<RelationR>(db, BooleanSR::one(), 6, 30);  // even
  add_fact<RelationR>(db, BooleanSR::one(), 7, 40);  // odd

  add_fact<RelationS>(db, BooleanSR::one(), 10, 8, 9);
  add_fact<RelationS>(db, BooleanSR::one(), 10, 9, 4);
  add_fact<RelationS>(db, BooleanSR::one(), 30, 6, 12);
  add_fact<RelationS>(db, BooleanSR::one(), 40, 7, 57);

  add_fact<RelationT>(db, BooleanSR::one(), 8, 2, 3);
  add_fact<RelationT>(db, BooleanSR::one(), 9, 3, 3);
  add_fact<RelationT>(db, BooleanSR::one(), 6, 6, 2);
  add_fact<RelationT>(db, BooleanSR::one(), 7, 7, 6);

  executor_if_test{}(db);
  // For non-iterative rules, data is written to FULL_VER, not NEW_VER
  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  rel_z.ensure_index(rel_z.get_default_index());

  // x = 2 (even), kept
  std::array<std::size_t, 3> tuple1 = {2, 10, 8};
  bool in_relation_2_10_8 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_REQUIRE(in_relation_2_10_8);

  // x = 3 (odd), filtered out
  std::array<std::size_t, 3> tuple2 = {3, 10, 9};
  bool in_relation_3_10_9 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_REQUIRE(!in_relation_3_10_9);

  // x = 6 (even), kept
  std::array<std::size_t, 3> tuple3 = {6, 30, 6};
  bool in_relation_6_30_6 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_REQUIRE(in_relation_6_30_6);

  // x = 7 (odd), filtered out
  std::array<std::size_t, 3> tuple4 = {7, 40, 7};
  bool in_relation_7_40_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple4);
  BOOST_REQUIRE(!in_relation_7_40_7);

  // Only even x values
  BOOST_REQUIRE_EQUAL(rel_z.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_let_clause_computation) {
  std::cout << "=== Starting test_let_clause_computation ===" << std::endl;
  using SR = BooleanSR;
  using Arena = boost::container::pmr::monotonic_buffer_resource;
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;

  using RelationR = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
  using RelationZ = AST::RelationSchema<decltype("Z"_s), SR, std::tuple<int, int, int>>;
  using LetDBBlueprint = AST::Database<RelationR, RelationZ>;

  using x_ = AST::Var<decltype("x"_s)>;
  using y_ = AST::Var<decltype("y"_s)>;
  using sum_ = AST::Var<decltype("sum"_s)>;

  // Rule: Z(x, y, sum) :- R(x, y), let sum = x + y
  using rule_let_test = AST::Rule<
      std::tuple<AST::Clause<RelationZ, NEW_VER, x_, y_, sum_>>,
      std::tuple<AST::Clause<RelationR, FULL_VER, x_, y_>,
                 // Let clause: bind sum to x + y
                 AST::LetClause<sum_, AST::CppExpr<std::tuple<x_, y_>,
                                                   [](int x, int y) -> int { return x + y; }>>>,
      AST::JoinPlan<std::tuple<x_, y_, sum_>>>;

  SemiNaiveDatabase<LetDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);
  using rule_set_let_test = AST::NonIterativeRuleSets<rule_let_test>;
  using executor_let_test = ExecuteRulesForNonIterativeRuleSets<rule_set_let_test>;

  // Add facts to relation R
  add_fact<RelationR>(db, BooleanSR::one(), 1, 2);    // sum = 3
  add_fact<RelationR>(db, BooleanSR::one(), 3, 4);    // sum = 7
  add_fact<RelationR>(db, BooleanSR::one(), 5, 6);    // sum = 11
  add_fact<RelationR>(db, BooleanSR::one(), 10, 20);  // sum = 30

  executor_let_test{}(db);
  // For non-iterative rules, data is written to FULL_VER, not NEW_VER
  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  rel_z.ensure_index(rel_z.get_default_index());

  // Verify results: Z(x, y, sum) where sum = x + y
  std::array<std::size_t, 3> tuple1 = {1, 2, 3};
  bool in_relation_1_2_3 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_REQUIRE(in_relation_1_2_3);

  std::array<std::size_t, 3> tuple2 = {3, 4, 7};
  bool in_relation_3_4_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_REQUIRE(in_relation_3_4_7);

  std::array<std::size_t, 3> tuple3 = {5, 6, 11};
  bool in_relation_5_6_11 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_REQUIRE(in_relation_5_6_11);

  std::array<std::size_t, 3> tuple4 = {10, 20, 30};
  bool in_relation_10_20_30 = is_in_relation<RelationZ, FULL_VER>(db, tuple4);
  BOOST_REQUIRE(in_relation_10_20_30);

  // Verify that incorrect sums are not in the relation
  std::array<std::size_t, 3> tuple5 = {1, 2, 4};
  bool in_relation_1_2_4 = is_in_relation<RelationZ, FULL_VER>(db, tuple5);
  BOOST_REQUIRE(!in_relation_1_2_4);

  // Verify total count
  BOOST_REQUIRE_EQUAL(rel_z.size(), 4);
}
