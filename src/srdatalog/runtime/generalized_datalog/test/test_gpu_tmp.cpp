/**
 * @file test_gpu_tmp.cpp
 * @brief Test compiled pipelines on GPU (Port of test_tmp.cpp) using high-level query compiler
 */

#define BOOST_TEST_MODULE GPUTMPQueryCompilerExecutor
#include <boost/test/included/unit_test.hpp>

#include "ast.h"
#include "gpu/runtime/query.h"  // High-level GPU query execution
#include "query.h"
#include "runtime.h"
#include "semiring.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

BOOST_AUTO_TEST_CASE(test_gpu_triangle_join) {
  std::cout << "=== Starting test_gpu_triangle_join ===" << std::endl;
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
      std::tuple<  // Body: R(x,y), S(y,z,h), T(z,x,f) with FULL_VER
          AST::Clause<RelationR, FULL_VER, x_, y_>, AST::Clause<RelationS, FULL_VER, y_, z_, h_>,
          AST::Clause<RelationT, FULL_VER, z_, x_, f_>>,
      AST::JoinPlan<std::tuple<x_, y_, z_>>>;

  // GPU Compiler currently supports Fixpoint for compilation.
  // We wrap our single rule in a Fixpoint. Since R, S, T are static (EDB) and Z is output (IDB),
  // this will reach fixpoint in one or two iterations.
  using FixpointType = AST::Fixpoint<rule1>;

  SemiNaiveDatabase<TriangleDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  // Get relation references directly to avoid add_fact compilation issues
  auto& rel_r = get_relation_by_schema<RelationR, FULL_VER>(db);
  auto& rel_s = get_relation_by_schema<RelationS, FULL_VER>(db);
  auto& rel_t = get_relation_by_schema<RelationT, FULL_VER>(db);

  // Add facts to relation R
  rel_r.push_row({1, 10}, BooleanSR::one());
  rel_r.push_row({2, 10}, BooleanSR::one());
  rel_r.push_row({3, 10}, BooleanSR::one());
  rel_r.push_row({4, 10}, BooleanSR::one());
  rel_r.push_row({5, 20}, BooleanSR::one());
  rel_r.push_row({6, 30}, BooleanSR::one());
  rel_r.push_row({7, 40}, BooleanSR::one());
  rel_r.push_row({8, 50}, BooleanSR::one());

  // Add facts to relation S
  rel_s.push_row({10, 7, 5}, BooleanSR::one());
  rel_s.push_row({10, 8, 9}, BooleanSR::one());
  rel_s.push_row({10, 9, 4}, BooleanSR::one());
  rel_s.push_row({10, 100, 3}, BooleanSR::one());
  rel_s.push_row({20, 7, 6}, BooleanSR::one());
  rel_s.push_row({30, 6, 12}, BooleanSR::one());
  rel_s.push_row({40, 7, 57}, BooleanSR::one());
  rel_s.push_row({50, 7, 34}, BooleanSR::one());

  // Add facts to relation T
  rel_t.push_row({7, 1, 4}, BooleanSR::one());
  rel_t.push_row({8, 2, 3}, BooleanSR::one());
  rel_t.push_row({9, 3, 3}, BooleanSR::one());
  rel_t.push_row({7, 5, 4}, BooleanSR::one());
  rel_t.push_row({6, 6, 2}, BooleanSR::one());
  rel_t.push_row({7, 7, 6}, BooleanSR::one());
  rel_t.push_row({7, 8, 2}, BooleanSR::one());
  rel_t.push_row({123, 4, 1}, BooleanSR::one());

  // Execution on GPU
  // This manages data mirroring, execution, and result reflection back to host.
  SRDatalog::GPU::execute_gpu_query<FixpointType>(db);

  // Verify results on host
  // execute_gpu_query reflects results back to FULL_VER in host DB.
  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  // Ensure index for checking
  rel_z.ensure_index(rel_z.get_default_index());

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Relation Z size: " << rel_z.size() << std::endl;

  BOOST_CHECK_EQUAL(rel_z.size(), 7);

  // Verify that the expected tuple (1, 10, 7) is in the result relation
  std::array<std::size_t, 3> tuple1 = {1, 10, 7};
  bool in_relation_1_10_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_CHECK(in_relation_1_10_7);
  std::array<std::size_t, 3> tuple2 = {2, 10, 8};
  bool in_relation_2_10_8 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_CHECK(in_relation_2_10_8);
  std::array<std::size_t, 3> tuple3 = {3, 10, 9};
  bool in_relation_3_10_9 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_CHECK(in_relation_3_10_9);

  // Check other results
  std::array<std::size_t, 3> t5 = {5, 20, 7};
  bool r5 = is_in_relation<RelationZ, FULL_VER>(db, t5);
  BOOST_CHECK(r5);

  std::array<std::size_t, 3> t6 = {6, 30, 6};
  bool r6 = is_in_relation<RelationZ, FULL_VER>(db, t6);
  BOOST_CHECK(r6);

  std::array<std::size_t, 3> t7 = {7, 40, 7};
  bool r7 = is_in_relation<RelationZ, FULL_VER>(db, t7);
  BOOST_CHECK(r7);

  std::array<std::size_t, 3> t8 = {8, 50, 7};
  bool r8 = is_in_relation<RelationZ, FULL_VER>(db, t8);
  BOOST_CHECK(r8);
}

BOOST_AUTO_TEST_CASE(test_gpu_if_clause_modulo_filter) {
  std::cout << "=== Starting test_gpu_if_clause_modulo_filter ===" << std::endl;
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

  // GPU Compiler currently supports Fixpoint for compilation.
  using FixpointType = AST::Fixpoint<rule_if_test>;

  SemiNaiveDatabase<TriangleDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  auto& rel_r = get_relation_by_schema<RelationR, FULL_VER>(db);
  auto& rel_s = get_relation_by_schema<RelationS, FULL_VER>(db);
  auto& rel_t = get_relation_by_schema<RelationT, FULL_VER>(db);

  // Add data
  rel_r.push_row({2, 10}, BooleanSR::one());  // even
  rel_r.push_row({3, 10}, BooleanSR::one());  // odd
  rel_r.push_row({6, 30}, BooleanSR::one());  // even
  rel_r.push_row({7, 40}, BooleanSR::one());  // odd

  rel_s.push_row({10, 8, 9}, BooleanSR::one());
  rel_s.push_row({10, 9, 4}, BooleanSR::one());
  rel_s.push_row({30, 6, 12}, BooleanSR::one());
  rel_s.push_row({40, 7, 57}, BooleanSR::one());

  rel_t.push_row({8, 2, 3}, BooleanSR::one());
  rel_t.push_row({9, 3, 3}, BooleanSR::one());
  rel_t.push_row({6, 6, 2}, BooleanSR::one());
  rel_t.push_row({7, 7, 6}, BooleanSR::one());

  // Execute on GPU
  SRDatalog::GPU::execute_gpu_query<FixpointType>(db);

  // Check results
  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  rel_z.ensure_index(rel_z.get_default_index());

  // x = 2 (even), kept
  std::array<std::size_t, 3> tuple1 = {2, 10, 8};
  bool in_relation_2_10_8 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_CHECK(in_relation_2_10_8);

  // x = 3 (odd), filtered out
  std::array<std::size_t, 3> tuple2 = {3, 10, 9};
  bool in_relation_3_10_9 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_CHECK(!in_relation_3_10_9);

  // x = 6 (even), kept
  std::array<std::size_t, 3> tuple3 = {6, 30, 6};
  bool in_relation_6_30_6 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_CHECK(in_relation_6_30_6);

  // x = 7 (odd), filtered out
  std::array<std::size_t, 3> tuple4 = {7, 40, 7};
  bool in_relation_7_40_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple4);
  BOOST_CHECK(!in_relation_7_40_7);

  // Only even x values (2 tuples)
  BOOST_CHECK_EQUAL(rel_z.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_gpu_let_clause_computation) {
  std::cout << "=== Starting test_gpu_let_clause_computation ===" << std::endl;
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

  // GPU Compiler usage
  using FixpointType = AST::Fixpoint<rule_let_test>;

  SemiNaiveDatabase<LetDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  auto& rel_r = get_relation_by_schema<RelationR, FULL_VER>(db);

  // Add facts to relation R
  rel_r.push_row({1, 2}, BooleanSR::one());    // sum = 3
  rel_r.push_row({3, 4}, BooleanSR::one());    // sum = 7
  rel_r.push_row({5, 6}, BooleanSR::one());    // sum = 11
  rel_r.push_row({10, 20}, BooleanSR::one());  // sum = 30

  // Execute on GPU
  using Plan = typename SRDatalog::GPU::GPUQueryExecutor<FixpointType>::Plan;

  SRDatalog::GPU::execute_gpu_query<FixpointType>(db);

  auto& rel_z = get_relation_by_schema<RelationZ, FULL_VER>(db);
  rel_z.ensure_index(rel_z.get_default_index());

  // Verify results: Z(x, y, sum) where sum = x + y
  std::array<std::size_t, 3> tuple1 = {1, 2, 3};
  bool in_relation_1_2_3 = is_in_relation<RelationZ, FULL_VER>(db, tuple1);
  BOOST_CHECK(in_relation_1_2_3);

  std::array<std::size_t, 3> tuple2 = {3, 4, 7};
  bool in_relation_3_4_7 = is_in_relation<RelationZ, FULL_VER>(db, tuple2);
  BOOST_CHECK(in_relation_3_4_7);

  std::array<std::size_t, 3> tuple3 = {5, 6, 11};
  bool in_relation_5_6_11 = is_in_relation<RelationZ, FULL_VER>(db, tuple3);
  BOOST_CHECK(in_relation_5_6_11);

  std::array<std::size_t, 3> tuple4 = {10, 20, 30};
  bool in_relation_10_20_30 = is_in_relation<RelationZ, FULL_VER>(db, tuple4);
  BOOST_CHECK(in_relation_10_20_30);

  // Verify that incorrect sums are not in the relation
  std::array<std::size_t, 3> tuple5 = {1, 2, 4};
  bool in_relation_1_2_4 = is_in_relation<RelationZ, FULL_VER>(db, tuple5);
  BOOST_CHECK(!in_relation_1_2_4);

  // Verify total count
  BOOST_CHECK_EQUAL(rel_z.size(), 4);
}
