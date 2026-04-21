/**
 * @file test_gpu_negation_native.cpp
 * @brief Unit test for native GPU negation (anti-join) operation.
 *
 * This test uses the native NegClause AST type and mir::Negation clause
 * instead of the AggCount workaround.
 * It tests the fundamental anti-join operation:
 *   result(X, Y) :- source(X, Y), !blocked(X, Y)
 */

#define BOOST_TEST_MODULE gpu_negation_native_test
#include "ast.h"
#include "mir.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

// GPU includes
#include "gpu/init.h"
#include "gpu/runtime/query.h"

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;
using namespace SRDatalog::GPU;

namespace {
using SR = BooleanSR;
using Arena = boost::container::pmr::monotonic_buffer_resource;

// =============================================================================
// Schema Definitions
// =============================================================================

// Source relation: Source(X, Y)
using SourceSchema = RelationSchema<decltype("Source"_s), SR, std::tuple<uint32_t, uint32_t>>;

// Blocked relation: Blocked(X, Y) - used for negation check
using BlockedSchema = RelationSchema<decltype("Blocked"_s), SR, std::tuple<uint32_t, uint32_t>>;

// Result relation: Result(X, Y)
using ResultSchema = RelationSchema<decltype("Result"_s), SR, std::tuple<uint32_t, uint32_t>>;

using NegationTestDB = Database<SourceSchema, BlockedSchema, ResultSchema>;

// =============================================================================
// Variable Definitions
// =============================================================================

using VarX = Var<decltype("X"_s)>;
using VarY = Var<decltype("Y"_s)>;

// =============================================================================
// Rule Definition using NegClause
// =============================================================================
// Rule: Result(X, Y) :- Source(X, Y), !Blocked(X, Y)

// Head clause: Result.New(X, Y)
using ResultHead = Clause<ResultSchema, NEW_VER, VarX, VarY>;

// Body clause 1: Source.Full(X, Y)
using SourceBody = Clause<SourceSchema, FULL_VER, VarX, VarY>;

// Body clause 2: !Blocked.Full(X, Y) - NegClause
using BlockedNeg = NegClause<BlockedSchema, FULL_VER, VarX, VarY>;

// Join plan
using NegationPlan = Plan<VarX, VarY>;

// Complete rule
using NegationRule = Rule<Head<ResultHead>, Body<SourceBody, BlockedNeg>, NegationPlan>;

// Wrap in Fixpoint (single iteration for non-recursive)
using NegationProgram = Fixpoint<NegationRule>;

// =============================================================================
// Test Cases
// =============================================================================

BOOST_AUTO_TEST_CASE(test_simple_negation) {
  /**
   * Test case:
   *   Source: (1,1), (2,2), (3,3), (4,4)
   *   Blocked: (2,2), (4,4)
   *
   *   Expected Result: (1,1), (3,3)
   *   Because !Blocked(1,1) and !Blocked(3,3) are true.
   */
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<NegationTestDB> db(&arena, &arena, &arena);

  // Add Source facts
  add_fact<SourceSchema>(db, SR::one(), 1u, 1u);
  add_fact<SourceSchema>(db, SR::one(), 2u, 2u);
  add_fact<SourceSchema>(db, SR::one(), 3u, 3u);
  add_fact<SourceSchema>(db, SR::one(), 4u, 4u);

  // Add Blocked facts (these should be filtered out)
  add_fact<BlockedSchema>(db, SR::one(), 2u, 2u);
  add_fact<BlockedSchema>(db, SR::one(), 4u, 4u);

  std::cout << "Source facts: 4, Blocked facts: 2" << std::endl;

  // CPU Reference
  SemiNaiveDatabase<NegationTestDB> db_cpu(&arena, &arena, &arena);
  add_fact<SourceSchema>(db_cpu, SR::one(), 1u, 1u);
  add_fact<SourceSchema>(db_cpu, SR::one(), 2u, 2u);
  add_fact<SourceSchema>(db_cpu, SR::one(), 3u, 3u);
  add_fact<SourceSchema>(db_cpu, SR::one(), 4u, 4u);
  add_fact<BlockedSchema>(db_cpu, SR::one(), 2u, 2u);
  add_fact<BlockedSchema>(db_cpu, SR::one(), 4u, 4u);

  // Execute CPU
  execute_query<NegationProgram>(db_cpu);
  auto& cpu_result = get_relation_by_schema<ResultSchema, FULL_VER>(db_cpu);
  size_t cpu_count = cpu_result.size();
  std::cout << "CPU Result count: " << cpu_count << std::endl;

  // Execute GPU
  execute_gpu_query<NegationProgram>(db);
  auto& gpu_result = get_relation_by_schema<ResultSchema, FULL_VER>(db);
  size_t gpu_count = gpu_result.size();
  std::cout << "GPU Result count: " << gpu_count << std::endl;

  // Verify
  BOOST_CHECK_EQUAL(cpu_count, 2u);  // (1,1) and (3,3)
  BOOST_CHECK_EQUAL(gpu_count, cpu_count);
}

BOOST_AUTO_TEST_CASE(test_negation_empty_blocked) {
  /**
   * Test case: Blocked relation is empty.
   * Expected: All Source tuples pass through.
   */
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<NegationTestDB> db(&arena, &arena, &arena);

  // Add Source facts
  add_fact<SourceSchema>(db, SR::one(), 1u, 1u);
  add_fact<SourceSchema>(db, SR::one(), 2u, 2u);
  add_fact<SourceSchema>(db, SR::one(), 3u, 3u);

  // No Blocked facts

  std::cout << "Source facts: 3, Blocked facts: 0" << std::endl;

  // Execute GPU
  execute_gpu_query<NegationProgram>(db);
  auto& gpu_result = get_relation_by_schema<ResultSchema, FULL_VER>(db);
  size_t gpu_count = gpu_result.size();
  std::cout << "GPU Result count (empty blocked): " << gpu_count << std::endl;

  // All 3 should pass
  BOOST_CHECK_EQUAL(gpu_count, 3u);
}

BOOST_AUTO_TEST_CASE(test_negation_all_blocked) {
  /**
   * Test case: All Source tuples are blocked.
   * Expected: Result is empty.
   */
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<NegationTestDB> db(&arena, &arena, &arena);

  // Add Source facts
  add_fact<SourceSchema>(db, SR::one(), 1u, 1u);
  add_fact<SourceSchema>(db, SR::one(), 2u, 2u);

  // Block all
  add_fact<BlockedSchema>(db, SR::one(), 1u, 1u);
  add_fact<BlockedSchema>(db, SR::one(), 2u, 2u);

  std::cout << "Source facts: 2, Blocked facts: 2 (all blocked)" << std::endl;

  // Execute GPU
  execute_gpu_query<NegationProgram>(db);
  auto& gpu_result = get_relation_by_schema<ResultSchema, FULL_VER>(db);
  size_t gpu_count = gpu_result.size();
  std::cout << "GPU Result count (all blocked): " << gpu_count << std::endl;

  // None should pass
  BOOST_CHECK_EQUAL(gpu_count, 0u);
}

}  // namespace
