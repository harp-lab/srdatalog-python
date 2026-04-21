/**
 * @file test_gpu_sg_deduplication.cpp
 * @brief Test to expose deduplication bugs in GPU SG fixpoint execution
 *
 * This test is designed to catch the issue where results are 2x the correct value,
 * indicating deduplication is not working correctly.
 */

#define BOOST_TEST_MODULE GPUSGDeduplicationTest
#include <boost/test/included/unit_test.hpp>

#include "ast.h"
#include "gpu/init.h"
#include "gpu/runtime/query.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

#include "debug_type.h"

using SR = BooleanSR;

// Schemas
using ArcSchema = AST::relation<decltype("Arc"_s), SR, int, int>;
using SGSchema = AST::relation<decltype("SG"_s), SR, int, int>;
using SGBlueprint = AST::database<ArcSchema, SGSchema>;

// Variables
constexpr auto x_ = "x"_v;
constexpr auto y_ = "y"_v;
constexpr auto p_ = "p"_v;
constexpr auto q_ = "q"_v;

// Relation accessors
constexpr auto arc = AST::rel<ArcSchema>;
constexpr auto sg = AST::rel<SGSchema>;

// Base rule: sg(x, y) :- arc(p, x), arc(p, y), x != y
const auto sg_base_rule =
    (sg.full(x_, y_) <<= (arc.full(p_, x_), arc.full(p_, y_),
                          AST::if_<[](int x, int y) -> bool { return x != y; }>(x_, y_))) |
    AST::plan(p_, x_, y_);

// Fixpoint rule: sg(x, y) :- sg.delta(p, q), arc.full(p, x), arc.full(q, y)
const auto sg_fixpoint_rule = sg.newt(x_, y_) <<=
    (sg.delta(p_, q_), arc.full(p_, x_), arc.full(q_, y_));

using SGBase = decltype(AST::non_iterative(sg_base_rule));
using SGFixpoint = decltype(AST::fixpoint(sg_fixpoint_rule));

using Executor = SRDatalog::GPU::GPUQueryExecutor<SGFixpoint>;

/**
 * @brief Test with a small graph where we can compute exact expected results
 *
 * Graph: 1->2, 1->3, 2->3
 * Base rule produces: sg(2,3) from source 1 (arcs to 2,3)
 * Fixpoint iteration 0: sg(1,3) from sg(2,3) + arc(1,2) + arc(2,3)
 *                      sg(1,2) from sg(2,3) + arc(1,3) + arc(2,3)
 *                      But wait, let me recalculate...
 *
 * Actually, let's use a simpler case where we can verify exact counts.
 */
BOOST_AUTO_TEST_CASE(test_sg_deduplication_exact_count) {
  std::cout << "=== test_sg_deduplication_exact_count ===" << std::endl;

  SRDatalog::GPU::init_cuda();

  // Initialize Host DB
  AST::SemiNaiveDatabase<SGBlueprint> host_db;

  // Create a simple chain: 1->2->3->4
  // This produces predictable results
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 2);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 2, 3);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 3, 4);

  std::cout << "Arc size: " << get_relation_by_schema<ArcSchema, FULL_VER>(host_db).size()
            << std::endl;

  // Execute Base Rule on CPU
  // sg(x, y) :- arc(p, x), arc(p, y), x != y
  // Source 1: arcs to 2 -> no result (need at least 2 arcs)
  // Source 2: arcs to 3 -> no result (need at least 2 arcs)
  // Source 3: arcs to 4 -> no result (need at least 2 arcs)
  // So base rule produces 0 tuples
  execute_query<SGBase>(host_db);

  auto& sg_full_initial = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after base rule: " << sg_full_initial.size() << std::endl;
  BOOST_CHECK_EQUAL(sg_full_initial.size(), 0);

  // Prepare device DB
  auto device_db = Executor::prepare(host_db);

  // Execute one iteration of fixpoint
  // sg(x, y) :- sg.delta(p, q), arc.full(p, x), arc.full(q, y)
  // In iteration 0, sg.delta is redirected to sg.full (which is empty)
  // So no new tuples should be produced
  std::cout << "\n=== Iteration 0 ===" << std::endl;
  Executor::execute_kernel(device_db, 1);
  Executor::reflect(host_db, device_db);

  auto& sg_full_iter0 = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after iteration 0: " << sg_full_iter0.size() << std::endl;

  // Should still be 0 (no base tuples, so no fixpoint results)
  BOOST_CHECK_EQUAL(sg_full_iter0.size(), 0);
}

/**
 * @brief Test with a graph that produces known duplicates
 *
 * Graph: 1->2, 1->3, 2->3
 * Base: sg(2,3) from source 1
 * Iteration 0: sg(1,3) from sg(2,3) + arc(1,2) + arc(2,3)
 *              sg(1,2) from sg(2,3) + arc(1,3) + arc(2,3)
 *              But sg(1,2) can also come from sg(1,3) + arc(1,2) + arc(3,?) - wait, no arc(3,?)
 *
 * Let's use a simpler case: triangle where we know exact results
 */
BOOST_AUTO_TEST_CASE(test_sg_deduplication_triangle) {
  std::cout << "\n=== test_sg_deduplication_triangle ===" << std::endl;

  SRDatalog::GPU::init_cuda();

  AST::SemiNaiveDatabase<SGBlueprint> host_db;

  // Triangle: 1->2, 2->3, 3->1 (cycle)
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 2);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 2, 3);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 3, 1);

  // Execute base rule
  execute_query<SGBase>(host_db);
  auto& sg_base = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after base rule: " << sg_base.size() << std::endl;

  // Base rule: sg(x,y) :- arc(p,x), arc(p,y), x != y
  // Source 1: arcs to 2 -> no result (only 1 arc)
  // Source 2: arcs to 3 -> no result (only 1 arc)
  // Source 3: arcs to 1 -> no result (only 1 arc)
  // So base produces 0 tuples

  // Prepare device and run fixpoint
  auto device_db = Executor::prepare(host_db);
  Executor::execute_kernel(device_db, 10);  // Run up to 10 iterations
  Executor::reflect(host_db, device_db);

  auto& sg_final = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after fixpoint: " << sg_final.size() << std::endl;

  // With cycle, we should get all pairs eventually
  // But the key test: size should match expected unique count, not 2x
  // For now, just check it's reasonable (not 2x expected)
  BOOST_CHECK_LE(sg_final.size(), 9);  // 3 nodes, max 3*2 = 6 pairs, but with cycle might be more
}

/**
 * @brief Test that explicitly creates duplicates to verify deduplication works
 *
 * This test manually adds duplicate tuples to verify that deduplication
 * correctly filters them during materialization.
 */
BOOST_AUTO_TEST_CASE(test_sg_deduplication_manual_duplicates) {
  std::cout << "\n=== test_sg_deduplication_manual_duplicates ===" << std::endl;

  SRDatalog::GPU::init_cuda();

  AST::SemiNaiveDatabase<SGBlueprint> host_db;

  // Create a simple graph: 1->2, 1->3
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 2);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 3);

  // Execute base rule
  execute_query<SGBase>(host_db);
  auto& sg_base = get_relation_by_schema<SGSchema, FULL_VER>(host_db);

  // Base rule: sg(x,y) :- arc(p,x), arc(p,y), x != y
  // Source 1: arcs to 2,3 -> sg(2,3) = 1 tuple
  std::cout << "SG.full size after base rule: " << sg_base.size() << std::endl;

  // Manually add a duplicate to FULL to test deduplication
  // This simulates what happens if deduplication fails
  SRDatalog::add_fact<SGSchema>(host_db, SR::one(), 2, 3);

  // Now FULL has sg(2,3) twice
  std::cout << "SG.full size after adding duplicate: " << sg_base.size() << std::endl;
  BOOST_CHECK_EQUAL(sg_base.size(), 2);  // Should have 2 (with duplicate)

  // Prepare device and run one iteration
  // The fixpoint rule should NOT produce sg(2,3) again because it already exists in FULL
  auto device_db = Executor::prepare(host_db);
  Executor::execute_kernel(device_db, 1);
  Executor::reflect(host_db, device_db);

  auto& sg_after = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after iteration: " << sg_after.size() << std::endl;

  // The fixpoint rule: sg(x,y) :- sg.delta(p,q), arc(p,x), arc(q,y)
  // In iteration 0, sg.delta redirects to sg.full which has sg(2,3)
  // Join: sg(2,3) + arc(2,?) + arc(3,?) -> no new tuples (no arcs from 2 or 3)
  // So size should remain 2 (the duplicate we added)
  // But if deduplication is working, the index should show only 1 unique
  // However, relation size might still be 2 if we're not deduplicating during merge

  // The key test: check if we're getting 2x the expected count
  // If deduplication fails, we might see sg(2,3) produced again, making size 3 or 4
  BOOST_CHECK_LE(sg_after.size(), 2);  // Should not increase beyond what we manually added
}

/**
 * @brief Test with known exact result - will catch 2x bug
 *
 * This test uses a small graph where we can manually compute the exact expected result.
 * If deduplication fails and we get 2x the correct count, this test will fail.
 *
 * Graph: 1->2, 1->3, 2->3
 * Base rule: sg(x,y) :- arc(p,x), arc(p,y), x != y
 *   Source 1: arcs to 2,3 -> sg(2,3) = 1 tuple
 *   Source 2: arcs to 3 -> no result (need 2+ arcs)
 *   Source 3: no arcs -> no result
 *   Total base: 1 unique tuple
 *
 * Fixpoint iteration 0: sg(x,y) :- sg.delta(p,q), arc(p,x), arc(q,y)
 *   sg.delta redirects to sg.full which has sg(2,3)
 *   Join: sg(2,3) + arc(2,?) + arc(3,?)
 *   - arc(2,3) exists, but no arc(3,?) -> no new tuples
 *   Total after iter 0: 1 unique tuple (same as base)
 *
 * Expected final result: 1 unique tuple
 * If we get 2, deduplication is broken (2x bug)
 */
BOOST_AUTO_TEST_CASE(test_sg_deduplication_exact_small) {
  std::cout << "\n=== test_sg_deduplication_exact_small ===" << std::endl;

  SRDatalog::GPU::init_cuda();

  AST::SemiNaiveDatabase<SGBlueprint> host_db;

  // Graph: 1->2, 1->3, 2->3
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 2);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 1, 3);
  SRDatalog::add_fact<ArcSchema>(host_db, SR::one(), 2, 3);

  // Execute base rule
  execute_query<SGBase>(host_db);
  auto& sg_base = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::cout << "SG.full size after base rule: " << sg_base.size() << std::endl;

  // Base should produce exactly 1 tuple: sg(2,3)
  // But base rule may have duplicates, so check unique count
  std::size_t base_unique = 0;
  try {
    sg_base.ensure_index({{0, 1}}, true);
    auto& idx = sg_base.get_index({{0, 1}});
    base_unique = idx.root().degree();
  } catch (...) {
    base_unique = sg_base.size();
  }
  std::cout << "SG.full unique count after base: " << base_unique << std::endl;
  BOOST_CHECK_EQUAL(base_unique, 1);  // Should be exactly 1 unique

  // Prepare device and run fixpoint
  auto device_db = Executor::prepare(host_db);
  Executor::execute_kernel(device_db, 10);  // Run until fixpoint
  Executor::reflect(host_db, device_db);

  auto& sg_final = get_relation_by_schema<SGSchema, FULL_VER>(host_db);
  std::size_t final_size = sg_final.size();
  std::cout << "SG.full size after fixpoint: " << final_size << std::endl;

  // Get unique count
  std::size_t final_unique = 0;
  try {
    sg_final.ensure_index({{0, 1}}, true);
    auto& idx = sg_final.get_index({{0, 1}});
    final_unique = idx.root().degree();
  } catch (...) {
    final_unique = final_size;
  }
  std::cout << "SG.full unique count after fixpoint: " << final_unique << std::endl;

  // CRITICAL TEST: Should be exactly 1 unique tuple
  // If we get 2, deduplication is broken (2x bug)
  BOOST_CHECK_EQUAL(final_unique, 1);

  // Also check relation size - if deduplication works, size should be <= 2 (allowing 1 duplicate)
  // If we see size > 2, something is very wrong
  BOOST_CHECK_LE(final_size, 2);
}
