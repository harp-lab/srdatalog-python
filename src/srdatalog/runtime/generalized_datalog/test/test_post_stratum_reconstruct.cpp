// Unit test for PostStratumReconstructInternCols instruction
// Validates that intern columns are correctly sized in multi-stratum scenarios

#define BOOST_TEST_MODULE post_stratum_reconstruct_test
#include "ast.h"
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
#include "mir_def.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

namespace {

using SR = BooleanSR;

// Test Schema - 4 column relation (like MethodLookup)
// Schema: Parent(a, b, c, d) - simulates MethodImplemented
using ParentSchema = AST::relation<decltype("Parent"_s), SR, int, int, int, int>;
// Schema: Derived(a, b, c, d) - simulates MethodLookup
using DerivedSchema = AST::relation<decltype("Derived"_s), SR, int, int, int, int>;
// Schema: Super(c, s) - simulates DirectSuperclass
using SuperSchema = AST::relation<decltype("Super"_s), SR, int, int>;

using TestDB = AST::database<ParentSchema, DerivedSchema, SuperSchema>;
using HostTestDBType = AST::SemiNaiveDatabase<TestDB>;

// Variables (must be short for _v operator)
constexpr auto a_ = "a"_v;
constexpr auto b_ = "b"_v;
constexpr auto c_ = "c"_v;
constexpr auto d_ = "d"_v;
constexpr auto s_ = "s"_v;

// Relation accessors
constexpr auto parent = AST::rel<ParentSchema>;
constexpr auto derived = AST::rel<DerivedSchema>;
constexpr auto super = AST::rel<SuperSchema>;

// Base rule: Derived(a, b, c, d) :- Parent(a, b, c, d)
const auto derived_base_rule =
    (derived.full(a_, b_, c_, d_) <<= parent.full(a_, b_, c_, d_)) | AST::plan(a_, b_, c_, d_);

// Recursive rule: Derived(a, b, s, d) :- Super(c, s), Derived.delta(a, b, c, d)
// This models MethodLookup's recursive structure - methods propagate UP the hierarchy
// If method (a,b) is in type c, and s is superclass of c, then method is also in s
const auto derived_recursive_rule = derived.newt(a_, b_, s_, d_) <<=
    (super.full(c_, s_), derived.delta(a_, b_, c_, d_));

using DerivedBase = decltype(AST::non_iterative(derived_base_rule));
using DerivedFixpoint = decltype(AST::fixpoint(derived_recursive_rule));
using Executor = GPU::GPUQueryExecutor<DerivedFixpoint>;

BOOST_AUTO_TEST_SUITE(post_stratum_reconstruct_suite)

/**
 * Test basic reconstruction using rel.size()
 */
BOOST_AUTO_TEST_CASE(test_basic_reconstruct_with_canonical_index) {
  GPU::init_cuda();

  HostTestDBType host_db;

  // Add some facts to Parent
  add_fact<ParentSchema>(host_db, SR::one(), 1, 2, 3, 4);
  add_fact<ParentSchema>(host_db, SR::one(), 5, 6, 7, 8);
  add_fact<ParentSchema>(host_db, SR::one(), 9, 10, 11, 12);

  // Copy to device
  auto device_db = GPU::copy_host_to_device(host_db);

  // Get the relation and build canonical index
  auto& rel = get_relation_by_schema<ParentSchema, FULL_VER>(device_db);
  SRDatalog::IndexSpec canonical_spec{{0, 1, 2, 3}};
  rel.ensure_index(canonical_spec, true);
  auto& idx = rel.get_index(canonical_spec);

  auto idx_degree = idx.root().degree();
  std::cout << "[Test] Index size after ensure_index: " << idx_degree << std::endl;
  BOOST_CHECK_EQUAL(idx_degree, 3u);
}

/**
 * Test multi-stratum scenario simulating MethodLookup
 *
 * This test simulates:
 * 1. Stratum 1: Base case populates Derived from Parent (10 tuples)
 * 2. Stratum 2: Recursive fixpoint expands Derived via Super (30 total)
 * 3. PostStratumReconstruct should correctly report 30
 */
BOOST_AUTO_TEST_CASE(test_multi_stratum_recursive_scenario) {
  std::cout << "\n=== test_multi_stratum_recursive_scenario ===" << std::endl;

  GPU::init_cuda();

  HostTestDBType host_db;

  // Add base facts to Parent (simulates MethodImplemented)
  // Create 10 methods with different (a, b) pairs
  for (int method = 0; method < 10; ++method) {
    int a = method;
    int b = method * 10;
    int c = 100 + method;  // Each method in its own type
    int d = method;
    add_fact<ParentSchema>(host_db, SR::one(), a, b, c, d);
  }

  // Create a class hierarchy for Super
  // Types 100-109 all inherit from Type 200
  // Type 200 inherits from Type 300
  for (int i = 0; i < 10; ++i) {
    add_fact<SuperSchema>(host_db, SR::one(), 100 + i, 200);  // 100+i -> 200
  }
  add_fact<SuperSchema>(host_db, SR::one(), 200, 300);  // 200 -> 300

  auto& parent_rel = get_relation_by_schema<ParentSchema, FULL_VER>(host_db);
  auto& super_rel = get_relation_by_schema<SuperSchema, FULL_VER>(host_db);
  std::cout << "Parent size: " << parent_rel.size() << std::endl;
  std::cout << "Super size: " << super_rel.size() << std::endl;

  // Execute base rule on CPU
  execute_query<DerivedBase>(host_db);

  auto& derived_after_base = get_relation_by_schema<DerivedSchema, FULL_VER>(host_db);
  std::cout << "Derived size after base rule: " << derived_after_base.size() << std::endl;
  BOOST_CHECK_EQUAL(derived_after_base.size(), 10u);  // Should have 10 base facts

  // Now run fixpoint on GPU
  auto device_db = Executor::prepare(host_db);

  std::cout << "\n=== Running Fixpoint ===\n" << std::endl;
  Executor::execute_kernel(device_db, 100);  // Max 100 iterations

  Executor::reflect(host_db, device_db);

  auto& derived_after_fixpoint = get_relation_by_schema<DerivedSchema, FULL_VER>(host_db);
  std::cout << "Derived size after fixpoint: " << derived_after_fixpoint.size() << std::endl;

  // Check that size() is correct - this is what PostStratumReconstruct ensures
  auto derived_size = derived_after_fixpoint.size();
  std::cout << "Derived size (from relation): " << derived_size << std::endl;

  // Expected:
  // - Base: 10 tuples (method i in type 100+i)
  // - After iter 1: 10 more tuples (method i now also in type 200)
  // - After iter 2: 10 more tuples (method i now also in type 300)
  // Total: 30 unique tuples

  std::cout << "Expected size: 30" << std::endl;
  BOOST_CHECK_EQUAL(derived_size, 30u);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
