#define BOOST_TEST_MODULE gpu_incomplete_materialize_test
#include "ast.h"
#include "mir.h"
#include "mir_printer.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

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

// Define schemas
// InsertInput: 4 ints
using InsertInputSchema =
    RelationSchema<decltype("InsertInput"_s), SR, std::tuple<int, int, int, int>>;

// Insert: 4 ints
using InsertSchema = RelationSchema<decltype("Insert"_s), SR, std::tuple<int, int, int, int>>;

// Assign: 5 ints
using AssignSchema = RelationSchema<decltype("Assign"_s), SR, std::tuple<int, int, int, int, int>>;

using CRDTBlueprint = Database<InsertInputSchema, InsertSchema, AssignSchema>;

// Variables
constexpr auto y = "y"_v;
constexpr auto x = "x"_v;
constexpr auto z = "z"_v;
constexpr auto w = "w"_v;

constexpr auto ctr = "ctr"_v;
constexpr auto n = "n"_v;
constexpr auto v1 = "v1"_v;
constexpr auto v2 = "v2"_v;

// Relation accessors
constexpr auto insert_input = rel<InsertInputSchema>;
constexpr auto insert = rel<InsertSchema>;
constexpr auto assign = rel<AssignSchema>;

// Rule 1: Copy Cartesian (arity 4)
// insert(y, x, z, w) :- insert_input(y, x, z, w)
auto copy_rule = insert.newt(y, x, z, w) <<= insert_input.full(y, x, z, w);

// Rule 2: Wildcard / Incomplete Materialization
// assign(ctr, n, ctr, n, n) :- insert(ctr, n, v1, v2)
auto wildcard_rule = assign.newt(ctr, n, ctr, n, n) <<= insert.full(ctr, n, v1, v2);

using CopyProgram = decltype(non_iterative(copy_rule));
using WildcardProgram = decltype(non_iterative(wildcard_rule));

BOOST_AUTO_TEST_SUITE(gpu_incomplete_materialize_test_suite)

BOOST_AUTO_TEST_CASE(test_copy_cartesian_arity_4) {
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<CRDTBlueprint> db(&arena, &arena, &arena);

  // 1. Add data to InsertInput
  // (1, 2, 3, 4), (5, 6, 7, 8)
  add_fact<InsertInputSchema>(db, BooleanSR::one(), 1, 2, 3, 4);
  add_fact<InsertInputSchema>(db, BooleanSR::one(), 5, 6, 7, 8);

  // 2. Execute Copy Rule
  // insert <- insert_input
  execute_gpu_query<CopyProgram>(db);

  // 3. Reflect to verify - check FULL_VER since NEW is cleared after merge
  auto& insert_rel = get_relation_by_schema<InsertSchema, FULL_VER>(db);

  // Create an index to check size/contents properly (though reflect should populate internal
  // vector)
  SRDatalog::IndexSpec all_cols{{0, 1, 2, 3}};
  insert_rel.ensure_index(all_cols, true);  // Force rebuild

  BOOST_CHECK_EQUAL(insert_rel.size(), 2);

  // Verify contents if possible, but size 2 implies both tuples copied
}

BOOST_AUTO_TEST_CASE(test_wildcard_optimization_check) {
  // Extract the single rule from WildcardProgram (NonIterativeRuleSets)
  // GetRulesFromQuery is in AST namespace
  using RuleTuple = typename SRDatalog::AST::GetRulesFromQuery<WildcardProgram>::type;
  using RuleType = std::tuple_element_t<0, RuleTuple>;

  // Compile to Pipeline using mir::CompileRuleToPipeline trait (in SRDatalog namespace)
  using PipelineType = typename SRDatalog::mir::CompileRuleToPipeline<RuleType>::type;

  std::stringstream ss;
  print_mir<PipelineType>(ss);
  std::string mir_output = ss.str();

  std::cout << ">>> MIR for Wildcard Rule <<<" << std::endl;
  std::cout << mir_output << std::endl;
  std::cout << ">>> End MIR <<<" << std::endl;
  bool v1_present = (mir_output.find("v1") != std::string::npos);
  bool v2_present = (mir_output.find("v2") != std::string::npos);

  if (v1_present || v2_present) {
    std::cout << "[INFO] Wildcard variables AR materialised in MIR." << std::endl;
  } else {
    std::cout << "[INFO] Wildcard variables are NOT materialised (Optimized!)." << std::endl;
  }

  // Also run it to ensure correctness (no crash)
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<CRDTBlueprint> db(&arena, &arena, &arena);

  // insert(10, 20, 100, 200) -> assign(10, 20, 10, 20, 20)
  add_fact<InsertSchema>(db, BooleanSR::one(), 10, 20, 100, 200);

  execute_gpu_query<WildcardProgram>(db);

  auto& assign_rel = get_relation_by_schema<AssignSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(assign_rel.size(), 1);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
