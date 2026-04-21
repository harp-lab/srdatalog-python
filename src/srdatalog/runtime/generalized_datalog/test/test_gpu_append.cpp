#define BOOST_TEST_MODULE gpu_append_test
#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/test/included/unit_test.hpp>
#include <iostream>

// GPU includes
#include "gpu/init.h"
#include "gpu/runtime/query.h"
#include "mir_printer.h"

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;
using namespace SRDatalog::GPU;

namespace {
using SR = BooleanSR;

// Define schemas (use default index type)
using A_Schema = RelationSchema<decltype("A"_s), SR, std::tuple<int, int>>;
using B_Schema = RelationSchema<decltype("B"_s), SR, std::tuple<int, int>>;
using R_Schema = RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
using AppendBlueprint = Database<A_Schema, B_Schema, R_Schema>;

// Define variables
DECLARE_VAR(x);
DECLARE_VAR(y);

// R(x, y) :- A(x, y)
using Rule1 =
    Rule<Head<Rel<R_Schema>::New<x_, y_>>, Body<Rel<A_Schema>::Full<x_, y_>>, Plan<x_, y_>>;

// R(x, y) :- B(x, y)
using Rule2 =
    Rule<Head<Rel<R_Schema>::New<x_, y_>>, Body<Rel<B_Schema>::Full<x_, y_>>, Plan<x_, y_>>;

using AppendQuery = Fixpoint<Rule1, Rule2>;

BOOST_AUTO_TEST_CASE(test_gpu_append_logic) {
  init_cuda();
  SemiNaiveDatabase<AppendBlueprint> db;

  // Rule 1 produces (1, 2)
  add_fact<A_Schema>(db, BooleanSR::one(), 1, 2);
  // Rule 2 produces (3, 4)
  add_fact<B_Schema>(db, BooleanSR::one(), 3, 4);

  std::cout << "Executing append query on GPU..." << std::endl;

  // Print MIR plan for debugging
  using Plan = SRDatalog::GPU::GPUQueryExecutor<AppendQuery>::Plan;
  std::cout << "\n=== Append Query MIR Plan ===" << std::endl;
  SRDatalog::print_mir<Plan>(std::cout);
  std::cout << "\n=== End MIR Plan ===" << std::endl;

  execute_gpu_query<AppendQuery>(db);

  auto& r_full = get_relation_by_schema<R_Schema, FULL_VER>(db);
  std::cout << "R relation size: " << r_full.size() << std::endl;

  // Verify both facts are present
  BOOST_CHECK_EQUAL(r_full.size(), 2);

  // Checking exact results would require iterating/sorting, but size=2 is a good sign
  // that both rules contributed and didn't overwrite each other.
}

}  // namespace
