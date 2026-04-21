#define BOOST_TEST_MODULE mir_printer_test
#include <boost/test/included/unit_test.hpp>

#include "ast.h"
#include "gpu/runtime/query.h"
#include "mir_printer.h"
#include "query.h"
#include "runtime.h"
#include "semiring.h"
#include <sstream>

namespace sd = SRDatalog;
namespace ast = sd::AST;

using ast::database;
using ast::fixpoint;
using ast::if_;
using ast::non_iterative;
using ast::plan;
using ast::rel;
using ast::relation;
using ast::operator""_v;
using ast::Literals::operator""_s;

using ::BooleanSR;

namespace {

using SR = BooleanSR;

// SG test case: same-gate computation
// Define schemas
using ArcSchema = relation<decltype("Arc"_s), SR, int, int>;
using SGSchema = relation<decltype("SG"_s), SR, int, int>;
using SGBlueprint = database<ArcSchema, SGSchema>;

// Variables
constexpr auto x_ = "x"_v;
constexpr auto y_ = "y"_v;
constexpr auto z_ = "z"_v;
constexpr auto p_ = "p"_v;
constexpr auto q_ = "q"_v;

// Relation accessors
constexpr auto arc = rel<ArcSchema>;
constexpr auto sg = rel<SGSchema>;

// sg(x, y) :- arc(p, x), arc(p, y)
const auto sg_base_rule =
    (sg.full(x_, y_) <<= (arc.full(p_, x_), arc.full(p_, y_),
                          if_<[](int x, int y) -> bool { return x != y; }>(x_, y_))) |
    plan(p_, x_, y_);

// sg(x, y) :- arc(p, x), sg(p, q), arc(q, y)
const auto sg_fixpoint_rule =
    (sg.newt(x_, y_) <<= (sg.delta(p_, q_), arc.full(p_, x_), arc.full(q_, y_))) |
    plan(p_, q_, x_, y_);

using SGBase = decltype(non_iterative(sg_base_rule));
using SGFixpoint = decltype(fixpoint(sg_fixpoint_rule));

// Get the GPU executor plan
using Executor = SRDatalog::GPU::GPUQueryExecutor<SGFixpoint>;
using SGPlan = typename Executor::Plan;

BOOST_AUTO_TEST_CASE(test_sg_plan_print) {
  std::stringstream ss;
  SRDatalog::print_mir<SGPlan>(ss);
  std::string output = ss.str();

  // Verify that the output contains expected MIR elements
  BOOST_CHECK(output.find("pipeline") != std::string::npos);
  BOOST_CHECK(output.find("cartesian-join") != std::string::npos);
  BOOST_CHECK(output.find("column-source") != std::string::npos);
  BOOST_CHECK(output.find("SG") != std::string::npos);
  BOOST_CHECK(output.find("Arc") != std::string::npos);

  // Print to stdout for manual inspection
  std::cout << "\n=== SG Plan MIR Output ===" << std::endl;
  std::cout << output << std::endl;
  std::cout << "=== End SG Plan MIR Output ===" << std::endl;
}

// Andersen test case: Andersen's Points-To Analysis
// Define schemas
using AddressOfSchema = relation<decltype("AddressOf"_s), SR, int, int>;
using AssignSchema = relation<decltype("Assign"_s), SR, int, int>;
using LoadSchema = relation<decltype("Load"_s), SR, int, int>;
using StoreSchema = relation<decltype("Store"_s), SR, int, int>;
using PointsToSchema = relation<decltype("PointsTo"_s), SR, int, int>;

using AndersenBlueprint =
    database<AddressOfSchema, AssignSchema, LoadSchema, StoreSchema, PointsToSchema>;

// Variables (reuse x_, y_, z_ from above; add w_)
constexpr auto w_ = "w"_v;

// Relation accessors
constexpr auto addressOf = rel<AddressOfSchema>;
constexpr auto assign = rel<AssignSchema>;
constexpr auto load = rel<LoadSchema>;
constexpr auto store = rel<StoreSchema>;
constexpr auto pointsTo = rel<PointsToSchema>;

// Base rule (non-iterative SCC): PointsTo(y, x) :- AddressOf(y, x)
const auto andersen_base_rule = pointsTo.newt(y_, x_) <<= addressOf.full(y_, x_);

// Rule 1: PointsTo(y, x) :- Assign(y, z), PointsTo(z, x)
const auto assign_rule = pointsTo.newt(y_, x_) <<= (pointsTo.delta(z_, x_), assign.full(y_, z_));

// Rule 2: PointsTo(y, w) :- Load(y, x), PointsTo(x, z), PointsTo(z, w)
const auto load_variant1 =
    (pointsTo.newt(y_, w_) <<= (pointsTo.delta(x_, z_), load.full(y_, x_), pointsTo.full(z_, w_)));
const auto load_variant2 =
    (pointsTo.newt(y_, w_) <<= (pointsTo.delta(z_, w_), pointsTo.full(x_, z_), load.full(y_, x_))) |
    plan(z_, x_, w_, y_);

// Rule 3: PointsTo(z, w) :- Store(y, x), PointsTo(y, z), PointsTo(x, w)
const auto store_variant1 = (pointsTo.newt(z_, w_) <<=
                             (pointsTo.delta(y_, z_), store.full(y_, x_), pointsTo.full(x_, w_))) |
                            plan(y_, x_, z_, w_);
const auto store_variant2 = (pointsTo.newt(z_, w_) <<=
                             (pointsTo.delta(x_, w_), store.full(y_, x_), pointsTo.full(y_, z_))) |
                            plan(x_, y_, w_, z_);

using AndersenBase = decltype(non_iterative(andersen_base_rule));
using AndersenFixpoint =
    decltype(fixpoint(assign_rule, load_variant1, load_variant2, store_variant1, store_variant2));

// Get the GPU executor plan
using AndersenExecutor = SRDatalog::GPU::GPUQueryExecutor<AndersenFixpoint>;
using AndersenPlan = typename AndersenExecutor::Plan;

BOOST_AUTO_TEST_CASE(test_andersen_plan_print) {
  // Note: Base rule (PointsTo(y, x) :- AddressOf(y, x)) uses NonIterativeRuleSets
  // which goes through a different code path than GPUQueryExecutor
  std::cout << "\n=== Andersen Base Rule ===" << std::endl;
  std::cout << "Base Rule: PointsTo(y, x) :- AddressOf(y, x)" << std::endl;
  std::cout << "(Uses NonIterativeRuleSets - separate code path)" << std::endl;

  // Print fixpoint rules MIR
  std::stringstream ss;
  SRDatalog::print_mir<AndersenPlan>(ss);
  SRDatalog::GPU::inspect<AndersenPlan>();

  std::string output = ss.str();

  // Verify that the output contains expected MIR elements
  BOOST_CHECK(output.find("pipeline") != std::string::npos);
  BOOST_CHECK(output.find("PointsTo") != std::string::npos);
  BOOST_CHECK(output.find("Assign") != std::string::npos ||
              output.find("Load") != std::string::npos ||
              output.find("Store") != std::string::npos);

  // Print to stdout for manual inspection
  std::cout << "\n=== Andersen Plan MIR Output ===" << std::endl;
  std::cout << output << std::endl;
  std::cout << "=== End Andersen Plan MIR Output ===" << std::endl;
}

}  // namespace
