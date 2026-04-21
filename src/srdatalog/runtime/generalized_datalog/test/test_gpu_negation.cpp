#define BOOST_TEST_MODULE gpu_negation_test
#include "ast.h"
#include "query.h"
#include "runtime.h"
#include "runtime/aggregation.h"
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

// GPU includes
#include "gpu/init.h"
#include "gpu/runtime/query.h"
#include "mir.h"

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;
using namespace SRDatalog::GPU;

namespace {
using SR = BooleanSR;
using Arena = boost::container::pmr::monotonic_buffer_resource;

// Schema Definitions
// nextSiblingAnc(StartCtr, StartN, NextCtr, NextN)
using NextSiblingAncSchema = RelationSchema<decltype("NextSiblingAnc"_s), SR,
                                            std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>;
// hasNextSibling(StartCtr, StartN)
using HasNextSiblingSchema =
    RelationSchema<decltype("HasNextSibling"_s), SR, std::tuple<uint32_t, uint32_t>>;
// insert(StartCtr, StartN, ParentCtr, ParentN)
using InsertSchema =
    RelationSchema<decltype("Insert"_s), SR, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>;

using NegationBlueprint = Database<NextSiblingAncSchema, HasNextSiblingSchema, InsertSchema>;

// Variables (Types for Template Arguments)
using VarStartCtr = Var<decltype("StartCtr"_s)>;
using VarStartN = Var<decltype("StartN"_s)>;
using VarNextCtr = Var<decltype("NextCtr"_s)>;
using VarNextN = Var<decltype("NextN"_s)>;
using VarParentCtr = Var<decltype("ParentCtr"_s)>;
using VarParentN = Var<decltype("ParentN"_s)>;
using VarCnt = Var<decltype("Cnt"_s)>;

// Relation accessors
constexpr auto nextSiblingAnc = rel<NextSiblingAncSchema>;
constexpr auto hasNextSibling = rel<HasNextSiblingSchema>;
constexpr auto insert = rel<InsertSchema>;

// Data Generation
void generate_data(SemiNaiveDatabase<NegationBlueprint>& db, int num_chains, int chain_length) {
  for (int i = 0; i < num_chains; ++i) {
    // Create a chain of insertions
    // insert(node, i, parent, i)
    // Root is (0, i)
    // Let's make a straight line ancestry: (j, i) -> (j-1, i)
    for (int j = 1; j < chain_length; ++j) {
      // insert(StartCtr, StartN, ParentCtr, ParentN)
      // insert(j, i, j-1, i)
      add_fact<InsertSchema>(db, BooleanSR::one(), (uint32_t)j, (uint32_t)i, (uint32_t)(j - 1),
                             (uint32_t)i);
    }

    // Populate HasNextSibling selectively
    // Mark even nodes as having sibling
    for (int j = 0; j < chain_length; ++j) {
      if (j % 2 == 0) {
        add_fact<HasNextSiblingSchema>(db, BooleanSR::one(), (uint32_t)j, (uint32_t)i);
      }
    }
  }
}

// Model the rule:
// nextSiblingAnc(StartCtr, StartN, NextCtr, NextN) :-
//    nextSiblingAnc(ParentCtr, ParentN, NextCtr, NextN),
//    !hasNextSibling(StartCtr, StartN),
//    insert(StartCtr, StartN, ParentCtr, ParentN).

// We model !hasNextSibling as AggCount + Filter(==0)

// Cpp Expr for Filter
constexpr auto is_zero = [](size_t cnt) { return cnt == 0; };
using IsZero = CppExpr<std::tuple<VarCnt>, is_zero>;

// Aggregation Clause
using AggClauseNeg =
    AggClause<VarCnt, AggCount, HasNextSiblingSchema, FULL_VER, VarStartCtr, VarStartN>;
using FilterClauseNeg = IfClause<IsZero>;

// Rule Structure
using HeadClause =
    Clause<NextSiblingAncSchema, NEW_VER, VarStartCtr, VarStartN, VarNextCtr, VarNextN>;
using BodyClause1 =
    Clause<InsertSchema, FULL_VER, VarStartCtr, VarStartN, VarParentCtr, VarParentN>;
using BodyClause2 =
    Clause<NextSiblingAncSchema, DELTA_VER, VarParentCtr, VarParentN, VarNextCtr, VarNextN>;

// Plan
using PlanVars =
    Plan<VarParentCtr, VarParentN, VarNextCtr, VarNextN, VarStartCtr, VarStartN, VarCnt>;

using RecursiveRule =
    Rule<std::tuple<HeadClause>,
         std::tuple<BodyClause2, BodyClause1, AggClauseNeg, FilterClauseNeg>, PlanVars>;

using FixpointProg = Fixpoint<RecursiveRule>;

BOOST_AUTO_TEST_CASE(test_gpu_negation_benchmark) {
  init_cuda();
  Arena arena;
  SemiNaiveDatabase<NegationBlueprint> db(&arena, &arena, &arena);

  // Data Gen
  int num_chains = 1;
  int chain_length = 20;
  generate_data(db, num_chains, chain_length);

  // Seed base facts for nextSiblingAnc
  // Roots are (0, i). Let's say they reach themselves (or some sink).
  for (int i = 0; i < num_chains; ++i) {
    add_fact<NextSiblingAncSchema>(db, BooleanSR::one(), (uint32_t)0, (uint32_t)i, (uint32_t)999,
                                   (uint32_t)999);
  }

  // CPU Execution (Reference)

  SemiNaiveDatabase<NegationBlueprint> db_cpu(&arena, &arena, &arena);
  generate_data(db_cpu, num_chains, chain_length);
  for (int i = 0; i < num_chains; ++i) {
    add_fact<NextSiblingAncSchema>(db_cpu, BooleanSR::one(), (uint32_t)0, (uint32_t)i,
                                   (uint32_t)999, (uint32_t)999);
  }

  // 1. CPU Run
  load_full_to_delta<NextSiblingAncSchema>(db_cpu);

  execute_query<FixpointProg>(db_cpu);

  auto& res_cpu = get_relation_by_schema<NextSiblingAncSchema, FULL_VER>(db_cpu);
  size_t cpu_size = res_cpu.size();

  // 2. GPU Run
  load_full_to_delta<NextSiblingAncSchema>(db);
  execute_gpu_query<FixpointProg>(db);

  // Reflect Result
  auto& res_gpu = get_relation_by_schema<NextSiblingAncSchema, FULL_VER>(db);
  size_t gpu_size = res_gpu.size();

  BOOST_CHECK_EQUAL(gpu_size, cpu_size);
  if (gpu_size > 0) {
    BOOST_CHECK(gpu_size > num_chains);  // Should have populated up the chain
  }
}

}  // namespace
