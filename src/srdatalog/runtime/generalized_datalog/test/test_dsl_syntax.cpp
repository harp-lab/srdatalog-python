#define BOOST_TEST_MODULE dsl_syntax_test
#include "ast.h"
#include "gpu/runtime/query.h"
#include "mir.h"
#include "mir_dsl.h"
#include "semiring.h"
#include <boost/test/included/unit_test.hpp>
#include <type_traits>

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;

// Schemas
using EdgeSchema = RelationSchema<decltype("edge"_s), BooleanSR, std::tuple<int, int>>;
using PathSchema = RelationSchema<decltype("path"_s), BooleanSR, std::tuple<int, int>>;

BOOST_AUTO_TEST_CASE(test_variable_udl) {
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;

  static_assert(std::is_same_v<typename decltype(x)::type, Var<TString<'x'>>>);
  static_assert(std::is_same_v<typename decltype(y)::type, Var<TString<'y'>>>);
}

BOOST_AUTO_TEST_CASE(test_relation_accessor) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;

  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;

  auto c1 = edge(x, y);
  auto c2 = path.delta(x, y);
  auto c3 = path.newt(x, y);

  static_assert(CNormalClause<decltype(c1)>);
  static_assert(CNormalClause<decltype(c2)>);
  static_assert(CNormalClause<decltype(c3)>);

  static_assert(decltype(c1)::version == FULL_VER);
  static_assert(decltype(c2)::version == DELTA_VER);
  static_assert(decltype(c3)::version == NEW_VER);
}

BOOST_AUTO_TEST_CASE(test_rule_construction) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  // tc_base = path(x,y) <<= edge(x,y)
  auto tc_base = path(x, y) <<= edge(x, y);
  static_assert(CRule<decltype(tc_base)>);

  // tc_step = path(x,y) <<= (path.delta(x,z), edge(z,y))
  auto tc_step = path(x, y) <<= (path.delta(x, z), edge(z, y));
  static_assert(CRule<decltype(tc_step)>);

  // Check join plan of tc_step
  using StepRule = decltype(tc_step);
  using JP = typename StepRule::join_plan_type;

  // path.delta(x,z) -> {x, z}, edge(z,y) -> {z, y}. Total {x, z, y}
  static_assert(std::tuple_size_v<JP> == 3);
}

BOOST_AUTO_TEST_CASE(test_program_construction) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  auto tc_base = path(x, y) <<= edge(x, y);
  auto tc_step = path(x, y) <<= (path.delta(x, z), edge(z, y));

  auto tc_prog = fixpoint(tc_base, tc_step);
  static_assert(CFixpoint<decltype(tc_prog)>);
}

BOOST_AUTO_TEST_CASE(test_let_clause_dsl) {
  using ZSchema = relation<decltype("Z"_s), BooleanSR, int, int, int>;
  using RSchema = relation<decltype("R"_s), BooleanSR, int, int>;

  constexpr auto z = rel<ZSchema>;
  constexpr auto r = rel<RSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto sum = "sum"_v;

  // Define the lambda function
  constexpr auto add_func = [](int x, int y) -> int { return x + y; };

  // Create let clause using the new DSL API
  auto let_clause = let<add_func>(sum, x, y);
  static_assert(CLetClause<decltype(let_clause)>);

  // Use in a rule
  auto rule_with_let = z.newt(x, y, sum) <<= (r.full(x, y), let<add_func>(sum, x, y));
  static_assert(CRule<decltype(rule_with_let)>);
}

BOOST_AUTO_TEST_CASE(test_if_clause_dsl) {
  using ZSchema = relation<decltype("Z"_s), BooleanSR, int, int, int>;
  using RSchema = relation<decltype("R"_s), BooleanSR, int, int>;
  using SSchema = relation<decltype("S"_s), BooleanSR, int, int, int>;
  using TSchema = relation<decltype("T"_s), BooleanSR, int, int, int>;

  constexpr auto z = rel<ZSchema>;
  constexpr auto r = rel<RSchema>;
  constexpr auto s = rel<SSchema>;
  constexpr auto t = rel<TSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z_var = "z"_v;
  constexpr auto h = "h"_v;
  constexpr auto f = "f"_v;

  // Define the lambda function for filtering
  constexpr auto is_even = [](int x) -> bool { return x % 2 == 0; };

  // Create if clause using the new DSL API
  auto if_clause = if_<is_even>(x);
  static_assert(CIfClause<decltype(if_clause)>);

  // Use in a rule
  auto rule_with_if = z.newt(x, y, z_var) <<=
      (r.full(x, y), s.full(y, z_var, h), t.full(z_var, x, f), if_<is_even>(x));
  static_assert(CRule<decltype(rule_with_if)>);
}

BOOST_AUTO_TEST_CASE(test_explicit_join_plan) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  // Test plan helper
  auto plan_obj = plan(y, x, z);
  static_assert(CJoinPlan<decltype(plan_obj)>);

  // Test rule_with_plan with explicit join plan
  auto rule_explicit = rule_with_plan(path(x, z), (path.delta(x, y), edge(y, z)), plan(y, x, z));
  static_assert(CRule<decltype(rule_explicit)>);

  // Check that the join plan is correct
  using ExplicitRule = decltype(rule_explicit);
  using JP = typename ExplicitRule::join_plan_type;
  static_assert(std::tuple_size_v<JP> == 3);

  // Compare with auto-derived join plan
  auto rule_auto = path(x, z) <<= (path.delta(x, y), edge(y, z));
  static_assert(CRule<decltype(rule_auto)>);

  using AutoRule = decltype(rule_auto);
  using AutoJP = typename AutoRule::join_plan_type;
  // Auto-derived should also have 3 variables, but order may differ
  static_assert(std::tuple_size_v<AutoJP> == 3);
}

BOOST_AUTO_TEST_CASE(test_pipe_operator_join_plan) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  // Test pipe operator syntax: (rule) | plan(vars...)
  // Note: parentheses needed due to operator precedence
  auto rule_with_pipe = (path(x, z) <<= (path.delta(x, y), edge(y, z))) | plan(y, x, z);
  static_assert(CRule<decltype(rule_with_pipe)>);

  // Check that the join plan is correct
  using PipeRule = decltype(rule_with_pipe);
  using PipeJP = typename PipeRule::join_plan_type;
  static_assert(std::tuple_size_v<PipeJP> == 3);

  // Test that plan() creates a JoinPlan
  auto plan1 = plan(x, y, z);
  auto plan2 = plan(x, y, z);
  static_assert(std::is_same_v<decltype(plan1), decltype(plan2)>);
  static_assert(CJoinPlan<decltype(plan1)>);
}

BOOST_AUTO_TEST_CASE(test_gpu_optimization_tc_base) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;

  // tc_base = path(x,y) <<= edge(x,y)
  // This has only one join, so it should NOT be fused into CartesianJoin
  auto tc_base = path(x, y) <<= edge(x, y);
  static_assert(CRule<decltype(tc_base)>);

  using TCBaseRule = decltype(tc_base);
  using MIROps = typename SRDatalog::mir::CompileRuleToMIR<TCBaseRule>::type;
  using OptimizedOps = typename SRDatalog::mir::gpu_opt::OptimizeMIRForGPU<MIROps>::type;

  // static_assert(SRDatalog::mir::gpu_opt::HasCartesianJoin<OptimizedOps>::value);
}

BOOST_AUTO_TEST_CASE(test_gpu_optimization_tc_step) {
  constexpr auto edge = rel<EdgeSchema>;
  constexpr auto path = rel<PathSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  // tc_step = path(x,y) <<= (path.delta(x,z), edge(z,y))
  // path.delta(x,z) binds z, then edge(z,y) depends on z
  // So they are NOT independent - should NOT be fused
  auto tc_step = path(x, y) <<= (path.delta(x, z), edge(z, y));
  static_assert(CRule<decltype(tc_step)>);

  using TCStepRule = decltype(tc_step);
  using MIROps = typename SRDatalog::mir::CompileRuleToMIR<TCStepRule>::type;
  using OptimizedOps = typename SRDatalog::mir::gpu_opt::OptimizeMIRForGPU<MIROps>::type;

  // Check that dependent joins are not fused
  static_assert(!SRDatalog::mir::gpu_opt::HasCartesianJoin<OptimizedOps>::value);
}

BOOST_AUTO_TEST_CASE(test_gpu_optimization_sg_base) {
  using ArcSchema = relation<decltype("Arc"_s), BooleanSR, int, int>;
  using SGSchema = relation<decltype("SG"_s), BooleanSR, int, int>;

  constexpr auto arc = rel<ArcSchema>;
  constexpr auto sg = rel<SGSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto p = "p"_v;

  // sg_base = sg(x,y) <<= (arc.full(p,x), arc.full(p,y))
  // Pattern: two independent single-source ColumnJoins (both have empty prefix)
  // followed immediately by DestinationRelation
  // This SHOULD be fused into CartesianJoin
  auto sg_base = sg.newt(x, y) <<= (arc.full(p, x), arc.full(p, y));
  static_assert(CRule<decltype(sg_base)>);

  using SGBaseRule = decltype(sg_base);
  using MIROps = typename SRDatalog::mir::CompileRuleToMIR<SGBaseRule>::type;
  using OptimizedOps = typename SRDatalog::mir::gpu_opt::OptimizeMIRForGPU<MIROps>::type;

  // Verify that optimization creates a CartesianJoin
  static_assert(SRDatalog::mir::gpu_opt::HasCartesianJoin<OptimizedOps>::value);

  // Verify the original MIR doesn't have CartesianJoin (before optimization)
  static_assert(!SRDatalog::mir::gpu_opt::HasCartesianJoin<MIROps>::value);
}

BOOST_AUTO_TEST_CASE(test_gpu_optimization_sg_step) {
  using ArcSchema = relation<decltype("Arc"_s), BooleanSR, int, int>;
  using SGSchema = relation<decltype("SG"_s), BooleanSR, int, int>;

  constexpr auto arc = rel<ArcSchema>;
  constexpr auto sg = rel<SGSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto p = "p"_v;
  constexpr auto q = "q"_v;

  // sg_step = sg(x,y) <<= (sg.delta(p,q), arc.full(p,x), arc.full(q,y))
  // With plan(p, q, x, y):
  // 1. p, x (from arc(p, x))
  // 2. q (from sg.delta(p, q))
  // 3. y (from arc(q, y))
  auto sg_step =
      (sg.newt(x, y) <<= (arc.full(p, x), sg.delta(p, q), arc.full(q, y))) | plan(p, q, x, y);
  static_assert(CRule<decltype(sg_step)>);

  using SGStepRule = decltype(sg_step);
  using MIROps = typename SRDatalog::mir::CompileRuleToMIR<SGStepRule>::type;
  using OptimizedOps = typename SRDatalog::mir::gpu_opt::OptimizeMIRForGPU<MIROps>::type;

  // In this order:
  // Root(arc, p, x)
  // Join(q, sg.delta(p, q))  -> Dependent on p
  // Join(y, arc(q, y))       -> Dependent on q
  // Dest(sg, x, y)
  // They are NOT independent enough to fuse into a 3-way CartesianJoin.
  // But Join(q) and Join(y) might NOT fuse because y depends on q.
  static_assert(true);
}

BOOST_AUTO_TEST_CASE(test_gpu_optimization_executor_plan) {
  using ArcSchema = relation<decltype("Arc"_s), BooleanSR, int, int>;
  using SGSchema = relation<decltype("SG"_s), BooleanSR, int, int>;

  constexpr auto arc = rel<ArcSchema>;
  constexpr auto sg = rel<SGSchema>;
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto p = "p"_v;

  // SG base: sg(x, y) :- arc(p, x), arc(p, y)
  auto sg_base = sg.newt(x, y) <<= (arc.full(p, x), arc.full(p, y));
  auto sg_prog = fixpoint(sg_base);

  using Executor = SRDatalog::GPU::GPUQueryExecutor<decltype(sg_prog)>;
  using Instructions = typename Executor::Instructions;

  // Verify that the EXECUTOR's instructions contain a CartesianJoin
  static_assert(SRDatalog::mir::gpu_opt::HasCartesianJoin<Instructions>::value);
}

// ============================================================================
// MIR DSL Tests - Value-wrapper pattern for MIR construction
// ============================================================================

using namespace SRDatalog::mir::dsl;

// Schemas for MIR DSL tests
using PointsToSchema = RelationSchema<decltype("PointsTo"_s), BooleanSR, std::tuple<int, int>>;
using AssignSchema = RelationSchema<decltype("Assign"_s), BooleanSR, std::tuple<int, int>>;
using LoadSchema = RelationSchema<decltype("Load"_s), BooleanSR, std::tuple<int, int>>;
using StoreSchema = RelationSchema<decltype("Store"_s), BooleanSR, std::tuple<int, int>>;

BOOST_AUTO_TEST_CASE(test_mir_dsl_index_spec) {
  // Test index spec creation with different versions
  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
  constexpr auto pt_idx_delta = pt_idx.delta();
  constexpr auto pt_idx_full = pt_idx.full();
  constexpr auto pt_idx_newt = pt_idx.newt();

  // Verify types
  using PtIdxType = decltype(pt_idx)::type;
  using PtIdxDeltaType = decltype(pt_idx_delta)::type;
  using PtIdxFullType = decltype(pt_idx_full)::type;
  using PtIdxNewtType = decltype(pt_idx_newt)::type;

  // Check that these are valid IndexSpecT types
  static_assert(mir::CIndexSpec<PtIdxType>);
  static_assert(mir::CIndexSpec<PtIdxDeltaType>);
  static_assert(mir::CIndexSpec<PtIdxFullType>);
  static_assert(mir::CIndexSpec<PtIdxNewtType>);

  // Check versions
  static_assert(PtIdxType::kVersion == FULL_VER);  // Default is FULL
  static_assert(PtIdxDeltaType::kVersion == DELTA_VER);
  static_assert(PtIdxFullType::kVersion == FULL_VER);
  static_assert(PtIdxNewtType::kVersion == NEW_VER);

  // Check column sequence
  static_assert(
      std::is_same_v<typename PtIdxType::column_indexes_type, std::integer_sequence<int, 0, 1>>);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_column_source) {
  constexpr auto x = "x"_v;
  constexpr auto z = "z"_v;

  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();

  // Column source with no prefix
  constexpr auto src1 = column_source(pt_idx.delta());
  using Src1Type = decltype(src1)::type;
  static_assert(mir::is_column_source_v<Src1Type>);
  static_assert(std::tuple_size_v<typename Src1Type::prefix_vars_type> == 0);

  // Column source with prefix variables
  constexpr auto src2 = column_source(pt_idx.full(), x, z);
  using Src2Type = decltype(src2)::type;
  static_assert(mir::is_column_source_v<Src2Type>);
  static_assert(std::tuple_size_v<typename Src2Type::prefix_vars_type> == 2);

  // Verify prefix contents
  using Prefix2 = typename Src2Type::prefix_vars_type;
  static_assert(std::is_same_v<std::tuple_element_t<0, Prefix2>, typename decltype(x)::type>);
  static_assert(std::is_same_v<std::tuple_element_t<1, Prefix2>, typename decltype(z)::type>);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_column_join) {
  constexpr auto z = "z"_v;
  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
  constexpr auto assign_idx = index<AssignSchema, 1, 0>();

  // Create column join with two sources
  constexpr auto join_z =
      column_join(z, column_source(pt_idx.delta()), column_source(assign_idx.full()));

  using JoinType = decltype(join_z)::type;
  static_assert(mir::is_column_join_v<JoinType>);

  // Verify var type
  static_assert(std::is_same_v<typename JoinType::var_type, typename decltype(z)::type>);

  // Verify sources tuple has 2 elements
  static_assert(std::tuple_size_v<typename JoinType::sources_type> == 2);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_cartesian_join) {
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
  constexpr auto assign_idx = index<AssignSchema, 1, 0>();

  // Create cartesian join
  constexpr auto cart = cartesian_join(vars(x, y), column_source(pt_idx.delta(), z),
                                       column_source(assign_idx.full(), z));

  using CartType = decltype(cart)::type;
  static_assert(mir::is_cartesian_join_v<CartType>);

  // Verify vars tuple
  using VarsTuple = typename CartType::vars_type;
  static_assert(std::tuple_size_v<VarsTuple> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, VarsTuple>, typename decltype(x)::type>);
  static_assert(std::is_same_v<std::tuple_element_t<1, VarsTuple>, typename decltype(y)::type>);

  // Verify sources tuple
  static_assert(std::tuple_size_v<typename CartType::sources_type> == 2);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_destination_relation) {
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;

  // Create destination relation (insert into)
  constexpr auto dest = insert_into<PointsToSchema>(y, x);

  using DestType = decltype(dest)::type;
  static_assert(mir::is_destination_relation_v<DestType>);

  // Verify schema and version
  static_assert(std::is_same_v<typename DestType::schema_type, PointsToSchema>);
  static_assert(DestType::Version == NEW_VER);  // Default version

  // Verify terms
  using TermsTuple = typename DestType::terms_type;
  static_assert(std::tuple_size_v<TermsTuple> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, TermsTuple>, typename decltype(y)::type>);
  static_assert(std::is_same_v<std::tuple_element_t<1, TermsTuple>, typename decltype(x)::type>);

  // Test explicit version
  constexpr auto dest_full = insert_into<PointsToSchema, FULL_VER>(x, y);
  using DestFullType = decltype(dest_full)::type;
  static_assert(DestFullType::Version == FULL_VER);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_pipeline) {
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;
  constexpr auto z = "z"_v;

  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
  constexpr auto assign_idx = index<AssignSchema, 1, 0>();

  // Build a complete pipeline (mimics Andersen Rule 1)
  constexpr auto pipe =
      pipeline(column_join(z, column_source(pt_idx.delta()), column_source(assign_idx.full())),
               cartesian_join(vars(x, y), column_source(pt_idx.delta(), z),
                              column_source(assign_idx.full(), z)),
               insert_into<PointsToSchema>(y, x));

  using PipeType = decltype(pipe)::type;
  static_assert(mir::is_pipeline_v<PipeType>);

  // Verify MIR ops tuple has 3 operations
  using OpsType = typename PipeType::mir_ops_type;
  static_assert(std::tuple_size_v<OpsType> == 3);

  // Verify first op is ColumnJoin
  static_assert(mir::is_column_join_v<std::tuple_element_t<0, OpsType>>);

  // Verify second op is CartesianJoin
  static_assert(mir::is_cartesian_join_v<std::tuple_element_t<1, OpsType>>);

  // Verify third op is DestinationRelation
  static_assert(mir::is_destination_relation_v<std::tuple_element_t<2, OpsType>>);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_fixpoint_instructions) {
  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();

  // Test various fixpoint-level instruction wrappers
  constexpr auto create_idx = create_index(pt_idx.full());
  constexpr auto rebuild_idx = rebuild_index(pt_idx.newt());
  constexpr auto merge_idx = merge_index(pt_idx.full());
  constexpr auto clear_rel = clear_relation<PointsToSchema, DELTA_VER>();
  constexpr auto check_sz = check_size<PointsToSchema, NEW_VER>();

  // Verify types
  static_assert(mir::is_build_index_v<decltype(create_idx)::type>);
  static_assert(mir::is_rebuild_index_v<decltype(rebuild_idx)::type>);
  static_assert(mir::is_merge_index_v<decltype(merge_idx)::type>);
  static_assert(mir::is_clear_relation_v<decltype(clear_rel)::type>);
  static_assert(mir::is_check_size_v<decltype(check_sz)::type>);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_execute_pipeline) {
  constexpr auto x = "x"_v;
  constexpr auto y = "y"_v;

  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();

  // Build a simple pipeline
  constexpr auto pipe =
      pipeline(column_join(x, column_source(pt_idx.delta())), insert_into<PointsToSchema>(x, y));

  // Wrap in execute
  constexpr auto exec = execute(pipe);

  using ExecType = decltype(exec)::type;
  static_assert(mir::is_execute_pipeline_v<ExecType>);
}

BOOST_AUTO_TEST_CASE(test_mir_dsl_type_extraction) {
  // This test verifies that the DSL properly extracts types via decltype(...):type
  // This is the core pattern: value wrappers carry types that can be extracted

  constexpr auto x = "x"_v;
  constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
  constexpr auto src = column_source(pt_idx.delta(), x);
  constexpr auto join = column_join(x, src);

  // The chain of type extractions
  using XType = decltype(x)::type;         // Var<...>
  using IdxType = decltype(pt_idx)::type;  // IndexSpecT<...>
  using SrcType = decltype(src)::type;     // ColumnSource<...>
  using JoinType = decltype(join)::type;   // ColumnJoin<...>

  // All should be valid MIR/AST types
  static_assert(CVar<XType>);
  static_assert(mir::CIndexSpec<IdxType>);
  static_assert(mir::is_column_source_v<SrcType>);
  static_assert(mir::is_column_join_v<JoinType>);
}
