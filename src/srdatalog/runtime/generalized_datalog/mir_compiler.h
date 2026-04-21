/**
 * @file mir_compiler.h
 * @brief AST to MIR compilation logic.
 *
 * @details This file contains the logic for compiling high-level AST structures
 * (Rules, Clauses) into MIR operations and FixpointPlans. It handles the
 * complexities of variable binding, index selection, and operation scheduling.
 */

#pragma once

#include "mir_def.h"
#include "mir_plan.h"

namespace SRDatalog {
using namespace AST;
namespace mir {

// Forward declarations for compilation helpers
template <CVar Var, tmp::CTuple BodyClauses>
struct FindLetClauses;

template <CVar Var, tmp::CTuple BodyClauses>
struct FindNormalClausesWithVar;

template <tmp::CTuple BodyClauses>
struct FindIfClauses;

template <typename Clause, CVar VarToBind, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct ClauseToColumnSource_impl;

template <CVar VarToBind, tmp::CTuple NormalClauses, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct MapClausesToColumnSources;

/**
 * @brief Computes the IndexSpec for a single normal clause.
 *
 * @details Given a clause and a global join plan, determines which index
 * should be used to access the clause's relation efficiently. The index is
 * chosen based on which variables from the join plan appear in the clause,
 * ensuring optimal access patterns for WCOJ algorithms.
 *
 * @tparam Clause The normal clause to analyze (must satisfy CNormalClause)
 * @tparam GlobalJoinPlan A std::tuple of Var types specifying the join order
 *
 * @note Only considers variables that appear in both the clause and the join
 *       plan, ensuring the index supports the join execution order.
 */
template <typename Clause>
struct GetClauseSchema;
template <CNormalClause C>
struct GetClauseSchema<C> {
  using type = typename GetNormalClauseSchema<C>::type;
};
template <CAggClause C>
struct GetClauseSchema<C> {
  using type = typename C::relation_schema;
};

template <typename Clause>
struct GetClauseTerms;
template <CNormalClause C>
struct GetClauseTerms<C> {
  using type = typename GetNormalClauseTerms<C>::type;
};
template <CAggClause C>
struct GetClauseTerms<C> {
  using type = typename C::terms_type;
};

template <typename Clause>
struct GetClauseVersion;
template <CNormalClause C>
struct GetClauseVersion<C> {
  static constexpr std::size_t value = GetNormalClauseVersion<C>::value;
};
template <CAggClause C>
struct GetClauseVersion<C> {
  static constexpr std::size_t value = C::version;
};

/**
 * @brief Computes the IndexSpec for a normal clause.
 */
template <typename Clause, tmp::CTuple GlobalJoinPlan>
struct GetIndexForClause;

template <CNormalClause Clause, tmp::CTuple GlobalJoinPlan>
struct GetIndexForClause<Clause, GlobalJoinPlan> {
  // For normal clauses, usage depends on join plan intersection
  using ClauseTerms = typename GetClauseTerms<Clause>::type;
  using ClauseVars = typename GetVarsFromTerms<ClauseTerms>::type;
  using RelevantPlan = tmp::TupleSetIntersection<GlobalJoinPlan, ClauseVars>;

  using ColumnIndexes = typename MapVarsToColumnIndexes<ClauseTerms, RelevantPlan>::type;
  using Schema = typename GetClauseSchema<Clause>::type;
  static constexpr std::size_t ClauseVer = GetClauseVersion<Clause>::value;

  using type = IndexSpecT<Schema, ColumnIndexes, ClauseVer>;
};

template <CAggClause Clause, tmp::CTuple GlobalJoinPlan>
struct GetIndexForClause<Clause, GlobalJoinPlan> {
  // For aggregation clauses, we need an index that covers ALL terms (Prefix +
  // AggVar) regardless of whether they appear in the join plan. The prefix
  // variables (bound) come first, followed by the free variable (unbound).
  // But since we want to cover all vars to iterate the free vars given the
  // prefix, we use all clause vars. Wait, ordering matters for efficiency. We
  // should order index columns based on: [PrefixVars in JoinPlan order] +
  // [FreeVars]

  // Use Clause::args_type to get only the Relation arguments (excluding
  // ResultVar) This lines up with the Relation Schema columns.
  using ClauseTerms = typename Clause::args_type;
  using ClauseVars = typename GetVarsFromTerms<ClauseTerms>::type;

  // Prefix vars are those in the join plan
  using PrefixVars = typename tmp::TupleSetIntersection<GlobalJoinPlan, ClauseVars>;
  // Free vars are the rest
  using FreeVars = typename tmp::TupleSetDifference<ClauseVars, PrefixVars>;

  // Desired index order: Prefix... , Free...
  using AccessOrder = typename tmp::TupleConcat<PrefixVars, FreeVars>;

  using ColumnIndexes = typename MapVarsToColumnIndexes<ClauseTerms, AccessOrder>::type;
  using Schema = typename GetClauseSchema<Clause>::type;
  static constexpr std::size_t ClauseVer = GetClauseVersion<Clause>::value;

  using type = IndexSpecT<Schema, ColumnIndexes, ClauseVer>;
};

/**
 * @brief Maps all body clauses to their required IndexSpecs.
 *
 * @details Filters body clauses to only normal clauses (excluding LetClauses)
 * and computes the IndexSpec for each. This produces the complete set of
 * indexes needed to execute the rule body efficiently.
 *
 * @tparam BodyClauses A std::tuple of clause types (normal and let)
 * @tparam GlobalJoinPlan A std::tuple of Var types specifying the join order
 */
template <typename Clause>
struct is_clause_needing_index {
  static constexpr bool value = is_normal_clause_v<Clause> || is_agg_clause_v<Clause>;
};

template <tmp::CTuple BodyClauses, tmp::CTuple GlobalJoinPlan>
struct MapClausesToSpecs_Impl {
  using ClausesToIndex = mp::mp_copy_if<BodyClauses, is_clause_needing_index>;

  template <typename Clause>
  using ToSpec = typename GetIndexForClause<Clause, GlobalJoinPlan>::type;
  using type = mp::mp_transform<ToSpec, ClausesToIndex>;
};

/**
 * @brief Computes the set of required IndexSpec ADTs for a rule.
 *
 * @details Analyzes a rule and determines all indexes needed to execute it
 * efficiently. Uses the join plan specified in the rule to determine optimal
 * index access patterns. The result is deduplicated to avoid redundant
 * indexes.
 *
 * @tparam rule The rule to analyze (must satisfy CRule)
 *
 * @note Currently uses the naive (first-appearance) variable order from the
 *       join plan. Future versions may support more sophisticated join
 * ordering.
 */
template <CRule rule>
struct ComputeNaiveIndices;

template <tmp::CTuple Head, tmp::CTuple Body, CJoinPlan JoinPlan>
struct ComputeNaiveIndices<Rule<Head, Body, JoinPlan>> {
  // Pass 1: Get the global join plan (appearance order)
  // TODO: consider adding better TMP to handle this
  using GlobalJoinPlan = typename JoinPlan::vars_type;
  // Pass 2: Map all body clauses to IndexSpecs using that plan
  using AllSpecs = typename MapClausesToSpecs_Impl<Body, GlobalJoinPlan>::type;

  /**
   * @brief The final, unique list of required IndexSpec ADTs.
   *
   * @details Contains one IndexSpec for each unique (schema, column-indexes,
   *          version) combination needed by the rule.
   */
  using type = typename tmp::TupleUnique<AllSpecs>;
};

/**
 * @brief Helper to filter a list of body clauses for a specific LetClause.
 *
 * @details Type predicate to check if a clause is a LetClause that binds a
 * specific variable. Used during MIR compilation to find let operations for
 * a variable being bound.
 *
 * @tparam Var The variable to check for
 * @tparam Clause The clause type to check
 */
template <CVar Var, typename Clause>
struct is_let_clause_for_var_impl : std::false_type {};

template <CVar Var, CCppExpr Expr>
struct is_let_clause_for_var_impl<Var, LetClause<Var, Expr>> : std::true_type {};

/**
 * @brief Finds all LetClauses in a body that bind a specific variable.
 *
 * @details Filters the body clauses to find all LetClauses that introduce
 * the specified variable. Used during MIR compilation to generate let
 * operations.
 *
 * @tparam Var The variable to find LetClauses for
 * @tparam BodyClauses A std::tuple of clause types
 */
template <CVar Var, tmp::CTuple BodyClauses>
struct FindLetClauses {
  template <typename Clause>
  using predicate = is_let_clause_for_var_impl<Var, Clause>;
  using type = mp::mp_copy_if<BodyClauses, predicate>;
};

/**
 * @brief Finds all normal clauses in a body that reference a specific
 * variable.
 *
 * @details Filters the body clauses to find all normal clauses (relation
 * predicates) that use the specified variable. Used during MIR compilation
 * to generate ColumnSource operations.
 *
 * @tparam Var The variable to find clauses for
 * @tparam BodyClauses A std::tuple of clause types
 */
template <CVar Var, tmp::CTuple BodyClauses>
struct FindNormalClausesWithVar {
  template <typename Clause>
  using predicate = mp::mp_eval_if_not<is_normal_clause<Clause>, mp::mp_bool<false>, tmp::Contains,
                                       Var, typename GetClauseVars<Clause>::type>;
  using type = mp::mp_copy_if<BodyClauses, predicate>;
};

// create a concept for all other clauses
/// @note **C++20 feature**: Uses concept disjunction (||) to combine multiple
///       clause concepts
template <typename Clause>
concept COtherClause = CLetClause<Clause> || CIfClause<Clause>;

template <CVar Var, COtherClause Head, typename... Tail>
struct FindNormalClausesWithVar<Var, std::tuple<Head, Tail...>> {
  // Skip all other clauses - they're not normal clauses and don't generate
  // ColumnSources
  using type = typename FindNormalClausesWithVar<Var, std::tuple<Tail...>>::type;
};

/**
 * @brief Finds all IfClause operations in a body.
 *
 * @details Filters the body clauses to find all IfClause operations.
 * Used during MIR compilation to generate IfClause MIR operations.
 * IfClause operations should be added after all variable bindings are
 * complete.
 *
 * @tparam BodyClauses A std::tuple of clause types
 */
template <tmp::CTuple BodyClauses>
struct FindIfClauses {
  template <typename Clause>
  using predicate = SRDatalog::AST::is_if_clause<Clause>;
  using type = mp::mp_copy_if<BodyClauses, predicate>;
};

/**
 * @brief Converts a normal clause to a ColumnSource MIR operation.
 *
 * @details Analyzes a clause in the context of bound variables and join plan
 * to determine how it should be accessed. Produces a ColumnSource that
 * specifies which index to use and which variables are already bound
 * (prefix).
 *
 * @tparam Clause The normal clause to convert
 * @tparam VarToBind The variable currently being bound (for context)
 * @tparam BoundVars A std::tuple of variables already bound
 * @tparam JoinPlan The global join plan (variable order)
 */
template <typename Clause, CVar VarToBind, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct ClauseToColumnSource_impl {
  using ClauseVars = typename GetClauseVars<Clause>::type;
  using ClauseTerms = typename GetNormalClauseTerms<Clause>::type;
  using Schema = typename GetNormalClauseSchema<Clause>::type;

  // 1. Find the prefix vars (those *already* bound)
  using PrefixVars = typename tmp::TupleSetIntersection<BoundVars, ClauseVars>;

  // 2. Find the free vars
  using FreeVars = typename tmp::TupleSetDifference<ClauseVars, PrefixVars>;

  // 3. The access order for this *one* clause
  // using AccessOrder = tmp::TupleConcat<PrefixVars, FreeVars>;
  // TODO: the access order is based on clause variable order in join plan
  using AccessOrder = typename tmp::TupleSetIntersection<JoinPlan, ClauseVars>;

  // 4. The column indexes for this access order
  using ColumnIndexes = typename MapVarsToColumnIndexes<ClauseTerms, AccessOrder>::type;

  // 5. Extract version from clause and create IndexSpec ADT
  static constexpr std::size_t ClauseVer = GetNormalClauseVersion<Clause>::value;
  using Index = IndexSpecT<Schema, ColumnIndexes, ClauseVer>;

  // 6. The new ColumnSource MIR op (version is now in IndexSpec)
  using type = mir::ColumnSource<Index, PrefixVars>;
};

/**
 * @brief Maps multiple normal clauses to ColumnSource MIR operations.
 *
 * @details Converts a list of normal clauses into their corresponding
 * ColumnSource operations, all in the context of the same bound variables
 * and join plan.
 *
 * @tparam VarToBind The variable currently being bound
 * @tparam NormalClauses A std::tuple of normal clause types
 * @tparam BoundVars A std::tuple of variables already bound
 * @tparam JoinPlan The global join plan
 */
template <CVar VarToBind, tmp::CTuple NormalClauses, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct MapClausesToColumnSources {
  template <typename Clause>
  using ToColumnSource =
      typename ClauseToColumnSource_impl<Clause, VarToBind, BoundVars, JoinPlan>::type;

  using type = mp::mp_transform<ToColumnSource, NormalClauses>;
};

/**
 * @brief Helper to filter a list of body clauses for a specific AggClause.
 */
template <CVar Var, typename Clause>
struct is_agg_clause_for_var_impl : std::false_type {};

template <CVar Var, typename AggFunc, typename RelSchema, std::size_t Ver, typename... Terms>
struct is_agg_clause_for_var_impl<Var, AggClause<Var, AggFunc, RelSchema, Ver, Terms...>>
    : std::true_type {};

/**
 * @brief Finds all AggClauses in a body that bind a specific variable.
 */
template <CVar Var, tmp::CTuple BodyClauses>
struct FindAggClauses {
  template <typename Clause>
  using predicate = is_agg_clause_for_var_impl<Var, Clause>;
  using type = mp::mp_copy_if<BodyClauses, predicate>;
};

// [REMOVED] Incomplete TransformToCartesian and FindCartesianMatch logic
// The TransformToCartesian logic requires GetOutputVars and specific Cartesian detection strategies
// that were not fully implemented. Reverting to avoid build breakages.
// The user's original request was likely just to fix standard template arguments.

template <typename Clause, CVar VarToBind, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct ClauseToAggregate_impl {
  using Schema = typename Clause::relation_schema;
  using ClauseTerms = typename Clause::args_type;
  // Get variables from the terms tuple
  using ClauseVars = typename GetVarsFromTerms<ClauseTerms>::type;

  // 1. Prefix vars (already bound)
  using PrefixVars = typename tmp::TupleSetIntersection<BoundVars, ClauseVars>;
  // 2. The unbound var (aggregation target)
  using FreeVars = typename tmp::TupleSetDifference<ClauseVars, PrefixVars>;

  // 3. Access Order: Prefix + Free
  using AccessOrder = typename tmp::TupleConcat<PrefixVars, FreeVars>;

  // 4. Column Indexes
  using ColumnIndexes = typename MapVarsToColumnIndexes<ClauseTerms, AccessOrder>::type;

  static constexpr std::size_t ClauseVer = Clause::version;
  using Index = IndexSpecT<Schema, ColumnIndexes, ClauseVer>;

  // Use dummy handles for initial AST->MIR mapping; explicit handles assigned later in compilation
  using type = mir::Aggregate<VarToBind, Index, PrefixVars, typename Clause::agg_func, 0, void>;
};

template <CVar VarToBind, tmp::CTuple AggClauses, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct MapAggClausesToAggregate {
  template <typename Clause>
  using ToAgg = typename ClauseToAggregate_impl<Clause, VarToBind, BoundVars, JoinPlan>::type;
  using type = mp::mp_transform<ToAgg, AggClauses>;
};

/**
 * @brief Finds all NegClause operations in a body.
 */
template <tmp::CTuple BodyClauses>
struct FindNegClauses {
  template <typename Clause>
  using predicate = SRDatalog::AST::is_neg_clause<Clause>;
  using type = mp::mp_copy_if<BodyClauses, predicate>;
};

/**
 * @brief Converts a negation clause to a Negation MIR operation.
 */
template <typename Clause, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct ClauseToNegation_impl {
  using Schema = typename Clause::relation_schema;
  using ClauseTerms = typename Clause::terms_type;
  using ClauseVars = typename GetVarsFromTerms<ClauseTerms>::type;

  // Prefix vars: all variables in the negation (must be previously bound)
  using PrefixVars = typename tmp::TupleSetIntersection<BoundVars, ClauseVars>;

  // Access order: based on global join plan
  using AccessOrder = typename tmp::TupleSetIntersection<JoinPlan, ClauseVars>;

  // Column indexes
  using ColumnIndexes = typename MapVarsToColumnIndexes<ClauseTerms, AccessOrder>::type;

  static constexpr std::size_t ClauseVer = Clause::version;
  using Index = IndexSpecT<Schema, ColumnIndexes, ClauseVer>;

  // Use dummy handles for initial AST->MIR mapping; explicit handles assigned later in compilation
  using type = mir::Negation<Index, PrefixVars, 0, void>;
};

/**
 * @brief Maps multiple NegClause operations to Negation MIR operations.
 */
template <tmp::CTuple NegClauses, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct MapNegClausesToNegations {
  template <typename Clause>
  using ToNeg = typename ClauseToNegation_impl<Clause, BoundVars, JoinPlan>::type;
  using type = mp::mp_transform<ToNeg, NegClauses>;
};

/**
 * @brief Helper to extract version from ColumnSource operations.
 *
 * @details Extracts the version number from the first ColumnSource in a
 * tuple. All ColumnSources in a join should have the same version.
 *
 * @tparam Sources A std::tuple of ColumnSource types
 */
template <tmp::CTuple Sources>
struct GetVersionFromSources;
template <CIndexSpec Index, tmp::CTuple BVars>
struct GetVersionFromSources<std::tuple<mir::ColumnSource<Index, BVars>>> {
  static constexpr std::size_t value = Index::kVersion;
};
template <CIndexSpec Index, tmp::CTuple BVars, typename... Rest>
struct GetVersionFromSources<std::tuple<mir::ColumnSource<Index, BVars>, Rest...>> {
  static constexpr std::size_t value = Index::kVersion;
};

/**
 * @brief State structure for the fold operation over join plan variables.
 *
 * @details Tracks which variables have been bound and which MIR operations
 * have been accumulated so far during rule body compilation.
 *
 * @tparam BoundVars A std::tuple of variables that have been bound
 * @tparam AccumulatedOps A std::tuple of MIR operations generated so far
 */
// Helper to get active handles for a specific variable from the state map
template <typename Var, typename ActiveHandlesMap>
struct GetHandlesForVar {
  static constexpr auto get() {
    if constexpr (hana::contains(ActiveHandlesMap{}, hana::type_c<Var>)) {
      return hana::at_key(ActiveHandlesMap{}, hana::type_c<Var>);
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }
  using type = decltype(get());
};

// Helper to build VarToHandlesMap for an operation
template <typename VarsTuple, typename ActiveHandlesMap>
struct BuildVarToHandlesMap;

template <typename... Vars, typename ActiveHandlesMap>
struct BuildVarToHandlesMap<std::tuple<Vars...>, ActiveHandlesMap> {
  static constexpr auto make() {
    return hana::make_map(hana::make_pair(
        hana::type_c<Vars>, typename GetHandlesForVar<Vars, ActiveHandlesMap>::type{})...);
  }
  using type = decltype(make());
};

// State structure for the fold operation over join plan variables.
// Now tracks active handles and handle counter.
template <tmp::CTuple BoundVars, tmp::CTuple AccumulatedOps, typename ActiveHandlesMap,
          std::size_t HandleCounter>
struct FoldState {
  using bound_vars = BoundVars;
  using ops = AccumulatedOps;
  using active_handles = ActiveHandlesMap;
  static constexpr std::size_t handle_counter = HandleCounter;
};

// Forward declaration
template <typename State, typename VarToBind, tmp::CTuple AllJoinPlan, tmp::CTuple AllBodyClauses,
          tmp::CTuple Indices>
struct ProcessOneVariable_impl;

template <tmp::CTuple AllJoinPlan, tmp::CTuple AllBodyClauses, tmp::CTuple Indices>
struct ProcessOneVariable {
  template <typename State, typename VarToBind>
  using fn = typename ProcessOneVariable_impl<State, VarToBind, AllJoinPlan, AllBodyClauses,
                                              Indices>::type;
};

template <typename State, typename VarToBind, tmp::CTuple AllJoinPlan, tmp::CTuple AllBodyClauses,
          tmp::CTuple Indices>
struct ProcessOneVariable_impl {
  // Extract state
  using BoundVars = typename State::bound_vars;
  using AccumulatedOps = typename State::ops;
  using ActiveHandlesMap = typename State::active_handles;
  static constexpr std::size_t CurrentHandleStart = State::handle_counter;

  // 1. Let Clauses (HandleStart=0/dummy, no map needed really as they don't consume handles in
  // this sense)
  using LetOps = typename FindLetClauses<VarToBind, AllBodyClauses>::type;

  // 2. Aggregate operations
  using AggClauses = typename FindAggClauses<VarToBind, AllBodyClauses>::type;
  // Map Agg ops (need to inject handle info)
  // For now, AggOps usually consume 1 handle.
  // We need to a custom mapper that injects HandleStart and VarToHandlesMap
  template <typename Op>
  struct EnrichAggOp {
    using VarsNeeded = typename Op::prefix_vars_type;  // Aggregates depend on prefix vars
    using Map = typename BuildVarToHandlesMap<VarsNeeded, ActiveHandlesMap>::type;
    using type = mir::Aggregate<typename Op::var_type, typename Op::index_spec_type,
                                typename Op::prefix_vars_type, typename Op::agg_func_type,
                                CurrentHandleStart, Map>;
  };
  // Base AggOps from previous helper
  using BaseAggOps =
      typename MapAggClausesToAggregate<VarToBind, AggClauses, BoundVars, AllJoinPlan>::type;
  using AggOps = mp::mp_transform_q<mp::mp_quote_trait<EnrichAggOp>, BaseAggOps>;

  static constexpr std::size_t AggHandlesConsumed = mp::mp_size<AggOps>::value;

  // 3. Join operations (ColumnJoin)
  using NormalClauses = typename FindNormalClausesWithVar<VarToBind, AllBodyClauses>::type;
  using ColumnSources =
      typename MapClausesToColumnSources<VarToBind, NormalClauses, BoundVars, AllJoinPlan>::type;

  static constexpr bool has_join = !mp::mp_empty<ColumnSources>::value;
  static constexpr std::size_t JoinHandlesConsumed =
      has_join ? mp::mp_size<ColumnSources>::value : 0;

  // Create Join Op with explicit handles
  template <bool Enable>
  struct CreateJoin {
    // Collect all prefix vars from sources
    template <typename S>
    using GetPrefix = typename S::prefix_vars_type;
    using SourcesList = mp::mp_rename<ColumnSources, mp::mp_list>;

    // Convert a tuple<T...> to mp_list<T...>
    template <typename T>
    using ToMpList = mp::mp_rename<T, mp::mp_list>;

    using AllPrefixVars = mp::mp_unique<
        mp::mp_flatten<mp::mp_transform<ToMpList, mp::mp_transform<GetPrefix, SourcesList>>>>;
    using AllPrefixVarsTuple = mp::mp_rename<AllPrefixVars, std::tuple>;

    // Build LOCAL handles map from THIS operation's sources only
    // For each prefix var, find which sources (at offset CurrentHandleStart + AggHandlesConsumed +
    // i) have that var in their prefix
    static constexpr std::size_t JoinHandleStart = CurrentHandleStart + AggHandlesConsumed;

    template <typename Var, std::size_t... Is>
    static constexpr auto get_handles_for_var_impl(std::index_sequence<Is...>) {
      constexpr auto check_source = []<std::size_t I>() {
        using Source = mp::mp_at_c<ColumnSources, I>;
        using Prefix = typename Source::prefix_vars_type;
        return mp::mp_contains<Prefix, Var>::value;
      };

      return []<bool... Contains>(std::integer_sequence<bool, Contains...>) {
        constexpr std::size_t count = (std::size_t{Contains} + ... + 0);
        std::array<std::size_t, count> result{};
        std::size_t idx = 0;
        ((Contains ? (result[idx++] = JoinHandleStart + Is, void()) : void()), ...);
        return result;
      }(std::integer_sequence<bool, check_source.template operator()<Is>()...>{});
    }

    template <typename Var>
    static constexpr auto get_handles_for_var() {
      constexpr auto arr = get_handles_for_var_impl<Var>(
          std::make_index_sequence<mp::mp_size<ColumnSources>::value>{});
      return [&]<std::size_t... Js>(std::index_sequence<Js...>) {
        return std::integer_sequence<std::size_t, arr[Js]...>{};
      }(std::make_index_sequence<arr.size()>{});
    }

    template <typename... Vars>
    static constexpr auto build_local_map(std::tuple<Vars...>) {
      return hana::make_map(hana::make_pair(hana::type_c<Vars>, get_handles_for_var<Vars>())...);
    }

    using Map = decltype(build_local_map(AllPrefixVarsTuple{}));

    using type = std::tuple<
        mir::ColumnJoin<VarToBind, ColumnSources, mir::DefaultJoinStrategy, JoinHandleStart, Map>>;
  };

  using MaybeJoin =
      typename std::conditional_t<has_join, CreateJoin<true>, mp::mp_identity<std::tuple<>>>::type;

  static constexpr std::size_t TotalHandlesConsumed = AggHandlesConsumed + JoinHandlesConsumed;

  // Update ActiveHandlesMap
  // The new variable `VarToBind` is bound by these new handles.
  // For ColumnJoin, the handles are [Start, Start + NumSources).
  // For Agg, [Start, Start + 1).
  // We need to map VarToBind to these new handle indices.

  template <std::size_t Start, std::size_t Count>
  struct MakeHandleSeq {
    using type = std::integer_sequence<std::size_t>;  // Placeholder if count 0
  };
  // Generate integer sequence directly: [0, 1, 2, ..., TotalHandlesConsumed-1]
  using NewHandleIndices = std::make_integer_sequence<std::size_t, TotalHandlesConsumed>;

  // Offset them by CurrentHandleStart
  template <typename Seq>
  struct OffsetSeq;
  template <std::size_t... Is>
  struct OffsetSeq<std::integer_sequence<std::size_t, Is...>> {
    using type = std::integer_sequence<std::size_t, (CurrentHandleStart + Is)...>;
  };
  using NewHandles = typename OffsetSeq<NewHandleIndices>::type;

  // Add new mapping: ActiveHandlesMap + { VarToBind -> NewHandles }
  using NewActiveHandlesMap = decltype(hana::insert(
      ActiveHandlesMap{}, hana::make_pair(hana::type_c<VarToBind>, NewHandles{})));

  // Return new state
  using type = FoldState<mp::mp_unique<mp::mp_push_back<BoundVars, VarToBind>>,
                         mp::mp_append<AccumulatedOps, LetOps, AggOps, MaybeJoin>,
                         NewActiveHandlesMap, CurrentHandleStart + TotalHandlesConsumed>;
};

template <tmp::CTuple BodyClauses, tmp::CTuple GlobalJoinPlan, tmp::CTuple Indices>
struct CompileBodiesToMIR {
  // Initial state: No bound vars, no ops, empty active handles map, handle counter 0
  using InitialState = FoldState<std::tuple<>, std::tuple<>, decltype(hana::make_map()), 0>;

  using FinalState = mp::mp_fold_q<GlobalJoinPlan, InitialState,
                                   ProcessOneVariable<GlobalJoinPlan, BodyClauses, Indices>>;

  // Find all IfClause operations in the body
  using IfClauses = typename FindIfClauses<BodyClauses>::type;

  // Find all NegClause operations in the body
  using NegClauses = typename FindNegClauses<BodyClauses>::type;

  // Convert NegClause to MIR Negation with handles
  // Negation needs variables to be bound already.

  // We need to fold over NegClauses to assign handles sequentially
  // We need to fold over NegClauses to assign handles sequentially
  struct ProcessNegation {
    template <typename State, typename Clause>
    struct apply {
      using Ops = typename State::ops;
      static constexpr std::size_t Counter = State::handle_counter;

      using Base =
          typename ClauseToNegation_impl<Clause, typename State::bound_vars, GlobalJoinPlan>::type;
      using PrefixVars = typename Base::prefix_vars_type;
      using Map = typename BuildVarToHandlesMap<PrefixVars, typename State::active_handles>::type;

      using NewOp = mir::Negation<typename Base::index_spec_type, PrefixVars, Counter, Map>;

      using type = FoldState<typename State::bound_vars, mp::mp_push_back<Ops, NewOp>,
                             typename State::active_handles, Counter + 1>;
    };
    template <typename State, typename Clause>
    using fn = typename apply<State, Clause>::type;
  };

  using NegationState = mp::mp_fold_q<NegClauses, FinalState, ProcessNegation>;

  using IfClauseOps = IfClauses;
  using NegOps = typename NegationState::ops;

  using type = mp::mp_append<typename NegationState::ops, IfClauseOps>;
  using final_active_handles = typename NegationState::active_handles;
};

/**
 * @brief Compiles the rule head into MIR DestinationRelation operations.
 *
 * @details Converts each head clause into a DestinationRelation operation
 * that writes facts to the corresponding relation. The version is extracted
 * from each clause (typically NEW_VER for semi-naive evaluation).
 *
 * @tparam HeadClauses A std::tuple of normal clause types
 * @tparam ActiveHandlesMap The map of variables to handle positions (from body)
 */
template <tmp::CTuple HeadClauses, typename ActiveHandlesMap = void,
          typename AllIndexSpecs = std::tuple<>>
struct CompileHeadsToMIR {
  template <typename Clause>
  using ToDest = mir::DestinationRelation<
      typename GetClauseSchema<Clause>::type, typename GetClauseTerms<Clause>::type,
      GetClauseVersion<Clause>::value,
      typename GetCanonicalIndexSpecForSchema<typename GetClauseSchema<Clause>::type,
                                              AllIndexSpecs>::type,
      ActiveHandlesMap>;

  // Note: We don't need to compute index specs for head clauses as they are just simple writes?
  // Actually, standard DestinationRelation uses empty index spec by default.
  // But if we want to enforce an index for the write? Usually hash index.
  // The DestinationRelation struct has a default IndexSpec.
  // We should prob override it if we had info, but default is fine.

  // Wait, current DestinationRelation usage in code assumes 4 args.
  // We added 5th arg VHM.
  // ClauseToDest should use the new 5th arg.
  // But wait, the existing code didn't use ClauseToDest struct?
  // It used mp_transform.

  using type = mp::mp_transform<ToDest, HeadClauses>;
};

// Forward declaration for ComputeHandlePositions (defined later in this file)
template <typename OpsTuple, std::size_t StartOffset>
struct ComputeHandlePositions;

/**
 * @brief Compiles a complete Rule ADT into MIR operations.
 *
 * @details Transforms a high-level logical rule into a sequence of MIR
 * operations suitable for runtime execution. The compilation process:
 * 1. Computes required indexes
 * 2. Compiles body clauses into ColumnSource/ColumnJoin operations
 * 3. Compiles head clauses into DestinationRelation operations
 * 4. Combines all operations into a single tuple
 * 5. Enhances operations with precomputed handle positions (fast compilation)
 *
 * @tparam Rule The rule to compile (must satisfy CRule)
 *
 * @note Version information is extracted from clauses in both body and head.
 *       The body typically uses FULL_VER or DELTA_VER, while the head uses
 *       NEW_VER for semi-naive evaluation.
 */
template <CRule Rule>
struct CompileRuleToMIR;

template <tmp::CTuple Head, tmp::CTuple Body, CJoinPlan JoinPlan>
struct CompileRuleToMIR<Rule<Head, Body, JoinPlan>> {
  using IdxRes = ComputeNaiveIndices<Rule<Head, Body, JoinPlan>>;
  using Indices = typename IdxRes::type;
  using GlobalJoinPlan = typename IdxRes::GlobalJoinPlan;

  using BodyCompiler = CompileBodiesToMIR<Body, GlobalJoinPlan, Indices>;

  /// @brief The tuple of basic MIR operations (body operations followed by head operations)
  using BasicMIROps = mp::mp_append<
      typename BodyCompiler::type,
      typename CompileHeadsToMIR<Head, typename BodyCompiler::final_active_handles, Indices>::type>;

  /// @brief Enhanced MIR operations
  /// (ComputeHandlePositions is removed, handle map is embedded during body/head compilation)
  using type = BasicMIROps;
};

/**
 * @brief Compiles a Rule to a Pipeline.
 *
 * @details Wraps the MIR operations from a rule into a Pipeline ADT.
 * This provides a higher-level abstraction for rule execution. Also computes
 * and stores both the variable-to-position map (based on join plan) and the
 * IndexSpecT-to-iterator-position map (based on ColumnSource operations).
 *
 * @tparam Rule The rule to compile (must satisfy CRule)
 */
// --- ExtractSourceSpecsFromMIR implementation ---
template <typename T, typename = void>
struct HasIndexSpecType : std::false_type {};
template <typename T>
struct HasIndexSpecType<T, std::void_t<typename T::index_spec_type>> : std::true_type {};

template <typename T, typename = void>
struct HasSourcesType : std::false_type {};
template <typename T>
struct HasSourcesType<T, std::void_t<typename T::sources_type>> : std::true_type {};

// Forward declaration
// Forward declaration
template <typename Op, typename Enable = void>
struct GetOpSpecs;

template <typename Tuple>
struct GetSpecsFromTuple;

template <typename... Ops>
struct GetSpecsFromTuple<std::tuple<Ops...>> {
  using type = Mp::mp_append<typename GetOpSpecs<Ops>::type...>;
};

// --- Spec Computation Logic (Aligned with Join Plan) ---

// Helper to extract IndexSpec from a ColumnSource type
template <typename Op>
struct GetIndexSpecFromColumnSource;
template <CIndexSpec Index, typename Prefix>
struct GetIndexSpecFromColumnSource<mir::ColumnSource<Index, Prefix>> {
  using type = Index;
};

// Map clauses to IndexSpecs (following Join Plan logic)
template <CVar VarToBind, tmp::CTuple NormalClauses, tmp::CTuple BoundVars, tmp::CTuple JoinPlan>
struct MapClausesToSpecs {
  template <typename Clause>
  using ToColumnSource =
      typename ClauseToColumnSource_impl<Clause, VarToBind, BoundVars, JoinPlan>::type;

  template <typename Clause>
  using ToSpec = typename GetIndexSpecFromColumnSource<ToColumnSource<Clause>>::type;

  using type = mp::mp_transform<ToSpec, NormalClauses>;
};

// Fold state for spec computation
template <tmp::CTuple BoundVars, tmp::CTuple AccumulatedSpecs>
struct SpecFoldState {
  using bound_vars = BoundVars;
  using specs = AccumulatedSpecs;
};

// Process One Variable for Specs
template <typename State, typename VarToBind, tmp::CTuple AllJoinPlan, tmp::CTuple AllBodyClauses>
struct ProcessOneVariableForSpecs_impl {
  using BoundVars = typename State::bound_vars;
  using AccumulatedSpecs = typename State::specs;

  // Find normal clauses for this variable (consistent with CompileBodiesToMIR)
  using NormalClauses = typename FindNormalClausesWithVar<VarToBind, AllBodyClauses>::type;

  // Map to Specs
  using NewSpecs =
      typename MapClausesToSpecs<VarToBind, NormalClauses, BoundVars, AllJoinPlan>::type;

  // Handle AggClauses (which consume iterators in CountInputIterators)
  using AggClauses = typename FindAggClauses<VarToBind, AllBodyClauses>::type;
  using AggOps =
      typename MapAggClausesToAggregate<VarToBind, AggClauses, BoundVars, AllJoinPlan>::type;
  // Extract specs from AggOps
  template <typename AggOp>
  using GetAggSpec = typename AggOp::index_spec_type;
  using AggSpecs = mp::mp_transform<GetAggSpec, AggOps>;

  using type = SpecFoldState<mp::mp_unique<mp::mp_push_back<BoundVars, VarToBind>>,
                             mp::mp_append<AccumulatedSpecs, NewSpecs, AggSpecs>>;
};

template <tmp::CTuple AllJoinPlan, tmp::CTuple AllBodyClauses>
struct ProcessOneVariableForSpecs {
  template <typename State, typename VarToBind>
  using fn =
      typename ProcessOneVariableForSpecs_impl<State, VarToBind, AllJoinPlan, AllBodyClauses>::type;
};

template <typename Body, typename JoinPlan>
struct ComputeSourceSpecsFromPlan {
  // Fold over variables to get Normal + Agg specs
  using FinalState = mp::mp_fold_q<JoinPlan, SpecFoldState<std::tuple<>, std::tuple<>>,
                                   ProcessOneVariableForSpecs<JoinPlan, Body>>;
  using NormalAggSpecs = typename FinalState::specs;
  using BoundVars = typename FinalState::bound_vars;

  // Handle Negation Specs (collected at end, similar to CompileBodiesToMIR)
  using NegClauses = typename FindNegClauses<Body>::type;
  using NegOps = typename MapNegClausesToNegations<NegClauses, BoundVars, JoinPlan>::type;
  template <typename NegOp>
  using GetNegSpec = typename NegOp::index_spec_type;
  using NegSpecs = mp::mp_transform<GetNegSpec, NegOps>;

  using type = mp::mp_append<NormalAggSpecs, NegSpecs>;
};

// Destination Specs from Heads
template <typename Head, typename AllIndexSpecs = std::tuple<>>
struct ComputeDestSpecs {
  template <typename HeadClause>
  using ToSpec =
      typename GetCanonicalIndexSpecForSchema<typename GetNormalClauseSchema<HeadClause>::type,
                                              AllIndexSpecs>::type;

  using type = mp::mp_transform<ToSpec, Head>;
};

/**
 * @brief Compiles a rule into a Pipeline ADT.
 *
 * @tparam Rule The rule to compile (must satisfy CRule)
 */
template <typename Rule>
struct CompileRuleToPipeline;

template <typename Head, typename Body, typename JoinPlan>
struct CompileRuleToPipeline<Rule<Head, Body, JoinPlan>> {

  using RuleType = Rule<Head, Body, JoinPlan>;
  using MIROps = typename CompileRuleToMIR<RuleType>::type;
  // Extract the join plan directly from template arg
  using GlobalJoinPlan = typename JoinPlan::vars_type;

  // Compute the variable-to-position map
  using VarPosMap = typename mir::ComputeVarPosMap<GlobalJoinPlan>::type;

  // Compute source specs from Join Plan (Correct Approach)
  using SourceSpecs = typename ComputeSourceSpecsFromPlan<Body, GlobalJoinPlan>::type;

  // Compute index specs for this rule (used for canonical index lookup)
  using IdxRes = ComputeNaiveIndices<RuleType>;
  using RuleIndexSpecs = typename IdxRes::type;

  // Compute destination specs from Head using GetCanonicalIndexSpecForSchema with RuleIndexSpecs
  // This ensures we use the correct canonical index from AllIndexSpecs if available
  using DestSpecs = typename ComputeDestSpecs<Head, RuleIndexSpecs>::type;

  // IteratorPosMap no longer used
  using type = mir::Pipeline<MIROps, VarPosMap, SourceSpecs, DestSpecs>;
};

/**
 * @brief Converts IndexSpecs to BuildIndex MIR operations.
 *
 * @details Converts a tuple of IndexSpecT types to a tuple of BuildIndex
 * operations. These operations execute at the fixpoint level (outside nested
 * for-loops).
 *
 * @tparam IndexSpecs A std::tuple of IndexSpecT types
 */
template <tmp::CTuple IndexSpecs>
struct IndexSpecsToBuildOps {
  template <typename IndexSpecT>
  using ToBuildIndex = mir::CreateIndex<IndexSpecT>;
  using type = mp::mp_transform<ToBuildIndex, IndexSpecs>;
};

/**
 * @brief Converts IndexSpecs to RebuildIndex MIR operations.
 *
 * @details Converts a tuple of IndexSpecT types to a tuple of RebuildIndex
 * operations. These operations execute at the fixpoint level (outside nested
 * for-loops). If an IndexSpecT is NEW_VER, also generates a RebuildIndex for
 * the corresponding FULL_VER index. Be careful, this means you must call this
 * function after you filtered IndexSpecs to **only include incremental
 * relations** (for example, after calling `FilterIndexSpecsByRelations`).
 *
 * @tparam IndexSpecs A std::tuple of IndexSpecT types
 */
// Helper to extract NEW_VER rebuild operations (executed BEFORE
// ComputeDeltaOps)
template <tmp::CTuple IndexSpecs>
struct IndexSpecsToRebuildNewVerOps {
 private:
  template <CIndexSpec Spec>
  struct ToRebuildNewVer {
    using Schema = typename Spec::schema_type;
    using ColSeq = typename Spec::column_indexes_type;
    using NewVerIndexSpec = IndexSpecT<Schema, ColSeq, NEW_VER>;
    using RebuildNewVer = mir::RebuildIndex<NewVerIndexSpec>;

    // Always generate NEW_VER rebuild for incremental relations
    // The version in IndexSpecT doesn't matter - we use it to find the
    // canonical column sequence but always rebuild NEW_VER index (which is
    // where pipeline writes)
    using type = std::tuple<RebuildNewVer>;
  };

 public:
  template <CIndexSpec Spec>
  using GetRebuildNewVer = typename ToRebuildNewVer<Spec>::type;
  using RebuildNewVerOpsPerSpec = mp::mp_transform<GetRebuildNewVer, IndexSpecs>;
  using type = mp::mp_flatten<RebuildNewVerOpsPerSpec>;
};

// Helper to extract FULL_VER rebuild operations (executed AFTER
// ComputeDeltaOps)
template <tmp::CTuple IndexSpecs>
struct IndexSpecsToRebuildFullVerOps {
 private:
  template <CIndexSpec Spec>
  struct ToRebuildFullVer {
    using Schema = typename Spec::schema_type;
    using ColSeq = typename Spec::column_indexes_type;
    using FullVerIndexSpec = IndexSpecT<Schema, ColSeq, FULL_VER>;
    using RebuildFullVer = mir::RebuildIndex<FullVerIndexSpec>;

    // Always generate FULL_VER rebuild (for all versions, since we merge into
    // FULL_VER)
    using type = std::tuple<RebuildFullVer>;
  };

 public:
  template <CIndexSpec Spec>
  using GetRebuildFullVer = typename ToRebuildFullVer<Spec>::type;
  using RebuildFullVerOpsPerSpec = mp::mp_transform<GetRebuildFullVer, IndexSpecs>;
  using type = mp::mp_flatten<RebuildFullVerOpsPerSpec>;
};

/**
 * @brief Helper to find canonical index spec for a schema in
 * CanonicalIndexSpecs tuple.
 *
 * @details Searches the CanonicalIndexSpecs tuple for an IndexSpecT matching
 * the given Schema. Returns the found index spec, or creates a canonical one
 * if not found.
 *
 * @tparam Schema The relation schema to look up
 * @tparam CanonicalIndexSpecs A std::tuple of IndexSpecT types
 */
template <CRelationSchema Schema, typename CanonicalIndexSpecs>
struct FindCanonicalIndexForSchema {
 private:
  template <typename IndexSpecT>

  // For calling convention of mp::mp_find_if, we should find better way to do
  // this (or maybe is good enough, as long as we unify code style)
  struct MatchesSchema {
    static constexpr bool value = std::is_same_v<typename IndexSpecT::schema_type, Schema>;
  };
  static constexpr std::size_t found_idx =
      mp::mp_find_if<CanonicalIndexSpecs, MatchesSchema>::value;
  static constexpr bool found = (found_idx < mp::mp_size<CanonicalIndexSpecs>::value);

  // Helper to create canonical index spec (same pattern as
  // GetCanonicalIndexSpec)
  using canonical_seq = typename Schema::canonical_index_type;
  template <std::size_t... Is>
  static auto to_int_seq(std::index_sequence<Is...>)
      -> std::integer_sequence<int, static_cast<int>(Is)...>;
  using column_indexes_type = decltype(to_int_seq(canonical_seq{}));
  using DefaultCanonicalIndex = IndexSpecT<Schema, column_indexes_type, FULL_VER>;

 public:
  using type =
      std::conditional_t<found, mp::mp_at_c<CanonicalIndexSpecs, found_idx>, DefaultCanonicalIndex>;
};

/**
 * @brief Replaces a DestinationRelation with canonical index from
 * CanonicalIndexSpecs.
 *
 * @details Takes a DestinationRelation and looks up its canonical index spec
 * from the CanonicalIndexSpecs tuple, then creates a new DestinationRelation
 * with the canonical index. The version is preserved from the original
 * DestinationRelation.
 *
 * @tparam DestRel The DestinationRelation to replace
 * @tparam CanonicalIndexSpecs A std::tuple of IndexSpecT types (from
 * CollectCanonicalIndexSpecs)
 */
template <typename DestRel, typename CanonicalIndexSpecs>
struct ReplaceDestinationRelationWithCanonicalIndex;

template <CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver, CIndexSpec OldIndex,
          typename CanonicalIndexSpecs>
struct ReplaceDestinationRelationWithCanonicalIndex<
    mir::DestinationRelation<Schema, Terms, Ver, OldIndex>, CanonicalIndexSpecs> {
 private:
  using FoundCanonical = typename FindCanonicalIndexForSchema<Schema, CanonicalIndexSpecs>::type;
  // Preserve the version from the original DestinationRelation
  using CanonicalIndex = IndexSpecT<typename FoundCanonical::schema_type,
                                    typename FoundCanonical::column_indexes_type, Ver>;

 public:
  using type = mir::DestinationRelation<Schema, Terms, Ver, CanonicalIndex>;
};

// For non-DestinationRelation types, return unchanged
template <typename T, typename CanonicalIndexSpecs>
struct ReplaceDestinationRelationWithCanonicalIndex {
  using type = T;
};

/**
 * @brief Transforms all DestinationRelation operations in a tuple to use
 * canonical indices.
 *
 * @details Applies ReplaceDestinationRelationWithCanonicalIndex to each
 * element in the tuple, replacing DestinationRelation operations with
 * canonical indices.
 *
 * @tparam MIROps A std::tuple of MIR operations
 * @tparam CanonicalIndexSpecs A std::tuple of IndexSpecT types
 */
template <tmp::CTuple MIROps, typename CanonicalIndexSpecs>
struct ReplaceDestinationRelationsInTuple {
  template <typename Op>
  using ReplaceOp =
      typename ReplaceDestinationRelationWithCanonicalIndex<Op, CanonicalIndexSpecs>::type;
  using type = mp::mp_transform<ReplaceOp, MIROps>;
};

/**
 * @brief Transforms a Pipeline to use canonical indices in its
 * DestinationRelation operations.
 *
 * @details Replaces all DestinationRelation operations in the Pipeline's
 * MIROps with versions that use canonical indices from CanonicalIndexSpecs.
 *
 * @tparam Pipeline The Pipeline to transform
 * @tparam CanonicalIndexSpecs A std::tuple of IndexSpecT types
 */
/**
 * @brief Maps destination specs to canonical indices from CanonicalIndexSpecs.
 *
 * @details Takes the original destination specs (which contain schemas) and
 * replaces their indices with canonical indices from CanonicalIndexSpecs,
 * converting to FULL_VER.
 */
template <typename DestSpecs, typename CanonicalIndexSpecs>
struct MapDestSpecsToCanonical;

template <typename CanonicalIndexSpecs>
struct MapDestSpecsToCanonical<std::tuple<>, CanonicalIndexSpecs> {
  using type = std::tuple<>;
};

template <typename HeadSpec, typename... TailSpecs, typename CanonicalIndexSpecs>
struct MapDestSpecsToCanonical<std::tuple<HeadSpec, TailSpecs...>, CanonicalIndexSpecs> {
  using Schema = typename HeadSpec::schema_type;
  using FoundCanonical = typename FindCanonicalIndexForSchema<Schema, CanonicalIndexSpecs>::type;
  // Convert to FULL_VER (destination relations always use FULL_VER for deduplication)
  using CanonicalFullVer = IndexSpecT<typename FoundCanonical::schema_type,
                                      typename FoundCanonical::column_indexes_type, FULL_VER>;
  using TailMapped =
      typename MapDestSpecsToCanonical<std::tuple<TailSpecs...>, CanonicalIndexSpecs>::type;
  using type = mp::mp_push_front<TailMapped, CanonicalFullVer>;
};

template <typename Pipeline, typename CanonicalIndexSpecs>
struct ReplaceDestinationRelationsInPipeline {
  static_assert(mir::is_pipeline_v<Pipeline>, "Pipeline must satisfy CPipeline");
  using ReplacedOps = typename ReplaceDestinationRelationsInTuple<typename Pipeline::mir_ops_type,
                                                                  CanonicalIndexSpecs>::type;

  // Source specs don't change (they come from body clauses, not affected by destination
  // replacement) Use original Pipeline::source_specs_type (computed from AST) instead of
  // re-extracting from MIR
  using SourceSpecs = typename Pipeline::source_specs_type;

  // Destination specs change - map original destination specs to canonical indices
  // This is faster than extracting from MIR and more accurate
  using OriginalDestSpecs = typename Pipeline::destination_specs_type;
  using DestSpecs = typename MapDestSpecsToCanonical<OriginalDestSpecs, CanonicalIndexSpecs>::type;

  // Preserve var_pos_map from the original pipeline
  using type =
      mir::Pipeline<ReplacedOps, typename Pipeline::var_pos_map_type, SourceSpecs, DestSpecs>;
};

/**
 * @brief Compiles a Fixpoint to a FixpointPlan.
 *
 * @details Compiles all rules in a Fixpoint to Pipelines and collects
 * all required index specifications. This provides a higher-level
 * MIR representation that can be executed more efficiently.
 *
 * @tparam FP The Fixpoint to compile (must satisfy CFixpoint)
 */
template <CFixpoint FP>
struct CompileFixpointToPlan;

}  // namespace mir

// Forward declarations for external helpers (from runtime/query.h)
namespace detail {
template <typename RuleList>
struct CollectAllIndexSpecs;

template <typename IncrementalRelations, typename AllIndexSpecs>
struct CollectCanonicalIndexSpecs;
}  // namespace detail

template <CRule Rule>
struct GetHeadSchemas;

namespace mir {
template <CRule... Rules>
struct CompileFixpointToPlan<Fixpoint<Rules...>> {
 private:
  using RulesTuple = std::tuple<Rules...>;

  // Compile each rule to a Pipeline (these execute in nested for-loops)
  template <typename Rule>
  using ToPipeline = typename CompileRuleToPipeline<Rule>::type;
  using InitialPipelines = mp::mp_transform<ToPipeline, RulesTuple>;

  // Collect all IndexSpecs from all rules (defined in runtime/query.h in SRDatalog::detail)
  using AllIndexSpecs = typename ::SRDatalog::detail::CollectAllIndexSpecs<RulesTuple>::type;

  // Get incremental relations (from rule heads) (defined in query.h)
  template <typename Rule>
  using GetHeadSchemasForRule = typename ::SRDatalog::GetHeadSchemas<Rule>::type;
  using SchemasPerRule = mp::mp_transform<GetHeadSchemasForRule, RulesTuple>;
  using AllHeadSchemas = mp::mp_flatten<SchemasPerRule>;
  using IncrementalRelations = tmp::TupleUnique<AllHeadSchemas>;

  // Get canonical index specs for incremental relations (defined in runtime/query.h in
  // SRDatalog::detail)
  using CanonicalIndexSpecs =
      typename ::SRDatalog::detail::CollectCanonicalIndexSpecs<IncrementalRelations,
                                                               AllIndexSpecs>::type;

  // Replace DestinationRelation operations in Pipelines with canonical
  // indices
  template <typename Pipeline>
  using ReplaceDestRels =
      typename ReplaceDestinationRelationsInPipeline<Pipeline, CanonicalIndexSpecs>::type;
  using Pipelines = mp::mp_transform<ReplaceDestRels, InitialPipelines>;

  // Filter AllIndexSpecs to only include incremental relations (for index
  // rebuilding)
  using FilteredIndexSpecs =
      typename mir::FilterIndexSpecsByRelations<AllIndexSpecs, IncrementalRelations>::type;

  // Convert IndexSpecs to MIR operations (executed at fixpoint level, outside
  // nested for-loops) BuildIndex ops: empty (canonical indexes built once at
  // start, not in MIR)
  using BuildIndexOps = std::tuple<>;

  // SecondaryIndexSpecs: Indices for FULL relation that are NOT canonical
  using SecondaryIndexSpecs = tmp::TupleSetDifference<FilteredIndexSpecs, CanonicalIndexSpecs>;

  // RebuildIndex ops: split into NEW_VER (before ComputeDeltaOps) and
  // FULL_VER (after ComputeDeltaOps) NEW_VER rebuilds: executed BEFORE
  // ComputeDeltaOps (after pipeline execution, before swap) CRITICAL: Use
  // CanonicalIndexSpecs instead of FilteredIndexSpecs to ensure NEW_VER has
  // canonical indexes. After SwapRelations, DELTA becomes NEW, and
  // InsertFromRelation uses canonical IndexForInsert. If NEW has
  // non-canonical indexes, reconstruct_from_index will fail.
  using RebuildNewVerOps = typename IndexSpecsToRebuildNewVerOps<CanonicalIndexSpecs>::type;

  // Split SecondaryIndexSpecs into FULL and DELTA/NEW
  // FULL indices are maintained via MergeIndex.
  // DELTA indices must be rebuilt every iteration (using RebuildIndex).
  // NEW indices are handled by RebuildNewVerOps.

  template <std::size_t Ver>
  struct HasVersion {
    template <typename Spec>
    using fn = mp::mp_bool<(Spec::kVersion == Ver)>;
  };

  using SecondaryFullSpecsUnfiltered = mp::mp_filter_q<HasVersion<FULL_VER>, SecondaryIndexSpecs>;
  using SecondaryDeltaSpecs = mp::mp_filter_q<HasVersion<DELTA_VER>, SecondaryIndexSpecs>;

  // Convert CanonicalIndexSpecs to FULL_VER for merging
  // Canonical indices need to be merged from DELTA_VER to FULL_VER after
  // ComputeDeltaOps
  template <CIndexSpec Spec>
  using CanonicalToFullVer =
      IndexSpecT<typename Spec::schema_type, typename Spec::column_indexes_type, FULL_VER>;
  using CanonicalFullSpecs = mp::mp_transform<CanonicalToFullVer, CanonicalIndexSpecs>;

  // Deduplicate SecondaryFullSpecs first (smaller list, faster)
  // Note: Even though SecondaryIndexSpecs and CanonicalIndexSpecs are
  // disjoint, after converting CanonicalIndexSpecs to FULL_VER, overlaps can
  // occur. So we still need to deduplicate the combined list, but
  // deduplicating SecondaryFullSpecs first reduces the work (deduplicating
  // smaller lists is faster)
  using SecondaryFullSpecs = tmp::TupleUnique<SecondaryFullSpecsUnfiltered>;
  using AllFullSpecs = mp::mp_append<SecondaryFullSpecs, CanonicalFullSpecs>;
  using UniqueFullSpecs = tmp::TupleUnique<AllFullSpecs>;

  template <typename Spec>
  using ToMergeIndex = mir::MergeIndex<Spec>;
  // MergeOps: merge DELTA_VER indices into FULL_VER indices
  // Deduplicate at spec level for better compilation performance
  using MergeOps = mp::mp_transform<ToMergeIndex, UniqueFullSpecs>;

  template <typename Spec>
  using ToRebuildIndex = mir::RebuildIndex<Spec>;
  // RebuildDeltaOps: executed every iteration to build indices on fresh DELTA
  using RebuildDeltaOps = mp::mp_transform<ToRebuildIndex, SecondaryDeltaSpecs>;

  // FULL_VER rebuilds: executed AFTER ComputeDeltaOps and MergeOps (for
  // secondary indices only) Canonical indices are merged via MergeOps, not
  // rebuilt
  using RebuildFullVerOps =
      typename IndexSpecsToRebuildFullVerOps<SecondaryFullSpecsUnfiltered>::type;

  // Convert Pipelines to ExecutePipeline operations
  template <typename Pipeline>
  using ToExecutePipeline = mir::ExecutePipeline<Pipeline>;
  using ExecutePipelineOps = mp::mp_transform<ToExecutePipeline, Pipelines>;
  using ExecutePipelineOpsFlattened = mp::mp_rename<ExecutePipelineOps, std::tuple>;

  // Compile compute_delta operations for each incremental relation
  // Each compute_delta becomes: CheckSize<NEW_VER> ->
  // ClearRelation<DELTA_VER> -> InsertFromRelation<NEW_VER, FULL_VER> ->
  // SwapRelations -> SetVersion<DELTA_VER> -> SetVersion<NEW_VER>
  template <typename Schema>
  struct CompileComputeDelta {
    // Get canonical index spec for this schema from CanonicalIndexSpecs
    // CanonicalIndexSpecs already contains one IndexSpecT per incremental
    // relation
    template <typename IndexSpecT>
    struct MatchesSchema {
      static constexpr bool value = std::is_same_v<typename IndexSpecT::schema_type, Schema>;
    };
    static constexpr std::size_t found_idx =
        mp::mp_find_if<CanonicalIndexSpecs, MatchesSchema>::value;
    static constexpr bool found = (found_idx < mp::mp_size<CanonicalIndexSpecs>::value);

    // Helper to create canonical index spec (same pattern as
    // FindCanonicalIndexForSchema)
    using canonical_seq = typename Schema::canonical_index_type;
    template <std::size_t... Is>
    static auto to_int_seq(std::index_sequence<Is...>)
        -> std::integer_sequence<int, static_cast<int>(Is)...>;
    using column_indexes_type = decltype(to_int_seq(canonical_seq{}));
    using DefaultCanonicalIndex = IndexSpecT<Schema, column_indexes_type, FULL_VER>;

    // Use found canonical index, or create default if not found (shouldn't
    // happen)
    using FoundCanonical = std::conditional_t<found, mp::mp_at_c<CanonicalIndexSpecs, found_idx>,
                                              DefaultCanonicalIndex>;
    // Convert to FULL_VER for the index used in InsertFromRelation
    using IndexForInsert =
        IndexSpecT<Schema, typename FoundCanonical::column_indexes_type, FULL_VER>;

    // Create NEW_VER index spec for reconstruction after merge (before swap)
    using NewVerIndexSpec =
        IndexSpecT<Schema, typename FoundCanonical::column_indexes_type, NEW_VER>;

    using type = std::tuple<mir::ClearRelation<Schema, DELTA_VER>,  // Always clear
                                                                    // delta first
                            mir::CheckSize<Schema, NEW_VER>,        // Early return if empty
                                                                    // (after clearing delta)
                            // Note: RebuildIndex for NEW_VER is handled by
                            // RebuildNewVerOps (before ComputeDeltaOps)
                            mir::ComputeDelta<Schema, NEW_VER, FULL_VER, DELTA_VER,
                                              IndexForInsert>,   // Compute NEW -
                                                                 // FULL -> DELTA
                            mir::ClearRelation<Schema, NEW_VER>  // Clear NEW after
                                                                 // computing delta
                            >;
  };
  template <typename Schema>
  using CompileComputeDeltaOp = typename CompileComputeDelta<Schema>::type;
  using ComputeDeltaOpsPerSchema = mp::mp_transform<CompileComputeDeltaOp, IncrementalRelations>;
  using ComputeDeltaOps = mp::mp_flatten<ComputeDeltaOpsPerSchema>;

  // template <typename Schema>
  // using ToMergeRelation = mir::MergeRelation<Schema>;
  // using MergeRelationOps = mp::mp_transform<ToMergeRelation, IncrementalRelations>;

  // Order:
  // 1. BuildIndexOps (Init secondary, iter 0 only)
  // 2. Execute Pipelines (reads Full/Delta, writes New)
  // 3. RebuildNewVerOps (indexes on New - needed before ComputeDelta)
  // 4. ComputeDeltaOps (New -> Delta via set difference, Clear New)
  // 5. MergeOps (Delta -> Full for both canonical and secondary indices)
  // 6. RebuildDeltaOps (indices on Delta for next iter)
  // 7. RebuildFullVerOps (secondary FULL indices - canonical indices are
  // merged, not rebuilt)
  using Instructions = mp::mp_append<BuildIndexOps, ExecutePipelineOpsFlattened, RebuildNewVerOps,
                                     ComputeDeltaOps, MergeOps, RebuildDeltaOps>;

 public:
  // FixpointPlan is now monadic - just a flat list of instructions
  using type = mir::FixpointPlan<Instructions, IncrementalRelations, CanonicalIndexSpecs>;
};

// ============================================================================
// Enhanced MIR Operations with Handle Positions (Using Hana Maps)
// ============================================================================

namespace detail {

/**
 * @brief Type trait to check if an operation has handle positions
 */
template <typename Op, typename = void>
struct HasHandlePositions : std::false_type {};

template <typename Op>
struct HasHandlePositions<Op, std::void_t<decltype(Op::has_handles)>> : std::true_type {};

}  // namespace detail

}  // namespace mir
}  // namespace SRDatalog