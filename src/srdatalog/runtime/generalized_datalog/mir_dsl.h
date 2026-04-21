/**
 * @file mir_dsl.h
 * @brief Value-wrapper DSL for MIR (Mid-level Intermediate Representation).
 *
 * @details This file provides a value-based DSL for constructing MIR operations
 * ergonomically. Each wrapper struct holds a `type` alias that produces the actual
 * MIR type, enabling compile-time type computation while providing a readable syntax.
 *
 * This mirrors the AST DSL pattern (VarWrapper, RelWrapper) and enables writing:
 * ```cpp
 * constexpr auto x = "x"_v;
 * constexpr auto idx = index<Schema, 0, 1>();
 * constexpr auto src = column_source(idx.delta());
 * constexpr auto join = column_join(x, src);
 * // decltype(join)::type == mir::ColumnJoin<...>
 * ```
 *
 * @note **C++20 Features Used**: Concepts, requires clauses, consteval functions,
 *       auto NTTP, and fold expressions.
 */

#pragma once

#include "ast.h"
#include "mir.h"

namespace SRDatalog::mir::dsl {

using namespace SRDatalog::AST;

// ============================================================================
// Forward declarations for wrapper types
// ============================================================================

template <CRelationSchema Schema, typename ColSeq, std::size_t Ver>
struct IndexSpecWrapper;

template <typename... VarWrappers>
struct VarsTupleWrapper;

template <typename IndexW, typename PrefixVarsTuple>
struct ColumnSourceWrapper;

template <typename VarW, typename SourcesTuple, std::size_t HandleStart = 0,
          typename VarToHandlesMap = void>
struct ColumnJoinWrapper;

template <typename VarsTuple, typename SourcesTuple, std::size_t HandleStart = 0,
          typename VarToHandlesMap = void>
struct CartesianJoinWrapper;

template <CRelationSchema Schema, std::size_t Ver, typename TermsTuple>
struct DestinationRelationWrapper;

template <typename OpsTuple, typename VarPosMapT, typename SourceSpecs = std::tuple<>,
          typename DestSpecs = std::tuple<>, typename AllSchemas = std::tuple<>,
          typename RuleNameT = void>
struct PipelineWrapper;

// ============================================================================
// IndexSpec Wrapper - supports versioning via .delta(), .full(), .newt()
// ============================================================================

/**
 * @brief Wrapper for IndexSpecT with version methods.
 *
 * @details Allows writing `index<Schema, 0, 1>().delta()` to get a
 * DELTA_VER index spec. The `type` alias extracts the actual MIR type.
 */
template <CRelationSchema Schema, typename ColSeq, std::size_t Ver>
struct IndexSpecWrapper {
  using type = IndexSpecT<Schema, ColSeq, Ver>;
  using schema_type = Schema;
  using column_seq_type = ColSeq;
  static constexpr std::size_t version = Ver;

  /// @brief Get this index spec with DELTA_VER
  constexpr auto delta() const {
    return IndexSpecWrapper<Schema, ColSeq, DELTA_VER>{};
  }

  /// @brief Get this index spec with FULL_VER
  constexpr auto full() const {
    return IndexSpecWrapper<Schema, ColSeq, FULL_VER>{};
  }

  /// @brief Get this index spec with NEW_VER
  constexpr auto newt() const {
    return IndexSpecWrapper<Schema, ColSeq, NEW_VER>{};
  }
};

/**
 * @brief Create an index spec wrapper.
 *
 * @tparam Schema The relation schema
 * @tparam Cols... The column indices (ints)
 * @return IndexSpecWrapper with FULL_VER by default
 *
 * @example
 * ```cpp
 * constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
 * constexpr auto delta_idx = pt_idx.delta();  // DELTA_VER version
 * ```
 */
template <CRelationSchema Schema, int... Cols>
constexpr auto index() {
  return IndexSpecWrapper<Schema, std::integer_sequence<int, Cols...>, FULL_VER>{};
}

/**
 * @brief Get a relation wrapper with default (canonical) index.
 *
 * @tparam Schema The relation schema
 * @return IndexSpecWrapper with canonical indices and FULL_VER
 *
 * @example
 * ```cpp
 * constexpr auto r = rel<PointsToSchema>();
 * constexpr auto delta = r.delta();
 * ```
 */
template <CRelationSchema Schema>
constexpr auto rel() {
  using Seq = typename Schema::canonical_index_type;
  return IndexSpecWrapper<Schema, Seq, FULL_VER>{};
}

// ============================================================================
// ColumnSource Wrapper
// ============================================================================

/**
 * @brief Wrapper for ColumnSource MIR operation.
 *
 * @tparam IndexW The IndexSpecWrapper type
 * @tparam PrefixVarsTuple std::tuple of Var types (not wrappers!)
 */
template <typename IndexW, typename PrefixVarsTuple>
struct ColumnSourceWrapper {
  using index_spec_type = typename IndexW::type;
  using prefix_vars_type = PrefixVarsTuple;
  using type = ColumnSource<index_spec_type, prefix_vars_type>;
};

// Helper to extract Var types from VarWrapper arguments
template <typename... VarWrappers>
struct ExtractVarTypesToTuple {
  using type = std::tuple<typename VarWrappers::type...>;
};

// Specialization for empty
template <>
struct ExtractVarTypesToTuple<> {
  using type = std::tuple<>;
};

/**
 * @brief Create a ColumnSource wrapper.
 *
 * @param idx The IndexSpecWrapper (determines schema, columns, version)
 * @param prefix_vars... Optional VarWrapper arguments for prefix variables
 * @return ColumnSourceWrapper
 *
 * @example
 * ```cpp
 * constexpr auto x = "x"_v;
 * constexpr auto pt_idx = index<PointsToSchema, 0, 1>();
 * constexpr auto src1 = column_source(pt_idx.delta());        // no prefix
 * constexpr auto src2 = column_source(pt_idx.full(), x);      // prefix = (x)
 * ```
 */
template <typename IndexW, typename... PrefixVarWrappers>
constexpr auto column_source(IndexW, PrefixVarWrappers...) {
  using PrefixTuple = typename ExtractVarTypesToTuple<PrefixVarWrappers...>::type;
  return ColumnSourceWrapper<IndexW, PrefixTuple>{};
}

// ============================================================================
// Scan Wrapper
// ============================================================================

/**
 * @brief Wrapper for Scan MIR operation with explicit handle positions.
 *
 * @tparam VarsTuple std::tuple of Var types
 * @tparam IndexW The IndexSpecWrapper type
 * @tparam PrefixVarsTuple std::tuple of Var types (already bound)
 * @tparam HandleStart The starting handle index for this operation
 * @tparam VarToHandlesMap Hana map from prefix vars to handle indices
 */
template <typename VarsTuple, typename IndexW, typename PrefixVarsTuple, std::size_t HandleStart,
          typename VarToHandlesMap>
struct ScanWrapper {
  using vars_type = VarsTuple;
  using index_spec_type = typename IndexW::type;
  using prefix_vars_type = PrefixVarsTuple;
  using type = Scan<vars_type, index_spec_type, prefix_vars_type, HandleStart, VarToHandlesMap>;
};

/**
 * @brief Create a Scan wrapper (legacy/default - uses dummy values 0/void).
 */
template <typename... ScanVarWrappers, typename IndexW, typename... PrefixVarWrappers>
constexpr auto scan(VarsTupleWrapper<ScanVarWrappers...>, IndexW, PrefixVarWrappers...) {
  using ScanVarsTuple = std::tuple<typename ScanVarWrappers::type...>;
  using PrefixTuple = typename ExtractVarTypesToTuple<PrefixVarWrappers...>::type;
  // Use 0 and void as defaults, but MIR op requires explicit parameters now
  // We'll pass 0 and void, assuming the user will only use this during AST lowering
  // or will update to use explicit handles if needed manually.
  return ScanWrapper<ScanVarsTuple, IndexW, PrefixTuple, 0, void>{};
}

/**
 * @brief Create a Scan wrapper with explicit handle positions.
 */
template <std::size_t HS, typename VHM, typename... ScanVarWrappers, typename IndexW,
          typename... PrefixVarWrappers>
constexpr auto scan_h(VarsTupleWrapper<ScanVarWrappers...>, IndexW, PrefixVarWrappers...) {
  using ScanVarsTuple = std::tuple<typename ScanVarWrappers::type...>;
  using PrefixTuple = typename ExtractVarTypesToTuple<PrefixVarWrappers...>::type;
  return ScanWrapper<ScanVarsTuple, IndexW, PrefixTuple, HS, VHM>{};
}

// ============================================================================
// ColumnJoin Wrapper
// ============================================================================

// Helper to extract ColumnSourceWrapper::type into a tuple
template <typename... SourceWrappers>
struct ExtractSourceTypes {
  using type = std::tuple<typename SourceWrappers::type...>;
};

/**
 * @brief Wrapper for ColumnJoin MIR operation with explicit handle positions.
 *
 * @tparam VarW The VarWrapper type
 * @tparam SourcesTuple std::tuple of ColumnSource types
 * @tparam HandleStart The starting handle index for this operation
 * @tparam VarToHandlesMap Hana map from prefix vars to handle indices
 */
template <typename VarW, typename SourcesTuple, std::size_t HandleStart, typename VarToHandlesMap>
struct ColumnJoinWrapper {
  using var_type = typename VarW::type;
  using sources_type = SourcesTuple;
  using type =
      ColumnJoin<var_type, sources_type, DefaultJoinStrategy, HandleStart, VarToHandlesMap>;
};

/**
 * @brief Create a ColumnJoin wrapper (legacy/default).
 */
template <typename VarW, typename... SourceWrappers>
constexpr auto column_join(VarW, SourceWrappers...) {
  using SourcesTuple = typename ExtractSourceTypes<SourceWrappers...>::type;
  return ColumnJoinWrapper<VarW, SourcesTuple, 0, void>{};
}

/**
 * @brief Create a ColumnJoin wrapper with explicit handle positions.
 */
template <std::size_t HS, typename VHM, typename VarW, typename... SourceWrappers>
constexpr auto column_join_h(VarW, SourceWrappers...) {
  using SourcesTuple = typename ExtractSourceTypes<SourceWrappers...>::type;
  return ColumnJoinWrapper<VarW, SourcesTuple, HS, VHM>{};
}

// ============================================================================
// CartesianJoin Wrapper
// ============================================================================

/**
 * @brief Wrapper for CartesianJoin MIR operation with explicit handle positions.
 *
 * @tparam VarsTuple std::tuple of Var types
 * @tparam SourcesTuple std::tuple of ColumnSource types
 * @tparam HandleStart The starting handle index for this operation
 * @tparam VarToHandlesMap Hana map from prefix vars to handle indices
 */
template <typename VarsTuple, typename SourcesTuple, std::size_t HandleStart,
          typename VarToHandlesMap>
struct CartesianJoinWrapper {
  using vars_type = VarsTuple;
  using sources_type = SourcesTuple;
  using type = CartesianJoin<vars_type, sources_type, HandleStart, VarToHandlesMap>;
};

/**
 * @brief Helper to wrap multiple variables for CartesianJoin.
 */
template <typename... VarWrappers>
struct VarsTupleWrapper {
  using type = std::tuple<typename VarWrappers::type...>;
};

template <typename... VarWrappers>
constexpr auto vars(VarWrappers...) {
  return VarsTupleWrapper<VarWrappers...>{};
}

/**
 * @brief Create a CartesianJoin wrapper (legacy/default).
 */
template <typename... VarWrappers, typename... SourceWrappers>
constexpr auto cartesian_join(VarsTupleWrapper<VarWrappers...>, SourceWrappers...) {
  using VarsTuple = std::tuple<typename VarWrappers::type...>;
  using SourcesTuple = typename ExtractSourceTypes<SourceWrappers...>::type;
  return CartesianJoinWrapper<VarsTuple, SourcesTuple, 0, void>{};
}

/**
 * @brief Create a CartesianJoin wrapper with explicit handle positions.
 */
template <std::size_t HS, typename VHM, typename... VarWrappers, typename... SourceWrappers>
constexpr auto cartesian_join_h(VarsTupleWrapper<VarWrappers...>, SourceWrappers...) {
  using VarsTuple = std::tuple<typename VarWrappers::type...>;
  using SourcesTuple = typename ExtractSourceTypes<SourceWrappers...>::type;
  return CartesianJoinWrapper<VarsTuple, SourcesTuple, HS, VHM>{};
}

// ============================================================================
// Filter Wrapper
// ============================================================================

/**
 * @brief Wrapper for Filter MIR operation.
 *
 * @tparam VarsTuple std::tuple of Var types that are passed to predicate
 * @tparam Predicate The lambda/functor type for filtering
 */
template <typename VarsTuple, typename Predicate>
struct FilterWrapper {
  using vars_type = VarsTuple;
  using predicate_type = Predicate;
  using type = Filter<VarsTuple, Predicate>;
};

/**
 * @brief Create a Filter wrapper.
 */
template <typename... VarWrappers, typename Predicate>
constexpr auto filter(VarsTupleWrapper<VarWrappers...>, Predicate pred) {
  using VarsTuple = std::tuple<typename VarWrappers::type...>;
  return FilterWrapper<VarsTuple, Predicate>{};
}

// ============================================================================
// Aggregate Wrapper
// ============================================================================

/**
 * @brief Wrapper for Aggregate MIR operation with explicit handle positions.
 */
template <CVar VarToBind, CIndexSpec IndexSpec, typename PrefixVars, typename AggFunc,
          std::size_t HandleStart, typename VarToHandlesMap>
struct AggregateWrapper {
  using var_type = VarToBind;
  using index_type = IndexSpec;
  using prefix_vars_type = PrefixVars;
  using agg_func_type = AggFunc;
  using type = Aggregate<VarToBind, IndexSpec, PrefixVars, AggFunc, HandleStart, VarToHandlesMap>;
};

// ============================================================================
// Negation Wrapper
// ============================================================================

/**
 * @brief Wrapper for Negation MIR operation with ConstArgs support.
 */
template <CIndexSpec IndexSpec, typename PrefixVars, std::size_t HS, typename VHM,
          typename ConstArgs = std::tuple<>>
struct NegationWrapper {
  using index_type = IndexSpec;
  using prefix_vars_type = PrefixVars;
  using const_args_type = ConstArgs;
  using type = mir::Negation<IndexSpec, PrefixVars, HS, VHM, ConstArgs>;
};

/**
 * @brief Create a Negation operation (legacy/default).
 */
template <typename IndexWrapper, typename... VarWrappers>
constexpr auto negation(IndexWrapper, VarsTupleWrapper<VarWrappers...>) {
  using IndexSpec = typename IndexWrapper::type;
  using PrefixVars = std::tuple<typename VarWrappers::type...>;
  return NegationWrapper<IndexSpec, PrefixVars, 0, void, std::tuple<>>{};
}

/**
 * @brief Create a Negation operation with explicit handle positions.
 */
template <std::size_t HS, typename VHM, typename IndexW, typename... VarWrappers>
constexpr auto negation_h(IndexW /*unused*/, VarsTupleWrapper<VarWrappers...> /*unused*/) {
  using Index = typename IndexW::type;
  using PrefixVars = std::tuple<typename VarWrappers::type...>;
  return NegationWrapper<Index, PrefixVars, HS, VHM, std::tuple<>>{};
}

/**
 * @brief Create a Negation operation with explicit handle positions and ConstArgs.
 */
template <std::size_t HS, typename VHM, typename ConstArgs, typename IndexW,
          typename... VarWrappers>
constexpr auto negation_h(IndexW /*unused*/, VarsTupleWrapper<VarWrappers...> /*unused*/) {
  using Index = typename IndexW::type;
  using PrefixVars = std::tuple<typename VarWrappers::type...>;
  return NegationWrapper<Index, PrefixVars, HS, VHM, ConstArgs>{};
}

/**
 * @brief Create an Aggregate wrapper (legacy/default).
 */
template <typename AggFunc, typename VarWrapper, typename IndexWrapper,
          typename... PrefixVarWrappers>
constexpr auto aggregate(VarWrapper, IndexWrapper, VarsTupleWrapper<PrefixVarWrappers...>) {
  using VarToBind = typename VarWrapper::type;
  using IndexSpec = typename IndexWrapper::type;
  using PrefixVars = std::tuple<typename PrefixVarWrappers::type...>;
  return AggregateWrapper<VarToBind, IndexSpec, PrefixVars, AggFunc, 0, void>{};
}

/**
 * @brief Create an Aggregate wrapper with explicit handle positions.
 */
template <std::size_t HS, typename VHM, typename AggFunc, typename VarW, typename IndexW,
          typename... PrefixVarWrappers>
constexpr auto aggregate_h(VarW /*unused*/, IndexW /*unused*/,
                           VarsTupleWrapper<PrefixVarWrappers...> /*unused*/) {
  using VarToBind = typename VarW::type;
  using IndexSpec = typename IndexW::type;
  using PrefixVars = std::tuple<typename PrefixVarWrappers::type...>;
  return AggregateWrapper<VarToBind, IndexSpec, PrefixVars, AggFunc, HS, VHM>{};
}

// ============================================================================
// DestinationRelation Wrapper
// ============================================================================

/**
 * @brief Wrapper for DestinationRelation MIR operation.
 *
 * @tparam Schema The target relation schema
 * @tparam Ver The version (typically NEW_VER)
 * @tparam TermsTuple std::tuple of Term types (Var or Const)
 */
template <CRelationSchema Schema, std::size_t Ver, typename TermsTuple>
struct DestinationRelationWrapper {
  using schema_type = Schema;
  using terms_type = TermsTuple;
  static constexpr std::size_t version = Ver;
  // Use default index spec (will be computed by runtime)
  using type = DestinationRelation<Schema, TermsTuple, Ver>;
};

// Helper to check if a type is an IndexSpecWrapper
template <typename T>
struct is_index_spec_wrapper : std::false_type {};

template <CRelationSchema Schema, typename ColSeq, std::size_t Ver>
struct is_index_spec_wrapper<IndexSpecWrapper<Schema, ColSeq, Ver>> : std::true_type {};

template <typename T>
inline constexpr bool is_index_spec_wrapper_v = is_index_spec_wrapper<T>::value;

/**
 * @brief Create a DestinationRelation wrapper (insert into relation).
 *
 * @tparam Schema The target relation schema
 * @tparam Ver The version (default: NEW_VER)
 * @tparam TermWrappers VarWrapper or ConstWrapper arguments
 * @return DestinationRelationWrapper
 *
 * @example
 * ```cpp
 * constexpr auto dest = insert_into<PointsToSchema>(y, x);  // NEW_VER by default
 * constexpr auto dest2 = insert_into<PointsToSchema, FULL_VER>(y, x);
 * ```
 */
template <CRelationSchema Schema, std::size_t Ver = NEW_VER, typename... TermWrappers>
constexpr auto insert_into(TermWrappers...) {
  using TermsTuple = typename ExtractVarTypesToTuple<TermWrappers...>::type;
  return DestinationRelationWrapper<Schema, Ver, TermsTuple>{};
}

/**
 * @brief Create a DestinationRelation wrapper with explicit IndexSpec.
 *
 * @tparam Schema The target relation schema
 * @tparam Ver The version (default: NEW_VER)
 * @tparam IndexSpecWrapperT The IndexSpec wrapper type (from index<Schema, cols...>().ver())
 * @param terms... VarWrapper or ConstWrapper arguments
 * @return DestinationRelation with explicit dedup IndexSpec
 */
template <CRelationSchema Schema, std::size_t Ver, typename IndexSpecWrapperT,
          typename... TermWrappers>
  requires is_index_spec_wrapper_v<IndexSpecWrapperT>
constexpr auto insert_into(TermWrappers...) {
  using TermsTuple = typename ExtractVarTypesToTuple<TermWrappers...>::type;
  using IndexSpec =
      typename IndexSpecWrapperT::type;  // IndexSpecWrapper uses 'type' not 'index_spec_type'
  return DestinationRelation<Schema, TermsTuple, Ver, IndexSpec>{};
}

// ============================================================================
// Pipeline Wrapper
// ============================================================================

/**
 * @brief Wrapper for Pipeline MIR structure.
 *
 * @tparam OpsTuple std::tuple of MIR operation types
 * @tparam VarPosMapT The variable position map type
 * @tparam SourceSpecs Tuple of source IndexSpecs
 * @tparam DestSpecs Tuple of destination IndexSpecs
 * @tparam AllSchemas Tuple of all schema types
 * @tparam RuleNameT TString type for rule name (or void for unnamed)
 */
template <typename OpsTuple, typename VarPosMapT, typename SourceSpecs, typename DestSpecs,
          typename AllSchemas, typename RuleNameT>
struct PipelineWrapper {
  using ops_type = OpsTuple;
  using var_pos_map_type = VarPosMapT;
  using source_specs_type = SourceSpecs;
  using dest_specs_type = DestSpecs;
  using all_schemas_type = AllSchemas;
  using rule_name_type = RuleNameT;
  using type = Pipeline<ops_type, var_pos_map_type, source_specs_type, dest_specs_type,
                        all_schemas_type, rule_name_type>;
};
// Helper to detect if a type has ::type member (is a wrapper)
template <typename T, typename = void>
struct has_type_member : std::false_type {};

template <typename T>
struct has_type_member<T, std::void_t<typename T::type>> : std::true_type {};

// Helper to unwrap a type if it's a wrapper, otherwise use it directly
// Use partial specialization to avoid evaluating T::type when it doesn't exist
template <typename T, bool HasType = has_type_member<T>::value>
struct UnwrapOpType {
  using type = T;  // No ::type member, use T directly
};

template <typename T>
struct UnwrapOpType<T, true> {
  using type = typename T::type;  // Has ::type member, unwrap
};

// Helper to extract operation types from wrappers or raw MIR types
template <typename... OpWrappers>
struct ExtractOpTypes {
  using type = std::tuple<typename UnwrapOpType<OpWrappers>::type...>;
};

// Helper to collect all variables from operations (simplified - extracts from
// ColumnJoin/CartesianJoin)
template <typename OpsTuple>
struct CollectVarsFromOps;

template <>
struct CollectVarsFromOps<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename Head, typename... Tail>
struct CollectVarsFromOps<std::tuple<Head, Tail...>> {
 private:
  // Extract var from ColumnJoin
  template <typename Op>
  struct ExtractVar {
    using type = std::tuple<>;
  };

  template <CVar V, typename Sources, typename Strategy, std::size_t HS, typename VHM>
  struct ExtractVar<ColumnJoin<V, Sources, Strategy, HS, VHM>> {
    using type = std::tuple<V>;
  };

  template <typename Vars, typename Sources, std::size_t HS, typename VHM>
  struct ExtractVar<CartesianJoin<Vars, Sources, HS, VHM>> {
    using type = Vars;  // Already a tuple of vars
  };

  template <CVar VarToBind, CIndexSpec IndexSpec, typename PrefixVars, typename AggFunc,
            std::size_t HS, typename VHM>
  struct ExtractVar<Aggregate<VarToBind, IndexSpec, PrefixVars, AggFunc, HS, VHM>> {
    using type = std::tuple<VarToBind>;
  };

  template <typename Vars, typename IndexSpec, typename Prefix, std::size_t HS, typename VHM>
  struct ExtractVar<mir::Scan<Vars, IndexSpec, Prefix, HS, VHM>> {
    using type = Vars;
  };

  // LetClause binds a single variable
  template <CVar VarToBind, typename Expr>
  struct ExtractVar<AST::LetClause<VarToBind, Expr>> {
    using type = std::tuple<VarToBind>;
  };

  using HeadVars = typename ExtractVar<Head>::type;
  using TailVars = typename CollectVarsFromOps<std::tuple<Tail...>>::type;

 public:
  using type = tmp::TupleConcat<HeadVars, TailVars>;
};

/**
 * @brief Create a Pipeline wrapper.
 *
 * @param ops... Operation wrappers (ColumnJoinWrapper, CartesianJoinWrapper,
 *               DestinationRelationWrapper, etc.)
 * @return PipelineWrapper
 *
 * @example
 * ```cpp
 * constexpr auto pipe = pipeline(
 *   column_join(z, column_source(pt_idx.delta()), column_source(assign_idx.full())),
 *   cartesian_join(vars(x, y), column_source(pt_idx.delta(), z), column_source(assign_idx.full(),
 * z)), insert_into<PointsToSchema>(y, x)
 * );
 * ```
 */
template <typename... OpWrappers>
constexpr auto pipeline(OpWrappers...) {
  using OpsTuple = typename ExtractOpTypes<OpWrappers...>::type;

  // Compute VarPosMap from the collected variables
  using AllVars = tmp::TupleUnique<typename CollectVarsFromOps<OpsTuple>::type>;
  using VarPosMapT = typename ComputeVarPosMap<AllVars>::type;

  return PipelineWrapper<OpsTuple, VarPosMapT>{};
}

/**
 * @brief Create a pipeline with explicit source/destination specs.
 *
 * @tparam ExplicitSourceSpecs Tuple of IndexSpecT for sources
 * @tparam ExplicitDestSpecs Tuple of IndexSpecT for destinations
 * @param ops... The MIR operations
 */
template <typename ExplicitSourceSpecs, typename ExplicitDestSpecs, typename... OpWrappers>
constexpr auto pipeline(OpWrappers...) {
  using OpsTuple = typename ExtractOpTypes<OpWrappers...>::type;

  // Compute VarPosMap from the collected variables
  using AllVars = tmp::TupleUnique<typename CollectVarsFromOps<OpsTuple>::type>;
  using VarPosMapT = typename ComputeVarPosMap<AllVars>::type;

  return PipelineWrapper<OpsTuple, VarPosMapT, ExplicitSourceSpecs, ExplicitDestSpecs>{};
}

/**
 * @brief Create a pipeline with explicit source/destination specs AND explicit all schemas.
 *
 * @tparam ExplicitSourceSpecs Tuple of IndexSpecT for sources
 * @tparam ExplicitDestSpecs Tuple of IndexSpecT for destinations
 * @tparam ExplicitAllSchemas Tuple of Schema types
 * @param ops... The MIR operations
 */
template <typename ExplicitSourceSpecs, typename ExplicitDestSpecs, typename ExplicitAllSchemas,
          typename... OpWrappers>
constexpr auto pipeline(OpWrappers...) {
  using OpsTuple = typename ExtractOpTypes<OpWrappers...>::type;

  // Compute VarPosMap from the collected variables
  using AllVars = tmp::TupleUnique<typename CollectVarsFromOps<OpsTuple>::type>;
  using VarPosMapT = typename ComputeVarPosMap<AllVars>::type;

  using Wrapper = PipelineWrapper<OpsTuple, VarPosMapT, ExplicitSourceSpecs, ExplicitDestSpecs,
                                  ExplicitAllSchemas>;
  // static_assert(sizeof(Wrapper) > 0, "Wrapper is valid");
  return Wrapper{};
}

/**
 * @brief Create a pipeline with explicit source/destination specs, schemas, AND rule name.
 *
 * @tparam ExplicitSourceSpecs Tuple of IndexSpecT for sources
 * @tparam ExplicitDestSpecs Tuple of IndexSpecT for destinations
 * @tparam ExplicitAllSchemas Tuple of Schema types
 * @tparam RuleNameT TString type for rule name (for NVTX profiling)
 * @param ops... The MIR operations
 */
template <typename ExplicitSourceSpecs, typename ExplicitDestSpecs, typename ExplicitAllSchemas,
          typename RuleNameT, typename... OpWrappers>
constexpr auto pipeline(OpWrappers...) {
  using OpsTuple = typename ExtractOpTypes<OpWrappers...>::type;

  // Compute VarPosMap from the collected variables
  using AllVars = tmp::TupleUnique<typename CollectVarsFromOps<OpsTuple>::type>;
  using VarPosMapT = typename ComputeVarPosMap<AllVars>::type;

  using Wrapper = PipelineWrapper<OpsTuple, VarPosMapT, ExplicitSourceSpecs, ExplicitDestSpecs,
                                  ExplicitAllSchemas, RuleNameT>;
  return Wrapper{};
}

// ============================================================================
// ExecutePipeline Wrapper
// ============================================================================

/**
 * @brief Wrapper for ExecutePipeline instruction.
 */
template <typename PipelineW>
struct ExecutePipelineWrapper {
  using pipeline_type = typename PipelineW::type;
  using type = ExecutePipeline<pipeline_type>;
};

/**
 * @brief Create an ExecutePipeline wrapper.
 *
 * @param pipe PipelineWrapper
 * @return ExecutePipelineWrapper
 */
template <typename PipelineW>
constexpr auto execute(PipelineW) {
  return ExecutePipelineWrapper<PipelineW>{};
}

/**
 * @brief Create a JitExecutePipeline for JIT codegen.
 */
template <typename JitKernel, typename SourceSpecs, typename DestSpecs, typename AllSchemas,
          typename RuleNameT>
constexpr auto jit_execute() {
  struct W {
    using type = JitExecutePipeline<
        JitKernel, Pipeline<std::tuple<>, void, SourceSpecs, DestSpecs, AllSchemas, RuleNameT>>;
  };
  return W{};
}

/**
 * @brief Create a MaterializedExecutePipeline for host-side Thrust executors.
 *
 * @details MaterializedExecutePipeline uses a JitRunner struct with a static
 * execute(db, iteration) method that runs Thrust primitives on the host.
 * The runner struct is generated in a separate JIT batch file.
 *
 * @tparam JitRunner The JitRunner struct (e.g., JitRunner_CGE_Materialized)
 */
template <typename JitRunner>
constexpr auto materialized_execute() {
  struct W {
    using type = MaterializedExecutePipeline<JitRunner>;
  };
  return W{};
}

// ============================================================================
// Other Fixpoint-level instruction wrappers
// ============================================================================

/**
 * @brief Wrapper for CreateIndex instruction.
 */
template <typename IndexW>
struct CreateIndexWrapper {
  using type = CreateIndex<typename IndexW::type>;
};

template <typename IndexW>
constexpr auto create_index(IndexW) {
  return CreateIndexWrapper<IndexW>{};
}

/**
 * @brief Wrapper for RebuildIndex instruction.
 */
template <typename IndexW>
struct RebuildIndexWrapper {
  using type = RebuildIndex<typename IndexW::type>;
};

template <typename IndexW>
constexpr auto rebuild_index(IndexW) {
  return RebuildIndexWrapper<IndexW>{};
}

/**
 * @brief Wrapper for ClearRelation instruction.
 */
template <CRelationSchema Schema, std::size_t Ver>
struct ClearRelationWrapper {
  using type = ClearRelation<Schema, Ver>;
};

template <CRelationSchema Schema, std::size_t Ver = NEW_VER>
constexpr auto clear_relation() {
  return ClearRelationWrapper<Schema, Ver>{};
}

/**
 * @brief Wrapper for CheckSize instruction.
 */
template <CRelationSchema Schema, std::size_t Ver>
struct CheckSizeWrapper {
  using type = CheckSize<Schema, Ver>;
};

template <CRelationSchema Schema, std::size_t Ver = NEW_VER>
constexpr auto check_size() {
  return CheckSizeWrapper<Schema, Ver>{};
}

/**
 * @brief Wrapper for PostStratumReconstructInternCols instruction.
 *
 * @details MUST only be emitted AFTER moFixpointPlan, never inside fixpoint loop.
 * Reconstructs intern column values from the canonical index to column storage.
 */
template <typename IndexW>
struct PostStratumReconstructInternColsWrapper {
  using type = PostStratumReconstructInternCols<typename IndexW::type>;
};

/**
 * @brief Create a PostStratumReconstructInternCols wrapper.
 *
 * @param idx The IndexSpecWrapper for the canonical index
 * @return PostStratumReconstructInternColsWrapper
 */
template <typename IndexW>
constexpr auto post_stratum_reconstruct_intern_cols(IndexW) {
  return PostStratumReconstructInternColsWrapper<IndexW>{};
}

/**
 * @brief Wrapper for MergeIndex instruction.
 */
template <typename IndexW>
struct MergeIndexWrapper {
  using type = MergeIndex<typename IndexW::type>;
};

template <typename IndexW>
constexpr auto merge_index(IndexW) {
  return MergeIndexWrapper<IndexW>{};
}

/**
 * @brief Wrapper for ComputeDelta instruction.
 */
template <typename IndexW>
struct ComputeDeltaWrapper {
  using index_spec_type = typename IndexW::type;
  using schema_type = typename index_spec_type::schema_type;
  // Standard delta computation: NEW - FULL -> DELTA
  using type =
      SRDatalog::mir::ComputeDelta<schema_type, NEW_VER, FULL_VER, DELTA_VER, index_spec_type>;
};

template <typename IndexW>
constexpr auto compute_delta(IndexW) {
  return ComputeDeltaWrapper<IndexW>{};
}

// ============================================================================
// FixpointPlan Wrapper
// ============================================================================

/**
 * @brief Wrapper for FixpointPlan.
 */
template <typename InstructionsTuple, typename IncrementalRelations = std::tuple<>,
          typename CanonicalIndexSpecs = std::tuple<>, typename AllSourceSpecs = std::tuple<>>
struct FixpointPlanWrapper {
  using instructions_type = InstructionsTuple;
  using incremental_relations_type = IncrementalRelations;
  using canonical_index_specs_type = CanonicalIndexSpecs;
  using all_source_specs_type = AllSourceSpecs;
  using type =
      FixpointPlan<InstructionsTuple, IncrementalRelations, CanonicalIndexSpecs, AllSourceSpecs>;
};

// Helper to extract instruction types
template <typename... InstrWrappers>
struct ExtractInstructionTypes {
  using type = std::tuple<typename InstrWrappers::type...>;
};

/**
 * @brief Create a FixpointPlan wrapper (basic version - metadata auto-extracted).
 *
 * @param instructions... Instruction wrappers (ExecutePipelineWrapper, etc.)
 * @return FixpointPlanWrapper with auto-extracted metadata
 */
template <typename... InstrWrappers>
constexpr auto fixpoint_plan(InstrWrappers... instructions) {
  using InstrTuple = typename ExtractInstructionTypes<InstrWrappers...>::type;
  return FixpointPlanWrapper<InstrTuple>{};
}

/**
 * @brief Create a FixpointPlan wrapper with explicit metadata.
 *
 * @tparam IncrementalRelationsTuple Tuple of relation schemas that are modified
 * @tparam CanonicalIndexSpecsTuple Tuple of canonical index specs
 * @param instructions... Instruction wrappers (ExecutePipelineWrapper, etc.)
 * @return FixpointPlanWrapper with explicit metadata
 *
 * @example
 * ```cpp
 * using IncRels = std::tuple<PointsToSchema>;
 * using CanonicalSpecs = std::tuple<IndexSpecT<PointsToSchema, ...>>;
 * auto plan = fixpoint_plan<IncRels, CanonicalSpecs>(execute(...), rebuild_index(...));
 * ```
 */
template <typename IncrementalRelationsTuple, typename CanonicalIndexSpecsTuple,
          typename... InstrWrappers>
constexpr auto fixpoint_plan(InstrWrappers... instructions) {
  using InstrTuple = typename ExtractInstructionTypes<InstrWrappers...>::type;
  return FixpointPlanWrapper<InstrTuple, IncrementalRelationsTuple, CanonicalIndexSpecsTuple>{};
}

/**
 * @brief Create a FixpointPlan wrapper with explicit AllSourceSpecs optimization.
 *
 * @tparam IncrementalRelationsTuple Tuple of incremental relations
 * @tparam CanonicalIndexSpecsTuple Tuple of canonical index specs
 * @tparam AllSourceSpecsTuple Tuple of ALL source index specs (pre-computed)
 * @param instructions... Instruction wrappers
 */
template <typename IncrementalRelationsTuple, typename CanonicalIndexSpecsTuple,
          typename AllSourceSpecsTuple, typename... InstrWrappers>
constexpr auto fixpoint_plan(InstrWrappers... instructions) {
  using InstrTuple = typename ExtractInstructionTypes<InstrWrappers...>::type;
  return FixpointPlanWrapper<InstrTuple, IncrementalRelationsTuple, CanonicalIndexSpecsTuple,
                             AllSourceSpecsTuple>{};
}

// ============================================================================
// InjectCppHook Wrapper
// ============================================================================

/**
 * @brief Wrapper for InjectCppHook instruction.
 *
 * @details Allows injecting arbitrary C++ code at fixpoint level with database access.
 * The functor receives (const auto& db) and can access any relation via:
 *   get_relation_by_schema<Schema, Version>(db).canonical_index().root().degree()
 *
 * @tparam Functor A callable type that takes (const auto& db) and returns void
 */
template <typename Functor>
struct InjectCppHookWrapper {
  using type = InjectCppHook<Functor>;
};

/**
 * @brief Create an InjectCppHook wrapper.
 *
 * @param functor The lambda/functor that receives database reference
 * @return InjectCppHookWrapper
 *
 * @example
 * ```cpp
 * auto hook = inject_cpp_hook([](const auto& db) {
 *   auto& rel = get_relation_by_schema<MySchema, FULL_VER>(db);
 *   std::cout << "Size: " << rel.canonical_index().root().degree() << std::endl;
 * });
 * ```
 */
template <typename Functor>
constexpr auto inject_cpp_hook(Functor) {
  return InjectCppHookWrapper<Functor>{};
}

}  // namespace SRDatalog::mir::dsl
