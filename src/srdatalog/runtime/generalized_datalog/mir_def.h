/**
 * @file mir_def.h
 * @brief Core definitions for Mid-level Intermediate Representation (MIR).
 *
 * @details This file contains the core MIR AST structures (ColumnSource, ColumnJoin,
 * DestinationRelation, etc.) and type predicates. It defines the structure of the
 * IR without the compilation logic or execution plan details.
 */

#pragma once

#include "ast.h"
#include "tmp.h"
#include <boost/hana.hpp>
#include <boost/mp11.hpp>

namespace Mp = boost::mp11;
namespace hana = boost::hana;

namespace SRDatalog {
using namespace AST;
namespace mir {
namespace mp = SRDatalog::tmp::mp;

/**
 * @brief Abstract Data Type (ADT) for an index specification.
 *
 * @details Specifies which index to use for a relation, including the relation
 * schema and the column indexes that form the index key. This is a compile-time
 * ADT used during query compilation, not the runtime index structure. The
 * version specifies which relation version to access during semi-naive
 * evaluation.
 *
 * @warning This is a compile-time ADT, not the actual runtime index type from
 * relation_col.h
 *
 * @tparam schema The relation schema this index is for
 * @tparam ColumnIndexes A std::integer_sequence<int, ...> specifying which
 * columns are indexed (complete permutation with all columns)
 * @tparam Ver The version number for semi-naive evaluation (default: FULL_VER)
 * @tparam PrefixLen The number of "real" join/lookup columns before completion
 * (default: 0 = all columns are real). For completed partial indexes, this
 * tracks how many columns were originally specified for joins/lookups.
 */
template <CRelationSchema schema, typename ColumnIndexes, std::size_t Ver = FULL_VER,
          std::size_t PrefixLen = 0>
struct IndexSpecT {
  /// @brief The relation schema type
  using schema_type = schema;
  /// @brief The column indexes sequence type
  using column_indexes_type = ColumnIndexes;
  /// @brief The version number for semi-naive evaluation
  static constexpr std::size_t kVersion = Ver;
  /// @brief The number of "real" join/lookup columns (0 = all columns are real)
  static constexpr std::size_t kPrefixLen = PrefixLen;
};
// Type predicate for index spec (Scheme/Racket style: (index-spec? x))
DEFINE_TYPE_PREDICATE(is_index_spec, IndexSpecT, CIndexSpec,
                      (CRelationSchema schema, typename ColumnIndexes, std::size_t Ver,
                       std::size_t PrefixLen),
                      (schema, ColumnIndexes, Ver, PrefixLen))

// @brief ADT for Version
using Version = std::size_t;

namespace detail {
template <typename Seq>
struct ToIntSeq;
template <std::size_t... Is>
struct ToIntSeq<std::integer_sequence<std::size_t, Is...>> {
  using type = std::integer_sequence<int, static_cast<int>(Is)...>;
};
}  // namespace detail

// Forward declaration for ComputeVarToHandlesMap (needed for handles_for_var in operations)
namespace detail {
template <typename Op, std::size_t HandleStart, typename HandleIndices>
struct ComputeVarToHandlesMap;

// Forward declaration for ComputeVarToHandlesMapSafe (needed for handles_for_var in operations)
template <typename Op, std::size_t StartOffset>
struct ComputeVarToHandlesMapSafe;
}  // namespace detail

/**
 * @brief Mid-level Intermediate Representation (MIR) namespace.
 *
 * @details Contains the MIR AST structures used for column-oriented join
 * execution. MIR is generated from the high-level AST during compilation and is
 * closer to the actual execution model.
 */
/**
 * @brief Abstract Data Type (ADT) for a list of bound variables.
 *
 * @details Represents a compile-time list of variables that have been bound
 * during join execution. Used in MIR to track which variables are available
 * in the current execution context.
 *
 * @tparam Vars Variadic list of CVar types that are currently bound
 */
template <CVar... Vars>
using BoundVars = std::tuple<Vars...>;
// Type predicate for bound vars (Scheme/Racket style: (bound-vars? x))
template <typename T>
struct is_bound_vars : std::false_type {};
template <CVar... Vars>
struct is_bound_vars<BoundVars<Vars...>> : std::true_type {};
template <typename T>
inline constexpr bool is_bound_vars_v = is_bound_vars<T>::value;
template <typename T>
concept CBoundVars = is_bound_vars<T>::value;

/**
 * @brief Abstract Data Type (ADT) for a column source in MIR.
 *
 * @details Represents a source of data for a column-oriented join operation.
 * Specifies which index to use, which prefix variables are already bound.
 * The version is extracted from the IndexSpec ADT.
 *
 * @tparam Index The IndexSpec specifying which index to use (includes version)
 * @tparam BVars A BoundVars tuple of variables that are already bound (prefix)
 */
template <CIndexSpec Index, tmp::CTuple BVars>
struct ColumnSource {
  using index_spec_type = Index;
  using prefix_vars_type = BVars;
};
// Type predicate for column source (Scheme/Racket style: (column-source? x))
DEFINE_TYPE_PREDICATE(is_column_source, ColumnSource, CColumnSource,
                      (CIndexSpec Index, tmp::CTuple BVars), (Index, BVars))

/**
 * @brief Abstract Data Type (ADT) for a linear scan operation in MIR.
 *
 * @details Represents a linear scan over an index (relation or intermediate) that
 * binds multiple variables from the columns of the index. This is used when standard
 * ColumnSource (single variable binding) is not sufficient, e.g. for scanning
 * a multi-column relation that was not joined previously (Cartesian-like but linear).
 *
 * @tparam VarsTuple Tuple of variables to bind from the scan
 * @tparam IndexSpec The IndexSpec specifying which index to use
 * @tparam PrefixVars Variables that are already bound (prefix)
 */

template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HandleStart,
          typename VarToHandlesMap>
struct Scan {
  using vars_type = VarsTuple;
  using index_spec_type = IndexSpec;
  using prefix_vars_type = PrefixVars;
  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;

  // Helper: check if map present
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable (using Hana map lookup)
  template <CVar V>
  static constexpr auto handles_for_var() {
    if constexpr (has_handles) {
      // Use the explicitly provided map
      constexpr auto map_val = VarToHandlesMap{};
      if constexpr (hana::contains(map_val, hana::type_c<V>)) {
        return hana::at_key(map_val, hana::type_c<V>);
      } else {
        return std::integer_sequence<std::size_t>{};
      }
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());
};
// Type predicate
DEFINE_TYPE_PREDICATE(is_scan, Scan, CScan,
                      (typename VarsTuple, typename IndexSpec, typename PrefixVars,
                       std::size_t HandleStart, typename VarToHandlesMap),
                      (VarsTuple, IndexSpec, PrefixVars, HandleStart, VarToHandlesMap))

/**
 * @brief Abstract Data Type (ADT) for a join strategy.
 *
 * @details Represents the strategy used to execute a join operation.
 * Different strategies may imply different algorithms (e.g., WCOJ,
 * Sort-Merge) and execution models.
 */
struct JoinStrategy {};

/**
 * @brief The default join strategy (WCOJ / Intersection).
 */
struct DefaultJoinStrategy : JoinStrategy {};

/**
 * @brief Strategy using TVJoin (Sort-Merge pipeline).
 */
struct TVJoinStrategy : JoinStrategy {};

/**
 * @brief Abstract Data Type (ADT) for a column join operation in MIR.
 *
 * @details Represents a multi-way join operation that binds a variable.
 * The execution strategy is determined by the `Strategy` template parameter.
 *
 * @tparam Var The variable to bind through this join
 * @tparam Sources A std::tuple of ColumnSource types to intersect
 * @tparam Strategy The join strategy to use (default: DefaultJoinStrategy)
 */

template <CVar Var, tmp::CTuple Sources, typename Strategy, std::size_t HandleStart,
          typename VarToHandlesMap>
struct ColumnJoin {
  using var_type = Var;
  using sources_type = Sources;
  using strategy_type = Strategy;

  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable (using Hana map lookup)
  template <CVar V>
  static constexpr auto handles_for_var() {
    if constexpr (has_handles) {
      // Use the explicitly provided map
      constexpr auto map_val = VarToHandlesMap{};
      if constexpr (hana::contains(map_val, hana::type_c<V>)) {
        return hana::at_key(map_val, hana::type_c<V>);
      } else {
        return std::integer_sequence<std::size_t>{};
      }
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());
};
// Type predicate for column join (Scheme/Racket style: (column-join? x))
DEFINE_TYPE_PREDICATE(is_column_join, ColumnJoin, CColumnJoin,
                      (CVar Var, tmp::CTuple Sources, typename Strategy, std::size_t HandleStart,
                       typename VarToHandlesMap),
                      (Var, Sources, Strategy, HandleStart, VarToHandlesMap))

/**
 * @brief Abstract Data Type (ADT) for a cartesian product join operation in
 * MIR.
 *
 * @details Represents a join operation that binds multiple variables
 * simultaneously via cartesian product of multiple sources. Unlike ColumnJoin
 * which binds one variable at a time through nested loops, CartesianJoin
 * binds all variables at once using flat cartesian product iteration over
 * unmaterialized handle spans.
 *
 * This enables efficient flat iteration over all combinations without the
 * overhead of nested loops and handle materialization.
 *
 * @tparam Vars A std::tuple of CVar types to bind simultaneously
 * @tparam Sources A std::tuple of ColumnSource types to take cartesian
 * product over
 *
 * @example
 * ```cpp
 * // Binds both VarX and VarZ from their respective sources
 * using JoinXZ = mir::CartesianJoin<
 *   std::tuple<VarX, VarZ>,
 *   std::tuple<SourcePath_X, SourceEdge_Z>
 * >;
 * ```
 */

template <tmp::CTuple Vars, tmp::CTuple Sources, std::size_t HandleStart, typename VarToHandlesMap>
struct CartesianJoin {
  using vars_type = Vars;
  using sources_type = Sources;

  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable (using Hana map lookup)
  template <CVar V>
  static constexpr auto handles_for_var() {
    if constexpr (has_handles) {
      // Use the explicitly provided map
      constexpr auto map_val = VarToHandlesMap{};
      if constexpr (hana::contains(map_val, hana::type_c<V>)) {
        return hana::at_key(map_val, hana::type_c<V>);
      } else {
        return std::integer_sequence<std::size_t>{};
      }
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());
};
// Type predicate for cartesian join
DEFINE_TYPE_PREDICATE(is_cartesian_join, CartesianJoin, CCartesianJoin,
                      (tmp::CTuple Vars, tmp::CTuple Sources, std::size_t HandleStart,
                       typename VarToHandlesMap),
                      (Vars, Sources, HandleStart, VarToHandlesMap))

/**
 * @brief Abstract Data Type (ADT) for a filter operation in MIR.
 *
 * @details Represents an operation that filters tuples based on a predicate
 * function. The predicate receives bound variables and returns true if the
 * tuple should be kept.
 *
 * @tparam BoundVars Tuple of variables that are passed to the predicate
 * @tparam Predicate Lambda/functor type that takes bound vars and returns
 * bool
 */
template <tmp::CTuple BoundVars, typename Predicate>
struct Filter {
  using bound_vars_type = BoundVars;
  using predicate_type = Predicate;
};
// Type predicate for filter (Scheme/Racket style: (filter? x))
DEFINE_TYPE_PREDICATE(is_filter, Filter, CFilter, (tmp::CTuple BoundVars, typename Predicate),
                      (BoundVars, Predicate))

/**
 * @brief Abstract Data Type (ADT) for an aggregation operation in MIR.
 *
 * @details Represents an operation that aggregates values from a relation
 * cursor. The cursor is restricted by prefix variables. The aggregation
 * function is applied to the values in the cursor's range to produce a single
 * result, which is bound to VarToBind.
 *
 * @tparam VarToBind The variable to bind with the aggregation result
 * @tparam Index The IndexSpec specifying which index to use
 * @tparam PrefixVars Variables that are already bound (prefix)
 * @tparam AggFunc The aggregation functor type
 */

template <CVar VarToBind, CIndexSpec Index, tmp::CTuple PrefixVars, typename AggFunc,
          std::size_t HandleStart, typename VarToHandlesMap>
struct Aggregate {
  using var_type = VarToBind;
  using index_spec_type = Index;
  using prefix_vars_type = PrefixVars;
  using agg_func_type = AggFunc;

  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable (using Hana map lookup)
  template <CVar V>
  static constexpr auto handles_for_var() {
    if constexpr (has_handles) {
      // Use the explicitly provided map
      constexpr auto map_val = VarToHandlesMap{};
      if constexpr (hana::contains(map_val, hana::type_c<V>)) {
        return hana::at_key(map_val, hana::type_c<V>);
      } else {
        return std::integer_sequence<std::size_t>{};
      }
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());
};
// Type predicate for aggregate (Scheme/Racket style: (aggregate? x))
// Type predicate
DEFINE_TYPE_PREDICATE(is_aggregate, Aggregate, CAggregate,
                      (CVar VarToBind, CIndexSpec Index, tmp::CTuple PrefixVars, typename AggFunc,
                       std::size_t HandleStart, typename VarToHandlesMap),
                      (VarToBind, Index, PrefixVars, AggFunc, HandleStart, VarToHandlesMap))

/**
 * @brief Abstract Data Type (ADT) for a negation operation in MIR.
 *
 * @details Represents an operation that filters tuples based on the absence
 * of matching tuples in a specific relation. This is an anti-join operation.
 *
 * @tparam Index The IndexSpec specifying which index to use for the check
 * @tparam PrefixVars Variables that are already bound (prefix)
 * @tparam ConstArgs Tuple of std::pair<std::integral_constant<int, ColIdx>,
 *                   std::integral_constant<int64_t, Val>> for constant filtering
 */
template <CIndexSpec Index, tmp::CTuple PrefixVars, std::size_t HandleStart,
          typename VarToHandlesMap, typename ConstArgs = std::tuple<>>
struct Negation {
  using index_spec_type = Index;
  using prefix_vars_type = PrefixVars;
  using const_args_type = ConstArgs;

  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable (using Hana map lookup)
  template <CVar V>
  static constexpr auto handles_for_var() {
    if constexpr (has_handles) {
      // Use the explicitly provided map
      constexpr auto map_val = VarToHandlesMap{};
      if constexpr (hana::contains(map_val, hana::type_c<V>)) {
        return hana::at_key(map_val, hana::type_c<V>);
      } else {
        return std::integer_sequence<std::size_t>{};
      }
    } else {
      return std::integer_sequence<std::size_t>{};
    }
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());

  // Wrapper compatibility
  using type = Negation;
};
// Type predicate for negation (Scheme/Racket style: (negation? x))
DEFINE_TYPE_PREDICATE(is_negation, Negation, CNegation,
                      (CIndexSpec Index, tmp::CTuple PrefixVars, std::size_t HandleStart,
                       typename VarToHandlesMap, typename ConstArgs),
                      (Index, PrefixVars, HandleStart, VarToHandlesMap, ConstArgs))

/**
 * @brief Abstract Data Type (ADT) for a destination relation in MIR.
 *
 * @details Represents an operation that writes a fact to a relation. This is
 * generated from rule heads during compilation. The terms specify what values
 * to write, which may include variables bound earlier in the join plan.
 * The IndexSpec specifies which index to use for deduplication checks.
 *
 * @tparam Schema The relation schema to write to
 * @tparam Terms A std::tuple of CTerm types specifying the values to write
 * @tparam Ver The version number (typically NEW_VER) for semi-naive
 * evaluation
 * @tparam IndexSpec The IndexSpecT type specifying which index to use
 * (defaults to empty index as placeholder)
 */
template <CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver,
          CIndexSpec IndexSpec = IndexSpecT<Schema, std::integer_sequence<int>, 0>,
          typename VarToHandlesMap = void>
struct DestinationRelation {
  using schema_type = Schema;
  using terms_type = Terms;
  static constexpr std::size_t Version = Ver;
  using index_spec_type = IndexSpec;
  using var_to_handles_map = VarToHandlesMap;
  static constexpr bool has_handles = !std::is_same_v<VarToHandlesMap, void>;

  // Helper to get handles that depend on a variable
  // NOTE: DestinationRelation is a WRITE operation, not a read operation.
  // It doesn't have handles that need prefixing during iteration, so always return empty.
  template <CVar V>
  static constexpr auto handles_for_var() {
    // DestinationRelation doesn't participate in handle updates
    return std::integer_sequence<std::size_t>{};
  }

  template <CVar V>
  using handles_for_var_t = decltype(handles_for_var<V>());
};
// Type predicate for destination relation (Scheme/Racket style:
// (destination-relation? x))
DEFINE_TYPE_PREDICATE(is_destination_relation, DestinationRelation, CDestinationRelation,
                      (CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver,
                       CIndexSpec IndexSpec, typename VarToHandlesMap),
                      (Schema, Terms, Ver, IndexSpec, VarToHandlesMap))

// @brief NOPE, this is for debug, always at the end of each rule
struct Nope {};

/**
 * @brief Computes a hana::map from variable types to their stack positions.
 *
 * @details Given a join plan (tuple of variables in binding order), creates
 * a compile-time hana::map that maps each variable type to its position
 * (0-based index) in the join plan. This map is used for tuple-based state
 * management to determine where each variable's value is stored.
 *
 * @tparam JoinPlan A std::tuple of Var types in binding order
 */
template <tmp::CTuple JoinPlan>
struct ComputeVarPosMap;

template <typename... Vars>
  requires(CVar<Vars> && ...)
struct ComputeVarPosMap<std::tuple<Vars...>> {
  // Create pairs for all variables with their positions
  template <std::size_t... Is>
  static constexpr auto make_map_impl(std::index_sequence<Is...>) {
    return hana::make_map(hana::make_pair(
        hana::type_c<std::tuple_element_t<Is, std::tuple<Vars...>>>, hana::int_c<Is>)...);
  }

  /// @brief The hana::map type mapping variable types to positions
  using type = decltype(make_map_impl(std::make_index_sequence<sizeof...(Vars)>{}));
};

/**
 * @brief Trait to count input iterators (ColumnSources/Aggregates) in a MIR
 * operation/tuple. Used to calculate offsets for linear iterator mapping.
 */
template <typename T>
struct CountInputIterators {
  static constexpr std::size_t value = 0;
};

// Tuple summation
template <typename... Ts>
struct CountInputIterators<std::tuple<Ts...>> {
  static constexpr std::size_t value = (CountInputIterators<Ts>::value + ... + 0);
};

// ColumnSource -> 1
template <CIndexSpec IndexSpecT, tmp::CTuple PrefixVars>
struct CountInputIterators<mir::ColumnSource<IndexSpecT, PrefixVars>> {
  static constexpr std::size_t value = 1;
};

// Aggregate -> 1
template <typename VarToBind, typename IndexSpecT, typename PrefixVars, typename AggFunc,
          std::size_t HS, typename VHM>
struct CountInputIterators<mir::Aggregate<VarToBind, IndexSpecT, PrefixVars, AggFunc, HS, VHM>> {
  static constexpr std::size_t value = 1;
};

// Negation -> 1
template <CIndexSpec IndexSpecT, tmp::CTuple PrefixVars, std::size_t HS, typename VHM,
          typename ConstArgs>
struct CountInputIterators<mir::Negation<IndexSpecT, PrefixVars, HS, VHM, ConstArgs>> {
  static constexpr std::size_t value = 1;
};

// ColumnJoin -> Sum of sources
template <typename Var, typename Sources, typename Strategy, std::size_t HS, typename VHM>
struct CountInputIterators<mir::ColumnJoin<Var, Sources, Strategy, HS, VHM>> {
  static constexpr std::size_t value = CountInputIterators<Sources>::value;
};

// CartesianJoin -> Sum of sources
template <typename Vars, typename Sources, std::size_t HS, typename VHM>
struct CountInputIterators<mir::CartesianJoin<Vars, Sources, HS, VHM>> {
  static constexpr std::size_t value = CountInputIterators<Sources>::value;
};

template <typename Vars, typename IndexSpec, typename Prefix, std::size_t HS, typename VHM>
struct CountInputIterators<mir::Scan<Vars, IndexSpec, Prefix, HS, VHM>> {
  static constexpr std::size_t value = 1;
};

/**
 * @brief Computes a hana::map from IndexSpecT types to their iterator
 * positions.
 *
 * @details Given a tuple of unique IndexSpecT types, creates a compile-time
 * hana::map that maps each IndexSpecT type to its position (0-based index)
 * in the iterator tuple. This map is used to determine which iterator to use
 * for a given index during execution.
 *
 * @tparam IndexSpecs A std::tuple of unique IndexSpecT types
 */
template <tmp::CTuple IndexSpecs>
struct ComputeIndexSpecToIteratorPosMap;

template <typename... IndexSpecs>
  requires(CIndexSpec<IndexSpecs> && ...)
struct ComputeIndexSpecToIteratorPosMap<std::tuple<IndexSpecs...>> {
  // Create pairs for all IndexSpecT types with their positions
  template <std::size_t... Is>
  static constexpr auto make_map_impl(std::index_sequence<Is...>) {
    return hana::make_map(hana::make_pair(
        hana::type_c<std::tuple_element_t<Is, std::tuple<IndexSpecs...>>>, hana::int_c<Is>)...);
  }

  /// @brief The hana::map type mapping IndexSpecT types to iterator positions
  using type = decltype(make_map_impl(std::make_index_sequence<sizeof...(IndexSpecs)>{}));
};

/**
 * @brief Abstract Data Type (ADT) for a Pipeline of MIR operations.
 *
 * @details Represents a sequence of MIR operations compiled from a single
 * rule. A Pipeline encapsulates all the operations needed to execute one
 * rule, including joins, let clauses, and destination relations. Also stores
 * compile-time maps from variable types to stack positions and from
 * IndexSpecT types to iterator positions.
 *
 * @tparam MIROps A std::tuple of MIR operation types (ColumnSource,
 * ColumnJoin, DestinationRelation, etc.)
 * @tparam VarPosMap A compile-time hana::map mapping variable types to
 * positions
 * @tparam RequiredSourceSpecs Explicit tuple of IndexSpecT for sources
 * (default: empty)
 * @tparam DestinationSpecs Explicit tuple of IndexSpecT for destinations
 * (default: empty)
 */
template <tmp::CTuple MIROps, typename VarPosMap, typename RequiredSourceSpecs = std::tuple<>,
          typename DestinationSpecs = std::tuple<>, typename AllSchemas = std::tuple<>,
          typename RuleNameT = void>
struct Pipeline {
  /// @brief The tuple of MIR operations in this pipeline
  using mir_ops_type = MIROps;
  /// @brief The compile-time map from variable types to stack positions
  using var_pos_map_type = VarPosMap;
  /// @brief Explicit list of required source indices
  using source_specs_type = RequiredSourceSpecs;
  /// @brief Explicit list of destination indices
  using destination_specs_type = DestinationSpecs;
  /// @brief Explicit list of all unique schemas used in the pipeline
  using all_schemas_type = AllSchemas;
  /// @brief Rule name type for NVTX profiling (TString or void)
  using rule_name_type = RuleNameT;

  /// @brief Get rule name as const char* for NVTX (returns "unnamed" if RuleNameT is void)
  static const char* get_rule_name() {
    if constexpr (std::is_void_v<RuleNameT>) {
      return "unnamed";
    } else {
      // RuleNameT is a TString type, use tstring_to_string
      static const std::string name = AST::tstring_to_string<RuleNameT>();
      return name.c_str();
    }
  }
};
// Type predicate for pipeline (Scheme/Racket style: (pipeline? x))
DEFINE_TYPE_PREDICATE(is_pipeline, Pipeline, CPipeline,
                      (tmp::CTuple MIROps, typename VarPosMap, typename SourceSpecs,
                       typename DestSpecs, typename AllSchemas, typename RuleNameT),
                      (MIROps, VarPosMap, SourceSpecs, DestSpecs, AllSchemas, RuleNameT))

/// @brief Check if tuple contains more ColumnJoins
template <typename Ops>
struct HasMoreColumnJoins;

template <>
struct HasMoreColumnJoins<std::tuple<>> {
  static constexpr bool value = false;
};

template <typename Head, typename... Tail>
struct HasMoreColumnJoins<std::tuple<Head, Tail...>> {
  static constexpr bool value =
      is_column_join_v<Head> || HasMoreColumnJoins<std::tuple<Tail...>>::value;
};

/**
 * @brief Helper to find the first IndexSpec matching a given schema.
 */
template <CRelationSchema Schema, typename AllIndexSpecs>
struct FindFirstIndexSpecForSchema;

template <CRelationSchema Schema>
struct FindFirstIndexSpecForSchema<Schema, std::tuple<>> {
  using type = std::tuple<>;
};

template <CRelationSchema Schema, typename FirstSpec, typename... RestSpecs>
struct FindFirstIndexSpecForSchema<Schema, std::tuple<FirstSpec, RestSpecs...>> {
  using type = std::conditional_t<
      std::is_same_v<typename FirstSpec::schema_type, Schema>, FirstSpec,
      typename FindFirstIndexSpecForSchema<Schema, std::tuple<RestSpecs...>>::type>;
};

/**
 * @brief Helper to get canonical index spec for a schema.
 */
template <CRelationSchema Schema, std::size_t Ver = FULL_VER>
struct GetCanonicalIndexSpec {
  using canonical_seq = typename Schema::canonical_index_type;
  // Convert std::index_sequence to std::integer_sequence<int, ...>
  template <std::size_t... Is>
  static auto to_int_seq(std::index_sequence<Is...>)
      -> std::integer_sequence<int, static_cast<int>(Is)...>;
  using column_indexes_type = decltype(to_int_seq(canonical_seq{}));
  using type = IndexSpecT<Schema, column_indexes_type, Ver>;
};

/**
 * @brief Computes canonical index spec for a schema from AllIndexSpecs or
 * creates canonical.
 */
template <CRelationSchema Schema, typename AllIndexSpecs>
struct GetCanonicalIndexSpecForSchema {
 private:
  using FoundSpec = typename FindFirstIndexSpecForSchema<Schema, AllIndexSpecs>::type;

 public:
  using type =
      std::conditional_t<std::is_same_v<FoundSpec, std::tuple<>>,
                         typename GetCanonicalIndexSpec<Schema, FULL_VER>::type, FoundSpec>;
};

namespace detail {
/**
 * @brief Collects canonical index specs for incremental relations.
 */
template <typename IncrementalRelations, typename AllIndexSpecs>
struct CollectCanonicalIndexSpecs;

template <typename... Schemas, typename AllIndexSpecs>
struct CollectCanonicalIndexSpecs<std::tuple<Schemas...>, AllIndexSpecs> {
  // For each schema, get its canonical index spec
  template <typename Schema>
  using GetCanonicalSpec = typename GetCanonicalIndexSpecForSchema<Schema, AllIndexSpecs>::type;

  /// @brief Tuple of canonical IndexSpecT types (one per incremental relation)
  using type = std::tuple<GetCanonicalSpec<Schemas>...>;
};
}  // namespace detail

/// @brief Compute position map from unique types
template <typename Types>
struct ComputePositionMap;

template <typename... Types>
struct ComputePositionMap<std::tuple<Types...>> {
  template <std::size_t... Is>
  static constexpr auto make_map(std::index_sequence<Is...>) {
    return hana::make_map(hana::make_pair(hana::type_c<Types>, hana::int_c<Is>)...);
  }
  using type = decltype(make_map(std::make_index_sequence<sizeof...(Types)>{}));
};

/// @brief Extract ColumnSources that have a specific Var in PrefixVars
template <typename MIROps, typename Var>
struct ExtractColumnSourcesWithVarInPrefix;

template <typename Var>
struct ExtractColumnSourcesWithVarInPrefix<std::tuple<>, Var> {
  using type = std::tuple<>;
};

template <typename Head, typename... Tail, typename Var>
struct ExtractColumnSourcesWithVarInPrefix<std::tuple<Head, Tail...>, Var> {
 private:
  template <typename Op>
  struct Extract {
    using type = std::tuple<>;
  };

  template <typename IndexSpec, typename PrefixVars>
  struct Extract<ColumnSource<IndexSpec, PrefixVars>> {
    using type = std::conditional_t<tmp::contains_v<Var, PrefixVars>,
                                    std::tuple<ColumnSource<IndexSpec, PrefixVars>>, std::tuple<>>;
  };

  template <typename V, typename Sources, typename Strategy, std::size_t HS, typename VHM>
  struct Extract<ColumnJoin<V, Sources, Strategy, HS, VHM>> {
    using type = typename ExtractColumnSourcesWithVarInPrefix<Sources, Var>::type;
  };

  template <typename VarToBind, typename IndexSpec, typename PrefixVars, typename AggFunc,
            std::size_t HS, typename VHM>
  struct Extract<Aggregate<VarToBind, IndexSpec, PrefixVars, AggFunc, HS, VHM>> {
    using type = std::conditional_t<
        tmp::contains_v<Var, PrefixVars>,
        std::tuple<Aggregate<VarToBind, IndexSpec, PrefixVars, AggFunc, HS, VHM>>, std::tuple<>>;
  };

  using HeadSources = typename Extract<Head>::type;
  using TailSources = typename ExtractColumnSourcesWithVarInPrefix<std::tuple<Tail...>, Var>::type;

 public:
  using type = Mp::mp_append<HeadSources, TailSources>;
};

}  // namespace mir
}  // namespace SRDatalog