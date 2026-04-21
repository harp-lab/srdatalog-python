/**
 * @file query.h
 * @brief Query compilation and index management for Datalog programs.
 *
 * @details This file provides template metaprogramming utilities for:
 * - Compiling high-level AST rules into Mid-level IR (MIR) operations
 * - Computing required indexes for efficient join execution
 * - Building runtime indexes from compile-time index specifications
 * - Converting between compile-time schemas and runtime relation objects
 *
 * The compilation process transforms logical Datalog rules into column-oriented
 * join operations suitable for Worst-Case Optimal Join (WCOJ) algorithms.
 *
 * @note **C++20/23 Features Used**: This file uses C++20 concepts for type
 *       constraints, `if constexpr` for compile-time branch selection, and
 *       `requires` clauses for template constraints.
 */

#pragma once

#include "ast.h"
#include "mir.h"  // For MIR compilation functions (CompileRuleToMIR, etc.)
#include "tmp.h"
#include <boost/mp11.hpp>
#include <iostream>

/**
 * @defgroup QueryMacros Query Helper Macros
 * @brief Macros for defining relations and adding facts at runtime.
 * @{
 */

/**
 * @brief Define a relation variable with a given schema.
 *
 * @details Creates a type alias and runtime variable for a relation with the
 * specified semiring and attribute types. This macro simplifies relation
 * declaration by combining type definition and variable declaration.
 *
 * @param RELATION_NAME The name of the relation variable to create
 * @param SR The semiring type (e.g., BooleanSR, MinPlusSR)
 * @param ... Variadic list of attribute types (e.g., int, int for binary relation)
 *
 * @example
 * ```cpp
 * DEFINE_RELATION(edge, BooleanSR, int, int);
 * // Expands to:
 * // using edge_type = Relation<BooleanSR, int, int>;
 * // edge_type edge;
 * ```
 */
#define DEFINE_RELATION(RELATION_NAME, SR, ...)                                                    \
  using RELATION_NAME##_type = Relation<SR, std::tuple<__VA_ARGS__>>;                              \
  RELATION_NAME##_type RELATION_NAME;

/**
 * @brief Add a fact (tuple) to a relation with an annotation.
 *
 * @details Inserts a row into the relation with the given annotation value
 * from the semiring. The annotation is combined with any existing annotation
 * for the same tuple using the semiring's addition operation.
 *
 * @param RELATION_NAME The relation variable to add the fact to
 * @param ANNOTATION The semiring annotation value (e.g., BooleanSR::one())
 * @param ... Variadic list of attribute values matching the relation schema
 *
 * @example
 * ```cpp
 * FACT(edge, BooleanSR::one(), 1, 2);
 * // Adds tuple (1, 2) with annotation true to the edge relation
 * ```
 */
#define FACT(RELATION_NAME, ANNOTATION, ...) RELATION_NAME.push_row({__VA_ARGS__}, ANNOTATION);

/** @} */  // end of QueryMacros group

/**
 * @defgroup IndexMacros Index Access Macros
 * @brief Macros for accessing relation indexes using name mangling.
 *
 * @details These macros generate unique variable names for indexes based on
 * their column specifications. This enables compile-time name resolution for
 * index variables.
 * @{
 */

// Internal helper macros for argument counting and name mangling
#define COUNT_ARGS_IMPL(_1, _2, _3, _4, N, ...) N
#define COUNT_ARGS(...) COUNT_ARGS_IMPL(__VA_ARGS__, 4, 3, 2, 1)
#define CAT(a, b) a##b
#define PASTE(a, b) CAT(a, b)
#define PASTE_NAME_1(REL, a) REL##_##a##_idx
#define PASTE_NAME_2(REL, a, b) REL##_##a##_##b##_idx
#define PASTE_NAME_3(REL, a, b, c) REL##_##a##_##b##_##c##_idx
#define PASTE_NAME_4(REL, a, b, c, d) REL##_##a##_##b##_##c##_##d##_idx
#define PASTE_NAME_5(REL, a, b, c, d, e) REL##_##a##_##b##_##c##_##d##_##e##_idx
#define PASTE_NAME_6(REL, a, b, c, d, e, f) REL##_##a##_##b##_##c##_##d##_##e##_##f##_idx

#define DISPATCH_PASTE(REL, N, ...) PASTE(PASTE_NAME_, N)(REL, __VA_ARGS__)

/**
 * @brief Get the name-mangled index variable for a relation.
 *
 * @details Generates a unique variable name for an index based on the relation
 * name and column specification. The name follows the pattern
 * `RELATION_COL1_COL2_..._COLN_idx`.
 *
 * @param REL The relation variable name
 * @param ... Variadic list of column indices (up to 6 columns)
 *
 * @example
 * ```cpp
 * INDEX(edge, 0, 1)  // Generates: edge_0_1_idx
 * INDEX(path, 0)     // Generates: path_0_idx
 * ```
 */
#define INDEX(REL, ...) DISPATCH_PASTE(REL, COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

/**
 * @brief Define and ensure an index exists on a relation.
 *
 * @details Creates a const reference to an index, ensuring it exists on the
 * relation. If the index doesn't exist, it will be created. The index name
 * is generated using the INDEX macro.
 *
 * @param REL The relation variable name
 * @param ... Variadic list of column indices specifying the index key
 *
 * @example
 * ```cpp
 * DEFINE_INDEX(edge, 0, 1);  // Ensures edge has index on columns {0, 1}
 * ```
 */
#define DEFINE_INDEX(REL, ...)                                                                     \
  const auto& INDEX(REL, __VA_ARGS__) = REL.ensure_index({{__VA_ARGS__}});

/**
 * @brief Access the root node of a trie-based index.
 *
 * @details For trie-structured indexes, accesses the root node of the trie.
 * The root provides access to the entire index structure.
 *
 * @param IDX The index variable name (from INDEX macro)
 * @param ... Column indices (for name mangling, same as INDEX)
 *
 * @example
 * ```cpp
 * INDEX_ROOT(INDEX(edge, 0, 1), 0, 1)  // Accesses root of edge_0_1_idx
 * ```
 */
#define INDEX_ROOT(IDX, ...) PASTE(INDEX(IDX, __VA_ARGS__), _root)

/**
 * @brief Access a specific postfix node in a trie-based index.
 *
 * @details For trie-structured indexes, accesses a node at a specific path
 * defined by bound variables. The postfix is typically a name-mangled
 * representation of the bound variable values.
 *
 * @param IDX The index variable name (from INDEX macro)
 * @param POSTFIX The postfix identifier (e.g., "x_y" for variables x and y)
 * @param ... Column indices (for name mangling, same as INDEX)
 *
 * @example
 * ```cpp
 * INDEX_AT(INDEX(edge, 0, 1), x_y, 0, 1)  // Accesses edge_0_1_idx[x][y]
 * ```
 */
#define INDEX_AT(IDX, POSTFIX, ...) PASTE(INDEX(IDX, __VA_ARGS__), POSTFIX)

/** @} */  // end of IndexMacros group

namespace SRDatalog {

/**
 * @brief Query compilation namespace.
 *
 * @details Contains template metaprogramming utilities for compiling
 * high-level Datalog AST rules into Mid-level IR (MIR) operations and
 * managing indexes for efficient join execution.
 */
using namespace AST;

/**
 * @brief Join plan for Worst-Case Optimal Join (WCOJ) algorithms.
 *
 * @details A join plan specifies the order in which logical variables should
 * be bound during join execution. The naive join plan uses the appearance
 * order of variables in the rule body. More sophisticated plans can optimize
 * join order based on selectivity and data distribution.
 */

// MIR compilation functions (GetIndexForClause, MapClausesToSpecs_Impl, ComputeNaiveIndices)
// have been moved to mir.h

/**
 * @brief Computes indexes for multiple rules.
 *
 * @details Processes a list of rules and collects all required indexes,
 * flattening the result into a single tuple.
 *
 * @tparam RuleList A std::tuple of CRule types
 */
template <tmp::CTuple RuleList>
struct ComputeIndicesForRules {
  template <typename Rule>
  using GetIndices = typename mir::ComputeNaiveIndices<Rule>::type;
  using type = mp::mp_flatten<mp::mp_transform<GetIndices, RuleList>>;
};

/**
 * @brief Gets all IndexSpecs required by a list of queries.
 *
 * @details Extracts all rules from the queries (whether Fixpoint or
 * NonIterativeRuleSets) and computes their combined index requirements.
 *
 * @tparam QueryList A std::tuple of CQuery types (Fixpoint or
 *                   NonIterativeRuleSets)
 *
 * @example
 * ```cpp
 * using MyQueries = std::tuple<Fixpoint<Rule1, Rule2>>;
 * using AllIndexes = GetAllIndexSpecs<MyQueries>;
 * // Contains all IndexSpecs needed by Rule1 and Rule2
 * ```
 */
template <tmp::CTuple QueryList>
using GetAllIndexSpecs =
    typename ComputeIndicesForRules<typename GetAllRulesFromQueries<QueryList>::type>::type;

namespace detail {
/**
 * @brief Helper to find the 0-based index of a relation type in a database tuple.
 *
 * @details Locates the position of a relation type in the database schema tuple,
 * enabling runtime access to the corresponding relation object.
 *
 * @tparam T The relation type to find
 * @tparam Tuple The database schema tuple (std::tuple of RelationSchema types)
 */
template <typename T, typename Tuple>
struct IndexRelInDBRels {
  static constexpr std::size_t value = mp::mp_find<Tuple, T>::value;
  static_assert(value < mp::mp_size<Tuple>::value,
                "Relation type not found in provided runtime database tuple.");
};

/**
 * @brief Helper to check if a type is a reference to a Relation.
 *
 * @details Type trait to determine if a type is a reference to a Relation
 * template instantiation. Used for type checking in template metaprogramming.
 *
 * @tparam T The type to check
 */
template <typename T>
struct is_relation_ref : std::false_type {};
template <Semiring SR, ColumnElementTuple AttrTuple>
struct is_relation_ref<Relation<SR, AttrTuple>&> : std::true_type {};

}  // namespace detail

/**
 * @brief Gets a runtime relation object from a SemiNaiveDatabase using its schema.
 *
 * @details Retrieves a reference to a specific relation version (full, delta,
 * or newt) from a SemiNaiveDatabase based on the compile-time schema and version
 * number. This bridges the gap between compile-time type information and runtime
 * data structures.
 *
 * @note **C++17/C++20 feature**: Uses `if constexpr` for compile-time branch
 *       selection based on the version parameter. This enables efficient
 *       compile-time dispatch without runtime overhead.
 *
 * @tparam Schema The relation schema (compile-time ADT)
 * @tparam Ver The version number (0=full, 1=delta, 2=newt)
 * @tparam Database The SemiNaiveDatabase type
 *
 * @param rel_tuple Reference to the SemiNaiveDatabase instance
 * @return Reference to the requested relation version
 *
 * @note The version parameter uses numeric constants: 0=FULL_VER, 1=DELTA_VER, 2=NEW_VER
 */
template <CRelationSchema Schema, std::size_t Ver, CSemiNaiveDatabase Database>
inline auto& get_relation_by_schema(Database& rel_tuple) {
  constexpr std::size_t index =
      detail::IndexRelInDBRels<Schema, typename Database::schema_type>::value;
  // rel_tuple.verify_distinct_addresses();
  if constexpr (Ver == FULL_VER) {
    return std::get<index>(rel_tuple.full);
  } else if constexpr (Ver == DELTA_VER) {
    return std::get<index>(rel_tuple.delta);
  } else if constexpr (Ver == NEW_VER) {
    return std::get<index>(rel_tuple.newt);
  } else {
    static_assert(false, "Invalid version");
  }
}

/**
 * @brief Builds a single index on a relation object from an IndexSpec.
 *
 * @details Creates an index on the specified columns of a relation. The index
 * is built on the full version of the relation and is used for efficient join
 * execution. This function bridges compile-time index specifications with
 * runtime index construction.
 *
 * @tparam Schema The relation schema (compile-time ADT)
 * @tparam Cols Variadic list of column indices (as int template parameters)
 * @tparam RelationsTuple The type of the relations tuple (typically from std::tie)
 *
 * @param seq An std::integer_sequence containing the column indices
 * @param all_relations Reference to the tuple of relation objects
 *
 * @note Currently prints debug information about which index is being built.
 *       The index is always built on the FULL_VER relation.
 */
template <CRelationSchema Schema, std::size_t Ver, int... Cols, typename RelationsTuple>
void build_index_from_spec(std::integer_sequence<int, Cols...>, RelationsTuple& all_relations) {
  auto& target_relation = get_relation_by_schema<Schema, Ver>(all_relations);
  // debug print
  // std::cout << "  -> Building index for relation '" << Schema::name
  //           << "' on columns {";
  ((std::cout << Cols << ", "), ...);
  std::cout << "}" << std::endl;

  target_relation.ensure_index({Cols...});
}

namespace detail {
/**
 * @brief Helper functor to build a single index from an IndexSpec.
 *
 * @details Used with mp::mp_for_each to iterate over a list of IndexSpec types
 * and build each corresponding runtime index.
 *
 * @tparam AllRelationsTuple The type of the relations tuple
 */
template <typename AllRelationsTuple, std::size_t Ver>
struct BuildIndexHelper {
  AllRelationsTuple& all_relations;

  /**
   * @brief Builds an index from an IndexSpec type.
   *
   * @tparam IndexSpec The IndexSpec ADT type (compile-time)
   */
  template <typename IndexSpec>
  void operator()(IndexSpec) const {
    using Schema = IndexSpec::schema_type;
    using ColSeq = IndexSpec::column_indexes_type;
    build_index_from_spec<Schema, Ver>(ColSeq{}, all_relations);
  }
};

/**
 * @brief Implementation helper for building all indexes.
 *
 * @details Template metaprogramming helper that uses mp::mp_for_each to iterate
 * over all IndexSpec types and build their corresponding runtime indexes.
 *
 * @tparam IndexSpecList A std::tuple of IndexSpec types
 * @tparam AllRelationsTuple The type of the relations tuple
 */
template <typename IndexSpecList, std::size_t Ver, typename AllRelationsTuple>
struct BuildAllIndices_Impl {
  /**
   * @brief Builds all indexes specified in IndexSpecList.
   *
   * @param all_relations Reference to the tuple of relation objects
   */
  static void build(AllRelationsTuple& all_relations) {
    mp::mp_for_each<IndexSpecList>(BuildIndexHelper<AllRelationsTuple, Ver>{all_relations});
  }
};
}  // namespace detail

/**
 * @brief Builds all required runtime indexes from compile-time IndexSpecs.
 *
 * @details Iterates over a tuple of IndexSpec types (computed via template
 * metaprogramming) and builds each corresponding runtime index on the provided
 * relations. This function bridges compile-time index analysis with runtime
 * index construction.
 *
 * @tparam IndexSpecList The std::tuple of IndexSpec types (TMP result)
 * @tparam RelationsTuple A tuple of references to runtime Relation objects,
 *                        typically created with std::tie(r, s, ...)
 *
 * @param runtime_database Reference to the tuple of relation objects
 *
 * @example
 * ```cpp
 * using RequiredIndexes = GetAllIndexSpecs<MyQueries>;
 * auto relations = std::tie(edge, path);
 * build_all_indices_from_specs<RequiredIndexes>(relations);
 * ```
 */
template <typename IndexSpecList, std::size_t Ver, typename RelationsTuple>
void build_all_indices_from_specs(RelationsTuple& runtime_database) {
  std::cout << "Building all required indices..." << std::endl;

  // Start the TMP recursive "loop", passing the runtime DB tuple directly
  detail::BuildAllIndices_Impl<IndexSpecList, Ver, RelationsTuple>::build(runtime_database);

  std::cout << "Index building complete." << std::endl;
}

/**
 * @brief Converts a Database ADT to a tuple of relation types.
 *
 * @details Transforms a compile-time Database (tuple of RelationSchema types)
 * into a tuple of corresponding Relation types. This is used to create the
 * runtime data structure matching the compile-time schema.
 *
 * @tparam DB The Database ADT type (must satisfy CDatabase)
 */
template <CDatabase DB>
using ConvertDatabaseToRelationsTuple = mp::mp_transform<AST::SchemaToRelation, DB>;

/**
 * @brief Computes and builds all indexes required by a Program.
 *
 * @details Analyzes a Program ADT to determine all required indexes and
 * provides a method to build them on a runtime relations tuple.
 *
 * @tparam Program The Program ADT (must satisfy CProgram)
 * @tparam RelationsTuple The type of the relations tuple
 */
template <CProgram Program, typename RelationsTuple>
struct AllIndicesFromProgram {
  /// @brief The tuple of IndexSpec types required by the program
  using type = GetAllIndexSpecs<typename Program::query_list_type>;

  /**
   * @brief Builds all required indexes on the relations tuple.
   *
   * @param runtime_database Reference to the tuple of relation objects
   */
  template <std::size_t Ver>
  static void build(RelationsTuple& runtime_database) {
    build_all_indices_from_specs<type, Ver>(runtime_database);
  }
};

/**
 * @brief Creates a runtime database from a Program ADT.
 *
 * @details Instantiates all relations specified in the program's database schema
 * and builds all required indexes. This is a convenience function that combines
 * database creation and index building.
 *
 * @tparam Program The Program ADT (must satisfy CProgram)
 *
 * @return A tuple of Relation objects with all required indexes built
 *
 * @example
 * ```cpp
 * using MyProgram = Program<MyDB, std::tuple<MyQuery>>;
 * auto relations = build_database<MyProgram>();
 * // relations is a tuple of Relation objects with indexes ready
 * ```
 */
template <CProgram Program>
[[nodiscard]] inline auto build_database() {
  using RelationsTuple = typename ConvertDatabaseToRelationsTuple<typename Program::db_type>::type;
  RelationsTuple relations;
  AllIndicesFromProgram<Program, RelationsTuple>::build(relations);
  return relations;
}

// MIR compilation functions have been moved to mir.h
// The following functions are now in mir.h:
// - is_let_clause_for_var_impl, FindLetClauses, FindNormalClausesWithVar, FindIfClauses
// - ClauseToColumnSource_impl, MapClausesToColumnSources, GetVersionFromSources
// - FoldState, ProcessOneVariable, ProcessOneVariable_impl
// - CompileBodiesToMIR, CompileHeadsToMIR, CompileRuleToMIR

// End of moved MIR compilation functions

/**
 * @brief Extracts all relation schemas from rule head clauses.
 *
 * @details Analyzes a rule's head to determine which relations it writes to.
 * This is useful for dependency analysis and determining which relations are
 * derived (not base relations).
 *
 * @tparam Rule The rule to analyze (must satisfy CRule)
 *
 * @example
 * ```cpp
 * // Rule with head: Path(x,y), Result(x,y,z)
 * using HeadSchemas = GetHeadSchemas<MyRule>::type;
 * // HeadSchemas is std::tuple<PathSchema, ResultSchema>
 * ```
 */
template <CRule Rule>
struct GetHeadSchemas;

template <tmp::CTuple Head, tmp::CTuple Body, CJoinPlan JoinPlan>
struct GetHeadSchemas<Rule<Head, Body, JoinPlan>> {
  // Extract schema from each head clause
  template <typename HeadClause>
  using extract_schema = typename GetNormalClauseSchema<HeadClause>::type;

  /// @brief Tuple of relation schemas referenced in the rule head
  using type = mp::mp_transform<extract_schema, Head>;
};

/**
 * @brief Helper function to create an IndexSpec from a canonical index sequence.
 *
 * @details Creates an IndexSpec that indexes all columns of a relation in
 * canonical order (0, 1, 2, ..., arity-1). This is useful for creating
 * default indexes.
 *
 * @tparam Schema The relation schema
 * @tparam Ver The version number (default: FULL_VER)
 * @tparam Is The index sequence (0, 1, 2, ..., arity-1)
 *
 * @param seq An std::index_sequence representing the canonical column order
 * @return An IndexSpecT for the canonical index
 */
template <CRelationSchema Schema, std::size_t Ver = FULL_VER, std::size_t... Is>
auto make_indexSpec_from_canonical(std::index_sequence<Is...>)
    -> mir::IndexSpecT<Schema, std::integer_sequence<int, static_cast<int>(Is)...>, Ver>;

/**
 * @brief Type alias for the canonical index specification of a relation.
 *
 * @details Represents an index on all columns of a relation in canonical order.
 * This is the "full" index that can be used to scan the entire relation.
 *
 * @tparam Schema The relation schema
 * @tparam Ver The version number (default: FULL_VER)
 */
template <CRelationSchema Schema, std::size_t Ver = FULL_VER>
using CanonicalIndexSpec =
    decltype(make_indexSpec_from_canonical<Schema, Ver>(typename Schema::canonical_index_type{}));

}  // namespace SRDatalog
