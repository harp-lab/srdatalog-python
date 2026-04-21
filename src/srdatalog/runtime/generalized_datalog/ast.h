/**
 * @file ast.h
 * @brief Abstract Syntax Tree (AST) for Datalog using Template Metaprogramming.
 *
 * @details This file defines the compile-time AST structures for representing
 * Datalog programs in C++. The AST uses template metaprogramming to encode
 * programs as types, enabling:
 *
 * - Type-safe program representation
 * - Compile-time analysis and transformation
 * - Zero-overhead runtime execution
 * - Integration with C++ type system
 *
 * The AST includes:
 * - Basic constructs: Var, Const, CppExpr, Clause
 * - Rule structures: Rule, Fixpoint, NonIterativeRuleSets
 * - Program structure: Program, Database
 * - MIR (Mid-level IR): ColumnSource, ColumnJoin, DestinationRelation
 * - Template metaprogramming utilities for AST analysis
 *
 * All AST nodes are compile-time types, not runtime values. The actual
 * execution happens in runtime.h which interprets MIR operations.
 *
 * @note **C++20/23 Features Used**: This file extensively uses C++20 concepts
 *       for type constraints, `auto` non-type template parameters for lambda
 *       expressions, lambda in unevaluated contexts, fold expressions, and
 *       `requires` clauses for template constraints.
 */

#pragma once

#include "relation_col.h"
#include "semiring.h"
#include "tmp.h"
#include <tuple>
#include <type_traits>
// for boost::pmr allocator
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#ifndef BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#define BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#endif
#include <boost/hana/string.hpp>
#include <boost/mp11.hpp>

namespace Mp = boost::mp11;
static_assert(Mp::mp_size<Mp::mp_list<>>::value == 0, "mp alias sanity check");

#define CONST(V) SRDatalog::AST::Const<[]() { return V; }>

#define FULL_VER 0
#define DELTA_VER 1
#define NEW_VER 2
#define UNKNOWN_VER 10

namespace SRDatalog::AST {

using boost::container::pmr::memory_resource;

// #include <algorithm>

// Helper macros to unwrap parenthesized arguments
#define DEFINE_TYPE_PREDICATE_IMPL(PredName, VarName, ConceptName, TEMPLATE_PARAMS, TEMPLATE_ARGS) \
  template <typename T>                                                                            \
  struct PredName : std::false_type {};                                                            \
  template <TEMPLATE_PARAMS>                                                                       \
  struct PredName<VarName<TEMPLATE_ARGS>> : std::true_type {};                                     \
  template <typename T>                                                                            \
  inline constexpr bool PredName##_v = PredName<T>::value;                                         \
  template <typename T>                                                                            \
  concept ConceptName = PredName<T>::value;

// Macro to define type predicate boilerplate
// Usage: DEFINE_TYPE_PREDICATE(is_xxx, Xxx, CXxx, (template_params),
// (template_args)) For simple cases: DEFINE_TYPE_PREDICATE(is_var, Var, CVar,
// (CTString Name), (Name)) For parameter packs:
// DEFINE_TYPE_PREDICATE(is_tstring, TString, CTString, (char... chars),
// (chars...)) For multiple params: DEFINE_TYPE_PREDICATE(is_xxx, Xxx, CXxx, (T1
// p1, T2 p2), (p1, p2)) Note: Parentheses in arguments are stripped by the
// preprocessor when used in template contexts
#define STRIP_PARENS(...) __VA_ARGS__
#define DEFINE_TYPE_PREDICATE(PredName, VarName, ConceptName, TEMPLATE_PARAMS, TEMPLATE_ARGS)      \
  DEFINE_TYPE_PREDICATE_IMPL(PredName, VarName, ConceptName, STRIP_PARENS TEMPLATE_PARAMS,         \
                             STRIP_PARENS TEMPLATE_ARGS)

/**
 * @brief A C++20 utility to allow string literals as template parameters.
 *
 * @details This type alias allows using string literals as compile-time
 * template parameters. It leverages `boost::hana::string` to model compile-time
 * strings with rich functionality such as iteration and comparisons.
 *
 * @tparam chars The characters of the string literal
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using MyString = decltype("hello"_s);          // preferred literal form
 * using LegacyMyString = TString<'h','e','l','l','o'>; // legacy form
 * ```
 */
template <char... chars>
using TString = boost::hana::string<chars...>;

/// @brief Re-export Boost.Hana string literals for convenient usage.
namespace Literals {
using boost::hana::literals::operator""_s;
}  // namespace Literals

// Type predicate for TString (Scheme/Racket style: (string? x))
DEFINE_TYPE_PREDICATE(IsTstring, TString, CTString, (char... chars), (chars...))

// Helper struct for tstring_to_string (structs can be partially specialized)
template <CTString T>
struct TStringToStringImpl;

template <char... chars>
struct TStringToStringImpl<TString<chars...>> {
  static std::string get() {
    // char char_array[] = {chars..., '\0'};
    std::array<char, sizeof...(chars) + 1> char_array = {chars..., '\0'};
    return std::string(char_array.data());
  }
};

// function get string from TString
template <CTString T>
std::string tstring_to_string() {
  return TStringToStringImpl<T>::get();
}

/**
 * @brief Abstract Data Type (ADT) for a relation schema.
 *
 * @details Defines the compile-time structure of a relation, including its
 * name, semiring type, and attribute types. This is a template metaprogramming
 * construct used to represent database schemas at compile time.
 *
 * @tparam Name The compile-time string name of the relation
 * @tparam SR The semiring type used for annotations (e.g., BooleanSR,
 * MinPlusSR)
 * @tparam AttrTsTuple A std::tuple of attribute types (e.g., std::tuple<int,
 * int>) All elements must satisfy the ColumnElement concept.
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using EdgeSchema =
 *     RelationSchema<decltype("edge"_s), BooleanSR, std::tuple<int, int>>;
 * ```
 */
template <CTString Name, Semiring SR, ColumnElementTuple AttrTsTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType = HashTrieIndex,
          typename ValueType = uint32_t>
struct RelationSchema {
  /// @brief The arity (number of attributes) of the relation
  static constexpr std::size_t Arity = std::tuple_size_v<AttrTsTuple>;

  /// @brief The runtime relation type corresponding to this schema
  using relation_type = Relation<SR, AttrTsTuple, IndexType, HostRelationPolicy, ValueType>;
  /// @brief The name type of the relation
  using name_type = Name;
  /// @brief Type for interned values (size_t or uint32_t)
  using intern_value_type = ValueType;
  /// @brief The semiring type used for annotations
  using semiring_type = SR;
  /// @brief Tuple of attribute types
  using attr_ts_type = AttrTsTuple;

  /// @brief The index template type
  template <Semiring S, ColumnElementTuple A, typename... Args>
  using IndexTemplate = IndexType<S, A, Args...>;

  /// @brief the canonical index of the relation, a sequence from 0 to arity - 1
  using canonical_index_type = std::make_index_sequence<Arity>;
};

// Type predicate for RelationSchema (Scheme/Racket style: (relation-schema? x))
// Type predicate for RelationSchema
// Type predicate for RelationSchema
DEFINE_TYPE_PREDICATE(IsRelationSchema, RelationSchema, CRelationSchema,
                      (CTString Name, Semiring SR, ColumnElementTuple AttrTsTuple,
                       template <Semiring, ColumnElementTuple, typename...> class IndexType,
                       typename ValueType),
                      (Name, SR, AttrTsTuple, IndexType, ValueType))

/**
 * @brief Concise DSL alias for RelationSchema.
 */
template <CTString Name, Semiring SR, typename... Ts>
using relation = RelationSchema<Name, SR, std::tuple<Ts...>>;

template <CRelationSchema Schema>
using schema_rel_type = typename Schema::relation_type;

/**
 * @brief Abstract Data Type (ADT) for a logical variable.
 *
 * @details Represents a logical variable in Datalog rules. Variables are
 * identified by their compile-time string name and are used to represent
 * placeholders in clauses that will be bound during query execution.
 *
 * @tparam Name The compile-time string name of the variable (e.g., 'x', 'y')
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using VarX = Var<decltype("x"_s)>;
 * using VarY = Var<decltype("y"_s)>;
 * ```
 */
template <CTString Name>
struct Var {
  /// @brief The name type of the variable
  using name_type = Name;
};

/**
 * @brief Macro to define a Datalog variable concisely.
 * @details usage: DECLARE_VAR(x) -> using x_ = Var<decltype("x"_s)>
 */
#define DECLARE_VAR(name) using name##_ = Var<decltype(#name##_s)>

/**
 * @brief Abstract Data Type (ADT) for inline C++ expression in clause
 * arguments.
 *
 * @details Allows embedding arbitrary C++ lambda expressions within Datalog
 * rules. The expression can capture logical variables from the rule context and
 * perform arbitrary computations. Used for computed terms and let clauses.
 *
 * @note **C++17/C++20 feature**: Uses `auto` non-type template parameter to
 *       accept lambda expressions and function pointers as template arguments.
 *
 * @todo **Future Optimization**: Currently, argument types for encode/decode
 *       operations are deduced at compile-time from the function signature using
 *       SFINAE-based type extraction (e.g., via function_arg_types_t). This can
 *       result in long compile times, especially with complex function signatures
 *       or multiple decode operations. A future optimization could make the types
 *       explicit in the CppExpr template, e.g., by adding an additional template
 *       parameter:
 *       ```cpp
 *       template <typename CapturedVars, typename ArgTypes, auto Func>
 *       struct CppExpr;
 *       ```
 *       This would eliminate the need for compile-time type deduction and
 *       significantly improve compilation speed. The ArgTypes parameter would be
 *       a std::tuple of the actual argument types (e.g., std::tuple<int, int>)
 *       matching the function signature, allowing direct type extraction without
 *       SFINAE.
 *
 * @tparam CapturedVars A std::tuple of Var types that the expression depends on
 * @tparam Func A C++ lambda or function pointer that will be called at runtime
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using Expr = CppExpr<std::tuple<Var<decltype("x"_s)>>,
 *                      [](int x) { return x + 1; }>;
 * ```
 */
template <typename CapturedVars, auto func>
struct CppExpr {
  template <typename... Args>
#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
      auto
      operator()(Args&&... args) const {
    return func(std::forward<Args>(args)...);
  }
};

/**
 * @brief Abstract Data Type (ADT) for a compile-time constant.
 *
 * @details Represents a constant value in Datalog rules. A constant is a
 * no-capture C++ expression that evaluates to a compile-time value. This is
 * used for literal values in clauses.
 *
 * @note **C++17/C++20 feature**: Uses `auto` non-type template parameter to
 *       accept lambda expressions as template arguments. The lambda must be
 *       constexpr-evaluable.
 *
 * @tparam Func A no-capture C++ lambda that returns the constant value
 *
 * @example
 * ```cpp
 * using ConstFive = Const<[]() { return 5; }>;
 * // Or use the macro: CONST(5)
 * ```
 */
template <auto Func>
struct Const {
  /// @brief The compile-time constant value
  static constexpr auto value = Func();
};

// Type predicates (Scheme/Racket style: (var? x), (cpp-expr? x), (const? x))
DEFINE_TYPE_PREDICATE(is_var, Var, CVar, (CTString Name), (Name))

// function get string from Var
template <CVar V>
std::string var_to_string() {
  return tstring_to_string<typename V::name_type>();
}

template <typename T>
struct is_cpp_expr : std::false_type {};
template <typename... Vars, auto func>
struct is_cpp_expr<CppExpr<std::tuple<Vars...>, func>> : std::true_type {};
template <typename T>
inline constexpr bool is_cpp_expr_v = is_cpp_expr<T>::value;

DEFINE_TYPE_PREDICATE(IsConst, Const, CConst, (auto func), (func))

// Concepts for use in template parameters (compile-time predicates)
// CVar and CConst are defined by DEFINE_TYPE_PREDICATE macros above
// @note **C++20 feature**: Uses C++20 concepts for type constraints
template <typename T>
concept CCppExpr = is_cpp_expr<T>::value;
// Union type predicate: term is var OR cpp-expr OR const
// @note **C++20 feature**: Uses concept disjunction (||) to combine multiple
//       concepts
template <typename T>
concept CTerm = CVar<T> || CCppExpr<T> || CConst<T>;

// @brief Error helper template type for arity mismatch debugging
// @details Example: ArityMismatchError<"r", 2, 3>, replace when we have c++26
// error checking
// C++20: Using constexpr variable for cleaner syntax
template <CRelationSchema schema, std::size_t ExpectedArity, std::size_t ActualArity>
struct ArityMismatchError {
  static constexpr bool value = ExpectedArity == ActualArity;
};
template <CRelationSchema schema, std::size_t ExpectedArity, std::size_t ActualArity>
inline constexpr bool arity_mismatch_error_v =
    ArityMismatchError<schema, ExpectedArity, ActualArity>::value;

/**
 * @brief Abstract Data Type (ADT) for a normal clause (relation predicate).
 *
 * @details Represents a clause that references a relation with a list of terms.
 * Terms can be variables, constants, or C++ expressions. The clause arity must
 * match the relation schema arity. The version specifies which relation version
 * to access during semi-naive evaluation (FULL_VER, DELTA_VER, NEW_VER).
 *
 * @tparam schema The relation schema this clause references
 * @tparam Ver The version number for semi-naive evaluation (default: FULL_VER)
 * @tparam Vars The terms (variables, constants, or expressions) of the clause
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using EdgeClause = Clause<EdgeSchema,
 *                           FULL_VER,
 *                           Var<decltype("x"_s)>,
 *                           Var<decltype("y"_s)>>;
 * ```
 */
template <CRelationSchema schema, std::size_t Ver = FULL_VER, CTerm... Vars>
struct Clause {
  /// @brief The arity (number of terms) of this clause
  static constexpr std::size_t arity = sizeof...(Vars);
  /// @brief The version number for semi-naive evaluation
  static constexpr std::size_t version = Ver;
  /// @brief Compile-time assertion ensuring clause arity matches schema arity
  static_assert(arity_mismatch_error_v<schema, schema::Arity, arity>, "Clause arity mismatch");
};
// Type predicate for normal clause (Scheme/Racket style: (normal-clause? x))
DEFINE_TYPE_PREDICATE(is_normal_clause, Clause, CNormalClause,
                      (CRelationSchema schema, std::size_t Ver, CTerm... Vars),
                      (schema, Ver, Vars...))

/**
 * @brief Abstract Data Type (ADT) for a let clause.
 *
 * @details Represents a let binding that introduces a new variable bound to
 * the result of a C++ expression. The expression can reference previously bound
 * variables in the rule context.
 *
 * @tparam var The variable to bind (must be a Var type)
 * @tparam expr The C++ expression that computes the value (must be a CppExpr
 * type)
 *
 * @example
 * ```cpp
 * using LetClause = LetClause<Var<decltype("z"_s)>,
 *                             CppExpr<std::tuple<Var<decltype("x"_s)>,
 * Var<decltype("y"_s)>>,
 *                                     [](int x, int y) { return x + y; }>>;
 * ```
 */
template <CVar var, CCppExpr expr>
struct LetClause {};
// Type predicate for let clause (Scheme/Racket style: (let-clause? x))
DEFINE_TYPE_PREDICATE(is_let_clause, LetClause, CLetClause, (CVar var, CCppExpr expr), (var, expr))

/**
 * @brief Abstract Data Type (ADT) for an aggregation clause.
 *
 * @tparam ResultVar The variable that will hold the aggregation result.
 * @tparam AggFunc The aggregation functor type (e.g. AggCount).
 * @tparam RelSchema The schema of the relation being aggregated.
 * @tparam Ver The relation version (FULL_VER, DELTA_VER, etc.).
 * @tparam Terms The arguments (logic vars or constants) passed to the relation.
 */
template <CVar ResultVar, typename AggFuncT, CRelationSchema RelSchema, std::size_t Ver,
          typename... Terms>
struct AggClause {
  using result_var_type = ResultVar;
  using agg_func = AggFuncT;
  using relation_schema = RelSchema;
  static constexpr std::size_t version = Ver;
  using terms_type = std::tuple<ResultVar, Terms...>;
  using args_type = std::tuple<Terms...>;
};

// Type predicate for agg clause
DEFINE_TYPE_PREDICATE(is_agg_clause, AggClause, CAggClause,
                      (CVar ResultVar, typename AggFuncT, CRelationSchema RelSchema,
                       std::size_t Ver, typename... Terms),
                      (ResultVar, AggFuncT, RelSchema, Ver, Terms...))

/**
 * @brief Abstract Data Type (ADT) for a negation clause (Anti-Join).
 *
 * @tparam RelSchema The schema of the negated relation.
 * @tparam Ver The relation version.
 * @tparam Terms The arguments passed to the relation.
 */
template <CRelationSchema RelSchema, std::size_t Ver, typename... Terms>
struct NegClause {
  using relation_schema = RelSchema;
  static constexpr std::size_t version = Ver;
  using terms_type = std::tuple<Terms...>;
};

// Type predicate for neg clause
DEFINE_TYPE_PREDICATE(is_neg_clause, NegClause, CNegClause,
                      (CRelationSchema RelSchema, std::size_t Ver, typename... Terms),
                      (RelSchema, Ver, Terms...))

template <CCppExpr Expr>
struct IfClause {
  using expr_type = Expr;
};
// Type predicate for if clause (Scheme/Racket style: (if-clause? x))
DEFINE_TYPE_PREDICATE(is_if_clause, IfClause, CIfClause, (CCppExpr Expr), (Expr))

// Union type predicate: clause is let-clause OR normal-clause OR if-clause OR agg-clause OR
// neg-clause
template <typename T>
concept CClause =
    CLetClause<T> || CNormalClause<T> || CIfClause<T> || CAggClause<T> || CNegClause<T>;

/**
 * @brief Abstract Data Type (ADT) for a user-provided join plan.
 *
 * @details Specifies the order in which variables should be bound during join
 * execution. This allows users to control the join order for optimal
 * performance in Worst-Case Optimal Join (WCOJ) algorithms.
 *
 * @tparam Vars A std::tuple of Var types specifying the join order
 *
 * @example
 * ```cpp
 * using Plan = JoinPlan<std::tuple<Var<decltype("x"_s)>,
 *                                   Var<decltype("y"_s)>,
 *                                   Var<decltype("z"_s)>>>;
 * ```
 */
template <tmp::CTuple Vars>
struct JoinPlan {
  /// @brief The tuple of variables in join order
  using vars_type = Vars;
};
// Type predicate for join plan (Scheme/Racket style: (join-plan? x))
DEFINE_TYPE_PREDICATE(is_join_plan, JoinPlan, CJoinPlan, (tmp::CTuple Vars), (Vars))

/**
 * @brief Abstract Data Type (ADT) for a Datalog rule.
 *
 * @details Represents a Datalog rule with a head (conclusion) and body
 * (premises). The head is a tuple of clauses that define what facts to derive,
 * and the body is a tuple of clauses (normal or let) that define the
 * conditions. The join plan specifies the variable binding order for efficient
 * join execution.
 *
 * @tparam Head A std::tuple of Clause types representing the rule head
 * @tparam Body A std::tuple of Clause types (normal or LetClause) representing
 * the rule body
 * @tparam JoinPlan A JoinPlan specifying the variable binding order
 *
 * @example
 * ```cpp
 * using MyRule = Rule<
 *   std::tuple<Clause<PathSchema, VarX, VarY>>,  // head
 *   std::tuple<Clause<EdgeSchema, VarX, VarY>>,  // body
 *   JoinPlan<std::tuple<VarX, VarY>>              // join plan
 * >;
 * ```
 */
template <tmp::CTuple Head, tmp::CTuple Body, CJoinPlan JoinPlan>
struct Rule {
  /// @brief Number of clauses in the rule head
  static constexpr std::size_t head_size = std::tuple_size_v<Head>;
  /// @brief Number of clauses in the rule body
  static constexpr std::size_t body_size = std::tuple_size_v<Body>;
  /// @brief The join plan variable ordering
  using join_plan_type = JoinPlan::vars_type;

  // Pipe operator to override join plan
  template <CJoinPlan NewPlan>
  constexpr auto operator|(NewPlan) const {
    return Rule<Head, Body, NewPlan>{};
  }
};
// Type predicate for rule (Scheme/Racket style: (rule? x))
DEFINE_TYPE_PREDICATE(is_rule, Rule, CRule,
                      (tmp::CTuple Head, tmp::CTuple Body, CJoinPlan JoinPlan),
                      (Head, Body, JoinPlan))

/**
 * @brief Abstract Data Type (ADT) for a fixpoint computation.
 *
 * @details Represents a set of recursive rules that are evaluated until a
 * fixpoint is reached. The rules are repeatedly applied using semi-naive
 * evaluation until no new facts are derived.
 *
 * @tparam Rules Variadic list of CRule types to evaluate until fixpoint
 *
 * @example
 * ```cpp
 * using TransitiveClosure = Fixpoint<
 *   Rule<...>,  // Rule 1
 *   Rule<...>   // Rule 2
 * >;
 * ```
 */
template <CRule... Rules>
struct Fixpoint {
  /// @brief Number of rules in this fixpoint
  static constexpr std::size_t rule_size = sizeof...(Rules);
};

/**
 * @brief Helper struct to wrap clause creation for a relation schema.
 */
template <CRelationSchema S>
struct Rel {
  template <CTerm... Ts>
  using Full = Clause<S, FULL_VER, Ts...>;
  template <CTerm... Ts>
  using Delta = Clause<S, DELTA_VER, Ts...>;
  template <CTerm... Ts>
  using New = Clause<S, NEW_VER, Ts...>;
};

template <typename... Clauses>
using Head = std::tuple<Clauses...>;
template <typename... Clauses>
using Body = std::tuple<Clauses...>;
template <CVar... Vars>
using Plan = JoinPlan<std::tuple<Vars...>>;

// Type predicate for fixpoint (Scheme/Racket style: (fixpoint? x))
DEFINE_TYPE_PREDICATE(is_fixpoint, Fixpoint, CFixpoint, (CRule... Rules), (Rules...))

/// @brief A util template to convert Fixpoint to mp11 list
template <CFixpoint FP>
struct FixpointToMp11List;
template <CRule... Rules>
struct FixpointToMp11List<Fixpoint<Rules...>> {
  using type = mp::mp_list<Rules...>;
};

/**
 * @brief Extracts the tuple of rules from a Fixpoint ADT.
 *
 * @details Template metaprogramming trait that unwraps a Fixpoint to get
 * the underlying tuple of rules. Used during query analysis and compilation.
 *
 * @tparam FP The Fixpoint type (must satisfy CFixpoint)
 */
template <CFixpoint FP>
struct GetRulesFromFixpoint;
template <CRule... Rules>
struct GetRulesFromFixpoint<Fixpoint<Rules...>> {
  /// @brief The tuple of rules contained in the fixpoint
  using type = std::tuple<Rules...>;
};

/**
 * @brief Abstract Data Type (ADT) for non-iterative rule sets.
 *
 * @details Represents a set of rules that are evaluated once (non-recursively).
 * Unlike Fixpoint, these rules do not iterate until fixpoint - they are
 * evaluated exactly once.
 *
 * @tparam Rules Variadic list of rule types to evaluate once
 *
 * @example
 * ```cpp
 * using SinglePass = NonIterativeRuleSets<
 *   Rule<...>,  // Rule 1
 *   Rule<...>   // Rule 2
 * >;
 * ```
 */
template <typename... Rules>
struct NonIterativeRuleSets {
  /// @brief Number of rules in this rule set
  static constexpr std::size_t rule_size = sizeof...(Rules);
};
// Type predicate for non-iterative rule sets (Scheme/Racket style predicate)
DEFINE_TYPE_PREDICATE(is_non_iterative_rule_sets, NonIterativeRuleSets, CNonIterativeRuleSets,
                      (typename... Rules), (Rules...))

/**
 * @brief Extracts the tuple of rules from a NonIterativeRuleSets ADT.
 *
 * @details Template metaprogramming trait that unwraps a NonIterativeRuleSets
 * to get the underlying tuple of rules. Used during query analysis and
 * compilation.
 *
 * @tparam Rules The NonIterativeRuleSets type (must satisfy
 * CNonIterativeRuleSets)
 */
template <CNonIterativeRuleSets Rules>
struct GetRulesFromNonIterativeRuleSets;
template <typename... Rules>
struct GetRulesFromNonIterativeRuleSets<NonIterativeRuleSets<Rules...>> {
  /// @brief The tuple of rules contained in the rule set
  using type = std::tuple<Rules...>;
};

// Union type predicate: query is fixpoint OR non-iterative-rule-sets
template <typename T>
concept CQuery = CFixpoint<T> || CNonIterativeRuleSets<T>;

/**
 * @brief Abstract Data Type (ADT) for a database schema.
 *
 * @details Represents the compile-time schema of a database as a tuple of
 * relation schemas. This defines the structure of all relations in the
 * database.
 *
 * @tparam Schemas Variadic list of CRelationSchema types
 *
 * @example
 * ```cpp
 * using namespace SRDatalog::AST::Literals;
 * using MyDB = Database<
 *   RelationSchema<decltype("edge"_s), BooleanSR, std::tuple<int, int>>,
 *   RelationSchema<decltype("path"_s), BooleanSR, std::tuple<int, int>>
 * >;
 * ```
 */
template <CRelationSchema... Schemas>
using Database = std::tuple<Schemas...>;

/**
 * @brief Concise DSL alias for Database.
 */
template <CRelationSchema... Schemas>
using database = Database<Schemas...>;

// Type predicate for database (Scheme/Racket style: (database? x))
template <typename T>
struct is_database : std::false_type {};
template <CRelationSchema... Schemas>
struct is_database<Database<Schemas...>> : std::true_type {};
template <typename T>
inline constexpr bool is_database_v = is_database<T>::value;
template <typename T>
concept CDatabase = is_database<T>::value;

/**
 * @brief Abstract Data Type (ADT) for a complete Datalog program.
 *
 * @details Represents a full Datalog program consisting of a database schema
 * and a list of queries (fixpoints or non-iterative rule sets) to execute.
 * This is the top-level AST structure that encapsulates the entire program.
 *
 * @tparam db The database schema (CDatabase type)
 * @tparam QueryList A std::tuple of CQuery types (Fixpoint or
 * NonIterativeRuleSets)
 *
 * @example
 * ```cpp
 * using MyProgram = Program<
 *   Database<EdgeSchema, PathSchema>,
 *   std::tuple<Fixpoint<MyRule>, NonIterativeRuleSets<OtherRule>>
 * >;
 * ```
 */
template <CDatabase db, tmp::CTuple QueryList>
struct Program {
  /// @brief The database schema type
  using db_type = db;
  /// @brief The list of queries to execute
  using query_list_type = QueryList;
};

// type fn take a database type and return tuple of relation types

// Type predicate for program (Scheme/Racket style: (program? x))
DEFINE_TYPE_PREDICATE(is_program, Program, CProgram, (CDatabase db, tmp::CTuple QueryList),
                      (db, QueryList))

/**
 * @brief Converts a Database ADT to a tuple of relation types.
 *
 * @details Template metaprogramming trait that extracts the relation_type
 * from each RelationSchema in a Database, producing a tuple of Relation types.
 * This is used to create runtime data structures matching compile-time schemas.
 *
 * @tparam DB The Database ADT type (must satisfy CDatabase)
 */
template <CDatabase DB>
struct DatabaseToRelationTypes;
template <CRelationSchema... Schemas>
struct DatabaseToRelationTypes<Database<Schemas...>> {
  /// @brief Tuple of Relation types corresponding to the database schemas
  using type = std::tuple<typename Schemas::relation_type...>;
};

/**
 * @brief Helper struct for initializing a database with a custom memory
 * resource.
 *
 * @details Template metaprogramming helper that creates a tuple of Relation
 * objects, each initialized with the provided memory resource. This enables
 * custom memory management for database relations.
 *
 * @tparam DB The Database ADT type (must satisfy CDatabase)
 */
template <typename Schema>
using SchemaToRelation = typename Schema::relation_type;

template <CDatabase DB, template <typename> typename RelationMapper>
struct InitDatabaseHelper;

template <CRelationSchema... Schemas, template <typename> typename RelationMapper>
struct InitDatabaseHelper<Database<Schemas...>, RelationMapper> {
  template <typename RelType>
  static RelType create_relation(std::string name, memory_resource* resource, std::size_t version) {
    if constexpr (std::is_constructible_v<RelType, std::string, memory_resource*, std::size_t>) {
      return RelType(name, resource, version);
    } else {
      RelType r;
      if constexpr (requires(RelType& x) { x.set_name(std::string{}); }) {
        r.set_name(name);
      }
      if constexpr (requires(RelType& x) { x.set_version(std::size_t{}); }) {
        r.set_version(version);
      }
      return r;
    }
  }

  static auto apply(memory_resource* resource, std::size_t version = FULL_VER) {
    return std::make_tuple(create_relation<RelationMapper<Schemas>>(
        tstring_to_string<typename Schemas::name_type>(), resource, version)...);
  }
};

/**
 * @brief Initializes a database with a custom memory resource.
 *
 * @details Convenience function that creates all relations in a database schema
 * using the provided memory resource. This is used to create runtime database
 * instances with custom memory management.
 *
 * @tparam DB The Database ADT type (must satisfy CDatabase)
 * @param resource The memory resource to use for all relations
 * @param version The version number to assign to all relations (default: FULL_VER)
 * @return A tuple of Relation objects initialized with the memory resource
 */
template <CDatabase DB, template <typename> typename RelationMapper = SchemaToRelation>
auto init_database(memory_resource* resource, std::size_t version = FULL_VER) {
  return InitDatabaseHelper<DB, RelationMapper>::apply(resource, version);
}

// --- Utils ---

/**
 * @brief Template metaprogramming trait to extract variables from a term.
 *
 * @details Given a term (Var, Const, or CppExpr), returns a std::tuple of
 * the variables that term depends on. Used for dependency analysis during
 * compilation.
 *
 * @tparam T The term type (must satisfy CTerm concept)
 *
 * @note
 * - For Var: returns std::tuple<Var>
 * - For Const: returns std::tuple<> (constants have no dependencies)
 * - For CppExpr: returns the CapturedVars tuple
 */
template <CTerm T>
struct GetVarsFromTerm;
/// @brief Specialization for Var: returns the variable itself
template <CVar V>
struct GetVarsFromTerm<V> {
  using type = std::tuple<V>;
};
/// @brief Specialization for Const: no variables
template <CConst C>
struct GetVarsFromTerm<C> {
  using type = std::tuple<>;
};
/// @brief Specialization for CppExpr: returns captured variables
template <tmp::CTuple CVars, auto F>
struct GetVarsFromTerm<CppExpr<CVars, F>> {
  using type = CVars;
};

/**
 * @brief Template metaprogramming trait to extract all variables from a list of
 * terms.
 *
 * @details Recursively extracts variables from each term in a tuple and
 * combines them into a unique tuple. Used to find all variables referenced in a
 * clause.
 *
 * @tparam Terms A std::tuple of CTerm types
 *
 * @note The result is deduplicated using TupleUnique, so each variable appears
 *       at most once in the result.
 */
template <tmp::CTuple Terms>
struct GetVarsFromTerms {
  template <typename Term>
  using ExtractVars = typename GetVarsFromTerm<Term>::type;

  using type = tmp::TupleUnique<mp::mp_apply<mp::mp_append, mp::mp_transform<ExtractVars, Terms>>>;
};

/**
 * @brief Template metaprogramming trait to extract terms from a normal clause.
 *
 * @details Extracts the tuple of terms (variables, constants, expressions) from
 * a normal Clause ADT. This is used to access the arguments of a relation
 * predicate.
 *
 * @tparam C The clause type (must satisfy CNormalClause concept)
 */
template <CNormalClause C>
struct GetNormalClauseTerms;
/// @brief Specialization for Clause: extracts the variadic terms
template <CRelationSchema S, std::size_t Ver, CTerm... Vars>
struct GetNormalClauseTerms<Clause<S, Ver, Vars...>> {
  using type = std::tuple<Vars...>;
};

/**
 * @brief Template metaprogramming trait to extract the schema from a normal
 * clause.
 *
 * @details Extracts the RelationSchema type from a normal Clause ADT. This is
 * used to identify which relation a clause references.
 *
 * @tparam C The clause type (must satisfy CNormalClause concept)
 */
template <CNormalClause C>
struct GetNormalClauseSchema;
/// @brief Specialization for Clause: extracts the schema parameter
template <CRelationSchema S, std::size_t Ver, CTerm... Vars>
struct GetNormalClauseSchema<Clause<S, Ver, Vars...>> {
  using type = S;
};

/**
 * @brief Template metaprogramming trait to extract the version from a normal
 * clause.
 *
 * @details Extracts the version number from a normal Clause ADT. This is
 * used to determine which relation version to access during semi-naive
 * evaluation.
 *
 * @tparam C The clause type (must satisfy CNormalClause concept)
 */
template <CNormalClause C>
struct GetNormalClauseVersion;
/// @brief Specialization for Clause: extracts the version parameter
template <CRelationSchema S, std::size_t Ver, CTerm... Vars>
struct GetNormalClauseVersion<Clause<S, Ver, Vars...>> {
  static constexpr std::size_t value = Ver;
};

// TODO: check if this is needed
/**
 * @brief Trait to extract the dependency list (CapturedVars)
 * from a CCppExpr ADT.
 */
template <CCppExpr Expr>
struct GetExprDependencies;

/**
 * @brief Partial specialization to pattern-match the CppExpr ADT
 * and extract the first template parameter.
 */
template <tmp::CTuple CapturedVars, auto Func>
struct GetExprDependencies<CppExpr<CapturedVars, Func>> {
  /**
   * @brief The std::tuple<Var...> of variables that the
   * CppExpr's lambda depends on.
   */
  using type = CapturedVars;
};

/**
 * @brief Template metaprogramming trait to extract variables from a LetClause.
 *
 * @details Extracts all variables that a LetClause depends on: both the
 * variable being bound and the variables captured by its C++ expression. Used
 * for dependency analysis during compilation.
 *
 * @tparam LC The LetClause type (must satisfy CLetClause concept)
 *
 * @note A LetClause introduces its bound variable AND its dependencies, so both
 *       are included in the result.
 */
template <CLetClause LC>
struct GetLetClauseVars;
/// @brief Specialization for LetClause: combines bound var and captured vars
template <CVar V, tmp::CTuple CVars, auto F>
struct GetLetClauseVars<LetClause<V, CppExpr<CVars, F>>> {
  // A LetClause introduces its bound var AND its dependencies
  using type = tmp::TupleUnique<tmp::TupleConcat<std::tuple<V>, CVars>>;
};

/**
 * @brief Template metaprogramming trait to extract variables from a IfClause.
 *
 * @details Extracts all variables that an IfClause depends on: the variables
 * captured by its C++ expression. Used for dependency analysis during
 * compilation.
 *
 * @tparam IC The IfClause type (must satisfy CIfClause concept)
 */
template <CIfClause IC>
struct GetIfClauseVars;
/// @brief Specialization for IfClause: extracts vars from CppExpr
template <CCppExpr Expr>
struct GetIfClauseVars<IfClause<Expr>> {
  // An IfClause depends on the variables captured by its CppExpr
  using type = typename GetExprDependencies<Expr>::type;
};

/**
 * @brief Template metaprogramming trait to extract variables from any clause
 * type.
 *
 * @details Unified interface for extracting variables from normal clauses,
 * let clauses, and if clauses. Dispatches to the appropriate specialization
 * based on clause type.
 *
 * @tparam C The clause type (must satisfy CClause concept)
 */
template <CClause C>
struct GetClauseVars;
/// @brief Specialization for normal clauses: extracts vars from terms
template <CNormalClause C>
struct GetClauseVars<C> {
  using type = typename GetVarsFromTerms<typename GetNormalClauseTerms<C>::type>::type;
};
/// @brief Specialization for let clauses: uses LetClause-specific extraction
template <CLetClause C>
struct GetClauseVars<C> {
  using type = typename GetLetClauseVars<C>::type;
};
/// @brief Specialization for if clauses: uses IfClause-specific extraction
template <CIfClause C>
struct GetClauseVars<C> {
  using type = typename GetIfClauseVars<C>::type;
};

/// @brief Specialization for agg clauses: extracts result var + relation terms
template <CAggClause C>
struct GetClauseVars<C> {
  // C is AggClause<ResultVar, AggFunc, RelSchema, Ver, Terms...>
  // We skip extracting type params and get variables from Terms + ResultVar
  template <typename ResultVar, typename AggFunc, typename RelSchema, std::size_t Ver,
            typename... Terms>
  struct Extract {
    using TermVars = typename GetVarsFromTerms<std::tuple<Terms...>>::type;
    using type = tmp::TupleUnique<tmp::TupleConcat<std::tuple<ResultVar>, TermVars>>;
  };
  // Apply Extract to the concrete AggClause type
  // We assume C is AggClause<...> so we can partial specialize?
  // Or we can just use a helper trait.
  template <typename T>
  struct Helper;
  template <typename RV, typename AF, typename RS, std::size_t V, typename... Ts>
  struct Helper<AggClause<RV, AF, RS, V, Ts...>> {
    using type = typename Extract<RV, AF, RS, V, Ts...>::type;
  };
  using type = typename Helper<C>::type;
};

/// @brief Specialization for neg clauses: extracts vars from negation terms
template <CNegClause C>
struct GetClauseVars<C> {
  // C is NegClause<RelSchema, Ver, Terms...>
  template <typename RelSchema, std::size_t Ver, typename... Terms>
  struct Extract {
    using type = typename GetVarsFromTerms<std::tuple<Terms...>>::type;
  };
  template <typename T>
  struct Helper;
  template <typename RS, std::size_t V, typename... Ts>
  struct Helper<NegClause<RS, V, Ts...>> {
    using type = typename Extract<RS, V, Ts...>::type;
  };
  using type = typename Helper<C>::type;
};
/**
 * @brief Finds the index of a term in a term list.
 *
 * @details Template metaprogramming trait that locates a term's position in
 * a tuple of terms. Returns -1 if the term is not found. Used during clause
 * analysis to map variables to column positions.
 *
 * @tparam T The term to find (must satisfy CTerm)
 * @tparam TermList A std::tuple of term types
 */
template <CTerm T, tmp::CTuple TermList>
struct FindTermIndex {
  static constexpr std::size_t idx = mp::mp_find<TermList, T>::value;
  /// @brief The index of the term, or -1 if not found
  static constexpr int value = (idx < mp::mp_size<TermList>::value) ? static_cast<int>(idx) : -1;
};

/**
 * @brief Maps variables to their column indexes in a clause.
 *
 * @details Template metaprogramming trait that converts a tuple of variables
 * (in access order) to an std::integer_sequence of column indexes. Each
 * variable is mapped to its position in the clause's term list. Used to
 * determine which columns to index for efficient access.
 *
 * @tparam ClauseTerms A std::tuple of terms from a clause
 * @tparam VarAccessOrder A std::tuple of Var types specifying the access order
 *
 * @example
 * ```cpp
 * // Clause: Edge(x, y) with terms [x, y]
 * // VarAccessOrder: [x, y]
 * // Result: std::integer_sequence<int, 0, 1>
 * ```
 */
template <tmp::CTuple ClauseTerms, tmp::CTuple VarAccessOrder>
struct MapVarsToColumnIndexes {
  template <typename Var>
  using VarToIndex = mp::mp_int<FindTermIndex<Var, ClauseTerms>::value>;

  using IndexList = mp::mp_transform<VarToIndex, VarAccessOrder>;

  template <typename... Ints>
  static auto to_seq(mp::mp_list<Ints...>) -> std::integer_sequence<int, Ints::value...>;

  /// @brief The integer sequence of column indexes
  using type = decltype(to_seq(mp::mp_rename<IndexList, mp::mp_list>{}));
};

/**
 * @brief Extracts the tuple of rules from a Query ADT.
 *
 * @details Template metaprogramming trait that unwraps either a Fixpoint
 * or NonIterativeRuleSets to get the underlying tuple of rules. Provides a
 * unified interface for extracting rules from different query types.
 *
 * @tparam Q The query type (must satisfy CQuery)
 */
template <CQuery Q>
struct GetRulesFromQuery;
/// @brief Specialization for Fixpoint: extracts rules tuple
template <CRule... Rules>
struct GetRulesFromQuery<Fixpoint<Rules...>> {
  using type = std::tuple<Rules...>;
};
/// @brief Specialization for NonIterativeRuleSets: extracts rules tuple
template <CRule... Rules>
struct GetRulesFromQuery<NonIterativeRuleSets<Rules...>> {
  using type = std::tuple<Rules...>;
};

/**
 * @brief Extracts all rules from a list of queries.
 *
 * @details Template metaprogramming trait that processes a tuple of queries
 * (Fixpoint or NonIterativeRuleSets) and collects all their rules into a
 * single flattened tuple. Used during program analysis to find all rules
 * that need to be compiled.
 *
 * @tparam QueryList A std::tuple of CQuery types
 */
template <tmp::CTuple QueryList>
struct GetAllRulesFromQueries {
  template <typename Query>
  using ExtractRules = typename GetRulesFromQuery<Query>::type;

  /// @brief Flattened tuple of all rules from all queries
  using type = mp::mp_apply<mp::mp_append, mp::mp_transform<ExtractRules, QueryList>>;
};

// MIR definitions (IndexSpecT, Version, mir namespace) have been moved to mir.h
// Include mir.h in files that need MIR definitions

/**
 * @brief Runtime database structure for semi-naive evaluation.
 *
 * @details Represents a database with three versions of each relation:
 * - full: The complete relation (all facts derived so far)
 * - delta: Facts derived in the previous iteration
 * - newt: Facts derived in the current iteration
 *
 * This structure enables semi-naive evaluation, an optimization for recursive
 * rules that avoids redundant computation by only processing new facts.
 *
 * @tparam DBType The compile-time database schema type (CDatabase)
 * @tparam AllocFull The allocator type for the full relations
 * @tparam AllocDelta The allocator type for the delta relations
 * @tparam AllocNewt The allocator type for the new relations
 */
template <CDatabase DBType, template <typename> typename RelationMapper = SchemaToRelation>
struct SemiNaiveDatabase {
  /// @brief The database schema type
  using schema_type = DBType;
  /// @brief The tuple type of all relation types in the database
  using data_type = mp::mp_transform<RelationMapper, DBType>;
  /// @brief The full relations (all facts)
  data_type full;
  /// @brief The delta relations (previous iteration)
  data_type delta;
  /// @brief The new relations (current iteration)
  data_type newt;

  // constructor
  SemiNaiveDatabase() {
    full = init_database<DBType, RelationMapper>(boost::container::pmr::get_default_resource(),
                                                 FULL_VER);
    delta = init_database<DBType, RelationMapper>(boost::container::pmr::get_default_resource(),
                                                  DELTA_VER);
    newt = init_database<DBType, RelationMapper>(boost::container::pmr::get_default_resource(),
                                                 NEW_VER);
  }
  SemiNaiveDatabase(memory_resource* full_resource, memory_resource* delta_resource,
                    memory_resource* newt_resource) {
    full = init_database<DBType, RelationMapper>(full_resource, FULL_VER);
    delta = init_database<DBType, RelationMapper>(delta_resource, DELTA_VER);
    newt = init_database<DBType, RelationMapper>(newt_resource, NEW_VER);
  }

  // Move constructor - explicitly defined to ensure proper move semantics
  SemiNaiveDatabase(SemiNaiveDatabase&& other) noexcept
      : full(std::move(other.full)), delta(std::move(other.delta)), newt(std::move(other.newt)) {}

  // Move assignment operator
  SemiNaiveDatabase& operator=(SemiNaiveDatabase&& other) noexcept {
    if (this != &other) {
      full = std::move(other.full);
      delta = std::move(other.delta);
      newt = std::move(other.newt);
    }
    return *this;
  }

  void verify_distinct_addresses() const {
    std::cout << "SemiNaiveDatabase constructed:" << std::endl;
    std::cout << "  full addr:  " << &full << std::endl;
    std::cout << "  delta addr: " << &delta << std::endl;
    std::cout << "  newt addr:  " << &newt << std::endl;

    constexpr std::size_t N = std::tuple_size_v<data_type>;
    mp::mp_for_each<mp::mp_iota_c<N>>([&](auto I) {
      std::cout << "  Relation " << I << ":" << std::endl;
      std::cout << "    full:  " << &std::get<I>(full) << std::endl;
      std::cout << "    delta: " << &std::get<I>(delta) << std::endl;
      std::cout << "    newt:  " << &std::get<I>(newt) << std::endl;

      // Verify they're different
      assert(&std::get<I>(full) != &std::get<I>(delta));
      assert(&std::get<I>(delta) != &std::get<I>(newt));
      assert(&std::get<I>(full) != &std::get<I>(newt));
    });
  }
};

/**
 * @brief Initialize a destination database from a source database.
 *
 * @details Iterates over all relations in full, delta, and newt versions
 *          and initializes the destination relation from the source relation.
 *          This relies on the destination relation type having a static `to_device`
 *          method (or compatible API) if it's a device relation, or copy constructor/assignment.
 *
 *          TODO: Standardize the initialization API (e.g. `from()` or `copy_from()`).
 *          Currently we use `to_device` for Host->Device.
 *
 * @tparam SrcDB The source database type
 * @tparam DstDB The destination database type
 */
template <typename SrcDB, typename DstDB>
struct DatabaseInitializer {
  template <std::size_t Ver>
  static void copy_tuple(SrcDB& src, DstDB& dst) {
    using SchemaTuple = typename SrcDB::schema_type;
    constexpr std::size_t N = std::tuple_size_v<SchemaTuple>;

    mp::mp_for_each<mp::mp_iota_c<N>>([&](auto I) {
      using Schema = std::tuple_element_t<I, SchemaTuple>;
      // Access source and dest relations by index/version
      // We can use get_relation_by_schema but generic access via index is safer/faster here
      auto& src_rel = [&]() -> auto& {
        if constexpr (Ver == FULL_VER)
          return std::get<I>(src.full);
        else if constexpr (Ver == DELTA_VER)
          return std::get<I>(src.delta);
        else if constexpr (Ver == NEW_VER)
          return std::get<I>(src.newt);
      }();

      auto& dst_rel = [&]() -> auto& {
        if constexpr (Ver == FULL_VER)
          return std::get<I>(dst.full);
        else if constexpr (Ver == DELTA_VER)
          return std::get<I>(dst.delta);
        else if constexpr (Ver == NEW_VER)
          return std::get<I>(dst.newt);
      }();

      using DstRelType = std::decay_t<decltype(dst_rel)>;

      // Heuristic for initialization:
      // 1. If DstRelType has static to_device(SrcRel), use it (Host -> Device)
      // 2. Default: Assignment/Copy constructor (Host -> Host, or compatible types)

      // We check for to_device existence via SFINAE/Concepts
      // For now, assume this pattern based on GPU runtime usage:
      // device_rel = DeviceRelationType<Schema>::to_device(host_rel);

      if constexpr (requires { DstRelType::to_device(src_rel); }) {
        dst_rel = DstRelType::to_device(src_rel);
      } else {
        // Fallback to assignment (works for Host->Host)
        dst_rel = src_rel;
      }

      // Set version and name (standard practice)
      dst_rel.set_version(Ver);
      dst_rel.set_name(src_rel.name() +
                       (Ver == FULL_VER ? "_Full" : (Ver == DELTA_VER ? "_Delta" : "_New")));
    });
  }

  static void execute(SrcDB& src, DstDB& dst) {
    copy_tuple<FULL_VER>(src, dst);
    copy_tuple<DELTA_VER>(src, dst);
    copy_tuple<NEW_VER>(src, dst);
  }
};
// predicate for SemiNaiveDatabase
// predicate for SemiNaiveDatabase
template <typename T>
struct is_semi_naive_database : std::false_type {};

template <CDatabase DBType, template <typename> typename RelationMapper>
struct is_semi_naive_database<SemiNaiveDatabase<DBType, RelationMapper>> : std::true_type {};

template <typename T>
inline constexpr bool is_semi_naive_database_v =
    is_semi_naive_database<std::remove_cvref_t<T>>::value;

template <typename T>
concept CSemiNaiveDatabase = is_semi_naive_database<std::remove_cvref_t<T>>::value;

// --- DSL Extensions ---

/**
 * @brief DSL Variable Wrapper to support expression building and UDLs.
 */
template <typename V>
struct VarWrapper {
  using type = V;
};

/**
 * @brief Internal helper for C++20 string literal as template parameter.
 */
template <std::size_t N>
struct DSLStringLiteral {
  char value[N];
  constexpr DSLStringLiteral(const char (&str)[N]) {
    for (std::size_t i = 0; i < N; ++i)
      value[i] = str[i];
  }
};

template <DSLStringLiteral str, std::size_t... Is>
consteval auto make_tstring_impl(std::index_sequence<Is...>) {
  return TString<str.value[Is]...>{};
}

/**
 * @brief UDL for variables: "x"_v produces a VarWrapper holding Var<"x">.
 */
template <DSLStringLiteral str>
consteval auto operator""_v() {
  using TStr = decltype(make_tstring_impl<str>(std::make_index_sequence<sizeof(str.value) - 1>{}));
  return VarWrapper<Var<TStr>>{};
}

/**
 * @brief DSL Relation Wrapper for concise clause creation.
 */
template <CRelationSchema S>
struct RelWrapper {
  template <typename... Ts>
  auto operator()(Ts...) const {
    return Clause<S, FULL_VER, typename Ts::type...>{};
  }
  template <typename... Ts>
  auto delta(Ts...) const {
    return Clause<S, DELTA_VER, typename Ts::type...>{};
  }
  template <typename... Ts>
  auto newt(Ts...) const {
    return Clause<S, NEW_VER, typename Ts::type...>{};
  }
  template <typename... Ts>
  auto full(Ts...) const {
    return Clause<S, FULL_VER, typename Ts::type...>{};
  }
};

/**
 * @brief Global helper to create a RelWrapper for a schema.
 */
template <CRelationSchema S>
constexpr auto rel = RelWrapper<S>{};

/**
 * @brief Rule Building: Accumulate body clauses into a tuple.
 */
template <CClause C1, CClause C2>
auto operator,(C1, C2) {
  return std::make_tuple(C1{}, C2{});
}

template <typename... Clauses, CClause Next>
auto operator,(std::tuple<Clauses...>, Next) {
  return std::make_tuple(Clauses{}..., Next{});
}

/**
 * @brief Helper to extract Var types from VarWrapper arguments.
 */
template <typename... VarWrappers>
struct ExtractVarTypes {
  using type = std::tuple<typename VarWrappers::type...>;
};

/**
 * @brief DSL helper for creating a JoinPlan from variables.
 * @details Creates a JoinPlan that specifies the order in which variables should be bound.
 *
 * @param vars The variables in join order (VarWrapper, e.g., "x"_v, "y"_v, "z"_v)
 *
 * @example
 * ```cpp
 * constexpr auto x = "x"_v;
 * constexpr auto y = "y"_v;
 * constexpr auto z = "z"_v;
 * auto plan = join_plan(x, y, z);
 * ```
 */
template <typename... VarWrappers>
constexpr auto plan(VarWrappers...) {
  using VarsTuple = typename ExtractVarTypes<VarWrappers...>::type;
  return JoinPlan<VarsTuple>{};
}

/**
 * @brief Rule Building: Define a rule using Head <<= Body syntax.
 * @details Automatically derives a JoinPlan from all variables found in the body.
 * Can be chained with operator| to specify an explicit join plan.
 *
 * @example Auto-derived join plan:
 * ```cpp
 * auto rule = path(x, y) <<= edge(x, y);  // Join plan auto-derived
 * ```
 *
 * @example Explicit join plan with pipe operator:
 * ```cpp
 * auto rule = path(x, z) <<= (path.delta(x, y), edge(y, z)) | plan(y, x, z);
 * ```
 */
template <typename HeadExpr, typename BodyExpr>
auto operator<<=(HeadExpr, BodyExpr) {
  using HeadTuple = std::conditional_t<tmp::is_tuple_v<HeadExpr>, HeadExpr, std::tuple<HeadExpr>>;
  using BodyTuple = std::conditional_t<tmp::is_tuple_v<BodyExpr>, BodyExpr, std::tuple<BodyExpr>>;

  // Helper trait to extract variables from all clauses in a tuple
  using AllBodyVarsList = mp::mp_transform_q<mp::mp_quote_trait<GetClauseVars>, BodyTuple>;
  using FlatBodyVars = mp::mp_flatten<AllBodyVarsList>;
  using UniqueBodyVars = tmp::TupleUnique<FlatBodyVars>;

  return Rule<HeadTuple, BodyTuple, JoinPlan<UniqueBodyVars>>{};
}

/**
 * @brief Rule Building: Alternative syntax for rule with explicit join plan.
 * @details Creates a rule with a user-specified join plan.
 *
 * @param head The rule head (clause or tuple of clauses)
 * @param body The rule body (clause or tuple of clauses)
 * @param plan The explicit join plan (created with join_plan())
 *
 * @example
 * ```cpp
 * constexpr auto x = "x"_v;
 * constexpr auto y = "y"_v;
 * constexpr auto z = "z"_v;
 * auto rule = rule_with_plan(path(x, z), (path.delta(x, y), edge(y, z)), join_plan(y, x, z));
 * ```
 */
template <typename HeadExpr, typename BodyExpr, CJoinPlan Plan>
constexpr auto rule_with_plan(HeadExpr, BodyExpr, Plan) {
  using HeadTuple = std::conditional_t<tmp::is_tuple_v<HeadExpr>, HeadExpr, std::tuple<HeadExpr>>;
  using BodyTuple = std::conditional_t<tmp::is_tuple_v<BodyExpr>, BodyExpr, std::tuple<BodyExpr>>;
  return Rule<HeadTuple, BodyTuple, Plan>{};
}

/**
 * @brief DSL helper for creating LetClause.
 * @details Creates a LetClause that binds a variable to the result of a C++ expression.
 * The lambda function must be passed as a template parameter.
 *
 * @tparam func The C++ lambda expression that computes the value
 * @param result_var The variable to bind (VarWrapper, e.g., "sum"_v)
 * @param dep_vars The variables the expression depends on (VarWrapper, e.g., "x"_v, "y"_v)
 *
 * @example
 * ```cpp
 * constexpr auto sum = "sum"_v;
 * constexpr auto x = "x"_v;
 * constexpr auto y = "y"_v;
 * constexpr auto add_func = [](int x, int y) -> int { return x + y; };
 * auto let_clause = let<add_func>(sum, x, y);
 * ```
 */
template <auto func, typename ResultVarWrapper, typename... DepVarWrappers>
constexpr auto let(ResultVarWrapper, DepVarWrappers...) {
  using ResultVar = typename ResultVarWrapper::type;
  using DepVars = typename ExtractVarTypes<DepVarWrappers...>::type;
  using Expr = CppExpr<DepVars, func>;
  return LetClause<ResultVar, Expr>{};
}

/**
 * @brief DSL helper for creating IfClause.
 * @details Creates an IfClause that filters tuples based on a C++ predicate.
 * The lambda function must be passed as a template parameter.
 *
 * @tparam func The C++ lambda expression that returns a boolean
 * @param dep_vars The variables the predicate depends on (VarWrapper, e.g., "x"_v)
 *
 * @example
 * ```cpp
 * constexpr auto x = "x"_v;
 * constexpr auto is_even = [](int x) -> bool { return x % 2 == 0; };
 * auto if_clause = if_<is_even>(x);
 * ```
 */
template <auto func, typename... DepVarWrappers>
constexpr auto if_(DepVarWrappers...) {
  using DepVars = typename ExtractVarTypes<DepVarWrappers...>::type;
  using Expr = CppExpr<DepVars, func>;
  return IfClause<Expr>{};
}

/**
 * @brief Program Building helpers.
 */
template <CRule... Rules>
auto fixpoint(Rules...) {
  return Fixpoint<Rules...>{};
}

template <CRule... Rules>
auto non_iterative(Rules...) {
  return NonIterativeRuleSets<Rules...>{};
}

}  // namespace SRDatalog::AST