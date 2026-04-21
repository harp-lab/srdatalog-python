/**
 * @file mir_gpu.h
 * @brief GPU optimization passes for MIR.
 *
 * @details This file contains optimization passes that transform MIR operations
 * to be more efficient for GPU execution, specifically fusing sequences of
 * single-source joins into CartesianJoins.
 */

#pragma once

#include "mir_def.h"
#include "mir_plan.h"

namespace SRDatalog {
using namespace AST;
namespace mir {
namespace gpu_opt {

// ============================================================================
// Pattern Match Result
// ============================================================================

template <std::size_t JoinCount, typename Vars, typename Sources, typename MiddleOps,
          typename RemainingOps, std::size_t HandleStart = 0,
          typename VarToHandlesMap = decltype(hana::make_map())>
struct PatternMatchResult {
  static constexpr std::size_t join_count = JoinCount;
  using vars_type = Vars;
  using sources_type = Sources;
  using middle_ops_type = MiddleOps;
  using remaining_ops_type = RemainingOps;
  static constexpr std::size_t handle_start = HandleStart;
  using var_to_handles_map = VarToHandlesMap;
};

using NoMatch = PatternMatchResult<0, std::tuple<>, std::tuple<>, std::tuple<>, std::tuple<>, 0,
                                   decltype(hana::make_map())>;

// ============================================================================
// Predicates (modern constexpr style)
// ============================================================================

/**
 * @brief Check if a ColumnSource has an empty prefix (no dependencies)
 */
template <typename Source>
constexpr bool hasEmptyPrefix() {
  if constexpr (requires { typename Source::prefix_vars_type; }) {
    return std::tuple_size_v<typename Source::prefix_vars_type> == 0;
  }
  return false;
}

/**
 * @brief Check if an operation is a Let or If clause
 */
template <typename Op>
constexpr bool isLetOrIf() {
  return CLetClause<Op> || CIfClause<Op>;
}

/**
 * @brief Check if an operation is a DestinationRelation
 */
template <typename Op>
constexpr bool isDestinationRelation() {
  return mir::is_destination_relation_v<Op>;
}

/**
 * @brief Check if a join is independent of collected variables
 *
 * Previously: A join was independent only if its prefix didn't contain
 * collected vars. Now: All single-source ColumnJoins are considered
 * "independent" for CartesianJoin. Batch dedup at index/merge is more
 * efficient than avoiding duplicates here.
 */
template <typename Op, typename CollectedVars>
constexpr bool isIndependentJoin() {
  if constexpr (requires { typename Op::sources_type; }) {
    using Sources = typename Op::sources_type;
    if constexpr (std::tuple_size_v<Sources> == 1) {
      using Source = std::tuple_element_t<0, Sources>;
      // Check for dependencies on CollectedVars
      if constexpr (requires { typename Source::prefix_vars_type; }) {
        using Prefix = typename Source::prefix_vars_type;
        using CommonVars = tmp::TupleSetIntersection<Prefix, CollectedVars>;
        // Independent only if no overlap with already collected variables
        return std::tuple_size_v<CommonVars> == 0;
      }
      return true;
    }
  }
  return false;
}

// ============================================================================
// MatchPattern: Use specialization for extraction, constexpr for dispatch
// ============================================================================

/**
 * @brief Match pattern: single joins + (Let/If)* + DestinationRelation
 *
 * This recursively collects:
 * 1. Consecutive independent single-source ColumnJoins
 * 2. Optional Let/If operations after the joins
 * 3. A DestinationRelation that completes the pattern
 */
template <typename OpsTuple, typename CollectedVars, typename CollectedSources>
struct MatchPattern {
  using type = NoMatch;
};

// Base case: empty tuple
template <typename CollectedVars, typename CollectedSources>
struct MatchPattern<std::tuple<>, CollectedVars, CollectedSources> {
  using type = NoMatch;
};

// ============================================================================
// Case 1: ColumnJoin - use specialization to extract Var, Source
// ============================================================================

template <typename Var, typename IndexSpec, typename PrefixVars, typename Strategy,
          std::size_t HandleStart, typename VarToHandlesMap, typename... Tail,
          typename CollectedVars, typename CollectedSources>
struct MatchPattern<
    std::tuple<mir::ColumnJoin<Var, std::tuple<mir::ColumnSource<IndexSpec, PrefixVars>>, Strategy,
                               HandleStart, VarToHandlesMap>,
               Tail...>,
    CollectedVars, CollectedSources> {
 private:
  using CurrentJoin = mir::ColumnJoin<Var, std::tuple<mir::ColumnSource<IndexSpec, PrefixVars>>,
                                      Strategy, HandleStart, VarToHandlesMap>;
  using CurrentSource = mir::ColumnSource<IndexSpec, PrefixVars>;

  // Check if we can merge this source with the last collected source
  // (optimization for A(x,y))
  template <typename Srcs, typename Vars>
  struct CanMergeWithLast {
    static constexpr bool value = false;
  };

  template <typename... Srcs, typename... Vars>
  struct CanMergeWithLast<std::tuple<Srcs...>, std::tuple<Vars...>> {
    using LastSource = std::tuple_element_t<sizeof...(Srcs) - 1, std::tuple<Srcs...>>;
    using LastVar = std::tuple_element_t<sizeof...(Vars) - 1, std::tuple<Vars...>>;
    using LastSpec = typename LastSource::index_spec_type;
    using CurrentSpec = typename CurrentSource::index_spec_type;

    // Check if schemas and versions match
    static constexpr bool same_schema =
        std::is_same_v<typename LastSpec::schema_type, typename CurrentSpec::schema_type>;
    static constexpr bool same_version = (LastSpec::kVersion == CurrentSpec::kVersion);

    // Check if current prefix is extension of last prefix
    using LastPrefix = typename LastSource::prefix_vars_type;
    using ExpectedCurrentPrefix = Mp::mp_append<LastPrefix, std::tuple<Vars...>>;
    static constexpr bool prefix_matches =
        std::is_same_v<typename CurrentSource::prefix_vars_type, ExpectedCurrentPrefix>;

    // DEBUG PROBE REMOVED
    static constexpr bool value = same_schema && same_version && prefix_matches;
  };

  // Base case for empty sources
  template <typename... Vars>
  struct CanMergeWithLast<std::tuple<>, std::tuple<Vars...>> {
    static constexpr bool value = false;
  };

  // NOTE: isIndependent is checked first because CartesianJoin-eligible joins
  // should NOT merge sources. The downstream VarToHandlesMap expects separate handles.
  static constexpr bool isIndependent = isIndependentJoin<CurrentJoin, CollectedVars>();
  // Only merge if NOT independent (merging for CartesianJoin breaks handle indexing)
  static constexpr bool canMergeFromRelation =
      CanMergeWithLast<CollectedSources, CollectedVars>::value;
  static constexpr bool canMerge = canMergeFromRelation && !isIndependent;

  // Helper to concatenate two integer sequences
  template <typename Seq1, typename Seq2>
  struct concat_sequences;

  template <std::size_t... Is1, std::size_t... Is2>
  struct concat_sequences<std::integer_sequence<std::size_t, Is1...>,
                          std::integer_sequence<std::size_t, Is2...>> {
    using type = std::integer_sequence<std::size_t, Is1..., Is2...>;
  };

  // Helper to merge two hana::maps, concatenating integer sequences for duplicate keys
  template <typename Map1, typename Map2>
  static constexpr auto merge_maps() {
    return hana::fold_left(Map2{}, Map1{}, [](auto acc, auto pair) {
      auto key = hana::first(pair);
      auto val = hana::second(pair);
      if constexpr (hana::contains(acc, key)) {
        // Merge: concatenate integer sequences
        auto existing = hana::at_key(acc, key);
        using ConcatSeq = typename concat_sequences<decltype(existing), decltype(val)>::type;
        return hana::insert(hana::erase_key(acc, key), hana::make_pair(key, ConcatSeq{}));
      } else {
        return hana::insert(acc, hana::make_pair(key, val));
      }
    });
  }

  // Modern: use if constexpr in a constexpr lambda to compute type
  static constexpr auto computeResult() {
    if constexpr (canMerge) {
      using NewVars = Mp::mp_push_back<CollectedVars, Var>;

      // Merge sources: remove last, add modified current source (with SAME
      // BASE prefix) The new source represents the flattened scan over the
      // relation covering multiple vars
      using SourcesMinusLast = Mp::mp_pop_back<CollectedSources>;
      using LastSource =
          std::tuple_element_t<std::tuple_size_v<CollectedSources> - 1, CollectedSources>;
      using LastPrefix = typename LastSource::prefix_vars_type;
      using ModifiedSource = mir::ColumnSource<IndexSpec, LastPrefix>;

      // Pop last source
      // using SourcesMinusLast = Mp::mp_pop_back<CollectedSources>; // Moved
      // up Push modified source
      using NewSources = Mp::mp_push_back<SourcesMinusLast, ModifiedSource>;

      using TailMatch = typename MatchPattern<std::tuple<Tail...>, NewVars, NewSources>::type;

      // Merge VarToHandlesMap from current join with tail
      using MergedMap =
          decltype(merge_maps<VarToHandlesMap, typename TailMatch::var_to_handles_map>());

      // Use HandleStart from first collected join (when CollectedVars was empty, use current)
      static constexpr std::size_t effective_handle_start =
          (std::tuple_size_v<CollectedVars> == 0) ? HandleStart : TailMatch::handle_start;

      return PatternMatchResult<
          TailMatch::join_count, typename TailMatch::vars_type, typename TailMatch::sources_type,
          typename TailMatch::middle_ops_type, typename TailMatch::remaining_ops_type,
          effective_handle_start, MergedMap>{};
    } else if constexpr (isIndependent) {
      using NewVars = Mp::mp_push_back<CollectedVars, Var>;
      using NewSources = Mp::mp_push_back<CollectedSources, CurrentSource>;
      using TailMatch = typename MatchPattern<std::tuple<Tail...>, NewVars, NewSources>::type;

      // Merge VarToHandlesMap from current join with tail
      using MergedMap =
          decltype(merge_maps<VarToHandlesMap, typename TailMatch::var_to_handles_map>());

      // Use HandleStart from first collected join (when CollectedVars was empty, use current)
      static constexpr std::size_t effective_handle_start =
          (std::tuple_size_v<CollectedVars> == 0) ? HandleStart : TailMatch::handle_start;

      return PatternMatchResult<
          TailMatch::join_count, typename TailMatch::vars_type, typename TailMatch::sources_type,
          typename TailMatch::middle_ops_type, typename TailMatch::remaining_ops_type,
          effective_handle_start, MergedMap>{};
    } else {
      return PatternMatchResult<0, std::tuple<>, std::tuple<>, std::tuple<>,
                                std::tuple<CurrentJoin, Tail...>, 0, decltype(hana::make_map())>{};
    }
  }

 public:
  using type = decltype(computeResult());
};

// ============================================================================
// Case 2: LetClause - use concept constraint
// ============================================================================

template <CVar V, CCppExpr E, typename... Tail, typename CollectedVars, typename CollectedSources>
struct MatchPattern<std::tuple<AST::LetClause<V, E>, Tail...>, CollectedVars, CollectedSources> {
 private:
  using Head = AST::LetClause<V, E>;
  static constexpr bool hasJoins = (std::tuple_size_v<CollectedVars> >= 2);

  using TailMatch =
      typename MatchPattern<std::tuple<Tail...>, CollectedVars, CollectedSources>::type;
  static constexpr bool tailHasPattern = (TailMatch::join_count > 0);

  static constexpr auto computeResult() {
    if constexpr (hasJoins || tailHasPattern) {
      return PatternMatchResult<TailMatch::join_count, typename TailMatch::vars_type,
                                typename TailMatch::sources_type,
                                Mp::mp_push_front<typename TailMatch::middle_ops_type, Head>,
                                typename TailMatch::remaining_ops_type, TailMatch::handle_start,
                                typename TailMatch::var_to_handles_map>{};
    } else {
      return PatternMatchResult<0, std::tuple<>, std::tuple<>, std::tuple<>,
                                std::tuple<Head, Tail...>, 0, decltype(hana::make_map())>{};
    }
  }

 public:
  using type = decltype(computeResult());
};

// ============================================================================
// Case 3: IfClause - similar to LetClause
// ============================================================================

template <CCppExpr E, typename... Tail, typename CollectedVars, typename CollectedSources>
struct MatchPattern<std::tuple<AST::IfClause<E>, Tail...>, CollectedVars, CollectedSources> {
 private:
  using Head = AST::IfClause<E>;
  static constexpr bool hasJoins = (std::tuple_size_v<CollectedVars> >= 2);

  using TailMatch =
      typename MatchPattern<std::tuple<Tail...>, CollectedVars, CollectedSources>::type;
  static constexpr bool tailHasPattern = (TailMatch::join_count > 0);

  static constexpr auto computeResult() {
    if constexpr (hasJoins || tailHasPattern) {
      return PatternMatchResult<TailMatch::join_count, typename TailMatch::vars_type,
                                typename TailMatch::sources_type,
                                Mp::mp_push_front<typename TailMatch::middle_ops_type, Head>,
                                typename TailMatch::remaining_ops_type, TailMatch::handle_start,
                                typename TailMatch::var_to_handles_map>{};
    } else {
      return PatternMatchResult<0, std::tuple<>, std::tuple<>, std::tuple<>,
                                std::tuple<Head, Tail...>, 0, decltype(hana::make_map())>{};
    }
  }

 public:
  using type = decltype(computeResult());
};

// ============================================================================
// Case 4: DestinationRelation - use specialization to match structure
// ============================================================================

template <CRelationSchema Schema, typename Terms, std::size_t Ver, CIndexSpec IdxSpec,
          typename ActiveHandlesMap, typename... Tail, typename CollectedVars,
          typename CollectedSources>
struct MatchPattern<
    std::tuple<mir::DestinationRelation<Schema, Terms, Ver, IdxSpec, ActiveHandlesMap>, Tail...>,
    CollectedVars, CollectedSources> {
 private:
  using Head = mir::DestinationRelation<Schema, Terms, Ver, IdxSpec, ActiveHandlesMap>;
  static constexpr bool hasPattern = (std::tuple_size_v<CollectedVars> >= 2);

  static constexpr auto computeResult() {
    if constexpr (hasPattern) {
      return PatternMatchResult<std::tuple_size_v<CollectedVars>, CollectedVars, CollectedSources,
                                std::tuple<>, std::tuple<Head, Tail...>, 0,
                                decltype(hana::make_map())>{};
    } else {
      return PatternMatchResult<0, std::tuple<>, std::tuple<>, std::tuple<>,
                                std::tuple<Head, Tail...>, 0, decltype(hana::make_map())>{};
    }
  }

 public:
  using type = decltype(computeResult());
};

// ============================================================================
// Transformation (modern style)
// ============================================================================

/**
 * @brief Transform MIR operations: fuse single joins before
 * DestinationRelation into CartesianJoin
 */
template <typename OpsTuple>
struct TransformToCartesian;

template <>
struct TransformToCartesian<std::tuple<>> {
  using type = std::tuple<>;
};

namespace detail {

// Helper to build VarToHandlesMap for CartesianJoin from its sources
template <typename SourcesTuple, std::size_t HandleStart>
struct BuildCartesianHandleMap {
 private:
  // Collect all unique prefix variables from sources
  template <typename S>
  using GetPrefix = typename S::prefix_vars_type;

  using SourcesList = boost::mp11::mp_rename<SourcesTuple, boost::mp11::mp_list>;

  template <typename T>
  using ToMpList = boost::mp11::mp_rename<T, boost::mp11::mp_list>;

  using AllPrefixVars = boost::mp11::mp_unique<boost::mp11::mp_flatten<
      boost::mp11::mp_transform<ToMpList, boost::mp11::mp_transform<GetPrefix, SourcesList>>>>;

  // Convert list to tuple for iteration
  using PrefixVarsTuple = boost::mp11::mp_rename<AllPrefixVars, std::tuple>;

  // For each variable, find which sources have it in their prefix
  template <typename Var, std::size_t... Is>
  static constexpr auto get_handles_for_var_impl(std::index_sequence<Is...>) {
    // Build list of handle indices where source[Is] has Var in its prefix
    constexpr auto check_source = []<std::size_t I>() {
      using Source = std::tuple_element_t<I, SourcesTuple>;
      using Prefix = typename Source::prefix_vars_type;
      return boost::mp11::mp_contains<Prefix, Var>::value;
    };

    // Collect indices where source contains Var in prefix
    return []<bool... Contains>(std::integer_sequence<bool, Contains...>) {
      constexpr std::size_t count = (std::size_t{Contains} + ... + 0);
      std::array<std::size_t, count> result{};
      std::size_t idx = 0;
      ((Contains ? (result[idx++] = HandleStart + Is, void()) : void()), ...);
      return result;
    }(std::integer_sequence<bool, check_source.template operator()<Is>()...>{});
  }

  template <typename Var>
  static constexpr auto get_handles_for_var() {
    constexpr auto arr =
        get_handles_for_var_impl<Var>(std::make_index_sequence<std::tuple_size_v<SourcesTuple>>{});
    return [&]<std::size_t... Js>(std::index_sequence<Js...>) {
      return std::integer_sequence<std::size_t, arr[Js]...>{};
    }(std::make_index_sequence<arr.size()>{});
  }

  template <typename... Vars>
  static constexpr auto build_map_impl(std::tuple<Vars...>) {
    return hana::make_map(hana::make_pair(hana::type_c<Vars>, get_handles_for_var<Vars>())...);
  }

 public:
  using type = decltype(build_map_impl(PrefixVarsTuple{}));
};

// Helper to build VarToHandlesMap for Scan - all vars map to same handle
template <typename VarsTuple, std::size_t HandleStart>
struct BuildScanHandleMap {
 private:
  // For Scan, all variables are bound from the same iterator (HandleStart)
  // Each variable maps to a sequence containing just HandleStart
  template <typename Var>
  static constexpr auto get_handle_for_var() {
    return std::integer_sequence<std::size_t, HandleStart>{};
  }

  template <typename... Vars>
  static constexpr auto build_map_impl(std::tuple<Vars...>) {
    return hana::make_map(hana::make_pair(hana::type_c<Vars>, get_handle_for_var<Vars>())...);
  }

 public:
  using type = decltype(build_map_impl(VarsTuple{}));
};

}  // namespace detail

template <typename Head, typename... Tail>
struct TransformToCartesian<std::tuple<Head, Tail...>> {
 private:
  using Match = typename MatchPattern<std::tuple<Head, Tail...>, std::tuple<>, std::tuple<>>::type;
  static constexpr bool hasMatch = (Match::join_count >= 2);

  static constexpr auto computeResult() {
    if constexpr (hasMatch) {
      // Check if this is a copy/project rule (multiple vars, single source)
      constexpr std::size_t num_sources = std::tuple_size_v<typename Match::sources_type>;
      constexpr std::size_t num_vars = std::tuple_size_v<typename Match::vars_type>;

      if constexpr (num_sources == 1 && num_vars >= 2) {
        // Copy/Project rule: use Scan instead of CartesianJoin
        // Scan linearly iterates all tuples, binding all variables at once
        using SingleSource = std::tuple_element_t<0, typename Match::sources_type>;
        using IndexSpec = typename SingleSource::index_spec_type;
        using ScanVars = typename Match::vars_type;
        using PrefixVars = std::tuple<>;  // No prefix for root scan

        // Build handle map for Scan
        using CorrectHandleMap =
            typename detail::BuildCartesianHandleMap<typename Match::sources_type,
                                                     Match::handle_start>::type;

        using ScanOp =
            mir::Scan<ScanVars, IndexSpec, PrefixVars, Match::handle_start, CorrectHandleMap>;
        using RemainingTransformed =
            typename TransformToCartesian<typename Match::remaining_ops_type>::type;
        using WithMiddle = Mp::mp_append<typename Match::middle_ops_type, RemainingTransformed>;

        return Mp::mp_push_front<WithMiddle, ScanOp>{};
      } else {
        // True CartesianJoin: multiple sources
        // Build new VarToHandlesMap based on CartesianJoin's sources
        using CorrectHandleMap =
            typename detail::BuildCartesianHandleMap<typename Match::sources_type,
                                                     Match::handle_start>::type;

        using Cartesian =
            mir::CartesianJoin<typename Match::vars_type, typename Match::sources_type,
                               Match::handle_start, CorrectHandleMap>;
        using RemainingTransformed =
            typename TransformToCartesian<typename Match::remaining_ops_type>::type;
        using WithMiddle = Mp::mp_append<typename Match::middle_ops_type, RemainingTransformed>;

        return Mp::mp_push_front<WithMiddle, Cartesian>{};
      }
    } else {
      using TailTransformed = typename TransformToCartesian<std::tuple<Tail...>>::type;
      return Mp::mp_push_front<TailTransformed, Head>{};
    }
  }

 public:
  using type = decltype(computeResult());
};

// ============================================================================
// Obsolete handle computation traits removed.
// Handle mapping is now performed during AST-to-MIR compilation.
// ============================================================================

namespace detail {

// Helper for ComputeVarToHandlesMapSafe - now just a placeholder or deleted
template <typename Op, std::size_t StartOffset>
struct ComputeVarToHandlesMapSafe {
  static constexpr auto value = hana::make_map();
};

}  // namespace detail

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Optimize MIR operations for GPU execution
 *
 * @details This pass replaces sequences of independent single-source
 * ColumnJoins (followed by optional Let/If operations) immediately before
 * DestinationRelation with CartesianJoin. This is beneficial for GPU because:
 * - CartesianJoin materializes all variables at once using flat iteration
 * - Avoids nested loops and handle materialization overhead
 * - Better suited for GPU's parallel execution model
 */
template <typename MIROps>
struct OptimizeMIRForGPU {
  using type = typename TransformToCartesian<MIROps>::type;
};

/**
 * @brief Extract source IndexSpecs from a ColumnSource
 */
template <typename T>
struct ExtractSourceSpec {
  using type = std::tuple<>;
};

template <typename IndexSpec, typename Prefix>
struct ExtractSourceSpec<mir::ColumnSource<IndexSpec, Prefix>> {
  using type = std::tuple<IndexSpec>;
};

/**
 * @brief Extract source IndexSpecs from MIR operations recursively
 */
template <typename Op>
struct ExtractSourceSpecsFromOp {
  using type = std::tuple<>;
};

// ColumnJoin: extract specs from sources
template <typename Var, typename Sources, typename Strategy, std::size_t HS, typename VHM>
struct ExtractSourceSpecsFromOp<mir::ColumnJoin<Var, Sources, Strategy, HS, VHM>> {
 private:
  template <typename... Srcs>
  static auto extract_from_tuple(std::tuple<Srcs...>) {
    return std::tuple_cat(typename ExtractSourceSpec<Srcs>::type{}...);
  }

 public:
  using type = decltype(extract_from_tuple(Sources{}));
};

// CartesianJoin: extract specs from sources
template <typename Vars, typename Sources, std::size_t HS, typename VHM>
struct ExtractSourceSpecsFromOp<mir::CartesianJoin<Vars, Sources, HS, VHM>> {
 private:
  template <typename... Srcs>
  static auto extract_from_tuple(std::tuple<Srcs...>) {
    return std::tuple_cat(typename ExtractSourceSpec<Srcs>::type{}...);
  }

 public:
  using type = decltype(extract_from_tuple(Sources{}));
};

// Aggregate: extract spec
template <typename Var, typename Idx, typename Prefix, typename Func, std::size_t HS, typename VHM>
struct ExtractSourceSpecsFromOp<mir::Aggregate<Var, Idx, Prefix, Func, HS, VHM>> {
  using type = std::tuple<Idx>;
};

// Negation: extract spec
template <typename Idx, typename Prefix, std::size_t HS, typename VHM>
struct ExtractSourceSpecsFromOp<mir::Negation<Idx, Prefix, HS, VHM>> {
  using type = std::tuple<Idx>;
};

// Scan: extract spec from the index
template <typename Vars, typename Idx, typename Prefix, std::size_t HS, typename VHM>
struct ExtractSourceSpecsFromOp<mir::Scan<Vars, Idx, Prefix, HS, VHM>> {
  using type = std::tuple<Idx>;
};

/**
 * @brief Extract all source specs from a tuple of MIR operations
 */
template <typename OpsTuple>
struct ExtractSourceSpecsFromOps;

template <typename... Ops>
struct ExtractSourceSpecsFromOps<std::tuple<Ops...>> {
  using type = Mp::mp_append<typename ExtractSourceSpecsFromOp<Ops>::type...>;
};

/**
 * @brief Optimize a Pipeline for GPU execution
 *
 * @details Applies GPU-specific optimizations to the MIR operations in a
 * Pipeline, and recomputes source_specs from the optimized operations.
 */
template <typename Pipeline>
struct OptimizePipelineForGPU {
  static_assert(mir::is_pipeline_v<Pipeline>, "Pipeline must satisfy CPipeline");
  using OptimizedOps = typename OptimizeMIRForGPU<typename Pipeline::mir_ops_type>::type;
  // Recompute source_specs from the optimized operations (handles merged sources)
  using NewSourceSpecs = typename ExtractSourceSpecsFromOps<OptimizedOps>::type;
  using type = mir::Pipeline<OptimizedOps, typename Pipeline::var_pos_map_type, NewSourceSpecs,
                             typename Pipeline::destination_specs_type>;
};

// Remove RebuildIndex (specialization inside OptimizeInstructionsForGPU)

// ============================================================================
// Fixpoint instruction predicates using concepts
// ============================================================================

/**
 * @brief Concept for fixpoint-level instructions (not rule-level MIR
 * operations)
 *
 * @details Fixpoint-level instructions are operations that execute at the
 * fixpoint iteration level, outside of nested for-loops. These include:
 * - ExecutePipeline<Pipeline> - executes a rule's pipeline
 * - CreateIndex<IndexSpecT> - builds an index
 * - RebuildIndex<IndexSpecT> - rebuilds an index
 * - ClearRelation<Schema, Ver> - clears a relation
 * - CheckSize<Schema, Ver> - checks relation size
 * - InsertFromRelation<...> - inserts from one relation to another
 * - SwapRelations<Schema> - swaps relation versions
 * - SetVersion<Schema, Ver> - sets relation version
 *
 * These are distinct from rule-level MIR operations (ColumnJoin,
 * CartesianJoin, DestinationRelation, etc.) which are inside
 * Pipeline::mir_ops_type.
 */
template <typename T>
concept FixpointInstruction =
    mir::is_execute_pipeline_v<T> || mir::is_build_index_v<T> || mir::is_rebuild_index_v<T> ||
    mir::is_clear_relation_v<T> || mir::is_check_size_v<T> || mir::is_insert_from_relation_v<T> ||
    mir::is_swap_relations_v<T> || mir::is_set_version_v<T> ||
    mir::is_compute_delta_and_merge_v<T> || mir::is_compute_delta_v<T> || mir::is_merge_index_v<T>;

// ============================================================================
// Instruction optimization
// ============================================================================

/**
 * @brief Optimize instructions in a FixpointPlan for GPU execution
 *
 * @details Applies GPU-specific optimizations to all ExecutePipeline
 * instructions. Only optimizes fixpoint-level instructions (ExecutePipeline,
 * RebuildIndex, etc.), not rule-level MIR operations (which are inside
 * Pipeline::mir_ops_type).
 */
template <tmp::CTuple Instructions>
struct OptimizeInstructionsForGPU;

// Base case: empty tuple
template <>
struct OptimizeInstructionsForGPU<std::tuple<>> {
  using type = std::tuple<>;
};

// Process one fixpoint-level instruction
template <typename Head, typename... Tail>
struct OptimizeInstructionsForGPU<std::tuple<Head, Tail...>> {
 private:
  // Optimize fixpoint-level instructions (not rule-level MIR operations)
  template <typename Inst>
  struct OptimizeInstruction {
    using type = Inst;  // Default: keep instruction as-is
  };

  // Specialization: Optimize ExecutePipeline by optimizing its Pipeline's
  // MIROps
  template <mir::CPipeline Pipeline>
  struct OptimizeInstruction<mir::ExecutePipeline<Pipeline>> {
    // Optimize the Pipeline's rule-level MIR operations (ColumnJoin ->
    // CartesianJoin, etc.)
    using OptimizedPipeline = typename OptimizePipelineForGPU<Pipeline>::type;
    using type = mir::ExecutePipeline<OptimizedPipeline>;
  };

  // GPU-specific optimization: Remove FULL_VER RebuildIndex operations
  // On GPU, we merge indexes to FULL instead of rebuilding FULL indexes from
  // scratch However, we keep NEW_VER RebuildIndex operations because they're
  // needed to reconstruct NEW_VER from index before merge
  template <CIndexSpec IndexSpecT>
  struct OptimizeInstruction<mir::RebuildIndex<IndexSpecT>> {
    // Remove if FULL_VER, keep if NEW_VER
    static constexpr bool is_full_ver = (IndexSpecT::kVersion == FULL_VER);
    using type = std::conditional_t<is_full_ver, void, mir::RebuildIndex<IndexSpecT>>;
  };

  // Helper to filter out SetVersion instructions (not needed after
  // ComputeDeltaAndMerge)
  template <typename Tuple>
  struct FilterSetVersion;

  template <>
  struct FilterSetVersion<std::tuple<>> {
    using type = std::tuple<>;
  };

  template <typename H, typename... Ts>
  struct FilterSetVersion<std::tuple<H, Ts...>> {
    using FilteredTail = typename FilterSetVersion<std::tuple<Ts...>>::type;
    using type = std::conditional_t<mir::is_set_version_v<H>, FilteredTail,
                                    mp::mp_push_front<FilteredTail, H>>;
  };

  // Note: InsertFromRelation + SwapRelations pattern is now handled at
  // compile time in CompileFixpointToPlan, which directly generates
  // ComputeDelta + ClearRelation. No runtime optimization needed here - the
  // base plan already has the correct operations.

  // Filter out void (removed instructions)
  template <typename T, typename TailTuple>
  struct PushFront {
    using type = TailTuple;  // Default: if T is void, just return TailTuple
  };

  // Specialization: when T is not void, prepend it to the tuple
  template <typename T, typename... Ts>
    requires(!std::is_same_v<T, void>)
  struct PushFront<T, std::tuple<Ts...>> {
    using type = std::tuple<T, Ts...>;
  };

 public:
  // Note: InsertFromRelation + SwapRelations pattern is now handled at
  // compile time in CompileFixpointToPlan, which directly generates
  // ComputeDelta + ClearRelation. No runtime optimization needed here - the
  // base plan already has the correct operations.
  using type =
      typename PushFront<typename OptimizeInstruction<Head>::type,
                         typename OptimizeInstructionsForGPU<std::tuple<Tail...>>::type>::type;
};

/**
 * @brief Optimize a FixpointPlan for GPU execution
 * Helpers to extract metadata from Plan types with SFINAE fallback
 */
// SFINAE helper to check if Plan has incremental_relations_type
template <typename Plan, typename = void>
struct gpu_extract_incremental_relations_impl {
  using type = std::tuple<>;  // Default to empty tuple
};

template <typename Plan>
struct gpu_extract_incremental_relations_impl<
    Plan, std::void_t<typename Plan::incremental_relations_type>> {
  using type = typename Plan::incremental_relations_type;
};

template <typename Plan>
using gpu_extract_incremental_relations_t =
    typename gpu_extract_incremental_relations_impl<Plan>::type;

// SFINAE helper to check if Plan has canonical_index_specs_type
template <typename Plan, typename = void>
struct gpu_extract_canonical_index_specs_impl {
  using type = std::tuple<>;  // Default to empty tuple
};

template <typename Plan>
struct gpu_extract_canonical_index_specs_impl<
    Plan, std::void_t<typename Plan::canonical_index_specs_type>> {
  using type = typename Plan::canonical_index_specs_type;
};

template <typename Plan>
using gpu_extract_canonical_index_specs_t =
    typename gpu_extract_canonical_index_specs_impl<Plan>::type;

/**
 * @brief Logic to optimize a FixpointPlan for GPU execution.
 *
 * @details Applies GPU-specific optimizations to all Pipelines in a
 * FixpointPlan.
 */
template <typename Plan>
struct OptimizeFixpointPlanForGPU {
  // Note: is_fixpoint_plan_v is defined outside mir namespace, so we check
  // manually Just assume Plan has instructions_type (it's a FixpointPlan)
  using Instructions = typename Plan::instructions_type;
  using OptimizedInstructions = typename OptimizeInstructionsForGPU<Instructions>::type;

  // Propagate metadata (IncrementalRelations and CanonicalIndexSpecs)
  // Default to empty tuples if not present (for backward compatibility or
  // partial plans)
  using IncrementalRelations = gpu_extract_incremental_relations_t<Plan>;
  using CanonicalIndexSpecs = gpu_extract_canonical_index_specs_t<Plan>;

  using type = mir::FixpointPlan<OptimizedInstructions, IncrementalRelations, CanonicalIndexSpecs>;
};

/**
 * @brief Check if a tuple of MIR operations contains a CartesianJoin
 */
template <typename OpsTuple>
struct HasCartesianJoin {
 private:
  template <typename Op>
  struct CheckOp {
    static constexpr bool value = mir::is_cartesian_join_v<Op>;
  };

  template <typename T>
  struct CheckOp<mir::ExecutePipeline<T>> {
    static constexpr bool value = HasCartesianJoin<typename T::mir_ops_type>::value;
  };

  template <typename... Alternatives>
  struct CheckOp<std::variant<Alternatives...>> {
    static constexpr bool value = (CheckOp<Alternatives>::value || ...);
  };

 public:
  static constexpr bool value = []<typename... Ops>(std::tuple<Ops...>) {
    return (CheckOp<Ops>::value || ...);
  }(OpsTuple{});
};

}  // namespace gpu_opt
}  // namespace mir
}  // namespace SRDatalog