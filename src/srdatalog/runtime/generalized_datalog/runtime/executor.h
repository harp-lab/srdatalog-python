/**
 * @file runtime/executor.h
 * @brief Pipeline executor - core template metaprogramming interpreter for pipelines.
 *
 * @details This file implements the PipelineExecutor, which executes pipeline
 * operations (ColumnJoin, LetClause, DestinationRelation) using tuple-based state
 * management. It uses precompiled hana::maps from the Pipeline for variable and
 * iterator position lookups. This is separate from fixpoint-level operations
 * (BuildIndex, ClearRelation, etc.) which have different execution semantics.
 */

#pragma once

#include "../query.h"      // For get_relation_by_schema
#include "mir.h"           // For MIR types (mir::ColumnSource, etc.) and IndexSpecT
#include "relation_col.h"  // For column_intersect
#include "runtime/aggregation.h"
#include "runtime/iterator.h"
#include <limits>

namespace SRDatalog {
using namespace AST;

namespace detail {
// ExtractIndexSpecsFromMIR removed - production code uses Pipeline::source_specs_type
// which is computed from AST via ComputeNaiveIndices, not extracted from MIR

/**
 * @brief Helper trait to extract IndexSpecT from a ColumnSource.
 *
 * @details Extracts the index_spec_type from a ColumnSource template.
 */
template <typename Source>
struct ExtractIndexSpecT {
  // For ColumnSource, extract the index_spec_type member
  using type = typename Source::index_spec_type;
};

/**
 * @brief Helper to extract all FULL_VER IndexSpecT from DestinationRelation operations in MIR plan.
 *
 * @details Similar to ExtractIndexSpecsFromMIR but only extracts from DestinationRelation
 * operations and converts them to FULL_VER IndexSpecT for output_full_iterators.
 */
template <typename MIRPlan>
struct ExtractDestinationRelationIndexSpecs {
  // Base case: empty tuple -> empty list
  using type = std::tuple<>;
};

template <typename HeadOp, typename... TailOps>
struct ExtractDestinationRelationIndexSpecs<std::tuple<HeadOp, TailOps...>> {
  static_assert(sizeof(HeadOp) == 0, "DEBUG: ExtractDestinationRelationIndexSpecs entered");

 private:
  // Extract from HeadOp if it's a DestinationRelation
  template <typename Op>
  struct ExtractFromOp {
    using type = std::tuple<>;
  };

  // Specialization for DestinationRelation
  template <CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver, mir::CIndexSpec IndexSpec,
            typename ActiveHandlesMap>
  struct ExtractFromOp<mir::DestinationRelation<Schema, Terms, Ver, IndexSpec, ActiveHandlesMap>> {
    // Convert to FULL_VER IndexSpecT for output_full_iterators
    // If the IndexSpec has empty column_indexes_type (placeholder), use canonical index
    using ColSeq = typename IndexSpec::column_indexes_type;
    using CanonicalSeq = typename Schema::canonical_index_type;
    // Convert std::index_sequence to std::integer_sequence<int, ...>
    template <std::size_t... Is>
    static auto to_int_seq(std::index_sequence<Is...>)
        -> std::integer_sequence<int, static_cast<int>(Is)...>;
    using CanonicalIntSeq = decltype(to_int_seq(CanonicalSeq{}));
    // Use canonical index if IndexSpec is empty (placeholder), otherwise use IndexSpec's columns
    using FinalColSeq = std::conditional_t<std::is_same_v<ColSeq, std::integer_sequence<int>>,
                                           CanonicalIntSeq, ColSeq>;
    using FullVerIndexSpec = mir::IndexSpecT<Schema, FinalColSeq, FULL_VER>;
    using type = std::tuple<FullVerIndexSpec>;
  };

  // Extract from TailOps
  using TailSpecs = typename ExtractDestinationRelationIndexSpecs<std::tuple<TailOps...>>::type;
  using HeadSpecs = typename ExtractFromOp<HeadOp>::type;

 public:
  using type = mp::mp_append<HeadSpecs, TailSpecs>;
};

// Forward declare UpdateIterators
template <typename Ops, CVar Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIterators;

// Helper to update linear iterators recursively
template <typename Op, CVar Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIteratorForOp {
  static void apply(IteratorTuple&, const auto&) {}
};

// Update ColumnSource
template <mir::CIndexSpec Index, tmp::CTuple Prefix, CVar Var, std::size_t Offset,
          typename IteratorTuple>
struct UpdateIteratorForOp<mir::ColumnSource<Index, Prefix>, Var, Offset, IteratorTuple> {
  static void apply(IteratorTuple& iterators, const auto& val) {
    if constexpr (tmp::contains_v<Var, Prefix>) {
      // Bounds check: ensure Offset is within the iterator tuple size
      constexpr std::size_t tuple_size = std::tuple_size_v<IteratorTuple>;
      if constexpr (Offset < tuple_size) {
        auto& iter = std::get<Offset>(iterators);
        // Reuse logic from state::update_iterator_with_prefixed but inline/adapted
        if constexpr (tmp::is_tuple_v<std::decay_t<decltype(val)>>) {
          std::apply([&](const auto&... vals) { iter = iter.prefix(vals...); }, val);
        } else {
          iter = iter.prefix(val);
        }
      }
    }
  }
};

// Update Aggregate
template <CVar V, mir::CIndexSpec Index, tmp::CTuple Prefix, typename Func, CVar Var,
          std::size_t Offset, typename IteratorTuple>
struct UpdateIteratorForOp<mir::Aggregate<V, Index, Prefix, Func, 0, void>, Var, Offset,
                           IteratorTuple> {
  static void apply(IteratorTuple& iterators, const auto& val) {
    if constexpr (tmp::contains_v<Var, Prefix>) {
      auto& iter = std::get<Offset>(iterators);
      if constexpr (tmp::is_tuple_v<std::decay_t<decltype(val)>>) {
        std::apply([&](const auto&... vals) { iter = iter.prefix(vals...); }, val);
      } else {
        iter = iter.prefix(val);
      }
    }
  }
};

// Update Negation
template <mir::CIndexSpec Index, tmp::CTuple Prefix, CVar Var, std::size_t Offset,
          typename IteratorTuple>
struct UpdateIteratorForOp<mir::Negation<Index, Prefix, 0, void>, Var, Offset, IteratorTuple> {
  static void apply(IteratorTuple& iterators, const auto& val) {
    if constexpr (tmp::contains_v<Var, Prefix>) {
      auto& iter = std::get<Offset>(iterators);
      if constexpr (tmp::is_tuple_v<std::decay_t<decltype(val)>>) {
        std::apply([&](const auto&... vals) { iter = iter.prefix(vals...); }, val);
      } else {
        iter = iter.prefix(val);
      }
    }
  }
};

// Update ColumnJoin -> recurse sources
// Update ColumnJoin -> recurse sources
template <typename V, typename Sources, typename Strategy, std::size_t HS, typename VHM,
          typename Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIteratorForOp<mir::ColumnJoin<V, Sources, Strategy, HS, VHM>, Var, Offset,
                           IteratorTuple> {
  static void apply(IteratorTuple& iterators, const auto& val) {
    // Recurse into sources
    UpdateIterators<Sources, Var, Offset, IteratorTuple>::apply(iterators, val);
  }
};

// Main UpdateIterators
template <typename Ops, CVar Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIterators;

template <CVar Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIterators<std::tuple<>, Var, Offset, IteratorTuple> {
  static void apply(IteratorTuple&, const auto&) {}
};

template <typename Head, typename... Tail, CVar Var, std::size_t Offset, typename IteratorTuple>
struct UpdateIterators<std::tuple<Head, Tail...>, Var, Offset, IteratorTuple> {
  static void apply(IteratorTuple& iterators, const auto& val) {
    // Update Head
    UpdateIteratorForOp<Head, Var, Offset, IteratorTuple>::apply(iterators, val);
    // Update Tail
    constexpr std::size_t head_size = mir::CountInputIterators<Head>::value;
    UpdateIterators<std::tuple<Tail...>, Var, Offset + head_size, IteratorTuple>::apply(iterators,
                                                                                        val);
  }
};

// Helper to build all indexes from a tuple of IndexSpecTs
template <typename IndexSpecList, CSemiNaiveDatabase DB>
void build_all_indexes_from_mir(DB& db, std::size_t iteration) {
  mp::mp_for_each<IndexSpecList>([&]<typename IndexSpecT>(IndexSpecT) {
    using Schema = typename IndexSpecT::schema_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    constexpr std::size_t ver = IndexSpecT::kVersion;

    // In iteration 0, DELTA_VER relations are redirected to FULL_VER in GenerateIterator.
    // We must ensure the index exists on the correct relation.
    auto& rel = [&]() -> auto& {
      if constexpr (ver == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
      }
      return get_relation_by_schema<Schema, ver>(db);
    }();

    auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    // std::cout << "build_all_indexes_from_mir: ensuring index " << runtime_spec.to_string() << "
    // ver " << IndexSpecT::version << std::endl;
    rel.ensure_index(runtime_spec);
  });
}

/**
 * @brief A helper function to add canonicalized index to MIR destination relation (incremental
 * relations are relations in rule heads), if no index exists in its NEW/DELTA/FULL version,
 *
 * @tparam MIRPlan A std::tuple of the *remaining* MIR operations to execute
 * @tparam DB The type of the runtime database
 * @param db The runtime database
 */
template <typename MIRPlan, CSemiNaiveDatabase DB>
void build_canonicalized_index_to_mir_destination_relation(DB& db) {
  // mp_for_each over MIRPlan collect all destination relations
  mp::mp_for_each<MIRPlan>([&]<typename Op>(Op) {
    if constexpr (mir::is_destination_relation_v<Op>) {
      // get the schema
      using Schema = typename Op::schema_type;
      auto& incr_new = get_relation_by_schema<Schema, NEW_VER>(db);
      auto& incr_delta = get_relation_by_schema<Schema, DELTA_VER>(db);
      auto& incr_full = get_relation_by_schema<Schema, FULL_VER>(db);
      // Construct canonical index spec from schema arity
      auto canonical_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        return IndexSpec{{Cols...}};
      }(std::make_integer_sequence<int, Schema::Arity>{});
      incr_new.ensure_index(canonical_spec);
      incr_delta.ensure_index(canonical_spec);
      incr_full.ensure_index(canonical_spec);
    }
  });
}

/**
 * @brief The main template metaprogramming interpreter for pipeline execution.
 *
 * @details This is the core of the pipeline execution engine. It recursively
 * executes pipeline operations by pattern matching on the head operation and
 * dispatching to appropriate specializations. Uses tuple-based state management
 * with precompiled position maps from the Pipeline.
 *
 * Execution model:
 * 1. Pattern match on the first MIR operation in the plan
 * 2. Execute that operation, which may:
 *    - Bind new variables (ColumnJoin, LetClause) - updates StateTuple
 *    - Update iterators (ColumnJoin) - updates IteratorTuple
 *    - Write facts (DestinationRelation)
 *    - Continue with the rest of the plan
 * 3. Recursively execute the remaining operations
 *
 * @tparam MIRPlan A std::tuple of the *remaining* MIR operations to execute
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam OutputFullIteratorPosMap Compile-time hana::map mapping FULL_VER IndexSpecT types to
 * positions
 * @tparam StateTuple Runtime tuple of variable values (fixed size, all variables)
 * @tparam IteratorTuple Runtime tuple of iterators (fixed size, one per unique IndexSpecT)
 * @tparam OutputFullIteratorTuple Runtime tuple of full index iterators (fixed size, one per
 * DestinationRelation)
 */
template <typename MIRPlan, typename VarPosMap, std::size_t IteratorOffset,
          typename OutputFullIteratorPosMap, typename StateTuple, typename IteratorTuple,
          typename OutputFullIteratorTuple>
struct PipelineExecutor;

// --- Base Case: No more operations ---
template <typename VarPosMap, std::size_t IteratorOffset, typename OutputFullIteratorPosMap,
          typename StateTuple, typename IteratorTuple, typename OutputFullIteratorTuple>
struct PipelineExecutor<std::tuple<>, VarPosMap, IteratorOffset, OutputFullIteratorPosMap,
                        StateTuple, IteratorTuple, OutputFullIteratorTuple> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB&, StateTuple&, IteratorTuple&, OutputFullIteratorTuple&, std::size_t,
                      bool) {}
};

// --- Recursive Case: Process the HeadOp ---
template <typename HeadOp, typename... TailOps, typename VarPosMap, std::size_t IteratorOffset,
          typename OutputFullIteratorPosMap, typename StateTuple, typename IteratorTuple,
          typename OutputFullIteratorTuple>
struct PipelineExecutor<std::tuple<HeadOp, TailOps...>, VarPosMap, IteratorOffset,
                        OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                        OutputFullIteratorTuple> {
  // This is the recursive call for the *rest* of the plan
  template <CSemiNaiveDatabase DB>
  static void execute_tail(DB& db, StateTuple& state, IteratorTuple& iterators,
                           OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                           bool non_iterative) {
    constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;
    PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                     OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                     OutputFullIteratorTuple>::execute(db, state, iterators, output_full_iterators,
                                                       iteration, non_iterative);
  }
};

/**
 * @brief Specialization for ColumnJoin
 */
template <CVar VarToBind, tmp::CTuple Sources, typename Strategy, typename... TailOps,
          typename VarPosMap, std::size_t IteratorOffset, typename OutputFullIteratorPosMap,
          typename StateTuple, typename IteratorTuple, typename OutputFullIteratorTuple,
          std::size_t HandleStart, typename VarToHandlesMap>
struct PipelineExecutor<
    std::tuple<mir::ColumnJoin<VarToBind, Sources, Strategy, HandleStart, VarToHandlesMap>,
               TailOps...>,
    VarPosMap, IteratorOffset, OutputFullIteratorPosMap, StateTuple, IteratorTuple,
    OutputFullIteratorTuple> {

  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    // 1. Get all iterators for the intersection linearly starting at IteratorOffset
    auto iter_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::tuple_cat([&]<std::size_t I>() {
        // Get the I-th source type
        using SourceType = std::tuple_element_t<I, Sources>;
        constexpr std::size_t count = mir::CountInputIterators<SourceType>::value;
        if constexpr (count > 0) {
          // Only if the source consumes an iterator, we try to fetch it
          // Note: This assumes source consumes exactly 1 iterator if count > 0
          // For nested joins that consume multiple, we might need to unpack all of them
          return std::make_tuple(
              std::get<IteratorOffset +
                       mir::CountInputIterators<typename tmp::take_t<I, Sources>>::value>(
                  iterators));
        } else {
          return std::tuple<>{};
        }
      }.template operator()<Is>()...);
    }(std::make_index_sequence<std::tuple_size_v<Sources>>{});

    // 2. Get relation logic (same as before)
    using FirstSource = std::tuple_element_t<0, Sources>;
    // Extract IndexSpecT from ColumnSource template parameters
    using FirstIndexSpecT = typename detail::ExtractIndexSpecT<FirstSource>::type;
    using FirstSchema = typename FirstIndexSpecT::schema_type;
    constexpr std::size_t first_ver = FirstIndexSpecT::kVersion;
    auto& rel = [&]() -> auto& {
      if constexpr (first_ver == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<FirstSchema, FULL_VER>(db);
        }
        return get_relation_by_schema<FirstSchema, DELTA_VER>(db);
      } else {
        return get_relation_by_schema<FirstSchema, first_ver>(db);
      }
    }();

    // 3. Call column_intersect
    auto intersection = rel.column_intersect(iter_tuple);

    // 4. Loop
    for (const auto& new_val : intersection) {
      // Set variable
      state::set_value_in_tuple<VarPosMap, StateTuple, VarToBind>(state, new_val);

      // Shallow copy iterators
      auto iterators_copy = iterators;

      // Update future iterators (TailOps) using UpdateIterators
      // Offset for TailOps starts after current Join's iterators (IteratorOffset + Count<HeadOp>)
      constexpr std::size_t HeadSize = mir::CountInputIterators<
          mir::ColumnJoin<VarToBind, Sources, Strategy, HandleStart, VarToHandlesMap>>::value;
      constexpr std::size_t TailOffset = IteratorOffset + HeadSize;

      detail::UpdateIterators<std::tuple<TailOps...>, VarToBind, TailOffset, IteratorTuple>::apply(
          iterators_copy, new_val);

      // Recurse
      PipelineExecutor<std::tuple<TailOps...>, VarPosMap, TailOffset, OutputFullIteratorPosMap,
                       StateTuple, IteratorTuple,
                       OutputFullIteratorTuple>::execute(db, state, iterators_copy,
                                                         output_full_iterators, iteration,
                                                         non_iterative);
    }
  }
};

// --- Specialization for Aggregate ---
template <CVar VarToBind, mir::CIndexSpec Index, tmp::CTuple PrefixVars, typename AggFunc,
          typename... TailOps, typename VarPosMap, std::size_t IteratorOffset,
          typename OutputFullIteratorPosMap, typename StateTuple, typename IteratorTuple,
          typename OutputFullIteratorTuple, std::size_t HandleStart, typename VarToHandlesMap>
struct PipelineExecutor<
    std::tuple<mir::Aggregate<VarToBind, Index, PrefixVars, AggFunc, HandleStart, VarToHandlesMap>,
               TailOps...>,
    VarPosMap, IteratorOffset, OutputFullIteratorPosMap, StateTuple, IteratorTuple,
    OutputFullIteratorTuple> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    // 1. Get the root iterator directly from the relation (not from tuple, which may be prefixed)
    // The iterator in the tuple may have been updated by previous operations, so we need the root
    using Schema = typename Index::schema_type;
    constexpr std::size_t index_ver = Index::kVersion;
    auto& rel = [&]() -> auto& {
      if constexpr (index_ver == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
        return get_relation_by_schema<Schema, DELTA_VER>(db);
      } else {
        return get_relation_by_schema<Schema, index_ver>(db);
      }
    }();

    using ColSeq = typename Index::column_indexes_type;
    auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    // Ensure the index exists and is built before using it
    auto root_handle = rel.ensure_index(runtime_spec).root();

    // 2. Init accumulator
    using AccType = typename AggFunc::ResultType;
    AccType acc = AggFunc::init();

    // 3. Aggregate
    auto cur_handle = root_handle;
    bool prefix_valid = true;

    // Apply lookups
    [&]<typename... Vars>(std::tuple<Vars...>) {
      return (([&]() {
                if (!prefix_valid)
                  return;
                if (!cur_handle.valid()) {
                  prefix_valid = false;
                  return;
                }

                auto val_encoded = state::get_value_from_tuple<VarPosMap, StateTuple, Vars>(state);
                cur_handle = cur_handle.prefix(static_cast<std::uint32_t>(val_encoded));

                if (!cur_handle.valid())
                  prefix_valid = false;
              }()),
              ...);
    }(PrefixVars{});

    if (prefix_valid && cur_handle.valid()) {
      using ColSeq = typename Index::column_indexes_type;
      using Schema = typename Index::schema_type;
      constexpr int LastColIdx = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        int cols[] = {Cols...};
        return cols[sizeof...(Cols) - 1];
      }(ColSeq{});
      using AttrTuple = typename Schema::attr_ts_type;
      using ValType = std::tuple_element_t<LastColIdx, AttrTuple>;

      if (cur_handle.is_leaf()) {
        // At a leaf node, count the number of rows for aggregation
        // For a 1-ary relation, degree() returns the number of rows with this value
        // (degree() = end_ - begin_ for leaf nodes, same as rows().size())
        std::size_t row_count = cur_handle.rows().size();
        // For AggCount, directly add row_count (more efficient than loop)
        if constexpr (std::is_same_v<AggFunc, AggCount>) {
          acc = static_cast<AccType>(acc + row_count);
        } else {
          // For other aggregations, iterate over rows
          for (std::size_t i = 0; i < row_count; ++i) {
            ValType val{};
            acc = AggFunc::apply(acc, val);
          }
        }
      } else {
        for (const auto& val : cur_handle.values()) {
          auto decoded_val = decode_from_size_t<ValType>(val);
          acc = AggFunc::apply(acc, decoded_val);
        }
      }
    }

    // 4. Bind var
    auto acc_encoded = encode_to_size_t(acc);
    state::set_value_in_tuple<VarPosMap, StateTuple, VarToBind>(state, acc_encoded);

    // 5. Recurse
    using HeadOp =
        mir::Aggregate<VarToBind, Index, PrefixVars, AggFunc, HandleStart, VarToHandlesMap>;
    constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;  // = 1
    PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                     OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                     OutputFullIteratorTuple>::execute(db, state, iterators, output_full_iterators,
                                                       iteration, non_iterative);
  }
};

// --- Specialization for LetClause ---
template <CVar VarToBind, tmp::CTuple CapturedVars, auto Func, typename... TailOps,
          typename VarPosMap, std::size_t IteratorOffset, typename OutputFullIteratorPosMap,
          typename StateTuple, typename IteratorTuple, typename OutputFullIteratorTuple>
struct PipelineExecutor<std::tuple<LetClause<VarToBind, CppExpr<CapturedVars, Func>>, TailOps...>,
                        VarPosMap, IteratorOffset, OutputFullIteratorPosMap, StateTuple,
                        IteratorTuple, OutputFullIteratorTuple> {

  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    using TypesTuple = function_arg_types_t<decltype(Func)>;
    auto dep_vals =
        state::extract_values_tuple_decoded<VarPosMap, StateTuple, CapturedVars, TypesTuple>(state);
    auto new_val = std::apply(Func, dep_vals);
    auto new_val_encoded = encode_to_size_t(new_val);
    state::set_value_in_tuple<VarPosMap, StateTuple, VarToBind>(state, new_val_encoded);

    using HeadOp = LetClause<VarToBind, CppExpr<CapturedVars, Func>>;
    constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;  // 0
    PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                     OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                     OutputFullIteratorTuple>::execute(db, state, iterators, output_full_iterators,
                                                       iteration, non_iterative);
  }
};

// --- Specialization for IfClause ---
template <tmp::CTuple BoundVars, auto Func, typename... TailOps, typename VarPosMap,
          std::size_t IteratorOffset, typename OutputFullIteratorPosMap, typename StateTuple,
          typename IteratorTuple, typename OutputFullIteratorTuple>
struct PipelineExecutor<std::tuple<IfClause<CppExpr<BoundVars, Func>>, TailOps...>, VarPosMap,
                        IteratorOffset, OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                        OutputFullIteratorTuple> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    using TypesTuple = function_arg_types_t<decltype(Func)>;
    auto dep_vals =
        state::extract_values_tuple_decoded<VarPosMap, StateTuple, BoundVars, TypesTuple>(state);
    bool result = std::apply(Func, dep_vals);
    if (result) {
      using HeadOp = IfClause<CppExpr<BoundVars, Func>>;
      constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;  // 0
      PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                       OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                       OutputFullIteratorTuple>::execute(db, state, iterators,
                                                         output_full_iterators, iteration,
                                                         non_iterative);
    }
  }
};

// --- Specialization for Negation ---
template <mir::CIndexSpec Index, tmp::CTuple PrefixVars, typename... TailOps, typename VarPosMap,
          std::size_t IteratorOffset, typename OutputFullIteratorPosMap, typename StateTuple,
          typename IteratorTuple, typename OutputFullIteratorTuple, std::size_t HandleStart,
          typename VarToHandlesMap>
struct PipelineExecutor<
    std::tuple<mir::Negation<Index, PrefixVars, HandleStart, VarToHandlesMap>, TailOps...>,
    VarPosMap, IteratorOffset, OutputFullIteratorPosMap, StateTuple, IteratorTuple,
    OutputFullIteratorTuple> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    // 1. Get root iterator from the relation (not from tuple, which may be partially prefixed)
    using Schema = typename Index::schema_type;
    constexpr std::size_t index_ver = Index::kVersion;
    auto& rel = [&]() -> auto& {
      if constexpr (index_ver == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
        return get_relation_by_schema<Schema, DELTA_VER>(db);
      } else {
        return get_relation_by_schema<Schema, index_ver>(db);
      }
    }();

    using ColSeq = typename Index::column_indexes_type;
    auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    // Ensure the index exists and get root handle
    auto cur_handle = rel.ensure_index(runtime_spec).root();

    // 2. Apply prefix navigation with bound variables
    bool prefix_valid = true;
    [&]<typename... Vars>(std::tuple<Vars...>) {
      return (([&]() {
                if (!prefix_valid)
                  return;
                if (!cur_handle.valid()) {
                  prefix_valid = false;
                  return;
                }
                auto val_encoded = state::get_value_from_tuple<VarPosMap, StateTuple, Vars>(state);
                cur_handle = cur_handle.prefix(static_cast<std::uint32_t>(val_encoded));
                if (!cur_handle.valid())
                  prefix_valid = false;
              }()),
              ...);
    }(PrefixVars{});

    // 3. Check if match exists (anti-join: continue only if NOT exists)
    // A tuple exists if prefix navigation led to a valid leaf node
    bool exists = prefix_valid && cur_handle.valid() && cur_handle.is_leaf();

    if (!exists) {
      using HeadOp = mir::Negation<Index, PrefixVars, HandleStart, VarToHandlesMap>;
      constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;  // 1
      PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                       OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                       OutputFullIteratorTuple>::execute(db, state, iterators,
                                                         output_full_iterators, iteration,
                                                         non_iterative);
    }
  }
};

/**
 * @brief Helper to get an iterator from OutputFullIteratorTuple using OutputFullIteratorPosMap.
 *
 * @details Similar to GenerateIterator but for output_full_iterators. Looks up the iterator
 * from the output_full_iterators tuple using the FULL_VER IndexSpecT.
 *
 * @tparam DestinationRelationOp The DestinationRelation MIR operation
 * @tparam OutputFullIteratorPosMap Compile-time hana::map mapping FULL_VER IndexSpecT types to
 * positions
 * @tparam OutputFullIteratorTuple Runtime tuple of full index iterators
 */
template <typename DestinationRelationOp, typename OutputFullIteratorPosMap,
          typename OutputFullIteratorTuple>
struct GetOutputFullIterator;

// Helper to convert std::index_sequence to std::integer_sequence<int, ...>
template <std::size_t... Is>
constexpr auto to_int_seq_helper(std::index_sequence<Is...>)
    -> std::integer_sequence<int, static_cast<int>(Is)...>;

template <CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver, mir::CIndexSpec IndexSpecT,
          typename ActiveHandlesMap, typename OutputFullIteratorPosMap,
          typename OutputFullIteratorTuple>
struct GetOutputFullIterator<
    mir::DestinationRelation<Schema, Terms, Ver, IndexSpecT, ActiveHandlesMap>,
    OutputFullIteratorPosMap, OutputFullIteratorTuple> {
  static auto get(const OutputFullIteratorTuple& output_full_iterators) {
    // 1. Convert IndexSpecT to FULL_VER IndexSpecT (same logic as
    // ExtractDestinationRelationIndexSpecs)
    using ColSeq = typename IndexSpecT::column_indexes_type;
    using CanonicalSeq = typename Schema::canonical_index_type;
    using CanonicalIntSeq = decltype(to_int_seq_helper(CanonicalSeq{}));
    using FinalColSeq = std::conditional_t<std::is_same_v<ColSeq, std::integer_sequence<int>>,
                                           CanonicalIntSeq, ColSeq>;
    using FullVerIndexSpec = mir::IndexSpecT<Schema, FinalColSeq, FULL_VER>;

    // 2. Look up the iterator position from the compile-time map
    constexpr auto iter_pos = std::decay_t<decltype(hana::at_key(
        OutputFullIteratorPosMap{}, hana::type_c<FullVerIndexSpec>))>::value;

    // 3. Get and return the iterator from the output_full_iterators tuple
    return std::get<iter_pos>(output_full_iterators);
  }
};

/**
 * @brief Specialization for DestinationRelation: executes a fact insertion
 * operation.
 *
 * @details This specialization handles the DestinationRelation MIR operation,
 * which writes a fact to a relation. The execution proceeds as:
 * 1. Get the runtime relation from the database
 * 2. Use precomputed full index iterator from output_full_iterators
 * 3. Extract runtime values for all terms from StateTuple
 * 4. Insert the fact into the relation with the appropriate semiring value
 * 5. Continue with the rest of the plan (e.g., another head clause)
 *
 * This is generated from rule heads during compilation.
 *
 * @tparam Schema The relation schema to write to
 * @tparam Terms The tuple of terms (Var, Const, or CppExpr) specifying values
 * @tparam Ver The version number (typically NEW_VER) for semi-naive evaluation
 * @tparam IndexSpecT The IndexSpec for deduplication
 * @tparam TailOps The remaining MIR operations to execute after this insertion
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam OutputFullIteratorPosMap Compile-time hana::map mapping FULL_VER IndexSpecT types to
 * positions
 * @tparam StateTuple Runtime tuple of variable values
 * @tparam IteratorTuple Runtime tuple of iterators
 * @tparam OutputFullIteratorTuple Runtime tuple of full index iterators
 */
// --- Specialization for DestinationRelation ---
template <CRelationSchema Schema, tmp::CTuple Terms, std::size_t Ver, mir::CIndexSpec IndexSpecT,
          typename ActiveHandlesMap, typename... TailOps, typename VarPosMap,
          std::size_t IteratorOffset, typename OutputFullIteratorPosMap, typename StateTuple,
          typename IteratorTuple, typename OutputFullIteratorTuple>
struct PipelineExecutor<
    std::tuple<mir::DestinationRelation<Schema, Terms, Ver, IndexSpecT, ActiveHandlesMap>,
               TailOps...>,
    VarPosMap, IteratorOffset, OutputFullIteratorPosMap, StateTuple, IteratorTuple,
    OutputFullIteratorTuple> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, StateTuple& state, IteratorTuple& iterators,
                      OutputFullIteratorTuple& output_full_iterators, std::size_t iteration,
                      bool non_iterative) {
    // DEBUG: Force error to see types
    // static_assert(IteratorOffset == 99999, "DEBUGGING TYPES");
    // static_assert(std::tuple_size_v<IteratorTuple> > 9999, "DEBUGGING TUPLE SIZE");
    // 1. Get the runtime relation
    // For non-iterative rules, if Ver is NEW_VER, write directly to FULL_VER
    auto& rel = [&]() -> auto& {
      if constexpr (Ver == NEW_VER) {
        if (non_iterative) {
          return get_relation_by_schema<Schema, FULL_VER>(db);
        }
        return get_relation_by_schema<Schema, NEW_VER>(db);
      }
      return get_relation_by_schema<Schema, Ver>(db);
    }();
    // 2. Use the precomputed full index iterator from output_full_iterators
    // The iterator is already a root iterator for the FULL_VER index, precomputed at pipeline start
    auto handle = GetOutputFullIterator<
        mir::DestinationRelation<Schema, Terms, Ver, IndexSpecT, ActiveHandlesMap>,
        OutputFullIteratorPosMap, OutputFullIteratorTuple>::get(output_full_iterators);
    // 3. Convert IndexSpecT to get ColSeq for prefix iteration
    using ColSeq = typename IndexSpecT::column_indexes_type;
    bool exists = [&]<int... Is>(std::integer_sequence<int, Is...>) {
      // Short-circuit: check validity after each prefix call
      return (
          ((handle = handle.prefix(
                state::get_term_value_tuple<
                    VarPosMap, StateTuple,
                    std::tuple_element_t<static_cast<std::size_t>(Is), Terms>  // Get I-th Term type
                    >(state))),
           handle.valid()) &&
          ...);
    }(ColSeq{});

    if (!exists) {
      auto sr_val = Schema::semiring_type::one();

      // Call push_row with unpacked arguments, directly decoding values
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        rel.push_row(
            std::make_tuple(
                decode_from_size_t<std::tuple_element_t<Is, typename Schema::attr_ts_type>>(
                    state::get_term_value_tuple<VarPosMap, StateTuple,
                                                std::tuple_element_t<Is, Terms>>(state))...),
            sr_val);
      }(std::make_index_sequence<std::tuple_size_v<Terms>>{});
    }

    // std::cout << "try to push row to relation : ";
    // std::apply(
    //     [&](auto... vals) {
    //       std::cout << "(";
    //       int n = 0;
    //       ((std::cout << (n++ ? ", " : "") << vals), ...);
    //       std::cout << ") ";
    //     },
    //     payload_vals);
    // std::cout << std::endl;

    // 4. Recurse on the rest of the plan (e.g., another 'Head' clause)
    using HeadOp = mir::DestinationRelation<Schema, Terms, Ver, IndexSpecT>;
    constexpr std::size_t HeadSize = mir::CountInputIterators<HeadOp>::value;  // 0
    PipelineExecutor<std::tuple<TailOps...>, VarPosMap, IteratorOffset + HeadSize,
                     OutputFullIteratorPosMap, StateTuple, IteratorTuple,
                     OutputFullIteratorTuple>::execute(db, state, iterators, output_full_iterators,
                                                       iteration, non_iterative);
  }
};

/**
 * @brief Helper to create a root iterator for a single IndexSpecT.
 *
 * @details Gets the relation, accesses the index, and returns the root iterator.
 * Handles DELTA_VER redirection in iteration 0.
 *
 * @tparam IndexSpecT The IndexSpecT type
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 * @param iteration The current iteration number
 * @return The root iterator for this IndexSpecT
 */
template <mir::CIndexSpec IndexSpecT, CSemiNaiveDatabase DB>
auto create_root_iterator(DB& db, std::size_t iteration) {
  using Schema = typename IndexSpecT::schema_type;
  using ColSeq = typename IndexSpecT::column_indexes_type;
  constexpr std::size_t ver = IndexSpecT::kVersion;

  // Get the relation (handle DELTA_VER redirection in iteration 0)
  auto& rel = [&]() -> auto& {
    if constexpr (ver == DELTA_VER) {
      if (iteration == 0) {
        return get_relation_by_schema<Schema, FULL_VER, DB>(db);
      }
      return get_relation_by_schema<Schema, DELTA_VER, DB>(db);
    } else {
      return get_relation_by_schema<Schema, ver, DB>(db);
    }
  }();

  // Get the index and return root iterator
  auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
    return IndexSpec{{Cols...}};
  }(ColSeq{});
  auto& index = rel.get_index(runtime_spec);
  return index.root();
}

// Helper to create initial iterators from a linear list of IndexSpecs
template <typename IndexSpecsTuple, CSemiNaiveDatabase DB>
auto create_initial_iterator_tuple_linear(DB& db, std::size_t iteration) {
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(
        create_root_iterator<std::tuple_element_t<Is, IndexSpecsTuple>, DB>(db, iteration)...);
  }(std::make_index_sequence<std::tuple_size_v<IndexSpecsTuple>>{});
}

/**
 * @brief Helper to create a full index iterator for a single FULL_VER IndexSpecT.
 *
 * @details Gets the FULL_VER relation, accesses the index, and returns the root iterator.
 * This is used for output_full_iterators which always use FULL_VER.
 *
 * @tparam IndexSpecT The IndexSpecT type (must be FULL_VER)
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 * @return The root iterator for this FULL_VER IndexSpecT
 */
template <mir::CIndexSpec IndexSpecT, CSemiNaiveDatabase DB>
auto create_full_index_iterator(DB& db) {
  static_assert(IndexSpecT::kVersion == FULL_VER,
                "IndexSpecT must be FULL_VER for output_full_iterators");
  using Schema = typename IndexSpecT::schema_type;
  using ColSeq = typename IndexSpecT::column_indexes_type;

  // Always use FULL_VER relation
  auto& rel_full = get_relation_by_schema<Schema, FULL_VER, DB>(db);

  // Get the index and return root iterator
  auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
    return IndexSpec{{Cols...}};
  }(ColSeq{});
  auto& index = rel_full.get_index(runtime_spec);
  return index.root();
}

/**
 * @brief Creates full index iterators for all DestinationRelation IndexSpecT in
 * OutputFullIteratorPosMap order.
 *
 * @details Extracts IndexSpecT types directly from OutputFullIteratorPosMap in position order
 * (0, 1, 2, ...), then creates full index iterators for each. This guarantees the order
 * matches exactly what's used in OutputFullIteratorPosMap lookups.
 *
 * @tparam OutputFullIteratorPosMap Compile-time hana::map mapping FULL_VER IndexSpecT types to
 * positions
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 * @return A tuple of full index iterators in OutputFullIteratorPosMap order
 */
template <typename OutputFullIteratorPosMap, CSemiNaiveDatabase DB>
auto create_initial_output_full_iterator_tuple(DB& db) {
  constexpr auto map_size = hana::size(OutputFullIteratorPosMap{});
  using IndexSpecsTuple = typename state::ExtractTypesFromPosMap<OutputFullIteratorPosMap,
                                                                 decltype(map_size)::value>::type;

  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(
        create_full_index_iterator<typename std::tuple_element_t<Is, IndexSpecsTuple>::type, DB>(
            db)...);
  }(std::make_index_sequence<std::tuple_size_v<IndexSpecsTuple>>{});
}
}  // namespace detail

/**
 * @brief Executes a pre-compiled Pipeline.
 *
 * @details Executes a Pipeline using tuple-based state management with precompiled
 * position maps. Creates initial state tuples (variables and iterators) and
 * executes the pipeline operations.
 *
 * @tparam Pipeline The Pipeline to execute (must satisfy CPipeline)
 * @tparam DB The semi-naive database type
 * @param runtime_db The runtime database
 * @param iteration The current iteration number (0 for first iteration)
 * @param non_iterative If true, writes directly to FULL_VER instead of NEW_VER
 */
template <typename Pipeline, CSemiNaiveDatabase DB>
void execute_pipeline(DB& runtime_db, std::size_t iteration = 0, bool non_iterative = false) {
  static_assert(mir::is_pipeline_v<Pipeline>, "Pipeline must satisfy CPipeline");

  using MIROps = typename Pipeline::mir_ops_type;
  using VarPosMap = typename Pipeline::var_pos_map_type;
  // IteratorPosMap is invalid/removed
  // using IteratorPosMap = typename Pipeline::iterator_pos_map_type;

  // 1. Use Pipeline::source_specs_type (computed from AST, not MIR)
  // This is faster and more accurate than extracting from MIR
  using ExtractedIndexSpecs = typename Pipeline::source_specs_type;

  // 4. Use explicit DestinationRelation IndexSpecTs from Pipeline
  using DestinationRelationIndexSpecs = typename Pipeline::destination_specs_type;

  // 2. Build all indexes explicitly (both inputs and outputs)
  // Input specs
  detail::build_all_indexes_from_mir<ExtractedIndexSpecs>(runtime_db, iteration);
  // Output specs (if any)
  detail::build_all_indexes_from_mir<DestinationRelationIndexSpecs>(runtime_db, iteration);

  // 3. Create initial iterator tuple directly from ExtractedIndexSpecs (linear)
  // This ensures a 1:1 mapping with MIR operations logic
  auto iterators =
      detail::create_initial_iterator_tuple_linear<ExtractedIndexSpecs, DB>(runtime_db, iteration);

  // 6. OutputFullIteratorPosMap logic remains the same (unique by type)
  // Make them unique (each IndexSpecT appears only once)
  using UniqueDestRelIndexSpecsList = mp::mp_unique<DestinationRelationIndexSpecs>;
  using UniqueDestRelIndexSpecs = mp::mp_rename<UniqueDestRelIndexSpecsList, std::tuple>;
  // Create the map from FULL_VER IndexSpecT type to position
  using OutputFullIteratorPosMap =
      typename mir::ComputeIndexSpecToIteratorPosMap<UniqueDestRelIndexSpecs>::type;

  // 5. Create initial output_full_iterators tuple directly from OutputFullIteratorPosMap
  auto output_full_iterators =
      detail::create_initial_output_full_iterator_tuple<OutputFullIteratorPosMap, DB>(runtime_db);

  // 6. Create initial state tuple (all variables, fixed size based on join plan)
  // All values initialized to MAX_INT (unbound variables)
  constexpr auto state_size = hana::size(VarPosMap{});
  constexpr std::size_t max_val = std::numeric_limits<std::size_t>::max();
  auto state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(((void)Is, max_val)...);
  }(std::make_index_sequence<decltype(state_size)::value>{});

  // 7. Start the recursive pipeline executor with IteratorOffset = 0
  using StateTuple = decltype(state);
  using IteratorTuple = decltype(iterators);
  using OutputFullIteratorTuple = decltype(output_full_iterators);
  detail::PipelineExecutor<MIROps, VarPosMap, 0, OutputFullIteratorPosMap, StateTuple,
                           IteratorTuple, OutputFullIteratorTuple>::execute(runtime_db, state,
                                                                            iterators,
                                                                            output_full_iterators,
                                                                            iteration,
                                                                            non_iterative);
}

}  // namespace SRDatalog
