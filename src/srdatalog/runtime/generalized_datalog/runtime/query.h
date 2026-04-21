/**
 * @file runtime/query.h
 * @brief High-level query execution functions.
 *
 * @details This file provides high-level functions for executing queries,
 * including fixpoint evaluation, delta computation, and semi-naive evaluation
 * support.
 */

#pragma once

#include "../query.h"  // For get_relation_by_schema
#include "logging.h"
#include "mir.h"               // For mir::CompileRuleToMIR and MIR definitions
#include "relation_col.h"      // For column_intersect
#include "runtime/executor.h"  // For execute_pipeline
#include "runtime/state.h"     // For get_type_name
#include <climits>
#include <iostream>

namespace SRDatalog {
using namespace AST;

namespace detail {
// Forward declaration for IterateInsertFull (defined later in this namespace)
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateInsertFull;
// Forward declaration for IterateComputeDelta (defined later in this namespace)
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateComputeDelta;

/**
 * @brief Interpreter for FixpointPlan execution plan.
 *
 * @details Processes a flat sequence of fixpoint-level operations (BuildIndex,
 *          ExecutePipeline, RebuildIndex) sequentially. When ExecutePipeline
 *          is encountered, it executes the pipeline (nested for-loops) and
 *          then continues to the next operation.
 *
 * @tparam ExecutionPlan A std::tuple of fixpoint-level operations
 */
template <typename ExecutionPlan>
struct MIRExecutor;

// Base case: empty plan
template <>
struct MIRExecutor<std::tuple<>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB&, std::size_t) {
    // Done
  }
};

// Specialization for BuildIndex
template <mir::CIndexSpec IndexSpecT, typename... TailOps>
struct MIRExecutor<std::tuple<mir::CreateIndex<IndexSpecT>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing CreateIndex instruction: "
    //           << get_type_name<mir::CreateIndex<IndexSpecT>>();
    using Schema = typename IndexSpecT::schema_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    constexpr std::size_t ver = IndexSpecT::kVersion;

    // In iteration 0, DELTA_VER relations are redirected to FULL_VER
    auto& rel = [&]() -> auto& {
      if constexpr (ver == DELTA_VER) {
        return get_relation_by_schema<Schema, FULL_VER>(db);
      }
      return get_relation_by_schema<Schema, ver>(db);
    }();

    auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    rel.ensure_index(runtime_spec);

    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for RebuildIndex
template <mir::CIndexSpec IndexSpecT, typename... TailOps>
struct MIRExecutor<std::tuple<mir::RebuildIndex<IndexSpecT>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing RebuildIndex instruction: "
    //           << get_type_name<mir::RebuildIndex<IndexSpecT>>();
    using Schema = typename IndexSpecT::schema_type;
    using ColSeq = typename IndexSpecT::column_indexes_type;
    constexpr std::size_t ver = IndexSpecT::kVersion;

    auto& rel = get_relation_by_schema<Schema, ver>(db);
    auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    rel.ensure_index(runtime_spec);

    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for CheckSize - conditional execution
template <CRelationSchema Schema, std::size_t Ver, typename... TailOps>
struct MIRExecutor<std::tuple<mir::CheckSize<Schema, Ver>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing CheckSize instruction: "
    //           << get_type_name<mir::CheckSize<Schema, Ver>>();
    auto& rel = get_relation_by_schema<Schema, Ver>(db);
    if (rel.size() == 0) {
      LOG_DEBUG << "CheckSize: relation is empty, skipping remaining operations";
      // Early return - skip remaining operations
      return;
    }
    // LOG_DEBUG << "CheckSize: relation size = " << rel.size() << ", continuing";
    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for ClearRelation
template <CRelationSchema Schema, std::size_t Ver, typename... TailOps>
struct MIRExecutor<std::tuple<mir::ClearRelation<Schema, Ver>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing ClearRelation instruction: "
    //           << get_type_name<mir::ClearRelation<Schema, Ver>>();
    auto& rel = get_relation_by_schema<Schema, Ver>(db);
    rel.clear();
    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for InsertFromRelation
template <CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer,
          mir::CIndexSpec IndexSpecT, typename... TailOps>
struct MIRExecutor<
    std::tuple<mir::InsertFromRelation<Schema, FromVer, ToVer, IndexSpecT>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing InsertFromRelation instruction: "
    //           << get_type_name<mir::InsertFromRelation<Schema, FromVer, ToVer, IndexSpecT>>();
    auto& from_rel = get_relation_by_schema<Schema, FromVer>(db);
    auto& to_rel = get_relation_by_schema<Schema, ToVer>(db);

    // Use the precomputed index from IndexSpecT - it must be built beforehand,
    // if not throw an error, compiler is broken
    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto index_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});
    const auto& from_index = from_rel.get_index(index_spec);
    const auto& from_root = from_index.root();

    // Iterate over all tuples in from_rel and insert into to_rel
    // LOG_DEBUG << "Full relation size: >>>>>>>>>>>>>>>>>>>>>>>>> BEFORE INSERT " << to_rel.size();
    std::array<std::size_t, Schema::Arity> values{};
    // Pass IndexSpecT type (not runtime value) for compile-time unrolling
    IterateInsertFull<Schema::Arity, decltype(from_root), Schema>::iterate(from_root, to_rel,
                                                                           values, IndexSpecT{});
    // print full relation
    // LOG_DEBUG << "From relation size: >>>>>>>>>>>>>>>>>>>>>>>>> AFTER INSERT " <<
    // from_rel.size(); LOG_DEBUG << "Full relation size: >>>>>>>>>>>>>>>>>>>>>>>>> AFTER INSERT "
    // << to_rel.size(); Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for SwapRelations
template <CRelationSchema Schema, typename... TailOps>
struct MIRExecutor<std::tuple<mir::SwapRelations<Schema>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing SwapRelations instruction: "
    //           << get_type_name<mir::SwapRelations<Schema>>();
    auto& delta = get_relation_by_schema<Schema, DELTA_VER>(db);
    auto& newt = get_relation_by_schema<Schema, NEW_VER>(db);
    std::swap(delta, newt);
    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for SetVersion
template <CRelationSchema Schema, std::size_t Ver, typename... TailOps>
struct MIRExecutor<std::tuple<mir::SetVersion<Schema, Ver>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing SetVersion instruction: "
    //           << get_type_name<mir::SetVersion<Schema, Ver>>();
    auto& rel = get_relation_by_schema<Schema, Ver>(db);
    rel.set_version(Ver);
    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for ComputeDelta
template <CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
          mir::CIndexSpec IndexSpecT, typename... TailOps>
struct MIRExecutor<
    std::tuple<mir::ComputeDelta<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing ComputeDelta instruction: "
    //           << get_type_name<mir::ComputeDelta<Schema, FromVer, ToVer, DeltaVer,
    //           IndexSpecT>>();
    auto& new_rel = get_relation_by_schema<Schema, FromVer>(db);
    auto& full_rel = get_relation_by_schema<Schema, ToVer>(db);
    auto& delta_rel = get_relation_by_schema<Schema, DeltaVer>(db);

    // Clear delta relation at the start
    delta_rel.clear();

    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto index_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});

    // Ensure indexes exist
    new_rel.ensure_index(index_spec);
    const auto& new_idx = new_rel.get_index(index_spec);

    if (iteration == 0) {
      full_rel.ensure_index(index_spec, false);
    } else {
      full_rel.ensure_index(index_spec, false);  // Safe check
    }
    const auto& full_idx = full_rel.get_index(index_spec);

    // Compute Delta = NEW - FULL by iterating over NEW index
    // For each tuple in NEW, check if it exists in FULL
    // If not, add to DELTA
    // Use index-based iteration (similar to InsertFromRelation)
    std::array<std::size_t, Schema::Arity> values{};
    auto new_root = new_idx.root();
    // Use IterateComputeDelta to iterate over NEW index and compute difference
    if (new_root.valid()) {
      detail::IterateComputeDelta<Schema::Arity, decltype(new_root), Schema>::iterate(
          new_root, full_idx, delta_rel, values, IndexSpecT{});
    }

    // After populating DELTA relation intern values, ensure canonical DELTA index is built
    // For CPU: ensure_index builds the index FROM the relation intern values, so no reconstruction
    // needed For GPU: ComputeDelta builds index first, then reconstructs relation from it (handled
    // in GPU executor)
    delta_rel.ensure_index(
        index_spec, true);  // Force rebuild canonical DELTA index from relation intern values

    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for MergeIndex
template <mir::CIndexSpec IndexSpecT, typename... TailOps>
struct MIRExecutor<std::tuple<mir::MergeIndex<IndexSpecT>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    using Schema = typename IndexSpecT::schema_type;
    static_assert(IndexSpecT::kVersion == FULL_VER, "MergeIndex must be for FULL_VER index");

    auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(db);
    auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(db);

    using ColSeq = typename IndexSpecT::column_indexes_type;
    auto index_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
      return IndexSpec{{Cols...}};
    }(ColSeq{});

    // Ensure indexes exist
    if (iteration == 0) {
      full_rel.ensure_index(index_spec, false);
    } else {
      full_rel.ensure_index(index_spec, false);  // Safe check
    }
    auto& full_idx = full_rel.get_index(index_spec);

    delta_rel.ensure_index(index_spec);
    const auto& delta_idx = delta_rel.get_index(index_spec);

    // Merge DELTA index into FULL index
    if (!delta_idx.empty()) {
      std::size_t row_offset = full_idx.size();
      full_idx.merge(delta_idx, row_offset);
    }

    // Continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

// Specialization for ExecutePipeline
template <typename Pipeline, typename... TailOps>
  requires mir::is_pipeline_v<Pipeline>
struct MIRExecutor<std::tuple<mir::ExecutePipeline<Pipeline>, TailOps...>> {
  template <CSemiNaiveDatabase DB>
  static void execute(DB& db, std::size_t iteration) {
    // LOG_DEBUG << "Executing ExecutePipeline instruction: "
    //           << get_type_name<mir::ExecutePipeline<Pipeline>>();
    // LOG_DEBUG << "Executing ExecutePipeline instruction";
    // Execute the pipeline using the new tuple-based execution engine
    execute_pipeline<Pipeline>(db, iteration, false);
    // Pipeline execution complete, continue with next operation
    MIRExecutor<std::tuple<TailOps...>>::execute(db, iteration);
  }
};

/**
 * @brief Clear all indexes from all relations in a database (for debugging).
 *
 * @details Iterates over all relations in full, delta, and newt versions
 *          and clears all their indexes. This is useful for debugging to
 *          ensure indexes are rebuilt from scratch on each query execution.
 *
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 */
template <AST::CSemiNaiveDatabase DB>
void clear_all_indexes_for_debug(DB& db) {
  using DBType = typename DB::schema_type;
  constexpr std::size_t N = std::tuple_size_v<typename DB::data_type>;

  // Clear indexes from all relations in full, delta, and newt
  mp::mp_for_each<mp::mp_iota_c<N>>([&](auto I) {
    auto& rel_full = std::get<I>(db.full);
    auto& rel_delta = std::get<I>(db.delta);
    auto& rel_newt = std::get<I>(db.newt);

    // TODO: Clear all indexes in each relation by iterating over index_specs
    // and clearing each index (using const_cast to get non-const access for clearing)
    for (const auto& spec : rel_full.get_all_indexes()) {
      try {
        auto& idx = const_cast<typename std::decay_t<decltype(rel_full)>::IndexTypeInst&>(
            rel_full.get_index(spec));
        idx.clear();
      } catch (...) {
        // Index might not exist, skip
      }
    }
    for (const auto& spec : rel_delta.get_all_indexes()) {
      try {
        auto& idx = const_cast<typename std::decay_t<decltype(rel_delta)>::IndexTypeInst&>(
            rel_delta.get_index(spec));
        idx.clear();
      } catch (...) {
        // Index might not exist, skip
      }
    }
    for (const auto& spec : rel_newt.get_all_indexes()) {
      try {
        auto& idx = const_cast<typename std::decay_t<decltype(rel_newt)>::IndexTypeInst&>(
            rel_newt.get_index(spec));
        idx.clear();
      } catch (...) {
        // Index might not exist, skip
      }
    }
  });
}

}  // namespace detail

// (defined in query.h, but may not be fully processed yet due to include order)
// this is defensive avoid include order changed break the build
template <CRule Rule>
struct GetHeadSchemas;
// mir::CompileRuleToMIR is now defined in mir.h (included above)

/**
 * @brief Add a fact to a relation in the runtime database
 * @param runtime_db The runtime database
 * @param values The values to add to the relation
 * @param semiring_val The semiring value to add to the relation
 * @tparam Schema The schema of the relation
 * @tparam Values The values to add to the relation
 * @tparam DB The type of the runtime database
 */
template <CRelationSchema Schema, typename... Values, typename DB>
inline void add_fact(DB& runtime_db, [[maybe_unused]] const auto&& semiring_val, Values... values) {
  auto& rel = get_relation_by_schema<Schema, FULL_VER>(runtime_db);
  if constexpr (has_provenance_v<typename Schema::semiring_type>) {
    // Use the schema's semiring type to get the correct one() value
    rel.push_row({values...}, Schema::semiring_type::one());
  } else {
    rel.push_row_no_prov({values...});
  }
}

/**
 * @brief Add a fact to a relation without explicit semiring value (auto uses one())
 * @param runtime_db The runtime database
 * @param values The values to add to the relation
 * @tparam Schema The schema of the relation
 * @tparam Values The values to add to the relation
 * @tparam DB The type of the runtime database
 */
template <CRelationSchema Schema, typename... Values, typename DB>
inline void add_fact_simple(DB& runtime_db, Values... values) {
  auto& rel = get_relation_by_schema<Schema, FULL_VER>(runtime_db);
  if constexpr (has_provenance_v<typename Schema::semiring_type>) {
    rel.push_row({values...}, Schema::semiring_type::one());
  } else {
    rel.push_row_no_prov({values...});
  }
}

/**
 * @brief Print the contents of a relation in the runtime database
 * @param runtime_db The runtime database
 * @param os The output stream to print to, default to std::cout
 * @param n The number of rows to print
 * @tparam Schema The schema of the relation
 * @tparam Ver The version of the relation to print
 * @tparam DB The type of the runtime database
 */
template <CRelationSchema Schema, std::size_t Ver, typename DB>
void print_relation(DB& runtime_db, std::ostream& os = std::cout, std::size_t n = 10) {
  auto& rel = get_relation_by_schema<Schema, Ver>(runtime_db);
  rel.head(n, os);
}

/**
 * @brief Print the interned (encoded) values of a relation in the runtime database
 * @param runtime_db The runtime database
 * @param os The output stream to print to, default to std::cout
 * @param n The number of rows to print
 * @tparam Schema The schema of the relation
 * @tparam Ver The version of the relation to print
 * @tparam DB The type of the runtime database
 */
template <CRelationSchema Schema, std::size_t Ver, typename DB>
void print_relation_interned(DB& runtime_db, std::ostream& os = std::cout, std::size_t n = 10) {
  auto& rel = get_relation_by_schema<Schema, Ver>(runtime_db);
  rel.head_interned(n, os);
}

// check if a tuple is in a relation
template <CRelationSchema Schema, std::size_t Ver, typename DB>
bool is_in_relation(DB& runtime_db, const std::array<std::size_t, Schema::Arity>& encoded_tuple) {
  // Access relation directly via tuple indexing to avoid get_relation_by_schema visibility issues
  // Since query.h is included, we can use IndexRelInDBRels from SRDatalog::detail namespace
  // Note: runtime/query.h has its own detail namespace, so we need to fully qualify
  using DB_SchemaTuple = typename DB::schema_type;
  // Use boost::mp11 to find the index directly
  using SchemaTuple = typename DB::schema_type;
  constexpr std::size_t rel_index = boost::mp11::mp_find<SchemaTuple, Schema>::value;
  auto& rel = [&]() -> auto& {
    if constexpr (Ver == FULL_VER) {
      return std::get<rel_index>(runtime_db.full);
    } else if constexpr (Ver == DELTA_VER) {
      return std::get<rel_index>(runtime_db.delta);
    } else if constexpr (Ver == NEW_VER) {
      return std::get<rel_index>(runtime_db.newt);
    }
  }();
  // Construct canonical index spec from schema arity (determined at compile time)
  // This uses Schema::canonical_index_type which is std::make_index_sequence<Arity>
  auto canonical_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
    return IndexSpec{{Cols...}};
  }(std::make_integer_sequence<int, Schema::Arity>{});
  // Ensure the canonical index exists (canonical index is always [0,1,2,...,arity-1])
  // This is not "generating at runtime" - the canonical index spec is determined at compile time
  const auto& idx = rel.ensure_index(canonical_spec).root();
  // reorder the tuple
  auto reordered_tuple = reorder_tuple(canonical_spec, encoded_tuple);
  return idx.prefix(reordered_tuple).valid();
}
// left value for array

template <AST::CFixpoint FP>
struct GetIncrementalRelationsFromFixpoint;

template <AST::CRule... Rules>
struct GetIncrementalRelationsFromFixpoint<AST::Fixpoint<Rules...>> {
 private:
  using RuleList = std::tuple<Rules...>;
  template <AST::CRule R>
  using GetSchemasFromRule = typename GetHeadSchemas<R>::type;
  using SchemasPerRule = mp::mp_transform<GetSchemasFromRule, RuleList>;
  using AllHeadSchemas = mp::mp_flatten<SchemasPerRule>;

 public:
  using type = tmp::TupleUnique<AllHeadSchemas>;
};

/**
 * @brief Extract rules from a Fixpoint.
 *
 * @details Returns a std::tuple of all rules in the Fixpoint.
 */
template <AST::CFixpoint FP>
struct ExtractRulesFromFixpoint;

template <AST::CRule... Rules>
struct ExtractRulesFromFixpoint<AST::Fixpoint<Rules...>> {
  using type = std::tuple<Rules...>;
};

/**
 * @brief Extract incremental relations from a tuple of rules.
 *
 * @details This trait extracts all relation schemas that appear in the head
 *          of the given rules. These are the relations that will be updated
 *          during rule execution and need delta tracking for semi-naive
 *          evaluation.
 *
 * @tparam RuleList A std::tuple of CRule types
 *
 * @example
 * ```cpp
 * using Rules = std::tuple<Rule1, Rule2>;
 * using IncrementalRelations = GetIncrementalRelationsFromRules<Rules>::type;
 * // IncrementalRelations is std::tuple<Schema1, Schema2, ...>
 * ```
 */
template <tmp::CTuple RuleList>
struct GetIncrementalRelationsFromRules;

template <AST::CRule... Rules>
struct GetIncrementalRelationsFromRules<std::tuple<Rules...>> {
 private:
  template <AST::CRule R>
  using GetSchemasFromRule = typename GetHeadSchemas<R>::type;
  using SchemasPerRule = mp::mp_transform<GetSchemasFromRule, std::tuple<Rules...>>;
  using AllHeadSchemas = mp::mp_flatten<SchemasPerRule>;

 public:
  /// @brief Unique tuple of relation schemas from all rule heads
  using type = tmp::TupleUnique<AllHeadSchemas>;
};

namespace detail {
/**
 * @brief Helper to filter IndexSpecList to only include schemas in IncrementalRelations.
 *
 * @details Filters an IndexSpecList to only keep IndexSpecs whose schema_type
 *          appears in the IncrementalRelations tuple. This ensures we only
 *          rebuild indexes for relations that were written to (in rule heads),
 *          not relations that were only read from (in rule bodies).
 *
 * @tparam IndexSpecList A std::tuple of IndexSpecT types
 * @tparam IncrementalRelations A std::tuple of relation schemas (from rule heads)
 */
template <typename IndexSpecList, typename IncrementalRelations>
struct FilterIndexSpecsByRelations;

template <typename... IndexSpecs, typename... Schemas>
struct FilterIndexSpecsByRelations<std::tuple<IndexSpecs...>, std::tuple<Schemas...>> {
 private:
  using RelationList = std::tuple<Schemas...>;

  // Predicate: check if IndexSpecT's schema_type is in RelationList
  template <typename IndexSpecT>
  struct SchemaInList {
    using Schema = typename IndexSpecT::schema_type;
    static constexpr bool value = mp::mp_contains<RelationList, Schema>::value;
  };

 public:
  using type = mp::mp_copy_if<std::tuple<IndexSpecs...>, SchemaInList>;
};

// Helper to extract Indices from a Rule
template <typename Rule>
struct GetRuleIndices {
  using type = typename mir::CompileRuleToMIR<Rule>::Indices;
};

// Helper to collect and flatten all IndexSpecs from a RuleList
template <typename RuleList>
struct CollectAllIndexSpecs;

template <typename... Rules>
struct CollectAllIndexSpecs<std::tuple<Rules...>> {
  using type = mp::mp_flatten<std::tuple<typename mir::CompileRuleToMIR<Rules>::Indices...>>;
};

/**
 * @brief Collects canonical index specs for incremental relations.
 *
 * @details For each relation in IncrementalRelations, finds the first matching IndexSpec
 *          in AllIndexSpecs. If found, uses that IndexSpec. If not found, creates a
 *          canonical index spec (0, 1, 2, ..., arity-1).
 *
 *          Returns a compile-time tuple of IndexSpecT types that can be:
 *          1. Iterated over with mp::mp_for_each to build all indexes
 *          2. Looked up by schema type to get canonical index for DestinationRelation
 *
 * @tparam IncrementalRelations A std::tuple of CRelationSchema types
 * @tparam AllIndexSpecs A std::tuple of IndexSpecT types
 *
 * @example
 * ```cpp
 * using IncrementalRelations = std::tuple<PathSchema, ResultSchema>;
 * using AllIndexSpecs = typename detail::CollectAllIndexSpecs<Rules>::type;
 * using CanonicalIndexSpecs = detail::CollectCanonicalIndexSpecs<IncrementalRelations,
 * AllIndexSpecs>;
 *
 * // Use case 1: Iterate and build all indexes
 * mp::mp_for_each<typename CanonicalIndexSpecs::type>([&]<typename IndexSpecT>(IndexSpecT) {
 *   using Schema = typename IndexSpecT::schema_type;
 *   auto& rel = get_relation_by_schema<Schema, FULL_VER>(db);
 *   auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
 *     return IndexSpec{{Cols...}};
 *   }(typename IndexSpecT::column_indexes_type{});
 *   rel.ensure_index(runtime_spec);
 * });
 *
 * // Use case 2: Look up canonical index for a schema
 * using PathCanonical = typename CanonicalIndexSpecs::GetCanonicalSpec<PathSchema>::type;
 * ```
 */
template <typename IncrementalRelations, typename AllIndexSpecs>
struct CollectCanonicalIndexSpecs;

template <typename... Schemas, typename AllIndexSpecs>
struct CollectCanonicalIndexSpecs<std::tuple<Schemas...>, AllIndexSpecs> {
  // For each schema, get its canonical index spec
  template <typename Schema>
  using GetCanonicalSpec =
      typename mir::GetCanonicalIndexSpecForSchema<Schema, AllIndexSpecs>::type;

  /// @brief Tuple of canonical IndexSpecT types (one per incremental relation)
  using type = std::tuple<GetCanonicalSpec<Schemas>...>;

  /**
   * @brief Lookup canonical IndexSpecT for a given schema type.
   *
   * @details Finds the canonical IndexSpecT for the given Schema in the collected
   *          canonical index specs. This can be used in DestinationRelation execution
   *          to get the canonical index type.
   *
   * @tparam Schema The relation schema to look up
   */
  template <CRelationSchema Schema>
  struct GetCanonicalSpecForSchema {
    using type = typename mir::GetCanonicalIndexSpecForSchema<Schema, AllIndexSpecs>::type;
  };
};

/**
 * @brief Compile-time helper to find index position for a schema column.
 *
 * @details For a given schema column position, finds which position in the index
 *          sequence contains that column. Uses compile-time template metaprogramming.
 *
 * @tparam SchemaCol The schema column position to find
 * @tparam IndexColSeq The integer sequence of index columns
 */
template <int SchemaCol, typename IndexColSeq>
struct FindIndexPositionForSchemaCol;

template <int SchemaCol>
struct FindIndexPositionForSchemaCol<SchemaCol, std::integer_sequence<int>> {
  // Base case: schema column not found in index (should not happen for full indexes)
  // Return 0 as fallback (will cause incorrect reordering if this case is hit)
  static constexpr std::size_t value = 0;
};

template <int SchemaCol, int First, int... Rest>
struct FindIndexPositionForSchemaCol<SchemaCol, std::integer_sequence<int, First, Rest...>> {
  static constexpr std::size_t value =
      (First == SchemaCol)
          ? 0
          : (1 +
             FindIndexPositionForSchemaCol<SchemaCol, std::integer_sequence<int, Rest...>>::value);
};

/**
 * @brief Compile-time inverse reorder: convert from index order back to schema order.
 *
 * @details When iterating over an index, values are in the order of the index columns.
 *          This function converts them back to schema order (0, 1, 2, ...) for insertion.
 *          Uses compile-time template metaprogramming to unroll the reordering.
 *
 * @tparam SchemaArity The arity of the schema
 * @tparam IndexColSeq The integer sequence of index columns (from IndexSpecT)
 * @param index_ordered_values Values in index column order
 * @return Values reordered to schema order
 */
template <std::size_t SchemaArity, typename IndexColSeq>
auto inverse_reorder_tuple_ct(const std::array<std::size_t, SchemaArity>& index_ordered_values)
    -> std::array<std::size_t, SchemaArity> {
  return [&]<std::size_t... SchemaPos>(std::index_sequence<SchemaPos...>) {
    return std::array<std::size_t, SchemaArity>{
        index_ordered_values[FindIndexPositionForSchemaCol<static_cast<int>(SchemaPos),
                                                           IndexColSeq>::value]...};
  }(std::make_index_sequence<SchemaArity>{});
}

/**
 * @brief Recursive TMP helper to iterate over all tuples in an index.
 *
 * @details This helper recursively generates nested loops to iterate over
 *          all tuples in a relation index. For arity N, it generates N
 *          nested loops, each iterating over a column using column_intersect.
 *          Values are collected in index order and then reordered to schema order
 *          before insertion.
 *
 * @tparam Arity The remaining arity to process
 * @tparam NodeHandleType The type of the current node handle
 * @tparam Schema The relation schema (for semiring type)
 */
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateInsertFull;

// Helper to compute delta: iterate over NEW index, check against FULL, insert into DELTA if not
// exists
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateComputeDelta;

// Base case: arity 0, check if tuple exists in FULL, if not insert into DELTA
template <typename NodeHandleType, CRelationSchema Schema>
struct IterateComputeDelta<0, NodeHandleType, Schema> {
  template <typename FullIndexType, typename DeltaRelType, typename ValuesArray,
            mir::CIndexSpec IndexSpecT>
  static void iterate(const NodeHandleType& /* node */, FullIndexType& full_idx,
                      DeltaRelType& delta_rel, ValuesArray& values,
                      const IndexSpecT& /* index_spec */) {
    // prefix_lookup expects values in index column order (which is what we have in values)
    // Check if tuple exists in FULL using prefix_lookup with index-ordered values
    auto full_handle = full_idx.prefix_lookup(values);
    bool exists_in_full = full_handle.valid() && full_handle.is_leaf();

    if (!exists_in_full) {
      // Tuple doesn't exist in FULL, add to DELTA
      // Reorder values from index order to schema order for insertion
      using IndexColSeq = typename IndexSpecT::column_indexes_type;
      auto schema_ordered_values = inverse_reorder_tuple_ct<Schema::Arity, IndexColSeq>(values);
      [&schema_ordered_values, &delta_rel]<std::size_t... Is>(std::index_sequence<Is...>) {
        if constexpr (has_provenance_v<typename Schema::semiring_type>) {
          delta_rel.push_row({schema_ordered_values[Is]...}, Schema::semiring_type::one());
        } else {
          delta_rel.push_row_no_prov({schema_ordered_values[Is]...});
        }
      }(std::make_index_sequence<Schema::Arity>{});
    }
  }
};

// Recursive case: iterate over current column, then recurse
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateComputeDelta {
  template <typename FullIndexType, typename DeltaRelType, typename ValuesArray,
            mir::CIndexSpec IndexSpecT>
  static void iterate(const NodeHandleType& node, FullIndexType& full_idx, DeltaRelType& delta_rel,
                      ValuesArray& values, const IndexSpecT& /* index_spec */) {
    constexpr std::size_t current_idx = Schema::Arity - Arity;
    for (auto val : DeltaRelType::column_intersect(node)) {
      values[current_idx] = val;
      auto next_node = node.prefix(val);
      IterateComputeDelta<Arity - 1, decltype(next_node), Schema>::iterate(
          next_node, full_idx, delta_rel, values, IndexSpecT{});
    }
  }
};

// Base case: arity 0, push the row (reorder values from index order to schema order)
template <typename NodeHandleType, CRelationSchema Schema>
struct IterateInsertFull<0, NodeHandleType, Schema> {
  template <typename RelType, typename ValuesArray, mir::CIndexSpec IndexSpecT>
  static void iterate(const NodeHandleType& /* node */, RelType& relation_full, ValuesArray& values,
                      const IndexSpecT& /* index_spec */) {
    // Reorder values from index order to schema order using compile-time unrolling
    using IndexColSeq = typename IndexSpecT::column_indexes_type;
    // Check if reordering is needed (if index order matches schema order [0,1,2,...], no reordering
    // needed) For now, always reorder to be safe
    auto schema_ordered_values = inverse_reorder_tuple_ct<Schema::Arity, IndexColSeq>(values);
    // Convert array to tuple and push row
    [&schema_ordered_values, &relation_full]<std::size_t... Is>(std::index_sequence<Is...>) {
      if constexpr (has_provenance_v<typename Schema::semiring_type>) {
        relation_full.push_row({schema_ordered_values[Is]...}, Schema::semiring_type::one());
      } else {
        relation_full.push_row_no_prov({schema_ordered_values[Is]...});
      }
    }(std::make_index_sequence<Schema::Arity>{});
  }
};

// Recursive case: iterate over current column, then recurse
template <std::size_t Arity, typename NodeHandleType, CRelationSchema Schema>
struct IterateInsertFull {
  template <typename RelType, typename ValuesArray, mir::CIndexSpec IndexSpecT>
  static void iterate(const NodeHandleType& node, RelType& relation_full, ValuesArray& values,
                      const IndexSpecT& /* index_spec */) {
    constexpr std::size_t current_idx = Schema::Arity - Arity;
    // current_idx is the position in the index order (0, 1, 2, ... for index columns)
    // We're iterating in index order, so values[current_idx] gets the value for index column
    // current_idx
    for (auto val : RelType::column_intersect(node)) {
      values[current_idx] = val;
      auto next_node = node.prefix(val);
      IterateInsertFull<Arity - 1, decltype(next_node), Schema>::iterate(next_node, relation_full,
                                                                         values, IndexSpecT{});
    }
  }
};
}  // namespace detail

/**
 * @brief Load full relation into delta for fixpoint initialization.
 *
 * @details This function initializes the delta relation by cloning the
 *          full relation. This is used at the start of fixpoint computation
 *          to initialize delta with all existing facts, so that the first
 *          iteration processes all facts.
 *
 * @tparam Schema The relation schema to initialize
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 */
template <CRelationSchema Schema, AST::CSemiNaiveDatabase DB>
void load_full_to_delta(DB& db) {
  auto& relation_full = get_relation_by_schema<Schema, FULL_VER>(db);
  auto& relation_delta = get_relation_by_schema<Schema, DELTA_VER>(db);
  // db.verify_distinct_addresses();
  relation_full.clone_into(relation_delta);
}

/**
 * @brief Functor for executing rules in a fixpoint computation.
 *
 * @details This struct executes recursive rules until a fixpoint is reached.
 *          It uses semi-naive evaluation to avoid redundant computation by
 *          only processing newly derived facts in each iteration.
 *
 *          The execution process:
 *          1. Extracts rules and incremental relations from the fixpoint
 *          2. Iteratively executes all rules
 *          3. After each iteration, computes delta (merges newt into full)
 *          4. Stops when no new facts are derived (delta == 0)
 *
 *          Exposes the compiled Plan type as a member type for debugging and clarity.
 *
 * @tparam FP A CFixpoint type containing the recursive rules
 */
template <CFixpoint FP>
struct ExecuteRulesForFixpoint {
  // Compile Fixpoint to monadic FixpointPlan (flat instruction list)
  using Plan = typename mir::CompileFixpointToPlan<FP>::type;

  // Extract incremental relations from original Fixpoint (needed for delta computation)
  using IncrementalRelations = typename GetIncrementalRelationsFromFixpoint<FP>::type;

  // Extract all index specs from Fixpoint rules (needed for canonical index computation)
  using RulesTuple = typename ExtractRulesFromFixpoint<FP>::type;
  using AllIndexSpecs = typename detail::CollectAllIndexSpecs<RulesTuple>::type;

  // Build canonical indexes for FULL_VER for all incremental relations
  // This is needed because DestinationRelation uses FULL_VER canonical index for deduplication
  using CanonicalIndexSpecs =
      typename detail::CollectCanonicalIndexSpecs<IncrementalRelations, AllIndexSpecs>::type;

  /**
   * @brief Execute the fixpoint computation.
   *
   * @tparam DB The semi-naive database type
   * @param db The runtime database
   * @param max_iterations Maximum number of iterations (default: unlimited)
   * @return Number of iterations executed
   */
  template <AST::CSemiNaiveDatabase DB>
  std::size_t operator()(DB& db, std::size_t max_iterations = SIZE_MAX) const {
    // std::cout << "Plan: " << get_type_name<Plan>() << std::endl;
    std::cout << get_type_name<CanonicalIndexSpecs>() << std::endl;

    mp::mp_for_each<CanonicalIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
      using Schema = typename IndexSpecT::schema_type;
      using ColSeq = typename IndexSpecT::column_indexes_type;
      auto& rel_full = get_relation_by_schema<Schema, FULL_VER>(db);
      auto canonical_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        return IndexSpec{{Cols...}};
      }(ColSeq{});
      // be careful here
      rel_full.ensure_index(canonical_spec);
    });

    std::size_t iteration = 0;
    while (iteration < max_iterations) {
      LOG_INFO << "Iteration " << iteration;

      // Check DELTA_VER size at the START of iteration (except iteration 0)
      // After delta computation in previous iteration, NEW_VER was cleared
      // So if DELTA_VER is empty at the start of this iteration, no new facts were derived
      // In future we want to make this check happens more like a IR instruction check instead of a
      // runtime check
      if (iteration > 0) {
        std::size_t total_new_facts = 0;
        mp::mp_for_each<IncrementalRelations>([&]<typename Schema>(Schema) {
          auto& newt = get_relation_by_schema<Schema, DELTA_VER>(db);
          total_new_facts += newt.size();
        });

        if (total_new_facts == 0) {
          LOG_INFO << "No new facts in DELTA_VER at start of iteration " << iteration
                   << ", fixpoint reached after " << iteration << " iterations";
          break;
        }

        LOG_INFO << "Found " << total_new_facts << " new facts in DELTA_VER at start of iteration "
                 << iteration;
      }

      // Execute the flat instruction list sequentially using interpreter
      // No MP needed - interpreter just processes instructions in order
      // When ExecutePipeline is encountered, it executes the pipeline (nested for-loops)
      // and then continues to the next instruction
      // ComputeDelta operations are now part of the MIR instructions, so they execute here
      detail::MIRExecutor<typename Plan::instructions_type>::execute(db, iteration);

      iteration++;
    }
    if (iteration >= max_iterations) {
      LOG_WARNING << "Reached maximum iterations (" << max_iterations << "), terminating early";
    }

    // For debugging: clear all indexes after query execution
    // detail::clear_all_indexes_for_debug(db);

    return iteration;
  }
};

/**
 * @brief Functor for executing non-iterative rule sets.
 *
 * @details This struct executes a set of non-recursive rules exactly once.
 *          Unlike fixpoint rules, these rules are evaluated only once and
 *          do not iterate until convergence.
 *
 *          Exposes compile-time types for debugging and clarity:
 *          - RuleList: The list of rules to execute
 *          - IncrementalRelations: Relations that appear in rule heads
 *          - AllIndexSpecs: All index specifications from the rules
 *          - CanonicalIndexSpecs: Canonical index specs for incremental relations
 *          - MIRPlans: Tuple of compiled MIR plans (one per rule)
 *          - FilteredIndexSpecs: Index specs for incremental relations only
 *
 * @tparam Rules A CNonIterativeRuleSets type containing the rules
 */
template <CNonIterativeRuleSets Rules>
struct ExecuteRulesForNonIterativeRuleSets {
  // Extract rule list
  using RuleList = typename GetRulesFromNonIterativeRuleSets<Rules>::type;

  // Collect all index specs and compute canonical indices (similar to mir::CompileFixpointToPlan)
  using IncrementalRelations = typename GetIncrementalRelationsFromRules<RuleList>::type;
  using AllIndexSpecs = typename detail::CollectAllIndexSpecs<RuleList>::type;
  using CanonicalIndexSpecs =
      typename detail::CollectCanonicalIndexSpecs<IncrementalRelations, AllIndexSpecs>::type;

  // Compile MIR plans for each rule (with canonical indices replaced)
  template <typename Rule>
  using CompileRuleMIR =
      typename mir::ReplaceDestinationRelationsInTuple<typename mir::CompileRuleToMIR<Rule>::type,
                                                       CanonicalIndexSpecs>::type;
  using MIRPlans = mp::mp_transform<CompileRuleMIR, RuleList>;

  // Filtered index specs for rebuilding (only incremental relations)
  using FilteredIndexSpecs =
      typename detail::FilterIndexSpecsByRelations<AllIndexSpecs, IncrementalRelations>::type;

  /**
   * @brief Execute the non-iterative rule sets.
   *
   * @tparam DB The semi-naive database type
   * @param db The runtime database
   */
  template <AST::CSemiNaiveDatabase DB>
  void operator()(DB& db) const {
    // Build canonical indexes for FULL_VER for all incremental relations BEFORE executing rules
    // This is needed because DestinationRelation uses FULL_VER canonical index for deduplication
    // Same logic as fixpoint execution to avoid collision
    mp::mp_for_each<CanonicalIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
      using Schema = typename IndexSpecT::schema_type;
      using ColSeq = typename IndexSpecT::column_indexes_type;
      auto& rel_full = get_relation_by_schema<Schema, FULL_VER>(db);
      auto canonical_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        return IndexSpec{{Cols...}};
      }(ColSeq{});
      rel_full.ensure_index(canonical_spec);
    });

    // Execute each rule with canonical indices replaced in MIR
    mp::mp_for_each<RuleList>([&]<typename Rule>(Rule) {
      // Use mir::CompileRuleToPipeline which extracts source specs from AST (ComputeNaiveIndices)
      // instead of extracting from MIR - this is faster and more accurate
      using Pipeline = typename mir::CompileRuleToPipeline<Rule>::type;

      // Execute using the new tuple-based pipeline execution engine
      execute_pipeline<Pipeline>(db, 0, true);  // non_iterative = true
    });

    // Rebuild indexes only for IncrementalRelations (relations in rule heads) on FULL_VER
    mp::mp_for_each<FilteredIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
      using Schema = typename IndexSpecT::schema_type;
      using ColSeq = typename IndexSpecT::column_indexes_type;
      auto& rel = get_relation_by_schema<Schema, FULL_VER>(db);
      auto runtime_spec = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        return IndexSpec{{Cols...}};
      }(ColSeq{});
      rel.ensure_index(runtime_spec);
    });

    // For debugging: clear all indexes after query execution
    detail::clear_all_indexes_for_debug(db);
  }
};

/**
 * @brief Functor for executing queries (non-iterative rule sets or fixpoint).
 *
 * @details This is the main entry point for executing queries. It dispatches
 *          to the appropriate execution functor based on the query type:
 *          - CFixpoint: uses ExecuteRulesForFixpoint functor
 *          - CNonIterativeRuleSets: uses ExecuteRulesForNonIterativeRuleSets functor
 *
 *          Exposes compile-time types for debugging and clarity by providing
 *          access to the underlying executor types.
 *
 * @tparam Q A CQuery type (either CFixpoint or CNonIterativeRuleSets)
 */
template <CQuery Q>
struct ExecuteQuery {
  /**
   * @brief Execute the query.
   *
   * @tparam DB The semi-naive database type
   * @param db The runtime database
   * @param max_iterations Maximum number of iterations for fixpoint queries (default: unlimited)
   * @return Number of iterations executed
   */
  template <AST::CSemiNaiveDatabase DB>
  std::size_t operator()(DB& db, std::size_t max_iterations = SIZE_MAX) const {
    if constexpr (CFixpoint<Q>) {
      return ExecuteRulesForFixpoint<Q>{}(db, max_iterations);
    } else if constexpr (CNonIterativeRuleSets<Q>) {
      ExecuteRulesForNonIterativeRuleSets<Q>{}(db);
      return 1;
    } else {
      static_assert(false, "Unknown query type");
    }
  }
};

// Specialization for CFixpoint to expose plan types
template <CFixpoint Q>
struct ExecuteQuery<Q> {
  using Executor = ExecuteRulesForFixpoint<Q>;
  using Plan = typename Executor::Plan;
  using IncrementalRelations = typename Executor::IncrementalRelations;
  using AllIndexSpecs = typename Executor::AllIndexSpecs;
  using CanonicalIndexSpecs = typename Executor::CanonicalIndexSpecs;

  template <AST::CSemiNaiveDatabase DB>
  std::size_t operator()(DB& db, std::size_t max_iterations = SIZE_MAX) const {
    return Executor{}(db, max_iterations);
  }
};

// Specialization for CNonIterativeRuleSets to expose plan types
template <CNonIterativeRuleSets Q>
struct ExecuteQuery<Q> {
  using Executor = ExecuteRulesForNonIterativeRuleSets<Q>;
  using RuleList = typename Executor::RuleList;
  using IncrementalRelations = typename Executor::IncrementalRelations;
  using AllIndexSpecs = typename Executor::AllIndexSpecs;
  using CanonicalIndexSpecs = typename Executor::CanonicalIndexSpecs;
  using MIRPlans = typename Executor::MIRPlans;
  using FilteredIndexSpecs = typename Executor::FilteredIndexSpecs;

  template <AST::CSemiNaiveDatabase DB>
  std::size_t operator()(DB& db, std::size_t max_iterations = SIZE_MAX) const {
    Executor{}(db);
    return 1;
  }
};

/**
 * @brief Convenience function wrapper for ExecuteQuery functor (API compatibility).
 *
 * @details This function provides backward compatibility for code that uses
 *          the function-style API. It simply instantiates the functor and calls it.
 *
 * @tparam Q A CQuery type (either CFixpoint or CNonIterativeRuleSets)
 * @tparam DB The semi-naive database type
 * @param db The runtime database
 * @param max_iterations Maximum number of iterations for fixpoint queries (default: unlimited)
 * @return Number of iterations executed
 */
template <CQuery Q, AST::CSemiNaiveDatabase DB>
std::size_t execute_query(DB& db, std::size_t max_iterations = SIZE_MAX) {
  return ExecuteQuery<Q>{}(db, max_iterations);
}

}  // namespace SRDatalog
