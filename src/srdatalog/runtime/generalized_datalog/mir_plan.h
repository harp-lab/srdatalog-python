/**
 * @file mir_plan.h
 * @brief Fixpoint execution plan definitions for MIR.
 *
 * @details This file defines the FixpointPlan structure and the instruction set
 * for executing a fixpoint computation, including pipeline execution, index
 * building, and relation maintenance operations.
 */

#pragma once

#include "mir_def.h"

namespace SRDatalog {
using namespace AST;
namespace mir {

/**
 * @brief Abstract Data Type (ADT) for executing a Pipeline.
 *
 * @details Represents an operation that executes a Pipeline (which contains
 * nested for-loops for joins). When this operation completes, the nested
 * for-loops have finished and execution continues to the next operation.
 *
 * @tparam Pipeline The Pipeline to execute (must satisfy CPipeline)
 */
template <typename Pipeline>
struct ExecutePipeline {
  static_assert(is_pipeline_v<Pipeline>, "Pipeline must satisfy CPipeline");
  using pipeline_type = Pipeline;
};
// Type predicate for execute pipeline (Scheme/Racket style:
// (execute-pipeline? x))
template <typename T>
struct is_execute_pipeline : std::false_type {};
template <typename Pipeline>
  requires is_pipeline_v<Pipeline>
struct is_execute_pipeline<ExecutePipeline<Pipeline>> : std::true_type {};
template <typename T>
inline constexpr bool is_execute_pipeline_v = is_execute_pipeline<T>::value;
template <typename T>
concept CExecutePipeline = is_execute_pipeline<T>::value;

/**
 * @brief Abstract Data Type (ADT) for executing a Pipeline using JIT kernels.
 *
 * @details Similar to ExecutePipeline, but uses a JIT-compiled kernel functor
 * instead of the TMP-based pipeline executor. The JitKernel is a struct with
 * an operator() that implements the join logic, generated at compile-time
 * by the Nim JIT codegen.
 *
 * @tparam JitKernel The JIT kernel functor struct (e.g., Kernel_RuleName)
 * @tparam Pipeline The Pipeline for metadata (source/dest specs, schemas)
 */
template <typename JitKernel, typename Pipeline>
struct JitExecutePipeline {
  static_assert(is_pipeline_v<Pipeline>, "Pipeline must satisfy CPipeline");
  using jit_kernel_type = JitKernel;
  using pipeline_type = Pipeline;
};

// Type predicate for JIT execute pipeline
template <typename T>
struct is_jit_execute_pipeline : std::false_type {};
template <typename JitKernel, typename Pipeline>
  requires is_pipeline_v<Pipeline>
struct is_jit_execute_pipeline<JitExecutePipeline<JitKernel, Pipeline>> : std::true_type {};
template <typename T>
inline constexpr bool is_jit_execute_pipeline_v = is_jit_execute_pipeline<T>::value;
template <typename T>
concept CJitExecutePipeline = is_jit_execute_pipeline<T>::value;

/**
 * @brief Abstract Data Type (ADT) for executing a materialized binary join.
 *
 * @details Similar to JitExecutePipeline, but uses a host-side Thrust executor
 * that runs entirely on the host with Thrust primitives. The JitRunner struct
 * contains a static execute(db, iteration) method that performs:
 * 1. Probe phase: count matches, compute offsets, materialize row ID pairs
 * 2. Gather phase: extract columns from source views using row IDs
 * 3. Insert phase: copy gathered data to destination relation
 *
 * @tparam JitRunner The JIT runner struct with static execute(DB&, uint32_t)
 */
template <typename JitRunner>
struct MaterializedExecutePipeline {
  using jit_runner_type = JitRunner;
};

// Type predicate for Materialized execute pipeline
template <typename T>
struct is_materialized_execute_pipeline : std::false_type {};
template <typename JitRunner>
struct is_materialized_execute_pipeline<MaterializedExecutePipeline<JitRunner>> : std::true_type {};
template <typename T>
inline constexpr bool is_materialized_execute_pipeline_v =
    is_materialized_execute_pipeline<T>::value;
template <typename T>
concept CMaterializedExecutePipeline = is_materialized_execute_pipeline<T>::value;

/**
 * @brief Helper to filter IndexSpecList to only include schemas in
 * IncrementalRelations.
 *
 * @details Filters an IndexSpecList to only keep IndexSpecs whose schema_type
 *          appears in the IncrementalRelations tuple. This ensures we only
 *          rebuild indexes for relations that were written to (in rule
 * heads), not relations that were only read from (in rule bodies).
 *
 * @tparam IndexSpecList A std::tuple of IndexSpecT types
 * @tparam IncrementalRelations A std::tuple of relation schemas (from rule
 * heads)
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

/**
 * @brief Abstract Data Type (ADT) for a Fixpoint Plan.
 *
 * @details Represents a compiled fixpoint computation at the MIR level.
 * Contains a flat sequence of operations (BuildIndex, ExecutePipeline,
 * RebuildIndex) that are interpreted sequentially. Pipelines execute in
 * nested for-loops, while index operations execute at the fixpoint level.
 *
 * The plan is monadic (flat list) - the interpreter simply iterates through
 * instructions in order without needing to identify different sections.
 * This design allows for future reordering optimizations at the IR level.
 *
 * @tparam Instructions A std::tuple of fixpoint-level operations (BuildIndex,
 *                      ExecutePipeline, RebuildIndex) in execution order
 */
template <tmp::CTuple Instructions, typename IncrementalRelations = std::tuple<>,
          typename CanonicalIndexSpecs = std::tuple<>, typename AllSourceSpecs = std::tuple<>>
struct FixpointPlan {
  /// @brief Flat sequence of instructions to execute
  using instructions_type = Instructions;
  using incremental_relations_type = IncrementalRelations;
  using canonical_index_specs_type = CanonicalIndexSpecs;
  using all_source_specs_type = AllSourceSpecs;
};

// Type predicate for fixpoint plan (Scheme/Racket style: (fixpoint-plan? x))
template <typename T>
struct is_fixpoint_plan : std::false_type {};
// Match FixpointPlan with all three template parameters
template <tmp::CTuple Instructions, typename IncrementalRelations, typename CanonicalIndexSpecs,
          typename AllSourceSpecs>
struct is_fixpoint_plan<
    mir::FixpointPlan<Instructions, IncrementalRelations, CanonicalIndexSpecs, AllSourceSpecs>>
    : std::true_type {};
// Match FixpointPlan with just Instructions (other two use defaults)
template <tmp::CTuple Instructions>
struct is_fixpoint_plan<mir::FixpointPlan<Instructions>> : std::true_type {};
template <typename T>
inline constexpr bool is_fixpoint_plan_v = is_fixpoint_plan<T>::value;
template <typename T>
concept CFixpointPlan = is_fixpoint_plan<T>::value;

/**
 * @brief Abstract Data Type (ADT) for building an index on a relation.
 *
 * @details Represents an operation that builds an index on a relation at a
 * specific version. This is used to ensure indexes exist before pipeline
 * execution.
 *
 * @tparam IndexSpecT The IndexSpecT type specifying which index to build
 */
template <CIndexSpec IndexSpecT>
struct CreateIndex {
  using index_spec_type = IndexSpecT;
};
// Type predicate for build index (Scheme/Racket style: (build-index? x))
DEFINE_TYPE_PREDICATE(is_build_index, CreateIndex, CBuildIndex, (CIndexSpec IndexSpecT),
                      (IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for rebuilding an index on a relation.
 *
 * @details Represents an operation that rebuilds an index on a relation at a
 * specific version. This is used after pipeline execution to rebuild indexes
 * that may have become dirty.
 *
 * @tparam IndexSpecT The IndexSpecT type specifying which index to rebuild
 */
template <CIndexSpec IndexSpecT>
struct RebuildIndex {
  using index_spec_type = IndexSpecT;
};
// Type predicate for rebuild index (Scheme/Racket style: (rebuild-index? x))
DEFINE_TYPE_PREDICATE(is_rebuild_index, RebuildIndex, CRebuildIndex, (CIndexSpec IndexSpecT),
                      (IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for checking if a relation is empty.
 *
 * @details Represents a conditional check on relation size. Used for early
 * returns in delta computation. If the relation is empty, execution can skip
 * subsequent operations.
 *
 * @tparam Schema The relation schema to check
 * @tparam Ver The version number to check
 */
template <CRelationSchema Schema, std::size_t Ver>
struct CheckSize {
  using schema_type = Schema;
  static constexpr std::size_t Version = Ver;
};
// Type predicate for check size (Scheme/Racket style: (check-size? x))
DEFINE_TYPE_PREDICATE(is_check_size, CheckSize, CCheckSize,
                      (CRelationSchema Schema, std::size_t Ver), (Schema, Ver))

/**
 * @brief Abstract Data Type (ADT) for clearing a relation.
 *
 * @details Represents an operation that clears all facts from a relation at a
 * specific version. Used in delta computation to reset the delta relation.
 *
 * @tparam Schema The relation schema to clear
 * @tparam Ver The version number to clear
 */
template <CRelationSchema Schema, std::size_t Ver>
struct ClearRelation {
  using schema_type = Schema;
  static constexpr std::size_t Version = Ver;
};
// Type predicate for clear relation (Scheme/Racket style: (clear-relation?
// x))
DEFINE_TYPE_PREDICATE(is_clear_relation, ClearRelation, CClearRelation,
                      (CRelationSchema Schema, std::size_t Ver), (Schema, Ver))

/**
 * @brief Abstract Data Type (ADT) for inserting all tuples from one relation
 * version to another.
 *
 * @details Represents an operation that iterates over all tuples in a source
 * relation (using a precomputed index) and inserts them into a destination
 * relation. Used in delta computation to merge newt into full.
 *
 * @tparam Schema The relation schema
 * @tparam FromVer The source version to read from
 * @tparam ToVer The destination version to write to
 * @tparam IndexSpecT The precomputed IndexSpecT to use for iteration
 */
template <CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer, CIndexSpec IndexSpecT>
struct InsertFromRelation {
  using schema_type = Schema;
  static constexpr std::size_t FromVersion = FromVer;
  static constexpr std::size_t ToVersion = ToVer;
  using index_spec_type = IndexSpecT;
};
// Type predicate for insert from relation (Scheme/Racket style:
// (insert-from-relation? x))
DEFINE_TYPE_PREDICATE(is_insert_from_relation, InsertFromRelation, CInsertFromRelation,
                      (CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer,
                       CIndexSpec IndexSpecT),
                      (Schema, FromVer, ToVer, IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for swapping two relation versions.
 *
 * @details Represents an operation that swaps the contents of DELTA_VER and
 * NEW_VER relations. Used in delta computation after merging newt into full.
 *
 * @tparam Schema The relation schema to swap
 */
template <CRelationSchema Schema>
struct SwapRelations {
  using schema_type = Schema;
};
// Type predicate for swap relations (Scheme/Racket style: (swap-relations?
// x))
DEFINE_TYPE_PREDICATE(is_swap_relations, SwapRelations, CSwapRelations, (CRelationSchema Schema),
                      (Schema))

/**
 * @brief Abstract Data Type (ADT) for computing delta (NEW - FULL -> DELTA).
 *
 * @details Computes the set difference `Delta = NEW - FULL` and populates the
 * DELTA relation. This is the first step of the optimized delta computation.
 * The merge into FULL is handled separately by MergeIndex operations.
 *
 * @tparam Schema The relation schema
 * @tparam FromVer The source version (NEW_VER)
 * @tparam ToVer The target version (FULL_VER)
 * @tparam DeltaVer The delta version (DELTA_VER)
 * @tparam IndexSpecT The IndexSpecT to use for the computation
 */
template <CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
          CIndexSpec IndexSpecT>
struct ComputeDelta {
  using schema_type = Schema;
  static constexpr std::size_t FromVersion = FromVer;
  static constexpr std::size_t ToVersion = ToVer;
  static constexpr std::size_t DeltaVersion = DeltaVer;
  using index_spec_type = IndexSpecT;
};
// Type predicate
DEFINE_TYPE_PREDICATE(is_compute_delta, ComputeDelta, CComputeDelta,
                      (CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer,
                       std::size_t DeltaVer, CIndexSpec IndexSpecT),
                      (Schema, FromVer, ToVer, DeltaVer, IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for merging an index from DELTA to FULL.
 *
 * @details Represents an operation that merges an index from DELTA_VER
 * relation into FULL_VER relation of the same schema. Used for efficient
 * incremental maintenance of secondary indices.
 *
 * @tparam IndexSpecT The IndexSpecT type specifying which index to merge.
 *                    Must be the FULL_VER index spec.
 */
template <CIndexSpec IndexSpecT>
struct MergeIndex {
  using index_spec_type = IndexSpecT;
};
// Type predicate for merge index (Scheme/Racket style: (merge-index? x))
DEFINE_TYPE_PREDICATE(is_merge_index, MergeIndex, CMergeIndex, (CIndexSpec IndexSpecT),
                      (IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for computing delta and merging (GPU
 * Optimization).
 * @deprecated Use ComputeDelta + MergeIndex instead for better modularity.
 *
 * @details Represents an optimized operation that combines
 * `InsertFromRelation` and `SwapRelations`. Instead of merging `NEW` into
 * `FULL` and then swapping `NEW` into `DELTA`, this operation:
 * 1. Computes `Delta = NEW - FULL` (using set difference)
 * 2. Merges `Delta` into `FULL`
 * 3. Populates `DELTA` relation with the computed `Delta`
 * 4. Clears `NEW` relation (effectively)
 *
 * This avoids the inefficiency of swapping `NEW` (which contains duplicates
 * and old data) into `DELTA` for the next iteration.
 *
 * @tparam Schema The relation schema
 * @tparam FromVer The source version (NEW_VER)
 * @tparam ToVer The destination version (FULL_VER)
 * @tparam DeltaVer The delta version (DELTA_VER)
 * @tparam IndexSpecT The IndexSpecT to use
 */
template <CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
          CIndexSpec IndexSpecT>
struct ComputeDeltaAndMerge {
  using schema_type = Schema;
  static constexpr std::size_t FromVersion = FromVer;
  static constexpr std::size_t ToVersion = ToVer;
  static constexpr std::size_t DeltaVersion = DeltaVer;
  using index_spec_type = IndexSpecT;
};
// Type predicate
DEFINE_TYPE_PREDICATE(is_compute_delta_and_merge, ComputeDeltaAndMerge, CComputeDeltaAndMerge,
                      (CRelationSchema Schema, std::size_t FromVer, std::size_t ToVer,
                       std::size_t DeltaVer, CIndexSpec IndexSpecT),
                      (Schema, FromVer, ToVer, DeltaVer, IndexSpecT))

/**
 * @brief Abstract Data Type (ADT) for setting a relation version.
 *
 * @details Represents an operation that sets the version number of a
 * relation. Used after swapping to ensure version metadata is correct.
 *
 * @tparam Schema The relation schema
 * @tparam Ver The version number to set
 */
template <CRelationSchema Schema, std::size_t Ver>
struct SetVersion {
  using schema_type = Schema;
  static constexpr std::size_t Version = Ver;
};
// Type predicate for set version (Scheme/Racket style: (set-version? x))
DEFINE_TYPE_PREDICATE(is_set_version, SetVersion, CSetVersion,
                      (CRelationSchema Schema, std::size_t Ver), (Schema, Ver))

/**
 * @brief Abstract Data Type (ADT) for injecting arbitrary C++ code at fixpoint level.
 *
 * @details Represents an operation that invokes a user-provided functor with the
 * entire database reference. Used for debugging, profiling, and inspecting relation
 * sizes during fixpoint iterations.
 *
 * The functor receives (const Database&) and can access any relation via:
 *   get_relation_by_schema<Schema, Version>(db).canonical_index().root().degree()
 *
 * @tparam Functor A callable type that takes (const auto& db) and returns void
 */
template <typename Functor>
struct InjectCppHook {
  using functor_type = Functor;
};

// Type predicate for inject cpp hook
template <typename T>
struct is_inject_cpp_hook : std::false_type {};
template <typename Functor>
struct is_inject_cpp_hook<InjectCppHook<Functor>> : std::true_type {};
template <typename T>
inline constexpr bool is_inject_cpp_hook_v = is_inject_cpp_hook<T>::value;
template <typename T>
concept CInjectCppHook = is_inject_cpp_hook<T>::value;

/**
 * @brief Abstract Data Type (ADT) for reconstructing intern columns from canonical index.
 *
 * @details MUST be emitted ONLY after moFixpointPlan completes, NEVER inside fixpoint loop.
 * Reconstructs intern column values FROM the canonical index TO the relation's column storage.
 * This ensures that after fixpoint iteration, the relation's linear storage is synchronized
 * with the authoritative index state.
 *
 * @tparam IndexSpecT The canonical index specification to use as source of truth
 */
template <CIndexSpec IndexSpecT>
struct PostStratumReconstructInternCols {
  using index_spec_type = IndexSpecT;
  using schema_type = typename IndexSpecT::schema_type;
};

// Type predicate for post_stratum_reconstruct_intern_cols
template <typename T>
struct is_post_stratum_reconstruct_intern_cols : std::false_type {};
template <CIndexSpec IndexSpecT>
struct is_post_stratum_reconstruct_intern_cols<PostStratumReconstructInternCols<IndexSpecT>>
    : std::true_type {};
template <typename T>
inline constexpr bool is_post_stratum_reconstruct_intern_cols_v =
    is_post_stratum_reconstruct_intern_cols<T>::value;
template <typename T>
concept CPostStratumReconstructInternCols = is_post_stratum_reconstruct_intern_cols<T>::value;

/// @brief Extract relation schemas from MIR ops
template <typename MIROps>
struct ExtractRelationSchemas;

template <>
struct ExtractRelationSchemas<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename Head, typename... Tail>
struct ExtractRelationSchemas<std::tuple<Head, Tail...>> {
 private:
  template <typename Op>
  struct Extract {
    using type = std::tuple<>;
  };

  template <typename IndexSpec, typename PrefixVars>
  struct Extract<ColumnSource<IndexSpec, PrefixVars>> {
    using type = std::tuple<typename IndexSpec::schema_type>;
  };

  template <typename Var, typename Sources, typename Strategy, std::size_t HS, typename VHM>
  struct Extract<ColumnJoin<Var, Sources, Strategy, HS, VHM>> {
    using type = typename ExtractRelationSchemas<Sources>::type;
  };

  template <typename VarsTuple, typename SourcesTuple, std::size_t HS, typename VHM>
  struct Extract<CartesianJoin<VarsTuple, SourcesTuple, HS, VHM>> {
    using type = typename ExtractRelationSchemas<SourcesTuple>::type;
  };

  template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HS,
            typename VHM>
  struct Extract<Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>> {
    using type = std::tuple<typename IndexSpec::schema_type>;
  };

  template <typename VarToBind, typename IndexSpec, typename PrefixVars, typename AggFunc,
            std::size_t HS, typename VHM>
  struct Extract<Aggregate<VarToBind, IndexSpec, PrefixVars, AggFunc, HS, VHM>> {
    using type = std::tuple<typename IndexSpec::schema_type>;
  };

  template <typename IndexSpec, typename PrefixVars, std::size_t HS, typename VHM>
  struct Extract<Negation<IndexSpec, PrefixVars, HS, VHM>> {
    using type = std::tuple<typename IndexSpec::schema_type>;
  };

  template <typename Pipeline>
  struct Extract<ExecutePipeline<Pipeline>> {
    using type = typename ExtractRelationSchemas<typename Pipeline::mir_ops_type>::type;
  };

  template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
            typename IndexSpecT>
  struct Extract<ComputeDelta<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>> {
    using type = std::tuple<Schema>;
  };

  template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
            typename IndexSpecT>
  struct Extract<ComputeDeltaAndMerge<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>> {
    using type = std::tuple<Schema>;
  };

  using HeadSchemas = typename Extract<Head>::type;
  using TailSchemas = typename ExtractRelationSchemas<std::tuple<Tail...>>::type;

 public:
  using type = Mp::mp_append<HeadSchemas, TailSchemas>;
};

/// @brief Extract destination relation schemas from MIR ops
template <typename MIROps>
struct ExtractDestinationSchemas;

template <>
struct ExtractDestinationSchemas<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename Head, typename... Tail>
struct ExtractDestinationSchemas<std::tuple<Head, Tail...>> {
 private:
  template <typename Op>
  struct Extract {
    using type = std::tuple<>;
  };

  template <typename Schema, typename Terms, std::size_t Ver, typename IndexSpec, typename VHM>
  struct Extract<DestinationRelation<Schema, Terms, Ver, IndexSpec, VHM>> {
    using type = std::tuple<Schema>;
  };

  template <typename Pipeline>
  struct Extract<ExecutePipeline<Pipeline>> {
    using type = typename ExtractDestinationSchemas<typename Pipeline::mir_ops_type>::type;
  };

  template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
            typename IndexSpecT>
  struct Extract<ComputeDelta<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>> {
    using type = std::tuple<Schema>;
  };

  template <typename Schema, std::size_t FromVer, std::size_t ToVer, std::size_t DeltaVer,
            typename IndexSpecT>
  struct Extract<ComputeDeltaAndMerge<Schema, FromVer, ToVer, DeltaVer, IndexSpecT>> {
    using type = std::tuple<Schema>;
  };

  template <typename Schema, std::size_t FromVer, std::size_t ToVer, typename IndexSpecT>
  struct Extract<InsertFromRelation<Schema, FromVer, ToVer, IndexSpecT>> {
    using type = std::tuple<Schema>;
  };

  using HeadSchemas = typename Extract<Head>::type;
  using TailSchemas = typename ExtractDestinationSchemas<std::tuple<Tail...>>::type;

 public:
  using type = Mp::mp_unique<Mp::mp_append<HeadSchemas, TailSchemas>>;
};

}  // namespace mir
}  // namespace SRDatalog