#pragma once

#include "gpu/device_2level_index.h"
#include "gpu/device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/gpu_api.h"
#include "gpu/index_traits.h"  // For TVJoinMarker, is_tvjoin_marker_v
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include "gpu/runtime/runtime.h"
#include "query.h"
#include "relation_col.h"
#include "runtime/query.h"
#include "type_name.h"
#include <boost/mp11.hpp>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace SRDatalog::GPU {
namespace mp = boost::mp11;

/**
 * @brief defines the Host Relation type for a given Schema.
 */
template <typename Schema>
using HostRelationType = Relation<typename Schema::semiring_type, typename Schema::attr_ts_type,
                                  HashTrieIndex, SRDatalog::HostRelationPolicy>;

// Extract the index template from a RelationSchema
template <typename T>
struct ExtractIndexTemplate;

template <AST::CTString N, Semiring S, SRDatalog::ColumnElementTuple A,
          template <Semiring, SRDatalog::ColumnElementTuple, typename...> class Idx, typename V>
struct ExtractIndexTemplate<AST::RelationSchema<N, S, A, Idx, V>> {
  template <Semiring S2, SRDatalog::ColumnElementTuple A2, typename... Args>
  using type = Idx<S2, A2, Args...>;

  static constexpr bool is_hash_trie = std::is_same_v<Idx<S, A>, HashTrieIndex<S, A>>;
  static constexpr bool is_tvjoin = is_tvjoin_marker_v<Idx<S, A>>;
};

// Forward declare DeviceTVJoinIndex
template <typename SR, typename T, std::size_t Arity, std::size_t... IndexCols>
class DeviceTVJoinIndex;

// Helper metafunction to get the actual device index
template <typename Schema, bool IsHashTrie, bool IsTVJoin>
struct GetDeviceIndexType {
  // Default: passthrough
  template <typename S, typename A, typename... Args>
  using type = typename ExtractIndexTemplate<Schema>::template type<S, A, Args...>;
};

template <typename Schema>
struct GetDeviceIndexType<Schema, true, false> {
  // HashTrie maps to DeviceSortedArrayIndex
  template <typename S, typename A, typename... Args>
  using type = DeviceSortedArrayIndex<S, A, Args...>;
};

template <typename Schema>
struct GetDeviceIndexType<Schema, false, true> {
  // TVJoin: map to DeviceTVJoinIndex with signature adaptation
  // Use intern_value_type from Schema (not raw AttrType) to match Relation's storage
  template <typename SR, typename AttrTuple, typename... Args>
  using type =
      DeviceTVJoinIndex<SR, typename Schema::intern_value_type, std::tuple_size_v<AttrTuple>, 0,
                        1  // Default binary relation index order
                        >;
};

// Wrapper that computes bools and inherits from the right specialization
template <typename Schema>
using SelectDeviceIndex = GetDeviceIndexType<Schema, ExtractIndexTemplate<Schema>::is_hash_trie,
                                             ExtractIndexTemplate<Schema>::is_tvjoin>;

/**
 * @brief Defines the Device Relation type for a given Schema.
 */
template <typename Schema>
using DeviceRelationType =
    Relation<typename Schema::semiring_type, typename Schema::attr_ts_type,
             SelectDeviceIndex<Schema>::template type, SRDatalog::DeviceRelationPolicy,
             typename Schema::intern_value_type>;

/**
 * @brief Helper to copy a Host Database to a Device Database (H2D).
 *
 * @details This is a database-level operation that copies all relations (FULL, DELTA, NEWT)
 * from a host database to a device database. This is query-agnostic and can be used
 * independently of any query executor. The function automatically creates a DeviceDB
 * with the same schema as the HostDB.
 *
 * @tparam HostDB The Host Database type (must have schema_type)
 * @param host_db Reference to the source Host Database (CPU memory)
 * @return A fully initialized DeviceDB instance residing on GPU memory
 *
 * @example
 * ```cpp
 * // Copy host database to device (query-agnostic)
 * auto device_db = copy_host_to_device(host_db);
 * // Now device_db can be used for GPU operations
 * ```
 */
template <typename HostDB>
auto copy_host_to_device(HostDB& host_db) {
  using SchemaTuple = typename HostDB::schema_type;
  using DeviceDBType = AST::SemiNaiveDatabase<SchemaTuple, DeviceRelationType>;

  DeviceDBType device_db;
  AST::DatabaseInitializer<HostDB, DeviceDBType>::execute(host_db, device_db);
  // Return value will be moved (RVO or explicit move constructor)
  // SemiNaiveDatabase has explicit move constructors, ensuring efficient transfer
  return device_db;
}

/**
 * @brief Helper metafunction to build a tuple of DeviceRelationTypes.
 *
 * @details Transforms a tuple of Schemas into a tuple of corresponding `DeviceRelationType`s.
 * Used for defining the structure of relation bundles on the device (internal usage).
 *
 * @tparam SchemaTuple A `std::tuple` of Schema types
 */
template <typename SchemaTuple>
struct BuildDeviceRelationTuple;

template <typename... Schemas>
struct BuildDeviceRelationTuple<std::tuple<Schemas...>> {
  using type = std::tuple<DeviceRelationType<Schemas>...>;
};

/**
 * @brief Executor for running queries on GPU (fixpoint or non-iterative).
 *
 * @details Handles the lifecycle of a GPU query execution:
 * 1. `prepare`: Copies data from Host DB to Device DB.
 * 2. `execute_kernel`: Runs the query on GPU (fixpoint loop or single execution).
 * 3. `reflect`: Copies results back from Device DB to Host DB.
 *
 * @tparam Q The query type (CFixpoint or CNonIterativeRuleSets)
 */
template <AST::CQuery Q>
struct GPUQueryExecutor;

/**
 * @brief Decoupled Executor for pre-compiled MIR plans.
 *
 * @details Contains the runtime logic for executing a query plan (fixpoint loop, index management,
 * state sync). Does NOT perform any AST analysis or compilation. Expects all necessary metadata
 * (Instructions, IncrementalRelations, CanonicalIndexSpecs) to be provided as template arguments.
 *
 * @tparam Instructions The tuple of MIR instructions to execute.
 * @tparam IncrementalRelations Tuple of Schemas that are modified/incremental (for state sync).
 * @tparam CanonicalIndexSpecs Tuple of IndexSpecs used to reconstruct relations from indexes.
 */
template <typename Instructions, typename IncrementalRelations, typename CanonicalIndexSpecs>
struct GPUPlanExecutor {
  /**
   * @brief Executes the plan on the given Device Database.
   *
   * @tparam DeviceDBType The Device Database type.
   * @param device_db The Device Database instance.
   * @param max_iterations Maximum number of fixpoint iterations.
   */
  template <typename DeviceDBType>
  static void execute(DeviceDBType& device_db, std::size_t max_iterations = 100) {
    // Helper to extract IndexSpec from a single instruction
    auto build_indexes_from_instruction = [&]<typename Inst>(Inst) {
      if constexpr (mir::is_rebuild_index_v<Inst>) {
        using IndexSpecT = typename Inst::index_spec_type;
        using Schema = typename IndexSpecT::schema_type;
        constexpr std::size_t ver = IndexSpecT::kVersion;
        using ColSeq = typename IndexSpecT::column_indexes_type;
        auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
          return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
        }(ColSeq{});
        auto& rel = get_relation_by_schema<Schema, ver>(device_db);
        rel.ensure_index(runtime_spec);
      } else if constexpr (mir::is_insert_from_relation_v<Inst>) {
        using IndexSpecT = typename Inst::index_spec_type;
        using Schema = typename Inst::schema_type;
        constexpr std::size_t from_ver = Inst::FromVersion;
        constexpr std::size_t to_ver = Inst::ToVersion;
        using ColSeq = typename IndexSpecT::column_indexes_type;
        auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
          return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
        }(ColSeq{});
        auto& from_rel = get_relation_by_schema<Schema, from_ver>(device_db);
        from_rel.ensure_index(runtime_spec);
        auto& to_rel = get_relation_by_schema<Schema, to_ver>(device_db);
        to_rel.ensure_index(runtime_spec);
      } else if constexpr (mir::is_execute_pipeline_v<Inst>) {
        using Pipeline = typename Inst::pipeline_type;
        using ExtractedIndexSpecs = typename Pipeline::source_specs_type;
        mp::mp_for_each<ExtractedIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
          using Schema = typename IndexSpecT::schema_type;
          constexpr std::size_t ver = IndexSpecT::kVersion;
          using ColSeq = typename IndexSpecT::column_indexes_type;
          auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
            return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
          }(ColSeq{});
          auto& rel = get_relation_by_schema<Schema, ver>(device_db);
          rel.ensure_index(runtime_spec);
          // Also build FULL index for DELTA sources to support iteration 0 redirection
          // If nim has issue check this code here, this is brittle and might give use useless
          // indices
          if constexpr (ver == DELTA_VER) {
            auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(device_db);
            full_rel.ensure_index(runtime_spec);
          }
        });
      } else if constexpr (mir::is_jit_execute_pipeline_v<Inst>) {
        // JitExecutePipeline uses the same pattern as ExecutePipeline for index building
        using Pipeline = typename Inst::pipeline_type;
        using ExtractedIndexSpecs = typename Pipeline::source_specs_type;
        mp::mp_for_each<ExtractedIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
          using Schema = typename IndexSpecT::schema_type;
          constexpr std::size_t ver = IndexSpecT::kVersion;
          using ColSeq = typename IndexSpecT::column_indexes_type;
          auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
            return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
          }(ColSeq{});
          auto& rel = get_relation_by_schema<Schema, ver>(device_db);
          rel.ensure_index(runtime_spec);
          // Also build FULL index for DELTA sources to support iteration 0 redirection
          if constexpr (ver == DELTA_VER) {
            auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(device_db);
            full_rel.ensure_index(runtime_spec);
          }
        });
      }
    };

    // Build all indexes from all instructions
    mp::mp_for_each<Instructions>(build_indexes_from_instruction);

    // Verify if CartesianJoin is present in any pipeline
    mp::mp_for_each<Instructions>([](auto inst) {
      using Inst = decltype(inst);
      if constexpr (mir::is_execute_pipeline_v<Inst>) {
        using Pipeline = typename Inst::pipeline_type;
        using MIROps = typename Pipeline::mir_ops_type;
        if constexpr (mir::gpu_opt::HasCartesianJoin<MIROps>::value) {
        } else {
        }
      }
    });

    using Executor = GPUFixpointExecutor<Instructions, DeviceDBType>;

    for (std::size_t i = 0; i < max_iterations; ++i) {
      std::cout << "[Fixpoint] Iteration " << i << std::endl;
      // Check DELTA_VER size at the START of iteration (except iteration 0)
      if (i > 0) {
        std::size_t total_new_facts = 0;

        mp::mp_for_each<CanonicalIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
          using Schema = typename IndexSpecT::schema_type;
          using ColSeq = typename IndexSpecT::column_indexes_type;
          auto canonical_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
            return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
          }(ColSeq{});

          auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(device_db);

          try {
            // Delta index already built by compute_delta_fn — use get_index() directly.
            // ensure_index() would trigger is_dirty() rebuild from empty intern cols.
            auto& delta_idx = delta_rel.get_index(canonical_spec);
            total_new_facts += delta_idx.root().degree();
          } catch (...) {
          }
        });

        std::cout << "[Host] Iter " << i << " Total Delta Facts: " << total_new_facts << std::endl;

        // DEBUG: For last few iterations or small deltas, print delta degree distribution
        // if (i >= 40) {
        //   std::cout << "[DEBUG_SKEW] Entering skew check for iter " << i << std::endl;
        //   mp::mp_for_each<CanonicalIndexSpecs>([&]<typename IndexSpecT>(IndexSpecT) {
        //     using Schema = typename IndexSpecT::schema_type;
        //     using ColSeq = typename IndexSpecT::column_indexes_type;
        //     auto canonical_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
        //       return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
        //     }(ColSeq{});

        //     auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(device_db);
        //     try {
        //       auto& delta_idx = delta_rel.ensure_index(canonical_spec);
        //       auto delta_size = delta_idx.size();
        //       if (delta_size > 0) {
        //         auto unique_roots = delta_idx.num_unique_root_values();
        //         double avg_deg =
        //             unique_roots > 0 ? static_cast<double>(delta_size) / unique_roots : 0;

        //         // Get max degree by sampling first 1000 root values
        //         auto root_handle = delta_idx.root();
        //         uint32_t max_deg = 0;
        //         for (size_t j = 0; j < std::min(unique_roots, size_t(1000)); ++j) {
        //           auto child = root_handle.child(j);
        //           if (child.degree() > max_deg)
        //             max_deg = child.degree();
        //         }

        //         std::cout << "[DELTA_SKEW] Iter " << i << " "
        //                   << ": size=" << delta_size << ", unique=" << unique_roots
        //                   << ", avg_deg=" << avg_deg << ", max_deg=" << max_deg << std::endl;
        //       }
        //     } catch (...) {
        //     }
        //   });
        // }

        if (total_new_facts == 0) {
          break;
        }
      }

      bool changed = Executor::execute(device_db, i, max_iterations);
      (void)changed;
    }
    GPU_DEVICE_SYNCHRONIZE();

    // Sync state: Reconstruct relation storage from canonical indices
    mp::mp_for_each<IncrementalRelations>([&]<typename Schema>(Schema) {
      auto& rel = get_relation_by_schema<Schema, FULL_VER>(device_db);

      using FoundCanonical =
          typename SRDatalog::mir::FindCanonicalIndexForSchema<Schema, CanonicalIndexSpecs>::type;
      using Spec = mir::IndexSpecT<typename FoundCanonical::schema_type,
                                   typename FoundCanonical::column_indexes_type, FULL_VER>;

      using ColSeq = typename Spec::column_indexes_type;
      auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
        return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
      }(ColSeq{});

      try {
        auto& idx = rel.get_index(runtime_spec);
        // Reconstruct for all semirings (columns always need copying; NoProvenance overload exists)
        idx.reconstruct_to_relation(runtime_spec, rel.unsafe_interned_columns(), rel.provenance());
      } catch (...) {
      }
    });
  }
};

// Specialization for fixpoint queries
template <AST::CFixpoint FP>
struct GPUQueryExecutor<FP> {
  /// @brief Compile the fixpoint query into an execution plan
  using InitialPlan = typename SRDatalog::mir::CompileFixpointToPlan<FP>::type;
  /// @brief Apply GPU-specific MIR optimizations
  using Plan = typename mir::gpu_opt::OptimizeFixpointPlanForGPU<InitialPlan>::type;
  /// @brief The sequence of instructions derived from the optimized plan
  using Instructions = typename Plan::instructions_type;

  // Extract all IndexSpecs from the fixpoint to know what indexes DELTA needs
  using RulesTuple = typename ExtractRulesFromFixpoint<FP>::type;
  using AllIndexSpecs = typename SRDatalog::detail::CollectAllIndexSpecs<RulesTuple>::type;

  // Extract incremental relations and canonical index specs (needed for building indexes at start)
  using IncrementalRelations = typename GetIncrementalRelationsFromFixpoint<FP>::type;
  using CanonicalIndexSpecs =
      typename SRDatalog::detail::CollectCanonicalIndexSpecs<IncrementalRelations,
                                                             AllIndexSpecs>::type;

  template <AST::CDatabase SchemaTuple>
  using DeviceDB = AST::SemiNaiveDatabase<SchemaTuple, DeviceRelationType>;

  template <typename HostDB>
  using DeviceDBFromHost = DeviceDB<typename HostDB::schema_type>;

  template <typename HostDB>
  static DeviceDBFromHost<HostDB> prepare(HostDB& host_db) {
    return copy_host_to_device(host_db);
  }

  template <AST::CDatabase SchemaTuple>
  static void execute_kernel(DeviceDB<SchemaTuple>& device_db, std::size_t max_iterations = 100) {
    // Delegate to GPUPlanExecutor
    using Executor = GPUPlanExecutor<Instructions, IncrementalRelations, CanonicalIndexSpecs>;
    Executor::execute(device_db, max_iterations);
  }

  // Overload for automatic schema deduction
  template <typename DeviceDBType>
  static void execute_kernel(DeviceDBType& device_db, std::size_t max_iterations = 100)
    requires requires { typename DeviceDBType::schema_type; }
  {
    using SchemaTuple = typename DeviceDBType::schema_type;
    execute_kernel<SchemaTuple>(device_db, max_iterations);
  }

  template <typename HostDB>
  static void reflect(HostDB& host_db, DeviceDBFromHost<HostDB>& device_db) {
    using SchemaTuple = typename HostDB::schema_type;
    using DeviceDBType = DeviceDB<SchemaTuple>;
    ResultReflector<HostDB, typename DeviceDBType::data_type, FULL_VER>::execute(host_db,
                                                                                 device_db.full);
  }

  template <typename HostDB>
  static void execute(HostDB& host_db, std::size_t max_iterations = 100) {
    auto device_db = prepare(host_db);
    using SchemaTuple = typename HostDB::schema_type;
    execute_kernel<SchemaTuple>(device_db, max_iterations);
    reflect(host_db, device_db);
  }

  // ResultReflector definition
  template <typename DB, typename DeviceTuple, std::size_t Ver>
  struct ResultReflector {
    static void execute(DB& host_db, DeviceTuple& device_tuple) {
      using SchemaTuple = typename DB::schema_type;
      constexpr std::size_t N = std::tuple_size_v<SchemaTuple>;

      mp::mp_for_each<mp::mp_iota_c<N>>([&](auto I) {
        using Schema = std::tuple_element_t<I, SchemaTuple>;
        auto& device_rel = std::get<I>(device_tuple);
        auto host_rel_copy = HostRelationType<Schema>::to_host(device_rel);
        host_rel_copy.reconstruct_columns_from_interned();

        auto& host_rel = [&]() -> auto& {
          if constexpr (Ver == FULL_VER) {
            return std::get<I>(host_db.full);
          } else if constexpr (Ver == DELTA_VER) {
            return std::get<I>(host_db.delta);
          } else if constexpr (Ver == NEW_VER) {
            return std::get<I>(host_db.newt);
          }
        }();
        host_rel.clear();
        host_rel.concat(host_rel_copy);
      });
    }
  };
};

/**
 * @brief Helper to convert NonIterativeRuleSets to Fixpoint type for executor access.
 *
 * @details Converts a NonIterativeRuleSets query to a Fixpoint type so it can use
 * the fixpoint executor. This allows non-iterative rules to reuse the fixpoint
 * executor's logic (which handles iteration 0 correctly by aliasing Delta to Full
 * and merging results).
 *
 * @tparam Rules The NonIterativeRuleSets type
 *
 * @example
 * ```cpp
 * using NonIterativeQuery = NonIterativeRuleSets<Rule1, Rule2>;
 * using FixpointType = NonIterativeToFixpoint<NonIterativeQuery>::type;
 * using Executor = NonIterativeToFixpoint<NonIterativeQuery>::Executor;
 * // Now you can access Executor::Plan, Executor::execute_kernel, etc.
 * ```
 */
template <AST::CNonIterativeRuleSets Rules>
struct NonIterativeToFixpoint {
  using RulesTuple = typename AST::GetRulesFromNonIterativeRuleSets<Rules>::type;

  /// @brief The Fixpoint type constructed from the non-iterative rules
  using type = mp::mp_apply<AST::Fixpoint, RulesTuple>;

  /// @brief Alias to access the executor type directly
  using Executor = GPUQueryExecutor<type>;
};

/**
 * @brief Main entry point to execute a query on GPU.
 *
 * @details Convenience wrapper around `GPUQueryExecutor::execute`.
 * Automatically deduces the database type and runs the full prepare-execute-reflect pipeline.
 * Supports both fixpoint and non-iterative queries.
 *
 * Checks if the query is a fixpoint query or a non-iterative rule set.
 * If non-iterative, it wraps the rules in a `Fixpoint` type and executes it with
 * `max_iterations = 1` to reuse the robust fixpoint execution logic (which handles
 * iteration 0 correctly by aliasing Delta to Full and merging results).
 *
 * It also detects if the input DB is already a DeviceDB (by checking if the first relation
 * has is_device traits). If so, it skips `prepare` and `reflect` and calls `execute_kernel`
 * directly.
 *
 * @tparam Q The query type (CFixpoint or CNonIterativeRuleSets)
 * @tparam DB The Database type (HostDB or DeviceDB)
 * @param db The Database instance to execute the query on
 * @param max_iterations Maximum number of iterations for fixpoint queries (default: 100)
 */
template <AST::CQuery Q, typename DB>
void execute_gpu_query(DB& db, std::size_t max_iterations = 100) {
  // Helper to detect if DB is a DeviceDB
  constexpr bool is_device_db = []() {
    using FirstRel = std::tuple_element_t<0, typename DB::data_type>;
    return FirstRel::StorageTraits::is_device;
  }();

  if constexpr (AST::CFixpoint<Q>) {
    if constexpr (is_device_db) {
      GPUQueryExecutor<Q>::execute_kernel(db, max_iterations);
    } else {
      GPUQueryExecutor<Q>::execute(db, max_iterations);
    }
  } else if constexpr (AST::CNonIterativeRuleSets<Q>) {
    // Use helper to convert to Fixpoint and access executor
    using FixpointType = typename NonIterativeToFixpoint<Q>::type;
    using Executor = typename NonIterativeToFixpoint<Q>::Executor;
    // Run for exactly 1 iteration
    if constexpr (is_device_db) {
      Executor::execute_kernel(db, 1);
    } else {
      Executor::execute(db, 1);
    }
  } else {
    static_assert(false, "Unknown query type");
  }
}

/**
 * @brief Extract all IndexSpecs from fixpoint-level instructions.
 *
 * @details Extracts index specs from ExecutePipeline (from Pipeline::mir_ops_type)
 *          and RebuildIndex operations. This is needed for API-generated plans
 *          that don't have AllIndexSpecs metadata.
 */
namespace detail {
template <typename Instructions>
struct ExtractAllIndexSpecsFromInstructions;

template <>
struct ExtractAllIndexSpecsFromInstructions<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename Head, typename... Tail>
struct ExtractAllIndexSpecsFromInstructions<std::tuple<Head, Tail...>> {
 private:
  template <typename Inst>
  struct ExtractFromInst {
    using type = std::tuple<>;
  };

  // Extract from ExecutePipeline using Pipeline::source_specs_type (from AST)
  template <typename Pipeline>
  struct ExtractFromInst<mir::ExecutePipeline<Pipeline>> {
    // Use Pipeline::source_specs_type (computed from AST) instead of extracting from MIR
    using type = typename Pipeline::source_specs_type;
  };

  // Extract from RebuildIndex
  template <mir::CIndexSpec IndexSpecT>
  struct ExtractFromInst<mir::RebuildIndex<IndexSpecT>> {
    using type = std::tuple<IndexSpecT>;
  };

  using HeadSpecs = typename ExtractFromInst<Head>::type;
  using TailSpecs = typename ExtractAllIndexSpecsFromInstructions<std::tuple<Tail...>>::type;

 public:
  using type = mp::mp_append<HeadSpecs, TailSpecs>;
};
}  // namespace detail

/**
 * @brief Executes a pre-compiled MIR plan on the GPU.
 *
 * @details Direct execution of MIR without AST recompilation.
 *
 * @tparam Plan The MIR Plan type (FixpointPlan with Instructions/IncRels/IdxSpecs)
 * @tparam DB The Database type (HostDB or DeviceDB)
 * @param db The Database instance
 * @param max_iterations Maximum iterations for recursive steps (default: 100)
 */
template <mir::CFixpointPlan Plan, typename DB>
void execute_gpu_mir_query(DB& db, std::size_t max_iterations = 100) {
  // Use metadata directly from the Plan type
  // Use metadata directly from the Plan type, or compute defaults if missing
  using Instructions = typename Plan::instructions_type;
  using IncrementalRelationsRaw = typename Plan::incremental_relations_type;
  using CanonicalIndexSpecsRaw = typename Plan::canonical_index_specs_type;

  // If IncrementalRelations is empty, extract from instructions
  using IncrementalRelations =
      std::conditional_t<std::tuple_size_v<IncrementalRelationsRaw> == 0,
                         typename mir::ExtractDestinationSchemas<Instructions>::type,
                         IncrementalRelationsRaw>;

  // If CanonicalIndexSpecs is empty, compute from IncrementalRelations
  // For API-generated plans, extract AllIndexSpecs from instructions (slow fallback)
  // For Nim-generated plans, use pre-computed all_source_specs_type (O(1))
  using PreComputedSpecs = typename Plan::all_source_specs_type;
  using AllIndexSpecsFromInstructions =
      std::conditional_t<std::tuple_size_v<PreComputedSpecs> != 0, PreComputedSpecs,
                         typename detail::ExtractAllIndexSpecsFromInstructions<Instructions>::type>;
  using CanonicalIndexSpecs =
      std::conditional_t<std::tuple_size_v<CanonicalIndexSpecsRaw> == 0,
                         typename ::SRDatalog::detail::CollectCanonicalIndexSpecs<
                             IncrementalRelations, AllIndexSpecsFromInstructions>::type,
                         CanonicalIndexSpecsRaw>;

  using SchemaTuple = typename DB::schema_type;

  constexpr bool is_device_db = []() {
    using FirstRel = std::tuple_element_t<0, typename DB::data_type>;
    return FirstRel::StorageTraits::is_device;
  }();

  if constexpr (is_device_db) {
    // If input is DeviceDB, just execute
    using Executor = GPUPlanExecutor<Instructions, IncrementalRelations, CanonicalIndexSpecs>;
    Executor::execute(db, max_iterations);
  } else {
    // If input is HostDB, copy to device, execute
    // Note: Reflection logic is omitted as per user direction ("reflect is broken").
    // The user handles data retrieval manually or this function acts as purely execution.
    auto device_db = copy_host_to_device(db);
    using DeviceDBType = decltype(device_db);

    using Executor = GPUPlanExecutor<Instructions, IncrementalRelations, CanonicalIndexSpecs>;
    Executor::execute(device_db, max_iterations);
  }
}

/**
 * @brief Executes a single PostStratumReconstructInternCols instruction.
 *
 * @details This is a standalone instruction that runs AFTER fixpoint completion.
 * It reconstructs intern column values from the canonical index to column storage.
 * NOTE: The canonical index may not have all columns, so we use size() not reconstruct_to_relation.
 */
template <mir::CPostStratumReconstructInternCols Instr, typename DB>
void execute_gpu_mir_query(DB& db, std::size_t /*max_iterations*/ = 100) {
  using IndexSpecT = typename Instr::index_spec_type;
  using Schema = typename IndexSpecT::schema_type;
  constexpr std::size_t arity = std::tuple_size_v<typename Schema::attr_ts_type>;

  auto& rel = get_relation_by_schema<Schema, FULL_VER>(db);

  // Extract runtime index spec from compile-time IndexSpecT (this is the canonical index)
  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto canonical_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // Get the canonical index WITHOUT rebuilding (it's authoritative after merge_index)
  // CRITICAL: ensure_index(_, false) preserves the merged index data
  // ensure_index(_, true) would rebuild from stale intern columns and destroy the merged data!
  rel.ensure_index(canonical_spec, false);
  auto& canonical_idx = rel.get_index(canonical_spec);
  std::size_t idx_size = canonical_idx.size();

  // Build a full-column index spec for reconstruction (must contain all columns)
  SRDatalog::IndexSpec full_spec;
  full_spec.cols.reserve(arity);
  for (std::size_t i = 0; i < arity; ++i) {
    full_spec.cols.push_back(static_cast<int>(i));
  }

  // Get full index WITHOUT rebuilding - use existing merged data
  // If full_spec == canonical_spec, this is the same index
  // Otherwise, we need a full-column index for reconstruction
  rel.ensure_index(full_spec, false);
  auto& full_idx = rel.get_index(full_spec);

  // Resize intern columns to match index size
  rel.resize_interned_columns(idx_size);
  if constexpr (has_provenance_v<typename Schema::semiring_type>) {
    rel.provenance().resize(idx_size);
  }

  // Copy values from full index to intern columns
  // For NoProvenance, std::monostate overload exists for reconstruct_to_relation
  full_idx.reconstruct_to_relation(full_spec, rel.unsafe_interned_columns(), rel.provenance());
}

/**
 * @brief Executes a standalone RebuildIndex instruction.
 *
 * @details This is a standalone instruction that can run at any point in the program.
 * It ensures the specified index exists and is built from current storage.
 * Used by the PreReconstructRebuild MIR pass to refresh indices between strata.
 */
template <mir::CRebuildIndex Instr, typename DB>
void execute_gpu_mir_query(DB& db, std::size_t /*max_iterations*/ = 100) {
  using IndexSpecT = typename Instr::index_spec_type;
  using Schema = typename IndexSpecT::schema_type;
  constexpr std::size_t ver = IndexSpecT::kVersion;

  auto& rel = get_relation_by_schema<Schema, ver>(db);

  // Extract runtime index spec from compile-time IndexSpecT
  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // Rebuild the index from storage (force rebuild)
  rel.ensure_index(runtime_spec, true);
}

/**
 * @brief Clangd inspection utils, hover here to see the type
 */
template <typename T>
void inspect() {}

}  // namespace SRDatalog::GPU
