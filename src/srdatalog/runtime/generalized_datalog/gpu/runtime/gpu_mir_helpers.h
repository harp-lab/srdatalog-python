/**
 * @file gpu_mir_helpers.h
 * @brief Standalone free functions for GPU MIR operations.
 *
 * @details Extracted from executor_impl/ specializations with tuple dispatch removed.
 * Each function is a self-contained operation that takes explicit template parameters
 * instead of relying on TailOps... introspection. Used by the Nim orchestrator to
 * generate imperative C++ fixpoint loops.
 *
 * These functions are the "primitive provider" layer - Nim decides the order of
 * operations, and these functions execute each individual operation.
 */

#pragma once

#include "gpu/index_ops.h"
#include "gpu_fixpoint_executor_common.h"
#include <chrono>

namespace SRDatalog::GPU::mir_helpers {

// ============================================================================
// compute_delta_fn - Extracted from executor_compute_delta.h
// ============================================================================

/**
 * @brief Compute DELTA = NEW - FULL using set_difference_update.
 *
 * @details Clears delta, computes set difference from NEW against FULL,
 * reconstructs delta from index, binds storage, and clears NEW.
 *
 * @tparam Schema The relation schema type
 * @tparam IndexSpecT The canonical index spec for this relation
 * @tparam DB The database type
 * @param db The database reference
 * @return true if delta has new tuples
 */
template <typename Schema, typename IndexSpecT, typename DB>
bool compute_delta_fn(DB& db) {
  constexpr std::size_t FromVer = NEW_VER;
  constexpr std::size_t ToVer = FULL_VER;
  constexpr std::size_t DeltaVer = DELTA_VER;

  auto& new_rel = get_relation_by_schema<Schema, FromVer>(db);
  auto& full_rel = get_relation_by_schema<Schema, ToVer>(db);
  auto& delta_rel = get_relation_by_schema<Schema, DeltaVer>(db);

  delta_rel.clear();

  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // NEW_VER index was just built by rebuild_index_fn — use get_index() directly.
  auto& new_idx = new_rel.get_index(runtime_spec);

  full_rel.ensure_index(runtime_spec, false);
  auto& full_idx = full_rel.get_index(runtime_spec);

  delta_rel.ensure_index(runtime_spec);
  auto& delta_idx = delta_rel.get_index(runtime_spec);
  // set_difference_update populates delta_idx directly (index-to-index).
  // DELTA is fully index-only — no intern cols touched.
  index_ops::set_difference(new_idx, full_idx, delta_idx);
  new_rel.clear();
  return !delta_idx.empty();
}

// ============================================================================
// compute_delta_index_fn - Index-only delta computation (JIT path)
// ============================================================================

/**
 * @brief Compute DELTA = NEW - FULL at index level only.
 *
 * @details No intern cols are touched. DELTA is fully index-only.
 * Uses explicit index specs for NEW, FULL, and DELTA versions.
 *
 * @tparam NewSpecT The NEW_VER canonical index spec
 * @tparam FullSpecT The FULL_VER canonical index spec (same columns as NewSpecT)
 * @tparam DeltaSpecT The DELTA_VER canonical index spec (same columns)
 * @tparam DB The database type
 * @param db The database reference
 * @return true if delta has new tuples
 */
template <typename NewSpecT, typename FullSpecT, typename DeltaSpecT, typename DB>
bool compute_delta_index_fn(DB& db) {
  using Schema = typename NewSpecT::schema_type;

  auto& new_rel = get_relation_by_schema<Schema, NEW_VER>(db);
  auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(db);
  auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(db);

  delta_rel.clear();

  using ColSeq = typename NewSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // NEW_VER index was just built by rebuild_index_fn — use get_index() directly.
  auto& new_idx = new_rel.get_index(runtime_spec);

  full_rel.ensure_index(runtime_spec, false);
  auto& full_idx = full_rel.get_index(runtime_spec);

  delta_rel.ensure_index(runtime_spec, false);
  auto& delta_idx = delta_rel.get_index(runtime_spec);

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t new_size = new_idx.size();
    std::size_t full_size = full_idx.size();
    index_ops::set_difference(new_idx, full_idx, delta_idx);
  }

  // Clear NEW (intern cols + indices)
  new_rel.clear();

  return !delta_idx.empty();
}

// ============================================================================
// merge_index_fn - Extracted from executor_merge_index.h
// ============================================================================

/**
 * @brief Merge DELTA index into FULL index using CPO dispatch.
 *
 * @tparam IndexSpecT The index spec (must be FULL_VER)
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename IndexSpecT, typename DB>
void merge_index_fn(DB& db) {
  using Schema = typename IndexSpecT::schema_type;
  static_assert(IndexSpecT::kVersion == FULL_VER, "MergeIndex must be for FULL_VER index");

  auto& full_rel = get_relation_by_schema<Schema, FULL_VER>(db);
  auto& delta_rel = get_relation_by_schema<Schema, DELTA_VER>(db);

  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  full_rel.ensure_index(runtime_spec, false);
  auto& full_idx = full_rel.get_index(runtime_spec);

  // Delta index was just built by compute_delta_fn — use get_index() directly.
  auto& delta_idx = delta_rel.get_index(runtime_spec);

  if (!delta_idx.empty()) {
    std::size_t full_before = full_idx.size();
    std::size_t delta_size = delta_idx.size();
    index_ops::merge_index(full_idx, delta_idx, full_rel, delta_rel);
  }
}

// ============================================================================
// check_size_fn - Extracted from executor_check_size.h
// ============================================================================

/**
 * @brief Check/rebuild canonical index for a relation.
 *
 * @details The original TMP version used mp::mp_find_if to search TailOps for the
 * canonical IndexSpec. In the orchestrator, the Nim codegen knows this statically
 * and passes CanonicalSpecT explicitly.
 *
 * @tparam Schema The relation schema type
 * @tparam Ver The version to check (typically NEW_VER)
 * @tparam CanonicalSpecT The canonical index spec (explicitly provided by Nim)
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename Schema, std::size_t Ver, typename CanonicalSpecT, typename DB>
void check_size_fn(DB& db) {
  auto& r = get_relation_by_schema<Schema, Ver>(db);

  using ColSeq = typename CanonicalSpecT::column_indexes_type;
  auto runtime_canonical_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // Force rebuild to ensure index reflects relation data
  r.ensure_index(runtime_canonical_spec, true);
}

// ============================================================================
// rebuild_index_fn - Extracted from executor_rebuild_index.h
// ============================================================================

/**
 * @brief Rebuild an index for a relation.
 *
 * @tparam IndexSpecT The index spec to rebuild
 * @tparam DB The database type
 * @param db The database reference
 * @return true if NEW_VER had new tuples (only meaningful for NEW_VER)
 */
template <typename IndexSpecT, typename DB>
bool rebuild_index_fn(DB& db) {
  using Schema = typename IndexSpecT::schema_type;
  using ColSeq = typename IndexSpecT::column_indexes_type;
  constexpr std::size_t ver = IndexSpecT::kVersion;

  auto& rel = get_relation_by_schema<Schema, ver>(db);
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  bool new_tuples = false;
  nvtxRangePushA("RebuildIndex");
  if constexpr (ver == NEW_VER) {
    // One-shot ownership transfer: intern cols → index (zero-copy for identity spec).
    // After this call, intern cols are empty and the index is the sole data owner.
    auto& new_idx = rel.build_index_take_ownership(runtime_spec);
    std::size_t root_degree = new_idx.root().degree();
    new_tuples = root_degree > 0;
  } else if constexpr (ver == FULL_VER) {
    rel.ensure_index(runtime_spec, false);
  } else {
    // DELTA_VER: fallback for TMP executor.
    // JIT path uses the explicit rebuild_index_from_index_fn instead.
    rel.ensure_index(runtime_spec, true);
  }
  nvtxRangePop();

  return new_tuples;
}

// ============================================================================
// rebuild_index_from_index_fn - Explicit index-to-index build (JIT path)
// ============================================================================

/**
 * @brief Build target index from source index (index-to-index permutation).
 *
 * @details Uses build_from_index(const& source) on the target index.
 * Source and target must be from the same relation and version.
 * No intern cols are involved.
 *
 * @tparam SourceSpecT The source index spec (e.g., canonical DELTA [1,0])
 * @tparam TargetSpecT The target index spec (e.g., non-canonical DELTA [0,1])
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename SourceSpecT, typename TargetSpecT, typename DB>
void rebuild_index_from_index_fn(DB& db) {
  using Schema = typename SourceSpecT::schema_type;
  constexpr std::size_t ver = SourceSpecT::kVersion;
  static_assert(SourceSpecT::kVersion == TargetSpecT::kVersion,
                "Source and target must have the same version");

  auto& rel = get_relation_by_schema<Schema, ver>(db);

  using SourceColSeq = typename SourceSpecT::column_indexes_type;
  auto source_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(SourceColSeq{});

  using TargetColSeq = typename TargetSpecT::column_indexes_type;
  auto target_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(TargetColSeq{});

  // Get existing source index (must already exist from compute_delta_index_fn)
  auto& source_idx = rel.get_index(source_spec);

  if (source_idx.empty()) {
    // No data in source — ensure target exists but leave it empty
    rel.ensure_index(target_spec, false);
    return;
  }

  // Ensure target entry exists (don't build from intern cols)
  rel.ensure_index(target_spec, false);
  auto& target_idx = rel.get_index(target_spec);

  // std::cerr << "[rebuild_index_from_index] " << rel.name() << " source=" <<
  // source_spec.to_string()
  //           << " size=" << source_idx.size() << " -> target=" << target_spec.to_string()
  //           << std::endl;

  // Direct index-to-index build using const ref source
  target_idx.build_from_index(source_spec, target_spec, source_idx);

  // std::cerr << "[rebuild_index_from_index] " << rel.name() << " target=" <<
  // target_spec.to_string()
  //           << " built size=" << target_idx.size() << std::endl;
}

// ============================================================================
// clear_relation_fn - Extracted from executor_relation_ops.h
// ============================================================================

/**
 * @brief Clear a relation.
 *
 * @tparam Schema The relation schema type
 * @tparam Ver The version to clear
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename Schema, std::size_t Ver, typename DB>
void clear_relation_fn(DB& db) {
  auto& r = get_relation_by_schema<Schema, Ver>(db);
  r.clear();
}

// ============================================================================
// swap_relations_fn - Extracted from executor_relation_ops.h
// ============================================================================

/**
 * @brief Swap DELTA and NEWT relations.
 *
 * @tparam Schema The relation schema type
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename Schema, typename DB>
void swap_relations_fn(DB& db) {
  auto& r1 = get_relation_by_schema<Schema, DELTA_VER>(db);
  auto& r2 = get_relation_by_schema<Schema, NEW_VER>(db);
  using std::swap;
  swap(r1, r2);
}

// ============================================================================
// set_version_fn - Extracted from executor_relation_ops.h
// ============================================================================

/**
 * @brief Set the version of a relation.
 *
 * @tparam Schema The relation schema type
 * @tparam Ver The version to set
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename Schema, std::size_t Ver, typename DB>
void set_version_fn(DB& db) {
  auto& r = get_relation_by_schema<Schema, Ver>(db);
  r.set_version(Ver);
}

// ============================================================================
// insert_merge_fn - Extracted from executor_insert_merge.h
// ============================================================================

/**
 * @brief Insert/merge from one relation version into another.
 *
 * @tparam Schema The relation schema type
 * @tparam FromVer Source version
 * @tparam ToVer Destination version
 * @tparam IndexSpecT The index spec to use
 * @tparam DB The database type
 * @param db The database reference
 * @param iteration Current iteration (0 redirects DELTA reads to FULL)
 */
template <typename Schema, std::size_t FromVer, std::size_t ToVer, typename IndexSpecT, typename DB>
void insert_merge_fn(DB& db, std::size_t iteration) {
  auto& from = [&]() -> auto& {
    if constexpr (FromVer == DELTA_VER) {
      if (iteration == 0) {
        return get_relation_by_schema<Schema, FULL_VER>(db);
      }
    }
    return get_relation_by_schema<Schema, FromVer>(db);
  }();

  auto& to = get_relation_by_schema<Schema, ToVer>(db);

  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  auto& from_idx = from.ensure_index(runtime_spec);

  if (iteration == 0) {
    to.ensure_index(runtime_spec);
  } else {
    try {
      auto& test_idx = to.get_index(runtime_spec);
      (void)test_idx;
    } catch (...) {
      to.ensure_index(runtime_spec);
    }
  }
  auto& to_idx = to.get_index(runtime_spec);

  std::size_t row_offset = to_idx.size();
  to_idx.merge(from_idx, row_offset);
}

// ============================================================================
// create_index_fn - Extracted from executor_create_index.h
// ============================================================================

/**
 * @brief Create/ensure an index exists for a relation.
 *
 * @tparam IndexSpecT The index spec to create
 * @tparam DB The database type
 * @param db The database reference
 * @param iteration Current iteration (0 redirects DELTA reads to FULL)
 */
template <typename IndexSpecT, typename DB>
void create_index_fn(DB& db, std::size_t iteration) {
  using Schema = typename IndexSpecT::schema_type;
  using ColSeq = typename IndexSpecT::column_indexes_type;
  constexpr std::size_t ver = IndexSpecT::kVersion;

  auto& rel = [&]() -> auto& {
    if constexpr (ver == DELTA_VER) {
      if (iteration == 0) {
        return get_relation_by_schema<Schema, FULL_VER>(db);
      }
    }
    return get_relation_by_schema<Schema, ver>(db);
  }();

  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  rel.ensure_index(runtime_spec);
}

// ============================================================================
// create_flat_view_fn - register an unsorted "flat" index over intern columns
// ============================================================================

/**
 * @brief Create an index slot without sorting/building.
 *
 * @details Used by the negation-pushdown split lowering: Pipeline A writes into
 * a temp relation's interned columns, and Pipeline B then scans that temp
 * relation. The scan does not need a sorted index — only the index slot needs
 * to exist so downstream view-construction code can find it. This mirrors
 * @ref create_index_fn but passes `build=false` to skip the GPU sort.
 *
 * @tparam IndexSpecT The index spec to register
 * @tparam DB The database type
 */
template <typename IndexSpecT, typename DB>
void create_flat_view_fn(DB& db) {
  using Schema = typename IndexSpecT::schema_type;
  using ColSeq = typename IndexSpecT::column_indexes_type;
  constexpr std::size_t ver = IndexSpecT::kVersion;

  auto& rel = get_relation_by_schema<Schema, ver>(db);

  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  // Build the index over the temp relation's freshly-written intern columns
  // so Pipeline B's scan sees a valid iterable view.
  rel.ensure_index(runtime_spec, true);
}

// ============================================================================
// reconstruct_fn - Extracted from executor_post_stratum_reconstruct.h
// ============================================================================

/**
 * @brief Reconstruct intern column values from canonical index to column storage.
 *
 * @details Called after fixpoint completes to ensure downstream strata have
 * consistent linear column storage.
 *
 * @tparam IndexSpecT The canonical index spec
 * @tparam DB The database type
 * @param db The database reference
 */
template <typename IndexSpecT, typename DB>
void reconstruct_fn(DB& db) {
  using Schema = typename IndexSpecT::schema_type;

  auto& rel = get_relation_by_schema<Schema, FULL_VER>(db);

  using ColSeq = typename IndexSpecT::column_indexes_type;
  auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
    return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
  }(ColSeq{});

  rel.ensure_index(runtime_spec, false);
  auto& idx = rel.get_index(runtime_spec);

  std::size_t idx_size = idx.size();
  rel.resize_interned_columns(idx_size);
  if constexpr (has_provenance_v<typename Schema::semiring_type>) {
    rel.provenance().resize(idx_size);
  }

  idx.reconstruct_to_relation(runtime_spec, rel.unsafe_interned_columns(), rel.provenance());
}

}  // namespace SRDatalog::GPU::mir_helpers
