#pragma once

/**
 * @file index_concepts.h
 * @brief C++20 concepts for GPU index data structures (BYODS contract)
 *
 * @details Every GPU index type must satisfy IndexReader + IndexWriter.
 * These concepts cover the FULL interface that the runtime (gpu_mir_helpers.h,
 * jit_executor.h, relation_col.h) actually calls on an index.
 *
 * To add a new index type, implement all methods required by IndexWriter,
 * then optionally provide ADL overloads for CPOs in index_ops.h.
 *
 * @see index_ops.h for customization point objects (merge, set_difference)
 */

#include <concepts>
#include <cstddef>

namespace SRDatalog::GPU {

/**
 * @brief Concept for index read operations (used in joins/scans)
 *
 * @details Required by: kernel launcher (jit_executor.h), pipeline executor
 */
template <typename Idx>
concept IndexReader = requires(const Idx& idx) {
  // ── Types ──
  typename Idx::NodeView;
  typename Idx::NodeHandle;
  typename Idx::ValueType;
  { Idx::arity } -> std::convertible_to<std::size_t>;

  // ── Core reads ──
  { idx.root() } -> std::same_as<typename Idx::NodeHandle>;
  { idx.view() } -> std::same_as<typename Idx::NodeView>;
  { idx.size() } -> std::convertible_to<std::size_t>;
  { idx.empty() } -> std::convertible_to<bool>;

  // ── Kernel work partitioning ──
  // The kernel launcher distributes work across warps using unique first-column
  // values. Every index must provide these (jit_executor.h lines 203-206).
  { idx.num_unique_root_values() } -> std::convertible_to<std::size_t>;
  { idx.root_unique_values() };  // returns DeviceArray<ValueType>&

  // ── Dirty-checking ──
  // The relation uses rows_processed() to detect when new data was added since
  // the last index build (relation_col.h is_dirty()). Every index must track this.
  { idx.rows_processed() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for index write operations (merge, set_difference, build, reconstruct)
 *
 * @details Required by: gpu_mir_helpers.h (merge_index_fn, compute_delta_index_fn,
 * rebuild_index_fn, reconstruct_fn)
 */
template <typename Idx>
concept IndexWriter = IndexReader<Idx> && requires(Idx& idx) {
  // ── Semi-naive operations ──
  { idx.merge(std::declval<Idx&>(), std::size_t{}) };
  { idx.set_difference_update(std::declval<Idx&>(), std::declval<Idx&>()) };
  { idx.clear() };

  // ── Dirty-checking (mutable) ──
  { idx.update_rows_processed(std::size_t{}) };
};

// ═══════════════════════════════════════════════════════════════════════════
// Storage Layout Traits
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Storage layout for relation column data
 *
 * @details Determines whether the relation stores columns as SoA (NDDeviceArray)
 * or AoS (AoSDeviceArray). Used by relation_col.h to select DeviceColsType.
 * Default is SoA. Specialize IndexStorageTraits for indices that need AoS.
 */
enum class StorageLayout {
  SoA,  // Structure of Arrays (Default, e.g., DeviceSortedArrayIndex)
  AoS   // Array of Structures (e.g., DeviceTVJoinIndex)
};

template <typename Idx>
struct IndexStorageTraits {
  static constexpr StorageLayout layout = StorageLayout::SoA;
};

}  // namespace SRDatalog::GPU
