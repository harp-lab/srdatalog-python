/**
 * @file device_2level_index.h
 * @brief 2-Level LSM Index wrapping two DeviceSortedArrayIndex instances (HEAD + FULL)
 *
 * @details Addresses the O(N+M) merge bottleneck in DeviceSortedArrayIndex when DELTA << FULL.
 * Instead of merging DELTA directly into FULL (expensive thrust::merge copying entire FULL),
 * we maintain two sorted segments:
 *
 * - HEAD: small sorted buffer, receives DELTA merges cheaply O(D+H)
 * - FULL: large sorted array, never touched during fixpoint iteration
 *
 * No compaction during fixpoint:
 * - merge_index: DELTA → HEAD, O(D+H) instead of O(D+F)
 * - set_difference: 2-step diff against FULL then HEAD, avoids O(H+F) compaction
 * - view()/root(): returns FULL segment only — safe because semi-naive join only
 *   reads DELTA version (single segment, fresh from set_difference)
 * - reconstruct (post-stratum): compacts HEAD→FULL once for final result
 *
 * @tparam SR Semiring type
 * @tparam AttrTuple Column element tuple type
 * @tparam ValueType Encoded value type (default: uint32_t)
 * @tparam RowIdType Row ID type (default: uint32_t)
 */

#pragma once

#include "device_sorted_array_index.h"
#include "index_concepts.h"
#include <cstddef>

namespace SRDatalog::GPU {

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueTypeParam = uint32_t,
          typename RowIdType = uint32_t>
class Device2LevelIndex {
 public:
  using DSAI = DeviceSortedArrayIndex<SR, AttrTuple, ValueTypeParam, RowIdType>;
  using NodeView = typename DSAI::NodeView;
  using NodeHandle = typename DSAI::NodeHandle;
  // Required by IndexReader concept
  using ValueType = ValueTypeParam;
  static constexpr std::size_t arity = DSAI::arity;

  Device2LevelIndex() = default;
  ~Device2LevelIndex() = default;

  // Non-copyable
  Device2LevelIndex(const Device2LevelIndex&) = delete;
  Device2LevelIndex& operator=(const Device2LevelIndex&) = delete;

  // Movable
  Device2LevelIndex(Device2LevelIndex&&) noexcept = default;
  Device2LevelIndex& operator=(Device2LevelIndex&&) noexcept = default;

  // ── IndexReader interface ──────────────────────────────────────────────
  // view()/root() return FULL segment only — used for single-segment reads (DELTA).
  // For FULL reads in non-linear recursion, kernel must iterate both segments
  // using full_view()/head_view() and full_root()/head_root() separately.
  // No compaction on read. Compaction happens only:
  //   (a) in merge_index_impl when HEAD > kCompactRatio * FULL
  //   (b) in reconstruct_to_relation at fixpoint exit

  [[nodiscard]] NodeHandle root() const {
    return full_.root();
  }

  [[nodiscard]] NodeView view() const {
    return full_.view();
  }

  // ── Segment-specific access for multi-segment kernel iteration ─────────
  // Used by codegen when join reads FULL version of a 2-level relation.
  // Kernel iterates both segments independently (no compaction).

  [[nodiscard]] NodeView full_view() const {
    return full_.view();
  }

  [[nodiscard]] NodeHandle full_root() const {
    return full_.root();
  }

  [[nodiscard]] NodeView head_view() const {
    return head_.view();
  }

  [[nodiscard]] NodeHandle head_root() const {
    return head_.root();
  }

  /// @brief Total size across both segments
  [[nodiscard]] std::size_t size() const noexcept {
    return head_.size() + full_.size();
  }

  /// @brief Check if both segments are empty
  [[nodiscard]] bool empty() const noexcept {
    return head_.empty() && full_.empty();
  }

  // ── IndexWriter interface ──────────────────────────────────────────────

  /// @brief Merge DELTA into HEAD (cheap O(H+M) instead of O(N+M))
  /// @param other Source index (DELTA) to merge
  /// @param row_id_offset Offset for row IDs
  void merge(const Device2LevelIndex& other, std::size_t row_id_offset) {
    // Merge other's content into head_
    // other may have content in both head_ and full_ — compact it first
    // But since we receive DELTA which is typically a single DSAI, we merge both segments
    if (!other.full_.empty()) {
      head_.merge(other.full_, row_id_offset);
    }
    if (!other.head_.empty()) {
      head_.merge(other.head_, row_id_offset + other.full_.size());
    }
  }

  /// @brief Set difference: (this - full_idx) → delta_idx
  /// @details When HEAD is non-empty, uses a fused kernel that probes both FULL and HEAD
  ///          in a single pass with atomic compaction — eliminates the old 2-step approach
  ///          that created an intermediate temp index with separate kernel chains.
  void set_difference_update(Device2LevelIndex& full_idx, Device2LevelIndex& delta_idx) {
    // this (NEW) is always single-segment (built fresh from pipeline output)
    compact();

    if (full_idx.head_.empty()) {
      // Fast path: no HEAD segment, single diff against FULL
      full_.set_difference_update(full_idx.full_, delta_idx.full_);
    } else {
      // Fused path: single kernel probes both FULL and HEAD
      full_.set_difference_update_dual(full_idx.full_, full_idx.head_, delta_idx.full_);
    }
  }

  /// @brief Clear both segments
  void clear() noexcept {
    head_.clear();
    full_.clear();
  }

  // ── Build interface (delegates to full_) ───────────────────────────────

  void build_from_encoded_device(const IndexSpec& spec,
                                 NDDeviceArray<ValueType, arity>& encoded_cols,
                                 DeviceArray<semiring_value_t<SR>>& provenance) {
    full_.build_from_encoded_device(spec, encoded_cols, provenance);
  }

  void build_from_encoded_device(const IndexSpec& spec,
                                 NDDeviceArray<ValueType, arity>& encoded_cols,
                                 [[maybe_unused]] std::monostate& provenance) {
    full_.build_from_encoded_device(spec, encoded_cols, provenance);
  }

  void build_take_ownership(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& encoded_cols,
                            DeviceArray<semiring_value_t<SR>>& provenance) {
    full_.build_take_ownership(spec, encoded_cols, provenance);
  }

  void build_take_ownership(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& encoded_cols,
                            [[maybe_unused]] std::monostate& provenance) {
    full_.build_take_ownership(spec, encoded_cols, provenance);
  }

  void build_from_index(const IndexSpec& source_spec, const IndexSpec& target_spec,
                        const Device2LevelIndex& source) {
    // Source must have all data visible — compact it
    const_cast<Device2LevelIndex&>(source).compact();
    full_.build_from_index(source_spec, target_spec, source.full_);
  }

  // ── Reconstruction ─────────────────────────────────────────────────────

  void reconstruct_to_relation(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& output_cols,
                               DeviceArray<semiring_value_t<SR>>& output_prov) const {
    const_cast<Device2LevelIndex*>(this)->compact();
    full_.reconstruct_to_relation(spec, output_cols, output_prov);
  }

  void reconstruct_to_relation(const IndexSpec& spec, NDDeviceArray<ValueType, arity>& output_cols,
                               [[maybe_unused]] std::monostate& output_prov) const {
    const_cast<Device2LevelIndex*>(this)->compact();
    full_.reconstruct_to_relation(spec, output_cols, output_prov);
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return full_.rows_processed() + head_.rows_processed();
  }

  void update_rows_processed(std::size_t new_value) noexcept {
    // After reconstruct, all data is in full_
    full_.update_rows_processed(new_value);
    head_.update_rows_processed(0);
  }

  [[nodiscard]] std::size_t bytes_used() const {
    return head_.bytes_used() + full_.bytes_used();
  }

  [[nodiscard]] std::size_t num_unique_root_values() const noexcept {
    return full_.num_unique_root_values();
  }

  [[nodiscard]] auto& root_unique_values() const {
    return full_.root_unique_values();
  }

  // Segment-specific root_unique_values for BG 2-segment iteration.
  // Both FULL and HEAD are DSAI instances with their own unique root arrays
  // already on device — no data movement needed, just pass the pointers.

  [[nodiscard]] std::size_t full_num_unique_root_values() const noexcept {
    return full_.num_unique_root_values();
  }

  [[nodiscard]] auto& full_root_unique_values() const {
    return full_.root_unique_values();
  }

  [[nodiscard]] std::size_t head_num_unique_root_values() const noexcept {
    return head_.num_unique_root_values();
  }

  [[nodiscard]] auto& head_root_unique_values() const {
    return head_.root_unique_values();
  }

  [[nodiscard]] auto& data() const {
    return full_.data();
  }

  [[nodiscard]] thrust::device_ptr<semiring_value_t<SR>> provenance_ptr() const noexcept {
    return full_.provenance_ptr();
  }

  void clone_from(const Device2LevelIndex& other) {
    full_.clone_from(other.full_);
    head_.clone_from(other.head_);
  }

  void print_debug() const {
    printf("Device2LevelIndex: HEAD[%zu] + FULL[%zu]\n", head_.size(), full_.size());
    printf("  HEAD: ");
    head_.print_debug();
    printf("  FULL: ");
    full_.print_debug();
  }

  // ── Direct access to segments (for testing/advanced use) ───────────────

  [[nodiscard]] const DSAI& head() const noexcept {
    return head_;
  }
  [[nodiscard]] const DSAI& full() const noexcept {
    return full_;
  }
  [[nodiscard]] DSAI& head() noexcept {
    return head_;
  }
  [[nodiscard]] DSAI& full() noexcept {
    return full_;
  }

  /// @brief Force compaction (merge HEAD into FULL)
  void compact() {
    if (head_.empty())
      return;
    if (full_.empty()) {
      // Just swap head into full
      full_ = std::move(head_);
      head_ = DSAI{};
    } else {
      full_.merge(head_, full_.size());
      head_.clear();
    }
  }

 private:
  DSAI head_;  // Small sorted buffer — receives DELTA merges
  DSAI full_;  // Large sorted array — only touched during reconstruction
};

// ── ADL customization for index_ops ────────────────────────────────────────

/// @brief Custom merge_index_impl for Device2LevelIndex
/// @details Merges DELTA into FULL's HEAD segment, then compacts HEAD→FULL
///          when HEAD exceeds a fraction of FULL. This keeps HEAD small so
///          each merge is O(D+H) where H << F, and compaction O(H+F) is
///          amortized over many cheap merges.
///
/// Compaction policy: compact when HEAD > kCompactRatio * FULL.
/// With ratio 0.1: HEAD stays ≤10% of FULL. Each merge is O(D + 0.1·F).
/// Compaction happens every ~0.1·F/D iterations, costs O(1.1·F).
/// Amortized per merge: O(D + 0.1·F) + O(1.1·F)/(0.1·F/D) = O(D + 0.1·F + 11·D) ≈ O(12·D + 0.1·F)
/// vs DSAI's O(D+F) per merge. Net win when F >> 120·D (i.e., when FULL is large).
template <Semiring SR, ColumnElementTuple AttrTuple, typename VT, typename RI, typename FullRel,
          typename DeltaRel>
void merge_index_impl(Device2LevelIndex<SR, AttrTuple, VT, RI>& full,
                      Device2LevelIndex<SR, AttrTuple, VT, RI>& delta, FullRel& /*full_rel*/,
                      DeltaRel& /*delta_rel*/) {
  static constexpr double kCompactRatio = 0.1;  // Compact when HEAD > 10% of FULL

  // Delta's content (from set_difference_update) is in delta.full_
  // Merge it into full's head_ for cheap O(D+H) merge
  if (!delta.full().empty()) {
    full.head().merge(delta.full(), full.size());
  }
  if (!delta.head().empty()) {
    full.head().merge(delta.head(), full.size());
  }

  // Compact HEAD→FULL when HEAD grows too large relative to FULL.
  // This keeps individual merges cheap by bounding HEAD size.
  std::size_t h = full.head().size();
  std::size_t f = full.full().size();
  if (h > 0 && (f == 0 || h > static_cast<std::size_t>(kCompactRatio * static_cast<double>(f)))) {
    full.compact();
  }
}

// ── Concept checks ─────────────────────────────────────────────────────────
// Cannot static_assert here (concepts need full type info); unit tests verify.

}  // namespace SRDatalog::GPU
