#pragma once

#include "../../TVJoin/join_skew/tuples_with_prov.cuh"
#include "adapter/tvjoin_adapter.h"
#include "device_array.h"
#include "gpu_api.h"
#include "index.h"
#include "index_concepts.h"

namespace SRDatalog::GPU {
// Forward declare for traits specialization
template <typename SR, typename T, std::size_t Arity, std::size_t... IndexCols>
class DeviceTVJoinIndex;

template <typename SR, typename T, std::size_t Arity, std::size_t... IndexCols>
struct IndexStorageTraits<DeviceTVJoinIndex<SR, T, Arity, IndexCols...>> {
  static constexpr StorageLayout layout = StorageLayout::AoS;
};
}  // namespace SRDatalog::GPU

#include "nd_device_array.h"
#include "semiring.h"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <vector>

namespace SRDatalog::GPU {

/**
 * @brief Lightweight view of the TVJoin Index for kernels.
 *        Deals directly with multi-level pointers.
 */
template <typename T>
struct TVJoinNodeView {
  const T* const* const* sorted_arr_ptrs;
  const std::size_t* sorted_arr_sizes;
  std::size_t num_levels;
};

/**
 * @brief Wrapper around TVJoinIndex to satisfy SRDatalog Index concepts.
 */
template <typename SR, typename T, std::size_t Arity, std::size_t... IndexCols>
class DeviceTVJoinIndex {
 public:
  using IndexImpl = TVJoinIndex<T, Arity, IndexCols...>;
  using NodeHandle = typename IndexImpl::NodeHandle;
  using NodeView = NodeHandle;

  using ValueType = T;
  static constexpr std::size_t arity = Arity;

  DeviceTVJoinIndex() = default;

  // Rule of 5: Move-only
  DeviceTVJoinIndex(DeviceTVJoinIndex&&) = default;
  DeviceTVJoinIndex& operator=(DeviceTVJoinIndex&&) = default;
  DeviceTVJoinIndex(const DeviceTVJoinIndex&) = delete;
  DeviceTVJoinIndex& operator=(const DeviceTVJoinIndex&) = delete;

  // --- IndexReader Concept ---

  [[nodiscard]] NodeHandle root() const {
    return impl_.get_node_handle();
  }

  [[nodiscard]] NodeView view() const {
    return impl_.get_node_handle();
  }

  [[nodiscard]] std::size_t size() const {
    return impl_.size();
  }

  [[nodiscard]] bool empty() const {
    return impl_.size() == 0;
  }

  /// @brief Returns processed rows for dirty-checking (matches DeviceSortedArrayIndex API)
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return rows_processed_;
  }

  /// @brief Update rows_processed to match the relation size after reconstruct
  void update_rows_processed(std::size_t new_value) noexcept {
    rows_processed_ = new_value;
  }

  // --- IndexWriter Concept ---

  void append(AoSDeviceArray<T, Arity>& data) {
    impl_.append(data);
    update_root_unique_values();
  }

  // Build directly from AoS (Zero Copy / Move)
  // Replaces old build_from_encoded_device which did SoA->AoS conversion
  void build_from_encoded_device(const IndexSpec& spec, AoSDeviceArray<T, Arity>& aos_data,
                                 DeviceArray<semiring_value_t<SR>>& provenance) {
    append(aos_data);
  }

  // Helper kernel for reconstruction (Index -> AoS)
  template <typename Handle>
  __global__ static void reconstruct_aos_kernel(Handle root, T* output,
                                                semiring_value_t<SR>* prov) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < root.size()) {
      const T* tuple = root.get_tuple_at(idx);
      if (tuple) {
#pragma unroll
        for (std::size_t c = 0; c < Arity; ++c) {
          output[idx * Arity + c] = tuple[c];
        }
      }
    }
    if (prov) {
      prov[idx] = true;
    }
  }

  // Reconstruct (Index -> AoS for Relation)
  void reconstruct_to_relation(const IndexSpec& spec, AoSDeviceArray<T, Arity>& output_aos,
                               DeviceArray<semiring_value_t<SR>>& output_prov) const {
    std::size_t n = size();
    if (n == 0)
      return;

    T* out_ptr = output_aos.data();

    size_t block_size = 256;
    size_t num_blocks = (n + block_size - 1) / block_size;

    reconstruct_aos_kernel<NodeHandle>
        <<<num_blocks, block_size>>>(root(), out_ptr, output_prov.data());
  }

  // Build from encoded host (Host -> Device -> Index)
  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const T>, Arity>& encoded_rows) {
    size_t n = encoded_rows[0].size();
    if (n == 0)
      return;

    // 1. Interleave on host (simplest)
    std::vector<T> aos_host(n * Arity);
    for (size_t i = 0; i < n; ++i) {
      for (size_t c = 0; c < Arity; ++c) {
        aos_host[i * Arity + c] = encoded_rows[c][i];
      }
    }

    // 2. To Device
    AoSDeviceArray<T, Arity> aos_dev(n);  // Allocate
    cudaMemcpy(aos_dev.data(), aos_host.data(), n * Arity * sizeof(T), cudaMemcpyHostToDevice);

    // 3. Append
    append(aos_dev);
  }

  void clear() {
    impl_.clear();
    root_unique_values_.clear();
  }

  /// @brief Merge another index into this one
  /// @param other Source index to merge from (consumed - becomes empty after)
  /// @param row_id_offset Unused for TVJoin (provenance handled internally)
  /// @note This matches the DeviceSortedArrayIndex::merge() concept
  /// @brief Merge another index into this one
  /// @param other Source index to merge from (consumed - becomes empty after)
  /// @param row_id_offset Unused for TVJoin (provenance handled internally)
  /// @note This matches the DeviceSortedArrayIndex::merge() concept
  /// @brief Merge another index into this one
  /// @param other Source index to merge from (consumed - becomes empty after)
  /// @param row_id_offset Unused for TVJoin (provenance handled internally)
  /// @note This matches the DeviceSortedArrayIndex::merge() concept
  void merge(DeviceTVJoinIndex& other, [[maybe_unused]] std::size_t row_id_offset = 0) {
    if (other.empty()) {
      return;
    }
    // Delegate to adapter implementation which handles zero-copy move of internals
    impl_.merge(other.impl_);

    // Set-union of root unique values (both are already sorted and unique)
    if (other.root_unique_values_.empty()) {
      return;  // Nothing to union
    }
    if (root_unique_values_.empty()) {
      // Copy other's values (DeviceArray is non-copyable, so resize + copy)
      root_unique_values_.resize(other.root_unique_values_.size());
      thrust::copy(thrust::device, other.root_unique_values_.begin(),
                   other.root_unique_values_.end(), root_unique_values_.begin());
      return;
    }

    // Both have values - do sorted set-union
    DeviceArray<T> merged;
    merged.resize(root_unique_values_.size() + other.root_unique_values_.size());

    auto end_iter = thrust::set_union(thrust::device, root_unique_values_.begin(),
                                      root_unique_values_.end(), other.root_unique_values_.begin(),
                                      other.root_unique_values_.end(), merged.begin());

    std::size_t merged_size = end_iter - merged.begin();
    merged.resize(merged_size);
    root_unique_values_ = std::move(merged);
  }

  // --- Antijoin Probe Kernel for Set Difference ---
  // For each tuple in newt, probe full index to check existence
  template <typename Handle>
  __global__ static void antijoin_probe_kernel(Handle newt_root, Handle full_root,
                                               bool* exists_mask, std::size_t newt_size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= newt_size)
      return;

    const T* tuple = newt_root.get_tuple_at(idx);
    if (!tuple) {
      exists_mask[idx] = false;
      return;
    }

    // Probe full index by chaining prefix calls for each indexed column
    Handle h = full_root;
    for (std::size_t c = 0; c < Arity && h.valid(); ++c) {
      // Use index_cols to get the column order
      std::size_t col_idx = h.index_cols[c];
      h = h.prefix(tuple[col_idx]);
    }

    // If handle is still valid after all prefixes, tuple exists in full
    exists_mask[idx] = h.valid();
  }

  // --- Antijoin Probe Kernel for Raw AoS Data ---
  // Probes full index with raw AoS tuples (no index needed for newt)
  template <typename Handle>
  __global__ static void antijoin_probe_raw_kernel(const T* newt_data, Handle full_root,
                                                   bool* exists_mask, std::size_t newt_size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= newt_size)
      return;

    const T* tuple = newt_data + idx * Arity;

    // Probe full index by chaining prefix calls for each indexed column
    Handle h = full_root;
    for (std::size_t c = 0; c < Arity && h.valid(); ++c) {
      std::size_t col_idx = h.index_cols[c];
      h = h.prefix(tuple[col_idx]);
    }

    exists_mask[idx] = h.valid();
  }

  // --- Extract Tuple Kernel for Compaction ---
  template <typename Handle>
  __global__ static void extract_tuple_kernel(Handle root, std::size_t idx, T* output) {
    const T* tuple = root.get_tuple_at(idx);
    if (tuple) {
#pragma unroll
      for (std::size_t c = 0; c < Arity; ++c) {
        output[c] = tuple[c];
      }
    }
  }

  /// @brief Compute set difference: delta = this - full
  /// @param full_idx Full index to compare against
  /// @param delta_idx Output: truly new tuples (not in full)
  /// @note Uses TVJoin's native dedupSAsWithISA for efficient in-place filtering
  void set_difference_update(DeviceTVJoinIndex& full_idx, DeviceTVJoinIndex& delta_idx) {
    delta_idx.clear();

    if (empty()) {
      return;  // Nothing to diff
    }

    if (full_idx.empty()) {
      // If full is empty, all of newt is new -> move to delta
      std::swap(*this, delta_idx);
      return;
    }

    // Use TVJoin's native deduplication primitive
    // dedupSAsWithISA filters in-place by partitioning sorted_arrs
    auto& my_isa = impl_.isa();
    auto& full_isa = full_idx.impl().isa();

    // Collect sorted_arr pointers for deduplication
    std::deque<::Vector<T*, MemType::Device>*> tuple_arr_ptrs;
    for (auto& sa : my_isa.sorted_arrs) {
      if (sa.size() > 0) {
        tuple_arr_ptrs.push_back(&sa);
      }
    }

    if (tuple_arr_ptrs.empty()) {
      return;  // No data to deduplicate
    }

    // Dedup against full_idx's ISA (in-place filtering via stable_partition)
    std::size_t dedup_size = dedupSAsWithISA(&full_isa, tuple_arr_ptrs);

    if (dedup_size == 0) {
      clear();  // All tuples were duplicates
      return;
    }

    // Move filtered data to delta
    std::swap(*this, delta_idx);
    delta_idx.update_root_unique_values();
  }

  /// @brief Compute set difference from raw AoS data: delta = newt_data - full
  /// @param newt_data Raw AoS tuples from join output (not an index)
  /// @param full_idx Full index to compare against
  /// @param delta_idx Output: truly new tuples (not in full)
  /// @note Uses TVJoin's native Tuples and dedup primitives for in-place filtering
  static void set_difference_from_data(AoSDeviceArray<T, Arity>& newt_data,
                                       DeviceTVJoinIndex& full_idx, DeviceTVJoinIndex& delta_idx) {
    delta_idx.clear();
    std::size_t newt_size = newt_data.num_rows();
    if (newt_size == 0)
      return;

    if (full_idx.empty()) {
      delta_idx.append(newt_data);
      return;
    }

    // 1. Create TVJoin Tuples from AoSDeviceArray (zero-copy via swap)
    using Vector_t = ::Vector<T, MemType::Device>;
    Vector_t data_vec;
    newt_data.swap(data_vec);  // Zero-copy ownership transfer

    // Wrap in Tuples (creates pointer array)
    Tuples<T, MemType::Device> tuples(std::move(data_vec), Arity, newt_size);

    // 2. Sort the tuples (in-place on tuple_ptrs, data unchanged)
    sortTuples<T, MemType::Device>(tuples, std::index_sequence<IndexCols...>{});

    // 3. Self-dedup (in-place on tuple_ptrs)
    selfDedupTuples<T, MemType::Device, IndexCols...>(tuples);

    // 4. Dedup against full (in-place filtering via stable_partition on tuple_ptrs)
    std::size_t dedup_size = dedupTuplesWithISA(&full_idx.impl().isa(), tuples);

    if (dedup_size == 0)
      return;

    // 5. Move result into delta (zero-copy)
    MergeCleanTuples(&delta_idx.impl().isa(), std::move(tuples));
    delta_idx.update_root_unique_values();
  }

  /// @brief Compute set difference from raw AoS data with provenance tracking
  /// @param newt_data Raw AoS tuples from join output (not an index)
  /// @param newt_prov Provenance values corresponding to newt_data
  /// @param full_idx Full index to compare against
  /// @param delta_idx Output: truly new tuples (not in full)
  /// @param delta_prov Output: provenance for truly new tuples
  /// @note Uses provenance-aware TVJoin primitives for in-place filtering
  template <typename ProvT>
  static void set_difference_from_data_with_prov(AoSDeviceArray<T, Arity>& newt_data,
                                                 DeviceArray<ProvT>& newt_prov,
                                                 DeviceTVJoinIndex& full_idx,
                                                 DeviceTVJoinIndex& delta_idx,
                                                 DeviceArray<ProvT>& delta_prov) {

    delta_idx.clear();
    delta_prov.resize(0);
    std::size_t newt_size = newt_data.num_rows();

    if (newt_size == 0)
      return;

    if (full_idx.empty()) {
      delta_idx.append(newt_data);
      // Copy provenance
      delta_prov.resize(newt_size);
      thrust::copy(thrust::device, newt_prov.begin(), newt_prov.end(), delta_prov.begin());
      return;
    }

    // 1. Create TVJoin TuplesWithProvenance from AoSDeviceArray (zero-copy via swap)
    using Vector_t = ::Vector<T, MemType::Device>;
    using ProvVector_t = ::Vector<ProvT, MemType::Device>;

    Vector_t data_vec;
    newt_data.swap(data_vec);  // Zero-copy ownership transfer

    ProvVector_t prov_vec;
    newt_prov.swap(prov_vec);  // Zero-copy ownership transfer

    // Wrap in TuplesWithProvenance (creates pointer array)
    TuplesWithProvenance<T, MemType::Device, ProvT> tuples(std::move(data_vec), std::move(prov_vec),
                                                           Arity, newt_size);

    // 2. Sort the tuples with provenance (in-place on tuple_ptrs + prov_arr)
    sortTuplesWithProv<T, MemType::Device, ProvT>(tuples, std::index_sequence<IndexCols...>{});

    // 3. Self-dedup with provenance (in-place on tuple_ptrs + prov_arr)
    selfDedupTuplesWithProv<T, MemType::Device, ProvT, IndexCols...>(tuples);

    // 4. Dedup against full with provenance (in-place filtering via stable_partition)
    std::size_t dedup_size = dedupTuplesWithISAWithProv(&full_idx.impl().isa(), tuples);

    if (dedup_size == 0)
      return;

    // 5. Move result into delta (zero-copy for data, extract provenance)
    auto [merged_size, result_prov] =
        MergeCleanTuplesWithProv(&delta_idx.impl().isa(), std::move(tuples));
    delta_idx.update_root_unique_values();

    // 6. Transfer provenance to output
    delta_prov.resize(result_prov.size());
    thrust::copy(thrust::device, result_prov.begin(), result_prov.end(), delta_prov.begin());
  }

  // Access underlying impl for testing
  IndexImpl& impl() {
    return impl_;
  }
  const IndexImpl& impl() const {
    return impl_;
  }

  [[nodiscard]] std::size_t num_unique_root_values() const noexcept {
    return root_unique_values_.size();
  }

  [[nodiscard]] auto& root_unique_values() const {
    return root_unique_values_;
  }

 private:
  IndexImpl impl_;
  DeviceArray<T> root_unique_values_;
  std::size_t rows_processed_{0};  // For dirty-checking (matches DeviceSortedArrayIndex API)

  // Extract root key column from IndexCols... (First one)
  static constexpr std::size_t RootColIndex = []() {
    // Helper to strip first from pack
    constexpr std::size_t cols[] = {IndexCols...};
    return cols[0];
  }();

  // Kernel to extract just the root key
  __global__ static void extract_root_keys(NodeHandle root, T* out_keys) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < root.size()) {
      const T* tuple = root.get_tuple_at(idx);
      if (tuple) {
        out_keys[idx] = tuple[RootColIndex];
      }
    }
  }

  void update_root_unique_values() {
    size_t n = size();
    if (n == 0) {
      root_unique_values_.resize(0);
      return;
    }

    // 1. Resize/Allocate
    root_unique_values_.resize(n);

    // 2. Extract
    size_t block_size = 256;
    size_t num_blocks = (n + block_size - 1) / block_size;
    extract_root_keys<<<num_blocks, block_size>>>(root(), root_unique_values_.data());

    // 3. Sort (Only if multiple levels, otherwise it's partially sorted concatenation if we
    // iterate) Actually, extract_root_keys just dumps blindly. Concatenation of sorted arrays. If
    // num_levels <= 1, it IS sorted.
    if (root().num_levels > 1) {
      thrust::sort(thrust::device, root_unique_values_.begin(), root_unique_values_.end());
    }

    // 4. Unique
    auto new_end =
        thrust::unique(thrust::device, root_unique_values_.begin(), root_unique_values_.end());

    // 5. Shrink
    size_t new_size = new_end - root_unique_values_.begin();
    root_unique_values_.resize(new_size);
  }
};

}  // namespace SRDatalog::GPU
