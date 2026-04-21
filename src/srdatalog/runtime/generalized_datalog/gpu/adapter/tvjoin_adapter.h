#pragma once

#include "gpu/aos_device_array.h"
#include "gpu/gpu_api.h"
#include "join_skew/RA.cuh"
#include <boost/mp11.hpp>
#include <memory>
#include <rmm/device_vector.hpp>
#include <thrust/device_ptr.h>  // For casting if needed

namespace SRDatalog::GPU {

/**
 * @brief Iterator that transparently traverses multiple segments (LSM levels).
 *        Simplistic implementation for InputIterator.
 */
/**
 * @brief Iterator that transparently traverses multiple segments (LSM levels).
 *        Simplistic implementation for InputIterator.
 */
template <typename T>
struct SegmentedIterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = T*;
  using difference_type = std::ptrdiff_t;
  using pointer = T**;
  using reference = T*;

  const T* const* const* level_ptrs;  // shape: [num_levels][size_of_level]

  // Bounds for each level
  // We use fixed size array if Config::sorted_arr_levels is small, or pointers.
  // To keep it simple and portable, let's store bounds in the iterator directly if small enough,
  // or use a pointer to NodeHandle's data if possible.
  // Given iterator copy semantics, value semantics are best.
  // Let's assume max levels is small (e.g. < 32).
  static constexpr size_t MAX_LEVELS = Config::sorted_arr_levels;

  size_t starts[MAX_LEVELS];
  size_t ends[MAX_LEVELS];
  size_t num_levels;

  std::size_t current_level;
  std::size_t current_idx;

  __host__ __device__ SegmentedIterator(const T* const* const* ptrs, const size_t* start_indices,
                                        const size_t* end_indices, std::size_t n_levels)
      : level_ptrs(ptrs), num_levels(n_levels), current_level(0) {

    for (size_t i = 0; i < n_levels; ++i) {
      starts[i] = start_indices ? start_indices[i] : 0;
      ends[i] = end_indices ? end_indices[i] : 0;
    }
    // For "end" iterator, current_level will be num_levels
    current_idx = (num_levels > 0) ? starts[0] : 0;
    skip_empty();
  }

  // Constructor for creating an "End" iterator
  __host__ __device__ SegmentedIterator(size_t n_levels)
      : level_ptrs(nullptr), num_levels(n_levels), current_level(n_levels), current_idx(0) {
    for (size_t i = 0; i < n_levels; ++i) {
      starts[i] = 0;
      ends[i] = 0;
    }
  }

  __host__ __device__ void skip_empty() {
    while (current_level < num_levels && current_idx >= ends[current_level]) {
      current_level++;
      if (current_level < num_levels) {
        current_idx = starts[current_level];
      }
    }
  }

  __host__ __device__ reference operator*() const {
    return (T*)level_ptrs[current_level][current_idx];
  }

  __host__ __device__ SegmentedIterator& operator++() {
    current_idx++;
    skip_empty();
    return *this;
  }

  __host__ __device__ SegmentedIterator operator++(int) {
    SegmentedIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  __host__ __device__ bool operator==(const SegmentedIterator& other) const {
    // End iterator check: if both are at end (current_level == num_levels)
    if (current_level == num_levels && other.current_level == other.num_levels)
      return true;

    return current_level == other.current_level && current_idx == other.current_idx;
  }

  __host__ __device__ bool operator!=(const SegmentedIterator& other) const {
    return !(*this == other);
  }
};

/**
 * @brief Adapter for TVJoin's ISA (Indexed Sorted Array) to be used as a SRDatalog Index.
 */
template <typename T, std::size_t Arity, std::size_t... IndexCols>
class TVJoinIndex {
 public:
  using value_type = T;

  // TVJoin REQUIRES that the index sequence covers ALL columns (Arity).
  // SRDatalog requests an index on subset (IndexCols).
  // We must append the remaining columns to the index columns to create a "FullOrder".

  // 1. AllCols = {0, 1, ..., Arity-1}
  using AllCols = boost::mp11::mp_iota_c<Arity>;

  // 2. IndexColsList = {IndexCols...}
  using IndexColsList = boost::mp11::mp_list_c<std::size_t, IndexCols...>;

  // 3. Filter: Remove cols that are already in IndexColsList
  template <typename I>
  using IsInIndex = boost::mp11::mp_contains<IndexColsList, I>;

  using RemainingCols = boost::mp11::mp_remove_if<AllCols, IsInIndex>;

  // 4. Combined: IndexCols + RemainingCols
  using FullOrderList = boost::mp11::mp_append<IndexColsList, RemainingCols>;

  // 5. Instantiate ISA with FullOrder
  template <typename Seq>
  struct MakeISA;
  template <std::size_t... Is>
  struct MakeISA<boost::mp11::mp_list_c<std::size_t, Is...>> {
    using type = ISA<T, MemType::Device, Arity, Is...>;
  };

  using ISA_t = typename MakeISA<FullOrderList>::type;

  TVJoinIndex() = default;

  /// @brief Zero-copy merge from another TVJoinIndex
  /// @note Non-consuming: source's sorted_arrs remain valid (pointing to data now owned by this)
  void merge(TVJoinIndex& other) {
    if (other.isa_.size() == 0)
      return;

    using DataArrayType = typename ISA_t::DataArrayType;  // std::deque<Vector<T, M>>
    using SortedArrayTypeTuple =
        std::deque<typename ISA_t::sorted_arr_t>;  // std::deque<Vector<T*, M>>

    // 1. Move data arrays (ownership transfer)
    DataArrayType data_arrs;
    for (auto& da : other.isa_.data_arrs) {
      if (!da.empty()) {
        data_arrs.emplace_back(std::move(da));
      }
    }

    // 2. COPY sorted arrays (source remains valid)
    SortedArrayTypeTuple tuple_arrs;
    for (const auto& sa : other.isa_.sorted_arrs) {
      if (!sa.empty()) {
        tuple_arrs.push_back(sa);  // Copy, not move
      }
    }

    // 3. Merge into this ISA (non-consuming for sorted_arrs)
    MergeSAsIntoISA_KeepSource(&this->isa_, std::move(data_arrs), tuple_arrs);
  }

  void append(AoSDeviceArray<T, Arity>& data) {
    if (data.empty())
      return;

    // 1. Create Empty RMM vector (Target)
    using Vector_t = ::Vector<T, MemType::Device>;
    Vector_t data_vec;

    // 2. Swap ownership (Zero Copy)
    // AoSDeviceArray internal storage is now rmm::device_vector
    data.swap(data_vec);

    // 3. Wrap in TVJoin::Tuples
    // Constructor: Tuples(Vector<T, M>&& data_buffer, size_t col_size, size_t row_size)
    // data_vec size is num_rows * Arity.
    // logical row size is data_vec.size() / Arity.
    Tuples<T, MemType::Device> tuples(std::move(data_vec), Arity, data_vec.size() / Arity);

    // 4. Insert into ISA
    // insert() sorts, dedups, and merges.
    isa_.insert(std::move(tuples));
  }

  [[nodiscard]] std::size_t size() const {
    return isa_.size();
  }

  void clear() {
    // TVJoin's ISA doesn't have a clear() method exposed directly?
    // We might need to re-construct it or impl clear in ISA.
    // For now, let's reset it if possible.
    // Actually, looking at TVJoin code, we might just need to replace the object.
    // But since ISA holds device vectors, we should be careful.
    // Let's defer proper clear() implementation or use reassignment.
    isa_ = ISA_t{};
  }

  static constexpr std::size_t index_cols_array[] = {IndexCols...};

  struct NodeHandle {
    const T* const* const* sorted_arr_ptrs;
    const std::size_t* sorted_arr_sizes;
    std::size_t num_levels;
    std::size_t total_size;
    uint32_t depth;

    // explicitly store cols to avoid static member access issues on device
    uint8_t index_cols[Arity];

    // Constructor
    __host__ NodeHandle(const T* const* const* ptrs, const std::size_t* sizes, size_t n_levels,
                        size_t tot_sz, const std::size_t* host_sizes_init = nullptr, uint32_t d = 0)
        : sorted_arr_ptrs(ptrs), sorted_arr_sizes(sizes), num_levels(n_levels), total_size(tot_sz),
          depth(d) {

      // Init index cols from template param pack helper
      // We can't access parameter pack Is... here easily unless we capture them.
      // But we can access TVJoinIndex::index_cols_array?
      // Better: Hardcode initialization loop from static.
      // Or pass it in. But we want default copy.
      // Accessing static constexpr from host/device should be fine IF it's simple.
      // But let's copy it.
#pragma unroll
      for (size_t i = 0; i < Arity; ++i) {
        index_cols[i] = (uint8_t)index_cols_array[i];
      }

      for (size_t i = 0; i < SegmentedIterator<T>::MAX_LEVELS; ++i) {
        if (i < n_levels) {
          starts[i] = 0;
          if (host_sizes_init) {
            ends[i] = host_sizes_init[i];
          } else {
#ifdef __CUDA_ARCH__
            ends[i] = sorted_arr_sizes[i];
#else
            ends[i] = 0;
#endif
          }
        } else {
          starts[i] = 0;
          ends[i] = 0;
        }
      }
    }

    // View alias (self, since NodeHandle holds pointers)
    using View = NodeHandle;

    // Explicitly store bounds for each level
    size_t starts[SegmentedIterator<T>::MAX_LEVELS];
    size_t ends[SegmentedIterator<T>::MAX_LEVELS];

    // Helper to get full range
    __host__ __device__ auto begin() const {
      return SegmentedIterator<T>(sorted_arr_ptrs, starts, ends, num_levels);
    }

    __host__ __device__ auto end() const {
      return SegmentedIterator<T>(num_levels);
    }

    [[nodiscard]] __host__ __device__ std::size_t size() const {
      return total_size;
    }

    [[nodiscard]] __host__ __device__ bool valid() const {
      return total_size > 0;
    }

    // Degree: Sum of ranges
    [[nodiscard]] __host__ __device__ std::size_t degree() const {
      std::size_t count = 0;
      for (size_t i = 0; i < num_levels; ++i) {
        if (ends[i] > starts[i]) {
          count += (ends[i] - starts[i]);
        }
      }
      return count;
    }

    // Unified value access (Direct)
    // view is just self (or compatible NodeHandle)
    [[nodiscard]] __host__ __device__ T get_value_at(const View& view, std::size_t idx) const {
      // Find which level 'idx' falls into
      size_t current_offset = 0;
      for (size_t i = 0; i < num_levels; ++i) {
        size_t level_size = (ends[i] > starts[i]) ? (ends[i] - starts[i]) : 0;
        if (idx < current_offset + level_size) {
          size_t local_idx = starts[i] + (idx - current_offset);
          // Value at next column (depth points to column index for NEXT prefix, but here we want
          // the value AT THIS LEVEL's column) If depth=0, we are at root level, but keys are
          // col[0]. Wait, 'prefix' uses col[depth]. BUT 'get_value_at' implies iterating over the
          // children. If we have handle H from root.prefix(k), H represents rows where col[0]=k.
          // H.degree() is number of such rows.
          // H.get_value_at(j) should return col[1] for the j-th row.
          // H.depth should be 1.
          // So we use index_cols_array[depth].
          size_t col_idx = index_cols[depth];
          const T* tuple_ptr = (const T*)sorted_arr_ptrs[i][local_idx];
          return tuple_ptr[col_idx];
        }
        current_offset += level_size;
      }
      return T{};
    }

    // Unified provenance access (Placeholder for BooleanSR)
    [[nodiscard]] __host__ __device__ bool get_provenance_at(const View& view,
                                                             std::size_t idx) const {
      // TVJoin doesn't store explicit provenance yet.
      // Assume BooleanSR (existence = true).
      // Future: load from dedicated provenance array if added to TVJoin.
      return true;
    }

    // Get value of specific column at global idx (Handles SoA offset)
    [[nodiscard]] __host__ __device__ T get_column_at(std::size_t idx, std::size_t col_idx) const {
      size_t current_offset = 0;
      for (size_t i = 0; i < num_levels; ++i) {
        size_t level_size = (ends[i] > starts[i]) ? (ends[i] - starts[i]) : 0;
        if (idx < current_offset + level_size) {
          size_t local_idx = starts[i] + (idx - current_offset);
          // Pointer based SoA: sorted_arr_ptrs[i] points to Array of Pointers to keys?
          // If TVJoin sorts pointers:
          const T* key_ptr = (const T*)sorted_arr_ptrs[i][local_idx];
// key_ptr points to Col 0 element in data buffer.
// In SoA, Col C is at offset (C * SizeOfLevel).
// But `sorted_arr_sizes[i]` tracks SizeOfLevel (Rows).

// However, key_ptr is absolute address.
// We assume key_ptr points to `Buf + Offset`.
// Col C is at `key_ptr + (C * sorted_arr_sizes[i])`.
// Wait, this assumes `key_ptr` is `&Buf[k]`.
// `&Buf[k + Size]` is `key_ptr + Size`.
// Yes.

// Need to dereference.
#ifdef __CUDA_ARCH__
          // Use __ldg?
          // Or just standard.
          // For AoS, the tuple is contiguous. key_ptr points to START of tuple.
          // So key_ptr[col_idx] gives the value of that column.
          return key_ptr[col_idx];
#else
          return key_ptr[col_idx];
#endif
        }
        current_offset += level_size;
      }
      return T{};
    }

    // Get pointer to the full tuple at idx (Deprecated/Invalid for SoA usually, unless offset
    // known)
    [[nodiscard]] __host__ __device__ const T* get_tuple_at(std::size_t idx) const {
      // For SoA, returning a single pointer is ambiguous unless caller knows stride.
      // We return pointer to Col 0.
      size_t current_offset = 0;
      for (size_t i = 0; i < num_levels; ++i) {
        size_t level_size = (ends[i] > starts[i]) ? (ends[i] - starts[i]) : 0;
        if (idx < current_offset + level_size) {
          size_t local_idx = starts[i] + (idx - current_offset);
          return (const T*)sorted_arr_ptrs[i][local_idx];
        }
        current_offset += level_size;
      }
      return nullptr;
    }

    // Custom Binary Search Helper (Scalar Descent with __ldg)
    template <typename Key>
    __device__ size_t device_lower_bound_optimized(const T* const* level_base, size_t start_idx,
                                                   size_t end_idx, const Key& key) const {
      size_t left = start_idx;
      size_t right = end_idx;
      // Use depth to pick column
      size_t col_idx = index_cols[depth];

      while (left < right) {
        size_t mid = left + (right - left) / 2;
        const T* tuple_ptr = (const T*)__ldg((const uintptr_t*)&level_base[mid]);
        auto val = tuple_ptr[col_idx];

        if (val < key) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      return left;
    }

    // Custom Upper Bound Helper
    template <typename Key>
    __device__ size_t device_upper_bound_optimized(const T* const* level_base, size_t start_idx,
                                                   size_t end_idx, const Key& key) const {
      size_t left = start_idx;
      size_t right = end_idx;
      size_t col_idx = index_cols[depth];

      while (left < right) {
        size_t mid = left + (right - left) / 2;
        const T* tuple_ptr = (const T*)__ldg((const uintptr_t*)&level_base[mid]);
        auto val = tuple_ptr[col_idx];

        if (key < val) {
          right = mid;
        } else {
          left = mid + 1;
        }
      }
      return left;
    }

    // Hybrid Cooperative Search
    template <typename Key, typename Group>
    __device__ size_t device_lower_bound_hybrid(const T* const* level_base, size_t start_idx,
                                                size_t end_idx, const Key& key, Group& g) const {
      size_t left = start_idx;
      size_t right = end_idx;
      size_t col_idx = index_cols[depth];

      while (right - left > 32) {
        size_t mid = left + (right - left) / 2;
        const T* tuple_ptr = (const T*)__ldg((const uintptr_t*)&level_base[mid]);
        if (tuple_ptr[col_idx] < key) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }

      size_t count = right - left;
      size_t lane = g.thread_rank();
      bool active = lane < count;
      bool pred = false;

      if (active) {
        const T* ptr = level_base[left + lane];
        pred = ptr[col_idx] < key;
      }

      // 2. Ballot and Popcount to find exact position
      // mask bits are 1 if (val < key)
      // We want the count of such elements to determine 'lower_bound' index.
      // Use g.ballot() to support arbitrary group sizes (e.g. half-warp)
      unsigned mask = g.ballot(active && pred);

      // If group < 32, we need correct mask handling.
      // Assuming Warp tile for now as per "Warp-Compatible".
      // Mask contains 1s for all items < key.
      // Since list is sorted, 1s are contiguous at LSB.
      // Total count of 1s = index offset.
      int offset = __popc(mask);

      return left + offset;
    }

    // Hybrid Upper Bound
    template <typename Key, typename Group>
    __device__ size_t device_upper_bound_hybrid(const T* const* level_base, size_t start_idx,
                                                size_t end_idx, const Key& key, Group& g) const {
      size_t left = start_idx;
      size_t right = end_idx;
      size_t col_idx = index_cols[depth];

      while (right - left > 32) {
        size_t mid = left + (right - left) / 2;
        const T* tuple_ptr = (const T*)__ldg((const uintptr_t*)&level_base[mid]);
        if (key < tuple_ptr[col_idx]) {
          right = mid;
        } else {
          left = mid + 1;
        }
      }

      size_t count = right - left;
      size_t lane = g.thread_rank();
      bool active = lane < count;
      bool pred = false;

      if (active) {
        const T* ptr = level_base[left + lane];
        // upper_bound: Logic is !(key < val) -> val <= key is "less portion"
        // actually standard upper_bound returns > key.
        // So we count items where !(key < val) i.e. val <= key
        pred = !(key < ptr[col_idx]);
      }

      unsigned mask = g.ballot(active && pred);
      int offset = __popc(mask);
      return left + offset;
    }

    // Cooperative Prefix Overload
    template <typename Key, typename Group, typename View>
    __device__ NodeHandle prefix(const Key& key, Group& g, const View& v) const {
      // If group is too small to handle the 32-element tail of hybrid search,
      // fallback to scalar optimized search.
      if (g.size() < 32) {
        return prefix(key);
      }
      NodeHandle sub_handle = *this;
      sub_handle.depth = depth + 1;  // Advance depth
      size_t new_total_size = 0;

      for (size_t i = 0; i < num_levels; ++i) {
#ifdef __CUDA_ARCH__
        sub_handle.starts[i] =
            device_lower_bound_hybrid(sorted_arr_ptrs[i], starts[i], ends[i], key, g);
        sub_handle.ends[i] =
            device_upper_bound_hybrid(sorted_arr_ptrs[i], sub_handle.starts[i], ends[i], key, g);
#endif
        new_total_size += (sub_handle.ends[i] - sub_handle.starts[i]);
      }
      sub_handle.total_size = new_total_size;
      return sub_handle;
    }

    // Standard Prefix Search
    template <typename Key>
    __host__ __device__ NodeHandle prefix(const Key& key) const {
      NodeHandle sub_handle = *this;
      sub_handle.depth = depth + 1;  // Advance depth

      // Get column index for host fallback
      size_t col_idx = index_cols[depth];

      size_t new_total_size = 0;

      for (size_t i = 0; i < num_levels; ++i) {
        // Device: Use optimized custom binary search (uses depth implicitly)
        sub_handle.starts[i] =
            device_lower_bound_optimized(sorted_arr_ptrs[i], starts[i], ends[i], key);
        sub_handle.ends[i] =
            device_upper_bound_optimized(sorted_arr_ptrs[i], sub_handle.starts[i], ends[i], key);

        new_total_size += (sub_handle.ends[i] - sub_handle.starts[i]);
      }
      sub_handle.total_size = new_total_size;
      return sub_handle;
    }

    // Standard lower_bound (iterator to first element >= key)
    template <typename Key>
    __host__ __device__ auto lower_bound(const Key& key) const {
      // Instead of wrapping prefix(), which narrows both start and end,
      // we only update starts to lower_bound(key) and keep original ends.
      NodeHandle lb_handle = *this;

      // Get column index for host fallback
      size_t col_idx = index_cols[depth];

      for (size_t i = 0; i < num_levels; ++i) {
        lb_handle.starts[i] =
            device_lower_bound_optimized(sorted_arr_ptrs[i], starts[i], ends[i], key);
        // ends[i] remains unchanged from *this, allowing iteration to continue
      }
      return lb_handle.begin();
    }
  };

  NodeHandle get_node_handle() const {
    // Ensure specific device pointers are accessible
    // ISA maintains dev_sorted_arr_ptrs and dev_sa_sizes

    // Copy sizes to host to initialize the root NodeHandle
    std::vector<size_t> host_sizes(isa_.dev_sa_sizes.size());
    if (!host_sizes.empty()) {
      cudaMemcpy(host_sizes.data(), thrust::raw_pointer_cast(isa_.dev_sa_sizes.data()),
                 sizeof(size_t) * host_sizes.size(), cudaMemcpyDeviceToHost);
    }

    return NodeHandle(thrust::raw_pointer_cast(isa_.dev_sorted_arr_ptrs.data()),
                      thrust::raw_pointer_cast(isa_.dev_sa_sizes.data()),
                      isa_.dev_sorted_arr_ptrs.size(), isa_.size(), host_sizes.data());
  }

  /// @brief Access underlying ISA for TVJoin primitives
  ISA_t& isa() {
    return isa_;
  }
  const ISA_t& isa() const {
    return isa_;
  }

 protected:
  ISA_t isa_;
};

}  // namespace SRDatalog::GPU
