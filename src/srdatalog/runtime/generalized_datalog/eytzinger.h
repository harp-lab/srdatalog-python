#pragma once
// For reference: https://algorithmica.org/en/b-tree
// The core idea is search on SA is cache-oblivious, if search is a very hot path, (in Datalog, join
// search call is hot path, though, might not be most expensive operation in Datalog), we pay
// cache miss price once when building index (use BFS order). We trade off building cache miss price
// for search time. This is also reason why entire SA can't be stored using this index, this must be
// paired with vectorization processing, where functionally its a Btree, where each node is vector
// can fits into L1/L2 cache, so building index is not that expensive (theoratically divided by
// vectorization size, extra overhead + cache miss penalty to just sorting).

#include "system.h"  // Vector
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <tuple>

#include "column.h"
#include "index.h"  // Your Concept Definitions
#include "logging.h"
#include "search.h"  // SRDatalog::search
#include "semiring.h"
#include "skip.h"  // DedupRange
#include "sort.h"  // hwy_lexsort

namespace SRDatalog {

// --- 1. Internal Helper: Eytzinger Index (L3 Cache Resident) ---
// This manages the "Block Separators". It is cache-oblivious.
template <typename T>
class EytzingerMap {
 public:
  Vector<T> data;
  Vector<size_t> heap_to_block;  // Mapping from Eytzinger heap index to original block index

  void build(const Vector<T>& sorted_separators) {
    size_t n = sorted_separators.size();
    data.resize(n);
    heap_to_block.resize(n);
    if (n == 0)
      return;
    size_t idx = 0;
    size_t block_idx = 0;
    build_recursive(sorted_separators, idx, block_idx, 1);
  }

  // Returns the Block ID (0-based) - the original block index, not the heap index
  [[nodiscard]] size_t search(T key) const noexcept {
    if (data.empty()) [[unlikely]]
      return 0;

    size_t k = 1;
    size_t n = data.size();

    // HPC Optimization: Prefetch future tree levels
    // 64 bytes (cache line) / sizeof(T) gives the lookahead distance
    constexpr size_t k_prefetch_dist = 64 / sizeof(T);

    while (k <= n) {
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(&data[(k * k_prefetch_dist) - 1]);
#endif
      // Branchless descent
      k = 2 * k + (data[k - 1] < key);
    }

    // C++20 Decode: Backtrack to find the lower bound leaf
    k >>= (std::countr_one(k) + 1);
    size_t heap_idx = k - 1;

    // Map heap index to original block index
    if (heap_idx < heap_to_block.size()) {
      return heap_to_block[heap_idx];
    }
    return heap_idx;  // Fallback if mapping is invalid
  }

  void clear() {
    data.clear();
    heap_to_block.clear();
  }

 private:
  void build_recursive(const Vector<T>& src, size_t& i, size_t& block_idx, size_t k) {
    if (k > src.size())
      return;
    build_recursive(src, i, block_idx, 2 * k);
    size_t heap_pos = k - 1;
    data[heap_pos] = src[i];
    heap_to_block[heap_pos] = block_idx;  // Store mapping: heap position -> block index
    i++;
    block_idx++;
    build_recursive(src, i, block_idx, 2 * k + 1);
  }
};

// --- 2. Accelerated Iterator and Range View (for use with Eytzinger index) ---
template <typename T, std::size_t BlockSize = 64>
class AcceleratedArrayIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;

  AcceleratedArrayIterator() = default;

  // Constructor
  AcceleratedArrayIterator(const T* base, const T* ptr, const T* end, const EytzingerMap<T>* idx)
      : base_(base), ptr_(ptr), end_(end), idx_(idx) {}

  reference operator*() const {
    return *ptr_;
  }
  pointer operator->() const {
    return ptr_;
  }

  // 1. Advance (Linear)
  // Use the exact same logic as SortedArrayIndex to ensure consistency
  AcceleratedArrayIterator& operator++() {
    if (ptr_ < end_) {
      // Use scalar loop or your trusted skip_duplicates
      ptr_ = SRDatalog::skip_duplicates(ptr_, end_);
    }
    return *this;
  }

  AcceleratedArrayIterator operator++(int) {
    AcceleratedArrayIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  void seek(const T& target) {
    if (ptr_ >= end_)
      return;
    if (*ptr_ >= target)
      return;

    // 1. Fallback (No Index)
    if (idx_ == nullptr || idx_->data.empty()) {
      size_t len = end_ - ptr_;
      size_t offset = SRDatalog::search::adaptive_lower_bound(ptr_, len, target);
      ptr_ += offset;
      return;
    }

    // 2. Eytzinger Lookup
    size_t block_idx = idx_->search(target);
    if (block_idx >= idx_->data.size())
      block_idx = idx_->data.size();

    // Safety: Backtrack one block to handle boundary duplicates
    if (block_idx > 0)
      block_idx--;

    // 3. Calculate Jump
    const T* block_start_ptr = base_ + (block_idx * BlockSize);

    // Forward Only Rule
    const T* search_start = (block_start_ptr > ptr_) ? block_start_ptr : ptr_;

    if (search_start >= end_) {
      ptr_ = end_;
      return;
    }

    // 4. THE CRITICAL FIX: Window Clamping
    // We know the target is in Block[k] or Block[k+1] (due to backtrack).
    // We do NOT need to search to 'end_'.
    // Clamp length to 256 items (4 blocks). This guarantees SIMD usage
    // (assuming adaptive_lower_bound threshold is > 64).

    size_t dist_to_end = end_ - search_start;
    size_t scan_len = std::min(dist_to_end, size_t{256});

    // asm volatile("nop; nop; nop; nop; nop");
    // This will now use SIMD Linear Scan because 256 is "small enough"
    // (or close enough to the threshold) to avoid the overhead of Binary Search setup.
    // If your adaptive threshold is strictly 128, change 256 to 128.
    size_t offset = SRDatalog::search::adaptive_lower_bound(search_start, scan_len, target);

    ptr_ = search_start + offset;

    // 5. Paranoia Safety / Soft Fallback
    // If we scanned the 256-item window and didn't find the target (hit the end of window),
    // AND we haven't hit the true end, it means Eytzinger was wrong (unlikely)
    // or duplicates are extremely long. We resume searching from here.
    if (offset == scan_len && ptr_ < end_ && *ptr_ < target) {
      size_t remaining = end_ - ptr_;
      ptr_ += SRDatalog::search::adaptive_lower_bound(ptr_, remaining, target);
    }
  }

  bool operator==(const AcceleratedArrayIterator& other) const {
    return ptr_ == other.ptr_;
  }
  bool operator!=(const AcceleratedArrayIterator& other) const {
    return ptr_ != other.ptr_;
  }

 private:
  const T* base_{nullptr};
  const T* ptr_{nullptr};
  const T* end_{nullptr};
  const EytzingerMap<T>* idx_{nullptr};
};

/**
 * @brief A range view that provides accelerated iteration over deduplicated values.
 * @details Similar to DedupRange but uses AcceleratedArrayIterator which leverages
 *          Eytzinger index for faster seeks in large sorted arrays.
 * @tparam T The value type
 * @tparam BlockSize The block size for Eytzinger indexing (default: 64)
 */
template <typename T, std::size_t BlockSize = 64>
struct AcceleratedDedupRange {
  const T* begin_ptr;
  const T* end_ptr;
  const T* base_ptr;                     // Base pointer for Eytzinger offset calculation
  const EytzingerMap<T>* eytzinger_idx;  // Pointer to Eytzinger index (can be nullptr)

  using Iterator = AcceleratedArrayIterator<T, BlockSize>;

  [[nodiscard]] Iterator begin() const {
    if (begin_ptr >= end_ptr) {
      return Iterator(base_ptr, end_ptr, end_ptr, eytzinger_idx);
    }
    return Iterator(base_ptr, begin_ptr, end_ptr, eytzinger_idx);
  }

  [[nodiscard]] Iterator end() const {
    return Iterator(base_ptr, end_ptr, end_ptr, eytzinger_idx);
  }
};

// --- 3. The Main Implicit B-Tree Index ---

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType = uint32_t,
          typename RowIdType = uint32_t>
class ImplicitBTreeIndex {
  struct Impl;

 public:
  using RowId = RowIdType;
  using RowRange = std::span<const RowId>;
  using ValueRange = std::span<ValueType>;
  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  // BlockSize aligned to cache lines (64 elements * 4 bytes = 256 bytes = 4 cache lines)
  // Matches AVX2/AVX-512 throughput sweet spots.
  static constexpr std::size_t BlockSize = 64;

  // --- NodeHandle Implementation ---
  class NodeHandle {
   public:
    NodeHandle() = default;

    [[nodiscard]] bool valid() const noexcept {
      return impl_ != nullptr && begin_ < end_;
    }
    [[nodiscard]] bool is_leaf() const noexcept {
      return valid() && depth_ == impl_->index_arity;
    }
    [[nodiscard]] uint32_t depth() const noexcept {
      return depth_;
    }
    [[nodiscard]] std::size_t degree() const noexcept {
      return end_ - begin_;
    }

    // Returns deduplicated view of values with Eytzinger acceleration when available.
    [[nodiscard]] auto values() const {
      if (!valid() || is_leaf()) {
        // Return empty view
        return SRDatalog::AcceleratedDedupRange<ValueType, BlockSize>{nullptr, nullptr, nullptr,
                                                                      nullptr};
      }

      const auto& col = impl_->cols[depth_];

      // CORRECT: Absolute start of the vector.
      // The Eytzinger offsets (0, 64, 128...) are calculated relative to THIS address.
      const ValueType* base = col.data();

      // Relative pointers for the range
      const ValueType* current = base + begin_;
      const ValueType* end_ptr = base + end_;

      // Index is only valid for the Root Level
      const EytzingerMap<ValueType>* acc =
          (depth_ == 0 && !impl_->eytzinger_idx.data.empty()) ? &impl_->eytzinger_idx : nullptr;

      return SRDatalog::AcceleratedDedupRange<ValueType, BlockSize>{current, end_ptr, base, acc};
    }

    [[nodiscard]] RowRange rows() const {
      if (!is_leaf())
        return {};
      return RowRange(impl_->postings.data() + begin_, end_ - begin_);
    }

    // --- The High-Performance Seek ---
    [[nodiscard]] NodeHandle prefix(ValueType key) const;

    template <std::size_t N>
    NodeHandle prefix(const EncodedKeyPrefix<N>& suffix) const {
      NodeHandle cur = *this;
      for (size_t i = 0; i < N; ++i) {
        cur = cur.prefix(static_cast<ValueType>(suffix[i]));
        if (!cur.valid())
          break;
      }
      return cur;
    }

    [[nodiscard]] bool contains_value(ValueType key) const noexcept {
      if (!valid() || is_leaf())
        return false;
      const auto& col = impl_->cols[depth_];
      const ValueType* ptr = col.data();

      size_t idx = begin_;
      size_t len = end_ - begin_;

      if (depth_ == 0 && !impl_->eytzinger_idx.data.empty()) {
        size_t block = impl_->eytzinger_idx.search(key);
        size_t block_start = block * BlockSize;
        if (block_start >= end_)
          return false;
        if (block_start > begin_) {
          idx = block_start;
          len = end_ - idx;
        }
      }

      size_t found = SRDatalog::search::adaptive_lower_bound(ptr + idx, len, key);
      return (idx + found < end_) && (ptr[idx + found] == key);
    }

    // Required by concept: Count unique values
    [[nodiscard]] std::size_t unique_count() const noexcept {
      if (!valid() || is_leaf())
        return 0;
      // values() returns DedupRange which already deduplicates, so just count elements
      auto v = values();
      size_t count = 0;
      for ([[maybe_unused]] auto val : v) {
        count++;
      }
      return count;
    }

   private:
    friend class ImplicitBTreeIndex;
    const Impl* impl_{nullptr};
    std::size_t begin_{0};
    std::size_t end_{0};
    uint32_t depth_{0};

    NodeHandle(const Impl* impl, std::size_t b, std::size_t e, uint32_t d)
        : impl_(impl), begin_(b), end_(e), depth_(d) {}
  };

  // --- Constructor / Destructor ---
  explicit ImplicitBTreeIndex(memory_resource* resource = nullptr)
      : p_(std::make_unique<Impl>((resource != nullptr) ? resource : default_memory_resource())) {}

  ~ImplicitBTreeIndex() = default;
  ImplicitBTreeIndex(ImplicitBTreeIndex&&) noexcept = default;
  ImplicitBTreeIndex& operator=(ImplicitBTreeIndex&&) noexcept = default;

  // --- Interface Methods (IndexLike) ---

  [[nodiscard]] NodeHandle root() const {
    if (empty())
      return {};
    return NodeHandle(p_.get(), 0, p_->size, 0);
  }

  [[nodiscard]] std::size_t bytes_used() const {
    if (!p_ || p_->size == 0)
      return 0;
    size_t b = sizeof(Impl);
    for (const auto& c : p_->cols)
      b += c.capacity() * sizeof(ValueType);
    b += p_->postings.capacity() * sizeof(RowId);
    b += p_->eytzinger_idx.data.capacity() * sizeof(ValueType);
    return b;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return p_ ? p_->size : 0;
  }
  [[nodiscard]] bool empty() const noexcept {
    return !p_ || p_->size == 0;
  }
  /// @brief Get the number of rows processed before deduplication
  /// @details This tracks how many rows from the relation have been indexed,
  ///          before deduplication. Used to determine if index needs rebuilding.
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return p_ ? p_->rows_processed_ : 0;
  }

  void clear() noexcept {
    p_->cols.clear();
    p_->postings.clear();
    p_->eytzinger_idx.data.clear();
    p_->size = 0;
    p_->rows_processed_ = 0;
  }

  template <std::size_t N>
  [[nodiscard]] NodeHandle prefix_lookup(const EncodedKeyPrefix<N>& k) const {
    return root().template prefix<N>(k);
  }

  // --- Build Method (using hwy_lexsort) ---
  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const ValueType>, arity>& encoded_rows,
                          memory_resource* resource = nullptr);

  // Deep clone
  void clone_from(const ImplicitBTreeIndex& other, memory_resource* resource = nullptr);

  // --- Merge Method ---
  void merge(const ImplicitBTreeIndex& other, std::size_t row_id_offset);

  /**
   * @brief LFTJ Intersection (galloping intersection, "seek").
   * @details Returns a View that scans the first handle and probes the rest.
   *          Uses LeapfrogView for efficient intersection of sorted ranges.
   */
  template <typename FirstHandle, typename... OtherHandles>
  static auto intersect(const FirstHandle& first, const OtherHandles&... others);

 private:
  struct Impl {
    using VecSizeT = Vector<ValueType>;
    using VecRowId = Vector<RowIdType>;

    explicit Impl(memory_resource* r) : resource(r), cols(r), postings(r) {}
    memory_resource* resource;

    // Structure of Arrays (Columns)
    // Match sorted_array_index container type for compatibility with soa_inplace_merge
    Vector<VecSizeT> cols;
    VecRowId postings;

    // Implicit Index (Acceleration for Col 0)
    EytzingerMap<ValueType> eytzinger_idx;

    size_t size{0};
    std::size_t index_arity{0};      // Number of indexed columns (spec.cols.size())
    std::size_t rows_processed_{0};  // Number of rows processed (before deduplication)

    void clear() {
      cols.clear();
      postings.clear();
      eytzinger_idx.clear();
      size = 0;
      rows_processed_ = 0;
    }
  };

  std::unique_ptr<Impl> p_;
};

}  // namespace SRDatalog

#include "eytzinger.ipp"

// --- 3. Static Assertions to Verify "Plugin" Capability ---
// Note: These must come after including eytzinger.ipp so the intersect implementation is visible

namespace SRDatalog {

// Verify that ImplicitBTreeIndex satisfies IndexLike
static_assert(IndexLike<ImplicitBTreeIndex<BooleanSR, std::tuple<int, int>>>,
              "ImplicitBTreeIndex must satisfy IndexLike concept");

// Verify that NodeHandle satisfies CNodeHandle
static_assert(CNodeHandle<typename ImplicitBTreeIndex<BooleanSR, std::tuple<int, int>>::NodeHandle>,
              "ImplicitBTreeIndex::NodeHandle must satisfy CNodeHandle concept");

}  // namespace SRDatalog
