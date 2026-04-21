/**
 * @file hashmap.h
 * @brief A pure hashmap based index datastructure. For WCOJ, we will vectical partition the hash
 * table, for each prefix in relation we will create a hashmap contains value to next level node.
 */

#pragma once

#include "column.h"
#include "index.h"
#include "semiring.h"
#include "system.h"
#include "tmp.h"
#include <array>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>

namespace SRDatalog::index {

// Template metaprogramming helper to generate vertical partition structure
namespace detail {
namespace mp = boost::mp11;

// Helper: Create a tuple type with N elements of type T
template <std::size_t N, typename T>
struct TupleRepeat {
  using type = mp::mp_rename<mp::mp_repeat_c<mp::mp_list<T>, N>, std::tuple>;
};

template <std::size_t N, typename T>
using TupleRepeat_t = typename TupleRepeat<N, T>::type;

// Helper: Generate map type for layer L (0-indexed)
// Layer 0: Map<std::tuple<>, Set<ValueType>> - maps empty tuple to all possible first values
// Layer 1: Map<std::tuple<ValueType>, Set<ValueType>> - maps (v0) to all possible second values
// Layer L: Map<std::tuple<ValueType, ... (L times)>, Set<ValueType>> - maps prefix of L values to
// next values Leaf: Map<std::tuple<ValueType, ... (arity times)>, Set<RowId>> - maps full prefix to
// set of row IDs
template <std::size_t Layer, typename ValueType, typename RowId, typename IndexType,
          std::size_t Arity>
struct LayerMapType;

// Non-leaf layers: map prefix tuple to set of next values (ensures uniqueness)
// Layer 0: uses std::tuple<> (empty tuple) as key
// Layer L: uses std::tuple<ValueType, ... (L times)> as key
// All layers 0 to Arity-1 are non-leaf (path goes through all columns)
template <std::size_t Layer, typename ValueType, typename RowId, typename IndexType,
          std::size_t Arity>
  requires(Layer < Arity)
struct LayerMapType<Layer, ValueType, RowId, IndexType, Arity> {
  using KeyType =
      TupleRepeat_t<Layer, ValueType>;  // Tuple of Layer ValueTypes (empty tuple for Layer 0)
  using ValueSetType = Set<IndexType>;  // Set ensures uniqueness of values
  using type = Map<KeyType, ValueSetType>;
};

// Leaf layer: map full prefix tuple to set of row IDs (separate layer after all columns)
// Leaf is at layer Arity (after all non-leaf layers 0 to Arity-1)
template <std::size_t Layer, typename ValueType, typename RowId, typename IndexType,
          std::size_t Arity>
  requires(Layer == Arity)
struct LayerMapType<Layer, ValueType, RowId, IndexType, Arity> {
  using KeyType = TupleRepeat_t<Arity, ValueType>;  // Tuple of Arity ValueTypes
  using ValueSetType = RowId;  // Just store one RowId per unique key (memory efficient)
  using type = Map<KeyType, ValueSetType>;
};

template <std::size_t Layer, typename ValueType, typename RowId, typename IndexType,
          std::size_t Arity>
using LayerMapType_t = typename LayerMapType<Layer, ValueType, RowId, IndexType, Arity>::type;

// Generate tuple of map types for vertical partitioning
// For arity N, we need N layers:
//   - Layer 0: Map<ValueType, Vector<ValueType>>
//   - Layer 1: Map<std::tuple<ValueType, ValueType>, Vector<ValueType>>
//   - ...
//   - Layer N-1: Map<std::tuple<ValueType, ... (N times)>, RowId>
template <std::size_t Arity, typename ValueType, typename RowId, typename IndexType = ValueType>
struct VerticalPartitionTypes;

// Base case: arity 0 (empty tuple)
template <typename ValueType, typename RowId, typename IndexType>
struct VerticalPartitionTypes<0, ValueType, RowId, IndexType> {
  using type = std::tuple<>;
};

// Generate tuple with one map per layer
// For arity N, we have N+1 layers: layers 0 to N-1 are non-leaf, layer N is leaf
template <std::size_t Arity, typename ValueType, typename RowId, typename IndexType>
struct VerticalPartitionTypes {
  template <std::size_t... Is>
  static auto make_map_tuple(std::index_sequence<Is...>) {
    return std::tuple<LayerMapType_t<Is, ValueType, RowId, IndexType, Arity>...>{};
  }
  // Create arity+1 layers: 0 to arity-1 (non-leaf) + arity (leaf)
  using type = decltype(make_map_tuple(std::make_index_sequence<Arity + 1>{}));
};

// Special case: arity 1 (layer 0 non-leaf + layer 1 leaf)
template <typename ValueType, typename RowId, typename IndexType>
struct VerticalPartitionTypes<1, ValueType, RowId, IndexType> {
  using type = std::tuple<LayerMapType_t<0, ValueType, RowId, IndexType, 1>,
                          LayerMapType_t<1, ValueType, RowId, IndexType, 1>>;
};
}  // namespace detail

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType = uint32_t,
          typename RowIdType = uint32_t, typename IndexType = ValueType>
class HashmapIndex {
  struct Impl;

 public:
  // Expose template parameters as type aliases for relation_col.ipp
  template <typename T>
  struct value_type_helper {
    using type = T;
  };
  using ValueTypeAlias = typename value_type_helper<ValueType>::type;
  using RowId = RowIdType;
  using RowRange = std::span<const RowId>;
  using Index = IndexType;
  // ValueRange: simple iterator range wrapper (no copying, no ranges)
  // Stores a pointer to the set to keep iterators valid
  struct ValueRange {
    using value_type = IndexType;
    using iterator = typename Set<IndexType>::const_iterator;

    struct type {
      const Set<IndexType>* set_ptr_;  // Keep reference to set to keep iterators valid

      type(const Set<IndexType>* ptr) : set_ptr_(ptr) {}

      iterator begin() const {
        return set_ptr_ ? set_ptr_->begin() : iterator{};
      }
      iterator end() const {
        return set_ptr_ ? set_ptr_->end() : iterator{};
      }
    };
  };
  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  /**
   * @brief Cursor for navigating and querying nodes in a HashmapIndex.
   * @details NodeHandle represents a position at a specific depth/layer in the vertical partition.
   *          It tracks the prefix path and provides navigation methods.
   */
  class NodeHandle {
   public:
    /// @brief Default constructor creates an invalid handle.
    NodeHandle() = default;

    /// @brief Check if this handle points to a valid node.
    [[nodiscard]] bool valid() const noexcept;

    /// @brief Check if this node is a leaf node.
    [[nodiscard]] bool is_leaf() const noexcept;

    /// @brief Get the depth of this node in the index.
    [[nodiscard]] uint32_t depth() const noexcept;

    /// @brief Enumerate encoded keys for the next level (internal nodes only).
    /// @return Lazy view that filters rows matching prefix and extracts values at current depth
    [[nodiscard]] typename ValueRange::type values() const;

    /// @brief Get row IDs matching the full key prefix (leaf nodes only).
    [[nodiscard]] RowRange rows() const;

    /// @brief Descend one level using an encoded key.
    NodeHandle prefix(ValueType next_key) const;

    /// @brief Descend multiple levels using an encoded key prefix.
    template <std::size_t N>
    NodeHandle prefix(const EncodedKeyPrefix<N>& suffix) const;

    /// @brief Check if a specific encoded key exists at the next level.
    [[nodiscard]] bool contains_value(ValueType key) const noexcept;

    /// @brief Get the number of children (out-degree) of this node.
    [[nodiscard]] std::size_t degree() const noexcept;

    /// @brief Get the count of unique values (size of values() range).
    [[nodiscard]] std::size_t unique_count() const noexcept;

   private:
    template <Semiring SR2, ColumnElementTuple AttrTuple2, typename ValueType2, typename RowIdType2,
              typename IndexType2>
    friend class HashmapIndex;
    NodeHandle(Impl* impl, uint32_t depth, const std::array<ValueType, arity>& prefix_path) noexcept
        : impl_(impl), depth_(depth), prefix_path_(prefix_path) {}

    Impl* impl_{nullptr};
    uint32_t depth_{0};
    std::array<ValueType, arity> prefix_path_{};
  };
  // Note: Concept check moved to end of file after all methods are defined
  // static_assert(CNodeHandle<NodeHandle>, "NodeHandle must satisfy CNodeHandle concept");

  /// @brief Construct an empty HashmapIndex.
  explicit HashmapIndex(memory_resource* resource);

  /// @brief Destructor.
  ~HashmapIndex();

  // Non-copyable
  HashmapIndex(const HashmapIndex&) = delete;
  HashmapIndex& operator=(const HashmapIndex&) = delete;

  // Movable
  HashmapIndex(HashmapIndex&&) noexcept = default;
  HashmapIndex& operator=(HashmapIndex&&) noexcept = default;

  /// @brief Build the index from pre-encoded row data.
  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const ValueType>, arity>& encoded_rows,
                          memory_resource* resource = nullptr);

  /// @brief Get a cursor to the root node of the index.
  NodeHandle root() const;

  /// @brief Look up a node by encoded key prefix.
  template <std::size_t N>
  NodeHandle prefix_lookup(const EncodedKeyPrefix<N>& k) const;

  /// @brief Get approximate memory footprint of the index.
  [[nodiscard]] std::size_t bytes_used() const;

  /// @brief Get the number of rows indexed.
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  /// @brief Get the number of rows processed (before deduplication).
  /// @details For HashmapIndex, this tracks the number of rows processed, not unique rows stored.
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return p_ ? p_->rows_processed_ : 0;
  }

  /// @brief Clear the index.
  void clear() noexcept;

  /// @brief Check if the index is empty.
  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  /// @brief Merge another HashmapIndex into this index.
  void merge(const HashmapIndex& other, std::size_t row_id_offset);

  /**
   * @brief Generic Join Intersection (Scan-Probe).
   * @details Returns a View that scans the first handle and probes the rest.
   */
  template <typename FirstHandle, typename... OtherHandles>
  static auto intersect(const FirstHandle& first, const OtherHandles&... others);

 private:
  template <std::size_t Layer>
  void merge_layer(const HashmapIndex& other, std::size_t row_id_offset);

 private:
  struct Impl {
    using RowId = typename HashmapIndex::RowId;
    using Index = typename HashmapIndex::Index;
    using VerticalPartitionTuple =
        typename detail::VerticalPartitionTypes<arity, ValueType, RowId, Index>::type;

    memory_resource* resource_;
    VerticalPartitionTuple partitions_;
    std::size_t size_{0};            // Number of unique rows stored (set semantics)
    std::size_t rows_processed_{0};  // Number of rows processed (before deduplication)
    uint32_t index_arity_{0};        // Number of columns in the index (from spec.cols.size())

    explicit Impl(memory_resource* resource)
        : resource_(resource),
          partitions_(make_partitions(std::make_index_sequence<arity + 1>{}, resource)) {}

    void clear() {
      clear_partitions(std::make_index_sequence<arity + 1>{});
      size_ = 0;
      rows_processed_ = 0;
    }

    [[nodiscard]] std::size_t bytes_used() const {
      std::size_t total = sizeof(*this);
      bytes_used_partitions(std::make_index_sequence<arity + 1>{}, total);
      // Return 0 if index is empty (no data stored)
      if (size_ == 0 && total == sizeof(*this)) {
        return 0;
      }
      return total;
    }

    /// @brief Get access to a specific layer's map by index
    /// @tparam Layer The layer index (0 to arity, where arity is the leaf)
    /// @return Reference to the map at the specified layer
    /// @details Layer 0 to arity-1: non-leaf (Map<tuple, Set<IndexType>>)
    ///          Layer arity: leaf (Map<tuple<arity>, RowId>)
    template <std::size_t Layer>
    [[nodiscard]] constexpr auto& layer() {
      static_assert(Layer <= arity, "Layer index out of bounds");
      return std::get<Layer>(partitions_);
    }

    /// @brief Get const access to a specific layer's map by index
    template <std::size_t Layer>
    [[nodiscard]] constexpr const auto& layer() const {
      static_assert(Layer <= arity, "Layer index out of bounds");
      return std::get<Layer>(partitions_);
    }

    /// @brief Get the map type at a specific layer
    template <std::size_t Layer>
    using LayerMapType = std::tuple_element_t<Layer, VerticalPartitionTuple>;

    /// @brief Check if a layer is a leaf (last layer)
    template <std::size_t Layer>
    static constexpr bool is_leaf_layer() {
      return Layer == arity;
    }

    /// @brief Get access to a specific layer's map by runtime index (dispatch helper)
    /// @param layer The layer index (0 to arity, where arity is the leaf)
    /// @param fn Function to call with the layer as a template parameter
    template <typename F>
    void dispatch_layer(std::size_t layer, F&& fn) {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        bool found = false;
        ((!found && layer == Is ? (found = true, fn.template operator()<Is>(), true) : false) ||
         ...);
        (void)found;
      }(std::make_index_sequence<arity + 1>{});
    }
    template <typename F>
    void dispatch_layer(std::size_t layer, F&& fn) const {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        bool found = false;
        ((!found && layer == Is ? (found = true, fn.template operator()<Is>(), true) : false) ||
         ...);
        (void)found;
      }(std::make_index_sequence<arity + 1>{});
    }

   private:
    template <std::size_t I>
    static auto make_partition_element(memory_resource* resource) {
      using MapType = detail::LayerMapType_t<I, ValueType, RowId, Index, arity>;
      // All layers now use Set as value type, so initialization is the same
      return MapType(16, resource);
    }

    template <std::size_t... Is>
    static VerticalPartitionTuple make_partitions(std::index_sequence<Is...>,
                                                  memory_resource* resource) {
      return std::make_tuple(make_partition_element<Is>(resource)...);
    }

    template <std::size_t... Is>
    void clear_partitions(std::index_sequence<Is...>) {
      (std::get<Is>(partitions_).clear(), ...);
    }

    template <std::size_t... Is>
    void bytes_used_partitions(std::index_sequence<Is...>, std::size_t& total) const {
      ((total += std::get<Is>(partitions_).size() *
                 (sizeof(ValueType) + sizeof(std::conditional_t < Is == arity, RowId, Index >))),
       ...);
    }
  };

  std::unique_ptr<Impl> p_;
  std::size_t size_{0};
};

}  // namespace SRDatalog::index