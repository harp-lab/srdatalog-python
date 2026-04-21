#pragma once

#include "hashmap.h"
#include "logging.h"
#include <algorithm>
#include <optional>
#include <ranges>

namespace SRDatalog::index {

namespace detail {
// Helper template to construct tuple from array using TMP
// Works for any compile-time size N
template <std::size_t N, typename ValueType, std::size_t MaxArity>
auto make_tuple_from_array(const std::array<ValueType, MaxArity>& arr) {
  return [&arr]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(arr[Is]...);
  }(std::make_index_sequence<N>{});
}

// Dispatch helper to call make_tuple_from_array with runtime layer value
// Uses if-constexpr to dispatch to compile-time template instantiations
template <std::size_t MaxLayer, typename ValueType, std::size_t MaxArity>
auto make_tuple_from_array_dispatch(const std::array<ValueType, MaxArity>& arr, std::size_t layer) {
  // Dispatch to appropriate template instantiation based on layer value
  // This generates code for all possible layer values up to MaxLayer
  if constexpr (MaxLayer >= 0) {
    if (layer == 0)
      return make_tuple_from_array<1>(arr);
  }
  if constexpr (MaxLayer >= 1) {
    if (layer == 1)
      return make_tuple_from_array<2>(arr);
  }
  if constexpr (MaxLayer >= 2) {
    if (layer == 2)
      return make_tuple_from_array<3>(arr);
  }
  if constexpr (MaxLayer >= 3) {
    if (layer == 3)
      return make_tuple_from_array<4>(arr);
  }
  // For larger layers, we'd need more cases, but arity is typically small
  // Fallback: use a generic approach (this shouldn't be reached if arity is reasonable)
  return make_tuple_from_array<MaxLayer + 1>(arr);
}
}  // namespace detail

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
bool HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::valid()
    const noexcept {
  return impl_ != nullptr && impl_->index_arity_ > 0 && depth_ <= impl_->index_arity_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
bool HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::is_leaf()
    const noexcept {
  // Leaf is at depth == index_arity (after all columns are indexed)
  return valid() && impl_ && depth_ == impl_->index_arity_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
uint32_t HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::depth()
    const noexcept {
  return depth_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::values() const ->
    typename HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::ValueRange::type {
  if (!valid() || is_leaf()) {
    // Return empty iterator range (no materialization)
    static thread_local const Set<IndexType> empty_set(
        0, boost::hash<IndexType>{}, std::equal_to<>{},
        boost::container::pmr::polymorphic_allocator<IndexType>(nullptr));
    typename ValueRange::type result{&empty_set};
    return result;
  }

  // Query the map at current depth using the full prefix tuple as key
  // Return a view into the existing Set in the map (no copying)
  const Set<IndexType>* found_set = nullptr;

  // All layers 0 to arity-1 are non-leaf, use tuple keys uniformly
  // Layer 0: std::tuple<> (empty tuple)
  // Layer L: std::tuple<ValueType, ... (L times)>
  impl_->dispatch_layer(depth_, [&]<std::size_t L>() {
    // All layers 0 to arity-1 are non-leaf
    if constexpr (L < arity) {
      const auto& map = impl_->template layer<L>();
      if (depth_ == 0) {
        // Root: layer 0 uses std::tuple<> as key
        if constexpr (L == 0) {
          auto it = map.find(std::tuple<>{});
          if (it != map.end()) {
            found_set = &it->second;  // Get pointer to the set in the map
          }
        }
      } else {
        // Non-root: construct prefix tuple of size L from prefix_path_
        // At depth L, prefix_path_ contains values from depth 0 to L-1
        // make_tuple_from_array<L> takes first L elements from prefix_path_
        auto prefix_key = detail::make_tuple_from_array<L>(prefix_path_);
        auto it = map.find(prefix_key);
        if (it != map.end()) {
          found_set = &it->second;  // Get pointer to the set in the map
        }
      }
    }
  });

  if (found_set) {
    // Return iterator range wrapper (no copying)
    // Store pointer to set to keep iterators valid
    typename ValueRange::type result{found_set};
    return result;
  }

  // Return empty iterator range if not found
  static thread_local const Set<IndexType> empty_set(
      0, boost::hash<IndexType>{}, std::equal_to<>{},
      boost::container::pmr::polymorphic_allocator<IndexType>(nullptr));
  typename ValueRange::type result{&empty_set};
  return result;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::rows() const
    -> RowRange {
  if (!valid() || !is_leaf()) {
    return RowRange{static_cast<const RowId*>(nullptr), 0};
  }

  // At leaf level, look up the full prefix tuple in the leaf map
  // Leaf layer is at layer arity (after all non-leaf layers 0 to arity-1)

  // Dispatch to the correct leaf layer and construct tuple inside lambda where size is compile-time
  // Leaf layer now stores just one RowId per unique key
  static thread_local RowId row_id_storage;
  bool found = false;

  // Leaf layer is at layer arity
  impl_->dispatch_layer(arity, [&]<std::size_t L>() {
    if constexpr (L == arity) {
      // Leaf layer expects tuple of size arity
      auto full_prefix = detail::make_tuple_from_array<arity>(prefix_path_);
      const auto& leaf_map = impl_->template layer<L>();
      auto it = leaf_map.find(full_prefix);
      if (it != leaf_map.end()) {
        // Leaf layer stores just one RowId
        row_id_storage = it->second;
        found = true;
      }
    }
  });

  if (!found) {
    return RowRange{static_cast<const RowId*>(nullptr), 0};
  }
  return RowRange{&row_id_storage, 1};  // Return span with one element
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::prefix(
    ValueType next_key) const -> NodeHandle {
  if (!valid() || is_leaf()) {
    return NodeHandle{};
  }

  // Check if key exists at current layer using dispatch
  bool key_found = false;
  std::array<ValueType, arity> new_prefix = prefix_path_;

  std::size_t index_arity = impl_->index_arity_;

  // Check if we're navigating to the leaf
  // When depth_ == index_arity - 1, we're at the last non-leaf layer, navigating to leaf
  // When depth_ == index_arity, we're at the leaf (no further navigation)
  if (depth_ == index_arity - 1) {
    // At depth index_arity - 1, navigating to leaf (depth index_arity)
    // Look in leaf layer (arity) with full prefix tuple
    impl_->dispatch_layer(arity, [&]<std::size_t L>() {
      if constexpr (L == arity) {
        const auto& leaf_map = impl_->template layer<L>();
        // Construct full prefix tuple including next_key
        std::array<ValueType, arity> full_prefix_arr = prefix_path_;
        full_prefix_arr[depth_] = next_key;
        auto full_prefix = detail::make_tuple_from_array<arity>(full_prefix_arr);
        auto it = leaf_map.find(full_prefix);
        if (it != leaf_map.end()) {
          key_found = true;
          new_prefix[depth_] = next_key;
        }
      }
    });
  } else {
    // Non-leaf navigation: look in layer at current depth
    // For depth_ < index_arity - 1, we're at a non-leaf layer
    // All layers 0 to arity-1 are non-leaf
    impl_->dispatch_layer(depth_, [&]<std::size_t L>() {
      // All layers 0 to arity-1 are non-leaf (L < arity)
      if constexpr (L < arity) {
        const auto& layer_map = impl_->template layer<L>();

        if (depth_ == 0) {
          // Root: layer 0 uses std::tuple<> as key
          if constexpr (L == 0) {
            auto it = layer_map.find(std::tuple<>{});
            if (it != layer_map.end()) {
              const auto& values_set = it->second;
              if (values_set.contains(next_key)) {
                key_found = true;
                new_prefix[0] = next_key;
              }
            }
          }
        } else {
          // Non-root non-leaf: construct prefix tuple and check if next_key is in the set
          auto prefix_key = detail::make_tuple_from_array<L>(prefix_path_);
          auto it = layer_map.find(prefix_key);
          if (it != layer_map.end()) {
            const auto& values_set = it->second;
            if (values_set.contains(next_key)) {
              key_found = true;
              new_prefix[depth_] = next_key;
            }
          }
        }
      }
    });
  }

  if (!key_found) {
    return NodeHandle{};
  }

  // Create new handle with updated depth and prefix_path_
  // new_prefix now has prefix_path_[0..depth_] = values from root to current depth
  return NodeHandle{impl_, depth_ + 1, new_prefix};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
template <std::size_t N>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::prefix(
    const EncodedKeyPrefix<N>& suffix) const -> NodeHandle {
  if (!valid()) {
    return NodeHandle{};
  }

  auto cursor = *this;
  for (std::size_t i = 0; i < N; ++i) {
    cursor = cursor.prefix(static_cast<ValueType>(suffix[i]));
    if (!cursor.valid()) {
      break;
    }
  }
  return cursor;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
bool HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::contains_value(
    ValueType key) const noexcept {
  if (!valid() || is_leaf()) {
    return false;
  }

  bool found = false;
  impl_->dispatch_layer(depth_, [&]<std::size_t L>() {
    if constexpr (L < arity - 1) {
      const auto& layer_map = impl_->template layer<L>();

      if (depth_ == 0) {
        // Root: layer 0 uses std::tuple<> as key
        if constexpr (L == 0) {
          auto it = layer_map.find(std::tuple<>{});
          if (it != layer_map.end()) {
            const auto& values_set = it->second;
            found = values_set.contains(key);
          }
        }
      } else {
        // Non-root: construct prefix tuple and check if key is in the set
        auto prefix_key = detail::make_tuple_from_array<L>(prefix_path_);
        auto it = layer_map.find(prefix_key);
        if (it != layer_map.end()) {
          const auto& values_set = it->second;
          found = values_set.contains(key);
        }
      }
    }
  });
  return found;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
std::size_t HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::degree()
    const noexcept {
  if (!valid() || is_leaf()) {
    return 0;
  }

  std::size_t result = 0;
  impl_->dispatch_layer(depth_, [&]<std::size_t L>() {
    if constexpr (L < arity - 1) {
      const auto& layer_map = impl_->template layer<L>();

      if (depth_ == 0) {
        // Root: layer 0 uses std::tuple<> as key - return size of the set
        if constexpr (L == 0) {
          auto it = layer_map.find(std::tuple<>{});
          if (it != layer_map.end()) {
            const auto& values_set = it->second;
            result = values_set.size();
          }
        }
      } else {
        // Non-root: construct prefix tuple and return size of the set
        auto prefix_key = detail::make_tuple_from_array<L>(prefix_path_);
        auto it = layer_map.find(prefix_key);
        if (it != layer_map.end()) {
          const auto& values_set = it->second;
          result = values_set.size();
        }
      }
    }
  });
  return result;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
std::size_t HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::NodeHandle::unique_count()
    const noexcept {
  return degree();
}

// ============================================================================
// HashmapIndex implementation
// ============================================================================

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::HashmapIndex(
    memory_resource* resource)
    : p_(std::make_unique<Impl>(resource)) {}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::~HashmapIndex() = default;

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
void HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::build_from_encoded(
    const IndexSpec& spec, const std::array<std::span<const ValueType>, arity>& encoded_rows,
    memory_resource* resource) {
  auto row_size = encoded_rows[0].size();
  if (row_size == 0) {
    return;
  }

  // Use provided resource or existing one
  memory_resource* res = resource ? resource : (p_ ? p_->resource_ : nullptr);
  if (!res) {
    res = boost::container::pmr::get_default_resource();
  }

  // Check if this is first build or incremental
  bool needs_init = !p_ || size_ == 0;
  std::size_t index_arity = spec.cols.size();

  // Enforce that index must contain all columns (index_arity == arity)
  if (index_arity != arity) {
    throw std::runtime_error(
        "build_from_encoded: index must contain all columns (index_arity must equal arity)");
  }

  // Validate that spec.cols contains all columns from 0 to arity-1
  if (spec.cols.size() != arity) {
    throw std::runtime_error("build_from_encoded: spec.cols must contain exactly arity columns");
  }
  for (std::size_t i = 0; i < arity; ++i) {
    if (std::find(spec.cols.begin(), spec.cols.end(), i) == spec.cols.end()) {
      throw std::runtime_error(
          "build_from_encoded: spec.cols must contain all columns from 0 to arity-1");
    }
  }

  if (needs_init) {
    // Initialize fresh index
    p_ = std::make_unique<Impl>(res);
    p_->index_arity_ = static_cast<uint32_t>(index_arity);
    size_ = 0;
    p_->rows_processed_ = 0;
  } else {
    // Validate for incremental build
    if (p_->index_arity_ != index_arity) {
      throw std::runtime_error("build_from_encoded: index arity mismatch");
    }
  }

  std::size_t base_row_id = p_->rows_processed_;  // Current rows processed before adding new rows
  std::size_t unique_count = 0;                   // Count unique keys inserted
  for (std::size_t row_id = 0; row_id < row_size; ++row_id) {
    // Build the key path for this row using columns from spec.cols
    std::array<ValueType, arity> key_path;
    for (std::size_t i = 0; i < index_arity; ++i) {
      std::size_t col_idx = spec.cols[i];
      key_path[i] = encoded_rows[col_idx][row_id];
    }

    // Insert into non-leaf layers (0 to index_arity-1) and leaf layer (arity)
    // All layers 0 to arity-1 are non-leaf (path goes through all columns)
    for (std::size_t layer = 0; layer < index_arity; ++layer) {
      // Non-leaf layer: add the value at this layer to the set for the prefix tuple
      // Layer 0: insert key_path[0] (first column value) into set for empty tuple
      // Layer L: insert key_path[L] (L-th column value) into set for tuple(key_path[0..L-1])
      ValueType layer_value = key_path[layer];

      // Dispatch to the correct non-leaf layer and construct tuple inside lambda where layer is
      // compile-time
      p_->dispatch_layer(layer, [&]<std::size_t L>() {
        // All layers 0 to arity-1 are non-leaf
        if constexpr (L < arity) {
          auto& non_leaf_map = p_->template layer<L>();
          // Construct prefix tuple of size L using compile-time L
          // Layer 0: std::tuple<> (empty tuple)
          // Layer L: std::tuple<ValueType, ... (L times)>
          if constexpr (L == 0) {
            // Layer 0: use empty tuple as key, insert value from column 0
            auto [it, inserted] = non_leaf_map.try_emplace(
                std::tuple<>{},
                Set<IndexType>(
                    0, boost::hash<IndexType>{}, std::equal_to<>{},
                    boost::container::pmr::polymorphic_allocator<IndexType>(p_->resource_)));
            auto& values_set = it->second;
            values_set.insert(layer_value);
          } else {
            // Layer L > 0: construct tuple from first L values, insert value from column L
            auto prefix_key = detail::make_tuple_from_array<L>(key_path);
            auto [it, inserted] = non_leaf_map.try_emplace(
                prefix_key, Set<IndexType>(0, boost::hash<IndexType>{}, std::equal_to<>{},
                                           boost::container::pmr::polymorphic_allocator<IndexType>(
                                               p_->resource_)));
            auto& values_set = it->second;
            values_set.insert(layer_value);
          }
        }
      });
    }

    // Insert into leaf layer (layer arity) after all columns are indexed
    p_->dispatch_layer(arity, [&]<std::size_t L>() {
      if constexpr (L == arity) {
        // Leaf layer: map type is Map<KeyType, RowId> (just store one RowId per unique key)
        auto& leaf_map = p_->template layer<L>();
        // Leaf layer expects tuple of size arity
        auto full_prefix = detail::make_tuple_from_array<arity>(key_path);
        // Get or create entry for this prefix key
        auto [it, inserted] =
            leaf_map.try_emplace(full_prefix, static_cast<RowId>(base_row_id + row_id));
        // Count unique keys: only count when the key is newly inserted
        if (inserted) {
          unique_count++;
        }
        // If key already exists, we keep the first row ID (don't overwrite)
      }
    });
  }

  // Update size_ to count unique rows stored (set semantics)
  size_ += unique_count;
  // Update rows_processed_ to track all rows processed (for incremental builds)
  p_->rows_processed_ = base_row_id + row_size;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::root() const -> NodeHandle {
  if (!p_) {
    return NodeHandle{};
  }
  // Root has empty prefix path (all zeros)
  std::array<ValueType, arity> empty_path{};
  return NodeHandle{p_.get(), 0, empty_path};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
template <std::size_t N>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::prefix_lookup(
    const EncodedKeyPrefix<N>& k) const -> NodeHandle {
  if (!p_ || N > arity) {
    return NodeHandle{};
  }

  // Directly navigate to layer N-1
  // Layer 0: uses std::tuple<> (empty tuple) as key
  // Layer L: uses std::tuple<ValueType, ... (L times)> as key
  // For prefix of size N, we look in layer N-1
  if (N == 0) {
    // Empty prefix - return root
    return root();
  }

  // Construct prefix tuple from first N values using TMP
  // For N=1, we look in layer 0 which uses std::tuple<> - but we have one value, so we need to
  // check Actually, for N=1, we should look in layer 0 with empty tuple and check if the value is
  // in the set For N>1, we construct tuple of size N-1 and look in layer N-1
  if constexpr (N == 1) {
    // For N=1, look in layer 0 (uses std::tuple<> as key) and check if k[0] is in the set
    const auto& map = p_->template layer<0>();
    auto it = map.find(std::tuple<>{});
    if (it == map.end()) {
      return NodeHandle{};
    }
    const auto& values_set = it->second;
    if (!values_set.contains(static_cast<ValueType>(k[0]))) {
      return NodeHandle{};
    }
    // Value found - create handle with depth 1
    std::array<ValueType, arity> prefix_path{};
    prefix_path[0] = static_cast<ValueType>(k[0]);
    return NodeHandle{p_.get(), 1, prefix_path};
  } else if constexpr (N == arity) {
    // Leaf case: look up full key in leaf layer (arity)
    auto full_tuple = [&k]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(static_cast<ValueType>(k[Is])...);
    }(std::make_index_sequence<arity>{});

    // Look up in leaf layer (layer arity, not arity - 1)
    const auto& leaf_map = p_->template layer<arity>();
    auto it = leaf_map.find(full_tuple);
    if (it == leaf_map.end()) {
      return NodeHandle{};  // Key not found
    }
    // Found - create handle with depth arity (leaf)
    std::array<ValueType, arity> prefix_path{};
    for (std::size_t i = 0; i < arity; ++i) {
      prefix_path[i] = static_cast<ValueType>(k[i]);
    }
    return NodeHandle{p_.get(), static_cast<uint32_t>(arity), prefix_path};
  } else {
    // For 1 < N < arity, construct tuple of size N-1 and look in layer N-1
    // All layers 0 to arity-1 are non-leaf
    auto prefix_tuple = [&k]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(static_cast<ValueType>(k[Is])...);
    }(std::make_index_sequence<N - 1>{});

    // Look up in layer N-1
    const auto& map = p_->template layer<N - 1>();
    auto it = map.find(prefix_tuple);
    if (it == map.end()) {
      return NodeHandle{};  // Prefix not found
    }
    // Check if k[N-1] is in the set
    const auto& values_set = it->second;
    if (!values_set.contains(static_cast<ValueType>(k[N - 1]))) {
      return NodeHandle{};
    }
    // Found - create handle with depth N
    std::array<ValueType, arity> prefix_path{};
    for (std::size_t i = 0; i < N && i < arity; ++i) {
      prefix_path[i] = static_cast<ValueType>(k[i]);
    }
    return NodeHandle{p_.get(), static_cast<uint32_t>(N), prefix_path};
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
std::size_t HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::bytes_used() const {
  return p_ ? p_->bytes_used() : 0;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
void HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::clear() noexcept {
  if (p_) {
    p_->clear();
  }
  size_ = 0;
  // rows_processed_ is cleared in p_->clear()
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
void HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::merge(
    const HashmapIndex& other, std::size_t row_id_offset) {
  if (!other.p_ || !p_) {
    return;
  }

  // Merge each layer (0 to arity, including leaf)
  [this, &other, row_id_offset]<std::size_t... Is>(std::index_sequence<Is...>) {
    ((this->template merge_layer<Is>(other, row_id_offset)), ...);
  }(std::make_index_sequence<arity + 1>{});

  // size_ is updated in merge_layer for leaf layer (only counts newly inserted keys)
  // Don't add other.size_ here because merge_layer already handles counting unique keys
  // rows_processed_ tracks all rows processed
  p_->rows_processed_ += other.p_->rows_processed_;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
template <std::size_t Layer>
void HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::merge_layer(
    const HashmapIndex& other, std::size_t row_id_offset) {
  auto& this_map = p_->template layer<Layer>();
  const auto& other_map = other.p_->template layer<Layer>();

  if constexpr (Layer == arity) {
    // Leaf layer: merge row IDs with offset (just store one RowId per unique key)
    for (const auto& [key, other_row_id] : other_map) {
      auto [it, inserted] = this_map.try_emplace(
          key, static_cast<RowId>(static_cast<std::size_t>(other_row_id) + row_id_offset));
      // Update size_ only when a new unique key is inserted
      if (inserted) {
        size_++;
      }
      // If key already exists, keep the first row ID (don't overwrite)
    }
  } else {
    // Non-leaf layer: merge sets for each key
    // Optimize: use bulk insert with iterator range for better performance
    for (const auto& [key, other_set] : other_map) {
      auto [it, inserted] = this_map.try_emplace(
          key,
          Set<IndexType>(0, boost::hash<IndexType>{}, std::equal_to<>{},
                         boost::container::pmr::polymorphic_allocator<IndexType>(p_->resource_)));
      auto& this_set = it->second;
      // Merge values from other_set into this_set using bulk insert
      // This is more efficient than individual inserts
      this_set.insert(other_set.begin(), other_set.end());
    }
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType,
          typename IndexType>
template <typename FirstHandle, typename... OtherHandles>
auto HashmapIndex<SR, AttrTuple, ValueType, RowIdType, IndexType>::intersect(
    const FirstHandle& first, const OtherHandles&... others) {
  // If only one handle, just return its values (no intersection needed)
  if constexpr (sizeof...(OtherHandles) == 0) {
    return first.values();
  } else {
    // Multiple handles: use filter view for hash-based intersection
    auto others_tuple = std::make_tuple(others...);

    // Define Probe Predicate
    auto predicate = [others_tuple](const auto& key) {
      bool all_match = true;
      std::apply([&](const auto&... h) { ((all_match = all_match && h.contains_value(key)), ...); },
                 others_tuple);
      return all_match;
    };

    // Return Filter View
    return std::ranges::views::filter(first.values(), predicate);
  }
}

}  // namespace SRDatalog::index
