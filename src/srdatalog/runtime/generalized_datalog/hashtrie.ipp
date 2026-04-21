#pragma once

#include "hashtrie.h"
#include "logging.h"

namespace SRDatalog {

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
struct HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::Impl {
  using RowId = typename HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::RowId;

  struct Node {

    using MapType = Map<ValueType, uint32_t>;
    using VecKeys = boost::container::pmr::vector<ValueType>;
    using VecRows = boost::container::pmr::vector<RowId>;

    MapType children;
    VecKeys keys_cache;
    VecRows postings;
    uint32_t depth{0};

    explicit Node(memory_resource* r, bool is_leaf_node)
        : children(is_leaf_node ? 0 : 16, r),  // Construct with size directly!
          keys_cache(r), postings(r) {}

    Node() = delete;

    [[nodiscard]] bool is_leaf() const noexcept {
      return !postings.empty() || children.empty();
    }
    [[nodiscard]] std::size_t degree() const noexcept {
      return keys_cache.size();
    }
    [[nodiscard]] bool contains(ValueType k) const noexcept {
      // std::cout << "contains function; " << k << std::endl;
      assert(!is_leaf() && "Node::contains called on leaf node");
      assert(children.get_allocator().resource() != nullptr &&
             "Node::contains: children allocator has null resource");
      assert(k != std::numeric_limits<ValueType>::max() &&
             "Node::contains: key value appears invalid");
      return children.contains(k);
    }
  };

  memory_resource* resource_;                // Handle to the arena
  boost::container::pmr::deque<Node> nodes;  // Deque using the arena
  uint32_t arity{0};

  explicit Impl(memory_resource* resource)
      : resource_(resource), nodes(resource) {  // Initialize deque with resource
    nodes.emplace_back(this->resource_, /*is_leaf=*/false);
  }

  Node create_node() const {
    return Node(resource_);
  }

  // Helper to safely get default resource if nullptr provided
  static memory_resource* ensure_resource(memory_resource* r) {
    return r ? r : boost::container::pmr::get_default_resource();
  }

  [[nodiscard]] uint32_t root_id() const noexcept {
    return 0;
  }

  uint32_t ensure_child(uint32_t nid, ValueType key, uint32_t child_depth) {
    const uint32_t potential_cid = static_cast<uint32_t>(nodes.size());

    auto [it, inserted] = nodes[nid].children.try_emplace(key, potential_cid);

    if (inserted) {
      bool is_child_leaf = (child_depth == arity);
      nodes.emplace_back(resource_, is_child_leaf);
      nodes.back().depth = child_depth;
      return potential_cid;
    }

    return it->second;
  }

  [[nodiscard]] uint32_t find_child(uint32_t nid, ValueType key) const {
    const auto& n = nodes[nid];
    auto it = n.children.find(key);
    return (it == n.children.end()) ? UINT32_MAX : it->second;
  }

  [[nodiscard]] std::size_t bytes_used() const {
    std::size_t b = sizeof(*this) + (nodes.size() * sizeof(Node));
    for (const auto& n : nodes) {
      b += n.children.size() * (sizeof(std::size_t) + sizeof(uint32_t));
      b += n.keys_cache.capacity() * sizeof(std::size_t);
      b += n.postings.capacity() * sizeof(RowId);
    }
    return b;
  }

  void clear() {
    nodes.clear();
    // Re-create root node
    nodes.emplace_back(resource_, /*is_leaf=*/false);
  }

  /// @brief Merge another Impl's trie structure into this Impl
  /// @param other The source Impl to merge from
  /// @param row_id_offset Offset to add to row IDs from the source index
  void merge(const Impl& other, std::size_t row_id_offset) {
    if (other.nodes.empty()) {
      return;
    }
    if (nodes.empty()) {
      nodes.emplace_back(resource_, /*is_leaf=*/false);
    }
    merge_nodes(root_id(), other, other.root_id(), row_id_offset);
  }

  /// @brief Recursively merge nodes from source trie into destination trie
  /// @param dst_nid Destination node ID in this Impl
  /// @param src Source Impl to merge from
  /// @param src_nid Source node ID
  /// @param offset Row ID offset to apply to postings
  void merge_nodes(uint32_t dst_nid, const Impl& src, uint32_t src_nid, std::size_t offset) {
    const auto& src_node = src.nodes[src_nid];
    if (src_node.is_leaf()) {
      auto& dst_node = nodes[dst_nid];
      if (offset == 0) {
        dst_node.postings.insert(dst_node.postings.end(), src_node.postings.begin(),
                                 src_node.postings.end());
      } else {
        std::size_t needed = dst_node.postings.size() + src_node.postings.size();
        if (dst_node.postings.capacity() < needed) {
          dst_node.postings.reserve(needed);
        }
        for (const auto& r : src_node.postings) {
          dst_node.postings.push_back(r + offset);
        }
      }
      return;
    }
    auto& dst_node = nodes[dst_nid];
    for (const auto& [key, src_child_id] : src_node.children) {
      uint32_t dst_child_id = ensure_child(dst_nid, key, nodes[dst_nid].depth + 1);
      merge_nodes(dst_child_id, src, src_child_id, offset);
    }
    // Simplest way is clear the keys_cache, but we need find better way (a set union?, if we can
    // make sure the insertions are in order)
    dst_node.keys_cache.clear();
  }
};

// The implementaiton for the declared function of hashtires
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::HashTrieIndex(memory_resource* resource)
    : p_(std::make_shared<Impl>(resource)) {}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::~HashTrieIndex() = default;

// root()
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::root() const -> NodeHandle {
  return NodeHandle{p_ ? p_.get() : nullptr, p_ ? p_->root_id() : 0};
}

// prefix_lookup(KeyPrefix)
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <std::size_t N>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::prefix_lookup(
    const EncodedKeyPrefix<N>& k) const -> NodeHandle {
  if (!p_)
    return NodeHandle{};
  uint32_t nid = p_->root_id();
  // uint32_t depth = 0;
  // for (std::size_t comp : k.comps) {
  //   const uint32_t cid = p_->find_child(nid, comp);
  //   if (cid == UINT32_MAX)
  //     return NodeHandle{}; // invalid
  //   nid = cid;
  //   // ++depth;
  // }
  // return NodeHandle{p_, nid};
  for (std::size_t i = 0; i < N; ++i) {
    const uint32_t cid = p_->find_child(nid, k[i]);
    if (cid == UINT32_MAX)
      return NodeHandle{};
    nid = cid;
  }
  return NodeHandle{p_.get(), nid};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::bytes_used() const {
  return p_ ? p_->bytes_used() : 0;
}

// The implementation for the method of the NodeHandles
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::valid() const noexcept {
  if (!impl_) {
    return false;
  }
  return nid_ < static_cast<uint32_t>(impl_->nodes.size());
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::is_leaf()
    const noexcept {
  if (!valid())
    return false;
  const auto& n = impl_->nodes[nid_];
  // We decide whether a node is leaf or non-leaf node
  // based on its current depth and compared with the arity
  return n.depth == impl_->arity;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline uint32_t HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::depth()
    const noexcept {
  if (!valid())
    return 0;
  return impl_->nodes[nid_].depth;
}

// template <Semiring SR, ColumnElementTuple AttrTuple>
// inline auto Relation<SR, AttrTuple>::HashTrieIndex::NodeHandle::values() const -> ValueRange {
//   static const std::size_t* kNull = nullptr;
//   if (!valid())
//     return ValueRange{kNull, 0};
//   const auto& n = impl_->nodes[nid_];
//   return ValueRange{n.keys_cache.data(), n.keys_cache.size()};
// }
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::values() const
    -> ValueRange {
  if (!valid())
    return ValueRange{static_cast<const ValueType*>(nullptr), 0};

  auto& n = const_cast<Impl::Node&>(impl_->nodes[nid_]);

  if (n.keys_cache.empty() && !n.children.empty()) {
    n.keys_cache.reserve(n.children.size());
    for (auto const& [key, val] : n.children) {
      n.keys_cache.push_back(key);
    }
  }

  return ValueRange{n.keys_cache.data(), n.keys_cache.size()};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::rows() const
    -> RowRange {
  static const RowId* kNull = nullptr;
  if (!valid())
    return RowRange{kNull, 0};
  const auto& n = impl_->nodes[nid_];
  return RowRange{n.postings.data(), n.postings.size()};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::prefix(
    ValueType next_key) const -> NodeHandle {
  if (!valid())
    return NodeHandle{};
  const uint32_t cid = impl_->find_child(nid_, next_key);
  if (cid == UINT32_MAX)
    return NodeHandle{};
  return NodeHandle{impl_, cid};
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <std::size_t N>
inline auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::prefix(
    const EncodedKeyPrefix<N>& suffix) const -> NodeHandle {
  if (!valid())
    return NodeHandle{};
  auto cursor = *this;
  // Convert std::size_t to ValueType for prefix lookup
  for (std::size_t i = 0; i < N; ++i) {
    cursor = cursor.prefix(static_cast<ValueType>(suffix[i]));
    if (!cursor.valid())
      break;
  }
  return cursor;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline bool HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::contains_value(
    ValueType key) const noexcept {
  if (!valid())
    return false;
  assert(impl_ != nullptr && "NodeHandle::contains_value: impl is null");
  assert(nid_ < impl_->nodes.size() && "NodeHandle::contains_value: invalid node id");
  const auto* node_ptr = &impl_->nodes[nid_];
  // std::cout << "contains_value last step" << std::endl;
  assert(node_ptr != nullptr && "NodeHandle::contains_value: node pointer is null");
  // std::cout << "contains_value last step_1" << std::endl;
  return node_ptr->contains(key);
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::degree()
    const noexcept {
  if (!valid())
    return 0;
  const auto& n = impl_->nodes[nid_];
  return n.degree();
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline std::size_t HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::NodeHandle::unique_count()
    const noexcept {
  if (!valid() || is_leaf())
    return 0;
  // For HashTrieIndex, values() returns a span which has .size()
  // Since HashTrieIndex stores unique keys, this equals the size
  return values().size();
}

// build the indexing from pre-encoded rows
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
inline void HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::build_from_encoded(
    const IndexSpec& spec, const std::array<std::span<const ValueType>, arity>& encoded_rows,
    memory_resource* resource) {
  // constexpr std::size_t N = std::tuple_size_v<AttrTuple>;
  auto row_size = encoded_rows[0].size();
  if (row_size == 0) {
    return;  // Nothing to do
  }
  // Check if this is first build or incremental
  bool needs_init = !p_ || p_->nodes.empty() || size_ == 0;

  if (needs_init) {
    // Initialize fresh trie
    p_ = std::make_shared<Impl>(resource);
    // Root node is already created by Impl constructor with proper PMR initialization
    // No need to clear and re-add
    p_->arity = static_cast<uint32_t>(spec.cols.size());
    size_ = 0;
  } else {
    // Validate for incremental build
    if (p_->arity != spec.cols.size()) {
      throw std::runtime_error("build_from_encoded: arity mismatch");
    }
  }

  uint32_t root = p_->root_id();
  std::size_t base_row_id = size_;  // Current index size

  // Determine memory resource: use provided, or get from existing index, or use default
  // Use the Impl's resource_ member for consistency (always non-null)
  memory_resource* cache_resource = resource ? Impl::ensure_resource(resource) : p_->resource_;

  // Cache for last processed row to avoid repeated hashmap lookups
  // This optimizes the case when multiple rows share the same prefix (e.g., (1,2), (1,3), (1,4))
  // Use PMR-aware containers for consistency with the rest of the index structure
  Vector<uint32_t> cached_node_ids(cache_resource);  // node ID at each depth level
  Vector<std::size_t> cached_keys(cache_resource);   // keys at each depth level for comparison
  cached_node_ids.reserve(spec.cols.size() + 1);
  cached_keys.reserve(spec.cols.size());
  std::size_t shared_prefix_length = 0;

  // Insert all rows (works for both full and incremental)
  for (std::size_t row_id = 0; row_id < row_size; ++row_id) {
    uint32_t nid = root;
    uint32_t depth = 0;

    // Find shared prefix length with cached path
    shared_prefix_length = 0;
    if (!cached_keys.empty() && row_id > 0) {
      for (std::size_t i = 0; i < spec.cols.size() && i < cached_keys.size(); ++i) {
        ValueType key = encoded_rows[spec.cols[i]][row_id];
        if (key == cached_keys[i] && (i + 1) < cached_node_ids.size()) {
          shared_prefix_length = i + 1;
        } else {
          break;
        }
      }
    }

    // Reuse cached nodes up to shared_prefix_length
    if (shared_prefix_length > 0) {
      nid = cached_node_ids[shared_prefix_length];
      depth = shared_prefix_length;
    } else {
      // Reset cache if no shared prefix
      cached_node_ids.clear();
      cached_keys.clear();
      cached_node_ids.push_back(root);
    }

    // Navigate/create path through trie for remaining levels
    cached_node_ids.resize(shared_prefix_length + 1);  // Keep shared prefix
    cached_keys.resize(shared_prefix_length);

    for (std::size_t i = shared_prefix_length; i < spec.cols.size(); ++i) {
      std::size_t col_idx = spec.cols[i];
      ValueType key = encoded_rows[col_idx][row_id];
      const uint32_t cid = p_->ensure_child(nid, key, depth + 1);
      nid = cid;
      ++depth;

      // Update cache
      cached_node_ids.push_back(nid);
      cached_keys.push_back(key);
    }

    // Append absolute row_id to leaf
    p_->nodes[nid].postings.push_back(static_cast<RowId>(base_row_id + row_id));
  }

  // Regenerate key caches for all nodes
  // p_->finalize_keys_cache();

  // Update size
  size_ += row_size;
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::clone_from(const HashTrieIndex& other,
                                                                    memory_resource* resource) {
  // If source is empty, clear this index
  if (!other.p_ || other.p_->nodes.empty()) {
    // Always require a resource (use provided or source's, or default)
    memory_resource* target_resource = resource;
    if (!target_resource && other.p_) {
      target_resource = other.p_->resource_;
    }
    if (!target_resource) {
      target_resource = default_memory_resource();
    }
    p_ = std::make_shared<Impl>(target_resource);
    size_ = 0;
    return;
  }

  // Determine memory resource: use provided, or get from source, or use default
  memory_resource* target_resource = resource;
  if (!target_resource && other.p_) {
    // Use source's resource member if not explicitly provided
    target_resource = other.p_->resource_;
  }
  target_resource = Impl::ensure_resource(target_resource);

  // Create new Impl with target resource
  p_ = std::make_shared<Impl>(target_resource);
  p_->arity = other.p_->arity;
  size_ = other.size_;

  // Reserve space for nodes
  // p_->nodes.reserve(other.p_->nodes.size());

  // Deep copy all nodes
  for (const auto& src_node : other.p_->nodes) {
    typename Impl::Node dst_node;
    dst_node.depth = src_node.depth;

    // Deep copy children map - construct with explicit resource
    // Use the range constructor with allocator
    dst_node.children = Map<std::size_t, uint32_t>(
        src_node.children.begin(), src_node.children.end(), src_node.children.size(),
        boost::container::pmr::polymorphic_allocator<std::pair<const std::size_t, uint32_t>>(
            target_resource));

    // Deep copy keys_cache vector
    dst_node.keys_cache = Vector<std::size_t>(target_resource);
    dst_node.keys_cache.assign(src_node.keys_cache.begin(), src_node.keys_cache.end());

    // Deep copy postings vector
    dst_node.postings = Vector<RowId>(target_resource);
    dst_node.postings.assign(src_node.postings.begin(), src_node.postings.end());

    p_->nodes.push_back(std::move(dst_node));
  }
}

// ------------------------ merge() implementation ------------------------

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
void HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::merge(const HashTrieIndex& other,
                                                               uint32_t row_id_offset) {
  if (other.empty() || !other.p_) {
    return;
  }
  if (empty() || !p_) {
    // If this index is empty, initialize it
    if (!p_) {
      // Use other's resource if available, or default
      memory_resource* r = other.p_ ? other.p_->resource_ : nullptr;
      p_ = std::make_shared<Impl>(Impl::ensure_resource(r));
    }
    p_->arity = other.p_->arity;
    size_ = 0;
  }
  // Delegate to Impl::merge
  p_->merge(*other.p_, row_id_offset);
  size_ += other.size();
}

template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
template <typename FirstHandle, typename... OtherHandles>
auto HashTrieIndex<SR, AttrTuple, ValueType, RowIdType>::intersect(const FirstHandle& first,
                                                                   const OtherHandles&... others) {
  // If only one handle, just return its values (no intersection needed)
  if constexpr (sizeof...(OtherHandles) == 0) {
    return first.values();
  } else {
    // Multiple handles: use filter view for hash-based intersection
    // 1. Capture others in a tuple for the predicate
    auto others_tuple = std::make_tuple(others...);

    // 2. Define Probe Predicate
    auto predicate = [others_tuple](const auto& key) {
      bool all_match = true;
      std::apply([&](const auto&... h) { ((all_match = all_match && h.contains_value(key)), ...); },
                 others_tuple);
      return all_match;
    };

    // 3. Return Filter View
    // std::views::filter is lazy and works with any forward iterator
    return std::ranges::views::filter(first.values(), predicate);
  }
}

}  // namespace SRDatalog
