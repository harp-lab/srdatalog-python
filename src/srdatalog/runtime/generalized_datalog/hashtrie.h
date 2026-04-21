#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>

#include "column.h"
#include "index.h"  // CNodeHandle concept
#include "semiring.h"
#include "tmp.h"
#include <boost/mp11.hpp>

namespace SRDatalog {

// EncodedKeyPrefix is defined in index.h (included above)

// ============================================================
// HashTrie index
// ============================================================
/**
 * @brief Multi-level hash trie index for efficient prefix lookups on
 * relations.
 *
 * @details HashTrieIndex provides a hierarchical indexing structure optimized
 * for Datalog evaluation. It organizes relation rows into a multi-level trie
 * where each level corresponds to one indexed column.
 *
 * @section hashtrie_structure Structure
 *
 * The trie is organized as follows:
 * - **Root node**: Entry point at depth 0
 * - **Internal nodes**: Each node at depth `d` maps encoded values (size_t)
 *   to child nodes at depth `d+1`
 * - **Leaf nodes**: Nodes at depth equal to the index arity contain posting
 *   lists of row IDs matching the full key prefix
 *
 * @section hashtrie_features Key Features
 *
 * - **Prefix indexing**: Efficiently supports partial key lookups (e.g., find
 *   all rows where column 0 = 5)
 * - **Encoded keys**: Uses size_t-encoded values for efficient hashing and
 *   comparison
 * - **Lazy building**: Indexes are built on-demand when first accessed
 * - **Memory efficient**: Uses polymorphic memory resources (PMR) for
 *   flexible allocation strategies
 * - **Immutable after build**: Once built, the index structure is read-only
 *   (rebuild required for updates)
 *
 * @section hashtrie_usage Usage
 *
 * Indexes are typically accessed through Relation::ensure_index():
 *
 * @code{.cpp}
 * // Build index on columns 0 and 1
 * IndexSpec spec;
 * spec.cols = {0, 1};
 * const auto& idx = rel.ensure_index(spec);
 *
 * // Query with full key
 * auto prefix = Prefix<int, int>{5, 10};
 * auto node = idx.prefix_lookup(prefix.encoded());
 * if (node.valid() && node.is_leaf()) {
 *   for (auto row_id : node.rows()) {
 *     // Process matching rows
 *   }
 * }
 *
 * // Query with partial key (prefix lookup)
 * auto partial = Prefix<int>{5};
 * auto node2 = idx.prefix_lookup(partial.encoded());
 * if (node2.valid() && !node2.is_leaf()) {
 *   for (auto next_key : node2.values()) {
 *     // Enumerate possible values for column 1 when column 0 = 5
 *   }
 * }
 * @endcode
 *
 * @section hashtrie_navigation Navigation
 *
 * Navigation uses NodeHandle cursors:
 * - `root()`: Get handle to root node (depth 0)
 * - `prefix_lookup(key)`: Jump directly to node matching full/partial key
 * - `NodeHandle::prefix(key)`: Descend one level using encoded key
 * - `NodeHandle::values()`: Enumerate next-level keys (non-leaf nodes)
 * - `NodeHandle::rows()`: Get row IDs matching full key (leaf nodes)
 *
 * @section hashtrie_memory Memory Management
 *
 * - Uses polymorphic memory resources for flexible allocation
 * - Non-copyable (to avoid accidental heavy copies)
 * - Movable (cheap handle moves)
 * - NodeHandle is a non-owning view - must ensure HashTrieIndex outlives any NodeHandles
 *
 * @section hashtrie_building Building
 *
 * Indexes are built from pre-encoded row data:
 * - `build_from_encoded(spec, encoded_rows)`: Build from pre-encoded row
 *   data. The Relation class handles encoding and calls this method.
 * - Indexes are typically built through `Relation::ensure_index()`, which
 *   manages encoding and incremental building
 *
 * @note The implementation uses a pimpl (Impl) pattern to keep the header
 * lightweight. Actual trie structure is hidden in the implementation file.
 *
 * @see NodeHandle for cursor-based navigation API
 * @see IndexSpec for index specification format
 * @see Relation::ensure_index for typical usage pattern
 */
template <Semiring SR, ColumnElementTuple AttrTuple, typename ValueType = uint32_t,
          typename RowIdType = uint32_t>
class HashTrieIndex {
  struct Impl;

 public:
  using RowId = RowIdType;
  using RowRange = std::span<const RowId>;
  using ValueRange = std::span<const ValueType>;
  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  /**
   * @brief Cursor for navigating and querying nodes in a HashTrieIndex.
   *
   * @details NodeHandle provides a lightweight, cursor-based API for
   * traversing the trie structure. It acts as a non-owning view into a specific node
   * and provides methods to query the node's state and navigate to children.
   *
   * @section nodehandle_lifecycle Lifecycle
   *
   * - Default-constructed handles are invalid (valid() returns false)
   * - Handles are created by HashTrieIndex methods (root(), prefix_lookup())
   * - Handles remain valid only as long as the parent HashTrieIndex exists
   *   (non-owning view - caller must ensure HashTrieIndex lifetime)
   * - Handles are trivially copyable (POD-style: just a raw pointer and node ID)
   *
   * @section nodehandle_types Node Types
   *
   * Nodes can be either:
   * - **Internal nodes** (is_leaf() == false): Contain mappings to child
   *   nodes. Use values() to enumerate possible next-level keys.
   * - **Leaf nodes** (is_leaf() == true): Contain posting lists of row IDs.
   *   Use rows() to get matching row IDs.
   *
   * @section nodehandle_usage Usage
   *
   * @code{.cpp}
   * auto node = idx.root();  // Start at root
   * if (node.valid() && !node.is_leaf()) {
   *   // Enumerate all possible values at first level
   *   for (auto key : node.values()) {
   *     auto child = node.prefix(key);
   *     if (child.valid() && child.is_leaf()) {
   *       // Process rows matching this key
   *       for (auto row_id : child.rows()) {
   *         // ...
   *       }
   *     }
   *   }
   * }
   * @endcode
   */
  class NodeHandle {
   public:
    /// @brief Default constructor creates an invalid handle.
    /// @details Default-constructed handles are invalid until assigned from
    ///          a HashTrieIndex method.
    NodeHandle() = default;

    /// @brief Check if this handle points to a valid node.
    /// @return true if the handle is valid and points to an existing node
    /// @details Invalid handles are returned when navigation fails (e.g.,
    ///          key not found). Always check valid() before using other
    ///          methods.
    [[nodiscard]] bool valid() const noexcept;

    /// @brief Check if this node is a leaf node.
    /// @return true if depth() equals the index arity (leaf node)
    /// @details Leaf nodes contain row IDs, while internal nodes contain
    ///          mappings to child nodes. Only call rows() on leaf nodes.
    [[nodiscard]] bool is_leaf() const noexcept;

    /// @brief Get the depth of this node in the trie.
    /// @return Depth in range [0, arity], where 0 is root and arity is leaf
    /// @details Depth indicates how many key components have been matched
    ///          to reach this node.
    [[nodiscard]] uint32_t depth() const noexcept;

    /// @brief Enumerate encoded keys for the next level (internal nodes
    /// only).
    /// @return ValueRange view of encoded keys that exist at the next level
    /// @details This method is only valid for non-leaf nodes. It returns a
    ///          view of all possible encoded values that can be used with
    ///          prefix() to descend to child nodes.
    /// @pre !is_leaf() && valid()
    /// @note The returned range is not guaranteed to be sorted.
    [[nodiscard]] ValueRange values() const;

    /// @brief Get row IDs matching the full key prefix (leaf nodes only).
    /// @return RowRange view of row IDs stored at this leaf node
    /// @details This method is only valid for leaf nodes. It returns all
    ///          row IDs from the relation that match the full key prefix
    ///          leading to this node.
    /// @pre is_leaf() && valid()
    [[nodiscard]] RowRange rows() const;

    /// @brief Descend one level using an encoded key.
    /// @param next_key Encoded value for the next level
    /// @return NodeHandle to the child node, or invalid handle if key not
    /// found
    /// @details Navigates from the current node to a child node at depth+1.
    ///          Returns invalid handle if the key doesn't exist at this
    ///          level.
    /// @pre valid() && !is_leaf()
    NodeHandle prefix(ValueType next_key) const;

    /// @brief Descend multiple levels using an encoded key prefix.
    /// @tparam N Number of levels to descend (length of suffix)
    /// @param suffix Encoded key prefix for the next N levels
    /// @return NodeHandle to the node reached after descending, or invalid
    ///         handle if any key component not found
    /// @details Efficiently navigates multiple levels in one call. Equivalent
    ///          to calling prefix() N times sequentially, but more efficient.
    /// @pre valid() && depth() + N <= arity
    template <std::size_t N>
    NodeHandle prefix(const EncodedKeyPrefix<N>& suffix) const;

    /// @brief Check if a specific encoded key exists at the next level.
    /// @param key Encoded value to check
    /// @return true if the key exists as a child of this node
    /// @details This is more efficient than enumerating values() when you
    ///          only need to check existence.
    /// @pre valid() && !is_leaf()
    [[nodiscard]] bool contains_value(ValueType key) const noexcept;

    /// @brief Get the number of children (out-degree) of this node.
    /// @return Number of distinct keys at the next level
    /// @details For internal nodes, this equals unique_count(). For leaf
    ///          nodes, this is 0 (no children).
    [[nodiscard]] std::size_t degree() const noexcept;

    /// @brief Get the count of unique values (size of values() range).
    /// @return Number of unique values at the next level
    /// @details This is the size of the view returned by values().
    ///          For HashTrieIndex, this equals values().size() since values are already unique.
    ///          For consistency with SortedArrayIndex which deduplicates, use this method instead
    ///          of values().size().
    [[nodiscard]] std::size_t unique_count() const noexcept;

   private:
    template <Semiring SR2, ColumnElementTuple AttrTuple2, typename ValueType2, typename RowIdType2>
    friend class HashTrieIndex;
    NodeHandle(Impl* impl, uint32_t nid) noexcept : impl_(impl), nid_(nid) {}

    /// @brief Non-owning raw pointer to the internal trie object.
    /// @details Must outlive this handle. Points to HashTrieIndex::Impl.
    Impl* impl_{nullptr};

    /// @brief Opaque node identifier within Impl.
    /// @details Node ID used to index into the Impl's nodes array.
    uint32_t nid_{0};
  };
  // compile time check if the NodeHandle is a valid according to the interface concept
  static_assert(CNodeHandle<NodeHandle>, "NodeHandle must be a valid NodeHandle");

  /// @brief Construct an empty HashTrieIndex.
  /// @param resource Memory resource for allocation (must match the relation's resource)
  /// @details Creates an empty index that must be built before use.
  ///          The resource must be explicitly provided to ensure consistency with
  ///          the parent relation's memory management.
  explicit HashTrieIndex(memory_resource* resource);

  /// @brief Destructor.
  ~HashTrieIndex();

  // Non-copyable (to avoid accidental heavy copies / shared state confusion)
  HashTrieIndex(const HashTrieIndex&) = delete;
  HashTrieIndex& operator=(const HashTrieIndex&) = delete;

  // Movable (cheap handle moves)
  HashTrieIndex(HashTrieIndex&&) noexcept = default;
  HashTrieIndex& operator=(HashTrieIndex&&) noexcept = default;

  /// @brief Build the index from pre-encoded row data.
  /// @param spec Index specification defining which columns to index
  /// @param encoded_rows Array of spans, one per column, containing encoded
  ///                     values (size_t). Each span must have the same
  ///                     length.
  /// @param resource Optional memory resource for index storage
  /// @details This is the primary build method. It:
  ///          - Builds the trie structure incrementally from existing size
  ///          - Expects all values to be pre-encoded as size_t
  ///          - Requires encoded_rows[col_idx][row_id] format
  ///          - Only processes new rows since last build (incremental)
  /// @note The Relation class handles encoding and calls this method.
  ///       For direct usage, ensure data is pre-encoded.
  void build_from_encoded(const IndexSpec& spec,
                          const std::array<std::span<const ValueType>, arity>& encoded_rows,
                          boost::container::pmr::memory_resource* resource =
                              boost::container::pmr::get_default_resource());

  /// @brief Get a cursor to the root node of the trie.
  /// @return NodeHandle pointing to the root node (depth 0)
  /// @details The root node is always valid (if the index has been built).
  ///          Use this to start navigation from the beginning of the trie.
  NodeHandle root() const;

  /// @brief Look up a node by encoded key prefix.
  /// @tparam N Length of the key prefix (must match spec.cols.size() for
  ///            full key lookup)
  /// @param k Encoded key prefix to look up
  /// @return NodeHandle to the matching node, or invalid handle if not found
  /// @details This method efficiently navigates multiple levels of the trie
  ///          in one call. The key prefix can be:
  ///          - Full key (N == spec.cols.size()): Returns leaf node with rows
  ///          - Partial key (N < spec.cols.size()): Returns internal node
  ///          with
  ///            values for next level
  /// @note Returns invalid handle if any component of the key doesn't exist
  ///       in the trie.
  template <std::size_t N>
  NodeHandle prefix_lookup(const EncodedKeyPrefix<N>& k) const;

  /// @brief Get approximate memory footprint of the index.
  /// @return Approximate number of bytes used by the index structure
  /// @details This includes:
  ///          - Node storage
  ///          - Hash maps for children
  ///          - Key caches
  ///          - Posting lists
  /// @note This is an approximation and may not account for all PMR overhead.
  [[nodiscard]] std::size_t bytes_used() const;

  /// @brief Get the number of rows indexed.
  /// @return Total number of relation rows that have been inserted into this
  ///         index
  /// @details This tracks how many rows from the relation are currently
  ///          indexed. Used to determine if index needs rebuilding.
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }
  /// @brief Get the number of rows processed (same as size() for HashTrieIndex).
  /// @return Total number of relation rows that have been processed into this index
  /// @details For HashTrieIndex, this is the same as size() since it doesn't deduplicate.
  ///          Used to determine if index needs rebuilding.
  [[nodiscard]] std::size_t rows_processed() const noexcept {
    return size_;
  }

  // ---- Deep clone
  /// @brief Deep clone from another HashTrieIndex
  /// @details Performs a deep copy of the entire trie structure including all
  /// nodes,
  ///          children maps, keys_cache, and postings.
  /// @param other The source HashTrieIndex to clone from
  /// @param resource Optional memory resource for the cloned index (defaults
  /// to source's resource)
  void clone_from(const HashTrieIndex& other, memory_resource* resource = nullptr);

  /// @brief Clear the index.
  /// @details Clears the index and resets the size to 0.
  void clear() noexcept {
    p_->clear();
    size_ = 0;
  }

  /// @brief Check if the index is empty.
  /// @details Checks if the index is empty by checking if the size is 0.
  /// @return True if the index is empty, false otherwise.
  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  /// @brief Merge another HashTrieIndex into this index
  /// @details Merges the trie structure from another index into this index,
  ///          applying a row ID offset to the merged postings.
  /// @param other The source HashTrieIndex to merge from
  /// @param row_id_offset Offset to add to row IDs from the source index
  void merge(const HashTrieIndex& other, uint32_t row_id_offset);

  /**
   * @brief Generic Join Intersection (Scan-Probe).
   * @details Returns a View that scans the first handle and probes the rest.
   */
  template <typename FirstHandle, typename... OtherHandles>
  static auto intersect(const FirstHandle& first, const OtherHandles&... others);

 private:
  std::shared_ptr<Impl> p_;
  std::size_t size_{0};

  // Compile-time concept validation (SML/Racket-like module checking)
  // This static member ensures the concept is checked when the template is instantiated
  static constexpr bool concept_validated_ = []() {
    static_assert(IndexLike<HashTrieIndex<SR, AttrTuple>>,
                  "HashTrieIndex must satisfy IndexLike concept");
    static_assert(CNodeHandle<NodeHandle>,
                  "HashTrieIndex::NodeHandle must satisfy CNodeHandle concept");
    return true;
  }();
};

// Compile-time validation helper: ensures HashTrieIndex satisfies IndexLike concept
// This provides SML/Racket-like module/typeclass checking at compile time
// The variable template is instantiated when HashTrieIndex is used, triggering validation
template <Semiring SR, ColumnElementTuple AttrTuple>
inline constexpr bool is_valid_hash_trie_index_v = IndexLike<HashTrieIndex<SR, AttrTuple>>;

}  // namespace SRDatalog

// Include implementation file after namespace closes
// This ensures any Boost headers included by hashtrie.ipp are at global scope
#include "hashtrie.ipp"
