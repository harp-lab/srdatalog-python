#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "column.h"
#include "semiring.h"
#include "tmp.h"

// Forward declaration for memory_resource
namespace boost::container::pmr {
class memory_resource;
}
using memory_resource = boost::container::pmr::memory_resource;

namespace SRDatalog {

// Forward declaration
// Forward declaration
struct IndexSpec;

// Common type for encoded key prefixes (used by both Relation and HashTrieIndex)
template <std::size_t N>
using EncodedKeyPrefix = std::array<std::size_t, N>;

/// @brief Concept for NodeHandle types from index implementations.
/// @details Detects types that provide the NodeHandle interface for navigating
///          trie structures. This enables better IDE support and type checking
///          in generic code.
/// @note **C++20 feature**: Uses C++20 concepts with `requires` expressions to
///       check interface requirements, including `std::ranges::range` concept
///       for range-based operations.
/// @tparam T The type to check
template <typename T>
concept CNodeHandle = requires(const T& handle) {
  { handle.valid() } -> std::convertible_to<bool>;
  // Allow both CPU-style is_leaf() and GPU-style is_leaf(view)
  (
      requires {
        { handle.is_leaf() } -> std::convertible_to<bool>;
      } ||
      requires(const typename T::View& view) {
        { handle.is_leaf(view) } -> std::convertible_to<bool>;
      });
  { handle.depth() } -> std::convertible_to<uint32_t>;
  // Check for range-like operations (begin/end) instead of std::ranges::range
  // This works for both std::span, cuda::std::span, and other range types
  // Allow CPU-style values(), GPU-style values(view), OR indexed accessor get_value_at
  (
      requires {
        requires requires(decltype(handle.values()) r) {
          { r.begin() };
          { r.end() };
        };
      } ||
      requires(const typename T::View& view) {
        requires requires(decltype(handle.values(view)) r) {
          { r.begin() };
          { r.end() };
        };
      } ||
      // Alternative: indexed accessor pattern (for GPU LSM support)
      requires(T& h, const typename T::View& view, std::size_t idx) {
        { h.get_value_at(view, idx) };
      });
  // Allow both CPU-style rows() and GPU-style rows(view), OR omit rows() entirely
  // for GPU handles that use indexed accessors
  (
      requires {
        requires requires(decltype(handle.rows()) r) {
          { r.begin() };
          { r.end() };
        };
      } ||
      requires(const typename T::View& view) {
        requires requires(decltype(handle.rows(view)) r) {
          { r.begin() };
          { r.end() };
        };
      } ||
      // GPU handles without rows() method (use degree() + get_value_at instead)
      requires(const T& h) {
        { h.degree() } -> std::convertible_to<std::size_t>;
      });
  // prefix and contains_value should accept the value type from the values() range
  // For spans, use element_type; for other ranges, use range_value_t
  // Handle both CPU-style (values()) and GPU-style (values(view)), or indexed accessors
  requires(
      requires(const T& h) {
        // Try element_type first (for spans), then fall back to range_value_t
        requires(
            requires { typename std::remove_cvref_t<decltype(h.values())>::element_type; } ||
            requires { typename std::ranges::range_value_t<decltype(h.values())>; });
      } ||
      requires(const T& h, const typename T::View& view) {
        // GPU-style: get value type from values(view)
        requires(
            requires { typename std::remove_cvref_t<decltype(h.values(view))>::element_type; } ||
            requires { typename std::ranges::range_value_t<decltype(h.values(view))>; });
      } ||
      // Alternative: indexed accessor pattern - value type inferred from get_value_at
      requires(T& h, const typename T::View& view, std::size_t idx) {
        { h.get_value_at(view, idx) };
      });
  // Check prefix and contains_value with the appropriate value type
  // Allow both CPU-style prefix(key) and GPU-style template prefix(key, tile, view)
  // For GPU handles with template prefix, we can't fully validate in a concept
  // so we relax the check for device handles
  requires(
      // CPU-style: prefix(key) returns same type
      requires(const T& h, typename std::remove_cvref_t<decltype(h.values())>::element_type key) {
        { h.prefix(key) } -> std::same_as<T>;
        { h.contains_value(key) } -> std::convertible_to<bool>;
      } ||
      requires(const T& h, std::ranges::range_value_t<decltype(h.values())> key) {
        { h.prefix(key) } -> std::same_as<T>;
        { h.contains_value(key) } -> std::convertible_to<bool>;
      } ||
      // Alternative GPU-style check: has View type and prefix takes view
      requires(T& h, const typename T::View& view) {
        typename T::View;
        // Just check that View type exists, actual prefix validation is in CDeviceHandle
      });
  // degree() can return uint32_t or std::size_t
  { handle.degree() } -> std::convertible_to<std::size_t>;
  // unique_count returns the number of unique values (size of values() view)
  // Can return uint32_t or std::size_t
  { handle.unique_count() } -> std::convertible_to<std::size_t>;
};

/// @brief Device-only handle concept used by GPU kernels.
/// @details Refines CNodeHandle and requires accessors that expose raw device
///          pointers to root-level values and provenance data. This is the
///          interface expected by GPU kernels such as project_edge_to_path_*.
template <typename T>
concept CDeviceHandle = CNodeHandle<T> && requires(T& handle, const typename T::View& view) {
  // Pointer to provenance/semiring values (used at leaf level)
  { handle.provenance_ptr(view) };
};

/// @brief Concept for index-like types with trie navigation interface.
/// @note **C++20 feature**: Uses nested `requires` clauses (requires CNodeHandle)
///       and `requires` expressions to check for required types and operations.
/// @details This concept defines the interface that any index type must satisfy to be used
///          with Relation. It checks for:
///          - Required nested types (NodeHandle, RowId, RowRange, ValueRange)
///          - Required query methods (root(), bytes_used(), size(), clear(), empty(),
///          prefix_lookup())
///          - Required build methods (build() for constructing the index)
///          - Required merge method (merge() for merging another index into this one)
///          - NodeHandle must satisfy CNodeHandle concept
/// @note This is in the main namespace to allow for future index implementations beyond
/// HashTrieIndex
template <typename T>
concept IndexLike =
    requires(const T& idx, T& non_const_idx, memory_resource* resource) {
      typename T::NodeHandle;
      typename T::RowId;
      typename T::RowRange;
      typename T::ValueRange;
      // ValueRange should be a range (span or view) of some value type
      // For views, we check range_value_t instead of element_type
      requires std::ranges::range<typename T::ValueRange>;
      requires std::is_same_v < std::ranges::range_value_t<typename T::ValueRange>,
                   std::conditional_t < requires {typename T::ValueRange::element_type;
    },
        typename T::ValueRange::element_type,
        std::ranges::range_value_t < typename T::ValueRange >>>
    ;
requires CNodeHandle<typename T::NodeHandle>;
{idx.root()}->std::same_as<typename T::NodeHandle>;
{idx.bytes_used()}->std::convertible_to<std::size_t>;
{idx.size()}->std::convertible_to<std::size_t>;
{non_const_idx.clear()}->std::same_as<void>;
{idx.empty()}->std::convertible_to<bool>;
// Check that the Index type has a variadic static intersect method
// This is used for set intersection of columns in WCOJ
// Check with 2 arguments (minimum for intersection)
requires requires(const typename T::NodeHandle& h1, const typename T::NodeHandle& h2) {
  { T::intersect(h1, h2) } -> std::ranges::view;
};
// Also check with 3 arguments to verify variadic support
requires requires(const typename T::NodeHandle& h1, const typename T::NodeHandle& h2,
                  const typename T::NodeHandle& h3) {
  { T::intersect(h1, h2, h3) } -> std::ranges::view;
};
// Check prefix_lookup with different key sizes
// This is tricky, but I am not sure any other way to check if a class has a template method in
// C++20
requires requires { idx.template prefix_lookup<1>(std::array<std::size_t, 1>{}); };
// Check that build_from_encoded method exists (concrete signature, easier to check)
// This is the primary build method that accepts pre-encoded data
// Relation handles encoding and calls this method - this keeps Index independent
// Note: We check with a fixed arity (e.g., 2) to verify the method signature exists
// The actual arity will be checked at template instantiation time
// The encoded_rows type should match ValueRange (which is std::span<const ValueType>)
requires requires(
    T& idx_mut, const IndexSpec& spec,
    const std::array<std::span<const typename T::ValueRange::value_type>, 2>& encoded_rows,
    memory_resource* res) {
  { idx_mut.build_from_encoded(spec, encoded_rows, res) } -> std::same_as<void>;
};
// Check that merge method exists
// Merges another index into this index, applying a row ID offset to the merged postings
requires requires(T& idx_mut, const T& other_idx, std::size_t row_id_offset) {
  { idx_mut.merge(other_idx, row_id_offset) } -> std::same_as<void>;
};
};

/// @brief Type trait for identifying index implementations.
template <typename T>
struct is_index : std::bool_constant<IndexLike<T>> {};

/// @brief Convenience variable template for index detection.
template <typename T>
inline constexpr bool is_index_v = is_index<T>::value;

/// @brief Concept for index types supporting trie navigation.
template <typename T>
concept CIndex = is_index_v<T>;

/// @brief Helper to check if an index template type is valid.
/// @details Checks that a template class can be instantiated with SR and AttrTuple
///          and satisfies the IndexLike concept.
template <template <Semiring, ColumnElementTuple> class IndexType, Semiring SR,
          ColumnElementTuple AttrTuple>
inline constexpr bool is_index_template_v = IndexLike<IndexType<SR, AttrTuple>>;

/// @brief Concept for index template types (template template parameter).
/// @details Checks that a template class can be instantiated with SR and AttrTuple
///          and satisfies the IndexLike concept.
/// @note This is a type alias for use in requires clauses - the actual check is done via
/// is_index_template_v
template <template <Semiring, ColumnElementTuple> class IndexType, Semiring SR,
          ColumnElementTuple AttrTuple>
using CIndexTemplate = std::enable_if_t<is_index_template_v<IndexType, SR, AttrTuple>, bool>;

/**
 * @brief Specification for a multi-column index on a relation.
 *
 * @details IndexSpec defines which columns of a relation should be indexed
 * together. The columns form a composite key where values are concatenated
 * left-to-right. This is used to create HashTrieIndex structures for efficient
 * prefix lookups and joins.
 *
 * @section indexspec_structure Structure
 *
 * An IndexSpec contains:
 * - `cols`: A vector of column indices (0-based) that form the index key
 * - The order matters: columns are indexed in the order specified
 * - Duplicate column indices are allowed but generally not useful
 *
 * @section indexspec_usage Usage
 *
 * @code{.cpp}
 * // Create an index on the first column only
 * IndexSpec spec1;
 * spec1.cols = {0};
 *
 * // Create an index on columns 0 and 1 (composite key)
 * IndexSpec spec2;
 * spec2.cols = {0, 1};
 *
 * // Create an index on columns 1, 0 (different order = different index)
 * IndexSpec spec3;
 * spec3.cols = {1, 0};  // Note: spec2 != spec3
 *
 * // Use with Relation::ensure_index()
 * const auto& idx = rel.ensure_index(spec2);
 * @endcode
 *
 * @section indexspec_key_format Key Format
 *
 * The index key is formed by concatenating encoded values from the specified
 * columns in order:
 * - For IndexSpec with cols = {0, 1}, the key is (col0_value, col1_value)
 * - Prefix lookups work on prefixes of this key (e.g., just col0_value)
 * - Full key lookups require all components
 *
 * @section indexspec_examples Examples
 *
 * For a relation R(x: int, y: int, z: int):
 * - `{0}` indexes on column x only
 * - `{0, 1}` indexes on (x, y) composite key
 * - `{1, 2}` indexes on (y, z) composite key
 * - `{0, 1, 2}` indexes on all columns (full key)
 *
 * @section indexspec_comparison Comparison
 *
 * IndexSpec supports equality comparison:
 * - Two IndexSpecs are equal if their `cols` vectors are identical
 * - Order matters: {0, 1} != {1, 0}
 * - Can be used as keys in std::map and std::set
 *
 * @note Column indices must be valid (0 <= index < relation arity) when used
 *       to build an index. Invalid indices will cause errors during index
 *       construction.
 *
 * @see HashTrieIndex for the index structure created from this specification
 * @see Relation::ensure_index for building indexes from specifications
 * @see IndexSpecHash for use in unordered containers
 */
struct IndexSpec {
  /// @brief Column indices that form the index key (left-to-right order).
  /// @details Each integer is a 0-based column index. The values from these
  ///          columns are concatenated in order to form the composite key.
  ///          Empty vector is valid but creates an index with no key columns.
  std::vector<int> cols;

  /// @brief String representation of the index specification.
  /// @details Constructed when IndexSpec is built, representing the column
  ///          indices in a human-readable format (e.g., "[0,1,2]").
  mutable std::string str_;

  /// @brief Default constructor.
  IndexSpec() = default;

  /// @brief Constructor from initializer list of column indices.
  /// @param col_list Initializer list of column indices
  /// @details Constructs the IndexSpec and builds the string representation.
  IndexSpec(std::initializer_list<int> col_list) : cols(col_list) {
    // update_string();
  }

  /// @brief Copy constructor.
  /// @param other The IndexSpec to copy from
  IndexSpec(const IndexSpec& other) : cols(other.cols), str_(other.str_) {}

  /// @brief Move constructor.
  /// @param other The IndexSpec to move from
  IndexSpec(IndexSpec&& other) noexcept
      : cols(std::move(other.cols)), str_(std::move(other.str_)) {}

  /// @brief Copy assignment operator.
  /// @param other The IndexSpec to copy from
  /// @return Reference to this IndexSpec
  IndexSpec& operator=(const IndexSpec& other) {
    if (this != &other) {
      cols = other.cols;
      str_ = other.str_;
    }
    return *this;
  }

  /// @brief Move assignment operator.
  /// @param other The IndexSpec to move from
  /// @return Reference to this IndexSpec
  IndexSpec& operator=(IndexSpec&& other) noexcept {
    if (this != &other) {
      cols = std::move(other.cols);
      str_ = std::move(other.str_);
    }
    return *this;
  }

  /// @brief Destructor.
  ~IndexSpec() = default;

  /// @brief Update the string representation from the current cols vector.
  /// @details Should be called after modifying cols directly to keep str_ in sync.
  void update_string() const {
    str_.clear();
    str_.push_back('[');
    for (size_t i = 0; i < cols.size(); ++i) {
      if (i > 0) {
        str_.append(",");
      }
      str_.append(std::to_string(cols[i]));
    }
    str_.push_back(']');
  }

  /// @brief Get the string representation of this IndexSpec.
  /// @return String representation in format "[col0,col1,...]" (e.g., "[0,1,2]")
  /// @details Returns the cached string representation. If str_ is empty,
  ///          constructs it from cols first.
  std::string to_string() const {
    if (str_.empty() && !cols.empty()) {
      update_string();
    } else if (str_.empty()) {
      return "[]";
    }
    return str_;
  }

  /// @brief Equality comparison operator.
  /// @param o Other IndexSpec to compare
  /// @return true if both IndexSpecs have identical `cols` vectors
  /// @details Two IndexSpecs are equal if their column vectors match exactly
  ///          (same elements in same order).
  bool operator==(const IndexSpec& o) const noexcept {
    return cols == o.cols;
  }

  /// @brief Inequality comparison operator.
  /// @param o Other IndexSpec to compare
  /// @return true if IndexSpecs differ
  bool operator!=(const IndexSpec& o) const noexcept {
    return !(*this == o);
  }
};

/**
 * @brief Hash function for IndexSpec to enable use in unordered containers.
 *
 * @details IndexSpecHash provides a custom hash function that allows IndexSpec
 * to be used as a key in std::unordered_map and std::unordered_set. The hash
 * combines the size of the column vector and all column indices using a
 * FNV-inspired hash with golden ratio mixing.
 *
 * @section indexspechash_algorithm Algorithm
 *
 * The hash function:
 * - Starts with an FNV-like seed constant
 * - Mixes in the vector size
 * - Mixes in each column index with bit rotations and golden ratio constant
 * - Provides good distribution for typical index specifications
 *
 * @note This hash function is deterministic and satisfies the requirements
 *       for std::hash: if two IndexSpecs are equal (operator==), they must
 *       have the same hash value.
 *
 * @see IndexSpec for the type being hashed
 * @see std::unordered_map for usage in hash tables
 */
struct IndexSpecHash {
  /// @brief Compute hash value for an IndexSpec.
  /// @param s The IndexSpec to hash
  /// @return Hash value suitable for use in unordered containers
  /// @details Uses FNV-inspired hashing with golden ratio mixing for good
  ///          distribution. The hash combines the vector size and all column
  ///          indices.
  std::size_t operator()(const IndexSpec& s) const noexcept {
    // Start from a large odd constant; mix size and each element (like FNV +
    // golden ratio salt)
    std::size_t h = 1469598103934665603ull;  // FNV-ish seed
    h ^= s.cols.size() + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    for (int c : s.cols) {
      h ^= std::hash<int>{}(c) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return h;
  }
};

}  // namespace SRDatalog
