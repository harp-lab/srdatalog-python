#pragma once
// Include external library headers at global scope BEFORE entering namespace
// This prevents their internal code from looking in SRDatalog namespace

// Include Boost.Log headers at global scope
#ifdef ENABLE_LOGGING
#include "logging_boost.h"
#endif

// Include Highway headers at global scope
// Highway's vqsort.h needs float16_t and other types to be in global scope
// Must be included before any headers that use Highway (like sort.h)
#include <hwy/aligned_allocator.h>
#include <hwy/contrib/sort/vqsort.h>
#include <hwy/highway.h>

#include "column.h"
#include "index.h"     // IndexLike concept and related traits
#include "semiring.h"  // Semiring concept + semiring_value_t
#include "tmp.h"
// Include index headers - must be after Highway to avoid namespace issues
#include "hashtrie.h"
#include "sorted_array_index.h"
// Include GPU headers for device storage (when GPU compilation is enabled)
// Simplified: Check for RMM/hipMM (works for both CUDA and ROCm via gpu/macro.h)
// gpu/macro.h already defines SRDATALOG_GPU_AVAILABLE based on USE_CUDA/USE_ROCm
#include "gpu/index_concepts.h"  // Always include concepts/traits references
#include "gpu/macro.h"           // Provides SRDATALOG_GPU_AVAILABLE and platform detection

#if SRDATALOG_GPU_AVAILABLE
// GPU support available (CUDA or ROCm) - include GPU headers
// RMM (CUDA) or hipMM (ROCm) will be available via the build system
#include "gpu/aos_device_array.h"
#include "gpu/device_array.h"
#include "gpu/nd_device_array.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#else
// No GPU support - provide forward declarations
namespace SRDatalog::GPU {
template <typename T>
class DeviceArray;
template <typename T, std::size_t N>
class NDDeviceArray;
template <typename T, std::size_t N>
class AoSDeviceArray;
}  // namespace SRDatalog::GPU
#endif
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <ostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
// for std::hash
#include <concepts>
#include <functional>
#include <iostream>

// boost pmr includes
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <boost/container/pmr/vector.hpp>

#include <boost/unordered/unordered_map.hpp>

namespace SRDatalog {
/**
 * @file relation_col.h
 * @brief Defines the column-oriented Relation class and associated
 * views/indexes.
 *
 * @note **C++20/23 Features Used**: This file uses C++20 concepts extensively
 *       for type constraints, `std::ranges` for lazy evaluation and range
 *       operations, `std::bit_cast` for type-punning, `std::span` for
 *       non-owning range views, `if constexpr` for compile-time conditionals,
 *       `requires` clauses for template constraints, and fold expressions.
 *
 * @section relation_design Relation Design
 *
 * This file implements a column-oriented (SoA - Structure of Arrays) relation
 * storage system optimized for analytical Datalog workloads. The design
 * provides:
 *
 * @subsection column_oriented Column-Oriented Storage
 *
 * - **Attribute Columns**: Each attribute type is stored in a separate column
 *   (Column<T>), providing better cache locality for columnar operations
 * - **Annotation Column**: Semiring annotations are stored separately,
 *   enabling efficient aggregation and provenance tracking
 * - **Interned Columns**: Encoded (size_t) versions of attributes for
 *   efficient indexing and comparison. All values are converted to uniform
 *   size_t representation, enabling better join performance, easier distributed
 *   communication, and improved cache locality. See @ref codec_motivation for
 *   detailed rationale.
 *
 * @subsection indexing_strategy Indexing Strategy
 *
 * The system supports pluggable index implementations (default: HashTrieIndex):
 *
 * - **Pluggable Index Types**: Relation accepts any index type satisfying the
 *   IndexLike concept via template parameter (e.g., HashTrieIndex, BTreeIndex)
 * - **Prefix Indexes**: Indexes are built on prefixes of columns (e.g., {0,1}
 *   indexes the first two columns)
 * - **Lazy Building**: Indexes are built on-demand when first accessed
 * - **Automatic Invalidation**: Indexes are marked dirty on writes and
 *   rebuilt when needed
 * - **Default Implementation**: HashTrieIndex provides multi-level trie structure
 *   where each level corresponds to an indexed column, enabling efficient prefix lookups
 *
 * @subsection memory_management Memory Management
 *
 * All storage uses polymorphic memory resources (PMR):
 *
 * - Relations can be allocated on custom memory resources
 * - Enables arena allocation, memory pooling, and custom allocators
 * - Indexes share memory resources with their parent relations
 *
 * @subsection example_usage Example Usage
 *
 * @code{.cpp}
 * using namespace SRDatalog;
 * using namespace SRDatalog::AST::Literals;
 * using SR = BooleanSR;
 *
 * // Define a relation schema: R(x: int, y: int) with Boolean semiring
 * using RSchema = AST::RelationSchema<
 *     decltype("R"_s), SR, std::tuple<int, int>>;
 *
 * // Create relation with custom memory resource
 * Arena arena;
 * Relation<SR, std::tuple<int, int>> rel(&arena);
 *
 * // Add facts
 * rel.push_row({1, 2}, BooleanSR::one());
 * rel.push_row({2, 3}, BooleanSR::one());
 *
 * // Build index on first column
 * IndexSpec spec;
 * spec.cols = {0};
 * const auto& idx = rel.ensure_index(spec);
 *
 * // Query using index
 * auto node = rel.indexed_range_query(spec, {1});
 * for (auto row_id : node.rows()) {
 *   // Process matching rows
 * }
 * @endcode
 */

// memory_resource alias is defined in index.h (included above)

// helper function to make a tuple with a resource
namespace detail {
template <typename TupleType, std::size_t... Is>
auto make_tuple_with_resource_impl(boost::container::pmr::memory_resource* resource,
                                   std::index_sequence<Is...>) {
  return std::make_tuple(std::tuple_element_t<Is, TupleType>(resource)...);
}
}  // namespace detail

template <typename TupleType>
auto make_tuple_with_resource(boost::container::pmr::memory_resource* resource) {
  return detail::make_tuple_with_resource_impl<TupleType>(
      resource, std::make_index_sequence<std::tuple_size_v<TupleType>>{});
}
// CNodeHandle concept moved to index.h

/// @brief Type-safe key prefix for index lookups
/// @details Stores a heterogeneous sequence of encoded values.
///          Can be constructed from typed values which are automatically
///          encoded.
template <typename... Ts>
  requires(Codec<Ts> && ...)
struct Prefix {
  std::tuple<Ts...> comps;
  explicit Prefix(Ts... values) : comps(std::move(values)...) {}
  explicit Prefix(std::tuple<Ts...> tuple) : comps(std::move(tuple)) {}
  auto encoded() const {
    return std::apply([](const auto&... vals) { return std::array{encode_to_size_t(vals)...}; },
                      comps);
  }

  static constexpr std::size_t size() noexcept {
    return sizeof...(Ts);
  }
};

// Deduction guide get rid of make_tuple
template <typename... Ts>
Prefix(Ts...) -> Prefix<Ts...>;

/// @brief Concept for prefix types used in index lookups.
/// @note **C++20 feature**: Uses C++20 concepts with multiple `requires`
///       expressions to check both static members (size()) and instance
///       methods (encoded()) with type constraints.
template <typename T>
concept CPrefix = std::movable<T> &&  // Efficient passing
                  requires {
                    { T::size() } -> std::same_as<std::size_t>;
                  } && requires(const T& prefix) {
                    { prefix.encoded() } -> std::same_as<std::array<std::size_t, T::size()>>;
                    prefix.comps;
                  };

// EncodedKeyPrefix is defined in index.h (included above)

/**
 * @brief Column-oriented Relation class.
 *
 * @details This class implements a column-oriented relation storage system
 * optimized for Datalog evaluation. Key features:
 *
 * - **Column-oriented storage (SoA)**: Each attribute type is stored in a
 *   separate column, providing better cache locality for analytical workloads
 * - **Semiring annotations**: One annotation column stores semiring values
 *   (SR::value_type) for each row, enabling provenance tracking and aggregation
 * - **Interned columns**: Encoded versions of attributes (as size_t) for
 *   efficient indexing and comparison
 * - **Pluggable indexing**: Supports any index type satisfying IndexLike concept
 *   (default: HashTrieIndex for multi-level trie indexes)
 * - **Lazy index building**: Indexes are built on-demand when first accessed
 *
 * @tparam SR The semiring type used for the annotation (must satisfy Semiring
 *            concept)
 * @tparam AttrTuple The tuple type containing the attribute types (all elements
 *                   must satisfy ColumnElement concept)
 *
 * @section relation_structure Relation Structure
 *
 * A Relation consists of:
 * - `cols_`: Tuple of Column<T> objects, one per attribute
 * - `ann_`: Vector of annotation values (semiring values)
 * - `interned_cols_`: Array of encoded columns (Vector<size_t>)
 * - `indexes_`: Map of IndexSpec -> IndexType instances for efficient lookups
 *
 * @section indexing_usage Indexing Usage
 *
 * Indexes are specified using IndexSpec, which defines which columns form
 * the index key:
 *
 * @code{.cpp}
 * // Create index on columns 0 and 1
 * IndexSpec spec;
 * spec.cols = {0, 1};
 * const auto& idx = rel.ensure_index(spec);
 *
 * // Query using encoded prefix
 * auto prefix = Prefix<int, int>{5, 10};
 * auto node = rel.indexed_range_query(spec, prefix.encoded());
 * for (auto row_id : node.rows()) {
 *   // Process matching rows
 * }
 * @endcode
 *
 * @see IndexLike for the index interface requirements
 * @see HashTrieIndex for the default index implementation
 * @see IndexSpec for index specification format
 */

/// @brief Enum to select memory location for relation storage.
/// @details Determines whether relation data should be stored on host (CPU) or device (GPU)
/// memory.
enum class MemoryLocation {
  Host,   ///< Store relation data in host (CPU) memory
  Device  ///< Store relation data in device (GPU) memory
};

/// @brief Policy struct for configuring relation behavior.
/// @details Contains enums and settings that control how relations are allocated and managed.
///          Used as a template parameter to Relation to specify compile-time behavior.
/// @tparam MemoryLoc The memory location for relation storage (default: Host)
template <MemoryLocation MemoryLoc = MemoryLocation::Host>
struct RelationPolicy {
  /// @brief The memory location specified by this policy.
  static constexpr MemoryLocation memory_location = MemoryLoc;
};

/// @brief Default policy for host (CPU) memory relations.
using HostRelationPolicy = RelationPolicy<MemoryLocation::Host>;

/// @brief Policy for device (GPU) memory relations.
using DeviceRelationPolicy = RelationPolicy<MemoryLocation::Device>;

// Helper to select the appropriate index instantiation
namespace detail {
// Helper type alias that selects the appropriate instantiation
// Uses requires expression to check if ValueType and RowIdType can be passed and satisfies
// IndexLike
template <template <Semiring, ColumnElementTuple, typename...> class IndexType, Semiring SR,
          ColumnElementTuple AttrTuple, typename ValueType, typename RowIdType>
using SelectIndexType = std::conditional_t<requires {
  typename IndexType<SR, AttrTuple, ValueType, RowIdType>;
}, IndexType<SR, AttrTuple, ValueType, RowIdType>, IndexType<SR, AttrTuple>>;

// ============================================================
// Storage Traits for Hybrid Model (C++20 Concepts)
// ============================================================

/// @brief C++20 concept to check if policy is host-based
template <typename Policy>
concept IsHostPolicy = requires {
  { Policy::memory_location } -> std::same_as<const MemoryLocation&>;
  requires Policy::memory_location == MemoryLocation::Host;
};

/// @brief C++20 concept to check if policy is device-based
template <typename Policy>
concept IsDevicePolicy = requires {
  { Policy::memory_location } -> std::same_as<const MemoryLocation&>;
  requires Policy::memory_location == MemoryLocation::Device;
};

/// @brief Storage traits using C++20 concepts
/// @details Provides type aliases for storage types based on Policy
template <typename Policy>
struct RelationStorageTraits;

/// @brief Storage traits for host relations
template <IsHostPolicy Policy>
struct RelationStorageTraits<Policy> {
  template <typename T>
  using ColumnStorage = Column<T>;

  template <typename T>
  using VectorStorage = Vector<T>;

  using MemoryResource = memory_resource*;

  static constexpr bool is_device = false;
};

/// @brief Storage traits for device relations
template <IsDevicePolicy Policy>
struct RelationStorageTraits<Policy> {
  // Check if GPU support is available at compile time
  static_assert(SRDATALOG_GPU_AVAILABLE == 1,
                "DeviceRelationPolicy requires GPU support. Compile with CUDA (--nvidia=y) or "
                "ensure RMM (RAPIDS Memory Manager) is available.");

  template <typename T, std::size_t N>
  using ColumnStorage = GPU::NDDeviceArray<T, N>;

  template <typename T>
  using VectorStorage = GPU::DeviceArray<T>;

  using MemoryResource = void;  // No PMR for device

  static constexpr bool is_device = true;
};

}  // namespace detail

// Helper storage struct for conditional storage

/// @brief Column-oriented Relation class with pluggable index types.
/// @details The IndexType template parameter allows using different index
///          implementations. Any type satisfying the IndexLike concept can be used.
///          Default is HashTrieIndex, but you can provide custom implementations
///          like BTreeIndex, RadixTrieIndex, etc.
///          For SortedArrayIndex, ValueType and RowIdType can be specified to
///          control the internal storage types.
/// @tparam SR The semiring type
/// @tparam AttrTuple The attribute tuple type
/// @tparam Policy The relation policy specifying memory location and other behaviors.
///                 Default is HostRelationPolicy (host memory).
/// @tparam IndexType Template template parameter for the index type. Must satisfy
///                   IndexLike concept when instantiated with SR and AttrTuple.
///                   Can accept variadic parameters (e.g., SortedArrayIndex with
///                   ValueType/RowIdType). Default is HashTrieIndex.
/// @tparam ValueType Type for encoded values in SortedArrayIndex (default: std::size_t)
/// @tparam RowIdType Type for row IDs in SortedArrayIndex (default: uint32_t)
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType = HashTrieIndex,
          typename Policy = HostRelationPolicy, typename ValueType = uint32_t,
          typename RowIdType = uint32_t>
class Relation {
  // Friend declaration removed in favor of unsafe accessors

 public:
  using semiring_type = SR;  // For has_provenance_v detection
  using AnnotationVal = typename SR::value_type;
  using RowId = std::size_t;
  using value_type = ValueType;  // Expose ValueType for use in io.h and other generic code
  // For all index types, pass ValueType and RowIdType if they support it
  // Uses concept-based detection: any Index type that can accept these parameters will get them
  using IndexTypeInst = detail::SelectIndexType<IndexType, SR, AttrTuple, ValueType, RowIdType>;

  static constexpr std::size_t arity = std::tuple_size_v<AttrTuple>;

  // Determine storage layout from Index Traits (Default SoA)
  using DbIndexTraits = GPU::IndexStorageTraits<IndexTypeInst>;
  static constexpr GPU::StorageLayout Layout = DbIndexTraits::layout;

  // Define Device Storage Type
  using DeviceColsType =
      std::conditional_t<Layout == GPU::StorageLayout::AoS, GPU::AoSDeviceArray<ValueType, arity>,
                         GPU::NDDeviceArray<ValueType, arity>>;

  using StorageTraits = detail::RelationStorageTraits<Policy>;

  // constructor
  // Note: Default version is 10 (UNKNOWN_VER) to avoid circular dependency with ast.h
  Relation() : Relation(10) {}

  // Default constructor - uses if constexpr internally
  explicit Relation(std::size_t version) : version_(version) {
    if constexpr (StorageTraits::is_device) {
      // Initialize device storage (ONLY encoded data as uint32_t)
      // storage_ union constructor handles initialization
    } else {
      // Initialize host storage (both original and encoded columns)
      host_storage().cols_ =
          make_tuple_with_resource<ColumnTuple<AttrTuple>>(default_memory_resource());
      host_storage().resource_ = default_memory_resource();
    }
  }

  // Host constructor - only available when Policy is HostRelationPolicy
  template <typename P = Policy>
  explicit Relation(memory_resource* resource, std::size_t version = 10)
    requires(!detail::IsDevicePolicy<P>)
      : version_(version) {
    host_storage().cols_ = make_tuple_with_resource<ColumnTuple<AttrTuple>>(resource);
    host_storage().ann_ = Vector<AnnotationVal>(resource);
    host_storage().resource_ = resource ? resource : default_memory_resource();
  }

  // Device constructor - only available when Policy is DeviceRelationPolicy
  template <typename P = Policy>
  explicit Relation(std::size_t version = 10)
    requires detail::IsDevicePolicy<P>
      : version_(version) {
    // Device storage is initialized by union constructor
  }

  Relation(std::string name, memory_resource* resource, std::size_t version = 10)
    requires(!StorageTraits::is_device)
      : name_(std::move(name)), version_(version) {
    host_storage().cols_ = make_tuple_with_resource<ColumnTuple<AttrTuple>>(resource);
    host_storage().ann_ = Vector<AnnotationVal>(resource);
    host_storage().resource_ = resource ? resource : default_memory_resource();
  }

  explicit Relation(std::string name, std::size_t version = 10)
    requires(!StorageTraits::is_device)
      : name_(std::move(name)), version_(version) {
    host_storage().cols_ =
        make_tuple_with_resource<ColumnTuple<AttrTuple>>(default_memory_resource());
    host_storage().resource_ = default_memory_resource();
  }

  // Copy constructor - DELETED to prevent accidental expensive copies.
  // Relations are large database tables that can contain millions of rows.
  // Use clone_into() for explicit copying when needed, or move operations
  // for efficient transfer of ownership.
  Relation(const Relation& other) = delete;

  // Move constructor
  Relation(Relation&& other) noexcept
      : col_names_(std::move(other.col_names_)), name_(std::move(other.name_)),
        version_(other.version_), index_specs_(std::move(other.index_specs_)),
        indexes_(std::move(other.indexes_)) {
    if constexpr (StorageTraits::is_device) {
      new (&storage_.device_) decltype(storage_.device_){};
      storage_.device_.device_interned_cols_ =
          std::move(other.storage_.device_.device_interned_cols_);
      storage_.device_.device_ann_ = std::move(other.storage_.device_.device_ann_);
    } else {
      new (&storage_.host_) decltype(storage_.host_){};
      storage_.host_.cols_ = std::move(other.storage_.host_.cols_);
      storage_.host_.interned_cols_ = std::move(other.storage_.host_.interned_cols_);
      storage_.host_.ann_ = std::move(other.storage_.host_.ann_);
      storage_.host_.resource_ = other.storage_.host_.resource_;
    }
  }

  Relation& operator=(const Relation& other) = delete;

  // Move assignment
  Relation& operator=(Relation&& other) noexcept {
    if (this != &other) {
      col_names_ = std::move(other.col_names_);
      name_ = std::move(other.name_);
      version_ = other.version_;
      index_specs_ = std::move(other.index_specs_);
      indexes_ = std::move(other.indexes_);

      if constexpr (StorageTraits::is_device) {
        device_storage().device_interned_cols_ =
            std::move(other.device_storage().device_interned_cols_);
        device_storage().device_ann_ = std::move(other.device_storage().device_ann_);
      } else {
        host_storage().cols_ = std::move(other.host_storage().cols_);
        host_storage().interned_cols_ = std::move(other.host_storage().interned_cols_);
        host_storage().ann_ = std::move(other.host_storage().ann_);
        host_storage().resource_ = other.host_storage().resource_;
      }
    }
    return *this;
  }

  /// @brief Create a host relation from a device relation.
  /// @details Copies encoded data from device to host, converting uint32_t to size_t.
  ///          Note: Original columns are not available on device, so the host relation
  ///          will only have encoded columns (interned_cols_), not original columns (cols_).
  /// @tparam DeviceIndexType The index type of the device relation
  /// @tparam DevicePolicy The policy type of the device relation (must be DeviceRelationPolicy)
  /// @tparam DeviceValueType ValueType of the device relation
  /// @tparam DeviceRowIdType RowIdType of the device relation
  /// @param device_rel The device relation to convert
  /// @param resource Optional memory resource for the host relation (default:
  /// default_memory_resource)
  /// @return A new host relation with copied data
  template <template <typename, typename, typename...> class DeviceIndexType, typename DevicePolicy,
            typename DeviceValueType = uint32_t, typename DeviceRowIdType = uint32_t>
  static Relation<SR, AttrTuple, IndexType, HostRelationPolicy, ValueType, RowIdType> to_host(
      const Relation<SR, AttrTuple, DeviceIndexType, DevicePolicy, DeviceValueType,
                     DeviceRowIdType>& device_rel,
      memory_resource* resource = nullptr);

  /// @brief Create a device relation from a host relation.
  /// @details Copies encoded data from host to device, converting size_t to uint32_t.
  ///          Original columns are not copied (device relations only store encoded data).
  /// @tparam HostIndexType The index type of the host relation
  /// @tparam HostPolicy The policy type of the host relation (must be HostRelationPolicy)
  /// @tparam HostValueType ValueType of the host relation
  /// @tparam HostRowIdType RowIdType of the host relation
  /// @param host_rel The host relation to convert
  /// @return A new device relation with copied data
  template <template <typename, typename, typename...> class HostIndexType, typename HostPolicy,
            typename HostValueType = uint32_t, typename HostRowIdType = uint32_t>
  static Relation<SR, AttrTuple, IndexType, DeviceRelationPolicy, ValueType, RowIdType> to_device(
      const Relation<SR, AttrTuple, HostIndexType, HostPolicy, HostValueType, HostRowIdType>&
          host_rel);

  /// @brief Get the name of the relation.
  /// @details Returns the relation name for debugging purposes.
  /// @return A const reference to the relation name string.
  [[nodiscard]] const std::string& name() const noexcept {
    return name_;
  }

  /// @brief Set the name of the relation.
  /// @param name The new relation name.
  void set_name(const std::string& name) {
    name_ = name;
  }

  /// @brief Get the version of the relation.
  /// @details Returns the relation version (FULL_VER, DELTA_VER, or NEW_VER).
  /// @return The version number.
  std::size_t version() const noexcept {
    return version_;
  }

  /// @brief Set the version of the relation.
  /// @param version The new version number.
  void set_version(std::size_t version) {
    version_ = version;
  }

  /// @brief Get the names of the columns.
  /// @details This can be used to match the column names during debugging.
  /// @return A const reference to the vector of column names.
  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept {
    return col_names_;
  }
  /// @brief Set the names of the columns.
  /// @param names The new column names.
  /// @throw std::invalid_argument if the size of the names vector is not equal
  /// to the arity of the relation.
  void set_column_names(std::vector<std::string> names) {
    if (!names.empty() && names.size() != arity) {
      throw std::invalid_argument("set_column_names: size must be 0 or == arity");
    }
    col_names_ = std::move(names);
  }

  /// @brief Get the size of the relation.
  /// @return The number of rows in the relation.
  /// @note **C++17/C++20 feature**: Uses `if constexpr` for compile-time
  ///       conditional compilation based on relation arity and storage type.
  [[nodiscard]] std::size_t size() const noexcept {
    if constexpr (StorageTraits::is_device) {
      // Device: Size is based on encoded columns (only storage)
      return device_storage().device_interned_cols_.num_rows();
    } else {
      // Host: Size is max of original or encoded columns
      if constexpr (arity == 0)
        return host_storage().ann_.size();
      return std::max(std::get<0>(host_storage().interned_cols_).size(),
                      std::get<0>(host_storage().cols_).size());
    }
  }

  /// @brief Get the size of the interned relation.
  /// @return The number of rows in the interned relation.
  [[nodiscard]] std::size_t interned_size() const noexcept {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_interned_cols_.num_rows();
    } else {
      return std::get<0>(host_storage().interned_cols_).size();
    }
  }
  /// @brief Reserve capacity for the relation.
  /// @details This is a function when you need bulk operations on the relation.
  ///          It will more useful when used together with SIMT style
  ///          parallelism.
  /// @param n The number of rows to reserve capacity for.
  void reserve(std::size_t n) {
    if constexpr (StorageTraits::is_device) {
      device_storage().device_interned_cols_.reserve(n);
      if constexpr (has_provenance_v<SR>) {
        device_storage().device_ann_.reserve(n);
      }
    } else {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((host_storage().interned_cols_[Is].reserve(n),
          std::get<Is>(host_storage().cols_).reserve(n)),
         ...);
      }(std::make_index_sequence<arity>{});
    }
  }

  /// @brief stride of the interned columns
  std::size_t interned_stride() const noexcept {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_interned_cols_.stride();
    } else {
      return host_storage().interned_cols_[0].size();
    }
  }

  void reserve_interned(std::size_t n) {
    if constexpr (StorageTraits::is_device) {
      device_storage().device_interned_cols_.reserve(n);
    } else {
      for (auto& col : host_storage().interned_cols_) {
        col.reserve(n);
      }
    }
  }

  /// @brief Concatenate another relation to this relation.
  /// @details Appends all rows from the other relation to this relation.
  ///          For device relations: only concatenates encoded data.
  ///          The operation reserves capacity for efficiency and preserves
  ///          the order of rows. Indexes are not automatically rebuilt after
  ///          concatenation.
  /// @param other The relation to concatenate from (source remains unchanged)
  template <typename OtherRelation>
  void concat(const OtherRelation& other) {
    static_assert(std::is_same_v<typename OtherRelation::AnnotationVal, AnnotationVal>,
                  "Relations must have the same semiring type");
    static_assert(OtherRelation::arity == arity, "Relations must have the same arity");

    if constexpr (StorageTraits::is_device) {
      if constexpr (OtherRelation::StorageTraits::is_device) {
        // Device to device: Only concat encoded data
        device_storage().device_interned_cols_.concat(other.unsafe_interned_columns());
        if constexpr (has_provenance_v<SR>) {
          device_storage().device_ann_.concat(other.provenance());
        }
      } else {
        // std::array<std::span<const std::size_t>, arity> host_encoded;
        // [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //   ((host_encoded[Is] = std::span(other.template interned_column<Is>().data(),
        //                                  other.template interned_column<Is>().size())),
        //    ...);
        // }(std::make_index_sequence<arity>{});

        throw std::runtime_error(
            "Host-to-device concat not yet implemented. Use transfer utilities.");
      }
    } else {
      static_assert(!OtherRelation::StorageTraits::is_device || !StorageTraits::is_device,
                    "Cross-storage concat (host<->device) not yet implemented");

      auto other_size = other.size();
      host_storage().ann_.reserve(host_storage().ann_.size() + other_size);
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(host_storage().cols_).concat(other.template column<Is>())), ...);
      }(std::make_index_sequence<arity>{});
      // also concat the interned columns
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((host_storage().interned_cols_[Is].insert(host_storage().interned_cols_[Is].end(),
                                                   other.template interned_column<Is>().begin(),
                                                   other.template interned_column<Is>().end())),
         ...);
      }(std::make_index_sequence<arity>{});
      host_storage().ann_.insert(host_storage().ann_.end(), other.provenance().begin(),
                                 other.provenance().end());
    }
  }

  /// @brief Clear the relation.
  /// @details This will clear the relation and mark the indexes as dirty.
  void clear();

  // ---------- Row appends (write path) ----------

  /// @brief Push a row into the relation.
  /// @details This will push a row into the relation, encode values to interned
  /// columns, and mark the indexes as dirty.
  /// @param tup The tuple of attributes to push.
  /// @param a The annotation value to push.
  /// @note For device relations: encodes on host as size_t, converts to uint32_t for device
  void push_row(const AttrTuple& tup, const AnnotationVal& a) {
    if constexpr (StorageTraits::is_device) {
      std::array<std::size_t, arity> encoded_host;

      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((encoded_host[Is] = encode_to_size_t(std::get<Is>(tup))), ...);
      }(std::make_index_sequence<arity>{});

      std::array<uint32_t, arity> encoded_device;
      std::transform(encoded_host.begin(), encoded_host.end(), encoded_device.begin(),
                     [](std::size_t val) { return static_cast<uint32_t>(val); });

      if constexpr (Layout == GPU::StorageLayout::AoS) {
        device_storage().device_interned_cols_.push_back(encoded_device);
      } else {
        device_storage().device_interned_cols_.push_back(encoded_device);
      }
      if constexpr (has_provenance_v<SR>) {
        device_storage().device_ann_.push_back(a);
      }

      // Note: Original typed columns are NOT stored on device
    } else {
      if constexpr (has_provenance_v<SR>) {
        host_storage().ann_.push_back(a);
      }
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(host_storage().cols_).push_back(std::get<Is>(tup))), ...);
        ((host_storage().interned_cols_[Is].push_back(encode_to_size_t(std::get<Is>(tup)))), ...);
      }(std::make_index_sequence<arity>{});
    }
  }

  /// @brief Push a row without provenance (for NoProvenance semiring).
  /// @details This method is used when has_provenance_v<SR> is false.
  /// @param tup The tuple of attributes to push.
  void push_row_no_prov(const AttrTuple& tup) {
    if constexpr (StorageTraits::is_device) {
      std::array<std::size_t, arity> encoded_host;

      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((encoded_host[Is] = encode_to_size_t(std::get<Is>(tup))), ...);
      }(std::make_index_sequence<arity>{});

      std::array<uint32_t, arity> encoded_device;
      std::transform(encoded_host.begin(), encoded_host.end(), encoded_device.begin(),
                     [](std::size_t val) { return static_cast<uint32_t>(val); });

      device_storage().device_interned_cols_.push_back(encoded_device);
      // No provenance for NoProvenance semiring
    } else {
      // No provenance for NoProvenance semiring
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(host_storage().cols_).push_back(std::get<Is>(tup))), ...);
        ((host_storage().interned_cols_[Is].push_back(encode_to_size_t(std::get<Is>(tup)))), ...);
      }(std::make_index_sequence<arity>{});
    }
  }

  /// @brief Push a row with pre-encoded interned values.
  /// @details This will push a row into the relation using pre-encoded values.
  /// For device relations: converts ValueType to uint32_t during push.
  /// @param interned_tup The tuple of encoded (ValueType) attribute values to push.
  /// @param a The annotation value to push.
  void push_intern_row(const std::array<ValueType, arity>& interned_tup, const AnnotationVal& a) {
    if constexpr (StorageTraits::is_device) {
      // Device: Convert ValueType to uint32_t and push
      std::array<uint32_t, arity> encoded_device;
      std::transform(interned_tup.begin(), interned_tup.end(), encoded_device.begin(),
                     [](ValueType val) { return static_cast<uint32_t>(val); });

      if constexpr (Layout == GPU::StorageLayout::AoS) {
        device_storage().device_interned_cols_.push_back(encoded_device);
      } else {
        device_storage().device_interned_cols_.push_back(encoded_device.data());
      }
      if constexpr (has_provenance_v<SR>) {
        device_storage().device_ann_.push_back(a);
      }
    } else {
      // Host: Push encoded values directly (as size_t)
      if constexpr (has_provenance_v<SR>) {
        host_storage().ann_.push_back(a);
      }
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((host_storage().interned_cols_[Is].push_back(interned_tup[Is])), ...);
      }(std::make_index_sequence<arity>{});
    }
  }

  /// @brief Reconstruct column values from interned values.
  /// @details This function decodes all interned columns back to their original
  /// column values. This is useful when you want to reconstruct columns from
  /// encoded data (e.g., after loading from serialized format).
  /// @note Device relations don't store original columns - this is host-only
  /// @note This overwrites existing column values with decoded values from
  /// interned columns. Use with caution if columns and interned columns are
  /// out of sync.
  void reconstruct_columns_from_interned() {
    std::size_t n = interned_size();

    if constexpr (StorageTraits::is_device) {
#if SRDATALOG_GPU_AVAILABLE == 1
      // Device: First fully copy encoded data from device to host, then decode on host
      std::array<std::vector<uint32_t>, arity> host_encoded_cols;
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (
            [&]<std::size_t I>() {
              host_encoded_cols[I].resize(n);
              // Copy from device column to host vector using thrust
              auto device_col_ptr = device_storage().device_interned_cols_.template column_ptr<I>();
              thrust::copy_n(thrust::device, device_col_ptr, n, host_encoded_cols[I].data());
            }.template operator()<Is>(),
            ...);
      }(std::make_index_sequence<arity>{});
      throw std::runtime_error("reconstruct_columns_from_interned() on device relations: Data "
                               "copied to host and decoded, "
                               "but device relations don't store original columns. Use host "
                               "relations if you need this functionality.");
#else
      // GPU not available - this shouldn't be called
      static_assert(StorageTraits::is_device == false, "Device relations require GPU support");
#endif
    } else {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (
            [&]<std::size_t I>() {
              using ColType = std::tuple_element_t<I, AttrTuple>;
              auto& col = std::get<I>(host_storage().cols_);
              const auto& interned_col = host_storage().interned_cols_[I];
              col.clear();
              col.reserve(n);
              for (std::size_t r = 0; r < n; ++r) {
                if constexpr (std::is_same_v<ColType, std::string>) {
                  col.push_back(decode_from_size_t(interned_col[r]));
                } else {
                  col.push_back(decode_from_size_t<ColType>(interned_col[r]));
                }
              }
            }.template operator()<Is>(),
            ...);
      }(std::make_index_sequence<arity>{});
    }
  }

  // @brief get one index (any one from all indexes), is not return canonical index.
  /// @details Lazily initializes the canonical index if no indexes exist yet.
  /// @return A const reference to the index.
  [[nodiscard]] const IndexSpec& get_default_index() {
    // generate canonical index
    if (indexes_.empty()) {
      IndexSpec spec;
      spec.cols.reserve(arity);
      for (std::size_t i = 0; i < arity; ++i) {
        spec.cols.push_back(static_cast<int>(i));
      }

      // pmr awareness support (not avalible on device)
      if constexpr (StorageTraits::is_device) {
        // Device: Indexes don't take memory_resource
        indexes_.emplace(std::move(spec), IndexTypeInst{});
      } else {
        if constexpr (requires { IndexTypeInst{host_storage().resource_}; }) {
          indexes_.emplace(std::move(spec), IndexTypeInst{host_storage().resource_});
        } else {
          indexes_.emplace(std::move(spec), IndexTypeInst{});
        }
      }
    }
    return indexes_.begin()->first;
  }

  // @brief get all indexes
  /// @return A const reference to the vector of indexes.
  [[nodiscard]] const std::vector<IndexSpec>& get_all_indexes() const {
    return index_specs_;
  }

  /// @brief Get a reference to the column at index I.
  /// @details Example:
  ///          auto& col = R.column<0>(); // get the first column
  ///          const auto& col = R.column<0>(); // get the first column const
  ///          reference
  /// @note Device relations don't store original columns - this throws for device relations
  /// @param I The index of the column to get.
  /// @return A reference to the column at index I.
  template <std::size_t I>
    requires(!StorageTraits::is_device)
  Column<std::tuple_element_t<I, AttrTuple>>& column() {
    return std::get<I>(host_storage().cols_);
  }

  template <std::size_t I>
    requires(!StorageTraits::is_device)
  const Column<std::tuple_element_t<I, AttrTuple>>& column() const {
    return std::get<I>(host_storage().cols_);
  }

  // Device column access - NOT AVAILABLE (device only stores encoded data)
  template <std::size_t I>
    requires StorageTraits::is_device
  auto column() {
    throw std::runtime_error("Device relations don't store original columns. Use interned_column() "
                             "to access encoded data.");
  }

  template <std::size_t I>
    requires StorageTraits::is_device
  auto column() const {
    throw std::runtime_error("Device relations don't store original columns. Use interned_column() "
                             "to access encoded data.");
  }

  /// @brief Get a reference to the interned column at index I.
  /// @details Interned columns store encoded values:
  ///          - Host: Vector<ValueType> (encoded as ValueType)
  ///          - Device: uint32_t* (device pointer to encoded data as uint32_t)
  ///          The encoding is typically done via encode_to_size_t().
  /// @tparam I The index of the column to get (must be < arity)
  /// @return Host: reference to Vector<ValueType>, Device: uint32_t* (device pointer)
  template <std::size_t I>
    requires(!StorageTraits::is_device)
  Vector<ValueType>& interned_column() {
    return host_storage().interned_cols_[I];
  }

  template <std::size_t I>
    requires(!StorageTraits::is_device)
  [[nodiscard]] const Vector<ValueType>& interned_column() const {
    return host_storage().interned_cols_[I];
  }

  template <std::size_t I>
    requires StorageTraits::is_device
  auto interned_column() {
    if constexpr (Layout == GPU::StorageLayout::AoS) {
      // AoS storage doesn't support separate column pointers
      throw std::runtime_error("interned_column<I>() not supported for AoS layout. Use "
                               "unsafe_interned_columns() to access raw AoS data.");
      return (uint32_t*)nullptr;  // Valid return type for signature deduction
    } else {
      // Returns uint32_t* (device pointer to encoded data)
      return storage_.device_.device_interned_cols_.template column_ptr<I>();
    }
  }

  template <std::size_t I>
    requires StorageTraits::is_device
  auto interned_column() const {
    if constexpr (Layout == GPU::StorageLayout::AoS) {
      throw std::runtime_error("interned_column<I>() not supported for AoS layout.");
      return (const uint32_t*)nullptr;
    } else {
      return storage_.device_.device_interned_cols_.template column_ptr<I>();
    }
  }

  /// @brief Resize all interned columns to the specified size.
  /// @details This function resizes all interned column vectors to size n.
  ///          If n is larger than the current size, new elements are
  ///          default-constructed (typically 0 for size_t). If n is smaller,
  ///          elements are removed from the end.
  /// @param n The new size for all interned columns
  void resize_interned_columns(std::size_t n) {
    if constexpr (StorageTraits::is_device) {
      device_storage().device_interned_cols_.resize(n);
    } else {
      for (auto& col : host_storage().interned_cols_) {
        col.resize(n);
      }
    }
  }

  /// @brief Resize all interned columns on a specific stream (stream-ordered)
#if SRDATALOG_GPU_AVAILABLE == 1
  void resize_interned_columns(std::size_t n, GPU_STREAM_T stream) {
    if constexpr (StorageTraits::is_device) {
      device_storage().device_interned_cols_.resize(n, stream);
    } else {
      // CPU path: no stream, just resize
      for (auto& col : host_storage().interned_cols_) {
        col.resize(n);
      }
    }
  }
#endif

  /// @brief Get all provenance values
  /// @details This is for debugging purposes. In real code provenance needs to
  /// be computed by
  ///          query specific tuple.
  /// @return A const reference to the vector of provenance values (host) or device array (device).
  /// @return A reference to the vector of provenance values (host) or device array (device).
  const auto& provenance() const noexcept {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_ann_;
    } else {
      return host_storage().ann_;
    }
  }
  auto& provenance() noexcept {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_ann_;
    } else {
      return host_storage().ann_;
    }
  }

  /// @brief Print the first n rows of the relation.
  /// @details Debugging purposes. print the column names and the rows to
  ///          the output stream or files.
  /// @param n The number of rows to print.
  /// @param os The output stream to print to.
  void head(std::size_t n, std::ostream& os) const;

  /// @brief Print the first n rows of the relation's interned (encoded) values.
  /// @details Debugging purposes. Print the encoded intern values instead of
  ///          decoded column values. For device relations, this copies data
  ///          from device to host before printing.
  /// @param n The number of rows to print.
  /// @param os The output stream to print to.
  void head_interned(std::size_t n, std::ostream& os) const;

  /// @brief Project the relation onto a set of columns.
  /// @details This projection is lazy and will not materialize the rows until
  /// the result is used.
  ///          Example:
  ///          auto pv = R.project<0, 1>(); // (id, name, ann)
  ///          // materialize first 5 rows of the projection
  ///          for (auto&& [id, name, ann] : pv | std::ranges::views::take(5)) {
  ///            std::cout << "  ( " << id << ", " << name << " ), " << ann <<
  ///            "\n";
  ///          }
  /// @note **C++20 features**: Uses `std::span` for efficient range views and
  ///       `std::ranges::views` for lazy evaluation. The projection returns a
  ///       lazy view that materializes rows on-demand.
  /// @param row_ids: the span of row IDs to project onto.
  /// @return A new relation with the projected rows.
  // template <int... Idx>
  // auto project(const std::span<const RowId>& row_ids = {}) const;

  /// @brief RowRange is a view over a range of row IDs.
  /// @details This is used for range queries on the index,
  ///          fetching all tuples that match the range.
  /// @note **C++20 feature**: Uses `std::span` for efficient non-owning range
  ///       views. Provides zero-cost abstraction for accessing contiguous
  ///       sequences without copying.
  using RowRange = std::span<const RowId>;  // view over RowId[]
  /// @brief ValueRange is a view over encoded next-level keys.
  /// @note **C++20 feature**: Uses `std::span` for efficient non-owning range
  ///       views.
  using ValueRange = std::span<const std::size_t>;  // view over encoded next-level keys

  // ============================================================
  // Index management on Relation, eager building and range query
  // ============================================================

  // @brief Get the index for a given specification. This will ensure the index
  // is built and fresh.
  /// @param spec The index specification.
  /// @return A const reference to the index.
  const IndexTypeInst& ensure_index(const IndexSpec& spec, bool build = true) const;
  // @brief Get the index for a given specification. don't build the index.
  /// @param spec The index specification.
  /// @return A const reference to the index.
  const IndexTypeInst& get_index(const IndexSpec& spec) const {
    if (indexes_.find(spec) == indexes_.end()) {
      std::cout << "Relation " << name() << " get_index: NOT FOUND! Looking for "
                << spec.to_string() << std::endl;
      std::cout << "Available keys:" << std::endl;
      for (const auto& [k, v] : indexes_) {
        std::cout << "  " << k.to_string() << std::endl;
      }
    }
    return indexes_.at(spec);
  }

  // @brief Get the index for a given specification (non-const version).
  /// @param spec The index specification.
  /// @return A non-const reference to the index.
  IndexTypeInst& get_index(const IndexSpec& spec) {
    if (indexes_.find(spec) == indexes_.end()) {
      std::cout << "Relation " << name() << " get_index: NOT FOUND! Looking for "
                << spec.to_string() << std::endl;
      std::cout << "Available keys:" << std::endl;
      for (const auto& [k, v] : indexes_) {
        std::cout << "  " << k.to_string() << std::endl;
      }
    }
    return indexes_.at(spec);
  }

  // check if an index exists (instantiated)
  [[nodiscard]] bool has_index(const IndexSpec& spec) const {
    return indexes_.find(spec) != indexes_.end();
  }

  // check if a index is dirty
  [[nodiscard]] bool is_dirty(const IndexSpec& spec) const {
    // Check if index exists first
    auto it = indexes_.find(spec);
    if (it == indexes_.end()) {
      // Index doesn't exist, so it's dirty (needs to be built)
      return true;
    }
    // Compare interned_size() to rows_processed() to handle deduplication correctly.
    // For HashTrieIndex: rows_processed() == size() (no deduplication)
    // For SortedArrayIndex: rows_processed() >= size() (after deduplication)
    // This ensures we correctly detect when an index needs rebuilding even if
    // the index has fewer entries due to deduplication.
    return interned_size() != it->second.rows_processed();
  }

  inline auto ensure_index_nc_(const IndexSpec& spec, bool build = true) -> const IndexTypeInst&;

  /// @brief Build index by taking ownership of intern columns (zero-copy for identity spec).
  /// @details After this call, the index is the sole owner of sorted/deduped data.
  ///          Intern columns are cleared. Use get_index(spec) to access the result.
  ///          Do NOT call ensure_index() on this relation after this call.
  /// @param spec The index specification.
  /// @return A const reference to the built index.
  const IndexTypeInst& build_index_take_ownership(const IndexSpec& spec);

  /// @brief Build a non-canonical index from an existing index (index-to-index).
  /// @details Finds the first existing index in this relation and builds the target
  ///          index from its column data, remapping columns per their specs.
  ///          This avoids the roundtrip through intern cols.
  ///          Used for DELTA non-canonical index builds after compute-delta.
  /// @param target_spec The target index specification to build.
  /// @return A const reference to the built index.
  const IndexTypeInst& rebuild_index_from_existing(const IndexSpec& target_spec);

  /// @brief Range query on an indexed column.
  /// @details This will return a NodeHandle to the node in the index that
  /// matches the key.
  ///          Example:
  ///          // find all tuples with key (1, 2) in the index {0, 1}
  ///          auto node = R.indexed_range_query({0, 1}, {1, 2});
  /// @param spec The index specification.
  /// @param key The key to query on.
  /// @return A NodeHandle to the node in the index that matches the key.
  template <std::size_t N>
  auto indexed_range_query(const IndexSpec& spec, const EncodedKeyPrefix<N>& key) const ->
      typename IndexTypeInst::NodeHandle {
    const auto& idx = ensure_index(spec);
    return idx.prefix_lookup(key);
  }

  // eagerly build all indexes
  void build_all_indexes() {
    for (const auto& spec : index_specs_) {
      ensure_index(spec);
    }
  }

  /// @brief Get a reference to the list of index specifications.
  /// @details Returns a reference to the internal vector of IndexSpec objects
  ///          that define which indexes are registered for this relation.
  ///          This allows external code to inspect or modify the index
  ///          specifications.
  /// @return A reference to the vector of index specifications
  std::vector<IndexSpec>& index_specs() {
    return index_specs_;
  }

  /// @brief Get a const reference to the list of index specifications.
  /// @return A const reference to the vector of index specifications
  const std::vector<IndexSpec>& index_specs() const {
    return index_specs_;
  }

  /// @brief Get the encoded rows from the relation.
  /// @details This will return a std::array of spans of the encoded rows.
  ///          This is used to build the index from pre-encoded rows.
  ///          NOTE: consider change end to length in future
  /// @param start The start index of the rows to get.
  /// @param end The end index of the rows to get.
  /// @return Host: std::array of std::span<const ValueType>, Device: std::array of std::span<const
  /// uint32_t>
  auto encoded_rows(std::size_t start, std::size_t end) const
    requires(!StorageTraits::is_device)
  {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::array<std::span<const ValueType>, arity>{
          std::span<const ValueType>(interned_col_at<Is>(*this).data() + start, end - start)...};
    }(std::make_index_sequence<arity>{});
  }

  // Device version - returns uint32_t spans
  auto encoded_rows(std::size_t start, std::size_t end) const
    requires StorageTraits::is_device
  {
    const std::size_t size = end - start;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      // C++20: std::span for non-owning views of encoded data (uint32_t on device)
      if constexpr (Layout == GPU::StorageLayout::AoS) {
        // Returning spans for AoS is hard because data is strided.
        // We can't return std::span<const uint32_t> that represents a column.
        // This function is mainly used for building index from encoded rows.
        // For AoS index, we might not need this if we build directly from AoS.
        // But if we do, we might need a strided_span or copy.
        throw std::runtime_error("encoded_rows() not supported for AoS layout directly.");
        return std::array<std::span<const uint32_t>, arity>{};
      } else {
        return std::array<std::span<const uint32_t>, arity>{std::span<const uint32_t>(
            storage_.device_.device_interned_cols_.template column_ptr<Is>() + start, size)...};
      }
    }(std::make_index_sequence<arity>{});
  }

  /// @brief Static function to intersect NodeHandles using this Relation's Index type
  /// @details This delegates to the IndexTypeInst's static intersect method.
  ///          This allows column_intersect to use the Relation's Index type without
  ///          requiring callers to explicitly pass the Index type.
  /// @tparam NodeHandleTs The variadic pack of NodeHandle types
  /// @param handles The NodeHandle arguments to intersect
  /// @return A C++20 lazy view of the intersection of all columns.
  template <CNodeHandle... NodeHandleTs>
  static auto column_intersect(const NodeHandleTs&... handles) noexcept {
    return IndexTypeInst::intersect(handles...);
  }

  /// @brief Static function to intersect NodeHandles from a tuple using this Relation's Index type
  /// @details This unpacks the tuple and calls the Index's static intersect method.
  /// @tparam TupleOfHandles A std::tuple of column value handles
  /// @param handles A tuple of handles (e.g., from std::make_tuple(a, b, c))
  /// @return A C++20 lazy view of the intersection of all columns.
  template <tmp::CTuple TupleOfHandles>
  static auto column_intersect(const TupleOfHandles& handles) noexcept {
    return std::apply([](const auto&... handles) { return IndexTypeInst::intersect(handles...); },
                      handles);
  }

  /// @brief Explicitly clone this relation into another relation.
  /// @details Performs a deep copy of all relation data into the target
  ///          relation. This includes:
  ///          - All attribute columns (host only)
  ///          - All annotation values
  ///          - All interned columns
  ///
  ///          Note: Indexes are NOT cloned or rebuilt. The target relation's
  ///          indexes will be marked as dirty and will need to be rebuilt
  ///          if needed.
  /// @param other The target relation to clone into (will be cleared first)
  template <typename OtherRelation>
  void clone_into(OtherRelation& other) const {
    static_assert(std::is_same_v<typename OtherRelation::AnnotationVal, AnnotationVal>,
                  "Relations must have the same semiring type");
    static_assert(OtherRelation::arity == arity, "Relations must have the same arity");
    // clear the other relation
    // other.clear();

    auto& prov = other.provenance();
    if constexpr (StorageTraits::is_device) {
      device_storage().device_interned_cols_.clone_into(other.unsafe_interned_columns());
      device_storage().device_ann_.clone_into(other.provenance());
    } else {
      prov.insert(prov.end(), host_storage().ann_.begin(), host_storage().ann_.end());

      // Check target storage layout (if it has one)
      if constexpr (requires { OtherRelation::Layout; }) {
        if constexpr (OtherRelation::Layout == GPU::StorageLayout::AoS &&
                      OtherRelation::StorageTraits::is_device) {
          // Host (SoA) -> Device (AoS)
          // Interleave host columns into AoS buffer
          std::size_t n = size();
          if (n > 0) {
            // We need to construct interleaved data on host then copy to device (or use specialized
            // kernel if data is on device? No, data is on Host) Access host columns
            std::vector<typename OtherRelation::ValueType> interleaved(n * arity);
            // Assume ValueType matches or is compatible
            for (std::size_t i = 0; i < n; ++i) {
              [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((interleaved[i * arity + Is] = static_cast<typename OtherRelation::ValueType>(
                      host_storage().interned_cols_[Is][i])),
                 ...);
              }(std::make_index_sequence<arity>{});
            }

            // Append directly to AoSDeviceArray from host pointer?
            // data() gives device pointer.
            // We typically use push_back for rows or resize + memcpy.
            // AoSDeviceArray::push_back takes ONE row.
            // We should add a bulk append or use resize + memcpy.
            // AoSDeviceArray helper in benchmark did memcpy.
            // Let's use resize on target and memcpy.
            auto& other_aos = other.unsafe_interned_columns();
            std::size_t start_idx = other_aos.num_rows();
            other_aos.resize(start_idx + n);
// Copy to device
// Assuming GPU_MEMCPY is available via helper or direct
// GPU::GPU_MEMCPY(other_aos.data() + start_idx * arity, interleaved.data(), interleaved.size() *
// sizeof(typename OtherRelation::ValueType), GPU_HOST_TO_DEVICE); Relation doesn't include gpu_api
// directly? Check if we can use other.unsafe_interned_columns(). AoSDeviceArray doesn't leak
// GPU_MEMCPY directly but we can use cudaMemcpy via wrapper? Or maybe AoSDeviceArray has bulk
// insertion? Let's assume we can call gpu_api if included or use a helper. Since this is generic
// Relation code, dependent on GPU support.
#if SRDATALOG_GPU_AVAILABLE == 1
            cudaMemcpy(other_aos.data() + start_idx * arity, interleaved.data(),
                       interleaved.size() * sizeof(typename OtherRelation::ValueType),
                       cudaMemcpyHostToDevice);
#endif
          }
          return;  // Done
        }
      }

      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(host_storage().cols_)
              .clone_into(std::get<Is>(other.unsafe_attribute_columns()))),
         ...);
      }(std::make_index_sequence<arity>{});
      // clone the interned columns into the other relation (SoA)
      for (std::size_t I = 0; I < arity; ++I) {
        other.unsafe_interned_columns()[I].assign(host_storage().interned_cols_[I].begin(),
                                                  host_storage().interned_cols_[I].end());
      }
    }
  }

  /// @brief Reconstruct interned columns and provenance from an index.
  /// @details This will reconstruct the interned columns and provenance from an index. Usually
  /// called on NEW_VER after materializing join results. Please avoid calling this function on full
  /// version during fixpoint computation.
  /// @param spec The index specification.
  void reconstruct_from_index(const IndexSpec& spec);

  // ============================================================
  // Unsafe Accessors (for system-level ops like to_host/to_device)
  // ============================================================

  /// @brief Access interned columns storage directly.
  /// @warning Internal use only. Exposes implementation details.
  auto& unsafe_interned_columns() {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_interned_cols_;
    } else {
      return host_storage().interned_cols_;
    }
  }

  /// @brief Access interned columns storage directly (const).
  /// @warning Internal use only. Exposes implementation details.
  const auto& unsafe_interned_columns() const {
    if constexpr (StorageTraits::is_device) {
      return device_storage().device_interned_cols_;
    } else {
      return host_storage().interned_cols_;
    }
  }

  /// @brief Access attribute columns storage directly (Host only).
  /// @warning Internal use only. Exposes implementation details.
  auto& unsafe_attribute_columns() {
    static_assert(!StorageTraits::is_device, "Attribute columns only exist on host");
    return host_storage().cols_;
  }

  /// @brief Access attribute columns storage directly (const, Host only).
  /// @warning Internal use only. Exposes implementation details.
  const auto& unsafe_attribute_columns() const {
    static_assert(!StorageTraits::is_device, "Attribute columns only exist on host");
    return host_storage().cols_;
  }

 private:
  // Conditional storage based on Policy using union
  union StorageUnion {
    // Host storage
    struct HostData {
      ColumnTuple<AttrTuple> cols_;
      std::array<Vector<ValueType>, arity> interned_cols_;
      Vector<AnnotationVal> ann_;
      memory_resource* resource_{nullptr};
    } host_;

    // Device storage (only encoded data as uint32_t, NOT size_t)
    // Only available when GPU headers are included (CUDA or RMM available)
#if SRDATALOG_GPU_AVAILABLE == 1
    struct DeviceData {
      DeviceColsType device_interned_cols_;
      // Conditional provenance storage: zero size when NoProvenance
      [[no_unique_address]]
      std::conditional_t<has_provenance_v<SR>, GPU::DeviceArray<AnnotationVal>, std::monostate>
          device_ann_;
    } device_;
#else
    // GPU not available - DeviceData is empty, DeviceRelationPolicy will fail to compile
    struct DeviceData {
      // Empty struct - DeviceRelationPolicy cannot be used without GPU support
      // The static_assert in RelationStorageTraits will catch this
    } device_;
#endif

    StorageUnion() {
      if constexpr (StorageTraits::is_device) {
#if SRDATALOG_GPU_AVAILABLE == 1
        new (&device_) DeviceData{};
#else
        // DeviceData is empty when GPU not available - static_assert in RelationStorageTraits will
        // catch this
        static_assert(StorageTraits::is_device == false,
                      "DeviceRelationPolicy requires GPU support");
#endif
      } else {
        new (&host_) HostData{};
      }
    }

    ~StorageUnion() {
      if constexpr (StorageTraits::is_device) {
#if SRDATALOG_GPU_AVAILABLE == 1
        device_.~DeviceData();
#else
        // DeviceData is empty when GPU not available
#endif
      } else {
        host_.~HostData();
      }
    }

    // Union is managed by Relation's copy/move operations
    // Marked as deleted to prevent accidental direct use
    StorageUnion(const StorageUnion&) = delete;
    StorageUnion& operator=(const StorageUnion&) = delete;
    StorageUnion(StorageUnion&&) = delete;
    StorageUnion& operator=(StorageUnion&&) = delete;
  } storage_;

  // Helper accessors
  auto& host_storage() {
    static_assert(!StorageTraits::is_device, "host_storage() only available for host relations");
    return storage_.host_;
  }
  const auto& host_storage() const {
    static_assert(!StorageTraits::is_device, "host_storage() only available for host relations");
    return storage_.host_;
  }
  auto& device_storage() {
    static_assert(StorageTraits::is_device, "device_storage() only available for device relations");
    return storage_.device_;
  }
  const auto& device_storage() const {
    static_assert(StorageTraits::is_device, "device_storage() only available for device relations");
    return storage_.device_;
  }

  std::vector<std::string> col_names_;
  std::string name_;     // relation name for debugging
  std::size_t version_;  // relation version (2=FULL_VER, 3=DELTA_VER, 1=NEW_VER, 10=UNKNOWN_VER)

  // Index registry (spec → concrete HashTrie)
  std::vector<IndexSpec> index_specs_;  // (1) stored specifications
  // Do we need allocate this on allocator?
  std::unordered_map<IndexSpec, IndexTypeInst, IndexSpecHash> indexes_;

  template <std::size_t... I>
  void reserve_impl_(std::size_t n, std::index_sequence<I...>);

  /// @brief Swap with another relation
  void swap(Relation& other) noexcept {
    using std::swap;
    swap(col_names_, other.col_names_);
    swap(name_, other.name_);
    swap(version_, other.version_);
    swap(index_specs_, other.index_specs_);
    swap(indexes_, other.indexes_);

    if constexpr (StorageTraits::is_device) {
      swap(device_storage().device_interned_cols_, other.device_storage().device_interned_cols_);
      swap(device_storage().device_ann_, other.device_storage().device_ann_);
    } else {
      swap(host_storage().cols_, other.host_storage().cols_);
      swap(host_storage().interned_cols_, other.host_storage().interned_cols_);
      swap(host_storage().ann_, other.host_storage().ann_);
      swap(host_storage().resource_, other.host_storage().resource_);
    }
  }

  friend void swap(Relation& a, Relation& b) noexcept {
    a.swap(b);
  }

  void register_canonical_index_() {
    IndexSpec spec;
    spec.cols.reserve(arity);
    for (std::size_t i = 0; i < arity; ++i) {
      spec.cols.push_back(static_cast<int>(i));
    }

    // Avoid duplicates if called multiple times (e.g., via move assignment)
    if (std::ranges::find(index_specs_, spec) != index_specs_.end()) {
      return;
    }

    index_specs_.push_back(spec);
    // Use relation's resource to ensure consistent memory management
    // Use SFINAE to check if index accepts memory_resource in constructor
    if constexpr (StorageTraits::is_device) {
      // Device: Indexes don't take memory_resource
      indexes_.emplace(spec, IndexTypeInst{});
    } else {
      if constexpr (requires { IndexTypeInst{host_storage().resource_}; }) {
        indexes_.emplace(spec, IndexTypeInst{host_storage().resource_});
      } else {
        indexes_.emplace(spec, IndexTypeInst{});
      }
    }
  }
};

template <typename T>
struct is_node_handle : std::false_type {};

template <typename T>
inline constexpr bool is_node_handle_v = is_node_handle<T>::value;

/// @brief anti-join version of column_intersect
/// @details This will return a lazy view of the anti-intersection of all columns.
///          It iterates the *first* argument 'a' and checks if 'key' is not
///          contained in any of the *other* arguments (b, c, ...).
///          It's not yet clear is more than two arguments is needed. please be careful when using
///          this function.
/// @note **C++20 features**: Uses `std::ranges::filter_view` for lazy
///       evaluation, fold expressions for predicate checking, and concepts
///       for type constraints. The result is a lazy view that filters values
///       on-demand without materializing intermediate results.
/// @tparam NodeHandleT The first Relation's column value handle.
/// @tparam OtherNodeHandleTs The variadic pack of other column handles.
/// @param a The first Relation's column value handle (ideally the smallest).
/// @param others The other column handles to anti-intersect against.
/// @return A C++20 lazy view of the anti-intersection of all columns.
template <CNodeHandle NodeHandleT, CNodeHandle... OtherNodeHandleTs>
auto column_anti_intersect(const NodeHandleT& a, const OtherNodeHandleTs&... others) noexcept;

// util function to lookup with unencoded key
template <CIndex IndexT, typename... Ts>
auto lookup_prefix(const IndexT& index, const Prefix<Ts...>& prefix) noexcept ->
    typename IndexT::NodeHandle;

/// @brief Overload that takes Index type as template parameter and calls its static intersect
/// method
/// @details This is the new preferred way to call column_intersect. It delegates to the Index's
///          static intersect method, which allows each Index type to implement its own
///          intersection strategy (e.g., HashTrieIndex uses hash-probe, SortedArrayIndex uses
///          leapfrog join).
/// @tparam IndexType The Index type (must satisfy CIndex concept)
/// @tparam NodeHandleTs The variadic pack of NodeHandle types
/// @param handles The NodeHandle arguments to intersect
/// @return A C++20 lazy view of the intersection of all columns.
template <CIndex IndexType, CNodeHandle... NodeHandleTs>
auto column_intersect(const NodeHandleTs&... handles) noexcept;

/// @brief Overload for intersecting columns provided in a std::tuple with Index type
/// @details This unpacks the tuple and calls the Index's static intersect method.
/// @tparam IndexType The Index type (must satisfy CIndex concept)
/// @tparam TupleOfHandles A std::tuple of column value handles
/// @param handles A tuple of handles (e.g., from std::make_tuple(a, b, c))
/// @return A C++20 lazy view of the intersection of all columns.
template <CIndex IndexType, tmp::CTuple TupleOfHandles>
auto column_intersect(const TupleOfHandles& handles) noexcept;

// @brief reoder a tuple based on the order of the a IndexSpec
template <std::size_t N>
auto reorder_tuple(const IndexSpec& spec, const std::array<std::size_t, N>& array)
    -> std::array<std::size_t, N> {
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::array{array[spec.cols[Is]]...};
  }(std::make_index_sequence<N>{});
}

}  // namespace SRDatalog

// TODO(user): MPI function to distribute the data ranks based on the hash value of
// the key

// Class for Relation buffer which is row-oriented for commmunication purpose
// After communication, the buffer is convereted back to oriented column

#include "hashtrie.h"
#include "relation_col.ipp"
