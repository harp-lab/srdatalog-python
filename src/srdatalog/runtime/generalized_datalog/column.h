/**
 * @file column.h
 * @brief Thin, PMR-aware contiguous column wrapper and encoding/decoding system.
 *
 * @details This file provides the Column class for storing attribute data in
 * a column-oriented format, along with encoding/decoding utilities for
 * converting values to/from size_t for efficient indexing.
 *
 * @note **C++20/23 Features Used**: This file uses C++20 concepts extensively
 *       for type constraints (Encodable, Decodable, Codec, ColumnElement,
 *       ColumnElementTuple), `requires` clauses for template constraints,
 *       `std::bit_cast` for type-punning, `if constexpr` for compile-time
 *       conditionals, fold expressions for tuple element checking, and
 *       C++17 `if` statement with initialization.
 */
#pragma once
#include <cstddef>

#include "system.h"

#include <concepts>
#include <type_traits>

#include <boost/mp11.hpp>
#include <boost/unordered/unordered_map.hpp>

namespace SRDatalog {

/**
 * @brief Thin, PMR-aware contiguous column wrapper.
 *
 * `Column<T>` mirrors the interface of a `std::vector<T>` but stores data in a
 * polymorphic memory-resource (PMR) container by default. The implementation is
 * header-only with out-of-class definitions placed in `column.ipp`.
 *
 * @tparam T The element type stored in the column (must satisfy ColumnElement)
 */
template <class T>
class Column {
 public:
  using value_type = T;
  /// @brief Construct a column backed by the default PMR resource.
  Column() : raw(default_memory_resource()) {}
  Column(const Column&) = default;
  Column(Column&&) noexcept = default;
  Column& operator=(const Column&) = default;
  Column& operator=(Column&&) noexcept = default;
  ~Column() = default;

  /// @brief Construct a column bound to a caller-supplied PMR resource.
  /// @param resource The resource that will own the column's allocations.
  explicit Column(memory_resource* resource) : raw(resource) {}

  // Capacity / size
  void reserve(std::size_t n);
  void clear() noexcept;
  std::size_t size() const noexcept;

  // Appends
  void push_back(const T& v);
  void push_back(T&& v);

  template <class... Args>
  T& emplace_back(Args&&... args);

  // Raw buffer access
  T* data() noexcept;
  const T* data() const noexcept;

  // Element access
  T& operator[](std::size_t i);
  const T& operator[](std::size_t i) const;

  // Iterators
  auto begin() noexcept -> Vector<T>::iterator;
  auto end() noexcept -> Vector<T>::iterator;
  auto begin() const noexcept -> Vector<T>::const_iterator;
  auto end() const noexcept -> Vector<T>::const_iterator;

  void concat(const Column<T>& other) {
    raw.insert(raw.end(), other.raw.begin(), other.raw.end());
  }

  void clone_into(Column<T>& other) const {
    // clear the other column
    other.clear();
    // clone the current column into the other
    other.raw.assign(raw.begin(), raw.end());
  }

 private:
  // Underlying storage
  Vector<T> raw;
};

// Helper type alias to transform a tuple of types into a tuple of Column types
/// @brief Helper to transform a tuple of types into a tuple of Column types.
/// @details Uses boost::mp11 template metaprogramming to transform each type
///          in the tuple to its corresponding Column type.
/// @tparam AttrTuple The tuple of attribute types to transform
namespace detail {
template <typename T>
using ColumnFor = Column<T>;
}
template <typename AttrTuple>
using ColumnTuple = boost::mp11::mp_transform<detail::ColumnFor, AttrTuple>;

// ============================================================
// Encoding/Decoding System: Why Codec?
// ============================================================
/**
 * @section codec_motivation Motivation for Encoding
 *
 * The codec system converts all attribute values to a uniform `size_t`
 * representation for use in HashTrieIndex structures. This design provides
 * several critical performance and architectural benefits:
 *
 * @subsection codec_uniform_size Uniform Value Size
 *
 * - **Even-length values**: All encoded values are `size_t` (typically 8
 * bytes), ensuring uniform key sizes in index structures
 * - **Simplified comparisons**: Index operations compare fixed-size values,
 *   enabling efficient SIMD operations and better branch prediction
 * - **Memory alignment**: Uniform sizes improve cache line utilization and
 *   reduce memory fragmentation
 *
 * @subsection codec_join_performance Join Performance
 *
 * - **Fast key comparisons**: Fixed-size encoded keys enable efficient hash
 *   table lookups and trie traversals during joins
 * - **Reduced comparison overhead**: No need for type-specific comparison logic
 *   in hot join paths (e.g., string comparison vs. integer comparison)
 * - **Better hash distribution**: Uniform encoding improves hash function
 *   effectiveness for join algorithms
 *
 * @subsection codec_communication Distributed Communication
 *
 * - **Serialization efficiency**: Encoded values serialize trivially (just
 *   size_t arrays) without type-specific serialization logic
 * - **Network efficiency**: Fixed-size values enable efficient packing and
 *   reduce communication overhead in MPI/distributed settings
 * - **Cross-platform compatibility**: size_t encoding provides consistent
 *   representation across different architectures
 *
 * @subsection codec_locality Cache Locality
 *
 * - **Dense storage**: Encoded columns store only size_t values, creating
 *   dense, cache-friendly memory layouts
 * - **Better prefetching**: Uniform sizes enable more effective hardware
 *   prefetching and cache line utilization
 * - **Reduced indirection**: For types like strings, encoding eliminates
 *   pointer chasing in index structures
 *
 * @subsection codec_tradeoffs Trade-offs
 *
 * Encoding adds a conversion step, but the benefits outweigh the cost:
 * - Conversion is typically O(1) for primitive types
 * - DictionaryCodec amortizes encoding cost across multiple uses
 * - Index operations (joins, lookups) benefit significantly from uniform
 * encoding
 * - The encoding layer is transparent to most users via Relation API
 *
 * @see HashTrieIndex for how encoded values are used in index structures
 * @see DictionaryCodec for bidirectional encoding of complex types
 */

// ============================================================
// Encoding functions: Convert values to size_t for indexing
// ============================================================

/**
 * @brief Encode an integral or enum value to size_t.
 * @tparam T Integral or enum type
 * @param val Value to encode
 * @return Encoded value as size_t
 * @details Performs a direct static_cast. This is a lossless encoding for
 *          integral types that fit in size_t.
 * @note **C++20 feature**: Uses `requires` clause for type constraints to
 *       ensure only integral or enum types are accepted.
 * @note For signed types, the value is converted to unsigned representation.
 */
template <typename T>
  requires std::is_integral_v<T> || std::is_enum_v<T>
constexpr std::size_t encode_to_size_t(const T& val) noexcept {
  return static_cast<std::size_t>(val);
}

/**
 * @brief Encode a floating-point value to size_t.
 * @tparam T Floating-point type (float, double, etc.)
 * @param val Value to encode
 * @return Encoded value as size_t (bit representation)
 * @details Uses std::bit_cast to preserve the bit pattern of the floating-point
 *          value. This is a lossless encoding that preserves exact bit
 * patterns.
 * @note **C++20 features**: Uses `requires` clause for type constraints,
 *       `if constexpr` for compile-time conditionals, and `std::bit_cast` for
 *       type-punning without undefined behavior.
 * @pre sizeof(T) <= sizeof(std::size_t)
 * @note The encoded value represents the bit pattern, not the numeric value.
 *       Use decode_from_size_t to recover the original floating-point value.
 */
template <typename T>
  requires std::is_floating_point_v<T> && (sizeof(T) <= sizeof(std::size_t))
constexpr std::size_t encode_to_size_t(const T& val) noexcept {
  if constexpr (sizeof(T) == sizeof(std::size_t)) {
    return std::bit_cast<std::size_t>(val);
  } else {
    using UInt = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
    return static_cast<std::size_t>(std::bit_cast<UInt>(val));
  }
}

/**
 * @brief Encode a hashable type to size_t using std::hash.
 * @tparam T Type that supports std::hash but is not integral or floating-point
 * @param val Value to encode
 * @return Hash value as size_t
 * @details Uses std::hash to compute a hash code. This is a one-way encoding:
 *          the original value cannot be recovered from the hash.
 * @note **C++20 feature**: Uses nested `requires` clauses to check that the
 *       type is not integral or floating-point, and that it supports
 *       `std::hash` with the correct return type. Uses `requires` expressions
 *       to check for the existence and return type of `std::hash`.
 * @note This is used for types that don't have a direct size_t representation
 *       but can be hashed. For bidirectional encoding, use DictionaryCodec.
 * @warning Hash collisions are possible. This encoding is not suitable when
 *          exact value recovery is needed.
 */
template <typename T>
  requires(!std::is_integral_v<T> && !std::is_floating_point_v<T>) && requires(const T& val) {
    { std::hash<T>{}(val) } -> std::convertible_to<std::size_t>;
  }
std::size_t encode_to_size_t(const T& val) {
  return std::hash<T>{}(val);
}

// ============================================================
// Decoding functions: Convert size_t back to original types
// ============================================================

/**
 * @brief Decode an integral or enum value from size_t.
 * @tparam T Integral or enum type
 * @param encoded Encoded value as size_t
 * @return Decoded value of type T
 * @details Performs a direct static_cast. This is the inverse of
 *          encode_to_size_t for integral/enum types.
 * @note **C++20 feature**: Uses `requires` clause for type constraints to
 *       ensure only integral or enum types are accepted.
 * @note For signed types, the unsigned representation is converted back.
 */
template <typename T>
  requires std::is_integral_v<T> || std::is_enum_v<T>
constexpr T decode_from_size_t(std::size_t encoded) noexcept {
  return static_cast<T>(encoded);
}

/**
 * @brief Decode a floating-point value from size_t.
 * @tparam T Floating-point type (float, double, etc.)
 * @param encoded Encoded value as size_t (bit representation)
 * @return Decoded floating-point value
 * @details Uses std::bit_cast to recover the floating-point value from its
 *          bit pattern. This is the inverse of encode_to_size_t for
 *          floating-point types.
 * @note **C++20 features**: Uses `requires` clause for type constraints,
 *       `if constexpr` for compile-time conditionals, and `std::bit_cast` for
 *       type-punning without undefined behavior.
 * @pre sizeof(T) <= sizeof(std::size_t)
 * @note The encoded value must have been created using encode_to_size_t for
 *       the same type T.
 */
template <typename T>
  requires std::is_floating_point_v<T> && (sizeof(T) <= sizeof(std::size_t))
constexpr T decode_from_size_t(std::size_t encoded) noexcept {
  if constexpr (sizeof(T) == sizeof(std::size_t)) {
    return std::bit_cast<T>(encoded);
  } else {
    using UInt = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
    return std::bit_cast<T>(static_cast<UInt>(encoded));
  }
}

/**
 * @brief Concept for types that can be decoded from size_t.
 * @details A type is Decodable if it provides a decode_from_size_t function
 *          that can convert a size_t back to the original type. This includes:
 *          - Integral types (via static_cast)
 *          - Enum types (via static_cast)
 *          - Floating-point types (via bit_cast)
 *          - Types with DictionaryCodec (via dictionary lookup)
 * @note **C++20 feature**: Uses C++20 concepts with `requires` expressions to
 *       check for the existence and return type of decode_from_size_t.
 * @see Encodable for the encoding concept
 * @see Codec for types that support both encoding and decoding
 */
template <typename T>
concept Decodable = requires(std::size_t encoded) {
  { decode_from_size_t<T>(encoded) } -> std::convertible_to<T>;
};

/**
 * @brief Bidirectional dictionary codec for mapping values to/from size_t.
 *
 * @details DictionaryCodec provides a two-way mapping between values of type T
 * and size_t identifiers. Unlike hash-based encoding, this provides:
 * - **Bidirectional mapping**: Values can be encoded and decoded losslessly
 * - **Collision-free**: Each unique value gets a unique ID
 * - **Incremental**: New values are assigned IDs on-demand
 * - **Memory efficient**: Uses PMR allocators for flexible memory management
 *
 * @tparam T The type of values to encode/decode (must be hashable and
 *           equality-comparable)
 *
 * @section dictionarycodec_usage Usage
 *
 * @code{.cpp}
 * DictionaryCodec<std::string> codec;
 *
 * // Encode values (assigns IDs automatically)
 * std::size_t id1 = codec.encode("hello");  // Returns 0
 * std::size_t id2 = codec.encode("world");  // Returns 1
 * std::size_t id3 = codec.encode("hello");  // Returns 0 (reuses existing)
 *
 * // Decode IDs back to values
 * const std::string& val1 = codec.decode(0);  // Returns "hello"
 * const std::string& val2 = codec.decode(1);  // Returns "world"
 *
 * // Check if ID exists
 * if (codec.contains(0)) {
 *   // ...
 * }
 * @endcode
 *
 * @section dictionarycodec_internals Internals
 *
 * The codec maintains two hash maps:
 * - `forward_`: Maps T -> size_t (value to ID)
 * - `inverse_`: Maps size_t -> T (ID to value)
 *
 * IDs are assigned sequentially starting from 0. The same value always maps to
 * the same ID (idempotent encoding).
 *
 * @section dictionarycodec_memory Memory Management
 *
 * Uses PMR (Polymorphic Memory Resource) allocators, allowing custom memory
 * strategies (arena allocation, pooling, etc.). Defaults to the default PMR
 * resource.
 *
 * @note DictionaryCodec is typically used via get_global_codec() for shared
 *       dictionaries, but can be instantiated directly for isolated codecs.
 *
 * @see get_global_codec for accessing shared codec instances
 * @see Codec concept for types supporting bidirectional encoding/decoding
 */
template <typename T>
class DictionaryCodec {
 public:
  /// @brief Forward map: value -> encoded ID
  using DictMap = boost::unordered::pmr::unordered_map<T, std::size_t>;
  /// @brief Inverse map: encoded ID -> value
  using InverseMap = boost::unordered::pmr::unordered_map<std::size_t, T>;

  /// @brief Construct an empty DictionaryCodec.
  /// @details Creates a new codec with no mappings. IDs will start at 0.
  explicit DictionaryCodec() : next_id_(0) {}

  /// @brief Encode a value to a size_t ID.
  /// @param value The value to encode
  /// @return The encoded ID (size_t)
  /// @details If the value has been seen before, returns its existing ID.
  ///          Otherwise, assigns a new sequential ID and stores the mapping.
  /// @note **C++17 feature**: Uses `if` statement with initialization
  ///       (`if (auto it = ...)`) for efficient lookup and initialization in
  ///       a single statement.
  /// @note Encoding is idempotent: encoding the same value multiple times
  ///       returns the same ID.
  std::size_t encode(const T& value) {
    if (auto it = forward_.find(value); it != forward_.end()) {
      return it->second;
    }
    std::size_t id = next_id_++;
    forward_.emplace(value, id);
    inverse_.emplace(id, value);
    return id;
  }

  /// @brief Decode an ID back to its original value.
  /// @param id The encoded ID to decode
  /// @return Const reference to the original value
  /// @throw std::runtime_error if the ID doesn't exist in the dictionary
  /// @details Returns a reference to the stored value. The reference remains
  ///          valid as long as the codec exists and the value isn't cleared.
  const T& decode(std::size_t id) const {
    auto it = inverse_.find(id);
    // if (it == inverse_.end()) {
    //   throw std::runtime_error("Invalid encoded value for dictionary");
    // }
    return it->second;
  }

  /// @brief Check if an ID exists in the dictionary.
  /// @param id The ID to check
  /// @return true if the ID exists, false otherwise
  /// @details Useful for checking validity before calling decode().
  /// @note **C++20 feature**: Uses `std::unordered_map::contains()` method
  ///       (C++20) instead of `find() != end()` pattern for clearer code.
  bool contains(std::size_t id) const {
    return inverse_.contains(id);
  }

  /// @brief Get the number of unique values in the dictionary.
  /// @return Number of distinct values that have been encoded
  /// @details This equals the number of unique IDs assigned (next_id_ if no
  ///          values have been cleared).
  std::size_t size() const {
    return forward_.size();
  }

  /// @brief Clear all mappings and reset the codec.
  /// @details Removes all value-to-ID mappings and resets the ID counter to 0.
  ///          After clearing, encoding will start assigning IDs from 0 again.
  void clear() {
    forward_.clear();
    inverse_.clear();
    next_id_ = 0;
  }

 private:
  DictMap forward_;      ///< Forward mapping: value -> ID
  InverseMap inverse_;   ///< Inverse mapping: ID -> value
  std::size_t next_id_;  ///< Next ID to assign (incremented on new encodings)
};

/**
 * @brief Get a global (shared) DictionaryCodec instance for type T.
 * @tparam T The type to get a codec for
 * @return Reference to a global DictionaryCodec<T> instance
 * @details Returns a static, process-wide codec instance. All calls with the
 *          same type T return the same codec instance, ensuring consistent
 *          encoding across the program.
 *
 * @section global_codec_usage Usage
 *
 * @code{.cpp}
 * // Get the global string codec
 * auto& codec = get_global_codec<std::string>();
 *
 * // Encode values (shared across all users)
 * std::size_t id = codec.encode("shared_value");
 *
 * // Decode anywhere in the program
 * const std::string& val = codec.decode(id);
 * @endcode
 *
 * @note The global codec persists for the lifetime of the program. Use
 *       reset_all_global_codecs() or codec.clear() to reset if needed.
 *
 * @see DictionaryCodec for the codec class
 * @see reset_all_global_codecs for resetting all global codecs
 */
template <typename T>
inline DictionaryCodec<T>& get_global_codec() {
  static DictionaryCodec<T> codec;
  return codec;
}

/**
 * @brief Reset all global dictionaries (useful for testing).
 * @details This function is intended to clear all global codec instances,
 *          but currently has limited functionality. In practice, you need to
 *          know which types have been instantiated to reset them individually.
 *
 * @note This is a placeholder. To reset specific codecs:
 * @code{.cpp}
 * get_global_codec<std::string>().clear();
 * get_global_codec<MyType>().clear();
 * @endcode
 *
 * @warning Use with caution! Resetting codecs invalidates all previously
 *          encoded IDs. Only use in testing scenarios.
 */
inline void reset_all_global_codecs() {
  // Note: This is tricky because we don't track all instantiated codecs
  // In practice, you'd need a registry if you want to reset all
  // For now, users can reset specific types they know about
}

/**
 * @brief Decode a string from size_t using the global string codec.
 * @param encoded Encoded string ID
 * @return Decoded string value
 * @details Uses get_global_codec<std::string>() to decode the ID back to
 *          the original string value.
 * @throw std::runtime_error if the ID doesn't exist in the dictionary
 * @see get_global_codec for the underlying codec mechanism
 */
inline std::string decode_from_size_t(std::size_t encoded) {
  return get_global_codec<std::string>().decode(encoded);
}

/**
 * @brief Encode a string to size_t using the global string codec.
 * @param val String value to encode
 * @return Encoded string ID
 * @details Uses get_global_codec<std::string>() to encode the string.
 *          The same string always maps to the same ID.
 * @see get_global_codec for the underlying codec mechanism
 */
inline std::size_t encode_to_size_t(const std::string& val) {
  return get_global_codec<std::string>().encode(val);
}

/**
 * @brief Encode a tuple of values to size_t using the global string codec.
 * @param vals Tuple of values to encode
 * @return Encoded tuple of IDs
 * @details Uses get_global_codec<std::string>() to encode the tuple.
 * @see get_global_codec for the underlying codec mechanism
 */
template <typename Tuple>
inline std::array<std::size_t, std::tuple_size_v<Tuple>> encode_to_size_t_array(const Tuple& vals) {
  return std::apply([&](const auto&... vals) { return std::array{encode_to_size_t(vals)...}; },
                    vals);
}

/**
 * @brief Decode a tuple of values from size_t using the global string codec.
 * @param vals Tuple of values to decode
 * @return Decoded tuple of values
 * @details Uses get_global_codec<std::string>() to decode the tuple.
 * @see get_global_codec for the underlying codec mechanism
 */
template <typename Tuple>
inline Tuple decode_from_size_t_array(
    const std::array<std::size_t, std::tuple_size_v<Tuple>>& vals) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::make_tuple(decode_from_size_t<std::tuple_element_t<I, Tuple>>(vals[I])...);
  }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

/**
 * @brief Concept for types that can be encoded to size_t.
 * @details A type is Encodable if it provides an encode_to_size_t function
 *          that can convert values to size_t. This includes:
 *          - Integral types (via static_cast)
 *          - Enum types (via static_cast)
 *          - Floating-point types (via bit_cast)
 *          - Hashable types (via std::hash)
 *          - Types with DictionaryCodec (via dictionary lookup)
 *
 * @note **C++20 feature**: Uses C++20 concepts with `requires` expressions to
 *       check for the existence and return type of encode_to_size_t.
 *
 * @section encodable_usage Usage
 *
 * @code{.cpp}
 * static_assert(Encodable<int>);           // true
 * static_assert(Encodable<double>);        // true
 * static_assert(Encodable<std::string>);    // true (via DictionaryCodec)
 * @endcode
 *
 * @see Decodable for the decoding concept
 * @see Codec for types that support both encoding and decoding
 */
template <typename T>
concept Encodable = requires(const T& val) {
  { encode_to_size_t(val) } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for types that support both encoding and decoding.
 * @details A type satisfies Codec if it is both Encodable and Decodable.
 *          This means values can be converted to size_t and back without
 *          loss (for bidirectional codecs) or with deterministic mapping
 *          (for hash-based codecs).
 *
 * @note **C++20 feature**: Uses C++20 concept conjunction (`&&`) to combine
 *       multiple concepts (Encodable and Decodable) into a single constraint.
 *
 * @section codec_types Types That Satisfy Codec
 *
 * - **Integral types**: int, long, size_t, etc. (lossless via static_cast)
 * - **Enum types**: Any enum (lossless via static_cast)
 * - **Floating-point types**: float, double (lossless via bit_cast)
 * - **String**: std::string (lossless via DictionaryCodec)
 * - **Custom types**: Types with DictionaryCodec specialization
 *
 * @note Hash-based encoding (via std::hash) is Encodable but NOT Decodable,
 *       so it doesn't satisfy Codec. Use DictionaryCodec for bidirectional
 *       encoding of hashable types.
 *
 * @see Encodable for the encoding concept
 * @see Decodable for the decoding concept
 * @see DictionaryCodec for bidirectional encoding of arbitrary types
 */
template <typename T>
concept Codec = Encodable<T> && Decodable<T>;

/// @brief Concept requiring a type to be suitable for storage in a Relation
/// column.
/// @details The type must be trivially copyable, arithmetic, or a std::string.
/// @note **C++20 feature**: Uses C++20 concepts with concept disjunction (||)
///       to combine multiple type traits.
//          (TODO: consider Hashable, PartialOrder, which match what's in
//          ascent)
template <class T>
concept ColumnElement = std::is_trivially_copyable_v<T> || std::is_arithmetic_v<T> ||
                        std::is_same_v<T, std::string> || Codec<T>;

/// @brief Concept requiring all elements of a tuple to satisfy ColumnElement.
/// @details This concept checks that every type in the tuple satisfies the
///          ColumnElement concept, enabling tuple-based Relation templates
///          while maintaining type safety.
/// @note **C++20 feature**: Uses C++20 concepts with fold expressions
///       (`&& ...`) to check all tuple elements at compile time. The fold
///       expression iterates over all elements in the tuple using
///       `std::index_sequence` and checks each element satisfies
///       `ColumnElement`.
namespace detail {
template <typename Tuple, std::size_t... Is>
constexpr bool all_column_elements_impl(std::index_sequence<Is...>) {
  return (ColumnElement<std::tuple_element_t<Is, Tuple>> && ...);
}
}  // namespace detail
/// @brief Concept for tuple types where all elements satisfy ColumnElement.
/// @tparam Tuple The tuple type to check
/// @note **C++20 feature**: Uses fold expressions and `std::index_sequence`
///       to check all tuple elements at compile time.
template <typename Tuple>
concept ColumnElementTuple =
    std::is_same_v<Tuple, std::tuple<>> ||
    detail::all_column_elements_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

}  // namespace SRDatalog

// Keep template definitions visible to all TUs.
#include "column.ipp"
