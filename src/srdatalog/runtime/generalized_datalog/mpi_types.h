#pragma once
#include <type_traits>
#include <typeindex>
#include <string>
#include <cstdint>
#include <cstddef>   // size_t, ptrdiff_t
#include <mpi.h>

/**
 * mpi_type.h — portable mapping between C++ types and MPI_Datatype
 *
 * What you get:
 *  - Compile-time trait  : is_mpi_direct_mappable_v<T>
 *  - Compile-time mapper : mpi_datatype_of<T>()  (for directly mappable types)
 *  - Runtime mappers     : mpi_datatype_from_type_index(...),
 *                          mpi_datatype_match_size_integer(...),
 *                          mpi_datatype_of_size_t(), mpi_datatype_of_ptrdiff_t()
 *  - Helpers for creating derived types for PODs: create_contiguous_datatype<T>()
 *  - Guidance for std::string packing (length + bytes)
 *
 * Notes:
 *  - "Directly mappable" means a standard MPI predefined datatype exists.
 *  - For non-direct types (e.g., std::string, custom structs), either pack/unpack
 *    manually or create a derived datatype.
 *  - For size-dependent integer types (size_t, ptrdiff_t), use the runtime helpers
 *    that call MPI_Type_match_size; do NOT guess underlying types.
 */

// ===============================
// 1) Pure compile-time trait
// ===============================

// Base template: not directly mappable by default
template<class T>
struct is_mpi_direct_mappable : std::false_type {};

// Signed/Unsigned 8/16/32/64-bit integers
template<> struct is_mpi_direct_mappable<std::int8_t>   : std::true_type {};
template<> struct is_mpi_direct_mappable<std::uint8_t>  : std::true_type {};
template<> struct is_mpi_direct_mappable<std::int16_t>  : std::true_type {};
template<> struct is_mpi_direct_mappable<std::uint16_t> : std::true_type {};
template<> struct is_mpi_direct_mappable<std::int32_t>  : std::true_type {};
template<> struct is_mpi_direct_mappable<std::uint32_t> : std::true_type {};
template<> struct is_mpi_direct_mappable<std::int64_t>  : std::true_type {};
template<> struct is_mpi_direct_mappable<std::uint64_t> : std::true_type {};

// Floating
template<> struct is_mpi_direct_mappable<float>         : std::true_type {};
template<> struct is_mpi_direct_mappable<double>        : std::true_type {};
template<> struct is_mpi_direct_mappable<long double>   : std::true_type {};

// Booleans / char family
template<> struct is_mpi_direct_mappable<bool>          : std::true_type {};
template<> struct is_mpi_direct_mappable<char>          : std::true_type {};
 
// Convenient alias
template<class T>
inline constexpr bool is_mpi_direct_mappable_v = is_mpi_direct_mappable<T>::value;

// A stricter concept you can use to constrain Relation column types.
// (You can relax this later to allow user-provided packers.)
template<class T>
concept MpiCompatible = is_mpi_direct_mappable_v<T>;


// ==============================================
// 2) Compile-time mapper for directly-mappable T
// ==============================================
//
// WARNING: For size_t/ptrdiff_t and non-direct types (e.g., std::string),
//          do not call this. Use the runtime helpers below.
//
template<class T>
inline MPI_Datatype mpi_datatype_of() {
// if not apple use this
#if !defined(__APPLE__)
  static_assert(is_mpi_direct_mappable_v<T>,
                "mpi_datatype_of<T> called for a non-directly-mappable type. "
                "Use packing or runtime helpers (e.g., MPI_Type_match_size).");
#endif
  // Integers by width
  if constexpr (std::is_same_v<T, std::int8_t>)        return MPI_INT8_T;
  else if constexpr (std::is_same_v<T, std::uint8_t>)  return MPI_UINT8_T;
  else if constexpr (std::is_same_v<T, std::int16_t>)  return MPI_INT16_T;
  else if constexpr (std::is_same_v<T, std::uint16_t>) return MPI_UINT16_T;
  else if constexpr (std::is_same_v<T, std::int32_t>)  return MPI_INT32_T;
  else if constexpr (std::is_same_v<T, std::uint32_t>) return MPI_UINT32_T;
  else if constexpr (std::is_same_v<T, std::int64_t>)  return MPI_INT64_T;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return MPI_UINT64_T;

  // Floating
  else if constexpr (std::is_same_v<T, float>)         return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, double>)        return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, long double>)   return MPI_LONG_DOUBLE;

  // Booleans / char family
  else if constexpr (std::is_same_v<T, bool>)          return MPI_C_BOOL;
  else if constexpr (std::is_same_v<T, char>)          return MPI_CHAR;
  else if constexpr (std::is_same_v<T, signed char>)   return MPI_SIGNED_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>) return MPI_UNSIGNED_CHAR;

  // MPI address integer
  else if constexpr (std::is_same_v<T, MPI_Aint>)      return MPI_AINT;

  // Should be unreachable due to static_assert
  else return MPI_DATATYPE_NULL;
}


// ===============================================
// 3) Runtime helpers (size-based & type-indexed)
// ===============================================

// Map an integer type *by size in bytes* using MPI_Type_match_size.
// Requires MPI to be initialized. Returns MPI_DATATYPE_NULL on failure.
inline MPI_Datatype mpi_datatype_match_size_integer(int nbytes) {
  MPI_Datatype dt = MPI_DATATYPE_NULL;
  if (MPI_Type_match_size(MPI_TYPECLASS_INTEGER, nbytes, &dt) == MPI_SUCCESS)
    return dt;
  return MPI_DATATYPE_NULL;
}

// Portable mapping for size_t (do NOT guess the underlying type)
inline MPI_Datatype mpi_datatype_of_size_t() {
  return mpi_datatype_match_size_integer(static_cast<int>(sizeof(std::size_t)));
}

// Portable mapping for ptrdiff_t
inline MPI_Datatype mpi_datatype_of_ptrdiff_t() {
  return mpi_datatype_match_size_integer(static_cast<int>(sizeof(std::ptrdiff_t)));
}

// Runtime mapper from std::type_index.
// (Use for reflection-like paths or heterogeneous containers.)
inline MPI_Datatype mpi_datatype_from_type_index(std::type_index ti) {
  // Integer family
  if (ti == typeid(std::int8_t))     return MPI_INT8_T;
  if (ti == typeid(std::uint8_t))    return MPI_UINT8_T;
  if (ti == typeid(std::int16_t))    return MPI_INT16_T;
  if (ti == typeid(std::uint16_t))   return MPI_UINT16_T;
  if (ti == typeid(std::int32_t))    return MPI_INT32_T;
  if (ti == typeid(std::uint32_t))   return MPI_UINT32_T;
  if (ti == typeid(std::int64_t))    return MPI_INT64_T;
  if (ti == typeid(std::uint64_t))   return MPI_UINT64_T;

  // Floating
  if (ti == typeid(float))           return MPI_FLOAT;
  if (ti == typeid(double))          return MPI_DOUBLE;
  if (ti == typeid(long double))     return MPI_LONG_DOUBLE;

  // Booleans / char family
  if (ti == typeid(bool))            return MPI_C_BOOL;
  if (ti == typeid(char))            return MPI_CHAR;
  if (ti == typeid(signed char))     return MPI_SIGNED_CHAR;
  if (ti == typeid(unsigned char))   return MPI_UNSIGNED_CHAR;

  // MPI address integer
  if (ti == typeid(MPI_Aint))        return MPI_AINT;

  // size_t / ptrdiff_t via size-based match
  if (ti == typeid(std::size_t))     return mpi_datatype_of_size_t();
  if (ti == typeid(std::ptrdiff_t))  return mpi_datatype_of_ptrdiff_t();

  // Strings (variable length) and everything else: not directly mappable
  if (ti == typeid(std::string))     return MPI_DATATYPE_NULL;

  return MPI_DATATYPE_NULL;
}


// ===================================================
// 4) Derived datatype helper for trivially copyable T
// ===================================================
//
// Creates a contiguous byte representation for any trivially copyable POD.
// Caller owns the resulting MPI_Datatype and must free it with MPI_Type_free.
// This is a simple “byte blob” representation (no field alignment semantics).
//
template<class T>
inline MPI_Datatype create_contiguous_datatype() {
  static_assert(std::is_trivially_copyable_v<T>,
                "create_contiguous_datatype<T>() requires trivially copyable T.");
  MPI_Datatype dt = MPI_DATATYPE_NULL;
  // Represents sizeof(T) bytes; portable across homogeneous nodes.
  MPI_Type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE, &dt);
  MPI_Type_commit(&dt);
  return dt;
}


// =====================================
// 5) Guidance for std::string transport
// =====================================
//
// std::string is NOT directly mappable. Use a simple length+bytes scheme:
//   1) Send length as a fixed-size integer (e.g., uint64_t with MPI_UINT64_T)
//   2) Send raw bytes (char buffer) with MPI_CHAR
// Or serialize with your favorite codec. Document this policy where used.


// =====================================
// 6) Quick coverage checks (debug only)
// =====================================
//
// Example: ensure your semiring annotation types are covered.
// static_assert(is_mpi_direct_mappable_v<double>, "ProbIndep::value_type must map");
// static_assert(is_mpi_direct_mappable_v<std::uint64_t>, "Bag::value_type must map");
