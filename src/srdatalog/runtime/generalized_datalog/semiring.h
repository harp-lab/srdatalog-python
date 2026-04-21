#pragma once
#include "gpu/macro.h"  // Provides GPU_HD and all GPU platform detection macros
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <variant>  // For std::monostate (used by NoProvenance)
// Note: gpu/macro.h already includes the appropriate runtime headers (cuda_runtime.h or
// hip/hip_runtime.h)

//
// Semiring concept
//
/// @brief Concept defining the interface for semiring types.
/// @details A semiring must provide a value_type, identity elements (zero, one),
///          and operations (add, mul). This concept ensures type safety and
///          enables generic programming over different semiring implementations.
/// @note **C++20 feature**: Uses C++20 concepts with `requires` expressions to
///       check for required types and operations. The concept uses
///       `std::same_as` for type constraints.
/// @tparam SR The semiring type to check
template <class SR>
concept Semiring = requires(typename SR::value_type a, typename SR::value_type b) {
  // associated value type
  typename SR::value_type;

  // identities
  { SR::zero() } -> std::same_as<typename SR::value_type>;
  { SR::one() } -> std::same_as<typename SR::value_type>;

  // operations
  { SR::add(a, b) } -> std::same_as<typename SR::value_type>;
  { SR::mul(a, b) } -> std::same_as<typename SR::value_type>;
};

//
// Atomic Semiring concept
//
/// @brief Concept defining the interface for semirings with atomic operations.
/// @details An atomic semiring extends the base Semiring concept with atomic
///          versions of add and mul operations. These are essential for
///          thread-safe parallel updates in GPU kernels where multiple threads
///          may update the same memory location concurrently.
/// @note Atomic operations take a pointer to the value to update and a value to
///       combine with. The operation is performed atomically, ensuring
///       thread-safety in parallel contexts.
/// @tparam SR The semiring type to check
template <class SR>
concept AtomicSemiring =
    Semiring<SR> && requires(typename SR::value_type* ptr, typename SR::value_type val) {
      // Atomic operations: atomically update *ptr using SR operations
      { SR::atomic_add(ptr, val) } -> std::same_as<void>;
      { SR::atomic_mul(ptr, val) } -> std::same_as<void>;
    };

//
// Traits (override by specialization if needed)
//
template <class SR>
struct semiring_traits {
  static constexpr bool add_idempotent = false;  // e.g., Boolean has true
  static constexpr bool mul_commutative = true;  // set false if your SR needs it
  static constexpr bool has_provenance = true;   // default: provenance tracking enabled
};

//
// Reference semirings
//

// Natural numbers (bag semantics): counts derivations
struct NaturalBag {
  using value_type = std::uint64_t;
  GPU_HD static constexpr value_type zero() noexcept {
    return 0;
  }
  GPU_HD static constexpr value_type one() noexcept {
    return 1;
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type add(value_type a, value_type b) noexcept {
    return a + b;
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type mul(value_type a, value_type b) noexcept {
    return a * b;
  }

#if defined(__CUDACC__) || defined(__HIP__)
  // Atomic operations for GPU (CUDA and HIP)
  // Note: Requires sizeof(value_type) <= 8 bytes (64 bits)
  // For larger types, use a different atomic strategy or lock-based approach
  static_assert(sizeof(value_type) <= 8,
                "NaturalBag value_type must be <= 8 bytes for atomic operations");

  __device__ static void atomic_add(value_type* ptr, value_type val) noexcept {
    if constexpr (sizeof(value_type) == 8) {
      // 64-bit: use unsigned long long
      atomicAdd(reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(val));
    } else if constexpr (sizeof(value_type) == 4) {
      // 32-bit: use unsigned int
      atomicAdd(reinterpret_cast<unsigned int*>(ptr), static_cast<unsigned int>(val));
    } else {
      // For smaller types (1 or 2 bytes), use compare-and-swap loop
      // CUDA atomicAdd only supports 32-bit and 64-bit types
      // atomicCAS supports unsigned short (2 bytes) on SM 7.0+, but not unsigned char
      value_type old = *ptr;
      value_type expected = old;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      do {
        expected = old;
        value_type updated = old + val;
        if constexpr (sizeof(value_type) == 2) {
          // 2 bytes: use unsigned short (requires SM 7.0+)
          old = atomicCAS(reinterpret_cast<unsigned short*>(ptr),
                          static_cast<unsigned short>(expected),
                          static_cast<unsigned short>(updated));
        } else {
          // 1 byte: cast to unsigned int and mask (atomicCAS doesn't support unsigned char)
          unsigned int* int_ptr = reinterpret_cast<unsigned int*>(
              reinterpret_cast<char*>(ptr) - (reinterpret_cast<uintptr_t>(ptr) & 3));
          unsigned int int_old = *int_ptr;
          unsigned int int_expected = int_old;
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
          do {
            int_expected = int_old;
            unsigned int byte_offset = reinterpret_cast<uintptr_t>(ptr) & 3;
            unsigned int byte_mask = 0xFFU << (byte_offset * 8);
            unsigned int byte_val = static_cast<unsigned char>(updated) << (byte_offset * 8);
            unsigned int int_updated = (int_old & ~byte_mask) | byte_val;
            int_old = atomicCAS(int_ptr, int_expected, int_updated);
          } while (int_old != int_expected);
          old = static_cast<value_type>((int_old >> ((reinterpret_cast<uintptr_t>(ptr) & 3) * 8)) &
                                        0xFFU);
        }
      } while (old != expected);
    }
  }
  __device__ static void atomic_mul(value_type* ptr, value_type val) noexcept {
    // atomicMul doesn't exist, use compare-and-swap loop
    // Note: This assumes value_type fits in 64 bits
    value_type old = *ptr;
    value_type expected = old;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      expected = old;
      value_type updated = old * val;
      if constexpr (sizeof(value_type) == 8) {
        old = atomicCAS(reinterpret_cast<unsigned long long*>(ptr),
                        static_cast<unsigned long long>(expected),
                        static_cast<unsigned long long>(updated));
      } else if constexpr (sizeof(value_type) == 4) {
        old = atomicCAS(reinterpret_cast<unsigned int*>(ptr), static_cast<unsigned int>(expected),
                        static_cast<unsigned int>(updated));
      } else {
        // For smaller types, cast to appropriate size
        // using atomic_type =
        //     std::conditional_t<sizeof(value_type) == 2, unsigned short, unsigned char>;
        // old = atomicCAS(reinterpret_cast<atomic_type*>(ptr), static_cast<atomic_type>(expected),
        //                 static_cast<atomic_type>(updated));
      }
    } while (old != expected);
  }
#else
  // Host fallback: not truly atomic, but allows compilation
  // On HIP, this needs to be __device__ to be callable from kernels
  GPU_HD static void atomic_add(value_type* ptr, value_type val) noexcept {
    *ptr = add(*ptr, val);
  }
  GPU_HD static void atomic_mul(value_type* ptr, value_type val) noexcept {
    *ptr = mul(*ptr, val);
  }
#endif
};

// Back compat alias for your existing name if you want it:
using Bag = NaturalBag;

// Boolean semiring (classic Datalog)
struct BooleanSR {
  using value_type = bool;
  GPU_HD static constexpr value_type zero() noexcept {
    return false;
  }
  GPU_HD static constexpr value_type one() noexcept {
    return true;
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type add(value_type a, value_type b) noexcept {
    return a || b;
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type mul(value_type a, value_type b) noexcept {
    return a && b;
  }

#if defined(__CUDACC__) || defined(__HIP__)
  // Atomic operations for GPU (CUDA and HIP)
  __device__ static void atomic_add(value_type* ptr, value_type val) noexcept {
    // Boolean OR: Set *ptr = *ptr || val
    // Since value_type is bool (1 byte), we cannot use atomicOr (requires 4 bytes).
    // Use word-aligned atomicCAS loop.
    if (!val)
      return;  // OR with false does nothing

    unsigned int* int_ptr =
        reinterpret_cast<unsigned int*>(reinterpret_cast<uintptr_t>(ptr) & ~3UL);
    unsigned int byte_offset = reinterpret_cast<uintptr_t>(ptr) & 3;
    unsigned int shift = byte_offset * 8;
    unsigned int mask = 0xFFU << shift;

    unsigned int old = *int_ptr;
    unsigned int assumed;
    do {
      assumed = old;
      // If the bit is already set in assumed, we don't need to do anything
      // But we need to check the specific byte.
      bool current_val = (assumed >> shift) & 0xFFU;
      if (current_val)
        return;  // Already true

      // Set the byte to true (1)
      unsigned int val_shifted = 1U << shift;
      unsigned int updated =
          (assumed & ~mask) | (val_shifted & mask);  // Actually just | val_shifted

      old = atomicCAS(int_ptr, assumed, updated);
    } while (assumed != old);
  }

  __device__ static void atomic_mul(value_type* ptr, value_type val) noexcept {
    // Boolean AND: Set *ptr = *ptr && val
    // If val is true, no change. If val is false, set *ptr to false.
    if (val)
      return;

    unsigned int* int_ptr =
        reinterpret_cast<unsigned int*>(reinterpret_cast<uintptr_t>(ptr) & ~3UL);
    unsigned int byte_offset = reinterpret_cast<uintptr_t>(ptr) & 3;
    unsigned int shift = byte_offset * 8;
    unsigned int mask = 0xFFU << shift;

    unsigned int old = *int_ptr;
    unsigned int assumed;
    do {
      assumed = old;
      bool current_val = (assumed >> shift) & 0xFFU;
      if (!current_val)
        return;  // Already false

      // Set byte to false (0)
      unsigned int updated = (assumed & ~mask);

      old = atomicCAS(int_ptr, assumed, updated);
    } while (assumed != old);
  }
#else
  // Host fallback: not truly atomic, but allows compilation
  // On HIP, this needs to be __device__ to be callable from kernels
  GPU_HD static void atomic_add(value_type* ptr, value_type val) noexcept {
    *ptr = add(*ptr, val);
  }
  GPU_HD static void atomic_mul(value_type* ptr, value_type val) noexcept {
    *ptr = mul(*ptr, val);
  }
#endif
};
template <>
struct semiring_traits<BooleanSR> {
  static constexpr bool add_idempotent = true;
  static constexpr bool mul_commutative = true;
  static constexpr bool has_provenance = true;
};

#include <algorithm>

struct ProbIndep {
  using value_type = double;

  static constexpr value_type zero() noexcept {
    return 0.0;
  }  // P(false)
  static constexpr value_type one() noexcept {
    return 1.0;
  }  // P(true)

  GPU_HD GPU_FORCE_INLINE static value_type add(value_type a, value_type b) noexcept {
    value_type r = a + b - (a * b);  // 1 - (1-a)(1-b)
#if defined(__CUDACC__) || defined(__HIP__)
    return fmax(0.0, fmin(1.0, r));
#else
    return std::clamp(r, 0.0, 1.0);
#endif
  }

  GPU_HD GPU_FORCE_INLINE static value_type mul(value_type a, value_type b) noexcept {
    value_type r = a * b;
#if defined(__CUDACC__) || defined(__HIP__)
    return fmax(0.0, fmin(1.0, r));
#else
    return std::clamp(r, 0.0, 1.0);
#endif
  }

#if defined(__CUDACC__) || defined(__HIP__)
  // Atomic operations for GPU (CUDA and HIP)
  // Note: double is 64 bits, requires __double_as_longlong for atomicCAS
  static_assert(sizeof(value_type) == 8, "ProbIndep value_type must be double (8 bytes)");

  __device__ static void atomic_add(value_type* ptr, value_type val) noexcept {
    // For probabilistic OR: use compare-and-swap loop with double
    value_type old = *ptr;
    value_type expected = old;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      expected = old;
      value_type updated = add(old, val);
      old = __longlong_as_double(atomicCAS(reinterpret_cast<unsigned long long*>(ptr),
                                           __double_as_longlong(expected),
                                           __double_as_longlong(updated)));
    } while (old != expected);
  }
  __device__ static void atomic_mul(value_type* ptr, value_type val) noexcept {
    // For probabilistic AND: use compare-and-swap loop with double
    value_type old = *ptr;
    value_type expected = old;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      expected = old;
      value_type updated = mul(old, val);
      old = __longlong_as_double(atomicCAS(reinterpret_cast<unsigned long long*>(ptr),
                                           __double_as_longlong(expected),
                                           __double_as_longlong(updated)));
    } while (old != expected);
  }
#else
  // Host fallback: not truly atomic, but allows compilation
  // On HIP, this needs to be __device__ to be callable from kernels
  GPU_HD static void atomic_add(value_type* ptr, value_type val) noexcept {
    *ptr = add(*ptr, val);
  }
  GPU_HD static void atomic_mul(value_type* ptr, value_type val) noexcept {
    *ptr = mul(*ptr, val);
  }
#endif
};

template <>
struct semiring_traits<ProbIndep> {
  static constexpr bool add_idempotent = false;
  static constexpr bool mul_commutative = true;
  static constexpr bool has_provenance = true;
};

//
// EnhancedID Provenance
//
// Layout (64-bit):
// [63]    Flag (1 bit)
// [48-62] Scope (15 bits)
// [0-47]  ID (48 bits)
//
struct EnhancedID {
  using value_type = std::uint64_t;
  value_type data_;

  static constexpr value_type ID_MASK = 0x0000FFFFFFFFFFFFULL;
  static constexpr value_type SCOPE_MASK = 0x7FFF000000000000ULL;
  static constexpr value_type FLAG_MASK = 0x8000000000000000ULL;
  static constexpr int SCOPE_SHIFT = 48;
  static constexpr int FLAG_SHIFT = 63;

  GPU_HD EnhancedID() : data_(0) {}
  GPU_HD EnhancedID(value_type val) : data_(val) {}
  GPU_HD EnhancedID(value_type id, uint16_t scope, bool flag) {
    data_ = (id & ID_MASK) | ((static_cast<value_type>(scope) << SCOPE_SHIFT) & SCOPE_MASK) |
            (flag ? FLAG_MASK : 0);
  }

  GPU_HD value_type id() const {
    return data_ & ID_MASK;
  }

  GPU_HD uint16_t scope() const {
    return static_cast<uint16_t>((data_ & SCOPE_MASK) >> SCOPE_SHIFT);
  }

  GPU_HD bool flag() const {
    return (data_ & FLAG_MASK) != 0;
  }

  // Dereferencing returns the ID
  GPU_HD value_type operator*() const {
    return id();
  }

  GPU_HD bool operator==(const EnhancedID& other) const {
    return data_ == other.data_;
  }

  GPU_HD bool operator!=(const EnhancedID& other) const {
    return data_ != other.data_;
  }
};

struct EnhancedIDSR {
  using value_type = EnhancedID;

  GPU_HD static constexpr value_type zero() noexcept {
    return EnhancedID(0, 0, false);
  }

  GPU_HD static constexpr value_type one() noexcept {
    return EnhancedID(0, 0, true);  // Default "valid" state
  }

  // Add: Combine two IDs (Union).
  // Strategy: Keep the one with higher ID. If equal, OR the flags.
  GPU_HD GPU_FORCE_INLINE static value_type add(value_type a, value_type b) noexcept {
    if (a.id() > b.id())
      return a;
    if (b.id() > a.id())
      return b;
    return EnhancedID(a.id(), a.scope(), a.flag() || b.flag());
  }

  // Mul: Combine two IDs (Intersection/Join).
  // Strategy: If both valid, result is valid. Logic depends on app,
  // here we just return 'a' generally or handle scoping logic.
  // For now: arithmetic sum of IDs just to show combination?
  // No, let's say "Extension": new ID dominates if larger?
  // Let's implement simple "Last Writer Wins" or "Max" for now.
  GPU_HD GPU_FORCE_INLINE static value_type mul(value_type a, value_type b) noexcept {
    // A placeholder logic: In a join, we might want to track the 'latest' or 'max' provenance
    // or combine them.
    // Let's take Max ID.
    return (a.data_ > b.data_) ? a : b;
  }

#if defined(__CUDACC__) || defined(__HIP__)
  static_assert(sizeof(value_type) == 8, "EnhancedID must be 8 bytes");

  __device__ static void atomic_add(value_type* ptr, value_type val) noexcept {
    unsigned long long* u64_ptr = reinterpret_cast<unsigned long long*>(ptr);
    unsigned long long old = *u64_ptr;
    unsigned long long expected;
    do {
      expected = old;
      value_type current(old);
      value_type updated = add(current, val);
      old = atomicCAS(u64_ptr, expected, updated.data_);
    } while (old != expected);
  }

  __device__ static void atomic_mul(value_type* ptr, value_type val) noexcept {
    unsigned long long* u64_ptr = reinterpret_cast<unsigned long long*>(ptr);
    unsigned long long old = *u64_ptr;
    unsigned long long expected;
    do {
      expected = old;
      value_type current(old);
      value_type updated = mul(current, val);
      old = atomicCAS(u64_ptr, expected, updated.data_);
    } while (old != expected);
  }
#else
  GPU_HD static void atomic_add(value_type* ptr, value_type val) noexcept {
    *ptr = add(*ptr, val);
  }
  GPU_HD static void atomic_mul(value_type* ptr, value_type val) noexcept {
    *ptr = mul(*ptr, val);
  }
#endif
};

// Traits
template <>
struct semiring_traits<EnhancedIDSR> {
  static constexpr bool add_idempotent = true;
  static constexpr bool mul_commutative = true;
  static constexpr bool has_provenance = true;
};

//
// NoProvenance Sentinel Semiring
//
/// @brief Sentinel semiring indicating no provenance tracking.
/// @details When used as the SR template parameter, TMP dispatch via has_provenance_v
/// eliminates all provenance-related storage, computation, and memory traffic.
/// Use this for pure Boolean Datalog (set semantics) where provenance overhead is unnecessary.
struct NoProvenance {
  using value_type = std::monostate;  // Zero-size type (from <variant>)

  GPU_HD static constexpr value_type zero() noexcept {
    return {};
  }
  GPU_HD static constexpr value_type one() noexcept {
    return {};
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type add(value_type, value_type) noexcept {
    return {};
  }
  GPU_HD GPU_FORCE_INLINE static constexpr value_type mul(value_type, value_type) noexcept {
    return {};
  }
  GPU_HD static void atomic_add(value_type*, value_type) noexcept {}
  GPU_HD static void atomic_mul(value_type*, value_type) noexcept {}
};

template <>
struct semiring_traits<NoProvenance> {
  static constexpr bool add_idempotent = true;  // Vacuously true
  static constexpr bool mul_commutative = true;
  static constexpr bool has_provenance = false;  // Key TMP dispatch flag
};

//
// Helper wrappers for uniform call sites
//
template <Semiring SR>
using semiring_value_t = typename SR::value_type;

template <Semiring SR>
GPU_HD GPU_FORCE_INLINE constexpr semiring_value_t<SR> sr_zero() {
  return SR::zero();
}

template <Semiring SR>
GPU_HD GPU_FORCE_INLINE constexpr semiring_value_t<SR> sr_one() {
  return SR::one();
}

template <Semiring SR>
GPU_HD GPU_FORCE_INLINE constexpr semiring_value_t<SR> sr_add(semiring_value_t<SR> a,
                                                              semiring_value_t<SR> b) {
  return SR::add(a, b);
}

template <Semiring SR>
GPU_HD GPU_FORCE_INLINE constexpr semiring_value_t<SR> sr_mul(semiring_value_t<SR> a,
                                                              semiring_value_t<SR> b) {
  return SR::mul(a, b);
}

//
// Atomic operation wrappers
//
template <AtomicSemiring SR>
GPU_HD GPU_FORCE_INLINE void sr_atomic_add(semiring_value_t<SR>* ptr, semiring_value_t<SR> val) {
  SR::atomic_add(ptr, val);
}

template <AtomicSemiring SR>
GPU_HD GPU_FORCE_INLINE void sr_atomic_mul(semiring_value_t<SR>* ptr, semiring_value_t<SR> val) {
  SR::atomic_mul(ptr, val);
}

//
// TMP helper to check if provenance tracking is enabled
//
/// @brief Compile-time check for whether a semiring has provenance tracking enabled.
/// @details Use this in `if constexpr` to conditionally compile provenance-related code.
/// Returns false for NoProvenance, true for all other semirings (BooleanSR, NaturalBag, etc.)
template <Semiring SR>
inline constexpr bool has_provenance_v = semiring_traits<SR>::has_provenance;
