#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

// Macros for user-friendly aggregation functor definition
// These ensure the methods are inlined and device-compatible (when compiled with GPU toolchain)
#include "gpu/macro.h"  // Provides GPU_HD and GPU_FORCE_INLINE macros
#define SRD_AGG_INIT GPU_HD GPU_FORCE_INLINE
#define SRD_AGG_FUNC GPU_HD GPU_FORCE_INLINE

namespace SRDatalog {

/**
 * @brief Count aggregation functor.
 * Valid for any value type. Returns count as size_t (or uint32_t/uint64_t).
 */
struct AggCount {
  using ResultType = std::size_t;

  SRD_AGG_INIT static ResultType init() {
    return 0;
  }

  template <typename ValueType>
  SRD_AGG_FUNC static ResultType apply(ResultType acc, [[maybe_unused]] ValueType val) {
    return acc + 1;
  }

  SRD_AGG_FUNC static ResultType combine(ResultType a, ResultType b) {
    return a + b;
  }
};

/**
 * @brief Sum aggregation functor.
 * Valid for numeric value types.
 */
template <typename T>
struct AggSum {
  using ResultType = T;

  SRD_AGG_INIT static ResultType init() {
    return 0;
  }

  SRD_AGG_FUNC static ResultType apply(ResultType acc, T val) {
    return acc + val;
  }

  SRD_AGG_FUNC static ResultType combine(ResultType a, ResultType b) {
    return a + b;
  }
};

/**
 * @brief Min aggregation functor.
 */
template <typename T>
struct AggMin {
  using ResultType = T;

  SRD_AGG_INIT static ResultType init() {
    return std::numeric_limits<T>::max();
  }

  SRD_AGG_FUNC static ResultType apply(ResultType acc, T val) {
    return (val < acc) ? val : acc;
  }

  SRD_AGG_FUNC static ResultType combine(ResultType a, ResultType b) {
    return (a < b) ? a : b;
  }
};

/**
 * @brief Max aggregation functor.
 */
template <typename T>
struct AggMax {
  using ResultType = T;

  SRD_AGG_INIT static ResultType init() {
    return std::numeric_limits<T>::min();
  }

  SRD_AGG_FUNC static ResultType apply(ResultType acc, T val) {
    return (val > acc) ? val : acc;
  }

  SRD_AGG_FUNC static ResultType combine(ResultType a, ResultType b) {
    return (a > b) ? a : b;
  }
};

/**
 * @brief Exists aggregation functor (for Negation).
 * Returns true if any value exists (count > 0 conceptually, but boolean).
 */
struct AggExists {
  using ResultType = bool;

  SRD_AGG_INIT static ResultType init() {
    return false;
  }

  template <typename ValueType>
  SRD_AGG_FUNC static ResultType apply([[maybe_unused]] ResultType acc,
                                       [[maybe_unused]] ValueType val) {
    return true;  // Found one, result is true. Can short-circuit in executor?
  }

  SRD_AGG_FUNC static ResultType combine(ResultType a, ResultType b) {
    return a || b;
  }
};

}  // namespace SRDatalog
