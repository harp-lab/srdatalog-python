#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Highway SIMD library
#include "hwy/highway.h"

// -------------------------------------------------------------------------
// Generic Scalar Implementation (Used for non-SIMD types or huge fallbacks)
// -------------------------------------------------------------------------
namespace SRDatalog::search {

/**
 * @brief Arithmetic Branchless Binary Search
 * @details Uses pure arithmetic (no 'if' or ternary operators) to update
 * indices. This forces the compiler to use CMOV/Arithmetic instead
 * of Jump instructions (jae/jb), eliminating pipeline flushes
 * on large random arrays.
 */
template <typename T>
inline std::size_t branchless_lower_bound(const T* base, std::size_t len, T val) {
  std::size_t idx = 0;

  while (len > 0) {
    std::size_t half = len >> 1;
    std::size_t mid = idx + half;

    // Comparison result: 1 if Go Right, 0 if Go Left.
    // Casting to integer prevents compiler from optimizing back to branches.
    std::size_t cmp = (base[mid] < val);

    // Arithmetic Update:
    // If Right: idx += half + 1
    // If Left:  idx += 0
    idx += cmp * (half + 1);

    // Length Update:
    // If Right: len = len - half - 1
    // If Left:  len = half
    // Formula: half + cmp * (remainder - half)
    // Simplified: half + cmp * (len - 2*half - 1)
    len = half + cmp * (len - (half << 1) - 1);
  }

  return idx;
}

}  // namespace SRDatalog::search

// -------------------------------------------------------------------------
// Highway SIMD Implementation (Target Specific)
// -------------------------------------------------------------------------
HWY_BEFORE_NAMESPACE();
namespace SRDatalog::search {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Pure SIMD Linear Scan
 * Checks elements in parallel using AVX/NEON/SVE.
 */
template <typename T>
HWY_INLINE std::size_t simd_linear_scan(const T* HWY_RESTRICT base, std::size_t len, T val) {
  const hn::ScalableTag<T> d;
  const std::size_t lanes_count = hn::Lanes(d);

  std::size_t idx = 0;

  // Process full vectors
  if (len >= lanes_count) {
    const auto target = hn::Set(d, val);

    while (len >= lanes_count) {
      auto data = hn::LoadU(d, base + idx);

      // Compare: mask bits set where data[i] < val
      auto lt_mask = hn::Lt(data, target);

      // If NOT all are less than val, we found the transition point (>= val)
      if (!hn::AllTrue(d, lt_mask)) {
        auto ge_mask = hn::Not(lt_mask);
        std::size_t first_ge_lane = hn::FindFirstTrue(d, ge_mask);
        return idx + first_ge_lane;
      }

      idx += lanes_count;
      len -= lanes_count;
    }
  }

  // Scalar fallback for the tail
  for (std::size_t i = 0; i < len; ++i) {
    if (base[idx + i] >= val) {
      return idx + i;
    }
  }

  return idx + len;
}

/**
 * @brief Hybrid Search: Branchless Binary Search + SIMD Scan
 * * 1. Uses Arithmetic Binary Search to narrow range from N -> 128.
 * 2. Uses SIMD Linear Scan for the final 128 elements.
 */
template <typename T>
HWY_INLINE std::size_t highway_hybrid_lower_bound(const T* base, std::size_t len, T val) {
  // Optimization for tiny arrays (avoid SIMD setup overhead)
  // Optimization for tiny arrays (avoid SIMD setup overhead)
  if (len <= 4) {
    for (std::size_t i = 0; i < len; ++i) {
      if (base[i] >= val)
        return i;
    }
    return len;
  }

  // Threshold: When to switch from Logarithmic to Linear.
  // 128 is optimal for AVX2 (4 unrolled YMM loads).
  constexpr std::size_t kSimdThreshold = 128;

  // Phase 1: Narrow range using Branchless Binary Search
  std::size_t left = 0;
  std::size_t range = len;

  while (range > kSimdThreshold) {
    std::size_t half = range >> 1;
    std::size_t mid = left + half;

    // Arithmetic Logic (No Jumps)
    std::size_t cmp = (base[mid] < val);

    left += cmp * (half + 1);
    range = half + cmp * (range - (half << 1) - 1);
  }

  // Phase 2: SIMD Scan on the remaining window
  // Range is guaranteed <= kSimdThreshold here.
  return left + simd_linear_scan(base + left, range, val);
}

}  // namespace HWY_NAMESPACE
}  // namespace SRDatalog::search
HWY_AFTER_NAMESPACE();

// -------------------------------------------------------------------------
// Public Dispatcher
// -------------------------------------------------------------------------
#if HWY_ONCE
namespace SRDatalog::search {

/**
 * @brief Dispatch wrapper for Highway implementation
 */
template <typename T>
HWY_INLINE std::size_t simd_lower_bound(const T* base, std::size_t len, T val) {
  return HWY_STATIC_DISPATCH(highway_hybrid_lower_bound)(base, len, val);
}

/**
 * @brief Adaptive Lower Bound (The Main API)
 * * Automatically selects the best strategy:
 * - Non-Integrals: Generic scalar branchless search.
 * - Integrals (32/64): Highway Hybrid (Binary -> SIMD).
 */
template <typename T>
inline std::size_t adaptive_lower_bound(const T* base, std::size_t len, T val) {
  // Only use SIMD path for types supported by our logic (int32/64, uint32/64)
  if constexpr (std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
    return simd_lower_bound(base, len, val);
  } else {
    return branchless_lower_bound(base, len, val);
  }
}

}  // namespace SRDatalog::search
#endif  // HWY_ONCE