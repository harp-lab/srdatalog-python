/**
 * @file search.h
 * @brief GPU-based group-cooperative search functions.
 *
 * @note Most functions have both group-based versions (accepting cooperative groups)
 *       and warp-based convenience wrappers for backward compatibility.
 *       Group-based versions support arbitrary group sizes including sub-warp groups
 *       and even sequential execution (group_size == 1).
 * Returns indices relative to input array (0 to len).
 */

#pragma once

#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#include "macro.h"

#include <cuda/std/limits>
#include <thrust/pair.h>

// Use platform-aware warp size from macro.h
#ifndef GPU_WARP_SIZE
#define WARP_SIZE 32  // Fallback for CPU-only builds
#else
#define WARP_SIZE GPU_WARP_SIZE
#endif

namespace SRDatalog::GPU {

/// @brief Group-cooperative lower_bound (works with arbitrary group sizes)
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] >= key, or len if all < key
/// @details Uses adaptive search: coalesced linear scan for small ranges (≤512),
///          binary search for larger ranges. Threshold based on benchmark results
///          showing linear scan is faster for small ranges due to 100% memory utilization.
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_lower_bound(Iterator data, IndexType len, T key,
                                                        Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());

  // Adaptive search threshold: use linear scan for small ranges
  // Benchmark results show linear scan wins for ranges up to ~512-1024 elements
  // due to coalesced memory access (100% utilization vs ~12.5% for binary search)
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 512;

  if (len <= LINEAR_SCAN_THRESHOLD) {
    // Small range: use coalesced linear scan directly
    // All threads in the warp read consecutive elements for 100% memory transaction utilization
    const IndexType lane = static_cast<IndexType>(tile.thread_rank());
    IndexType result = len;  // Default: not found (return len)

    for (IndexType base = 0; base < len; base += group_size) {
      IndexType idx = base + lane;
      T val = (idx < len) ? data[idx] : sentinel;
      // FIX: Only vote TRUE if index is valid. Lanes with idx >= len should NOT participate.
      // bool found = (idx < len) && (val >= key);
      bool found = (val >= key);
      auto mask = tile.ballot(found);
      if (mask != 0) {
        // Found first element >= key in this chunk
#if defined(USE_ROCm)
        IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
        IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
        result = base + first_set;
        break;
      }
    }
    return result;
  }

  IndexType lo = 0;
  IndexType hi = len;

// Binary search until range fits in the group
// For ROCm with 64-thread wavefronts, we need special handling only when group_size == 64
// For smaller groups (e.g., 32), use the standard logic
#if defined(USE_ROCm)
  if (group_size == 64) {
    // On ROCm with 64 threads, continue binary search until range <= 32 to ensure accurate parallel
    // scan This is because with 64 threads, we might have issues when hi - lo is between 33-64
    while (hi - lo > 32) {
      IndexType mid = lo + ((hi - lo) >> 1);
      if (data[mid] < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
  } else {
    // For smaller groups on ROCm, use standard logic
    while (hi - lo > group_size) {
      IndexType mid = lo + ((hi - lo) >> 1);
      if (data[mid] < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
  }
#else
  // For CUDA, use the original logic
  while (hi - lo > group_size) {
    IndexType mid = lo + ((hi - lo) >> 1);
    if (data[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
#endif

  // Parallel scan of final <= group_size elements
  // For sequential case (group_size == 1), this becomes a simple loop
  if (group_size == 1) {
    // Sequential case: simple linear search
    IndexType idx = lo;
    while (idx < hi && data[idx] < key) {
      ++idx;
    }
    return idx;
  }

  // Parallel case: use cooperative group primitives
  IndexType lane = static_cast<IndexType>(tile.thread_rank());
  IndexType idx = lo + lane;
  // Only threads with valid indices participate in the comparison
  // Threads beyond hi will read sentinel, which is > any key, so they won't contribute to count
  T val = (idx < hi) ? data[idx] : sentinel;

// FIX: Use CG API consistently - tile.ballot() returns a mask, but we should use CG API for
// counting too However, CG API doesn't provide popcount, so we need manual popcount but ensure
// consistency On ROCm, ensure threads beyond hi don't participate in ballot This is critical when
// hi - lo < group_size (64 threads but fewer elements)
#if defined(USE_ROCm)
  // Only participate in ballot if we're in the valid range
  bool participate = (idx < hi);
  unsigned long long mask = tile.ballot(participate && (val < key));
#else
  unsigned long long mask = tile.ballot(val < key);
#endif

  // Count number of set bits in mask (population count)
  // CRITICAL: Use tile.size() to determine which popcount to use, not hardcoded values
  // This ensures consistency between CG API (tile.ballot) and manual popcount
  IndexType count;
#if defined(USE_ROCm)
  // ROCm: CDNA (MI series) has 64-thread wavefronts, RDNA (RX series) has 32-thread wavefronts
  // Use tile.size() to determine actual wavefront size at runtime
  // For now, assume 64 for CDNA (most common for compute), but this should be runtime-detected
  if (group_size <= 32) {
    // RDNA or sub-warp: use 32-bit popcount
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    // CDNA: use 64-bit popcount
    count = static_cast<IndexType>(__popcll(mask));
  }
#else
// CUDA uses 32-thread warps
#ifdef GPU_WARP_SIZE
  if (group_size <= GPU_WARP_SIZE) {
#else
  if (group_size <= 32) {  // Fallback
#endif
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#endif

  return lo + count;
}

/// @brief Group-cooperative parallel exponential (galloping) lower_bound
/// @note Simplified hybrid approach:
///       - For ranges ≤256: coalesced linear scan (100% memory utilization)
///       - For ranges >256: exponential galloping to find range, then binary search
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] >= key, or len if all < key
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_exp_lower_bound(Iterator data, IndexType len, T key,
                                                            Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());
  const IndexType lane = static_cast<IndexType>(tile.thread_rank());

  if (len == 0) {
    return 0;
  }

  // Threshold: use linear scan for small ranges (≤2048 elements)
  // Coalesced reads give 100% memory transaction utilization
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 2048;

  // ============================================================================
  // Small range: coalesced linear scan
  // ============================================================================
  if (len <= LINEAR_SCAN_THRESHOLD) {
    if (group_size == 1) {
      // Sequential fallback
      IndexType idx = 0;
      while (idx < len && data[idx] < key) {
        ++idx;
      }
      return idx;
    }

    IndexType result = len;  // Default: not found
    for (IndexType base = 0; base < len; base += group_size) {
      IndexType idx = base + lane;
      T val = (idx < len) ? data[idx] : sentinel;
      bool found = (val >= key);
      auto mask = tile.ballot(found);
      if (mask != 0) {
#if defined(USE_ROCm)
        IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
        IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
        result = base + first_set;
        break;
      }
    }
    return result;
  }

  // ============================================================================
  // Large range: K-ary parallel narrowing
  // Each iteration narrows the range by factor `group_size` using parallel
  // probing (all threads probe `group_size` positions evenly, then ballot
  // finds the slab containing the answer). O(log_K(N)) iterations, each with
  // one memory read per thread + one ballot.
  // Verified 1.1-4.7x faster than old sequential gallop in unit tests.
  // ============================================================================
  IndexType lo = 0;
  IndexType hi = len;
  while (hi - lo > group_size) {
    IndexType range = hi - lo;
    // Each lane probes position lo + lane * (range - 1) / (group_size - 1)
    // Lane 0 probes lo; lane (group_size-1) probes hi-1.
    IndexType idx = lo + (static_cast<uint64_t>(lane) * (range - 1)) / (group_size - 1);
    if (idx >= hi)
      idx = hi - 1;
    T val = data[idx];
    auto mask = tile.ballot(val < key);
#if defined(USE_ROCm)
    IndexType count;
    if (group_size <= 32) {
      count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
    } else {
      count = static_cast<IndexType>(__popcll(mask));
    }
#else
    IndexType count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
#endif
    // count = number of lanes where probe < key → answer is between probe_{count-1} and probe_count
    if (count == 0) {
      // Answer is at lo or earlier; but lo is already smallest, so answer in [lo, probe_1_idx]
      // Narrow hi to where lane 1 would probe + 1
      IndexType probe1 = lo + (static_cast<uint64_t>(1) * (range - 1)) / (group_size - 1);
      hi = probe1 + 1;
    } else if (count == group_size) {
      // All probes < key → answer after last probe
      IndexType probe_last =
          lo + (static_cast<uint64_t>(group_size - 1) * (range - 1)) / (group_size - 1);
      lo = probe_last + 1;
    } else {
      IndexType new_lo =
          lo + (static_cast<uint64_t>(count - 1) * (range - 1)) / (group_size - 1) + 1;
      IndexType new_hi = lo + (static_cast<uint64_t>(count) * (range - 1)) / (group_size - 1) + 1;
      lo = new_lo;
      hi = new_hi;
    }
  }

  // Final phase: parallel ballot scan for remaining group_size elements
  IndexType idx = lo + lane;
  T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
  bool participate = (idx < hi);
  unsigned long long mask = tile.ballot(participate && (val < key));
#else
  unsigned long long mask = tile.ballot(val < key);
#endif

  IndexType count;
#if defined(USE_ROCm)
  if (group_size <= 32) {
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#else
#ifdef GPU_WARP_SIZE
  if (group_size <= GPU_WARP_SIZE) {
#else
  if (group_size <= 32) {
#endif
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#endif

  return lo + count;
}

/// @brief Group-cooperative parallel exponential (galloping) upper_bound
/// @note Simplified hybrid approach (mirrors group_exp_lower_bound):
///       - For ranges ≤256: coalesced linear scan (100% memory utilization)
///       - For ranges >256: exponential galloping to find range, then binary search
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] > key, or len if all <= key
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_exp_upper_bound(Iterator data, IndexType len, T key,
                                                            Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());
  const IndexType lane = static_cast<IndexType>(tile.thread_rank());

  if (len == 0) {
    return 0;
  }

  // Threshold: use linear scan for small ranges (1024 elements)
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 2048;

  // ============================================================================
  // Small range: coalesced linear scan
  // ============================================================================
  if (len <= LINEAR_SCAN_THRESHOLD) {
    if (group_size == 1) {
      // Sequential fallback
      IndexType idx = 0;
      while (idx < len && data[idx] <= key) {  // upper_bound: find first > key
        ++idx;
      }
      return idx;
    }

    IndexType result = len;  // Default: not found
    for (IndexType base = 0; base < len; base += group_size) {
      IndexType idx = base + lane;
      T val = (idx < len) ? data[idx] : sentinel;
      bool found = (idx < len) && (val > key);  // upper_bound: val > key
      auto mask = tile.ballot(found);
      if (mask != 0) {
#if defined(USE_ROCm)
        IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
        IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
        result = base + first_set;
        break;
      }
    }
    return result;
  }

  // ============================================================================
  // Large range: K-ary parallel narrowing (same strategy as lower_bound)
  // For upper_bound, the comparison is `val <= key` instead of `val < key`.
  // ============================================================================
  IndexType lo = 0;
  IndexType hi = len;
  while (hi - lo > group_size) {
    IndexType range = hi - lo;
    IndexType idx = lo + (static_cast<uint64_t>(lane) * (range - 1)) / (group_size - 1);
    if (idx >= hi)
      idx = hi - 1;
    T val = data[idx];
    auto mask = tile.ballot(val <= key);  // upper_bound: count lanes with val <= key
#if defined(USE_ROCm)
    IndexType count;
    if (group_size <= 32) {
      count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
    } else {
      count = static_cast<IndexType>(__popcll(mask));
    }
#else
    IndexType count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
#endif
    if (count == 0) {
      IndexType probe1 = lo + (static_cast<uint64_t>(1) * (range - 1)) / (group_size - 1);
      hi = probe1 + 1;
    } else if (count == group_size) {
      IndexType probe_last =
          lo + (static_cast<uint64_t>(group_size - 1) * (range - 1)) / (group_size - 1);
      lo = probe_last + 1;
    } else {
      IndexType new_lo =
          lo + (static_cast<uint64_t>(count - 1) * (range - 1)) / (group_size - 1) + 1;
      IndexType new_hi = lo + (static_cast<uint64_t>(count) * (range - 1)) / (group_size - 1) + 1;
      lo = new_lo;
      hi = new_hi;
    }
  }

  // Final phase: parallel ballot scan for remaining group_size elements
  IndexType idx = lo + lane;
  T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
  bool participate = (idx < hi);
  unsigned long long mask = tile.ballot(participate && (val <= key));
#else
  unsigned long long mask = tile.ballot(val <= key);
#endif

  IndexType count;
#if defined(USE_ROCm)
  if (group_size <= 32) {
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#else
#ifdef GPU_WARP_SIZE
  if (group_size <= GPU_WARP_SIZE) {
#else
  if (group_size <= 32) {
#endif
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#endif

  return lo + count;
}

/// @brief Group-cooperative linear lower_bound with hybrid approach
/// @note For ranges ≤ 512 elements, uses fully coalesced warp-parallel linear scan.
///       For larger ranges, uses binary search to narrow to 512 elements first, then linear scan.
///       Benchmark shows linear scan wins for ranges ≤ 1K elements (~22% faster).
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] >= key, or len if all < key
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_linear_lower_bound(Iterator data, IndexType len, T key,
                                                               Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());

  // Threshold: benchmark shows linear scan wins for ranges ≤ 1024 elements
  // Use 512 as conservative threshold for binary → linear switch
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 512;

  // Handle empty array
  if (len == 0) {
    return 0;
  }

  IndexType lo = 0;
  IndexType hi = len;

  // ============================================================================
  // Phase 1: Binary search to narrow to linear scan threshold
  // ============================================================================
  while (hi - lo > LINEAR_SCAN_THRESHOLD) {
    IndexType mid = lo + ((hi - lo) >> 1);
    if (data[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // ============================================================================
  // Phase 2: Linear scan with full warp parallelism (coalesced access)
  // ============================================================================
  if (group_size == 1) {
    // Sequential fallback
    IndexType idx = lo;
    while (idx < hi && data[idx] < key) {
      ++idx;
    }
    return idx;
  }

  IndexType lane = static_cast<IndexType>(tile.thread_rank());

  // Linear scan: process group_size elements per iteration (coalesced reads)
  while (hi - lo > group_size) {
    IndexType idx = lo + lane;
    T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
    bool participate = (idx < hi);
    unsigned long long mask = tile.ballot(participate && (val >= key));
#else
    unsigned long long mask = tile.ballot(val >= key);
#endif

    if (mask != 0) {
      // Found at least one element >= key, count how many are < key
#if defined(USE_ROCm)
      unsigned long long lt_mask = tile.ballot(participate && (val < key));
#else
      unsigned long long lt_mask = tile.ballot(val < key);
#endif
      IndexType count;
      if (group_size <= 32) {
        count = static_cast<IndexType>(__popc(static_cast<uint32_t>(lt_mask)));
      } else {
        count = static_cast<IndexType>(__popcll(lt_mask));
      }
      return lo + count;
    }

    // All elements in this batch are < key, advance to next batch
    lo += group_size;
  }

  // ============================================================================
  // Phase 3: Final ballot scan for remaining elements
  // ============================================================================
  IndexType idx = lo + lane;
  T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
  bool participate = (idx < hi);
  unsigned long long mask = tile.ballot(participate && (val < key));
#else
  unsigned long long mask = tile.ballot(val < key);
#endif

  IndexType count;
#if defined(USE_ROCm)
  if (group_size <= 32) {
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#else
#ifdef GPU_WARP_SIZE
  if (group_size <= GPU_WARP_SIZE) {
#else
  if (group_size <= 32) {
#endif
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#endif

  return lo + count;
}

/// @brief Warp-cooperative lower_bound (convenience wrapper for backward compatibility)
/// @deprecated Use group_lower_bound with explicit tile parameter to support tree-shape
/// decomposition
/// @return First index where data[i] >= key, or len if all < key
template <typename T, typename IndexType = uint32_t,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
[[deprecated("Use group_lower_bound with explicit tile parameter")]]
__device__ GPU_FORCE_INLINE IndexType warp_lower_bound(Iterator data, IndexType len, T key) {
  auto tile =
      cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
  return group_lower_bound<T, IndexType, decltype(tile), sentinel>(data, len, key, tile);
}

/// @brief Group-cooperative upper_bound (works with arbitrary group sizes)
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] > key, or len if all <= key
/// @details Uses adaptive search: linear scan for small (≤512), exponential for medium (512-2048),
///          binary search for large ranges. Exponential search is faster for medium ranges.
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_upper_bound(Iterator data, IndexType len, T key,
                                                        Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());
  const IndexType lane = static_cast<IndexType>(tile.thread_rank());

  // Adaptive search thresholds
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 512;
  constexpr IndexType EXP_SEARCH_THRESHOLD = 512;  // Disabled: same as LINEAR to skip exp path

  if (len <= LINEAR_SCAN_THRESHOLD) {
    // Small range: use coalesced linear scan directly for upper_bound
    IndexType result = len;

    for (IndexType base = 0; base < len; base += group_size) {
      IndexType idx = base + lane;
      T val = (idx < len) ? data[idx] : sentinel;
      // FIX: Only vote if within bounds. sentinel > key is TRUE which causes spurious matches!
      bool found = (idx < len) && (val > key);  // upper_bound: first element > key
      auto mask = tile.ballot(found);
      if (mask != 0) {
#if defined(USE_ROCm)
        IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
        IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
        result = base + first_set;
        break;
      }
    }
    return result;
  }

  if (len <= EXP_SEARCH_THRESHOLD) {
    // Medium range: use exponential (galloping) search
    // Phase 1: Parallel galloping to find the range containing the answer
    IndexType offset = 0;
    IndexType jump = group_size;  // Start with group_size, double each iteration

    while (offset + jump <= len) {
      // Each lane reads one element from [offset, offset + group_size)
      IndexType idx = offset + lane;
      T val = (idx < len) ? data[idx] : sentinel;

      // Check if any lane found data > key
      bool found = (idx < len) && (val > key);
      auto mask = tile.ballot(found);

      if (mask != 0) {
        // Found! The answer is in [offset, offset + group_size)
        // Break and refine with binary search
        break;
      }

      // All elements <= key, jump forward
      offset += jump;
      jump <<= 1;  // Double the jump for exponential growth

      // Cap jump to avoid overshooting
      if (offset + jump > len) {
        jump = len - offset;
      }
    }

    // Now we know: all elements before 'offset' are <= key
    // The answer is in [offset, min(offset + jump, len))
    IndexType lo = offset;
    IndexType hi = (offset + jump < len) ? (offset + jump) : len;

    // Phase 2: Binary search to narrow to group_size elements
    while (hi - lo > group_size) {
      IndexType mid = lo + ((hi - lo) >> 1);
      if (data[mid] <= key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    // Phase 3: Parallel ballot scan for final group_size elements
    IndexType idx = lo + lane;
    T val = (idx < hi) ? data[idx] : sentinel;
    bool found = (idx < hi) && (val > key);  // upper_bound: first element > key
    auto mask = tile.ballot(found);

    if (mask != 0) {
#if defined(USE_ROCm)
      IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
      IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
      return lo + first_set;
    }
    return hi;  // All elements <= key, return hi (which is upper_bound)
  }

  // Large range: use binary search to narrow to LINEAR_SCAN_THRESHOLD
  IndexType lo = 0;
  IndexType hi = len;

  // Binary search until range fits in LINEAR_SCAN_THRESHOLD
  // Then hand off to linear scan which is faster for small ranges
  while (hi - lo > LINEAR_SCAN_THRESHOLD) {
    IndexType mid = lo + ((hi - lo) >> 1);
    // DEBUG: Check for out-of-bounds before access
    if (mid >= len) {
      printf("[SEARCH_DEBUG] OOB! mid=%u >= len=%u, lo=%u, hi=%u\\n", mid, len, lo, hi);
    }
    if (data[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Linear scan for remaining <= LINEAR_SCAN_THRESHOLD elements
  // This is the same logic as the linear scan path above
  IndexType result = hi;
  for (IndexType base = lo; base < hi; base += group_size) {
    IndexType idx = base + lane;
    T val = (idx < hi) ? data[idx] : sentinel;
    bool found = (idx < hi) && (val > key);  // upper_bound: first element > key
    auto mask = tile.ballot(found);
    if (mask != 0) {
#if defined(USE_ROCm)
      IndexType first_set = static_cast<IndexType>(__ffsll(mask) - 1);
#else
      IndexType first_set = static_cast<IndexType>(__ffs(static_cast<int>(mask)) - 1);
#endif
      return base + first_set;
    }
  }
  return result;
}

/// @brief Group-cooperative linear upper_bound with hybrid approach
/// @note For ranges ≤ 512 elements, uses fully coalesced warp-parallel linear scan.
///       For larger ranges, uses binary search to narrow to 512 elements first, then linear scan.
///       Benchmark shows linear scan wins for ranges ≤ 1K elements (~22% faster).
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return First index where data[i] > key, or len if all <= key
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_linear_upper_bound(Iterator data, IndexType len, T key,
                                                               Group tile) {
  const IndexType group_size = static_cast<IndexType>(tile.size());

  // Threshold: benchmark shows linear scan wins for ranges ≤ 1024 elements
  // Use 512 as conservative threshold for binary → linear switch
  constexpr IndexType LINEAR_SCAN_THRESHOLD = 512;

  // Handle empty array
  if (len == 0) {
    return 0;
  }

  IndexType lo = 0;
  IndexType hi = len;

  // ============================================================================
  // Phase 1: Binary search to narrow to linear scan threshold
  // ============================================================================
  while (hi - lo > LINEAR_SCAN_THRESHOLD) {
    IndexType mid = lo + ((hi - lo) >> 1);
    if (data[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // ============================================================================
  // Phase 2: Linear scan with full warp parallelism (coalesced access)
  // ============================================================================
  if (group_size == 1) {
    // Sequential fallback
    IndexType idx = lo;
    while (idx < hi && data[idx] <= key) {
      ++idx;
    }
    return idx;
  }

  IndexType lane = static_cast<IndexType>(tile.thread_rank());

  // Linear scan: process group_size elements per iteration (coalesced reads)
  while (hi - lo > group_size) {
    IndexType idx = lo + lane;
    T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
    bool participate = (idx < hi);
    unsigned long long mask = tile.ballot(participate && (val > key));
#else
    unsigned long long mask = tile.ballot(val > key);
#endif

    if (mask != 0) {
      // Found at least one element > key, count how many are <= key
#if defined(USE_ROCm)
      unsigned long long le_mask = tile.ballot(participate && (val <= key));
#else
      unsigned long long le_mask = tile.ballot(val <= key);
#endif
      IndexType count;
      if (group_size <= 32) {
        count = static_cast<IndexType>(__popc(static_cast<uint32_t>(le_mask)));
      } else {
        count = static_cast<IndexType>(__popcll(le_mask));
      }
      return lo + count;
    }

    // All elements in this batch are <= key, advance to next batch
    lo += group_size;
  }

  // ============================================================================
  // Phase 3: Final ballot scan for remaining elements
  // ============================================================================
  IndexType idx = lo + lane;
  T val = (idx < hi) ? data[idx] : sentinel;

#if defined(USE_ROCm)
  bool participate = (idx < hi);
  unsigned long long mask = tile.ballot(participate && (val <= key));
#else
  unsigned long long mask = tile.ballot(val <= key);
#endif

  IndexType count;
#if defined(USE_ROCm)
  if (group_size <= 32) {
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#else
#ifdef GPU_WARP_SIZE
  if (group_size <= GPU_WARP_SIZE) {
#else
  if (group_size <= 32) {
#endif
    count = static_cast<IndexType>(__popc(static_cast<uint32_t>(mask)));
  } else {
    count = static_cast<IndexType>(__popcll(mask));
  }
#endif

  return lo + count;
}

/// @brief Warp-cooperative upper_bound (convenience wrapper for backward compatibility)
/// @deprecated Use group_upper_bound with explicit tile parameter to support tree-shape
/// decomposition
/// @return First index where data[i] > key, or len if all <= key
template <typename T, typename IndexType = uint32_t,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
[[deprecated("Use group_upper_bound with explicit tile parameter")]]
__device__ GPU_FORCE_INLINE IndexType warp_upper_bound(Iterator data, IndexType len, T key) {
  auto tile =
      cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
  return group_upper_bound<T, IndexType, decltype(tile), sentinel>(data, len, key, tile);
}

/// @brief Group-cooperative equal_range (works with arbitrary group sizes)
/// @param tile Cooperative group (can be warp, sub-warp, or even sequential)
/// @return {lower_bound, upper_bound} pair
template <typename T, typename IndexType = uint32_t, typename Group, typename Iterator>
__device__ GPU_FORCE_INLINE thrust::pair<IndexType, IndexType> group_equal_range(Iterator data,
                                                                                 IndexType len,
                                                                                 T key,
                                                                                 Group tile) {
  // Use exponential search for better performance on large arrays
  IndexType lb = group_exp_lower_bound<T, IndexType>(data, len, key, tile);
  if (lb == len) {
    return {len, len};
  }
  if (data[lb] != key) {
    return {lb, lb};
  }
  // For sub-range search, we need to adjust the key comparison
  // group_exp_upper_bound searches in [data + lb, data + len), so we pass the adjusted length
  IndexType ub = group_exp_upper_bound<T, IndexType, Group>(data + lb, len - lb, key, tile);
  return {lb, lb + ub};
}

/// @brief Warp-cooperative equal_range (convenience wrapper for backward compatibility)
/// @deprecated Use group_equal_range with explicit tile parameter to support tree-shape
/// decomposition
/// @return {lower_bound, upper_bound} pair
template <typename T, typename IndexType = uint32_t, typename Iterator>
[[deprecated("Use group_equal_range with explicit tile parameter")]]
__device__ GPU_FORCE_INLINE thrust::pair<IndexType, IndexType> warp_equal_range(Iterator data,
                                                                                IndexType len,
                                                                                T key) {
  auto tile =
      cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
  return group_equal_range<T, IndexType>(data, len, key, tile);
}

/// @brief Group-cooperative contains check (works with arbitrary group sizes)
/// @param tile Cooperative group (can be warp, sub-warp, or single-thread)
/// @return true if key exists in sorted array
/// @note Optimized for binary join: uses single lower_bound + equality check
///       instead of full equal_range (saves 50% search work for semijoin checks)
template <typename T, typename IndexType = uint32_t, typename Group, typename Iterator>
__device__ GPU_FORCE_INLINE bool group_contains(Iterator data, IndexType len, T key, Group tile) {
  IndexType lb = group_lower_bound<T, IndexType>(data, len, key, tile);
  // Single equality check after lower_bound - no need for upper_bound
  return (lb < len && data[lb] == key);
}

/// @brief Check if key exists in sorted array
/// @deprecated Use group_contains with explicit tile parameter to support tree-shape decomposition
template <typename T, typename IndexType = uint32_t,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
[[deprecated("Use group_contains with explicit tile parameter")]]
__device__ GPU_FORCE_INLINE bool warp_contains(Iterator data, IndexType len, T key) {
  auto [lo, hi] = warp_equal_range<T, IndexType>(data, len, key);
  return lo < hi;
}

/// @brief Group-cooperative find (works with arbitrary group sizes)
/// @param tile Cooperative group (can be warp, sub-warp, or single-thread)
/// @return Position of key if found, len if not found
template <typename T, typename IndexType = uint32_t, typename Group,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
__device__ GPU_FORCE_INLINE IndexType group_find(Iterator data, IndexType len, T key, Group tile) {
  IndexType pos = group_lower_bound<T, IndexType, Group, sentinel>(data, len, key, tile);
  return (pos < len && data[pos] == key) ? pos : len;
}

/// @brief Find exact key, returns len if not found
/// @deprecated Use group_find with explicit tile parameter to support tree-shape decomposition
template <typename T, typename IndexType = uint32_t,
          T sentinel = cuda::std::numeric_limits<T>::max(), typename Iterator>
[[deprecated("Use group_find with explicit tile parameter")]]
__device__ GPU_FORCE_INLINE IndexType warp_find(Iterator data, IndexType len, T key) {
  IndexType pos = warp_lower_bound<T, IndexType, sentinel>(data, len, key);
  return (pos < len && data[pos] == key) ? pos : len;
}

// =============================================================================
// Sequential (non-cooperative) search functions
// For use inside Cartesian loops where each thread has a different key
// =============================================================================

/// @brief Sequential lower_bound - standard binary search, no cooperative primitives
template <typename T, typename IndexType = uint32_t, typename Iterator>
__device__ GPU_FORCE_INLINE IndexType seq_lower_bound(Iterator data, IndexType len, T key) {
  IndexType lo = 0;
  IndexType hi = len;
  while (lo < hi) {
    IndexType mid = lo + ((hi - lo) >> 1);
    if (data[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/// @brief Sequential upper_bound - standard binary search, no cooperative primitives
template <typename T, typename IndexType = uint32_t, typename Iterator>
__device__ GPU_FORCE_INLINE IndexType seq_upper_bound(Iterator data, IndexType len, T key) {
  IndexType lo = 0;
  IndexType hi = len;
  while (lo < hi) {
    IndexType mid = lo + ((hi - lo) >> 1);
    if (data[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/// @brief Sequential equal_range - for single-thread search without cooperative primitives
template <typename T, typename IndexType = uint32_t, typename Iterator>
__device__ GPU_FORCE_INLINE thrust::pair<IndexType, IndexType> seq_equal_range(Iterator data,
                                                                               IndexType len,
                                                                               T key) {
  IndexType lb = seq_lower_bound<T, IndexType>(data, len, key);
  if (lb == len || data[lb] != key) {
    return {lb, lb};
  }
  IndexType ub = seq_upper_bound<T, IndexType>(data + lb, len - lb, key);
  return {lb, lb + ub};
}

/// @brief Sequential contains check - single lower_bound + equality (optimized for binary join)
/// @return true if key exists in sorted array
template <typename T, typename IndexType = uint32_t, typename Iterator>
__device__ GPU_FORCE_INLINE bool seq_contains(Iterator data, IndexType len, T key) {
  IndexType lb = seq_lower_bound<T, IndexType>(data, len, key);
  return (lb < len && data[lb] == key);
}

}  // namespace SRDatalog::GPU
