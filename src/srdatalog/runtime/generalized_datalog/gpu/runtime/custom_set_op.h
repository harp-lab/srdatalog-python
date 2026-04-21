/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This file is a modified version of the set_diffference in NVIDIA's thrust library.
 * The original code can be found at
 *https://github.com/NVIDIA/thrust/blob/master/thrust/set_operations.h
 *
 ******************************************************************************/
#pragma once

#ifdef USE_ROCm
// Include CCCL compatibility shim before thrust headers
#include "gpu/shim/thrust/cccl_compat.h"
// Include HIP target macros (NV_IF_TARGET, etc.) before CUDA backend headers
// This ensures CUDA backend code can use HIP-specific macros
#include <thrust/system/hip/detail/nv/detail/__target_macros.h>
// Include CUDA error constants shim before thrust/system/cuda headers
#include "gpu/shim/thrust/system/cuda/cuda_error_constants.h"
#endif

// Include GPU API abstraction for hipMemcpy, hipStreamSynchronize, etc.
// Use GPU API abstraction layer for both CUDA and ROCm
#include "gpu/gpu_api.h"

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif  // no system header

#ifdef USE_ROCm
// On ROCm, use hipcub which provides CUB compatibility
#include <hipcub/block/block_load.hpp>
#include <hipcub/iterator/cache_modified_input_iterator.hpp>
#else
#include <cub/block/block_load.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#endif

#include <thrust/detail/alignment.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/extrema.h>
#include <thrust/set_operations.h>
#include <thrust/system/cuda/detail/cdp_dispatch.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/system/cuda/detail/util.h>

#ifdef USE_ROCm
// On HIP, also include HIP execution policy for compatibility
// Note: rocThrust uses CUDA backend headers but with HIP execution policies
#include <thrust/system/hip/detail/execution_policy.h>
#endif

#ifdef USE_ROCm
// HIP/ROCm workaround: HIP dispatch.h expects thrust::detail::int32_t
// Provide it if not already defined
#include <cstdint>
THRUST_NAMESPACE_BEGIN
namespace detail {
using int32_t = ::std::int32_t;
using int64_t = ::std::int64_t;
}  // namespace detail
THRUST_NAMESPACE_END

// Workaround: HIP dispatch.h macro uses const_max() but it's a static constexpr value
// Redefine THRUST_DOUBLE_INDEX_TYPE_DISPATCH to work correctly
// Use a simplified version that always uses int64_t (matches CUDA behavior when
// THRUST_FORCE_64_BIT_OFFSET_TYPE is not set)
#undef THRUST_DOUBLE_INDEX_TYPE_DISPATCH
#define THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, call, count1, count2, arguments)                 \
  {                                                                                                \
    auto THRUST_PP_CAT2(count1, _fixed) = static_cast<std::int64_t>(count1);                       \
    auto THRUST_PP_CAT2(count2, _fixed) = static_cast<std::int64_t>(count2);                       \
    status = call arguments;                                                                       \
  }
#endif

#ifdef USE_ROCm
// On ROCm, use std headers (following macro.h pattern: namespace cuda_std = std)
// Thrust code uses ::cuda::std:: types, which we map via namespace alias in cuda_error_constants.h
#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>
#else
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#endif

#ifdef USE_ROCm
// nvtx3 is NVIDIA-specific profiling - not available on ROCm
// ROCm uses roctx, but for this code we can just skip the profiling markers
#define nvtxRangePushA(name) ((void)0)
#define nvtxRangePop() ((void)0)
#else
#include <nvtx3/nvToolsExt.h>
#endif
#include <thrust/system/cuda/detail/core/util.h>

#define _CUDA_CUB_RET_IF_FAIL_CUSTOM(e)                                                            \
  {                                                                                                \
    auto const error = (e);                                                                        \
    if (cub::Debug(error, __FILE__, __LINE__))                                                     \
      return error;                                                                                \
  }

namespace patch {
template <typename It>
using it_value_t = typename ::cuda::std::iterator_traits<It>::value_type;

template <typename It>
using it_difference_t = typename ::cuda::std::iterator_traits<It>::difference_type;
// from cuda 13.0

namespace detail {
template <cub::CacheLoadModifier LoadModifier, typename Iterator>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto try_make_cache_modified_iterator(Iterator it) {
  if constexpr (::cuda::std::contiguous_iterator<Iterator>) {
    return cub::CacheModifiedInputIterator<LoadModifier, it_value_t<Iterator>,
                                           it_difference_t<Iterator>>{
        THRUST_NS_QUALIFIER::raw_pointer_cast(&*it)};
  } else {
    return it;
  }
}
}  // namespace detail
template <cub::CacheLoadModifier LoadModifier, typename Iterator>
using try_make_cache_modified_iterator_t =
    decltype(detail::try_make_cache_modified_iterator<LoadModifier>(
        ::cuda::std::declval<Iterator>()));
}  // namespace patch

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {
namespace __set_operations {
template <bool UpperBound, class IntT, class Size, class It, class T, class Comp>
__device__ _CCCL_FORCEINLINE void binary_search_iteration_custom(It data, Size& begin, Size& end,
                                                                 T key, int shift, Comp comp) {
  IntT scale = (1 << shift) - 1;
  Size mid = (begin + scale * end) >> shift;

  T key2 = data[mid];
  bool pred = UpperBound ? !comp(key, key2) : comp(key2, key);
  if (pred) {
    begin = mid + 1;
  } else {
    end = mid;
  }
}

template <bool UpperBound, class Size, class T, class It, class Comp>
__device__ _CCCL_FORCEINLINE Size binary_search_custom(It data, Size count, T key, Comp comp) {
  Size begin = 0;
  Size end = count;
  while (begin < end) {
    binary_search_iteration_custom<UpperBound, int>(data, begin, end, key, 1, comp);
  }
  return begin;
}

template <bool UpperBound, class IntT, class Size, class T, class It, class Comp>
__device__ _CCCL_FORCEINLINE Size biased_binary_search_custom(It data, Size count, T key,
                                                              IntT levels, Comp comp) {
  Size begin = 0;
  Size end = count;

  if (levels >= 4 && begin < end) {
    binary_search_iteration_custom<UpperBound, IntT>(data, begin, end, key, 9, comp);
  }
  if (levels >= 3 && begin < end) {
    binary_search_iteration_custom<UpperBound, IntT>(data, begin, end, key, 7, comp);
  }
  if (levels >= 2 && begin < end) {
    binary_search_iteration_custom<UpperBound, IntT>(data, begin, end, key, 5, comp);
  }
  if (levels >= 1 && begin < end) {
    binary_search_iteration_custom<UpperBound, IntT>(data, begin, end, key, 4, comp);
  }

  while (begin < end) {
    binary_search_iteration_custom<UpperBound, IntT>(data, begin, end, key, 1, comp);
  }
  return begin;
}

template <bool UpperBound, class Size, class It1, class It2, class Comp>
__device__ _CCCL_FORCEINLINE Size merge_path_custom(It1 a, Size aCount, It2 b, Size bCount,
                                                    Size diag, Comp comp) {
  // using T = thrust::detail::it_value_t<It1>;
  using T = patch::it_value_t<It1>;

  Size begin = ::cuda::std::max<Size>(0, diag - bCount);
  Size end = ::cuda::std::min<Size>(diag, aCount);

  while (begin < end) {
    Size mid = (begin + end) >> 1;
    T aKey = a[mid];
    T bKey = b[diag - 1 - mid];
    bool pred = UpperBound ? comp(aKey, bKey) : !comp(bKey, aKey);
    if (pred) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin;
}

template <class It1, class It2, class Size, class Size2, class CompareOp>
__device__ _CCCL_FORCEINLINE ::cuda::std::pair<Size, Size> balanced_path_custom(
    It1 keys1, It2 keys2, Size num_keys1, Size num_keys2, Size diag, Size2 levels,
    CompareOp compare_op) {
  // using T = thrust::detail::it_value_t<It1>;
  using T = patch::it_value_t<It1>;

  Size index1 = merge_path_custom<false>(keys1, num_keys1, keys2, num_keys2, diag, compare_op);
  Size index2 = diag - index1;

  bool star = false;
  if (index2 < num_keys2) {
    T x = keys2[index2];

    // Search for the beginning of the duplicate run in both A and B.
    Size start1 = biased_binary_search_custom<false>(keys1, index1, x, levels, compare_op);
    Size start2 = biased_binary_search_custom<false>(keys2, index2, x, levels, compare_op);

    // The distance between x's merge path and its lower_bound is its rank.
    // We add up the a and b ranks and evenly distribute them to
    // get a stairstep path.
    Size run1 = index1 - start1;
    Size run2 = index2 - start2;
    Size total_run = run1 + run2;

    // Attempt to advance b and regress a.
    Size advance2 = max<Size>(total_run >> 1, total_run - run1);
    Size end2 = min<Size>(num_keys2, start2 + advance2 + 1);

    Size run_end2 =
        index2 + binary_search_custom<true>(keys2 + index2, end2 - index2, x, compare_op);
    run2 = run_end2 - start2;

    advance2 = min<Size>(advance2, run2);
    Size advance1 = total_run - advance2;

    bool round_up = (advance1 == advance2 + 1) && (advance2 < run2);
    if (round_up) {
      star = true;
    }

    index1 = start1 + advance1;
  }
  return ::cuda::std::make_pair(index1, (diag - index1) + star);
}  // func balanced_path_custom

template <int _BLOCK_THREADS, int _ITEMS_PER_THREAD = 1,
          cub::BlockLoadAlgorithm _LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier _LOAD_MODIFIER = cub::LOAD_LDG,
          cub::BlockScanAlgorithm _SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS>
struct PtxPolicyCustom {
  static constexpr int BLOCK_THREADS = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = _BLOCK_THREADS * _ITEMS_PER_THREAD - 1;

  static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
  static const cub::CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;
  static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};  // PtxPolicyCustom

template <class Arch, class T, class U>
struct TuningCustom {
  static constexpr int MAX_INPUT_BYTES = static_cast<int>(::cuda::std::max(sizeof(T), sizeof(U)));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(T) + sizeof(U);
  static constexpr int NOMINAL_4B_ITEMS_PER_THREAD =
      10;  // Reduced from 19 to 10 to check ShMem limits
  static constexpr int ITEMS_PER_THREAD = ::cuda::std::min(
      NOMINAL_4B_ITEMS_PER_THREAD,
      ::cuda::std::max(
          1, static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) + COMBINED_INPUT_BYTES - 1) /
                              COMBINED_INPUT_BYTES)));

  using type = PtxPolicyCustom<512, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE,
                               cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
};

template <class T, class U>
struct TuningCustom<core::sm52, T, U> {
  static constexpr int MAX_INPUT_BYTES = static_cast<int>(::cuda::std::max(sizeof(T), sizeof(U)));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(T);  // + sizeof(U)
  static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 15;
  static constexpr int ITEMS_PER_THREAD = ::cuda::std::min(
      NOMINAL_4B_ITEMS_PER_THREAD,
      ::cuda::std::max(
          1, static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) + COMBINED_INPUT_BYTES - 1) /
                              COMBINED_INPUT_BYTES)));

  using type = PtxPolicyCustom<256, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE,
                               cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
};  // tuning sm52

template <class T, class U>
struct TuningCustom<core::sm60, T, U> {
  static constexpr int MAX_INPUT_BYTES = static_cast<int>(::cuda::std::max(sizeof(T), sizeof(U)));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(T);  // + sizeof(U),
  static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 19;
  static constexpr int ITEMS_PER_THREAD = ::cuda::std::min(
      NOMINAL_4B_ITEMS_PER_THREAD,
      ::cuda::std::max(
          1, static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) + COMBINED_INPUT_BYTES - 1) /
                              COMBINED_INPUT_BYTES)));

  using type = PtxPolicyCustom<512, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE,
                               cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
};  // tuning sm60

// a helper metaprogram that returns type of a block loader
template <class PtxPlan, class It, class T = patch::it_value_t<It>>
using BlockLoad = cub::BlockLoad<T, PtxPlan::BLOCK_THREADS, PtxPlan::ITEMS_PER_THREAD,
                                 PtxPlan::LOAD_ALGORITHM, 1, 1>;

template <class KeysIt1, class KeysIt2, class ValuesIt1, class ValuesIt2, class KeysOutputIt,
          class ValuesOutputIt, class Size, class CompareOp, class SetOp, class HAS_VALUES>
struct SetOpAgentCustom {
  using key_type = patch::it_value_t<KeysIt1>;
  using value_type = patch::it_value_t<ValuesIt1>;
  using key2_type = patch::it_value_t<KeysIt2>;
  using value2_type = patch::it_value_t<ValuesIt2>;

  using ScanTileState = cub::ScanTileState<Size>;

  template <class Arch>
  struct PtxPlan : TuningCustom<Arch, key_type, value_type>::type {
    using tuning = TuningCustom<Arch, key_type, value_type>;

    using KeysLoadIt1 = patch::try_make_cache_modified_iterator_t<PtxPlan::LOAD_MODIFIER, KeysIt1>;
    using KeysLoadIt2 = patch::try_make_cache_modified_iterator_t<PtxPlan::LOAD_MODIFIER, KeysIt2>;
    using ValuesLoadIt1 =
        patch::try_make_cache_modified_iterator_t<PtxPlan::LOAD_MODIFIER, ValuesIt1>;
    using ValuesLoadIt2 =
        patch::try_make_cache_modified_iterator_t<PtxPlan::LOAD_MODIFIER, ValuesIt2>;

    using BlockLoadKeys1 = BlockLoad<PtxPlan, KeysLoadIt1>;
    using BlockLoadKeys2 = BlockLoad<PtxPlan, KeysLoadIt2>;
    using BlockLoadValues1 = BlockLoad<PtxPlan, ValuesLoadIt1>;
    using BlockLoadValues2 = BlockLoad<PtxPlan, ValuesLoadIt2>;

    using TilePrefixCallback = cub::TilePrefixCallbackOp<Size, ::cuda::std::plus<>, ScanTileState>;

    using BlockScan = cub::BlockScan<Size, PtxPlan::BLOCK_THREADS, PtxPlan::SCAN_ALGORITHM, 1, 1>;

    // gather required temporary storage in a union
    //
    union TempStorage {
      struct ScanStorage {
        typename BlockScan::TempStorage scan;
        typename TilePrefixCallback::TempStorage prefix;
      } scan_storage;

      struct alignas(16) LoadStorage {
        core::uninitialized_array<int, PtxPlan::BLOCK_THREADS> offset;

        // Keys and values need to coexist (not union!)
        alignas(16) core::uninitialized_array<value_type, PtxPlan::ITEMS_PER_TILE +
                                                              PtxPlan::BLOCK_THREADS> values_shared;

        alignas(16) core::uninitialized_array<key_type, PtxPlan::ITEMS_PER_TILE +
                                                            PtxPlan::BLOCK_THREADS> keys_shared;
      } load_storage;
    };
  };  // struct PtxPlan

  using ptx_plan = typename core::specialize_plan_msvc10_war<PtxPlan>::type::type;

  using KeysLoadIt1 = typename ptx_plan::KeysLoadIt1;
  using KeysLoadIt2 = typename ptx_plan::KeysLoadIt2;
  using ValuesLoadIt1 = typename ptx_plan::ValuesLoadIt1;
  using ValuesLoadIt2 = typename ptx_plan::ValuesLoadIt2;

  using BlockLoadKeys1 = typename ptx_plan::BlockLoadKeys1;
  using BlockLoadKeys2 = typename ptx_plan::BlockLoadKeys2;
  using BlockLoadValues1 = typename ptx_plan::BlockLoadValues1;
  using BlockLoadValues2 = typename ptx_plan::BlockLoadValues2;

  using TilePrefixCallback = typename ptx_plan::TilePrefixCallback;
  using BlockScan = typename ptx_plan::BlockScan;

  using TempStorage = typename ptx_plan::TempStorage;

  static constexpr int ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD;
  static constexpr int BLOCK_THREADS = ptx_plan::BLOCK_THREADS;

  struct impl {
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    TempStorage& storage;
    ScanTileState& tile_state;
    KeysLoadIt1 keys1_in;
    KeysLoadIt2 keys2_in;
    ValuesLoadIt1 values1_in;
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    ValuesLoadIt2 values2_in;
    // NEW: Global pointer for in-place updates
    using ValueType2 = typename thrust::iterator_value<ValuesIt2>::type;
    ValueType2* values2_raw;
    Size keys1_count;
    Size keys2_count;
    KeysOutputIt keys_out;
    ValuesOutputIt values_out;
    CompareOp compare_op;
    SetOp set_op;
    ::cuda::std::pair<Size, Size>* partitions;
    std::size_t* output_count;

    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    template <bool IS_FULL_TILE, class T, class It1, class It2>
    __device__ _CCCL_FORCEINLINE void gmem_to_reg(T (&output)[ITEMS_PER_THREAD], It1 input1,
                                                  It2 input2, int count1, int count2) {
      if (IS_FULL_TILE) {
// _CCCL_PRAGMA_UNROLL_FULL()
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ++ITEM) {
          int idx = BLOCK_THREADS * ITEM + threadIdx.x;
          output[ITEM] =
              (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
        }

        // last ITEM might be a conditional load even for full tiles
        // please check first before attempting to load.
        int ITEM = ITEMS_PER_THREAD - 1;
        int idx = BLOCK_THREADS * ITEM + threadIdx.x;
        if (idx < count1 + count2) {
          output[ITEM] =
              (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
        }
      } else {
// _CCCL_PRAGMA_UNROLL_FULL()
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
          int idx = BLOCK_THREADS * ITEM + threadIdx.x;
          if (idx < count1 + count2) {
            output[ITEM] =
                (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
          }
        }
      }
    }

    template <class T, class It>
    __device__ _CCCL_FORCEINLINE void reg_to_shared(It output, T (&input)[ITEMS_PER_THREAD]) {
// _CCCL_PRAGMA_UNROLL_FULL()
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int idx = BLOCK_THREADS * ITEM + threadIdx.x;
        output[idx] = input[ITEM];
      }
    }

    template <class OutputIt, class T, class SharedIt>
    void __device__ _CCCL_FORCEINLINE scatter(OutputIt output, T (&input)[ITEMS_PER_THREAD],
                                              SharedIt shared, int active_mask,
                                              Size thread_output_prefix, Size tile_output_prefix,
                                              int tile_output_count) {
      int local_scatter_idx = thread_output_prefix - tile_output_prefix;

// _CCCL_PRAGMA_UNROLL_FULL()
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if (active_mask & (1 << ITEM)) {
          shared[local_scatter_idx++] = input[ITEM];
        }
      }
      __syncthreads();

      for (int item = threadIdx.x; item < tile_output_count; item += BLOCK_THREADS) {
        output[tile_output_prefix + item] = shared[item];
      }
    }

    int __device__ _CCCL_FORCEINLINE serial_set_op_with_values(
        key_type* keys, value_type* values, int keys1_beg, int keys2_beg, int keys1_count,
        int keys2_count, ValueType2* values2_global, int values2_global_base_offset,
        key_type (&output_keys)[ITEMS_PER_THREAD], value_type (&output_values)[ITEMS_PER_THREAD],
        int (&indices)[ITEMS_PER_THREAD], CompareOp compare_op, SetOp set_op) {
      int active_mask =
          set_op(keys, values, keys1_beg, keys2_beg, keys1_count, keys2_count, values2_global,
                 values2_global_base_offset, output_keys, output_values, indices, compare_op);
      return active_mask;
    }

    template <bool IS_LAST_TILE>
    void __device__ _CCCL_FORCEINLINE consume_tile_with_match(Size tile_idx) {
      using core::uninitialized_array;

      ::cuda::std::pair<Size, Size> partition_beg = partitions[tile_idx + 0];
      ::cuda::std::pair<Size, Size> partition_end = partitions[tile_idx + 1];

      Size keys1_beg = partition_beg.first;
      Size keys1_end = partition_end.first;
      Size keys2_beg = partition_beg.second;
      Size keys2_end = partition_end.second;

      int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
      int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

      // ============ LOAD KEYS ============
      key_type keys_loc[ITEMS_PER_THREAD];
      gmem_to_reg<!IS_LAST_TILE>(keys_loc, keys1_in + keys1_beg, keys2_in + keys2_beg, num_keys1,
                                 num_keys2);
      reg_to_shared(&storage.load_storage.keys_shared[0], keys_loc);
      __syncthreads();

      // ============ LOAD VALUES (BEFORE serial op!) ============
      // Need separate shared memory for values during serial op
      // We'll use a trick: store values in registers, pass pointer to serial op
      value_type values_loc[ITEMS_PER_THREAD];
      if constexpr (HAS_VALUES::value) {
        gmem_to_reg<!IS_LAST_TILE>(values_loc, values1_in + keys1_beg, values2_in + keys2_beg,
                                   num_keys1, num_keys2);
        reg_to_shared(&storage.load_storage.values_shared[0], values_loc);
        __syncthreads();
      }

      // ============ PARTITION (same as before) ============
      int diag_loc = min<int>(ITEMS_PER_THREAD * threadIdx.x, num_keys1 + num_keys2);

      ::cuda::std::pair<int, int> partition_loc = balanced_path_custom(
          &storage.load_storage.keys_shared[0], &storage.load_storage.keys_shared[num_keys1],
          num_keys1, num_keys2, diag_loc, 4, compare_op);

      int keys1_beg_loc = partition_loc.first;
      int keys2_beg_loc = partition_loc.second;

      int value = threadIdx.x == 0 ? (num_keys1 << 16) | num_keys2
                                   : (partition_loc.first << 16) | partition_loc.second;
      int dst = threadIdx.x == 0 ? BLOCK_THREADS - 1 : threadIdx.x - 1;
      storage.load_storage.offset[dst] = value;
      __syncthreads();

      ::cuda::std::pair<int, int> partition1_loc =
          ::cuda::std::make_pair(storage.load_storage.offset[threadIdx.x] >> 16,
                                 storage.load_storage.offset[threadIdx.x] & 0xFFFF);

      int keys1_end_loc = partition1_loc.first;
      int keys2_end_loc = partition1_loc.second;
      int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
      int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

      // ============ SERIAL SET OP WITH VALUES ============
      int indices[ITEMS_PER_THREAD];
      value_type output_values[ITEMS_PER_THREAD];  // NEW: output values from serial op

      // Pass values_shared to serial op so it can access provenance
      // NEW: Pass global pointer and offset for in-place updates
      // keys2_beg is the global index where this tile's B-partition starts
      int active_mask = serial_set_op_with_values(
          &storage.load_storage.keys_shared[0], &storage.load_storage.values_shared[0],
          keys1_beg_loc, keys2_beg_loc + num_keys1, num_keys1_loc, num_keys2_loc, values2_raw,
          static_cast<int>(keys2_beg) -
              num_keys1,  // Pass adjusted global base offset (GlobalStart - num_keys1)
          keys_loc, output_values, indices, compare_op, set_op);

      __syncthreads();

      // ============ SCAN (same as before) ============
      Size tile_output_count = 0;
      Size thread_output_prefix = 0;
      Size tile_output_prefix = 0;
      Size thread_output_count =
          static_cast<Size>(::cuda::std::popcount(static_cast<unsigned>(active_mask)));

      if (tile_idx == 0) {
        BlockScan(storage.scan_storage.scan)
            .ExclusiveSum(thread_output_count, thread_output_prefix, tile_output_count);
        // For the first tile, tile_output_prefix should be 0
        tile_output_prefix = 0;
        if (threadIdx.x == 0 && !IS_LAST_TILE) {
          tile_state.SetInclusive(0, tile_output_count);
        }
      } else {
        TilePrefixCallback prefix_cb(tile_state, storage.scan_storage.prefix, ::cuda::std::plus<>{},
                                     tile_idx);
        BlockScan(storage.scan_storage.scan)
            .ExclusiveSum(thread_output_count, thread_output_prefix, prefix_cb);
        tile_output_count = prefix_cb.GetBlockAggregate();
        tile_output_prefix = prefix_cb.GetExclusivePrefix();
      }
      __syncthreads();

      // ============ SCATTER KEYS ============
      scatter(keys_out, keys_loc, &storage.load_storage.keys_shared[0], active_mask,
              thread_output_prefix, tile_output_prefix, tile_output_count);

      // ============ SCATTER VALUES (already computed!) ============
      if constexpr (HAS_VALUES::value) {
        __syncthreads();
        scatter(values_out, output_values, &storage.load_storage.values_shared[0], active_mask,
                thread_output_prefix, tile_output_prefix, tile_output_count);
      }

      if (IS_LAST_TILE && threadIdx.x == 0) {
        std::size_t final_count = tile_output_prefix + tile_output_count;
        if (output_count != nullptr) {
#ifdef USE_ROCm
          // On ROCm, use atomic store for better memory ordering
          __atomic_store_n(output_count, final_count, __ATOMIC_RELEASE);
#else
          // Use volatile write to ensure it's not optimized away
          volatile std::size_t* volatile_output_count =
              const_cast<volatile std::size_t*>(output_count);
          *volatile_output_count = final_count;
#endif
          // Memory fence to ensure write is visible
          __threadfence_system();
        }
      }
    }

    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    __device__ _CCCL_FORCEINLINE impl(TempStorage& storage_, ScanTileState& tile_state_,
                                      KeysIt1 keys1_, KeysIt2 keys2_, ValuesIt1 values1_,
                                      ValuesIt2 values2_, Size keys1_count_, Size keys2_count_,
                                      KeysOutputIt keys_out_, ValuesOutputIt values_out_,
                                      CompareOp compare_op_, SetOp set_op_,
                                      ::cuda::std::pair<Size, Size>* partitions_,
                                      std::size_t* output_count_)
        : storage(storage_), tile_state(tile_state_),
          keys1_in(
              patch::detail::try_make_cache_modified_iterator<ptx_plan::LOAD_MODIFIER>(keys1_)),
          keys2_in(
              patch::detail::try_make_cache_modified_iterator<ptx_plan::LOAD_MODIFIER>(keys2_)),
          values1_in(
              patch::detail::try_make_cache_modified_iterator<ptx_plan::LOAD_MODIFIER>(values1_)),
          values2_in(
              patch::detail::try_make_cache_modified_iterator<ptx_plan::LOAD_MODIFIER>(values2_)),
          // NEW: Store raw global iterator - ONLY if not bool (to avoid bit-packed iterator issues)
          values2_raw(nullptr), keys1_count(keys1_count_), keys2_count(keys2_count_),
          keys_out(keys_out_), values_out(values_out_), compare_op(compare_op_), set_op(set_op_),
          partitions(partitions_), output_count(output_count_) {

      // Initialize values2_raw only if not bool
      if constexpr (!::cuda::std::is_same<ValueType2, bool>::value) {
        values2_raw = thrust::raw_pointer_cast(&*values2_);
      }

      int tile_idx = blockIdx.x;
      int num_tiles = gridDim.x;

      if (tile_idx < num_tiles - 1) {
        consume_tile_with_match<false>(tile_idx);
      } else {
        consume_tile_with_match<true>(tile_idx);
      }
    }
  };  // struct impl

  //---------------------------------------------------------------------
  // Agent entry point
  //---------------------------------------------------------------------

  THRUST_AGENT_ENTRY(KeysIt1 keys1, KeysIt2 keys2, ValuesIt1 values1, ValuesIt2 values2,
                     Size keys1_count, Size keys2_count, KeysOutputIt keys_output,
                     ValuesOutputIt values_output, CompareOp compare_op, SetOp set_op,
                     ::cuda::std::pair<Size, Size>* partitions, std::size_t* output_count,
                     ScanTileState tile_state, char* shmem) {
    TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

    impl(storage, tile_state, keys1, keys2, values1, values2, keys1_count, keys2_count, keys_output,
         values_output, compare_op, set_op, partitions, output_count);
  }
};  // struct SetOpAgentCustom

template <class KeysIt1, class KeysIt2, class Size, class CompareOp>
struct PartitionAgentCustom {
  template <class Arch>
  struct PtxPlan : PtxPolicyCustom<256> {};

  using ptx_plan = core::specialize_plan<PtxPlan>;

  //---------------------------------------------------------------------
  // Agent entry point
  //---------------------------------------------------------------------

  THRUST_AGENT_ENTRY(KeysIt1 keys1, KeysIt2 keys2, Size keys1_count, Size keys2_count,
                     Size num_partitions, ::cuda::std::pair<Size, Size>* partitions,
                     CompareOp compare_op, int items_per_tile, char* /*shmem*/) {
    Size partition_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (partition_idx < num_partitions) {
      Size partition_at = min<Size>(partition_idx * items_per_tile, keys1_count + keys2_count);
      ::cuda::std::pair<Size, Size> diag = balanced_path_custom(
          keys1, keys2, keys1_count, keys2_count, partition_at, 4ll, compare_op);
      partitions[partition_idx] = diag;
    }
  }
};  // struct PartitionAgentCustom

template <class ScanTileState, class Size>
struct InitAgentCustom {
  template <class Arch>
  struct PtxPlan : PtxPolicyCustom<128> {};

  using ptx_plan = core::specialize_plan<PtxPlan>;

  //---------------------------------------------------------------------
  // Agent entry point
  //---------------------------------------------------------------------

  THRUST_AGENT_ENTRY(ScanTileState tile_state, Size num_tiles, char* /*shmem*/) {
    tile_state.InitializeStatus(num_tiles);
  }
};  // struct InitAgentCustom

// serial_set_difference_with_match_custom
// For matches: calls match_op(key, value1, value2) -> (emit?, output_value)
template <typename MatchOp>
struct serial_set_difference_with_match_custom {
  MatchOp match_op;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE serial_set_difference_with_match_custom(MatchOp op)
      : match_op(op) {}

  template <class KeyT, class ValueT, class ValuesGlobalIt, class CompareOp, int ITEMS_PER_THREAD>
  int __device__ _CCCL_FORCEINLINE operator()(KeyT* keys,
                                              ValueT* values,  // values in shared memory
                                              int keys1_beg, int keys2_beg, int keys1_count,
                                              int keys2_count, ValuesGlobalIt values2_global,
                                              int values2_global_base_offset,  // NEW: Global access
                                              KeyT (&output_keys)[ITEMS_PER_THREAD],
                                              ValueT (&output_values)[ITEMS_PER_THREAD],
                                              int (&indices)[ITEMS_PER_THREAD],
                                              CompareOp compare_op) {
    int active_mask = 0;

    int aBegin = keys1_beg;
    int bBegin = keys2_beg;
    int aEnd = keys1_beg + keys1_count;
    int bEnd = keys2_beg + keys2_count;
    int end = aEnd + bEnd;

    KeyT aKey = keys[aBegin];
    KeyT bKey = keys[bBegin];

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      bool pB = aBegin >= aEnd;
      bool pA = !pB && bBegin >= bEnd;

      if (!pA && !pB) {
        pA = compare_op(aKey, bKey);
        pB = !pA && compare_op(bKey, aKey);
      }

      output_keys[i] = aKey;
      indices[i] = aBegin;

      if (aBegin + bBegin < end) {
        if (pA) {
          // A < B: DIFFERENCE (INSERT) - emit with original value
          active_mask |= 1 << i;
          output_values[i] = values[aBegin];  // value from A
        } else if (!pA && !pB) {
          // A == B: MATCH - call match_op to decide
          ValueT val_a = values[aBegin];
          ValueT val_b = values[bBegin];  // bBegin already offset by num_keys1

          auto [should_emit, combined_value] = match_op(aKey, val_a, val_b);

          // NEW: In-place update of Full array side effect
          // keys2_beg was the start of B in shared memory (already offset by num_keys1)
          // bBegin is the current cursor in shared memory
          // The local offset in B partition is (bBegin - keys2_beg)
          // So global index is values2_global_base_offset + (bBegin - keys2_beg)

          // Only perform update if not bool (idempotent addition for bool makes this redundant,
          // and avoids bit-packed iterator issues)
          if constexpr (!::cuda::std::is_same<ValueT, bool>::value) {
            values2_global[values2_global_base_offset + bBegin] = combined_value;
          }

          if (should_emit) {
            active_mask |= 1 << i;
            output_values[i] = combined_value;
          }
        }
        // else: B < A, don't emit
      }

      if (!pB) {
        aKey = keys[++aBegin];
      }
      if (!pA) {
        bKey = keys[++bBegin];
      }
    }
    return active_mask;
  }
};

template <class HAS_VALUES, class KeysIt1, class KeysIt2, class ValuesIt1, class ValuesIt2,
          class Size, class KeysOutputIt, class ValuesOutputIt, class CompareOp, class SetOp>
cudaError_t THRUST_RUNTIME_FUNCTION
doit_step_custom(void* d_temp_storage, size_t& temp_storage_size, KeysIt1 keys1, KeysIt2 keys2,
                 ValuesIt1 values1, ValuesIt2 values2, Size num_keys1, Size num_keys2,
                 KeysOutputIt keys_output, ValuesOutputIt values_output, std::size_t* output_count,
                 CompareOp compare_op, SetOp set_op, cudaStream_t stream) {
  Size keys_total = num_keys1 + num_keys2;
  if (keys_total == 0) {
    return cudaErrorNotSupported;
  }

  if (d_temp_storage == nullptr && output_count == nullptr) {
    // First call - just querying temp storage size
  } else if (output_count != nullptr) {
    // Second call - actually executing
  }

  cudaError_t status = cudaSuccess;

  using core::AgentLauncher;
  using core::AgentPlan;

  using set_op_agent =
      AgentLauncher<SetOpAgentCustom<KeysIt1, KeysIt2, ValuesIt1, ValuesIt2, KeysOutputIt,
                                     ValuesOutputIt, Size, CompareOp, SetOp, HAS_VALUES>>;

  using partition_agent = AgentLauncher<PartitionAgentCustom<KeysIt1, KeysIt2, Size, CompareOp>>;

  using ScanTileState = typename set_op_agent::ScanTileState;
  using init_agent = AgentLauncher<InitAgentCustom<ScanTileState, Size>>;

  AgentPlan set_op_plan = set_op_agent::get_plan(stream);
  AgentPlan init_plan = init_agent::get_plan();
  AgentPlan partition_plan = partition_agent::get_plan();

  int tile_size = set_op_plan.items_per_tile;
  Size num_tiles = (keys_total + tile_size - 1) / tile_size;

  size_t tile_agent_storage;
  status = ScanTileState::AllocationSize(static_cast<int>(num_tiles), tile_agent_storage);
  _CUDA_CUB_RET_IF_FAIL_CUSTOM(status);

  size_t vshmem_storage = core::vshmem_size(set_op_plan.shared_memory_size, num_tiles);
  size_t partition_agent_storage = (num_tiles + 1) * sizeof(Size) * 2;

  void* allocations[3] = {nullptr, nullptr, nullptr};
  size_t allocation_sizes[3] = {tile_agent_storage, partition_agent_storage, vshmem_storage};

  status = core::alias_storage(d_temp_storage, temp_storage_size, allocations, allocation_sizes);
  _CUDA_CUB_RET_IF_FAIL_CUSTOM(status);

  if (d_temp_storage == nullptr) {
    return status;
  }

  ScanTileState tile_state;
  status = tile_state.Init(static_cast<int>(num_tiles), allocations[0], allocation_sizes[0]);
  _CUDA_CUB_RET_IF_FAIL_CUSTOM(status);

  ::cuda::std::pair<Size, Size>* partitions = (::cuda::std::pair<Size, Size>*)allocations[1];
  char* vshmem_ptr = vshmem_storage > 0 ? (char*)allocations[2] : nullptr;

  init_agent ia(init_plan, num_tiles, stream, "set_op::init_agent");
  ia.launch(tile_state, num_tiles);
  _CUDA_CUB_RET_IF_FAIL_CUSTOM(cudaPeekAtLastError());

  partition_agent pa(partition_plan, num_tiles + 1, stream, "set_op::partition agent");
  pa.launch(keys1, keys2, num_keys1, num_keys2, num_tiles + 1, partitions, compare_op, tile_size);
  _CUDA_CUB_RET_IF_FAIL_CUSTOM(cudaPeekAtLastError());

  set_op_agent sa(set_op_plan, keys_total, stream, vshmem_ptr, "set_op::set_op_agent");

  sa.launch(keys1, keys2, values1, values2, num_keys1, num_keys2, keys_output, values_output,
            compare_op, set_op, partitions, output_count, tile_state);

  // Check for launch errors immediately
  cudaError_t launch_err = GPU_GET_LAST_ERROR();
  if (launch_err != GPU_SUCCESS) {
    std::cerr << "ERROR: Kernel launch failed: " << GPU_GET_ERROR_STRING(launch_err) << std::endl;
    return launch_err;
  }

  return status;
}

template <typename HAS_VALUES, typename Derived, typename KeysIt1, typename KeysIt2,
          typename ValuesIt1, typename ValuesIt2, typename KeysOutputIt, typename ValuesOutputIt,
          typename CompareOp, typename SetOp>
THRUST_RUNTIME_FUNCTION ::cuda::std::pair<KeysOutputIt, ValuesOutputIt> set_operations_custom(
    Derived policy, KeysIt1 keys1_first, KeysIt1 keys1_last, KeysIt2 keys2_first,
    KeysIt2 keys2_last, ValuesIt1 values1_first, ValuesIt2 values2_first, KeysOutputIt keys_output,
    ValuesOutputIt values_output, CompareOp compare_op, SetOp set_op) {
  using size_type = patch::it_difference_t<KeysIt1>;

  size_type num_keys1 = static_cast<size_type>(::cuda::std::distance(keys1_first, keys1_last));
  size_type num_keys2 = static_cast<size_type>(::cuda::std::distance(keys2_first, keys2_last));

  if (num_keys1 + num_keys2 == 0) {
    return ::cuda::std::make_pair(keys_output, values_output);
  }

  size_t temp_storage_bytes = 0;
#ifdef USE_ROCm
  // On HIP, use default stream (nullptr) for thrust::hip_rocprim::par_t
  // If policy is an execution_policy with a stream, extract it; otherwise use default
  cudaStream_t stream = nullptr;  // Default stream
  if constexpr (!std::is_same_v<Derived, thrust::hip_rocprim::par_t>) {
    // Try to extract stream if policy is not the default par_t
    stream = thrust::hip_rocprim::stream(policy);
  }
#else
  cudaStream_t stream = cuda_cub::stream(policy);
#endif

  cudaError_t status;
  THRUST_DOUBLE_INDEX_TYPE_DISPATCH(
      status, doit_step_custom<HAS_VALUES>, num_keys1, num_keys2,
      (nullptr, temp_storage_bytes, keys1_first, keys2_first, values1_first, values2_first,
       num_keys1_fixed, num_keys2_fixed, keys_output, values_output,
       static_cast<std::size_t*>(nullptr), compare_op, set_op, stream));
#ifdef USE_ROCm
  thrust::hip_rocprim::throw_on_error(status, "set_operations failed on 1st step");
#else
  cuda_cub::throw_on_error(status, "set_operations failed on 1st step");
#endif

  size_t allocation_sizes[2] = {sizeof(std::size_t) * 2,
                                temp_storage_bytes};  // *2 for output_count + marker
  void* allocations[2] = {nullptr, nullptr};

  size_t storage_size = 0;

  status = core::alias_storage(nullptr, storage_size, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "set_operations failed on 1st alias_storage");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status = core::alias_storage(ptr, storage_size, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "set_operations failed on 2nd alias_storage");

  std::size_t* d_output_count =
      thrust::detail::aligned_reinterpret_cast<std::size_t*>(allocations[0]);

  // Initialize output count to 0 on device (important for ROCm)
  // Also initialize a marker value to verify kernel execution
  std::size_t zero = 0;
  std::size_t marker = 0xDEADBEEF;
  status = GPU_MEMCPY(d_output_count, &zero, sizeof(std::size_t), GPU_HOST_TO_DEVICE);
  cuda_cub::throw_on_error(status, "set_operations failed to initialize output count");
  // Write marker to the next word to verify kernel execution
  std::size_t* d_marker = d_output_count + 1;
  status = GPU_MEMCPY(d_marker, &marker, sizeof(std::size_t), GPU_HOST_TO_DEVICE);
  cuda_cub::throw_on_error(status, "set_operations failed to initialize marker");

  THRUST_DOUBLE_INDEX_TYPE_DISPATCH(
      status, doit_step_custom<HAS_VALUES>, num_keys1, num_keys2,
      (allocations[1], temp_storage_bytes, keys1_first, keys2_first, values1_first, values2_first,
       num_keys1_fixed, num_keys2_fixed, keys_output, values_output, d_output_count, compare_op,
       set_op, stream));
  cuda_cub::throw_on_error(status, "set_operations failed on 2nd step");

#ifdef USE_ROCm
  // On ROCm, always use manual memcpy to get output count (more reliable than get_value)
  // Extract stream from policy if available, otherwise use default stream
  cudaStream_t stream_to_sync = nullptr;
  if constexpr (std::is_same_v<Derived, thrust::hip_rocprim::par_t>) {
    stream_to_sync = nullptr;  // Default stream
  } else {
    // Try to extract stream from policy
    stream_to_sync = thrust::hip_rocprim::stream(policy);
  }
  status = GPU_STREAM_SYNCHRONIZE(stream_to_sync);
  thrust::hip_rocprim::throw_on_error(status, "set_operations failed to synchronize");

  // Check for kernel errors after synchronization
  GPU_ERROR_T sync_err = GPU_GET_LAST_ERROR();
  if (sync_err != GPU_SUCCESS) {
    std::cerr << "WARNING: GPU error after synchronization: " << GPU_GET_ERROR_STRING(sync_err)
              << std::endl;
  }

  // Copy output count from device
  std::size_t h_output_count = 0;
  status = GPU_MEMCPY(&h_output_count, d_output_count, sizeof(std::size_t), GPU_DEVICE_TO_HOST);
  thrust::hip_rocprim::throw_on_error(status, "set_operations failed to get output count");
  return ::cuda::std::make_pair(keys_output + h_output_count, values_output + h_output_count);
#else
  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "set_operations failed to synchronize");
  std::size_t output_count = cuda_cub::get_value(policy, d_output_count);
  return ::cuda::std::make_pair(keys_output + output_count, values_output + output_count);
#endif
}
}  // namespace __set_operations

template <class Derived, class KeysIt1, class KeysIt2, class ValuesIt1, class ValuesIt2,
          class KeysOutputIt, class ValuesOutputIt, class CompareOp, class MatchOp>
::cuda::std::pair<KeysOutputIt, ValuesOutputIt> _CCCL_HOST_DEVICE
set_difference_by_key_with_match_custom(
#ifdef USE_ROCm
    // On HIP, accept any execution policy (Thrust will dispatch to HIP backend)
    // Use decltype to extract the underlying policy type from thrust::device
    const Derived& policy,
#else
    // On CUDA, use cuda_cub::execution_policy
    const execution_policy<Derived>& policy,
#endif
    KeysIt1 keys1_first, KeysIt1 keys1_last, KeysIt2 keys2_first, KeysIt2 keys2_last,
    ValuesIt1 values1_first,  // new provenance
    ValuesIt2 values2_first,  // full provenance
    KeysOutputIt keys_result, ValuesOutputIt values_result, CompareOp compare_op,
    MatchOp match_op)  // NEW: match callback
{
#if !defined(__CUDA_ARCH__)
  nvtxRangePushA("set_difference_by_key_with_match_custom");
#endif
  auto set_op = __set_operations::serial_set_difference_with_match_custom<MatchOp>(match_op);

#ifdef USE_ROCm
  // On HIP, use thrust::hip_rocprim::par directly
  // This is the default parallel execution policy for HIP
  // The temporary_array and other operations will work with this type
  auto result = __set_operations::set_operations_custom<thrust::detail::true_type>(
      thrust::hip_rocprim::par, keys1_first, keys1_last, keys2_first, keys2_last, values1_first,
      values2_first, keys_result, values_result, compare_op, set_op);
#else
  auto result = __set_operations::set_operations_custom<thrust::detail::true_type>(
      static_cast<const Derived&>(policy), keys1_first, keys1_last, keys2_first, keys2_last,
      values1_first, values2_first, keys_result, values_result, compare_op, set_op);
#endif
#if !defined(__CUDA_ARCH__)
  nvtxRangePop();
#endif
  return result;
}

}  // namespace cuda_cub

#ifdef USE_ROCm
// On ROCm, add namespace alias for compatibility
// This allows code to use thrust::cuda_cub::set_difference_by_key_with_match_custom
// with thrust::device policy, which will be automatically converted to
// hip_rocprim::execution_policy
namespace hip_rocprim {
using cuda_cub::set_difference_by_key_with_match_custom;
}  // namespace hip_rocprim
#endif

THRUST_NAMESPACE_END