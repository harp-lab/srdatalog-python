#pragma once

/**
 * @file launcher.h
 * @brief Host-side launcher for GPU pipeline execution
 */

#include "../gpu_api.h"  // GPU API abstraction
#include "./root_executor.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace SRDatalog::GPU {

template <typename Pipeline, Semiring SR, typename ValueType, typename HandleType>
struct GPUPipelineLauncher {
  using Executor = GPURootExecutor<Pipeline, SR, ValueType, HandleType>;

  static constexpr int BlockSize = 256;
// Use platform-aware default group size (32 for CUDA, 64 for ROCm)
#ifdef GPU_DEFAULT_GROUP_SIZE
  static constexpr int GroupSize = GPU_DEFAULT_GROUP_SIZE;
#else
  static constexpr int GroupSize = 32;  // Fallback for CPU-only builds
#endif
  static constexpr std::size_t NumHandles = Executor::NumHandles;
  static constexpr std::size_t NumFullHandles = Executor::NumFullHandles;

  /**
   * @brief Execute pipeline with two-phase approach
   *
   * @param root_handles Array of root handles for indexed relations
   * @param full_handles Array of root handles for full relations (dedup)
   * @param root_views Array of root views for indexed relations
   * @param full_views Array of full views for full relations (dedup)
   * @param root_unique_values Unique root values for partitioning (from first source index)
   * @param num_unique_root_keys Number of unique root keys
   * @param num_root_keys Total number of root keys to process
   * @param output_data Pre-allocated output buffer (column-major)
   * @param output_prov Pre-allocated provenance buffer
   * @param output_stride Stride for column-major layout
   * @param stream CUDA stream
   * @return Total number of output tuples
   */
  static std::size_t execute(const HandleType* root_handles, const HandleType* full_handles,
                             const typename Executor::ViewType* root_views,
                             const typename Executor::ViewType* full_views,
                             const ValueType* root_unique_values, uint32_t num_unique_root_keys,
                             uint32_t num_root_keys, ValueType* output_data,
                             semiring_value_t<SR>* output_prov, std::size_t output_stride,
                             cudaStream_t stream = 0) {

    // Calculate grid dimensions based on unique root keys for better partitioning
    uint32_t num_threads = ((num_unique_root_keys + GroupSize - 1) / GroupSize) * GroupSize;
    num_threads = std::max(num_threads, static_cast<uint32_t>(BlockSize));
    uint32_t num_blocks = (num_threads + BlockSize - 1) / BlockSize;

    // Ensure minimum grid size to utilize GPU: at least 2 blocks per SM
    // This prevents severe underutilization in early iterations when num_unique_root_keys is small
    int num_sms = 0;
    GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);
    uint32_t min_blocks = static_cast<uint32_t>(num_sms) * 2;
    num_blocks = std::max(num_blocks, min_blocks);
    num_threads = num_blocks * BlockSize;

    // Allocate histogram
    DeviceArray<uint32_t> thread_counts(num_threads + 1);

    // Phase 1: Count
    Executor::template kernel_count<BlockSize, GroupSize><<<num_blocks, BlockSize, 0, stream>>>(
        root_handles, full_handles, root_views, full_views, root_unique_values,
        num_unique_root_keys, num_root_keys, thread_counts.data());

    // Prefix sum
    // Use thrust::device policy to avoid potential issues with stream-based policy/invalid device
    // ordinal
    thrust::exclusive_scan(rmm::exec_policy{}, thread_counts.begin(), thread_counts.end(),
                           thread_counts.begin());

    // Get total count
    uint32_t total_count;
    GPU_MEMCPY_ASYNC(&total_count, thread_counts.data() + num_threads, sizeof(uint32_t),
                     GPU_DEVICE_TO_HOST, stream);
    GPU_STREAM_SYNCHRONIZE(stream);

    if (total_count == 0) {
      return 0;
    }

    // Phase 2: Materialize
    Executor::template kernel_materialize<BlockSize, GroupSize>
        <<<num_blocks, BlockSize, 0, stream>>>(root_handles, full_handles, root_views, full_views,
                                               root_unique_values, num_unique_root_keys,
                                               num_root_keys, thread_counts.data(), output_data,
                                               output_prov, output_stride);

    return total_count;
  }
};

}  // namespace SRDatalog::GPU
