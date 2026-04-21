/**
 * @file jit_executor.h
 * @brief JIT kernel execution wrapper for GPU pipelines
 *
 * Provides execute_jit() function that wraps JIT kernel execution
 * with proper workload partitioning matching TMP backend.
 *
 * Key design: Work is partitioned across warps via warp_id/num_warps,
 * and root_unique_values provides the work items to distribute.
 */

#pragma once

#include "../../device_sorted_array_index.h"
#include "../mir.h"
#include "../output_context.h"
#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace cg = cooperative_groups;

namespace SRDatalog::GPU::JIT {

/**
 * @brief Execute a JIT kernel with automatic handle/view setup from IndexSpecs
 *
 * This is the main entry point for running JIT kernels. It:
 * 1. Extracts handles/views from the database based on IndexSpecs
 * 2. Gets root_unique_values for work partitioning
 * 3. Copies them to device memory
 * 4. Runs the JIT kernel with two-phase execution (count + materialize)
 * 5. Returns the output in the NEW tuple space
 *
 * @tparam JitKernel The JIT-generated kernel functor
 * @tparam SourceSpecs Tuple of IndexSpec types for source relations
 * @tparam DestSpecs Tuple of IndexSpec types for destination relations
 * @tparam AllSchemas Tuple of all schema types used
 * @tparam DB Database type
 * @tparam SR Semiring type
 */
template <typename JitKernel, typename SourceSpecs, typename DestSpecs, typename AllSchemas,
          Semiring SR>
struct ExecuteJit {
  // Derive ValueType from first source schema's intern type (same as TMP executor)
  using FirstSpec = std::tuple_element_t<0, SourceSpecs>;
  using FirstSchema = typename FirstSpec::schema_type;
  using ValueType = typename FirstSchema::intern_value_type;

  static constexpr auto Layout = StorageLayout::SoA;

  // Derive OutputArity from first DestSpec schema
  template <typename T>
  struct GetArity;
  template <typename First, typename... Rest>
  struct GetArity<std::tuple<First, Rest...>> {
    static constexpr std::size_t value = First::schema_type::Arity;
  };
  template <>
  struct GetArity<std::tuple<>> {
    static constexpr std::size_t value = 0;
  };
  static constexpr std::size_t OutputArity = GetArity<DestSpecs>::value;

  // Extract destination schema
  template <typename T>
  struct GetDestSchema;
  template <typename First, typename... Rest>
  struct GetDestSchema<std::tuple<First, Rest...>> {
    using type = typename First::schema_type;
  };
  using DestSchema = typename GetDestSchema<DestSpecs>::type;

  static constexpr int kBlockSize = 256;
#ifdef GPU_DEFAULT_GROUP_SIZE
  static constexpr int kGroupSize = GPU_DEFAULT_GROUP_SIZE;
#else
  static constexpr int kGroupSize = 32;
#endif

  /**
   * @brief Count kernel with proper workload partitioning
   *
   * Each warp handles different work items based on warp_id and num_warps.
   * This matches TMP's `execute_root` pattern.
   *
   * Note: Handles are NOT passed as parameters - they are computed from views
   * inside the generated kernel code (handle = cursor, computable from view metadata).
   */
  template <typename ViewType>
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_count(const ViewType* __restrict__ views,
                   const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
                   uint32_t num_root_keys, uint32_t* __restrict__ thread_counts) {

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;

    using OutputCtx = OutputContext<ValueType, SR, true, Layout, OutputArity>;
    OutputCtx output_ctx{nullptr, nullptr, 0, 0};

    // Call JIT kernel with workload partitioning info
    // Note: handles computed from views inside kernel
    JitKernel{}(tile, views, root_unique_values, num_unique_root_keys, num_root_keys, warp_id,
                num_warps, output_ctx);

    thread_counts[thread_id] = output_ctx.count();
  }

  /**
   * @brief Materialize kernel with proper workload partitioning
   */
  template <typename ViewType>
  static __global__ void __launch_bounds__(kBlockSize)
      kernel_materialize(const ViewType* __restrict__ views,
                         const ValueType* __restrict__ root_unique_values,
                         uint32_t num_unique_root_keys, uint32_t num_root_keys,
                         const uint32_t* __restrict__ thread_offsets,
                         ValueType* __restrict__ output_data,
                         semiring_value_t<SR>* __restrict__ output_prov,
                         std::size_t output_stride) {

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t write_base = thread_offsets[thread_id];

    using OutputCtx = OutputContext<ValueType, SR, false, Layout, OutputArity>;
    OutputCtx output_ctx{output_data, output_prov, output_stride, write_base};

    // Call JIT kernel with workload partitioning info
    // Note: handles computed from views inside kernel
    JitKernel{}(tile, views, root_unique_values, num_unique_root_keys, num_root_keys, warp_id,
                num_warps, output_ctx);
  }

  template <typename DB>
  static void run(DB& db, uint32_t iteration, const char* rule_name) {
    nvtxRangePushA(rule_name);

    // Get index type from DB's first source relation
    using FirstSpec = std::tuple_element_t<0, SourceSpecs>;
    using FirstSchema = typename FirstSpec::schema_type;
    constexpr std::size_t FirstVer = FirstSpec::kVersion;
    using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, FirstVer>(db))>;
    using IndexType = typename RelType::IndexTypeInst;  // Use IndexTypeInst, not IndexType!
    using ViewType = typename IndexType::NodeView;

    // Setup views from SourceSpecs (handles computed from views in kernel)
    std::vector<ViewType> views_vec;

    auto fill_source_views = [&]<typename... Specs>(std::tuple<Specs...>) {
      (..., [&](auto SpecType) {
        using Spec = decltype(SpecType);
        using Schema = typename Spec::schema_type;
        using ColSeq = typename Spec::column_indexes_type;
        constexpr std::size_t ver = Spec::kVersion;

        auto& rel = [&]() -> auto& {
          if constexpr (ver == DELTA_VER) {
            if (iteration == 0) {
              return get_relation_by_schema<Schema, FULL_VER>(db);
            }
          }
          return get_relation_by_schema<Schema, ver>(db);
        }();

        auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
          return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
        }(ColSeq{});

        auto& idx = rel.ensure_index(runtime_spec, false);
        views_vec.push_back(idx.view());
      }(Specs{}));
    };
    fill_source_views(SourceSpecs{});

    // Get root unique values for workload partitioning (from first source)
    using FirstColSeq = typename FirstSpec::column_indexes_type;
    auto first_runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
      return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
    }(FirstColSeq{});

    auto& first_rel = [&]() -> auto& {
      if constexpr (FirstVer == DELTA_VER) {
        if (iteration == 0) {
          return get_relation_by_schema<FirstSchema, FULL_VER>(db);
        }
      }
      return get_relation_by_schema<FirstSchema, FirstVer>(db);
    }();

    auto& first_idx = first_rel.get_index(first_runtime_spec);
    uint32_t num_root_keys = first_idx.root().degree();
    uint32_t num_unique_root_keys = static_cast<uint32_t>(first_idx.num_unique_root_values());

    const ValueType* root_unique_values_ptr =
        (num_unique_root_keys > 0) ? first_idx.root_unique_values().data() : nullptr;

    // Copy only views to device (handles computed from views in kernel)
    DeviceArray<ViewType> d_views(views_vec.size());
    GPU_MEMCPY(d_views.data(), views_vec.data(), views_vec.size() * sizeof(ViewType),
               GPU_HOST_TO_DEVICE);
    GPU_DEVICE_SYNCHRONIZE();

    // Launch configuration - based on unique root keys for better partitioning
    int num_sms = 0;
    GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);

    // Match TMP's grid sizing strategy
    uint32_t num_threads = ((num_unique_root_keys + kGroupSize - 1) / kGroupSize) * kGroupSize;
    num_threads = std::max(num_threads, static_cast<uint32_t>(kBlockSize));
    uint32_t num_blocks = (num_threads + kBlockSize - 1) / kBlockSize;

    // Ensure minimum grid size: at least 4 blocks per SM for better latency hiding
    uint32_t min_blocks = static_cast<uint32_t>(num_sms) * 4;
    num_blocks = std::max(num_blocks, min_blocks);
    num_threads = num_blocks * kBlockSize;

    std::cerr << "[JIT] " << rule_name << ": Launching kernel_count with " << num_blocks
              << " blocks, " << kBlockSize << " threads/block, " << num_unique_root_keys
              << " unique root keys" << std::endl;

    // Count phase
    DeviceArray<uint32_t> thread_counts(num_threads + 1);
    kernel_count<ViewType><<<num_blocks, kBlockSize>>>(d_views.data(), root_unique_values_ptr,
                                                       num_unique_root_keys, num_root_keys,
                                                       thread_counts.data());
    std::cerr << "[JIT] " << rule_name << ": kernel_count launched, syncing..." << std::endl;
    GPU_DEVICE_SYNCHRONIZE();
    std::cerr << "[JIT] " << rule_name << ": kernel_count sync done" << std::endl;

    GPU_ERROR_T err = GPU_GET_LAST_ERROR();
    if (err != GPU_SUCCESS) {
      std::cerr << "JIT kernel_count failed: " << GPU_GET_ERROR_STRING(err) << std::endl;
    }

    // // DEBUG: Dump thread count histogram
    // {
    //   std::vector<uint32_t> h_counts(num_threads);
    //   GPU_MEMCPY(h_counts.data(), thread_counts.data(), num_threads * sizeof(uint32_t),
    //              GPU_DEVICE_TO_HOST);

    //   // Calculate histogram buckets: 0, 1-10, 11-100, 101-1000, 1001-10000, 10001+
    //   std::array<uint32_t, 6> histogram = {0, 0, 0, 0, 0, 0};
    //   uint32_t total = 0, max_count = 0, nonzero = 0;
    //   uint32_t max_thread = 0;

    //   for (uint32_t i = 0; i < num_threads; ++i) {
    //     uint32_t c = h_counts[i];
    //     total += c;
    //     if (c > max_count) {
    //       max_count = c;
    //       max_thread = i;
    //     }
    //     if (c > 0)
    //       nonzero++;

    //     if (c == 0)
    //       histogram[0]++;
    //     else if (c <= 10)
    //       histogram[1]++;
    //     else if (c <= 100)
    //       histogram[2]++;
    //     else if (c <= 1000)
    //       histogram[3]++;
    //     else if (c <= 10000)
    //       histogram[4]++;
    //     else
    //       histogram[5]++;
    //   }

    //   std::cerr << "\n=== THREAD COUNT HISTOGRAM (Load Balance) ===" << std::endl;
    //   std::cerr << "  Total threads: " << num_threads << ", Non-zero: " << nonzero << " ("
    //             << (100.0 * nonzero / num_threads) << "%)" << std::endl;
    //   std::cerr << "  Total tuples: " << total << std::endl;
    //   std::cerr << "  Max count: " << max_count << " (thread " << max_thread << ")" << std::endl;
    //   std::cerr << "  Bucket [0]:       " << histogram[0] << " threads" << std::endl;
    //   std::cerr << "  Bucket [1-10]:    " << histogram[1] << " threads" << std::endl;
    //   std::cerr << "  Bucket [11-100]:  " << histogram[2] << " threads" << std::endl;
    //   std::cerr << "  Bucket [101-1K]:  " << histogram[3] << " threads" << std::endl;
    //   std::cerr << "  Bucket [1K-10K]:  " << histogram[4] << " threads" << std::endl;
    //   std::cerr << "  Bucket [10K+]:    " << histogram[5] << " threads" << std::endl;
    //   std::cerr << "=============================================\n" << std::endl;
    // }

    // Prefix sum for write offsets
    auto& dest_rel = get_relation_by_schema<DestSchema, NEW_VER>(db);
    uint32_t old_size = static_cast<uint32_t>(dest_rel.size());

    thrust::exclusive_scan(rmm::exec_policy{}, thread_counts.begin(), thread_counts.end(),
                           thread_counts.begin(), old_size, thrust::plus<uint32_t>());

    // Get total count
    uint32_t total_count = 0;
    GPU_MEMCPY(&total_count, thread_counts.data() + num_threads, sizeof(uint32_t),
               GPU_DEVICE_TO_HOST);
    total_count -= old_size;  // Subtract old_size since we used it as initial value

    if (total_count == 0) {
      nvtxRangePop();
      return;
    }

    // Resize output relation
    uint32_t new_size = old_size + total_count;
    dest_rel.resize_interned_columns(new_size);

    if constexpr (has_provenance_v<SR>) {
      dest_rel.provenance().resize(new_size);
    }

    // Materialize phase
    using ProvPtrType = semiring_value_t<SR>*;
    ProvPtrType prov_ptr = nullptr;
    if constexpr (has_provenance_v<SR>) {
      prov_ptr = dest_rel.provenance().data();
    }

    kernel_materialize<ViewType><<<num_blocks, kBlockSize>>>(
        d_views.data(), root_unique_values_ptr, num_unique_root_keys, num_root_keys,
        thread_counts.data(), dest_rel.template interned_column<0>(), prov_ptr,
        dest_rel.interned_stride());
    GPU_DEVICE_SYNCHRONIZE();

    err = GPU_GET_LAST_ERROR();
    if (err != GPU_SUCCESS) {
      std::cerr << "JIT kernel_materialize failed: " << GPU_GET_ERROR_STRING(err) << std::endl;
    }

    nvtxRangePop();
  }
};

}  // namespace SRDatalog::GPU::JIT
