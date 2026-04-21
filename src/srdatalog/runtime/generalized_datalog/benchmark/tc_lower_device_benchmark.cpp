// #include <mimalloc-new-delete.h>
// implement tc with relation API
#include "ast.h"
#include "eytzinger.h"
#include "hashmap.h"
#include "query.h"
// Include implementation for auto return type deduction
#include "gpu/device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include "gpu/init.h"
#include "gpu/macro.h"  // For JOIN_SIZE_ONLY and JOIN_MATERIALIZED
#include "hashmap.ipp"
#include "runtime.h"
#include "semiring.h"
#include <boost/unordered/unordered_flat_set.hpp>
#include <cassert>
#include <chrono>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <sys/mman.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================================
// VTune ITT (Instrumentation and Tracing Technology) for block-wise profiling
// ============================================================================
// Compile with -DENABLE_VTUNE_ITT to enable VTune user-defined regions
// Link with: -L/opt/intel/oneapi/vtune/latest/sdk/lib64 -littnotify
// ============================================================================
#ifdef ENABLE_VTUNE_ITT
#include <ittnotify.h>
static __itt_domain* vtune_domain = __itt_domain_create("SRDatalog");
static __itt_string_handle* handle_copy_edge_to_path =
    __itt_string_handle_create("block: copy_edge_to_path");
static __itt_string_handle* handle_tc_join_loop = __itt_string_handle_create("block: tc_join_loop");
static __itt_string_handle* handle_tc_join_loop_dedup_full =
    __itt_string_handle_create("block: tc_join_loop_dedup_full");
static __itt_string_handle* handle_tc_join_loop_size =
    __itt_string_handle_create("block: tc_join_loop_size");
static __itt_string_handle* handle_build_index_path_newt =
    __itt_string_handle_create("block: build_index_path_newt");
static __itt_string_handle* handle_copy_path_newt =
    __itt_string_handle_create("block: copy_path_newt_to_full");
static __itt_string_handle* handle_merge_index = __itt_string_handle_create("block: merge_index");

#define VTUNE_TASK_BEGIN(handle) __itt_task_begin(vtune_domain, __itt_null, __itt_null, handle)
#define VTUNE_TASK_END() __itt_task_end(vtune_domain)
#else
#define VTUNE_TASK_BEGIN(handle) ((void)0)
#define VTUNE_TASK_END() ((void)0)
#endif

namespace {
template <typename IndexType, typename NodeHandle, int GROUP_SIZE = WARP_SIZE>
__global__ void project_edge_to_path_count(
    const typename IndexType::ValueType* edge_unique_x_values, NodeHandle edge_root,
    typename NodeHandle::View edge_view, uint32_t num_unique_keys, uint32_t* counts) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<GROUP_SIZE>(block);
  uint32_t warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP_SIZE;
  uint32_t total_warps = (gridDim.x * blockDim.x) / GROUP_SIZE;
  uint32_t lane = tile.thread_rank();

  if (!edge_unique_x_values) {
    return;  // Should not happen in normal operation
  }

  for (uint32_t i = warp_global_id; i < num_unique_keys; i += total_warps) {
    // each warp processes one unique outer-most column key
    typename IndexType::ValueType x;
    if (lane == 0) {
      x = edge_unique_x_values[i];  // Use unique values for partitioning
    }
    x = tile.shfl(x, 0);

    auto handle = edge_root.prefix(x, tile, edge_view);
    // only lane 0 in each warp reports the count
    if (lane == 0) {
      counts[i] = handle.valid() ? handle.degree() : 0;
    }
  }
}

template <typename IndexType, typename NodeHandle, typename OutputIter, typename ProvIter,
          int GROUP_SIZE = WARP_SIZE>
__global__ void project_edge_to_path_materialized(
    const typename IndexType::ValueType* edge_unique_x_values, NodeHandle edge_root,
    typename NodeHandle::View edge_view, uint32_t num_unique_keys, OutputIter output,
    std::size_t output_stride, ProvIter output_prov) {

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<GROUP_SIZE>(block);

  uint32_t warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP_SIZE;
  uint32_t total_warps = (gridDim.x * blockDim.x) / GROUP_SIZE;
  uint32_t lane = tile.thread_rank();

  if (!edge_unique_x_values) {
    return;  // Should not happen in normal operation
  }

  for (uint32_t i = warp_global_id; i < num_unique_keys; i += total_warps) {
    // each warp processes one unique outer-most column key
    typename IndexType::ValueType x;
    if (lane == 0) {
      x = edge_unique_x_values[i];  // Use unique values for partitioning
    }
    x = tile.shfl(x, 0);
    typename IndexType::NodeHandle edge_x = edge_root.prefix(x, tile, edge_view);
    if (!edge_x.valid()) {
      continue;
    }
    size_t write_start_index = 0;
    if (lane == 0) {
      write_start_index = edge_x.begin();
    }
    write_start_index = tile.shfl(write_start_index, 0);
    uint32_t degree = edge_x.degree();
    for (uint32_t j = lane; j < degree; j += GROUP_SIZE) {
      auto y = edge_x.get_value_at(edge_view, j);
      auto prov = edge_x.get_provenance_at(edge_view, j);
      size_t target_row = write_start_index + j;
      output[(0 * output_stride) + target_row] = x;
      output[(1 * output_stride) + target_row] = y;
      output_prov[target_row] = prov;
    }
  }
}

template <typename IndexOuterMost, typename IndexOutput, SRDatalog::CDeviceHandle NodeHandle,
          Semiring SR, typename ProvIter, typename MaskIter, bool SFlag, int GROUP_SIZE = WARP_SIZE>
__global__ void tc_join_kernel(const typename IndexOuterMost::ValueType* path_root_unique_ptr,
                               NodeHandle path_root, NodeHandle edge_root,
                               NodeHandle path_full_root, typename NodeHandle::View path_view,
                               typename NodeHandle::View edge_view,
                               typename NodeHandle::View path_full_view, uint32_t num_root_keys,
                               uint32_t* counts_or_offset, typename IndexOutput::ValueType* output,
                               std::size_t output_stride, ProvIter output_prov, MaskIter mask) {

  static_assert(SFlag == JOIN_SIZE_ONLY || SFlag == JOIN_MATERIALIZED, "Invalid SFlag");

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<GROUP_SIZE>(block);
  auto single_tile = cooperative_groups::tiled_partition<1>(block);

  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_global_id = thread_id / GROUP_SIZE;
  uint32_t total_warps = (gridDim.x * blockDim.x) / GROUP_SIZE;
  uint32_t lane = tile.thread_rank();

  uint32_t thread_local_new_count = 0;

  uint32_t thread_write_base = 0;
  if constexpr (SFlag == JOIN_MATERIALIZED) {
    thread_write_base = counts_or_offset[thread_id];
  }

  // Helper lambda to emit a new tuple
  auto emit = [&](const auto& x_, const auto& z_, const auto& prov_, const auto& mask_) {
    if constexpr (SFlag == JOIN_MATERIALIZED) {
      uint32_t output_pos = thread_write_base + thread_local_new_count;
      output[(0 * output_stride) + output_pos] = x_;
      output[(1 * output_stride) + output_pos] = z_;
      output_prov[output_pos] = prov_;
      mask[output_pos] = mask_;
    }
    thread_local_new_count++;
  };

  for (uint32_t y_idx = warp_global_id; y_idx < num_root_keys; y_idx += total_warps) {
    typename IndexOuterMost::ValueType y;
    if (lane == 0) {
      // Use root_unique_values for partitioning (exactly what it's designed for)
      y = path_root_unique_ptr[y_idx];
    }
    y = tile.shfl(y, 0);

    NodeHandle path_y = path_root.prefix(y, tile, path_view);
    NodeHandle edge_y = edge_root.prefix(y, tile, edge_view);

    if (!path_y.valid() || !edge_y.valid()) {
      continue;
    }

    uint32_t x_count = path_y.degree();
    uint32_t z_count = edge_y.degree();
    uint32_t total_pairs = x_count * z_count;

    const bool z_major = (z_count >= x_count);

    for (uint32_t pair_idx = lane; pair_idx < total_pairs; pair_idx += GROUP_SIZE) {

      uint32_t x_idx, z_idx;
      if (z_major) {
        x_idx = pair_idx / z_count;
        z_idx = pair_idx % z_count;
      } else {
        z_idx = pair_idx / x_count;
        x_idx = pair_idx % x_count;
      }

      auto x = path_y.get_value_at(path_view, x_idx);
      auto z = edge_y.get_value_at(edge_view, z_idx);
      auto new_prov = sr_mul<SR>(path_y.get_provenance_at(path_view, x_idx),
                                 edge_y.get_provenance_at(edge_view, z_idx));

      auto path_full_z = path_full_root.prefix(z, single_tile, path_full_view);
      if (!path_full_z.valid()) {
        emit(x, z, new_prov, false);
        continue;
      }

      auto path_full_z_x = path_full_z.prefix(x, single_tile, path_full_view);
      if (!path_full_z_x.valid()) {
        emit(x, z, new_prov, false);
        continue;
      }

      auto old_prov = path_full_z_x.get_provenance_at(path_full_view, 0);
      auto combined_prov = sr_add<SR>(new_prov, old_prov);
      if (combined_prov != old_prov) {
        // in pure datalog this never happens
        // Atomic in-place update for existing tuple in full
        // path_full_z_x.begin() gives the absolute index in the column/provenance array
        auto* prov_ptr = path_full_view.provenance_ptr_;
        if (prov_ptr) {
          sr_atomic_add<SR>(prov_ptr + path_full_z_x.begin(), new_prov);
        }
        emit(x, z, new_prov, true);
      }
    }
  }

  if constexpr (SFlag == JOIN_SIZE_ONLY) {
    counts_or_offset[thread_id] = thread_local_new_count;
  }
}

template <typename PathVer, typename PathFull, typename EdgeRootHandle, typename PathFullRootHandle,
          typename PathDeviceType>
bool perform_tc_join(
    PathVer& path_ver_1_0, EdgeRootHandle edge_full_0_1, PathFullRootHandle path_full_1_0,
    typename PathVer::NodeHandle::View path_ver_view, typename EdgeRootHandle::View edge_full_view,
    typename PathFullRootHandle::View path_full_view, PathDeviceType& path_newt_device,
    SRDatalog::GPU::DeviceArray<uint32_t>& join_hist, SRDatalog::GPU::DeviceArray<bool>& mask,
    uint32_t num_blocks, uint32_t BLOCK_SIZE) {
  VTUNE_TASK_BEGIN(handle_tc_join_loop_size);
  // compute the join size
  // Get number of unique Y values from path_ver_1_0 for partitioning
  auto num_outer_most_keys_join = static_cast<uint32_t>(path_ver_1_0.num_unique_root_values());
  // join_hist is pre-allocated, no need to resize
  assert(join_hist.size() >= (num_blocks * BLOCK_SIZE) + 1);
  // std::cout << "join_hist size: " << join_hist.size() << std::endl;
  using ProvType = decltype(path_newt_device.provenance().data());
  using PathVerNodeHandle = typename PathVer::NodeHandle;
  tc_join_kernel<PathVer, PathFull, PathVerNodeHandle, BooleanSR, ProvType, bool*, JOIN_SIZE_ONLY,
                 WARP_SIZE><<<num_blocks, BLOCK_SIZE>>>(
      path_ver_1_0.root_unique_values().data(), path_ver_1_0.root(), edge_full_0_1, path_full_1_0,
      path_ver_view, edge_full_view, path_full_view, num_outer_most_keys_join, join_hist.data(),
      path_newt_device.template interned_column<0>(), path_newt_device.interned_stride(),
      path_newt_device.provenance().data(), mask.data());
  GPU_DEVICE_SYNCHRONIZE();
  VTUNE_TASK_END();
  VTUNE_TASK_BEGIN(handle_tc_join_loop);
  thrust::exclusive_scan(thrust::device, join_hist.begin(), join_hist.end(), join_hist.begin(), 0,
                         thrust::plus<uint32_t>());
  auto total_count = join_hist[num_blocks * BLOCK_SIZE];
  std::cout << "total_count: " << total_count << std::endl;
  path_newt_device.resize_interned_columns(total_count);
  path_newt_device.provenance().resize(total_count);
  mask.resize(total_count);
  tc_join_kernel<PathVer, PathFull, PathVerNodeHandle, BooleanSR, ProvType, bool*,
                 JOIN_MATERIALIZED, WARP_SIZE><<<num_blocks, BLOCK_SIZE>>>(
      path_ver_1_0.root_unique_values().data(), path_ver_1_0.root(), edge_full_0_1, path_full_1_0,
      path_ver_view, edge_full_view, path_full_view, num_outer_most_keys_join, join_hist.data(),
      path_newt_device.template interned_column<0>(), path_newt_device.interned_stride(),
      path_newt_device.provenance().data(), mask.data());
  GPU_DEVICE_SYNCHRONIZE();
  // catch error
  VTUNE_TASK_END();
  GPU_ERROR_T err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    LOG_ERROR << "TC join materialized kernel failed: " << GPU_GET_ERROR_STRING(err);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  SRDatalog::GPU::init_cuda();

  // First, load edge data into host relation
  SRDatalog::Relation<BooleanSR, std::tuple<int, int>, SRDatalog::HashTrieIndex,
                      SRDatalog::HostRelationPolicy>
      edge_full_host;
  edge_full_host.set_version(FULL_VER);

  const std::string edge_path = argv[1];
  SRDatalog::load_file(edge_full_host, edge_path);

  // Convert edge from host to device
  using EdgeDeviceType =
      SRDatalog::Relation<BooleanSR, std::tuple<int, int>, SRDatalog::GPU::DeviceSortedArrayIndex,
                          SRDatalog::DeviceRelationPolicy, uint32_t>;

  EdgeDeviceType edge_full_device = EdgeDeviceType::to_device(edge_full_host);
  std::cout << "Edge full device has " << edge_full_device.interned_size() << " rows" << std::endl;

  // Create path relations directly on device (they will be populated during computation)
  using PathDeviceType =
      SRDatalog::Relation<BooleanSR, std::tuple<int, int>, SRDatalog::GPU::DeviceSortedArrayIndex,
                          SRDatalog::DeviceRelationPolicy, uint32_t>;
  PathDeviceType path_full_device;
  path_full_device.set_version(FULL_VER);
  PathDeviceType path_delta_device;
  path_delta_device.set_version(DELTA_VER);
  PathDeviceType path_newt_device;
  path_newt_device.set_version(NEW_VER);

  const auto& edge_full_idx = edge_full_device.ensure_index({{0, 1}});

  auto start_time = std::chrono::high_resolution_clock::now();

  // copy edge to path
  auto edge_full_idx_root = edge_full_idx.root();
  // get the number of unique root keys for partitioning
  uint32_t num_unique_root_keys = static_cast<uint32_t>(edge_full_idx.num_unique_root_values());
  constexpr uint32_t BLOCK_SIZE = 256;
  SRDatalog::GPU::DeviceArray<uint32_t> counts(num_unique_root_keys + 1);
  int num_sms = 0;
  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);
  uint32_t num_blocks = num_sms * 4;  // 4 blocks per SM typically good
  project_edge_to_path_count<EdgeDeviceType::IndexTypeInst, decltype(edge_full_idx_root), WARP_SIZE>
      <<<num_blocks, BLOCK_SIZE>>>(edge_full_idx.root_unique_values().data(), edge_full_idx_root,
                                   edge_full_idx.view(), num_unique_root_keys, counts.data());
  GPU_DEVICE_SYNCHRONIZE();
  thrust::exclusive_scan(thrust::device, counts.begin(), counts.end(), counts.begin(), 0,
                         thrust::plus<uint32_t>());
  auto total_count = counts[num_unique_root_keys];
  std::cout << "Total count: " << total_count << std::endl;
  // path_newt_device.reserve(total_count);
  path_full_device.resize_interned_columns(total_count);
  path_full_device.provenance().resize(total_count);
  auto* path_full_data = path_full_device.interned_column<0>();
  auto* path_full_prov = path_full_device.provenance().data();
  auto path_full_stride = path_full_device.interned_stride();
  project_edge_to_path_materialized<EdgeDeviceType::IndexTypeInst><<<num_blocks, BLOCK_SIZE>>>(
      edge_full_idx.root_unique_values().data(), edge_full_idx_root, edge_full_idx.view(),
      num_unique_root_keys, path_full_data, path_full_stride, path_full_prov);
  GPU_DEVICE_SYNCHRONIZE();
  const auto& path_full_1_0 = path_full_device.ensure_index({{1, 0}});
  // path_full_1_0.data().print_first(10);
  std::cout << "Total count: " << total_count << std::endl;

  start_time = std::chrono::high_resolution_clock::now();

  // TC
  std::size_t iteration = 0;
  constexpr std::size_t max_iterations = std::numeric_limits<std::size_t>::max();
  // Pre-allocate join_hist with max size to avoid reallocation every iteration
  SRDatalog::GPU::DeviceArray<uint32_t> join_hist((num_blocks * BLOCK_SIZE) + 1);
  while (true) {
    auto path_full_1_0 = path_full_device.get_index({{1, 0}}).root();
    auto edge_full_0_1 = edge_full_device.get_index({{0, 1}}).root();

    path_newt_device.reserve_interned(path_delta_device.size() * 2);

    auto& path_ver = (iteration == 0) ? path_full_device : path_delta_device;
    auto& path_ver_1_0 = path_ver.get_index({{1, 0}});
    using PathVer_1_0Type = typename std::decay_t<decltype(path_ver_1_0)>;
    auto path_ver_1_0_root = path_ver_1_0.root();
    SRDatalog::GPU::DeviceArray<bool> delta_mask;

    auto& path_full_1_0_idx = path_full_device.get_index({{1, 0}});
    using PathFull_1_0Type = typename std::decay_t<decltype(path_full_1_0_idx)>;
    if (!perform_tc_join<PathVer_1_0Type, PathFull_1_0Type, decltype(edge_full_0_1),
                         decltype(path_full_1_0), decltype(path_newt_device)>(

            path_ver_1_0, edge_full_0_1, path_full_1_0, path_ver_1_0.view(),
            edge_full_device.get_index({{0, 1}}).view(),
            path_full_device.get_index({{1, 0}}).view(), path_newt_device, join_hist, delta_mask,
            num_blocks, BLOCK_SIZE)) {
      break;
    }

    // Split Strategy:
    // 1. Calculate how many are unmasked (new valid paths)
    // 2. If ALL are unmasked, shortcut: merge path_newt (as delta) directly to full
    // 3. If PARTIAL masked, filter unmasked to path_persist, merge persist to full

    // Count unmasked (persist)
    auto raw_mask = delta_mask.data();
    std::size_t persist_count = thrust::count_if(
        thrust::device, delta_mask.begin(), delta_mask.end(), [] __device__(bool m) { return !m; });

    std::size_t current_total_count = path_newt_device.interned_size();

    if (persist_count == current_total_count) {
      // Optimization: All tuples are new (unmasked).
      // path_newt contains exactly the tuples we want to add to full.

      path_newt_device.ensure_index({{1, 0}});
      const auto& path_newt_1_0 = path_newt_device.get_index({{1, 0}});
      std::size_t path_full_start_size = path_full_device.get_index({{1, 0}}).size();
      VTUNE_TASK_BEGIN(handle_merge_index);
      try {
        auto& path_full_idx = const_cast<PathDeviceType&>(path_full_device).get_index({{1, 0}});
        path_full_idx.merge(path_newt_1_0, path_full_start_size);
      } catch (const std::exception& e) {
        std::cerr << "Error during direct merge: " << e.what() << std::endl;
        throw;
      }
      VTUNE_TASK_END();

      // Finalize delta for next iter
      VTUNE_TASK_BEGIN(handle_build_index_path_newt);
      path_newt_device.reconstruct_from_index({{1, 0}});
      VTUNE_TASK_END();

    } else if (persist_count > 0) {
      // Fallback: Filter unmasked to path_persist
      PathDeviceType path_persist_device;
      path_persist_device.set_version(NEW_VER);
      path_persist_device.resize_interned_columns(persist_count);
      path_persist_device.provenance().resize(persist_count);

      // Copy unmasked
      auto raw_newt_col0 = path_newt_device.interned_column<0>();
      auto raw_newt_col1 = path_newt_device.interned_column<1>();
      auto raw_newt_prov = path_newt_device.provenance().data();

      auto raw_persist_col0 = path_persist_device.interned_column<0>();
      auto raw_persist_col1 = path_persist_device.interned_column<1>();
      auto raw_persist_prov = path_persist_device.provenance().data();

      auto newt_zip = thrust::make_zip_iterator(
          thrust::make_tuple(raw_newt_col0, raw_newt_col1, raw_newt_prov));
      auto persist_zip = thrust::make_zip_iterator(
          thrust::make_tuple(raw_persist_col0, raw_persist_col1, raw_persist_prov));

      thrust::copy_if(thrust::device, newt_zip, newt_zip + current_total_count, raw_mask,
                      persist_zip, [] __device__(bool m) { return !m; });

      // Build persist index and merge to full
      path_persist_device.ensure_index({{1, 0}});
      const auto& path_persist_1_0 = path_persist_device.get_index({{1, 0}});

      VTUNE_TASK_BEGIN(handle_merge_index);
      auto& path_full_idx = const_cast<PathDeviceType&>(path_full_device).get_index({{1, 0}});
      path_full_idx.merge(path_persist_1_0, path_full_idx.size());
      VTUNE_TASK_END();

      path_persist_device.clear();

      // Prepare delta (path_newt) for next iter
      path_newt_device.ensure_index({{1, 0}});
      VTUNE_TASK_BEGIN(handle_build_index_path_newt);
      path_newt_device.reconstruct_from_index({{1, 0}});
      VTUNE_TASK_END();

    } else {
      // persist_count == 0 (all masked, only in-place prov updates)
      // Nothing to merge to full (already there).
      // Just prepare delta for propagation.
      path_newt_device.ensure_index({{1, 0}});
      VTUNE_TASK_BEGIN(handle_build_index_path_newt);
      path_newt_device.reconstruct_from_index({{1, 0}});
      VTUNE_TASK_END();
    }

    std::swap(path_newt_device, path_delta_device);
    path_newt_device.clear();

    if (path_delta_device.size() == 0) {
      break;
    }

    iteration++;
    if (iteration >= max_iterations) {
      break;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "TC benchmark finished in " << duration.count() << " milliseconds" << std::endl;

  // reconstruct full
  path_full_device.reconstruct_from_index({{1, 0}});
  auto& path_full_idx = path_full_device.get_index({{1, 0}});
  LOG_INFO << "path_full_device size: " << path_full_idx.size();
  // path_full_device.get_index({{1, 0}}).data().print_first(20);

  // Print RMM memory usage report
  SRDatalog::GPU::print_rmm_memory_report();
}