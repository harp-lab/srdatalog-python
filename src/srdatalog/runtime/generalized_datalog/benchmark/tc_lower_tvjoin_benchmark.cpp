// #include <mimalloc-new-delete.h>
// implement tc with TVJoin Index
// #include "ast.h" // Not needed if we remove Relation usage?
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
// #include "eytzinger.h"
// #include "hashmap.h"
// #include "query.h"
// Include implementation for auto return return type deduction
#include "../relation_col.h"  // Relative include
#include "../semiring.h"      // Relative include
#include "gpu/device_array.h"
#include "gpu/device_tvjoin_index.h"  // CHANGED
#include "gpu/gpu_api.h"
#include "gpu/init.h"
#include "gpu/macro.h"
#include "gpu/nd_device_array.h"  // Added
#include <boost/unordered/unordered_flat_set.hpp>
#include <cassert>
#include <chrono>
#include <nvtx3/nvtx3.hpp>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <sys/mman.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

using namespace SRDatalog;
using namespace SRDatalog::GPU;

namespace {

// ... project_edge_to_path_count (Same as original) ...
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
    return;
  }

  for (uint32_t i = warp_global_id; i < num_unique_keys; i += total_warps) {
    typename IndexType::ValueType x;
    if (lane == 0) {
      x = edge_unique_x_values[i];
    }
    x = tile.shfl(x, 0);

    auto handle = edge_root.prefix(x, tile, edge_view);
    if (lane == 0) {
      counts[i] = handle.valid() ? handle.degree() : 0;
    }
  }
}

// ... project_edge_to_path_materialized (Same as original) ...
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
    return;
  }

  for (uint32_t i = warp_global_id; i < num_unique_keys; i += total_warps) {
    typename IndexType::ValueType x;
    if (lane == 0) {
      x = edge_unique_x_values[i];
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
      // AoS:
      output[target_row * 2 + 0] = x;
      output[target_row * 2 + 1] = y;
      output_prov[target_row] = prov;
    }
  }
}

// ... tc_join_kernel (Same as original) ...
template <typename IndexOuterMost, typename IndexOutput, typename PathHandle, typename EdgeHandle,
          typename PathFullHandle, Semiring SR, typename ProvIter, bool SFlag, int TILE_SIZE = 32>
__global__ void tc_join_kernel(const uint32_t num_root_keys, const int* path_root_unique_ptr,
                               PathHandle path_root, EdgeHandle edge_root,
                               PathFullHandle path_full_root, typename PathHandle::View path_view,
                               typename EdgeHandle::View edge_view,
                               typename PathFullHandle::View path_full_view,
                               uint32_t* counts_or_offset, typename IndexOutput::value_type* output,
                               std::size_t output_stride, ProvIter output_prov) {

  static_assert(SFlag == JOIN_SIZE_ONLY || SFlag == JOIN_MATERIALIZED, "Invalid SFlag");

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<TILE_SIZE>(block);
  auto single_tile = cooperative_groups::tiled_partition<1>(block);

  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_global_id = thread_id / TILE_SIZE;
  uint32_t total_warps = (gridDim.x * blockDim.x) / TILE_SIZE;
  uint32_t lane = tile.thread_rank();

  uint32_t thread_local_new_count = 0;

  uint32_t thread_write_base = 0;
  if constexpr (SFlag == JOIN_MATERIALIZED) {
    thread_write_base = counts_or_offset[thread_id];
  }

  auto emit = [&](const auto& x_, const auto& z_, const auto& prov_) {
    if constexpr (SFlag == JOIN_MATERIALIZED) {
      uint32_t output_pos = thread_write_base + thread_local_new_count;
      // AoS Write: [x, z, x, z...]
      output[output_pos * 2 + 0] = x_;
      output[output_pos * 2 + 1] = z_;
      output_prov[output_pos] = prov_;
    }
    thread_local_new_count++;
  };

  for (uint32_t y_idx = warp_global_id; y_idx < num_root_keys; y_idx += total_warps) {
    typename IndexOuterMost::ValueType y;
    if (lane == 0) {
      y = path_root_unique_ptr[y_idx];
    }
    y = tile.shfl(y, 0);

    // Use scalar prefix to isolate Hybrid search bug
    auto path_y = path_root.prefix(y);
    auto edge_y = edge_root.prefix(y);
    // NodeHandle path_y = path_root.prefix(y, tile, path_view);
    // NodeHandle edge_y = edge_root.prefix(y, tile, edge_view);

    if (!path_y.valid() || !edge_y.valid()) {
      continue;
    }

    uint32_t x_count = path_y.degree();
    uint32_t z_count = edge_y.degree();
    uint32_t total_pairs = x_count * z_count;

    const bool z_major = (z_count >= x_count);

    for (uint32_t pair_idx = lane; pair_idx < total_pairs; pair_idx += TILE_SIZE) {

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

      emit(x, z, new_prov);
    }
  }

  if constexpr (SFlag == JOIN_SIZE_ONLY) {
    counts_or_offset[thread_id] = thread_local_new_count;
  }
}

// ... perform_tc_join adapted for generic Index ...
template <typename PathVerIndex, typename PathFullIndex, typename EdgeIndex, typename PathRelation>
bool perform_tc_join(PathVerIndex& path_ver_1_0, EdgeIndex& edge_full_0_1,
                     PathFullIndex& path_full_1_0, PathRelation& path_newt_rel,
                     DeviceArray<uint32_t>& join_hist, uint32_t num_blocks, uint32_t BLOCK_SIZE,
                     size_t iteration_id) {

  using PathHandle = typename PathVerIndex::NodeHandle;
  using EdgeHandle = typename EdgeIndex::NodeHandle;
  using PathFullHandle = typename PathFullIndex::NodeHandle;

  auto num_outer_most_keys_join = static_cast<uint32_t>(path_ver_1_0.num_unique_root_values());
  using ProvType = decltype(path_newt_rel.provenance().data());

  tc_join_kernel<PathVerIndex, typename PathRelation::DeviceColsType, PathHandle, EdgeHandle,
                 PathFullHandle, BooleanSR, ProvType, JOIN_SIZE_ONLY, WARP_SIZE>
      <<<num_blocks, BLOCK_SIZE>>>(
          num_outer_most_keys_join, path_ver_1_0.root_unique_values().data(), path_ver_1_0.root(),
          edge_full_0_1.root(), path_full_1_0.root(), path_ver_1_0.view(), edge_full_0_1.view(),
          path_full_1_0.view(), join_hist.data(), path_newt_rel.unsafe_interned_columns().data(),
          0,  // Stride unused for AoS logic hardcoded 2
          path_newt_rel.provenance().data());

  GPU_DEVICE_SYNCHRONIZE();

  thrust::exclusive_scan(thrust::device, join_hist.begin(), join_hist.end(), join_hist.begin(), 0,
                         thrust::plus<uint32_t>());
  auto total_count = join_hist[num_blocks * BLOCK_SIZE];
  std::cout << "total_count: " << total_count << std::endl;

  // Resize storage
  path_newt_rel.resize_interned_columns(total_count);
  path_newt_rel.provenance().resize(total_count);

  tc_join_kernel<PathVerIndex, typename PathRelation::DeviceColsType, PathHandle, EdgeHandle,
                 PathFullHandle, BooleanSR, ProvType, JOIN_MATERIALIZED, WARP_SIZE>
      <<<num_blocks, BLOCK_SIZE>>>(
          num_outer_most_keys_join, path_ver_1_0.root_unique_values().data(), path_ver_1_0.root(),
          edge_full_0_1.root(), path_full_1_0.root(), path_ver_1_0.view(), edge_full_0_1.view(),
          path_full_1_0.view(), join_hist.data(), path_newt_rel.unsafe_interned_columns().data(),
          0,  // Stride (unused for AoS)
          path_newt_rel.provenance().data());

  GPU_DEVICE_SYNCHRONIZE();

  GPU_ERROR_T err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    LOG_ERROR << "TC join materialized kernel failed: " << GPU_GET_ERROR_STRING(err);
    return false;
  }
  return true;
}

// StorageWrapper removed.
// Uses Relation<BooleanSR, tuple<int, int>, DeviceTVJoinIndex, DeviceRelationPolicy> instead.

std::vector<std::tuple<int, int>> load_edge_list(const char* filename) {
  std::vector<std::tuple<int, int>> edges;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return edges;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;  // Skip comments and empty lines
    int u, v;
    char comma;
    std::istringstream iss(line);
    // Try "u,v" format first
    if (iss >> u >> comma >> v && comma == ',') {
      edges.emplace_back(u, v);
    } else {
      // Fallback to "u v" format
      iss.clear();
      iss.str(line);
      if (iss >> u >> v) {
        edges.emplace_back(u, v);
      }
    }
  }
  return edges;
}

// Adaptor for (0, 1) index
template <typename SR, typename AttrTuple, typename... Args>
using EdgeTVJoinIndex = DeviceTVJoinIndex<SR, typename std::tuple_element_t<0, AttrTuple>,
                                          std::tuple_size_v<AttrTuple>, 0, 1>;

// Adaptor for (1, 0) index
template <typename SR, typename AttrTuple, typename... Args>
using PathTVJoinIndex = DeviceTVJoinIndex<SR, typename std::tuple_element_t<0, AttrTuple>,
                                          std::tuple_size_v<AttrTuple>, 1, 0>;

}  // namespace

int main(int argc, char** argv) {
  init_cuda();

  std::vector<std::tuple<int, int>> host_edges;
  if (argc > 1) {
    host_edges = load_edge_list(argv[1]);
  } else {
    // Gen random?
    std::cout << "Usage: " << argv[0] << " <edge_file>" << std::endl;
    return 1;
  }

  if (host_edges.empty()) {
    std::cout << "No edges loaded." << std::endl;
    return 0;
  }

  // Convert to Device TVJoin
  // 1. Storage
  using EdgeRelation = SRDatalog::Relation<BooleanSR, std::tuple<int, int>, EdgeTVJoinIndex,
                                           DeviceRelationPolicy, int>;
  EdgeRelation edge_storage;
  size_t n = host_edges.size();
  edge_storage.resize_interned_columns(n);

  // Copy columns
  std::vector<int> col0(n), col1(n);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = std::get<0>(host_edges[i]);
    col1[i] = std::get<1>(host_edges[i]);
  }

  // AoS Ingestion: Strided copy manually?
  // Or copy tuple by tuple?
  // Host is SoA vectors (col0, col1).
  // Target is AoS [c0, c1, c0, c1].
  // We can't just memcpy columns.
  // We need to interleave.
  std::vector<int> interleaved(n * 2);
  for (size_t i = 0; i < n; ++i) {
    interleaved[i * 2 + 0] = col0[i];
    interleaved[i * 2 + 1] = col1[i];
  }
  cudaMemcpy(edge_storage.unsafe_interned_columns().data(), interleaved.data(), n * 2 * sizeof(int),
             cudaMemcpyHostToDevice);

  // Set all provenance to true
  thrust::fill(thrust::device, edge_storage.provenance().begin(), edge_storage.provenance().end(),
               true);

  // 2. Index (TVJoin)
  using EdgeIndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0, 1>;
  EdgeIndexType edge_idx_0_1;
  IndexSpec spec_0_1{{0, 1}};

  // Clone data for destructive build
  AoSDeviceArray<int, 2> temp_cols;
  edge_storage.unsafe_interned_columns().clone_into(temp_cols);
  edge_idx_0_1.build_from_encoded_device(spec_0_1, temp_cols, edge_storage.provenance());

  std::cout << "Edge full device has " << edge_idx_0_1.size() << " rows" << std::endl;

  // Path Relations
  using PathIndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 1, 0>;
  PathIndexType path_full_idx_1_0;
  PathIndexType path_delta_idx_1_0;

  using PathRelation = SRDatalog::Relation<BooleanSR, std::tuple<int, int>, PathTVJoinIndex,
                                           DeviceRelationPolicy, int>;
  PathRelation path_newt_rel;

  // Initial step: Path = Edge
  // Project Edge -> Path (1,0)
  edge_storage.unsafe_interned_columns().clone_into(temp_cols);
  path_full_idx_1_0.build_from_encoded_device(spec_0_1, temp_cols, edge_storage.provenance());

  // Also Delta = Edge initially?
  edge_storage.unsafe_interned_columns().clone_into(temp_cols);
  path_delta_idx_1_0.build_from_encoded_device(spec_0_1, temp_cols, edge_storage.provenance());

  // DIAGNOSTIC PROBE
  // Probe Edge(0,1) for y=2. Expect z=3.
  {
    auto root = edge_idx_0_1.root();
    auto view = edge_idx_0_1.view();
    int key = 2;
    using H = decltype(root);
    using V = decltype(view);

    // Launch small kernel
    // Lambda kernel require extended lambda support or global.
    // Use standard kernel if poss, or cheat with new method.
  }
  // Defined outside main ideally.

  std::size_t iteration = 0;
  constexpr std::size_t max_iterations = 1000;  // Limit for bench

  uint32_t num_blocks = 128;  // Tweak
  uint32_t BLOCK_SIZE = 256;
  DeviceArray<uint32_t> join_hist((num_blocks * BLOCK_SIZE) + 1);

  auto start_time = std::chrono::high_resolution_clock::now();

  while (true) {
    std::cout << "Iteration " << iteration << " delta size: " << path_delta_idx_1_0.size()
              << std::endl;

    path_newt_rel.clear();
    path_newt_rel.resize_interned_columns(path_delta_idx_1_0.size());
    path_newt_rel.provenance().resize(path_delta_idx_1_0.size());

    // My setup:
    // path_delta_idx_1_0 (y, x). Correct.
    // edge_idx_0_1 (y, z). Correct.
    // Output: x (from path), z (from edge).

    bool not_empty;
    {
      nvtx3::scoped_range range{"Join"};
      not_empty = perform_tc_join(path_delta_idx_1_0,  // ver
                                  edge_idx_0_1,        // full
                                  path_full_idx_1_0,   // full check
                                  path_newt_rel, join_hist, num_blocks, BLOCK_SIZE, iteration);
    }

    if (!not_empty || path_newt_rel.interned_size() == 0)
      break;

    IndexSpec spec{{1, 0}};

    // Set difference: write directly into path_delta_idx_1_0 (reuses memory via RMM VMM)
    {
      nvtx3::scoped_range range{"SetDifference"};
      PathIndexType::set_difference_from_data(path_newt_rel.unsafe_interned_columns(),
                                              path_full_idx_1_0, path_delta_idx_1_0);
    }

    if (path_delta_idx_1_0.empty())
      break;

    std::cout << "Iteration " << iteration << " delta size: " << path_delta_idx_1_0.size()
              << std::endl;

    // Merge delta into full (delta's sorted_arrs remain valid after merge)
    {
      nvtx3::scoped_range range{"Merge"};
      path_full_idx_1_0.merge(path_delta_idx_1_0, 0);
    }

    // path_delta_idx_1_0 is now ready for next iteration - its sorted_arrs
    // point to data now owned by full (still valid!)

    iteration++;
    if (iteration >= max_iterations)
      break;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::cout << "TC benchmark finished in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << " ms" << std::endl;
  std::cout << "Final Full Path Size: " << path_full_idx_1_0.size() << std::endl;

  // PathRelation final_storage;
  // final_storage.resize_interned_columns(path_full_idx_1_0.size());
  // final_storage.provenance().resize(path_full_idx_1_0.size());
  // path_full_idx_1_0.reconstruct_to_relation(spec_0_1, final_storage.unsafe_interned_columns(),
  //                                           final_storage.provenance());

  // std::vector<int> h_aos(final_storage.interned_size() * 2);
  // cudaMemcpy(h_aos.data(), final_storage.unsafe_interned_columns().data(),
  //            h_aos.size() * sizeof(int), cudaMemcpyDeviceToHost);

  // std::cout << "Tuples:" << std::endl;
  // for (size_t i = 0; i < final_storage.num_rows(); ++i) {
  //   int x = h_aos[i * 2 + 0];
  //   int y = h_aos[i * 2 + 1];
  //   std::cout << x << " -> " << y << std::endl;
  // }

  /*
    std::vector<int> h_col0(final_storage.num_rows());
    std::vector<int> h_col1(final_storage.num_rows());

    cudaMemcpy(h_col0.data(), final_storage.column_ptr<0>(), h_col0.size() * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col1.data(), final_storage.column_ptr<1>(), h_col1.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "Tuples:" << std::endl;
    for (size_t i = 0; i < h_col0.size(); ++i) {
      std::cout << h_col0[i] << " -> " << h_col1[i] << std::endl;
    }
  */

  return 0;
}
