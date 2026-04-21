#include <benchmark/benchmark.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <tuple>
#include <vector>

#include "gpu/aos_device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/device_tvjoin_index.h"
#include "gpu/init.h"
#include "gpu/nd_device_array.h"
#include "semiring.h"

using namespace SRDatalog::GPU;
using SR = BooleanSR;
namespace cg = cooperative_groups;

using Spec = SRDatalog::IndexSpec;
const Spec spec = {0, 1};

// ------------------------------------------------------------------------------------------------
// DEVICE GENERATION
// ------------------------------------------------------------------------------------------------

// Simple PCG hash for fast random numbers
// Using explicit __device__ function
__device__ uint32_t hash_pcg(uint32_t i) {
  uint32_t state = i * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

struct GenerateRows {
  size_t n;
  __device__ void operator()(size_t idx, uint32_t* c0, uint32_t* c1) {
    if (idx >= n)
      return;
    uint32_t h = hash_pcg(idx);
    // Keys in range [0, n/10] to create collisions
    c0[idx] = h % (n / 10);
    c1[idx] = h % n;
  }
};

struct GenerateFlat {
  size_t n;
  __device__ void operator()(size_t idx, int* io_data) {
    // idx is tuple index
    if (idx >= n)
      return;
    uint32_t h = hash_pcg(idx);
    io_data[2 * idx] = h % (n / 10);
    io_data[2 * idx + 1] = h % n;

    // io_data size is 2*n, we write 2 ints per thread
  }
};

// Launcher for Flat Gen
__global__ void generate_flat_kernel(size_t n, int* data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    GenerateFlat{n}(tid, data);  // Writes 2*tid and 2*tid+1
  }
}

// Launcher for Col Gen
__global__ void generate_cols_kernel(size_t n, uint32_t* c0, uint32_t* c1) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    GenerateRows{n}(tid, c0, c1);
  }
}

// ------------------------------------------------------------------------------------------------
// 3-ARITY GENERATION
// ------------------------------------------------------------------------------------------------

struct GenerateRows3 {
  size_t n;
  __device__ void operator()(size_t idx, uint32_t* c0, uint32_t* c1, uint32_t* c2) {
    if (idx >= n)
      return;
    uint32_t h = hash_pcg(idx);
    // Dense keys [0, 10) for deep join interaction
    c0[idx] = h % 10;
    c1[idx] = (h >> 2) % 10;
    c2[idx] = (h >> 4) % 10;
  }
};

struct GenerateFlat3 {
  size_t n;
  __device__ void operator()(size_t idx, int* io_data) {
    if (idx >= n)
      return;
    uint32_t h = hash_pcg(idx);
    // Dense keys [0, 10) for deep join interaction
    io_data[3 * idx] = h % 10;
    io_data[3 * idx + 1] = (h >> 2) % 10;
    io_data[3 * idx + 2] = (h >> 4) % 10;
  }
};

__global__ void generate_flat_kernel3(size_t n, int* data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    GenerateFlat3{n}(tid, data);
  }
}

__global__ void generate_cols_kernel3(size_t n, uint32_t* c0, uint32_t* c1, uint32_t* c2) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    GenerateRows3{n}(tid, c0, c1, c2);
  }
}

// ------------------------------------------------------------------------------------------------
// PROBE KERNELS
// ------------------------------------------------------------------------------------------------
template <typename Handle>
__global__ void tvjoin_probe_kernel(Handle handle, const int* keys, int num_keys,
                                    unsigned long long* total_matches) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_keys)
    return;
  int key = keys[tid];
  auto sub_handle = handle.prefix(key);
  unsigned long long count = sub_handle.degree();
  atomicAdd(total_matches, count);
}

template <typename Handle>
__global__ void tvjoin_probe_kernel_3arity(Handle root, const int* keys, int num_keys,
                                           unsigned long long* total_matches) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_keys)
    return;
  // keys is flat [k1, k2, k3] ...
  // Wait, benchmark usually generates 1 key to probe?
  // Or do we probe with full tuples?
  // "one arity value per time binary search"
  // Let's assume we probe with (k1, k2, k3) to check full existence/degree.

  int k1 = keys[3 * tid];
  int k2 = keys[3 * tid + 1];
  int k3 = keys[3 * tid + 2];

  auto h1 = root.prefix(k1);
  auto h2 = h1.prefix(k2);
  auto h3 = h2.prefix(k3);

  unsigned long long count = 0;
  if (h3.valid()) {
    // Leaf level degree?
    // h3 is handle after 3 prefixes. If valid, it points to a range in the hypothetical next level
    // (provenance?). For TVJoin, degree() on handle returns size of range.
    count = h3.degree();
  }
  atomicAdd(total_matches, count);
}

template <typename Handle, typename View>
__global__ void sorted_probe_kernel(Handle root, View view, const int* keys, int num_keys,
                                    unsigned long long* total_matches) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_keys)
    return;
  int key = keys[tid];
  auto group = cg::tiled_partition<1>(cg::this_thread_block());
  auto handle = root.prefix(key, group, view);
  unsigned long long count = 0;
  if (handle.valid())
    count = handle.degree();
  atomicAdd(total_matches, count);
}

template <typename Handle, typename View>
__global__ void sorted_probe_kernel_3arity(Handle root, View view, const int* keys, int num_keys,
                                           unsigned long long* total_matches) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_keys)
    return;

  int k1 = keys[3 * tid];
  int k2 = keys[3 * tid + 1];
  int k3 = keys[3 * tid + 2];

  auto group = cg::tiled_partition<1>(cg::this_thread_block());

  auto h1 = root.prefix(k1, group, view);
  auto h2 = h1.prefix(k2, group, view);
  auto h3 = h2.prefix(k3, group, view);

  unsigned long long count = 0;
  if (h3.valid())
    count = h3.degree();
  atomicAdd(total_matches, count);
}

// ------------------------------------------------------------------------------------------------
// GENERIC COOPERATIVE KERNELS (Templated on TileSize)
// ------------------------------------------------------------------------------------------------

template <int TileSize, typename Handle>
__global__ void tvjoin_cooperative_probe_kernel(Handle root, const int* keys, int num_keys,
                                                unsigned long long* total_matches) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(cta);

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tile_id = global_tid / TileSize;
  int num_tiles = (gridDim.x * blockDim.x) / TileSize;

  for (int k = global_tile_id; k < num_keys; k += num_tiles) {
    int key = keys[k];
    auto handle = root.prefix(key, tile, 0);

    if (tile.thread_rank() == 0) {
      unsigned long long count = handle.degree();
      atomicAdd(total_matches, count);
    }
  }
}

template <int TileSize, typename Handle, typename View>
__global__ void sorted_cooperative_probe_kernel(Handle root, View view, const int* keys,
                                                int num_keys, unsigned long long* total_matches) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(cta);

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tile_id = global_tid / TileSize;
  int num_tiles = (gridDim.x * blockDim.x) / TileSize;

  for (int k = global_tile_id; k < num_keys; k += num_tiles) {
    int key = keys[k];
    auto handle = root.prefix(key, tile, view);

    if (tile.thread_rank() == 0) {
      unsigned long long count = 0;
      if (handle.valid())
        count = handle.degree();
      atomicAdd(total_matches, count);
    }
  }
}

// 3-Arity Cooperative
template <int TileSize, typename Handle>
__global__ void tvjoin_cooperative_probe_kernel_3arity(Handle root, const int* keys, int num_keys,
                                                       unsigned long long* total_matches) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(cta);

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tile_id = global_tid / TileSize;
  int num_tiles = (gridDim.x * blockDim.x) / TileSize;

  for (int k = global_tile_id; k < num_keys; k += num_tiles) {
    int k1 = keys[3 * k];
    int k2 = keys[3 * k + 1];
    int k3 = keys[3 * k + 2];

    auto h1 = root.prefix(k1, tile, 0);
    auto h2 = h1.prefix(k2, tile, 0);
    auto h3 = h2.prefix(k3, tile, 0);

    if (tile.thread_rank() == 0) {
      unsigned long long count = 0;
      if (h3.valid())
        count = h3.degree();
      atomicAdd(total_matches, count);
    }
  }
}

template <int TileSize, typename Handle, typename View>
__global__ void sorted_cooperative_probe_kernel_3arity(Handle root, View view, const int* keys,
                                                       int num_keys,
                                                       unsigned long long* total_matches) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(cta);

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tile_id = global_tid / TileSize;
  int num_tiles = (gridDim.x * blockDim.x) / TileSize;

  for (int k = global_tile_id; k < num_keys; k += num_tiles) {
    int k1 = keys[3 * k];
    int k2 = keys[3 * k + 1];
    int k3 = keys[3 * k + 2];

    auto h1 = root.prefix(k1, tile, view);
    auto h2 = h1.prefix(k2, tile, view);
    auto h3 = h2.prefix(k3, tile, view);

    if (tile.thread_rank() == 0) {
      unsigned long long count = 0;
      if (h3.valid())
        count = h3.degree();
      atomicAdd(total_matches, count);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// BENCHMARKS
// ------------------------------------------------------------------------------------------------

static void BM_TVJoin_Append(benchmark::State& state) {
  size_t n = state.range(0);
  AoSDeviceArray<int, 2> aos_data(n);  // Pre-allocate

  // Warmup / Generate once to ensure kernel compiles
  generate_flat_kernel<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  cudaDeviceSynchronize();

  for (auto _ : state) {
    state.PauseTiming();
    // Regenerate to be fair (data generation is typically NOT part of index *ingest* time,
    // but here we are measuring "Raw Data on Device -> Index").
    // Actually, usually we assume data exists.
    // But TVJoin `append` "swaps" or "copies".
    // My previous implem of `append` takes `AoSDeviceArray` and SWAPS it.
    // So we MUST generate FRESH data every time because `append` CONSUMES the buffer (if swap
    // used)! TVJoin adapter uses `swap` to take ownership. So we re-generate.

    // Re-allocate if consumed?
    // AoSDeviceArray internal vector is swapped out.
    // So aos_data is empty after append.
    aos_data.resize(n);  // Restore size
    generate_flat_kernel<<<(n + 255) / 256, 256>>>(n, aos_data.data());
    cudaDeviceSynchronize();

    state.ResumeTiming();

    DeviceTVJoinIndex<SR, int, 2, 0, 1> index;
    index.append(aos_data);  // Zero copy (swap)

    cudaDeviceSynchronize();
  }
}

static void BM_SortedArray_Build(benchmark::State& state) {
  size_t n = state.range(0);

  // NDDeviceArray for columns
  NDDeviceArray<uint32_t, 2> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);

  // Wrapper for spans to use build_from_encoded_device
  // Wait, build_from_encoded_device performs Sort IN PLACE or OUT OF PLACE?
  // It usually sorts indices or data.
  // DeviceSortedArrayIndex internally stores `NDDeviceArray`.
  // It TAKES `NDDeviceArray` by reference/move?
  // Let's check signature: `void build_from_encoded_device(spec, NDDeviceArray& cols, DeviceArray&
  // prov)` It likely consumes or copies. If it *consumes* or modifies, we need to regenerate.

  for (auto _ : state) {
    state.PauseTiming();
    // Regenerate
    cols.resize(n);        // Ensure size
    provenance.resize(n);  // Dummy prov
    generate_cols_kernel<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>());
    cudaDeviceSynchronize();
    state.ResumeTiming();

    DeviceSortedArrayIndex<SR, std::tuple<int, int>> index;
    // This function builds index on device, presumably sorting `cols`
    index.build_from_encoded_device(spec, cols, provenance);

    cudaDeviceSynchronize();
  }
}

static void BM_TVJoin_Probe(benchmark::State& state) {
  size_t n = state.range(0);

  // Prepare Index
  AoSDeviceArray<int, 2> aos_data(n);
  generate_flat_kernel<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  DeviceTVJoinIndex<SR, int, 2, 0, 1> index;
  index.append(aos_data);
  cudaDeviceSynchronize();

  // Prepare Keys
  // We want keys that exist. We generated `h % (n/10)` for col0.
  // Let's generate probe keys [0 ... n/10]
  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes, rmm::cuda_stream_default);
  // Fill with pattern matching generation: 0, 1, ...
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // Modulo n/10 to match domain
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_keys.begin(),
                    [limit = n / 10] __device__(int x) { return x % limit; });

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);

  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    tvjoin_probe_kernel<<<blocks, threads>>>(index.root(), d_keys.data(), num_probes,
                                             d_matches.data());
    cudaDeviceSynchronize();
  }
}

static void BM_SortedArray_Probe(benchmark::State& state) {
  size_t n = state.range(0);

  // Prepare Index
  NDDeviceArray<uint32_t, 2> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>());

  DeviceSortedArrayIndex<SR, std::tuple<int, int>> index;
  index.build_from_encoded_device(spec, cols, provenance);
  cudaDeviceSynchronize();

  // Probe Keys
  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes, rmm::cuda_stream_default);
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_keys.begin(),
                    [limit = n / 10] __device__(int x) { return x % limit; });

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    sorted_probe_kernel<<<blocks, threads>>>(index.root(), index.view(), d_keys.data(), num_probes,
                                             d_matches.data());
    cudaDeviceSynchronize();
  }
}

const Spec spec3 = {0, 1, 2};

__global__ void generate_probes_kernel3(size_t n_data, size_t n_probes, int* data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_probes) {
    GenerateFlat3{n_data}(tid, data);
  }
}

// Helper for Cooperative Benchmarks
template <int TileSize>
void BM_TVJoin_Probe_Cooperative_Templated(benchmark::State& state) {
  size_t n = state.range(0);
  // Prepare Index
  AoSDeviceArray<int, 2> aos_data(n);
  generate_flat_kernel<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  DeviceTVJoinIndex<SR, int, 2, 0, 1> index;
  index.append(aos_data);
  cudaDeviceSynchronize();

  // Prepare Keys
  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes, rmm::cuda_stream_default);
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_keys.begin(),
                    [limit = n / 10] __device__(int x) { return x % limit; });

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);

  // Launch Config: 1 tile per key
  int threads = 256;
  int tiles_per_block = threads / TileSize;
  int blocks = (num_probes + tiles_per_block - 1) / tiles_per_block;
  blocks = std::min(blocks, 640 * 2);

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    tvjoin_cooperative_probe_kernel<TileSize>
        <<<blocks, threads>>>(index.root(), d_keys.data(), num_probes, d_matches.data());
    cudaDeviceSynchronize();
  }
}

template <int TileSize>
void BM_SortedArray_Probe_Cooperative_Templated(benchmark::State& state) {
  size_t n = state.range(0);
  NDDeviceArray<uint32_t, 2> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>());
  DeviceSortedArrayIndex<SR, std::tuple<int, int>> index;
  index.build_from_encoded_device(spec, cols, provenance);
  cudaDeviceSynchronize();

  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes, rmm::cuda_stream_default);
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_keys.begin(),
                    [limit = n / 10] __device__(int x) { return x % limit; });

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int tiles_per_block = threads / TileSize;
  int blocks = (num_probes + tiles_per_block - 1) / tiles_per_block;
  blocks = std::min(blocks, 640 * 2);

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    sorted_cooperative_probe_kernel<TileSize><<<blocks, threads>>>(
        index.root(), index.view(), d_keys.data(), num_probes, d_matches.data());
    cudaDeviceSynchronize();
  }
}

template <int TileSize>
void BM_TVJoin_Probe_3Arity_Cooperative_Templated(benchmark::State& state) {
  size_t n = state.range(0);
  AoSDeviceArray<int, 3> aos_data(n);
  generate_flat_kernel3<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  DeviceTVJoinIndex<SR, int, 3, 0, 1, 2> index;
  index.append(aos_data);
  cudaDeviceSynchronize();

  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int tiles_per_block = threads / TileSize;
  int blocks = (num_probes + tiles_per_block - 1) / tiles_per_block;
  blocks = std::min(blocks, 640 * 2);

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    tvjoin_cooperative_probe_kernel_3arity<TileSize>
        <<<blocks, threads>>>(index.root(), d_keys.data(), num_probes, d_matches.data());
    cudaDeviceSynchronize();
  }
}

template <int TileSize>
void BM_SortedArray_Probe_3Arity_Cooperative_Templated(benchmark::State& state) {
  size_t n = state.range(0);
  NDDeviceArray<uint32_t, 3> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel3<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>(),
                                                  cols.column_ptr<2>());
  DeviceSortedArrayIndex<SR, std::tuple<int, int, int>> index;
  index.build_from_encoded_device(spec3, cols, provenance);
  cudaDeviceSynchronize();

  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int tiles_per_block = threads / TileSize;
  int blocks = (num_probes + tiles_per_block - 1) / tiles_per_block;
  blocks = std::min(blocks, 640 * 2);

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    sorted_cooperative_probe_kernel_3arity<TileSize><<<blocks, threads>>>(
        index.root(), index.view(), d_keys.data(), num_probes, d_matches.data());
    cudaDeviceSynchronize();
  }
}

static void BM_TVJoin_Probe_3Arity(benchmark::State& state) {
  size_t n = state.range(0);

  // Prepare Index (3 columns)
  // Use generate_flat_kernel3
  AoSDeviceArray<int, 3> aos_data(n);
  generate_flat_kernel3<<<(n + 255) / 256, 256>>>(n, aos_data.data());

  // TVJoin Index 3-arity
  DeviceTVJoinIndex<SR, int, 3, 0, 1, 2> index;
  index.append(aos_data);
  cudaDeviceSynchronize();

  // Prepare Keys (3 columns flat)
  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);

  // Use generate_probes_kernel3 to ensure keys match data distribution (guarantee hits)
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    tvjoin_probe_kernel_3arity<<<blocks, threads>>>(index.root(), d_keys.data(), num_probes,
                                                    d_matches.data());
    cudaDeviceSynchronize();
  }
}

static void BM_SortedArray_Probe_3Arity(benchmark::State& state) {
  size_t n = state.range(0);

  // Prepare Index
  NDDeviceArray<uint32_t, 3> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel3<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>(),
                                                  cols.column_ptr<2>());

  DeviceSortedArrayIndex<SR, std::tuple<int, int, int>> index;
  index.build_from_encoded_device(spec3, cols, provenance);
  cudaDeviceSynchronize();

  // Probe Keys
  size_t num_probes = 100000;
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);
  // Match data distribution
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  rmm::device_uvector<unsigned long long> d_matches(1, rmm::cuda_stream_default);
  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  for (auto _ : state) {
    cudaMemset(d_matches.data(), 0, sizeof(unsigned long long));
    sorted_probe_kernel_3arity<<<blocks, threads>>>(index.root(), index.view(), d_keys.data(),
                                                    num_probes, d_matches.data());
    cudaDeviceSynchronize();
  }
}

// ------------------------------------------------------------------------------------------------
// Define BENCHMARKS using macros to instantiate templates
// ------------------------------------------------------------------------------------------------

// TVJoin Cooperative
BENCHMARK_TEMPLATE(BM_TVJoin_Probe_Cooperative_Templated, 32)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_TVJoin_Probe_Cooperative_Templated, 16)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);

// SortedArray Cooperative
BENCHMARK_TEMPLATE(BM_SortedArray_Probe_Cooperative_Templated, 32)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_SortedArray_Probe_Cooperative_Templated, 16)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);

// 3-Arity Cooperative
BENCHMARK_TEMPLATE(BM_TVJoin_Probe_3Arity_Cooperative_Templated, 32)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_TVJoin_Probe_3Arity_Cooperative_Templated, 16)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_SortedArray_Probe_3Arity_Cooperative_Templated, 32)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_SortedArray_Probe_3Arity_Cooperative_Templated, 16)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Unit(benchmark::kMillisecond);

// Keep existing SCALAR benchmarks
BENCHMARK(BM_TVJoin_Append)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SortedArray_Build)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TVJoin_Probe)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SortedArray_Probe)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TVJoin_Probe_3Arity)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SortedArray_Probe_3Arity)->Arg(1000000)->Arg(10000000)->Unit(benchmark::kMillisecond);

// Verification for Cooperative 3-Arity (using Tile=16 to test sub-warp)
void verify_results_3arity_coop() {
  size_t n = 1000000;
  size_t num_probes = 100000;
  std::cout << "Verifying 3-Arity COOP(16) results with N=" << n << " Probes=" << num_probes
            << std::endl;

  // 1. Data Generation
  AoSDeviceArray<int, 3> aos_data(n);
  generate_flat_kernel3<<<(n + 255) / 256, 256>>>(n, aos_data.data());

  NDDeviceArray<uint32_t, 3> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel3<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>(),
                                                  cols.column_ptr<2>());

  // 2. Build Indices
  DeviceTVJoinIndex<SR, int, 3, 0, 1, 2> tv_index;
  tv_index.append(aos_data);

  DeviceSortedArrayIndex<SR, std::tuple<int, int, int>> sorted_index;
  sorted_index.build_from_encoded_device(spec3, cols, provenance);
  cudaDeviceSynchronize();

  // 3. Probe Generation
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  // 4. Run Cooperative Probe (Tile=16)
  rmm::device_uvector<unsigned long long> d_matches_tv(1, rmm::cuda_stream_default);
  rmm::device_uvector<unsigned long long> d_matches_sorted(1, rmm::cuda_stream_default);
  cudaMemset(d_matches_tv.data(), 0, sizeof(unsigned long long));
  cudaMemset(d_matches_sorted.data(), 0, sizeof(unsigned long long));

  int threads = 256;
  int tile_size = 16;
  int tiles_per_block = threads / tile_size;
  int blocks = (num_probes + tiles_per_block - 1) / tiles_per_block;

  tvjoin_cooperative_probe_kernel_3arity<16>
      <<<blocks, threads>>>(tv_index.root(), d_keys.data(), num_probes, d_matches_tv.data());
  sorted_cooperative_probe_kernel_3arity<16><<<blocks, threads>>>(
      sorted_index.root(), sorted_index.view(), d_keys.data(), num_probes, d_matches_sorted.data());
  cudaDeviceSynchronize();

  // 5. Check
  unsigned long long h_tv, h_sorted;
  cudaMemcpy(&h_tv, d_matches_tv.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sorted, d_matches_sorted.data(), sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  std::cout << "TVJoin COOP(16) Matches: " << h_tv << std::endl;
  std::cout << "Sorted COOP(16) Matches: " << h_sorted << std::endl;

  if (h_tv != h_sorted) {
    std::cout << "MISMATCH DETECTED!" << std::endl;
    exit(1);
  }
  std::cout << "Counts match." << std::endl;
}

void verify_results_3arity() {
  size_t n = 1000000;  // 1M
  size_t num_probes = 100000;
  std::cout << "Verifying 3-Arity results with N=" << n << " Probes=" << num_probes << std::endl;

  // 1. Setup TVJoin
  AoSDeviceArray<int, 3> aos_data(n);
  generate_flat_kernel3<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  cudaDeviceSynchronize();

  DeviceTVJoinIndex<SR, int, 3, 0, 1, 2> tv_index;
  tv_index.append(aos_data);
  cudaDeviceSynchronize();

  // 2. Setup SortedArray
  NDDeviceArray<uint32_t, 3> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel3<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>(),
                                                  cols.column_ptr<2>());
  cudaDeviceSynchronize();

  DeviceSortedArrayIndex<SR, std::tuple<int, int, int>> sorted_index;
  sorted_index.build_from_encoded_device(spec3, cols, provenance);
  cudaDeviceSynchronize();

  // 3. Setup Keys
  rmm::device_uvector<int> d_keys(num_probes * 3, rmm::cuda_stream_default);
  generate_probes_kernel3<<<(num_probes + 255) / 256, 256>>>(n, num_probes, d_keys.data());

  rmm::device_uvector<unsigned long long> tv_matches(1, rmm::cuda_stream_default);
  rmm::device_uvector<unsigned long long> sorted_matches(1, rmm::cuda_stream_default);
  cudaMemset(tv_matches.data(), 0, sizeof(unsigned long long));
  cudaMemset(sorted_matches.data(), 0, sizeof(unsigned long long));

  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  tvjoin_probe_kernel_3arity<<<blocks, threads>>>(tv_index.root(), d_keys.data(), num_probes,
                                                  tv_matches.data());

  sorted_probe_kernel_3arity<<<blocks, threads>>>(sorted_index.root(), sorted_index.view(),
                                                  d_keys.data(), num_probes, sorted_matches.data());

  cudaDeviceSynchronize();

  unsigned long long tv_count = 0;
  unsigned long long sorted_count = 0;
  cudaMemcpy(&tv_count, tv_matches.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sorted_count, sorted_matches.data(), sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  std::cout << "TVJoin Matches: " << tv_count << std::endl;
  std::cout << "Sorted Matches: " << sorted_count << std::endl;

  if (tv_count != sorted_count) {
    std::cout << "MISMATCH DETECTED!" << std::endl;
  } else {
    std::cout << "Counts match." << std::endl;
  }
}

void verify_results() {
  size_t n = 1000000;  // 1M
  size_t num_probes = 100000;
  std::cout << "Verifying results with N=" << n << " Probes=" << num_probes << std::endl;

  // 1. Setup TVJoin
  AoSDeviceArray<int, 2> aos_data(n);
  generate_flat_kernel<<<(n + 255) / 256, 256>>>(n, aos_data.data());
  cudaDeviceSynchronize();

  DeviceTVJoinIndex<SR, int, 2, 0, 1> tv_index;
  tv_index.append(aos_data);
  cudaDeviceSynchronize();

  // 2. Setup SortedArray
  NDDeviceArray<uint32_t, 2> cols;
  cols.resize(n);
  DeviceArray<semiring_value_t<SR>> provenance(n);
  generate_cols_kernel<<<(n + 255) / 256, 256>>>(n, cols.column_ptr<0>(), cols.column_ptr<1>());
  cudaDeviceSynchronize();

  DeviceSortedArrayIndex<SR, std::tuple<int, int>> sorted_index;
  sorted_index.build_from_encoded_device(spec, cols, provenance);
  cudaDeviceSynchronize();

  // 3. Setup Keys
  rmm::device_uvector<int> d_keys(num_probes, rmm::cuda_stream_default);
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_keys.begin(),
                    [limit = n / 10] __device__(int x) { return x % limit; });

  rmm::device_uvector<unsigned long long> tv_matches(1, rmm::cuda_stream_default);
  rmm::device_uvector<unsigned long long> sorted_matches(1, rmm::cuda_stream_default);
  cudaMemset(tv_matches.data(), 0, sizeof(unsigned long long));
  cudaMemset(sorted_matches.data(), 0, sizeof(unsigned long long));

  int threads = 256;
  int blocks = (num_probes + threads - 1) / threads;

  tvjoin_probe_kernel<<<blocks, threads>>>(tv_index.root(), d_keys.data(), num_probes,
                                           tv_matches.data());

  // Check Cooperative
  rmm::device_uvector<unsigned long long> cooperative_matches(1, rmm::cuda_stream_default);
  cudaMemset(cooperative_matches.data(), 0, sizeof(unsigned long long));

  // Cooperative Grid sizing
  int coop_threads = 256;
  int coop_blocks = (num_probes * 32 + coop_threads - 1) / coop_threads;
  tvjoin_cooperative_probe_kernel<32><<<coop_blocks, coop_threads>>>(
      tv_index.root(), d_keys.data(), num_probes, cooperative_matches.data());

  // Check SortedArray
  // Check if root is valid?
  if (sorted_index.size() == 0) {
    std::cout << "WARNING: SortedIndex size is 0!" << std::endl;
  }
  sorted_probe_kernel<<<blocks, threads>>>(sorted_index.root(), sorted_index.view(), d_keys.data(),
                                           num_probes, sorted_matches.data());

  cudaDeviceSynchronize();

  unsigned long long tv_count = 0;
  unsigned long long coop_count = 0;
  unsigned long long sorted_count = 0;
  cudaMemcpy(&tv_count, tv_matches.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&coop_count, cooperative_matches.data(), sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&sorted_count, sorted_matches.data(), sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  std::cout << "TVJoin Matches: " << tv_count << std::endl;
  std::cout << "Coop Matches:   " << coop_count << std::endl;
  std::cout << "Sorted Matches: " << sorted_count << std::endl;

  if (tv_count != sorted_count || coop_count != sorted_count) {
    std::cout << "MISMATCH DETECTED!" << std::endl;
  } else {
    std::cout << "Counts match." << std::endl;
  }
}

int main(int argc, char** argv) {
  SRDatalog::GPU::init_cuda();
  verify_results_3arity();

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::RunSpecifiedBenchmarks())
    return 0;
  return 0;
}
