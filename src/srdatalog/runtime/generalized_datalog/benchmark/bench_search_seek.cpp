/**
 * @file bench_search_seek.cpp
 * @brief Benchmark for GPU search seek performance: binary vs exponential search.
 *
 * Compares:
 *   - group_lower_bound (binary search)
 *   - group_linear_lower_bound (coalesced linear search)
 *
 * Usage:
 *   bench_search_seek <data_dir>
 *   where data_dir contains VarPointsTo.csv from batik_interned dataset
 *
 * Benchmark scenarios:
 *   1. Random keys across the full array
 *   2. Keys from the first 1% (best case for exp search)
 *   3. Keys from the last 1% (worst case for exp search)
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gpu/device_array.h"
#include "gpu/gpu_api.h"
#include "gpu/search.h"

using namespace SRDatalog::GPU;

// =============================================================================
// Data Loading
// =============================================================================

static std::vector<uint32_t> g_sorted_keys;  // Loaded VarPointsTo keys (heap column)
static DeviceArray<uint32_t>* g_device_keys = nullptr;
static std::string g_data_dir;

// Load VarPointsTo.csv and extract the first column (heap) as sorted keys
void load_var_points_to_data(const std::string& data_dir) {
  std::string filepath = data_dir + "/VarPointsTo.csv";
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open " << filepath << std::endl;
    exit(1);
  }

  std::cout << "Loading VarPointsTo data from: " << filepath << std::endl;

  std::string line;
  while (std::getline(file, line)) {
    // Tab-separated: heap<TAB>var
    std::istringstream iss(line);
    uint32_t heap;
    if (iss >> heap) {
      g_sorted_keys.push_back(heap);
    }
  }
  file.close();

  std::cout << "Loaded " << g_sorted_keys.size() << " tuples" << std::endl;

  // Sort the keys (maintains duplicates - we search on full sorted array)
  std::sort(g_sorted_keys.begin(), g_sorted_keys.end());

  std::cout << "Sorted keys count: " << g_sorted_keys.size() << " (~"
            << (g_sorted_keys.size() * sizeof(uint32_t) / (1024 * 1024)) << " MB)" << std::endl;

  // Copy to device
  g_device_keys = new DeviceArray<uint32_t>(g_sorted_keys.size());
  cudaMemcpy(g_device_keys->data(), g_sorted_keys.data(), g_sorted_keys.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
}

// =============================================================================
// CUDA Kernels for Benchmarking
// =============================================================================

template <bool UseLinearSearch>
__global__ void bench_lower_bound_kernel(const uint32_t* data, uint32_t len, const uint32_t* keys,
                                         uint32_t num_keys, uint32_t* results) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  // Persistent grid-stride loop for better load balancing with skewed keys
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t grid_stride = (gridDim.x * blockDim.x) / 32;

  for (uint32_t i = warp_id; i < num_keys; i += grid_stride) {
    uint32_t key = keys[i];
    uint32_t result;

    if constexpr (UseLinearSearch) {
      result = group_linear_lower_bound<uint32_t, uint32_t>(data, len, key, tile);
    } else {
      result = group_lower_bound<uint32_t, uint32_t>(data, len, key, tile);
    }

    // Only lane 0 writes the result
    if (tile.thread_rank() == 0) {
      results[i] = result;
    }
  }
}

// Kernel for small fixed-range search benchmark
enum class SearchType { Binary, Linear, Exponential };

template <SearchType ST>
__global__ void bench_small_range_kernel(const uint32_t* data, uint32_t range_size,
                                         const uint32_t* start_offsets, const uint32_t* keys,
                                         uint32_t num_keys, uint32_t* results) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t grid_stride = (gridDim.x * blockDim.x) / 32;

  for (uint32_t i = warp_id; i < num_keys; i += grid_stride) {
    uint32_t offset = start_offsets[i];
    uint32_t key = keys[i];
    uint32_t result;

    // Search within small range starting at offset
    if constexpr (ST == SearchType::Linear) {
      result = group_linear_lower_bound<uint32_t, uint32_t>(data + offset, range_size, key, tile);
    } else if constexpr (ST == SearchType::Exponential) {
      result = group_exp_lower_bound<uint32_t, uint32_t>(data + offset, range_size, key, tile);
    } else {
      result = group_lower_bound<uint32_t, uint32_t>(data + offset, range_size, key, tile);
    }

    if (tile.thread_rank() == 0) {
      results[i] = offset + result;
    }
  }
}

// =============================================================================
// Benchmark Helper
// =============================================================================

enum class KeyDistribution {
  Random,     // Keys uniformly distributed across the array
  FirstOne,   // Keys from the first 1% of the array (best case for exp search)
  LastOne,    // Keys from the last 1% of the array (worst case for exp search)
  Sequential  // Simulate sequential access pattern
};

template <bool UseLinearSearch>
void BM_LowerBound(benchmark::State& state, KeyDistribution dist) {
  if (g_sorted_keys.empty()) {
    state.SkipWithError("Data not loaded");
    return;
  }

  const size_t num_searches = static_cast<size_t>(state.range(0));
  const size_t data_len = g_sorted_keys.size();

  // Generate search keys based on distribution
  std::vector<uint32_t> search_keys(num_searches);
  std::mt19937 rng(42);

  switch (dist) {
    case KeyDistribution::Random: {
      std::uniform_int_distribution<size_t> dist(0, data_len - 1);
      for (size_t i = 0; i < num_searches; ++i) {
        search_keys[i] = g_sorted_keys[dist(rng)];
      }
      break;
    }
    case KeyDistribution::FirstOne: {
      size_t range_end = std::max(data_len / 100, size_t(1));
      std::uniform_int_distribution<size_t> dist(0, range_end - 1);
      for (size_t i = 0; i < num_searches; ++i) {
        search_keys[i] = g_sorted_keys[dist(rng)];
      }
      break;
    }
    case KeyDistribution::LastOne: {
      size_t range_start = data_len - std::max(data_len / 100, size_t(1));
      std::uniform_int_distribution<size_t> dist(range_start, data_len - 1);
      for (size_t i = 0; i < num_searches; ++i) {
        search_keys[i] = g_sorted_keys[dist(rng)];
      }
      break;
    }
    case KeyDistribution::Sequential: {
      // Sequential access: keys[i], keys[i+1], keys[i+2], ...
      size_t start = rng() % (data_len - num_searches);
      for (size_t i = 0; i < num_searches; ++i) {
        search_keys[i] = g_sorted_keys[start + i];
      }
      break;
    }
  }

  // Allocate device memory for keys and results
  DeviceArray<uint32_t> d_keys(num_searches);
  DeviceArray<uint32_t> d_results(num_searches);
  cudaMemcpy(d_keys.data(), search_keys.data(), num_searches * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  // Launch configuration: 1 warp per key
  const int threads_per_block = 256;
  const int warps_per_block = threads_per_block / 32;
  const int num_blocks = (num_searches + warps_per_block - 1) / warps_per_block;

  // Warmup
  bench_lower_bound_kernel<UseLinearSearch><<<num_blocks, threads_per_block>>>(
      g_device_keys->data(), static_cast<uint32_t>(data_len), d_keys.data(),
      static_cast<uint32_t>(num_searches), d_results.data());
  cudaDeviceSynchronize();

  // Benchmark loop
  for (auto _ : state) {
    bench_lower_bound_kernel<UseLinearSearch><<<num_blocks, threads_per_block>>>(
        g_device_keys->data(), static_cast<uint32_t>(data_len), d_keys.data(),
        static_cast<uint32_t>(num_searches), d_results.data());
    cudaDeviceSynchronize();
  }

  // Report metrics
  state.SetItemsProcessed(state.iterations() * num_searches);
  state.counters["searches_per_sec"] =
      benchmark::Counter(num_searches, benchmark::Counter::kIsRate);
}

// =============================================================================
// Benchmark Registration
// =============================================================================

// Binary Search Benchmarks
static void BM_Binary_Random(benchmark::State& state) {
  BM_LowerBound<false>(state, KeyDistribution::Random);
}
static void BM_Binary_First1Pct(benchmark::State& state) {
  BM_LowerBound<false>(state, KeyDistribution::FirstOne);
}
static void BM_Binary_Last1Pct(benchmark::State& state) {
  BM_LowerBound<false>(state, KeyDistribution::LastOne);
}
static void BM_Binary_Sequential(benchmark::State& state) {
  BM_LowerBound<false>(state, KeyDistribution::Sequential);
}

// Linear Search Benchmarks
static void BM_Linear_Random(benchmark::State& state) {
  BM_LowerBound<true>(state, KeyDistribution::Random);
}
static void BM_Linear_First1Pct(benchmark::State& state) {
  BM_LowerBound<true>(state, KeyDistribution::FirstOne);
}
static void BM_Linear_Last1Pct(benchmark::State& state) {
  BM_LowerBound<true>(state, KeyDistribution::LastOne);
}
static void BM_Linear_Sequential(benchmark::State& state) {
  BM_LowerBound<true>(state, KeyDistribution::Sequential);
}

// Register benchmarks with different search counts
BENCHMARK(BM_Binary_Random)->Arg(10000)->Arg(100000)->Arg(1000000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear_Random)->Arg(10000)->Arg(100000)->Arg(1000000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Binary_First1Pct)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear_First1Pct)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Binary_Last1Pct)->Arg(10000)->Arg(100000)->Arg(1000000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear_Last1Pct)->Arg(10000)->Arg(100000)->Arg(1000000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Binary_Sequential)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear_Sequential)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

// =============================================================================
// Small Range Benchmarks (to find break-even point)
// =============================================================================

template <SearchType ST>
void BM_SmallRange(benchmark::State& state) {
  if (g_sorted_keys.empty()) {
    state.SkipWithError("Data not loaded");
    return;
  }

  const size_t num_searches = 100000;
  const size_t range_size = static_cast<size_t>(state.range(0));  // 256, 512, 1024
  const size_t data_len = g_sorted_keys.size();

  if (range_size > data_len) {
    state.SkipWithError("Range size larger than data");
    return;
  }

  // Generate random start offsets and keys within each range
  std::vector<uint32_t> start_offsets(num_searches);
  std::vector<uint32_t> search_keys(num_searches);
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> offset_dist(0, data_len - range_size);

  for (size_t i = 0; i < num_searches; ++i) {
    size_t offset = offset_dist(rng);
    start_offsets[i] = static_cast<uint32_t>(offset);
    // Pick a key from within the range
    std::uniform_int_distribution<size_t> key_dist(offset, offset + range_size - 1);
    search_keys[i] = g_sorted_keys[key_dist(rng)];
  }

  // Allocate device memory
  DeviceArray<uint32_t> d_offsets(num_searches);
  DeviceArray<uint32_t> d_keys(num_searches);
  DeviceArray<uint32_t> d_results(num_searches);
  cudaMemcpy(d_offsets.data(), start_offsets.data(), num_searches * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_keys.data(), search_keys.data(), num_searches * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  const int threads_per_block = 256;
  const int warps_per_block = threads_per_block / 32;
  const int num_blocks = (num_searches + warps_per_block - 1) / warps_per_block;

  // Warmup
  bench_small_range_kernel<ST><<<num_blocks, threads_per_block>>>(
      g_device_keys->data(), static_cast<uint32_t>(range_size), d_offsets.data(), d_keys.data(),
      static_cast<uint32_t>(num_searches), d_results.data());
  cudaDeviceSynchronize();

  for (auto _ : state) {
    bench_small_range_kernel<ST><<<num_blocks, threads_per_block>>>(
        g_device_keys->data(), static_cast<uint32_t>(range_size), d_offsets.data(), d_keys.data(),
        static_cast<uint32_t>(num_searches), d_results.data());
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(state.iterations() * num_searches);
  state.counters["range_size"] = range_size;
}

static void BM_Binary_SmallRange(benchmark::State& state) {
  BM_SmallRange<SearchType::Binary>(state);
}
static void BM_Linear_SmallRange(benchmark::State& state) {
  BM_SmallRange<SearchType::Linear>(state);
}
static void BM_Exp_SmallRange(benchmark::State& state) {
  BM_SmallRange<SearchType::Exponential>(state);
}

// Test range sizes: 64, 128, 256, 512, 1024, 2048
BENCHMARK(BM_Binary_SmallRange)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Linear_SmallRange)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Exp_SmallRange)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
  // Parse data directory from command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <data_dir> [benchmark_args...]" << std::endl;
    std::cerr << "  data_dir: Path to batik_interned dataset containing VarPointsTo.csv"
              << std::endl;
    return 1;
  }

  g_data_dir = argv[1];

  // Load data (CUDA initializes implicitly on first API call)
  load_var_points_to_data(g_data_dir);

  // Remove data_dir from argv before passing to benchmark
  char* benchmark_argv[argc];
  benchmark_argv[0] = argv[0];
  for (int i = 2; i < argc; ++i) {
    benchmark_argv[i - 1] = argv[i];
  }
  int benchmark_argc = argc - 1;

  // Run benchmarks
  ::benchmark::Initialize(&benchmark_argc, benchmark_argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  delete g_device_keys;

  return 0;
}
