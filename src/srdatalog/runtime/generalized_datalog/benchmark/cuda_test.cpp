// Configure Thrust to use TBB backend for host-side operations
// This must be defined BEFORE including Thrust headers
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include <benchmark/benchmark.h>
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)

#include <chrono>  // Required for manual timing
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/system/tbb/execution_policy.h>

#include <boost/sort/parallel_stable_sort/parallel_stable_sort.hpp>
#include <tbb/parallel_sort.h>

// IPS²Ra is CPU-only, include only for host code (not CUDA device)
#ifndef __CUDA_ARCH__
// IPS²Ra parallel namespace requires _REENTRANT or _OPENMP to be defined
#ifndef _REENTRANT
#define _REENTRANT
#endif
// Suppress template syntax warning for IPS²Ra (compiler compatibility issue)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"
#include <ips2ra.hpp>  // IPS²Ra main header
#pragma clang diagnostic pop
#endif

#include "gpu/device_array.h"
#include "gpu/rand.h"
#include "sort.h"
#include "system.h"

// This file benchmarks CUDA Thrust sorting and Highway CPU sorting for comparison

using SRDatalog::make_aligned_vector;
using SRDatalog::GPU::DeviceArray;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
namespace config {
constexpr size_t seed = 42;
}  // namespace config

// Global CUDA context initialization (thread-safe)
static std::once_flag g_cuda_init_flag;

// -----------------------------------------------------------------------------
// Setup & Teardown
// -----------------------------------------------------------------------------
static void InitCuda() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << ">> Device: " << prop.name << "\n";
}

// -----------------------------------------------------------------------------
// Data Generation
// -----------------------------------------------------------------------------
__device__ inline uint64_t gpu_rand_device(uint64_t index, uint64_t global_seed) {
  uint64_t x = index + global_seed;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * 0x2545F4914F6CDD1DULL;
}

__device__ inline uint32_t gpu_rand_device(uint32_t index, uint32_t global_seed) {
  return static_cast<uint32_t>(
      gpu_rand_device(static_cast<uint64_t>(index), static_cast<uint64_t>(global_seed)));
}

__global__ void generate_random_kernel(int* data, size_t n, size_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] =
        static_cast<int>(gpu_rand_device(static_cast<uint32_t>(idx), static_cast<uint32_t>(seed)));
  }
}

static void generate_random_gpu(int* data, size_t n, size_t seed_offset) {
  std::call_once(g_cuda_init_flag, InitCuda);

  size_t final_seed = config::seed + seed_offset;

  const int threads_per_block = 256;
  const int blocks = (n + threads_per_block - 1) / threads_per_block;

  generate_random_kernel<<<blocks, threads_per_block>>>(data, n, final_seed);
  GPU_DEVICE_SYNCHRONIZE();
}

static void generate_random_cpu(auto& data, size_t seed_offset) {
  size_t final_seed = config::seed + seed_offset;
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] =
        static_cast<int>(gpu_rand(static_cast<uint32_t>(i), static_cast<uint32_t>(final_seed)));
  }
}

// -----------------------------------------------------------------------------
// Sorting Kernels
// -----------------------------------------------------------------------------
namespace kernels {

void sort_thrust(DeviceArray<int>& data) {
  thrust::sort(data.begin(), data.end());
}

void sort_thrust_stable(DeviceArray<int>& data) {
  thrust::stable_sort(data.begin(), data.end());
}

template <typename VectorType>
void sort_highway(VectorType& data) {
  hwy_lexsort::unstable_sort(data.data(), data.size());
}

template <typename VectorType>
void sort_boost_parallel_stable(VectorType& data) {
  boost::sort::parallel_stable_sort(data.begin(), data.end());
}

template <typename VectorType>
void sort_tbb_parallel(VectorType& data) {
  tbb::parallel_sort(data.begin(), data.end());
}

template <typename VectorType>
void sort_thrust_host_tbb(VectorType& data) {
  thrust::sort(thrust::tbb::par, data.begin(), data.end());
}

template <typename VectorType>
void sort_thrust_host_stable_tbb(VectorType& data) {
  thrust::stable_sort(thrust::tbb::par, data.begin(), data.end());
}

template <typename VectorType>
void sort_thrust_host_merge_tbb(VectorType& data) {
  thrust::stable_sort(thrust::tbb::par, data.begin(), data.end());
}

template <typename VectorType>
void sort_thrust_host_merge_stable_tbb(VectorType& data) {
  thrust::stable_sort(thrust::tbb::par, data.begin(), data.end());
}

// IPS²Ra - CPU-based parallel radix sort (host-only)
#ifndef __CUDA_ARCH__
template <typename VectorType>
void sort_ips2ra(VectorType& data) {
  ips2ra::parallel::sort(data.begin(), data.end());
}
#endif
}  // namespace kernels

// -----------------------------------------------------------------------------
// Benchmarks using Manual Timing (SetIterationTime)
// -----------------------------------------------------------------------------

static void BM_Sort_CUDA_Thrust(benchmark::State& state) {
  std::call_once(g_cuda_init_flag, InitCuda);
  const size_t size = static_cast<size_t>(state.range(0));
  DeviceArray<int> data(size);

  for (auto _ : state) {
    // Setup (Heavy, untimed)
    generate_random_gpu(data.data(), data.size(), 0);
    GPU_DEVICE_SYNCHRONIZE();  // Ensure generation finished

    // Manual Timer Start
    auto start = std::chrono::high_resolution_clock::now();

    kernels::sort_thrust(data);
    GPU_DEVICE_SYNCHRONIZE();  // Ensure kernel finished

    // Manual Timer Stop
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    // Report precise GPU time
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_CUDA_Thrust_Stable(benchmark::State& state) {
  std::call_once(g_cuda_init_flag, InitCuda);
  const size_t size = static_cast<size_t>(state.range(0));
  DeviceArray<int> data(size);

  for (auto _ : state) {
    generate_random_gpu(data.data(), data.size(), 0);
    GPU_DEVICE_SYNCHRONIZE();

    auto start = std::chrono::high_resolution_clock::now();

    kernels::sort_thrust_stable(data);
    GPU_DEVICE_SYNCHRONIZE();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Highway_VQSort(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();

    kernels::sort_highway(cpu_data);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Boost_ParallelStable(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_boost_parallel_stable(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_TBB_Parallel(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_tbb_parallel(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Thrust_Host_TBB(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_thrust_host_tbb(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Thrust_Host_Stable_TBB(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_thrust_host_stable_tbb(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Thrust_Host_Merge_TBB(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_thrust_host_merge_tbb(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_Sort_Thrust_Host_Merge_Stable_TBB(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  auto cpu_data = make_aligned_vector<int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_thrust_host_merge_stable_tbb(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

#ifndef __CUDA_ARCH__
static void BM_Sort_IPS2Ra(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  // IPS²Ra requires unsigned types for radix sort
  auto cpu_data = make_aligned_vector<unsigned int>(size);

  for (auto _ : state) {
    generate_random_cpu(cpu_data, 0);

    auto start = std::chrono::high_resolution_clock::now();
    kernels::sort_ips2ra(cpu_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}
#endif

// -----------------------------------------------------------------------------
// Register Benchmarks (Using Manual Time)
// -----------------------------------------------------------------------------
// NOTE: UseManualTime() is REQUIRED because we call SetIterationTime()

BENCHMARK(BM_Sort_CUDA_Thrust)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_CUDA_Thrust_Stable)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Highway_VQSort)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Boost_ParallelStable)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_TBB_Parallel)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Thrust_Host_TBB)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Thrust_Host_Stable_TBB)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Thrust_Host_Merge_TBB)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_Sort_Thrust_Host_Merge_Stable_TBB)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

#ifndef __CUDA_ARCH__
#ifndef __CUDA_ARCH__
BENCHMARK(BM_Sort_IPS2Ra)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();
#endif
#endif

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
BENCHMARK_MAIN();
