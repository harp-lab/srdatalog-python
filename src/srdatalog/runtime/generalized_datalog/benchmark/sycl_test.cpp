#include "gpu/rand.h"
#include "sort.h"
#include <AdaptiveCpp/algorithms/algorithm.hpp>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
namespace config {
constexpr size_t seed = 42;
constexpr size_t default_size = 10'000'000;
}  // namespace config

using SharedAlloc = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
using DataVector = std::vector<int, SharedAlloc>;
using CpuDataVector = std::vector<int>;  // CPU memory for Highway

// Global SYCL queue (initialized once, thread-safe)
static sycl::queue* g_queue = nullptr;
static std::once_flag g_queue_init_flag;

// -----------------------------------------------------------------------------
// Setup & Teardown
// -----------------------------------------------------------------------------
static void InitQueue() {
  g_queue = new sycl::queue(sycl::gpu_selector_v);
  // Print device info only once (thread-safe via call_once)
  std::cout << ">> Device: " << g_queue->get_device().get_info<sycl::info::device::name>() << "\n";
}

static sycl::queue& GetQueue() {
  std::call_once(g_queue_init_flag, InitQueue);
  return *g_queue;
}

// -----------------------------------------------------------------------------
// Data Generation
// -----------------------------------------------------------------------------
// Generate random data on GPU (for SYCL benchmark)
static void generate_random_gpu(sycl::queue& q, DataVector& data, size_t seed_offset) {
  const size_t N = data.size();
  int* raw_ptr = data.data();
  size_t final_seed = config::seed + seed_offset;

  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
     raw_ptr[i] =
         static_cast<int>(gpu_rand(static_cast<uint32_t>(i), static_cast<uint32_t>(final_seed)));
   }).wait();
}

// Generate random data on CPU (for Highway benchmark)
// Uses same gpu_rand function to generate identical data as GPU version
static void generate_random_cpu(CpuDataVector& data, size_t seed_offset) {
  size_t final_seed = config::seed + seed_offset;
  for (size_t i = 0; i < data.size(); ++i) {
    // Use same gpu_rand function for identical random sequence
    data[i] =
        static_cast<int>(gpu_rand(static_cast<uint32_t>(i), static_cast<uint32_t>(final_seed)));
  }
}

// -----------------------------------------------------------------------------
// Sorting Kernels
// -----------------------------------------------------------------------------
namespace kernels {

// 1. AdaptiveCpp stdpar (Standard C++ Parallel Algorithms) - GPU-based
// This is the "Modern C++" way. AdaptiveCpp intercepts this and runs it on GPU.
void sort_stdpar(DataVector& data, sycl::queue&) {
  std::sort(std::execution::par_unseq, data.begin(), data.end(), std::less<>());
}

// 2. Highway VQSort - CPU-based SIMD sorting
// Works directly on CPU memory (no GPU involved)
void sort_highway(CpuDataVector& data) {
  hwy_lexsort::unstable_sort(data.data(), data.size());
}
}  // namespace kernels

// -----------------------------------------------------------------------------
// Benchmark: SYCL std::sort (par_unseq) - GPU
// -----------------------------------------------------------------------------
static void BM_Sort_SYCL_ParUnseq(benchmark::State& state) {
  sycl::queue& q = GetQueue();
  const size_t size = static_cast<size_t>(state.range(0));

  // Setup: Allocate and generate data (excluded from timing)
  state.PauseTiming();
  DataVector data(size, SharedAlloc(q));
  generate_random_gpu(q, data, 0);
  state.ResumeTiming();

  // Set label before timing loop
  state.SetLabel("SYCL_par_unseq_GPU");

  for (auto _ : state) {
    kernels::sort_stdpar(data, q);
    q.wait();  // Ensure completion
    benchmark::DoNotOptimize(data.data());
  }

  // Verification (once, after all iterations)
  state.PauseTiming();
  using early_exit_flag_t = acpp::algorithms::detail::early_exit_flag_t;
  auto* result_ptr = sycl::malloc_shared<early_exit_flag_t>(1, q);
  *result_ptr = 0;
  acpp::algorithms::is_sorted(q, data.begin(), data.end(), result_ptr).wait();
  bool is_sorted = (*result_ptr == 1);
  sycl::free(result_ptr, q);

  if (!is_sorted) {
    state.SkipWithError("Array is not sorted!");
  }
  state.ResumeTiming();
}

// -----------------------------------------------------------------------------
// Benchmark: Highway VQSort - CPU
// -----------------------------------------------------------------------------
static void BM_Sort_Highway_VQSort(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));

  // Set label before timing
  state.SetLabel("Highway_VQSort_CPU");

  // Setup: Allocate and generate data on CPU (excluded from timing)
  state.PauseTiming();
  CpuDataVector data(size);
  generate_random_cpu(data, 0);
  state.ResumeTiming();

  for (auto _ : state) {
    kernels::sort_highway(data);
    benchmark::DoNotOptimize(data.data());
  }

  // Verification (once, after all iterations)
  state.PauseTiming();
  bool is_sorted = true;
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1] > data[i]) {
      is_sorted = false;
      break;
    }
  }

  if (!is_sorted) {
    state.SkipWithError("Array is not sorted!");
  }
  state.ResumeTiming();
}

// -----------------------------------------------------------------------------
// Register Benchmarks
// -----------------------------------------------------------------------------
// Register SYCL benchmark with multiple sizes
BENCHMARK(BM_Sort_SYCL_ParUnseq)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Register Highway benchmark with multiple sizes
BENCHMARK(BM_Sort_Highway_VQSort)
    ->Arg(5'000'000)
    ->Arg(20'000'000)
    ->Arg(100'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
BENCHMARK_MAIN();