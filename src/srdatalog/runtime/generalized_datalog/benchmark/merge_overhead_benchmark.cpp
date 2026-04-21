#include <benchmark/benchmark.h>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "gpu/deduplication.h"
#include "gpu/device_array.h"
#include "gpu/gpu_api.h"  // GPU API abstraction
#include "gpu/merge.h"
#include "gpu/nd_device_array.h"
#include "semiring.h"

using namespace SRDatalog::GPU;

// Define a simple semiring for testing
struct CountSR {
  using value_type = int;
  static __device__ __host__ int zero() {
    return 0;
  }
  static __device__ __host__ int one() {
    return 1;
  }
  static __device__ __host__ int add(int a, int b) {
    return a + b;
  }
  static __device__ __host__ int mul(int a, int b) {
    return a * b;
  }
};

// Helper to create test data
template <typename T>
void create_test_data(size_t size_a, size_t size_b, NDDeviceArray<T, 2>& cols_a,
                      DeviceArray<int>& prov_a, NDDeviceArray<T, 2>& cols_b,
                      DeviceArray<int>& prov_b, float overlap_ratio = 0.1f) {

  // Resize arrays
  cols_a.resize(size_a);
  prov_a.resize(size_a);
  cols_b.resize(size_b);
  prov_b.resize(size_b);

  // Fill provenance with 1s
  thrust::fill(thrust::device, prov_a.begin(), prov_a.end(), 1);
  thrust::fill(thrust::device, prov_b.begin(), prov_b.end(), 1);

  // Fill columns
  // A: 0, 2, 4, ...
  thrust::sequence(thrust::device, cols_a.template column_ptr<0>(),
                   cols_a.template column_ptr<0>() + size_a, 0, 2);
  thrust::fill(thrust::device, cols_a.template column_ptr<1>(),
               cols_a.template column_ptr<1>() + size_a,
               0);  // Col 1 is 0

  // B: overlap some range
  size_t overlap_count = size_b * overlap_ratio;
  size_t non_overlap = size_b - overlap_count;

  // Overlapping part: 0, 2, 4... (matches A)
  thrust::sequence(thrust::device, cols_b.template column_ptr<0>(),
                   cols_b.template column_ptr<0>() + overlap_count, 0, 2);

  // Non-overlapping part: 1, 3, 5... (distinct from A)
  thrust::sequence(thrust::device, cols_b.template column_ptr<0>() + overlap_count,
                   cols_b.template column_ptr<0>() + size_b, 1, 2);

  thrust::fill(thrust::device, cols_b.template column_ptr<1>(),
               cols_b.template column_ptr<1>() + size_b, 0);

  // Ensure B is sorted
  // (It is sorted by construction: 0, 2, ... then 1, 3, ... is NOT sorted globally)
  // We need B sorted.
  // Let's just make B: start from size_a * (1-overlap)
  // A: [0, size_a)
  // B: [size_a - overlap, size_a - overlap + size_b)

  thrust::sequence(thrust::device, cols_a.template column_ptr<0>(),
                   cols_a.template column_ptr<0>() + size_a);

  T start_b = size_a > overlap_count ? size_a - overlap_count : 0;
  thrust::sequence(thrust::device, cols_b.template column_ptr<0>(),
                   cols_b.template column_ptr<0>() + size_b, start_b);
}

static void Benchmark_Baseline(benchmark::State& state) {
  size_t size = state.range(0);

  NDDeviceArray<uint32_t, 2> cols_a, cols_b, cols_out;
  DeviceArray<int> prov_a, prov_b, prov_out;
  DeviceArray<uint32_t> csr_val;

  create_test_data(size, size, cols_a, prov_a, cols_b, prov_b, 0.1f);

  // Pre-allocate temp buffers to simulate reuse
  NDDeviceArray<uint32_t, 2> temp_cols;
  DeviceArray<int> temp_prov;
  temp_cols.resize(size * 2);
  temp_prov.resize(size * 2);

  for (auto _ : state) {
    // 1. Merge
    size_t out_size = size * 2;

    // Prepare iterators
    auto keys_a = thrust::make_zip_iterator(
        thrust::make_tuple(cols_a.template column_ptr<0>(), cols_a.template column_ptr<1>()));
    auto keys_b = thrust::make_zip_iterator(
        thrust::make_tuple(cols_b.template column_ptr<0>(), cols_b.template column_ptr<1>()));
    auto keys_out = thrust::make_zip_iterator(
        thrust::make_tuple(temp_cols.template column_ptr<0>(), temp_cols.template column_ptr<1>()));

    thrust::merge_by_key(thrust::device, keys_a, keys_a + size, keys_b, keys_b + size,
                         prov_a.begin(), prov_b.begin(), keys_out, temp_prov.begin());

    // 2. Dedup
    deduplicate_aggregate_and_build_csr<CountSR>(temp_cols, temp_prov, csr_val);

    GPU_DEVICE_SYNCHRONIZE();
  }

  // metric: Gelems/s (output elements / time)
  // Output size is roughly 2*size (minus overlap)
  state.SetItemsProcessed(state.iterations() * size * 2);
}

static void Benchmark_OptimizedFused(benchmark::State& state) {
  size_t size = state.range(0);

  NDDeviceArray<uint32_t, 2> cols_a, cols_b, cols_out;
  DeviceArray<int> prov_a, prov_b, prov_out;
  DeviceArray<uint32_t> csr_val;

  create_test_data(size, size, cols_a, prov_a, cols_b, prov_b, 0.1f);

  for (auto _ : state) {
    merge_deduplicate_aggregate<CountSR>(cols_a, prov_a, cols_b, prov_b, cols_out, prov_out,
                                         csr_val);
  }
  state.SetItemsProcessed(state.iterations() * size * 2);
}

static void Benchmark_ThrustMergeOnly(benchmark::State& state) {
  size_t size = state.range(0);

  NDDeviceArray<uint32_t, 2> cols_a, cols_b, cols_out;
  DeviceArray<int> prov_a, prov_b, prov_out;

  create_test_data(size, size, cols_a, prov_a, cols_b, prov_b, 0.1f);
  cols_out.resize(size * 2);
  prov_out.resize(size * 2);

  for (auto _ : state) {
    auto keys_a = thrust::make_zip_iterator(
        thrust::make_tuple(cols_a.template column_ptr<0>(), cols_a.template column_ptr<1>()));
    auto keys_b = thrust::make_zip_iterator(
        thrust::make_tuple(cols_b.template column_ptr<0>(), cols_b.template column_ptr<1>()));
    auto keys_out = thrust::make_zip_iterator(
        thrust::make_tuple(cols_out.template column_ptr<0>(), cols_out.template column_ptr<1>()));

    thrust::merge_by_key(thrust::device, keys_a, keys_a + size, keys_b, keys_b + size,
                         prov_a.begin(), prov_b.begin(), keys_out, prov_out.begin());
    GPU_DEVICE_SYNCHRONIZE();
  }
  state.SetItemsProcessed(state.iterations() * size * 2);
}

// Register benchmarks
BENCHMARK(Benchmark_Baseline)
    ->RangeMultiplier(4)
    ->Range(1 << 20, 1 << 24)
    ->Unit(benchmark::kMillisecond);  // 1M to 16M
BENCHMARK(Benchmark_OptimizedFused)
    ->RangeMultiplier(4)
    ->Range(1 << 20, 1 << 24)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(Benchmark_ThrustMergeOnly)
    ->RangeMultiplier(4)
    ->Range(1 << 20, 1 << 24)
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  // Use RMM pool for better perf
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr,
                                                                       1024 * 1024 * 1024);
  rmm::mr::set_current_device_resource(&pool_mr);

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
