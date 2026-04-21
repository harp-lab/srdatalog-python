#define BOOST_TEST_MODULE gpu_semiring_atomic_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/device_array.h"
#include "semiring.h"
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <thrust/copy.h>
#include <vector>

using ::BooleanSR;
using ::NaturalBag;
using ::ProbIndep;
using SRDatalog::GPU::DeviceArray;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

template <typename T>
std::vector<T> device_to_host(const DeviceArray<T>& dev_array) {
  std::vector<T> host(dev_array.size());
  thrust::copy(dev_array.begin(), dev_array.end(), host.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return host;
}

template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return dev;
}

// ============================================================================
// CUDA Kernels for Atomic Operations
// ============================================================================

// Kernel: Multiple threads atomically add to the same location
template <typename SR>
__global__ void test_atomic_add_kernel(typename SR::value_type* result,
                                       const typename SR::value_type* values,
                                       uint32_t num_threads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_threads) {
    SR::atomic_add(result, values[tid]);
  }
}

// Kernel: Multiple threads atomically multiply to the same location
template <typename SR>
__global__ void test_atomic_mul_kernel(typename SR::value_type* result,
                                       const typename SR::value_type* values,
                                       uint32_t num_threads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_threads) {
    SR::atomic_mul(result, values[tid]);
  }
}

// Kernel: Multiple threads atomically add to different locations (stress test)
template <typename SR>
__global__ void test_atomic_add_multi_kernel(typename SR::value_type* results,
                                             uint32_t num_locations,
                                             const typename SR::value_type* values,
                                             const uint32_t* indices, uint32_t num_ops) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_ops) {
    uint32_t idx = indices[tid];
    if (idx < num_locations) {
      SR::atomic_add(results + idx, values[tid]);
    }
  }
}

}  // namespace

// ============================================================================
// Tests for NaturalBag (uint64_t)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_natural_bag_atomic_add_parallel) {
  // Test: 100 threads each add 1, result should be 100
  constexpr uint32_t num_threads = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<uint64_t> dev_result(1);
  dev_result[0] = 0;  // Initialize to zero

  std::vector<uint64_t> values(num_threads, 1);
  DeviceArray<uint64_t> dev_values = host_to_device(values);

  test_atomic_add_kernel<NaturalBag>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint64_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], num_threads);
}

BOOST_AUTO_TEST_CASE(test_natural_bag_atomic_add_large_values) {
  // Test: Multiple threads adding larger values
  constexpr uint32_t num_threads = 50;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<uint64_t> dev_result(1);
  dev_result[0] = 10;  // Start with 10

  std::vector<uint64_t> values(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i) {
    values[i] = i + 1;  // Values: 1, 2, 3, ..., 50
  }
  DeviceArray<uint64_t> dev_values = host_to_device(values);

  test_atomic_add_kernel<NaturalBag>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint64_t> result = device_to_host(dev_result);
  // Expected: 10 + sum(1..50) = 10 + 50*51/2 = 10 + 1275 = 1285
  uint64_t expected = 10ULL + (static_cast<uint64_t>(num_threads) * (num_threads + 1ULL)) / 2ULL;
  BOOST_REQUIRE_EQUAL(result[0], expected);
}

BOOST_AUTO_TEST_CASE(test_natural_bag_atomic_mul_parallel) {
  // Test: Multiple threads multiplying (should be commutative)
  // Start with 1, multiply by 2, 3, 4, 5
  constexpr uint32_t num_threads = 4;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<uint64_t> dev_result(1);
  dev_result[0] = 1;  // Start with 1

  std::vector<uint64_t> values = {2, 3, 4, 5};
  DeviceArray<uint64_t> dev_values = host_to_device(values);

  test_atomic_mul_kernel<NaturalBag>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint64_t> result = device_to_host(dev_result);
  // Expected: 1 * 2 * 3 * 4 * 5 = 120
  BOOST_REQUIRE_EQUAL(result[0], 120u);
}

BOOST_AUTO_TEST_CASE(test_natural_bag_atomic_add_multi_location) {
  // Test: Multiple threads updating different locations
  constexpr uint32_t num_locations = 10;
  constexpr uint32_t num_ops = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_ops + threads_per_block - 1) / threads_per_block;

  DeviceArray<uint64_t> dev_results(num_locations);
  // Initialize all to 0
  for (uint32_t i = 0; i < num_locations; ++i) {
    dev_results[i] = 0;
  }

  std::vector<uint64_t> values(num_ops, 1);
  std::vector<uint32_t> indices(num_ops);
  for (uint32_t i = 0; i < num_ops; ++i) {
    indices[i] = i % num_locations;  // Round-robin assignment
  }

  DeviceArray<uint64_t> dev_values = host_to_device(values);
  DeviceArray<uint32_t> dev_indices = host_to_device(indices);

  test_atomic_add_multi_kernel<NaturalBag><<<num_blocks, threads_per_block>>>(
      dev_results.data(), num_locations, dev_values.data(), dev_indices.data(), num_ops);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint64_t> results = device_to_host(dev_results);
  // Each location should have 10 additions (100 ops / 10 locations)
  for (uint32_t i = 0; i < num_locations; ++i) {
    BOOST_REQUIRE_EQUAL(results[i], 10u);
  }
}

// ============================================================================
// Tests for BooleanSR (bool)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_boolean_sr_atomic_add_parallel) {
  // Test: Multiple threads ORing values
  // If any thread sets true, result should be true
  constexpr uint32_t num_threads = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<bool> dev_result(1);
  dev_result[0] = false;  // Start with false

  std::vector<bool> values(num_threads, false);
  values[50] = true;  // One thread sets true
  DeviceArray<bool> dev_values = host_to_device(values);

  test_atomic_add_kernel<BooleanSR>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], true);  // OR of all values should be true
}

BOOST_AUTO_TEST_CASE(test_boolean_sr_atomic_add_all_false) {
  // Test: All threads OR false, result should be false
  constexpr uint32_t num_threads = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<bool> dev_result(1);
  dev_result[0] = false;

  std::vector<bool> values(num_threads, false);
  DeviceArray<bool> dev_values = host_to_device(values);

  test_atomic_add_kernel<BooleanSR>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], false);
}

BOOST_AUTO_TEST_CASE(test_boolean_sr_atomic_mul_parallel) {
  // Test: Multiple threads ANDing values
  // If any thread sets false, result should be false
  constexpr uint32_t num_threads = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<bool> dev_result(1);
  dev_result[0] = true;  // Start with true

  std::vector<bool> values(num_threads, true);
  values[50] = false;  // One thread sets false
  DeviceArray<bool> dev_values = host_to_device(values);

  test_atomic_mul_kernel<BooleanSR>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], false);  // AND of all values should be false
}

BOOST_AUTO_TEST_CASE(test_boolean_sr_atomic_mul_all_true) {
  // Test: All threads AND true, result should be true
  constexpr uint32_t num_threads = 100;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<bool> dev_result(1);
  dev_result[0] = true;

  std::vector<bool> values(num_threads, true);
  DeviceArray<bool> dev_values = host_to_device(values);

  test_atomic_mul_kernel<BooleanSR>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], true);
}

// ============================================================================
// Tests for ProbIndep (double)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_prob_indep_atomic_add_parallel) {
  // Test: Multiple threads computing probabilistic OR
  // P(A OR B) = 1 - (1-P(A)) * (1-P(B))
  constexpr uint32_t num_threads = 10;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<double> dev_result(1);
  dev_result[0] = 0.0;  // Start with 0.0 (P(false))

  // Each thread adds probability 0.1
  std::vector<double> values(num_threads, 0.1);
  DeviceArray<double> dev_values = host_to_device(values);

  test_atomic_add_kernel<ProbIndep>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<double> result = device_to_host(dev_result);
  // Expected: 1 - (1-0.1)^10 = 1 - 0.9^10 ≈ 0.6513
  // Allow some tolerance for floating point
  double expected = 1.0 - std::pow(0.9, static_cast<double>(num_threads));
  BOOST_CHECK_CLOSE(result[0], expected, 0.1);        // 0.1% tolerance
  BOOST_CHECK(result[0] >= 0.0 && result[0] <= 1.0);  // Should be clamped
}

BOOST_AUTO_TEST_CASE(test_prob_indep_atomic_mul_parallel) {
  // Test: Multiple threads computing probabilistic AND
  // P(A AND B) = P(A) * P(B)
  constexpr uint32_t num_threads = 5;
  constexpr uint32_t threads_per_block = 32;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<double> dev_result(1);
  dev_result[0] = 1.0;  // Start with 1.0 (P(true))

  // Each thread multiplies by 0.5
  std::vector<double> values(num_threads, 0.5);
  DeviceArray<double> dev_values = host_to_device(values);

  test_atomic_mul_kernel<ProbIndep>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<double> result = device_to_host(dev_result);
  // Expected: 1.0 * 0.5^5 = 0.03125
  double expected = std::pow(0.5, static_cast<double>(num_threads));
  BOOST_CHECK_CLOSE(result[0], expected, 0.1);        // 0.1% tolerance
  BOOST_CHECK(result[0] >= 0.0 && result[0] <= 1.0);  // Should be clamped
}

BOOST_AUTO_TEST_CASE(test_prob_indep_atomic_add_high_contention) {
  // Test: High contention - many threads updating same location
  constexpr uint32_t num_threads = 1000;
  constexpr uint32_t threads_per_block = 256;
  constexpr uint32_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  DeviceArray<double> dev_result(1);
  dev_result[0] = 0.0;

  // Each thread adds a small probability
  std::vector<double> values(num_threads, 0.001);
  DeviceArray<double> dev_values = host_to_device(values);

  test_atomic_add_kernel<ProbIndep>
      <<<num_blocks, threads_per_block>>>(dev_result.data(), dev_values.data(), num_threads);
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<double> result = device_to_host(dev_result);
  // Expected: 1 - (1-0.001)^1000 ≈ 0.632
  double expected = 1.0 - std::pow(0.999, static_cast<double>(num_threads));
  BOOST_CHECK_CLOSE(result[0], expected, 1.0);  // 1% tolerance for high contention
  BOOST_CHECK(result[0] >= 0.0 && result[0] <= 1.0);
}
