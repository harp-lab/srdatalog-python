#define BOOST_TEST_MODULE gpu_search_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include "gpu/macro.h"    // For WARP_SIZE
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include "gpu/device_array.h"
#include "gpu/search.h"

using SRDatalog::GPU::DeviceArray;
using SRDatalog::GPU::group_lower_bound;
using SRDatalog::GPU::warp_contains;
using SRDatalog::GPU::warp_equal_range;
using SRDatalog::GPU::warp_find;
using SRDatalog::GPU::warp_lower_bound;
using SRDatalog::GPU::warp_upper_bound;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

// Helper: Copy device array to host vector
template <typename T>
std::vector<T> device_to_host(const DeviceArray<T>& dev_array) {
  std::vector<T> host(dev_array.size());
  thrust::copy(dev_array.begin(), dev_array.end(), host.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return host;
}

// Helper: Copy host vector to device array
template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return dev;
}

// Helper: CPU reference implementation for lower_bound
template <typename T>
uint32_t cpu_lower_bound(const T* data, uint32_t len, T key) {
  const T* first = data;
  const T* last = data + len;
  const T* it = std::lower_bound(first, last, key);
  return static_cast<uint32_t>(it - first);
}

// Helper: CPU reference implementation for upper_bound
template <typename T>
uint32_t cpu_upper_bound(const T* data, uint32_t len, T key) {
  const T* first = data;
  const T* last = data + len;
  const T* it = std::upper_bound(first, last, key);
  return static_cast<uint32_t>(it - first);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel to test warp_lower_bound
// All threads in warp must participate, but only thread 0 writes result
template <typename T, typename IndexType>
__global__ void test_warp_lower_bound_kernel(const T* data, IndexType len, T key,
                                             IndexType* result) {
  // All threads in warp participate in the search
  IndexType res = warp_lower_bound<T, IndexType>(data, len, key);
  // Only thread 0 writes the result
  if (threadIdx.x == 0) {
    result[0] = res;
  }
}

// Kernel to test warp_upper_bound
template <typename T, typename IndexType>
__global__ void test_warp_upper_bound_kernel(const T* data, IndexType len, T key,
                                             IndexType* result) {
  IndexType res = warp_upper_bound<T, IndexType>(data, len, key);
  if (threadIdx.x == 0) {
    result[0] = res;
  }
}

// Kernel to test warp_equal_range
template <typename T, typename IndexType>
__global__ void test_warp_equal_range_kernel(const T* data, IndexType len, T key,
                                             IndexType* result) {
  auto [lower, upper] = warp_equal_range<T, IndexType>(data, len, key);
  if (threadIdx.x == 0) {
    result[0] = lower;
    result[1] = upper;
  }
}

// Kernel to test warp_contains
template <typename T, typename IndexType>
__global__ void test_warp_contains_kernel(const T* data, IndexType len, T key, bool* result) {
  bool res = warp_contains<T, IndexType>(data, len, key);
  if (threadIdx.x == 0) {
    result[0] = res;
  }
}

// Kernel to test warp_find
template <typename T, typename IndexType>
__global__ void test_warp_find_kernel(const T* data, IndexType len, T key, IndexType* result) {
  IndexType res = warp_find<T, IndexType>(data, len, key);
  if (threadIdx.x == 0) {
    result[0] = res;
  }
}

// Kernel to test multiple keys in parallel (one per warp)
// Each warp processes one key - all threads in warp must participate
template <typename T, typename IndexType>
__global__ void test_warp_lower_bound_multi_kernel(const T* data, IndexType len, const T* keys,
                                                   uint32_t num_keys, IndexType* results) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

  // All threads in warp participate in the search
  // Only process if this warp has a valid key
  T key = (warp_id < num_keys) ? keys[warp_id] : 0;
  IndexType res = warp_lower_bound<T, IndexType>(data, len, key);

  // Only lane 0 writes the result
  if (warp_id < num_keys && lane == 0) {
    results[warp_id] = res;
  }
}

// Kernel to test group_lower_bound with arbitrary group sizes
// All threads in the group must participate
template <typename T, typename IndexType, int GROUP_SIZE>
__global__ void test_group_lower_bound_kernel(const T* data, IndexType len, T key,
                                              IndexType* result) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<GROUP_SIZE>(block);

  // All threads in the group participate in the search
  IndexType res = group_lower_bound<T, IndexType>(data, len, key, tile);

  // Only thread 0 in the tile writes the result
  if (tile.thread_rank() == 0) {
    *result = res;
  }
}

// ============================================================================
// Tests
// ============================================================================

// Test 1: Basic lower_bound - key exists
BOOST_AUTO_TEST_CASE(test_lower_bound_key_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 7;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);  // Index of 7
}

// Test 2: Basic lower_bound - key doesn't exist
BOOST_AUTO_TEST_CASE(test_lower_bound_key_not_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 6;  // Between 5 and 7
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);  // Should point to 7
}

// Test 3: Basic lower_bound - key smaller than all
BOOST_AUTO_TEST_CASE(test_lower_bound_key_smaller) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 0;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
}

// Test 4: Basic lower_bound - key larger than all
BOOST_AUTO_TEST_CASE(test_lower_bound_key_larger) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 100;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], host_data.size());
}

// Test 5: Basic upper_bound
BOOST_AUTO_TEST_CASE(test_upper_bound) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 7, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 7;
  test_warp_upper_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 6u);  // Points after all 7s
}

// Test 6: Basic equal_range
BOOST_AUTO_TEST_CASE(test_equal_range) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 7, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 7;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 3u);  // First 7
  BOOST_REQUIRE_EQUAL(result[1], 6u);  // After last 7
}

// Test 7: equal_range - key not found
BOOST_AUTO_TEST_CASE(test_equal_range_not_found) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 6;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], result[1]);  // Empty range
  BOOST_REQUIRE_EQUAL(result[0], 3u);         // Both point to insertion point
}

// Test 8: warp_contains - key exists
BOOST_AUTO_TEST_CASE(test_contains_key_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<bool> dev_result(1);

  uint32_t key = 7;
  test_warp_contains_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], true);
}

// Test 9: warp_contains - key doesn't exist
BOOST_AUTO_TEST_CASE(test_contains_key_not_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<bool> dev_result(1);

  uint32_t key = 6;
  test_warp_contains_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<bool> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], false);
}

// Test 10: warp_find - key exists
BOOST_AUTO_TEST_CASE(test_find_key_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 7;
  test_warp_find_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 3u);
}

// Test 11: warp_find - key doesn't exist
BOOST_AUTO_TEST_CASE(test_find_key_not_exists) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 6;
  test_warp_find_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], host_data.size());  // Returns len when not found
}

// Test 12: warp_lower_bound - small array (replaces linear_scan test)
BOOST_AUTO_TEST_CASE(test_lower_bound_small) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 5;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 2u);
}

// Test 13: Large array test
BOOST_AUTO_TEST_CASE(test_lower_bound_large_array) {
  const size_t n = 1000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);  // Even numbers: 0, 2, 4, 6, ...
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 500;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
}

// Test 14: Multiple keys in parallel
BOOST_AUTO_TEST_CASE(test_lower_bound_multi) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  std::vector<uint32_t> host_keys = {0, 5, 10, 15, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_keys = host_to_device(host_keys);
  DeviceArray<uint32_t> dev_results(host_keys.size());

// On ROCm with 64-thread warps, we need more threads to get enough warps
// 256 threads = 4 warps on ROCm (256/64), but we need at least 5 warps for 5 keys
#ifdef USE_ROCm
  const int threads_per_block = 320;  // 320/64 = 5 warps, enough for 5 keys
#else
  const int threads_per_block = 256;  // 256/32 = 8 warps on CUDA
#endif
  const int blocks = (host_keys.size() + threads_per_block - 1) / threads_per_block;
  test_warp_lower_bound_multi_kernel<uint32_t, uint32_t><<<blocks, threads_per_block>>>(
      dev_data.data(), host_data.size(), dev_keys.data(), host_keys.size(), dev_results.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> results = device_to_host(dev_results);
  for (size_t i = 0; i < host_keys.size(); ++i) {
    uint32_t expected =
        cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), host_keys[i]);
    BOOST_REQUIRE_EQUAL(results[i], expected);
  }
}

// Test 15: Edge case - empty array
BOOST_AUTO_TEST_CASE(test_lower_bound_empty) {
  std::vector<uint32_t> host_data;
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 5;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), 0, key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
}

// Test 16: Edge case - single element
BOOST_AUTO_TEST_CASE(test_lower_bound_single) {
  std::vector<uint32_t> host_data = {42};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 42;
  test_warp_lower_bound_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
}

// Test 17: All duplicates
BOOST_AUTO_TEST_CASE(test_equal_range_all_duplicates) {
  std::vector<uint32_t> host_data(10, 5);  // All 5s
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 5;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
  BOOST_REQUIRE_EQUAL(result[1], host_data.size());
}

// Test 18: 64-bit Index Type
BOOST_AUTO_TEST_CASE(test_lower_bound_64bit) {
  std::vector<uint64_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint64_t> dev_data = host_to_device(host_data);
  DeviceArray<uint64_t> dev_result(1);

  uint64_t key = 7;
  test_warp_lower_bound_kernel<uint64_t, uint64_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint64_t> result = device_to_host(dev_result);
  // CPU reference
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);
}

// Test 19: equal_range - Single element array, key found
BOOST_AUTO_TEST_CASE(test_equal_range_single_element_found) {
  std::vector<uint32_t> host_data = {42};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 42;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
  BOOST_REQUIRE_EQUAL(result[1], 1u);
}

// Test 20: equal_range - Single element array, key not found (less than)
BOOST_AUTO_TEST_CASE(test_equal_range_single_element_not_found_less) {
  std::vector<uint32_t> host_data = {42};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 10;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 0u);  // Insertion point
  BOOST_REQUIRE_EQUAL(result[1], 0u);  // Empty range
}

// Test 21: equal_range - Single element array, key not found (greater than)
BOOST_AUTO_TEST_CASE(test_equal_range_single_element_not_found_greater) {
  std::vector<uint32_t> host_data = {42};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 100;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 1u);  // Insertion point
  BOOST_REQUIRE_EQUAL(result[1], 1u);  // Empty range
}

// Test 22: equal_range - Two element array, key is first element
BOOST_AUTO_TEST_CASE(test_equal_range_two_elements_key_first) {
  std::vector<uint32_t> host_data = {10, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 10;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
  BOOST_REQUIRE_EQUAL(result[1], 1u);
}

// Test 23: equal_range - Two element array, key is second element (THE BUG CASE)
BOOST_AUTO_TEST_CASE(test_equal_range_two_elements_key_second) {
  std::vector<uint32_t> host_data = {10, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 20;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 1u);
  BOOST_REQUIRE_EQUAL(result[1], 2u);  // This is the failing case!
  if (result[0] != lower || result[1] != upper) {
    BOOST_TEST_MESSAGE("FAILED: expected lower=" << lower << ", upper=" << upper << ", got lower="
                                                 << result[0] << ", upper=" << result[1]);
  }
}

// Test 24: equal_range - Two element array, key not found (between)
BOOST_AUTO_TEST_CASE(test_equal_range_two_elements_key_between) {
  std::vector<uint32_t> host_data = {10, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 15;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 1u);  // Insertion point
  BOOST_REQUIRE_EQUAL(result[1], 1u);  // Empty range
}

// Test 25: equal_range - Two element array, key not found (less than first)
BOOST_AUTO_TEST_CASE(test_equal_range_two_elements_key_less) {
  std::vector<uint32_t> host_data = {10, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 5;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
  BOOST_REQUIRE_EQUAL(result[1], 0u);
}

// Test 26: equal_range - Two element array, key not found (greater than last)
BOOST_AUTO_TEST_CASE(test_equal_range_two_elements_key_greater) {
  std::vector<uint32_t> host_data = {10, 20};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 30;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 2u);
  BOOST_REQUIRE_EQUAL(result[1], 2u);
}

// Test 27: equal_range - Two duplicate elements
BOOST_AUTO_TEST_CASE(test_equal_range_two_duplicates) {
  std::vector<uint32_t> host_data = {10, 10};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 10;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
  BOOST_REQUIRE_EQUAL(result[1], 2u);
}

// Test 28: equal_range - Three element array, key in middle
BOOST_AUTO_TEST_CASE(test_equal_range_three_elements_key_middle) {
  std::vector<uint32_t> host_data = {10, 20, 30};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 20;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 1u);
  BOOST_REQUIRE_EQUAL(result[1], 2u);
}

// Test 29: equal_range - Three element array, key is last
BOOST_AUTO_TEST_CASE(test_equal_range_three_elements_key_last) {
  std::vector<uint32_t> host_data = {10, 20, 30};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 30;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
  BOOST_REQUIRE_EQUAL(result[0], 2u);
  BOOST_REQUIRE_EQUAL(result[1], 3u);
}

// Test 30: equal_range - Exactly WARP_SIZE elements
BOOST_AUTO_TEST_CASE(test_equal_range_warp_size_elements) {
  std::vector<uint32_t> host_data(32);
  for (uint32_t i = 0; i < 32; ++i) {
    host_data[i] = i * 2;  // 0, 2, 4, ..., 62
  }
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 30;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
}

// Test 31: equal_range - WARP_SIZE + 1 elements (33), triggers binary search
BOOST_AUTO_TEST_CASE(test_equal_range_warp_size_plus_one) {
  std::vector<uint32_t> host_data(33);
  for (uint32_t i = 0; i < 33; ++i) {
    host_data[i] = i * 2;  // 0, 2, 4, ..., 64
  }
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 32;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
}

// Test 32: equal_range - Large array (100 elements)
BOOST_AUTO_TEST_CASE(test_equal_range_large_array) {
  std::vector<uint32_t> host_data(100);
  for (uint32_t i = 0; i < 100; ++i) {
    host_data[i] = i * 3;  // 0, 3, 6, ..., 297
  }
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(2);

  uint32_t key = 150;
  test_warp_equal_range_kernel<uint32_t, uint32_t>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t lower = cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  uint32_t upper = cpu_upper_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], lower);
  BOOST_REQUIRE_EQUAL(result[1], upper);
}

// Test 33: group_lower_bound with group size 1 (sequential)
BOOST_AUTO_TEST_CASE(test_group_lower_bound_sequential) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 7;
  // Use 1 thread per block to test sequential case
  test_group_lower_bound_kernel<uint32_t, uint32_t, 1>
      <<<1, 1>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);  // Index of 7
}

// Test 34: group_lower_bound with group size 8 (sub-warp)
BOOST_AUTO_TEST_CASE(test_group_lower_bound_subwarp_8) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 15;
  // Use 8 threads per block to test sub-warp group
  test_group_lower_bound_kernel<uint32_t, uint32_t, 8>
      <<<1, 8>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 7u);  // Index of 15
}

// Test 35: group_lower_bound with group size 16 (sub-warp)
BOOST_AUTO_TEST_CASE(test_group_lower_bound_subwarp_16) {
  std::vector<uint32_t> host_data = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                                     23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43,
                                     45, 47, 49, 51, 53, 55, 57, 59, 61, 63};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 31;
  // Use 16 threads per block to test sub-warp group
  test_group_lower_bound_kernel<uint32_t, uint32_t, 16>
      <<<1, 16>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 15u);  // Index of 31
}

// Test 36: group_lower_bound with group size WARP_SIZE (full warp, should match warp_lower_bound)
BOOST_AUTO_TEST_CASE(test_group_lower_bound_warp_32) {
  std::vector<uint32_t> host_data = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                                     23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43,
                                     45, 47, 49, 51, 53, 55, 57, 59, 61, 63};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 47;
  // Use WARP_SIZE threads per block to test full warp group
  test_group_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 23u);  // Index of 47
}

// Test 37: group_lower_bound with different group sizes on large array
BOOST_AUTO_TEST_CASE(test_group_lower_bound_large_array_all_sizes) {
  const size_t n = 1000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);  // Even numbers: 0, 2, 4, 6, ...
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  uint32_t key = 500;

  // Test all group sizes
  const std::vector<int> group_sizes = {1, 8, 16, 32};
  const std::vector<int> thread_counts = {1, 8, 16, 32};

  for (size_t i = 0; i < group_sizes.size(); ++i) {
    DeviceArray<uint32_t> dev_result(1);
    int group_size = group_sizes[i];
    int thread_count = thread_counts[i];

    if (group_size == 1) {
      test_group_lower_bound_kernel<uint32_t, uint32_t, 1>
          <<<1, 1>>>(dev_data.data(), host_data.size(), key, dev_result.data());
    } else if (group_size == 8) {
      test_group_lower_bound_kernel<uint32_t, uint32_t, 8>
          <<<1, 8>>>(dev_data.data(), host_data.size(), key, dev_result.data());
    } else if (group_size == 16) {
      test_group_lower_bound_kernel<uint32_t, uint32_t, 16>
          <<<1, 16>>>(dev_data.data(), host_data.size(), key, dev_result.data());
    } else if (group_size == 32) {
      test_group_lower_bound_kernel<uint32_t, uint32_t, 32>
          <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
    }

    GPU_DEVICE_SYNCHRONIZE();

    std::vector<uint32_t> result = device_to_host(dev_result);
    uint32_t expected =
        cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
    BOOST_REQUIRE_EQUAL(result[0], expected);
  }
}

// Test 38: group_lower_bound with group size 1 - key not found
BOOST_AUTO_TEST_CASE(test_group_lower_bound_sequential_key_not_found) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 6;  // Between 5 and 7
  test_group_lower_bound_kernel<uint32_t, uint32_t, 1>
      <<<1, 1>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);  // Should point to 7
}

// Test 39: group_lower_bound with group size 8 - key larger than all
BOOST_AUTO_TEST_CASE(test_group_lower_bound_subwarp_key_larger) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 100;
  test_group_lower_bound_kernel<uint32_t, uint32_t, 8>
      <<<1, 8>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], host_data.size());
}

// =============================================================================
// Linear Search Tests (group_linear_lower_bound)
// =============================================================================

using SRDatalog::GPU::group_linear_lower_bound;

// Kernel to test group_linear_lower_bound with arbitrary group sizes
template <typename T, typename IndexType, int GROUP_SIZE>
__global__ void test_group_linear_lower_bound_kernel(const T* data, IndexType len, T key,
                                                     IndexType* result) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<GROUP_SIZE>(block);

  // All threads in the group participate in the search
  IndexType res = group_linear_lower_bound<T, IndexType>(data, len, key, tile);

  // Only thread 0 in the tile writes the result
  if (tile.thread_rank() == 0) {
    *result = res;
  }
}

// Test 40: Exponential search - key at beginning (best case)
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_key_at_beginning) {
  const size_t n = 1000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  // Key at position 5 (value = 10)
  uint32_t key = 10;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 5u);
}

// Test 41: Exponential search - key at middle
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_key_at_middle) {
  const size_t n = 1000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  // Key at position 500 (value = 1000)
  uint32_t key = 1000;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
}

// Test 42: Exponential search - key at end (worst case for exp search)
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_key_at_end) {
  const size_t n = 1000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  // Key at last position (value = 1998)
  uint32_t key = 1998;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 999u);
}

// Test 43: Exponential search vs binary search - correctness comparison
BOOST_AUTO_TEST_CASE(test_exp_vs_binary_correctness) {
  const size_t n = 10000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 3);  // 0, 3, 6, 9, ...
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result_exp(1);
  DeviceArray<uint32_t> dev_result_bin(1);

  // Test multiple keys
  std::vector<uint32_t> test_keys = {0, 3, 15, 100, 999, 5000, 15000, 29997, 30000};

  for (uint32_t key : test_keys) {
    // Test exponential search
    test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
        <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result_exp.data());

    // Test binary search
    test_group_lower_bound_kernel<uint32_t, uint32_t, 32>
        <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result_bin.data());

    GPU_DEVICE_SYNCHRONIZE();

    std::vector<uint32_t> result_exp = device_to_host(dev_result_exp);
    std::vector<uint32_t> result_bin = device_to_host(dev_result_bin);

    BOOST_REQUIRE_EQUAL(result_exp[0], result_bin[0]);
  }
}

// Test 44: Exponential search - empty array
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_empty) {
  std::vector<uint32_t> host_data;
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 5;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), 0, key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
}

// Test 45: Exponential search - small array (<= group_size)
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_small_array) {
  std::vector<uint32_t> host_data = {1, 3, 5, 7, 9, 11, 13, 15};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 7;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 3u);
}

// Test 46: Exponential search - key smaller than all
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_key_smaller) {
  std::vector<uint32_t> host_data = {10, 20, 30, 40, 50};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 5;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], 0u);
}

// Test 47: Exponential search - key larger than all
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_key_larger) {
  std::vector<uint32_t> host_data = {10, 20, 30, 40, 50};
  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  uint32_t key = 100;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  BOOST_REQUIRE_EQUAL(result[0], host_data.size());
}

// Test 48: Exponential search - very large array (100K elements)
BOOST_AUTO_TEST_CASE(test_exp_lower_bound_very_large_array) {
  const size_t n = 100000;
  std::vector<uint32_t> host_data(n);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = static_cast<uint32_t>(i * 2);
  }

  DeviceArray<uint32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_result(1);

  // Key at position 100 (early in array - exp search should be faster)
  uint32_t key = 200;
  test_group_linear_lower_bound_kernel<uint32_t, uint32_t, 32>
      <<<1, WARP_SIZE>>>(dev_data.data(), host_data.size(), key, dev_result.data());
  GPU_DEVICE_SYNCHRONIZE();

  std::vector<uint32_t> result = device_to_host(dev_result);
  uint32_t expected =
      cpu_lower_bound(host_data.data(), static_cast<uint32_t>(host_data.size()), key);
  BOOST_REQUIRE_EQUAL(result[0], expected);
  BOOST_REQUIRE_EQUAL(result[0], 100u);
}

}  // namespace
