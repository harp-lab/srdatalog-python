#define BOOST_TEST_MODULE gpu_sort_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "gpu/device_array.h"
#include "gpu/sort.h"
#include "system.h"
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <thrust/copy.h>
#include <thrust/device_vector.h>

using SRDatalog::GPU::DeviceArray;
using SRDatalog::GPU::SortContext;
using SRDatalog::GPU::stable_lex_sort;
using SRDatalog::GPU::unstable_sort;

using SRDatalog::AlignedVector;
using SRDatalog::make_aligned_vector;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

// Helper: Copy device array to host vector
template <typename T>
std::vector<T> device_to_host(const DeviceArray<T>& dev_array) {
  std::vector<T> host(dev_array.size());
  thrust::copy(dev_array.begin(), dev_array.end(), host.begin());
  GPU_DEVICE_SYNCHRONIZE();  // Ensure copy is complete
  return host;
}

// Helper: Copy host vector to device array
template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  GPU_DEVICE_SYNCHRONIZE();  // Ensure copy is complete
  return dev;
}

// Helper: Apply permutation p to src, storing result in dst
template <typename T, typename I>
void apply_perm(const T* src, T* dst, const I* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[p[i]];
  }
}

// Helper: Check if array is sorted
template <typename T>
bool is_sorted(const T* arr, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    if (arr[i - 1] > arr[i])
      return false;
  }
  return true;
}

// Helper: Check if permutation is valid (contains 0..N-1 exactly once)
template <typename I>
bool is_valid_permutation(const I* perm, size_t n) {
  std::vector<bool> seen(n, false);
  for (size_t i = 0; i < n; ++i) {
    if (perm[i] >= static_cast<I>(n) || seen[perm[i]])
      return false;
    seen[perm[i]] = true;
  }
  return true;
}

// ============================================================================
// Tests
// ============================================================================

// Test 1: Basic single column sort (int32_t)
BOOST_AUTO_TEST_CASE(test_single_column_int32) {
  const size_t n = 100;
  std::vector<int32_t> host_data(n);
  std::vector<uint32_t> host_ids(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), n, order, dev_data.data());

  // Copy results back
  host_ids = device_to_host(dev_ids);
  std::vector<int32_t> host_data_after = device_to_host(dev_data);

  // Verify permutation validity
  BOOST_REQUIRE(is_valid_permutation(host_ids.data(), n));

  // Verify sorted order by applying permutation
  std::vector<int32_t> sorted_data(n);
  apply_perm(host_data.data(), sorted_data.data(), host_ids.data(), n);
  BOOST_REQUIRE(is_sorted(sorted_data.data(), n));

  // Verify original data was not modified
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(host_data[i], host_data_after[i]);
  }
}

// Test 2: Multi-column lexicographic sort
BOOST_AUTO_TEST_CASE(test_multi_column_sort) {
  const size_t n = 50;
  std::vector<int32_t> host_col0(n);
  std::vector<int32_t> host_col1(n);
  std::vector<uint32_t> host_ids(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 5);
  for (size_t i = 0; i < n; ++i) {
    host_col0[i] = dist(gen);
    host_col1[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_col0 = host_to_device(host_col0);
  DeviceArray<int32_t> dev_col1 = host_to_device(host_col1);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0, 1};
  stable_lex_sort(dev_ids.data(), n, order, dev_col0.data(), dev_col1.data());

  host_ids = device_to_host(dev_ids);
  host_col0 = device_to_host(dev_col0);
  host_col1 = device_to_host(dev_col1);

  // Verify lexicographic order using indices
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids[i];
    uint32_t prev = host_ids[i - 1];

    BOOST_REQUIRE_LE(host_col0[prev], host_col0[curr]);
    if (host_col0[prev] == host_col0[curr]) {
      BOOST_REQUIRE_LE(host_col1[prev], host_col1[curr]);
    }
  }
}

// Test 3: Stability test
BOOST_AUTO_TEST_CASE(test_stability) {
  const size_t n = 20;
  std::vector<int32_t> host_keys(n);
  std::vector<int32_t> host_position(n);

  // Create data with duplicates
  for (size_t i = 0; i < n; ++i) {
    host_keys[i] = static_cast<int32_t>(i / 4);  // Many duplicates
    host_position[i] = static_cast<int32_t>(i);
  }

  // Shuffle to test stability
  std::mt19937 gen(42);
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  std::vector<int32_t> shuffled_keys(n);
  std::vector<int32_t> shuffled_position(n);
  for (size_t i = 0; i < n; ++i) {
    shuffled_keys[i] = host_keys[indices[i]];
    shuffled_position[i] = static_cast<int32_t>(i);
  }

  DeviceArray<int32_t> dev_keys = host_to_device(shuffled_keys);
  DeviceArray<int32_t> dev_position = host_to_device(shuffled_position);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), n, order, dev_keys.data());

  std::vector<uint32_t> host_ids = device_to_host(dev_ids);
  std::vector<int32_t> host_keys_after = device_to_host(dev_keys);

  // For equal keys, relative order from shuffled array should be preserved
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids[i];
    uint32_t prev = host_ids[i - 1];

    if (host_keys_after[prev] == host_keys_after[curr]) {
      BOOST_REQUIRE_LE(shuffled_position[prev], shuffled_position[curr]);
    }
  }
}

// Test 4: Edge case - empty array
BOOST_AUTO_TEST_CASE(test_empty_array) {
  DeviceArray<int32_t> dev_data(0);
  DeviceArray<uint32_t> dev_ids(0);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), 0, order, dev_data.data());
  // Empty array should not crash
  BOOST_REQUIRE(true);
}

// Test 5: Edge case - single element
BOOST_AUTO_TEST_CASE(test_single_element) {
  std::vector<int32_t> host_data = {42};
  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_ids(1);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), 1, order, dev_data.data());

  std::vector<uint32_t> host_ids = device_to_host(dev_ids);
  std::vector<int32_t> host_data_after = device_to_host(dev_data);

  BOOST_REQUIRE_EQUAL(host_ids[0], 0u);
  BOOST_REQUIRE_EQUAL(host_data_after[0], 42);
}

// Test 6: Edge case - all duplicates
BOOST_AUTO_TEST_CASE(test_all_duplicates) {
  const size_t n = 20;
  std::vector<int32_t> host_data(n, 42);
  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), n, order, dev_data.data());

  std::vector<uint32_t> host_ids = device_to_host(dev_ids);
  // Permutation should be identity (0, 1, 2, ...) because it's stable
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(host_ids[i], static_cast<uint32_t>(i));
  }
}

// Test 7: Different sort order
BOOST_AUTO_TEST_CASE(test_different_sort_order) {
  const size_t n = 30;
  std::vector<int32_t> host_col0(n);
  std::vector<int32_t> host_col1(n);
  std::vector<uint32_t> host_ids(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 5);
  for (size_t i = 0; i < n; ++i) {
    host_col0[i] = dist(gen);
    host_col1[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_col0 = host_to_device(host_col0);
  DeviceArray<int32_t> dev_col1 = host_to_device(host_col1);
  DeviceArray<uint32_t> dev_ids(n);

  // Sort by col1 first, then col0
  std::vector<int> order = {1, 0};
  stable_lex_sort(dev_ids.data(), n, order, dev_col0.data(), dev_col1.data());

  host_ids = device_to_host(dev_ids);
  host_col0 = device_to_host(dev_col0);
  host_col1 = device_to_host(dev_col1);

  // Verify lexicographic order (col1, col0)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids[i];
    uint32_t prev = host_ids[i - 1];

    BOOST_REQUIRE_LE(host_col1[prev], host_col1[curr]);
    if (host_col1[prev] == host_col1[curr]) {
      BOOST_REQUIRE_LE(host_col0[prev], host_col0[curr]);
    }
  }
}

// Test 8: Large array
BOOST_AUTO_TEST_CASE(test_large_array) {
  const size_t n = 10000;
  std::vector<int32_t> host_data(n);
  std::vector<uint32_t> host_ids(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-100000, 100000);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0};
  stable_lex_sort(dev_ids.data(), n, order, dev_data.data());

  host_ids = device_to_host(dev_ids);
  host_data = device_to_host(dev_data);

  BOOST_REQUIRE(is_valid_permutation(host_ids.data(), n));

  // Check order
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(host_data[host_ids[i - 1]], host_data[host_ids[i]]);
  }
}

// Test 9: Context reuse
BOOST_AUTO_TEST_CASE(test_reuse_sort_context) {
  const size_t n = 100;
  std::vector<int32_t> host_col0(n);
  std::vector<int32_t> host_col1(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 10);
  for (size_t i = 0; i < n; ++i) {
    host_col0[i] = dist(gen);
    host_col1[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_col0 = host_to_device(host_col0);
  DeviceArray<int32_t> dev_col1 = host_to_device(host_col1);

  SortContext<uint32_t> ctx;

  // First sort: by col0, then col1
  std::vector<int> order1 = {0, 1};
  stable_lex_sort(n, order1, ctx, dev_col0.data(), dev_col1.data());
  std::vector<uint32_t> host_ids1(n);
  thrust::copy(ctx.permutation.begin(), ctx.permutation.end(), host_ids1.begin());

  // Second sort: by col1, then col0 (reusing the same context)
  std::vector<int> order2 = {1, 0};
  stable_lex_sort(n, order2, ctx, dev_col0.data(), dev_col1.data());
  std::vector<uint32_t> host_ids2(n);
  thrust::copy(ctx.permutation.begin(), ctx.permutation.end(), host_ids2.begin());

  host_col0 = device_to_host(dev_col0);
  host_col1 = device_to_host(dev_col1);

  // Verify both sorts are correct
  BOOST_REQUIRE(is_valid_permutation(host_ids1.data(), n));
  BOOST_REQUIRE(is_valid_permutation(host_ids2.data(), n));

  // Verify lexicographic order for first sort (col0, col1)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids1[i];
    uint32_t prev = host_ids1[i - 1];
    BOOST_REQUIRE_LE(host_col0[prev], host_col0[curr]);
    if (host_col0[prev] == host_col0[curr]) {
      BOOST_REQUIRE_LE(host_col1[prev], host_col1[curr]);
    }
  }

  // Verify lexicographic order for second sort (col1, col0)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids2[i];
    uint32_t prev = host_ids2[i - 1];
    BOOST_REQUIRE_LE(host_col1[prev], host_col1[curr]);
    if (host_col1[prev] == host_col1[curr]) {
      BOOST_REQUIRE_LE(host_col0[prev], host_col0[curr]);
    }
  }
}

// Test 10: reorder_data flag
BOOST_AUTO_TEST_CASE(test_reorder_data_flag) {
  const size_t n = 100;
  std::vector<int32_t> host_data(n);
  std::vector<int32_t> host_data_copy(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
    host_data_copy[i] = host_data[i];
  }

  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0};

  // Test 1: reorder_data = false (default behavior - only sort indices)
  {
    dev_data = host_to_device(host_data);
    stable_lex_sort(dev_ids.data(), n, order, false, dev_data.data());

    std::vector<uint32_t> host_ids = device_to_host(dev_ids);
    std::vector<int32_t> host_data_after = device_to_host(dev_data);

    BOOST_REQUIRE(is_valid_permutation(host_ids.data(), n));
    for (size_t i = 1; i < n; ++i) {
      BOOST_REQUIRE_LE(host_data[host_ids[i - 1]], host_data[host_ids[i]]);
    }

    // Verify original data was NOT modified
    for (size_t i = 0; i < n; ++i) {
      BOOST_REQUIRE_EQUAL(host_data[i], host_data_after[i]);
    }
  }

  // Test 2: reorder_data = true (data should be reordered in place)
  {
    dev_data = host_to_device(host_data);
    stable_lex_sort(dev_ids.data(), n, order, true, dev_data.data());

    std::vector<int32_t> host_data_after = device_to_host(dev_data);

    // Verify data is now sorted in place
    BOOST_REQUIRE(is_sorted(host_data_after.data(), n));
  }
}

// Test 11: Three column sort
BOOST_AUTO_TEST_CASE(test_three_column_sort) {
  const size_t n = 30;
  std::vector<int32_t> host_col0(n);
  std::vector<int32_t> host_col1(n);
  std::vector<int32_t> host_col2(n);
  std::vector<uint32_t> host_ids(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 3);
  for (size_t i = 0; i < n; ++i) {
    host_col0[i] = dist(gen);
    host_col1[i] = dist(gen);
    host_col2[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_col0 = host_to_device(host_col0);
  DeviceArray<int32_t> dev_col1 = host_to_device(host_col1);
  DeviceArray<int32_t> dev_col2 = host_to_device(host_col2);
  DeviceArray<uint32_t> dev_ids(n);

  std::vector<int> order = {0, 1, 2};
  stable_lex_sort(dev_ids.data(), n, order, dev_col0.data(), dev_col1.data(), dev_col2.data());

  host_ids = device_to_host(dev_ids);
  host_col0 = device_to_host(dev_col0);
  host_col1 = device_to_host(dev_col1);
  host_col2 = device_to_host(dev_col2);

  // Verify lexicographic order
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = host_ids[i];
    uint32_t prev = host_ids[i - 1];

    BOOST_REQUIRE_LE(host_col0[prev], host_col0[curr]);
    if (host_col0[prev] == host_col0[curr]) {
      BOOST_REQUIRE_LE(host_col1[prev], host_col1[curr]);
      if (host_col1[prev] == host_col1[curr]) {
        BOOST_REQUIRE_LE(host_col2[prev], host_col2[curr]);
      }
    }
  }
}

// ============================================================================
// Unstable Sort Tests
// ============================================================================

// Test 12: Unstable sort - int32_t (large array)
BOOST_AUTO_TEST_CASE(test_unstable_sort_int32_large) {
  const size_t n = 10000;
  std::vector<int32_t> host_data(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  unstable_sort(dev_data.data(), n);

  host_data = device_to_host(dev_data);
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(host_data[i - 1], host_data[i]);
  }
}

// Test 13: Unstable sort - float
BOOST_AUTO_TEST_CASE(test_unstable_sort_float) {
  const size_t n = 1000;
  std::vector<float> host_data(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<float> dev_data = host_to_device(host_data);
  unstable_sort(dev_data.data(), n);

  host_data = device_to_host(dev_data);
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(host_data[i - 1], host_data[i]);
  }
}

// Test 14: Unstable sort - double
BOOST_AUTO_TEST_CASE(test_unstable_sort_double) {
  const size_t n = 2000;
  std::vector<double> host_data(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000000.0, 1000000.0);
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<double> dev_data = host_to_device(host_data);
  unstable_sort(dev_data.data(), n);

  host_data = device_to_host(dev_data);
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(host_data[i - 1], host_data[i]);
  }
}

// Test 15: Unstable sort - with duplicates
BOOST_AUTO_TEST_CASE(test_unstable_sort_with_duplicates) {
  const size_t n = 100;
  std::vector<int32_t> host_data(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 10);  // Many duplicates
  for (size_t i = 0; i < n; ++i) {
    host_data[i] = dist(gen);
  }

  DeviceArray<int32_t> dev_data = host_to_device(host_data);
  unstable_sort(dev_data.data(), n);

  host_data = device_to_host(dev_data);
  // Verify sorted order (unstable sort may not preserve order of equal elements)
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(host_data[i - 1], host_data[i]);
  }
}

}  // namespace
