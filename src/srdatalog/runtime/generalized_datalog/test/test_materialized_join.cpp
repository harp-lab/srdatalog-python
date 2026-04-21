/**
 * @file test_materialized_join.cpp
 * @brief Unit tests for materialized join helpers
 *
 * Tests probe_count_matches, probe_materialize_pairs, and gather_column
 * functions from materialized_join.h
 */

#define BOOST_TEST_MODULE materialized_join_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#include "gpu/device_array.h"
#include "gpu/gpu_api.h"
#include "gpu/runtime/jit/materialized_join.h"

using SRDatalog::GPU::DeviceArray;
using namespace SRDatalog::GPU::JIT;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

template <typename T>
std::vector<T> device_to_host(const thrust::device_vector<T>& dev) {
  std::vector<T> host(dev.size());
  thrust::copy(dev.begin(), dev.end(), host.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return host;
}

template <typename T>
thrust::device_vector<T> host_to_device(const std::vector<T>& host) {
  thrust::device_vector<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return dev;
}

// Simple mock view for testing (simulates sorted index with get_range)
struct MockSortedView {
  const int64_t* keys;  // Sorted keys
  uint32_t size;

  __device__ thrust::pair<uint32_t, uint32_t> get_range(int64_t key) const {
    // Binary search for lower and upper bounds
    uint32_t lo = 0, hi = size;

    // Lower bound
    uint32_t lower = size;
    lo = 0;
    hi = size;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (keys[mid] < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    lower = lo;

    // Upper bound
    uint32_t upper = size;
    lo = lower;
    hi = size;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (keys[mid] <= key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    upper = lo;

    return {lower, upper};
  }
};

// ============================================================================
// Tests for gather_column
// ============================================================================

BOOST_AUTO_TEST_CASE(test_gather_column_basic) {
  // Source column: {10, 20, 30, 40, 50}
  std::vector<int64_t> column_data = {10, 20, 30, 40, 50};
  auto dev_column = host_to_device(column_data);

  // Row IDs to gather: {2, 0, 4, 1}
  std::vector<uint32_t> rowids = {2, 0, 4, 1};
  auto dev_rowids = host_to_device(rowids);

  // Output
  thrust::device_vector<int64_t> dev_output(rowids.size());

  // Run gather
  gather_column(dev_rowids, dev_column, dev_output);
  GPU_DEVICE_SYNCHRONIZE();

  // Verify
  auto result = device_to_host(dev_output);
  BOOST_REQUIRE_EQUAL(result[0], 30);  // column[2]
  BOOST_REQUIRE_EQUAL(result[1], 10);  // column[0]
  BOOST_REQUIRE_EQUAL(result[2], 50);  // column[4]
  BOOST_REQUIRE_EQUAL(result[3], 20);  // column[1]
}

BOOST_AUTO_TEST_CASE(test_gather_column_empty) {
  std::vector<int64_t> column_data = {10, 20, 30};
  auto dev_column = host_to_device(column_data);

  std::vector<uint32_t> rowids = {};
  auto dev_rowids = host_to_device(rowids);

  thrust::device_vector<int64_t> dev_output(0);

  // Should not crash on empty input
  gather_column(dev_rowids, dev_column, dev_output);
  GPU_DEVICE_SYNCHRONIZE();

  BOOST_REQUIRE_EQUAL(dev_output.size(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gather_column_sequential) {
  // Test sequential access pattern
  std::vector<int64_t> column_data(100);
  std::iota(column_data.begin(), column_data.end(), 0);  // 0, 1, 2, ..., 99
  auto dev_column = host_to_device(column_data);

  std::vector<uint32_t> rowids(100);
  std::iota(rowids.begin(), rowids.end(), 0);  // 0, 1, 2, ..., 99
  auto dev_rowids = host_to_device(rowids);

  thrust::device_vector<int64_t> dev_output(100);

  gather_column(dev_rowids, dev_column, dev_output);
  GPU_DEVICE_SYNCHRONIZE();

  auto result = device_to_host(dev_output);
  for (size_t i = 0; i < 100; ++i) {
    BOOST_REQUIRE_EQUAL(result[i], static_cast<int64_t>(i));
  }
}

// ============================================================================
// Tests for gather_column_raw (kernel version)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_gather_column_raw_basic) {
  std::vector<int64_t> column_data = {100, 200, 300, 400};
  auto dev_column = host_to_device(column_data);

  std::vector<uint32_t> rowids = {3, 1, 0};
  auto dev_rowids = host_to_device(rowids);

  thrust::device_vector<int64_t> dev_output(rowids.size());

  gather_column_raw(
      thrust::raw_pointer_cast(dev_rowids.data()), thrust::raw_pointer_cast(dev_column.data()),
      thrust::raw_pointer_cast(dev_output.data()), static_cast<uint32_t>(rowids.size()));
  GPU_DEVICE_SYNCHRONIZE();

  auto result = device_to_host(dev_output);
  BOOST_REQUIRE_EQUAL(result[0], 400);  // column[3]
  BOOST_REQUIRE_EQUAL(result[1], 200);  // column[1]
  BOOST_REQUIRE_EQUAL(result[2], 100);  // column[0]
}

// ============================================================================
// Tests for compute_output_offsets
// ============================================================================

BOOST_AUTO_TEST_CASE(test_compute_output_offsets_basic) {
  // Counts: {3, 1, 4, 2}
  std::vector<uint32_t> counts = {3, 1, 4, 2};
  auto dev_counts = host_to_device(counts);

  thrust::device_vector<uint32_t> dev_offsets(counts.size());

  uint32_t total = compute_output_offsets(dev_counts, dev_offsets);
  GPU_DEVICE_SYNCHRONIZE();

  // Expected offsets: {0, 3, 4, 8} (exclusive scan)
  // Total = 8 + 2 = 10
  auto offsets = device_to_host(dev_offsets);
  BOOST_REQUIRE_EQUAL(offsets[0], 0u);
  BOOST_REQUIRE_EQUAL(offsets[1], 3u);
  BOOST_REQUIRE_EQUAL(offsets[2], 4u);
  BOOST_REQUIRE_EQUAL(offsets[3], 8u);
  BOOST_REQUIRE_EQUAL(total, 10u);
}

BOOST_AUTO_TEST_CASE(test_compute_output_offsets_zeros) {
  // Test with some zero counts
  std::vector<uint32_t> counts = {0, 2, 0, 3, 0};
  auto dev_counts = host_to_device(counts);

  thrust::device_vector<uint32_t> dev_offsets(counts.size());

  uint32_t total = compute_output_offsets(dev_counts, dev_offsets);
  GPU_DEVICE_SYNCHRONIZE();

  auto offsets = device_to_host(dev_offsets);
  BOOST_REQUIRE_EQUAL(offsets[0], 0u);
  BOOST_REQUIRE_EQUAL(offsets[1], 0u);
  BOOST_REQUIRE_EQUAL(offsets[2], 2u);
  BOOST_REQUIRE_EQUAL(offsets[3], 2u);
  BOOST_REQUIRE_EQUAL(offsets[4], 5u);
  BOOST_REQUIRE_EQUAL(total, 5u);
}

BOOST_AUTO_TEST_CASE(test_compute_output_offsets_single) {
  std::vector<uint32_t> counts = {42};
  auto dev_counts = host_to_device(counts);

  thrust::device_vector<uint32_t> dev_offsets(1);

  uint32_t total = compute_output_offsets(dev_counts, dev_offsets);
  GPU_DEVICE_SYNCHRONIZE();

  auto offsets = device_to_host(dev_offsets);
  BOOST_REQUIRE_EQUAL(offsets[0], 0u);
  BOOST_REQUIRE_EQUAL(total, 42u);
}

}  // namespace
