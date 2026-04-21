#define BOOST_TEST_MODULE cub_sort_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "gpu/cub_sort.h"
#include "gpu/device_array.h"
#include "gpu/gpu_api.h"
#include "gpu/sort.h"  // Original Thrust sort for comparison
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

using SRDatalog::GPU::cub_lex_sort;
using SRDatalog::GPU::CubSortStorage;
using SRDatalog::GPU::DeviceArray;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

template <typename T>
std::vector<T> d2h(const DeviceArray<T>& dev) {
  std::vector<T> host(dev.size());
  thrust::copy(dev.begin(), dev.end(), host.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return host;
}

template <typename T>
DeviceArray<T> h2d(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  GPU_DEVICE_SYNCHRONIZE();
  return dev;
}

/// Generate random uint32 data
std::vector<uint32_t> rand_u32(size_t n, uint32_t lo, uint32_t hi, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dist(lo, hi);
  std::vector<uint32_t> v(n);
  for (auto& x : v)
    x = dist(gen);
  return v;
}

/// CPU reference: lexicographic sort of N columns, returns sorted row indices
std::vector<uint32_t> cpu_lex_sort(const std::vector<std::vector<uint32_t>>& cols,
                                   const std::vector<int>& order) {
  size_t n = cols[0].size();
  std::vector<uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
    for (int c : order) {
      if (cols[c][a] != cols[c][b])
        return cols[c][a] < cols[c][b];
    }
    return false;  // equal
  });
  return perm;
}

/// Verify that device columns are in lexicographic order according to `order`
void verify_lex_sorted(const std::vector<std::vector<uint32_t>>& cols,
                       const std::vector<int>& order, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    for (int c : order) {
      if (cols[c][i - 1] < cols[c][i])
        break;  // strictly less → ok
      BOOST_REQUIRE_LE(cols[c][i - 1], cols[c][i]);
      if (cols[c][i - 1] == cols[c][i])
        continue;  // equal → check next column
      break;
    }
  }
}

/// Verify CUB sort matches CPU reference exactly (same multiset of rows)
void verify_matches_cpu(const std::vector<std::vector<uint32_t>>& host_cols_orig,
                        const std::vector<std::vector<uint32_t>>& host_cols_sorted,
                        const std::vector<int>& order) {
  size_t n = host_cols_orig[0].size();
  auto ref_perm = cpu_lex_sort(host_cols_orig, order);

  for (size_t i = 0; i < n; ++i) {
    for (size_t c = 0; c < host_cols_orig.size(); ++c) {
      BOOST_REQUIRE_EQUAL(host_cols_sorted[c][i], host_cols_orig[c][ref_perm[i]]);
    }
  }
}

/// Run CUB sort on device columns in-place, return host results
std::vector<std::vector<uint32_t>> run_cub_sort(const std::vector<std::vector<uint32_t>>& host_cols,
                                                const std::vector<int>& order,
                                                GPU_STREAM_T stream = 0) {
  int num_cols = static_cast<int>(host_cols.size());
  uint32_t n = static_cast<uint32_t>(host_cols[0].size());

  std::vector<DeviceArray<uint32_t>> dev_cols;
  std::vector<uint32_t*> col_ptrs;
  for (int c = 0; c < num_cols; ++c) {
    dev_cols.push_back(h2d(host_cols[c]));
    col_ptrs.push_back(dev_cols.back().data());
  }

  CubSortStorage storage;
  cub_lex_sort(col_ptrs.data(), num_cols, n, order.data(), stream, storage);
  GPU_STREAM_SYNCHRONIZE(stream);

  std::vector<std::vector<uint32_t>> result;
  for (int c = 0; c < num_cols; ++c) {
    result.push_back(d2h(dev_cols[c]));
  }
  return result;
}

// ============================================================================
// Correctness Tests: Arity 1
// ============================================================================

BOOST_AUTO_TEST_CASE(arity1_basic) {
  auto col = rand_u32(100, 0, 1000);
  auto result = run_cub_sort({col}, {0});
  verify_lex_sorted(result, {0}, 100);
  verify_matches_cpu({col}, result, {0});
}

BOOST_AUTO_TEST_CASE(arity1_empty) {
  auto result = run_cub_sort({{}}, {0});
  BOOST_REQUIRE(result[0].empty());
}

BOOST_AUTO_TEST_CASE(arity1_single) {
  auto result = run_cub_sort({{42}}, {0});
  BOOST_REQUIRE_EQUAL(result[0][0], 42u);
}

BOOST_AUTO_TEST_CASE(arity1_all_equal) {
  std::vector<uint32_t> col(50, 7);
  auto result = run_cub_sort({col}, {0});
  for (auto v : result[0])
    BOOST_REQUIRE_EQUAL(v, 7u);
}

BOOST_AUTO_TEST_CASE(arity1_already_sorted) {
  std::vector<uint32_t> col(100);
  std::iota(col.begin(), col.end(), 0u);
  auto result = run_cub_sort({col}, {0});
  verify_matches_cpu({col}, result, {0});
}

BOOST_AUTO_TEST_CASE(arity1_reverse_sorted) {
  std::vector<uint32_t> col(100);
  std::iota(col.begin(), col.end(), 0u);
  std::reverse(col.begin(), col.end());
  auto result = run_cub_sort({col}, {0});
  verify_lex_sorted(result, {0}, 100);
  verify_matches_cpu({col}, result, {0});
}

// ============================================================================
// Correctness Tests: Arity 2
// ============================================================================

BOOST_AUTO_TEST_CASE(arity2_order_01) {
  auto c0 = rand_u32(200, 0, 10);
  auto c1 = rand_u32(200, 0, 10);
  auto result = run_cub_sort({c0, c1}, {0, 1});
  verify_lex_sorted(result, {0, 1}, 200);
  verify_matches_cpu({c0, c1}, result, {0, 1});
}

BOOST_AUTO_TEST_CASE(arity2_order_10) {
  auto c0 = rand_u32(200, 0, 10);
  auto c1 = rand_u32(200, 0, 10);
  auto result = run_cub_sort({c0, c1}, {1, 0});
  verify_lex_sorted(result, {1, 0}, 200);
  verify_matches_cpu({c0, c1}, result, {1, 0});
}

BOOST_AUTO_TEST_CASE(arity2_large) {
  auto c0 = rand_u32(100000, 0, 100000);
  auto c1 = rand_u32(100000, 0, 100000);
  auto result = run_cub_sort({c0, c1}, {0, 1});
  verify_lex_sorted(result, {0, 1}, 100000);
  verify_matches_cpu({c0, c1}, result, {0, 1});
}

BOOST_AUTO_TEST_CASE(arity2_many_duplicates) {
  auto c0 = rand_u32(1000, 0, 3);
  auto c1 = rand_u32(1000, 0, 3);
  auto result = run_cub_sort({c0, c1}, {0, 1});
  verify_lex_sorted(result, {0, 1}, 1000);
  verify_matches_cpu({c0, c1}, result, {0, 1});
}

BOOST_AUTO_TEST_CASE(arity2_col0_constant) {
  std::vector<uint32_t> c0(500, 5);
  auto c1 = rand_u32(500, 0, 100);
  auto result = run_cub_sort({c0, c1}, {0, 1});
  verify_lex_sorted(result, {0, 1}, 500);
  verify_matches_cpu({c0, c1}, result, {0, 1});
}

// ============================================================================
// Correctness Tests: Arity 3
// ============================================================================

BOOST_AUTO_TEST_CASE(arity3_order_012) {
  auto c0 = rand_u32(300, 0, 5);
  auto c1 = rand_u32(300, 0, 5);
  auto c2 = rand_u32(300, 0, 5);
  auto result = run_cub_sort({c0, c1, c2}, {0, 1, 2});
  verify_lex_sorted(result, {0, 1, 2}, 300);
  verify_matches_cpu({c0, c1, c2}, result, {0, 1, 2});
}

BOOST_AUTO_TEST_CASE(arity3_order_120) {
  auto c0 = rand_u32(300, 0, 5);
  auto c1 = rand_u32(300, 0, 5);
  auto c2 = rand_u32(300, 0, 5);
  auto result = run_cub_sort({c0, c1, c2}, {1, 2, 0});
  verify_lex_sorted(result, {1, 2, 0}, 300);
  verify_matches_cpu({c0, c1, c2}, result, {1, 2, 0});
}

BOOST_AUTO_TEST_CASE(arity3_order_210) {
  auto c0 = rand_u32(300, 0, 5);
  auto c1 = rand_u32(300, 0, 5);
  auto c2 = rand_u32(300, 0, 5);
  auto result = run_cub_sort({c0, c1, c2}, {2, 1, 0});
  verify_lex_sorted(result, {2, 1, 0}, 300);
  verify_matches_cpu({c0, c1, c2}, result, {2, 1, 0});
}

BOOST_AUTO_TEST_CASE(arity3_all_permutations) {
  // Test all 6 orderings of 3 columns
  auto c0 = rand_u32(100, 0, 5);
  auto c1 = rand_u32(100, 0, 5);
  auto c2 = rand_u32(100, 0, 5);

  int orders[][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (auto& o : orders) {
    std::vector<int> order = {o[0], o[1], o[2]};
    auto result = run_cub_sort({c0, c1, c2}, order);
    verify_lex_sorted(result, order, 100);
    verify_matches_cpu({c0, c1, c2}, result, order);
  }
}

BOOST_AUTO_TEST_CASE(arity3_large) {
  auto c0 = rand_u32(100000, 0, 100000);
  auto c1 = rand_u32(100000, 0, 100000);
  auto c2 = rand_u32(100000, 0, 100000);
  auto result = run_cub_sort({c0, c1, c2}, {0, 1, 2});
  verify_lex_sorted(result, {0, 1, 2}, 100000);
  verify_matches_cpu({c0, c1, c2}, result, {0, 1, 2});
}

// ============================================================================
// Correctness Tests: Arity 4, 5, 6 (arbitrary)
// ============================================================================

BOOST_AUTO_TEST_CASE(arity4_basic) {
  auto c0 = rand_u32(200, 0, 5, 1);
  auto c1 = rand_u32(200, 0, 5, 2);
  auto c2 = rand_u32(200, 0, 5, 3);
  auto c3 = rand_u32(200, 0, 5, 4);
  auto result = run_cub_sort({c0, c1, c2, c3}, {0, 1, 2, 3});
  verify_lex_sorted(result, {0, 1, 2, 3}, 200);
  verify_matches_cpu({c0, c1, c2, c3}, result, {0, 1, 2, 3});
}

BOOST_AUTO_TEST_CASE(arity4_reverse_order) {
  auto c0 = rand_u32(200, 0, 5, 1);
  auto c1 = rand_u32(200, 0, 5, 2);
  auto c2 = rand_u32(200, 0, 5, 3);
  auto c3 = rand_u32(200, 0, 5, 4);
  auto result = run_cub_sort({c0, c1, c2, c3}, {3, 2, 1, 0});
  verify_lex_sorted(result, {3, 2, 1, 0}, 200);
  verify_matches_cpu({c0, c1, c2, c3}, result, {3, 2, 1, 0});
}

BOOST_AUTO_TEST_CASE(arity5_basic) {
  std::vector<std::vector<uint32_t>> cols;
  for (int i = 0; i < 5; ++i)
    cols.push_back(rand_u32(150, 0, 4, 10 + i));
  std::vector<int> order = {0, 1, 2, 3, 4};
  auto result = run_cub_sort(cols, order);
  verify_lex_sorted(result, order, 150);
  verify_matches_cpu(cols, result, order);
}

BOOST_AUTO_TEST_CASE(arity6_basic) {
  std::vector<std::vector<uint32_t>> cols;
  for (int i = 0; i < 6; ++i)
    cols.push_back(rand_u32(120, 0, 3, 20 + i));
  std::vector<int> order = {5, 3, 1, 4, 2, 0};  // arbitrary order
  auto result = run_cub_sort(cols, order);
  verify_lex_sorted(result, order, 120);
  verify_matches_cpu(cols, result, order);
}

// ============================================================================
// Edge Cases
// ============================================================================

BOOST_AUTO_TEST_CASE(edge_two_elements) {
  auto result = run_cub_sort({{5, 3}, {2, 7}}, {0, 1});
  BOOST_REQUIRE_EQUAL(result[0][0], 3u);
  BOOST_REQUIRE_EQUAL(result[0][1], 5u);
  BOOST_REQUIRE_EQUAL(result[1][0], 7u);
  BOOST_REQUIRE_EQUAL(result[1][1], 2u);
}

BOOST_AUTO_TEST_CASE(edge_large_values) {
  // Near uint32 max
  std::vector<uint32_t> c0 = {UINT32_MAX, 0, UINT32_MAX - 1, 1};
  std::vector<uint32_t> c1 = {0, UINT32_MAX, 1, UINT32_MAX - 1};
  auto result = run_cub_sort({c0, c1}, {0, 1});
  verify_lex_sorted(result, {0, 1}, 4);
  verify_matches_cpu({c0, c1}, result, {0, 1});
}

// ============================================================================
// Stream Tests: verify sort works on non-default stream
// ============================================================================

BOOST_AUTO_TEST_CASE(stream_nondefault) {
  GPU_STREAM_T stream;
  GPU_STREAM_CREATE(&stream);

  auto c0 = rand_u32(1000, 0, 100);
  auto c1 = rand_u32(1000, 0, 100);
  auto result = run_cub_sort({c0, c1}, {0, 1}, stream);
  verify_lex_sorted(result, {0, 1}, 1000);
  verify_matches_cpu({c0, c1}, result, {0, 1});

  GPU_STREAM_DESTROY(stream);
}

BOOST_AUTO_TEST_CASE(stream_multiple_sorts_different_streams) {
  // Two independent sorts on two different streams — should not interfere
  GPU_STREAM_T s1, s2;
  GPU_STREAM_CREATE(&s1);
  GPU_STREAM_CREATE(&s2);

  uint32_t n = 5000;
  auto c0a = rand_u32(n, 0, 100, 1);
  auto c1a = rand_u32(n, 0, 100, 2);
  auto c0b = rand_u32(n, 0, 100, 3);
  auto c1b = rand_u32(n, 0, 100, 4);

  // Setup device arrays
  auto d_c0a = h2d(c0a);
  auto d_c1a = h2d(c1a);
  auto d_c0b = h2d(c0b);
  auto d_c1b = h2d(c1b);

  std::vector<int> order = {0, 1};
  CubSortStorage stor1, stor2;

  // Launch both sorts without synchronizing between them
  std::vector<uint32_t*> ptrsA = {d_c0a.data(), d_c1a.data()};
  std::vector<uint32_t*> ptrsB = {d_c0b.data(), d_c1b.data()};
  cub_lex_sort(ptrsA, n, order, s1, stor1);
  cub_lex_sort(ptrsB, n, order, s2, stor2);

  // Sync both streams
  GPU_STREAM_SYNCHRONIZE(s1);
  GPU_STREAM_SYNCHRONIZE(s2);

  // Verify both
  auto rA0 = d2h(d_c0a);
  auto rA1 = d2h(d_c1a);
  auto rB0 = d2h(d_c0b);
  auto rB1 = d2h(d_c1b);

  verify_lex_sorted({rA0, rA1}, order, n);
  verify_matches_cpu({c0a, c1a}, {rA0, rA1}, order);

  verify_lex_sorted({rB0, rB1}, order, n);
  verify_matches_cpu({c0b, c1b}, {rB0, rB1}, order);

  GPU_STREAM_DESTROY(s1);
  GPU_STREAM_DESTROY(s2);
}

BOOST_AUTO_TEST_CASE(stream_four_parallel_sorts_arity3) {
  // Simulates 4 secondary index rebuilds running in parallel on different streams
  GPU_STREAM_T streams[4];
  for (auto& s : streams)
    GPU_STREAM_CREATE(&s);

  uint32_t n = 10000;
  int orders[][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

  // Generate one source dataset
  auto c0 = rand_u32(n, 0, 1000);
  auto c1 = rand_u32(n, 0, 1000);
  auto c2 = rand_u32(n, 0, 1000);

  // Create 4 independent copies on device
  struct SortJob {
    DeviceArray<uint32_t> d0, d1, d2;
    CubSortStorage storage;
  };
  std::vector<SortJob> jobs(4);

  for (int j = 0; j < 4; ++j) {
    jobs[j].d0 = h2d(c0);
    jobs[j].d1 = h2d(c1);
    jobs[j].d2 = h2d(c2);
  }

  // Launch all 4 sorts in parallel on different streams
  for (int j = 0; j < 4; ++j) {
    std::vector<uint32_t*> ptrs = {jobs[j].d0.data(), jobs[j].d1.data(), jobs[j].d2.data()};
    std::vector<int> order = {orders[j][0], orders[j][1], orders[j][2]};
    cub_lex_sort(ptrs, n, order, streams[j], jobs[j].storage);
  }

  // Sync all
  for (auto& s : streams)
    GPU_STREAM_SYNCHRONIZE(s);

  // Verify all 4
  for (int j = 0; j < 4; ++j) {
    auto r0 = d2h(jobs[j].d0);
    auto r1 = d2h(jobs[j].d1);
    auto r2 = d2h(jobs[j].d2);
    std::vector<int> order = {orders[j][0], orders[j][1], orders[j][2]};

    verify_lex_sorted({r0, r1, r2}, order, n);
    verify_matches_cpu({c0, c1, c2}, {r0, r1, r2}, order);
  }

  for (auto& s : streams)
    GPU_STREAM_DESTROY(s);
}

// ============================================================================
// Microbenchmark: CUB vs Thrust at various sizes
// ============================================================================

BOOST_AUTO_TEST_CASE(bench_cub_vs_thrust_arity2) {
  std::cout << "\n=== Microbenchmark: CUB vs Thrust, arity-2, order {0,1} ===" << std::endl;
  std::cout << "    N        CUB(us)   Thrust(us)  Ratio" << std::endl;

  for (uint32_t n : {100u, 1000u, 10000u, 100000u, 1000000u, 5000000u}) {
    auto c0 = rand_u32(n, 0, n);
    auto c1 = rand_u32(n, 0, n);
    std::vector<int> order = {0, 1};

    // Warmup
    {
      auto d0 = h2d(c0);
      auto d1 = h2d(c1);
      std::vector<uint32_t*> ptrs = {d0.data(), d1.data()};
      CubSortStorage stor;
      cub_lex_sort(ptrs, n, order, 0, stor);
      GPU_DEVICE_SYNCHRONIZE();
    }

    // CUB timing
    double cub_us = 0;
    {
      constexpr int REPS = 5;
      for (int r = 0; r < REPS; ++r) {
        auto d0 = h2d(c0);
        auto d1 = h2d(c1);
        std::vector<uint32_t*> ptrs = {d0.data(), d1.data()};
        CubSortStorage stor;
        GPU_DEVICE_SYNCHRONIZE();
        auto t0 = std::chrono::high_resolution_clock::now();
        cub_lex_sort(ptrs, n, order, 0, stor);
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        cub_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      cub_us /= REPS;
    }

    // Thrust timing
    double thrust_us = 0;
    {
      constexpr int REPS = 5;
      for (int r = 0; r < REPS; ++r) {
        auto d0 = h2d(c0);
        auto d1 = h2d(c1);
        GPU_DEVICE_SYNCHRONIZE();
        auto t0 = std::chrono::high_resolution_clock::now();
        SRDatalog::GPU::zip_sort_columns(n, d0.data(), d1.data());
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        thrust_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      thrust_us /= REPS;
    }

    printf("  %9u  %8.0f  %10.0f  %.2fx\n", n, cub_us, thrust_us, thrust_us / cub_us);
  }
}

BOOST_AUTO_TEST_CASE(bench_cub_vs_thrust_arity3) {
  std::cout << "\n=== Microbenchmark: CUB vs Thrust, arity-3, order {0,1,2} ===" << std::endl;
  std::cout << "    N        CUB(us)   Thrust(us)  Ratio" << std::endl;

  for (uint32_t n : {100u, 1000u, 10000u, 100000u, 1000000u, 5000000u}) {
    auto c0 = rand_u32(n, 0, n);
    auto c1 = rand_u32(n, 0, n);
    auto c2 = rand_u32(n, 0, n);
    std::vector<int> order = {0, 1, 2};

    // Warmup
    {
      auto d0 = h2d(c0);
      auto d1 = h2d(c1);
      auto d2 = h2d(c2);
      std::vector<uint32_t*> ptrs = {d0.data(), d1.data(), d2.data()};
      CubSortStorage stor;
      cub_lex_sort(ptrs, n, order, 0, stor);
      GPU_DEVICE_SYNCHRONIZE();
    }

    // CUB timing
    double cub_us = 0;
    {
      constexpr int REPS = 5;
      for (int r = 0; r < REPS; ++r) {
        auto d0 = h2d(c0);
        auto d1 = h2d(c1);
        auto d2 = h2d(c2);
        std::vector<uint32_t*> ptrs = {d0.data(), d1.data(), d2.data()};
        CubSortStorage stor;
        GPU_DEVICE_SYNCHRONIZE();
        auto t0 = std::chrono::high_resolution_clock::now();
        cub_lex_sort(ptrs, n, order, 0, stor);
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        cub_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      cub_us /= REPS;
    }

    // Thrust timing (multi-pass stable_lex_sort)
    double thrust_us = 0;
    {
      constexpr int REPS = 5;
      for (int r = 0; r < REPS; ++r) {
        auto d0 = h2d(c0);
        auto d1 = h2d(c1);
        auto d2 = h2d(c2);
        GPU_DEVICE_SYNCHRONIZE();
        auto t0 = std::chrono::high_resolution_clock::now();
        SRDatalog::GPU::zip_sort_columns(n, d0.data(), d1.data(), d2.data());
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        thrust_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      thrust_us /= REPS;
    }

    printf("  %9u  %8.0f  %10.0f  %.2fx\n", n, cub_us, thrust_us, thrust_us / cub_us);
  }
}

BOOST_AUTO_TEST_CASE(bench_stream_parallel_benefit) {
  std::cout << "\n=== Microbenchmark: 4 sorts sequential vs parallel (arity-3) ===" << std::endl;
  std::cout << "    N        Seq(us)   Par(us)    Speedup" << std::endl;

  for (uint32_t n : {1000u, 10000u, 100000u, 1000000u, 5000000u}) {
    auto c0 = rand_u32(n, 0, n);
    auto c1 = rand_u32(n, 0, n);
    auto c2 = rand_u32(n, 0, n);
    int orders[][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

    // Sequential: 4 sorts on stream 0
    double seq_us = 0;
    {
      constexpr int REPS = 3;
      for (int r = 0; r < REPS; ++r) {
        // Create 4 copies
        DeviceArray<uint32_t> d0[4], d1[4], d2[4];
        CubSortStorage stor[4];
        for (int j = 0; j < 4; ++j) {
          d0[j] = h2d(c0);
          d1[j] = h2d(c1);
          d2[j] = h2d(c2);
        }
        GPU_DEVICE_SYNCHRONIZE();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 4; ++j) {
          std::vector<uint32_t*> ptrs = {d0[j].data(), d1[j].data(), d2[j].data()};
          std::vector<int> order = {orders[j][0], orders[j][1], orders[j][2]};
          cub_lex_sort(ptrs, n, order, 0, stor[j]);
        }
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      seq_us /= REPS;
    }

    // Parallel: 4 sorts on 4 different streams
    double par_us = 0;
    {
      GPU_STREAM_T streams[4];
      for (auto& s : streams)
        GPU_STREAM_CREATE(&s);

      constexpr int REPS = 3;
      for (int r = 0; r < REPS; ++r) {
        DeviceArray<uint32_t> d0[4], d1[4], d2[4];
        CubSortStorage stor[4];
        for (int j = 0; j < 4; ++j) {
          d0[j] = h2d(c0);
          d1[j] = h2d(c1);
          d2[j] = h2d(c2);
        }
        GPU_DEVICE_SYNCHRONIZE();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 4; ++j) {
          std::vector<uint32_t*> ptrs = {d0[j].data(), d1[j].data(), d2[j].data()};
          std::vector<int> order = {orders[j][0], orders[j][1], orders[j][2]};
          cub_lex_sort(ptrs, n, order, streams[j], stor[j]);
        }
        for (auto& s : streams)
          GPU_STREAM_SYNCHRONIZE(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        par_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      par_us /= REPS;

      for (auto& s : streams)
        GPU_STREAM_DESTROY(s);
    }

    printf("  %9u  %8.0f  %8.0f    %.2fx\n", n, seq_us, par_us, seq_us / par_us);
  }
}

// Test: does Thrust zip_sort_columns parallelize across streams natively?
BOOST_AUTO_TEST_CASE(bench_thrust_stream_parallel) {
  std::cout << "\n=== Thrust zip_sort: sequential vs multi-stream (arity-3) ===" << std::endl;
  std::cout << "    N        Seq(us)   Par(us)    Speedup" << std::endl;

  for (uint32_t n : {1000u, 10000u, 100000u, 1000000u, 5000000u}) {
    auto c0 = rand_u32(n, 0, n);
    auto c1 = rand_u32(n, 0, n);
    auto c2 = rand_u32(n, 0, n);

    // Sequential: 4 Thrust sorts on default stream
    double seq_us = 0;
    {
      constexpr int REPS = 3;
      for (int r = 0; r < REPS; ++r) {
        DeviceArray<uint32_t> d0[4], d1[4], d2[4];
        for (int j = 0; j < 4; ++j) {
          d0[j] = h2d(c0);
          d1[j] = h2d(c1);
          d2[j] = h2d(c2);
        }
        GPU_DEVICE_SYNCHRONIZE();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 4; ++j) {
          SRDatalog::GPU::zip_sort_columns(n, d0[j].data(), d1[j].data(), d2[j].data());
        }
        GPU_DEVICE_SYNCHRONIZE();
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      seq_us /= REPS;
    }

    // Parallel: 4 Thrust sorts on 4 different streams using rmm::exec_policy(stream)
    double par_us = 0;
    {
      GPU_STREAM_T streams[4];
      for (auto& s : streams)
        GPU_STREAM_CREATE(&s);

      constexpr int REPS = 3;
      for (int r = 0; r < REPS; ++r) {
        DeviceArray<uint32_t> d0[4], d1[4], d2[4];
        for (int j = 0; j < 4; ++j) {
          d0[j] = h2d(c0);
          d1[j] = h2d(c1);
          d2[j] = h2d(c2);
        }
        GPU_DEVICE_SYNCHRONIZE();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 4; ++j) {
          auto keys = thrust::make_zip_iterator(
              thrust::make_tuple(thrust::device_ptr<uint32_t>(d0[j].data()),
                                 thrust::device_ptr<uint32_t>(d1[j].data()),
                                 thrust::device_ptr<uint32_t>(d2[j].data())));
          thrust::sort(rmm::exec_policy(streams[j]), keys, keys + n);
        }
        for (auto& s : streams)
          GPU_STREAM_SYNCHRONIZE(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        par_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }
      par_us /= REPS;

      for (auto& s : streams)
        GPU_STREAM_DESTROY(s);
    }

    printf("  %9u  %8.0f  %8.0f    %.2fx\n", n, seq_us, par_us, seq_us / par_us);
  }
}

}  // namespace
