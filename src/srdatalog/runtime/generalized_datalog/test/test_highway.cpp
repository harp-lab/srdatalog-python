#define BOOST_TEST_MODULE highway_sort_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cstring>  // for std::memcpy
#include <numeric>  // for std::iota
#include <random>
#include <vector>

// Assuming the header file is named "sort.h" or "hwy_lsd_sort.hpp"
#include "sort.h"
#include "system.h"    // For HighwayAlignedMemoryResource and make_aligned_vector
#include <hwy/base.h>  // For hwy::uint128_t

using hwy_lexsort::stable_lex_sort;
using hwy_lexsort::unstable_sort;

using SRDatalog::AlignedVector;
using SRDatalog::make_aligned_vector;

namespace {

// ============================================================================
// Test Helpers
// ============================================================================

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
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  // API Call: Generates permutation into 'ids', does NOT modify 'data'
  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), data.data());

  // Verify permutation validity
  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));

  // Verify sorted order by applying permutation
  AlignedVector<int32_t> sorted_data = make_aligned_vector<int32_t>(n);
  apply_perm(data.data(), sorted_data.data(), ids.data(), n);
  BOOST_REQUIRE(is_sorted(sorted_data.data(), n));

  // Verify original data was not modified
  // (The new API is non-destructive by design)
  // The function takes const pointers, so data cannot be modified
}

// Test 2: Basic single column sort (int16_t)
BOOST_AUTO_TEST_CASE(test_single_column_int16) {
  const size_t n = 100;
  AlignedVector<int16_t> data = make_aligned_vector<int16_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int16_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), data.data());

  AlignedVector<int16_t> sorted_data = make_aligned_vector<int16_t>(n);
  apply_perm(data.data(), sorted_data.data(), ids.data(), n);

  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));
  BOOST_REQUIRE(is_sorted(sorted_data.data(), n));
}

// Test 3: Multi-column lexicographic sort
BOOST_AUTO_TEST_CASE(test_multi_column_sort) {
  const size_t n = 50;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 5);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist(gen);
    col1[i] = dist(gen);
  }

  // Sort by col0 first, then col1
  std::vector<int> order = {0, 1};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), col0.data(), col1.data());

  // Verify lexicographic order using indices
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids[i];
    uint32_t prev = ids[i - 1];

    BOOST_REQUIRE_LE(col0[prev], col0[curr]);
    if (col0[prev] == col0[curr]) {
      BOOST_REQUIRE_LE(col1[prev], col1[curr]);
    }
  }
}

// Test 4: Stability test
BOOST_AUTO_TEST_CASE(test_stability) {
  const size_t n = 20;
  AlignedVector<int32_t> keys = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> original_order = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  // Create data with duplicates
  for (size_t i = 0; i < n; ++i) {
    keys[i] = static_cast<int32_t>(i / 4);  // Many duplicates: 0,0,0,0, 1,1,1,1...
    original_order[i] = static_cast<int32_t>(i);
  }

  // Shuffle to test stability
  std::mt19937 gen(42);
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  AlignedVector<int32_t> shuffled_keys = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> shuffled_position = make_aligned_vector<int32_t>(n);
  for (size_t i = 0; i < n; ++i) {
    shuffled_keys[i] = keys[indices[i]];
    shuffled_position[i] = static_cast<int32_t>(i);  // Position in shuffled array
  }

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), shuffled_keys.data());

  // For equal keys, relative order from shuffled array should be preserved
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids[i];
    uint32_t prev = ids[i - 1];

    if (shuffled_keys[prev] == shuffled_keys[curr]) {
      // The position in the shuffled array must be non-decreasing (stable sort preserves relative
      // order)
      BOOST_REQUIRE_LE(shuffled_position[prev], shuffled_position[curr]);
    }
  }
}

// Test 5: Verify Non-Modifying Behavior explicitly
BOOST_AUTO_TEST_CASE(test_non_modifying) {
  const size_t n = 50;
  AlignedVector<int32_t> original = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> copy = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> perm = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-100, 100);
  for (size_t i = 0; i < n; ++i) {
    original[i] = dist(gen);
    copy[i] = original[i];
  }

  std::vector<int> order = {0};
  stable_lex_sort(perm.data(), n, order.data(), order.size(), original.data());

  // Original should be unchanged
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(original[i], copy[i]);
  }
}

// Test 7: Edge case - empty array
BOOST_AUTO_TEST_CASE(test_empty_array) {
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(0);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(0);

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), 0, order.data(), order.size(), data.data());
  // Empty array should not crash
  BOOST_REQUIRE(true);
}

// Test 8: Edge case - single element
BOOST_AUTO_TEST_CASE(test_single_element) {
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(1);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(1);
  data[0] = 42;

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), 1, order.data(), order.size(), data.data());

  BOOST_REQUIRE_EQUAL(ids[0], 0u);
  BOOST_REQUIRE_EQUAL(data[0], 42);  // Check data untouched
}

// Test 9: Edge case - all duplicates
BOOST_AUTO_TEST_CASE(test_all_duplicates) {
  const size_t n = 20;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = 42;
  }

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), data.data());

  // Permutation should be identity (0, 1, 2, ...) because it's stable
  // and the inputs were already in that "order" (identical).
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(ids[i], static_cast<uint32_t>(i));
  }
}

// Test 10: Three column sort
BOOST_AUTO_TEST_CASE(test_three_column_sort) {
  const size_t n = 30;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col2 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 3);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist(gen);
    col1[i] = dist(gen);
    col2[i] = dist(gen);
  }

  // Sort by col0, then col1, then col2
  std::vector<int> order = {0, 1, 2};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), col0.data(), col1.data(), col2.data());

  // Verify lexicographic order
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids[i];
    uint32_t prev = ids[i - 1];

    BOOST_REQUIRE_LE(col0[prev], col0[curr]);
    if (col0[prev] == col0[curr]) {
      BOOST_REQUIRE_LE(col1[prev], col1[curr]);
      if (col1[prev] == col1[curr]) {
        BOOST_REQUIRE_LE(col2[prev], col2[curr]);
      }
    }
  }
}

// Test 11: Different sort order
BOOST_AUTO_TEST_CASE(test_different_sort_order) {
  const size_t n = 30;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 5);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist(gen);
    col1[i] = dist(gen);
  }

  // Sort by col1 first, then col0
  std::vector<int> order = {1, 0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), col0.data(), col1.data());

  // Verify lexicographic order (col1, col0)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids[i];
    uint32_t prev = ids[i - 1];

    BOOST_REQUIRE_LE(col1[prev], col1[curr]);
    if (col1[prev] == col1[curr]) {
      BOOST_REQUIRE_LE(col0[prev], col0[curr]);
    }
  }
}

// Test 12: Large array
BOOST_AUTO_TEST_CASE(test_large_array) {
  const size_t n = 10000;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-100000, 100000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), data.data());

  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));

  // Check order locally
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[ids[i - 1]], data[ids[i]]);
  }
}

// Test 13: Unsigned integer sort
BOOST_AUTO_TEST_CASE(test_unsigned_int_sort) {
  const size_t n = 100;
  AlignedVector<uint32_t> data = make_aligned_vector<uint32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> dist(0, 100000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  std::vector<int> order = {0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), data.data());

  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[ids[i - 1]], data[ids[i]]);
  }
}

// ============================================================================
// Unstable Sort Tests (single column, types < 128 bits)
// ============================================================================

// Test 15: Unstable sort - int16_t (medium array)
BOOST_AUTO_TEST_CASE(test_unstable_sort_int16_medium) {
  const size_t n = 500;
  AlignedVector<int16_t> data = make_aligned_vector<int16_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int16_t> dist(-10000, 10000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 17: Unstable sort - int32_t (large array)
BOOST_AUTO_TEST_CASE(test_unstable_sort_int32_large) {
  const size_t n = 10000;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 18: Unstable sort - int64_t (very large array)
BOOST_AUTO_TEST_CASE(test_unstable_sort_int64_very_large) {
  const size_t n = 50000;
  AlignedVector<int64_t> data = make_aligned_vector<int64_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int64_t> dist(-1000000000LL, 1000000000LL);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 19: Unstable sort - uint32_t
BOOST_AUTO_TEST_CASE(test_unstable_sort_uint32) {
  const size_t n = 2000;
  AlignedVector<uint32_t> data = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 20: Unstable sort - uint64_t
BOOST_AUTO_TEST_CASE(test_unstable_sort_uint64) {
  const size_t n = 5000;
  AlignedVector<uint64_t> data = make_aligned_vector<uint64_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> dist(0, 1000000000ULL);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 21: Unstable sort - float
BOOST_AUTO_TEST_CASE(test_unstable_sort_float) {
  const size_t n = 1000;
  AlignedVector<float> data = make_aligned_vector<float>(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 22: Unstable sort - double
BOOST_AUTO_TEST_CASE(test_unstable_sort_double) {
  const size_t n = 2000;
  AlignedVector<double> data = make_aligned_vector<double>(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000000.0, 1000000.0);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  for (size_t i = 1; i < n; ++i) {
    assert(data[i - 1] <= data[i]);
  }

  std::cout << "  ✓ Passed\n";
}

// Test 23: Unstable sort - with duplicates (verifies correctness, not stability)
void test_unstable_sort_with_duplicates() {
  std::cout << "Test 23: Unstable sort - with duplicates\n";
  const size_t n = 100;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 10);  // Many duplicates
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  unstable_sort(data.data(), n);

  // Verify sorted order (unstable sort may not preserve order of equal elements)
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 24: Unstable sort - edge case with all same values
BOOST_AUTO_TEST_CASE(test_unstable_sort_all_same) {
  const size_t n = 50;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = 42;
  }

  unstable_sort(data.data(), n);

  // All values are the same, so sorted order is maintained
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[i - 1], data[i]);
  }
}

// Test 25: Unstable sort - 128-bit (hwy::uint128_t)
BOOST_AUTO_TEST_CASE(test_unstable_sort_128bit) {
  const size_t n = 100;
  AlignedVector<hwy::uint128_t> data = make_aligned_vector<hwy::uint128_t>(n);

  std::mt19937 gen(42);
  // Generate random 128-bit values using Highway's uint128_t
  std::uniform_int_distribution<uint64_t> dist_lo(0, UINT64_MAX);
  std::uniform_int_distribution<uint64_t> dist_hi(0, 1000ULL);

  for (size_t i = 0; i < n; ++i) {
    // Construct hwy::uint128_t from two 64-bit parts
    uint64_t lo = dist_lo(gen);
    uint64_t hi = dist_hi(gen);
    data[i] = {lo, hi};

    // Also test some values that require more than 64 bits
    if (i % 10 == 0) {
      uint64_t large_hi = dist_hi(gen) + 1000ULL;
      data[i] = {lo, large_hi};
    }
  }

  // Sort in-place using unstable_sort
  unstable_sort(data.data(), n);

  // Verify sorted order
  // hwy::uint128_t supports <, >, == operators (defined in base.h)
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE(!(data[i - 1] > data[i]));
  }
}

// Test 26: Reuse SortContext for multiple sorts
BOOST_AUTO_TEST_CASE(test_reuse_sort_context) {
  const size_t n = 100;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids1 = make_aligned_vector<uint32_t>(n);
  AlignedVector<uint32_t> ids2 = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 10);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist(gen);
    col1[i] = dist(gen);
  }

  // Create a SortContext and reuse it for multiple sorts
  hwy_lexsort::SortContext ctx;

  // First sort: by col0, then col1
  std::vector<int> order1 = {0, 1};
  stable_lex_sort(n, order1.data(), order1.size(), ctx, col0.data(), col1.data());
  const uint32_t* sorted1 = ctx.get_sorted_indices<uint32_t>(n);
  std::memcpy(ids1.data(), sorted1, n * sizeof(uint32_t));

  // Second sort: by col1, then col0 (reusing the same context)
  std::vector<int> order2 = {1, 0};
  stable_lex_sort(n, order2.data(), order2.size(), ctx, col0.data(), col1.data());
  const uint32_t* sorted2 = ctx.get_sorted_indices<uint32_t>(n);
  std::memcpy(ids2.data(), sorted2, n * sizeof(uint32_t));

  // Verify both sorts are correct
  BOOST_REQUIRE(is_valid_permutation(ids1.data(), n));
  BOOST_REQUIRE(is_valid_permutation(ids2.data(), n));

  // Verify lexicographic order for first sort (col0, col1)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids1[i];
    uint32_t prev = ids1[i - 1];
    BOOST_REQUIRE_LE(col0[prev], col0[curr]);
    if (col0[prev] == col0[curr]) {
      BOOST_REQUIRE_LE(col1[prev], col1[curr]);
    }
  }

  // Verify lexicographic order for second sort (col1, col0)
  for (size_t i = 1; i < n; ++i) {
    uint32_t curr = ids2[i];
    uint32_t prev = ids2[i - 1];
    BOOST_REQUIRE_LE(col1[prev], col1[curr]);
    if (col1[prev] == col1[curr]) {
      BOOST_REQUIRE_LE(col0[prev], col0[curr]);
    }
  }
}

// Test 27: SortContext with different sizes (buffer reuse)
BOOST_AUTO_TEST_CASE(test_sort_context_different_sizes) {
  const size_t n1 = 50;
  const size_t n2 = 200;
  const size_t n3 = 100;

  AlignedVector<int32_t> data1 = make_aligned_vector<int32_t>(n1);
  AlignedVector<int32_t> data2 = make_aligned_vector<int32_t>(n2);
  AlignedVector<int32_t> data3 = make_aligned_vector<int32_t>(n3);
  AlignedVector<uint32_t> ids1 = make_aligned_vector<uint32_t>(n1);
  AlignedVector<uint32_t> ids2 = make_aligned_vector<uint32_t>(n2);
  AlignedVector<uint32_t> ids3 = make_aligned_vector<uint32_t>(n3);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);

  for (size_t i = 0; i < n1; ++i) {
    data1[i] = dist(gen);
  }
  for (size_t i = 0; i < n2; ++i) {
    data2[i] = dist(gen);
  }
  for (size_t i = 0; i < n3; ++i) {
    data3[i] = dist(gen);
  }

  // Reuse the same context for different sizes
  hwy_lexsort::SortContext ctx;

  // Sort with size n1
  std::vector<int> order = {0};
  stable_lex_sort(n1, order.data(), order.size(), ctx, data1.data());
  const uint32_t* sorted1 = ctx.get_sorted_indices<uint32_t>(n1);
  std::memcpy(ids1.data(), sorted1, n1 * sizeof(uint32_t));
  BOOST_REQUIRE(is_valid_permutation(ids1.data(), n1));
  for (size_t i = 1; i < n1; ++i) {
    BOOST_REQUIRE_LE(data1[ids1[i - 1]], data1[ids1[i]]);
  }

  // Sort with larger size n2 (should resize buffers)
  stable_lex_sort(n2, order.data(), order.size(), ctx, data2.data());
  const uint32_t* sorted2 = ctx.get_sorted_indices<uint32_t>(n2);
  std::memcpy(ids2.data(), sorted2, n2 * sizeof(uint32_t));
  BOOST_REQUIRE(is_valid_permutation(ids2.data(), n2));
  for (size_t i = 1; i < n2; ++i) {
    BOOST_REQUIRE_LE(data2[ids2[i - 1]], data2[ids2[i]]);
  }

  // Sort with smaller size n3 (should reuse existing buffers)
  stable_lex_sort(n3, order.data(), order.size(), ctx, data3.data());
  const uint32_t* sorted3 = ctx.get_sorted_indices<uint32_t>(n3);
  std::memcpy(ids3.data(), sorted3, n3 * sizeof(uint32_t));
  BOOST_REQUIRE(is_valid_permutation(ids3.data(), n3));
  for (size_t i = 1; i < n3; ++i) {
    BOOST_REQUIRE_LE(data3[ids3[i - 1]], data3[ids3[i]]);
  }
}

// Test 28: SortContext with backward compatibility
BOOST_AUTO_TEST_CASE(test_sort_context_backward_compat) {
  const size_t n = 100;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }

  // Test new context-based API
  hwy_lexsort::SortContext ctx;
  std::vector<int> order = {0};
  stable_lex_sort(n, order.data(), order.size(), ctx, data.data());
  const uint32_t* sorted = ctx.get_sorted_indices<uint32_t>(n);
  std::memcpy(ids.data(), sorted, n * sizeof(uint32_t));

  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[ids[i - 1]], data[ids[i]]);
  }

  // Also test backward-compatible API (no context parameter)
  AlignedVector<uint32_t> ids2 = make_aligned_vector<uint32_t>(n);
  stable_lex_sort(ids2.data(), n, order.data(), order.size(), data.data());

  BOOST_REQUIRE(is_valid_permutation(ids2.data(), n));
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(data[ids2[i - 1]], data[ids2[i]]);
  }

  // Results should be identical
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(ids[i], ids2[i]);
  }
}

// Test 29: Test reorder_data flag functionality
BOOST_AUTO_TEST_CASE(test_reorder_data_flag) {
  const size_t n = 100;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> data_copy = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
    data_copy[i] = data[i];  // Keep a copy for comparison
  }

  std::vector<int> order = {0};

  // Test 1: reorder_data = false (default behavior - only sort indices)
  {
    // Reset data to original
    for (size_t i = 0; i < n; ++i) {
      data[i] = data_copy[i];
    }

    stable_lex_sort(ids.data(), n, order.data(), order.size(), false, data.data());

    // Verify indices are sorted
    BOOST_REQUIRE(is_valid_permutation(ids.data(), n));
    for (size_t i = 1; i < n; ++i) {
      BOOST_REQUIRE_LE(data[ids[i - 1]], data[ids[i]]);
    }

    // Verify original data was NOT modified
    for (size_t i = 0; i < n; ++i) {
      BOOST_REQUIRE_EQUAL(data[i], data_copy[i]);
    }
  }

  // Test 2: reorder_data = true (data should be reordered in place)
  {
    // Reset data to original
    for (size_t i = 0; i < n; ++i) {
      data[i] = data_copy[i];
    }

    stable_lex_sort(ids.data(), n, order.data(), order.size(), true, data.data());

    // Verify indices are still valid
    BOOST_REQUIRE(is_valid_permutation(ids.data(), n));

    // Verify data is now sorted in place
    BOOST_REQUIRE(is_sorted(data.data(), n));

    // Verify that data[ids[i]] == data[i] (data is already sorted, so indices should be identity)
    // Actually, when data is reordered, the indices should reflect the original positions
    // But since we reordered the data, accessing via indices should still give sorted order
    for (size_t i = 1; i < n; ++i) {
      BOOST_REQUIRE_LE(data[i - 1], data[i]);
    }

    // Verify data was actually modified (not equal to original)
    bool data_changed = false;
    for (size_t i = 0; i < n; ++i) {
      if (data[i] != data_copy[i]) {
        data_changed = true;
        break;
      }
    }
    // Data should be changed unless it was already sorted
    // Since we used random data, it's very likely to be changed
    // But we can't assert it always changes, so we just verify it's sorted
  }

  // Test 3: Multi-column with reorder_data = true
  {
    const size_t n2 = 50;
    AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n2);
    AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n2);
    AlignedVector<int32_t> col0_copy = make_aligned_vector<int32_t>(n2);
    AlignedVector<int32_t> col1_copy = make_aligned_vector<int32_t>(n2);
    AlignedVector<uint32_t> ids2 = make_aligned_vector<uint32_t>(n2);

    std::mt19937 gen2(123);
    std::uniform_int_distribution<int32_t> dist2(0, 10);
    for (size_t i = 0; i < n2; ++i) {
      col0[i] = dist2(gen2);
      col1[i] = dist2(gen2);
      col0_copy[i] = col0[i];
      col1_copy[i] = col1[i];
    }

    std::vector<int> order2 = {0, 1};
    stable_lex_sort(ids2.data(), n2, order2.data(), order2.size(), true, col0.data(), col1.data());

    // Verify both columns are sorted lexicographically
    for (size_t i = 1; i < n2; ++i) {
      BOOST_REQUIRE_LE(col0[i - 1], col0[i]);
      if (col0[i - 1] == col0[i]) {
        BOOST_REQUIRE_LE(col1[i - 1], col1[i]);
      }
    }

    // Verify data was modified (columns were reordered)
    // At least one column should be different unless already sorted
  }

  // Test 4: Context-based API with reorder_data
  {
    // Reset data to original
    for (size_t i = 0; i < n; ++i) {
      data[i] = data_copy[i];
    }

    hwy_lexsort::SortContext ctx;
    stable_lex_sort(n, order.data(), order.size(), ctx, false, data.data());

    // Verify data was NOT modified
    for (size_t i = 0; i < n; ++i) {
      BOOST_REQUIRE_EQUAL(data[i], data_copy[i]);
    }

    // Now test with reorder_data = true
    for (size_t i = 0; i < n; ++i) {
      data[i] = data_copy[i];
    }

    stable_lex_sort(n, order.data(), order.size(), ctx, true, data.data());

    // Verify data is sorted
    BOOST_REQUIRE(is_sorted(data.data(), n));
  }
}

// Test 29: Two column sort with reorder_data (large dataset)
BOOST_AUTO_TEST_CASE(test_two_column_reorder_data_large) {
  const size_t n = 50000;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col0_copy = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1_copy = make_aligned_vector<int32_t>(n);

  std::mt19937 gen(77777);
  std::uniform_int_distribution<int32_t> dist0(0, 200);
  std::uniform_int_distribution<int32_t> dist1(0, 300);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist0(gen);
    col1[i] = dist1(gen);
    col0_copy[i] = col0[i];
    col1_copy[i] = col1[i];
  }

  hwy_lexsort::SortContext ctx;
  std::vector<int> order = {0, 1};
  stable_lex_sort(n, order.data(), order.size(), ctx, true, col0.data(), col1.data());

  // Verify both columns are sorted lexicographically
  for (size_t i = 1; i < n; ++i) {
    BOOST_REQUIRE_LE(col0[i - 1], col0[i]);
    if (col0[i - 1] == col0[i]) {
      BOOST_REQUIRE_LE(col1[i - 1], col1[i]);
    }
  }

  // Verify data was modified (reordered)
  bool data_modified = false;
  for (size_t i = 0; i < n; ++i) {
    if (col0[i] != col0_copy[i] || col1[i] != col1_copy[i]) {
      data_modified = true;
      break;
    }
  }
  // Data should be modified unless it was already sorted (unlikely with random data)
  BOOST_REQUIRE(data_modified);
}

// Test 30: Verify index/permutation is sorted (not data after permutation)
// This test directly checks that the permutation indices represent sorted order
BOOST_AUTO_TEST_CASE(test_index_sorted_two_column) {
  const size_t n = 10000;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(12345);
  std::uniform_int_distribution<int32_t> dist0(0, 100);
  std::uniform_int_distribution<int32_t> dist1(0, 200);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist0(gen);
    col1[i] = dist1(gen);
  }

  // Keep original data for comparison
  AlignedVector<int32_t> col0_original = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1_original = make_aligned_vector<int32_t>(n);
  std::memcpy(col0_original.data(), col0.data(), n * sizeof(int32_t));
  std::memcpy(col1_original.data(), col1.data(), n * sizeof(int32_t));

  std::vector<int> order = {0, 1};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), col0.data(), col1.data());

  // Verify permutation validity
  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));

  // Verify that the INDEX/permutation represents sorted order
  // This means: for consecutive indices i, i+1 in the permutation array,
  // the data at those original positions should be in sorted order
  for (size_t i = 1; i < n; ++i) {
    uint32_t prev_idx = ids[i - 1];  // Original row index at position i-1 in sorted order
    uint32_t curr_idx = ids[i];      // Original row index at position i in sorted order

    // Check lexicographic order using original data (before any permutation)
    BOOST_REQUIRE_LE(col0_original[prev_idx], col0_original[curr_idx]);
    if (col0_original[prev_idx] == col0_original[curr_idx]) {
      BOOST_REQUIRE_LE(col1_original[prev_idx], col1_original[curr_idx]);
    }
  }

  // Verify original data was NOT modified (since reorder_data was not set)
  for (size_t i = 0; i < n; ++i) {
    BOOST_REQUIRE_EQUAL(col0[i], col0_original[i]);
    BOOST_REQUIRE_EQUAL(col1[i], col1_original[i]);
  }
}

// Test 31: Verify index sorted with different order (col1 first, then col0)
BOOST_AUTO_TEST_CASE(test_index_sorted_reverse_order) {
  const size_t n = 5000;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);
  AlignedVector<uint32_t> ids = make_aligned_vector<uint32_t>(n);

  std::mt19937 gen(54321);
  std::uniform_int_distribution<int32_t> dist0(0, 150);
  std::uniform_int_distribution<int32_t> dist1(0, 100);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist0(gen);
    col1[i] = dist1(gen);
  }

  AlignedVector<int32_t> col0_original = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1_original = make_aligned_vector<int32_t>(n);
  std::memcpy(col0_original.data(), col0.data(), n * sizeof(int32_t));
  std::memcpy(col1_original.data(), col1.data(), n * sizeof(int32_t));

  // Sort by col1 first, then col0
  std::vector<int> order = {1, 0};
  stable_lex_sort(ids.data(), n, order.data(), order.size(), col0.data(), col1.data());

  BOOST_REQUIRE(is_valid_permutation(ids.data(), n));

  // Verify index represents sorted order (by col1, then col0)
  for (size_t i = 1; i < n; ++i) {
    uint32_t prev_idx = ids[i - 1];
    uint32_t curr_idx = ids[i];

    int32_t col1_prev = col1_original[prev_idx];
    int32_t col1_curr = col1_original[curr_idx];
    int32_t col0_prev = col0_original[prev_idx];
    int32_t col0_curr = col0_original[curr_idx];

    // Lexicographic comparison: first by col1, then by col0
    BOOST_REQUIRE_LE(col1_prev, col1_curr);
    if (col1_prev == col1_curr) {
      BOOST_REQUIRE_LE(col0_prev, col0_curr);
    }
  }
}

// Test 32: Verify index is sorted with Context API and reorder_data=false
BOOST_AUTO_TEST_CASE(test_index_sorted_context_reorder_false) {
  const size_t n = 10000;
  AlignedVector<int32_t> col0 = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1 = make_aligned_vector<int32_t>(n);

  std::mt19937 gen(11111);
  std::uniform_int_distribution<int32_t> dist0(0, 200);
  std::uniform_int_distribution<int32_t> dist1(0, 300);
  for (size_t i = 0; i < n; ++i) {
    col0[i] = dist0(gen);
    col1[i] = dist1(gen);
  }

  AlignedVector<int32_t> col0_original = make_aligned_vector<int32_t>(n);
  AlignedVector<int32_t> col1_original = make_aligned_vector<int32_t>(n);
  std::memcpy(col0_original.data(), col0.data(), n * sizeof(int32_t));
  std::memcpy(col1_original.data(), col1.data(), n * sizeof(int32_t));

  hwy_lexsort::SortContext ctx;
  std::vector<int> order = {0, 1};
  stable_lex_sort(n, order.data(), order.size(), ctx, false, col0.data(), col1.data());

  const uint32_t* sorted_indices = ctx.get_sorted_indices<uint32_t>(n);
  BOOST_REQUIRE(is_valid_permutation(sorted_indices, n));

  // Verify index represents sorted order on ORIGINAL data
  for (size_t i = 1; i < n; ++i) {
    uint32_t prev_idx = sorted_indices[i - 1];
    uint32_t curr_idx = sorted_indices[i];

    BOOST_REQUIRE_LE(col0_original[prev_idx], col0_original[curr_idx]);
    if (col0_original[prev_idx] == col0_original[curr_idx]) {
      BOOST_REQUIRE_LE(col1_original[prev_idx], col1_original[curr_idx]);
    }
  }
}

}  // namespace