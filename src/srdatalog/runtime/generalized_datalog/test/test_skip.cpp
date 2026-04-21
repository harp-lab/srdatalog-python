#define BOOST_TEST_MODULE skip_duplicates_test
#include <boost/test/included/unit_test.hpp>

#include "skip.h"
#include "system.h"  // For AlignedVector and make_aligned_vector

#include <algorithm>
#include <vector>
#include <random>

using SRDatalog::skip_duplicates;
using SRDatalog::AlignedVector;
using SRDatalog::make_aligned_vector;

namespace {

// Helper function to verify skip_duplicates result
template <typename T>
void verify_skip_result(const T* original_ptr, const T* result_ptr, const T* end_ptr, 
                        const T expected_value) {
  // Result should be within bounds
  BOOST_REQUIRE_GE(result_ptr, original_ptr);
  BOOST_REQUIRE_LE(result_ptr, end_ptr);
  
  // All skipped elements should equal expected_value
  for (const T* p = original_ptr; p < result_ptr; ++p) {
    BOOST_REQUIRE_EQUAL(*p, expected_value);
  }
  
  // The result pointer should point to first non-duplicate (or end)
  if (result_ptr < end_ptr) {
    BOOST_REQUIRE_NE(*result_ptr, expected_value);
  }
}

// Test 1: Basic functionality - skip duplicates
BOOST_AUTO_TEST_CASE(test_basic_skip_duplicates) {
  std::vector<int32_t> data = {1, 1, 1, 2, 3, 4, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 3);  // Should skip 3 ones
  BOOST_REQUIRE_EQUAL(*result, 2);       // Should point to first non-duplicate
}

// Test 2: All duplicates - should return end
BOOST_AUTO_TEST_CASE(test_all_duplicates) {
  std::vector<int32_t> data = {5, 5, 5, 5, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, end);  // Should return end pointer
}

// Test 3: No duplicates - should return ptr + 1
BOOST_AUTO_TEST_CASE(test_no_duplicates) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, ptr + 1);  // Should skip only the first element
  BOOST_REQUIRE_EQUAL(*result, 2);
}

// Test 4: Single element - should return end
BOOST_AUTO_TEST_CASE(test_single_element) {
  std::vector<int32_t> data = {42};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, end);
}

// Test 5: Empty array - should return end
BOOST_AUTO_TEST_CASE(test_empty_array) {
  std::vector<int32_t> data;
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, end);
}

// Test 6: Many duplicates - test SIMD path
BOOST_AUTO_TEST_CASE(test_many_duplicates) {
  const size_t n = 1000;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  // Fill with many duplicates followed by different values
  for (size_t i = 0; i < 500; ++i) {
    data[i] = 7;
  }
  for (size_t i = 500; i < n; ++i) {
    data[i] = static_cast<int32_t>(i);
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 500);  // Should skip 500 sevens
  BOOST_REQUIRE_EQUAL(*result, 500);
  
  verify_skip_result(ptr, result, end, 7);
}

// Test 7: Duplicates at the end
BOOST_AUTO_TEST_CASE(test_duplicates_at_end) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 5, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, ptr + 1);  // Should skip only first element
  BOOST_REQUIRE_EQUAL(*result, 2);
}

// Test 8: Different integer types - int64_t
BOOST_AUTO_TEST_CASE(test_int64_duplicates) {
  std::vector<int64_t> data = {100LL, 100LL, 100LL, 200LL, 300LL};
  const int64_t* ptr = data.data();
  const int64_t* end = data.data() + data.size();
  
  const int64_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 3);
  BOOST_REQUIRE_EQUAL(*result, 200LL);
}

// Test 9: Different integer types - uint32_t
BOOST_AUTO_TEST_CASE(test_uint32_duplicates) {
  std::vector<uint32_t> data = {42U, 42U, 42U, 43U, 44U};
  const uint32_t* ptr = data.data();
  const uint32_t* end = data.data() + data.size();
  
  const uint32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 3);
  BOOST_REQUIRE_EQUAL(*result, 43U);
}

// Test 10: Large array with SIMD-sized chunks
BOOST_AUTO_TEST_CASE(test_large_array_simd_chunks) {
  const size_t n = 10000;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  // Fill with duplicates in SIMD-sized chunks
  const int32_t target_val = 99;
  for (size_t i = 0; i < n; ++i) {
    if (i < 8000) {
      data[i] = target_val;
    } else {
      data[i] = static_cast<int32_t>(i);
    }
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 8000);
  BOOST_REQUIRE_EQUAL(*result, 8000);
  
  verify_skip_result(ptr, result, end, target_val);
}

// Test 11: Duplicates spanning multiple SIMD vectors
BOOST_AUTO_TEST_CASE(test_duplicates_spanning_simd_vectors) {
  const size_t n = 500;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  // Fill all with same value to test SIMD optimization
  const int32_t target_val = 123;
  for (size_t i = 0; i < n; ++i) {
    data[i] = target_val;
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result, end);  // All are duplicates, should return end
}

// Test 12: Duplicates with mismatch in middle of SIMD vector
BOOST_AUTO_TEST_CASE(test_mismatch_in_simd_vector) {
  const size_t n = 100;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  const int32_t target_val = 50;
  // Fill first 20 with duplicates
  for (size_t i = 0; i < 20; ++i) {
    data[i] = target_val;
  }
  // Then different values
  for (size_t i = 20; i < n; ++i) {
    data[i] = static_cast<int32_t>(i);
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 20);
  BOOST_REQUIRE_EQUAL(*result, 20);
  
  verify_skip_result(ptr, result, end, target_val);
}

// Test 13: Tail handling - small remainder after SIMD chunks
BOOST_AUTO_TEST_CASE(test_tail_handling) {
  // Create array size that leaves a small tail after SIMD processing
  const size_t simd_lanes = 16;  // Approximate for AVX2
  const size_t n = simd_lanes * 10 + 5;  // 10 full vectors + 5 tail elements
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  const int32_t target_val = 77;
  // Fill all but last 3 with duplicates
  for (size_t i = 0; i < n - 3; ++i) {
    data[i] = target_val;
  }
  // Last 3 are different
  for (size_t i = n - 3; i < n; ++i) {
    data[i] = static_cast<int32_t>(i);
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, n - 3);
  BOOST_REQUIRE_EQUAL(*result, static_cast<int32_t>(n - 3));
  
  verify_skip_result(ptr, result, end, target_val);
}

// Test 14: Negative values
BOOST_AUTO_TEST_CASE(test_negative_values) {
  std::vector<int32_t> data = {-10, -10, -10, -5, 0, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 3);
  BOOST_REQUIRE_EQUAL(*result, -5);
}

// Test 15: Zero values
BOOST_AUTO_TEST_CASE(test_zero_values) {
  std::vector<int32_t> data = {0, 0, 0, 1, 2, 3};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 3);
  BOOST_REQUIRE_EQUAL(*result, 1);
}

// Test 16: Very small array (smaller than SIMD width)
BOOST_AUTO_TEST_CASE(test_very_small_array) {
  std::vector<int32_t> data = {1, 1, 2};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, 2);
  BOOST_REQUIRE_EQUAL(*result, 2);
}

// Test 17: Random test with verification
BOOST_AUTO_TEST_CASE(test_random_with_verification) {
  const size_t n = 2000;
  AlignedVector<int32_t> data = make_aligned_vector<int32_t>(n);
  
  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, 100);
  
  // Generate sorted array with duplicates
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(gen);
  }
  std::sort(data.data(), data.data() + n);
  
  // Find first value and count duplicates
  const int32_t first_val = data[0];
  size_t duplicate_count = 0;
  for (size_t i = 0; i < n && data[i] == first_val; ++i) {
    duplicate_count++;
  }
  
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + n;
  
  const int32_t* result = skip_duplicates(ptr, end);
  
  BOOST_REQUIRE_EQUAL(result - ptr, static_cast<ptrdiff_t>(duplicate_count));
  
  if (result < end) {
    BOOST_REQUIRE_NE(*result, first_val);
  }
  
  verify_skip_result(ptr, result, end, first_val);
}

// Test 18: Multiple calls on same array (pointer advancement)
BOOST_AUTO_TEST_CASE(test_multiple_calls) {
  std::vector<int32_t> data = {1, 1, 2, 2, 2, 3, 3, 4, 5};
  const int32_t* ptr = data.data();
  const int32_t* end = data.data() + data.size();
  
  // First skip: should skip the 1s
  const int32_t* result1 = skip_duplicates(ptr, end);
  BOOST_REQUIRE_EQUAL(result1 - ptr, 2);
  BOOST_REQUIRE_EQUAL(*result1, 2);
  
  // Second skip: should skip the 2s
  const int32_t* result2 = skip_duplicates(result1, end);
  BOOST_REQUIRE_EQUAL(result2 - result1, 3);
  BOOST_REQUIRE_EQUAL(*result2, 3);
  
  // Third skip: should skip the 3s
  const int32_t* result3 = skip_duplicates(result2, end);
  BOOST_REQUIRE_EQUAL(result3 - result2, 2);
  BOOST_REQUIRE_EQUAL(*result3, 4);
  
  // Fourth skip: should skip the 4
  const int32_t* result4 = skip_duplicates(result3, end);
  BOOST_REQUIRE_EQUAL(result4 - result3, 1);
  BOOST_REQUIRE_EQUAL(*result4, 5);
  
  // Fifth skip: should skip the 5
  const int32_t* result5 = skip_duplicates(result4, end);
  BOOST_REQUIRE_EQUAL(result5, end);
}

}  // namespace

