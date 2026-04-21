
#define BOOST_TEST_MODULE aos_device_array_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/aos_device_array.h"
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <tuple>
#include <vector>

using namespace SRDatalog::GPU;

// Helper to copy from device to host
template <typename T, std::size_t N>
std::vector<T> to_host(const AoSDeviceArray<T, N>& device_arr) {
  std::size_t size = device_arr.num_rows() * N;
  std::vector<T> host_vec(size);
  if (size > 0) {
    GPU_MEMCPY(host_vec.data(), device_arr.data(), size * sizeof(T), GPU_DEVICE_TO_HOST);
  }
  return host_vec;
}

// Check CUDA availability fixture
struct CudaCheckFixture {
  CudaCheckFixture() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      BOOST_TEST_MESSAGE("No CUDA device available, skipping tests");
      // In Boost Test, we can't easily skip all tests globally from a fixture constructor
      // without throwing an exception or aborting.
      // Individual tests should check if they can run, or we assume test environment has GPU.
    }
  }
};

BOOST_GLOBAL_FIXTURE(CudaCheckFixture);

BOOST_AUTO_TEST_CASE(test_basic_construction) {
  AoSDeviceArray<int, 2> arr(10);
  BOOST_CHECK_EQUAL(arr.num_rows(), 10);
  BOOST_CHECK_GE(arr.capacity(), 10);
  BOOST_CHECK(arr.data() != nullptr);
}

BOOST_AUTO_TEST_CASE(test_push_back_row) {
  AoSDeviceArray<int, 2> arr;

  int row1[] = {1, 2};
  int row2[] = {3, 4};

  arr.push_back(static_cast<const int*>(row1));
  arr.push_back(static_cast<const int*>(row2));

  BOOST_CHECK_EQUAL(arr.num_rows(), 2);

  auto host_data = to_host(arr);
  BOOST_REQUIRE_EQUAL(host_data.size(), 4);
  BOOST_CHECK_EQUAL(host_data[0], 1);
  BOOST_CHECK_EQUAL(host_data[1], 2);
  BOOST_CHECK_EQUAL(host_data[2], 3);
  BOOST_CHECK_EQUAL(host_data[3], 4);
}

BOOST_AUTO_TEST_CASE(test_push_back_tuple) {
  AoSDeviceArray<int, 3> arr;

  arr.push_back(std::make_tuple(10, 20, 30));
  arr.push_back(std::make_tuple(40, 50, 60));

  BOOST_CHECK_EQUAL(arr.num_rows(), 2);

  auto host_data = to_host(arr);
  BOOST_REQUIRE_EQUAL(host_data.size(), 6);
  BOOST_CHECK_EQUAL(host_data[0], 10);
  BOOST_CHECK_EQUAL(host_data[1], 20);
  BOOST_CHECK_EQUAL(host_data[2], 30);
  BOOST_CHECK_EQUAL(host_data[3], 40);
  BOOST_CHECK_EQUAL(host_data[4], 50);
  BOOST_CHECK_EQUAL(host_data[5], 60);
}

BOOST_AUTO_TEST_CASE(test_resize) {
  AoSDeviceArray<int, 2> arr;
  arr.push_back(std::make_tuple(1, 1));
  arr.push_back(std::make_tuple(2, 2));

  // Grow
  arr.resize(4);
  BOOST_CHECK_EQUAL(arr.num_rows(), 4);
  BOOST_CHECK_GE(arr.capacity(), 4);

  // Check old data preserved
  auto host_data = to_host(arr);
  BOOST_CHECK_EQUAL(host_data[0], 1);
  BOOST_CHECK_EQUAL(host_data[1], 1);
  BOOST_CHECK_EQUAL(host_data[2], 2);
  BOOST_CHECK_EQUAL(host_data[3], 2);

  // Shrink
  arr.resize(1);
  BOOST_CHECK_EQUAL(arr.num_rows(), 1);
  host_data = to_host(arr);
  BOOST_REQUIRE_EQUAL(host_data.size(), 2);  // 1 row * 2 cols
  BOOST_CHECK_EQUAL(host_data[0], 1);
  BOOST_CHECK_EQUAL(host_data[1], 1);
}

BOOST_AUTO_TEST_CASE(test_reserve) {
  AoSDeviceArray<int, 2> arr;
  arr.reserve(100);
  BOOST_CHECK_GE(arr.capacity(), 100);
  BOOST_CHECK_EQUAL(arr.num_rows(), 0);

  arr.push_back(std::make_tuple(1, 2));
  BOOST_CHECK_EQUAL(arr.num_rows(), 1);
  // Capacity should ideally remain unchanged if it was large enough
  BOOST_CHECK_GE(arr.capacity(), 100);
}

BOOST_AUTO_TEST_CASE(test_copy_clone) {
  AoSDeviceArray<int, 2> arr;
  arr.push_back(std::make_tuple(1, 2));
  arr.push_back(std::make_tuple(3, 4));

  AoSDeviceArray<int, 2> copy;
  arr.clone_into(copy);

  BOOST_CHECK_EQUAL(copy.num_rows(), 2);
  auto host_data = to_host(copy);
  BOOST_CHECK_EQUAL(host_data[0], 1);
  BOOST_CHECK_EQUAL(host_data[1], 2);
  BOOST_CHECK_EQUAL(host_data[2], 3);
  BOOST_CHECK_EQUAL(host_data[3], 4);
}

BOOST_AUTO_TEST_CASE(test_concat) {
  AoSDeviceArray<int, 2> arr1;
  arr1.push_back(std::make_tuple(1, 2));

  AoSDeviceArray<int, 2> arr2;
  arr2.push_back(std::make_tuple(3, 4));
  arr2.push_back(std::make_tuple(5, 6));

  arr1.concat(arr2);

  BOOST_CHECK_EQUAL(arr1.num_rows(), 3);
  auto host_data = to_host(arr1);
  BOOST_REQUIRE_EQUAL(host_data.size(), 6);
  BOOST_CHECK_EQUAL(host_data[0], 1);
  BOOST_CHECK_EQUAL(host_data[1], 2);
  BOOST_CHECK_EQUAL(host_data[2], 3);
  BOOST_CHECK_EQUAL(host_data[3], 4);
  BOOST_CHECK_EQUAL(host_data[4], 5);
  BOOST_CHECK_EQUAL(host_data[5], 6);
}
