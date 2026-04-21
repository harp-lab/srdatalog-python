#define BOOST_TEST_MODULE gpu_tvjoin_index_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/aos_device_array.h"
#include "gpu/device_array.h"
#include "gpu/device_tvjoin_index.h"
#include "index.h"
#include "semiring.h"
#include <tuple>
#include <vector>

using namespace SRDatalog::GPU;
using SRDatalog::IndexSpec;

// Helper kernel to verify NodeHandle access from wrapper
template <typename NodeHandle>
__global__ void verify_wrapper_handle_kernel(NodeHandle handle, int* count_out) {
  int count = 0;
  // Iterate
  auto end = handle.end();
  for (auto it = handle.begin(); it != end; ++it) {
    count++;
  }
  *count_out = count;
}

BOOST_AUTO_TEST_CASE(test_device_tvjoin_index_basic) {
  // Define Index: T=int, Arity=2, IndexCols={0}
  // SR=DummySR
  using IndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0>;
  IndexType index;

  BOOST_CHECK(index.empty());
  BOOST_CHECK_EQUAL(index.size(), 0);

  // Append data
  AoSDeviceArray<int, 2> data;
  data.push_back(std::make_tuple(10, 1));
  data.push_back(std::make_tuple(20, 2));

  index.append(data);

  BOOST_CHECK_EQUAL(index.size(), 2);
  BOOST_CHECK(!index.empty());

  // Verify clear
  // index.clear();
  // BOOST_CHECK(index.empty());
}

BOOST_AUTO_TEST_CASE(test_device_tvjoin_index_handle) {
  using IndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0>;
  IndexType index;

  AoSDeviceArray<int, 2> data;
  for (int i = 0; i < 100; ++i) {
    data.push_back(std::make_tuple(i, i));
  }
  index.append(data);

  // Get Handle via root()
  auto handle = index.root();

  int* d_count;
  cudaMalloc(&d_count, sizeof(int));
  cudaMemset(d_count, 0, sizeof(int));

  verify_wrapper_handle_kernel<<<1, 1>>>(handle, d_count);
  GPU_CHECK_ERROR(cudaDeviceSynchronize());

  int h_count;
  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_count, 100);

  cudaFree(d_count);
}

BOOST_AUTO_TEST_CASE(test_device_tvjoin_set_difference_update) {
  // Test set_difference_update with BooleanSR
  // Full: { (1,2), (3,4) }
  // Newt: { (1,2), (5,6) }  -- (1,2) exists, (5,6) is new
  // Delta should be { (5,6) }

  using IndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0, 1>;

  // Create Full index
  IndexType full_idx;
  {
    AoSDeviceArray<int, 2> full_data;
    full_data.push_back(std::make_tuple(1, 2));
    full_data.push_back(std::make_tuple(3, 4));
    full_idx.append(full_data);
  }
  BOOST_CHECK_EQUAL(full_idx.size(), 2);

  // Create Newt index (has one duplicate, one new)
  IndexType newt_idx;
  {
    AoSDeviceArray<int, 2> newt_data;
    newt_data.push_back(std::make_tuple(1, 2));  // Exists in full
    newt_data.push_back(std::make_tuple(5, 6));  // New
    newt_idx.append(newt_data);
  }
  BOOST_CHECK_EQUAL(newt_idx.size(), 2);

  // Compute set difference
  IndexType delta_idx;
  newt_idx.set_difference_update(full_idx, delta_idx);

  // Delta should have only the new tuple
  BOOST_CHECK_EQUAL(delta_idx.size(), 1);

  // Verify the content by reconstructing
  AoSDeviceArray<int, 2> delta_aos(delta_idx.size());
  DeviceArray<semiring_value_t<BooleanSR>> delta_prov;
  delta_prov.resize(delta_idx.size());
  IndexSpec spec{{0, 1}};
  delta_idx.reconstruct_to_relation(spec, delta_aos, delta_prov);

  // Read back to host
  std::vector<int> h_delta(delta_idx.size() * 2);
  cudaMemcpy(h_delta.data(), delta_aos.data(), h_delta.size() * sizeof(int),
             cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_delta[0], 5);  // First column of (5,6)
  BOOST_CHECK_EQUAL(h_delta[1], 6);  // Second column of (5,6)

  std::cout << "set_difference_update test passed: delta = {(" << h_delta[0] << "," << h_delta[1]
            << ")}" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_device_tvjoin_set_difference_empty_full) {
  // Test when full is empty - all of newt should become delta
  using IndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0, 1>;

  IndexType full_idx;  // Empty
  BOOST_CHECK(full_idx.empty());

  IndexType newt_idx;
  {
    AoSDeviceArray<int, 2> newt_data;
    newt_data.push_back(std::make_tuple(1, 2));
    newt_data.push_back(std::make_tuple(3, 4));
    newt_idx.append(newt_data);
  }
  BOOST_CHECK_EQUAL(newt_idx.size(), 2);

  IndexType delta_idx;
  newt_idx.set_difference_update(full_idx, delta_idx);

  // All of newt should be in delta
  BOOST_CHECK_EQUAL(delta_idx.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_device_tvjoin_set_difference_all_exist) {
  // Test when all newt tuples exist in full - delta should be empty
  using IndexType = DeviceTVJoinIndex<BooleanSR, int, 2, 0, 1>;

  IndexType full_idx;
  {
    AoSDeviceArray<int, 2> full_data;
    full_data.push_back(std::make_tuple(1, 2));
    full_data.push_back(std::make_tuple(3, 4));
    full_idx.append(full_data);
  }

  IndexType newt_idx;
  {
    AoSDeviceArray<int, 2> newt_data;
    newt_data.push_back(std::make_tuple(1, 2));  // Exists
    newt_data.push_back(std::make_tuple(3, 4));  // Exists
    newt_idx.append(newt_data);
  }

  IndexType delta_idx;
  newt_idx.set_difference_update(full_idx, delta_idx);

  // Delta should be empty
  BOOST_CHECK(delta_idx.empty());
}
