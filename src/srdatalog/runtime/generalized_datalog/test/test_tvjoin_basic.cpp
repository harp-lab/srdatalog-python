
#define BOOST_TEST_MODULE tvjoin_basic_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/adapter/tvjoin_adapter.h"
#include "gpu/aos_device_array.h"
#include <vector>

using namespace SRDatalog::GPU;

// Check CUDA availability fixture (standard boilerplate)
struct CudaCheckFixture {
  CudaCheckFixture() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      BOOST_TEST_MESSAGE("No CUDA device available, skipping tests");
    }
  }
};

BOOST_GLOBAL_FIXTURE(CudaCheckFixture);

BOOST_AUTO_TEST_CASE(test_instantiation) {
  // Test instantiating an index on Arity=2, IndexCols={0}
  TVJoinIndex<int, 2, 0> index;
  BOOST_CHECK_EQUAL(index.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_append) {
  TVJoinIndex<int, 2, 0> index;

  // Create AoS data
  AoSDeviceArray<int, 2> aos_data;
  aos_data.push_back(std::make_tuple(10, 20));
  aos_data.push_back(std::make_tuple(30, 40));
  aos_data.push_back(std::make_tuple(10, 20));  // Duplicate

  // Append to index
  index.append(aos_data);

  // Check size
  // TVJoin insert does self-deduplication.
  // (10, 20) is duplicated, so we expect 2 unique tuples.
  BOOST_CHECK_EQUAL(index.size(), 2);

  // Append more data
  AoSDeviceArray<int, 2> more_data;
  more_data.push_back(std::make_tuple(50, 60));
  index.append(more_data);

  BOOST_CHECK_EQUAL(index.size(), 3);
}

BOOST_AUTO_TEST_CASE(test_append_multi_col_index) {
  // Test index on columns {1, 0} (reverse order)
  TVJoinIndex<int, 2, 1, 0> index;

  AoSDeviceArray<int, 2> aos_data;
  aos_data.push_back(std::make_tuple(10, 20));
  aos_data.push_back(std::make_tuple(30, 40));

  index.append(aos_data);
  BOOST_CHECK_EQUAL(index.size(), 2);
}

// Kernel to verify iteration
template <typename Handle>
__global__ void verify_handle_kernel(Handle handle, int* count_out, long long* sum_out) {
  int count = 0;
  long long sum = 0;
  for (auto it = handle.begin(); it != handle.end(); ++it) {
    int* tuple = *it;
    // tuple points to [col0, col1]
    sum += (long long)tuple[0] + tuple[1];
    count++;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *count_out = count;
    *sum_out = sum;
  }
}

BOOST_AUTO_TEST_CASE(test_node_handle_iteration) {
  TVJoinIndex<int, 2, 0> index;
  AoSDeviceArray<int, 2> aos_data;
  // Insert 1000 tuples: (i, i+1)
  for (int i = 0; i < 1000; ++i) {
    aos_data.push_back(std::make_tuple(i, i + 1));
  }
  index.append(aos_data);

  auto handle = index.get_node_handle();

  // Allocate device memory for results
  int* d_count;
  long long* d_sum;
  cudaMalloc(&d_count, sizeof(int));
  cudaMalloc(&d_sum, sizeof(long long));
  cudaMemset(d_count, 0, sizeof(int));
  cudaMemset(d_sum, 0, sizeof(long long));

  verify_handle_kernel<<<1, 1>>>(handle, d_count, d_sum);
  GPU_CHECK_ERROR(cudaDeviceSynchronize());

  int h_count;
  long long h_sum;
  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);

  cudaFree(d_count);
  cudaFree(d_sum);

  BOOST_CHECK_EQUAL(h_count, 1000);
  // Expected Sum calculation:
  // sum(i) for i in 0..999 = 999*1000/2 = 499500
  // sum(i+1) = sum(i) + 1000 = 500500
  // Total = 499500 + 500500 = 1000000
  BOOST_CHECK_EQUAL(h_sum, 1000000);
}

// Kernel to verify search (lower_bound)
template <typename Handle, typename Key>
__global__ void verify_search_kernel(Handle handle, Key key, int* count_out, long long* sum_out) {
  int count = 0;
  long long sum = 0;
  // Perform search ON DEVICE
  auto it = handle.lower_bound(key);
  auto end = handle.end();

  for (; it != end; ++it) {
    int* tuple = *it;
    sum += (long long)tuple[0] + tuple[1];
    count++;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *count_out = count;
    *sum_out = sum;
  }
}

BOOST_AUTO_TEST_CASE(test_lower_bound) {
  TVJoinIndex<int, 2, 0> index;
  AoSDeviceArray<int, 2> aos_data;

  aos_data.push_back(std::make_tuple(10, 1));
  aos_data.push_back(std::make_tuple(20, 1));
  aos_data.push_back(std::make_tuple(30, 1));
  index.append(aos_data);

  auto handle = index.get_node_handle();

  int* d_count;
  long long* d_sum;
  cudaMalloc(&d_count, sizeof(int));
  cudaMalloc(&d_sum, sizeof(long long));
  cudaMemset(d_count, 0, sizeof(int));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Search for 20
  verify_search_kernel<<<1, 1>>>(handle, 20, d_count, d_sum);
  GPU_CHECK_ERROR(cudaDeviceSynchronize());

  int h_count;
  long long h_sum;
  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);

  // Expect (20, 1) and (30, 1) -> 2 items. Sum: 21 + 31 = 52.
  BOOST_CHECK_EQUAL(h_count, 2);
  BOOST_CHECK_EQUAL(h_sum, 52);

  cudaFree(d_count);
  cudaFree(d_sum);
}

BOOST_AUTO_TEST_CASE(test_lower_bound_multi_col) {
  // Index on {1, 0}
  TVJoinIndex<int, 2, 1, 0> index;
  AoSDeviceArray<int, 2> aos_data;

  aos_data.push_back(std::make_tuple(10, 20));
  aos_data.push_back(std::make_tuple(30, 40));
  aos_data.push_back(std::make_tuple(5, 40));

  index.append(aos_data);

  auto handle = index.get_node_handle();

  int* d_count;
  long long* d_sum;
  cudaMalloc(&d_count, sizeof(int));
  cudaMalloc(&d_sum, sizeof(long long));
  cudaMemset(d_count, 0, sizeof(int));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Search for 30 on Primary Key (Col 1)
  verify_search_kernel<<<1, 1>>>(handle, 30, d_count, d_sum);
  GPU_CHECK_ERROR(cudaDeviceSynchronize());

  int h_count;
  long long h_sum;
  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);

  cudaFree(d_count);
  cudaFree(d_sum);

  // Expect (5, 40) and (30, 40) (since 40 >= 30).
  // Sum: 45 + 70 = 115.

  BOOST_CHECK_EQUAL(h_count, 2);
  BOOST_CHECK_EQUAL(h_sum, 115);
}
