// Unit tests for DeviceSortedArrayIndex clone operations
#define BOOST_TEST_MODULE device_sorted_array_index_clone_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/nd_device_array.h"
#include "semiring.h"
#include <thrust/copy.h>
#include <vector>

using SRDatalog::GPU::DeviceArray;
using SRDatalog::GPU::DeviceSortedArrayIndex;
using SRDatalog::GPU::NDDeviceArray;
using BooleanSR = ::BooleanSR;  // BooleanSR is in global namespace
using SRDatalog::IndexSpec;

namespace {

// Helper: Copy host vector to device array
template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  return dev;
}

// Helper: Copy device array to host vector
template <typename T>
std::vector<T> device_to_host(const DeviceArray<T>& dev_array) {
  std::vector<T> host(dev_array.size());
  thrust::copy(dev_array.begin(), dev_array.end(), host.begin());
  return host;
}

// Helper: Synchronize device via a Thrust operation
inline void device_sync_via_thrust() {
  DeviceArray<int> dummy(1);
  thrust::fill(dummy.begin(), dummy.end(), 0);
}

// Helper function to build index from vectors
template <typename SR, typename DSAI>
void build_index_from_vectors(DSAI& idx, const std::vector<uint32_t>& c0,
                              const std::vector<uint32_t>& c1) {
  NDDeviceArray<uint32_t, 2> encoded(c0.size());
  DeviceArray<uint32_t> d0 = host_to_device(c0);
  DeviceArray<uint32_t> d1 = host_to_device(c1);
  thrust::copy(d0.begin(), d0.end(), encoded.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), encoded.column_ptr(1));
  DeviceArray<typename SR::value_type> prov(c0.size());
  idx.build_from_encoded_device(IndexSpec{{0, 1}}, encoded, prov);
  device_sync_via_thrust();
}

// Kernel to check handle validity
template <typename Handle>
__global__ void check_valid_kernel(const Handle* h_ptr, bool* valid_out) {
  if (threadIdx.x == 0) {
    *valid_out = h_ptr->valid();
  }
}

// Kernel to extract values from handle
template <typename Handle, typename ValueType>
__global__ void extract_values_kernel(const Handle* h_ptr, const typename Handle::View& view,
                                      ValueType* values_out, uint32_t* count_out) {
  if (threadIdx.x == 0) {
    const Handle& handle = *h_ptr;
    if (handle.valid()) {
      uint32_t count = static_cast<uint32_t>(handle.degree());
      *count_out = count;
      for (uint32_t i = 0; i < count && i < 32; ++i) {
        values_out[i] = handle.get_value_at(view, i);
      }
    } else {
      *count_out = 0;
    }
  }
}

// Kernel to get prefix handle - all threads must participate for cooperative groups operations
template <typename Handle, typename ValueType>
__global__ void test_prefix_kernel(Handle handle, const typename Handle::View& view, ValueType key,
                                   Handle* result_out) {
  // All threads call prefix() - needed for cooperative groups operations
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<32>(block);
  Handle result = handle.prefix(key, tile, view);
  if (threadIdx.x == 0) {
    *result_out = result;
  }
}

// Helper: Check handle values using GPU kernel
template <typename Handle, typename ValueType>
void check_handle_values_gpu(const Handle& h, const typename Handle::View& view,
                             const std::vector<ValueType>& expected) {
  // Copy handle to device memory
  DeviceArray<Handle> dev_handle(1);
  Handle h_copy = h;
  thrust::copy_n(&h_copy, 1, dev_handle.begin());

  // Copy view to device memory
  DeviceArray<typename Handle::View> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  DeviceArray<ValueType> dev_values(32);
  DeviceArray<uint32_t> dev_count(1);
  DeviceArray<bool> dev_valid(1);

  // Check validity
  check_valid_kernel<<<1, 32>>>(dev_handle.data(), dev_valid.data());
  device_sync_via_thrust();

  std::vector<bool> valid_host = device_to_host(dev_valid);
  BOOST_REQUIRE(valid_host[0]);

  // Extract values
  extract_values_kernel<<<1, 32>>>(dev_handle.data(), *dev_view.data(), dev_values.data(),
                                   dev_count.data());
  device_sync_via_thrust();

  std::vector<uint32_t> count_host = device_to_host(dev_count);
  uint32_t actual_count = count_host[0];
  std::vector<ValueType> actual = device_to_host(dev_values);
  actual.resize(actual_count);

  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_REQUIRE_EQUAL(actual[i], expected[i]);
  }
}

BOOST_AUTO_TEST_CASE(test_clone_empty_into_empty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  idx1.clone_from(idx2);
  BOOST_REQUIRE(idx1.empty());
  BOOST_REQUIRE_EQUAL(idx1.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_clone_nonempty_into_empty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with data
  build_index_from_vectors<SR>(idx2, {1, 2}, {10, 20});
  BOOST_REQUIRE_EQUAL(idx2.size(), 2);
  BOOST_REQUIRE(!idx2.empty());

  // Clone idx2 into empty idx1
  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  BOOST_REQUIRE(!idx1.empty());
  BOOST_REQUIRE_EQUAL(idx1.size(), idx2.size());

  // Verify root values
  auto root1 = idx1.root();
  auto root2 = idx2.root();
  auto view1 = idx1.view();
  auto view2 = idx2.view();
  BOOST_REQUIRE(root1.valid());
  BOOST_REQUIRE(root2.valid());

  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{1, 2});
  check_handle_values_gpu(root2, view2, std::vector<uint32_t>{1, 2});

  // Verify navigation works
  using Handle = decltype(root1);
  DeviceArray<Handle> dev_n1_1(1), dev_n1_2(1);
  DeviceArray<typename Handle::View> dev_view1(1), dev_view2(1);
  thrust::copy_n(&view1, 1, dev_view1.begin());
  thrust::copy_n(&view2, 1, dev_view2.begin());
  test_prefix_kernel<<<1, 32>>>(root1, *dev_view1.data(), static_cast<uint32_t>(1),
                                dev_n1_1.data());
  test_prefix_kernel<<<1, 32>>>(root2, *dev_view2.data(), static_cast<uint32_t>(1),
                                dev_n1_2.data());
  device_sync_via_thrust();

  std::vector<Handle> n1_1_host = device_to_host(dev_n1_1);
  std::vector<Handle> n1_2_host = device_to_host(dev_n1_2);
  Handle n1_1 = n1_1_host[0];
  Handle n1_2 = n1_2_host[0];

  BOOST_REQUIRE(n1_1.valid());
  BOOST_REQUIRE(n1_2.valid());
  check_handle_values_gpu(n1_1, view1, std::vector<uint32_t>{10});
  check_handle_values_gpu(n1_2, view2, std::vector<uint32_t>{10});
}

BOOST_AUTO_TEST_CASE(test_clone_nonempty_into_nonempty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx1 with data
  build_index_from_vectors<SR>(idx1, {1, 2}, {10, 20});
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Build idx2 with different data
  build_index_from_vectors<SR>(idx2, {3, 4, 5}, {30, 40, 50});
  BOOST_REQUIRE_EQUAL(idx2.size(), 3);

  // Clone idx2 into idx1 (should overwrite)
  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 3);
  BOOST_REQUIRE_EQUAL(idx1.size(), idx2.size());

  // Verify idx1 now has idx2's data
  auto root1 = idx1.root();
  auto view1 = idx1.view();
  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{3, 4, 5});

  // Verify idx2 is unchanged
  auto root2 = idx2.root();
  auto view2 = idx2.view();
  check_handle_values_gpu(root2, view2, std::vector<uint32_t>{3, 4, 5});

  // Verify navigation
  using Handle = decltype(root1);
  DeviceArray<Handle> dev_n3(1);
  DeviceArray<typename Handle::View> dev_view1(1);
  thrust::copy_n(&view1, 1, dev_view1.begin());
  test_prefix_kernel<<<1, 32>>>(root1, *dev_view1.data(), static_cast<uint32_t>(3), dev_n3.data());
  device_sync_via_thrust();
  std::vector<Handle> n3_host = device_to_host(dev_n3);
  Handle n3 = n3_host[0];
  check_handle_values_gpu(n3, view1, std::vector<uint32_t>{30});
}

BOOST_AUTO_TEST_CASE(test_clone_single_element) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with single element
  build_index_from_vectors<SR>(idx2, {42}, {100});
  BOOST_REQUIRE_EQUAL(idx2.size(), 1);

  // Verify idx2 works before clone
  auto root2 = idx2.root();
  auto view2 = idx2.view();
  BOOST_REQUIRE(root2.valid());
  check_handle_values_gpu(root2, view2, std::vector<uint32_t>{42});

  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 1);
  BOOST_REQUIRE_EQUAL(idx1.size(), idx2.size());

  // Verify idx2 still works after clone
  root2 = idx2.root();
  view2 = idx2.view();
  BOOST_REQUIRE(root2.valid());

  // Verify idx1 works after clone
  auto root1 = idx1.root();
  auto view1 = idx1.view();
  BOOST_REQUIRE(root1.valid());
  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{42});
}

BOOST_AUTO_TEST_CASE(test_clone_large_index) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with many elements
  std::vector<uint32_t> c0, c1;
  for (uint32_t i = 0; i < 100; ++i) {
    c0.push_back(i);
    c1.push_back(i * 10);
  }
  build_index_from_vectors<SR>(idx2, c0, c1);
  BOOST_REQUIRE_EQUAL(idx2.size(), 100);

  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 100);

  // Verify first few values
  auto root = idx1.root();
  BOOST_REQUIRE(root.valid());

  // Check that we can navigate to first element
  auto view = idx1.view();
  using Handle = decltype(root);
  DeviceArray<Handle> dev_n0(1);
  DeviceArray<typename Handle::View> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_kernel<<<1, 32>>>(root, *dev_view.data(), static_cast<uint32_t>(0), dev_n0.data());
  device_sync_via_thrust();
  std::vector<Handle> n0_host = device_to_host(dev_n0);
  Handle n0 = n0_host[0];
  BOOST_REQUIRE(n0.valid());
  check_handle_values_gpu(n0, view, std::vector<uint32_t>{0});
}

BOOST_AUTO_TEST_CASE(test_clone_with_overlapping_keys) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with data that has same first column values
  build_index_from_vectors<SR>(idx2, {1, 1, 2}, {10, 20, 30});
  BOOST_REQUIRE_EQUAL(idx2.size(), 3);

  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 3);
  auto root = idx1.root();
  auto view = idx1.view();
  // Root values should be all column 0 values: [1, 1, 2] (includes duplicates, not unique values)
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 1, 2});

  // Verify navigation to key 1 returns both values
  using Handle = decltype(root);
  DeviceArray<Handle> dev_n1(1);
  DeviceArray<typename Handle::View> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_kernel<<<1, 32>>>(root, *dev_view.data(), static_cast<uint32_t>(1), dev_n1.data());
  device_sync_via_thrust();
  std::vector<Handle> n1_host = device_to_host(dev_n1);
  Handle n1 = n1_host[0];
  BOOST_REQUIRE(n1.valid());
  check_handle_values_gpu(n1, view, std::vector<uint32_t>{10, 20});
}

BOOST_AUTO_TEST_CASE(test_clone_three_columns) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with 3-column data
  std::vector<uint32_t> c0 = {1, 2};
  std::vector<uint32_t> c1 = {10, 20};
  std::vector<uint32_t> c2 = {100, 200};

  NDDeviceArray<uint32_t, 3> encoded(2);
  DeviceArray<uint32_t> d0 = host_to_device(c0);
  DeviceArray<uint32_t> d1 = host_to_device(c1);
  DeviceArray<uint32_t> d2 = host_to_device(c2);
  thrust::copy(d0.begin(), d0.end(), encoded.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), encoded.column_ptr(1));
  thrust::copy(d2.begin(), d2.end(), encoded.column_ptr(2));
  DeviceArray<SR::value_type> prov(2);
  idx2.build_from_encoded_device(IndexSpec{{0, 1, 2}}, encoded, prov);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx2.size(), 2);

  idx1.clone_from(idx2);
  device_sync_via_thrust();

  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  auto root = idx1.root();
  auto view = idx1.view();
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2});
}

BOOST_AUTO_TEST_CASE(test_clone_independence) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2
  build_index_from_vectors<SR>(idx2, {1, 2}, {10, 20});

  // Clone
  idx1.clone_from(idx2);
  device_sync_via_thrust();

  // Modify idx2
  idx2.clear();
  BOOST_REQUIRE(idx2.empty());

  // Verify idx1 is unchanged (independent copy)
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  auto root1 = idx1.root();
  auto view1 = idx1.view();
  BOOST_REQUIRE(root1.valid());
  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{1, 2});
}

BOOST_AUTO_TEST_CASE(test_clone_then_modify_source) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2
  build_index_from_vectors<SR>(idx2, {1, 2}, {10, 20});

  // Clone
  idx1.clone_from(idx2);
  device_sync_via_thrust();

  // Build new data in idx2
  build_index_from_vectors<SR>(idx2, {5, 6, 7}, {50, 60, 70});
  BOOST_REQUIRE_EQUAL(idx2.size(), 3);

  // Verify idx1 still has old data (independent copy)
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  auto root1 = idx1.root();
  auto view1 = idx1.view();
  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{1, 2});

  // Verify idx2 has new data
  auto root2 = idx2.root();
  auto view2 = idx2.view();
  check_handle_values_gpu(root2, view2, std::vector<uint32_t>{5, 6, 7});
}

BOOST_AUTO_TEST_CASE(test_clone_then_build_clone) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2
  build_index_from_vectors<SR>(idx2, {1, 2}, {10, 20});

  // Clone
  idx1.clone_from(idx2);
  device_sync_via_thrust();

  // Build new data in idx1
  build_index_from_vectors<SR>(idx1, {8, 9}, {80, 90});
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Verify idx2 is unchanged
  BOOST_REQUIRE_EQUAL(idx2.size(), 2);
  auto root2 = idx2.root();
  auto view2 = idx2.view();
  check_handle_values_gpu(root2, view2, std::vector<uint32_t>{1, 2});

  // Verify idx1 has new data
  auto root1 = idx1.root();
  auto view1 = idx1.view();
  check_handle_values_gpu(root1, view1, std::vector<uint32_t>{8, 9});
}

}  // namespace
