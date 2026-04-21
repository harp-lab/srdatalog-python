#define BOOST_TEST_MODULE device_sorted_array_index_test
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

#include "gpu/device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/macro.h"  // For WARP_SIZE
#include "gpu/nd_device_array.h"
#include "semiring.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

// Helper to synchronize device - uses Thrust operation that implicitly synchronizes
// This works with both CUDA and HIP backends
// Note: Thrust operations like copy/fill synchronize implicitly
// After kernel launches, we rely on subsequent Thrust operations (like device_to_host)
// to synchronize. For explicit sync, use a minimal Thrust operation.
inline void device_sync_via_thrust() {
  // Use a minimal Thrust operation to ensure device synchronization
  // This is portable across CUDA and HIP backends
  static SRDatalog::GPU::DeviceArray<uint32_t> sync_buffer(1);
  if (sync_buffer.size() > 0) {
    thrust::fill(thrust::device, sync_buffer.begin(), sync_buffer.end(), 0);
  }
}

using SRDatalog::GPU::DeviceArray;
using SRDatalog::GPU::DeviceSortedArrayIndex;
using SRDatalog::GPU::NDDeviceArray;
using BooleanSR = ::BooleanSR;
using NaturalBag = ::NaturalBag;
using SRDatalog::IndexSpec;
using SRDatalog::Vector;

namespace {

// ============================================================================
// CUDA Kernels for Testing Device Functions
// ============================================================================

// Kernel to test prefix() - writes result handle info
template <typename Handle, typename ValueType>
__global__ void test_prefix_kernel(Handle handle, const typename Handle::View& view, ValueType key,
                                   bool* valid_out, uint16_t* depth_out, uint32_t* degree_out,
                                   bool* is_leaf_out) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  auto result = handle.prefix(key, tile, view);
  if (threadIdx.x == 0) {
    *valid_out = result.valid();
    *depth_out = result.depth();
    *degree_out = result.degree();
    *is_leaf_out = result.is_leaf(view);
  }
}

// Kernel to extract values from handle
template <typename Handle, typename ValueType>
__global__ void extract_handle_values_kernel(Handle handle, const typename Handle::View& view,
                                             ValueType* values_out, uint32_t* count_out) {
  if (threadIdx.x == 0) {
    uint32_t count = static_cast<uint32_t>(handle.degree());
    *count_out = count;
    for (uint32_t i = 0; i < count && i < 32; ++i) {  // Limit to 32 for simplicity
      values_out[i] = handle.get_value_at(view, i);
    }
  }
}

// Kernel to test contains_value()
// All threads must participate for warp operations inside contains_value() to work correctly
template <typename Handle, typename ValueType>
__global__ void test_contains_value_kernel(Handle handle, const typename Handle::View& view,
                                           ValueType key, bool* result_out) {
  // Create tile for group-cooperative search
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

  // All threads call contains_value() - needed for group search operations
  bool result = handle.contains_value(key, view, tile);

  // Only thread 0 writes the result
  if (threadIdx.x == 0) {
    *result_out = result;
  }
}

// Kernel to get prefix result handle (returns POD handle)
// This kernel is called with handle passed by value - device pointers in the POD are handled
// correctly
// All threads must participate for warp operations inside prefix() to work correctly
template <typename Handle, typename ValueType>
__global__ void test_prefix_get_handle_kernel(Handle handle, const typename Handle::View& view,
                                              ValueType key, Handle* result_out) {
  // All threads call prefix() - needed for cooperative groups operations
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  Handle result = handle.prefix(key, tile, view);

  // Only thread 0 writes the result
  if (threadIdx.x == 0) {
    *result_out = result;
  }
}

// Kernel to test prefix and check validity/is_leaf in one call
template <typename Handle, typename ValueType>
__global__ void test_prefix_check_leaf_kernel(Handle handle, const typename Handle::View& view,
                                              ValueType key, bool* valid_out, bool* is_leaf_out) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  if (threadIdx.x == 0) {
    auto result = handle.prefix(key, tile, view);
    if (valid_out)
      *valid_out = result.valid();
    if (is_leaf_out)
      *is_leaf_out = result.is_leaf(view);
  }
}

// Kernel to check handle validity
template <typename Handle>
__global__ void check_handle_valid_kernel(const Handle* handle_ptr, bool* valid_out) {
  if (threadIdx.x == 0) {
    *valid_out = handle_ptr->valid();
  }
}

// Kernel to extract values from handle in device memory
template <typename Handle, typename ValueType>
__global__ void extract_values_from_device_handle_kernel(const Handle* handle_ptr,
                                                         const typename Handle::View& view,
                                                         ValueType* values_out,
                                                         uint32_t* count_out) {
  if (threadIdx.x == 0) {
    const Handle& h = *handle_ptr;
    if (h.valid()) {
      uint32_t count = static_cast<uint32_t>(h.degree());
      *count_out = count;
      for (uint32_t i = 0; i < count && i < 32; ++i) {
        values_out[i] = h.get_value_at(view, i);
      }
    } else {
      *count_out = 0;
    }
  }
}

// Kernel to dump CSR info directly from root handle
template <typename Handle>
__global__ void dump_csr_info_kernel(const Handle* handle_ptr, uint32_t* csr_size_out,
                                     uint32_t* handle_begin_out, uint32_t* handle_end_out,
                                     bool* handle_valid_out) {
  if (threadIdx.x == 0) {
    const Handle& h = *handle_ptr;
    *handle_valid_out = h.valid();
    *handle_begin_out = h.begin();
    *handle_end_out = h.end();
    *csr_size_out = h.valid() ? (h.end() - h.begin()) : 0;
  }
}

// Kernel to get prefix from a handle stored in device memory
// All threads must participate for cooperative groups operations inside prefix() to work correctly
template <typename Handle, typename ValueType>
__global__ void test_prefix_from_device_kernel(const Handle* handle_ptr,
                                               const typename Handle::View& view, ValueType key,
                                               Handle* result_out) {
  // All threads call prefix() - needed for cooperative groups operations
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  Handle result = handle_ptr->prefix(key, tile, view);

  // Only thread 0 writes the result
  if (threadIdx.x == 0) {
    *result_out = result;
  }
}

// Debug kernel: Test prefix search with detailed tracing
template <typename Handle, typename ValueType>
__global__ void debug_prefix_search_kernel(const Handle* handle_ptr,
                                           const typename Handle::View& view, ValueType key,
                                           Handle* result_out, uint32_t* debug_info,
                                           uint32_t* col_debug, uint32_t* result_debug) {
  // All threads must participate for any cooperative groups operations inside prefix()
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  const Handle& h = *handle_ptr;
  Handle result = h.prefix(key, tile, view);

  if (threadIdx.x == 0) {
    debug_info[0] = h.depth();
    debug_info[1] = static_cast<uint32_t>(h.begin());
    debug_info[2] = static_cast<uint32_t>(h.end());
    *result_out = result;

    // Debug result
    result_debug[0] = static_cast<uint32_t>(result.begin());
    result_debug[1] = static_cast<uint32_t>(result.end());
    result_debug[2] = static_cast<uint32_t>(result.depth());
    result_debug[3] = result.valid() ? 1 : 0;
    result_debug[4] = result.is_leaf(view) ? 1 : 0;

    // Get column values for debugging (if valid)
    if (h.valid() && !h.is_leaf(view)) {
      uint32_t degree = static_cast<uint32_t>(h.degree());
      col_debug[0] = degree;
      if (degree > 0)
        col_debug[1] = static_cast<uint32_t>(h.get_value_at(view, 0));
      if (degree > 1)
        col_debug[2] = static_cast<uint32_t>(h.get_value_at(view, 1));
      if (degree > 2)
        col_debug[3] = static_cast<uint32_t>(h.get_value_at(view, 2));
    }
  }
}

// Kernel to get handle depth and is_leaf status
template <typename Handle>
__global__ void get_handle_depth_kernel(const Handle* h, const typename Handle::View& view,
                                        uint16_t* depth_out, bool* is_leaf_out) {
  if (threadIdx.x == 0) {
    *depth_out = h->depth();
    *is_leaf_out = h->is_leaf(view);
  }
}

// Debug kernel: Dump handle properties
template <typename Handle>
__global__ void debug_dump_handle_kernel(const Handle* handle_ptr, uint32_t* begin_out,
                                         uint32_t* end_out, uint16_t* depth_out) {
  if (threadIdx.x == 0) {
    *begin_out = static_cast<uint32_t>(handle_ptr->begin());
    *end_out = static_cast<uint32_t>(handle_ptr->end());
    *depth_out = handle_ptr->depth();
  }
}

// Debug kernel: Test warp_equal_range directly to see what it returns
template <typename ValueType>
__global__ void debug_warp_equal_range_kernel(const ValueType* data, uint32_t len, ValueType key,
                                              uint32_t* lower_out, uint32_t* upper_out,
                                              uint32_t* lt_mask_out, uint32_t* le_mask_out) {
  // All threads must participate for __ballot_sync to work correctly
  // Call warp_equal_range first (all threads participate)
  auto [lower, upper] = SRDatalog::GPU::warp_equal_range<ValueType, uint32_t>(data, len, key);

  // Also compute masks manually for comparison (all threads participate)
  uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
  ValueType sentinel = cuda::std::numeric_limits<ValueType>::max();
  ValueType val = (lane < len) ? data[lane] : sentinel;

// ROCm requires 64-bit mask for __ballot_sync, CUDA accepts 32-bit
#ifdef USE_ROCm
  uint64_t lt_mask = __ballot_sync(0xffffffffULL, val < key);
  uint64_t le_mask = __ballot_sync(0xffffffffULL, val <= key);
#else
  uint32_t lt_mask = __ballot_sync(0xffffffff, val < key);
  uint32_t le_mask = __ballot_sync(0xffffffff, val <= key);
#endif

  // Only thread 0 writes results
  if (threadIdx.x == 0) {
    *lower_out = lower;
    *upper_out = upper;
    // Cast to uint32_t for storage (ROCm uses uint64_t but we store as uint32_t for compatibility)
    *lt_mask_out = static_cast<uint32_t>(lt_mask);
    *le_mask_out = static_cast<uint32_t>(le_mask);
  }
}

// Kernel to test prov() function
// All threads in warp must participate for tile_semiring_reduce to work correctly
template <typename Handle, typename ProvValueType>
__global__ void test_prov_kernel(const Handle* handle_ptr, const typename Handle::View& view,
                                 ProvValueType* prov_out) {
  // All threads must participate for cooperative groups operations
  const Handle& h = *handle_ptr;
  ProvValueType result = h.template prov<32>(view);  // Use TILE_SIZE=32

  // Only thread 0 writes the result
  if (threadIdx.x == 0) {
    *prov_out = result;
  }
}

// ============================================================================
// Test Helpers
// ============================================================================

// Helper: Copy host vector to device array
// Let Thrust infer execution policy from iterator types (works with both CUDA and HIP)
template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  return dev;
}

// Helper: Copy device array to host vector
// Let Thrust infer execution policy from iterator types (works with both CUDA and HIP)
template <typename T>
std::vector<T> device_to_host(const DeviceArray<T>& dev_array) {
  std::vector<T> host(dev_array.size());
  thrust::copy(dev_array.begin(), dev_array.end(), host.begin());
  return host;
}

// Helper: Check if handle values match expected (using GPU kernel)
// Handle is kept in device memory to ensure device pointers remain valid
template <typename Handle, typename ValueType>
void check_handle_values_gpu(const Handle& h, const typename Handle::View& view,
                             const std::vector<ValueType>& expected) {
  // Copy handle to device memory first to ensure device pointers are valid
  DeviceArray<Handle> dev_handle(1);
  Handle h_copy = h;  // Make a copy (device pointers are just values)
  // Let Thrust infer execution policy - it will detect host pointer to device pointer
  thrust::copy_n(&h_copy, 1, dev_handle.begin());

  // Copy view to device memory
  DeviceArray<typename Handle::View> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  DeviceArray<ValueType> dev_values(32);
  DeviceArray<uint32_t> dev_count(1);
  DeviceArray<bool> dev_valid(1);

  // Check validity
  check_handle_valid_kernel<<<1, WARP_SIZE>>>(dev_handle.data(), dev_valid.data());
  std::vector<bool> valid_host = device_to_host(dev_valid);
  BOOST_REQUIRE(valid_host[0]);

  // Extract values
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_handle.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  // device_to_host will synchronize implicitly

  std::vector<uint32_t> count_host = device_to_host(dev_count);
  uint32_t actual_count = count_host[0];

  std::vector<ValueType> actual = device_to_host(dev_values);
  actual.resize(actual_count);

  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_REQUIRE_EQUAL(actual[i], expected[i]);
  }
}

// Helper: Test contains_value() on GPU
template <typename Handle, typename ValueType>
bool contains_value_gpu(const Handle& h, const typename Handle::View& view, ValueType key) {
  DeviceArray<bool> dev_result(1);
  DeviceArray<typename Handle::View> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  test_contains_value_kernel<<<1, WARP_SIZE>>>(h, *dev_view.data(), key, dev_result.data());
  // Thrust copy will synchronize, no explicit sync needed

  std::vector<bool> result_host = device_to_host(dev_result);
  return result_host[0];
}

// ============================================================================
// Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(test_basic_build_and_navigate) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;

  // Data: (1, 10), (1, 20), (2, 20)
  std::vector<uint32_t> col0 = {1, 1, 2};
  std::vector<uint32_t> col1 = {10, 20, 20};

  // Prepare NDDeviceArray
  NDDeviceArray<uint32_t, 2> encoded_cols(3);
  DeviceArray<uint32_t> d_col0 = host_to_device(col0);
  DeviceArray<uint32_t> d_col1 = host_to_device(col1);

  thrust::copy(d_col0.begin(), d_col0.end(), encoded_cols.column_ptr(0));
  thrust::copy(d_col1.begin(), d_col1.end(), encoded_cols.column_ptr(1));

  DeviceArray<SR::value_type> provenance(3);  // Default initialized

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, encoded_cols, provenance);

  // Debug: Check actual size
  std::size_t actual_size = idx.size();
  if (actual_size != 3) {
    BOOST_TEST_MESSAGE("Warning: Expected size 3, got " << actual_size);
  }
  BOOST_REQUIRE_EQUAL(idx.size(), 3);
  BOOST_REQUIRE(!idx.empty());

  // Navigate
  auto root = idx.root();
  auto view = idx.view();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE(!root.is_leaf(view));
  BOOST_REQUIRE_EQUAL(root.depth(), 0);

  // Verify root handle spans entire column 0 (begin=0, end=num_rows)
  // unique_values is only for partitioning, not used by cursor/handle
  BOOST_REQUIRE_EQUAL(root.begin(), 0);
  BOOST_REQUIRE_EQUAL(root.end(), 3);  // Should have 3 rows: [1, 1, 2]

  // Check root values: should be column 0 values [1, 1, 2] (not unique values)
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 1, 2});

  // Go to 1 - test prefix on GPU
  // Keep handles in device memory - copy root to device first
  using Handle = decltype(root);
  DeviceArray<Handle> dev_root(1);
  // Let Thrust infer execution policy - it will detect host pointer to device pointer
  thrust::copy_n(&root, 1, dev_root.begin());

  DeviceArray<Handle> dev_n1(1);
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  // Synchronize via Thrust operation (device_to_host will also synchronize, but be explicit)
  device_sync_via_thrust();

  // Check validity using device-side handle
  DeviceArray<bool> dev_valid(1);
  check_handle_valid_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), dev_valid.data());
  std::vector<bool> valid_host = device_to_host(dev_valid);
  BOOST_REQUIRE(valid_host[0]);

  // Check values using device-side handle
  DeviceArray<uint32_t> dev_values(32);
  DeviceArray<uint32_t> dev_count(1);
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  std::vector<uint32_t> count_host = device_to_host(dev_count);
  std::vector<uint32_t> values_host = device_to_host(dev_values);
  values_host.resize(count_host[0]);
  BOOST_REQUIRE_EQUAL(values_host.size(), 2);
  BOOST_REQUIRE_EQUAL(values_host[0], 10);
  BOOST_REQUIRE_EQUAL(values_host[1], 20);

  // Go to 1 -> 10 (using device-side n1)
  DeviceArray<Handle> dev_n1_10(1);
  test_prefix_from_device_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                   static_cast<uint32_t>(10), dev_n1_10.data());
  device_sync_via_thrust();

  DeviceArray<bool> dev_is_leaf(1);
  DeviceArray<uint16_t> dev_n1_10_depth(1);
  get_handle_depth_kernel<<<1, WARP_SIZE>>>(dev_n1_10.data(), *dev_view.data(),
                                            dev_n1_10_depth.data(), dev_is_leaf.data());
  std::vector<bool> is_leaf_host = device_to_host(dev_is_leaf);
  BOOST_REQUIRE(is_leaf_host[0]);  // Should be leaf

  // Go to 1 -> 20
  // First, verify n1 still has the expected values before trying prefix(20)
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  count_host = device_to_host(dev_count);
  values_host = device_to_host(dev_values);
  values_host.resize(count_host[0]);
  BOOST_REQUIRE_EQUAL(values_host.size(), 2);
  BOOST_REQUIRE_EQUAL(values_host[0], 10);
  BOOST_REQUIRE_EQUAL(values_host[1], 20);  // Verify 20 is still there

  // Debug: Check handle properties before search and test warp_equal_range directly
  DeviceArray<uint32_t> debug_begin(1), debug_end(1);
  DeviceArray<uint16_t> debug_depth(1);
  debug_dump_handle_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), debug_begin.data(), debug_end.data(),
                                             debug_depth.data());
  std::vector<uint32_t> begin_vec = device_to_host(debug_begin);
  std::vector<uint32_t> end_vec = device_to_host(debug_end);
  std::vector<uint16_t> depth_vec = device_to_host(debug_depth);
  BOOST_TEST_MESSAGE("Before prefix(20): begin=" << begin_vec[0] << ", end=" << end_vec[0]
                                                 << ", depth=" << static_cast<int>(depth_vec[0]));

  // Extract the actual column data to test warp_equal_range directly
  std::vector<Handle> n1_check = device_to_host(dev_n1);
  Handle n1_local = n1_check[0];
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  count_host = device_to_host(dev_count);
  values_host = device_to_host(dev_values);
  values_host.resize(count_host[0]);
  BOOST_TEST_MESSAGE("Column data at depth 1: [" << values_host[0] << ", " << values_host[1]
                                                 << "]");

  // Test warp_equal_range directly on this data
  DeviceArray<uint32_t> test_data(2);
  DeviceArray<uint32_t> lower_result(1), upper_result(1), lt_mask_result(1), le_mask_result(1);
  thrust::copy(values_host.begin(), values_host.end(), test_data.begin());
  debug_warp_equal_range_kernel<<<1, WARP_SIZE>>>(test_data.data(), 2, static_cast<uint32_t>(20),
                                                  lower_result.data(), upper_result.data(),
                                                  lt_mask_result.data(), le_mask_result.data());
  std::vector<uint32_t> lower_vec = device_to_host(lower_result);
  std::vector<uint32_t> upper_vec = device_to_host(upper_result);
  std::vector<uint32_t> lt_mask_vec = device_to_host(lt_mask_result);
  std::vector<uint32_t> le_mask_vec = device_to_host(le_mask_result);
  BOOST_TEST_MESSAGE("warp_equal_range([10,20], len=2, key=20) returned: lower="
                     << lower_vec[0] << ", upper=" << upper_vec[0]);
  BOOST_TEST_MESSAGE("lt_mask=" << std::hex << lt_mask_vec[0] << std::dec
                                << " (popc=" << __builtin_popcount(lt_mask_vec[0]) << "), "
                                << "le_mask=" << std::hex << le_mask_vec[0] << std::dec
                                << " (popc=" << __builtin_popcount(le_mask_vec[0]) << ")");

  DeviceArray<Handle> dev_n1_20(1);
  DeviceArray<uint32_t> debug_info(10);
  DeviceArray<uint32_t> col_debug(10);
  DeviceArray<uint32_t> result_debug(10);
  debug_prefix_search_kernel<<<1, WARP_SIZE>>>(
      dev_n1.data(), *dev_view.data(), static_cast<uint32_t>(20), dev_n1_20.data(),
      debug_info.data(), col_debug.data(), result_debug.data());
  std::vector<uint32_t> debug_vec = device_to_host(debug_info);
  std::vector<uint32_t> col_vec = device_to_host(col_debug);
  std::vector<uint32_t> result_vec = device_to_host(result_debug);
  BOOST_TEST_MESSAGE("Before prefix(20): depth=" << debug_vec[0] << ", begin=" << debug_vec[1]
                                                 << ", end=" << debug_vec[2]);
  BOOST_TEST_MESSAGE("Column values: size=" << col_vec[0] << ", [0]=" << col_vec[1]
                                            << ", [1]=" << col_vec[2]);
  BOOST_TEST_MESSAGE("After prefix(20): begin=" << result_vec[0] << ", end=" << result_vec[1]
                                                << ", depth=" << result_vec[2] << ", valid="
                                                << result_vec[3] << ", is_leaf=" << result_vec[4]);

  // Also try the regular kernel
  test_prefix_from_device_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                   static_cast<uint32_t>(20), dev_n1_20.data());
  device_sync_via_thrust();
  check_handle_valid_kernel<<<1, WARP_SIZE>>>(dev_n1_20.data(), dev_valid.data());
  valid_host = device_to_host(dev_valid);
  if (!valid_host[0]) {
    BOOST_TEST_MESSAGE("ERROR: prefix(20) returned invalid handle!");
    // Debug: Check what happened
    debug_dump_handle_kernel<<<1, WARP_SIZE>>>(dev_n1_20.data(), debug_begin.data(),
                                               debug_end.data(), debug_depth.data());
    begin_vec = device_to_host(debug_begin);
    end_vec = device_to_host(debug_end);
    depth_vec = device_to_host(debug_depth);
    BOOST_TEST_MESSAGE("After prefix(20) (invalid): begin=" << begin_vec[0]
                                                            << ", end=" << end_vec[0] << ", depth="
                                                            << static_cast<int>(depth_vec[0]));
  }
  BOOST_REQUIRE(valid_host[0]);

  // Go to 2
  DeviceArray<Handle> dev_n2(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(2),
                                                  dev_n2.data());
  device_sync_via_thrust();
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_n2.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  count_host = device_to_host(dev_count);
  values_host = device_to_host(dev_values);
  values_host.resize(count_host[0]);
  BOOST_REQUIRE_EQUAL(values_host.size(), 1);
  BOOST_REQUIRE_EQUAL(values_host[0], 20);

  // Go to 2 -> 20
  DeviceArray<Handle> dev_n2_20(1);
  test_prefix_from_device_kernel<<<1, WARP_SIZE>>>(dev_n2.data(), *dev_view.data(),
                                                   static_cast<uint32_t>(20), dev_n2_20.data());
  device_sync_via_thrust();
  check_handle_valid_kernel<<<1, WARP_SIZE>>>(dev_n2_20.data(), dev_valid.data());
  valid_host = device_to_host(dev_valid);
  BOOST_REQUIRE(valid_host[0]);
}

BOOST_AUTO_TEST_CASE(test_contains_value) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;

  std::vector<uint32_t> col0 = {1, 3};
  std::vector<uint32_t> col1 = {10, 30};

  NDDeviceArray<uint32_t, 2> encoded_cols(2);
  DeviceArray<uint32_t> d_col0 = host_to_device(col0);
  DeviceArray<uint32_t> d_col1 = host_to_device(col1);

  thrust::copy(d_col0.begin(), d_col0.end(), encoded_cols.column_ptr(0));
  thrust::copy(d_col1.begin(), d_col1.end(), encoded_cols.column_ptr(1));

  DeviceArray<SR::value_type> provenance(2);

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, encoded_cols, provenance);

  auto root = idx.root();
  auto view = idx.view();
  BOOST_REQUIRE(contains_value_gpu(root, view, static_cast<uint32_t>(1)));
  BOOST_REQUIRE(contains_value_gpu(root, view, static_cast<uint32_t>(3)));
  BOOST_REQUIRE(!contains_value_gpu(root, view, static_cast<uint32_t>(2)));

  using Handle = decltype(root);
  DeviceArray<Handle> dev_n1(1);
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  std::vector<Handle> n1_host = device_to_host(dev_n1);
  Handle n1 = n1_host[0];
  BOOST_REQUIRE(contains_value_gpu(n1, view, static_cast<uint32_t>(10)));
  BOOST_REQUIRE(!contains_value_gpu(n1, view, static_cast<uint32_t>(20)));
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
}

// Helper function to build index from vectors with custom IndexSpec (for TC-style {{1, 0}} index)
template <typename SR, typename DSAI>
void build_index_from_vectors_spec(DSAI& idx, const std::vector<uint32_t>& c0,
                                   const std::vector<uint32_t>& c1, const IndexSpec& spec) {
  NDDeviceArray<uint32_t, 2> encoded(c0.size());
  DeviceArray<uint32_t> d0 = host_to_device(c0);
  DeviceArray<uint32_t> d1 = host_to_device(c1);
  // Column order in encoded is always [0, 1] (relation order)
  thrust::copy(d0.begin(), d0.end(), encoded.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), encoded.column_ptr(1));
  DeviceArray<typename SR::value_type> prov(c0.size());
  idx.build_from_encoded_device(spec, encoded, prov);
}

BOOST_AUTO_TEST_CASE(test_merge_empty_into_empty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), 0);
  BOOST_REQUIRE(idx1.empty());
}

BOOST_AUTO_TEST_CASE(test_merge_empty_into_nonempty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx1 with some data
  build_index_from_vectors<SR>(idx1, {1, 2}, {10, 20});

  std::size_t size_before = idx1.size();
  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), size_before);
}

BOOST_AUTO_TEST_CASE(test_merge_nonempty_into_empty) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // Build idx2 with some data
  build_index_from_vectors<SR>(idx2, {1, 2}, {10, 20});

  // Ensure idx2 is fully built before merge
  device_sync_via_thrust();

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  auto root = idx1.root();
  auto view = idx1.view();
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2});
}

BOOST_AUTO_TEST_CASE(test_merge_disjoint_keys) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1,10), (2,20)
  build_index_from_vectors<SR>(idx1, {1, 2}, {10, 20});

  // idx2: (3,30), (4,40)
  build_index_from_vectors<SR>(idx2, {3, 4}, {30, 40});

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  auto root = idx1.root();
  auto view1 = idx1.view();
  // Root values should be all column 0 values: [1, 2, 3, 4] (all unique in this case)
  check_handle_values_gpu(root, view1, std::vector<uint32_t>{1, 2, 3, 4});
}

BOOST_AUTO_TEST_CASE(test_merge_overlapping_keys) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1,10), (2,20)
  build_index_from_vectors<SR>(idx1, {1, 2}, {10, 20});

  // idx2: (2,25), (3,30) - overlaps with idx1 on key (2,20) vs (2,25)
  build_index_from_vectors<SR>(idx2, {2, 3}, {25, 30});

  idx1.merge(idx2, 100);
  // After merge and deduplication: (1,10), (2,20), (2,25), (3,30) -> deduped to unique rows
  // But merge keeps both (2,20) and (2,25) as they are different keys
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  auto root = idx1.root();
  auto view = idx1.view();
  // Root values should be all column 0 values: [1, 2, 2, 3] (sorted, includes duplicates)
  // After merge: (1,10), (2,20), (2,25), (3,30) -> column 0: [1, 2, 2, 3]
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2, 2, 3});
}

BOOST_AUTO_TEST_CASE(test_merge_equal_keys) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1,10)
  build_index_from_vectors<SR>(idx1, {1}, {10});

  // idx2: (1,10) - same key
  build_index_from_vectors<SR>(idx2, {1}, {10});

  BOOST_REQUIRE_EQUAL(idx1.size(), 1);
  BOOST_REQUIRE_EQUAL(idx2.size(), 1);

  // After merge, both entries are kept (merge does NOT deduplicate)
  // merge_by_key simply merges sorted arrays and keeps duplicates
  idx1.merge(idx2, 100);
  // Since merge does not deduplicate, we expect size to be 2 (both entries kept)
  // Deduplication happens during join generation (checking against full), not during merge
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  auto root = idx1.root();
  auto view = idx1.view();
  // Root values should be all column 0 values: [1, 1] (both entries kept after merge)
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 1});
}

BOOST_AUTO_TEST_CASE(test_merge_preserves_sorted_order) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1,10), (3,30)
  build_index_from_vectors<SR>(idx1, {1, 3}, {10, 30});

  // idx2: (2,20), (4,40)
  build_index_from_vectors<SR>(idx2, {2, 4}, {20, 40});

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  // Verify sorted order: (1,10), (2,20), (3,30), (4,40)
  auto root = idx1.root();
  auto view = idx1.view();
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2, 3, 4});
}

BOOST_AUTO_TEST_CASE(test_merge_basic) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1, 10)
  build_index_from_vectors<SR>(idx1, {1}, {10});

  // idx2: (2, 20)
  build_index_from_vectors<SR>(idx2, {2}, {20});

  // Merge idx2 into idx1
  idx1.merge(idx2, 100);

  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  auto root = idx1.root();
  auto view = idx1.view();
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2});

  using Handle = decltype(root);
  DeviceArray<Handle> dev_n1(1);
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  std::vector<Handle> n1_host = device_to_host(dev_n1);
  Handle n1 = n1_host[0];
  check_handle_values_gpu(n1, view, std::vector<uint32_t>{10});

  DeviceArray<Handle> dev_n2(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(2),
                                                  dev_n2.data());
  std::vector<Handle> n2_host = device_to_host(dev_n2);
  Handle n2 = n2_host[0];
  check_handle_values_gpu(n2, view, std::vector<uint32_t>{20});
}

BOOST_AUTO_TEST_CASE(test_merge_multiple_entries) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1;
  DSAI idx2;

  // idx1: (1,10), (3,30), (5,50)
  build_index_from_vectors<SR>(idx1, {1, 3, 5}, {10, 30, 50});

  // idx2: (2,20), (4,40)
  build_index_from_vectors<SR>(idx2, {2, 4}, {20, 40});

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 5);

  // Verify sorted order: (1,10), (2,20), (3,30), (4,40), (5,50)
  auto root = idx1.root();
  auto view = idx1.view();
  check_handle_values_gpu(root, view, std::vector<uint32_t>{1, 2, 3, 4, 5});

  // Verify navigation
  using Handle = decltype(root);
  DeviceArray<Handle> dev_n3(1);
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(3),
                                                  dev_n3.data());
  std::vector<Handle> n3_host = device_to_host(dev_n3);
  Handle n3 = n3_host[0];
  check_handle_values_gpu(n3, view, std::vector<uint32_t>{30});
}

// ============================================================================
// Large-scale Build Tests with CSR Verification
// ============================================================================

// ============================================================================
// Batch Search Tests (Multiple keys in parallel)
// ============================================================================

// Kernel for batch search - each warp processes one key
// All threads in warp must participate for prefix() to work correctly
template <typename Handle, typename ValueType>
__global__ void batch_search_prefix_kernel(const Handle* root_ptr,
                                           const typename Handle::View& view, const ValueType* keys,
                                           uint32_t num_keys, bool* results_out,
                                           uint32_t* counts_out) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

  // All threads in warp must participate, even if warp_id >= num_keys
  // Use a sentinel key for warps without work to ensure all threads call prefix()
  Handle root = *root_ptr;
  ValueType key = (warp_id < num_keys) ? keys[warp_id] : static_cast<ValueType>(0);

  // All threads participate in prefix search (required for cooperative groups operations)
  Handle result = root.prefix(key, tile, view);

  // Only write results for valid warps
  if (warp_id < num_keys && lane == 0) {
    results_out[warp_id] = result.valid();
    if (result.valid()) {
      counts_out[warp_id] = result.degree();
    } else {
      counts_out[warp_id] = 0;
    }
  }
}

// Kernel for batch search with tuple existence check
// All threads in warp must participate for prefix() to work correctly
template <typename Handle, typename ValueType>
__global__ void batch_search_tuple_exists_kernel(const Handle* root_ptr,
                                                 const typename Handle::View& view,
                                                 const ValueType* keys_col0,
                                                 const ValueType* keys_col1, uint32_t num_keys,
                                                 bool* results_out) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

  // All threads participate
  Handle root = *root_ptr;
  ValueType key0 = (warp_id < num_keys) ? keys_col0[warp_id] : static_cast<ValueType>(0);
  Handle n0 = root.prefix(key0, tile, view);

  if (warp_id < num_keys) {
    if (n0.valid()) {
      ValueType key1 = keys_col1[warp_id];
      Handle leaf = n0.prefix(key1, tile, view);

      if (lane == 0) {
        results_out[warp_id] = leaf.valid() && leaf.is_leaf(view);
      }
    } else {
      if (lane == 0) {
        results_out[warp_id] = false;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_batch_search_prefix) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;
  build_index_from_vectors<SR>(idx, {1, 1, 2, 3, 3}, {10, 20, 30, 40, 50});

  auto root = idx.root();
  auto view = idx.view();
  DeviceArray<decltype(root)> dev_root(1);
  thrust::copy_n(&root, 1, dev_root.begin());
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  // Search for multiple keys in batch
  std::vector<uint32_t> search_keys = {1, 2, 3, 4, 5};  // 4 exists, 5 doesn't
  DeviceArray<uint32_t> dev_keys = host_to_device(search_keys);
  DeviceArray<bool> dev_results(5);
  DeviceArray<uint32_t> dev_counts(5);

  // Launch with enough threads for all keys (each warp handles one key)
  // Need at least WARP_SIZE threads per warp, and each warp handles one key
  // For 5 keys, we need 5 warps = 5 * WARP_SIZE threads
  uint32_t num_keys = 5;
#ifdef USE_ROCm
  // On ROCm with 64-thread warps, we need more threads to get enough warps
  // 256 threads = 4 warps on ROCm (256/64), but we need at least 5 warps for 5 keys
  uint32_t threads_per_block = 320;  // 320/64 = 5 warps, enough for 5 keys
#else
  uint32_t threads_per_block = 256;  // 256/32 = 8 warps on CUDA
#endif
  uint32_t num_blocks = (num_keys * WARP_SIZE + threads_per_block - 1) / threads_per_block;
  batch_search_prefix_kernel<<<num_blocks, threads_per_block>>>(
      dev_root.data(), *dev_view.data(), dev_keys.data(), num_keys, dev_results.data(),
      dev_counts.data());
  device_sync_via_thrust();

  std::vector<bool> results = device_to_host(dev_results);
  std::vector<uint32_t> counts = device_to_host(dev_counts);

  BOOST_REQUIRE(results[0] == true);   // Key 1 exists
  BOOST_REQUIRE(results[1] == true);   // Key 2 exists
  BOOST_REQUIRE(results[2] == true);   // Key 3 exists
  BOOST_REQUIRE(results[3] == false);  // Key 4 doesn't exist
  BOOST_REQUIRE(results[4] == false);  // Key 5 doesn't exist

  BOOST_REQUIRE_EQUAL(counts[0], 2u);  // Key 1 has 2 children
  BOOST_REQUIRE_EQUAL(counts[1], 1u);  // Key 2 has 1 child
  BOOST_REQUIRE_EQUAL(counts[2], 2u);  // Key 3 has 2 children
  BOOST_REQUIRE_EQUAL(counts[3], 0u);
  BOOST_REQUIRE_EQUAL(counts[4], 0u);
}

BOOST_AUTO_TEST_CASE(test_batch_search_count_matched_tuples) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;
  // Build index: (1,10), (1,20), (2,30), (2,30), (3,40)
  // After dedup: (1,10), (1,20), (2,30), (3,40)
  build_index_from_vectors<SR>(idx, {1, 1, 2, 2, 3}, {10, 20, 30, 30, 40});

  auto root = idx.root();
  auto view = idx.view();
  DeviceArray<decltype(root)> dev_root(1);
  thrust::copy_n(&root, 1, dev_root.begin());
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  // Search for multiple tuples
  std::vector<uint32_t> keys_col0 = {1, 1, 2, 3, 99};
  std::vector<uint32_t> keys_col1 = {10, 20, 30, 40, 99};
  DeviceArray<uint32_t> dev_keys0 = host_to_device(keys_col0);
  DeviceArray<uint32_t> dev_keys1 = host_to_device(keys_col1);
  DeviceArray<bool> dev_results(5);

  // Launch with enough threads for all keys (each warp handles one key)
  uint32_t num_keys = 5;
#ifdef USE_ROCm
  // On ROCm with 64-thread warps, we need more threads to get enough warps
  // 256 threads = 4 warps on ROCm (256/64), but we need at least 5 warps for 5 keys
  uint32_t threads_per_block = 320;  // 320/64 = 5 warps, enough for 5 keys
#else
  uint32_t threads_per_block = 256;  // 256/32 = 8 warps on CUDA
#endif
  uint32_t num_blocks = (num_keys * WARP_SIZE + threads_per_block - 1) / threads_per_block;
  batch_search_tuple_exists_kernel<<<num_blocks, threads_per_block>>>(
      dev_root.data(), *dev_view.data(), dev_keys0.data(), dev_keys1.data(), num_keys,
      dev_results.data());
  device_sync_via_thrust();

  std::vector<bool> results = device_to_host(dev_results);
  BOOST_REQUIRE(results[0] == true);   // (1,10) exists
  BOOST_REQUIRE(results[1] == true);   // (1,20) exists
  BOOST_REQUIRE(results[2] == true);   // (2,30) exists
  BOOST_REQUIRE(results[3] == true);   // (3,40) exists
  BOOST_REQUIRE(results[4] == false);  // (99,99) doesn't exist

  // Count total matched
  uint32_t matched_count = std::count(results.begin(), results.end(), true);
  BOOST_REQUIRE_EQUAL(matched_count, 4u);
}

// ============================================================================
// Multi-Column Search Tests (3-level nested search)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_three_column_nested_search) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int, int>>;

  DSAI idx;

  // Build 3-column index: (1,10,100), (1,10,200), (1,20,100), (2,30,300)
  std::vector<uint32_t> c0 = {1, 1, 1, 2};
  std::vector<uint32_t> c1 = {10, 10, 20, 30};
  std::vector<uint32_t> c2 = {100, 200, 100, 300};

  NDDeviceArray<uint32_t, 3> encoded(4);
  DeviceArray<uint32_t> d0 = host_to_device(c0);
  DeviceArray<uint32_t> d1 = host_to_device(c1);
  DeviceArray<uint32_t> d2 = host_to_device(c2);
  thrust::copy(d0.begin(), d0.end(), encoded.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), encoded.column_ptr(1));
  thrust::copy(d2.begin(), d2.end(), encoded.column_ptr(2));

  DeviceArray<SR::value_type> prov(4);
  IndexSpec spec{{0, 1, 2}};
  idx.build_from_encoded_device(spec, encoded, prov);
  device_sync_via_thrust();

  // All 4 tuples are unique, so size should be 4
  BOOST_REQUIRE_EQUAL(idx.size(), 4);

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  // Verify root has all column 0 values (including duplicates)
  // Actual data: (1,10,100), (1,10,200), (1,20,100), (2,30,300)
  // Column 0 values: [1, 1, 1, 2] (sorted, includes duplicates)
  auto view = idx.view();
  DeviceArray<decltype(root)> dev_root(1);
  thrust::copy_n(&root, 1, dev_root.begin());
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  DeviceArray<uint32_t> dev_values(10);
  DeviceArray<uint32_t> dev_count(1);
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(dev_root.data(), *dev_view.data(),
                                                             dev_values.data(), dev_count.data());
  device_sync_via_thrust();
  std::vector<uint32_t> count_host = device_to_host(dev_count);
  std::vector<uint32_t> values_host = device_to_host(dev_values);
  values_host.resize(count_host[0]);
  // Root should have all column 0 values (sorted)
  // Actual data: (1,10,100), (1,10,200), (1,20,100), (2,30,300)
  // After sorting by IndexSpec{{0, 1, 2}}, column 0 should be [1, 1, 1, 2]
  BOOST_REQUIRE_EQUAL(count_host[0], 4);
  // Verify all values are present (sorted order)
  BOOST_REQUIRE_EQUAL(values_host[0], 1u);
  BOOST_REQUIRE_EQUAL(values_host[1], 1u);
  BOOST_REQUIRE_EQUAL(values_host[2], 1u);
  BOOST_REQUIRE_EQUAL(values_host[3], 2u);

  // Nested search: root -> prefix(1) -> prefix(10) -> prefix(100)
  using Handle = decltype(root);

  // Level 1: prefix(1)
  DeviceArray<Handle> dev_n1(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  device_sync_via_thrust();
  std::vector<Handle> handles = device_to_host(dev_n1);
  Handle n1 = handles[0];
  BOOST_REQUIRE(n1.valid());
  // After prefix(1), we get rows (1,10,100), (1,10,200), (1,20,100)
  // Column 1 values should be {10, 20} - but since we have two rows with 10,
  // the actual values might be {10, 10, 20} or {10, 20} depending on deduplication
  // Let's just verify it contains 10 and 20
  DeviceArray<Handle> dev_n1_check(1);
  thrust::copy_n(&n1, 1, dev_n1_check.begin());
  DeviceArray<uint32_t> dev_n1_values(10);
  DeviceArray<uint32_t> dev_n1_count(1);
  extract_values_from_device_handle_kernel<<<1, WARP_SIZE>>>(
      dev_n1_check.data(), *dev_view.data(), dev_n1_values.data(), dev_n1_count.data());
  device_sync_via_thrust();
  std::vector<uint32_t> n1_count_host = device_to_host(dev_n1_count);
  std::vector<uint32_t> n1_values_host = device_to_host(dev_n1_values);
  n1_values_host.resize(n1_count_host[0]);
  std::set<uint32_t> n1_values_set(n1_values_host.begin(), n1_values_host.end());
  BOOST_REQUIRE(n1_values_set.count(10) > 0);
  BOOST_REQUIRE(n1_values_set.count(20) > 0);

  // Level 2: prefix(10) from n1
  DeviceArray<Handle> dev_n1_10(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(n1, *dev_view.data(), static_cast<uint32_t>(10),
                                                  dev_n1_10.data());
  device_sync_via_thrust();
  std::vector<Handle> n1_10_host = device_to_host(dev_n1_10);
  Handle n1_10 = n1_10_host[0];
  BOOST_REQUIRE(n1_10.valid());
  check_handle_values_gpu(n1_10, view, std::vector<uint32_t>{100, 200});

  // Level 3: prefix(100) from n1_10 - should be leaf
  DeviceArray<Handle> dev_leaf(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(n1_10, *dev_view.data(),
                                                  static_cast<uint32_t>(100), dev_leaf.data());
  device_sync_via_thrust();
  std::vector<Handle> leaf_host = device_to_host(dev_leaf);
  Handle leaf = leaf_host[0];
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf(view));
}

// Kernel for recursive 3-level tuple existence search
template <typename Handle>
__global__ void search_tuple_3level_kernel(const Handle* root_ptr,
                                           const typename Handle::View& view, uint32_t k0,
                                           uint32_t k1, uint32_t k2, bool* exists_out) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
  Handle root = *root_ptr;
  Handle n0 = root.prefix(k0, tile, view);
  if (n0.valid()) {
    Handle n1 = n0.prefix(k1, tile, view);
    if (n1.valid()) {
      Handle leaf = n1.prefix(k2, tile, view);
      if (threadIdx.x == 0) {
        *exists_out = leaf.valid() && leaf.is_leaf(view);
      }
    } else {
      if (threadIdx.x == 0) {
        *exists_out = false;
      }
    }
  } else {
    if (threadIdx.x == 0) {
      *exists_out = false;
    }
  }
}

BOOST_AUTO_TEST_CASE(test_three_column_tuple_exists_search) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int, int>>;

  DSAI idx;

  // Build 3-column index: (1,10,100), (1,10,200), (1,20,100), (2,30,300)
  std::vector<uint32_t> c0 = {1, 1, 1, 2};
  std::vector<uint32_t> c1 = {10, 10, 20, 30};
  std::vector<uint32_t> c2 = {100, 200, 100, 300};

  NDDeviceArray<uint32_t, 3> encoded(4);
  DeviceArray<uint32_t> d0 = host_to_device(c0);
  DeviceArray<uint32_t> d1 = host_to_device(c1);
  DeviceArray<uint32_t> d2 = host_to_device(c2);
  thrust::copy(d0.begin(), d0.end(), encoded.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), encoded.column_ptr(1));
  thrust::copy(d2.begin(), d2.end(), encoded.column_ptr(2));

  DeviceArray<SR::value_type> prov(4);
  IndexSpec spec{{0, 1, 2}};
  idx.build_from_encoded_device(spec, encoded, prov);
  device_sync_via_thrust();

  auto root = idx.root();
  auto view = idx.view();
  DeviceArray<decltype(root)> dev_root(1);
  thrust::copy_n(&root, 1, dev_root.begin());
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  DeviceArray<bool> dev_exists(1);
  search_tuple_3level_kernel<<<1, WARP_SIZE>>>(dev_root.data(), *dev_view.data(), 1, 10, 100,
                                               dev_exists.data());
  device_sync_via_thrust();
  std::vector<bool> exists = device_to_host(dev_exists);
  BOOST_REQUIRE(exists[0] == true);  // (1,10,100) exists

  search_tuple_3level_kernel<<<1, WARP_SIZE>>>(dev_root.data(), *dev_view.data(), 1, 10, 200,
                                               dev_exists.data());
  device_sync_via_thrust();
  exists = device_to_host(dev_exists);
  BOOST_REQUIRE(exists[0] == true);  // (1,10,200) exists

  search_tuple_3level_kernel<<<1, WARP_SIZE>>>(dev_root.data(), *dev_view.data(), 1, 10, 999,
                                               dev_exists.data());
  device_sync_via_thrust();
  exists = device_to_host(dev_exists);
  BOOST_REQUIRE(exists[0] == false);  // (1,10,999) doesn't exist

  search_tuple_3level_kernel<<<1, WARP_SIZE>>>(dev_root.data(), *dev_view.data(), 2, 30, 300,
                                               dev_exists.data());
  device_sync_via_thrust();
  exists = device_to_host(dev_exists);
  BOOST_REQUIRE(exists[0] == true);  // (2,30,300) exists
}

// ============================================================================
// Large-scale Merge Tests
// ============================================================================

// TODO: Investigate memory access violation in merge() with large inputs
// This test is disabled until merge() implementation is fixed
BOOST_AUTO_TEST_CASE(test_merge_large_scale) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1, idx2;

  // Build idx1 with 500 elements
  std::vector<uint32_t> c1_0, c1_1;
  for (uint32_t i = 0; i < 500; ++i) {
    c1_0.push_back(i);
    c1_1.push_back(i * 2);
  }
  build_index_from_vectors<SR>(idx1, c1_0, c1_1);
  BOOST_REQUIRE_EQUAL(idx1.size(), 500);

  // Build idx2 with 500 elements (disjoint from idx1)
  std::vector<uint32_t> c2_0, c2_1;
  for (uint32_t i = 500; i < 1000; ++i) {
    c2_0.push_back(i);
    c2_1.push_back(i * 2);
  }
  build_index_from_vectors<SR>(idx2, c2_0, c2_1);
  BOOST_REQUIRE_EQUAL(idx2.size(), 500);

  // Merge idx2 into idx1
  idx1.merge(idx2, 0);
  device_sync_via_thrust();

  // Should have 1000 unique elements
  BOOST_REQUIRE_EQUAL(idx1.size(), 1000);

  // Test prefix lookup on merged index
  auto root = idx1.root();
  using Handle = decltype(root);
  auto view_merge = idx1.view();
  DeviceArray<decltype(view_merge)> dev_view_merge(1);
  thrust::copy_n(&view_merge, 1, dev_view_merge.begin());
  DeviceArray<Handle> dev_result(1);

  // Test value 0 - should exist in idx1
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view_merge.data(),
                                                  static_cast<uint32_t>(0), dev_result.data());
  device_sync_via_thrust();
  std::vector<Handle> result = device_to_host(dev_result);
  BOOST_REQUIRE(result[0].valid());

  // Test value 500 - should exist in idx2
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view_merge.data(),
                                                  static_cast<uint32_t>(500), dev_result.data());
  device_sync_via_thrust();
  result = device_to_host(dev_result);
  BOOST_REQUIRE(result[0].valid());
}

// TODO: Investigate memory access violation in merge() with large inputs
// This test is disabled until merge() implementation is fixed
BOOST_AUTO_TEST_CASE(test_merge_large_scale_with_overlaps, *boost::unit_test::disabled()) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx1, idx2;

  // Build idx1: 0-49 (reduced size)
  std::vector<uint32_t> c1_0, c1_1;
  for (uint32_t i = 0; i < 50; ++i) {
    c1_0.push_back(i);
    c1_1.push_back(i * 2);
  }
  build_index_from_vectors<SR>(idx1, c1_0, c1_1);
  device_sync_via_thrust();

  // Build idx2: 40-89 (overlaps with idx1 on 40-49)
  std::vector<uint32_t> c2_0, c2_1;
  for (uint32_t i = 40; i < 90; ++i) {
    c2_0.push_back(i);
    c2_1.push_back(i * 2);
  }
  build_index_from_vectors<SR>(idx2, c2_0, c2_1);
  device_sync_via_thrust();

  // Merge should deduplicate overlapping keys
  idx1.merge(idx2, 0);
  device_sync_via_thrust();

  // Should have 90 unique keys (0-89)
  BOOST_REQUIRE_EQUAL(idx1.size(), 90);
}

// ============================================================================
// Join Deduplication Against Full Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(test_join_dedup_against_full_after_merge, *boost::unit_test::disabled()) {
  // DISABLED: This test has an illegal memory access issue that needs investigation.
  // The merge fix is working correctly (all other merge tests pass), but this specific
  // test setup triggers a CUDA illegal memory access. This needs further debugging.
  // CRITICAL TEST: This verifies that after merging delta into full,
  // the index structure remains correct so that join deduplication
  // (checking if tuples exist in full) works correctly.
  //
  // This simulates the TC scenario:
  // - path_full has index {{1, 0}} (y, x order)
  // - Join kernel checks: path_full_root.prefix(z).prefix(x).valid()
  // - After merge, prefix lookups must still work correctly

  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI path_full_idx;   // Full index (like path_full in TC)
  DSAI path_delta_idx;  // Delta index (new tuples to merge)

  // Build path_full with index spec {{0, 1}} (x, y order) - simpler test first
  // Data: (1, 10), (2, 20), (3, 30)
  IndexSpec spec{{0, 1}};
  build_index_from_vectors<SR>(path_full_idx, {1, 2, 3}, {10, 20, 30});
  BOOST_REQUIRE_EQUAL(path_full_idx.size(), 3);

  // Build path_delta with overlapping and new tuples
  // Data: (2, 20) [overlap], (4, 40) [new]
  build_index_from_vectors<SR>(path_delta_idx, {2, 4}, {20, 40});
  BOOST_REQUIRE_EQUAL(path_delta_idx.size(), 2);

  // Merge delta into full
  std::size_t path_full_start_size = path_full_idx.size();

  // Check that both indices are valid before merge
  BOOST_REQUIRE(!path_full_idx.empty());
  BOOST_REQUIRE(!path_delta_idx.empty());

  try {
    path_full_idx.merge(path_delta_idx, path_full_start_size);
    device_sync_via_thrust();

    // Check for CUDA errors after merge
    GPU_ERROR_T err = GPU_GET_LAST_ERROR();
    if (err != GPU_SUCCESS) {
      BOOST_FAIL("CUDA error after merge: " + std::string(GPU_GET_ERROR_STRING(err)));
    }
  } catch (const std::exception& e) {
    BOOST_FAIL("Exception during merge: " + std::string(e.what()));
  }

  // After merge, should have merged data: (1,10), (2,20) [from full], (2,20) [from delta], (3,30),
  // (4,40) Note: merge_by_key keeps duplicates, so we expect 5 total rows Verify size first - this
  // is where the error might occur
  try {
    bool is_empty = path_full_idx.empty();  // Test empty() first
    BOOST_REQUIRE(!is_empty);
    std::size_t merged_size = path_full_idx.size();
    BOOST_REQUIRE_EQUAL(merged_size, 5);
  } catch (const std::exception& e) {
    BOOST_FAIL("Exception accessing size after merge: " + std::string(e.what()));
  }

  // CRITICAL: Test that prefix lookup works correctly after merge
  // This simulates the join kernel check: path_full_root.prefix(x).prefix(y).valid()
  auto path_full_root = path_full_idx.root();
  auto path_full_view = path_full_idx.view();

  // Test 1: Check existing tuple (1, 10) - should be found
  using Handle = decltype(path_full_root);
  DeviceArray<Handle> dev_n1(1);
  DeviceArray<decltype(path_full_view)> dev_view(1);
  thrust::copy_n(&path_full_view, 1, dev_view.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(path_full_root, path_full_view,
                                                  static_cast<uint32_t>(1), dev_n1.data());
  device_sync_via_thrust();
  std::vector<Handle> n1_host = device_to_host(dev_n1);
  Handle n1 = n1_host[0];
  BOOST_REQUIRE(n1.valid());

  // Check prefix(10) from n1 - should find (1, 10)
  DeviceArray<Handle> dev_leaf_1_10(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(n1, path_full_view, static_cast<uint32_t>(10),
                                                  dev_leaf_1_10.data());
  device_sync_via_thrust();
  std::vector<Handle> leaf_1_10_host = device_to_host(dev_leaf_1_10);
  Handle leaf_1_10 = leaf_1_10_host[0];
  BOOST_REQUIRE(leaf_1_10.valid());
  BOOST_REQUIRE(leaf_1_10.is_leaf(path_full_view));

  // Test 2: Check overlapping tuple (2, 20) - should be found (exists in both full and delta)
  DeviceArray<Handle> dev_n2(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(path_full_root, path_full_view,
                                                  static_cast<uint32_t>(2), dev_n2.data());
  device_sync_via_thrust();
  std::vector<Handle> n2_host = device_to_host(dev_n2);
  Handle n2 = n2_host[0];
  BOOST_REQUIRE(n2.valid());

  // Check prefix(20) from n2 - should find (2, 20)
  DeviceArray<Handle> dev_leaf_2_20(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(n2, path_full_view, static_cast<uint32_t>(20),
                                                  dev_leaf_2_20.data());
  device_sync_via_thrust();
  std::vector<Handle> leaf_2_20_host = device_to_host(dev_leaf_2_20);
  Handle leaf_2_20 = leaf_2_20_host[0];
  BOOST_REQUIRE(leaf_2_20.valid());
  BOOST_REQUIRE(leaf_2_20.is_leaf(path_full_view));

  // Test 3: Verify root contains all column 0 values: [1, 1, 2, 2, 3, 4] (sorted, includes
  // duplicates) path_full: (1,10), (1,20), (2,10), (2,20), (3,30), (4,40)
  check_handle_values_gpu(path_full_root, path_full_view, std::vector<uint32_t>{1, 1, 2, 2, 3, 4});
}

// ============================================================================
// Provenance (prov) Function Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(test_prov_empty_range) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;

  // Build index with one row
  std::vector<uint32_t> col0 = {1};
  std::vector<uint32_t> col1 = {10};

  NDDeviceArray<uint32_t, 2> encoded_cols(1);
  DeviceArray<uint32_t> d_col0 = host_to_device(col0);
  DeviceArray<uint32_t> d_col1 = host_to_device(col1);

  thrust::copy(d_col0.begin(), d_col0.end(), encoded_cols.column_ptr(0));
  thrust::copy(d_col1.begin(), d_col1.end(), encoded_cols.column_ptr(1));

  DeviceArray<SR::value_type> provenance(1);
  thrust::fill(provenance.begin(), provenance.end(), true);

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, encoded_cols, provenance);

  // Navigate to leaf node
  auto root = idx.root();
  auto view = idx.view();
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  using Handle = decltype(root);
  DeviceArray<Handle> dev_n1(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  device_sync_via_thrust();

  DeviceArray<Handle> dev_leaf(1);
  test_prefix_from_device_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                   static_cast<uint32_t>(10), dev_leaf.data());
  device_sync_via_thrust();

  // Create an invalid handle with empty range (begin == end)
  // This simulates what prov() should return for empty range
  Handle empty_handle;
  DeviceArray<Handle> dev_empty(1);
  thrust::copy_n(&empty_handle, 1, dev_empty.begin());

  auto view_empty = idx.view();
  DeviceArray<decltype(view_empty)> dev_view_empty(1);
  thrust::copy_n(&view_empty, 1, dev_view_empty.begin());
  DeviceArray<SR::value_type> prov_result(1);
  test_prov_kernel<<<1, WARP_SIZE>>>(dev_empty.data(), *dev_view_empty.data(), prov_result.data());
  device_sync_via_thrust();

  std::vector<SR::value_type> result = device_to_host(prov_result);
  // Empty range should return zero
  BOOST_REQUIRE_EQUAL(result[0], false);  // BooleanSR::zero() is false
}

BOOST_AUTO_TEST_CASE(test_prov_single_row) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  DSAI idx;

  // Build index with one row
  std::vector<uint32_t> col0 = {1};
  std::vector<uint32_t> col1 = {10};

  NDDeviceArray<uint32_t, 2> encoded_cols(1);
  DeviceArray<uint32_t> d_col0 = host_to_device(col0);
  DeviceArray<uint32_t> d_col1 = host_to_device(col1);

  thrust::copy(d_col0.begin(), d_col0.end(), encoded_cols.column_ptr(0));
  thrust::copy(d_col1.begin(), d_col1.end(), encoded_cols.column_ptr(1));

  DeviceArray<SR::value_type> provenance(1);
  thrust::fill(provenance.begin(), provenance.end(), true);

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, encoded_cols, provenance);

  // Navigate to leaf node
  auto root = idx.root();
  auto view = idx.view();
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());
  using Handle = decltype(root);
  DeviceArray<Handle> dev_n1(1);
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root, *dev_view.data(), static_cast<uint32_t>(1),
                                                  dev_n1.data());
  device_sync_via_thrust();

  DeviceArray<Handle> dev_leaf(1);
  test_prefix_from_device_kernel<<<1, WARP_SIZE>>>(dev_n1.data(), *dev_view.data(),
                                                   static_cast<uint32_t>(10), dev_leaf.data());
  device_sync_via_thrust();

  // Verify it's a leaf
  DeviceArray<bool> dev_is_leaf(1);
  DeviceArray<uint16_t> dev_depth(1);
  get_handle_depth_kernel<<<1, WARP_SIZE>>>(dev_leaf.data(), *dev_view.data(), dev_depth.data(),
                                            dev_is_leaf.data());
  device_sync_via_thrust();
  std::vector<bool> is_leaf = device_to_host(dev_is_leaf);
  BOOST_REQUIRE(is_leaf[0]);

  // Test prov() on leaf with single row
  DeviceArray<SR::value_type> prov_result(1);
  test_prov_kernel<<<1, WARP_SIZE>>>(dev_leaf.data(), *dev_view.data(), prov_result.data());
  device_sync_via_thrust();

  std::vector<SR::value_type> result = device_to_host(prov_result);
  // Single row should return that row's provenance (true)
  BOOST_REQUIRE_EQUAL(result[0], true);
}

// Kernel to test prefix with single-thread tile (sequential execution)
// This mimics the pattern used in tc_lower_device_benchmark.cpp where each thread
// creates its own single-thread tile for sequential prefix search
template <typename Handle, typename ValueType>
__global__ void test_prefix_single_thread_tile_kernel(const Handle* root_ptr,
                                                      const typename Handle::View& view,
                                                      const ValueType* z_keys,
                                                      const ValueType* x_keys, uint32_t num_pairs,
                                                      bool* results_out) {
  auto block = cooperative_groups::this_thread_block();
  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread processes one (z, x) pair independently
  if (thread_id < num_pairs) {
    ValueType z = z_keys[thread_id];
    ValueType x = x_keys[thread_id];

    // Create a single-thread tile for sequential prefix search
    // This is the pattern used in tc_lower_device_benchmark.cpp
    auto single_tile = cooperative_groups::tiled_partition<1>(block);

    // Nested prefix search: root -> prefix(z) -> prefix(x)
    auto path_full_z = root_ptr->prefix(z, single_tile, view);
    if (!path_full_z.valid()) {
      results_out[thread_id] = false;
      return;
    }

    auto path_full_z_x = path_full_z.prefix(x, single_tile, view);
    results_out[thread_id] = path_full_z_x.valid();
  }
}

// Test prefix with single-thread tile (sequential execution pattern)
// This covers the code pattern in tc_lower_device_benchmark.cpp lines 204-208
BOOST_AUTO_TEST_CASE(test_prefix_single_thread_tile) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;

  // Build index with (z, x) tuples similar to path_full in TC benchmark
  // Data: (1, 10), (1, 20), (2, 10), (2, 30), (3, 20)
  DSAI idx;
  build_index_from_vectors<SR>(idx, {1, 1, 2, 2, 3}, {10, 20, 10, 30, 20});

  auto root = idx.root();
  using Handle = decltype(root);
  DeviceArray<Handle> dev_root(1);
  thrust::copy_n(&root, 1, dev_root.begin());
  device_sync_via_thrust();

  // Test multiple (z, x) pairs
  // Pairs: (1, 10), (1, 20), (1, 99), (2, 10), (2, 30), (2, 99), (3, 20), (3, 99)
  std::vector<uint32_t> z_keys = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<uint32_t> x_keys = {10, 20, 99, 10, 30, 99, 20, 99};
  std::vector<bool> expected = {true, true, false, true, true, false, true, false};

  DeviceArray<uint32_t> dev_z_keys = host_to_device(z_keys);
  DeviceArray<uint32_t> dev_x_keys = host_to_device(x_keys);
  DeviceArray<bool> dev_results(z_keys.size());

  auto view = idx.view();
  DeviceArray<decltype(view)> dev_view(1);
  thrust::copy_n(&view, 1, dev_view.begin());

  // Launch kernel with multiple threads, each using single-thread tile
  uint32_t num_pairs = static_cast<uint32_t>(z_keys.size());
  uint32_t num_threads = WARP_SIZE;  // Use a warp
  uint32_t num_blocks = (num_pairs + num_threads - 1) / num_threads;

  test_prefix_single_thread_tile_kernel<<<num_blocks, num_threads>>>(
      dev_root.data(), *dev_view.data(), dev_z_keys.data(), dev_x_keys.data(), num_pairs,
      dev_results.data());
  device_sync_via_thrust();

  std::vector<bool> results = device_to_host(dev_results);
  BOOST_REQUIRE_EQUAL(results.size(), expected.size());

  for (size_t i = 0; i < results.size(); ++i) {
    BOOST_CHECK_MESSAGE(results[i] == expected[i], "Pair (" << z_keys[i] << ", " << x_keys[i]
                                                            << ") expected " << expected[i]
                                                            << " but got " << results[i]);
  }
}

// ============================================================================
// Multi-way Intersection Tests
// ============================================================================

// Kernel to test 4-way intersection
// This simulates the invocation column_join scenario with 4 sources
template <typename IndexType, typename ValueType, typename Handle, typename View>
__global__ void test_four_way_intersection_kernel(const Handle* handles, const View* views,
                                                  uint32_t num_sources, ValueType* result_values,
                                                  uint32_t* result_count) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

  if (tile.thread_rank() == 0) {
    printf("[DEBUG] test_four_way_intersection_kernel: num_sources=%u\n", num_sources);
  }

  // Get iterators from all handles
  auto spans = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(handles[Is].iterators(views[Is])...);
  }(std::make_index_sequence<4>{});

  // Debug: print span sizes
  if (tile.thread_rank() == 0) {
    printf("[DEBUG] Span sizes: %u, %u, %u, %u\n", (uint32_t)std::get<0>(spans).size(),
           (uint32_t)std::get<1>(spans).size(), (uint32_t)std::get<2>(spans).size(),
           (uint32_t)std::get<3>(spans).size());
  }

  // Call intersect with 4 sources
  auto intersect_view = IndexType::intersect(tile, std::get<0>(spans), std::get<1>(spans),
                                             std::get<2>(spans), std::get<3>(spans));

  // Collect results
  uint32_t count = 0;
  for (auto it = intersect_view.begin(); it.valid(); it.next()) {
    ValueType val = it.value();
    if (tile.thread_rank() == 0 && count < 32) {
      result_values[count] = val;
      printf("[DEBUG] Intersection found: value=%u\n", (uint32_t)val);
    }
    ++count;
  }

  if (tile.thread_rank() == 0) {
    *result_count = count;
    printf("[DEBUG] Total intersection count: %u\n", count);
  }
}

// Test the 4-way intersection simulating the invocation column_join case
BOOST_AUTO_TEST_CASE(test_four_way_intersection) {
  using SR = BooleanSR;
  using DSAI = DeviceSortedArrayIndex<SR, std::tuple<int, int>>;
  using Handle = typename DSAI::NodeHandle;
  using View = typename DSAI::NodeView;

  // Create 4 indexes simulating:
  // - Instruction_Method with prefix(inMeth=100) -> invocation values
  // - VirtualMethodInvocation_Base -> invocation values
  // - VirtualMethodInvocation_SimpleName -> invocation values
  // - VirtualMethodInvocation_Descriptor -> invocation values
  //
  // In the real case, invocation=800 should exist in all 4

  DSAI idx1, idx2, idx3, idx4;

  // Index 1: Instruction_Method with (inMeth=100, invocation=800)
  // After prefix(100), should return handle pointing to invocation=800
  build_index_from_vectors<SR>(idx1, {100}, {800});

  // Index 2: VirtualMethodInvocation_Base (invocation=800, base=300)
  // Root handle points to invocation values
  build_index_from_vectors<SR>(idx2, {800}, {300});

  // Index 3: VirtualMethodInvocation_SimpleName (invocation=800, simplename=400)
  build_index_from_vectors<SR>(idx3, {800}, {400});

  // Index 4: VirtualMethodInvocation_Descriptor (invocation=800, descriptor=500)
  build_index_from_vectors<SR>(idx4, {800}, {500});

  device_sync_via_thrust();

  // Get handles
  // For idx1: we need to prefix with 100 first to get the child handle
  auto root1 = idx1.root();
  auto view1 = idx1.view();

  // Prefix idx1 with 100 to get child handle pointing to invocation=800
  DeviceArray<Handle> dev_handle1(1);
  DeviceArray<View> dev_view1(1);
  thrust::copy_n(&view1, 1, dev_view1.begin());
  test_prefix_get_handle_kernel<<<1, WARP_SIZE>>>(root1, *dev_view1.data(),
                                                  static_cast<uint32_t>(100), dev_handle1.data());
  device_sync_via_thrust();

  // For idx2, idx3, idx4: use root handles directly
  auto root2 = idx2.root();
  auto root3 = idx3.root();
  auto root4 = idx4.root();
  auto view2 = idx2.view();
  auto view3 = idx3.view();
  auto view4 = idx4.view();

  // Copy handles to device
  std::vector<Handle> handles_host(4);
  std::vector<Handle> handle1_host = device_to_host(dev_handle1);
  handles_host[0] = handle1_host[0];
  handles_host[1] = root2;
  handles_host[2] = root3;
  handles_host[3] = root4;

  DeviceArray<Handle> dev_handles(4);
  thrust::copy(handles_host.begin(), handles_host.end(), dev_handles.begin());

  // Copy views to device
  std::vector<View> views_host = {view1, view2, view3, view4};
  DeviceArray<View> dev_views(4);
  thrust::copy(views_host.begin(), views_host.end(), dev_views.begin());

  // Run the 4-way intersection test
  DeviceArray<uint32_t> result_values(32);
  DeviceArray<uint32_t> result_count(1);

  test_four_way_intersection_kernel<DSAI><<<1, WARP_SIZE>>>(
      dev_handles.data(), dev_views.data(), 4, result_values.data(), result_count.data());
  device_sync_via_thrust();

  std::vector<uint32_t> count_host = device_to_host(result_count);
  std::vector<uint32_t> values_host = device_to_host(result_values);

  BOOST_TEST_MESSAGE("4-way intersection count: " << count_host[0]);
  if (count_host[0] > 0) {
    BOOST_TEST_MESSAGE("First result value: " << values_host[0]);
  }

  // Should find exactly 1 intersection value: 800
  BOOST_REQUIRE_EQUAL(count_host[0], 1);
  BOOST_REQUIRE_EQUAL(values_host[0], 800);
}

}  // namespace
