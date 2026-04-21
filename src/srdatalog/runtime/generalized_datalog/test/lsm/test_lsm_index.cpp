
/**
 * @file test_gpu_lsm_index.cpp
 * @brief Unit tests for DeviceLsmIndex
 */

#define BOOST_TEST_MODULE GPULsmIndexTest
#include <boost/test/included/unit_test.hpp>

#include "gpu/device_array.h"
#include "gpu/device_lsm_index.h"
#include "semiring.h"

// RMM
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

using namespace SRDatalog;
using namespace SRDatalog::GPU;

// Type Aliases
using MySemiring = BooleanSR;
using MyTuple = std::tuple<uint32_t, uint32_t>;  // 2 columns
using MyIndex = DeviceLsmIndex<MySemiring, MyTuple>;

BOOST_AUTO_TEST_CASE(test_lsm_build_and_traverse) {
  std::cout << "=== Testing LsmIndex Build and Traverse ===" << std::endl;

  // Initialize RMM
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr, 10 * 1024 * 1024);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Setup Data
  // Col 0: [3, 1, 2, 3]
  // Col 1: [0, 9, 5, 1]

  std::vector<uint32_t> h_col0 = {3, 1, 2, 3};
  std::vector<uint32_t> h_col1 = {0, 9, 5, 1};
  std::vector<uint8_t> h_prov = {1, 1, 1, 1};  // Use uint8_t for bool transfer

  size_t num_rows = h_col0.size();

  NDDeviceArray<uint32_t, 2> device_cols(num_rows);
  DeviceArray<MySemiring::value_type> device_prov(num_rows);

  // Copy to GPU logic (simplified)
  // We need to copy into the flat buffer of NDDeviceArray
  // Layout: [Col0...][Col1...]

  cudaMemcpy(device_cols.data(), h_col0.data(), num_rows * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_cols.data() + num_rows, h_col1.data(), num_rows * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_prov.data(), h_prov.data(), num_rows * sizeof(bool), cudaMemcpyHostToDevice);

  // Build LsmIndex
  MyIndex idx;
  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, device_cols, device_prov);

  // Verify Size
  BOOST_CHECK_EQUAL(idx.size(), num_rows);

  // Verify Sorted Order via Traversing Root
  // Expected Sort Order of (Col0, Col1):
  // (1, 9) @ row 1
  // (2, 5) @ row 2
  // (3, 0) @ row 0
  // (3, 1) @ row 3

  // Actually check the implementation details via View (Whitebox testing)
  auto view = idx.view();
  std::vector<uint32_t> h_indices(num_rows);
  cudaMemcpy(h_indices.data(), view.sorted_indices_, num_rows * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  std::vector<uint32_t> expected_indices = {1, 2, 0, 3};  // Based on sort order above
  BOOST_CHECK_EQUAL_COLLECTIONS(h_indices.begin(), h_indices.end(), expected_indices.begin(),
                                expected_indices.end());

  // Test NodeHandle Traversal (Simulating Scan)
  // Iterate root handle
  auto root = idx.root();

  // We can't use generic 'values()' easily because return type is span of contiguous memory.
  // But we can check our 'prefix' logic.

  // Test Prefix Search for "3"
  // Should find range [2, 4) in indices array (row 0 and row 3)

  // Launch kernel to test device-side prefix search
}

// Kernel for testing prefix search on device
template <typename IndexType>
__global__ void test_prefix_kernel(typename IndexType::NodeView view,
                                   typename IndexType::NodeHandle root, bool* results) {
  // Warning: single thread only for simplified test

  // Search for first column = 3
  // Use cooperative group placeholder (not used in our simplistic generic implementation yet)
  namespace cg = cooperative_groups;
  auto tile = cg::tiled_partition<1>(cg::this_thread_block());  // dummy

  auto h_sub = root.prefix(3, tile, view);

  if (!h_sub.valid()) {
    results[0] = false;
    return;
  }

  // Check range [begin, end)
  // indices[begin] should be row 0
  // indices[begin+1] should be row 3
  // end - begin should be 2
  if (h_sub.begin() != 2 || h_sub.end() != 4) {
    results[1] = false;  // Range mismatch
                         // printf("Debug: %d %d\n", h_sub.begin(), h_sub.end());
  } else {
    results[1] = true;
  }

  // Drill down to next level (Col 1)
  // Values at depth 1 for these rows are 0 and 1.

  // Debug: Check values
  if (h_sub.valid()) {
    // Expecting range [2, 4)
    // Check indices
    uint32_t row_at_2 = view.sorted_indices_[2];
    uint32_t row_at_3 = view.sorted_indices_[3];
    // printf("Debug: Depth 1 Range [%d, %d). Row@2=%d, Row@3=%d\n", h_sub.begin(), h_sub.end(),
    // row_at_2, row_at_3);

    // Check values at depth 1
    uint32_t val_at_2 = view.get_value(1, 2);
    uint32_t val_at_3 = view.get_value(1, 3);
    // printf("Debug: Vals at Depth 1: Val@2=%d, Val@3=%d. Looking for 1.\n", val_at_2, val_at_3);

    if (val_at_3 != 1) {
      // If this assertion fails, issue is data access
      results[5] = false;
    } else {
      results[5] = true;
    }
  }

  auto h_leaf = h_sub.prefix(1, tile, view);  // Search for (3, 1)
  if (!h_leaf.valid()) {
    results[2] = false;
    // printf("Debug: Leaf invalid.\n");
  } else {
    results[2] = true;  // Mark as found
    if (h_leaf.is_leaf(view))
      results[3] = true;
    // Should match exactly 1 row (row 3, which is at index 3)
    if (h_leaf.begin() == 3 && h_leaf.end() == 4)
      results[4] = true;
    else {
      results[4] = false;
      // printf("Debug: Leaf range mismatch [%d, %d)\n", h_leaf.begin(), h_leaf.end());
    }
  }

  results[0] = true;  // Reach end
}

BOOST_AUTO_TEST_CASE(test_lsm_prefix_kernel) {
  // Reuse setup code or structure... for brevity merging into one big test logic style
  // ... (Setup Data same as above) ...

  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr, 10 * 1024 * 1024);
  rmm::mr::set_current_device_resource(&pool_mr);

  std::vector<uint32_t> h_col0 = {3, 1, 2, 3};
  std::vector<uint32_t> h_col1 = {0, 9, 5, 1};
  std::vector<uint8_t> h_prov = {1, 1, 1, 1};  // Use uint8_t
  size_t num_rows = h_col0.size();

  NDDeviceArray<uint32_t, 2> device_cols(num_rows);
  DeviceArray<MySemiring::value_type> device_prov(num_rows);

  cudaMemcpy(device_cols.data(), h_col0.data(), num_rows * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_cols.data() + num_rows, h_col1.data(), num_rows * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_prov.data(), h_prov.data(), num_rows * sizeof(bool), cudaMemcpyHostToDevice);

  MyIndex idx;
  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, device_cols, device_prov);

  DeviceArray<bool> d_results(10);  // buffer for flags
  cudaMemset(d_results.data(), 0, 10);

  test_prefix_kernel<MyIndex><<<1, 1>>>(idx.view(), idx.root(), d_results.data());
  cudaDeviceSynchronize();

  std::vector<uint8_t> h_results(10);
  cudaMemcpy(h_results.data(), d_results.data(), 10 * sizeof(bool), cudaMemcpyDeviceToHost);

  BOOST_CHECK(h_results[0]);  // Kernel ran ok
  BOOST_CHECK(h_results[1]);  // First level range ok
  BOOST_CHECK(h_results[2]);  // Second level search ok
  BOOST_CHECK(h_results[3]);  // Is leaf
  BOOST_CHECK(h_results[4]);  // Leaf range ok
}

BOOST_AUTO_TEST_CASE(test_lsm_compact) {
  std::cout << "=== Testing LsmIndex Compact ===" << std::endl;

  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr, 10 * 1024 * 1024);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Setup Unsorted Data
  // Col: [3, 1, 2]
  std::vector<uint32_t> h_col0 = {3, 1, 2};
  std::vector<uint32_t> h_col1 = {10, 20, 30};  // dummy
  size_t num_rows = 3;

  NDDeviceArray<uint32_t, 2> device_cols(num_rows);
  DeviceArray<MySemiring::value_type> device_prov(num_rows);  // all true

  cudaMemcpy(device_cols.data(), h_col0.data(), num_rows * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(device_cols.data() + num_rows, h_col1.data(), num_rows * 4, cudaMemcpyHostToDevice);

  MyIndex idx;
  IndexSpec spec{{0, 1}};
  idx.build_from_encoded_device(spec, device_cols, device_prov);

  // Check Indices: Should be [1, 2, 0] (values 1, 2, 3)
  auto view = idx.view();
  std::vector<uint32_t> h_indices(num_rows);
  cudaMemcpy(h_indices.data(), view.sorted_indices_, num_rows * 4, cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_indices[0], 1);
  BOOST_CHECK_EQUAL(h_indices[1], 2);
  BOOST_CHECK_EQUAL(h_indices[2], 0);

  // Compact!
  idx.compact(device_cols, device_prov);

  // After compact:
  // 1. Data should be sorted: [1, 2, 3] in physical memory
  // 2. Indices should be reset to [0, 1, 2]

  std::vector<uint32_t> h_new_indices(num_rows);
  cudaMemcpy(h_new_indices.data(), idx.view().sorted_indices_, num_rows * 4,
             cudaMemcpyDeviceToHost);

  // Check indices reset
  BOOST_CHECK_EQUAL(h_new_indices[0], 0);
  BOOST_CHECK_EQUAL(h_new_indices[1], 1);
  BOOST_CHECK_EQUAL(h_new_indices[2], 2);

  // Check Data
  std::vector<uint32_t> h_new_col0(num_rows);
  cudaMemcpy(h_new_col0.data(), device_cols.data(), num_rows * 4, cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_new_col0[0], 1);
  BOOST_CHECK_EQUAL(h_new_col0[1], 2);
  BOOST_CHECK_EQUAL(h_new_col0[2], 3);

  // Verify Col1 also moved (Permutation applies to all columns)
  // Original Col1: [10, 20, 30] corresponding to [3, 1, 2]
  // Sorted Order tuples: (1, 20), (2, 30), (3, 10)
  std::vector<uint32_t> h_new_col1(num_rows);
  cudaMemcpy(h_new_col1.data(), device_cols.data() + num_rows, num_rows * 4,
             cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_new_col1[0], 20);
  BOOST_CHECK_EQUAL(h_new_col1[1], 30);
  BOOST_CHECK_EQUAL(h_new_col1[2], 10);
}

BOOST_AUTO_TEST_CASE(test_lsm_merge) {
  std::cout << "=== Testing LsmIndex Merge ===" << std::endl;

  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr, 10 * 1024 * 1024);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Data 1: [(3,0), (1,9)] -> sorted: (1,9), (3,0)  [RowIDs: 1, 0]
  // Data 2: [(2,5)]        -> sorted: (2,5)       [RowIDs: 0 relative to Data 2]

  // Combined Data (logical): [(3,0), (1,9), (2,5)]
  // RowIDs: 0, 1, 2
  // We want to merge index of Data 1 and index of Data 2.
  // Index 1 (local): [1, 0]
  // Index 2 (local): [0]
  // Offset for Index 2: 2 (Data 1 has size 2)
  // Shifted Index 2: [0+2] = [2]
  // Merged Index: (1,9) < (2,5) < (3,0) -> RowIDs [1, 2, 0]

  std::vector<uint32_t> h_col0_1 = {3, 1};
  std::vector<uint32_t> h_col1_1 = {0, 9};
  std::vector<uint8_t> h_prov_1 = {1, 1};

  std::vector<uint32_t> h_col0_2 = {2};
  std::vector<uint32_t> h_col1_2 = {5};
  std::vector<uint8_t> h_prov_2 = {1};

  // Build Index 1
  size_t size1 = 2;
  NDDeviceArray<uint32_t, 2> cols1(size1);
  DeviceArray<MySemiring::value_type> prov1(size1);
  cudaMemcpy(cols1.data(), h_col0_1.data(), size1 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(cols1.data() + size1, h_col1_1.data(), size1 * 4, cudaMemcpyHostToDevice);

  MyIndex idx1;
  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded_device(spec, cols1, prov1);

  // Build Index 2
  size_t size2 = 1;
  NDDeviceArray<uint32_t, 2> cols2(size2);
  DeviceArray<MySemiring::value_type> prov2(size2);
  cudaMemcpy(cols2.data(), h_col0_2.data(), size2 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(cols2.data() + size2, h_col1_2.data(), size2 * 4, cudaMemcpyHostToDevice);

  MyIndex idx2;
  idx2.build_from_encoded_device(spec, cols2, prov2);

  // Create Unified Data Buffer for Merge
  // This simulates Relation appending data
  size_t total_size = size1 + size2;
  NDDeviceArray<uint32_t, 2> total_cols(total_size);
  DeviceArray<MySemiring::value_type> total_prov(total_size);

  // Copy part 1
  cudaMemcpy(total_cols.data(), cols1.data(), size1 * 4, cudaMemcpyDeviceToDevice);
  cudaMemcpy(total_cols.data() + total_size, cols1.data() + size1, size1 * 4,
             cudaMemcpyDeviceToDevice);  // stride logic!

  // Wait, stride for total_cols is `total_size`.
  // We need to be careful copying columns.
  // Col 0 part 1 -> total_cols.data()
  // Col 0 part 2 -> total_cols.data() + size1
  // Col 1 part 1 -> total_cols.data() + total_size
  // Col 1 part 2 -> total_cols.data() + total_size + size1

  cudaMemcpy(total_cols.data(), cols1.data(), size1 * 4, cudaMemcpyDeviceToDevice);
  cudaMemcpy(total_cols.data() + size1, cols2.data(), size2 * 4,
             cudaMemcpyDeviceToDevice);  // Append Col 0

  cudaMemcpy(total_cols.data() + total_size, cols1.data() + size1, size1 * 4,
             cudaMemcpyDeviceToDevice);  // Col 1 part 1
  cudaMemcpy(total_cols.data() + total_size + size1, cols2.data() + size2, size2 * 4,
             cudaMemcpyDeviceToDevice);  // Append Col 1

  // Update data pointer in idx1 (the target of merge)
  idx1.bind_data(total_cols, total_prov);

  // Merge
  idx1.merge(idx2, size1);

  // Verify size
  BOOST_CHECK_EQUAL(idx1.size(), 3);

  // Verify Indices: [1, 2, 0]
  auto view = idx1.view();
  std::vector<uint32_t> h_indices(3);
  cudaMemcpy(h_indices.data(), view.sorted_indices_, 3 * 4, cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(h_indices[0], 1);
  BOOST_CHECK_EQUAL(h_indices[1], 2);
  BOOST_CHECK_EQUAL(h_indices[2], 0);
}
