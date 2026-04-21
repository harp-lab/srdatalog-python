
/**
 * @file test_gpu_enhanced_id.cpp
 * @brief Unit tests for EnhancedID provenance on GPU
 */

#define BOOST_TEST_MODULE GPUEnhancedIDTest
#include <boost/test/included/unit_test.hpp>

#include "gpu/device_array.h"
#include "semiring.h"

// RMM for device memory
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

using namespace SRDatalog;

// wrapper for calling device functions from host for testing packing/unpacking
// Since EnhancedID functions are GPU_HD, we can test them on host directly.
BOOST_AUTO_TEST_CASE(test_host_packing) {
  std::cout << "=== Testing EnhancedID Packing on Host ===" << std::endl;

  uint64_t id = 0x123456789ABC;  // 48 bits (max is FFFFFFFFFFFF)
  uint16_t scope = 0x7FED;       // 15 bits (max is 7FFF)
  bool flag = true;

  EnhancedID eid(id, scope, flag);

  BOOST_CHECK_EQUAL(eid.id(), id);
  BOOST_CHECK_EQUAL(eid.scope(), scope);
  BOOST_CHECK_EQUAL(eid.flag(), flag);

  // Check * operator
  BOOST_CHECK_EQUAL(*eid, id);
}

// Global kernel to test atomic operations on GPU
__global__ void test_atomic_add_kernel(EnhancedID* data, EnhancedID val) {
  EnhancedIDSR::atomic_add(data, val);
}

__global__ void test_atomic_mul_kernel(EnhancedID* data, EnhancedID val) {
  EnhancedIDSR::atomic_mul(data, val);
}

BOOST_AUTO_TEST_CASE(test_gpu_atomic_add) {
  std::cout << "=== Testing EnhancedID Atomic Add on GPU ===" << std::endl;

  // Initialize RMM
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&cuda_mr,
                                                                       1024 * 1024);  // 1MB pool
  rmm::mr::set_current_device_resource(&pool_mr);

  // Setup data
  EnhancedID initial(100, 1, false);
  rmm::device_uvector<EnhancedID> check_vec(1, rmm::cuda_stream_default);

  // Copy initial value to GPU behavior
  // rmm::device_uvector doesn't have easy host->device assignment for single element without stream
  // construction easier to use cudamemcpy
  cudaMemcpy(check_vec.data(), &initial, sizeof(EnhancedID), cudaMemcpyHostToDevice);

  // Value to add: Higher ID (200), same scope, flag true
  EnhancedID val(200, 1, true);

  // Launch kernel (1 block, 1 thread)
  test_atomic_add_kernel<<<1, 1>>>(check_vec.data(), val);
  cudaDeviceSynchronize();

  // Copy pack
  EnhancedID result;
  cudaMemcpy(&result, check_vec.data(), sizeof(EnhancedID), cudaMemcpyDeviceToHost);

  // Expected: ID 200 (max of 100, 200), Scope 1, Flag true (false || true)
  BOOST_CHECK_EQUAL(result.id(), 200);
  BOOST_CHECK_EQUAL(result.scope(), 1);
  BOOST_CHECK_EQUAL(result.flag(), true);

  // Test Case 2: Lower ID, Flag True
  EnhancedID val2(50, 2, true);
  // Reset
  cudaMemcpy(check_vec.data(), &initial, sizeof(EnhancedID),
             cudaMemcpyHostToDevice);  // Back to 100
  test_atomic_add_kernel<<<1, 1>>>(check_vec.data(), val2);
  cudaDeviceSynchronize();
  cudaMemcpy(&result, check_vec.data(), sizeof(EnhancedID), cudaMemcpyDeviceToHost);

  // Expected: ID 100 (max of 100, 50), Scope 1 (kept from winner), Flag true (initial.flag(false)
  // || val2.flag(true) ? wait, add logic is: if a > b returns a. if b > a returns b. if a == b
  // returns merged. Here 100 > 50, so should return initial (100, 1, false). Ah, the logic in
  // semiring.h was:
  /*
  if (a.id() > b.id()) return a;
  if (b.id() > a.id()) return b;
  return EnhancedID(a.id(), a.scope(), a.flag() || b.flag());
  */
  // So adding (50) to (100) -> 100 wins completely. Scope/Flag of 50 are ignored.

  BOOST_CHECK_EQUAL(result.id(), 100);
  BOOST_CHECK_EQUAL(result.scope(), 1);
  BOOST_CHECK_EQUAL(result.flag(), false);
}
