#define BOOST_TEST_MODULE gpu_runtime_test
#include <boost/test/included/unit_test.hpp>

#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include "ast.h"
#include "gpu/runtime/store.h"
#include "gpu/runtime/tmp_helpers.h"
#include "mir.h"

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;
using namespace SRDatalog::GPU;
using namespace SRDatalog::GPU::state;
using namespace SRDatalog::mir;

namespace {

// Define Var types using proper CTString syntax
using VarY = Var<decltype("y"_s)>;
using VarZ = Var<decltype("z"_s)>;
using VarX = Var<decltype("x"_s)>;

// Create VarPosMap using ComputeVarPosMap (same approach as the runtime)
// Maps VarY -> position 0, VarZ -> position 1, VarX -> position 2
using JoinPlan = std::tuple<VarY, VarZ, VarX>;
using MyVarPosMap = typename ComputeVarPosMap<JoinPlan>::type;

using ValueType = uint32_t;

// ============================================================================
// CUDA Kernel to Test VarStore
// ============================================================================

__global__ void test_varstore_kernel(ValueType* results) {
  VarStore<MyVarPosMap, ValueType> vars;

  // Set variables
  vars.set<VarY>(100);  // Stores at position 0
  vars.set<VarZ>(200);  // Stores at position 1
  vars.set<VarX>(300);  // Stores at position 2

  // Get variables
  ValueType y = vars.get<VarY>();  // Should return 100
  ValueType z = vars.get<VarZ>();  // Should return 200
  ValueType x = vars.get<VarX>();  // Should return 300

  // Store results
  results[0] = y;
  results[1] = z;
  results[2] = x;

  // Test ExtractTermValues
  using OutputTerms = std::tuple<VarX, VarZ>;
  ValueType output[2];
  ExtractTermValues<MyVarPosMap, ValueType, OutputTerms>::get_array(vars, output);

  // output[0] should be 300 (x), output[1] should be 200 (z)
  results[3] = output[0];  // x value
  results[4] = output[1];  // z value
}

// ============================================================================
// Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(test_varstore_basic) {
  // Allocate device memory for results
  thrust::device_vector<ValueType> dev_results(5);

  // Launch kernel
  test_varstore_kernel<<<1, 1>>>(thrust::raw_pointer_cast(dev_results.data()));

  // Wait for kernel to complete
  GPU_ERROR_T err = GPU_DEVICE_SYNCHRONIZE();
  BOOST_REQUIRE_EQUAL(err, GPU_SUCCESS);

  // Copy results back to host
  std::vector<ValueType> host_results(5);
  thrust::copy(dev_results.begin(), dev_results.end(), host_results.begin());

  // Verify results
  BOOST_REQUIRE_EQUAL(host_results[0], 100u);  // y = 100
  BOOST_REQUIRE_EQUAL(host_results[1], 200u);  // z = 200
  BOOST_REQUIRE_EQUAL(host_results[2], 300u);  // x = 300
  BOOST_REQUIRE_EQUAL(host_results[3], 300u);  // output[0] = x = 300
  BOOST_REQUIRE_EQUAL(host_results[4], 200u);  // output[1] = z = 200
}

// Kernel for testing ExtractTermValues
__global__ void test_extract_term_values_kernel(ValueType* results) {
  VarStore<MyVarPosMap, ValueType> vars;
  vars.set<VarX>(42);
  vars.set<VarY>(100);
  vars.set<VarZ>(200);

  using OutputTerms = std::tuple<VarX, VarZ>;
  ExtractTermValues<MyVarPosMap, ValueType, OutputTerms>::get_array(vars, results);
}

BOOST_AUTO_TEST_CASE(test_varstore_extract_term_values) {
  // Allocate device memory for results
  thrust::device_vector<ValueType> dev_results(2);

  // Launch kernel
  test_extract_term_values_kernel<<<1, 1>>>(thrust::raw_pointer_cast(dev_results.data()));

  GPU_ERROR_T err = GPU_DEVICE_SYNCHRONIZE();
  BOOST_REQUIRE_EQUAL(err, GPU_SUCCESS);

  std::vector<ValueType> host_results(2);
  thrust::copy(dev_results.begin(), dev_results.end(), host_results.begin());

  BOOST_REQUIRE_EQUAL(host_results[0], 42u);   // x = 42
  BOOST_REQUIRE_EQUAL(host_results[1], 200u);  // z = 200
}

}  // namespace
