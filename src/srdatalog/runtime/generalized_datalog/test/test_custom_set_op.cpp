
#include <iostream>

// Include rocprim headers before thrust/custom_set_op to ensure namespace visibility
// This is needed for hipcub/rocprim backend on ROCm 7.0/7.1
#ifdef USE_ROCm
#include <rocprim/config.hpp>  // Ensure namespace macros are defined
#include <rocprim/device/device_scan.hpp>  // Needed for inclusive_scan
#include <rocprim/block/block_radix_rank.hpp>  // Needed for hipcub BlockRadixRank
#endif

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <vector>

// Include the custom set operation header
#include "gpu/runtime/custom_set_op.h"
#include "semiring.h"  // For NaturalBag
// Note: We avoid including gpu/init.h here because it includes RMM headers
// which may not be available when compiling with -x hip. Instead, we manually
// initialize the GPU device.
#include "gpu/gpu_api.h"  // For GPU API macros

// Define a simple MatchOp that merges values using the Semiring
template <typename SR>
struct MergeMatchOp {
  using ValueType = typename SR::value_type;

  bool emit_matches;

  __host__ __device__ MergeMatchOp(bool emit = false) : emit_matches(emit) {}

  // Function called on match: key, value_from_new, value_from_full
  __device__ auto operator()(int /*key*/, ValueType val_new, ValueType val_full) {
    ValueType merged = SR::add(val_new, val_full);
    // If emit_matches is true, we emit the merged value.
    // If false, we (conceptually) update full in place and don't emit to difference result.
    // Note: In-place update of 'val_full' here doesn't affect global memory because it's a copy.
    // check if merged value different from val_full, if so, change emit_matches to true
    if (merged != val_full) {
      return cuda::std::make_pair(true, merged);
    }
    return cuda::std::make_pair(emit_matches, merged);
  }
};

void test_set_difference_with_merge() {
  std::cout << "Testing set_difference_by_key_with_match_custom..." << std::endl;

  using SR = NaturalBag;  // Use NaturalBag for addition (uint64_t)
  using ValueType = typename SR::value_type;

  // Data setup
  // New Version (A): Keys [1, 3, 5], Values [10, 0, 30]
  std::vector<int> h_keys_new = {1, 3, 5};
  std::vector<ValueType> h_vals_new = {10, 0, 30};

  // Full Version (B): Keys [3, 5, 7], Values [0, 200, 300]
  std::vector<int> h_keys_full = {3, 5, 7};
  std::vector<ValueType> h_vals_full = {0, 200, 300};

  // Copy to device
  thrust::device_vector<int> d_keys_new = h_keys_new;
  thrust::device_vector<ValueType> d_vals_new = h_vals_new;
  thrust::device_vector<int> d_keys_full = h_keys_full;
  thrust::device_vector<ValueType> d_vals_full = h_vals_full;

  // Allocate output (max size = size of new + full)
  thrust::device_vector<int> d_keys_out(d_keys_new.size() + d_keys_full.size());
  thrust::device_vector<ValueType> d_vals_out(d_vals_new.size() + d_vals_full.size());

  // Use a non-const execution policy variable to match the function signature
  auto policy = thrust::device;

  // 1. Test Standard Diff Behavior (Matches suppressed UNLESS value changes)
  // Since we use NaturalBag addition, values WILL change (100+20 != 100),
  // so the new match_op logic (merged != val_full) returns true.
  // We expect ALL 3 keys to be emitted now.
  {
    MergeMatchOp<SR> match_op(false);

    // Initial check of full values
    thrust::host_vector<ValueType> h_vals_full_before = d_vals_full;
    if (h_vals_full_before[0] != 0 || h_vals_full_before[1] != 200 ||
        h_vals_full_before[2] != 300) {
      std::cerr << "FAIL: Pre-condition check failed for Full values." << std::endl;
      exit(1);
    }

    auto result_end = thrust::cuda_cub::set_difference_by_key_with_match_custom(
        policy, d_keys_new.begin(), d_keys_new.end(), d_keys_full.begin(), d_keys_full.end(),
        d_vals_new.begin(), d_vals_full.begin(), d_keys_out.begin(), d_vals_out.begin(),
        thrust::less<int>(), match_op);

    int out_size = result_end.first - d_keys_out.begin();
    std::cout << "Test 1 Output Size: " << out_size << std::endl;

    // We expect 2 because:
    // 1 (New) -> Diff -> Emit
    // 3 (Match) -> 0 (New) + 0 (Full) = 0. Merged(0) == Full(0). -> Suppressed.
    // 5 (Match) -> 30+200=230 != 200 -> Force Emit
    if (out_size != 2) {
      std::cerr << "FAIL: Expected output size 2, got " << out_size << std::endl;
      exit(1);
    }

    // Verify Output values
    thrust::host_vector<int> h_keys_res(d_keys_out.begin(), result_end.first);
    thrust::host_vector<ValueType> h_vals_res(d_vals_out.begin(), result_end.second);

    // Check specific values
    // Index 0: {1, 10}
    if (h_keys_res[0] != 1) {
      std::cerr << "FAIL: Expected key 1 at index 0." << std::endl;
      exit(1);
    }
    // Index 1: {5, 230} (Key 3 skipped)
    if (h_keys_res[1] != 5 || h_vals_res[1] != 230) {
      std::cerr << "FAIL: Expected output {5, 230} at index 1, got {" << h_keys_res[1] << ", "
                << h_vals_res[1] << "}" << std::endl;
      exit(1);
    }

    // Verify Side Effect: In-Place Update of Full Array
    thrust::host_vector<ValueType> h_vals_full_after = d_vals_full;
    std::cout << "Verifying In-Place Update of Full Array..." << std::endl;

    // Key 3 is at index 0 of Full (keys={3,5,7}) -> Should remain 0 (0+0=0)
    if (h_vals_full_after[0] != 0) {
      std::cerr << "FAIL: In-place update check for key 3. Expected 0, got " << h_vals_full_after[0]
                << std::endl;
      exit(1);
    }
    // Key 5 is at index 1 of Full -> Should be updated to 230
    if (h_vals_full_after[1] != 230) {
      std::cerr << "FAIL: In-place update failed for key 5. Expected 230, got "
                << h_vals_full_after[1] << std::endl;
      exit(1);
    }
    // Key 7 is at index 2 of Full -> Should remain 300
    if (h_vals_full_after[2] != 300) {
      std::cerr << "FAIL: In-place update failed for key 7. Expected 300, got "
                << h_vals_full_after[2] << std::endl;
      exit(1);
    }

    std::cout << "PASS: Diff logic + Suppression + In-Place Update correct." << std::endl;
  }

  // Reset d_vals_full for next test
  d_vals_full = h_vals_full;

  // 2. Test Merge Logic (Explicit Emit=True)
  // Should behave identically now since values change, but good to keep for regression.
  {
    MergeMatchOp<SR> match_op(true);
    // ... rest of test 2 logic ...
    auto result_end = thrust::cuda_cub::set_difference_by_key_with_match_custom(
        policy, d_keys_new.begin(), d_keys_new.end(), d_keys_full.begin(), d_keys_full.end(),
        d_vals_new.begin(), d_vals_full.begin(), d_keys_out.begin(), d_vals_out.begin(),
        thrust::less<int>(), match_op);

    int out_size = result_end.first - d_keys_out.begin();
    std::cout << "Test 2 Output Size: " << out_size << std::endl;
    if (out_size != 3) {
      std::cerr << "FAIL: Expected output size 3, got " << out_size << std::endl;
      exit(1);
    }
    std::cout << "PASS: Explicit emit logic correct." << std::endl;
  }
  // 3. Test Larger Dataset (e.g., > 1024 items to trigger multiple tiles/blocks)
  {
    std::cout << "Testing Large Dataset (1024 items)..." << std::endl;
    MergeMatchOp<SR> match_op(false);

    int size_new = 1024;
    int size_full = 1024;

    // Generate data
    // New: Even numbers [0, 2, ..., 2046]
    // Full: Multiples of 3 [0, 3, ..., 3069]
    thrust::host_vector<int> h_keys_new_large(size_new);
    thrust::host_vector<ValueType> h_vals_new_large(size_new);
    for (int i = 0; i < size_new; ++i) {
      h_keys_new_large[i] = i * 2;
      h_vals_new_large[i] = 10;
    }

    thrust::host_vector<int> h_keys_full_large(size_full);
    thrust::host_vector<ValueType> h_vals_full_large(size_full);
    for (int i = 0; i < size_full; ++i) {
      h_keys_full_large[i] = i * 3;
      h_vals_full_large[i] = 100;
    }

    thrust::device_vector<int> d_keys_new_large = h_keys_new_large;
    thrust::device_vector<ValueType> d_vals_new_large = h_vals_new_large;
    thrust::device_vector<int> d_keys_full_large = h_keys_full_large;
    thrust::device_vector<ValueType> d_vals_full_large = h_vals_full_large;

    thrust::device_vector<int> d_keys_out_large(size_new + size_full);
    thrust::device_vector<ValueType> d_vals_out_large(size_new + size_full);

    // Expected logic:
    // Matches occur at multiples of 6 (LCM of 2 and 3).
    // 0, 6, 12...
    // Max key approx 2046. 2046 / 6 = 341 matches.
    // For matches: New(10) + Full(100) = 110 != 100. -> Emit? No, match_op(false) means suppress
    // unless we force it? Wait, match_op logic: "if merged != val_full -> return (true, merged)"
    // 110 != 100 -> returns (true, 110). So matches ARE emitted.

    // Difference (New - Full): Keys in New not in Full.
    // i*2 is NOT div by 3.
    // Count: Total New (1024) - Matches (341 + 1 for 0? 0 is div by 6. range [0..2046]. 2046/6=341.
    // inclusive 0..341 = 342 matches?) Let's count exact. 0 is match. 6 is match. ... 341*6=2046 is
    // match. So 0..341 indices -> 342 matches. Diff count = 1024 - 342 = 682.

    // Total Output = Diff (682) + Matches(342) = 1024.
    // Wait, effectively all New keys are either Diff or Match-with-update.
    // All New keys should be emitted?
    // - If Key in New AND NOT in Full: Emit (Diff).
    // - If Key in New AND in Full: Match. Value updates 100->110. 110!=100 -> Emit (Match).
    // So YES, all 1024 New keys should be in output.

    auto result_end = thrust::cuda_cub::set_difference_by_key_with_match_custom(
        policy, d_keys_new_large.begin(), d_keys_new_large.end(), d_keys_full_large.begin(),
        d_keys_full_large.end(), d_vals_new_large.begin(), d_vals_full_large.begin(),
        d_keys_out_large.begin(), d_vals_out_large.begin(), thrust::less<int>(), match_op);

    int out_size = result_end.first - d_keys_out_large.begin();
    std::cout << "Test 3 Output Size: " << out_size << std::endl;

    // Verify Update Side Effect on a sample match
    // Key 6 should be updated in Full.
    // Key 6 is at index 2 in Full (0, 3, 6).
    thrust::host_vector<ValueType> h_vals_full_after = d_vals_full_large;
    if (h_vals_full_after[2] != 110) {
      std::cerr << "FAIL: Large test in-place update for key 6. Expected 110, got "
                << h_vals_full_after[2] << std::endl;
      exit(1);
    }

    if (out_size != 1024) {
      std::cerr << "FAIL: Expected output size 1024, got " << out_size << std::endl;
      exit(1);
    }
    std::cout << "PASS: Large dataset test passed." << std::endl;
  }
}  // End function

int main() {
  try {
    // Initialize GPU device (required for GPU operations)
    // We do this manually instead of calling init_cuda() to avoid RMM header issues
    GPU_ERROR_T err = GPU_SET_DEVICE(0);
    if (err != GPU_SUCCESS) {
      std::cerr << "Failed to set GPU device: " << GPU_GET_ERROR_STRING(err) << std::endl;
      return 1;
    }
    
    test_set_difference_with_merge();
    std::cout << "[OK] All custom set op tests passed." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << " (Possible device error)" << std::endl;
    return 1;
  }
  return 0;
}
