#ifdef USE_ROCm
// HIP/ROCm shim for cub/util_device.cuh
// Provide device utility functions for ROCm
#include <hip/hip_runtime.h>
namespace cub {
// PtxVersion - HIP doesn't have PTX, return success
inline hipError_t PtxVersion(int& version) {
  version = 0;  // HIP doesn't use PTX version
  return hipSuccess;
}

// CurrentDevice - get current HIP device
inline int CurrentDevice() {
  int device;
  hipGetDevice(&device);
  return device;
}

// MaxSmOccupancy - get max occupancy (stub for HIP)
template<typename Kernel>
inline hipError_t MaxSmOccupancy(int& occupancy, Kernel k, int block_threads) {
  // HIP doesn't have exact CUDA occupancy API, return success with placeholder
  occupancy = 1;  // Placeholder
  return hipSuccess;
}
} // namespace cub
#else
#include <cub/util_device.cuh>
#endif
