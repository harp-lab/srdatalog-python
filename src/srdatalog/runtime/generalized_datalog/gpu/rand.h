/**
 * @file rand.h
 * @brief Random number generation for GPU. Use simple static xorshift* algorithm. or some wrapper
 * for vendor friendly one
 */

#pragma once

#include <cstdint>

inline uint64_t gpu_rand(uint64_t index, uint64_t global_seed) {
  // Initialize state with index and global seed
  uint64_t x = index + global_seed;

  // Xorshift64* algorithm
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * 0x2545F4914F6CDD1DULL;
}

// u32 version - default implementation using xorshift
inline uint32_t gpu_rand(uint32_t index, uint32_t global_seed) {
  return gpu_rand(static_cast<uint64_t>(index), static_cast<uint64_t>(global_seed));
}

// // Vendor-provided RNG implementations
// #ifdef __CUDA_ARCH__
// // CUDA: Use curand for NVIDIA GPUs
// #include <curand_kernel.h>

// inline uint32_t gpu_vendor_rand(uint32_t index, uint32_t global_seed) {
//   curandState state;
//   curand_init(static_cast<unsigned long long>(global_seed), static_cast<unsigned long
//   long>(index),
//               0, &state);
//   return static_cast<uint32_t>(curand(&state));
// }
// #elif defined(__HIP_DEVICE_COMPILE__) || (defined(__HIP__) && !defined(__HIP_HOST_COMPILE__))
// // HIP: Use rocrand for AMD GPUs
// #include <rocrand/rocrand_kernel.h>

// inline uint32_t gpu_vendor_rand(uint32_t index, uint32_t global_seed) {
//   rocrand_state_xorwow state;
//   rocrand_init(static_cast<unsigned long long>(global_seed), static_cast<unsigned long
//   long>(index),
//                0, &state);
//   return static_cast<uint32_t>(rocrand(&state));
// }
// #endif
