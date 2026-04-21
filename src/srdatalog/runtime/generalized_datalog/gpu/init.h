#pragma once

// Use GPU API abstraction instead of direct CUDA/HIP includes
#include "gpu/gpu_api.h"
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef __CUDA_ARCH__
#ifndef __HIP_DEVICE_COMPILE__
// Workaround for spdlog/fmt consteval issue with clang-cuda/clang-hip
#ifndef SPDLOG_USE_STD_FORMAT
#define SPDLOG_USE_STD_FORMAT
#endif
#include <rmm/mr/device/per_device_resource.hpp>
#endif
#endif

// Forward declare RMM types in global namespace (not inside SRDatalog::GPU)
// This prevents namespace resolution issues when RMM headers aren't included
#ifndef __CUDA_ARCH__
#ifndef __HIP_DEVICE_COMPILE__
namespace rmm { namespace mr { class device_memory_resource; } }
#endif
#endif

// Forward declare RMM initialization (defined in device_array_rmm_impl.h)
namespace SRDatalog::GPU {
void init_rmm_pool();
#ifndef __CUDA_ARCH__
#ifndef __HIP_DEVICE_COMPILE__
rmm::mr::device_memory_resource* get_gpu_pool_memory_resource();
#endif
#endif
}  // namespace SRDatalog::GPU

namespace SRDatalog::GPU {

/**
 * @brief Thread-safe GPU device initialization (works with both CUDA and HIP)
 * @details Initializes GPU device 0, sets up RMM/hipMM global pool, and prints device information.
 *          Uses std::once_flag to ensure initialization happens only once,
 *          even in multi-threaded environments.
 * @param device_id The GPU device ID to initialize (default: 0)
 * @throw std::runtime_error if no GPU devices are found or initialization fails
 * @note Also initializes the global RMM/hipMM pool memory resource, which is set as the
 *       default per-device resource for all RMM allocations
 */
inline void init_cuda(int device_id = 0) {
  // Keep name for backward compatibility, but works with both CUDA and HIP
  static std::once_flag init_flag;

  // Ensure device is set for the calling thread (GPU setDevice is thread-local)
  GPU_ERROR_T err = GPU_SET_DEVICE(device_id);
  if (err != GPU_SUCCESS) {
    throw std::runtime_error("Failed to set GPU device " + std::to_string(device_id) + ": " +
                             std::string(GPU_GET_ERROR_STRING(err)));
  }

  std::call_once(init_flag, [device_id]() {
    int device_count = 0;
    GPU_ERROR_T err = GPU_GET_DEVICE_COUNT(&device_count);
    if (err != GPU_SUCCESS || device_count == 0) {
      throw std::runtime_error("No GPU devices found or GPU error: " +
                               std::string(GPU_GET_ERROR_STRING(err)));
    }

    if (device_id < 0 || device_id >= device_count) {
      throw std::runtime_error("Invalid GPU device ID: " + std::to_string(device_id) +
                               " (available devices: " + std::to_string(device_count) + ")");
    }

    GPU_DEVICE_PROP_T prop{};
    err = GPU_GET_DEVICE_PROPERTIES(&prop, device_id);
    if (err == GPU_SUCCESS) {
#if SRDATALOG_GPU_PLATFORM_CUDA
      std::cout << ">> CUDA Device: " << prop.name << " (Compute " << prop.major << "."
                << prop.minor << ")" << std::endl;
#elif SRDATALOG_GPU_PLATFORM_ROCm
      std::cout << ">> HIP Device: " << prop.name << " (Architecture: " << prop.gcnArchName << ")" << std::endl;
#else
      std::cout << ">> GPU Device: " << prop.name << std::endl;
#endif
    }

    // Initialize RMM/hipMM global pool
    init_rmm_pool();

    // Set pool as the default per-device resource so all RMM allocations use it
    // This ensures device_uvector, rmm::device_vector, and other RMM containers
    // all use the same pool, and Thrust temporary allocations (via rmm::exec_policy) can use it too
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIP__)
    // Only set RMM resource when compiling host code (not device code)
    // Note: __HIP__ is defined when compiling with -x hip, but we still want to execute this
    // The key is that __HIP_DEVICE_COMPILE__ is only set inside device functions
    auto* pool = get_gpu_pool_memory_resource();
    rmm::mr::set_current_device_resource(pool);
#elif !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    // For HIP host code compilation (__HIP__ defined but not __HIP_DEVICE_COMPILE__)
    auto* pool = get_gpu_pool_memory_resource();
    rmm::mr::set_current_device_resource(pool);
#endif
  });
}

}  // namespace SRDatalog::GPU
