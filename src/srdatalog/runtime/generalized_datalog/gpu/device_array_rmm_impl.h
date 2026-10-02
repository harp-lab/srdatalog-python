/**
 * @file device_array_rmm_impl.h
 * @brief RMM implementation details for DeviceArray (host-only)
 *
 * This file contains RMM-specific implementation that should only be compiled
 * for host code to avoid spdlog consteval issues in device code.
 */

#pragma once

#ifndef __CUDA_ARCH__
#ifndef __HIP_DEVICE_COMPILE__

// Note: Requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined
// (defined in xmake.lua when nvidia or rocm config is enabled)

// Workaround for spdlog/fmt consteval issue with clang-cuda/clang-hip
// Use std::format instead of fmt to avoid consteval issues
#ifndef SPDLOG_USE_STD_FORMAT
#define SPDLOG_USE_STD_FORMAT
#endif

// Use GPU API abstraction instead of direct CUDA includes
#include "gpu/gpu_api.h"

// This file is host-only, so RMM/hipMM headers (which pull in spdlog) are safe here
// We need the full RMM headers here for pool_memory_resource and cuda_memory_resource types
// Note: hipMM maintains RMM API compatibility, so these headers work for both CUDA and HIP
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <rmm/cuda_device.hpp>  // For rmm::available_device_memory() - works with hipMM too
#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>  // hipMM maintains this API
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#if SRDATALOG_GPU_PLATFORM_CUDA
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#endif
#include <stdexcept>
#include <string>

namespace SRDatalog::GPU {

/**
 * @brief Configuration for RMM pool memory resource
 * @details Pool sizes can be configured via environment variables:
 *   - SRDATALOG_RMM_RESOURCE: "pool" (default) or CUDA-only "cuda_async"
 *     Both use device memory; cuda_async is not a managed-memory fallback.
 *   - SRDATALOG_RMM_POOL_INITIAL_SIZE: Initial pool size in bytes (default: 1GB)
 *   - SRDATALOG_RMM_POOL_MAX_SIZE: Maximum pool size in bytes (default: unlimited)
 *     Set to 0 or use environment variable to specify a limit, otherwise unlimited
 * @note If environment variables are not set, uses default values (unlimited max size)
 */
namespace RMMConfig {
// Default pool sizes (can be overridden via environment variables)
constexpr std::size_t DEFAULT_INITIAL_SIZE = 1024ULL * 1024 * 1024;  // 1GB
// Default max size is unlimited (use a very large value that's a multiple of 256)
// Using UINT64_MAX rounded down to multiple of 256: 0xFFFFFFFFFFFFFE00
// RMM requires max_size to be a multiple of 256 bytes
constexpr std::size_t DEFAULT_MAX_SIZE = 0xFFFFFFFFFFFFFE00ULL;  // ~18 exabytes, multiple of 256

inline std::size_t get_initial_size() {
  const char* env = std::getenv("SRDATALOG_RMM_POOL_INITIAL_SIZE");
  if (env != nullptr) {
    return std::strtoull(env, nullptr, 0);
  }
  return DEFAULT_INITIAL_SIZE;
}

inline std::size_t get_max_size() {
  const char* env = std::getenv("SRDATALOG_RMM_POOL_MAX_SIZE");
  if (env != nullptr) {
    std::size_t value = std::strtoull(env, nullptr, 0);
    // Allow 0 to mean unlimited (use default unlimited value)
    if (value == 0) {
      return DEFAULT_MAX_SIZE;
    }
    // Round to nearest multiple of 256 (round down) - RMM requirement
    return (value / 256) * 256;
  }
  return DEFAULT_MAX_SIZE;  // Unlimited by default
}

inline std::optional<std::size_t> get_max_size_optional() {
  const char* env = std::getenv("SRDATALOG_RMM_POOL_MAX_SIZE");
  if (env != nullptr) {
    std::size_t value = std::strtoull(env, nullptr, 0);
    // Allow 0 or empty to mean unlimited (return std::nullopt)
    if (value == 0) {
      return std::nullopt;
    }
    // Round to nearest multiple of 256 (round down) - RMM requirement
    return std::make_optional((value / 256) * 256);
  }
  return std::nullopt;  // Unlimited by default (no max size limit)
}
}  // namespace RMMConfig

/**
 * @brief Thread-safe singleton that provides a global GPU pool memory resource
 * @note GPU (CUDA or HIP) must be initialized before this is called (call init_cuda() first)
 * @note This function is host-only (allocation happens on host)
 * @note The pool is also set as the global per-device resource so all RMM/hipMM allocations use it
 */
inline rmm::mr::device_memory_resource* get_gpu_pool_memory_resource() {
  struct ResourceOwner {
    // Destroy the pool before its non-owning upstream resource.
    std::unique_ptr<rmm::mr::cuda_memory_resource> upstream;
    std::unique_ptr<rmm::mr::device_memory_resource> resource;
  };
  static ResourceOwner owner = []() {
    int current_device = -1;
    GPU_ERROR_T err = GPU_GET_DEVICE(&current_device);
    if (err != GPU_SUCCESS) {
      throw std::runtime_error(
          "get_gpu_pool_memory_resource: GPU not initialized. Call init_cuda() first. Error: " +
          std::string(GPU_GET_ERROR_STRING(err)));
    }
    ResourceOwner result;
    const char* kind = std::getenv("SRDATALOG_RMM_RESOURCE");
    const auto initial_size = RMMConfig::get_initial_size();
    const auto max_size = RMMConfig::get_max_size_optional();
    if (kind != nullptr && std::strcmp(kind, "cuda_async") == 0) {
#if SRDATALOG_GPU_PLATFORM_CUDA
      if (max_size.has_value()) {
        throw std::invalid_argument(
            "SRDATALOG_RMM_POOL_MAX_SIZE is not supported by cuda_async; use pool for a hard cap");
      }
      result.resource = std::make_unique<rmm::mr::cuda_async_memory_resource>(initial_size);
#else
      throw std::invalid_argument("SRDATALOG_RMM_RESOURCE=cuda_async requires CUDA");
#endif
    } else if (kind == nullptr || kind[0] == '\0' || std::strcmp(kind, "pool") == 0) {
      result.upstream = std::make_unique<rmm::mr::cuda_memory_resource>();
      result.resource =
          std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
              result.upstream.get(), initial_size, max_size);
    } else {
      throw std::invalid_argument("SRDATALOG_RMM_RESOURCE must be pool or cuda_async");
    }
    rmm::mr::set_current_device_resource(result.resource.get());
    return result;
  }();
  return owner.resource.get();
}

/**
 * @brief Initialize and set up the global RMM pool memory resource
 * @note This should be called early in the program (typically in init_cuda())
 * @note This function is idempotent - safe to call multiple times
 */
inline void init_rmm_pool() {
  // Simply access the pool to trigger its initialization
  (void)get_gpu_pool_memory_resource();
}


/**
 * @brief Print RMM pool memory usage report
 * @note This function is host-only and intended for debugging/monitoring
 */
inline void print_rmm_memory_report() {
  auto* resource = get_gpu_pool_memory_resource();
  auto* pool =
      dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(resource);
  if (pool != nullptr) {
    std::cout << "\n=== RMM Pool Memory Report ===" << std::endl;

    // Get GPU memory information
    auto const [free, total] = rmm::available_device_memory();
    std::cout << "GPU free memory: " << free << " bytes (" << (free / (1024.0 * 1024.0)) << " MB)"
              << std::endl;
    std::cout << "GPU total memory: " << total << " bytes (" << (total / (1024.0 * 1024.0))
              << " MB)" << std::endl;

    // Get pool size
    std::size_t pool_size = pool->pool_size();
    std::cout << "Pool size: " << pool_size << " bytes (" << (pool_size / (1024.0 * 1024.0))
              << " MB)" << std::endl;

// Try to call print() if RMM_DEBUG_PRINT is defined
#ifdef RMM_DEBUG_PRINT
    pool->print();
#else
    std::cout << "Note: Enable RMM_DEBUG_PRINT for detailed block information" << std::endl;
#endif

    std::cout << "==============================\n" << std::endl;
  }
#if SRDATALOG_GPU_PLATFORM_CUDA
  else if (auto* async = dynamic_cast<rmm::mr::cuda_async_memory_resource*>(resource)) {
    std::uint64_t used = 0, reserved = 0;
    const auto used_status =
        cudaMemPoolGetAttribute(async->pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used);
    const auto reserved_status =
        cudaMemPoolGetAttribute(async->pool_handle(), cudaMemPoolAttrReservedMemCurrent, &reserved);
    if (used_status == cudaSuccess && reserved_status == cudaSuccess) {
      std::cout << "CUDA async device pool: used=" << used << " reserved=" << reserved << " bytes\n";
    } else {
      std::cout << "CUDA async device pool statistics unavailable\n";
    }
  }
#endif
}

}  // namespace SRDatalog::GPU

#endif  // __HIP_DEVICE_COMPILE__
#endif  // __CUDA_ARCH__
