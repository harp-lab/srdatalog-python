#pragma once

// GPU platform detection and abstraction
// Supports both CUDA (NVIDIA) and HIP (AMD/ROCm)

#if defined(USE_CUDA) && defined(__CUDACC__)
// CUDA compilation (NVIDIA)
#include <cuda/std/span>
namespace cuda_std = cuda::std;
#define GPU_HD __host__ __device__
#define GPU_DEVICE __device__
#define GPU_FORCE_INLINE __forceinline__
#define SRDATALOG_GPU_AVAILABLE 1
#define SRDATALOG_GPU_PLATFORM_CUDA 1
#define SRDATALOG_GPU_PLATFORM_ROCm 0
#define GPU_WARP_SIZE 32           // CUDA warp size
#define GPU_DEFAULT_GROUP_SIZE 32  // Default cooperative group size for CUDA
#elif defined(USE_ROCm) && (defined(__HIP__) || defined(__HIP_DEVICE_COMPILE__))
// HIP compilation (AMD/ROCm) - only when actually compiling with HIP
#include <hip/hip_runtime.h>
#include <span>
namespace cuda_std = std;  // HIP uses std::span, not cuda::std::span
#define GPU_HD __host__ __device__
#define GPU_DEVICE __device__
#define GPU_FORCE_INLINE __forceinline__
#define SRDATALOG_GPU_AVAILABLE 1
#define SRDATALOG_GPU_PLATFORM_CUDA 0
#define SRDATALOG_GPU_PLATFORM_ROCm 1

// AMD GPU wavefront size
// CDNA (MI100, MI200, MI300): 64-wide wavefronts
// RDNA (RX 6000, RX 7000): 32-wide wavefronts (wave32 default)
// For maximum compatibility and portability, we use 32 as the default group size
// This works on all AMD GPUs (CDNA uses wave64 but supports 32-thread subgroups)
// and matches CUDA's warp size for consistent behavior
#define GPU_WARP_SIZE 32
#define GPU_DEFAULT_GROUP_SIZE 32
#else
// CPU-only compilation
#include <span>
namespace cuda_std = std;
#define GPU_HD
#define GPU_DEVICE
#define GPU_FORCE_INLINE inline
#define SRDATALOG_GPU_AVAILABLE 0
#define SRDATALOG_GPU_PLATFORM_CUDA 0
#define SRDATALOG_GPU_PLATFORM_ROCm 0
#endif

#include <limits>

template <typename ValueType>
constexpr auto SENTINEL = std::numeric_limits<ValueType>::max();

// Compiler-specific attribute macros
#ifdef __clang__
// Clang-specific attributes for lifetime checking
// See: https://clang.llvm.org/docs/AttributeReference.html#lifetimebound
//      https://clang.llvm.org/docs/AttributeReference.html#lifetime-capture-by
#define CLANG_LIFETIMEBOUND [[clang::lifetimebound]]
// Note: lifetime_capture_by is available but not currently used in this codebase.
// It's useful for functions that take a view parameter and store it elsewhere, e.g.:
//   void store_view(NDArrayView v [[clang::lifetime_capture_by(container)]],
//                   std::vector<NDArrayView>& container) { container.push_back(v); }
// Use: #define CLANG_LIFETIME_CAPTURE_BY(X) [[clang::lifetime_capture_by(X)]]
#else
// GCC, NVCC, and other compilers don't support lifetime attributes
#define CLANG_LIFETIMEBOUND
#endif

// constant for decide if join is materialized or compute size only
constexpr bool JOIN_MATERIALIZED = true;
constexpr bool JOIN_SIZE_ONLY = false;
