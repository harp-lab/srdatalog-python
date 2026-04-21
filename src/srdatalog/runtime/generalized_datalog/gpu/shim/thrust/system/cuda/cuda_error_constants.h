#ifdef USE_ROCm
// HIP/ROCm shim for CUDA error constants used by thrust/system/cuda/error.h
// HIP provides hipSuccess, hipError*, but thrust expects cudaSuccess, cudaError*
// Include this BEFORE thrust/system/cuda/error.h

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <utility>  // For std::pair, std::make_pair
#include <iterator>  // For std::iterator_traits
#include <concepts>  // For std::contiguous_iterator
#include <algorithm>  // For std::max, std::min
#include <functional>  // For std::plus
#include <bit>  // For std::popcount
#include <exception>  // For std::terminate

// libhipcxx (bundled in hipMM) provides many cuda::std types
// libhipcxx provides: conditional, ceil_div, void_t, is_same, pair, iterator_traits,
//                     contiguous_iterator, max, min, distance, plus, popcount, declval, etc.
// libhipcxx doesn't provide: terminate (function, not a type)
// 
// IMPORTANT: We should NOT redefine any types that libhipcxx provides to avoid ambiguity.
// Only provide what's truly missing (like terminate).
// Try to include libhipcxx headers if available to ensure types are accessible
#if __has_include(<cuda/std/utility>)
  #include <cuda/std/utility>  // libhipcxx provides cuda::std::pair here
#endif
#if __has_include(<cuda/std/iterator>)
  #include <cuda/std/iterator>  // libhipcxx provides cuda::std::iterator_traits, contiguous_iterator here
#endif
#if __has_include(<cuda/std/algorithm>)
  #include <cuda/std/algorithm>  // libhipcxx provides cuda::std::max, min, distance here
#endif
#if __has_include(<cuda/std/functional>)
  #include <cuda/std/functional>  // libhipcxx provides cuda::std::plus here
#endif
#if __has_include(<cuda/std/bit>)
  #include <cuda/std/bit>  // libhipcxx provides cuda::std::popcount here
#endif

namespace cuda {
namespace std {
  // libhipcxx provides most types we need: pair, iterator_traits, contiguous_iterator,
  // max, min, distance, plus, popcount, declval, random_access_iterator_tag, make_pair, etc.
  // We only provide what libhipcxx doesn't provide to avoid ambiguity.
  
  // Note: libhipcxx provides make_pair via <cuda/std/utility>, so we don't need to add it
  // If we add "using ::std::make_pair", it would conflict with libhipcxx's version
  using ::std::terminate;  // libhipcxx doesn't provide terminate
  // Note: Don't provide any types/functions that libhipcxx provides:
  // - pair, iterator_traits, contiguous_iterator, max, min, distance, plus, popcount,
  //   declval, random_access_iterator_tag, conditional, ceil_div, void_t, is_same, make_pair
} // namespace std
} // namespace cuda

// HIP doesn't provide cudaError_t and cudaStream_t as typedefs by default
// We need to define them for CUDA backend compatibility (following gpu_api.h pattern)
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipMemcpyKind cudaMemcpyKind;

// Thrust code sometimes uses 'cudaError' instead of 'cudaError_t' (typo in thrust)
typedef hipError_t cudaError;

// CUDA stream constants
#define cudaStreamLegacy ((cudaStream_t)0)  // HIP uses nullptr for default stream
#define cudaStreamPerThread ((cudaStream_t)1)  // HIP doesn't have per-thread streams, use a placeholder

// HIP provides cudaGetErrorString and cudaGetErrorName as aliases
// Check if they exist, if not define them
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetErrorName
#define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif
#ifndef cudaPeekAtLastError
#define cudaPeekAtLastError hipPeekAtLastError
#endif
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif

// CUDA device attributes (HIP equivalents)
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

// CUDA memory functions (HIP equivalents)
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamSynchronize hipStreamSynchronize

// Define CUDA error constants as aliases to HIP equivalents
#define cudaSuccess hipSuccess
#define cudaErrorMissingConfiguration hipErrorMissingConfiguration
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorInitializationError hipErrorInitializationError
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorLaunchTimeout hipErrorLaunchTimeOut
#define cudaErrorLaunchOutOfResources hipErrorLaunchOutOfResources
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorInvalidPitchValue hipErrorInvalidPitchValue
#define cudaErrorInvalidSymbol hipErrorInvalidSymbol
#define cudaErrorMapBufferObjectFailed hipErrorMapBufferObjectFailed
#define cudaErrorUnmapBufferObjectFailed hipErrorUnmapFailed  // HIP uses hipErrorUnmapFailed
#define cudaErrorInvalidTexture hipErrorInvalidTexture
#define cudaErrorInvalidTextureBinding hipErrorInvalidTexture  // HIP doesn't have separate binding error
#define cudaErrorInvalidChannelDescriptor hipErrorInvalidChannelDescriptor
#define cudaErrorInvalidMemcpyDirection hipErrorInvalidMemcpyDirection
#define cudaErrorInvalidFilterSetting hipErrorUnknown  // HIP doesn't have this error
#define cudaErrorInvalidNormSetting hipErrorUnknown  // HIP doesn't have this error
#define cudaErrorCudartUnloading hipErrorUnknown  // HIP doesn't have this error (HIP runtime doesn't unload)
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorSetOnActiveProcess hipErrorSetOnActiveProcess
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorECCUncorrectable hipErrorECCNotCorrectable
#define cudaErrorStartupFailure hipErrorUnknown  // HIP doesn't have this error
#define cudaErrorNotSupported hipErrorNotSupported
#endif
