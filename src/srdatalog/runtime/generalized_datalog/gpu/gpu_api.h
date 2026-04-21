/**
 * @file gpu_api.h
 * @brief CUDA/HIP API abstraction layer
 *
 * Provides macros and type aliases to abstract differences between CUDA and HIP APIs.
 * This allows the same code to work with both NVIDIA CUDA and AMD ROCm/HIP.
 *
 * Usage:
 *   - Include this header instead of directly including <cuda_runtime.h> or <hip/hip_runtime.h>
 *   - Use GPU_* macros instead of CUDA_* or HIP_* directly
 *   - Use GPU_* types instead of cuda* or hip* types
 */

#pragma once

#if defined(USE_CUDA) && defined(__CUDACC__)
// CUDA (NVIDIA) platform
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA API macros
#define GPU_SUCCESS cudaSuccess
#define GPU_ERROR_INVALID_VALUE cudaErrorInvalidValue
#define GPU_ERROR_OUT_OF_MEMORY cudaErrorMemoryAllocation
#define GPU_ERROR_NOT_READY cudaErrorNotReady

// CUDA device attribute macros
#define GPU_DEV_ATTR_MULTIPROCESSOR_COUNT cudaDevAttrMultiProcessorCount

// CUDA API functions
#define GPU_GET_ERROR_STRING cudaGetErrorString
#define GPU_GET_LAST_ERROR cudaGetLastError
#define GPU_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define GPU_SET_DEVICE cudaSetDevice
#define GPU_GET_DEVICE cudaGetDevice
#define GPU_GET_DEVICE_COUNT cudaGetDeviceCount
#define GPU_GET_DEVICE_PROPERTIES cudaGetDeviceProperties
#define GPU_DEVICE_GET_ATTRIBUTE cudaDeviceGetAttribute
#define GPU_MALLOC cudaMalloc
#define GPU_FREE cudaFree
#define GPU_MEMCPY cudaMemcpy
#define GPU_MEMCPY_ASYNC cudaMemcpyAsync
#define GPU_STREAM_CREATE cudaStreamCreate
#define GPU_STREAM_DESTROY cudaStreamDestroy
#define GPU_STREAM_SYNCHRONIZE cudaStreamSynchronize
#define GPU_EVENT_CREATE cudaEventCreate
#define GPU_EVENT_DESTROY cudaEventDestroy
#define GPU_EVENT_RECORD cudaEventRecord
#define GPU_EVENT_SYNCHRONIZE cudaEventSynchronize
#define GPU_EVENT_ELAPSED_TIME cudaEventElapsedTime
#define GPU_STREAM_WAIT_EVENT cudaStreamWaitEvent
#define GPU_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GPU_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define GPU_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice

// CUDA types
#define GPU_ERROR_T cudaError_t
#define GPU_STREAM_T cudaStream_t
#define GPU_EVENT_T cudaEvent_t
#define GPU_DEVICE_PROP_T cudaDeviceProp

// CUDA memory resource namespace
#define GPU_MR_NAMESPACE cuda::mr

// Thrust namespace (CUDA uses thrust directly)
#define GPU_THRUST_NAMESPACE thrust

#elif defined(USE_ROCm) && (defined(__HIP__) || defined(__HIP_DEVICE_COMPILE__))
// HIP (AMD/ROCm) platform - only when actually compiling with HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// HIP API macros (HIP uses same error codes as CUDA for compatibility)
#define GPU_SUCCESS hipSuccess
#define GPU_ERROR_INVALID_VALUE hipErrorInvalidValue
#define GPU_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define GPU_ERROR_NOT_READY hipErrorNotReady

// HIP device attribute macros
#define GPU_DEV_ATTR_MULTIPROCESSOR_COUNT hipDeviceAttributeMultiprocessorCount

// HIP API functions
#define GPU_GET_ERROR_STRING hipGetErrorString
#define GPU_GET_LAST_ERROR hipGetLastError
#define GPU_DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define GPU_SET_DEVICE hipSetDevice
#define GPU_GET_DEVICE hipGetDevice
#define GPU_GET_DEVICE_COUNT hipGetDeviceCount
#define GPU_GET_DEVICE_PROPERTIES hipGetDeviceProperties
#define GPU_DEVICE_GET_ATTRIBUTE hipDeviceGetAttribute
#define GPU_MALLOC hipMalloc
#define GPU_FREE hipFree
#define GPU_MEMCPY hipMemcpy
#define GPU_MEMCPY_ASYNC hipMemcpyAsync
#define GPU_STREAM_CREATE hipStreamCreate
#define GPU_STREAM_DESTROY hipStreamDestroy
#define GPU_STREAM_SYNCHRONIZE hipStreamSynchronize
#define GPU_EVENT_CREATE hipEventCreate
#define GPU_EVENT_DESTROY hipEventDestroy
#define GPU_EVENT_RECORD hipEventRecord
#define GPU_EVENT_SYNCHRONIZE hipEventSynchronize
#define GPU_EVENT_ELAPSED_TIME hipEventElapsedTime
#define GPU_STREAM_WAIT_EVENT hipStreamWaitEvent
#define GPU_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define GPU_HOST_TO_DEVICE hipMemcpyHostToDevice
#define GPU_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice

// HIP types
#define GPU_ERROR_T hipError_t
#define GPU_STREAM_T hipStream_t
#define GPU_EVENT_T hipEvent_t
#define GPU_DEVICE_PROP_T hipDeviceProp_t

// HIP device property access (different from CUDA)
#define GPU_GET_DEVICE_PROPERTIES hipGetDeviceProperties

// HIP memory resource namespace (hipMM maintains RMM API compatibility)
// Note: hipMM uses rmm namespace for compatibility, but internally uses HIP
#define GPU_MR_NAMESPACE cuda::mr  // hipMM maintains RMM API, so use cuda::mr namespace

// Thrust namespace (ROCm uses rocthrust)
#define GPU_THRUST_NAMESPACE roc::thrust

#else
// CPU-only compilation - provide stub definitions
#define GPU_SUCCESS 0
#define GPU_ERROR_INVALID_VALUE 1
#define GPU_ERROR_OUT_OF_MEMORY 2
#define GPU_ERROR_NOT_READY 3

// Device attribute macros (stub values)
#define GPU_DEV_ATTR_MULTIPROCESSOR_COUNT 0

// Stub types (won't be used in CPU-only code)
typedef int GPU_ERROR_T;
typedef void* GPU_STREAM_T;
typedef void* GPU_EVENT_T;
struct GPU_DEVICE_PROP_T {};

// Stub functions (won't be called in CPU-only code)
#define GPU_GET_ERROR_STRING(x) "CPU-only build"
#define GPU_GET_LAST_ERROR() GPU_SUCCESS
#define GPU_DEVICE_SYNCHRONIZE()
#define GPU_SET_DEVICE(x)
#define GPU_GET_DEVICE(x)
#define GPU_GET_DEVICE_COUNT(x)
#define GPU_DEVICE_GET_ATTRIBUTE(x, y, z) GPU_ERROR_INVALID_VALUE
#define GPU_MALLOC(x, y) GPU_ERROR_INVALID_VALUE
#define GPU_FREE(x)
#define GPU_MEMCPY(x, y, z, w) GPU_ERROR_INVALID_VALUE
#define GPU_MEMCPY_ASYNC(x, y, z, w, s) GPU_ERROR_INVALID_VALUE
#define GPU_STREAM_CREATE(x) GPU_ERROR_INVALID_VALUE
#define GPU_STREAM_DESTROY(x) ((void)0)
#define GPU_STREAM_SYNCHRONIZE(x) ((void)0)
#define GPU_EVENT_CREATE(x) GPU_ERROR_INVALID_VALUE
#define GPU_EVENT_DESTROY(x) ((void)0)
#define GPU_EVENT_RECORD(x, y) ((void)0)
#define GPU_EVENT_SYNCHRONIZE(x) ((void)0)
#define GPU_EVENT_ELAPSED_TIME(x, y, z) ((void)0)
#define GPU_STREAM_WAIT_EVENT(s, e, f) ((void)0)
#define GPU_DEVICE_TO_HOST 1
#define GPU_HOST_TO_DEVICE 2
#define GPU_DEVICE_TO_DEVICE 3

#define GPU_MR_NAMESPACE
#define GPU_THRUST_NAMESPACE

#endif

// Common abstraction macros
#if defined(USE_CUDA) || defined(USE_ROCm)
#define GPU_CHECK_ERROR(call)                                                                      \
  do {                                                                                             \
    GPU_ERROR_T err = (call);                                                                      \
    if (err != GPU_SUCCESS) {                                                                      \
      throw std::runtime_error("GPU error: " + std::string(GPU_GET_ERROR_STRING(err)) + " at " +   \
                               std::string(__FILE__) + ":" + std::to_string(__LINE__));            \
    }                                                                                              \
  } while (0)
#else
#define GPU_CHECK_ERROR(call) ((void)0)  // No-op for CPU-only
#endif
