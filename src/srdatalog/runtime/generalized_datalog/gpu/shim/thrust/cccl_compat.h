#ifdef USE_ROCm
// HIP/ROCm shim for CCCL macros used by thrust
// Include this BEFORE any thrust headers that use _CCCL_COMPILER or _CCCL_SUPPRESS_DEPRECATED

// Define THRUST_IGNORE_CUB_VERSION_CHECK to avoid CUB version compatibility errors on ROCm
#ifndef THRUST_IGNORE_CUB_VERSION_CHECK
#define THRUST_IGNORE_CUB_VERSION_CHECK
#endif

// Enable CUDA backend compatibility on HIP
// This allows CUDA backend headers (like agent_launcher.h) to be compiled on HIP
#ifndef _CCCL_HAS_CUDA_COMPILER
#define _CCCL_HAS_CUDA_COMPILER 1
#endif

// On HIP, we need the _kernel_agent template to be available for host code
// The agent_launcher.h checks for __CUDA_ARCH__ || _NVHPC_CUDA to define _kernel_agent template
// On HIP, we need to enable this so the kernel template is instantiated
// However, we need to be careful - _NVHPC_CUDA might cause issues with some code
// Let's try defining it conditionally only when needed
#ifdef __HIP__
// Define _NVHPC_CUDA to enable kernel template instantiation on HIP
// This allows _kernel_agent template to be defined, which is needed for kernel launches
#ifndef _NVHPC_CUDA
#define _NVHPC_CUDA 1
#endif
#endif

#ifndef _CCCL_COMPILER
// _CCCL_COMPILER is a function-like macro for compiler detection
// Format: _CCCL_COMPILER(COMPILER_NAME) returns 1 if using that compiler, 0 otherwise
// On HIP, we're not using NVRTC, NVCC, etc., so always return 0
#define _CCCL_COMPILER(...) 0
#endif

// _CCCL_CUDA_COMPILER macro (function-like, similar to _CCCL_COMPILER)
#ifndef _CCCL_CUDA_COMPILER
#define _CCCL_CUDA_COMPILER(...) 0  // Not using CUDA compiler on HIP
#endif

// CCCL diagnostic suppression macros
#ifndef _CCCL_DIAG_SUPPRESS_GCC
#define _CCCL_DIAG_SUPPRESS_GCC(...)  // No-op on HIP
#endif
#ifndef _CCCL_DIAG_SUPPRESS_CLANG
#define _CCCL_DIAG_SUPPRESS_CLANG(...)  // No-op on HIP
#endif
#ifndef _CCCL_DIAG_SUPPRESS_NVHPC
#define _CCCL_DIAG_SUPPRESS_NVHPC(...)  // No-op on HIP
#endif

// CCCL kernel attributes
#ifndef CCCL_DETAIL_KERNEL_ATTRIBUTES
#define CCCL_DETAIL_KERNEL_ATTRIBUTES  // No-op on HIP (kernel attributes)
#endif

// THRUST_DETAIL_KERNEL_ATTRIBUTES - used by _kernel_agent template
// On HIP, this needs to include __global__ to make the function a kernel
#ifndef THRUST_DETAIL_KERNEL_ATTRIBUTES
#define THRUST_DETAIL_KERNEL_ATTRIBUTES __global__  // Add __global__ for HIP kernel functions
#endif

// CCCL execution space macros - map to HIP equivalents
#ifndef _CCCL_HOST
#define _CCCL_HOST __host__
#endif
#ifndef _CCCL_DEVICE
#define _CCCL_DEVICE __device__
#endif
#ifndef _CCCL_HOST_DEVICE
#define _CCCL_HOST_DEVICE __host__ __device__
#endif
#ifndef _CCCL_FORCEINLINE
#define _CCCL_FORCEINLINE __forceinline__
#endif

#ifndef _CCCL_SUPPRESS_DEPRECATED_PUSH
// Use THRUST versions as fallback (these are defined in thrust/detail/config/diagnostic.h)
// But we need to define them before diagnostic.h is included, so use empty macros for now
// and let diagnostic.h override them
#define _CCCL_SUPPRESS_DEPRECATED_PUSH
#define _CCCL_SUPPRESS_DEPRECATED_POP
#endif

// Additional CCCL macros used by thrust
#ifndef _CCCL_EXEC_CHECK_DISABLE
#define _CCCL_EXEC_CHECK_DISABLE  // No-op on HIP
#endif

// CCCL deprecated macro
#ifndef CCCL_DEPRECATED_BECAUSE
#define CCCL_DEPRECATED_BECAUSE(...)  // No-op on HIP
#endif
#ifndef CCCL_DEPRECATED
#define CCCL_DEPRECATED  // No-op on HIP
#endif

// CCCL visibility macros
#ifndef _CCCL_VISIBILITY_HIDDEN
#define _CCCL_VISIBILITY_HIDDEN  // No-op on HIP (visibility attribute)
#endif

// NV_IF_TARGET and related macros are defined in thrust/system/hip/detail/nv/detail/__target_macros.h
// We include that header in custom_set_op.h before CUDA backend headers
// So we don't need to define them here - they'll be available from the HIP headers
#endif
