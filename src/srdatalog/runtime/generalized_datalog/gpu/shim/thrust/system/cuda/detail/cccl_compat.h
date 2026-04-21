#ifdef USE_ROCm
// HIP/ROCm shim for CCCL macros used by thrust
// These macros should be defined by libcudacxx but aren't available on ROCm

// _CCCL_COMPILER is a function-like macro for compiler detection
// Format: _CCCL_COMPILER(COMPILER_NAME) returns 1 if using that compiler, 0 otherwise
#define _CCCL_COMPILER(...) 0  // On HIP, we're not using NVRTC, NVCC, etc.

// _CCCL_SUPPRESS_DEPRECATED macros - use THRUST versions as fallback
#include <thrust/detail/config/diagnostic.h>
#define _CCCL_SUPPRESS_DEPRECATED_PUSH THRUST_SUPPRESS_DEPRECATED_PUSH
#define _CCCL_SUPPRESS_DEPRECATED_POP THRUST_SUPPRESS_DEPRECATED_POP
#else
// On CUDA, these should be defined by libcudacxx
#endif
