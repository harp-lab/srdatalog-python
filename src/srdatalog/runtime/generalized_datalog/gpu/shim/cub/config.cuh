#ifdef USE_ROCm
// HIP/ROCm shim for cub/config.cuh
// Provide CUB version and namespace macros for thrust compatibility
#define CUB_VERSION 200800  // Compatible version number
#define CUB_MAJOR_VERSION 2
#define CUB_MINOR_VERSION 8
#define CUB_PATCH_VERSION 0
#define THRUST_IGNORE_CUB_VERSION_CHECK  // Ignore version check on ROCm

// CUB runtime function macro (used by thrust)
#ifndef CUB_RUNTIME_FUNCTION
#define CUB_RUNTIME_FUNCTION  // No-op on HIP
#endif
#else
#include <cub/config.cuh>
#endif
