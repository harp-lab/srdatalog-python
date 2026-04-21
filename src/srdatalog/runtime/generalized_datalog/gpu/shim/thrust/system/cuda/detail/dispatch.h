#ifdef USE_ROCm
// HIP/ROCm shim for thrust/system/cuda/detail/dispatch.h
// When using CUDA backend on ROCm, we need CUDA dispatch.h macros, not HIP ones
// Redirect to CUDA dispatch.h which has the correct macro definitions for CUDA backend
#pragma once

// Include the actual CUDA dispatch.h - it should work for CUDA backend even on HIP
#include <thrust/system/cuda/detail/dispatch.h>

#else
#include <thrust/system/cuda/detail/dispatch.h>
#endif
