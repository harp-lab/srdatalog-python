#ifdef USE_ROCm
// HIP/ROCm shim for cuda/std/__cccl/cuda_capabilities.h
// CUDA capabilities are CUDA-specific - on HIP/ROCm, we can stub this out
// Most CUDA capability queries can be mapped to HIP equivalents if needed
#pragma once

// Minimal stub - provides empty definitions
// If specific capability queries are needed, they can be mapped to HIP device attributes
#else
#include <cuda/std/__cccl/cuda_capabilities.h>
#endif
