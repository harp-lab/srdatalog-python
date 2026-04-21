#ifdef USE_ROCm
// HIP/ROCm doesn't need CUDA runtime detection
#define CUB_RUNTIME_ENABLED 1
#else
#include <cub/detail/detect_cuda_runtime.cuh>
#endif
