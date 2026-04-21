#ifdef USE_ROCm
// HIP/ROCm shim for cub/util_debug.cuh
// CUB debug utilities - minimal implementation for ROCm
#include <hip/hip_runtime.h>
namespace cub {
// Debug function - in HIP, cudaError_t is hipError_t
template<typename ErrorType>
inline bool Debug(ErrorType error, const char* file, int line) {
    if (error != hipSuccess) {
        // Could add logging here if needed
        return true;
    }
    return false;
}
} // namespace cub

// CubDebug is a macro/alias for cub::Debug used by thrust
// It's typically defined in thrust/system/cuda/detail/core/util.h
// For HIP compatibility, define it as calling cub::Debug
#define CubDebug(expr) cub::Debug((expr), __FILE__, __LINE__)
#else
#include <cub/util_debug.cuh>
#endif
