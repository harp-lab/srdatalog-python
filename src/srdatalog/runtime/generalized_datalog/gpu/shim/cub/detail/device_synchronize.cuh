#ifdef USE_ROCm
#ifndef CUB_DETAIL_DEVICE_SYNCHRONIZE_CUH
#define CUB_DETAIL_DEVICE_SYNCHRONIZE_CUH
// HIP/ROCm shim for cub/detail/device_synchronize.cuh
// Note: cudaError_t is defined by thrust/system/cuda headers
// This header is included before those, so we use auto return type
#include <hip/hip_runtime.h>
namespace cub {
namespace detail {
// Use auto to avoid needing cudaError_t type at declaration time
// The actual return type will be resolved when the function is used
inline auto DeviceSynchronize() -> decltype(hipDeviceSynchronize()) {
    return hipDeviceSynchronize();
}

// SyncStream function used by thrust/system/cuda/detail/util.h
inline auto SyncStream(hipStream_t stream) -> decltype(hipStreamSynchronize(stream)) {
    return hipStreamSynchronize(stream);
}
inline auto DebugSyncStream(hipStream_t stream) -> decltype(hipStreamSynchronize(stream)) {
    return hipStreamSynchronize(stream);
}
} // namespace detail
} // namespace cub

// SyncStream is used as cub::SyncStream (not in a namespace)
// The function is already defined above in cub::detail, but thrust expects it in cub::
// So we provide it directly in cub namespace
namespace cub {
inline auto SyncStream(hipStream_t stream) -> decltype(hipStreamSynchronize(stream)) {
    return hipStreamSynchronize(stream);
}
} // namespace cub
#endif // CUB_DETAIL_DEVICE_SYNCHRONIZE_CUH
#else
#include <cub/detail/device_synchronize.cuh>
#endif
