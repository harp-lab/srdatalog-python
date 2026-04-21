#pragma once

// HIP shim for cuda/stream_ref compatibility
// Provides CUDA API compatibility for hipMM when using ROCm
// hipMM expects <cuda/stream_ref> but ROCm doesn't provide it natively

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace cuda {

class stream_ref {
    hipStream_t stream_;
public:
    stream_ref() noexcept : stream_(nullptr) {}  // Default stream
    stream_ref(hipStream_t s) noexcept : stream_(s) {}
    
    hipStream_t get() const noexcept { return stream_; }
    operator hipStream_t() const noexcept { return stream_; }
    
    void synchronize() const {
        hipStreamSynchronize(stream_);
    }
    
    bool ready() const {
        return hipStreamQuery(stream_) == hipSuccess;
    }
};

}  // namespace cuda
