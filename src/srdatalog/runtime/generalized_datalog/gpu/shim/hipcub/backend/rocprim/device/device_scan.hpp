// Shim header to fix hipcub/rocprim compatibility issue in ROCm 7.1.1
// hipcub calls rocprim::inclusive_scan with 4 template parameters, but rocprim 7.1.1 expects 5
// This shim provides a compatibility wrapper

#pragma once

#ifdef USE_ROCm

// Include rocprim to get the correct API
#include <rocprim/device/device_scan.hpp>
#include <rocprim/type_traits.hpp>

// Create a compatibility namespace that fixes the API mismatch
namespace rocprim_compat {
    // Wrapper that matches hipcub's expected signature (4 template params)
    // but calls rocprim with the correct signature (5 template params)
    template<class Config,
             class InputIterator,
             class OutputIterator,
             class BinaryFunction>
    inline hipError_t inclusive_scan(void*             temporary_storage,
                                    size_t&           storage_size,
                                    InputIterator     input,
                                    OutputIterator    output,
                                    const size_t      size,
                                    BinaryFunction    scan_op           = BinaryFunction(),
                                    const hipStream_t stream            = 0,
                                    bool              debug_synchronous = false)
    {
        // Use rocprim's default AccType deduction
        using AccType = ::rocprim::accumulator_t<BinaryFunction,
                                                 typename std::iterator_traits<InputIterator>::value_type>;
        
        return ::rocprim::inclusive_scan<Config, InputIterator, OutputIterator, BinaryFunction, AccType>(
            temporary_storage,
            storage_size,
            input,
            output,
            size,
            scan_op,
            stream,
            debug_synchronous);
    }
}

// Temporarily redirect rocprim::inclusive_scan to our compatibility wrapper
// This is a workaround for the hipcub/rocprim version mismatch
#define rocprim_inclusive_scan_4param rocprim_compat::inclusive_scan

// Now include the original hipcub header
// We'll need to patch the calls, but since we can't easily do that,
// we'll use a different strategy: create a macro that fixes the call

#else
// For non-ROCm builds, include normally
#include <hipcub/backend/rocprim/device/device_scan.hpp>
#endif
