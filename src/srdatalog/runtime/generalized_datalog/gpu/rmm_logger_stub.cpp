/**
 * @file rmm_logger_stub.cpp
 * @brief Stub implementation for RMM logger symbols when using header-only hipMM
 * 
 * hipMM is header-only and doesn't provide the RMM logger implementation library.
 * This file provides stub implementations of the missing logger symbols to allow
 * linking to succeed. Logging is effectively disabled (all functions are no-ops).
 * 
 * NOTE: This file must be compiled as HOST code only (not HIP device code).
 */

#ifdef USE_ROCm
#ifndef __HIP_DEVICE_COMPILE__
#ifndef __CUDA_ARCH__
// Only compile this stub for ROCm builds where hipMM is used
// For CUDA builds, RMM provides the actual logger implementation

// Include RMM headers AFTER defining the implementation details
// We need to provide complete definitions for incomplete types

namespace rmm {
namespace detail {

// Provide complete definition of logger_impl (normally provided by RMM library)
// Use class to match RMM header declaration
class logger_impl {
public:
    // Minimal stub - just enough to satisfy the linker
    // The actual implementation would use spdlog, but we don't need that
};

// Provide complete definition of sink_impl
// Use class to match RMM header declaration
class sink_impl {
public:
    // Minimal stub
    ~sink_impl() = default;
};

}  // namespace detail
}  // namespace rmm

// Now include the RMM headers which will use our complete definitions
#include <rmm/logger.hpp>
#include <memory>
#include <vector>
#include <string>

namespace rmm {

// Stub implementation of logger class
logger::logger(std::string name, std::vector<sink_ptr> sinks) : sinks_(*this, sinks) {
    // No-op constructor - initialize sinks_ with parent reference
    (void)name;  // Suppress unused parameter warning
}

logger::~logger() {
    // No-op destructor
}

void logger::set_level(level_enum) {
    // No-op: logging is disabled
}

void logger::log(level_enum, std::string const&) {
    // No-op: logging is disabled
}

// Stub implementation of sink base class
sink::sink(std::unique_ptr<detail::sink_impl> impl) : impl(std::move(impl)) {
    // No-op constructor - store the impl pointer
}

sink::~sink() {
    // No-op destructor
}

// Stub implementation of stderr_sink_mt
stderr_sink_mt::stderr_sink_mt() : sink(std::make_unique<detail::sink_impl>()) {
    // No-op constructor - initialize base class with stub sink_impl
}

// Stub implementation of basic_file_sink_mt
basic_file_sink_mt::basic_file_sink_mt(std::string const&, bool) : sink(std::make_unique<detail::sink_impl>()) {
    // No-op constructor - initialize base class with stub sink_impl
}

}  // namespace rmm

#endif  // __CUDA_ARCH__
#endif  // __HIP_DEVICE_COMPILE__
#endif  // USE_ROCm
