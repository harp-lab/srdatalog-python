#ifdef USE_ROCm
// HIP/ROCm shim for thrust::detail::int32_t and related integer types
// These are used by THRUST_DOUBLE_INDEX_TYPE_DISPATCH macro
#pragma once

#include <cstdint>

THRUST_NAMESPACE_BEGIN
namespace detail {
  // Provide int32_t and int64_t typedefs in thrust::detail namespace
  // These are used by integer_traits
  using int32_t = ::std::int32_t;
  using int64_t = ::std::int64_t;
} // namespace detail
THRUST_NAMESPACE_END

#else
// On CUDA, these should be provided by Thrust headers
#endif
