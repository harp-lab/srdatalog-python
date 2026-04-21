#ifdef USE_ROCm
// HIP/ROCm shim for cub/util_temporary_storage.cuh
// hipcub provides this header
#include <hipcub/util_temporary_storage.hpp>
// Provide cub::detail::AliasTemporaries alias for hipcub::AliasTemporaries
namespace cub {
namespace detail {
  using hipcub::AliasTemporaries;
} // namespace detail
} // namespace cub
#else
#include <cub/util_temporary_storage.cuh>
#endif
