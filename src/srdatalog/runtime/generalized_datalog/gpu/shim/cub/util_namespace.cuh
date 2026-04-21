#ifdef USE_ROCm
// HIP/ROCm shim for cub/util_namespace.cuh
// Provide CUB namespace macros for thrust compatibility
// Modern macros (when THRUST_IGNORE_CUB_VERSION_CHECK is not set)
#define CUB_NAMESPACE_BEGIN namespace cub {
#define CUB_NAMESPACE_END }
// Old-style macros (when THRUST_IGNORE_CUB_VERSION_CHECK is set)
#define CUB_NS_PREFIX
#define CUB_NS_POSTFIX
#define CUB_NS_QUALIFIER ::cub

// Include hipcub agent header to get ScanTileState and TilePrefixCallbackOp
#include <hipcub/agent/single_pass_scan_operators.hpp>

// Provide cub:: namespace that aliases hipcub types
// This allows code using cub::BlockLoad, cub::CacheModifiedInputIterator, etc. to work
namespace cub {
  // Bring hipcub types into cub:: namespace
  using namespace hipcub;
  
  // ScanTileState and TilePrefixCallbackOp are in hipcub namespace (BEGIN_HIPCUB_NAMESPACE = namespace hipcub {)
  // Create template aliases to bring them into cub:: namespace for compatibility
  template<typename T, bool SINGLE_WORD = (sizeof(T) <= 7)>
  using ScanTileState = hipcub::ScanTileState<T, SINGLE_WORD>;
  
  template<typename T, typename ScanOpT, typename ScanTileStateT, int LEGACY_PTX_ARCH = 0, typename DelayConstructorT = void>
  using TilePrefixCallbackOp = hipcub::TilePrefixCallbackOp<T, ScanOpT, ScanTileStateT, LEGACY_PTX_ARCH, DelayConstructorT>;
} // namespace cub
#else
#include <cub/util_namespace.cuh>
#endif
