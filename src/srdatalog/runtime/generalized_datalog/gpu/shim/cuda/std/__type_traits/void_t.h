#ifdef USE_ROCm
// HIP/ROCm shim for cuda/std/__type_traits/void_t.h
// On ROCm, use std::void_t from standard library
#include <type_traits>
namespace cuda {
namespace std {
  using ::std::void_t;
} // namespace std
} // namespace cuda
#else
#include <cuda/std/__type_traits/void_t.h>
#endif
