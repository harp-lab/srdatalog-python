#ifdef USE_ROCm
// HIP/ROCm shim for cub/block/block_store.cuh
// Redirect to hipcub which provides CUB compatibility
#include <hipcub/block/block_store.hpp>
#else
#include <cub/block/block_store.cuh>
#endif
