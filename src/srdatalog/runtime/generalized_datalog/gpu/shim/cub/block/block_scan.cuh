#ifdef USE_ROCm
// HIP/ROCm shim for cub/block/block_scan.cuh
// Redirect to hipcub which provides CUB compatibility
#include <hipcub/block/block_scan.hpp>
#else
#include <cub/block/block_scan.cuh>
#endif
