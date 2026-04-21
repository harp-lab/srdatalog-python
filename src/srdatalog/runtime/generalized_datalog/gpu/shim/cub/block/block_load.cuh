#ifdef USE_ROCm
// HIP/ROCm shim for cub/block/block_load.cuh
// Redirect to hipcub which provides CUB compatibility
// Note: hipcub types are in hipcub:: namespace, but code expects cub::
// This shim includes hipcub headers - the namespace compatibility
// should be handled by hipcub itself or we need namespace aliases
#include <hipcub/block/block_load.hpp>
#else
#include <cub/block/block_load.cuh>
#endif
