#pragma once

#include "../semiring.h"
#include "macro.h"
#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
// CUB is not actually used in this file, so we don't include it
// (cub/cub.cuh was included but never referenced)
// #include <cub/cub.cuh>

namespace SRDatalog::GPU {

template <Semiring SR, typename Group>
__device__ GPU_FORCE_INLINE semiring_value_t<SR> warp_reduce_sr(semiring_value_t<SR> val,
                                                                Group tile) {
#pragma unroll
  for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
    auto other = tile.shfl_down(val, offset);
    val = sr_add<SR>(val, other);
  }
  return tile.shfl(val, 0);
}

template <Semiring SR, int TILE_SIZE>
__device__ GPU_FORCE_INLINE semiring_value_t<SR> tile_semiring_reduce(
    const semiring_value_t<SR>* prov_data, uint32_t begin, uint32_t end) {
  auto tile =
      cooperative_groups::tiled_partition<TILE_SIZE>(cooperative_groups::this_thread_block());
  auto n = end - begin;
  auto lane = tile.thread_rank();

  semiring_value_t<SR> local_sum = sr_zero<SR>();
  // stride load, optimize with batch loading in future
  for (uint32_t i = lane; i < n; i += TILE_SIZE) {
    local_sum = sr_add<SR>(local_sum, prov_data[begin + i]);
  }
  return warp_reduce_sr<SR>(local_sum, tile);
}

}  // namespace SRDatalog::GPU
