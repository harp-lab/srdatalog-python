#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for DestinationRelation: Emits result tuple.
 */
template <typename Schema, typename Terms, std::size_t Ver, typename IndexSpec,
          typename ActiveHandlesMap, typename... TailOpsDest, typename VarPosMapDest,
          std::size_t IteratorOffsetDest, typename RelationPosMapDest, int GroupSizeDest,
          bool IsLeafLevelDest>
struct GPUPipelineExecutor<
    std::tuple<mir::DestinationRelation<Schema, Terms, Ver, IndexSpec, ActiveHandlesMap>,
               TailOpsDest...>,
    VarPosMapDest, IteratorOffsetDest, RelationPosMapDest, GroupSizeDest, IsLeafLevelDest> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile,
                                 state::VarStore<VarPosMapDest, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapDest, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    auto combined_prov = provs.combine_all();

    if (true) {  // should_emit (simplified from commented out deduplication)
      // When GroupSizeDest == 1, each thread processes independently, so always emit
      // When GroupSizeDest > 1, only thread 0 emits (cooperative group behavior)
      if constexpr (GroupSizeDest == 1) {
        output_ctx.template emit<Schema, Terms>(vars, combined_prov);
      } else {
        if (tile.thread_rank() == 0) {
          output_ctx.template emit<Schema, Terms>(vars, combined_prov);
        }
      }
    }

    // Continue: DestinationRelation consumes 0 iterators
    constexpr std::size_t kNextOffset = IteratorOffsetDest;

    GPUPipelineExecutor<std::tuple<TailOpsDest...>, VarPosMapDest, kNextOffset, RelationPosMapDest,
                        GroupSizeDest, IsLeafLevelDest>::template execute<SR>(tile, vars, handles,
                                                                              provs, output_ctx);
  }
};

}  // namespace SRDatalog::GPU
