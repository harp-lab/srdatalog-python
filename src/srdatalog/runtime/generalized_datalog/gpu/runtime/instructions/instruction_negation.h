#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for Negation (Anti-Join): Filters if tuple exists in relation.
 *
 * Supports ConstArgs for filtering on constant values within negated literals.
 * ConstArgs is a tuple of std::pair<std::integral_constant<int, ColIdx>,
 *                                   std::integral_constant<int64_t, Val>>
 */
template <typename Index, typename PrefixVars, std::size_t HS, typename VHM, typename ConstArgs,
          typename... TailOpsNeg, typename VarPosMapNeg, std::size_t IteratorOffsetNeg,
          typename RelationPosMapNeg, int GroupSizeNeg, bool IsLeafLevelNeg>
struct GPUPipelineExecutor<
    std::tuple<mir::Negation<Index, PrefixVars, HS, VHM, ConstArgs>, TailOpsNeg...>, VarPosMapNeg,
    IteratorOffsetNeg, RelationPosMapNeg, GroupSizeNeg, IsLeafLevelNeg> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile,
                                 state::VarStore<VarPosMapNeg, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapNeg, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    // Access handle at HS (HandleStart) - where this Negation's handle is stored
    // NOTE: We use HS, NOT IteratorOffsetNeg. HS is the statically-computed handle position
    // for this Negation operation, while IteratorOffsetNeg is the pipeline offset.
    auto handle = handles[HS];

    // First, apply constant prefix navigation (ConstArgs)
    // Each pair is std::pair<std::integral_constant<int, ColIdx>, std::integral_constant<int64_t,
    // Val>>
    [&]<typename... ConstPairs>(std::tuple<ConstPairs...>) {
      ((handle = handle.prefix(static_cast<ValueType>(ConstPairs::second_type::value), tile,
                               handles.get_view(HS))),
       ...);
    }(ConstArgs{});

    // Then, apply variable prefix navigation using PrefixVars (bound variables)
    // Each variable in PrefixVars should be used to navigate the handle
    [&]<typename... Vars>(std::tuple<Vars...>) {
      ((handle = handle.prefix(vars.template get<Vars>(), tile, handles.get_view(HS))), ...);
    }(PrefixVars{});

    // Check if any matching tuple exists for the current prefix
    // For Anti-Join, we short-circuit if found (degree > 0)
    bool exists = false;
    if (handle.valid()) {
      exists = (handle.degree() > 0);
    }

    // Anti-Join: Continue ONLY if NOT exists
    if (!exists) {
      // Continue pipeline - Negation consumes 1 iterator
      constexpr std::size_t kNextOffset = IteratorOffsetNeg + 1;

      GPUPipelineExecutor<std::tuple<TailOpsNeg...>, VarPosMapNeg, kNextOffset, RelationPosMapNeg,
                          GroupSizeNeg, IsLeafLevelNeg>::template execute<SR>(tile, vars, handles,
                                                                              provs, output_ctx);
    }
    // Else: Short-circuit (filter out this tuple)
  }
};

}  // namespace SRDatalog::GPU
