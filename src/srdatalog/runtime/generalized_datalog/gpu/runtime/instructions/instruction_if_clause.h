#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for IfClause: Filters execution via C++ predicate.
 */
template <typename BoundVars, auto Func, typename... TailOpsIf, typename VarPosMapIf,
          std::size_t IteratorOffsetIf, typename RelationPosMapIf, int GroupSizeIf,
          bool IsLeafLevelIf>
struct GPUPipelineExecutor<std::tuple<AST::IfClause<AST::CppExpr<BoundVars, Func>>, TailOpsIf...>,
                           VarPosMapIf, IteratorOffsetIf, RelationPosMapIf, GroupSizeIf,
                           IsLeafLevelIf> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile, state::VarStore<VarPosMapIf, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapIf, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    using TypesTuple = function_arg_types_t<decltype(Func)>;
    using VarStoreType = state::VarStore<VarPosMapIf, ValueType, NumVars>;

    auto dep_vals =
        DeviceDecodedValueExtractor<VarPosMapIf, VarStoreType, BoundVars, TypesTuple>::extract(
            vars);

    bool result = std::apply(Func, dep_vals);

    if (result) {
      // Recurse: IfClause consumes 0 iterators
      constexpr std::size_t kNextOffset = IteratorOffsetIf;

      GPUPipelineExecutor<std::tuple<TailOpsIf...>, VarPosMapIf, kNextOffset, RelationPosMapIf,
                          GroupSizeIf, IsLeafLevelIf>::template execute<SR>(tile, vars, handles,
                                                                            provs, output_ctx);
    }
  }
};

}  // namespace SRDatalog::GPU
