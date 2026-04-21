#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for LetClause: Computes a new variable via C++ expression.
 */
template <typename VarToBind, typename CapturedVars, auto Func, typename... TailOpsLet,
          typename VarPosMapLet, std::size_t IteratorOffsetLet, typename RelationPosMapLet,
          int GroupSizeLet, bool IsLeafLevelLet>
struct GPUPipelineExecutor<
    std::tuple<AST::LetClause<VarToBind, AST::CppExpr<CapturedVars, Func>>, TailOpsLet...>,
    VarPosMapLet, IteratorOffsetLet, RelationPosMapLet, GroupSizeLet, IsLeafLevelLet> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile,
                                 state::VarStore<VarPosMapLet, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapLet, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    using TypesTuple = function_arg_types_t<decltype(Func)>;
    using VarStoreType = state::VarStore<VarPosMapLet, ValueType, NumVars>;

    auto dep_vals =
        DeviceDecodedValueExtractor<VarPosMapLet, VarStoreType, CapturedVars, TypesTuple>::extract(
            vars);

    auto new_val = std::apply(Func, dep_vals);

    size_t new_val_encoded = 0;
    if constexpr (std::is_same_v<decltype(new_val), bool>) {
      new_val_encoded = new_val ? 1 : 0;
    } else {
      if constexpr (std::is_integral_v<decltype(new_val)> || std::is_enum_v<decltype(new_val)>) {
        new_val_encoded = static_cast<size_t>(new_val);
      } else if constexpr (std::is_floating_point_v<decltype(new_val)>) {
        if constexpr (sizeof(new_val) == 8)
          new_val_encoded = *reinterpret_cast<size_t*>(&new_val);
        else {
          uint32_t temp = *reinterpret_cast<uint32_t*>(&new_val);
          new_val_encoded = static_cast<size_t>(temp);
        }
      } else {
        new_val_encoded = static_cast<size_t>(new_val);
      }
    }

    vars.template set<VarToBind>(new_val_encoded);

    // Recurse: LetClause consumes 0 iterators
    constexpr std::size_t kNextOffset = IteratorOffsetLet;

    GPUPipelineExecutor<std::tuple<TailOpsLet...>, VarPosMapLet, kNextOffset, RelationPosMapLet,
                        GroupSizeLet, IsLeafLevelLet>::template execute<SR>(tile, vars, handles,
                                                                            provs, output_ctx);
  }
};

}  // namespace SRDatalog::GPU
