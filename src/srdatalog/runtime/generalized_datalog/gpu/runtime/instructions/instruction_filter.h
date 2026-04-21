#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for Filter: Applies predicate to bound variables.
 *
 * @details Extracts bound variables from VarStore, evaluates the predicate,
 * and only continues pipeline if predicate returns true.
 */
template <typename BoundVars, typename Predicate, typename... TailOpsFilter,
          typename VarPosMapFilter, std::size_t IteratorOffsetFilter, typename RelationPosMapFilter,
          int GroupSizeFilter, bool IsLeafLevelFilter>
struct GPUPipelineExecutor<std::tuple<mir::Filter<BoundVars, Predicate>, TailOpsFilter...>,
                           VarPosMapFilter, IteratorOffsetFilter, RelationPosMapFilter,
                           GroupSizeFilter, IsLeafLevelFilter> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile,
                                 state::VarStore<VarPosMapFilter, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapFilter, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    // Extract bound variables and evaluate predicate
    bool should_continue = evaluate_predicate<BoundVars, Predicate>(vars);

    if (should_continue) {
      // Continue pipeline - Filter consumes 0 iterators
      constexpr std::size_t kNextOffset = IteratorOffsetFilter;

      GPUPipelineExecutor<std::tuple<TailOpsFilter...>, VarPosMapFilter, kNextOffset,
                          RelationPosMapFilter, GroupSizeFilter,
                          IsLeafLevelFilter>::template execute<SR>(tile, vars, handles, provs,
                                                                   output_ctx);
    }
    // If predicate is false, don't continue - tuple is filtered out
  }

 private:
  // Helper to extract vars and call predicate
  template <typename VarsTuple, typename Pred, typename VarStore>
  __device__ static bool evaluate_predicate(const VarStore& vars) {
    return invoke_predicate_with_vars<VarsTuple, Pred>(
        vars, std::make_index_sequence<std::tuple_size_v<VarsTuple>>{});
  }

  template <typename VarsTuple, typename Pred, typename VarStore, std::size_t... Is>
  __device__ static bool invoke_predicate_with_vars(const VarStore& vars,
                                                    std::index_sequence<Is...>) {
    // Get each variable's value and call predicate
    Pred pred{};
    return pred(vars.template get<std::tuple_element_t<Is, VarsTuple>>()...);
  }
};

}  // namespace SRDatalog::GPU
