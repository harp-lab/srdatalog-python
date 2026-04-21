#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for Aggregation: Computes aggregate over a variable.
 */
template <typename VarToBind, typename Index, typename PrefixVars, typename AggFunc, std::size_t HS,
          typename VHM, typename... TailOpsAgg, typename VarPosMapAgg,
          std::size_t IteratorOffsetAgg, typename RelationPosMapAgg, int GroupSizeAgg,
          bool IsLeafLevelAgg>
struct GPUPipelineExecutor<
    std::tuple<mir::Aggregate<VarToBind, Index, PrefixVars, AggFunc, HS, VHM>, TailOpsAgg...>,
    VarPosMapAgg, IteratorOffsetAgg, RelationPosMapAgg, GroupSizeAgg, IsLeafLevelAgg> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile,
                                 state::VarStore<VarPosMapAgg, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMapAgg, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    // Access handle at IteratorOffsetAgg
    auto handle = handles[IteratorOffsetAgg];

    // Apply prefix navigation using PrefixVars (bound variables)
    // Each variable in PrefixVars should be used to navigate the handle
    // Use the existing tile for prefix (avoid tiled_partition<1> on a size-1 tile)
    [&]<typename... Vars>(std::tuple<Vars...>) {
      ((handle =
            handle.prefix(vars.template get<Vars>(), tile, handles.get_view(IteratorOffsetAgg))),
       ...);
    }(PrefixVars{});

    using AccType = typename AggFunc::ResultType;
    AccType acc = AggFunc::init();

    if (handle.valid()) {
      // Determine schema and value type same as CPU
      using ColSeq = typename Index::column_indexes_type;
      using Schema = typename Index::schema_type;
      // Last column is the aggregated one
      constexpr int kLastColIdx = []<int... Cols>(std::integer_sequence<int, Cols...>) {
        int cols[] = {Cols...};
        return cols[sizeof...(Cols) - 1];
      }(ColSeq{});
      using AttrTuple = typename Schema::attr_ts_type;
      using ValType = std::tuple_element_t<kLastColIdx, AttrTuple>;

      auto& agg_view = handles.get_view(IteratorOffsetAgg);
      uint32_t count = handle.degree();

      if constexpr (GroupSizeAgg > 1 && std::is_arithmetic_v<AccType>) {
        // Parallel reduction optimization for arithmetic types
        uint32_t lane = tile.thread_rank();
        AccType thread_acc = AggFunc::init();
        for (uint32_t i = lane; i < count; i += GroupSizeAgg) {
          auto val = device_decode_from_size_t<ValType>(handle.get_value_at(agg_view, i));
          thread_acc = AggFunc::apply(thread_acc, val);
        }

        // Warp reduction using shuffle
        // We assume GroupSizeAgg is power of 2
        for (int offset = GroupSizeAgg / 2; offset > 0; offset /= 2) {
          AccType other = tile.shfl_down(thread_acc, offset);
          thread_acc = AggFunc::combine(thread_acc, other);
        }

        // Broadcast result from rank 0 to all
        acc = tile.shfl(thread_acc, 0);

      } else {
        // Serial fallback (single thread or non-arithmetic)
        if (tile.thread_rank() == 0) {
          for (uint32_t i = 0; i < count; ++i) {
            auto val = device_decode_from_size_t<ValType>(handle.get_value_at(agg_view, i));
            acc = AggFunc::apply(acc, val);
          }
        }
        // If GroupSizeAgg > 1, we must broadcast the result!
        if constexpr (GroupSizeAgg > 1) {
          if constexpr (std::is_arithmetic_v<AccType>) {
            acc = tile.shfl(acc, 0);  // easy broadcast
          } else {
            // Limitation: structured types
          }
        }
      }
    }

    // Encoding result
    size_t encoded_res = 0;
    if constexpr (std::is_same_v<AccType, bool>) {
      encoded_res = acc ? 1 : 0;
    } else {
      if constexpr (std::is_integral_v<AccType> || std::is_enum_v<AccType>) {
        encoded_res = static_cast<size_t>(acc);
      } else if constexpr (std::is_floating_point_v<AccType>) {
        if constexpr (sizeof(AccType) == 8)
          encoded_res = *reinterpret_cast<const size_t*>(&acc);
        else {
          uint32_t temp = *reinterpret_cast<const uint32_t*>(&acc);
          encoded_res = static_cast<size_t>(temp);
        }
      } else {
        encoded_res = static_cast<size_t>(acc);
      }
    }

    vars.template set<VarToBind>(encoded_res);

    // Recurse: Aggregate consumes 1 iterator
    constexpr std::size_t kNextOffset = IteratorOffsetAgg + 1;

    GPUPipelineExecutor<std::tuple<TailOpsAgg...>, VarPosMapAgg, kNextOffset, RelationPosMapAgg,
                        GroupSizeAgg, IsLeafLevelAgg>::template execute<SR>(tile, vars, handles,
                                                                            provs, output_ctx);
  }
};

}  // namespace SRDatalog::GPU
