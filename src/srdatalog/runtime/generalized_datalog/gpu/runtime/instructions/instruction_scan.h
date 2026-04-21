#pragma once

#include "../pipeline_executor_base.h"
#include <cstdio>  // For printf

namespace SRDatalog::GPU {

// Specialization for Scan
template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HS, typename VHM,
          typename... TailOps, typename VarPosMap, std::size_t IteratorOffset,
          typename RelationPosMap, int GroupSize, bool IsLeafLevel>
struct GPUPipelineExecutor<
    std::tuple<mir::Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>, TailOps...>, VarPosMap,
    IteratorOffset, RelationPosMap, GroupSize, IsLeafLevel> {

  static constexpr std::size_t kCurrentOffset = IteratorOffset;
  static constexpr std::size_t kNextOffset = IteratorOffset + 1;
  static constexpr std::size_t num_scan_vars = std::tuple_size_v<VarsTuple>;
  using TailOpsTuple = std::tuple<TailOps...>;

  static constexpr int kNextGroupSize = (GroupSize > 1) ? (GroupSize / 2) : 1;
  using NextPipeline = GPUPipelineExecutor<TailOpsTuple, VarPosMap, kNextOffset, RelationPosMap,
                                           kNextGroupSize, IsLeafLevel>;

  using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

  template <typename V>
  using IndicesForVar =
      typename detail::GetIndicesToUpdate<TailOpsTuple, V,
                                          boost::mp11::mp_size_t<kNextOffset>>::type;

  using IndicesLists = boost::mp11::mp_transform<IndicesForVar, VarsList>;
  using FlattenedIndices = boost::mp11::mp_apply<boost::mp11::mp_append, IndicesLists>;
  using IndicesToUpdate = boost::mp11::mp_unique<FlattenedIndices>;

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    auto& handle = handles[kCurrentOffset];
    auto view = handles.get_view(kCurrentOffset);

    if (!handle.valid())
      return;

    uint32_t total_rows = handle.degree();
    uint32_t start_offset = handle.begin();
    uint32_t lane = tile.thread_rank();

    // DEBUG: Print scan range
    // if (tile.thread_rank() == 0) {
    //   if constexpr (kCurrentOffset == 0) {  // Only print for the very first handle
    //     printf("[GPU] Scan Handle %lu (Root). Range: [%u, %u) Rows: %u\n", kCurrentOffset,
    //            start_offset, handle.end(), total_rows);
    //   }
    // }

    using Schema = typename IndexSpec::schema_type;

    constexpr bool has_indices_to_update = (boost::mp11::mp_size<IndicesToUpdate>::value > 0);

    auto saved_handles = [&]() {
      if constexpr (has_indices_to_update) {
        return SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);
      } else {
        return std::tuple<>{};
      }
    }();

    // Iterate over scan range
    if constexpr (kNextGroupSize < GroupSize) {
      auto sub_tile = cg::tiled_partition<kNextGroupSize>(tile);
      uint32_t subgroup_id = tile.thread_rank() / kNextGroupSize;
      uint32_t num_subgroups = GroupSize / kNextGroupSize;

      for (uint32_t i = subgroup_id; i < total_rows; i += num_subgroups) {
        // Restore handles
        if constexpr (has_indices_to_update) {
          SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_handles);
        }

        uint32_t row_idx = start_offset + i;

        // Bind variables using unified accessor (supports both DSAI and LSM)
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          ((vars.template set<boost::mp11::mp_at_c<VarsTuple, Is>>(view.get_value(Is, row_idx))),
           ...);
        }(std::make_index_sequence<num_scan_vars>{});

        // Set provenance using unified accessor
        provs.template set<Schema>(view.get_provenance(row_idx));

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          (scan_update_handles_for_var<TailOpsTuple, boost::mp11::mp_at_c<VarsTuple, Is>,
                                       decltype(handles), ValueType, decltype(sub_tile),
                                       kNextOffset>(handles, view.get_value(Is, row_idx), sub_tile),
           ...);
        }(std::make_index_sequence<num_scan_vars>{});

        // Recurse
        NextPipeline::template execute<SR>(sub_tile, vars, handles, provs, output_ctx);
      }
    } else {
      // Sequential fallback
      for (uint32_t i = 0; i < total_rows; i++) {
        if constexpr (has_indices_to_update) {
          SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_handles);
        }

        uint32_t row_idx = start_offset + i;

        // Bind vars using unified accessor (supports both DSAI and LSM)
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          ((vars.template set<boost::mp11::mp_at_c<VarsTuple, Is>>(view.get_value(Is, row_idx))),
           ...);
        }(std::make_index_sequence<num_scan_vars>{});

        provs.template set<Schema>(view.get_provenance(row_idx));

        // Update dependent handles
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          (scan_update_handles_for_var<TailOpsTuple, boost::mp11::mp_at_c<VarsTuple, Is>,
                                       decltype(handles), ValueType, decltype(tile), kNextOffset>(
               handles, view.get_value(Is, row_idx), tile),
           ...);
        }(std::make_index_sequence<num_scan_vars>{});

        NextPipeline::template execute<SR>(tile, vars, handles, provs, output_ctx);
      }
    }
  }

 private:
  template <typename TupleOfOps, typename Var, typename HandleStoreT, typename ValueType,
            typename Group, std::size_t Offset>
  __device__ static void scan_update_handles_for_var(HandleStoreT& store, ValueType val,
                                                     Group tile) {
    using IndicesToUpdateSub =
        typename detail::GetIndicesToUpdate<TupleOfOps, Var, boost::mp11::mp_size_t<Offset>>::type;
    boost::mp11::mp_for_each<IndicesToUpdateSub>([&](auto I) {
      constexpr std::size_t idx = decltype(I)::value;
      auto& handle = store[idx];
      auto new_handle = handle.prefix(val, tile, store.get_view(idx));
      handle = new_handle;
    });
  }
};

}  // namespace SRDatalog::GPU
