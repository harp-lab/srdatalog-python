#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

/**
 * @brief Abstract Join Executor - Dispatches execution based on Strategy.
 */
template <typename Strategy>
struct JoinExecutor;

/**
 * @brief Helper to find index of a type in a tuple
 */
template <typename T, typename Tuple>
struct TypeIndex;

template <typename T, typename... Types>
struct TypeIndex<T, std::tuple<T, Types...>> {
  static constexpr std::size_t value = 0;
};

template <typename T, typename U, typename... Types>
struct TypeIndex<T, std::tuple<U, Types...>> {
  static constexpr std::size_t value = 1 + TypeIndex<T, std::tuple<Types...>>::value;
};

/**
 * @brief Default Join Executor (WCOJ / Intersection).
 *
 * Implements the standard LFTJ intersection logic.
 */
template <>
struct JoinExecutor<mir::DefaultJoinStrategy> {
  template <typename VarToBind, typename SourcesTuple, typename TailOpsTuple, typename VarPosMap,
            std::size_t IteratorOffset, typename RelationPosMap, int GroupSize, bool IsLeafLevel,
            Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    // Helper alias for the next pipeline stage
    constexpr int kNextGroupSize = (GroupSize > 1) ? (GroupSize / 2) : 1;
    constexpr std::size_t kNextOffset =
        IteratorOffset + mir::CountInputIterators<SourcesTuple>::value;

    using NextPipeline = GPUPipelineExecutor<TailOpsTuple, VarPosMap, kNextOffset, RelationPosMap,
                                             kNextGroupSize, IsLeafLevel>;
    // Helper alias for single-thread pipeline
    using SingleThreadPipeline =
        GPUPipelineExecutor<TailOpsTuple, VarPosMap, kNextOffset, RelationPosMap, 1, true>;

    // Helper lambda to invoke execute_intersection/single_source with Sources... unpacked
    auto dispatch = [&]<typename... Sources>(std::tuple<Sources...>) {
      constexpr std::size_t num_sources = sizeof...(Sources);
      constexpr bool is_intersection = (num_sources > 1);

      if constexpr (is_intersection) {
        execute_intersection<VarToBind, TailOpsTuple, GroupSize, Sources...>(
            tile, vars, handles, provs, output_ctx, static_cast<NextPipeline*>(nullptr),
            std::integral_constant<std::size_t, kNextOffset>{});
      } else {
        execute_single_source<VarToBind, TailOpsTuple, GroupSize, Sources...>(
            tile, vars, handles, provs, output_ctx, static_cast<SingleThreadPipeline*>(nullptr),
            std::integral_constant<std::size_t, kNextOffset>{});
      }
    };

    dispatch(SourcesTuple{});
  }

 private:
  // Get source handles using linear offsets
  template <typename... Sources, typename HandleStoreT>
  __device__ static auto get_source_handles(HandleStoreT& handles, std::size_t StartOffset) {
    constexpr std::size_t num_sources = sizeof...(Sources);
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      // Each source i is at StartOffset + i
      return std::make_tuple(handles[StartOffset + Is]...);
    }(std::make_index_sequence<num_sources>{});
  }

  // Metafunction to map a handle index to its position in the sources tuple
  // Returns -1 if the handle is not a source
  template <std::size_t HandleOffset, std::size_t NumSources, std::size_t StartOffset>
  static constexpr int get_source_index() {
    if (HandleOffset >= StartOffset && HandleOffset < StartOffset + NumSources) {
      return HandleOffset - StartOffset;
    }
    return -1;
  }

  template <typename IndicesList, std::size_t NumSources, std::size_t StartOffset,
            typename HandleStoreT, typename ValueType, typename Group, typename PositionsArray>
  __device__ static void update_handles_with_indices(HandleStoreT& store,
                                                     [[maybe_unused]] ValueType val,
                                                     [[maybe_unused]] Group tile,
                                                     const PositionsArray& positions) {
    boost::mp11::mp_for_each<IndicesList>([&](auto I) {
      constexpr std::size_t idx = decltype(I)::value;
      constexpr int source_idx = get_source_index<idx, NumSources, StartOffset>();

      auto& handle = store[idx];
      if constexpr (source_idx >= 0) {
        // This handle was a source in the intersection.
        // Intersection iterator gives us exact position - just create single-row child handle.
        // No search needed - the position is known from the iterator.
        auto pos = positions[source_idx];
        handle = handle.child(handle.begin() + pos, handle.begin() + pos + 1);
      } else {
        // This handle was NOT a source (e.g., dependency). Use standard prefix search.
        handle = handle.prefix(val, tile, store.get_view(idx));
      }
    });
  }

  // Multiple sources: warp-cooperative intersection
  template <typename VarToBind, typename TailOpsTuple, int GroupSize, typename... Sources,
            typename Semiring, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename VarPosMap,
            typename RelationPosMap, typename Group, typename OutputCtx, typename NextPipeline,
            std::size_t NextOffset>
  __device__ static void execute_intersection(
      Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, Semiring, NumRelations>& provs, OutputCtx& output_ctx,
      NextPipeline* /*dummy*/, std::integral_constant<std::size_t, NextOffset> /*offset_tag*/) {

    // Current offset is NextOffset - num_sources
    constexpr std::size_t kCurrentOffset = NextOffset - sizeof...(Sources);
    constexpr std::size_t kNumSources = sizeof...(Sources);
    auto source_handles = get_source_handles<Sources...>(handles, kCurrentOffset);

    using FirstSource = std::tuple_element_t<0, std::tuple<Sources...>>;
    using FirstIndexSpec = typename FirstSource::index_spec_type;
    using Schema = typename FirstIndexSpec::schema_type;
    using IndexType = DeviceSortedArrayIndex<typename Schema::semiring_type,
                                             typename Schema::attr_ts_type, ValueType, uint32_t>;

    auto intersect_view = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return IndexType::intersect(
          tile, std::get<Is>(source_handles).iterators(handles.get_view(kCurrentOffset + Is))...);
    }(std::make_index_sequence<sizeof...(Sources)>{});

    constexpr unsigned int CurrentGroupSize = GroupSize;
    constexpr bool has_more_column_joins = mir::HasMoreColumnJoins<TailOpsTuple>::value;
    constexpr unsigned int NextGroupSize =
        (has_more_column_joins && CurrentGroupSize > 1)
            ? CurrentGroupSize
            : ((CurrentGroupSize > 1) ? (unsigned int)(CurrentGroupSize / 2) : 1u);

    // Save handles that will be updated
    using IndicesToUpdate =
        typename detail::GetIndicesToUpdate<TailOpsTuple, VarToBind,
                                            boost::mp11::mp_size_t<NextOffset>>::type;
    constexpr bool has_indices_to_update = (boost::mp11::mp_size<IndicesToUpdate>::value > 0);

    // Helper macro to reduce code duplication in the loop
#define PROCESS_INTERSECTION_MATCH(SUB_TILE)                                                       \
  ValueType val = it.value();                                                                      \
  auto positions = it.positions();                                                                 \
  vars.template set<VarToBind>(val);                                                               \
                                                                                                   \
  if constexpr (has_indices_to_update) {                                                           \
    auto saved_handles = SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);        \
    update_handles_with_indices<IndicesToUpdate, kNumSources, kCurrentOffset>(                     \
        handles, val, SUB_TILE, positions);                                                        \
    NextPipeline::template execute<Semiring>(SUB_TILE, vars, handles, provs, output_ctx);          \
    SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_handles);         \
  } else {                                                                                         \
    NextPipeline::template execute<Semiring>(SUB_TILE, vars, handles, provs, output_ctx);          \
  }

    if constexpr (NextGroupSize < CurrentGroupSize) {
      auto sub_tile = cg::tiled_partition<NextGroupSize>(tile);
      uint32_t subgroup_id = tile.thread_rank() / NextGroupSize;
      uint32_t num_subgroups = CurrentGroupSize / NextGroupSize;

      uint32_t loop_idx = 0;
      for (auto it = intersect_view.begin(); it.valid(); it.next(), ++loop_idx) {
        if (loop_idx % num_subgroups == subgroup_id) {
          PROCESS_INTERSECTION_MATCH(sub_tile);
        }
      }
    } else if constexpr (NextGroupSize == CurrentGroupSize) {
      for (auto it = intersect_view.begin(); it.valid(); it.next()) {
        PROCESS_INTERSECTION_MATCH(tile);
      }
    } else {
      // Sequential fallback? (Should not be hit if NextGroupSize valid)
      for (auto it = intersect_view.begin(); it.valid(); it.next()) {
        PROCESS_INTERSECTION_MATCH(tile);
      }
    }
#undef PROCESS_INTERSECTION_MATCH
  }

  // Single source
  template <typename VarToBind, typename TailOpsTuple, int GroupSize, typename... Sources,
            typename Semiring, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename VarPosMap,
            typename RelationPosMap, typename Group, typename OutputCtx, typename NextPipeline,
            std::size_t NextOffset>
  __device__ static void execute_single_source(
      Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, Semiring, NumRelations>& provs, OutputCtx& output_ctx,
      NextPipeline* /*dummy*/, std::integral_constant<std::size_t, NextOffset> /*offset_tag*/) {

    constexpr std::size_t kCurrentOffset = NextOffset - sizeof...(Sources);  // Should be 1 source

    auto& handle = handles[kCurrentOffset];
    using SingleSource = std::tuple_element_t<0, std::tuple<Sources...>>;
    using IndexSpec = typename SingleSource::index_spec_type;
    using Schema = typename IndexSpec::schema_type;

    if (!handle.valid()) {
      return;
    }

    if (!handle.valid()) {
      return;
    }

    auto& view = handles.get_view(kCurrentOffset);
    uint32_t degree = handle.degree();
    uint32_t lane = tile.thread_rank();

    constexpr unsigned int CurrentGroupSize = GroupSize;

    using IndicesToUpdate =
        typename detail::GetIndicesToUpdate<TailOpsTuple, VarToBind,
                                            boost::mp11::mp_size_t<NextOffset>>::type;
    constexpr bool has_indices_to_update = (boost::mp11::mp_size<IndicesToUpdate>::value > 0);

    // For single source, we don't need positions() as much (it's just idx),
    // but we can still use the optimized update path.
    // However, execute_single_source loop is different.
    // It iterates by index directly.

    auto execute_body = [&](uint32_t idx, auto SUB_TILE) {
      ValueType val = handle.get_value_at(view, idx);
      auto prov = handle.get_provenance_at(view, idx);

      vars.template set<VarToBind>(val);
      provs.template set<Schema>(prov);

      if constexpr (has_indices_to_update) {
        auto saved_handles = SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);

        // Construct a single-element positions array for the update helper
        std::array<uint32_t, 1> positions = {idx};
        constexpr std::size_t kNumSources = 1;
        update_handles_with_indices<IndicesToUpdate, kNumSources, kCurrentOffset>(
            handles, val, SUB_TILE, positions);

        NextPipeline::template execute<Semiring>(SUB_TILE, vars, handles, provs, output_ctx);
        SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_handles);
      } else {
        NextPipeline::template execute<Semiring>(SUB_TILE, vars, handles, provs, output_ctx);
      }
    };

    if constexpr (CurrentGroupSize > 1) {
      auto single_tile = cg::tiled_partition<1>(tile);
      for (uint32_t idx = lane; idx < degree; idx += CurrentGroupSize) {
        execute_body(idx, single_tile);
      }
    } else {
      for (uint32_t idx = lane; idx < degree; idx += CurrentGroupSize) {
        execute_body(idx, tile);
      }
    }
  }
};

/**
 * @brief Specialization for ColumnJoin: Delegates to JoinExecutor based on Strategy.
 */
template <typename VarToBind, typename... Sources, typename Strategy, std::size_t HS, typename VHM,
          typename... TailOps, typename VarPosMap, std::size_t IteratorOffset,
          typename RelationPosMap, int GroupSize, bool IsLeafLevel>
struct GPUPipelineExecutor<
    std::tuple<mir::ColumnJoin<VarToBind, std::tuple<Sources...>, Strategy, HS, VHM>, TailOps...>,
    VarPosMap, IteratorOffset, RelationPosMap, GroupSize, IsLeafLevel> {

  // Expose OpsTuple type for helpers
  using ops_type =
      std::tuple<mir::ColumnJoin<VarToBind, std::tuple<Sources...>, Strategy, HS, VHM>, TailOps...>;

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    // Delegate to JoinExecutor
    JoinExecutor<Strategy>::template execute<VarToBind, std::tuple<Sources...>,
                                             std::tuple<TailOps...>, VarPosMap, IteratorOffset,
                                             RelationPosMap, GroupSize, IsLeafLevel>(
        tile, vars, handles, provs, output_ctx);
  }
};

}  // namespace SRDatalog::GPU
