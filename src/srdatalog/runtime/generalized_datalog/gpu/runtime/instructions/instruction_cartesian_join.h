#pragma once

#include "../pipeline_executor_base.h"

namespace SRDatalog::GPU {

namespace detail {

/**
 * @brief Helper to execute next pipeline with proper handle save/restore and VHM prefixing.
 *
 * This extracts the common pattern of:
 * 1. Computing which handles need to be updated for downstream operations
 * 2. Saving those handles
 * 3. Prefixing handles with bound variable values
 * 4. Executing the next pipeline stage
 * 5. Restoring the saved handles
 */
template <Semiring SR, typename TailOpsTuple, std::size_t NextOffset, typename VarsTuple,
          typename VarPosMap, typename ValueType, std::size_t NumVars, typename HandleType,
          std::size_t NumHandles, typename RelationPosMap, std::size_t NumRelations, typename Group,
          typename SingleTileGroup, typename OutputCtx>
__device__ void execute_next_with_handle_management(
    Group& tile, SingleTileGroup& single_tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
    state::HandleStore<HandleType, NumHandles>& handles,
    state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx) {

  using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;
  constexpr std::size_t num_vars = std::tuple_size_v<VarsTuple>;

  // Compute all handle indices that need to be updated across all variables
  using AllIndicesToUpdate = boost::mp11::mp_unique<
      boost::mp11::mp_apply<boost::mp11::mp_append,
                            boost::mp11::mp_transform_q<
                                boost::mp11::mp_quote<GetIndicesToUpdateBound<
                                    TailOpsTuple, boost::mp11::mp_size_t<NextOffset>>::template fn>,
                                VarsList>>>;

  constexpr bool has_indices_to_update = (boost::mp11::mp_size<AllIndicesToUpdate>::value > 0);

  using NextPipeline =
      GPUPipelineExecutor<TailOpsTuple, VarPosMap, NextOffset, RelationPosMap, 1, true>;

  if constexpr (has_indices_to_update) {
    // Save handles that will be modified
    auto saved_handles = SRDatalog::state::save_handles_to_tuple<AllIndicesToUpdate>(handles);

    // Update handles for each bound variable
    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<num_vars>>([&](auto I_var) {
      constexpr std::size_t i_var = decltype(I_var)::value;
      using VarType = boost::mp11::mp_at_c<VarsList, i_var>;
      using IndicesToUpdate = typename GetIndicesToUpdate<TailOpsTuple, VarType,
                                                          boost::mp11::mp_size_t<NextOffset>>::type;

      boost::mp11::mp_for_each<IndicesToUpdate>([&](auto I_idx) {
        constexpr std::size_t h_idx = decltype(I_idx)::value;
        handles[h_idx] = handles[h_idx].prefix(vars.template get<VarType>(), single_tile,
                                               handles.get_view(h_idx));
      });
    });

    NextPipeline::template execute<SR>(single_tile, vars, handles, provs, output_ctx);
    SRDatalog::state::restore_handles_from_tuple<AllIndicesToUpdate>(handles, saved_handles);
  } else {
    NextPipeline::template execute<SR>(single_tile, vars, handles, provs, output_ctx);
  }
}

}  // namespace detail

/**
 * @brief Specialization for CartesianJoin: Flat cartesian product iteration.
 *
 * @details Binds multiple variables simultaneously via flat cartesian product
 * iteration over unmaterialized handle spans. This avoids nested loops and
 * handle materialization overhead.
 */
template <typename VarsTuple, typename SourcesTuple, std::size_t HS, typename VHM,
          typename... TailOps, typename VarPosMap, std::size_t IteratorOffset,
          typename RelationPosMap, int GroupSize, bool IsLeafLevel>
struct GPUPipelineExecutor<
    std::tuple<mir::CartesianJoin<VarsTuple, SourcesTuple, HS, VHM>, TailOps...>, VarPosMap,
    IteratorOffset, RelationPosMap, GroupSize, IsLeafLevel> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                 state::HandleStore<HandleType, NumHandles>& handles,
                                 state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
                                 OutputCtx& output_ctx) {

    constexpr std::size_t kNextOffset =
        IteratorOffset + mir::CountInputIterators<SourcesTuple>::value;

    // Get source handles
    auto source_handles = get_source_handles_from_tuple<SourcesTuple>(handles, IteratorOffset);

    // Execute flat cartesian product iteration
    execute_cartesian_product<SR>(tile, vars, handles, provs, output_ctx, source_handles,
                                  std::tuple<TailOps...>{},
                                  std::integral_constant<std::size_t, kNextOffset>{},
                                  std::integral_constant<std::size_t, IteratorOffset>{});
  }

 private:
  // Helper to get source handles from tuple of Sources
  template <typename SourcesTupleType, typename HandleStoreT>
  __device__ static auto get_source_handles_from_tuple(HandleStoreT& handles,
                                                       std::size_t StartOffset) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(handles[StartOffset + Is]...);
    }(std::make_index_sequence<std::tuple_size_v<SourcesTupleType>>{});
  }

  // Helper to compute how many variables a source binds at compile time
  template <typename Source>
  constexpr static std::size_t get_source_var_count() {
    if constexpr (mir::is_scan_v<Source>) {
      return std::tuple_size_v<typename Source::vars_type>;
    } else if constexpr (mir::is_column_source_v<Source>) {
      // ColumnSource binds (total columns - prefix vars) variables
      using IndexSpec = typename Source::index_spec_type;
      using PrefixVars = typename Source::prefix_vars_type;
      constexpr std::size_t total_cols = IndexSpec::schema_type::Arity;
      constexpr std::size_t prefix_len = std::tuple_size_v<PrefixVars>;
      return total_cols - prefix_len;
    } else {
      return 0;
    }
  }

  // Helper template to compute cumulative variable offset for source at index I
  template <typename SourcesTupleType, std::size_t I>
  constexpr static std::size_t compute_var_offset() {
    if constexpr (I == 0) {
      return 0;
    } else {
      using PrevSource = std::tuple_element_t<I - 1, SourcesTupleType>;
      return compute_var_offset<SourcesTupleType, I - 1>() + get_source_var_count<PrevSource>();
    }
  }

  // Execute flat cartesian product iteration
  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx,
            typename TailOpsTupleType, std::size_t NextOffset, std::size_t CurrentIteratorOffset>
  __device__ static void execute_cartesian_product(
      Group tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx,
      auto& source_handles_tuple, TailOpsTupleType /*tail_ops*/,
      std::integral_constant<std::size_t, NextOffset> /*offset*/,
      std::integral_constant<std::size_t, CurrentIteratorOffset> /*iterator_offset*/) {

    constexpr std::size_t num_vars = std::tuple_size_v<VarsTuple>;
    constexpr std::size_t num_sources = std::tuple_size_v<SourcesTuple>;
    using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

    // Check if all source handles are valid
    bool all_valid = true;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((all_valid = all_valid && std::get<Is>(source_handles_tuple).valid()), ...);
    }(std::make_index_sequence<num_sources>{});

    if (!all_valid)
      return;

    // Extract degrees and compute total combinations
    uint32_t degrees[num_sources];
    uint32_t total = 1;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((degrees[Is] = std::get<Is>(source_handles_tuple).degree(), total *= degrees[Is]), ...);
    }(std::make_index_sequence<num_sources>{});

    if (total == 0)
      return;

    uint32_t lane = tile.thread_rank();
    uint32_t CurrentGroupSize = tile.size();

    // Create single_tile once for leaf-level operations
    auto single_tile = [&]() {
      if constexpr (GroupSize > 1) {
        return cg::tiled_partition<1>(tile);
      } else {
        return tile;
      }
    }();

    // Fast path for 2-source, 2-variable case (most common, e.g., TC join)
    if constexpr (num_vars == 2 && num_sources == 2) {
      execute_fast_path_2arity<SR, TailOpsTupleType, NextOffset, CurrentIteratorOffset>(
          tile, single_tile, vars, handles, provs, output_ctx, source_handles_tuple, degrees, total,
          lane, CurrentGroupSize);
    } else {
      execute_general_case<SR, TailOpsTupleType, NextOffset, CurrentIteratorOffset>(
          tile, single_tile, vars, handles, provs, output_ctx, source_handles_tuple, degrees, total,
          lane, CurrentGroupSize);
    }
  }

  // Fast path: 2 sources, 2 variables
  template <Semiring SR, typename TailOpsTupleType, std::size_t NextOffset,
            std::size_t CurrentIteratorOffset, typename Group, typename ValueType,
            std::size_t NumVars, typename HandleType, std::size_t NumHandles,
            std::size_t NumRelations, typename OutputCtx>
  __device__ static void execute_fast_path_2arity(
      Group& tile, auto& single_tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx,
      auto& source_handles_tuple, uint32_t* degrees, uint32_t total, uint32_t lane,
      uint32_t CurrentGroupSize) {

    uint32_t degree0 = degrees[0];
    uint32_t degree1 = degrees[1];
    const bool major_is_1 = (degree1 >= degree0);

    auto& handle0 = std::get<0>(source_handles_tuple);
    auto& handle1 = std::get<1>(source_handles_tuple);
    auto& view0 = handles.get_view(CurrentIteratorOffset + 0);
    auto& view1 = handles.get_view(CurrentIteratorOffset + 1);

    using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

#pragma unroll 2
    for (uint32_t flat_idx = lane; flat_idx < total; flat_idx += CurrentGroupSize) {
      uint32_t idx0 = 0, idx1 = 0;
      if (major_is_1) {
        idx0 = flat_idx / degree1;
        idx1 = flat_idx % degree1;
      } else {
        idx1 = flat_idx / degree0;
        idx0 = flat_idx % degree0;
      }

      // Bind variables using indexed accessors
      vars.template set<boost::mp11::mp_at_c<VarsList, 0>>(handle0.get_value_at(view0, idx0));
      vars.template set<boost::mp11::mp_at_c<VarsList, 1>>(handle1.get_value_at(view1, idx1));

      // Set provenance using indexed accessors (only if semiring has provenance)
      if constexpr (has_provenance_v<SR>) {
        using FirstSource = std::tuple_element_t<0, SourcesTuple>;
        using FirstIndexSpec = typename FirstSource::index_spec_type;
        using FirstSchema = typename FirstIndexSpec::schema_type;
        provs.template set<FirstSchema>(sr_mul<SR>(handle0.get_provenance_at(view0, idx0),
                                                   handle1.get_provenance_at(view1, idx1)));
      }

      // Execute next pipeline with handle management
      detail::execute_next_with_handle_management<SR, TailOpsTupleType, NextOffset, VarsTuple>(
          tile, single_tile, vars, handles, provs, output_ctx);
    }
  }

  // General case: N sources, M variables
  template <Semiring SR, typename TailOpsTupleType, std::size_t NextOffset,
            std::size_t CurrentIteratorOffset, typename Group, typename ValueType,
            std::size_t NumVars, typename HandleType, std::size_t NumHandles,
            std::size_t NumRelations, typename OutputCtx>
  __device__ static void execute_general_case(
      Group& tile, auto& single_tile, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx,
      auto& source_handles_tuple, uint32_t* degrees, uint32_t total, uint32_t lane,
      uint32_t CurrentGroupSize) {

    constexpr std::size_t num_sources = std::tuple_size_v<SourcesTuple>;
    using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

    // Compute cumulative products for index decomposition
    uint32_t cumulative_products[num_sources + 1];
    cumulative_products[num_sources] = 1;
    for (int i = num_sources - 1; i >= 0; --i) {
      cumulative_products[i] = cumulative_products[i + 1] * degrees[i];
    }

#pragma unroll 2
    for (uint32_t flat_idx = lane; flat_idx < total; flat_idx += CurrentGroupSize) {
      // Compute source indices
      uint32_t source_idx[num_sources];
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((source_idx[Is] = (flat_idx / cumulative_products[Is + 1]) % degrees[Is]), ...);
      }(std::make_index_sequence<num_sources>{});

      // Bind variables for each source
      bind_variables_from_sources<CurrentIteratorOffset>(vars, handles, source_handles_tuple,
                                                         source_idx);

      // Compute and set combined provenance (only if semiring has provenance)
      if constexpr (has_provenance_v<SR>) {
        set_combined_provenance<SR, CurrentIteratorOffset>(provs, handles, source_handles_tuple,
                                                           source_idx);
      }

      // Execute next pipeline with handle management
      detail::execute_next_with_handle_management<SR, TailOpsTupleType, NextOffset, VarsTuple>(
          tile, single_tile, vars, handles, provs, output_ctx);
    }
  }

  // Helper to bind variables from all sources
  template <std::size_t CurrentIteratorOffset, typename ValueType, std::size_t NumVars,
            typename HandleType, std::size_t NumHandles>
  __device__ static void bind_variables_from_sources(
      state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles, auto& source_handles_tuple,
      uint32_t* source_idx) {

    constexpr std::size_t num_sources = std::tuple_size_v<SourcesTuple>;
    using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

    [&]<std::size_t... SourceIndices>(std::index_sequence<SourceIndices...>) {
      (bind_source_variables<SourceIndices, CurrentIteratorOffset>(
           vars, handles, source_handles_tuple, source_idx[SourceIndices]),
       ...);
    }(std::make_index_sequence<num_sources>{});
  }

  // Helper to bind variables from a single source
  template <std::size_t SourceIdx, std::size_t CurrentIteratorOffset, typename ValueType,
            std::size_t NumVars, typename HandleType, std::size_t NumHandles>
  __device__ static void bind_source_variables(state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                               state::HandleStore<HandleType, NumHandles>& handles,
                                               auto& source_handles_tuple, uint32_t row_idx) {

    using SourceType = std::tuple_element_t<SourceIdx, SourcesTuple>;
    auto& source_handle = std::get<SourceIdx>(source_handles_tuple);
    auto source_view = handles.get_view(CurrentIteratorOffset + SourceIdx);
    using VarsList = boost::mp11::mp_rename<VarsTuple, boost::mp11::mp_list>;

    if constexpr (mir::is_scan_v<SourceType>) {
      // Scan source: bind multiple variables from same row
      using ScanVarsTuple = typename SourceType::vars_type;
      constexpr std::size_t num_scan_vars = std::tuple_size_v<ScanVarsTuple>;
      using PrefixVars = typename SourceType::prefix_vars_type;
      constexpr std::size_t prefix_len = std::tuple_size_v<PrefixVars>;

      [&]<std::size_t... ScanVarIndices>(std::index_sequence<ScanVarIndices...>) {
        ((vars.template set<std::tuple_element_t<ScanVarIndices, ScanVarsTuple>>([&]() {
           // TODO: Verify this is correct - using position in index, not original column
           // This may have the same bug that was fixed for ColumnSource
           constexpr std::size_t col_idx = prefix_len + ScanVarIndices;
           return source_view.get_value(col_idx, source_handle.begin() + row_idx);
         }())),
         ...);
      }(std::make_index_sequence<num_scan_vars>{});

    } else if constexpr (mir::is_column_source_v<SourceType>) {
      // ColumnSource: bind multiple variables from row
      using PrefixVars = typename SourceType::prefix_vars_type;
      using NumSourceVars = boost::mp11::mp_size_t<get_source_var_count<SourceType>()>;
      using VarOffset = boost::mp11::mp_size_t<compute_var_offset<SourcesTuple, SourceIdx>()>;
      using PrefixLen = boost::mp11::mp_size_t<std::tuple_size_v<PrefixVars>>;

      [&]<std::size_t... VarIndices>(std::index_sequence<VarIndices...>) {
        ((vars.template set<boost::mp11::mp_at_c<VarsList, VarOffset::value + VarIndices>>([&]() {
           // Column position in INDEX is (prefix_len + var_idx)
           // NOT the original column index from IndexSpec!
           constexpr std::size_t col_idx = PrefixLen::value + VarIndices;
           auto global_row = source_handle.begin() + row_idx;
           return source_view.get_value(col_idx, global_row);
         }())),
         ...);
      }(std::make_index_sequence<NumSourceVars::value>{});
    }
  }

  // Helper to compute and set combined provenance
  template <Semiring SR, std::size_t CurrentIteratorOffset, typename HandleType,
            std::size_t NumHandles, std::size_t NumRelations>
  __device__ static void set_combined_provenance(
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
      state::HandleStore<HandleType, NumHandles>& handles, auto& source_handles_tuple,
      uint32_t* source_idx) {

    constexpr std::size_t num_sources = std::tuple_size_v<SourcesTuple>;

    semiring_value_t<SR> combined_prov =
        std::get<0>(source_handles_tuple)
            .get_provenance_at(handles.get_view(CurrentIteratorOffset + 0), source_idx[0]);

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((combined_prov = sr_mul<SR>(
            combined_prov, std::get<Is + 1>(source_handles_tuple)
                               .get_provenance_at(handles.get_view(CurrentIteratorOffset + Is + 1),
                                                  source_idx[Is + 1]))),
       ...);
    }(std::make_index_sequence<num_sources - 1>{});

    using FirstSource = std::tuple_element_t<0, SourcesTuple>;
    using FirstIndexSpec = typename FirstSource::index_spec_type;
    using FirstSchema = typename FirstIndexSpec::schema_type;
    provs.template set<FirstSchema>(combined_prov);
  }
};

}  // namespace SRDatalog::GPU
