#pragma once
#include "./tmp_helpers.h"
#include <cstdio>  // For printf
//

#include "../../runtime/state.h"
#include "./output_context.h"
#include "./pipeline_executor.h"
#include "./store.h"

#include "../device_sorted_array_index.h"
#include "../mir.h"
#include "../semiring.h"
#include <boost/mp11.hpp>
#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
#else
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

namespace SRDatalog::GPU {

/**
 * @brief Root-level executor - handles first ColumnJoin and kernel launch
 *
 * @tparam Layout Storage layout for output (SoA or AoS)
 * @tparam OutputArity Number of columns in destination relation
 */
template <typename Pipeline, Semiring SR, typename ValueType, typename HandleType,
          StorageLayout Layout = StorageLayout::SoA, std::size_t OutputArity = 2>
struct GPURootExecutor {
  using MIROps = typename Pipeline::mir_ops_type;
  using VarPosMap = typename Pipeline::var_pos_map_type;

  // Compute NumHandles linearly
  static constexpr std::size_t NumHandles = mir::CountInputIterators<MIROps>::value;

  // First op can be ColumnJoin or CartesianJoin
  using FirstOp = std::tuple_element_t<0, MIROps>;
  static_assert(mir::is_column_join_v<FirstOp> || mir::is_cartesian_join_v<FirstOp> ||
                    mir::is_scan_v<FirstOp>,
                "First MIR op must be ColumnJoin or CartesianJoin or Scan");

  // Check if first op is CartesianJoin or Scan
  static constexpr bool is_cartesian_root = mir::is_cartesian_join_v<FirstOp>;
  static constexpr bool is_scan_root = mir::is_scan_v<FirstOp>;

  // Extract root operation details
  // For ColumnJoin: single variable and sources
  // For CartesianJoin/Scan: tuple of variables and sources
  template <typename Op>
  struct ExtractRootVar {
    using type = typename Op::var_type;
  };
  template <typename VarsTuple, typename SourcesTuple, std::size_t HS, typename VHM>
  struct ExtractRootVar<mir::CartesianJoin<VarsTuple, SourcesTuple, HS, VHM>> {
    using type = void;  // CartesianJoin doesn't have a single var_type
  };
  template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HS,
            typename VHM>
  struct ExtractRootVar<mir::Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>> {
    using type = void;
  };

  template <typename Op>
  struct ExtractRootVars {
    using type = void;
  };
  template <typename VarsTuple, typename SourcesTuple, std::size_t HS, typename VHM>
  struct ExtractRootVars<mir::CartesianJoin<VarsTuple, SourcesTuple, HS, VHM>> {
    using type = VarsTuple;
  };
  template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HS,
            typename VHM>
  struct ExtractRootVars<mir::Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>> {
    using type = VarsTuple;
  };

  // Extract root sources
  template <typename Op>
  struct ExtractRootSources {
    using type = typename Op::sources_type;
  };
  template <typename VarsTuple, typename IndexSpec, typename PrefixVars, std::size_t HS,
            typename VHM>
  struct ExtractRootSources<mir::Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>> {
    using type = std::tuple<mir::Scan<VarsTuple, IndexSpec, PrefixVars, HS, VHM>>;
  };

  using RootVar = typename ExtractRootVar<FirstOp>::type;
  using RootVarsTuple = typename ExtractRootVars<FirstOp>::type;
  using RootSources = typename ExtractRootSources<FirstOp>::type;
  using RemainingOps = SRDatalog::tmp::TupleTail_t<MIROps>;

  // For Scan, we don't have sources like ColumnJoin/CartesianJoin (it's a terminal)
  // But logic below expects num_root_sources.
  // For Scan, num_root_sources is 1 (the scan itself).
  // We need to adjust usages of RootSources.
  static constexpr std::size_t num_root_sources = std::tuple_size_v<RootSources>;
  static constexpr bool root_is_intersection =
      (!is_cartesian_root && !is_scan_root) && (num_root_sources > 1);

  // Compute maps
  // Compute maps
  // OPTIMIZATION: Use pre-computed AllSchemas from Pipeline (O(1)) if available
  // Fallback to extraction for legacy/test pipelines that don't provide it
  using PrecomputedSchemas = typename Pipeline::all_schemas_type;
  using UniqueSchemas =
      std::conditional_t<std::is_same_v<PrecomputedSchemas, std::tuple<>>,
                         boost::mp11::mp_unique<typename mir::ExtractRelationSchemas<MIROps>::type>,
                         PrecomputedSchemas>;
  using RelationPosMap = typename mir::ComputePositionMap<UniqueSchemas>::type;

  using DestSpecs = typename Pipeline::destination_specs_type;  // O(1) lookup
  template <typename Spec>
  struct GetSpecSchema {
    using type = typename Spec::schema_type;
  };
  using DestSchemas = boost::mp11::mp_unique<boost::mp11::mp_transform<GetSpecSchema, DestSpecs>>;
  using FullHandleMap = typename mir::ComputePositionMap<DestSchemas>::type;

  static constexpr std::size_t NumVars = decltype(boost::hana::size(VarPosMap{}))::value;
  static constexpr std::size_t NumRelations = boost::mp11::mp_size<UniqueSchemas>::value;
  static constexpr std::size_t NumFullHandles = boost::mp11::mp_size<DestSchemas>::value;

  // Layout-aware OutputContext types
  using OutputCtxType =
      OutputContext<ValueType, SR, false, Layout, OutputArity, FullHandleMap, HandleType>;
  using OutputCtxCountType =
      OutputContext<ValueType, SR, true, Layout, OutputArity, FullHandleMap, HandleType>;

  using ViewType = typename HandleType::View;

  /**
   * @brief Kernel to count the number of resulting tuples from the root join.
   */
  // Default GroupSize: 32 for CUDA, 64 for ROCm
#ifdef GPU_DEFAULT_GROUP_SIZE
  template <int BlockSize = 256, int GroupSize = GPU_DEFAULT_GROUP_SIZE>
#else
  template <int BlockSize = 256, int GroupSize = 32>
#endif
  static __global__ void __launch_bounds__(BlockSize)
      kernel_count(const HandleType* __restrict__ root_handles,
                   const HandleType* __restrict__ full_handles,
                   const ViewType* __restrict__ root_views, const ViewType* __restrict__ full_views,
                   const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
                   uint32_t num_root_keys, uint32_t* __restrict__ thread_counts) {

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<GroupSize>(block);

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / GroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / GroupSize;

    // DEBUG: Trace kernel launch
    // if (thread_id == 0) {
    //   printf("[GPU] Kernel Count Launch. Grid: %u Block: %u Group: %d NumHandles: %lu\n",
    //   gridDim.x,
    //          blockDim.x, GroupSize, NumHandles);
    // }
    // Initialize stores
    state::VarStore<VarPosMap, ValueType, NumVars> vars;
    state::HandleStore<HandleType, NumHandles> handles;
    state::ProvStore<RelationPosMap, SR, NumRelations> provs;

    // Create ViewStore locally on stack (contains pointers to global memory)
    state::ViewStore<ViewType, NumHandles> view_store(root_views);
    handles.set_view_store(&view_store);

#pragma unroll
    for (std::size_t i = 0; i < NumHandles; ++i) {
      handles[i] = root_handles[i];
    }

    // Create output context
    OutputCtxCountType output_ctx{nullptr, nullptr, 0, 0};
    init_full_handles(output_ctx, full_handles, full_views);

    // Execute root level
    execute_root<GroupSize, true>(tile, warp_id, num_warps, root_unique_values,
                                  num_unique_root_keys, num_root_keys, vars, handles, provs,
                                  output_ctx);

    uint32_t count = output_ctx.count();
    thread_counts[thread_id] = count;
  }

  /**
   * @brief Kernel to materialize the resulting tuples.
   */
  // Default GroupSize: 32 for CUDA, 64 for ROCm
#ifdef GPU_DEFAULT_GROUP_SIZE
  template <int BlockSize = 256, int GroupSize = GPU_DEFAULT_GROUP_SIZE>
#else
  template <int BlockSize = 256, int GroupSize = 32>
#endif
  static __global__ void __launch_bounds__(BlockSize) kernel_materialize(
      const HandleType* __restrict__ root_handles, const HandleType* __restrict__ full_handles,
      const ViewType* __restrict__ root_views, const ViewType* __restrict__ full_views,
      const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
      uint32_t num_root_keys, const uint32_t* __restrict__ thread_offsets,
      ValueType* __restrict__ output_data, semiring_value_t<SR>* __restrict__ output_prov,
      std::size_t output_stride) {

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<GroupSize>(block);

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / GroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / GroupSize;

    // DEBUG: Trace kernel launch
    // if (thread_id == 0) {
    //   printf("[GPU] Kernel Materialize Launch. Grid: %u Block: %u Group: %d NumHandles: %lu\n",
    //          gridDim.x, blockDim.x, GroupSize, NumHandles);
    // }

    uint32_t write_base = thread_offsets[thread_id];

    state::VarStore<VarPosMap, ValueType, NumVars> vars;
    state::HandleStore<HandleType, NumHandles> handles;
    state::ProvStore<RelationPosMap, SR, NumRelations> provs;

    // Create ViewStore locally
    state::ViewStore<ViewType, NumHandles> view_store(root_views);
    handles.set_view_store(&view_store);

#pragma unroll
    for (std::size_t i = 0; i < NumHandles; ++i) {
      handles[i] = root_handles[i];
    }

    OutputCtxType output_ctx{output_data, output_prov, output_stride, write_base};
    init_full_handles(output_ctx, full_handles, full_views);

    execute_root<GroupSize, false>(tile, warp_id, num_warps, root_unique_values,
                                   num_unique_root_keys, num_root_keys, vars, handles, provs,
                                   output_ctx);
  }

 private:
  /**
   * @brief Dispatcher for root execution strategy.
   */
  template <int GroupSize, bool SizeOnly, typename Group, typename OutputCtx>
  __device__ static void execute_root(Group tile, uint32_t warp_id, uint32_t num_warps,
                                      const ValueType* __restrict__ root_unique_values,
                                      uint32_t num_unique_root_keys, uint32_t num_root_keys,
                                      state::VarStore<VarPosMap, ValueType, NumVars>& vars,
                                      state::HandleStore<HandleType, NumHandles>& handles,
                                      state::ProvStore<RelationPosMap, SR, NumRelations>& provs,
                                      OutputCtx& output_ctx) {
    if constexpr (is_cartesian_root) {
      execute_root_cartesian<GroupSize, SizeOnly>(tile, warp_id, num_warps, root_unique_values,
                                                  num_unique_root_keys, num_root_keys, vars,
                                                  handles, provs, output_ctx);
    } else if constexpr (is_scan_root) {
      execute_root_scan<GroupSize, SizeOnly>(tile, warp_id, num_warps, root_unique_values,
                                             num_unique_root_keys, num_root_keys, vars, handles,
                                             provs, output_ctx);
    } else if constexpr (root_is_intersection) {
      execute_root_intersection<GroupSize, SizeOnly>(tile, warp_id, num_warps, root_unique_values,
                                                     num_unique_root_keys, num_root_keys, vars,
                                                     handles, provs, output_ctx);
    } else {
      execute_root_single<GroupSize, SizeOnly>(tile, warp_id, num_warps, root_unique_values,
                                               num_unique_root_keys, num_root_keys, vars, handles,
                                               provs, output_ctx);
    }
  }

  /**
   * @brief Executes root Scan operation - linear scan of index columns.
   */
  template <int GroupSize, bool SizeOnly, typename Group, typename OutputCtx>
  __device__ static void execute_root_scan(
      Group tile, uint32_t warp_id, uint32_t num_warps,
      const ValueType* __restrict__ /*root_unique_values*/, uint32_t /*num_unique_root_keys*/,
      uint32_t /*num_root_keys*/, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx) {

    using IndexSpec = typename FirstOp::index_spec_type;
    using Schema = typename IndexSpec::schema_type;
    constexpr std::size_t num_scan_vars = std::tuple_size_v<RootVarsTuple>;

    // Scan handles 0 (always first iterator)
    auto& handle = handles[0];
    auto view = handles.get_view(0);

    // Calculate total size from handle / view
    // Since this is root, handle should cover full index usually.
    // Or we rely on view.num_rows_ if handle is invalid?
    // Actually handle initialized generally to root().
    if (!handle.valid())
      return;

    // For Scan at root, we iterate over all rows in the index/handle
    // Handle degree gives size of range
    uint32_t total_rows = handle.degree();
    uint32_t start_offset = handle.begin();

    // Parallel linear scan
    uint32_t global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = global_tid; i < total_rows; i += stride) {
      uint32_t row_idx = start_offset + i;

      // Bind variables from columns using unified accessor (supports DSAI and LSM)
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((vars.template set<boost::mp11::mp_at_c<RootVarsTuple, Is>>(view.get_value(Is, row_idx))),
         ...);
      }(std::make_index_sequence<num_scan_vars>{});

      // Set provenance using unified accessor (only if semiring has provenance)
      if constexpr (has_provenance_v<typename Schema::semiring_type>) {
        provs.template set<Schema>(view.get_provenance(row_idx));
      }

      // Execute remaining pipeline
      constexpr std::size_t NextOffset = 1;
      // Just execute the pipeline. Downstream ColumnJoins will handle their own handle prefixing.
      auto single_tile = cg::tiled_partition<1>(tile);
      GPUPipelineExecutor<RemainingOps, VarPosMap, NextOffset, RelationPosMap, 1,
                          true>::template execute<SR>(single_tile, vars, handles, provs,
                                                      output_ctx);
    }
  }

  /**
   * @brief Executes root CartesianJoin - iterates over cartesian product of root values.
   */
  template <int GroupSize, bool SizeOnly, typename Group, typename OutputCtx>
  __device__ static void execute_root_cartesian(
      Group tile, uint32_t warp_id, uint32_t num_warps,
      const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
      uint32_t /* num_root_keys */, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx) {

    constexpr int next_group_size = GroupSize;
    constexpr std::size_t num_vars = std::tuple_size_v<RootVarsTuple>;
    constexpr std::size_t num_sources = num_root_sources;

    uint32_t lane = tile.thread_rank();
    auto single_tile = cg::tiled_partition<1>(tile);

    // Simplified Cartesian Join: Parallelize over first relation, iterate second.
    // Ignores root_unique_values and prefix() navigation (which fails for arity=1 relations).

    // 1. Setup handles
    auto& first_handle = handles[0];
    auto& first_view = handles.get_view(0);

    if (!first_handle.valid())
      return;

    uint32_t first_total = first_handle.degree();
    uint32_t first_begin = first_handle.begin();

    // 2. Parallel loop over first relation
    uint32_t global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = global_tid; i < first_total; i += stride) {
      // Get value from first relation
      ValueType val0 = first_handle.get_value_at(first_view, i);

      // Set First Variable
      using Var0 = boost::mp11::mp_at_c<RootVarsTuple, 0>;
      vars.template set<Var0>(val0);

      // Provenance for first relation
      semiring_value_t<SR> prov0 = first_handle.get_provenance_at(first_view, i);

      // 3. For each row in first, iterate remaining sources
      if constexpr (num_root_sources >= 2) {
        auto& second_handle = handles[1];
        auto& second_view = handles.get_view(1);

        if (second_handle.valid()) {
          uint32_t second_total = second_handle.degree();
          for (uint32_t j = 0; j < second_total; ++j) {
            ValueType val1 = second_handle.get_value_at(second_view, j);

            using Var1 = boost::mp11::mp_at_c<RootVarsTuple, 1>;
            vars.template set<Var1>(val1);

            auto prov1 = second_handle.get_provenance_at(second_view, j);

            // Combine provenance
            using FirstIndexSpec = typename std::tuple_element_t<0, RootSources>::index_spec_type;
            using FirstSchema = typename FirstIndexSpec::schema_type;

            provs.template set<FirstSchema>(sr_mul<SR>(prov0, prov1));

            // 4. Update Handles for dependent ops
            constexpr std::size_t NextOffset = num_root_sources;

            using IndicesToUpdate0 =
                typename detail::GetIndicesToUpdate<RemainingOps, Var0,
                                                    boost::mp11::mp_size_t<NextOffset>>::type;
            using IndicesToUpdate1 =
                typename detail::GetIndicesToUpdate<RemainingOps, Var1,
                                                    boost::mp11::mp_size_t<NextOffset>>::type;
            using IndicesToUpdate =
                boost::mp11::mp_unique<boost::mp11::mp_append<IndicesToUpdate0, IndicesToUpdate1>>;
            constexpr bool has_updates = boost::mp11::mp_size<IndicesToUpdate>::value > 0;

            auto saved = [&]() {
              if constexpr (has_updates)
                return SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);
              else
                return std::tuple<>{};
            }();

            if constexpr (has_updates) {
              boost::mp11::mp_for_each<IndicesToUpdate0>([&](auto I) {
                constexpr std::size_t idx = decltype(I)::value;
                handles[idx] = coordinates_prefix_helper(handles[idx], val0, single_tile,
                                                         handles.get_view(idx));
              });
              boost::mp11::mp_for_each<IndicesToUpdate1>([&](auto I) {
                constexpr std::size_t idx = decltype(I)::value;
                handles[idx] = coordinates_prefix_helper(handles[idx], val1, single_tile,
                                                         handles.get_view(idx));
              });
            }

            provs.reset();
            GPUPipelineExecutor<RemainingOps, VarPosMap, NextOffset, RelationPosMap, 1,
                                true>::template execute<SR>(single_tile, vars, handles, provs,
                                                            output_ctx);

            if constexpr (has_updates) {
              SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved);
            }
          }
        }
      } else {
        // Single source case actions
        using FirstIndexSpec = typename std::tuple_element_t<0, RootSources>::index_spec_type;
        using FirstSchema = typename FirstIndexSpec::schema_type;
        provs.template set<FirstSchema>(prov0);

        constexpr std::size_t NextOffset = num_root_sources;

        using IndicesToUpdate =
            typename detail::GetIndicesToUpdate<RemainingOps, Var0,
                                                boost::mp11::mp_size_t<NextOffset>>::type;
        constexpr bool has_updates = boost::mp11::mp_size<IndicesToUpdate>::value > 0;

        auto saved = [&]() {
          if constexpr (has_updates)
            return SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);
          else
            return std::tuple<>{};
        }();

        if constexpr (has_updates) {
          boost::mp11::mp_for_each<IndicesToUpdate>([&](auto I) {
            constexpr std::size_t idx = decltype(I)::value;
            handles[idx] =
                coordinates_prefix_helper(handles[idx], val0, single_tile, handles.get_view(idx));
          });
        }

        provs.reset();
        GPUPipelineExecutor<RemainingOps, VarPosMap, NextOffset, RelationPosMap, 1,
                            true>::template execute<SR>(single_tile, vars, handles, provs,
                                                        output_ctx);

        if constexpr (has_updates) {
          SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved);
        }
      }
    }
  }

  /**
   * @brief Executes root join with specialized intersection logic.
   */
  template <int GroupSize, bool SizeOnly, typename Group, typename OutputCtx>
  __device__ static void execute_root_intersection(
      Group tile, uint32_t warp_id, uint32_t num_warps,
      const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
      uint32_t /* num_root_keys */, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx) {

    uint32_t lane = tile.thread_rank();

    using DependentIndices =
        typename detail::GetIndicesToUpdate<RemainingOps, RootVar,
                                            boost::mp11::mp_size_t<num_root_sources>>::type;
    using RootIndices = boost::mp11::mp_iota_c<num_root_sources>;
    using IndicesToUpdate =
        boost::mp11::mp_unique<boost::mp11::mp_append<RootIndices, DependentIndices>>;
    constexpr bool has_indices_to_update = (boost::mp11::mp_size<IndicesToUpdate>::value > 0);

    auto saved_root_handles = [&]() {
      if constexpr (has_indices_to_update) {
        return SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);
      } else {
        return std::tuple<>{};
      }
    }();

    for (uint32_t y_idx = warp_id; y_idx < num_unique_root_keys; y_idx += num_warps) {
      if constexpr (has_indices_to_update) {
        SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_root_handles);
      }
      ValueType root_val;
      if (lane == 0) {
        root_val = root_unique_values[y_idx];
      }
      root_val = tile.shfl(root_val, 0);

      auto source_handles = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(handles[Is].prefix(root_val, tile, handles.get_view(Is))...);
      }(std::make_index_sequence<num_root_sources>{});

      bool all_valid = true;
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((all_valid = all_valid && std::get<Is>(source_handles).valid()), ...);
      }(std::make_index_sequence<num_root_sources>{});

      if (!all_valid)
        continue;

      vars.template set<RootVar>(root_val);

      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        auto update_handle = [&](std::size_t idx, auto&& val) { handles[idx] = val; };
        (update_handle(Is, std::get<Is>(source_handles)), ...);
      }(std::make_index_sequence<num_root_sources>{});

#ifdef GPU_DEFAULT_GROUP_SIZE
      constexpr unsigned int NextGroupSize = GPU_DEFAULT_GROUP_SIZE;
#else
      constexpr unsigned int NextGroupSize = 32;
#endif
      static_assert(GroupSize % NextGroupSize == 0, "GroupSize must be divisible by NextGroupSize");
      constexpr std::size_t NextOffset = num_root_sources;

      if constexpr (NextGroupSize < GroupSize) {
        auto sub_tile = cg::tiled_partition<NextGroupSize>(tile);
        update_handles_after_root(handles, root_val, sub_tile);
        provs.reset();
        GPUPipelineExecutor<
            RemainingOps, VarPosMap, NextOffset, RelationPosMap, NextGroupSize,
            !mir::HasMoreColumnJoins<RemainingOps>::value>::template execute<SR>(sub_tile, vars,
                                                                                 handles, provs,
                                                                                 output_ctx);
      } else {
        update_handles_after_root(handles, root_val, tile);
        provs.reset();
        GPUPipelineExecutor<
            RemainingOps, VarPosMap, NextOffset, RelationPosMap, NextGroupSize,
            !mir::HasMoreColumnJoins<RemainingOps>::value>::template execute<SR>(tile, vars,
                                                                                 handles, provs,
                                                                                 output_ctx);
      }
    }
  }

  /**
   * @brief Executes root single source join.
   */
  template <int GroupSize, bool SizeOnly, typename Group, typename OutputCtx>
  __device__ static void execute_root_single(
      Group tile, uint32_t warp_id, uint32_t num_warps,
      const ValueType* __restrict__ root_unique_values, uint32_t num_unique_root_keys,
      uint32_t /* num_root_keys */, state::VarStore<VarPosMap, ValueType, NumVars>& vars,
      state::HandleStore<HandleType, NumHandles>& handles,
      state::ProvStore<RelationPosMap, SR, NumRelations>& provs, OutputCtx& output_ctx) {

    auto& root_handle = handles[0];
    if (!root_handle.valid())
      return;

    uint32_t lane = tile.thread_rank();

    for (uint32_t y_idx = warp_id; y_idx < num_unique_root_keys; y_idx += num_warps) {
      ValueType root_val;
      if (lane == 0) {
        root_val = root_unique_values[y_idx];
      }
      root_val = tile.shfl(root_val, 0);

#ifdef GPU_DEFAULT_GROUP_SIZE
      constexpr unsigned int NextGroupSize = GPU_DEFAULT_GROUP_SIZE;
#else
      constexpr unsigned int NextGroupSize = 32;
#endif
      static_assert(GroupSize % NextGroupSize == 0, "GroupSize must be divisible by NextGroupSize");
      constexpr std::size_t NextOffset = num_root_sources;

      using DependentIndices =
          typename detail::GetIndicesToUpdate<RemainingOps, RootVar,
                                              boost::mp11::mp_size_t<NextOffset>>::type;
      using RootIndices = boost::mp11::mp_iota_c<1>;
      using IndicesToUpdate =
          boost::mp11::mp_unique<boost::mp11::mp_append<RootIndices, DependentIndices>>;
      constexpr bool has_indices_to_update = (boost::mp11::mp_size<IndicesToUpdate>::value > 0);

      auto saved_handles = [&]() {
        if constexpr (has_indices_to_update) {
          return SRDatalog::state::save_handles_to_tuple<IndicesToUpdate>(handles);
        } else {
          return std::tuple<>{};
        }
      }();

      auto navigated_handle = root_handle.prefix(root_val, tile, handles.get_view(0));
      if (!navigated_handle.valid()) {
        continue;
      }

      handles[0] = navigated_handle;
      vars.template set<RootVar>(root_val);

      if constexpr (NextGroupSize < GroupSize) {
        auto sub_tile = cg::tiled_partition<NextGroupSize>(tile);
        update_handles_after_root(handles, root_val, sub_tile);
        provs.reset();
        GPUPipelineExecutor<
            RemainingOps, VarPosMap, NextOffset, RelationPosMap, NextGroupSize,
            !mir::HasMoreColumnJoins<RemainingOps>::value>::template execute<SR>(sub_tile, vars,
                                                                                 handles, provs,
                                                                                 output_ctx);
      } else {
        update_handles_after_root(handles, root_val, tile);
        provs.reset();
        GPUPipelineExecutor<
            RemainingOps, VarPosMap, NextOffset, RelationPosMap, NextGroupSize,
            !mir::HasMoreColumnJoins<RemainingOps>::value>::template execute<SR>(tile, vars,
                                                                                 handles, provs,
                                                                                 output_ctx);
      }

      if constexpr (has_indices_to_update) {
        SRDatalog::state::restore_handles_from_tuple<IndicesToUpdate>(handles, saved_handles);
      }
    }
  }

  __device__ static auto get_root_source_handles(
      state::HandleStore<HandleType, NumHandles>& handles) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(handles[Is]...);
    }(std::make_index_sequence<num_root_sources>{});
  }

  template <typename Group>
  __device__ static void update_handles_after_root(
      state::HandleStore<HandleType, NumHandles>& handles, ValueType root_val, Group tile) {
    constexpr std::size_t Offset = num_root_sources;

    // Get handles that need prefixing from:
    // 1. The RemainingOps (downstream operations that have RootVar in their VarToHandlesMap)
    // 2. The FirstOp's VarToHandlesMap (the root ColumnJoin's own handle dependencies)
    using DownstreamIndices =
        typename detail::GetIndicesToUpdate<RemainingOps, RootVar,
                                            boost::mp11::mp_size_t<Offset>>::type;

    // Also get indices from FirstOp's VarToHandlesMap (uses SFINAE to handle void VarToHandlesMap)
    using FirstOpIndices =
        typename detail::GetIndicesToUpdateSimple<std::tuple<FirstOp>, RootVar>::type;

    // Combine and deduplicate
    using IndicesToUpdate =
        boost::mp11::mp_unique<boost::mp11::mp_append<DownstreamIndices, FirstOpIndices>>;

    boost::mp11::mp_for_each<IndicesToUpdate>([&](auto I) {
      constexpr std::size_t idx = decltype(I)::value;
      auto& handle = handles[idx];
      handle = handle.prefix(root_val, tile, handles.get_view(idx));
    });
  }

  template <typename Group>
  __device__ static void update_handles_after_root_cartesian(
      state::HandleStore<HandleType, NumHandles>& /*handles*/, Group /*tile*/) {}

  template <typename OutputCtx>
  __device__ static void init_full_handles(OutputCtx& ctx, const HandleType* full_handles,
                                           const ViewType* full_views) {
    if constexpr (NumFullHandles > 0) {
      init_full_handles_impl<0>(ctx, full_handles, full_views);
    }
  }

  template <std::size_t Idx, typename OutputCtx>
  __device__ static void init_full_handles_impl(OutputCtx& ctx, const HandleType* full_handles,
                                                const ViewType* full_views) {
    if constexpr (Idx < NumFullHandles) {
      using Schema = std::tuple_element_t<Idx, DestSchemas>;
      auto handle = full_handles[Idx];
      // Pass the view directly
      ctx.template set_full_handle<Schema>(handle, full_views[Idx]);
      init_full_handles_impl<Idx + 1>(ctx, full_handles, full_views);
    }
  }

  // helper to allow coordinates_prefix_helper usage if available or direct prefix
  template <typename H, typename T, typename G, typename V>
  __device__ static auto coordinates_prefix_helper(H&& h, T&& val, G&& tile, V&& view) {
    // If coordinates_prefix_helper was needed, it should be defined or we use prefix.
    // In original code (Step 105 view), we didn't see coordinates_prefix_helper.
    // We saw `handles[idx].prefix`.
    // My manual merge above used `coordinates_prefix_helper` in one spot (CartesianJoin).
    // I should fix that to use `.prefix` directly to be safe, matching other spots.
    return h.prefix(std::forward<T>(val), std::forward<G>(tile), std::forward<V>(view));
  }
};

}  // namespace SRDatalog::GPU
