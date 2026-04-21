#pragma once

#include "../gpu_fixpoint_executor_common.h"
#include "../pipeline_executor.h"

namespace SRDatalog::GPU {

/**
 * @brief Specialization for ExecutePipeline: Launches a GPU pipeline.
 */
template <typename Pipeline, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::ExecutePipeline<Pipeline>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    using MIROps = typename Pipeline::mir_ops_type;
    using VarPosMap = typename Pipeline::var_pos_map_type;

    using FirstOp = std::tuple_element_t<0, MIROps>;
    using GlobalIndexSpecs = typename Pipeline::source_specs_type;

    constexpr std::size_t kNumHandles = std::tuple_size_v<GlobalIndexSpecs>;

    if constexpr (kNumHandles == 0) {
      return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
    } else {
      // Push NVTX range with rule name for profiling
      nvtxRangePushA(Pipeline::get_rule_name());
      using FirstSpec = std::tuple_element_t<0, GlobalIndexSpecs>;
      using FirstSchema = typename FirstSpec::schema_type;
      using ValueType = typename FirstSchema::intern_value_type;
      // Derive index type from the database's actual relation type (avoids circular include)
      using FirstRelationType =
          std::remove_cvref_t<decltype(get_relation_by_schema<FirstSchema, FULL_VER>(
              std::declval<DB&>()))>;
      using IndexType = typename FirstRelationType::IndexTypeInst;
      using HandleType = typename IndexType::NodeHandle;
      using ConcreteHandleType = HandleType;
      using ViewType = typename HandleType::View;

      using SR = typename FirstSchema::semiring_type;
      using Launcher = GPUPipelineLauncher<Pipeline, SR, ValueType, HandleType>;

      std::vector<ConcreteHandleType> root_handles_vec;
      root_handles_vec.reserve(kNumHandles);
      std::vector<ViewType> root_views_vec;
      root_views_vec.reserve(kNumHandles);

      auto fill_handles = [&]<typename... Specs>(std::tuple<Specs...>) {
        (..., [&](auto SpecType) {
          using Spec = decltype(SpecType);
          using Schema = typename Spec::schema_type;
          using ColSeq = typename Spec::column_indexes_type;
          constexpr std::size_t ver = Spec::kVersion;

          auto& rel = [&]() -> auto& {
            if constexpr (ver == DELTA_VER) {
              if (iteration == 0) {
                return get_relation_by_schema<Schema, FULL_VER>(db);
              }
            }
            return get_relation_by_schema<Schema, ver>(db);
          }();

          auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
            return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
          }(ColSeq{});

          // DONT CHANGE THIS
          rel.ensure_index(runtime_spec, false);
          auto& idx = rel.get_index(runtime_spec);
          root_handles_vec.push_back(idx.root());
          root_views_vec.push_back(idx.view());
        }(Specs{}));
      };
      fill_handles(GlobalIndexSpecs{});

      using DestSpecs = typename Pipeline::destination_specs_type;
      using UniqueDestSpecsList = mp::mp_unique<DestSpecs>;
      using UniqueDestSpecs = mp::mp_rename<UniqueDestSpecsList, std::tuple>;

      std::vector<ConcreteHandleType> full_handles_vec;
      std::vector<ViewType> full_views_vec;

      auto fill_dest_handles = [&]<typename... Specs>(std::tuple<Specs...>) {
        (..., [&](auto SpecType) {
          using Spec = decltype(SpecType);
          using Schema = typename Spec::schema_type;
          using ColSeq = typename Spec::column_indexes_type;
          constexpr std::size_t ver = FULL_VER;

          auto& rel = get_relation_by_schema<Schema, ver>(db);

          auto runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
            return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
          }(ColSeq{});

          auto& idx = rel.ensure_index(runtime_spec, false);
          std::size_t full_index_size = idx.root().degree();

          full_handles_vec.push_back(idx.root());
          full_views_vec.push_back(idx.view());
        }(Specs{}));
      };
      fill_dest_handles(UniqueDestSpecs{});

      DeviceArray<ConcreteHandleType> d_root_handles(root_handles_vec.size());
      DeviceArray<ConcreteHandleType> d_full_handles(
          full_handles_vec.size() > 0 ? full_handles_vec.size() : 1);

      // Ensure device is ready before copy
      GPU_DEVICE_SYNCHRONIZE();

      GPU_MEMCPY(d_root_handles.data(), root_handles_vec.data(),
                 root_handles_vec.size() * sizeof(ConcreteHandleType), GPU_HOST_TO_DEVICE);

      // Sync after copy and check for errors
      GPU_DEVICE_SYNCHRONIZE();
      GPU_ERROR_T err = GPU_GET_LAST_ERROR();
      if (err != GPU_SUCCESS) {
        std::cerr << "[ERROR] GPU_MEMCPY failed: " << err << std::endl;
      }

      if (!full_handles_vec.empty()) {
        GPU_MEMCPY(d_full_handles.data(), full_handles_vec.data(),
                   full_handles_vec.size() * sizeof(ConcreteHandleType), GPU_HOST_TO_DEVICE);
      }

      DeviceArray<ViewType> d_root_views(root_views_vec.size());
      DeviceArray<ViewType> d_full_views(full_views_vec.size() > 0 ? full_views_vec.size() : 1);

      GPU_MEMCPY(d_root_views.data(), root_views_vec.data(),
                 root_views_vec.size() * sizeof(ViewType), GPU_HOST_TO_DEVICE);
      if (!full_views_vec.empty()) {
        GPU_MEMCPY(d_full_views.data(), full_views_vec.data(),
                   full_views_vec.size() * sizeof(ViewType), GPU_HOST_TO_DEVICE);
      }
      GPU_DEVICE_SYNCHRONIZE();

      using DestRelOp = std::tuple_element_t<std::tuple_size_v<MIROps> - 1, MIROps>;
      using OutSchema = typename DestRelOp::schema_type;
      auto& out_rel = get_relation_by_schema<OutSchema, NEW_VER>(db);

      uint32_t num_root_keys = root_handles_vec[0].degree();

      using FirstSourceSpec = std::tuple_element_t<0, GlobalIndexSpecs>;
      using FirstColSeq = typename FirstSourceSpec::column_indexes_type;
      constexpr std::size_t first_ver = FirstSourceSpec::kVersion;
      auto first_runtime_spec = []<typename T, T... Cols>(std::integer_sequence<T, Cols...>) {
        return SRDatalog::IndexSpec{{static_cast<int>(Cols)...}};
      }(FirstColSeq{});

      auto& first_rel = [&]() -> auto& {
        if constexpr (first_ver == DELTA_VER) {
          if (iteration == 0) {
            return get_relation_by_schema<FirstSchema, FULL_VER>(db);
          }
        }
        return get_relation_by_schema<FirstSchema, first_ver>(db);
      }();

      auto& first_idx = first_rel.get_index(first_runtime_spec);
      uint32_t num_unique_root_keys = static_cast<uint32_t>(first_idx.num_unique_root_values());

      const ValueType* root_unique_values_ptr =
          (num_unique_root_keys > 0) ? first_idx.root_unique_values().data() : nullptr;

      using RootExec = GPURootExecutor<Pipeline, SR, ValueType, ConcreteHandleType>;

      DeviceArray<uint32_t> join_hist;
      int num_sms = 0;
      GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);
      uint32_t num_blocks = static_cast<uint32_t>(num_sms) * 4;
      constexpr uint32_t kResultBlockSize = 256;
      join_hist.resize((static_cast<size_t>(num_blocks) * kResultBlockSize) + 1);

      if (iteration == 0) {
        GPU_EVENT_T start_count, stop_count, start_materialize, stop_materialize;
        GPU_EVENT_CREATE(&start_count);
        GPU_EVENT_CREATE(&stop_count);
        GPU_EVENT_CREATE(&start_materialize);
        GPU_EVENT_CREATE(&stop_materialize);

        nvtxRangePushA("Count_Phase");
        GPU_EVENT_RECORD(start_count);

        RootExec::template kernel_count<256, GPU_DEFAULT_GROUP_SIZE>
            <<<num_blocks, kResultBlockSize>>>(d_root_handles.data(), d_full_handles.data(),
                                               d_root_views.data(), d_full_views.data(),
                                               root_unique_values_ptr, num_unique_root_keys,
                                               num_root_keys, join_hist.data());
        GPU_ERROR_T launch_err = GPU_GET_LAST_ERROR();
        if (launch_err != GPU_SUCCESS) {
          std::cerr << "ERROR: kernel_count launch failed in iteration " << iteration << ": "
                    << GPU_GET_ERROR_STRING(launch_err) << std::endl;
        }
        GPU_EVENT_RECORD(stop_count);
        GPU_EVENT_SYNCHRONIZE(stop_count);
        GPU_ERROR_T sync_err = GPU_GET_LAST_ERROR();
        if (sync_err != GPU_SUCCESS) {
          std::cerr << "ERROR: kernel_count execution failed in iteration " << iteration << ": "
                    << GPU_GET_ERROR_STRING(sync_err) << std::endl;
        }
        nvtxRangePop();
        float count_time_ms = 0;
        GPU_EVENT_ELAPSED_TIME(&count_time_ms, start_count, stop_count);

        auto old_size = static_cast<uint32_t>(out_rel.size());

        nvtxRangePushA("Exclusive_Scan");
        thrust::exclusive_scan(rmm::exec_policy{}, join_hist.begin(), join_hist.end(),
                               join_hist.begin(), old_size, thrust::plus<uint32_t>());
        nvtxRangePop();
        uint32_t total_count = join_hist[static_cast<size_t>(num_blocks) * kResultBlockSize];

        if (total_count > old_size) {
          nvtxRangePushA("Memory_Resize");
          out_rel.resize_interned_columns(total_count);
          if constexpr (has_provenance_v<typename OutSchema::semiring_type>) {
            out_rel.provenance().resize(total_count);
          }
          nvtxRangePop();

          nvtxRangePushA("Materialize_Phase");
          GPU_EVENT_RECORD(start_materialize);
          using ProvPtrType = semiring_value_t<typename OutSchema::semiring_type>*;
          ProvPtrType prov_ptr = nullptr;
          if constexpr (has_provenance_v<typename OutSchema::semiring_type>) {
            prov_ptr = out_rel.provenance().data();
          }
          RootExec::template kernel_materialize<256, GPU_DEFAULT_GROUP_SIZE>
              <<<num_blocks, kResultBlockSize>>>(
                  d_root_handles.data(), d_full_handles.data(), d_root_views.data(),
                  d_full_views.data(), root_unique_values_ptr, num_unique_root_keys, num_root_keys,
                  join_hist.data(), out_rel.template interned_column<0>(), prov_ptr,
                  out_rel.interned_stride());
          GPU_ERROR_T materialize_launch_err = GPU_GET_LAST_ERROR();
          if (materialize_launch_err != GPU_SUCCESS) {
            std::cerr << "ERROR: kernel_materialize launch failed in iteration " << iteration
                      << ": " << GPU_GET_ERROR_STRING(materialize_launch_err) << std::endl;
          }
          GPU_EVENT_RECORD(stop_materialize);
          GPU_EVENT_SYNCHRONIZE(stop_materialize);
          GPU_ERROR_T materialize_sync_err = GPU_GET_LAST_ERROR();
          if (materialize_sync_err != GPU_SUCCESS) {
            std::cerr << "ERROR: kernel_materialize execution failed in iteration " << iteration
                      << ": " << GPU_GET_ERROR_STRING(materialize_sync_err) << std::endl;
          }
          nvtxRangePop();
          float materialize_time_ms = 0;
          GPU_EVENT_ELAPSED_TIME(&materialize_time_ms, start_materialize, stop_materialize);
        }

        GPU_EVENT_DESTROY(start_count);
        GPU_EVENT_DESTROY(stop_count);
        GPU_EVENT_DESTROY(start_materialize);
        GPU_EVENT_DESTROY(stop_materialize);
      } else {
        nvtxRangePushA("Count_Phase");
        RootExec::template kernel_count<256, GPU_DEFAULT_GROUP_SIZE>
            <<<num_blocks, kResultBlockSize>>>(d_root_handles.data(), d_full_handles.data(),
                                               d_root_views.data(), d_full_views.data(),
                                               root_unique_values_ptr, num_unique_root_keys,
                                               num_root_keys, join_hist.data());
        GPU_ERROR_T launch_err = GPU_GET_LAST_ERROR();
        if (launch_err != GPU_SUCCESS) {
          std::cerr << "ERROR: kernel_count launch failed in iteration " << iteration << ": "
                    << GPU_GET_ERROR_STRING(launch_err) << std::endl;
        }
        GPU_DEVICE_SYNCHRONIZE();
        GPU_ERROR_T sync_err = GPU_GET_LAST_ERROR();
        if (sync_err != GPU_SUCCESS) {
          std::cerr << "ERROR: kernel_count execution failed in iteration " << iteration << ": "
                    << GPU_GET_ERROR_STRING(sync_err) << std::endl;
        }
        nvtxRangePop();

        uint32_t old_size = static_cast<uint32_t>(out_rel.size());

        nvtxRangePushA("Exclusive_Scan");
        thrust::exclusive_scan(rmm::exec_policy{}, join_hist.begin(), join_hist.end(),
                               join_hist.begin(), old_size, thrust::plus<uint32_t>());
        nvtxRangePop();
        uint32_t total_count = join_hist[static_cast<size_t>(num_blocks) * kResultBlockSize];

        if (total_count > old_size) {
          nvtxRangePushA("Memory_Resize");
          out_rel.resize_interned_columns(total_count);
          if constexpr (has_provenance_v<typename OutSchema::semiring_type>) {
            out_rel.provenance().resize(total_count);
          }
          nvtxRangePop();

          nvtxRangePushA("Materialize_Phase");
          using ProvPtrType2 = semiring_value_t<typename OutSchema::semiring_type>*;
          ProvPtrType2 prov_ptr2 = nullptr;
          if constexpr (has_provenance_v<typename OutSchema::semiring_type>) {
            prov_ptr2 = out_rel.provenance().data();
          }
          RootExec::template kernel_materialize<256, GPU_DEFAULT_GROUP_SIZE>
              <<<num_blocks, kResultBlockSize>>>(
                  d_root_handles.data(), d_full_handles.data(), d_root_views.data(),
                  d_full_views.data(), root_unique_values_ptr, num_unique_root_keys, num_root_keys,
                  join_hist.data(), out_rel.template interned_column<0>(), prov_ptr2,
                  out_rel.interned_stride());
          GPU_ERROR_T materialize_launch_err = GPU_GET_LAST_ERROR();
          if (materialize_launch_err != GPU_SUCCESS) {
            std::cerr << "ERROR: kernel_materialize launch failed in iteration " << iteration
                      << ": " << GPU_GET_ERROR_STRING(materialize_launch_err) << std::endl;
          }
          GPU_DEVICE_SYNCHRONIZE();
          GPU_ERROR_T materialize_sync_err = GPU_GET_LAST_ERROR();
          if (materialize_sync_err != GPU_SUCCESS) {
            std::cerr << "ERROR: kernel_materialize execution failed in iteration " << iteration
                      << ": " << GPU_GET_ERROR_STRING(materialize_sync_err) << std::endl;
          }
          nvtxRangePop();
        }
      }

      nvtxRangePop();  // Rule name (Pipeline::rule_name)

      auto result =
          GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
      GPU_DEVICE_SYNCHRONIZE();
      return result;
    }
  }
};

}  // namespace SRDatalog::GPU
