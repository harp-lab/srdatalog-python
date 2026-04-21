-- Test target definitions
local test_targets = {
    {
        name = "test_column",
        source = "test_column.cpp",
        headerfiles = {"column.h", "column.ipp"}
    },
    {
        name = "test_semiring",
        source = "test_simiring.cpp",
        headerfiles = {"semiring.h"}
    },
    {
        name = "test_relation",
        source = "test_relation.cpp",
        headerfiles = {
            "relation_col.h", "relation_col.ipp",
            "column.h", "column.ipp",
            "semiring.h"
        }
    },
    {
        name = "test_sorted_array_index",
        source = "test_sorted_array_index.cpp",
        headerfiles = {
            "sorted_array_index.h", "sorted_array_index.ipp",
            "column.h", "column.ipp",
            "index.h", "semiring.h"
        }
    },
    {
        name = "test_hashmap_index",
        source = "test_hashmap_index.cpp",
        headerfiles = {
            "hashmap.h", "hashmap.ipp",
            "column.h", "column.ipp",
            "index.h", "semiring.h", "system.h"
        }
    },
    {
        name = "test_io",
        source = "test_io.cpp"
    },
    {
        name = "test_highway",
        source = "test_highway.cpp",
        headerfiles = {"sort.h", "system.h"}
    },
    {
        name = "test_gpu_sort",
        source = "test_gpu_sort.cpp",
        headerfiles = {"gpu/sort.h", "gpu/device_array.h", "system.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_cub_sort",
        source = "test_cub_sort.cpp",
        headerfiles = {"gpu/cub_sort.h", "gpu/sort.h", "gpu/device_array.h", "system.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_search",
        source = "test_gpu_search.cpp",
        headerfiles = {"gpu/search.h", "gpu/device_array.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_materialized_join",
        source = "test_materialized_join.cpp",
        headerfiles = {"gpu/runtime/jit/materialized_join.h", "gpu/device_array.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {

        name = "test_custom_set_op",
        source = "test_custom_set_op.cpp",
        headerfiles = {"gpu/runtime/custom_set_op.h", "gpu/device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_semiring_atomic",
        source = "test_gpu_semiring_atomic.cpp",
        headerfiles = {"gpu/device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_device_sorted_array_index",
        source = "test_device_sorted_array_index.cpp",
        headerfiles = {"gpu/device_sorted_array_index.h", "gpu/device_array.h", "gpu/nd_device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_device_sorted_array_index_clone",
        source = "test_device_sorted_array_index_clone.cpp",
        headerfiles = {"gpu/device_sorted_array_index.h", "gpu/device_array.h", "gpu/nd_device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_device_2level_index",
        source = "test_device_2level_index.cpp",
        headerfiles = {"gpu/device_2level_index.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "gpu/nd_device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_runtime",
        source = "test_gpu_runtime.cpp",
        headerfiles = {"gpu/runtime/store.h", "ast.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_pipeline_executor_let_if",
        source = "test_gpu_pipeline_executor_let_if.cpp",
        headerfiles = {"gpu/runtime/**.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_query",
        source = "test_gpu_query.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_device_2level_index_tc",
        source = "test_device_2level_index_tc.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/device_2level_index.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_sg",
        source = "test_gpu_sg.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_sg_fixpoint",
        source = "test_gpu_sg_fixpoint.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h", "query.h", "runtime.h", "test_util.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_append",
        source = "test_gpu_append.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_gpu_tmp",
        source = "test_gpu_tmp.cpp",
        headerfiles = {"gpu/device_array.h", "gpu/device_sorted_array_index.h", "gpu/nd_device_array.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/runtime/runtime.h", "ast.h", "query.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_skip",
        source = "test_skip.cpp",
        headerfiles = {"skip.h", "system.h"}
    },
    {
        name = "test_eytzinger",
        source = "test_eytzinger.cpp",
        headerfiles = {"eytzinger.h", "system.h"}
    },
    {
        name = "test_implicit_btree_index",
        source = "test_implicit_btree_index.cpp",
        headerfiles = {
            "eytzinger.h", "eytzinger.ipp",
            "column.h", "column.ipp",
            "index.h", "semiring.h", "query.h"
        }
    },
    {
        name = "test_gpu_enhanced_id",
        source = "test_gpu_enhanced_id.cpp",
        headerfiles = {"gpu/device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_jit_executor",
        source = "jit/test_jit_executor.cpp",
        headerfiles = {"gpu/device_sorted_array_index.h", "gpu/runtime/output_context.h", "gpu/device_array.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_aos_device_array",
        source = "test_aos_device_array.cpp",
        headerfiles = {"gpu/aos_device_array.h", "gpu/device_array.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    },
    {
        name = "test_tvjoin_basic",
        source = "test_tvjoin_basic.cpp",
        headerfiles = {"gpu/adapter/tvjoin_adapter.h", "gpu/aos_device_array.h", "gpu/device_array.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}, "tbb"},
        gpu_test = true,
        includedirs = {
            "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include",
            "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/include" -- Add target include just in case
        },
        defines = {
            "SRDATALOG_NO_CUCO",
            "__CUDACC__",
            "__CUDACC_VER_MAJOR__=12",
            "__CUDACC_VER_MINOR__=9",
            "__CUDACC_EXTENDED_LAMBDA__"
        }
    },
    {
        name = "test_gpu_tvjoin_index",
        source = "test_gpu_tvjoin_index.cpp",
        headerfiles = {"gpu/device_tvjoin_index.h", "gpu/adapter/tvjoin_adapter.h", "gpu/aos_device_array.h", "gpu/device_array.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}, "tbb"},
        gpu_test = true,
        includedirs = {
            "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include",
            "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/include"
        },
        defines = {
            "SRDATALOG_NO_CUCO",
            "__CUDACC__",
            "__CUDACC_VER_MAJOR__=12",
            "__CUDACC_VER_MINOR__=9",
            "__CUDACC_EXTENDED_LAMBDA__"
        }
    }
}

if has_config("build_test_tmp") then
    table.insert(test_targets, {
        name = "test_tmp",
        source = "test_tmp.cpp",
        headerfiles = {"ast.h", "query.h", "runtime.h", "tmp.h"}
    })
    
    table.insert(test_targets, {
        name = "test_tc",
        source = "test_tc.cpp",
        headerfiles = {"ast.h", "query.h", "runtime.h", "semiring.h"}
    })
    
    table.insert(test_targets, {
        name = "test_tc_sorted_array",
        source = "test_tc_sorted_array.cpp"
    })
    
    table.insert(test_targets, {
        name = "test_sorted_array_merge",
        source = "test_sorted_array_merge.cpp"
    })

    -- DISABLED: C++ MIR compiler has a handle duplication bug that causes incorrect results.
    -- These tests will be re-enabled when negation is implemented in the Nim MIR compiler.
    -- table.insert(test_targets, {
    --     name = "test_gpu_negation",
    --     source = "test_gpu_negation.cpp",
    --     headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h", "query.h", "runtime.h", "test_util.h"},
    --     packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
    --     gpu_test = true
    -- })

    -- table.insert(test_targets, {
    --     name = "test_gpu_negation_native",
    --     source = "test_gpu_negation_native.cpp",
    --     headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h", "query.h", "runtime.h", "test_util.h"},
    --     packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
    --     gpu_test = true
    -- })

    table.insert(test_targets, {
        name = "test_debug",
        source = "test_debug.cpp",
        headerfiles = {"ast.h", "mir.h"},
    })

    table.insert(test_targets, {
        name = "test_mir_printer",
        source = "test_mir_printer.cpp",
        headerfiles = {"mir_printer.h", "ast.h", "mir.h", "gpu/runtime/query.h", "query.h", "runtime.h", "semiring.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    })

    table.insert(test_targets, {
        name = "test_self_join",
        source = "test_self_join.cpp",
        headerfiles = {"ast.h", "query.h", "runtime.h", "semiring.h"}
    })

    table.insert(test_targets, {
        name = "test_dsl_syntax",
        source = "test_dsl_syntax.cpp",
        headerfiles = {"ast.h", "semiring.h", "tmp.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    })

    table.insert(test_targets, {
        name = "test_gpu_incomplete_materialize",
        source = "test_gpu_incomplete_materialize.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "mir_printer.h", "semiring.h", "query.h", "runtime.h", "test_util.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    })

    table.insert(test_targets, {
        name = "test_post_stratum_reconstruct",
        source = "test_post_stratum_reconstruct.cpp",
        headerfiles = {"gpu/runtime/query.h", "gpu/runtime/gpu_fixpoint_executor.h", "gpu/device_sorted_array_index.h", "gpu/device_array.h", "ast.h", "mir.h", "semiring.h", "query.h", "runtime.h"},
        packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
        gpu_test = true
    })
end

create_test_targets(test_targets)


-- test_nim, test_nim_gpu, and test_mir have been removed. 
-- Nim targets are now built/tested via `nimble` using vendored dependencies.
