-- Benchmark target definitions
local benchmark_targets = {
    {
        name = "tc_benchmarks",
        source = "tc_benchmark.cpp",
        config = function(target)
            -- code_dir is defined in xmake.lua and should be accessible here
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    },
    {
        name = "reach_benchmarks",
        source = "reach_benchmark.cpp",
        config = function(target)
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    },
    {
        name = "andersen_benchmarks",
        source = "andersen_benchmark.cpp",
        config = function(target)
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    },
    {
        name = "index_benchmarks",
        source = "index_benchmark.cpp",
        profiling = true,
        config = function(target)
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    },
    {
        name = "join_benchmark",
        source = "join_benchmark.cpp",
        config = function(target)
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            
            if has_config("nvidia") then
               set_toolchains("clang-cuda")
               add_rules("clang-cuda.target")
               add_packages("RMM")
               add_defines("SR_ENABLE_GPU")
            elseif has_config("rocm") then
               set_toolchains("clang-hip")
               add_rules("clang-hip.target")
               add_packages("hipMM")
               add_defines("SR_ENABLE_GPU")
            end
        end
    }
}

-- GPU-only benchmark(s)
-- Keep these behind --nvidia=y or --rocm=y so CPU-only builds don't break on GPU/thrust usage.
if has_config("nvidia") or has_config("rocm") then
    local gpu_rmm_package = has_config("nvidia") and "RMM" or "hipMM"
    
    -- Search seek benchmark (binary vs exponential search on VarPointsTo data)
    table.insert(benchmark_targets, {
        name = "search_seek_benchmark",
        source = "bench_search_seek.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}, "benchmark"},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3", "-fno-omit-frame-pointer")
            end
            
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    })
    table.insert(benchmark_targets, {
        name = "tc_lower_device_benchmark",
        source = "tc_lower_device_benchmark.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}},
        config = function(target)
            -- only set if use releasedbg mode
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3", "-fno-omit-frame-pointer")
            end
            
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            
            -- Add VTune ITT support if enabled
            -- Enable with: xmake build tc_lower_device_benchmark -m releasedbg --vtune-itt=true
            if has_config("vtune-itt") then
                local vtune_path = "/opt/intel/oneapi/vtune/latest"
                local sdk_path = path.join(vtune_path, "sdk")
                local lib_path = path.join(sdk_path, "lib64")
                if os.isdir(sdk_path) then
                    add_defines("ENABLE_VTUNE_ITT")
                    add_includedirs(path.join(sdk_path, "include"))
                    add_linkdirs(lib_path)
                    -- Link against static library (libittnotify.a)
                    add_links("ittnotify", {force = true})
                else
                    print("Warning: VTune SDK not found at " .. sdk_path)
                    print("  Install VTune or set correct path")
                end
            end
        end
    })
    table.insert(benchmark_targets, {
        name = "tc_lower_tvjoin_benchmark",
        source = "tc_lower_tvjoin_benchmark.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}, "tbb"},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3", "-fno-omit-frame-pointer")
            end
            
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            
            -- Ensure GPU features are enabled
            add_defines("SR_ENABLE_GPU")
            add_defines("SRDATALOG_NO_CUCO")
        end
    })
    table.insert(benchmark_targets, {
        name = "tc_device_benchmark",
        source = "tc_device_benchmark.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    })
    table.insert(benchmark_targets, {
        name = "runtime_tc_benchmark",
        source = "runtime_tc_instantiation.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    })
    table.insert(benchmark_targets, {
        name = "runtime_tc_tvjoin_benchmark",
        source = "runtime_tc_tvjoin_instantiation.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}, "tbb"},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            add_defines("SR_ENABLE_GPU")
            add_defines("SRDATALOG_NO_CUCO")
        end
    })
    table.insert(benchmark_targets, {
        name = "sg_device_benchmark",
        source = "sg_device_benchmark.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
                if has_config("nvidia") then

                    add_linkdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
                    add_rpathdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
                end
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
        end
    })
    table.insert(benchmark_targets, {
        name = "andersen_device_benchmark",
        source = "andersen_device_benchmark.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
                if has_config("nvidia") then

                    add_linkdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
                    add_rpathdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
                end
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            
            -- Note: clang-cuda toolchain is set by create_benchmark_target helper when nvidia config is enabled
            add_defines("SR_ENABLE_GPU")
        end,
        -- Ensure CUDA compiler gets nvbench include path (package's cuflags may not propagate)
        -- after_load = function(target)
        --     local nvbench_pkg = target:dep("nvbench")
        --     if nvbench_pkg then
        --         local nvbench_include = nvbench_pkg:installdir("include")
        --         if nvbench_include and os.isdir(nvbench_include) then
        --             print("andersen_device_benchmark: Adding nvbench cuflags: -I" .. nvbench_include)
        --             target:add("cuflags", "-I" .. nvbench_include, {force = true})
        --         else
        --             print("andersen_device_benchmark: nvbench include dir not found: " .. tostring(nvbench_include))
        --         end
        --     else
        --         print("andersen_device_benchmark: nvbench package dependency not found")
        --     end
        -- end
    })
    -- table.insert(benchmark_targets, {
    --     name = "merge_overhead_benchmark",
    --     source = "merge_overhead_benchmark.cpp",
    --     profiling = true,
    --     packages = {{"RMM", {system = false, optional = false, version = "v24.12.01"}}},
    --     config = function(target)
    --         if is_mode("releasedbg") then
    --             set_symbols("debug")
    --             set_strip("none")
    --             add_cxflags("-g", "-O3")
    --         end
    --         if code_dir then
    --             add_includedirs(code_dir)
    --         else
    --             add_includedirs("generalized_datalog")
    --         end
    --     end
    --     end
    -- })

    table.insert(benchmark_targets, {
        name = "tvjoin_vs_lsm_benchmark",
        source = "benchmark_tvjoin_vs_lsm.cpp",
        profiling = true,
        packages = {{gpu_rmm_package, {system = false, optional = false, version = "v24.12.01"}}, "benchmark", "tbb"},
        config = function(target)
            if is_mode("releasedbg") then
                set_symbols("debug")
                set_strip("none")
                add_cxflags("-g", "-O3")
                if has_config("nvidia") then
                     add_includedirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include")
                end
            end
            if code_dir then
                add_includedirs(code_dir)
            else
                add_includedirs("generalized_datalog")
            end
            -- Add TVJoin includes explicitly if not covered by code_dir
            -- TVJoin is at project root usually? "TVJoin" folder.
            -- xmake.lua adds "TVJoin" to global includes.
            -- But we also need to ensure we link properly.
            add_defines("SRDATALOG_NO_CUCO")
        end
    })
end

create_benchmark_targets(benchmark_targets)

if has_config("nvidia") then
    target("cuda_test_clang")
        set_kind("binary")
        set_group("benchmarks")
        set_toolchains("clang-cuda")
        add_rules("clang-cuda.target")
        add_files(path.join(os.projectdir(), code_dir, "benchmark/cuda_test.cpp"))
        add_includedirs(code_dir, path.join(code_dir, "build"))
        add_packages("boost", {configs = {all = true}, system = false})
        add_packages("mimalloc")
        add_packages("benchmark", {system = false})
        add_packages("RMM", {system = false, optional = false, version = "v24.12.01"})
        add_packages("highway", {configs = {contrib = true}, system = false})
        add_packages("tbb", {system = false})
        add_syslinks("pthread", "atomic")
        -- Fix: Ensure hwy is linked AFTER hwy_contrib
        add_ldflags("-lhwy", {force = true})
        after_load(function (target)
            local ips2ra_cache = path.join(os.getenv("HOME") or os.tmpdir(), ".xmake", "cache", "ips2ra")
            local ips2ra_source = path.join(ips2ra_cache, "ips2ra-source")
            local ips2ra_include = path.join(ips2ra_source, "include")
            
            if not os.isdir(ips2ra_include) then
                print("Cloning IPS²Ra repository for cuda_test_clang...")
                os.mkdir(ips2ra_cache)
                os.vrunv("git", {"clone", "--depth", "1", "--branch", "master",
                                "--recurse-submodules", "https://github.com/ips4o/ips2ra.git", ips2ra_source})
            end
            
            if os.isdir(ips2ra_include) then
                target:add("includedirs", ips2ra_include)
                target:add("cxflags", "-I" .. ips2ra_include, {force = true})
                print("cuda_test_clang: Added IPS²Ra include directory: " .. ips2ra_include)
            else
                print("Warning: IPS²Ra include directory not found: " .. ips2ra_include)
            end
        end)
end
