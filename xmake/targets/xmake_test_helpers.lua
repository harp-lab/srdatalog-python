-- xmake_test_helpers.lua
-- Helper functions to reduce duplication in test and benchmark target definitions
-- Note: code_dir must be defined in xmake.lua before this file is included
-- MPI support is available via add_rules("mpi") rule defined in xmake.lua

-- Store code_dir locally when this file is included
-- This ensures it's available when functions are called later
local _code_dir = code_dir or "generalized_datalog"
if not _code_dir or _code_dir == "" then
    error("code_dir is not set. It must be defined in xmake.lua before including xmake_test_helpers.lua")
end

-- Common packages used by tests and benchmarks
local test_packages = {
    {"boost", {configs = {all = true}}},
    "mimalloc",
    "highway"
}

local benchmark_packages = {
    "benchmark",
    {"boost", {configs = {all = true}}},
    "mimalloc",
    "highway"
}

local function add_packages_to_target(packages)
    for _, pkg in ipairs(packages) do
        if type(pkg) == "table" and #pkg >= 2 then
            add_packages(pkg[1], pkg[2])
        elseif type(pkg) == "string" then
            add_packages(pkg)
        end
    end
end

local function configure_common_settings(target)
    if type(import) == "function" then
        local find_tool = import("lib.detect.find_tool")
        if find_tool then
            local lld = find_tool("ld.lld")
            if lld then
                target:add("ldflags", "-fuse-ld=lld", {force = true})
            end
        end
    end
    target:add("cxflags", "-D_GLIBCXX_USE_CXX11_ABI=1", {force = true})
    -- Fix: Ensure hwy is linked after hwy_contrib
    target:add("ldflags", "-lhwy", {force = true})
end

function create_test_target(name, source_file, options)
    options = options or {}
    local headerfiles = options.headerfiles or {}
    local extra_packages = options.packages or {}
    local extra_config = options.config or function(target) end
    local is_gpu_test = options.gpu_test or false
    local is_mpi_test = options.mpi_test or false
    local extra_cxflags = options.cxflags or {}
    local extra_defines = options.defines or {}
    local extra_includedirs = options.includedirs or {}
    
    if not source_file then
        error("source_file cannot be nil for target: " .. tostring(name))
    end
    
    target(name)
        set_kind("binary")
        add_files(path.join(os.projectdir(), _code_dir, "test", source_file))
        add_packages_to_target(test_packages)
        -- Only add extra packages (like RMM/hipMM) if GPU is configured for GPU tests
        -- This is a safety check in case create_test_target is called directly
        -- (GPU tests should be skipped at create_test_targets level when GPU is not configured)
        if not is_gpu_test or has_config("nvidia") or has_config("rocm") then
            add_packages_to_target(extra_packages)
        end
        

        add_rules("generate_shims")
        set_group("tests")
        add_tests("default")
        
        if is_gpu_test then
            if has_config("nvidia") then
                set_toolchains("clang-cuda")
                add_rules("clang-cuda.target", "gpu.test")
            elseif has_config("rocm") then
                set_toolchains("clang-hip")
                add_rules("clang-hip.target", "gpu.test")
            end
        end
        
        if is_mpi_test then
            add_rules("mpi")
        end
        
        if #extra_cxflags > 0 then
            add_cxflags(extra_cxflags, {force = true})
        end
        if #extra_defines > 0 then
            add_defines(extra_defines)
        end
        if #extra_includedirs > 0 then
            add_includedirs(extra_includedirs)
        end
        
        on_config(function(target)
            configure_common_settings(target)
        end)
        
        on_load(function(tgt)
            extra_config(tgt)
        end)
end

function create_benchmark_target(name, source_file, options)
    options = options or {}
    local headerfiles = options.headerfiles or {}
    local extra_packages = options.packages or {}
    local extra_config = options.config or function(target) end
    local extra_after_load = options.after_load or function(target) end
    local profiling = options.profiling or false
    
    if not source_file then
        error("source_file cannot be nil for target: " .. tostring(name))
    end
    
    target(name)
        set_kind("binary")
        if has_config("nvidia") then
            set_toolchains("clang-cuda")
            add_rules("clang-cuda.target")

            add_linkdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
            add_rpathdirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/targets/x86_64-linux/lib")
            -- Add RMM for NVIDIA GPU benchmarks (needed for device_array.h and GPU memory management)
            add_packages("RMM", {system = false, optional = false, version = "v24.12.01"})
        elseif has_config("rocm") then
            set_toolchains("clang-hip")
            add_rules("clang-hip.target")
            -- Add hipMM for ROCm GPU benchmarks (needed for device_array.h and GPU memory management)
            add_packages("hipMM", {system = false, optional = false, version = "release/rocmds-25.10"})
        end
        add_files(path.join(os.projectdir(), _code_dir, "benchmark", source_file))
        add_packages_to_target(benchmark_packages)
        add_packages_to_target(extra_packages)
        
        if headerfiles and #headerfiles > 0 then
            for _, hdr in ipairs(headerfiles) do
                if hdr then
                    add_headerfiles(path.join(_code_dir, hdr))
                end
            end
        end
        
        add_rules("generate_shims")
        set_group("benchmarks")
        
        on_config(function(target)
            configure_common_settings(target)
            if profiling and is_mode("releasedbg") then
                target:set("symbols", "debug")
                target:set("strip", "none")
                target:add("cxflags", "-g", "-O3")
            end
        end)
        
        extra_config(target)
        
        -- Support after_load hook for accessing package dependencies
        after_load(function(target)
            extra_after_load(target)
        end)
end

function create_test_targets(test_list)
    for _, test_info in ipairs(test_list) do
        if not test_info.name then
            error("Test definition missing 'name' field")
        end
        if not test_info.source then
            error("Test definition missing 'source' field for: " .. tostring(test_info.name))
        end
        -- Skip GPU tests if neither nvidia nor rocm is configured
        -- This ensures CI builds only compile CPU tests when GPU support is unavailable
        if test_info.gpu_test and not has_config("nvidia") and not has_config("rocm") then
            -- Skip GPU test target creation - prevents compilation errors from missing GPU dependencies
            -- (e.g., RMM/hipMM headers, CUDA/HIP toolchain, etc.)
        else
            -- Swap RMM for hipMM when ROCm is enabled
            local packages = test_info.packages or {}
            if has_config("rocm") then
                packages = {}
                for _, pkg in ipairs(test_info.packages or {}) do
                    if type(pkg) == "table" and pkg[1] == "RMM" then
                        -- Replace RMM with hipMM for ROCm (use same version as benchmarks)
                        table.insert(packages, {"hipMM", pkg[2] or {system = false, optional = false, version = "release/rocmds-25.10"}})
                    else
                        table.insert(packages, pkg)
                    end
                end
            end
            
            create_test_target(
                test_info.name,
                test_info.source,
                {
                    headerfiles = test_info.headerfiles or {},
                    packages = packages,
                    config = test_info.config or function() end,
                    gpu_test = test_info.gpu_test or false,
                    mpi_test = test_info.mpi_test or false,
                    cxflags = test_info.cxflags,
                    defines = test_info.defines,
                    includedirs = test_info.includedirs
                }
            )
        end
    end
end

function create_benchmark_targets(benchmark_list)
    for _, bench_info in ipairs(benchmark_list) do
        if not bench_info.name then
            error("Benchmark definition missing 'name' field")
        end
        if not bench_info.source then
            error("Benchmark definition missing 'source' field for: " .. tostring(bench_info.name))
        end
        -- Swap RMM for hipMM when ROCm is enabled (similar to create_test_targets)
        local packages = bench_info.packages or {}
        if has_config("rocm") then
            packages = {}
            for _, pkg in ipairs(bench_info.packages or {}) do
                if type(pkg) == "table" and pkg[1] == "RMM" then
                    -- Replace RMM with hipMM for ROCm
                    table.insert(packages, {"hipMM", pkg[2] or {system = false, optional = false, version = "release/rocmds-25.10"}})
                else
                    table.insert(packages, pkg)
                end
            end
        end
        create_benchmark_target(
            bench_info.name,
            bench_info.source,
            {
                headerfiles = bench_info.headerfiles or {},
                packages = packages,
                config = bench_info.config or function() end,
                profiling = bench_info.profiling or false
            }
        )
    end
end
