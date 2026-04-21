-- xmake.lua - Build configuration matching CMakeLists.txt
-- Column-oriented Datalog core + semiring + relation + small tests
-- Located at project root, builds code in generalized_datalog/

-- Project settings
add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_project("SRDatalog")
set_version("0.1.0")

set_languages("cxx23")

add_repositories("xmake-repo https://github.com/xmake-io/xmake-repo.git")

-- Build options
option("build_test_tmp", {description = "Build test_tmp executable (requires Boost)", default = true})
option("nvidia", {description = "Enable NVIDIA CUDA support (builds cuda_test_clang target)", default = false})
option("rocm", {description = "Enable AMD ROCm/HIP support (builds HIP targets)", default = false})
option("adaptivecpp", {description = "Enable AdaptiveCpp support (SYCL/GPU acceleration)", default = false})
option("enable_logging", {description = "Enable Boost.Log support (can be disabled for zero overhead)", default = false})
option("enable_diagnostics", {description = "Enable diagnostic logging (detailed debug output)", default = false})
option("vtune-itt", {description = "Enable VTune ITT support", default = false})

option("souffle_test", {description = "Enable Souffle integration for testing", default = false})

-- Load toolchains
includes("xmake/toolchains/clang-cuda.lua")
if has_config("rocm") then
    includes("xmake/toolchains/clang-hip.lua")
end

-- Load rules
includes("xmake/rules/mpi.lua")
includes("xmake/rules/clang-cuda.lua")
if has_config("rocm") then
    includes("xmake/rules/clang-hip.lua")
    includes("xmake/rules/rocm-host-compiler.lua")
end
includes("xmake/rules/gpu-test.lua")
if has_config("adaptivecpp") then
    includes("xmake/rules/sycl.lua")
end

-- Load AdaptiveCpp toolchain and package definition only if enabled
if has_config("adaptivecpp") then
    includes("xmake/toolchains/acpp.lua")
    includes("xmake/packages/AdaptiveCpp.lua")
    includes("xmake/packages/CUBThrust.lua")
    add_requires("AdaptiveCpp", {system = false, optional = true})
    add_requires("CUBThrust", {system = false, optional = true})
end

-- When ROCm is enabled, packages need libc++ for C++23 support
local package_cxflags = {}
local package_ldflags = {}
local package_cmake_flags = {}
if has_config("rocm") then
    -- Find libc++ and add flags for packages
    -- Prefer llvm-20 (has C++23 <format> support), then newer versions
    local libcxx_include = nil
    local libcxx_paths = {
        "/usr/lib/llvm-20/include/c++/v1",  -- Prefer llvm-20 (has C++23 format support)
        "/usr/lib/llvm-16/include/c++/v1",
        "/usr/lib/llvm-15/include/c++/v1",
        "/usr/lib/llvm-14/include/c++/v1",
        "/usr/include/c++/v1",
    }
    for _, test_path in ipairs(libcxx_paths) do
        if os.isdir(test_path) and os.isfile(path.join(test_path, "format")) then
            libcxx_include = test_path
            break
        end
    end
    if libcxx_include then
        -- Compiler flags for packages
        table.insert(package_cxflags, "-I" .. libcxx_include)
        table.insert(package_cxflags, "-stdlib=libc++")
        table.insert(package_ldflags, "-lc++")
        table.insert(package_ldflags, "-lc++abi")
        
        -- Add library path for llvm-20
        if libcxx_include:match("llvm-20") then
            table.insert(package_ldflags, "-L/usr/lib/llvm-20/lib")
        end
        
        -- Also add global flags so package checking uses them
        add_includedirs(libcxx_include)
        add_cxflags("-stdlib=libc++")
        add_ldflags("-lc++", "-lc++abi")
        if libcxx_include:match("llvm-20") then
            add_linkdirs("/usr/lib/llvm-20/lib")
        end
        
        -- CMake flags for packages built with CMake
        local cxx_flags = "-I" .. libcxx_include .. " -stdlib=libc++"
        table.insert(package_cmake_flags, "-DCMAKE_CXX_FLAGS=" .. cxx_flags)
        table.insert(package_cmake_flags, "-DCMAKE_CXX_STANDARD=23")
        local linker_flags = "-lc++ -lc++abi"
        if libcxx_include:match("llvm-20") then
            linker_flags = linker_flags .. " -L/usr/lib/llvm-20/lib"
        end
        table.insert(package_cmake_flags, "-DCMAKE_EXE_LINKER_FLAGS=" .. linker_flags)
        table.insert(package_cmake_flags, "-DCMAKE_SHARED_LINKER_FLAGS=" .. linker_flags)
        table.insert(package_cmake_flags, "-DCMAKE_MODULE_LINKER_FLAGS=" .. linker_flags)
        
        -- Note: Global flags added so package checking also uses libc++
    end
end

add_requires("boost", {configs = {all = true}, system = false, 
    build = {cxflags = package_cxflags, ldflags = package_ldflags}})
add_requires("benchmark", {system = false,
    build = {cxflags = package_cxflags, ldflags = package_ldflags, cmake = package_cmake_flags}})
-- Force spdlog to use std::format and be header-only to avoid fmt conflicts
add_requires("spdlog", {configs = {std_format = true, header_only = true},
    build = {cxflags = package_cxflags, ldflags = package_ldflags}})
-- Explicitly require fmt v11 for nvbench compatibility -- REMOVED to fix link order linkage
-- add_requires("fmt", {version = "11.x", system = false})
add_requires("highway", {system = false, configs = {contrib = true},
    build = {cxflags = package_cxflags, ldflags = package_ldflags, cmake = package_cmake_flags}})
add_requires("tbb", {system = false,
    build = {cxflags = package_cxflags, ldflags = package_ldflags}})
-- IPS²Ra is header-only - added manually in target configuration to avoid package validation issues
-- add_requires("ips2ra", {system = false, configs = {optimize_for_native = true}})

-- Load package definitions
-- Note: Umpire package definition kept for reference but no longer required
-- includes("xmake/packages/Umpire.lua")  -- No longer needed - using RMM/hipMM instead
includes("xmake/packages/RMM.lua")
if has_config("rocm") then
    includes("xmake/packages/hipMM.lua")
end
includes("xmake/packages/Cuco.lua")
includes("xmake/packages/Jitify.lua")
includes("xmake/packages/souffle.lua")
includes("xmake/packages/nvbench.lua")

-- includes("xmake/packages/ips2ra.lua")  -- Not needed since we're adding headers manually
-- Add RMM (RAPIDS Memory Manager) for GPU memory management (NVIDIA CUDA)
-- Note: Using --nvidia=y instead of --cuda=y to avoid conflict with xmake's internal CUDA toolchain detection
if has_config("nvidia") then
    -- Use RMM v24.12.01 for CUDA 12.9 compatibility
    add_requires("RMM", {system = false, optional = true, version = "v24.12.01"})
    -- Jitify for CUDA Runtime Compilation (NVRTC)
    add_requires("Jitify", {system = false, optional = true, version = "jitify2"})
    add_requires("Cuco", {system = false, optional = true})
    
    -- NVBench for GPU benchmarking
    -- Package is installed with clang toolchain, targets using clang-cuda access it via after_load
    -- add_requires("nvbench", {system = false, optional = false})
end

-- Add hipMM (HIP Memory Manager) for GPU memory management (AMD ROCm)
-- Note: Using --rocm=y to avoid conflict with xmake namespace
if has_config("rocm") then
    -- Use hipMM v3.0.0 for ROCm compatibility
    add_requires("hipMM", {system = false, optional = true, version = "release/rocmds-25.10"})
end

-- Souffle dependency for testing
if has_config("souffle_test") then
    add_requires("souffle", {optional = true, version = "master"})
end


-- Compiler flags (matching CMakeLists.txt warnings)
-- Compiler flags (matching CMakeLists.txt warnings)
if is_plat("windows") then
    add_cxflags("/W4", "/permissive-")
    -- Enable AVX on Windows (use /arch:AVX2 for AVX2, /arch:AVX for AVX)
    add_cxflags("/arch:AVX2")
else
    add_cxflags("-Wall", "-Wextra", "-Wpedantic")
    if is_plat("macosx") or is_plat("linux") then
        -- Enable all CPU features including AVX/AVX2/AVX-512
        -- Highway library benefits from -march=native for optimal SIMD performance
        add_cxflags("-march=native")
        if is_kind("gcc-13") then
            add_cxflags("-fmax-errors=1")
        elseif is_kind("clang") then
            add_cxflags("-ferror-limit=1")
        end
    end
end

-- Add logging define if enabled
if has_config("enable_logging") then
    add_defines("ENABLE_LOGGING")
end

-- Add souffle_test define if enabled
if has_config("souffle_test") then
    add_defines("SOUFFLE_TEST")
end

-- Add diagnostics define if enabled
if has_config("enable_diagnostics") then
    add_defines("ENABLE_DIAGNOSTICS")
end

-- Enable CCCL's experimental memory_resource API for RMM compatibility
-- RMM v24.12.01 requires cuda::mr namespace from CCCL
if has_config("nvidia") then
    add_defines("LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
    add_defines("USE_CUDA")
    
    -- Fix for Boost.Atomic 1.90+ with Clang on x86
    -- Clang rejects __sync builtins for 16-byte structs, which Boost tries to use
    -- Disable 128-bit atomic support to avoid this error
    add_defines("BOOST_ATOMIC_NO_CMPXCHG16B")
    add_cxflags("-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16")
    
    -- Workaround for spdlog/fmt consteval issue with clang-cuda
    -- Use std::format instead of fmt to avoid consteval issues with fmt::basic_format_string
    -- This avoids the fmt::basic_format_string consteval problem entirely
    -- See: https://forums.developer.nvidia.com/t/spdlog-doesnt-work-with-nvcc-12-6-and-c-20-bug-in-nvcc/332729/3
    add_defines("SPDLOG_USE_STD_FORMAT")
    

end

-- Enable HIP/ROCm support
if has_config("rocm") then
    add_defines("USE_ROCm")
    add_defines("__HIP_PLATFORM_AMD__")
    add_defines("LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
    -- hipMM maintains RMM API compatibility, so same define works
    
    -- Check if system clang-20 is available (preferred - has libc++ with C++23 support)
    -- Note: /usr/bin/clang++-20 is now a wrapper that always adds -stdlib=libc++
    local system_clang20 = "/usr/bin/clang-20"
    local system_clangxx20 = "/usr/bin/clang++-20"
    local use_system_clang20 = os.isfile(system_clangxx20)
    
    -- Check if compiler was already set via command line
    local cc_config = get_config("cc")
    local cxx_config = get_config("cxx")
    
    -- Check if clang-20 toolchain is being used (it will handle compiler setup)
    local toolchain_config = get_config("toolchain")
    
    if use_system_clang20 and (not cc_config or not cxx_config) and toolchain_config ~= "clang-20" then
        -- Use system clang-20 directly (clang++-20 is already a wrapper with -stdlib=libc++)
        set_config("cc", system_clang20)
        set_config("cxx", system_clangxx20)
        
        -- Add libc++ flags for packages (clang++-20 wrapper already adds -stdlib=libc++, but CMake needs explicit flags)
        table.insert(package_cmake_flags, "-DCMAKE_C_COMPILER=" .. system_clang20)
        table.insert(package_cmake_flags, "-DCMAKE_CXX_COMPILER=" .. system_clangxx20)
        table.insert(package_cmake_flags, "-DCMAKE_CXX_FLAGS=-stdlib=libc++ -I/usr/lib/llvm-20/include/c++/v1")
        table.insert(package_cmake_flags, "-DCMAKE_EXE_LINKER_FLAGS=-lc++ -lc++abi -L/usr/lib/llvm-20/lib")
        table.insert(package_cmake_flags, "-DCMAKE_SHARED_LINKER_FLAGS=-lc++ -lc++abi -L/usr/lib/llvm-20/lib")
        table.insert(package_cmake_flags, "-DCMAKE_MODULE_LINKER_FLAGS=-lc++ -lc++abi -L/usr/lib/llvm-20/lib")
        
        print("ROCm: Using system clang-20 (wrapper automatically adds -stdlib=libc++ for C++23 <format> support)")
        print("  C compiler: " .. system_clang20)
        print("  C++ compiler: " .. system_clangxx20)
    elseif cc_config or cxx_config then
        -- User specified compiler via command line, use it (system clang++-20 is already a wrapper)
        print("ROCm: Using user-specified compiler")
        if cc_config then
            print("  C compiler: " .. cc_config)
        end
        if cxx_config then
            print("  C++ compiler: " .. cxx_config)
        end
    else
        -- Fallback: manual libc++ configuration (shouldn't happen if clang-20 is installed)
        local libcxx_include = nil
        local libcxx_paths = {
            "/usr/lib/llvm-20/include/c++/v1",
            "/usr/lib/llvm-16/include/c++/v1",
            "/usr/lib/llvm-15/include/c++/v1",
            "/usr/lib/llvm-14/include/c++/v1",
            "/usr/include/c++/v1",
        }
        for _, test_path in ipairs(libcxx_paths) do
            if os.isdir(test_path) and os.isfile(path.join(test_path, "format")) then
                libcxx_include = test_path
                break
            end
        end
        
        if libcxx_include then
            add_includedirs(libcxx_include)
            add_cxflags("-stdlib=libc++")
            add_ldflags("-lc++", "-lc++abi")
            add_linkdirs("/usr/lib/x86_64-linux-gnu")
            print("ROCm: Using libc++ from " .. libcxx_include .. " for C++23 support (manual config)")
        else
            print("ROCm: Warning - libc++ headers not found, using default libstdc++ (may lack C++23 features)")
        end
    end
end

-- Source code directory
-- Note: Not local so it can be used in included files (xmake_tests.lua, xmake_benchmarks.lua)
code_dir = "generalized_datalog"

-- Include directories (for header files)
add_includedirs(code_dir)
add_includedirs(path.join(code_dir, "build"))
add_includedirs("TVJoin")
add_includedirs(path.join(code_dir, "gpu/runtime"))
add_includedirs(path.join(code_dir, "gpu/runtime/instructions"))
add_includedirs(path.join(code_dir, "gpu/runtime/executor_impl"))
-- Only add CUDA headers for NVIDIA builds (not for ROCm)
if has_config("nvidia") then
    add_includedirs("/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include") -- System CUDA headers for RMM/CCCL
end


rule("run_doxygen")
    on_build(function (target)
        local doxyfile = target:values("doxyfile")
        if not doxyfile or doxyfile == "" then
            doxyfile = path.join(os.projectdir(), "docs", "Doxyfile")
        end
        os.execv("doxygen", {doxyfile})
    end)
    on_clean(function (target)
        local output_dir = target:values("doxygen_output")
        if not output_dir or output_dir == "" then
            output_dir = path.join(os.projectdir(), "docs", "build")
        end
        os.tryrm(output_dir)
        os.tryrm(path.join(os.projectdir(), "docs", "doxygen-warnings.log"))
    end)

rule("generate_shims")
    on_config(function (target)
        -- Ensure code_dir is available (it should be defined before this rule)
        local dir = code_dir or "generalized_datalog"
        local buildir = path.join(os.projectdir(), dir, "build")
        os.mkdir(buildir)
        io.writefile(path.join(buildir, "semiring.hpp"), [[#pragma once
#include "semiring.h"
]])
        io.writefile(path.join(buildir, "mpi_type.h"), [[#pragma once
#include "mpi_types.h"
]])
        -- Add build directory to include path for this target
        target:add("includedirs", buildir)
    end)

-- Include test and benchmark target definitions
-- These files use helper functions to create targets with common configuration
includes("xmake/targets/xmake_test_helpers.lua")
includes("xmake/targets/xmake_tests.lua")
includes("xmake/targets/xmake_benchmarks.lua")

-- Special configuration for test_gpu_sg to include Souffle code
if has_config("souffle_test") then
    target("test_gpu_sg")
        add_files("souffle/sg.cpp")
        add_packages("souffle")
        add_defines("__EMBEDDED_SOUFFLE__")
        add_cxflags("-gdwarf-4")
end

-- Benchmarks phony target (all benchmarks)
target("benchmarks")
    set_kind("phony")
    add_deps("tc_benchmarks", "reach_benchmarks", "index_benchmarks", "andersen_benchmarks")
    if has_config("nvidia") then
        add_deps("tc_lower_device_benchmark", "tc_device_benchmark")
    end
    set_group("benchmarks")
    set_default(false)

-- Target for bootstrapping Nim dependencies
target("nim_vendor")
    set_kind("phony")
    if has_config("nvidia") then
        add_packages("RMM")
    elseif has_config("rocm") then
        add_packages("hipMM")
    end
    add_packages("boost", "highway", "spdlog")
    
    on_install(function (target)
        local vendor_dir = path.join(os.projectdir(), "src/srdatalog/vendor")
        print(" vendoring dependencies to " .. vendor_dir)
        
        -- Clean existing vendor dir to avoid stale files
        os.tryrm(vendor_dir)
        os.mkdir(vendor_dir)
        os.mkdir(path.join(vendor_dir, "include"))
        os.mkdir(path.join(vendor_dir, "lib"))

        -- Track package paths for config.nims generation
        local include_dirs = {}
        local lib_dirs = {}

        for _, pkgname in ipairs(target:get("packages")) do
            local pkg = target:pkg(pkgname)
            if pkg then
                local installdir = pkg:installdir()
                if installdir then
                    print("  copying " .. pkgname .. " from " .. installdir)
                    if os.isdir(path.join(installdir, "include")) then
                        os.cp(path.join(installdir, "include", "*"), path.join(vendor_dir, "include"))
                        table.insert(include_dirs, path.join(installdir, "include"))
                    end
                    if os.isdir(path.join(installdir, "lib")) then
                        os.cp(path.join(installdir, "lib", "*"), path.join(vendor_dir, "lib"))
                        table.insert(lib_dirs, path.join(installdir, "lib"))
                    end
                end
            end
        end
        
        -- Generate config.nims
        print(" generating config.nims...")
        local config_nims = path.join(os.projectdir(), "config.nims")
        local projectdir = os.projectdir()
        
        local content = {}
        table.insert(content, '# Auto-generated by xmake nim_vendor target')
        table.insert(content, '# Regenerate with: xmake install nim_vendor')
        table.insert(content, '')
        table.insert(content, 'import os')
        table.insert(content, '')
        table.insert(content, '# Project root (portable - works on any machine)')
        table.insert(content, 'const projectDir = currentSourcePath().parentDir()')
        table.insert(content, '')
        table.insert(content, 'switch("backend", "cpp")')
        table.insert(content, 'switch("cc", "clang")')
        
        -- Use acpp_wrapper.sh with relative path
        table.insert(content, '')
        table.insert(content, '# Compiler wrapper (relative to project root)')
        table.insert(content, 'switch("clang.exe", projectDir / "acpp_wrapper.sh")')
        table.insert(content, 'switch("clang.linkerexe", projectDir / "acpp_wrapper.sh")')
        table.insert(content, 'switch("clang.cpp.exe", projectDir / "acpp_wrapper.sh")')
        table.insert(content, 'switch("clang.cpp.linkerexe", projectDir / "acpp_wrapper.sh")')
        table.insert(content, 'switch("clangpp.exe", projectDir / "acpp_wrapper.sh")')
        table.insert(content, 'switch("clangpp.linkerexe", projectDir / "acpp_wrapper.sh")')
        
        -- Use relative paths for includes
        table.insert(content, '')
        table.insert(content, '# Include directories (relative to project root)')
        table.insert(content, 'switch("passC", "-I" & (projectDir / "src/srdatalog/vendor/include"))')
        table.insert(content, 'switch("passC", "-I" & (projectDir / "generalized_datalog"))')
        table.insert(content, 'switch("passC", "-std=c++23")')
        table.insert(content, 'switch("passC", "-I" & (projectDir / "generalized_datalog"))')
        table.insert(content, 'switch("passC", "-I" & (projectDir / "generalized_datalog/build"))')
        
        -- CUDA/GPU settings
        if has_config("nvidia") then
            table.insert(content, 'switch("passC", "-x cuda")')
            table.insert(content, 'switch("passC", "-Qunused-arguments")')
            
            -- Detect GPU architecture (default to sm_89 for Ada, but can be changed)
            local cuda_arch = os.getenv("SRDATALOG_CUDA_ARCH") or "sm_89"
            table.insert(content, string.format('switch("passC", "--cuda-gpu-arch=%s")', cuda_arch))
            
            -- Find CUDA path
            local cuda_paths = {
                "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9",
                "/usr/local/cuda",
                "/usr/local/cuda-12",
            }
            local cuda_path = nil
            for _, p in ipairs(cuda_paths) do
                if os.isdir(p) then
                    cuda_path = p
                    break
                end
            end
            if cuda_path then
                table.insert(content, string.format('switch("passC", "--cuda-path=%s")', cuda_path))
                table.insert(content, string.format('switch("passC", "-I%s/include")', cuda_path))
                table.insert(content, string.format('switch("passC", "-I%s/targets/x86_64-linux/include")', cuda_path))
                table.insert(content, string.format('switch("passC", "-I%s/targets/x86_64-linux/include/crt")', cuda_path))
                -- Add math_libs include path for curand headers
                table.insert(content, string.format('switch("passC", "-I%s/../../math_libs/12.9/targets/x86_64-linux/include")', cuda_path))
                table.insert(content, string.format('switch("passL", "-L%s/lib64")', cuda_path))
            end
            
            table.insert(content, 'switch("passC", "-m64")')
            table.insert(content, 'switch("passC", "-DSRDATALOG_GPU_AVAILABLE=1")')
            table.insert(content, 'switch("passC", "-DUSE_CUDA")')
            table.insert(content, 'switch("passC", "-DBOOST_ATOMIC_NO_CMPXCHG16B")')
            table.insert(content, 'switch("passC", "-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16")')
            table.insert(content, 'switch("passC", "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")')
            table.insert(content, 'switch("passC", "-DSPDLOG_USE_STD_FORMAT")')
            table.insert(content, 'switch("passL", "-lcudart")')
        end
        
        if has_config("enable_logging") then
            table.insert(content, 'switch("passC", "-DENABLE_LOGGING")')
        end
        

        
        -- Common defines
        table.insert(content, 'switch("passC", "-DNDEBUG")')
        table.insert(content, 'switch("passC", "-D_GLIBCXX_USE_CXX11_ABI=1")')
        table.insert(content, 'switch("passC", "-O3")')
        table.insert(content, 'switch("passC", "-g")')
        
        
        -- Fix for Nim CUDA compilation (use relative path)
        local fix_header = path.join(projectdir, "src/fix_nim_cuda.h")
        if os.isfile(fix_header) then
            table.insert(content, '')
            table.insert(content, '# Fix for Nim CUDA compilation')
            table.insert(content, 'switch("passC", "-include")')
            table.insert(content, 'switch("passC", projectDir / "src/fix_nim_cuda.h")')
        end
        
        -- Vendor library directory (portable - relative to project root)
        table.insert(content, '')
        table.insert(content, '# Library paths')
        table.insert(content, 'switch("passL", "-L" & (projectDir / "src/srdatalog/vendor/lib"))')
        
        -- Boost libraries
        table.insert(content, 'switch("passL", "-lboost_container")')
        table.insert(content, 'switch("passL", "-lboost_log")')
        table.insert(content, 'switch("passL", "-lboost_log_setup")')
        table.insert(content, 'switch("passL", "-lboost_thread")')
        table.insert(content, 'switch("passL", "-lboost_filesystem")')
        table.insert(content, 'switch("passL", "-lboost_atomic")')
        table.insert(content, 'switch("passL", "-DDISABLE_MIMALLOC=true")')
        -- Note: spdlog is header-only when using std_format, no -lspdlog needed
        
        -- Nimble config
        table.insert(content, '')
        table.insert(content, '# begin Nimble config (version 2)')
        table.insert(content, 'when withDir(thisDir(), system.fileExists("nimble.paths")):')
        table.insert(content, '  include "nimble.paths"')
        table.insert(content, '# end Nimble config')
        
        io.writefile(config_nims, table.concat(content, "\n") .. "\n")
        print("  generated " .. config_nims)
    end)
    set_default(false)

-- Default build target (all C++ tests)
target("tests")
    set_kind("phony")
    add_deps("test_column", "test_semiring", "test_relation", "test_io", "test_highway", "test_sorted_array_index", "test_hashmap_index", "test_eytzinger")
    if has_config("build_test_tmp") then
        add_deps("test_tmp", "test_tc", "test_tc_sorted_array", "test_sorted_array_merge")
        -- test_nim removed as we use nimble for that now
    end
    set_default(true)

-- Doxygen documentation pipeline
target("docs")
    set_kind("phony")
    add_rules("run_doxygen")
    add_packages("doxygen", {optional = true, system = true})
    set_values("doxyfile", path.join(os.projectdir(), "docs", "Doxyfile"))
    set_values("doxygen_output", path.join(os.projectdir(), "docs", "build"))
    set_group("docs")
    set_default(false)
