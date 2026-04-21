package("nvbench")
    set_homepage("https://github.com/NVIDIA/nvbench")
    set_description("CUDA Kernel Benchmarking Library")
    set_license("Apache-2.0")

    add_urls("https://github.com/NVIDIA/nvbench.git")
    add_versions("main", "main")
    
    add_deps("cmake")
    add_deps("ninja")
    -- Use external fmt dependency in header-only mode
    add_deps("fmt", {configs = {header_only = true}}) 

    -- Don't force re-installation checks - nvbench is header-only and works with any toolchain
    -- set_policy("package.install_always", true)

    -- Ensure package metadata is set when package is loaded (after installation)
    on_load(function (package)
        local install_dir = package:installdir()
        if install_dir and os.isdir(install_dir) then
            local include_dir = path.join(install_dir, "include")
            local lib_dir = path.join(install_dir, "lib")
            
            if os.isdir(include_dir) then
                package:set("includedirs", {include_dir})
                -- Also set cuflags so CUDA compiler can find headers
                -- Force flag ensures it's applied even if toolchain differs
                package:add("cuflags", "-I" .. include_dir, {force = true})
            end
            
            if os.isdir(lib_dir) then
                package:set("linkdirs", {lib_dir})
                package:add("links", "nvbench")
                
                -- Add CUDA driver API
                package:add("syslinks", "cuda")
                
                -- Add NVML (NVIDIA Management Library) - required by nvbench
                -- Use NVHPC CUDA library path (same as used by clang-cuda toolchain)
                local cuda_path = os.getenv("ACPP_CUDA_PATH") or 
                                  os.getenv("CUDA_PATH") or 
                                  os.getenv("CUDA_HOME") or
                                  "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
                local nvml_stubs_path = path.join(cuda_path, "targets", "x86_64-linux", "lib", "stubs")
                if os.isdir(nvml_stubs_path) then
                    -- NVML is in the stubs directory in NVHPC
                    package:add("linkdirs", nvml_stubs_path)
                    package:add("ldflags", "-L" .. nvml_stubs_path, {force = true})
                    package:add("ldflags", "-lnvidia-ml", {force = true})
                else
                    -- Fallback: try main lib directory or syslink
                    local nvml_lib_path = path.join(cuda_path, "targets", "x86_64-linux", "lib")
                    if os.isdir(nvml_lib_path) then
                        package:add("linkdirs", nvml_lib_path)
                        package:add("ldflags", "-L" .. nvml_lib_path, {force = true})
                    end
                    package:add("syslinks", "nvidia-ml")
                end
                
                -- FORCE CUPTI linking using ldflags (bypassing xmake check which seems to fail)
                local cupti_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/extras/CUPTI/lib64"
                if os.isdir(cupti_lib) then
                    print("NVBench: Found CUPTI lib at " .. cupti_lib)
                    package:add("linkdirs", cupti_lib)
                    package:add("ldflags", "-L" .. cupti_lib)
                    package:add("ldflags", "-lcupti", "-lnvperf_host", "-lnvperf_target")
                else
                    print("NVBench WARNING: CUPTI lib not found at " .. cupti_lib)
                end
            end
            
            -- Force consumers to see fmt as header-only
            package:add("defines", "FMT_HEADER_ONLY")
            
            -- Debug include dir
            print("NVBench: Exporting include_dir: " .. include_dir)
        end
    end)

    on_install("linux", function (package)
        local configs = {}
        table.insert(configs, "-DNVBENCH_ENABLE_EXAMPLE=OFF")
        table.insert(configs, "-DNVBENCH_ENABLE_TESTING=OFF")
        table.insert(configs, "-DBUILD_SHARED_LIBS=OFF") 
        table.insert(configs, "-DCMAKE_BUILD_TYPE=Release")
        
        -- Force internal build to use header-only fmt
        table.insert(configs, "-DCMAKE_CXX_FLAGS=-DFMT_HEADER_ONLY")
        table.insert(configs, "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DFMT_HEADER_ONLY")
        
        -- Pass explicit FMT_INCLUDE_DIR for our manual target injection
        local fmt = package:dep("fmt")
        local fmt_include_dir = ""
        if fmt then
             fmt_include_dir = fmt:installdir("include")
             -- Use forward slashes for cmake compatibility
             fmt_include_dir = fmt_include_dir:gsub("\\", "/")
        end
        
        local cxx = package:build_getenv("CXX")
        if not cxx then cxx = package:tool("cxx") end
        if cxx then
             table.insert(configs, "-DCMAKE_CUDA_HOST_COMPILER=" .. cxx)
             table.insert(configs, "-DCMAKE_CXX_COMPILER=" .. cxx)
        end
        
        -- Patch NVBenchDependencies.cmake to manually inject fmt target
        local dep_file = "cmake/NVBenchDependencies.cmake"
        if os.isfile(dep_file) then
            print("NVBench: Patching " .. dep_file .. " to inject manual fmt target...")
            local content = io.readfile(dep_file)
            
            local start_marker = "rapids_cpm_find%(fmt 11%.2%.0"
            local end_marker = "CMAKE_POSITION_INDEPENDENT_CODE ON\"\n%)"
            
            -- Constructed manual target block
            local injection = ""
            if fmt_include_dir ~= "" then
                injection = string.format([[
if(NOT TARGET fmt::fmt)
  add_library(fmt::fmt INTERFACE IMPORTED)
  target_include_directories(fmt::fmt INTERFACE "%s")
  target_compile_definitions(fmt::fmt INTERFACE FMT_HEADER_ONLY=1)
  set(fmt_ADDED TRUE) 
endif()
# Replaced rapids_cpm_find with manual target injection
]], fmt_include_dir)
            else
                print("NVBench warning: FMT_INCLUDE_DIR is empty! Patch might fail at build time.")
                injection = "message(FATAL_ERROR \"FMT_INCLUDE_DIR not found explicitly.\")"
            end

            local s, e = content:find(start_marker)
            if s then
                 local _, end_idx = content:find(end_marker, s)
                 if end_idx then
                      local pre = content:sub(1, s-1)
                      local post = content:sub(end_idx+1)
                      content = pre .. injection .. post
                      io.writefile(dep_file, content)
                      print("NVBench: Patch applied successfully.")
                 else
                      print("NVBench warning: Could not find end of fmt block in " .. dep_file)
                 end
            else
                 -- Fallback regex replacement
                 content = content:gsub("rapids_cpm_find%(fmt.-%)", injection)
                 io.writefile(dep_file, content)
                 print("NVBench: Patch applied via regex fallback.")
            end
        else
            print("NVBench warning: " .. dep_file .. " not found. Cannot patch.")
        end

        import("package.tools.cmake").install(package, configs)
        
        -- Remove pkgconfig and cmake config to prevent xmake from picking up bad paths
        -- We rely strictly on manual on_load settings
        local lib_dir = package:installdir("lib")
        if lib_dir then
            os.rm(path.join(lib_dir, "pkgconfig"))
            os.rm(path.join(lib_dir, "cmake"))
        end
    end)

    on_test(function (package)
    end)
package_end()
