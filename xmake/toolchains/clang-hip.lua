-- Clang HIP toolchain for xmake
-- Uses clang@hipcc syntax so xmake-repo recognizes it as clang-compatible
-- hipcc handles HIP device code compilation automatically (similar to acpp)

toolchain("clang-hip")
    set_kind("clang")
    
    on_check(function (toolchain)
        -- Check for hipcc compiler
        import("lib.detect.find_tool")
        local hipcc = find_tool("hipcc")
        if hipcc then
            return true
        end
        
        -- Try common ROCm installation paths
        local rocm_paths = {
            "/opt/rocm/bin/hipcc",
            "/usr/bin/hipcc",
        }
        for _, hipcc_path in ipairs(rocm_paths) do
            if os.isfile(hipcc_path) then
                return true
            end
        end
        return false
    end)
    
    on_load(function (toolchain)
        import("lib.detect.find_tool")
        
        -- Find hipcc compiler
        local hipcc = find_tool("hipcc")
        if not hipcc then
            -- Try common installation paths
            local hipcc_paths = {
                "/opt/rocm/bin/hipcc",
                "/usr/bin/hipcc",
            }
            for _, hipcc_path in ipairs(hipcc_paths) do
                if os.isfile(hipcc_path) then
                    hipcc = {program = hipcc_path}
                    break
                end
            end
        end
        
        if not hipcc then
            raise("hipcc compiler not found! Please install ROCm with HIP support.")
        end
        
        print("clang-hip: Using hipcc compiler: " .. (hipcc.program or "hipcc"))
        
        -- Use "clang@hipcc" syntax for C++ compiler so xmake recognizes it as clang-compatible
        -- hipcc handles HIP device code compilation and automatically uses ROCm clang internally
        -- This is similar to how clang-cuda uses "clang@acpp" and acpp uses "clang@acpp"
        toolchain:set("toolset", "cc", "clang")
        toolchain:set("toolset", "cxx", "clang@hipcc")
        toolchain:set("toolset", "ld", "clang++@hipcc")
        toolchain:set("toolset", "sh", "clang++@hipcc")
        print("clang-hip: Using clang@hipcc syntax (hipcc handles HIP device code compilation)")
        print("  C compiler: clang")
        print("  C++ compiler: clang@hipcc")
        
        -- Use lld linker
        toolchain:add("ldflags", "-fuse-ld=lld")
        toolchain:add("shflags", "-fuse-ld=lld")
        
        toolchain:set("toolset", "ar", "ar")
        toolchain:set("toolset", "strip", "strip")
        
        -- Find ROCm installation path
        local rocm_path = nil
        
        -- Priority 1: Use ROCM_PATH environment variable
        rocm_path = os.getenv("ROCM_PATH") or os.getenv("ROCM_HOME")
        
        -- Priority 2: Try common default paths
        if not rocm_path then
            local default_paths = {
                "/opt/rocm",
                "/usr/local/rocm",
            }
            for _, test_path in ipairs(default_paths) do
                if os.isdir(test_path) then
                    rocm_path = test_path
                    break
                end
            end
        end
        
        -- Priority 3: Infer from hipcc location
        if not rocm_path and hipcc.program then
            local hipcc_dir = path.directory(hipcc.program)
            if hipcc_dir:match("/rocm") then
                -- Extract ROCm path (e.g., /opt/rocm/bin/hipcc -> /opt/rocm)
                rocm_path = hipcc_dir:match("(.*/rocm[^/]*)")
            end
        end
        
        if not rocm_path then
            raise("ROCm path not found! Please set ROCM_PATH or install ROCm.")
        end
        
        -- Print detected ROCm path for debugging
        print("clang-hip: Using ROCm path: " .. rocm_path)
        
        local rocm_lib_path = path.join(rocm_path, "lib")
        local rocm_include_path = path.join(rocm_path, "include")
        
        -- Auto-detect GPU architecture if not explicitly set
        local gpu_arch = get_config("hip-arch") or get_config("rocm-arch")
        if not gpu_arch then
            -- Try to auto-detect using rocm_agent_enumerator (preferred method)
            local rocm_agent_enum = path.join(rocm_path, "bin", "rocm_agent_enumerator")
            if os.isfile(rocm_agent_enum) then
                local result = os.iorun(rocm_agent_enum)
                if result and result:match("gfx%d+") then
                    -- Extract first gfx architecture found
                    gpu_arch = result:match("(gfx%d+)")
                    print("clang-hip: Auto-detected GPU architecture: " .. gpu_arch)
                end
            end
            
            -- Fallback: try rocminfo if rocm_agent_enumerator didn't work
            if not gpu_arch then
                local rocminfo_path = path.join(rocm_path, "bin", "rocminfo")
                if os.isfile(rocminfo_path) then
                    local result = os.iorun(rocminfo_path)
                    if result and result:match("gfx%d+") then
                        -- Extract first gfx architecture found
                        gpu_arch = result:match("(gfx%d+)")
                        print("clang-hip: Auto-detected GPU architecture from rocminfo: " .. gpu_arch)
                    end
                end
            end
            
            -- Final fallback: default to gfx942 for MI300
            if not gpu_arch then
                gpu_arch = "gfx942"
                print("clang-hip: Using default GPU architecture: " .. gpu_arch .. " (auto-detection failed)")
            end
        else
            print("clang-hip: Using user-specified GPU architecture: " .. gpu_arch)
        end
        
        -- Add HIP compiler flags
        -- hipcc automatically handles HIP compilation, but we can add architecture flags
        toolchain:add("cxflags", "-std=c++23", {force = true})
        toolchain:add("cxflags", "--offload-arch=" .. gpu_arch)
        
        -- Add ROCm include paths
        toolchain:add("includedirs", 
            rocm_include_path,
            path.join(rocm_include_path, "hip"),
            path.join(rocm_include_path, "rocblas"),
            path.join(rocm_include_path, "rocsparse"),
            path.join(rocm_include_path, "rocthrust"),
            path.join(rocm_include_path, "rocprim"),  -- rocPRIM headers needed for hipCUB rocprim backend
            path.join(rocm_include_path, "hipcub"),
            path.join(rocm_include_path, "hipcub", "backend")  -- For CUB headers (cub/block/block_load.cuh)
        )
        
        -- Add ROCm library paths and libraries
        if os.isdir(rocm_lib_path) then
            toolchain:add("linkdirs", rocm_lib_path)
            -- hip_hcc is deprecated in newer ROCm versions, use amdhip64 only
            toolchain:add("syslinks", "amdhip64", "hsa-runtime64")
            toolchain:add("ldflags", "-L" .. rocm_lib_path)
            toolchain:add("ldflags", "-Wl,-rpath," .. rocm_lib_path)
            
            -- Set LIBRARY_PATH for linker
            local existing_lib_path = os.getenv("LIBRARY_PATH") or ""
            local new_lib_path = rocm_lib_path .. (existing_lib_path ~= "" and (":" .. existing_lib_path) or "")
            toolchain:add("runenvs", "LIBRARY_PATH", new_lib_path)
            toolchain:add("runenvs", "ROCM_PATH", rocm_path)
            toolchain:add("runenvs", "ROCM_HOME", rocm_path)
        end
        
        -- Store ROCm path for later use
        toolchain:set("rocm_path", rocm_path)
        toolchain:set("rocm_lib_path", rocm_lib_path)
        toolchain:set("gpu_arch", gpu_arch)
        
        -- Set LIBRARY_PATH in environment for xmake process
        if os.isdir(rocm_lib_path) then
            local existing_lib_path = os.getenv("LIBRARY_PATH") or ""
            local new_lib_path = rocm_lib_path .. (existing_lib_path ~= "" and (":" .. existing_lib_path) or "")
            os.setenv("LIBRARY_PATH", new_lib_path)
        end
        
        -- Add define to indicate HIP/ROCm build
        toolchain:add("defines", "__HIP_PLATFORM_AMD__")
    end)
    
    -- Package includes for HIP device code are handled by the clang-hip.target rule
    -- (defined in xmake/rules/clang-hip.lua)

toolchain_end()


