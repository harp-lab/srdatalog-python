-- Clang CUDA toolchain for xmake
-- Uses clang++ (or acpp clang wrapper) with -x cuda flag for CUDA compilation
-- Prefers AdaptiveCpp (acpp) if available, as it's a clang wrapper with CUDA support

toolchain("clang-cuda")
    set_kind("clang")
    
    on_check(function (toolchain)
        -- Check if AdaptiveCpp is configured
        if has_config("adaptivecpp") then
            -- If AdaptiveCpp is configured, check for acpp
            import("lib.detect.find_tool")
            local acpp = find_tool("acpp")
            if acpp then
                return true
            end
            
            -- Try common acpp installation paths
            local acpp_paths = {"/opt/sycl/bin/acpp"}
            for _, acpp_path in ipairs(acpp_paths) do
                if os.isfile(acpp_path) then
                    return true
                end
            end
            return false
        else
            -- If AdaptiveCpp is not configured, check for clang++
            import("lib.detect.find_tool")
            local clang = find_tool("clang++")
            if clang then
                return true
            end
            return false
        end
    end)
    
    on_load(function (toolchain)
        -- Dispatch based on AdaptiveCpp configuration
        import("lib.detect.find_tool")
        local compiler = nil
        local compiler_name = nil
        local use_acpp = false
        
        if has_config("adaptivecpp") then
            -- AdaptiveCpp is configured, use acpp
            local acpp = find_tool("acpp")
            if not acpp then
                -- Try common acpp installation paths
                local acpp_paths = {"/opt/sycl/bin/acpp"}
                for _, acpp_path in ipairs(acpp_paths) do
                    if os.isfile(acpp_path) then
                        acpp = {program = acpp_path}
                        break
                    end
                end
            end
            
            if not acpp then
                raise("AdaptiveCpp is configured (--adaptivecpp=y) but acpp compiler not found! Please install AdaptiveCpp.")
            end
            
            compiler = acpp
            compiler_name = "acpp"
            use_acpp = true
            print("clang-cuda: Using AdaptiveCpp (acpp) compiler: " .. acpp.program)
        else
            -- AdaptiveCpp is not configured, use normal clang++
            compiler = find_tool("clang++")
            if not compiler then
                raise("clang++ not found! Please install clang with CUDA support, or enable AdaptiveCpp with --adaptivecpp=y")
            end
            compiler_name = "clang++"
            print("clang-cuda: Using clang++ compiler: " .. (compiler.program or "clang++"))
        end
        
        -- Set compiler toolsets
        -- Use "clang@acpp" syntax for acpp so xmake-repo recognizes it as clang-compatible
        if use_acpp then
            toolchain:set("toolset", "cc", "clang")
            toolchain:set("toolset", "cxx", "clang@acpp")
            toolchain:set("toolset", "ld", "clang++@acpp")
            toolchain:set("toolset", "sh", "clang++@acpp")
        else
            toolchain:set("toolset", "cc", "clang")
            toolchain:set("toolset", "cxx", "clang++")
            toolchain:set("toolset", "ld", "clang++")
            toolchain:set("toolset", "sh", "clang++")
        end
        toolchain:set("toolset", "ar", "ar")
        toolchain:set("toolset", "strip", "strip")
         
        local cuda_path = nil
        
        -- Priority 1: Use NVHPC environment variables if set
        -- These are typically set by NVHPC module system
        local nvcompilers = os.getenv("NVCOMPILERS")
        local nvarch_env = os.getenv("NVARCH")
        local nvhpc_ver = os.getenv("NVHPC_VER")
        
        -- Ensure nvarch is a string (default if not set or invalid)
        local nvarch = "Linux_x86_64"  -- Default architecture
        if nvarch_env and type(nvarch_env) == "string" and nvarch_env ~= "" then
            nvarch = nvarch_env
        end
        
        -- Only use NVHPC env vars if nvcompilers and nvhpc_ver are valid strings
        if nvcompilers and type(nvcompilers) == "string" and nvcompilers ~= "" and
           nvhpc_ver and type(nvhpc_ver) == "string" and nvhpc_ver ~= "" then
            -- Build path from NVHPC env vars: $NVCOMPILERS/$NVARCH/$NVHPC_VER/cuda/12.9
            -- Prefer 12.9 for clang-20 compatibility (clang-20 doesn't work with CUDA 13)
            local nvhpc_cuda_12_9 = path.join(nvcompilers, nvarch, nvhpc_ver, "cuda", "12.9")
            
            if os.isdir(nvhpc_cuda_12_9) then
                cuda_path = nvhpc_cuda_12_9
            end
        end
        
        -- Priority 2: Use standard CUDA environment variables
        if not cuda_path then
            cuda_path = os.getenv("CUDA_PATH") or 
                       os.getenv("CUDA_HOME") or 
                       os.getenv("ACPP_CUDA_PATH")
            
            -- If env var is set, check if it points to 13.0 and fall back to 12.9
            if cuda_path then
                -- Ensure we have the full CUDA path with version
                if cuda_path:match("/cuda$") and not cuda_path:match("/cuda/%d") then
                    if os.isdir(cuda_path .. "/12.9") then
                        cuda_path = cuda_path .. "/12.9"
                    end
                end
                
            end
        end
        
        -- Priority 3: Try common default paths (prefer 12.9 over 13.0)
        if not cuda_path then
            -- Try common NVHPC SDK paths
            local default_paths = {
                "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9",
            }
            
            for _, test_path in ipairs(default_paths) do
                if os.isdir(test_path) then
                    -- Prefer 12.9 over 13.0
                    if test_path:match("/12.9") then
                        cuda_path = test_path
                        break
                    end
                end
            end
        end
        
        if not cuda_path then
            raise("CUDA path not found! Please set NVHPC_VER, CUDA_PATH, or install NVIDIA HPC SDK with CUDA 12.9")
        end
        
        -- Print detected CUDA path for debugging
        print("clang-cuda: Using CUDA path: " .. cuda_path)
        
        local cuda_lib_path = path.join(cuda_path, "targets", "x86_64-linux", "lib")
        
        -- Auto-detect GPU architecture if not explicitly set
        local gpu_arch = get_config("clang-cuda-arch") or get_config("cuda-arch")
        if not gpu_arch then
            -- Try to auto-detect using nvidia-smi
            local nvidia_smi = find_tool("nvidia-smi")
            if nvidia_smi then
                local result = os.iorun(nvidia_smi.program .. " --query-gpu=compute_cap --format=csv,noheader")
                if result and result:match("%d+%.%d+") then
                    -- Extract compute capability (e.g., "8.9" -> "sm_89")
                    local major, minor = result:match("(%d+)%.(%d+)")
                    if major and minor then
                        gpu_arch = "sm_" .. major .. minor
                        print("clang-cuda: Auto-detected GPU architecture: " .. gpu_arch .. " (compute capability " .. major .. "." .. minor .. ")")
                    end
                end
            end
        else
            print("clang-cuda: Using user-specified GPU architecture: " .. gpu_arch)
        end
        
        -- Add CUDA compiler flags
        -- -x cuda tells clang to treat input as CUDA code
        -- --cuda-gpu-arch specifies architecture
        toolchain:add("cxflags", "-x", "cuda", {force = true})
        -- use c++23 standard
        toolchain:add("cxflags", "-std=c++23", {force = true})
        
        toolchain:add("cxflags", "--cuda-gpu-arch=" .. gpu_arch)
        toolchain:add("cxflags", "--cuda-path=" .. cuda_path, {force = true})
        
        -- Add CUDA include paths
        local target_include = path.join(cuda_path, "targets/x86_64-linux/include")
        local cccl_include = path.join(target_include, "cccl")
        local crt_include = path.join(target_include, "crt")
        
        if os.isdir(cccl_include) then
            toolchain:add("includedirs", cccl_include)
        end
        
        toolchain:add("includedirs", 
            path.join(cuda_path, "include"),
            target_include
        )
        
        -- Add crt directory for CUDA runtime headers (needed by clang CUDA runtime wrapper)
        if os.isdir(crt_include) then
            toolchain:add("includedirs", crt_include)
        end
        
        -- Add math_libs include directories (needed for curand_mtgp32_kernel.h and other math library headers)
        -- math_libs is a sibling directory of cuda, so go up from cuda/12.9 to get to the parent, then into math_libs
        local math_libs_path = path.join(cuda_path, "..", "..", "math_libs")
        if os.isdir(math_libs_path) then
            for _, version_dir in ipairs(os.dirs(path.join(math_libs_path, "*"))) do
                local include_dir = path.join(version_dir, "targets", "x86_64-linux", "include")
                if os.isdir(include_dir) then
                    toolchain:add("includedirs", include_dir)
                end
            end
        end
        
        -- Add CUDA library paths and libraries
        if os.isdir(cuda_lib_path) then
            toolchain:add("linkdirs", cuda_lib_path)
            toolchain:add("syslinks", "cudart_static", "cudadevrt", "cudart")
            toolchain:add("ldflags", "-L" .. cuda_lib_path)
            toolchain:add("ldflags", "-Wl,-rpath," .. cuda_lib_path)
            
            -- Set LIBRARY_PATH for linker
            local existing_lib_path = os.getenv("LIBRARY_PATH") or ""
            local new_lib_path = cuda_lib_path .. (existing_lib_path ~= "" and (":" .. existing_lib_path) or "")
            toolchain:add("runenvs", "LIBRARY_PATH", new_lib_path)
            toolchain:add("runenvs", "CUDA_HOME", cuda_path)
            toolchain:add("runenvs", "CUDA_PATH", cuda_path)
            
            -- Also set ACPP_CUDA_PATH if using acpp (AdaptiveCpp uses this env var)
            if use_acpp then
                toolchain:add("runenvs", "ACPP_CUDA_PATH", cuda_path)
            end
        end
        
        -- Store CUDA path for later use
        toolchain:set("cuda_path", cuda_path)
        toolchain:set("cuda_lib_path", cuda_lib_path)
        toolchain:set("gpu_arch", gpu_arch)
        
        -- Set LIBRARY_PATH in environment for xmake process (must be done here)
        -- This ensures it's available during both compilation and linking
        if os.isdir(cuda_lib_path) then
            local existing_lib_path = os.getenv("LIBRARY_PATH") or ""
            local new_lib_path = cuda_lib_path .. (existing_lib_path ~= "" and (":" .. existing_lib_path) or "")
            os.setenv("LIBRARY_PATH", new_lib_path)
        end
    end)
toolchain_end()

