-- xmake/rules/sycl.lua
-- SYCL/AdaptiveCpp rule: automatically configures SYCL targets with AdaptiveCpp
-- This rule handles ACPP path detection, CUDA paths, includes, libraries, and rpath
-- Usage: add_rules("sycl.target") in your target definition

rule("sycl.target")
    on_config(function (target)
        -- Detect ACPP path
        local acpp_path = os.getenv("ACPP_PATH") or "/home/stargazermiao/sycl"
        
        -- Detect CUDA path for AdaptiveCpp
        -- Use CUDA 13.0 (in HPC SDK 25.9) if available, otherwise fall back to 12.9
        local acpp_cuda_path = os.getenv("ACPP_CUDA_PATH") or 
                               (os.isdir("/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0") and 
                                "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0") or
                               "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
        
        -- Suppress CUDA header conflicts with C++ standard library
        -- CUDA headers declare abs() as non-constexpr, conflicting with C++ stdlib
        -- Use -fpermissive to allow this conflict (treats it as a warning)
        target:add("cxflags", "-fpermissive")
        
        -- Add CUDA include paths
        target:add("includedirs",
            path.join(acpp_cuda_path, "include"),
            path.join(acpp_cuda_path, "targets/x86_64-linux/include")
        )
        
        -- Add CUDA libraries for Thrust (if CUBThrust is available)
        local cuda_lib_path = path.join(acpp_cuda_path, "targets", "x86_64-linux", "lib")
        if os.isdir(cuda_lib_path) then
            target:add("linkdirs", cuda_lib_path)
            target:add("syslinks", "cudart", "cudart_static")
        end
        
        -- Add math_libs include paths (for curand, etc.)
        local math_libs_path = path.join(acpp_cuda_path, "..", "math_libs")
        if os.isdir(math_libs_path) then
            for _, version_dir in ipairs(os.dirs(path.join(math_libs_path, "*"))) do
                local include_dir = path.join(version_dir, "targets", "x86_64-linux", "include")
                if os.isdir(include_dir) then
                    target:add("includedirs", include_dir)
                end
            end
        end
    end)
    
    on_load(function (target)
        -- Set rpath for AdaptiveCpp libraries
        local acpp_path = os.getenv("ACPP_PATH") or "/home/stargazermiao/sycl"
        local acpp_lib_path = path.join(acpp_path, "lib")
        if os.isdir(acpp_lib_path) then
            target:set("rpathdirs", acpp_lib_path)
        end
    end)
rule_end()

