-- AdaptiveCpp (acpp) toolchain for xmake
-- Uses clang@acpp syntax so xmake-repo recognizes it as clang-compatible

toolchain("acpp")
    set_kind("clang")
    
    on_check(function (toolchain)
        -- Check if acpp is available
        import("lib.detect.find_tool")
        local acpp = find_tool("acpp")
        if not acpp then
            -- Try common installation paths
            local acpp_paths = {"/home/stargazermiao/sycl/bin/acpp", "/opt/sycl/bin/acpp"}
            for _, acpp_path in ipairs(acpp_paths) do
                if os.isfile(acpp_path) then
                    return true
                end
            end
            return false
        end
        return true
    end)
    
    on_load(function (toolchain)
        -- Find acpp compiler
        import("lib.detect.find_tool")
        local acpp = find_tool("acpp")
        if not acpp then
            local acpp_paths = {"/home/stargazermiao/sycl/bin/acpp", "/opt/sycl/bin/acpp"}
            for _, acpp_path in ipairs(acpp_paths) do
                if os.isfile(acpp_path) then
                    acpp = {program = acpp_path}
                    break
                end
            end
        end
        
        if not acpp then
            raise("acpp compiler not found! Please install AdaptiveCpp.")
        end
        
        -- Use "clang@acpp" syntax for C++ compiler so xmake-repo recognizes it as clang-compatible
        -- acpp is C++ only, so use clang for C compiler
        toolchain:set("toolset", "cc", "clang")
        toolchain:set("toolset", "cxx", "clang@acpp")
        toolchain:set("toolset", "ld", "clang++@acpp")
        toolchain:set("toolset", "sh", "clang++@acpp")
        toolchain:set("toolset", "ar", "ar")
        toolchain:set("toolset", "strip", "strip")
        
        -- Add compiler flags for AdaptiveCpp
        toolchain:add("cxflags", "--acpp-stdpar")
        
        -- Add linker flags and environment variables
        toolchain:add("ldflags", "-latomic")
        toolchain:add("shflags", "-latomic")
        toolchain:add("runenvs", "ACPP_CUDA_PATH", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9")
        
        toolchain:set("acpp_path", acpp.program)
        toolchain:set("acpp_gpu_arch", get_config("acpp-gpu-arch") or "sm_89")
    end)

toolchain_end()

